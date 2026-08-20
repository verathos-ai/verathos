"""Streaming full-sequence anchors for bounded post-nonce execution audits.

Only one 32-byte leaf hash per runtime row is consumed while inference runs;
the accumulator retains O(log(context)) state.  A hard audit later rebuilds
the same tree while reconstructing the validator-selected rows.  Matching the
pre-nonce root binds those bounded witnesses to the original full sequence
without retaining every activation between the response and nonce reveal.

Mesh port of proof-v3 GLEIPNIR's execution anchor (byte-identical
construction: same lane geometry, leaf/node hashing, MMR frontier, and
inventory digest), so an anchor produced by either runtime verifies the
same way.  The v3 class names are kept deliberately for
cross-referenceability; only the error types are local because the mesh
tree does not ship the proof_v3 package.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass

from zkllm.crypto.merkle import (
    MerkleTree,
    hash_leaf,
    hash_node,
    verify_merkle_path_with_leaf_hash,
)
from zkllm.types import MerklePath


class ProofV3Error(RuntimeError):
    """Anchor construction or replay violated the streaming contract."""


class ProofV3VerificationError(ProofV3Error):
    """An anchor opening failed verification."""

EXECUTION_ANCHOR_ABI_V3 = "execution.anchor.streaming_rows.v2"

_DEFAULT_LANE_BYTES = 2048
_COMPACT_GDN_LANE_BYTES = 256
_COMPACT_GDN_SUFFIXES = (
    ".gdn_qkvz_output",
    ".gdn_ba_output",
    ".gdn_o_input",
)

_INVENTORY_DOMAIN = b"VERATHOS/PROOF_V3/EXECUTION_ANCHOR/INVENTORY/V1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,95}$")

__all__ = [
    "EXECUTION_ANCHOR_ABI_V3",
    "ExecutionAnchorCommitmentV3",
    "ExecutionAnchorLaneOpeningV3",
    "ExecutionAnchorReplayStageV3",
    "StreamingExecutionAnchorV3",
    "build_execution_anchor_lane_opening_v3",
    "build_execution_anchor_tree_v3",
    "execution_anchor_inventory_digest_v3",
    "execution_anchor_lane_bytes_v3",
    "execution_anchor_row_leaf_from_chunks_v3",
    "execution_anchor_row_leaf_hash_v3",
    "verify_execution_anchor_lane_v3",
    "verify_execution_anchor_row_v3",
]


def execution_anchor_lane_bytes_v3(stage_id: str) -> int:
    """Return the canonical inner-leaf width for one execution stage."""

    _stage(stage_id)
    return (
        _COMPACT_GDN_LANE_BYTES
        if stage_id.endswith(_COMPACT_GDN_SUFFIXES)
        else _DEFAULT_LANE_BYTES
    )


def _stage(stage_id: str) -> bytes:
    if not isinstance(stage_id, str) or _IDENTIFIER.fullmatch(stage_id) is None:
        raise ProofV3Error("execution anchor stage id is malformed")
    return stage_id.encode("ascii")


def execution_anchor_row_leaf_hash_v3(
    *,
    stage_id: str,
    row_index: int,
    row_width: int,
    row_bytes: bytes,
) -> bytes:
    """Canonical pre-hashed Merkle leaf for one runtime row."""

    stage = _stage(stage_id)
    if (
        isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or not 0 <= row_index < 1 << 32
        or isinstance(row_width, bool)
        or not isinstance(row_width, int)
        or not 0 < row_width < 1 << 24
        or not isinstance(row_bytes, bytes)
        or len(row_bytes) != row_width
    ):
        raise ProofV3Error("execution anchor row is malformed")
    # Wide runtime rows are a deterministic two-level tree: hash_leaf() each
    # zero-padded canonical lane, then odd-duplicate Merkle-reduce those lane
    # hashes to one outer leaf.  The compact GDN stages use 256-byte lanes so
    # a post-nonce audit need not transmit unrelated heads.
    # Stage/geometry are bound by the ordered inventory record and row position
    # by the outer Merkle path.
    del stage, row_index
    lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
    padded = row_bytes + bytes((-len(row_bytes)) % lane_bytes)
    chunks = tuple(
        hash_leaf(padded[offset:offset + lane_bytes])
        for offset in range(0, len(padded), lane_bytes)
    )
    return execution_anchor_row_leaf_from_chunks_v3(chunks)


def execution_anchor_row_leaf_from_chunks_v3(
    chunk_hashes: tuple[bytes, ...],
) -> bytes:
    """Odd-duplicate Merkle root of pre-hashed 2KiB row chunks."""

    if (
        not isinstance(chunk_hashes, tuple)
        or not chunk_hashes
        or any(not isinstance(item, bytes) or len(item) != 32
               for item in chunk_hashes)
    ):
        raise ProofV3Error("execution anchor row chunk hashes are malformed")
    level = list(chunk_hashes)
    while len(level) > 1:
        level = [
            hash_node(
                level[index],
                level[index + 1] if index + 1 < len(level) else level[index],
            )
            for index in range(0, len(level), 2)
        ]
    return level[0]


@dataclass(frozen=True, slots=True)
class ExecutionAnchorCommitmentV3:
    stage_id: str
    row_count: int
    row_width: int
    root: bytes

    def __post_init__(self) -> None:
        _stage(self.stage_id)
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or not 0 < self.row_count < 1 << 32
            or isinstance(self.row_width, bool)
            or not isinstance(self.row_width, int)
            or not 0 < self.row_width < 1 << 24
            or not isinstance(self.root, bytes)
            or len(self.root) != 32
            or self.root == bytes(32)
        ):
            raise ProofV3Error("execution anchor commitment is malformed")

    def canonical_bytes(self) -> bytes:
        stage = _stage(self.stage_id)
        return (
            struct.pack("<HII", len(stage), self.row_count, self.row_width)
            + stage
            + self.root
        )


@dataclass(frozen=True, slots=True)
class ExecutionAnchorLaneOpeningV3:
    """One canonical lane inside one full-sequence runtime row."""

    row_index: int
    lane_index: int
    lane_bytes: bytes
    lane_sibling_hashes: tuple[bytes, ...]
    row_sibling_hashes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or not 0 <= self.row_index < 1 << 32
            or isinstance(self.lane_index, bool)
            or not isinstance(self.lane_index, int)
            or not 0 <= self.lane_index < 1 << 24
            or not isinstance(self.lane_bytes, bytes)
            or len(self.lane_bytes) not in (
                _COMPACT_GDN_LANE_BYTES,
                _DEFAULT_LANE_BYTES,
            )
        ):
            raise ProofV3Error("execution anchor lane opening is malformed")
        for name, siblings in (
            ("lane", self.lane_sibling_hashes),
            ("row", self.row_sibling_hashes),
        ):
            if (
                not isinstance(siblings, tuple)
                or len(siblings) > 32
                or any(
                    not isinstance(item, bytes) or len(item) != 32
                    for item in siblings
                )
            ):
                raise ProofV3Error(
                    f"execution anchor {name} sibling path is malformed"
                )


@dataclass(frozen=True, slots=True)
class ExecutionAnchorReplayStageV3:
    """Bounded post-nonce rebuild material for one selected runtime stage.

    Only 32-byte outer leaf hashes are retained across the replay.  Raw
    activation bytes exist solely for the validator-selected rows.  The
    prover builds one temporary Merkle tree at a time, so peak host memory is
    O(sequence) rather than O(sequence * selected stages).

    For attention K/V equality coordinates that become known only after the
    succinct transcript is frozen, ``retained_lane_*`` may carry a bounded
    predeclared lane subset for every row.  It contains no miner-selected
    coordinates: the signed geometry and pre-replay challenge choose the
    subset, while the final verifier-derived coordinates choose the openings.
    """

    stage_id: str
    row_count: int
    row_width: int
    leaf_hashes: bytes
    selected_rows: tuple[tuple[int, bytes], ...]
    retained_lane_indices: tuple[int, ...] = ()
    retained_lane_hashes: bytes = b""
    retained_lane_values: bytes = b""

    def __post_init__(self) -> None:
        _stage(self.stage_id)
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or not 0 < self.row_count < 1 << 32
            or isinstance(self.row_width, bool)
            or not isinstance(self.row_width, int)
            or not 0 < self.row_width < 1 << 24
            or not isinstance(self.leaf_hashes, bytes)
            or len(self.leaf_hashes) != self.row_count * 32
        ):
            raise ProofV3Error(
                "execution anchor replay stage geometry is malformed"
            )
        rows = tuple(self.selected_rows)
        indices = tuple(index for index, _row in rows)
        if (
            indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.row_count
                for index in indices
            )
            or any(
                not isinstance(row, bytes) or len(row) != self.row_width
                for _index, row in rows
            )
        ):
            raise ProofV3Error(
                "execution anchor replay selected rows are malformed"
            )
        object.__setattr__(self, "selected_rows", rows)
        lane_indices = tuple(self.retained_lane_indices)
        lane_bytes = execution_anchor_lane_bytes_v3(self.stage_id)
        lane_count = (self.row_width + lane_bytes - 1) // lane_bytes
        if (
            lane_indices != tuple(sorted(set(lane_indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < lane_count
                for index in lane_indices
            )
            or not isinstance(self.retained_lane_hashes, bytes)
            or not isinstance(self.retained_lane_values, bytes)
            or (
                bool(lane_indices)
                != bool(
                    self.retained_lane_hashes
                    and self.retained_lane_values
                )
            )
            or len(self.retained_lane_hashes)
            != (
                self.row_count * lane_count * 32
                if lane_indices
                else 0
            )
            or len(self.retained_lane_values)
            != (
                self.row_count * len(lane_indices) * lane_bytes
                if lane_indices
                else 0
            )
        ):
            raise ProofV3Error(
                "execution anchor replay retained lanes are malformed"
            )
        object.__setattr__(self, "retained_lane_indices", lane_indices)


class StreamingExecutionAnchorV3:
    """O(log n) odd-dup Merkle accumulator, byte-identical to MerkleTree."""

    __slots__ = ("stage_id", "row_width", "_count", "_peaks")

    def __init__(self, *, stage_id: str, row_width: int) -> None:
        _stage(stage_id)
        if (
            isinstance(row_width, bool)
            or not isinstance(row_width, int)
            or not 0 < row_width < 1 << 24
        ):
            raise ProofV3Error("execution anchor row width is malformed")
        self.stage_id = stage_id
        self.row_width = row_width
        self._count = 0
        self._peaks: list[bytes | None] = []

    @property
    def row_count(self) -> int:
        return self._count

    @property
    def frontier_hashes(self) -> tuple[bytes | None, ...]:
        """Current binary Merkle frontier, for bounded native updates."""

        return tuple(self._peaks)

    def append(self, row_bytes: bytes) -> None:
        leaf = execution_anchor_row_leaf_hash_v3(
            stage_id=self.stage_id,
            row_index=self._count,
            row_width=self.row_width,
            row_bytes=row_bytes,
        )
        self.append_leaf_hash(leaf)

    def append_leaf_hash(self, leaf_hash: bytes) -> None:
        if (
            not isinstance(leaf_hash, bytes)
            or len(leaf_hash) != 32
            or self._count >= (1 << 32) - 1
        ):
            raise ProofV3Error("execution anchor leaf hash is malformed")
        carry = leaf_hash
        level = 0
        while level < len(self._peaks) and self._peaks[level] is not None:
            carry = hash_node(self._peaks[level], carry)
            self._peaks[level] = None
            level += 1
        if level == len(self._peaks):
            self._peaks.append(carry)
        else:
            self._peaks[level] = carry
        self._count += 1

    def replace_frontier_after_append(
        self,
        *,
        added_rows: int,
        frontier: tuple[bytes | None, ...],
    ) -> None:
        """Install a native-computed frontier after an exact append.

        Occupancy is fully determined by the resulting row count.  Requiring
        the exact 32-level shape and zero/None absence prevents a native or
        caller shape error from silently changing the authenticated tree.
        """

        if (
            isinstance(added_rows, bool)
            or not isinstance(added_rows, int)
            or added_rows < 0
            or self._count + added_rows >= 1 << 32
            or not isinstance(frontier, tuple)
            or len(frontier) != 32
        ):
            raise ProofV3Error(
                "execution anchor native frontier update is malformed"
            )
        updated_count = self._count + added_rows
        for level, item in enumerate(frontier):
            occupied = bool(updated_count & (1 << level))
            if occupied != (item is not None):
                raise ProofV3Error(
                    "execution anchor native frontier occupancy is inconsistent"
                )
            if item is not None and (
                not isinstance(item, bytes) or len(item) != 32
            ):
                raise ProofV3Error(
                    "execution anchor native frontier hash is malformed"
                )
        self._count = updated_count
        self._peaks = list(frontier[:updated_count.bit_length()])

    @property
    def root(self) -> bytes:
        if self._count == 0:
            raise ProofV3Error("execution anchor has no rows")
        occupied = [
            (level, peak)
            for level, peak in enumerate(self._peaks)
            if peak is not None
        ]
        level, carry = occupied[0]
        for next_level, peak in occupied[1:]:
            while level < next_level:
                carry = hash_node(carry, carry)
                level += 1
            if level != next_level:
                raise ProofV3Error("execution anchor accumulator is inconsistent")
            carry = hash_node(peak, carry)
            level += 1
        return carry

    def commitment(self) -> ExecutionAnchorCommitmentV3:
        return ExecutionAnchorCommitmentV3(
            stage_id=self.stage_id,
            row_count=self._count,
            row_width=self.row_width,
            root=self.root,
        )


def build_execution_anchor_tree_v3(
    *, stage_id: str, rows: tuple[bytes, ...]
) -> tuple[ExecutionAnchorCommitmentV3, MerkleTree]:
    if not isinstance(rows, tuple) or not rows:
        raise ProofV3Error("execution anchor rows are malformed")
    width = len(rows[0])
    leaves = [
        execution_anchor_row_leaf_hash_v3(
            stage_id=stage_id,
            row_index=index,
            row_width=width,
            row_bytes=row,
        )
        for index, row in enumerate(rows)
    ]
    tree = MerkleTree.from_leaf_hashes(leaves)
    return (
        ExecutionAnchorCommitmentV3(
            stage_id=stage_id,
            row_count=len(rows),
            row_width=width,
            root=tree.root,
        ),
        tree,
    )


def build_execution_anchor_lane_opening_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    row_index: int,
    row_bytes: bytes,
    row_tree: MerkleTree,
    lane_index: int,
) -> ExecutionAnchorLaneOpeningV3:
    """Build one nested lane opening against an execution-anchor tree."""

    if (
        not isinstance(commitment, ExecutionAnchorCommitmentV3)
        or not isinstance(row_tree, MerkleTree)
        or row_tree.root != commitment.root
        or row_tree.num_leaves != commitment.row_count
        or isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or not 0 <= row_index < commitment.row_count
        or not isinstance(row_bytes, bytes)
        or len(row_bytes) != commitment.row_width
    ):
        raise ProofV3Error("execution anchor lane source is malformed")
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    padded = row_bytes + bytes((-len(row_bytes)) % lane_bytes)
    lanes = tuple(
        padded[offset:offset + lane_bytes]
        for offset in range(0, len(padded), lane_bytes)
    )
    if (
        isinstance(lane_index, bool)
        or not isinstance(lane_index, int)
        or not 0 <= lane_index < len(lanes)
    ):
        raise ProofV3Error("execution anchor lane index is out of range")
    lane_tree = MerkleTree.from_leaf_hashes(
        [hash_leaf(lane) for lane in lanes]
    )
    lane_path = lane_tree.get_path(lane_index)
    row_path = row_tree.get_path(row_index)
    return ExecutionAnchorLaneOpeningV3(
        row_index=row_index,
        lane_index=lane_index,
        lane_bytes=lanes[lane_index],
        lane_sibling_hashes=tuple(
            sibling for sibling, _is_left in lane_path.siblings
        ),
        row_sibling_hashes=tuple(
            sibling for sibling, _is_left in row_path.siblings
        ),
    )


def verify_execution_anchor_lane_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    opening: ExecutionAnchorLaneOpeningV3,
) -> bytes:
    """Authenticate one canonical row lane; return its non-padding bytes."""

    if not isinstance(commitment, ExecutionAnchorCommitmentV3) or not isinstance(
        opening, ExecutionAnchorLaneOpeningV3
    ):
        raise ProofV3VerificationError(
            "execution anchor lane opening has an unexpected type"
        )
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    lane_count = (commitment.row_width + lane_bytes - 1) // lane_bytes
    if (
        opening.row_index >= commitment.row_count
        or opening.lane_index >= lane_count
        or len(opening.lane_bytes) != lane_bytes
        or len(opening.lane_sibling_hashes)
        != (lane_count - 1).bit_length()
        or len(opening.row_sibling_hashes)
        != (commitment.row_count - 1).bit_length()
    ):
        raise ProofV3VerificationError(
            "execution anchor lane opening geometry is malformed"
        )
    lane_data_len = min(
        lane_bytes,
        commitment.row_width - opening.lane_index * lane_bytes,
    )
    if opening.lane_bytes[lane_data_len:] != bytes(
        lane_bytes - lane_data_len
    ):
        raise ProofV3VerificationError(
            "execution anchor final lane has non-canonical padding"
        )
    row_leaf = hash_leaf(opening.lane_bytes)
    for level, sibling_hash in enumerate(opening.lane_sibling_hashes):
        row_leaf = (
            hash_node(sibling_hash, row_leaf)
            if (opening.lane_index >> level) & 1
            else hash_node(row_leaf, sibling_hash)
        )
    row_path = MerklePath(
        leaf_index=opening.row_index,
        siblings=[
            (
                sibling,
                bool((opening.row_index >> level) & 1),
            )
            for level, sibling in enumerate(opening.row_sibling_hashes)
        ],
    )
    if not verify_merkle_path_with_leaf_hash(
        commitment.root, row_leaf, row_path
    ):
        raise ProofV3VerificationError(
            "execution anchor lane does not match the pre-nonce root"
        )
    return opening.lane_bytes[:lane_data_len]


def verify_execution_anchor_row_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    row_index: int,
    row_bytes: bytes,
    path: MerklePath,
) -> None:
    if (
        not isinstance(path, MerklePath)
        or path.leaf_index != row_index
        or not 0 <= row_index < commitment.row_count
    ):
        raise ProofV3VerificationError(
            "execution anchor row opening is malformed"
        )
    leaf = execution_anchor_row_leaf_hash_v3(
        stage_id=commitment.stage_id,
        row_index=row_index,
        row_width=commitment.row_width,
        row_bytes=row_bytes,
    )
    if not verify_merkle_path_with_leaf_hash(commitment.root, leaf, path):
        raise ProofV3VerificationError(
            f"execution anchor {commitment.stage_id} row {row_index} "
            "does not match the pre-nonce root"
        )


def execution_anchor_inventory_digest_v3(
    commitments: tuple[ExecutionAnchorCommitmentV3, ...],
) -> bytes:
    if (
        not isinstance(commitments, tuple)
        or not commitments
        or tuple(sorted(item.stage_id for item in commitments))
        != tuple(item.stage_id for item in commitments)
        or len({item.stage_id for item in commitments}) != len(commitments)
    ):
        raise ProofV3Error(
            "execution anchor inventory must be ordered and distinct"
        )
    payload = b"".join(
        struct.pack("<I", len(item.canonical_bytes())) + item.canonical_bytes()
        for item in commitments
    )
    return hashlib.sha256(
        _INVENTORY_DOMAIN + struct.pack("<I", len(commitments)) + payload
    ).digest()
