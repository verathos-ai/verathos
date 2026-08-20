"""Bounded nonce-selected row openings against streaming execution anchors.

DORMANT IN PRODUCTION. Nothing on the serving path reaches this module; see
worker.load_anchored_rows_for_trace for the two blockers and for why lifting
them would buy no latency. The cost claims below describe what this shape
WOULD cost if wired up, not a property of any audit that ships today.

Were it live, this would be the O(log context) half of the hard audit. The
serve commits one 32-byte leaf per activation row; the audit opens only a
fixed number of rows, chosen entirely by the post-nonce beacon, with a
Merkle path each. Prove and verify cost is therefore
O(selected_rows * log context) rather than O(context), and no full GEMM
output is ever materialized. Note this bounds the ACTIVATION side only,
which profiling put at 0.09 s of a 3.1 s prove; it does not touch the PCS
proving over weight geometry that dominates.

Selection determinism is the security property that matters here: the
prover cannot influence which rows it must produce, because the row
indexes come from the beacon, the frozen anchor commitment, and the signed
stage identity alone.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping

from verallm.mesh.anchor_streams import AnchorStageStream
from verallm.mesh.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ProofV3Error,
    ProofV3VerificationError,
    verify_execution_anchor_row_v3,
)
from zkllm.crypto.merkle import MerkleTree
from zkllm.types import MerklePath

ANCHOR_AUDIT_ABI = "mesh.anchor.row_openings.v1"
_SELECT_DOMAIN = b"VERATHOS/MESH/ANCHOR/ROW_SELECT/V1"
_SELECT_OP_DOMAIN = b"VERATHOS/MESH/ANCHOR/OP_ROW_SELECT/V1"
_SELECT_ENTRY_DOMAIN = b"VERATHOS/MESH/ANCHOR/ENTRY_ROW_SELECT/V1"

# Rows opened per anchored stage. Fixed by design: the audit's cost must not
# grow with context, and the openings are a uniform sample of a stage the
# beacon also chose. Two extra bounded rows (first and last) pin the
# sequence ends, where a prover truncating or padding the stream would
# otherwise have the most freedom.
ANCHOR_AUDIT_ROWS_PER_STAGE = 4


def select_anchor_audit_rows(
    *,
    beacon: str | bytes,
    commitment: ExecutionAnchorCommitmentV3,
    rows_per_stage: int = ANCHOR_AUDIT_ROWS_PER_STAGE,
) -> tuple[int, ...]:
    """Derive the audited row indexes for one anchored stage.

    Unbiased rejection sampling over a counter stream seeded by the beacon
    and the frozen commitment. Always includes row 0 and the final row, so
    the ends of the committed sequence are covered on every audit.
    """

    return _sample_rows(
        seed=hashlib.sha256(
            _SELECT_DOMAIN
            + _beacon_bytes(beacon)
            + commitment.canonical_bytes()
        ).digest(),
        row_count=int(commitment.row_count),
        rows_per_stage=rows_per_stage,
    )


def _beacon_bytes(beacon: str | bytes) -> bytes:
    """Validate a 32-byte audit beacon, hex or raw."""

    if isinstance(beacon, str):
        try:
            raw = bytes.fromhex(beacon)
        except ValueError as exc:
            raise ProofV3Error("anchor audit beacon must be hex") from exc
    else:
        raw = bytes(beacon)
    if len(raw) != 32:
        raise ProofV3Error("anchor audit beacon must be 32 bytes")
    return raw


def select_anchor_audit_rows_for_entry(
    *,
    beacon: str | bytes,
    op_manifest_entry_hash: str,
    row_count: int,
    rows_per_stage: int = ANCHOR_AUDIT_ROWS_PER_STAGE,
) -> tuple[int, ...]:
    """Derive the audited row set from PRE-NONCE state plus the nonce.

    This is the selection the mesh audit uses. Every input is fixed before
    the validator's nonce exists: the op-manifest entry hash is Merkle
    committed into ``proof_op_manifest_root`` in the origin receipt, and
    ``row_count`` is that entry's own ``src1_shape[1]``. The beacon adds
    the nonce. Nothing the prover produces after the nonce can move the
    draw.

    That property is the point. ``select_anchor_audit_rows_for_op`` seeds
    from the stream commitments instead, which is sound only when those
    commitments are themselves frozen pre-nonce. When the anchors are
    built during the post-nonce replay the prover could otherwise re-run
    the replay, perturbing rows it does not intend to open, until the draw
    landed somewhere convenient: one prefill per grind attempt.
    """

    if not isinstance(op_manifest_entry_hash, str) or not op_manifest_entry_hash:
        raise ProofV3Error("anchor audit needs the op manifest entry hash")
    try:
        entry_hash = bytes.fromhex(op_manifest_entry_hash)
    except ValueError as exc:
        raise ProofV3Error("op manifest entry hash must be hex") from exc
    return _sample_rows(
        seed=hashlib.sha256(
            _SELECT_ENTRY_DOMAIN
            + _beacon_bytes(beacon)
            + struct.pack("<Q", int(row_count))
            + entry_hash
        ).digest(),
        row_count=int(row_count),
        rows_per_stage=rows_per_stage,
    )


def select_anchor_audit_rows_for_op(
    *,
    beacon: str | bytes,
    src1_commitment: ExecutionAnchorCommitmentV3,
    dst_commitment: ExecutionAnchorCommitmentV3,
    rows_per_stage: int = ANCHOR_AUDIT_ROWS_PER_STAGE,
) -> tuple[int, ...]:
    """Derive ONE audited row set shared by both sides of an anchored op.

    Seeds from the beacon and BOTH stream commitments, so it is sound ONLY
    where those commitments are frozen before the nonce. The mesh audit
    anchors during the post-nonce replay and therefore uses
    ``select_anchor_audit_rows_for_entry`` instead; this form is retained
    for a future origin-anchored lane.
    """

    if src1_commitment.row_count != dst_commitment.row_count:
        raise ProofV3Error(
            "anchored op src1/dst streams disagree on row count"
        )
    return _sample_rows(
        seed=hashlib.sha256(
            _SELECT_OP_DOMAIN
            + _beacon_bytes(beacon)
            + src1_commitment.canonical_bytes()
            + dst_commitment.canonical_bytes()
        ).digest(),
        row_count=int(src1_commitment.row_count),
        rows_per_stage=rows_per_stage,
    )


def _sample_rows(
    *, seed: bytes, row_count: int, rows_per_stage: int
) -> tuple[int, ...]:
    if rows_per_stage < 1:
        raise ProofV3Error("anchor audit needs at least one row")
    if row_count < 1:
        raise ProofV3Error("anchor audit needs a non-empty stream")
    selected: list[int] = [0]
    if row_count > 1:
        selected.append(row_count - 1)
    ceiling = (1 << 64) - ((1 << 64) % row_count) if row_count else 0
    counter = 0
    while len(selected) < min(rows_per_stage, row_count):
        draw = hashlib.sha256(seed + struct.pack("<I", counter)).digest()
        counter += 1
        if counter > 1 << 20:
            raise ProofV3Error("anchor audit row selection did not converge")
        candidate = int.from_bytes(draw[:8], "big")
        if ceiling and candidate >= ceiling:
            continue
        index = candidate % row_count
        if index not in selected:
            selected.append(index)
    return tuple(sorted(selected))


def anchor_commitment_to_dict(
    commitment: ExecutionAnchorCommitmentV3,
) -> dict[str, Any]:
    return {
        "stage_id": commitment.stage_id,
        "row_count": int(commitment.row_count),
        "row_width": int(commitment.row_width),
        "root": commitment.root.hex(),
    }


def anchor_commitment_from_dict(
    data: Mapping[str, Any],
) -> ExecutionAnchorCommitmentV3:
    try:
        root = bytes.fromhex(str(data["root"]))
        return ExecutionAnchorCommitmentV3(
            stage_id=str(data["stage_id"]),
            row_count=int(data["row_count"]),
            row_width=int(data["row_width"]),
            root=root,
        )
    except (KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "anchor commitment is malformed"
        ) from exc


def _require_valid_row_set(rows: tuple[int, ...], row_count: int) -> None:
    if not rows:
        raise ProofV3Error("anchor audit row set is empty")
    if tuple(sorted(set(rows))) != tuple(rows):
        raise ProofV3Error(
            "anchor audit row set must be sorted and distinct"
        )
    if rows[0] < 0 or rows[-1] >= row_count:
        raise ProofV3Error("anchor audit row set is out of range")


@dataclass(frozen=True)
class AnchorRowOpening:
    """One authenticated activation row from a frozen anchor."""

    stage_id: str
    row_index: int
    row_bytes: bytes
    sibling_hashes: tuple[bytes, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "row_index": int(self.row_index),
            "row_hex": self.row_bytes.hex(),
            "sibling_hashes": [item.hex() for item in self.sibling_hashes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnchorRowOpening":
        try:
            row_bytes = bytes.fromhex(str(data["row_hex"]))
            siblings = tuple(
                bytes.fromhex(str(item)) for item in data["sibling_hashes"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofV3VerificationError(
                "anchor row opening is malformed"
            ) from exc
        if any(len(item) != 32 for item in siblings):
            raise ProofV3VerificationError(
                "anchor row opening sibling is not 32 bytes"
            )
        return cls(
            stage_id=str(data.get("stage_id", "")),
            row_index=int(data.get("row_index", -1)),
            row_bytes=row_bytes,
            sibling_hashes=siblings,
        )


def build_anchor_row_openings(
    *,
    beacon: str | bytes,
    stream: AnchorStageStream,
    rows: Mapping[int, bytes],
    rows_per_stage: int = ANCHOR_AUDIT_ROWS_PER_STAGE,
    expected_rows: tuple[int, ...] | None = None,
) -> list[AnchorRowOpening]:
    """Open the beacon-selected rows of one anchored stage.

    ``rows`` supplies the raw bytes for exactly the selected indexes, which
    is all the replay has to reproduce. The Merkle paths come from the
    committed leaf stream, so the prover never rebuilds activations it was
    not asked for.  ``expected_rows`` overrides the per-stage selection for
    joint per-op audits, where both sides open the same
    select_anchor_audit_rows_for_op result.
    """

    selected = (
        expected_rows
        if expected_rows is not None
        else select_anchor_audit_rows(
            beacon=beacon,
            commitment=stream.commitment,
            rows_per_stage=rows_per_stage,
        )
    )
    _require_valid_row_set(selected, stream.commitment.row_count)
    tree = MerkleTree.from_leaf_hashes(list(stream.leaf_hashes))
    if tree.root != stream.commitment.root:
        raise ProofV3Error(
            "anchor leaf stream no longer rebuilds its committed root"
        )
    openings: list[AnchorRowOpening] = []
    for index in selected:
        row = rows.get(index)
        if row is None:
            raise ProofV3Error(
                f"anchor audit is missing replayed row {index} for "
                f"{stream.stage_id}"
            )
        if len(row) != stream.row_width:
            raise ProofV3Error(
                f"anchor audit row {index} for {stream.stage_id} has the "
                "wrong width"
            )
        path = tree.get_path(index)
        openings.append(
            AnchorRowOpening(
                stage_id=stream.stage_id,
                row_index=index,
                row_bytes=row,
                sibling_hashes=tuple(
                    sibling for sibling, _is_left in path.siblings
                ),
            )
        )
    return openings


def verify_anchor_row_openings(
    *,
    beacon: str | bytes,
    commitment: ExecutionAnchorCommitmentV3,
    openings: list[Mapping[str, Any] | AnchorRowOpening],
    rows_per_stage: int = ANCHOR_AUDIT_ROWS_PER_STAGE,
    expected_rows: tuple[int, ...] | None = None,
) -> dict[int, bytes]:
    """Authenticate row openings; return the verified rows by index.

    The verifier derives the expected row set itself and requires an exact
    match, so a prover cannot answer with rows it happens to like, and
    every returned row is Merkle-bound to the pre-nonce commitment.  When
    ``expected_rows`` is given (joint per-op audits) the caller has already
    derived the set from select_anchor_audit_rows_for_op; it is validated
    against this commitment's range and demanded exactly.
    """

    parsed = [
        item
        if isinstance(item, AnchorRowOpening)
        else AnchorRowOpening.from_dict(item)
        for item in openings
    ]
    expected = (
        expected_rows
        if expected_rows is not None
        else select_anchor_audit_rows(
            beacon=beacon,
            commitment=commitment,
            rows_per_stage=rows_per_stage,
        )
    )
    _require_valid_row_set(expected, commitment.row_count)
    if tuple(sorted(item.row_index for item in parsed)) != expected:
        raise ProofV3VerificationError(
            "anchor audit openings do not match the beacon-selected rows"
        )
    depth = (commitment.row_count - 1).bit_length()
    verified: dict[int, bytes] = {}
    for opening in parsed:
        if opening.stage_id != commitment.stage_id:
            raise ProofV3VerificationError(
                "anchor audit opening belongs to a different stage"
            )
        if len(opening.sibling_hashes) != depth:
            raise ProofV3VerificationError(
                "anchor audit opening path depth is wrong"
            )
        if len(opening.row_bytes) != commitment.row_width:
            raise ProofV3VerificationError(
                "anchor audit opening row width is wrong"
            )
        path = MerklePath(
            leaf_index=opening.row_index,
            siblings=[
                (sibling, bool((opening.row_index >> level) & 1))
                for level, sibling in enumerate(opening.sibling_hashes)
            ],
        )
        verify_execution_anchor_row_v3(
            commitment=commitment,
            row_index=opening.row_index,
            row_bytes=opening.row_bytes,
            path=path,
        )
        verified[opening.row_index] = opening.row_bytes
    return verified
