"""Pre-nonce oracle commitments for the economic_recompute_v3 hard audit.

Miner side: every audited tensor -- projection X, its exact int8 surrogate
output, residual boundaries, per-layer K/V cache pages, the final hidden and
the surrogate logits -- is committed to a REAL Goldilocks capture-tree root
BEFORE the validator nonce is revealed.  Nothing stays a plain dictionary:
reveals later open sampled leaves against these roots.

Every oracle root is bound to (request precommit binding, oracle_id, phase,
layer, operation, dims, quantization scale) through a derived per-oracle
binding digest, so a reveal cannot be re-scoped to a different oracle or a
different request after the nonce.  The full ordered inventory is folded into
``envelope.execution_root`` (:func:`economic_wire.economic_execution_root_v3`).

Validator side: :func:`verify_economic_oracle_opening_v3` re-derives the
binding and leaf geometry from the oracle metadata it already holds (never
trusting the opening's own claims) and authenticates revealed leaves.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache

from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicOracleCommitmentV3,
    economic_execution_root_v3,
    scale_to_bits_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

_ORACLE_BINDING_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_ORACLE_BINDING/SHA256"
_HALF = GOLDILOCKS_MODULUS >> 1

# The exact per-layer and global operation sets a conforming economic proof
# must commit.  Enforced as an EXACT inventory: missing, duplicate or extra
# oracles fail verification (economic_recompute_adapter).
FULL_ATTENTION_LAYER_OPERATIONS_V3 = (
    # alphabetical: matches the wire's canonical (phase, layer, operation)
    # oracle ordering exactly.
    #
    # Full per-layer transition witnesses (GAP1 closure):
    #   *_x  = int8 reduce of the CAPTURED runtime input of a projection
    #   *_s  = exact integer surrogate X.W over the served weights
    #   *_y  = int8 reduce of the CAPTURED runtime output of a projection
    #   q/k/v_cache   = int8 reduce of the captured QKV output slices
    #   mid_residual  = int8 reduce of captured residual_after_attention
    "attn_o_s",
    "attn_o_x",
    "attn_o_y",
    "down_s",
    "down_x",
    "down_y",
    "gate_up_s",
    "gate_up_x",
    "gate_up_y",
    "k_cache",
    "mid_residual",
    "q_cache",
    "qkv_s",
    "qkv_x",
    "residual_in",
    "residual_out",
    "v_cache",
)
GDN_LAYER_OPERATIONS_V3 = (
    "down_s",
    "down_x",
    "down_y",
    "gate_up_s",
    "gate_up_x",
    "gate_up_y",
    "gdn_ba_s",
    "gdn_ba_x",
    "gdn_ba_y",
    "gdn_o_s",
    "gdn_o_x",
    "gdn_o_y",
    "gdn_qkvz_s",
    "gdn_qkvz_x",
    "gdn_qkvz_y",
    "mid_residual",
    "residual_in",
    "residual_out",
)
DENSE_MLP_LAYER_OPERATIONS_V3 = (
    "down_s",
    "down_x",
    "down_y",
    "gate_up_s",
    "gate_up_x",
    "gate_up_y",
)
# Compatibility name for full-attention-only callers. Exact inventory
# construction below is architecture-aware and never uses this alias.
LAYER_OPERATIONS_V3 = FULL_ATTENTION_LAYER_OPERATIONS_V3
GLOBAL_OPERATIONS_V3 = ("final_hidden", "logits")
STREAMING_GLOBAL_OPERATIONS_V3 = (
    "final_hidden",
    "logits",
    "response_stamp_input",
)

# Chunk-leaf layout: one tree leaf commits ORACLE_CHUNK_WIDTH_V3 consecutive
# cells of the padded row-major grid (or one full padded row when narrower).
# The width always divides col_pad, so a chunk never straddles rows.
ORACLE_CHUNK_WIDTH_V3 = 32

# One reference capture tree holds at most 2^16 leaves, so the committed
# surrogate logits (decode x vocab, vocab can be 150k+) are decomposed into
# deterministic vocab-column blocks.  The bound is on chunk leaves, not raw
# cells: failing to account for the 32-cell leaf ABI creates 32 undersized
# trees and turns fixed per-tree setup into the light-path bottleneck.
# Both sides derive the SAME geometry from (decode_rows, vocab) alone -- a
# miner cannot choose it.
LOGITS_BLOCK_LEAVES_CAP_V3 = 1 << 16


def logits_block_geometry_v3(*, decode_rows: int, vocab: int) -> tuple[int, int]:
    """Return ``(block_cols, block_count)`` for the logits block oracles."""

    if decode_rows < 1 or vocab < 1:
        raise ProofV3Error("logits block geometry is malformed")
    row_pad = 1 << max(0, (decode_rows - 1).bit_length())
    if row_pad > LOGITS_BLOCK_LEAVES_CAP_V3:
        raise ProofV3Error("logits block geometry cannot fit the decode rows")
    block_cols = (
        LOGITS_BLOCK_LEAVES_CAP_V3
        * ORACLE_CHUNK_WIDTH_V3
        // row_pad
    )
    block_count = (vocab + block_cols - 1) // block_cols
    return block_cols, block_count


def logits_block_oracle_id_v3(block_index: int) -> str:
    return f"logits{block_index:04d}"

__all__ = [
    "GLOBAL_OPERATIONS_V3",
    "STREAMING_GLOBAL_OPERATIONS_V3",
    "LAYER_OPERATIONS_V3",
    "FULL_ATTENTION_LAYER_OPERATIONS_V3",
    "GDN_LAYER_OPERATIONS_V3",
    "DENSE_MLP_LAYER_OPERATIONS_V3",
    "LOGITS_BLOCK_LEAVES_CAP_V3",
    "logits_block_geometry_v3",
    "logits_block_oracle_id_v3",
    "EconomicCommittedOracleV3",
    "EconomicOracleSetV3",
    "commit_economic_oracle_tensor_v3",
    "commit_economic_oracle_v3",
    "economic_oracle_binding_v3",
    "expected_economic_inventory_v3",
    "field_to_signed_v3",
    "ORACLE_CHUNK_WIDTH_V3",
    "oracle_cell_count_v3",
    "oracle_leaf_count_v3",
    "oracle_leaf_index_v3",
    "oracle_leaf_width_v3",
    "signed_to_field_v3",
    "verify_economic_oracle_opening_v3",
]


def signed_to_field_v3(value: int) -> int:
    value = int(value)
    if not -_HALF < value < _HALF:
        raise ProofV3Error("oracle value exceeds the signed field range")
    return value + GOLDILOCKS_MODULUS if value < 0 else value


def field_to_signed_v3(value: int) -> int:
    value = int(value)
    if not 0 <= value < GOLDILOCKS_MODULUS:
        raise ProofV3VerificationError("oracle field value is out of range")
    return value - GOLDILOCKS_MODULUS if value > _HALF else value


@lru_cache(maxsize=4096)
def _pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _canonical_leaf_indices(indices) -> tuple[int, ...]:
    """Exact ``tuple(sorted(set(int(i) for i in indices)))`` with a
    C-speed fast path for already strictly-increasing plain-int input
    (the common case: concatenated contiguous row ranges)."""

    try:
        import numpy as _np

        arr = _np.fromiter(indices, dtype=_np.int64)
    except (TypeError, ValueError, OverflowError):
        return tuple(sorted({int(index) for index in indices}))
    if arr.size and bool((_np.diff(arr) > 0).all()):
        return tuple(arr.tolist())
    if not arr.size:
        return ()
    return tuple(_np.unique(arr).tolist())


def oracle_leaf_width_v3(col_count: int) -> int:
    return min(ORACLE_CHUNK_WIDTH_V3, _pow2(col_count))


def oracle_cell_count_v3(row_count: int, col_count: int) -> int:
    return _pow2(row_count) * _pow2(col_count)


def oracle_leaf_count_v3(row_count: int, col_count: int) -> int:
    return oracle_cell_count_v3(row_count, col_count) // oracle_leaf_width_v3(
        col_count
    )


def oracle_leaf_index_v3(row: int, col: int, col_count: int) -> int:
    """Flat CELL position in the padded row-major grid (NOT a tree leaf:
    tree leaves are chunks; chunk = cell // width, offset = cell % width)."""

    if not 0 <= col < col_count:
        raise ProofV3Error("oracle leaf column is out of range")
    return row * _pow2(col_count) + col


def economic_oracle_binding_v3(
    *,
    base_binding: bytes,
    oracle_id: str,
    phase: str,
    layer_index: int,
    operation: str,
    row_count: int,
    col_count: int,
    scale_bits: int,
) -> bytes:
    """Per-oracle capture binding: request context + full oracle identity."""

    if not isinstance(base_binding, bytes) or len(base_binding) != 32:
        raise ProofV3Error("oracle base binding must be exactly 32 bytes")
    ident = f"{oracle_id}|{phase}|{operation}".encode("ascii")
    return hashlib.sha256(
        _ORACLE_BINDING_DOMAIN
        + base_binding
        + struct.pack("<H", len(ident))
        + ident
        + struct.pack("<IIIQ", layer_index, row_count, col_count, scale_bits)
    ).digest()


@dataclass(frozen=True, slots=True)
class EconomicCommittedOracleV3:
    """Miner-side committed oracle: wire record + retained leaf vector.

    ``tree`` is the ONE Merkle tree built at commit time (GPU-fused root
    when the fused reference acceleration is installed); every opening
    reuses it instead of rebuilding the tree per reveal.
    """

    commitment: EconomicOracleCommitmentV3
    binding: bytes
    raw_values: tuple          # padded row-major field vector the root commits
    row_pad: int
    col_pad: int
    tree: object = None
    int_rows_cpu: object = None   # tensor path: original signed rows (CPU)

    @property
    def leaf_width(self) -> int:
        return min(ORACLE_CHUNK_WIDTH_V3, self.col_pad)

    def leaf_indices_for_row(self, row: int) -> tuple[int, ...]:
        """Tree-leaf (chunk) indices covering one committed row."""

        if not 0 <= row < self.commitment.row_count:
            raise ProofV3Error("oracle row is out of range")
        width = self.leaf_width
        base = row * (self.col_pad // width)
        used = -(-self.commitment.col_count // width)
        return tuple(range(base, base + used))

    def open_leaves(
        self, leaf_indices, *, value_mode: int = 1,
        bounded_width: int | None = None,
    ) -> EconomicMerkleOpeningV3:
        """Open the given chunk leaves against the committed root."""

        if self.tree is not None:
            opening = self.tree.open(_canonical_leaf_indices(leaf_indices))
        else:
            import hashlib as _hashlib

            from verallm.proof_v3.goldilocks_capture_commitment_reference import (
                _CAPTURE_DOMAIN,
            )
            from verallm.proof_v3.goldilocks_merkle_reference import (
                GoldilocksMerkleTreeReference,
            )

            width = self.leaf_width
            leaf_binding = _hashlib.sha256(
                _CAPTURE_DOMAIN + b"LEAVES/" + self.binding
            ).digest()
            tree = GoldilocksMerkleTreeReference.from_rows(
                tuple(
                    tuple(self.raw_values[start:start + width])
                    for start in range(0, len(self.raw_values), width)
                ),
                binding_digest=leaf_binding,
            )
            opening = tree.open(_canonical_leaf_indices(leaf_indices))
        return EconomicMerkleOpeningV3.from_reference(
            opening, value_mode=value_mode, bounded_width=bounded_width)

    def open_rows(
        self, rows, *, value_mode: int = 1, bounded_width: int | None = None
    ) -> tuple[tuple[int, ...], EconomicMerkleOpeningV3]:
        leaves: list[int] = []
        for row in rows:
            leaves.extend(self.leaf_indices_for_row(row))
        indices = _canonical_leaf_indices(leaves)
        return indices, self.open_leaves(
            indices, value_mode=value_mode, bounded_width=bounded_width)

    def open_cells(
        self, cells, *, value_mode: int = 1, bounded_width: int | None = None
    ) -> tuple[tuple[int, ...], EconomicMerkleOpeningV3]:
        """Open the chunks holding individual ``(row, col)`` cells."""

        width = self.leaf_width
        leaves = {
            oracle_leaf_index_v3(row, col, self.commitment.col_count) // width
            for row, col in cells
        }
        indices = tuple(sorted(leaves))
        return indices, self.open_leaves(
            indices, value_mode=value_mode, bounded_width=bounded_width)

    def signed_value(self, row: int, col: int) -> int:
        if self.int_rows_cpu is not None:
            return int(self.int_rows_cpu[row, col])
        index = oracle_leaf_index_v3(row, col, self.commitment.col_count)
        return field_to_signed_v3(self.raw_values[index])


# p - 2^64 as a signed 64-bit constant: adding the Goldilocks modulus to a
# negative value modulo 2^64 equals subtracting (2^32 - 1) in int64 bits.
_FIELD_BITS_NEGATIVE_OFFSET = -((1 << 32) - 1)


class _LazyReferenceTreeV3:
    """Opening-capable view over a GPU-committed chunk-leaf vector.

    Retains only the HOST-side signed source rows (int8 when the capture
    fits, int64 otherwise): the padded device field vector is rebuilt on
    demand for the oracles the challenge actually opens, re-hashed
    device-resident with SPARSE sibling extraction, and the recomputed
    root must byte-match the committed root.  Device residency between
    commit and audit: zero.
    """

    __slots__ = ("_leaf_binding", "_source_cpu", "_row_pad", "_col_pad",
                 "_root", "_leaf_width")

    def __init__(
        self, *, leaf_binding: bytes, source_cpu, row_pad: int,
        col_pad: int, root: bytes, leaf_width: int,
    ) -> None:
        self._leaf_binding = leaf_binding
        self._source_cpu = source_cpu
        self._row_pad = int(row_pad)
        self._col_pad = int(col_pad)
        self._root = root
        self._leaf_width = int(leaf_width)

    @property
    def commitment(self) -> bytes:
        return self._root

    def open(self, indices):
        from verallm.proof_v3.goldilocks_merkle_reference import (
            GoldilocksMerkleMultiOpeningReference,
            GoldilocksMerkleSiblingReference,
            _expected_sibling_coordinates,
        )
        from verallm.proof_v3.native_cuda_tree_backend import (
            fused_merkle_sibling_paths_wn_streamed,
        )
        from verallm.proof_v3.native_reference_tree_accelerator import (
            _STATE as _accel_state,
        )

        tree_ext = (
            _accel_state.get("tree_ext")
            if _accel_state.get("installed")
            else None
        )
        if tree_ext is None:
            raise ProofV3Error(
                "fused tree extension is unavailable for opening"
            )
        width = self._leaf_width
        leaf_count = self._row_pad * self._col_pad // width
        selected = _canonical_leaf_indices(indices)
        commitment, paths = fused_merkle_sibling_paths_wn_streamed(
            tree_ext, self._source_cpu,
            binding_digest=self._leaf_binding,
            leaf_width=width, row_pad=self._row_pad,
            col_pad=self._col_pad,
            negative_offset=_FIELD_BITS_NEGATIVE_OFFSET,
            leaf_indices=selected,
        )
        if commitment != self._root:
            raise ProofV3Error(
                "fused oracle root does not match the opening rebuild"
            )
        # opened chunk values come straight from the retained host rows
        # (pad cells are committed zeros); tiny relative to the openings
        source = self._source_cpu
        src_rows = int(source.shape[0])
        src_cols = int(source.shape[1])
        chunks_per_row = self._col_pad // width
        rows = []
        for chunk in selected:
            row = chunk // chunks_per_row
            col0 = (chunk % chunks_per_row) * width
            if row < src_rows and col0 < src_cols:
                cells = source[row, col0:min(col0 + width, src_cols)].tolist()
            else:
                cells = []
            if len(cells) < width:
                cells = list(cells) + [0] * (width - len(cells))
            rows.append(tuple(
                value + GOLDILOCKS_MODULUS if value < 0 else value
                for value in cells
            ))
        siblings = tuple(
            GoldilocksMerkleSiblingReference(
                level=level,
                index=index,
                digest=paths[(level, index)],
            )
            for level, index in _expected_sibling_coordinates(
                leaf_count=leaf_count, indices=selected
            )
        )
        return GoldilocksMerkleMultiOpeningReference(
            binding_digest=self._leaf_binding,
            leaf_count=leaf_count,
            leaf_width=width,
            indices=selected,
            rows=tuple(rows),
            siblings=siblings,
        )


def commit_economic_oracle_tensor_v3(
    *,
    base_binding: bytes,
    oracle_id: str,
    phase: str,
    layer_index: int,
    operation: str,
    int_rows,
    scale: float,
) -> EconomicCommittedOracleV3:
    """Tensor-native commit: pad + signed->field mapping on device, fused
    GPU root, NO per-element Python pass.  Byte-identical to
    :func:`commit_economic_oracle_v3` (asserted lazily on first opening).

    ``int_rows`` is a 2-D integer torch tensor (values in the signed field
    range).  Falls back to the list path when the fused tree extension is
    unavailable or the tensor is too small for it.
    """

    import torch

    if int_rows.dim() != 2 or int_rows.numel() == 0:
        raise ProofV3Error("oracle tensor must be non-empty 2-D")
    row_count, col_count = int(int_rows.shape[0]), int(int_rows.shape[1])
    scale_bits = scale_to_bits_v3(scale)
    binding = economic_oracle_binding_v3(
        base_binding=base_binding,
        oracle_id=oracle_id,
        phase=phase,
        layer_index=layer_index,
        operation=operation,
        row_count=row_count,
        col_count=col_count,
        scale_bits=scale_bits,
    )
    row_pad = _pow2(row_count)
    col_pad = _pow2(col_count)

    from verallm.proof_v3.native_reference_tree_accelerator import (
        _STATE as _accel_state,
    )

    tree_ext = _accel_state.get("tree_ext") if _accel_state.get("installed") else None
    if tree_ext is None or not torch.cuda.is_available():
        rows_list = int_rows.cpu().tolist()
        return commit_economic_oracle_v3(
            base_binding=base_binding,
            oracle_id=oracle_id,
            phase=phase,
            layer_index=layer_index,
            operation=operation,
            rows=rows_list,
            scale=scale,
        )

    import hashlib as _hashlib

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
    )
    from verallm.proof_v3.native_cuda_tree_backend import (
        fused_merkle_root_wn_streamed,
    )

    leaf_binding = _hashlib.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + binding
    ).digest()
    leaf_width = oracle_leaf_width_v3(col_count)
    # streamed build: the padded field grid never materializes (device
    # transients stay bounded at any context/width, so the always-on
    # commit coexists with production gpu-mem)
    root = fused_merkle_root_wn_streamed(
        tree_ext, int_rows, binding_digest=leaf_binding,
        leaf_width=leaf_width, row_pad=row_pad, col_pad=col_pad,
        negative_offset=_FIELD_BITS_NEGATIVE_OFFSET,
    )
    record = EconomicOracleCommitmentV3(
        oracle_id=oracle_id,
        phase=phase,
        layer_index=layer_index,
        operation=operation,
        row_count=row_count,
        col_count=col_count,
        scale_bits=scale_bits,
        root=root,
    )
    # Captures retain on HOST at the narrowest exact dtype (int8 ->
    # int32 -> int64; every int8-dot surrogate fits int32); the lazy
    # tree rebuilds the padded device vector from this SAME retained
    # tensor only for oracles the challenge opens, so nothing stays
    # device-resident between commit and audit.
    if int_rows.numel():
        low, high = int(int_rows.min()), int(int_rows.max())
    else:
        low = high = 0
    if -128 <= low and high <= 127:
        int_rows_cpu = int_rows.char().to("cpu")
    elif -(1 << 31) <= low and high < (1 << 31):
        int_rows_cpu = int_rows.to(dtype=torch.int32).to("cpu")
    else:
        int_rows_cpu = int_rows.to("cpu")
    return EconomicCommittedOracleV3(
        commitment=record,
        binding=binding,
        raw_values=(),
        row_pad=row_pad,
        col_pad=col_pad,
        tree=_LazyReferenceTreeV3(
            leaf_binding=leaf_binding, source_cpu=int_rows_cpu,
            row_pad=row_pad, col_pad=col_pad, root=root,
            leaf_width=leaf_width,
        ),
        int_rows_cpu=int_rows_cpu,
    )


def commit_economic_oracle_v3(
    *,
    base_binding: bytes,
    oracle_id: str,
    phase: str,
    layer_index: int,
    operation: str,
    rows,
    scale: float,
) -> EconomicCommittedOracleV3:
    """Commit one oracle tensor (integer ``rows[r][c]``) to a capture root.

    Builds the capture Merkle tree ONCE and retains it for openings.  With
    :func:`native_reference_tree_accelerator.install_fused_reference_acceleration`
    installed, the root is GPU-fused and the CPU tree materialises lazily
    only for oracles the challenge actually opens.
    """

    import hashlib as _hashlib

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
    )
    from verallm.proof_v3.goldilocks_merkle_reference import (
        GoldilocksMerkleTreeReference,
    )

    rows = [tuple(int(v) for v in row) for row in rows]
    if not rows or not rows[0]:
        raise ProofV3Error("oracle tensor must be non-empty")
    col_count = len(rows[0])
    if any(len(row) != col_count for row in rows):
        raise ProofV3Error("oracle tensor rows are ragged")
    row_count = len(rows)
    scale_bits = scale_to_bits_v3(scale)
    binding = economic_oracle_binding_v3(
        base_binding=base_binding,
        oracle_id=oracle_id,
        phase=phase,
        layer_index=layer_index,
        operation=operation,
        row_count=row_count,
        col_count=col_count,
        scale_bits=scale_bits,
    )
    row_pad = _pow2(row_count)
    col_pad = _pow2(col_count)
    flat: list[int] = []
    for r in range(row_pad):
        row = rows[r] if r < row_count else ()
        for c in range(col_pad):
            flat.append(signed_to_field_v3(row[c]) if c < len(row) else 0)
    # chunk-leaf construction (byte-identical to the fused GPU commit);
    # the tree is built once and retained for every later opening
    leaf_binding = _hashlib.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + binding
    ).digest()
    width = oracle_leaf_width_v3(col_count)
    tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(
            tuple(flat[start:start + width])
            for start in range(0, len(flat), width)
        ),
        binding_digest=leaf_binding,
    )
    commitment_root = tree.commitment
    record = EconomicOracleCommitmentV3(
        oracle_id=oracle_id,
        phase=phase,
        layer_index=layer_index,
        operation=operation,
        row_count=row_count,
        col_count=col_count,
        scale_bits=scale_bits,
        root=commitment_root,
    )
    return EconomicCommittedOracleV3(
        commitment=record,
        binding=binding,
        raw_values=tuple(flat),
        row_pad=row_pad,
        col_pad=col_pad,
        tree=tree,
    )


class EconomicOracleSetV3:
    """Ordered miner-side oracle collection for one audited request."""

    def __init__(self, *, base_binding: bytes) -> None:
        if not isinstance(base_binding, bytes) or len(base_binding) != 32:
            raise ProofV3Error("oracle base binding must be exactly 32 bytes")
        self._base_binding = base_binding
        self._oracles: list[EconomicCommittedOracleV3] = []
        self._by_id: dict[str, int] = {}

    def commit(
        self,
        *,
        oracle_id: str,
        phase: str,
        layer_index: int,
        operation: str,
        rows,
        scale: float,
    ) -> EconomicCommittedOracleV3:
        if oracle_id in self._by_id:
            raise ProofV3Error("oracle_id is already committed")
        oracle = commit_economic_oracle_v3(
            base_binding=self._base_binding,
            oracle_id=oracle_id,
            phase=phase,
            layer_index=layer_index,
            operation=operation,
            rows=rows,
            scale=scale,
        )
        self._by_id[oracle_id] = len(self._oracles)
        self._oracles.append(oracle)
        return oracle

    def commit_tensor(
        self,
        *,
        oracle_id: str,
        phase: str,
        layer_index: int,
        operation: str,
        int_rows,
        scale: float,
    ) -> EconomicCommittedOracleV3:
        """Tensor-native commit (GPU-fused root when available)."""

        if oracle_id in self._by_id:
            raise ProofV3Error("oracle_id is already committed")
        oracle = commit_economic_oracle_tensor_v3(
            base_binding=self._base_binding,
            oracle_id=oracle_id,
            phase=phase,
            layer_index=layer_index,
            operation=operation,
            int_rows=int_rows,
            scale=scale,
        )
        self._by_id[oracle_id] = len(self._oracles)
        self._oracles.append(oracle)
        return oracle

    def index_of(self, oracle_id: str) -> int:
        if oracle_id not in self._by_id:
            raise ProofV3Error(f"oracle {oracle_id!r} is not committed")
        return self._by_id[oracle_id]

    def get(self, oracle_id: str) -> EconomicCommittedOracleV3:
        return self._oracles[self.index_of(oracle_id)]

    def wire_oracles(self) -> tuple[EconomicOracleCommitmentV3, ...]:
        return tuple(oracle.commitment for oracle in self._oracles)

    def execution_root(self, *, capture_chain_digest: bytes) -> bytes:
        return economic_execution_root_v3(
            oracles=self.wire_oracles(),
            capture_chain_digest=capture_chain_digest,
        )


def expected_economic_inventory_v3(
    *,
    layer_indices,
    layer_kinds,
    global_layer_index: int,
    logits_block_count: int = 1,
    selected_layer_indices=None,
    moe_layer_indices=(),
) -> tuple[tuple[str, str, int, str], ...]:
    """The EXACT ordered ``(oracle_id, phase, layer, operation)`` inventory.

    Legacy mode (``selected_layer_indices is None``) carries every operation
    for every layer.  Streaming mode carries complete operation witnesses
    only for the nonce-selected layers and just ``residual_in/out`` for every
    other layer so the prompt-to-output chain remains fully connected.  The
    global final-hidden oracle and, for legacy profiles,
    ``logits_block_count`` surrogate-logit block oracles follow.  Compact-v9
    profiles pass zero because terminal work is selected post-nonce.
    Streaming mode additionally carries the bounded
    pre-nonce prompt-tail input used by the nonce-free response stamp.  The
    Sparse-MoE layers omit the six dense MLP projection oracles because their
    signed router, selected-expert, shared-expert and capacity openings use
    the dedicated bounded MoE wire.  The adapter compares the wire inventory
    against this set exactly -- a
    missing, duplicated, reordered or extra oracle fails verification.
    """

    layer_indices = tuple(int(layer) for layer in layer_indices)
    if not layer_indices or layer_indices != tuple(sorted(set(layer_indices))):
        raise ProofV3Error("inventory layer indices must be sorted and distinct")
    try:
        kinds = {int(layer): str(kind) for layer, kind in layer_kinds.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error("inventory layer kinds are malformed") from exc
    if set(kinds) != set(layer_indices) or any(
        kind not in {"full_attention", "gdn"} for kind in kinds.values()
    ):
        raise ProofV3Error(
            "inventory layer kinds do not match the signed layer set"
        )
    if any(layer >= global_layer_index for layer in layer_indices):
        raise ProofV3Error("inventory global layer index must exceed all layers")
    if not 0 <= int(logits_block_count) <= 9_999:
        raise ProofV3Error("inventory logits block count is out of range")
    selected = None
    if selected_layer_indices is not None:
        selected = tuple(int(layer) for layer in selected_layer_indices)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or any(layer not in kinds for layer in selected)
        ):
            raise ProofV3Error(
                "inventory selected layers must be ordered within the "
                "signed layer set"
            )
    moe_layers = tuple(int(layer) for layer in moe_layer_indices)
    if (
        moe_layers != tuple(sorted(set(moe_layers)))
        or any(layer not in kinds for layer in moe_layers)
    ):
        raise ProofV3Error(
            "inventory MoE layers must be ordered within the signed layer set"
        )
    moe_set = frozenset(moe_layers)
    dense_mlp = frozenset(DENSE_MLP_LAYER_OPERATIONS_V3)
    inventory: list[tuple[str, str, int, str]] = []
    for layer in layer_indices:
        if selected is not None and layer not in selected:
            operations = ("residual_in", "residual_out")
        else:
            operations = (
                FULL_ATTENTION_LAYER_OPERATIONS_V3
                if kinds[layer] == "full_attention"
                else GDN_LAYER_OPERATIONS_V3
            )
            if layer in moe_set:
                operations = tuple(
                    operation
                    for operation in operations
                    if operation not in dense_mlp
                )
        for operation in operations:
            inventory.append(
                (f"l{layer}.{operation}", "global", layer, operation)
            )
    inventory.append(
        ("final_hidden", "global", global_layer_index, "final_hidden")
    )
    for block in range(int(logits_block_count)):
        inventory.append(
            (
                logits_block_oracle_id_v3(block),
                "global",
                global_layer_index,
                "logits",
            )
        )
    if selected is not None:
        inventory.append(
            (
                "response_stamp_input",
                "global",
                global_layer_index,
                "response_stamp_input",
            )
        )
    return tuple(inventory)


def verify_economic_oracle_opening_v3(
    *,
    oracle: EconomicOracleCommitmentV3,
    base_binding: bytes,
    expected_indices,
    opening: EconomicMerkleOpeningV3,
    expected_mode: int = 1,
    external_values=None,
    expected_bounded_width: int | None = None,
    return_values: bool = True,
) -> dict[int, int]:
    """Validator-side: authenticate revealed cells against one oracle root.

    Binding and leaf geometry are re-derived from the oracle metadata the
    validator already validated against the frozen inventory.
    ``expected_indices`` are validator-derived CELL positions
    (:func:`oracle_leaf_index_v3`); the verifier maps them to the
    chunk-leaf set itself -- COMPACT openings (wire v3) never carry
    indices, and Merkle reconstruction rejects any other set.
    ``expected_mode`` is the SITE CONTRACT for how values travel
    (external / field / int8); a mismatched mode fails closed.
    ``external_values``: canonical field values aligned to the sorted
    expected cells, supplied from data already authenticated exactly
    once on the wire; every other cell of an externally-opened chunk
    must be grid padding (committed zero).  Returns
    ``{cell_index: signed_value}``."""

    import hashlib as _hashlib

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
    )
    from verallm.proof_v3.goldilocks_merkle_reference import (
        verify_goldilocks_merkle_multiopening_reference,
    )

    if not isinstance(oracle, EconomicOracleCommitmentV3):
        raise ProofV3VerificationError("oracle record has an unexpected type")
    if not isinstance(opening, EconomicMerkleOpeningV3):
        raise ProofV3VerificationError("oracle opening has an unexpected type")
    if int(opening.value_mode) != int(expected_mode):
        raise ProofV3VerificationError(
            "oracle opening value mode does not match the site contract"
        )
    if opening.bounded_width != expected_bounded_width:
        raise ProofV3VerificationError(
            "oracle opening byte width does not match the derived bound"
        )
    expected = _canonical_leaf_indices(expected_indices)
    if not expected:
        raise ProofV3VerificationError("oracle opening has no expected leaves")
    col_pad = _pow2(oracle.col_count)
    cell_count = _pow2(oracle.row_count) * col_pad
    if expected[-1] >= cell_count:
        raise ProofV3VerificationError("oracle opening index exceeds the tree")
    width = oracle_leaf_width_v3(oracle.col_count)
    leaf_count = cell_count // width
    chunks = tuple(sorted({index // width for index in expected}))
    binding = economic_oracle_binding_v3(
        base_binding=base_binding,
        oracle_id=oracle.oracle_id,
        phase=oracle.phase,
        layer_index=oracle.layer_index,
        operation=oracle.operation,
        row_count=oracle.row_count,
        col_count=oracle.col_count,
        scale_bits=oracle.scale_bits,
    )
    external = None
    if external_values is not None:
        supplied = tuple(int(value) for value in external_values)
        if len(supplied) != len(expected):
            raise ProofV3VerificationError(
                "external opening values do not align with the derived cells"
            )
        by_cell = dict(zip(expected, supplied))
        external = []
        for chunk in chunks:
            for offset in range(width):
                cell = chunk * width + offset
                value = by_cell.get(cell)
                if value is None:
                    if (
                        cell % col_pad < oracle.col_count
                        and cell // col_pad < oracle.row_count
                    ):
                        raise ProofV3VerificationError(
                            "external opening chunk covers underived cells"
                        )
                    value = 0
                external.append(value)
    leaf_binding = _hashlib.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + binding
    ).digest()
    flat = _fast_verified_chunk_values(
        oracle_root=oracle.root,
        leaf_binding=leaf_binding,
        leaf_count=leaf_count,
        width=width,
        chunks=chunks,
        opening=opening,
        external=external,
    )
    if flat is None:
        # reference path (no compiled reconstruction available)
        try:
            reference = opening.to_reference_with(
                indices=chunks, external_values=external, leaf_width=width)
        except ProofV3Error as exc:
            raise ProofV3VerificationError(str(exc))
        verify_goldilocks_merkle_multiopening_reference(
            oracle.root,
            reference,
            expected_binding_digest=leaf_binding,
            expected_leaf_count=leaf_count,
            expected_leaf_width=width,
            expected_indices=chunks,
        )
        flat = [value for row in reference.rows for value in row]
    if not return_values:
        return {}
    # values are canonical field elements here (any other value produces
    # different leaf bytes than the committed canonical encoding, so the
    # root reconstruction above already rejected it); inline the signed
    # mapping -- per-cell function calls dominated the verify profile
    pos_of = {chunk: slot for slot, chunk in enumerate(chunks)}
    half = _HALF
    modulus = GOLDILOCKS_MODULUS
    out: dict[int, int] = {}
    for index in expected:
        value = flat[pos_of[index // width] * width + index % width]
        out[index] = value - modulus if value > half else value
    return out


def verify_external_oracle_rows_fast_v3(
    *,
    oracle: EconomicOracleCommitmentV3,
    base_binding: bytes,
    row_indices,
    packed_rows,
    opening: EconomicMerkleOpeningV3,
) -> bool:
    """Authenticate complete packed int8 rows without per-cell Python state.

    The fast path applies only when one row is an exact number of chunk
    leaves. Other geometries return ``False`` so the caller can use the
    general sparse/padded verifier.
    """

    if not isinstance(oracle, EconomicOracleCommitmentV3):
        raise ProofV3VerificationError("oracle record has an unexpected type")
    if not isinstance(opening, EconomicMerkleOpeningV3):
        raise ProofV3VerificationError("oracle opening has an unexpected type")
    if opening.value_mode != 0:
        raise ProofV3VerificationError(
            "external row opening value mode does not match the site contract"
        )
    if opening.bounded_width is not None:
        raise ProofV3VerificationError(
            "external row opening unexpectedly carries a bounded width"
        )
    rows = tuple(int(row) for row in row_indices)
    if (
        not rows
        or rows != tuple(sorted(set(rows)))
        or rows[0] < 0
        or rows[-1] >= oracle.row_count
    ):
        raise ProofV3VerificationError(
            "external row opening indices are malformed"
        )
    values = tuple(packed_rows)
    if len(values) != len(rows):
        raise ProofV3VerificationError(
            "external row opening values do not align with the rows"
        )
    width = oracle_leaf_width_v3(oracle.col_count)
    if oracle.col_count % width:
        return False
    if any(
        not isinstance(row, bytes) or len(row) != oracle.col_count
        for row in values
    ):
        raise ProofV3VerificationError(
            "external row opening has a malformed packed row"
        )
    try:
        import numpy as np
    except ImportError:
        return False
    col_pad = _pow2(oracle.col_count)
    chunks_per_row = oracle.col_count // width
    chunks = tuple(
        chunk
        for row in rows
        for chunk in range(
            row * col_pad // width,
            row * col_pad // width + chunks_per_row,
        )
    )
    signed = np.frombuffer(b"".join(values), dtype=np.int8).astype(
        np.int64,
        copy=False,
    )
    unsigned = signed.view(np.uint64)
    canonical = np.where(
        signed < 0,
        unsigned - np.uint64((1 << 64) - GOLDILOCKS_MODULUS),
        unsigned,
    )
    binding = economic_oracle_binding_v3(
        base_binding=base_binding,
        oracle_id=oracle.oracle_id,
        phase=oracle.phase,
        layer_index=oracle.layer_index,
        operation=oracle.operation,
        row_count=oracle.row_count,
        col_count=oracle.col_count,
        scale_bits=oracle.scale_bits,
    )
    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
    )

    leaf_binding = hashlib.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + binding
    ).digest()
    cell_count = _pow2(oracle.row_count) * col_pad
    leaf_count = cell_count // width
    flat = _fast_verified_chunk_values(
        oracle_root=oracle.root,
        leaf_binding=leaf_binding,
        leaf_count=leaf_count,
        width=width,
        chunks=chunks,
        opening=opening,
        external=canonical,
    )
    return flat is not None


def _fast_verified_chunk_values(
    *, oracle_root, leaf_binding, leaf_count, width, chunks, opening,
    external,
):
    """Compiled multiproof verification straight from wire values.

    Returns the FLAT canonical field values of the opened chunks after
    the root reconstructed, or ``None`` to fall back to the reference
    path.  Every reference-path expectation is replicated: exact
    sibling coordinate schedule, leaf-count binding, value alignment;
    the byte-exact root compare rejects any value the committed tree
    does not hold."""

    import struct as _struct

    from verallm.proof_v3.c_multiopen import (
        reconstruct_wn,
        sibling_coordinates,
    )
    from verallm.proof_v3.goldilocks_merkle_reference import (
        _LEAF_DOMAIN,
        _NODE_DOMAIN,
        _ROOT_DOMAIN,
    )

    coords = sibling_coordinates(leaf_count, chunks)
    if coords is None:
        return None
    if tuple((s.level, s.index) for s in opening.siblings) != tuple(coords):
        raise ProofV3VerificationError(
            "Merkle opening sibling coordinates are incomplete, duplicate, "
            "or reordered"
        )
    if int(opening.leaf_count) != int(leaf_count):
        raise ProofV3VerificationError(
            "Merkle opening tree metadata is unexpected"
        )
    if opening.value_mode == 0 and external is None:
        raise ProofV3VerificationError(
            "external-value opening needs site-supplied values"
        )
    if external is not None:
        dtype = getattr(external, "dtype", None)
        if (
            dtype is not None
            and dtype.kind == "u"
            and dtype.itemsize == 8
            and getattr(external, "ndim", 0) == 1
        ):
            flat = external
        else:
            flat = [int(v) for v in external]
    else:
        mode = opening.value_mode
        if mode == 1:
            flat = list(opening.values)
        else:
            # int8 / bounded signed transport -> canonical field mapping
            flat = [
                v + GOLDILOCKS_MODULUS if v < 0 else v
                for v in opening.values
            ]
    if len(flat) != len(chunks) * width:
        raise ProofV3VerificationError(
            "opening values do not align with the derived leaves"
        )
    import hashlib as _hl

    header = leaf_binding + _struct.pack("<II", leaf_count, width)
    result = reconstruct_wn(
        _LEAF_DOMAIN + header,
        _NODE_DOMAIN + header,
        leaf_count,
        list(chunks),
        flat,
        width,
        [c[0] for c in coords],
        [c[1] for c in coords],
        b"".join(s.digest for s in opening.siblings),
    )
    if result is None:
        return None
    if not isinstance(result, bytes):
        raise ProofV3VerificationError(
            "Merkle opening reconstruction is incomplete"
        )
    commitment = _hl.sha256(_ROOT_DOMAIN + header + result).digest()
    if commitment != oracle_root:
        raise ProofV3VerificationError(
            "Merkle opening does not match commitment"
        )
    return flat
