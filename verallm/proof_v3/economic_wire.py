"""Canonical bounded wire ABI for the economic_recompute_v3 hard audit.

This is the ONLY transport for an economic recompute proof.  Every field is
bytes, int, or a canonical ASCII identifier -- no objects, dicts or callables
travel on the wire.  Encoding is strict and versioned: lengths and counts are
bounded, indices must be sorted and distinct, and decoding re-encodes the
parsed structure and requires byte equality, so a malformed, truncated,
non-canonical or trailing-bytes payload always fails closed.

Wire sections:

* header + envelope binding (envelope/profile digests, request-bound capture
  digest) -- ties the proof to one frozen commitment envelope;
* EXECUTION ANCHORS -- pre-nonce full-sequence raw runtime roots plus bounded
  post-nonce row openings for the exact validator-selected positions;
* ORACLE INVENTORY -- exact-int8 recompute commitments with bound
  phase/layer/operation/dims/scale.  Legacy profiles freeze the complete
  inventory pre-nonce.  Streaming profiles freeze the execution anchors and
  bounded response-stamp input plus terminal output commitments instead;
  selected layer oracles are rebuilt post-nonce and must be cross-bound to
  authenticated raw anchor openings (see
  :func:`economic_execution_root_v3`);
* PROJECTION reveals -- capture-opened X rows + surrogate cell openings +
  manifest weight-row chunk openings for the exact recompute;
* CHAIN reveal -- residual[0] bottom anchor (embedding rows), complete
  per-layer boundary openings for connectivity, at challenge-sampled rows;
* FINAL reveal -- final-hidden opening + LM-head manifest rows and either
  sampled top-k certification or an exact full-vocabulary escalation.

The verifier (:mod:`economic_recompute_adapter`) derives every sampled
coordinate from the nonce-bound transcript; the reveal structures here must
match those verifier-derived selections exactly or verification fails.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorLaneOpeningV3,
    execution_anchor_inventory_digest_v3,
)
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    LM_HEAD_CATALOG_FOLD_COUNT_V3,
    MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3,
)
from verallm.proof_v3.lean_projection_batch import (
    MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3,
    decode_lean_projection_batch_v3,
)
from verallm.proof_v3.prefix_cache import (
    MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3,
    MAX_PREFIX_CACHE_STATE_STAGES_V3,
    PrefixCacheBlockOpeningV3,
    PrefixCacheBlockRecordV3,
    PrefixCacheCommitmentV3,
    PrefixCacheLaneRevealV3,
    PrefixCachePostnonceProofV3,
    PrefixCacheStateOpeningV3,
    PrefixCacheStateRecordV3,
)
from zkllm.types import MerklePath
# v21/v22: a bounded nested sparse-MoE section authenticates complete router,
# selected expert/shared-path, sampled down-row and route-independent capacity
# openings. v22 is the prefix-cache counterpart.
# v19/v20: lean compact proofs carry exact bounded replay cells for the
# nonce-selected fused SwiGLU relation. v20 is the prefix-cache counterpart.
# v17: checkpointed GDN couplings carry exact bounded replay source rows for
# RMSNorm, avoiding non-conservative intervals when candidate-pool int8 rows
# clip at later nonce-selected decode positions.
# v15: compact projection proofs carry a succinct complete-output relation;
# unpredictable escalations retain the complete-row relation.
# v14: compact terminal proofs omit pre-nonce full-vocabulary logits oracles;
# normal proofs carry a sampled top-k certificate and unpredictable
# escalations carry one complete signed-i64 logits row.
# v13: lean GDN suffix replay carries only nonce-selected Q/K/V/Z/BA/output
# coordinates while retaining every decode transition.
# v12: complete-output projection reveals omit redundant output indices while
# retaining only the nonce-sampled manifest rows used by the local couplings;
# the nested lean batch authenticates the complete projection relation.
# v11: compact complete-output projection batch for lean corridor proofs.
# v10: rational attention transport uses the scored-scheme-v2 section.
# v9: weightless full-vocabulary LM-head catalog folds.
# v8: manifest weight rows are reconstructed from their authenticated
# contiguous chunk opening instead of being serialized a second time.
# v7: architecture-specific GDN coupling reveals.
# v6: bounded nested lane openings into streaming execution-anchor rows.
# v5: streaming execution-anchor commitments and selected-row openings.
# v4: rational attention sections carry mandatory economic o_x openings,
# joining the attention transport to the transition oracle inventory.
# v3: COMPACT openings -- leaf indices never ride the wire (validator
# derives them from the nonce challenge); values travel per-mode
# (external / u64 field / packed int8). Pre-v4 proofs reject cleanly.
# v16: complete selected-trace proof bytes replace the legacy raw recompute
# sections in compact-only mode. Static/dynamic oracle inventories and
# execution-anchor commitments stay in the authenticated outer envelope.
ECONOMIC_WIRE_FORMAT_VERSION = 17
ECONOMIC_PREFIX_CACHE_WIRE_FORMAT_VERSION = 18
ECONOMIC_MLP_ACTIVATION_WIRE_FORMAT_VERSION = 19
ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION = 20
ECONOMIC_MOE_WIRE_FORMAT_VERSION = 21
ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION = 22
_WIRE_MAGIC = b"V3EW"

VALUE_MODE_EXTERNAL = 0
VALUE_MODE_FIELD = 1
VALUE_MODE_INT8 = 2
# bounded signed values at a narrow byte width both sides derive from the
# SIGNED oracle dims (int8 x int8 dot products: |value| < 128*128*in_dim)
VALUE_MODE_BOUNDED = 3


def bounded_byte_width_v3(in_dim: int) -> int:
    """Transport byte width of an int8-dot surrogate bounded by
    ``128 * 128 * in_dim`` (two's complement, sign bit included)."""

    if not isinstance(in_dim, int) or isinstance(in_dim, bool) or in_dim < 1:
        raise ProofV3Error("bounded width in_dim is malformed")
    return ((128 * 128 * in_dim).bit_length() + 8) // 8

GOLDILOCKS_FIELD_MODULUS = (1 << 64) - (1 << 32) + 1
_INVENTORY_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_ORACLE_INVENTORY/SHA256"
_EXECUTION_ROOT_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_EXECUTION_ROOT/SHA256"
_ANCHORED_EXECUTION_ROOT_DOMAIN = (
    b"VERATHOS/PROOF_V3/ECONOMIC_EXECUTION_ROOT/STREAMING_V1/SHA256"
)

# Sized by the SIGNED audit policy, not the context: with compact
# openings the hard proof is ~20MB at the release policy on 0.5B
# geometry (32 rows x 4 projections + full-boundary chain) and scales
# with model WIDTH, not context. 128MiB bounds the widest supported
# geometry (27B-class) with margin while staying a hard verifier-side
# parse bound.
MAX_ECONOMIC_WIRE_BYTES = 128 << 20
MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3 = 17 << 17
MAX_SELECTED_TRACE_WIRE_BYTES_V3 = 2 << 20
MAX_ORACLES = 4096
MAX_PROJECTION_REVEALS = 512
MAX_COUPLING_REVEALS = 64
MAX_REVEALED_ROWS = 256
MAX_REVEALED_LOGITS_V3 = 1 << 20
# The widest qualified Qwen3.6 GDN geometry has 256-wide Q/K heads and
# 128-wide V/Z heads. Seven selected value heads may occupy seven distinct
# key-head groups, requiring 7 * (2 * 256 + 2 * 128) authenticated output
# coordinates. The v9 selection ABI additionally unions 16 generic
# nonce-selected coordinates. Keep both parser bounds at that exact production
# maximum rather than assuming the narrower 128-wide Q/K geometry or forcing
# the structural and generic samples to overlap.
MAX_GDN_RUNTIME_PROJECTION_COLUMNS = 7 * (2 * 256 + 2 * 128)
MAX_PROJECTION_WEIGHT_ROWS = MAX_GDN_RUNTIME_PROJECTION_COLUMNS + 16
MAX_PROJECTION_OUT_INDICES = MAX_PROJECTION_WEIGHT_ROWS
# Wide enough for current and qualified near-term vocabulary rows while the
# signed profile fixes the exact model dimension and the outer transport cap
# still bounds total allocation/decoding work.
MAX_ROW_WIDTH = 1 << 20
MAX_OPENING_INDICES = 1 << 20
MAX_OPENING_SIBLINGS = 1 << 20
MAX_WEIGHT_CHUNK_BYTES = 1 << 16
MAX_WEIGHT_CHUNKS_PER_ROW = 4096
# contiguous-range multiproof: at most two edge siblings per tree level
MAX_WEIGHT_RANGE_SIBLINGS = 80
MAX_BOUNDARY_OPENINGS = 1024
MAX_CAPTURE_CHAIN_DIGEST_BYTES = 128
MAX_IDENTIFIER_BYTES = 64
MAX_EXECUTION_ANCHORS = 4096
MAX_EXECUTION_ANCHOR_REVEALS = 4096
MAX_EXECUTION_ANCHOR_ROWS = 256
MAX_EXECUTION_ANCHOR_LANE_REVEALS = 8192
MAX_EXECUTION_ANCHOR_ROW_BYTES = (1 << 24) - 1
MAX_EXECUTION_ANCHOR_SIBLINGS = 32

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,63}$")
_PHASE_CODES = {"prefill": 1, "decode": 2, "global": 3}
_PHASE_NAMES = {code: name for name, code in _PHASE_CODES.items()}

__all__ = [
    "ECONOMIC_WIRE_FORMAT_VERSION",
    "ECONOMIC_PREFIX_CACHE_WIRE_FORMAT_VERSION",
    "ECONOMIC_MLP_ACTIVATION_WIRE_FORMAT_VERSION",
    "ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION",
    "ECONOMIC_MOE_WIRE_FORMAT_VERSION",
    "ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION",
    "MAX_ECONOMIC_WIRE_BYTES",
    "MAX_SELECTED_TRACE_WIRE_BYTES_V3",
    "MAX_REVEALED_LOGITS_V3",
    "EconomicOracleCommitmentV3",
    "EconomicExecutionAnchorRowV3",
    "EconomicExecutionAnchorRevealV3",
    "EconomicExecutionAnchorLaneRevealV3",
    "EconomicMerkleSiblingV3",
    "EconomicLayerCouplingRevealV3",
    "EconomicGdnLayerCouplingRevealV3",
    "EconomicMlpActivationRevealV3",
    "EconomicMerkleOpeningV3",
    "EconomicWeightRowRevealV3",
    "EconomicProjectionRevealV3",
    "EconomicBoundaryOpeningV3",
    "EconomicAttentionRequestSectionV3",
    "EconomicChainRevealV3",
    "EconomicFinalRevealV3",
    "EconomicRecomputeProofV3",
    "economic_oracle_inventory_digest_v3",
    "economic_execution_root_v3",
    "encode_int8_row_v3",
    "decode_int8_row_v3",
    "scale_to_bits_v3",
    "bits_to_scale_v3",
    "bounded_byte_width_v3",
]


# ---------------------------------------------------------------------------
# primitive validators / codecs
# ---------------------------------------------------------------------------

def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _u_range(value: int, name: str, *, bits: int, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned {bits}-bit integer")
    if value < (1 if positive else 0) or value >= 1 << bits:
        raise ProofV3Error(f"{name} is out of the unsigned {bits}-bit range")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value


def _sorted_distinct(values, name: str, *, bits: int, maximum: int):
    values = tuple(values)
    if not values:
        raise ProofV3Error(f"{name} must not be empty")
    if len(values) > maximum:
        raise ProofV3Error(f"{name} count exceeds the wire bound")
    limit = 1 << bits
    previous = -1
    for item in values:
        # exact fast check; bool (an int subclass) falls to _u_range below
        if type(item) is int and previous < item < limit:
            previous = item
            continue
        _u_range(item, f"{name} entry", bits=bits)
        if item <= previous:
            raise ProofV3Error(f"{name} must be strictly increasing")
        previous = item
    return values


def encode_int8_row_v3(row) -> bytes:
    """Two's-complement int8 encoding of one revealed row."""

    try:
        # exact fast path: struct range-checks every value as signed int8
        return struct.pack(f"<{len(row)}b", *row)
    except (struct.error, TypeError):
        pass
    out = bytearray()
    for value in row:
        value = int(value)
        if not -128 <= value <= 127:
            raise ProofV3Error("int8 row value is out of range")
        out.append(value & 0xFF)
    return bytes(out)


def decode_int8_row_v3(data: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(data)}b", data)


def scale_to_bits_v3(scale: float) -> int:
    """Canonical u64 (IEEE-754 little-endian bits) for a quantization scale."""

    bits = struct.unpack("<Q", struct.pack("<d", float(scale)))[0]
    _validate_scale_bits(bits, "quantization scale")
    return bits


def bits_to_scale_v3(bits: int) -> float:
    _validate_scale_bits(bits, "quantization scale")
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _validate_scale_bits(bits: int, name: str) -> None:
    _u_range(bits, f"{name} bits", bits=64)
    value = struct.unpack("<d", struct.pack("<Q", bits))[0]
    if not (value > 0.0) or value != value or value in (float("inf"),):
        raise ProofV3Error(f"{name} must be a finite positive value")


class _Writer:
    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def raw(self, data: bytes) -> None:
        self._parts.append(data)

    def pack(self, format_string: str, *values) -> None:
        self._parts.append(struct.pack(format_string, *values))

    def identifier(self, value: str, name: str) -> None:
        encoded = _identifier(value, name).encode("ascii")
        self.pack("<B", len(encoded))
        self.raw(encoded)

    def vbytes(self, value: bytes, name: str, maximum: int) -> None:
        if not isinstance(value, bytes) or not value:
            raise ProofV3Error(f"{name} must be non-empty bytes")
        if len(value) > maximum:
            raise ProofV3Error(f"{name} exceeds the wire bound")
        self.pack("<I", len(value))
        self.raw(value)

    def finish(self) -> bytes:
        return b"".join(self._parts)


class _Reader:
    def __init__(self, encoded: bytes, name: str) -> None:
        if not isinstance(encoded, bytes):
            raise ProofV3Error(f"{name} must be bytes")
        if len(encoded) > MAX_ECONOMIC_WIRE_BYTES:
            raise ProofV3Error(f"{name} exceeds the wire byte limit")
        self._encoded = encoded
        self._offset = 0
        self._name = name

    def read(self, size: int) -> bytes:
        if size < 0 or self._offset + size > len(self._encoded):
            raise ProofV3Error(f"{self._name} is truncated")
        value = self._encoded[self._offset : self._offset + size]
        self._offset += size
        return value

    def unpack(self, format_string: str) -> tuple:
        return struct.unpack(format_string, self.read(struct.calcsize(format_string)))

    def identifier(self, name: str) -> str:
        size = self.unpack("<B")[0]
        if size == 0 or size > MAX_IDENTIFIER_BYTES:
            raise ProofV3Error(f"{name} length is out of range")
        try:
            value = self.read(size).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV3Error(f"{name} is not ASCII") from exc
        return _identifier(value, name)

    def vbytes(self, name: str, maximum: int) -> bytes:
        size = self.unpack("<I")[0]
        if size == 0 or size > maximum:
            raise ProofV3Error(f"{name} length is out of range")
        return self.read(size)

    def count(self, name: str, maximum: int, *, allow_zero: bool = False) -> int:
        value = self.unpack("<I")[0]
        if (value == 0 and not allow_zero) or value > maximum:
            raise ProofV3Error(f"{name} count is out of range")
        return value

    def finish(self) -> None:
        if self._offset != len(self._encoded):
            raise ProofV3Error(f"{self._name} has trailing bytes")


# ---------------------------------------------------------------------------
# streaming full-sequence execution anchors
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EconomicExecutionAnchorRowV3:
    """One raw runtime row plus its outer execution-anchor Merkle path."""

    row_index: int
    row_bytes: bytes
    sibling_hashes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _u_range(self.row_index, "execution anchor row_index", bits=32)
        if (
            not isinstance(self.row_bytes, bytes)
            or not 0 < len(self.row_bytes) <= MAX_EXECUTION_ANCHOR_ROW_BYTES
        ):
            raise ProofV3Error("execution anchor row bytes are malformed")
        siblings = tuple(self.sibling_hashes)
        if len(siblings) > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error(
                "execution anchor path exceeds the supported depth"
            )
        for sibling in siblings:
            _fixed32(sibling, "execution anchor sibling")
        object.__setattr__(self, "sibling_hashes", siblings)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<I", self.row_index)
        writer.vbytes(
            self.row_bytes,
            "execution anchor row",
            MAX_EXECUTION_ANCHOR_ROW_BYTES,
        )
        writer.pack("<B", len(self.sibling_hashes))
        for sibling in self.sibling_hashes:
            writer.raw(sibling)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicExecutionAnchorRowV3":
        row_index = reader.unpack("<I")[0]
        row_bytes = reader.vbytes(
            "execution anchor row",
            MAX_EXECUTION_ANCHOR_ROW_BYTES,
        )
        sibling_count = reader.unpack("<B")[0]
        if sibling_count > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error(
                "execution anchor path exceeds the supported depth"
            )
        return cls(
            row_index=row_index,
            row_bytes=row_bytes,
            sibling_hashes=tuple(
                reader.read(32) for _ in range(sibling_count)
            ),
        )


@dataclass(frozen=True, slots=True)
class EconomicExecutionAnchorRevealV3:
    """Canonical selected-row openings for one indexed anchor stage."""

    commitment_index: int
    rows: tuple[EconomicExecutionAnchorRowV3, ...]

    def __post_init__(self) -> None:
        _u_range(
            self.commitment_index,
            "execution anchor commitment_index",
            bits=32,
        )
        rows = tuple(self.rows)
        if not 0 < len(rows) <= MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error(
                "execution anchor revealed-row count is out of range"
            )
        if any(not isinstance(row, EconomicExecutionAnchorRowV3)
               for row in rows):
            raise ProofV3Error(
                "execution anchor row has an unexpected type"
            )
        indices = tuple(row.row_index for row in rows)
        if indices != tuple(sorted(set(indices))):
            raise ProofV3Error(
                "execution anchor rows must be sorted and distinct"
            )
        object.__setattr__(self, "rows", rows)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<II", self.commitment_index, len(self.rows))
        for row in self.rows:
            row.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicExecutionAnchorRevealV3":
        commitment_index, row_count = reader.unpack("<II")
        if not 0 < row_count <= MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error(
                "execution anchor revealed-row count is out of range"
            )
        return cls(
            commitment_index=commitment_index,
            rows=tuple(
                EconomicExecutionAnchorRowV3.decode(reader)
                for _ in range(row_count)
            ),
        )


@dataclass(frozen=True, slots=True)
class EconomicExecutionAnchorLaneRevealV3:
    """Canonical nested 2 KiB lane opening for one indexed anchor stage."""

    commitment_index: int
    opening: ExecutionAnchorLaneOpeningV3

    def __post_init__(self) -> None:
        _u_range(
            self.commitment_index,
            "execution anchor lane commitment_index",
            bits=32,
        )
        if not isinstance(self.opening, ExecutionAnchorLaneOpeningV3):
            raise ProofV3Error(
                "execution anchor lane opening has an unexpected type"
            )

    def encode(self, writer: _Writer) -> None:
        opening = self.opening
        writer.pack(
            "<IIIH",
            self.commitment_index,
            opening.row_index,
            opening.lane_index,
            len(opening.lane_bytes),
        )
        writer.raw(opening.lane_bytes)
        writer.pack("<B", len(opening.lane_sibling_hashes))
        for sibling in opening.lane_sibling_hashes:
            writer.raw(sibling)
        writer.pack("<B", len(opening.row_sibling_hashes))
        for sibling in opening.row_sibling_hashes:
            writer.raw(sibling)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicExecutionAnchorLaneRevealV3":
        commitment_index, row_index, lane_index, lane_byte_count = (
            reader.unpack("<IIIH")
        )
        if lane_byte_count not in (256, 2048):
            raise ProofV3Error(
                "execution anchor lane byte count is unsupported"
            )
        lane_bytes = reader.read(lane_byte_count)
        lane_sibling_count = reader.unpack("<B")[0]
        if lane_sibling_count > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error(
                "execution anchor lane path exceeds the supported depth"
            )
        lane_siblings = tuple(
            reader.read(32) for _ in range(lane_sibling_count)
        )
        row_sibling_count = reader.unpack("<B")[0]
        if row_sibling_count > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error(
                "execution anchor row path exceeds the supported depth"
            )
        return cls(
            commitment_index=commitment_index,
            opening=ExecutionAnchorLaneOpeningV3(
                row_index=row_index,
                lane_index=lane_index,
                lane_bytes=lane_bytes,
                lane_sibling_hashes=lane_siblings,
                row_sibling_hashes=tuple(
                    reader.read(32) for _ in range(row_sibling_count)
                ),
            ),
        )


# ---------------------------------------------------------------------------
# oracle inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EconomicOracleCommitmentV3:
    """One pre-nonce capture-root commitment with bound metadata.

    The root commits the exact tensor (Goldilocks capture tree); the
    metadata binds oracle identity, phase, layer, operation, dimensions and
    quantization scale so a reveal cannot be re-interpreted under different
    semantics after the nonce.
    """

    oracle_id: str
    phase: str
    layer_index: int
    operation: str
    row_count: int
    col_count: int
    scale_bits: int
    root: bytes

    def __post_init__(self) -> None:
        _identifier(self.oracle_id, "oracle_id")
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("oracle phase is not supported")
        _u_range(self.layer_index, "oracle layer_index", bits=32)
        _identifier(self.operation, "oracle operation")
        _u_range(self.row_count, "oracle row_count", bits=32, positive=True)
        _u_range(self.col_count, "oracle col_count", bits=32, positive=True)
        if self.col_count > MAX_ROW_WIDTH:
            raise ProofV3Error("oracle col_count exceeds the wire bound")
        _validate_scale_bits(self.scale_bits, "oracle scale")
        _fixed32(self.root, "oracle root", nonzero=True)

    def encode(self, writer: _Writer) -> None:
        writer.identifier(self.oracle_id, "oracle_id")
        writer.pack("<B", _PHASE_CODES[self.phase])
        writer.identifier(self.operation, "oracle operation")
        writer.pack(
            "<IIIQ",
            self.layer_index,
            self.row_count,
            self.col_count,
            self.scale_bits,
        )
        writer.raw(self.root)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicOracleCommitmentV3":
        oracle_id = reader.identifier("oracle_id")
        phase_code = reader.unpack("<B")[0]
        phase = _PHASE_NAMES.get(phase_code)
        if phase is None:
            raise ProofV3Error("oracle phase code is not supported")
        operation = reader.identifier("oracle operation")
        layer_index, row_count, col_count, scale_bits = reader.unpack("<IIIQ")
        return cls(
            oracle_id=oracle_id,
            phase=phase,
            layer_index=layer_index,
            operation=operation,
            row_count=row_count,
            col_count=col_count,
            scale_bits=scale_bits,
            root=reader.read(32),
        )

    def canonical_bytes(self) -> bytes:
        writer = _Writer()
        self.encode(writer)
        return writer.finish()


def economic_oracle_inventory_digest_v3(
    oracles: tuple[EconomicOracleCommitmentV3, ...],
) -> bytes:
    """Digest of the exact ordered oracle inventory (all roots + metadata)."""

    if not oracles or len(oracles) > MAX_ORACLES:
        raise ProofV3Error("oracle inventory count is out of range")
    hasher = hashlib.sha256(_INVENTORY_DOMAIN + struct.pack("<I", len(oracles)))
    for oracle in oracles:
        if not isinstance(oracle, EconomicOracleCommitmentV3):
            raise ProofV3Error("oracle inventory entry has an unexpected type")
        encoded = oracle.canonical_bytes()
        hasher.update(struct.pack("<I", len(encoded)))
        hasher.update(encoded)
    return hasher.digest()


def economic_execution_root_v3(
    *,
    oracles: tuple[EconomicOracleCommitmentV3, ...],
    capture_chain_digest: bytes,
    execution_anchors: tuple[ExecutionAnchorCommitmentV3, ...] = (),
) -> bytes:
    """The value ``envelope.execution_root`` MUST equal for an economic proof.

    Legacy mode freezes the complete oracle inventory.  Streaming mode freezes
    full-sequence runtime anchors plus a bounded prompt-tail input and terminal
    hidden commitment. Legacy streaming profiles additionally freeze logits;
    compact-v9 profiles select their terminal relation post-nonce. The exact
    signed inventory check decides which form is admissible.
    """

    if (
        not isinstance(capture_chain_digest, bytes)
        or not capture_chain_digest
        or len(capture_chain_digest) > MAX_CAPTURE_CHAIN_DIGEST_BYTES
    ):
        raise ProofV3Error("capture_chain_digest length is out of range")
    execution_anchors = tuple(execution_anchors)
    if execution_anchors:
        # In streaming mode, layer-oracle rows are reconstructed only after
        # the nonce selects absolute sequence positions.  They must therefore
        # not influence the nonce transcript.  The full runtime anchor
        # inventory is frozen instead, together with the bounded response
        # stamp input and terminal hidden commitment. Legacy profiles may
        # additionally include pre-nonce logits.
        precommit_oracles = tuple(
            oracle
            for oracle in oracles
            if oracle.operation
            in ("final_hidden", "logits", "response_stamp_input")
        )
        operations = {oracle.operation for oracle in precommit_oracles}
        if not {"final_hidden", "response_stamp_input"}.issubset(
            operations
        ):
            raise ProofV3Error(
                "streaming execution root lacks its bounded pre-nonce "
                "response-stamp or terminal commitments"
            )
        return hashlib.sha256(
            _ANCHORED_EXECUTION_ROOT_DOMAIN
            + execution_anchor_inventory_digest_v3(execution_anchors)
            + economic_oracle_inventory_digest_v3(precommit_oracles)
            + struct.pack("<I", len(capture_chain_digest))
            + capture_chain_digest
        ).digest()
    return hashlib.sha256(
        _EXECUTION_ROOT_DOMAIN
        + economic_oracle_inventory_digest_v3(oracles)
        + struct.pack("<I", len(capture_chain_digest))
        + capture_chain_digest
    ).digest()


# ---------------------------------------------------------------------------
# capture-tree openings (serialized Goldilocks multiproofs, leaf width 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EconomicMerkleSiblingV3:
    level: int
    index: int
    digest: bytes

    def __post_init__(self) -> None:
        _u_range(self.level, "opening sibling level", bits=32)
        _u_range(self.index, "opening sibling index", bits=32)
        _fixed32(self.digest, "opening sibling digest")


@dataclass(frozen=True, slots=True)
class EconomicMerkleOpeningV3:
    """One serialized capture-tree multiproof (Goldilocks chunk leaves).

    COMPACT wire form (format v3): leaf INDICES never ride the wire --
    the validator derives them from the nonce challenge and rejects
    any disagreement through Merkle reconstruction. ``value_mode``
    picks how leaf values travel:

    * ``VALUE_MODE_EXTERNAL`` (0): not carried; the verification site
      supplies them from data already on the wire exactly once (e.g.
      the packed int8 X rows) -- reconstruction against the committed
      root is what authenticates them;
    * ``VALUE_MODE_FIELD`` (1): canonical u64 field values, 8 bytes
      each (wide-range payloads);
    * ``VALUE_MODE_INT8`` (2): two's-complement int8, 1 byte each
      (int8-reduced captures: residuals, hidden rows);
    * ``VALUE_MODE_BOUNDED`` (3): two's-complement at ``bounded_width``
      bytes (int8-dot surrogates/logits; the verifier re-derives the
      width from the SIGNED dims and rejects any other).
    """

    binding_digest: bytes
    leaf_count: int
    value_mode: int
    values: tuple[int, ...] | None
    siblings: tuple[EconomicMerkleSiblingV3, ...]
    bounded_width: int | None = None

    def __post_init__(self) -> None:
        _fixed32(self.binding_digest, "opening binding_digest")
        _u_range(self.leaf_count, "opening leaf_count", bits=64, positive=True)
        mode = self.value_mode
        if type(mode) is not int or mode not in (
            VALUE_MODE_EXTERNAL, VALUE_MODE_FIELD, VALUE_MODE_INT8,
            VALUE_MODE_BOUNDED,
        ):
            raise ProofV3Error("opening value_mode is malformed")
        width = self.bounded_width
        if mode == VALUE_MODE_BOUNDED:
            if type(width) is not int or not 1 <= width <= 8:
                raise ProofV3Error("bounded opening width is malformed")
        elif width is not None:
            raise ProofV3Error(
                "opening width is only meaningful for bounded mode")
        values = self.values
        if mode == VALUE_MODE_EXTERNAL:
            if values is not None:
                raise ProofV3Error(
                    "external-value opening must not carry values")
        else:
            if values is None:
                raise ProofV3Error("opening carries no values for its mode")
            values = tuple(values)
            if not 0 < len(values) <= MAX_OPENING_INDICES:
                raise ProofV3Error("opening value count is out of range")
            if mode == VALUE_MODE_FIELD:
                for value in values:
                    if type(value) is not int or not 0 <= value < (1 << 64):
                        _u_range(value, "opening value", bits=64)
            elif mode == VALUE_MODE_BOUNDED:
                bound = 1 << (8 * width - 1)
                for value in values:
                    if type(value) is not int or not -bound <= value < bound:
                        raise ProofV3Error(
                            "bounded opening value is out of range")
            else:
                for value in values:
                    if type(value) is not int or not -128 <= value <= 127:
                        raise ProofV3Error(
                            "int8 opening value is out of range")
        siblings = tuple(self.siblings)
        if len(siblings) > MAX_OPENING_SIBLINGS:
            raise ProofV3Error("opening sibling count exceeds the wire bound")
        for sibling in siblings:
            if not isinstance(sibling, EconomicMerkleSiblingV3):
                raise ProofV3Error("opening sibling has an unexpected type")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "siblings", siblings)

    def encode(self, writer: _Writer) -> None:
        writer.raw(self.binding_digest)
        count = 0 if self.values is None else len(self.values)
        writer.pack(
            "<QBII", self.leaf_count, self.value_mode, count,
            len(self.siblings))
        if self.value_mode == VALUE_MODE_FIELD:
            writer.raw(struct.pack(f"<{count}Q", *self.values))
        elif self.value_mode == VALUE_MODE_INT8:
            writer.raw(struct.pack(f"<{count}b", *self.values))
        elif self.value_mode == VALUE_MODE_BOUNDED:
            width = self.bounded_width
            writer.pack("<B", width)
            writer.raw(b"".join(
                value.to_bytes(width, "little", signed=True)
                for value in self.values
            ))
        for sibling in self.siblings:
            writer.pack("<II", sibling.level, sibling.index)
            writer.raw(sibling.digest)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMerkleOpeningV3":
        binding_digest = reader.read(32)
        leaf_count, value_mode, count, sibling_count = reader.unpack("<QBII")
        if value_mode not in (
            VALUE_MODE_EXTERNAL, VALUE_MODE_FIELD, VALUE_MODE_INT8,
            VALUE_MODE_BOUNDED,
        ):
            raise ProofV3Error("opening value_mode is malformed")
        bounded_width = None
        if value_mode == VALUE_MODE_EXTERNAL:
            if count != 0:
                raise ProofV3Error(
                    "external-value opening must not carry values")
            values = None
        else:
            if count == 0 or count > MAX_OPENING_INDICES:
                raise ProofV3Error("opening value count is out of range")
            if value_mode == VALUE_MODE_FIELD:
                values = struct.unpack(f"<{count}Q", reader.read(8 * count))
            elif value_mode == VALUE_MODE_BOUNDED:
                bounded_width = reader.unpack("<B")[0]
                if not 1 <= bounded_width <= 8:
                    raise ProofV3Error("bounded opening width is malformed")
                blob = reader.read(bounded_width * count)
                values = tuple(
                    int.from_bytes(
                        blob[start:start + bounded_width], "little",
                        signed=True,
                    )
                    for start in range(0, len(blob), bounded_width)
                )
            else:
                values = struct.unpack(f"<{count}b", reader.read(count))
        if sibling_count > MAX_OPENING_SIBLINGS:
            raise ProofV3Error("opening sibling count is out of range")
        siblings = []
        for _ in range(sibling_count):
            level, index = reader.unpack("<II")
            siblings.append(
                EconomicMerkleSiblingV3(
                    level=level, index=index, digest=reader.read(32)
                )
            )
        return cls(
            binding_digest=binding_digest,
            leaf_count=leaf_count,
            value_mode=value_mode,
            values=values,
            siblings=tuple(siblings),
            bounded_width=bounded_width,
        )

    def to_reference_with(self, *, indices, external_values=None,
                          leaf_width: int = 1):
        """Rebuild the reference multiproof for verification.

        ``indices``: the VALIDATOR-derived sorted leaf indices (never
        trusted from the wire). ``leaf_width``: the VALIDATOR-derived
        tree leaf width; wire values are flat and group into
        ``leaf_width``-cell leaf rows. ``external_values``: canonical
        field values for ``VALUE_MODE_EXTERNAL`` openings, flat and
        aligned to ``indices`` x ``leaf_width`` -- reconstruction
        against the committed root is what authenticates them."""

        from verallm.proof_v3.goldilocks_merkle_reference import (
            GoldilocksMerkleMultiOpeningReference,
            GoldilocksMerkleSiblingReference,
        )

        width = int(leaf_width)
        if width < 1:
            raise ProofV3Error("opening leaf width is malformed")
        indices = tuple(int(index) for index in indices)
        if self.value_mode == VALUE_MODE_EXTERNAL:
            if external_values is None:
                raise ProofV3Error(
                    "external-value opening needs site-supplied values")
            values = tuple(int(v) for v in external_values)
        elif self.value_mode in (VALUE_MODE_INT8, VALUE_MODE_BOUNDED):
            values = tuple(
                value % GOLDILOCKS_FIELD_MODULUS for value in self.values)
        else:
            values = self.values
        if len(values) != len(indices) * width:
            raise ProofV3Error(
                "opening values do not align with the derived leaves")
        return GoldilocksMerkleMultiOpeningReference(
            binding_digest=self.binding_digest,
            leaf_count=self.leaf_count,
            leaf_width=width,
            indices=indices,
            rows=tuple(
                values[start:start + width]
                for start in range(0, len(values), width)
            ),
            siblings=tuple(
                GoldilocksMerkleSiblingReference(
                    level=sibling.level,
                    index=sibling.index,
                    digest=sibling.digest,
                )
                for sibling in self.siblings
            ),
        )

    @classmethod
    def from_reference(
        cls, opening, *, value_mode: int = 1, bounded_width: int | None = None
    ) -> "EconomicMerkleOpeningV3":
        # any reference leaf width: wire values are flat; the verifier
        # re-derives the width from the oracle geometry it owns
        from itertools import chain

        if value_mode == VALUE_MODE_EXTERNAL:
            values = None
        else:
            raw = tuple(chain.from_iterable(opening.rows))
            if value_mode == VALUE_MODE_INT8:
                values = []
                for value in raw:
                    signed = (
                        value - GOLDILOCKS_FIELD_MODULUS
                        if value > (GOLDILOCKS_FIELD_MODULUS >> 1)
                        else value
                    )
                    if not -128 <= signed <= 127:
                        raise ProofV3Error(
                            "int8 opening mode over a non-int8 capture")
                    values.append(int(signed))
                values = tuple(values)
            elif value_mode == VALUE_MODE_BOUNDED:
                if type(bounded_width) is not int or not 1 <= bounded_width <= 8:
                    raise ProofV3Error("bounded opening width is malformed")
                bound = 1 << (8 * bounded_width - 1)
                values = []
                for value in raw:
                    signed = (
                        value - GOLDILOCKS_FIELD_MODULUS
                        if value > (GOLDILOCKS_FIELD_MODULUS >> 1)
                        else value
                    )
                    if not -bound <= signed < bound:
                        raise ProofV3Error(
                            "bounded opening mode over an out-of-bound capture")
                    values.append(int(signed))
                values = tuple(values)
            else:
                values = tuple(int(v) for v in raw)
        return cls(
            binding_digest=opening.binding_digest,
            leaf_count=opening.leaf_count,
            value_mode=int(value_mode),
            values=values,
            bounded_width=(
                bounded_width if value_mode == VALUE_MODE_BOUNDED else None
            ),
            siblings=tuple(
                EconomicMerkleSiblingV3(
                    level=sibling.level,
                    index=sibling.index,
                    digest=sibling.digest,
                )
                for sibling in opening.siblings
            ),
        )


# ---------------------------------------------------------------------------
# manifest weight-row reveals (FlatWeightMerkle chunk openings)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EconomicWeightRowRevealV3:
    """One int8 weight row opened against a signed-manifest FlatWeightMerkle.

    The row's required chunks are CONTIGUOUS leaves of the flat tree, so
    the reveal carries their raw bytes plus the deduplicated RANGE
    multiproof: at most two edge siblings per tree level in climb order,
    instead of one full authentication path per chunk.  The verifier
    re-derives the exact chunk range, per-chunk byte lengths and total
    leaf count from the SIGNED manifest geometry, reconstructs the row
    directly from those authenticated chunks, and rejects a blob or sibling
    set for any other range."""

    row_index: int
    chunk_blob: bytes
    range_siblings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _u_range(self.row_index, "weight row_index", bits=32)
        if (
            not isinstance(self.chunk_blob, bytes)
            or not self.chunk_blob
            or len(self.chunk_blob)
            > MAX_WEIGHT_CHUNKS_PER_ROW * MAX_WEIGHT_CHUNK_BYTES
        ):
            raise ProofV3Error("weight row chunk blob length is out of range")
        siblings = tuple(self.range_siblings)
        if len(siblings) > MAX_WEIGHT_RANGE_SIBLINGS:
            raise ProofV3Error("weight row range proof exceeds the wire bound")
        for digest in siblings:
            _fixed32(digest, "weight row range sibling")
        object.__setattr__(self, "range_siblings", siblings)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<I", self.row_index)
        writer.vbytes(
            self.chunk_blob, "weight row chunk blob",
            MAX_WEIGHT_CHUNKS_PER_ROW * MAX_WEIGHT_CHUNK_BYTES)
        writer.pack("<H", len(self.range_siblings))
        for digest in self.range_siblings:
            writer.raw(digest)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicWeightRowRevealV3":
        row_index = reader.unpack("<I")[0]
        chunk_blob = reader.vbytes(
            "weight row chunk blob",
            MAX_WEIGHT_CHUNKS_PER_ROW * MAX_WEIGHT_CHUNK_BYTES)
        sibling_count = reader.unpack("<H")[0]
        if sibling_count > MAX_WEIGHT_RANGE_SIBLINGS:
            raise ProofV3Error("weight row range proof exceeds the wire bound")
        siblings = tuple(reader.read(32) for _ in range(sibling_count))
        return cls(
            row_index=row_index, chunk_blob=chunk_blob,
            range_siblings=siblings,
        )


# ---------------------------------------------------------------------------
# projection / chain / final reveal sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EconomicProjectionRevealV3:
    """Sampled exact-recompute reveal for one committed projection."""

    x_oracle_index: int
    s_oracle_index: int
    manifest_name: str
    token_indices: tuple[int, ...]
    x_rows: tuple[bytes, ...]
    x_opening: EconomicMerkleOpeningV3
    out_indices: tuple[int, ...]
    s_opening: EconomicMerkleOpeningV3
    weight_rows: tuple[EconomicWeightRowRevealV3, ...]
    complete_output: bool = False
    succinct_output: bool = False

    def __post_init__(self) -> None:
        _u_range(self.x_oracle_index, "projection x_oracle_index", bits=16)
        _u_range(self.s_oracle_index, "projection s_oracle_index", bits=16)
        if self.x_oracle_index == self.s_oracle_index:
            raise ProofV3Error("projection X and surrogate oracles must differ")
        _identifier(self.manifest_name, "projection manifest_name")
        tokens = _sorted_distinct(
            self.token_indices,
            "projection token_indices",
            bits=32,
            maximum=MAX_REVEALED_ROWS,
        )
        rows = tuple(self.x_rows)
        if len(rows) != len(tokens):
            raise ProofV3Error("projection X rows do not match token_indices")
        for row in rows:
            if (
                not isinstance(row, bytes)
                or not row
                or len(row) > MAX_ROW_WIDTH
            ):
                raise ProofV3Error("projection X row length is out of range")
        if not isinstance(self.x_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error("projection x_opening has an unexpected type")
        complete = self.complete_output
        if type(complete) is not bool:
            raise ProofV3Error(
                "projection complete_output must be boolean"
            )
        succinct = self.succinct_output
        if type(succinct) is not bool:
            raise ProofV3Error(
                "projection succinct_output must be boolean"
            )
        if complete and succinct:
            raise ProofV3Error(
                "projection output modes are mutually exclusive"
            )
        if complete:
            outs = tuple(self.out_indices)
            if outs:
                raise ProofV3Error(
                    "complete projection must omit output indices"
                )
        else:
            outs = _sorted_distinct(
                self.out_indices,
                "projection out_indices",
                bits=32,
                maximum=MAX_PROJECTION_OUT_INDICES,
            )
        if not isinstance(self.s_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error("projection s_opening has an unexpected type")
        weight_rows = tuple(self.weight_rows)
        if complete:
            signed_norm_qkv = self.manifest_name.endswith(".qkv")
            if (
                (not weight_rows and not signed_norm_qkv)
                or len(weight_rows) > MAX_PROJECTION_WEIGHT_ROWS
                or not all(
                    isinstance(reveal, EconomicWeightRowRevealV3)
                    for reveal in weight_rows
                )
            ):
                raise ProofV3Error(
                    "complete projection sampled weight rows are malformed"
                )
            indices = tuple(reveal.row_index for reveal in weight_rows)
            if indices != tuple(sorted(set(indices))):
                raise ProofV3Error(
                    "complete projection sampled weight rows are malformed"
                )
        elif succinct:
            if weight_rows:
                raise ProofV3Error(
                    "succinct projection must omit sampled weight rows"
                )
        elif len(weight_rows) != len(outs):
            raise ProofV3Error("projection weight rows do not match out_indices")
        if not succinct:
            for reveal, out_index in zip(
                weight_rows,
                outs,
                strict=not complete,
            ):
                if not isinstance(reveal, EconomicWeightRowRevealV3):
                    raise ProofV3Error(
                        "projection weight row has an unexpected type"
                    )
                if not complete and reveal.row_index != out_index:
                    raise ProofV3Error(
                        "projection weight rows must match out_indices in order"
                    )
        object.__setattr__(self, "token_indices", tokens)
        object.__setattr__(self, "x_rows", rows)
        object.__setattr__(self, "out_indices", outs)
        object.__setattr__(self, "weight_rows", weight_rows)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<HH", self.x_oracle_index, self.s_oracle_index)
        writer.identifier(self.manifest_name, "projection manifest_name")
        writer.pack(
            "<B",
            1 if self.complete_output else 2 if self.succinct_output else 0,
        )
        writer.pack("<I", len(self.token_indices))
        for token in self.token_indices:
            writer.pack("<I", token)
        for row in self.x_rows:
            writer.vbytes(row, "projection X row", MAX_ROW_WIDTH)
        self.x_opening.encode(writer)
        if not self.complete_output:
            writer.pack("<I", len(self.out_indices))
            for out_index in self.out_indices:
                writer.pack("<I", out_index)
        self.s_opening.encode(writer)
        writer.pack("<I", len(self.weight_rows))
        for reveal in self.weight_rows:
            reveal.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicProjectionRevealV3":
        x_oracle_index, s_oracle_index = reader.unpack("<HH")
        manifest_name = reader.identifier("projection manifest_name")
        output_mode = reader.unpack("<B")[0]
        if output_mode not in (0, 1, 2):
            raise ProofV3Error(
                "projection output mode is not canonical"
            )
        complete_output = output_mode == 1
        succinct_output = output_mode == 2
        token_count = reader.count("projection tokens", MAX_REVEALED_ROWS)
        tokens = tuple(reader.unpack("<I")[0] for _ in range(token_count))
        rows = tuple(
            reader.vbytes("projection X row", MAX_ROW_WIDTH)
            for _ in range(token_count)
        )
        x_opening = EconomicMerkleOpeningV3.decode(reader)
        if complete_output:
            outs = ()
        else:
            out_count = reader.count(
                "projection outs", MAX_PROJECTION_OUT_INDICES
            )
            outs = tuple(reader.unpack("<I")[0] for _ in range(out_count))
        s_opening = EconomicMerkleOpeningV3.decode(reader)
        weight_count = reader.count(
            "projection weight rows",
            (
                MAX_PROJECTION_WEIGHT_ROWS
                if complete_output
                else MAX_PROJECTION_OUT_INDICES
            ),
            allow_zero=(
                succinct_output
                or (
                    complete_output
                    and manifest_name.endswith(".qkv")
                )
            ),
        )
        weight_rows = tuple(
            EconomicWeightRowRevealV3.decode(reader)
            for _ in range(weight_count)
        )
        return cls(
            x_oracle_index=x_oracle_index,
            s_oracle_index=s_oracle_index,
            manifest_name=manifest_name,
            token_indices=tokens,
            x_rows=rows,
            x_opening=x_opening,
            out_indices=outs,
            s_opening=s_opening,
            weight_rows=weight_rows,
            complete_output=complete_output,
            succinct_output=succinct_output,
        )


@dataclass(frozen=True, slots=True)
class EconomicMlpActivationRevealV3:
    """Exact selected runtime cells for one fused SwiGLU relation.

    Each row carries gate, up and fused down-input values at the validator's
    nonce-derived intermediate columns. Values use the signed execution-anchor
    fp16/bf16 encoding and are joined to the independently opened int8
    projection cells by the verifier.
    """

    layer_index: int
    rows: tuple[tuple[int, bytes, bytes, bytes], ...] = ()

    def __post_init__(self) -> None:
        _u_range(self.layer_index, "MLP activation layer_index", bits=32)
        rows = tuple(self.rows)
        if not rows or len(rows) > MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error("MLP activation row count is out of range")
        previous_position = -1
        expected_width = None
        for record in rows:
            if not isinstance(record, tuple) or len(record) != 4:
                raise ProofV3Error("MLP activation row is malformed")
            position, gate_bytes, up_bytes, down_bytes = record
            _u_range(position, "MLP activation row position", bits=64)
            if position <= previous_position:
                raise ProofV3Error(
                    "MLP activation rows must be strictly increasing"
                )
            previous_position = position
            widths = tuple(
                len(value)
                for value in (gate_bytes, up_bytes, down_bytes)
                if isinstance(value, bytes)
            )
            if (
                len(widths) != 3
                or not widths[0]
                or len(set(widths)) != 1
                or widths[0] > MAX_EXECUTION_ANCHOR_ROW_BYTES
                or (expected_width is not None and widths[0] != expected_width)
            ):
                raise ProofV3Error(
                    "MLP activation cell bytes have inconsistent widths"
                )
            expected_width = widths[0]
        object.__setattr__(self, "rows", rows)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<II", self.layer_index, len(self.rows))
        for position, gate_bytes, up_bytes, down_bytes in self.rows:
            writer.pack("<Q", position)
            for value in (gate_bytes, up_bytes, down_bytes):
                writer.vbytes(
                    value,
                    "MLP activation cells",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                )

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMlpActivationRevealV3":
        layer_index, row_count = reader.unpack("<II")
        if not row_count or row_count > MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error("MLP activation row count is out of range")
        rows = tuple(
            (
                reader.unpack("<Q")[0],
                reader.vbytes(
                    "MLP gate cells", MAX_EXECUTION_ANCHOR_ROW_BYTES
                ),
                reader.vbytes(
                    "MLP up cells", MAX_EXECUTION_ANCHOR_ROW_BYTES
                ),
                reader.vbytes(
                    "MLP down-input cells", MAX_EXECUTION_ANCHOR_ROW_BYTES
                ),
            )
            for _ in range(row_count)
        )
        return cls(layer_index=layer_index, rows=rows)


@dataclass(frozen=True, slots=True)
class EconomicLayerCouplingRevealV3:
    """Per-selected-layer coupling reveal: binds the captured runtime
    outputs (Y), the residual stream and the K/V cache to the audited
    surrogates -- corridors, residual compositions, the elementwise MLP
    link and both RMSNorm links (GAP1 closure)."""

    layer_index: int
    attn_o_y_oracle_index: int
    attn_o_y_opening: EconomicMerkleOpeningV3   # FULL rows @ sampled tokens
    down_y_oracle_index: int
    down_y_opening: EconomicMerkleOpeningV3     # FULL rows @ sampled tokens
    mid_oracle_index: int
    mid_opening: EconomicMerkleOpeningV3        # FULL rows @ sampled tokens
    gate_up_y_oracle_index: int
    gate_up_y_opening: EconomicMerkleOpeningV3  # sampled corridor + mlp cells
    k_oracle_index: int
    k_opening: EconomicMerkleOpeningV3          # tokens x sampled kv cols
    v_oracle_index: int
    v_opening: EconomicMerkleOpeningV3          # tokens x sampled kv cols
    qkv_s_kv_opening: EconomicMerkleOpeningV3   # qkv_s cells @ kv-corridor outs
    qkv_kv_weight_rows: tuple[EconomicWeightRowRevealV3, ...]
    input_norm_row: EconomicWeightRowRevealV3
    post_norm_row: EconomicWeightRowRevealV3
    # Manifest-bound projection biases: (audited-projection index, bias row)
    # pairs, strictly increasing by index.  WHICH projections must reveal a
    # bias is validator-derived from the signed manifest (an entry named
    # "l{L}.{proj}_bias"), never from the proof -- model families without
    # biased projections simply register none.
    bias_rows: tuple[tuple[int, EconomicWeightRowRevealV3], ...] = ()

    def __post_init__(self) -> None:
        _u_range(self.layer_index, "coupling layer_index", bits=32)
        for name in (
            "attn_o_y_oracle_index",
            "down_y_oracle_index",
            "mid_oracle_index",
            "gate_up_y_oracle_index",
            "k_oracle_index",
            "v_oracle_index",
        ):
            _u_range(getattr(self, name), f"coupling {name}", bits=16)
        for name in (
            "attn_o_y_opening",
            "down_y_opening",
            "mid_opening",
            "gate_up_y_opening",
            "k_opening",
            "v_opening",
            "qkv_s_kv_opening",
        ):
            if not isinstance(getattr(self, name), EconomicMerkleOpeningV3):
                raise ProofV3Error(f"coupling {name} has an unexpected type")
        weight_rows = tuple(self.qkv_kv_weight_rows)
        if len(weight_rows) > MAX_WEIGHT_CHUNKS_PER_ROW:
            raise ProofV3Error("coupling kv weight row count is out of range")
        previous = -1
        for row in weight_rows:
            if not isinstance(row, EconomicWeightRowRevealV3):
                raise ProofV3Error("coupling kv weight row has an unexpected type")
            if row.row_index <= previous:
                raise ProofV3Error(
                    "coupling kv weight rows must be strictly increasing"
                )
            previous = row.row_index
        for name in ("input_norm_row", "post_norm_row"):
            if not isinstance(getattr(self, name), EconomicWeightRowRevealV3):
                raise ProofV3Error(f"coupling {name} has an unexpected type")
        object.__setattr__(self, "qkv_kv_weight_rows", weight_rows)
        bias_rows = tuple(self.bias_rows)
        if len(bias_rows) > 16:
            raise ProofV3Error("coupling bias row count is out of range")
        previous_index = -1
        for pair in bias_rows:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ProofV3Error("coupling bias entry has an unexpected shape")
            index, row = pair
            _u_range(index, "coupling bias projection index", bits=8)
            if index <= previous_index:
                raise ProofV3Error(
                    "coupling bias entries must be strictly increasing"
                )
            previous_index = index
            if not isinstance(row, EconomicWeightRowRevealV3):
                raise ProofV3Error("coupling bias row has an unexpected type")
        object.__setattr__(self, "bias_rows", bias_rows)

    def encode(self, writer: _Writer) -> None:
        writer.pack(
            "<IHHHHHH",
            self.layer_index,
            self.attn_o_y_oracle_index,
            self.down_y_oracle_index,
            self.mid_oracle_index,
            self.gate_up_y_oracle_index,
            self.k_oracle_index,
            self.v_oracle_index,
        )
        self.attn_o_y_opening.encode(writer)
        self.down_y_opening.encode(writer)
        self.mid_opening.encode(writer)
        self.gate_up_y_opening.encode(writer)
        self.k_opening.encode(writer)
        self.v_opening.encode(writer)
        self.qkv_s_kv_opening.encode(writer)
        writer.pack("<H", len(self.qkv_kv_weight_rows))
        for row in self.qkv_kv_weight_rows:
            row.encode(writer)
        self.input_norm_row.encode(writer)
        self.post_norm_row.encode(writer)
        writer.pack("<B", len(self.bias_rows))
        for index, row in self.bias_rows:
            writer.pack("<B", index)
            row.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicLayerCouplingRevealV3":
        (
            layer_index,
            attn_o_y_oracle_index,
            down_y_oracle_index,
            mid_oracle_index,
            gate_up_y_oracle_index,
            k_oracle_index,
            v_oracle_index,
        ) = reader.unpack("<IHHHHHH")
        attn_o_y_opening = EconomicMerkleOpeningV3.decode(reader)
        down_y_opening = EconomicMerkleOpeningV3.decode(reader)
        mid_opening = EconomicMerkleOpeningV3.decode(reader)
        gate_up_y_opening = EconomicMerkleOpeningV3.decode(reader)
        k_opening = EconomicMerkleOpeningV3.decode(reader)
        v_opening = EconomicMerkleOpeningV3.decode(reader)
        qkv_s_kv_opening = EconomicMerkleOpeningV3.decode(reader)
        row_count = reader.unpack("<H")[0]
        if row_count > MAX_WEIGHT_CHUNKS_PER_ROW:
            raise ProofV3Error("coupling kv weight row count is out of range")
        weight_rows = tuple(
            EconomicWeightRowRevealV3.decode(reader) for _ in range(row_count)
        )
        input_norm_row = EconomicWeightRowRevealV3.decode(reader)
        post_norm_row = EconomicWeightRowRevealV3.decode(reader)
        bias_count = reader.unpack("<B")[0]
        if bias_count > 16:
            raise ProofV3Error("coupling bias row count is out of range")
        bias_rows = tuple(
            (
                reader.unpack("<B")[0],
                EconomicWeightRowRevealV3.decode(reader),
            )
            for _ in range(bias_count)
        )
        return cls(
            layer_index=layer_index,
            attn_o_y_oracle_index=attn_o_y_oracle_index,
            attn_o_y_opening=attn_o_y_opening,
            down_y_oracle_index=down_y_oracle_index,
            down_y_opening=down_y_opening,
            mid_oracle_index=mid_oracle_index,
            mid_opening=mid_opening,
            gate_up_y_oracle_index=gate_up_y_oracle_index,
            gate_up_y_opening=gate_up_y_opening,
            k_oracle_index=k_oracle_index,
            k_opening=k_opening,
            v_oracle_index=v_oracle_index,
            v_opening=v_opening,
            qkv_s_kv_opening=qkv_s_kv_opening,
            qkv_kv_weight_rows=weight_rows,
            input_norm_row=input_norm_row,
            post_norm_row=post_norm_row,
            bias_rows=bias_rows,
        )


@dataclass(frozen=True, slots=True)
class EconomicGdnLayerCouplingRevealV3:
    """Per-selected-GDN-layer projection and residual coupling reveal.

    The recurrent-state transition itself is verified from authenticated raw
    execution-anchor rows.  This section binds those runtime rows to the
    registered qkvz/BA/output projection weights and to the common
    RMSNorm/MLP/residual chain.
    """

    layer_index: int
    qkvz_y_oracle_index: int
    qkvz_y_opening: EconomicMerkleOpeningV3
    ba_y_oracle_index: int
    ba_y_opening: EconomicMerkleOpeningV3
    gdn_o_y_oracle_index: int
    gdn_o_y_opening: EconomicMerkleOpeningV3
    down_y_oracle_index: int
    down_y_opening: EconomicMerkleOpeningV3
    mid_oracle_index: int
    mid_opening: EconomicMerkleOpeningV3
    gate_up_y_oracle_index: int
    gate_up_y_opening: EconomicMerkleOpeningV3
    input_norm_row: EconomicWeightRowRevealV3
    post_norm_row: EconomicWeightRowRevealV3
    bias_rows: tuple[tuple[int, EconomicWeightRowRevealV3], ...] = ()
    runtime_rows: tuple[
        tuple[int, bytes, bytes, bytes], ...
    ] = ()
    norm_source_rows: tuple[tuple[int, bytes, bytes], ...] = ()

    def __post_init__(self) -> None:
        _u_range(self.layer_index, "GDN coupling layer_index", bits=32)
        for name in (
            "qkvz_y_oracle_index",
            "ba_y_oracle_index",
            "gdn_o_y_oracle_index",
            "down_y_oracle_index",
            "mid_oracle_index",
            "gate_up_y_oracle_index",
        ):
            _u_range(getattr(self, name), f"GDN coupling {name}", bits=16)
        for name in (
            "qkvz_y_opening",
            "ba_y_opening",
            "gdn_o_y_opening",
            "down_y_opening",
            "mid_opening",
            "gate_up_y_opening",
        ):
            if not isinstance(getattr(self, name), EconomicMerkleOpeningV3):
                raise ProofV3Error(
                    f"GDN coupling {name} has an unexpected type"
                )
        for name in ("input_norm_row", "post_norm_row"):
            if not isinstance(getattr(self, name), EconomicWeightRowRevealV3):
                raise ProofV3Error(
                    f"GDN coupling {name} has an unexpected type"
                )
        bias_rows = tuple(self.bias_rows)
        if len(bias_rows) > 16:
            raise ProofV3Error("GDN coupling bias row count is out of range")
        previous_index = -1
        for pair in bias_rows:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ProofV3Error(
                    "GDN coupling bias entry has an unexpected shape"
                )
            index, row = pair
            _u_range(index, "GDN coupling bias projection index", bits=8)
            if index <= previous_index:
                raise ProofV3Error(
                    "GDN coupling bias entries must be strictly increasing"
                )
            previous_index = index
            if not isinstance(row, EconomicWeightRowRevealV3):
                raise ProofV3Error(
                    "GDN coupling bias row has an unexpected type"
                )
        object.__setattr__(self, "bias_rows", bias_rows)
        runtime_rows = tuple(self.runtime_rows)
        if len(runtime_rows) > MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error("GDN runtime row count is out of range")
        previous_position = -1
        for record in runtime_rows:
            if not isinstance(record, tuple) or len(record) != 4:
                raise ProofV3Error("GDN runtime row is malformed")
            position, qkvz_bytes, ba_bytes, output_bytes = record
            _u_range(position, "GDN runtime row position", bits=64)
            if position <= previous_position:
                raise ProofV3Error(
                    "GDN runtime rows must be strictly increasing"
                )
            previous_position = position
            for value in (qkvz_bytes, ba_bytes, output_bytes):
                if (
                    not isinstance(value, bytes)
                    or not 0 < len(value) <= MAX_EXECUTION_ANCHOR_ROW_BYTES
                ):
                    raise ProofV3Error(
                        "GDN runtime row bytes are out of range"
                    )
        object.__setattr__(self, "runtime_rows", runtime_rows)
        norm_source_rows = tuple(self.norm_source_rows)
        if len(norm_source_rows) > MAX_EXECUTION_ANCHOR_ROWS:
            raise ProofV3Error("GDN norm-source row count is out of range")
        previous_position = -1
        for record in norm_source_rows:
            if not isinstance(record, tuple) or len(record) != 3:
                raise ProofV3Error("GDN norm-source row is malformed")
            position, input_bytes, post_bytes = record
            _u_range(position, "GDN norm-source row position", bits=64)
            if position <= previous_position:
                raise ProofV3Error(
                    "GDN norm-source rows must be strictly increasing"
                )
            previous_position = position
            for value in (input_bytes, post_bytes):
                if (
                    not isinstance(value, bytes)
                    or not 0 < len(value) <= MAX_EXECUTION_ANCHOR_ROW_BYTES
                ):
                    raise ProofV3Error(
                        "GDN norm-source row bytes are out of range"
                    )
        object.__setattr__(self, "norm_source_rows", norm_source_rows)

    def encode(self, writer: _Writer) -> None:
        writer.pack(
            "<IHHHHHH",
            self.layer_index,
            self.qkvz_y_oracle_index,
            self.ba_y_oracle_index,
            self.gdn_o_y_oracle_index,
            self.down_y_oracle_index,
            self.mid_oracle_index,
            self.gate_up_y_oracle_index,
        )
        self.qkvz_y_opening.encode(writer)
        self.ba_y_opening.encode(writer)
        self.gdn_o_y_opening.encode(writer)
        self.down_y_opening.encode(writer)
        self.mid_opening.encode(writer)
        self.gate_up_y_opening.encode(writer)
        self.input_norm_row.encode(writer)
        self.post_norm_row.encode(writer)
        writer.pack("<B", len(self.bias_rows))
        for index, row in self.bias_rows:
            writer.pack("<B", index)
            row.encode(writer)
        writer.pack("<I", len(self.runtime_rows))
        for position, qkvz_bytes, ba_bytes, output_bytes in self.runtime_rows:
            writer.pack("<Q", position)
            for value in (qkvz_bytes, ba_bytes, output_bytes):
                writer.vbytes(
                    value,
                    "GDN runtime row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                )
        writer.pack("<I", len(self.norm_source_rows))
        for position, input_bytes, post_bytes in self.norm_source_rows:
            writer.pack("<Q", position)
            for value in (input_bytes, post_bytes):
                writer.vbytes(
                    value,
                    "GDN norm-source row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                )

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicGdnLayerCouplingRevealV3":
        (
            layer_index,
            qkvz_y_oracle_index,
            ba_y_oracle_index,
            gdn_o_y_oracle_index,
            down_y_oracle_index,
            mid_oracle_index,
            gate_up_y_oracle_index,
        ) = reader.unpack("<IHHHHHH")
        qkvz_y_opening = EconomicMerkleOpeningV3.decode(reader)
        ba_y_opening = EconomicMerkleOpeningV3.decode(reader)
        gdn_o_y_opening = EconomicMerkleOpeningV3.decode(reader)
        down_y_opening = EconomicMerkleOpeningV3.decode(reader)
        mid_opening = EconomicMerkleOpeningV3.decode(reader)
        gate_up_y_opening = EconomicMerkleOpeningV3.decode(reader)
        input_norm_row = EconomicWeightRowRevealV3.decode(reader)
        post_norm_row = EconomicWeightRowRevealV3.decode(reader)
        bias_count = reader.unpack("<B")[0]
        if bias_count > 16:
            raise ProofV3Error("GDN coupling bias row count is out of range")
        bias_rows = tuple(
            (
                reader.unpack("<B")[0],
                EconomicWeightRowRevealV3.decode(reader),
            )
            for _ in range(bias_count)
        )
        runtime_count = reader.count(
            "GDN runtime rows",
            MAX_EXECUTION_ANCHOR_ROWS,
            allow_zero=True,
        )
        runtime_rows = tuple(
            (
                reader.unpack("<Q")[0],
                reader.vbytes(
                    "GDN qkvz runtime row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                ),
                reader.vbytes(
                    "GDN ba runtime row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                ),
                reader.vbytes(
                    "GDN output runtime row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                ),
            )
            for _ in range(runtime_count)
        )
        norm_source_count = reader.count(
            "GDN norm-source rows",
            MAX_EXECUTION_ANCHOR_ROWS,
            allow_zero=True,
        )
        norm_source_rows = tuple(
            (
                reader.unpack("<Q")[0],
                reader.vbytes(
                    "GDN input norm-source row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                ),
                reader.vbytes(
                    "GDN post norm-source row",
                    MAX_EXECUTION_ANCHOR_ROW_BYTES,
                ),
            )
            for _ in range(norm_source_count)
        )
        return cls(
            layer_index=layer_index,
            qkvz_y_oracle_index=qkvz_y_oracle_index,
            qkvz_y_opening=qkvz_y_opening,
            ba_y_oracle_index=ba_y_oracle_index,
            ba_y_opening=ba_y_opening,
            gdn_o_y_oracle_index=gdn_o_y_oracle_index,
            gdn_o_y_opening=gdn_o_y_opening,
            down_y_oracle_index=down_y_oracle_index,
            down_y_opening=down_y_opening,
            mid_oracle_index=mid_oracle_index,
            mid_opening=mid_opening,
            gate_up_y_oracle_index=gate_up_y_oracle_index,
            gate_up_y_opening=gate_up_y_opening,
            input_norm_row=input_norm_row,
            post_norm_row=post_norm_row,
            bias_rows=bias_rows,
            runtime_rows=runtime_rows,
            norm_source_rows=norm_source_rows,
        )


@dataclass(frozen=True, slots=True)
class EconomicBoundaryOpeningV3:
    """Residual in/out openings for one layer (chain connectivity)."""

    layer_index: int
    in_oracle_index: int
    out_oracle_index: int
    in_opening: EconomicMerkleOpeningV3
    out_opening: EconomicMerkleOpeningV3

    def __post_init__(self) -> None:
        _u_range(self.layer_index, "boundary layer_index", bits=32)
        _u_range(self.in_oracle_index, "boundary in_oracle_index", bits=16)
        _u_range(self.out_oracle_index, "boundary out_oracle_index", bits=16)
        if self.in_oracle_index == self.out_oracle_index:
            raise ProofV3Error("boundary in/out oracles must differ")
        for name, opening in (
            ("boundary in_opening", self.in_opening),
            ("boundary out_opening", self.out_opening),
        ):
            if not isinstance(opening, EconomicMerkleOpeningV3):
                raise ProofV3Error(f"{name} has an unexpected type")

    def encode(self, writer: _Writer) -> None:
        writer.pack(
            "<IHH", self.layer_index, self.in_oracle_index, self.out_oracle_index
        )
        self.in_opening.encode(writer)
        self.out_opening.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicBoundaryOpeningV3":
        layer_index, in_oracle_index, out_oracle_index = reader.unpack("<IHH")
        return cls(
            layer_index=layer_index,
            in_oracle_index=in_oracle_index,
            out_oracle_index=out_oracle_index,
            in_opening=EconomicMerkleOpeningV3.decode(reader),
            out_opening=EconomicMerkleOpeningV3.decode(reader),
        )


@dataclass(frozen=True, slots=True)
class EconomicChainRevealV3:
    """Bottom anchor + complete boundary connectivity at sampled rows."""

    residual0_oracle_index: int
    residual0_opening: EconomicMerkleOpeningV3
    embedding_rows: tuple[EconomicWeightRowRevealV3, ...]
    boundaries: tuple[EconomicBoundaryOpeningV3, ...]

    def __post_init__(self) -> None:
        _u_range(
            self.residual0_oracle_index, "chain residual0_oracle_index", bits=16
        )
        if not isinstance(self.residual0_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error("chain residual0_opening has an unexpected type")
        embedding_rows = tuple(self.embedding_rows)
        if not embedding_rows or len(embedding_rows) > MAX_REVEALED_ROWS:
            raise ProofV3Error("chain embedding row count is out of range")
        for reveal in embedding_rows:
            if not isinstance(reveal, EconomicWeightRowRevealV3):
                raise ProofV3Error("chain embedding row has an unexpected type")
        boundaries = tuple(self.boundaries)
        if not boundaries or len(boundaries) > MAX_BOUNDARY_OPENINGS:
            raise ProofV3Error("chain boundary count is out of range")
        previous = -1
        for boundary in boundaries:
            if not isinstance(boundary, EconomicBoundaryOpeningV3):
                raise ProofV3Error("chain boundary has an unexpected type")
            if boundary.layer_index <= previous:
                raise ProofV3Error("chain boundaries must be strictly increasing")
            previous = boundary.layer_index
        object.__setattr__(self, "embedding_rows", embedding_rows)
        object.__setattr__(self, "boundaries", boundaries)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<H", self.residual0_oracle_index)
        self.residual0_opening.encode(writer)
        writer.pack("<I", len(self.embedding_rows))
        for reveal in self.embedding_rows:
            reveal.encode(writer)
        writer.pack("<I", len(self.boundaries))
        for boundary in self.boundaries:
            boundary.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicChainRevealV3":
        residual0_oracle_index = reader.unpack("<H")[0]
        residual0_opening = EconomicMerkleOpeningV3.decode(reader)
        embedding_count = reader.count("chain embedding rows", MAX_REVEALED_ROWS)
        embedding_rows = tuple(
            EconomicWeightRowRevealV3.decode(reader) for _ in range(embedding_count)
        )
        boundary_count = reader.count("chain boundaries", MAX_BOUNDARY_OPENINGS)
        boundaries = tuple(
            EconomicBoundaryOpeningV3.decode(reader) for _ in range(boundary_count)
        )
        return cls(
            residual0_oracle_index=residual0_oracle_index,
            residual0_opening=residual0_opening,
            embedding_rows=embedding_rows,
            boundaries=boundaries,
        )


@dataclass(frozen=True, slots=True)
class EconomicFinalRevealV3:
    """Final-hidden -> LM-head -> observed-token top-anchor reveal.

    Legacy profiles use ``logits_openings`` into pre-nonce block oracles.
    Compact-v9 profiles instead carry a bounded miner-claimed top-k row set;
    normal proofs certify it against nonce-sampled rows, while escalations
    additionally carry one complete ``revealed_logits`` row.  The verifier
    derives which form is mandatory from the signed profile and nonce.
    """

    final_oracle_index: int
    audited_position: int
    final_opening: EconomicMerkleOpeningV3
    lm_head_rows: tuple[EconomicWeightRowRevealV3, ...]
    logits_openings: tuple[tuple[int, EconomicMerkleOpeningV3], ...]
    candidate_token_rows: tuple[int, ...] = ()
    revealed_logits: tuple[int, ...] = ()
    lm_head_catalog_folds: tuple[bytes, ...] = ()
    # FINAL NORM LINK: the last layer's residual_out row that produced the
    # audited final hidden row, plus the signed final-norm gain vector --
    # binds the top anchor to the audited residual chain.
    last_residual_oracle_index: int = 0
    last_residual_opening: EconomicMerkleOpeningV3 | None = None
    final_norm_row: EconomicWeightRowRevealV3 | None = None

    def __post_init__(self) -> None:
        _u_range(self.final_oracle_index, "final final_oracle_index", bits=16)
        _u_range(self.audited_position, "final audited_position", bits=32)
        _u_range(
            self.last_residual_oracle_index,
            "final last_residual_oracle_index",
            bits=16,
        )
        if self.last_residual_opening is None or self.final_norm_row is None:
            raise ProofV3Error("final reveal requires the final-norm link")
        if not isinstance(self.last_residual_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error(
                "final last_residual_opening has an unexpected type"
            )
        if not isinstance(self.final_norm_row, EconomicWeightRowRevealV3):
            raise ProofV3Error("final final_norm_row has an unexpected type")
        if not isinstance(self.final_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error("final final_opening has an unexpected type")
        lm_head_rows = tuple(self.lm_head_rows)
        if not lm_head_rows or len(lm_head_rows) > MAX_REVEALED_ROWS:
            raise ProofV3Error("final lm_head row count is out of range")
        previous = -1
        for reveal in lm_head_rows:
            if not isinstance(reveal, EconomicWeightRowRevealV3):
                raise ProofV3Error("final lm_head row has an unexpected type")
            if reveal.row_index <= previous:
                raise ProofV3Error("final lm_head rows must be strictly increasing")
            previous = reveal.row_index
        logits_openings = tuple(self.logits_openings)
        if len(logits_openings) > MAX_REVEALED_ROWS:
            raise ProofV3Error("final logits opening count is out of range")
        previous = -1
        for entry in logits_openings:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ProofV3Error("final logits opening entry is malformed")
            oracle_index, opening = entry
            _u_range(oracle_index, "final logits oracle_index", bits=16)
            if oracle_index <= previous:
                raise ProofV3Error(
                    "final logits openings must be strictly increasing"
                )
            previous = oracle_index
            if oracle_index == self.final_oracle_index:
                raise ProofV3Error("final hidden and logits oracles must differ")
            if not isinstance(opening, EconomicMerkleOpeningV3):
                raise ProofV3Error("final logits opening has an unexpected type")
        object.__setattr__(self, "lm_head_rows", lm_head_rows)
        object.__setattr__(self, "logits_openings", logits_openings)
        candidates = tuple(self.candidate_token_rows)
        if candidates:
            candidates = _sorted_distinct(
                candidates,
                "final candidate token rows",
                bits=32,
                maximum=32,
            )
        object.__setattr__(self, "candidate_token_rows", candidates)
        revealed_logits = tuple(self.revealed_logits)
        if len(revealed_logits) > MAX_REVEALED_LOGITS_V3:
            raise ProofV3Error("final revealed logits exceed the wire bound")
        for value in revealed_logits:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -(1 << 63) <= value < 1 << 63
            ):
                raise ProofV3Error(
                    "final revealed logit is outside signed 64-bit range"
                )
        if logits_openings and revealed_logits:
            raise ProofV3Error(
                "final legacy openings and revealed logits are mutually exclusive"
            )
        object.__setattr__(self, "revealed_logits", revealed_logits)
        catalog_folds = tuple(self.lm_head_catalog_folds)
        if len(catalog_folds) not in (0, LM_HEAD_CATALOG_FOLD_COUNT_V3):
            raise ProofV3Error("final LM-head catalog fold count is invalid")
        if any(
            not isinstance(value, bytes)
            or not 0 < len(value) <= MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3
            for value in catalog_folds
        ):
            raise ProofV3Error("final LM-head catalog fold is malformed")
        object.__setattr__(self, "lm_head_catalog_folds", catalog_folds)

    def encode(self, writer: _Writer) -> None:
        writer.pack(
            "<HIH",
            self.final_oracle_index,
            self.audited_position,
            self.last_residual_oracle_index,
        )
        self.final_opening.encode(writer)
        self.last_residual_opening.encode(writer)
        self.final_norm_row.encode(writer)
        writer.pack("<I", len(self.lm_head_rows))
        for reveal in self.lm_head_rows:
            reveal.encode(writer)
        writer.pack("<I", len(self.candidate_token_rows))
        for row in self.candidate_token_rows:
            writer.pack("<I", row)
        writer.pack("<I", len(self.logits_openings))
        for oracle_index, opening in self.logits_openings:
            writer.pack("<H", oracle_index)
            opening.encode(writer)
        writer.pack("<I", len(self.revealed_logits))
        for start in range(0, len(self.revealed_logits), 8_192):
            chunk = self.revealed_logits[start : start + 8_192]
            writer.pack(f"<{len(chunk)}q", *chunk)
        writer.pack("<I", len(self.lm_head_catalog_folds))
        for folded_weights in self.lm_head_catalog_folds:
            writer.vbytes(
                folded_weights,
                "final LM-head catalog fold",
                MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3,
            )

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicFinalRevealV3":
        (
            final_oracle_index,
            audited_position,
            last_residual_oracle_index,
        ) = reader.unpack("<HIH")
        final_opening = EconomicMerkleOpeningV3.decode(reader)
        last_residual_opening = EconomicMerkleOpeningV3.decode(reader)
        final_norm_row = EconomicWeightRowRevealV3.decode(reader)
        row_count = reader.count("final lm_head rows", MAX_REVEALED_ROWS)
        lm_head_rows = tuple(
            EconomicWeightRowRevealV3.decode(reader) for _ in range(row_count)
        )
        candidate_count = reader.count(
            "final candidate token rows",
            32,
            allow_zero=True,
        )
        candidate_token_rows = tuple(
            reader.unpack("<I")[0] for _ in range(candidate_count)
        )
        opening_count = reader.count(
            "final logits openings",
            MAX_REVEALED_ROWS,
            allow_zero=True,
        )
        logits_openings = []
        for _ in range(opening_count):
            oracle_index = reader.unpack("<H")[0]
            logits_openings.append(
                (oracle_index, EconomicMerkleOpeningV3.decode(reader))
            )
        logits_count = reader.count(
            "final revealed logits",
            MAX_REVEALED_LOGITS_V3,
            allow_zero=True,
        )
        revealed_logits = []
        remaining = logits_count
        while remaining:
            chunk_count = min(remaining, 8_192)
            revealed_logits.extend(
                reader.unpack(f"<{chunk_count}q")
            )
            remaining -= chunk_count
        fold_count = reader.count(
            "final LM-head catalog folds",
            LM_HEAD_CATALOG_FOLD_COUNT_V3,
            allow_zero=True,
        )
        catalog_folds = tuple(
            reader.vbytes(
                "final LM-head catalog fold",
                MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3,
            )
            for _ in range(fold_count)
        )
        return cls(
            final_oracle_index=final_oracle_index,
            audited_position=audited_position,
            final_opening=final_opening,
            lm_head_rows=lm_head_rows,
            logits_openings=tuple(logits_openings),
            candidate_token_rows=candidate_token_rows,
            revealed_logits=tuple(revealed_logits),
            lm_head_catalog_folds=catalog_folds,
            last_residual_oracle_index=last_residual_oracle_index,
            last_residual_opening=last_residual_opening,
            final_norm_row=final_norm_row,
        )


# ---------------------------------------------------------------------------
# the complete wire proof
# ---------------------------------------------------------------------------


MAX_ATTENTION_BUNDLE_BYTES = 64 << 20
MAX_ATTENTION_POOL_ROWS = 4096
MAX_ATTENTION_LAYERS = 128
MAX_ATTENTION_QUERY_ROWS = 8192
MAX_ATTENTION_QUERY_ROW_BYTES = 1 << 20
MAX_ATTENTION_QUERY_ROWS_BYTES = 16 << 20


@dataclass(frozen=True, slots=True)
class EconomicAttentionRequestSectionV3:
    """SCORED_SCHEME_RATIONAL_V2 attention audit riding the economic
    proof: the capture-kv rational bundle wire plus its transport
    inputs, in ONE bounded canonical request section.

    ``base_capture_digest`` is the request's capture chain BEFORE the
    transport fold; the verifier recomputes
    ``fold(base, commitment(roots, binding, pool, key_count))`` and
    requires equality with the proof's ``capture_chain_digest`` -- the
    envelope's execution root then authenticates everything here.
    ``pool`` is the candidate row pool in POOL ORDER (order is
    protocol-significant); ``roots_by_layer`` is sorted by layer with
    (k, v, ox[, gate]) roots each. ``economic_ox_openings`` authenticates
    the same nonce-selected rows against the economic transition oracle,
    joining the attention proof to the registered-weight/residual chain."""

    base_capture_digest: bytes
    binding: bytes
    key_count: int
    pool: tuple[int, ...]
    roots_by_layer: tuple[tuple[int, tuple[bytes, ...]], ...]
    economic_ox_openings: tuple[
        tuple[int, int, EconomicMerkleOpeningV3], ...
    ]
    bundle_wire: bytes
    query_rows: tuple[tuple[int, int, bytes], ...] = ()

    def __post_init__(self) -> None:
        _fixed32(self.base_capture_digest, "attention base_capture_digest")
        _fixed32(self.binding, "attention transport binding")
        _u_range(self.key_count, "attention key_count", bits=64, positive=True)
        pool = tuple(self.pool)
        if not 0 < len(pool) <= MAX_ATTENTION_POOL_ROWS:
            raise ProofV3Error("attention pool size is out of range")
        seen: set[int] = set()
        for position in pool:
            _u_range(position, "attention pool position", bits=64)
            if position in seen:
                raise ProofV3Error("attention pool positions must be distinct")
            seen.add(position)
        layers = tuple(self.roots_by_layer)
        if not 0 < len(layers) <= MAX_ATTENTION_LAYERS:
            raise ProofV3Error("attention layer count is out of range")
        previous = -1
        for layer, roots in layers:
            _u_range(layer, "attention audited layer", bits=32)
            if layer <= previous:
                raise ProofV3Error(
                    "attention layers must be strictly increasing")
            previous = layer
            roots = tuple(roots)
            if not 3 <= len(roots) <= 4:
                raise ProofV3Error(
                    "attention layer needs (k, v, ox[, gate]) roots")
            for root in roots:
                _fixed32(root, "attention capture root")
        economic_openings = tuple(self.economic_ox_openings)
        if len(economic_openings) != len(layers):
            raise ProofV3Error(
                "attention economic o_x openings must cover every layer")
        expected_layers = tuple(layer for layer, _roots in layers)
        previous = -1
        for layer, oracle_index, opening in economic_openings:
            _u_range(layer, "attention economic o_x layer", bits=32)
            if layer <= previous:
                raise ProofV3Error(
                    "attention economic o_x layers must be strictly increasing")
            previous = layer
            _u_range(
                oracle_index, "attention economic o_x oracle index", bits=16)
            if not isinstance(opening, EconomicMerkleOpeningV3):
                raise ProofV3Error(
                    "attention economic o_x opening has an unexpected type")
        if tuple(
            layer for layer, _index, _opening in economic_openings
        ) != expected_layers:
            raise ProofV3Error(
                "attention economic o_x layers disagree with capture roots")
        wire = self.bundle_wire
        if (
            not isinstance(wire, (bytes, bytearray))
            or not 0 < len(wire) <= MAX_ATTENTION_BUNDLE_BYTES
        ):
            raise ProofV3Error("attention bundle wire is out of range")
        query_rows = tuple(self.query_rows)
        if len(query_rows) > MAX_ATTENTION_QUERY_ROWS:
            raise ProofV3Error("attention query row count is out of range")
        previous = (-1, -1)
        total_query_bytes = 0
        for layer, position, row_bytes in query_rows:
            _u_range(layer, "attention query row layer", bits=32)
            _u_range(position, "attention query row position", bits=64)
            if (layer, position) <= previous:
                raise ProofV3Error(
                    "attention query rows must be strictly ordered"
                )
            previous = (layer, position)
            if (
                not isinstance(row_bytes, bytes)
                or not 0 < len(row_bytes) <= MAX_ATTENTION_QUERY_ROW_BYTES
            ):
                raise ProofV3Error("attention query row width is out of range")
            total_query_bytes += len(row_bytes)
        if total_query_bytes > MAX_ATTENTION_QUERY_ROWS_BYTES:
            raise ProofV3Error("attention query rows exceed the wire bound")
        object.__setattr__(self, "pool", pool)
        object.__setattr__(
            self,
            "roots_by_layer",
            tuple((layer, tuple(roots)) for layer, roots in layers),
        )
        object.__setattr__(
            self, "economic_ox_openings", economic_openings)
        object.__setattr__(self, "bundle_wire", bytes(wire))
        object.__setattr__(self, "query_rows", query_rows)

    def encode(self, writer: _Writer) -> None:
        writer.raw(self.base_capture_digest)
        writer.raw(self.binding)
        writer.pack("<QII", self.key_count, len(self.pool),
                    len(self.roots_by_layer))
        writer.raw(struct.pack(f"<{len(self.pool)}Q", *self.pool))
        for layer, roots in self.roots_by_layer:
            writer.pack("<IB", layer, len(roots))
            for root in roots:
                writer.raw(root)
        writer.pack("<I", len(self.economic_ox_openings))
        for layer, oracle_index, opening in self.economic_ox_openings:
            writer.pack("<IH", layer, oracle_index)
            opening.encode(writer)
        writer.pack("<I", len(self.query_rows))
        for layer, position, row_bytes in self.query_rows:
            writer.pack("<IQ", layer, position)
            writer.vbytes(
                row_bytes,
                "attention query row",
                MAX_ATTENTION_QUERY_ROW_BYTES,
            )
        writer.vbytes(self.bundle_wire, "attention bundle wire",
                      MAX_ATTENTION_BUNDLE_BYTES)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicAttentionRequestSectionV3":
        base = reader.read(32)
        binding = reader.read(32)
        key_count, pool_count, layer_count = reader.unpack("<QII")
        if not 0 < pool_count <= MAX_ATTENTION_POOL_ROWS:
            raise ProofV3Error("attention pool size is out of range")
        if not 0 < layer_count <= MAX_ATTENTION_LAYERS:
            raise ProofV3Error("attention layer count is out of range")
        pool = struct.unpack(f"<{pool_count}Q", reader.read(8 * pool_count))
        layers = []
        for _ in range(layer_count):
            layer, root_count = reader.unpack("<IB")
            if not 3 <= root_count <= 4:
                raise ProofV3Error(
                    "attention layer needs (k, v, ox[, gate]) roots")
            layers.append(
                (layer,
                 tuple(reader.read(32) for _ in range(root_count)))
            )
        opening_count = reader.count(
            "attention economic o_x openings", MAX_ATTENTION_LAYERS)
        economic_openings = tuple(
            (
                reader.unpack("<I")[0],
                reader.unpack("<H")[0],
                EconomicMerkleOpeningV3.decode(reader),
            )
            for _ in range(opening_count)
        )
        query_row_count = reader.count(
            "attention query rows",
            MAX_ATTENTION_QUERY_ROWS,
            allow_zero=True,
        )
        query_rows = tuple(
            (
                *reader.unpack("<IQ"),
                reader.vbytes(
                    "attention query row",
                    MAX_ATTENTION_QUERY_ROW_BYTES,
                ),
            )
            for _ in range(query_row_count)
        )
        wire = reader.vbytes(
            "attention bundle wire", MAX_ATTENTION_BUNDLE_BYTES)
        return cls(
            base_capture_digest=base, binding=binding,
            key_count=key_count, pool=pool,
            roots_by_layer=tuple(layers),
            economic_ox_openings=economic_openings,
            bundle_wire=wire,
            query_rows=query_rows)


def _encode_prefix_cache_path_v3(writer: _Writer, path: MerklePath) -> None:
    if (
        not isinstance(path, MerklePath)
        or isinstance(path.leaf_index, bool)
        or not isinstance(path.leaf_index, int)
        or not 0 <= path.leaf_index < 1 << 32
        or len(path.siblings) > MAX_EXECUTION_ANCHOR_SIBLINGS
    ):
        raise ProofV3Error("prefix-cache Merkle path is malformed")
    writer.pack("<IB", path.leaf_index, len(path.siblings))
    for level, (sibling, is_left) in enumerate(path.siblings):
        _fixed32(sibling, "prefix-cache Merkle sibling")
        if is_left != bool((path.leaf_index >> level) & 1):
            raise ProofV3Error(
                "prefix-cache Merkle path direction is noncanonical"
            )
        writer.raw(sibling)


def _decode_prefix_cache_path_v3(reader: _Reader) -> MerklePath:
    leaf_index, count = reader.unpack("<IB")
    if count > MAX_EXECUTION_ANCHOR_SIBLINGS:
        raise ProofV3Error("prefix-cache Merkle path exceeds its depth bound")
    return MerklePath(
        leaf_index=leaf_index,
        siblings=[
            (reader.read(32), bool((leaf_index >> level) & 1))
            for level in range(count)
        ],
    )


def _encode_prefix_cache_section_v3(
    writer: _Writer,
    section: PrefixCachePostnonceProofV3,
) -> None:
    commitment = section.commitment
    writer.raw(section.base_capture_digest)
    writer.raw(commitment.execution_profile_digest)
    writer.raw(commitment.prompt_token_root)
    writer.raw(commitment.cache_salt_digest)
    writer.raw(commitment.executed_suffix_digest)
    writer.raw(commitment.block_inventory_root)
    writer.raw(
        bytes(32)
        if commitment.gdn_boundary_root is None
        else commitment.gdn_boundary_root
    )
    writer.pack(
        "<IIII",
        commitment.context_token_count,
        commitment.cached_token_count,
        commitment.block_token_count,
        commitment.block_count,
    )
    writer.pack("<I", len(section.block_opening.records))
    for record, path in zip(
        section.block_opening.records,
        section.block_opening.paths,
        strict=True,
    ):
        writer.pack(
            "<III",
            record.block_index,
            record.token_start,
            record.token_count,
        )
        writer.raw(record.content_digest)
        writer.raw(record.state_root)
        _encode_prefix_cache_path_v3(writer, path)
    writer.pack("<I", len(section.state_openings))
    for opening in section.state_openings:
        writer.pack("<I", len(opening.records))
        for record, path in zip(opening.records, opening.paths, strict=True):
            writer.pack("<I", record.block_index)
            writer.identifier(record.stage_id, "prefix-cache state stage_id")
            writer.pack("<II", record.row_count, record.row_width)
            writer.raw(record.value_root)
            _encode_prefix_cache_path_v3(writer, path)
    writer.pack("<I", len(section.lane_reveals))
    for reveal in section.lane_reveals:
        writer.pack("<I", reveal.block_index)
        writer.identifier(reveal.stage_id, "prefix-cache lane stage_id")
        opening = reveal.opening
        writer.pack(
            "<IIH",
            opening.row_index,
            opening.lane_index,
            len(opening.lane_bytes),
        )
        writer.raw(opening.lane_bytes)
        writer.pack("<B", len(opening.lane_sibling_hashes))
        for sibling in opening.lane_sibling_hashes:
            writer.raw(sibling)
        writer.pack("<B", len(opening.row_sibling_hashes))
        for sibling in opening.row_sibling_hashes:
            writer.raw(sibling)


def _decode_prefix_cache_section_v3(
    reader: _Reader,
) -> PrefixCachePostnonceProofV3:
    base_capture_digest = reader.read(32)
    profile_digest = reader.read(32)
    prompt_root = reader.read(32)
    salt_digest = reader.read(32)
    suffix_digest = reader.read(32)
    inventory_root = reader.read(32)
    raw_gdn_root = reader.read(32)
    context, cached, width, block_count = reader.unpack("<IIII")
    commitment = PrefixCacheCommitmentV3(
        execution_profile_digest=profile_digest,
        prompt_token_root=prompt_root,
        cache_salt_digest=salt_digest,
        executed_suffix_digest=suffix_digest,
        block_inventory_root=inventory_root,
        gdn_boundary_root=(None if raw_gdn_root == bytes(32) else raw_gdn_root),
        context_token_count=context,
        cached_token_count=cached,
        block_token_count=width,
        block_count=block_count,
    )
    opened_blocks = reader.count(
        "prefix-cache opened blocks",
        MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3,
    )
    block_records = []
    block_paths = []
    for _ in range(opened_blocks):
        block_index, token_start, token_count = reader.unpack("<III")
        block_records.append(PrefixCacheBlockRecordV3(
            block_index=block_index,
            token_start=token_start,
            token_count=token_count,
            content_digest=reader.read(32),
            state_root=reader.read(32),
        ))
        block_paths.append(_decode_prefix_cache_path_v3(reader))
    state_opening_count = reader.count(
        "prefix-cache state openings",
        MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3,
    )
    state_openings = []
    for _ in range(state_opening_count):
        state_count = reader.count(
            "prefix-cache opened states",
            MAX_PREFIX_CACHE_STATE_STAGES_V3,
        )
        records = []
        paths = []
        for _ in range(state_count):
            block_index = reader.unpack("<I")[0]
            stage_id = reader.identifier("prefix-cache state stage_id")
            row_count, row_width = reader.unpack("<II")
            records.append(PrefixCacheStateRecordV3(
                block_index=block_index,
                stage_id=stage_id,
                row_count=row_count,
                row_width=row_width,
                value_root=reader.read(32),
            ))
            paths.append(_decode_prefix_cache_path_v3(reader))
        state_openings.append(PrefixCacheStateOpeningV3(
            records=tuple(records),
            paths=tuple(paths),
        ))
    lane_count = reader.count(
        "prefix-cache lane reveals",
        MAX_EXECUTION_ANCHOR_LANE_REVEALS,
    )
    lanes = []
    for _ in range(lane_count):
        block_index = reader.unpack("<I")[0]
        stage_id = reader.identifier("prefix-cache lane stage_id")
        row_index, lane_index, lane_byte_count = reader.unpack("<IIH")
        if lane_byte_count not in (256, 2048):
            raise ProofV3Error("prefix-cache lane byte count is unsupported")
        lane_bytes = reader.read(lane_byte_count)
        lane_sibling_count = reader.unpack("<B")[0]
        if lane_sibling_count > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error("prefix-cache lane path exceeds its depth bound")
        lane_siblings = tuple(
            reader.read(32) for _ in range(lane_sibling_count)
        )
        row_sibling_count = reader.unpack("<B")[0]
        if row_sibling_count > MAX_EXECUTION_ANCHOR_SIBLINGS:
            raise ProofV3Error("prefix-cache row path exceeds its depth bound")
        lanes.append(PrefixCacheLaneRevealV3(
            block_index=block_index,
            stage_id=stage_id,
            opening=ExecutionAnchorLaneOpeningV3(
                row_index=row_index,
                lane_index=lane_index,
                lane_bytes=lane_bytes,
                lane_sibling_hashes=lane_siblings,
                row_sibling_hashes=tuple(
                    reader.read(32) for _ in range(row_sibling_count)
                ),
            ),
        ))
    return PrefixCachePostnonceProofV3(
        commitment=commitment,
        base_capture_digest=base_capture_digest,
        block_opening=PrefixCacheBlockOpeningV3(
            records=tuple(block_records),
            paths=tuple(block_paths),
        ),
        state_openings=tuple(state_openings),
        lane_reveals=tuple(lanes),
    )


@dataclass(frozen=True, slots=True)
class EconomicRecomputeProofV3:
    """The complete canonical economic recompute proof for one request."""

    commitment_envelope_digest: bytes
    execution_profile_digest: bytes
    signed_bound_digest: bytes
    capture_chain_digest: bytes
    execution_anchors: tuple[ExecutionAnchorCommitmentV3, ...] = field(
        default_factory=tuple
    )
    execution_anchor_reveals: tuple[
        EconomicExecutionAnchorRevealV3, ...
    ] = field(default_factory=tuple)
    execution_anchor_lane_reveals: tuple[
        EconomicExecutionAnchorLaneRevealV3, ...
    ] = field(default_factory=tuple)
    oracles: tuple[EconomicOracleCommitmentV3, ...] = field(default_factory=tuple)
    projections: tuple[EconomicProjectionRevealV3, ...] = field(
        default_factory=tuple
    )
    couplings: tuple[EconomicLayerCouplingRevealV3, ...] = field(
        default_factory=tuple
    )
    gdn_couplings: tuple[EconomicGdnLayerCouplingRevealV3, ...] = field(
        default_factory=tuple
    )
    mlp_activation_reveals: tuple[
        EconomicMlpActivationRevealV3, ...
    ] = field(default_factory=tuple)
    lean_projection_batch_wire: bytes = b""
    succinct_projection_batch_wire: bytes = b""
    selected_trace_wire: bytes = b""
    moe_wire: bytes = b""
    chain: EconomicChainRevealV3 | None = None
    final: EconomicFinalRevealV3 | None = None
    attention: EconomicAttentionRequestSectionV3 | None = None
    prefix_cache: PrefixCachePostnonceProofV3 | None = None

    def __post_init__(self) -> None:
        _fixed32(self.commitment_envelope_digest, "commitment_envelope_digest")
        _fixed32(self.execution_profile_digest, "execution_profile_digest")
        _fixed32(self.signed_bound_digest, "signed_bound_digest")
        if (
            not isinstance(self.capture_chain_digest, bytes)
            or not self.capture_chain_digest
            or len(self.capture_chain_digest) > MAX_CAPTURE_CHAIN_DIGEST_BYTES
        ):
            raise ProofV3Error("capture_chain_digest length is out of range")
        execution_anchors = tuple(self.execution_anchors)
        if len(execution_anchors) > MAX_EXECUTION_ANCHORS:
            raise ProofV3Error(
                "execution anchor inventory count exceeds the wire bound"
            )
        if execution_anchors:
            if any(not isinstance(item, ExecutionAnchorCommitmentV3)
                   for item in execution_anchors):
                raise ProofV3Error(
                    "execution anchor commitment has an unexpected type"
                )
            # Also enforces exact canonical order and distinct stage ids.
            execution_anchor_inventory_digest_v3(execution_anchors)
        anchor_reveals = tuple(self.execution_anchor_reveals)
        if len(anchor_reveals) > MAX_EXECUTION_ANCHOR_REVEALS:
            raise ProofV3Error(
                "execution anchor reveal count exceeds the wire bound"
            )
        previous_anchor_index = -1
        for reveal in anchor_reveals:
            if not isinstance(reveal, EconomicExecutionAnchorRevealV3):
                raise ProofV3Error(
                    "execution anchor reveal has an unexpected type"
                )
            if reveal.commitment_index <= previous_anchor_index:
                raise ProofV3Error(
                    "execution anchor reveals are not canonically ordered"
                )
            if reveal.commitment_index >= len(execution_anchors):
                raise ProofV3Error(
                    "execution anchor reveal references an unknown commitment"
                )
            previous_anchor_index = reveal.commitment_index
        anchor_lane_reveals = tuple(self.execution_anchor_lane_reveals)
        if len(anchor_lane_reveals) > MAX_EXECUTION_ANCHOR_LANE_REVEALS:
            raise ProofV3Error(
                "execution anchor lane reveal count exceeds the wire bound"
            )
        previous_lane_key = None
        for reveal in anchor_lane_reveals:
            if not isinstance(reveal, EconomicExecutionAnchorLaneRevealV3):
                raise ProofV3Error(
                    "execution anchor lane reveal has an unexpected type"
                )
            if reveal.commitment_index >= len(execution_anchors):
                raise ProofV3Error(
                    "execution anchor lane reveal references an unknown "
                    "commitment"
                )
            key = (
                reveal.commitment_index,
                reveal.opening.row_index,
                reveal.opening.lane_index,
            )
            if previous_lane_key is not None and key <= previous_lane_key:
                raise ProofV3Error(
                    "execution anchor lane reveals are not canonically ordered"
                )
            previous_lane_key = key
        oracles = tuple(self.oracles)
        if not oracles or len(oracles) > MAX_ORACLES:
            raise ProofV3Error("oracle inventory count is out of range")
        seen_ids: set[str] = set()
        previous_key = None
        for oracle in oracles:
            if not isinstance(oracle, EconomicOracleCommitmentV3):
                raise ProofV3Error("oracle inventory entry has an unexpected type")
            if oracle.oracle_id in seen_ids:
                raise ProofV3Error("oracle inventory contains a duplicate oracle_id")
            seen_ids.add(oracle.oracle_id)
            key = (
                _PHASE_CODES[oracle.phase],
                oracle.layer_index,
                oracle.operation,
                oracle.oracle_id,
            )
            if previous_key is not None and key <= previous_key:
                raise ProofV3Error("oracle inventory is not canonically ordered")
            previous_key = key
        projections = tuple(self.projections)
        if len(projections) > MAX_PROJECTION_REVEALS:
            raise ProofV3Error("projection reveal count exceeds the wire bound")
        previous_index = -1
        for reveal in projections:
            if not isinstance(reveal, EconomicProjectionRevealV3):
                raise ProofV3Error("projection reveal has an unexpected type")
            if reveal.x_oracle_index <= previous_index:
                raise ProofV3Error(
                    "projection reveals must be strictly increasing by X oracle"
                )
            previous_index = reveal.x_oracle_index
            for oracle_index in (reveal.x_oracle_index, reveal.s_oracle_index):
                if oracle_index >= len(oracles):
                    raise ProofV3Error(
                        "projection reveal references an unknown oracle"
                    )
        couplings = tuple(self.couplings)
        if len(couplings) > MAX_COUPLING_REVEALS:
            raise ProofV3Error("coupling reveal count exceeds the wire bound")
        previous_layer = -1
        for coupling in couplings:
            if not isinstance(coupling, EconomicLayerCouplingRevealV3):
                raise ProofV3Error("coupling reveal has an unexpected type")
            if coupling.layer_index <= previous_layer:
                raise ProofV3Error(
                    "coupling reveals must be strictly increasing by layer"
                )
            previous_layer = coupling.layer_index
            for oracle_index in (
                coupling.attn_o_y_oracle_index,
                coupling.down_y_oracle_index,
                coupling.mid_oracle_index,
                coupling.gate_up_y_oracle_index,
                coupling.k_oracle_index,
                coupling.v_oracle_index,
            ):
                if oracle_index >= len(oracles):
                    raise ProofV3Error(
                        "coupling reveal references an unknown oracle"
                    )
        gdn_couplings = tuple(self.gdn_couplings)
        if len(gdn_couplings) > MAX_COUPLING_REVEALS:
            raise ProofV3Error("GDN coupling reveal count exceeds the wire bound")
        previous_layer = -1
        for coupling in gdn_couplings:
            if not isinstance(coupling, EconomicGdnLayerCouplingRevealV3):
                raise ProofV3Error(
                    "GDN coupling reveal has an unexpected type"
                )
            if coupling.layer_index <= previous_layer:
                raise ProofV3Error(
                    "GDN coupling reveals must be strictly increasing by layer"
                )
            previous_layer = coupling.layer_index
            for oracle_index in (
                coupling.qkvz_y_oracle_index,
                coupling.ba_y_oracle_index,
                coupling.gdn_o_y_oracle_index,
                coupling.down_y_oracle_index,
                coupling.mid_oracle_index,
                coupling.gate_up_y_oracle_index,
            ):
                if oracle_index >= len(oracles):
                    raise ProofV3Error(
                        "GDN coupling reveal references an unknown oracle"
                    )
        mlp_activation_reveals = tuple(self.mlp_activation_reveals)
        if len(mlp_activation_reveals) > MAX_COUPLING_REVEALS:
            raise ProofV3Error(
                "MLP activation reveal count exceeds the wire bound"
            )
        previous_layer = -1
        for reveal in mlp_activation_reveals:
            if not isinstance(reveal, EconomicMlpActivationRevealV3):
                raise ProofV3Error(
                    "MLP activation reveal has an unexpected type"
                )
            if reveal.layer_index <= previous_layer:
                raise ProofV3Error(
                    "MLP activation reveals must be strictly increasing by layer"
                )
            previous_layer = reveal.layer_index
        lean_projection_batch_wire = self.lean_projection_batch_wire
        if not isinstance(lean_projection_batch_wire, bytes):
            raise ProofV3Error(
                "lean projection batch wire must be canonical bytes"
            )
        if (
            len(lean_projection_batch_wire)
            > MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3
        ):
            raise ProofV3Error(
                "lean projection batch wire exceeds the protocol bound"
            )
        if lean_projection_batch_wire:
            # Parsing here makes malformed nested proof bytes a construction
            # failure, never a later "not requested" verifier outcome.
            decode_lean_projection_batch_v3(lean_projection_batch_wire)
        succinct_projection_batch_wire = self.succinct_projection_batch_wire
        if not isinstance(succinct_projection_batch_wire, bytes):
            raise ProofV3Error(
                "succinct projection batch wire must be canonical bytes"
            )
        if (
            len(succinct_projection_batch_wire)
            > MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3
        ):
            raise ProofV3Error(
                "succinct projection batch wire exceeds the protocol bound"
            )
        if succinct_projection_batch_wire:
            from verallm.proof_v3.succinct_projection_batch import (
                decode_succinct_projection_batch_v3,
            )

            decode_succinct_projection_batch_v3(
                succinct_projection_batch_wire
            )
        if lean_projection_batch_wire and succinct_projection_batch_wire:
            raise ProofV3Error(
                "projection batch modes are mutually exclusive"
            )
        selected_trace_wire = self.selected_trace_wire
        if not isinstance(selected_trace_wire, bytes):
            raise ProofV3Error(
                "selected trace wire must be canonical bytes"
            )
        if len(selected_trace_wire) > MAX_SELECTED_TRACE_WIRE_BYTES_V3:
            raise ProofV3Error(
                "selected trace wire exceeds the protocol bound"
            )
        if selected_trace_wire:
            from verallm.proof_v3.goldilocks_selected_trace_wire import (
                decode_goldilocks_selected_trace_v3,
            )

            decode_goldilocks_selected_trace_v3(selected_trace_wire)
            if (
                anchor_reveals
                or anchor_lane_reveals
                or projections
                or couplings
                or gdn_couplings
                or lean_projection_batch_wire
                or succinct_projection_batch_wire
                or self.chain is not None
                or self.final is not None
                or self.attention is not None
            ):
                raise ProofV3Error(
                    "selected trace and legacy recompute sections are "
                    "mutually exclusive"
                )
        moe_wire = self.moe_wire
        if not isinstance(moe_wire, bytes):
            raise ProofV3Error("MoE wire must be canonical bytes")
        if moe_wire:
            from verallm.proof_v3.economic_moe_wire import (
                MAX_ECONOMIC_MOE_WIRE_BYTES_V3,
                decode_economic_moe_wire_v3,
            )

            if len(moe_wire) > MAX_ECONOMIC_MOE_WIRE_BYTES_V3:
                raise ProofV3Error("MoE wire exceeds the protocol bound")
            decode_economic_moe_wire_v3(moe_wire)
        if self.chain is not None:
            if not isinstance(self.chain, EconomicChainRevealV3):
                raise ProofV3Error("chain reveal has an unexpected type")
            if self.chain.residual0_oracle_index >= len(oracles):
                raise ProofV3Error("chain reveal references an unknown oracle")
            for boundary in self.chain.boundaries:
                if (
                    boundary.in_oracle_index >= len(oracles)
                    or boundary.out_oracle_index >= len(oracles)
                ):
                    raise ProofV3Error("chain boundary references an unknown oracle")
        if self.final is not None:
            if not isinstance(self.final, EconomicFinalRevealV3):
                raise ProofV3Error("final reveal has an unexpected type")
            if self.final.final_oracle_index >= len(oracles) or any(
                oracle_index >= len(oracles)
                for oracle_index, _opening in self.final.logits_openings
            ):
                raise ProofV3Error("final reveal references an unknown oracle")
        if self.attention is not None and not isinstance(
            self.attention, EconomicAttentionRequestSectionV3
        ):
            raise ProofV3Error("attention section has an unexpected type")
        if self.prefix_cache is not None and not isinstance(
            self.prefix_cache, PrefixCachePostnonceProofV3
        ):
            raise ProofV3Error("prefix-cache section has an unexpected type")
        object.__setattr__(self, "oracles", oracles)
        object.__setattr__(self, "execution_anchors", execution_anchors)
        object.__setattr__(
            self, "execution_anchor_reveals", anchor_reveals)
        object.__setattr__(
            self, "execution_anchor_lane_reveals", anchor_lane_reveals)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "gdn_couplings", gdn_couplings)
        object.__setattr__(
            self, "mlp_activation_reveals", mlp_activation_reveals
        )
        object.__setattr__(
            self,
            "lean_projection_batch_wire",
            lean_projection_batch_wire,
        )
        object.__setattr__(
            self,
            "succinct_projection_batch_wire",
            succinct_projection_batch_wire,
        )
        object.__setattr__(
            self,
            "selected_trace_wire",
            selected_trace_wire,
        )
        object.__setattr__(self, "moe_wire", moe_wire)

    def oracle_inventory_digest(self) -> bytes:
        return economic_oracle_inventory_digest_v3(self.oracles)

    def expected_execution_root(self) -> bytes:
        return economic_execution_root_v3(
            oracles=self.oracles,
            capture_chain_digest=self.capture_chain_digest,
            execution_anchors=self.execution_anchors,
        )

    def canonical_bytes(self) -> bytes:
        writer = _Writer()
        if self.moe_wire:
            version = (
                ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION
                if self.prefix_cache is not None
                else ECONOMIC_MOE_WIRE_FORMAT_VERSION
            )
        elif self.mlp_activation_reveals:
            version = (
                ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION
                if self.prefix_cache is not None
                else ECONOMIC_MLP_ACTIVATION_WIRE_FORMAT_VERSION
            )
        else:
            version = (
                ECONOMIC_PREFIX_CACHE_WIRE_FORMAT_VERSION
                if self.prefix_cache is not None
                else ECONOMIC_WIRE_FORMAT_VERSION
            )
        writer.pack("<4sH", _WIRE_MAGIC, version)
        writer.raw(self.commitment_envelope_digest)
        writer.raw(self.execution_profile_digest)
        writer.raw(self.signed_bound_digest)
        writer.vbytes(
            self.capture_chain_digest,
            "capture_chain_digest",
            MAX_CAPTURE_CHAIN_DIGEST_BYTES,
        )
        writer.pack("<I", len(self.execution_anchors))
        for commitment in self.execution_anchors:
            writer.identifier(
                commitment.stage_id,
                "execution anchor stage_id",
            )
            writer.pack(
                "<II",
                commitment.row_count,
                commitment.row_width,
            )
            writer.raw(commitment.root)
        writer.pack("<I", len(self.execution_anchor_reveals))
        for reveal in self.execution_anchor_reveals:
            reveal.encode(writer)
        writer.pack("<I", len(self.execution_anchor_lane_reveals))
        for reveal in self.execution_anchor_lane_reveals:
            reveal.encode(writer)
        writer.pack("<I", len(self.oracles))
        for oracle in self.oracles:
            oracle.encode(writer)
        writer.pack("<I", len(self.projections))
        for reveal in self.projections:
            reveal.encode(writer)
        writer.pack("<I", len(self.couplings))
        for coupling in self.couplings:
            coupling.encode(writer)
        writer.pack("<I", len(self.gdn_couplings))
        for coupling in self.gdn_couplings:
            coupling.encode(writer)
        if self.moe_wire:
            writer.pack("<I", len(self.mlp_activation_reveals))
            for reveal in self.mlp_activation_reveals:
                reveal.encode(writer)
        elif self.mlp_activation_reveals:
            writer.pack("<I", len(self.mlp_activation_reveals))
            for reveal in self.mlp_activation_reveals:
                reveal.encode(writer)
        writer.pack("<B", 1 if self.lean_projection_batch_wire else 0)
        if self.lean_projection_batch_wire:
            writer.vbytes(
                self.lean_projection_batch_wire,
                "lean projection batch wire",
                MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3,
            )
        writer.pack(
            "<B", 1 if self.succinct_projection_batch_wire else 0
        )
        if self.succinct_projection_batch_wire:
            writer.vbytes(
                self.succinct_projection_batch_wire,
                "succinct projection batch wire",
                MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
            )
        writer.pack("<B", 1 if self.selected_trace_wire else 0)
        if self.selected_trace_wire:
            writer.vbytes(
                self.selected_trace_wire,
                "selected trace wire",
                MAX_SELECTED_TRACE_WIRE_BYTES_V3,
            )
        if self.moe_wire:
            from verallm.proof_v3.economic_moe_wire import (
                MAX_ECONOMIC_MOE_WIRE_BYTES_V3,
            )

            writer.vbytes(
                self.moe_wire,
                "MoE wire",
                MAX_ECONOMIC_MOE_WIRE_BYTES_V3,
            )
        writer.pack("<B", 1 if self.chain is not None else 0)
        if self.chain is not None:
            self.chain.encode(writer)
        writer.pack("<B", 1 if self.final is not None else 0)
        if self.final is not None:
            self.final.encode(writer)
        writer.pack("<B", 1 if self.attention is not None else 0)
        if self.attention is not None:
            self.attention.encode(writer)
        if self.prefix_cache is not None:
            _encode_prefix_cache_section_v3(writer, self.prefix_cache)
        encoded = writer.finish()
        if len(encoded) > MAX_ECONOMIC_WIRE_BYTES:
            raise ProofV3Error("economic proof exceeds the wire byte limit")
        return encoded

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "EconomicRecomputeProofV3":
        reader = _Reader(encoded, "economic recompute proof")
        magic, version = reader.unpack("<4sH")
        if magic != _WIRE_MAGIC or version not in {
            ECONOMIC_WIRE_FORMAT_VERSION,
            ECONOMIC_PREFIX_CACHE_WIRE_FORMAT_VERSION,
            ECONOMIC_MLP_ACTIVATION_WIRE_FORMAT_VERSION,
            ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION,
            ECONOMIC_MOE_WIRE_FORMAT_VERSION,
            ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION,
        }:
            raise ProofV3Error("economic recompute proof header is not supported")
        commitment_envelope_digest = reader.read(32)
        execution_profile_digest = reader.read(32)
        signed_bound_digest = reader.read(32)
        capture_chain_digest = reader.vbytes(
            "capture_chain_digest", MAX_CAPTURE_CHAIN_DIGEST_BYTES
        )
        anchor_count = reader.count(
            "execution anchor inventory",
            MAX_EXECUTION_ANCHORS,
            allow_zero=True,
        )
        execution_anchors = tuple(
            ExecutionAnchorCommitmentV3(
                stage_id=reader.identifier("execution anchor stage_id"),
                row_count=reader.unpack("<I")[0],
                row_width=reader.unpack("<I")[0],
                root=reader.read(32),
            )
            for _ in range(anchor_count)
        )
        anchor_reveal_count = reader.count(
            "execution anchor reveals",
            MAX_EXECUTION_ANCHOR_REVEALS,
            allow_zero=True,
        )
        execution_anchor_reveals = tuple(
            EconomicExecutionAnchorRevealV3.decode(reader)
            for _ in range(anchor_reveal_count)
        )
        anchor_lane_reveal_count = reader.count(
            "execution anchor lane reveals",
            MAX_EXECUTION_ANCHOR_LANE_REVEALS,
            allow_zero=True,
        )
        execution_anchor_lane_reveals = tuple(
            EconomicExecutionAnchorLaneRevealV3.decode(reader)
            for _ in range(anchor_lane_reveal_count)
        )
        oracle_count = reader.count("oracle inventory", MAX_ORACLES)
        oracles = tuple(
            EconomicOracleCommitmentV3.decode(reader) for _ in range(oracle_count)
        )
        projection_count = reader.count(
            "projection reveals", MAX_PROJECTION_REVEALS, allow_zero=True
        )
        projections = tuple(
            EconomicProjectionRevealV3.decode(reader)
            for _ in range(projection_count)
        )
        coupling_count = reader.count(
            "coupling reveals", MAX_COUPLING_REVEALS, allow_zero=True
        )
        couplings = tuple(
            EconomicLayerCouplingRevealV3.decode(reader)
            for _ in range(coupling_count)
        )
        gdn_coupling_count = reader.count(
            "GDN coupling reveals", MAX_COUPLING_REVEALS, allow_zero=True
        )
        gdn_couplings = tuple(
            EconomicGdnLayerCouplingRevealV3.decode(reader)
            for _ in range(gdn_coupling_count)
        )
        mlp_activation_reveals = ()
        if version in {
            ECONOMIC_MLP_ACTIVATION_WIRE_FORMAT_VERSION,
            ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION,
            ECONOMIC_MOE_WIRE_FORMAT_VERSION,
            ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION,
        }:
            mlp_activation_count = reader.count(
                "MLP activation reveals",
                MAX_COUPLING_REVEALS,
                allow_zero=version
                in {
                    ECONOMIC_MOE_WIRE_FORMAT_VERSION,
                    ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION,
                },
            )
            mlp_activation_reveals = tuple(
                EconomicMlpActivationRevealV3.decode(reader)
                for _ in range(mlp_activation_count)
            )
        lean_projection_batch_flag = reader.unpack("<B")[0]
        if lean_projection_batch_flag not in (0, 1):
            raise ProofV3Error(
                "lean projection batch presence flag is not canonical"
            )
        lean_projection_batch_wire = (
            reader.vbytes(
                "lean projection batch wire",
                MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3,
            )
            if lean_projection_batch_flag
            else b""
        )
        succinct_projection_batch_flag = reader.unpack("<B")[0]
        if succinct_projection_batch_flag not in (0, 1):
            raise ProofV3Error(
                "succinct projection batch presence flag is not canonical"
            )
        succinct_projection_batch_wire = (
            reader.vbytes(
                "succinct projection batch wire",
                MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
            )
            if succinct_projection_batch_flag
            else b""
        )
        selected_trace_flag = reader.unpack("<B")[0]
        if selected_trace_flag not in (0, 1):
            raise ProofV3Error(
                "selected trace presence flag is not canonical"
            )
        selected_trace_wire = (
            reader.vbytes(
                "selected trace wire",
                MAX_SELECTED_TRACE_WIRE_BYTES_V3,
            )
            if selected_trace_flag
            else b""
        )
        moe_wire = b""
        if version in {
            ECONOMIC_MOE_WIRE_FORMAT_VERSION,
            ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION,
        }:
            from verallm.proof_v3.economic_moe_wire import (
                MAX_ECONOMIC_MOE_WIRE_BYTES_V3,
            )

            moe_wire = reader.vbytes(
                "MoE wire",
                MAX_ECONOMIC_MOE_WIRE_BYTES_V3,
            )
        chain_flag = reader.unpack("<B")[0]
        if chain_flag not in (0, 1):
            raise ProofV3Error("economic chain flag is not canonical")
        chain = EconomicChainRevealV3.decode(reader) if chain_flag else None
        final_flag = reader.unpack("<B")[0]
        if final_flag not in (0, 1):
            raise ProofV3Error("economic final flag is not canonical")
        final = EconomicFinalRevealV3.decode(reader) if final_flag else None
        attention_flag = reader.unpack("<B")[0]
        if attention_flag not in (0, 1):
            raise ProofV3Error("economic attention flag is not canonical")
        attention = (
            EconomicAttentionRequestSectionV3.decode(reader)
            if attention_flag else None
        )
        prefix_cache = (
            _decode_prefix_cache_section_v3(reader)
            if version in {
                ECONOMIC_PREFIX_CACHE_WIRE_FORMAT_VERSION,
                ECONOMIC_MLP_ACTIVATION_PREFIX_CACHE_WIRE_FORMAT_VERSION,
                ECONOMIC_MOE_PREFIX_CACHE_WIRE_FORMAT_VERSION,
            }
            else None
        )
        reader.finish()
        result = cls(
            commitment_envelope_digest=commitment_envelope_digest,
            execution_profile_digest=execution_profile_digest,
            signed_bound_digest=signed_bound_digest,
            capture_chain_digest=capture_chain_digest,
            execution_anchors=execution_anchors,
            execution_anchor_reveals=execution_anchor_reveals,
            execution_anchor_lane_reveals=execution_anchor_lane_reveals,
            oracles=oracles,
            projections=projections,
            couplings=couplings,
            gdn_couplings=gdn_couplings,
            mlp_activation_reveals=mlp_activation_reveals,
            lean_projection_batch_wire=lean_projection_batch_wire,
            succinct_projection_batch_wire=succinct_projection_batch_wire,
            selected_trace_wire=selected_trace_wire,
            moe_wire=moe_wire,
            attention=attention,
            chain=chain,
            final=final,
            prefix_cache=prefix_cache,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("economic recompute proof is not canonical")
        return result
