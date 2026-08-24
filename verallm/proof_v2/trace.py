"""Canonical causal execution-trace commitments for proof protocol v2.

This module only defines the commitment and opening boundary.  Arithmetic
verification of registered projections remains in :mod:`verallm.proof_v2.engine`.
An opened layer is accepted only when the verifier also checks the corresponding
operation rows under the authenticated X/runtime-Y roots.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from verallm.crypto.merkle import MerkleTree, verify_merkle_path
from zkllm.types import MerklePath

TRACE_PROFILE_QWEN_HYBRID_DENSE_V1 = "qwen_hybrid_dense_trace_v1"
TRACE_PROFILE_QWEN_HYBRID_SPARSE_MOE_SHELL_V1 = (
    "qwen_hybrid_sparse_moe_shell_trace_v1"
)
TRACE_ATTENTION_FULL_AUDIT_ONLY = "full_attention_audit_only"
TRACE_ATTENTION_FULL_TRANSITION_V1 = "full_attention_transition_v1"
TRACE_ATTENTION_GDN_AUDIT_ONLY = "gdn_attention_audit_only"
TRACE_ATTENTION_GDN_TRANSITION_V1 = "gdn_attention_transition_v1"
# vLLM's prefill cache is the state *after* the final prompt token.  Trace
# row 0 is that final prompt-token forward pass, therefore a cache opening can
# only start the independently replayed suffix at row 1.  Keeping this as a
# protocol constant prevents a post-prefill state from being mislabeled as the
# state before row 0.
GDN_DECODE_SUFFIX_TOKEN_START_V1 = 1

_TENSOR_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_TENSOR/SHA256"
_LAYER_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_LAYER/SHA256"
_TOKEN_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_TOKEN/SHA256"
_IO_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_IO/SHA256"
_TRACE_PROOF_MAGIC_V3 = b"V2T3"
_TRACE_PROOF_MAGIC_V4 = b"V2T4"
_TAIL_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_TAIL/SHA256"
_RESIDUAL_BOUNDARY_DOMAIN = b"VERATHOS/PROOF_V2/TRACE_RESIDUAL_BOUNDARY/SHA256"
_ATTENTION_STATE_BOUNDARY_DOMAIN = (
    b"VERATHOS/PROOF_V2/TRACE_ATTENTION_STATE_BOUNDARY/SHA256"
)
_GDN_STATE_DOMAIN = b"VERATHOS/PROOF_V2/GDN_STATE/SHA256"
_FULL_ATTENTION_HEAD_STATE_DOMAIN = (
    b"VERATHOS/PROOF_V2/FULL_ATTENTION_HEAD_STATE/SHA256"
)
_GDN_TRANSITION_WITNESS_DOMAIN = (
    b"VERATHOS/PROOF_V2/GDN_TRANSITION_WITNESS/SHA256"
)
_FULL_ATTENTION_HEAD_WITNESS_DOMAIN = (
    b"VERATHOS/PROOF_V2/FULL_ATTENTION_HEAD_WITNESS/SHA256"
)
_GDN_INITIAL_STATE_MAGIC = b"GSI2"
_FULL_ATTENTION_STATE_MAGIC = b"FAI2"
_GDN_TRANSITION_WITNESS_MAGIC = b"GTW2"
_FULL_ATTENTION_HEAD_WITNESS_MAGIC = b"FHW2"
_GDN_STATE_DTYPE_CODES = {"f16": 1, "f32": 2}
MAX_TRACE_TENSOR_BYTES = 8 << 20
MAX_TRACE_LAYER_BYTES = 16 << 20
MAX_TRACE_PROOF_BYTES = 24 << 20
MAX_GDN_STATE_BYTES = 8 << 20
MAX_FULL_ATTENTION_HEAD_STATE_BYTES = 16 << 20
MAX_FULL_ATTENTION_STATE_PATHS = 16_384
MAX_TRACE_TOKENS = 16_384
MAX_TRACE_LAYERS = 16_384
MAX_TRACE_OPENED_LAYERS = 16_384

_DTYPE_SIZES = {
    "i8": 1,
    "i32": 4,
    "f16": 2,
    "f32": 4,
    "digest32": 32,
}

_COMMON_LAYER_FIELDS = (
    "residual_in",
    "x_attn",
    "attention_core_out",
    "attention_out_proj",
    "residual_after_attention",
    "x_ffn",
    "mlp_gate_up",
    "mlp_hidden",
    "mlp_down",
    "residual_out",
)

_FULL_ATTENTION_FIELDS = (
    "attention_qkv",
    "kv_before_digest",
    "kv_after_digest",
)

_GDN_ATTENTION_FIELDS = (
    "gdn_qkvz",
    "gdn_ba",
    "gdn_conv_before_digest",
    "gdn_conv_after_digest",
    "gdn_recurrent_before_digest",
    "gdn_recurrent_after_digest",
)


class ProofV2TraceError(ValueError):
    """A causal trace commitment or opening is malformed."""


def trace_tail_digest_v2(label: str, value: bytes) -> bytes:
    """Hash a canonical final-hidden or LM-head row into a token leaf."""

    encoded_label = _text(label, "trace tail label")
    if not isinstance(value, bytes) or not value or len(value) > MAX_TRACE_TENSOR_BYTES:
        raise ProofV2TraceError("trace tail value length is out of range")
    return hashlib.sha256(
        _TAIL_DOMAIN
        + encoded_label
        + _u32(len(value), "trace tail value length")
        + value
    ).digest()


def trace_residual_boundary_digest_v2(value: bytes) -> bytes:
    """Commit one canonical fp16 residual row shared by adjacent layers."""

    if not isinstance(value, bytes) or not value or len(value) > MAX_TRACE_TENSOR_BYTES:
        raise ProofV2TraceError("trace residual boundary length is out of range")
    return hashlib.sha256(
        _RESIDUAL_BOUNDARY_DOMAIN
        + _u32(len(value), "trace residual boundary length")
        + value
    ).digest()


def attention_state_tensor_names_v2(
    attention_profile: str,
    *,
    before: bool,
) -> tuple[str, ...]:
    """Return the exact state fields folded into one compact layer boundary."""

    suffix = "before_digest" if before else "after_digest"
    if attention_profile in (
        TRACE_ATTENTION_FULL_AUDIT_ONLY,
        TRACE_ATTENTION_FULL_TRANSITION_V1,
    ):
        return (f"kv_{suffix}",)
    if attention_profile in (
        TRACE_ATTENTION_GDN_AUDIT_ONLY,
        TRACE_ATTENTION_GDN_TRANSITION_V1,
    ):
        return (
            f"gdn_conv_{suffix}",
            f"gdn_recurrent_{suffix}",
        )
    raise ProofV2TraceError("attention trace profile is not supported")


def trace_attention_state_boundary_digest_v2(
    attention_profile: str,
    values: Sequence[bytes],
) -> bytes:
    """Fold the verifier-owned KV/GDN state tuple into one boundary digest."""

    encoded_profile = _text(attention_profile, "attention trace profile")
    try:
        states = tuple(values)
    except TypeError as exc:
        raise ProofV2TraceError("attention state boundary must be a sequence") from exc
    expected_count = (
        1
        if attention_profile
        in (TRACE_ATTENTION_FULL_AUDIT_ONLY, TRACE_ATTENTION_FULL_TRANSITION_V1)
        else 2
        if attention_profile
        in (TRACE_ATTENTION_GDN_AUDIT_ONLY, TRACE_ATTENTION_GDN_TRANSITION_V1)
        else 0
    )
    if len(states) != expected_count:
        raise ProofV2TraceError("attention state boundary count is not exact")
    for index, state in enumerate(states):
        _fixed32(state, f"attention state boundary {index}")
    return hashlib.sha256(
        _ATTENTION_STATE_BOUNDARY_DOMAIN
        + encoded_profile
        + struct.pack("<B", len(states))
        + b"".join(states)
    ).digest()


def _gdn_state_dtype_code(dtype: str) -> int:
    try:
        return _GDN_STATE_DTYPE_CODES[dtype]
    except KeyError as exc:
        raise ProofV2TraceError("GDN state dtype is not supported") from exc


def _gdn_state_dtype_from_code(code: int) -> str:
    for dtype, known_code in _GDN_STATE_DTYPE_CODES.items():
        if code == known_code:
            return dtype
    raise ProofV2TraceError("GDN state dtype is not supported")


def _gdn_state_item_size(dtype: str) -> int:
    return 2 if _gdn_state_dtype_code(dtype) == 1 else 4


def gdn_state_digest_v2(
    component: str,
    values: bytes,
    *,
    dtype: str = "f16",
) -> bytes:
    """Commit one canonical live GDN cache component.

    This digest intentionally has no token position: the post-state of one
    token must be byte-for-byte the pre-state of the next token.  The opening
    and signed transition parameters provide the layer-specific dimensions.
    """

    if component not in ("conv", "recurrent"):
        raise ProofV2TraceError("GDN state component is not supported")
    if (
        not isinstance(values, bytes)
        or not values
        or len(values) > MAX_GDN_STATE_BYTES
        or len(values) % _gdn_state_item_size(dtype)
    ):
        raise ProofV2TraceError("GDN state value length is out of range")
    return hashlib.sha256(
        _GDN_STATE_DOMAIN
        + _text(component, "GDN state component")
        + struct.pack("<B", _gdn_state_dtype_code(dtype))
        + _u32(len(values), "GDN state value length")
        + values
    ).digest()


def _u32(value: int, name: str) -> bytes:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < (1 << 32)
    ):
        raise ProofV2TraceError(f"{name} must be an unsigned 32-bit integer")
    return struct.pack("<I", value)


def _text(value: str, name: str) -> bytes:
    if not isinstance(value, str):
        raise ProofV2TraceError(f"{name} must be text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV2TraceError(f"{name} must be ASCII") from exc
    if not encoded or len(encoded) > 255:
        raise ProofV2TraceError(f"{name} length is out of range")
    return struct.pack("<B", len(encoded)) + encoded


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV2TraceError(f"{name} must be exactly 32 bytes")
    return value


class _TraceReader:
    def __init__(self, encoded: bytes, maximum: int, name: str):
        if not isinstance(encoded, bytes) or not encoded or len(encoded) > maximum:
            raise ProofV2TraceError(f"{name} length is out of range")
        self.encoded = encoded
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.encoded) - self.offset

    def read(self, length: int) -> bytes:
        if length < 0 or length > self.remaining:
            raise ProofV2TraceError("execution trace payload is truncated")
        start = self.offset
        self.offset += length
        return self.encoded[start : start + length]

    def unpack(self, format_: str) -> tuple:
        size = struct.calcsize(format_)
        try:
            return struct.unpack(format_, self.read(size))
        except struct.error as exc:
            raise ProofV2TraceError("execution trace payload is malformed") from exc

    def text(self, name: str) -> str:
        length = self.unpack("<B")[0]
        if length == 0:
            raise ProofV2TraceError(f"{name} length is out of range")
        try:
            return self.read(length).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV2TraceError(f"{name} must be ASCII") from exc

    def finish(self) -> None:
        if self.remaining:
            raise ProofV2TraceError("execution trace payload contains trailing data")


def required_layer_fields_v2(attention_profile: str) -> tuple[str, ...]:
    """Return the verifier-owned exact field set for one trace layer."""

    if attention_profile in (
        TRACE_ATTENTION_FULL_AUDIT_ONLY,
        TRACE_ATTENTION_FULL_TRANSITION_V1,
    ):
        attention = _FULL_ATTENTION_FIELDS
    elif attention_profile in (
        TRACE_ATTENTION_GDN_AUDIT_ONLY,
        TRACE_ATTENTION_GDN_TRANSITION_V1,
    ):
        attention = _GDN_ATTENTION_FIELDS
    else:
        raise ProofV2TraceError("attention trace profile is not supported")
    return tuple(sorted((*_COMMON_LAYER_FIELDS, *attention)))


@dataclass(frozen=True, order=True)
class TraceTensorV2:
    """One canonically encoded tensor or fixed state digest in a trace leaf."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    values: bytes

    def __post_init__(self) -> None:
        _text(self.name, "trace tensor name")
        if self.dtype not in _DTYPE_SIZES:
            raise ProofV2TraceError("trace tensor dtype is not supported")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise ProofV2TraceError("trace tensor shape must be a sequence") from exc
        if not shape or len(shape) > 4:
            raise ProofV2TraceError("trace tensor rank is out of range")
        element_count = 1
        for dimension in shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or not 0 < dimension < (1 << 32)
            ):
                raise ProofV2TraceError("trace tensor dimensions must be positive")
            element_count *= dimension
        expected = element_count * _DTYPE_SIZES[self.dtype]
        if not isinstance(self.values, bytes) or len(self.values) != expected:
            raise ProofV2TraceError(
                "trace tensor byte length does not match its dtype and shape"
            )
        if self.dtype == "digest32" and shape != (1,):
            raise ProofV2TraceError("trace state digests must use shape (1,)")
        object.__setattr__(self, "shape", shape)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray()
        encoded.extend(_text(self.name, "trace tensor name"))
        encoded.extend(_text(self.dtype, "trace tensor dtype"))
        encoded.extend(struct.pack("<B", len(self.shape)))
        for dimension in self.shape:
            encoded.extend(_u32(dimension, "trace tensor dimension"))
        encoded.extend(_u32(len(self.values), "trace tensor byte length"))
        encoded.extend(self.values)
        return bytes(encoded)

    def digest(self) -> bytes:
        return hashlib.sha256(_TENSOR_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "TraceTensorV2":
        reader = _TraceReader(encoded, MAX_TRACE_TENSOR_BYTES, "trace tensor")
        name = reader.text("trace tensor name")
        dtype = reader.text("trace tensor dtype")
        rank = reader.unpack("<B")[0]
        if rank == 0 or rank > 4:
            raise ProofV2TraceError("trace tensor rank is out of range")
        shape = tuple(reader.unpack("<I")[0] for _ in range(rank))
        value_length = reader.unpack("<I")[0]
        if value_length > MAX_TRACE_TENSOR_BYTES:
            raise ProofV2TraceError("trace tensor value length is out of range")
        result = cls(name, dtype, shape, reader.read(value_length))
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("trace tensor is not canonical")
        return result


@dataclass(frozen=True)
class LayerExecutionTraceV2:
    """Canonical opened boundary for one generated token and model layer."""

    token_index: int
    layer_idx: int
    attention_profile: str
    tensors: tuple[TraceTensorV2, ...]

    def __post_init__(self) -> None:
        _u32(self.token_index, "trace token index")
        _u32(self.layer_idx, "trace layer index")
        required = required_layer_fields_v2(self.attention_profile)
        try:
            tensors = tuple(self.tensors)
        except TypeError as exc:
            raise ProofV2TraceError("trace tensors must be a sequence") from exc
        if not all(isinstance(item, TraceTensorV2) for item in tensors):
            raise ProofV2TraceError("trace tensors have an unexpected type")
        names = tuple(item.name for item in tensors)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ProofV2TraceError("trace tensor names must be sorted and unique")
        if names != required:
            raise ProofV2TraceError("trace tensor set does not match its profile")
        object.__setattr__(self, "tensors", tensors)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray()
        encoded.extend(_u32(self.token_index, "trace token index"))
        encoded.extend(_u32(self.layer_idx, "trace layer index"))
        encoded.extend(_text(self.attention_profile, "attention trace profile"))
        encoded.extend(_u32(len(self.tensors), "trace tensor count"))
        for tensor in self.tensors:
            item = tensor.canonical_bytes()
            encoded.extend(_u32(len(item), "trace tensor encoding length"))
            encoded.extend(item)
        return bytes(encoded)

    def digest(self) -> bytes:
        return hashlib.sha256(_LAYER_DOMAIN + self.canonical_bytes()).digest()

    def tensor(self, name: str) -> TraceTensorV2:
        for item in self.tensors:
            if item.name == name:
                return item
        raise ProofV2TraceError(f"trace tensor {name!r} is missing")

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "LayerExecutionTraceV2":
        reader = _TraceReader(encoded, MAX_TRACE_LAYER_BYTES, "layer trace")
        token_index, layer_idx = reader.unpack("<II")
        attention_profile = reader.text("attention trace profile")
        tensor_count = reader.unpack("<I")[0]
        if tensor_count == 0 or tensor_count > 64:
            raise ProofV2TraceError("trace tensor count is out of range")
        tensors = []
        for _ in range(tensor_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > MAX_TRACE_TENSOR_BYTES:
                raise ProofV2TraceError("trace tensor length is out of range")
            tensors.append(TraceTensorV2.from_canonical_bytes(reader.read(length)))
        reader.finish()
        result = cls(token_index, layer_idx, attention_profile, tuple(tensors))
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("layer trace is not canonical")
        return result


@dataclass(frozen=True)
class TokenExecutionTraceV2:
    """One generated-token trace leaf binding every layer and decode tail."""

    token_index: int
    input_token_id: int
    output_token_id: int
    layer_digests: tuple[bytes, ...]
    residual_boundary_digests: tuple[bytes, ...]
    attention_state_before_digests: tuple[bytes, ...]
    attention_state_after_digests: tuple[bytes, ...]
    final_hidden_digest: bytes
    lm_head_digest: bytes
    previous_io_digest: bytes
    final_hidden_f16: bytes
    # One root per layer for the compact nonlinear-transition witness.  Older
    # trace leaves do not carry these roots and remain parseable only for
    # non-hard compatibility tests; a production hard audit requires them.
    transition_witness_roots: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        _u32(self.token_index, "trace token index")
        _u32(self.input_token_id, "trace input token id")
        _u32(self.output_token_id, "trace output token id")
        try:
            layers = tuple(self.layer_digests)
        except TypeError as exc:
            raise ProofV2TraceError("layer digests must be a sequence") from exc
        if not layers:
            raise ProofV2TraceError("token trace must contain at least one layer")
        for index, digest in enumerate(layers):
            _fixed32(digest, f"layer digest {index}")
        try:
            residuals = tuple(self.residual_boundary_digests)
            state_before = tuple(self.attention_state_before_digests)
            state_after = tuple(self.attention_state_after_digests)
        except TypeError as exc:
            raise ProofV2TraceError(
                "token trace boundary digests must be sequences"
            ) from exc
        if len(residuals) != len(layers) + 1:
            raise ProofV2TraceError("token trace residual boundary count is not exact")
        if len(state_before) != len(layers) or len(state_after) != len(layers):
            raise ProofV2TraceError(
                "token trace attention-state boundary count is not exact"
            )
        for label, values in (
            ("residual boundary", residuals),
            ("attention state before", state_before),
            ("attention state after", state_after),
        ):
            for index, digest in enumerate(values):
                _fixed32(digest, f"{label} digest {index}")
        _fixed32(self.final_hidden_digest, "final hidden digest")
        _fixed32(self.lm_head_digest, "LM-head digest")
        _fixed32(self.previous_io_digest, "previous IO digest")
        if (
            not isinstance(self.final_hidden_f16, bytes)
            or not self.final_hidden_f16
            or len(self.final_hidden_f16) > MAX_TRACE_TENSOR_BYTES
            or len(self.final_hidden_f16) % 2
        ):
            raise ProofV2TraceError("final hidden row length is out of range")
        if self.final_hidden_digest != trace_tail_digest_v2(
            "final_hidden_f16",
            self.final_hidden_f16,
        ):
            raise ProofV2TraceError("final hidden row does not match its token digest")
        try:
            transition_roots = tuple(self.transition_witness_roots)
        except TypeError as exc:
            raise ProofV2TraceError(
                "transition witness roots must be a sequence"
            ) from exc
        if transition_roots:
            if len(transition_roots) != len(layers):
                raise ProofV2TraceError(
                    "transition witness root count is not exact"
                )
            for index, root in enumerate(transition_roots):
                _fixed32(root, f"transition witness root {index}")
        object.__setattr__(self, "layer_digests", layers)
        object.__setattr__(self, "residual_boundary_digests", residuals)
        object.__setattr__(self, "attention_state_before_digests", state_before)
        object.__setattr__(self, "attention_state_after_digests", state_after)
        object.__setattr__(self, "transition_witness_roots", transition_roots)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray()
        encoded.extend(_u32(self.token_index, "trace token index"))
        encoded.extend(_u32(self.input_token_id, "trace input token id"))
        encoded.extend(_u32(self.output_token_id, "trace output token id"))
        encoded.extend(_u32(len(self.layer_digests), "trace layer count"))
        for digest in self.layer_digests:
            encoded.extend(digest)
        for digest in self.residual_boundary_digests:
            encoded.extend(digest)
        for digest in self.attention_state_before_digests:
            encoded.extend(digest)
        for digest in self.attention_state_after_digests:
            encoded.extend(digest)
        encoded.extend(self.final_hidden_digest)
        encoded.extend(self.lm_head_digest)
        encoded.extend(self.previous_io_digest)
        encoded.extend(_u32(len(self.final_hidden_f16), "final hidden row length"))
        encoded.extend(self.final_hidden_f16)
        if self.transition_witness_roots:
            encoded.extend(
                _u32(
                    len(self.transition_witness_roots),
                    "transition witness root count",
                )
            )
            encoded.extend(b"".join(self.transition_witness_roots))
        return bytes(encoded)

    def digest(self) -> bytes:
        return hashlib.sha256(_TOKEN_DOMAIN + self.canonical_bytes()).digest()

    def io_digest(self) -> bytes:
        return hashlib.sha256(_IO_DOMAIN + self.digest()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "TokenExecutionTraceV2":
        reader = _TraceReader(encoded, 1 << 20, "token trace")
        token_index, input_token_id, output_token_id, layer_count = reader.unpack(
            "<IIII"
        )
        if layer_count == 0 or layer_count > MAX_TRACE_LAYERS:
            raise ProofV2TraceError("token trace layer count is out of range")
        layer_digests = tuple(reader.read(32) for _ in range(layer_count))
        residual_digests = tuple(reader.read(32) for _ in range(layer_count + 1))
        state_before_digests = tuple(reader.read(32) for _ in range(layer_count))
        state_after_digests = tuple(reader.read(32) for _ in range(layer_count))
        final_hidden_digest = reader.read(32)
        lm_head_digest = reader.read(32)
        previous_io_digest = reader.read(32)
        final_hidden_length = reader.unpack("<I")[0]
        if (
            final_hidden_length == 0
            or final_hidden_length > MAX_TRACE_TENSOR_BYTES
            or final_hidden_length % 2
        ):
            raise ProofV2TraceError("final hidden row length is out of range")
        final_hidden_f16 = reader.read(final_hidden_length)
        transition_witness_roots = ()
        if reader.remaining:
            root_count = reader.unpack("<I")[0]
            if root_count != layer_count:
                raise ProofV2TraceError(
                    "transition witness root count is not exact"
                )
            transition_witness_roots = tuple(reader.read(32) for _ in range(root_count))
        result = cls(
            token_index,
            input_token_id,
            output_token_id,
            layer_digests,
            residual_digests,
            state_before_digests,
            state_after_digests,
            final_hidden_digest,
            lm_head_digest,
            previous_io_digest,
            final_hidden_f16,
            transition_witness_roots,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("token trace is not canonical")
        return result


@dataclass(frozen=True)
class ExecutionTraceCommitmentV2:
    """Request-level roots frozen before transcript challenge selection."""

    profile: str
    token_count: int
    num_layers: int
    token_trace_root: bytes
    final_io_digest: bytes

    def __post_init__(self) -> None:
        if self.profile != TRACE_PROFILE_QWEN_HYBRID_DENSE_V1:
            raise ProofV2TraceError("execution trace profile is not supported")
        _u32(self.token_count, "trace token count")
        _u32(self.num_layers, "trace layer count")
        if self.token_count == 0:
            raise ProofV2TraceError("trace token count must be positive")
        if self.num_layers == 0:
            raise ProofV2TraceError("trace layer count must be positive")
        _fixed32(self.token_trace_root, "token trace root")
        _fixed32(self.final_io_digest, "final IO digest")

    def canonical_bytes(self) -> bytes:
        return (
            _text(self.profile, "execution trace profile")
            + _u32(self.token_count, "trace token count")
            + _u32(self.num_layers, "trace layer count")
            + self.token_trace_root
            + self.final_io_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            b"VERATHOS/PROOF_V2/TRACE_COMMITMENT/SHA256" + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ExecutionTraceCommitmentV2":
        if not isinstance(encoded, bytes) or not encoded:
            raise ProofV2TraceError("execution trace commitment must be bytes")
        profile_length = encoded[0]
        expected_length = 1 + profile_length + 4 + 4 + 32 + 32
        if profile_length == 0 or len(encoded) != expected_length:
            raise ProofV2TraceError(
                "execution trace commitment length is not canonical"
            )
        try:
            profile = encoded[1 : 1 + profile_length].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV2TraceError(
                "execution trace commitment profile is not ASCII"
            ) from exc
        offset = 1 + profile_length
        token_count, num_layers = struct.unpack_from("<II", encoded, offset)
        offset += 8
        result = cls(
            profile=profile,
            token_count=token_count,
            num_layers=num_layers,
            token_trace_root=encoded[offset : offset + 32],
            final_io_digest=encoded[offset + 32 : offset + 64],
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("execution trace commitment is not canonical")
        return result


@dataclass(frozen=True)
class TokenTraceOpeningV2:
    token: TokenExecutionTraceV2
    merkle_path: MerklePath

    def verify(self, commitment: ExecutionTraceCommitmentV2) -> bool:
        if not isinstance(commitment, ExecutionTraceCommitmentV2):
            raise ProofV2TraceError("trace commitment has an unexpected type")
        if self.token.token_index >= commitment.token_count:
            return False
        if len(self.token.layer_digests) != commitment.num_layers:
            return False
        if self.merkle_path.leaf_index != self.token.token_index:
            return False
        return verify_merkle_path(
            commitment.token_trace_root,
            self.token.canonical_bytes(),
            self.merkle_path,
        )


def build_execution_trace_commitment_v2(
    tokens: Sequence[TokenExecutionTraceV2],
    *,
    profile: str = TRACE_PROFILE_QWEN_HYBRID_DENSE_V1,
) -> tuple[ExecutionTraceCommitmentV2, MerkleTree]:
    """Build the exact token tree and validate its rolling IO chain."""

    try:
        canonical = tuple(tokens)
    except TypeError as exc:
        raise ProofV2TraceError("token traces must be a sequence") from exc
    if not canonical:
        raise ProofV2TraceError("execution trace must contain at least one token")
    if not all(isinstance(item, TokenExecutionTraceV2) for item in canonical):
        raise ProofV2TraceError("token traces have an unexpected type")
    num_layers = len(canonical[0].layer_digests)
    previous = b"\x00" * 32
    previous_output_token_id = None
    previous_attention_state = None
    for index, token in enumerate(canonical):
        if token.token_index != index:
            raise ProofV2TraceError("token trace indices are not canonical")
        if len(token.layer_digests) != num_layers:
            raise ProofV2TraceError("token trace layer counts are inconsistent")
        if (
            previous_output_token_id is not None
            and token.input_token_id != previous_output_token_id
        ):
            raise ProofV2TraceError("generated-token IO chain is discontinuous")
        if token.previous_io_digest != previous:
            raise ProofV2TraceError("token trace IO chain is discontinuous")
        if (
            previous_attention_state is not None
            and token.attention_state_before_digests != previous_attention_state
        ):
            raise ProofV2TraceError(
                "token trace attention-state chain is discontinuous"
            )
        previous = token.io_digest()
        previous_output_token_id = token.output_token_id
        previous_attention_state = token.attention_state_after_digests
    tree = MerkleTree([token.canonical_bytes() for token in canonical])
    commitment = ExecutionTraceCommitmentV2(
        profile=profile,
        token_count=len(canonical),
        num_layers=num_layers,
        token_trace_root=tree.root,
        final_io_digest=previous,
    )
    return commitment, tree


def validate_layer_trace_set_v2(
    token: TokenExecutionTraceV2,
    layers: Iterable[LayerExecutionTraceV2],
) -> None:
    """Require exact ordered layer openings for one token trace."""

    try:
        canonical = tuple(layers)
    except TypeError as exc:
        raise ProofV2TraceError("layer traces must be a sequence") from exc
    if len(canonical) != len(token.layer_digests):
        raise ProofV2TraceError("layer trace opening count is not exact")
    for layer_idx, (expected_digest, layer) in enumerate(
        zip(token.layer_digests, canonical)
    ):
        if not isinstance(layer, LayerExecutionTraceV2):
            raise ProofV2TraceError("layer trace has an unexpected type")
        if layer.token_index != token.token_index or layer.layer_idx != layer_idx:
            raise ProofV2TraceError("layer trace position is not canonical")
        if layer.digest() != expected_digest:
            raise ProofV2TraceError("layer trace does not match its token commitment")
        if layer_idx:
            prior = canonical[layer_idx - 1].tensor("residual_out")
            current = layer.tensor("residual_in")
            if (
                prior.dtype != current.dtype
                or prior.shape != current.shape
                or prior.values != current.values
            ):
                raise ProofV2TraceError("cross-layer residual chain is discontinuous")


def validate_cross_token_state_continuity_v2(
    previous: Sequence[LayerExecutionTraceV2],
    current: Sequence[LayerExecutionTraceV2],
) -> None:
    """Check the committed KV/GDN state boundary between adjacent tokens."""

    try:
        before = tuple(previous)
        after = tuple(current)
    except TypeError as exc:
        raise ProofV2TraceError("layer traces must be sequences") from exc
    if not before or len(before) != len(after):
        raise ProofV2TraceError("cross-token layer trace sets are not exact")
    for layer_idx, (left, right) in enumerate(zip(before, after)):
        if left.layer_idx != layer_idx or right.layer_idx != layer_idx:
            raise ProofV2TraceError("cross-token layer positions are not canonical")
        if right.token_index != left.token_index + 1:
            raise ProofV2TraceError("cross-token trace positions are not adjacent")
        if left.attention_profile != right.attention_profile:
            raise ProofV2TraceError("cross-token attention profiles differ")
        if left.attention_profile in (
            TRACE_ATTENTION_FULL_AUDIT_ONLY,
            TRACE_ATTENTION_FULL_TRANSITION_V1,
        ):
            links = (("kv_after_digest", "kv_before_digest"),)
        elif left.attention_profile in (
            TRACE_ATTENTION_GDN_AUDIT_ONLY,
            TRACE_ATTENTION_GDN_TRANSITION_V1,
        ):
            links = (
                ("gdn_conv_after_digest", "gdn_conv_before_digest"),
                (
                    "gdn_recurrent_after_digest",
                    "gdn_recurrent_before_digest",
                ),
            )
        else:  # pragma: no cover - construction already rejects this
            raise ProofV2TraceError("attention trace profile is not supported")
        for left_name, right_name in links:
            if left.tensor(left_name).values != right.tensor(right_name).values:
                raise ProofV2TraceError(
                    f"layer {layer_idx} cross-token state chain is discontinuous"
                )


@dataclass(frozen=True, order=True)
class GDNTransitionWitnessV2:
    """Compact committed nonlinear witness for one GDN decode row.

    The full layer trace remains the arithmetic witness for the nonce-selected
    row.  This object commits only the values needed to replay every decode
    state transition, which keeps the hard proof independent of MLP width.
    """

    token_index: int
    layer_idx: int
    qkvz_f16: bytes
    ba_f16: bytes
    core_output_f16: bytes
    conv_before_digest: bytes
    conv_after_digest: bytes
    recurrent_before_digest: bytes
    recurrent_after_digest: bytes

    def __post_init__(self) -> None:
        _u32(self.token_index, "GDN transition witness token")
        _u32(self.layer_idx, "GDN transition witness layer")
        for value, name in (
            (self.qkvz_f16, "GDN transition QKVZ"),
            (self.ba_f16, "GDN transition BA"),
            (self.core_output_f16, "GDN transition core output"),
        ):
            if (
                not isinstance(value, bytes)
                or not value
                or len(value) > MAX_TRACE_TENSOR_BYTES
                or len(value) % 2
            ):
                raise ProofV2TraceError(f"{name} length is out of range")
        for value, name in (
            (self.conv_before_digest, "GDN transition convolution pre-state"),
            (self.conv_after_digest, "GDN transition convolution post-state"),
            (
                self.recurrent_before_digest,
                "GDN transition recurrent pre-state",
            ),
            (
                self.recurrent_after_digest,
                "GDN transition recurrent post-state",
            ),
        ):
            _fixed32(value, name)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(
            _GDN_TRANSITION_WITNESS_MAGIC
            + _u32(self.token_index, "GDN transition witness token")
            + _u32(self.layer_idx, "GDN transition witness layer")
        )
        for value, name in (
            (self.qkvz_f16, "GDN transition QKVZ"),
            (self.ba_f16, "GDN transition BA"),
            (self.core_output_f16, "GDN transition core output"),
        ):
            encoded.extend(_u32(len(value), f"{name} length"))
            encoded.extend(value)
        encoded.extend(self.conv_before_digest)
        encoded.extend(self.conv_after_digest)
        encoded.extend(self.recurrent_before_digest)
        encoded.extend(self.recurrent_after_digest)
        return bytes(encoded)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _GDN_TRANSITION_WITNESS_DOMAIN + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "GDNTransitionWitnessV2":
        reader = _TraceReader(
            encoded,
            3 * MAX_TRACE_TENSOR_BYTES + 148,
            "GDN transition witness",
        )
        if reader.read(4) != _GDN_TRANSITION_WITNESS_MAGIC:
            raise ProofV2TraceError("GDN transition witness header is unsupported")
        token_index, layer_idx = reader.unpack("<II")
        values = []
        for name in ("QKVZ", "BA", "core output"):
            length = reader.unpack("<I")[0]
            if length == 0 or length > MAX_TRACE_TENSOR_BYTES or length % 2:
                raise ProofV2TraceError(
                    f"GDN transition witness {name} length is invalid"
                )
            values.append(reader.read(length))
        result = cls(
            token_index,
            layer_idx,
            values[0],
            values[1],
            values[2],
            reader.read(32),
            reader.read(32),
            reader.read(32),
            reader.read(32),
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("GDN transition witness is not canonical")
        return result


@dataclass(frozen=True, order=True)
class FullAttentionHeadWitnessV2:
    """One Merkle-opened Q/K/V/core head for a committed decode row."""

    token_index: int
    layer_idx: int
    query_head_idx: int
    head_dim: int
    query_f16: bytes
    key_f16: bytes
    value_f16: bytes
    core_output_f16: bytes
    kv_before_root: bytes
    kv_after_root: bytes
    merkle_path: MerklePath

    def __post_init__(self) -> None:
        for value, name in (
            (self.token_index, "full-attention witness token"),
            (self.layer_idx, "full-attention witness layer"),
            (self.query_head_idx, "full-attention witness query head"),
            (self.head_dim, "full-attention witness head dimension"),
        ):
            _u32(value, name)
        if self.head_dim == 0:
            raise ProofV2TraceError("full-attention witness head dimension is invalid")
        expected = self.head_dim * 2
        for value, name in (
            (self.query_f16, "full-attention witness query"),
            (self.key_f16, "full-attention witness key"),
            (self.value_f16, "full-attention witness value"),
            (self.core_output_f16, "full-attention witness core output"),
        ):
            if not isinstance(value, bytes) or len(value) != expected:
                raise ProofV2TraceError(f"{name} length is invalid")
        _fixed32(self.kv_before_root, "full-attention witness pre-state root")
        _fixed32(self.kv_after_root, "full-attention witness post-state root")
        if not isinstance(self.merkle_path, MerklePath) or (
            self.merkle_path.leaf_index != self.query_head_idx
        ):
            raise ProofV2TraceError("full-attention witness path is invalid")

    def leaf_bytes(self) -> bytes:
        return (
            _FULL_ATTENTION_HEAD_WITNESS_MAGIC
            + _u32(self.token_index, "full-attention witness token")
            + _u32(self.layer_idx, "full-attention witness layer")
            + _u32(self.query_head_idx, "full-attention witness query head")
            + _u32(self.head_dim, "full-attention witness head dimension")
            + self.query_f16
            + self.key_f16
            + self.value_f16
            + self.core_output_f16
            + self.kv_before_root
            + self.kv_after_root
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _FULL_ATTENTION_HEAD_WITNESS_DOMAIN + self.leaf_bytes()
        ).digest()

    def canonical_bytes(self) -> bytes:
        return self.leaf_bytes() + _encode_merkle_path_v2(
            self.merkle_path,
            "full-attention witness",
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "FullAttentionHeadWitnessV2":
        reader = _TraceReader(
            encoded,
            MAX_TRACE_TENSOR_BYTES,
            "full-attention head witness",
        )
        if reader.read(4) != _FULL_ATTENTION_HEAD_WITNESS_MAGIC:
            raise ProofV2TraceError(
                "full-attention witness header is unsupported"
            )
        token_index, layer_idx, query_head_idx, head_dim = reader.unpack("<IIII")
        if head_dim == 0:
            raise ProofV2TraceError("full-attention witness head dimension is invalid")
        values = tuple(reader.read(head_dim * 2) for _ in range(4))
        result = cls(
            token_index,
            layer_idx,
            query_head_idx,
            head_dim,
            values[0],
            values[1],
            values[2],
            values[3],
            reader.read(32),
            reader.read(32),
            _decode_merkle_path_v2(reader, "full-attention witness"),
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("full-attention witness is not canonical")
        return result


def build_full_attention_witness_root_v2(
    head_witnesses: Sequence[FullAttentionHeadWitnessV2],
) -> tuple[bytes, MerkleTree]:
    """Commit every logical query-head witness for one token/layer pair."""

    try:
        witnesses = tuple(head_witnesses)
    except TypeError as exc:
        raise ProofV2TraceError("full-attention witnesses are invalid") from exc
    if not witnesses or len(witnesses) > 16_384:
        raise ProofV2TraceError("full-attention witness head count is invalid")
    first = witnesses[0]
    positions = tuple(item.query_head_idx for item in witnesses)
    if (
        not all(isinstance(item, FullAttentionHeadWitnessV2) for item in witnesses)
        or positions != tuple(range(len(witnesses)))
        or any(
            item.token_index != first.token_index
            or item.layer_idx != first.layer_idx
            or item.head_dim != first.head_dim
            for item in witnesses
        )
    ):
        raise ProofV2TraceError("full-attention witness set is not canonical")
    tree = MerkleTree([item.digest() for item in witnesses])
    return tree.root, tree


@dataclass(frozen=True, order=True)
class GDNInitialStateOpeningV2:
    """Raw prompt-bound GDN state for one nonce-selected layer.

    Token leaves contain compact state digests.  This opening reveals the
    state at the prompt/decode boundary only after challenge selection, so the
    verifier can replay the committed decode suffix without trusting a
    miner-created digest update.  For the v1 vLLM ABI, ``token_start`` is
    exactly :data:`GDN_DECODE_SUFFIX_TOKEN_START_V1` because the captured cache
    follows the final prompt-token trace row.
    """

    layer_idx: int
    conv_state: bytes
    recurrent_state: bytes
    token_start: int = 0
    conv_state_dtype: str = "f16"
    recurrent_state_dtype: str = "f16"

    def __post_init__(self) -> None:
        _u32(self.layer_idx, "GDN state layer")
        _u32(self.token_start, "GDN state token start")
        for name, value, dtype in (
            ("GDN convolution state", self.conv_state, self.conv_state_dtype),
            ("GDN recurrent state", self.recurrent_state, self.recurrent_state_dtype),
        ):
            if (
                not isinstance(value, bytes)
                or not value
                or len(value) > MAX_GDN_STATE_BYTES
                or len(value) % _gdn_state_item_size(dtype)
            ):
                raise ProofV2TraceError(f"{name} length is out of range")

    def canonical_bytes(self) -> bytes:
        return (
            _GDN_INITIAL_STATE_MAGIC
            + _u32(self.layer_idx, "GDN state layer")
            + _u32(self.token_start, "GDN state token start")
            + struct.pack("<B", _gdn_state_dtype_code(self.conv_state_dtype))
            + struct.pack("<B", _gdn_state_dtype_code(self.recurrent_state_dtype))
            + _u32(len(self.conv_state), "GDN convolution state length")
            + _u32(len(self.recurrent_state), "GDN recurrent state length")
            + self.conv_state
            + self.recurrent_state
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "GDNInitialStateOpeningV2":
        reader = _TraceReader(encoded, 2 * MAX_GDN_STATE_BYTES + 22, "GDN state")
        magic = reader.read(4)
        if magic == _GDN_INITIAL_STATE_MAGIC:
            (
                layer_idx,
                token_start,
                conv_dtype_code,
                recurrent_dtype_code,
                conv_length,
                recurrent_length,
            ) = reader.unpack("<IIBBII")
            conv_dtype = _gdn_state_dtype_from_code(conv_dtype_code)
            recurrent_dtype = _gdn_state_dtype_from_code(recurrent_dtype_code)
        else:
            raise ProofV2TraceError("GDN state opening header is unsupported")
        if (
            conv_length == 0
            or recurrent_length == 0
            or conv_length > MAX_GDN_STATE_BYTES
            or recurrent_length > MAX_GDN_STATE_BYTES
            or conv_length % _gdn_state_item_size(conv_dtype)
            or recurrent_length % _gdn_state_item_size(recurrent_dtype)
        ):
            raise ProofV2TraceError("GDN state opening length is out of range")
        result = cls(
            layer_idx,
            reader.read(conv_length),
            reader.read(recurrent_length),
            token_start,
            conv_dtype,
            recurrent_dtype,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("GDN state opening is not canonical")
        return result


def full_attention_head_state_leaf_v2(
    *,
    kv_head_idx: int,
    head_dim: int,
    key_values_f16: bytes,
    value_values_f16: bytes,
) -> bytes:
    """Commit one logical K/V head history before or after a trace row."""

    _u32(kv_head_idx, "full-attention KV head")
    _u32(head_dim, "full-attention head dimension")
    if head_dim == 0:
        raise ProofV2TraceError("full-attention head dimension is invalid")
    for value, name in (
        (key_values_f16, "full-attention key state"),
        (value_values_f16, "full-attention value state"),
    ):
        if (
            not isinstance(value, bytes)
            or len(value) > MAX_FULL_ATTENTION_HEAD_STATE_BYTES
            or len(value) % (head_dim * 2)
        ):
            raise ProofV2TraceError(f"{name} length is out of range")
    if len(key_values_f16) != len(value_values_f16):
        raise ProofV2TraceError("full-attention K/V state lengths differ")
    return hashlib.sha256(
        _FULL_ATTENTION_HEAD_STATE_DOMAIN
        + _u32(kv_head_idx, "full-attention KV head")
        + _u32(head_dim, "full-attention head dimension")
        + _u32(
            len(key_values_f16) // (head_dim * 2),
            "full-attention state row count",
        )
        + key_values_f16
        + value_values_f16
    ).digest()


def build_full_attention_state_root_v2(
    head_leaves: Sequence[bytes],
) -> tuple[bytes, MerkleTree]:
    """Build the per-KV-head logical-cache root retained in trace leaves."""

    try:
        leaves = tuple(head_leaves)
    except TypeError as exc:
        raise ProofV2TraceError("full-attention head leaves are invalid") from exc
    if not leaves or len(leaves) > 16_384:
        raise ProofV2TraceError("full-attention head count is out of range")
    for index, leaf in enumerate(leaves):
        _fixed32(leaf, f"full-attention head leaf {index}")
    tree = MerkleTree(list(leaves))
    return tree.root, tree


def _encode_merkle_path_v2(path: MerklePath, name: str) -> bytes:
    if not isinstance(path, MerklePath):
        raise ProofV2TraceError(f"{name} path is invalid")
    _u32(path.leaf_index, f"{name} leaf index")
    siblings = tuple(path.siblings)
    if len(siblings) > 32:
        raise ProofV2TraceError(f"{name} path is too deep")
    encoded = bytearray(
        _u32(path.leaf_index, f"{name} leaf index")
        + _u32(len(siblings), f"{name} path length")
    )
    for index, sibling in enumerate(siblings):
        if (
            not isinstance(sibling, tuple)
            or len(sibling) != 2
            or not isinstance(sibling[1], bool)
        ):
            raise ProofV2TraceError(f"{name} sibling {index} is invalid")
        encoded.extend(_fixed32(sibling[0], f"{name} sibling {index}"))
        encoded.extend(struct.pack("<?", sibling[1]))
    return bytes(encoded)


def _decode_merkle_path_v2(reader: _TraceReader, name: str) -> MerklePath:
    leaf_index, count = reader.unpack("<II")
    if count > 32:
        raise ProofV2TraceError(f"{name} path is too deep")
    siblings = []
    for _ in range(count):
        digest = reader.read(32)
        is_left = reader.unpack("<?")[0]
        siblings.append((digest, is_left))
    return MerklePath(leaf_index=leaf_index, siblings=siblings)


@dataclass(frozen=True, order=True)
class FullAttentionHeadStateOpeningV2:
    """Nonce-selected logical K/V-head prefix plus every trace-root path.

    The opening reveals only one K/V head.  The per-token before/after paths
    prove that this head is part of the committed full-cache roots while the
    verifier derives the missing suffix K/V rows from the opened QKV trace.
    """

    layer_idx: int
    kv_head_idx: int
    head_dim: int
    position_start: int
    prefix_keys_f16: bytes
    prefix_values_f16: bytes
    before_paths: tuple[MerklePath, ...]
    after_paths: tuple[MerklePath, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.layer_idx, "full-attention opening layer"),
            (self.kv_head_idx, "full-attention opening KV head"),
            (self.head_dim, "full-attention opening head dimension"),
            (self.position_start, "full-attention opening position start"),
        ):
            _u32(value, name)
        if self.head_dim == 0:
            raise ProofV2TraceError("full-attention opening head dimension is invalid")
        keys = self.prefix_keys_f16
        values = self.prefix_values_f16
        expected_multiple = self.head_dim * 2
        if (
            not isinstance(keys, bytes)
            or not isinstance(values, bytes)
            or len(keys) != len(values)
            or len(keys) > MAX_FULL_ATTENTION_HEAD_STATE_BYTES
            or len(keys) % expected_multiple
        ):
            raise ProofV2TraceError("full-attention opening K/V prefix is invalid")
        before = tuple(self.before_paths)
        after = tuple(self.after_paths)
        if (
            not before
            or len(before) != len(after)
            or len(before) > MAX_FULL_ATTENTION_STATE_PATHS
            or not all(isinstance(item, MerklePath) for item in before + after)
            or any(item.leaf_index != self.kv_head_idx for item in before + after)
        ):
            raise ProofV2TraceError("full-attention opening path set is invalid")
        object.__setattr__(self, "before_paths", before)
        object.__setattr__(self, "after_paths", after)

    @property
    def trace_row_count(self) -> int:
        return len(self.before_paths)

    @property
    def prefix_token_count(self) -> int:
        return len(self.prefix_keys_f16) // (self.head_dim * 2)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(
            _FULL_ATTENTION_STATE_MAGIC
            + struct.pack(
                "<IIIII",
                self.layer_idx,
                self.kv_head_idx,
                self.head_dim,
                self.position_start,
                self.trace_row_count,
            )
            + _u32(len(self.prefix_keys_f16), "full-attention key prefix length")
            + self.prefix_keys_f16
            + _u32(len(self.prefix_values_f16), "full-attention value prefix length")
            + self.prefix_values_f16
        )
        for index, (before, after) in enumerate(
            zip(self.before_paths, self.after_paths)
        ):
            for label, path in (("before", before), ("after", after)):
                item = _encode_merkle_path_v2(
                    path,
                    f"full-attention {label} path {index}",
                )
                encoded.extend(_u32(len(item), "full-attention path length"))
                encoded.extend(item)
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "FullAttentionHeadStateOpeningV2":
        reader = _TraceReader(
            encoded,
            2 * MAX_FULL_ATTENTION_HEAD_STATE_BYTES + (MAX_FULL_ATTENTION_STATE_PATHS * 2_048),
            "full-attention state opening",
        )
        if reader.read(4) != _FULL_ATTENTION_STATE_MAGIC:
            raise ProofV2TraceError("full-attention state opening header is unsupported")
        layer_idx, kv_head_idx, head_dim, position_start, row_count = reader.unpack(
            "<IIIII"
        )
        if row_count == 0 or row_count > MAX_FULL_ATTENTION_STATE_PATHS:
            raise ProofV2TraceError("full-attention opening row count is invalid")
        key_length = reader.unpack("<I")[0]
        if key_length > MAX_FULL_ATTENTION_HEAD_STATE_BYTES:
            raise ProofV2TraceError("full-attention key prefix is too large")
        keys = reader.read(key_length)
        value_length = reader.unpack("<I")[0]
        if value_length > MAX_FULL_ATTENTION_HEAD_STATE_BYTES:
            raise ProofV2TraceError("full-attention value prefix is too large")
        values = reader.read(value_length)
        before = []
        after = []
        for index in range(row_count):
            before_length = reader.unpack("<I")[0]
            if before_length == 0 or before_length > 1_024:
                raise ProofV2TraceError("full-attention before path is invalid")
            before_reader = _TraceReader(
                reader.read(before_length), 1_024, "full-attention before path"
            )
            before.append(
                _decode_merkle_path_v2(
                    before_reader,
                    f"full-attention before path {index}",
                )
            )
            before_reader.finish()
            after_length = reader.unpack("<I")[0]
            if after_length == 0 or after_length > 1_024:
                raise ProofV2TraceError("full-attention after path is invalid")
            after_reader = _TraceReader(
                reader.read(after_length), 1_024, "full-attention after path"
            )
            after.append(
                _decode_merkle_path_v2(
                    after_reader,
                    f"full-attention after path {index}",
                )
            )
            after_reader.finish()
        reader.finish()
        result = cls(
            layer_idx,
            kv_head_idx,
            head_dim,
            position_start,
            keys,
            values,
            tuple(before),
            tuple(after),
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("full-attention state opening is not canonical")
        return result


@dataclass(frozen=True)
class ExecutionTraceProofV2:
    """All compact token leaves plus the exact nonce-selected layer openings."""

    tokens: tuple[TokenExecutionTraceV2, ...]
    opened_layers: tuple[LayerExecutionTraceV2, ...]
    gdn_initial_state_openings: tuple[GDNInitialStateOpeningV2, ...] = ()
    full_attention_state_openings: tuple[FullAttentionHeadStateOpeningV2, ...] = ()
    gdn_transition_witnesses: tuple[GDNTransitionWitnessV2, ...] = ()
    full_attention_head_witnesses: tuple[FullAttentionHeadWitnessV2, ...] = ()

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        layers = tuple(self.opened_layers)
        gdn_states = tuple(self.gdn_initial_state_openings)
        full_attention_states = tuple(self.full_attention_state_openings)
        gdn_witnesses = tuple(self.gdn_transition_witnesses)
        full_attention_witnesses = tuple(self.full_attention_head_witnesses)
        if not tokens or len(tokens) > MAX_TRACE_TOKENS:
            raise ProofV2TraceError("trace proof token count is out of range")
        if len(layers) > MAX_TRACE_OPENED_LAYERS:
            raise ProofV2TraceError("opened trace layer count is out of range")
        if not all(isinstance(item, TokenExecutionTraceV2) for item in tokens):
            raise ProofV2TraceError("trace proof tokens have an unexpected type")
        if tuple(item.token_index for item in tokens) != tuple(range(len(tokens))):
            raise ProofV2TraceError("trace proof tokens are not exact and ordered")
        if not all(isinstance(item, LayerExecutionTraceV2) for item in layers):
            raise ProofV2TraceError("opened trace layers have an unexpected type")
        if not all(isinstance(item, GDNInitialStateOpeningV2) for item in gdn_states):
            raise ProofV2TraceError("GDN state openings have an unexpected type")
        if not all(
            isinstance(item, FullAttentionHeadStateOpeningV2)
            for item in full_attention_states
        ):
            raise ProofV2TraceError(
                "full-attention state openings have an unexpected type"
            )
        if not all(isinstance(item, GDNTransitionWitnessV2) for item in gdn_witnesses):
            raise ProofV2TraceError("GDN transition witnesses have an unexpected type")
        if not all(
            isinstance(item, FullAttentionHeadWitnessV2)
            for item in full_attention_witnesses
        ):
            raise ProofV2TraceError(
                "full-attention transition witnesses have an unexpected type"
            )
        if tuple(item.layer_idx for item in gdn_states) != tuple(
            sorted(item.layer_idx for item in gdn_states)
        ) or len({item.layer_idx for item in gdn_states}) != len(gdn_states):
            raise ProofV2TraceError(
                "GDN state opening layers must be sorted and unique"
            )
        full_positions = tuple(
            (item.layer_idx, item.kv_head_idx) for item in full_attention_states
        )
        if full_positions != tuple(sorted(full_positions)) or len(full_positions) != len(
            set(full_positions)
        ):
            raise ProofV2TraceError(
                "full-attention state openings must be sorted and unique"
            )
        positions = tuple((item.token_index, item.layer_idx) for item in layers)
        if positions != tuple(sorted(positions)) or len(positions) != len(
            set(positions)
        ):
            raise ProofV2TraceError(
                "opened trace layer positions must be sorted and unique"
            )
        gdn_positions = tuple(
            (item.token_index, item.layer_idx) for item in gdn_witnesses
        )
        if gdn_positions != tuple(sorted(gdn_positions)) or len(gdn_positions) != len(
            set(gdn_positions)
        ):
            raise ProofV2TraceError(
                "GDN transition witnesses must be sorted and unique"
            )
        full_witness_positions = tuple(
            (item.token_index, item.layer_idx, item.query_head_idx)
            for item in full_attention_witnesses
        )
        if full_witness_positions != tuple(sorted(full_witness_positions)) or len(
            full_witness_positions
        ) != len(set(full_witness_positions)):
            raise ProofV2TraceError(
                "full-attention transition witnesses must be sorted and unique"
            )
        for item in layers:
            if item.token_index >= len(tokens):
                raise ProofV2TraceError("opened trace token position is out of range")
            token = tokens[item.token_index]
            if item.layer_idx >= len(token.layer_digests):
                raise ProofV2TraceError("opened trace layer position is out of range")
            if item.digest() != token.layer_digests[item.layer_idx]:
                raise ProofV2TraceError(
                    "opened layer does not match its committed token leaf"
                )
            residual_in = trace_residual_boundary_digest_v2(
                item.tensor("residual_in").values
            )
            residual_out = trace_residual_boundary_digest_v2(
                item.tensor("residual_out").values
            )
            if (
                residual_in != token.residual_boundary_digests[item.layer_idx]
                or residual_out != token.residual_boundary_digests[item.layer_idx + 1]
            ):
                raise ProofV2TraceError(
                    "opened layer does not match the committed residual chain"
                )
            before_names = attention_state_tensor_names_v2(
                item.attention_profile,
                before=True,
            )
            after_names = attention_state_tensor_names_v2(
                item.attention_profile,
                before=False,
            )
            before_digest = trace_attention_state_boundary_digest_v2(
                item.attention_profile,
                tuple(item.tensor(name).values for name in before_names),
            )
            after_digest = trace_attention_state_boundary_digest_v2(
                item.attention_profile,
                tuple(item.tensor(name).values for name in after_names),
            )
            if (
                before_digest != token.attention_state_before_digests[item.layer_idx]
                or after_digest != token.attention_state_after_digests[item.layer_idx]
            ):
                raise ProofV2TraceError(
                    "opened layer does not match the committed attention-state chain"
                )
        for item in gdn_witnesses:
            if (
                item.token_index >= len(tokens)
                or not tokens[item.token_index].transition_witness_roots
                or item.layer_idx
                >= len(tokens[item.token_index].transition_witness_roots)
                or item.digest()
                != tokens[item.token_index].transition_witness_roots[item.layer_idx]
            ):
                raise ProofV2TraceError(
                    "GDN transition witness does not match its token leaf"
                )
        for item in full_attention_witnesses:
            if (
                item.token_index >= len(tokens)
                or not tokens[item.token_index].transition_witness_roots
                or item.layer_idx
                >= len(tokens[item.token_index].transition_witness_roots)
                or not verify_merkle_path(
                    tokens[item.token_index].transition_witness_roots[item.layer_idx],
                    item.digest(),
                    item.merkle_path,
                )
            ):
                raise ProofV2TraceError(
                    "full-attention witness does not match its token leaf"
                )
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "opened_layers", layers)
        object.__setattr__(self, "gdn_initial_state_openings", gdn_states)
        object.__setattr__(
            self,
            "full_attention_state_openings",
            full_attention_states,
        )
        object.__setattr__(self, "gdn_transition_witnesses", gdn_witnesses)
        object.__setattr__(
            self,
            "full_attention_head_witnesses",
            full_attention_witnesses,
        )

    def canonical_bytes(self) -> bytes:
        has_transition_witnesses = bool(
            self.gdn_transition_witnesses or self.full_attention_head_witnesses
        )
        encoded = bytearray()
        if has_transition_witnesses:
            encoded.extend(
                struct.pack(
                    "<4sIIIIII",
                    _TRACE_PROOF_MAGIC_V4,
                    len(self.tokens),
                    len(self.opened_layers),
                    len(self.gdn_initial_state_openings),
                    len(self.full_attention_state_openings),
                    len(self.gdn_transition_witnesses),
                    len(self.full_attention_head_witnesses),
                )
            )
        else:
            encoded.extend(
                struct.pack(
                    "<4sIIII",
                    _TRACE_PROOF_MAGIC_V3,
                    len(self.tokens),
                    len(self.opened_layers),
                    len(self.gdn_initial_state_openings),
                    len(self.full_attention_state_openings),
                )
            )
        for token in self.tokens:
            item = token.canonical_bytes()
            encoded.extend(_u32(len(item), "token trace encoding length"))
            encoded.extend(item)
        for layer in self.opened_layers:
            item = layer.canonical_bytes()
            encoded.extend(_u32(len(item), "layer trace encoding length"))
            encoded.extend(item)
        for opening in self.gdn_initial_state_openings:
            item = opening.canonical_bytes()
            encoded.extend(_u32(len(item), "GDN state opening length"))
            encoded.extend(item)
        for opening in self.full_attention_state_openings:
            item = opening.canonical_bytes()
            encoded.extend(_u32(len(item), "full-attention state opening length"))
            encoded.extend(item)
        for witness in self.gdn_transition_witnesses:
            item = witness.canonical_bytes()
            encoded.extend(_u32(len(item), "GDN transition witness length"))
            encoded.extend(item)
        for witness in self.full_attention_head_witnesses:
            item = witness.canonical_bytes()
            encoded.extend(
                _u32(len(item), "full-attention transition witness length")
            )
            encoded.extend(item)
        if len(encoded) > MAX_TRACE_PROOF_BYTES:
            raise ProofV2TraceError("execution trace proof exceeds the protocol limit")
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ExecutionTraceProofV2":
        reader = _TraceReader(encoded, MAX_TRACE_PROOF_BYTES, "execution trace proof")
        magic = reader.read(4)
        if magic == _TRACE_PROOF_MAGIC_V3:
            (
                token_count,
                layer_count,
                gdn_state_count,
                full_attention_state_count,
            ) = reader.unpack("<IIII")
            gdn_witness_count = 0
            full_attention_witness_count = 0
        elif magic == _TRACE_PROOF_MAGIC_V4:
            (
                token_count,
                layer_count,
                gdn_state_count,
                full_attention_state_count,
                gdn_witness_count,
                full_attention_witness_count,
            ) = reader.unpack("<IIIIII")
        else:
            raise ProofV2TraceError("execution trace proof header is not supported")
        if not 0 < token_count <= MAX_TRACE_TOKENS:
            raise ProofV2TraceError("trace proof token count is out of range")
        if layer_count > MAX_TRACE_OPENED_LAYERS:
            raise ProofV2TraceError("opened trace layer count is out of range")
        if gdn_state_count > MAX_TRACE_OPENED_LAYERS:
            raise ProofV2TraceError("GDN state opening count is out of range")
        if full_attention_state_count > MAX_TRACE_OPENED_LAYERS:
            raise ProofV2TraceError(
                "full-attention state opening count is out of range"
            )
        if (
            gdn_witness_count > MAX_TRACE_OPENED_LAYERS
            or full_attention_witness_count > MAX_TRACE_OPENED_LAYERS
        ):
            raise ProofV2TraceError("transition witness count is out of range")
        tokens = []
        for _ in range(token_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > (1 << 20):
                raise ProofV2TraceError("token trace length is out of range")
            tokens.append(
                TokenExecutionTraceV2.from_canonical_bytes(reader.read(length))
            )
        layers = []
        for _ in range(layer_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > MAX_TRACE_LAYER_BYTES:
                raise ProofV2TraceError("layer trace length is out of range")
            layers.append(
                LayerExecutionTraceV2.from_canonical_bytes(reader.read(length))
            )
        gdn_states = []
        for _ in range(gdn_state_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > 2 * MAX_GDN_STATE_BYTES + 22:
                raise ProofV2TraceError("GDN state opening length is out of range")
            gdn_states.append(
                GDNInitialStateOpeningV2.from_canonical_bytes(reader.read(length))
            )
        full_attention_states = []
        for _ in range(full_attention_state_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > (2 * MAX_FULL_ATTENTION_HEAD_STATE_BYTES):
                raise ProofV2TraceError(
                    "full-attention state opening length is out of range"
                )
            full_attention_states.append(
                FullAttentionHeadStateOpeningV2.from_canonical_bytes(
                    reader.read(length)
                )
            )
        gdn_witnesses = []
        for _ in range(gdn_witness_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > 3 * MAX_TRACE_TENSOR_BYTES + 148:
                raise ProofV2TraceError("GDN transition witness length is out of range")
            gdn_witnesses.append(
                GDNTransitionWitnessV2.from_canonical_bytes(reader.read(length))
            )
        full_attention_witnesses = []
        for _ in range(full_attention_witness_count):
            length = reader.unpack("<I")[0]
            if length == 0 or length > MAX_TRACE_TENSOR_BYTES:
                raise ProofV2TraceError(
                    "full-attention transition witness length is out of range"
                )
            full_attention_witnesses.append(
                FullAttentionHeadWitnessV2.from_canonical_bytes(reader.read(length))
            )
        reader.finish()
        result = cls(
            tuple(tokens),
            tuple(layers),
            tuple(gdn_states),
            tuple(full_attention_states),
            tuple(gdn_witnesses),
            tuple(full_attention_witnesses),
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TraceError("execution trace proof is not canonical")
        return result

    def verify(
        self,
        commitment: ExecutionTraceCommitmentV2,
        *,
        output_token_ids: Sequence[int],
        expected_layer_positions: Iterable[tuple[int, int]],
        expected_first_input_token_id: int,
    ) -> None:
        rebuilt, _tree = build_execution_trace_commitment_v2(
            self.tokens,
            profile=commitment.profile,
        )
        if rebuilt != commitment:
            raise ProofV2TraceError(
                "execution trace proof does not match its pre-challenge root"
            )
        expected_outputs = tuple(int(value) for value in output_token_ids)
        if tuple(item.output_token_id for item in self.tokens) != expected_outputs:
            raise ProofV2TraceError(
                "execution trace output-token chain does not match the response"
            )
        if self.tokens[0].input_token_id != expected_first_input_token_id:
            raise ProofV2TraceError(
                "execution trace does not start at the committed prompt boundary"
            )
        expected_positions = tuple(sorted(set(expected_layer_positions)))
        actual_positions = tuple(
            (item.token_index, item.layer_idx) for item in self.opened_layers
        )
        if actual_positions != expected_positions:
            raise ProofV2TraceError("execution trace layer opening set is not exact")


__all__ = [
    "ExecutionTraceCommitmentV2",
    "ExecutionTraceProofV2",
    "FullAttentionHeadStateOpeningV2",
    "GDN_DECODE_SUFFIX_TOKEN_START_V1",
    "GDNInitialStateOpeningV2",
    "LayerExecutionTraceV2",
    "ProofV2TraceError",
    "TRACE_ATTENTION_FULL_AUDIT_ONLY",
    "TRACE_ATTENTION_FULL_TRANSITION_V1",
    "TRACE_ATTENTION_GDN_AUDIT_ONLY",
    "TRACE_ATTENTION_GDN_TRANSITION_V1",
    "TRACE_PROFILE_QWEN_HYBRID_DENSE_V1",
    "TRACE_PROFILE_QWEN_HYBRID_SPARSE_MOE_SHELL_V1",
    "TokenExecutionTraceV2",
    "TokenTraceOpeningV2",
    "TraceTensorV2",
    "attention_state_tensor_names_v2",
    "build_execution_trace_commitment_v2",
    "build_full_attention_state_root_v2",
    "full_attention_head_state_leaf_v2",
    "gdn_state_digest_v2",
    "required_layer_fields_v2",
    "trace_attention_state_boundary_digest_v2",
    "trace_residual_boundary_digest_v2",
    "trace_tail_digest_v2",
    "validate_cross_token_state_continuity_v2",
    "validate_layer_trace_set_v2",
]
