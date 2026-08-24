"""Canonical signed relation schema required by a proof-v3 hard profile.

The schema is intentionally declarative.  It does not verify a computation by
itself; a qualified adapter must implement every signed relation ABI before it
can be registered.  Its job is to make the statement complete, canonical, and
unambiguous before a verifier sees opaque proof bytes.
"""

from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError


RELATION_SPEC_FORMAT_VERSION = 6
# The enclosing signed profile document is capped at 1 MiB too. Reserve space
# for its profile metadata, hexadecimal signatures, and JSON framing.
MAX_RELATION_SPEC_BYTES = 960 << 10
MAX_IDENTIFIER_BYTES = 128
MAX_DECLARED_DIMENSIONS = 128
MAX_TENSORS = 65_535
MAX_RELATION_NODES = 65_535
MAX_OPERATION_REFERENCES = 65_535
MAX_STATIC_BINDINGS = 65_535
MAX_STATIC_TABLE_BINDINGS = 65_535
MAX_TOLERANCE_PROFILES = 256
MAX_TENSOR_RANK = 8
MAX_NODE_TENSORS = 32
GLOBAL_EXECUTION_RELATION_ABI_V3 = "global_execution.air.v2"
GLOBAL_EXECUTION_COVERAGE_MODE_V3 = "all_graph_nodes"
ECONOMIC_SHELL_COVERAGE_MODE_V3 = "economic_shell_only"
POSTCOMMIT_GLOBAL_NONCE_ABI_V3 = "postcommit.global_fold.v1"
POSTCOMMIT_AUDIT_TIER_ABI_V3 = "postcommit.audit_tier.v1"
STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3 = "stratified.layer_head.v1"
RECURSIVE_CHUNK_ACCUMULATOR_ABI_V3 = "recursive.chunk.v2"
SEQUENCE_DOMAIN_RELATION_ID_V3 = "context_plus_decode.v1"
# Version one named only K/V pages.  Version two is deliberately generic so a
# profile can describe both paged-attention tables and recurrent GDN state
# tables without smuggling one family through the other's semantics.
LOGICAL_KV_ADDRESS_ABI_V3 = "logical_kv.address.v1"
LOGICAL_CACHE_ADDRESS_ABI_V3 = "logical_cache.address.v2"
LOGICAL_CACHE_TABLES_SEMANTICS_V3 = "logical_cache_tables.v2"
PAGED_KV_CACHE_SEMANTICS_V3 = "paged_kv.logical.v1"
GDN_STATE_CACHE_SEMANTICS_V3 = "gdn.recurrent_state.v1"
GPU_RETENTION_LEASE_ABI_V3 = "gpu.retention_lease.v1"
PREFIX_CACHE_LOGICAL_ADDRESS_ABI_V3 = "logical_cache.address.v3.prefix_blocks"
PREFIX_CACHE_TABLES_SEMANTICS_V3 = (
    "logical_cache_tables.v4.prefix_blocks.cross_geometry_corridor"
)
PREFIX_CACHE_RETENTION_LEASE_ABI_V3 = "gpu.retention_lease.v2.prefix_blocks"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_COMMITMENT_ROLES = frozenset(
    {
        "request_input",
        "runtime_state",
        "cache",
        "final_hidden",
        "logits",
        "token_output",
    }
)
_STATIC_BINDING_KINDS = frozenset(
    {
        "attention",
        "bridge",
        "embedding",
        "final_norm",
        "model_boundary",
        "moe",
        "rope",
        "sampler",
        "transition",
    }
)
_STATIC_TABLE_SUBJECT_KINDS = frozenset({"operation", "static_parameter"})
_EXACT_COMPARATORS = frozenset({"exact.field.v1", "exact.integer.v1"})
_BOUNDED_COMPARATORS = frozenset({"bounded.fixed_point.v1"})
_RELATION_ADAPTERS = {
    "embedding": frozenset({"embedding.v1"}),
    "linear": frozenset({"global.linear.v1"}),
    "full_attention": frozenset({"full_attention.v1"}),
    "gdn": frozenset({"gdn.v1"}),
    "bridge": frozenset({"residual.bridge.v1"}),
    "moe": frozenset({"fused_moe.qwen3_5.v1"}),
    "final_norm": frozenset({"final_norm.v1"}),
    "sampler": frozenset({"canonical.sampler.v1"}),
}
_LAYER_RELATIONS = frozenset({"full_attention", "gdn", "bridge", "moe"})
_NODE_BINDING_KINDS = {
    "embedding": frozenset({"embedding", "model_boundary"}),
    "full_attention": frozenset({"attention", "rope", "transition"}),
    "gdn": frozenset({"bridge", "transition"}),
    "bridge": frozenset({"bridge"}),
    "moe": frozenset({"moe"}),
    "final_norm": frozenset({"final_norm", "model_boundary"}),
    "sampler": frozenset({"sampler"}),
}
_CACHE_FAMILY_KINDS = frozenset({"attention_kv", "gdn_state"})
_TRANSITION_KINDS = frozenset({"full_attention", "gdn"})
GLOBAL_HARD_AUDIT_MIN_LAYERS_V3 = 4
GLOBAL_HARD_AUDIT_MIN_FULL_ATTENTION_LAYERS_V3 = 2
GLOBAL_HARD_AUDIT_MIN_GDN_LAYERS_V3 = 1
GLOBAL_HARD_AUDIT_MIN_HEADS_PER_FULL_ATTENTION_LAYER_V3 = 2
MAX_HARD_AUDIT_QUERY_ROWS_V3 = 256


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} must be ASCII") from exc
    if len(encoded) > MAX_IDENTIFIER_BYTES or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(
            f"{name} must be a lowercase canonical identifier no longer than "
            f"{MAX_IDENTIFIER_BYTES} bytes"
        )
    return value


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 64:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 64-bit integer")
    return value


def _i32(value: object, name: str, *, minimum: int = -(1 << 31)) -> int:
    if type(value) is not int or not minimum <= value < 1 << 31:
        raise ProofV3Error(f"{name} must be a signed 32-bit integer")
    return value


def _tuple(value: object, name: str, maximum: int) -> tuple[object, ...]:
    if not isinstance(value, tuple) or len(value) > maximum:
        raise ProofV3Error(f"{name} must be a bounded tuple")
    return value


def _strictly_sorted(
    values: Sequence[object], keys: Sequence[object], name: str
) -> None:
    if len(values) != len(keys) or tuple(keys) != tuple(sorted(keys)):
        raise ProofV3Error(f"{name} must be sorted canonically")
    if len(set(keys)) != len(keys):
        raise ProofV3Error(f"{name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class DeclaredDimensionV3:
    """One closed symbolic dimension usable by signed tensor descriptors."""

    dimension_id: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _identifier(self.dimension_id, "dimension_id")
        minimum = _u32(self.minimum, "dimension minimum")
        maximum = _u32(self.maximum, "dimension maximum", positive=True)
        if minimum > maximum:
            raise ProofV3Error("dimension minimum exceeds its maximum")

    @property
    def exact_value(self) -> int | None:
        """Return a static width only when the signed range is a singleton."""

        return self.minimum if self.minimum == self.maximum else None

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension_id": self.dimension_id,
            "maximum": self.maximum,
            "minimum": self.minimum,
        }


@dataclass(frozen=True, slots=True)
class SequenceDomainV3:
    """The exact combined token domain consumed by decode-time cache access.

    A long-context proof must not treat prompt and decode cache rows as
    unrelated bounded dimensions. This signed affine rule defines the only
    allowed logical sequence coordinate: ``context_tokens + decode_tokens``.
    The verifier binds its concrete operands from the validator-owned request
    and observed output counts before any adapter sees proof bytes.
    """

    context_dimension_id: str
    decode_dimension_id: str
    sequence_dimension_id: str
    relation_id: str = SEQUENCE_DOMAIN_RELATION_ID_V3

    def __post_init__(self) -> None:
        _identifier(self.context_dimension_id, "context_dimension_id")
        _identifier(self.decode_dimension_id, "decode_dimension_id")
        _identifier(self.sequence_dimension_id, "sequence_dimension_id")
        if (
            len(
                {
                    self.context_dimension_id,
                    self.decode_dimension_id,
                    self.sequence_dimension_id,
                }
            )
            != 3
        ):
            raise ProofV3Error("sequence-domain dimensions must be distinct")
        if self.relation_id != SEQUENCE_DOMAIN_RELATION_ID_V3:
            raise ProofV3Error("sequence-domain relation_id is not supported")

    def validate_dimensions(
        self,
        dimensions: Mapping[str, DeclaredDimensionV3],
    ) -> None:
        """Require an exact signed range for the affine token domain."""

        required = (
            self.context_dimension_id,
            self.decode_dimension_id,
            self.sequence_dimension_id,
        )
        if any(dimension_id not in dimensions for dimension_id in required):
            raise ProofV3Error("sequence-domain dimensions are not declared")
        context = dimensions[self.context_dimension_id]
        decode = dimensions[self.decode_dimension_id]
        sequence = dimensions[self.sequence_dimension_id]
        if context.minimum < 1:
            raise ProofV3Error("context token domain must exclude an empty prompt")
        if sequence.minimum != context.minimum + decode.minimum:
            raise ProofV3Error(
                "sequence token minimum does not match context plus decode"
            )
        if sequence.maximum != context.maximum + decode.maximum:
            raise ProofV3Error(
                "sequence token maximum does not match context plus decode"
            )

    def bind_counts(
        self,
        *,
        dimensions: Mapping[str, DeclaredDimensionV3],
        context_token_count: int,
        decode_token_count: int,
    ) -> int:
        """Validate one concrete request/output count pair and return its sum."""

        self.validate_dimensions(dimensions)
        context_count = _u32(
            context_token_count,
            "context_token_count",
            positive=True,
        )
        decode_count = _u32(decode_token_count, "decode_token_count")
        context = dimensions[self.context_dimension_id]
        decode = dimensions[self.decode_dimension_id]
        sequence = dimensions[self.sequence_dimension_id]
        if not context.minimum <= context_count <= context.maximum:
            raise ProofV3Error("context token count is outside the signed domain")
        if not decode.minimum <= decode_count <= decode.maximum:
            raise ProofV3Error("decode token count is outside the signed domain")
        sequence_count = context_count + decode_count
        if sequence_count >= 1 << 32:
            raise ProofV3Error("combined token coordinate space overflows")
        if not sequence.minimum <= sequence_count <= sequence.maximum:
            raise ProofV3Error("sequence token count is outside the signed domain")
        return sequence_count

    def to_dict(self) -> dict[str, object]:
        return {
            "context_dimension_id": self.context_dimension_id,
            "decode_dimension_id": self.decode_dimension_id,
            "relation_id": self.relation_id,
            "sequence_dimension_id": self.sequence_dimension_id,
        }


ShapeDimensionV3 = int | str


@dataclass(frozen=True, slots=True)
class TensorDescriptorV3:
    """One committed runtime tensor ABI with a closed concrete/symbolic shape."""

    tensor_id: str
    shape: tuple[ShapeDimensionV3, ...]
    encoding_id: str
    layout_id: str
    scale_rule_id: str
    commitment_role: str

    def __post_init__(self) -> None:
        _identifier(self.tensor_id, "tensor_id")
        shape = _tuple(self.shape, "tensor shape", MAX_TENSOR_RANK)
        if not shape:
            raise ProofV3Error("tensor shape must not be empty")
        for index, dimension in enumerate(shape):
            if type(dimension) is int:
                _u32(dimension, f"tensor shape[{index}]", positive=True)
            elif isinstance(dimension, str):
                _identifier(dimension, f"tensor shape[{index}]")
            else:
                raise ProofV3Error(
                    f"tensor shape[{index}] must be a positive integer or dimension id"
                )
        _identifier(self.encoding_id, "encoding_id")
        _identifier(self.layout_id, "layout_id")
        _identifier(self.scale_rule_id, "scale_rule_id")
        if self.commitment_role not in _COMMITMENT_ROLES:
            raise ProofV3Error("tensor commitment_role is not supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "commitment_role": self.commitment_role,
            "encoding_id": self.encoding_id,
            "layout_id": self.layout_id,
            "scale_rule_id": self.scale_rule_id,
            "shape": list(self.shape),
            "tensor_id": self.tensor_id,
        }


@dataclass(frozen=True, slots=True)
class RegisteredOperationReferenceV3:
    """Exact identity of one PCS operation in the authenticated static manifest."""

    layer_index: int
    operation_id: str
    expert_id: int | None = None
    rows: int = 0
    cols: int = 0
    weight_commitment: bytes = b""
    descriptor_digest: bytes = b""

    def __post_init__(self) -> None:
        _i32(self.layer_index, "operation layer_index", minimum=-1)
        _identifier(self.operation_id, "operation_id")
        if self.expert_id is not None:
            _u32(self.expert_id, "operation expert_id")
        _u32(self.rows, "operation rows", positive=True)
        _u32(self.cols, "operation cols", positive=True)
        _fixed32(self.weight_commitment, "operation weight_commitment", nonzero=True)
        _fixed32(self.descriptor_digest, "operation descriptor_digest", nonzero=True)

    def sort_key(self) -> tuple[int, str, int]:
        return (
            self.layer_index,
            self.operation_id,
            -1 if self.expert_id is None else self.expert_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cols": self.cols,
            "descriptor_digest": self.descriptor_digest.hex(),
            "expert_id": self.expert_id,
            "layer_index": self.layer_index,
            "operation_id": self.operation_id,
            "rows": self.rows,
            "weight_commitment": self.weight_commitment.hex(),
        }


@dataclass(frozen=True, slots=True)
class StaticParameterBindingV3:
    """One signed non-PCS static parameter family consumed by graph relations."""

    binding_id: str
    binding_kind: str
    binding_digest: bytes
    layer_index: int = -1

    def __post_init__(self) -> None:
        _identifier(self.binding_id, "binding_id")
        if self.binding_kind not in _STATIC_BINDING_KINDS:
            raise ProofV3Error("static binding_kind is not supported")
        _fixed32(self.binding_digest, "binding_digest", nonzero=True)
        layer = _i32(self.layer_index, "binding layer_index", minimum=-1)
        if self.binding_kind in {
            "attention",
            "bridge",
            "moe",
            "rope",
            "transition",
        }:
            if layer < 0:
                raise ProofV3Error("layer transition bindings require a layer index")
        elif layer != -1:
            raise ProofV3Error("model-bound static bindings must use layer -1")

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_digest": self.binding_digest.hex(),
            "binding_id": self.binding_id,
            "binding_kind": self.binding_kind,
            "layer_index": self.layer_index,
        }


@dataclass(frozen=True, slots=True)
class StaticTableBindingV3:
    """Exact signed use of one dynamic static-table descriptor.

    The V2 PCS operation catalog and the Goldilocks fixed-table catalog are
    separate commitments. A future native adapter must not choose a table by a
    local label. This record joins every table to exactly one signed operation
    or static parameter, and repeats its expected range/encoding so the loaded
    table layout is checked rather than merely hash-matched.
    """

    binding_id: str
    subject_kind: str
    operation_descriptor_digest: bytes | None
    static_binding_id: str | None
    static_binding_digest: bytes | None
    table_id: str
    static_table_descriptor_digest: bytes
    logical_leaf_start: int
    logical_leaf_count: int
    page_start: int
    page_count: int
    element_encoding_id: str
    scale_encoding_id: str

    def __post_init__(self) -> None:
        _identifier(self.binding_id, "static table binding_id")
        if self.subject_kind not in _STATIC_TABLE_SUBJECT_KINDS:
            raise ProofV3Error("static table subject_kind is not supported")
        _identifier(self.table_id, "static table binding table_id")
        _fixed32(
            self.static_table_descriptor_digest,
            "static table descriptor_digest",
            nonzero=True,
        )
        _u64(self.logical_leaf_start, "static table logical_leaf_start")
        _u64(
            self.logical_leaf_count,
            "static table logical_leaf_count",
            positive=True,
        )
        _u64(self.page_start, "static table page_start")
        _u64(self.page_count, "static table page_count", positive=True)
        _identifier(self.element_encoding_id, "static table element_encoding_id")
        _identifier(self.scale_encoding_id, "static table scale_encoding_id")
        if self.subject_kind == "operation":
            _fixed32(
                self.operation_descriptor_digest,
                "static table operation_descriptor_digest",
                nonzero=True,
            )
            if (
                self.static_binding_id is not None
                or self.static_binding_digest is not None
            ):
                raise ProofV3Error(
                    "operation static table bindings must not carry static parameter fields"
                )
        else:
            if self.operation_descriptor_digest is not None:
                raise ProofV3Error(
                    "static parameter table bindings must not carry an operation digest"
                )
            _identifier(self.static_binding_id, "static table static_binding_id")
            _fixed32(
                self.static_binding_digest,
                "static table static_binding_digest",
                nonzero=True,
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "element_encoding_id": self.element_encoding_id,
            "logical_leaf_count": self.logical_leaf_count,
            "logical_leaf_start": self.logical_leaf_start,
            "operation_descriptor_digest": (
                None
                if self.operation_descriptor_digest is None
                else self.operation_descriptor_digest.hex()
            ),
            "page_count": self.page_count,
            "page_start": self.page_start,
            "scale_encoding_id": self.scale_encoding_id,
            "static_binding_digest": (
                None
                if self.static_binding_digest is None
                else self.static_binding_digest.hex()
            ),
            "static_binding_id": self.static_binding_id,
            "static_table_descriptor_digest": self.static_table_descriptor_digest.hex(),
            "subject_kind": self.subject_kind,
            "table_id": self.table_id,
        }


@dataclass(frozen=True, slots=True)
class RelationToleranceV3:
    """One named, signed comparator/tolerance rule used by graph nodes."""

    tolerance_id: str
    comparator_id: str
    absolute_q32: int
    relative_bps: int

    def __post_init__(self) -> None:
        _identifier(self.tolerance_id, "tolerance_id")
        if self.comparator_id not in _EXACT_COMPARATORS | _BOUNDED_COMPARATORS:
            raise ProofV3Error("tolerance comparator_id is not supported")
        _u32(self.absolute_q32, "absolute_q32")
        _u32(self.relative_bps, "relative_bps")
        if self.relative_bps > 10_000:
            raise ProofV3Error("relative_bps must not exceed 10000")
        if self.comparator_id in _EXACT_COMPARATORS and (
            self.absolute_q32 != 0 or self.relative_bps != 0
        ):
            raise ProofV3Error("exact tolerance comparators must have zero tolerance")

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_q32": self.absolute_q32,
            "comparator_id": self.comparator_id,
            "relative_bps": self.relative_bps,
            "tolerance_id": self.tolerance_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionRelationNodeV3:
    """One ordered global-relation node in a signed execution graph."""

    node_id: str
    relation_id: str
    transition_adapter_id: str
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    tolerance_id: str
    static_binding_ids: tuple[str, ...] = ()
    operation_reference: RegisteredOperationReferenceV3 | None = None
    layer_index: int = -1

    def __post_init__(self) -> None:
        _identifier(self.node_id, "node_id")
        _identifier(self.relation_id, "relation_id")
        _identifier(self.transition_adapter_id, "transition_adapter_id")
        if self.relation_id not in _RELATION_ADAPTERS:
            raise ProofV3Error("relation_id is not supported by relation format v3")
        if self.transition_adapter_id not in _RELATION_ADAPTERS[self.relation_id]:
            raise ProofV3Error("transition_adapter_id is not valid for relation_id")
        _i32(self.layer_index, "node layer_index", minimum=-1)
        if self.relation_id in _LAYER_RELATIONS and self.layer_index < 0:
            raise ProofV3Error("layer relation nodes require a nonnegative layer index")
        if self.relation_id in {"embedding", "final_norm", "sampler"} and (
            self.layer_index != -1
        ):
            raise ProofV3Error("model-bound relation nodes must use layer -1")
        inputs = _tuple(
            self.input_tensor_ids, "node input_tensor_ids", MAX_NODE_TENSORS
        )
        outputs = _tuple(
            self.output_tensor_ids,
            "node output_tensor_ids",
            MAX_NODE_TENSORS,
        )
        if not inputs or not outputs:
            raise ProofV3Error("relation nodes require input and output tensors")
        for index, tensor_id in enumerate(inputs):
            _identifier(tensor_id, f"node input_tensor_ids[{index}]")
        for index, tensor_id in enumerate(outputs):
            _identifier(tensor_id, f"node output_tensor_ids[{index}]")
        if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
            raise ProofV3Error("relation node tensor lists must not contain duplicates")
        _identifier(self.tolerance_id, "node tolerance_id")
        binding_ids = _tuple(
            self.static_binding_ids,
            "node static_binding_ids",
            MAX_NODE_TENSORS,
        )
        for index, binding_id in enumerate(binding_ids):
            _identifier(binding_id, f"node static_binding_ids[{index}]")
        if len(set(binding_ids)) != len(binding_ids):
            raise ProofV3Error(
                "relation node static bindings must not contain duplicates"
            )
        if self.relation_id == "linear":
            if not isinstance(self.operation_reference, RegisteredOperationReferenceV3):
                raise ProofV3Error(
                    "linear relation nodes require an operation reference"
                )
            if self.operation_reference.layer_index != self.layer_index:
                raise ProofV3Error(
                    "linear relation layer does not match its PCS operation"
                )
        elif self.operation_reference is not None:
            raise ProofV3Error(
                "only linear relation nodes may reference PCS operations"
            )
        if self.relation_id != "linear" and not binding_ids:
            raise ProofV3Error(
                "nonlinear relation nodes require static parameter bindings"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tensor_ids": list(self.input_tensor_ids),
            "layer_index": self.layer_index,
            "node_id": self.node_id,
            "operation_reference": (
                None
                if self.operation_reference is None
                else self.operation_reference.to_dict()
            ),
            "output_tensor_ids": list(self.output_tensor_ids),
            "relation_id": self.relation_id,
            "static_binding_ids": list(self.static_binding_ids),
            "tolerance_id": self.tolerance_id,
            "transition_adapter_id": self.transition_adapter_id,
        }


@dataclass(frozen=True, slots=True)
class LayerCacheRelationV3:
    """One layer-local logical cache family governed by a RAM relation.

    The tensor identifiers are logical tables, never physical vLLM cache
    pointers. A qualified adapter must prove their versioned read/write
    relation, but the signed profile already prevents one shared K/V table from
    standing in for every transformer layer.
    """

    layer_index: int
    cache_kind: str
    cache_semantics_id: str
    key_tensor_id: str | None = None
    value_tensor_id: str | None = None
    state_tensor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _i32(self.layer_index, "cache layer_index", minimum=0)
        if self.cache_kind not in _CACHE_FAMILY_KINDS:
            raise ProofV3Error("cache family kind is not supported")
        _identifier(self.cache_semantics_id, "cache_semantics_id")
        if self.cache_kind == "attention_kv":
            if self.cache_semantics_id != PAGED_KV_CACHE_SEMANTICS_V3:
                raise ProofV3Error(
                    "attention K/V cache family has an unsupported semantics ABI"
                )
            _identifier(self.key_tensor_id, "cache key_tensor_id")
            _identifier(self.value_tensor_id, "cache value_tensor_id")
            if self.key_tensor_id == self.value_tensor_id:
                raise ProofV3Error("cache key and value tensors must differ")
        else:
            if self.cache_semantics_id != GDN_STATE_CACHE_SEMANTICS_V3:
                raise ProofV3Error("GDN cache family has an unsupported semantics ABI")
            if self.key_tensor_id is not None or self.value_tensor_id is not None:
                raise ProofV3Error("GDN cache families must not declare K/V tensors")
        states = _tuple(
            self.state_tensor_ids, "cache state_tensor_ids", MAX_NODE_TENSORS
        )
        for index, tensor_id in enumerate(states):
            _identifier(tensor_id, f"cache state_tensor_ids[{index}]")
        if len(set(states)) != len(states):
            raise ProofV3Error("cache state tensors must not contain duplicates")
        if (self.key_tensor_id is not None and self.key_tensor_id in states) or (
            self.value_tensor_id is not None and self.value_tensor_id in states
        ):
            raise ProofV3Error("cache state tensors must not alias key or value")
        if self.cache_kind == "gdn_state" and not states:
            raise ProofV3Error("GDN cache families require recurrent state tensors")

    def tensor_ids(self) -> tuple[str, ...]:
        """Return all logical tables belonging to this layer-local family."""

        return (
            *(() if self.key_tensor_id is None else (self.key_tensor_id,)),
            *(() if self.value_tensor_id is None else (self.value_tensor_id,)),
            *self.state_tensor_ids,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_kind": self.cache_kind,
            "cache_semantics_id": self.cache_semantics_id,
            "key_tensor_id": self.key_tensor_id,
            "layer_index": self.layer_index,
            "state_tensor_ids": list(self.state_tensor_ids),
            "value_tensor_id": self.value_tensor_id,
        }


@dataclass(frozen=True, slots=True)
class CacheRelationSpecV3:
    """Qualified logical-table ABI with a distinct family for every layer.

    The outer collection ABI only fixes the address/lease/table inventory
    mechanics. Each :class:`LayerCacheRelationV3` separately fixes the
    attention or recurrent-state semantics that a native RAM adapter must
    prove.
    """

    logical_address_abi_id: str
    cache_semantics_id: str
    retention_lease_abi_id: str
    page_token_count: int
    max_cache_tokens: int
    layer_caches: tuple[LayerCacheRelationV3, ...]
    allows_sliding_window: bool = False
    allows_prefix_cache_sharing: bool = False
    allows_speculative_decode: bool = False
    prefix_cache_k_cell_delta_max: int = 0
    prefix_cache_k_row_sq_delta_max: int = 0
    prefix_cache_v_cell_delta_max: int = 0
    prefix_cache_v_row_sq_delta_max: int = 0

    def __post_init__(self) -> None:
        if type(self.allows_prefix_cache_sharing) is not bool:
            raise ProofV3Error("allows_prefix_cache_sharing must be a boolean")
        if self.allows_prefix_cache_sharing:
            expected_address = PREFIX_CACHE_LOGICAL_ADDRESS_ABI_V3
            expected_semantics = PREFIX_CACHE_TABLES_SEMANTICS_V3
            expected_lease = PREFIX_CACHE_RETENTION_LEASE_ABI_V3
            mode = "prefix-cache"
        else:
            expected_address = LOGICAL_CACHE_ADDRESS_ABI_V3
            expected_semantics = LOGICAL_CACHE_TABLES_SEMANTICS_V3
            expected_lease = GPU_RETENTION_LEASE_ABI_V3
            mode = "logical cache"
        if self.logical_address_abi_id != expected_address:
            raise ProofV3Error(f"{mode} address ABI is not supported")
        if self.cache_semantics_id != expected_semantics:
            raise ProofV3Error(f"{mode} table semantics are not supported")
        if self.retention_lease_abi_id != expected_lease:
            raise ProofV3Error(f"{mode} retention lease ABI is not supported")
        _u32(self.page_token_count, "page_token_count", positive=True)
        _u32(self.max_cache_tokens, "max_cache_tokens", positive=True)
        families = _tuple(self.layer_caches, "layer_caches", MAX_RELATION_NODES)
        if not families or not all(
            isinstance(item, LayerCacheRelationV3) for item in families
        ):
            raise ProofV3Error("cache relation requires layer-local cache families")
        _strictly_sorted(
            families,
            tuple(item.layer_index for item in families),
            "layer-local cache families",
        )
        all_tensor_ids = tuple(
            tensor_id for family in families for tensor_id in family.tensor_ids()
        )
        if len(all_tensor_ids) != len(set(all_tensor_ids)):
            raise ProofV3Error("layer-local cache families must not share tensors")
        if self.page_token_count > self.max_cache_tokens:
            raise ProofV3Error("cache page size exceeds the declared cache capacity")
        cache_bounds = (
            (
                self.prefix_cache_k_cell_delta_max,
                "prefix_cache_k_cell_delta_max",
                8_190,
            ),
            (
                self.prefix_cache_v_cell_delta_max,
                "prefix_cache_v_cell_delta_max",
                254,
            ),
        )
        for value, name, maximum in cache_bounds:
            _u32(value, name, positive=self.allows_prefix_cache_sharing)
            if value > maximum:
                raise ProofV3Error(f"{name} exceeds its physical quantized range")
        for value, name in (
            (
                self.prefix_cache_k_row_sq_delta_max,
                "prefix_cache_k_row_sq_delta_max",
            ),
            (
                self.prefix_cache_v_row_sq_delta_max,
                "prefix_cache_v_row_sq_delta_max",
            ),
        ):
            _u64(value, name, positive=self.allows_prefix_cache_sharing)
        if self.allows_prefix_cache_sharing:
            if (
                self.prefix_cache_k_row_sq_delta_max
                < self.prefix_cache_k_cell_delta_max**2
                or self.prefix_cache_v_row_sq_delta_max
                < self.prefix_cache_v_cell_delta_max**2
            ):
                raise ProofV3Error(
                    "prefix-cache row corridor cannot be smaller than its cell corridor"
                )
        elif any(
            (
                self.prefix_cache_k_cell_delta_max,
                self.prefix_cache_k_row_sq_delta_max,
                self.prefix_cache_v_cell_delta_max,
                self.prefix_cache_v_row_sq_delta_max,
            )
        ):
            raise ProofV3Error(
                "prefix-cache replay corridors require prefix-cache sharing"
            )
        for flag, name, display_name in (
            (self.allows_sliding_window, "allows_sliding_window", "sliding-window"),
            (
                self.allows_speculative_decode,
                "allows_speculative_decode",
                "speculative decode",
            ),
        ):
            if type(flag) is not bool:
                raise ProofV3Error(f"{name} must be a boolean")
            if flag:
                raise ProofV3Error(
                    f"{display_name} cache profiles are not yet qualified"
                )

    def by_layer(self) -> dict[int, LayerCacheRelationV3]:
        """Return the canonical layer-to-family mapping after validation."""

        return {family.layer_index: family for family in self.layer_caches}

    def to_dict(self) -> dict[str, object]:
        result = {
            "allows_prefix_cache_sharing": self.allows_prefix_cache_sharing,
            "allows_sliding_window": self.allows_sliding_window,
            "allows_speculative_decode": self.allows_speculative_decode,
            "cache_semantics_id": self.cache_semantics_id,
            "layer_caches": [item.to_dict() for item in self.layer_caches],
            "logical_address_abi_id": self.logical_address_abi_id,
            "max_cache_tokens": self.max_cache_tokens,
            "page_token_count": self.page_token_count,
            "retention_lease_abi_id": self.retention_lease_abi_id,
        }
        if self.allows_prefix_cache_sharing:
            result.update(
                {
                    "prefix_cache_k_cell_delta_max": self.prefix_cache_k_cell_delta_max,
                    "prefix_cache_k_row_sq_delta_max": self.prefix_cache_k_row_sq_delta_max,
                    "prefix_cache_v_cell_delta_max": self.prefix_cache_v_cell_delta_max,
                    "prefix_cache_v_row_sq_delta_max": self.prefix_cache_v_row_sq_delta_max,
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class LayerAuditPlanV3:
    """Signed, complete audit contract for one selectable transformer layer.

    A selected layer is not merely a layer number.  The plan pins the exact
    transition and bridge graph nodes and the complete PCS operation set the
    native adapter must prove for that layer.  Full-attention plans also pin
    the logical query/KV geometry used for post-commitment head selection.
    The model-specific semantics (RoPE, GQA, causal mask, paged-KV mapping)
    remain in the authenticated constraint-system artifact selected by
    ``attention_semantics_id``; an unknown adapter must fail closed.
    """

    layer_index: int
    transition_node_id: str
    bridge_node_id: str
    required_operation_references: tuple[RegisteredOperationReferenceV3, ...]
    attention_query_head_count: int = 0
    attention_key_value_head_count: int = 0
    attention_head_dimension: int = 0
    attention_semantics_id: str | None = None

    def __post_init__(self) -> None:
        _i32(self.layer_index, "layer audit layer_index", minimum=0)
        _identifier(self.transition_node_id, "layer audit transition_node_id")
        _identifier(self.bridge_node_id, "layer audit bridge_node_id")
        if self.transition_node_id == self.bridge_node_id:
            raise ProofV3Error("layer audit transition and bridge nodes must differ")
        operations = _tuple(
            self.required_operation_references,
            "layer audit required_operation_references",
            MAX_OPERATION_REFERENCES,
        )
        if not operations or not all(
            isinstance(item, RegisteredOperationReferenceV3) for item in operations
        ):
            raise ProofV3Error("layer audit requires registered PCS operations")
        _strictly_sorted(
            operations,
            tuple(item.sort_key() for item in operations),
            "layer audit registered PCS operations",
        )
        if any(item.layer_index != self.layer_index for item in operations):
            raise ProofV3Error(
                "layer audit operations must all belong to their declared layer"
            )
        query_heads = _u32(
            self.attention_query_head_count,
            "attention_query_head_count",
        )
        kv_heads = _u32(
            self.attention_key_value_head_count,
            "attention_key_value_head_count",
        )
        head_dimension = _u32(
            self.attention_head_dimension,
            "attention_head_dimension",
        )
        if query_heads == kv_heads == head_dimension == 0:
            if self.attention_semantics_id is not None:
                raise ProofV3Error(
                    "non-attention layer audits must not declare attention semantics"
                )
            return
        if not query_heads or not kv_heads or not head_dimension:
            raise ProofV3Error("full-attention audit geometry is incomplete")
        if query_heads % kv_heads:
            raise ProofV3Error(
                "full-attention query heads must divide evenly into KV groups"
            )
        if self.attention_semantics_id is None:
            raise ProofV3Error(
                "full-attention layer audits require attention semantics"
            )
        _identifier(self.attention_semantics_id, "attention_semantics_id")

    @property
    def is_full_attention(self) -> bool:
        return self.attention_query_head_count > 0

    def validate_for_transition_kind(self, transition_kind: str) -> None:
        """Check geometry only after the graph resolves the transition type."""

        if transition_kind not in _TRANSITION_KINDS:
            raise ProofV3Error("layer audit transition kind is not supported")
        if transition_kind == "full_attention" and not self.is_full_attention:
            raise ProofV3Error(
                "full-attention transitions require signed head geometry"
            )
        if transition_kind == "gdn" and self.is_full_attention:
            raise ProofV3Error(
                "GDN transitions must not declare attention head geometry"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "attention_head_dimension": self.attention_head_dimension,
            "attention_key_value_head_count": self.attention_key_value_head_count,
            "attention_query_head_count": self.attention_query_head_count,
            "attention_semantics_id": self.attention_semantics_id,
            "bridge_node_id": self.bridge_node_id,
            "layer_index": self.layer_index,
            "required_operation_references": [
                item.to_dict() for item in self.required_operation_references
            ],
            "transition_node_id": self.transition_node_id,
        }


@dataclass(frozen=True, slots=True)
class RuntimeHardAuditPolicyV3:
    """Validator-local effective policy supplied to every eligible request.

    This is not miner-controlled payload data. Runtime wiring must derive it
    from the active scheduler/probation configuration for the current request.
    It deliberately does **not** carry a caller-asserted hard/light bit: the
    post-commitment tier draw derives that result only after the envelope is
    frozen under the validator nonce.
    """

    request_kind: str
    effective_organic_hard_bps: int
    effective_canary_hard_bps: int
    effective_probation_failures: int
    nonce_selection_abi_id: str
    tier_selection_abi_id: str
    selection_abi_id: str

    def __post_init__(self) -> None:
        if self.request_kind not in {"organic", "canary"}:
            raise ProofV3Error("runtime hard-audit request_kind is not supported")
        _u32(
            self.effective_organic_hard_bps,
            "effective_organic_hard_bps",
        )
        _u32(
            self.effective_canary_hard_bps,
            "effective_canary_hard_bps",
        )
        _u32(
            self.effective_probation_failures,
            "effective_probation_failures",
            positive=True,
        )
        _identifier(self.nonce_selection_abi_id, "runtime nonce_selection_abi_id")
        _identifier(self.tier_selection_abi_id, "runtime tier_selection_abi_id")
        _identifier(self.selection_abi_id, "runtime selection_abi_id")
        if self.effective_organic_hard_bps > 10_000:
            raise ProofV3Error("effective_organic_hard_bps must not exceed 10000")
        if self.effective_canary_hard_bps > 10_000:
            raise ProofV3Error("effective_canary_hard_bps must not exceed 10000")


@dataclass(frozen=True, slots=True)
class HardAuditPolicyV3:
    """Signed minimum hard-audit policy; validators may only be stricter."""

    coverage_mode: str
    nonce_selection_abi_id: str
    tier_selection_abi_id: str
    selection_abi_id: str
    minimum_organic_hard_bps: int
    minimum_canary_hard_bps: int
    probation_failures: int
    selected_layer_count: int
    minimum_full_attention_layers: int
    minimum_gdn_layers: int
    full_attention_heads_per_layer: int
    transition_query_rows: int

    def __post_init__(self) -> None:
        if self.coverage_mode not in {
            GLOBAL_EXECUTION_COVERAGE_MODE_V3,
            ECONOMIC_SHELL_COVERAGE_MODE_V3,
        }:
            raise ProofV3Error("hard audit coverage_mode is not supported")
        _identifier(self.nonce_selection_abi_id, "nonce_selection_abi_id")
        _identifier(self.tier_selection_abi_id, "tier_selection_abi_id")
        _identifier(self.selection_abi_id, "selection_abi_id")
        if self.tier_selection_abi_id != POSTCOMMIT_AUDIT_TIER_ABI_V3:
            raise ProofV3Error("postcommit audit-tier selection ABI is not supported")
        _u32(self.minimum_organic_hard_bps, "minimum_organic_hard_bps")
        _u32(self.minimum_canary_hard_bps, "minimum_canary_hard_bps")
        _u32(self.probation_failures, "probation_failures", positive=True)
        _u32(self.selected_layer_count, "selected_layer_count", positive=True)
        _u32(
            self.minimum_full_attention_layers,
            "minimum_full_attention_layers",
        )
        _u32(self.minimum_gdn_layers, "minimum_gdn_layers")
        _u32(
            self.full_attention_heads_per_layer,
            "full_attention_heads_per_layer",
            positive=True,
        )
        _u32(
            self.transition_query_rows,
            "transition_query_rows",
            positive=True,
        )
        if self.minimum_organic_hard_bps > 10_000:
            raise ProofV3Error("minimum_organic_hard_bps must not exceed 10000")
        if self.minimum_canary_hard_bps > 10_000:
            raise ProofV3Error("minimum_canary_hard_bps must not exceed 10000")
        if self.transition_query_rows > MAX_HARD_AUDIT_QUERY_ROWS_V3:
            raise ProofV3Error("transition_query_rows exceeds the protocol maximum")
        if (
            self.minimum_full_attention_layers + self.minimum_gdn_layers
            > self.selected_layer_count
        ):
            raise ProofV3Error(
                "required transition coverage exceeds selected layer count"
            )
        if self.coverage_mode == GLOBAL_EXECUTION_COVERAGE_MODE_V3 and (
            self.nonce_selection_abi_id != POSTCOMMIT_GLOBAL_NONCE_ABI_V3
            or self.tier_selection_abi_id != POSTCOMMIT_AUDIT_TIER_ABI_V3
            or self.selection_abi_id != STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3
            or self.minimum_organic_hard_bps < 1_000
            or self.minimum_canary_hard_bps != 10_000
            or self.probation_failures != 1
            or self.selected_layer_count < GLOBAL_HARD_AUDIT_MIN_LAYERS_V3
            or self.full_attention_heads_per_layer
            < GLOBAL_HARD_AUDIT_MIN_HEADS_PER_FULL_ATTENTION_LAYER_V3
        ):
            raise ProofV3Error(
                "global execution profiles require postcommit tier selection and "
                "stratified folding, "
                "at least 1000 organic bps, 10000 canary bps, one-failure "
                "probation, four selected layers, and two selected heads"
            )

    @property
    def global_execution_qualified(self) -> bool:
        return self.coverage_mode == GLOBAL_EXECUTION_COVERAGE_MODE_V3

    def validate_runtime(self, runtime_policy: RuntimeHardAuditPolicyV3) -> None:
        """Reject an eligible request under a weaker live validator policy.

        This validates configuration only.  Whether this individual request
        is hard is derived later from the hidden nonce and frozen envelope.
        """

        if not isinstance(runtime_policy, RuntimeHardAuditPolicyV3):
            raise ProofV3VerificationError(
                "runtime hard-audit policy has an unexpected type"
            )
        if runtime_policy.nonce_selection_abi_id != self.nonce_selection_abi_id:
            raise ProofV3VerificationError(
                "runtime nonce-selection ABI does not match the signed profile"
            )
        if runtime_policy.tier_selection_abi_id != self.tier_selection_abi_id:
            raise ProofV3VerificationError(
                "runtime tier-selection ABI does not match the signed profile"
            )
        if runtime_policy.selection_abi_id != self.selection_abi_id:
            raise ProofV3VerificationError(
                "runtime selection ABI does not match the signed profile"
            )
        if runtime_policy.effective_organic_hard_bps < self.minimum_organic_hard_bps:
            raise ProofV3VerificationError(
                "runtime organic hard-audit rate is weaker than the signed profile"
            )
        if runtime_policy.effective_canary_hard_bps < self.minimum_canary_hard_bps:
            raise ProofV3VerificationError(
                "runtime canary hard-audit rate is weaker than the signed profile"
            )
        if runtime_policy.effective_probation_failures > self.probation_failures:
            raise ProofV3VerificationError(
                "runtime probation policy is weaker than the signed profile"
            )
        if (
            runtime_policy.request_kind == "canary"
            and runtime_policy.effective_canary_hard_bps != 10_000
        ):
            raise ProofV3VerificationError(
                "canary hard-audit requests must run at 10000 basis points"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage_mode": self.coverage_mode,
            "full_attention_heads_per_layer": self.full_attention_heads_per_layer,
            "minimum_canary_hard_bps": self.minimum_canary_hard_bps,
            "minimum_full_attention_layers": self.minimum_full_attention_layers,
            "minimum_gdn_layers": self.minimum_gdn_layers,
            "minimum_organic_hard_bps": self.minimum_organic_hard_bps,
            "nonce_selection_abi_id": self.nonce_selection_abi_id,
            "probation_failures": self.probation_failures,
            "selected_layer_count": self.selected_layer_count,
            "selection_abi_id": self.selection_abi_id,
            "tier_selection_abi_id": self.tier_selection_abi_id,
            "transition_query_rows": self.transition_query_rows,
        }


@dataclass(frozen=True, slots=True)
class ExecutionRelationSpecV3:
    """Complete signed statement ABI for one qualified model/quantization path."""

    relation_abi_id: str
    capture_abi_digest: bytes
    constraint_system_digest: bytes
    tokenizer_binding_digest: bytes
    sampler_abi_id: str
    sampler_abi_digest: bytes
    quantization_binding_digest: bytes
    recursive_accumulator_abi_id: str
    recursive_accumulator_digest: bytes
    prefill_chunk_tokens: int
    decode_chunk_tokens: int
    request_tensor_id: str
    final_token_tensor_id: str
    sequence_domain: SequenceDomainV3
    dimensions: tuple[DeclaredDimensionV3, ...]
    tensors: tuple[TensorDescriptorV3, ...]
    registered_operations: tuple[RegisteredOperationReferenceV3, ...]
    static_bindings: tuple[StaticParameterBindingV3, ...]
    static_table_bindings: tuple[StaticTableBindingV3, ...]
    tolerances: tuple[RelationToleranceV3, ...]
    nodes: tuple[ExecutionRelationNodeV3, ...]
    layer_audits: tuple[LayerAuditPlanV3, ...]
    cache: CacheRelationSpecV3
    audit_policy: HardAuditPolicyV3
    relation_spec_version: int = RELATION_SPEC_FORMAT_VERSION

    def __post_init__(self) -> None:
        _identifier(self.relation_abi_id, "relation_abi_id")
        _fixed32(self.capture_abi_digest, "capture_abi_digest", nonzero=True)
        _fixed32(
            self.constraint_system_digest,
            "constraint_system_digest",
            nonzero=True,
        )
        _fixed32(
            self.tokenizer_binding_digest,
            "tokenizer_binding_digest",
            nonzero=True,
        )
        _identifier(self.sampler_abi_id, "sampler_abi_id")
        _fixed32(self.sampler_abi_digest, "sampler_abi_digest", nonzero=True)
        _fixed32(
            self.quantization_binding_digest,
            "quantization_binding_digest",
            nonzero=True,
        )
        _identifier(self.recursive_accumulator_abi_id, "recursive_accumulator_abi_id")
        _fixed32(
            self.recursive_accumulator_digest,
            "recursive_accumulator_digest",
            nonzero=True,
        )
        _u32(self.prefill_chunk_tokens, "prefill_chunk_tokens", positive=True)
        _u32(self.decode_chunk_tokens, "decode_chunk_tokens", positive=True)
        _identifier(self.request_tensor_id, "request_tensor_id")
        _identifier(self.final_token_tensor_id, "final_token_tensor_id")
        if not isinstance(self.sequence_domain, SequenceDomainV3):
            raise ProofV3Error("sequence_domain has an unexpected type")
        _u32(
            self.relation_spec_version,
            "relation_spec_version",
            positive=True,
        )
        if self.relation_spec_version != RELATION_SPEC_FORMAT_VERSION:
            raise ProofV3Error("relation spec version is not supported")
        dimensions = _tuple(self.dimensions, "dimensions", MAX_DECLARED_DIMENSIONS)
        tensors = _tuple(self.tensors, "tensors", MAX_TENSORS)
        operations = _tuple(
            self.registered_operations,
            "registered_operations",
            MAX_OPERATION_REFERENCES,
        )
        bindings = _tuple(self.static_bindings, "static_bindings", MAX_STATIC_BINDINGS)
        table_bindings = _tuple(
            self.static_table_bindings,
            "static_table_bindings",
            MAX_STATIC_TABLE_BINDINGS,
        )
        tolerances = _tuple(
            self.tolerances,
            "tolerances",
            MAX_TOLERANCE_PROFILES,
        )
        nodes = _tuple(self.nodes, "relation nodes", MAX_RELATION_NODES)
        layer_audits = _tuple(
            self.layer_audits,
            "layer_audits",
            MAX_RELATION_NODES,
        )
        if (
            not dimensions
            or not tensors
            or not operations
            or not bindings
            or not table_bindings
            or not tolerances
        ):
            raise ProofV3Error("relation spec has an incomplete signed statement")
        if not nodes:
            raise ProofV3Error("relation spec requires ordered graph nodes")
        if not layer_audits:
            raise ProofV3Error("relation spec requires per-layer audit plans")
        if not all(isinstance(item, DeclaredDimensionV3) for item in dimensions):
            raise ProofV3Error("relation dimensions have an unexpected type")
        if not all(isinstance(item, TensorDescriptorV3) for item in tensors):
            raise ProofV3Error("relation tensors have an unexpected type")
        if not all(
            isinstance(item, RegisteredOperationReferenceV3) for item in operations
        ):
            raise ProofV3Error("relation operations have an unexpected type")
        if not all(isinstance(item, StaticParameterBindingV3) for item in bindings):
            raise ProofV3Error("relation static bindings have an unexpected type")
        if not all(isinstance(item, StaticTableBindingV3) for item in table_bindings):
            raise ProofV3Error("relation static table bindings have an unexpected type")
        if not all(isinstance(item, RelationToleranceV3) for item in tolerances):
            raise ProofV3Error("relation tolerances have an unexpected type")
        if not all(isinstance(item, ExecutionRelationNodeV3) for item in nodes):
            raise ProofV3Error("relation nodes have an unexpected type")
        if not all(isinstance(item, LayerAuditPlanV3) for item in layer_audits):
            raise ProofV3Error("layer audit plans have an unexpected type")
        if not isinstance(self.cache, CacheRelationSpecV3):
            raise ProofV3Error("relation cache has an unexpected type")
        if not isinstance(self.audit_policy, HardAuditPolicyV3):
            raise ProofV3Error("relation audit policy has an unexpected type")
        _strictly_sorted(
            dimensions,
            tuple(item.dimension_id for item in dimensions),
            "relation dimensions",
        )
        _strictly_sorted(
            tensors,
            tuple(item.tensor_id for item in tensors),
            "relation tensors",
        )
        _strictly_sorted(
            operations,
            tuple(item.sort_key() for item in operations),
            "registered operations",
        )
        _strictly_sorted(
            bindings,
            tuple(item.binding_id for item in bindings),
            "static bindings",
        )
        _strictly_sorted(
            table_bindings,
            tuple(item.binding_id for item in table_bindings),
            "static table bindings",
        )
        _strictly_sorted(
            tolerances,
            tuple(item.tolerance_id for item in tolerances),
            "tolerances",
        )
        _strictly_sorted(
            layer_audits,
            tuple(item.layer_index for item in layer_audits),
            "layer audit plans",
        )
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ProofV3Error("relation nodes must not contain duplicate ids")
        self._validate_graph()

    def _validate_graph(self) -> None:
        dimensions = {item.dimension_id: item for item in self.dimensions}
        dimension_ids = set(dimensions)
        tensors = {item.tensor_id: item for item in self.tensors}
        operations = set(self.registered_operations)
        bindings = {item.binding_id: item for item in self.static_bindings}
        table_bindings = tuple(self.static_table_bindings)
        binding_ids = set(bindings)
        tolerance_ids = {item.tolerance_id for item in self.tolerances}
        if (
            self.request_tensor_id not in tensors
            or self.final_token_tensor_id not in tensors
        ):
            raise ProofV3Error("relation source or final tensor is not declared")
        if self.request_tensor_id == self.final_token_tensor_id:
            raise ProofV3Error("relation source and final tensors must differ")
        if tensors[self.request_tensor_id].commitment_role != "request_input":
            raise ProofV3Error("request tensor must have request_input role")
        if tensors[self.final_token_tensor_id].commitment_role != "token_output":
            raise ProofV3Error("final token tensor must have token_output role")
        operation_by_descriptor = {
            item.descriptor_digest: item for item in self.registered_operations
        }
        if len(operation_by_descriptor) != len(self.registered_operations):
            raise ProofV3Error("registered operation descriptor digests are duplicated")
        operation_table_bindings = tuple(
            item for item in table_bindings if item.subject_kind == "operation"
        )
        if {
            item.operation_descriptor_digest for item in operation_table_bindings
        } != set(operation_by_descriptor):
            raise ProofV3Error(
                "static table bindings do not exactly cover registered operations"
            )
        if len(operation_table_bindings) != len(operation_by_descriptor):
            raise ProofV3Error(
                "registered operation has multiple static table bindings"
            )
        static_parameter_table_bindings = tuple(
            item for item in table_bindings if item.subject_kind == "static_parameter"
        )
        for item in static_parameter_table_bindings:
            assert item.static_binding_id is not None
            assert item.static_binding_digest is not None
            bound = bindings.get(item.static_binding_id)
            if bound is None or bound.binding_digest != item.static_binding_digest:
                raise ProofV3Error(
                    "static table binding does not match a signed static parameter"
                )
        if {
            item.static_binding_id for item in static_parameter_table_bindings
        } != binding_ids:
            raise ProofV3Error(
                "static table bindings do not cover signed static parameters"
            )
        table_ids = tuple(item.table_id for item in table_bindings)
        if len(table_ids) != len(set(table_ids)):
            raise ProofV3Error("static table bindings reuse a table descriptor")
        descriptor_digests = tuple(
            item.static_table_descriptor_digest for item in table_bindings
        )
        if len(descriptor_digests) != len(set(descriptor_digests)):
            raise ProofV3Error("static table bindings reuse a descriptor digest")
        self.sequence_domain.validate_dimensions(dimensions)
        context_dimension_id = self.sequence_domain.context_dimension_id
        decode_dimension_id = self.sequence_domain.decode_dimension_id
        sequence_dimension_id = self.sequence_domain.sequence_dimension_id
        if context_dimension_id not in tensors[self.request_tensor_id].shape:
            raise ProofV3Error("request tensor must be indexed by context_tokens")
        if decode_dimension_id not in tensors[self.final_token_tensor_id].shape:
            raise ProofV3Error("final token tensor must be indexed by decode_tokens")
        for tensor in self.tensors:
            for dimension in tensor.shape:
                if isinstance(dimension, str) and dimension not in dimension_ids:
                    raise ProofV3Error(
                        "tensor shape references an undeclared dimension"
                    )
        final_hidden_tensors = {
            tensor.tensor_id
            for tensor in self.tensors
            if tensor.commitment_role == "final_hidden"
        }
        logit_tensors = {
            tensor.tensor_id
            for tensor in self.tensors
            if tensor.commitment_role == "logits"
        }
        if not final_hidden_tensors or not logit_tensors:
            raise ProofV3Error("relation requires final-hidden and logits tensors")
        for tensor_id in final_hidden_tensors | logit_tensors:
            if decode_dimension_id not in tensors[tensor_id].shape:
                raise ProofV3Error(
                    "final-hidden and logits tensors must cover every decode token"
                )
        cache_by_layer = self.cache.by_layer()
        cache_tensors_by_layer = {
            layer_index: frozenset(family.tensor_ids())
            for layer_index, family in cache_by_layer.items()
        }
        cache_tensors = frozenset(
            tensor_id
            for tensor_ids in cache_tensors_by_layer.values()
            for tensor_id in tensor_ids
        )
        for tensor_id in cache_tensors:
            if tensor_id not in tensors:
                raise ProofV3Error("cache tensors are not declared")
            if tensors[tensor_id].commitment_role != "cache":
                raise ProofV3Error("cache tensors must have cache commitment role")
            if sequence_dimension_id not in tensors[tensor_id].shape:
                raise ProofV3Error("cache tensors must cover the full sequence domain")

        available = {self.request_tensor_id}
        used_operations: set[RegisteredOperationReferenceV3] = set()
        used_bindings: set[str] = set()
        cache_producers: set[str] = set()
        cache_consumers: set[str] = set()
        transition_nodes_by_layer: dict[int, str] = {}
        transition_node_ids_by_layer: dict[int, str] = {}
        bridge_node_ids_by_layer: dict[int, str] = {}

        def exact_tail_width(tensor_id: str) -> int | None:
            tail = tensors[tensor_id].shape[-1]
            if type(tail) is int:
                return tail
            return dimensions[tail].exact_value

        for index, node in enumerate(self.nodes):
            inputs = set(node.input_tensor_ids)
            outputs = set(node.output_tensor_ids)
            if not inputs.issubset(tensors) or not outputs.issubset(tensors):
                raise ProofV3Error("relation node references an undeclared tensor")
            if not inputs.issubset(available):
                raise ProofV3Error(
                    f"relation node {index} consumes a tensor before its producer"
                )
            if self.request_tensor_id in outputs or outputs & available:
                raise ProofV3Error("relation graph has duplicate tensor producers")
            if node.tolerance_id not in tolerance_ids:
                raise ProofV3Error("relation node references an undeclared tolerance")
            if not set(node.static_binding_ids).issubset(binding_ids):
                raise ProofV3Error(
                    "relation node references an undeclared static binding"
                )
            used_bindings.update(node.static_binding_ids)
            if node.operation_reference is not None:
                if node.operation_reference not in operations:
                    raise ProofV3Error(
                        "linear relation node references an undeclared PCS operation"
                    )
                if len(node.input_tensor_ids) != 1 or len(node.output_tensor_ids) != 1:
                    raise ProofV3Error(
                        "linear relation nodes require one X and one Y tensor"
                    )
                if (
                    exact_tail_width(node.input_tensor_ids[0])
                    != node.operation_reference.rows
                ):
                    raise ProofV3Error("linear X tensor width does not match PCS rows")
                if (
                    exact_tail_width(node.output_tensor_ids[0])
                    != node.operation_reference.cols
                ):
                    raise ProofV3Error("linear Y tensor width does not match PCS cols")
                if node.static_binding_ids:
                    raise ProofV3Error(
                        "linear relation nodes may not add static bindings"
                    )
                if node.layer_index >= 0 and (
                    sequence_dimension_id not in tensors[node.input_tensor_ids[0]].shape
                    or sequence_dimension_id
                    not in tensors[node.output_tensor_ids[0]].shape
                ):
                    raise ProofV3Error(
                        "layer PCS operations must cover the full sequence domain"
                    )
                used_operations.add(node.operation_reference)
            else:
                allowed_kinds = _NODE_BINDING_KINDS[node.relation_id]
                for binding_id in node.static_binding_ids:
                    binding = bindings[binding_id]
                    if binding.layer_index != node.layer_index:
                        raise ProofV3Error(
                            "relation node binding layer does not match the node"
                        )
                    if binding.binding_kind not in allowed_kinds:
                        raise ProofV3Error(
                            "relation node binding kind does not match the relation"
                        )
            if cache_tensors & outputs:
                cache_producers.update(cache_tensors & outputs)
            if cache_tensors & inputs:
                cache_consumers.update(cache_tensors & inputs)
            if node.relation_id in {"full_attention", "gdn"}:
                family = cache_by_layer.get(node.layer_index)
                if family is None:
                    raise ProofV3Error("transition layer has no signed cache family")
                expected_cache_kind = (
                    "attention_kv"
                    if node.relation_id == "full_attention"
                    else "gdn_state"
                )
                if family.cache_kind != expected_cache_kind:
                    raise ProofV3Error(
                        "transition relation has an incompatible cache family"
                    )
                if not cache_tensors_by_layer[node.layer_index].issubset(outputs):
                    raise ProofV3Error(
                        "transition relations must produce their complete layer cache family"
                    )
                if not any(
                    sequence_dimension_id in tensors[tensor_id].shape
                    and tensors[tensor_id].commitment_role == "runtime_state"
                    for tensor_id in inputs
                ) or not any(
                    sequence_dimension_id in tensors[tensor_id].shape
                    and tensors[tensor_id].commitment_role == "runtime_state"
                    for tensor_id in outputs
                ):
                    raise ProofV3Error(
                        "transition relations must bind sequence-indexed runtime state"
                    )
                if node.layer_index in transition_nodes_by_layer:
                    raise ProofV3Error(
                        "each layer may have only one transition relation"
                    )
                transition_nodes_by_layer[node.layer_index] = node.relation_id
                transition_node_ids_by_layer[node.layer_index] = node.node_id
            if node.relation_id in {"bridge", "moe"}:
                family = cache_by_layer.get(node.layer_index)
                if family is None:
                    raise ProofV3Error("bridge layer has no signed cache family")
                if not cache_tensors_by_layer[node.layer_index].issubset(inputs):
                    raise ProofV3Error(
                        "bridge relations must consume their complete layer cache family"
                    )
                if node.layer_index in bridge_node_ids_by_layer:
                    raise ProofV3Error("each layer may have only one bridge relation")
                if node.relation_id == "moe":
                    runtime_inputs = tuple(
                        tensor_id
                        for tensor_id in node.input_tensor_ids
                        if tensors[tensor_id].commitment_role == "runtime_state"
                        and sequence_dimension_id in tensors[tensor_id].shape
                    )
                    runtime_outputs = tuple(
                        tensor_id
                        for tensor_id in node.output_tensor_ids
                        if tensors[tensor_id].commitment_role == "runtime_state"
                        and sequence_dimension_id in tensors[tensor_id].shape
                    )
                    if len(runtime_inputs) != 2 or len(runtime_outputs) != 1:
                        raise ProofV3Error(
                            "MoE relations require the layer residual, transition "
                            "output and one resulting runtime state"
                        )
                bridge_node_ids_by_layer[node.layer_index] = node.node_id
            available.update(outputs)
        if self.final_token_tensor_id not in self.nodes[-1].output_tensor_ids:
            raise ProofV3Error("final token tensor must be produced by the last node")
        if self.nodes[-1].relation_id != "sampler":
            raise ProofV3Error("last relation node must be the canonical sampler")
        if self.nodes[-1].transition_adapter_id != self.sampler_abi_id:
            raise ProofV3Error("final sampler node does not match the sampler ABI")
        if set(self.nodes[-1].output_tensor_ids) != {self.final_token_tensor_id}:
            raise ProofV3Error("last relation node may only produce the final token")
        lm_head_nodes = [
            node
            for node in self.nodes
            if node.operation_reference is not None
            and node.operation_reference.layer_index == -1
        ]
        if len(lm_head_nodes) != 1:
            raise ProofV3Error("relation requires exactly one registered LM-head node")
        lm_head_node = lm_head_nodes[0]
        lm_head_input = lm_head_node.input_tensor_ids[0]
        lm_head_output = lm_head_node.output_tensor_ids[0]
        if lm_head_input not in final_hidden_tensors:
            raise ProofV3Error(
                "LM-head input must be an authenticated final-hidden tensor"
            )
        if lm_head_output not in logit_tensors:
            raise ProofV3Error("LM-head output must be an authenticated logits tensor")
        if set(self.nodes[-1].input_tensor_ids) != {lm_head_output}:
            raise ProofV3Error(
                "canonical sampler must consume the registered LM-head logits"
            )
        if used_operations != operations:
            raise ProofV3Error(
                "every registered PCS operation must appear in the graph"
            )
        if used_bindings != binding_ids:
            raise ProofV3Error("every static binding must appear in the graph")
        if available != set(tensors):
            raise ProofV3Error("every declared tensor must have one graph producer")
        if cache_producers != cache_tensors or cache_consumers != cache_tensors:
            raise ProofV3Error(
                "relation graph must produce and consume both cache tensors"
            )
        required_layers = {
            operation.layer_index
            for operation in self.registered_operations
            if operation.layer_index >= 0
        }
        if set(cache_by_layer) != required_layers:
            raise ProofV3Error("every registered layer requires one cache family")
        if set(transition_nodes_by_layer) != required_layers:
            raise ProofV3Error(
                "every registered layer requires one full-attention or GDN transition"
            )
        if set(bridge_node_ids_by_layer) != required_layers:
            raise ProofV3Error("every registered layer requires one bridge transition")

        audit_by_layer = {audit.layer_index: audit for audit in self.layer_audits}
        if set(audit_by_layer) != required_layers:
            raise ProofV3Error("every registered layer requires one signed audit plan")
        full_attention_audits: list[LayerAuditPlanV3] = []
        gdn_audits: list[LayerAuditPlanV3] = []
        for layer_index in sorted(required_layers):
            audit = audit_by_layer[layer_index]
            transition_kind = transition_nodes_by_layer[layer_index]
            audit.validate_for_transition_kind(transition_kind)
            if audit.transition_node_id != transition_node_ids_by_layer[layer_index]:
                raise ProofV3Error(
                    "layer audit plan does not reference its transition node"
                )
            if audit.bridge_node_id != bridge_node_ids_by_layer[layer_index]:
                raise ProofV3Error(
                    "layer audit plan does not reference its bridge node"
                )
            expected_operations = {
                operation
                for operation in self.registered_operations
                if operation.layer_index == layer_index
            }
            if set(audit.required_operation_references) != expected_operations:
                raise ProofV3Error(
                    "layer audit plan does not cover every registered layer operation"
                )
            if transition_kind == "full_attention":
                full_attention_audits.append(audit)
            else:
                gdn_audits.append(audit)

        policy = self.audit_policy
        if policy.selected_layer_count > len(audit_by_layer):
            raise ProofV3Error(
                "hard-audit layer count exceeds the signed execution graph"
            )
        required_full_attention = min(
            GLOBAL_HARD_AUDIT_MIN_FULL_ATTENTION_LAYERS_V3,
            len(full_attention_audits),
        )
        required_gdn = min(
            GLOBAL_HARD_AUDIT_MIN_GDN_LAYERS_V3,
            len(gdn_audits),
        )
        if policy.minimum_full_attention_layers < required_full_attention:
            raise ProofV3Error(
                "hard-audit policy weakens signed full-attention coverage"
            )
        if policy.minimum_gdn_layers < required_gdn:
            raise ProofV3Error("hard-audit policy weakens signed GDN coverage")
        if policy.minimum_full_attention_layers > len(full_attention_audits):
            raise ProofV3Error(
                "hard-audit policy requires unavailable full-attention layers"
            )
        if policy.minimum_gdn_layers > len(gdn_audits):
            raise ProofV3Error("hard-audit policy requires unavailable GDN layers")
        for audit in full_attention_audits:
            if audit.attention_query_head_count < policy.full_attention_heads_per_layer:
                raise ProofV3Error(
                    "full-attention audit has fewer heads than the signed policy"
                )

        needed = {self.final_token_tensor_id}
        for node in reversed(self.nodes):
            outputs = set(node.output_tensor_ids)
            if not outputs.issubset(needed):
                raise ProofV3Error("relation graph contains a disconnected output")
            needed.update(node.input_tensor_ids)
        if needed != {self.request_tensor_id} | {
            tensor_id
            for node in self.nodes
            for tensor_id in node.input_tensor_ids + node.output_tensor_ids
        }:
            raise ProofV3Error("relation graph is disconnected from the final token")

    def validate_for_limits(
        self,
        *,
        max_context_tokens: int,
        max_decode_tokens: int,
    ) -> None:
        """Check profile limits against the signed chunk/cache statement."""

        context_limit = _u32(
            max_context_tokens,
            "max_context_tokens",
            positive=True,
        )
        decode_limit = _u32(
            max_decode_tokens,
            "max_decode_tokens",
            positive=True,
        )
        if self.prefill_chunk_tokens > context_limit:
            raise ProofV3Error("prefill chunk exceeds the signed context limit")
        if self.decode_chunk_tokens > decode_limit:
            raise ProofV3Error("decode chunk exceeds the signed decode limit")
        if context_limit + decode_limit >= 1 << 32:
            raise ProofV3Error("logical token coordinate space overflows")
        if self.cache.max_cache_tokens < context_limit + decode_limit:
            raise ProofV3Error("cache capacity does not cover signed token limits")
        dimensions = {item.dimension_id: item for item in self.dimensions}
        self.sequence_domain.validate_dimensions(dimensions)
        context_dimension = dimensions[self.sequence_domain.context_dimension_id]
        decode_dimension = dimensions[self.sequence_domain.decode_dimension_id]
        sequence_dimension = dimensions[self.sequence_domain.sequence_dimension_id]
        if context_dimension.maximum != context_limit:
            raise ProofV3Error(
                "context token domain must exactly match the profile limit"
            )
        if decode_dimension.maximum != decode_limit:
            raise ProofV3Error(
                "decode token domain must exactly match the profile limit"
            )
        if sequence_dimension.maximum != context_limit + decode_limit:
            raise ProofV3Error(
                "sequence token domain must exactly match profile limits"
            )

    def validate_execution_counts(
        self,
        *,
        context_token_count: int,
        decode_token_count: int,
    ) -> int:
        """Bind concrete validator counts to the signed prompt/decode/cache domain."""

        dimensions = {item.dimension_id: item for item in self.dimensions}
        return self.sequence_domain.bind_counts(
            dimensions=dimensions,
            context_token_count=context_token_count,
            decode_token_count=decode_token_count,
        )

    @property
    def global_execution_qualified(self) -> bool:
        return (
            self.relation_abi_id == GLOBAL_EXECUTION_RELATION_ABI_V3
            and self.audit_policy.global_execution_qualified
            and self.recursive_accumulator_abi_id == RECURSIVE_CHUNK_ACCUMULATOR_ABI_V3
            and all(
                tolerance.comparator_id in _EXACT_COMPARATORS
                for tolerance in self.tolerances
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_policy": self.audit_policy.to_dict(),
            "cache": self.cache.to_dict(),
            "capture_abi_digest": self.capture_abi_digest.hex(),
            "constraint_system_digest": self.constraint_system_digest.hex(),
            "decode_chunk_tokens": self.decode_chunk_tokens,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "final_token_tensor_id": self.final_token_tensor_id,
            "layer_audits": [item.to_dict() for item in self.layer_audits],
            "nodes": [item.to_dict() for item in self.nodes],
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "recursive_accumulator_abi_id": self.recursive_accumulator_abi_id,
            "recursive_accumulator_digest": self.recursive_accumulator_digest.hex(),
            "registered_operations": [
                item.to_dict() for item in self.registered_operations
            ],
            "relation_abi_id": self.relation_abi_id,
            "relation_spec_version": self.relation_spec_version,
            "request_tensor_id": self.request_tensor_id,
            "sampler_abi_id": self.sampler_abi_id,
            "sampler_abi_digest": self.sampler_abi_digest.hex(),
            "quantization_binding_digest": self.quantization_binding_digest.hex(),
            "sequence_domain": self.sequence_domain.to_dict(),
            "static_bindings": [item.to_dict() for item in self.static_bindings],
            "static_table_bindings": [
                item.to_dict() for item in self.static_table_bindings
            ],
            "tensors": [item.to_dict() for item in self.tensors],
            "tokenizer_binding_digest": self.tokenizer_binding_digest.hex(),
            "tolerances": [item.to_dict() for item in self.tolerances],
        }

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_RELATION_SPEC_BYTES:
            raise ProofV3Error("relation spec exceeds the protocol size limit")
        return encoded

    @classmethod
    def from_canonical_json_bytes(cls, encoded: bytes) -> "ExecutionRelationSpecV3":
        if (
            not isinstance(encoded, bytes)
            or not encoded
            or len(encoded) > MAX_RELATION_SPEC_BYTES
        ):
            raise ProofV3Error("relation spec byte length is out of range")
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except ProofV3Error:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("relation spec is not canonical JSON") from exc
        result = execution_relation_spec_from_dict(value)
        if result.canonical_json_bytes() != encoded:
            raise ProofV3Error("relation spec is not canonical JSON")
        return result


def _object(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ProofV3Error(f"{name} fields do not match the canonical schema")
    return value


def _list(value: object, name: str, maximum: int) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ProofV3Error(f"{name} must be a bounded list")
    return value


def _json_identifier(value: object, name: str) -> str:
    return _identifier(value, name)


def _json_u32(value: object, name: str, *, positive: bool = False) -> int:
    return _u32(value, name, positive=positive)


def _json_u64(value: object, name: str, *, positive: bool = False) -> int:
    return _u64(value, name, positive=positive)


def _json_i32(value: object, name: str, *, minimum: int = -(1 << 31)) -> int:
    return _i32(value, name, minimum=minimum)


def _json_hex(value: object, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ProofV3Error(f"{name} must be lowercase 32-byte hexadecimal")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ProofV3Error(f"{name} must be hexadecimal") from exc
    return _fixed32(result, name, nonzero=True)


def _optional_json_hex(value: object, name: str) -> bytes | None:
    return None if value is None else _json_hex(value, name)


def _dimension_from_dict(value: object) -> DeclaredDimensionV3:
    item = _object(value, {"dimension_id", "maximum", "minimum"}, "dimension")
    return DeclaredDimensionV3(
        dimension_id=_json_identifier(item["dimension_id"], "dimension_id"),
        minimum=_json_u32(item["minimum"], "dimension minimum"),
        maximum=_json_u32(item["maximum"], "dimension maximum", positive=True),
    )


def _tensor_from_dict(value: object) -> TensorDescriptorV3:
    item = _object(
        value,
        {
            "commitment_role",
            "encoding_id",
            "layout_id",
            "scale_rule_id",
            "shape",
            "tensor_id",
        },
        "tensor",
    )
    shape_value = _list(item["shape"], "tensor shape", MAX_TENSOR_RANK)
    shape: list[ShapeDimensionV3] = []
    for index, dimension in enumerate(shape_value):
        if type(dimension) is int:
            shape.append(_json_u32(dimension, f"tensor shape[{index}]", positive=True))
        elif isinstance(dimension, str):
            shape.append(_json_identifier(dimension, f"tensor shape[{index}]"))
        else:
            raise ProofV3Error("tensor shape values are not canonical")
    return TensorDescriptorV3(
        tensor_id=_json_identifier(item["tensor_id"], "tensor_id"),
        shape=tuple(shape),
        encoding_id=_json_identifier(item["encoding_id"], "encoding_id"),
        layout_id=_json_identifier(item["layout_id"], "layout_id"),
        scale_rule_id=_json_identifier(item["scale_rule_id"], "scale_rule_id"),
        commitment_role=_json_identifier(item["commitment_role"], "commitment_role"),
    )


def _operation_from_dict(value: object) -> RegisteredOperationReferenceV3:
    item = _object(
        value,
        {
            "cols",
            "descriptor_digest",
            "expert_id",
            "layer_index",
            "operation_id",
            "rows",
            "weight_commitment",
        },
        "registered operation",
    )
    expert_value = item["expert_id"]
    if expert_value is not None:
        expert_value = _json_u32(expert_value, "operation expert_id")
    return RegisteredOperationReferenceV3(
        layer_index=_json_i32(item["layer_index"], "operation layer_index", minimum=-1),
        operation_id=_json_identifier(item["operation_id"], "operation_id"),
        expert_id=expert_value,
        rows=_json_u32(item["rows"], "operation rows", positive=True),
        cols=_json_u32(item["cols"], "operation cols", positive=True),
        weight_commitment=_json_hex(
            item["weight_commitment"],
            "operation weight_commitment",
        ),
        descriptor_digest=_json_hex(
            item["descriptor_digest"],
            "operation descriptor_digest",
        ),
    )


def _binding_from_dict(value: object) -> StaticParameterBindingV3:
    item = _object(
        value,
        {"binding_digest", "binding_id", "binding_kind", "layer_index"},
        "static binding",
    )
    return StaticParameterBindingV3(
        binding_id=_json_identifier(item["binding_id"], "binding_id"),
        binding_kind=_json_identifier(item["binding_kind"], "binding_kind"),
        binding_digest=_json_hex(item["binding_digest"], "binding_digest"),
        layer_index=_json_i32(item["layer_index"], "binding layer_index", minimum=-1),
    )


def _static_table_binding_from_dict(value: object) -> StaticTableBindingV3:
    item = _object(
        value,
        {
            "binding_id",
            "element_encoding_id",
            "logical_leaf_count",
            "logical_leaf_start",
            "operation_descriptor_digest",
            "page_count",
            "page_start",
            "scale_encoding_id",
            "static_binding_digest",
            "static_binding_id",
            "static_table_descriptor_digest",
            "subject_kind",
            "table_id",
        },
        "static table binding",
    )
    return StaticTableBindingV3(
        binding_id=_json_identifier(item["binding_id"], "static table binding_id"),
        subject_kind=_json_identifier(
            item["subject_kind"], "static table subject_kind"
        ),
        operation_descriptor_digest=_optional_json_hex(
            item["operation_descriptor_digest"],
            "static table operation_descriptor_digest",
        ),
        static_binding_id=_optional_identifier(
            item["static_binding_id"],
            "static table static_binding_id",
        ),
        static_binding_digest=_optional_json_hex(
            item["static_binding_digest"],
            "static table static_binding_digest",
        ),
        table_id=_json_identifier(item["table_id"], "static table table_id"),
        static_table_descriptor_digest=_json_hex(
            item["static_table_descriptor_digest"],
            "static table descriptor_digest",
        ),
        logical_leaf_start=_json_u64(
            item["logical_leaf_start"], "static table logical_leaf_start"
        ),
        logical_leaf_count=_json_u64(
            item["logical_leaf_count"],
            "static table logical_leaf_count",
            positive=True,
        ),
        page_start=_json_u64(item["page_start"], "static table page_start"),
        page_count=_json_u64(
            item["page_count"], "static table page_count", positive=True
        ),
        element_encoding_id=_json_identifier(
            item["element_encoding_id"], "static table element_encoding_id"
        ),
        scale_encoding_id=_json_identifier(
            item["scale_encoding_id"], "static table scale_encoding_id"
        ),
    )


def _tolerance_from_dict(value: object) -> RelationToleranceV3:
    item = _object(
        value,
        {"absolute_q32", "comparator_id", "relative_bps", "tolerance_id"},
        "tolerance",
    )
    return RelationToleranceV3(
        tolerance_id=_json_identifier(item["tolerance_id"], "tolerance_id"),
        comparator_id=_json_identifier(item["comparator_id"], "comparator_id"),
        absolute_q32=_json_u32(item["absolute_q32"], "absolute_q32"),
        relative_bps=_json_u32(item["relative_bps"], "relative_bps"),
    )


def _identifier_tuple(value: object, name: str) -> tuple[str, ...]:
    values = _list(value, name, MAX_NODE_TENSORS)
    return tuple(
        _json_identifier(item, f"{name}[{index}]") for index, item in enumerate(values)
    )


def _sequence_domain_from_dict(value: object) -> SequenceDomainV3:
    item = _object(
        value,
        {
            "context_dimension_id",
            "decode_dimension_id",
            "relation_id",
            "sequence_dimension_id",
        },
        "sequence domain",
    )
    return SequenceDomainV3(
        context_dimension_id=_json_identifier(
            item["context_dimension_id"],
            "context_dimension_id",
        ),
        decode_dimension_id=_json_identifier(
            item["decode_dimension_id"],
            "decode_dimension_id",
        ),
        sequence_dimension_id=_json_identifier(
            item["sequence_dimension_id"],
            "sequence_dimension_id",
        ),
        relation_id=_json_identifier(item["relation_id"], "relation_id"),
    )


def _node_from_dict(value: object) -> ExecutionRelationNodeV3:
    item = _object(
        value,
        {
            "input_tensor_ids",
            "layer_index",
            "node_id",
            "operation_reference",
            "output_tensor_ids",
            "relation_id",
            "static_binding_ids",
            "tolerance_id",
            "transition_adapter_id",
        },
        "relation node",
    )
    operation_value = item["operation_reference"]
    if operation_value is not None and not isinstance(operation_value, Mapping):
        raise ProofV3Error("node operation_reference is not canonical")
    return ExecutionRelationNodeV3(
        node_id=_json_identifier(item["node_id"], "node_id"),
        layer_index=_json_i32(item["layer_index"], "node layer_index", minimum=-1),
        relation_id=_json_identifier(item["relation_id"], "relation_id"),
        transition_adapter_id=_json_identifier(
            item["transition_adapter_id"],
            "transition_adapter_id",
        ),
        input_tensor_ids=_identifier_tuple(
            item["input_tensor_ids"], "input_tensor_ids"
        ),
        output_tensor_ids=_identifier_tuple(
            item["output_tensor_ids"], "output_tensor_ids"
        ),
        tolerance_id=_json_identifier(item["tolerance_id"], "tolerance_id"),
        static_binding_ids=_identifier_tuple(
            item["static_binding_ids"],
            "static_binding_ids",
        ),
        operation_reference=(
            None if operation_value is None else _operation_from_dict(operation_value)
        ),
    )


def _optional_identifier(value: object, name: str) -> str | None:
    return None if value is None else _json_identifier(value, name)


def _layer_cache_from_dict(value: object) -> LayerCacheRelationV3:
    item = _object(
        value,
        {
            "cache_kind",
            "cache_semantics_id",
            "key_tensor_id",
            "layer_index",
            "state_tensor_ids",
            "value_tensor_id",
        },
        "layer cache family",
    )
    return LayerCacheRelationV3(
        layer_index=_json_i32(item["layer_index"], "cache layer_index", minimum=0),
        cache_kind=_json_identifier(item["cache_kind"], "cache_kind"),
        cache_semantics_id=_json_identifier(
            item["cache_semantics_id"],
            "cache_semantics_id",
        ),
        key_tensor_id=_optional_identifier(item["key_tensor_id"], "key_tensor_id"),
        value_tensor_id=_optional_identifier(
            item["value_tensor_id"], "value_tensor_id"
        ),
        state_tensor_ids=_identifier_tuple(
            item["state_tensor_ids"],
            "cache state_tensor_ids",
        ),
    )


def _cache_from_dict(value: object) -> CacheRelationSpecV3:
    if not isinstance(value, Mapping):
        raise ProofV3Error("cache must be an object")
    prefix_cache = value.get("allows_prefix_cache_sharing") is True
    replay_bound_fields = {
        "prefix_cache_k_cell_delta_max",
        "prefix_cache_k_row_sq_delta_max",
        "prefix_cache_v_cell_delta_max",
        "prefix_cache_v_row_sq_delta_max",
    }
    item = _object(
        value,
        {
            "allows_prefix_cache_sharing",
            "allows_sliding_window",
            "allows_speculative_decode",
            "cache_semantics_id",
            "layer_caches",
            "logical_address_abi_id",
            "max_cache_tokens",
            "page_token_count",
            "retention_lease_abi_id",
        } | (replay_bound_fields if prefix_cache else set()),
        "cache",
    )
    for field in (
        "allows_prefix_cache_sharing",
        "allows_sliding_window",
        "allows_speculative_decode",
    ):
        if type(item[field]) is not bool:
            raise ProofV3Error(f"{field} must be a boolean")
    return CacheRelationSpecV3(
        logical_address_abi_id=_json_identifier(
            item["logical_address_abi_id"],
            "logical_address_abi_id",
        ),
        cache_semantics_id=_json_identifier(
            item["cache_semantics_id"],
            "cache_semantics_id",
        ),
        retention_lease_abi_id=_json_identifier(
            item["retention_lease_abi_id"],
            "retention_lease_abi_id",
        ),
        page_token_count=_json_u32(
            item["page_token_count"],
            "page_token_count",
            positive=True,
        ),
        max_cache_tokens=_json_u32(
            item["max_cache_tokens"],
            "max_cache_tokens",
            positive=True,
        ),
        layer_caches=tuple(
            _layer_cache_from_dict(layer_cache)
            for layer_cache in _list(
                item["layer_caches"],
                "layer_caches",
                MAX_RELATION_NODES,
            )
        ),
        allows_sliding_window=item["allows_sliding_window"],
        allows_prefix_cache_sharing=item["allows_prefix_cache_sharing"],
        allows_speculative_decode=item["allows_speculative_decode"],
        prefix_cache_k_cell_delta_max=(
            _json_u32(
                item["prefix_cache_k_cell_delta_max"],
                "prefix_cache_k_cell_delta_max",
                positive=True,
            )
            if prefix_cache
            else 0
        ),
        prefix_cache_k_row_sq_delta_max=(
            _json_u64(
                item["prefix_cache_k_row_sq_delta_max"],
                "prefix_cache_k_row_sq_delta_max",
                positive=True,
            )
            if prefix_cache
            else 0
        ),
        prefix_cache_v_cell_delta_max=(
            _json_u32(
                item["prefix_cache_v_cell_delta_max"],
                "prefix_cache_v_cell_delta_max",
                positive=True,
            )
            if prefix_cache
            else 0
        ),
        prefix_cache_v_row_sq_delta_max=(
            _json_u64(
                item["prefix_cache_v_row_sq_delta_max"],
                "prefix_cache_v_row_sq_delta_max",
                positive=True,
            )
            if prefix_cache
            else 0
        ),
    )


def _layer_audit_from_dict(value: object) -> LayerAuditPlanV3:
    item = _object(
        value,
        {
            "attention_head_dimension",
            "attention_key_value_head_count",
            "attention_query_head_count",
            "attention_semantics_id",
            "bridge_node_id",
            "layer_index",
            "required_operation_references",
            "transition_node_id",
        },
        "layer audit plan",
    )
    return LayerAuditPlanV3(
        layer_index=_json_i32(
            item["layer_index"], "layer audit layer_index", minimum=0
        ),
        transition_node_id=_json_identifier(
            item["transition_node_id"],
            "layer audit transition_node_id",
        ),
        bridge_node_id=_json_identifier(
            item["bridge_node_id"],
            "layer audit bridge_node_id",
        ),
        required_operation_references=tuple(
            _operation_from_dict(entry)
            for entry in _list(
                item["required_operation_references"],
                "layer audit required_operation_references",
                MAX_OPERATION_REFERENCES,
            )
        ),
        attention_query_head_count=_json_u32(
            item["attention_query_head_count"],
            "attention_query_head_count",
        ),
        attention_key_value_head_count=_json_u32(
            item["attention_key_value_head_count"],
            "attention_key_value_head_count",
        ),
        attention_head_dimension=_json_u32(
            item["attention_head_dimension"],
            "attention_head_dimension",
        ),
        attention_semantics_id=_optional_identifier(
            item["attention_semantics_id"],
            "attention_semantics_id",
        ),
    )


def _audit_policy_from_dict(value: object) -> HardAuditPolicyV3:
    item = _object(
        value,
        {
            "coverage_mode",
            "full_attention_heads_per_layer",
            "minimum_canary_hard_bps",
            "minimum_full_attention_layers",
            "minimum_gdn_layers",
            "minimum_organic_hard_bps",
            "nonce_selection_abi_id",
            "probation_failures",
            "selected_layer_count",
            "selection_abi_id",
            "tier_selection_abi_id",
            "transition_query_rows",
        },
        "audit policy",
    )
    return HardAuditPolicyV3(
        coverage_mode=_json_identifier(item["coverage_mode"], "coverage_mode"),
        nonce_selection_abi_id=_json_identifier(
            item["nonce_selection_abi_id"],
            "nonce_selection_abi_id",
        ),
        tier_selection_abi_id=_json_identifier(
            item["tier_selection_abi_id"],
            "tier_selection_abi_id",
        ),
        selection_abi_id=_json_identifier(
            item["selection_abi_id"],
            "selection_abi_id",
        ),
        minimum_organic_hard_bps=_json_u32(
            item["minimum_organic_hard_bps"],
            "minimum_organic_hard_bps",
        ),
        minimum_canary_hard_bps=_json_u32(
            item["minimum_canary_hard_bps"],
            "minimum_canary_hard_bps",
        ),
        probation_failures=_json_u32(
            item["probation_failures"],
            "probation_failures",
            positive=True,
        ),
        selected_layer_count=_json_u32(
            item["selected_layer_count"],
            "selected_layer_count",
            positive=True,
        ),
        minimum_full_attention_layers=_json_u32(
            item["minimum_full_attention_layers"],
            "minimum_full_attention_layers",
        ),
        minimum_gdn_layers=_json_u32(
            item["minimum_gdn_layers"],
            "minimum_gdn_layers",
        ),
        full_attention_heads_per_layer=_json_u32(
            item["full_attention_heads_per_layer"],
            "full_attention_heads_per_layer",
            positive=True,
        ),
        transition_query_rows=_json_u32(
            item["transition_query_rows"],
            "transition_query_rows",
            positive=True,
        ),
    )


def execution_relation_spec_from_dict(value: object) -> ExecutionRelationSpecV3:
    """Parse an exact relation-spec JSON object and re-run all invariants."""

    item = _object(
        value,
        {
            "audit_policy",
            "cache",
            "capture_abi_digest",
            "constraint_system_digest",
            "decode_chunk_tokens",
            "dimensions",
            "final_token_tensor_id",
            "layer_audits",
            "nodes",
            "prefill_chunk_tokens",
            "recursive_accumulator_abi_id",
            "recursive_accumulator_digest",
            "registered_operations",
            "relation_abi_id",
            "relation_spec_version",
            "request_tensor_id",
            "sampler_abi_id",
            "sampler_abi_digest",
            "quantization_binding_digest",
            "sequence_domain",
            "static_bindings",
            "static_table_bindings",
            "tensors",
            "tokenizer_binding_digest",
            "tolerances",
        },
        "relation spec",
    )
    return ExecutionRelationSpecV3(
        relation_abi_id=_json_identifier(item["relation_abi_id"], "relation_abi_id"),
        capture_abi_digest=_json_hex(item["capture_abi_digest"], "capture_abi_digest"),
        constraint_system_digest=_json_hex(
            item["constraint_system_digest"],
            "constraint_system_digest",
        ),
        tokenizer_binding_digest=_json_hex(
            item["tokenizer_binding_digest"],
            "tokenizer_binding_digest",
        ),
        sampler_abi_id=_json_identifier(item["sampler_abi_id"], "sampler_abi_id"),
        sampler_abi_digest=_json_hex(item["sampler_abi_digest"], "sampler_abi_digest"),
        quantization_binding_digest=_json_hex(
            item["quantization_binding_digest"],
            "quantization_binding_digest",
        ),
        recursive_accumulator_abi_id=_json_identifier(
            item["recursive_accumulator_abi_id"],
            "recursive_accumulator_abi_id",
        ),
        recursive_accumulator_digest=_json_hex(
            item["recursive_accumulator_digest"],
            "recursive_accumulator_digest",
        ),
        prefill_chunk_tokens=_json_u32(
            item["prefill_chunk_tokens"],
            "prefill_chunk_tokens",
            positive=True,
        ),
        decode_chunk_tokens=_json_u32(
            item["decode_chunk_tokens"],
            "decode_chunk_tokens",
            positive=True,
        ),
        request_tensor_id=_json_identifier(
            item["request_tensor_id"],
            "request_tensor_id",
        ),
        final_token_tensor_id=_json_identifier(
            item["final_token_tensor_id"],
            "final_token_tensor_id",
        ),
        sequence_domain=_sequence_domain_from_dict(item["sequence_domain"]),
        dimensions=tuple(
            _dimension_from_dict(entry)
            for entry in _list(
                item["dimensions"],
                "dimensions",
                MAX_DECLARED_DIMENSIONS,
            )
        ),
        tensors=tuple(
            _tensor_from_dict(entry)
            for entry in _list(item["tensors"], "tensors", MAX_TENSORS)
        ),
        registered_operations=tuple(
            _operation_from_dict(entry)
            for entry in _list(
                item["registered_operations"],
                "registered_operations",
                MAX_OPERATION_REFERENCES,
            )
        ),
        static_bindings=tuple(
            _binding_from_dict(entry)
            for entry in _list(
                item["static_bindings"],
                "static_bindings",
                MAX_STATIC_BINDINGS,
            )
        ),
        static_table_bindings=tuple(
            _static_table_binding_from_dict(entry)
            for entry in _list(
                item["static_table_bindings"],
                "static_table_bindings",
                MAX_STATIC_TABLE_BINDINGS,
            )
        ),
        tolerances=tuple(
            _tolerance_from_dict(entry)
            for entry in _list(
                item["tolerances"],
                "tolerances",
                MAX_TOLERANCE_PROFILES,
            )
        ),
        nodes=tuple(
            _node_from_dict(entry)
            for entry in _list(item["nodes"], "nodes", MAX_RELATION_NODES)
        ),
        layer_audits=tuple(
            _layer_audit_from_dict(entry)
            for entry in _list(
                item["layer_audits"],
                "layer_audits",
                MAX_RELATION_NODES,
            )
        ),
        cache=_cache_from_dict(item["cache"]),
        audit_policy=_audit_policy_from_dict(item["audit_policy"]),
        relation_spec_version=_json_u32(
            item["relation_spec_version"],
            "relation_spec_version",
            positive=True,
        ),
    )


def validate_registered_operations_against_manifest_v3(
    profile_operations: Sequence[RegisteredOperationReferenceV3],
    *,
    manifest_digest: bytes,
    expected_manifest_digest: bytes,
    manifest_operations: Sequence[object],
) -> None:
    """Check exact PCS-operation coverage against an already authenticated manifest.

    The caller owns manifest parsing/signature verification.  This helper only
    accepts objects exposing ``layer``, ``operation_id``, and ``expert_id`` and
    deliberately does not infer bridge/cache semantics from untyped metadata.
    A qualified adapter must still validate all static parameter bindings.
    """

    if _fixed32(manifest_digest, "manifest_digest") != _fixed32(
        expected_manifest_digest,
        "expected_manifest_digest",
    ):
        raise ProofV3VerificationError("execution profile has an unexpected manifest")
    if isinstance(manifest_operations, (str, bytes)) or not isinstance(
        manifest_operations,
        Sequence,
    ):
        raise ProofV3VerificationError("manifest operations have an unexpected type")
    actual: set[RegisteredOperationReferenceV3] = set()
    try:
        for descriptor in manifest_operations:
            operation = RegisteredOperationReferenceV3(
                layer_index=descriptor.layer,
                operation_id=descriptor.operation_id,
                expert_id=descriptor.expert_id,
                rows=descriptor.rows,
                cols=descriptor.cols,
                weight_commitment=descriptor.commitment,
                descriptor_digest=hashlib.sha256(descriptor.canonical_bytes()).digest(),
            )
            if operation in actual:
                raise ProofV3VerificationError("manifest operations are duplicated")
            actual.add(operation)
    except (AttributeError, ProofV3Error, TypeError) as exc:
        raise ProofV3VerificationError("manifest operation is malformed") from exc
    expected = set(profile_operations)
    if not expected or actual != expected:
        raise ProofV3VerificationError(
            "signed relation operations do not exactly cover the static manifest"
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3Error("relation spec JSON has duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ProofV3Error("relation spec JSON constants are not allowed")


__all__ = [
    "ECONOMIC_SHELL_COVERAGE_MODE_V3",
    "GLOBAL_HARD_AUDIT_MIN_FULL_ATTENTION_LAYERS_V3",
    "GLOBAL_HARD_AUDIT_MIN_GDN_LAYERS_V3",
    "GLOBAL_HARD_AUDIT_MIN_HEADS_PER_FULL_ATTENTION_LAYER_V3",
    "GLOBAL_HARD_AUDIT_MIN_LAYERS_V3",
    "GLOBAL_EXECUTION_COVERAGE_MODE_V3",
    "GLOBAL_EXECUTION_RELATION_ABI_V3",
    "GDN_STATE_CACHE_SEMANTICS_V3",
    "GPU_RETENTION_LEASE_ABI_V3",
    "LOGICAL_CACHE_ADDRESS_ABI_V3",
    "LOGICAL_CACHE_TABLES_SEMANTICS_V3",
    "PREFIX_CACHE_LOGICAL_ADDRESS_ABI_V3",
    "PREFIX_CACHE_RETENTION_LEASE_ABI_V3",
    "PREFIX_CACHE_TABLES_SEMANTICS_V3",
    "LOGICAL_KV_ADDRESS_ABI_V3",
    "MAX_RELATION_SPEC_BYTES",
    "MAX_HARD_AUDIT_QUERY_ROWS_V3",
    "PAGED_KV_CACHE_SEMANTICS_V3",
    "POSTCOMMIT_AUDIT_TIER_ABI_V3",
    "POSTCOMMIT_GLOBAL_NONCE_ABI_V3",
    "RELATION_SPEC_FORMAT_VERSION",
    "SEQUENCE_DOMAIN_RELATION_ID_V3",
    "STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3",
    "CacheRelationSpecV3",
    "DeclaredDimensionV3",
    "ExecutionRelationNodeV3",
    "ExecutionRelationSpecV3",
    "HardAuditPolicyV3",
    "LayerAuditPlanV3",
    "LayerCacheRelationV3",
    "RegisteredOperationReferenceV3",
    "RelationToleranceV3",
    "RuntimeHardAuditPolicyV3",
    "SequenceDomainV3",
    "StaticParameterBindingV3",
    "StaticTableBindingV3",
    "TensorDescriptorV3",
    "execution_relation_spec_from_dict",
    "validate_registered_operations_against_manifest_v3",
]
