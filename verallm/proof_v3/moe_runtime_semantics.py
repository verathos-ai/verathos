"""Authenticated sparse-MoE execution semantics for proof-v3.

The artifact describes the logical computation shared by qualified fused-MoE
backends.  Physical kernel choices may change dispatch order or padding size,
but they must preserve the signed logical token/expert map and make every
padding row non-contributing.  Model weights remain in the authenticated
projection inventory; this artifact binds its exact inventory digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError


MOE_RUNTIME_SEMANTICS_VERSION_V3 = 1
MOE_RUNTIME_SEMANTICS_ABI_V3 = "moe.runtime_semantics.sparse.v1"
QWEN35_MOE_ADAPTER_V3 = "fused_moe.qwen3_5.v1"
QWEN35_LOGICAL_PADDING_MULTIPLE_V3 = 16
SOFTMAX_TOPK_ROUTER_V3 = "softmax.all_experts.topk.v1"
SCORE_DESC_EXPERT_ASC_TIEBREAK_V3 = "score_desc.expert_asc.v1"
SELECTED_WEIGHT_RENORMALIZATION_V3 = "selected.sum_to_one.v1"
TOKEN_MAJOR_TOPK_DISPATCH_V3 = "logical.token_major.topk_slots.v1"
MASKED_PADDING_V3 = "physical.padding.masked_no_contribution.v1"
SILU_AND_MUL_EXPERT_V3 = "silu_gate_mul_up.v1"
SIGMOID_SHARED_GATE_V3 = "sigmoid.scalar_gate.v1"
ROUTED_PLUS_SHARED_AGGREGATION_V3 = "routed_weighted_sum_plus_shared.v1"
POST_ATTENTION_RESIDUAL_V3 = "post_attention_residual.moe_add.v1"
LOGICAL_EXPERT_IDS_V3 = "logical_expert_ids.no_eplb.v1"
W13_GATE_THEN_UP_LAYOUT_V3 = "w13.gate_then_up.v1"

_DOMAIN = b"VERATHOS/PROOF_V3/MOE_RUNTIME_SEMANTICS/V1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")
_RUNTIME_ENCODINGS = frozenset({"fp16.v1", "bf16.v1"})
_ROUTER_ENCODINGS = frozenset({"bf16.v1", "fp32.v1"})
_ENCODING_BYTES = {
    "bf16.v1": 2,
    "fp16.v1": 2,
    "fp32.v1": 4,
}

__all__ = [
    "LOGICAL_EXPERT_IDS_V3",
    "MASKED_PADDING_V3",
    "MOE_RUNTIME_SEMANTICS_ABI_V3",
    "MOE_RUNTIME_SEMANTICS_VERSION_V3",
    "MoeLayerRuntimeSemanticsV3",
    "MoeRuntimeSemanticsV3",
    "POST_ATTENTION_RESIDUAL_V3",
    "QWEN35_MOE_ADAPTER_V3",
    "QWEN35_LOGICAL_PADDING_MULTIPLE_V3",
    "ROUTED_PLUS_SHARED_AGGREGATION_V3",
    "SCORE_DESC_EXPERT_ASC_TIEBREAK_V3",
    "SELECTED_WEIGHT_RENORMALIZATION_V3",
    "SIGMOID_SHARED_GATE_V3",
    "SILU_AND_MUL_EXPERT_V3",
    "SOFTMAX_TOPK_ROUTER_V3",
    "TOKEN_MAJOR_TOPK_DISPATCH_V3",
    "W13_GATE_THEN_UP_LAYOUT_V3",
    "dump_moe_runtime_semantics_v3",
    "load_moe_runtime_semantics_v3",
    "moe_runtime_encoding_bytes_v3",
]


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3Error(f"MoE {name} must be a string")
    try:
        valid = _IDENTIFIER.fullmatch(value) is not None
    except (TypeError, UnicodeEncodeError):
        valid = False
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError:
        encoded = b""
        valid = False
    if not valid or not encoded:
        raise ProofV3Error(f"MoE {name} is malformed")
    return value


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
        or value >= 1 << 32
    ):
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"MoE {name} must be a {qualifier}u32")
    return value


def _digest(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32 or value == bytes(32):
        raise ProofV3Error(f"MoE {name} must be a nonzero 32-byte digest")
    return value


def moe_runtime_encoding_bytes_v3(encoding_id: str) -> int:
    """Return the exact encoded bytes per scalar for a qualified MoE row."""

    try:
        return _ENCODING_BYTES[encoding_id]
    except (KeyError, TypeError) as exc:
        raise ProofV3Error("MoE runtime encoding is not qualified") from exc


@dataclass(frozen=True, slots=True)
class MoeLayerRuntimeSemanticsV3:
    """Qualified numeric corridors for one sparse-MoE layer."""

    layer_index: int
    router_logit_atol_q24: int
    routing_weight_atol_q24: int
    expert_output_atol_q24: int
    shared_output_atol_q24: int
    aggregate_output_atol_q24: int
    residual_output_atol_q24: int
    moe_input_atol_q24: int = 0

    def __post_init__(self) -> None:
        _u32(self.layer_index, "layer index")
        for name in (
            "router_logit_atol_q24",
            "routing_weight_atol_q24",
            "expert_output_atol_q24",
            "shared_output_atol_q24",
            "aggregate_output_atol_q24",
            "residual_output_atol_q24",
            "moe_input_atol_q24",
        ):
            value = _u32(getattr(self, name), name)
            if value > 16 * (1 << 24):
                raise ProofV3Error(f"MoE {name} is out of range")

    def to_dict(self) -> dict[str, int]:
        return {
            "aggregate_output_atol_q24": self.aggregate_output_atol_q24,
            "expert_output_atol_q24": self.expert_output_atol_q24,
            "layer_index": self.layer_index,
            "moe_input_atol_q24": self.moe_input_atol_q24,
            "residual_output_atol_q24": self.residual_output_atol_q24,
            "router_logit_atol_q24": self.router_logit_atol_q24,
            "routing_weight_atol_q24": self.routing_weight_atol_q24,
            "shared_output_atol_q24": self.shared_output_atol_q24,
        }


@dataclass(frozen=True, slots=True)
class MoeRuntimeSemanticsV3:
    """Complete signed logical ABI for one sparse fused-MoE profile."""

    adapter_id: str
    num_experts: int
    top_k: int
    hidden_size: int
    expert_intermediate_size: int
    shared_expert_intermediate_size: int
    padding_multiple: int
    expert_inventory_digest: bytes
    layers: tuple[MoeLayerRuntimeSemanticsV3, ...]
    capacity_experts_per_audited_layer: int = 2
    runtime_encoding_id: str = "bf16.v1"
    router_encoding_id: str = "bf16.v1"
    router_id: str = SOFTMAX_TOPK_ROUTER_V3
    topk_tiebreak_id: str = SCORE_DESC_EXPERT_ASC_TIEBREAK_V3
    routing_weight_normalization_id: str = SELECTED_WEIGHT_RENORMALIZATION_V3
    dispatch_id: str = TOKEN_MAJOR_TOPK_DISPATCH_V3
    padding_id: str = MASKED_PADDING_V3
    expert_activation_id: str = SILU_AND_MUL_EXPERT_V3
    shared_gate_id: str = SIGMOID_SHARED_GATE_V3
    aggregation_id: str = ROUTED_PLUS_SHARED_AGGREGATION_V3
    residual_id: str = POST_ATTENTION_RESIDUAL_V3
    expert_id_space_id: str = LOGICAL_EXPERT_IDS_V3
    expert_weight_layout_id: str = W13_GATE_THEN_UP_LAYOUT_V3
    version: int = MOE_RUNTIME_SEMANTICS_VERSION_V3

    def __post_init__(self) -> None:
        if self.version != MOE_RUNTIME_SEMANTICS_VERSION_V3:
            raise ProofV3Error("MoE runtime semantics version is unsupported")
        if _identifier(self.adapter_id, "adapter id") != QWEN35_MOE_ADAPTER_V3:
            raise ProofV3Error("MoE runtime adapter is not qualified")
        numeric = (
            ("expert count", self.num_experts, 2, 4096),
            ("top-k", self.top_k, 1, 64),
            ("hidden size", self.hidden_size, 1, 1 << 20),
            (
                "expert intermediate size",
                self.expert_intermediate_size,
                1,
                1 << 20,
            ),
            (
                "shared expert intermediate size",
                self.shared_expert_intermediate_size,
                1,
                1 << 20,
            ),
            ("padding multiple", self.padding_multiple, 1, 1025),
            (
                "capacity expert count",
                self.capacity_experts_per_audited_layer,
                1,
                65,
            ),
        )
        for name, value, minimum, maximum in numeric:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value < maximum
            ):
                raise ProofV3Error(f"MoE {name} is out of range")
        if self.padding_multiple != QWEN35_LOGICAL_PADDING_MULTIPLE_V3:
            raise ProofV3Error(
                "MoE logical padding multiple is not canonical for the adapter"
            )
        if self.top_k > self.num_experts:
            raise ProofV3Error("MoE top-k exceeds the expert count")
        if self.capacity_experts_per_audited_layer > self.num_experts:
            raise ProofV3Error("MoE capacity sample exceeds the expert count")
        _digest(self.expert_inventory_digest, "expert inventory digest")
        if self.runtime_encoding_id not in _RUNTIME_ENCODINGS:
            raise ProofV3Error("MoE runtime encoding is not qualified")
        if self.router_encoding_id not in _ROUTER_ENCODINGS:
            raise ProofV3Error("MoE router encoding is not qualified")
        expected_ids = {
            "router_id": SOFTMAX_TOPK_ROUTER_V3,
            "topk_tiebreak_id": SCORE_DESC_EXPERT_ASC_TIEBREAK_V3,
            "routing_weight_normalization_id": SELECTED_WEIGHT_RENORMALIZATION_V3,
            "dispatch_id": TOKEN_MAJOR_TOPK_DISPATCH_V3,
            "padding_id": MASKED_PADDING_V3,
            "expert_activation_id": SILU_AND_MUL_EXPERT_V3,
            "shared_gate_id": SIGMOID_SHARED_GATE_V3,
            "aggregation_id": ROUTED_PLUS_SHARED_AGGREGATION_V3,
            "residual_id": POST_ATTENTION_RESIDUAL_V3,
            "expert_id_space_id": LOGICAL_EXPERT_IDS_V3,
            "expert_weight_layout_id": W13_GATE_THEN_UP_LAYOUT_V3,
        }
        for name, expected in expected_ids.items():
            actual = _identifier(getattr(self, name), name)
            if actual != expected:
                raise ProofV3Error(f"MoE {name} is not qualified")
        layers = tuple(self.layers)
        if (
            not layers
            or not all(isinstance(item, MoeLayerRuntimeSemanticsV3) for item in layers)
            or tuple(item.layer_index for item in layers)
            != tuple(sorted({item.layer_index for item in layers}))
        ):
            raise ProofV3Error("MoE layers must be ordered and distinct")
        object.__setattr__(self, "layers", layers)

    def layer_for(self, layer_index: int) -> MoeLayerRuntimeSemanticsV3:
        for item in self.layers:
            if item.layer_index == int(layer_index):
                return item
        raise ProofV3VerificationError(
            f"authenticated MoE semantics have no layer {int(layer_index)}"
        )

    def to_dict(self, *, include_digest: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "aggregation_id": self.aggregation_id,
            "capacity_experts_per_audited_layer": (
                self.capacity_experts_per_audited_layer
            ),
            "dispatch_id": self.dispatch_id,
            "expert_activation_id": self.expert_activation_id,
            "expert_id_space_id": self.expert_id_space_id,
            "expert_intermediate_size": self.expert_intermediate_size,
            "expert_inventory_digest": self.expert_inventory_digest.hex(),
            "expert_weight_layout_id": self.expert_weight_layout_id,
            "hidden_size": self.hidden_size,
            "layers": [item.to_dict() for item in self.layers],
            "num_experts": self.num_experts,
            "padding_multiple": self.padding_multiple,
            "padding_id": self.padding_id,
            "residual_id": self.residual_id,
            "router_encoding_id": self.router_encoding_id,
            "router_id": self.router_id,
            "routing_weight_normalization_id": self.routing_weight_normalization_id,
            "runtime_encoding_id": self.runtime_encoding_id,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "shared_gate_id": self.shared_gate_id,
            "top_k": self.top_k,
            "topk_tiebreak_id": self.topk_tiebreak_id,
            "version": self.version,
        }
        if include_digest:
            result["digest"] = self.digest().hex()
        return result

    def canonical_bytes(self) -> bytes:
        body = json.dumps(
            self.to_dict(include_digest=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return _DOMAIN + body

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


def dump_moe_runtime_semantics_v3(
    semantics: MoeRuntimeSemanticsV3,
) -> dict[str, object]:
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("MoE runtime semantics have an unexpected type")
    return semantics.to_dict(include_digest=True)


def load_moe_runtime_semantics_v3(source) -> MoeRuntimeSemanticsV3:
    if isinstance(source, dict):
        value = source
    elif isinstance(source, bytes):
        try:
            value = json.loads(source.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("MoE runtime semantics JSON is malformed") from exc
    elif isinstance(source, str):
        stripped = source.lstrip()
        try:
            if stripped.startswith("{"):
                value = json.loads(source)
            else:
                with open(source, "rb") as handle:
                    value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("MoE runtime semantics source is malformed") from exc
    else:
        raise ProofV3Error("MoE runtime semantics source is unsupported")
    if not isinstance(value, dict):
        raise ProofV3Error("MoE runtime semantics object is malformed")
    expected_keys = {
        "adapter_id",
        "aggregation_id",
        "capacity_experts_per_audited_layer",
        "digest",
        "dispatch_id",
        "expert_activation_id",
        "expert_id_space_id",
        "expert_intermediate_size",
        "expert_inventory_digest",
        "expert_weight_layout_id",
        "hidden_size",
        "layers",
        "num_experts",
        "padding_multiple",
        "padding_id",
        "residual_id",
        "router_encoding_id",
        "router_id",
        "routing_weight_normalization_id",
        "runtime_encoding_id",
        "shared_expert_intermediate_size",
        "shared_gate_id",
        "top_k",
        "topk_tiebreak_id",
        "version",
    }
    layer_keys = {
        "aggregate_output_atol_q24",
        "expert_output_atol_q24",
        "layer_index",
        "moe_input_atol_q24",
        "residual_output_atol_q24",
        "router_logit_atol_q24",
        "routing_weight_atol_q24",
        "shared_output_atol_q24",
    }
    if set(value) != expected_keys or not isinstance(value.get("layers"), list):
        raise ProofV3Error("MoE runtime semantics object is malformed")
    if any(
        not isinstance(item, dict) or set(item) != layer_keys
        for item in value["layers"]
    ):
        raise ProofV3Error("MoE runtime layer object is malformed")
    try:
        semantics = MoeRuntimeSemanticsV3(
            adapter_id=value["adapter_id"],
            aggregation_id=value["aggregation_id"],
            capacity_experts_per_audited_layer=value[
                "capacity_experts_per_audited_layer"
            ],
            dispatch_id=value["dispatch_id"],
            expert_activation_id=value["expert_activation_id"],
            expert_id_space_id=value["expert_id_space_id"],
            expert_intermediate_size=value["expert_intermediate_size"],
            expert_inventory_digest=bytes.fromhex(value["expert_inventory_digest"]),
            expert_weight_layout_id=value["expert_weight_layout_id"],
            hidden_size=value["hidden_size"],
            layers=tuple(
                MoeLayerRuntimeSemanticsV3(**item) for item in value["layers"]
            ),
            num_experts=value["num_experts"],
            padding_multiple=value["padding_multiple"],
            padding_id=value["padding_id"],
            residual_id=value["residual_id"],
            router_encoding_id=value["router_encoding_id"],
            router_id=value["router_id"],
            routing_weight_normalization_id=value["routing_weight_normalization_id"],
            runtime_encoding_id=value["runtime_encoding_id"],
            shared_expert_intermediate_size=value["shared_expert_intermediate_size"],
            shared_gate_id=value["shared_gate_id"],
            top_k=value["top_k"],
            topk_tiebreak_id=value["topk_tiebreak_id"],
            version=value["version"],
        )
        expected_digest = bytes.fromhex(value["digest"])
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("MoE runtime semantics object is malformed") from exc
    if len(expected_digest) != 32 or semantics.digest() != expected_digest:
        raise ProofV3Error("MoE runtime semantics digest does not match")
    return semantics
