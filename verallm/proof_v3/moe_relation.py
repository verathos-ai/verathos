"""Reference sparse-MoE routing, dispatch and residual relation.

This is the CPU/reference statement used by qualification and negative tests.
Production proof adapters authenticate the same inputs and selected projection
operations through compact commitments rather than shipping full weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.moe_runtime_semantics import (
    MoeLayerRuntimeSemanticsV3,
    MoeRuntimeSemanticsV3,
)


@dataclass(frozen=True, slots=True, order=True)
class MoeDispatchRowV3:
    expert_id: int
    token_index: int
    slot_index: int

    @property
    def padded(self) -> bool:
        return self.token_index == -1


@dataclass(frozen=True, slots=True)
class MoeDispatchPlanV3:
    padding_multiple: int
    rows: tuple[MoeDispatchRowV3, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.padding_multiple, bool)
            or not isinstance(self.padding_multiple, int)
            or not 1 <= self.padding_multiple <= 1024
            or not isinstance(self.rows, tuple)
            or any(not isinstance(row, MoeDispatchRowV3) for row in self.rows)
        ):
            raise ProofV3Error("MoE dispatch plan is malformed")
        seen_padding: set[int] = set()
        previous_expert = -1
        for row in self.rows:
            if (
                isinstance(row.expert_id, bool)
                or not isinstance(row.expert_id, int)
                or not 0 <= row.expert_id < 1 << 32
                or row.expert_id < previous_expert
                or (row.token_index == -1) != (row.slot_index == -1)
                or row.token_index < -1
                or row.slot_index < -1
                or (row.expert_id in seen_padding and not row.padded)
            ):
                raise ProofV3Error("MoE dispatch rows are not canonical")
            if row.padded:
                seen_padding.add(row.expert_id)
            previous_expert = row.expert_id


@dataclass(frozen=True, slots=True)
class MoeReferenceWeightsV3:
    router: torch.Tensor
    expert_gate_up: tuple[torch.Tensor, ...]
    expert_down: tuple[torch.Tensor, ...]
    shared_gate: torch.Tensor
    shared_gate_up: torch.Tensor
    shared_down: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoeRuntimeWitnessV3:
    router_logits: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    dispatch: MoeDispatchPlanV3
    expert_slot_outputs: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    aggregate_output: torch.Tensor
    residual_output: torch.Tensor


__all__ = [
    "MoeDispatchPlanV3",
    "MoeDispatchRowV3",
    "MoeReferenceWeightsV3",
    "MoeRuntimeWitnessV3",
    "build_moe_dispatch_plan_v3",
    "canonical_qwen_router_v3",
    "evaluate_qwen_moe_reference_v3",
    "verify_moe_runtime_witness_v3",
]


def _matrix(value: object, name: str, shape: tuple[int, int]) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ProofV3Error(f"MoE {name} has malformed geometry or values")
    return value.detach().to(device="cpu", dtype=torch.float64)


def canonical_qwen_router_v3(
    router_logits: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply softmax, score-desc/expert-asc top-k and selected renormalization."""

    if (
        not isinstance(router_logits, torch.Tensor)
        or router_logits.ndim != 2
        or not router_logits.is_floating_point()
        or not bool(torch.isfinite(router_logits).all())
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= int(router_logits.shape[1])
    ):
        raise ProofV3Error("MoE router logits or top-k are malformed")
    logits = router_logits.detach().to(device="cpu", dtype=torch.float64)
    ids = torch.empty((logits.shape[0], top_k), dtype=torch.int64)
    weights = torch.empty_like(ids, dtype=torch.float64)
    for token in range(int(logits.shape[0])):
        row = tuple(float(value) for value in logits[token])
        selected = tuple(
            sorted(range(len(row)), key=lambda expert: (-row[expert], expert))
        )[:top_k]
        maximum = max(row)
        all_exp = tuple(math.exp(value - maximum) for value in row)
        denominator = math.fsum(all_exp)
        softmax = tuple(value / denominator for value in all_exp)
        selected_total = math.fsum(softmax[expert] for expert in selected)
        ids[token] = torch.tensor(selected, dtype=torch.int64)
        weights[token] = torch.tensor(
            tuple(softmax[expert] / selected_total for expert in selected),
            dtype=torch.float64,
        )
    return ids, weights


def build_moe_dispatch_plan_v3(
    *,
    selected_experts: torch.Tensor,
    num_experts: int,
    padding_multiple: int,
) -> MoeDispatchPlanV3:
    if (
        not isinstance(selected_experts, torch.Tensor)
        or selected_experts.ndim != 2
        or selected_experts.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64}
        or isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or num_experts <= 0
        or isinstance(padding_multiple, bool)
        or not isinstance(padding_multiple, int)
        or not 1 <= padding_multiple <= 1024
    ):
        raise ProofV3Error("MoE dispatch inputs are malformed")
    selected = selected_experts.detach().to(device="cpu", dtype=torch.int64)
    if bool((selected < 0).any()) or bool((selected >= num_experts).any()):
        raise ProofV3Error("MoE dispatch expert id is out of range")
    if any(
        len(set(int(value) for value in row)) != int(row.numel()) for row in selected
    ):
        raise ProofV3Error("MoE token routes to a duplicate expert")
    by_expert: list[list[tuple[int, int]]] = [[] for _ in range(num_experts)]
    for token in range(int(selected.shape[0])):
        for slot in range(int(selected.shape[1])):
            by_expert[int(selected[token, slot])].append((token, slot))
    rows: list[MoeDispatchRowV3] = []
    for expert, routes in enumerate(by_expert):
        rows.extend(MoeDispatchRowV3(expert, token, slot) for token, slot in routes)
        padding = (-len(routes)) % padding_multiple
        rows.extend(MoeDispatchRowV3(expert, -1, -1) for _ in range(padding))
    return MoeDispatchPlanV3(padding_multiple, tuple(rows))


def _validate_weights(
    weights: MoeReferenceWeightsV3,
    semantics: MoeRuntimeSemanticsV3,
) -> MoeReferenceWeightsV3:
    if not isinstance(weights, MoeReferenceWeightsV3):
        raise ProofV3Error("MoE reference weights have an unexpected type")
    hidden = semantics.hidden_size
    intermediate = semantics.expert_intermediate_size
    shared_intermediate = semantics.shared_expert_intermediate_size
    if (
        len(weights.expert_gate_up) != semantics.num_experts
        or len(weights.expert_down) != semantics.num_experts
    ):
        raise ProofV3Error("MoE reference weights do not cover every expert")
    return MoeReferenceWeightsV3(
        router=_matrix(weights.router, "router", (semantics.num_experts, hidden)),
        expert_gate_up=tuple(
            _matrix(value, f"expert {index} gate-up", (2 * intermediate, hidden))
            for index, value in enumerate(weights.expert_gate_up)
        ),
        expert_down=tuple(
            _matrix(value, f"expert {index} down", (hidden, intermediate))
            for index, value in enumerate(weights.expert_down)
        ),
        shared_gate=_matrix(weights.shared_gate, "shared gate", (1, hidden)),
        shared_gate_up=_matrix(
            weights.shared_gate_up,
            "shared gate-up",
            (2 * shared_intermediate, hidden),
        ),
        shared_down=_matrix(
            weights.shared_down,
            "shared down",
            (hidden, shared_intermediate),
        ),
    )


def _silu_and_mul(gate_up: torch.Tensor, intermediate: int) -> torch.Tensor:
    gate = gate_up[..., :intermediate]
    up = gate_up[..., intermediate:]
    return torch.nn.functional.silu(gate) * up


def evaluate_qwen_moe_reference_v3(
    *,
    moe_input: torch.Tensor,
    residual_after_attention: torch.Tensor,
    weights: MoeReferenceWeightsV3,
    semantics: MoeRuntimeSemanticsV3,
    layer_index: int,
) -> MoeRuntimeWitnessV3:
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("MoE semantics have an unexpected type")
    semantics.layer_for(layer_index)
    hidden = semantics.hidden_size
    if not isinstance(moe_input, torch.Tensor) or moe_input.ndim != 2:
        raise ProofV3Error("MoE input has malformed geometry or values")
    x = _matrix(moe_input, "input", (int(moe_input.shape[0]), hidden))
    residual = _matrix(
        residual_after_attention,
        "residual after attention",
        tuple(x.shape),
    )
    trusted = _validate_weights(weights, semantics)
    router_logits = x @ trusted.router.T
    selected, routing_weights = canonical_qwen_router_v3(
        router_logits,
        top_k=semantics.top_k,
    )
    dispatch = build_moe_dispatch_plan_v3(
        selected_experts=selected,
        num_experts=semantics.num_experts,
        padding_multiple=semantics.padding_multiple,
    )
    slot_outputs = torch.empty(
        (x.shape[0], semantics.top_k, hidden),
        dtype=torch.float64,
    )
    for token in range(int(x.shape[0])):
        for slot in range(semantics.top_k):
            expert = int(selected[token, slot])
            gate_up = x[token] @ trusted.expert_gate_up[expert].T
            activated = _silu_and_mul(gate_up, semantics.expert_intermediate_size)
            slot_outputs[token, slot] = activated @ trusted.expert_down[expert].T
    routed = torch.sum(slot_outputs * routing_weights.unsqueeze(-1), dim=1)
    shared_gate_up = x @ trusted.shared_gate_up.T
    shared_activated = _silu_and_mul(
        shared_gate_up,
        semantics.shared_expert_intermediate_size,
    )
    shared = shared_activated @ trusted.shared_down.T
    shared = shared * torch.sigmoid(x @ trusted.shared_gate.T)
    aggregate = routed + shared
    return MoeRuntimeWitnessV3(
        router_logits=router_logits,
        selected_experts=selected,
        routing_weights=routing_weights,
        dispatch=dispatch,
        expert_slot_outputs=slot_outputs,
        routed_output=routed,
        shared_output=shared,
        aggregate_output=aggregate,
        residual_output=residual + aggregate,
    )


def _assert_close(
    actual: object,
    expected: torch.Tensor,
    *,
    atol_q24: int,
    name: str,
) -> None:
    if (
        not isinstance(actual, torch.Tensor)
        or tuple(actual.shape) != tuple(expected.shape)
        or not actual.is_floating_point()
        or not bool(torch.isfinite(actual).all())
    ):
        raise ProofV3VerificationError(f"MoE {name} witness is malformed")
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    tolerance = int(atol_q24) / float(1 << 24)
    if not bool(torch.all(torch.abs(actual64 - expected) <= tolerance)):
        raise ProofV3VerificationError(f"MoE {name} relation failed")


def verify_moe_runtime_witness_v3(
    *,
    witness: MoeRuntimeWitnessV3,
    moe_input: torch.Tensor,
    residual_after_attention: torch.Tensor,
    weights: MoeReferenceWeightsV3,
    semantics: MoeRuntimeSemanticsV3,
    layer_index: int,
) -> None:
    if not isinstance(witness, MoeRuntimeWitnessV3):
        raise ProofV3VerificationError("MoE runtime witness has an unexpected type")
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3VerificationError("MoE semantics have an unexpected type")
    if witness.dispatch.padding_multiple != semantics.padding_multiple:
        raise ProofV3VerificationError(
            "MoE dispatch padding differs from authenticated semantics"
        )
    expected = evaluate_qwen_moe_reference_v3(
        moe_input=moe_input,
        residual_after_attention=residual_after_attention,
        weights=weights,
        semantics=semantics,
        layer_index=layer_index,
    )
    layer: MoeLayerRuntimeSemanticsV3 = semantics.layer_for(layer_index)
    if (
        not isinstance(witness.selected_experts, torch.Tensor)
        or witness.selected_experts.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64}
        or not torch.equal(
            witness.selected_experts.detach().to(device="cpu", dtype=torch.int64),
            expected.selected_experts,
        )
    ):
        raise ProofV3VerificationError("MoE selected-expert relation failed")
    if witness.dispatch != expected.dispatch:
        raise ProofV3VerificationError("MoE dispatch or padding relation failed")
    for name, actual, expected_value, tolerance in (
        (
            "router logits",
            witness.router_logits,
            expected.router_logits,
            layer.router_logit_atol_q24,
        ),
        (
            "routing weights",
            witness.routing_weights,
            expected.routing_weights,
            layer.routing_weight_atol_q24,
        ),
        (
            "expert outputs",
            witness.expert_slot_outputs,
            expected.expert_slot_outputs,
            layer.expert_output_atol_q24,
        ),
        (
            "routed aggregation",
            witness.routed_output,
            expected.routed_output,
            layer.aggregate_output_atol_q24,
        ),
        (
            "shared output",
            witness.shared_output,
            expected.shared_output,
            layer.shared_output_atol_q24,
        ),
        (
            "aggregate output",
            witness.aggregate_output,
            expected.aggregate_output,
            layer.aggregate_output_atol_q24,
        ),
        (
            "residual output",
            witness.residual_output,
            expected.residual_output,
            layer.residual_output_atol_q24,
        ),
    ):
        _assert_close(actual, expected_value, atol_q24=tolerance, name=name)
