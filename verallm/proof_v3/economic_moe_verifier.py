"""Authenticate compact sparse-MoE weight openings before numeric replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.economic_moe_wire import (
    EconomicMoeWireV3,
    decode_economic_moe_wire_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.moe_projection_inventory import (
    MOE_EXPERT_DOWN_OPERATION_V3,
    MOE_EXPERT_GATE_UP_OPERATION_V3,
    MOE_ROUTER_OPERATION_V3,
    MOE_SHARED_DOWN_OPERATION_V3,
    MOE_SHARED_GATE_OPERATION_V3,
    MOE_SHARED_GATE_UP_OPERATION_V3,
    moe_projection_name_v3,
    select_capacity_experts_v3,
    select_capacity_rows_v3,
    validate_moe_projection_inventory_v3,
)
from verallm.proof_v3.moe_runtime_semantics import MoeRuntimeSemanticsV3
from verallm.proof_v3.moe_runtime_qualification import (
    MOE_RUNTIME_QUALIFICATION_RELATIONS_V3,
)


# Qualification-only telemetry. When set to a list by an owner-side lifecycle
# harness, numeric corridors record their exact honest error instead of using
# provisional bounds. Production never initializes this sink.
_MOE_RUNTIME_QUALIFICATION_REPORT = None

# Sparse runtime semantics v1 predates a relation-metric identifier.  Preserve
# its historical absolute aggregate corridor for registered predecessors; the
# Qwen3.6 FP8 profile qualifies its chained projection error in authenticated
# projection-cell units so cancellation and activation scale cannot distort
# the frozen held-out gate.
_PROJECTION_NORMALIZED_AGGREGATE_MODEL_IDS_V3 = frozenset(
    {"Qwen/Qwen3.6-35B-A3B-FP8"}
)


@dataclass(frozen=True, slots=True)
class AuthenticatedMoeExpertWeightsV3:
    expert_id: int
    gate_up_i8: bytes
    down_rows_i8: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedMoeLayerWeightsV3:
    layer_index: int
    token_position: int
    padding_multiple: int
    hidden_indices: tuple[int, ...]
    router_i8: bytes
    selected_experts: tuple[AuthenticatedMoeExpertWeightsV3, ...]
    shared_gate_i8: bytes
    shared_gate_up_i8: bytes
    shared_down_rows_i8: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class EconomicMoeReplayResultV3:
    layer_index: int
    token_position: int
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]
    aggregate_values: tuple[float, ...]


__all__ = [
    "AuthenticatedMoeExpertWeightsV3",
    "AuthenticatedMoeLayerWeightsV3",
    "EconomicMoeReplayResultV3",
    "MOE_RUNTIME_QUALIFICATION_RELATIONS_V3",
    "authenticate_economic_moe_openings_v3",
    "verify_economic_moe_numeric_replay_v3",
]


def _full_range(
    *,
    artifacts: EconomicVerifiedArtifactsV3,
    name: str,
    reveal,
    expected_rows: int,
) -> bytes:
    if reveal.first_row != 0 or reveal.row_count != expected_rows:
        raise ProofV3VerificationError(
            f"MoE projection {name!r} does not reveal its complete row range"
        )
    return artifacts.verify_weight_range(name=name, reveal=reveal)


def authenticate_economic_moe_openings_v3(
    *,
    wire: bytes | EconomicMoeWireV3,
    artifacts: EconomicVerifiedArtifactsV3,
    semantics: MoeRuntimeSemanticsV3,
    challenge: EconomicChallengeV3,
    minimum_token_position: int = 0,
) -> tuple[AuthenticatedMoeLayerWeightsV3, ...]:
    """Authenticate exact nonce-selected MoE inventory and return int8 rows."""

    if not isinstance(artifacts, EconomicVerifiedArtifactsV3):
        raise ProofV3VerificationError("MoE artifacts have an unexpected type")
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3VerificationError("MoE semantics have an unexpected type")
    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3VerificationError("MoE challenge has an unexpected type")
    authenticated_semantics = artifacts.moe_runtime_semantics
    if (
        not isinstance(authenticated_semantics, MoeRuntimeSemanticsV3)
        or authenticated_semantics.digest() != semantics.digest()
        or artifacts.manifest.moe_runtime_semantics_digest != semantics.digest()
    ):
        raise ProofV3VerificationError(
            "MoE semantics were not authenticated by the signed manifest"
        )
    try:
        validate_moe_projection_inventory_v3(
            manifest=artifacts.manifest,
            semantics=semantics,
        )
    except ProofV3Error as exc:
        if isinstance(exc, ProofV3VerificationError):
            raise
        raise ProofV3VerificationError(
            "signed MoE projection inventory is not qualified"
        ) from exc
    try:
        parsed = (
            decode_economic_moe_wire_v3(wire)
            if isinstance(wire, bytes)
            else wire
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError(
            "MoE wire failed canonical decoding"
        ) from exc
    if not isinstance(parsed, EconomicMoeWireV3):
        raise ProofV3VerificationError("MoE wire has an unexpected type")
    if parsed.semantics_digest != semantics.digest():
        raise ProofV3VerificationError(
            "MoE wire does not match authenticated runtime semantics"
        )
    semantic_layers = {layer.layer_index for layer in semantics.layers}
    expected_layers = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer in semantic_layers
    )
    if tuple(layer.layer_index for layer in parsed.layers) != expected_layers:
        raise ProofV3VerificationError(
            "MoE wire does not cover the selected sparse layers exactly"
        )
    result = []
    for layer in parsed.layers:
        layer_index = layer.layer_index
        if layer.token_position != challenge.moe_token_position_for(
            layer_index=layer_index,
            minimum_position=minimum_token_position,
        ):
            raise ProofV3VerificationError(
                "MoE token position is not the validator-derived selection"
            )
        hidden = challenge.residual_cols_for(
            layer_index=layer_index,
            hidden_dim=semantics.hidden_size,
        )
        if layer.hidden_indices != hidden:
            raise ProofV3VerificationError(
                "MoE hidden rows are not the validator-derived selection"
            )
        if layer.padding_multiple != semantics.padding_multiple:
            raise ProofV3VerificationError(
                "MoE padding multiple differs from qualification"
            )

        router_name = moe_projection_name_v3(
            layer_index=layer_index,
            operation_id=MOE_ROUTER_OPERATION_V3,
        )
        router = _full_range(
            artifacts=artifacts,
            name=router_name,
            reveal=layer.router_rows,
            expected_rows=semantics.num_experts,
        )
        if len(layer.selected_experts) != semantics.top_k or any(
            expert.expert_id >= semantics.num_experts
            for expert in layer.selected_experts
        ):
            raise ProofV3VerificationError(
                "MoE selected expert inventory is malformed"
            )
        selected = []
        for expert in layer.selected_experts:
            gate_name = moe_projection_name_v3(
                layer_index=layer_index,
                operation_id=MOE_EXPERT_GATE_UP_OPERATION_V3,
                expert_id=expert.expert_id,
            )
            down_name = moe_projection_name_v3(
                layer_index=layer_index,
                operation_id=MOE_EXPERT_DOWN_OPERATION_V3,
                expert_id=expert.expert_id,
            )
            selected.append(
                AuthenticatedMoeExpertWeightsV3(
                    expert_id=expert.expert_id,
                    gate_up_i8=_full_range(
                        artifacts=artifacts,
                        name=gate_name,
                        reveal=expert.gate_up_rows,
                        expected_rows=2 * semantics.expert_intermediate_size,
                    ),
                    down_rows_i8=tuple(
                        artifacts.verify_weight_row(name=down_name, reveal=row)
                        for row in expert.down_rows
                    ),
                )
            )

        shared_gate_name = moe_projection_name_v3(
            layer_index=layer_index,
            operation_id=MOE_SHARED_GATE_OPERATION_V3,
        )
        shared_gate_up_name = moe_projection_name_v3(
            layer_index=layer_index,
            operation_id=MOE_SHARED_GATE_UP_OPERATION_V3,
        )
        shared_down_name = moe_projection_name_v3(
            layer_index=layer_index,
            operation_id=MOE_SHARED_DOWN_OPERATION_V3,
        )
        shared_gate = _full_range(
            artifacts=artifacts,
            name=shared_gate_name,
            reveal=layer.shared_gate_rows,
            expected_rows=1,
        )
        shared_gate_up = _full_range(
            artifacts=artifacts,
            name=shared_gate_up_name,
            reveal=layer.shared_gate_up_rows,
            expected_rows=2 * semantics.shared_expert_intermediate_size,
        )
        shared_down = tuple(
            artifacts.verify_weight_row(name=shared_down_name, reveal=row)
            for row in layer.shared_down_rows
        )

        expected_capacity = tuple(
            sorted(
                select_capacity_experts_v3(
                    nonce=challenge.selection_seed,
                    semantics=semantics,
                    layer_index=layer_index,
                )
            )
        )
        if tuple(item.expert_id for item in layer.capacity_experts) != expected_capacity:
            raise ProofV3VerificationError(
                "MoE capacity experts are not the post-commit selection"
            )
        for capacity in layer.capacity_experts:
            expected_gate_row, expected_down_row = select_capacity_rows_v3(
                nonce=challenge.selection_seed,
                semantics=semantics,
                layer_index=layer_index,
                expert_id=capacity.expert_id,
            )
            if (
                capacity.gate_up_row.row_index != expected_gate_row
                or capacity.down_row.row_index != expected_down_row
            ):
                raise ProofV3VerificationError(
                    "MoE capacity rows are not the post-commit selection"
                )
            artifacts.verify_weight_row(
                name=moe_projection_name_v3(
                    layer_index=layer_index,
                    operation_id=MOE_EXPERT_GATE_UP_OPERATION_V3,
                    expert_id=capacity.expert_id,
                ),
                reveal=capacity.gate_up_row,
            )
            artifacts.verify_weight_row(
                name=moe_projection_name_v3(
                    layer_index=layer_index,
                    operation_id=MOE_EXPERT_DOWN_OPERATION_V3,
                    expert_id=capacity.expert_id,
                ),
                reveal=capacity.down_row,
            )

        result.append(
            AuthenticatedMoeLayerWeightsV3(
                layer_index=layer_index,
                token_position=layer.token_position,
                padding_multiple=layer.padding_multiple,
                hidden_indices=hidden,
                router_i8=router,
                selected_experts=tuple(selected),
                shared_gate_i8=shared_gate,
                shared_gate_up_i8=shared_gate_up,
                shared_down_rows_i8=shared_down,
            )
        )
    return tuple(result)


def _runtime_row(
    *,
    rows: Mapping[str, Mapping[int, bytes]],
    stage_id: str,
    token_position: int,
    encoding_id: str,
    width: int,
):
    try:
        raw = rows[stage_id][token_position]
    except (KeyError, TypeError) as exc:
        raise ProofV3VerificationError(
            f"MoE runtime anchor {stage_id!r} is missing the selected row"
        ) from exc
    if not isinstance(raw, bytes):
        raise ProofV3VerificationError(
            f"MoE runtime anchor {stage_id!r} row is malformed"
        )
    try:
        import numpy as np

        if encoding_id == "fp16.v1":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float64)
        elif encoding_id == "bf16.v1":
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            values = (words << 16).view("<f4").astype(np.float64)
        elif encoding_id == "fp32.v1":
            values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        else:
            raise ProofV3VerificationError(
                "MoE runtime anchor encoding is not qualified"
            )
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            f"MoE runtime anchor {stage_id!r} row is malformed"
        ) from exc
    if int(values.size) != width or not bool(np.isfinite(values).all()):
        raise ProofV3VerificationError(
            f"MoE runtime anchor {stage_id!r} row has invalid values"
        )
    return values


def _matrix_i8(raw: bytes, *, rows: int, cols: int, scale: float, name: str):
    try:
        import numpy as np

        values = np.frombuffer(raw, dtype=np.int8)
        if int(values.size) != rows * cols:
            raise ValueError("wrong element count")
        matrix = values.reshape(rows, cols).astype(np.float64) * float(scale)
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            f"authenticated MoE projection {name!r} has malformed bytes"
        ) from exc
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ProofV3VerificationError(
            f"authenticated MoE projection {name!r} has malformed scale"
        )
    return matrix


def _rows_i8(values, *, rows: int, cols: int, scale: float, name: str):
    try:
        import numpy as np

        if len(values) != rows:
            raise ValueError("wrong row count")
        matrix = np.asarray(values, dtype=np.int8)
        if matrix.shape != (rows, cols):
            raise ValueError("wrong row geometry")
        result = matrix.astype(np.float64) * float(scale)
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            f"authenticated MoE projection {name!r} has malformed rows"
        ) from exc
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ProofV3VerificationError(
            f"authenticated MoE projection {name!r} has malformed scale"
        )
    return result


def _canonical_router(logits, top_k: int):
    try:
        import numpy as np

        row = tuple(float(value) for value in logits)
        selected = tuple(
            sorted(range(len(row)), key=lambda expert: (-row[expert], expert))
        )[:top_k]
        maximum = max(row)
        exponentials = tuple(math.exp(value - maximum) for value in row)
        denominator = math.fsum(exponentials)
        softmax = tuple(value / denominator for value in exponentials)
        selected_total = math.fsum(softmax[expert] for expert in selected)
        routing = np.asarray(
            tuple(softmax[expert] / selected_total for expert in selected),
            dtype=np.float64,
        )
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProofV3VerificationError("MoE router values are malformed") from exc
    if not bool(np.isfinite(routing).all()):
        raise ProofV3VerificationError("MoE routing weights are non-finite")
    return selected, routing


def _routing_weights_for_experts(logits, expert_ids):
    """Normalize router probabilities for an identity-keyed expert set."""

    try:
        import numpy as np

        row = tuple(float(value) for value in logits)
        selected = tuple(int(expert_id) for expert_id in expert_ids)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(expert_id < 0 or expert_id >= len(row) for expert_id in selected)
        ):
            raise ValueError("invalid expert selection")
        maximum = max(row)
        exponentials = tuple(math.exp(value - maximum) for value in row)
        denominator = math.fsum(exponentials)
        softmax = tuple(value / denominator for value in exponentials)
        selected_total = math.fsum(softmax[expert] for expert in selected)
        routing = np.asarray(
            tuple(softmax[expert] / selected_total for expert in selected),
            dtype=np.float64,
        )
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProofV3VerificationError("MoE router values are malformed") from exc
    if not bool(np.isfinite(routing).all()):
        raise ProofV3VerificationError("MoE routing weights are non-finite")
    return routing


def _runtime_precision_cell_bounds_v3(values, *, encoding_id: str):
    """Return the round-to-nearest storage cell for runtime values."""
    try:
        import numpy as np

        from verallm.proof_v3.attention_anchor_binding import (
            _runtime_precision_neighbor_bounds_v3,
        )

        runtime_values = np.asarray(values, dtype=np.float64)
        neighbor_lower, neighbor_upper = (
            _runtime_precision_neighbor_bounds_v3(
                runtime_values,
                encoding_id=encoding_id,
                ulps=1,
            )
        )
        cell_lower = (neighbor_lower + runtime_values) / 2.0
        cell_upper = (neighbor_upper + runtime_values) / 2.0
        if not (
            bool(np.isfinite(cell_lower).all())
            and bool(np.isfinite(cell_upper).all())
        ):
            raise ValueError("non-finite storage cell")
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "MoE runtime precision values are malformed"
        ) from exc
    return cell_lower, cell_upper


def _runtime_precision_interval_delta_v3(
    actual,
    expected_lower,
    expected_upper,
    *,
    encoding_id: str,
):
    """Measure distance between runtime and authenticated input cells."""

    try:
        import numpy as np

        actual_values = np.asarray(actual, dtype=np.float64)
        lower_values = np.asarray(expected_lower, dtype=np.float64)
        upper_values = np.asarray(expected_upper, dtype=np.float64)
        if (
            actual_values.shape != lower_values.shape
            or actual_values.shape != upper_values.shape
            or not bool((lower_values <= upper_values).all())
        ):
            raise ValueError("shape mismatch")
        cell_lower, cell_upper = _runtime_precision_cell_bounds_v3(
            actual_values,
            encoding_id=encoding_id,
        )
        delta = np.maximum(
            np.maximum(
                cell_lower - upper_values,
                lower_values - cell_upper,
            ),
            0.0,
        )
        if not bool(np.isfinite(delta).all()):
            raise ValueError("non-finite error")
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "MoE runtime precision values are malformed"
        ) from exc
    return delta


def _runtime_precision_absolute_delta_v3(
    actual,
    expected,
    *,
    encoding_id: str,
):
    """Measure distance from the authenticated runtime storage cell."""

    return _runtime_precision_interval_delta_v3(
        actual,
        expected,
        expected,
        encoding_id=encoding_id,
    )


def _record_moe_runtime_qualification_v3(
    *,
    layer_index: int,
    relation_id: str,
    delta,
) -> bool:
    """Record exact honest error when the owner qualification sink is active."""

    report = _MOE_RUNTIME_QUALIFICATION_REPORT
    if report is None:
        return False
    if (
        not isinstance(report, list)
        or isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or relation_id not in MOE_RUNTIME_QUALIFICATION_RELATIONS_V3
    ):
        raise ProofV3VerificationError(
            "MoE runtime qualification telemetry is malformed"
        )
    try:
        import numpy as np

        values = np.asarray(delta, dtype=np.float64)
        if values.size <= 0 or not bool(np.isfinite(values).all()):
            raise ValueError("non-finite or empty error")
        maximum = float(np.max(values))
        mean = float(np.mean(values))
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "MoE runtime qualification telemetry is malformed"
        ) from exc
    report.append(
        {
            "cell_count": int(values.size),
            "layer": layer_index,
            "maximum_absolute_error": maximum,
            "mean_absolute_error": mean,
            "relation": relation_id,
        }
    )
    return True


def _require_close(
    actual,
    expected,
    *,
    atol_q24: int,
    name: str,
    layer_index: int,
    relation_id: str,
    encoding_id: str | None = None,
) -> None:
    try:
        import numpy as np

        actual_values = np.asarray(actual)
        expected_values = np.asarray(expected)
        if actual_values.shape != expected_values.shape:
            raise ValueError("shape mismatch")
        delta = (
            np.abs(actual_values - expected_values)
            if encoding_id is None
            else _runtime_precision_absolute_delta_v3(
                actual_values,
                expected_values,
                encoding_id=encoding_id,
            )
        )
        tolerance = int(atol_q24) / float(1 << 24)
        finite = bool(np.isfinite(delta).all())
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(f"MoE {name} values are malformed") from exc
    if not finite:
        raise ProofV3VerificationError(f"MoE {name} values are malformed")
    if _record_moe_runtime_qualification_v3(
        layer_index=layer_index,
        relation_id=relation_id,
        delta=delta,
    ):
        return
    valid = bool((delta <= tolerance).all())
    if not valid:
        raise ProofV3VerificationError(
            f"MoE {name} is outside its signed runtime corridor"
        )


def _require_runtime_interval_close(
    actual,
    expected_lower,
    expected_upper,
    *,
    atol_q24: int,
    name: str,
    layer_index: int,
    relation_id: str,
    encoding_id: str,
) -> None:
    try:
        import numpy as np

        delta = _runtime_precision_interval_delta_v3(
            actual,
            expected_lower,
            expected_upper,
            encoding_id=encoding_id,
        )
        tolerance = int(atol_q24) / float(1 << 24)
        finite = bool(np.isfinite(delta).all())
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(f"MoE {name} values are malformed") from exc
    if not finite:
        raise ProofV3VerificationError(f"MoE {name} values are malformed")
    if _record_moe_runtime_qualification_v3(
        layer_index=layer_index,
        relation_id=relation_id,
        delta=delta,
    ):
        return
    if not bool((delta <= tolerance).all()):
        raise ProofV3VerificationError(
            f"MoE {name} is outside its signed runtime corridor"
        )


def _require_runtime_scaled_close(
    actual,
    expected,
    normalizer,
    *,
    atol_q24: int,
    name: str,
    layer_index: int,
    relation_id: str,
    encoding_id: str,
) -> None:
    """Check storage-cell error in authenticated projection-cell units."""

    try:
        import numpy as np

        actual_values = np.asarray(actual, dtype=np.float64)
        expected_values = np.asarray(expected, dtype=np.float64)
        normalizer_values = np.asarray(normalizer, dtype=np.float64)
        if (
            actual_values.shape != expected_values.shape
            or actual_values.shape != normalizer_values.shape
            or not bool(np.isfinite(normalizer_values).all())
            or not bool((normalizer_values > 0.0).all())
        ):
            raise ValueError("shape mismatch")
        delta = _runtime_precision_absolute_delta_v3(
            actual_values,
            expected_values,
            encoding_id=encoding_id,
        ) / normalizer_values
        tolerance = int(atol_q24) / float(1 << 24)
        finite = bool(np.isfinite(delta).all())
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(f"MoE {name} values are malformed") from exc
    if not finite:
        raise ProofV3VerificationError(f"MoE {name} values are malformed")
    if _record_moe_runtime_qualification_v3(
        layer_index=layer_index,
        relation_id=relation_id,
        delta=delta,
    ):
        return
    if not bool((delta <= tolerance).all()):
        raise ProofV3VerificationError(
            f"MoE {name} is outside its signed runtime corridor"
        )


def _sigmoid(values):
    import numpy as np

    value = np.asarray(values, dtype=np.float64)
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _projection_output_uncertainty_v3(
    *,
    input_values,
    gate_values,
    up_values,
    activated_values,
    down_rows,
    gate_up_scale: float,
    down_scale: float,
):
    """Propagate one authenticated gate/up/down projection corridor.

    Projection roots bind round-to-nearest absmax/127 INT8 weights.  The
    registered FP weight represented by each interior cell therefore has the
    same half-step rounding variance used by the ordinary projection
    corridor.  A first-order SwiGLU propagation keeps each stage's standard
    error local to the actual authenticated input and selected down rows.  The
    stage radii add by the triangle rule; the protocol-fixed coefficient is
    applied after complete routed/shared composition.
    """

    try:
        import numpy as np

        inputs = np.asarray(input_values, dtype=np.float64)
        gates = np.asarray(gate_values, dtype=np.float64)
        ups = np.asarray(up_values, dtype=np.float64)
        activated = np.asarray(activated_values, dtype=np.float64)
        down = np.asarray(down_rows, dtype=np.float64)
        if (
            inputs.ndim != 1
            or gates.ndim != 1
            or ups.shape != gates.shape
            or activated.shape != gates.shape
            or down.ndim != 2
            or down.shape[1] != gates.size
            or not math.isfinite(float(gate_up_scale))
            or not math.isfinite(float(down_scale))
            or gate_up_scale <= 0.0
            or down_scale <= 0.0
        ):
            raise ValueError("invalid projection geometry")
        gate_up_variance = (
            float(gate_up_scale) * float(gate_up_scale) / 12.0
        ) * float(np.dot(inputs, inputs))
        sigmoid_gate = _sigmoid(gates)
        silu_gate = gates * sigmoid_gate
        silu_derivative = sigmoid_gate * (
            1.0 + gates * (1.0 - sigmoid_gate)
        )
        activation_variance = gate_up_variance * (
            (ups * silu_derivative) ** 2 + silu_gate**2
        )
        down_weight_variance = (
            (float(down_scale) * float(down_scale) / 12.0)
            * float(np.dot(activated, activated))
        )
        activation_output_variance = (down * down) @ activation_variance
        result = np.sqrt(down_weight_variance) + np.sqrt(
            activation_output_variance
        )
        if not bool(np.isfinite(result).all()) or bool((result < 0.0).any()):
            raise ValueError("invalid propagated uncertainty")
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "MoE projection uncertainty is malformed"
        ) from exc
    return result


def _projection_sigma_cap_v3() -> float:
    """Protocol-fixed concentration cap for chained projection cells."""

    from verallm.proof_v3.economic_challenge import (
        CORRIDOR_QUANT_COEFF_DEN_V3,
        CORRIDOR_QUANT_COEFF_NUM_V3,
    )

    return CORRIDOR_QUANT_COEFF_NUM_V3 / CORRIDOR_QUANT_COEFF_DEN_V3


def verify_economic_moe_numeric_replay_v3(
    *,
    authenticated_layers,
    artifacts: EconomicVerifiedArtifactsV3,
    semantics: MoeRuntimeSemanticsV3,
    runtime_rows: Mapping[str, Mapping[int, bytes]],
) -> tuple[EconomicMoeReplayResultV3, ...]:
    """Replay bounded sparse-MoE cells from signed weights and runtime roots.

    ``runtime_rows`` must be the already-authenticated result of execution-
    anchor verification. Each sparse layer supplies a complete normalized MoE
    input row, complete router-logit row, and complete mid/aggregate/output
    rows for the one validator-derived token selected in the nested wire.
    """

    layers = tuple(authenticated_layers)
    if (
        not layers
        or not all(isinstance(item, AuthenticatedMoeLayerWeightsV3) for item in layers)
        or not isinstance(artifacts, EconomicVerifiedArtifactsV3)
        or not isinstance(semantics, MoeRuntimeSemanticsV3)
        or not isinstance(runtime_rows, Mapping)
    ):
        raise ProofV3VerificationError("MoE numeric replay inputs are malformed")
    if (
        not isinstance(artifacts.moe_runtime_semantics, MoeRuntimeSemanticsV3)
        or artifacts.moe_runtime_semantics.digest() != semantics.digest()
    ):
        raise ProofV3VerificationError(
            "MoE numeric replay semantics are not authenticated"
        )

    import numpy as np

    hidden = semantics.hidden_size
    intermediate = semantics.expert_intermediate_size
    shared_intermediate = semantics.shared_expert_intermediate_size
    projection_normalized_aggregate = (
        artifacts.manifest.model_id
        in _PROJECTION_NORMALIZED_AGGREGATE_MODEL_IDS_V3
    )
    results = []
    for layer in layers:
        signed = semantics.layer_for(layer.layer_index)
        prefix = f"l{layer.layer_index}"
        token = layer.token_position
        moe_input = _runtime_row(
            rows=runtime_rows,
            stage_id=f"{prefix}.moe_input",
            token_position=token,
            encoding_id=semantics.runtime_encoding_id,
            width=hidden,
        )
        runtime_logits = _runtime_row(
            rows=runtime_rows,
            stage_id=f"{prefix}.moe_router_logits",
            token_position=token,
            encoding_id=semantics.router_encoding_id,
            width=semantics.num_experts,
        )
        mid = _runtime_row(
            rows=runtime_rows,
            stage_id=f"{prefix}.residual_after_attention",
            token_position=token,
            encoding_id=semantics.runtime_encoding_id,
            width=hidden,
        )
        runtime_aggregate = _runtime_row(
            rows=runtime_rows,
            stage_id=f"{prefix}.moe_aggregate_output",
            token_position=token,
            encoding_id=semantics.runtime_encoding_id,
            width=hidden,
        )
        residual_out = _runtime_row(
            rows=runtime_rows,
            stage_id=f"{prefix}.residual_out",
            token_position=token,
            encoding_id=semantics.runtime_encoding_id,
            width=hidden,
        )

        router_name = moe_projection_name_v3(
            layer_index=layer.layer_index,
            operation_id=MOE_ROUTER_OPERATION_V3,
        )
        router = _matrix_i8(
            layer.router_i8,
            rows=semantics.num_experts,
            cols=hidden,
            scale=artifacts.scale_for(router_name),
            name=router_name,
        )
        surrogate_logits = router @ moe_input
        _require_close(
            runtime_logits,
            surrogate_logits,
            atol_q24=signed.router_logit_atol_q24,
            name="router logits",
            layer_index=layer.layer_index,
            relation_id="router_logits",
            encoding_id=semantics.router_encoding_id,
        )
        selected_ids, routing_weights = _canonical_router(
            runtime_logits,
            semantics.top_k,
        )
        wire_ids = tuple(item.expert_id for item in layer.selected_experts)
        if selected_ids != wire_ids:
            raise ProofV3VerificationError(
                "MoE selected experts do not match the authenticated router"
            )
        # The signed int8 router is an approximation of the authenticated
        # runtime projection. Close or tied logits can therefore rank the same
        # boundary experts differently. Membership is defined exactly by the
        # pre-nonce runtime row; compare the surrogate probabilities for those
        # same expert identities instead of recomputing membership from it.
        surrogate_routing = _routing_weights_for_experts(
            surrogate_logits,
            selected_ids,
        )
        _require_close(
            routing_weights,
            surrogate_routing,
            atol_q24=signed.routing_weight_atol_q24,
            name="routing weights",
            layer_index=layer.layer_index,
            relation_id="routing_weights",
        )

        hidden_indices = np.asarray(layer.hidden_indices, dtype=np.int64)
        routed = np.zeros(len(layer.hidden_indices), dtype=np.float64)
        routed_uncertainty = (
            np.zeros(len(layer.hidden_indices), dtype=np.float64)
            if projection_normalized_aggregate
            else None
        )
        for slot, expert in enumerate(layer.selected_experts):
            gate_name = moe_projection_name_v3(
                layer_index=layer.layer_index,
                operation_id=MOE_EXPERT_GATE_UP_OPERATION_V3,
                expert_id=expert.expert_id,
            )
            down_name = moe_projection_name_v3(
                layer_index=layer.layer_index,
                operation_id=MOE_EXPERT_DOWN_OPERATION_V3,
                expert_id=expert.expert_id,
            )
            gate_up_scale = artifacts.scale_for(gate_name)
            gate_up_weights = _matrix_i8(
                expert.gate_up_i8,
                rows=2 * intermediate,
                cols=hidden,
                scale=gate_up_scale,
                name=gate_name,
            )
            gate_up = gate_up_weights @ moe_input
            gate = gate_up[:intermediate]
            up = gate_up[intermediate:]
            activated = (gate * _sigmoid(gate)) * up
            down_scale = artifacts.scale_for(down_name)
            down_weights = _rows_i8(
                expert.down_rows_i8,
                rows=len(layer.hidden_indices),
                cols=intermediate,
                scale=down_scale,
                name=down_name,
            )
            down = down_weights @ activated
            route_weight = float(routing_weights[slot])
            routed += route_weight * down
            if routed_uncertainty is not None:
                routed_uncertainty += abs(route_weight) * (
                    _projection_output_uncertainty_v3(
                        input_values=moe_input,
                        gate_values=gate,
                        up_values=up,
                        activated_values=activated,
                        down_rows=down_weights,
                        gate_up_scale=gate_up_scale,
                        down_scale=down_scale,
                    )
                )

        shared_gate_name = moe_projection_name_v3(
            layer_index=layer.layer_index,
            operation_id=MOE_SHARED_GATE_OPERATION_V3,
        )
        shared_gate_up_name = moe_projection_name_v3(
            layer_index=layer.layer_index,
            operation_id=MOE_SHARED_GATE_UP_OPERATION_V3,
        )
        shared_down_name = moe_projection_name_v3(
            layer_index=layer.layer_index,
            operation_id=MOE_SHARED_DOWN_OPERATION_V3,
        )
        shared_gate_scale = artifacts.scale_for(shared_gate_name)
        shared_gate = float(
            _matrix_i8(
                layer.shared_gate_i8,
                rows=1,
                cols=hidden,
                scale=shared_gate_scale,
                name=shared_gate_name,
            )[0]
            @ moe_input
        )
        shared_gate_up_scale = artifacts.scale_for(shared_gate_up_name)
        shared_gate_up_weights = _matrix_i8(
            layer.shared_gate_up_i8,
            rows=2 * shared_intermediate,
            cols=hidden,
            scale=shared_gate_up_scale,
            name=shared_gate_up_name,
        )
        shared_gate_up = shared_gate_up_weights @ moe_input
        shared_gate_values = shared_gate_up[:shared_intermediate]
        shared_up_values = shared_gate_up[shared_intermediate:]
        shared_activated = (
            shared_gate_values * _sigmoid(shared_gate_values)
        ) * shared_up_values
        shared_down_scale = artifacts.scale_for(shared_down_name)
        shared_down_weights = _rows_i8(
            layer.shared_down_rows_i8,
            rows=len(layer.hidden_indices),
            cols=shared_intermediate,
            scale=shared_down_scale,
            name=shared_down_name,
        )
        shared = shared_down_weights @ shared_activated
        shared_uncertainty = (
            _projection_output_uncertainty_v3(
                input_values=moe_input,
                gate_values=shared_gate_values,
                up_values=shared_up_values,
                activated_values=shared_activated,
                down_rows=shared_down_weights,
                gate_up_scale=shared_gate_up_scale,
                down_scale=shared_down_scale,
            )
            if projection_normalized_aggregate
            else None
        )
        shared_gate_probability = float(_sigmoid((shared_gate,))[0])
        shared_gate_variance = (
            shared_gate_scale * shared_gate_scale / 12.0
        ) * float(np.dot(moe_input, moe_input))
        if shared_uncertainty is not None:
            shared_uncertainty = (
                shared_gate_probability * shared_uncertainty
                + np.abs(shared)
                * shared_gate_probability
                * (1.0 - shared_gate_probability)
                * math.sqrt(shared_gate_variance)
            )
        shared *= shared_gate_probability
        aggregate = routed + shared
        if projection_normalized_aggregate:
            aggregate_standard_uncertainty = (
                routed_uncertainty + shared_uncertainty
            )
            if not bool(
                np.isfinite(aggregate_standard_uncertainty).all()
            ) or bool((aggregate_standard_uncertainty <= 0.0).any()):
                raise ProofV3VerificationError(
                    "MoE aggregate projection uncertainty is malformed"
                )
            _require_runtime_scaled_close(
                runtime_aggregate[hidden_indices],
                aggregate,
                _projection_sigma_cap_v3()
                * aggregate_standard_uncertainty,
                atol_q24=signed.aggregate_output_atol_q24,
                name="aggregate output",
                layer_index=layer.layer_index,
                relation_id="aggregate_output",
                encoding_id=semantics.runtime_encoding_id,
            )
        else:
            _require_close(
                runtime_aggregate[hidden_indices],
                aggregate,
                atol_q24=signed.aggregate_output_atol_q24,
                name="aggregate output",
                layer_index=layer.layer_index,
                relation_id="aggregate_output",
                encoding_id=semantics.runtime_encoding_id,
            )
        mid_lower, mid_upper = _runtime_precision_cell_bounds_v3(
            mid[hidden_indices],
            encoding_id=semantics.runtime_encoding_id,
        )
        aggregate_lower, aggregate_upper = _runtime_precision_cell_bounds_v3(
            runtime_aggregate[hidden_indices],
            encoding_id=semantics.runtime_encoding_id,
        )
        _require_runtime_interval_close(
            residual_out[hidden_indices],
            mid_lower + aggregate_lower,
            mid_upper + aggregate_upper,
            atol_q24=signed.residual_output_atol_q24,
            name="residual output",
            layer_index=layer.layer_index,
            relation_id="residual_output",
            encoding_id=semantics.runtime_encoding_id,
        )
        results.append(
            EconomicMoeReplayResultV3(
                layer_index=layer.layer_index,
                token_position=token,
                selected_experts=selected_ids,
                routing_weights=tuple(float(value) for value in routing_weights),
                aggregate_values=tuple(float(value) for value in aggregate),
            )
        )
    return tuple(results)
