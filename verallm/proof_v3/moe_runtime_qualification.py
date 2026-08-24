"""Validation for calibration-derived sparse-MoE runtime corridors."""

from __future__ import annotations

import math

from verallm.proof_v3.errors import ProofV3Error


MOE_RUNTIME_QUALIFICATION_ABI_V3 = "verathos.proof_v3.moe_runtime_qualification.v1"
MOE_RUNTIME_QUALIFICATION_RELATIONS_V3 = frozenset(
    {
        "moe_input",
        "router_logits",
        "routing_weights",
        "aggregate_output",
        "residual_output",
    }
)

__all__ = [
    "MOE_RUNTIME_QUALIFICATION_ABI_V3",
    "MOE_RUNTIME_QUALIFICATION_RELATIONS_V3",
    "validate_moe_runtime_qualification_v3",
    "validate_qualified_moe_runtime_semantics_v3",
]


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def validate_moe_runtime_qualification_v3(
    value: object,
    *,
    layer_indices: tuple[int, ...],
    calibration_minimum: int,
    heldout_minimum: int,
) -> dict[tuple[int, str], int]:
    """Validate frozen held-out gates and return each signed Q24 corridor."""

    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "abi",
            "coverage_gate_passed",
            "heldout_gate_passed",
            "relations",
            "safety_margin",
        }
        or value.get("abi") != MOE_RUNTIME_QUALIFICATION_ABI_V3
        or value.get("coverage_gate_passed") is not True
        or value.get("heldout_gate_passed") is not True
        or not layer_indices
        or tuple(sorted(set(layer_indices))) != layer_indices
        or not _positive_integer(calibration_minimum)
        or not _positive_integer(heldout_minimum)
        or not _finite_nonnegative(value.get("safety_margin"))
        or float(value["safety_margin"]) < 1.0
    ):
        raise ProofV3Error("sparse-MoE runtime qualification is malformed")
    rows = value["relations"]
    expected = tuple(
        (layer, relation)
        for layer in layer_indices
        for relation in sorted(MOE_RUNTIME_QUALIFICATION_RELATIONS_V3)
    )
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ProofV3Error("sparse-MoE runtime qualification coverage is incomplete")
    parsed: dict[tuple[int, str], int] = {}
    for expected_key, row in zip(expected, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "calibration_maximum_absolute_error",
                "calibration_sample_count",
                "heldout_maximum_absolute_error",
                "heldout_passed",
                "heldout_sample_count",
                "layer",
                "qualified_atol_q24",
                "relation",
            }
            or (row.get("layer"), row.get("relation")) != expected_key
            or not _finite_nonnegative(row.get("calibration_maximum_absolute_error"))
            or not _finite_nonnegative(row.get("heldout_maximum_absolute_error"))
            or not _positive_integer(row.get("calibration_sample_count"))
            or row["calibration_sample_count"] < calibration_minimum
            or not _positive_integer(row.get("heldout_sample_count"))
            or row["heldout_sample_count"] < heldout_minimum
            or row.get("heldout_passed") is not True
            or isinstance(row.get("qualified_atol_q24"), bool)
            or not isinstance(row.get("qualified_atol_q24"), int)
        ):
            raise ProofV3Error("sparse-MoE runtime qualification relation is malformed")
        expected_q24 = math.ceil(
            float(row["calibration_maximum_absolute_error"])
            * float(value["safety_margin"])
            * (1 << 24)
        )
        qualified_q24 = row["qualified_atol_q24"]
        heldout_maximum = float(row["heldout_maximum_absolute_error"])
        if (
            qualified_q24 != expected_q24
            or not 0 <= qualified_q24 <= 16 * (1 << 24)
            or heldout_maximum > qualified_q24 / float(1 << 24)
        ):
            raise ProofV3Error(
                "sparse-MoE runtime qualification did not freeze its "
                "calibration bounds"
            )
        parsed[expected_key] = qualified_q24
    return parsed


def validate_qualified_moe_runtime_semantics_v3(
    value: object,
    *,
    semantics,
    calibration_minimum: int,
    heldout_minimum: int,
) -> None:
    """Require signed semantics to contain exactly the derived corridors."""

    from verallm.proof_v3.moe_runtime_semantics import MoeRuntimeSemanticsV3

    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("sparse-MoE runtime semantics have an unexpected type")
    layer_indices = tuple(layer.layer_index for layer in semantics.layers)
    corridors = validate_moe_runtime_qualification_v3(
        value,
        layer_indices=layer_indices,
        calibration_minimum=calibration_minimum,
        heldout_minimum=heldout_minimum,
    )
    for layer in semantics.layers:
        expected = {
            "aggregate_output_atol_q24": corridors[
                (layer.layer_index, "aggregate_output")
            ],
            "moe_input_atol_q24": corridors[(layer.layer_index, "moe_input")],
            "residual_output_atol_q24": corridors[
                (layer.layer_index, "residual_output")
            ],
            "router_logit_atol_q24": corridors[(layer.layer_index, "router_logits")],
            "routing_weight_atol_q24": corridors[
                (layer.layer_index, "routing_weights")
            ],
        }
        if (
            any(getattr(layer, name) != value for name, value in expected.items())
            or layer.expert_output_atol_q24 != 0
            or layer.shared_output_atol_q24 != 0
        ):
            raise ProofV3Error(
                "sparse-MoE runtime semantics were not calibration derived"
            )
