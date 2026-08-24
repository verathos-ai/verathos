"""Canonical projection-corridor negative qualification records.

The signed corridor bounds admit honest runtime quantization error.  A release
qualification must therefore show that the same frozen bounds reject the
model-independent negative classes exercised by the hard verifier.  This
module validates the small, content-addressed result record; it does not run
the qualification cases or trust miner-provided evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

from verallm.proof_v3.errors import ProofV3Error

PROJECTION_CORRIDOR_QUALIFICATION_ABI_V2 = (
    "verathos.proof_v3.projection_corridor_qualification.v2"
)
PROJECTION_CORRIDOR_HONEST_QUALIFICATION_ABI_V2 = (
    "verathos.proof_v3.projection_corridor_honest_qualification.v2"
)
PROJECTION_CORRIDOR_QUALIFICATION_ABI_V3 = (
    "verathos.proof_v3.projection_corridor_qualification.v3"
)
PROJECTION_CORRIDOR_NEGATIVE_GATE_ABI_V3 = (
    "verathos.proof_v3.projection_corridor_negative_gate.v1"
)

MINIMUM_CORRIDOR_CALIBRATION_LAYER_SELECTIONS_V3 = 2
MINIMUM_CORRIDOR_HELDOUT_LAYER_SELECTIONS_V3 = 1
MINIMUM_CORRIDOR_LENGTH_SAMPLES_V3 = 2

_COMMON_NEGATIVE_CASES_V3 = frozenset(
    (
        "registered_weight_mismatch",
        "detached_runtime_trace",
        "zero_runtime_state",
        "wrong_terminal_output",
    )
)
_FULL_ATTENTION_NEGATIVE_CASE_V3 = (
    "self_consistent_full_attention_substitute"
)
_GDN_NEGATIVE_CASE_V3 = "self_consistent_gdn_substitute"
_MLP_ACTIVATION_NEGATIVE_CASES_V3 = frozenset(
    (
        "mlp_activation_down_mutation",
        "mlp_activation_gate_mutation",
        "mlp_activation_up_mutation",
        "saturated_mlp_activation_substitute",
    )
)
_MOE_RUNTIME_NEGATIVE_CASES_V3 = frozenset(
    (
        "moe_aggregate_output_mutation",
        "moe_expert_down_mutation",
        "moe_expert_gate_up_mutation",
        "moe_padding_substitution",
        "moe_postcommit_capacity_substitution",
        "moe_residual_output_mutation",
        "moe_router_output_mutation",
        "moe_routing_weight_mutation",
        "moe_selected_expert_substitution",
        "moe_shared_expert_mutation",
    )
)
_EVIDENCE_CLASSES_V3 = frozenset(
    (
        "real_model_wire",
        "qualified_relation_fixture",
        "retained_bundle_replay",
    )
)
_CASE_EVIDENCE_CLASSES_V3 = {
    "registered_weight_mismatch": frozenset(("real_model_wire",)),
    "detached_runtime_trace": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    "zero_runtime_state": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    "wrong_terminal_output": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    _FULL_ATTENTION_NEGATIVE_CASE_V3: frozenset(
        ("real_model_wire", "qualified_relation_fixture")
    ),
    _GDN_NEGATIVE_CASE_V3: frozenset(
        ("real_model_wire", "qualified_relation_fixture")
    ),
    **{
        case_id: frozenset(("qualified_relation_fixture",))
        for case_id in _MLP_ACTIVATION_NEGATIVE_CASES_V3
    },
    **{
        case_id: frozenset(("qualified_relation_fixture",))
        for case_id in _MOE_RUNTIME_NEGATIVE_CASES_V3
    },
}


def required_projection_corridor_decode_lengths_v3(
    maximum: int,
) -> tuple[int, ...]:
    """Return canonical octave checkpoints through the admitted decode reach."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ProofV3Error("projection-corridor decode reach is malformed")
    current = min(64, maximum)
    values = [current]
    while current < maximum:
        current = min(maximum, current * 2)
        values.append(current)
    return tuple(values)


def required_projection_corridor_context_lengths_v3(
    maximum: int,
) -> tuple[int, ...]:
    """Return canonical 4x context checkpoints through the admitted reach."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise ProofV3Error("projection-corridor context reach is malformed")
    current = min(2_048, maximum)
    values = [current]
    while current < maximum:
        current = min(maximum, current * 4)
        values.append(current)
    return tuple(values)


def projection_corridor_manifest_layers_v3(
    manifest_entry_names: Iterable[str],
) -> tuple[int, ...]:
    """Extract the complete canonical runtime-layer inventory from operations."""

    layers = set()
    for raw_name in manifest_entry_names:
        name = str(raw_name)
        prefix = name.split(".", 1)[0]
        if len(prefix) < 2 or prefix[0] != "l" or not prefix[1:].isdigit():
            continue
        layers.add(int(prefix[1:]))
    if not layers:
        raise ProofV3Error(
            "projection-corridor manifest layer inventory is empty"
        )
    ordered = tuple(sorted(layers))
    if ordered != tuple(range(ordered[-1] + 1)):
        raise ProofV3Error(
            "projection-corridor manifest layer inventory is not contiguous"
        )
    return ordered


def projection_corridor_dense_mlp_layers_v3(
    manifest_entry_names: Iterable[str],
) -> tuple[int, ...]:
    """Return layers whose activation coverage uses the dense-MLP relation."""

    names = tuple(str(name) for name in manifest_entry_names)
    layers = projection_corridor_manifest_layers_v3(names)
    moe_layers = {
        int(name.split(".", 1)[0][1:])
        for name in names
        if ".moe." in name
        and name.split(".", 1)[0].startswith("l")
        and name.split(".", 1)[0][1:].isdigit()
    }
    return tuple(layer for layer in layers if layer not in moe_layers)


def _canonical_positive_counts(
    value,
    *,
    key_name: str,
    inventory_name: str,
) -> dict[int, int]:
    if not isinstance(value, list) or not value:
        raise ProofV3Error(
            f"projection-corridor {inventory_name} inventory is malformed"
        )
    result = {}
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {key_name, "count"}
        ):
            raise ProofV3Error(
                f"projection-corridor {inventory_name} inventory is malformed"
            )
        key = item[key_name]
        count = item["count"]
        if (
            isinstance(key, bool)
            or not isinstance(key, int)
            or key < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or key in result
        ):
            raise ProofV3Error(
                f"projection-corridor {inventory_name} inventory is malformed"
            )
        result[key] = count
    if tuple(result) != tuple(sorted(result)):
        raise ProofV3Error(
            f"projection-corridor {inventory_name} inventory is not canonical"
        )
    return result


def _canonical_layer_counts(value, *, name: str) -> dict[int, tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ProofV3Error(f"projection-corridor {name} is malformed")
    result = {}
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"layer", "count", "distinct_plan_count"}
        ):
            raise ProofV3Error(f"projection-corridor {name} is malformed")
        layer = item["layer"]
        count = item["count"]
        plan_count = item["distinct_plan_count"]
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or isinstance(plan_count, bool)
            or not isinstance(plan_count, int)
            or not 1 <= plan_count <= count
            or layer in result
        ):
            raise ProofV3Error(f"projection-corridor {name} is malformed")
        result[layer] = (count, plan_count)
    if tuple(result) != tuple(sorted(result)):
        raise ProofV3Error(f"projection-corridor {name} is not canonical")
    return result


def validate_projection_corridor_honest_coverage_v3(
    value,
    *,
    manifest_entry_names: Iterable[str],
) -> None:
    """Validate final-profile layer, nonce, context and decode coverage.

    Calibration observations derive the signed thresholds. Held-out cases are
    checked against those frozen thresholds and therefore cannot silently
    widen them.
    """

    if not isinstance(value, dict):
        raise ProofV3Error("projection-corridor qualification is malformed")
    required = {
        "qualified_decode_tokens",
        "qualified_context_tokens",
        "qualified_decode_token_lengths",
        "qualified_context_token_lengths",
        "minimum_calibration_layer_selections",
        "minimum_heldout_layer_selections",
        "minimum_length_samples",
        "selected_layer_count",
        "calibration_layer_counts",
        "heldout_layer_counts",
        "decode_length_counts",
        "context_length_counts",
        "geometry_counts",
        "maximum_geometry_case_count",
        "honest_coverage_gate_passed",
        "mlp_activation_coverage_gate_passed",
        "mlp_activation_layer_coverage",
        "heldout_gate_passed",
        "heldout_maximum_required_sigma",
        "heldout_maximum_chi2",
        "heldout_maximum_required_fixed_quant_coeff",
        "qualified_corridor_sigma",
        "qualified_corridor_chi2",
        "protocol_fixed_quant_coeff",
        "cases",
    }
    if not required.issubset(value) or value.get(
        "honest_coverage_gate_passed"
    ) is not True or value.get(
        "mlp_activation_coverage_gate_passed"
    ) is not True or value.get("heldout_gate_passed") is not True:
        raise ProofV3Error(
            "projection-corridor honest coverage is incomplete"
        )
    decode_max = value["qualified_decode_tokens"]
    context_max = value["qualified_context_tokens"]
    if (
        isinstance(decode_max, bool)
        or not isinstance(decode_max, int)
        or decode_max < 1
        or isinstance(context_max, bool)
        or not isinstance(context_max, int)
        or context_max < 2
    ):
        raise ProofV3Error("projection-corridor qualified reach is malformed")
    try:
        decode_lengths = tuple(value["qualified_decode_token_lengths"])
        context_lengths = tuple(value["qualified_context_token_lengths"])
    except TypeError as exc:
        raise ProofV3Error(
            "projection-corridor qualified length coverage is malformed"
        ) from exc
    if (
        any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in decode_lengths
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 2
            for item in context_lengths
        )
        or not decode_lengths
        or not context_lengths
        or decode_lengths != tuple(sorted(set(decode_lengths)))
        or context_lengths != tuple(sorted(set(context_lengths)))
        or decode_lengths
        != required_projection_corridor_decode_lengths_v3(decode_max)
        or context_lengths
        != required_projection_corridor_context_lengths_v3(context_max)
    ):
        raise ProofV3Error(
            "projection-corridor qualified length coverage is incomplete"
        )
    calibration_minimum = value["minimum_calibration_layer_selections"]
    heldout_minimum = value["minimum_heldout_layer_selections"]
    length_minimum = value["minimum_length_samples"]
    selected_layer_count = value["selected_layer_count"]
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (calibration_minimum, heldout_minimum, length_minimum)
    ) or (
        calibration_minimum < MINIMUM_CORRIDOR_CALIBRATION_LAYER_SELECTIONS_V3
        or heldout_minimum < MINIMUM_CORRIDOR_HELDOUT_LAYER_SELECTIONS_V3
        or length_minimum < MINIMUM_CORRIDOR_LENGTH_SAMPLES_V3
        or isinstance(selected_layer_count, bool)
        or not isinstance(selected_layer_count, int)
        or selected_layer_count < 1
    ):
        raise ProofV3Error(
            "projection-corridor coverage minimum is too weak"
        )
    entry_names = tuple(str(name) for name in manifest_entry_names)
    layers = projection_corridor_manifest_layers_v3(entry_names)
    dense_mlp_layers = projection_corridor_dense_mlp_layers_v3(entry_names)
    calibration_counts = _canonical_layer_counts(
        value["calibration_layer_counts"],
        name="calibration layer counts",
    )
    heldout_counts = _canonical_layer_counts(
        value["heldout_layer_counts"],
        name="held-out layer counts",
    )
    mlp_coverage_value = value["mlp_activation_layer_coverage"]
    mlp_coverage_fields = {
        "layer",
        "calibration_case_count",
        "calibration_selected_cell_count",
        "calibration_selected_rail_cell_count",
        "calibration_selected_near_zero_cell_count",
        "calibration_scanned_cell_count",
        "calibration_rail_cell_count",
        "calibration_near_zero_cell_count",
        "heldout_case_count",
        "heldout_selected_cell_count",
        "heldout_selected_rail_cell_count",
        "heldout_selected_near_zero_cell_count",
        "heldout_scanned_cell_count",
        "heldout_rail_cell_count",
        "heldout_near_zero_cell_count",
    }
    if not isinstance(mlp_coverage_value, list):
        raise ProofV3Error(
            "projection-corridor MLP activation coverage is malformed"
        )
    mlp_coverage = {}
    for item in mlp_coverage_value:
        if not isinstance(item, dict) or set(item) != mlp_coverage_fields:
            raise ProofV3Error(
                "projection-corridor MLP activation coverage is malformed"
            )
        layer = item["layer"]
        counts = tuple(
            item[name]
            for name in sorted(mlp_coverage_fields - {"layer"})
        )
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer in mlp_coverage
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for count in counts
            )
        ):
            raise ProofV3Error(
                "projection-corridor MLP activation coverage is malformed"
            )
        mlp_coverage[layer] = item
    if tuple(mlp_coverage) != dense_mlp_layers:
        raise ProofV3Error(
            "projection-corridor MLP activation layer coverage is incomplete"
        )
    decode_counts = _canonical_positive_counts(
        value["decode_length_counts"],
        key_name="decode_tokens",
        inventory_name="decode length counts",
    )
    context_counts = _canonical_positive_counts(
        value["context_length_counts"],
        key_name="context_tokens",
        inventory_name="context length counts",
    )
    geometry_counts_value = value["geometry_counts"]
    if not isinstance(geometry_counts_value, list) or not geometry_counts_value:
        raise ProofV3Error(
            "projection-corridor geometry count inventory is malformed"
        )
    geometry_counts = {}
    for item in geometry_counts_value:
        if (
            not isinstance(item, dict)
            or set(item) != {"context_tokens", "decode_tokens", "count"}
        ):
            raise ProofV3Error(
                "projection-corridor geometry count inventory is malformed"
            )
        key = (item["context_tokens"], item["decode_tokens"])
        count = item["count"]
        if (
            any(isinstance(part, bool) or not isinstance(part, int) for part in key)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or key in geometry_counts
        ):
            raise ProofV3Error(
                "projection-corridor geometry count inventory is malformed"
            )
        geometry_counts[key] = count
    required_geometries = tuple(
        (context, decode)
        for context in context_lengths
        for decode in decode_lengths
    )
    if tuple(geometry_counts) != required_geometries:
        raise ProofV3Error(
            "projection-corridor geometry count inventory is not canonical"
        )
    maximum_geometry_count = value["maximum_geometry_case_count"]
    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise ProofV3Error("projection-corridor honest cases are malformed")
    case_ids = set()
    derived_layer_counts = {
        split: {layer: [0, set()] for layer in layers}
        for split in ("calibration", "heldout")
    }
    derived_decode_counts: dict[int, int] = {}
    derived_context_counts: dict[int, int] = {}
    derived_geometry_counts: dict[tuple[int, int], int] = {}
    derived_maximum_geometry_count = 0
    for case in cases:
        required_case_fields = {
            "id",
            "split",
            "target_context_tokens",
            "context_tokens",
            "decode_tokens",
            "selected_layer_indices",
            "selection_plan_sha256",
        }
        if not isinstance(case, dict) or not required_case_fields.issubset(case):
            raise ProofV3Error(
                "projection-corridor honest case is malformed"
            )
        case_id = case["id"]
        split = case["split"]
        target_context = case["target_context_tokens"]
        context = case["context_tokens"]
        decode = case["decode_tokens"]
        selected_layers = case["selected_layer_indices"]
        plan_digest = case["selection_plan_sha256"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or split not in derived_layer_counts
            or isinstance(target_context, bool)
            or not isinstance(target_context, int)
            or not 2 <= target_context <= context_max
            or isinstance(context, bool)
            or not isinstance(context, int)
            or not 2 <= context <= context_max
            or isinstance(decode, bool)
            or not isinstance(decode, int)
            or not 1 <= decode <= decode_max
            or not isinstance(selected_layers, list)
            or len(selected_layers) != selected_layer_count
            or any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer not in derived_layer_counts[split]
                for layer in selected_layers
            )
            or len(set(selected_layers)) != len(selected_layers)
            or not isinstance(plan_digest, str)
            or len(plan_digest) != 64
            or plan_digest.lower() != plan_digest
            or any(char not in "0123456789abcdef" for char in plan_digest)
        ):
            raise ProofV3Error(
                "projection-corridor honest case is malformed"
            )
        case_ids.add(case_id)
        for layer in selected_layers:
            layer_count, plans = derived_layer_counts[split][layer]
            derived_layer_counts[split][layer][0] = layer_count + 1
            plans.add(plan_digest)
        derived_decode_counts[decode] = derived_decode_counts.get(decode, 0) + 1
        derived_context_counts[target_context] = (
            derived_context_counts.get(target_context, 0) + 1
        )
        geometry = (target_context, decode)
        derived_geometry_counts[geometry] = (
            derived_geometry_counts.get(geometry, 0) + 1
        )
        if decode == decode_max and target_context == context_max:
            derived_maximum_geometry_count += 1

    derived_calibration_counts = {
        layer: (count, len(plans))
        for layer, (count, plans) in derived_layer_counts["calibration"].items()
    }
    derived_heldout_counts = {
        layer: (count, len(plans))
        for layer, (count, plans) in derived_layer_counts["heldout"].items()
    }
    heldout_numeric_names = (
        "heldout_maximum_required_sigma",
        "heldout_maximum_chi2",
        "heldout_maximum_required_fixed_quant_coeff",
        "qualified_corridor_sigma",
        "qualified_corridor_chi2",
        "protocol_fixed_quant_coeff",
    )
    heldout_numeric = {}
    for name in heldout_numeric_names:
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise ProofV3Error(
                "projection-corridor held-out gate is malformed"
            )
        heldout_numeric[name] = float(item)
    if (
        tuple(calibration_counts) != layers
        or tuple(heldout_counts) != layers
        or calibration_counts != derived_calibration_counts
        or heldout_counts != derived_heldout_counts
        or any(
            count < calibration_minimum or plans < calibration_minimum
            for count, plans in calibration_counts.values()
        )
        or any(
            count < heldout_minimum or plans < heldout_minimum
            for count, plans in heldout_counts.values()
        )
        or tuple(decode_counts) != decode_lengths
        or tuple(context_counts) != context_lengths
        or decode_counts != derived_decode_counts
        or context_counts != derived_context_counts
        or geometry_counts != derived_geometry_counts
        or any(count < length_minimum for count in decode_counts.values())
        or any(count < length_minimum for count in context_counts.values())
        or any(count < length_minimum for count in geometry_counts.values())
        or any(
            item[f"{split}_case_count"] < minimum
            or item[f"{split}_selected_cell_count"] < minimum
            or item[f"{split}_scanned_cell_count"] < minimum
            for item in mlp_coverage.values()
            for split, minimum in (
                ("calibration", calibration_minimum),
                ("heldout", heldout_minimum),
            )
        )
        or any(
            sum(
                item[f"{split}_selected_{kind}_cell_count"]
                for item in mlp_coverage.values()
            )
            < 1
            for split in ("calibration", "heldout")
            for kind in ("rail", "near_zero")
            if sum(
                item[f"{split}_{kind}_cell_count"]
                for item in mlp_coverage.values()
            )
            > 0
        )
        or isinstance(maximum_geometry_count, bool)
        or not isinstance(maximum_geometry_count, int)
        or maximum_geometry_count < length_minimum
        or maximum_geometry_count != derived_maximum_geometry_count
        or (
            isinstance(value.get("case_count"), int)
            and not isinstance(value.get("case_count"), bool)
            and value["case_count"] != len(cases)
        )
        or heldout_numeric["heldout_maximum_required_sigma"]
        > heldout_numeric["qualified_corridor_sigma"]
        or heldout_numeric["heldout_maximum_chi2"]
        > heldout_numeric["qualified_corridor_chi2"]
        or heldout_numeric["heldout_maximum_required_fixed_quant_coeff"]
        > heldout_numeric["protocol_fixed_quant_coeff"]
    ):
        raise ProofV3Error(
            "projection-corridor honest coverage did not meet its minimum"
        )


def required_projection_corridor_negative_cases_v3(
    manifest_entry_names: Iterable[str],
) -> tuple[str, ...]:
    """Return the fail-closed negative inventory for one signed manifest."""

    names = tuple(str(name) for name in manifest_entry_names)
    required = set(_COMMON_NEGATIVE_CASES_V3)
    if any(name.rsplit(".", 1)[-1] == "qkv" for name in names):
        required.add(_FULL_ATTENTION_NEGATIVE_CASE_V3)
    if any(name.rsplit(".", 1)[-1] == "gdn_ba" for name in names):
        required.add(_GDN_NEGATIVE_CASE_V3)
    if any(
        name.rsplit(".", 1)[-1] in {"down", "mlp_down"}
        for name in names
    ):
        required.update(_MLP_ACTIVATION_NEGATIVE_CASES_V3)
    if any(".moe." in name for name in names):
        required.update(_MOE_RUNTIME_NEGATIVE_CASES_V3)
    return tuple(sorted(required))


def validate_projection_corridor_negative_gate_v3(
    value,
    *,
    model_id: str,
    manifest_digest: str,
    manifest_entry_names: Iterable[str],
) -> tuple[dict, ...]:
    """Validate a completed, model-bound negative qualification result."""

    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "abi",
            "model_id",
            "manifest_digest",
            "case_count",
            "cases",
            "passed",
        }
        or value.get("abi") != PROJECTION_CORRIDOR_NEGATIVE_GATE_ABI_V3
        or value.get("model_id") != model_id
        or value.get("manifest_digest") != manifest_digest
        or value.get("passed") is not True
    ):
        raise ProofV3Error(
            "projection-corridor negative qualification is malformed"
        )
    cases = value.get("cases")
    case_count = value.get("case_count")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or not isinstance(cases, list)
        or case_count != len(cases)
        or case_count < 1
    ):
        raise ProofV3Error(
            "projection-corridor negative qualification inventory is malformed"
        )

    parsed = []
    for item in cases:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "case_id",
                "evidence_class",
                "trial_count",
                "rejected_count",
                "accepted_count",
                "evidence_sha256",
            }
        ):
            raise ProofV3Error(
                "projection-corridor negative qualification case is malformed"
            )
        case_id = item.get("case_id")
        evidence_class = item.get("evidence_class")
        trial_count = item.get("trial_count")
        rejected_count = item.get("rejected_count")
        accepted_count = item.get("accepted_count")
        digest = item.get("evidence_sha256")
        if (
            not isinstance(case_id, str)
            or not case_id
            or evidence_class not in _EVIDENCE_CLASSES_V3
            or evidence_class
            not in _CASE_EVIDENCE_CLASSES_V3.get(case_id, ())
            or isinstance(trial_count, bool)
            or not isinstance(trial_count, int)
            or trial_count < 1
            or isinstance(rejected_count, bool)
            or not isinstance(rejected_count, int)
            or isinstance(accepted_count, bool)
            or not isinstance(accepted_count, int)
            or rejected_count != trial_count
            or accepted_count != 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ProofV3Error(
                "projection-corridor negative qualification case did not pass"
            )
        parsed.append(dict(item))

    ids = tuple(item["case_id"] for item in parsed)
    if len(set(ids)) != len(ids) or ids != tuple(sorted(ids)):
        raise ProofV3Error(
            "projection-corridor negative qualification cases are not canonical"
        )
    required = required_projection_corridor_negative_cases_v3(
        manifest_entry_names
    )
    if ids != required:
        raise ProofV3Error(
            "projection-corridor negative qualification is incomplete"
        )
    return tuple(parsed)
