"""Deterministic operation and block layout for proof protocol v2."""

from __future__ import annotations

import hashlib
import math
import struct

from verallm.challenge.v2 import (
    MODEL_LM_HEAD_OPERATION_ID,
    MODEL_OPERATION_LAYER_IDX,
    OperationKeyV2,
    RegisteredOperationV2,
    RUNTIME_Y_COMMITMENT_BLOCK_COLS,
    RUNTIME_Y_COMMITMENT_BLOCK_ROWS,
    canonical_axis_segments_v2,
)
from verallm.proof_v2.manifest import (
    OperationDescriptor,
    StaticWeightCommitmentManifest,
    WEIGHT_SCALE_BLOCK_COLS,
)
from verallm.proof_v2.trace import (
    TRACE_ATTENTION_FULL_AUDIT_ONLY,
    TRACE_ATTENTION_FULL_TRANSITION_V1,
    TRACE_ATTENTION_GDN_AUDIT_ONLY,
    TRACE_ATTENTION_GDN_TRANSITION_V1,
    TRACE_PROFILE_QWEN_HYBRID_DENSE_V1,
    TRACE_PROFILE_QWEN_HYBRID_SPARSE_MOE_SHELL_V1,
)
from verallm.proof_v2.transition import (
    FULL_ATTENTION_TRANSITION_PROFILE_V1,
    FullAttentionTransitionParametersV2,
    GDN_TRANSITION_PROFILE_V1,
    GDNTransitionParametersV2,
    ProofV2TransitionError,
)
from zkllm.crypto.pcs_v2 import MAX_VECTOR_LEN


_LAYOUT_DOMAIN = b"VERATHOS/PROOF_V2/BLOCK_LAYOUT/SHA256"
_TREE_CONTEXT_DOMAIN = b"VERATHOS/PROOF_V2/COMMITMENT_TREE_CONTEXT/SHA256"
MAX_BLOCK_AXIS = 16
DENSE_RUNTIME_OPERATION_ID = "mlp.gate_proj"
MLP_GATE_UP_OPERATION_ID = "mlp.gate_up_proj"
MLP_DOWN_OPERATION_ID = "mlp.down_proj"
FULL_QKV_OPERATION_ID = "attention.qkv_proj"
FULL_OUTPUT_OPERATION_ID = "attention.o_proj"
GDN_QKVZ_OPERATION_ID = "gdn.in_proj_qkvz"
GDN_BA_OPERATION_ID = "gdn.in_proj_ba"
GDN_OUTPUT_OPERATION_ID = "gdn.out_proj"

_COMMON_EXECUTION_OPERATION_IDS = frozenset(
    (MLP_GATE_UP_OPERATION_ID, MLP_DOWN_OPERATION_ID)
)
_EXECUTION_OPERATION_IDS = {
    TRACE_ATTENTION_FULL_AUDIT_ONLY: frozenset(
        (
            *_COMMON_EXECUTION_OPERATION_IDS,
            FULL_QKV_OPERATION_ID,
            FULL_OUTPUT_OPERATION_ID,
        )
    ),
    TRACE_ATTENTION_FULL_TRANSITION_V1: frozenset(
        (
            *_COMMON_EXECUTION_OPERATION_IDS,
            FULL_QKV_OPERATION_ID,
            FULL_OUTPUT_OPERATION_ID,
        )
    ),
    TRACE_ATTENTION_GDN_AUDIT_ONLY: frozenset(
        (
            *_COMMON_EXECUTION_OPERATION_IDS,
            GDN_QKVZ_OPERATION_ID,
            GDN_BA_OPERATION_ID,
            GDN_OUTPUT_OPERATION_ID,
        )
    ),
    TRACE_ATTENTION_GDN_TRANSITION_V1: frozenset(
        (
            *_COMMON_EXECUTION_OPERATION_IDS,
            GDN_QKVZ_OPERATION_ID,
            GDN_BA_OPERATION_ID,
            GDN_OUTPUT_OPERATION_ID,
        )
    ),
}

_EXECUTION_TRACE_FIELDS = {
    MLP_GATE_UP_OPERATION_ID: ("x_ffn", "mlp_gate_up"),
    MLP_DOWN_OPERATION_ID: ("mlp_hidden", "mlp_down"),
    FULL_QKV_OPERATION_ID: ("x_attn", "attention_qkv"),
    FULL_OUTPUT_OPERATION_ID: ("attention_core_out", "attention_out_proj"),
    GDN_QKVZ_OPERATION_ID: ("x_attn", "gdn_qkvz"),
    GDN_BA_OPERATION_ID: ("x_attn", "gdn_ba"),
    GDN_OUTPUT_OPERATION_ID: ("attention_core_out", "attention_out_proj"),
}

_EXECUTION_CAPTURE_SUFFIXES = {
    MLP_GATE_UP_OPERATION_ID: ("mlp_gate_up_input", "mlp_gate_up_output"),
    MLP_DOWN_OPERATION_ID: ("mlp_down_input", "mlp_down_output"),
    FULL_QKV_OPERATION_ID: ("attention_qkv_input", "attention_qkv_output"),
    FULL_OUTPUT_OPERATION_ID: ("attention_o_input", "attention_o_output"),
    GDN_QKVZ_OPERATION_ID: ("gdn_qkvz_input", "gdn_qkvz_output"),
    GDN_BA_OPERATION_ID: ("gdn_ba_input", "gdn_ba_output"),
    GDN_OUTPUT_OPERATION_ID: ("gdn_o_input", "gdn_o_output"),
}


class ProofV2LayoutError(ValueError):
    """Manifest operation dimensions cannot use the canonical v2 layout."""


def execution_capture_suffixes_for_operation(
    operation_id: str,
) -> tuple[str, str]:
    """Return the runtime capture keys for one signed shell operation."""

    try:
        return _EXECUTION_CAPTURE_SUFFIXES[operation_id]
    except KeyError as exc:
        raise ProofV2LayoutError(
            f"operation {operation_id!r} has no execution capture ABI"
        ) from exc


def padded_inner_dimension(inner_dim: int) -> int:
    if isinstance(inner_dim, bool) or not isinstance(inner_dim, int) or inner_dim <= 0:
        raise ProofV2LayoutError("inner dimension must be a positive integer")
    padded = 1 << (inner_dim - 1).bit_length()
    if padded > MAX_VECTOR_LEN:
        raise ProofV2LayoutError("inner dimension exceeds the PCS vector limit")
    return padded


def block_axis_for_inner(inner_dim: int) -> int:
    """Return the fixed row/column sampling axis for one supported operation."""

    padded_inner_dimension(inner_dim)
    return MAX_BLOCK_AXIS


def registered_operations_from_manifest(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Convert all layer-level manifest operations into the exact challenge set."""

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV2LayoutError("manifest has an unexpected type")
    operations = []
    for descriptor in manifest.operations:
        if descriptor.layer < 0:
            continue
        expert_idx = -1 if descriptor.expert_id is None else descriptor.expert_id
        key = OperationKeyV2(
            descriptor.layer,
            descriptor.operation_id,
            expert_idx,
        )
        axis = block_axis_for_inner(descriptor.rows)
        operation = RegisteredOperationV2(
            key=key,
            inner_dim=descriptor.rows,
            output_dim=descriptor.cols,
            block_rows=axis,
            block_cols=axis,
            weight_commitment_root=descriptor.commitment,
        )
        try:
            operation.validate(manifest.model_spec.num_layers)
        except (TypeError, ValueError) as exc:
            raise ProofV2LayoutError(
                f"manifest operation {descriptor.operation_id!r} is not usable by proof v2"
            ) from exc
        operations.append(operation)
    canonical = tuple(sorted(operations, key=lambda item: item.key))
    if not canonical:
        raise ProofV2LayoutError("manifest contains no layer-level proof operations")
    keys = [item.key for item in canonical]
    if len(keys) != len(set(keys)):
        raise ProofV2LayoutError("manifest layer-level operation keys are not unique")
    return canonical


def registered_model_operations_from_manifest(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Convert authenticated model-level operations into v2 operation records."""

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV2LayoutError("manifest has an unexpected type")
    operations = []
    for descriptor in manifest.operations:
        if descriptor.layer >= 0:
            continue
        if (
            descriptor.layer != -1
            or descriptor.operation_id != MODEL_LM_HEAD_OPERATION_ID
            or descriptor.expert_id is not None
        ):
            raise ProofV2LayoutError(
                "manifest contains an unsupported model-level operation"
            )
        operation = RegisteredOperationV2(
            key=OperationKeyV2(
                MODEL_OPERATION_LAYER_IDX,
                descriptor.operation_id,
                -1,
            ),
            inner_dim=descriptor.rows,
            output_dim=descriptor.cols,
            block_rows=block_axis_for_inner(descriptor.rows),
            block_cols=MAX_BLOCK_AXIS,
            weight_commitment_root=descriptor.commitment,
        )
        try:
            operation.validate(manifest.model_spec.num_layers)
        except (TypeError, ValueError) as exc:
            raise ProofV2LayoutError(
                f"manifest operation {descriptor.operation_id!r} is not usable by proof v2"
            ) from exc
        operations.append(operation)
    canonical = tuple(sorted(operations, key=lambda item: item.key))
    keys = [item.key for item in canonical]
    if len(keys) != len(set(keys)):
        raise ProofV2LayoutError("manifest model-level operation keys are not unique")
    return canonical


def registered_all_operations_from_manifest(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Return the exact authenticated layer and model-level operation set."""

    operations = tuple(
        sorted(
            (
                *registered_operations_from_manifest(manifest),
                *registered_model_operations_from_manifest(manifest),
            ),
            key=lambda item: item.key,
        )
    )
    keys = [item.key for item in operations]
    if len(keys) != len(set(keys)):
        raise ProofV2LayoutError("manifest operation keys are not unique")
    return operations


def registered_lm_head_operation_from_manifest(
    manifest: StaticWeightCommitmentManifest,
) -> RegisteredOperationV2 | None:
    """Return the authenticated model-level LM-head operation when present."""

    operations = registered_model_operations_from_manifest(manifest)
    if not operations:
        return None
    if len(operations) != 1:
        raise ProofV2LayoutError(
            "proof v2 supports exactly one model-level LM-head operation"
        )
    operation = operations[0]
    spec = manifest.model_spec
    if (
        operation.inner_dim != spec.hidden_dim
        or operation.output_dim != spec.vocab_size
    ):
        raise ProofV2LayoutError(
            "model-level LM-head dimensions do not match the ModelSpec"
        )
    return operation


def validate_dense_runtime_manifest_profile(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Validate the exact manifest shape supported by the current miner runtime.

    The manifest format intentionally supports future operation families, but
    the current inference capture and weight extraction code supports exactly
    one dense gate projection for every transformer layer.  Runtime loaders
    must reject every other signed manifest instead of accepting an operation
    set that the miner cannot faithfully capture or prove.
    """

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV2LayoutError("manifest has an unexpected type")
    spec = manifest.model_spec
    if spec.num_experts != 0 or spec.expert_w_num_cols != 0:
        raise ProofV2LayoutError(
            "proof v2 runtime currently supports dense models only"
        )
    layer_descriptors = tuple(
        descriptor for descriptor in manifest.operations if descriptor.layer >= 0
    )
    if len(layer_descriptors) != spec.num_layers:
        raise ProofV2LayoutError(
            "dense proof v2 runtime requires exactly one operation per layer"
        )
    model_operations = registered_model_operations_from_manifest(manifest)
    if len(model_operations) != 1:
        raise ProofV2LayoutError(
            "dense proof v2 runtime requires one authenticated LM-head operation"
        )
    registered_lm_head_operation_from_manifest(manifest)

    for expected_layer, descriptor in enumerate(layer_descriptors):
        if descriptor.layer != expected_layer:
            raise ProofV2LayoutError(
                "dense proof v2 runtime requires every layer exactly once"
            )
        if (
            descriptor.operation_id != DENSE_RUNTIME_OPERATION_ID
            or descriptor.expert_id is not None
        ):
            raise ProofV2LayoutError(
                "dense proof v2 runtime operation identity is unsupported"
            )
        if (
            descriptor.rows != spec.hidden_dim
            or descriptor.cols != spec.intermediate_dim
        ):
            raise ProofV2LayoutError(
                "dense proof v2 runtime operation dimensions do not match the ModelSpec"
            )

    operations = registered_operations_from_manifest(manifest)
    if len(operations) != spec.num_layers:
        raise ProofV2LayoutError("dense proof v2 runtime operation set is not exact")
    return operations


def validate_qwen_hybrid_execution_manifest_profile(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Validate the exact authority-signed Qwen hybrid execution shell.

    Attention/GDN internal transition checks are selected by the signed
    per-layer profile. Dense manifests require both MLP projections. Sparse
    manifests require the same attention shell but deliberately exclude MLP
    operations because routed/shared matrices use individually signed Merkle
    openings in proof v3. Unknown or additional operations fail closed.
    """

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV2LayoutError("manifest has an unexpected type")
    if manifest.execution_profile == TRACE_PROFILE_QWEN_HYBRID_DENSE_V1:
        sparse_moe_shell = False
    elif (
        manifest.execution_profile
        == TRACE_PROFILE_QWEN_HYBRID_SPARSE_MOE_SHELL_V1
    ):
        sparse_moe_shell = True
    else:
        raise ProofV2LayoutError(
            "proof v2 manifest is missing the supported causal execution profile"
        )
    spec = manifest.model_spec
    if sparse_moe_shell and (
        spec.num_experts < 2 or spec.expert_w_num_cols <= 0
    ):
        raise ProofV2LayoutError(
            "sparse-MoE shell requires exact expert ModelSpec geometry"
        )
    if not sparse_moe_shell and (
        spec.num_experts != 0 or spec.expert_w_num_cols != 0
    ):
        raise ProofV2LayoutError(
            "dense Qwen hybrid execution profile rejects expert geometry"
        )
    if spec.activation != "silu" or spec.norm_type != "rmsnorm":
        raise ProofV2LayoutError(
            "Qwen hybrid execution proof requires exact SiLU and RMSNorm semantics"
        )
    if len(manifest.layer_execution) != spec.num_layers:
        raise ProofV2LayoutError(
            "execution manifest must describe every model layer exactly once"
        )
    if manifest.model_execution is None:
        raise ProofV2LayoutError(
            "execution manifest is missing model-boundary parameters"
        )
    has_full_attention_transition = any(
        item.attention_profile == TRACE_ATTENTION_FULL_TRANSITION_V1
        for item in manifest.layer_execution
    )
    if has_full_attention_transition:
        policy = manifest.model_execution.audit_policy
        if policy is None:
            raise ProofV2LayoutError(
                "full-attention transition profiles require a signed hard-audit policy"
            )
        if policy.hard_layer_count > spec.num_layers:
            raise ProofV2LayoutError(
                "signed hard-audit layer count exceeds the ModelSpec"
            )
        if policy.full_attention_heads_per_layer < 2:
            raise ProofV2LayoutError(
                "full-attention transition profiles require at least two "
                "nonce-selected heads per layer"
            )

    def validate_f16_vector(value: bytes, name: str) -> None:
        if len(value) != 2 * spec.hidden_dim:
            raise ProofV2LayoutError(
                f"{name} does not contain exactly hidden_dim fp16 values"
            )
        if any(not math.isfinite(item[0]) for item in struct.iter_unpack("<e", value)):
            raise ProofV2LayoutError(f"{name} contains a non-finite value")

    validate_f16_vector(
        manifest.model_execution.final_norm_weight_f16,
        "final_norm_weight_f16",
    )

    layer_descriptors: dict[int, dict[str, OperationDescriptor]] = {
        layer: {} for layer in range(spec.num_layers)
    }
    for descriptor in manifest.operations:
        if descriptor.layer < 0:
            continue
        if descriptor.layer not in layer_descriptors:
            raise ProofV2LayoutError("execution operation layer is out of range")
        if descriptor.expert_id is not None:
            raise ProofV2LayoutError(
                "Qwen shell execution operations must not specify an expert"
            )
        by_id = layer_descriptors[descriptor.layer]
        if descriptor.operation_id in by_id:
            raise ProofV2LayoutError(
                "execution operation identity is duplicated within a layer"
            )
        by_id[descriptor.operation_id] = descriptor

    for layer_profile in manifest.layer_execution:
        validate_f16_vector(
            layer_profile.input_norm_weight_f16,
            f"layer {layer_profile.layer} input_norm_weight_f16",
        )
        validate_f16_vector(
            layer_profile.post_attention_norm_weight_f16,
            f"layer {layer_profile.layer} post_attention_norm_weight_f16",
        )
        if layer_profile.norm_epsilon_q32 <= 0:
            raise ProofV2LayoutError(
                f"layer {layer_profile.layer} norm epsilon is not positive"
            )
        if layer_profile.attention_profile == TRACE_ATTENTION_GDN_TRANSITION_V1:
            if layer_profile.transition_profile != GDN_TRANSITION_PROFILE_V1:
                raise ProofV2LayoutError(
                    "GDN transition layer has an unsupported transition profile"
                )
            try:
                transition = GDNTransitionParametersV2.from_canonical_bytes(
                    layer_profile.transition_parameters
                )
            except ProofV2TransitionError as exc:
                raise ProofV2LayoutError(
                    "GDN transition layer parameters are malformed"
                ) from exc
            if transition.rms_epsilon_q32 != layer_profile.norm_epsilon_q32:
                raise ProofV2LayoutError("GDN transition and bridge RMS epsilon differ")
            if transition.conv_kernel_size <= 1:
                raise ProofV2LayoutError(
                    "GDN transition cache must retain a nonempty convolution state"
                )
            if (
                transition.num_key_heads != spec.num_heads
                or transition.key_head_dim != spec.head_dim
            ):
                raise ProofV2LayoutError(
                    "GDN transition key-head layout does not match the ModelSpec"
                )
        elif layer_profile.attention_profile == TRACE_ATTENTION_FULL_TRANSITION_V1:
            if layer_profile.transition_profile != FULL_ATTENTION_TRANSITION_PROFILE_V1:
                raise ProofV2LayoutError(
                    "full-attention transition layer has an unsupported transition profile"
                )
            try:
                transition = FullAttentionTransitionParametersV2.from_canonical_bytes(
                    layer_profile.transition_parameters
                )
            except ProofV2TransitionError as exc:
                raise ProofV2LayoutError(
                    "full-attention transition layer parameters are malformed"
                ) from exc
            if transition.rms_epsilon_q32 != layer_profile.norm_epsilon_q32:
                raise ProofV2LayoutError(
                    "full-attention transition and bridge RMS epsilon differ"
                )
            if (
                transition.num_query_heads != spec.num_heads
                or transition.head_dim != spec.head_dim
            ):
                raise ProofV2LayoutError(
                    "full-attention head layout does not match the ModelSpec"
                )
            if (
                manifest.model_execution.audit_policy.full_attention_heads_per_layer
                > transition.num_query_heads
            ):
                raise ProofV2LayoutError(
                    "full-attention head sampling exceeds the signed query-head count"
                )
        elif any(
            value not in (None, b"")
            for value in (
                layer_profile.transition_profile,
                layer_profile.transition_parameter_root,
                layer_profile.transition_parameters,
            )
        ):
            raise ProofV2LayoutError(
                "audit-only attention profiles cannot carry transition parameters"
            )
        expected_ids = _EXECUTION_OPERATION_IDS.get(
            layer_profile.attention_profile
        )
        if expected_ids is None:
            raise ProofV2LayoutError(
                "execution manifest contains an unsupported attention profile"
            )
        if sparse_moe_shell:
            expected_ids = expected_ids - _COMMON_EXECUTION_OPERATION_IDS
        actual = layer_descriptors[layer_profile.layer]
        if frozenset(actual) != expected_ids:
            raise ProofV2LayoutError(
                f"layer {layer_profile.layer} execution operation set is not exact"
            )
        if layer_profile.attention_profile == TRACE_ATTENTION_GDN_TRANSITION_V1:
            expected_qkvz_width = (
                2 * transition.num_key_heads * transition.key_head_dim
                + 2 * transition.num_value_heads * transition.value_head_dim
            )
            if (
                actual[GDN_QKVZ_OPERATION_ID].cols != expected_qkvz_width
                or actual[GDN_BA_OPERATION_ID].cols != 2 * transition.num_value_heads
                or actual[GDN_OUTPUT_OPERATION_ID].rows
                != transition.num_value_heads * transition.value_head_dim
            ):
                raise ProofV2LayoutError(
                    "GDN transition operation dimensions do not match its signed parameters"
                )
        elif layer_profile.attention_profile == TRACE_ATTENTION_FULL_TRANSITION_V1:
            if (
                actual[FULL_QKV_OPERATION_ID].cols != transition.qkv_width
                or actual[FULL_OUTPUT_OPERATION_ID].rows != transition.q_width
            ):
                raise ProofV2LayoutError(
                    "full-attention transition operation dimensions do not match its signed parameters"
                )

        for operation_id, descriptor in actual.items():
            if (
                operation_id
                in {
                    MLP_GATE_UP_OPERATION_ID,
                    FULL_QKV_OPERATION_ID,
                    GDN_QKVZ_OPERATION_ID,
                    GDN_BA_OPERATION_ID,
                }
                and descriptor.rows != spec.hidden_dim
            ):
                raise ProofV2LayoutError(
                    f"layer {layer_profile.layer} {operation_id} input dimension "
                    "does not match the ModelSpec"
                )
            # ModelSpec.intermediate_dim is the registered fused gate+up
            # width.  Registration derives it from gate_up_proj's logical
            # output size, not the single SiLU branch width.
            if operation_id == MLP_GATE_UP_OPERATION_ID and (
                descriptor.cols != spec.intermediate_dim
            ):
                raise ProofV2LayoutError(
                    "MLP gate/up output dimension does not match the ModelSpec"
                )
            if operation_id == MLP_DOWN_OPERATION_ID and (
                spec.intermediate_dim % 2
                or descriptor.rows != spec.intermediate_dim // 2
                or descriptor.cols != spec.hidden_dim
            ):
                raise ProofV2LayoutError(
                    "MLP down-projection dimensions do not match the ModelSpec"
                )
            if (
                operation_id
                in {
                    MLP_DOWN_OPERATION_ID,
                    FULL_OUTPUT_OPERATION_ID,
                    GDN_OUTPUT_OPERATION_ID,
                }
                and descriptor.cols != spec.hidden_dim
            ):
                raise ProofV2LayoutError(
                    f"layer {layer_profile.layer} {operation_id} output dimension "
                    "does not match the ModelSpec"
                )

    model_operations = registered_model_operations_from_manifest(manifest)
    if len(model_operations) != 1:
        raise ProofV2LayoutError(
            "execution manifest requires one authenticated LM-head operation"
        )
    registered_lm_head_operation_from_manifest(manifest)

    operations = registered_operations_from_manifest(manifest)
    expected_count = sum(
        len(
            _EXECUTION_OPERATION_IDS[item.attention_profile]
            - (
                _COMMON_EXECUTION_OPERATION_IDS
                if sparse_moe_shell
                else frozenset()
            )
        )
        for item in manifest.layer_execution
    )
    if len(operations) != expected_count:
        raise ProofV2LayoutError("execution operation set is not exact")
    return operations


def require_independent_execution_transitions_v2(
    manifest: StaticWeightCommitmentManifest,
) -> tuple[RegisteredOperationV2, ...]:
    """Require an execution shell with no hash-only attention transition.

    A valid execution manifest can still describe ``*_audit_only`` attention
    profiles for development and arithmetic plumbing.  Those profiles are not
    suitable for a signed runtime artifact: their KV/GDN state links are
    prover-authored digest chains rather than verifier-replayed transitions.
    Keep that distinction at the artifact/runtime boundary so a manually
    assembled document cannot advertise a hard proof profile with a known
    transition gap.

    This gate deliberately says nothing about complete verified inference.  It
    only establishes the narrower prerequisite that every attention state
    transition selected by the current execution shell has an independent
    verifier implementation.
    """

    operations = validate_qwen_hybrid_execution_manifest_profile(manifest)
    if (
        any(
            item.attention_profile == TRACE_ATTENTION_FULL_TRANSITION_V1
            for item in manifest.layer_execution
        )
        and manifest.model_execution.audit_policy is None
    ):
        raise ProofV2LayoutError(
            "full-attention transition profiles require a signed hard-audit policy"
        )
    unresolved = tuple(
        item.layer
        for item in manifest.layer_execution
        if item.attention_profile
        not in (
            TRACE_ATTENTION_GDN_TRANSITION_V1,
            TRACE_ATTENTION_FULL_TRANSITION_V1,
        )
    )
    if unresolved:
        rendered = ",".join(str(layer) for layer in unresolved)
        raise ProofV2LayoutError(
            "execution manifest is not eligible for a hard proof-v2 release: "
            "layers without an independently verified attention transition: " + rendered
        )
    return operations


def execution_trace_fields_for_operation(operation_id: str) -> tuple[str, str]:
    """Return the verifier-owned trace input/output names for an operation."""

    try:
        return _EXECUTION_TRACE_FIELDS[operation_id]
    except (KeyError, TypeError) as exc:
        raise ProofV2LayoutError(
            "operation is not part of the causal execution trace profile"
        ) from exc


def operation_descriptor_by_key(
    manifest: StaticWeightCommitmentManifest,
) -> dict[OperationKeyV2, OperationDescriptor]:
    result = {}
    for descriptor in manifest.operations:
        layer_idx = (
            MODEL_OPERATION_LAYER_IDX if descriptor.layer == -1 else descriptor.layer
        )
        key = OperationKeyV2(
            layer_idx,
            descriptor.operation_id,
            -1 if descriptor.expert_id is None else descriptor.expert_id,
        )
        if key in result:
            raise ProofV2LayoutError("manifest operation keys are not unique")
        result[key] = descriptor
    return result


def operation_weight_scale_q32_v2(
    descriptor: OperationDescriptor,
    column_offset: int,
) -> int:
    """Return the authority-signed scale for one challenged output block."""

    if (
        isinstance(column_offset, bool)
        or not isinstance(column_offset, int)
        or column_offset < 0
        or column_offset >= descriptor.cols
    ):
        raise ProofV2LayoutError("weight scale column offset is not canonical")
    scales = descriptor.weight_block_scales_q32
    if not scales:
        return descriptor.weight_scale_q32
    if column_offset % WEIGHT_SCALE_BLOCK_COLS:
        raise ProofV2LayoutError("weight scale column offset is not canonical")
    index = column_offset // WEIGHT_SCALE_BLOCK_COLS
    try:
        return scales[index]
    except IndexError as exc:
        raise ProofV2LayoutError(
            "weight block scale set does not cover the challenged columns"
        ) from exc


def row_segments(
    operation: RegisteredOperationV2, row_count: int
) -> tuple[tuple[int, int], ...]:
    del operation
    return canonical_axis_segments_v2(
        row_count,
        RUNTIME_Y_COMMITMENT_BLOCK_ROWS,
    )


def column_segments(operation: RegisteredOperationV2) -> tuple[tuple[int, int], ...]:
    return canonical_axis_segments_v2(operation.output_dim, operation.block_cols)


def runtime_y_column_segments(
    operation: RegisteredOperationV2,
) -> tuple[tuple[int, int], ...]:
    """Return fixed-width commitment segments for captured runtime output.

    GEMM challenges remain 16 columns wide. Runtime output is committed in
    256-column segments so freezing every layer before challenge selection is
    inexpensive; the opened segment still authenticates the exact challenged
    16-column slice. A short final segment is zero-padded to the fixed width.
    """

    return tuple(
        (
            offset,
            min(
                RUNTIME_Y_COMMITMENT_BLOCK_COLS,
                operation.output_dim - offset,
            ),
        )
        for offset in range(
            0,
            operation.output_dim,
            RUNTIME_Y_COMMITMENT_BLOCK_COLS,
        )
    )


def layout_digest(operation: RegisteredOperationV2) -> bytes:
    """Bind operation identity, dimensions, padding, and fixed block convention."""

    padded = padded_inner_dimension(operation.inner_dim)
    encoded = operation.key.canonical_bytes() + struct.pack(
        "<IIIII",
        operation.inner_dim,
        operation.output_dim,
        padded,
        operation.block_rows,
        operation.block_cols,
    )
    return hashlib.sha256(_LAYOUT_DOMAIN + encoded).digest()


def commitment_tree_context(
    operation: RegisteredOperationV2,
    *,
    tensor: str,
    logical_axis_length: int,
) -> bytes:
    """Return the non-circular context used to hash X or W commitment leaves."""

    if tensor not in ("x", "w"):
        raise ProofV2LayoutError("commitment tree tensor must be 'x' or 'w'")
    if (
        isinstance(logical_axis_length, bool)
        or not isinstance(logical_axis_length, int)
        or not 0 < logical_axis_length < (1 << 32)
    ):
        raise ProofV2LayoutError("logical axis length is out of range")
    return hashlib.sha256(
        _TREE_CONTEXT_DOMAIN
        + tensor.encode("ascii")
        + layout_digest(operation)
        + struct.pack("<I", logical_axis_length)
    ).digest()


__all__ = [
    "DENSE_RUNTIME_OPERATION_ID",
    "MODEL_LM_HEAD_OPERATION_ID",
    "MODEL_OPERATION_LAYER_IDX",
    "MAX_BLOCK_AXIS",
    "RUNTIME_Y_COMMITMENT_BLOCK_COLS",
    "RUNTIME_Y_COMMITMENT_BLOCK_ROWS",
    "ProofV2LayoutError",
    "block_axis_for_inner",
    "column_segments",
    "commitment_tree_context",
    "layout_digest",
    "operation_descriptor_by_key",
    "operation_weight_scale_q32_v2",
    "padded_inner_dimension",
    "registered_all_operations_from_manifest",
    "registered_lm_head_operation_from_manifest",
    "registered_model_operations_from_manifest",
    "registered_operations_from_manifest",
    "require_independent_execution_transitions_v2",
    "row_segments",
    "runtime_y_column_segments",
    "validate_dense_runtime_manifest_profile",
    "validate_qwen_hybrid_execution_manifest_profile",
]
