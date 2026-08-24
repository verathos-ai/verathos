"""Production economic proof-v3 profile construction.

The economic verifier uses the authority-authenticated projection manifest as
its exact weight, normalization, attention/GDN, and LM-head inventory.  This
module derives the signed execution shell from that manifest and its
digest-bound runtime semantics.  It deliberately contains no model weights
and no test-fixture constants.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import replace
from typing import Sequence

from verallm.proof_v3.attention_runtime_semantics import (
    ATTENTION_RUNTIME_SEMANTICS_VERSION_V3,
    AttentionRuntimeSemanticsV3,
    Q_GATE_INTERLEAVED_LAYOUT_V3,
)
from verallm.proof_v3.constraint_system import (
    constraint_system_relation_projection_bytes_v3,
)
from verallm.proof_v3.economic_challenge import (
    ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    ECONOMIC_COMPACT_PCS_QUERY_COUNT_V3,
    ECONOMIC_COMPACT_SELECTION_ABI_V3,
    ECONOMIC_ESCALATION_PCS_QUERY_COUNT_V3,
    ECONOMIC_SELECTION_ABI_V3,
    ECONOMIC_STREAMING_SELECTION_ABI_V3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import EXECUTION_ANCHOR_ABI_V3
from verallm.proof_v3.gdn_decode_corridor import (
    GDN_DECODE_CORRIDOR_ABI_V3,
)
from verallm.proof_v3.gdn_projection_input import (
    GDN_CANONICAL_NORM_INPUT_ABI_V3,
)
from verallm.proof_v3.lean_execution_anchor import (
    LEAN_EXECUTION_ANCHOR_ABI_V3,
)
from verallm.proof_v3.lean_projection_batch import (
    LEAN_PROJECTION_BATCH_ABI_V3,
)
from verallm.proof_v3.lean_projection_native import (
    LEAN_PROJECTION_NATIVE_FOLD_ABI_V3,
)
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_ABI_V3,
)
from verallm.proof_v3.gdn_runtime_semantics import GdnRuntimeSemanticsV3
from verallm.proof_v3.moe_projection_inventory import (
    MoeProjectionIdentityV3,
    parse_moe_projection_name_v3,
    validate_moe_projection_inventory_v3,
)
from verallm.proof_v3.moe_runtime_semantics import MoeRuntimeSemanticsV3
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.projection_manifest import (
    LM_HEAD_CATALOG_BINDING_V3,
    ProjectionManifestEntryV3,
    ProjectionManifestV3,
)
from verallm.proof_v3.relation import (
    ECONOMIC_SHELL_COVERAGE_MODE_V3,
    GLOBAL_EXECUTION_RELATION_ABI_V3,
    GDN_STATE_CACHE_SEMANTICS_V3,
    GPU_RETENTION_LEASE_ABI_V3,
    LOGICAL_CACHE_ADDRESS_ABI_V3,
    LOGICAL_CACHE_TABLES_SEMANTICS_V3,
    PAGED_KV_CACHE_SEMANTICS_V3,
    POSTCOMMIT_AUDIT_TIER_ABI_V3,
    PREFIX_CACHE_LOGICAL_ADDRESS_ABI_V3,
    PREFIX_CACHE_RETENTION_LEASE_ABI_V3,
    PREFIX_CACHE_TABLES_SEMANTICS_V3,
    CacheRelationSpecV3,
    DeclaredDimensionV3,
    ExecutionRelationNodeV3,
    ExecutionRelationSpecV3,
    HardAuditPolicyV3,
    LayerAuditPlanV3,
    LayerCacheRelationV3,
    RegisteredOperationReferenceV3,
    RelationToleranceV3,
    SequenceDomainV3,
    StaticParameterBindingV3,
    StaticTableBindingV3,
    TensorDescriptorV3,
)
from verallm.proof_v3.scored_calibration_set import ScoredCalibrationSetV3
from verallm.proof_v3.scheduler_geometry import SCHEDULER_GEOMETRY_ABI_V3

ECONOMIC_PROFILE_ADAPTER_ID_V3 = "verathos.economic_runtime_capture"
ECONOMIC_PROFILE_ADAPTER_VERSION_V3 = "v1"
ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3 = "v2"
ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3 = "v8"
ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3 = "v9"
ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3 = "v10"
ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3 = "v11"
ECONOMIC_COMPACT_PROJECTION_ABI_V3 = (
    "projection.sampled_output.merkle_catalog_sumcheck.v5"
)
ECONOMIC_QUANTIZATION_SEMANTICS_ID_V3 = "int8.symmetric.v1"
ECONOMIC_NONCE_SELECTION_ABI_V3 = "economic.recompute.nonce.v1"
ECONOMIC_RECURSIVE_ACCUMULATOR_ABI_V3 = "economic.openings.v1"
ECONOMIC_MLP_ACTIVATION_REPLAY_ABI_V3 = (
    "mlp.silu.selected_runtime_cells.fp16_bf16.ulp2.v1"
)

_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_PROFILE/V1/"
_LAYER_ENTRY = re.compile(r"^l([0-9]+)\.([a-z0-9_]+)$")
_GLOBAL_ENTRY_NAMES = frozenset({"embed_tokens", "final_norm", "lm_head"})
_FULL_ATTENTION_ENTRY_NAMES = frozenset(
    {"down", "gate_up", "input_norm", "o", "post_norm", "qkv"}
)
_GDN_ENTRY_NAMES = frozenset(
    {
        "down",
        "gate_up",
        "gdn_ba",
        "gdn_o",
        "gdn_qkvz",
        "input_norm",
        "post_norm",
    }
)
_FULL_ATTENTION_MOE_ENTRY_NAMES = _FULL_ATTENTION_ENTRY_NAMES - {
    "down",
    "gate_up",
}
_GDN_MOE_ENTRY_NAMES = _GDN_ENTRY_NAMES - {"down", "gate_up"}
_MOE_ENTRY_PREFIX = re.compile(r"^l([0-9]+)\.moe\.")

__all__ = [
    "ECONOMIC_PROFILE_ADAPTER_ID_V3",
    "ECONOMIC_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3",
    "ECONOMIC_COMPACT_PROJECTION_ABI_V3",
    "ECONOMIC_MLP_ACTIVATION_REPLAY_ABI_V3",
    "GDN_CANONICAL_NORM_INPUT_ABI_V3",
    "ECONOMIC_QUANTIZATION_SEMANTICS_ID_V3",
    "build_economic_execution_profile_v3",
    "economic_static_artifact_digest_v3",
    "economic_verifier_digest_v3",
    "economic_profile_is_compact_v3",
    "economic_profile_has_full_row_escalation_v3",
    "economic_profile_is_lean_v3",
    "economic_profile_uses_canonical_gdn_input_v3",
    "economic_profile_uses_exact_mlp_activation_v3",
    "economic_profile_uses_selected_trace_v3",
    "validate_economic_execution_profile_v3",
]


def _adapter_version(profile_or_version: object) -> object:
    return getattr(profile_or_version, "adapter_version", profile_or_version)


def economic_profile_is_lean_v3(profile_or_version: object) -> bool:
    """Whether a profile uses the authenticated lean execution inventory."""

    return _adapter_version(profile_or_version) in {
        ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
    }


def economic_profile_is_compact_v3(profile_or_version: object) -> bool:
    """Whether coverage is selected by the compact v9 terminal ABI."""

    return _adapter_version(profile_or_version) in {
        ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
    }


def economic_profile_uses_selected_trace_v3(
    profile_or_version: object,
) -> bool:
    """Whether compact hard audits use the shared selected-trace proof."""

    return _adapter_version(profile_or_version) in {
        ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
    }


def economic_profile_has_full_row_escalation_v3(
    profile_or_version: object,
) -> bool:
    """Whether the signed compact profile enables complete-row escalation."""

    return _adapter_version(profile_or_version) in {
        ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
    }


def _digest(label: bytes, *parts: bytes) -> bytes:
    return hashlib.sha256(_DIGEST_DOMAIN + label + b"".join(parts)).digest()


def _entry_bytes(entry: ProjectionManifestEntryV3) -> bytes:
    return json.dumps(
        {
            "in_dim": int(entry.in_dim),
            "name": entry.name,
            "orientation": entry.orientation,
            "out_dim": int(entry.out_dim),
            "root": entry.root.hex(),
            "scale_bits": int(entry.scale_bits),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _entry_digest(entry: ProjectionManifestEntryV3) -> bytes:
    return _digest(b"MANIFEST_ENTRY/", _entry_bytes(entry))


def economic_static_artifact_digest_v3(
    manifest: ProjectionManifestV3,
) -> bytes:
    """Digest of the complete validator-owned economic static artifact."""

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("economic manifest has an unexpected type")
    return _digest(b"STATIC_ARTIFACT/", manifest.digest())


def economic_verifier_digest_v3(
    *,
    lean: bool = False,
    compact_projection: bool = False,
    compact_full_row_escalation: bool = False,
    selected_trace: bool = False,
    exact_mlp_activation: bool = True,
    canonical_gdn_input: bool | None = None,
) -> bytes:
    """Versioned verifier identity for the non-keyed economic adapter."""

    if canonical_gdn_input is None:
        canonical_gdn_input = exact_mlp_activation
    if canonical_gdn_input and not exact_mlp_activation:
        raise ProofV3Error(
            "canonical GDN inputs require exact MLP activation semantics"
        )

    if compact_projection and not lean:
        raise ProofV3Error(
            "compact projection verification requires lean execution anchors"
        )
    if selected_trace and not compact_projection:
        raise ProofV3Error(
            "selected-trace verification requires compact projection"
        )
    selected_trace_binding = b""
    if selected_trace:
        from verallm.proof_v3.goldilocks_execution_anchor_pcs_binding import (
            GOLDILOCKS_EXECUTION_ANCHOR_PCS_BINDING_ABI_V3,
        )
        from verallm.proof_v3.goldilocks_selected_trace import (
            GOLDILOCKS_SELECTED_TRACE_ABI_V3,
        )
        from verallm.proof_v3.goldilocks_selected_trace_wire import (
            GOLDILOCKS_SELECTED_TRACE_WIRE_ABI_V3,
        )

        selected_trace_binding = (
            b"\0"
            + GOLDILOCKS_SELECTED_TRACE_ABI_V3.encode("ascii")
            + b"\0"
            + GOLDILOCKS_SELECTED_TRACE_WIRE_ABI_V3.encode("ascii")
            + b"\0"
            + GOLDILOCKS_EXECUTION_ANCHOR_PCS_BINDING_ABI_V3.encode("ascii")
        )
    return _digest(
        b"VERIFIER/",
        b"economic_recompute_v3",
        (
            (
                ECONOMIC_COMPACT_SELECTION_ABI_V3
                if compact_full_row_escalation
                else ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3
            )
            if compact_projection
            else ECONOMIC_STREAMING_SELECTION_ABI_V3
        ).encode("ascii"),
        (
            LEAN_EXECUTION_ANCHOR_ABI_V3.encode("ascii")
            + b"\0"
            + GDN_DECODE_CORRIDOR_ABI_V3.encode("ascii")
            + (
                b"\0"
                + ECONOMIC_MLP_ACTIVATION_REPLAY_ABI_V3.encode("ascii")
                + (
                    b"\0"
                    + GDN_CANONICAL_NORM_INPUT_ABI_V3.encode("ascii")
                    if canonical_gdn_input
                    else b""
                )
                if exact_mlp_activation and not selected_trace
                else b""
            )
            + b"\0"
            + LEAN_PROJECTION_BATCH_ABI_V3.encode("ascii")
            + b"\0"
            + LEAN_PROJECTION_NATIVE_FOLD_ABI_V3.encode("ascii")
            + (
                b"\0"
                + ECONOMIC_COMPACT_PROJECTION_ABI_V3.encode("ascii")
                + b"\0"
                + LEAN_PROJECTION_FOLD_ABI_V3.encode("ascii")
                + b"\0"
                + struct.pack(
                    "<II",
                    ECONOMIC_COMPACT_PCS_QUERY_COUNT_V3,
                    ECONOMIC_ESCALATION_PCS_QUERY_COUNT_V3,
                )
                + selected_trace_binding
                if compact_projection
                else b""
            )
            if lean
            else b"full_inventory.v1"
        ),
    )


def economic_profile_uses_exact_mlp_activation_v3(
    profile: ExecutionSecurityProfileV3,
) -> bool:
    """Whether the signed verifier digest selects exact fused-SwiGLU cells."""

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("economic profile has an unexpected type")
    if (
        not economic_profile_is_lean_v3(profile)
        or economic_profile_uses_selected_trace_v3(profile)
    ):
        return False
    common = dict(
        lean=True,
        compact_projection=economic_profile_is_compact_v3(profile),
        compact_full_row_escalation=(
            economic_profile_has_full_row_escalation_v3(profile)
        ),
        selected_trace=False,
        exact_mlp_activation=True,
    )
    return profile.verifier_key_digest in {
        economic_verifier_digest_v3(
            **common,
            canonical_gdn_input=True,
        ),
        economic_verifier_digest_v3(
            **common,
            canonical_gdn_input=False,
        ),
    }


def economic_profile_uses_canonical_gdn_input_v3(
    profile: ExecutionSecurityProfileV3,
) -> bool:
    """Whether the profile selects authenticated canonical GDN input rows."""

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("economic profile has an unexpected type")
    if (
        not economic_profile_is_lean_v3(profile)
        or economic_profile_uses_selected_trace_v3(profile)
    ):
        return False
    return profile.verifier_key_digest == economic_verifier_digest_v3(
        lean=True,
        compact_projection=economic_profile_is_compact_v3(profile),
        compact_full_row_escalation=(
            economic_profile_has_full_row_escalation_v3(profile)
        ),
        selected_trace=False,
        exact_mlp_activation=True,
        canonical_gdn_input=True,
    )


def _manifest_inventory(
    manifest: ProjectionManifestV3,
    layer_kinds: Sequence[str],
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None,
    *,
    allow_unvalidated_moe: bool = False,
) -> tuple[
    dict[str, ProjectionManifestEntryV3],
    dict[int, dict[str, ProjectionManifestEntryV3]],
    dict[MoeProjectionIdentityV3, ProjectionManifestEntryV3],
]:
    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("economic manifest has an unexpected type")
    if (
        not isinstance(manifest.model_id, str)
        or not manifest.model_id
        or not isinstance(manifest.chunk_size, int)
        or manifest.chunk_size < 1
        or manifest.lm_head_binding not in (
            1,
            LM_HEAD_CATALOG_BINDING_V3,
        )
    ):
        raise ProofV3Error(
            "economic manifest header or mandatory LM-head binding is malformed"
        )
    by_name: dict[str, ProjectionManifestEntryV3] = {}
    by_layer: dict[int, dict[str, ProjectionManifestEntryV3]] = {}
    moe_entry_names: set[str] = set()
    for entry in manifest.entries:
        if (
            not isinstance(entry, ProjectionManifestEntryV3)
            or not isinstance(entry.name, str)
            or not entry.name
            or entry.name in by_name
            or not isinstance(entry.root, bytes)
            or len(entry.root) != 32
            or entry.root == bytes(32)
            or entry.orientation != "out_in"
            or type(entry.in_dim) is not int
            or type(entry.out_dim) is not int
            or entry.in_dim < 1
            or entry.out_dim < 1
            or type(entry.scale_bits) is not int
            or entry.scale_bits <= 0
        ):
            raise ProofV3Error("economic manifest entry is malformed")
        by_name[entry.name] = entry
        if _MOE_ENTRY_PREFIX.match(entry.name) is not None:
            if parse_moe_projection_name_v3(entry.name) is None:
                raise ProofV3Error(
                    "economic manifest contains an unsupported MoE projection name"
                )
            moe_entry_names.add(entry.name)
            continue
        match = _LAYER_ENTRY.fullmatch(entry.name)
        if match is not None:
            layer_index = int(match.group(1))
            suffix = match.group(2)
            layer = by_layer.setdefault(layer_index, {})
            if suffix in layer:
                raise ProofV3Error("economic manifest layer entry is duplicated")
            layer[suffix] = entry

    layer_entry_names = {
        entry.name for entries in by_layer.values() for entry in entries.values()
    }
    global_names = set(by_name) - layer_entry_names - moe_entry_names
    if global_names != _GLOBAL_ENTRY_NAMES:
        raise ProofV3Error(
            "economic manifest global inventory is incomplete or unsupported"
        )
    kinds = tuple(str(kind) for kind in layer_kinds)
    if (
        not kinds
        or any(kind not in {"full_attention", "gdn"} for kind in kinds)
        or tuple(sorted(by_layer)) != tuple(range(len(kinds)))
    ):
        raise ProofV3Error(
            "economic manifest layer inventory does not match the profile"
        )
    if moe_runtime_semantics is not None:
        moe_layers = frozenset(
            item.layer_index for item in moe_runtime_semantics.layers
        )
    elif allow_unvalidated_moe:
        moe_layers = frozenset(
            int(match.group(1))
            for name in moe_entry_names
            if (match := _MOE_ENTRY_PREFIX.match(name)) is not None
        )
    else:
        moe_layers = frozenset()
    if any(layer_index >= len(kinds) for layer_index in moe_layers):
        raise ProofV3Error("MoE runtime semantics reference an unknown layer")
    if (
        moe_runtime_semantics is None
        and not allow_unvalidated_moe
        and (moe_entry_names or manifest.moe_runtime_semantics_digest)
    ):
        raise ProofV3Error(
            "economic manifest carries unauthenticated MoE inventory"
        )
    for layer_index, kind in enumerate(kinds):
        actual = set(by_layer[layer_index])
        uses_moe = layer_index in moe_layers
        if kind == "full_attention":
            expected = (
                _FULL_ATTENTION_MOE_ENTRY_NAMES
                if uses_moe
                else _FULL_ATTENTION_ENTRY_NAMES
            )
            if actual not in (expected, expected | {"qkv_bias"}):
                raise ProofV3Error(
                    f"full-attention layer {layer_index} has an unsupported "
                    "manifest inventory"
                )
        elif actual != (
            _GDN_MOE_ENTRY_NAMES if uses_moe else _GDN_ENTRY_NAMES
        ):
            raise ProofV3Error(
                f"GDN layer {layer_index} has an unsupported manifest inventory"
            )
    if moe_runtime_semantics is None:
        if allow_unvalidated_moe:
            if bool(moe_entry_names) != bool(
                manifest.moe_runtime_semantics_digest
            ):
                raise ProofV3Error(
                    "economic manifest MoE inventory and semantics digest disagree"
                )
        moe_entries = {}
    else:
        if (
            moe_runtime_semantics.digest()
            != manifest.moe_runtime_semantics_digest
        ):
            raise ProofV3Error(
                "MoE profile lacks exact authenticated runtime semantics"
            )
        moe_entries = validate_moe_projection_inventory_v3(
            manifest=manifest,
            semantics=moe_runtime_semantics,
        )
        if {entry.name for entry in moe_entries.values()} != moe_entry_names:
            raise ProofV3Error(
                "economic manifest contains unsupported MoE projection names"
            )
    return by_name, by_layer, moe_entries


def infer_economic_manifest_layer_kinds_v3(
    manifest: ProjectionManifestV3,
) -> tuple[str, ...]:
    """Derive the exact layer architecture from a signed manifest inventory.

    This is the weightless validator path. It does not infer from model names
    or configuration supplied by a miner: every accepted layer must have one
    of the complete, disjoint inventories enforced by
    :func:`_manifest_inventory`.
    """

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("economic manifest has an unexpected type")
    by_layer: dict[int, set[str]] = {}
    for entry in manifest.entries:
        if not isinstance(entry, ProjectionManifestEntryV3):
            raise ProofV3Error("economic manifest entry is malformed")
        if _MOE_ENTRY_PREFIX.match(entry.name) is not None:
            if parse_moe_projection_name_v3(entry.name) is None:
                raise ProofV3Error(
                    "economic manifest contains an unsupported MoE projection name"
                )
            continue
        match = _LAYER_ENTRY.fullmatch(entry.name)
        if match is None:
            continue
        layer_index = int(match.group(1))
        suffix = match.group(2)
        layer = by_layer.setdefault(layer_index, set())
        if suffix in layer:
            raise ProofV3Error("economic manifest layer entry is duplicated")
        layer.add(suffix)
    if not by_layer or tuple(sorted(by_layer)) != tuple(range(len(by_layer))):
        raise ProofV3Error(
            "economic manifest layer inventory is not contiguous"
        )
    kinds: list[str] = []
    for layer_index in range(len(by_layer)):
        actual = by_layer[layer_index]
        if actual in (
            _FULL_ATTENTION_ENTRY_NAMES,
            _FULL_ATTENTION_ENTRY_NAMES | {"qkv_bias"},
            _FULL_ATTENTION_MOE_ENTRY_NAMES,
            _FULL_ATTENTION_MOE_ENTRY_NAMES | {"qkv_bias"},
        ):
            kinds.append("full_attention")
        elif actual in {_GDN_ENTRY_NAMES, _GDN_MOE_ENTRY_NAMES}:
            kinds.append("gdn")
        else:
            raise ProofV3Error(
                f"layer {layer_index} has an unsupported manifest inventory"
            )
    result = tuple(kinds)
    _manifest_inventory(
        manifest,
        result,
        allow_unvalidated_moe=True,
    )
    return result


def _attention_geometry(
    *,
    manifest: ProjectionManifestV3,
    by_layer: dict[int, dict[str, ProjectionManifestEntryV3]],
    layer_kinds: tuple[str, ...],
    calibration_set: ScoredCalibrationSetV3 | None,
    semantics: AttentionRuntimeSemanticsV3 | None,
) -> tuple[int, int, int]:
    full_layers = tuple(
        index for index, kind in enumerate(layer_kinds)
        if kind == "full_attention"
    )
    if not full_layers:
        if (
            calibration_set is not None
            or semantics is not None
            or manifest.attn_runtime_semantics_digest
            or manifest.attn_calibration_set_digest
        ):
            raise ProofV3Error(
                "non-attention profile carries attention qualification data"
            )
        return 0, 0, 0
    if (
        manifest.attn_audit_required != 1
        or manifest.attn_scheme != 2
        or calibration_set is None
        or calibration_set.digest != manifest.attn_calibration_set_digest
        or semantics is None
        or semantics.digest() != manifest.attn_runtime_semantics_digest
    ):
        raise ProofV3Error(
            "full-attention profile lacks exact authenticated qualification data"
        )
    if tuple(sorted(calibration_set.bands[0].calibration.layers)) != full_layers:
        raise ProofV3Error(
            "attention calibration does not cover the exact layer inventory"
        )
    head_count = 0
    head_dim = 0
    for band in calibration_set.bands:
        if tuple(sorted(band.calibration.layers)) != full_layers:
            raise ProofV3Error(
                "attention calibration bands disagree on the layer inventory"
            )
        for layer_index in full_layers:
            heads = tuple(band.calibration.heads_for(layer_index))
            if not heads:
                raise ProofV3Error("attention calibration has no heads")
            dimensions = {int(params.head_dim) for params, _bounds in heads}
            if len(dimensions) != 1:
                raise ProofV3Error(
                    "attention calibration head dimensions are inconsistent"
                )
            current_dim = next(iter(dimensions))
            if head_count == 0:
                head_count, head_dim = len(heads), current_dim
            elif (len(heads), current_dim) != (head_count, head_dim):
                raise ProofV3Error(
                    "attention calibration geometry is inconsistent"
                )
    if semantics.rotary_dimension > head_dim:
        raise ProofV3Error(
            "attention runtime semantics exceed the calibrated head dimension"
        )
    if (
        semantics.version == ATTENTION_RUNTIME_SEMANTICS_VERSION_V3
        and semantics.rope_coefficient_row_count
        < max(int(band.hi) for band in calibration_set.bands)
    ):
        raise ProofV3Error(
            "attention runtime coefficient table does not cover the "
            "calibrated context"
        )
    multiplier = 2 if semantics.qkv_layout_id == Q_GATE_INTERLEAVED_LAYOUT_V3 else 1
    kv_count: int | None = None
    for layer_index in full_layers:
        qkv = by_layer[layer_index]["qkv"]
        remainder = qkv.out_dim - multiplier * head_count * head_dim
        if remainder <= 0 or remainder % (2 * head_dim):
            raise ProofV3Error(
                f"attention layer {layer_index} QKV width disagrees with "
                "authenticated head geometry"
            )
        current_kv = remainder // (2 * head_dim)
        if head_count % current_kv:
            raise ProofV3Error(
                "attention query heads do not divide into authenticated KV groups"
            )
        if kv_count is None:
            kv_count = current_kv
        elif current_kv != kv_count:
            raise ProofV3Error(
                "full-attention layers disagree on KV-head geometry"
            )
    assert kv_count is not None
    return head_count, kv_count, head_dim


def _operation_reference(
    entry: ProjectionManifestEntryV3,
    *,
    layer_index: int,
    operation_id: str,
) -> RegisteredOperationReferenceV3:
    return RegisteredOperationReferenceV3(
        layer_index=layer_index,
        operation_id=operation_id,
        rows=entry.in_dim,
        cols=entry.out_dim,
        weight_commitment=entry.root,
        descriptor_digest=_entry_digest(entry),
    )


def _table_binding(
    *,
    binding_id: str,
    subject_kind: str,
    subject_digest: bytes,
    leaf_start: int,
    leaf_count: int,
) -> StaticTableBindingV3:
    page_count = max(1, math.ceil(leaf_count / 256))
    descriptor = _digest(
        b"STATIC_TABLE/",
        binding_id.encode("ascii"),
        subject_digest,
        int(leaf_start).to_bytes(8, "little"),
        int(leaf_count).to_bytes(8, "little"),
    )
    return StaticTableBindingV3(
        binding_id=binding_id,
        subject_kind=subject_kind,
        operation_descriptor_digest=(
            subject_digest if subject_kind == "operation" else None
        ),
        static_binding_id=(
            None if subject_kind == "operation"
            else binding_id.removeprefix("static.")
        ),
        static_binding_digest=(
            None if subject_kind == "operation" else subject_digest
        ),
        table_id=binding_id,
        static_table_descriptor_digest=descriptor,
        logical_leaf_start=leaf_start,
        logical_leaf_count=leaf_count,
        page_start=leaf_start // 256,
        page_count=page_count,
        element_encoding_id="int8.row_major.v1",
        scale_encoding_id="float64.bits.v1",
    )


def build_economic_execution_profile_v3(
    *,
    manifest: ProjectionManifestV3,
    layer_kinds: Sequence[str],
    calibration_set: ScoredCalibrationSetV3 | None,
    attention_runtime_semantics: AttentionRuntimeSemanticsV3 | None,
    gdn_runtime_semantics: GdnRuntimeSemanticsV3 | None,
    tokenizer_binding_digest: bytes,
    runtime_encoding_id: str,
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None,
    max_context_tokens: int | None = None,
    max_decode_tokens: int = 4_096,
    streaming: bool = True,
    lean: bool = False,
    compact_projection: bool = False,
    compact_full_row_escalation: bool = False,
    selected_trace: bool = False,
    prefix_cache_sharing: bool = False,
    prefix_cache_page_tokens: int | None = None,
    prefix_cache_k_cell_delta_max: int | None = None,
    prefix_cache_k_row_sq_delta_max: int | None = None,
    prefix_cache_v_cell_delta_max: int | None = None,
    prefix_cache_v_row_sq_delta_max: int | None = None,
) -> ExecutionSecurityProfileV3:
    """Build the exact signed economic profile for one qualified model."""

    from verallm.proof_v3.economic_recompute_adapter import (
        ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
    )

    if not isinstance(lean, bool):
        raise ProofV3Error("economic lean-profile selector must be boolean")
    if not isinstance(compact_projection, bool):
        raise ProofV3Error(
            "economic compact-projection selector must be boolean"
        )
    if not isinstance(compact_full_row_escalation, bool):
        raise ProofV3Error(
            "economic compact escalation selector must be boolean"
        )
    if not isinstance(selected_trace, bool):
        raise ProofV3Error(
            "economic selected-trace selector must be boolean"
        )
    if not isinstance(prefix_cache_sharing, bool):
        raise ProofV3Error(
            "economic prefix-cache selector must be boolean"
        )
    if prefix_cache_sharing:
        if (
            isinstance(prefix_cache_page_tokens, bool)
            or not isinstance(prefix_cache_page_tokens, int)
            or not 1 <= prefix_cache_page_tokens < 1 << 32
        ):
            raise ProofV3Error(
                "prefix-cache sharing requires an explicit qualified page width"
            )
        replay_bounds = (
            prefix_cache_k_cell_delta_max,
            prefix_cache_k_row_sq_delta_max,
            prefix_cache_v_cell_delta_max,
            prefix_cache_v_row_sq_delta_max,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in replay_bounds):
            raise ProofV3Error(
                "prefix-cache sharing requires explicit qualified replay corridors"
            )
    elif prefix_cache_page_tokens is not None:
        raise ProofV3Error(
            "prefix-cache page width requires prefix-cache sharing"
        )
    elif any(
        value is not None
        for value in (
            prefix_cache_k_cell_delta_max,
            prefix_cache_k_row_sq_delta_max,
            prefix_cache_v_cell_delta_max,
            prefix_cache_v_row_sq_delta_max,
        )
    ):
        raise ProofV3Error(
            "prefix-cache replay corridors require prefix-cache sharing"
        )
    if compact_full_row_escalation and not compact_projection:
        raise ProofV3Error(
            "complete-row escalation requires compact projection"
        )
    if compact_projection and not lean:
        raise ProofV3Error(
            "compact projection profiles require lean execution anchors"
        )
    if selected_trace and not compact_projection:
        raise ProofV3Error(
            "selected-trace profiles require compact projection"
        )
    if prefix_cache_sharing and (
        not lean
        or not compact_projection
        or compact_full_row_escalation
        or selected_trace
    ):
        raise ProofV3Error(
            "prefix-cache sharing requires compact-v9 projection proofs"
        )
    if lean and not streaming:
        raise ProofV3Error("lean economic profiles require streaming selection")
    kinds = tuple(str(kind) for kind in layer_kinds)
    by_name, by_layer, moe_entries = _manifest_inventory(
        manifest,
        kinds,
        moe_runtime_semantics,
    )
    if (
        not isinstance(tokenizer_binding_digest, bytes)
        or len(tokenizer_binding_digest) != 32
        or tokenizer_binding_digest == bytes(32)
    ):
        raise ProofV3Error("economic tokenizer binding digest is malformed")
    if runtime_encoding_id not in {"fp16.v1", "bf16.v1"}:
        raise ProofV3Error("economic runtime encoding is not qualified")
    if (
        type(max_decode_tokens) is not int
        or not 1 <= max_decode_tokens < 1 << 32
    ):
        raise ProofV3Error("economic maximum decode count is malformed")
    if max_context_tokens is None:
        if calibration_set is None:
            raise ProofV3Error(
                "economic maximum context needs an authenticated calibration set"
            )
        max_context_tokens = calibration_set.bands[-1].hi
    if (
        type(max_context_tokens) is not int
        or not 1 <= max_context_tokens < 1 << 32
        or max_context_tokens + max_decode_tokens >= 1 << 32
    ):
        raise ProofV3Error("economic maximum context count is malformed")
    if (
        calibration_set is not None
        and max_context_tokens != calibration_set.bands[-1].hi
    ):
        raise ProofV3Error(
            "economic context limit must equal the calibrated domain"
        )

    hidden = by_name["embed_tokens"].in_dim
    vocab = by_name["embed_tokens"].out_dim
    moe_layers = frozenset(
        item.layer_index for item in moe_runtime_semantics.layers
    ) if moe_runtime_semantics is not None else frozenset()
    if moe_runtime_semantics is not None and (
        moe_runtime_semantics.hidden_size != hidden
        or moe_runtime_semantics.runtime_encoding_id != runtime_encoding_id
    ):
        raise ProofV3Error(
            "MoE runtime geometry or encoding disagrees with the execution profile"
        )
    if (
        by_name["lm_head"].in_dim != hidden
        or by_name["lm_head"].out_dim != vocab
        or by_name["final_norm"].in_dim != hidden
    ):
        raise ProofV3Error(
            "embedding, final norm and LM-head dimensions are inconsistent"
        )
    for layer_index, entries in by_layer.items():
        malformed = (
            entries["input_norm"].in_dim != hidden
            or entries["post_norm"].in_dim != hidden
        )
        if layer_index not in moe_layers:
            malformed = malformed or (
                entries["gate_up"].in_dim != hidden
                or entries["down"].out_dim != hidden
                or entries["gate_up"].out_dim != 2 * entries["down"].in_dim
            )
        if malformed:
            raise ProofV3Error(
                f"layer {layer_index} dimensions are inconsistent"
            )

    nh, nkv, head_dim = _attention_geometry(
        manifest=manifest,
        by_layer=by_layer,
        layer_kinds=kinds,
        calibration_set=calibration_set,
        semantics=attention_runtime_semantics,
    )
    gdn_layers = tuple(
        index for index, kind in enumerate(kinds) if kind == "gdn"
    )
    if gdn_layers:
        if (
            gdn_runtime_semantics is None
            or gdn_runtime_semantics.digest()
            != manifest.gdn_runtime_semantics_digest
            or tuple(item.layer_index for item in gdn_runtime_semantics.layers)
            != gdn_layers
        ):
            raise ProofV3Error(
                "GDN profile lacks exact authenticated runtime semantics"
            )
        if any(
            item.runtime_encoding_id != runtime_encoding_id
            for item in gdn_runtime_semantics.layers
        ):
            raise ProofV3Error(
                "GDN runtime encoding disagrees with the execution profile"
            )
        if (
            gdn_runtime_semantics.decode_checkpoint_stride
            and max_decode_tokens
            > gdn_runtime_semantics.max_hard_audit_decode_tokens
        ):
            raise ProofV3Error(
                "economic decode limit exceeds the signed GDN hard-audit "
                "reach"
            )
    elif gdn_runtime_semantics is not None or manifest.gdn_runtime_semantics_digest:
        raise ProofV3Error("dense profile carries unexpected GDN semantics")

    anchor_entries = {
        layer_index: entries[
            "qkv" if kinds[layer_index] == "full_attention" else "gdn_qkvz"
        ]
        for layer_index, entries in by_layer.items()
    }
    references = tuple(
        sorted(
            (
                *(
                    _operation_reference(
                        anchor_entries[layer_index],
                        layer_index=layer_index,
                        operation_id=anchor_entries[layer_index].name,
                    )
                    for layer_index in range(len(kinds))
                ),
                _operation_reference(
                    by_name["lm_head"],
                    layer_index=-1,
                    operation_id="model.lm_head",
                ),
            ),
            key=lambda item: item.sort_key(),
        )
    )
    reference_by_layer = {
        item.layer_index: item for item in references if item.layer_index >= 0
    }
    lm_reference = next(item for item in references if item.layer_index == -1)

    inventory_digest = _digest(
        b"MANIFEST_INVENTORY/",
        manifest.digest(),
        *(kind.encode("ascii") + b"\0" for kind in kinds),
    )
    static_bindings: list[StaticParameterBindingV3] = [
        StaticParameterBindingV3(
            "embedding_params",
            "embedding",
            _entry_digest(by_name["embed_tokens"]),
            -1,
        ),
        StaticParameterBindingV3(
            "final_norm_params",
            "final_norm",
            _entry_digest(by_name["final_norm"]),
            -1,
        ),
        StaticParameterBindingV3(
            "sampler_params",
            "sampler",
            _digest(b"SAMPLER/", manifest.digest()),
            -1,
        ),
    ]
    moe_entries_by_layer: dict[int, list[ProjectionManifestEntryV3]] = {}
    for identity, entry in moe_entries.items():
        moe_entries_by_layer.setdefault(identity.layer_index, []).append(entry)
    for layer_index, kind in enumerate(kinds):
        complete_layer_entries = tuple(by_layer[layer_index].values()) + tuple(
            moe_entries_by_layer.get(layer_index, ())
        )
        layer_digest = _digest(
            b"LAYER_INVENTORY/",
            inventory_digest,
            layer_index.to_bytes(4, "little"),
            *(
                _entry_bytes(entry)
                for entry in sorted(
                    complete_layer_entries,
                    key=lambda item: item.name,
                )
            ),
        )
        semantics_digest = (
            attention_runtime_semantics.digest()
            if kind == "full_attention"
            else gdn_runtime_semantics.digest()
        )
        static_bindings.append(
            StaticParameterBindingV3(
                f"transition_params_l{layer_index}",
                "attention" if kind == "full_attention" else "transition",
                _digest(b"TRANSITION/", layer_digest, semantics_digest),
                layer_index,
            )
        )
        if layer_index in moe_layers:
            assert moe_runtime_semantics is not None
            static_bindings.append(
                StaticParameterBindingV3(
                    f"moe_params_l{layer_index}",
                    "moe",
                    _digest(
                        b"MOE/",
                        layer_digest,
                        moe_runtime_semantics.digest(),
                    ),
                    layer_index,
                )
            )
        else:
            static_bindings.append(
                StaticParameterBindingV3(
                    f"bridge_params_l{layer_index}",
                    "bridge",
                    _digest(b"BRIDGE/", layer_digest),
                    layer_index,
                )
            )
    static_bindings = sorted(static_bindings, key=lambda item: item.binding_id)

    table_bindings: list[StaticTableBindingV3] = []
    leaf_start = 0
    for reference in references:
        leaf_start = ((leaf_start + 255) // 256) * 256
        count = reference.rows * reference.cols
        binding_id = (
            f"op.l{reference.layer_index}.{reference.operation_id}"
            if reference.layer_index >= 0
            else "op.model.lm_head"
        )
        table_bindings.append(
            _table_binding(
                binding_id=binding_id,
                subject_kind="operation",
                subject_digest=reference.descriptor_digest,
                leaf_start=leaf_start,
                leaf_count=count,
            )
        )
        leaf_start += count
    for binding in static_bindings:
        leaf_start = ((leaf_start + 255) // 256) * 256
        table_bindings.append(
            _table_binding(
                binding_id=f"static.{binding.binding_id}",
                subject_kind="static_parameter",
                subject_digest=binding.binding_digest,
                leaf_start=leaf_start,
                leaf_count=1,
            )
        )
        leaf_start += 1

    tensors: list[TensorDescriptorV3] = [
        TensorDescriptorV3(
            "prompt_tokens",
            ("context_tokens",),
            "token_ids.u32.v1",
            "sequence.v1",
            "exact.v1",
            "request_input",
        ),
        TensorDescriptorV3(
            "state_l0",
            ("sequence_tokens", "hidden_dim"),
            runtime_encoding_id,
            "row_major.v1",
            "fixed.v1",
            "runtime_state",
        ),
        TensorDescriptorV3(
            "final_hidden",
            ("decode_tokens", "hidden_dim"),
            runtime_encoding_id,
            "row_major.v1",
            "fixed.v1",
            "final_hidden",
        ),
        TensorDescriptorV3(
            "logits",
            ("decode_tokens", "vocab_size"),
            runtime_encoding_id,
            "row_major.v1",
            "fixed.v1",
            "logits",
        ),
        TensorDescriptorV3(
            "token",
            ("decode_tokens",),
            "token_ids.u32.v1",
            "sequence.v1",
            "exact.v1",
            "token_output",
        ),
    ]
    layer_caches: list[LayerCacheRelationV3] = []
    nodes: list[ExecutionRelationNodeV3] = [
        ExecutionRelationNodeV3(
            "embedding",
            "embedding",
            "embedding.v1",
            ("prompt_tokens",),
            ("state_l0",),
            "exact",
            ("embedding_params",),
        )
    ]
    layer_audits: list[LayerAuditPlanV3] = []
    for layer_index, kind in enumerate(kinds):
        anchor = anchor_entries[layer_index]
        anchor_tensor = f"transition_anchor_l{layer_index}"
        transition_tensor = f"transition_out_l{layer_index}"
        state_out = f"state_l{layer_index + 1}"
        tensors.extend(
            (
                TensorDescriptorV3(
                    anchor_tensor,
                    ("sequence_tokens", anchor.out_dim),
                    runtime_encoding_id,
                    "row_major.v1",
                    "fixed.v1",
                    "runtime_state",
                ),
                TensorDescriptorV3(
                    transition_tensor,
                    ("sequence_tokens", "hidden_dim"),
                    runtime_encoding_id,
                    "row_major.v1",
                    "fixed.v1",
                    "runtime_state",
                ),
                TensorDescriptorV3(
                    state_out,
                    ("sequence_tokens", "hidden_dim"),
                    runtime_encoding_id,
                    "row_major.v1",
                    "fixed.v1",
                    "runtime_state",
                ),
            )
        )
        if kind == "full_attention":
            cache_tensor_ids = (
                f"cache_key_l{layer_index}",
                f"cache_value_l{layer_index}",
            )
            kv_width = nkv * head_dim
            for tensor_id in cache_tensor_ids:
                tensors.append(
                    TensorDescriptorV3(
                        tensor_id,
                        ("sequence_tokens", kv_width),
                        runtime_encoding_id,
                        "paged_kv.logical.v1",
                        "fixed.v1",
                        "cache",
                    )
                )
            layer_caches.append(
                LayerCacheRelationV3(
                    layer_index=layer_index,
                    cache_kind="attention_kv",
                    cache_semantics_id=PAGED_KV_CACHE_SEMANTICS_V3,
                    key_tensor_id=cache_tensor_ids[0],
                    value_tensor_id=cache_tensor_ids[1],
                )
            )
            transition_id = f"full_attention_l{layer_index}"
            relation_id = "full_attention"
            adapter_id = "full_attention.v1"
        else:
            assert gdn_runtime_semantics is not None
            layer_semantics = gdn_runtime_semantics.layer_for(layer_index)
            params = layer_semantics.parameters()
            conv_width = (
                (params.conv_kernel_size - 1)
                * (
                    2 * params.num_key_heads * params.key_head_dim
                    + params.num_value_heads * params.value_head_dim
                )
            )
            recurrent_width = (
                params.num_value_heads
                * params.value_head_dim
                * params.key_head_dim
            )
            cache_tensor_ids = (
                f"gdn_conv_state_l{layer_index}",
                f"gdn_recurrent_state_l{layer_index}",
            )
            tensors.extend(
                (
                    TensorDescriptorV3(
                        cache_tensor_ids[0],
                        ("sequence_tokens", conv_width),
                        layer_semantics.conv_state_encoding_id,
                        "gdn.state_trace.v1",
                        "fixed.v1",
                        "cache",
                    ),
                    TensorDescriptorV3(
                        cache_tensor_ids[1],
                        ("sequence_tokens", recurrent_width),
                        layer_semantics.recurrent_state_encoding_id,
                        "gdn.state_trace.v1",
                        "fixed.v1",
                        "cache",
                    ),
                )
            )
            layer_caches.append(
                LayerCacheRelationV3(
                    layer_index=layer_index,
                    cache_kind="gdn_state",
                    cache_semantics_id=GDN_STATE_CACHE_SEMANTICS_V3,
                    state_tensor_ids=cache_tensor_ids,
                )
            )
            transition_id = f"gdn_l{layer_index}"
            relation_id = "gdn"
            adapter_id = "gdn.v1"
        if layer_index in moe_layers:
            assert moe_runtime_semantics is not None
            bridge_id = f"moe_l{layer_index}"
            bridge_node = ExecutionRelationNodeV3(
                bridge_id,
                "moe",
                moe_runtime_semantics.adapter_id,
                (
                    f"state_l{layer_index}",
                    transition_tensor,
                    *cache_tensor_ids,
                ),
                (state_out,),
                "exact",
                (f"moe_params_l{layer_index}",),
                layer_index=layer_index,
            )
        else:
            bridge_id = f"bridge_l{layer_index}"
            bridge_node = ExecutionRelationNodeV3(
                bridge_id,
                "bridge",
                "residual.bridge.v1",
                (transition_tensor, *cache_tensor_ids),
                (state_out,),
                "exact",
                (f"bridge_params_l{layer_index}",),
                layer_index=layer_index,
            )
        nodes.extend(
            (
                ExecutionRelationNodeV3(
                    f"linear_l{layer_index}",
                    "linear",
                    "global.linear.v1",
                    (f"state_l{layer_index}",),
                    (anchor_tensor,),
                    "exact",
                    operation_reference=reference_by_layer[layer_index],
                    layer_index=layer_index,
                ),
                ExecutionRelationNodeV3(
                    transition_id,
                    relation_id,
                    adapter_id,
                    (anchor_tensor,),
                    (transition_tensor, *cache_tensor_ids),
                    "exact",
                    (f"transition_params_l{layer_index}",),
                    layer_index=layer_index,
                ),
                bridge_node,
            )
        )
        layer_audits.append(
            LayerAuditPlanV3(
                layer_index=layer_index,
                transition_node_id=transition_id,
                bridge_node_id=bridge_id,
                required_operation_references=(reference_by_layer[layer_index],),
                attention_query_head_count=nh if kind == "full_attention" else 0,
                attention_key_value_head_count=(
                    nkv if kind == "full_attention" else 0
                ),
                attention_head_dimension=(
                    head_dim if kind == "full_attention" else 0
                ),
                attention_semantics_id=(
                    attention_runtime_semantics.adapter_id
                    if kind == "full_attention"
                    else None
                ),
            )
        )
    nodes.extend(
        (
            ExecutionRelationNodeV3(
                "final_norm",
                "final_norm",
                "final_norm.v1",
                (f"state_l{len(kinds)}",),
                ("final_hidden",),
                "exact",
                ("final_norm_params",),
            ),
            ExecutionRelationNodeV3(
                "lm_head",
                "linear",
                "global.linear.v1",
                ("final_hidden",),
                ("logits",),
                "exact",
                operation_reference=lm_reference,
            ),
            ExecutionRelationNodeV3(
                "sampler",
                "sampler",
                "canonical.sampler.v1",
                ("logits",),
                ("token",),
                "exact",
                ("sampler_params",),
            ),
        )
    )

    selected_layers = min(4, len(kinds))
    full_count = kinds.count("full_attention")
    gdn_count = kinds.count("gdn")
    if full_count and gdn_count:
        # A hybrid hard audit spends its fixed four-layer budget evenly
        # across the two registered transition families.  The post-commit
        # selection remains uniform within each family.
        full_minimum = min(2, full_count)
        gdn_minimum = min(2, gdn_count)
    else:
        full_minimum = min(full_count, selected_layers)
        gdn_minimum = min(gdn_count, selected_layers)
    if calibration_set is not None:
        heads_per_layer = calibration_set.policy.heads_per_layer
    else:
        heads_per_layer = 1
    selection_abi = (
        (
            (
                ECONOMIC_COMPACT_SELECTION_ABI_V3
                if compact_full_row_escalation
                else ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3
            )
            if compact_projection
            else ECONOMIC_STREAMING_SELECTION_ABI_V3
        )
        if streaming
        else ECONOMIC_SELECTION_ABI_V3
    )
    provisional = ExecutionRelationSpecV3(
        relation_abi_id=GLOBAL_EXECUTION_RELATION_ABI_V3,
        capture_abi_digest=_digest(
            b"CAPTURE_ABI/",
            manifest.digest(),
            runtime_encoding_id.encode("ascii"),
            (
                LEAN_EXECUTION_ANCHOR_ABI_V3.encode("ascii")
                + b"\0"
                + LEAN_PROJECTION_BATCH_ABI_V3.encode("ascii")
                + b"\0"
                + LEAN_PROJECTION_NATIVE_FOLD_ABI_V3.encode("ascii")
                + (
                    b"\0"
                    + ECONOMIC_COMPACT_PROJECTION_ABI_V3.encode("ascii")
                    + b"\0"
                    + LEAN_PROJECTION_FOLD_ABI_V3.encode("ascii")
                    if compact_projection
                    else b""
                )
                if lean
                else EXECUTION_ANCHOR_ABI_V3.encode("ascii")
            ),
            SCHEDULER_GEOMETRY_ABI_V3.encode("ascii"),
            *(kind.encode("ascii") + b"\0" for kind in kinds),
        ),
        constraint_system_digest=_digest(b"CONSTRAINT_PLACEHOLDER/"),
        tokenizer_binding_digest=tokenizer_binding_digest,
        sampler_abi_id="canonical.sampler.v1",
        sampler_abi_digest=_digest(b"SAMPLER_ABI/greedy.v1"),
        quantization_binding_digest=_digest(
            b"QUANTIZATION_BINDING/",
            manifest.digest(),
            ECONOMIC_QUANTIZATION_SEMANTICS_ID_V3.encode("ascii"),
        ),
        recursive_accumulator_abi_id=ECONOMIC_RECURSIVE_ACCUMULATOR_ABI_V3,
        recursive_accumulator_digest=_digest(b"OPENING_ACCUMULATOR/v1"),
        prefill_chunk_tokens=min(4_096, max_context_tokens),
        decode_chunk_tokens=min(64, max_decode_tokens),
        request_tensor_id="prompt_tokens",
        final_token_tensor_id="token",
        sequence_domain=SequenceDomainV3(
            context_dimension_id="context_tokens",
            decode_dimension_id="decode_tokens",
            sequence_dimension_id="sequence_tokens",
        ),
        dimensions=tuple(
            sorted(
                (
                    DeclaredDimensionV3(
                        "context_tokens", 1, max_context_tokens
                    ),
                    DeclaredDimensionV3(
                        "decode_tokens", 0, max_decode_tokens
                    ),
                    DeclaredDimensionV3("hidden_dim", hidden, hidden),
                    DeclaredDimensionV3(
                        "sequence_tokens",
                        1,
                        max_context_tokens + max_decode_tokens,
                    ),
                    DeclaredDimensionV3("vocab_size", vocab, vocab),
                ),
                key=lambda item: item.dimension_id,
            )
        ),
        tensors=tuple(sorted(tensors, key=lambda item: item.tensor_id)),
        registered_operations=references,
        static_bindings=tuple(static_bindings),
        static_table_bindings=tuple(
            sorted(table_bindings, key=lambda item: item.binding_id)
        ),
        tolerances=(
            RelationToleranceV3("exact", "exact.integer.v1", 0, 0),
        ),
        nodes=tuple(nodes),
        layer_audits=tuple(layer_audits),
        cache=CacheRelationSpecV3(
            logical_address_abi_id=(
                PREFIX_CACHE_LOGICAL_ADDRESS_ABI_V3
                if prefix_cache_sharing
                else LOGICAL_CACHE_ADDRESS_ABI_V3
            ),
            cache_semantics_id=(
                PREFIX_CACHE_TABLES_SEMANTICS_V3
                if prefix_cache_sharing
                else LOGICAL_CACHE_TABLES_SEMANTICS_V3
            ),
            retention_lease_abi_id=(
                PREFIX_CACHE_RETENTION_LEASE_ABI_V3
                if prefix_cache_sharing
                else GPU_RETENTION_LEASE_ABI_V3
            ),
            page_token_count=(
                prefix_cache_page_tokens
                if prefix_cache_sharing
                else min(128, max_context_tokens + max_decode_tokens)
            ),
            max_cache_tokens=max_context_tokens + max_decode_tokens,
            layer_caches=tuple(layer_caches),
            allows_prefix_cache_sharing=prefix_cache_sharing,
            prefix_cache_k_cell_delta_max=(
                int(prefix_cache_k_cell_delta_max)
                if prefix_cache_sharing
                else 0
            ),
            prefix_cache_k_row_sq_delta_max=(
                int(prefix_cache_k_row_sq_delta_max)
                if prefix_cache_sharing
                else 0
            ),
            prefix_cache_v_cell_delta_max=(
                int(prefix_cache_v_cell_delta_max)
                if prefix_cache_sharing
                else 0
            ),
            prefix_cache_v_row_sq_delta_max=(
                int(prefix_cache_v_row_sq_delta_max)
                if prefix_cache_sharing
                else 0
            ),
        ),
        audit_policy=HardAuditPolicyV3(
            coverage_mode=ECONOMIC_SHELL_COVERAGE_MODE_V3,
            nonce_selection_abi_id=ECONOMIC_NONCE_SELECTION_ABI_V3,
            tier_selection_abi_id=POSTCOMMIT_AUDIT_TIER_ABI_V3,
            selection_abi_id=selection_abi,
            minimum_organic_hard_bps=0,
            minimum_canary_hard_bps=10_000,
            probation_failures=1,
            selected_layer_count=selected_layers,
            minimum_full_attention_layers=full_minimum,
            minimum_gdn_layers=gdn_minimum,
            full_attention_heads_per_layer=heads_per_layer,
            transition_query_rows=4 if streaming else 64,
        ),
    )
    relation = replace(
        provisional,
        constraint_system_digest=_digest(
            b"ECONOMIC_RELATION/",
            constraint_system_relation_projection_bytes_v3(provisional),
        ),
    )
    return ExecutionSecurityProfileV3(
        static_manifest_digest=manifest.digest(),
        static_artifact_digest=economic_static_artifact_digest_v3(manifest),
        adapter_id=ECONOMIC_PROFILE_ADAPTER_ID_V3,
        adapter_version=(
            (
                (
                    ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3
                    if compact_full_row_escalation
                    else ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3
                )
                if selected_trace
                else (
                    ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3
                    if compact_full_row_escalation
                    else ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3
                )
            )
            if compact_projection
            else (
                ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3
                if lean
                else ECONOMIC_PROFILE_ADAPTER_VERSION_V3
            )
        ),
        proof_system_id=ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
        verifier_key_digest=economic_verifier_digest_v3(
            lean=lean,
            compact_projection=compact_projection,
            compact_full_row_escalation=compact_full_row_escalation,
            selected_trace=selected_trace,
        ),
        quantization_semantics_id=ECONOMIC_QUANTIZATION_SEMANTICS_ID_V3,
        max_verified_context_tokens=max_context_tokens,
        max_verified_decode_tokens=max_decode_tokens,
        relation_spec=relation,
    )


def validate_economic_execution_profile_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    manifest: ProjectionManifestV3,
    calibration_set: ScoredCalibrationSetV3 | None,
    attention_runtime_semantics: AttentionRuntimeSemanticsV3 | None,
    gdn_runtime_semantics: GdnRuntimeSemanticsV3 | None,
    tokenizer_binding_digest: bytes | None,
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None,
) -> None:
    """Rebuild and compare the complete signed economic profile."""

    try:
        kinds = tuple(
            "full_attention" if plan.is_full_attention else "gdn"
            for plan in profile.relation_spec.layer_audits
        )
        runtime_encodings = {
            tensor.encoding_id
            for tensor in profile.relation_spec.tensors
            if tensor.commitment_role
            in {"runtime_state", "final_hidden", "logits"}
        }
        if len(runtime_encodings) != 1:
            raise ProofV3VerificationError(
                "economic profile runtime encoding is ambiguous"
            )
        expected = build_economic_execution_profile_v3(
            manifest=manifest,
            layer_kinds=kinds,
            calibration_set=calibration_set,
            attention_runtime_semantics=attention_runtime_semantics,
            gdn_runtime_semantics=gdn_runtime_semantics,
            moe_runtime_semantics=moe_runtime_semantics,
            tokenizer_binding_digest=tokenizer_binding_digest,
            runtime_encoding_id=next(iter(runtime_encodings)),
            max_context_tokens=profile.max_verified_context_tokens,
            max_decode_tokens=profile.max_verified_decode_tokens,
            streaming=(
                profile.relation_spec.audit_policy.selection_abi_id
                in {
                    ECONOMIC_STREAMING_SELECTION_ABI_V3,
                    ECONOMIC_COMPACT_SELECTION_ABI_V3,
                    ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
                }
            ),
            lean=economic_profile_is_lean_v3(profile),
            compact_projection=economic_profile_is_compact_v3(profile),
            compact_full_row_escalation=(
                economic_profile_has_full_row_escalation_v3(profile)
            ),
            selected_trace=economic_profile_uses_selected_trace_v3(profile),
            prefix_cache_sharing=bool(
                profile.relation_spec.cache.allows_prefix_cache_sharing
            ),
            prefix_cache_page_tokens=(
                profile.relation_spec.cache.page_token_count
                if profile.relation_spec.cache.allows_prefix_cache_sharing
                else None
            ),
            prefix_cache_k_cell_delta_max=(
                profile.relation_spec.cache.prefix_cache_k_cell_delta_max
                if profile.relation_spec.cache.allows_prefix_cache_sharing
                else None
            ),
            prefix_cache_k_row_sq_delta_max=(
                profile.relation_spec.cache.prefix_cache_k_row_sq_delta_max
                if profile.relation_spec.cache.allows_prefix_cache_sharing
                else None
            ),
            prefix_cache_v_cell_delta_max=(
                profile.relation_spec.cache.prefix_cache_v_cell_delta_max
                if profile.relation_spec.cache.allows_prefix_cache_sharing
                else None
            ),
            prefix_cache_v_row_sq_delta_max=(
                profile.relation_spec.cache.prefix_cache_v_row_sq_delta_max
                if profile.relation_spec.cache.allows_prefix_cache_sharing
                else None
            ),
        )
    except ProofV3VerificationError:
        raise
    except (ProofV3Error, AttributeError, TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            f"economic execution profile qualification failed: {exc}"
        ) from exc
    if expected.canonical_bytes() != profile.canonical_bytes():
        exact_mlp_legacy = replace(
            expected,
            verifier_key_digest=economic_verifier_digest_v3(
                lean=economic_profile_is_lean_v3(expected),
                compact_projection=economic_profile_is_compact_v3(expected),
                compact_full_row_escalation=(
                    economic_profile_has_full_row_escalation_v3(expected)
                ),
                selected_trace=economic_profile_uses_selected_trace_v3(
                    expected
                ),
                exact_mlp_activation=True,
                canonical_gdn_input=False,
            ),
        )
        pre_exact_legacy = replace(
            expected,
            verifier_key_digest=economic_verifier_digest_v3(
                lean=economic_profile_is_lean_v3(expected),
                compact_projection=economic_profile_is_compact_v3(expected),
                compact_full_row_escalation=(
                    economic_profile_has_full_row_escalation_v3(expected)
                ),
                selected_trace=economic_profile_uses_selected_trace_v3(
                    expected
                ),
                exact_mlp_activation=False,
            ),
        )
        if profile.canonical_bytes() not in {
            exact_mlp_legacy.canonical_bytes(),
            pre_exact_legacy.canonical_bytes(),
        }:
            raise ProofV3VerificationError(
                "economic execution profile does not match the authenticated "
                "artifacts"
            )
