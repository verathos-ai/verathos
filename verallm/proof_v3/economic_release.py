"""Load one authority-authenticated economic proof-v3 runtime release.

The projection manifest signature is checked against an authority set supplied
by the caller (normally resolved from ``ModelRegistry.owner()``).  The
artifact's informational ``authority`` field is never a trust input.

This loader authenticates the manifest-bound calibration and runtime-semantics
files and deterministically reconstructs the economic execution profile.  A
validator still authenticates the separately signed execution-profile
document before issuing a challenge session; this module does not replace
that qualification boundary.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from verallm.proof_v3.attention_runtime_semantics import (
    AttentionRuntimeSemanticsV3,
    load_attention_runtime_semantics_v3,
)
from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_profile import (
    build_economic_execution_profile_v3,
    infer_economic_manifest_layer_kinds_v3,
)
from verallm.proof_v3.economic_registry import QualifiedEconomicAdapterV3
from verallm.proof_v3.document import (
    load_signed_execution_profile_document_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.gdn_runtime_semantics import (
    GdnRuntimeSemanticsV3,
    load_gdn_runtime_semantics_v3,
)
from verallm.proof_v3.moe_runtime_semantics import (
    MoeRuntimeSemanticsV3,
    load_moe_runtime_semantics_v3,
)
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    EconomicLmHeadCatalogArtifactV3,
)
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.projection_manifest import ProjectionManifestV3
from verallm.proof_v3.scored_calibration_set import (
    ScoredCalibrationSetV3,
    load_scored_calibration_set_v3,
)
from verallm.proof_v3.session import QualifiedExecutionProfileV3

__all__ = [
    "EconomicProofV3RuntimeRelease",
    "QualifiedEconomicProofV3Release",
    "load_economic_proof_v3_runtime_release",
    "load_qualified_economic_proof_v3_release",
]


def _load_json_object(path, name: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofV3Error(f"{name} could not be loaded") from exc
    if not isinstance(value, dict):
        raise ProofV3Error(f"{name} must be a JSON object")
    return value


def _manifest_signatures(artifact: dict) -> tuple[bytes, ...]:
    has_single = "signature" in artifact
    has_plural = "signatures" in artifact
    if has_single == has_plural:
        raise ProofV3Error(
            "proof-v3 manifest artifact must contain exactly one signature "
            "field"
        )
    values = (
        (artifact["signature"],)
        if has_single
        else artifact["signatures"]
    )
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ProofV3Error("proof-v3 manifest signatures are malformed")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ProofV3Error(
                "proof-v3 manifest signature must be hexadecimal"
            )
        encoded = value[2:] if value.startswith("0x") else value
        if len(encoded) != 130 or encoded.lower() != encoded:
            raise ProofV3Error(
                "proof-v3 manifest signature must be canonical hexadecimal"
            )
        try:
            signature = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ProofV3Error(
                "proof-v3 manifest signature must be hexadecimal"
            ) from exc
        if len(signature) != 65:
            raise ProofV3Error(
                "proof-v3 manifest signature must be exactly 65 bytes"
            )
        result.append(signature)
    return tuple(result)


def _require_qualified_projection_corridor_v3(
    manifest: ProjectionManifestV3,
) -> None:
    """Reject lean hard-audit releases with unsigned fallback corridors."""

    missing = []
    if not manifest.corridor_sigma_bits:
        missing.append("corridor_sigma_bits")
    if not manifest.corridor_chi2_bits:
        missing.append("corridor_chi2_bits")
    if not manifest.corridor_qualification_digest:
        missing.append("corridor_qualification_digest")
    if missing:
        raise ProofV3VerificationError(
            "lean proof-v3 release lacks signed projection-corridor "
            f"qualification: {', '.join(missing)}"
        )


@dataclass(frozen=True, slots=True)
class EconomicProofV3RuntimeRelease:
    """Authenticated static runtime inputs for one exact qualified model."""

    manifest: ProjectionManifestV3
    calibration_set: ScoredCalibrationSetV3
    attention_runtime_semantics: AttentionRuntimeSemanticsV3
    gdn_runtime_semantics: GdnRuntimeSemanticsV3 | None
    lm_head_catalog: EconomicLmHeadCatalogArtifactV3 | None
    artifacts: EconomicVerifiedArtifactsV3
    profile: ExecutionSecurityProfileV3
    layer_kinds: tuple[str, ...]
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None


@dataclass(frozen=True, slots=True)
class QualifiedEconomicProofV3Release:
    """One runtime release plus its independently signed profile."""

    runtime: EconomicProofV3RuntimeRelease
    qualified_profile: QualifiedExecutionProfileV3


def load_economic_proof_v3_runtime_release(
    *,
    manifest_artifact_path,
    calibration_set_path,
    attention_runtime_semantics_path,
    gdn_runtime_semantics_path=None,
    moe_runtime_semantics_path=None,
    lm_head_catalog_path=None,
    expected_model_id: str,
    expected_authorities: Collection[str | bytes],
    authority_threshold: int,
    layer_kinds,
    tokenizer_binding_digest: bytes,
    runtime_encoding_id: str,
    max_decode_tokens: int = 4_096,
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
    verified_projection_manifest=None,
    weight_catalog=None,
    projection_catalog_validation_context=None,
) -> EconomicProofV3RuntimeRelease:
    """Authenticate and assemble the model's production runtime artifacts."""

    if not isinstance(lean, bool):
        raise ProofV3Error("proof-v3 lean release selector must be boolean")
    if not isinstance(compact_projection, bool):
        raise ProofV3Error(
            "proof-v3 compact projection selector must be boolean"
        )
    if not isinstance(compact_full_row_escalation, bool):
        raise ProofV3Error(
            "proof-v3 compact escalation selector must be boolean"
        )
    if not isinstance(selected_trace, bool):
        raise ProofV3Error(
            "proof-v3 selected-trace selector must be boolean"
        )
    if not isinstance(prefix_cache_sharing, bool):
        raise ProofV3Error(
            "proof-v3 prefix-cache selector must be boolean"
        )
    if compact_full_row_escalation and not compact_projection:
        raise ProofV3Error(
            "complete-row escalation requires compact projection"
        )
    if selected_trace and not compact_projection:
        raise ProofV3Error(
            "selected-trace release requires compact projection"
        )
    if compact_projection and not lean:
        raise ProofV3Error(
            "compact projection release requires lean artifacts"
        )
    has_projection_manifest = verified_projection_manifest is not None
    has_projection_catalog = weight_catalog is not None
    if has_projection_manifest != has_projection_catalog:
        raise ProofV3VerificationError(
            "projection manifest and catalog must be configured together"
        )
    if lean != has_projection_manifest:
        raise ProofV3VerificationError(
            "projection catalog availability does not match the signed profile"
        )

    artifact = _load_json_object(
        manifest_artifact_path,
        "proof-v3 manifest artifact",
    )
    if "manifest" not in artifact:
        raise ProofV3Error("proof-v3 manifest artifact lacks its manifest")
    try:
        manifest = ProjectionManifestV3.from_json(
            json.dumps(
                artifact["manifest"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except Exception as exc:
        if isinstance(exc, (ProofV3Error, ProofV3VerificationError)):
            raise
        raise ProofV3Error("proof-v3 manifest payload is malformed") from exc
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
        or manifest.model_id != expected_model_id
    ):
        raise ProofV3VerificationError(
            "proof-v3 manifest model does not match the selected model"
        )

    artifacts = EconomicVerifiedArtifactsV3.from_signed(
        manifest,
        signatures=_manifest_signatures(artifact),
        expected_authorities=expected_authorities,
        authority_threshold=authority_threshold,
    )
    if lean:
        _require_qualified_projection_corridor_v3(manifest)
    from verallm.proof_v3.projection_manifest import (
        LM_HEAD_CATALOG_BINDING_V3,
    )

    catalog_required = (
        manifest.lm_head_binding == LM_HEAD_CATALOG_BINDING_V3
    )
    if catalog_required != bool(lm_head_catalog_path):
        raise ProofV3VerificationError(
            "LM-head catalog availability does not match the signed manifest"
        )
    lm_head_catalog = (
        EconomicLmHeadCatalogArtifactV3.load(lm_head_catalog_path)
        if lm_head_catalog_path
        else None
    )
    if lm_head_catalog is not None:
        artifacts.authenticate_lm_head_catalog_v3(lm_head_catalog)
    calibration = load_scored_calibration_set_v3(str(calibration_set_path))
    if calibration.digest != manifest.attn_calibration_set_digest:
        raise ProofV3VerificationError(
            "attention calibration set does not match the signed manifest"
        )
    artifacts.attn_calibration_set = calibration

    attention = load_attention_runtime_semantics_v3(
        str(attention_runtime_semantics_path)
    )
    artifacts.authenticate_attention_runtime_semantics_v3(attention)

    kinds = (
        infer_economic_manifest_layer_kinds_v3(manifest)
        if layer_kinds is None
        else tuple(str(kind) for kind in layer_kinds)
    )
    has_gdn = any(kind == "gdn" for kind in kinds)
    if has_gdn != bool(gdn_runtime_semantics_path):
        raise ProofV3VerificationError(
            "GDN runtime semantics do not match the qualified layer inventory"
        )
    gdn = (
        load_gdn_runtime_semantics_v3(str(gdn_runtime_semantics_path))
        if gdn_runtime_semantics_path
        else None
    )
    if gdn is not None:
        artifacts.authenticate_gdn_runtime_semantics_v3(gdn)
    elif manifest.gdn_runtime_semantics_digest:
        raise ProofV3VerificationError(
            "signed manifest requires GDN runtime semantics"
        )

    has_moe = bool(manifest.moe_runtime_semantics_digest)
    if has_moe != bool(moe_runtime_semantics_path):
        raise ProofV3VerificationError(
            "MoE runtime semantics do not match the qualified manifest"
        )
    moe = (
        load_moe_runtime_semantics_v3(str(moe_runtime_semantics_path))
        if moe_runtime_semantics_path
        else None
    )
    if moe is not None:
        artifacts.authenticate_moe_runtime_semantics_v3(moe)

    artifacts.authenticate_tokenizer_binding_v3(tokenizer_binding_digest)
    profile = build_economic_execution_profile_v3(
        manifest=manifest,
        layer_kinds=kinds,
        calibration_set=calibration,
        attention_runtime_semantics=attention,
        gdn_runtime_semantics=gdn,
        moe_runtime_semantics=moe,
        tokenizer_binding_digest=tokenizer_binding_digest,
        runtime_encoding_id=runtime_encoding_id,
        max_decode_tokens=max_decode_tokens,
        lean=lean,
        compact_projection=compact_projection,
        compact_full_row_escalation=compact_full_row_escalation,
        selected_trace=selected_trace,
        prefix_cache_sharing=prefix_cache_sharing,
        prefix_cache_page_tokens=prefix_cache_page_tokens,
        prefix_cache_k_cell_delta_max=prefix_cache_k_cell_delta_max,
        prefix_cache_k_row_sq_delta_max=prefix_cache_k_row_sq_delta_max,
        prefix_cache_v_cell_delta_max=prefix_cache_v_cell_delta_max,
        prefix_cache_v_row_sq_delta_max=prefix_cache_v_row_sq_delta_max,
    )
    if lean:
        from verallm.proof_v3.catalog import VerifiedV2CatalogBindingV3

        projection_binding = (
            VerifiedV2CatalogBindingV3.from_verified_projection_manifest(
                verified_manifest=verified_projection_manifest,
                weight_catalog=weight_catalog,
                quantization_semantics_id=profile.quantization_semantics_id,
                static_bindings=profile.relation_spec.static_bindings,
                validation_context=(
                    projection_catalog_validation_context
                ),
            )
        )
        artifacts.authenticate_lean_projection_catalog_v3(
            projection_binding
        )
    QualifiedEconomicAdapterV3(artifacts).validate_profile(profile=profile)
    return EconomicProofV3RuntimeRelease(
        manifest=manifest,
        calibration_set=calibration,
        attention_runtime_semantics=attention,
        gdn_runtime_semantics=gdn,
        moe_runtime_semantics=moe,
        lm_head_catalog=lm_head_catalog,
        artifacts=artifacts,
        profile=profile,
        layer_kinds=kinds,
    )


def load_qualified_economic_proof_v3_release(
    *,
    signed_profile_path,
    manifest_artifact_path,
    calibration_set_path,
    attention_runtime_semantics_path,
    gdn_runtime_semantics_path=None,
    moe_runtime_semantics_path=None,
    lm_head_catalog_path=None,
    expected_model_id: str,
    expected_authorities: Collection[str | bytes],
    authority_threshold: int,
    layer_kinds,
    tokenizer_binding_digest: bytes,
    runtime_encoding_id: str,
    max_decode_tokens: int = 4_096,
    verified_projection_manifest=None,
    weight_catalog=None,
    projection_catalog_validation_context=None,
) -> QualifiedEconomicProofV3Release:
    """Load the complete validator admission artifact for one exact model.

    The manifest and profile signatures are independently checked against the
    validator-selected authority set. The signed profile must equal the
    deterministic profile reconstructed from the authenticated manifest,
    calibration and runtime semantics; there is no miner-selected fallback.
    """

    document = load_signed_execution_profile_document_v3(signed_profile_path)
    from verallm.proof_v3.economic_profile import (
        economic_profile_has_full_row_escalation_v3,
        economic_profile_is_compact_v3,
        economic_profile_is_lean_v3,
        economic_profile_uses_selected_trace_v3,
    )

    lean = economic_profile_is_lean_v3(document.profile)
    compact_projection = economic_profile_is_compact_v3(document.profile)
    compact_full_row_escalation = (
        economic_profile_has_full_row_escalation_v3(document.profile)
    )
    selected_trace = economic_profile_uses_selected_trace_v3(
        document.profile
    )
    prefix_cache_sharing = bool(
        document.profile.relation_spec.cache.allows_prefix_cache_sharing
    )
    runtime = load_economic_proof_v3_runtime_release(
        manifest_artifact_path=manifest_artifact_path,
        calibration_set_path=calibration_set_path,
        attention_runtime_semantics_path=attention_runtime_semantics_path,
        gdn_runtime_semantics_path=gdn_runtime_semantics_path,
        moe_runtime_semantics_path=moe_runtime_semantics_path,
        lm_head_catalog_path=lm_head_catalog_path,
        expected_model_id=expected_model_id,
        expected_authorities=expected_authorities,
        authority_threshold=authority_threshold,
        layer_kinds=layer_kinds,
        tokenizer_binding_digest=tokenizer_binding_digest,
        runtime_encoding_id=runtime_encoding_id,
        max_decode_tokens=max_decode_tokens,
        lean=lean,
        compact_projection=compact_projection,
        compact_full_row_escalation=compact_full_row_escalation,
        selected_trace=selected_trace,
        prefix_cache_sharing=prefix_cache_sharing,
        prefix_cache_page_tokens=(
            document.profile.relation_spec.cache.page_token_count
            if prefix_cache_sharing
            else None
        ),
        prefix_cache_k_cell_delta_max=(
            document.profile.relation_spec.cache.prefix_cache_k_cell_delta_max
            if prefix_cache_sharing
            else None
        ),
        prefix_cache_k_row_sq_delta_max=(
            document.profile.relation_spec.cache.prefix_cache_k_row_sq_delta_max
            if prefix_cache_sharing
            else None
        ),
        prefix_cache_v_cell_delta_max=(
            document.profile.relation_spec.cache.prefix_cache_v_cell_delta_max
            if prefix_cache_sharing
            else None
        ),
        prefix_cache_v_row_sq_delta_max=(
            document.profile.relation_spec.cache.prefix_cache_v_row_sq_delta_max
            if prefix_cache_sharing
            else None
        ),
        verified_projection_manifest=verified_projection_manifest,
        weight_catalog=weight_catalog,
        projection_catalog_validation_context=(
            projection_catalog_validation_context
        ),
    )
    registration = QualifiedEconomicAdapterV3(runtime.artifacts)
    if document.profile != runtime.profile:
        # Transition profiles differ only in a verifier-digest-gated relation
        # variant.  Validate the exact signed profile against the already
        # authenticated artifacts before selecting it as the runtime profile;
        # arbitrary or stale profiles still fail closed here.
        try:
            registration.validate_profile(profile=document.profile)
        except ProofV3VerificationError as exc:
            raise ProofV3VerificationError(
                "signed execution profile was not selected by the "
                "validator catalog"
            ) from exc
        runtime = replace(runtime, profile=document.profile)
    expected_profile_digest = runtime.profile.digest()
    qualified = QualifiedExecutionProfileV3.from_signed_document(
        document=document,
        expected_static_manifest_digest=runtime.manifest.digest(),
        expected_execution_profile_digest=expected_profile_digest,
        expected_authority_signers=expected_authorities,
        authority_threshold=authority_threshold,
        registration=registration,
    )
    if qualified.profile != runtime.profile:
        raise ProofV3VerificationError(
            "signed execution profile differs from the authenticated runtime "
            "release"
        )
    return QualifiedEconomicProofV3Release(
        runtime=runtime,
        qualified_profile=qualified,
    )
