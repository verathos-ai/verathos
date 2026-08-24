"""Qualified registration + fail-closed admission for economic_recompute_v3.

The economic proof system is the current HARD audit tier.  Its admission
REUSES the one Stack-A fail-closed preparation flow
(:func:`verifier.validate_execution_envelope_binding_v3`: signed-profile
authentication, envelope-vs-validator-owned prompt/output validation, nonce
commitment check) instead of duplicating it, then:

* validates the runtime hard-audit policy against the signed minimum;
* re-derives the postcommit HARD/LIGHT decision from the hidden nonce and
  REQUIRES ``hard_audit_selected == True`` -- economic_recompute_v3 is the
  hard tier, a light-tier request never dispatches an economic proof;
* binds proof, envelope, and profile digests;
* derives the complete validator challenge from the nonce-bound transcript;
* performs the exact-profile-digest registry lookup (fail-closed: no
  default adapter, wrong registration type rejected).

The strong ``global_folded_execution_v3`` path is untouched: its registry,
its ``require_hard_execution_capability`` gate, and its adapter protocol
stay exactly as they were.  An economic profile has a different profile
digest, so a validator that selected the strong digest can never dispatch
an economic proof (no downgrade), and vice versa.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass, field

from verallm.proof_v3 import ECONOMIC_SHELL_COVERAGE_MODE_V3
from verallm.proof_v3.challenge import (
    PostCommitAuditDecisionV3,
    derive_postcommit_audit_decision_v3,
)
from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_challenge import (
    ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    ECONOMIC_COMPACT_SELECTION_ABI_V3,
    ECONOMIC_SELECTION_ABI_V3,
    ECONOMIC_STREAMING_SELECTION_ABI_V3,
    EconomicChallengeV3,
    derive_economic_challenge_v3,
)
from verallm.proof_v3.economic_recompute_adapter import (
    ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
    verify_economic_recompute_v3,
)
from verallm.proof_v3.economic_wire import EconomicRecomputeProofV3
from verallm.proof_v3.errors import ProofV3UnavailableError, ProofV3VerificationError
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    ObservedExecutionOutputV3,
    ValidatorExecutionRequestContextV3,
)
from verallm.proof_v3.verifier import (
    AdapterKeyV3,
    adapter_key_v3,
    validate_execution_envelope_binding_v3,
)

__all__ = [
    "QualifiedEconomicAdapterV3",
    "require_economic_recompute_capability_v3",
    "verify_economic_execution_proof_v3",
]


def require_economic_recompute_capability_v3(
    profile: ExecutionSecurityProfileV3,
) -> None:
    """Reject any profile that is not an exact economic recompute statement.

    This is the economic sibling of ``require_hard_execution_capability`` --
    it never accepts a strong global-folded profile (that path keeps its own
    stricter gate), and it never accepts a profile signed for a different
    coverage mode or selection ABI.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("profile has an unexpected type")
    if profile.proof_system_id != ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
        raise ProofV3VerificationError(
            "execution profile is not an economic recompute profile"
        )
    policy = profile.relation_spec.audit_policy
    if policy.coverage_mode != ECONOMIC_SHELL_COVERAGE_MODE_V3:
        raise ProofV3VerificationError(
            "economic profile has an unexpected coverage mode"
        )
    if policy.selection_abi_id not in (
        ECONOMIC_SELECTION_ABI_V3,
        ECONOMIC_STREAMING_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    ):
        raise ProofV3VerificationError(
            "economic profile is not signed for the economic selection ABI"
        )


@dataclass(frozen=True, slots=True)
class QualifiedEconomicAdapterV3:
    """One economic verifier registered for one exact signed profile digest.

    ``artifacts`` is the validator-owned, authority-signature-verified
    projection/embedding/LM-head manifest view -- the only trusted weight
    source of the economic audit.
    """

    artifacts: EconomicVerifiedArtifactsV3
    _validated_profile_digest: bytes = field(
        default=b"",
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, EconomicVerifiedArtifactsV3):
            raise ProofV3VerificationError(
                "economic adapter artifacts are not a verified manifest view"
            )

    def validate_profile(self, *, profile: ExecutionSecurityProfileV3) -> None:
        """Bind this verified weight manifest to one exact signed profile."""

        require_economic_recompute_capability_v3(profile)
        from verallm.proof_v3.economic_profile import (
            ECONOMIC_PROFILE_ADAPTER_ID_V3,
            validate_economic_execution_profile_v3,
        )

        if profile.adapter_id != ECONOMIC_PROFILE_ADAPTER_ID_V3:
            raise ProofV3VerificationError(
                "economic profile adapter id is not qualified"
            )
        profile_digest = profile.digest()
        if self._validated_profile_digest:
            if hmac.compare_digest(
                self._validated_profile_digest,
                profile_digest,
            ):
                return
            raise ProofV3VerificationError(
                "economic adapter is already bound to another profile"
            )
        validate_economic_execution_profile_v3(
            profile=profile,
            manifest=self.artifacts.manifest,
            calibration_set=self.artifacts.attn_calibration_set,
            attention_runtime_semantics=(
                self.artifacts.attention_runtime_semantics
            ),
            gdn_runtime_semantics=self.artifacts.gdn_runtime_semantics,
            moe_runtime_semantics=self.artifacts.moe_runtime_semantics,
            tokenizer_binding_digest=(
                self.artifacts.tokenizer_binding_digest
            ),
        )
        object.__setattr__(
            self,
            "_validated_profile_digest",
            profile_digest,
        )

    def maximum_hard_audit_decode_tokens(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
    ) -> int:
        """Return the authenticated decode limit for a possible hard audit."""

        require_economic_recompute_capability_v3(profile)
        semantics = self.artifacts.gdn_runtime_semantics
        if semantics is None:
            return profile.max_verified_decode_tokens
        if semantics.decode_checkpoint_stride:
            # Checkpointed semantics qualify one bounded replay window, not
            # the complete suffix from the prompt boundary. The signed
            # empirical qualification still limits which absolute decode
            # positions may be selected for a hard audit.
            return min(
                profile.max_verified_decode_tokens,
                semantics.max_hard_audit_decode_tokens,
            )
        row_limits = tuple(
            layer.max_decode_replay_rows for layer in semantics.layers
        )
        if not row_limits:
            raise ProofV3VerificationError(
                "qualified GDN semantics have no replay-row bounds"
            )
        # A decode of N output tokens forwards N-1 rows after the committed
        # prompt boundary. The hard verifier replays that exact suffix.
        return min(
            profile.max_verified_decode_tokens,
            min(row_limits) + 1,
        )

    def validate_envelope(
        self,
        *,
        envelope: ProofV3CommitmentEnvelope,
        proof: EconomicRecomputeProofV3,
    ) -> None:
        """Require the canonical economic meaning of generic envelope roots."""

        from verallm.proof_v3.economic_envelope import (
            validate_economic_commitment_envelope_roots_v3,
        )

        validate_economic_commitment_envelope_roots_v3(
            envelope=envelope,
            capture_chain_digest=proof.capture_chain_digest,
        )


def verify_economic_execution_proof_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    proof: EconomicRecomputeProofV3,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    validator_request_context: ValidatorExecutionRequestContextV3,
    observed_output: ObservedExecutionOutputV3,
    runtime_policy: RuntimeHardAuditPolicyV3,
    adapters: Mapping[AdapterKeyV3, QualifiedEconomicAdapterV3] | None = None,
    audit_decision: PostCommitAuditDecisionV3 | None = None,
) -> EconomicChallengeV3:
    """Admit + verify one economic wire proof through the shared flow."""

    if not isinstance(proof, EconomicRecomputeProofV3):
        raise ProofV3VerificationError(
            "economic execution proof has an unexpected type"
        )
    require_economic_recompute_capability_v3(profile)

    # -- shared Stack-A admission: profile/envelope/nonce/prompt/output -----
    binding = validate_execution_envelope_binding_v3(
        profile=profile,
        envelope=envelope,
        validator_nonce=validator_nonce,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_execution_profile_digest=expected_execution_profile_digest,
        validator_request_context=validator_request_context,
        observed_output=observed_output,
        capability_requirement=require_economic_recompute_capability_v3,
    )

    # -- runtime policy + postcommit HARD/LIGHT decision --------------------
    try:
        profile.relation_spec.audit_policy.validate_runtime(runtime_policy)
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "runtime hard-audit policy is malformed"
        ) from exc
    try:
        expected_decision = derive_postcommit_audit_decision_v3(
            validator_nonce=validator_nonce,
            profile=profile,
            envelope=envelope,
            runtime_policy=runtime_policy,
        )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "postcommit hard-audit decision is malformed"
        ) from exc
    if audit_decision is not None:
        if not isinstance(audit_decision, PostCommitAuditDecisionV3):
            raise ProofV3VerificationError(
                "postcommit hard-audit decision has an unexpected type"
            )
        if audit_decision != expected_decision:
            raise ProofV3VerificationError(
                "postcommit hard-audit decision is not bound to the validator "
                "transcript"
            )
    # economic_recompute_v3 IS the hard tier: a light-tier draw never
    # dispatches an economic proof.
    if not expected_decision.hard_audit_selected:
        raise ProofV3VerificationError(
            "postcommit audit decision did not select a hard proof"
        )

    # -- digest binding: proof <-> envelope <-> profile ---------------------
    if proof.commitment_envelope_digest != envelope.digest():
        raise ProofV3VerificationError(
            "proof is bound to a different commitment envelope"
        )
    if proof.execution_profile_digest != binding.profile_digest:
        raise ProofV3VerificationError(
            "proof is bound to a different execution profile"
        )

    # -- validator-derived challenge from the nonce-bound transcript --------
    challenge = derive_economic_challenge_v3(
        transcript_digest=expected_decision.transcript_digest,
        profile=profile,
        envelope=envelope,
    )

    # -- exact-profile registry lookup (fail-closed) ------------------------
    registration = (adapters or {}).get(adapter_key_v3(profile))
    if registration is None:
        raise ProofV3UnavailableError(
            "no qualified economic adapter is registered for this execution "
            "profile"
        )
    if not isinstance(registration, QualifiedEconomicAdapterV3):
        raise ProofV3VerificationError(
            "economic adapter registration is not a qualified economic adapter"
        )
    registration.validate_profile(profile=profile)
    registration.validate_envelope(envelope=envelope, proof=proof)

    verify_economic_recompute_v3(
        proof_system_id=profile.proof_system_id,
        proof=proof,
        profile=profile,
        envelope=envelope,
        challenge=challenge,
        artifacts=registration.artifacts,
        precommit_context=binding.precommit_context,
        prompt_token_ids=validator_request_context.prompt_token_ids,
        observed_output_token_ids=observed_output.output_token_ids,
        observed_text_utf8=observed_output.emitted_text_utf8,
        finish_reason=observed_output.finish_reason,
    )
    return challenge
