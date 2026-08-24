"""Canonical retained proof-v3 hard bundle and deterministic replay.

The hard-audit validator observes the request and output directly, then
verifies the post-nonce proof.  Other validators need the same public inputs
to repeat that verification without issuing another nonce or trusting the
miner's verdict.  This module carries exactly those inputs in one bounded,
canonical frame.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

from verallm.proof_v3.challenge import derive_postcommit_audit_decision_v3
from verallm.proof_v3.economic_recompute_adapter import (
    ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
)
from verallm.proof_v3.economic_registry import (
    require_economic_recompute_capability_v3,
    verify_economic_execution_proof_v3,
)
from verallm.proof_v3.economic_transport import (
    MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
    MAX_ECONOMIC_TRANSPORT_BYTES,
    decode_economic_proof_transport_v3,
    economic_proof_transport_maximum_bytes_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.payload import (
    ProofV3CommitmentEnvelope,
    commitment_envelope_from_bytes,
)
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    MAX_FINISH_REASON_BYTES_V3,
    MAX_OUTPUT_STREAM_BYTES_V3,
    ObservedExecutionOutputV3,
    PreExecutionRequestContextV3,
    ValidatorExecutionRequestContextV3,
    commit_validator_nonce_v3,
)
from verallm.proof_v3.session import (
    MAX_NONCE_REVEAL_BYTES_V3,
    NonceRevealV3,
    QualifiedExecutionProfileV3,
)
from verallm.proof_v3.verifier import (
    validate_execution_envelope_against_precommit_v3,
)

HARD_BUNDLE_FORMAT_VERSION_V3 = 1
MAX_DENSE_HARD_BUNDLE_BYTES_V3 = 64 << 20
MAX_HARD_BUNDLE_BYTES_V3 = 128 << 20
MAX_HARD_BUNDLE_TOKEN_IDS_V3 = 1 << 20
HARD_BUNDLE_MEDIA_TYPE_V3 = (
    "application/vnd.verathos.proof-v3-hard-bundle+octet-stream"
)

_MAGIC = b"V3HB"
_FLAGS_DENSE = 0
_FLAGS_SPARSE_MOE = 1
_HEADER = struct.Struct("<4sHHIIIIIIII")
_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/RETAINED_HARD_BUNDLE/V1/SHA256"

__all__ = [
    "HARD_BUNDLE_FORMAT_VERSION_V3",
    "HARD_BUNDLE_MEDIA_TYPE_V3",
    "MAX_DENSE_HARD_BUNDLE_BYTES_V3",
    "MAX_HARD_BUNDLE_BYTES_V3",
    "RetainedHardProofBundleV3",
    "runtime_policy_from_nonce_reveal_v3",
    "verify_retained_hard_proof_bundle_v3",
]


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3VerificationError(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3VerificationError(f"{name} must not be the zero digest")
    return value


def _retained_proof_transport_maximum_v3(encoded: bytes) -> int:
    """Select a bundle lane while preserving bounded failed-proof capture."""

    try:
        return economic_proof_transport_maximum_bytes_v3(
            encoded,
            allow_sparse_moe=True,
        )
    except ProofV3Error:
        # Validators intentionally retain malformed rejected responses for
        # deterministic replay. Preserve that diagnostic behavior only inside
        # the legacy dense byte ceiling; the larger sparse lane always needs a
        # valid, explicit transport header.
        if isinstance(encoded, bytes) and 0 < len(encoded) <= (
            MAX_ECONOMIC_TRANSPORT_BYTES
        ):
            return MAX_ECONOMIC_TRANSPORT_BYTES
        raise


def _token_ids(values, name: str, *, positive: bool) -> tuple[int, ...]:
    if isinstance(values, (bytes, str)) or not hasattr(values, "__len__"):
        raise ProofV3VerificationError(f"{name} must be a token-id sequence")
    count = len(values)
    if count > MAX_HARD_BUNDLE_TOKEN_IDS_V3 or (positive and count == 0):
        raise ProofV3VerificationError(f"{name} count is out of range")
    result: list[int] = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 1 << 32
        ):
            raise ProofV3VerificationError(
                f"{name}[{index}] is not a uint32 token id"
            )
        result.append(value)
    return tuple(result)


def runtime_policy_from_nonce_reveal_v3(
    *,
    profile,
    reveal: NonceRevealV3,
) -> RuntimeHardAuditPolicyV3:
    """Reconstruct the signed-compatible runtime policy from a hard reveal."""

    if not isinstance(reveal, NonceRevealV3):
        raise ProofV3VerificationError(
            "retained hard bundle nonce reveal has an unexpected type"
        )
    signed = profile.relation_spec.audit_policy
    decision = reveal.audit_decision
    organic_bps = signed.minimum_organic_hard_bps
    canary_bps = signed.minimum_canary_hard_bps
    if decision.request_kind == "organic":
        organic_bps = decision.effective_hard_bps
    elif decision.request_kind == "canary":
        canary_bps = decision.effective_hard_bps
    else:
        raise ProofV3VerificationError(
            "retained hard bundle request kind is unsupported"
        )
    policy = RuntimeHardAuditPolicyV3(
        request_kind=decision.request_kind,
        effective_organic_hard_bps=organic_bps,
        effective_canary_hard_bps=canary_bps,
        effective_probation_failures=signed.probation_failures,
        nonce_selection_abi_id=signed.nonce_selection_abi_id,
        tier_selection_abi_id=signed.tier_selection_abi_id,
        selection_abi_id=signed.selection_abi_id,
    )
    signed.validate_runtime(policy)
    return policy


@dataclass(frozen=True, slots=True)
class RetainedHardProofBundleV3:
    """Public inputs and proof needed to replay one completed hard audit."""

    precommit_context: PreExecutionRequestContextV3
    prompt_token_ids: tuple[int, ...]
    observed_output: ObservedExecutionOutputV3
    envelope: ProofV3CommitmentEnvelope
    nonce_reveal: NonceRevealV3
    encoded_proof: bytes

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommit_context,
            PreExecutionRequestContextV3,
        ):
            raise ProofV3VerificationError(
                "retained hard bundle precommit context has an unexpected type"
            )
        prompt = _token_ids(
            self.prompt_token_ids,
            "prompt_token_ids",
            positive=True,
        )
        if (
            not isinstance(self.prompt_token_ids, tuple)
            or prompt != self.prompt_token_ids
        ):
            object.__setattr__(self, "prompt_token_ids", prompt)
        if not isinstance(self.observed_output, ObservedExecutionOutputV3):
            raise ProofV3VerificationError(
                "retained hard bundle output has an unexpected type"
            )
        output_ids = _token_ids(
            self.observed_output.output_token_ids,
            "output_token_ids",
            positive=True,
        )
        text = self.observed_output.emitted_text_utf8
        if not isinstance(text, bytes) or len(text) > MAX_OUTPUT_STREAM_BYTES_V3:
            raise ProofV3VerificationError(
                "retained hard bundle output text is out of range"
            )
        finish = self.observed_output.finish_reason
        if not isinstance(finish, str):
            raise ProofV3VerificationError(
                "retained hard bundle finish reason is malformed"
            )
        try:
            finish_bytes = finish.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProofV3VerificationError(
                "retained hard bundle finish reason is malformed"
            ) from exc
        if not 0 < len(finish_bytes) <= MAX_FINISH_REASON_BYTES_V3:
            raise ProofV3VerificationError(
                "retained hard bundle finish reason is out of range"
            )
        if (
            not isinstance(self.observed_output.output_token_ids, tuple)
            or output_ids != tuple(self.observed_output.output_token_ids)
        ):
            object.__setattr__(
                self,
                "observed_output",
                ObservedExecutionOutputV3(
                    output_token_ids=output_ids,
                    emitted_text_utf8=text,
                    finish_reason=finish,
                ),
            )
        if not isinstance(self.envelope, ProofV3CommitmentEnvelope):
            raise ProofV3VerificationError(
                "retained hard bundle envelope has an unexpected type"
            )
        if not isinstance(self.nonce_reveal, NonceRevealV3):
            raise ProofV3VerificationError(
                "retained hard bundle reveal has an unexpected type"
            )
        try:
            proof_maximum = _retained_proof_transport_maximum_v3(
                self.encoded_proof
            )
        except ProofV3Error as exc:
            raise ProofV3VerificationError(
                "retained hard bundle proof transport is malformed"
            ) from exc
        if not self.encoded_proof or len(self.encoded_proof) > proof_maximum:
            raise ProofV3VerificationError(
                "retained hard bundle proof bytes are out of range"
            )
        if self.precommit_context.context_token_count != len(prompt):
            raise ProofV3VerificationError(
                "retained hard bundle prompt count does not match precommit"
            )
        if self.envelope.digest() != self.nonce_reveal.commitment_envelope_digest:
            raise ProofV3VerificationError(
                "retained hard bundle reveal does not match the envelope"
            )
        if (
            self.precommit_context.proof_challenge_id
            != self.nonce_reveal.proof_challenge_id
            or self.precommit_context.digest()
            != self.nonce_reveal.precommit_context_digest
            or self.precommit_context.execution_profile_digest
            != self.nonce_reveal.execution_profile_digest
            or self.precommit_context.cache_lease_digest
            != self.nonce_reveal.cache_lease_digest
        ):
            raise ProofV3VerificationError(
                "retained hard bundle public contexts do not match"
            )
        if not self.nonce_reveal.audit_decision.hard_audit_selected:
            raise ProofV3VerificationError(
                "retained hard bundle does not carry a hard decision"
            )
        output_binding = self.observed_output.derive_output_binding()
        if (
            self.envelope.output_token_root != output_binding.output_token_root
            or self.envelope.output_stream_digest
            != output_binding.output_stream_digest
            or self.envelope.decode_token_count != output_binding.decode_token_count
        ):
            raise ProofV3VerificationError(
                "retained hard bundle output does not match the envelope"
            )

    @property
    def commitment_envelope_digest(self) -> bytes:
        return self.envelope.digest()

    @property
    def proof_challenge_id(self) -> bytes:
        return self.precommit_context.proof_challenge_id

    def validator_request_context(self) -> ValidatorExecutionRequestContextV3:
        return ValidatorExecutionRequestContextV3(
            proof_challenge_id=self.precommit_context.proof_challenge_id,
            validator_identity_digest=(
                self.precommit_context.validator_identity_digest
            ),
            miner_identity_digest=self.precommit_context.miner_identity_digest,
            validator_nonce_commitment=(
                self.precommit_context.validator_nonce_commitment
            ),
            prompt_token_ids=self.prompt_token_ids,
            cache_lease_digest=self.precommit_context.cache_lease_digest,
            sampler_config_digest=self.precommit_context.sampler_config_digest,
        )

    def canonical_bytes(self) -> bytes:
        proof_maximum = _retained_proof_transport_maximum_v3(
            self.encoded_proof
        )
        sparse_moe = proof_maximum == MAX_ECONOMIC_MOE_TRANSPORT_BYTES
        flags = _FLAGS_SPARSE_MOE if sparse_moe else _FLAGS_DENSE
        bundle_maximum = (
            MAX_HARD_BUNDLE_BYTES_V3
            if sparse_moe
            else MAX_DENSE_HARD_BUNDLE_BYTES_V3
        )
        precommit = self.precommit_context.canonical_bytes()
        envelope = self.envelope.canonical_bytes()
        reveal = self.nonce_reveal.canonical_bytes()
        text = self.observed_output.emitted_text_utf8
        finish = self.observed_output.finish_reason.encode("utf-8")
        prompt = struct.pack(
            f"<{len(self.prompt_token_ids)}I",
            *self.prompt_token_ids,
        )
        output_ids = tuple(self.observed_output.output_token_ids)
        output = struct.pack(f"<{len(output_ids)}I", *output_ids)
        encoded = b"".join(
            (
                _HEADER.pack(
                    _MAGIC,
                    HARD_BUNDLE_FORMAT_VERSION_V3,
                    flags,
                    len(precommit),
                    len(envelope),
                    len(reveal),
                    len(self.prompt_token_ids),
                    len(output_ids),
                    len(text),
                    len(finish),
                    len(self.encoded_proof),
                ),
                precommit,
                envelope,
                reveal,
                prompt,
                output,
                text,
                finish,
                self.encoded_proof,
            )
        )
        if len(encoded) > bundle_maximum:
            raise ProofV3VerificationError(
                "retained hard bundle exceeds its byte limit"
            )
        return encoded

    def digest(self) -> bytes:
        return hashlib.sha256(_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "RetainedHardProofBundleV3":
        if (
            not isinstance(encoded, bytes)
            or len(encoded) < _HEADER.size
            or len(encoded) > MAX_HARD_BUNDLE_BYTES_V3
        ):
            raise ProofV3VerificationError(
                "retained hard bundle byte length is out of range"
            )
        try:
            (
                magic,
                version,
                flags,
                precommit_len,
                envelope_len,
                reveal_len,
                prompt_count,
                output_count,
                text_len,
                finish_len,
                proof_len,
            ) = _HEADER.unpack_from(encoded)
        except struct.error as exc:
            raise ProofV3VerificationError(
                "retained hard bundle header is malformed"
            ) from exc
        if (
            magic != _MAGIC
            or version != HARD_BUNDLE_FORMAT_VERSION_V3
            or flags not in {_FLAGS_DENSE, _FLAGS_SPARSE_MOE}
        ):
            raise ProofV3VerificationError(
                "retained hard bundle header is unsupported"
            )
        sparse_moe = flags == _FLAGS_SPARSE_MOE
        bundle_maximum = (
            MAX_HARD_BUNDLE_BYTES_V3
            if sparse_moe
            else MAX_DENSE_HARD_BUNDLE_BYTES_V3
        )
        proof_maximum = (
            MAX_ECONOMIC_MOE_TRANSPORT_BYTES
            if sparse_moe
            else MAX_ECONOMIC_TRANSPORT_BYTES
        )
        if len(encoded) > bundle_maximum:
            raise ProofV3VerificationError(
                "retained hard bundle byte length is out of range"
            )
        if (
            prompt_count == 0
            or output_count == 0
            or prompt_count > MAX_HARD_BUNDLE_TOKEN_IDS_V3
            or output_count > MAX_HARD_BUNDLE_TOKEN_IDS_V3
            or text_len > MAX_OUTPUT_STREAM_BYTES_V3
            or not 0 < finish_len <= MAX_FINISH_REASON_BYTES_V3
            or not 0 < proof_len <= proof_maximum
            or not 0 < reveal_len <= MAX_NONCE_REVEAL_BYTES_V3
        ):
            raise ProofV3VerificationError(
                "retained hard bundle declared lengths are out of range"
            )
        total = (
            _HEADER.size
            + precommit_len
            + envelope_len
            + reveal_len
            + 4 * prompt_count
            + 4 * output_count
            + text_len
            + finish_len
            + proof_len
        )
        if total != len(encoded):
            raise ProofV3VerificationError(
                "retained hard bundle length is inconsistent"
            )
        offset = _HEADER.size

        def take(size: int) -> bytes:
            nonlocal offset
            value = encoded[offset : offset + size]
            offset += size
            return value

        precommit = PreExecutionRequestContextV3.from_canonical_bytes(
            take(precommit_len)
        )
        envelope = commitment_envelope_from_bytes(take(envelope_len))
        reveal = NonceRevealV3.from_canonical_bytes(take(reveal_len))
        prompt = struct.unpack(f"<{prompt_count}I", take(4 * prompt_count))
        output = struct.unpack(f"<{output_count}I", take(4 * output_count))
        text = take(text_len)
        try:
            finish = take(finish_len).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProofV3VerificationError(
                "retained hard bundle finish reason is not UTF-8"
            ) from exc
        proof = take(proof_len)
        try:
            observed_proof_maximum = _retained_proof_transport_maximum_v3(
                proof
            )
        except ProofV3Error as exc:
            raise ProofV3VerificationError(
                "retained hard bundle proof transport is malformed"
            ) from exc
        if observed_proof_maximum != proof_maximum:
            raise ProofV3VerificationError(
                "retained hard bundle sparse-MoE flag is inconsistent"
            )
        if offset != len(encoded):
            raise ProofV3VerificationError(
                "retained hard bundle has trailing bytes"
            )
        result = cls(
            precommit_context=precommit,
            prompt_token_ids=tuple(prompt),
            observed_output=ObservedExecutionOutputV3(
                output_token_ids=tuple(output),
                emitted_text_utf8=text,
                finish_reason=finish,
            ),
            envelope=envelope,
            nonce_reveal=reveal,
            encoded_proof=proof,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3VerificationError(
                "retained hard bundle is not canonical"
            )
        return result


def verify_retained_hard_proof_bundle_v3(
    *,
    bundle: RetainedHardProofBundleV3,
    qualified_profile: QualifiedExecutionProfileV3,
    expected_validator_identity_digest: bytes,
    expected_miner_identity_digest: bytes,
    expected_commitment_envelope_digest: bytes | None = None,
    expected_capture_chain_digest: bytes | None = None,
):
    """Deterministically replay one retained hard proof from public inputs."""

    if not isinstance(bundle, RetainedHardProofBundleV3):
        raise ProofV3VerificationError(
            "retained hard bundle has an unexpected type"
        )
    if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
        raise ProofV3VerificationError(
            "retained hard bundle qualified profile has an unexpected type"
        )
    qualified_profile.require_qualification_provenance()
    validator_identity = _fixed32(
        expected_validator_identity_digest,
        "expected_validator_identity_digest",
        nonzero=True,
    )
    miner_identity = _fixed32(
        expected_miner_identity_digest,
        "expected_miner_identity_digest",
        nonzero=True,
    )
    if not hmac.compare_digest(
        bundle.precommit_context.validator_identity_digest,
        validator_identity,
    ):
        raise ProofV3VerificationError(
            "retained hard bundle belongs to another validator"
        )
    if not hmac.compare_digest(
        bundle.precommit_context.miner_identity_digest,
        miner_identity,
    ):
        raise ProofV3VerificationError(
            "retained hard bundle belongs to another miner"
        )
    if expected_commitment_envelope_digest is not None and not hmac.compare_digest(
        bundle.commitment_envelope_digest,
        _fixed32(
            expected_commitment_envelope_digest,
            "expected_commitment_envelope_digest",
        ),
    ):
        raise ProofV3VerificationError(
            "retained hard bundle does not match the signed receipt"
        )

    profile = qualified_profile.profile
    if profile.proof_system_id != ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
        raise ProofV3VerificationError(
            "retained hard bundle requires economic_recompute_v3"
        )
    if (
        bundle.precommit_context.static_manifest_digest
        != qualified_profile.expected_static_manifest_digest
        or bundle.precommit_context.execution_profile_digest
        != qualified_profile.expected_execution_profile_digest
        or profile.digest()
        != qualified_profile.expected_execution_profile_digest
    ):
        raise ProofV3VerificationError(
            "retained hard bundle profile binding is stale"
        )
    validator_context = bundle.validator_request_context()
    reconstructed = validator_context.derive_precommit_context(
        static_manifest_digest=qualified_profile.expected_static_manifest_digest,
        execution_profile_digest=(
            qualified_profile.expected_execution_profile_digest
        ),
    )
    if reconstructed.canonical_bytes() != bundle.precommit_context.canonical_bytes():
        raise ProofV3VerificationError(
            "retained hard bundle prompt does not match its precommit"
        )
    expected_nonce_commitment = commit_validator_nonce_v3(
        validator_nonce=bundle.nonce_reveal.validator_nonce,
        nonce_context_digest=bundle.precommit_context.nonce_context_digest(),
    )
    if not hmac.compare_digest(
        expected_nonce_commitment,
        bundle.precommit_context.validator_nonce_commitment,
    ):
        raise ProofV3VerificationError(
            "retained hard bundle nonce does not match its commitment"
        )
    runtime_policy = runtime_policy_from_nonce_reveal_v3(
        profile=profile,
        reveal=bundle.nonce_reveal,
    )
    decision = derive_postcommit_audit_decision_v3(
        validator_nonce=bundle.nonce_reveal.validator_nonce,
        profile=profile,
        envelope=bundle.envelope,
        runtime_policy=runtime_policy,
    )
    if not hmac.compare_digest(
        decision.canonical_bytes(),
        bundle.nonce_reveal.audit_decision.canonical_bytes(),
    ):
        raise ProofV3VerificationError(
            "retained hard bundle audit decision is inconsistent"
        )
    validate_execution_envelope_against_precommit_v3(
        profile=profile,
        envelope=bundle.envelope,
        expected_static_manifest_digest=(
            qualified_profile.expected_static_manifest_digest
        ),
        expected_execution_profile_digest=(
            qualified_profile.expected_execution_profile_digest
        ),
        precommit_context=bundle.precommit_context,
        output_binding=bundle.observed_output.derive_output_binding(),
        capability_requirement=require_economic_recompute_capability_v3,
    )
    try:
        proof = decode_economic_proof_transport_v3(
            bundle.encoded_proof,
            allow_sparse_moe=bool(
                getattr(
                    getattr(
                        qualified_profile.registration,
                        "artifacts",
                        None,
                    ),
                    "moe_runtime_semantics",
                    None,
                )
            ),
        )
    except ProofV3Error:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "retained hard bundle proof transport is malformed"
        ) from exc
    if (
        expected_capture_chain_digest is not None
        and not hmac.compare_digest(
            proof.capture_chain_digest,
            _fixed32(
                expected_capture_chain_digest,
                "expected_capture_chain_digest",
                nonzero=True,
            ),
        )
    ):
        raise ProofV3VerificationError(
            "retained hard bundle does not match the receipt capture chain"
        )
    return verify_economic_execution_proof_v3(
        profile=profile,
        envelope=bundle.envelope,
        proof=proof,
        validator_nonce=bundle.nonce_reveal.validator_nonce,
        expected_static_manifest_digest=(
            qualified_profile.expected_static_manifest_digest
        ),
        expected_execution_profile_digest=(
            qualified_profile.expected_execution_profile_digest
        ),
        validator_request_context=validator_context,
        observed_output=bundle.observed_output,
        runtime_policy=runtime_policy,
        adapters={
            qualified_profile.expected_execution_profile_digest: (
                qualified_profile.registration
            ),
        },
        audit_decision=decision,
    )
