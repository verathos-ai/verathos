"""Atomic validator lifecycle for one post-commitment proof-v3 nonce.

The nonce is generated before serving but is never exposed through the public
request context.  A validator accepts exactly one canonical commitment envelope
after the last visible token, then reveals the nonce and a derived hard/light
decision exactly once.  A hard decision accepts exactly one final proof bound
to that envelope; the currently unregistered light state does not claim a
verified receipt.  This is local state: a multi-worker integration must keep a
request sticky to one worker or provide an equivalent shared compare-and-swap
record.
"""

from __future__ import annotations

import secrets
import struct
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from enum import Enum

from verallm.proof_v3.accumulator import PostCommitOpeningTicketV3
from verallm.proof_v3.challenge import (
    FoldedExecutionChallengeV3,
    PostCommitAuditDecisionV3,
    _audit_tier_draw_from_precommit_context_v3,
    derive_postcommit_audit_decision_v3,
)
from verallm.proof_v3.document import (
    SignedExecutionProfileDocumentV3,
    verify_signed_execution_profile_v3,
)
from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.payload import (
    FoldedExecutionProofV3,
    ProofV3CommitmentEnvelope,
    commitment_envelope_from_bytes,
    folded_execution_proof_from_bytes,
)
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    ObservedExecutionOutputV3,
    PreExecutionRequestContextV3,
    ValidatorExecutionRequestContextV3,
    commit_validator_nonce_v3,
    derive_pre_nonce_context_digest_v3,
)
from verallm.proof_v3.verifier import (
    EnvelopeBindingV3,
    QualifiedExecutionAdapterV3,
    _verify_folded_execution_proof_against_binding_v3,
    validate_execution_envelope_against_precommit_v3,
)


DEFAULT_PROOF_ARRIVAL_BUDGET_NS_V3 = 1_000_000_000
DEFAULT_PRECOMMIT_ARRIVAL_BUDGET_NS_V3 = 5_000_000_000
DEFAULT_LOCAL_NONCE_REVEAL_BUDGET_NS_V3 = 10_000_000_000
DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3 = 360_000_000_000
LEGACY_MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3 = 540_000_000_000
MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3 = 1_800_000_000_000
HARD_PROOF_EXTENDED_DECODE_START_TOKENS_V3 = 4096
HARD_PROOF_EXTENDED_DECODE_LIMIT_TOKENS_V3 = 8192
MAX_NONCE_REVEAL_HOLD_BUDGET_NS_V3 = 930_000_000_000
_MAX_PAIRED_TIER_NONCE_ATTEMPTS_V3 = 4096
NONCE_REVEAL_FORMAT_VERSION_V3 = 1
MAX_NONCE_REVEAL_BYTES_V3 = 512
_NONCE_REVEAL_MAGIC_V3 = b"V3NR"
_QUALIFIED_PROFILE_FACTORY_TOKEN = object()
_CHALLENGE_SESSION_FACTORY_TOKEN = object()
_ISSUE_WITH_ENTROPY_FACTORY_TOKEN = object()


class ChallengeSessionStateV3(str, Enum):
    """The only valid transitions for one eligible V3 proof request."""

    AWAITING_PRECOMMIT = "awaiting_precommit"
    PRECOMMIT_ACCEPTED = "precommit_accepted"
    HARD_SELECTED = "hard_selected"
    NONCE_REVEALED = "nonce_revealed"
    LIGHT_REVEALED = "light_revealed"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    ABORTED = "aborted"


_TERMINAL_STATES = frozenset(
    {
        # A light draw deliberately records no V3 proof success. It is still
        # terminal so it cannot later turn into an expiry/proof failure after
        # the validator has told the miner no hard opening is due.
        ChallengeSessionStateV3.LIGHT_REVEALED,
        ChallengeSessionStateV3.VERIFIED,
        ChallengeSessionStateV3.FAILED,
        ChallengeSessionStateV3.EXPIRED,
        ChallengeSessionStateV3.ABORTED,
    }
)


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3VerificationError(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3VerificationError(f"{name} must not be the zero digest")
    return value


def _monotonic_ns(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProofV3VerificationError(f"{name} must be a monotonic nanosecond value")
    return value


def _arrival_budget_ns(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= DEFAULT_PRECOMMIT_ARRIVAL_BUDGET_NS_V3
    ):
        raise ProofV3VerificationError(
            "proof_arrival_budget_ns must be between one nanosecond and five seconds"
        )
    return value


def _hard_arrival_budget_ns(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3
    ):
        raise ProofV3VerificationError(
            "hard_proof_arrival_budget_ns must be between one nanosecond "
            "and thirty minutes"
        )
    return value


def hard_proof_arrival_budget_for_decode_v3(
    decode_tokens: int,
    *,
    ordinary_decode_tokens: int = HARD_PROOF_EXTENDED_DECODE_START_TOKENS_V3,
    max_decode_tokens: int = HARD_PROOF_EXTENDED_DECODE_LIMIT_TOKENS_V3,
    ordinary_budget_ns: int = DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3,
    max_budget_ns: int = LEGACY_MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3,
) -> int:
    """Return the bounded validator deadline for one authenticated geometry."""

    if (
        isinstance(decode_tokens, bool)
        or not isinstance(decode_tokens, int)
        or decode_tokens <= 0
    ):
        raise ProofV3VerificationError(
            "hard-proof decode token count must be a positive integer"
        )
    for value, name in (
        (ordinary_decode_tokens, "ordinary_decode_tokens"),
        (max_decode_tokens, "max_decode_tokens"),
        (ordinary_budget_ns, "ordinary_budget_ns"),
        (max_budget_ns, "max_budget_ns"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProofV3VerificationError(
                f"hard-proof deadline {name} must be a positive integer"
            )
    if ordinary_decode_tokens > max_decode_tokens:
        raise ProofV3VerificationError(
            "hard-proof ordinary decode boundary exceeds its maximum"
        )
    if ordinary_budget_ns > max_budget_ns:
        raise ProofV3VerificationError(
            "hard-proof ordinary budget exceeds its maximum"
        )
    if max_budget_ns > MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3:
        raise ProofV3VerificationError(
            "hard-proof deadline exceeds the implementation safety ceiling"
        )
    if decode_tokens > max_decode_tokens:
        raise ProofV3VerificationError(
            "hard-proof decode token count exceeds the qualified deadline geometry"
        )
    if decode_tokens <= ordinary_decode_tokens:
        return ordinary_budget_ns
    token_span = max_decode_tokens - ordinary_decode_tokens
    if token_span <= 0:
        return max_budget_ns
    extension_span = max_budget_ns - ordinary_budget_ns
    extension_tokens = decode_tokens - ordinary_decode_tokens
    extension_ns = (
        extension_tokens * extension_span + token_span - 1
    ) // token_span
    return ordinary_budget_ns + extension_ns


def _nonce_reveal_hold_budget_ns(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= MAX_NONCE_REVEAL_HOLD_BUDGET_NS_V3
    ):
        raise ProofV3VerificationError(
            "nonce_reveal_hold_budget_ns must be between one nanosecond "
            "and 930 seconds"
        )
    return value


@dataclass(frozen=True, slots=True)
class NonceRevealV3:
    """The sole nonce reveal bound to one already accepted envelope."""

    proof_challenge_id: bytes
    precommit_context_digest: bytes
    execution_profile_digest: bytes
    cache_lease_digest: bytes
    commitment_envelope_digest: bytes
    validator_nonce: bytes
    audit_decision: PostCommitAuditDecisionV3

    def __post_init__(self) -> None:
        _fixed32(self.proof_challenge_id, "proof_challenge_id")
        _fixed32(self.precommit_context_digest, "precommit_context_digest")
        _fixed32(self.execution_profile_digest, "execution_profile_digest")
        _fixed32(self.cache_lease_digest, "cache_lease_digest", nonzero=True)
        _fixed32(self.commitment_envelope_digest, "commitment_envelope_digest")
        _fixed32(self.validator_nonce, "validator_nonce", nonzero=True)
        if not isinstance(self.audit_decision, PostCommitAuditDecisionV3):
            raise ProofV3VerificationError(
                "nonce reveal audit_decision has an unexpected type"
            )
        if (
            self.audit_decision.commitment_envelope_digest
            != self.commitment_envelope_digest
        ):
            raise ProofV3VerificationError(
                "nonce reveal audit decision has an unexpected envelope"
            )

    def opening_ticket(self) -> PostCommitOpeningTicketV3:
        """Return the one-use ticket a native retained-state sidecar consumes."""

        if not self.audit_decision.hard_audit_selected:
            raise ProofV3VerificationError(
                "a light postcommit decision does not permit a hard opening ticket"
            )

        return PostCommitOpeningTicketV3(
            proof_challenge_id=self.proof_challenge_id,
            precommit_context_digest=self.precommit_context_digest,
            execution_profile_digest=self.execution_profile_digest,
            cache_lease_digest=self.cache_lease_digest,
            commitment_envelope_digest=self.commitment_envelope_digest,
            validator_nonce=self.validator_nonce,
        )

    def canonical_bytes(self) -> bytes:
        """Serialize the exact inline postcommit nonce-reveal frame."""

        decision = self.audit_decision.canonical_bytes()
        if len(decision) > 0xFFFF:
            raise ProofV3VerificationError("nonce reveal decision exceeds its wire limit")
        return (
            _NONCE_REVEAL_MAGIC_V3
            + struct.pack("<H", NONCE_REVEAL_FORMAT_VERSION_V3)
            + self.proof_challenge_id
            + self.precommit_context_digest
            + self.execution_profile_digest
            + self.cache_lease_digest
            + self.commitment_envelope_digest
            + self.validator_nonce
            + struct.pack("<H", len(decision))
            + decision
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "NonceRevealV3":
        """Parse one bounded nonce reveal without trusting its contents."""

        fixed_size = 4 + 2 + 6 * 32 + 2
        if (
            not isinstance(encoded, bytes)
            or len(encoded) < fixed_size + 1
            or len(encoded) > MAX_NONCE_REVEAL_BYTES_V3
        ):
            raise ProofV3VerificationError("nonce reveal byte length is invalid")
        if encoded[:4] != _NONCE_REVEAL_MAGIC_V3:
            raise ProofV3VerificationError("nonce reveal header is unsupported")
        version = struct.unpack_from("<H", encoded, 4)[0]
        if version != NONCE_REVEAL_FORMAT_VERSION_V3:
            raise ProofV3VerificationError("nonce reveal version is unsupported")
        offset = 6
        proof_challenge_id = encoded[offset : offset + 32]
        offset += 32
        precommit_context_digest = encoded[offset : offset + 32]
        offset += 32
        execution_profile_digest = encoded[offset : offset + 32]
        offset += 32
        cache_lease_digest = encoded[offset : offset + 32]
        offset += 32
        commitment_envelope_digest = encoded[offset : offset + 32]
        offset += 32
        validator_nonce = encoded[offset : offset + 32]
        offset += 32
        decision_length = struct.unpack_from("<H", encoded, offset)[0]
        offset += 2
        if decision_length == 0 or offset + decision_length != len(encoded):
            raise ProofV3VerificationError("nonce reveal decision framing is invalid")
        try:
            decision = PostCommitAuditDecisionV3.from_canonical_bytes(
                encoded[offset:]
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError("nonce reveal decision is malformed") from exc
        result = cls(
            proof_challenge_id=proof_challenge_id,
            precommit_context_digest=precommit_context_digest,
            execution_profile_digest=execution_profile_digest,
            cache_lease_digest=cache_lease_digest,
            commitment_envelope_digest=commitment_envelope_digest,
            validator_nonce=validator_nonce,
            audit_decision=decision,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3VerificationError("nonce reveal is not canonical")
        return result


@dataclass(frozen=True, slots=True, init=False)
class QualifiedExecutionProfileV3:
    """Validator-owned profile qualified before any request can be served.

    Construction through :meth:`from_signed_document` verifies the authority
    threshold, selected static-manifest digest, exact profile digest, loaded
    artifacts, and native adapter once at catalog-load time.  A request session
    accepts only this already-qualified object; it never accepts a raw miner-
    or caller-selected profile/document.
    """

    profile: ExecutionSecurityProfileV3
    expected_static_manifest_digest: bytes
    expected_execution_profile_digest: bytes
    registration: object
    verified_signers: tuple[str, ...]
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "qualified profiles must be created through from_signed_document"
        )

    @classmethod
    def _construct_from_signed_document(
        cls,
        *,
        profile: ExecutionSecurityProfileV3,
        expected_static_manifest_digest: bytes,
        expected_execution_profile_digest: bytes,
        registration: object,
        verified_signers: tuple[str, ...],
        _factory_token: object | None = None,
    ) -> "QualifiedExecutionProfileV3":
        """Mint the opaque result of a completed authority qualification."""

        if _factory_token is not _QUALIFIED_PROFILE_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "qualified profiles must be created through from_signed_document"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "profile", profile)
        object.__setattr__(
            result,
            "expected_static_manifest_digest",
            expected_static_manifest_digest,
        )
        object.__setattr__(
            result,
            "expected_execution_profile_digest",
            expected_execution_profile_digest,
        )
        object.__setattr__(result, "registration", registration)
        object.__setattr__(result, "verified_signers", verified_signers)
        object.__setattr__(result, "_factory_token", _QUALIFIED_PROFILE_FACTORY_TOKEN)
        result._validate_qualification(validate_registration=True)
        return result

    def _validate_qualification(self, *, validate_registration: bool) -> None:
        if self._factory_token is not _QUALIFIED_PROFILE_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "qualified profile requires signed-document qualification provenance"
            )
        if not isinstance(self.profile, ExecutionSecurityProfileV3):
            raise ProofV3VerificationError("qualified profile has an unexpected type")
        static_digest = _fixed32(
            self.expected_static_manifest_digest,
            "expected_static_manifest_digest",
        )
        profile_digest = _fixed32(
            self.expected_execution_profile_digest,
            "expected_execution_profile_digest",
        )
        if self.profile.static_manifest_digest != static_digest:
            raise ProofV3VerificationError(
                "qualified profile has an unexpected static manifest"
            )
        if self.profile.digest() != profile_digest:
            raise ProofV3VerificationError(
                "qualified profile has an unexpected execution profile digest"
            )
        from verallm.proof_v3.economic_recompute_adapter import (
            ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
        )
        from verallm.proof_v3.economic_registry import QualifiedEconomicAdapterV3
        from verallm.proof_v3.profile import (
            GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3,
        )

        if self.profile.proof_system_id == GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3:
            if not isinstance(self.registration, QualifiedExecutionAdapterV3):
                raise ProofV3VerificationError(
                    "qualified profile has an unqualified folded adapter "
                    "registration"
                )
            self.profile.require_hard_execution_capability()
        elif self.profile.proof_system_id == ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
            if not isinstance(self.registration, QualifiedEconomicAdapterV3):
                raise ProofV3VerificationError(
                    "qualified profile has an unqualified economic adapter "
                    "registration"
                )
        else:
            raise ProofV3VerificationError(
                "qualified profile selects an unsupported proof system"
            )
        signers = tuple(self.verified_signers)
        if not signers or signers != tuple(sorted(set(signers))):
            raise ProofV3VerificationError(
                "qualified profile signer set is not canonical"
            )
        object.__setattr__(self, "verified_signers", signers)
        if validate_registration:
            self.registration.validate_profile(profile=self.profile)

    def require_qualification_provenance(self) -> None:
        """Check the factory-minted authority qualification without reloading.

        This is an accidental-misuse guard for trusted validator code. It does
        not claim to protect against code already able to modify that validator
        process; the authority signatures are verified by
        :meth:`from_signed_document` before this record is minted.
        """

        self._validate_qualification(validate_registration=False)

    @classmethod
    def from_signed_document(
        cls,
        *,
        document: SignedExecutionProfileDocumentV3,
        expected_static_manifest_digest: bytes,
        expected_execution_profile_digest: bytes,
        expected_authority_signers: Collection[str | bytes],
        authority_threshold: int,
        registration: object,
    ) -> "QualifiedExecutionProfileV3":
        """Authenticate and qualify one immutable profile catalog entry."""

        if not isinstance(document, SignedExecutionProfileDocumentV3):
            raise ProofV3VerificationError(
                "signed execution profile document has an unexpected type"
            )
        static_digest = _fixed32(
            expected_static_manifest_digest,
            "expected_static_manifest_digest",
        )
        profile_digest = _fixed32(
            expected_execution_profile_digest,
            "expected_execution_profile_digest",
        )
        profile = document.profile
        if profile.digest() != profile_digest:
            raise ProofV3VerificationError(
                "signed execution profile was not selected by the validator catalog"
            )
        try:
            verified_signers = verify_signed_execution_profile_v3(
                profile,
                document.signatures,
                expected_static_manifest_digest=static_digest,
                expected_authority_signers=expected_authority_signers,
                authority_threshold=authority_threshold,
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "signed execution profile authority validation failed"
            ) from exc
        return cls._construct_from_signed_document(
            profile=profile,
            expected_static_manifest_digest=static_digest,
            expected_execution_profile_digest=profile_digest,
            registration=registration,
            verified_signers=verified_signers,
            _factory_token=_QUALIFIED_PROFILE_FACTORY_TOKEN,
        )


class ProofV3ChallengeSession:
    """Thread-safe, fail-closed lifecycle for one eligible proof request.

    The object retains only fixed-size canonical request/output roots after
    issue/accept. It never retains the original prompt or observed token
    sequence, which keeps the validator-side control plane bounded at 250k and
    1m context lengths.
    """

    __slots__ = (
        "_binding",
        "_audit_decision",
        "_deadline_monotonic_ns",
        "_envelope",
        "_expected_execution_profile_digest",
        "_expected_static_manifest_digest",
        "_lock",
        "_nonce",
        "_precommit_context",
        "_precommit_received_monotonic_ns",
        "_profile",
        "_precommit_arrival_budget_ns",
        "_hard_proof_arrival_budget_ns",
        "_nonce_reveal_hold_budget_ns",
        "_registration",
        "_runtime_policy",
        "_nonce_revealed_monotonic_ns",
        "_tier_selected_monotonic_ns",
        "_verified_capture_chain_digest",
        "_state",
    )

    def __init__(
        self,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        precommit_context: PreExecutionRequestContextV3,
        validator_nonce: bytes,
        runtime_policy: RuntimeHardAuditPolicyV3,
        proof_arrival_budget_ns: int,
        hard_proof_arrival_budget_ns: int = (
            DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3
        ),
        nonce_reveal_hold_budget_ns: int | None = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CHALLENGE_SESSION_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "challenge sessions must be issued through ProofV3ChallengeSession.issue"
            )
        if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
            raise ProofV3VerificationError("qualified_profile has an unexpected type")
        qualified_profile.require_qualification_provenance()
        if not isinstance(precommit_context, PreExecutionRequestContextV3):
            raise ProofV3VerificationError("precommit context has an unexpected type")
        profile = qualified_profile.profile
        static_digest = qualified_profile.expected_static_manifest_digest
        profile_digest = qualified_profile.expected_execution_profile_digest
        nonce = _fixed32(validator_nonce, "validator_nonce", nonzero=True)
        precommit_budget = _arrival_budget_ns(proof_arrival_budget_ns)
        hard_proof_budget = _hard_arrival_budget_ns(
            hard_proof_arrival_budget_ns
        )
        nonce_reveal_hold_budget = _nonce_reveal_hold_budget_ns(
            nonce_reveal_hold_budget_ns
        )
        profile.relation_spec.audit_policy.validate_runtime(runtime_policy)
        if profile.static_manifest_digest != static_digest:
            raise ProofV3VerificationError(
                "execution profile has an unexpected static manifest"
            )
        if profile.digest() != profile_digest:
            raise ProofV3VerificationError(
                "execution profile was not selected by the validator catalog"
            )
        if precommit_context.static_manifest_digest != static_digest:
            raise ProofV3VerificationError(
                "precommit context has an unexpected static manifest"
            )
        if precommit_context.execution_profile_digest != profile_digest:
            raise ProofV3VerificationError(
                "precommit context has an unexpected profile"
            )
        expected_nonce_commitment = commit_validator_nonce_v3(
            validator_nonce=nonce,
            nonce_context_digest=precommit_context.nonce_context_digest(),
        )
        if precommit_context.validator_nonce_commitment != expected_nonce_commitment:
            raise ProofV3VerificationError(
                "validator nonce does not match the pre-request commitment"
            )
        self._profile = profile
        self._expected_static_manifest_digest = static_digest
        self._expected_execution_profile_digest = profile_digest
        self._precommit_context = precommit_context
        self._nonce: bytes | None = nonce
        self._runtime_policy = runtime_policy
        self._precommit_arrival_budget_ns = precommit_budget
        self._hard_proof_arrival_budget_ns = hard_proof_budget
        self._nonce_reveal_hold_budget_ns = nonce_reveal_hold_budget
        self._registration = qualified_profile.registration
        self._binding: EnvelopeBindingV3 | None = None
        self._audit_decision: PostCommitAuditDecisionV3 | None = None
        self._envelope: ProofV3CommitmentEnvelope | None = None
        self._deadline_monotonic_ns: int | None = None
        self._precommit_received_monotonic_ns: int | None = None
        self._nonce_revealed_monotonic_ns: int | None = None
        self._tier_selected_monotonic_ns: int | None = None
        self._verified_capture_chain_digest: bytes | None = None
        self._state = ChallengeSessionStateV3.AWAITING_PRECOMMIT
        self._lock = threading.Lock()

    @classmethod
    def issue(
        cls,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        proof_challenge_id: bytes,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        prompt_token_ids: Sequence[int],
        sampler_config_digest: bytes,
        runtime_policy: RuntimeHardAuditPolicyV3,
        proof_arrival_budget_ns: int = DEFAULT_PRECOMMIT_ARRIVAL_BUDGET_NS_V3,
        hard_proof_arrival_budget_ns: int = (
            DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3
        ),
        nonce_reveal_hold_budget_ns: int | None = None,
        expected_hard_audit: bool | None = None,
    ) -> tuple["ProofV3ChallengeSession", ValidatorExecutionRequestContextV3]:
        """Issue a production session with validator-owned fresh entropy."""

        return cls._issue_with_entropy(
            qualified_profile=qualified_profile,
            proof_challenge_id=proof_challenge_id,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            prompt_token_ids=prompt_token_ids,
            sampler_config_digest=sampler_config_digest,
            runtime_policy=runtime_policy,
            proof_arrival_budget_ns=proof_arrival_budget_ns,
            hard_proof_arrival_budget_ns=hard_proof_arrival_budget_ns,
            nonce_reveal_hold_budget_ns=nonce_reveal_hold_budget_ns,
            expected_hard_audit=expected_hard_audit,
            nonce_source=secrets.token_bytes,
            cache_lease_source=secrets.token_bytes,
            _factory_token=_ISSUE_WITH_ENTROPY_FACTORY_TOKEN,
        )

    @classmethod
    def _issue_with_entropy(
        cls,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        proof_challenge_id: bytes,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        prompt_token_ids: Sequence[int],
        sampler_config_digest: bytes,
        runtime_policy: RuntimeHardAuditPolicyV3,
        proof_arrival_budget_ns: int,
        nonce_source: Callable[[int], bytes],
        cache_lease_source: Callable[[int], bytes],
        hard_proof_arrival_budget_ns: int = (
            DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3
        ),
        nonce_reveal_hold_budget_ns: int | None = None,
        expected_hard_audit: bool | None = None,
        _factory_token: object | None = None,
    ) -> tuple["ProofV3ChallengeSession", ValidatorExecutionRequestContextV3]:
        """Issue a public context while retaining the nonce only in this session.

        This private helper receives the entropy source only after the public
        production method has selected ``secrets.token_bytes``. The cache lease
        is validator-generated, so a miner cannot choose a stale or
        cross-request cache namespace.
        """

        if _factory_token is not _ISSUE_WITH_ENTROPY_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "challenge entropy issuance must use the public issue factory"
            )
        if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
            raise ProofV3VerificationError("qualified_profile has an unexpected type")
        qualified_profile.require_qualification_provenance()
        if not callable(nonce_source) or not callable(cache_lease_source):
            raise ProofV3VerificationError(
                "nonce and cache lease sources must be callable"
            )
        nonce = _fixed32(nonce_source(32), "validator_nonce", nonzero=True)
        cache_lease_digest = _fixed32(
            cache_lease_source(32),
            "cache_lease_digest",
            nonzero=True,
        )
        if expected_hard_audit is not None and not isinstance(
            expected_hard_audit,
            bool,
        ):
            raise ProofV3VerificationError(
                "expected_hard_audit must be a boolean when provided"
            )
        if (
            expected_hard_audit is not None
            and nonce_reveal_hold_budget_ns is None
        ):
            raise ProofV3VerificationError(
                "paired tier conditioning requires a nonce-reveal hold budget"
            )
        static_digest = qualified_profile.expected_static_manifest_digest
        profile_digest = qualified_profile.expected_execution_profile_digest
        nonce_context = derive_pre_nonce_context_digest_v3(
            proof_challenge_id=proof_challenge_id,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            static_manifest_digest=static_digest,
            execution_profile_digest=profile_digest,
            prompt_token_ids=prompt_token_ids,
            cache_lease_digest=cache_lease_digest,
            sampler_config_digest=sampler_config_digest,
        )
        profile = qualified_profile.profile
        profile.relation_spec.audit_policy.validate_runtime(runtime_policy)
        effective_hard_bps = (
            runtime_policy.effective_canary_hard_bps
            if runtime_policy.request_kind == "canary"
            else runtime_policy.effective_organic_hard_bps
        )
        if expected_hard_audit is True and effective_hard_bps == 0:
            raise ProofV3VerificationError(
                "paired hard tier cannot use a zero-rate runtime policy"
            )
        if expected_hard_audit is False and effective_hard_bps == 10_000:
            raise ProofV3VerificationError(
                "paired light tier cannot use an always-hard runtime policy"
            )
        for _attempt in range(_MAX_PAIRED_TIER_NONCE_ATTEMPTS_V3):
            if _attempt:
                nonce = _fixed32(
                    nonce_source(32),
                    "validator_nonce",
                    nonzero=True,
                )
            public_context = ValidatorExecutionRequestContextV3(
                proof_challenge_id=proof_challenge_id,
                validator_identity_digest=validator_identity_digest,
                miner_identity_digest=miner_identity_digest,
                validator_nonce_commitment=commit_validator_nonce_v3(
                    validator_nonce=nonce,
                    nonce_context_digest=nonce_context,
                ),
                prompt_token_ids=prompt_token_ids,
                cache_lease_digest=cache_lease_digest,
                sampler_config_digest=sampler_config_digest,
            )
            precommit_context = public_context.derive_precommit_context(
                static_manifest_digest=static_digest,
                execution_profile_digest=profile_digest,
            )
            if expected_hard_audit is None:
                break
            draw = _audit_tier_draw_from_precommit_context_v3(
                validator_nonce=nonce,
                profile=profile,
                precommit_context_digest=precommit_context.digest(),
                runtime_policy=runtime_policy,
                effective_hard_bps=effective_hard_bps,
            )
            if (draw < effective_hard_bps) is expected_hard_audit:
                break
        else:
            raise ProofV3VerificationError(
                "unable to issue a validator nonce for the paired audit tier"
            )
        return (
            cls(
                qualified_profile=qualified_profile,
                precommit_context=precommit_context,
                validator_nonce=nonce,
                runtime_policy=runtime_policy,
                proof_arrival_budget_ns=proof_arrival_budget_ns,
                hard_proof_arrival_budget_ns=hard_proof_arrival_budget_ns,
                nonce_reveal_hold_budget_ns=nonce_reveal_hold_budget_ns,
                _factory_token=_CHALLENGE_SESSION_FACTORY_TOKEN,
            ),
            public_context,
        )

    @property
    def state(self) -> ChallengeSessionStateV3:
        """Return the current lifecycle state without exposing witness data."""

        with self._lock:
            return self._state

    @property
    def precommit_context(self) -> PreExecutionRequestContextV3:
        """Return the fixed-size public context sent before inference."""

        return self._precommit_context

    @property
    def nonce_reveal_hold_budget_ns(self) -> int | None:
        """Return the bounded post-precommit hold configured at issuance."""

        return self._nonce_reveal_hold_budget_ns

    @property
    def hard_proof_arrival_budget_ns(self) -> int:
        """Return the validator-owned post-nonce arrival budget."""

        return self._hard_proof_arrival_budget_ns

    @property
    def maximum_hard_proof_transport_bytes(self) -> int:
        """Return the adapter-authenticated hard-proof transport ceiling."""

        from verallm.proof_v3.economic_transport import (
            MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
            MAX_ECONOMIC_TRANSPORT_BYTES,
        )

        return (
            MAX_ECONOMIC_MOE_TRANSPORT_BYTES
            if getattr(
                getattr(self._registration, "artifacts", None),
                "moe_runtime_semantics",
                None,
            ) is not None
            else MAX_ECONOMIC_TRANSPORT_BYTES
        )

    @property
    def verified_capture_chain_digest(self) -> bytes | None:
        """Return the authenticated economic capture chain after verification."""

        with self._lock:
            return self._verified_capture_chain_digest

    def _fail_locked(self, *, state: ChallengeSessionStateV3) -> None:
        self._state = state
        self._nonce = None
        self._binding = None
        self._audit_decision = None
        self._envelope = None
        self._precommit_received_monotonic_ns = None
        self._nonce_revealed_monotonic_ns = None
        self._tier_selected_monotonic_ns = None
        self._verified_capture_chain_digest = None

    def _require_arrival_before_deadline_locked(
        self, received_monotonic_ns: int
    ) -> int:
        received = _monotonic_ns(received_monotonic_ns, "received_monotonic_ns")
        deadline = self._deadline_monotonic_ns
        if deadline is None or received > deadline:
            state = self._state.value
            overdue_ms = (
                "unknown"
                if deadline is None
                else f"{(received - deadline) / 1_000_000:.3f}"
            )
            self._fail_locked(state=ChallengeSessionStateV3.EXPIRED)
            raise ProofV3VerificationError(
                "proof-v3 response arrived after its deadline "
                f"(state={state}, overdue_ms={overdue_ms})"
            )
        return received

    def accept_precommit_bytes(
        self,
        *,
        encoded_envelope: bytes,
        observed_output: ObservedExecutionOutputV3,
        last_visible_token_monotonic_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        """Atomically accept the one final envelope before nonce disclosure.

        Any malformed, late, duplicate, or mismatched envelope is terminal. The
        validator does not reveal its nonce on any failure path. Runtime
        integration must obtain both timestamps from the validator's local
        monotonic clock at its observed SSE boundaries; miner-provided timing
        metadata is not an acceptable source for either value.
        """

        with self._lock:
            if self._state != ChallengeSessionStateV3.AWAITING_PRECOMMIT:
                raise ProofV3VerificationError(
                    "proof-v3 session does not accept another precommit envelope"
                )
            try:
                last_visible = _monotonic_ns(
                    last_visible_token_monotonic_ns,
                    "last_visible_token_monotonic_ns",
                )
                deadline = last_visible + self._precommit_arrival_budget_ns
                if deadline >= 1 << 63:
                    raise ProofV3VerificationError(
                        "proof-v3 arrival deadline overflows"
                    )
                self._deadline_monotonic_ns = deadline
                received = self._require_arrival_before_deadline_locked(
                    received_monotonic_ns
                )
                if received < last_visible:
                    raise ProofV3VerificationError(
                        "proof-v3 precommit envelope arrived before the last visible token"
                    )
                envelope = commitment_envelope_from_bytes(encoded_envelope)
                if not isinstance(observed_output, ObservedExecutionOutputV3):
                    raise ProofV3VerificationError(
                        "observed_output has an unexpected type"
                    )
                capability_requirement = None
                from verallm.proof_v3.economic_recompute_adapter import (
                    ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
                )

                if (
                    self._profile.proof_system_id
                    == ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3
                ):
                    from verallm.proof_v3.economic_registry import (
                        require_economic_recompute_capability_v3,
                    )

                    capability_requirement = (
                        require_economic_recompute_capability_v3
                    )
                unverified_context = (
                    self._precommit_context.context_token_count
                    > self._profile.max_verified_context_tokens
                )
                if unverified_context:
                    effective_hard_bps = (
                        self._runtime_policy.effective_canary_hard_bps
                        if self._runtime_policy.request_kind == "canary"
                        else self._runtime_policy.effective_organic_hard_bps
                    )
                    if effective_hard_bps != 0:
                        raise ProofV3VerificationError(
                            "context above the signed hard-proof ceiling "
                            "requires a light-only runtime policy"
                        )
                binding = validate_execution_envelope_against_precommit_v3(
                    profile=self._profile,
                    envelope=envelope,
                    expected_static_manifest_digest=self._expected_static_manifest_digest,
                    expected_execution_profile_digest=self._expected_execution_profile_digest,
                    precommit_context=self._precommit_context,
                    output_binding=observed_output.derive_output_binding(),
                    capability_requirement=capability_requirement,
                    allow_unverified_context=unverified_context,
                )
            except ProofV3VerificationError:
                if self._state != ChallengeSessionStateV3.EXPIRED:
                    self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise
            except Exception as exc:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise ProofV3VerificationError(
                    "proof-v3 precommit envelope is malformed or mismatched"
                ) from exc
            self._envelope = envelope
            self._binding = binding
            self._precommit_received_monotonic_ns = received
            self._state = ChallengeSessionStateV3.PRECOMMIT_ACCEPTED

    def _select_audit_tier_locked(
        self,
        *,
        selected_monotonic_ns: int,
    ) -> PostCommitAuditDecisionV3:
        if self._state != ChallengeSessionStateV3.PRECOMMIT_ACCEPTED:
            raise ProofV3VerificationError(
                "proof-v3 session has no accepted precommit to select"
            )
        try:
            selected = _monotonic_ns(
                selected_monotonic_ns,
                "selected_monotonic_ns",
            )
            if (
                self._precommit_received_monotonic_ns is None
                or selected < self._precommit_received_monotonic_ns
            ):
                raise ProofV3VerificationError(
                    "proof-v3 audit tier was selected before its accepted "
                    "precommit envelope"
                )
        except ProofV3VerificationError:
            if self._state != ChallengeSessionStateV3.EXPIRED:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
            raise
        assert self._envelope is not None
        assert self._nonce is not None
        try:
            decision = derive_postcommit_audit_decision_v3(
                validator_nonce=self._nonce,
                profile=self._profile,
                envelope=self._envelope,
                runtime_policy=self._runtime_policy,
            )
        except Exception as exc:
            self._fail_locked(state=ChallengeSessionStateV3.FAILED)
            raise ProofV3VerificationError(
                "proof-v3 postcommit audit decision is malformed"
            ) from exc
        if not decision.hard_audit_selected:
            self._fail_locked(state=ChallengeSessionStateV3.LIGHT_REVEALED)
            return decision
        self._audit_decision = decision
        self._tier_selected_monotonic_ns = selected
        if self._nonce_reveal_hold_budget_ns is not None:
            assert self._precommit_received_monotonic_ns is not None
            hold_deadline = (
                self._precommit_received_monotonic_ns
                + self._nonce_reveal_hold_budget_ns
            )
            if hold_deadline >= 1 << 63:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise ProofV3VerificationError(
                    "proof-v3 nonce-reveal hold deadline overflows"
                )
            self._deadline_monotonic_ns = hold_deadline
            self._require_arrival_before_deadline_locked(selected)
        else:
            # Tier selection is validator-local work after the peer has met
            # its precommit deadline. Give the immediately following local
            # nonce serialization its own bounded window instead of reusing
            # the already-consumed peer arrival deadline.
            reveal_deadline = (
                selected + DEFAULT_LOCAL_NONCE_REVEAL_BUDGET_NS_V3
            )
            if reveal_deadline >= 1 << 63:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise ProofV3VerificationError(
                    "proof-v3 local nonce-reveal deadline overflows"
                )
            self._deadline_monotonic_ns = reveal_deadline
        self._state = ChallengeSessionStateV3.HARD_SELECTED
        return decision

    def select_audit_tier_once(
        self,
        *,
        selected_monotonic_ns: int,
    ) -> PostCommitAuditDecisionV3:
        """Select hard/light locally without disclosing the nonce on light."""

        with self._lock:
            return self._select_audit_tier_locked(
                selected_monotonic_ns=selected_monotonic_ns
            )

    def reveal_nonce_once(self, *, revealed_monotonic_ns: int) -> NonceRevealV3:
        """Reveal the validator nonce exactly once, and only for a hard tier."""

        with self._lock:
            if self._state == ChallengeSessionStateV3.PRECOMMIT_ACCEPTED:
                decision = self._select_audit_tier_locked(
                    selected_monotonic_ns=revealed_monotonic_ns
                )
                if not decision.hard_audit_selected:
                    raise ProofV3VerificationError(
                        "proof-v3 light decision does not reveal validator nonce"
                    )
            elif (
                self._state != ChallengeSessionStateV3.HARD_SELECTED
                or self._audit_decision is None
            ):
                raise ProofV3VerificationError(
                    "proof-v3 session has no hard decision to reveal"
                )
            try:
                revealed = self._require_arrival_before_deadline_locked(
                    revealed_monotonic_ns
                )
                if (
                    self._precommit_received_monotonic_ns is None
                    or revealed < self._precommit_received_monotonic_ns
                    or self._tier_selected_monotonic_ns is None
                    or revealed < self._tier_selected_monotonic_ns
                ):
                    raise ProofV3VerificationError(
                        "proof-v3 nonce was revealed before its accepted "
                        "precommit envelope"
                    )
            except ProofV3VerificationError:
                if self._state != ChallengeSessionStateV3.EXPIRED:
                    self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise
            assert self._envelope is not None
            assert self._nonce is not None
            assert self._audit_decision is not None
            try:
                decision = self._audit_decision
                reveal = NonceRevealV3(
                    proof_challenge_id=self._precommit_context.proof_challenge_id,
                    precommit_context_digest=self._precommit_context.digest(),
                    execution_profile_digest=self._expected_execution_profile_digest,
                    cache_lease_digest=self._precommit_context.cache_lease_digest,
                    commitment_envelope_digest=self._envelope.digest(),
                    validator_nonce=self._nonce,
                    audit_decision=decision,
                )
            except Exception as exc:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise ProofV3VerificationError(
                    "proof-v3 postcommit nonce reveal is malformed"
                ) from exc
            proof_deadline = revealed + self._hard_proof_arrival_budget_ns
            if proof_deadline >= 1 << 63:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise ProofV3VerificationError(
                    "proof-v3 hard proof deadline overflows"
                )
            self._nonce_revealed_monotonic_ns = revealed
            self._deadline_monotonic_ns = proof_deadline
            self._state = ChallengeSessionStateV3.NONCE_REVEALED
            return reveal

    def verify_proof_bytes_once(
        self,
        *,
        encoded_proof: bytes,
        received_monotonic_ns: int,
        validator_request_context: ValidatorExecutionRequestContextV3 | None = None,
        observed_output: ObservedExecutionOutputV3 | None = None,
    ) -> object:
        """Verify one post-reveal proof bound to the accepted envelope only."""

        with self._lock:
            if self._state != ChallengeSessionStateV3.NONCE_REVEALED:
                if self._state == ChallengeSessionStateV3.LIGHT_REVEALED:
                    raise ProofV3VerificationError(
                        "proof-v3 light decision does not accept a hard proof"
                    )
                raise ProofV3VerificationError(
                    "proof-v3 session does not accept another proof"
                )
            try:
                received = self._require_arrival_before_deadline_locked(
                    received_monotonic_ns
                )
                if (
                    self._nonce_revealed_monotonic_ns is None
                    or received < self._nonce_revealed_monotonic_ns
                ):
                    raise ProofV3VerificationError(
                        "proof-v3 proof arrived before its nonce reveal"
                    )
            except ProofV3VerificationError:
                if self._state != ChallengeSessionStateV3.EXPIRED:
                    self._fail_locked(state=ChallengeSessionStateV3.FAILED)
                raise
            assert self._envelope is not None
            assert self._binding is not None
            assert self._nonce is not None
            assert self._audit_decision is not None
            envelope = self._envelope
            binding = self._binding
            nonce = self._nonce
            audit_decision = self._audit_decision
            adapters = {
                self._expected_execution_profile_digest: self._registration,
            }
            self._state = ChallengeSessionStateV3.VERIFYING
        verified_capture_chain_digest: bytes | None = None
        try:
            from verallm.proof_v3.economic_recompute_adapter import (
                ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
            )

            if self._profile.proof_system_id == ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
                from verallm.proof_v3.economic_registry import (
                    verify_economic_execution_proof_v3,
                )
                from verallm.proof_v3.economic_transport import (
                    decode_economic_proof_transport_v3,
                )

                if not isinstance(
                    validator_request_context,
                    ValidatorExecutionRequestContextV3,
                ) or not isinstance(observed_output, ObservedExecutionOutputV3):
                    raise ProofV3VerificationError(
                        "economic proof verification requires validator-owned "
                        "request and output context"
                    )
                proof = decode_economic_proof_transport_v3(
                    encoded_proof,
                    allow_sparse_moe=bool(
                        getattr(
                            getattr(self._registration, "artifacts", None),
                            "moe_runtime_semantics",
                            None,
                        )
                    ),
                )
                result = verify_economic_execution_proof_v3(
                    profile=self._profile,
                    envelope=envelope,
                    proof=proof,
                    validator_nonce=nonce,
                    expected_static_manifest_digest=(
                        self._expected_static_manifest_digest
                    ),
                    expected_execution_profile_digest=(
                        self._expected_execution_profile_digest
                    ),
                    validator_request_context=validator_request_context,
                    observed_output=observed_output,
                    runtime_policy=self._runtime_policy,
                    adapters=adapters,
                    audit_decision=audit_decision,
                )
                verified_capture_chain_digest = proof.capture_chain_digest
            else:
                proof = folded_execution_proof_from_bytes(encoded_proof)
                result = _verify_folded_execution_proof_against_binding_v3(
                    profile=self._profile,
                    envelope=envelope,
                    proof=proof,
                    validator_nonce=nonce,
                    expected_static_manifest_digest=(
                        self._expected_static_manifest_digest
                    ),
                    expected_execution_profile_digest=(
                        self._expected_execution_profile_digest
                    ),
                    binding=binding,
                    runtime_policy=self._runtime_policy,
                    audit_decision=audit_decision,
                    adapters=adapters,
                    registration_prequalified=True,
                )
        except Exception:
            with self._lock:
                if self._state == ChallengeSessionStateV3.VERIFYING:
                    self._fail_locked(state=ChallengeSessionStateV3.FAILED)
            raise
        with self._lock:
            if self._state != ChallengeSessionStateV3.VERIFYING:
                raise ProofV3VerificationError(
                    "proof-v3 session terminated while verification was running"
                )
            self._state = ChallengeSessionStateV3.VERIFIED
            self._nonce = None
            self._binding = None
            self._audit_decision = None
            self._envelope = None
            self._precommit_received_monotonic_ns = None
            self._nonce_revealed_monotonic_ns = None
            self._tier_selected_monotonic_ns = None
            self._verified_capture_chain_digest = verified_capture_chain_digest
        return result

    def expire(self, *, now_monotonic_ns: int) -> None:
        """Fail a pending session whose validator-local arrival budget elapsed."""

        with self._lock:
            if self._state in _TERMINAL_STATES:
                return
            now = _monotonic_ns(now_monotonic_ns, "now_monotonic_ns")
            if (
                self._deadline_monotonic_ns is None
                or now <= self._deadline_monotonic_ns
            ):
                return
            self._fail_locked(state=ChallengeSessionStateV3.EXPIRED)

    def fail_closed(self) -> None:
        """Discard all retained session state after a transport/runtime failure."""

        with self._lock:
            if self._state not in _TERMINAL_STATES:
                self._fail_locked(state=ChallengeSessionStateV3.FAILED)

    def abort(self) -> None:
        """Discard a locally cancelled session before it can reveal a nonce."""

        with self._lock:
            if self._state not in _TERMINAL_STATES:
                self._fail_locked(state=ChallengeSessionStateV3.ABORTED)


__all__ = [
    "ChallengeSessionStateV3",
    "DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3",
    "DEFAULT_LOCAL_NONCE_REVEAL_BUDGET_NS_V3",
    "DEFAULT_PROOF_ARRIVAL_BUDGET_NS_V3",
    "DEFAULT_PRECOMMIT_ARRIVAL_BUDGET_NS_V3",
    "HARD_PROOF_EXTENDED_DECODE_LIMIT_TOKENS_V3",
    "HARD_PROOF_EXTENDED_DECODE_START_TOKENS_V3",
    "MAX_NONCE_REVEAL_HOLD_BUDGET_NS_V3",
    "MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3",
    "MAX_NONCE_REVEAL_BYTES_V3",
    "NONCE_REVEAL_FORMAT_VERSION_V3",
    "NonceRevealV3",
    "ProofV3ChallengeSession",
    "QualifiedExecutionProfileV3",
    "hard_proof_arrival_budget_for_decode_v3",
]
