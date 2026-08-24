"""Validator-owned HTTP lifecycle for one proof-v3 inference request.

This module is deliberately independent of scoring and probation policy.  It
drives the already-qualified :class:`ProofV3ChallengeSession` over the miner's
SSE response and authenticated hard-opening endpoint, and returns one explicit
hard/light verdict to its caller.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)

import httpx

from verallm.api.economic_proof_v3 import (
    ECONOMIC_PROOF_V3_CHALLENGE_PATH,
    ECONOMIC_PROOF_V3_MEDIA_TYPE,
    ECONOMIC_PROOF_V3_RETENTION_PATH,
)
from verallm.api.proof_protocol import PROOF_PROTOCOL_V3
from verallm.proof_v3.challenge import PostCommitAuditDecisionV3
from verallm.proof_v3.errors import (
    ProofV3Error,
    ProofV3UnavailableError,
    ProofV3VerificationError,
)
from verallm.proof_v3.payload import commitment_envelope_from_bytes
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import ObservedExecutionOutputV3
from verallm.proof_v3.session import (
    ChallengeSessionStateV3,
    DEFAULT_HARD_PROOF_ARRIVAL_BUDGET_NS_V3,
    DEFAULT_PRECOMMIT_ARRIVAL_BUDGET_NS_V3,
    ProofV3ChallengeSession,
    QualifiedExecutionProfileV3,
)

MAX_PROOF_V3_SSE_EVENT_BYTES = 1 << 20

_FAILED_HARD_BUNDLE_COUNT = 8
_FAILED_HARD_BUNDLE_BYTES = 512 << 20
_FAILED_HARD_BUNDLE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _peer_error_summary(data: Mapping[str, object]) -> str:
    """Return a bounded, single-line diagnostic for a miner SSE error.

    The peer payload is untrusted and must never replace the stable public
    verifier error.  Keeping only its error/code fields in operator logs makes
    production failures diagnosable without reflecting arbitrary payload data
    to API clients.
    """

    parts: list[str] = []
    for key in ("code", "error"):
        value = data.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            normalized = " ".join(str(value).split())
            if normalized:
                parts.append(f"{key}={normalized[:384]}")
    return " ".join(parts) or "unspecified"


def _retain_failed_hard_bundle(encoded_bundle: bytes) -> Path:
    """Keep a bounded private replay artifact after verifier rejection.

    Failed hard proofs are otherwise absent from the receipt-matched miner
    bundle index, which makes an intermittent verifier rejection impossible to
    reproduce.  Owner hard audits are canaries, so this stores only bounded
    canary token ids/output plus the already-received public proof.  It has no
    effect on admission, verification, receipts, or the wire protocol.
    """

    from verallm.proof_v3.hard_bundle import MAX_HARD_BUNDLE_BYTES_V3

    if (
        not isinstance(encoded_bundle, bytes)
        or not encoded_bundle
        or len(encoded_bundle) > MAX_HARD_BUNDLE_BYTES_V3
    ):
        raise ProofV3VerificationError(
            "failed proof-v3 replay bundle is out of range"
        )
    base = Path(
        os.environ.get("VERALLM_DATA_DIR", str(Path.home() / ".verathos"))
    ).expanduser()
    directory = base / "proof_v3_failed_bundles"
    with _FAILED_HARD_BUNDLE_LOCK:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        # The retained-bundle digest is its final 32 bytes.  Include a
        # nanosecond timestamp so two independently malformed payloads for the
        # same envelope cannot overwrite one another.
        digest = encoded_bundle[-32:].hex()
        path = directory / f"{time.time_ns()}-{digest}.bundle"
        temporary = directory / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded_bundle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

        entries = sorted(
            directory.glob("*.bundle"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        retained_bytes = 0
        for index, entry in enumerate(entries):
            size = entry.stat().st_size
            retained_bytes += size
            if (
                index >= _FAILED_HARD_BUNDLE_COUNT
                or retained_bytes > _FAILED_HARD_BUNDLE_BYTES
            ):
                entry.unlink(missing_ok=True)
        return path


class ProofV3PeerFailure(ProofV3VerificationError):
    """The selected miner failed a v3 protocol obligation."""


class ProofV3PeerServiceFailure(ProofV3PeerFailure):
    """The selected miner reported an inference-service failure."""


class ProofV3FirstTokenTimeout(ProofV3PeerServiceFailure):
    """The peer emitted no token within the caller's routing deadline."""


class ProofV3PostcommitFailure(ProofV3PeerFailure):
    """A hard exchange failed after the validator disclosed its nonce."""


@dataclass(frozen=True, slots=True)
class ProofV3ExchangeResult:
    """Bounded result of one completed validator-owned v3 exchange."""

    text: str
    input_tokens: int
    output_token_ids: tuple[int, ...]
    finish_reason: str
    commitment_envelope_digest: bytes
    capture_chain_digest: bytes
    audit_decision: PostCommitAuditDecisionV3
    proof_verified: bool
    verification_result: object | None
    validator_request_start_ts: float
    validator_request_end_ts: float
    validator_request_ms: float
    round_trip_ms: float
    ttft_ms: float
    inference_ms: float
    last_token_to_precommit_ms: float
    nonce_to_proof_ms: float | None
    verification_ms: float | None
    proof_wire_bytes: int


class ProofV3ValidatorExchange:
    """Strict event consumer around one validator-owned challenge session."""

    def __init__(
        self,
        *,
        session: ProofV3ChallengeSession,
        validator_request_context,
        max_output_tokens: int,
    ) -> None:
        if not isinstance(session, ProofV3ChallengeSession):
            raise ProofV3VerificationError(
                "proof-v3 exchange session has an unexpected type"
            )
        self.session = session
        self.validator_request_context = validator_request_context
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 0 < max_output_tokens < 1 << 32
        ):
            raise ProofV3VerificationError(
                "proof-v3 exchange output-token bound is malformed"
            )
        self._max_output_tokens = max_output_tokens
        self._text_parts: list[str] = []
        self._token_ids: list[int] = []
        self._first_token_ns: int | None = None
        self._last_token_ns: int | None = None
        self._precommit_bytes: bytes | None = None
        self._precommit_received_ns: int | None = None
        self._observed_output: ObservedExecutionOutputV3 | None = None
        self._decision: PostCommitAuditDecisionV3 | None = None
        self._reveal_ns: int | None = None
        self._nonce_reveal_bytes: bytes | None = None
        self._verification_result: object | None = None
        self._verification_ns: int | None = None
        self._proof_received_ns: int | None = None
        self._proof_wire_bytes = 0
        self._request_started_wall = 0.0
        self._request_started_ns: int | None = None
        self._stream_finished_wall = 0.0
        self._stream_finished_ns: int | None = None

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
    ) -> "ProofV3ValidatorExchange":
        session, context = ProofV3ChallengeSession.issue(
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
        )
        return cls(
            session=session,
            validator_request_context=context,
            max_output_tokens=(
                qualified_profile.profile.max_verified_decode_tokens
            ),
        )

    def request_fields(self) -> dict[str, object]:
        """Return the exact fields added to the inference request."""

        return {
            "proof_protocol_version": PROOF_PROTOCOL_V3,
            "proof_v3_preexecution_context": (
                self.session.precommit_context.canonical_bytes().hex()
            ),
            "proof_v3_hard_proof_arrival_budget_ns": (
                self.session.hard_proof_arrival_budget_ns
            ),
        }

    @property
    def audit_decision(self) -> PostCommitAuditDecisionV3 | None:
        return self._decision

    @property
    def requires_hard_proof(self) -> bool:
        return bool(
            self._decision is not None
            and self._decision.hard_audit_selected
        )

    @property
    def hard_nonce_revealed(self) -> bool:
        return self._reveal_ns is not None

    @property
    def precommit_received(self) -> bool:
        """Whether the miner froze its signed request envelope on the wire."""

        return self._precommit_bytes is not None

    def fail_closed(self) -> None:
        self.session.fail_closed()

    def _require_dict(self, data: object) -> Mapping[str, object]:
        if not isinstance(data, Mapping):
            raise ProofV3VerificationError("proof-v3 SSE data is not an object")
        return data

    def _observe_token(
        self, data: Mapping[str, object], received_monotonic_ns: int
    ) -> None:
        if self._precommit_bytes is not None or self._observed_output is not None:
            raise ProofV3VerificationError(
                "proof-v3 token arrived after the precommit"
            )
        text = data.get("text")
        token_ids = data.get("token_ids")
        if not isinstance(text, str) or not isinstance(token_ids, list):
            raise ProofV3VerificationError(
                "proof-v3 token event lacks exact text or token ids"
            )
        parsed: list[int] = []
        for token_id in token_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < 1 << 32
            ):
                raise ProofV3VerificationError(
                    "proof-v3 output token id is malformed"
                )
            parsed.append(token_id)
        if not text and not parsed:
            raise ProofV3VerificationError("proof-v3 token event is empty")
        if len(self._token_ids) + len(parsed) > self._max_output_tokens:
            raise ProofV3VerificationError(
                "proof-v3 output token sequence exceeds the signed profile"
            )
        self._text_parts.append(text)
        self._token_ids.extend(parsed)
        if self._first_token_ns is None:
            self._first_token_ns = received_monotonic_ns
        self._last_token_ns = received_monotonic_ns

    def record_stream_timing(
        self,
        *,
        request_started_wall: float,
        request_started_ns: int,
        stream_finished_wall: float,
        stream_finished_ns: int,
    ) -> None:
        """Record validator-observed timing for the inference stream only."""

        if (
            self._request_started_ns is not None
            or request_started_ns <= 0
            or stream_finished_ns < request_started_ns
            or request_started_wall <= 0
            or stream_finished_wall < request_started_wall
        ):
            self.fail_closed()
            raise ProofV3UnavailableError(
                "proof-v3 validator stream timing is malformed or duplicated"
            )
        self._request_started_wall = request_started_wall
        self._request_started_ns = request_started_ns
        self._stream_finished_wall = stream_finished_wall
        self._stream_finished_ns = stream_finished_ns
        if self._observed_output is not None and self._decision is None:
            # Tier selection is validator-local and must start only after the
            # canonical SSE stream has reached EOF.  Starting the bounded
            # local reveal clock at the earlier ``done`` event allowed stream
            # teardown or worker rescheduling to consume that clock and made
            # an honest miner appear late before it had even received a nonce.
            try:
                self._decision = self.session.select_audit_tier_once(
                    selected_monotonic_ns=stream_finished_ns,
                )
            except ProofV3Error as exc:
                self.fail_closed()
                raise ProofV3UnavailableError(
                    "proof-v3 validator could not select the post-stream "
                    "audit tier"
                ) from exc

    def _observe_precommit(
        self, data: Mapping[str, object], received_monotonic_ns: int
    ) -> None:
        if self._precommit_bytes is not None:
            raise ProofV3VerificationError(
                "proof-v3 stream contains a duplicate precommit"
            )
        if self._observed_output is not None or self._last_token_ns is None:
            raise ProofV3VerificationError(
                "proof-v3 precommit has invalid stream ordering"
            )
        if data.get("proof_protocol_version") != PROOF_PROTOCOL_V3:
            raise ProofV3VerificationError(
                "proof-v3 precommit has an unexpected protocol version"
            )
        if (
            data.get("proof_challenge_id")
            != self.session.precommit_context.proof_challenge_id.hex()
        ):
            raise ProofV3VerificationError(
                "proof-v3 precommit has an unexpected challenge id"
            )
        encoded_hex = data.get("commitment_envelope")
        if not isinstance(encoded_hex, str) or len(encoded_hex) % 2:
            raise ProofV3VerificationError(
                "proof-v3 precommit envelope is not canonical hexadecimal"
            )
        try:
            encoded = bytes.fromhex(encoded_hex)
        except ValueError as exc:
            raise ProofV3VerificationError(
                "proof-v3 precommit envelope is not hexadecimal"
            ) from exc
        if encoded.hex() != encoded_hex:
            raise ProofV3VerificationError(
                "proof-v3 precommit envelope is not canonical hexadecimal"
            )
        envelope = commitment_envelope_from_bytes(encoded)
        if envelope.canonical_bytes() != encoded:
            raise ProofV3VerificationError(
                "proof-v3 precommit envelope is not canonical"
            )
        self._precommit_bytes = encoded
        self._precommit_received_ns = received_monotonic_ns

    def _observe_done(
        self, data: Mapping[str, object], received_monotonic_ns: int
    ) -> None:
        if self._observed_output is not None:
            raise ProofV3VerificationError(
                "proof-v3 stream contains a duplicate done event"
            )
        if (
            self._precommit_bytes is None
            or self._precommit_received_ns is None
            or self._last_token_ns is None
        ):
            raise ProofV3VerificationError(
                "proof-v3 done event arrived before token/precommit"
            )
        if data.get("proof_protocol_version") != PROOF_PROTOCOL_V3:
            raise ProofV3VerificationError(
                "proof-v3 done event has an unexpected protocol version"
            )
        text = "".join(self._text_parts)
        if data.get("output_text") != text:
            raise ProofV3VerificationError(
                "proof-v3 done text differs from the observed token stream"
            )
        if data.get("input_tokens") != self.session.precommit_context.context_token_count:
            raise ProofV3VerificationError(
                "proof-v3 done prompt length differs from the request"
            )
        if data.get("output_tokens") != len(self._token_ids):
            raise ProofV3VerificationError(
                "proof-v3 done token count differs from the observed stream"
            )
        finish_reason = data.get("finish_reason")
        if not isinstance(finish_reason, str):
            raise ProofV3VerificationError(
                "proof-v3 done finish reason is malformed"
            )
        observed = ObservedExecutionOutputV3(
            output_token_ids=tuple(self._token_ids),
            emitted_text_utf8=text.encode("utf-8"),
            finish_reason=finish_reason,
        )
        self.session.accept_precommit_bytes(
            encoded_envelope=self._precommit_bytes,
            observed_output=observed,
            last_visible_token_monotonic_ns=self._last_token_ns,
            received_monotonic_ns=self._precommit_received_ns,
        )
        self._observed_output = observed

    def observe_sse_event(
        self,
        *,
        event_type: str,
        data: object,
        received_monotonic_ns: int,
    ) -> None:
        """Consume one miner SSE event and fail the session on any mismatch."""

        try:
            if not isinstance(event_type, str):
                raise ProofV3VerificationError(
                    "proof-v3 SSE event type is malformed"
                )
            payload = self._require_dict(data)
            if event_type == "token":
                self._observe_token(payload, received_monotonic_ns)
            elif event_type == "proof_precommit":
                self._observe_precommit(payload, received_monotonic_ns)
            elif event_type == "done":
                self._observe_done(payload, received_monotonic_ns)
            elif event_type == "error":
                logger.warning(
                    "Proof-v3 miner emitted an error event: %s",
                    _peer_error_summary(payload),
                )
                raise ProofV3PeerServiceFailure(
                    "miner returned an error during proof-v3 inference"
                )
            else:
                raise ProofV3VerificationError(
                    "proof-v3 stream contains an unexpected event"
                )
        except Exception:
            self.fail_closed()
            raise

    def require_stream_complete(self) -> PostCommitAuditDecisionV3:
        if self._observed_output is None:
            self.fail_closed()
            raise ProofV3VerificationError(
                "proof-v3 stream ended before its canonical done event"
            )
        if self._decision is None:
            self.fail_closed()
            raise ProofV3UnavailableError(
                "proof-v3 validator did not select a post-stream audit tier"
            )
        return self._decision

    def retention_request_fields(self, *, action: str) -> dict[str, object]:
        """Build a post-precommit hold or light-release request."""

        decision = self.require_stream_complete()
        if action == "hold":
            hold_budget_ns = self.session.nonce_reveal_hold_budget_ns
            if hold_budget_ns is None:
                raise ProofV3UnavailableError(
                    "proof-v3 exchange has no precommit hold budget"
                )
        elif action == "release":
            if decision.hard_audit_selected:
                raise ProofV3UnavailableError(
                    "proof-v3 hard exchange cannot use light release"
                )
            hold_budget_ns = 0
        else:
            raise ProofV3UnavailableError(
                "proof-v3 retention action is unsupported"
            )
        return {
            "action": action,
            "proof_challenge_id": (
                self.session.precommit_context.proof_challenge_id.hex()
            ),
            "commitment_envelope_digest": (
                decision.commitment_envelope_digest.hex()
            ),
            "hold_budget_ns": hold_budget_ns,
        }

    def reveal_hard_nonce(self, *, revealed_monotonic_ns: int) -> bytes:
        try:
            reveal = self.session.reveal_nonce_once(
                revealed_monotonic_ns=revealed_monotonic_ns,
            )
        except ProofV3Error as exc:
            # Nothing has been disclosed to the miner yet.  Selection,
            # serialization and the call into this method are validator-local
            # work, so failure here must never become a peer proof strike.
            self.fail_closed()
            raise ProofV3UnavailableError(
                "proof-v3 validator missed its local nonce-reveal window"
            ) from exc
        except Exception:
            self.fail_closed()
            raise
        self._reveal_ns = revealed_monotonic_ns
        encoded = reveal.canonical_bytes()
        self._nonce_reveal_bytes = encoded
        return encoded

    def _failed_hard_bundle_bytes(self, *, encoded_proof: bytes) -> bytes:
        """Build a replayable bundle from validator-owned exchange state."""

        from verallm.proof_v3.hard_bundle import RetainedHardProofBundleV3
        from verallm.proof_v3.session import NonceRevealV3

        if (
            self._precommit_bytes is None
            or self._observed_output is None
            or self._nonce_reveal_bytes is None
        ):
            raise ProofV3VerificationError(
                "failed proof-v3 exchange lacks replay context"
            )
        bundle = RetainedHardProofBundleV3(
            precommit_context=self.session.precommit_context,
            prompt_token_ids=tuple(
                self.validator_request_context.prompt_token_ids
            ),
            observed_output=self._observed_output,
            envelope=commitment_envelope_from_bytes(self._precommit_bytes),
            nonce_reveal=NonceRevealV3.from_canonical_bytes(
                self._nonce_reveal_bytes
            ),
            encoded_proof=encoded_proof,
        )
        return bundle.canonical_bytes()

    def verify_hard_proof(
        self,
        *,
        encoded_proof: bytes,
        received_monotonic_ns: int,
    ) -> object:
        if self._observed_output is None:
            self.fail_closed()
            raise ProofV3VerificationError(
                "proof-v3 proof arrived before the output was observed"
            )
        started = time.monotonic_ns()
        try:
            result = self.session.verify_proof_bytes_once(
                encoded_proof=encoded_proof,
                received_monotonic_ns=received_monotonic_ns,
                validator_request_context=self.validator_request_context,
                observed_output=self._observed_output,
            )
        except Exception:
            try:
                retained = _retain_failed_hard_bundle(
                    self._failed_hard_bundle_bytes(
                        encoded_proof=encoded_proof,
                    )
                )
                logger.warning(
                    "Retained failed proof-v3 hard bundle for deterministic "
                    "replay: %s",
                    retained,
                )
            except Exception as retention_error:
                # Retention is diagnostic only.  Never replace, suppress, or
                # alter the actual fail-closed verifier result.
                logger.warning(
                    "Failed proof-v3 hard bundle could not be retained: %s",
                    retention_error,
                )
            self.fail_closed()
            raise
        self._verification_ns = time.monotonic_ns() - started
        self._proof_received_ns = received_monotonic_ns
        self._proof_wire_bytes = len(encoded_proof)
        self._verification_result = result
        return result

    def result(self) -> ProofV3ExchangeResult:
        decision = self.require_stream_complete()
        hard = decision.hard_audit_selected
        if hard and self.session.state is not ChallengeSessionStateV3.VERIFIED:
            raise ProofV3UnavailableError(
                "proof-v3 hard exchange has not verified its proof"
            )
        if not hard and self.session.state is not ChallengeSessionStateV3.LIGHT_REVEALED:
            raise ProofV3UnavailableError(
                "proof-v3 light exchange has an unexpected state"
            )
        assert self._last_token_ns is not None
        assert self._first_token_ns is not None
        assert self._precommit_received_ns is not None
        assert self._observed_output is not None
        if self._request_started_ns is None or self._stream_finished_ns is None:
            raise ProofV3UnavailableError(
                "proof-v3 exchange lacks validator-observed stream timing"
            )
        ttft_ms = max(
            0.0,
            (self._first_token_ns - self._request_started_ns) / 1_000_000,
        )
        inference_ms = max(
            0.0,
            (self._last_token_ns - self._request_started_ns) / 1_000_000,
        )
        round_trip_ms = max(
            0.0,
            (self._stream_finished_ns - self._request_started_ns) / 1_000_000,
        )
        return ProofV3ExchangeResult(
            text="".join(self._text_parts),
            input_tokens=self.session.precommit_context.context_token_count,
            output_token_ids=tuple(self._token_ids),
            finish_reason=self._observed_output.finish_reason,
            commitment_envelope_digest=decision.commitment_envelope_digest,
            capture_chain_digest=(
                self.session.verified_capture_chain_digest or b""
            ),
            audit_decision=decision,
            proof_verified=hard,
            verification_result=self._verification_result,
            validator_request_start_ts=self._request_started_wall,
            validator_request_end_ts=self._stream_finished_wall,
            validator_request_ms=round_trip_ms,
            round_trip_ms=round_trip_ms,
            ttft_ms=ttft_ms,
            inference_ms=inference_ms,
            last_token_to_precommit_ms=max(
                0.0,
                (self._precommit_received_ns - self._last_token_ns) / 1_000_000,
            ),
            nonce_to_proof_ms=(
                None
                if self._reveal_ns is None or self._proof_received_ns is None
                else max(
                    0.0,
                    (self._proof_received_ns - self._reveal_ns) / 1_000_000,
                )
            ),
            verification_ms=(
                None
                if self._verification_ns is None
                else self._verification_ns / 1_000_000
            ),
            proof_wire_bytes=self._proof_wire_bytes,
        )


def _iter_sse_events(response) -> Iterator[tuple[str, Mapping[str, object]]]:
    event_type: str | None = None
    data_lines: list[str] = []
    data_size = 0

    def consume() -> tuple[str, Mapping[str, object]] | None:
        nonlocal event_type, data_lines, data_size
        if not data_lines:
            event_type = None
            data_size = 0
            return None
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ProofV3VerificationError(
                "proof-v3 SSE event is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProofV3VerificationError(
                "proof-v3 SSE event data is not an object"
            )
        selected = event_type or parsed.get("event")
        event_type = None
        data_lines = []
        data_size = 0
        if not isinstance(selected, str) or not selected:
            raise ProofV3VerificationError(
                "proof-v3 SSE event has no event type"
            )
        return selected, parsed

    for line in response.iter_lines():
        if not isinstance(line, str):
            raise ProofV3VerificationError("proof-v3 SSE line is malformed")
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            value = line[len("data:") :].strip()
            data_size += len(value.encode("utf-8"))
            if data_size > MAX_PROOF_V3_SSE_EVENT_BYTES:
                raise ProofV3VerificationError(
                    "proof-v3 SSE event exceeds its byte limit"
                )
            data_lines.append(value)
        elif line == "":
            item = consume()
            if item is not None:
                yield item
    item = consume()
    if item is not None:
        yield item


def _read_bounded_hard_proof(
    response,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    from verallm.proof_v3.economic_transport import (
        MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
        MAX_ECONOMIC_TRANSPORT_BYTES,
    )

    maximum = (
        MAX_ECONOMIC_TRANSPORT_BYTES
        if maximum_bytes is None
        else maximum_bytes
    )
    if maximum not in {
        MAX_ECONOMIC_TRANSPORT_BYTES,
        MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
    }:
        raise ProofV3VerificationError(
            "proof-v3 hard-proof transport bound is invalid"
        )
    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            declared_length = int(declared)
        except ValueError as exc:
            raise ProofV3VerificationError(
                "proof-v3 response content length is malformed"
            ) from exc
        if not 0 < declared_length <= maximum:
            raise ProofV3VerificationError(
                "proof-v3 response content length is out of range"
            )
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > maximum:
            raise ProofV3VerificationError(
                "proof-v3 hard proof exceeds its transport limit"
            )
        chunks.append(chunk)
    if size == 0:
        raise ProofV3VerificationError("proof-v3 hard proof is empty")
    if declared is not None and size != int(declared):
        raise ProofV3VerificationError(
            "proof-v3 response content length is inconsistent"
        )
    return b"".join(chunks)


async def _iter_sse_events_async(
    response,
) -> AsyncIterator[tuple[str, Mapping[str, object]]]:
    event_type: str | None = None
    data_lines: list[str] = []
    data_size = 0

    def consume() -> tuple[str, Mapping[str, object]] | None:
        nonlocal event_type, data_lines, data_size
        if not data_lines:
            event_type = None
            data_size = 0
            return None
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ProofV3VerificationError(
                "proof-v3 SSE event is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProofV3VerificationError(
                "proof-v3 SSE event data is not an object"
            )
        selected = event_type or parsed.get("event")
        event_type = None
        data_lines = []
        data_size = 0
        if not isinstance(selected, str) or not selected:
            raise ProofV3VerificationError(
                "proof-v3 SSE event has no event type"
            )
        return selected, parsed

    async for line in response.aiter_lines():
        if not isinstance(line, str):
            raise ProofV3VerificationError("proof-v3 SSE line is malformed")
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            value = line[len("data:") :].strip()
            data_size += len(value.encode("utf-8"))
            if data_size > MAX_PROOF_V3_SSE_EVENT_BYTES:
                raise ProofV3VerificationError(
                    "proof-v3 SSE event exceeds its byte limit"
                )
            data_lines.append(value)
        elif line == "":
            item = consume()
            if item is not None:
                yield item
    item = consume()
    if item is not None:
        yield item


async def _read_bounded_hard_proof_async(
    response,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    from verallm.proof_v3.economic_transport import (
        MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
        MAX_ECONOMIC_TRANSPORT_BYTES,
    )

    maximum = (
        MAX_ECONOMIC_TRANSPORT_BYTES
        if maximum_bytes is None
        else maximum_bytes
    )
    if maximum not in {
        MAX_ECONOMIC_TRANSPORT_BYTES,
        MAX_ECONOMIC_MOE_TRANSPORT_BYTES,
    }:
        raise ProofV3VerificationError(
            "proof-v3 hard-proof transport bound is invalid"
        )
    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            declared_length = int(declared)
        except ValueError as exc:
            raise ProofV3VerificationError(
                "proof-v3 response content length is malformed"
            ) from exc
        if not 0 < declared_length <= maximum:
            raise ProofV3VerificationError(
                "proof-v3 response content length is out of range"
            )
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > maximum:
            raise ProofV3VerificationError(
                "proof-v3 hard proof exceeds its transport limit"
            )
        chunks.append(chunk)
    if size == 0:
        raise ProofV3VerificationError("proof-v3 hard proof is empty")
    if declared is not None and size != int(declared):
        raise ProofV3VerificationError(
            "proof-v3 response content length is inconsistent"
        )
    return b"".join(chunks)


def _raise_exchange_failure(
    exchange: ProofV3ValidatorExchange,
    exc: Exception,
) -> None:
    exchange.fail_closed()
    if isinstance(exc, ProofV3UnavailableError):
        raise exc
    if exchange.hard_nonce_revealed:
        if isinstance(
            exc,
            (ProofV3Error, httpx.HTTPStatusError, httpx.TransportError),
        ):
            raise ProofV3PostcommitFailure(str(exc)) from exc
        raise ProofV3UnavailableError(
            "proof-v3 local verifier failed after nonce disclosure"
        ) from exc
    if isinstance(exc, ProofV3PeerServiceFailure):
        raise exc
    if isinstance(exc, ProofV3Error):
        raise ProofV3PeerFailure(str(exc)) from exc
    if (
        exchange.precommit_received
        and isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError))
    ):
        raise ProofV3PeerFailure(
            "proof-v3 peer abandoned the exchange after precommit"
        ) from exc
    if isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError)):
        raise exc
    raise ProofV3UnavailableError(
        "proof-v3 local exchange failed closed"
    ) from exc


def run_proof_v3_precommit_sync(
    *,
    client,
    miner_url: str,
    inference_path: str,
    request_body: Mapping[str, object],
    exchange: ProofV3ValidatorExchange,
    stream_callback: Callable[[str], Any] | None = None,
) -> ProofV3ValidatorExchange:
    """Run one synchronous v3 inference through accepted precommit only."""

    if not isinstance(exchange, ProofV3ValidatorExchange):
        raise TypeError("exchange has an unexpected type")
    if inference_path not in ("/inference", "/chat"):
        raise ValueError("proof-v3 inference path is unsupported")
    body: MutableMapping[str, object] = dict(request_body)
    fields = exchange.request_fields()
    for key, value in fields.items():
        if key in body and body[key] != value:
            raise ProofV3VerificationError(
                "proof-v3 request contains conflicting protocol fields"
            )
        body[key] = value
    base_url = miner_url.rstrip("/")
    request_started_wall = time.time()
    request_started_ns = time.monotonic_ns()
    try:
        with client.stream(
            "POST",
            f"{base_url}{inference_path}",
            json=dict(body),
        ) as response:
            if response.is_error:
                response.read()
            response.raise_for_status()
            for event_type, data in _iter_sse_events(response):
                received = time.monotonic_ns()
                exchange.observe_sse_event(
                    event_type=event_type,
                    data=data,
                    received_monotonic_ns=received,
                )
                if event_type == "token" and stream_callback is not None:
                    stream_callback(str(data.get("text", "")))
        exchange.record_stream_timing(
            request_started_wall=request_started_wall,
            request_started_ns=request_started_ns,
            stream_finished_wall=time.time(),
            stream_finished_ns=time.monotonic_ns(),
        )
        exchange.require_stream_complete()
        return exchange
    except Exception as exc:
        _raise_exchange_failure(exchange, exc)
        raise AssertionError("unreachable")


def hold_proof_v3_precommit_sync(
    *,
    client,
    miner_url: str,
    exchange: ProofV3ValidatorExchange,
) -> None:
    """Install one bounded post-precommit hold without revealing the tier."""

    if not isinstance(exchange, ProofV3ValidatorExchange):
        raise TypeError("exchange has an unexpected type")
    try:
        _post_retention_sync(
            client=client,
            url=f"{miner_url.rstrip('/')}{ECONOMIC_PROOF_V3_RETENTION_PATH}",
            fields=exchange.retention_request_fields(action="hold"),
        )
    except Exception as exc:
        _raise_exchange_failure(exchange, exc)


def _post_retention_sync(*, client, url: str, fields: Mapping[str, object]) -> None:
    """POST one idempotent retention operation with one transport retry."""

    for attempt in range(2):
        try:
            response = client.post(url, json=dict(fields))
            if response.is_error:
                response.read()
            response.raise_for_status()
            return
        except httpx.TransportError:
            if attempt == 0:
                continue
            raise


async def _post_retention_async(
    *,
    client,
    url: str,
    fields: Mapping[str, object],
) -> None:
    """Async counterpart of the bounded idempotent retention POST."""

    for attempt in range(2):
        try:
            response = await client.post(url, json=dict(fields))
            if response.is_error:
                await response.aread()
            response.raise_for_status()
            return
        except httpx.TransportError:
            if attempt == 0:
                continue
            raise


def finalize_proof_v3_exchange_sync(
    *,
    client,
    miner_url: str,
    exchange: ProofV3ValidatorExchange,
) -> ProofV3ExchangeResult:
    """Finalize one already accepted light or hard synchronous exchange."""

    if not isinstance(exchange, ProofV3ValidatorExchange):
        raise TypeError("exchange has an unexpected type")
    base_url = miner_url.rstrip("/")
    try:
        decision = exchange.require_stream_complete()
        if decision.hard_audit_selected:
            reveal_ns = time.monotonic_ns()
            reveal = exchange.reveal_hard_nonce(
                revealed_monotonic_ns=reveal_ns,
            )
            with client.stream(
                "POST",
                f"{base_url}{ECONOMIC_PROOF_V3_CHALLENGE_PATH}",
                json={"nonce_reveal": reveal.hex()},
                timeout=(
                    getattr(
                        getattr(exchange, "session", None),
                        "hard_proof_arrival_budget_ns",
                        540_000_000_000,
                    )
                    / 1_000_000_000
                    + 1.0
                ),
            ) as response:
                if response.is_error:
                    response.read()
                response.raise_for_status()
                media_type = response.headers.get("content-type", "").split(
                    ";", 1
                )[0].strip().lower()
                if media_type != ECONOMIC_PROOF_V3_MEDIA_TYPE:
                    raise ProofV3VerificationError(
                        "proof-v3 hard proof has an unexpected media type"
                    )
                encoded_proof = _read_bounded_hard_proof(
                    response,
                    maximum_bytes=getattr(
                        getattr(exchange, "session", None),
                        "maximum_hard_proof_transport_bytes",
                        None,
                    ),
                )
                received_ns = time.monotonic_ns()
            exchange.verify_hard_proof(
                encoded_proof=encoded_proof,
                received_monotonic_ns=received_ns,
            )
        else:
            _post_retention_sync(
                client=client,
                url=f"{base_url}{ECONOMIC_PROOF_V3_RETENTION_PATH}",
                fields=exchange.retention_request_fields(action="release"),
            )
        return exchange.result()
    except Exception as exc:
        _raise_exchange_failure(exchange, exc)
        raise AssertionError("unreachable")


def run_proof_v3_exchange_sync(
    *,
    client,
    miner_url: str,
    inference_path: str,
    request_body: Mapping[str, object],
    exchange: ProofV3ValidatorExchange,
    stream_callback: Callable[[str], Any] | None = None,
) -> ProofV3ExchangeResult:
    """Run one synchronous v3 SSE + conditional hard-opening exchange."""

    run_proof_v3_precommit_sync(
        client=client,
        miner_url=miner_url,
        inference_path=inference_path,
        request_body=request_body,
        exchange=exchange,
        stream_callback=stream_callback,
    )
    return finalize_proof_v3_exchange_sync(
        client=client,
        miner_url=miner_url,
        exchange=exchange,
    )


async def run_proof_v3_exchange_async(
    *,
    client,
    miner_url: str,
    inference_path: str,
    request_body: Mapping[str, object],
    exchange: ProofV3ValidatorExchange,
    stream_callback: Callable[[str], Any] | None = None,
    first_token_timeout_seconds: float | None = None,
) -> ProofV3ExchangeResult:
    """Run one asynchronous v3 SSE + conditional hard-opening exchange."""

    if not isinstance(exchange, ProofV3ValidatorExchange):
        raise TypeError("exchange has an unexpected type")
    if inference_path not in ("/inference", "/chat"):
        raise ValueError("proof-v3 inference path is unsupported")
    if (
        first_token_timeout_seconds is not None
        and (
            isinstance(first_token_timeout_seconds, bool)
            or not isinstance(first_token_timeout_seconds, (int, float))
            or first_token_timeout_seconds <= 0
        )
    ):
        raise ValueError("first-token timeout must be positive")
    body: MutableMapping[str, object] = dict(request_body)
    fields = exchange.request_fields()
    for key, value in fields.items():
        if key in body and body[key] != value:
            raise ProofV3VerificationError(
                "proof-v3 request contains conflicting protocol fields"
            )
        body[key] = value
    base_url = miner_url.rstrip("/")
    request_started_wall = time.time()
    request_started_ns = time.monotonic_ns()
    try:
        async with client.stream(
            "POST",
            f"{base_url}{inference_path}",
            json=dict(body),
        ) as response:
            if response.is_error:
                await response.aread()
            response.raise_for_status()
            events = _iter_sse_events_async(response)
            first_token_seen = False
            first_token_deadline = (
                asyncio.get_running_loop().time()
                + float(first_token_timeout_seconds)
                if first_token_timeout_seconds is not None
                else None
            )
            try:
                while True:
                    try:
                        if not first_token_seen and first_token_deadline is not None:
                            remaining = (
                                first_token_deadline
                                - asyncio.get_running_loop().time()
                            )
                            if remaining <= 0:
                                raise asyncio.TimeoutError
                            event_type, data = await asyncio.wait_for(
                                anext(events),
                                timeout=remaining,
                            )
                        else:
                            event_type, data = await anext(events)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as exc:
                        raise ProofV3FirstTokenTimeout(
                            "miner produced no token before the routing deadline"
                        ) from exc

                    received = time.monotonic_ns()
                    if event_type in ("proof_precommit", "done"):
                        # Canonical envelope parsing and the complete done-event
                        # admission path are CPU work.  Running them on the
                        # shared asyncio loop serializes otherwise-independent
                        # streams and can make already-arrived precommits appear
                        # late under a wide co-batch.  Timestamp receipt first,
                        # then validate off-loop; ordering within this exchange
                        # remains strict because this coroutine awaits each call.
                        await asyncio.to_thread(
                            exchange.observe_sse_event,
                            event_type=event_type,
                            data=data,
                            received_monotonic_ns=received,
                        )
                    else:
                        exchange.observe_sse_event(
                            event_type=event_type,
                            data=data,
                            received_monotonic_ns=received,
                        )
                    if event_type == "token":
                        first_token_seen = True
                        if stream_callback is not None:
                            callback_result = stream_callback(
                                str(data.get("text", ""))
                            )
                            if inspect.isawaitable(callback_result):
                                await callback_result
            finally:
                await events.aclose()
        exchange.record_stream_timing(
            request_started_wall=request_started_wall,
            request_started_ns=request_started_ns,
            stream_finished_wall=time.time(),
            stream_finished_ns=time.monotonic_ns(),
        )
        decision = exchange.require_stream_complete()
        if decision.hard_audit_selected:
            reveal_ns = time.monotonic_ns()
            reveal = exchange.reveal_hard_nonce(
                revealed_monotonic_ns=reveal_ns,
            )
            async with client.stream(
                "POST",
                f"{base_url}{ECONOMIC_PROOF_V3_CHALLENGE_PATH}",
                json={"nonce_reveal": reveal.hex()},
                timeout=(
                    getattr(
                        getattr(exchange, "session", None),
                        "hard_proof_arrival_budget_ns",
                        540_000_000_000,
                    )
                    / 1_000_000_000
                    + 1.0
                ),
            ) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                media_type = response.headers.get("content-type", "").split(
                    ";", 1
                )[0].strip().lower()
                if media_type != ECONOMIC_PROOF_V3_MEDIA_TYPE:
                    raise ProofV3VerificationError(
                        "proof-v3 hard proof has an unexpected media type"
                    )
                encoded_proof = await _read_bounded_hard_proof_async(
                    response,
                    maximum_bytes=getattr(
                        getattr(exchange, "session", None),
                        "maximum_hard_proof_transport_bytes",
                        None,
                    ),
                )
                received_ns = time.monotonic_ns()
            exchange.verify_hard_proof(
                encoded_proof=encoded_proof,
                received_monotonic_ns=received_ns,
            )
        else:
            await _post_retention_async(
                client=client,
                url=f"{base_url}{ECONOMIC_PROOF_V3_RETENTION_PATH}",
                fields=exchange.retention_request_fields(action="release"),
            )
        return exchange.result()
    except Exception as exc:
        exchange.fail_closed()
        if isinstance(exc, ProofV3UnavailableError):
            raise
        if exchange.hard_nonce_revealed:
            if isinstance(
                exc,
                (ProofV3Error, httpx.HTTPStatusError, httpx.TransportError),
            ):
                raise ProofV3PostcommitFailure(str(exc)) from exc
            raise ProofV3UnavailableError(
                "proof-v3 local verifier failed after nonce disclosure"
            ) from exc
        if isinstance(exc, ProofV3PeerServiceFailure):
            raise
        if isinstance(exc, ProofV3Error):
            raise ProofV3PeerFailure(str(exc)) from exc
        if (
            exchange.precommit_received
            and isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError))
        ):
            raise ProofV3PeerFailure(
                "proof-v3 peer abandoned the exchange after precommit"
            ) from exc
        if isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError)):
            raise
        raise ProofV3UnavailableError(
            "proof-v3 local exchange failed closed"
        ) from exc


__all__ = [
    "MAX_PROOF_V3_SSE_EVENT_BYTES",
    "ProofV3ExchangeResult",
    "ProofV3FirstTokenTimeout",
    "ProofV3PeerFailure",
    "ProofV3PeerServiceFailure",
    "ProofV3PostcommitFailure",
    "ProofV3ValidatorExchange",
    "finalize_proof_v3_exchange_sync",
    "hold_proof_v3_precommit_sync",
    "run_proof_v3_precommit_sync",
    "run_proof_v3_exchange_async",
    "run_proof_v3_exchange_sync",
]
