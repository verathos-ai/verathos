"""Validator-side private-mesh canary execution and verification.

The active canary path talks only to the coordinator registered on chain.  It
binds each request to a validator-pinned, endpoint-free verification snapshot,
authenticates the exact HTTP body with the validator hotkey, and locally
verifies the returned proof artifact and coordinator signature.

The active canary protocol is two-phase.  The signed inference request carries
only a commitment to a secret validator challenge.  After the coordinator has
returned and signed the response-bound origin receipt, the validator reveals
the challenge in a separately signed postcommit request and verifies the final
proof artifact against that frozen origin.

The deferred-audit helper at the bottom of this module is retained only as an
experimental compatibility surface.  It is not selected by
``run_mesh_canary``.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from neurons import request_signing
from verallm.mesh.proof import mesh_validator_challenge_nonce_commitment
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationSnapshot,
    verify_mesh_verification_snapshot_signature,
)


MESH_INFERENCE_PATH = "/v1/mesh/inference"
# Canaries deliberately use the same path and the same streaming shape as
# organic user traffic.  A canary that posted to /v1/mesh/inference with
# stream=false while real users streamed to /v1/chat/completions was separable
# on the URL alone, before the coordinator parsed a body, which let it serve
# canaries honestly and cheat the endpoint carrying the actual traffic.
MESH_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MESH_POSTCOMMIT_AUDIT_PATH = "/v1/mesh/proof/postcommit-audit"
MAX_MESH_ARTIFACT_RESPONSE_BYTES = 128 * 1024 * 1024
MESH_POSTCOMMIT_MAX_ATTEMPTS = 3
MESH_POSTCOMMIT_RETRY_DELAYS = (0.5, 1.0)
MESH_POSTCOMMIT_FINALIZATION_ERROR_CODE = (
    "postcommit_finalization_in_progress"
)
MESH_POSTCOMMIT_CAPACITY_ERROR_CODE = "postcommit_capacity_unavailable"
MESH_POSTCOMMIT_POLLABLE_ERROR_CODES = frozenset(
    {
        MESH_POSTCOMMIT_FINALIZATION_ERROR_CODE,
        MESH_POSTCOMMIT_CAPACITY_ERROR_CODE,
    }
)
# The coordinator refused a request pinned to the snapshot this validator
# committed to for the epoch.  That is a refusal to be verified under the
# pinned terms rather than an outage, so it must never reach the retry or
# busy-forgiveness path no matter which status code carries it.
MESH_SNAPSHOT_MISMATCH_ERROR_CODE = "verification_snapshot_mismatch"
# Final proof generation is globally serialized and the coordinator reserves
# capacity for at most four admitted origins.  At the 120-second proof budget,
# a later honest origin can wait roughly 480 seconds.  Six hundred seconds
# covers that bounded queue with headroom while still preventing an unbounded
# validator thread/task hold.  The hard poll cap also covers the minimum
# accepted one-second Retry-After cadence across the whole window.
MESH_POSTCOMMIT_FINALIZATION_WINDOW_SECONDS = 600.0
MESH_POSTCOMMIT_FINALIZATION_MAX_POLLS = 640
MESH_POSTCOMMIT_FINALIZATION_POLL_DEFAULT_SECONDS = 2.0
MESH_POSTCOMMIT_FINALIZATION_POLL_MIN_SECONDS = 1.0
MESH_POSTCOMMIT_FINALIZATION_POLL_MAX_SECONDS = 5.0
MESH_HTTP_ERROR_METADATA_MAX_BYTES = 4096

# Experimental compatibility only.  The active validator no longer schedules
# deferred mesh audits.
MESH_DEFERRED_AUDIT_DELAY_BLOCKS = 5

MeshCanaryTransport = Callable[[str, bytes, Mapping[str, str], float], bytes]


class MeshCanaryTransportError(RuntimeError):
    """Typed coordinator transport/status failure used by validator retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
        phase: str = "inference",
        streamed_content: bool = False,
        error_code: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.phase = str(phase)
        self.streamed_content = bool(streamed_content)
        self.error_code = _normalize_mesh_error_code(error_code)
        self.retry_after_seconds = _bounded_mesh_retry_after(
            retry_after_seconds
        )


class MeshValidatorVerificationError(RuntimeError):
    """Local verifier/infrastructure failure that is not attributable to a miner."""

    def __init__(self, message: str, *, streamed_content: bool = False) -> None:
        super().__init__(message)
        self.streamed_content = bool(streamed_content)


def is_validator_verification_fault(exc: BaseException) -> bool:
    """Accept only an explicitly labelled trusted local-boundary failure.

    Native verifier exceptions and their cause chains may contain
    miner-influenced failure classes.  Import/executor boundaries must wrap
    genuine validator infrastructure faults in
    :class:`MeshValidatorVerificationError` themselves; no generic exception
    class or nested cause is implicitly forgiven here.
    """

    return type(exc) is MeshValidatorVerificationError


def _normalize_mesh_error_code(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        return ""
    if not value.isascii() or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in value
    ):
        return ""
    return value


def _bounded_mesh_retry_after(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return min(
        MESH_POSTCOMMIT_FINALIZATION_POLL_MAX_SECONDS,
        max(MESH_POSTCOMMIT_FINALIZATION_POLL_MIN_SECONDS, parsed),
    )


@dataclass
class MeshCanaryResult:
    """Outcome of one coordinator request and local proof verification."""

    ok: bool
    reason: str = ""
    # True only when no coordinator artifact was served (including an HTTP
    # status without an artifact). Malformed or unverifiable artifacts are
    # protocol failures, not transport errors.
    transport_error: bool = False
    transport_status_code: int | None = None
    transport_retryable: bool = False
    transport_phase: str = ""
    # A local verifier/import/executor failure is indeterminate.  Serving still
    # fails closed, but it must never become a miner proof strike.
    validator_error: bool = False
    # Retained for DB/result compatibility.  The active path always leaves it
    # false because canaries require inline proof.
    deferred_pending: bool = False
    full_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # None means unavailable, which now only happens when the stream carried no
    # visible content chunk before its terminal event.
    ttft_ms: float | None = None
    # Validator-observed phase-one latency through receipt of the frozen
    # response.  This excludes postcommit proof generation and verification.
    inference_ms: float = 0.0
    # Actual validator wall-clock interval enclosing the phase-one coordinator
    # request.  Keep this separate from inference_ms: the former is signed into
    # receipts while the latter is the monotonic duration used for throughput.
    phase_one_start_ts: float = 0.0
    phase_one_end_ts: float = 0.0
    total_ms: float = 0.0
    receipt: dict = field(default_factory=dict)
    artifact: dict = field(default_factory=dict)
    openai_request: dict = field(default_factory=dict)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _coordinator_route_url(endpoint: str, path: str) -> str:
    if not isinstance(endpoint, str):
        raise ValueError("mesh coordinator endpoint must be a string")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("mesh coordinator endpoint must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("mesh coordinator endpoint must not contain user information")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("mesh coordinator endpoint must be an origin without a path")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return origin + path


def _coordinator_inference_url(endpoint: str) -> str:
    return _coordinator_route_url(endpoint, MESH_INFERENCE_PATH)


def _coordinator_chat_url(endpoint: str) -> str:
    return _coordinator_route_url(endpoint, MESH_CHAT_COMPLETIONS_PATH)


def _coordinator_postcommit_url(endpoint: str) -> str:
    return _coordinator_route_url(endpoint, MESH_POSTCOMMIT_AUDIT_PATH)


def _default_mesh_canary_transport(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    phase = (
        "postcommit"
        if url.endswith(MESH_POSTCOMMIT_AUDIT_PATH)
        else "inference"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout,
        ) as response:
            payload = response.read(MAX_MESH_ARTIFACT_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        try:
            error_body = exc.read(MESH_HTTP_ERROR_METADATA_MAX_BYTES + 1)
        except Exception:
            error_body = b""
        error_code, retry_after_seconds = _mesh_http_error_metadata(
            error_body[:MESH_HTTP_ERROR_METADATA_MAX_BYTES],
            exc.headers,
        )
        raise MeshCanaryTransportError(
            f"mesh coordinator returned HTTP {status_code}",
            status_code=status_code,
            retryable=(
                (status_code in {408, 429, 503} or status_code >= 500)
                and error_code != MESH_SNAPSHOT_MISMATCH_ERROR_CODE
            ),
            phase=phase,
            error_code=error_code,
            retry_after_seconds=retry_after_seconds,
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise MeshCanaryTransportError(
            f"mesh coordinator transport failed: {type(exc).__name__}",
            retryable=True,
            phase=phase,
        ) from exc
    return payload


# Filled by the default streaming transport and drained by the caller in the
# same thread, immediately after the transport returns.  An out-parameter
# rather than a return value so the transport signature, and therefore every
# injected test transport, stays unchanged.  Thread-local because the validator
# runs canaries on a thread pool and a shared list would attribute one miner's
# first-token timing to another.
_canary_timing = threading.local()


def _canary_first_delta() -> list[float]:
    slot = getattr(_canary_timing, "first_delta", None)
    if slot is None:
        slot = []
        _canary_timing.first_delta = slot
    return slot


def _mesh_canary_stream_transport(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> bytes:
    """POST a streaming canary and return its terminal artifact as JSON bytes.

    The canary streams exactly like organic traffic, so the coordinator cannot
    separate the two on transport shape.  The terminal ``done`` event carries
    the same receipt and response the non-streaming route used to return in one
    piece, so this rebuilds that artifact and the rest of the canary path is
    unchanged.

    Byte budgets mirror the non-streaming reader: an untrusted coordinator must
    not be able to hold the validator open or exhaust it by streaming forever.
    """

    request = urllib.request.Request(
        url,
        data=body,
        headers={**dict(headers), "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout,
        ) as response:
            return _read_mesh_sse_artifact(
                response, first_delta=_canary_first_delta(),
            )
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        try:
            error_body = exc.read(MESH_HTTP_ERROR_METADATA_MAX_BYTES + 1)
        except Exception:
            error_body = b""
        error_code, retry_after_seconds = _mesh_http_error_metadata(
            error_body[:MESH_HTTP_ERROR_METADATA_MAX_BYTES],
            exc.headers,
        )
        raise MeshCanaryTransportError(
            f"mesh coordinator returned HTTP {status_code}",
            status_code=status_code,
            retryable=(
                (status_code in {408, 429, 503} or status_code >= 500)
                and error_code != MESH_SNAPSHOT_MISMATCH_ERROR_CODE
            ),
            phase="inference",
            error_code=error_code,
            retry_after_seconds=retry_after_seconds,
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise MeshCanaryTransportError(
            f"mesh coordinator transport failed: {type(exc).__name__}",
            retryable=True,
            phase="inference",
        ) from exc


def _read_mesh_sse_artifact(
    response,
    *,
    first_delta: list[float] | None = None,
) -> bytes:
    """Consume an SSE stream and return the terminal artifact as JSON bytes.

    ``first_delta`` collects the monotonic timestamp of the first visible
    content chunk, which is what time to first token actually measures.  It is
    an out-parameter so the transport keeps returning plain bytes and injected
    transports stay compatible.
    """

    consumed = 0
    terminal: dict[str, Any] | None = None
    while True:
        line = response.readline(MAX_MESH_ARTIFACT_RESPONSE_BYTES + 1)
        if not line:
            break
        consumed += len(line)
        if consumed > MAX_MESH_ARTIFACT_RESPONSE_BYTES:
            raise MeshCanaryTransportError(
                "mesh canary stream exceeded the artifact byte limit",
                retryable=False,
                phase="inference",
                streamed_content=True,
            )
        if not line.startswith(b"data:"):
            continue
        raw = line[len(b"data:"):].strip()
        if not raw or raw == b"[DONE]":
            continue
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value is forbidden: {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            # Delta chunks are not the validator's business; only the terminal
            # event is. A malformed delta is still a protocol failure though,
            # because the coordinator controls the whole stream.
            raise MeshCanaryTransportError(
                "mesh canary stream contained a malformed event",
                retryable=False,
                phase="inference",
                streamed_content=True,
            )
        if isinstance(payload, dict) and payload.get("event") == "error":
            # The coordinator reports failures that occur AFTER the 200 +
            # SSE headers are on the wire as an in-stream error event
            # (worker _send_mesh_failure). Silently skipping it used to
            # collapse every such verdict into "ended without a terminal
            # artifact" (non-retryable) — including the RETRYABLE
            # slots_busy verdict a validator-lane stream receives when an
            # exclusive replay window does not drain within its wait
            # budget. An audit-window collision is availability, never
            # eligibility.
            error_code = str(
                payload.get("error_code") or payload.get("type") or ""
            )
            message = str(
                payload.get("error") or "mesh canary stream reported an error"
            )
            if (
                str(payload.get("type") or "") == "slots_busy"
                or payload.get("retryable") is True
            ):
                # Surface exactly like the pre-stream slots_busy 503 so
                # the ordinary busy-skip / forgiveness machinery prices
                # it. Busy is raised while arming capture, before any
                # content token, so the retry is pre-first-byte.
                raise MeshCanaryTransportError(
                    message,
                    status_code=503,
                    retryable=True,
                    phase="inference",
                    streamed_content=bool(first_delta),
                    error_code=error_code,
                )
            raise MeshCanaryTransportError(
                message,
                retryable=False,
                phase="inference",
                streamed_content=True,
                error_code=error_code,
            )
        if not isinstance(payload, dict) or payload.get("event") != "done":
            if first_delta is not None and not first_delta:
                choices = payload.get("choices") if isinstance(payload, dict) else None
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if isinstance(delta, dict) and delta.get("content"):
                        first_delta.append(time.monotonic())
            continue
        if terminal is not None:
            raise MeshCanaryTransportError(
                "mesh canary stream returned multiple terminal artifacts",
                retryable=False,
                phase="inference",
                streamed_content=True,
            )
        terminal = payload
    if terminal is None:
        raise MeshCanaryTransportError(
            "mesh canary stream ended without a terminal artifact",
            retryable=False,
            phase="inference",
            streamed_content=True,
        )
    metadata = terminal.get("verathos_mesh")
    response_body = terminal.get("response")
    if not isinstance(metadata, dict) or not isinstance(response_body, dict):
        raise MeshCanaryTransportError(
            "mesh canary terminal event is missing its artifact",
            retryable=False,
            phase="inference",
            streamed_content=True,
        )
    artifact = dict(metadata)
    artifact["response"] = response_body
    return json.dumps(artifact, sort_keys=True).encode("utf-8")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field in mesh artifact: {key}")
        result[key] = value
    return result


def _mesh_http_error_metadata(
    raw_body: bytes,
    headers: Mapping[str, Any] | None,
) -> tuple[str, float | None]:
    """Extract bounded retry metadata from an untrusted HTTP error."""

    error_code = ""
    if isinstance(raw_body, bytes) and len(raw_body) <= (
        MESH_HTTP_ERROR_METADATA_MAX_BYTES
    ):
        try:
            payload = json.loads(
                raw_body.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(
                        f"non-finite JSON value is forbidden: {value}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            error_code = _normalize_mesh_error_code(payload.get("error_code"))

    retry_after_raw: Any = None
    if headers is not None:
        try:
            retry_after_raw = headers.get("Retry-After")
            if retry_after_raw is None:
                retry_after_raw = headers.get("retry-after")
        except Exception:
            retry_after_raw = None
    if isinstance(retry_after_raw, str):
        try:
            retry_after_raw = float(retry_after_raw.strip())
        except ValueError:
            retry_after_raw = None
    return error_code, _bounded_mesh_retry_after(retry_after_raw)


def _parse_artifact_response(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise ValueError("mesh canary transport must return bytes")
    if len(raw) > MAX_MESH_ARTIFACT_RESPONSE_BYTES:
        raise ValueError("mesh inference artifact response is too large")
    try:
        artifact = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value is forbidden: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("mesh inference response is not valid JSON") from exc
    if not isinstance(artifact, dict):
        raise ValueError("mesh inference response must be a JSON object")
    return artifact


def _require_lower_hex_digest(value: Any, *, field_name: str) -> str:
    raw = str(value or "")
    if (
        len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return raw


def _proof_bound_mesh_token_counts(
    response: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    require_response_usage: bool = False,
) -> tuple[int, int]:
    """Return proof-bound counts and reject conflicting response metadata.

    The final receipt's token counts are bound to the verified token-id
    artifacts.  OpenAI ``usage`` is coordinator-controlled metadata and must
    never replace those counts for receipts, analytics, or scoring.  It may be
    omitted unless a caller explicitly requires it, but when supplied it must
    be an exact, non-negative JSON-integer restatement of the bound counts.
    """

    if not isinstance(response, Mapping):
        raise ValueError("mesh response must be an object")
    if not isinstance(receipt, Mapping):
        raise ValueError("mesh final receipt must be an object")

    bound: dict[str, int] = {}
    for field in ("prompt_token_count", "completion_token_count"):
        value = receipt.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"mesh final receipt {field} must be a non-negative JSON integer"
            )
        bound[field] = value

    if "usage" not in response:
        if require_response_usage:
            raise ValueError("mesh response usage is required")
        return bound["prompt_token_count"], bound["completion_token_count"]

    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("mesh response usage must be an object")
    usage_bindings = (
        ("prompt_tokens", "prompt_token_count"),
        ("completion_tokens", "completion_token_count"),
    )
    for usage_field, receipt_field in usage_bindings:
        value = usage.get(usage_field)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"mesh response usage.{usage_field} must be a non-negative "
                "JSON integer"
            )
        if value != bound[receipt_field]:
            raise ValueError(
                f"mesh response usage.{usage_field} does not match "
                f"proof-bound {receipt_field}"
            )
    if "total_tokens" in usage:
        total_tokens = usage.get("total_tokens")
        if type(total_tokens) is not int or total_tokens < 0:
            raise ValueError(
                "mesh response usage.total_tokens must be a non-negative "
                "JSON integer"
            )
        if total_tokens != (
            bound["prompt_token_count"] + bound["completion_token_count"]
        ):
            raise ValueError(
                "mesh response usage.total_tokens does not match "
                "proof-bound token counts"
            )
    return bound["prompt_token_count"], bound["completion_token_count"]


def _validate_canary_snapshot(
    snapshot: MeshVerificationSnapshot,
    expected_coordinator: MeshCoordinatorIdentity,
    *,
    now_unix: int,
) -> None:
    if not isinstance(snapshot, MeshVerificationSnapshot):
        raise ValueError(
            "verification_snapshot must be a fetched MeshVerificationSnapshot"
        )
    if not isinstance(expected_coordinator, MeshCoordinatorIdentity):
        raise ValueError("expected_coordinator must be MeshCoordinatorIdentity")
    expected_coordinator.validate()
    snapshot.validate(require_signature=True)
    snapshot.validate_expected_bindings(expected_coordinator=expected_coordinator)
    snapshot.validate_freshness(now_unix=now_unix)
    if not verify_mesh_verification_snapshot_signature(
        snapshot,
        expected_hotkey=expected_coordinator.coordinator_hotkey,
        expected_epoch=snapshot.epoch,
        expected_mesh_id=snapshot.mesh_id,
        expected_generation=snapshot.generation,
        expected_coordinator=expected_coordinator,
        expected_model=snapshot.model,
        expected_policy=snapshot.policy,
        now_unix=now_unix,
    ):
        raise ValueError("mesh verification snapshot signature is invalid")
    if snapshot.policy.base_proof_sample_bps != 10_000:
        raise ValueError("mesh canaries require base proof on every response")
    # Decode rate 0 or full, uniform on both paths: the light tier carries
    # no decode obligation, and decode
    # verification lives in hard draws, which the canary scheduler forces
    # on its own cadence. Requiring the full decode rate here made every
    # canary indeterminate the moment the rate moved to zero (live: two
    # consecutive all-burn weight closes with a healthy serving fleet).
    if snapshot.policy.canary_decode_sample_bps not in (0, 10_000):
        raise ValueError("mesh canary decode rate must be zero or full")
    if (
        snapshot.policy.organic_decode_sample_bps
        != snapshot.policy.canary_decode_sample_bps
    ):
        raise ValueError(
            "mesh decode audit rate must be identical for organic and canary "
            "traffic so canaries are not identifiable at phase one"
        )
    if snapshot.policy.deferred_proof_enabled:
        raise ValueError("active mesh verification snapshot enables deferred proof")
    if snapshot.policy.challenge_scheme != "validator_postcommit_v1":
        raise ValueError(
            "mesh canaries require validator_postcommit_v1 challenge scheme"
        )


def _validate_canary_inputs(
    *,
    model_id: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    timeout: float,
    validator_hotkey_ss58: str,
    validator_hotkey_seed: bytes,
) -> None:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite non-negative number")
    if not math.isfinite(float(temperature)) or float(temperature) != 0.0:
        raise ValueError(
            "mesh proof canaries require temperature=0 for canonical replay"
        )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError("timeout must be a positive finite number")
    if not isinstance(validator_hotkey_ss58, str) or not validator_hotkey_ss58:
        raise ValueError("validator_hotkey_ss58 must be a non-empty string")
    if not isinstance(validator_hotkey_seed, bytes) or len(validator_hotkey_seed) < 32:
        raise ValueError("validator_hotkey_seed must contain at least 32 bytes")


def run_mesh_canary(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    timeout: float = 300.0,
    verify_artifact_fn=None,
    deferred: bool = False,
    *,
    verification_snapshot: MeshVerificationSnapshot,
    expected_coordinator: MeshCoordinatorIdentity,
    validator_hotkey_ss58: str,
    validator_hotkey_seed: bytes,
    transport: MeshCanaryTransport | None = None,
    nonce_factory: Callable[[], str] | None = None,
    request_id_factory: Callable[[], str] | None = None,
    clock: Callable[[], float] = time.time,
    wall_clock: Callable[[], float] | None = None,
    verify_origin_artifact_fn=None,
    retry_sleep: Callable[[float], None] = time.sleep,
    postcommit_clock: Callable[[], float] | None = None,
    audit_tier: str = "",
) -> MeshCanaryResult:
    """Run a proof-bearing canary against one registered coordinator.

    The snapshot must already have been fetched and pinned from the
    coordinator's authenticated snapshot route.  It is revalidated here and
    bound to the independently expected coordinator identity before any
    network request is made.  Private worker endpoints and the runtime
    ``MeshSpec`` are neither requested nor accepted by this client.
    """

    if deferred:
        raise ValueError(
            "deferred mesh canaries are experimental and disabled in the active path"
        )
    if audit_tier not in ("", "hard"):
        raise ValueError("mesh canary audit_tier must be '' or 'hard'")
    _validate_canary_inputs(
        model_id=model_id,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        timeout=timeout,
        validator_hotkey_ss58=validator_hotkey_ss58,
        validator_hotkey_seed=validator_hotkey_seed,
    )
    now = clock()
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("clock must return a Unix timestamp")
    now_float = float(now)
    if not math.isfinite(now_float) or now_float <= 0:
        raise ValueError("clock must return a positive finite Unix timestamp")
    _validate_canary_snapshot(
        verification_snapshot,
        expected_coordinator,
        now_unix=int(now_float),
    )
    if model_id != verification_snapshot.model.model_id:
        raise ValueError("canary model_id does not match verification snapshot")

    nonce = (nonce_factory or (lambda: secrets.token_hex(32)))()
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise ValueError("validator nonce must be 32 bytes of lowercase hex")
    validator_request_id = (
        request_id_factory or (lambda: secrets.token_hex(32))
    )()
    if (
        not isinstance(validator_request_id, str)
        or len(validator_request_id) != 64
        or any(
            character not in "0123456789abcdef"
            for character in validator_request_id
        )
    ):
        raise ValueError(
            "validator request id must be 32 bytes of lowercase hex"
        )

    snapshot_hash = verification_snapshot.snapshot_hash_hex()
    challenge_commitment = mesh_validator_challenge_nonce_commitment(
        nonce,
        validator_request_id=validator_request_id,
        verification_snapshot_hash=snapshot_hash,
    )
    openai_request: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        # The proxy puts do_sample on every organic request unconditionally,
        # so a request without it is a canary. The verified GGUF sampler
        # requires it false anyway, which is what the proxy sends for the
        # deterministic profile, so matching costs nothing.
        "do_sample": False,
        # Canaries stream because organic traffic streams. See
        # MESH_CHAT_COMPLETIONS_PATH: any difference here is a free signal that
        # separates canaries from the traffic a coordinator would rather cheat.
        "stream": True,
        "verathos": {
            "challenge_nonce_commitment": challenge_commitment,
            "validator_request_id": validator_request_id,
            # Must be byte-identical to what the proxy sends on organic
            # traffic, or this field alone identifies canaries at phase one.
            "decode_audit_bps": int(
                verification_snapshot.policy.decode_sample_bps
            ),
            "verification_snapshot_hash": snapshot_hash,
        },
    }
    # The chat route takes the OpenAI request as the body directly, with no
    # envelope, which is exactly what the proxy signs for organic streaming
    # traffic. Matching it byte for byte is the whole point.
    body_bytes = _canonical_json_bytes(openai_request)
    auth_headers = request_signing.sign_request(
        method="POST",
        path=MESH_CHAT_COMPLETIONS_PATH,
        body=body_bytes,
        hotkey_ss58=validator_hotkey_ss58,
        hotkey_seed=validator_hotkey_seed,
    )
    headers = {"Content-Type": "application/json", **auth_headers}
    url = _coordinator_chat_url(endpoint)

    phase_one_wall_clock = wall_clock or time.time

    def _read_phase_one_wall_clock() -> float:
        value = phase_one_wall_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("wall_clock must return a Unix timestamp")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(
                "wall_clock must return a positive finite Unix timestamp"
            )
        return parsed

    phase_one_start_ts = _read_phase_one_wall_clock()
    phase_one_end_ts = phase_one_start_ts

    def _phase_one_result(**kwargs: Any) -> MeshCanaryResult:
        return MeshCanaryResult(
            phase_one_start_ts=phase_one_start_ts,
            phase_one_end_ts=phase_one_end_ts,
            **kwargs,
        )

    start = time.monotonic()
    # Stale timing from this thread's previous canary must not leak into this
    # one, and an injected transport never fills it at all.
    _canary_first_delta().clear()
    try:
        raw_artifact = (transport or _mesh_canary_stream_transport)(
            url,
            body_bytes,
            headers,
            float(timeout),
        )
    except MeshCanaryTransportError as exc:
        phase_one_end_ts = _read_phase_one_wall_clock()
        return _phase_one_result(
            ok=False,
            reason=f"mesh inference request failed: {exc}",
            transport_error=True,
            transport_status_code=exc.status_code,
            transport_retryable=exc.retryable,
            transport_phase="inference",
            openai_request=openai_request,
        )
    except Exception as exc:
        phase_one_end_ts = _read_phase_one_wall_clock()
        return _phase_one_result(
            ok=False,
            reason=f"mesh inference request failed: {exc}",
            transport_error=True,
            transport_retryable=isinstance(
                exc,
                (TimeoutError, ConnectionError, OSError),
            ),
            transport_phase="inference",
            openai_request=openai_request,
        )
    response_received = time.monotonic()
    phase_one_end_ts = _read_phase_one_wall_clock()
    origin_ms = (response_received - start) * 1000.0
    # Canaries stream now, so time to first token is a real measurement rather
    # than the unavailable sentinel the scoring path used to receive.
    _first_delta = _canary_first_delta()
    canary_ttft_ms = (
        (_first_delta[0] - start) * 1000.0 if _first_delta else None
    )

    try:
        origin_artifact = _parse_artifact_response(raw_artifact)
    except Exception as exc:
        return _phase_one_result(
            ok=False,
            reason=f"mesh inference response invalid: {exc}",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            openai_request=openai_request,
        )

    origin_receipt = origin_artifact.get("receipt")
    response = origin_artifact.get("response")
    if not isinstance(origin_receipt, dict) or not isinstance(response, dict):
        return _phase_one_result(
            ok=False,
            reason="mesh artifact missing receipt or response",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            artifact=origin_artifact,
            openai_request=openai_request,
        )

    if str(origin_receipt.get("proof_validator_hotkey", "")) != str(
        validator_hotkey_ss58
    ):
        return _phase_one_result(
            ok=False,
            reason="mesh postcommit origin validator hotkey mismatch",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )

    if verify_origin_artifact_fn is None:
        try:
            from verallm.mesh.worker import verify_mesh_inference_artifact
        except Exception as exc:
            return _phase_one_result(
                ok=False,
                reason=f"mesh origin verifier unavailable: {exc}",
                validator_error=True,
                inference_ms=origin_ms,
                total_ms=origin_ms,
                receipt=origin_receipt,
                artifact=origin_artifact,
                openai_request=openai_request,
            )

        verify_origin_artifact_fn = verify_mesh_inference_artifact

    try:
        verified = bool(
            verify_origin_artifact_fn(
                origin_artifact,
                openai_request,
                require_configured_proof=True,
                require_cryptographic_proof=False,
                expected_coordinator_hotkey=(
                    expected_coordinator.coordinator_hotkey
                ),
                expected_coordinator_uid=expected_coordinator.coordinator_uid,
                expected_validator_hotkey=validator_hotkey_ss58,
                require_coordinator_signature=True,
                verification_snapshot=verification_snapshot,
                verification_snapshot_now_unix=int(now_float),
                require_validator_request_id=True,
            )
        )
    except Exception as exc:
        if is_validator_verification_fault(exc):
            return _phase_one_result(
                ok=False,
                reason=f"mesh origin verifier unavailable: {exc}",
                validator_error=True,
                inference_ms=origin_ms,
                total_ms=origin_ms,
                receipt=origin_receipt,
                artifact=origin_artifact,
                openai_request=openai_request,
            )
        return _phase_one_result(
            ok=False,
            reason=f"mesh origin verification error: {exc}",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )
    if not verified:
        return _phase_one_result(
            ok=False,
            reason="mesh origin verification failed",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )

    try:
        origin_receipt_hash = _require_lower_hex_digest(
            origin_receipt.get("receipt_hash"),
            field_name="origin receipt_hash",
        )
        response_commitment_hash = _require_lower_hex_digest(
            origin_receipt.get("mesh_response_commitment_hash"),
            field_name="origin mesh_response_commitment_hash",
        )
        receipt_snapshot_hash = _require_lower_hex_digest(
            origin_receipt.get("verification_snapshot_hash"),
            field_name="origin verification_snapshot_hash",
        )
        if receipt_snapshot_hash != snapshot_hash:
            raise ValueError(
                "origin verification_snapshot_hash does not match pinned snapshot"
            )
    except Exception as exc:
        return _phase_one_result(
            ok=False,
            reason=f"mesh origin commitment invalid: {exc}",
            inference_ms=origin_ms,
            total_ms=origin_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )

    postcommit_body = {
        "validator_request_id": validator_request_id,
        "origin_receipt_hash": origin_receipt_hash,
        "mesh_response_commitment_hash": response_commitment_hash,
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": nonce,
    }
    if audit_tier:
        # The signed stricter-only tier demand: this reveal must resolve the
        # HARD relation regardless of the nonce-derived draw (canary hard
        # slots). Omitted on every other request so the demand itself is the
        # only distinguisher, and only after the origin froze the response.
        postcommit_body["audit_tier"] = audit_tier
    postcommit_body_bytes = _canonical_json_bytes(postcommit_body)
    raw_final_artifact: bytes | None = None
    postcommit_error: Exception | None = None
    last_auth_timestamp = 0
    ordinary_retry_count = 0
    finalization_poll_count = 0
    finalization_deadline: float | None = None
    poll_clock = postcommit_clock or time.monotonic
    while True:
        request_timeout = float(timeout)
        if finalization_deadline is not None:
            before_request = float(poll_clock())
            if not math.isfinite(before_request):
                break
            remaining_window = finalization_deadline - before_request
            if remaining_window <= 0:
                break
            request_timeout = min(request_timeout, remaining_window)
        # The reveal body and all request/origin bindings remain byte-identical,
        # while a strictly increasing signed timestamp gives each authenticated
        # retrieval attempt a fresh replay fingerprint.
        auth_timestamp = max(int(time.time()), last_auth_timestamp + 1)
        last_auth_timestamp = auth_timestamp
        postcommit_auth_headers = request_signing.sign_request(
            method="POST",
            path=MESH_POSTCOMMIT_AUDIT_PATH,
            body=postcommit_body_bytes,
            hotkey_ss58=validator_hotkey_ss58,
            hotkey_seed=validator_hotkey_seed,
            timestamp=auth_timestamp,
        )
        postcommit_headers = {
            "Content-Type": "application/json",
            **postcommit_auth_headers,
        }
        retryable = False
        transport_error: MeshCanaryTransportError | None = None
        try:
            raw_final_artifact = (transport or _default_mesh_canary_transport)(
                _coordinator_postcommit_url(endpoint),
                postcommit_body_bytes,
                postcommit_headers,
                request_timeout,
            )
            break
        except MeshCanaryTransportError as exc:
            postcommit_error = exc
            retryable = exc.retryable
            transport_error = exc
        except Exception as exc:
            postcommit_error = exc
            retryable = isinstance(
                exc,
                (TimeoutError, ConnectionError, OSError),
            )
        if (
            transport_error is not None
            and transport_error.retryable
            and transport_error.error_code in MESH_POSTCOMMIT_POLLABLE_ERROR_CODES
        ):
            now_poll = float(poll_clock())
            if not math.isfinite(now_poll):
                break
            if finalization_deadline is None:
                finalization_deadline = (
                    now_poll + MESH_POSTCOMMIT_FINALIZATION_WINDOW_SECONDS
                )
            finalization_poll_count += 1
            delay = (
                transport_error.retry_after_seconds
                if transport_error.retry_after_seconds is not None
                else MESH_POSTCOMMIT_FINALIZATION_POLL_DEFAULT_SECONDS
            )
            if (
                finalization_poll_count >= MESH_POSTCOMMIT_FINALIZATION_MAX_POLLS
                or now_poll >= finalization_deadline
                or delay > finalization_deadline - now_poll
            ):
                break
            retry_sleep(delay)
            continue
        ordinary_retry_count += 1
        if (
            not retryable
            or ordinary_retry_count >= MESH_POSTCOMMIT_MAX_ATTEMPTS
        ):
            break
        retry_sleep(
            MESH_POSTCOMMIT_RETRY_DELAYS[ordinary_retry_count - 1]
        )

    if raw_final_artifact is None:
        total_ms = (time.monotonic() - start) * 1000.0
        return _phase_one_result(
            ok=False,
            reason=(
                "mesh postcommit proof obligation unavailable after exact "
                f"retrieval retries: {postcommit_error}"
            ),
            # A signed origin commits the coordinator to open the proof after
            # seeing the challenge.  Exhausted retrieval is therefore a proof
            # failure, not an availability skip; marking it nonpunitive would
            # permit selective abort of sampled decode checks.
            transport_error=False,
            inference_ms=origin_ms,
            total_ms=total_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )
    total_ms = (time.monotonic() - start) * 1000.0

    try:
        final_artifact = _parse_artifact_response(raw_final_artifact)
    except Exception as exc:
        return _phase_one_result(
            ok=False,
            reason=f"mesh postcommit proof response invalid: {exc}",
            inference_ms=origin_ms,
            total_ms=total_ms,
            receipt=origin_receipt,
            artifact=origin_artifact,
            openai_request=openai_request,
        )

    final_receipt = final_artifact.get("receipt")
    if not isinstance(final_receipt, dict):
        return _phase_one_result(
            ok=False,
            reason="mesh postcommit artifact missing receipt",
            inference_ms=origin_ms,
            total_ms=total_ms,
            receipt=origin_receipt,
            artifact=final_artifact,
            openai_request=openai_request,
        )

    if verify_artifact_fn is None:
        try:
            from verallm.mesh.worker import verify_mesh_postcommit_artifact
        except Exception as exc:
            completed_total_ms = (time.monotonic() - start) * 1000.0
            return _phase_one_result(
                ok=False,
                reason=f"mesh postcommit verifier unavailable: {exc}",
                validator_error=True,
                inference_ms=origin_ms,
                total_ms=completed_total_ms,
                receipt=final_receipt,
                artifact=final_artifact,
                openai_request=openai_request,
            )

        verify_artifact_fn = verify_mesh_postcommit_artifact

    try:
        verified = bool(
            verify_artifact_fn(
                final_artifact,
                origin_artifact,
                openai_request,
                challenge_nonce=nonce,
                expected_coordinator_hotkey=(
                    expected_coordinator.coordinator_hotkey
                ),
                expected_coordinator_uid=expected_coordinator.coordinator_uid,
                expected_validator_hotkey=validator_hotkey_ss58,
                require_coordinator_signature=True,
                verification_snapshot=verification_snapshot,
                verification_snapshot_now_unix=int(now_float),
                audit_tier=audit_tier,
            )
        )
    except Exception as exc:
        completed_total_ms = (time.monotonic() - start) * 1000.0
        if is_validator_verification_fault(exc):
            return _phase_one_result(
                ok=False,
                reason=f"mesh postcommit verifier unavailable: {exc}",
                validator_error=True,
                inference_ms=origin_ms,
                total_ms=completed_total_ms,
                receipt=final_receipt,
                artifact=final_artifact,
                openai_request=openai_request,
            )
        return _phase_one_result(
            ok=False,
            reason=f"mesh postcommit artifact verification error: {exc}",
            inference_ms=origin_ms,
            total_ms=completed_total_ms,
            receipt=final_receipt,
            artifact=final_artifact,
            openai_request=openai_request,
        )
    if not verified:
        completed_total_ms = (time.monotonic() - start) * 1000.0
        return _phase_one_result(
            ok=False,
            reason="mesh postcommit artifact verification failed",
            inference_ms=origin_ms,
            total_ms=completed_total_ms,
            receipt=final_receipt,
            artifact=final_artifact,
            openai_request=openai_request,
        )

    try:
        input_tokens, output_tokens = _proof_bound_mesh_token_counts(
            response,
            final_receipt,
        )
    except ValueError as exc:
        completed_total_ms = (time.monotonic() - start) * 1000.0
        return _phase_one_result(
            ok=False,
            reason=f"mesh proof-bound token accounting invalid: {exc}",
            inference_ms=origin_ms,
            total_ms=completed_total_ms,
            receipt=final_receipt,
            artifact=final_artifact,
            openai_request=openai_request,
        )
    full_text = ""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        message = None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            full_text = content
        # Reasoning models emit their thinking as message.reasoning_content
        # (llama-server --reasoning-format deepseek). That text IS generated
        # output: a small canary budget can be consumed entirely by thinking,
        # and the output-sanity guard must count it instead of failing the
        # canary as empty_visible_output.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            full_text = f"{full_text}\n{reasoning}" if full_text else reasoning
    completed_total_ms = (time.monotonic() - start) * 1000.0

    return _phase_one_result(
        ok=True,
        full_text=full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=canary_ttft_ms,
        inference_ms=origin_ms,
        total_ms=completed_total_ms,
        receipt=final_receipt,
        artifact=final_artifact,
        openai_request=openai_request,
        deferred_pending=False,
    )


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Legacy unsigned transport used only by the experimental audit helper."""

    request = urllib.request.Request(
        url,
        data=_canonical_json_bytes(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def resolve_mesh_deferred_audit(
    endpoint: str,
    artifact: dict[str, Any],
    openai_request: dict[str, Any],
    deferred_randomness: str,
    timeout: float = 180.0,
    post_fn=None,
    verify_bundle_fn=None,
) -> tuple[bool, str]:
    """Experimental compatibility helper for legacy deferred audit bundles.

    The active mesh canary path does not call this function.  It remains here
    only so previously persisted experimental audits can be inspected while
    that protocol is retired.
    """

    if post_fn is None:
        post_fn = _post_json
    try:
        bundle = post_fn(
            endpoint.rstrip("/") + "/v1/mesh/proof/deferred-audit",
            {
                "artifact": artifact,
                "openai_request": openai_request,
                "deferred_randomness": deferred_randomness,
            },
            timeout=timeout,
        )
    except Exception as exc:
        return False, f"deferred audit fetch failed: {exc}"

    if verify_bundle_fn is None:
        from verallm.mesh.worker import verify_deferred_mesh_audit_bundle

        verify_bundle_fn = verify_deferred_mesh_audit_bundle
    try:
        ok = bool(verify_bundle_fn(bundle, artifact, openai_request))
    except Exception as exc:
        return False, f"deferred audit bundle verification error: {exc}"
    if not ok:
        return False, "deferred audit bundle verification failed"
    return True, ""


__all__ = [
    "MAX_MESH_ARTIFACT_RESPONSE_BYTES",
    "MESH_DEFERRED_AUDIT_DELAY_BLOCKS",
    "MESH_INFERENCE_PATH",
    "MESH_POSTCOMMIT_AUDIT_PATH",
    "MeshCanaryResult",
    "MeshCanaryTransport",
    "resolve_mesh_deferred_audit",
    "run_mesh_canary",
]
