"""Fetch, verify, and pin private-mesh verification snapshots.

The coordinator is authoritative only for the signed snapshot bytes.  Every
value that grants the snapshot meaning is supplied independently by the
validator through :class:`MeshSnapshotExpectations` and compared exactly
before the snapshot reaches persistent state.
"""

from __future__ import annotations

import json
import math
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from neurons import request_signing
from neurons.validator_db import ValidatorStateDB
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    assert_endpoint_free_payload,
    verify_mesh_verification_snapshot_signature,
)


MESH_VERIFICATION_SNAPSHOT_PATH = "/v1/mesh/verification-snapshot"
MAX_SNAPSHOT_RESPONSE_BYTES = 1_048_576

#: Per-attempt fetch timeout. A coordinator under audit/organic load can
#: stall the snapshot route for tens of seconds while /health stays sub-
#: second. Load tolerance belongs in this budget;
#: liveness enforcement belongs to the caller's bounded retry window.
MESH_SNAPSHOT_FETCH_TIMEOUT_S = 30.0

SnapshotFetch = Callable[[str, Mapping[str, str], float], bytes]


def is_transient_snapshot_fetch_error(exc: BaseException) -> bool:
    """Whether a snapshot fetch failure is load/transport-shaped.

    Transient failures (timeouts, connection errors, 5xx/429 refusals,
    truncated responses) deserve a bounded retry: a coordinator that is
    merely busy — or mid-relaunch — presents exactly these. Everything
    else (signature/binding/parse failures, auth refusals) is a
    definitive verdict about the response and must never be retried into
    acceptance.
    """

    import http.client

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 425, 429, 500, 502, 503, 504)
    if isinstance(exc, http.client.HTTPException):
        return True
    # URLError, socket.timeout/TimeoutError, and every ConnectionError
    # flavour are OSError subclasses.
    return isinstance(exc, OSError)


@dataclass(frozen=True)
class MeshSnapshotExpectations:
    """Validator/chain-approved facts a coordinator cannot self-authorize."""

    mesh_id: str
    generation: int
    epoch: int
    coordinator: MeshCoordinatorIdentity
    model: MeshModelAnchors
    policy: MeshVerificationPolicy

    def validate(self) -> None:
        if not isinstance(self.mesh_id, str) or not self.mesh_id:
            raise ValueError("expected mesh_id must be a non-empty string")
        if type(self.generation) is not int or not 1 <= self.generation < 2**63:
            raise ValueError("expected generation must be a positive 63-bit integer")
        if type(self.epoch) is not int or not 0 <= self.epoch < 2**63:
            raise ValueError("expected epoch must be a non-negative 63-bit integer")
        if not isinstance(self.coordinator, MeshCoordinatorIdentity):
            raise ValueError("expected coordinator must be MeshCoordinatorIdentity")
        if not isinstance(self.model, MeshModelAnchors):
            raise ValueError("expected model must be MeshModelAnchors")
        if not isinstance(self.policy, MeshVerificationPolicy):
            raise ValueError("expected policy must be MeshVerificationPolicy")
        self.coordinator.validate()
        self.model.validate()
        self.policy.validate()


@dataclass(frozen=True)
class MeshSnapshotTrustAnchors:
    """Chain/config facts for discovering a coordinator-authored mesh view.

    A private coordinator is authorized to choose its opaque ``mesh_id`` and
    monotonically increasing snapshot generation.  The validator independently
    anchors the coordinator, model, epoch, and proof policy, then relies on the
    immutable DB high-water mark to reject rollback or equivocation.
    """

    epoch: int
    coordinator: MeshCoordinatorIdentity
    model: MeshModelAnchors
    policy: MeshVerificationPolicy

    def validate(self) -> None:
        if type(self.epoch) is not int or not 0 <= self.epoch < 2**63:
            raise ValueError("expected epoch must be a non-negative 63-bit integer")
        if not isinstance(self.coordinator, MeshCoordinatorIdentity):
            raise ValueError("expected coordinator must be MeshCoordinatorIdentity")
        if not isinstance(self.model, MeshModelAnchors):
            raise ValueError("expected model must be MeshModelAnchors")
        if not isinstance(self.policy, MeshVerificationPolicy):
            raise ValueError("expected policy must be MeshVerificationPolicy")
        self.coordinator.validate()
        self.model.validate()
        self.policy.validate()


def _snapshot_url(coordinator_endpoint: str) -> str:
    if not isinstance(coordinator_endpoint, str):
        raise ValueError("coordinator_endpoint must be a string")
    parsed = urlsplit(coordinator_endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("coordinator_endpoint must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("coordinator_endpoint must not contain user information")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("coordinator_endpoint must be an origin without a path")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return origin + MESH_VERIFICATION_SNAPSHOT_PATH


def _default_fetch(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers=dict(headers),
        method="GET",
    )
    # Stock mesh endpoints deliberately support self-signed TLS when an
    # operator has no domain.  Transport encryption still protects the
    # request in flight; authenticity comes from the validator-signed request
    # and the coordinator-signed snapshot, whose exact chain/config bindings
    # are verified below.  This must match the production endpoint gate in
    # ``verallm.mesh.probe`` or an endpoint can pass deployment and then be
    # excluded by every validator solely because its certificate is private.
    context = (
        ssl._create_unverified_context()  # noqa: S323
        if urlsplit(url).scheme == "https"
        else None
    )
    with urllib.request.urlopen(  # noqa: S310
        request,
        timeout=timeout,
        context=context,
    ) as response:
        payload = response.read(MAX_SNAPSHOT_RESPONSE_BYTES + 1)
    if len(payload) > MAX_SNAPSHOT_RESPONSE_BYTES:
        raise ValueError("mesh verification snapshot response is too large")
    return payload


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field in mesh snapshot: {key}")
        result[key] = value
    return result


def _parse_snapshot_response(raw: bytes) -> MeshVerificationSnapshot:
    if not isinstance(raw, bytes):
        raise ValueError("snapshot fetcher must return bytes")
    if len(raw) > MAX_SNAPSHOT_RESPONSE_BYTES:
        raise ValueError("mesh verification snapshot response is too large")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value is forbidden: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("mesh verification snapshot response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("mesh verification snapshot response must be a JSON object")
    assert_endpoint_free_payload(payload)
    return MeshVerificationSnapshot.from_dict(payload)


def fetch_and_pin_mesh_verification_snapshot(
    *,
    coordinator_endpoint: str,
    validator_hotkey_ss58: str,
    validator_hotkey_seed: bytes,
    expectations: MeshSnapshotExpectations,
    state_db: ValidatorStateDB,
    timeout: float = MESH_SNAPSHOT_FETCH_TIMEOUT_S,
    fetch: SnapshotFetch | None = None,
    clock: Callable[[], float] = time.time,
    max_future_skew_s: int = 30,
) -> MeshVerificationSnapshot:
    """Fetch one authenticated snapshot, verify exact bindings, and pin it.

    ``expectations`` must come from validator-owned configuration and fresh
    chain/registry reads.  In particular, callers must never construct it from
    the coordinator response being checked.

    The signed GET has an exactly empty body and the fixed path
    :data:`MESH_VERIFICATION_SNAPSHOT_PATH`.  A return value means the exact
    snapshot was accepted by the rollback/equivocation-safe DB transaction.
    """

    if not isinstance(expectations, MeshSnapshotExpectations):
        raise ValueError("expectations must be MeshSnapshotExpectations")
    expectations.validate()
    if not isinstance(state_db, ValidatorStateDB):
        raise ValueError("state_db must be ValidatorStateDB")
    if not isinstance(validator_hotkey_ss58, str) or not validator_hotkey_ss58:
        raise ValueError("validator_hotkey_ss58 must be a non-empty string")
    if not isinstance(validator_hotkey_seed, bytes) or len(validator_hotkey_seed) < 32:
        raise ValueError("validator_hotkey_seed must contain at least 32 bytes")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if type(max_future_skew_s) is not int or max_future_skew_s < 0:
        raise ValueError("max_future_skew_s must be a non-negative integer")

    now = clock()
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("clock must return a Unix timestamp")
    fetched_at = float(now)
    if not math.isfinite(fetched_at) or fetched_at <= 0:
        raise ValueError("clock must return a positive finite Unix timestamp")
    now_unix = int(fetched_at)

    url = _snapshot_url(coordinator_endpoint)
    request_body = b""
    auth_headers = request_signing.sign_request(
        method="GET",
        path=MESH_VERIFICATION_SNAPSHOT_PATH,
        body=request_body,
        hotkey_ss58=validator_hotkey_ss58,
        hotkey_seed=validator_hotkey_seed,
    )
    raw = (fetch or _default_fetch)(url, auth_headers, timeout)
    snapshot = _parse_snapshot_response(raw)

    snapshot.validate(require_signature=True)
    snapshot.validate_expected_bindings(
        expected_mesh_id=expectations.mesh_id,
        expected_generation=expectations.generation,
        expected_coordinator=expectations.coordinator,
        expected_model=expectations.model,
        expected_policy=expectations.policy,
    )
    snapshot.validate_freshness(
        now_unix=now_unix,
        expected_epoch=expectations.epoch,
        max_future_skew_s=max_future_skew_s,
    )
    if not verify_mesh_verification_snapshot_signature(
        snapshot,
        expected_hotkey=expectations.coordinator.coordinator_hotkey,
        expected_epoch=expectations.epoch,
        expected_mesh_id=expectations.mesh_id,
        expected_generation=expectations.generation,
        expected_coordinator=expectations.coordinator,
        expected_model=expectations.model,
        expected_policy=expectations.policy,
        now_unix=now_unix,
        max_future_skew_s=max_future_skew_s,
    ):
        raise ValueError("mesh verification snapshot signature is invalid")

    state_db.upsert_mesh_verification_snapshot(
        coordinator_address=expectations.coordinator.coordinator_evm_address,
        model_index=expectations.coordinator.model_index,
        epoch=expectations.epoch,
        generation=expectations.generation,
        snapshot_hash=snapshot.snapshot_hash_hex(),
        snapshot_json=snapshot.to_dict(),
        fetched_at=fetched_at,
    )
    return snapshot


def discover_and_pin_mesh_verification_snapshot(
    *,
    coordinator_endpoint: str,
    validator_hotkey_ss58: str,
    validator_hotkey_seed: bytes,
    trust_anchors: MeshSnapshotTrustAnchors,
    state_db: ValidatorStateDB,
    timeout: float = MESH_SNAPSHOT_FETCH_TIMEOUT_S,
    fetch: SnapshotFetch | None = None,
    clock: Callable[[], float] = time.time,
    max_future_skew_s: int = 30,
    allow_mesh_change: bool = False,
) -> MeshVerificationSnapshot:
    """Discover, authenticate, and pin one private mesh snapshot.

    Unlike :func:`fetch_and_pin_mesh_verification_snapshot`, this production
    discovery form does not pretend ``mesh_id`` or ``generation`` come from a
    contract that does not currently store them.  The signed coordinator may
    author those two operational identifiers; all security-sensitive facts are
    matched to validator-owned chain/config anchors and the DB enforces their
    monotonic history.
    """

    if not isinstance(trust_anchors, MeshSnapshotTrustAnchors):
        raise ValueError("trust_anchors must be MeshSnapshotTrustAnchors")
    trust_anchors.validate()
    if not isinstance(state_db, ValidatorStateDB):
        raise ValueError("state_db must be ValidatorStateDB")
    if not isinstance(validator_hotkey_ss58, str) or not validator_hotkey_ss58:
        raise ValueError("validator_hotkey_ss58 must be a non-empty string")
    if not isinstance(validator_hotkey_seed, bytes) or len(validator_hotkey_seed) < 32:
        raise ValueError("validator_hotkey_seed must contain at least 32 bytes")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if type(max_future_skew_s) is not int or max_future_skew_s < 0:
        raise ValueError("max_future_skew_s must be a non-negative integer")

    now = clock()
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("clock must return a Unix timestamp")
    fetched_at = float(now)
    if not math.isfinite(fetched_at) or fetched_at <= 0:
        raise ValueError("clock must return a positive finite Unix timestamp")
    now_unix = int(fetched_at)

    url = _snapshot_url(coordinator_endpoint)
    auth_headers = request_signing.sign_request(
        method="GET",
        path=MESH_VERIFICATION_SNAPSHOT_PATH,
        body=b"",
        hotkey_ss58=validator_hotkey_ss58,
        hotkey_seed=validator_hotkey_seed,
    )
    raw = (fetch or _default_fetch)(url, auth_headers, timeout)
    snapshot = _parse_snapshot_response(raw)
    snapshot.validate(require_signature=True)
    snapshot.validate_expected_bindings(
        expected_coordinator=trust_anchors.coordinator,
        expected_model=trust_anchors.model,
        expected_policy=trust_anchors.policy,
    )
    snapshot.validate_freshness(
        now_unix=now_unix,
        expected_epoch=trust_anchors.epoch,
        max_future_skew_s=max_future_skew_s,
    )
    if not verify_mesh_verification_snapshot_signature(
        snapshot,
        expected_hotkey=trust_anchors.coordinator.coordinator_hotkey,
        expected_epoch=trust_anchors.epoch,
        expected_mesh_id=snapshot.mesh_id,
        expected_generation=snapshot.generation,
        expected_coordinator=trust_anchors.coordinator,
        expected_model=trust_anchors.model,
        expected_policy=trust_anchors.policy,
        now_unix=now_unix,
        max_future_skew_s=max_future_skew_s,
    ):
        raise ValueError("mesh verification snapshot signature is invalid")

    state_db.upsert_mesh_verification_snapshot(
        coordinator_address=trust_anchors.coordinator.coordinator_evm_address,
        model_index=trust_anchors.coordinator.model_index,
        epoch=trust_anchors.epoch,
        generation=snapshot.generation,
        snapshot_hash=snapshot.snapshot_hash_hex(),
        snapshot_json=snapshot.to_dict(),
        fetched_at=fetched_at,
        allow_mesh_change=allow_mesh_change,
    )
    return snapshot


__all__ = [
    "MAX_SNAPSHOT_RESPONSE_BYTES",
    "MESH_SNAPSHOT_FETCH_TIMEOUT_S",
    "MESH_VERIFICATION_SNAPSHOT_PATH",
    "MeshSnapshotExpectations",
    "MeshSnapshotTrustAnchors",
    "SnapshotFetch",
    "discover_and_pin_mesh_verification_snapshot",
    "fetch_and_pin_mesh_verification_snapshot",
    "is_transient_snapshot_fetch_error",
]
