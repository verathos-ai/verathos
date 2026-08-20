"""HTTP authentication helpers for mesh coordinator and worker routes.

The validator helper consumes the existing ``X-Validator-*`` wire format from
``neurons.request_signing``.  Callers must pass the request method, path, and
body exactly as received; in a :class:`http.server.BaseHTTPRequestHandler`
those values are ``handler.command``, ``handler.path``, and the raw bytes read
from ``handler.rfile``.

Internal coordinator/worker calls use a separate, domain-separated HMAC
format.  Validator signatures and internal MACs are therefore not reusable
across the two protocols.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    MAX_CLOCK_SKEW,
    verify_request,
)

HDR_MESH_SIGNATURE = "X-Mesh-Signature"
HDR_MESH_TIMESTAMP = "X-Mesh-Timestamp"
HDR_MESH_NONCE = "X-Mesh-Nonce"

_INTERNAL_HTTP_DOMAIN = b"verathos.mesh.internal-http.v1\x00"
_VALIDATOR_REPLAY_DOMAIN = b"verathos.mesh.validator-replay.v1\x00"
_INTERNAL_REPLAY_DOMAIN = b"verathos.mesh.internal-replay.v1\x00"

# The miner refreshes this file every five minutes.  Three refresh intervals
# allow brief RPC/retry trouble without leaving a validator identity trusted
# indefinitely after the refresh process dies.
DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS = 15 * 60.0


@dataclass(frozen=True)
class AuthResult:
    """Result returned by the non-raising request verification helpers."""

    ok: bool
    status_code: int
    reason: str
    principal: str | None = None


class RequestReplayCache:
    """Thread-safe, bounded cache for already accepted signed requests.

    Only requests whose signature or MAC has already verified should be added.
    Live entries are never evicted to admit new ones: when the cache is full it
    fails closed until an entry expires.
    """

    def __init__(
        self,
        *,
        max_entries: int = 100_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = int(max_entries)
        self._clock = clock
        self._entries: dict[bytes, float] = {}
        self._lock = threading.Lock()

    def claim(
        self,
        fingerprint: bytes,
        *,
        expires_at: float,
        now: float | None = None,
    ) -> bool:
        """Atomically record ``fingerprint`` or reject it if already present.

        ``False`` also means the cache was full or the requested expiry was no
        longer in the future.  All three cases must fail authentication.
        """

        current = float(self._clock() if now is None else now)
        expiry = float(expires_at)
        key = bytes(fingerprint)
        with self._lock:
            expired = [item for item, deadline in self._entries.items() if deadline <= current]
            for item in expired:
                del self._entries[item]

            if key in self._entries:
                return False
            if expiry <= current or len(self._entries) >= self._max_entries:
                return False
            self._entries[key] = expiry
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _read_validator_allowlist_document(
    path: str | Path,
) -> tuple[dict[str, object], frozenset[str]]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("validator allowlist must be a JSON object")
    validators = data.get("validators")
    if not isinstance(validators, list):
        raise ValueError("validator allowlist must contain a validators list")

    hotkeys: set[str] = set()
    for entry in validators:
        if not isinstance(entry, dict):
            raise ValueError("validator allowlist entries must be JSON objects")
        hotkey = entry.get("hotkey_ss58")
        if not isinstance(hotkey, str) or not hotkey.strip():
            raise ValueError("validator allowlist entry has no hotkey_ss58")
        if hotkey != hotkey.strip():
            raise ValueError("validator hotkey must not contain surrounding whitespace")
        hotkeys.add(hotkey)
    return data, frozenset(hotkeys)


def read_validator_allowlist(path: str | Path) -> frozenset[str]:
    """Read SS58 hotkeys from the validator allowlist written by the miner.

    The existing schema is ``{"validators": [{"hotkey_ss58": ...}, ...]}``;
    additional top-level and per-validator fields are intentionally ignored.
    Missing, malformed, or incomplete data raises instead of returning a
    partially trusted set.
    """

    _data, hotkeys = _read_validator_allowlist_document(path)
    return hotkeys


def _validator_allowlist_updated_at(
    data: Mapping[str, object],
    *,
    max_file_age_seconds: float,
    clock: Callable[[], float],
) -> float:
    updated_at = data.get("updated_at")
    if (
        not isinstance(updated_at, (int, float))
        or isinstance(updated_at, bool)
        or not math.isfinite(float(updated_at))
        or float(updated_at) <= 0
    ):
        raise ValueError("validator allowlist has no valid numeric updated_at")
    timestamp = float(updated_at)
    current = float(clock())
    if not math.isfinite(current):
        raise ValueError("validator allowlist clock is invalid")
    age = current - timestamp
    if age < -MAX_CLOCK_SKEW or age > float(max_file_age_seconds):
        raise ValueError("validator allowlist is outside its accepted age")
    return timestamp


def read_fresh_validator_allowlist(
    path: str | Path,
    *,
    max_file_age_seconds: float = DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
    clock: Callable[[], float] = time.time,
) -> frozenset[str]:
    """Read an allowlist only when its metagraph refresh is still current."""

    if (
        not math.isfinite(float(max_file_age_seconds))
        or float(max_file_age_seconds) <= 0
    ):
        raise ValueError("max_file_age_seconds must be positive")
    data, hotkeys = _read_validator_allowlist_document(path)
    _validator_allowlist_updated_at(
        data,
        max_file_age_seconds=float(max_file_age_seconds),
        clock=clock,
    )
    return hotkeys


class ValidatorAllowlist:
    """Reloading, fail-closed view of the miner's validator JSON file."""

    def __init__(
        self,
        path: str | Path,
        *,
        reload_interval_seconds: float = 60.0,
        max_file_age_seconds: float | None = (
            DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS
        ),
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if reload_interval_seconds < 0:
            raise ValueError("reload_interval_seconds must not be negative")
        if (
            max_file_age_seconds is not None
            and (
                not math.isfinite(float(max_file_age_seconds))
                or float(max_file_age_seconds) <= 0
            )
        ):
            raise ValueError("max_file_age_seconds must be positive")
        self._path = Path(path)
        self._reload_interval = float(reload_interval_seconds)
        self._max_file_age = (
            None if max_file_age_seconds is None else float(max_file_age_seconds)
        )
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._hotkeys: frozenset[str] = frozenset()
        self._available = False
        self._updated_at: float | None = None
        self._last_attempt: float | None = None
        self._lock = threading.Lock()
        self.refresh(force=True)

    def _cached_timestamp_is_fresh(self) -> bool:
        if self._max_file_age is None:
            return True
        if self._updated_at is None:
            return False
        current = float(self._clock())
        if not math.isfinite(current):
            return False
        age = current - self._updated_at
        return -MAX_CLOCK_SKEW <= age <= self._max_file_age

    def _clear(self) -> None:
        self._hotkeys = frozenset()
        self._available = False
        self._updated_at = None

    def refresh(self, *, force: bool = False) -> bool:
        """Reload the file, clearing all trust if it cannot be validated."""

        with self._lock:
            now_mono = float(self._monotonic_clock())
            if (
                not force
                and self._last_attempt is not None
                and now_mono - self._last_attempt < self._reload_interval
            ):
                if self._available and not self._cached_timestamp_is_fresh():
                    self._clear()
                return self._available
            self._last_attempt = now_mono

            try:
                updated_at: float | None = None
                if self._max_file_age is not None:
                    data, hotkeys = _read_validator_allowlist_document(self._path)
                    updated_at = _validator_allowlist_updated_at(
                        data,
                        max_file_age_seconds=self._max_file_age,
                        clock=self._clock,
                    )
                else:
                    hotkeys = read_validator_allowlist(self._path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
                # Never retain a previously loaded allowlist after a failed
                # reload: stale trust is more dangerous than a temporary deny.
                self._clear()
                return False

            self._hotkeys = hotkeys
            self._available = True
            self._updated_at = updated_at
            return True

    def snapshot(self) -> tuple[bool, frozenset[str]]:
        """Return ``(file_valid, allowed_hotkeys)`` after a due reload."""

        self.refresh()
        with self._lock:
            return self._available, self._hotkeys


def _single_header(headers: Mapping[str, str], name: str) -> str | None:
    """Return one case-insensitive header value, rejecting duplicates."""

    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name, [])
        if len(values) != 1:
            return None
        value = values[0]
        return value if isinstance(value, str) and value else None

    matches = [value for key, value in headers.items() if str(key).lower() == name.lower()]
    if len(matches) != 1:
        return None
    value = matches[0]
    return value if isinstance(value, str) and value else None


def _signature_bytes(signature_hex: str, *, expected_length: int) -> bytes | None:
    text = signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
    if len(text) != expected_length * 2:
        return None
    try:
        value = bytes.fromhex(text)
    except ValueError:
        return None
    return value if len(value) == expected_length else None


def verify_validator_http_request(
    *,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    allowlist: ValidatorAllowlist,
    replay_cache: RequestReplayCache,
) -> AuthResult:
    """Authenticate one exact raw HTTP request from a subnet validator."""

    available, allowed_hotkeys = allowlist.snapshot()
    if not available:
        return AuthResult(False, 503, "validator allowlist unavailable")

    hotkey = _single_header(headers, HDR_HOTKEY)
    signature_hex = _single_header(headers, HDR_SIGNATURE)
    timestamp = _single_header(headers, HDR_TIMESTAMP)
    if hotkey is None or signature_hex is None or timestamp is None:
        return AuthResult(False, 401, "missing or duplicate validator auth header")
    if hotkey not in allowed_hotkeys:
        return AuthResult(False, 403, "validator hotkey is not allowed")

    ok, _reason = verify_request(
        method=method,
        path=path,
        body=body,
        hotkey_ss58=hotkey,
        signature_hex=signature_hex,
        timestamp_str=timestamp,
    )
    if not ok:
        return AuthResult(False, 401, "invalid validator request signature")

    signature = _signature_bytes(signature_hex, expected_length=64)
    try:
        signed_at = int(timestamp)
    except (TypeError, ValueError):
        signed_at = 0
    if signature is None or signed_at <= 0:
        return AuthResult(False, 401, "invalid validator request signature")

    fingerprint = hashlib.sha256(
        _VALIDATOR_REPLAY_DOMAIN + hotkey.encode("utf-8") + signature
    ).digest()
    if not replay_cache.claim(
        fingerprint,
        expires_at=float(signed_at + MAX_CLOCK_SKEW),
    ):
        return AuthResult(False, 401, "validator request was already accepted")
    return AuthResult(True, 200, "ok", principal=hotkey)


def _secret_bytes(secret: bytes | bytearray | memoryview | str) -> bytes:
    value = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if not value:
        raise ValueError("mesh HTTP authentication secret must not be empty")
    return value


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def build_internal_signing_message(
    method: str,
    path: str,
    body: bytes,
    timestamp: str,
    nonce: str,
) -> bytes:
    """Build the unambiguous, domain-separated internal mesh MAC message."""

    fields = (
        method.encode("utf-8"),
        path.encode("utf-8"),
        bytes(body),
        timestamp.encode("ascii"),
        nonce.encode("ascii"),
    )
    return _INTERNAL_HTTP_DOMAIN + b"".join(_length_prefixed(field) for field in fields)


def sign_internal_http_request(
    *,
    secret: bytes | bytearray | memoryview | str,
    method: str,
    path: str,
    body: bytes,
    timestamp: int | str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Return HMAC headers for one exact coordinator/worker HTTP request."""

    timestamp_text = str(int(time.time()) if timestamp is None else timestamp)
    # Validate the timestamp before producing a credential that no verifier
    # would accept.
    if not timestamp_text.isascii() or not timestamp_text.isdecimal():
        raise ValueError("timestamp must be unsigned Unix seconds")
    nonce_text = secrets.token_hex(16) if nonce is None else str(nonce)
    if len(nonce_text) != 32 or any(ch not in "0123456789abcdef" for ch in nonce_text):
        raise ValueError("nonce must be 16 bytes encoded as lowercase hex")
    message = build_internal_signing_message(
        method,
        path,
        body,
        timestamp_text,
        nonce_text,
    )
    signature = hmac.digest(_secret_bytes(secret), message, "sha256")
    return {
        HDR_MESH_SIGNATURE: signature.hex(),
        HDR_MESH_TIMESTAMP: timestamp_text,
        HDR_MESH_NONCE: nonce_text,
    }


def verify_internal_http_request(
    *,
    secret: bytes | bytearray | memoryview | str,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    replay_cache: RequestReplayCache | None = None,
    freshness_seconds: int = MAX_CLOCK_SKEW,
    now: float | None = None,
) -> AuthResult:
    """Authenticate an internal mesh request with constant-time MAC checking."""

    if freshness_seconds <= 0:
        raise ValueError("freshness_seconds must be positive")
    signature_hex = _single_header(headers, HDR_MESH_SIGNATURE)
    timestamp = _single_header(headers, HDR_MESH_TIMESTAMP)
    nonce = _single_header(headers, HDR_MESH_NONCE)
    if signature_hex is None or timestamp is None or nonce is None:
        return AuthResult(False, 401, "missing or duplicate internal auth header")
    if not timestamp.isascii() or not timestamp.isdecimal():
        return AuthResult(False, 401, "invalid internal request timestamp")

    current = float(time.time() if now is None else now)
    try:
        signed_at = int(timestamp)
    except (ValueError, OverflowError):
        return AuthResult(False, 401, "invalid internal request timestamp")
    if abs(current - signed_at) > freshness_seconds:
        return AuthResult(False, 401, "internal request timestamp is outside freshness window")
    if len(nonce) != 32 or any(ch not in "0123456789abcdef" for ch in nonce):
        return AuthResult(False, 401, "invalid internal request nonce")

    signature = _signature_bytes(signature_hex, expected_length=32)
    if signature is None:
        return AuthResult(False, 401, "invalid internal request signature")
    try:
        message = build_internal_signing_message(method, path, body, timestamp, nonce)
        expected = hmac.digest(_secret_bytes(secret), message, "sha256")
    except (UnicodeError, ValueError):
        return AuthResult(False, 503, "internal authentication unavailable")
    if not hmac.compare_digest(expected, signature):
        return AuthResult(False, 401, "invalid internal request signature")

    if replay_cache is not None:
        fingerprint = hashlib.sha256(
            _INTERNAL_REPLAY_DOMAIN + nonce.encode("ascii") + signature
        ).digest()
        if not replay_cache.claim(
            fingerprint,
            expires_at=float(signed_at + freshness_seconds),
            now=current,
        ):
            return AuthResult(False, 401, "internal request was already accepted")
    return AuthResult(True, 200, "ok", principal="mesh-internal")


__all__ = [
    "AuthResult",
    "HDR_MESH_NONCE",
    "HDR_MESH_SIGNATURE",
    "HDR_MESH_TIMESTAMP",
    "RequestReplayCache",
    "ValidatorAllowlist",
    "build_internal_signing_message",
    "read_validator_allowlist",
    "read_fresh_validator_allowlist",
    "DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS",
    "sign_internal_http_request",
    "verify_internal_http_request",
    "verify_validator_http_request",
]
