"""Validator authentication middleware for the miner server.

Only allows requests signed by validators registered on the subnet.
The validator list is read from a JSON file (written by neurons/miner.py
which queries the metagraph periodically).

Exempt endpoints: /health, /model_spec (always public, rate-limited per IP).

Package boundary: this module does NOT import bittensor. The metagraph
lookup is done in neurons/miner.py which writes the allowlist file.
Signature verification uses substrateinterface (Sr25519).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Default path for the validator allowlist file
DEFAULT_VALIDATORS_PATH = "/tmp/verathos_validators.json"

# How often to re-read the file (seconds)
FILE_RELOAD_INTERVAL = 60

# Endpoints that don't require validator auth
PUBLIC_ENDPOINTS = {"/health", "/model_spec", "/docs", "/openapi.json", "/identity/challenge", "/tee/info"}

# Rate limiting for public endpoints (per IP).
# 60/min allows 20 validator proxies health-checking every 10s (= 120/min)
# since each proxy has its own IP. Random scanners get blocked quickly.
PUBLIC_RATE_LIMIT = 60       # max requests per window
PUBLIC_RATE_WINDOW = 60.0    # window in seconds
_MAX_TRACKED_IPS = 10_000    # cap to prevent memory exhaustion


class _PublicRateLimiter:
    """Lightweight in-memory sliding-window rate limiter keyed by IP."""

    __slots__ = ("_limit", "_window", "_hits")

    def __init__(self, limit: int = PUBLIC_RATE_LIMIT, window: float = PUBLIC_RATE_WINDOW):
        self._limit = limit
        self._window = window
        # IP → deque of timestamps
        self._hits: dict[str, collections.deque] = {}

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = self._hits.get(ip)
        if bucket is None:
            if len(self._hits) >= _MAX_TRACKED_IPS:
                # Evict oldest bucket to cap memory
                oldest_ip = next(iter(self._hits))
                del self._hits[oldest_ip]
            bucket = collections.deque()
            self._hits[ip] = bucket

        # Expire old entries
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self._limit:
            return False

        bucket.append(now)
        return True


class ValidatorAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid validator signature on all non-public endpoints.

    Reads the allowed validator SS58 hotkeys from a JSON file on disk.
    The file is re-read every FILE_RELOAD_INTERVAL seconds.

    When no validators file exists, blocks all non-public requests (deny by
    default).
    """

    def __init__(self, app, validators_path: Optional[str] = None):
        super().__init__(app)
        self._validators_path = Path(
            validators_path
            or os.environ.get("VERATHOS_VALIDATORS_PATH", DEFAULT_VALIDATORS_PATH)
        )
        self._allowed_ss58: Set[str] = set()  # SS58-encoded hotkey addresses
        self._last_load: float = 0.0
        self._no_file_warned: bool = False  # Avoid spamming the missing-file warning
        self._public_limiter = _PublicRateLimiter()
        self._load_validators()

    def _load_validators(self) -> None:
        """Load validator SS58 hotkeys from the JSON file."""
        now = time.time()
        if now - self._last_load < FILE_RELOAD_INTERVAL:
            return
        self._last_load = now

        if not self._validators_path.exists():
            if not self._no_file_warned:
                logger.warning(
                    "Validators file not found at %s — non-public requests will be "
                    "blocked until the file is created.",
                    self._validators_path,
                )
                self._no_file_warned = True
            return

        try:
            data = json.loads(self._validators_path.read_text())
            validators = data.get("validators", [])
            new_ss58 = {v["hotkey_ss58"] for v in validators if v.get("hotkey_ss58")}

            if new_ss58 != self._allowed_ss58:
                logger.info(
                    "Loaded %d validator hotkeys from %s",
                    len(new_ss58), self._validators_path,
                )
            self._allowed_ss58 = new_ss58
            self._no_file_warned = False
        except Exception as e:
            logger.warning("Failed to load validators file: %s", e)

    async def dispatch(self, request: Request, call_next):
        # Reload validators periodically
        self._load_validators()

        # Public endpoints — rate-limited per IP
        if request.url.path in PUBLIC_ENDPOINTS:
            client_ip = request.client.host if request.client else "unknown"
            if not self._public_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"},
                    headers={"Retry-After": str(int(PUBLIC_RATE_WINDOW))},
                )
            return await call_next(request)

        # No validators file yet — deny all non-public requests
        if not self._allowed_ss58:
            return JSONResponse(
                status_code=503,
                content={"error": "Miner is starting up — validator allowlist not yet loaded"},
            )

        # Extract auth headers
        hotkey_ss58 = request.headers.get("x-validator-hotkey", "")
        signature_hex = request.headers.get("x-validator-signature", "")
        timestamp_str = request.headers.get("x-validator-timestamp", "")

        if not hotkey_ss58 or not signature_hex or not timestamp_str:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing validator auth headers (X-Validator-Hotkey, X-Validator-Signature, X-Validator-Timestamp)"},
            )

        # Check if hotkey is in the allowed set
        if hotkey_ss58 not in self._allowed_ss58:
            return JSONResponse(
                status_code=403,
                content={"error": "Hotkey is not a registered validator on this subnet"},
            )

        # Verify signature
        # Import here to avoid circular imports at module level
        from neurons.request_signing import verify_request

        body = await request.body()
        ok, reason = verify_request(
            method=request.method,
            path=request.url.path,
            body=body,
            hotkey_ss58=hotkey_ss58,
            signature_hex=signature_hex,
            timestamp_str=timestamp_str,
        )

        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": f"Validator signature verification failed: {reason}"},
            )

        # Store validated hotkey on request state for downstream logging
        request.state.validator_hotkey = hotkey_ss58

        return await call_next(request)
