"""Centralized rate limiting for the validator proxy.

Provides a sliding-window rate limiter with two backends:

- **In-memory** (default): ``defaultdict(list)`` of timestamps per key.
  Suitable for single-proxy deployments.  Lost on restart.
- **PostgreSQL**: ``proxy_rate_limits`` table with atomic
  ``INSERT ... ON CONFLICT DO UPDATE``.  Shared across proxy instances.

No external dependencies — uses stdlib for in-memory and the existing
:class:`~neurons.db.DBBackend` for PostgreSQL.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Set

from starlette.requests import Request

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Outcome of a rate limit check."""

    allowed: bool
    remaining: int
    retry_after: int  # seconds until next window (0 if allowed)


class RateLimiter:
    """Sliding-window rate limiter with in-memory or PostgreSQL backend.

    Parameters
    ----------
    backend
        A :class:`~neurons.db.PostgresBackend` instance to use the DB-backed
        limiter.  Pass ``None`` (default) for in-memory only.
    """

    def __init__(self, backend=None):
        self._use_pg = backend is not None and getattr(backend, "is_postgres", False)
        self._backend = backend

        # In-memory state
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = time.monotonic()
        self._CLEANUP_INTERVAL = 60.0  # seconds

        if self._use_pg:
            self._init_pg_schema()

    # ── Public API ───────────────────────────────────────────────

    def check(self, key: str, limit: int, window: int) -> RateLimitResult:
        """Check and increment the counter for *key*.

        Returns a :class:`RateLimitResult` indicating whether the request
        is allowed and how many requests remain in the window.
        """
        if self._use_pg:
            return self._check_pg(key, limit, window)
        return self._check_memory(key, limit, window)

    # ── In-memory backend ────────────────────────────────────────

    def _check_memory(self, key: str, limit: int, window: int) -> RateLimitResult:
        now = time.monotonic()
        with self._lock:
            # Periodic cleanup of stale keys
            if now - self._last_cleanup > self._CLEANUP_INTERVAL:
                self._cleanup(now, window)
                self._last_cleanup = now

            timestamps = self._buckets[key]
            cutoff = now - window
            # Prune expired entries
            timestamps[:] = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= limit:
                retry_after = int(timestamps[0] - cutoff) + 1
                return RateLimitResult(allowed=False, remaining=0, retry_after=max(1, retry_after))

            timestamps.append(now)
            remaining = limit - len(timestamps)
            return RateLimitResult(allowed=True, remaining=remaining, retry_after=0)

    def _cleanup(self, now: float, default_window: int = 300) -> None:
        """Remove keys with no recent activity (called under lock)."""
        cutoff = now - default_window
        stale = [k for k, ts in self._buckets.items() if not ts or ts[-1] < cutoff]
        for k in stale:
            del self._buckets[k]

    # ── PostgreSQL backend ───────────────────────────────────────

    def _init_pg_schema(self) -> None:
        try:
            self._backend.execute("""
                CREATE TABLE IF NOT EXISTS proxy_rate_limits (
                    key       TEXT NOT NULL,
                    window_ts BIGINT NOT NULL,
                    count     INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (key, window_ts)
                )
            """)
            self._backend.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_limits_ts
                    ON proxy_rate_limits(window_ts)
            """)
            self._backend.commit()
        except Exception as e:
            logger.warning("Failed to create rate_limits table: %s", e)

    def _check_pg(self, key: str, limit: int, window: int) -> RateLimitResult:
        now = int(time.time())
        window_ts = (now // window) * window  # floor to window boundary
        prev_ts = window_ts - window

        # Probabilistic cleanup (~1% of calls)
        if random.random() < 0.01:
            try:
                cleanup_cutoff = now - window * 3
                self._backend.execute(
                    "DELETE FROM proxy_rate_limits WHERE window_ts < ?",
                    (cleanup_cutoff,),
                )
                self._backend.commit()
            except Exception:
                pass

        # Atomic upsert + return count
        try:
            if hasattr(self._backend, "execute_in_transaction"):
                return self._check_pg_atomic(key, limit, window, window_ts, prev_ts, now)
            else:
                # Fallback: non-atomic (shouldn't happen with PostgresBackend)
                return self._check_pg_simple(key, limit, window, window_ts, prev_ts, now)
        except Exception as e:
            logger.debug("Rate limit DB error (allowing request): %s", e)
            return RateLimitResult(allowed=True, remaining=limit, retry_after=0)

    def _check_pg_atomic(self, key: str, limit: int, window: int,
                         window_ts: int, prev_ts: int, now: int) -> RateLimitResult:
        def _do(conn):
            # Get previous window count (for sliding approximation)
            cur = conn.execute(
                "SELECT count FROM proxy_rate_limits WHERE key = %s AND window_ts = %s",
                (key, prev_ts),
            )
            prev_row = cur.fetchone()
            prev_count = prev_row[0] if prev_row else 0

            # Upsert current window
            cur = conn.execute(
                """INSERT INTO proxy_rate_limits (key, window_ts, count)
                   VALUES (%s, %s, 1)
                   ON CONFLICT (key, window_ts)
                   DO UPDATE SET count = proxy_rate_limits.count + 1
                   RETURNING count""",
                (key, window_ts),
            )
            current_count = cur.fetchone()[0]

            # Sliding window approximation:
            # weight = fraction of previous window still in scope
            elapsed = now - window_ts
            weight = max(0.0, 1.0 - elapsed / window)
            approx = int(prev_count * weight) + current_count

            if approx > limit:
                retry_after = window - elapsed
                # Rollback the increment (we already incremented but should deny)
                conn.execute(
                    "UPDATE proxy_rate_limits SET count = count - 1 WHERE key = %s AND window_ts = %s",
                    (key, window_ts),
                )
                return RateLimitResult(
                    allowed=False, remaining=0, retry_after=max(1, int(retry_after))
                )

            remaining = max(0, limit - approx)
            return RateLimitResult(allowed=True, remaining=remaining, retry_after=0)

        return self._backend.execute_in_transaction(_do)

    def _check_pg_simple(self, key: str, limit: int, window: int,
                         window_ts: int, prev_ts: int, now: int) -> RateLimitResult:
        """Non-atomic fallback (less accurate under concurrency)."""
        prev_row = self._backend.fetchone(
            "SELECT count FROM proxy_rate_limits WHERE key = ? AND window_ts = ?",
            (key, prev_ts),
        )
        prev_count = prev_row[0] if prev_row else 0

        cur_row = self._backend.fetchone(
            "SELECT count FROM proxy_rate_limits WHERE key = ? AND window_ts = ?",
            (key, window_ts),
        )
        current_count = cur_row[0] if cur_row else 0

        elapsed = now - window_ts
        weight = max(0.0, 1.0 - elapsed / window)
        approx = int(prev_count * weight) + current_count + 1

        if approx > limit:
            retry_after = window - elapsed
            return RateLimitResult(
                allowed=False, remaining=0, retry_after=max(1, int(retry_after))
            )

        # Increment
        if cur_row:
            self._backend.execute(
                "UPDATE proxy_rate_limits SET count = count + 1 WHERE key = ? AND window_ts = ?",
                (key, window_ts),
            )
        else:
            self._backend.execute(
                "INSERT INTO proxy_rate_limits (key, window_ts, count) VALUES (?, ?, 1)",
                (key, window_ts),
            )
        self._backend.commit()

        remaining = max(0, limit - approx)
        return RateLimitResult(allowed=True, remaining=remaining, retry_after=0)


# ── IP Resolution ────────────────────────────────────────────────────


def get_client_ip(request: Request, trusted_proxies: Optional[Set[str]] = None) -> str:
    """Extract the real client IP from the request.

    When ``trusted_proxies`` is configured, parses ``X-Forwarded-For`` by
    walking from right and skipping trusted proxy IPs.

    When ``trusted_proxies`` is **not** configured, ``X-Forwarded-For`` and
    ``X-Real-IP`` headers are **ignored** (they are user-controlled and could
    be spoofed to bypass rate limits). Falls back to ``request.client.host``.

    Parameters
    ----------
    request
        The incoming Starlette/FastAPI request.
    trusted_proxies
        Set of proxy IPs to skip when parsing ``X-Forwarded-For``.
        **Must** be set when running behind a reverse proxy (nginx).
    """
    if trusted_proxies:
        # X-Forwarded-For: client, proxy1, proxy2
        xff = request.headers.get("x-forwarded-for")
        if xff:
            ips = [ip.strip() for ip in xff.split(",")]
            # Walk from right, skip trusted proxies, return first untrusted
            for ip in reversed(ips):
                if ip not in trusted_proxies:
                    return ip
            # All IPs are trusted proxies — fall back to direct connection
            return request.client.host if request.client else "unknown"

        # X-Real-IP (set by nginx) — only trust when trusted_proxies configured
        xri = request.headers.get("x-real-ip")
        if xri:
            return xri.strip()

    # Direct connection (or no trusted proxies — ignore XFF to prevent spoofing)
    if request.client:
        return request.client.host

    return "unknown"
