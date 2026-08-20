"""Private directory of mesh coordinator control endpoints.

Coordinators register themselves here over a signed announce
(``POST /v1/operator/announce``); the proxy then pulls each coordinator's
SANITIZED viewer-lane operator overview and serves that cached data at
``GET /v1/operator/fleet``. The control endpoint itself is confidential:
it lives only in this table, is never included in any served payload, and
is never logged above debug level.

Design notes (operator console plan):
- Announce is address-book only (startup + once per epoch). Data freshness
  comes from the proxy's demand-driven overview pulls, not from announce
  frequency.
- The announce is self-attested (signed by the coordinator's own hotkey)
  and affects only how that hotkey's own fleet is DISPLAYED. It never
  feeds scoring or validation, which keeps the trust model of
  validator-signed, endpoint-free mesh snapshots unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: One scoring epoch (360 blocks x 12 s). Announces older than two epochs
#: are marked stale; rows not renewed for a day are dropped.
EPOCH_SECONDS = 360 * 12
STALE_AFTER_S = 2 * EPOCH_SECONDS
DROP_AFTER_S = 24 * 3600
MAX_ANNOUNCE_BODY = 4 * 1024

#: Data-plane cadences: a viewed fleet refreshes when older than
#: OVERVIEW_TTL_S, recently-viewed coordinators stay warm for WARM_FOR_S,
#: and every registered coordinator gets a /healthz once per HEALTH_EVERY_S.
OVERVIEW_TTL_S = 15.0
WARM_FOR_S = 600.0
WARM_EVERY_S = 30.0
HEALTH_EVERY_S = 60.0


def _strip_endpoint_shapes(value: Any) -> Any:
    """Defensively remove endpoint-shaped data from a payload.

    The viewer-lane overview already omits endpoints; this is belt and
    braces so a future upstream field can never leak a coordinator or
    worker address through the fleet route.
    """

    if isinstance(value, dict):
        return {
            k: _strip_endpoint_shapes(v)
            for k, v in value.items()
            if "endpoint" not in str(k).lower() and str(k).lower() != "url"
        }
    if isinstance(value, list):
        return [_strip_endpoint_shapes(v) for v in value]
    if isinstance(value, str) and value.startswith(("http://", "https://", "ws://", "wss://")):
        return None
    return value


def valid_control_endpoint(endpoint: str) -> bool:
    """Shape + SSRF check for a self-announced coordinator endpoint.

    The proxy will POST/GET this address on its own network, so private,
    loopback, link-local, and metadata ranges are rejected — an announce
    passing the known-hotkey gate must not turn the proxy into an
    internal port prober (the cached overview is publicly served).
    """
    try:
        parsed = urlparse(endpoint)
    except Exception:
        return False
    if not (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    ):
        return False
    try:
        if parsed.port is not None and not (0 < parsed.port < 65536):
            return False
    except ValueError:
        return False
    host = parsed.hostname or ""
    if host.lower() in ("localhost",):
        return False
    try:
        import ipaddress

        addr = ipaddress.ip_address(host)
        if not addr.is_global:
            return False
    except ValueError:
        pass  # DNS name — allowed (coordinators are public by definition)
    return True


@dataclass
class _Entry:
    endpoint: str
    pool_id: str
    netuid: int
    announced_at: float
    last_ok_at: float = 0.0
    overview: Optional[dict] = None
    overview_at: float = 0.0
    last_viewed_at: float = 0.0
    refreshing: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OperatorDirectory:
    """In-memory registry + overview cache, persisted through the proxy DB."""

    def __init__(self, backend: Any):
        self._backend = backend
        self._entries: dict[str, _Entry] = {}
        self._init_schema()
        self._load()

    # -- persistence -----------------------------------------------------

    def _init_schema(self) -> None:
        self._backend.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_coordinators (
                hotkey_ss58 TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                pool_id TEXT NOT NULL DEFAULT '',
                netuid INTEGER NOT NULL DEFAULT 0,
                announced_at REAL NOT NULL DEFAULT 0,
                last_ok_at REAL NOT NULL DEFAULT 0
            )
            """
        )

    def _load(self) -> None:
        now = time.time()
        for row in self._backend.fetchall(
            "SELECT hotkey_ss58, endpoint, pool_id, netuid, announced_at, "
            "last_ok_at FROM operator_coordinators"
        ):
            hotkey, endpoint, pool_id, netuid, announced_at, last_ok_at = row
            if now - float(announced_at) > DROP_AFTER_S:
                continue
            self._entries[str(hotkey)] = _Entry(
                endpoint=str(endpoint),
                pool_id=str(pool_id),
                netuid=int(netuid),
                announced_at=float(announced_at),
                last_ok_at=float(last_ok_at),
            )
        if self._entries:
            logger.info(
                "operator directory: %d coordinator(s) restored", len(self._entries)
            )

    # -- announce (write) ------------------------------------------------

    def register(
        self, hotkey_ss58: str, endpoint: str, pool_id: str, netuid: int
    ) -> None:
        now = time.time()
        entry = self._entries.get(hotkey_ss58)
        if entry is not None and entry.endpoint == endpoint:
            entry.announced_at = now
            entry.pool_id = pool_id
            entry.netuid = netuid
        else:
            self._entries[hotkey_ss58] = _Entry(
                endpoint=endpoint,
                pool_id=pool_id,
                netuid=netuid,
                announced_at=now,
            )
        self._backend.execute(
            "INSERT INTO operator_coordinators "
            "(hotkey_ss58, endpoint, pool_id, netuid, announced_at, last_ok_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(hotkey_ss58) DO UPDATE SET endpoint=excluded.endpoint, "
            "pool_id=excluded.pool_id, netuid=excluded.netuid, "
            "announced_at=excluded.announced_at",
            (hotkey_ss58, endpoint, pool_id, int(netuid), now, 0.0),
        )
        # The sqlite backend does NOT autocommit (postgres does) — without
        # this, every registration dies with the process.
        self._backend.commit()
        logger.debug("operator directory: announce accepted for %s", hotkey_ss58)

    # -- fleet (read) ----------------------------------------------------

    async def fleet(self, hotkey_ss58: str) -> dict:
        """Cached sanitized overview for a hotkey's coordinator, or absent."""
        entry = self._entries.get(hotkey_ss58)
        if entry is None:
            return {"registered": False, "coordinators": []}
        now = time.time()
        entry.last_viewed_at = now
        if now - entry.overview_at > OVERVIEW_TTL_S:
            await self._refresh(hotkey_ss58, entry)
        return {
            "registered": True,
            "coordinators": [
                {
                    "pool_id": entry.pool_id,
                    "netuid": entry.netuid,
                    "announced_at": int(entry.announced_at),
                    "last_ok_at": int(entry.last_ok_at),
                    "stale": (now - entry.announced_at) > STALE_AFTER_S
                    or (entry.last_ok_at > 0 and now - entry.last_ok_at > STALE_AFTER_S),
                    "overview_age_s": (
                        round(max(0.0, time.time() - entry.overview_at), 1)
                        if entry.overview
                        else None
                    ),
                    "overview": entry.overview,
                }
            ],
        }

    async def _refresh(self, hotkey_ss58: str, entry: _Entry) -> None:
        import httpx

        async with entry.lock:
            now = time.time()
            if now - entry.overview_at <= OVERVIEW_TTL_S:
                return
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(8.0, connect=4.0), verify=False
                ) as client:
                    res = await client.post(
                        f"{entry.endpoint}/v1/operator/overview", json={}
                    )
                if res.status_code == 200:
                    entry.overview = _strip_endpoint_shapes(res.json())
                    entry.overview_at = time.time()
                    entry.last_ok_at = entry.overview_at
                    self._backend.execute(
                        "UPDATE operator_coordinators SET last_ok_at = ? "
                        "WHERE hotkey_ss58 = ?",
                        (entry.last_ok_at, hotkey_ss58),
                    )
                    self._backend.commit()
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug(
                    "operator overview refresh failed for %s: %s", hotkey_ss58, exc
                )

    # -- background upkeep ----------------------------------------------

    async def upkeep_loop(self) -> None:
        """Warm recently-viewed fleets, health-check the rest, expire rows."""
        import httpx

        while True:
            try:
                now = time.time()
                for hotkey, entry in list(self._entries.items()):
                    if now - entry.announced_at > DROP_AFTER_S:
                        self._entries.pop(hotkey, None)
                        self._backend.execute(
                            "DELETE FROM operator_coordinators WHERE hotkey_ss58 = ?",
                            (hotkey,),
                        )
                        self._backend.commit()
                        continue
                    if now - entry.last_viewed_at < WARM_FOR_S:
                        await self._refresh(hotkey, entry)
                    elif now - entry.last_ok_at > HEALTH_EVERY_S:
                        try:
                            async with httpx.AsyncClient(
                                timeout=httpx.Timeout(5.0, connect=3.0), verify=False
                            ) as client:
                                res = await client.get(f"{entry.endpoint}/healthz")
                            if res.status_code == 200:
                                entry.last_ok_at = time.time()
                        except httpx.HTTPError:
                            pass
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("operator directory upkeep failed: %s", exc)
            await asyncio.sleep(WARM_EVERY_S)
