"""Validator endpoint discovery for hot-capacity audit artifact submission."""

from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import bittensor as bt
import httpx


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(default)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    value = int(default)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


DISCOVERY_CACHE_TTL_S = _env_float(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_DISCOVERY_CACHE_S",
    1800.0,
    minimum=1.0,
)
DISCOVERY_RETRY_S = _env_float(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_DISCOVERY_RETRY_S",
    120.0,
    minimum=1.0,
)
DISCOVERY_PROBE_TIMEOUT_S = _env_float(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_PROBE_TIMEOUT_S",
    2.0,
    minimum=0.2,
)
ENDPOINT_FAILURE_THRESHOLD = _env_int(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_ENDPOINT_FAILURE_THRESHOLD",
    2,
    minimum=1,
)
ENDPOINT_TRANSIENT_FAILURE_THRESHOLD = _env_int(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_ENDPOINT_TRANSIENT_FAILURE_THRESHOLD",
    12,
    minimum=1,
)
ENDPOINT_QUARANTINE_S = _env_float(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_ENDPOINT_QUARANTINE_S",
    900.0,
    minimum=1.0,
)
DISCOVERY_PROBE_MAX_WORKERS = _env_int(
    "VERATHOS_CAPACITY_AUDIT_VALIDATOR_PROBE_WORKERS",
    32,
    minimum=1,
)
ALLOWED_CACHED_ENDPOINT_SOURCES = {"axon"}


@dataclass
class ValidatorAuditEndpoint:
    address: str
    endpoint: str
    uid: int = -1
    source: str = "registry"
    is_active: bool = True


def _validator_uid(value) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def normalize_audit_endpoint(endpoint: str) -> str:
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    scheme = parsed.scheme or "http"
    if scheme not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").strip()
    if not host:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port is not None:
        netloc = f"{netloc}:{int(port)}"
    return f"{scheme}://{netloc}{parsed.path or ''}".rstrip("/")


def _endpoint_cache_path() -> Path:
    explicit = os.environ.get("VERATHOS_CAPACITY_AUDIT_VALIDATOR_ENDPOINT_CACHE_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    data_dir = os.environ.get("VERATHOS_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser() / "capacity_audit_validator_endpoints.json"
    return Path.home() / ".verathos" / "capacity_audit_validator_endpoints.json"


def _call_or_value(value):
    return value() if callable(value) else value


def _validator_permits_from_metagraph(mg) -> list:
    permits = getattr(mg, "validator_permit", None)
    if permits is None:
        permits = getattr(mg, "validator_permits", [])
    if hasattr(permits, "tolist"):
        return list(permits.tolist())
    return list(permits or [])


def _metagraph_validator_has_permit(mg, uid: int) -> bool:
    try:
        return bool(_validator_permits_from_metagraph(mg)[int(uid)])
    except Exception:
        return False


def _clean_axon_host(value) -> str:
    host = _call_or_value(value)
    if isinstance(host, int):
        try:
            return str(ipaddress.ip_address(host))
        except Exception:
            return ""
    text = str(host or "").strip()
    if text.startswith("/ipv4/") or text.startswith("/ip4/"):
        text = text.split("/", 2)[2]
    elif text.startswith("/ipv6/") or text.startswith("/ip6/"):
        text = text.split("/", 2)[2]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.startswith("[") and "]" in text:
        return text[1:text.index("]")]
    if ":" in text:
        maybe_host, maybe_port = text.rsplit(":", 1)
        if maybe_port.isdigit() and maybe_host:
            text = maybe_host
    return text.strip()


def _parse_axon_host_port(axon) -> tuple[str, int]:
    host = (
        _clean_axon_host(getattr(axon, "external_ip", ""))
        or _clean_axon_host(getattr(axon, "ip", ""))
        or _clean_axon_host(getattr(axon, "ip_str", ""))
    )
    port = (
        _call_or_value(getattr(axon, "external_port", 0))
        or _call_or_value(getattr(axon, "port", 0))
    )
    try:
        port = int(port or 0)
    except Exception:
        port = 0
    return host, port


def _axon_endpoint(axon, *, scheme: str = "http") -> str:
    if not bool(_call_or_value(getattr(axon, "is_serving", False))):
        return ""
    host, port = _parse_axon_host_port(axon)
    if not host or port <= 0:
        return ""
    if host in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
        return ""
    return normalize_audit_endpoint(f"{scheme}://{host}:{port}")


def validator_axon_audit_endpoints_from_metagraph(mg, *, scheme: str = "http") -> list[ValidatorAuditEndpoint]:
    hotkeys = list(getattr(mg, "hotkeys", []) or [])
    axons = list(getattr(mg, "axons", []) or [])
    permits = _validator_permits_from_metagraph(mg)
    count = min(len(hotkeys), len(axons), len(permits))
    endpoints: list[ValidatorAuditEndpoint] = []
    for uid in range(count):
        try:
            if not bool(permits[uid]):
                continue
            endpoint = _axon_endpoint(axons[uid], scheme=scheme)
            if not endpoint:
                continue
            endpoints.append(ValidatorAuditEndpoint(
                address=f"axon:{hotkeys[uid]}",
                endpoint=endpoint,
                uid=uid,
                source="axon",
                is_active=True,
            ))
        except Exception:
            continue
    return endpoints


def _merge_endpoints(endpoints: Iterable[ValidatorAuditEndpoint]) -> list[ValidatorAuditEndpoint]:
    merged: list[ValidatorAuditEndpoint] = []
    seen_endpoints: set[str] = set()
    seen_uids: set[int] = set()
    for item in endpoints:
        endpoint = normalize_audit_endpoint(getattr(item, "endpoint", ""))
        if not endpoint:
            continue
        uid = _validator_uid(getattr(item, "uid", -1))
        if uid >= 0 and uid in seen_uids:
            continue
        key = endpoint.lower()
        if key in seen_endpoints:
            continue
        item.endpoint = endpoint
        merged.append(item)
        seen_endpoints.add(key)
        if uid >= 0:
            seen_uids.add(uid)
    return merged


class CapacityAuditEndpointResolver:
    """Discovers and tracks validator audit ingest endpoints."""

    def __init__(
        self,
        config,
        *,
        manual_urls: Iterable[str] = (),
        cache_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.manual_urls = tuple(
            endpoint for endpoint in (normalize_audit_endpoint(u) for u in manual_urls) if endpoint
        )
        self.cache_path = Path(cache_path).expanduser() if cache_path else _endpoint_cache_path()
        self.cache_ttl_s = DISCOVERY_CACHE_TTL_S
        self.discovery_retry_s = DISCOVERY_RETRY_S
        self.probe_timeout_s = DISCOVERY_PROBE_TIMEOUT_S
        self.failure_threshold = ENDPOINT_FAILURE_THRESHOLD
        self.transient_failure_threshold = ENDPOINT_TRANSIENT_FAILURE_THRESHOLD
        self.quarantine_s = ENDPOINT_QUARANTINE_S
        self.role_probe_enabled = _env_bool(
            "VERATHOS_CAPACITY_AUDIT_VALIDATOR_ENDPOINT_PROBE",
            True,
        )
        self._endpoints: list[ValidatorAuditEndpoint] = []
        self._cache_updated_at = 0.0
        self._next_refresh_after = 0.0
        self._endpoint_failures: dict[str, int] = {}
        self._endpoint_transient_failures: dict[str, int] = {}
        self._endpoint_quarantine_until: dict[str, float] = {}
        self._background_refresh_lock = threading.Lock()
        self._background_refreshing = False
        self._load_cache()

    def current_urls(self, *, force_refresh: bool = False) -> tuple[str, ...]:
        now = time.time()
        cached = tuple(
            item.endpoint
            for item in self._endpoints
            if item.endpoint and not self._is_quarantined(item.endpoint, now)
        )
        if not force_refresh and cached:
            if (
                now - self._cache_updated_at >= self.cache_ttl_s
                and now >= self._next_refresh_after
            ):
                self._request_background_refresh()
            return cached

        self.refresh(force=force_refresh)
        now = time.time()
        return tuple(
            item.endpoint
            for item in self._endpoints
            if item.endpoint and not self._is_quarantined(item.endpoint, now)
        )

    def _request_background_refresh(self) -> None:
        """Refresh stale chain discovery without blocking artifact publication."""
        with self._background_refresh_lock:
            if self._background_refreshing:
                return
            self._background_refreshing = True

        def _run() -> None:
            try:
                self.refresh(force=False)
            except Exception as exc:
                bt.logging.debug(
                    f"Capacity audit validator background refresh failed: {exc}"
                )
            finally:
                with self._background_refresh_lock:
                    self._background_refreshing = False

        threading.Thread(
            target=_run,
            name="capacity-audit-validator-refresh",
            daemon=True,
        ).start()

    def refresh(self, *, force: bool = False) -> list[ValidatorAuditEndpoint]:
        now = time.time()
        if self.manual_urls:
            manual = [
                ValidatorAuditEndpoint(
                    address=f"manual:{idx}",
                    endpoint=url,
                    uid=-1,
                    source="manual",
                    is_active=True,
                )
                for idx, url in enumerate(self.manual_urls)
            ]
            self._set_endpoints(manual, source="manual override")
            return list(self._endpoints)
        if not force and self._endpoints and now - self._cache_updated_at < self.cache_ttl_s:
            return list(self._endpoints)
        if not force and now < self._next_refresh_after:
            return list(self._endpoints)

        try:
            discovered = self._discover_from_chain()
            usable = self._filter_usable_endpoints(discovered)
            if usable:
                self._set_endpoints(usable, source="chain discovery")
            elif not self._endpoints:
                bt.logging.warning("Capacity audit validator discovery returned no usable endpoints")
                self._next_refresh_after = now + self.discovery_retry_s
            else:
                bt.logging.debug(
                    "Capacity audit validator discovery returned no usable endpoints; "
                    f"keeping {len(self._endpoints)} cached endpoint(s)"
                )
                self._next_refresh_after = now + self.discovery_retry_s
        except Exception as exc:
            self._next_refresh_after = now + self.discovery_retry_s
            if not self._endpoints:
                bt.logging.warning(f"Capacity audit validator discovery failed with empty cache: {exc}")
            else:
                bt.logging.debug(
                    f"Capacity audit validator discovery failed; keeping "
                    f"{len(self._endpoints)} cached endpoint(s): {exc}"
                )
        return list(self._endpoints)

    def record_publish_result(
        self,
        endpoint: str,
        success: bool,
        *,
        failure_kind: str = "hard",
    ) -> None:
        endpoint = normalize_audit_endpoint(endpoint)
        if not endpoint:
            return
        key = endpoint.lower()
        changed = False
        if success:
            if self._endpoint_failures.pop(key, None) is not None:
                changed = True
            if self._endpoint_transient_failures.pop(key, None) is not None:
                changed = True
            if self._endpoint_quarantine_until.pop(key, None) is not None:
                changed = True
            if changed:
                self._save_cache()
            return

        kind = str(failure_kind or "hard").lower()
        if kind == "client":
            if self._endpoint_failures.pop(key, None) is not None:
                changed = True
            if self._endpoint_transient_failures.pop(key, None) is not None:
                changed = True
            if changed:
                self._save_cache()
            return

        if kind == "transient":
            failures = int(self._endpoint_transient_failures.get(key, 0) or 0) + 1
            self._endpoint_transient_failures[key] = failures
            threshold = self.transient_failure_threshold
            failure_label = "transient"
        else:
            failures = int(self._endpoint_failures.get(key, 0) or 0) + 1
            self._endpoint_failures[key] = failures
            threshold = self.failure_threshold
            failure_label = "hard"

        if failures >= threshold:
            until = time.time() + self.quarantine_s
            previous = float(self._endpoint_quarantine_until.get(key, 0) or 0)
            self._endpoint_quarantine_until[key] = until
            if previous <= time.time():
                bt.logging.warning(
                    f"Quarantining capacity-audit validator endpoint {endpoint} "
                    f"for {self.quarantine_s:.0f}s after {failures} "
                    f"{failure_label} failed publish attempt(s)"
                )
        self._save_cache()

    def _discover_from_chain(self) -> list[ValidatorAuditEndpoint]:
        mg = None
        try:
            SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            subtensor = SubtensorCls(network=self.config.subtensor_network)
            try:
                mg = subtensor.metagraph(self.config.netuid)
            finally:
                close = getattr(subtensor, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
        except Exception as exc:
            bt.logging.debug(f"Capacity audit metagraph discovery unavailable: {exc}")

        native_endpoints = validator_axon_audit_endpoints_from_metagraph(mg) if mg is not None else []
        return _merge_endpoints(native_endpoints)

    def _filter_usable_endpoints(
        self,
        endpoints: Iterable[ValidatorAuditEndpoint],
    ) -> list[ValidatorAuditEndpoint]:
        if not self.role_probe_enabled:
            return list(endpoints)
        endpoint_list = list(endpoints)
        if not endpoint_list:
            return []
        usable: list[ValidatorAuditEndpoint] = []
        dropped = 0
        inconclusive = 0
        max_workers = min(DISCOVERY_PROBE_MAX_WORKERS, max(1, len(endpoint_list)))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = {pool.submit(self._probe_endpoint, item.endpoint): item for item in endpoint_list}
        handled = set()

        def record_result(item: ValidatorAuditEndpoint, status: str) -> None:
            nonlocal dropped, inconclusive
            if status == "bad":
                dropped += 1
                return
            usable.append(item)
            if status != "ok":
                inconclusive += 1

        try:
            try:
                for future in as_completed(
                    futures,
                    timeout=max(self.probe_timeout_s + 0.5, self.probe_timeout_s * 2.0),
                ):
                    item = futures[future]
                    try:
                        status = future.result()
                    except Exception:
                        status = "unknown"
                    handled.add(future)
                    record_result(item, status)
            except _FuturesTimeout:
                pending = 0
                for future, item in futures.items():
                    if future in handled:
                        continue
                    if future.done():
                        try:
                            status = future.result()
                        except Exception:
                            status = "unknown"
                        record_result(item, status)
                        continue
                    pending += 1
                    future.cancel()
                    usable.append(item)
                inconclusive += pending
                if pending:
                    bt.logging.warning(
                        f"Capacity audit discovery probe timed out for {pending} endpoint(s); "
                        "retaining chain-discovered targets"
                    )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if dropped:
            bt.logging.warning(
                f"Capacity audit discovery dropped {dropped} endpoint(s) without usable audit ingest"
            )
        if inconclusive:
            bt.logging.info(
                f"Capacity audit discovery retained {inconclusive} chain-discovered "
                "endpoint(s) after inconclusive health probes"
            )
        return usable

    def _probe_endpoint(self, endpoint: str) -> str:
        try:
            resp = httpx.get(
                f"{endpoint.rstrip('/')}/capacity/audit/v1/health",
                timeout=self.probe_timeout_s,
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                if not isinstance(data, dict):
                    return "bad"
                service = str(data.get("service") or "").strip().lower()
                if service in {
                    "verathos-capacity-audit-ingest",
                }:
                    return "ok"
                if bool(data.get("capacity_audit")):
                    return "ok"
                return "bad"
            if resp.status_code in {401, 403, 404}:
                return "bad"
            return "unknown"
        except Exception:
            return "unknown"

    def _set_endpoints(self, endpoints: Iterable[ValidatorAuditEndpoint], *, source: str) -> None:
        fresh = _merge_endpoints(endpoints)
        if not fresh:
            return
        self._endpoints = fresh
        self._cache_updated_at = time.time()
        self._next_refresh_after = self._cache_updated_at + self.cache_ttl_s
        valid = {item.endpoint.lower() for item in self._endpoints}
        for key in list(self._endpoint_failures):
            if key not in valid:
                self._endpoint_failures.pop(key, None)
        for key in list(self._endpoint_transient_failures):
            if key not in valid:
                self._endpoint_transient_failures.pop(key, None)
        for key in list(self._endpoint_quarantine_until):
            if key not in valid:
                self._endpoint_quarantine_until.pop(key, None)
        self._save_cache()
        bt.logging.info(
            f"Capacity audit validator endpoints set from {source}: {len(fresh)} endpoint(s)"
        )

    def _is_quarantined(self, endpoint: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        key = normalize_audit_endpoint(endpoint).lower()
        until = float(self._endpoint_quarantine_until.get(key, 0) or 0)
        if until <= 0:
            return False
        if until <= now:
            self._endpoint_quarantine_until.pop(key, None)
            self._save_cache()
            return False
        return True

    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text())
            rows = data.get("endpoints") if isinstance(data, dict) else data
            endpoints = []
            dropped = 0
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                source = str(row.get("source") or "cache")
                if source not in ALLOWED_CACHED_ENDPOINT_SOURCES:
                    dropped += 1
                    continue
                endpoints.append(
                    ValidatorAuditEndpoint(
                        address=str(row.get("address") or ""),
                        endpoint=str(row.get("endpoint") or ""),
                        uid=_validator_uid(row.get("uid")),
                        source=source,
                        is_active=bool(row.get("is_active", True)),
                    )
                )
            self._endpoints = _merge_endpoints(endpoints)
            self._cache_updated_at = float(data.get("updated_at") or 0) if isinstance(data, dict) else 0.0
            health = data.get("endpoint_health") if isinstance(data, dict) else {}
            if isinstance(health, dict):
                now = time.time()
                for key, row in health.items():
                    key = normalize_audit_endpoint(key).lower()
                    if not key or not isinstance(row, dict):
                        continue
                    failures = int(row.get("failures") or 0)
                    transient_failures = int(row.get("transient_failures") or 0)
                    quarantine_until = float(row.get("quarantine_until") or 0)
                    if failures > 0:
                        self._endpoint_failures[key] = failures
                    if transient_failures > 0:
                        self._endpoint_transient_failures[key] = transient_failures
                    if quarantine_until > now:
                        self._endpoint_quarantine_until[key] = quarantine_until
            if self._endpoints:
                bt.logging.info(
                    f"Loaded capacity audit validator endpoint cache: {len(self._endpoints)} endpoint(s)"
                )
            elif dropped:
                self._cache_updated_at = 0.0
                bt.logging.info(
                    f"Ignored {dropped} stale capacity audit validator endpoint cache entries"
                )
                try:
                    self.cache_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    bt.logging.debug(f"Capacity audit stale endpoint cache cleanup failed: {exc}")
        except FileNotFoundError:
            return
        except Exception as exc:
            bt.logging.warning(f"Ignoring invalid capacity audit endpoint cache {self.cache_path}: {exc}")

    def _save_cache(self) -> None:
        if not self._endpoints:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            now = time.time()
            health = {}
            keys = (
                set(self._endpoint_failures)
                | set(self._endpoint_transient_failures)
                | set(self._endpoint_quarantine_until)
            )
            for key in keys:
                failures = int(self._endpoint_failures.get(key, 0) or 0)
                transient_failures = int(self._endpoint_transient_failures.get(key, 0) or 0)
                quarantine_until = float(self._endpoint_quarantine_until.get(key, 0) or 0)
                if failures <= 0 and transient_failures <= 0 and quarantine_until <= now:
                    continue
                health[key] = {
                    "failures": failures,
                    "transient_failures": transient_failures,
                    "quarantine_until": quarantine_until,
                }
            payload = {
                "updated_at": self._cache_updated_at or time.time(),
                "endpoints": [asdict(item) for item in self._endpoints],
                "endpoint_health": health,
            }
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True))
            tmp.replace(self.cache_path)
        except Exception as exc:
            bt.logging.debug(f"Capacity audit endpoint cache write failed: {exc}")
