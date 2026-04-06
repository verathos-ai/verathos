"""TTL cache for on-chain reads."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """Simple in-memory TTL cache.

    Used to avoid redundant on-chain reads:
    - ModelSpec: TTL ~1 hour (changes only on re-registration)
    - Miner list: TTL ~5 minutes (miners come and go)
    """

    def __init__(self, default_ttl: float = 300.0):
        self._default_ttl = default_ttl
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        t = ttl if ttl is not None else self._default_ttl
        self._store[key] = _CacheEntry(value=value, expires_at=time.monotonic() + t)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
