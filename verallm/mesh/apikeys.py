"""Operator-issued API keys for the private pool OpenAI API.

A pool operator can use their own meshes regardless of subnet
registration: the pool manager exposes an OpenAI-compatible surface
(GET /v1/models, POST /v1/chat/completions) authenticated by keys the
operator mints here. This is deliberately DISTINCT from the
validator-hosted subnet API (which fronts every registered mesh with
payment): these keys front ONE pool, they are stored as sha256 hashes in
the owner-only pool state (the ``pool-token.txt`` trust model), and they
never touch a database. Semantics mirror ``neurons/auth.py``'s
APIKeyManager (prefix + hash-at-rest + revoke/list) without its DB
dependency.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any, Mapping

POOL_API_KEY_PREFIX = "vrt_pk_"


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def mint_pool_api_key(
    state: dict[str, Any], *, name: str = ""
) -> tuple[str, dict[str, Any]]:
    """Create a key, store its hash + metadata in ``state``, return it.

    The cleartext key exists only in the return value; persist the state
    (caller's ``_save``) and hand the key to the operator exactly once.
    """

    key = POOL_API_KEY_PREFIX + secrets.token_urlsafe(32)
    record = {
        "name": str(name or "").strip()[:64],
        "created_at_unix": int(time.time()),
        "disabled": False,
        "last_used_unix": 0,
    }
    state.setdefault("api_keys", {})[_key_hash(key)] = record
    return key, dict(record)


def list_pool_api_keys(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Metadata rows (hash-prefixed id, never the key) for the operator."""

    rows = []
    for key_hash, record in (state.get("api_keys") or {}).items():
        if not isinstance(record, Mapping):
            continue
        rows.append(
            {
                "id": str(key_hash)[:12],
                "name": str(record.get("name", "") or ""),
                "created_at_unix": int(record.get("created_at_unix", 0) or 0),
                "disabled": bool(record.get("disabled", False)),
                "last_used_unix": int(record.get("last_used_unix", 0) or 0),
            }
        )
    rows.sort(key=lambda row: row["created_at_unix"])
    return rows


def revoke_pool_api_key(state: dict[str, Any], key_id: str) -> bool:
    """Revoke by the hash-prefix id from ``list_pool_api_keys``."""

    key_id = str(key_id or "").strip()
    if not key_id:
        return False
    keys = state.get("api_keys") or {}
    for key_hash in list(keys):
        if str(key_hash).startswith(key_id):
            keys.pop(key_hash)
            return True
    return False


def authenticate_pool_api_key(
    state: Mapping[str, Any], authorization_header: str
) -> str:
    """Check ``Authorization: Bearer vrt_pk_...`` against the stored hashes.

    Returns the matching key's hash ("" on failure) so callers can rate
    limit per key without ever holding the cleartext. Constant-time
    comparison over the hash; a disabled or unknown key fails identically.
    Updates ``last_used_unix`` best-effort (callers persist lazily; the
    timestamp is operator convenience, not audit)."""

    header = str(authorization_header or "").strip()
    if not header.lower().startswith("bearer "):
        return ""
    key = header[7:].strip()
    if not key.startswith(POOL_API_KEY_PREFIX):
        return ""
    candidate = _key_hash(key)
    for key_hash, record in (state.get("api_keys") or {}).items():
        if secrets.compare_digest(str(key_hash), candidate):
            if isinstance(record, Mapping) and record.get("disabled"):
                return ""
            if isinstance(record, dict):
                record["last_used_unix"] = int(time.time())
            return str(key_hash)
    return ""
