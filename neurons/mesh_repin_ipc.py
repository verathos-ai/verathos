"""Owner-only proxy-to-validator mesh re-pin request spool.

A relaunched mesh presents a new verification-snapshot generation; until
the validator re-pins it, the proxy cannot route organics to a perfectly
healthy coordinator.  The validator already owns a trust-anchored,
rate-capped re-pin path (``_try_repin_mesh_snapshot``), but its only
trigger was a canary refusal — so every relaunch bought minutes of
"No miners available" while waiting for the next canary or epoch
boundary to happen by.

This spool closes ONLY the trigger latency.  The proxy detects the pin
gap on the first routing attempt and asks the co-located validator to
run its existing re-pin immediately.  Requests carry no authority: the
validator validates the replacement snapshot against the same trust
anchors it uses at the epoch boundary, keeps its per-slot-per-epoch
rate cap, and may refuse.  A hostile or buggy requester can therefore
cause at most a bounded number of snapshot fetches, never an un-audited
pin.

Requests are idempotent liveness signals, not events: one file per
(address, model_index), overwritten in place, deleted once processed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from verallm.mesh.private_files import read_owner_only_text, write_owner_only_json

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 4 * 1024
MAX_REASON_CHARS = 300
# A fresh request younger than this is not rewritten (proxy-side debounce)
# and the validator's per-slot epoch cap bounds the fetch rate regardless.
REQUEST_DEBOUNCE_SECONDS = 30.0
_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")


def mesh_repin_spool_dir(shared_state_path: str = "") -> Path:
    """Return the shared local request directory for one validator/proxy pair."""

    explicit = os.environ.get("VERATHOS_MESH_REPIN_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if str(shared_state_path or "").strip():
        shared = Path(shared_state_path).expanduser()
        return shared.with_name(shared.name + ".mesh-repin")
    data_dir = Path(os.environ.get("VERALLM_DATA_DIR", "/tmp")).expanduser()
    return data_dir / "verathos_mesh_repin_requests"


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("mesh repin spool must be a non-symlink directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise PermissionError("mesh repin spool must be owned by the current user")
    os.chmod(path, 0o700)


def _validated_request(payload: dict[str, Any]) -> dict[str, Any]:
    address = str(payload.get("address", "")).strip().lower()
    if not _EVM_ADDRESS_RE.fullmatch(address):
        raise ValueError("mesh repin address is invalid")
    model_index = payload.get("model_index")
    if type(model_index) is not int or model_index < 0:
        raise ValueError("mesh repin model_index is invalid")
    requested_at = payload.get("requested_at")
    if type(requested_at) not in (int, float) or requested_at < 0:
        raise ValueError("mesh repin requested_at is invalid")
    reason = str(payload.get("reason", ""))[:MAX_REASON_CHARS]
    return {
        "address": address,
        "model_index": int(model_index),
        "requested_at": float(requested_at),
        "reason": reason,
    }


def enqueue_mesh_repin_request(
    *,
    address: str,
    model_index: int,
    reason: str,
    shared_state_path: str = "",
) -> Path | None:
    """Record one re-pin request; returns None when debounced."""

    directory = mesh_repin_spool_dir(shared_state_path)
    _ensure_private_directory(directory)
    target = directory / f"repin-{str(address).lower()}-{int(model_index)}.json"
    if target.exists():
        try:
            age = time.time() - target.stat().st_mtime
        except OSError:
            age = REQUEST_DEBOUNCE_SECONDS
        if age < REQUEST_DEBOUNCE_SECONDS:
            return None
    request = _validated_request(
        {
            "address": str(address).strip().lower(),
            "model_index": int(model_index),
            "requested_at": time.time(),
            "reason": str(reason or "")[:MAX_REASON_CHARS],
        }
    )
    return write_owner_only_json(target, request)


def pending_mesh_repin_requests(
    *,
    shared_state_path: str = "",
    limit: int = 32,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load pending validated requests; quarantine anything malformed."""

    directory = mesh_repin_spool_dir(shared_state_path)
    if not directory.exists():
        return []
    _ensure_private_directory(directory)
    requests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("repin-*.json"))[: max(1, int(limit))]:
        try:
            metadata = os.lstat(path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_size > MAX_REQUEST_BYTES
            ):
                raise ValueError("mesh repin request file is invalid")
            payload = json.loads(
                read_owner_only_text(path, label="mesh repin request")
            )
            if not isinstance(payload, dict):
                raise ValueError("mesh repin request must be a JSON object")
            requests.append((path, _validated_request(payload)))
        except Exception as exc:
            quarantine = path.with_suffix(path.suffix + ".invalid")
            try:
                os.replace(path, quarantine)
            except OSError:
                logger.exception(
                    "Could not quarantine invalid mesh repin request %s", path
                )
            else:
                logger.error(
                    "Quarantined invalid mesh repin request %s: %s", path, exc
                )
    return requests


def acknowledge_mesh_repin_request(path: str | Path) -> None:
    """Delete one request after the validator has acted on it."""

    Path(path).unlink(missing_ok=True)


__all__ = [
    "acknowledge_mesh_repin_request",
    "enqueue_mesh_repin_request",
    "mesh_repin_spool_dir",
    "pending_mesh_repin_requests",
]
