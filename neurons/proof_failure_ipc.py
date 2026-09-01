"""Owner-only proxy-to-validator proof-failure event spool.

The proxy can reject a served artifact before it has a valid receipt to push
back through the miner.  This local spool lets the co-located validator learn
that failure directly instead of relying on the untrusted miner to preserve a
negative receipt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from verallm.mesh.private_files import read_owner_only_text, write_owner_only_json


logger = logging.getLogger(__name__)

MAX_EVENT_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 1_000
MAX_ENDPOINT_CHARS = 2_048
_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")


def proof_failure_spool_dir(shared_state_path: str = "") -> Path:
    """Return the shared local event directory for one validator/proxy pair."""

    explicit = os.environ.get("VERATHOS_PROXY_PROOF_FAILURE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if str(shared_state_path or "").strip():
        shared = Path(shared_state_path).expanduser()
        return shared.with_name(shared.name + ".proof-failures")
    data_dir = Path(
        os.environ.get("VERALLM_DATA_DIR", "/tmp")
    ).expanduser()
    return data_dir / "verathos_proxy_proof_failures"


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("proof failure spool must be a non-symlink directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise PermissionError(
            "proof failure spool must be owned by the current user"
        )
    os.chmod(path, 0o700)


def _validated_event(event: dict[str, Any]) -> dict[str, Any]:
    required = {
        "version",
        "event_id",
        "address",
        "model_index",
        "epoch_number",
        "timestamp",
        "endpoint",
        "traffic_class",
        "message",
    }
    if set(event) != required:
        raise ValueError("proof failure event fields are invalid")
    if event["version"] != 1:
        raise ValueError("unsupported proof failure event version")
    event_id = str(event["event_id"])
    try:
        uuid.UUID(event_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("proof failure event_id is invalid") from exc
    address = str(event["address"]).strip().lower()
    if not _EVM_ADDRESS_RE.fullmatch(address):
        raise ValueError("proof failure address is invalid")
    for field_name in ("model_index", "epoch_number", "timestamp"):
        value = event[field_name]
        if type(value) is not int or value < 0:
            raise ValueError(f"proof failure {field_name} is invalid")
    endpoint = str(event["endpoint"])
    traffic_class = str(event["traffic_class"])
    message = str(event["message"])
    if len(endpoint) > MAX_ENDPOINT_CHARS:
        raise ValueError("proof failure endpoint is too long")
    if len(traffic_class) > 32:
        raise ValueError("proof failure traffic_class is too long")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError("proof failure message is too long")
    return {
        "version": 1,
        "event_id": event_id,
        "address": address,
        "model_index": event["model_index"],
        "epoch_number": event["epoch_number"],
        "timestamp": event["timestamp"],
        "endpoint": endpoint,
        "traffic_class": traffic_class,
        "message": message,
    }


def enqueue_proxy_proof_failure(
    *,
    address: str,
    model_index: int,
    epoch_number: int,
    endpoint: str,
    traffic_class: str,
    message: str,
    shared_state_path: str = "",
    event_id: str = "",
) -> Path:
    """Atomically enqueue one locally verified organic proof failure."""

    stable_event_id = str(event_id or uuid.uuid4())
    event = _validated_event(
        {
            "version": 1,
            "event_id": stable_event_id,
            "address": str(address).strip().lower(),
            "model_index": int(model_index),
            "epoch_number": max(0, int(epoch_number)),
            "timestamp": int(time.time()),
            "endpoint": str(endpoint or "")[:MAX_ENDPOINT_CHARS],
            "traffic_class": str(traffic_class or "unknown")[:32],
            "message": str(message or "")[:MAX_MESSAGE_CHARS],
        }
    )
    directory = proof_failure_spool_dir(shared_state_path)
    _ensure_private_directory(directory)
    target = directory / f"event-{event['event_id']}.json"
    if target.exists():
        # Retries for one local probation incident use a stable event ID.
        # Keep the first owner-only payload until the validator acknowledges
        # it instead of creating duplicate financial penalties.
        read_owner_only_text(target, label="proof failure event")
        return target
    return write_owner_only_json(target, event)


def pending_proxy_proof_failures(
    *,
    shared_state_path: str = "",
    limit: int = 100,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load a bounded batch of validated events without deleting them."""

    if type(limit) is not int or limit <= 0:
        raise ValueError("proof failure event limit must be positive")
    directory = proof_failure_spool_dir(shared_state_path)
    if not directory.exists():
        return []
    _ensure_private_directory(directory)
    events: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json"))[:limit]:
        try:
            metadata = os.lstat(path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_size > MAX_EVENT_BYTES
            ):
                raise ValueError("proof failure event file is invalid")
            raw = read_owner_only_text(path, label="proof failure event")
            if len(raw.encode("utf-8")) > MAX_EVENT_BYTES:
                raise ValueError("proof failure event file is too large")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("proof failure event must be a JSON object")
            events.append((path, _validated_event(payload)))
        except Exception as exc:
            quarantine = path.with_suffix(path.suffix + ".invalid")
            try:
                os.replace(path, quarantine)
            except OSError:
                logger.exception(
                    "Could not quarantine invalid proof failure event %s",
                    path,
                )
            else:
                logger.error(
                    "Quarantined invalid proof failure event %s: %s",
                    path,
                    exc,
                )
    return events


def acknowledge_proxy_proof_failure(path: str | Path) -> None:
    """Delete one event after the validator durably applied it."""

    Path(path).unlink(missing_ok=True)


__all__ = [
    "acknowledge_proxy_proof_failure",
    "enqueue_proxy_proof_failure",
    "pending_proxy_proof_failures",
    "proof_failure_spool_dir",
]
