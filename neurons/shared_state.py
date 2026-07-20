"""Shared state file between validator and proxy processes.

The validator writes a JSON file with scores + epoch info after each epoch close.
The proxy reads it periodically for score-weighted routing and receipt creation.

The file is written atomically via ``os.replace`` (POSIX-safe), so the proxy
never sees a half-written file.  If the file is missing or corrupt, the proxy
falls back to uniform miner selection — the system degrades gracefully.
"""

from __future__ import annotations

import json
import logging
import bittensor as bt
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "/tmp/verathos_validator_state.json"


@dataclass
class MinerEntry:
    """Minimal miner info for shared state (proxy fallback when RPC is down)."""

    address: str
    endpoint: str
    model_id: str
    model_index: int
    quant: str
    max_context_len: int
    uid: Optional[int] = None
    hotkey_ss58: str = ""
    coldkey_ss58: str = ""
    tee_enabled: bool = False
    tee_platform: str = ""
    enclave_public_key: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    compute_capability: str = ""
    gpu_uuids: List[str] = field(default_factory=list)


@dataclass
class AuditDrain:
    """Temporary endpoint routing exclusion for a capacity audit window."""

    audit_id: str
    address: str
    model_index: int
    endpoint: str
    until_ts: float
    reason: str = "capacity_audit"


@dataclass
class ValidatorSharedState:
    """State shared from the validator process to the proxy process."""

    epoch_number: int = 0
    epoch_start_block: int = 0
    # Per-model scores: address -> {model_index (str) -> ema_score}.
    # Prior format (address -> float) is auto-migrated on read.
    miner_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Miners on probation: address -> list of model_indices.
    # Proxy should exclude these from organic traffic routing.
    probation_miners: Dict[str, List[int]] = field(default_factory=dict)
    # Discovered miner endpoints (proxy fallback when chain RPC is unavailable).
    miner_endpoints: List[MinerEntry] = field(default_factory=list)
    # Temporary audit drains consumed by public proxies. These are not verdicts
    # and do not imply punishment; they only avoid routing user work into the
    # timed audit window.
    audit_drains: List[AuditDrain] = field(default_factory=list)
    # Last normalized weights sent to set_weights() (uid -> weight).
    # Includes burn UID. Empty until first weight-setting boundary.
    last_weights: Dict[int, float] = field(default_factory=dict)
    # Per-model demand scores (model_id -> bps 0-10000) computed from organic
    # traffic.  Proxy serves these via /v1/network/stats for the webapp dashboard.
    demand_scores: Dict[str, int] = field(default_factory=dict)
    # EVM address (lowercase) → {hotkey_ss58, coldkey_ss58} for all known miners.
    # Proxy uses this to resolve SS58 for miners discovered on-chain.
    ss58_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # SubnetConfig blacklist — addresses whose weights the validator zeros.
    # Proxy reads this to also zero the score in /v1/network/stats.
    blacklisted_addresses: List[str] = field(default_factory=list)
    # Stale EVM addresses excluded by validator UID ownership checks. Proxy
    # must not route or display these addresses.
    stale_miner_addresses: List[str] = field(default_factory=list)
    updated_at: float = 0.0


def write_shared_state(
    state: ValidatorSharedState,
    path: str = DEFAULT_STATE_PATH,
) -> None:
    """Atomically write shared state (validator side).

    Uses write-to-tmp + ``os.replace`` so the proxy never reads a partial file.
    """
    data = {
        "epoch_number": state.epoch_number,
        "epoch_start_block": state.epoch_start_block,
        "miner_scores": state.miner_scores,
        "probation_miners": state.probation_miners,
        "miner_endpoints": [
            {"address": m.address, "endpoint": m.endpoint,
             "model_id": m.model_id, "model_index": m.model_index,
             "quant": m.quant, "max_context_len": m.max_context_len,
             "uid": m.uid, "hotkey_ss58": m.hotkey_ss58,
             "coldkey_ss58": m.coldkey_ss58,
             "tee_enabled": m.tee_enabled, "tee_platform": m.tee_platform,
             "enclave_public_key": m.enclave_public_key,
             "gpu_name": m.gpu_name, "gpu_count": m.gpu_count,
             "vram_gb": m.vram_gb, "compute_capability": m.compute_capability,
             "gpu_uuids": m.gpu_uuids}
            for m in state.miner_endpoints
        ],
        "audit_drains": [
            {"audit_id": d.audit_id, "address": d.address,
             "model_index": d.model_index, "endpoint": d.endpoint,
             "until_ts": d.until_ts, "reason": d.reason}
            for d in state.audit_drains
        ],
        "last_weights": {str(k): v for k, v in state.last_weights.items()},
        "demand_scores": state.demand_scores,
        "ss58_map": state.ss58_map,
        "blacklisted_addresses": list(state.blacklisted_addresses),
        "stale_miner_addresses": list(state.stale_miner_addresses),
        "updated_at": time.time(),
    }
    tmp_path = path + ".tmp"
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)  # Atomic on POSIX
    except Exception as exc:
        bt.logging.warning(f"Failed to write shared state to {path}: {exc}")
        # Clean up tmp file if it exists
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_shared_state(
    path: str = DEFAULT_STATE_PATH,
) -> Optional[ValidatorSharedState]:
    """Read shared state (proxy side).

    Returns ``None`` if the file is missing, unreadable, or corrupt.
    The caller should treat ``None`` as "no validator data available" and
    fall back to uniform miner selection.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        miner_endpoints = [
            MinerEntry(**{k: v for k, v in m.items()
                          if k in MinerEntry.__dataclass_fields__})
            for m in data.get("miner_endpoints", [])
        ]
        audit_drains = [
            AuditDrain(**{k: v for k, v in d.items()
                          if k in AuditDrain.__dataclass_fields__})
            for d in data.get("audit_drains", [])
        ]
        # Migrate old flat scores (address -> float) to per-model format
        raw_scores = data.get("miner_scores", {})
        miner_scores: Dict[str, Dict[str, float]] = {}
        for addr, val in raw_scores.items():
            if isinstance(val, (int, float)):
                # Old format: treat as model_index 0
                miner_scores[addr] = {"0": float(val)}
            elif isinstance(val, dict):
                miner_scores[addr] = val
            else:
                miner_scores[addr] = {"0": 1.0}

        return ValidatorSharedState(
            epoch_number=data.get("epoch_number", 0),
            epoch_start_block=data.get("epoch_start_block", 0),
            miner_scores=miner_scores,
            probation_miners=data.get("probation_miners", {}),
            miner_endpoints=miner_endpoints,
            audit_drains=audit_drains,
            demand_scores=data.get("demand_scores", {}),
            ss58_map=data.get("ss58_map", {}),
            blacklisted_addresses=data.get("blacklisted_addresses", []),
            stale_miner_addresses=data.get("stale_miner_addresses", []),
            updated_at=data.get("updated_at", 0.0),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def write_miner_debug_state(data: Dict[str, Any], path: str) -> None:
    """Atomically write the optional public miner diagnostics cache."""
    tmp_path = path + ".tmp"
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception as exc:
        bt.logging.warning(f"Failed to write miner debug state to {path}: {exc}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_miner_debug_state(path: str) -> Optional[Dict[str, Any]]:
    """Read the last complete miner diagnostics cache."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None
