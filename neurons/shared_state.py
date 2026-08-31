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
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neurons.runtime import is_mesh_quant

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "/tmp/verathos_validator_state.json"
_EVM_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
_HEX_32_RE = re.compile(r"[0-9a-f]{64}")
_MESH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def mesh_snapshot_slot_key(address: str, model_index: int) -> str:
    """Return the stable JSON key for one coordinator/model slot."""

    return f"{str(address).lower()}:{int(model_index)}"


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
    mesh_enabled: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    compute_capability: str = ""
    gpu_uuids: List[str] = field(default_factory=list)
    # On-chain lease expiry (unix seconds) as read at this epoch's discovery.
    # Refreshed on every shared-state write, so renewals are reflected; 0
    # means the lease is unknown, never that it expired.
    expires_at: int = 0

    def __post_init__(self) -> None:
        self.mesh_enabled = bool(self.mesh_enabled) or is_mesh_quant(self.quant)


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

    # Network identity for every score, endpoint, and snapshot in this file.
    # Legacy files leave these unset; security-sensitive consumers must refuse
    # to attribute their scores to a configured validator pool until both
    # values are present and match its immutable chain binding.
    chain_id: Optional[int] = None
    netuid: Optional[int] = None
    epoch_number: int = 0
    epoch_start_block: int = 0
    # Per-model proxy-routing scores: address -> {model_index (str) -> score}.
    # Active zero-EMA entries may have a small routing floor and inactive
    # entries are zeroed; use miner_ema_scores for the unmodified score.
    # Prior format (address -> float) is auto-migrated on read.
    miner_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Raw per-model validator EMA, before proxy-routing substitutions.  Unlike
    # ``miner_scores``, this does not replace active zero-EMA entries with the
    # 0.01 routing floor or force inactive entries to zero.  Operator surfaces
    # use this field when they need to display the score itself honestly.
    miner_ema_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-model score provenance:
    # address -> {model_index (str) -> {
    #   "scored_epochs": non-negative int,
    #   "last_scored_epoch": non-negative int | None,
    #   "score_epoch": latest completed sample epoch,
    #   "latest_completed_ema": EMA at completion,
    #   "ema_scope": "coordinator_model_slot",
    #   "ema_carries_across_topology": true,
    #   "latest_mesh_sample": exact signed topology identity (mesh only),
    # }}.
    # ``scored_epochs == 0`` distinguishes a newly registered, never-scored
    # zero EMA from a real zero produced by validator scoring.  Consumers can
    # compare ``last_scored_epoch`` with ``epoch_number`` when they require
    # evidence that the score was refreshed in the current epoch.
    miner_score_metadata: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )
    # Recent validator-observed decode throughput and TTFT per miner/model.
    # Values are derived from accepted validator-owned receipts; miners never
    # supply these routing or display signals directly.
    miner_tps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    miner_ttft_ms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Miners on probation: address -> list of model_indices.
    # Proxy should exclude these from organic traffic routing.
    probation_miners: Dict[str, List[int]] = field(default_factory=dict)
    # Discovered miner endpoints (proxy fallback when chain RPC is unavailable).
    miner_endpoints: List[MinerEntry] = field(default_factory=list)
    # Validator-authenticated, coordinator-signed and endpoint-free snapshots,
    # keyed by ``mesh_snapshot_slot_key(address, model_index)``.  The proxy
    # independently revalidates the signature, epoch, coordinator identity,
    # freshness, and canonical proof policy before every mesh request.
    mesh_verification_snapshots: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
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
    # Owner-signed hard-audit failures, retained independently of miners so
    # follower validators can consume no-show and invalid-proof outcomes.
    proof_v3_hard_failures: List[dict] = field(default_factory=list)
    # Exact canonical bytes of the owner's latest signed verdict snapshot,
    # hex-encoded for JSON transport. The proxy serves this string verbatim.
    verdict_snapshot: str = ""
    # Bounded immutable snapshots keyed by exact epoch. Followers use this
    # history at close so advancing the latest pointer cannot race their fetch.
    verdict_snapshots: Dict[str, str] = field(default_factory=dict)
    updated_at: float = 0.0


def write_shared_state(
    state: ValidatorSharedState,
    path: str = DEFAULT_STATE_PATH,
) -> None:
    """Atomically write shared state (validator side).

    Uses write-to-tmp + ``os.replace`` so the proxy never reads a partial file.
    """
    data = {
        "chain_id": state.chain_id,
        "netuid": state.netuid,
        "epoch_number": state.epoch_number,
        "epoch_start_block": state.epoch_start_block,
        "miner_scores": state.miner_scores,
        "miner_ema_scores": state.miner_ema_scores,
        "miner_score_metadata": state.miner_score_metadata,
        "miner_tps": state.miner_tps,
        "miner_ttft_ms": state.miner_ttft_ms,
        "probation_miners": state.probation_miners,
        "miner_endpoints": [
            {"address": m.address, "endpoint": m.endpoint,
             "model_id": m.model_id, "model_index": m.model_index,
             "quant": m.quant, "max_context_len": m.max_context_len,
             "uid": m.uid, "hotkey_ss58": m.hotkey_ss58,
             "coldkey_ss58": m.coldkey_ss58,
             "tee_enabled": m.tee_enabled, "tee_platform": m.tee_platform,
             "enclave_public_key": m.enclave_public_key,
             "mesh_enabled": m.mesh_enabled,
             "gpu_name": m.gpu_name, "gpu_count": m.gpu_count,
             "vram_gb": m.vram_gb, "compute_capability": m.compute_capability,
             "gpu_uuids": m.gpu_uuids, "expires_at": m.expires_at}
            for m in state.miner_endpoints
        ],
        "mesh_verification_snapshots": state.mesh_verification_snapshots,
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
        "proof_v3_hard_failures": list(state.proof_v3_hard_failures),
        "verdict_snapshot": str(state.verdict_snapshot or ""),
        "verdict_snapshots": {
            str(epoch): str(snapshot)
            for epoch, snapshot in state.verdict_snapshots.items()
        },
        "updated_at": time.time(),
    }
    # Unique tmp name per write: the validator writes shared state from
    # several threads (epoch loop, re-pin publish, audit drains).  A fixed
    # ".tmp" name let two concurrent writers race — one os.replace consumed
    # the other's tmp file ("[Errno 2] ... .tmp -> ..."
    # 13:44), and worse, could publish a half-written interleaved file.  A
    # per-write name keeps every rename atomic and collision-free.
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
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
        # Migrate old flat scores (address -> float) to per-model format.
        def normalize_scores(
            raw_scores,
            *,
            legacy_invalid_default: bool,
        ) -> Dict[str, Dict[str, float]]:
            normalized: Dict[str, Dict[str, float]] = {}
            if not isinstance(raw_scores, dict):
                return normalized
            for addr, val in raw_scores.items():
                if isinstance(val, (int, float)):
                    normalized[addr] = {"0": float(val)}
                elif isinstance(val, dict):
                    normalized[addr] = val
                elif legacy_invalid_default:
                    normalized[addr] = {"0": 1.0}
            return normalized

        raw_scores = data.get("miner_scores", {})
        miner_scores = normalize_scores(raw_scores, legacy_invalid_default=True)
        # Deliberately do not fall back to miner_scores for older files: that
        # field may contain the active-entry routing floor rather than raw EMA.
        miner_ema_scores = normalize_scores(
            data.get("miner_ema_scores", {}),
            legacy_invalid_default=False,
        )

        def normalize_score_metadata(
            raw_metadata: object,
        ) -> Dict[str, Dict[str, Dict[str, Any]]]:
            normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
            if not isinstance(raw_metadata, dict):
                return normalized
            for address, per_model in raw_metadata.items():
                if not isinstance(per_model, dict):
                    continue
                normalized_models: Dict[str, Dict[str, Any]] = {}
                for model_index, metadata in per_model.items():
                    if not isinstance(metadata, dict):
                        continue
                    scored_epochs = metadata.get("scored_epochs")
                    last_scored_epoch = metadata.get("last_scored_epoch")
                    if (
                        isinstance(scored_epochs, bool)
                        or not isinstance(scored_epochs, int)
                        or scored_epochs < 0
                    ):
                        continue
                    if last_scored_epoch is not None and (
                        isinstance(last_scored_epoch, bool)
                        or not isinstance(last_scored_epoch, int)
                        or last_scored_epoch < 0
                    ):
                        continue
                    normalized_metadata: Dict[str, Any] = {
                        "scored_epochs": scored_epochs,
                        "last_scored_epoch": last_scored_epoch,
                    }
                    score_epoch = metadata.get("score_epoch")
                    completed_ema = metadata.get("latest_completed_ema")
                    if (
                        type(score_epoch) is int
                        and score_epoch >= 0
                        and type(completed_ema) in {int, float}
                        and not isinstance(completed_ema, bool)
                        and float(completed_ema) not in {
                            float("inf"),
                            float("-inf"),
                        }
                        and float(completed_ema) == float(completed_ema)
                        and score_epoch == last_scored_epoch
                        and metadata.get("ema_scope")
                        == "coordinator_model_slot"
                        and metadata.get("ema_carries_across_topology") is True
                    ):
                        normalized_metadata.update(
                            {
                                "score_epoch": score_epoch,
                                "latest_completed_ema": float(completed_ema),
                                "ema_scope": "coordinator_model_slot",
                                "ema_carries_across_topology": True,
                            }
                        )
                        mesh_sample = metadata.get("latest_mesh_sample")
                        expected_mesh_sample_fields = {
                            "chain_id",
                            "netuid",
                            "coordinator_address",
                            "model_index",
                            "model_id",
                            "mesh_id",
                            "verification_snapshot_hash",
                            "snapshot_generation",
                        }
                        if (
                            isinstance(mesh_sample, dict)
                            and set(mesh_sample) == expected_mesh_sample_fields
                            and type(mesh_sample.get("chain_id")) is int
                            and 1 <= mesh_sample["chain_id"] < 2**63
                            and type(mesh_sample.get("netuid")) is int
                            and 0 <= mesh_sample["netuid"] <= 65_535
                            and type(mesh_sample.get("model_index")) is int
                            and mesh_sample["model_index"] >= 0
                            and type(mesh_sample.get("snapshot_generation")) is int
                            and 1 <= mesh_sample["snapshot_generation"] < 2**63
                            and type(mesh_sample.get("coordinator_address")) is str
                            and _EVM_ADDRESS_RE.fullmatch(
                                mesh_sample["coordinator_address"]
                            )
                            and type(mesh_sample.get("model_id")) is str
                            and 0 < len(mesh_sample["model_id"]) <= 256
                            and type(mesh_sample.get("mesh_id")) is str
                            and _MESH_ID_RE.fullmatch(mesh_sample["mesh_id"])
                            and type(
                                mesh_sample.get("verification_snapshot_hash")
                            ) is str
                            and _HEX_32_RE.fullmatch(
                                mesh_sample["verification_snapshot_hash"]
                            )
                        ):
                            normalized_metadata["latest_mesh_sample"] = {
                                "chain_id": mesh_sample["chain_id"],
                                "netuid": mesh_sample["netuid"],
                                "coordinator_address": mesh_sample[
                                    "coordinator_address"
                                ].lower(),
                                "model_index": mesh_sample["model_index"],
                                "model_id": mesh_sample["model_id"],
                                "mesh_id": mesh_sample["mesh_id"],
                                "verification_snapshot_hash": mesh_sample[
                                    "verification_snapshot_hash"
                                ],
                                "snapshot_generation": mesh_sample[
                                    "snapshot_generation"
                                ],
                            }
                    normalized_models[str(model_index)] = normalized_metadata
                if normalized_models:
                    normalized[str(address)] = normalized_models
            return normalized

        def normalize_network_id(
            value: object,
            *,
            minimum: int,
            maximum: int,
        ) -> Optional[int]:
            if type(value) is not int or not minimum <= value <= maximum:
                return None
            return int(value)

        return ValidatorSharedState(
            chain_id=normalize_network_id(
                data.get("chain_id"), minimum=1, maximum=2**64 - 1
            ),
            netuid=normalize_network_id(
                data.get("netuid"), minimum=0, maximum=65_535
            ),
            epoch_number=data.get("epoch_number", 0),
            epoch_start_block=data.get("epoch_start_block", 0),
            miner_scores=miner_scores,
            miner_ema_scores=miner_ema_scores,
            miner_score_metadata=normalize_score_metadata(
                data.get("miner_score_metadata", {})
            ),
            miner_tps={
                str(addr).lower(): {
                    str(index): float(value)
                    for index, value in values.items()
                    if float(value) > 0.0
                }
                for addr, values in (
                    data.get("miner_tps", {})
                    if isinstance(data.get("miner_tps", {}), dict)
                    else {}
                ).items()
                if isinstance(values, dict)
            },
            miner_ttft_ms={
                str(addr).lower(): {
                    str(index): float(value)
                    for index, value in values.items()
                    if float(value) >= 0.0
                }
                for addr, values in (
                    data.get("miner_ttft_ms", {})
                    if isinstance(data.get("miner_ttft_ms", {}), dict)
                    else {}
                ).items()
                if isinstance(values, dict)
            },
            probation_miners=data.get("probation_miners", {}),
            miner_endpoints=miner_endpoints,
            mesh_verification_snapshots={
                str(key): value
                for key, value in data.get(
                    "mesh_verification_snapshots", {}
                ).items()
                if isinstance(value, dict)
            }
            if isinstance(data.get("mesh_verification_snapshots", {}), dict)
            else {},
            audit_drains=audit_drains,
            demand_scores=data.get("demand_scores", {}),
            ss58_map=data.get("ss58_map", {}),
            blacklisted_addresses=data.get("blacklisted_addresses", []),
            stale_miner_addresses=data.get("stale_miner_addresses", []),
            proof_v3_hard_failures=[
                value
                for value in data.get("proof_v3_hard_failures", [])
                if isinstance(value, dict)
            ][:16_384],
            verdict_snapshot=(
                str(data.get("verdict_snapshot") or "")
                if len(str(data.get("verdict_snapshot") or ""))
                <= 64 * 1024 * 1024
                else ""
            ),
            verdict_snapshots={
                str(epoch): str(snapshot)
                for epoch, snapshot in (
                    data.get("verdict_snapshots", {})
                    if isinstance(data.get("verdict_snapshots", {}), dict)
                    else {}
                ).items()
                if (
                    str(epoch).isdigit()
                    and len(str(snapshot)) <= 64 * 1024 * 1024
                )
            },
            updated_at=data.get("updated_at", 0.0),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def write_miner_debug_state(data: Dict[str, Any], path: str) -> None:
    """Atomically write the optional public miner diagnostics cache."""
    # Per-pid tmp name: three validators on one box export to the same path,
    # and a shared ".tmp" raced the atomic rename (ENOENT when a sibling
    # consumed it first).
    tmp_path = f"{path}.tmp.{os.getpid()}"
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
