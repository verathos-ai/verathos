"""Signed GPU roster for mesh capacity audits.

A vLLM endpoint is one process on one GPU, so its capacity-audit obligation
is self-evident: the slot's GPU proves.  A mesh endpoint is one chain entry
served by SEVERAL workers whose GPUs the chain never sees, so the audit
needs a document that pins down exactly which physical GPUs stand behind
the slot.  That document is the roster: the pool manager (which formed the
mesh and holds the coordinator wallet) enumerates every member's GPUs and
signs the enumeration with the same coordinator EVM key the chain entry
registered.  Validators verify the signature against the on-chain address,
so a miner cannot under-declare GPUs to shrink its proof obligation without
the coordinator key holder co-signing the lie — and the VRAM floor check
(declared CUDA VRAM must fit the registered model+context) catches the lie
anyway.

The roster is deliberately NOT part of the verification snapshot: the
snapshot is a strict-schema, endpoint-free document validators pin per
epoch, and coupling audit rollout to snapshot compatibility would force a
lockstep upgrade.  Workers embed the signed roster in their capacity-audit
receipts instead (~120 bytes per GPU).

Ordinals: every CUDA GPU in the roster gets one global ordinal, assigned by
sorting on ``(worker_id, local_gpu_index)``.  The ordinal feeds the existing
``derive_proof_seed(..., gpu_index=ordinal)`` derivations unchanged, so each
GPU's synthetic workload has a distinct, publicly derivable seed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

ROSTER_VERSION = 1
ROSTER_SIGNING_PREFIX = "VERATHOS_CAPACITY_ROSTER_V1"

#: Normalized backend labels. Anything that is not recognisably CUDA is
#: treated as non-auditable (the synthetic workload is CUDA-only); an
#: all-non-CUDA roster makes the slot audit-ineligible rather than failed.
BACKEND_CUDA = "cuda"
BACKEND_METAL = "metal"
BACKEND_OTHER = "other"

_MAX_ROSTER_WORKERS = 64
_MAX_GPUS_PER_WORKER = 16
_MAX_TOKEN_LEN = 128


def normalize_backend(value: object) -> str:
    """Collapse a worker's device string into an auditable backend label."""

    text = str(value or "").strip().lower()
    if text.startswith("cuda"):
        return BACKEND_CUDA
    if text.startswith(("metal", "mps")):
        return BACKEND_METAL
    return BACKEND_OTHER


def host_group_hints(endpoint_or_host: str) -> tuple[str, str]:
    """Return ``(ip24, registered_domain)`` grouping hints for a worker host.

    Mirrors the token derivation in ``neurons.capacity_audit`` so roster
    hints collide with chain-endpoint tokens where they should (same /24 →
    same cohort-stress group), without importing the neurons package into
    the mesh layer.
    """

    raw = str(endpoint_or_host or "").strip()
    if not raw:
        return "", ""
    host = raw
    if "//" in raw:
        host = urlparse(raw).hostname or ""
    else:
        # bare host[:port]
        host = raw.split("/", 1)[0].rsplit(":", 1)[0] if raw.count(":") == 1 else raw
    host = str(host or "").strip().lower().rstrip(".")
    if not host:
        return "", ""
    try:
        ip = ipaddress.ip_address(host)
        if ip.version == 4:
            return ".".join(str(ip).split(".")[:3]), ""
        return "", ""
    except ValueError:
        pass
    if "." not in host:
        return "", ""
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return "", ""
    return "", ".".join(labels[-2:])


@dataclass(frozen=True)
class RosterGpu:
    """One CUDA GPU's place in the roster's global ordinal ordering."""

    ordinal: int
    worker_id: str
    local_gpu_index: int
    gpu_name: str
    vram_gb: int


def build_roster(
    *,
    chain_id: int,
    netuid: int,
    address: str,
    model_index: int,
    roster_epoch: int,
    workers: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the canonical roster document (unsigned).

    Each entry in ``workers`` needs ``worker_id``, ``backend`` (raw device
    text is fine — it is normalized here), ``gpu_names``, ``per_gpu_vram_gb``
    and optionally ``host`` (advertise host or endpoint, reduced to public
    grouping hints; the raw host never enters the roster).
    """

    rows: list[dict[str, Any]] = []
    for worker in workers:
        gpu_names = [str(n or "") for n in (worker.get("gpu_names") or [])]
        per_gpu = [int(v or 0) for v in (worker.get("per_gpu_vram_gb") or [])]
        if len(per_gpu) < len(gpu_names):
            per_gpu = per_gpu + [0] * (len(gpu_names) - len(per_gpu))
        elif len(per_gpu) > len(gpu_names):
            per_gpu = per_gpu[: len(gpu_names)]
        ip24, regdom = host_group_hints(str(worker.get("host", "") or ""))
        rows.append(
            {
                "worker_id": str(worker.get("worker_id", "") or ""),
                "backend": normalize_backend(worker.get("backend")),
                "gpu_names": gpu_names,
                "per_gpu_vram_gb": per_gpu,
                "host_ip24": ip24,
                "host_regdom": regdom,
            }
        )
    rows.sort(key=lambda row: row["worker_id"])
    roster = {
        "version": ROSTER_VERSION,
        "slot": {
            "chain_id": int(chain_id),
            "netuid": int(netuid),
            "address": str(address or "").lower(),
            "model_index": int(model_index),
        },
        "roster_epoch": int(roster_epoch),
        "workers": rows,
    }
    validate_roster(roster)
    return roster


def validate_roster(roster: Mapping[str, Any]) -> None:
    """Reject malformed or oversized roster documents.

    Runs on both sides: the manager before signing, the validator before
    trusting a signature.  Bounds keep a hostile roster from inflating
    receipts or smuggling unbounded strings into the DB.
    """

    if not isinstance(roster, Mapping):
        raise ValueError("roster must be a JSON object")
    if int(roster.get("version", 0) or 0) != ROSTER_VERSION:
        raise ValueError("unsupported roster version")
    slot = roster.get("slot")
    if not isinstance(slot, Mapping):
        raise ValueError("roster.slot must be a JSON object")
    chain_id = slot.get("chain_id")
    netuid = slot.get("netuid")
    model_index = slot.get("model_index")
    for name, value in (("chain_id", chain_id), ("netuid", netuid)):
        if type(value) is not int or value < 0:
            raise ValueError(f"roster.slot.{name} must be a non-negative integer")
    if type(model_index) is not int or model_index < 0:
        raise ValueError("roster.slot.model_index must be a non-negative integer")
    address = str(slot.get("address", "") or "")
    if not address.startswith("0x") or len(address) != 42 or address != address.lower():
        raise ValueError("roster.slot.address must be a lowercase 0x EVM address")
    if type(roster.get("roster_epoch")) is not int or int(roster["roster_epoch"]) < 0:
        raise ValueError("roster.roster_epoch must be a non-negative integer")
    workers = roster.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("roster.workers must be a non-empty list")
    if len(workers) > _MAX_ROSTER_WORKERS:
        raise ValueError("roster declares too many workers")
    seen_ids: set[str] = set()
    for row in workers:
        if not isinstance(row, Mapping):
            raise ValueError("roster worker entries must be JSON objects")
        worker_id = str(row.get("worker_id", "") or "")
        if not worker_id or len(worker_id) > _MAX_TOKEN_LEN:
            raise ValueError("roster worker_id is missing or too long")
        if worker_id in seen_ids:
            raise ValueError("roster worker_id entries must be unique")
        seen_ids.add(worker_id)
        if str(row.get("backend", "") or "") not in (
            BACKEND_CUDA,
            BACKEND_METAL,
            BACKEND_OTHER,
        ):
            raise ValueError("roster worker backend must be normalized")
        gpu_names = row.get("gpu_names")
        per_gpu = row.get("per_gpu_vram_gb")
        if not isinstance(gpu_names, list) or not isinstance(per_gpu, list):
            raise ValueError("roster worker gpu lists must be lists")
        if len(gpu_names) != len(per_gpu):
            raise ValueError("roster worker gpu name/vram lists must align")
        if len(gpu_names) > _MAX_GPUS_PER_WORKER:
            raise ValueError("roster worker declares too many GPUs")
        if str(row.get("backend")) == BACKEND_CUDA and not gpu_names:
            raise ValueError("a CUDA roster worker must declare its GPUs")
        for name in gpu_names:
            if not isinstance(name, str) or len(name) > _MAX_TOKEN_LEN:
                raise ValueError("roster gpu_names entries must be short strings")
        for vram in per_gpu:
            if type(vram) is not int or vram < 0 or vram > 4096:
                raise ValueError("roster per_gpu_vram_gb entries must be sane integers")
        for hint_key in ("host_ip24", "host_regdom"):
            hint = row.get(hint_key, "")
            if not isinstance(hint, str) or len(hint) > _MAX_TOKEN_LEN:
                raise ValueError(f"roster {hint_key} must be a short string")


def canonical_roster_json(roster: Mapping[str, Any]) -> str:
    return json.dumps(
        roster, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def roster_digest(roster: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_roster_json(roster).encode("utf-8")
    ).hexdigest()


def roster_signing_text(roster: Mapping[str, Any]) -> str:
    return ROSTER_SIGNING_PREFIX + "\n" + canonical_roster_json(roster)


def sign_roster(roster: Mapping[str, Any], private_key: str) -> str:
    """EIP-191 signature over the canonical roster, coordinator EVM key."""

    from eth_account import Account
    from eth_account.messages import encode_defunct

    validate_roster(roster)
    signed = Account.sign_message(
        encode_defunct(text=roster_signing_text(roster)),
        private_key=private_key,
    )
    return signed.signature.hex()


def recover_roster_signer(roster: Mapping[str, Any], signature: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    return Account.recover_message(
        encode_defunct(text=roster_signing_text(roster)),
        signature=str(signature or ""),
    )


def verify_roster_signature(
    roster: Mapping[str, Any],
    signature: str,
    expected_address: str,
) -> bool:
    try:
        validate_roster(roster)
        recovered = recover_roster_signer(roster, signature)
    except Exception:
        return False
    return recovered.lower() == str(expected_address or "").lower()


def roster_cuda_gpus(roster: Mapping[str, Any]) -> list[RosterGpu]:
    """Global ordinal assignment over every CUDA GPU in the roster.

    Deterministic from the roster alone: sorted by ``(worker_id,
    local_gpu_index)``, ordinals 0..N-1.  Non-CUDA workers contribute no
    ordinals (the synthetic workload cannot run there); an all-non-CUDA
    roster yields an empty list, which callers treat as audit-ineligible.
    """

    entries: list[tuple[str, int, str, int]] = []
    for row in roster.get("workers") or []:
        if str(row.get("backend", "") or "") != BACKEND_CUDA:
            continue
        worker_id = str(row.get("worker_id", "") or "")
        gpu_names = row.get("gpu_names") or []
        per_gpu = row.get("per_gpu_vram_gb") or []
        for local_index, gpu_name in enumerate(gpu_names):
            vram = int(per_gpu[local_index]) if local_index < len(per_gpu) else 0
            entries.append((worker_id, local_index, str(gpu_name), vram))
    entries.sort(key=lambda item: (item[0], item[1]))
    return [
        RosterGpu(
            ordinal=ordinal,
            worker_id=worker_id,
            local_gpu_index=local_index,
            gpu_name=gpu_name,
            vram_gb=vram,
        )
        for ordinal, (worker_id, local_index, gpu_name, vram) in enumerate(entries)
    ]


def roster_cuda_vram_total_gb(roster: Mapping[str, Any]) -> int:
    return sum(gpu.vram_gb for gpu in roster_cuda_gpus(roster))
