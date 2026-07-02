"""Hot-capacity audit protocol helpers.

The audit protocol is non-interactive after current-head chain data is available:
validators and miners independently derive audit windows, cohorts, seeds, and
proof challenges from public chain data plus registered endpoint metadata.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


PROTOCOL_VERSION = "verathos-capacity-audit-v1"
DEFAULT_MAX_PROOF_PAYLOAD_BYTES = 32 * 1024 * 1024
DEFAULT_BLOCK_TIME_S = 12.0
AUDIT_PREFIX = b"VERATHOS_AUDIT_V1"
WINDOW_TRIGGER_PREFIX = b"VERATHOS_CAPACITY_WINDOW_TRIGGER_V1"
LEASE_PREFIX = b"VERATHOS_CAPACITY_LEASE_V1"
PROOF_SEED_PREFIX = b"VERATHOS_HOT_CAPACITY_SEED_V1"
TRANSCRIPT_PREFIX = b"VERATHOS_HOT_CAPACITY_TRANSCRIPT_V1"
PROOF_PASS_PREFIX = b"VERATHOS_HOT_CAPACITY_PROOF_PASS_V1"
PROOF_CHALLENGE_PREFIX = b"VERATHOS_HOT_CAPACITY_CHALLENGE_V1"
SIGNING_PREFIX = "VERATHOS_CAPACITY_AUDIT_ARTIFACT_V1"


# Deterministic fallback for common multi-label public suffixes.  Do not fetch
# public suffix data at runtime: miners and validators must derive identical
# cohorts from the same endpoint metadata.
_MULTI_LABEL_PUBLIC_SUFFIXES = {
    "ac.jp",
    "ac.uk",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.uk",
    "com.ar",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.mx",
    "com.sg",
    "com.tr",
    "com.tw",
    "gov.uk",
    "ne.jp",
    "net.au",
    "net.cn",
    "net.nz",
    "net.uk",
    "org.au",
    "org.cn",
    "org.nz",
    "org.uk",
}

_PUBLIC_GROUP_PREFIXES = (
    "addr:",
    "host:",
    "ip:",
    "ip24:",
    "regdom:",
    "addr_model:",
    "host_model:",
    "ip_model:",
    "ip24_model:",
    "regdom_model:",
)


@dataclass(frozen=True)
class CapacityGpuClass:
    """Validator-controlled workload row for one calibrated GPU class."""

    match_gpu_name: str
    vram_gb: int = 0
    workload_version: str = "hot_capacity_combined"
    arena_mb: int = 500
    matrix_dim: int = 8960
    rounds_per_pass: int = 8
    memory_mix_rounds: int = 0
    passes: int = 0
    capacity_passes: int = 0
    capacity_rounds: int = 1
    capacity_warmup_passes: int = 1
    capacity_block_size: int = 64
    transition_mix_rounds: int = 0
    transition_fanout: int = 1
    capacity_tail_passes: int = 0
    capacity_tail_rounds: int = 1
    capacity_tail_warmup_passes: int = 1
    capacity_tail_transition_mix_rounds: int = 1
    capacity_tail_transition_fanout: int = 8
    fp64_matrix_dim: int = 4096
    fp64_passes: int = 0
    fp64_rounds: int = 1
    fp64_warmup_passes: int = 1
    fp64_block_size: int = 64
    spot_checks: int = 1
    deadline_s: float = 30.0
    mode: str = "observe"
    calibrated: bool = False


DEFAULT_GPU_CLASSES: tuple[CapacityGpuClass, ...] = (
    CapacityGpuClass(
        "NVIDIA GeForce RTX 4090",
        24,
        passes=222,
        capacity_passes=222,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA GeForce RTX 3090",
        24,
        passes=90,
        capacity_passes=90,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA GeForce RTX 3090 Ti",
        24,
        passes=115,
        capacity_passes=115,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA GeForce RTX 5090",
        32,
        passes=337,
        capacity_passes=337,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX 4000 Ada Generation",
        20,
        passes=62,
        capacity_passes=62,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX A4500",
        20,
        passes=53,
        capacity_passes=53,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX A5000",
        24,
        passes=69,
        capacity_passes=69,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX 5000 Ada Generation",
        32,
        passes=144,
        capacity_passes=144,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA L4",
        23,
        passes=45,
        capacity_passes=45,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA A40",
        46,
        passes=93,
        capacity_passes=93,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX A6000",
        49,
        passes=96,
        capacity_passes=96,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA A100-SXM4-40GB",
        40,
        passes=674,
        capacity_passes=6,
        capacity_tail_passes=14,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=654,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA A100 80GB PCIe",
        80,
        passes=699,
        capacity_passes=9,
        capacity_tail_passes=9,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=681,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA A100-SXM4-80GB",
        80,
        passes=662,
        capacity_passes=8,
        capacity_tail_passes=9,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=645,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA L40",
        46,
        passes=150,
        capacity_passes=150,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA L40S",
        46,
        passes=196,
        capacity_passes=196,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX 6000 Ada Generation",
        49,
        passes=156,
        capacity_passes=156,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX PRO 4000 Blackwell",
        24,
        passes=105,
        capacity_passes=105,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX PRO 4500 Blackwell",
        32,
        passes=154,
        capacity_passes=154,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX PRO 5000 Blackwell",
        48,
        passes=195,
        capacity_passes=195,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA H100 80GB HBM3",
        80,
        passes=1649,
        capacity_passes=16,
        capacity_tail_passes=4,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=1629,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA H100 PCIe",
        80,
        passes=1345,
        capacity_passes=9,
        capacity_tail_passes=1,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=1335,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA H100 NVL",
        96,
        passes=1480,
        capacity_passes=14,
        capacity_tail_passes=4,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=1462,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA H200",
        144,
        passes=1661,
        capacity_passes=20,
        capacity_tail_passes=12,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=1629,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        98,
        passes=311,
        capacity_passes=311,
        capacity_rounds=8,
        calibrated=True,
    ),
    CapacityGpuClass(
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        98,
        passes=393,
        capacity_passes=393,
        capacity_rounds=8,
        calibrated=False,
    ),
    CapacityGpuClass(
        "NVIDIA B200",
        183,
        passes=1154,
        capacity_passes=43,
        capacity_tail_passes=1,
        transition_mix_rounds=3,
        transition_fanout=64,
        fp64_passes=1110,
        calibrated=True,
    ),
)


@dataclass(frozen=True)
class CapacityAuditRuntimeConfig:
    """Runtime policy for validator-side audit scheduling and enforcement."""

    enabled: bool = False
    mode: str = "observe"  # observe | score_gate | soft_gate | enforce
    cohort_min: int = 100
    cohort_fraction: float = 0.025
    cohort_max: int = 250
    windows_per_epoch: int = 30
    max_drain_fraction: float = 0.05
    group_stress_fraction: float = 0.35
    group_stress_min_size: int = 2
    group_stress_slots_per_group: int = 2
    beacon_hash_count: int = 1
    min_registration_age_s: float = 0.0
    lead_blocks: int = 5
    proof_challenge_delay_blocks: int = 4
    drain_seconds: float = 30.0
    deadline_s: float = 30.0
    transport_grace_s: float = 3.0
    payload_deadline_s: float = 60.0
    max_proof_payload_bytes: int = DEFAULT_MAX_PROOF_PAYLOAD_BYTES
    require_proof_payload: bool = True
    repeat_window_epochs: int = 20
    timing_misses_for_zero_score: int = 2
    hard_proof_misses_for_zero_score: int = 2
    allow_timing_only_score_gate: bool = True
    validator_urls: tuple[str, ...] = ()
    gpu_classes: tuple[CapacityGpuClass, ...] = DEFAULT_GPU_CLASSES


def validate_capacity_audit_runtime_config(
    cfg: CapacityAuditRuntimeConfig,
    *,
    block_time_s: float = DEFAULT_BLOCK_TIME_S,
) -> None:
    group_fraction = float(getattr(cfg, "group_stress_fraction", 0.0) or 0.0)
    if group_fraction < 0.0 or group_fraction > 0.5:
        raise ValueError("capacity audit group_stress_fraction must be between 0.0 and 0.5")
    proof_blocks = max(1, int(getattr(cfg, "proof_challenge_delay_blocks", 1) or 1))
    proof_delay_s = proof_blocks * max(0.001, float(block_time_s or DEFAULT_BLOCK_TIME_S))
    receipt_deadline_s = float(getattr(cfg, "deadline_s", 0.0) or 0.0) + float(
        getattr(cfg, "transport_grace_s", 0.0) or 0.0
    )
    if receipt_deadline_s >= proof_delay_s:
        raise ValueError(
            "capacity audit deadline_s + transport_grace_s must be smaller than "
            "proof_challenge_delay_blocks * block_time"
        )
    max_payload = int(getattr(cfg, "max_proof_payload_bytes", 0) or 0)
    if max_payload <= 0:
        raise ValueError("capacity audit max_proof_payload_bytes must be positive")


@dataclass(frozen=True)
class CapacitySlot:
    """Endpoint slot selected for capacity audit."""

    chain_id: int
    netuid: int
    address: str
    model_index: int
    endpoint: str
    model_id: str
    quant: str = ""
    max_context_len: int = 0
    miner_uid: int | None = None
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    group_key: str = ""

    @property
    def address_lower(self) -> str:
        return self.address.lower()


@dataclass(frozen=True)
class CapacityWindow:
    audit_id: str
    epoch_number: int
    selection_block: int
    audit_block: int
    selection_block_hash: str
    audit_block_hash: str = ""
    cohort_seed: str = ""
    created_at: float = field(default_factory=time.time)


def _bytes_from_hash(value: bytes | str) -> bytes:
    if isinstance(value, bytes):
        return value
    raw = str(value).strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    try:
        return bytes.fromhex(raw)
    except ValueError:
        return raw.encode("utf-8")


def _hash_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _u64(value: int) -> bytes:
    return struct.pack(">Q", int(value))


def _i64(value: int) -> bytes:
    return struct.pack(">q", int(value))


def canonical_json(data: Mapping[str, Any], *, exclude_signature: bool = False) -> str:
    payload = dict(data)
    if exclude_signature:
        payload.pop("miner_signature", None)
        payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def slot_id(slot: CapacitySlot | Mapping[str, Any]) -> str:
    if isinstance(slot, CapacitySlot):
        chain_id = slot.chain_id
        netuid = slot.netuid
        address = slot.address_lower
        model_index = slot.model_index
    else:
        chain_id = int(slot.get("chain_id", 0))
        netuid = int(slot.get("netuid", 0))
        address = str(slot.get("address") or slot.get("miner_address") or "").lower()
        model_index = int(slot.get("model_index", 0))
    return _hash_hex(
        b"VERATHOS_CAPACITY_SLOT_V1",
        _u64(chain_id),
        _u64(netuid),
        address.encode("utf-8"),
        _i64(model_index),
    )


def lease_id(slot: CapacitySlot, audit_epoch: int) -> str:
    return _hash_hex(
        LEASE_PREFIX,
        _u64(slot.chain_id),
        _u64(slot.netuid),
        slot.address_lower.encode("utf-8"),
        _i64(slot.model_index),
        slot.endpoint.encode("utf-8"),
        slot.model_id.encode("utf-8"),
        slot.quant.encode("utf-8"),
        _u64(slot.max_context_len),
        _u64(audit_epoch),
    )


def derive_audit_seed(selection_block_hash: bytes | str, subnet_epoch: int) -> str:
    return _hash_hex(AUDIT_PREFIX, _bytes_from_hash(selection_block_hash), _u64(subnet_epoch))


def derive_audit_seed_from_hashes(
    selection_block_hashes: Sequence[bytes | str],
    subnet_epoch: int,
) -> str:
    """Derive the public audit beacon from one or more finalized block hashes."""
    parts = [AUDIT_PREFIX, _u64(subnet_epoch), _u64(len(selection_block_hashes))]
    for value in selection_block_hashes:
        parts.append(_bytes_from_hash(value))
    return _hash_hex(*parts)


def derive_audit_id(
    *,
    chain_id: int,
    netuid: int,
    epoch_number: int,
    selection_block: int,
    audit_block: int,
    cohort_seed: str,
) -> str:
    return _hash_hex(
        b"VERATHOS_CAPACITY_AUDIT_ID_V1",
        _u64(chain_id),
        _u64(netuid),
        _u64(epoch_number),
        _u64(selection_block),
        _u64(audit_block),
        _bytes_from_hash(cohort_seed),
    )


def derive_proof_seed(
    audit_block_hash: bytes | str,
    slot_identifier: str,
    gpu_index: int,
) -> str:
    return _hash_hex(
        PROOF_SEED_PREFIX,
        _bytes_from_hash(audit_block_hash),
        _bytes_from_hash(slot_identifier),
        _u64(gpu_index),
    )


def derive_proof_challenge_seed(
    transcript_root_hex: bytes | str,
    challenge_block_hash: bytes | str,
    lease_identifier: str,
    slot_identifier: str,
    gpu_index: int,
) -> str:
    """Derive the public post-commit proof challenge seed.

    The timed final receipt commits ``transcript_root_hex`` before the future
    challenge block hash is known. The proof sample must be drawn from this
    seed, not from the miner-controlled transcript alone.
    """
    return _hash_hex(
        PROOF_CHALLENGE_PREFIX,
        _bytes_from_hash(transcript_root_hex),
        _bytes_from_hash(challenge_block_hash),
        lease_identifier.encode("utf-8"),
        _bytes_from_hash(slot_identifier),
        _u64(gpu_index),
    )


def transcript_root(pass_roots: Sequence[Any]) -> str:
    h = hashlib.sha256()
    h.update(TRANSCRIPT_PREFIX)
    for idx, root in enumerate(pass_roots):
        h.update(_u64(idx))
        if isinstance(root, (list, tuple)):
            h.update(canonical_json({"root": list(root)}).encode("utf-8"))
        else:
            h.update(str(root).encode("utf-8"))
    return h.hexdigest()


def root_words_digest(root_words: Any) -> str:
    """Digest one compact CUDA root vector exactly as miner artifacts encode it."""
    payload = json.dumps(root_words, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_sampled_pass_index(root_hex: str, lease_identifier: str, gpu_index: int, passes: int) -> int:
    if passes <= 0:
        return 0
    digest = hashlib.sha256(
        PROOF_PASS_PREFIX
        + _bytes_from_hash(root_hex)
        + lease_identifier.encode("utf-8")
        + _u64(gpu_index)
    ).digest()
    return int.from_bytes(digest[:8], "big") % passes


def deterministic_sample_slots(
    slots: Iterable[CapacitySlot],
    audit_seed_hex: str,
    budget: int,
) -> list[CapacitySlot]:
    ordered = sorted(slots, key=lambda s: (s.address_lower, s.model_index, s.endpoint))
    if budget <= 0 or not ordered:
        return []
    if budget >= len(ordered):
        return ordered
    rng = random.Random(int(audit_seed_hex[:16], 16))
    return sorted(rng.sample(ordered, budget), key=lambda s: (s.address_lower, s.model_index))


def cohort_budget(endpoint_count: int, cfg: CapacityAuditRuntimeConfig) -> int:
    if endpoint_count <= 0:
        return 0
    return min(
        int(cfg.cohort_max),
        max(int(cfg.cohort_min), int(endpoint_count * float(cfg.cohort_fraction))),
    )


def window_cohort_budget(endpoint_count: int, cfg: CapacityAuditRuntimeConfig) -> int:
    """Exact per-window budget while preserving the configured per-epoch target."""
    epoch_budget = cohort_budget(endpoint_count, cfg)
    if epoch_budget <= 0:
        return 0
    windows = max(1, int(cfg.windows_per_epoch or 1))
    per_window = max(1, int(math.ceil(epoch_budget / windows)))
    max_drain_fraction = float(cfg.max_drain_fraction or 0.0)
    if max_drain_fraction > 0:
        drain_cap = max(1, int(math.floor(endpoint_count * max_drain_fraction)))
        per_window = min(per_window, drain_cap)
    return min(epoch_budget, per_window, endpoint_count)


def capacity_audit_window_triggered(
    block_number: int,
    selection_block_hash: bytes | str,
    epoch_blocks: int,
    cfg: CapacityAuditRuntimeConfig,
) -> bool:
    """Return True when this current-head block opens a surprise audit window.

    ``windows_per_epoch`` is a target rate, not a fixed offset schedule. Every
    current-head block hash can become ``B_select``; before the head is observed,
    neither validators nor miners can know whether that block will trigger.
    """
    epoch_blocks = max(1, int(epoch_blocks or 1))
    windows = max(0, int(cfg.windows_per_epoch or 0))
    if windows <= 0:
        return False
    if windows >= epoch_blocks:
        return True

    digest = hashlib.sha256(
        WINDOW_TRIGGER_PREFIX
        + _bytes_from_hash(selection_block_hash)
        + _u64(block_number)
        + _u64(epoch_blocks)
        + _u64(windows)
    ).digest()
    sample = int.from_bytes(digest[:16], "big")
    threshold = ((1 << 128) * windows) // epoch_blocks
    return sample < threshold


def capacity_audit_window_fits_epoch(
    selection_block: int,
    epoch_blocks: int,
    cfg: CapacityAuditRuntimeConfig,
) -> bool:
    """Return True when all public audit timing blocks stay in one tempo.

    Late surprise selections are ignored instead of carried into the next
    epoch. Carrying would give the selected miner advance notice for a later
    tempo and would mix audit evidence across score epochs.
    """
    epoch_blocks = max(1, int(epoch_blocks or 1))
    selection_block = int(selection_block)
    epoch_end_exclusive = ((selection_block // epoch_blocks) + 1) * epoch_blocks
    audit_block = selection_block + int(cfg.lead_blocks or 0)
    proof_challenge_block = audit_block + max(
        1,
        int(cfg.proof_challenge_delay_blocks or 1),
    )
    return audit_block < epoch_end_exclusive and proof_challenge_block < epoch_end_exclusive


def build_capacity_slot_group_key(
    *,
    address: str,
    endpoint: str,
    model_id: str,
    gpu_name: str = "",
    miner_uid: int | None = None,
) -> str:
    """Build public-only grouping tokens from chain-advertised slot metadata.

    ``gpu_name`` and ``miner_uid`` are accepted for backwards-compatible call
    sites, but they are intentionally ignored. Cohort grouping must not depend
    on miner-reported health data or validator-local UID cache state.
    """
    parsed = urlparse(endpoint or "")
    host = (parsed.hostname or "").lower()
    ip_literal = ""
    ip24 = ""
    regdom = registered_domain_from_host(host)
    if host:
        try:
            ip = ipaddress.ip_address(host)
            if ip.version == 4:
                ip_literal = str(ip)
                ip24 = ".".join(ip_literal.split(".")[:3])
        except ValueError:
            pass
    model = _group_token_value(model_id)
    addr = _group_token_value(str(address or "").lower())
    tokens = [
        f"addr:{addr}" if addr else "",
        f"host:{_group_token_value(host)}" if host else "",
        f"ip:{ip_literal}" if ip_literal else "",
        f"ip24:{ip24}" if ip24 else "",
        f"regdom:{_group_token_value(regdom)}" if regdom else "",
        f"addr_model:{addr}::{model}" if addr and model else "",
        f"host_model:{_group_token_value(host)}::{model}" if host and model else "",
        f"ip_model:{ip_literal}::{model}" if ip_literal and model else "",
        f"ip24_model:{ip24}::{model}" if ip24 and model else "",
        f"regdom_model:{_group_token_value(regdom)}::{model}" if regdom and model else "",
    ]
    return "|".join(t for t in tokens if t)


def _group_token_value(value: str) -> str:
    return str(value or "").strip().lower().replace("|", "%7c")


def _public_group_tokens_from_metadata(
    *,
    address: str,
    endpoint: str,
    model_id: str,
) -> set[str]:
    key = build_capacity_slot_group_key(
        address=address,
        endpoint=endpoint,
        model_id=model_id,
    )
    return {
        token
        for token in key.split("|")
        if token and token.startswith(_PUBLIC_GROUP_PREFIXES)
    }


def _is_public_group_token(token: str) -> bool:
    return bool(token) and token.startswith(_PUBLIC_GROUP_PREFIXES)


def registered_domain_from_host(host: str) -> str:
    """Return a deterministic registered-domain grouping token for a hostname.

    This intentionally avoids live DNS and runtime public-suffix updates.  It is
    a scheduling hint, not evidence: exact host/IP groups remain separate.
    """
    normalized = (host or "").strip().lower().rstrip(".")
    if not normalized or "." not in normalized:
        return ""
    try:
        ipaddress.ip_address(normalized)
        return ""
    except ValueError:
        pass
    try:
        ascii_host = normalized.encode("idna").decode("ascii")
    except UnicodeError:
        ascii_host = normalized
    labels = [label for label in ascii_host.split(".") if label]
    if len(labels) < 2 or any(label.startswith("-") or label.endswith("-") for label in labels):
        return ""
    suffix2 = ".".join(labels[-2:])
    if suffix2 in _MULTI_LABEL_PUBLIC_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix2


def capacity_slot_group_keys(slot: CapacitySlot) -> tuple[str, ...]:
    tokens = {
        token.strip()
        for token in str(slot.group_key or "").split("|")
        if _is_public_group_token(token.strip())
    }
    tokens.update(
        _public_group_tokens_from_metadata(
            address=slot.address_lower,
            endpoint=slot.endpoint,
            model_id=slot.model_id,
        )
    )
    return tuple(sorted(tokens))


def _selection_probability(cfg: CapacityAuditRuntimeConfig) -> float:
    windows = max(1, int(cfg.windows_per_epoch or 1))
    cohort_fraction = max(0.0, float(cfg.cohort_fraction or 0.0))
    target = cohort_fraction / windows
    drain = max(0.0, float(cfg.max_drain_fraction or 0.0))
    if drain > 0:
        target = min(target, drain)
    return min(1.0, max(0.0, target))


def _selection_score_hex(*parts: str) -> str:
    h = hashlib.sha256()
    h.update(b"VERATHOS_CAPACITY_SELECTION_SCORE_V1")
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _selection_hit(audit_seed_hex: str, namespace: str, key: str, probability: float) -> bool:
    p = min(1.0, max(0.0, float(probability or 0.0)))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    score = int(_selection_score_hex(audit_seed_hex, namespace, key), 16)
    return score < int(p * (1 << 256))


def capacity_audit_slot_selected(
    slot: CapacitySlot,
    audit_seed_hex: str,
    cfg: CapacityAuditRuntimeConfig,
) -> bool:
    """Return whether one endpoint slot is selected without global state."""
    target = _selection_probability(cfg)
    if target <= 0.0:
        return False
    group_fraction = min(1.0, max(0.0, float(cfg.group_stress_fraction or 0.0)))
    base_p = target * (1.0 - group_fraction)
    if _selection_hit(audit_seed_hex, "base", slot_id(slot), base_p):
        return True

    keys = capacity_slot_group_keys(slot)
    if not keys or group_fraction <= 0.0:
        return False
    per_key_p = min(1.0, target * group_fraction)
    return any(
        _selection_hit(audit_seed_hex, "group", key, per_key_p)
        for key in keys
    )


def select_capacity_audit_slots(
    slots: Iterable[CapacitySlot],
    audit_seed_hex: str,
    cfg: CapacityAuditRuntimeConfig,
) -> list[CapacitySlot]:
    """Select slots by per-slot public hash predicates.

    The selection is deterministic from the public seed plus each endpoint's
    chain-advertised metadata. It intentionally does not require miners to know
    the global miner set or exact cohort budget.
    """
    unique = {slot_id(slot): slot for slot in slots}
    selected = [
        slot for slot in unique.values()
        if capacity_audit_slot_selected(slot, audit_seed_hex, cfg)
    ]
    return sorted(selected, key=lambda s: (s.address_lower, s.model_index, s.endpoint))


def match_gpu_class(gpu_name: str, vram_gb: int, cfg: CapacityAuditRuntimeConfig) -> CapacityGpuClass | None:
    normalized = (gpu_name or "").strip().lower()
    for row in cfg.gpu_classes:
        if row.match_gpu_name.strip().lower() != normalized:
            continue
        if row.vram_gb and vram_gb and abs(int(vram_gb) - int(row.vram_gb)) > 2:
            continue
        return row
    return None


def calibrated_gpu_class_names(cfg: CapacityAuditRuntimeConfig | None = None) -> tuple[str, ...]:
    runtime_cfg = cfg or CapacityAuditRuntimeConfig()
    return tuple(row.match_gpu_name for row in runtime_cfg.gpu_classes if row.calibrated)


def capacity_audit_gpu_support_status(
    gpu_name: str,
    vram_gb: int,
    cfg: CapacityAuditRuntimeConfig | None = None,
) -> tuple[bool, str, CapacityGpuClass | None]:
    runtime_cfg = cfg or CapacityAuditRuntimeConfig()
    row = match_gpu_class(gpu_name, vram_gb, runtime_cfg)
    if row is None:
        return False, "unsupported_gpu_class", None
    if not row.calibrated:
        return False, "uncalibrated_gpu_class", row
    return True, "supported", row


def capacity_gpu_pass_count(row: CapacityGpuClass) -> int:
    if str(getattr(row, "workload_version", "") or "") == "hot_capacity_combined":
        total = (
            int(getattr(row, "capacity_passes", 0) or 0)
            + int(getattr(row, "capacity_tail_passes", 0) or 0)
            + int(getattr(row, "fp64_passes", 0) or 0)
        )
        if total > 0:
            return total
    return int(getattr(row, "passes", 0) or 0)


def capacity_gpu_workload_spec(row: CapacityGpuClass) -> dict[str, Any]:
    """Return the miner CLI/runtime spec for a calibrated GPU row."""
    return {
        "workload_version": str(row.workload_version or "hot_capacity_combined"),
        "capacity_matrix_dim": int(row.matrix_dim or 8960),
        "capacity_passes": int(row.capacity_passes or row.passes or 0),
        "capacity_rounds": int(row.capacity_rounds or 1),
        "capacity_warmup_passes": int(row.capacity_warmup_passes or 0),
        "capacity_block_size": int(row.capacity_block_size or 64),
        "transition_mix_rounds": int(row.transition_mix_rounds),
        "transition_fanout": int(row.transition_fanout or 1),
        "capacity_tail_passes": int(row.capacity_tail_passes or 0),
        "capacity_tail_rounds": int(row.capacity_tail_rounds or 1),
        "capacity_tail_warmup_passes": int(row.capacity_tail_warmup_passes or 0),
        "capacity_tail_transition_mix_rounds": int(row.capacity_tail_transition_mix_rounds or 1),
        "capacity_tail_transition_fanout": int(row.capacity_tail_transition_fanout or 1),
        "fp64_matrix_dim": int(row.fp64_matrix_dim or 4096),
        "fp64_passes": int(row.fp64_passes or 0),
        "fp64_rounds": int(row.fp64_rounds or 1),
        "fp64_warmup_passes": int(row.fp64_warmup_passes or 0),
        "fp64_block_size": int(row.fp64_block_size or 64),
        "spot_checks": int(row.spot_checks or 1),
        "pass_count": capacity_gpu_pass_count(row),
    }


def signing_message(data: Mapping[str, Any]) -> str:
    return SIGNING_PREFIX + "\n" + canonical_json(data, exclude_signature=True)


def sign_artifact(data: Mapping[str, Any], private_key: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = encode_defunct(text=signing_message(data))
    signed = Account.sign_message(msg, private_key=private_key)
    return signed.signature.hex()


def recover_artifact_signer(data: Mapping[str, Any]) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    sig = str(data.get("miner_signature") or data.get("signature") or "")
    if not sig:
        raise ValueError("missing miner_signature")
    msg = encode_defunct(text=signing_message(data))
    return Account.recover_message(msg, signature=sig)


def verify_artifact_signature(data: Mapping[str, Any], expected_address: str) -> bool:
    try:
        recovered = recover_artifact_signer(data)
    except Exception:
        return False
    return recovered.lower() == expected_address.lower()
