"""Public runtime subnet config for validator/miner tuning parameters."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from neurons.capacity_audit import (
    DEFAULT_GPU_CLASSES,
    CapacityAuditRuntimeConfig,
    CapacityGpuClass,
    validate_capacity_audit_runtime_config,
)
from verallm.chain.types import ScoringParams


DEFAULT_SUBNET_CONFIG_URL = "https://api.verathos.ai/v1/subnet-config"
DEFAULT_SUBNET_CONFIG_PATH = "/opt/verathos/config/subnet_config.json"
DEFAULT_SUBNET_CONFIG_REFRESH_SECONDS = 120.0
DEFAULT_SUBNET_CONFIG_TIMEOUT_SECONDS = 5.0
SUBNET_CONFIG_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


class SubnetRuntimeConfigError(ValueError):
    """Raised when a hosted subnet config is incomplete or invalid."""


@dataclass(frozen=True)
class MaintenanceGraceConfig:
    enabled: bool = False
    until_epoch: Optional[int] = None
    until_unix_ts: Optional[int] = None
    reason: str = ""
    suppress_score_zeroing: bool = True
    suppress_probation: bool = True
    suppress_capacity_score_gate: bool = True
    suppress_report_offline: bool = True
    suppress_proxy_proof_strikes: bool = True


@dataclass(frozen=True)
class RuntimeSubnetConfig:
    schema_version: int
    version: int
    effective_epoch: Optional[int]
    updated_at: str
    refresh_seconds: float
    scoring: ScoringParams
    probation_escalation_epochs: int
    demand_bonus_enabled: bool
    epoch_blocks: int
    epoch_grace_blocks: int
    set_weights_epoch_blocks: int
    canary_small_count: int
    canary_full_context_count: int
    canary_inference_timeout_s: float
    canary_full_context_inference_timeout_s: float
    epoch_receipt_pull_timeout_s: float
    epoch_receipt_pull_overall_timeout_s: float
    capacity_audit: CapacityAuditRuntimeConfig
    capacity_audit_worker_poll_s: float
    capacity_audit_slot_refresh_blocks: int
    capacity_audit_slot_snapshot_stale_blocks: int
    capacity_audit_proof_verify_workers: int
    maintenance_grace: MaintenanceGraceConfig
    payload: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    source: str = ""

    @property
    def cache_key(self) -> tuple[int, Optional[int], str]:
        return (self.version, self.effective_epoch, self.source)


def _data_dir_cache_path() -> Path:
    explicit = os.environ.get("VERATHOS_SUBNET_CONFIG_CACHE_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    data_dir = (
        os.environ.get("VERATHOS_DATA_DIR", "").strip()
        or os.environ.get("VERALLM_DATA_DIR", "").strip()
    )
    if data_dir:
        return Path(data_dir).expanduser() / "subnet_config_cache.json"
    return Path.home() / ".verathos" / "subnet_config_cache.json"


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise SubnetRuntimeConfigError(f"missing or invalid section: {key}")
    return value


def _reject_bool(value: Any, key: str) -> None:
    if isinstance(value, bool):
        raise SubnetRuntimeConfigError(f"{key} must not be boolean")


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in data:
        raise SubnetRuntimeConfigError(f"missing required field: {key}")
    value = data[key]
    _reject_bool(value, key)
    if not isinstance(value, int):
        raise SubnetRuntimeConfigError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise SubnetRuntimeConfigError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise SubnetRuntimeConfigError(f"{key} must be <= {maximum}")
    return int(value)


def _require_optional_int(
    data: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if key not in data:
        raise SubnetRuntimeConfigError(f"missing required field: {key}")
    value = data[key]
    if value is None:
        return None
    _reject_bool(value, key)
    if not isinstance(value, int):
        raise SubnetRuntimeConfigError(f"{key} must be an integer or null")
    if minimum is not None and value < minimum:
        raise SubnetRuntimeConfigError(f"{key} must be >= {minimum}")
    return int(value)


def _require_float(
    data: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in data:
        raise SubnetRuntimeConfigError(f"missing required field: {key}")
    value = data[key]
    _reject_bool(value, key)
    if not isinstance(value, (int, float)):
        raise SubnetRuntimeConfigError(f"{key} must be numeric")
    out = float(value)
    if minimum is not None and out < minimum:
        raise SubnetRuntimeConfigError(f"{key} must be >= {minimum}")
    if maximum is not None and out > maximum:
        raise SubnetRuntimeConfigError(f"{key} must be <= {maximum}")
    return out


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    if key not in data:
        raise SubnetRuntimeConfigError(f"missing required field: {key}")
    value = data[key]
    if not isinstance(value, bool):
        raise SubnetRuntimeConfigError(f"{key} must be boolean")
    return bool(value)


def _require_str(
    data: Mapping[str, Any],
    key: str,
    *,
    allowed: set[str] | None = None,
    nonempty: bool = False,
) -> str:
    if key not in data:
        raise SubnetRuntimeConfigError(f"missing required field: {key}")
    value = data[key]
    if not isinstance(value, str):
        raise SubnetRuntimeConfigError(f"{key} must be a string")
    out = value.strip() if nonempty else value
    if nonempty and not out:
        raise SubnetRuntimeConfigError(f"{key} must be non-empty")
    if allowed is not None and out not in allowed:
        raise SubnetRuntimeConfigError(f"{key} must be one of {sorted(allowed)}")
    return out


def _optional_bool(
    data: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    if key not in data:
        return bool(default)
    value = data[key]
    if not isinstance(value, bool):
        raise SubnetRuntimeConfigError(f"{key} must be boolean")
    return bool(value)


def _optional_nullable_int(
    data: Mapping[str, Any],
    key: str,
    default: int | None,
    *,
    minimum: int | None = None,
) -> int | None:
    if key not in data:
        return default
    value = data[key]
    if value is None:
        return None
    _reject_bool(value, key)
    if not isinstance(value, int):
        raise SubnetRuntimeConfigError(f"{key} must be an integer or null")
    if minimum is not None and value < minimum:
        raise SubnetRuntimeConfigError(f"{key} must be >= {minimum}")
    return int(value)


def _optional_str(
    data: Mapping[str, Any],
    key: str,
    default: str,
    *,
    maximum_length: int | None = None,
) -> str:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str):
        raise SubnetRuntimeConfigError(f"{key} must be a string")
    out = value.strip()
    if maximum_length is not None and len(out) > maximum_length:
        raise SubnetRuntimeConfigError(f"{key} must be <= {maximum_length} characters")
    return out


def _gpu_class_to_dict(row: CapacityGpuClass) -> dict[str, Any]:
    return dict(asdict(row))


def _parse_gpu_class(data: Mapping[str, Any], index: int) -> CapacityGpuClass:
    if not isinstance(data, Mapping):
        raise SubnetRuntimeConfigError(f"gpu_classes[{index}] must be an object")
    values: dict[str, Any] = {}
    for f in fields(CapacityGpuClass):
        key = f.name
        label = f"gpu_classes[{index}].{key}"
        if key in {"match_gpu_name", "workload_version", "mode"}:
            values[key] = _require_str(data, key, nonempty=(key == "match_gpu_name"))
        elif key == "calibrated":
            values[key] = _require_bool(data, key)
        elif key == "deadline_s":
            values[key] = _require_float(data, key, minimum=0.001)
        else:
            values[key] = _require_int(data, key, minimum=0)
        if label and key not in data:
            raise SubnetRuntimeConfigError(f"missing required field: {label}")
    return CapacityGpuClass(**values)


def _parse_gpu_classes(data: Mapping[str, Any]) -> tuple[CapacityGpuClass, ...]:
    raw = data.get("gpu_classes")
    if not isinstance(raw, list) or not raw:
        raise SubnetRuntimeConfigError("capacity_audit.gpu_classes must be a non-empty list")
    return tuple(_parse_gpu_class(row, idx) for idx, row in enumerate(raw))


def _maintenance_grace_to_dict(row: MaintenanceGraceConfig) -> dict[str, Any]:
    return dict(asdict(row))


def _parse_maintenance_grace(payload: Mapping[str, Any]) -> MaintenanceGraceConfig:
    raw = payload.get("maintenance_grace", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise SubnetRuntimeConfigError("maintenance_grace must be an object")

    cfg = MaintenanceGraceConfig(
        enabled=_optional_bool(raw, "enabled", False),
        until_epoch=_optional_nullable_int(raw, "until_epoch", None, minimum=0),
        until_unix_ts=_optional_nullable_int(raw, "until_unix_ts", None, minimum=0),
        reason=_optional_str(raw, "reason", "", maximum_length=160),
        suppress_score_zeroing=_optional_bool(raw, "suppress_score_zeroing", True),
        suppress_probation=_optional_bool(raw, "suppress_probation", True),
        suppress_capacity_score_gate=_optional_bool(raw, "suppress_capacity_score_gate", True),
        suppress_report_offline=_optional_bool(raw, "suppress_report_offline", True),
        suppress_proxy_proof_strikes=_optional_bool(raw, "suppress_proxy_proof_strikes", True),
    )
    if cfg.enabled and cfg.until_epoch is None and cfg.until_unix_ts is None:
        raise SubnetRuntimeConfigError(
            "maintenance_grace requires until_epoch or until_unix_ts when enabled"
        )
    return cfg


def maintenance_grace_active(
    grace: MaintenanceGraceConfig | None,
    *,
    current_epoch: int | None = None,
    now: float | None = None,
) -> bool:
    if grace is None or not grace.enabled:
        return False
    epoch_active = False
    if grace.until_epoch is not None and current_epoch is not None:
        epoch_active = int(current_epoch) <= int(grace.until_epoch)
    time_active = False
    if grace.until_unix_ts is not None:
        time_active = float(now if now is not None else time.time()) <= float(
            grace.until_unix_ts
        )
    return epoch_active or time_active


def build_default_subnet_config_payload(
    *,
    scoring: ScoringParams | None = None,
    neuron_config: Any | None = None,
    version: int = 1,
    effective_epoch: int | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build a complete server config from current code defaults."""
    if neuron_config is None:
        from neurons.config import NeuronConfig

        neuron_config = NeuronConfig()
    scoring = scoring or ScoringParams()
    updated_at = updated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": SUBNET_CONFIG_SCHEMA_VERSION,
        "version": int(version),
        "effective_epoch": effective_epoch,
        "updated_at": updated_at,
        "refresh_seconds": DEFAULT_SUBNET_CONFIG_REFRESH_SECONDS,
        "scoring": {
            "tee_bonus_bps": int(scoring.tee_bonus_bps),
            "ema_alpha_bps": int(scoring.ema_alpha_bps),
            "throughput_power_bps": int(scoring.throughput_power_bps),
            "proof_sample_rate_bps": int(scoring.proof_sample_rate_bps),
            "probation_required_passes": int(scoring.probation_required_passes),
            "demand_bonus_max_bps": int(scoring.demand_bonus_max_bps),
            "emission_burn_bps": int(scoring.emission_burn_bps),
        },
        "runtime": {
            "probation_escalation_epochs": int(neuron_config.probation_escalation_epochs),
            "demand_bonus_enabled": bool(neuron_config.demand_bonus_enabled),
        },
        "epoch": {
            "epoch_blocks": int(neuron_config.epoch_blocks),
            "epoch_grace_blocks": int(neuron_config.epoch_grace_blocks),
            "set_weights_epoch_blocks": int(neuron_config.set_weights_epoch_blocks),
        },
        "canary": {
            "small_count": int(neuron_config.canary_small_count),
            "full_context_count": int(neuron_config.canary_full_context_count),
            "inference_timeout_s": float(neuron_config.canary_inference_timeout),
            "full_context_inference_timeout_s": float(
                neuron_config.canary_full_context_inference_timeout
            ),
            "epoch_receipt_pull_timeout_s": float(neuron_config.epoch_receipt_pull_timeout),
            "epoch_receipt_pull_overall_timeout_s": float(
                neuron_config.epoch_receipt_pull_overall_timeout
            ),
        },
        "capacity_audit": {
            "enabled": bool(neuron_config.capacity_audit_enabled),
            "mode": str(neuron_config.capacity_audit_mode),
            "worker_poll_s": float(neuron_config.capacity_audit_worker_poll_s),
            "cohort_min": int(neuron_config.capacity_audit_cohort_min),
            "cohort_fraction": float(neuron_config.capacity_audit_cohort_fraction),
            "cohort_max": int(neuron_config.capacity_audit_cohort_max),
            "windows_per_epoch": int(neuron_config.capacity_audit_windows_per_epoch),
            "max_drain_fraction": float(neuron_config.capacity_audit_max_drain_fraction),
            "group_stress_fraction": float(neuron_config.capacity_audit_group_stress_fraction),
            "group_stress_min_size": int(neuron_config.capacity_audit_group_stress_min_size),
            "group_stress_slots_per_group": int(
                neuron_config.capacity_audit_group_stress_slots_per_group
            ),
            "beacon_hash_count": int(neuron_config.capacity_audit_beacon_hash_count),
            "min_registration_age_s": float(neuron_config.capacity_audit_min_registration_age_s),
            "lead_blocks": int(neuron_config.capacity_audit_lead_blocks),
            "proof_challenge_delay_blocks": int(
                neuron_config.capacity_audit_proof_challenge_delay_blocks
            ),
            "drain_seconds": float(neuron_config.capacity_audit_drain_seconds),
            "deadline_s": float(neuron_config.capacity_audit_deadline_s),
            "transport_grace_s": float(neuron_config.capacity_audit_transport_grace_s),
            "payload_deadline_s": float(neuron_config.capacity_audit_payload_deadline_s),
            "max_proof_payload_bytes": int(
                neuron_config.capacity_audit_max_proof_payload_bytes
            ),
            "require_proof_payload": bool(neuron_config.capacity_audit_require_proof_payload),
            "repeat_window_epochs": int(neuron_config.capacity_audit_repeat_window_epochs),
            "timing_misses_for_zero_score": int(
                neuron_config.capacity_audit_timing_misses_for_zero_score
            ),
            "hard_proof_misses_for_zero_score": int(
                neuron_config.capacity_audit_hard_proof_misses_for_zero_score
            ),
            "allow_timing_only_score_gate": bool(
                neuron_config.capacity_audit_allow_timing_only_score_gate
            ),
            "slot_refresh_blocks": int(neuron_config.capacity_audit_slot_refresh_blocks),
            "slot_snapshot_stale_blocks": int(
                neuron_config.capacity_audit_slot_snapshot_stale_blocks
            ),
            "proof_verify_workers": int(neuron_config.capacity_audit_proof_verify_workers),
            "gpu_classes": [_gpu_class_to_dict(row) for row in DEFAULT_GPU_CLASSES],
        },
        "maintenance_grace": _maintenance_grace_to_dict(
            maintenance_grace_config_from_neuron_config(neuron_config)
        ),
    }


def validate_subnet_config_payload(
    payload: Mapping[str, Any],
    *,
    source: str = "",
) -> RuntimeSubnetConfig:
    if not isinstance(payload, Mapping):
        raise SubnetRuntimeConfigError("subnet config must be a JSON object")

    schema_version = _require_int(payload, "schema_version", minimum=1)
    if schema_version != SUBNET_CONFIG_SCHEMA_VERSION:
        raise SubnetRuntimeConfigError(
            f"unsupported schema_version: {schema_version}"
        )
    version = _require_int(payload, "version", minimum=1)
    effective_epoch = _require_optional_int(payload, "effective_epoch", minimum=0)
    updated_at = _require_str(payload, "updated_at", nonempty=True)
    refresh_seconds = _require_float(payload, "refresh_seconds", minimum=1.0)

    scoring_data = _require_mapping(payload, "scoring")
    scoring = ScoringParams(
        tee_bonus_bps=_require_int(scoring_data, "tee_bonus_bps", minimum=0, maximum=10000),
        ema_alpha_bps=_require_int(scoring_data, "ema_alpha_bps", minimum=0, maximum=10000),
        throughput_power_bps=_require_int(
            scoring_data, "throughput_power_bps", minimum=0, maximum=50000
        ),
        proof_sample_rate_bps=_require_int(
            scoring_data, "proof_sample_rate_bps", minimum=0, maximum=10000
        ),
        probation_required_passes=_require_int(
            scoring_data, "probation_required_passes", minimum=1, maximum=100
        ),
        demand_bonus_max_bps=_require_int(
            scoring_data, "demand_bonus_max_bps", minimum=0, maximum=10000
        ),
        emission_burn_bps=_require_int(
            scoring_data, "emission_burn_bps", minimum=0, maximum=10000
        ),
    )

    runtime_data = _require_mapping(payload, "runtime")
    probation_escalation_epochs = _require_int(
        runtime_data, "probation_escalation_epochs", minimum=1
    )
    demand_bonus_enabled = _require_bool(runtime_data, "demand_bonus_enabled")

    epoch_data = _require_mapping(payload, "epoch")
    epoch_blocks = _require_int(epoch_data, "epoch_blocks", minimum=1)
    epoch_grace_blocks = _require_int(epoch_data, "epoch_grace_blocks", minimum=0)
    set_weights_epoch_blocks = _require_int(
        epoch_data, "set_weights_epoch_blocks", minimum=1
    )

    canary_data = _require_mapping(payload, "canary")
    canary_small_count = _require_int(canary_data, "small_count", minimum=0)
    canary_full_context_count = _require_int(canary_data, "full_context_count", minimum=0)
    canary_inference_timeout_s = _require_float(
        canary_data, "inference_timeout_s", minimum=1.0
    )
    canary_full_context_inference_timeout_s = _require_float(
        canary_data, "full_context_inference_timeout_s", minimum=1.0
    )
    epoch_receipt_pull_timeout_s = _require_float(
        canary_data, "epoch_receipt_pull_timeout_s", minimum=1.0
    )
    epoch_receipt_pull_overall_timeout_s = _require_float(
        canary_data, "epoch_receipt_pull_overall_timeout_s", minimum=1.0
    )

    audit_data = _require_mapping(payload, "capacity_audit")
    gpu_classes = _parse_gpu_classes(audit_data)
    audit_data_with_defaults = dict(audit_data)
    audit_data_with_defaults.setdefault("max_proof_payload_bytes", 32 * 1024 * 1024)
    capacity_audit = CapacityAuditRuntimeConfig(
        enabled=_require_bool(audit_data, "enabled"),
        mode=_require_str(
            audit_data,
            "mode",
            allowed={"observe", "score_gate", "soft_gate", "enforce"},
        ),
        cohort_min=_require_int(audit_data, "cohort_min", minimum=0),
        cohort_fraction=_require_float(audit_data, "cohort_fraction", minimum=0.0),
        cohort_max=_require_int(audit_data, "cohort_max", minimum=0),
        windows_per_epoch=_require_int(audit_data, "windows_per_epoch", minimum=0),
        max_drain_fraction=_require_float(
            audit_data, "max_drain_fraction", minimum=0.0, maximum=1.0
        ),
        group_stress_fraction=_require_float(
            audit_data, "group_stress_fraction", minimum=0.0, maximum=0.5
        ),
        group_stress_min_size=_require_int(audit_data, "group_stress_min_size", minimum=1),
        group_stress_slots_per_group=_require_int(
            audit_data, "group_stress_slots_per_group", minimum=1
        ),
        beacon_hash_count=_require_int(audit_data, "beacon_hash_count", minimum=1),
        min_registration_age_s=_require_float(
            audit_data, "min_registration_age_s", minimum=0.0
        ),
        lead_blocks=_require_int(audit_data, "lead_blocks", minimum=0),
        proof_challenge_delay_blocks=_require_int(
            audit_data, "proof_challenge_delay_blocks", minimum=1
        ),
        drain_seconds=_require_float(audit_data, "drain_seconds", minimum=0.001),
        deadline_s=_require_float(audit_data, "deadline_s", minimum=0.001),
        transport_grace_s=_require_float(audit_data, "transport_grace_s", minimum=0.0),
        payload_deadline_s=_require_float(audit_data, "payload_deadline_s", minimum=0.001),
        max_proof_payload_bytes=_require_int(
            audit_data_with_defaults,
            "max_proof_payload_bytes",
            minimum=1,
        ),
        require_proof_payload=_require_bool(audit_data, "require_proof_payload"),
        repeat_window_epochs=_require_int(audit_data, "repeat_window_epochs", minimum=1),
        timing_misses_for_zero_score=_require_int(
            audit_data, "timing_misses_for_zero_score", minimum=1
        ),
        hard_proof_misses_for_zero_score=_require_int(
            audit_data, "hard_proof_misses_for_zero_score", minimum=1
        ),
        allow_timing_only_score_gate=_require_bool(
            audit_data, "allow_timing_only_score_gate"
        ),
        gpu_classes=gpu_classes,
    )
    try:
        validate_capacity_audit_runtime_config(capacity_audit)
    except ValueError as exc:
        raise SubnetRuntimeConfigError(str(exc)) from exc
    worker_poll_s = _require_float(audit_data, "worker_poll_s", minimum=0.1)
    slot_refresh_blocks = _require_int(audit_data, "slot_refresh_blocks", minimum=0)
    slot_snapshot_stale_blocks = _require_int(
        audit_data, "slot_snapshot_stale_blocks", minimum=0
    )
    proof_verify_workers = _require_int(audit_data, "proof_verify_workers", minimum=1)
    maintenance_grace = _parse_maintenance_grace(payload)

    normalized = build_default_subnet_config_payload(
        scoring=scoring,
        version=version,
        effective_epoch=effective_epoch,
        updated_at=updated_at,
    )
    normalized["refresh_seconds"] = refresh_seconds
    normalized["runtime"] = {
        "probation_escalation_epochs": probation_escalation_epochs,
        "demand_bonus_enabled": demand_bonus_enabled,
    }
    normalized["epoch"] = {
        "epoch_blocks": epoch_blocks,
        "epoch_grace_blocks": epoch_grace_blocks,
        "set_weights_epoch_blocks": set_weights_epoch_blocks,
    }
    normalized["canary"] = {
        "small_count": canary_small_count,
        "full_context_count": canary_full_context_count,
        "inference_timeout_s": canary_inference_timeout_s,
        "full_context_inference_timeout_s": canary_full_context_inference_timeout_s,
        "epoch_receipt_pull_timeout_s": epoch_receipt_pull_timeout_s,
        "epoch_receipt_pull_overall_timeout_s": epoch_receipt_pull_overall_timeout_s,
    }
    normalized["capacity_audit"] = {
        "enabled": capacity_audit.enabled,
        "mode": capacity_audit.mode,
        "worker_poll_s": worker_poll_s,
        "cohort_min": capacity_audit.cohort_min,
        "cohort_fraction": capacity_audit.cohort_fraction,
        "cohort_max": capacity_audit.cohort_max,
        "windows_per_epoch": capacity_audit.windows_per_epoch,
        "max_drain_fraction": capacity_audit.max_drain_fraction,
        "group_stress_fraction": capacity_audit.group_stress_fraction,
        "group_stress_min_size": capacity_audit.group_stress_min_size,
        "group_stress_slots_per_group": capacity_audit.group_stress_slots_per_group,
        "beacon_hash_count": capacity_audit.beacon_hash_count,
        "min_registration_age_s": capacity_audit.min_registration_age_s,
        "lead_blocks": capacity_audit.lead_blocks,
        "proof_challenge_delay_blocks": capacity_audit.proof_challenge_delay_blocks,
        "drain_seconds": capacity_audit.drain_seconds,
        "deadline_s": capacity_audit.deadline_s,
        "transport_grace_s": capacity_audit.transport_grace_s,
        "payload_deadline_s": capacity_audit.payload_deadline_s,
        "max_proof_payload_bytes": capacity_audit.max_proof_payload_bytes,
        "require_proof_payload": capacity_audit.require_proof_payload,
        "repeat_window_epochs": capacity_audit.repeat_window_epochs,
        "timing_misses_for_zero_score": capacity_audit.timing_misses_for_zero_score,
        "hard_proof_misses_for_zero_score": capacity_audit.hard_proof_misses_for_zero_score,
        "allow_timing_only_score_gate": capacity_audit.allow_timing_only_score_gate,
        "slot_refresh_blocks": slot_refresh_blocks,
        "slot_snapshot_stale_blocks": slot_snapshot_stale_blocks,
        "proof_verify_workers": proof_verify_workers,
        "gpu_classes": [_gpu_class_to_dict(row) for row in gpu_classes],
    }
    normalized["maintenance_grace"] = _maintenance_grace_to_dict(maintenance_grace)

    return RuntimeSubnetConfig(
        schema_version=schema_version,
        version=version,
        effective_epoch=effective_epoch,
        updated_at=updated_at,
        refresh_seconds=refresh_seconds,
        scoring=scoring,
        probation_escalation_epochs=probation_escalation_epochs,
        demand_bonus_enabled=demand_bonus_enabled,
        epoch_blocks=epoch_blocks,
        epoch_grace_blocks=epoch_grace_blocks,
        set_weights_epoch_blocks=set_weights_epoch_blocks,
        canary_small_count=canary_small_count,
        canary_full_context_count=canary_full_context_count,
        canary_inference_timeout_s=canary_inference_timeout_s,
        canary_full_context_inference_timeout_s=canary_full_context_inference_timeout_s,
        epoch_receipt_pull_timeout_s=epoch_receipt_pull_timeout_s,
        epoch_receipt_pull_overall_timeout_s=epoch_receipt_pull_overall_timeout_s,
        capacity_audit=capacity_audit,
        capacity_audit_worker_poll_s=worker_poll_s,
        capacity_audit_slot_refresh_blocks=slot_refresh_blocks,
        capacity_audit_slot_snapshot_stale_blocks=slot_snapshot_stale_blocks,
        capacity_audit_proof_verify_workers=proof_verify_workers,
        maintenance_grace=maintenance_grace,
        payload=normalized,
        source=source,
    )


def load_subnet_config_payload_from_path(path: str | os.PathLike[str]) -> RuntimeSubnetConfig:
    data = json.loads(Path(path).expanduser().read_text())
    return validate_subnet_config_payload(data, source=str(path))


def apply_runtime_config_to_neuron_config(
    runtime: RuntimeSubnetConfig,
    config: Any,
) -> Any:
    scoring = runtime.scoring
    config.ema_alpha = scoring.ema_alpha
    config.throughput_power = scoring.throughput_power
    config.canary_proof_sample_rate = scoring.proof_sample_rate
    config.probation_required_passes = scoring.probation_required_passes
    config.demand_bonus_max = scoring.demand_bonus_max
    config.probation_escalation_epochs = runtime.probation_escalation_epochs
    config.demand_bonus_enabled = runtime.demand_bonus_enabled
    config.epoch_blocks = runtime.epoch_blocks
    config.epoch_grace_blocks = runtime.epoch_grace_blocks
    config.set_weights_epoch_blocks = runtime.set_weights_epoch_blocks
    config.canary_small_count = runtime.canary_small_count
    config.canary_full_context_count = runtime.canary_full_context_count
    config.canary_inference_timeout = runtime.canary_inference_timeout_s
    config.canary_full_context_inference_timeout = (
        runtime.canary_full_context_inference_timeout_s
    )
    config.epoch_receipt_pull_timeout = runtime.epoch_receipt_pull_timeout_s
    config.epoch_receipt_pull_overall_timeout = (
        runtime.epoch_receipt_pull_overall_timeout_s
    )

    audit = runtime.capacity_audit
    config.capacity_audit_enabled = audit.enabled
    config.capacity_audit_mode = audit.mode
    config.capacity_audit_worker_poll_s = runtime.capacity_audit_worker_poll_s
    config.capacity_audit_cohort_min = audit.cohort_min
    config.capacity_audit_cohort_fraction = audit.cohort_fraction
    config.capacity_audit_cohort_max = audit.cohort_max
    config.capacity_audit_windows_per_epoch = audit.windows_per_epoch
    config.capacity_audit_max_drain_fraction = audit.max_drain_fraction
    config.capacity_audit_group_stress_fraction = audit.group_stress_fraction
    config.capacity_audit_group_stress_min_size = audit.group_stress_min_size
    config.capacity_audit_group_stress_slots_per_group = audit.group_stress_slots_per_group
    config.capacity_audit_beacon_hash_count = audit.beacon_hash_count
    config.capacity_audit_min_registration_age_s = audit.min_registration_age_s
    config.capacity_audit_lead_blocks = audit.lead_blocks
    config.capacity_audit_proof_challenge_delay_blocks = audit.proof_challenge_delay_blocks
    config.capacity_audit_drain_seconds = audit.drain_seconds
    config.capacity_audit_deadline_s = audit.deadline_s
    config.capacity_audit_transport_grace_s = audit.transport_grace_s
    config.capacity_audit_payload_deadline_s = audit.payload_deadline_s
    config.capacity_audit_max_proof_payload_bytes = audit.max_proof_payload_bytes
    config.capacity_audit_require_proof_payload = audit.require_proof_payload
    config.capacity_audit_repeat_window_epochs = audit.repeat_window_epochs
    config.capacity_audit_timing_misses_for_zero_score = (
        audit.timing_misses_for_zero_score
    )
    config.capacity_audit_hard_proof_misses_for_zero_score = (
        audit.hard_proof_misses_for_zero_score
    )
    config.capacity_audit_allow_timing_only_score_gate = (
        audit.allow_timing_only_score_gate
    )
    config.capacity_audit_slot_refresh_blocks = (
        runtime.capacity_audit_slot_refresh_blocks
    )
    config.capacity_audit_slot_snapshot_stale_blocks = (
        runtime.capacity_audit_slot_snapshot_stale_blocks
    )
    config.capacity_audit_proof_verify_workers = (
        runtime.capacity_audit_proof_verify_workers
    )
    grace = runtime.maintenance_grace
    config.maintenance_grace_enabled = grace.enabled
    config.maintenance_grace_until_epoch = grace.until_epoch
    config.maintenance_grace_until_unix_ts = grace.until_unix_ts
    config.maintenance_grace_reason = grace.reason
    config.maintenance_grace_suppress_score_zeroing = grace.suppress_score_zeroing
    config.maintenance_grace_suppress_probation = grace.suppress_probation
    config.maintenance_grace_suppress_capacity_score_gate = (
        grace.suppress_capacity_score_gate
    )
    config.maintenance_grace_suppress_report_offline = grace.suppress_report_offline
    config.maintenance_grace_suppress_proxy_proof_strikes = (
        grace.suppress_proxy_proof_strikes
    )
    return config


def maintenance_grace_config_from_neuron_config(config: Any) -> MaintenanceGraceConfig:
    return MaintenanceGraceConfig(
        enabled=bool(getattr(config, "maintenance_grace_enabled", False)),
        until_epoch=getattr(config, "maintenance_grace_until_epoch", None),
        until_unix_ts=getattr(config, "maintenance_grace_until_unix_ts", None),
        reason=str(getattr(config, "maintenance_grace_reason", "") or ""),
        suppress_score_zeroing=bool(
            getattr(config, "maintenance_grace_suppress_score_zeroing", True)
        ),
        suppress_probation=bool(
            getattr(config, "maintenance_grace_suppress_probation", True)
        ),
        suppress_capacity_score_gate=bool(
            getattr(config, "maintenance_grace_suppress_capacity_score_gate", True)
        ),
        suppress_report_offline=bool(
            getattr(config, "maintenance_grace_suppress_report_offline", True)
        ),
        suppress_proxy_proof_strikes=bool(
            getattr(config, "maintenance_grace_suppress_proxy_proof_strikes", True)
        ),
    )


def capacity_audit_config_from_neuron_config(config: Any) -> CapacityAuditRuntimeConfig:
    runtime = CapacityAuditRuntimeConfig(
        enabled=bool(getattr(config, "capacity_audit_enabled", False)),
        mode=str(getattr(config, "capacity_audit_mode", "observe") or "observe"),
        cohort_min=int(getattr(config, "capacity_audit_cohort_min", 100)),
        cohort_fraction=float(getattr(config, "capacity_audit_cohort_fraction", 0.025)),
        cohort_max=int(getattr(config, "capacity_audit_cohort_max", 250)),
        windows_per_epoch=int(getattr(config, "capacity_audit_windows_per_epoch", 30)),
        max_drain_fraction=float(getattr(config, "capacity_audit_max_drain_fraction", 0.05)),
        group_stress_fraction=float(getattr(config, "capacity_audit_group_stress_fraction", 0.35)),
        group_stress_min_size=int(getattr(config, "capacity_audit_group_stress_min_size", 2)),
        group_stress_slots_per_group=int(
            getattr(config, "capacity_audit_group_stress_slots_per_group", 2)
        ),
        beacon_hash_count=int(getattr(config, "capacity_audit_beacon_hash_count", 1)),
        min_registration_age_s=float(getattr(config, "capacity_audit_min_registration_age_s", 0.0)),
        lead_blocks=int(getattr(config, "capacity_audit_lead_blocks", 5)),
        proof_challenge_delay_blocks=int(
            getattr(config, "capacity_audit_proof_challenge_delay_blocks", 4)
        ),
        drain_seconds=float(getattr(config, "capacity_audit_drain_seconds", 30.0)),
        deadline_s=float(getattr(config, "capacity_audit_deadline_s", 30.0)),
        transport_grace_s=float(getattr(config, "capacity_audit_transport_grace_s", 3.0)),
        payload_deadline_s=float(getattr(config, "capacity_audit_payload_deadline_s", 60.0)),
        max_proof_payload_bytes=int(
            getattr(config, "capacity_audit_max_proof_payload_bytes", 32 * 1024 * 1024)
        ),
        require_proof_payload=bool(getattr(config, "capacity_audit_require_proof_payload", True)),
        repeat_window_epochs=int(getattr(config, "capacity_audit_repeat_window_epochs", 20)),
        timing_misses_for_zero_score=int(
            getattr(config, "capacity_audit_timing_misses_for_zero_score", 2)
        ),
        hard_proof_misses_for_zero_score=int(
            getattr(config, "capacity_audit_hard_proof_misses_for_zero_score", 2)
        ),
        allow_timing_only_score_gate=bool(
            getattr(config, "capacity_audit_allow_timing_only_score_gate", True)
        ),
        validator_urls=tuple(
            u.strip()
            for u in str(getattr(config, "capacity_audit_validator_urls", "") or "").split(",")
            if u.strip()
        ),
    )
    validate_capacity_audit_runtime_config(runtime)
    return runtime


class RuntimeSubnetConfigClient:
    """Fetch and cache the public runtime subnet config."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_SUBNET_CONFIG_URL,
        refresh_seconds: float = DEFAULT_SUBNET_CONFIG_REFRESH_SECONDS,
        timeout_seconds: float = DEFAULT_SUBNET_CONFIG_TIMEOUT_SECONDS,
        cache_path: str | os.PathLike[str] | None = None,
        disabled: bool = False,
        log: Any | None = None,
    ) -> None:
        self.url = str(url or "").strip()
        self.refresh_seconds = max(1.0, float(refresh_seconds or 1.0))
        self.timeout_seconds = max(0.2, float(timeout_seconds or 0.2))
        self.cache_path = Path(cache_path).expanduser() if cache_path else _data_dir_cache_path()
        self.disabled = bool(disabled)
        self.log = log or logger
        self._active: RuntimeSubnetConfig | None = None
        self._last_fetch_at = 0.0
        self._last_error = ""
        self._last_authoritative = False

    @property
    def last_authoritative(self) -> bool:
        return bool(self._last_authoritative)

    @classmethod
    def from_config(cls, config: Any, *, log: Any | None = None) -> "RuntimeSubnetConfigClient":
        return cls(
            url=getattr(config, "subnet_config_url", DEFAULT_SUBNET_CONFIG_URL),
            refresh_seconds=getattr(
                config,
                "subnet_config_refresh_seconds",
                DEFAULT_SUBNET_CONFIG_REFRESH_SECONDS,
            ),
            timeout_seconds=getattr(
                config,
                "subnet_config_timeout_seconds",
                DEFAULT_SUBNET_CONFIG_TIMEOUT_SECONDS,
            ),
            cache_path=getattr(config, "subnet_config_cache_path", "") or None,
            disabled=bool(getattr(config, "subnet_config_disable", False)),
            log=log,
        )

    def get(
        self,
        *,
        current_epoch: int | None = None,
        force: bool = False,
    ) -> RuntimeSubnetConfig | None:
        if self.disabled or not self.url:
            return None
        now = time.time()
        if (
            not force
            and self._active is not None
            and now - self._last_fetch_at < self.refresh_seconds
            and self._is_effective(self._active, current_epoch)
        ):
            return self._active

        try:
            cfg = self._fetch()
            self._last_fetch_at = now
            self._last_error = ""
            self.refresh_seconds = max(1.0, float(cfg.refresh_seconds or self.refresh_seconds))
            if self._is_effective(cfg, current_epoch):
                self._active = cfg
                self._last_authoritative = True
                return cfg
            self._last_authoritative = False
            return self._active if self._is_effective(self._active, current_epoch) else None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._last_authoritative = False
            self._warn(f"Runtime subnet config fetch failed: {self._last_error}")
            if self._active is not None and self._is_effective(self._active, current_epoch):
                return self._active
            cached = self._load_cache(current_epoch=current_epoch)
            if cached is not None:
                self._active = cached
                return cached
            return None

    def _is_effective(
        self,
        cfg: RuntimeSubnetConfig | None,
        current_epoch: int | None,
    ) -> bool:
        if cfg is None:
            return False
        if cfg.effective_epoch is None:
            return True
        if current_epoch is None:
            return False
        return int(cfg.effective_epoch) <= int(current_epoch)

    def _fetch(self) -> RuntimeSubnetConfig:
        resp = httpx.get(self.url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        cfg = validate_subnet_config_payload(data, source=self.url)
        self._save_cache(cfg)
        return cfg

    def _load_cache(self, *, current_epoch: int | None) -> RuntimeSubnetConfig | None:
        try:
            cfg = load_subnet_config_payload_from_path(self.cache_path)
            object.__setattr__(cfg, "source", str(self.cache_path))
            if self._is_effective(cfg, current_epoch):
                return cfg
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._warn(f"Ignoring invalid runtime subnet config cache: {exc}")
        return None

    def _save_cache(self, cfg: RuntimeSubnetConfig) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(cfg.payload, indent=2, sort_keys=True) + "\n")
            tmp.replace(self.cache_path)
        except Exception as exc:
            self._debug(f"Runtime subnet config cache write failed: {exc}")

    def _warn(self, message: str) -> None:
        warning = getattr(self.log, "warning", None)
        if callable(warning):
            warning(message)

    def _debug(self, message: str) -> None:
        debug = getattr(self.log, "debug", None)
        if callable(debug):
            debug(message)


def load_onchain_scoring_params(
    *,
    chain_config: str | None = None,
    subtensor_network: str | None = None,
) -> ScoringParams:
    import json

    from verallm.chain.config import ChainConfig
    from verallm.chain.subnet_config import SubnetConfigClient

    resolved = ChainConfig.resolve_config_path(chain_config, subtensor_network)
    if resolved:
        with open(resolved) as f:
            raw = json.load(f)
        overrides = {}
        if not raw.get("rpc_url"):
            network = subtensor_network
            if network is None:
                chain_id = int(raw.get("chain_id") or 0)
                if chain_id == 964:
                    network = "finney"
                elif chain_id == 945:
                    network = "test"
            rpc_url = ChainConfig.resolve_rpc_url(None, network)
            if rpc_url:
                overrides["rpc_url"] = rpc_url
        config = ChainConfig.from_json(resolved, **overrides)
    else:
        config = ChainConfig.from_env()
    return SubnetConfigClient(config).get_scoring_params()


def _write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or export Verathos subnet runtime config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", help="Validate an existing subnet config JSON file")
    group.add_argument("--export", help="Write a complete subnet config JSON file")
    parser.add_argument("--chain-config", default=None, help="Chain config JSON for live SubnetConfig scoring read")
    parser.add_argument("--subtensor-network", default=None, help="Network name used to resolve bundled chain config")
    parser.add_argument("--allow-code-defaults", action="store_true", help="Allow code scoring defaults if live chain read fails")
    parser.add_argument("--version", type=int, default=1, help="Config version to write with --export")
    parser.add_argument("--effective-epoch", type=int, default=None, help="Optional effective epoch for exported config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.check:
        cfg = load_subnet_config_payload_from_path(args.check)
        print(
            f"ok version={cfg.version} effective_epoch={cfg.effective_epoch} "
            f"gpu_classes={len(cfg.capacity_audit.gpu_classes)}"
        )
        return

    scoring: ScoringParams | None = None
    try:
        scoring = load_onchain_scoring_params(
            chain_config=args.chain_config,
            subtensor_network=args.subtensor_network,
        )
    except Exception as exc:
        if not args.allow_code_defaults:
            raise SystemExit(
                "live SubnetConfig scoring read failed; rerun with "
                f"--allow-code-defaults to export code defaults: {exc}"
            )
        print(f"warning: using code scoring defaults because chain read failed: {exc}")
    payload = build_default_subnet_config_payload(
        scoring=scoring,
        version=args.version,
        effective_epoch=args.effective_epoch,
    )
    validate_subnet_config_payload(payload, source="export")
    _write_json_atomic(args.export, payload)
    print(f"wrote {args.export}")


if __name__ == "__main__":
    main()
