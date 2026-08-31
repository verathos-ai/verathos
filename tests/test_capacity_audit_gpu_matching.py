from neurons.capacity_audit import (
    CapacityAuditRuntimeConfig,
    CapacityGpuClass,
    match_gpu_class,
)
from verallm.vram import normalize_vram_gb


def _config_with(row: CapacityGpuClass) -> CapacityAuditRuntimeConfig:
    return CapacityAuditRuntimeConfig(gpu_classes=(row,))


def test_h200_runtime_value_matches_default_class() -> None:
    cfg = CapacityAuditRuntimeConfig()

    row = match_gpu_class("NVIDIA H200", 141, cfg)

    assert row is not None
    assert row.vram_gb == 141


def test_legacy_h200_config_value_is_canonicalized() -> None:
    legacy_row = CapacityGpuClass("NVIDIA H200", 144, calibrated=True)

    assert match_gpu_class("NVIDIA H200", 141, _config_with(legacy_row)) == legacy_row


def test_legacy_b200_config_value_is_canonicalized() -> None:
    legacy_row = CapacityGpuClass("NVIDIA B200", 183, calibrated=True)

    assert match_gpu_class("NVIDIA B200", 192, _config_with(legacy_row)) == legacy_row


def test_unrelated_vram_size_still_does_not_match() -> None:
    row = CapacityGpuClass("NVIDIA H200", 141, calibrated=True)

    assert match_gpu_class("NVIDIA H200", 96, _config_with(row)) is None


def test_h200_measured_vram_normalizes_to_marketed_size() -> None:
    assert normalize_vram_gb(143_771 / 1024) == 141
