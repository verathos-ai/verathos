"""Dependency-free GPU-memory class normalization."""

from __future__ import annotations

# Physical/marketed capacities used by supported accelerator families. CUDA
# and nvidia-smi may report usable memory below these values, while historic
# capacity-audit rows may contain the provider's binary-GB representation.
MARKETED_VRAM_SIZES_GB = (
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    80,
    96,
    128,
    141,
    192,
    288,
)


def normalize_vram_gb(raw_gb: float) -> int:
    """Map a usable/provider VRAM value to its marketed memory class."""

    value = float(raw_gb)
    if value <= 0:
        return round(value)
    for marketed_gb in MARKETED_VRAM_SIZES_GB:
        if marketed_gb >= value * 0.95 and marketed_gb <= value * 1.25:
            return marketed_gb
    return round(value)
