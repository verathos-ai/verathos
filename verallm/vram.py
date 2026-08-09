"""Dependency-free helpers for canonicalizing reported GPU memory."""

from __future__ import annotations

MARKETED_VRAM_SIZES_GB = (16, 24, 32, 48, 80, 96, 128, 141, 192, 288)


def normalize_vram_gb(raw_gb: float) -> int:
    """Return the marketed VRAM size corresponding to a usable/raw value.

    CUDA and ``nvidia-smi`` report usable memory after reservations, while GPU
    class configuration may contain either that observed value or the marketed
    specification. Keeping this conversion independent of CUDA lets callers
    canonicalize both sides of a comparison with exactly the same rules.
    """
    for spec in MARKETED_VRAM_SIZES_GB:
        if spec >= raw_gb * 0.95 and spec <= raw_gb * 1.25:
            return spec

    return round(raw_gb)
