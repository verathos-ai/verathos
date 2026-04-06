"""GPU auto-detection for VRAM tier resolution.

Detects the available GPU VRAM and maps it to the nearest
:class:`~verallm.registry.models.VRAMTier`.

Requires ``torch`` with CUDA support — imported lazily so the pure-data
``models`` module stays dependency-free.
"""

from __future__ import annotations

from verallm.registry.models import VRAMTier


def detect_vram_gb(device: int = 0) -> int:
    """Return **marketed** VRAM in GB for the given CUDA device.

    ``torch.cuda.get_device_properties().total_memory`` reports *usable*
    VRAM after driver/firmware reservation, which is typically 5-12% less
    than the physical spec (e.g. an L40S advertises 48 GB but torch sees
    ~44.4 GB).  If we simply ``round()`` this we get 44, which maps to
    ``GB_32`` instead of the correct ``GB_48``.

    Strategy: round **up** to the nearest common power-of-2-ish spec size.
    This matches what nvidia-smi and vendor datasheets report.

    Raises ``RuntimeError`` if no CUDA GPU is available.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for GPU detection") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")

    total_bytes = torch.cuda.get_device_properties(device).total_memory
    raw_gb = total_bytes / (1024 ** 3)

    # Common marketed VRAM sizes.  Find the smallest spec >= raw_gb * 0.95
    # (allowing 5% tolerance for driver overhead) that is also within 25%
    # of the raw value (to avoid mapping a 24 GB card to 32 GB).
    _SPEC_SIZES = (16, 24, 32, 48, 80, 96, 128, 141, 192, 288)
    for spec in _SPEC_SIZES:
        if spec >= raw_gb * 0.95 and spec <= raw_gb * 1.25:
            return spec

    # Fallback: plain round for unusual sizes
    return round(raw_gb)


def detect_vram_tier(device: int = 0) -> VRAMTier:
    """Auto-detect the VRAM tier of the current GPU.

    Maps the detected VRAM (in GB) to the **largest** ``VRAMTier`` whose
    value is ``<=`` the actual VRAM.  For example, a GPU with 25 GB would
    map to ``GB_24``, and one with 80 GB maps to ``GB_80``.

    Raises ``RuntimeError`` if no CUDA GPU is available.
    Raises ``ValueError`` if the GPU has less than the minimum tier (24 GB).
    """
    vram_gb = detect_vram_gb(device)

    # Sorted tiers excluding MULTI_GPU sentinel
    real_tiers = sorted(
        (t for t in VRAMTier if t != VRAMTier.MULTI_GPU),
        key=lambda t: t.value,
    )

    best: VRAMTier | None = None
    for tier in real_tiers:
        if tier.value <= vram_gb:
            best = tier

    if best is None:
        raise ValueError(
            f"GPU has {vram_gb} GB VRAM, below the minimum tier "
            f"({real_tiers[0].name} = {real_tiers[0].value} GB)."
        )
    return best


def detect_gpu_info(device: int = 0) -> dict:
    """Return a summary dict of the detected GPU for display purposes."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(device)
    vram_gb = round(props.total_memory / (1024 ** 3))
    tier = detect_vram_tier(device)
    return {
        "available": True,
        "name": props.name,
        "vram_gb": vram_gb,
        "tier": tier,
        "compute_capability": f"{props.major}.{props.minor}",
    }
