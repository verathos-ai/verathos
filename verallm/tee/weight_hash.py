"""Compute a flat SHA-256 hash over model safetensors files.

This is the TEE path's model identity anchor — cheap to compute (single
sequential pass over files, no GPU, no quantization) and deterministic
for a given HuggingFace model revision.

The ZK path uses per-layer Merkle trees (expensive).  TEE miners only
need this flat hash, bound into the hardware attestation report_data.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_weight_file_hash(
    model_id: str,
    cache_dir: Optional[str] = None,
) -> bytes:
    """SHA-256 over safetensors shard files (sorted by name, concatenated).

    Deterministic: same HF revision + same files = same hash.
    Fast: sequential file read (~1 GB/s on SSD), no GPU, no quantization.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-8B"``).
    cache_dir : str, optional
        Explicit local path.  If *None*, resolves via ``huggingface_hub``.

    Returns
    -------
    bytes
        32-byte SHA-256 digest.
    """
    model_path = _resolve_model_path(model_id, cache_dir)
    st_files = sorted(model_path.glob("*.safetensors"))

    if not st_files:
        raise FileNotFoundError(
            f"No *.safetensors files in {model_path} for model {model_id}"
        )

    h = hashlib.sha256()
    total_bytes = 0
    for f in st_files:
        size = f.stat().st_size
        total_bytes += size
        with open(f, "rb") as fp:
            while True:
                chunk = fp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                h.update(chunk)

    digest = h.digest()
    logger.info(
        "weight_file_hash: %s — %d files, %.1f GB -> %s",
        model_id,
        len(st_files),
        total_bytes / 1e9,
        digest.hex()[:16] + "...",
    )
    return digest


def _resolve_model_path(model_id: str, cache_dir: Optional[str]) -> Path:
    """Resolve model to a local directory containing safetensors files."""
    if cache_dir is not None:
        p = Path(cache_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Explicit cache_dir not found: {cache_dir}")

    # Try huggingface_hub snapshot_download (local_files_only — no download)
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(model_id, local_files_only=True)
        return Path(local_dir)
    except Exception:
        pass

    # Fallback: treat model_id as a local path
    p = Path(model_id)
    if p.is_dir():
        return p

    raise FileNotFoundError(
        f"Cannot resolve model {model_id!r} to a local directory. "
        "Download it first or pass cache_dir explicitly."
    )
