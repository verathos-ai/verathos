"""
ModelSpec and Merkle tree disk caching.

Caches computed ModelSpec (weight Merkle roots) and FlatWeightMerkle trees
to avoid recomputation during development and testing.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directories at repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _REPO_ROOT / ".model_root_cache"
MERKLE_CACHE_DIR = _REPO_ROOT / ".merkle_tree_cache"


# ============================================================================
# ModelSpec Caching
# ============================================================================

def _get_cache_path(model_name: str, chunk_size: int, quant_mode: str = "fp16") -> Path:
    """Get cache file path for a model's ModelSpec."""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe_name}_chunk{chunk_size}_{quant_mode}.pkl"


def load_cached_model_spec(model_name: str, chunk_size: int, quant_mode: str = "fp16"):
    """
    Load a cached ModelSpec if it exists.

    Returns None if no cache exists or cache is invalid.
    """
    from verallm.types import ModelSpec

    cache_path = _get_cache_path(model_name, chunk_size, quant_mode)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        if isinstance(cached, ModelSpec) and cached.model_id == model_name:
            logger.debug("Registry: Loaded cached ModelSpec from %s", cache_path)
            return cached
    except Exception as e:
        logger.debug("Registry: Cache load failed (%s), will recompute", e)

    return None


def save_model_spec_to_cache(model_spec, chunk_size: int, quant_mode: str = "fp16") -> None:
    """Save a ModelSpec to cache for future use."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(model_spec.model_id, chunk_size, quant_mode)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(model_spec, f)
        logger.debug("Registry: Cached ModelSpec to %s", cache_path)
    except Exception as e:
        logger.debug("Registry: Failed to cache ModelSpec (%s)", e)


# ============================================================================
# Merkle Tree Disk Caching
# ============================================================================

def _get_merkle_cache_path(model_name: str, quant_mode: str,
                           layer_idx: int, expert_idx: int = -1,
                           tag: str = "") -> Path:
    """Get disk cache path for a FlatWeightMerkle tree."""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    if tag:
        return MERKLE_CACHE_DIR / f"{safe_name}_{quant_mode}_{tag}.pkl"
    if expert_idx >= 0:
        return MERKLE_CACHE_DIR / f"{safe_name}_{quant_mode}_L{layer_idx}_E{expert_idx}.pkl"
    return MERKLE_CACHE_DIR / f"{safe_name}_{quant_mode}_L{layer_idx}.pkl"


def _load_merkle_from_disk(model_name: str, quant_mode: str, layer_idx: int = 0,
                           expert_idx: int = -1,
                           expected_root: Optional[bytes] = None,
                           tag: str = ""):
    """Load a FlatWeightMerkle from disk cache.

    Args:
        expected_root: If provided, validates the cached root matches.
                       Catches stale caches from code changes.
        tag: Named cache key (e.g. "lm_head", "embedding"). Overrides layer_idx.
    Returns:
        FlatWeightMerkle (with _raw_bytes=None) or None on miss/mismatch.
    """
    from verallm.crypto.merkle import FlatWeightMerkle as _FWM
    cache_path = _get_merkle_cache_path(model_name, quant_mode, layer_idx, expert_idx, tag=tag)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        tree = _FWM.from_cached(data)
        if expected_root is not None and tree.root != expected_root:
            return None  # Stale cache
        return tree
    except Exception:
        return None


def _save_merkle_to_disk(tree, model_name: str, quant_mode: str,
                         layer_idx: int = 0, expert_idx: int = -1,
                         tag: str = "") -> None:
    """Save a FlatWeightMerkle to disk cache."""
    MERKLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_merkle_cache_path(model_name, quant_mode, layer_idx, expert_idx, tag=tag)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(tree.get_cache_data(), f)
    except Exception:
        pass  # Non-fatal
