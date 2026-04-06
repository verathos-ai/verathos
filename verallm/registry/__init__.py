"""
VeraLLM Registry — model catalogue, root computation, and caching.

* **Model catalogue** (``verallm.registry.models``): central list of all
  supported models with VRAM-tier configs, categories, and query helpers.
* **Root computation** (``verallm.registry.roots``): one-time weight Merkle
  root computation via ``compute_model_roots()``.
* **Caching** (``verallm.registry.cache``): ModelSpec / Merkle-tree
  persistence for development.
"""

from verallm.registry.roots import compute_model_roots
from verallm.registry.cache import (
    CACHE_DIR,
    MERKLE_CACHE_DIR,
    load_cached_model_spec,
    save_model_spec_to_cache,
    _get_cache_path,
    _get_merkle_cache_path,
    _load_merkle_from_disk,
    _save_merkle_to_disk,
)
from verallm.registry.models import (
    # Enums
    VRAMTier,
    ModelCategory,
    # Dataclasses
    QuantOption,
    TierConfig,
    TrainingConfig,
    ModelEntry,
    ModelPreset,
    TierMatch,
    ModelRecommendation,
    # Data
    ALL_MODELS,
    MODELS_BY_ID,
    MODEL_PRESETS,
    PRESETS_BY_ID,
    QUANT_QUALITY,
    # Query functions
    get_model,
    get_models_for_tier,
    get_models_by_category,
    get_single_gpu_models,
    get_multi_gpu_models,
    get_training_models,
    get_models_up_to_tier,
    resolve_model_for_tier,
    # Estimation & recommendation
    estimate_weight_gb,
    estimate_effective_context,
    recommend_models,
    # Backward-compat bridges
    get_inference_presets,
    get_training_presets,
    get_webapp_presets,
    get_presets_grouped,
)

__all__ = [
    # Roots & cache
    "compute_model_roots",
    "CACHE_DIR",
    "MERKLE_CACHE_DIR",
    "load_cached_model_spec",
    "save_model_spec_to_cache",
    "_get_cache_path",
    "_get_merkle_cache_path",
    "_load_merkle_from_disk",
    "_save_merkle_to_disk",
    # Model catalogue
    "VRAMTier",
    "ModelCategory",
    "QuantOption",
    "TierConfig",
    "TrainingConfig",
    "ModelEntry",
    "ModelPreset",
    "TierMatch",
    "ALL_MODELS",
    "MODELS_BY_ID",
    "MODEL_PRESETS",
    "PRESETS_BY_ID",
    "QUANT_QUALITY",
    "get_model",
    "get_models_for_tier",
    "get_models_by_category",
    "get_single_gpu_models",
    "get_multi_gpu_models",
    "get_training_models",
    "get_models_up_to_tier",
    "resolve_model_for_tier",
    "ModelRecommendation",
    "estimate_weight_gb",
    "estimate_effective_context",
    "recommend_models",
    "get_inference_presets",
    "get_training_presets",
    "get_webapp_presets",
    "get_presets_grouped",
]
