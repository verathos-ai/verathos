"""Cascading model/quant/context resolution for miner setup.

Resolves a miner's model configuration by cascading through:
1. Explicit CLI values (always win)
2. Registry recommendations for the detected GPU tier
3. Sensible defaults (fp16, error if context unknown)

Used by ``neurons.miner`` and ``scripts/register_miner_onchain.py``.
"""

from __future__ import annotations

import logging
import bittensor as bt
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """Fully resolved model configuration for a miner."""

    model_id: str
    quant: str
    max_context_len: int
    # Source info for logging
    model_source: str  # "cli", "auto", "auto+category"
    quant_source: str  # "cli", "registry", "default"
    context_source: str  # "cli", "registry"


@dataclass(frozen=True)
class RecommendedCapacityModel:
    model_id: str
    quant: str
    max_context_len: int
    tier_name: str
    registry_id: str


def vram_tier_for_gb(vram_gb: int):
    from verallm.registry import VRAMTier

    real_tiers = sorted(
        (t for t in VRAMTier if t != VRAMTier.MULTI_GPU),
        key=lambda t: t.value,
    )
    best = None
    for tier in real_tiers:
        if tier.value <= int(vram_gb or 0):
            best = tier
    return best


def capacity_gate_vram_gb(gpu_info: dict) -> int:
    tier = gpu_info.get("tier")
    tier_value = getattr(tier, "value", None)
    if tier_value is not None:
        return int(tier_value)
    return int(gpu_info.get("vram_gb") or 0)


def _filter_capacity_recommendations(recs, on_chain_models: Optional[Iterable[str]]):
    if on_chain_models is None:
        return list(recs)
    on_chain_set = {str(m).lower() for m in on_chain_models}
    return [
        r for r in recs
        if str(r.config.checkpoint).lower() in on_chain_set
    ]


def recommended_capacity_model_for_vram(
    vram_gb: int,
    *,
    on_chain_models: Optional[Iterable[str]] = None,
) -> RecommendedCapacityModel | None:
    from verallm.registry import recommend_models

    tier = vram_tier_for_gb(int(vram_gb or 0))
    if tier is None:
        return None
    recs = recommend_models(tier, verified_only=True)
    recs = _filter_capacity_recommendations(recs, on_chain_models)
    if not recs:
        return None
    best = recs[0]
    return RecommendedCapacityModel(
        model_id=str(best.config.checkpoint),
        quant=str(best.quant),
        max_context_len=int(best.est_context or 0),
        tier_name=str(tier.name),
        registry_id=str(best.model.id),
    )


def validate_capacity_recommended_model(
    *,
    model_id: str,
    quant: str,
    max_context_len: int,
    vram_gb: int,
    on_chain_models: Optional[Iterable[str]] = None,
) -> tuple[bool, str, RecommendedCapacityModel | None]:
    expected = recommended_capacity_model_for_vram(
        int(vram_gb or 0),
        on_chain_models=on_chain_models,
    )
    if expected is None:
        return False, f"no verified recommended model for {int(vram_gb or 0)}GB VRAM", None
    if str(model_id or "").lower() != expected.model_id.lower():
        return (
            False,
            f"capacity audit requires {expected.model_id} on {expected.tier_name}",
            expected,
        )
    if str(quant or "").lower() != expected.quant.lower():
        return (
            False,
            f"capacity audit requires quant={expected.quant} for {expected.model_id}",
            expected,
        )
    if int(max_context_len or 0) < int(expected.max_context_len or 0):
        return (
            False,
            f"capacity audit requires max_context_len>={expected.max_context_len} "
            f"for {expected.model_id}",
            expected,
        )
    return True, "", expected


def _get_on_chain_models(chain_config_path: str, subtensor_network: Optional[str] = None) -> Optional[list[str]]:
    """Query on-chain ModelRegistry for registered model IDs.

    Returns a list of model ID strings, or None on error (RPC failure, etc.).
    Cached for the process lifetime after first call.
    """
    cache_key = (str(chain_config_path or ""), str(subtensor_network or ""))
    cache = getattr(_get_on_chain_models, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    try:
        from verallm.chain.config import ChainConfig
        from verallm.chain.mock import create_clients

        rpc_override = ChainConfig.resolve_rpc_url(None, subtensor_network)
        cc = ChainConfig.from_json(
            chain_config_path,
            **({"rpc_url": rpc_override} if rpc_override else {}),
        )
        model_client, _, _ = create_clients(cc)
        models = model_client.get_model_list()
        bt.logging.info(f"On-chain ModelSpec: {len(models)} models registered")
        cache[cache_key] = models
        _get_on_chain_models._cache = cache
        return models
    except Exception as e:
        bt.logging.warning(f"Failed to query on-chain ModelRegistry: {e}")
        cache[cache_key] = None
        _get_on_chain_models._cache = cache
        return None


def resolve_model_config(
    *,
    model_id: Optional[str] = None,
    quant: Optional[str] = None,
    max_context_len: Optional[int] = None,
    auto: bool = False,
    category: Optional[str] = None,
    chain_config: Optional[str] = None,
    subtensor_network: Optional[str] = None,
    capacity_audit_required: bool = False,
) -> ResolvedModel:
    """Resolve a miner's model configuration with cascading fallback.

    Parameters
    ----------
    model_id : str or None
        Explicit model ID from CLI (``--model-id``).
    quant : str or None
        Explicit quantization from CLI (``--quant``).
    max_context_len : int or None
        Explicit max context length from CLI (``--max-context-len``).
    auto : bool
        If True, auto-select model from registry when model_id is None.
    category : str or None
        Category filter for auto-selection (e.g. "coding", "reasoning").

    Returns
    -------
    ResolvedModel
        Fully resolved configuration with source annotations.

    Raises
    ------
    SystemExit
        If the configuration cannot be resolved (no GPU, no models fit, etc.).
    """
    import sys

    from verallm.registry import (
        ModelCategory,
        recommend_models,
        resolve_model_for_tier,
        estimate_effective_context,
    )
    from verallm.registry.gpu import detect_gpu_info

    # --- GPU detection (always needed for registry lookups) ---
    gpu_info = detect_gpu_info()
    if not gpu_info["available"]:
        if capacity_audit_required:
            bt.logging.error("Capacity audit is enabled but no CUDA GPU was detected")
            sys.exit(1)
        if model_id and quant and max_context_len:
            # All explicit — no GPU needed for resolution
            bt.logging.warning("No GPU detected, using explicit config")
            return ResolvedModel(
                model_id=model_id,
                quant=quant,
                max_context_len=max_context_len,
                model_source="cli",
                quant_source="cli",
                context_source="cli",
            )
        bt.logging.error("No GPU detected and not all config values provided explicitly")
        sys.exit(1)

    tier = gpu_info["tier"]
    bt.logging.info(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB, tier={tier.name})")

    # --- Full auto or no model specified ---
    # --model-id auto is equivalent to --auto
    if model_id is not None and model_id.lower() == "auto":
        model_id = None
        auto = True
    if auto or model_id is None:
        cat = ModelCategory(category) if category else None
        recs = recommend_models(
            tier,
            category=cat,
            verified_only=bool(capacity_audit_required),
        )

        # Filter by on-chain ModelSpec availability (if chain config provided)
        if chain_config and recs:
            on_chain_models = _get_on_chain_models(chain_config, subtensor_network)
            if on_chain_models is None and capacity_audit_required:
                bt.logging.error(
                    "Capacity audit is enabled but on-chain ModelRegistry "
                    "recommendations could not be loaded"
                )
                sys.exit(1)
            if on_chain_models is not None:
                on_chain_set = {m.lower() for m in on_chain_models}
                filtered = _filter_capacity_recommendations(recs, on_chain_set)
                if filtered:
                    bt.logging.info(f"Filtered to {len(filtered)}/{len(recs)} models with on-chain ModelSpec")
                    recs = filtered
                else:
                    # Read contract address for the error message
                    import json as _json
                    try:
                        _cc = _json.load(open(chain_config))
                        _addr = _cc.get("model_registry_address", "unknown")
                    except Exception:
                        _addr = "unknown"
                    bt.logging.error(
                        "No recommended models are registered on-chain. "
                        "Miners can only serve models that the subnet owner has "
                        "registered on the ModelRegistry contract. "
                        f"ModelRegistry: {_addr} | "
                        f"Registered models: {sorted(on_chain_set) if on_chain_set else '(none)'}"
                    )
                    sys.exit(1)

        if not recs:
            cat_str = f" for category '{category}'" if category else ""
            bt.logging.error(f"No models fit GPU tier {tier.name}{cat_str}")
            sys.exit(1)

        # Show top 5 recommendations
        bt.logging.info("Top model recommendations:")
        for i, r in enumerate(recs[:5]):
            bt.logging.info(f"  {i + 1}. {r.model.id} (quant={r.quant}, ctx={r.est_context}, vram={r.est_weight_gb:.1f}GB, utility={r.utility:.1f})")

        best = recs[0]

        # Use the HF checkpoint name (not the registry shortname) as model_id
        # so that on-chain registration matches what the miner actually serves.
        resolved_quant = quant or best.quant
        resolved_context = max_context_len or best.est_context

        model_source = "cli" if model_id else ("auto+category" if category else "auto")
        quant_source = "cli" if quant else "registry"
        context_source = "cli" if max_context_len else "registry"

        if model_id:
            # CLI gave a shortname — resolve it through the registry to get checkpoint
            try:
                match = resolve_model_for_tier(model_id, tier)
                resolved_model_id = match.config.checkpoint
            except (KeyError, ValueError):
                # Not in registry — assume user passed an HF name directly
                resolved_model_id = model_id
        else:
            resolved_model_id = best.config.checkpoint

        bt.logging.info(f"Resolved: model={resolved_model_id} (src={model_source}) quant={resolved_quant} (src={quant_source}) ctx={resolved_context} (src={context_source})")

        return ResolvedModel(
            model_id=resolved_model_id,
            quant=resolved_quant,
            max_context_len=resolved_context,
            model_source=model_source,
            quant_source=quant_source,
            context_source=context_source,
        )

    # --- Explicit model_id, but quant/context may be auto ---
    assert model_id is not None

    from verallm.registry.models import MODELS_BY_ID as _models_by_id

    # Determine whether user passed an HF checkpoint (has '/') or a registry
    # shortname (e.g. 'qwen3.5-9b').  This controls whether we keep the
    # user's checkpoint or let the registry pick one for this tier.
    _user_gave_hf_checkpoint = "/" in model_id
    resolved_model_id = model_id

    # If all three are explicit, return immediately — no registry needed.
    if quant is not None and max_context_len is not None:
        if capacity_audit_required:
            on_chain_models = (
                _get_on_chain_models(chain_config, subtensor_network)
                if chain_config else None
            )
            if chain_config and on_chain_models is None:
                bt.logging.error(
                    "Capacity audit is enabled but on-chain ModelRegistry "
                    "recommendations could not be loaded"
                )
                sys.exit(1)
            ok, reason, expected = validate_capacity_recommended_model(
                model_id=resolved_model_id,
                quant=quant,
                max_context_len=max_context_len,
                vram_gb=capacity_gate_vram_gb(gpu_info),
                on_chain_models=on_chain_models,
            )
            if not ok:
                expected_text = ""
                if expected is not None:
                    expected_text = (
                        f" expected model={expected.model_id} "
                        f"quant={expected.quant} ctx>={expected.max_context_len}"
                    )
                bt.logging.error(f"{reason}.{expected_text}")
                sys.exit(1)
        return ResolvedModel(
            model_id=resolved_model_id,
            quant=quant,
            max_context_len=max_context_len,
            model_source="cli",
            quant_source="cli",
            context_source="cli",
        )

    # Look up registry entry: by shortname first, then reverse-lookup by
    # HF checkpoint name if the shortname doesn't match.
    _lookup_id = model_id
    _registry_model = _models_by_id.get(model_id)
    if _registry_model is None and _user_gave_hf_checkpoint:
        for _rm in _models_by_id.values():
            if any(tc.checkpoint.lower() == model_id.lower() for tc in _rm.tier_configs):
                _lookup_id = _rm.id
                bt.logging.info(f"Resolved HF checkpoint '{model_id}' → registry '{_lookup_id}'")
                break

    try:
        match = resolve_model_for_tier(
            _lookup_id, tier,
            quant=quant,
            checkpoint=model_id if _user_gave_hf_checkpoint else None,
        )
    except (KeyError, ValueError) as exc:
        bt.logging.warning(f"Model {model_id} not in registry for this tier ({exc}), using defaults")
        resolved_quant = quant or "fp16"
        if max_context_len is None:
            bt.logging.error(f"Model '{model_id}' not in registry — --max-context-len is required")
            sys.exit(1)
        return ResolvedModel(
            model_id=resolved_model_id,
            quant=resolved_quant,
            max_context_len=max_context_len,
            model_source="cli",
            quant_source="cli" if quant else "default",
            context_source="cli",
        )

    # Registry matched — use the tier config's checkpoint for shortnames,
    # keep the user's HF checkpoint if they specified one explicitly.
    if not _user_gave_hf_checkpoint:
        resolved_model_id = match.config.checkpoint

    resolved_quant = quant
    quant_source = "cli"
    if resolved_quant is None:
        # Pick the best quant from this tier config (first in list = preferred)
        if match.config.quant_configs:
            resolved_quant = match.config.quant_configs[0].quant
            quant_source = "registry"
        else:
            resolved_quant = "fp16"
            quant_source = "default"

    resolved_context = max_context_len
    context_source = "cli"
    if resolved_context is None:
        # Look up the max_model_len for this quant in the tier config
        for qo in match.config.quant_configs:
            if qo.quant == resolved_quant:
                resolved_context = qo.max_model_len
                context_source = "registry"
                break

        if resolved_context == 0:
            # 0 = let vLLM auto-fit (don't estimate, don't cap)
            context_source = "auto(vllm)"
        elif resolved_context is None:
            # Not found in registry — fall back to estimate
            is_moe = match.model.architecture == "moe"
            resolved_context = estimate_effective_context(
                match.model.total_params_b,
                match.model.active_params_b,
                resolved_quant,
                tier.value,
                match.model.native_context_len,
                is_moe=is_moe,
            )
            context_source = "registry(estimated)"

    bt.logging.info(f"Resolved: model={resolved_model_id} (cli={model_id}) quant={resolved_quant} (src={quant_source}) ctx={resolved_context} (src={context_source})")

    if capacity_audit_required:
        on_chain_models = (
            _get_on_chain_models(chain_config, subtensor_network)
            if chain_config else None
        )
        if chain_config and on_chain_models is None:
            bt.logging.error(
                "Capacity audit is enabled but on-chain ModelRegistry "
                "recommendations could not be loaded"
            )
            sys.exit(1)
        ok, reason, expected = validate_capacity_recommended_model(
            model_id=resolved_model_id,
            quant=resolved_quant,
            max_context_len=int(resolved_context or 0),
            vram_gb=capacity_gate_vram_gb(gpu_info),
            on_chain_models=on_chain_models,
        )
        if not ok:
            expected_text = ""
            if expected is not None:
                expected_text = (
                    f" expected model={expected.model_id} "
                    f"quant={expected.quant} ctx>={expected.max_context_len}"
                )
            bt.logging.error(f"{reason}.{expected_text}")
            sys.exit(1)

    return ResolvedModel(
        model_id=resolved_model_id,
        quant=resolved_quant,
        max_context_len=resolved_context,
        model_source="cli",
        quant_source=quant_source,
        context_source=context_source,
    )


def add_model_args(parser) -> None:
    """Add --model-id, --auto, --category, --quant, --max-context-len to an argparser."""
    parser.add_argument(
        "--model-id", default=None,
        help="Model ID for serving. Use 'auto' to auto-select best model for GPU "
             "(same as --auto flag)",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-select best model for detected GPU (same as --model-id auto)",
    )
    parser.add_argument(
        "--category", default=None,
        choices=["general", "coding", "agent_swe", "reasoning", "multimodal"],
        help="Filter auto-selection by model category",
    )
    parser.add_argument(
        "--quant", default=None,
        help="Quantization mode (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-context-len", type=int, default=None,
        help="Max context length in tokens (auto-detected if omitted)",
    )
