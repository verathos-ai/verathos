"""CLI for the model registry.

Usage::

    # Auto-detect GPU and list models that fit
    python -m verallm.registry

    # List models for a specific VRAM tier
    python -m verallm.registry --tier 80

    # Show all models (all tiers + multi-GPU)
    python -m verallm.registry --all

    # Resolve a specific model for your GPU
    python -m verallm.registry --resolve qwen3-8b

    # Recommend optimal model for your GPU (ranked by utility)
    python -m verallm.registry --recommend

    # Only show verified models in recommendations
    python -m verallm.registry --recommend --verified-only

    # Recommend by category (coding, reasoning, general, agent_swe, multimodal)
    python -m verallm.registry --recommend --category coding
"""

from __future__ import annotations

import argparse
import sys

from verallm.registry.models import (
    ALL_MODELS,
    VRAMTier,
    ModelCategory,
    get_models_up_to_tier,
    resolve_model_for_tier,
    recommend_models,
)


def _tier_from_int(value: int) -> VRAMTier:
    """Convert an integer GB value to the nearest VRAMTier."""
    for t in sorted(VRAMTier, key=lambda t: t.value):
        if t.value >= value and t != VRAMTier.MULTI_GPU:
            return t
    return VRAMTier.GB_288


def _format_model_line(
    name: str,
    checkpoint: str,
    quant: str,
    tier_label: str,
    verified: bool,
    native: bool,
    max_model_len: int,
) -> str:
    v_mark = "[verified]" if verified else ""
    n_mark = "" if native else " (inherited)"
    return (
        f"  {name:<45s} {tier_label:<8s} {quant:<12s} "
        f"ctx={max_model_len:<6d} {v_mark}{n_mark}"
    )


def cmd_list(tier: VRAMTier, show_all: bool) -> None:
    """List models available for a tier."""
    if show_all:
        print(f"All {len(ALL_MODELS)} registered models:\n")
        print(f"  {'Model':<45s} {'MinTier':<8s} {'Arch':<6s} "
              f"{'Params':<10s} {'Verified':<10s} {'MultiGPU'}")
        print(f"  {'-'*45} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        for m in ALL_MODELS:
            min_t = min(tc.tier for tc in m.tier_configs)
            multi = "YES (unsupported)" if m.multi_gpu else ""
            params = f"{m.total_params_b}B"
            if m.active_params_b != m.total_params_b:
                params += f" ({m.active_params_b}B active)"
            print(
                f"  {m.name:<45s} {min_t.name:<8s} {m.architecture:<6s} "
                f"{params:<10s} {'yes' if m.verified_inference else '':<10s} {multi}"
            )
        return

    matches = get_models_up_to_tier(tier)
    native_count = sum(1 for m in matches if m.native)
    inherited_count = len(matches) - native_count

    print(f"Models for {tier.name} ({tier.value} GB): "
          f"{len(matches)} total ({native_count} native, {inherited_count} inherited)\n")

    print(f"  {'Model':<45s} {'Tier':<8s} {'Quant':<12s} "
          f"{'Context':<8s} {'Status'}")
    print(f"  {'-'*45} {'-'*8} {'-'*12} {'-'*8} {'-'*20}")

    for tm in matches:
        for qo in tm.config.quant_configs:
            print(_format_model_line(
                tm.model.name,
                tm.config.checkpoint,
                qo.quant,
                tm.config.tier.name,
                tm.model.verified_inference,
                tm.native,
                qo.max_model_len,
            ))

    if inherited_count:
        print(f"\n  * (inherited) = uses config from a lower tier; "
              f"runs fine but not tier-optimised")


def _print_rec_table(
    recs: list,
    *,
    start: int = 1,
    header: bool = True,
) -> None:
    """Print a recommendation table (shared by flat and grouped views)."""
    if header:
        print(f"  {'#':<3s} {'Model':<44s} {'Quant':<6s} {'Wt GB':>5s} "
              f"{'EstCtx':>7s} {'Ctx%':>5s} {'VRAM%':>5s} {'Util':>6s}")
        print(f"  {'-'*3} {'-'*44} {'-'*6} {'-'*5} "
              f"{'-'*7} {'-'*5} {'-'*5} {'-'*6}")

    for i, r in enumerate(recs, start):
        native_mark = "" if r.native else " *"
        print(f"  {i:<3d} {r.model.name:<44s} {r.quant:<6s} {r.est_weight_gb:>5.1f} "
              f"{r.est_context:>7d} {r.context_utilization:>5.0%} "
              f"{r.vram_utilization:>5.0%} {r.utility:>6.2f}{native_mark}")


def _print_legend(recs: list) -> None:
    """Print shared legend footer."""
    inherited = sum(1 for r in recs if not r.native)
    print(f"\n  Legend:")
    print(f"    Quant  — quantization (bf16/fp16 = full, fp8 = 8-bit, int4 = 4-bit)")
    print(f"    Wt GB  — estimated GPU memory for model weights")
    print(f"    EstCtx — estimated max context (tokens) that fits in remaining VRAM")
    print(f"    Ctx%   — EstCtx / model's native context window")
    print(f"    VRAM%  — weight memory / total GPU VRAM")
    print(f"    Util   — utility score: log2(quality_params)^1.8 x log2(ctx/1K) x quant x gen_quality")
    print(f"             quality_params = active_params (dense) or moe_dense_equivalent (MoE)")
    if inherited:
        print(f"    *      — inherited config (from lower tier, not tier-optimised)")


_CATEGORY_LABELS: dict[str, str] = {
    "general": "General Purpose",
    "coding": "Coding",
    "agent_swe": "Agentic / SWE",
    "reasoning": "Reasoning",
    "multimodal": "Multimodal",
}


def cmd_recommend(
    tier: VRAMTier,
    verified_only: bool,
    category: ModelCategory | None = None,
) -> None:
    """Rank models by estimated utility for a tier, optionally by category."""
    if category is not None:
        # Single-category flat view
        recs = recommend_models(tier, verified_only=verified_only, category=category)
        if not recs:
            print(f"No {category.value} models fit this tier.")
            return

        cat_label = _CATEGORY_LABELS.get(category.value, category.value)
        v_label = "verified " if verified_only else ""
        print(f"Recommended {v_label}{cat_label} models for "
              f"{tier.name} ({tier.value} GB):\n")
        _print_rec_table(recs)
        _print_legend(recs)

        top = recs[0]
        print(f"\n  Best {cat_label.lower()} pick: {top.model.name} ({top.quant})")
        print(f"    Estimated {top.est_context:,} token context "
              f"({top.context_utilization:.0%} of native "
              f"{top.model.native_context_len:,})")
        print(f"\n  To start the miner:")
        print(f"    python -m verallm.api.server --model-id {top.model.id}")
        return

    # Default: grouped by category, showing top pick per category + full list
    all_recs = recommend_models(tier, verified_only=verified_only)
    if not all_recs:
        print("No models fit this tier.")
        return

    v_label = "verified " if verified_only else ""

    # -- Top pick per category --
    best_per_cat: dict[str, object] = {}
    for r in all_recs:
        for cat in r.model.categories:
            if cat.value not in best_per_cat:
                best_per_cat[cat.value] = r

    print(f"Best {v_label}model per category for "
          f"{tier.name} ({tier.value} GB):\n")
    for cat in ModelCategory:
        rec = best_per_cat.get(cat.value)
        if rec is None:
            continue
        cat_label = _CATEGORY_LABELS.get(cat.value, cat.value)
        native_mark = "" if rec.native else " *"
        print(f"  {cat_label:<20s} {rec.model.name:<40s} "
              f"{rec.quant:<6s} util={rec.utility:.2f}{native_mark}")

    # -- Full ranked list --
    print(f"\nFull ranking ({len(all_recs)} models):\n")
    _print_rec_table(all_recs)
    _print_legend(all_recs)

    top = all_recs[0]
    print(f"\n  Overall best: {top.model.name} ({top.quant})")
    print(f"    Estimated {top.est_context:,} token context "
          f"({top.context_utilization:.0%} of native "
          f"{top.model.native_context_len:,})")
    print(f"\n  To start the miner:")
    print(f"    python -m verallm.api.server --model-id {top.model.id}")
    print(f"\n  Filter by category:")
    print(f"    python -m verallm.registry --recommend --category coding")


def cmd_resolve(model_id: str, tier: VRAMTier) -> None:
    """Resolve a model ID to concrete config for a tier."""
    try:
        tm = resolve_model_for_tier(model_id, tier)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    m = tm.model
    tc = tm.config
    print(f"Model:       {m.name} ({m.id})")
    print(f"Tier:        {tier.name} ({tier.value} GB)")
    print(f"Native:      {'yes' if tm.native else 'no (inherited from ' + tc.tier.name + ')'}")
    print(f"Checkpoint:  {tc.checkpoint}")
    print(f"Quant options:")
    for qo in tc.quant_configs:
        print(f"  {qo.quant:<6s} max_model_len={qo.max_model_len:,}")
    print(f"Architecture:{m.architecture}")
    print(f"Params:      {m.total_params_b}B total, {m.active_params_b}B active")
    print(f"Verified:    {'yes' if m.verified_inference else 'no'}")
    if tc.notes:
        print(f"Notes:       {tc.notes}")
    print(f"\nTo start the miner:")
    print(f"  python -m verallm.api.server --model-id {m.id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m verallm.registry",
        description="Verathos model registry — list and resolve models for your GPU.",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="VRAM tier in GB (e.g. 24, 48, 80). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="show_all",
        help="Show all models across all tiers (including multi-GPU).",
    )
    parser.add_argument(
        "--resolve", metavar="MODEL_ID",
        help="Resolve a specific model ID to its config for the tier.",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Rank models by estimated utility for optimal GPU usage.",
    )
    parser.add_argument(
        "--verified-only", action="store_true",
        help="Only show verified models (use with --recommend or --list).",
    )
    cat_choices = [c.value for c in ModelCategory]
    parser.add_argument(
        "--category", choices=cat_choices, default=None,
        help=f"Filter recommendations by category ({', '.join(cat_choices)}).",
    )
    args = parser.parse_args()

    # Determine tier
    if args.tier is not None:
        tier = _tier_from_int(args.tier)
    else:
        try:
            from verallm.registry.gpu import detect_vram_tier, detect_gpu_info
            info = detect_gpu_info()
            tier = info["tier"]
            print(f"Detected GPU: {info['name']} ({info['vram_gb']} GB, "
                  f"tier {tier.name}, CC {info['compute_capability']})\n")
        except (RuntimeError, ImportError):
            print("No GPU detected, defaulting to tier GB_24 (RTX 4090).\n"
                  "Use --tier <GB> to specify manually.\n")
            tier = VRAMTier.GB_24

    # Parse category
    category = ModelCategory(args.category) if args.category else None

    if args.resolve:
        cmd_resolve(args.resolve, tier)
    elif args.recommend:
        cmd_recommend(tier, args.verified_only, category)
    else:
        cmd_list(tier, args.show_all)


if __name__ == "__main__":
    main()
