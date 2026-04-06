"""
Central model registry for Verathos.

Single source of truth for all supported models across inference and training.
Every model runs on a **miner** — the distinction is the VRAM tier required.
Multi-GPU models are included but NOT YET SUPPORTED by the protocol.

Usage::

    from verallm.registry.models import (
        ALL_MODELS, get_model, get_models_for_tier, VRAMTier,
    )

    # All models that fit on an RTX 4090
    for m in get_models_for_tier(VRAMTier.GB_24):
        print(m.name)

    # Backward-compat for Streamlit inference app
    from verallm.registry.models import get_inference_presets
    MODEL_PRESETS = get_inference_presets()  # same dict format as before
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VRAMTier(IntEnum):
    """VRAM tiers for GPU hardware classes.

    The integer value is the VRAM in GB, enabling natural comparison:
    ``if available_vram >= tier.value: ...``

    GPU models per tier:

    ====  ========  ==================================================
    Tier  VRAM      GPUs
    ====  ========  ==================================================
    12    12 GB     RTX 3060 12GB, RTX 4070, RTX 4070 Ti
    16    16 GB     RTX 4060 Ti 16GB, RTX 5060, A4000, T4
    24    24 GB     RTX 4090, RTX 3090/Ti, L4, A30, A10, RTX A5000
    32    32 GB     RTX 5090
    48    48 GB     RTX A6000, RTX 6000 Ada, RTX PRO 6000 Ada,
                    A40, L40, L40S
    80    80 GB     A100 80 GB SXM, H100 80 GB
    96    96 GB     RTX PRO 6000 Blackwell
    141   141 GB    H200 SXM
    192   192 GB    B200
    288   288 GB    B300 Blackwell Ultra
    ====  ========  ==================================================
    """
    GB_12     = 12    # RTX 3060 12GB, RTX 4070, RTX 4070 Ti
    GB_16     = 16    # RTX 4060 Ti 16GB, RTX 5060, A4000, T4
    GB_24     = 24    # RTX 4090, RTX 3090/Ti, L4, A30, RTX A5000
    GB_32     = 32    # RTX 5090
    GB_48     = 48    # RTX A6000, RTX 6000 Ada, A40, L40, L40S
    GB_80     = 80    # A100 80 GB SXM, H100 80 GB
    GB_96     = 96    # RTX PRO 6000 Blackwell (96 GB GDDR7)
    GB_141    = 141   # H200 SXM
    GB_192    = 192   # B200 (192 GB HBM3e)
    GB_288    = 288   # B300 Blackwell Ultra (288 GB HBM3e)
    MULTI_GPU = 999   # Multi-GPU clusters (not a real VRAM size)


class ModelCategory(str, Enum):
    """Broad capability category."""
    GENERAL    = "general"
    CODING     = "coding"
    AGENT_SWE  = "agent_swe"
    REASONING  = "reasoning"
    MULTIMODAL = "multimodal"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantOption:
    """A quantization mode with its per-tier max context length.

    Each (tier, checkpoint, quant) combination has a different amount of
    VRAM left for KV cache, so ``max_model_len`` must be per-quant.
    """
    quant: str                         # e.g. "int4", "fp8", "bf16"
    max_model_len: int                 # Max context tokens for this quant on this tier


@dataclass(frozen=True)
class TierConfig:
    """How a model runs on a specific VRAM tier.

    Each model can have multiple TierConfig entries — one per tier it supports,
    each with the appropriate checkpoint and quantization options.

    ``quant_configs`` holds one :class:`QuantOption` per supported quantization
    mode, each with its own ``max_model_len`` — the estimated safe context cap
    for that quant at this tier's VRAM budget.  Different quants leave different
    amounts of VRAM for KV cache (e.g. INT4 weights are half the size of FP8,
    freeing more room for context).

    When a model is served on a **higher** tier than its config targets
    (inherited / non-native match), the cap is ignored and vLLM is left
    to auto-size — the GPU has more VRAM than needed, so the model's
    full native context can typically be used.  Set to ``0`` to always
    let vLLM decide (useful for small models that fit any tier easily).
    """
    tier: VRAMTier
    checkpoint: str                    # HuggingFace checkpoint ID
    quant_configs: tuple[QuantOption, ...]  # Per-quant max context caps
    label: str = ""                    # Display label for UI (auto-generated if empty)
    notes: str = ""


@dataclass(frozen=True)
class TrainingConfig:
    """Training-specific metadata (attached to models that support training)."""
    default_steps: int = 20
    default_rounds: int = 2
    default_quant: str = "int8"
    size_b: float = 0.0               # Approximate base params in billions


@dataclass(frozen=True)
class ModelEntry:
    """A model in the Verathos registry.

    One entry per logical model.  The ``tier_configs`` list describes which
    VRAM tiers can run it and with what quantization / checkpoint.
    """
    id: str                            # Stable machine-readable ID
    name: str                          # Human-readable display name
    base_model: str                    # Base HuggingFace model ID (unquantized)
    architecture: str                  # "dense" or "moe"
    categories: tuple[ModelCategory, ...]
    tier_configs: tuple[TierConfig, ...]

    # Size metadata
    total_params_b: float = 0.0        # Total parameters in billions
    active_params_b: float = 0.0       # Active parameters per token (= total for dense)
    num_experts: int = 0               # MoE expert count (0 = dense)
    native_context_len: int = 32768    # Model's trained context window (max_position_embeddings)

    # Model generation quality multiplier (affects utility ranking).
    # 1.0 = modern baseline; <1 penalises older/superseded architectures.
    generation_quality: float = 1.0

    # MoE dense equivalent: the dense model size (in B) that this MoE model
    # matches in benchmark quality.  Used instead of ``active_params_b`` in
    # the utility formula so MoE models aren't unfairly penalised.
    # Set to 0.0 for dense models (falls back to ``active_params_b``).
    moe_dense_equivalent: float = 0.0

    # Multi-GPU flag — NOT YET SUPPORTED by the protocol
    multi_gpu: bool = False

    # Verification status
    verified_inference: bool = False    # True = passes full VeraLLM verification

    # Training support (None = inference only)
    training: TrainingConfig | None = None

    # Metadata
    family: str = ""                   # e.g. "qwen3", "llama4"
    provider: str = ""                 # e.g. "Qwen", "Meta", "Mistral"
    notes: str = ""


# ---------------------------------------------------------------------------
# Webapp backward-compat DTO (same fields as old config.py ModelPreset)
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    """Lightweight DTO used by the webapp backend."""
    id: str
    label: str
    model_name: str
    quant_options: tuple[QuantOption, ...]
    category: str


@dataclass(frozen=True)
class TierMatch:
    """Result of matching a model against a requested VRAM tier.

    Returned by :func:`get_models_up_to_tier`.

    Attributes:
        model: The matched model entry.
        config: The best TierConfig for the requested tier (highest tier <= requested).
        native: ``True`` if the config targets the exact requested tier.
                ``False`` if inherited from a lower tier — the model *can* run
                but isn't specifically optimized/tested for this hardware.
    """
    model: ModelEntry
    config: TierConfig
    native: bool


# ===================================================================
# ALL_MODELS — the complete registry
# ===================================================================
#
# Organisation:
#   1. Verified 24 GB models (existing, tested on RTX 4090)
#   2. Training-only small models
#   3. New single-GPU models from Excel (higher tiers, unverified)
#   4. Multi-GPU models (NOT YET SUPPORTED)
#
# Checkpoint IDs marked with [VERIFIED] have been tested end-to-end.
# Others use base model IDs; quantized checkpoints TBD when testing.
# ===================================================================

ALL_MODELS: tuple[ModelEntry, ...] = (

    # ================================================================
    # 24B-class Dense — Verified
    # ================================================================

    ModelEntry(
        id="mistral-small-3.1-24b-gptq",
        name="Mistral-Small-3.1-24B GPTQ",
        base_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "ISTA-DASLab/Mistral-Small-3.1-24B-Instruct-2503-GPTQ-4b-128g",  # [VERIFIED] [VERIFIED L40S] [VERIFIED H100]
                (QuantOption("int4", 33968),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Mistral-Small-3.1-24B GPTQ (Dense, 40L)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "ISTA-DASLab/Mistral-Small-3.1-24B-Instruct-2503-GPTQ-4b-128g",
                (QuantOption("int4", 45056),),
                notes="Same GPTQ checkpoint; extra 8 GB → more KV headroom",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                (QuantOption("fp8", 52224), QuantOption("int4", 88064)),
                notes="FP8 runtime quant or INT4 for max context",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                (QuantOption("fp8", 131072),),
                notes="FP8 — full native 131K context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                (QuantOption("bf16", 131072),),
                notes="BF16 max quality — full native context",
            ),
        ),
        total_params_b=24.0, active_params_b=24.0,
        native_context_len=131072,
        generation_quality=0.90,  # MMLU-Pro 66.8, GPQA 46.0 — below Qwen3 peers (~81+)
        verified_inference=True,
        family="mistral-small", provider="Mistral",
    ),

    ModelEntry(
        id="devstral-small-2-24b",
        name="Devstral-Small-2-24B",
        base_model="mistralai/Devstral-Small-2-24B-Instruct-2512",
        architecture="dense",
        categories=(ModelCategory.CODING, ModelCategory.AGENT_SWE),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit",  # [VERIFIED]
                (QuantOption("int4", 27280),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Devstral-Small-2-24B AWQ (Dense, Agent/SWE)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "mistralai/Devstral-Small-2-24B-Instruct-2512",
                (QuantOption("int4", 39936),),
                notes="INT4 only — FP8 (24 GB) leaves <2 GB KV on 32 GB",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "mistralai/Devstral-Small-2-24B-Instruct-2512",  # [VERIFIED L40S]
                (QuantOption("fp8", 45056),),
                notes="FP8 runtime quant (CutlassFP8 kernel on Ada+)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "mistralai/Devstral-Small-2-24B-Instruct-2512",  # [VERIFIED H100]
                (QuantOption("fp8", 125952),),
                notes="FP8 — plenty of VRAM for long context",
            ),
        ),
        total_params_b=24.0, active_params_b=24.0,
        native_context_len=262144,  # Mistral advertises 256K; config.json 393K is rope headroom
        generation_quality=1.0,
        verified_inference=True,
        family="devstral", provider="Mistral",
    ),

    # ================================================================
    # MoE GPTQ — Verified
    # ================================================================

    ModelEntry(
        id="qwen3-coder-30b-a3b-gptq",
        name="Qwen3-Coder-30B-A3B GPTQ",
        base_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        architecture="moe",
        categories=(ModelCategory.CODING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq",  # [VERIFIED]
                (QuantOption("int4", 44912),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Qwen3-Coder-30B-A3B GPTQ (MoE, Coding)",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq",  # [VERIFIED L40S]
                (QuantOption("int4", 192512),),
                label="Qwen3-Coder-30B-A3B GPTQ (MoE, Coding)",
                notes="INT4 GPTQ; ~15 GB weights → room for 192K context on 48 GB.",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                (QuantOption("fp8", 262144),),
                notes="FP8 runtime — full native 262K context",
            ),
        ),
        total_params_b=30.0, active_params_b=3.0, num_experts=128,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=14.0,  # Benchmarks ≈ Qwen3-14B dense (MMLU 81+)
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="qwen3-30b-a3b-2507-gptq",
        name="Qwen3-30B-A3B-2507 GPTQ",
        base_model="Qwen/Qwen3-30B-A3B",
        architecture="moe",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4",  # [VERIFIED]
                (QuantOption("int4", 44912),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Qwen3-30B-A3B-2507 GPTQ (MoE, 128e)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4",
                (QuantOption("int4", 97280),),
                notes="Same GPTQ checkpoint; extra 8 GB → ~97K context",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4",  # [VERIFIED RTX 6000 Ada]
                (QuantOption("int4", 262144),),  # measured: full 256K context fits, KV=278,208
                notes="INT4 — full native 256K context on 48 GB (MoE is compact)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3-30B-A3B",
                (QuantOption("fp8", 262144),),
                notes="FP8 runtime — full native 262K context",
            ),
        ),
        total_params_b=30.0, active_params_b=3.0, num_experts=128,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=14.0,  # Benchmarks ≈ Qwen3-14B dense
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="qwen3-30b-a3b-gptq",
        name="Qwen3-30B-A3B GPTQ",
        base_model="Qwen/Qwen3-30B-A3B",
        architecture="moe",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen3-30B-A3B-GPTQ-Int4",  # [VERIFIED]
                (QuantOption("int4", 40960),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Qwen3-30B-A3B GPTQ (MoE, 128e)",
            ),
        ),
        total_params_b=30.0, active_params_b=3.0, num_experts=128,
        native_context_len=32768,  # Qwen advertises 32K; config.json 40960 is prompt+output allocation
        generation_quality=1.0,
        moe_dense_equivalent=14.0,  # Benchmarks ≈ Qwen3-14B dense
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    # ================================================================
    # Dense 14B — Verified
    # ================================================================

    ModelEntry(
        id="deepseek-r1-distill-qwen-14b-gptq",
        name="DeepSeek-R1-Distill-Qwen-14B GPTQ",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",  # [VERIFIED]
                (QuantOption("int4", 56928),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="DeepSeek-R1-Distill-Qwen-14B GPTQ (Dense, Reasoning)",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                (QuantOption("fp8", 131072),),
                notes="FP8 runtime — full native 131K context",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                (QuantOption("bf16", 131072),),
                notes="BF16 max quality — full native context",
            ),
        ),
        total_params_b=14.0, active_params_b=14.0,
        native_context_len=131072,
        generation_quality=0.85,  # Qwen2.5 base arch; R1 reasoning distill keeps value
        verified_inference=True,
        family="deepseek-r1", provider="DeepSeek",
    ),

    ModelEntry(
        id="deepseek-r1-distill-qwen-14b-nf4",
        name="DeepSeek-R1-Distill-Qwen-14B NF4",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # [VERIFIED]
                (QuantOption("nf4", 56928),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="DeepSeek-R1-Distill-Qwen-14B NF4 (Dense, Reasoning)",
            ),
        ),
        total_params_b=14.0, active_params_b=14.0,
        native_context_len=131072,
        generation_quality=0.85,  # Qwen2.5 base arch; R1 reasoning distill keeps value
        verified_inference=True,
        family="deepseek-r1", provider="DeepSeek",
        notes="NF4 via bitsandbytes (uses base model checkpoint)",
    ),

    ModelEntry(
        id="phi-4-reasoning-plus",
        name="Phi-4-reasoning-plus",
        base_model="microsoft/Phi-4-reasoning-plus",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "meteorain/Phi-4-reasoning-plus-GPTQ-OpenR1",
                (QuantOption("int4", 61440),),
                label="Phi-4-reasoning-plus GPTQ (Dense, Reasoning)",
                notes="Compressed-tensors INT4; beats DS-R1-Distill-70B on reasoning benchmarks",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "cortecs/Phi-4-reasoning-plus-FP8-Dynamic",
                (QuantOption("fp8", 65536),),
                notes="FP8 dynamic — full native 64K context",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "microsoft/Phi-4-reasoning-plus",
                (QuantOption("bf16", 65536),),
                notes="BF16 max quality — full native context",
            ),
        ),
        total_params_b=14.7, active_params_b=14.7,
        native_context_len=65536,
        generation_quality=1.05,  # AIME 81.3 at 14B — exceptional reasoning per param
        family="phi-4", provider="Microsoft",
        notes="MIT license; extended to 64K via outcome-based RL",
    ),

    ModelEntry(
        id="qwen3-14b-gptq",
        name="Qwen3-14B GPTQ",
        base_model="Qwen/Qwen3-14B",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "JunHowie/Qwen3-14B-GPTQ-Int4",  # [VERIFIED]
                (QuantOption("int4", 40960),),  # measured: full native context fits on RTX 4090 @ 0.90 util
                label="Qwen3-14B GPTQ (Dense, 40L)",
            ),
        ),
        total_params_b=14.0, active_params_b=14.0,
        native_context_len=40960,  # HF config max_position_embeddings (Qwen advertises 32K but allocates 40K)
        generation_quality=1.0,
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    # ================================================================
    # Dense 7-8B — Verified (inference + training)
    # ================================================================

    ModelEntry(
        id="qwen3-8b",
        name="Qwen3-8B",
        base_model="Qwen/Qwen3-8B",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen3-8B",  # [VERIFIED]
                (QuantOption("fp16", 24368), QuantOption("int4", 40960)),  # fp16 @ 0.85 util, int4 @ 0.90 util — measured via vLLM auto-fit on RTX 4090
                label="Qwen3-8B (Dense)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/Qwen3-8B",
                (QuantOption("fp16", 40960), QuantOption("int4", 40960)),
                notes="FP16 at full native context (vs 24K on 24 GB)",
            ),
        ),
        total_params_b=8.0, active_params_b=8.0,
        native_context_len=40960,  # HF config max_position_embeddings (Qwen advertises 32K but allocates 40K)
        generation_quality=1.0,
        verified_inference=True,
        training=TrainingConfig(default_steps=20, default_rounds=2,
                                default_quant="int8", size_b=8),
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="deepseek-r1-distill-qwen-7b",
        name="DeepSeek-R1-Distill-Qwen-7B",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # [VERIFIED]
                (QuantOption("fp16", 81424), QuantOption("int4", 131072)),  # fp16 @ 0.85 util, int4 @ 0.90 util — measured via vLLM auto-fit on RTX 4090
                label="DeepSeek-R1-Distill-Qwen-7B (Dense)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                (QuantOption("fp16", 131072), QuantOption("int4", 131072)),
                notes="FP16 at full native 131K context (vs 81K on 24 GB)",
            ),
        ),
        total_params_b=7.0, active_params_b=7.0,
        native_context_len=131072,
        generation_quality=0.85,  # Qwen2.5 base arch; R1 reasoning distill keeps value
        verified_inference=True,
        training=TrainingConfig(default_steps=20, default_rounds=2,
                                default_quant="int8", size_b=7),
        family="deepseek-r1", provider="DeepSeek",
    ),

    ModelEntry(
        id="deepseek-r1-0528-qwen3-8b",
        name="DeepSeek-R1-0528-Qwen3-8B",
        base_model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                (QuantOption("fp16", 35840), QuantOption("int4", 131072)),
                label="DeepSeek-R1-0528-Qwen3-8B (Dense, Reasoning)",
                notes="Distilled from R1-0528 (improved May 2025 checkpoint)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                (QuantOption("fp16", 131072), QuantOption("int4", 131072)),
                notes="FP16 at full native 131K context",
            ),
        ),
        total_params_b=8.0, active_params_b=8.0,
        native_context_len=131072,
        generation_quality=1.05,  # AIME 86.0 at 8B — matches 235B models on reasoning
        family="deepseek-r1", provider="DeepSeek",
        notes="Distilled from R1-0528; uses Qwen3-8B base + DeepSeek tokenizer",
    ),

    ModelEntry(
        id="qwen2.5-7b-instruct",
        name="Qwen2.5-7B-Instruct",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen2.5-7B-Instruct",  # [VERIFIED]
                (QuantOption("fp16", 32768), QuantOption("int4", 32768)),  # fp16 @ 0.85 util (full native), int4 @ 0.90 util
                label="Qwen2.5-7B-Instruct (Dense)",
            ),
        ),
        total_params_b=7.0, active_params_b=7.0,
        native_context_len=32768,
        generation_quality=0.80,  # Superseded by Qwen3/3.5
        verified_inference=True,
        training=TrainingConfig(default_steps=20, default_rounds=2,
                                default_quant="int8", size_b=7),
        family="qwen2.5", provider="Qwen",
    ),

    ModelEntry(
        id="mistral-7b-v0.2",
        name="Mistral-7B-Instruct-v0.2",
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "mistralai/Mistral-7B-Instruct-v0.2",  # [VERIFIED]
                (QuantOption("fp16", 32768), QuantOption("int4", 32768)),  # fp16 @ 0.85 util (full native), int4 @ 0.90 util
                label="Mistral-7B-Instruct-v0.2 (Dense)",
            ),
        ),
        total_params_b=7.0, active_params_b=7.0,
        generation_quality=0.70,  # Jan 2024, oldest model in registry; MMLU ~60.1, 2+ years behind
        verified_inference=True,
        training=TrainingConfig(default_steps=20, default_rounds=2,
                                default_quant="int8", size_b=7),
        family="mistral", provider="Mistral",
    ),

    ModelEntry(
        id="llama-3.1-8b-gptq",
        name="Llama-3.1-8B GPTQ",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",  # [VERIFIED]
                (QuantOption("int4", 120080),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util
                label="Llama-3.1-8B GPTQ (Dense, 32L)",
            ),
        ),
        total_params_b=8.0, active_params_b=8.0,
        native_context_len=131072,
        generation_quality=0.80,  # Jul 2024, superseded by Llama 3.3/4
        verified_inference=True,
        family="llama3", provider="Meta",
    ),

    # ================================================================
    # Pre-quantized checkpoints — Verified
    # ================================================================

    ModelEntry(
        id="qwen2.5-7b-gptq",
        name="Qwen2.5-7B-GPTQ",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # [VERIFIED]
                (QuantOption("int4", 32768),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util (full native)
                label="Qwen2.5-7B-GPTQ (Dense)",
            ),
        ),
        total_params_b=7.0, active_params_b=7.0,
        native_context_len=32768,
        generation_quality=0.80,  # Superseded by Qwen3/3.5
        verified_inference=True,
        family="qwen2.5", provider="Qwen",
    ),

    ModelEntry(
        id="mistral-7b-awq",
        name="Mistral-7B-AWQ",
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",  # [VERIFIED]
                (QuantOption("int4", 32768),),  # measured via vLLM auto-fit on RTX 4090 @ 0.90 util (full native)
                label="Mistral-7B-AWQ (Dense)",
            ),
        ),
        total_params_b=7.0, active_params_b=7.0,
        generation_quality=0.70,  # Jan 2024, oldest model in registry; MMLU ~60.1, 2+ years behind
        verified_inference=True,
        family="mistral", provider="Mistral",
    ),

    # ================================================================
    # Compressed-Tensors GPTQ — Verified
    # ================================================================

    ModelEntry(
        id="qwen3-8b-gptq-ct",
        name="Qwen3-8B GPTQ (Compressed-Tensors)",
        base_model="Qwen/Qwen3-8B",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "RedHatAI/Qwen3-8B-quantized.w4a16",  # [VERIFIED]
                (QuantOption("int4", 40960),),  # measured: full native context fits on RTX 4090 @ 0.90 util
                label="Qwen3-8B GPTQ (Dense, 36L)",
            ),
        ),
        total_params_b=8.0, active_params_b=8.0,
        native_context_len=40960,  # HF config max_position_embeddings (Qwen advertises 32K but allocates 40K)
        generation_quality=1.0,
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="gemma-3-12b-gptq",
        name="Gemma-3-12B GPTQ",
        base_model="google/gemma-3-12b-it",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g",  # [VERIFIED]
                (QuantOption("int4", 131072),),  # measured: full native context fits on RTX 4090 @ 0.90 util
                label="Gemma-3-12B GPTQ (Dense, 48L)",
            ),
        ),
        total_params_b=12.0, active_params_b=12.0,
        native_context_len=131072,
        generation_quality=0.75,  # Superseded by Gemma 4; older arch
        verified_inference=True,
        family="gemma3", provider="Google",
    ),

    ModelEntry(
        id="qwen2.5-coder-14b",
        name="Qwen2.5-Coder-14B-Instruct",
        base_model="Qwen/Qwen2.5-Coder-14B-Instruct",
        architecture="dense",
        categories=(ModelCategory.CODING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
                (QuantOption("int4", 32768),),
                label="Qwen2.5-Coder-14B AWQ (Dense, Coding)",
                notes="Official Qwen AWQ INT4; full native 32K context on 24 GB",
            ),
        ),
        total_params_b=14.0, active_params_b=14.0,
        native_context_len=32768,
        generation_quality=0.85,  # Superseded by Qwen3-Coder/3.5
        family="qwen2.5", provider="Qwen",
    ),

    # ================================================================
    # Training-only small model
    # ================================================================

    ModelEntry(
        id="qwen2.5-0.5b",
        name="Qwen2.5-0.5B",
        base_model="Qwen/Qwen2.5-0.5B",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen2.5-0.5B",
                (QuantOption("fp16", 32768), QuantOption("int8", 32768), QuantOption("int4", 32768)),
            ),
        ),
        total_params_b=0.5, active_params_b=0.5,
        generation_quality=0.80,  # Superseded by Qwen3/3.5
        verified_inference=False,
        training=TrainingConfig(default_steps=50, default_rounds=3,
                                default_quant="fp16", size_b=0.5),
        family="qwen2.5", provider="Qwen",
        notes="Small model for fast training demos",
    ),

    # ================================================================
    # Higher-tier single-GPU models (from Excel, not yet verified)
    # ================================================================

    # ----------------------------------------------------------------
    # GPT-oss — OpenAI open-weight MoE models (Apache 2.0)
    # ----------------------------------------------------------------
    #
    # Both ship with native MXFP4 quantization (MoE experts 4.25 bits,
    # attention/norms in bf16).  AWQ-w4a16 community quants also
    # available for the 120b for wider hardware compatibility.
    #

    ModelEntry(
        id="gpt-oss-20b",
        name="GPT-oss-20B",
        base_model="openai/gpt-oss-20b",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.REASONING),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_16,
                "openai/gpt-oss-20b",
                (QuantOption("mxfp4", 4096),),
                label="GPT-oss-20B MXFP4 (MoE, Reasoning)",
                notes="Native MXFP4 ~14 GB; very tight on 16 GB — minimal KV headroom",
            ),
            TierConfig(
                VRAMTier.GB_24,
                "openai/gpt-oss-20b",
                (QuantOption("mxfp4", 65536),),
                notes="MXFP4 — ~10 GB KV headroom; GQA-8 keeps KV compact",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "openai/gpt-oss-20b",
                (QuantOption("mxfp4", 131072),),
                notes="MXFP4 — full native 131K context easily",
            ),
        ),
        total_params_b=21.0, active_params_b=3.6, num_experts=32,
        native_context_len=131072,
        generation_quality=1.05,  # MMLU 85.3, GPQA 71.5 — extraordinary for 3.6B active
        moe_dense_equivalent=14.0,  # Small but capable reasoning MoE; ~14B dense equiv
        verified_inference=True,
        family="gpt-oss", provider="OpenAI",
        notes="First OpenAI open-weight model; Apache 2.0; MXFP4 native (MoE experts "
              "quantized to 4.25 bits, attn in bf16); 32 experts, top-4 routing",
    ),

    ModelEntry(
        id="gpt-oss-120b",
        name="GPT-oss-120B",
        base_model="openai/gpt-oss-120b",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.REASONING),
        tier_configs=(
            # GB_48 removed: native MXFP4 is ~65 GB, won't fit on 48 GB.
            # twhitworth/gpt-oss-120b-awq-w4a16 is NOT actually AWQ — it's
            # FP16 weights mislabeled (no qweight/qzeros/scales tensors).
            TierConfig(
                VRAMTier.GB_80,
                "openai/gpt-oss-120b",
                (QuantOption("mxfp4", 32768),),
                notes="Native MXFP4 ~65 GB; tight on 80 GB, short context only. "
                      "A100 uses Marlin FP4 decompression (cc 8.0+).",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "openai/gpt-oss-120b",
                (QuantOption("mxfp4", 131072),),
                notes="Native MXFP4 ~65 GB; ~62 GB KV → full 131K context on H200",
            ),
            TierConfig(
                VRAMTier.GB_192,
                "RedHatAI/gpt-oss-120b-FP8-dynamic",
                (QuantOption("fp8", 131072),),
                notes="FP8 dynamic ~120 GB; ~53 GB KV → full 131K context on B200",
            ),
        ),
        total_params_b=117.0, active_params_b=5.1, num_experts=128,
        native_context_len=131072,
        generation_quality=1.05,  # MMLU-Pro 90.0, GPQA 80.9 — frontier-level benchmarks
        moe_dense_equivalent=80.0,  # MMLU-Pro 90.0 surpasses any open 70B dense; AIME ~92% no-tools
        verified_inference=True,
        family="gpt-oss", provider="OpenAI",
        notes="Apache 2.0; MXFP4 native; 128 experts, top-4 routing; GQA group-8; "
              "only 5/36 layers are MoE (rest dense), 16.7B actual params",
    ),

    # ----------------------------------------------------------------
    # Ministral 3 — Mistral dense edge models (Apache 2.0)
    # ----------------------------------------------------------------
    #
    # Ship natively in FP8 format.  All variants have vision (image)
    # capabilities.  Reasoning variants also available on HuggingFace
    # (Ministral-3-{8,14}B-Reasoning-2512).
    #

    ModelEntry(
        id="ministral-3-8b",
        name="Ministral-3-8B-Instruct",
        base_model="mistralai/Ministral-3-8B-Instruct-2512",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_16,
                "mistralai/Ministral-3-8B-Instruct-2512",
                (QuantOption("fp8", 32768),),
                label="Ministral-3-8B FP8 (Dense, General/Vision)",
                notes="Native FP8 ~8 GB; ~6 GB KV headroom on 16 GB",
            ),
            TierConfig(
                VRAMTier.GB_24,
                "mistralai/Ministral-3-8B-Instruct-2512",
                (QuantOption("fp8", 131072),),
                notes="FP8 — full native 131K context on 24 GB",
            ),
        ),
        total_params_b=8.0, active_params_b=8.0,
        native_context_len=131072,
        generation_quality=1.0,
        family="ministral", provider="Mistral",
        notes="Apache 2.0; native FP8 weights; vision (image); "
              "Reasoning variant: Ministral-3-8B-Reasoning-2512",
    ),

    ModelEntry(
        id="ministral-3-14b",
        name="Ministral-3-14B-Instruct",
        base_model="mistralai/Ministral-3-14B-Instruct-2512",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "mistralai/Ministral-3-14B-Instruct-2512",
                (QuantOption("fp8", 49152),),
                label="Ministral-3-14B FP8 (Dense, General/Vision)",
                notes="Native FP8 ~14 GB; ~8 GB KV headroom on 24 GB",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "mistralai/Ministral-3-14B-Instruct-2512",
                (QuantOption("fp8", 131072),),
                notes="FP8 — full native 131K context on 48 GB",
            ),
        ),
        total_params_b=14.0, active_params_b=14.0,
        native_context_len=131072,
        generation_quality=1.0,
        family="ministral", provider="Mistral",
        notes="Apache 2.0; native FP8 weights; vision (image); "
              "Reasoning variant: Ministral-3-14B-Reasoning-2512",
    ),

    ModelEntry(
        id="qwen3-32b",
        name="Qwen3-32B",
        base_model="Qwen/Qwen3-32B",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/Qwen3-32B-AWQ",  # [VERIFIED RTX 5090]
                (QuantOption("int4", 28496),),  # measured via vLLM auto-fit on RTX 5090 @ 0.90 util
                label="Qwen3-32B AWQ (Dense, General/Chat)",
                notes="AWQ INT4 native checkpoint",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/Qwen3-32B-AWQ",  # [VERIFIED RTX 6000 Ada]
                (QuantOption("int4", 40960),),  # measured via vLLM auto-fit @ 0.90 util
                notes="INT4 — full native context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3-32B",
                (QuantOption("int4", 32768), QuantOption("int8", 32768)),
                notes="INT8 preferred quality, INT4 also fits native context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "Qwen/Qwen3-32B",  # [VERIFIED H200]
                (QuantOption("bf16", 32768),),
                notes="BF16 max quality — full native context",
            ),
        ),
        total_params_b=32.0, active_params_b=32.0,
        native_context_len=32768,  # Qwen advertises 32K
        generation_quality=1.0,
        verified_inference=True,
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="qwen2.5-coder-32b",
        name="Qwen2.5-Coder-32B-Instruct",
        base_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        architecture="dense",
        categories=(ModelCategory.CODING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",  # [VERIFIED H200]
                (QuantOption("int4", 21504),),
                label="Qwen2.5-Coder-32B AWQ (Dense, Coding)",
                notes="AWQ INT4 native checkpoint",
            ),
        ),
        total_params_b=32.0, active_params_b=32.0,
        native_context_len=32768,
        generation_quality=0.85,  # Superseded by Qwen3-Coder/3.5
        verified_inference=True,
        family="qwen2.5", provider="Qwen",
    ),

    ModelEntry(
        id="gemma-3-27b",
        name="Gemma-3-27B-IT",
        base_model="google/gemma-3-27b-it",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_32,
                "RedHatAI/gemma-3-27b-it-quantized.w4a16",  # [VERIFIED RTX 5090]
                (QuantOption("int4", 120080),),  # measured via vLLM auto-fit on RTX 5090 @ 0.90 util
                label="Gemma-3-27B GPTQ (Dense, General/Multimodal)",
                notes="Compressed-tensors W4A16; ~13.5 GB weights; aggressive GQA → large KV pool",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "RedHatAI/gemma-3-27b-it-quantized.w4a16",
                (QuantOption("int4", 131072),),  # measured: full 131K native context fits (KV=43,296)
                notes="Compressed-tensors W4A16 — full native 131K context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "google/gemma-3-27b-it",
                (QuantOption("fp8", 114688),),
                notes="FP8 — ~115K context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "google/gemma-3-27b-it",
                (QuantOption("bf16", 131072),),
                notes="BF16 max quality — full native 131K context",
            ),
        ),
        total_params_b=27.0, active_params_b=27.0,
        native_context_len=131072,
        generation_quality=0.75,  # Superseded by Gemma 4; older arch
        verified_inference=True,
        family="gemma3", provider="Google",
        notes="Multimodal (text + image); Gemma license (permissive, commercial OK)",
    ),

    ModelEntry(
        id="llama-3.3-70b-instruct",
        name="Llama-3.3-70B-Instruct",
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        architecture="dense",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            # NOTE: 70B INT4 = ~35 GB weights — does NOT fit usably on 48 GB
            # (only ~4K context).  Minimum viable tier: GB_80.
            TierConfig(
                VRAMTier.GB_80,
                "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",
                (QuantOption("int4", 107296),),
                notes="INT4 Marlin, 107K context at 0.90 util [VERIFIED A100]",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "meta-llama/Llama-3.3-70B-Instruct",
                (QuantOption("fp8", 49152),),
                notes="FP8 — ~48 GB KV headroom",
            ),
            TierConfig(
                VRAMTier.GB_192,
                "meta-llama/Llama-3.3-70B-Instruct",
                (QuantOption("fp8", 93184),),
                notes="FP8 — est ~93K context at 0.85 util; capped below native 131K",
            ),
        ),
        total_params_b=70.0, active_params_b=70.0,
        native_context_len=131072,
        generation_quality=0.90,  # MMLU-Pro 68.9, GPQA 50.5 — mediocre for 70B; Qwen3 MoE far better
        verified_inference=True,
        family="llama3", provider="Meta",
    ),

    ModelEntry(
        id="deepseek-r1-distill-llama-70b",
        name="DeepSeek-R1-Distill-Llama-70B",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            # NOTE: 70B INT4 = ~35 GB weights — does NOT fit usably on 48 GB
            # (only ~6K context).  Minimum viable tier: GB_80.
            TierConfig(
                VRAMTier.GB_80,
                "RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16",  # [VERIFIED A100]
                (QuantOption("int4", 107296),),  # measured via vLLM auto-fit on A100 80GB @ 0.90 util
                notes="INT4 Marlin — 107K context (auto-fit from 131K)",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                (QuantOption("fp8", 56320), QuantOption("int4", 92160)),
                notes="FP8 better quality; INT4 for more context",
            ),
            TierConfig(
                VRAMTier.GB_192,
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                (QuantOption("fp8", 103424),),
                notes="FP8 — ~103K context",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                (QuantOption("bf16", 119808), QuantOption("fp8", 131072)),
                notes="BF16 max quality ~120K; FP8 full native 131K",
            ),
        ),
        total_params_b=70.0, active_params_b=70.0,
        native_context_len=131072,
        generation_quality=0.90,  # AIME 70.0 weak for 70B; beaten by 32B Qwen distill (72.6)
        verified_inference=True,
        family="deepseek-r1", provider="DeepSeek",
        notes="Largest dense reasoning distill; Llama 3.3 license",
    ),

    ModelEntry(
        id="llama-4-scout-17b-16e",
        name="Llama-4-Scout-17B-16E-Instruct",
        base_model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        architecture="moe",
        categories=(ModelCategory.MULTIMODAL,),
        tier_configs=(
            # NOTE: INT4 weights alone are ~54.5 GB — doesn't fit on 48 GB.
            # Lowest tier is GB_80 with INT4 (~13 GB KV headroom).
            TierConfig(
                VRAMTier.GB_80,
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                (QuantOption("int4", 16384),),
                label="Llama-4-Scout-17B-16E INT4 (MoE, Multimodal)",
                notes="INT4 ~54.5 GB weights; ~13 GB KV headroom on 80 GB",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                (QuantOption("fp8", 12288),),
                notes="FP8 (~109 GB); only ~9 GB KV headroom on H200",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                (QuantOption("bf16", 34816),),
                notes="Full precision; enormous 10M native context",
            ),
        ),
        total_params_b=109.0, active_params_b=17.0, num_experts=16,
        native_context_len=10485760,
        generation_quality=1.0,
        moe_dense_equivalent=30.0,  # 16 experts (fewer than 128), ~30B dense equiv
        family="llama4", provider="Meta",
    ),

    ModelEntry(
        id="devstral-2-123b",
        name="Devstral-2-123B",
        base_model="mistralai/Devstral-2-123B-Instruct-2512",
        architecture="dense",
        categories=(ModelCategory.CODING, ModelCategory.AGENT_SWE),
        tier_configs=(
            # NOTE: Base checkpoint ships with native FP8 quantization_config,
            # so BNB NF4 won't work.  FP8 at ~123 GB doesn't fit on H200
            # (only 8 GB KV headroom at 0.95 util).  Use the AWQ INT4
            # checkpoint for GB_141, FP8 for GB_192+.
            TierConfig(
                VRAMTier.GB_141,
                "cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit",  # [VERIFIED H200]
                (QuantOption("int4", 32768),),
                label="Devstral-2-123B AWQ (Dense, Coding/Agent)",
                notes="AWQ INT4 ~61.5 GB; ~33K context on H200",
            ),
            TierConfig(
                VRAMTier.GB_192,
                "mistralai/Devstral-2-123B-Instruct-2512",
                (QuantOption("fp8", 21504),),
                notes="FP8 native weights; higher quality than INT4",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "mistralai/Devstral-2-123B-Instruct-2512",
                (QuantOption("fp8", 69632),),
                notes="FP8 — BF16 (246 GB) doesn't fit; FP8 gives ~120 GB KV headroom",
            ),
        ),
        total_params_b=123.0, active_params_b=123.0,
        native_context_len=262144,
        generation_quality=1.0,
        verified_inference=True,
        family="devstral", provider="Mistral",
    ),

    ModelEntry(
        id="qwen3-235b-a22b",
        name="Qwen3-235B-A22B",
        base_model="Qwen/Qwen3-235B-A22B",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.REASONING),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_192,
                "Qwen/Qwen3-235B-A22B",
                (QuantOption("nvfp4", 32768),),
                label="Qwen3-235B-A22B NVFP4 (MoE, General)",
                notes="NVFP4 (FP4) — Blackwell-targeted",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "Qwen/Qwen3-235B-A22B",
                (QuantOption("fp8", 8192),),
                notes="FP8 — big quality uplift over FP4",
            ),
        ),
        total_params_b=235.0, active_params_b=22.0, num_experts=128,
        native_context_len=32768,  # Qwen advertises 32K
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Flagship MoE, competes with 70B+ dense
        family="qwen3", provider="Qwen",
    ),

    ModelEntry(
        id="deepseek-r1-distill-qwen-32b",
        name="DeepSeek-R1-Distill-Qwen-32B",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
                (QuantOption("int4", 20480),),
                label="DeepSeek-R1-Distill-Qwen-32B INT4 (Dense, Reasoning)",
                notes="INT4 Marlin ~18 GB weights; tight but viable on 24 GB with ~20K context",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                (QuantOption("nf4", 21504),),
                label="DeepSeek-R1-Distill-Qwen-32B NF4 (Dense, Reasoning)",
                notes="NF4 runtime quant via bitsandbytes — fits 32 GB",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # [VERIFIED RTX 6000 Ada]
                (QuantOption("nf4", 87088),),  # measured via vLLM auto-fit (bnb NF4) @ 0.90 util
                notes="NF4 runtime quant — 87K context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
                (QuantOption("int4", 131072),),
                notes="INT4 Marlin, full 131K context at 0.90 util [VERIFIED A100]",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                (QuantOption("fp8", 131072), QuantOption("bf16", 120832)),
                notes="FP8 or BF16 — plenty of VRAM",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                (QuantOption("fp8", 131072), QuantOption("bf16", 131072)),
                label="DeepSeek-R1-Distill-Qwen-32B (Dense, Reasoning)",
                notes="Full native context on B300",
            ),
        ),
        total_params_b=32.0, active_params_b=32.0,
        native_context_len=131072,
        generation_quality=0.85,  # Qwen2.5 base arch; R1 reasoning distill keeps value
        verified_inference=True,
        family="deepseek-r1", provider="DeepSeek",
    ),

    ModelEntry(
        id="qwq-32b",
        name="QwQ-32B",
        base_model="Qwen/QwQ-32B",
        architecture="dense",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/QwQ-32B-AWQ",
                (QuantOption("int4", 25088),),
                label="QwQ-32B AWQ (Dense, Reasoning)",
                notes="Official Qwen AWQ; dedicated reasoning model (92% AIME 2024)",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/QwQ-32B-AWQ",  # [VERIFIED RTX 6000 Ada]
                (QuantOption("int4", 40960),),  # measured via vLLM auto-fit @ 0.90 util
                notes="AWQ INT4 — full native 40K context on 48 GB (KV=93,696)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/QwQ-32B",
                (QuantOption("int4", 121856), QuantOption("fp8", 86016)),
                notes="INT4 for max context, FP8 for quality",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "Qwen/QwQ-32B",
                (QuantOption("fp8", 131072), QuantOption("bf16", 131072)),
                notes="FP8 or BF16 — full native 131K context",
            ),
        ),
        total_params_b=32.0, active_params_b=32.0,
        native_context_len=131072,
        generation_quality=0.85,  # Qwen2.5 base arch; superseded by Qwen3/3.5 reasoning
        verified_inference=True,
        family="qwen-qwq", provider="Qwen",
        notes="Dedicated reasoning model; distinct from Qwen3-32B (chat/general)",
    ),

    ModelEntry(
        id="qwen3-vl-30b-a3b",
        name="Qwen3-VL-30B-A3B-Instruct",
        base_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
                (QuantOption("int4", 32768),),
                label="Qwen3-VL-30B-A3B AWQ (MoE, Multimodal)",
                notes="AWQ INT4; ~15 GB weights + ~3 GB vision encoder; conservative ctx estimate",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
                (QuantOption("int4", 163840),),
                notes="INT4 — ~164K context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3-VL-30B-A3B-Instruct",
                (QuantOption("fp8", 262144),),
                notes="FP8 runtime — full native 256K context",
            ),
        ),
        total_params_b=30.0, active_params_b=3.0, num_experts=128,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=14.0,  # Same decoder as Qwen3-30B-A3B text
        family="qwen3-vl", provider="Qwen",
        notes="Multimodal (text + image + video); same MoE decoder as Qwen3-30B-A3B",
    ),

    ModelEntry(
        id="llama-4-maverick-17b-128e",
        name="Llama-4-Maverick-17B-128E-Instruct",
        base_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_288,
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                (QuantOption("int4", 60416),),
                label="Llama-4-Maverick-128E INT4 (MoE, General/Multimodal)",
                notes="400B MoE / 17B active; INT4 AWQ ~200 GB; 1M native context",
            ),
        ),
        total_params_b=400.0, active_params_b=17.0, num_experts=128,
        native_context_len=1048576,
        generation_quality=1.0,
        moe_dense_equivalent=50.0,  # 128 experts, competitive with 50B+ dense
        family="llama4", provider="Meta",
    ),

    ModelEntry(
        id="nemotron-ultra-253b",
        name="Nemotron-Ultra-253B",
        base_model="nvidia/Nemotron-Ultra-253B-v1",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.REASONING),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_288,
                "nvidia/Nemotron-Ultra-253B-v1",
                (QuantOption("nvfp4", 32768),),
                label="Nemotron-Ultra-253B NVFP4 (Dense, Reasoning)",
                notes="NVFP4 on Blackwell; ~130 GB",
            ),
        ),
        total_params_b=253.0, active_params_b=253.0,
        native_context_len=131072,
        generation_quality=1.0,
        family="nemotron", provider="NVIDIA",
        notes="Borderline single-GPU on B300 at NVFP4",
    ),

    # ================================================================
    # Qwen3.5 — Multimodal MoE/Dense with GDN hybrid attention
    # ================================================================
    #
    # All Qwen3.5 models are natively multimodal (text + image + video)
    # with Gated Delta Networks (GDN) hybrid attention — 3:1 ratio of
    # linear attention (O(1) KV per layer) to standard attention.
    # This reduces real KV cache to ~1/4 of standard transformers,
    # so context estimates below are conservative (vLLM day-0 support).
    #

    ModelEntry(
        id="qwen3.5-35b-a3b",
        name="Qwen3.5-35B-A3B",
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            # NOTE: 21.2 GiB GPU weights — does NOT fit on 24 GB (OOM verified).
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                (QuantOption("int4", 40960),),
                label="Qwen3.5-35B-A3B GPTQ (MoE, 256e, Multimodal)",
                notes="Official GPTQ-Int4 ~21 GB weights; tight on 32 GB; GDN hybrid attn",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                (QuantOption("int4", 262144),),
                notes="Official GPTQ-Int4 — full native 256K context on 48 GB (compact MoE)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3.5-35B-A3B-FP8",
                (QuantOption("fp8", 262144),),
                notes="Pre-quantized FP8 — full native 256K context",
            ),
        ),
        total_params_b=35.0, active_params_b=3.0, num_experts=256,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        moe_dense_equivalent=24.0,  # Benchmarks below Qwen3.5-27B dense; ~24B dense equiv
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention; "
              "256 experts (8 routed + 1 shared); successor to Qwen3-30B-A3B",
    ),

    ModelEntry(
        id="qwen3.5-27b",
        name="Qwen3.5-27B",
        base_model="Qwen/Qwen3.5-27B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.MULTIMODAL),
        tier_configs=(
            # NOTE: Does NOT fit on 24 GB (OOM on CUDA graph capture) or
            # 32 GB (GDN solve_tril Triton kernel OOMs even at gpu_mem=0.95,
            # max_model_len=2048; verified on RTX 5090).  Minimum tier: GB_48.
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/Qwen3.5-27B-GPTQ-Int4",  # [VERIFIED A6000 48GB with AWQ]
                (QuantOption("int4", 85000), QuantOption("fp8", 65536)),
                notes="Official GPTQ-Int4 ~16 GiB weights, 85K KV tokens; FP8 (31 GB) tighter",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3.5-27B",
                (QuantOption("fp8", 262144),),
                notes="FP8 runtime — full native 256K context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "Qwen/Qwen3.5-27B",
                (QuantOption("bf16", 262144),),
                notes="BF16 max quality — full native 256K context",
            ),
        ),
        total_params_b=27.0, active_params_b=27.0,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention; "
              "matches GPT-5-mini on SWE-Bench (72.4)",
    ),

    ModelEntry(
        id="qwen3.5-27b-claude-reasoning",
        name="Qwen3.5-27B Claude Opus Reasoning",
        base_model="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        architecture="dense",
        categories=(ModelCategory.REASONING, ModelCategory.CODING),
        tier_configs=(
            # Same GDN kernel OOM as base Qwen3.5-27B — min tier GB_48.
            TierConfig(
                VRAMTier.GB_48,
                "codgician/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4",
                (QuantOption("int4", 85000),),
                notes="Community GPTQ-Int4 ~16 GiB weights; GDN hybrid attn",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-FP8-Dynamic",
                (QuantOption("fp8", 262144),),
                notes="FP8 dynamic — full native 256K context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
                (QuantOption("bf16", 262144),),
                notes="BF16 max quality — full native 256K context",
            ),
        ),
        total_params_b=27.0, active_params_b=27.0,
        native_context_len=262144,
        generation_quality=1.0,  # Community LoRA from Claude 4.6 Opus; no formal benchmarks yet
        family="qwen3.5", provider="Community (Jackrong)",
        notes="LoRA finetune of Qwen3.5-27B on Claude 4.6 Opus reasoning traces; "
              "GDN hybrid attention; strong agentic/coding behaviour",
    ),

    ModelEntry(
        id="qwen3.5-122b-a10b",
        name="Qwen3.5-122B-A10B",
        base_model="Qwen/Qwen3.5-122B-A10B",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.MULTIMODAL),
        tier_configs=(
            # GB_80 removed: AWQ INT4 ~61 GB weights + CUDA graphs + MoE buffers
            # leaves zero room for KV cache on 80 GB — OOM even for a single token.
            TierConfig(
                VRAMTier.GB_141,
                "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
                (QuantOption("int4", 196608),),
                notes="Official GPTQ-Int4; ample KV headroom on H200",
            ),
            TierConfig(
                VRAMTier.GB_192,
                "Qwen/Qwen3.5-122B-A10B-FP8",
                (QuantOption("fp8", 196608),),
                notes="Pre-quantized FP8 ~122 GB weights",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "Qwen/Qwen3.5-122B-A10B-FP8",
                (QuantOption("fp8", 262144),),
                notes="FP8 — full native 256K context on B300",
            ),
        ),
        total_params_b=122.0, active_params_b=10.0, num_experts=256,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        moe_dense_equivalent=70.0,  # Strongest medium series; leads on knowledge/vision/BFCL
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention; "
              "256 experts (8 routed + 1 shared); strongest Qwen3.5 medium model",
    ),

    # ---- Qwen3.5 Small Dense (0.8B / 2B / 4B / 9B) ----

    ModelEntry(
        id="qwen3.5-9b",
        name="Qwen3.5-9B",
        base_model="Qwen/Qwen3.5-9B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_16,
                "QuantTrio/Qwen3.5-9B-AWQ",
                (QuantOption("int4", 131072),),
                label="Qwen3.5-9B AWQ (Dense, Multimodal)",
                notes="AWQ INT4 ~5 GB weights; GDN hybrid attn → generous KV headroom",
            ),
            TierConfig(
                VRAMTier.GB_24,
                "QuantTrio/Qwen3.5-9B-AWQ",
                (QuantOption("int4", 218000),),
                label="Qwen3.5-9B AWQ (Dense, Multimodal)",
                notes="AWQ INT4; kv_pool ~225K on 24GB (RTX 4090/3090)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "QuantTrio/Qwen3.5-9B-AWQ",
                (QuantOption("int4", 262144),),
                notes="AWQ INT4 full 256K context; kv_pool ~440K on RTX 5090",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "Qwen/Qwen3.5-9B",
                (QuantOption("fp16", 229152),),
                notes="FP16 on 32GB; kv_pool ~229K on RTX 5090 (6GB headroom)",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/Qwen3.5-9B",
                (QuantOption("fp16", 262144),),
                notes="FP16 full 256K context; kv_pool ~600K+ on A6000/L40S",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3.5-9B",
                (QuantOption("fp16", 262144),),
                notes="FP16 full 256K context",
            ),
        ),
        total_params_b=9.0, active_params_b=9.0,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention (24 GDN + 8 std attn); "
              "successor to Qwen3-8B with much more efficient KV",
    ),

    ModelEntry(
        id="qwen3.5-4b",
        name="Qwen3.5-4B",
        base_model="Qwen/Qwen3.5-4B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_16,
                "Qwen/Qwen3.5-4B",
                (QuantOption("fp16", 131072),),
                label="Qwen3.5-4B (Dense, Multimodal)",
                notes="FP16 ~10 GB weights (incl. vision encoder); GDN → low KV usage",
            ),
            TierConfig(
                VRAMTier.GB_24,
                "Qwen/Qwen3.5-4B",
                (QuantOption("fp16", 262144),),
                notes="FP16 — full native 256K context; plenty of VRAM headroom",
            ),
        ),
        total_params_b=4.0, active_params_b=4.0,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention (24 GDN + 8 std attn)",
    ),

    ModelEntry(
        id="qwen3.5-2b",
        name="Qwen3.5-2B",
        base_model="Qwen/Qwen3.5-2B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_12,
                "Qwen/Qwen3.5-2B",
                (QuantOption("fp16", 262144),),
                label="Qwen3.5-2B (Dense, Multimodal)",
                notes="FP16 ~4 GB weights; fits easily on 12 GB with full context",
            ),
        ),
        total_params_b=2.0, active_params_b=2.0,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        training=TrainingConfig(default_steps=30, default_rounds=2,
                                default_quant="fp16", size_b=2.0),
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention (18 GDN + 6 std attn); "
              "suitable for training demos and edge deployment",
    ),

    ModelEntry(
        id="qwen3.5-0.8b",
        name="Qwen3.5-0.8B",
        base_model="Qwen/Qwen3.5-0.8B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_12,
                "Qwen/Qwen3.5-0.8B",
                (QuantOption("fp16", 262144),),
                label="Qwen3.5-0.8B (Dense, Multimodal)",
                notes="FP16 ~1.8 GB weights; fits any GPU with full context",
            ),
        ),
        total_params_b=0.8, active_params_b=0.8,
        native_context_len=262144,
        generation_quality=1.05,  # GDN hybrid attn punches above weight class
        training=TrainingConfig(default_steps=50, default_rounds=3,
                                default_quant="fp16", size_b=0.8),
        verified_inference=True,
        family="qwen3.5", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention (18 GDN + 6 std attn); "
              "smallest Qwen3.5; training demos and edge deployment",
    ),

    # ================================================================
    # Gemma 4 — Google's best open models (Apache 2.0)
    # ================================================================
    #
    # Four sizes: E2B (2B), E4B (4B), 26B A4B MoE, 31B Dense.
    # Hybrid attention: sliding window (1024 tokens) + global attention.
    # Multimodal (text + image + video).  Released April 2, 2026.
    # Requires vLLM >= 0.19.0.
    #

    ModelEntry(
        id="gemma-4-26b-a4b",
        name="Gemma-4-26B-A4B",
        base_model="google/gemma-4-26B-A4B-it",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_24,
                "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                (QuantOption("int4", 44032),),
                label="Gemma-4-26B-A4B AWQ (MoE, 128e, Multimodal)",
                notes="Community AWQ INT4 ~13 GB weights + vision encoder; "
                      "~44K context on 24 GB (compact MoE)",
            ),
            TierConfig(
                VRAMTier.GB_32,
                "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                (QuantOption("int4", 89088),),
                notes="AWQ INT4 — ~89K context on 32 GB",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                (QuantOption("int4", 178176),),
                notes="AWQ INT4 — ~178K context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "google/gemma-4-26B-A4B-it",
                (QuantOption("fp8", 262144),),
                notes="FP8 runtime — full native 256K context",
            ),
        ),
        total_params_b=26.0, active_params_b=3.8, num_experts=128,
        native_context_len=262144,
        generation_quality=1.0,  # MMLU-Pro 82.6, GPQA 82.3 — solid but below Qwen3.6 MoE
        moe_dense_equivalent=20.0,  # MMLU-Pro 82.6 between 14B-27B dense; AIME 88.3 strong
        family="gemma4", provider="Google",
        notes="Multimodal (text + image + video); hybrid sliding-window + global attention; "
              "128 experts (8 active + 1 shared); Apache 2.0; requires vLLM >= 0.19.0",
    ),

    ModelEntry(
        id="gemma-4-31b",
        name="Gemma-4-31B",
        base_model="google/gemma-4-31B-it",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_32,
                "QuantTrio/gemma-4-31B-it-AWQ",
                (QuantOption("int4", 26624),),
                label="Gemma-4-31B AWQ (Dense, General/Coding/Multimodal)",
                notes="Community AWQ INT4 ~16 GB weights + ~1 GB vision encoder; "
                      "~27K context on 32 GB",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "QuantTrio/gemma-4-31B-it-AWQ",
                (QuantOption("int4", 59392),),
                notes="AWQ INT4 — ~59K context on 48 GB",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "google/gemma-4-31B-it",
                (QuantOption("fp8", 91136), QuantOption("int4", 126976)),
                notes="FP8 for quality (~91K ctx), INT4 for max context (~127K)",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "google/gemma-4-31B-it",
                (QuantOption("fp8", 218112), QuantOption("bf16", 146432)),
                notes="FP8 ~218K ctx; BF16 max quality ~146K ctx",
            ),
        ),
        total_params_b=31.0, active_params_b=31.0,
        native_context_len=262144,
        generation_quality=1.05,  # MMLU-Pro 85.2, GPQA 84.3, Codeforces 2150 — strong
        family="gemma4", provider="Google",
        notes="Multimodal (text + image + video); hybrid sliding-window + global attention; "
              "Apache 2.0; Codeforces ELO 2150; requires vLLM >= 0.19.0",
    ),

    # ================================================================
    # Qwen3.6 — Improved MoE/Dense with GDN hybrid attention
    # ================================================================
    #
    # Successor to Qwen3.5.  Same GDN hybrid attention architecture
    # (3:1 linear-to-standard ratio), but with improved training —
    # benchmarks ~3-5% above Qwen3.5 counterparts across the board.
    # Multimodal (text + image + video).  Requires vLLM >= 0.19.0.
    #

    ModelEntry(
        id="qwen3.6-35b-a3b",
        name="Qwen3.6-35B-A3B",
        base_model="Qwen/Qwen3.6-35B-A3B",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.MULTIMODAL),
        tier_configs=(
            # NOTE: ~21 GiB GPU weights at INT4 — does NOT fit on 24 GB (same as Qwen3.5).
            TierConfig(
                VRAMTier.GB_32,
                "QuantTrio/Qwen3.6-35B-A3B-AWQ",
                (QuantOption("int4", 40960),),
                label="Qwen3.6-35B-A3B AWQ (MoE, 256e, Multimodal)",
                notes="Community AWQ INT4 ~21 GB weights; tight on 32 GB; GDN hybrid attn",
            ),
            TierConfig(
                VRAMTier.GB_48,
                "QuantTrio/Qwen3.6-35B-A3B-AWQ",
                (QuantOption("int4", 262144),),
                notes="AWQ INT4 — full native 256K context on 48 GB (compact MoE)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3.6-35B-A3B-FP8",
                (QuantOption("fp8", 262144),),
                notes="Official pre-quantized FP8 — full native 256K context",
            ),
        ),
        total_params_b=35.0, active_params_b=3.0, num_experts=256,
        native_context_len=262144,
        generation_quality=1.10,  # MMLU-Pro 85.2, GPQA 86.0, SWE-Bench 73.4 — improved over 3.5
        moe_dense_equivalent=24.0,  # Benchmarks below Qwen3.6-27B dense; ~24B dense equiv
        family="qwen3.6", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention; "
              "256 experts (8 routed + 1 shared); successor to Qwen3.5-35B-A3B; "
              "requires vLLM >= 0.19.0",
    ),

    ModelEntry(
        id="qwen3.6-27b",
        name="Qwen3.6-27B",
        base_model="Qwen/Qwen3.6-27B",
        architecture="dense",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.MULTIMODAL),
        tier_configs=(
            # NOTE: Same GDN OOM as Qwen3.5-27B — does NOT fit on 24 GB or 32 GB.
            # GDN solve_tril Triton kernel OOMs even at gpu_mem=0.95.  Min tier: GB_48.
            TierConfig(
                VRAMTier.GB_48,
                "Qwen/Qwen3.6-27B",
                (QuantOption("nf4", 85000),),
                notes="NF4 runtime quant via bitsandbytes — GDN kernel requires ≥48 GB; "
                      "GPTQ-Int4 checkpoint pending (model released 2026-04-22)",
            ),
            TierConfig(
                VRAMTier.GB_80,
                "Qwen/Qwen3.6-27B-FP8",
                (QuantOption("fp8", 262144),),
                notes="Official pre-quantized FP8 — full native 256K context",
            ),
            TierConfig(
                VRAMTier.GB_141,
                "Qwen/Qwen3.6-27B",
                (QuantOption("bf16", 262144),),
                notes="BF16 max quality — full native 256K context",
            ),
        ),
        total_params_b=27.0, active_params_b=27.0,
        native_context_len=262144,
        generation_quality=1.10,  # MMLU-Pro 86.2, GPQA 87.8, SWE-Bench 77.2 — surpasses Qwen3.5-397B
        family="qwen3.6", provider="Qwen",
        notes="Multimodal (text + image + video); GDN hybrid attention; "
              "flagship-level coding (SWE-Bench 77.2); successor to Qwen3.5-27B; "
              "requires vLLM >= 0.19.0",
    ),

    # ================================================================
    # Multi-GPU models — NOT YET SUPPORTED by the protocol
    # ================================================================

    ModelEntry(
        id="kimi-k2-instruct",
        name="Kimi K2-Instruct",
        base_model="moonshotai/Kimi-K2-Instruct",
        architecture="moe",
        categories=(ModelCategory.AGENT_SWE,),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "moonshotai/Kimi-K2-Instruct",
                (QuantOption("int4", 4096),),
                notes="~489 GB at INT4; 8x H100 80GB or 6x B200",
            ),
        ),
        total_params_b=1000.0, active_params_b=32.0, num_experts=384,
        native_context_len=131072,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Top-tier agentic, ~70B dense equiv
        multi_gpu=True,
        family="kimi", provider="Moonshot AI",
        notes="NOT YET SUPPORTED — multi-GPU required (8x H100 min)",
    ),

    ModelEntry(
        id="kimi-k2.5",
        name="Kimi K2.5",
        base_model="moonshotai/Kimi-K2.5",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "moonshotai/Kimi-K2.5",
                (QuantOption("int4", 4096),),
                notes="~600 GB at INT4; 4x H200 or 8x H100",
            ),
        ),
        total_params_b=1000.0, active_params_b=32.0, num_experts=384,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Top-tier multimodal, ~70B dense equiv
        multi_gpu=True,
        family="kimi", provider="Moonshot AI",
        notes="NOT YET SUPPORTED — multi-GPU required; native vision + thinking",
    ),

    ModelEntry(
        id="deepseek-v3.2",
        name="DeepSeek-V3.2",
        base_model="deepseek-ai/DeepSeek-V3",
        architecture="moe",
        categories=(ModelCategory.GENERAL,),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "deepseek-ai/DeepSeek-V3",
                (QuantOption("fp8", 4096), QuantOption("int4", 4096)),
                notes="~700 GB FP8, ~386 GB INT4; 8x H100 (FP8) or 5x A100 (INT4)",
            ),
        ),
        total_params_b=685.0, active_params_b=37.0,
        native_context_len=163840,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Top-tier general, ~70B dense equiv
        multi_gpu=True,
        family="deepseek-v3", provider="DeepSeek",
        notes="NOT YET SUPPORTED — multi-GPU required (8x H100 min for FP8)",
    ),

    ModelEntry(
        id="deepseek-r1-671b",
        name="DeepSeek-R1 (671B full)",
        base_model="deepseek-ai/DeepSeek-R1",
        architecture="moe",
        categories=(ModelCategory.REASONING,),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "deepseek-ai/DeepSeek-R1",
                (QuantOption("fp8", 4096), QuantOption("int4", 4096)),
                notes="~700 GB FP8, ~386 GB INT4; 8x H100",
            ),
        ),
        total_params_b=671.0, active_params_b=37.0,
        native_context_len=163840,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Best open reasoning, ~70B dense equiv
        multi_gpu=True,
        family="deepseek-r1", provider="DeepSeek",
        notes="NOT YET SUPPORTED — multi-GPU required; best open reasoning model (full)",
    ),

    ModelEntry(
        id="mimo-v2-flash",
        name="MiMo-V2-Flash",
        base_model="XiaomiMiMo/MiMo-V2-Flash",
        architecture="moe",
        categories=(ModelCategory.CODING, ModelCategory.REASONING),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_288,
                "cyankiwi/MiMo-V2-Flash-AWQ-4bit",
                (QuantOption("int4", 65536),),
                label="MiMo-V2-Flash AWQ (MoE, Reasoning/Coding)",
                notes="AWQ INT4 ~170 GB; ~89 GB KV on B300; MLA keeps KV compact; "
                      "conservative ctx estimate",
            ),
            TierConfig(
                VRAMTier.MULTI_GPU,
                "XiaomiMiMo/MiMo-V2-Flash",
                (QuantOption("fp8", 4096), QuantOption("int4", 4096)),
                notes="FP8 ~309 GB (4x H100); INT4 ~170 GB (2x H100)",
            ),
        ),
        total_params_b=309.0, active_params_b=15.0,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=30.0,  # Fast reasoning MoE, ~30B dense equiv
        family="mimo", provider="Xiaomi",
        notes="Ultra-fast reasoning MoE; MLA attention (compact KV cache); "
              "B300 single-GPU viable at INT4; SGLang recommended by Xiaomi; "
              "multi-GPU for FP8 (4x H100)",
    ),

    ModelEntry(
        id="glm-5",
        name="GLM-5",
        base_model="zai-org/GLM-5",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.CODING, ModelCategory.REASONING),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "zai-org/GLM-5",
                (QuantOption("fp8", 4096), QuantOption("int4", 4096)),
                notes="~744B MoE; FP8 ~744 GB, INT4 ~372 GB; 4x B200 or 8x H100",
            ),
        ),
        total_params_b=744.0, active_params_b=40.0,
        native_context_len=202752,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # #1 open-source Feb 2026, rivals GPT-5.2
        multi_gpu=True,
        family="glm", provider="Zhipu AI / Z.ai",
        notes="NOT YET SUPPORTED — multi-GPU required; MIT license; #1 open-source (Feb 2026)",
    ),

    ModelEntry(
        id="mistral-large-3-675b",
        name="Mistral-Large-3-675B",
        base_model="mistralai/Mistral-Large-3-675B-Instruct-2512",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "mistralai/Mistral-Large-3-675B-Instruct-2512",
                (QuantOption("fp8", 4096), QuantOption("nvfp4", 4096)),
                notes="~675B MoE (41B active); FP8 ~675 GB; Apache 2.0",
            ),
        ),
        total_params_b=675.0, active_params_b=41.0,
        native_context_len=262144,
        generation_quality=1.0,
        moe_dense_equivalent=70.0,  # Mistral flagship, competitive with frontier models
        multi_gpu=True,
        family="mistral-large", provider="Mistral",
        notes="NOT YET SUPPORTED — multi-GPU required; Apache 2.0; multimodal (vision)",
    ),

    ModelEntry(
        id="qwen3.5-397b-a17b",
        name="Qwen3.5-397B-A17B",
        base_model="Qwen/Qwen3.5-397B-A17B",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.MULTIMODAL),
        tier_configs=(
            TierConfig(
                VRAMTier.MULTI_GPU,
                "Qwen/Qwen3.5-397B-A17B-FP8",
                (QuantOption("fp8", 4096), QuantOption("nvfp4", 4096)),
                notes="FP8 ~397 GB (5x H100 / 3x B200); NVFP4 ~199 GB (2x B200 / 1x B300)",
            ),
        ),
        total_params_b=397.0, active_params_b=17.0, num_experts=512,
        native_context_len=1048576,
        generation_quality=1.0,
        moe_dense_equivalent=100.0,  # Flagship, frontier-competitive
        multi_gpu=True,
        family="qwen3.5", provider="Qwen",
        notes="NOT YET SUPPORTED — multi-GPU required; multimodal (text + image + video); "
              "GDN hybrid attention; 512 experts (10 routed + 1 shared); 1M native context",
    ),

    ModelEntry(
        id="minimax-m2.5",
        name="MiniMax-M2.5",
        base_model="MiniMaxAI/MiniMax-M2.5",
        architecture="moe",
        categories=(ModelCategory.GENERAL, ModelCategory.REASONING, ModelCategory.CODING, ModelCategory.AGENT_SWE),
        tier_configs=(
            TierConfig(
                VRAMTier.GB_192,
                "lukealonso/MiniMax-M2.5-NVFP4",
                (QuantOption("nvfp4", 196608),),  # measured via vLLM auto-fit on B200 @ 0.90 util (full native context)
                label="MiniMax-M2.5 NVFP4 (MoE, Coding/Agent)",
                notes="NVFP4 experts + BF16 attn ~120 GB; full 196K context on B200",
            ),
            TierConfig(
                VRAMTier.GB_288,
                "MiniMaxAI/MiniMax-M2.5",
                (QuantOption("fp8", 60416), QuantOption("nvfp4", 196608)),
                notes="FP8 native ~229 GB → ~60K ctx; NVFP4 ~120 GB → full 196K ctx",
            ),
        ),
        total_params_b=229.0, active_params_b=10.0, num_experts=256,
        native_context_len=196608,
        generation_quality=1.0,
        moe_dense_equivalent=90.0,  # GPQA 85.2%, AIME 86.3%, SWE-Bench 80.2%; no MMLU-Pro published
        family="minimax-m2", provider="MiniMax",
        notes="FP8 native or NVFP4 (experts only); NVFP4 requires vLLM 0.15.1+ "
              "with --quantization modelopt_fp4",
    ),
)


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

MODELS_BY_ID: dict[str, ModelEntry] = {m.id: m for m in ALL_MODELS}

# Reverse index: HF checkpoint name -> ModelEntry (for on-chain model_id lookups)
MODELS_BY_CHECKPOINT: dict[str, ModelEntry] = {}
for _m in ALL_MODELS:
    for _tc in _m.tier_configs:
        MODELS_BY_CHECKPOINT.setdefault(_tc.checkpoint, _m)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_model(model_id: str) -> ModelEntry | None:
    """Look up a model by its stable ID or HF checkpoint name."""
    return MODELS_BY_ID.get(model_id) or MODELS_BY_CHECKPOINT.get(model_id)


def get_models_for_tier(tier: VRAMTier) -> list[ModelEntry]:
    """Return all models that have a TierConfig for the given VRAM tier."""
    return [m for m in ALL_MODELS
            if any(tc.tier == tier for tc in m.tier_configs)]


def get_models_by_category(category: ModelCategory) -> list[ModelEntry]:
    """Return all models matching a category."""
    return [m for m in ALL_MODELS if category in m.categories]


def get_single_gpu_models() -> list[ModelEntry]:
    """Return all single-GPU models (supported by the protocol)."""
    return [m for m in ALL_MODELS if not m.multi_gpu]


def get_multi_gpu_models() -> list[ModelEntry]:
    """Return all multi-GPU models (NOT YET SUPPORTED by the protocol)."""
    return [m for m in ALL_MODELS if m.multi_gpu]


def get_training_models() -> list[ModelEntry]:
    """Return all models that support training."""
    return [m for m in ALL_MODELS if m.training is not None]


def resolve_model_for_tier(
    model_id: str,
    tier: VRAMTier,
    *,
    quant: str | None = None,
    checkpoint: str | None = None,
) -> TierMatch:
    """Resolve a model ID to the best config for a VRAM tier.

    Picks the highest TierConfig whose tier is ``<= tier``.
    When multiple configs exist at the same tier (e.g. AWQ and FP16),
    ``quant`` or ``checkpoint`` narrow the match.

    Raises ``KeyError`` if the model ID is unknown.
    Raises ``ValueError`` if the model has no config that fits the tier.
    """
    model = MODELS_BY_ID.get(model_id)
    if model is None:
        available = ", ".join(sorted(MODELS_BY_ID))
        raise KeyError(f"Unknown model ID {model_id!r}. Available: {available}")

    # Collect all configs that fit this tier (tier <= requested).
    candidates: list[TierConfig] = []
    best_tier_val = -1
    for tc in model.tier_configs:
        if tc.tier <= tier:
            if tc.tier.value > best_tier_val:
                best_tier_val = tc.tier.value
                candidates = [tc]
            elif tc.tier.value == best_tier_val:
                candidates.append(tc)

    if not candidates:
        min_tier = min(tc.tier for tc in model.tier_configs)
        raise ValueError(
            f"Model {model_id!r} requires at least {min_tier.value} GB VRAM "
            f"(tier {min_tier.name}), but only {tier.value} GB available."
        )

    # Narrow by checkpoint or quant if requested.
    if checkpoint:
        filtered = [c for c in candidates if c.checkpoint.lower() == checkpoint.lower()]
        if filtered:
            candidates = filtered
    if quant and len(candidates) > 1:
        filtered = [c for c in candidates if any(qo.quant == quant for qo in c.quant_configs)]
        if filtered:
            candidates = filtered

    best = candidates[0]
    return TierMatch(model=model, config=best, native=(best.tier == tier))


def get_models_up_to_tier(
    tier: VRAMTier,
    *,
    verified_only: bool = False,
) -> list[TierMatch]:
    """Return all single-GPU models that can run on the given VRAM tier.

    For each model, picks the **best** available TierConfig — the one with
    the highest tier that is still ``<= tier``.  The result is wrapped in a
    :class:`TierMatch` with a ``native`` flag:

    - ``native=True``:  the model has a TierConfig specifically targeting
      the requested tier — optimised and (if verified) tested.
    - ``native=False``: the model uses a config from a **lower** tier.
      It will run fine (the GPU has more than enough VRAM) but the
      checkpoint / quant / max_model_len aren't tier-optimised.

    Multi-GPU models are always excluded.
    """
    results: list[TierMatch] = []
    for model in ALL_MODELS:
        if model.multi_gpu:
            continue
        if verified_only and not model.verified_inference:
            continue
        # Collect all configs that fit: highest tier level that's <= requested,
        # but include ALL configs at that tier level (e.g. both AWQ and fp16).
        best_tier_val = -1
        fitting: list[TierConfig] = []
        for tc in model.tier_configs:
            if tc.tier <= tier:
                if tc.tier.value > best_tier_val:
                    best_tier_val = tc.tier.value
                    fitting = [tc]
                elif tc.tier.value == best_tier_val:
                    fitting.append(tc)
        for tc in fitting:
            results.append(TierMatch(
                model=model,
                config=tc,
                native=(tc.tier == tier),
            ))
    return results


# ---------------------------------------------------------------------------
# VRAM estimation helpers
# ---------------------------------------------------------------------------

# Bytes per parameter for each quantization mode
_QUANT_BYTES: dict[str, float] = {
    "bf16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "int8": 1.0,
    "int4": 0.5, "nf4": 0.5, "nvfp4": 0.5, "mxfp4": 0.53,
}

# Quality penalty for quantization (1.0 = lossless, lower = more lossy)
QUANT_QUALITY: dict[str, float] = {
    "bf16": 1.0, "fp16": 1.0,
    "fp8": 0.98, "int8": 0.95,
    "int4": 0.90, "nf4": 0.90, "nvfp4": 0.92, "mxfp4": 0.91,
}


def estimate_weight_gb(total_params_b: float, quant: str) -> float:
    """Estimate GPU memory for model weights in GB.

    This is a rough estimate: ``params_b * bytes_per_param``.
    Real usage is ~10-20% higher due to buffers, norms, and embeddings
    that may stay in higher precision.
    """
    bpp = _QUANT_BYTES.get(quant, 2.0)
    return total_params_b * bpp


def estimate_effective_context(
    total_params_b: float,
    active_params_b: float,
    quant: str,
    vram_gb: int,
    native_context_len: int,
    *,
    gpu_memory_utilization: float = 0.90,
    overhead_gb: float = 1.5,
    is_moe: bool = False,
) -> int:
    """Estimate the maximum context length that fits in VRAM.

    Uses a simplified model::

        usable_vram = vram * utilization - weights - overhead
        kv_per_token ≈ kv_params * 15_000 bytes
        effective_context = usable_vram / kv_per_token

    The KV-cache constant (~15 KB per billion params) is calibrated for
    modern GQA architectures in the 7B-30B range.

    For **dense** models, ``kv_params = total_params_b`` (KV cache scales
    with the full model depth).

    For **MoE** models, ``kv_params = active_params_b * 3``.  The KV cache
    only depends on the shared attention layers, not the expert FFN layers.
    Attention is typically 30-50% of active params in MoE architectures,
    so ``active * 3`` approximates the "attention-equivalent" param count.
    (e.g. Qwen3-30B-A3B: 3B active * 3 = 9B → 135 KB/token est,
    real ~96 KB/token — slightly conservative.)

    Capped at the model's ``native_context_len``.
    """
    weight_gb = estimate_weight_gb(total_params_b, quant)
    usable_gb = vram_gb * gpu_memory_utilization - weight_gb - overhead_gb
    if usable_gb <= 0:
        return 0

    # KV cache bytes per token: ~15,000 bytes per billion "KV-relevant" params.
    if is_moe:
        # MoE: attention layers are a fraction of active params, not total.
        # Using active_params * 3 as proxy for attention-equivalent depth.
        kv_params = active_params_b * 3
    else:
        # Dense: all params contribute to model depth → KV scales with total.
        kv_params = total_params_b
    kv_bytes_per_token = kv_params * 15_000
    usable_bytes = usable_gb * (1024 ** 3)
    max_tokens = int(usable_bytes / kv_bytes_per_token)

    # Round down to nearest 1024 for cleanliness
    max_tokens = (max_tokens // 1024) * 1024
    return min(max(max_tokens, 0), native_context_len)


@dataclass(frozen=True)
class ModelRecommendation:
    """A scored recommendation for a miner on a specific VRAM tier.

    The ``utility`` score is a rough heuristic combining model quality,
    effective context, quantization quality, and VRAM utilization.
    Higher is better.  **Not a final scoring formula** — intended as a
    guide for miners choosing which model to serve.
    """
    model: ModelEntry
    config: TierConfig
    native: bool
    quant: str                         # Best quant mode for this config
    est_weight_gb: float               # Estimated weight memory
    est_context: int                   # Estimated effective context
    context_utilization: float         # est_context / native_context_len
    vram_utilization: float            # est_weight_gb / tier_vram
    utility: float                     # Composite score (higher = better)


def recommend_models(
    tier: VRAMTier,
    *,
    verified_only: bool = False,
    category: ModelCategory | None = None,
    demand_scores: dict[str, int] | None = None,
    demand_bonus_max: float = 0.20,
) -> list[ModelRecommendation]:
    """Rank all models for a VRAM tier by estimated miner utility.

    For each model, estimates:
    - Weight memory and effective context for the best quant
    - A composite utility score::

        log2(quality_params)^1.3 × log2(context/1K) × quant_quality × generation_quality

    The ``^1.3`` exponent on params gives model quality (size) a moderate
    edge over context — enough to prefer bigger models at similar context,
    but not so much that a large older model dominates a smaller newer one
    with vastly more context and a better architecture.

    ``quality_params`` is:
    - For dense models: ``active_params_b`` (same as total).
    - For MoE models: ``moe_dense_equivalent`` — the dense model size that
      the MoE matches on benchmarks.  Without this, MoE models are unfairly
      penalised (e.g. Qwen3-30B-A3B has 3B active but benchmarks like a 14B
      dense model).

    Context uses ``log2`` which naturally provides diminishing returns —
    doubling the context only adds +1 to the score.
    The ``generation_quality`` multiplier penalises older/superseded
    architectures (e.g. Qwen 2.5 vs Qwen 3).

    Args:
        tier: Target VRAM tier.
        verified_only: Only include models with verified inference.
        category: Filter to a specific category (e.g. ``ModelCategory.CODING``).
            ``None`` returns all categories.
        demand_scores: Per-model demand scores in basis points (0-10000),
            as posted on-chain by validators.  ``None`` disables demand bonus.
        demand_bonus_max: Maximum demand bonus fraction (default 0.20 = 20%).

    Returns recommendations sorted by utility (descending).
    """
    import math

    recs: list[ModelRecommendation] = []

    for model in ALL_MODELS:
        if model.multi_gpu:
            continue
        if verified_only and not model.verified_inference:
            continue
        if category is not None and category not in model.categories:
            continue

        is_moe = model.architecture == "moe"
        # For MoE models, use the benchmark-calibrated dense equivalent
        # instead of raw active_params_b, which unfairly penalises MoE.
        quality_params = (
            model.moe_dense_equivalent if model.moe_dense_equivalent > 0
            else model.active_params_b
        )
        # Exponent >1 on params makes model quality dominate over context
        # in the ranking — users prefer smarter (larger) models even if
        # they have less context headroom.
        quality = math.log2(max(quality_params, 1.0)) ** 1.3

        # Try ALL tier configs that fit (<= requested tier), and all quant
        # modes within each.  Keep the best per (model, quant) — higher tiers
        # supersede lower ones for the same quant, but different quants on
        # the same tier both appear (e.g. AWQ int4 and fp16 on GB_32).
        best_by_quant: dict[str, ModelRecommendation] = {}

        for tc in model.tier_configs:
            if tc.tier > tier:
                continue  # Config targets a higher tier
            native = (tc.tier == tier)
            for qo in tc.quant_configs:
                quant = qo.quant
                est_weight = estimate_weight_gb(model.total_params_b, quant)

                # Context: prefer registered max_model_len (measured/calibrated)
                # over formula estimate (inaccurate for GQA/GDN models).
                formula_ctx = estimate_effective_context(
                    model.total_params_b, model.active_params_b, quant,
                    tier.value, model.native_context_len,
                    is_moe=is_moe,
                )
                if qo.max_model_len > 0:
                    # Use measured value from registry (native or inherited).
                    # Don't inflate inherited values with formula estimates —
                    # measured values are more reliable, and higher-tier configs
                    # (with larger measured values) will win via best_by_quant.
                    est_ctx = qo.max_model_len
                else:
                    est_ctx = formula_ctx
                if est_ctx == 0:
                    continue  # Doesn't fit at this quant

                ctx_util = est_ctx / model.native_context_len if model.native_context_len > 0 else 0
                vram_util = est_weight / tier.value if tier.value > 0 else 0
                quant_q = QUANT_QUALITY.get(quant, 0.8)

                # log2 naturally provides diminishing returns —
                # doubling context only adds +1 to the score.
                context_value = math.log2(max(est_ctx / 1024, 1))
                utility = quality * context_value * quant_q * model.generation_quality

                # Demand bonus: models with organic user demand get a bonus
                if demand_scores is not None:
                    bps = demand_scores.get(model.id, 0)
                    clamped = max(0, min(bps, 10000))
                    utility *= 1.0 + (clamped / 10000) * demand_bonus_max

                rec = ModelRecommendation(
                    model=model, config=tc, native=native,
                    quant=quant, est_weight_gb=est_weight,
                    est_context=est_ctx, context_utilization=ctx_util,
                    vram_utilization=vram_util, utility=utility,
                )
                prev = best_by_quant.get(quant)
                if prev is None or utility > prev.utility or (
                    utility == prev.utility and tc.tier > prev.config.tier
                ):
                    best_by_quant[quant] = rec

        recs.extend(best_by_quant.values())

    recs.sort(key=lambda r: r.utility, reverse=True)
    return recs


# ---------------------------------------------------------------------------
# Backward-compatibility bridges
# ---------------------------------------------------------------------------

def get_inference_presets(
    tier: VRAMTier = VRAMTier.GB_24,
) -> dict[str, tuple[str, tuple[QuantOption, ...]]]:
    """Return presets for ``streamlit_vllm_chat.py``.

    Returns ``{label: (checkpoint, quant_configs)}``.
    Only includes models with ``verified_inference=True``.
    """
    result: dict[str, tuple[str, tuple[QuantOption, ...]]] = {}
    for model in ALL_MODELS:
        if not model.verified_inference:
            continue
        for tc in model.tier_configs:
            if tc.tier == tier:
                label = tc.label or model.name
                result[label] = (tc.checkpoint, tc.quant_configs)
    return result


def get_training_presets() -> dict[str, dict]:
    """Return presets in the old ``streamlit_training.py`` format.

    Returns ``{display_name: {name, default_steps, default_rounds, size_b, [quant]}}``.
    """
    result: dict[str, dict] = {}
    for model in ALL_MODELS:
        if model.training is None:
            continue
        tc = model.tier_configs[0] if model.tier_configs else None
        if tc is None:
            continue
        preset: dict = {
            "name": tc.checkpoint,
            "default_steps": model.training.default_steps,
            "default_rounds": model.training.default_rounds,
            "size_b": model.training.size_b,
        }
        if model.training.default_quant not in ("fp16", ""):
            preset["quant"] = model.training.default_quant
        result[model.name] = preset
    return result


def get_webapp_presets(
    tier: VRAMTier = VRAMTier.GB_24,
) -> list[ModelPreset]:
    """Return presets as webapp ``ModelPreset`` dataclass instances.

    Only includes models with ``verified_inference=True``.
    """
    result: list[ModelPreset] = []
    for model in ALL_MODELS:
        if not model.verified_inference:
            continue
        for tc in model.tier_configs:
            if tc.tier == tier:
                cat = model.categories[0].value if model.categories else "general"
                result.append(ModelPreset(
                    id=model.id,
                    label=tc.label or model.name,
                    model_name=tc.checkpoint,
                    quant_options=tc.quant_configs,
                    category=cat,
                ))
    return result


# Module-level webapp indexes (computed once at import time)
MODEL_PRESETS: list[ModelPreset] = get_webapp_presets()
PRESETS_BY_ID: dict[str, ModelPreset] = {p.id: p for p in MODEL_PRESETS}


def get_presets_grouped(
    tier: VRAMTier = VRAMTier.GB_24,
) -> dict[str, list[ModelPreset]]:
    """Return webapp-format presets grouped by category."""
    presets = get_webapp_presets(tier)
    groups: dict[str, list[ModelPreset]] = {}
    for p in presets:
        groups.setdefault(p.category, []).append(p)
    return groups
