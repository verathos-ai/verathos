"""
Verathos vLLM Plugin — activation capture for proof generation.

Entry point for vllm.general_plugins.  Must be called BEFORE LLM()
construction so the OOT class replacements are registered when vLLM
instantiates model layers.

Two modes of invocation:
1. Automatic: vLLM discovers the entry point via pyproject.toml
   [project.entry-points."vllm.general_plugins"]
2. Explicit: verallm.miner.vllm_backend calls register_verathos_plugin()
   before constructing the LLM instance.
"""

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_verathos_plugin() -> bool:
    """Register OOT CustomOp replacements for activation capture.

    Idempotent — safe to call multiple times.

    Returns True if registration succeeded, False if vLLM lacks support.
    """
    global _REGISTERED
    if _REGISTERED:
        return True

    from .compat import has_customop_oot

    if not has_customop_oot():
        logger.warning(
            "vLLM does not support CustomOp.register_oot. "
            "Falling back to enforce_eager=True with PyTorch hooks."
        )
        return False

    # Import triggers OOT registration via @CustomOp.register_oot decorator
    from .capture_fused_moe import _register
    _register()

    # Validate OOT registration keys up front so startup can safely fall back
    # to eager hooks if vLLM changes CustomOp lookup behavior again.
    try:
        from vllm.model_executor.custom_op import op_registry_oot
        required_keys = {"FusedMoE"}
        try:
            # Some MoE architectures (e.g., Qwen3) instantiate SharedFusedMoE.
            # If present in this vLLM build, we must register it too.
            from vllm.model_executor.layers.fused_moe.shared_fused_moe import (  # noqa: F401
                SharedFusedMoE,
            )
            required_keys.add("SharedFusedMoE")
        except Exception:
            pass

        missing = [k for k in sorted(required_keys) if k not in op_registry_oot]
        if missing:
            logger.warning(
                "CaptureFusedMoE OOT registration missing keys: %s. "
                "Falling back to enforce_eager=True with PyTorch hooks.",
                ",".join(missing),
            )
            return False
    except Exception:
        logger.exception(
            "Failed to validate CaptureFusedMoE OOT registration. "
            "Falling back to enforce_eager=True with PyTorch hooks."
        )
        return False

    # Register the verallm::capture custom op
    from . import ops  # noqa: F401

    _REGISTERED = True
    logger.info("Verathos vLLM plugin registered successfully")
    return True
