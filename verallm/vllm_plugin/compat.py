"""
Version detection for vLLM CUDA graph support.

Checks whether the installed vLLM supports:
- CustomOp.register_oot() for out-of-tree layer replacement
- CompilationConfig.splitting_ops for piecewise CUDA graphs
"""

import logging

logger = logging.getLogger(__name__)


def has_customop_oot() -> bool:
    """Check if vLLM supports CustomOp.register_oot."""
    try:
        from vllm.model_executor.custom_op import CustomOp, op_registry_oot  # noqa: F401
        return hasattr(CustomOp, 'register_oot')
    except ImportError:
        return False


def has_splitting_ops() -> bool:
    """Check if CompilationConfig supports splitting_ops."""
    try:
        from vllm.config import CompilationConfig  # noqa: F401
        return hasattr(CompilationConfig, 'splitting_ops')
    except ImportError:
        try:
            from vllm.config.compilation import CompilationConfig  # noqa: F401
            return hasattr(CompilationConfig, 'splitting_ops')
        except ImportError:
            return False


def can_use_cuda_graphs() -> bool:
    """Check if vLLM has both CustomOp OOT and splitting_ops support.

    Both are required for Phase 4 (piecewise CUDA graphs with
    activation capture at split points).
    """
    oot = has_customop_oot()
    split = has_splitting_ops()
    if oot and split:
        logger.info("vLLM supports CUDA graph capture (CustomOp OOT + splitting_ops)")
        return True
    if not oot:
        logger.info("vLLM lacks CustomOp.register_oot — falling back to hooks")
    if not split:
        logger.info("vLLM lacks CompilationConfig.splitting_ops — falling back to hooks")
    return False
