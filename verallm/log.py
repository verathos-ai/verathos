"""Centralized logging setup for the VeraLLM server subprocess.

Uses bt.logging (bittensor) for consistent format, colors, and log levels
across all Verathos processes (miner neuron, server, validator, proxy).

Provides:
- ``setup_server_logging(level)`` — configure bt.logging + register all
  verallm/zkllm loggers + suppress noisy third-party loggers.
- ``print_server_banner(...)`` — branded startup banner.
"""
from __future__ import annotations

import logging
from logging.handlers import QueueHandler

import bittensor as bt

from verallm import __version__

# All stdlib loggers used by verallm/zkllm library code.
# Must be registered as primary loggers so bt.logging doesn't silence them.
_PRIMARY_LOGGERS: list[str] = [
    "verallm",
    "verallm.api.server",
    "verallm.api.client",
    "verallm.api.receipt_store",
    "verallm.api.validator_auth",
    "verallm.chain",
    "verallm.chain.checkpoint",
    "verallm.chain.config",
    "verallm.chain.mock",
    "verallm.chain.miner_registry",
    "verallm.chain.model_registry",
    "verallm.chain.payment",
    "verallm.chain.provider",
    "verallm.chain.validator_registry",
    "verallm.chain.wallet",
    "verallm.miner.activation_tracker",
    "verallm.miner.admission",
    "verallm.miner.base",
    "verallm.miner.batch_engine",
    "verallm.miner.matmul",
    "verallm.miner.memory_budget",
    "verallm.miner.proof_pipeline",
    "verallm.miner.vllm_backend",
    "verallm.miner.vllm_utils",
    "verallm.moe.batch_hooks",
    "verallm.quantization.detection",
    "verallm.registry.cache",
    "verallm.registry.roots",
    "verallm.validator.core",
    "verallm.vllm_plugin",
    "verallm.vllm_plugin.capture_fused_moe",
    "verallm.vllm_plugin.capture_linear",
    "verallm.vllm_plugin.compat",
    "verallm.vllm_plugin.ops",
    "zkllm",
    "zkllm.cuda",
    "zkllm.crypto.sumcheck_fast",
    "zkllm.crypto.sumcheck",
    "zkllm.crypto.merkle",
]


def setup_server_logging(level: str = "info") -> None:
    """Configure bt.logging for the VeraLLM server process.

    Parameters
    ----------
    level:
        One of ``"debug"``, ``"info"``, ``"warning"`` (case-insensitive).
    """
    # Set bt.logging level
    if level.lower() == "trace":
        bt.logging.enable_trace()
    elif level.lower() == "debug":
        bt.logging.enable_debug()
    else:
        bt.logging.enable_info()

    # Prevent bt.logging from propagating to root (where vLLM adds handlers).
    # This is THE fix for duplicate log lines.
    bt.logging._logger.propagate = False

    # Register all verallm/zkllm loggers as primary loggers.
    # Clear existing QueueHandlers first to prevent duplicates.
    for name in _PRIMARY_LOGGERS:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, QueueHandler):
                lg.removeHandler(h)
        bt.logging.register_primary_logger(name)
        lg.propagate = False

    # Route uvicorn loggers through bt.logging, suppress access logs.
    for uv_name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(uv_name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        bt.logging.register_primary_logger(uv_name)
        lg.propagate = False
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "filelock", "urllib3", "urllib3.connectionpool", "requests"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.handlers.clear()
        lg.propagate = False

    # Suppress ALL vLLM logging — set the parent "vllm" logger to ERROR.
    # This covers all child loggers (vllm.v1.engine, vllm.model_executor, etc.)
    # even ones created later during warmup/inference.
    logging.getLogger("vllm").setLevel(logging.ERROR)
    logging.getLogger("vllm").handlers.clear()
    logging.getLogger("vllm").propagate = False
    # Nuke root handlers too — vLLM's dictConfig may have added some.
    logging.getLogger().handlers.clear()


def print_server_banner(
    *,
    model: str = "",
    quant: str = "",
    gpu_name: str = "",
    vram_gb: float = 0,
    sm: str = "",
    attention: str = "",
    tee: str = "",
    batch_mode: bool = True,
    k_layers: int | str = "",
    k_experts: int | str = "",
    port: int | str = "",
    **extra: str,
) -> None:
    """Print a branded startup banner for the VeraLLM miner server."""
    width = 60
    bar = "─" * width
    max_val = width - 20

    lines: list[str] = []
    lines.append(f"┌{bar}┐")
    lines.append(f"│  {'Verathos · Subnet 96':<{width - 2}}│")
    lines.append(f"│  {'VeraLLM Server v' + __version__:<{width - 2}}│")
    lines.append(f"├{bar}┤")

    rows: list[tuple[str, str]] = []
    if model:
        rows.append(("Model", model))
    if quant:
        rows.append(("Quantization", quant))
    if gpu_name:
        gpu_label = f"{gpu_name} ({vram_gb:.1f} GB" + (f", {sm}" if sm else "") + ")"
        rows.append(("GPU", gpu_label))
    if attention:
        rows.append(("Attention", attention))
    if tee:
        rows.append(("TEE", tee))
    rows.append(("Batch mode", "enabled" if batch_mode else "disabled"))
    if k_layers:
        proof_label = f"k_layers={k_layers}"
        if k_experts:
            proof_label += f", k_experts={k_experts}"
        rows.append(("Proof params", proof_label))
    if port:
        rows.append(("Port", str(port)))

    for key, val in extra.items():
        label = key.replace("_", " ").title()
        rows.append((label, str(val)))

    for label, value in rows:
        if len(value) > max_val:
            value = value[:max_val - 1] + "…"
        row_text = f"  {label + ':':<18}{value}"
        lines.append(f"│{row_text:<{width}}│")

    lines.append(f"└{bar}┘")

    for line in lines:
        bt.logging.info(line)
