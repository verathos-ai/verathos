"""Centralized logging setup for Verathos neuron processes.

Provides:
- ``setup_neuron_logging(args)`` — configure bt.logging from CLI flags and
  register all child loggers as primary loggers so they aren't silenced.
- ``print_banner(role, ...)`` — branded startup banner via bt.logging.
"""
from __future__ import annotations

import bittensor as bt

from neurons.version import (
    version_str as __version__,
    miner_version_str,
    validator_version_str,
)

# ---------------------------------------------------------------------------
# All stdlib loggers that may be created in-process by neuron code or by
# verallm modules imported directly into the neuron process (chain, validator,
# moe).  ``register_primary_logger`` requires exact names — no wildcards.
# ---------------------------------------------------------------------------
_PRIMARY_LOGGERS: list[str] = [
    # neurons/*
    "neurons",
    "neurons.auto_update",
    "neurons.auth",
    "neurons.base_deposits",
    "neurons.canary",
    "neurons.config",
    "neurons.credits",
    "neurons.db",
    "neurons.deposit_scanner",
    "neurons.discovery",
    "neurons.miner",
    "neurons.miner_pool",
    "neurons.model_resolve",
    "neurons.poi",
    "neurons.pricing",
    "neurons.proxy",
    "neurons.receipts",
    "neurons.request_signing",
    "neurons.scoring",
    "neurons.settlement",
    "neurons.shared_state",
    "neurons.user_manager",
    "neurons.validator",
    "neurons.validator_db",
    "neurons.x402_payment",
    # verallm modules imported in-process (not in the server subprocess)
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
    "verallm.validator.core",
    "verallm.moe.batch_hooks",
]


def setup_neuron_logging(args) -> None:
    """Configure bt.logging from parsed CLI ``args`` and register loggers.

    Call this once at the top of each neuron ``main()`` after parsing args.
    The argparse namespace should contain the flags added by
    ``bt.logging.add_args(parser)`` (``--logging.debug``, ``--logging.trace``,
    ``--logging.info``).
    """
    import logging as _logging
    import os as _os
    from logging.handlers import QueueHandler as _QH

    # Silence PyTorch's C++ glog noise (TorchDynamo, FakeTensor, inductor).
    # These use Google glog via stderr, not Python logging.
    _os.environ.setdefault("TORCH_LOGS", "-all")           # Python torch._logging
    _os.environ.setdefault("TORCHDYNAMO_LOG_LEVEL", "ERROR")
    _os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    _os.environ.setdefault("GLOG_minloglevel", "2")        # C++ glog: 0=INFO,1=WARN,2=ERROR

    # Determine level from CLI flags (trace > debug > info)
    trace = getattr(args, "logging.trace", False)
    debug = getattr(args, "logging.debug", False)

    if trace:
        bt.logging.enable_trace()
    elif debug:
        bt.logging.enable_debug()
    else:
        bt.logging.enable_info()

    # Prevent bt.logging from propagating to root logger.
    bt.logging._logger.propagate = False

    # Register all known child loggers so bt.logging doesn't silence them.
    # Clear any existing QueueHandlers first to prevent duplicate output
    # (bt.logging._initialize_bt_logger blindly adds a QueueHandler on
    # every register_primary_logger call).
    for name in _PRIMARY_LOGGERS:
        lg = _logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, _QH):
                lg.removeHandler(h)
        bt.logging.register_primary_logger(name)
        # Prevent propagation to root logger (which may have a default
        # stderr StreamHandler), so all output goes through bt.logging's
        # QueueListener → stdout StreamHandler only.
        lg.propagate = False

    # Route uvicorn loggers through bt.logging so they match our format.
    # Suppress noisy access logs (each HTTP request) — only show errors.
    for uv_name in ("uvicorn", "uvicorn.error"):
        lg = _logging.getLogger(uv_name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        bt.logging.register_primary_logger(uv_name)
        lg.propagate = False
    _logging.getLogger("uvicorn.access").setLevel(_logging.WARNING)
    _logging.getLogger("uvicorn.access").handlers.clear()
    _logging.getLogger("uvicorn.access").propagate = False


def _level_label() -> str:
    """Return current bt.logging level as a human-readable string."""
    level = bt.logging.get_level()
    _MAP = {5: "TRACE", 10: "DEBUG", 20: "INFO", 30: "WARNING"}
    return _MAP.get(level, str(level))


def print_banner(
    role: str,
    *,
    network: str = "",
    netuid: int | str = "",
    wallet: str = "",
    hotkey: str = "",
    evm: str = "",
    **extra: str,
) -> None:
    """Print a branded Verathos startup banner.

    Parameters
    ----------
    role:
        "Miner", "Validator", or "Proxy".
    network, netuid, wallet, hotkey, evm:
        Standard fields shown in every banner.
    **extra:
        Additional key-value pairs appended to the info section.
        Keys are title-cased for display (underscores → spaces).
    """
    width = 60
    bar = "─" * width
    # Max value length = width - 2 (padding) - 18 (label column)
    max_val = width - 20

    lines: list[str] = []
    lines.append(f"┌{bar}┐")
    lines.append(f"│  {'Verathos · Subnet 96':<{width - 2}}│")
    _role_ver = miner_version_str if role.lower() == "miner" else validator_version_str
    lines.append(f"│  {role + ' v' + _role_ver:<{width - 2}}│")
    lines.append(f"├{bar}┤")

    # Build info rows
    rows: list[tuple[str, str]] = []
    if network:
        net_label = f"{network} (netuid {netuid})" if netuid else network
        rows.append(("Network", net_label))
    if wallet:
        wallet_label = f"{wallet}/{hotkey}" if hotkey else wallet
        rows.append(("Wallet", wallet_label))
    if evm:
        # Truncate long addresses for readability
        evm_display = f"{evm[:6]}…{evm[-4:]}" if len(evm) > 14 else evm
        rows.append(("EVM", evm_display))

    for key, val in extra.items():
        label = key.replace("_", " ").title()
        rows.append((label, str(val)))

    rows.append(("Log level", _level_label()))

    for label, value in rows:
        # Truncate long values to fit the box
        if len(value) > max_val:
            value = value[:max_val - 1] + "…"
        row_text = f"  {label + ':':<18}{value}"
        lines.append(f"│{row_text:<{width}}│")

    lines.append(f"└{bar}┘")

    # Use bt.logging.info for each line so it flows through the standard
    # bt.logging pipeline (colors, file logging, etc.)
    for line in lines:
        bt.logging.info(line)
