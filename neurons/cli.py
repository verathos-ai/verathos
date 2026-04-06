#!/usr/bin/env python3
"""Verathos CLI — unified command for setup, status, and operations.

Registered as ``verathos`` via pyproject.toml console_scripts.

Usage::

    verathos setup              # interactive miner setup wizard
    verathos setup validator    # interactive validator setup wizard
    verathos status             # preflight readiness check
    verathos start              # pm2 start miner
    verathos stop               # pm2 stop miner
    verathos logs               # pm2 logs miner
    verathos models             # show model recommendations for your GPU
    verathos network            # network status dashboard
"""

from __future__ import annotations

import os
import subprocess
import sys


def _find_repo_root() -> str:
    """Find the Verathos repo root."""
    from neurons.wizard import _find_repo_root
    return str(_find_repo_root())


def cmd_setup(args: list[str]) -> None:
    """Launch the interactive setup wizard."""
    role = args[0] if args else "miner"
    if role not in ("miner", "validator"):
        print(f"Unknown role: {role}. Use 'miner' or 'validator'.")
        sys.exit(1)
    from neurons.wizard import run_wizard
    run_wizard(role)


def cmd_status(args: list[str]) -> None:
    """Run preflight readiness check."""
    from neurons.wizard import run_status
    run_status()


def cmd_start(args: list[str]) -> None:
    """Start a miner or validator via PM2."""
    name = args[0] if args else "miner"
    repo = _find_repo_root()
    config = os.path.join(repo, "ecosystem.config.js")
    if not os.path.exists(config):
        print(f"ecosystem.config.js not found in {repo}")
        print("Run 'verathos setup' first to generate it.")
        sys.exit(1)
    from pathlib import Path
    from neurons.wizard import _quick_start
    _quick_start(Path(repo), name)


def cmd_stop(args: list[str]) -> None:
    """Stop a miner or validator via PM2."""
    name = args[0] if args else "miner"
    subprocess.run(["pm2", "stop", name])


def cmd_logs(args: list[str]) -> None:
    """Tail PM2 logs."""
    name = args[0] if args else "miner"
    lines = "50"
    if len(args) > 1 and args[1].startswith("--lines"):
        lines = args[2] if len(args) > 2 else args[1].split("=")[-1]
    subprocess.run(["pm2", "logs", name, "--lines", lines])


def cmd_models(args: list[str]) -> None:
    """Show model recommendations for detected GPU."""
    from verallm.registry.__main__ import main as registry_main
    # Inject --recommend flag
    sys.argv = ["verallm.registry", "--recommend"] + args
    registry_main()


def cmd_network(args: list[str]) -> None:
    """Show network-wide status."""
    from neurons.network import cmd_network as _cmd_network
    _cmd_network(args)


def cmd_help() -> None:
    """Print usage."""
    print("""
  Verathos CLI

  Usage:
    verathos setup [miner|validator]   Interactive setup wizard
    verathos status                    Preflight readiness check
    verathos start [miner|validator]   Start via PM2
    verathos stop [miner|validator]    Stop via PM2
    verathos logs [miner|validator]    Tail PM2 logs
    verathos models [--category ...]   Model recommendations for your GPU
    verathos network [--json|--watch]  Network status dashboard
    verathos help                      Show this help
""")


COMMANDS = {
    "setup": cmd_setup,
    "status": cmd_status,
    "start": cmd_start,
    "stop": cmd_stop,
    "logs": cmd_logs,
    "models": cmd_models,
    "network": cmd_network,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        cmd_help()
        sys.exit(0)

    cmd_name = sys.argv[1]
    cmd_args = sys.argv[2:]

    handler = COMMANDS.get(cmd_name)
    if handler is None:
        print(f"Unknown command: {cmd_name}")
        cmd_help()
        sys.exit(1)

    handler(cmd_args)
    os._exit(0)  # Force-exit: bt.Subtensor WebSocket keeps non-daemon threads alive


if __name__ == "__main__":
    main()
