#!/usr/bin/env python3
"""Flag-driven mesh onboarding: coordinator and worker roles.

This is the agent/script door onto the shared onboarding machinery in
verallm/mesh/onboarding.py; the interactive human door is
``verathos mesh setup`` (verallm/mesh/setup_wizard.py). Neither implements
pool creation, PM2 naming, or health polling itself — one implementation,
two front-ends.

Separate from neurons/wizard.py on purpose. That module owns the vLLM
miner/validator flows and ecosystem.config.js, which stay untouched; this
one only borrows its prompt helpers. Mesh processes register with PM2
directly (pm2 start + pm2 save), never through ecosystem.config.js, so
`verathos status` keeps reporting the vLLM path correctly.

Roles:
  mesh-coordinator  Create the pool state, start the pool manager under PM2,
                    and print the worker join one-liner.
  mesh-worker       Hand off to scripts/join_pool.sh with the given token.

Usage:
  verathos setup mesh-coordinator [flags]
  verathos setup mesh-worker --token vtpool_... [join_pool.sh flags]
  python neurons/mesh_wizard.py mesh-coordinator [flags]

Every prompt has a flag so the whole flow runs non-interactively with --yes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from neurons.wizard import (
    _banner,
    _confirm,
    _header,
    _prompt,
    green,
    red,
)
from verallm.mesh.onboarding import (
    DEFAULT_MANAGER_PORT,
    MeshCoordinatorOptions,
    MeshCoordinatorResult,
    POOL_MANAGER_PM2_NAME,
    detect_public_ip as _detect_public_ip,
    find_repo_root as _find_repo_root,
    maybe_join_local_gpus,
    print_coordinator_result as _print_coordinator_result,
    resolve_manager_endpoint as _resolve_manager_endpoint,
    resolve_network,
    setup_mesh_coordinator,
    setup_mesh_worker,
    validator_preflight as _validator_preflight,
)

__all__ = [
    "DEFAULT_MANAGER_PORT",
    "MeshCoordinatorOptions",
    "MeshCoordinatorResult",
    "POOL_MANAGER_PM2_NAME",
    "main",
    "maybe_join_local_gpus",
    "run_mesh_wizard",
    "setup_mesh_coordinator",
    "setup_mesh_worker",
]


def _coordinator_options_from_args(args: argparse.Namespace) -> MeshCoordinatorOptions:
    opts = MeshCoordinatorOptions(
        serving_mode=args.serving_mode,
        manager_host=args.host,
        manager_port=args.port,
        manager_endpoint=args.manager_endpoint,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        api_tls_port=int(getattr(args, "api_tls_port", 0) or 0),
        owner_account=args.owner_account,
        coordinator_address=args.coordinator_address,
        validator_shared_state=args.validator_shared_state,
        chain_id=args.chain_id,
        netuid=args.netuid,
        coordinator_uid=args.coordinator_uid,
        epoch=args.epoch,
        snapshot_ttl_seconds=args.snapshot_ttl_seconds,
        pools_root=Path(args.pools_root),
        skip_install=args.skip_install,
        non_interactive=args.yes,
        join_local_gpus=args.join_local_gpus,
        allow_new_pool=True if args.new_pool else None,
        worker_wallet_name=args.worker_wallet_name,
        worker_wallet_hotkey=args.worker_wallet_hotkey,
        validator_allowlist_path=args.validator_allowlist_path,
    )
    if args.network:
        binding = resolve_network(args.network)
        if binding is None:
            opts.serving_mode = "dev"
        else:
            opts.serving_mode = "subnet"
            # Explicit --chain-id/--netuid stay overrides for unusual setups.
            opts.chain_id = opts.chain_id or binding["chain_id"]
            opts.netuid = opts.netuid or binding["netuid"]
            # The manager's lease renewer needs the chain config too, or the
            # on-chain registration quietly expires after its 24h lease.
            opts.chain_config = opts.chain_config or str(
                binding.get("chain_config", "") or ""
            )
    from verallm.mesh.onboarding import derive_subnet_setup_defaults

    for note in derive_subnet_setup_defaults(opts):
        print(f"  derived: {note}")
    return opts


def _interactive_fill(opts: MeshCoordinatorOptions) -> MeshCoordinatorOptions:
    _banner("mesh coordinator")
    _header(1, 3, "Where should this pool serve?")
    print("  dev:    local only; not on the subnet, earns nothing")
    print(
        "  subnet: on the subnet as a miner; validators verify the\n"
        "          pool's signed proofs"
    )
    opts.serving_mode = _prompt("Serving mode (dev/subnet)", opts.serving_mode)
    _header(2, 3, "Manager endpoint")
    if not opts.manager_host:
        detected = _detect_public_ip()
        opts.manager_host = _prompt("Address workers dial", detected or "127.0.0.1")
    if not (opts.tls_cert and opts.tls_key) and not opts.manager_endpoint:
        if _confirm("Serve TLS directly (have cert + key files)?", default=False):
            opts.tls_cert = _prompt("TLS certificate path")
            opts.tls_key = _prompt("TLS key path")
    if opts.serving_mode in ("subnet", "validator"):
        _header(3, 3, "Subnet binding")
        if opts.chain_id is None or opts.netuid is None:
            network = _prompt("Network (testnet/mainnet)", "testnet")
            binding = resolve_network(network)
            if binding is not None:
                opts.chain_id = opts.chain_id or binding["chain_id"]
                opts.netuid = opts.netuid or binding["netuid"]
        opts.owner_account = opts.owner_account or _prompt("Owner account (SS58)")
        opts.coordinator_address = opts.coordinator_address or _prompt(
            "Coordinator EVM address"
        )
        opts.validator_shared_state = opts.validator_shared_state or _prompt(
            "Validator shared state path"
        )
        opts.coordinator_uid = (
            opts.coordinator_uid
            if opts.coordinator_uid is not None
            else int(_prompt("Coordinator UID"))
        )
        opts.epoch = opts.epoch if opts.epoch is not None else int(_prompt("Epoch"))
        # The validator driver refuses to serve without a signing wallet
        # and allowlist; asking here beats a launch that dies minutes in.
        if not opts.worker_wallet_name:
            opts.worker_wallet_name = _prompt(
                "Worker signing wallet name (for locally joined GPUs)"
            )
        if opts.worker_wallet_name and not opts.worker_wallet_hotkey:
            opts.worker_wallet_hotkey = _prompt("Worker wallet hotkey", "default")
        if not opts.validator_allowlist_path:
            opts.validator_allowlist_path = _prompt(
                "Validator allowlist file path"
            )
    return opts


def run_mesh_wizard(role: str, argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    if role == "mesh-worker":
        parser = argparse.ArgumentParser(prog="verathos setup mesh-worker")
        parser.add_argument("--token", default="")
        parser.add_argument("--token-file", default="")
        args, passthrough = parser.parse_known_args(argv)
        if args.token_file:
            passthrough = ["--token-file", args.token_file, *passthrough]
        elif not args.token:
            raise SystemExit(
                "mesh-worker needs --token vtpool_... (from your coordinator)"
            )
        setup_mesh_worker(args.token, passthrough)
        return
    if role != "mesh-coordinator":
        raise SystemExit(f"unknown mesh role: {role}")

    parser = argparse.ArgumentParser(prog="verathos setup mesh-coordinator")
    parser.add_argument(
        "--serving-mode",
        default="dev",
        choices=["dev", "subnet", "validator"],
        help=(
            "dev = local only, no chain binding; subnet = on the subnet, "
            "signs snapshots that remote validators verify ('validator' is "
            "the deprecated alias)"
        ),
    )
    parser.add_argument(
        "--network",
        default="",
        choices=["", "local", "testnet", "mainnet"],
        help=(
            "shorthand: local = dev mode; testnet/mainnet = subnet mode with "
            "chain-id/netuid read from the shipped chain config"
        ),
    )
    parser.add_argument("--host", default="", help="Address workers dial")
    parser.add_argument("--port", type=int, default=DEFAULT_MANAGER_PORT)
    parser.add_argument(
        "--manager-endpoint",
        default="",
        help="Full public URL override (nginx TLS front)",
    )
    parser.add_argument("--tls-cert", default="")
    parser.add_argument("--tls-key", default="")
    parser.add_argument(
        "--api-tls-port",
        type=int,
        default=0,
        help=(
            "Self-minted pinned HTTPS API listener (recommended for "
            "remote-worker pools; the only TLS mode join tokens can pin)"
        ),
    )
    parser.add_argument("--owner-account", default="")
    parser.add_argument("--coordinator-address", default="")
    parser.add_argument("--validator-shared-state", default="")
    parser.add_argument("--chain-id", type=int, default=None)
    parser.add_argument("--netuid", type=int, default=None)
    parser.add_argument("--coordinator-uid", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--snapshot-ttl-seconds", type=int, default=86_400)
    parser.add_argument(
        "--pools-root", default=str(Path.home() / ".verathos" / "pools")
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Non-interactive; flags only"
    )
    parser.add_argument(
        "--new-pool",
        action="store_true",
        help=(
            "Mint a new pool even when this machine already has mesh state "
            "(without this, a re-run refuses instead of duplicating)"
        ),
    )
    parser.add_argument(
        "--worker-wallet-name",
        default="",
        help=(
            "Signing wallet for locally joined GPUs; a validator pool's "
            "driver refuses to serve without it"
        ),
    )
    parser.add_argument("--worker-wallet-hotkey", default="")
    parser.add_argument(
        "--validator-allowlist-path",
        default="",
        help="Allowlist file the validator driver requires",
    )
    join_group = parser.add_mutually_exclusive_group()
    join_group.add_argument(
        "--join-local-gpus",
        dest="join_local_gpus",
        action="store_true",
        default=None,
        help="Also enroll this machine's GPUs as worker units",
    )
    join_group.add_argument(
        "--no-join-local-gpus",
        dest="join_local_gpus",
        action="store_false",
        help="Coordinator only, even when GPUs are present",
    )
    args = parser.parse_args(argv)
    opts = _coordinator_options_from_args(args)
    if not opts.non_interactive:
        opts = _interactive_fill(opts)
    elif not opts.manager_host and not opts.manager_endpoint:
        opts.manager_host = _detect_public_ip() or "127.0.0.1"
    try:
        result = setup_mesh_coordinator(opts)
    except subprocess.CalledProcessError as exc:
        print(red(f"  Setup step failed: {exc}"))
        raise SystemExit(1) from exc
    _print_coordinator_result(result)
    maybe_join_local_gpus(result, opts)
    print(green("  Done."))


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: mesh_wizard.py mesh-coordinator|mesh-worker [flags]")
        raise SystemExit(2)
    run_mesh_wizard(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
