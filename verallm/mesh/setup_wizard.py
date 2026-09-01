"""``verathos mesh setup``: the interactive human door onto mesh onboarding.

All machinery lives in verallm/mesh/onboarding.py and is shared with the
flag-driven ``verathos setup mesh-coordinator`` / ``mesh-worker`` path, so
both doors mint pools the same way, use the same PM2 names and pool root,
and hit the same idempotency guard. This module only asks questions,
narrates, and finishes on something that demonstrably works (a verified
probe) instead of a wall of placeholder commands.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from verallm.mesh import flows, onboarding, render
from verallm.mesh.flows import (  # noqa: F401  (public seam for both doors)
    WizardIO,
    _pool_state,
)
from verallm.mesh.onboarding import (
    DEFAULT_MANAGER_PORT,
    MeshCoordinatorOptions,
    MeshCoordinatorResult,
)


def _ask_port(io: WizardIO, prompt: str, default: int) -> int:
    while True:
        raw = io.text(prompt, str(default))
        try:
            port = int(raw)
        except ValueError:
            io.out(render.warn("the port must be a number"))
            continue
        if 1 <= port <= 65535:
            return port
        io.out(render.warn("the port must be between 1 and 65535"))


def _ask_coordinator_options(
    io: WizardIO, *, allow_new_pool: bool | None
) -> MeshCoordinatorOptions:
    serving = io.choice(
        "where should this pool serve?",
        [
            ("local only", "try it out; not on the subnet, earns nothing"),
            (
                "on the subnet",
                "earn as a miner; validators verify your signed proofs",
            ),
        ],
        default=1,
    )
    opts = MeshCoordinatorOptions(
        serving_mode="dev" if serving == 1 else "subnet",
        skip_install=True,  # mesh setup runs inside an installed checkout
        non_interactive=False,
        allow_new_pool=allow_new_pool,
    )

    if serving == 2:
        network = io.choice(
            "which network?",
            [
                ("testnet", "netuid 405, no real TAO"),
                ("mainnet", "netuid 96, real emissions"),
            ],
            default=1,
        )
        binding = onboarding.resolve_network(
            "testnet" if network == 1 else "mainnet"
        )
        assert binding is not None
        opts.chain_id = binding["chain_id"]
        opts.netuid = binding["netuid"]
        io.out(
            render.ok(
                f"{binding['network']}: chain id {binding['chain_id']}, "
                f"netuid {binding['netuid']} "
                f"(from {Path(binding['chain_config']).name})"
            )
        )
        io.out("")
        io.out(
            render.dim(
                "  subnet pools sign snapshots, so they need your miner "
                "identity:"
            )
        )
        opts.owner_account = io.text(
            "owner coldkey address (SS58, receives your emissions)"
        )
        opts.coordinator_address = io.text(
            "coordinator EVM address (from your registered hotkey)"
        )
        opts.validator_shared_state = io.text(
            "validator shared-state path (read-only score mirror)",
            str(Path.home() / ".verathos" / "shared_state.json"),
        )
        io.out("")
        io.out(
            render.dim(
                "  chain node: snapshot epochs and the hot capacity audit "
                "follow the chain head; the public RPC is congested and "
                "NOT stable enough for either. Run your own subtensor "
                "node and point the pool (and its workers, who inherit "
                "it at join) at it."
            )
        )
        opts.subtensor_endpoint = io.text(
            "your subtensor node (ws://host:9944; empty = public default, "
            "not recommended)"
        ).strip()
        uid_raw = io.text("your miner UID on the subnet")
        epoch_raw = io.text("current epoch (btcli subnets show)")
        try:
            opts.coordinator_uid = int(uid_raw)
            opts.epoch = int(epoch_raw)
        except ValueError:
            raise SystemExit("UID and epoch must be integers")
        io.out("")
        io.out(
            render.dim(
                "  the DRIVER worker signs receipts, so locally joined GPUs "
                "need a signing wallet + validator allowlist:"
            )
        )
        opts.worker_wallet_name = io.text(
            "worker signing wallet name (empty = configure later)"
        )
        if opts.worker_wallet_name:
            opts.worker_wallet_hotkey = io.text(
                "worker wallet hotkey", "default"
            )
            opts.validator_allowlist_path = io.text(
                "validator allowlist file path"
            )

        # Subnet mode requires an HTTPS manager endpoint UNLESS the pool
        # stays on this box (loopback). Asking up front beats collecting
        # eight answers and then dying on the preflight.
        reach = io.choice(
            "will other machines join this pool?",
            [
                (
                    "no, single box",
                    "manager stays loopback-only; workers run on this "
                    "machine",
                ),
                (
                    "yes",
                    "the manager endpoint must be HTTPS for remote "
                    "workers",
                ),
            ],
            default=1,
        )
        if reach == 1:
            opts.manager_host = "127.0.0.1"
        else:
            tls = io.choice(
                "how should the manager serve HTTPS?",
                [
                    (
                        "self-minted certificate (recommended)",
                        "the manager mints its own pinned certificate; "
                        "join tokens authenticate against it with no "
                        "files to manage",
                    ),
                    (
                        "nginx front",
                        "you run scripts/setup_https.sh and enter the "
                        "public https:// URL",
                    ),
                    (
                        "certificate files",
                        "the manager serves TLS directly from cert + key "
                        "files",
                    ),
                ],
                default=1,
            )
            if tls == 1:
                # The pinned self-minted listener is the only TLS mode
                # that can mint working join commands; BYO certs and
                # nginx fronts cannot pin, so they stay opt-in.
                opts.api_tls_port = _ask_port(
                    io, "HTTPS API port", opts.manager_port + 1
                )
            elif tls == 2:
                opts.manager_endpoint = io.text(
                    "public manager URL (https://host:port)"
                ).strip()
                if not opts.manager_endpoint.startswith("https://"):
                    raise SystemExit(
                        "subnet pools reachable from other machines need "
                        "an https:// manager URL"
                    )
            else:
                opts.tls_cert = io.text("TLS certificate path")
                opts.tls_key = io.text("TLS key path")

    if not opts.manager_host and not opts.manager_endpoint:
        # "Local only" is about the SUBNET (no chain, no earnings), not
        # about network reach: a dev pool spanning several machines is
        # normal. The manager listens on 0.0.0.0 either way; this address
        # is only what the join token tells workers to dial, so a
        # loopback default would just make external joins impossible
        # until the pool is re-created.
        detected = onboarding.detect_public_ip()
        io.out("")
        if detected:
            io.out(render.dim(f"  detected public IP: {detected}"))
        io.out(
            render.dim(
                "  workers dial this address to join; enter 127.0.0.1 to "
                "keep the pool joinable from THIS machine only"
            )
        )
        opts.manager_host = io.text(
            "manager address workers dial", detected or "127.0.0.1"
        )
    opts.manager_port = _ask_port(io, "manager port", DEFAULT_MANAGER_PORT)
    opts.pools_root = Path(
        io.text("pool state directory", str(opts.pools_root))
    ).expanduser()
    return opts


def _repair_or_confirm_manager(
    io: WizardIO, result: MeshCoordinatorResult
) -> None:
    """After a reuse choice: make sure the pool's manager is actually up."""

    # This box hosts the manager, so dial loopback first: a coordinator
    # that cannot hairpin to its own public IP is still healthy.
    probe = onboarding.probe_manager(
        onboarding.loopback_equivalent(result.manager_endpoint)
    )
    if not probe.reachable and not onboarding.endpoint_is_loopback(
        result.manager_endpoint
    ):
        probe = onboarding.probe_manager(result.manager_endpoint)
    if probe.reachable and probe.pool_id == result.pool_id:
        io.out(render.ok(f"manager already healthy for {result.pool_id}"))
        return
    if probe.reachable:
        io.out(
            render.warn(
                f"the manager on {result.manager_endpoint} serves pool "
                f"{probe.pool_id!r}, not {result.pool_id!r}"
            )
        )
        raise SystemExit(
            "another pool's manager holds this endpoint; stop it first "
            "or reuse that pool instead"
        )
    io.out(
        render.warn(
            f"no manager answering on {result.manager_endpoint}; starting it"
        )
    )
    from urllib.parse import urlparse

    # A saved PM2 app still carries the manager's ORIGINAL argv (including
    # any TLS flags); restarting it is the only repair that cannot
    # silently downgrade an HTTPS manager to plain HTTP.
    apps = onboarding.pm2_mesh_apps()
    restart_name = next(
        (
            name
            for name in (
                result.pm2_name,
                onboarding.POOL_MANAGER_PM2_NAME,
                onboarding.manager_pm2_name_for_pool(result.pool_id),
            )
            if name in apps
        ),
        "",
    )
    if restart_name:
        io.out(render.dim(f"  $ pm2 restart {restart_name}"))
        subprocess.run(["pm2", "restart", restart_name], capture_output=True)
        probe = onboarding.probe_manager(
            onboarding.loopback_equivalent(result.manager_endpoint),
            timeout=5.0,
        )
        if probe.reachable and probe.pool_id == result.pool_id:
            crash = onboarding.manager_crash_loop_error(restart_name)
            if crash:
                raise SystemExit(
                    f"pool manager did not stay healthy: {crash}"
                )
            io.out(render.ok("manager healthy"))
            return
        io.out(
            render.warn(
                "the PM2 restart did not bring the manager back; starting "
                "it fresh"
            )
        )
    parsed = urlparse(result.manager_endpoint)
    if (parsed.scheme or "http") == "https":
        raise SystemExit(
            "this pool's manager serves HTTPS and its TLS flags are not "
            "recorded here; start it manually:\n"
            f"  verathos mesh pool serve --pool {result.pool_dir} "
            "--host 0.0.0.0 "
            f"--port {parsed.port or 443} --tls-cert <cert> --tls-key <key>"
        )
    opts = MeshCoordinatorOptions(
        manager_port=int(parsed.port or DEFAULT_MANAGER_PORT)
    )
    onboarding.start_manager_for_pool(result.pool_dir, opts)
    io.out(render.ok("manager healthy"))


def _coordinator_flow(io: WizardIO, *, join_gpus: bool) -> int:
    scan = onboarding.scan_existing_setup(
        Path.home() / ".verathos" / "pools", DEFAULT_MANAGER_PORT
    )
    result: MeshCoordinatorResult | None = None
    opts: MeshCoordinatorOptions | None = None

    if scan.anything:
        io.out("")
        io.out(render.section("  this machine already has mesh state:"))
        for pool_dir in scan.pools:
            io.out(render.ok(f"pool {pool_dir.name}  ({pool_dir})"))
        if scan.manager.reachable:
            io.out(
                render.ok(
                    f"a manager answers on port {DEFAULT_MANAGER_PORT} "
                    f"for pool {scan.manager.pool_id}"
                )
            )
        for app in scan.pm2_apps:
            io.out(render.ok(f"pm2 app {app}"))
        action = io.choice(
            "what do you want to do?",
            [
                ("reuse it", "keep the existing pool; restart its manager if down"),
                ("create another pool", "advanced: a second pool on this machine"),
                ("quit", "leave everything untouched"),
            ],
            default=1,
        )
        if action == 3:
            return 0
        if action == 1:
            if not scan.pools:
                raise SystemExit(
                    "a manager is running but its pool state dir was not "
                    "found on this machine; use --pool on the other commands"
                )
            pool_dir = scan.pools[0]
            if len(scan.pools) > 1:
                pick = io.choice(
                    "which pool?",
                    [(p.name, str(p)) for p in scan.pools],
                    default=1,
                )
                pool_dir = scan.pools[pick - 1]
            result = onboarding.result_from_existing_pool(pool_dir)
            _repair_or_confirm_manager(io, result)
            opts = MeshCoordinatorOptions(non_interactive=False)
        else:
            opts = _ask_coordinator_options(io, allow_new_pool=True)
            if opts.manager_port == DEFAULT_MANAGER_PORT and (
                scan.manager.reachable
            ):
                io.out(
                    render.warn(
                        f"port {DEFAULT_MANAGER_PORT} is taken by pool "
                        f"{scan.manager.pool_id}; pick a different port"
                    )
                )
                opts.manager_port = _ask_port(
                    io, "manager port", DEFAULT_MANAGER_PORT + 1
                )
    else:
        opts = _ask_coordinator_options(io, allow_new_pool=None)

    if result is None:
        assert opts is not None
        result = onboarding.setup_mesh_coordinator(opts)
    onboarding.print_coordinator_result(result)

    if join_gpus and opts is not None:
        onboarding.maybe_join_local_gpus(result, opts)

    _finish_with_probe(io, result.pool_dir)
    return 0


def _finish_with_probe(io: WizardIO, pool_dir: Path) -> None:
    """End on something that works: launch a model and verify a real chat.

    Model entries appear when workers advertise catalogs, so give freshly
    joined workers a moment before concluding there is nothing to launch.
    Day-2 operations live in ``verathos mesh manage``, which reuses the
    exact same flows.
    """

    if flows._serving_mesh(pool_dir):
        io.out(render.ok(f"a mesh is already serving on {pool_dir.name}"))
        flows._offer_deploy(io, pool_dir)
        flows._print_next_moves(io, pool_dir)
        return

    models = flows._pool_models(pool_dir)
    if not models and not flows._pool_view(pool_dir)["workers"]:
        # A worker's model catalog arrives WITH its join, so the only
        # thing worth waiting for is workers that have not joined yet.
        # A pool with joined workers and no models needs a model ADDED —
        # waiting would be 90 dead seconds with a nonsensical message.
        deadline = time.monotonic() + 90
        io.out("")
        with render.spinner(
            "waiting for the GPU workers to join the pool..."
        ) as spin:
            while time.monotonic() < deadline:
                view = flows._pool_view(pool_dir)
                if view["workers"]:
                    break
                spin.update(
                    "waiting for the GPU workers to join the pool... "
                    "(none joined yet)"
                )
                time.sleep(3)
        models = flows._pool_models(pool_dir)

    if not models and flows.pool_serving_mode(pool_dir) == "subnet":
        io.out(render.warn("the pool has no models yet"))
        io.out(
            render.dim(
                "  subnet pools serve models from the subnet owner's "
                "on-chain ModelSpec; the deploy pipeline reads it and "
                "sets everything up"
            )
        )
        flows._offer_deploy(io, pool_dir)
        flows._print_next_moves(io, pool_dir)
        return

    outcome = flows.launch_flow(io, pool_dir)
    if outcome == "verified":
        _offer_private_api(io, pool_dir)
        flows._offer_deploy(io, pool_dir)
    flows._print_next_moves(io, pool_dir)


def _offer_private_api(io: WizardIO, pool_dir: Path) -> None:
    """First-run offer: mint an API key for the freshly verified mesh.

    The natural "what do I get" moment is right after the first verified
    launch: one confirm mints the key, and the api-keys screen in
    `verathos mesh manage` (or `verathos mesh apikey expose`) takes it to
    the internet later. Declining costs nothing; the epilogue prints the
    verbs either way.
    """

    io.out("")
    io.out(
        render.dim(
            "  your mesh also serves an OpenAI-compatible API for your "
            "own tools (proof-verified replies, key-authed, rate limited)"
        )
    )
    if not io.confirm("create your first API key now?", default=True):
        return
    payload = onboarding.pool_management_request(
        pool_dir, "/v1/pool/api-keys", {"action": "create", "name": "first"}
    )
    if not payload or not payload.get("api_key"):
        io.out(render.warn("the manager did not answer; is it running?"))
        return
    io.out(render.ok("API key minted (shown ONCE, store it now):"))
    io.out(f"    {payload['api_key']}")
    endpoint = str(
        flows._pool_state(pool_dir).get("manager_endpoint", "")
        or "http://127.0.0.1:9500"
    ).rstrip("/")
    io.out("  use it with any OpenAI client:")
    io.out(
        render.cyan(
            f"    curl -H 'Authorization: Bearer <key>' {endpoint}/v1/models"
        )
    )
    io.out(
        render.dim(
            "  internet access later: verathos mesh apikey expose "
            "--port <public-port> (own https listener + live self-test)"
        )
    )


def _worker_flow(io: WizardIO) -> int:
    io.out("")
    io.out(
        render.dim(
            "  the join token comes from the coordinator machine: it is\n"
            "  printed at pool creation, and again by\n"
            "  `verathos mesh pool join-token`"
        )
    )
    raw = io.text("join token (vtpool_...) or a token file path")
    if not raw:
        io.out(render.warn("a worker needs the pool join token; aborting"))
        return 2

    # A file path and a pasted token get the SAME diagnostics: decode,
    # loopback warning, and a reachability probe before anything runs.
    token_text = raw
    if not raw.startswith("vtpool_"):
        token_path = Path(raw).expanduser()
        if not token_path.is_file():
            io.out(
                render.fail(
                    f"{raw!r} is neither a vtpool_ token nor a token file"
                )
            )
            return 2
        token_text = token_path.read_text(encoding="utf-8").strip()

    try:
        info = onboarding.describe_worker_token(token_text)
    except ValueError as exc:
        io.out(render.fail(f"that token does not decode: {exc}"))
        return 2
    io.out(
        render.ok(
            f"token for pool {info['pool_id']} at "
            f"{info['manager_endpoint']}"
        )
    )
    if info["loopback"]:
        io.out(
            render.warn(
                "this token points at 127.0.0.1, i.e. at THIS machine. "
                "A token minted on another box with a loopback manager "
                "endpoint can never reach its pool from here."
            )
        )
    if not info["reachable"]:
        io.out(
            render.warn(
                f"the manager does not answer from here: {info['error']}"
            )
        )
        if not io.confirm("join anyway (it will retry forever)?", False):
            return 2
    elif info["reachable_pool_id"] != info["pool_id"]:
        io.out(
            render.warn(
                f"the endpoint answers for pool "
                f"{info['reachable_pool_id']!r}, not {info['pool_id']!r} "
                "(stale manager?)"
            )
        )
        if not io.confirm("join anyway?", False):
            return 2

    passthrough: list[str] = []
    detected = onboarding.detect_public_ip()
    io.out(
        render.dim(
            "  mesh peers dial this machine directly; behind NAT the "
            "auto-detected interface address is unreachable from outside"
        )
    )
    advertise = io.text(
        "address other mesh members dial this box", detected or ""
    ).strip()
    if advertise:
        passthrough += ["--advertise-host", advertise]
    wallet = io.text(
        "signing wallet name (needed on subnet pools; empty = skip)"
    ).strip()
    if wallet:
        passthrough += ["--wallet-name", wallet]
        passthrough += ["--wallet-hotkey", io.text("wallet hotkey", "default")]
        allowlist = io.text(
            "validator allowlist file path (empty = skip)"
        ).strip()
        if allowlist:
            passthrough += ["--validator-allowlist-path", allowlist]
    onboarding.setup_mesh_worker(token_text, passthrough)
    return 0


def run_mesh_setup(args: Any, io: WizardIO | None = None) -> int:
    io = io or WizardIO()
    out = io.out

    from verallm import __version__

    out("")
    for line in render.brand_banner(
        [
            "VERATHOS · mesh setup",
            "sleipnir mesh inference",
            "gleipnir proof protocol",
            f"v{__version__} · verified GGUF inference, step by step",
            "every action is printed before it runs; Ctrl-C aborts safely",
        ]
    ):
        out(line)

    role = io.choice(
        "what should this machine be?",
        [
            ("coordinator", "runs the pool control plane (no GPU needed)"),
            ("GPU worker", "joins an existing pool with a join token"),
            ("both", "single-box pool: coordinator plus this machine's GPUs"),
        ],
        default=3,
    )
    if role == 2:
        return _worker_flow(io)
    return _coordinator_flow(io, join_gpus=(role == 3))
