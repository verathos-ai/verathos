"""``verathos mesh manage``: the day-2 operator board.

Setup runs once; this is every day after. One screen shows the machines
(with latency links), the running meshes with live state, and the model
catalog matched to this pool's hardware — then a small menu drives the
same shared flows setup uses: launch, stop, switch, chat, probe,
machines, go on the subnet. Model onboarding is the subnet owner's
process and deliberately absent here. Every action prints its
``$ verathos mesh ...`` equivalent first, so the flag lane for scripts
and agents stays discoverable.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from verallm.mesh import flows, render
from verallm.mesh.flows import WizardIO


def _board(
    io: WizardIO, pool_dir: Path, *, refresh_catalog: bool = False
) -> dict[str, Any]:
    """Render the full board; returns the view it rendered from."""

    from verallm.mesh import model_catalog, panels

    view = flows._pool_view(pool_dir)
    mode = flows.pool_mode_label(pool_dir)
    io.out("")
    for line in panels.pool_panel(view, pool_id=pool_dir.name, mode=mode):
        io.out(line)
    chain = (
        flows.pool_operator_score(pool_dir)
        if view.get("serving_mode") == "subnet"
        else {}
    )
    for line in panels.subnet_status_lines(view, chain=chain):
        io.out(line)
    io.out("")
    for line in panels.mesh_board(view, score_of=model_catalog.base_score):
        io.out(line)
    io.out("")
    io.out(render.section("  what can this pool serve?"))
    with render.spinner("assembling the model catalog..."):
        candidates = model_catalog.assemble_candidates(
            view,
            network=flows.pool_network(pool_dir),
            refresh=refresh_catalog,
        )
    lines, _launchable = panels.catalog_table(candidates)
    for line in lines:
        io.out(line)
    if not view["live"]:
        io.out(
            render.warn(
                "the pool manager is not answering; most actions need it "
                "(pm2 restart verathos-pool-manager)"
            )
        )
    return view


def _overview(io: WizardIO, pool_dir: Path) -> dict[str, Any]:
    """Machines + meshes between actions: state at a glance.

    The catalog table and the latency map stay on the FULL board
    (entry and the explicit redraw) - after a task the operator wants
    to see what changed (mesh states, leases, contexts), not re-read
    what the pool could hypothetically serve.
    """

    from verallm.mesh import model_catalog, panels

    view = flows._pool_view(pool_dir)
    mode = flows.pool_mode_label(pool_dir)
    io.out("")
    for line in panels.pool_panel(
        view, pool_id=pool_dir.name, mode=mode, links=False
    ):
        io.out(line)
    chain = (
        flows.pool_operator_score(pool_dir)
        if view.get("serving_mode") == "subnet"
        else {}
    )
    for line in panels.subnet_status_lines(view, chain=chain):
        io.out(line)
    io.out("")
    for line in panels.mesh_board(view, score_of=model_catalog.base_score):
        io.out(line)
    return view


def _pick_mesh(
    io: WizardIO,
    view: dict[str, Any],
    prompt: str,
    *,
    statuses: tuple[str, ...] = (),
) -> str:
    from verallm.mesh import model_catalog, panels

    meshes = {
        key: mesh
        for key, mesh in (view.get("meshes") or {}).items()
        if not statuses or str(mesh.get("status", "")) in statuses
    }
    if not meshes:
        # Say WHY, not just "no": the operator just saw a mesh table and
        # "no matching mesh" reads like a bug when everything is merely
        # stopping/forming (operator feedback).
        others = {
            str(mesh.get("model_id", "?") or "?"): str(
                mesh.get("status", "?") or "?"
            )
            for mesh in (view.get("meshes") or {}).values()
        }
        if others:
            detail = ", ".join(
                f"{model} is {status}" for model, status in sorted(others.items())
            )
            io.out(
                render.warn(
                    f"nothing is serving right now ({detail}); launch a "
                    "model (option 1) or wait for the transition to finish"
                )
            )
        else:
            io.out(
                render.warn(
                    "no mesh exists yet; launch a model (option 1) first"
                )
            )
        return ""

    def _order(key: str) -> tuple:
        model_id = str((meshes[key] or {}).get("model_id", "") or "")
        try:
            score = float(model_catalog.base_score(model_id) or 0.0)
        except Exception:
            score = 0.0
        return (-score, model_id, key)

    keys = sorted(meshes, key=_order)
    if len(keys) == 1:
        return keys[0]

    def _label(key: str) -> str:
        mesh = meshes[key]
        parts = [
            str(mesh.get("model_id", "?") or "?"),
            str(mesh.get("status", "?") or "?"),
        ]
        where = panels.mesh_hardware_label(view, mesh)
        if where:
            parts.append(where)
        return " · ".join(parts)

    pick = io.choice(
        prompt,
        [(key, _label(key)) for key in keys],
        default=1,
        back=True,
    )
    if pick == 0:
        return ""
    return keys[pick - 1]


def _stop_mesh(io: WizardIO, pool_dir: Path, view: dict[str, Any]) -> bool:
    mesh_key = _pick_mesh(io, view, "which mesh should stop?")
    if not mesh_key:
        return False
    model_id = str(
        (view["meshes"].get(mesh_key) or {}).get("model_id", "?") or "?"
    )
    if not io.confirm(f"stop {mesh_key} ({model_id})?", default=True):
        return False
    argv = [
        "pool",
        "stop",
        "--mesh-key",
        mesh_key,
        "--pool",
        str(pool_dir),
    ]
    io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
    code, output = flows._run_cli(argv)
    if code != 0:
        io.out(render.fail(f"stop failed:\n{output.strip()}"))
        return False
    deadline = time.monotonic() + 120
    with render.spinner(f"stopping {mesh_key}..."):
        while time.monotonic() < deadline:
            current = flows._pool_view(pool_dir)
            if mesh_key not in (current.get("meshes") or {}):
                io.out(render.ok(f"{mesh_key} stopped; its GPUs are free"))
                return True
            time.sleep(3)
    io.out(
        render.warn(
            f"{mesh_key} is still stopping (workers confirm asynchronously); "
            "check verathos mesh fleet"
        )
    )
    return True


def _watch_launch(io: WizardIO, pool_dir: Path, view: dict[str, Any]) -> None:
    """Re-attach to a forming mesh's live progress at any time.

    The same member-narrated wait the launch flow runs. Ctrl-C detaches
    back to the BOARD (the launch keeps going pool-side) instead of
    killing the whole manage session.
    """

    mesh_key = _pick_mesh(
        io,
        view,
        "status of which launch?",
        statuses=("fetching", "driving", "joining", "assigned"),
    )
    if not mesh_key:
        return
    model_id = str(
        (view["meshes"].get(mesh_key) or {}).get("model_id", "?") or "?"
    )
    try:
        ready = flows.wait_for_mesh(io, pool_dir, mesh_key, model_id)
        if ready:
            flows.run_verified_probe(io, pool_dir, mesh_key)
        elif ready is None:
            io.out(
                render.dim(
                    "  detached; the launch keeps going (watch it again "
                    "from the menu)"
                )
            )
    except KeyboardInterrupt:
        io.out("")
        io.out(
            render.dim(
                "  detached; the launch keeps going (watch it again from "
                "the menu)"
            )
        )


def _probe_mesh(io: WizardIO, pool_dir: Path, view: dict[str, Any]) -> None:
    mesh_key = _pick_mesh(
        io, view, "which mesh should the probe test?", statuses=("serving",)
    )
    if not mesh_key:
        return
    flows.run_verified_probe(io, pool_dir, mesh_key)


def _chat(io: WizardIO, pool_dir: Path, view: dict[str, Any]) -> None:
    mesh_key = _pick_mesh(
        io, view, "chat with which mesh?", statuses=("serving",)
    )
    if not mesh_key:
        return
    argv = [
        "chat",
        "--pool",
        str(pool_dir),
        "--mesh-key",
        mesh_key,
    ]
    io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
    # In-process, not a child interpreter: a fresh python re-pays every
    # import at startup, and on a box whose disk a 200 GB model fetch is
    # saturating that cold start was observed to look like a hard
    # hang between the printed command and the chat banner.
    from verallm.mesh import cli as mesh_cli

    ns = mesh_cli.build_parser().parse_args(argv)
    try:
        mesh_cli.cmd_mesh_chat(ns)
    except SystemExit as exc:
        if str(exc) not in ("", "0"):
            io.out(render.warn(str(exc)))
    except KeyboardInterrupt:
        io.out("")


def _subnet_registration(
    io: WizardIO, pool_dir: Path, view: dict[str, Any]
) -> None:
    """Per-mesh registration menu: EVERY serving mesh gets its own row.

    The registration store is per model, so a pool serving glm on one
    machine and qwen on another shows both: registered meshes offer a
    re-deploy, local-only meshes offer to register (probe gate + on-chain
    registration). Before this, the single stored registration was the
    whole menu and a second serving mesh had no path onto the subnet
    except the flag-driven CLI (observed on the manage board)."""

    from verallm.mesh.panels import _hours_text

    registrations = flows.pool_registrations(view)
    meshes = dict(view.get("meshes") or {})
    deploys = dict(view.get("deploys") or {})
    # Unregistered models lead the list (they are what this menu is FOR);
    # registered ones sit under their own colored header so a re-deploy
    # takes deliberate aim instead of an accidental Enter (the default
    # always points at the first unregistered row).
    unregistered: list[tuple[str, str, str, str]] = []
    registered: list[tuple[str, str, str, str]] = []
    serving_models: set[str] = set()
    for mesh_key, mesh in sorted(meshes.items()):
        if str(mesh.get("status", "") or "") != "serving":
            continue
        model_id = str(mesh.get("model_id", "") or "")
        serving_models.add(model_id)
        deploying = deploys.get(model_id)
        if deploying:
            # A deploy for this model is ALREADY running (its own process,
            # posting stage markers to the manager). Offering "register"
            # here was one keypress from two deploys racing the same chain
            # slot.
            unregistered.append(
                (
                    model_id,
                    "deploy in progress · stage: "
                    f"{deploying.get('stage', '?')} (wait for it; no "
                    "second deploy)",
                    "",
                    "",
                )
            )
            continue
        registration = registrations.get(model_id)
        if registration:
            lease = ""
            if registration.get("expires_at"):
                lease = " · lease " + _hours_text(
                    float(registration["expires_at"]) - time.time()
                )
            registered.append(
                (
                    model_id,
                    f"index {registration.get('index', '?')}{lease} · "
                    "serving",
                    mesh_key,
                    model_id,
                )
            )
        else:
            unregistered.append(
                (
                    model_id,
                    "local only · probe gate + on-chain registration",
                    mesh_key,
                    model_id,
                )
            )
    for model_id, registration in sorted(registrations.items()):
        if model_id in serving_models:
            continue
        # Registered but nothing serves it: the lease will lapse. Offering
        # the deploy relaunches it (or `verathos mesh retire` frees it).
        registered.append(
            (
                model_id,
                f"index {registration.get('index', '?')} · NOT serving · "
                "re-deploy relaunches it, or retire",
                str(registration.get("mesh_key", "") or ""),
                model_id,
            )
        )
    rows = unregistered + registered
    if not rows:
        io.out("")
        io.out(render.dim("  nothing is serving yet; launch a model first"))
        flows._offer_deploy(io, pool_dir)
        return
    io.out("")
    io.out(
        render.dim(
            "  chain truth: verathos mesh registration-status  ·  "
            "retire: verathos mesh retire <model>"
        )
    )
    options: list[tuple[str, str]] = []
    if unregistered:
        options.append(
            ("", render.green("  ready to register - not on the subnet yet"))
        )
        options += [(label, detail) for label, detail, _, _ in unregistered]
    if registered:
        options.append(("", ""))
        options.append(
            (
                "",
                render.yellow(
                    "  already registered - pick one only to RE-deploy "
                    "(re-verify + refresh anchors)"
                ),
            )
        )
        options += [(label, detail) for label, detail, _, _ in registered]
    picked = io.choice(
        "which model?",
        options + [("back", "")],
        default=1,
        back=True,
    )
    if picked == 0 or picked == len(rows) + 1:
        return
    _, _, mesh_key, model_id = rows[picked - 1]
    if not model_id:
        # "deploy in progress" row: informational, never a second deploy.
        return
    flows._offer_deploy(io, pool_dir, mesh_key=mesh_key, model_id=model_id)


def _machines(io: WizardIO, pool_dir: Path, view: dict[str, Any]) -> None:
    action = io.choice(
        "machines",
        [
            ("add a machine", "print the worker join one-liner"),
            ("restart idle workers", "PM2 restart; busy workers are left alone"),
            ("remove an offline worker", "delete a dead worker's record"),
            ("back", ""),
        ],
        default=1,
        back=True,
    )
    if action == 0:
        return
    if action == 1:
        argv = ["pool", "join-token", "--pool", str(pool_dir)]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        code, output = flows._run_cli(argv)
        io.out(output.rstrip())
        return
    if action == 2:
        busy = {
            worker_id: str(worker.get("status", "") or "")
            for worker_id, worker in (view.get("workers") or {}).items()
            if str(worker.get("status", "") or "") not in ("", "idle")
        }
        for worker_id, status in sorted(busy.items()):
            io.out(
                render.dim(f"  {worker_id}: left alone ({status})")
            )
        argv = ["start", "--all"]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        code, output = flows._run_cli(argv)
        io.out(output.rstrip() or render.ok("idle units restarted"))
        return
    if action == 3:
        offline = sorted(
            worker_id
            for worker_id, worker in (view.get("workers") or {}).items()
            if worker.get("stale")
        )
        if not offline:
            io.out(
                render.warn(
                    "no offline workers (live workers cannot be removed; "
                    "stop their process first)"
                )
            )
            return
        pick = io.choice(
            "remove which worker?",
            [(worker_id, "offline") for worker_id in offline],
            default=1,
            back=True,
        )
        if pick == 0:
            return
        worker_id = offline[pick - 1]
        if not io.confirm(f"remove {worker_id}?", default=False):
            return
        argv = [
            "pool",
            "remove-worker",
            "--worker-id",
            worker_id,
            "--pool",
            str(pool_dir),
        ]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        code, output = flows._run_cli(argv)
        io.out(output.rstrip())


def _api_keys(io: WizardIO, pool_dir: Path) -> None:
    """Private OpenAI API keys: this pool's meshes for the operator's own
    tools, independent of subnet registration."""

    from verallm.mesh import onboarding

    state = flows._pool_state(pool_dir)
    endpoint_hint = str(
        state.get("manager_endpoint", "") or "http://127.0.0.1:9500"
    )
    api_tls_port = int(state.get("api_tls_port", 0) or 0)
    io.out("")
    io.out(
        render.dim(
            "  the private API serves THIS pool's meshes to your own "
            "tools (OpenAI compatible, proofs always on, rate limited)"
        )
    )
    if api_tls_port:
        public_host = onboarding.detect_public_ip() or "<public-ip>"
        io.out(
            render.ok(
                f"  internet access ON: https://{public_host}:{api_tls_port}"
                "/v1/chat/completions and /v1/models"
            )
        )
        fingerprint = onboarding.api_tls_cert_fingerprint(pool_dir)
        if fingerprint:
            io.out(
                render.dim(
                    f"  certificate SHA256 fingerprint: {fingerprint}"
                )
            )
    else:
        io.out(
            render.dim(
                f"  local: {endpoint_hint}/v1/chat/completions and "
                "/v1/models (this machine only)"
            )
        )
    listing = onboarding.pool_management_request(
        pool_dir, "/v1/pool/api-keys", {"action": "list"}
    )
    keys = (listing or {}).get("keys") or []
    if keys:
        for row in keys:
            flag = " (disabled)" if row.get("disabled") else ""
            io.out(
                f"    {row.get('id')}  {row.get('name') or '(unnamed)'}"
                f"{flag}"
            )
    else:
        io.out(render.dim("    no keys yet"))
    action = io.choice(
        "api keys",
        [
            ("create a key", "printed once; store it in your client"),
            ("revoke a key", "by its id from the list above"),
            (
                (
                    "internet access (change port / disable)"
                    if api_tls_port
                    else "internet access (enable https)"
                ),
                "own https listener + live self-test; workers unaffected",
            ),
            ("back", ""),
        ],
        default=4,
    )
    if action == 1:
        name = io.text("key label (optional)", "")
        payload = onboarding.pool_management_request(
            pool_dir, "/v1/pool/api-keys", {"action": "create", "name": name}
        )
        if payload and payload.get("api_key"):
            io.out(render.ok("API key minted (shown ONCE, store it now):"))
            io.out(f"    {payload['api_key']}")
        else:
            io.out(render.warn("the manager did not answer; is it running?"))
    elif action == 2:
        key_id = io.text("key id to revoke")
        payload = onboarding.pool_management_request(
            pool_dir,
            "/v1/pool/api-keys",
            {"action": "revoke", "key_id": key_id},
        )
        if payload and payload.get("revoked"):
            io.out(render.ok("revoked"))
        else:
            io.out(render.warn("no key matched that id"))
    elif action == 3:
        io.out(
            render.dim(
                "  the manager opens a second https listener for the API "
                "and dashboard; workers keep their plain-http port. On "
                "rented boxes use a provider-mapped port. 0 disables."
            )
        )
        raw = io.text(
            "public https port", str(api_tls_port or 9543)
        ).strip()
        try:
            port = int(raw)
        except ValueError:
            io.out(render.warn("the port must be a number"))
            return
        # The CLI stays the reference implementation; the board drives the
        # exact same verb in-process (restart, health poll, live
        # self-test, fingerprint, test command).
        import argparse as argparse_module

        from verallm.mesh import cli as mesh_cli

        try:
            mesh_cli.cmd_apikey(
                argparse_module.Namespace(
                    action="expose",
                    port=port,
                    name="",
                    key_id="",
                    pool=str(pool_dir),
                    pool_token=None,
                    pool_token_file=None,
                    manager_ca_file="",
                    timeout=30.0,
                )
            )
        except SystemExit as exc:
            io.out(render.warn(str(exc)))


def _ask_subnet_binding(io: WizardIO, pool_dir: Path):
    """Collect the chain binding for an in-place upgrade (no host/port/root
    questions: the pool and its manager endpoint already exist)."""

    from verallm.mesh import onboarding

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
    opts = onboarding.MeshCoordinatorOptions(
        serving_mode="subnet",
        skip_install=True,
        non_interactive=False,
    )
    opts.chain_id = binding["chain_id"]
    opts.netuid = binding["netuid"]
    opts.chain_config = str(binding.get("chain_config", "") or "")
    io.out(
        render.ok(
            f"{binding['network']}: chain id {binding['chain_id']}, "
            f"netuid {binding['netuid']}"
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
    uid_raw = io.text("your miner UID on the subnet")
    epoch_raw = io.text("current epoch (btcli subnets show)")
    try:
        opts.coordinator_uid = int(uid_raw)
        opts.epoch = int(epoch_raw)
    except ValueError:
        io.out(render.warn("UID and epoch must be integers"))
        return None
    opts.worker_wallet_name = io.text(
        "worker signing wallet name (for locally joined GPUs)"
    )
    opts.worker_wallet_hotkey = (
        io.text("worker wallet hotkey", "default") or "default"
    )
    opts.validator_allowlist_path = io.text(
        "validator allowlist file path",
        str(Path.home() / ".verathos" / "validators.json"),
    )
    state = flows._pool_state(pool_dir)
    endpoint = str(state.get("manager_endpoint", "") or "")
    try:
        from urllib.parse import urlparse

        opts.manager_port = int(urlparse(endpoint).port or 9500)
    except (TypeError, ValueError):
        opts.manager_port = 9500
    return opts


def _local_gpu_group_spec() -> str:
    """The --gpus spec matching this machine's EXISTING worker topology.

    Rejoining after an upgrade must not silently reshape workers (a
    2+2 split collapsing into one 4-GPU worker changes ids, workdirs,
    and stage keys); the unit registry remembers the enrolled grouping.
    """

    from verallm.mesh import units

    try:
        registry = units.load_unit_registry()
        enrolled = units.units_from_registry(registry) if registry else []
    except Exception:
        return ""
    specs = [
        "+".join(str(i) for i in unit.gpu_indices)
        for unit in enrolled
        if unit.gpu_indices
    ]
    return ",".join(specs)


def _migrate_to_subnet(
    io: WizardIO, pool_dir: Path
) -> Path | None:
    """Bind a local-only pool to the subnet IN PLACE.

    The pool keeps its id, tokens, worker records, model registry, and
    every on-disk artifact; what changes is the chain binding in the
    state file, the manager restarting with the signing wallet, and the
    local workers rejoining with wallet + allowlist flags (rejoin binds
    each worker's stage identity, which subnet auth pins). Returns the
    pool dir, or None when the operator backs out.
    """

    from verallm.mesh import onboarding
    from verallm.mesh.pool import upgrade_pool_state_to_subnet

    io.out("")
    io.out(
        render.warn(
            "going on the subnet stops the running meshes and restarts "
            "the manager and this machine's workers with your signing "
            "wallet."
        )
    )
    io.out(
        render.dim(
            "  the pool keeps its id, tokens, workers, models, and proof "
            "caches; nothing re-downloads."
        )
    )
    if not io.confirm("bind this pool to the subnet now?", default=True):
        return None

    opts = _ask_subnet_binding(io, pool_dir)
    if opts is None:
        return None
    endpoint = str(
        flows._pool_state(pool_dir).get("manager_endpoint", "") or ""
    )
    try:
        onboarding.validator_preflight(opts, endpoint)
    except SystemExit as exc:
        io.out(render.warn(str(exc)))
        io.out(
            render.dim(
                "  nothing was changed; fix the endpoint (TLS) and retry"
            )
        )
        return None

    view = flows._pool_view(pool_dir)
    for mesh_key in sorted(view.get("meshes") or {}):
        argv = [
            "pool",
            "stop",
            "--mesh-key",
            mesh_key,
            "--pool",
            str(pool_dir),
        ]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        flows._run_cli(argv)
    if view.get("meshes"):
        deadline = time.monotonic() + 120
        with render.spinner("stopping the running meshes..."):
            while time.monotonic() < deadline:
                if not flows._pool_view(pool_dir).get("meshes"):
                    break
                time.sleep(3)

    # The live manager persists its in-memory (dev) state lazily and
    # would clobber the upgraded file; take it down BEFORE writing.
    subprocess.run(
        ["pm2", "delete", onboarding.POOL_MANAGER_PM2_NAME],
        capture_output=True,
    )
    try:
        _dir, token, endpoint_changed = upgrade_pool_state_to_subnet(
            pool_dir,
            owner_account=opts.owner_account,
            coordinator_address=opts.coordinator_address,
            validator_shared_state_path=opts.validator_shared_state,
            chain_id=opts.chain_id,
            netuid=opts.netuid,
            coordinator_uid=opts.coordinator_uid,
            epoch=opts.epoch,
        )
    except ValueError as exc:
        io.out(render.warn(f"upgrade refused: {exc}"))
        io.out(
            render.dim(
                "  restart the manager with: verathos mesh start --all"
            )
        )
        return None
    onboarding.start_manager_for_pool(pool_dir, opts)
    opts.join_gpu_groups = opts.join_gpu_groups or _local_gpu_group_spec()
    passthrough = onboarding.worker_passthrough(opts)
    if opts.join_gpu_groups:
        passthrough += ["--gpus", opts.join_gpu_groups]
    io.out("  rejoining this machine's workers with the signing wallet...")
    try:
        onboarding.setup_mesh_worker(token.encode(), passthrough)
    except SystemExit as exc:
        io.out(
            render.warn(
                f"local worker rejoin failed ({exc}); rejoin by hand with "
                "the join one-liner (verathos mesh pool join-token)"
            )
        )
    if endpoint_changed:
        io.out(
            render.warn(
                "the manager endpoint changed: remote workers need the "
                "re-printed token (verathos mesh pool join-token)"
            )
        )
    io.out(
        render.ok(
            f"pool {pool_dir.name} is live on the subnet (same pool, "
            "same tokens)"
        )
    )
    flows._offer_deploy(io, pool_dir)
    return pool_dir


def run_mesh_manage(args: Any, io: WizardIO | None = None) -> int:
    io = io or WizardIO()
    pool_dir = Path(
        str(getattr(args, "pool", "") or "") or _discover_pool_dir()
    ).expanduser()
    if not (pool_dir / "pool-state.json").is_file():
        raise SystemExit(
            f"no pool state under {pool_dir}; run `verathos mesh setup` "
            "first, or pass --pool DIR"
        )
    from verallm import __version__

    io.out("")
    for line in render.brand_banner(
        [
            "VERATHOS · mesh manage",
            "sleipnir mesh inference",
            "gleipnir proof protocol",
            f"v{__version__} · pool {pool_dir.name}",
        ]
    ):
        io.out(line)
    # The FULL board (catalog + latency map) renders at entry and on the
    # explicit redraw. Every return to the menu shows the compact
    # overview instead: machines + meshes, the state a completed action
    # just changed. (One line was too little - operators re-ran the full
    # redraw after every launch just to see the mesh row; three panels
    # were too much noise after a join token.)
    render_board = True
    refresh_catalog = False
    while True:
        if render_board:
            view = _board(io, pool_dir, refresh_catalog=refresh_catalog)
        else:
            view = _overview(io, pool_dir)
        render_board = False
        refresh_catalog = False
        _registrations = flows.pool_registrations(view)
        _golive = (
            (
                "subnet registration",
                f"{len(_registrations)} registered · register more / "
                "re-deploy / retire",
            )
            if _registrations
            else ("go live on the subnet", "probe gate + on-chain registration")
        )
        action = io.choice(
            "what do you want to do?",
            [
                ("launch a model", "catalog → placement → verified probe"),
                ("launch progress", "follow a forming mesh live; b or Ctrl-C detaches"),
                ("stop a mesh", "frees its GPUs; downloads resume later"),
                ("switch model", "stop a mesh, then launch another"),
                ("chat", "verified chat with a serving mesh"),
                ("probe", "one proof-verified test chat"),
                ("machines", "add / restart / remove workers"),
                _golive,
                (
                    "api keys",
                    "private OpenAI API for this pool (create/list/revoke)",
                ),
                ("board", "redraw machines/meshes/catalog (from cache; the chain list refreshes itself in the background)"),
                ("quit", ""),
            ],
            default=10,
        )
        # State-changing actions fall through to the compact overview
        # (machines + meshes) the loop prints on every menu return; only
        # the explicit redraw and a pool migration re-render the FULL
        # board with the catalog.
        if action == 1:
            flows.launch_flow(io, pool_dir, show_panel=False)
        elif action == 2:
            _watch_launch(io, pool_dir, view)
        elif action == 3:
            _stop_mesh(io, pool_dir, view)
        elif action == 4:
            if _stop_mesh(io, pool_dir, view):
                flows.launch_flow(io, pool_dir, show_panel=False)
        elif action == 5:
            _chat(io, pool_dir, view)
        elif action == 6:
            _probe_mesh(io, pool_dir, view)
        elif action == 7:
            _machines(io, pool_dir, view)
        elif action == 8:
            if flows.pool_serving_mode(pool_dir) == "subnet":
                _subnet_registration(io, pool_dir, view)
            else:
                migrated = _migrate_to_subnet(io, pool_dir)
                if migrated is not None:
                    pool_dir = migrated
                    render_board = True
        elif action == 9:
            _api_keys(io, pool_dir)
        elif action == 10:
            # Cache-only redraw: the forced inline chain re-read hung the
            # board for minutes on a congested node;
            # freshness now comes from the catalog's own background
            # refresh (stale-while-revalidate).
            render_board = True
        elif action == 11:
            return 0


def _discover_pool_dir() -> str:
    """The single local pool, mirroring the CLI's zero-flag discovery.

    Several pools on a terminal get a numbered picker (newest first, Enter
    picks it) instead of an error a human then has to retype as --pool."""

    import json as json_module
    import sys

    from verallm.mesh.pool import known_pool_dirs

    candidates = [
        pool_dir
        for pool_dir in known_pool_dirs()
        if (pool_dir / "pool-state.json").is_file()
    ]
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise SystemExit(
            "no pool on this machine; run `verathos mesh setup` first, or "
            "pass --pool DIR"
        )
    candidates.sort(
        key=lambda pool_dir: (pool_dir / "pool-state.json").stat().st_mtime,
        reverse=True,
    )
    if sys.stdin.isatty() and sys.stdout.isatty():
        print("several pools on this machine:")
        for index, pool_dir in enumerate(candidates, start=1):
            mode = ""
            try:
                state = json_module.loads(
                    (pool_dir / "pool-state.json").read_text()
                )
                mode = str(state.get("serving_mode", "") or "")
                if mode == "validator":  # pre-rename state files
                    mode = "subnet"
            except (OSError, ValueError):
                pass
            print(
                f"  {index}. {pool_dir.name}"
                + (f"  [{mode}]" if mode else "")
                + f"  ({pool_dir})"
            )
        while True:
            raw = input(
                f"select [1-{len(candidates)}, Enter=1 (newest)]: "
            ).strip()
            if not raw:
                raw = "1"
            if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                return str(candidates[int(raw) - 1])
            print("  pick a number from the list")
    listing = "\n  ".join(str(pool_dir) for pool_dir in candidates)
    raise SystemExit(
        "several pools on this machine; pick one with --pool DIR:\n  "
        + listing
    )
