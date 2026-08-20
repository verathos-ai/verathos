"""Interactive operator flows shared by ``mesh setup`` and ``mesh manage``.

Everything here needs only a pool directory and a :class:`WizardIO`; no
setup state, no coordinator options. The setup wizard uses these flows
for its first-launch quickstart, the manage board for every day after
that, and both print the equivalent ``$ verathos mesh ...`` command
before acting so the flag lane stays discoverable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from verallm.mesh import onboarding, render


@dataclass
class WizardIO:
    """Injectable I/O so every flow is testable without a TTY."""

    ask: Callable[[str], str] = input
    out: Callable[[str], None] = print
    answers: list[str] = field(default_factory=list)

    def choice(
        self,
        prompt: str,
        options: list[tuple[str, str]],
        default: int = 1,
        back: bool = False,
    ) -> int:
        """Numbered pick; with ``back=True``, b/back returns 0.

        An option with an EMPTY label is a section header: its detail is
        printed as-is (pre-styled by the caller), it gets no number and
        cannot be picked. The returned number counts selectable options
        only, in display order.
        """

        self.out("")
        number = 0
        for label, detail in options:
            if not label:
                self.out(detail)
                continue
            number += 1
            marker = render.dim(f"  {number}) ") + render.bold(label)
            styled_detail = (
                detail if "\x1b" in detail else render.dim(detail)
            )
            self.out(marker + ("  " + styled_detail if detail else ""))
        hint = f"[1-{number}, Enter={default}"
        hint += ", b=back]: " if back else "]: "
        while True:
            raw = self.ask(render.bold(f"{prompt} {hint}")).strip().lower()
            if not raw:
                return default
            if back and raw in ("b", "back"):
                return 0
            if raw.isdigit() and 1 <= int(raw) <= number:
                return int(raw)
            self.out(render.warn("pick a number from the list"))

    def text(self, prompt: str, default: str = "") -> str:
        suffix = render.dim(f"[{default}]: ") if default else ": "
        raw = self.ask(render.bold(f"{prompt} ") + suffix).strip()
        return raw or default

    def confirm(self, prompt: str, default: bool = True) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        raw = self.ask(render.bold(f"{prompt} {hint}: ")).strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")


def _run_cli(args: list[str]) -> tuple[int, str]:
    """Run a verathos mesh CLI subcommand, streaming nothing, capturing all."""

    env = dict(os.environ)
    if render.use_style():
        # The child's stdout is a pipe, but ITS output lands on THIS
        # terminal: forward the styling decision or every board-invoked
        # command prints colorless.
        env.setdefault("CLICOLOR_FORCE", "1")
    completed = subprocess.run(
        [sys.executable, "-m", "verallm.mesh.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.returncode, (completed.stdout or "") + (
        completed.stderr or ""
    )


def _pool_state(pool_dir: Path) -> dict[str, Any]:
    try:
        return json.loads(
            (pool_dir / "pool-state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def pool_serving_mode(pool_dir: Path) -> str:
    """Normalized serving mode from the state file: "subnet", "dev", or "".

    Pools minted before the mode rename persist "validator"; every branch
    on the mode must read it through here (or pool.is_subnet_serving_mode)
    instead of comparing raw literals.
    """

    from verallm.mesh.pool import is_subnet_serving_mode

    mode = str(_pool_state(pool_dir).get("serving_mode", "") or "")
    if is_subnet_serving_mode(mode):
        return "subnet"
    return mode.strip().lower()


def pool_mode_label(pool_dir: Path) -> str:
    """Human label for the pool's serving mode ("" when unknown)."""

    return {"dev": "local only", "subnet": "on the subnet"}.get(
        pool_serving_mode(pool_dir), ""
    )


def _pool_view(pool_dir: Path) -> dict[str, Any]:
    """The pool as it actually is right now.

    The LIVE manager is the only honest source: the state file is written
    lazily and survives manager restarts, so reading it directly once told
    an operator "a mesh is already serving" about a mesh that was long
    gone. Disk is only a fallback for a manager that is down (in which
    case nothing can launch anyway).
    """

    def _catalog_model_ids(workers: dict[str, Any]) -> set[str]:
        # A model a worker holds locally (catalog entry, no download
        # source) is just as launchable as a registry one: recommend()
        # places it on the workers that have the file.
        ids: set[str] = set()
        for worker in workers.values():
            for item in worker.get("catalog") or []:
                if item.get("model_id"):
                    ids.add(str(item["model_id"]))
        return ids

    def _subnet_fields(source: dict[str, Any]) -> dict[str, Any]:
        # Identity + registration surface for the board panels: which
        # network/hotkey this pool serves as and what is registered
        # on-chain. Missing on pools started before the identity persist;
        # panels render nothing for empty values.
        from verallm.mesh.pool import is_subnet_serving_mode

        raw_mode = str(source.get("serving_mode", "") or "")
        mode = "subnet" if is_subnet_serving_mode(raw_mode) else (
            raw_mode.strip().lower()
        )
        return {
            "serving_mode": mode,
            "subtensor_network": str(
                source.get("subtensor_network", "") or ""
            ),
            "wallet_name": str(source.get("wallet_name", "") or ""),
            "wallet_hotkey": str(source.get("wallet_hotkey", "") or ""),
            "coordinator_hotkey_ss58": str(
                source.get("coordinator_hotkey_ss58", "") or ""
            ),
            "validator_binding": dict(
                source.get("validator_binding") or {}
            ),
            "mesh_registration": dict(
                source.get("mesh_registration") or {}
            ),
            "mesh_registrations": dict(
                source.get("mesh_registrations") or {}
            ),
            "api_tls_port": int(source.get("api_tls_port", 0) or 0),
        }

    live = onboarding.pool_management_status(pool_dir)
    if live is not None:
        workers = dict(live.get("workers") or {})
        models_detail = dict(live.get("models") or {})
        return {
            "live": True,
            "models": sorted(
                set(models_detail.keys()) | _catalog_model_ids(workers)
            ),
            "models_detail": models_detail,
            "meshes": dict(live.get("meshes") or {}),
            "workers": workers,
            "deploys": dict(live.get("deploys") or {}),
            **_subnet_fields(live),
        }
    state = _pool_state(pool_dir)
    workers = dict(state.get("workers") or {})
    models_detail = dict(state.get("model_registry") or {})
    return {
        "live": False,
        "models": sorted(
            set(models_detail.keys()) | _catalog_model_ids(workers)
        ),
        "models_detail": models_detail,
        "meshes": dict(state.get("meshes") or {}),
        "workers": workers,
        **_subnet_fields(state),
    }


def pool_operator_score(pool_dir: Path) -> dict[str, Any]:
    """The manager's cached on-chain score view ({} when unavailable).

    The manager caches the metagraph read ~5 min, so board redraws stay
    cheap; a manager that is down or chain-blind returns {} and panels
    render the local fleet view only. The short timeout keeps the FIRST
    render snappy: the manager finishes the metagraph read and fills its
    cache even after this client gives up, so the next redraw has it.
    """

    payload = onboarding.pool_management_request(
        pool_dir, "/v1/operator/score", timeout=3.0
    )
    if not payload or payload.get("status") != "ok":
        return {}
    return {k: v for k, v in payload.items() if k != "status"}


def _pool_models(pool_dir: Path) -> list[str]:
    return list(_pool_view(pool_dir)["models"])


def _serving_mesh(pool_dir: Path) -> str:
    """The mesh key that is serving AND actually routable, else ''.

    A "serving" row whose workers stopped heartbeating is dead in
    practice; claiming it works sends the operator to a chat that 400s.
    Without a live manager NOTHING is claimable: chat, launch, and probe
    all go through the manager, so a disk-only "serving" row is
    unactionable even when the processes happen to be up.
    """

    view = _pool_view(pool_dir)
    if not view["live"]:
        return ""
    for mesh_key, mesh in view["meshes"].items():
        if str(mesh.get("status", "")) != "serving":
            continue
        if not bool(mesh.get("routing_ready", False)):
            continue
        return str(mesh_key)
    return ""


def _network_binding_for_pool(pool_dir: Path) -> dict[str, Any] | None:
    """Match the pool's stored chain binding back to a named network."""

    binding = _pool_state(pool_dir).get("validator_binding") or {}
    chain_id = int(binding.get("chain_id", 0) or 0)
    for name in onboarding.MESH_NETWORKS:
        try:
            resolved = onboarding.resolve_network(name)
        except SystemExit:
            continue
        if resolved and resolved["chain_id"] == chain_id:
            return resolved
    return None


def pool_network(pool_dir: Path) -> str:
    """The network whose catalog this pool browses (testnet for dev)."""

    binding = _network_binding_for_pool(pool_dir)
    return binding["network"] if binding else "testnet"


def _pick_model_from_catalog(
    io: WizardIO, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """One table of what this pool can serve; the operator just picks.

    Where a model comes from (disk, download, already in the pool) is an
    annotation, never a question — resolving the file is the tool's job
    after the pick.
    """

    from verallm.mesh import panels

    io.out("")
    io.out(render.section("  what can this pool serve?"))
    lines, launchable = panels.catalog_table(candidates)
    for line in lines:
        io.out(line)
    if not launchable:
        io.out(render.warn("nothing launchable right now"))
        return None
    # The table above is the broad overview (everything this pool could
    # ever serve). The CHOICE runs over what fits the machines that are
    # idle RIGHT NOW: picking a model whose capacity is fully occupied
    # only produced a placement failure after the fact.
    now = [
        cand
        for cand in launchable
        # A mesh already serving or forming for the model is not a new
        # launch (switch/stop flows own those). An ABSENT fits_now is
        # "unannotated", not "busy": only an explicit False may drop a
        # launchable model here (a missing key silently skipped every
        # launch for callers that never stamp the annotation).
        if cand.get("fits_now", True)
        and not str(cand.get("mesh_status", "") or "")
    ]
    if not now:
        io.out("")
        io.out(
            render.warn(
                "every machine with fitting capacity is busy serving; "
                "stop a mesh (option 3) to free GPUs, or add a machine "
                "(option 7)"
            )
        )
        return None
    if len(now) != len(launchable):
        io.out("")
        io.out(render.section("  launchable right now (on idle machines)"))
        for number, cand in enumerate(now, start=1):
            io.out(
                render.dim(f"  {number}) ")
                + render.bold(str(cand["model_id"]))
                + "  "
                + render.dim(str(cand.get("fit_now", "") or ""))
            )
    while True:
        raw = io.text(
            f"which model? [1-{len(now)}, Enter=1, b=back]"
        ).strip().lower()
        if raw in ("b", "back", "s", "skip", "q"):
            return None
        if not raw:
            return now[0]
        if raw.isdigit() and 1 <= int(raw) <= len(now):
            return now[int(raw) - 1]
        io.out(render.warn("pick a number from the list"))


def _prepare_model(
    io: WizardIO, pool_dir: Path, cand: dict[str, Any]
) -> bool:
    """Make the picked model servable; all plumbing, no questions."""

    if cand.get("in_pool"):
        return True
    if cand.get("disk_files") and cand.get("store_manifest"):
        # A local copy that matched the owner's published manifest wires
        # in without any build. Operators never build manifests: a local
        # file WITHOUT a published manifest never reaches this flow (the
        # catalog only lists registered models), and onboarding new
        # models is the subnet owner's process.
        first = str(cand["disk_files"][0])
        argv = [
            "add-model",
            first,
            "--model-id",
            cand["model_id"],
            "--manifest",
            str(cand["store_manifest"]),
        ]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        with render.spinner(
            f"preparing {cand['model_id']} from the local file..."
        ):
            code, output = _run_cli(argv)
        if code != 0:
            io.out(render.fail(f"prepare failed:\n{output.strip()}"))
            return False
        deadline = time.monotonic() + 120
        with render.spinner("workers are picking the model up..."):
            while time.monotonic() < deadline:
                if cand["model_id"] in _pool_models(pool_dir):
                    return True
                time.sleep(3)
        io.out(
            render.warn(
                "workers did not pick the model up within 2 minutes; "
                "check verathos mesh status"
            )
        )
        return False
    if cand.get("hf_repo") and cand.get("hf_files"):
        # Catalogue-sourced download: teach the pool the source so the
        # driver auto-fetches at launch. Metadata only, instant.
        argv = [
            "pool",
            "add-model-source",
            "--pool",
            str(pool_dir),
            "--model-id",
            cand["model_id"],
            "--hf-repo",
            str(cand["hf_repo"]),
            "--hf-files",
            ",".join(str(f) for f in cand["hf_files"]),
            "--model-bytes",
            str(int(cand.get("model_bytes", 0) or 0)),
            "--layers",
            str(int(cand.get("layers", 0) or 0)),
        ]
        # Chain anchors + store URLs let the DRIVER pull the owner's
        # published, root-verified manifest instead of rebuilding it
        # locally (minutes of hashing per launch, and the parallel
        # builder is the fragile step that once killed a 27B fetch).
        root = str(cand.get("tensor_manifest_root", "") or "")
        package = str(cand.get("model_package_hash", "") or "")
        tokenizer = str(cand.get("tokenizer_hash", "") or "")
        if root and package and tokenizer:
            argv += [
                "--model-tensor-manifest-root",
                root,
                "--model-package-hash",
                package,
                "--tokenizer-hash",
                tokenizer,
                "--quantization-scheme",
                str(cand.get("quant", "") or "gguf"),
            ]
        manifest_urls = [str(u) for u in cand.get("manifest_urls") or ()]
        if manifest_urls:
            argv += ["--manifest-urls", ",".join(manifest_urls)]
        io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
        code, output = _run_cli(argv)
        if code != 0:
            io.out(render.fail(f"prepare failed:\n{output.strip()}"))
            return False
        return True
    return cand["model_id"] in _pool_models(pool_dir)


def _pick_placement(
    io: WizardIO, pool_dir: Path, model_id: str
) -> dict[str, Any] | None:
    """The FULL placement recommendation (VRAM, RTT, link class), tabled."""

    from verallm.mesh import panels

    with render.spinner(f"measuring placement options for {model_id}..."):
        advice = onboarding.pool_recommend(pool_dir, model_id)
    view = _pool_view(pool_dir)
    io.out("")
    io.out(render.section(f"  recommended placement for {model_id}:"))
    lines, suggestions = panels.placement_table(
        model_id, advice, view["workers"]
    )
    for line in lines:
        io.out(line)
    if not suggestions:
        io.out(render.warn("no viable placement right now"))
        return None
    while True:
        raw = io.text(
            f"which placement? [1-{len(suggestions)}, Enter=1, b=back]"
        ).strip().lower()
        if raw in ("b", "back"):
            return "back"
        if not raw:
            return suggestions[0]
        if raw.isdigit() and 1 <= int(raw) <= len(suggestions):
            return suggestions[int(raw) - 1]
        io.out(render.warn("pick a number from the table"))


def wait_for_mesh(
    io: WizardIO,
    pool_dir: Path,
    mesh_key: str,
    model_id: str,
    *,
    deadline_s: float = 600.0,
) -> bool | None:
    """Watch ONE mesh form, narrating its members' live progress.

    Returns True when the mesh serves, False on the deadline, and None
    when the operator detaches with ``b`` + Enter (the launch keeps
    going pool-side either way). Ctrl-C also detaches: it explains that
    the launch continues (downloads resume; nothing to clean up) and
    re-raises.
    """

    import select

    deadline = time.monotonic() + deadline_s
    interactive = False
    try:
        interactive = sys.stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if interactive:
        io.out(
            render.dim(
                "  b + Enter detaches (the launch keeps going); Ctrl-C "
                "does the same"
            )
        )

    def _ready(view: dict[str, Any]) -> bool:
        if mesh_key:
            mesh = view["meshes"].get(mesh_key) or {}
            if str(mesh.get("status", "")) != "serving":
                return False
            return not view["live"] or bool(mesh.get("routing_ready"))
        return bool(_serving_mesh(pool_dir))

    def _wait_slice(seconds: float) -> bool:
        """Sleep in stdin-watching slices; True = operator detached."""

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not interactive:
                time.sleep(0.25)
                continue
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.25)
            except (OSError, ValueError):
                time.sleep(0.25)
                continue
            if readable:
                line = sys.stdin.readline().strip().lower()
                if line in ("b", "q", "back"):
                    return True
        return False

    try:
        with render.spinner(f"launching {model_id}...") as spin:
            while time.monotonic() < deadline:
                view = _pool_view(pool_dir)
                if _ready(view):
                    return True
                mesh_state = str(
                    (view["meshes"].get(mesh_key) or {}).get("status", "")
                )
                if mesh_state == "fetching":
                    # Downloads are disk-fit-prechecked and report live
                    # progress; they must not burn the serving deadline
                    # (a 17GB model on a home line is many minutes).
                    deadline = max(
                        deadline, time.monotonic() + deadline_s
                    )
                # Workers report REAL progress in their status strings
                # ("fetching 45%", "driving (warming proofs 30%)");
                # surface them live — but only THIS mesh's members.
                # Another mesh forming in parallel must not leak its
                # download percent into this launch line.
                members = set(
                    (view["meshes"].get(mesh_key) or {}).get("members") or []
                )
                states = sorted(
                    {
                        str(worker.get("status", "") or "")
                        for worker_id, worker in view["workers"].items()
                        if worker.get("status")
                        and (not members or worker_id in members)
                    }
                )
                if states:
                    spin.update(
                        f"launching {model_id}...  [{', '.join(states)}]"
                    )
                if _wait_slice(5):
                    return None
    except KeyboardInterrupt:
        # Aborting the WIZARD does not abort the LAUNCH: the mesh keeps
        # forming pool-side (downloads resume; nothing to clean up).
        io.out("")
        io.out(
            render.warn(
                "the launch continues on the pool; this only stopped the "
                "wizard's view of it."
            )
        )
        if mesh_key:
            io.out(
                render.dim(
                    "  didn't want it? "
                    f"verathos mesh pool stop --mesh-key {mesh_key} "
                    f"--pool {pool_dir}"
                )
            )
        io.out(
            render.dim(
                "  watch it: verathos mesh fleet   (partial downloads "
                "resume on the next launch)"
            )
        )
        raise
    return False


def run_verified_probe(io: WizardIO, pool_dir: Path, mesh_key: str) -> bool:
    """One proof-verified chat through the mesh; prints the result line."""

    probe_argv = ["pool", "probe", "--pool", str(pool_dir)]
    if mesh_key:
        # Pin the probe to the mesh WE launched: with two meshes serving,
        # auto-selection would refuse or pick the wrong one.
        probe_argv += ["--mesh-key", mesh_key]
    io.out(render.dim("  $ verathos mesh " + " ".join(probe_argv)))
    with render.spinner("proving one real chat end to end..."):
        code, output = _run_cli(probe_argv)
    payload: dict[str, Any] = {}
    try:
        start = output.index("{")
        payload = json.loads(output[start:])
    except ValueError:
        pass
    if code == 0 and payload.get("verified"):
        io.out(
            render.ok(
                f"verified end to end: {payload.get('proof_stages')} proof "
                f"stage(s), {payload.get('receipts')} receipt(s), "
                f"{payload.get('engine_tps')} tok/s engine, "
                f"ttft {payload.get('ttft_s')}s"
            )
        )
        return True
    io.out(render.fail(f"probe did not verify:\n{output.strip()}"))
    return False


def launch_and_verify(
    io: WizardIO,
    pool_dir: Path,
    model_id: str,
    placement: dict[str, Any],
) -> bool:
    """`pool launch` with the chosen placement, wait, verified probe."""

    launch_argv = [
        "pool",
        "launch",
        "--pool",
        str(pool_dir),
        "--model-id",
        model_id,
    ]
    workers = [str(w) for w in placement.get("workers") or []]
    driver = str(placement.get("driver", "") or "")
    if workers:
        launch_argv += ["--workers", ",".join(workers)]
    if driver:
        launch_argv += ["--driver", driver]
    # Return at SPAWN: wait_for_mesh below narrates the live formation.
    # Without --no-wait the child CLI itself blocks until routing-ready,
    # so the "mesh launching" tick appeared only after the whole launch
    # had finished, leaving the screen apparently frozen for minutes.
    launch_argv += ["--no-wait"]
    io.out(render.dim("  $ verathos mesh " + " ".join(launch_argv)))
    with render.spinner("requesting the launch..."):
        code, output = _run_cli(launch_argv)
    if code != 0:
        io.out(render.fail(f"launch failed:\n{output.strip()}"))
        return False
    mesh_key = ""
    try:
        payload = json.loads(output[output.index("{"):])
        mesh_key = str(payload.get("mesh_key", "") or "")
    except (ValueError, KeyError):
        pass
    io.out(render.ok("mesh launching (workers may download the model first)"))
    ready = wait_for_mesh(io, pool_dir, mesh_key, model_id)
    if ready is None:
        io.out(
            render.dim(
                "  detached; the launch keeps going (watch it again from "
                "the manage menu: launch progress)"
            )
        )
        return False
    if not ready:
        io.out(
            render.warn(
                "the mesh has not reached serving within 10 minutes; check "
                f"verathos mesh fleet --pool {pool_dir}"
            )
        )
        return False
    resolved_key = mesh_key or _serving_mesh(pool_dir)
    io.out(
        render.ok(
            f"mesh {resolved_key} is serving; running the verified probe"
        )
    )
    return run_verified_probe(io, pool_dir, resolved_key)


def launch_flow(
    io: WizardIO,
    pool_dir: Path,
    *,
    network: str = "",
    show_panel: bool = True,
) -> str:
    """Catalog pick -> prepare -> placement -> launch -> verified probe.

    Returns "verified", "launched" (up but probe failed), "skipped", or
    "failed".
    """

    from verallm.mesh import model_catalog, panels

    network = network or pool_network(pool_dir)
    view = _pool_view(pool_dir)
    if show_panel:
        mode = pool_mode_label(pool_dir)
        io.out("")
        for line in panels.pool_panel(
            view, pool_id=pool_dir.name, mode=mode
        ):
            io.out(line)
    with render.spinner("assembling the model catalog..."):
        candidates = model_catalog.assemble_candidates(view, network=network)
    if not candidates:
        io.out(
            render.warn(
                "no registered models are reachable right now: the "
                "catalog lists what the subnet owner has registered "
                "on-chain (model onboarding is the owner's process). "
                "Check the chain connection and try again."
            )
        )
        return "skipped"
    while True:
        picked = _pick_model_from_catalog(io, candidates)
        if picked is None:
            return "skipped"
        if not _prepare_model(io, pool_dir, picked):
            return "failed"
        model_id = picked["model_id"]
        placement = _pick_placement(io, pool_dir, model_id)
        if placement == "back":
            continue
        break
    if placement is None:
        return "failed"
    if launch_and_verify(io, pool_dir, model_id, placement):
        return "verified"
    return "launched"


def pool_registrations(view: dict) -> dict[str, dict]:
    """Per-model on-chain registrations from a pool view (compat-tolerant)."""

    registrations = {
        str(model_id): dict(entry)
        for model_id, entry in (view.get("mesh_registrations") or {}).items()
        if isinstance(entry, dict)
    }
    single = view.get("mesh_registration")
    if isinstance(single, dict) and single.get("model_id"):
        registrations.setdefault(str(single["model_id"]), dict(single))
    return registrations


def _offer_deploy(
    io: WizardIO,
    pool_dir: Path,
    mesh_key: str = "",
    model_id: str = "",
) -> None:
    """The guided lane onto the subnet: probe gate + on-chain registration.

    ``verathos mesh deploy`` stays the flag-driven agent lane; this asks
    for the same inputs interactively and then runs that exact pipeline,
    streaming its output. Only subnet ("validator") pools qualify — a
    local-only pool has no chain binding to register against.
    """

    if pool_serving_mode(pool_dir) != "subnet":
        io.out("")
        io.out(
            render.dim(
                "  this pool is local-only and never earns; going on the "
                "subnet means a subnet pool (verathos mesh setup, 'create "
                "another pool', then 'on the subnet')."
            )
        )
        return
    view = _pool_view(pool_dir)
    if not mesh_key:
        mesh_key = _serving_mesh(pool_dir)
    mesh = view["meshes"].get(mesh_key) or {}
    model_id = model_id or str(mesh.get("model_id", "") or "")
    registrations = pool_registrations(view)
    registration = dict(registrations.get(model_id) or {})
    already_registered = bool(model_id and registration)
    io.out("")
    if already_registered:
        # Offering to "register" an actively registered model is
        # contradictory (observed on the manage board). A re-run is
        # still a legitimate action — deploy is idempotent (reuse-active
        # registration; relaunch only when the chain anchors changed) —
        # but it must SAY that and default to no.
        prompt = (
            f"{model_id} is already registered at index "
            f"{registration.get('index', '?')} — re-run deploy (re-verify "
            "probe gates + refresh the mesh against current chain anchors)?"
        )
        default = False
    elif model_id:
        prompt = (
            f"register {model_id} on the subnet now (probe gate + on-chain "
            "registration)?"
        )
        default = True
    else:
        prompt = (
            "run the guided subnet deploy now (launch + probe gate + "
            "on-chain registration)?"
        )
        default = True
    if not io.confirm(prompt, default=default):
        io.out(
            render.dim(
                "  later: the `verathos mesh deploy` line under next moves "
                "runs the same pipeline."
            )
        )
        return
    if not model_id:
        model_id = io.text(
            "on-chain mesh model id (from your registered ModelSpec)"
        ).strip()
        if not model_id:
            io.out(render.warn("deploy needs a model id; skipping"))
            return

    binding = _network_binding_for_pool(pool_dir)
    if binding is None:
        io.out(
            render.fail(
                "the pool's chain binding matches no shipped chain config; "
                "run `verathos mesh deploy` with an explicit --chain-config"
            )
        )
        return
    io.out(
        render.ok(
            f"network {binding['network']} (netuid {binding['netuid']}, "
            f"from {Path(binding['chain_config']).name})"
        )
    )

    driver_id = str(mesh.get("driver", "") or "")
    driver = view["workers"].get(driver_id) or {}
    driver_mesh_endpoint = str(
        (driver.get("endpoints") or {}).get("mesh", "") or ""
    )
    default_endpoint = ""
    if driver_mesh_endpoint and not onboarding.endpoint_is_loopback(
        driver_mesh_endpoint
    ):
        # Keep the ADVERTISED scheme: the driver's coordinator serves
        # plain HTTP, and fabricating https:// here fails three hard
        # probe checks unless nginx already fronts that port. Plain HTTP
        # passes the gate with an advisory warning.
        default_endpoint = driver_mesh_endpoint
    io.out(
        render.dim(
            "  validators dial the DRIVER worker's mesh endpoint, not the "
            "pool manager; front it with HTTPS (scripts/setup_https.sh) "
            "for production"
        )
    )
    endpoint = io.text(
        "public endpoint validators dial", default_endpoint
    ).strip()
    if not endpoint:
        io.out(render.warn("deploy needs a public endpoint; skipping"))
        return
    # The pool already KNOWS its miner identity (persisted at setup, shown
    # in the board header); prompting for it here made every registration
    # re-type what the manager runs with. Prompt only for a pool that
    # predates the identity persist.
    wallet = str(view.get("wallet_name", "") or "")
    hotkey = str(view.get("wallet_hotkey", "") or "")
    if wallet:
        io.out(
            render.dim(
                f"  signing as {wallet}/{hotkey or 'default'} (the pool's "
                "miner identity)"
            )
        )
    else:
        wallet = io.text(
            "miner wallet name (holds the registered hotkey)"
        ).strip()
        if not wallet:
            io.out(render.warn("deploy needs a wallet; skipping"))
            return
        hotkey = io.text("hotkey name", "default").strip()
    hotkey = hotkey or "default"

    argv = [
        "deploy",
        model_id,
        "--pool-token-file",
        str(pool_dir / "pool-admin-token.txt"),
        "--chain-config",
        binding["chain_config"],
        "--endpoint",
        endpoint,
        "--wallet",
        wallet,
        "--hotkey",
        hotkey,
    ]
    # The pool's own subtensor node serves every chain step of the deploy;
    # without it the flow crawled through the congested public RPC even
    # though the pool ran its own node.
    pool_subtensor = str(
        (_pool_view(pool_dir).get("subtensor_network") or "")
    ).strip()
    if pool_subtensor.startswith(("ws://", "wss://", "http://", "https://")):
        argv += ["--subtensor-chain-endpoint", pool_subtensor]
    io.out("")
    io.out(render.dim("  $ verathos mesh " + " ".join(argv)))
    # Stream, do not capture: deploy narrates a 9-stage pipeline and asks
    # its own confirmation before the on-chain transaction.
    completed = subprocess.run(
        [sys.executable, "-m", "verallm.mesh.cli", *argv]
    )
    if completed.returncode == 0:
        io.out(render.ok("deployed: the pool manager now renews the lease"))
    else:
        io.out(
            render.fail(
                "deploy did not complete; fix the reported stage and re-run "
                "the printed command"
            )
        )


def _print_next_moves(io: WizardIO, pool_dir: Path) -> None:
    io.out("")
    io.out(render.section("  next moves"))
    io.out(render.dim("  (zero-flag forms work on this machine; --pool pins a specific pool)"))
    io.out("")
    io.out("  manage the pool day to day (launch/stop meshes, machines, probes):")
    io.out(render.cyan("    verathos mesh manage"))
    io.out("  chat with your mesh (proof tier picker included):")
    io.out(render.cyan("    verathos mesh chat"))
    io.out("  see workers, meshes, and placement advice:")
    io.out(render.cyan("    verathos mesh fleet"))
    io.out("  placement advice for one model (add --json for scripts/agents):")
    io.out(render.cyan("    verathos mesh fleet --model-id <model>"))
    io.out("  print the worker join one-liner again:")
    io.out(render.cyan(f"    verathos mesh pool join-token --pool {pool_dir}"))
    io.out("")
    io.out(
        "  use your meshes from your own tools (OpenAI compatible, "
        "proof-verified):"
    )
    io.out(render.cyan("    verathos mesh apikey create"))
    io.out("  reach that API over the internet (own https listener + self-test):")
    io.out(render.cyan("    verathos mesh apikey expose --port 9543"))
    io.out("")
    io.out("  go live on the subnet (launch + probe gate + on-chain registration):")
    io.out(
        render.cyan(
            "    verathos mesh deploy <model-id> \\\n"
            f"      --pool-token-file {pool_dir / 'pool-admin-token.txt'} \\\n"
            "      --chain-config chain_config_testnet.json \\\n"
            "      --endpoint https://<driver-worker-host>:9443 \\\n"
            "      --wallet <wallet-name> --hotkey <hotkey-name>"
        )
    )
    io.out("")
    io.out(render.dim("  roles and flow: docs/architecture/mesh_operator_flow.md"))
    io.out("")
