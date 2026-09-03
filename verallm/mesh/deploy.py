"""`verathos mesh deploy`: measure, gate, then register a mesh on-chain.

Pipeline, in the order an honest registration requires:

1. chain preconditions (ModelSpec anchors, approved scoring profile)
2. hotkey binding (the coordinator hotkey holds a UID on the target
   netuid of the target network; testnet and mainnet behave identically)
3. pool preconditions (subnet-mode pool, coordinator address matches)
4. measurement launch: the model runs UNREGISTERED first (operator lane
   only, no signed snapshot, no chain slot), because the registered
   max_context_len is DERIVED from the launch: min(measured KV auto-fit,
   time cap from a timing probe against the validator's canary budget),
   and neither number exists before a launch
5. probe gate: light proofs, an explicit hard-tier proof, the
   full-context probe at the measured context, throughput and TTFT: the
   "everything works, ready to register" confirmation
6. EVM binding + index prediction + pool model registration (anchors
   derived from chain, context taken from the measurement, never typed)
7. relaunch chain-bound (signed snapshot binding the predicted index)
8. final posture verification (public endpoint, snapshot binding, one
   hard probe through the chain-bound mesh)
9. MinerRegistry write, index-guarded; renewal state handed to the pool

On a failed gate the mesh keeps serving UNREGISTERED and the report says
exactly why; --force overrides gate checks but never the chain
preconditions or the auth-posture reachability checks.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from verallm.chain.miner_lifecycle import LifecycleRefusal, LifecyclePlan
from verallm.mesh import registration as registration_module
from verallm.mesh.probe import (
    FULL_CONTEXT_OUTPUT_TOKENS,
    GateCheck,
    GateReport,
    ProbeGateConfig,
    build_full_context_probe_prompt,
    check_public_endpoint,
    probe_sample_from_chat_result,
    run_probe_gate,
)
from verallm.mesh.registration import (
    MeshChainAnchors,
    RegistrationOutcome,
    build_registration_target,
)

DEFAULT_LAUNCH_WAIT_S = 2700.0  # a 27B fetch plus proof prewarm is slow
DEFAULT_POLL_INTERVAL_S = 2.0
# Matches neurons/miner.py _ctx_close_enough and the pool's
# MESH_CTX_JITTER_TOLERANCE: KV auto-fit varies a little between restarts
# (VRAM fragmentation, CUDA graph memory), so an explicit context override
# is accepted within this fraction of the measured value and refused beyond.
CTX_JITTER_TOLERANCE = 0.10
# Derived-context cap knobs. Without --max-context-len the registered value
# is min(measured KV auto-fit, time-derived cap): validators canary a miner
# AT its registered maximum and that canary must verify inside the
# validator's inference budget, so a raw KV fit far beyond what the hardware
# can prefill in time guarantees canary timeouts (observed: KV auto-fit
# measured 389120 while the full-context probe at 98304 already took 541 s
# against a 540 s gate budget).
CTX_ROUND_MULTIPLE = 1024
MIN_REGISTERED_CTX = 8192
# Timing-probe context: large enough that prefill dominates the wall time,
# small enough that the probe stays cheap even on a slow mesh. When the
# first probe's linear candidate extrapolates beyond the probe point, a
# SECOND probe (capped at TIMING_PROBE_CTX_HIGH) turns the model into a
# linear+quadratic fit: prefill attention is superlinear, and a single
# small probe overestimates the servable context.
TIMING_PROBE_CTX = 32_768
TIMING_PROBE_CTX_HIGH = 65_536
DEFAULT_CONTEXT_SAFETY_MARGIN = 0.75


def derive_time_capped_context(
    *,
    measured: int,
    samples: Sequence[tuple[int, float]],
    budget_s: float,
    safety_margin: float,
) -> tuple[int, int]:
    """Derive the registered context from timing-probe samples.

    One sample extrapolates linearly (seconds per context token). Two
    samples fit t(n) = a*n + b*n^2 through both points, capturing the
    superlinear prefill attention a single small probe cannot see. A
    non-physical fit (negative curvature or negative linear term, from
    probe noise) falls back conservatively: the slowest observed
    per-token rate, or a pure-quadratic through the larger point. The
    cap is the largest context whose predicted wall time stays inside
    safety_margin * budget_s. The registered value is min(measured KV
    fit, cap), rounded DOWN to a multiple of CTX_ROUND_MULTIPLE and
    clamped to at least MIN_REGISTERED_CTX, but never above the
    measurement (honesty beats the floor; the gate's certified
    full-context probe catches a floor the mesh cannot serve in budget).

    Returns (cap_tokens, registered_ctx).
    """
    time_allowance = float(safety_margin) * float(budget_s)
    points = sorted(
        {
            (int(ctx), float(wall_s))
            for ctx, wall_s in samples
            if int(ctx) > 0 and float(wall_s) > 0
        }
    )
    if not points:
        raise ValueError("derive_time_capped_context needs a timing sample")
    if len(points) == 1 or points[0][0] == points[-1][0]:
        ctx_1, wall_1 = points[-1]
        cap = int(time_allowance * ctx_1 / wall_1)
    else:
        (ctx_1, wall_1), (ctx_2, wall_2) = points[0], points[-1]
        rate_1 = wall_1 / ctx_1
        rate_2 = wall_2 / ctx_2
        quad = (rate_2 - rate_1) / float(ctx_2 - ctx_1)
        linear = rate_1 - quad * ctx_1
        if quad <= 0.0:
            cap = int(time_allowance / max(rate_1, rate_2))
        elif linear < 0.0:
            cap = int(ctx_2 * math.sqrt(time_allowance / wall_2))
        else:
            cap = int(
                (math.sqrt(linear * linear + 4.0 * quad * time_allowance) - linear)
                / (2.0 * quad)
            )
    registered = min(measured, cap)
    registered = (registered // CTX_ROUND_MULTIPLE) * CTX_ROUND_MULTIPLE
    registered = max(registered, MIN_REGISTERED_CTX)
    return cap, min(registered, measured)


@dataclass
class DeployConfig:
    model_id: str
    endpoint: str
    chain_config: Any
    private_key: str
    hotkey_seed: bytes | None = None
    coordinator_hotkey_ss58: str = ""
    subtensor_network: str = ""
    uid: int | None = None
    netuid: int | None = None
    # None = derive the registered context: min(measured KV auto-fit,
    # time-derived cap from a timing probe). An explicit number skips the
    # derivation and is accepted only within CTX_JITTER_TOLERANCE of the
    # measurement.
    max_context_len: int | None = None
    # Budget the time-derived context cap is computed against; mirrors the
    # validator's canary_full_context_inference_timeout (neurons/config.py,
    # read by neurons/validator.py; ~900 s). None = the probe gate's
    # full_context_budget_s, which already discounts the validator ceiling
    # for the WAN, TLS, and canary contention a loopback probe does not see.
    validator_budget_s: float | None = None
    # Fraction of validator_budget_s the derived cap may spend; the rest is
    # headroom for timing-probe noise and load at canary time.
    context_safety_margin: float = DEFAULT_CONTEXT_SAFETY_MARGIN
    workers: tuple[str, ...] = ()
    driver: str = ""
    hf_repo: str = ""
    hf_files: tuple[str, ...] = ()
    model_bytes: int = 0
    probe: ProbeGateConfig = field(default_factory=ProbeGateConfig)
    assume_yes: bool = False
    force: bool = False
    dry_run: bool = False
    # Exact registration previously committed by this pool manager.  It is
    # used only to prove an unambiguous endpoint replacement at the same chain
    # index; chain state is re-read and must match the complete stored tuple.
    previous_registration: Mapping[str, Any] | None = None
    launch_wait_s: float = DEFAULT_LAUNCH_WAIT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S


@dataclass
class DeployReport:
    stages: list[dict[str, Any]] = field(default_factory=list)
    anchors: MeshChainAnchors | None = None
    plan: LifecyclePlan | None = None
    placement: dict[str, Any] = field(default_factory=dict)
    mesh_key: str = ""
    measured_ctx_budget: int = 0
    registered_context_len: int = 0
    gate: GateReport | None = None
    registration: RegistrationOutcome | None = None
    failed: bool = False

    def stage(self, name: str, status: str, detail: str = "") -> None:
        self.stages.append({"name": name, "status": status, "detail": detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "mesh_key": self.mesh_key,
            "placement": self.placement,
            "measured_ctx_budget": self.measured_ctx_budget,
            "registered_context_len": self.registered_context_len,
            "plan": (
                {
                    "action": self.plan.action,
                    "predicted_index": self.plan.predicted_index,
                    "reason": self.plan.reason,
                }
                if self.plan
                else None
            ),
            "gate_passed": self.gate.passed if self.gate else None,
            "registration": (
                {
                    "action": self.registration.action,
                    "index": self.registration.index,
                    "tx_hash": self.registration.tx_hash,
                    "expires_at": self.registration.expires_at,
                }
                if self.registration
                else None
            ),
            "failed": self.failed,
        }


def _find_model_mesh(
    status: Mapping[str, Any], model_id: str
) -> tuple[str, dict[str, Any]]:
    # Meshes iterate in manager insertion order, so a lingering corpse of
    # an earlier generation (stopping/error, e.g. after a dead worker that
    # can never ack its teardown) always precedes the live mesh of the
    # same model and would shadow it. Pick by liveness instead: a serving
    # mesh is the one to measure against, a mid-launch mesh (fetching,
    # driving, or joining) is the one to adopt, and a corpse is only
    # relevant when nothing livelier exists.
    priority = {"serving": 0, "fetching": 1, "driving": 1, "joining": 1}
    best_key: str = ""
    best_mesh: dict[str, Any] = {}
    best_rank = 99
    for mesh_key, mesh in (status.get("meshes") or {}).items():
        if str(mesh.get("model_id", "")) != model_id:
            continue
        rank = priority.get(str(mesh.get("status", "")), 2)
        if rank < best_rank:
            best_key, best_mesh, best_rank = str(mesh_key), dict(mesh), rank
    return best_key, best_mesh


def mesh_already_bound(
    mesh: Mapping[str, Any],
    predicted_index: int,
    anchors: Any,
    registered_context: int | None = None,
) -> bool:
    """True when a serving mesh's signed snapshot already binds the CURRENT
    chain state: the predicted index AND every chain anchor. A mesh whose
    snapshot was signed against a since-updated ModelSpec (for example a
    tokenizer-anchor restamp) must relaunch so validators can pin its
    snapshot again.

    The registered context is part of that state: an in-place registration
    refresh that changes ONLY max_context_len left the serving mesh bound
    to the old value, and every validator pin then fails "snapshot model
    does not match expected model" for the rest of the epoch."""

    if registered_context is not None and int(
        mesh.get("max_context_len", 0) or 0
    ) != int(registered_context):
        return False
    return (
        mesh.get("model_index") is not None
        and int(mesh.get("model_index", -1)) == int(predicted_index)
        and bool(mesh.get("verification_snapshot_hash"))
        and str(mesh.get("model_package_hash", "")) == anchors.model_package_hash
        and str(mesh.get("model_tensor_manifest_root", ""))
        == anchors.model_tensor_manifest_root
        and str(mesh.get("tokenizer_hash", "")) == anchors.tokenizer_hash
        and str(mesh.get("quantization_scheme", ""))
        == anchors.chain_quantization_scheme
    )


def _measured_ctx(
    status: Mapping[str, Any],
    model_id: str,
    *,
    mesh_key: str = "",
    allow_model_fallback: bool = True,
) -> int:
    if mesh_key:
        mesh = dict((status.get("meshes") or {}).get(mesh_key) or {})
        try:
            measured = max(
                0, int(mesh.get("measured_ctx_budget", 0) or 0)
            )
        except (TypeError, ValueError):
            measured = 0
        if measured > 0:
            return measured
        if not allow_model_fallback:
            return 0
    entry = dict((status.get("models") or {}).get(model_id) or {})
    try:
        return max(0, int(entry.get("measured_ctx_budget", 0) or 0))
    except (TypeError, ValueError):
        return 0


def run_deploy(
    config: DeployConfig,
    *,
    call: Callable[[str, dict], dict],
    out: Callable[[str], None] = print,
    confirm: Callable[[str], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> DeployReport:
    report = DeployReport()

    def _notify_deploy_stage(stage_name: str) -> None:
        # Best-effort progress marker on the manager so boards can show a
        # running deploy instead of offering a second, racing one.
        try:
            call(
                "/v1/pool/deploy-status",
                {"model_id": config.model_id, "stage": stage_name},
            )
        except Exception:
            pass

    _record_stage = report.stage

    def _stage_and_notify(name: str, status: str, detail: str = "") -> None:
        _record_stage(name, status, detail)
        _notify_deploy_stage(
            "failed" if status in ("failed", "aborted") else name
        )

    report.stage = _stage_and_notify  # type: ignore[method-assign]
    _notify_deploy_stage("starting")

    def fail(stage: str, detail: str) -> DeployReport:
        report.stage(stage, "failed", detail)
        report.failed = True
        out(f"[{stage}] FAILED: {detail}")
        return report

    def wait_for_serving(mesh_key: str, driver_id: str) -> dict[str, Any] | None:
        """Poll until the mesh serves with routing ready; None on failure.

        The timeout is progress-aware: ``launch_wait_s`` bounds the time
        WITHOUT a state change, not the whole launch. A 200GB first-time
        fetch on a 100Mbit pipe takes hours while reporting progress the
        whole way; an absolute clock failed that launch mid-download and
        left the operator to babysit re-runs (observed). Known
        long-static phases (tensor-manifest verify, VRAM load, prewarm)
        legitimately sit on ONE status line for over an hour on the
        largest models, so those get a two-hour floor.
        """
        slow_static_phases = ("verifying", "loading", "prewarm")
        window = config.launch_wait_s
        deadline = clock() + window
        last_progress = ""
        while clock() < deadline:
            status = call("/v1/pool/status", {})
            mesh = dict((status.get("meshes") or {}).get(mesh_key) or {})
            if mesh.get("error"):
                fail("launch-wait", f"mesh error: {mesh['error']}")
                return None
            driver_status = str(
                ((status.get("workers") or {}).get(driver_id) or {}).get(
                    "status", ""
                )
            )
            progress = f"{mesh.get('status', '')} ({driver_status})"
            if progress != last_progress:
                out(f"[launch] {progress}")
                last_progress = progress
                window = config.launch_wait_s
                if any(phase in progress for phase in slow_static_phases):
                    window = max(window, 7200.0)
                deadline = clock() + window
            if mesh.get("status") == "serving" and mesh.get("routing_ready"):
                return mesh
            sleep(config.poll_interval_s)
        fail(
            "launch-wait",
            f"no launch progress for {window:.0f}s (last: {last_progress})",
        )
        return None

    def wait_for_workers_idle(
        workers: list[str], stage: str = "relaunch"
    ) -> bool:
        deadline = clock() + config.launch_wait_s
        states: dict[str, str] = {}
        while clock() < deadline:
            status = call("/v1/pool/status", {})
            states = {
                wid: str(
                    ((status.get("workers") or {}).get(wid) or {}).get(
                        "status", "gone"
                    )
                )
                for wid in workers
            }
            if all(state == "idle" for state in states.values()):
                return True
            sleep(config.poll_interval_s)
        fail(
            stage,
            "workers did not return to idle after stopping the "
            f"mesh: {states}",
        )
        return False

    # 1. Chain preconditions (fail-closed, no writes). Context is not an
    # input here: the registered value is measured by the launch.
    try:
        anchors = registration_module.resolve_mesh_chain_anchors(
            config.chain_config, config.model_id
        )
    except LifecycleRefusal as exc:
        return fail("chain-preconditions", str(exc))
    report.anchors = anchors
    report.stage(
        "chain-preconditions",
        "ok",
        f"{anchors.model_id} {anchors.registry_quant} "
        f"layers={anchors.total_layers} "
        f"manifest_root={anchors.model_tensor_manifest_root[:12]}",
    )
    out(
        f"[chain] {anchors.model_id}: quant={anchors.registry_quant}, "
        f"layers={anchors.total_layers}, anchors derived from ModelSpec"
    )

    # 2. Hotkey binding: the coordinator hotkey must hold a UID on the
    # target netuid, on testnet and mainnet alike. Wallet mode proves it
    # against the metagraph; key-file mode has no hotkey to check, so the
    # UID <-> EVM reconciliation in ensure_evm_registered is the only
    # binding evidence there.
    resolved_uid: int | None = None
    if config.coordinator_hotkey_ss58 and config.subtensor_network:
        if config.netuid is None:
            return fail("hotkey-binding", "no netuid in the chain config")
        try:
            resolved_uid = registration_module.resolve_uid_for_hotkey(
                config.subtensor_network,
                int(config.netuid),
                config.coordinator_hotkey_ss58,
            )
        except LifecycleRefusal as exc:
            return fail("hotkey-binding", str(exc))
        if config.uid is not None and int(config.uid) != resolved_uid:
            return fail(
                "hotkey-binding",
                f"--uid {config.uid} but hotkey "
                f"{config.coordinator_hotkey_ss58} is registered at UID "
                f"{resolved_uid} on netuid {config.netuid}",
            )
        report.stage(
            "hotkey-binding",
            "ok",
            f"UID {resolved_uid} on netuid {config.netuid} "
            f"({config.subtensor_network})",
        )
        out(
            f"[hotkey] {config.coordinator_hotkey_ss58} holds UID "
            f"{resolved_uid} on netuid {config.netuid} "
            f"({config.subtensor_network})"
        )
    else:
        report.stage(
            "hotkey-binding",
            "skipped",
            "no wallet hotkey or no --subtensor-network; UID comes from "
            "--uid / the pool binding and is only verified via the EVM "
            "registry binding",
        )

    # 3. Pool preconditions.
    from verallm.mesh.pool import is_subnet_serving_mode

    status = call("/v1/pool/status", {})
    serving_mode = str(status.get("serving_mode", ""))
    if not is_subnet_serving_mode(serving_mode):
        return fail(
            "pool-preconditions",
            f"pool serving mode is {serving_mode!r}; only a subnet-mode "
            "pool produces the signed snapshots validators verify "
            "(recreate with --serving-mode subnet)",
        )
    binding = dict(status.get("validator_binding") or {})
    from eth_account import Account

    signer_address = Account.from_key(config.private_key).address
    pool_coordinator = str(status.get("coordinator_address", "") or "")
    if pool_coordinator and pool_coordinator.lower() != signer_address.lower():
        return fail(
            "pool-preconditions",
            f"pool coordinator address {pool_coordinator} does not match the "
            f"signing wallet {signer_address}; every snapshot would bind the "
            "wrong coordinator",
        )
    chain_id = getattr(config.chain_config, "chain_id", None)
    if chain_id is not None and binding.get("chain_id") not in (None, chain_id):
        return fail(
            "pool-preconditions",
            f"pool chain_id {binding.get('chain_id')} != config {chain_id}",
        )
    netuid = config.netuid
    if netuid is not None and binding.get("netuid") not in (None, netuid):
        return fail(
            "pool-preconditions",
            f"pool netuid {binding.get('netuid')} != {netuid}",
        )
    uid = config.uid if config.uid is not None else resolved_uid
    if uid is None:
        uid = binding.get("coordinator_uid")
    if uid is None:
        return fail("pool-preconditions", "no coordinator UID (pass --uid)")
    if binding.get("coordinator_uid") not in (None, uid):
        return fail(
            "pool-preconditions",
            f"pool coordinator_uid {binding.get('coordinator_uid')} != {uid}",
        )
    report.stage("pool-preconditions", "ok", f"subnet pool, uid={uid}")

    # 4. Model source: flags > pool registry > the shipped catalogue.
    models = dict(status.get("models") or {})
    existing = dict(models.get(config.model_id) or {})
    hf_repo = config.hf_repo or str(existing.get("hf_repo", "") or "")
    hf_files = list(config.hf_files) or list(existing.get("hf_files") or [])
    model_bytes = int(config.model_bytes or existing.get("model_bytes", 0) or 0)
    if not hf_repo or not hf_files:
        from verallm.registry.models import mesh_model_source

        source = mesh_model_source(config.model_id)
        if source is not None:
            hf_repo, hf_files, model_bytes, _layers = source
            hf_files = list(hf_files)
    if not hf_repo or not hf_files:
        return fail(
            "model-source",
            f"no download source known for {config.model_id}; pass "
            "--hf-repo and --hf-files (and --model-bytes)",
        )
    report.stage("model-source", "ok", hf_repo)

    # Serving-correctness cap: the KV auto-fit measures MEMORY, and the
    # timing probes measure SPEED - neither proves the runtime serves the
    # context CORRECTLY. A runtime bug past a position threshold produces
    # degenerate output that light proofs verify and only the decode
    # audit catches. The catalogue
    # cap pins the measurement launch AND bounds the registration; an
    # explicit override beyond it is refused outright.
    from verallm.registry.models import mesh_model_serving_context_cap

    serving_cap = mesh_model_serving_context_cap(config.model_id)
    # `mesh deploy` is the calibration boundary.  A stored registration only
    # proves which chain index this pool may update; it never supplies the
    # replacement hardware's serving capacity.  Re-measure automatically on
    # every default deploy so a stronger replacement may advertise more and a
    # weaker one cannot inherit an unsafe contract.  Lease renewal does not
    # run this path.
    replacement_recalibration = (
        isinstance(config.previous_registration, Mapping)
        and str(config.previous_registration.get("model_id", ""))
        == config.model_id
    )
    same_endpoint_recalibration = (
        replacement_recalibration
        and str(config.previous_registration.get("endpoint", ""))
        == config.endpoint
    )
    if serving_cap > 0:
        if (
            config.max_context_len is not None
            and int(config.max_context_len) > serving_cap
        ):
            return fail(
                "context",
                f"--max-context-len {config.max_context_len} exceeds the "
                f"catalogue serving cap {serving_cap} for "
                f"{config.model_id}: the runtime does not serve that "
                "context correctly (validator canaries at the registered "
                "maximum would fail every audit)",
            )
        out(
            f"[context] catalogue serving cap {serving_cap} pins this "
            "launch (runtime correctness bound, not a memory limit)"
        )

    # 5. Placement.
    advice = call("/v1/pool/recommend", {"model_id": config.model_id})
    suggestions = list(advice.get("suggestions") or [])
    reasons = dict(advice.get("reasons") or {})
    mesh_key, mesh = _find_model_mesh(status, config.model_id)
    if config.workers:
        placement = {
            "workers": list(config.workers),
            "driver": config.driver or config.workers[0],
            "max_rtt_ms": 0.0,
        }
        for suggestion in suggestions:
            if set(suggestion.get("workers", [])) == set(config.workers):
                placement = dict(suggestion)
                break
    elif mesh:
        placement = {
            "workers": list(mesh.get("members") or []),
            "driver": str(mesh.get("driver", "")),
            "max_rtt_ms": 0.0,
        }
    elif suggestions:
        placement = dict(suggestions[0])
    else:
        lines = [f"  {wid}: {reason}" for wid, reason in sorted(reasons.items())]
        return fail(
            "placement",
            "no viable placement:\n" + "\n".join(lines),
        )
    report.placement = placement
    out(
        f"[placement] workers={','.join(placement['workers'])} "
        f"driver={placement.get('driver')} "
        f"link={placement.get('link_class', '?')} "
        f"rtt={float(placement.get('max_rtt_ms', 0.0)):.0f}ms"
    )
    if placement.get("warn"):
        out(f"[placement] warn: {placement['warn']}")
    for wid, reason in sorted(reasons.items()):
        out(f"[placement] {wid}: {reason}")
    from verallm.mesh.capacity_roster import mainnet_backend_gate_reason

    selected_workers = dict(status.get("workers") or {})
    backend_reason = mainnet_backend_gate_reason(
        [
            dict(
                (selected_workers.get(str(worker_id)) or {}).get("capability")
                or {}
            ).get("rpc_device", "")
            for worker_id in placement["workers"]
        ],
        netuid=netuid,
        chain_id=chain_id,
    )
    if backend_reason:
        return fail(
            "placement",
            f"{backend_reason}; mainnet mesh deployment currently requires "
            "an all-CUDA worker roster (Metal and other backends remain "
            "available on testnet)",
        )
    driver_id = str(placement.get("driver", ""))
    driver_view = dict((status.get("workers") or {}).get(driver_id) or {})
    driver_capability = dict(driver_view.get("capability") or {})
    if driver_view and not (
        driver_capability.get("subnet_driver_ready")
        # pre-rename workers advertise the old key
        or driver_capability.get("validator_driver_ready")
    ):
        return fail(
            "placement",
            f"driver {driver_id!r} is not subnet-ready (its worker needs a "
            "fresh, non-empty validator allowlist via "
            "--subtensor-network/--netuid; no wallet)",
        )
    report.stage("placement", "ok", ",".join(placement["workers"]))

    if config.dry_run:
        report.stage(
            "measurement", "skipped", "dry run stops before launching"
        )
        out("[dry-run] preflight complete; no launch, no transactions")
        return report

    # 6. Measurement: reuse a serving mesh for a first registration.  A
    # replacement deploy must instead stop the chain-bound instance and launch
    # an UNREGISTERED measurement instance on the selected hardware.  The
    # stored registration remains only as authorization to update the same
    # index after the fresh gate passes.
    mesh_status = str(mesh.get("status", "")) if mesh else ""
    launch_confirmed = False
    if replacement_recalibration:
        if confirm is not None and not config.assume_yes:
            verb = "Stop and recalibrate" if mesh else "Recalibrate"
            if not confirm(
                f"{verb} {config.model_id} on "
                f"{','.join(placement['workers'])} "
                f"(driver {placement.get('driver')})?"
            ):
                report.stage(
                    "measurement", "aborted", "operator declined calibration"
                )
                report.failed = True
                return report
            launch_confirmed = True
        # Persistently suspend lease renewal before any chain deactivation or
        # runtime stop, then wait out a renewal already in flight.  The final
        # successful registration-state write replaces this stored record and
        # therefore clears the suspension.  A failed deploy keeps renewal
        # suspended, which is the safe state for an inactive/unknown slot.
        renewal_deadline = clock() + min(
            max(config.poll_interval_s * 6, 30.0), 300.0
        )
        while True:
            try:
                suspension = call(
                    "/v1/pool/registration-state",
                    {
                        "model_id": config.model_id,
                        "suspend_renewal": True,
                    },
                )
            except (RuntimeError, OSError, ValueError) as exc:
                return fail(
                    "measurement",
                    f"could not suspend lease renewal before replacement: {exc}",
                )
            registrations = suspension.get("registrations") or {}
            stored = registrations.get(config.model_id) or {}
            renewal_in_progress = suspension.get("renewal_in_progress")
            if type(renewal_in_progress) is not bool:
                return fail(
                    "measurement",
                    "pool manager did not prove lease-renewal quiescence; "
                    "update the manager before recalibrating",
                )
            if not bool(stored.get("renewal_suspended")):
                return fail(
                    "measurement",
                    "pool manager did not confirm persistent lease-renewal "
                    "suspension; update the manager before recalibrating",
                )
            if not renewal_in_progress:
                break
            if clock() >= renewal_deadline:
                return fail(
                    "measurement",
                    "an in-flight lease renewal did not quiesce before the "
                    "replacement calibration deadline",
                )
            out("[measure] waiting for an in-flight lease renewal to finish")
            sleep(max(config.poll_interval_s, 0.1))
        report.stage(
            "measurement-renewal",
            "ok",
            "lease renewal suspended and no renewal transaction is in flight",
        )
        if same_endpoint_recalibration:
            out(
                "[measure] taking the exact same-endpoint registration "
                "offline before exposing an unbound calibration runtime"
            )
            try:
                inactive = (
                    registration_module.deactivate_mesh_endpoint_for_recalibration(
                        config.chain_config,
                        config.previous_registration,
                        private_key=config.private_key,
                        expected_endpoint=config.endpoint,
                    )
                )
            except LifecycleRefusal as exc:
                return fail("measurement", str(exc))
            report.stage(
                "measurement-slot",
                "ok",
                f"{inactive.action} index={inactive.index} "
                f"tx={inactive.tx_hash or 'none'}",
            )
    if replacement_recalibration and mesh:
        old_members = [str(member) for member in (mesh.get("members") or [])]
        out(
            f"[measure] stopping registered mesh {mesh_key} for fresh "
            "replacement-hardware calibration"
        )
        call("/v1/pool/stop", {"mesh_key": mesh_key})
        if not wait_for_workers_idle(
            list(
                dict.fromkeys(
                    old_members + [str(worker) for worker in placement["workers"]]
                )
            ),
            stage="measurement",
        ):
            return report
        mesh_key, mesh, mesh_status = "", {}, ""
    if mesh and mesh_status == "serving":
        report.mesh_key = mesh_key
        out(f"[measure] reusing serving mesh {mesh_key}")
    elif mesh and mesh_status in ("fetching", "driving", "joining"):
        # A mesh for this model is mid-launch - typically the orphan of a
        # deploy that died between launching and its chain write. Its
        # workers are busy, so launching fresh here would only fail with
        # "worker busy". Adopt it instead: wait for it
        # to serve and measure against it; the gate and the chain-bound
        # relaunch below supersede whatever state it carries.
        report.mesh_key = mesh_key
        out(
            f"[measure] adopting mesh {mesh_key} already mid-launch "
            f"(status {mesh_status})"
        )
        served = wait_for_serving(
            mesh_key,
            str(mesh.get("driver", "") or placement.get("driver", "")),
        )
        if served is None:
            return report
        mesh = served
    else:
        if mesh and mesh_status in ("stopping", "error"):
            # A dead or half-stopped mesh still pins its workers; clear it
            # first so the launch below does not fail with "worker busy".
            out(
                f"[measure] clearing {mesh_status} mesh {mesh_key} before "
                "launching"
            )
            call("/v1/pool/stop", {"mesh_key": mesh_key})
            if not wait_for_workers_idle(
                list(placement["workers"]), stage="measurement"
            ):
                return report
            mesh_key, mesh = "", {}
        if (
            confirm is not None
            and not config.assume_yes
            and not launch_confirmed
        ):
            if not confirm(
                f"Launch {config.model_id} on "
                f"{','.join(placement['workers'])} "
                f"(driver {placement.get('driver')})?"
            ):
                report.stage(
                    "measurement", "aborted", "operator declined placement"
                )
                report.failed = True
                return report
        launched = call(
            "/v1/pool/launch",
            {
                "model_id": config.model_id,
                "workers": list(placement["workers"]),
                "driver": placement.get("driver", ""),
                # Internal deploy/manager contract, not an operator flag: a
                # replacement measurement must not inherit the old chain-bound
                # max_context_len from pool state.  The later register/relaunch
                # stages restore the binding at the same index.
                "_deploy_measurement_unbound": replacement_recalibration,
                # Chain anchors ride the launch so a driver that must fetch
                # can pull the owner-published manifest from the store and
                # root-verify it, instead of rebuilding it locally (that is
                # the subnet owner's process, never a miner's). The pool
                # registry cannot carry these yet: registration commits the
                # measured context, which this launch is about to measure.
                "model_tensor_manifest_root": anchors.model_tensor_manifest_root,
                "manifest_urls": list(
                    getattr(
                        config.chain_config, "mesh_manifest_base_urls", ()
                    )
                    or ()
                ),
                # Measure at the intended registration budget when the
                # operator pinned one (or at the catalogue serving cap):
                # serving-survives at the contract is what matters, not
                # the theoretical VRAM maximum - and a request past the
                # cap must be REJECTED by the runtime, never served
                # degenerately.
                **(
                    {"max_context_len": int(config.max_context_len)}
                    if config.max_context_len is not None
                    else (
                        {"max_context_len": int(serving_cap)}
                        if serving_cap > 0
                        else {}
                    )
                ),
            },
        )
        mesh_key = str(launched.get("mesh_key", ""))
        report.mesh_key = mesh_key
        out(f"[measure] {mesh_key} launching (unregistered, operator lane)")
        served = wait_for_serving(mesh_key, driver_id)
        if served is None:
            return report
        mesh = served
        if replacement_recalibration and (
            mesh.get("model_index") is not None
            or bool(mesh.get("verification_snapshot_hash"))
        ):
            out(
                "[measure] FAIL: pool manager inherited the previous chain "
                "binding during replacement calibration; stopping it"
            )
            try:
                call("/v1/pool/stop", {"mesh_key": mesh_key})
                wait_for_workers_idle(
                    [str(worker) for worker in placement["workers"]],
                    stage="measurement",
                )
            except Exception as exc:
                out(f"[measure] cleanup after incompatible manager failed: {exc}")
            return fail(
                "measurement",
                "pool manager did not honor the unbound replacement-"
                "measurement contract; update the manager before retrying",
            )
    report.stage("measurement", "ok", mesh_key)

    # Early reachability abort: the measurement mesh has just bound the SAME
    # public port the registration will claim, so health and TLS are checkable
    # now. Validator-auth posture is not: an unregistered measurement has no
    # model index or signed verification snapshot and deliberately runs the
    # operator lane. Require validator-auth posture only after the chain-bound
    # relaunch below. This still avoids spending the full probe gate on an
    # unreachable endpoint or broken TLS listener.
    early_posture = check_public_endpoint(config.endpoint)
    early_required = {"public-health", "tls-certificate"}
    early_failures = [
        check
        for check in early_posture
        if check.name in early_required
        and check.kind == "hard"
        and not check.passed
    ]
    if early_failures:
        for check in early_failures:
            out(
                f"[endpoint] FAIL {check.name}: {check.observed} "
                f"(need {check.threshold}) - {check.remediation}"
            )
        return fail(
            "endpoint-reachability",
            f"{config.endpoint} is not publicly serving the measurement "
            "mesh; fix the port mapping / firewall / reverse proxy before "
            "spending the probe gate on it",
        )
    report.stage(
        "endpoint-reachability", "ok", f"{config.endpoint} answers publicly"
    )

    status = call("/v1/pool/status", {})
    measured = _measured_ctx(
        status,
        config.model_id,
        mesh_key=mesh_key,
        allow_model_fallback=not replacement_recalibration,
    )
    if replacement_recalibration and measured <= 0:
        return fail(
            "measurement",
            "replacement hardware did not report a fresh per-mesh context "
            "measurement; refusing to reuse the previous host's value",
        )
    report.measured_ctx_budget = measured
    if measured > 0:
        out(f"[measure] KV auto-fit measured {measured} context")
    if serving_cap > 0 and measured > serving_cap:
        # The measurement reports the VRAM fit even on a pinned launch;
        # every probe below must stay inside the runtime-correctness cap
        # (a probe past it exercised the exact degeneration the cap
        # exists to avoid).
        measured = serving_cap
        out(
            f"[measure] catalogue serving cap bounds the usable context "
            f"to {measured}"
        )
    elif measured <= 0:
        out(
            "[measure] WARNING: no measured context budget reported; the "
            "worker may predate measurement"
        )

    # 7. Registered context. An explicit override wins (validated against
    # the measurement within the restart-jitter tolerance, skipping the
    # derivation). Otherwise the registered value is min(measured KV
    # auto-fit, time-derived cap): validators canary at the registered
    # maximum, and that canary must verify inside the validator's
    # inference budget, so the KV fit alone is not registrable on hardware
    # that cannot prefill it in time. The decision uses a timing estimate
    # here; the gate's full-context probe below then certifies the final
    # value.
    derived_cap = 0
    if config.max_context_len is not None:
        requested = int(config.max_context_len)
        if measured > 0 and requested > int(
            measured * (1.0 + CTX_JITTER_TOLERANCE)
        ):
            return fail(
                "context",
                f"--max-context-len {requested} exceeds the measured context "
                f"{measured} by more than "
                f"{int(CTX_JITTER_TOLERANCE * 100)}%; validators canary at "
                "the registered maximum, so this registration would fail an "
                "honest audit. Omit the flag to register the measured value.",
            )
        registered_ctx = requested
    elif measured > 0:
        budget_s = (
            float(config.validator_budget_s)
            if config.validator_budget_s is not None
            else float(config.probe.full_context_budget_s)
        )
        margin = float(config.context_safety_margin)

        def _timing_probe(probe_ctx: int) -> tuple[float, str, str]:
            """One prefill-dominated probe; returns (wall_s, error, kind).

            kind distinguishes WHERE a failure happened: "transport" means
            the manager/mesh could not be reached (an infra problem this
            derivation must not paper over), "serve" means the mesh
            answered and the verified serve itself failed at this context.
            A serve-level failure is a property of the model+runtime at
            that context — canaries at it would fail the same way — so the
            caller treats it as a servable-context CEILING, not an error.
            """
            try:
                timing_result = call(
                    "/v1/pool/chat",
                    {
                        "mesh_key": mesh_key,
                        "probe": True,
                        "stream": True,
                        "thinking": False,
                        "messages": [
                            {
                                "role": "user",
                                "content": build_full_context_probe_prompt(
                                    probe_ctx
                                ),
                            }
                        ],
                        "max_tokens": FULL_CONTEXT_OUTPUT_TOKENS,
                        "timeout": min(budget_s + 30.0, 960.0),
                    },
                )
            except (RuntimeError, OSError) as exc:
                return 0.0, str(exc), "transport"
            timing = probe_sample_from_chat_result(timing_result)
            if not timing.ok or timing.total_s <= 0:
                error = timing.error[:200] or "no wall time reported"
                # Only VERIFICATION-class failures are evidence of a
                # context cliff; busy slots, deadlines and other transient
                # serve errors say nothing about the model and must fail
                # the stage exactly as before.
                kind = (
                    "serve"
                    if re.search(
                        r"audit|verif|proof", error, flags=re.IGNORECASE
                    )
                    else "transport"
                )
                return 0.0, error, kind
            return timing.total_s, "", ""

        def _bisect_servable_ceiling(
            lo_pass: int,
            hi_fail: int,
            samples: list[tuple[int, float]],
        ) -> int:
            """Largest probed context whose verified serve passes.

            A serve-level probe failure above a passing one is a real
            model+runtime cliff.
            Validators canary at the registered context, so registering
            past the cliff guarantees probation; erroring out entirely
            hides the honest window that DOES verify. Bisect the boundary
            and register below it.
            """
            lo, hi = int(lo_pass), int(hi_fail)
            while hi - lo > max(CTX_ROUND_MULTIPLE, 4096):
                mid = ((lo + hi) // 2 // CTX_ROUND_MULTIPLE) * CTX_ROUND_MULTIPLE
                if mid <= lo or mid >= hi:
                    break
                out(f"[context] bisecting servable ceiling: probing {mid}")
                wall, err, kind = _timing_probe(mid)
                if not err:
                    samples.append((mid, wall))
                    out(
                        f"[context] probe {wall:.1f}s at {mid} context: passes"
                    )
                    lo = mid
                elif kind == "serve":
                    out(f"[context] probe at {mid} context: fails verification")
                    hi = mid
                else:
                    # Transport errors say nothing about the model; stop
                    # bisecting and keep the last KNOWN-GOOD floor.
                    out(
                        f"[context] transport error at {mid}; keeping "
                        f"known-good {lo}"
                    )
                    break
            return lo

        servable_ceiling = int(measured)
        timing_ctx = min(measured, TIMING_PROBE_CTX)
        out(
            f"[context] timing probe at {timing_ctx} context to estimate "
            "verify seconds per context token"
        )
        wall_1, probe_error, probe_kind = _timing_probe(timing_ctx)
        if probe_error and probe_kind == "serve":
            # Even the base probe context is past the model's cliff; find
            # the honest window below it instead of refusing outright.
            out(
                f"[context] probe at {timing_ctx} fails verification: "
                f"{probe_error}"
            )
            samples: list[tuple[int, float]] = []
            floor = _bisect_servable_ceiling(
                CTX_ROUND_MULTIPLE, timing_ctx, samples
            )
            if not samples:
                return fail(
                    "context",
                    f"no context verifies (probed down from {timing_ctx}); "
                    f"first failure: {probe_error}",
                )
            servable_ceiling = floor
            timing_ctx, wall_1 = samples[-1]
        elif probe_error:
            return fail(
                "context",
                f"timing probe at context {timing_ctx} failed: {probe_error}",
            )
        else:
            samples = [(timing_ctx, wall_1)]
        seconds_per_token = wall_1 / float(timing_ctx)
        out(
            f"[context] timing probe {wall_1:.1f}s at {timing_ctx} context "
            f"({seconds_per_token * 1000.0:.3f} ms per context token)"
        )
        # Prefill attention is superlinear, so extrapolating the 32k rate
        # alone overestimates the servable context. Whenever the linear
        # candidate extrapolates beyond the probe point, a second probe
        # turns the model into a linear+quadratic fit. Its context stays
        # at or below the linear candidate's midpoint (the candidate
        # itself already costs the full time allowance to serve, and
        # reality is slower than the linear estimate).
        linear_candidate = min(
            measured, int(margin * budget_s / seconds_per_token)
        )
        second_ctx = min(
            measured,
            TIMING_PROBE_CTX_HIGH,
            (timing_ctx + linear_candidate) // 2,
        )
        if second_ctx >= timing_ctx * 9 // 8 and second_ctx <= servable_ceiling:
            out(
                f"[context] second timing probe at {second_ctx} context to "
                "fit the superlinear prefill curve"
            )
            wall_2, probe_error, probe_kind = _timing_probe(second_ctx)
            if probe_error and probe_kind == "serve":
                out(
                    f"[context] probe at {second_ctx} fails verification: "
                    f"{probe_error}"
                )
                servable_ceiling = _bisect_servable_ceiling(
                    timing_ctx, second_ctx, samples
                )
                out(
                    "[context] servable ceiling bisected to "
                    f"{servable_ceiling} (verification fails above it)"
                )
            elif probe_error:
                return fail(
                    "context",
                    f"timing probe at context {second_ctx} failed: "
                    f"{probe_error}",
                )
            else:
                samples.append((second_ctx, wall_2))
                out(
                    f"[context] second timing probe {wall_2:.1f}s at "
                    f"{second_ctx} context "
                    f"({wall_2 / second_ctx * 1000.0:.3f} ms per context token)"
                )
        derived_cap, registered_ctx = derive_time_capped_context(
            measured=min(int(measured), int(servable_ceiling)),
            samples=samples,
            budget_s=budget_s,
            safety_margin=margin,
        )
        if registered_ctx > servable_ceiling:
            registered_ctx = (
                servable_ceiling // CTX_ROUND_MULTIPLE
            ) * CTX_ROUND_MULTIPLE
        if serving_cap > 0 and registered_ctx > serving_cap:
            # A REUSED serving mesh may have measured its KV fit without
            # the catalogue pin; the registration must still respect the
            # runtime-correctness bound.
            registered_ctx = (
                serving_cap // CTX_ROUND_MULTIPLE
            ) * CTX_ROUND_MULTIPLE
            out(
                f"[context] catalogue serving cap clamps the registration "
                f"to {registered_ctx}"
            )
        out(
            f"[context] measured KV fit {measured}; "
            f"budget {budget_s:.0f}s x margin {margin:.2f} -> "
            f"time cap {derived_cap}; registering {registered_ctx}"
        )
    else:
        return fail(
            "context",
            "no measured context budget and no --max-context-len; update "
            "the worker so the launch reports its KV auto-fit, or pass an "
            "explicit value",
        )
    report.registered_context_len = registered_ctx
    report.stage(
        "context",
        "ok",
        f"registering {registered_ctx} (measured {measured or 'unknown'}"
        + (f", time cap {derived_cap}" if derived_cap else "")
        + ")",
    )
    out(f"[context] registering max_context_len={registered_ctx}")

    # 8. Probe gate on the measurement mesh: the ready-to-register
    # confirmation (light proofs, hard-tier proof, full-context at the
    # registered context, throughput, TTFT).
    gate_config = replace(
        config.probe,
        max_context_len=registered_ctx,
        measured_ctx_budget=measured,
    )
    gate = run_probe_gate(
        call=call,
        mesh_key=mesh_key,
        expected_snapshot_hash=str(
            mesh.get("verification_snapshot_hash", "") or ""
        ),
        expected_stage_count=len(mesh.get("members") or []),
        max_rtt_ms=float(placement.get("max_rtt_ms", 0.0) or 0.0),
        config=gate_config,
        log=lambda line: out(f"[gate] {line}"),
    )
    report.gate = gate
    out(gate.render())

    if not gate.passed and not config.force:
        report.failed = True
        report.stage("gate", "failed", "see checks above")
        best_alternative = next(
            (
                suggestion
                for suggestion in suggestions
                if set(suggestion.get("workers", []))
                != set(placement.get("workers", []))
            ),
            None,
        )
        out(
            "[gate] FAILED. The mesh keeps serving UNREGISTERED (invisible "
            "to validators); no transaction was sent and no reputation is "
            "at risk."
        )
        if best_alternative is not None:
            out(
                "[gate] best alternative placement: "
                f"workers={','.join(best_alternative.get('workers', []))} "
                f"link={best_alternative.get('link_class', '?')} "
                f"rtt={float(best_alternative.get('max_rtt_ms', 0.0)):.0f}ms"
            )
        out("[gate] override with --force to register anyway")
        return report
    if not gate.passed:
        report.stage("gate", "forced", "gate failed but --force was given")
        out("[gate] failed checks OVERRIDDEN by --force")
    else:
        report.stage("gate", "ok", "all hard checks passed")

    if confirm is not None and not config.assume_yes:
        if not confirm(
            f"Register {config.model_id} for UID {uid} at context "
            f"{registered_ctx} on netuid {netuid} and go live?"
        ):
            report.stage("register", "aborted", "operator declined")
            report.failed = True
            return report

    # 9. EVM binding (the first write; no reputation risk).
    if config.hotkey_seed is None or netuid is None:
        return fail(
            "evm-binding",
            "EVM reconciliation needs --wallet/--hotkey (hotkey seed) "
            "and the netuid",
        )
    # The two chain steps below can sit on a slow RPC for tens of seconds;
    # without an immediate line the confirmed deploy reads as hung (live
    # operator feedback).
    out("[chain] reconciling EVM binding (may submit a tx and wait for it)...")
    try:
        sent = registration_module.ensure_evm_registered(
            config.chain_config,
            uid=int(uid),
            hotkey_seed=config.hotkey_seed,
            netuid=int(netuid),
            private_key=config.private_key,
        )
    except LifecycleRefusal as exc:
        return fail("evm-binding", str(exc))
    report.stage("evm-binding", "ok", "registered" if sent else "already bound")

    # 10. Index prediction: handle_launch REQUIRES model_index for a
    # chain-bound mesh, so the index must be known before the relaunch and
    # verified after the chain write.
    target = build_registration_target(
        anchors=anchors,
        endpoint=config.endpoint,
        max_context_len=registered_ctx,
    )
    out("[chain] planning the registration slot (reading existing entries)...")
    try:
        plan_kwargs = (
            {"previous_registration": config.previous_registration}
            if config.previous_registration is not None
            else {}
        )
        plan = registration_module.plan_mesh_registration(
            config.chain_config, target, signer_address, **plan_kwargs
        )
    except LifecycleRefusal as exc:
        return fail("index-prediction", str(exc))
    report.plan = plan
    report.stage(
        "index-prediction", "ok", f"{plan.action} at index {plan.predicted_index}"
    )
    out(f"[index] {plan.action} at {plan.predicted_index}: {plan.reason}")

    # 11. Pool model registration: chain-derived anchors, measured context.
    call(
        "/v1/pool/register-model",
        {
            "model_id": config.model_id,
            "hf_repo": hf_repo,
            "hf_files": hf_files,
            "model_bytes": model_bytes,
            "layers": anchors.total_layers,
            "max_context_len": registered_ctx,
            "measured_ctx_budget": measured,
            "model_index": plan.predicted_index,
            "model_package_hash": anchors.model_package_hash,
            "model_tensor_manifest_root": anchors.model_tensor_manifest_root,
            "tokenizer_hash": anchors.tokenizer_hash,
            "quantization_scheme": anchors.chain_quantization_scheme,
        },
    )
    report.stage("pool-model", "ok", f"model_index={plan.predicted_index}")

    # 12. Relaunch chain-bound: the measurement mesh signed nothing, so it
    # must be replaced by a mesh whose snapshot binds the predicted index.
    # A mesh that is ALREADY chain-bound to the predicted index (re-deploy
    # of a live registration) keeps serving untouched — but only when its
    # snapshot still carries the CURRENT chain anchors; a mesh signed
    # against a since-updated ModelSpec must relaunch so validators can pin
    # its snapshot again.
    already_bound = mesh_already_bound(
        mesh, plan.predicted_index, anchors, registered_context=registered_ctx
    )
    if already_bound:
        out(
            f"[relaunch] mesh {mesh_key} is already chain-bound to index "
            f"{plan.predicted_index}; keeping it serving"
        )
    else:
        out("[relaunch] stopping the measurement mesh")
        call("/v1/pool/stop", {"mesh_key": mesh_key})
        if not wait_for_workers_idle(list(placement["workers"])):
            return report
        launched = call(
            "/v1/pool/launch",
            {
                "model_id": config.model_id,
                "workers": list(placement["workers"]),
                "driver": placement.get("driver", ""),
                # The binding this launch uses was written at stage 11 as
                # a PREDICTION; the chain write happens after this mesh's
                # snapshot verifies (stage 14). Only the deploy dance may
                # launch against an unconfirmed binding - a later launch
                # finding one is an aborted deploy's leftover and purges
                # it instead.
                "pending_binding_ok": True,
            },
        )
        mesh_key = str(launched.get("mesh_key", ""))
        report.mesh_key = mesh_key
        out(f"[relaunch] {mesh_key} launching (chain-bound)")
        mesh = wait_for_serving(mesh_key, driver_id)
        if mesh is None:
            return report
    snapshot_hash = str(mesh.get("verification_snapshot_hash", "") or "")
    if not snapshot_hash:
        return fail(
            "relaunch",
            "chain-bound mesh reports no verification snapshot hash; the "
            "coordinator did not sign its snapshot",
        )
    report.stage("relaunch", "ok", f"{mesh_key} snapshot {snapshot_hash[:12]}")

    # 13. Final verification: one hard-tier probe through the chain-bound
    # mesh (asserts the postcommit-shaped proof path and snapshot binding)
    # plus the public endpoint posture. Reachability failures are never
    # forceable: registering an unreachable endpoint or one with validator
    # auth disabled cannot be the operator's intent.
    verify_result = call(
        "/v1/pool/chat",
        {
            "mesh_key": mesh_key,
            "probe": True,
            "stream": True,
            "thinking": False,
            "proof_tier": "hard",
            "messages": [
                {"role": "user", "content": "Reply with the word ready."}
            ],
            "max_tokens": 16,
            "timeout": 900.0,
        },
    )
    verify_sample = probe_sample_from_chat_result(verify_result)
    snapshot_bound = (
        verify_sample.verification_snapshot_hash == snapshot_hash
    )
    if not (
        verify_sample.ok
        and verify_sample.verified
        and verify_sample.receipt_verified
        and snapshot_bound
    ):
        return fail(
            "final-verification",
            "chain-bound hard probe failed: "
            f"ok={verify_sample.ok} verified={verify_sample.verified} "
            f"receipt={verify_sample.receipt_verified} "
            f"snapshot_bound={snapshot_bound} "
            f"error={verify_sample.error[:200]}",
        )
    posture = check_public_endpoint(config.endpoint)
    if report.gate is not None:
        report.gate.reachability = posture
    posture_failures = [
        check for check in posture if check.kind == "hard" and not check.passed
    ]
    if posture_failures:
        for check in posture_failures:
            out(f"[posture] FAIL {check.name}: {check.observed}")
        return fail(
            "final-verification",
            "public endpoint posture failed (not forceable): "
            + "; ".join(check.name for check in posture_failures),
        )
    report.stage(
        "final-verification",
        "ok",
        "hard proof verified, snapshot bound, endpoint posture ok",
    )

    # 14. Register on MinerRegistry, index-guarded.
    try:
        registration_kwargs = (
            {"previous_registration": config.previous_registration}
            if config.previous_registration is not None
            else {}
        )
        outcome = registration_module.register_mesh_endpoint(
            config.chain_config,
            target,
            private_key=config.private_key,
            expected_index=plan.predicted_index,
            **registration_kwargs,
        )
    except Exception as exc:
        # Any registry-write or verification failure leaves the chain-bound
        # launch unconfirmed. This includes ambiguous RPC outcomes after an
        # endpoint update or refresh, not only explicit lifecycle refusals.
        # A mesh serving against unverified chain state would fail canaries or
        # renew a stale registration, so always stop it.
        try:
            call("/v1/pool/stop", {"mesh_key": mesh_key})
        except (RuntimeError, OSError):
            out("[register] WARNING: could not stop the mismatched mesh")
        return fail("register", f"{exc} (mesh {mesh_key} stopped)")
    report.registration = outcome
    report.stage(
        "register",
        "ok",
        f"{outcome.action} index={outcome.index} tx={outcome.tx_hash or 'none'}",
    )
    out(
        f"[register] {outcome.action} at index {outcome.index}"
        + (f" tx {outcome.tx_hash}" if outcome.tx_hash else " (no tx needed)")
    )

    # 15. Persist renewal state on the pool manager so its lease renewer
    # takes over; deploy may run on any machine holding the admin token.
    try:
        call(
            "/v1/pool/registration-state",
            {
                "registration": {
                    "model_id": target.model_id,
                    "endpoint": target.endpoint,
                    "quant": target.quant,
                    "max_context_len": target.max_context_len,
                    "model_spec_ref": bytes(target.model_spec_ref).hex(),
                    "index": outcome.index,
                    "mesh_key": mesh_key,
                    "expires_at": outcome.expires_at,
                }
            },
        )
        report.stage("renewal-state", "ok", "pool manager renews the lease")
    except Exception as exc:
        report.failed = True
        report.stage(
            "renewal-state",
            "failed",
            f"{exc}; re-run the same `verathos mesh deploy` command to "
            "verify the chain tuple and persist automatic renewal state",
        )
        out(
            f"[renewal] FAILED: {exc}; re-run the same mesh deploy command"
        )
    _notify_deploy_stage("failed" if report.failed else "done")
    return report
