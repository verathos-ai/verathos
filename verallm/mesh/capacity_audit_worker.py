"""Mesh-side hot-capacity audit worker.

Runs inside every ``mesh pool worker`` daemon whose worker is a member of a
chain-bound subnet mesh with CUDA GPUs.  Mirrors the non-interactive
protocol of the vLLM miner's ``CapacityAuditMinerWorker``: it watches
current-head chain data, derives the audit cohort locally, and when the
mesh's endpoint slot is selected it runs the synthetic CUDA workload on
EVERY local GPU named in the signed roster simultaneously — one opening per
global roster ordinal, all inside the same chain-anchored window.
Validators never send per-miner challenge commands.

Differences from the vLLM worker, all forced by the mesh topology:

* The slot's chain identity (coordinator EVM address, model_index,
  registered endpoint/quant/context) is DELIVERED by the pool manager in
  the audit context, because only the manager holds the chain registration
  state.  The worker holds no wallet; artifacts are signed through the
  manager's delegated ``capacity-audit-artifact`` purpose and verified
  locally against the coordinator address before publishing.
* One daemon may own several GPUs (a 4-GPU unit is ONE worker): it launches
  one ``bench_combined`` subprocess per local GPU, masked with
  ``CUDA_VISIBLE_DEVICES``, each seeded from its GLOBAL roster ordinal so
  every opening has a distinct, publicly derivable seed.
* While a window is active the worker writes a drain state file the serve
  process reads: admission returns the ORDINARY busy 503 during the drain,
  byte-identical to a saturation rejection, so validators cannot use the
  refusal to fingerprint audit windows (canary-oracle rule).

Boundary note: this verallm module lazy-imports protocol helpers from
``neurons.capacity_audit`` at function scope.  The audit protocol has
exactly one implementation and it lives in neurons/; duplicating the
derivations here would eventually fork them.  This follows the existing
exception precedent (``verallm/mesh/http_auth.py`` importing
``neurons.request_signing``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


class CapacityAuditWorkspaceError(RuntimeError):
    """Subnet worker cannot run capacity audits and must not serve.

    Raised at audit-worker start when the hot-capacity workspace is
    unavailable and the operator has not explicitly opted out. Callers
    treat this as fatal for the worker process: serving without audits
    means probating with no evidence and nothing in the logs.
    """


CAPACITY_DRAIN_FILE_NAME = "capacity-audit-drain.json"
CAPACITY_ROSTER_FILE_NAME = "capacity-roster.json"

_BLOCK_TIME_S = 12.0


def capacity_drain_file_path(workdir: str | os.PathLike[str]) -> Path:
    return Path(workdir) / CAPACITY_DRAIN_FILE_NAME


def capacity_roster_file_path(workdir: str | os.PathLike[str]) -> Path:
    return Path(workdir) / CAPACITY_ROSTER_FILE_NAME


def capacity_drain_active(path: str | os.PathLike[str]) -> bool:
    """Whether the drain file marks an active audit window right now.

    Read by the serve process on every admission; cheap (one small owner
    file), fail-open (an unreadable or absent file never blocks serving).
    """

    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return False
    except Exception:
        return False
    if not isinstance(data, dict) or not bool(data.get("active")):
        return False
    until_ts = float(data.get("until_ts", 0.0) or 0.0)
    # Unbounded drains are refused: the writer always stamps a deadline,
    # so a zero/missing bound is a truncated or foreign file — treating it
    # as active forever would 503 every chat and suppress self-audits
    # until someone deletes it.
    return until_ts > 0.0 and time.time() <= until_ts


@dataclass(frozen=True)
class LocalAuditGpu:
    """One local CUDA GPU's audit assignment for a selected window."""

    ordinal: int
    local_gpu_index: int
    gpu_name: str
    vram_gb: int


@dataclass(frozen=True)
class MeshAuditWindow:
    audit_id: str
    selection_block: int
    audit_block: int
    proof_challenge_block: int
    cohort_seed: str
    epoch_number: int
    gpus: tuple[LocalAuditGpu, ...]
    gpu_class_name: str
    passes: int
    deadline_s: float
    workload_spec: dict = field(default_factory=dict)


@dataclass
class _PreparedGpuAudit:
    gpu: LocalAuditGpu
    #: subprocess.Popen (fallback bench child) or _HolderBenchHandle (bench
    #: running inside the resident holder); same poll/communicate surface.
    proc: Any
    out_dir: Path
    challenge_file: Path
    start_file: Path
    ready_file: Path


#: Total VRAM footprint one workspace holder child pins per GPU: sized to
#: the measured ready-state of the combined bench (~1.7GB incl. its torch
#: CUDA context) plus margin. llama.cpp grows into ALL free VRAM over time
#: (observed 22.1->23.6GB on a 24GB card even at ub 512), so launch-time
#: reservations decay; only memory that is ALREADY ALLOCATED survives.
AUDIT_WORKSPACE_HOLD_MB = 1300
#: Free VRAM (beyond the hold) a GPU must have before a holder spawns; a
#: card too full for the holder falls back to the reserve/descent path.
AUDIT_WORKSPACE_HOLD_MARGIN_MB = 600

#: Audit children whose exit statuses their Popen owners still need. The
#: daemon's adopted-orphan zombie sweep (pool.reap_untracked_zombies) must
#: never steal these; weak refs mean a Popen nobody can reach anymore stops
#: shielding its pid automatically.
_TRACKED_CHILD_PROCS: "weakref.WeakSet[subprocess.Popen]" = weakref.WeakSet()


def _track_child(proc: subprocess.Popen) -> subprocess.Popen:
    """Register a spawned audit child so the daemon zombie sweep skips it."""

    try:
        _TRACKED_CHILD_PROCS.add(proc)
    except Exception:
        pass
    return proc


def tracked_child_pids() -> set[int]:
    """Pids of live-owned audit children (holders + fallback bench procs)."""

    pids: set[int] = set()
    for proc in list(_TRACKED_CHILD_PROCS):
        pid = getattr(proc, "pid", None)
        if pid is not None:
            pids.add(int(pid))
    return pids


def _holder_command(hold_mb: int) -> list[str]:
    """Command line for one resident holder child (verallm.mesh.vram_holder).

    The child pins the workspace AND executes audit benches in-process on
    command, so the workspace pages never leave its torch caching
    allocator during a window handoff (see the module docstring there).
    """
    return [
        sys.executable,
        "-m",
        "verallm.mesh.vram_holder",
        str(int(hold_mb)),
        str(os.getpid()),
    ]


class _HolderProc:
    """One resident holder child plus its pipe reader threads.

    The readers drain stdout/stderr continuously: the holder emits
    protocol events mid-life (and the in-process bench may log), so a
    blocking pipe would wedge the child. Never call ``communicate()`` on
    the wrapped Popen — it races the readers; use ``kill_and_wait``.
    """

    def __init__(self, proc: subprocess.Popen, local_gpu_index: int) -> None:
        self.proc = proc
        self.local_gpu_index = int(local_gpu_index)
        self.held = threading.Event()
        self.bench_requested = False
        self.bench_started = threading.Event()
        self.bench_done = threading.Event()
        self.bench_rc: Optional[int] = None
        self.bench_error = ""
        self.bench_held_after = True
        self.busy = False
        self.state_lock = threading.Lock()
        self.stdout_tail: deque[str] = deque(maxlen=50)
        self.stderr_tail: deque[str] = deque(maxlen=100)
        threading.Thread(
            target=self._read_stdout,
            name=f"vram-holder-out-{self.local_gpu_index}",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            name=f"vram-holder-err-{self.local_gpu_index}",
            daemon=True,
        ).start()

    def _read_stdout(self) -> None:
        from verallm.mesh.vram_holder import EVENT_PREFIX

        try:
            for raw in self.proc.stdout:  # type: ignore[union-attr]
                line = raw.rstrip("\n")
                if line.strip() == "held":
                    self.held.set()
                    continue
                if line.startswith(EVENT_PREFIX):
                    try:
                        event = json.loads(line[len(EVENT_PREFIX):])
                    except Exception:
                        continue
                    name = str(event.get("event", "") or "")
                    if name == "bench_started":
                        self.bench_started.set()
                    elif name == "bench_done":
                        with self.state_lock:
                            self.bench_rc = int(event.get("rc", 1) or 0)
                            self.bench_error = str(event.get("error") or "")
                            self.bench_held_after = bool(event.get("held"))
                            self.busy = False
                        self.bench_done.set()
                    continue
                self.stdout_tail.append(line)
        except Exception:
            pass
        finally:
            # Pipe EOF: the holder is gone. A bench in flight can never
            # complete now — surface a failure instead of hanging waiters.
            with self.state_lock:
                if self.bench_requested and not self.bench_done.is_set():
                    rc = self.proc.poll()
                    self.bench_rc = 1 if rc in (None, 0) else int(rc)
                    self.bench_error = (
                        self.bench_error or "holder exited during bench"
                    )
                    self.bench_held_after = False
                    self.busy = False
                    self.bench_done.set()

    def _read_stderr(self) -> None:
        try:
            for raw in self.proc.stderr:  # type: ignore[union-attr]
                self.stderr_tail.append(raw.rstrip("\n"))
        except Exception:
            pass

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill_and_wait(self, timeout: float = 15.0) -> None:
        for stream in (self.proc.stdin,):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except Exception:
            pass


class _HolderBenchHandle:
    """Popen-shaped view of a bench running inside a resident holder.

    Exposes exactly the surface the audit worker uses on a bench child
    (poll/returncode/communicate/terminate/kill). Completion means the
    BENCH finished — the holder process stays alive, already re-holding
    the workspace, so there is no re-hold gap for the serve to grow into.
    """

    def __init__(
        self,
        holders: "WorkspaceHolders",
        local_gpu_index: int,
        holder: _HolderProc,
    ) -> None:
        self._holders = holders
        self._local_gpu_index = int(local_gpu_index)
        self._holder = holder

    @property
    def returncode(self) -> Optional[int]:
        return self.poll()

    def poll(self) -> Optional[int]:
        holder = self._holder
        if holder.bench_done.is_set() and holder.bench_rc is not None:
            return holder.bench_rc
        rc = holder.proc.poll()
        if rc is not None:
            # The holder died before the bench finished.
            return int(rc) if rc != 0 else 1
        return None

    def communicate(
        self, timeout: Optional[float] = None
    ) -> tuple[str, str]:
        if not self._holder.bench_done.wait(timeout):
            raise subprocess.TimeoutExpired(
                cmd="holder-bench", timeout=float(timeout or 0.0)
            )
        stdout = "\n".join(self._holder.stdout_tail)
        stderr_lines = list(self._holder.stderr_tail)
        if self._holder.bench_error:
            stderr_lines.append(self._holder.bench_error)
        return stdout, "\n".join(stderr_lines)

    def terminate(self) -> None:
        # There is no way to stop only the in-process bench; discarding
        # the holder is the abort path, and ensure() replaces it.
        self._holders.discard(self._local_gpu_index, self._holder)

    def kill(self) -> None:
        self.terminate()


def _cuda_device_for_local(local_gpu_index: int) -> str:
    """Map a worker-local GPU index through the parent's CUDA mask.

    A worker pinned to a GPU subset (CUDA_VISIBLE_DEVICES="2,3") numbers
    its local GPUs 0..N-1; a child masked with the bare local index would
    land on a foreign physical GPU.
    """
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if mask:
        entries = [e.strip() for e in mask.split(",") if e.strip()]
        if 0 <= int(local_gpu_index) < len(entries):
            return entries[int(local_gpu_index)]
    return str(int(local_gpu_index))


def _physical_gpu_index(local_gpu_index: int) -> Optional[int]:
    """Physical nvidia-smi index for a worker-local GPU, when derivable."""
    device = _cuda_device_for_local(local_gpu_index)
    try:
        return int(device)
    except (TypeError, ValueError):
        return None


class WorkspaceHolders:
    """Resident per-GPU placeholder processes for the audit workspace.

    Held from daemon start so the serving stack (llama KV auto-fit, lazy
    graph growth) sizes itself around the audit workspace instead of
    racing it. At a window the bench runs INSIDE the holder process
    (``run_bench``): the workspace pages move hold-buffer -> bench
    tensors -> hold-buffer within one torch caching allocator and are
    never released to CUDA, so the serve cannot grow into the handoff.
    The kill-holder/spawn-bench swap this replaced left the workspace
    unowned for the seconds each fresh interpreter needed to boot, and
    llama ratcheted into it a little more at every window until benches
    OOM'd.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: dict[int, _HolderProc] = {}
        self._pending: set[int] = set()
        self._env_builder: Callable[[int], dict[str, str]] | None = None

    def configure(
        self, env_builder: Callable[[int], dict[str, str]]
    ) -> None:
        with self._lock:
            self._env_builder = env_builder

    def _spawn(self, local_gpu_index: int) -> Optional[_HolderProc]:
        """Spawn one holder child. RUNS WITHOUT THE LOCK.

        CUDA init in the child can hang on a wedged GPU; a blocking wait
        under the process-wide lock turned that into a whole-daemon
        deadlock risk (every serve launch consults holder state). The
        ready wait is event-based with a hard deadline; the reader
        threads own the pipes for the holder's whole life.
        """
        env = (
            self._env_builder(local_gpu_index)
            if self._env_builder is not None
            else {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": _cuda_device_for_local(
                    local_gpu_index
                ),
            }
        )
        try:
            proc = _track_child(
                subprocess.Popen(
                    _holder_command(AUDIT_WORKSPACE_HOLD_MB),
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        except Exception as exc:
            logger.warning(
                "audit workspace holder failed to spawn: gpu=%s %s",
                local_gpu_index,
                exc,
            )
            return None
        holder = _HolderProc(proc, local_gpu_index)
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if holder.held.wait(1.0):
                break
            if proc.poll() is not None:
                break
        if not holder.held.is_set():
            holder.kill_and_wait(timeout=10.0)
            logger.warning(
                "audit workspace holder did not allocate: gpu=%s "
                "stderr_tail=%s",
                local_gpu_index,
                "\n".join(holder.stderr_tail)[-200:],
            )
            return None
        logger.info(
            "audit workspace holder resident: gpu=%s hold_mb=%s",
            local_gpu_index,
            AUDIT_WORKSPACE_HOLD_MB,
        )
        return holder

    def ensure(self, local_gpu_indexes: list[int]) -> None:
        """Hold the workspace on every listed GPU that lacks a live holder.

        Spawns OUTSIDE the lock (CUDA init can hang; see _spawn). A GPU
        without enough free VRAM for hold+margin is skipped with a loud
        warning — the launch-time reserve stays the guard there.
        """
        per_gpu_free: list[int] = []
        try:
            from verallm.mesh.pool import _gpu_free_vram_mb

            per_gpu_free = _gpu_free_vram_mb()
        except Exception:
            per_gpu_free = []
        needed = AUDIT_WORKSPACE_HOLD_MB + AUDIT_WORKSPACE_HOLD_MARGIN_MB
        todo: list[int] = []
        with self._lock:
            for idx in local_gpu_indexes:
                idx = int(idx)
                holder = self._procs.get(idx)
                if holder is not None and holder.alive():
                    continue
                if idx in self._pending:
                    continue
                self._procs.pop(idx, None)
                self._pending.add(idx)
                todo.append(idx)
        for idx in todo:
            phys = _physical_gpu_index(idx)
            if (
                phys is not None
                and 0 <= phys < len(per_gpu_free)
                and per_gpu_free[phys] < needed
            ):
                logger.warning(
                    "audit workspace holder skipped: gpu=%s free=%dMB "
                    "< %dMB needed; capacity audits on this GPU depend "
                    "on the launch-time reserve alone",
                    idx,
                    per_gpu_free[phys],
                    needed,
                )
                with self._lock:
                    self._pending.discard(idx)
                continue
            holder = self._spawn(idx)
            with self._lock:
                self._pending.discard(idx)
                if holder is not None:
                    self._procs[idx] = holder

    def run_bench(
        self,
        local_gpu_index: int,
        *,
        argv: list[str],
        sys_path: list[str],
        lease_id: str,
        start_timeout_s: float = 90.0,
    ) -> Optional[_HolderBenchHandle]:
        """Run one audit bench INSIDE the resident holder; None = no holder.

        Returns once the bench has taken over the workspace (or already
        failed inside the holder). A holder that never acknowledges the
        command within the deadline is discarded so the caller's fallback
        path can spawn a plain bench child.
        """
        idx = int(local_gpu_index)
        with self._lock:
            holder = self._procs.get(idx)
            if holder is None or not holder.alive():
                return None
            with holder.state_lock:
                if holder.busy:
                    logger.warning(
                        "audit workspace holder already benching: gpu=%s",
                        idx,
                    )
                    return None
                holder.busy = True
                holder.bench_requested = True
                holder.bench_started.clear()
                holder.bench_done.clear()
                holder.bench_rc = None
                holder.bench_error = ""
        command = {
            "cmd": "bench",
            "argv": [str(a) for a in argv],
            "sys_path": [str(p) for p in sys_path],
            "lease_id": str(lease_id),
        }
        try:
            holder.proc.stdin.write(  # type: ignore[union-attr]
                json.dumps(command, sort_keys=True) + "\n"
            )
            holder.proc.stdin.flush()  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "audit workspace holder bench dispatch failed: gpu=%s %s",
                idx,
                exc,
            )
            self.discard(idx, holder)
            return None
        deadline = time.time() + max(1.0, float(start_timeout_s))
        while time.time() < deadline:
            if holder.bench_started.wait(0.25) or holder.bench_done.is_set():
                return _HolderBenchHandle(self, idx, holder)
            if not holder.alive():
                break
        logger.warning(
            "audit workspace holder never acknowledged the bench: gpu=%s",
            idx,
        )
        self.discard(idx, holder)
        return None

    def discard(self, local_gpu_index: int, holder: _HolderProc) -> None:
        """Kill one specific holder and forget it (abort/replace path)."""
        with self._lock:
            if self._procs.get(int(local_gpu_index)) is holder:
                self._procs.pop(int(local_gpu_index), None)
        holder.kill_and_wait()

    def release(self, local_gpu_index: int) -> None:
        """Free one GPU's held workspace (legacy bench-child fallback only)."""
        with self._lock:
            holder = self._procs.pop(int(local_gpu_index), None)
        if holder is None:
            return
        holder.kill_and_wait()

    def rehold(self, local_gpu_index: int) -> None:
        self.ensure([int(local_gpu_index)])

    def active(self, local_gpu_index: int) -> bool:
        with self._lock:
            holder = self._procs.get(int(local_gpu_index))
            return holder is not None and holder.alive()

    def stop_all(self) -> None:
        with self._lock:
            holders = list(self._procs.values())
            self._procs.clear()
        for holder in holders:
            holder.kill_and_wait(timeout=10.0)


#: One delegated-sign request in flight per daemon (see MeshCapacityAuditWorker._sign).
_SIGN_SERIALIZER = threading.Lock()

_WORKSPACE_HOLDERS = WorkspaceHolders()


def workspace_holders() -> WorkspaceHolders:
    """Process-wide holder registry shared by daemon and audit worker."""
    return _WORKSPACE_HOLDERS


class MeshCapacityAuditWorker:
    """Background capacity-audit worker for one mesh pool worker daemon."""

    def __init__(
        self,
        *,
        context: Mapping[str, Any],
        workdir: str | os.PathLike[str],
        repo_root: str | os.PathLike[str],
        sign_artifact_remote: Callable[[dict[str, Any]], dict[str, Any]],
        subtensor_network: str,
        netuid: int,
        worker_id: str,
        manual_validator_urls: tuple[str, ...] = (),
        poll_interval_s: float = 5.0,
    ) -> None:
        slot_ctx = dict(context.get("slot") or {})
        self.slot_ctx = slot_ctx
        self.roster = dict(context.get("roster") or {})
        self.roster_signature = str(context.get("roster_signature", "") or "")
        self.local_gpus = tuple(
            LocalAuditGpu(
                ordinal=int(gpu.get("ordinal", -1)),
                local_gpu_index=int(gpu.get("local_gpu_index", -1)),
                gpu_name=str(gpu.get("gpu_name", "") or ""),
                vram_gb=int(gpu.get("vram_gb", 0) or 0),
            )
            for gpu in (context.get("local_gpus") or [])
        )
        self.workdir = Path(workdir)
        self.repo_root = Path(repo_root)
        self.sign_artifact_remote = sign_artifact_remote
        self.subtensor_network = str(subtensor_network or "")
        self.netuid = int(netuid)
        self.worker_id = str(worker_id or "")
        self.poll_interval_s = max(0.5, float(poll_interval_s))
        self.drain_file = capacity_drain_file_path(self.workdir)

        from verallm.mesh.capacity_roster import roster_digest

        self.roster_digest = roster_digest(self.roster) if self.roster else ""

        # The neuron-shaped config the protocol helpers expect. All
        # capacity_audit_* policy fields resolve to their defaults until the
        # owner-published runtime subnet config is applied; without an
        # authoritative runtime config the audit stays OFF (enabled defaults
        # False), which is the safe dark state.

        # subnet_config_url MUST resolve per network: the client's module
        # default is the MAINNET URL, and a testnet worker running mainnet
        # runtime parameters (epoch_blocks and friends) derives disjoint
        # window schedules and epoch numbering from its validator — every
        # audit then silently expires as no_show. The chain slot's EVM chain_id is the
        # authoritative network identity (works even when the daemon was
        # launched with a raw ws:// endpoint instead of a network name).
        chain_id = int(slot_ctx.get("chain_id", 0) or 0)
        self.config = SimpleNamespace(
            netuid=self.netuid,
            chain_id=chain_id,
            subtensor_network=self.subtensor_network,
            epoch_blocks=360,
            subnet_config_url=self._resolve_subnet_config_url(
                chain_id=chain_id,
                subtensor_network=self.subtensor_network,
                context_url=str(context.get("subnet_config_url", "") or ""),
            ),
        )
        self._runtime_client = None
        self._runtime_key = None
        self._runtime_authoritative = False
        self.runtime_cfg = None  # set by _refresh_runtime_config

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._seen_audits: set[str] = set()
        self._audit_start_lock = threading.Lock()
        self._busy_selection_block_until = 0
        self._current_block_seen = 0
        self._active_audit_id = ""
        self._active_audit_until_ts = 0.0
        self._workspace_ready = False
        self._workspace_error = ""
        self._endpoint_resolver = None

    @staticmethod
    def _resolve_subnet_config_url(
        *,
        chain_id: int,
        subtensor_network: str,
        context_url: str,
    ) -> str:
        """Network-scoped runtime-config URL, strongest signal first.

        Priority: operator env override, manager-delivered context URL,
        the slot's EVM chain_id (945 = bittensor testnet), then the
        network-name mapping as the final fallback.
        """
        env_url = os.environ.get("VERATHOS_SUBNET_CONFIG_URL", "").strip()
        if env_url:
            return env_url
        if context_url.strip():
            return context_url.strip()
        from neurons.config import (
            TESTNET_SUBNET_CONFIG_URL,
            default_subnet_config_url,
        )

        if int(chain_id) == 945:
            return TESTNET_SUBNET_CONFIG_URL
        return default_subnet_config_url(subtensor_network)

    # ── lifecycle ────────────────────────────────────────────────

    def _publish_roster_file(self) -> None:
        """Publish the signed roster for the serve's GET /capacity/roster.

        Validators learn a mesh's GPU obligation by PULL, and the all-Metal
        exemption depends on SEEING an all-Metal roster: publication must
        not be gated on this worker actually auditing. It used to sit after
        the CUDA preflight, so Metal-only workers never exposed a roster and
        the model gate zeroed their entries as roster-less.
        """

        if not (self.roster and self.roster_signature):
            return
        roster_path = capacity_roster_file_path(self.workdir)
        try:
            tmp = roster_path.with_suffix(roster_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "roster": self.roster,
                        "roster_signature": self.roster_signature,
                    },
                    sort_keys=True,
                )
            )
            os.replace(tmp, roster_path)
        except Exception as exc:
            logger.warning("capacity roster file write failed: %s", exc)

    def start(self) -> bool:
        """Start the audit thread. Returns False (and logs) when disabled."""

        # Roster publication is eligibility data for EVERY backend; the
        # audit thread below stays CUDA-only.
        self._publish_roster_file()
        if not self.local_gpus:
            logger.info(
                "mesh capacity audit worker not started: no local CUDA GPUs "
                "in the roster assignment"
            )
            return False
        if not self.slot_ctx.get("address") or self.slot_ctx.get(
            "model_index"
        ) is None:
            logger.warning(
                "mesh capacity audit worker not started: audit context has "
                "no chain slot"
            )
            return False
        if not self.subtensor_network:
            logger.warning(
                "mesh capacity audit worker not started: no subtensor "
                "network (manager did not send chain coordinates at join)"
            )
            return False
        ok, error = self._preflight_workspace()
        if not ok:
            if os.environ.get("VERATHOS_ALLOW_NO_CAPACITY_AUDITS", "") == "1":
                logger.error(
                    "mesh capacity audit worker disabled: hot-capacity "
                    "workspace unavailable: %s (explicitly allowed via "
                    "VERATHOS_ALLOW_NO_CAPACITY_AUDITS=1)",
                    error[:300],
                )
                return False
            # Fail-closed: a subnet worker that serves without a working
            # audit workspace runs audit-blind, records no evidence, and
            # probates with nothing in its own logs explaining why. Dying
            # loudly at start is the only honest behavior; light-tier or
            # dev setups opt out explicitly with the env above.
            raise CapacityAuditWorkspaceError(
                "hot-capacity audit workspace unavailable on a subnet "
                f"worker: {error[:300]} - install the matching "
                "hot_capacity_workspace_cuda wheel from dist/ (the join "
                "script does this when dist/ is present), or set "
                "VERATHOS_ALLOW_NO_CAPACITY_AUDITS=1 to serve without "
                "capacity audits (the slot will accumulate strikes)"
            )
        # Ingest reachability probe. A worker that can run every audit but
        # cannot DELIVER the evidence is scored as a no-show and ends up
        # on probation with nothing in its own logs explaining why; probe
        # the return path once at startup so a broken egress is loud
        # before the first window, not after the first probation.
        self._probe_ingest_reachability()
        # Roster already published at the top of start() (backend-agnostic);
        # republish here in case it changed between the preflight and now.
        self._publish_roster_file()
        # Resident workspace holders: give them the exact per-GPU env the
        # bench children use, then make sure every audited GPU is held.
        # (The daemon also holds at startup, BEFORE the serve launches;
        # this covers audit contexts that name GPUs the daemon missed.)
        def _holder_env(local_gpu_index: int) -> dict[str, str]:
            probe = LocalAuditGpu(
                ordinal=-1,
                local_gpu_index=int(local_gpu_index),
                gpu_name="",
                vram_gb=0,
            )
            return self._workspace_env(probe)

        holders = workspace_holders()
        holders.configure(_holder_env)
        holders.ensure(
            [gpu.local_gpu_index for gpu in self.local_gpus]
        )
        self._refresh_runtime_config(force=True)
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="mesh-capacity-audit",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "mesh capacity audit worker started: model_index=%s gpus=%s "
            "ordinals=%s",
            self.slot_ctx.get("model_index"),
            len(self.local_gpus),
            [gpu.ordinal for gpu in self.local_gpus],
        )
        return True

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    # ── runtime config ───────────────────────────────────────────

    def _refresh_runtime_config(
        self,
        *,
        current_epoch: int | None = None,
        current_block: int | None = None,
        force: bool = False,
    ) -> None:
        from neurons.subnet_runtime_config import (
            RuntimeSubnetConfigClient,
            apply_runtime_config_to_neuron_config,
            capacity_audit_config_from_neuron_config,
        )

        if self._runtime_client is None:
            self._runtime_client = RuntimeSubnetConfigClient.from_config(
                self.config, log=logger
            )
        runtime = None
        try:
            runtime = self._runtime_client.get(
                current_epoch=current_epoch,
                current_block=current_block,
                force=force,
            )
        except Exception as exc:
            logger.debug("runtime subnet config fetch failed: %s", exc)
        if runtime is None:
            self._runtime_authoritative = False
        else:
            self._runtime_authoritative = True
            if runtime.cache_key != self._runtime_key:
                apply_runtime_config_to_neuron_config(runtime, self.config)
                self._runtime_key = runtime.cache_key
                logger.info(
                    "mesh capacity audit applied runtime config version=%s "
                    "effective_epoch=%s",
                    runtime.version,
                    runtime.effective_epoch,
                )
        self.runtime_cfg = capacity_audit_config_from_neuron_config(self.config)

    def _epoch_blocks(self) -> int:
        return max(1, int(getattr(self.config, "epoch_blocks", 360) or 360))

    def _audit_enabled(self) -> bool:
        return bool(
            self._runtime_authoritative
            and self.runtime_cfg is not None
            and getattr(self.runtime_cfg, "enabled", False)
        )

    # ── chain head poller (D4: poll, not subscribe) ──────────────

    def _subtensor(self):
        import bittensor as bt

        from verallm.mesh.allowlist import normalize_subtensor_network

        SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
        return SubtensorCls(
            network=normalize_subtensor_network(self.subtensor_network)
        )

    @staticmethod
    def _close_subtensor(subtensor) -> None:
        from neurons.subtensor_connection import close_owned_subtensor

        close_owned_subtensor(subtensor)

    @staticmethod
    def _coerce_block_hash(raw: object) -> Optional[bytes]:
        if isinstance(raw, bytes):
            return raw if len(raw) == 32 else None
        if raw is None:
            return None
        text = str(raw).strip()
        if text.startswith("0x"):
            text = text[2:]
        if len(text) != 64:
            return None
        try:
            return bytes.fromhex(text)
        except ValueError:
            return None

    def _get_block_hash(self, subtensor, block_number: int) -> Optional[bytes]:
        substrate = getattr(subtensor, "substrate", None)
        if substrate is not None:
            try:
                response = substrate.rpc_request(
                    "chain_getBlockHash", [int(block_number)]
                )
                raw = (
                    response.get("result")
                    if isinstance(response, dict)
                    else response
                )
                normalized = self._coerce_block_hash(raw)
                if normalized is not None:
                    return normalized
            except Exception:
                pass
        try:
            return self._coerce_block_hash(
                subtensor.get_block_hash(int(block_number))
            )
        except Exception:
            return None

    def _get_current_head(self, subtensor) -> int:
        substrate = getattr(subtensor, "substrate", None)
        if substrate is not None:
            try:
                response = substrate.rpc_request("chain_getHeader", [])
                header = (
                    response.get("result")
                    if isinstance(response, dict)
                    else response
                )
                if isinstance(header, dict):
                    number = header.get("number")
                    if number is not None:
                        return int(number, 0) if isinstance(number, str) else int(number)
            except Exception:
                pass
        try:
            return int(subtensor.get_current_block())
        except Exception:
            return 0

    def _run(self) -> None:
        last_block = 0
        while self._running:
            subtensor = None
            try:
                subtensor = self._subtensor()
                while self._running:
                    current = self._get_current_head(subtensor)
                    if current <= 0:
                        raise RuntimeError("chain head unavailable")
                    if last_block <= 0:
                        last_block = current - 1
                    # Bounded catch-up: after an outage longer than a few
                    # blocks the missed selections are unrecoverable anyway
                    # (their windows have passed), so skip ahead instead of
                    # replaying stale blocks as if they were current.
                    if current - last_block > 8:
                        last_block = current - 1
                    for block_number in range(last_block + 1, current + 1):
                        if not self._running:
                            break
                        block_hash = self._get_block_hash(subtensor, block_number)
                        if block_hash is None:
                            break
                        self._current_block_seen = max(
                            self._current_block_seen, block_number
                        )
                        self._on_block(block_number, block_hash, subtensor)
                        last_block = block_number
                    if self._stop_event.wait(self.poll_interval_s):
                        return
            except Exception as exc:
                logger.debug("mesh capacity audit head poll failed: %s", exc)
                if self._stop_event.wait(
                    min(60.0, self.poll_interval_s * 4.0)
                ):
                    return
            finally:
                if subtensor is not None:
                    self._close_subtensor(subtensor)

    # ── selection ────────────────────────────────────────────────

    def _capacity_slot(self):
        from neurons.capacity_audit import CapacitySlot, build_capacity_slot_group_key

        slot_ctx = self.slot_ctx
        return CapacitySlot(
            chain_id=int(slot_ctx.get("chain_id", 0) or 0),
            netuid=int(slot_ctx.get("netuid", 0) or 0),
            address=str(slot_ctx.get("address", "") or "").lower(),
            model_index=int(slot_ctx.get("model_index", 0) or 0),
            endpoint=str(slot_ctx.get("endpoint", "") or ""),
            model_id=str(slot_ctx.get("model_id", "") or ""),
            quant=str(slot_ctx.get("quant", "") or ""),
            max_context_len=int(slot_ctx.get("max_context_len", 0) or 0),
            group_key=build_capacity_slot_group_key(
                address=str(slot_ctx.get("address", "") or "").lower(),
                endpoint=str(slot_ctx.get("endpoint", "") or ""),
                model_id=str(slot_ctx.get("model_id", "") or ""),
            ),
        )

    def _selection_roster(self, epoch_number: int) -> Optional[dict[str, Any]]:
        """The roster used for group-key selection, or None.

        Determinism contract shared with the validator: only a roster frozen
        BEFORE the current epoch contributes grouping tokens, so both sides
        of the non-interactive protocol group on the same document.
        """

        if not self.roster:
            return None
        roster_epoch = int(self.roster.get("roster_epoch", -1))
        if roster_epoch < 0 or roster_epoch > int(epoch_number) - 1:
            return None
        return self.roster

    def _selection_hashes(
        self,
        selection_block: int,
        selection_block_hash: bytes,
        subtensor,
    ) -> Optional[list[bytes]]:
        count = max(1, int(getattr(self.runtime_cfg, "beacon_hash_count", 1) or 1))
        hashes = [selection_block_hash]
        if count <= 1 or subtensor is None:
            return hashes
        for offset in range(1, count):
            block_hash = self._get_block_hash(
                subtensor, int(selection_block) - offset
            )
            if block_hash is None:
                return None
            hashes.append(block_hash)
        return hashes

    def _has_active_local_audit(self) -> bool:
        if not self._active_audit_id:
            return False
        until_ts = float(self._active_audit_until_ts or 0.0)
        if until_ts > 0.0 and time.time() > until_ts:
            self._active_audit_id = ""
            self._active_audit_until_ts = 0.0
            return False
        return True

    def _on_block(self, block_number: int, block_hash: bytes, subtensor) -> None:
        from neurons.capacity_audit import (
            capacity_audit_slot_selected,
            capacity_audit_window_fits_epoch,
            capacity_audit_window_triggered,
            capacity_gpu_pass_count,
            capacity_gpu_workload_spec,
            derive_audit_id,
            derive_audit_seed,
            derive_audit_seed_from_hashes,
            match_gpu_class,
        )

        # Startup may have fetched a valid config whose effective epoch could
        # not yet be evaluated because no chain block was known. Resolve that
        # authenticated candidate against its own epoch width before deriving
        # this first window. This is a config-cache operation in the normal
        # path; it performs no additional chain RPC.
        if not self._runtime_authoritative:
            self._refresh_runtime_config(current_block=block_number)

        epoch_blocks = self._epoch_blocks()
        if block_number % epoch_blocks == 0:
            self._refresh_runtime_config(
                current_epoch=int(block_number // epoch_blocks), force=True
            )
            epoch_blocks = self._epoch_blocks()
        if not self._audit_enabled():
            return
        cfg = self.runtime_cfg
        if not capacity_audit_window_triggered(
            block_number, block_hash, epoch_blocks, cfg
        ):
            return
        if not capacity_audit_window_fits_epoch(block_number, epoch_blocks, cfg):
            return
        selection_hashes = self._selection_hashes(
            block_number, block_hash, subtensor
        )
        if selection_hashes is None:
            return
        epoch_number = block_number // epoch_blocks
        if len(selection_hashes) <= 1:
            seed = derive_audit_seed(selection_hashes[0], epoch_number)
        else:
            seed = derive_audit_seed_from_hashes(selection_hashes, epoch_number)
        slot = self._capacity_slot()
        if not capacity_audit_slot_selected(
            slot, seed, cfg, self._selection_roster(epoch_number)
        ):
            return
        if self._has_active_local_audit():
            logger.info(
                "mesh capacity audit skipping window while one is active: "
                "B_select=%s",
                block_number,
            )
            return
        if int(block_number) <= int(self._busy_selection_block_until or 0):
            return

        # Per-GPU calibration: every local CUDA GPU must map to a calibrated
        # class or this worker cannot honour the opening. The validator's
        # model gate reasons the slot on uncalibrated roster GPUs anyway,
        # so skipping here loses nothing scoreable.
        gpu_rows = []
        for gpu in self.local_gpus:
            row = match_gpu_class(gpu.gpu_name, gpu.vram_gb, cfg)
            if row is None or not row.calibrated or capacity_gpu_pass_count(row) <= 0:
                logger.warning(
                    "mesh capacity audit selected but local GPU is not "
                    "calibrated: ordinal=%s gpu=%r vram=%s",
                    gpu.ordinal,
                    gpu.gpu_name,
                    gpu.vram_gb,
                )
                return
            gpu_rows.append((gpu, row))
        first_row = gpu_rows[0][1]

        audit_block = int(block_number + cfg.lead_blocks)
        proof_challenge_block = int(
            audit_block + max(1, int(cfg.proof_challenge_delay_blocks or 1))
        )
        audit_id = derive_audit_id(
            chain_id=slot.chain_id,
            netuid=slot.netuid,
            epoch_number=epoch_number,
            selection_block=block_number,
            audit_block=audit_block,
            cohort_seed=seed,
        )
        with self._audit_start_lock:
            if audit_id in self._seen_audits:
                return
        window = MeshAuditWindow(
            audit_id=audit_id,
            selection_block=block_number,
            audit_block=audit_block,
            proof_challenge_block=proof_challenge_block,
            cohort_seed=seed,
            epoch_number=epoch_number,
            gpus=tuple(gpu for gpu, _row in gpu_rows),
            gpu_class_name=str(first_row.match_gpu_name),
            passes=capacity_gpu_pass_count(first_row),
            deadline_s=float(first_row.deadline_s or cfg.deadline_s),
            workload_spec=capacity_gpu_workload_spec(first_row),
        )
        self._busy_selection_block_until = max(
            int(self._busy_selection_block_until or 0),
            proof_challenge_block + 1,
        )
        self._mark_audit_drain(window, phase="selected")
        threading.Thread(
            target=self._await_and_run_window,
            args=(window,),
            name=f"mesh-capacity-start-{audit_id[:12]}",
            daemon=True,
        ).start()
        logger.info(
            "mesh capacity audit selected: audit_id=%s B_select=%s B_start=%s "
            "gpus=%s",
            audit_id[:12],
            block_number,
            audit_block,
            len(window.gpus),
        )

    # ── drain state ──────────────────────────────────────────────

    def _window_until_ts(self, window: MeshAuditWindow) -> float:
        cfg = self.runtime_cfg
        block_budget_s = (
            max(0, window.audit_block - window.selection_block) * _BLOCK_TIME_S
            + max(0, window.proof_challenge_block - window.audit_block)
            * _BLOCK_TIME_S
        )
        evidence_budget_s = (
            float(getattr(cfg, "deadline_s", 0.0) or 0.0)
            + float(getattr(cfg, "transport_grace_s", 0.0) or 0.0)
            + float(getattr(cfg, "payload_deadline_s", 0.0) or 0.0)
        )
        return time.time() + max(
            float(getattr(cfg, "drain_seconds", 0.0) or 0.0),
            block_budget_s + evidence_budget_s + 30.0,
            60.0,
        )

    def _write_drain_state(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at"] = time.time()
        tmp = self.drain_file.with_suffix(self.drain_file.suffix + ".tmp")
        try:
            self.drain_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, sort_keys=True))
            os.replace(tmp, self.drain_file)
        except Exception as exc:
            logger.warning("capacity drain state write failed: %s", exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def _mark_audit_drain(self, window: MeshAuditWindow, *, phase: str) -> None:
        until_ts = self._window_until_ts(window)
        self._active_audit_id = window.audit_id
        self._active_audit_until_ts = max(
            float(self._active_audit_until_ts or 0.0), until_ts
        )
        self._write_drain_state(
            {
                "active": True,
                "reason": "capacity_audit",
                "phase": phase,
                "audit_id": window.audit_id,
                "until_ts": until_ts,
            }
        )

    def _clear_audit_drain(self, audit_id: str) -> None:
        if not self._active_audit_id or self._active_audit_id == audit_id:
            self._active_audit_id = ""
            self._active_audit_until_ts = 0.0
        self._write_drain_state({"active": False, "last_audit_id": audit_id})

    # ── workspace subprocesses ───────────────────────────────────

    def _workspace_script(self) -> Path:
        return (
            self.repo_root
            / "scripts"
            / "hot_capacity_workspace"
            / "bench_combined.py"
        )

    def _workspace_command(self, script: Path) -> list[str]:
        if script.exists():
            return [sys.executable, str(script)]
        return [
            sys.executable,
            "-c",
            "from hot_capacity_workspace.bench_combined import main; main()",
        ]

    def _workspace_env(self, gpu: LocalAuditGpu) -> dict[str, str]:
        env = os.environ.copy()
        script_dir = self._workspace_script().parent
        if script_dir.exists():
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{script_dir}:{current}" if current else str(script_dir)
            )
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.exists():
                current = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{torch_lib}:{current}" if current else str(torch_lib)
                )
        except Exception:
            pass
        # One subprocess per GPU: mask exactly this roster entry's device,
        # mapped through the parent's own CUDA mask (a worker pinned to a
        # subset numbers its GPUs locally). Device selection rides this
        # mask + --device-index 0; the PROOF binding rides --gpu-index,
        # which carries the GLOBAL roster ordinal.
        env["CUDA_VISIBLE_DEVICES"] = _cuda_device_for_local(
            gpu.local_gpu_index
        )
        return env

    def _preflight_workspace(self) -> tuple[bool, str]:
        """Import-check the wheel once per daemon; disable, never crash."""

        if self._workspace_ready:
            return True, ""
        if self._workspace_error:
            return False, self._workspace_error
        script_dir = self._workspace_script().parent
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"p={str(script_dir)!r}; "
                "import pathlib; "
                "path=pathlib.Path(p); "
                "sys.path.insert(0, p) if path.exists() else None; "
                "import torch; "
                "import hot_capacity_workspace_cuda; "
            ),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=script_dir if script_dir.exists() else self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            self._workspace_error = str(exc)
            return False, self._workspace_error
        if proc.returncode == 0:
            self._workspace_ready = True
            return True, ""
        self._workspace_error = (proc.stderr or proc.stdout or "")[-500:]
        return False, self._workspace_error

    def _audit_lease(self, window: MeshAuditWindow) -> str:
        from neurons.capacity_audit import lease_id

        return lease_id(self._capacity_slot(), window.epoch_number)

    def _challenge_timeout_s(self, window: MeshAuditWindow) -> float:
        cfg = self.runtime_cfg
        return max(
            30.0,
            (window.proof_challenge_block - window.audit_block) * _BLOCK_TIME_S
            + float(getattr(cfg, "payload_deadline_s", 0.0) or 0.0),
        )

    def _prepare_gpu_audit(
        self,
        window: MeshAuditWindow,
        gpu: LocalAuditGpu,
        *,
        lease: str,
        start_timeout_s: float,
    ) -> Optional[_PreparedGpuAudit]:
        from neurons.capacity_audit_combined import (
            CURRENT_COMBINED_PROOF_PROTOCOL_VERSION,
        )

        out_dir = Path(
            tempfile.mkdtemp(
                prefix=f"verathos_mesh_capacity_g{gpu.ordinal}_"
            )
        )
        challenge_file = out_dir / f"{lease}_challenge.txt"
        start_file = out_dir / f"{lease}_start.json"
        ready_file = out_dir / f"{lease}_ready.json"
        script = self._workspace_script()
        bench_args = [
            "--child",
            "--out-dir", str(out_dir),
            "--lease-id", lease,
            # PROOF binding: --gpu-index enters every lane-seed derivation
            # (seed_for(lease, gpu_index, ...)) and must be the GLOBAL
            # roster ordinal the validator verifies with — passing 0 made
            # every ordinal>0 opening of a multi-GPU window an instant
            # invalid_payload. DEVICE selection
            # is separate: CUDA_VISIBLE_DEVICES masks one GPU and
            # --device-index defaults to 0 inside the mask.
            "--gpu-index", str(int(gpu.ordinal)),
            "--challenge-file", str(challenge_file),
            "--challenge-timeout-s", str(self._challenge_timeout_s(window)),
            "--ready-file", str(ready_file),
            "--start-file", str(start_file),
            "--start-timeout-s", str(max(1.0, float(start_timeout_s))),
        ]
        for key, value in dict(window.workload_spec or {}).items():
            if key in {"workload_version", "pass_count", "proof_protocol_version"}:
                continue
            bench_args.extend([f"--{key.replace('_', '-')}", str(value)])
        bench_args.extend(
            ["--proof-protocol-version", str(CURRENT_COMBINED_PROOF_PROTOCOL_VERSION)]
        )
        # Preferred path: run the bench INSIDE the resident holder. The
        # workspace pages move hold-buffer -> bench tensors -> hold-buffer
        # within one torch caching allocator and are never released to
        # CUDA, so the serve has NO gap to grow into (the old
        # kill-holder/spawn-child swap left the workspace unowned for the
        # seconds each interpreter boot took, and llama ratcheted into it
        # window after window until benches OOM'd).
        script_dir = script.parent
        handle = workspace_holders().run_bench(
            gpu.local_gpu_index,
            argv=bench_args,
            sys_path=[str(script_dir)] if script_dir.exists() else [],
            lease_id=lease,
        )
        if handle is not None:
            return _PreparedGpuAudit(
                gpu=gpu,
                proc=handle,
                out_dir=out_dir,
                challenge_file=challenge_file,
                start_file=start_file,
                ready_file=ready_file,
            )
        # Fallback (no resident holder, e.g. a GPU skipped as too full, or
        # a holder that failed mid-dispatch): legacy swap through a plain
        # bench child. There is no hold to protect on this GPU anyway.
        workspace_holders().release(gpu.local_gpu_index)
        try:
            proc = _track_child(
                subprocess.Popen(
                    [*self._workspace_command(script), *bench_args],
                    env=self._workspace_env(gpu),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        except Exception as exc:
            logger.warning(
                "mesh capacity workload failed to prelaunch: ordinal=%s %s",
                gpu.ordinal,
                exc,
            )
            # The holder was released for this spawn; take the space back
            # before the serve grows into it.
            workspace_holders().rehold(gpu.local_gpu_index)
            return None
        return _PreparedGpuAudit(
            gpu=gpu,
            proc=proc,
            out_dir=out_dir,
            challenge_file=challenge_file,
            start_file=start_file,
            ready_file=ready_file,
        )

    @staticmethod
    def _terminate_prepared(prepared: _PreparedGpuAudit) -> None:
        proc = prepared.proc
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            except Exception:
                pass
        # The bench child is gone either way; the holder takes the
        # workspace back before the serve can grow into it.
        workspace_holders().rehold(prepared.gpu.local_gpu_index)

    # ── window execution ─────────────────────────────────────────

    def _await_and_run_window(self, window: MeshAuditWindow) -> None:
        subtensor = None
        lease = self._audit_lease(window)
        lead_wait_s = max(
            0.0,
            float(window.audit_block - window.selection_block) * _BLOCK_TIME_S,
        )
        start_timeout_s = max(60.0, lead_wait_s + 60.0)
        prepared: list[_PreparedGpuAudit] = []
        try:
            for gpu in window.gpus:
                one = self._prepare_gpu_audit(
                    window, gpu, lease=lease, start_timeout_s=start_timeout_s
                )
                if one is None:
                    for other in prepared:
                        self._terminate_prepared(other)
                    self._clear_audit_drain(window.audit_id)
                    return
                prepared.append(one)

            deadline = time.time() + start_timeout_s
            audit_hash: Optional[bytes] = None
            while self._running and time.time() < deadline:
                try:
                    if subtensor is None:
                        subtensor = self._subtensor()
                    current = self._get_current_head(subtensor)
                    if current <= 0:
                        self._close_subtensor(subtensor)
                        subtensor = None
                        if self._stop_event.wait(0.5):
                            return
                        continue
                    if current >= window.audit_block:
                        audit_hash = self._get_block_hash(
                            subtensor, window.audit_block
                        )
                        if audit_hash is not None:
                            break
                        self._close_subtensor(subtensor)
                        subtensor = None
                except Exception:
                    self._close_subtensor(subtensor)
                    subtensor = None
                if self._stop_event.wait(0.5):
                    return
            if audit_hash is None:
                logger.warning(
                    "mesh capacity audit start wait timed out: audit_id=%s "
                    "B_start=%s",
                    window.audit_id[:12],
                    window.audit_block,
                )
                for one in prepared:
                    self._terminate_prepared(one)
                self._clear_audit_drain(window.audit_id)
                return
            self._run_window(window, audit_hash, prepared, lease, subtensor)
        except Exception as exc:
            logger.warning(
                "mesh capacity audit window failed: audit_id=%s %s",
                window.audit_id[:12],
                exc,
            )
            for one in prepared:
                self._terminate_prepared(one)
            self._clear_audit_drain(window.audit_id)
        finally:
            if subtensor is not None:
                self._close_subtensor(subtensor)
            # Single authoritative re-hold for EVERY window GPU: the
            # release/rehold pairing was unbalanced across early-exit
            # paths (Popen failure, pre-B_start death, stop-event exit)
            # and each miss left the GPU unprotected for the audit
            # worker's lifetime. ensure() is
            # idempotent for live holders, so double-rehold is free.
            try:
                workspace_holders().ensure(
                    [gpu.local_gpu_index for gpu in window.gpus]
                )
            except Exception:
                pass

    def _run_window(
        self,
        window: MeshAuditWindow,
        audit_hash: bytes,
        prepared: list[_PreparedGpuAudit],
        lease: str,
        subtensor,
    ) -> None:
        from neurons.capacity_audit import derive_proof_seed, slot_id

        with self._audit_start_lock:
            if window.audit_id in self._seen_audits:
                for one in prepared:
                    self._terminate_prepared(one)
                return
            self._seen_audits.add(window.audit_id)
        self._mark_audit_drain(window, phase="running")
        slot_identifier = slot_id(self._capacity_slot())

        released: list[_PreparedGpuAudit] = []
        for one in prepared:
            if one.proc.poll() is not None:
                stdout, stderr = one.proc.communicate()
                logger.warning(
                    "mesh capacity workload exited before B_start: "
                    "ordinal=%s rc=%s stderr_tail=%s",
                    one.gpu.ordinal,
                    one.proc.returncode,
                    (stderr or "")[-300:],
                )
                continue
            proof_seed = derive_proof_seed(
                audit_hash, slot_identifier, one.gpu.ordinal
            )
            start_payload = {
                "seed_hex": proof_seed,
                "audit_id": window.audit_id,
                "B_start": window.audit_block,
                "t": time.time(),
            }
            tmp_start = one.start_file.with_suffix(
                one.start_file.suffix + ".tmp"
            )
            tmp_start.write_text(
                json.dumps(start_payload, sort_keys=True) + "\n"
            )
            os.replace(tmp_start, one.start_file)
            released.append(one)
        logger.info(
            "mesh capacity audit released %d/%d openings: audit_id=%s",
            len(released),
            len(prepared),
            window.audit_id[:12],
        )
        # The drain exists for the TIMED workload: admissions stop at
        # selection so in-flight inference clears before B_start, and the
        # calibrated passes must own the GPUs until final timing is
        # captured. Everything AFTER that — waiting for the challenge
        # block, proof assembly (untimed, generous payload deadline), and
        # the WAN push — tolerates chat traffic, yet the drain used to
        # span it all: ~4 minutes of busy 503 per window on a healthy
        # box for ~30 seconds of protected compute; the drain protects
        # the timed workload only, never the aftermath. Each collector
        # reports the moment its GPU's timing is settled (final receipt
        # captured, or the workload conclusively failed); when every
        # opening has reported, chats resume while proofs assemble.
        timing_pending = {"n": len(released)}
        timing_lock = threading.Lock()

        def _timing_settled() -> None:
            with timing_lock:
                timing_pending["n"] -= 1
                done = timing_pending["n"] <= 0
            if done:
                self._clear_audit_drain(window.audit_id)

        collectors = [
            threading.Thread(
                target=self._collect_gpu_artifacts,
                args=(window, one, lease, slot_identifier, _timing_settled),
                name=f"mesh-capacity-collect-{one.gpu.ordinal}",
                daemon=True,
            )
            for one in released
        ]
        for thread in collectors:
            thread.start()
        for thread in collectors:
            thread.join()
        for one in released:
            workspace_holders().rehold(one.gpu.local_gpu_index)
        # Backstop (idempotent): a collector that died without reporting
        # must never leave the drain armed past the window.
        self._clear_audit_drain(window.audit_id)

    def _wait_for_challenge_seed(
        self,
        window: MeshAuditWindow,
        *,
        transcript: str,
        lease: str,
        slot_identifier: str,
        gpu_ordinal: int,
        timeout_s: float,
    ) -> Optional[str]:
        from neurons.capacity_audit import derive_proof_challenge_seed

        deadline = time.time() + max(1.0, float(timeout_s))
        subtensor = None
        try:
            while self._running and time.time() < deadline:
                try:
                    if subtensor is None:
                        subtensor = self._subtensor()
                    current = self._get_current_head(subtensor)
                    if current >= window.proof_challenge_block:
                        challenge_hash = self._get_block_hash(
                            subtensor, window.proof_challenge_block
                        )
                        if challenge_hash is not None:
                            return derive_proof_challenge_seed(
                                transcript,
                                challenge_hash,
                                lease,
                                slot_identifier,
                                gpu_ordinal,
                            )
                except Exception:
                    self._close_subtensor(subtensor)
                    subtensor = None
                if self._stop_event.wait(1.0):
                    return None
            return None
        finally:
            if subtensor is not None:
                self._close_subtensor(subtensor)

    # ── artifacts ────────────────────────────────────────────────

    def _base_artifact(
        self, window: MeshAuditWindow, gpu: LocalAuditGpu, slot_identifier: str
    ) -> dict[str, Any]:
        from neurons.capacity_audit import PROTOCOL_VERSION

        return {
            "protocol_version": PROTOCOL_VERSION,
            "audit_id": window.audit_id,
            "slot_id": slot_identifier,
            "address": str(self.slot_ctx.get("address", "") or "").lower(),
            "model_index": int(self.slot_ctx.get("model_index", 0) or 0),
            "claimed_gpu_class": window.gpu_class_name,
            # The GLOBAL roster ordinal: the validator derives seeds, the
            # sampled pass index and the proof challenge from this value,
            # so it must be the ordinal, never the local device index.
            "gpu_index": int(gpu.ordinal),
            "B_select": window.selection_block,
            "B_start": window.audit_block,
            "B_proof": window.proof_challenge_block,
            "pass_count": window.passes,
            "worker_id": self.worker_id,
            "local_gpu_index": int(gpu.local_gpu_index),
            "roster_digest": self.roster_digest,
        }

    def _sign(self, artifact: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Delegated signing: SERIALIZED per daemon, with short retries.

        Four collector threads signing simultaneously produced a connect
        burst that something between a WAN worker and the manager killed
        with TLS EOFs — reproducibly for the four ~254KiB payload signs,
        never for serial requests from the same box, and with no trace on
        the manager itself. One in-flight sign at a time removes the burst shape;
        the retries then have a healthy path to succeed on.
        """
        import contextlib

        # Only the large payload signs are serialized: they are the burst
        # that broke transport, and receipts must NEVER queue behind a
        # slow payload sign — the timing verdict rides on them
        # without delay.
        gate = (
            _SIGN_SERIALIZER
            if str(artifact.get("artifact_type", ""))
            == "capacity_audit_proof_payload"
            else contextlib.nullcontext()
        )
        last_exc: Exception | None = None
        for attempt in range(3):
            started = time.monotonic()
            try:
                with gate:
                    return dict(self.sign_artifact_remote(artifact))
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "mesh capacity sign attempt %d failed after %.2fs: "
                    "type=%s",
                    attempt,
                    time.monotonic() - started,
                    artifact.get("artifact_type"),
                    exc_info=True,
                )
                if attempt < 2 and self._stop_event.wait(
                    0.5 * (attempt + 1)
                ):
                    break
        logger.warning(
            "mesh capacity artifact signing failed: type=%s %s",
            artifact.get("artifact_type"),
            last_exc,
        )
        return None

    @staticmethod
    def _root_hex(root_words: object) -> str:
        from neurons.capacity_audit import root_words_digest

        if isinstance(root_words, str):
            text = root_words.strip()
            raw = text[2:] if text.startswith("0x") else text
            if len(raw) == 64:
                try:
                    bytes.fromhex(raw)
                    return raw
                except ValueError:
                    pass
        return root_words_digest(root_words)

    def _collect_gpu_artifacts(
        self,
        window: MeshAuditWindow,
        one: _PreparedGpuAudit,
        lease: str,
        slot_identifier: str,
        on_timing_settled: Callable[[], None] | None = None,
    ) -> None:
        from neurons.capacity_audit import transcript_root
        from neurons.capacity_audit_combined import (
            combined_commitment_from_final_timing,
            combined_proof_protocol_version,
        )

        timing_reported = False

        def _report_timing_settled() -> None:
            # Exactly-once per opening: the drain latch in _run_window
            # counts DOWN once per released GPU, on success AND failure.
            nonlocal timing_reported
            if timing_reported or on_timing_settled is None:
                return
            timing_reported = True
            try:
                on_timing_settled()
            except Exception:
                pass

        cfg = self.runtime_cfg
        gpu = one.gpu
        proc = one.proc
        out_dir = one.out_dir
        pass0_path = out_dir / f"{lease}_pass0.json"
        final_path = out_dir / f"{lease}_final_timing.json"
        final_summary_path = out_dir / f"{lease}_final.json"

        pass0_sent = False
        final_sent = False
        pass0_root = ""
        final_root = ""
        transcript = ""
        challenge_wait_s = (
            max(0, window.proof_challenge_block - window.audit_block)
            * _BLOCK_TIME_S
            + float(getattr(cfg, "payload_deadline_s", 0.0) or 0.0)
        )
        deadline = time.time() + max(
            120.0, window.deadline_s + 90.0 + challenge_wait_s
        )

        def publish_pass0(root: str) -> bool:
            nonlocal pass0_root, pass0_sent
            candidate = str(root or "").strip()
            if not candidate:
                return False
            pass0_root = candidate
            artifact = self._base_artifact(window, gpu, slot_identifier)
            artifact.update(
                {
                    "artifact_type": "capacity_audit_pass0_receipt",
                    "pass0_root": pass0_root,
                    "pass0_transcript_commit": pass0_root,
                    "roster": dict(self.roster),
                    "roster_signature": self.roster_signature,
                }
            )
            signed = self._sign(artifact)
            if signed is not None:
                self._publish_receipt(signed)
                pass0_sent = True
            return pass0_sent

        def read_pass0_file() -> bool:
            # Read-only here; DELIVERY runs in the background. The bench
            # writes pass0 and final_timing together, and one hanging
            # validator ingest can burn ~3x5s in pass0's sweep — publishing
            # inline here delayed the final receipt past deadline+grace
            # twice.
            nonlocal pass0_root, pass0_sent
            if not pass0_path.exists():
                return False
            data = json.loads(pass0_path.read_text())
            raw_root = data.get("root")
            if raw_root in (None, "", []):
                return False
            root = self._root_hex(raw_root)
            pass0_root = root
            pass0_sent = True
            threading.Thread(
                target=publish_pass0,
                args=(root,),
                name=f"mesh-capacity-pass0-{gpu.ordinal}",
                daemon=True,
            ).start()
            return True

        while time.time() < deadline:
            # The bench normally writes pass0 and final_timing together.  Give
            # the deadline-critical final receipt first access to delegated
            # signing when both are ready; otherwise a multi-GPU worker starts
            # one pass0 sign per GPU before any final sign and can exhaust the
            # receipt transport grace despite completing the timed work.
            final_ready = not final_sent and final_path.exists()
            if not pass0_sent and not final_ready:
                try:
                    read_pass0_file()
                except Exception:
                    pass
            if final_ready:
                data = json.loads(final_path.read_text())
                final_timing = data if isinstance(data, dict) else {}
                deferred_pass0_root = ""
                if not pass0_sent:
                    # The combined bench writes pass0 and final together, so
                    # pass0's delivery sweep must NOT delay the final: the
                    # timing verdict hangs on the final's arrival, and one
                    # slow validator answering pass0 can burn the 3s grace.
                    # Pass 0 needs only its root read
                    # synchronously; delivery runs in the background.
                    pass0_root_candidate = ""
                    try:
                        if pass0_path.exists():
                            pass0_root_candidate = self._root_hex(
                                json.loads(pass0_path.read_text()).get("root")
                                or []
                            )
                    except Exception:
                        pass0_root_candidate = ""
                    if not pass0_root_candidate:
                        raw_pass0_root = final_timing.get("pass0_root")
                        if raw_pass0_root:
                            pass0_root_candidate = self._root_hex(
                                raw_pass0_root
                            )
                    if pass0_root_candidate:
                        pass0_root = pass0_root_candidate
                        pass0_sent = True
                        deferred_pass0_root = pass0_root_candidate
                final_root = self._root_hex(data.get("root") or [])
                transcript = str(data.get("transcript_root") or "")
                if not transcript:
                    transcript = transcript_root([pass0_root, final_root])
                artifact = self._base_artifact(window, gpu, slot_identifier)
                combined_commit = combined_commitment_from_final_timing(
                    final_timing
                )
                artifact.update(
                    {
                        "artifact_type": "capacity_audit_final_receipt",
                        "pass0_root": pass0_root,
                        "final_root": final_root,
                        "final_transcript_commit": transcript,
                        "roster": dict(self.roster),
                        "roster_signature": self.roster_signature,
                    }
                )
                if combined_commit:
                    artifact["combined"] = combined_commit
                signed = self._sign(artifact)
                # Timed compute is settled as soon as its final receipt exists.
                # Signing and network delivery must not keep the serving drain
                # active after the protected workload has finished.
                _report_timing_settled()
                if signed is not None:
                    self._publish_receipt(signed)
                    final_sent = True
                if deferred_pass0_root:
                    threading.Thread(
                        target=publish_pass0,
                        args=(deferred_pass0_root,),
                        name=f"mesh-capacity-pass0-{gpu.ordinal}",
                        daemon=True,
                    ).start()
                challenge_seed = self._wait_for_challenge_seed(
                    window,
                    transcript=transcript,
                    lease=lease,
                    slot_identifier=slot_identifier,
                    gpu_ordinal=gpu.ordinal,
                    timeout_s=challenge_wait_s,
                )
                if challenge_seed:
                    tmp = one.challenge_file.with_suffix(
                        one.challenge_file.suffix + ".tmp"
                    )
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_text(challenge_seed)
                    tmp.replace(one.challenge_file)
                else:
                    logger.warning(
                        "mesh capacity proof challenge unavailable: "
                        "audit_id=%s ordinal=%s",
                        window.audit_id[:12],
                        gpu.ordinal,
                    )
                break
            if proc.poll() is not None and (final_sent or not final_path.exists()):
                break
            time.sleep(0.02)

        # Failure paths (deadline exhausted, workload died without a
        # final) settle here so a broken opening never pins the drain.
        _report_timing_settled()

        proof_assembly_timeout = max(
            5.0, float(getattr(cfg, "payload_deadline_s", 0.0) or 0.0) + 30.0
        )
        try:
            stdout, stderr = proc.communicate(timeout=proof_assembly_timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        if not final_sent:
            logger.warning(
                "mesh capacity workload produced no final receipt: "
                "ordinal=%s rc=%s stderr_tail=%s",
                gpu.ordinal,
                proc.poll(),
                (stderr or "")[-500:],
            )
            return

        final_summary: dict = {}
        if final_summary_path.exists():
            try:
                final_summary = json.loads(final_summary_path.read_text())
            except Exception:
                final_summary = {}
        proof_payload = final_summary.get("proof_payload")
        if not (
            isinstance(proof_payload, dict)
            and combined_proof_protocol_version(proof_payload) is not None
        ):
            logger.warning(
                "mesh capacity proof payload missing verifier proof: "
                "audit_id=%s ordinal=%s",
                window.audit_id[:12],
                gpu.ordinal,
            )
            return
        capacity_proof = proof_payload.get("capacity_proof") or {}
        sampled_raw = (
            capacity_proof.get("sampled")
            if isinstance(capacity_proof, dict)
            else None
        )
        try:
            sampled = int((sampled_raw or {}).get("pass_index"))
        except Exception:
            sampled = 0
        artifact = self._base_artifact(window, gpu, slot_identifier)
        artifact.update(
            {
                "artifact_type": "capacity_audit_proof_payload",
                "sampled_pass_index": sampled,
                "sampled_opening": {
                    "lease_id": lease,
                    "transcript_root": transcript,
                    "pass0_root": pass0_root,
                    "final_root": final_root,
                    "pass_index": sampled,
                },
                "sampled_pass_proof": proof_payload,
            }
        )
        signed = self._sign(artifact)
        if signed is not None:
            self._publish_proof(signed)

    # ── publishing ───────────────────────────────────────────────

    def _probe_ingest_reachability(self) -> None:
        """One HTTP roundtrip per validator ingest; WARN on each failure."""
        import httpx

        urls = self._validator_urls()
        if not urls:
            return
        reachable = []
        unreachable = []
        for endpoint in urls:
            try:
                resp = httpx.get(
                    f"{endpoint}/capacity/audit/v1/health", timeout=8.0
                )
                if resp.status_code >= 500:
                    unreachable.append(f"{endpoint} (status {resp.status_code})")
                else:
                    reachable.append(endpoint)
            except Exception as exc:
                unreachable.append(f"{endpoint} ({type(exc).__name__})")
        if unreachable:
            resolver = self._endpoint_resolver
            owner_path_reachable = bool(reachable) and any(
                not resolver.is_non_owner_endpoint(endpoint)
                for endpoint in reachable
            )
            only_non_owner_failed = all(
                resolver.is_non_owner_endpoint(
                    value.split(" (", 1)[0]
                )
                for value in unreachable
            )
            if owner_path_reachable or only_non_owner_failed:
                # Validators can retain stale/retired axon routes briefly.
                # A redundant non-owner path being down must not look like the
                # worker is doomed to probation while its scoring-owner path
                # is demonstrably live.
                logger.info(
                    "mesh capacity audit ingest ready: %d/%d validator "
                    "path(s) reachable; %d redundant path(s) unavailable",
                    len(reachable),
                    len(reachable) + len(unreachable),
                    len(unreachable),
                )
                logger.debug(
                    "unavailable mesh capacity audit paths: %s",
                    "; ".join(unreachable),
                )
                return
            logger.error(
                "mesh capacity audit ingest UNREACHABLE from this worker: "
                "%s — audit evidence cannot be delivered on this path and "
                "the validator will record no-shows (leading to probation) "
                "even though the audits run. Fix the network egress from "
                "this box to the validator audit endpoints.",
                "; ".join(unreachable),
            )

    def _validator_urls(self) -> tuple[str, ...]:
        from neurons.capacity_audit_discovery import CapacityAuditEndpointResolver

        if self._endpoint_resolver is None:
            self._endpoint_resolver = CapacityAuditEndpointResolver(
                self.config,
                manual_urls=tuple(
                    getattr(self, "manual_validator_urls", ()) or ()
                ),
            )
        try:
            return self._endpoint_resolver.current_urls()
        except Exception as exc:
            logger.warning(
                "mesh capacity validator endpoint discovery failed: %s", exc
            )
            return ()

    def _publish_receipt(self, artifact: dict[str, Any]) -> int:
        # Receipts race a seconds-scale deadline: two attempts with a short
        # retry keep the worst case per endpoint bounded near the transport
        # grace instead of tripling a hung connection's timeout.
        return self._publish_artifact(
            "/capacity/audit/v1/receipt",
            artifact,
            attempts=2,
            retry_delay_s=0.25,
        )

    def _publish_proof(self, artifact: dict[str, Any]) -> int:
        cfg = self.runtime_cfg
        attempts = max(
            1,
            int(
                float(getattr(cfg, "payload_deadline_s", 0.0) or 0.0) // 5.0
            ),
        )
        return self._publish_artifact(
            "/capacity/audit/v1/proof",
            artifact,
            attempts=min(12, attempts),
            retry_delay_s=5.0,
        )

    def _publish_artifact(
        self,
        path: str,
        artifact: dict[str, Any],
        *,
        attempts: int,
        retry_delay_s: float,
    ) -> int:
        """Deliver one artifact to every validator ingest CONCURRENTLY.

        Sequential delivery let one slow validator delay receipts to every
        other one —
        answering 400s, pushing the final receipt 3s past the timing
        deadline on the validator that mattered. One thread per endpoint;
        a slow or hostile validator now costs only its own delivery.
        """
        import httpx

        urls = self._validator_urls()
        if not urls:
            logger.warning(
                "mesh capacity publish has no validator endpoints: type=%s",
                artifact.get("artifact_type"),
            )
            return 0

        unknown_slot_refusals: list[str] = []

        def deliver(endpoint: str) -> int:
            for attempt in range(max(1, int(attempts))):
                try:
                    resp = httpx.post(
                        f"{endpoint}{path}", json=artifact, timeout=5.0
                    )
                except Exception as exc:
                    if attempt + 1 >= max(1, int(attempts)):
                        # The last attempt for this endpoint failed at the
                        # transport layer. This is the failure mode that
                        # silently turns a diligent worker into a no-show,
                        # so it must be visible at production log level.
                        logger.warning(
                            "mesh capacity publish undeliverable: url=%s "
                            "attempts=%s %s: %s",
                            endpoint,
                            attempts,
                            type(exc).__name__,
                            exc,
                        )
                    else:
                        logger.debug(
                            "mesh capacity publish error: url=%s %s",
                            endpoint,
                            exc,
                        )
                    resp = None
                if resp is not None:
                    if resp.status_code < 300:
                        return 1
                    if resp.status_code not in (409, 425, 429, 500, 502, 503, 504):
                        if "unknown audit slot" in str(resp.text or "").lower():
                            unknown_slot_refusals.append(endpoint)
                        logger.warning(
                            "mesh capacity publish refused: type=%s url=%s "
                            "status=%s body=%s",
                            artifact.get("artifact_type"),
                            endpoint,
                            resp.status_code,
                            resp.text[:200],
                        )
                        return 0
                if attempt + 1 < attempts and self._stop_event.wait(
                    max(0.1, retry_delay_s)
                ):
                    return 0
            return 0

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(8, len(urls)),
            thread_name_prefix="mesh-capacity-publish",
        ) as pool:
            delivered = sum(pool.map(deliver, urls))
        if delivered == 0:
            streak = int(getattr(self, "_publish_failure_streak", 0)) + 1
            self._publish_failure_streak = streak
            if streak >= 3:
                logger.error(
                    "mesh capacity audit evidence UNDELIVERABLE to every "
                    "validator ingest (%s consecutive artifacts). The "
                    "validator records these windows as no-shows and the "
                    "slot WILL be probated even though the audits run. "
                    "Check network egress from this box to the validator "
                    "audit endpoints.",
                    streak,
                )
        else:
            self._publish_failure_streak = 0
        if (
            delivered == 0
            and unknown_slot_refusals
            and str(artifact.get("audit_id") or "")
            and str(artifact.get("audit_id")) == str(self._active_audit_id or "")
        ):
            # EVERY validator refused this window's evidence as an unknown
            # slot: no scheduler recognizes it, so holding the local
            # active-window lock only makes the worker skip the windows the
            # validator DID schedule. Abandon immediately; the next scheduled window
            # re-converges both sides.
            audit_id = str(artifact.get("audit_id"))
            logger.warning(
                "mesh capacity audit abandoning window %s: every validator "
                "refused its evidence as an unknown slot",
                audit_id[:12],
            )
            self._active_audit_id = ""
            self._active_audit_until_ts = 0.0
            try:
                self._clear_audit_drain(audit_id)
            except Exception:
                pass
        return delivered
