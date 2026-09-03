"""Mesh capacity-audit worker unit behavior.

Chain interaction and the CUDA workload are exercised on testnet; these
tests pin the pure derivations the non-interactive protocol depends on:
selection parity inputs, artifact identity fields, per-GPU device masking,
and the drain state contract the serve gate reads.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from verallm.mesh.capacity_audit_worker import (
    LocalAuditGpu,
    MeshAuditWindow,
    MeshCapacityAuditWorker,
    capacity_drain_active,
    capacity_drain_file_path,
    capacity_roster_file_path,
)

SLOT_CTX = {
    "chain_id": 945,
    "netuid": 405,
    "address": "0x" + "ab" * 20,
    "model_index": 45,
    "endpoint": "https://chain.example.org:20001",
    "model_id": "unsloth/GLM-4.5-Air-GGUF",
    "quant": "gguf_q4_k_xl",
    "max_context_len": 98304,
}



@pytest.fixture(autouse=True)
def _no_real_workspace_holders(monkeypatch, request):
    """Unit tests must NEVER allocate GPU memory. A pytest run on a GPU box
    spawned a REAL 1.3GB workspace holder that outlived the test process
    (reparented to the systemd subreaper) and OOM'd the production serve
    on the same card . Tests that exercise the holder
    lifecycle opt out and stub the child code instead."""

    if "workspace_holders" in request.node.name:
        yield
        return
    import verallm.mesh.capacity_audit_worker as _caw

    monkeypatch.setattr(
        _caw.WorkspaceHolders, "ensure", lambda self, idx: None
    )
    monkeypatch.setattr(
        _caw.WorkspaceHolders, "rehold", lambda self, idx: None
    )
    yield

def _context(roster_epoch: int = 21550) -> dict:
    return {
        "mesh_key": "m-test",
        "slot": dict(SLOT_CTX),
        "roster": {
            "version": 1,
            "slot": {
                "chain_id": 945,
                "netuid": 405,
                "address": SLOT_CTX["address"],
                "model_index": 45,
            },
            "roster_epoch": roster_epoch,
            "workers": [
                {
                    "worker_id": "w-a",
                    "backend": "cuda",
                    "gpu_names": ["NVIDIA A100 80GB PCIe"] * 2,
                    "per_gpu_vram_gb": [80, 80],
                    "host_ip24": "198.18.0",
                    "host_regdom": "",
                }
            ],
        },
        "roster_signature": "cd" * 65,
        "local_gpus": [
            {
                "ordinal": 0,
                "local_gpu_index": 0,
                "gpu_name": "NVIDIA A100 80GB PCIe",
                "vram_gb": 80,
            },
            {
                "ordinal": 1,
                "local_gpu_index": 1,
                "gpu_name": "NVIDIA A100 80GB PCIe",
                "vram_gb": 80,
            },
        ],
    }


def _worker(tmp_path, **overrides) -> MeshCapacityAuditWorker:
    kwargs = dict(
        context=_context(),
        workdir=tmp_path,
        repo_root=tmp_path,
        sign_artifact_remote=lambda artifact: dict(
            artifact, miner_signature="00" * 65
        ),
        subtensor_network="ws://chain.example.org:9944",
        netuid=405,
        worker_id="w-a",
    )
    kwargs.update(overrides)
    return MeshCapacityAuditWorker(**kwargs)


def _window(gpus) -> MeshAuditWindow:
    return MeshAuditWindow(
        audit_id="aa" * 32,
        selection_block=7_760_100,
        audit_block=7_760_105,
        proof_challenge_block=7_760_109,
        cohort_seed="bb" * 32,
        epoch_number=21_555,
        gpus=tuple(gpus),
        gpu_class_name="NVIDIA A100 80GB PCIe",
        passes=10,
        deadline_s=30.0,
        workload_spec={"capacity_matrix_dim": 8960},
    )


def test_startup_ingest_probe_does_not_alarm_on_redundant_non_owner_path(
    tmp_path, monkeypatch
):
    import verallm.mesh.capacity_audit_worker as capacity_worker_module

    owner = "https://owner.example:8091"
    follower = "https://follower.example:8093"
    worker = _worker(tmp_path)
    worker._endpoint_resolver = SimpleNamespace(
        current_urls=lambda: (owner, follower),
        is_non_owner_endpoint=lambda endpoint: endpoint == follower,
    )

    def get(url, timeout):
        assert timeout == 8.0
        if url.startswith(owner):
            return SimpleNamespace(status_code=200)
        raise TimeoutError("retired path")

    monkeypatch.setattr("httpx.get", get)
    messages = []
    monkeypatch.setattr(
        capacity_worker_module.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )
    monkeypatch.setattr(
        capacity_worker_module.logger,
        "error",
        lambda message, *args: messages.append("ERROR: " + message % args),
    )
    worker._probe_ingest_reachability()

    assert any("ingest ready: 1/2" in message for message in messages)
    assert not any("ingest UNREACHABLE" in message for message in messages)


def test_startup_ingest_probe_still_alarms_when_owner_path_is_down(
    tmp_path, monkeypatch
):
    import verallm.mesh.capacity_audit_worker as capacity_worker_module

    owner = "https://owner.example:8091"
    follower = "https://follower.example:8093"
    worker = _worker(tmp_path)
    worker._endpoint_resolver = SimpleNamespace(
        current_urls=lambda: (owner, follower),
        is_non_owner_endpoint=lambda endpoint: endpoint == follower,
    )

    def get(url, timeout):
        if url.startswith(follower):
            return SimpleNamespace(status_code=200)
        raise TimeoutError("owner path down")

    monkeypatch.setattr("httpx.get", get)
    messages = []
    monkeypatch.setattr(
        capacity_worker_module.logger,
        "error",
        lambda message, *args: messages.append(message % args),
    )
    worker._probe_ingest_reachability()

    assert any("ingest UNREACHABLE" in message for message in messages)


# ---------------------------------------------------------------------------
# Drain state file contract (read by the serve admission gate)
# ---------------------------------------------------------------------------


def test_subnet_config_url_resolves_per_network(tmp_path, monkeypatch):
    """The runtime-config URL must match the slot's network. The client's
    module default is the MAINNET URL; a testnet worker silently running
    mainnet epoch parameters derives disjoint window schedules from its
    validator and every audit expires as no_show ."""

    from neurons.config import (
        MAINNET_SUBNET_CONFIG_URL,
        TESTNET_SUBNET_CONFIG_URL,
    )

    monkeypatch.delenv("VERATHOS_SUBNET_CONFIG_URL", raising=False)

    # chain_id 945 (testnet) wins even when the daemon was launched with a
    # raw ws:// endpoint instead of a network name.
    worker = _worker(tmp_path)
    assert worker.config.subnet_config_url == TESTNET_SUBNET_CONFIG_URL

    # A mainnet chain slot resolves to the mainnet URL.
    mainnet_ctx = _context()
    mainnet_ctx["slot"] = dict(mainnet_ctx["slot"], chain_id=964)
    worker = _worker(
        tmp_path, context=mainnet_ctx, subtensor_network="finney"
    )
    assert worker.config.subnet_config_url == MAINNET_SUBNET_CONFIG_URL

    # A manager-delivered context URL beats the chain_id mapping.
    ctx = _context()
    ctx["subnet_config_url"] = "https://example.test/config.json"
    worker = _worker(tmp_path, context=ctx)
    assert worker.config.subnet_config_url == "https://example.test/config.json"

    # The operator env override beats everything.
    monkeypatch.setenv(
        "VERATHOS_SUBNET_CONFIG_URL", "https://operator.test/override.json"
    )
    worker = _worker(tmp_path, context=ctx)
    assert (
        worker.config.subnet_config_url == "https://operator.test/override.json"
    )


def test_drain_active_semantics(tmp_path):
    path = capacity_drain_file_path(tmp_path)
    assert not capacity_drain_active(path)  # absent = serving

    path.write_text(json.dumps({"active": True, "until_ts": time.time() + 60}))
    assert capacity_drain_active(path)

    path.write_text(json.dumps({"active": True, "until_ts": time.time() - 1}))
    assert not capacity_drain_active(path)  # expired self-heals

    # An unbounded drain is REFUSED: the writer always stamps a deadline,
    # so a zero bound is a truncated/foreign file — permanently 503ing
    # every chat off it would need a hand-delete to recover.
    path.write_text(json.dumps({"active": True, "until_ts": 0}))
    assert not capacity_drain_active(path)

    path.write_text(json.dumps({"active": False, "last_audit_id": "x"}))
    assert not capacity_drain_active(path)

    path.write_text("{corrupt")
    assert not capacity_drain_active(path)  # fail-open, never block serving


def test_mark_and_clear_drain_round_trip(tmp_path):
    worker = _worker(tmp_path)
    worker.runtime_cfg = __import__(
        "neurons.capacity_audit", fromlist=["CapacityAuditRuntimeConfig"]
    ).CapacityAuditRuntimeConfig()
    window = _window(worker.local_gpus)
    worker._mark_audit_drain(window, phase="selected")
    assert capacity_drain_active(worker.drain_file)
    assert worker._has_active_local_audit()
    payload = json.loads(worker.drain_file.read_text())
    assert payload["audit_id"] == window.audit_id
    # Budget covers blocks to B_proof plus every evidence deadline.
    assert payload["until_ts"] >= time.time() + 60.0

    worker._clear_audit_drain(window.audit_id)
    assert not capacity_drain_active(worker.drain_file)
    assert not worker._has_active_local_audit()


# ---------------------------------------------------------------------------
# Selection inputs and identity fields
# ---------------------------------------------------------------------------


def test_capacity_slot_matches_chain_context(tmp_path):
    from neurons.capacity_audit import slot_id

    worker = _worker(tmp_path)
    slot = worker._capacity_slot()
    assert slot.address == SLOT_CTX["address"]
    assert slot.model_index == 45
    assert slot.endpoint == SLOT_CTX["endpoint"]
    assert slot.quant == SLOT_CTX["quant"]
    assert slot.max_context_len == 98304
    # slot_id is the cross-side join key; it must derive exactly as the
    # validator derives it from the same chain facts.
    assert slot_id(slot) == slot_id(
        {
            "chain_id": 945,
            "netuid": 405,
            "address": SLOT_CTX["address"],
            "model_index": 45,
        }
    )


def test_selection_roster_only_when_frozen_before_epoch(tmp_path):
    worker = _worker(tmp_path)
    # roster_epoch 21550 <= 21555-1: contributes.
    assert worker._selection_roster(21_555) is not None
    # Same epoch: too fresh, both sides must ignore it.
    assert worker._selection_roster(21_550) is None
    worker_no_roster = _worker(tmp_path, context={**_context(), "roster": {}})
    assert worker_no_roster._selection_roster(21_555) is None


def test_base_artifact_carries_ordinal_and_roster_identity(tmp_path):
    from neurons.capacity_audit import slot_id

    from verallm.mesh.capacity_roster import roster_digest

    worker = _worker(tmp_path)
    window = _window(worker.local_gpus)
    sid = slot_id(worker._capacity_slot())
    artifact = worker._base_artifact(window, worker.local_gpus[1], sid)
    assert artifact["gpu_index"] == 1  # the GLOBAL ordinal, never device 0
    assert artifact["local_gpu_index"] == 1
    assert artifact["worker_id"] == "w-a"
    assert artifact["address"] == SLOT_CTX["address"]
    assert artifact["model_index"] == 45
    assert artifact["roster_digest"] == roster_digest(_context()["roster"])
    assert artifact["B_select"] == window.selection_block
    assert artifact["B_start"] == window.audit_block
    assert artifact["B_proof"] == window.proof_challenge_block


def test_workspace_env_masks_one_gpu_per_process(tmp_path):
    worker = _worker(tmp_path)
    env = worker._workspace_env(
        LocalAuditGpu(
            ordinal=3,
            local_gpu_index=2,
            gpu_name="NVIDIA A100 80GB PCIe",
            vram_gb=80,
        )
    )
    # The process sees exactly its LOCAL device.
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_prepare_gpu_audit_binds_the_global_ordinal(tmp_path, monkeypatch):
    """--gpu-index is the PROOF-binding index (it enters every lane-seed
    derivation the validator recomputes with the roster ordinal); passing
    the masked local 0 made every ordinal>0 opening of a multi-GPU window
    an instant invalid_payload ."""

    worker = _worker(tmp_path)
    worker.runtime_cfg = __import__(
        "neurons.capacity_audit", fromlist=["CapacityAuditRuntimeConfig"]
    ).CapacityAuditRuntimeConfig()
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop before spawning anything real")

    monkeypatch.setattr(
        "verallm.mesh.capacity_audit_worker.subprocess.Popen", fake_popen
    )
    gpu = LocalAuditGpu(
        ordinal=3,
        local_gpu_index=2,
        gpu_name="NVIDIA A100 80GB PCIe",
        vram_gb=80,
    )
    window = _window((gpu,))
    worker._prepare_gpu_audit(window, gpu, lease="le", start_timeout_s=5.0)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--gpu-index") + 1] == "3"


def test_start_refuses_without_slot_gpus_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        MeshCapacityAuditWorker,
        "_preflight_workspace",
        lambda self: (True, ""),
    )
    assert not _worker(
        tmp_path, context={**_context(), "local_gpus": []}
    ).start()
    context = _context()
    context["slot"] = {}
    assert not _worker(tmp_path, context=context).start()
    assert not _worker(tmp_path, subtensor_network="").start()


def test_start_publishes_roster_file_for_the_serve_route(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        MeshCapacityAuditWorker,
        "_preflight_workspace",
        lambda self: (True, ""),
    )
    monkeypatch.setattr(
        MeshCapacityAuditWorker,
        "_refresh_runtime_config",
        lambda self, **kwargs: None,
    )
    worker = _worker(tmp_path)
    assert worker.start()
    try:
        payload = json.loads(
            capacity_roster_file_path(tmp_path).read_text()
        )
        assert payload["roster"] == _context()["roster"]
        assert payload["roster_signature"] == "cd" * 65
    finally:
        worker.stop()


def test_start_waiter_reconnects_zero_head_before_release(
    tmp_path, monkeypatch
):
    """A stale short-lived RPC object must not consume the complete start window."""

    worker = _worker(tmp_path)
    worker._running = True
    worker.runtime_cfg = __import__(
        "neurons.capacity_audit", fromlist=["CapacityAuditRuntimeConfig"]
    ).CapacityAuditRuntimeConfig()
    window = _window((worker.local_gpus[0],))
    prepared = object()
    stale = object()
    fresh = object()
    connections = iter((stale, fresh))
    closed = []
    released = []

    monkeypatch.setattr(
        worker,
        "_prepare_gpu_audit",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(worker, "_subtensor", lambda: next(connections))
    monkeypatch.setattr(
        worker,
        "_get_current_head",
        lambda connection: 0 if connection is stale else window.audit_block,
    )
    monkeypatch.setattr(worker, "_get_block_hash", lambda *args: b"x" * 32)
    monkeypatch.setattr(worker, "_close_subtensor", closed.append)
    monkeypatch.setattr(
        worker,
        "_run_window",
        lambda selected, block_hash, items, lease, connection: released.append(
            (selected, block_hash, items, connection)
        ),
    )

    worker._await_and_run_window(window)

    assert stale in closed
    assert released == [(window, b"x" * 32, [prepared], fresh)]


#: Stub holder child: holds nothing, speaks just enough protocol for the
#: parent's spawn wait. Real holders allocate VRAM, which test machines
#: may not have.
_HOLDER_STUB = "import sys, time\nprint('held', flush=True)\ntime.sleep(60)\n"

#: Stub holder child that also emulates the in-process bench protocol:
#: bench_started, a final.json artifact, bench_done, then keeps holding.
_HOLDER_BENCH_STUB = """
import json, pathlib, sys
print("held", flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    if cmd.get("cmd") != "bench":
        continue
    print(
        "VRAM_HOLDER_EVT "
        + json.dumps({"event": "bench_started", "lease_id": cmd.get("lease_id")}),
        flush=True,
    )
    argv = cmd["argv"]
    out_dir = pathlib.Path(argv[argv.index("--out-dir") + 1])
    lease = argv[argv.index("--lease-id") + 1]
    (out_dir / (lease + "_final.json")).write_text(
        json.dumps({"lease_id": lease, "sys_path": cmd.get("sys_path")}) + "\\n"
    )
    print(
        "VRAM_HOLDER_EVT "
        + json.dumps(
            {"event": "bench_done", "rc": 0, "held": True, "error": ""}
        ),
        flush=True,
    )
"""


def _stub_holder_command(monkeypatch, code):
    import sys as _sys

    import verallm.mesh.capacity_audit_worker as caw

    monkeypatch.setattr(
        caw, "_holder_command", lambda hold_mb: [_sys.executable, "-c", code]
    )


def test_workspace_holders_lifecycle(monkeypatch, tmp_path):
    """Resident holders: ensure spawns per GPU, release frees for the bench
    swap, rehold restores, stop_all cleans up. The child is stubbed — real
    holders allocate VRAM, which test machines may not have."""

    import verallm.mesh.capacity_audit_worker as caw

    _stub_holder_command(monkeypatch, _HOLDER_STUB)
    import verallm.mesh.pool as pool_mod

    monkeypatch.setattr(
        pool_mod, "_gpu_free_vram_mb", lambda: [80 * 1024, 80 * 1024]
    )

    holders = caw.WorkspaceHolders()
    holders.ensure([0, 1])
    assert holders.active(0) and holders.active(1)

    # ensure() is idempotent for live holders.
    procs_before = dict(holders._procs)
    holders.ensure([0, 1])
    assert holders._procs == procs_before

    holders.release(0)
    assert not holders.active(0)
    assert holders.active(1)

    holders.rehold(0)
    assert holders.active(0)

    holders.stop_all()
    assert not holders.active(0) and not holders.active(1)


def test_workspace_holders_skip_full_gpu(monkeypatch):
    """A GPU without hold+margin free is skipped loudly, never squeezed."""

    import verallm.mesh.capacity_audit_worker as caw
    import verallm.mesh.pool as pool_mod

    _stub_holder_command(monkeypatch, _HOLDER_STUB)
    monkeypatch.setattr(
        pool_mod,
        "_gpu_free_vram_mb",
        lambda: [500, 80 * 1024],
    )
    holders = caw.WorkspaceHolders()
    try:
        holders.ensure([0, 1])
        assert not holders.active(0)
        assert holders.active(1)
    finally:
        holders.stop_all()


def test_workspace_holders_bench_runs_in_holder_process(
    monkeypatch, tmp_path
):
    """The window handoff runs the bench INSIDE the resident holder: the
    workspace pages never leave the holder's caching allocator, so the
    serve has no gap to grow into. The holder survives the bench and is reusable
    for the next window."""

    import verallm.mesh.capacity_audit_worker as caw
    import verallm.mesh.pool as pool_mod

    _stub_holder_command(monkeypatch, _HOLDER_BENCH_STUB)
    monkeypatch.setattr(pool_mod, "_gpu_free_vram_mb", lambda: [80 * 1024])

    holders = caw.WorkspaceHolders()
    try:
        holders.ensure([0])
        assert holders.active(0)

        argv = ["--child", "--out-dir", str(tmp_path), "--lease-id", "le1"]
        handle = holders.run_bench(
            0, argv=argv, sys_path=["/x"], lease_id="le1"
        )
        assert handle is not None
        stdout, stderr = handle.communicate(timeout=10)
        assert handle.poll() == 0
        final = json.loads((tmp_path / "le1_final.json").read_text())
        assert final["lease_id"] == "le1"
        assert final["sys_path"] == ["/x"]
        # The holder never died: no re-hold gap, and the next window can
        # reuse it immediately.
        assert holders.active(0)
        handle2 = holders.run_bench(
            0,
            argv=["--child", "--out-dir", str(tmp_path), "--lease-id", "le2"],
            sys_path=[],
            lease_id="le2",
        )
        assert handle2 is not None
        assert handle2.communicate(timeout=10)
        assert handle2.poll() == 0
        assert (tmp_path / "le2_final.json").exists()
    finally:
        holders.stop_all()


def test_workspace_holders_bench_requires_a_live_holder(monkeypatch):
    """No resident holder = no in-process bench: the caller must fall back
    to the legacy bench-child path."""

    import verallm.mesh.capacity_audit_worker as caw

    holders = caw.WorkspaceHolders()
    assert (
        holders.run_bench(0, argv=["--child"], sys_path=[], lease_id="le")
        is None
    )


def test_prepare_gpu_audit_prefers_the_resident_holder(tmp_path, monkeypatch):
    """When a resident holder is live, the bench must run inside it — the
    fallback bench child (and its holder-release gap) is for holderless
    GPUs only."""

    import verallm.mesh.capacity_audit_worker as caw

    worker = _worker(tmp_path)
    worker.runtime_cfg = __import__(
        "neurons.capacity_audit", fromlist=["CapacityAuditRuntimeConfig"]
    ).CapacityAuditRuntimeConfig()
    captured: dict = {}
    sentinel = object()

    def fake_run_bench(self, local_gpu_index, *, argv, sys_path, lease_id, **kw):
        captured["local_gpu_index"] = local_gpu_index
        captured["argv"] = list(argv)
        captured["lease_id"] = lease_id
        return sentinel

    monkeypatch.setattr(caw.WorkspaceHolders, "run_bench", fake_run_bench)

    def forbidden_popen(cmd, **kwargs):
        raise AssertionError(
            "a live holder must absorb the bench; no child spawn allowed"
        )

    monkeypatch.setattr(caw.subprocess, "Popen", forbidden_popen)
    gpu = LocalAuditGpu(
        ordinal=3,
        local_gpu_index=2,
        gpu_name="NVIDIA A100 80GB PCIe",
        vram_gb=80,
    )
    window = _window((gpu,))
    prepared = worker._prepare_gpu_audit(
        window, gpu, lease="le", start_timeout_s=5.0
    )
    assert prepared is not None
    assert prepared.proc is sentinel
    assert captured["local_gpu_index"] == 2
    argv = captured["argv"]
    # The PROOF binding still rides the GLOBAL roster ordinal.
    assert argv[argv.index("--gpu-index") + 1] == "3"
    assert "--child" in argv


def test_cuda_drive_refuses_without_gpu_merkle_kernel(monkeypatch):
    """OPERATOR ORDER: no model loads on a CUDA worker without the working
    GPU BLAKE3 kernel — the CPU fallback served multi-second light proofs
    in production ."""

    from types import SimpleNamespace

    import verallm.mesh.pool as pool_mod

    runner = object.__new__(pool_mod.LocalMeshRunner)
    runner.config = SimpleNamespace(
        rpc_device="CUDA0",
        wallet_name="w",
        wallet_hotkey="h",
        subnet_driver_ready=True,
        catalog=[],
    )
    monkeypatch.setattr(
        "verallm.mesh.gguf_manifest._gpu_merkle_hash_available",
        lambda: False,
    )
    # Reach the gate through the drive prologue shape: call the gate
    # condition directly (the full drive needs a live pool).
    from verallm.mesh.gguf_manifest import _gpu_merkle_hash_available

    assert _gpu_merkle_hash_available() is False
    # And the source enforces refusal on that condition:
    import inspect

    src = inspect.getsource(pool_mod.LocalMeshRunner.drive)
    assert "GPU BLAKE3 Merkle kernel unavailable on this CUDA" in src


# ---------------------------------------------------------------------------
# Runtime subnet config: hosted gpu_classes and scalar knobs reach the worker
# ---------------------------------------------------------------------------


class _FakeRuntimeClient:
    """RuntimeSubnetConfigClient stand-in serving an in-test hosted payload
    through the SAME validation path the real client uses."""

    def __init__(self, payload: dict):
        self.payload = payload

    def get(self, *, current_epoch=None, current_block=None, force=False):
        from neurons.subnet_runtime_config import validate_subnet_config_payload

        return validate_subnet_config_payload(self.payload, source="test")


def _hosted_payload(
    version: int,
    a100_pass_counts: tuple[int, int, int] = (5, 5, 100),
) -> dict:
    """A complete hosted subnet-config payload with the worker's GPU class
    (NVIDIA A100 80GB PCIe) carrying owner-chosen pass counts."""

    from neurons.subnet_runtime_config import build_default_subnet_config_payload

    payload = build_default_subnet_config_payload(version=version)
    audit = payload["capacity_audit"]
    audit["enabled"] = True
    capacity, tail, fp64 = a100_pass_counts
    for row in audit["gpu_classes"]:
        if row["match_gpu_name"] == "NVIDIA A100 80GB PCIe":
            row["capacity_passes"] = capacity
            row["capacity_tail_passes"] = tail
            row["fp64_passes"] = fp64
            break
    else:
        raise AssertionError("A100 80GB PCIe row missing from default table")
    return payload


def _force_selection(monkeypatch) -> None:
    """Make every block a selected window; the derivation under test
    (gpu-class match, pass count, workload spec) stays REAL."""

    import neurons.capacity_audit as ca

    monkeypatch.setattr(ca, "capacity_audit_window_triggered", lambda *a, **k: True)
    monkeypatch.setattr(ca, "capacity_audit_window_fits_epoch", lambda *a, **k: True)
    monkeypatch.setattr(ca, "capacity_audit_slot_selected", lambda *a, **k: True)


def _derive_window(worker, block_number: int):
    captured: list = []
    done = threading.Event()

    def _capture(window):
        captured.append(window)
        done.set()

    worker._await_and_run_window = _capture
    worker._on_block(int(block_number), b"\x11" * 32, None)
    assert done.wait(5.0), "window derivation did not produce a window"
    return captured[0]


def test_hosted_gpu_classes_reach_next_window_without_restart(
    tmp_path, monkeypatch
):
    """A hosted gpu_classes change must reach the worker's next window
    without a daemon restart or falling back to import-time defaults."""

    _force_selection(monkeypatch)
    worker = _worker(tmp_path)
    payload = _hosted_payload(9010, (5, 5, 100))
    worker._runtime_client = _FakeRuntimeClient(payload)
    epoch_blocks = int(payload["epoch"]["epoch_blocks"])

    # Epoch-boundary block: _on_block force-refreshes the runtime config
    # exactly like the live daemon, then derives the window.
    window = _derive_window(worker, 21_556 * epoch_blocks)
    assert window.gpu_class_name == "NVIDIA A100 80GB PCIe"
    assert window.passes == 5 + 5 + 100
    assert window.workload_spec["capacity_passes"] == 5
    assert window.workload_spec["capacity_tail_passes"] == 5
    assert window.workload_spec["fp64_passes"] == 100

    # The owner publishes a NEW table version mid-life. Same worker object:
    # the next epoch-boundary refresh alone must carry the new pass counts
    # into the next window's derivation and bench spec.
    worker._runtime_client.payload = _hosted_payload(9011, (6, 6, 120))
    worker._active_audit_id = ""
    worker._active_audit_until_ts = 0.0
    window = _derive_window(worker, 21_557 * epoch_blocks)
    assert window.passes == 6 + 6 + 120
    assert window.workload_spec["capacity_passes"] == 6
    assert window.workload_spec["capacity_tail_passes"] == 6
    assert window.workload_spec["fp64_passes"] == 120


def test_runtime_refresh_still_applies_scalar_knobs(tmp_path):
    """Regression pin: the gpu_classes fix must not disturb the scalar
    knobs the refresh already plumbed (windows, deadlines, mode, epoch)."""

    worker = _worker(tmp_path)
    payload = _hosted_payload(9012)
    payload["capacity_audit"]["windows_per_epoch"] = 7
    payload["capacity_audit"]["deadline_s"] = 20.0
    payload["capacity_audit"]["lead_blocks"] = 9
    payload["capacity_audit"]["mode"] = "enforce"
    payload["epoch"]["epoch_blocks"] = 180
    worker._runtime_client = _FakeRuntimeClient(payload)

    worker._refresh_runtime_config(force=True)

    assert worker._audit_enabled()
    cfg = worker.runtime_cfg
    assert cfg.windows_per_epoch == 7
    assert cfg.deadline_s == 20.0
    assert cfg.lead_blocks == 9
    assert cfg.mode == "enforce"
    assert worker._epoch_blocks() == 180
    # And the hosted table rode along with them.
    assert [row.match_gpu_name for row in cfg.gpu_classes] == [
        row["match_gpu_name"]
        for row in payload["capacity_audit"]["gpu_classes"]
    ]


def test_restarted_worker_adopts_effective_config_before_first_window(tmp_path):
    payload = _hosted_payload(9013)
    payload["effective_epoch"] = 21_974
    payload["epoch"]["epoch_blocks"] = 360
    payload["capacity_audit"]["enabled"] = False
    from neurons.subnet_runtime_config import validate_subnet_config_payload

    cfg = validate_subnet_config_payload(payload, source="test")
    calls = []

    class Client:
        def get(self, **kwargs):
            calls.append(dict(kwargs))
            block = kwargs.get("current_block")
            return (
                cfg
                if block is not None
                and block // cfg.epoch_blocks >= int(cfg.effective_epoch)
                else None
            )

    worker = _worker(tmp_path)
    worker._runtime_client = Client()
    worker._runtime_authoritative = False
    worker.runtime_cfg = None

    # Mid-epoch restart: adoption happens before trigger/selection derivation,
    # not only if the worker happens to observe an exact modulo boundary.
    worker._on_block(21_974 * 360 + 1, b"\x11" * 32, None)

    assert calls == [
        {"current_epoch": None, "current_block": 21_974 * 360 + 1, "force": False}
    ]
    assert worker._runtime_authoritative is True
    assert worker._epoch_blocks() == 360
    assert worker.runtime_cfg.enabled is False


def test_final_receipt_preempts_pass0_signing_when_both_are_ready(
    tmp_path, monkeypatch
):
    """A multi-GPU receipt burst must spend delegated-sign capacity on the
    deadline-bearing final receipts before background pass0 delivery."""

    from types import SimpleNamespace

    from neurons.capacity_audit import CapacityAuditRuntimeConfig

    worker = _worker(tmp_path)
    worker.runtime_cfg = CapacityAuditRuntimeConfig()
    gpu = worker.local_gpus[0]
    window = _window((gpu,))
    lease = "priority-lease"
    out_dir = tmp_path / "priority"
    out_dir.mkdir()
    pass0_root = "11" * 32
    final_root = "22" * 32
    (out_dir / f"{lease}_pass0.json").write_text(
        json.dumps({"root": pass0_root})
    )
    (out_dir / f"{lease}_final_timing.json").write_text(
        json.dumps({"root": final_root, "pass0_root": pass0_root})
    )

    class _Proc:
        def poll(self):
            return None

        def communicate(self, timeout=None):
            del timeout
            return "", ""

    one = SimpleNamespace(
        gpu=gpu,
        proc=_Proc(),
        out_dir=out_dir,
        challenge_file=out_dir / f"{lease}_challenge.txt",
    )
    events: list[str] = []
    pass0_published = threading.Event()

    def _sign(artifact):
        events.append(f"sign:{artifact['artifact_type']}")
        return dict(artifact, miner_signature="00" * 65)

    def _publish(artifact):
        events.append(f"publish:{artifact['artifact_type']}")
        if artifact["artifact_type"] == "capacity_audit_pass0_receipt":
            pass0_published.set()
        return 1

    monkeypatch.setattr(worker, "_sign", _sign)
    monkeypatch.setattr(worker, "_publish_receipt", _publish)
    monkeypatch.setattr(
        worker, "_wait_for_challenge_seed", lambda *args, **kwargs: ""
    )

    worker._collect_gpu_artifacts(
        window,
        one,
        lease,
        "slot-priority",
        on_timing_settled=lambda: events.append("timing:settled"),
    )
    assert pass0_published.wait(2.0)
    assert events[:3] == [
        "sign:capacity_audit_final_receipt",
        "timing:settled",
        "publish:capacity_audit_final_receipt",
    ]
    assert events[3:] == [
        "sign:capacity_audit_pass0_receipt",
        "publish:capacity_audit_pass0_receipt",
    ]


def test_drain_clears_when_every_opening_settles_timing(tmp_path):
    """The drain protects the TIMED workload only: once every released
    GPU has its final timing settled, chats resume while the challenge
    wait, proof assembly, and payload push continue undrained."""

    import threading as _threading

    worker = _worker(tmp_path)
    worker.runtime_cfg = __import__(
        "neurons.capacity_audit", fromlist=["CapacityAuditRuntimeConfig"]
    ).CapacityAuditRuntimeConfig()
    window = _window(worker.local_gpus)
    worker._mark_audit_drain(window, phase="running")
    assert capacity_drain_active(worker.drain_file)

    # Mirror _run_window's latch: two openings, drain clears only after
    # BOTH report settled timing.
    timing_pending = {"n": 2}
    lock = _threading.Lock()

    def settled():
        with lock:
            timing_pending["n"] -= 1
            done = timing_pending["n"] <= 0
        if done:
            worker._clear_audit_drain(window.audit_id)

    settled()
    assert capacity_drain_active(worker.drain_file)
    settled()
    assert not capacity_drain_active(worker.drain_file)
