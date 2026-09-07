"""Worker pool orchestration tests (docs/architecture/mesh_orchestration_ux.md)."""

from __future__ import annotations

import json
import inspect
import threading
import time
import urllib.request
import stat
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from verallm.mesh import cli as mesh_cli
from verallm.mesh import pool as pool_module
from verallm.mesh.llama_cpp import build_llama_server_command
from verallm.mesh.pool import (
    LocalMeshRunner,
    MeshPoolToken,
    POOL_ADMIN_TOKEN_FILE,
    PoolManager,
    PoolWorkerConfig,
    create_pool_state,
    load_pool_token_file,
    pool_worker_loop,
    serve_pool_manager,
)
from verallm.mesh.worker import post_json
from verallm.mesh.types import MeshSpec


@pytest.mark.parametrize("owned,dead", [(True, False), (False, False), (True, True)])
def test_rpc_readiness_checks_owned_listener_without_connecting(monkeypatch, tmp_path, owned, dead):
    runner = object.__new__(LocalMeshRunner)
    runner.procs = [SimpleNamespace(pid=1234, poll=lambda: 1 if dead else None)]
    runner.config = SimpleNamespace(workdir=tmp_path)
    monkeypatch.setattr(runner, "_assert_command_active", lambda: None)
    monkeypatch.setattr(pool_module, "_proc_net_available", lambda: True)
    monkeypatch.setattr(pool_module, "_listeners_on_port", lambda port: ({5678}, True))
    monkeypatch.setattr(pool_module, "_pid_descends_from", lambda pid, roots: owned)
    def no_connect(*args, **kwargs):
        pytest.fail("readiness must not fill a single-client RPC accept queue")
    monkeypatch.setattr(pool_module.socket, "create_connection", no_connect)
    if dead:
        with pytest.raises(RuntimeError, match="process exited"):
            runner._wait_tcp("127.0.0.1", 20000)
    elif not owned:
        with pytest.raises(RuntimeError, match="not owned"):
            runner._wait_tcp("127.0.0.1", 20000)
    else:
        runner._wait_tcp("127.0.0.1", 20000)


def test_rpc_listener_parent_chain_fails_closed(monkeypatch):
    stats = {5678: "5678 (rpc server) S 1234 0", 1234: "1234 (worker) S 1 0"}
    monkeypatch.setattr(pool_module.Path, "read_text", lambda path: stats[int(path.parts[-2])])
    assert pool_module._pid_descends_from(5678, {1234})
    assert not pool_module._pid_descends_from(5678, {9999})


@dataclass(frozen=True)
class _PoolCredentials:
    worker: MeshPoolToken
    management: MeshPoolToken

    @property
    def pool_id(self):
        return self.worker.pool_id

    @property
    def manager_endpoint(self):
        return self.worker.manager_endpoint

    @property
    def pool_secret(self):
        return self.worker.pool_secret

    def encode(self):
        return self.worker.encode()


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


@pytest.fixture()
def pool(tmp_path):
    state_dir, token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    server = serve_pool_manager(state_dir, host="127.0.0.1", port=0)
    host, port = server.server_address
    endpoint = f"http://{host}:{port}"
    token = MeshPoolToken(
        pool_id=token.pool_id,
        manager_endpoint=endpoint,
        pool_secret=token.pool_secret,
        scope=token.scope,
    )
    admin_token = MeshPoolToken(
        pool_id=admin_token.pool_id,
        manager_endpoint=endpoint,
        pool_secret=admin_token.pool_secret,
        scope=admin_token.scope,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield endpoint, _PoolCredentials(worker=token, management=admin_token)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met within timeout")


def _call(endpoint, token, route, body):
    management_routes = {
        "/v1/pool/status",
        "/v1/pool/launch",
        "/v1/pool/stop",
        "/v1/pool/chat",
        "/v1/pool/register-model",
        "/v1/pool/remove-worker",
    }
    auth = (
        {"management_secret": token.management.pool_secret}
        if route in management_routes
        else {"pool_secret": token.worker.pool_secret}
    )
    return post_json(endpoint + route, {**auth, **body}, timeout=5.0)


def _join(endpoint, token, worker_id, *, vram=48, model_bytes=5_000_000_000):
    return _call(
        endpoint,
        token,
        "/v1/pool/join",
        {
            "worker_id": worker_id,
            "capability": {"gpu_name": "test", "vram_gb": vram},
            "catalog": [{"model_id": "m1", "model_bytes": model_bytes}],
            "endpoints": {
                "rpc": f"{worker_id}.local:50052",
                "proof": f"http://{worker_id}.local:9402",
                "mesh": f"http://{worker_id}.local:9500",
            },
        },
    )


def _assert_command_id(command):
    command_id = str(command.get("command_id", ""))
    assert command_id.startswith("cmd-")
    assert len(command_id) == len("cmd-") + 32
    assert all(character in "0123456789abcdef" for character in command_id[4:])
    command_digest = str(command.get("command_digest", ""))
    assert len(command_digest) == 64
    assert command_digest == command_digest.lower()
    assert all(character in "0123456789abcdef" for character in command_digest)
    return command_id


def _command_report(command, *, worker_id, event, **fields):
    """Build the command-bound lifecycle envelope emitted by a worker."""

    _assert_command_id(command)
    return {
        "worker_id": worker_id,
        "mesh_key": str(command["mesh_key"]),
        "command_id": str(command["command_id"]),
        "command_digest": str(command["command_digest"]),
        "event": event,
        **fields,
    }


def _mesh_score_metadata(
    *,
    coordinator,
    model_index,
    model_id,
    score_epoch,
    ema,
    scored_epochs=1,
    chain_id=945,
    netuid=405,
    mesh_id="scored-mesh",
    snapshot_hash="11" * 32,
    snapshot_generation=1,
):
    return {
        "scored_epochs": scored_epochs,
        "last_scored_epoch": score_epoch,
        "score_epoch": score_epoch,
        "latest_completed_ema": ema,
        "ema_scope": "coordinator_model_slot",
        "ema_carries_across_topology": True,
        "latest_mesh_sample": {
            "chain_id": chain_id,
            "netuid": netuid,
            "coordinator_address": coordinator,
            "model_index": model_index,
            "model_id": model_id,
            "mesh_id": mesh_id,
            "verification_snapshot_hash": snapshot_hash,
            "snapshot_generation": snapshot_generation,
        },
    }


def test_pool_token_roundtrip(tmp_path):
    _, token = create_pool_state(
        tmp_path, manager_endpoint="http://mgr:9500", serving_mode="dev"
    )
    decoded = MeshPoolToken.decode(token.encode())
    assert decoded == token
    with pytest.raises(ValueError):
        MeshPoolToken.decode("vtmesh_notpool")


def test_pool_manager_tls_wraps_server_socket(monkeypatch, tmp_path):
    state_dir, _token = create_pool_state(
        tmp_path,
        manager_endpoint="https://manager.example:9543",
        serving_mode="dev",
    )
    fake_server = SimpleNamespace(socket=object())
    fake_context = SimpleNamespace(
        minimum_version=None,
        options=0,
        loaded=None,
        wrapped=None,
    )

    def load_cert_chain(*, certfile, keyfile):
        fake_context.loaded = (certfile, keyfile)

    def wrap_socket(sock, *, server_side, do_handshake_on_connect):
        fake_context.wrapped = (sock, server_side, do_handshake_on_connect)
        return "tls-socket"

    fake_context.load_cert_chain = load_cert_chain
    fake_context.wrap_socket = wrap_socket
    key_path = tmp_path / "key.pem"
    key_path.write_text("test key")
    key_path.chmod(0o600)
    monkeypatch.setattr(
        pool_module,
        "ThreadingHTTPServer",
        lambda _address, _handler: fake_server,
    )
    monkeypatch.setattr(
        pool_module.ssl,
        "SSLContext",
        lambda _protocol: fake_context,
    )

    server = serve_pool_manager(
        state_dir,
        host="127.0.0.1",
        port=9543,
        tls_certfile=tmp_path / "cert.pem",
        tls_keyfile=key_path,
    )

    assert server.socket == "tls-socket"
    assert server.tls_enabled is True
    assert fake_context.loaded == (
        str(tmp_path / "cert.pem"),
        str(key_path),
    )
    assert fake_context.wrapped[1] is True
    # Handshake must be DEFERRED to the handler thread: handshaking in
    # accept() lets one silent client wedge the whole listener.
    assert fake_context.wrapped[2] is False
    assert fake_context.minimum_version == pool_module.ssl.TLSVersion.TLSv1_2


def test_pool_manager_tls_requires_cert_and_key(tmp_path):
    state_dir, _token = create_pool_state(
        tmp_path,
        manager_endpoint="https://manager.example:9543",
        serving_mode="dev",
    )
    with pytest.raises(ValueError, match="requires both"):
        serve_pool_manager(state_dir, tls_certfile=tmp_path / "cert.pem")
    with pytest.raises(ValueError, match="requires both"):
        serve_pool_manager(state_dir, tls_keyfile=tmp_path / "key.pem")


def test_pool_manager_tls_rejects_public_private_key(tmp_path):
    state_dir, _token = create_pool_state(
        tmp_path,
        manager_endpoint="https://manager.example:9543",
        serving_mode="dev",
    )
    key_path = tmp_path / "key.pem"
    key_path.write_text("test key")
    key_path.chmod(0o644)

    with pytest.raises(PermissionError, match="group or world"):
        serve_pool_manager(
            state_dir,
            tls_certfile=tmp_path / "cert.pem",
            tls_keyfile=key_path,
        )


def _local_runner_config(tmp_path, *, wallet: bool = False):
    return PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-test",
            manager_endpoint="http://manager.local:9500",
            pool_secret="test-secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "worker",
        advertise_host="worker.local",
        rpc_port=56152,
        proof_port=19602,
        mesh_port=19643,
        llama_server_binary="llama-server",
        rpc_worker_binary="rpc-server",
        catalog=[
            {
                "model_id": "m1",
                "llama_model": str(tmp_path / "model.gguf"),
                "manifest": str(tmp_path / "manifest.json"),
                "layers": 4,
                "model_bytes": 1,
            }
        ],
        wallet_name="miner" if wallet else "",
        wallet_hotkey="default" if wallet else "",
    )


def test_auto_backend_build_targets_detected_cuda_architecture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    patch_dir = tmp_path / "patches" / "llama.cpp"
    patch_dir.mkdir(parents=True)
    (patch_dir / "0001-verathos-proof-capture-cuda-cpu-rpc-server.patch").write_bytes(
        b"test patch"
    )
    # The CUDA build applies the anchor patch too; its bytes are part of the
    # build-manifest digest so a stale binary is rebuilt rather than served.
    (patch_dir / "0003-verathos-streaming-execution-anchors.patch").write_bytes(
        b"test anchor patch"
    )
    (patch_dir / "0005-verathos-mmid-decode-intra-parity.patch").write_bytes(
        b"test mmid parity patch"
    )
    (patch_dir / "0007-verathos-rpc-foreign-view-serialization.patch").write_bytes(
        b"test view split patch"
    )
    (patch_dir / "0008-verathos-wildcard-op-arming.patch").write_bytes(
        b"test wildcard arming patch"
    )
    (patch_dir / "0009-verathos-name-keyed-op-arming.patch").write_bytes(
        b"test name arming patch"
    )
    (patch_dir / "build.sh").write_text("#!/usr/bin/env bash\n")
    (patch_dir / "UPSTREAM_BASE.txt").write_text("base-commit\n")
    monkeypatch.setattr(pool_module, "_detect_backend_arch", lambda: ("cuda", "sm89"))

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        bin_dir = (
            tmp_path
            / ".cache"
            / "verathos-mesh-runtime"
            / "auto-cuda-sm89"
            / "build-verathos-cuda"
            / "bin"
        )
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server").write_bytes(b"llama")
        (bin_dir / "verathos-rpc-server").write_bytes(b"rpc")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pool_module.subprocess, "run", fake_run)

    llama, rpc = pool_module._auto_backend_binaries(
        _local_runner_config(tmp_path),
        lambda _status: None,
    )

    assert calls == [
        [
            "bash",
            str(patch_dir / "build.sh"),
            "--backend",
            "cuda",
            "--src",
            str(
                tmp_path
                / ".cache"
                / "verathos-mesh-runtime"
                / "auto-cuda-sm89"
            ),
            "--cuda-architectures",
            "89",
        ]
    ]
    assert llama.endswith("/auto-cuda-sm89/build-verathos-cuda/bin/llama-server")
    assert rpc.endswith(
        "/auto-cuda-sm89/build-verathos-cuda/bin/verathos-rpc-server"
    )
    manifest = (
        tmp_path
        / ".cache"
        / "verathos-mesh-runtime"
        / "auto-cuda-sm89"
        / "build-verathos-cuda"
        / "bin"
        / "verathos-build.json"
    )
    recorded = json.loads(manifest.read_text())
    assert recorded["upstream_base"] == "base-commit"
    assert recorded["build_script_sha256"]


def test_pool_runner_persists_distinct_stage_key_and_worker_flag(tmp_path):
    config = _local_runner_config(tmp_path, wallet=True)
    first = LocalMeshRunner(config)
    second = LocalMeshRunner(config)
    key_file = config.workdir / "stage-proof-key.seed"

    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert first._stage_proof_key_ss58 == second._stage_proof_key_ss58
    command = first._worker_serve_cmd(
        tmp_path / "mesh",
        config.catalog[0],
    )
    flag_index = command.index("--stage-proof-key-file")
    assert command[flag_index + 1] == str(key_file.resolve())
    assert "--wallet-name" not in command
    assert "--wallet-hotkey" not in command


def test_pool_runner_binds_worker_services_to_all_interfaces(tmp_path):
    """Behind NAT or a port-mapping fabric the advertised public address is
    not a local interface; binding it fails the whole serve process. Bind
    0.0.0.0 and keep the advertise host only in the ADVERTISED endpoints."""
    config = _local_runner_config(tmp_path)
    config.advertise_host = "203.0.113.9"
    runner = LocalMeshRunner(config)

    command = runner._worker_serve_cmd(
        tmp_path / "mesh",
        config.catalog[0],
    )
    serve_args = command[command.index("serve") :]
    parsed = mesh_cli.build_parser().parse_args(serve_args)

    assert parsed.host == "0.0.0.0"
    assert parsed.rpc_host == "0.0.0.0"


def test_serve_parallel_slots_floor_and_env(monkeypatch):
    # Production meshes serve CONCURRENT traffic: eight slots minimum.
    # This floor was silently lost once (a single-slot pin from before the
    # --parallel audit certification survived into every launch), so the
    # value is pinned here as a capability, not a tunable detail.
    monkeypatch.delenv("VERATHOS_MESH_SERVE_PARALLEL", raising=False)
    assert pool_module._serve_parallel_slots() == 8
    monkeypatch.setenv("VERATHOS_MESH_SERVE_PARALLEL", "16")
    assert pool_module._serve_parallel_slots() == 16
    # Below the floor (or garbage) clamps UP: concurrency cannot sneak
    # back to a serialized serve through an env typo.
    monkeypatch.setenv("VERATHOS_MESH_SERVE_PARALLEL", "1")
    assert pool_module._serve_parallel_slots() == 8
    monkeypatch.setenv("VERATHOS_MESH_SERVE_PARALLEL", "nope")
    assert pool_module._serve_parallel_slots() == 8


def test_drive_kv_autofit_descends_on_alloc_failure_and_caches(
    tmp_path,
    monkeypatch,
):
    """vLLM-style auto-fit by measurement: a model-max unified KV budget
    that fails ALLOCATION descends the bounded ladder (first stop 131072)
    and the fitted value is cached per model so later launches start
    there. Non-allocation failures re-raise untouched."""

    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    # Plenty of free VRAM: these tests exercise the KV ladder, and the
    # audit-reserve micro-batch descent must stay out of their way (the
    # real nvidia-smi on a busy dev box would otherwise trigger it).
    monkeypatch.setattr(
        pool_module, "_gpu_free_vram_mb", lambda: [80 * 1024]
    )

    ctx_per_spawn: list[str] = []

    def fake_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            ctx_per_spawn.append(
                command[command.index("--llama-ctx-size") + 1]
            )

    waits = {"n": 0}

    def fake_wait_http(url, **_kwargs):
        if "/health" not in url:
            return
        waits["n"] += 1
        if waits["n"] == 1:
            raise RuntimeError(
                "backend llama-server failed: cudaMalloc failed: "
                "out of memory"
            )

    class StopAfterFit(RuntimeError):
        pass

    def fake_join_mesh(**_kwargs):
        raise StopAfterFit

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", fake_spawn)
    monkeypatch.setattr(runner, "_wait_http", fake_wait_http)
    monkeypatch.setattr(
        runner, "_llama_first_batch_probe", lambda *_a, **_k: None
    )
    monkeypatch.setattr(runner, "_wait_tcp", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_scrape_backend_error", lambda: "")
    monkeypatch.setattr(pool_module, "_prewarm_proof_cache", finish_prewarm)
    monkeypatch.setattr(pool_module, "join_mesh", fake_join_mesh)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(StopAfterFit):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 2,
            }
        )

    # Attempt 1 at the model max (0), attempt 2 at the first ladder stop.
    assert ctx_per_spawn == ["0", "131072"]
    fit = json.loads((tmp_path / "worker" / "kv-fit-m1.json").read_text())
    assert fit["ctx_budget"] == 131072

    # A later drive starts from the cached fit instead of re-failing.
    ctx_per_spawn.clear()
    waits["n"] = 99  # health passes immediately now
    with pytest.raises(StopAfterFit):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 2,
            }
        )
    assert ctx_per_spawn == ["131072"]


def test_drive_first_batch_probe_descends_ladder_single_box(
    tmp_path,
    monkeypatch,
):
    """Health passing is "loads", not "serves": a single-box drive whose
    FIRST full batch dies on an allocation must descend the KV ladder
    (terminating the still-alive attempt), never ship a budget that
    crash-loops on request one."""

    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    # Plenty of free VRAM: these tests exercise the KV ladder, and the
    # audit-reserve micro-batch descent must stay out of their way (the
    # real nvidia-smi on a busy dev box would otherwise trigger it).
    monkeypatch.setattr(
        pool_module, "_gpu_free_vram_mb", lambda: [80 * 1024]
    )

    ctx_per_spawn: list[str] = []

    def fake_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            ctx_per_spawn.append(
                command[command.index("--llama-ctx-size") + 1]
            )

    probes = {"n": 0}

    def fake_probe(_port, _model_bytes=0):
        probes["n"] += 1
        if probes["n"] == 1:
            raise RuntimeError(
                "first-batch probe failed: allocating 33056.56 MiB on "
                "device 0: cudaMalloc failed: out of memory"
            )

    class StopAfterFit(RuntimeError):
        pass

    def fake_join_mesh(**_kwargs):
        raise StopAfterFit

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", fake_spawn)
    monkeypatch.setattr(runner, "_wait_http", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_wait_tcp", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_llama_first_batch_probe", fake_probe)
    monkeypatch.setattr(runner, "_scrape_backend_error", lambda: "")
    monkeypatch.setattr(pool_module, "_prewarm_proof_cache", finish_prewarm)
    monkeypatch.setattr(pool_module, "join_mesh", fake_join_mesh)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(StopAfterFit):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 1,
            }
        )

    # Attempt 1 at the model max died in the PROBE (after health), attempt
    # 2 at the first ladder stop survived it; the surviving value cached.
    assert ctx_per_spawn == ["0", "131072"]
    assert probes["n"] == 2
    fit = json.loads((tmp_path / "worker" / "kv-fit-m1.json").read_text())
    assert fit["ctx_budget"] == 131072


def test_drive_kv_ladder_raise_paths_terminate_attempt(
    tmp_path,
    monkeypatch,
):
    """The ladder must end the failed attempt on every exit, including
    the raise paths, and escalate to SIGKILL when SIGTERM is ignored."""

    import subprocess

    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)

    class WedgedProc:
        """Ignores SIGTERM (llama wedged in CUDA teardown)."""

        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("llama", timeout or 0)
            return 0

    spawned: list[WedgedProc] = []

    def fake_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            proc = WedgedProc()
            spawned.append(proc)
            runner.procs.append(proc)

    def fake_wait_http(url, **_kwargs):
        if "/health" in url:
            raise RuntimeError(
                "backend llama-server failed: cudaMalloc failed: "
                "out of memory"
            )

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", fake_spawn)
    monkeypatch.setattr(runner, "_wait_http", fake_wait_http)
    monkeypatch.setattr(runner, "_wait_tcp", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner, "_llama_first_batch_probe", lambda *_a, **_k: None
    )
    monkeypatch.setattr(runner, "_scrape_backend_error", lambda: "")

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(RuntimeError, match="does not fit"):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 1,
            }
        )

    # Every rung was attempted (model max + full ladder), and every
    # attempt - INCLUDING the final one whose failure raised - was
    # terminated and then killed (SIGTERM was ignored).
    # Every rung of the context ladder plus the micro-batch descent: when
    # the compute buffer OOMs at the ladder floor, drive halves -ub
    # 4096 -> 2048 -> 1024 -> 512 before declaring the attempt dead
    # (qwen3.8 q4-k-xl 24GB fix).
    assert len(spawned) == 1 + len(pool_module._KV_FIT_LADDER) + 3
    assert all(p.terminated and p.killed for p in spawned)
    assert runner.procs == []


def test_scrape_backend_error_scoped_to_marked_offset(tmp_path):
    """A previous rung's cudaMalloc line in the shared append-mode log
    must not classify the CURRENT attempt's failure ."""

    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    log = config.workdir / "pool-driver-coordinator.log"
    log.write_text("ggml_backend_cuda: cudaMalloc failed: out of memory\n")

    # Unmarked (fresh runner): full file visible - old behavior.
    assert "out of memory" in runner._scrape_backend_error()

    # Marked at the next attempt's spawn: the stale line is invisible...
    runner._mark_backend_log_offsets()
    assert runner._scrape_backend_error() == ""

    # ...but the CURRENT attempt's crash still surfaces.
    with log.open("a") as handle:
        handle.write("GGML_ASSERT(ok) failed at ggml-cuda.cu:4746\n")
    assert "GGML_ASSERT" in runner._scrape_backend_error()
    assert "out of memory" not in runner._scrape_backend_error()


def test_pool_driver_serves_concurrent_parallel_slots(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("VERATHOS_MESH_SERVE_PARALLEL", raising=False)
    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    captured: dict[str, list[str]] = {}

    class CoordinatorCommandCaptured(RuntimeError):
        pass

    def capture_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            captured["command"] = list(command)
            raise CoordinatorCommandCaptured

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", capture_spawn)
    monkeypatch.setattr(pool_module, "_prewarm_proof_cache", finish_prewarm)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(CoordinatorCommandCaptured):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 2,
                "max_context_len": 32_768,
            }
        )

    coordinator_command = captured["command"]
    serve_args = coordinator_command[coordinator_command.index("serve") :]
    parsed = mesh_cli.build_parser().parse_args(serve_args)
    # All interfaces: self-tests dial loopback (the internal-HMAC lane on
    # validator routes is loopback-only) while validators and members dial
    # the advertise address.
    assert parsed.host == "0.0.0.0"
    assert parsed.llama_extra_arg == [
        "--kv-unified",
        "-b",
        "8192",
        "-ub",
        "4096",
        "--parallel",
        "8",
    ]
    assert mesh_cli.llama_n_parallel_from_args(parsed) == 8
    assert parsed.llama_device == "RPC0,RPC1"
    assert parsed.llama_tensor_split == ""
    # Unified KV: --ctx-size is ONE shared budget (kv-unified above),
    # equal to the registry max_context_len, so a single request can use
    # the full advertised context and concurrent ones pack beside it.
    assert parsed.llama_ctx_size == 32_768

    backend_argv = build_llama_server_command(
        binary=parsed.llama_server_binary,
        model=parsed.llama_model,
        host=parsed.llama_host,
        port=parsed.llama_port,
        rpc_endpoints=["worker.local:50052"],
        device=parsed.llama_device,
        n_gpu_layers=parsed.llama_n_gpu_layers,
        ctx_size=parsed.llama_ctx_size,
        tensor_split=parsed.llama_tensor_split,
        alias=parsed.llama_alias or "m1",
        extra_args=parsed.llama_extra_arg,
    )
    parallel_index = backend_argv.index("--parallel")
    assert backend_argv[parallel_index + 1] == "8"
    assert "--kv-unified" in backend_argv
    context_index = backend_argv.index("--ctx-size")
    assert backend_argv[context_index + 1] == "32768"
    with pytest.raises(ValueError, match="must not override"):
        build_llama_server_command(
            binary=parsed.llama_server_binary,
            model=parsed.llama_model,
            ctx_size=parsed.llama_ctx_size,
            extra_args=["--ctx-size=65536"],
        )


def test_pool_driver_single_worker_dev_mesh_computes_locally(
    tmp_path,
    monkeypatch,
):
    """LOCAL STAGE: a single-worker dev mesh runs llama-server directly on
    the worker's own devices - no rpc-server, no --rpc, capture redirected
    into the member trace dir - removing the per-token loopback rpc hop."""
    config = _local_runner_config(tmp_path)
    config.rpc_device = "CUDA0,CUDA1"
    config.per_gpu_vram_gb = [80, 80]
    config.gpu_names = ["A100", "A100"]
    runner = LocalMeshRunner(config)
    captured: dict[str, list[str]] = {}

    class CoordinatorCommandCaptured(RuntimeError):
        pass

    def capture_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            captured["command"] = list(command)
            raise CoordinatorCommandCaptured

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", capture_spawn)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(CoordinatorCommandCaptured):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 1,
                "member_device_counts": [2],
            }
        )

    coordinator_command = captured["command"]
    serve_args = coordinator_command[coordinator_command.index("serve") :]
    parsed = mesh_cli.build_parser().parse_args(serve_args)
    assert parsed.llama_device == "CUDA0,CUDA1"
    assert parsed.llama_min_rpc_workers == 0
    assert parsed.llama_tensor_split == "80,80"
    # Trace dirs are per MODEL: a worker serves several models from one
    # workdir, and a shared traces/ let a template loader pick up the
    # PREVIOUS model's manifests (observed qwen -> glm contamination).
    assert parsed.llama_capture_trace_dir == str(
        pool_module.model_trace_dir(config.workdir, config.catalog[0]["model_id"])
    )
    # The local-stage coordinator stays orchestration-only: the layers
    # belong to the member's committed stage.
    assert not mesh_cli.coordinator_computes_from_args(parsed)


def test_worker_serve_cmd_without_rpc_worker(tmp_path):
    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    command = runner._worker_serve_cmd(
        tmp_path / "mesh", config.catalog[0], rpc_worker=False
    )
    assert "--rpc-worker" not in command
    assert "--rpc-port" not in command
    trace_index = command.index("--proof-trace-dir")
    assert command[trace_index + 1] == str(
        pool_module.model_trace_dir(config.workdir, config.catalog[0]["model_id"])
    )
    assert config.catalog[0]["model_id"] in command[trace_index + 1]


def test_kv_budget_fills_free_vram_instead_of_the_model_maximum(monkeypatch):
    """The KV budget is sized from what is actually free, vLLM-style.

    A 1M-context model whose weights take 238GB of a 320GB box cannot hold
    its trained maximum, and the coarse ladder's next rung (131072) throws
    away most of the ~400k tokens that DO fit.
    """
    # 79 layers, 1 kv head, k=576 v=512 -> 171,904 bytes per token.
    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 171_904)
    monkeypatch.setattr(pool_module, "_gpu_free_vram_mb", lambda: [320 * 1024])

    tokens = pool_module._kv_tokens_that_fit("/x/model.gguf", 238_577_580_768)
    assert tokens % 4096 == 0
    # 320GiB x 0.90 utilization - 238.6GB weights = ~70GB of KV.
    assert 380_000 < tokens < 430_000
    # Far better than the ladder rung it replaces, which is the whole point.
    assert tokens > pool_module._KV_FIT_LADDER[0]

    # No header (unreadable GGUF) or no GPU: fall back to the measured
    # ladder rather than inventing a number.
    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 0)
    assert pool_module._kv_tokens_that_fit("/x/model.gguf", 1) == 0
    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 171_904)
    monkeypatch.setattr(pool_module, "_gpu_free_vram_mb", lambda: [])
    assert pool_module._kv_tokens_that_fit("/x/model.gguf", 1) == 0
    # Weights alone exceed VRAM: nothing left for KV, so say so with 0.
    monkeypatch.setattr(pool_module, "_gpu_free_vram_mb", lambda: [80 * 1024])
    assert pool_module._kv_tokens_that_fit("/x/model.gguf", 238_577_580_768) == 0


def test_extra_arg_value_index_finds_paired_llama_flags():
    """The ubatch descent edits paired --llama-extra-arg entries in place."""

    cmd = [
        "serve",
        "--llama-extra-arg=-b",
        "--llama-extra-arg=8192",
        "--llama-extra-arg=-ub",
        "--llama-extra-arg=4096",
    ]
    ub = pool_module._extra_arg_value_index(cmd, "-ub")
    b = pool_module._extra_arg_value_index(cmd, "-b")
    assert cmd[ub] == "--llama-extra-arg=4096"
    assert cmd[b] == "--llama-extra-arg=8192"
    assert pool_module._extra_arg_value_index(cmd, "--missing") == -1


def test_kv_budget_reserves_audit_workspace_per_gpu(monkeypatch):
    """Every GPU keeps CAPACITY_AUDIT_VRAM_RESERVE_MB out of the KV budget.

    A budget that packs the card wall to wall makes the hot capacity audit
    workload die in cudaMalloc before B_start — a permanent no_show while
    serving looks healthy ."""

    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 171_904)
    monkeypatch.setattr(pool_module, "_gguf_trained_context", lambda path: 1_048_576)

    reserve_mb = pool_module.CAPACITY_AUDIT_VRAM_RESERVE_MB
    model_bytes = 238_577_580_768

    monkeypatch.setattr(
        pool_module, "_gpu_free_vram_mb", lambda: [80 * 1024] * 4
    )
    four_gpu = pool_module._kv_tokens_that_fit("/x/model.gguf", model_bytes)

    # The same capacity with no reserve would admit strictly more tokens;
    # the difference is exactly the per-GPU holdback (4 GPUs here).
    monkeypatch.setattr(pool_module, "CAPACITY_AUDIT_VRAM_RESERVE_MB", 0)
    no_reserve = pool_module._kv_tokens_that_fit("/x/model.gguf", model_bytes)
    held_back_tokens = (4 * reserve_mb * 1024 * 1024) // 171_904
    assert no_reserve - four_gpu >= held_back_tokens - 4096
    assert no_reserve - four_gpu <= held_back_tokens + 4096


def test_kv_estimate_never_caps_a_model_that_fits_its_trained_context(monkeypatch):
    """Models whose full context already fits must be left alone.

    DeepSeek Flash serves its full 1M and Qwen its 262k today; an estimate
    that quietly capped them (or that exceeded a small model's trained
    maximum) would be a regression, so 0 -- meaning 'use the trained
    maximum' -- is the answer whenever everything fits.
    """
    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 1024)
    monkeypatch.setattr(pool_module, "_gpu_free_vram_mb", lambda: [320 * 1024])

    # A small model on a big box: spare VRAM holds far more than it was
    # trained for, so do not pass a budget beyond the trained maximum.
    monkeypatch.setattr(pool_module, "_gguf_trained_context", lambda path: 262_144)
    assert pool_module._kv_tokens_that_fit("/x/small.gguf", 4_700_000_000) == 0

    # A model that genuinely cannot fit its trained context still gets a
    # measured budget, strictly below that maximum.
    monkeypatch.setattr(pool_module, "_kv_bytes_per_token", lambda path: 171_904)
    monkeypatch.setattr(pool_module, "_gguf_trained_context", lambda path: 1_048_576)
    fitted = pool_module._kv_tokens_that_fit("/x/big.gguf", 238_577_580_768)
    assert 0 < fitted < 1_048_576

    # Unknown trained context (unreadable header): keep the measured value
    # rather than guessing that everything fits.
    monkeypatch.setattr(pool_module, "_gguf_trained_context", lambda path: 0)
    assert pool_module._kv_tokens_that_fit("/x/big.gguf", 238_577_580_768) == fitted


def test_listener_lookup_reads_proc_without_lsof(tmp_path, monkeypatch):
    """A box with no lsof still identifies listeners: /proc is enough.

    lsof is absent from plain container images, and depending on it meant a
    worker could never stop a mesh (stuck 'stopping', then stale).
    """
    tcp = tmp_path / "tcp"
    tcp.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
        "retrnsmt   uid  timeout inode\n"
        "   0: 00000000:24BA 00000000:0000 0A 00000000:00000000 00:00000000 "
        "00000000     0        0 987654 1 0000 100 0 0 10 0\n"
        # Same port but ESTABLISHED, not LISTEN: must be ignored.
        "   1: 00000000:24BA 0100007F:9999 01 00000000:00000000 00:00000000 "
        "00000000     0        0 111111 1 0000 100 0 0 10 0\n"
        # A LISTENer on a different port: must be ignored.
        "   2: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 "
        "00000000     0        0 222222 1 0000 100 0 0 10 0\n"
    )
    assert pool_module._listening_inodes_from_proc(9402, [str(tcp)]) == {987654}
    assert pool_module._listening_inodes_from_proc(8080, [str(tcp)]) == {222222}
    assert pool_module._listening_inodes_from_proc(1234, [str(tcp)]) == set()

    # A listener whose owner cannot be attributed is still occupied, and the
    # caller must refuse rather than assume the port is free.
    monkeypatch.setattr(
        pool_module, "_listening_inodes_from_proc", lambda port: {5}
    )
    monkeypatch.setattr(pool_module, "_pids_holding_inodes", lambda inodes: set())
    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(rpc_port=55052, proof_port=19402, mesh_port=19443)
    runner.procs = []
    with pytest.raises(RuntimeError, match="could not be identified"):
        runner._free_own_ports(driving=False)


def test_pool_runner_reclaims_own_ports_from_escaped_descendants(monkeypatch):
    """A listener that persists on the runner's OWN configured port after the
    tracked children are gone is an escaped descendant (llama in its own
    session under the coordinator supervisor) and is reclaimed — refusing it
    wedged every stop/relaunch into a permanent error mesh until a daemon
    restart . Unidentifiable listeners are
    still refused."""

    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(
        rpc_port=55052,
        proof_port=19402,
        mesh_port=19443,
    )
    runner.procs = []
    runner._thread_command = SimpleNamespace(command_id="")
    monkeypatch.setattr(pool_module.time, "sleep", lambda s: None)
    killed: list[int] = []

    state = {"alive": True}

    def fake_listeners(port):
        if port != 55052:
            return (set(), False)
        return ({4242}, True) if state["alive"] else (set(), False)

    def fake_kill(pid, sig):
        killed.append(pid)
        state["alive"] = False

    monkeypatch.setattr(pool_module, "_listeners_on_port", fake_listeners)
    monkeypatch.setattr(pool_module.os, "kill", fake_kill)
    # Collapse the grace window so the test does not wait 15s.
    ticks = iter([0.0, 100.0] + [200.0] * 50)
    monkeypatch.setattr(
        pool_module.time, "monotonic", lambda: next(ticks, 500.0)
    )

    runner._free_own_ports(driving=False)
    assert killed == [4242]

    # A listener whose owner cannot be identified is never blind-killed.
    state["alive"] = True
    monkeypatch.setattr(
        pool_module, "_listeners_on_port", lambda port: (set(), True)
    )
    ticks2 = iter([0.0, 100.0] + [200.0] * 50)
    monkeypatch.setattr(
        pool_module.time, "monotonic", lambda: next(ticks2, 500.0)
    )
    with pytest.raises(RuntimeError, match="could not be identified"):
        runner._free_own_ports(driving=False)


def test_pool_runner_clears_only_tracked_process_groups(monkeypatch):
    fallback_kills: list[int] = []
    proc = SimpleNamespace(
        pid=7331,
        poll=lambda: None,
        kill=lambda: fallback_kills.append(7331),
    )
    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(
        rpc_port=55052,
        proof_port=19402,
        mesh_port=19443,
    )
    runner.procs = [proc]
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        pool_module.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    monkeypatch.setattr(
        pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="",
        ),
    )

    runner._free_own_ports(driving=True)
    assert killed == [(7331, pool_module.signal.SIGKILL)]
    assert fallback_kills == []
    assert runner.procs == []


def test_pool_runner_does_not_kill_reaped_process_group(monkeypatch):
    proc = SimpleNamespace(
        pid=7331,
        poll=lambda: 1,
        kill=lambda: pytest.fail("reaped process fallback kill must not run"),
    )
    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(
        rpc_port=55052,
        proof_port=19402,
        mesh_port=19443,
    )
    runner.procs = [proc]
    monkeypatch.setattr(
        pool_module.os,
        "killpg",
        lambda *_args: pytest.fail("reaped process group must not be killed"),
    )
    monkeypatch.setattr(
        pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="",
        ),
    )

    runner._free_own_ports(driving=False)
    assert runner.procs == []


def test_pool_runner_fails_closed_when_listener_inspection_fails(monkeypatch):
    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(
        rpc_port=55052,
        proof_port=19402,
        mesh_port=19443,
    )
    runner.procs = []
    # No /proc (macOS worker): the lsof branch runs, and a failed inspection
    # is never read as "the port is free".
    monkeypatch.setattr(pool_module, "_proc_net_available", lambda: False)
    monkeypatch.setattr(
        pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="permission denied",
        ),
    )

    with pytest.raises(RuntimeError, match="cannot verify ownership.*permission denied"):
        runner._free_own_ports(driving=False)


def test_pool_runner_surfaces_terminal_plan_failure(tmp_path):
    runner = object.__new__(LocalMeshRunner)
    runner.config = SimpleNamespace(workdir=tmp_path)
    runner.procs = []
    runner._thread_command = threading.local()
    failure = tmp_path / "backend-failure.json"
    failure.write_text(
        json.dumps(
            {
                "component": "llama-plan",
                "crashes": 0,
                "error": "RPC member layer ranges do not match committed tensor split",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="backend llama-plan failed.*do not match committed tensor split",
    ):
        runner._wait_http(
            "http://127.0.0.1:1/health",
            timeout=0.1,
            failure_file=failure,
        )


def test_pool_join_advertises_stage_public_key(monkeypatch, tmp_path):
    import verallm.mesh.pool as mesh_pool

    config = _local_runner_config(tmp_path)
    config.gpu_name = "Test GPU"
    config.vram_gb = 24
    runner = LocalMeshRunner(config)
    joined = MeshSpec.new_private_mesh(
        coordinator_uid=7,
        coordinator_hotkey="5Coordinator",
        endpoint="http://coordinator.local:9443",
        model_id="m1",
        model_package_hash="a" * 64,
        total_layers=4,
    )
    captured: dict[str, object] = {}

    def fake_join_mesh(**kwargs):
        captured.update(kwargs)
        return tmp_path / "joined", joined

    readiness = []
    monkeypatch.setattr(mesh_pool, "join_mesh", fake_join_mesh)
    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_wait_http",
        lambda url, **_kwargs: readiness.append(("http", url)),
    )
    monkeypatch.setattr(
        runner,
        "_wait_tcp",
        lambda host, port, **_kwargs: readiness.append(("tcp", host, port)),
    )

    assert runner.join({"join_token": "opaque"}) == {"event": "serving"}
    assert captured["hotkey"] == runner._stage_proof_key_ss58
    assert captured["gpu_name"] == "Test GPU"
    assert captured["vram_gb"] == 24
    # Readiness dials loopback: these are the runner's own children, and the
    # advertise host may not hairpin from inside a NAT.
    assert readiness == [
        ("http", f"http://127.0.0.1:{config.proof_port}/health"),
        ("tcp", "127.0.0.1", config.rpc_port),
    ]


def test_pool_driver_local_join_advertises_driver_vram(monkeypatch, tmp_path):
    config = _local_runner_config(tmp_path)
    config.gpu_name = "RTX 4090"
    config.vram_gb = 24
    runner = LocalMeshRunner(config)
    captured: dict[str, object] = {}

    class LocalJoinCaptured(RuntimeError):
        pass

    def capture_join(**kwargs):
        captured.update(kwargs)
        raise LocalJoinCaptured

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(pool_module, "join_mesh", capture_join)
    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_wait_http", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_llama_first_batch_probe", lambda *_a, **_k: None
    )
    monkeypatch.setattr(pool_module, "_prewarm_proof_cache", finish_prewarm)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    with pytest.raises(LocalJoinCaptured):
        runner.drive(
            {
                "action": "drive",
                "serving_mode": "dev",
                "model_id": "m1",
                "member_count": 1,
                "member_vram": [24],
            }
        )

    assert captured["gpu_name"] == "RTX 4090"
    assert captured["vram_gb"] == 24


def test_status_heals_orphaned_serving_mesh(tmp_path):
    """A mesh record no worker references any more can never advance again
    (nothing will ever report for it), yet it used to render "serving"
    indefinitely - a dead mesh key stayed "serving" in pool-state.json
    beside the live replacement mesh and misled operators and tooling
    that read pool state instead of asking the coordinator. The status read
    must heal it to an explicit,
    persisted error."""

    state_dir, token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    state_path = state_dir / "pool-state.json"
    state = json.loads(state_path.read_text())
    # The stale record: its member worker still exists but was released
    # and moved on (mesh reference cleared), so nothing will ever report
    # for this mesh again - yet its status still claims serving. (A mesh
    # whose worker RECORDS are gone entirely is already healed at manager
    # init; this is the runtime supersession case.)
    state.setdefault("workers", {})["ghost"] = {
        "worker_id": "ghost",
        "capability": {"gpu_name": "test", "vram_gb": 24},
        "catalog": [],
        "endpoints": {},
        "status": "idle",
        "mesh": "",
        "last_seen_unix": 0,
        "rtt_ms": {},
        "peer_rtt_ms": {},
        "commands": [],
        "command_inflight": None,
    }
    state["meshes"]["m-deadbeef00"] = {
        "mesh_key": "m-deadbeef00",
        "model_id": "m1",
        "members": ["ghost"],
        "driver": "ghost",
        "status": "serving",
        "serving": ["ghost"],
        "serving_mode": "dev",
        "created_at_unix": 0,
    }
    state_path.write_text(json.dumps(state))

    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    server = serve_pool_manager(state_dir, host="127.0.0.1", port=0)
    host, port = server.server_address
    endpoint = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = post_json(
            endpoint + "/v1/pool/status",
            {"management_secret": admin_token.pool_secret},
            timeout=5.0,
        )
        mesh = status["meshes"]["m-deadbeef00"]
        assert mesh["status"] == "error"
        assert "no worker is assigned" in mesh["error"]
        # The healing is persisted: file readers see reality too.
        persisted = json.loads(state_path.read_text())
        assert persisted["meshes"]["m-deadbeef00"]["status"] == "error"

        # A mesh whose members still reference it is untouched: launch a
        # real one and confirm the reconciliation never fires on it.
        worker_token = MeshPoolToken(
            pool_id=token.pool_id,
            manager_endpoint=endpoint,
            pool_secret=token.pool_secret,
            scope=token.scope,
        )
        creds = _PoolCredentials(
            worker=worker_token,
            management=MeshPoolToken(
                pool_id=admin_token.pool_id,
                manager_endpoint=endpoint,
                pool_secret=admin_token.pool_secret,
                scope=admin_token.scope,
            ),
        )
        _join(endpoint, creds, "alpha", vram=24)
        launched = _call(
            endpoint, creds, "/v1/pool/launch",
            {"model_id": "m1", "workers": ["alpha"], "driver": "alpha"},
        )
        status = _call(endpoint, creds, "/v1/pool/status", {})
        assert status["meshes"][launched["mesh_key"]]["status"] == "driving"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_heartbeat_persists_worker_status_transitions(pool, tmp_path):
    """Worker status transitions arriving via heartbeat must reach
    pool-state.json: the file is read by tooling that never dials the
    live status route, and heartbeat-only updates used to stay
    memory-only until an unrelated save (regression). Progress ticks
    within one phase stay off the disk-write path."""

    endpoint, token = pool
    _join(endpoint, token, "alpha", vram=24)
    # The pool fixture shares this test's tmp_path, so the state file
    # lives under it.
    candidates = sorted(tmp_path.rglob("pool-state.json"))
    assert candidates, "pool fixture must persist a pool-state.json"
    state_path = candidates[0]
    assert "alpha" in (
        json.loads(state_path.read_text()).get("workers") or {}
    )

    _call(
        endpoint, token, "/v1/pool/heartbeat",
        {"worker_id": "alpha", "status": "fetching 10%"},
    )
    persisted = json.loads(state_path.read_text())
    assert persisted["workers"]["alpha"]["status"] == "fetching 10%"

    # Same phase, new percentage: memory updates, disk stays quiet.
    _call(
        endpoint, token, "/v1/pool/heartbeat",
        {"worker_id": "alpha", "status": "fetching 42%"},
    )
    persisted = json.loads(state_path.read_text())
    assert persisted["workers"]["alpha"]["status"] == "fetching 10%"
    live = _call(endpoint, token, "/v1/pool/status", {})
    assert live["workers"]["alpha"]["status"] == "fetching 42%"


def test_pool_rejects_bad_secret(pool):
    endpoint, token = pool
    with pytest.raises(RuntimeError, match="HTTP 403"):
        post_json(endpoint + "/v1/pool/status", {"pool_secret": "wrong"}, timeout=5.0)


def test_join_launch_drive_join_flow(pool):
    endpoint, token = pool
    assert _join(endpoint, token, "alpha", vram=24)["worker_id"] == "alpha"
    assert _join(endpoint, token, "beta", vram=49)["worker_id"] == "beta"

    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m1", "workers": ["alpha", "beta"], "driver": "alpha"},
    )
    mesh_key = launched["mesh_key"]
    assert launched["driver"] == "alpha"

    # Driver's next heartbeat carries the drive command; beta has nothing yet.
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "alpha"})
    drive_command = beat["command"]
    drive_command_id = _assert_command_id(drive_command)
    assert drive_command["action"] == "drive"
    assert drive_command["member_count"] == 2
    assert "member_vram" not in drive_command
    assert _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "beta"})["command"] is None

    # Busy workers cannot be double-launched.
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _call(endpoint, token, "/v1/pool/launch", {"model_id": "m1", "workers": ["alpha"]})

    # drive_ready fans the join token out to the other member.
    _call(
        endpoint, token, "/v1/pool/report",
        _command_report(
            drive_command,
            worker_id="alpha",
            event="drive_ready",
            mesh_id="mesh-x",
            join_token="vtmesh_tok",
            coordinator_endpoint="http://alpha.local:9500",
        ),
    )
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "beta"})
    join_command = beat["command"]
    join_command_id = _assert_command_id(join_command)
    assert {
        key: join_command[key]
        for key in ("action", "mesh_key", "join_token", "coordinator_endpoint")
    } == {
        "action": "join", "mesh_key": mesh_key, "join_token": "vtmesh_tok",
        "coordinator_endpoint": "http://alpha.local:9500",
    }

    joined = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(join_command, worker_id="beta", event="serving"),
    )
    assert joined["command_completed"] == join_command_id
    # Member joined, but a multi-box mesh is not serving until the DRIVER has
    # verified the fully-joined split backend actually answers.
    assert _call(endpoint, token, "/v1/pool/status", {})["meshes"][mesh_key]["status"] == "joining"
    driven = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(drive_command, worker_id="alpha", event="serving"),
    )
    assert driven["command_completed"] == drive_command_id
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "serving"
    # The driver runs the coordinator AND serves its own stage, so once the
    # mesh is serving the driver is "serving" too (not merely "driving").
    assert status["workers"]["alpha"]["status"] == "serving"
    assert status["workers"]["beta"]["status"] == "serving"

    # Stop reserves both workers and retains a tombstone until each worker's
    # signed, command-bound completion confirms its local processes are gone.
    stopped = _call(endpoint, token, "/v1/pool/stop", {"mesh_key": mesh_key})
    assert stopped == {"status": "stopping", "mesh_key": mesh_key}
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "stopping"
    assert status["workers"]["alpha"]["status"] == "stopping"
    assert status["workers"]["beta"]["status"] == "stopping"

    stop_commands = {}
    for worker_id in ("alpha", "beta"):
        beat = _call(
            endpoint,
            token,
            "/v1/pool/heartbeat",
            {"worker_id": worker_id},
        )
        stop_commands[worker_id] = beat["command"]
        _assert_command_id(stop_commands[worker_id])
        assert beat["command"]["action"] == "stop"
        receipt = _call(
            endpoint,
            token,
            "/v1/pool/heartbeat",
            {
                "worker_id": worker_id,
                "command_ack": stop_commands[worker_id]["command_id"],
            },
        )
        assert receipt["command_acknowledged"] == stop_commands[worker_id][
            "command_id"
        ]
        assert receipt["command"] == stop_commands[worker_id]

    first_stopped = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            stop_commands["alpha"],
            worker_id="alpha",
            event="stopped",
        ),
    )
    assert first_stopped["command_completed"] == stop_commands["alpha"][
        "command_id"
    ]
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["workers"]["alpha"]["status"] == "idle"
    assert status["workers"]["beta"]["status"] == "stopping"
    assert status["meshes"][mesh_key]["status"] == "stopping"

    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            stop_commands["beta"],
            worker_id="beta",
            event="stopped",
        ),
    )
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["workers"]["alpha"]["status"] == "idle"
    assert status["workers"]["beta"]["status"] == "idle"
    assert mesh_key not in status["meshes"]


def test_pool_serializes_member_joins_in_committed_rpc_order(pool):
    endpoint, token = pool
    for worker_id, vram in (
        ("alpha", 24),
        ("beta", 49),
        ("gamma", 80),
    ):
        assert _join(
            endpoint,
            token,
            worker_id,
            vram=vram,
        )["worker_id"] == worker_id

    launched = _call(
        endpoint,
        token,
        "/v1/pool/launch",
        {
            "model_id": "m1",
            "workers": ["gamma", "alpha", "beta"],
            "driver": "alpha",
        },
    )
    mesh_key = launched["mesh_key"]
    drive = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha"},
    )["command"]
    drive_id = _assert_command_id(drive)
    assert drive["action"] == "drive"

    ready = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            drive,
            worker_id="alpha",
            event="drive_ready",
            mesh_id="mesh-ordered",
            join_token="vtmesh_ordered",
            coordinator_endpoint="http://alpha.local:9500",
        ),
    )
    assert ready == {"status": "ok", "command_phase": "drive_ready"}
    assert _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha"},
    )["command"]["command_id"] == drive_id
    first = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "gamma"},
    )["command"]
    _assert_command_id(first)
    assert first["action"] == "join"
    assert _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"] is None

    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(first, worker_id="gamma", event="serving"),
    )
    second = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"]
    _assert_command_id(second)
    assert second["action"] == "join"


def test_pool_heartbeat_status_cannot_advance_serial_join_or_driver_completion(pool):
    endpoint, token = pool
    for worker_id, vram in (
        ("alpha", 24),
        ("beta", 49),
        ("gamma", 80),
    ):
        _join(endpoint, token, worker_id, vram=vram)

    launched = _call(
        endpoint,
        token,
        "/v1/pool/launch",
        {
            "model_id": "m1",
            "workers": ["gamma", "alpha", "beta"],
            "driver": "alpha",
        },
    )
    mesh_key = launched["mesh_key"]
    drive = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha"},
    )["command"]
    _assert_command_id(drive)
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            drive,
            worker_id="alpha",
            event="drive_ready",
            mesh_id="mesh-heartbeat-recovery",
            join_token="vtmesh_heartbeat_recovery",
            coordinator_endpoint="http://alpha.local:9500",
        ),
    )
    first = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "gamma"},
    )["command"]
    first_id = _assert_command_id(first)
    assert first["action"] == "join"

    # Heartbeat status is observational. It must not complete the join command
    # or admit the next worker without the signed, command-bound report.
    _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "gamma", "status": "serving"},
    )
    second = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"]
    assert second is None
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "joining"

    completed_first = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(first, worker_id="gamma", event="serving"),
    )
    assert completed_first["command_completed"] == first_id
    second = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"]
    second_id = _assert_command_id(second)
    assert second["action"] == "join"

    _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta", "status": "serving"},
    )
    assert _call(endpoint, token, "/v1/pool/status", {})["meshes"][mesh_key][
        "status"
    ] == "joining"
    # Beta's heartbeat still cannot complete its join command.
    assert _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"]["command_id"] == second_id
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(second, worker_id="beta", event="serving"),
    )

    # The driver's serving heartbeat likewise cannot substitute for the final
    # terminal report that completes its still-inflight drive command.
    _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha", "status": "serving"},
    )
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "joining"
    assert status["meshes"][mesh_key]["driver_verified"] is False
    completed_drive = _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(drive, worker_id="alpha", event="serving"),
    )
    assert completed_drive["command_completed"] == drive["command_id"]
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "serving"
    assert "pending_join_members" not in status["meshes"][mesh_key]


def test_pool_tears_down_serving_mesh_when_child_process_exits(pool):
    endpoint, token = pool
    _join(endpoint, token, "alpha", vram=24)
    _join(endpoint, token, "beta", vram=49)
    mesh_key = _call(
        endpoint,
        token,
        "/v1/pool/launch",
        {
            "model_id": "m1",
            "workers": ["alpha", "beta"],
            "driver": "alpha",
        },
    )["mesh_key"]
    drive = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha"},
    )["command"]
    _assert_command_id(drive)
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            drive,
            worker_id="alpha",
            event="drive_ready",
            mesh_id="mesh-child-health",
            join_token="vtmesh_child_health",
            coordinator_endpoint="http://alpha.local:9500",
        ),
    )
    join = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "beta"},
    )["command"]
    _assert_command_id(join)
    assert join["action"] == "join"
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(join, worker_id="beta", event="serving"),
    )
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(drive, worker_id="alpha", event="serving"),
    )
    assert _call(endpoint, token, "/v1/pool/status", {})["meshes"][mesh_key][
        "routing_ready"
    ] is True

    failed = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {
            "worker_id": "beta",
            "status": "serving",
            "runtime_error": "pid 7331 exited with code 1",
        },
    )
    assert failed["command"]["action"] == "stop"
    beta_stop = failed["command"]
    _assert_command_id(beta_stop)
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "error"
    assert status["meshes"][mesh_key]["routing_ready"] is False
    assert "worker beta runtime failed" in status["meshes"][mesh_key]["error"]
    assert status["workers"]["alpha"]["status"] == "stopping"
    assert status["workers"]["beta"]["status"] == "stopping"
    alpha_stop = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "alpha"},
    )["command"]
    _assert_command_id(alpha_stop)
    assert alpha_stop["action"] == "stop"

    for worker_id, stop_command in (
        ("alpha", alpha_stop),
        ("beta", beta_stop),
    ):
        _call(
            endpoint,
            token,
            "/v1/pool/report",
            _command_report(
                stop_command,
                worker_id=worker_id,
                event="stopped",
            ),
        )
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["workers"]["alpha"]["status"] == "idle"
    assert status["workers"]["beta"]["status"] == "idle"
    assert status["meshes"][mesh_key]["status"] == "error"


def test_recommend_prefers_single_worker_and_best_fit(pool):
    endpoint, token = pool
    _join(endpoint, token, "small", vram=16, model_bytes=40_000_000_000)
    _join(endpoint, token, "mid", vram=24, model_bytes=40_000_000_000)
    _join(endpoint, token, "big", vram=48, model_bytes=40_000_000_000)

    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "m1"})["suggestions"]
    # 40 GB * 1.3 = 52 GB: no single card fits; pairs required. Best fit:
    # the SMALLEST pair that clears the bar (small+big = 64 GB) outranks
    # mid+big (72 GB); within the winning pair the bigger box drives.
    assert all(len(s["workers"]) == 2 for s in got)
    top = got[0]
    assert set(top["workers"]) == {"small", "big"} and top["driver"] == "big"

    # A model that fits one card ranks the single-worker mesh first.
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "solo",
            "capability": {"vram_gb": 48},
            "catalog": [{"model_id": "tiny", "model_bytes": 5_000_000_000}],
            "endpoints": {"proof": "http://solo.local:9402"},
        },
    )
    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "tiny"})["suggestions"]
    top = got[0]
    assert (top["workers"], top["driver"], top["max_rtt_ms"]) == (["solo"], "solo", 0.0)
    assert top["link_class"] == "local"  # singles never pay a member link


def test_recommend_best_fit_keeps_big_boxes_free(pool):
    """A small model must land on the SMALLEST fitting box, not the biggest.

    Biggest-first starred the pool's only 320 GB box (the one box able to
    hold the largest catalogue models) for a 21 GB model while a lone
    consumer card sat idle . Fit quality also outranks
    fetch convenience: a worker that must download the model but fits
    snugly beats an oversized worker that already holds it.
    """

    endpoint, token = pool
    # All three fit m2 (15 GB * 1.3 = 19.5 GB need) and already hold it.
    for wid, vram in (("gpu4090", 23), ("gpu5090x2", 62), ("a100x4", 320)):
        _call(
            endpoint, token, "/v1/pool/join",
            {
                "worker_id": wid,
                "capability": {"gpu_name": "test", "vram_gb": vram},
                "catalog": [{"model_id": "m2", "model_bytes": 15_000_000_000}],
                "endpoints": {
                    "rpc": f"{wid}.local:50052",
                    "proof": f"http://{wid}.local:9402",
                    "mesh": f"http://{wid}.local:9500",
                },
            },
        )
    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "m2"})["suggestions"]
    singles = [s for s in got if len(s["workers"]) == 1]
    assert [s["workers"][0] for s in singles[:3]] == [
        "gpu4090", "gpu5090x2", "a100x4",
    ], "smallest fitting worker must rank first"

    # Drop m2 from the smaller catalogs (rejoin without it): only the big box
    # holds it now, but the small boxes can fetch it — the snug fetch-needing
    # box must STILL outrank the oversized no-fetch box.
    _call(
        endpoint, token, "/v1/pool/register-model",
        {
            "model_id": "m2",
            "hf_repo": "test/m2",
            "hf_files": ["m2.gguf"],
            "model_bytes": 15_000_000_000,
            "layers": 1,
        },
    )
    for wid, vram in (("gpu4090", 23), ("gpu5090x2", 62)):
        _call(
            endpoint, token, "/v1/pool/join",
            {
                "worker_id": wid,
                "capability": {
                    "gpu_name": "test",
                    "vram_gb": vram,
                    "free_disk_gb": 500,
                },
                "catalog": [],
                "endpoints": {
                    "rpc": f"{wid}.local:50052",
                    "proof": f"http://{wid}.local:9402",
                    "mesh": f"http://{wid}.local:9500",
                },
            },
        )
    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "m2"})["suggestions"]
    singles = [s for s in got if len(s["workers"]) == 1]
    assert [s["workers"][0] for s in singles[:3]] == [
        "gpu4090", "gpu5090x2", "a100x4",
    ], "best fit outranks fetch convenience"
    assert singles[0].get("fetch") is True
    assert singles[2].get("fetch") is None


def test_chat_routes_through_driver_longpoll(pool, monkeypatch):
    """Operator chat is queued for the driver, delivered on its chat long-poll,
    and answered via /v1/pool/chat-result — no inbound connection to the mesh."""
    endpoint, token = pool
    _join(endpoint, token, "solo")
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m1", "workers": ["solo"], "driver": "solo"},
    )
    mesh_key = launched["mesh_key"]
    # Drive the single-worker mesh to serving.
    drive = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "solo"},
    )["command"]
    _call(
        endpoint, token, "/v1/pool/report",
        _command_report(
            drive,
            worker_id="solo",
            event="drive_ready",
            mesh_id="mesh-x",
            join_token="vtmesh_tok",
            coordinator_endpoint="http://127.0.0.1:9500",
        ),
    )
    assert _call(endpoint, token, "/v1/pool/status", {})["meshes"][mesh_key]["status"] == "serving"

    # A live driver's chat long-poll runs continuously from launch; do one poll
    # now so the manager records this driver as actively picking chats up (the
    # fail-fast guard refuses a mesh whose driver has gone silent).
    _call(endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 0})

    # handle_chat blocks until the driver reports back — run it in a thread and
    # play the driver by hand.
    result: dict = {}
    chatter = threading.Thread(
        target=lambda: result.update(
            _call(endpoint, token, "/v1/pool/chat",
                  {"mesh_key": mesh_key, "prompt": "ping", "timeout": 5})
        ),
        daemon=True,
    )
    chatter.start()

    # The queued request is delivered on the driver's chat long-poll, which
    # returns immediately once a chat is signalled.
    polled = _call(endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 5})
    assert len(polled["chat"]) == 1
    chat_id = polled["chat"][0]["chat_id"]
    delivery_token = polled["chat"][0]["delivery_token"]
    assert polled["chat"][0]["delivery_attempt"] == 1
    assert polled["chat"][0]["messages"][0]["content"] == "ping"
    assert _call(
        endpoint,
        token,
        "/v1/pool/chat-pickup",
        {
            "worker_id": "solo",
            "chat_id": chat_id,
            "delivery_token": delivery_token,
        },
    ) == {"status": "ok"}

    # While that chat is in flight, the next test beyond the slot
    # capacity is refused (forced to 1 here): admissions track the
    # backend's --parallel slots instead of piling up past them.
    monkeypatch.setattr(pool_module, "_serve_parallel_slots", lambda: 1)
    with pytest.raises(RuntimeError, match="already running"):
        _call(endpoint, token, "/v1/pool/chat",
              {"mesh_key": mesh_key, "prompt": "again", "timeout": 1})

    # Driver pushes the completion back; handle_chat unblocks with it.
    _call(
        endpoint, token, "/v1/pool/chat-result",
        {"worker_id": "solo", "chat_id": chat_id,
         "content": "pong", "verified": True, "receipts": 1},
    )
    chatter.join(timeout=5)
    assert result["status"] == "ok"
    assert result["content"] == "pong"
    assert result["verified"] is True

    # Internal chat queue never leaks into the operator-facing status.
    assert "chat_pending" not in _call(endpoint, token, "/v1/pool/status", {})["workers"]["solo"]


def test_launch_only_driver_needs_model(pool):
    """Only the DRIVER must have the model: its llama-server reads the GGUF and
    streams layer slices to members over RPC, and members join file-lessly
    (manifest fetched from the coordinator, proof blobs on demand). A driver
    without the model and without a known download source is refused with a
    clear message; a MEMBER without it is fine."""
    endpoint, token = pool
    _join(endpoint, token, "hasit")  # catalog has "m1"
    # Member lacking the model is allowed (file-less join).
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "bare",
            "capability": {"gpu_name": "test", "vram_gb": 48},
            "catalog": [],  # nothing on disk
            "endpoints": {"rpc": "bare.local:50052", "proof": "http://bare.local:9402",
                          "mesh": "http://bare.local:9500"},
        },
    )
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m1", "workers": ["hasit", "bare"], "driver": "hasit"},
    )
    assert launched["status"] == "launching"
    mesh_key = launched["mesh_key"]
    _call(endpoint, token, "/v1/pool/stop", {"mesh_key": mesh_key})
    for worker_id in ("hasit", "bare"):
        stop_command = _call(
            endpoint,
            token,
            "/v1/pool/heartbeat",
            {"worker_id": worker_id},
        )["command"]
        assert stop_command["action"] == "stop"
        _call(
            endpoint,
            token,
            "/v1/pool/report",
            _command_report(
                stop_command,
                worker_id=worker_id,
                event="stopped",
            ),
        )
    assert mesh_key not in _call(endpoint, token, "/v1/pool/status", {})[
        "meshes"
    ]

    # Driver lacking the model with NO registry source: clean refusal.
    with pytest.raises(RuntimeError, match="does not have model 'm1'"):
        _call(
            endpoint, token, "/v1/pool/launch",
            {"model_id": "m1", "workers": ["bare", "hasit"], "driver": "bare"},
        )


def test_launch_auto_fetches_shipped_model_no_worker_ever_advertised(pool):
    """A model from the SHIPPED catalogue is launchable on a pool that has
    never seen it advertised: the source comes from verallm/registry/models.py.

    Before this, a fresh box could not serve a model we ship until someone
    hand-fed a worker catalog, even though hf_repo/hf_files are compiled in.
    """
    from verallm.registry.models import MESH_GGUF_MODELS

    endpoint, token = pool
    shipped_id = next(iter(MESH_GGUF_MODELS))
    entry, variant = MESH_GGUF_MODELS[shipped_id]
    _join(endpoint, token, "fresh")  # holds only m1; nothing taught the pool

    # No explicit workers: placement must find the worker on its own, which
    # is the real flow and needs the shipped facts too (an earlier version of
    # this fix only reached the fetch path and still failed with "no feasible
    # worker set for this model").
    launched = _call(
        endpoint,
        token,
        "/v1/pool/launch",
        {"model_id": shipped_id},
    )
    assert (
        _call(endpoint, token, "/v1/pool/status", {})["meshes"][
            launched["mesh_key"]
        ]["status"]
        == "fetching"
    )
    fetch_command = _call(
        endpoint, token, "/v1/pool/heartbeat", {"worker_id": "fresh"}
    )["command"]
    _assert_command_id(fetch_command)
    assert fetch_command["action"] == "fetch"
    assert fetch_command["spec"]["hf_repo"] == (variant.hf_repo or entry.hf_repo)
    assert fetch_command["spec"]["hf_files"] == list(variant.hf_files)


def test_launch_manifest_anchors_reach_fetch_spec(pool):
    """Deploy passes the chain manifest root + store URLs with the launch;
    they must land in the fetch spec so a fresh driver downloads the
    owner-published manifest instead of rebuilding it locally (building
    manifests is the subnet owner's process, never a miner's)."""
    endpoint, token = pool
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "teacher2",
            "capability": {"gpu_name": "test", "vram_gb": 48},
            "catalog": [{"model_id": "m10", "model_bytes": 1000, "layers": 12,
                         "hf_repo": "org/repo", "hf_files": ["m10.gguf"]}],
            "endpoints": {"rpc": "t2.local:50052", "proof": "http://t2.local:9402",
                          "mesh": "http://t2.local:9500"},
        },
    )
    _join(endpoint, token, "learner2")  # has only m1
    root = "ab" * 32
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {
            "model_id": "m10",
            "workers": ["learner2"],
            "driver": "learner2",
            "model_tensor_manifest_root": root,
            "manifest_urls": ["https://store.example/gleipnir/testnet/"],
        },
    )
    assert launched["mesh_key"]
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "learner2"})
    fetch_command = beat["command"]
    assert fetch_command["action"] == "fetch"
    assert fetch_command["spec"]["model_tensor_manifest_root"] == root
    assert fetch_command["spec"]["manifest_urls"] == [
        "https://store.example/gleipnir/testnet"
    ]


def test_launch_auto_fetches_model_for_driver(pool):
    """A driver lacking the model auto-fetches when the pool knows a source:
    launch queues a fetch command (mesh 'fetching'), the driver's 'fetched'
    report banks the new catalog entry and dispatches the drive."""
    endpoint, token = pool
    # A worker advertising hf metadata teaches the pool's model registry.
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "teacher",
            "capability": {"gpu_name": "test", "vram_gb": 48},
            "catalog": [{"model_id": "m9", "model_bytes": 1000, "layers": 12,
                         "hf_repo": "org/repo", "hf_files": ["m9.gguf"]}],
            "endpoints": {"rpc": "t.local:50052", "proof": "http://t.local:9402",
                          "mesh": "http://t.local:9500"},
        },
    )
    _join(endpoint, token, "learner")  # has only m1
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m9", "workers": ["learner"], "driver": "learner"},
    )
    mesh_key = launched["mesh_key"]
    assert _call(endpoint, token, "/v1/pool/status", {})["meshes"][mesh_key]["status"] == "fetching"

    # The learner's next heartbeat carries the fetch command with the source.
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "learner"})
    fetch_command = beat["command"]
    fetch_id = _assert_command_id(fetch_command)
    assert fetch_command["action"] == "fetch"
    assert fetch_command["spec"]["hf_repo"] == "org/repo"

    # While fetching, the reservation survives the worker's self-reported idle
    # (the machine must not be double-booked into a second mesh).
    _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "learner", "status": "idle", "command_ack": fetch_id},
    )
    with pytest.raises(RuntimeError, match="busy"):
        _call(endpoint, token, "/v1/pool/launch",
              {"model_id": "m1", "workers": ["learner"], "driver": "learner"})

    # fetched -> catalog banked + drive dispatched.
    _call(
        endpoint, token, "/v1/pool/report",
        _command_report(
            fetch_command,
            worker_id="learner",
            event="fetched",
            entry={
                "model_id": "m9",
                "llama_model": "/x/m9.gguf",
                "manifest": "/x/m.json",
                "layers": 12,
                "model_bytes": 1000,
            },
        ),
    )
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "driving"
    models = {c["model_id"] for c in status["workers"]["learner"]["catalog"]}
    assert "m9" in models
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "learner"})
    _assert_command_id(beat["command"])
    assert beat["command"]["action"] == "drive"
    assert beat["command"]["model_id"] == "m9"


def _direct_manager(tmp_path):
    """A PoolManager with no HTTP server, for testing the chat-slot guard
    directly (no reliance on real-time lock ageing)."""
    from verallm.mesh.pool import PoolManager

    state_dir, _tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    return PoolManager(state_dir)


class _AcceptingKeypair:
    def verify(self, message, signature):
        return bool(message) and signature == b"signed"


def test_operator_wallet_owner_session_and_logout(tmp_path, monkeypatch):
    """The pool token bootstraps one owner account; later wallet sessions can
    manage without retaining that raw token, while other accounts stay viewers."""
    mgr = _direct_manager(tmp_path)
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    owner = "5" + "A" * 47
    viewer = "5" + "B" * 47

    challenge = mgr.handle_auth_challenge({"account": owner})
    session = mgr.handle_auth_verify(
        {
            "account": owner,
            "nonce": challenge["nonce"],
            "signature": b"signed".hex(),
            "management_secret": mgr.state["management_secret"],
        }
    )
    assert session["is_owner"] is True
    assert mgr.state["owner_account"] == owner
    assert mgr.handle_auth_session({"session": session["token"]}) == {
        "status": "ok", "account": owner, "is_owner": True, "signed_in": True,
    }
    mgr._auth_manage({"session": session["token"]})

    # A challenge is single-use even when the same signed payload is replayed.
    with pytest.raises(PermissionError, match="expired or unknown"):
        mgr.handle_auth_verify(
            {
                "account": owner,
                "nonce": challenge["nonce"],
                "signature": b"signed".hex(),
                "management_secret": mgr.state["management_secret"],
            }
        )

    other_challenge = mgr.handle_auth_challenge({"account": viewer})
    other_session = mgr.handle_auth_verify(
        {
            "account": viewer,
            "nonce": other_challenge["nonce"],
            "signature": b"signed".hex(),
        }
    )
    assert other_session["is_owner"] is False
    with pytest.raises(PermissionError, match="verified owner-wallet"):
        mgr.handle_operator_access({"session": other_session["token"]})

    access = mgr.handle_operator_access({"session": session["token"]})
    decoded = MeshPoolToken.decode(access["pool_token"])
    assert decoded.pool_id == mgr.state["pool_id"]
    assert decoded.pool_secret == mgr.state["pool_secret"]

    mgr.handle_auth_logout({"session": session["token"]})
    assert mgr.handle_auth_session({"session": session["token"]})["signed_in"] is False
    with pytest.raises(PermissionError):
        mgr._auth_manage({"session": session["token"]})


def test_operator_wallet_http_routes(pool, monkeypatch):
    endpoint, pool_token = pool
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    owner = "5" + "C" * 47
    challenge = post_json(
        endpoint + "/v1/auth/challenge", {"account": owner}, timeout=5.0
    )
    session = post_json(
        endpoint + "/v1/auth/verify",
        {
            "account": owner, "nonce": challenge["nonce"],
            "signature": b"signed".hex(),
            "management_secret": pool_token.management.pool_secret,
        },
        timeout=5.0,
    )
    access = post_json(
        endpoint + "/v1/operator/access", {"session": session["token"]}, timeout=5.0
    )
    assert MeshPoolToken.decode(access["pool_token"]).pool_secret == pool_token.pool_secret
    post_json(
        endpoint + "/v1/auth/logout", {"session": session["token"]}, timeout=5.0
    )
    who = post_json(
        endpoint + "/v1/auth/session", {"session": session["token"]}, timeout=5.0
    )
    assert who["signed_in"] is False


def test_public_operator_overview_is_sanitized_and_recommendations_work(pool):
    endpoint, token = pool
    _call(
        endpoint,
        token,
        "/v1/pool/join",
        {
            "worker_id": "private-box",
            "capability": {"gpu_name": "test", "vram_gb": 48, "internal": "no"},
            "catalog": [
                {
                    "model_id": "m-private", "model_bytes": 1000, "layers": 8,
                    "hf_repo": "private/repo", "hf_files": ["secret.gguf"],
                    "llama_model": "/srv/private/model.gguf",
                }
            ],
            "endpoints": {
                "rpc": "private-box.local:50052",
                "proof": "http://private-box.local:9402",
                "mesh": "http://private-box.local:9500",
            },
        },
    )
    public_recommend = post_json(
        endpoint + "/v1/pool/recommend", {"model_id": "m-private"}, timeout=5.0
    )
    assert public_recommend["suggestions"]
    launched = _call(
        endpoint,
        token,
        "/v1/pool/launch",
        {"model_id": "m-private", "workers": ["private-box"], "driver": "private-box"},
    )
    drive = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "private-box"},
    )["command"]
    _call(
        endpoint,
        token,
        "/v1/pool/report",
        _command_report(
            drive,
            worker_id="private-box",
            event="drive_ready",
            mesh_id="private-mesh",
            join_token="vtmesh_private",
            coordinator_endpoint="http://private:9500",
        ),
    )
    _call(
        endpoint, token, "/v1/pool/heartbeat",
        {"worker_id": "private-box", "status": "backend error at /srv/private"},
    )

    overview = post_json(endpoint + "/v1/operator/overview", {}, timeout=5.0)
    assert overview["pool_id"] == token.pool_id
    worker = overview["workers"]["private-box"]
    assert worker["status"] == "error"
    assert worker["capability"] == {"gpu_name": "test", "vram_gb": 48}
    assert worker["catalog"] == [
        {"model_id": "m-private", "model_bytes": 1000, "layers": 8}
    ]
    assert set(overview["meshes"][launched["mesh_key"]]) <= {
        "mesh_key", "model_id", "members", "driver", "status", "serving",
        "driver_verified", "driver_stale", "member_stale", "stale_members",
        "routing_ready", "error", "validator_score", "created_at_unix",
    }
    serialized = str(overview)
    for secret in ("vtmesh_private", "private/repo", "secret.gguf", "/srv/private"):
        assert secret not in serialized

    with pytest.raises(RuntimeError, match="HTTP 403"):
        post_json(
            endpoint + "/v1/pool/launch",
            {"model_id": "m-private", "workers": ["private-box"]},
            timeout=5.0,
        )


def test_operator_overview_maps_exact_validator_ema_to_mesh(tmp_path):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    coordinator = "0x" + "a1" * 20
    state_path = tmp_path / "isolated-validator-state.json"
    write_shared_state(
        ValidatorSharedState(
            epoch_number=23,
            miner_scores={coordinator: {"4": 2.75}},
            miner_ema_scores={coordinator: {"4": 2.75}},
            miner_score_metadata={
                coordinator: {
                    "4": _mesh_score_metadata(
                        coordinator=coordinator,
                        model_index=4,
                        model_id="gguf-model",
                        score_epoch=23,
                        ema=2.75,
                        scored_epochs=3,
                        mesh_id="scored-runtime",
                        snapshot_generation=4,
                    )
                }
            },
            probation_miners={coordinator: [4]},
            miner_endpoints=[
                MinerEntry(
                    address=coordinator,
                    endpoint="http://private-coordinator.internal:9999",
                    model_id="gguf-model",
                    model_index=4,
                    quant="gguf_mesh",
                    max_context_len=8192,
                    uid=17,
                    mesh_enabled=True,
                )
            ],
        ),
        str(state_path),
    )
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    mgr = PoolManager(
        state_dir,
        coordinator_address=coordinator.upper(),
        validator_shared_state_path=state_path,
    )
    mgr.state["workers"]["driver"] = {
        "status": "serving",
        "mesh": "m-scored",
        "last_seen_unix": int(time.time()),
        "capability": {"gpu_name": "test", "vram_gb": 24},
        "catalog": [],
    }
    mgr.state["meshes"]["m-scored"] = {
        "mesh_key": "m-scored",
        "model_id": "gguf-model",
        "members": ["driver"],
        "driver": "driver",
        "status": "serving",
    }

    public = mgr.handle_operator_overview({})
    assert public["meshes"]["m-scored"]["validator_score"] == {
        "available": False,
        "source": "validator_ema",
        "reason": "owner access is required to view validator scores",
    }
    overview = mgr.handle_operator_overview(
        {"management_secret": mgr.state["management_secret"]}
    )
    score = overview["meshes"]["m-scored"]["validator_score"]
    assert score["available"] is True
    assert score["score"] == 2.75
    assert score["score_kind"] == "current_coordinator_model_slot_ema"
    assert score["coordinator_address"] == coordinator
    assert score["model_index"] == 4
    assert score["uid"] == 17
    assert score["epoch_number"] == 23
    assert score["score_epoch"] == 23
    assert score["latest_score"] == {
        "ema_at_completion": 2.75,
        "score_epoch": 23,
        "chain_id": 945,
        "netuid": 405,
        "coordinator_address": coordinator,
        "model_index": 4,
        "model_id": "gguf-model",
        "scored_mesh_id": "scored-runtime",
        "scored_verification_snapshot_hash": "11" * 32,
        "scored_snapshot_generation": 4,
    }
    assert score["current_topology"]["mesh_id"] is None
    assert score["current_topology_matches_latest_score"] is False
    assert score["score_adjusted_since_completion"] is False
    assert score["probation"] is True
    assert score["blacklisted"] is False
    assert score["stale"] is False
    assert score["updated_at"] > 0
    assert score["age_seconds"] >= 0
    # The score projection must not disclose the validator path or private
    # coordinator endpoint used to establish the identity mapping.
    serialized = str(overview)
    assert str(state_path) not in serialized
    assert "private-coordinator.internal" not in serialized


def test_operator_mesh_score_never_guesses_missing_or_ambiguous_mapping(tmp_path):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    coordinator = "0x" + "b2" * 20
    state_path = tmp_path / "validator-state.json"
    entries = [
        MinerEntry(
            address=coordinator,
            endpoint=f"http://private-{index}:9999",
            model_id="same-model",
            model_index=index,
            quant="gguf_mesh",
            max_context_len=4096,
            mesh_enabled=True,
        )
        for index in (3, 7)
    ]
    entries.append(
        MinerEntry(
            address=coordinator,
            endpoint="http://private-vllm:9999",
            model_id="not-a-mesh",
            model_index=9,
            quant="int4",
            max_context_len=4096,
            mesh_enabled=False,
        )
    )
    write_shared_state(
        ValidatorSharedState(
            epoch_number=8,
            miner_scores={coordinator: {"3": 1.1, "7": 2.2, "9": 99.0}},
            miner_ema_scores={coordinator: {"3": 1.1, "7": 2.2, "9": 99.0}},
            miner_score_metadata={
                coordinator: {
                    str(model_index): _mesh_score_metadata(
                        coordinator=coordinator,
                        model_index=model_index,
                        model_id=(
                            "same-model" if model_index in (3, 7) else "not-a-mesh"
                        ),
                        score_epoch=8,
                        ema={3: 1.1, 7: 2.2, 9: 99.0}[model_index],
                        scored_epochs=2,
                        mesh_id=f"scored-{model_index}",
                    )
                    for model_index in (3, 7, 9)
                }
            },
            miner_endpoints=entries,
        ),
        str(state_path),
    )
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    mgr = PoolManager(
        state_dir,
        coordinator_address=coordinator,
        validator_shared_state_path=state_path,
    )

    scores = mgr._validator_scores_for_meshes(
        {
            "ambiguous": {"model_id": "same-model"},
            "pinned": {"model_id": "same-model", "model_index": 7},
            "non-mesh": {"model_id": "not-a-mesh"},
            "unknown": {"model_id": "missing"},
        }
    )
    assert scores["ambiguous"]["available"] is False
    assert scores["ambiguous"]["reason"] == "coordinator model mapping is ambiguous"
    assert scores["pinned"]["available"] is True
    assert scores["pinned"]["model_index"] == 7
    assert scores["pinned"]["score"] == 2.2
    assert scores["non-mesh"]["available"] is False
    assert scores["unknown"]["available"] is False
    assert all(
        "score" not in scores[key]
        for key in ("ambiguous", "non-mesh", "unknown")
    )

    # miner_scores is a proxy-routing value and may contain the 0.01 floor.
    # If raw EMA is absent, the operator view must stay unavailable instead
    # of presenting that routing substitution as a validator score.
    write_shared_state(
        ValidatorSharedState(
            epoch_number=9,
            miner_scores={coordinator: {"3": 0.01}},
            miner_endpoints=[entries[0]],
        ),
        str(state_path),
    )
    unavailable = mgr._validator_scores_for_meshes(
        {"old-state": {"model_id": "same-model", "model_index": 3}}
    )["old-state"]
    assert unavailable["available"] is False
    assert unavailable["reason"] == "validator EMA is unavailable in this shared state"
    assert "score" not in unavailable


def test_operator_pinned_score_requires_active_exact_mesh_slot(tmp_path):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    coordinator = "0x" + "c3" * 20
    state_path = tmp_path / "validator-state.json"
    entries = [
        MinerEntry(
            address=coordinator,
            endpoint="http://exact.internal:9999",
            model_id="exact-model",
            model_index=1,
            quant="gguf_mesh",
            max_context_len=4096,
            mesh_enabled=True,
        ),
        MinerEntry(
            address=coordinator,
            endpoint="http://other.internal:9999",
            model_id="other-model",
            model_index=2,
            quant="gguf_mesh",
            max_context_len=4096,
            mesh_enabled=True,
        ),
        MinerEntry(
            address=coordinator,
            endpoint="http://non-mesh.internal:9999",
            model_id="disabled-model",
            model_index=3,
            quant="int4",
            max_context_len=4096,
            mesh_enabled=False,
        ),
    ]
    write_shared_state(
        ValidatorSharedState(
            epoch_number=31,
            miner_ema_scores={
                coordinator: {
                    "1": 1.25,
                    "2": 2.5,
                    "3": 3.75,
                    # Historical EMA remains in shared state after the slot
                    # is inactive, but there is deliberately no active entry.
                    "4": 4.0,
                }
            },
            miner_score_metadata={
                coordinator: {
                    str(model_index): _mesh_score_metadata(
                        coordinator=coordinator,
                        model_index=model_index,
                        model_id={
                            1: "exact-model",
                            2: "other-model",
                            3: "disabled-model",
                            4: "historical-model",
                        }[model_index],
                        score_epoch=31,
                        ema={1: 1.25, 2: 2.5, 3: 3.75, 4: 4.0}[model_index],
                        scored_epochs=4,
                        mesh_id=f"scored-{model_index}",
                    )
                    for model_index in (1, 2, 3, 4)
                }
            },
            miner_endpoints=entries,
        ),
        str(state_path),
    )
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    manager = PoolManager(
        state_dir,
        coordinator_address=coordinator,
        validator_shared_state_path=state_path,
    )

    scores = manager._validator_scores_for_meshes(
        {
            "exact": {"model_id": "exact-model", "model_index": 1},
            "wrong-model": {"model_id": "exact-model", "model_index": 2},
            "non-mesh": {"model_id": "disabled-model", "model_index": 3},
            "historical": {"model_id": "historical-model", "model_index": 4},
        }
    )

    assert scores["exact"]["available"] is True
    assert scores["exact"]["score"] == 1.25
    assert scores["wrong-model"]["available"] is False
    assert (
        scores["wrong-model"]["reason"]
        == "pinned model index does not match this mesh"
    )
    assert scores["non-mesh"]["available"] is False
    assert (
        scores["non-mesh"]["reason"]
        == "pinned model index does not match this mesh"
    )
    assert scores["historical"]["available"] is False
    assert (
        scores["historical"]["reason"]
        == "pinned mesh slot is not active in validator state"
    )
    assert all(
        "score" not in scores[key]
        for key in ("wrong-model", "non-mesh", "historical")
    )


def test_operator_zero_score_requires_bound_completed_sample_provenance(tmp_path):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    coordinator = "0x" + "d4" * 20
    state_path = tmp_path / "validator-state.json"
    endpoint = MinerEntry(
        address=coordinator,
        endpoint="http://mesh.internal:9999",
        model_id="zero-model",
        model_index=5,
        quant="gguf_mesh",
        max_context_len=4096,
        mesh_enabled=True,
    )
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    manager = PoolManager(
        state_dir,
        coordinator_address=coordinator,
        validator_shared_state_path=state_path,
    )
    mesh = {"mesh": {"model_id": "zero-model", "model_index": 5}}

    def score_with(metadata):
        write_shared_state(
            ValidatorSharedState(
                epoch_number=41,
                miner_ema_scores={coordinator: {"5": 0.0}},
                miner_score_metadata={coordinator: {"5": metadata}},
                miner_endpoints=[endpoint],
            ),
            str(state_path),
        )
        return manager._validator_scores_for_meshes(mesh)["mesh"]

    never_scored = score_with(
        {"scored_epochs": 0, "last_scored_epoch": None}
    )
    assert never_scored["available"] is False
    assert (
        never_scored["reason"]
        == "validator has not scored this coordinator model yet"
    )
    assert "score" not in never_scored

    previous_epoch = score_with(
        _mesh_score_metadata(
            coordinator=coordinator,
            model_index=5,
            model_id="zero-model",
            score_epoch=40,
            ema=0.0,
        )
    )
    assert previous_epoch["available"] is True
    assert previous_epoch["score"] == 0.0
    assert previous_epoch["score_epoch"] == 40
    assert previous_epoch["epochs_since_score"] == 1
    assert previous_epoch["current_topology_matches_latest_score"] is False

    missing_sample = score_with(
        {"scored_epochs": 1, "last_scored_epoch": 41}
    )
    assert missing_sample["available"] is False
    assert missing_sample["reason"] == "validator has not scored this coordinator model yet"

    real_zero = score_with(
        _mesh_score_metadata(
            coordinator=coordinator,
            model_index=5,
            model_id="zero-model",
            score_epoch=41,
            ema=0.0,
        )
    )
    assert real_zero["available"] is True
    assert real_zero["score"] == 0.0
    assert real_zero["scored_epochs"] == 1
    assert real_zero["last_scored_epoch"] == 41
    assert real_zero["epoch_number"] == 41


def test_operator_mesh_score_requires_explicit_coordinator_identity(tmp_path):
    state_dir, _ = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
        validator_shared_state_path=tmp_path / "state.json",
    )
    mgr = PoolManager(state_dir)
    score = mgr._validator_scores_for_meshes({"mesh": {"model_id": "m"}})["mesh"]
    assert score == {
        "available": False,
        "source": "validator_ema",
        "reason": "coordinator identity is not configured",
    }
    with pytest.raises(ValueError, match="20-byte"):
        PoolManager(state_dir, coordinator_address="not-an-evm-address")


def test_validator_pool_shared_state_path_is_optional(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    common = {
        "manager_endpoint": "http://127.0.0.1:19500",
        "serving_mode": "validator",
        "owner_account": "5" + "D" * 47,
        "coordinator_address": "0x" + "e5" * 20,
        "chain_id": 945,
        "netuid": 405,
        "coordinator_uid": 1,
        "epoch": 52,
    }

    # validator_shared_state_path stays optional for subnet pools: it only
    # feeds the operator-board score view, which falls back to the chain
    # score cache when unset (public operators have no validator install).
    state_dir, _ = create_pool_state(tmp_path / "missing", **common)
    persisted = json.loads(
        (state_dir / pool_module.POOL_STATE_FILE).read_text(encoding="utf-8")
    )
    assert "validator_shared_state_path" not in persisted
    mgr = PoolManager(state_dir)
    assert mgr.validator_shared_state_path == ""

    # The chain binding is NOT optional: subnet pool state stripped of
    # validator_binding must refuse to load.
    state_dir, _ = create_pool_state(
        tmp_path / "persisted",
        validator_shared_state_path=tmp_path / "validator-state.json",
        **common,
    )
    state_path = state_dir / pool_module.POOL_STATE_FILE
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["validator_shared_state_path"] == str(
        tmp_path / "validator-state.json"
    )
    persisted.pop("validator_binding")
    state_path.write_text(json.dumps(persisted), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="subnet pool state has no validator_binding",
    ):
        PoolManager(state_dir)


def test_validator_pool_carries_slot_ema_without_relabeling_new_topology(
    tmp_path,
    monkeypatch,
):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    coordinator = "0x" + "f6" * 20
    shared_path = tmp_path / "validator-state.json"
    write_shared_state(
        ValidatorSharedState(
            chain_id=945,
            netuid=405,
            epoch_number=64,
            # A post-close proof penalty has already halved the live slot EMA;
            # the completed sample remains 6.25 with immutable provenance.
            miner_ema_scores={coordinator: {"6": 3.125}},
            miner_score_metadata={
                coordinator: {
                    "6": _mesh_score_metadata(
                        coordinator=coordinator,
                        model_index=6,
                        model_id="epoch-model",
                        score_epoch=63,
                        ema=6.25,
                        scored_epochs=5,
                        mesh_id="mesh-epoch-63",
                        snapshot_hash="63" * 32,
                        snapshot_generation=3,
                    )
                }
            },
            miner_endpoints=[
                MinerEntry(
                    address=coordinator,
                    endpoint="http://private-coordinator.internal:9999",
                    model_id="epoch-model",
                    model_index=6,
                    quant="gguf_mesh",
                    max_context_len=8192,
                    uid=21,
                    mesh_enabled=True,
                )
            ],
        ),
        str(shared_path),
    )
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="validator",
        owner_account="5" + "E" * 47,
        coordinator_address=coordinator,
        validator_shared_state_path=shared_path,
        chain_id=945,
        netuid=405,
        coordinator_uid=21,
        epoch=64,
    )
    manager = PoolManager(state_dir)

    scores = manager._validator_scores_for_meshes(
        {
            "scored-topology": {
                "model_id": "epoch-model",
                "model_index": 6,
                "validator_binding": {"epoch": 63},
                "mesh_id": "mesh-epoch-63",
                "verification_snapshot_hash": "63" * 32,
                "snapshot_generation": 3,
            },
            "current-topology": {
                "model_id": "epoch-model",
                "model_index": 6,
                "validator_binding": {"epoch": 64},
                "mesh_id": "mesh-epoch-64",
                "verification_snapshot_hash": "64" * 32,
                "snapshot_generation": 4,
            },
        }
    )

    scored = scores["scored-topology"]
    assert scored["available"] is True
    assert scored["current_topology_matches_latest_score"] is True
    assert scored["current_topology"]["epoch"] == 63

    current = scores["current-topology"]
    assert current["available"] is True
    assert current["score"] == 3.125
    assert current["latest_score"]["ema_at_completion"] == 6.25
    assert current["score_adjusted_since_completion"] is True
    assert current["scored_epochs"] == 5
    assert current["score_epoch"] == 63
    assert current["epoch_number"] == 64
    assert current["epochs_since_score"] == 1
    assert current["current_topology_matches_latest_score"] is False
    assert current["latest_score"]["scored_mesh_id"] == "mesh-epoch-63"
    assert current["current_topology"]["mesh_id"] == "mesh-epoch-64"
    assert "mesh_id" not in current
    assert "verification_snapshot_hash" not in current


def test_validator_pool_mesh_score_requires_matching_chain_and_netuid(
    tmp_path,
    monkeypatch,
):
    from neurons.shared_state import (
        MinerEntry,
        ValidatorSharedState,
        write_shared_state,
    )

    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    coordinator = "0x" + "a7" * 20
    shared_path = tmp_path / "validator-state.json"
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="validator",
        owner_account="5" + "G" * 47,
        coordinator_address=coordinator,
        validator_shared_state_path=shared_path,
        chain_id=945,
        netuid=405,
        coordinator_uid=22,
        epoch=64,
    )
    manager = PoolManager(state_dir)
    endpoint = MinerEntry(
        address=coordinator,
        endpoint="http://private-coordinator.internal:9999",
        model_id="bound-model",
        model_index=7,
        quant="gguf_mesh",
        max_context_len=8192,
        uid=22,
        mesh_enabled=True,
    )
    mesh = {
        "mesh": {
            "model_id": "bound-model",
            "model_index": 7,
            "validator_binding": {"epoch": 64},
        }
    }

    def score_for(*, chain_id, netuid):
        write_shared_state(
            ValidatorSharedState(
                chain_id=chain_id,
                netuid=netuid,
                epoch_number=64,
                miner_ema_scores={coordinator: {"7": 1.5}},
                miner_score_metadata={
                    coordinator: {
                        "7": _mesh_score_metadata(
                            coordinator=coordinator,
                            model_index=7,
                            model_id="bound-model",
                            score_epoch=64,
                            ema=1.5,
                        )
                    }
                },
                miner_endpoints=[endpoint],
            ),
            str(shared_path),
        )
        return manager._validator_scores_for_meshes(mesh)["mesh"]

    missing = score_for(chain_id=None, netuid=None)
    assert missing["available"] is False
    assert missing["reason"] == "validator score state has no chain/network binding"

    wrong = score_for(chain_id=945, netuid=96)
    assert wrong["available"] is False
    assert (
        wrong["reason"]
        == "validator score state belongs to a different chain or subnet"
    )

    exact = score_for(chain_id=945, netuid=405)
    assert exact["available"] is True
    assert exact["score"] == 1.5


def test_validator_pool_allows_one_active_mesh_per_model_index(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: _AcceptingKeypair(),
    )
    shared_path = tmp_path / "validator-state.json"
    state_dir, _ = create_pool_state(
        tmp_path / "pool",
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="validator",
        owner_account="5" + "F" * 47,
        coordinator_address="0x" + "07" * 20,
        validator_shared_state_path=shared_path,
        chain_id=945,
        netuid=405,
        coordinator_uid=22,
        epoch=71,
    )
    manager = PoolManager(state_dir)
    auth = {"management_secret": manager.state["management_secret"]}
    manager.handle_register_model(
        {
            **auth,
            "model_id": "one-slot-model",
            "hf_repo": "operator/one-slot-model",
            "hf_files": ["model.gguf"],
            "layers": 4,
            "model_bytes": 1024,
            "model_index": 7,
            "model_package_hash": "11" * 32,
            "model_tensor_manifest_root": "22" * 32,
            "tokenizer_hash": "33" * 32,
            "quantization_scheme": "gguf_q4_k_m",
            "max_context_len": 32_768,
        }
    )
    # Models a COMPLETED chain registration; only confirmed bindings
    # launch chain-bound.
    manager.state["model_registry"]["one-slot-model"]["chain_committed"] = True
    now = int(time.time())
    for worker_id in ("driver-a", "driver-b"):
        manager.state["workers"][worker_id] = {
            "status": "idle",
            "mesh": "",
            "last_seen_unix": now,
            "capability": {
                "vram_gb": 24,
                "subnet_driver_ready": True,
            },
            "catalog": [
                {
                    "model_id": "one-slot-model",
                    "layers": 4,
                    "model_bytes": 1024,
                }
            ],
            "commands": [],
        }

    first = manager.handle_launch(
        {
            **auth,
            "model_id": "one-slot-model",
            "workers": ["driver-a"],
            "driver": "driver-a",
        }
    )
    with pytest.raises(
        ValueError,
        match="already has a mesh for model index 7",
    ):
        manager.handle_launch(
            {
                **auth,
                "model_id": "one-slot-model",
                "workers": ["driver-b"],
                "driver": "driver-b",
            }
        )

    stopped = manager.handle_stop({**auth, "mesh_key": first["mesh_key"]})
    assert stopped == {"status": "stopping", "mesh_key": first["mesh_key"]}
    assert manager.state["meshes"][first["mesh_key"]]["status"] == "stopping"
    assert manager.state["workers"]["driver-a"]["status"] == "stopping"
    with pytest.raises(
        ValueError,
        match="already has a mesh for model index 7",
    ):
        manager.handle_launch(
            {
                **auth,
                "model_id": "one-slot-model",
                "workers": ["driver-b"],
                "driver": "driver-b",
            }
        )


def test_validator_score_source_freshness_is_explicit(monkeypatch):
    monkeypatch.setattr(pool_module.time, "time", lambda: 2_000.0)

    fresh = pool_module.PoolManager._validator_state_freshness(
        SimpleNamespace(updated_at=1_900.0)
    )
    stale = pool_module.PoolManager._validator_state_freshness(
        SimpleNamespace(updated_at=1_000.0)
    )
    unknown = pool_module.PoolManager._validator_state_freshness(
        SimpleNamespace(updated_at=0.0)
    )

    assert fresh == {"updated_at": 1900.0, "age_seconds": 100.0, "stale": False}
    assert stale == {"updated_at": 1000.0, "age_seconds": 1000.0, "stale": True}
    assert unknown == {"updated_at": 0.0, "age_seconds": None, "stale": True}


def test_operator_route_serves_unified_dashboard(pool):
    endpoint, _token = pool
    with urllib.request.urlopen(endpoint + "/operator", timeout=5.0) as response:
        html = response.read().decode()
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Frame-Options"] == "DENY"
    assert "Verathos Operator" in html
    assert "scripts/join_pool.sh" in html
    assert "injectedWeb3" in html
    assert "/v1/operator/overview" in html
    assert "/v1/pool/chat-stream" in html
    for forbidden in ("node" + "xo", "node" + "x", "naut" + "tiilus"):
        assert forbidden not in html.lower()


def test_serving_mesh_requires_every_member_heartbeat_for_routing(tmp_path):
    mgr = _direct_manager(tmp_path)
    now = int(time.time())
    mgr.state["workers"] = {
        "driver": {
            "worker_id": "driver",
            "last_seen_unix": now,
            "last_chat_poll_unix": time.time(),
            "status": "serving",
            "mesh": "m-live",
            "commands": [],
        },
        "member": {
            "worker_id": "member",
            "last_seen_unix": now,
            "status": "serving",
            "mesh": "m-live",
            "commands": [],
        },
    }
    mgr.state["meshes"]["m-live"] = {
        "mesh_key": "m-live",
        "model_id": "m1",
        "members": ["driver", "member"],
        "driver": "driver",
        "status": "serving",
    }
    admin = {"management_secret": mgr.state["management_secret"]}

    healthy = mgr.handle_status(admin)["meshes"]["m-live"]
    assert healthy["routing_ready"] is True
    assert healthy["stale_members"] == []
    assert healthy["driver_stale"] is False
    assert healthy["member_stale"] is False

    mgr.state["workers"]["member"]["last_seen_unix"] = (
        now - int(pool_module.WORKER_STALE_S) - 5
    )
    degraded = mgr.handle_status(admin)["meshes"]["m-live"]
    assert degraded["status"] == "serving"
    assert degraded["routing_ready"] is False
    assert degraded["stale_members"] == ["member"]
    assert degraded["driver_stale"] is False
    assert degraded["member_stale"] is True
    with pytest.raises(ValueError, match="offline member stage"):
        mgr.start_chat_stream({**admin, "mesh_key": "m-live", "prompt": "test"})

    mgr.state["workers"].pop("member")
    missing = mgr.handle_status(admin)["meshes"]["m-live"]
    assert missing["routing_ready"] is False
    assert missing["stale_members"] == ["member"]

    driverless = mgr._mesh_routing_health(
        {"status": "serving", "members": [], "driver": ""}
    )
    assert driverless["routing_ready"] is False
    assert driverless["driver_stale"] is True


def test_operator_models_with_unknown_size_are_not_launch_ready(tmp_path):
    mgr = _direct_manager(tmp_path)
    mgr.state["model_registry"] = {
        "unknown": {"model_bytes": 0, "layers": 4},
        "malformed": {"model_bytes": "not-a-number", "layers": 4},
        "known": {
            "model_bytes": 1024,
            "layers": 4,
            "max_context_len": 32_768,
        },
    }
    overview = mgr.handle_operator_overview(
        {"management_secret": mgr.state["management_secret"]}
    )
    assert overview["models"]["unknown"]["launch_ready"] is False
    assert overview["models"]["malformed"]["launch_ready"] is False
    assert overview["models"]["known"]["launch_ready"] is True
    assert overview["models"]["known"]["max_context_len"] == 32_768


def test_pool_chat_sse_disables_reverse_proxy_buffering() -> None:
    source = inspect.getsource(pool_module.serve_pool_manager)
    assert 'self.send_header("X-Accel-Buffering", "no")' in source


def test_streaming_chat_flushes_short_reasoning_before_final(tmp_path, monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[{"delta":{"reasoning_content":"brief thought"}}]}\n\n',
                    (
                        b'data: {"event":"done","response":{"choices":[{"message":'
                        b'{"content":"answer","reasoning_content":"brief thought"}}],'
                        b'"usage":{"completion_tokens":1}},'
                        b'"verathos_mesh":{"verified":true,'
                        b'"proof_receipt_count":1}}\n\n'
                    ),
                ]
            )

    sent = []
    coordinator_requests = []

    monkeypatch.setattr(
        pool_module.urllib.request,
        "urlopen",
        lambda request, timeout: coordinator_requests.append(request.full_url)
        or _Response(),
    )
    monkeypatch.setattr(
        pool_module,
        "post_json",
        lambda endpoint, body, timeout=0, **_kwargs: sent.append(
            (endpoint, dict(body))
        )
        or {"status": "ok"},
    )

    runner_config = _local_runner_config(tmp_path)
    pool_module._run_pool_chat(
        "https://manager.example",
        "worker-secret",
        "driver",
        runner_config,
        {
            "chat_id": "c-test",
            "model": "m1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    chunk_payloads = [
        body for endpoint, body in sent if endpoint.endswith("/v1/pool/chat-chunk")
    ]
    assert chunk_payloads == [
        {
            "pool_secret": "worker-secret",
            "worker_id": "driver",
            "chat_id": "c-test",
            "thinking": "brief thought",
            "seq": 1,
        }
    ]
    final = next(
        body for endpoint, body in sent if endpoint.endswith("/v1/pool/chat-result")
    )
    assert final["content"] == "answer"
    # The coordinator's committed aggregate carries the thinking text as
    # message.reasoning_content; the driver's final must relay it so the
    # private OpenAI API can pass it through.
    assert final["reasoning_content"] == "brief thought"
    assert final["verified"] is True
    assert final["receipts"] == 1
    # Loopback, never the advertise host: the coordinator's internal-HMAC
    # lane on validator routes rejects non-loopback clients.
    assert coordinator_requests == [
        f"http://127.0.0.1:{runner_config.mesh_port}/v1/chat/completions"
    ]


def test_chat_slot_fails_fast_when_driver_not_polling(tmp_path):
    """A driver that has not long-polled (crashed, or running pre-long-poll code
    like the stale 4090 worker) must be refused immediately with a clear reason
    — not queued into the void where it hangs until the SSE deadline and then
    every retry wrongly reports 'already running'."""
    mgr = _direct_manager(tmp_path)
    mgr.state["workers"]["drv"] = {"worker_id": "drv"}  # never polled
    worker = mgr.state["workers"]["drv"]

    with mgr.lock:
        with pytest.raises(ValueError, match="not picking up chats"):
            mgr._guard_chat_slot("m-1", worker)
        # The refusal must not have taken the lock (else the next attempt would
        # wrongly say 'already running').
        assert "m-1" not in mgr.chat_active

    # A driver that polled recently passes the guard and takes the slot.
    worker["last_chat_poll_unix"] = time.time()
    with mgr.lock:
        mgr._guard_chat_slot("m-1", worker)
        mgr._mark_chat_active("m-1", "c-live")
    assert "c-live" in mgr.chat_active["m-1"]

    # A driver that has since gone silent is refused again.
    mgr.state["workers"]["drv2"] = {"worker_id": "drv2", "last_chat_poll_unix": time.time() - 999}
    with mgr.lock, pytest.raises(ValueError, match="not picking up chats"):
        mgr._guard_chat_slot("m-2", mgr.state["workers"]["drv2"])


def test_chat_slot_reclaims_stale_lock(tmp_path, monkeypatch):
    """A live lock refuses a second test; a lock older than CHAT_ACTIVE_MAX_S is
    a chat whose driver died mid-flight — it is reclaimed (with its orphaned
    waiter/stream) so the mesh is usable again instead of pinned until 300s."""
    mgr = _direct_manager(tmp_path)
    worker = {"worker_id": "drv", "last_chat_poll_unix": time.time()}
    mgr.state["workers"]["drv"] = worker

    with mgr.lock:
        mgr._mark_chat_active("m-1", "c-old")
    mgr.chat_waiters["c-old"] = {"event": threading.Event(), "result": None}
    mgr.chat_streams["c-old"] = object()
    mgr.chat_contexts["c-old"] = {
        "driver": "drv",
        "mesh_key": "m-1",
        "state": "leased",
    }
    mgr.chat_pending["drv"] = [
        {"chat_id": "c-old", "messages": [{"role": "user", "content": "old"}]}
    ]

    # At capacity (forced to 1 here) a fresh slot refuses the next test.
    monkeypatch.setattr(pool_module, "_serve_parallel_slots", lambda: 1)
    with mgr.lock, pytest.raises(ValueError, match="already running"):
        mgr._guard_chat_slot("m-1", worker)

    # Age it past the cap: reclaimed, no raise, orphaned waiter+stream dropped.
    monkeypatch.setattr("verallm.mesh.pool.CHAT_ACTIVE_MAX_S", 0.0)
    with mgr.lock:
        mgr._guard_chat_slot("m-1", worker)  # must not raise
    assert "m-1" not in mgr.chat_active
    assert "c-old" not in mgr.chat_waiters
    assert "c-old" not in mgr.chat_streams
    assert "c-old" not in mgr.chat_contexts
    assert mgr.chat_pending.get("drv", []) == []


def test_chat_result_clears_lock_timestamp(tmp_path):
    """Delivering a result frees the lock AND its timestamp, so a mesh is
    immediately reusable and no stale timestamp lingers."""
    mgr = _direct_manager(tmp_path)
    mgr.state["workers"]["drv"] = {
        "worker_id": "drv",
        "status": "serving",
        "commands": [],
    }
    with mgr.lock:
        mgr._mark_chat_active("m-1", "c-1")
    assert "c-1" in mgr.chat_active["m-1"]
    mgr.chat_waiters["c-1"] = {"event": threading.Event(), "result": None}
    mgr.chat_contexts["c-1"] = {
        "driver": "drv",
        "mesh_key": "m-1",
        "expected_stage_count": 1,
        "state": "running",
        "last_seq": 0,
        "stream": False,
        "client_disconnected": False,
    }
    assert mgr.handle_chat_result({
        "pool_secret": mgr.state["pool_secret"], "worker_id": "drv",
        "chat_id": "c-1", "content": "pong", "verified": True,
    }) == {"status": "ok"}
    assert "m-1" not in mgr.chat_active


def test_pool_worker_loop_refuses_duplicate_daemon_on_workdir(tmp_path):
    """One daemon per workdir: a second instance must die loudly at start.

    Two daemons on one workdir race their control joins (each rotates the
    session pin, 403-ing the other's delegated signing) and double every
    capacity-audit workload —."""

    import fcntl
    from pathlib import Path

    _state_dir, token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:9", serving_mode="dev"
    )
    config = _worker_config(token, tmp_path, "dup")
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    holder = open(Path(config.workdir) / "pool-worker.lock", "w")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            pool_worker_loop(
                config,
                runner=_StubRunner(),
                stop_event=threading.Event(),
                max_beats=1,
            )
    finally:
        holder.close()


def test_worker_rejoins_after_manager_reset(tmp_path):
    """If the manager is reset and forgets a worker, the worker re-joins on the
    next heartbeat instead of heartbeating into the void (what dropped the Mac)."""
    state_dir, token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    server = serve_pool_manager(state_dir, host="127.0.0.1", port=0)
    host, port = server.server_address
    endpoint = f"http://{host}:{port}"
    token = MeshPoolToken(
        pool_id=token.pool_id, manager_endpoint=endpoint, pool_secret=token.pool_secret
    )
    credentials = _PoolCredentials(worker=token, management=admin_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manager = server.pool_manager  # type: ignore[attr-defined]
    stop = threading.Event()
    worker = threading.Thread(
        target=pool_worker_loop,
        args=(_worker_config(token, tmp_path, "ghost"),),
        kwargs={"runner": _StubRunner(), "stop_event": stop},
        daemon=True,
    )
    worker.start()
    try:
        _wait(lambda: "ghost" in _call(endpoint, credentials, "/v1/pool/status", {})["workers"])
        # Simulate a manager reset that forgets every worker.
        with manager.lock:
            manager.state["workers"].clear()
            manager._save()
        assert "ghost" not in _call(endpoint, credentials, "/v1/pool/status", {})["workers"]
        # Next heartbeat returns unknown-worker; the worker re-joins itself.
        _wait(lambda: "ghost" in _call(endpoint, credentials, "/v1/pool/status", {})["workers"])
    finally:
        stop.set()
        worker.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _StubRunner:
    def __init__(self):
        self.events = []

    def drive(self, command, progress=None):
        self.events.append(("drive", command["mesh_key"]))
        return {
            "event": "drive_ready", "mesh_id": "mesh-stub",
            "join_token": "vtmesh_stub", "coordinator_endpoint": "http://stub:1",
        }

    def join(self, command):
        self.events.append(("join", command["mesh_key"]))
        return {"event": "serving"}

    def verify_backend_ready(self):
        # Multi-box drivers verify the split backend off-thread after drive_ready;
        # the stub has no real backend, so this just lets the verifier proceed to
        # report "serving".
        self.events.append(("verify", ""))

    def stop(self, command):
        self.events.append(("stop", command.get("mesh_key", "")))
        return {"event": "stopped"}


def test_parent_worker_delegated_capacity_sign_retries_with_fresh_auth(
    monkeypatch,
    tmp_path,
):
    token = MeshPoolToken(
        pool_id="pool-sign-retry",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    runner = _StubRunner()
    from verallm.mesh.receipt_signing import ensure_stage_proof_key_file

    runner._stage_proof_keypair = ensure_stage_proof_key_file(
        tmp_path / "sign-worker-stage.seed"
    )
    runner._stage_proof_key_ss58 = str(
        runner._stage_proof_keypair.ss58_address
    )
    stop = threading.Event()
    sign_payloads = []

    def fake_post_json(url, body, **kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "sign-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            stop.wait(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {"status": "ok", "command": None, "probe": {}}
        if url.endswith("/v1/pool/coordinator-sign"):
            sign_payloads.append(dict(body))
            assert kwargs["timeout"] == 3.0
            assert kwargs["keepalive"] is True
            assert kwargs["keepalive_fallback"] is False
            if len(sign_payloads) == 1:
                raise RuntimeError("HTTP 429 from manager: retry")
            return {"status": "ok", "signature": "ab" * 65}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "verallm.mesh.delegated_signing.time.sleep", lambda _seconds: None
    )

    pool_worker_loop(
        _worker_config(token, tmp_path, "sign-worker"),
        runner=runner,
        stop_event=stop,
        max_beats=1,
    )
    response = runner._coordinator_sign_request(
        "coordinator-sign",
        {
            "purpose": "capacity-audit-artifact",
            "mesh_key": "mesh-1",
            "artifact": {"type": "capacity_audit_final_receipt"},
        },
    )
    stop.set()

    assert response["status"] == "ok"
    assert len(sign_payloads) == 2
    assert len({body["worker_auth_nonce"] for body in sign_payloads}) == 2
    assert all(
        body["purpose"] == "capacity-audit-artifact"
        for body in sign_payloads
    )
    assert sign_payloads[0]["artifact"] == sign_payloads[1]["artifact"]


def _worker_config(token, tmp_path, worker_id):
    return PoolWorkerConfig(
        token=token, repo_root=tmp_path, workdir=tmp_path / worker_id,
        advertise_host="127.0.0.1", rpc_port=50052, proof_port=9402,
        mesh_port=9500, llama_server_binary="llama-server",
        rpc_worker_binary="rpc-server", catalog=[{"model_id": "m1"}],
        worker_id=worker_id, vram_gb=48, heartbeat_s=0.05,
    )


class _BlockingValidatorRunner(_StubRunner):
    snapshot_hash = "ab" * 32

    def __init__(self):
        super().__init__()
        self.verify_started = threading.Event()
        self.release_verify = threading.Event()

    def verify_backend_ready(self):
        self.events.append(("verify", ""))
        self.verify_started.set()
        if not self.release_verify.wait(2.0):
            raise RuntimeError("test did not release split-backend verification")
        return self.snapshot_hash


@pytest.mark.parametrize(
    ("model_index", "snapshot_hash", "expected_event"),
    [
        (None, "", "serving"),
        (0, "ab" * 32, "serving"),
        (12, "ab" * 32, "serving"),
        (0, "", "error"),
        (12, "not-a-snapshot", "error"),
    ],
)
def test_worker_multibox_phase_retry_continues_once_without_report_flood(
    monkeypatch,
    tmp_path,
    model_index,
    snapshot_hash,
    expected_event,
):
    token = MeshPoolToken(
        pool_id="pool-phase-retry",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "d" * 32,
            "action": "drive",
            "mesh_key": "m-phase-retry",
            "model_id": "m1",
            "model_index": model_index,
            "member_count": 2,
            "serving_mode": "validator",
        }
    )
    runner = _BlockingValidatorRunner()
    runner.snapshot_hash = snapshot_hash
    stop = threading.Event()
    terminal = threading.Event()
    heartbeats: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "phase-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            stop.wait(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            heartbeats.append(dict(body))
            return {
                "status": "ok",
                "command": None if terminal.is_set() else dict(command),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            report = dict(body)
            reports.append(report)
            if report["event"] == "drive_ready":
                # Exhaust the first bounded delivery.  The next heartbeat must
                # retry the persisted phase report and start the continuation
                # only after that retry receives the manager's phase ACK.
                drive_attempts = sum(
                    item.get("event") == "drive_ready" for item in reports
                )
                if drive_attempts == 1:
                    time.sleep(0.02)
                    raise TimeoutError("lost initial phase delivery")
                return {"status": "ok", "command_phase": "drive_ready"}
            assert report["event"] == expected_event
            terminal.set()
            return {
                "status": "ok",
                "command_completed": command["command_id"],
            }
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pool_module, "WORKER_REPORT_DELIVERY_MAX_S", 0.01)
    monkeypatch.setattr(pool_module, "CHAT_POLL_RETRY_S", 0.0)
    config = _worker_config(token, tmp_path, "phase-worker")
    worker = threading.Thread(
        target=pool_worker_loop,
        args=(config,),
        kwargs={"runner": runner, "stop_event": stop, "max_beats": 20},
        daemon=True,
    )
    worker.start()

    assert runner.verify_started.wait(2.0)
    _wait(lambda: len(heartbeats) >= 4)
    assert [item["event"] for item in reports] == [
        "drive_ready",
        "drive_ready",
    ]
    runner.release_verify.set()
    assert terminal.wait(2.0)
    _wait(lambda: any(item.get("event") == expected_event for item in reports))
    journal_path = config.workdir / pool_module.POOL_WORKER_COMMAND_JOURNAL_FILE
    _wait(
        lambda: json.loads(journal_path.read_text())["commands"][
            command["command_id"]
        ]["state"]
        == "completed"
    )
    stop.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert runner.events == [
        ("drive", "m-phase-retry"),
        ("verify", ""),
    ]
    terminal_report = reports[-1]
    assert terminal_report["event"] == expected_event
    if expected_event == "error":
        assert "without a signed snapshot" in terminal_report["message"]
    elif model_index is not None:
        assert terminal_report["verification_snapshot_hash"] == runner.snapshot_hash
    else:
        assert "verification_snapshot_hash" not in terminal_report
    journal = json.loads(journal_path.read_text())
    assert journal["commands"][command["command_id"]]["state"] == "completed"


def test_worker_stale_multibox_phase_does_not_start_verifier(
    monkeypatch,
    tmp_path,
):
    token = MeshPoolToken(
        pool_id="pool-stale-phase",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "e" * 32,
            "action": "drive",
            "mesh_key": "m-stale-phase",
            "model_id": "m1",
            "member_count": 2,
            "serving_mode": "validator",
        }
    )
    runner = _BlockingValidatorRunner()
    stop = threading.Event()
    stale = threading.Event()
    reports: list[dict[str, object]] = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "stale-phase-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            stop.wait(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {
                "status": "ok",
                "command": None if stale.is_set() else dict(command),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            reports.append(dict(body))
            stale.set()
            return {"status": "stale", "command_id": command["command_id"]}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    config = _worker_config(token, tmp_path, "stale-phase-worker")
    worker = threading.Thread(
        target=pool_worker_loop,
        args=(config,),
        kwargs={"runner": runner, "stop_event": stop, "max_beats": 5},
        daemon=True,
    )
    worker.start()

    assert stale.wait(2.0)
    journal_path = config.workdir / pool_module.POOL_WORKER_COMMAND_JOURNAL_FILE
    _wait(
        lambda: json.loads(journal_path.read_text())["commands"][
            command["command_id"]
        ]["state"]
        == "completed"
    )
    worker.join(timeout=2.0)
    stop.set()

    assert not worker.is_alive()
    assert [item["event"] for item in reports] == ["drive_ready"]
    assert runner.events == [("drive", "m-stale-phase")]
    assert not runner.verify_started.is_set()
    journal = json.loads(journal_path.read_text())
    assert journal["commands"][command["command_id"]]["state"] == "completed"


@pytest.mark.parametrize(
    "malformation",
    ("missing-command-id", "missing-command-digest", "digest-mismatch"),
)
def test_worker_loop_refuses_unbound_command_before_claim_or_execution(
    monkeypatch,
    tmp_path,
    malformation,
):
    token = MeshPoolToken(
        pool_id="pool-malformed-command",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "9" * 32,
            "action": "drive",
            "mesh_key": "m-malformed",
            "model_id": "m1",
            "member_count": 1,
        }
    )
    if malformation == "missing-command-id":
        command.pop("command_id")
    elif malformation == "missing-command-digest":
        command.pop("command_digest")
    else:
        digest = str(command["command_digest"])
        command["command_digest"] = (
            ("0" if digest[0] != "0" else "1") + digest[1:]
        )
    reports = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "malformed-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            time.sleep(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {
                "status": "ok",
                "command": dict(command),
                "command_acknowledged": "",
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            reports.append(dict(body))
            return {"status": "ok"}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    runner = _StubRunner()
    stop = threading.Event()
    config = _worker_config(token, tmp_path, "malformed-worker")

    pool_worker_loop(
        config,
        runner=runner,
        stop_event=stop,
        max_beats=1,
    )
    stop.set()

    assert runner.events == []
    assert reports == []
    assert not (
        config.workdir / pool_module.POOL_WORKER_COMMAND_JOURNAL_FILE
    ).exists()


def test_worker_loop_executes_a_redelivered_command_id_once(
    monkeypatch,
    tmp_path,
):
    token = MeshPoolToken(
        pool_id="pool-duplicate-command",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    runner = _StubRunner()
    stop = threading.Event()
    duplicate = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "a" * 32,
            "action": "drive",
            "mesh_key": "m-duplicate",
            "model_id": "m1",
            "member_count": 1,
        }
    )
    reports = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "worker-one"}
        if url.endswith("/v1/pool/chat-poll"):
            stop.wait(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {
                "status": "ok",
                # Simulate the same persisted in-flight delivery appearing in
                # consecutive responses while its acknowledgement crosses.
                "command": dict(duplicate),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            reports.append(dict(body))
            return {"status": "ok"}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)

    pool_worker_loop(
        _worker_config(token, tmp_path, "worker-one"),
        runner=runner,
        stop_event=stop,
        max_beats=2,
    )
    _wait(lambda: len(reports) >= 1)
    stop.set()

    assert runner.events == [("drive", "m-duplicate")]
    assert all(
        report["command_id"] == duplicate["command_id"]
        for report in reports
    )
    assert all(
        report["command_digest"] == duplicate["command_digest"]
        for report in reports
    )


def test_worker_command_journal_replays_completion_after_restart_without_execution(
    monkeypatch,
    tmp_path,
):
    token = MeshPoolToken(
        pool_id="pool-command-journal",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "b" * 32,
            "action": "drive",
            "mesh_key": "m-journal",
            "model_id": "m1",
            "member_count": 1,
        }
    )
    reports = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "journal-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            time.sleep(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {
                "status": "ok",
                "command": dict(command),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            reports.append(dict(body))
            return {"status": "ok"}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    config = _worker_config(token, tmp_path, "journal-worker")

    first_runner = _StubRunner()
    first_stop = threading.Event()
    pool_worker_loop(
        config,
        runner=first_runner,
        stop_event=first_stop,
        max_beats=1,
    )
    _wait(lambda: len(reports) >= 1)
    first_stop.set()

    second_runner = _StubRunner()
    second_stop = threading.Event()
    pool_worker_loop(
        config,
        runner=second_runner,
        stop_event=second_stop,
        max_beats=1,
    )
    _wait(lambda: len(reports) >= 2)
    second_stop.set()

    assert first_runner.events == [("drive", "m-journal")]
    assert second_runner.events == []
    assert reports[0]["command_id"] == command["command_id"]
    assert reports[1]["command_id"] == command["command_id"]
    assert reports[0]["command_digest"] == command["command_digest"]
    assert reports[1]["command_digest"] == command["command_digest"]
    assert reports[0]["event"] == reports[1]["event"] == "drive_ready"


def test_worker_stop_command_reports_command_bound_stopped_completion(
    monkeypatch,
    tmp_path,
):
    token = MeshPoolToken(
        pool_id="pool-stop-command",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    stop_command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "c" * 32,
            "action": "stop",
            "mesh_key": "m-stop",
        }
    )
    reports = []

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            return {"status": "joined", "worker_id": "stop-worker"}
        if url.endswith("/v1/pool/chat-poll"):
            time.sleep(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            return {
                "status": "ok",
                "command": dict(stop_command),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            reports.append(dict(body))
            return {"status": "ok"}
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_args, **_kwargs: None)
    runner = _StubRunner()
    stop = threading.Event()

    pool_worker_loop(
        _worker_config(token, tmp_path, "stop-worker"),
        runner=runner,
        stop_event=stop,
        max_beats=1,
    )
    _wait(lambda: len(reports) == 1)
    stop.set()

    assert runner.events == [("stop", "m-stop")]
    assert reports[0]["event"] == "stopped"
    assert reports[0]["command_id"] == stop_command["command_id"]
    assert reports[0]["command_digest"] == stop_command["command_digest"]


def test_worker_loop_executes_drive_then_member_joins(pool, tmp_path):
    endpoint, credentials = pool
    token = MeshPoolToken(
        pool_id=credentials.pool_id,
        manager_endpoint=endpoint,
        pool_secret=credentials.pool_secret,
    )
    driver_runner, member_runner = _StubRunner(), _StubRunner()

    # Run each worker loop continuously in a thread (the real deployment shape:
    # a worker joins ONCE at start, then heartbeats — it does not re-join, so a
    # launch mid-flight is picked up without the manager treating it as a
    # restart and dropping the mesh).
    stops = [threading.Event(), threading.Event()]
    threads = [
        threading.Thread(
            target=pool_worker_loop,
            args=(_worker_config(token, tmp_path, wid),),
            kwargs={"runner": r, "stop_event": s},
            daemon=True,
        )
        for wid, r, s in (("drv", driver_runner, stops[0]), ("mem", member_runner, stops[1]))
    ]
    try:
        for t in threads:
            t.start()
        _wait(lambda: set(_call(endpoint, credentials, "/v1/pool/status", {})["workers"]) >= {"drv", "mem"})
        launched = _call(
            endpoint, credentials, "/v1/pool/launch",
            {"model_id": "m1", "workers": ["drv", "mem"], "driver": "drv"},
        )
        mesh_key = launched["mesh_key"]
        _wait(lambda: _call(endpoint, credentials, "/v1/pool/status", {})["meshes"].get(mesh_key, {}).get("status") == "serving")
    finally:
        for s in stops:
            s.set()
        for t in threads:
            t.join(timeout=2)

    # Driver drives, then verifies the fully-joined split backend before serving.
    assert driver_runner.events == [("drive", mesh_key), ("verify", "")]
    assert member_runner.events == [("join", mesh_key)]
    status = _call(endpoint, credentials, "/v1/pool/status", {})
    assert "join_token" not in status["meshes"][mesh_key]
    assert "vtmesh_stub" not in json.dumps(status, sort_keys=True)


def test_recommend_fileless_member_and_fetchable_driver(pool):
    """Only the driver needs the model: a worker WITH it pairs with a file-less
    worker (contributing VRAM), and a worker that can fetch the model can drive."""
    endpoint, token = pool
    # 'hasit' has the 40GB model; 'bare' has nothing (file-less member).
    _join(endpoint, token, "hasit", vram=24, model_bytes=40_000_000_000)
    _call(endpoint, token, "/v1/pool/join", {
        "worker_id": "bare", "capability": {"vram_gb": 40},
        "catalog": [], "endpoints": {"proof": "http://bare.local:9402"}})
    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "m1"})["suggestions"]
    # 40GB*1.3=52 > 24, so a pair is needed; hasit must be the driver, bare the
    # file-less member.
    pairs = [s for s in got if len(s["workers"]) == 2]
    assert pairs, "file-less member should enable a viable pair"
    assert pairs[0]["driver"] == "hasit"
    assert set(pairs[0]["workers"]) == {"hasit", "bare"}

    # A worker advertising an hf source can DRIVE a model it lacks (auto-fetch),
    # flagged with the download size.
    _call(endpoint, token, "/v1/pool/join", {
        "worker_id": "learner", "capability": {"vram_gb": 80},
        "catalog": [{"model_id": "m9", "model_bytes": 1000, "layers": 4,
                     "hf_repo": "org/repo", "hf_files": ["m9.gguf"]}],
        "endpoints": {"proof": "http://learner.local:9402"}})
    _call(endpoint, token, "/v1/pool/join", {
        "worker_id": "empty", "capability": {"vram_gb": 80},
        "catalog": [], "endpoints": {"proof": "http://empty.local:9402"}})
    got = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "m9"})["suggestions"]
    single = next(s for s in got if s["workers"] == ["empty"] or s["workers"] == ["learner"])
    # 'empty' lacks m9 and has no source -> cannot be a single driver; only
    # 'learner' (has the entry+source) can. The teacher's registry entry lets a
    # DIFFERENT fetchable-source worker drive... here only learner has it.
    assert single["driver"] == "learner"


def test_tolerance_survives_auto_fetch_path(pool):
    """Per-model proof tolerances must reach the drive command even when the
    DRIVER lacks the model: they ride the teacher's advert into the pool
    registry and from there into the mesh record + fetch spec + drive command.
    (Dropping them re-runs the default band on exactly the models that need
    the override — the q3_K intermittent proof-failure bug.)"""
    endpoint, token = pool
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "teacher",
            "capability": {"gpu_name": "test", "vram_gb": 48},
            "catalog": [{"model_id": "q3", "model_bytes": 1000, "layers": 12,
                         "hf_repo": "org/q3", "hf_files": ["q3.gguf"],
                         "proof_tolerance_rel": 0.10}],
            "endpoints": {"rpc": "t.local:50052", "proof": "http://t.local:9402",
                          "mesh": "http://t.local:9500"},
        },
    )
    _join(endpoint, token, "learner")  # has only m1, must fetch q3
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "q3", "workers": ["learner"], "driver": "learner"},
    )
    mesh_key = launched["mesh_key"]
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "learner"})
    fetch_command = beat["command"]
    fetch_id = _assert_command_id(fetch_command)
    assert fetch_command["action"] == "fetch"
    # The spec the worker downloads from carries the override (it banks it in
    # its new catalog entry for future launches).
    assert fetch_command["spec"]["proof_tolerance_rel"] == 0.10
    _call(
        endpoint, token, "/v1/pool/report",
        _command_report(
            fetch_command,
            worker_id="learner",
            event="fetched",
            entry={
                "model_id": "q3",
                "llama_model": "/x/q3.gguf",
                "manifest": "/x/q3.json",
                "layers": 12,
                "model_bytes": 1000,
            },
        ),
    )
    beat = _call(
        endpoint,
        token,
        "/v1/pool/heartbeat",
        {"worker_id": "learner", "command_ack": fetch_id},
    )
    _assert_command_id(beat["command"])
    assert beat["command"]["action"] == "drive"
    assert beat["command"]["proof_tolerance_rel"] == 0.10


def test_launch_refuses_fetch_that_cannot_fit_disk(pool):
    """A download bigger than the driver's advertised free disk is refused up
    front with a clear error (hf_hub only warns, then dies mid-file)."""
    endpoint, token = pool
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "teacher",
            "capability": {"gpu_name": "test", "vram_gb": 48},
            "catalog": [{"model_id": "big", "model_bytes": 35_000_000_000,
                         "layers": 80, "hf_repo": "org/big", "hf_files": ["b.gguf"]}],
            "endpoints": {"rpc": "t.local:50052", "proof": "http://t.local:9402",
                          "mesh": "http://t.local:9500"},
        },
    )
    _call(
        endpoint, token, "/v1/pool/join",
        {
            "worker_id": "cramped",
            "capability": {"gpu_name": "test", "vram_gb": 80, "free_disk_gb": 2.7},
            "catalog": [],
            "endpoints": {"rpc": "c.local:50052", "proof": "http://c.local:9402",
                          "mesh": "http://c.local:9500"},
        },
    )
    with pytest.raises(RuntimeError, match="free disk"):
        _call(endpoint, token, "/v1/pool/launch",
              {"model_id": "big", "workers": ["cramped"], "driver": "cramped"})
    # recommend() explains WHY the box can't drive instead of proposing a
    # download that would die at 7%.
    rec = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "big"})
    assert "free disk" in rec["reasons"]["cramped"] or "GB" in rec["reasons"]["cramped"]
    assert not any(
        s["driver"] == "cramped" for s in rec["suggestions"]
    )


def test_rejoin_stops_surviving_members(pool):
    """When a mesh member rejoins (worker restart), the OTHER members of its
    dead mesh get a stop command — otherwise their serve processes keep
    holding VRAM/ports while the manager thinks the boxes are idle."""
    endpoint, token = pool
    _join(endpoint, token, "w1", vram=24)
    _join(endpoint, token, "w2", vram=24)
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m1", "workers": ["w1", "w2"], "driver": "w1"},
    )
    mesh_key = launched["mesh_key"]
    # Driver restarts and rejoins -> both members remain reserved until they
    # complete the pre-emptive stop command for the now-invalid mesh.
    _join(endpoint, token, "w1", vram=24)
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][mesh_key]["status"] == "stopping"
    assert status["workers"]["w1"]["status"] == "stopping"
    assert status["workers"]["w2"]["status"] == "stopping"

    for worker_id in ("w1", "w2"):
        stop_command = _call(
            endpoint,
            token,
            "/v1/pool/heartbeat",
            {"worker_id": worker_id},
        )["command"]
        _assert_command_id(stop_command)
        assert {
            key: stop_command[key]
            for key in ("action", "mesh_key")
        } == {"action": "stop", "mesh_key": mesh_key}
        _call(
            endpoint,
            token,
            "/v1/pool/report",
            _command_report(
                stop_command,
                worker_id=worker_id,
                event="stopped",
            ),
        )

    status = _call(endpoint, token, "/v1/pool/status", {})
    assert mesh_key not in status["meshes"]
    assert status["workers"]["w1"]["status"] == "idle"
    assert status["workers"]["w2"]["status"] == "idle"


def _direct_manager_with_drive_command(tmp_path):
    state_dir, tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    mgr = PoolManager(state_dir)
    worker_body = {"pool_secret": tok.pool_secret}
    admin_body = {"management_secret": mgr.state["management_secret"]}
    mgr.handle_join({**worker_body, "worker_id": "w1",
                     "capability": {"gpu_name": "t", "vram_gb": 48},
                     "catalog": [{"model_id": "m1", "model_bytes": 1000}],
                     "endpoints": {"rpc": "w1:50052", "proof": "http://w1:9402",
                                   "mesh": "http://w1:9500"}})
    launched = mgr.handle_launch(
        {**admin_body, "model_id": "m1", "workers": ["w1"], "driver": "w1"}
    )
    return state_dir, tok, mgr, worker_body, launched["mesh_key"]


def test_lost_heartbeat_response_redelivers_the_same_command_id(tmp_path):
    _state_dir, _tok, mgr, worker_body, _mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )

    first = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
    first_command = first["command"]
    first_id = _assert_command_id(first_command)

    # Model a response lost after the manager persisted delivery: the worker
    # sends no acknowledgement because it never saw the first response.
    second = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
    assert second["command"] == first_command
    assert second["command"]["command_id"] == first_id
    assert mgr.state["workers"]["w1"]["command_inflight"] == first_command
    assert mgr.state["workers"]["w1"]["commands"] == []


def test_command_receipt_ack_does_not_clear_the_inflight_command(tmp_path):
    _state_dir, _tok, mgr, worker_body, _mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    command_id = _assert_command_id(delivered)

    acknowledged = mgr.handle_heartbeat(
        {
            **worker_body,
            "worker_id": "w1",
            "command_ack": command_id,
        }
    )
    assert acknowledged["command_acknowledged"] == command_id
    assert acknowledged["command"] == delivered
    assert mgr.state["workers"]["w1"]["command_inflight"] == delivered
    assert mgr.state["workers"]["w1"]["last_command_ack"] == command_id


def test_exact_terminal_report_clears_the_inflight_command(tmp_path):
    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    command_id = _assert_command_id(delivered)

    completed = mgr.handle_report(
        {
            **worker_body,
            **_command_report(
                delivered,
                worker_id="w1",
                event="drive_ready",
                mesh_id="mesh-terminal",
                join_token="vtmesh_terminal",
                coordinator_endpoint="http://w1:9500",
            ),
        }
    )

    assert completed == {
        "status": "ok",
        "command_completed": command_id,
    }
    assert mgr.state["workers"]["w1"]["command_inflight"] is None
    assert mgr.state["workers"]["w1"]["last_command_completion"][
        "command_digest"
    ] == delivered["command_digest"]
    assert mgr.state["meshes"][mesh_key]["status"] == "serving"


def test_command_reports_reject_missing_or_mismatched_command_binding(tmp_path):
    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    report = {
        **worker_body,
        **_command_report(
            delivered,
            worker_id="w1",
            event="drive_ready",
            mesh_id="mesh-rejection-matrix",
            join_token="vtmesh_rejection_matrix",
            coordinator_endpoint="http://w1:9500",
        ),
    }

    missing_id = dict(report)
    missing_id.pop("command_id")
    with pytest.raises(ValueError, match="valid command_id"):
        mgr.handle_report(missing_id)

    missing_digest = dict(report)
    missing_digest.pop("command_digest")
    with pytest.raises(ValueError, match="valid command_digest"):
        mgr.handle_report(missing_digest)

    with pytest.raises(PermissionError, match="digest mismatch"):
        mgr.handle_report({**report, "command_digest": "f" * 64})

    with pytest.raises(PermissionError, match="mesh does not match"):
        mgr.handle_report({**report, "mesh_key": "m-wrong-mesh"})

    with pytest.raises(ValueError, match="invalid for command action drive"):
        mgr.handle_report({**report, "event": "stopped"})

    mgr.handle_join(
        {
            **worker_body,
            "worker_id": "w2",
            "capability": {"gpu_name": "t", "vram_gb": 48},
            "catalog": [{"model_id": "m1", "model_bytes": 1000}],
            "endpoints": {
                "rpc": "w2:50052",
                "proof": "http://w2:9402",
                "mesh": "http://w2:9500",
            },
        }
    )
    cross_worker = mgr.handle_report({**report, "worker_id": "w2"})
    assert cross_worker == {
        "status": "stale",
        "command_id": delivered["command_id"],
    }
    assert mgr.state["workers"]["w1"]["command_inflight"] == delivered
    assert mgr.state["meshes"][mesh_key]["status"] == "driving"

    with mgr.lock:
        mgr._queue_worker_command(
            mgr.state["workers"]["w1"],
            {"action": "stop", "mesh_key": mesh_key},
        )
        mgr._save()
    stale = mgr.handle_report(report)
    assert stale == {
        "status": "stale",
        "command_id": delivered["command_id"],
    }
    assert mgr.state["workers"]["w1"]["command_inflight"]["action"] == "stop"


def test_stale_command_ack_cannot_clear_a_newer_inflight_command(tmp_path):
    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    first = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})["command"]
    first_id = _assert_command_id(first)
    acknowledged = mgr.handle_heartbeat(
        {**worker_body, "worker_id": "w1", "command_ack": first_id}
    )
    assert acknowledged["command"] == first
    with mgr.lock:
        mgr._queue_worker_command(
            mgr.state["workers"]["w1"],
            {"action": "stop", "mesh_key": mesh_key},
        )
        mgr._save()

    preempted = mgr.handle_heartbeat(
        {**worker_body, "worker_id": "w1", "command_ack": first_id}
    )
    second = preempted["command"]
    second_id = _assert_command_id(second)
    assert second_id != first_id
    assert preempted["command_acknowledged"] == first_id

    stale = mgr.handle_heartbeat(
        {**worker_body, "worker_id": "w1", "command_ack": first_id}
    )
    assert stale["command"] == second
    assert mgr.state["workers"]["w1"]["command_inflight"] == second
    assert mgr.state["workers"]["w1"]["last_command_ack"] == first_id


def test_manager_restart_retains_and_redelivers_inflight_command(tmp_path):
    state_dir, _tok, mgr, worker_body, _mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    command_id = _assert_command_id(delivered)

    # Reload from disk, as a restarted manager would. The command remains
    # in-flight and is redelivered until an exact terminal report completes it.
    mgr2 = PoolManager(state_dir)
    assert mgr2.state["workers"]["w1"]["commands"] == []
    assert mgr2.state["workers"]["w1"]["command_inflight"] == delivered
    replayed = mgr2.handle_heartbeat({**worker_body, "worker_id": "w1"})
    assert replayed["command"] == delivered
    assert replayed["command"]["command_id"] == command_id


def test_register_model_enables_auto_fetch_launch(pool):
    """An operator can teach the pool a model NO worker has: registering the
    download source makes it launchable (driver auto-fetches), including the
    per-model proof tolerances."""
    endpoint, token = pool
    _join(endpoint, token, "solo", vram=48)  # has only m1
    reg = _call(
        endpoint, token, "/v1/pool/register-model",
        {"model_id": "new9", "hf_repo": "org/new9", "hf_files": ["n9.gguf"],
         "layers": 64, "model_bytes": 2000, "proof_tolerance_rel": 0.07,
         "model_index": 4, "max_context_len": 32_768},
    )
    assert reg["status"] == "ok"
    assert reg["registry"]["model_index"] == 4
    assert reg["registry"]["max_context_len"] == 32_768
    # It shows up for the dashboard's model list.
    assert "new9" in _call(endpoint, token, "/v1/pool/status", {})["models"]
    # And recommend/launch treat the solo box as a fetch-driver.
    rec = _call(endpoint, token, "/v1/pool/recommend", {"model_id": "new9"})
    assert any(s.get("fetch") for s in rec["suggestions"])
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "new9", "workers": ["solo"], "driver": "solo"},
    )
    beat = _call(endpoint, token, "/v1/pool/heartbeat", {"worker_id": "solo"})
    assert beat["command"]["action"] == "fetch"
    assert beat["command"]["spec"]["hf_repo"] == "org/new9"
    assert beat["command"]["spec"]["proof_tolerance_rel"] == 0.07
    assert beat["command"]["spec"]["max_context_len"] == 32_768
    assert launched["mesh_key"]
    status = _call(endpoint, token, "/v1/pool/status", {})
    assert status["meshes"][launched["mesh_key"]]["model_index"] == 4


def test_launch_refuses_stale_worker(tmp_path):
    """Launching onto a worker with no recent heartbeat must be refused: it is
    offline (mesh would hang) or mid-restart, and a mid-restart worker's
    imminent (re)join drops the brand-new mesh as leftover state."""
    from verallm.mesh.pool import PoolManager, WORKER_STALE_S

    state_dir, tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    mgr = PoolManager(state_dir)
    worker_body = {"pool_secret": tok.pool_secret}
    admin_body = {"management_secret": mgr.state["management_secret"]}
    mgr.handle_join({**worker_body, "worker_id": "w1",
                     "capability": {"gpu_name": "t", "vram_gb": 48},
                     "catalog": [{"model_id": "m1", "model_bytes": 1000}],
                     "endpoints": {"rpc": "w1:50052", "proof": "http://w1:9402",
                                   "mesh": "http://w1:9500"}})
    mgr.state["workers"]["w1"]["last_seen_unix"] = int(time.time()) - int(WORKER_STALE_S) - 5
    with pytest.raises(ValueError, match="offline"):
        mgr.handle_launch({**admin_body, "model_id": "m1", "workers": ["w1"], "driver": "w1"})


def test_silent_driver_during_formation_queues_confirmed_teardown(tmp_path):
    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    admin_body = {"management_secret": mgr.state["management_secret"]}
    mgr.state["workers"]["w1"]["last_seen_unix"] = int(time.time()) - max(
        121,
        int(4 * pool_module.WORKER_STALE_S) + 1,
    )

    status = mgr.handle_status(admin_body)

    assert status["meshes"][mesh_key]["status"] == "error"
    assert "went silent" in status["meshes"][mesh_key]["error"]
    assert status["workers"]["w1"]["status"] == "stopping"
    stop_command = mgr.handle_heartbeat(
        {**worker_body, "worker_id": "w1"}
    )["command"]
    assert stop_command["action"] == "stop"
    assert stop_command["mesh_key"] == mesh_key


def test_requested_chat_proof_tier_accepts_operator_selection():
    from verallm.mesh.pool import _requested_chat_proof_tier

    assert _requested_chat_proof_tier({}) == ""
    assert _requested_chat_proof_tier({"proof_tier": "auto"}) == ""
    assert _requested_chat_proof_tier({"proof_tier": "light"}) == "light"
    assert _requested_chat_proof_tier({"proof_tier": "HARD"}) == "hard"


def test_requested_chat_proof_tier_refuses_probe_downgrade():
    # The probe gate asserts the hard relation; letting a probe request
    # light would silently gate deployments on a weaker claim.
    from verallm.mesh.pool import _requested_chat_proof_tier

    with pytest.raises(ValueError, match="probe cannot downgrade"):
        _requested_chat_proof_tier({"proof_tier": "light", "probe": True})
    assert _requested_chat_proof_tier({"proof_tier": "hard", "probe": True}) == "hard"


def test_requested_chat_proof_tier_rejects_unknown_values():
    from verallm.mesh.pool import _requested_chat_proof_tier

    with pytest.raises(ValueError, match="auto, light, or hard"):
        _requested_chat_proof_tier({"proof_tier": "none"})


def test_detect_gpu_capability_parses_and_fails_open(monkeypatch):
    import subprocess

    from verallm.mesh.cli import _detect_gpu_capability

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class _Ok:
        returncode = 0
        stdout = "0, NVIDIA GeForce RTX 5090, 32607\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Ok())
    assert _detect_gpu_capability() == (
        "NVIDIA GeForce RTX 5090",
        31,
        ["NVIDIA GeForce RTX 5090"],
        [31],
    )

    class _FourGpus:
        returncode = 0
        stdout = (
            "0, NVIDIA A100-SXM4-80GB, 81920\n"
            "1, NVIDIA A100-SXM4-80GB, 81920\n"
            "2, NVIDIA A100-SXM4-80GB, 81920\n"
            "3, NVIDIA A100-SXM4-80GB, 81920\n"
        )

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FourGpus())
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES; the mask is applied here.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    display, total, names, per_gpu = _detect_gpu_capability()
    assert display == "NVIDIA A100-SXM4-80GB x2"
    assert total == 160
    assert per_gpu == [80, 80]
    assert names == ["NVIDIA A100-SXM4-80GB"] * 2
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    display, total, names, per_gpu = _detect_gpu_capability()
    assert total == 320 and len(per_gpu) == 4

    def _boom(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _detect_gpu_capability() == ("", 0, [], [])


def test_pcs_prover_threads_stays_per_worker_conservative(monkeypatch):
    """The cap is per WORKER UNIT, and a box runs one unit per GPU.

    A single-prover microbenchmark favours far more threads (30-core box:
    8 -> 3.64s, 15 -> 3.37s, 20 -> 3.28s), but N co-located workers
    multiply this number, so cores // 2 would put 30 prover threads on 30
    cores before serving gets any. An end-to-end A/B could not confirm a
    gain either: at a fixed 8 threads, three identical runs gave
    15.4/30.1/15.4, 16.6/20.7/25.6 and 6.7/8.8/10.6 seconds, because which
    tensor the beacon draws dominates. Pinned so it is not raised again on
    single-prover evidence alone.
    """
    import os

    from verallm.mesh.pool import _pcs_prover_threads

    for cores, expected in ((1, 1), (2, 1), (9, 8), (30, 8), (256, 8)):
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda _pid, c=cores: set(range(c))
        )
        assert _pcs_prover_threads() == expected, cores


def test_manager_startup_reconciles_orphan_meshes(tmp_path):
    """Persisted meshes that no worker can ever serve or stop again must not
    reappear after a manager restart (the resurrected-mesh incident)."""

    state_dir, _tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    state = json.loads((state_dir / "pool-state.json").read_text())
    state["workers"] = {
        "w-live": {
            "worker_id": "w-live",
            "status": "serving",
            "mesh": "m-orphan-driver",
            "capability": {},
            "endpoints": {},
            "catalog": [],
            "commands": [],
            "command_inflight": None,
            "last_seen_unix": 0,
            "rtt_ms": {},
            "peer_rtt_ms": {},
        }
    }
    state["meshes"] = {
        "m-tombstone": {
            "mesh_key": "m-tombstone",
            "status": "stopping",
            "remove_after_stop": True,
            "members": ["w-gone"],
            "driver": "w-gone",
        },
        "m-orphan-driver": {
            "mesh_key": "m-orphan-driver",
            "status": "serving",
            "members": ["w-gone-driver", "w-live"],
            "driver": "w-gone-driver",
        },
        "m-fully-gone": {
            "mesh_key": "m-fully-gone",
            "status": "serving",
            "members": ["w-gone1"],
            "driver": "w-gone1",
        },
    }
    (state_dir / "pool-state.json").write_text(json.dumps(state))
    from verallm.mesh.pool import PoolManager

    mgr = PoolManager(state_dir)
    assert "m-tombstone" not in mgr.state["meshes"]
    assert "m-fully-gone" not in mgr.state["meshes"]
    orphan = mgr.state["meshes"]["m-orphan-driver"]
    # A removable "stopping" tombstone, NOT "error": the survivor's
    # eventual stopped report must still be able to retire the entry.
    assert orphan["status"] == "stopping"
    assert orphan["remove_after_stop"] is True
    # The surviving member is told to tear its stage down.
    live = mgr.state["workers"]["w-live"]
    queued = [
        *(live.get("commands") or []),
        *(
            [live["command_inflight"]]
            if isinstance(live.get("command_inflight"), dict)
            else []
        ),
    ]
    assert any(
        c.get("action") == "stop" and c.get("mesh_key") == "m-orphan-driver"
        for c in queued
    )
    # A healthy pool (no orphans) restarts without rewriting anything.
    mgr2 = PoolManager(state_dir)
    assert "m-orphan-driver" in mgr2.state["meshes"]

    # The survivor's stopped report retires the tombstone for good: the
    # heartbeat promotes the queued stop, the report completes it.
    secret = mgr2.state["pool_secret"]
    beat = mgr2.handle_heartbeat(
        {"pool_secret": secret, "worker_id": "w-live", "status": "serving"}
    )
    command = beat["command"]
    assert command["action"] == "stop"
    done = mgr2.handle_report(
        {
            "pool_secret": secret,
            "worker_id": "w-live",
            "command_id": command["command_id"],
            "command_digest": command["command_digest"],
            "mesh_key": "m-orphan-driver",
            "event": "stopped",
        }
    )
    assert done["status"] == "ok"
    assert "m-orphan-driver" not in mgr2.state["meshes"]
    assert mgr2.state["workers"]["w-live"]["status"] == "idle"


def test_failed_stop_keeps_the_tombstone_removable(tmp_path):
    """A stop that fails must not flip a remove_after_stop tombstone to
    "error": that pinned the mesh AND its workers forever, since removal
    is gated on the "stopping" status."""

    state_dir, _tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    from verallm.mesh.pool import PoolManager

    mgr = PoolManager(state_dir)
    secret = mgr.state["pool_secret"]
    mgr.state["workers"]["w-x"] = {
        "worker_id": "w-x",
        "status": "serving",
        "mesh": "m-w",
        "capability": {},
        "endpoints": {},
        "catalog": [],
        "commands": [],
        "command_inflight": None,
        "last_seen_unix": int(time.time()),
        "rtt_ms": {},
        "peer_rtt_ms": {},
    }
    mgr.state["meshes"]["m-w"] = {
        "mesh_key": "m-w",
        "status": "serving",
        "members": ["w-x"],
        "driver": "w-x",
    }
    mgr.handle_stop({"management_secret": mgr.state["management_secret"], "mesh_key": "m-w"})
    beat = mgr.handle_heartbeat(
        {"pool_secret": secret, "worker_id": "w-x", "status": "stopping"}
    )
    command = beat["command"]
    assert command["action"] == "stop"
    mgr.handle_report(
        {
            "pool_secret": secret,
            "worker_id": "w-x",
            "command_id": command["command_id"],
            "command_digest": command["command_digest"],
            "mesh_key": "m-w",
            "event": "error",
            "message": "port still owned by an orphan",
        }
    )
    # An errored stop now CONVERGES instead of wedging: the worker is
    # released and the remove_after_stop tombstone retires immediately.
    # The dominant stop-error class is "worker restarted while the command
    # was running", where the restart already killed the serve and the
    # old behavior wedged the mesh in "stopping" for 40+ minutes while
    # every relaunch was refused .
    assert "m-w" not in mgr.state["meshes"]
    assert mgr.state["workers"]["w-x"]["status"] == "idle"
    assert mgr.state["workers"]["w-x"]["mesh"] == ""


def test_worker_tears_down_mesh_when_manager_forgets_it(monkeypatch, tmp_path):
    """A manager reset must not leave the worker's old backend owning the
    GPU/ports while the worker rejoins as 'idle' (the double-booked-GPU
    half of the resurrection incident)."""

    token = MeshPoolToken(
        pool_id="pool-reset-teardown",
        manager_endpoint="http://manager.invalid",
        pool_secret="worker-secret",
    )
    command = PoolManager._command_with_id(
        {
            "command_id": "cmd-" + "a1" * 16,
            "action": "drive",
            "mesh_key": "m-reset",
            "model_id": "m1",
            "member_count": 1,
        }
    )
    runner = _StubRunner()
    stop = threading.Event()
    rejoined = threading.Event()
    joins = {"count": 0}

    def fake_post_json(url, body, **_kwargs):
        if url.endswith("/v1/pool/join"):
            joins["count"] += 1
            if joins["count"] > 1:
                rejoined.set()
            return {"status": "joined", "worker_id": "reset-teardown"}
        if url.endswith("/v1/pool/chat-poll"):
            stop.wait(0.01)
            return {"status": "ok", "chat": []}
        if url.endswith("/v1/pool/heartbeat"):
            if str(body.get("status", "")) == "serving":
                # The manager was reset and no longer knows this worker.
                return {"status": "unknown-worker"}
            return {
                "status": "ok",
                "command": dict(command),
                "command_acknowledged": str(body.get("command_ack", "")),
                "probe": {},
            }
        if url.endswith("/v1/pool/report"):
            return {
                "status": "ok",
                "command_completed": command["command_id"],
            }
        raise AssertionError(f"unexpected pool route: {url}")

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    monkeypatch.setattr(pool_module, "_tcp_rtt_ms", lambda *_a, **_k: None)
    config = _worker_config(token, tmp_path, "reset-teardown")
    worker = threading.Thread(
        target=pool_worker_loop,
        args=(config,),
        kwargs={"runner": runner, "stop_event": stop, "max_beats": 200},
        daemon=True,
    )
    worker.start()
    try:
        assert rejoined.wait(5.0), "worker never rejoined after the reset"
        assert ("stop", "m-reset") in runner.events, (
            "worker rejoined without tearing down its orphaned mesh runtime: "
            f"{runner.events}"
        )
    finally:
        stop.set()
        worker.join(timeout=2.0)


def test_crash_breaker_catches_a_slow_loop_the_rate_limit_cannot_see():
    """A big model's OOM-at-load crashes are ~2 min apart, so the 120s rate
    window empties between them and never trips. That left a mesh retrying a
    hopeless load while the pool said 'driving', and starved the KV auto-fit,
    which only descends once a backend failure is reported."""
    from verallm.mesh.cli import BackendCrashBreaker

    clock = [0.0]
    breaker = BackendCrashBreaker(now=lambda: clock[0])

    # Two slow crashes are not yet a verdict.
    for _ in range(2):
        clock[0] += 130.0
        tripped, _, _ = breaker.note("llama")
        assert not tripped
    clock[0] += 130.0
    tripped, crashes, reason = breaker.note("llama")
    assert tripped and crashes == 3 and reason == "3x in a row"

    # A fast loop trips just as surely, and the message carries the rate.
    fast = BackendCrashBreaker(now=lambda: clock[0])
    for _ in range(BackendCrashBreaker.CONSECUTIVE_LIMIT - 1):
        clock[0] += 1.0
        assert not fast.note("rpc")[0]
    clock[0] += 1.0
    tripped, crashes, reason = fast.note("rpc")
    assert tripped and crashes == BackendCrashBreaker.CONSECUTIVE_LIMIT
    assert "within" in reason

    # A backend that served for hours then died is not a startup failure:
    # its crash starts a new streak instead of tripping on old history.
    healthy = BackendCrashBreaker(now=lambda: clock[0])
    clock[0] += 100.0
    healthy.note("llama")
    clock[0] += 100.0
    healthy.note("llama")
    clock[0] += 10_000.0
    tripped, crashes, _ = healthy.note(
        "llama", uptime_s=BackendCrashBreaker.HEALTHY_UPTIME_S + 1
    )
    assert not tripped and crashes == 1


def test_pool_serving_mode_names_the_miner_side_not_the_audience():
    """A pool is MINER infrastructure: the modes are dev and subnet.

    The mode was originally spelled "validator", which read as if a miner
    were running a validator and repeatedly confused operators. That value
    still loads (pools minted before the rename persist it) but normalizes
    to the canonical name.
    """
    from verallm.mesh.pool import (
        POOL_SERVING_MODE_DEV,
        POOL_SERVING_MODE_SUBNET,
        POOL_SERVING_MODES,
        _normalize_pool_serving_mode,
    )

    assert POOL_SERVING_MODE_SUBNET == "subnet"
    assert POOL_SERVING_MODES == {POOL_SERVING_MODE_DEV, POOL_SERVING_MODE_SUBNET}
    assert _normalize_pool_serving_mode("subnet") == "subnet"
    assert _normalize_pool_serving_mode("dev") == "dev"
    # Legacy pool state and older scripts keep working.
    assert _normalize_pool_serving_mode("validator") == "subnet"
    assert _normalize_pool_serving_mode(" VALIDATOR ") == "subnet"
    for bad in ("", "miner", "prod", None):
        try:
            _normalize_pool_serving_mode(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError as exc:
            assert "dev or subnet" in str(exc)


def test_is_subnet_serving_mode_is_the_single_mode_predicate():
    """UI/flow branches must use the predicate, never raw literals.

    State files persist "subnet" since the rename but pools minted before
    it carry "validator"; comparing either literal directly shipped a board
    that offered to recreate an already-subnet pool. Unknown values are
    simply not subnet mode, never an error (display paths must not blow up
    on a corrupt state file).
    """
    from verallm.mesh.pool import is_subnet_serving_mode

    assert is_subnet_serving_mode("subnet")
    assert is_subnet_serving_mode("validator")
    assert is_subnet_serving_mode(" Subnet ")
    assert not is_subnet_serving_mode("dev")
    assert not is_subnet_serving_mode("")
    assert not is_subnet_serving_mode(None)
    assert not is_subnet_serving_mode("garbage")


def _subnet_manager(tmp_path):
    """A subnet-mode pool manager with chain facts but no registered models."""
    from verallm.mesh.pool import PoolManager, create_pool_state

    shared_state_path = tmp_path / "shared_state.json"
    shared_state_path.write_text("{}", encoding="utf-8")
    state_dir, _ = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="subnet",
        owner_account="5HEDSywHjCaxgdGZ5CFN36LywPbBtaG8juZdKrjcrWWuJnJh",
        coordinator_address="0x0eeba349CB4473c011805603AfD9Aee40a923A59",
        validator_shared_state_path=shared_state_path,
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=123,
    )
    return PoolManager(state_dir)


def test_subnet_register_model_uses_the_measured_context(tmp_path):
    """The registered max_context_len is the MEASURED auto-fit value.

    A miner is scored on context and validators canary at exactly the
    registered maximum, so this number may never be a hand-typed guess: an
    unregistered launch measures what the hardware holds, and registration
    commits that. Registering before any launch has measured it is refused
    with an actionable message rather than accepting a guess.
    """
    manager = _subnet_manager(tmp_path)
    secret = manager.state["management_secret"]
    base = {
        "management_secret": secret,
        "model_id": "glm-5.2-iq2-m",
        "hf_repo": "unsloth/GLM-5.2-GGUF",
        "hf_files": ["UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf"],
        "layers": 79,
        "model_index": 40,
        "quantization_scheme": "gguf_iq2_m",
        "model_package_hash": "aa" * 32,
        "model_tensor_manifest_root": "bb" * 32,
        "tokenizer_hash": "cc" * 32,
    }

    try:
        manager.handle_register_model(dict(base))
        raise AssertionError("expected a refusal without a measured context")
    except ValueError as exc:
        assert "no launch has measured" in str(exc)
        assert "mesh pool launch" in str(exc)

    # An unregistered launch measured what this hardware holds.
    registry = manager.state.setdefault("model_registry", {})
    registry.setdefault("glm-5.2-iq2-m", {})["measured_ctx_budget"] = 389_120

    result = manager.handle_register_model(dict(base))
    assert result["registry"]["max_context_len"] == 389_120

    # Jitter between restarts is tolerated exactly as the vLLM path does.
    from verallm.mesh.pool import MESH_CTX_JITTER_TOLERANCE

    assert MESH_CTX_JITTER_TOLERANCE == 0.10
    within = int(389_120 * 1.05)
    assert manager.handle_register_model(
        {**base, "max_context_len": within}
    )["registry"]["max_context_len"] == within

    try:
        manager.handle_register_model({**base, "max_context_len": 600_000})
        raise AssertionError("expected an over-claim to be refused")
    except ValueError as exc:
        assert "exceeds the measured" in str(exc)


def test_subnet_pool_launches_unregistered_model_for_verification(tmp_path):
    """A subnet pool must launch an UNREGISTERED model so the operator can
    probe it and read its measured context BEFORE committing anything on
    chain. Such a launch signs no snapshot and claims no chain slot, so it
    can never take scored traffic."""
    manager = _subnet_manager(tmp_path)
    registry = manager.state.setdefault("model_registry", {})
    registry["glm-5.2-iq2-m"] = {
        "hf_repo": "unsloth/GLM-5.2-GGUF",
        "hf_files": ["UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf"],
        "layers": 79,
    }
    manager.state["workers"] = {}
    try:
        manager.handle_launch(
            {
                "management_secret": manager.state["management_secret"],
                "model_id": "glm-5.2-iq2-m",
            }
        )
    except ValueError as exc:
        message = str(exc)
        # It must fail for lack of WORKERS, never for lack of chain binding:
        # the unregistered path is legitimate.
        assert "model_index" not in message
        assert "max_context_len" not in message
        assert "quantization_scheme" not in message


def _register_with_binding(manager, model_id="glm-5.2-iq2-m", index=40):
    registry = manager.state.setdefault("model_registry", {})
    registry.setdefault(model_id, {})["measured_ctx_budget"] = 98_304
    manager.handle_register_model(
        {
            "management_secret": manager.state["management_secret"],
            "model_id": model_id,
            "hf_repo": "unsloth/GLM-5.2-GGUF",
            "hf_files": ["UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf"],
            "layers": 79,
            "model_index": index,
            "quantization_scheme": "gguf_iq2_m",
            "model_package_hash": "aa" * 32,
            "model_tensor_manifest_root": "bb" * 32,
            "tokenizer_hash": "cc" * 32,
        }
    )


def test_deploy_measurement_recalibrates_without_mutating_durable_binding(
    tmp_path, monkeypatch
):
    """A replacement measurement is unbound; its final relaunch is not."""

    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    model_id = "glm-5.2-iq2-m"
    manager.state["model_registry"][model_id]["chain_committed"] = True
    manager.state["mesh_registrations"] = {
        model_id: {
            "model_id": model_id,
            "index": 40,
            "mesh_key": "m-durable",
            "mesh_id": "mesh-durable",
            "endpoint": "https://old.example:9443",
            "quant": "gguf_mesh_iq2_m",
            "max_context_len": 98_304,
            "model_spec_ref": "dd" * 32,
            "expires_at": int(time.time()) + 86_400,
            "suspended_by_operator": True,
        }
    }
    manager.state["workers"] = {
        "w-driver": {
            "status": "idle",
            "mesh": "",
            "last_seen_unix": int(time.time()),
            "capability": {
                "gpu_name": "test",
                "vram_gb": 320,
                "free_disk_gb": 1000,
                "subnet_driver_ready": True,
                "rpc_device": "CUDA0",
            },
            "catalog": [
                {
                    "model_id": model_id,
                    "layers": 79,
                    "model_bytes": 1_000_000,
                }
            ],
            "endpoints": {
                "rpc": "w-driver:50052",
                "proof": "http://w-driver:9402",
                "mesh": "http://w-driver:9500",
            },
            "rtt_ms": {},
            "peer_rtt_ms": {},
            "commands": [],
            "command_inflight": None,
        }
    }
    manager._save()
    admin = {"management_secret": manager.state["management_secret"]}
    monkeypatch.setattr(
        manager,
        "_auth_worker",
        lambda body, **_kwargs: (str(body.get("worker_id", "")), ""),
    )

    measured = manager.handle_launch(
        {
            **admin,
            "model_id": model_id,
            "workers": ["w-driver"],
            "driver": "w-driver",
            "_deploy_measurement_unbound": True,
        }
    )
    measured_mesh = manager.state["meshes"][measured["mesh_key"]]
    assert "model_index" not in measured_mesh
    assert "max_context_len" not in measured_mesh
    assert "resume_mesh_id" not in measured_mesh
    durable = manager.state["mesh_registrations"][model_id]
    assert durable["mesh_key"] == "m-durable"
    assert durable["mesh_id"] == "mesh-durable"
    assert durable["suspended_by_operator"] is True

    command = manager.handle_heartbeat({"worker_id": "w-driver"})["command"]
    assert "model_index" not in command
    assert "max_context_len" not in command
    manager.handle_report(
        _command_report(
            command,
            worker_id="w-driver",
            event="drive_ready",
            mesh_id="mesh-measurement",
            join_token="vtmesh_measurement",
            coordinator_endpoint="http://w-driver:9500",
            measured_ctx_budget=262_144,
        )
    )
    assert measured_mesh["measured_ctx_budget"] == 262_144
    assert manager.state["model_registry"][model_id][
        "measured_ctx_budget"
    ] == 262_144
    assert manager.state["mesh_registrations"][model_id]["mesh_id"] == (
        "mesh-durable"
    )

    manager.handle_stop({**admin, "mesh_key": measured["mesh_key"]})
    stop_command = manager.handle_heartbeat({"worker_id": "w-driver"})[
        "command"
    ]
    manager.handle_report(
        _command_report(
            stop_command,
            worker_id="w-driver",
            event="stopped",
        )
    )
    final = manager.handle_launch(
        {
            **admin,
            "model_id": model_id,
            "workers": ["w-driver"],
            "driver": "w-driver",
        }
    )
    final_mesh = manager.state["meshes"][final["mesh_key"]]
    assert final_mesh["model_index"] == 40
    assert final_mesh["max_context_len"] == 98_304
    assert final_mesh["resume_mesh_id"] == "mesh-durable"
    assert "suspended_by_operator" not in manager.state[
        "mesh_registrations"
    ][model_id]


def test_unconfirmed_chain_binding_is_purged_at_launch(tmp_path):
    """An aborted deploy's predicted binding is NOT a chain contract.

    A model that was never registered must be re-measured by every
    deploy; the leftover prediction is purged in code."""
    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    entry = manager.state["model_registry"]["glm-5.2-iq2-m"]
    # A binding written by register-model is a PREDICTION, never trusted.
    assert entry["chain_committed"] is False

    manager.state["workers"] = {}
    try:
        manager.handle_launch(
            {
                "management_secret": manager.state["management_secret"],
                "model_id": "glm-5.2-iq2-m",
            }
        )
    except ValueError as exc:
        # Fails for lack of workers - as an UNREGISTERED launch, never
        # as a chain-bound one riding the stale prediction.
        assert "model_index" not in str(exc)
    entry = manager.state["model_registry"]["glm-5.2-iq2-m"]
    for field_name in (
        "model_index",
        "chain_committed",
        "max_context_len",
        "measured_ctx_budget",
    ):
        assert field_name not in entry
    # The download source survives the purge.
    assert entry["hf_repo"] == "unsloth/GLM-5.2-GGUF"


def test_deploy_dance_may_launch_a_pending_binding(tmp_path):
    """Only the deploy flow (explicit pending_binding_ok) may launch
    against an unconfirmed binding - its snapshot must bind the predicted
    index BEFORE the chain write verifies it."""
    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    manager.state["workers"] = {}
    with pytest.raises(ValueError):
        # Still fails (no workers), but the binding must survive.
        manager.handle_launch(
            {
                "management_secret": manager.state["management_secret"],
                "model_id": "glm-5.2-iq2-m",
                "pending_binding_ok": True,
            }
        )
    entry = manager.state["model_registry"]["glm-5.2-iq2-m"]
    assert entry["model_index"] == 40
    assert entry["chain_committed"] is False


def test_lease_record_confirms_and_heals_a_binding(tmp_path):
    """The lease record only exists after a successful chain write, so a
    matching lease confirms a binding whose flag is absent (legacy) or
    stale-pending (a registered model's re-deploy aborted mid-flight)."""
    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    manager.state["mesh_registrations"] = {
        "glm-5.2-iq2-m": {
            "model_id": "glm-5.2-iq2-m",
            "endpoint": "http://mesh.example:20043",
            "quant": "gguf_mesh_iq2_m",
            "max_context_len": 98_304,
            "model_spec_ref": "dd" * 32,
            "index": 40,
            "mesh_key": "m-prior",
            "expires_at": int(time.time()) + 86_400,
        }
    }
    manager.state["workers"] = {}
    with pytest.raises(ValueError):
        manager.handle_launch(
            {
                "management_secret": manager.state["management_secret"],
                "model_id": "glm-5.2-iq2-m",
            }
        )
    entry = manager.state["model_registry"]["glm-5.2-iq2-m"]
    # Confirmed by the lease, healed in place, nothing purged.
    assert entry["model_index"] == 40
    assert entry["chain_committed"] is True
    assert entry["max_context_len"] == 98_304


def test_registration_clear_unbinds_the_model(tmp_path):
    """`mesh retire` deactivates the chain entry; the stored binding must
    die with it or the next launch serves chain-bound at a DEAD index
    . The next deploy re-measures fresh."""
    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    manager.state["model_registry"]["glm-5.2-iq2-m"]["chain_committed"] = True
    manager.state["mesh_registrations"] = {
        "glm-5.2-iq2-m": {"model_id": "glm-5.2-iq2-m", "index": 40}
    }
    manager.handle_registration_state(
        {
            "management_secret": manager.state["management_secret"],
            "clear": True,
            "model_id": "glm-5.2-iq2-m",
        }
    )
    entry = manager.state["model_registry"]["glm-5.2-iq2-m"]
    for field_name in (
        "model_index",
        "chain_committed",
        "max_context_len",
        "measured_ctx_budget",
    ):
        assert field_name not in entry
    # The download source survives; only the chain binding dies.
    assert entry["hf_repo"] == "unsloth/GLM-5.2-GGUF"


def test_registration_state_write_confirms_the_binding(tmp_path):
    """Deploy's post-chain lease write is the confirmation handshake."""
    manager = _subnet_manager(tmp_path)
    _register_with_binding(manager)
    assert (
        manager.state["model_registry"]["glm-5.2-iq2-m"]["chain_committed"]
        is False
    )
    manager.handle_registration_state(
        {
            "management_secret": manager.state["management_secret"],
            "registration": {
                "model_id": "glm-5.2-iq2-m",
                "endpoint": "http://mesh.example:20043",
                "quant": "gguf_mesh_iq2_m",
                "max_context_len": 98_304,
                "model_spec_ref": "dd" * 32,
                "index": 40,
                "mesh_key": "m-live",
                "expires_at": int(time.time()) + 86_400,
            },
        }
    )
    assert (
        manager.state["model_registry"]["glm-5.2-iq2-m"]["chain_committed"]
        is True
    )


def test_proof_flags_match_the_canonical_snapshot_policy():
    """Sampling rates are not tiers, and they are not local tuning knobs
    either: the signed snapshot pins the canonical policy and the
    coordinator refuses to serve chain-bound when the runtime disagrees.
    PROOF_FLAGS drifted to a 10% decode draw from before the always-on
    light tier and nothing caught it, because only chain-bound launches
    compare against a snapshot. Pin the drive flags to the same constants
    the canonical policy is built from."""
    from neurons.validator import (
        MESH_BASE_PROOF_SAMPLE_BPS,
        MESH_CANARY_DECODE_SAMPLE_BPS,
        MESH_ORGANIC_DECODE_SAMPLE_BPS,
    )
    from verallm.mesh.pool import PROOF_FLAGS

    flags = {
        PROOF_FLAGS[i]: PROOF_FLAGS[i + 1]
        for i in range(len(PROOF_FLAGS) - 1)
        if PROOF_FLAGS[i].startswith("--")
        and not PROOF_FLAGS[i + 1].startswith("--")
    }
    assert int(flags["--proof-sample-bps"]) == MESH_BASE_PROOF_SAMPLE_BPS
    assert int(flags["--decode-audit-bps"]) == MESH_ORGANIC_DECODE_SAMPLE_BPS
    assert int(flags["--decode-audit-bps"]) == MESH_CANARY_DECODE_SAMPLE_BPS


def test_chain_identity_persists_and_flows_into_status(tmp_path):
    """`mesh pool serve` persists the wallet/network it runs with.

    The identity historically lived only in the manager's PM2 argv, so no
    status surface could say which hotkey/network a pool serves as. The
    updater backfills existing pools at manager start; status then carries
    it (plus the stored registration) for the board and dashboard.
    """
    from verallm.mesh.pool import PoolManager

    manager = _subnet_manager(tmp_path)
    manager.update_chain_identity(
        subtensor_network="test",
        wallet_name="test_miner96",
        wallet_hotkey="default",
        coordinator_hotkey_ss58="5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv",
    )
    status = manager.handle_status(
        {"management_secret": manager.state["management_secret"]}
    )
    assert status["subtensor_network"] == "test"
    assert status["wallet_name"] == "test_miner96"
    assert status["wallet_hotkey"] == "default"
    assert status["coordinator_hotkey_ss58"].startswith("5ENhc47")
    assert status["mesh_registration"] == {}
    # Persisted: a fresh manager over the same state dir still knows it.
    reloaded = PoolManager(manager.state_path.parent)
    assert reloaded.state["subtensor_network"] == "test"
    assert reloaded.state["wallet_name"] == "test_miner96"


def test_chain_identity_network_falls_back_to_the_chain_id(tmp_path):
    """Pools started before the identity persist still resolve a network:
    the chain id already sits in validator_binding (945 = test, 964 =
    finney), so metagraph-backed status does not stay dark on them."""
    manager = _subnet_manager(tmp_path)  # chain_id 945, nothing persisted
    assert manager._subtensor_network() == "test"


def test_operator_score_reads_the_validator_binding(tmp_path, monkeypatch):
    """The score endpoint was dead on every wizard-created pool: it read
    state["netuid"]/state["subtensor"], which nothing ever wrote (netuid
    lives in validator_binding). It must resolve the coordinator hotkey's
    UID and report incentive/stake/emission; a successful chain read that
    finds NO UID reports registered=False (deregistration warning), which
    is distinct from an unreachable chain (available=False)."""
    import sys
    import types

    hotkey = "5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv"

    class _Metagraph:
        # Mimics the LIVE bittensor Metagraph shape: NO `trust` attribute
        # (observed AttributeError blanked the whole panel), stake
        # only as the single-letter tensor.
        hotkeys = [hotkey, "5FvtzEPother"]
        coldkeys = ["5Cold1", "5Cold2"]
        T = [0.9, 0.1]
        incentive = [0.0123, 0.001]
        S = [107.0, 1.0]
        emission = [0.5, 0.01]

    seen = {}

    class _Subtensor:
        def __init__(self, network=""):
            seen["network"] = network

        def metagraph(self, netuid):
            seen["netuid"] = netuid
            return _Metagraph()

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _Subtensor
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    manager = _subnet_manager(tmp_path)
    manager.update_chain_identity(
        subtensor_network="test", coordinator_hotkey_ss58=hotkey
    )
    payload = manager.handle_operator_score({})
    assert seen == {"network": "test", "netuid": 405}
    assert payload["available"] is True
    assert payload["registered"] is True
    assert payload["uid"] == 0
    assert payload["incentive"] == 0.0123
    assert payload["emission"] == 0.5
    assert payload["netuid"] == 405 and payload["network"] == "test"

    # Deregistered: chain answered, identity holds no UID. Drop the
    # persisted restart copy too, or the handler correctly serves it
    # stale instead of hitting the chain on the caller's thread.
    _Metagraph.hotkeys = ["5FvtzEPother"]
    manager._score_cache = {}
    manager._score_cache_path().unlink()
    dereg = manager.handle_operator_score({})
    assert dereg["available"] is True
    assert dereg["registered"] is False
    assert dereg["uid"] is None


def test_operator_score_serves_stale_through_rate_limits(
    tmp_path, monkeypatch
):
    """Public subtensor RPCs rate-limit hard (user-confirmed on testnet):
    a metagraph failure must serve the LAST GOOD value marked stale, not
    blank out the board, and must not retry faster than the backoff."""
    import sys
    import types

    hotkey = "5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv"
    fail = {"value": False}

    class _Metagraph:
        hotkeys = [hotkey]
        coldkeys = ["5Cold1"]
        trust = [0.9]
        incentive = [0.0123]
        stake = [107.0]
        emission = [0.5]

    class _Subtensor:
        def __init__(self, network=""):
            if fail["value"]:
                raise RuntimeError("429 too many requests")

        def metagraph(self, netuid):
            return _Metagraph()

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _Subtensor
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    manager = _subnet_manager(tmp_path)
    manager.update_chain_identity(
        subtensor_network="test", coordinator_hotkey_ss58=hotkey
    )
    good = manager.handle_operator_score({})
    assert good["available"] is True and good["uid"] == 0
    # RPC starts refusing; the cache has expired. The stale value is
    # served IMMEDIATELY (no blocking chain read on the caller's thread:
    # board clients only wait ~3s) and the refresh runs in the background.
    fail["value"] = True
    manager._score_cache = {
        k: {**v, "exp": 0} for k, v in manager._score_cache.items()
    }
    stale = manager.handle_operator_score({})
    assert stale["available"] is True
    assert stale["uid"] == 0
    assert stale["stale"] is True
    manager._score_refresh_thread.join(timeout=10)
    # The background failure was cached with a backoff: the next call
    # serves the stale value again without re-hitting the RPC.
    calls = {"n": 0}

    class _Counting(_Subtensor):
        def __init__(self, network=""):
            calls["n"] += 1
            raise RuntimeError("429")

    fake_bt.Subtensor = _Counting
    again = manager.handle_operator_score({})
    assert again["stale"] is True
    assert calls["n"] == 0


def test_operator_score_survives_manager_restart(tmp_path, monkeypatch):
    """A manager restart (pm2 restart, apikey expose, box reboot) used to
    wipe the in-memory score cache: the next board render blocked behind a
    10-30s metagraph read, timed out client-side, and showed "chain status
    unavailable" on every fresh install. The last good value is persisted
    in the pool dir and served immediately (marked stale) by the NEXT
    manager process while a background refresh runs."""
    import sys
    import types

    hotkey = "5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv"

    class _Metagraph:
        hotkeys = [hotkey]
        coldkeys = ["5Cold1"]
        incentive = [0.0123]
        S = [107.0]
        emission = [0.5]

    class _Subtensor:
        def __init__(self, network=""):
            pass

        def metagraph(self, netuid):
            return _Metagraph()

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _Subtensor
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    from verallm.mesh.pool import PoolManager

    manager = _subnet_manager(tmp_path)
    manager.update_chain_identity(
        subtensor_network="test", coordinator_hotkey_ss58=hotkey
    )
    assert manager.handle_operator_score({})["uid"] == 0

    # New process, cold cache, chain now slow/unreachable.
    class _Down(_Subtensor):
        def __init__(self, network=""):
            raise RuntimeError("429 too many requests")

    fake_bt.Subtensor = _Down
    restarted = PoolManager(manager.state_path.parent)
    payload = restarted.handle_operator_score({})
    assert payload["available"] is True
    assert payload["uid"] == 0
    assert payload["stale"] is True
    restarted._score_refresh_thread.join(timeout=10)


def test_operator_score_pending_while_first_fetch_inflight(
    tmp_path, monkeypatch
):
    """While the first-ever chain read runs (normally the startup warmer),
    concurrent callers get an instant ``pending: true`` instead of piling
    onto the RPC or timing out: the board renders "fetching", not
    "unavailable"."""
    manager = _subnet_manager(tmp_path)
    manager.update_chain_identity(
        subtensor_network="test",
        coordinator_hotkey_ss58="5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv",
    )
    account = str(manager.state.get("owner_account", "") or "")
    hotkey = str(manager.state.get("coordinator_hotkey_ss58", "") or "")
    manager._score_refresh_lock = __import__("threading").Lock()
    manager._score_refresh_inflight = {(account, hotkey)}
    payload = manager.handle_operator_score({})
    assert payload["available"] is False
    assert payload["pending"] is True


def test_epoch_follower_queues_rotate_for_stale_serving_meshes(
    tmp_path, monkeypatch
):
    """Validators pin one signed snapshot per scoring epoch, so the manager
    must keep every serving chain-bound mesh signed for the CURRENT epoch.
    The follower reads the chain block, stages the pool binding, and queues
    a driver-side rotate (local re-sign, no chain writes, no relaunch).
    Measurement meshes (no model_index) and non-serving meshes are never
    rotated, and a pending rotate is not duplicated."""
    import sys
    import types

    current_epoch = 21_528

    class _Subtensor:
        def __init__(self, network=""):
            pass

        def get_current_block(self):
            return current_epoch * 360 + 5

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _Subtensor
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    manager = _subnet_manager(tmp_path)
    # Pin the epoch length: this test asserts 360-block math and must not
    # depend on the LIVE hosted subnet-config (which now serves 180).
    manager._epoch_blocks_cache = {"value": 360, "exp": float("inf")}
    manager.update_chain_identity(subtensor_network="test")
    manager.state["workers"]["w-driver"] = {
        "status": "serving",
        "mesh": "m-live",
        "capability": {},
    }
    manager.state["meshes"]["m-live"] = {
        "mesh_key": "m-live",
        "model_id": "glm-5.2-iq2-m",
        "serving_mode": "subnet",
        "model_index": 41,
        "status": "serving",
        "driver": "w-driver",
        "members": ["w-driver"],
        "validator_binding": {**dict(manager.validator_binding), "epoch": 123},
        "snapshot_generation": 1,
    }
    manager.state["meshes"]["m-measure"] = {
        "mesh_key": "m-measure",
        "model_id": "glm-5.2-iq2-m",
        "serving_mode": "subnet",
        "model_index": None,
        "status": "serving",
        "driver": "w-driver",
        "members": ["w-driver"],
    }

    manager.follow_chain_epoch()

    commands = manager.state["workers"]["w-driver"].get("commands") or []
    rotates = [c for c in commands if c.get("action") == "rotate"]
    assert len(rotates) == 1
    assert rotates[0]["mesh_key"] == "m-live"
    assert rotates[0]["epoch"] == current_epoch
    # Pool-level binding staged for future launches too.
    assert int(manager.validator_binding["epoch"]) == current_epoch
    # A second tick must not queue a duplicate while one is pending.
    manager.follow_chain_epoch()
    commands = manager.state["workers"]["w-driver"].get("commands") or []
    assert len([c for c in commands if c.get("action") == "rotate"]) == 1

    # Once the driver reports the rotation, the mesh binding advances and
    # the follower goes quiet for this epoch.
    manager.state["workers"]["w-driver"]["commands"] = []
    mesh_binding = dict(manager.state["meshes"]["m-live"]["validator_binding"])
    mesh_binding["epoch"] = current_epoch
    manager.state["meshes"]["m-live"]["validator_binding"] = mesh_binding
    manager.follow_chain_epoch()
    assert not (manager.state["workers"]["w-driver"].get("commands") or [])


def test_epoch_follower_serves_stale_epoch_through_rpc_failures(
    tmp_path, monkeypatch
):
    """A dead RPC must neither crash the follower nor hot-loop it: the
    last known epoch is served through the backoff window."""
    import sys
    import types

    calls = {"n": 0}

    class _Flaky:
        def __init__(self, network=""):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("429 too many requests")

        def get_current_block(self):
            return 21_530 * 360 + 1

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _Flaky
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)

    manager = _subnet_manager(tmp_path)
    # Pin the epoch length: this test asserts 360-block math and must not
    # depend on the LIVE hosted subnet-config (which now serves 180).
    manager._epoch_blocks_cache = {"value": 360, "exp": float("inf")}
    manager.update_chain_identity(subtensor_network="test")
    assert manager._current_chain_epoch() == 21_530
    # Expire the cache; the next read fails and serves the stale epoch.
    manager._epoch_cache = {**manager._epoch_cache, "exp": 0}
    assert manager._current_chain_epoch() == 21_530


def test_pool_token_carries_optional_manager_cert_pin():
    """Remote join tokens pin the manager's TLS certificate; legacy tokens
    without the field must keep decoding (and encoding) unchanged."""
    pinned = MeshPoolToken(
        pool_id="pool-x",
        manager_endpoint="https://198.51.100.7:20102",
        pool_secret="s3cret",
        manager_ca_sha256="ab" * 32,
    )
    decoded = MeshPoolToken.decode(pinned.encode())
    assert decoded == pinned
    assert decoded.manager_ca_sha256 == "ab" * 32

    legacy = MeshPoolToken(
        pool_id="pool-x",
        manager_endpoint="http://127.0.0.1:9500",
        pool_secret="s3cret",
    )
    decoded_legacy = MeshPoolToken.decode(legacy.encode())
    assert decoded_legacy.manager_ca_sha256 == ""
    # The pin field is omitted entirely when unset (old tokens stay
    # byte-identical across versions).
    import base64 as _b64
    import json as _json

    raw = _json.loads(
        _b64.urlsafe_b64decode(legacy.encode()[len("vtpool_"):])
    )
    assert "manager_ca_sha256" not in raw


def test_worker_source_bundle_builds_and_caches(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "join_pool.sh").write_text("#!/bin/bash\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "big").write_text("x" * 100)
    (repo / "model.gguf").write_text("weights")
    # dist/ is the sanctioned prebuilt-wheel channel; join_pool.sh installs
    # zkllm + the capacity-audit workspace from it. Excluding it left every
    # bundle-installed worker without capacity audits .
    (repo / "dist").mkdir()
    (repo / "dist" / "hot_capacity_workspace_cuda-0.1.0-cp310-cp310-linux_x86_64.whl").write_text("wheel")
    (repo / "dist" / "zkllm-0.1.1-cp310-cp310-linux_x86_64.whl").write_text("wheel")
    (repo / "build").mkdir()
    (repo / "build" / "artifact.o").write_text("obj")

    state_dir, _token = create_pool_state(
        tmp_path / "pool", manager_endpoint="http://mgr:9500", serving_mode="dev"
    )
    import verallm.mesh.onboarding as onboarding

    monkeypatch.setattr(onboarding, "find_repo_root", lambda: repo)
    manager = PoolManager(state_dir)

    bundle = manager.ensure_worker_source_bundle()
    assert bundle.is_file()
    import tarfile

    with tarfile.open(bundle) as tf:
        names = tf.getnames()
    assert any(name.endswith("scripts/join_pool.sh") for name in names)
    assert not any(".git" in name for name in names)
    assert not any(name.endswith(".gguf") for name in names)
    assert any(
        "dist/hot_capacity_workspace_cuda-" in name for name in names
    ), "capacity-audit wheels must ship in the worker bundle"
    assert any("dist/zkllm-" in name for name in names)
    assert not any("build/" in name for name in names)
    # Provenance stamp: workers run the tree AS BUNDLED (dirty edits
    # included); the stamp records what that was (a mid-edit join shipped
    # a broken decode audit with no trace).
    assert any(
        name.endswith("repo/bundle-stamp.json") for name in names
    ), "bundle must carry its provenance stamp"

    # Cached: a second call returns the same file without regenerating.
    first_mtime = bundle.stat().st_mtime
    assert manager.ensure_worker_source_bundle() == bundle
    assert bundle.stat().st_mtime == first_mtime


def test_chain_epoch_read_never_blocks_past_its_deadline(monkeypatch, tmp_path):
    """A reused chain client whose websocket died silently blocks recv
    forever; the epoch read must abandon it at the deadline and drop the
    client so the next tick reconnects (observed: the follower thread
    hung for over an hour, the snapshot never rotated, and nothing was
    logged)."""
    import sys
    import types

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://mgr:9500", serving_mode="dev"
    )
    manager = PoolManager(state_dir)
    manager.state["subtensor_network"] = "test"

    class _HangingSub:
        def get_current_block(self):
            import time as _time

            _time.sleep(3600)

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = lambda network: _HangingSub()
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    monkeypatch.setattr(pool_module, "_call_bounded",
        lambda fn, timeout_s: (_ for _ in ()).throw(TimeoutError("deadline")))

    t0 = time.monotonic()
    result = manager._current_chain_epoch()
    elapsed = time.monotonic() - t0

    assert result is None  # no cache to fall back on
    assert elapsed < 5.0  # never waits out the hang
    assert getattr(manager, "_epoch_subtensor", "unset") is None  # dropped


def test_call_bounded_times_out_and_propagates_errors():
    import time as _time

    with pytest.raises(TimeoutError):
        pool_module._call_bounded(lambda: _time.sleep(30), timeout_s=0.2)
    assert pool_module._call_bounded(lambda: 42, timeout_s=5.0) == 42
    with pytest.raises(ValueError, match="boom"):
        pool_module._call_bounded(
            lambda: (_ for _ in ()).throw(ValueError("boom")), timeout_s=5.0
        )


def test_subnet_pool_accepts_source_only_registration_and_keeps_binding(
    tmp_path,
):
    """The board's launch flow teaches a download source BEFORE anything is
    measured or chain-registered; requiring model_index there broke every
    catalog launch on a subnet pool with HTTP 400 . A
    later source-only update must also never wipe a stored chain binding."""
    from verallm.mesh.pool import PoolManager

    manager = _subnet_manager(tmp_path)
    secret = manager.state["management_secret"]
    anchors = {
        "model_package_hash": "aa" * 32,
        "model_tensor_manifest_root": "bb" * 32,
        "tokenizer_hash": "cc" * 32,
        "quantization_scheme": "q4_k_m",
    }
    source = {
        "management_secret": secret,
        "model_id": "qwen3.6-27b-q4-k-m",
        "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
        "hf_files": ["Qwen3.6-27B-Q4_K_M.gguf"],
        "model_bytes": 16_817_244_384,
        "layers": 64,
        **anchors,
    }
    # Source-only: no model_index, no max_context_len - measurement lane.
    response = manager.handle_register_model(dict(source))
    assert response["status"] == "ok"
    entry = manager.state["model_registry"]["qwen3.6-27b-q4-k-m"]
    assert "model_index" not in entry

    # Full chain-bound registration afterwards (deploy stage 11).
    manager.handle_register_model(
        {
            **source,
            "model_index": 46,
            "measured_ctx_budget": 65_536,
            "max_context_len": 32_768,
        }
    )
    entry = manager.state["model_registry"]["qwen3.6-27b-q4-k-m"]
    assert entry["model_index"] == 46

    # A later source-only update (board re-run) preserves the binding.
    manager.handle_register_model(dict(source))
    entry = manager.state["model_registry"]["qwen3.6-27b-q4-k-m"]
    assert entry["model_index"] == 46
    assert entry["max_context_len"] == 32_768


def test_runner_reaps_previous_generation_process_groups(tmp_path):
    """Serve processes run in their own sessions, so a pm2 restart of the
    worker ORPHANS them: the fresh runner then cannot stop the old mesh
    (_free_own_ports rightly refuses unowned listeners) and the stop
    wedges forever . The persisted proc registry
    proves ownership across generations: construction reaps recorded
    groups, and a recycled pid (pgid mismatch) is never touched."""
    import os
    import signal as signal_module
    import subprocess
    import time as time_module

    from verallm.mesh.private_files import write_owner_only_json

    from pathlib import Path

    config = _local_runner_config(tmp_path)
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    orphan = subprocess.Popen(["sleep", "300"], start_new_session=True)
    bystander = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        write_owner_only_json(
            workdir / "runner-procs.json",
            [
                {"pid": orphan.pid, "pgid": os.getpgid(orphan.pid)},
                # Simulated pid recycling: recorded pgid does not match
                # the live process group -> must NOT be signalled.
                {"pid": bystander.pid, "pgid": os.getpgid(bystander.pid) + 1},
            ],
        )

        runner = LocalMeshRunner(config)
        # The reap runs in pool_worker_loop under the single-daemon flock,
        # never at construction, where a test configuration may point at
        # ports owned by another process. Invoke the daemon-path hook directly.
        runner._reap_previous_generation()

        deadline = time_module.monotonic() + 25.0
        while time_module.monotonic() < deadline:
            if orphan.poll() is not None:
                break
            time_module.sleep(0.2)
        assert orphan.poll() is not None, "orphan group was not reaped"
        assert bystander.poll() is None, "pgid-mismatched pid was killed"
        assert not (workdir / "runner-procs.json").exists()
    finally:
        for proc in (orphan, bystander):
            try:
                os.killpg(os.getpgid(proc.pid), signal_module.SIGKILL)
            except OSError:
                pass


def test_runner_spawn_records_process_group(tmp_path, monkeypatch):
    """Every spawn lands in the registry so the NEXT generation can prove
    ownership of what this one leaves behind."""
    import json as json_module
    from pathlib import Path

    config = _local_runner_config(tmp_path)
    runner = LocalMeshRunner(config)
    proc = runner._spawn(["sleep", "1"], "test-spawn.log")
    try:
        records = json_module.loads(
            (Path(config.workdir) / "runner-procs.json").read_text()
        )
        assert any(r.get("pid") == proc.pid for r in records)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Capacity-audit context delivery: heartbeats hand mesh workers their signed
# roster and ordinal share, and nothing else.
# ---------------------------------------------------------------------------


def _capacity_context_manager(tmp_path, monkeypatch):
    try:
        from bittensor_wallet import Keypair
    except ImportError:  # pragma: no cover - legacy stack
        from substrateinterface import Keypair

    seed = b"\x31" * 32
    keypair = Keypair.create_from_seed(seed.hex())
    state_dir, _tok = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:0",
        serving_mode="subnet",
        owner_account=keypair.ss58_address,
        coordinator_address="0x" + "11" * 20,
        validator_shared_state_path=tmp_path / "shared_state.json",
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=21550,
    )
    mgr = PoolManager(state_dir)
    mgr.state["wallet_name"] = "walletname"
    mgr.state["wallet_hotkey"] = "hotkeyname"
    mgr.state.setdefault("meshes", {})["m-cap"] = {
        "mesh_key": "m-cap",
        "model_id": "glm",
        "members": ["w-gpu", "w-mac"],
        "driver": "w-gpu",
        "model_index": 45,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 21550,
            "snapshot_ttl_seconds": 3600,
        },
    }
    mgr.state.setdefault("mesh_registrations", {})["glm"] = {
        "model_id": "glm",
        "endpoint": "https://198.18.0.221:20001",
        "quant": "gguf_q4_k_xl",
        "max_context_len": 98304,
        "model_spec_ref": "x",
        "index": 45,
        "mesh_key": "m-cap",
        "expires_at": int(time.time()) + 3600,
    }
    workers = mgr.state.setdefault("workers", {})
    workers["w-gpu"] = {
        "mesh": "m-cap",
        "capability": {
            "gpu_name": "NVIDIA A100 80GB PCIe",
            "vram_gb": 160,
            "gpu_names": ["NVIDIA A100 80GB PCIe"] * 2,
            "per_gpu_vram_gb": [80, 80],
            "rpc_device": "CUDA0",
        },
        "endpoints": {"proof": "http://203.0.113.137:20002"},
        "rtt_ms": {},
    }
    workers["w-mac"] = {
        "mesh": "m-cap",
        "capability": {
            "gpu_name": "Apple M1 Max",
            "vram_gb": 64,
            "gpu_names": ["Apple M1 Max"],
            "per_gpu_vram_gb": [64],
            "rpc_device": "metal",
        },
        "endpoints": {},
        "rtt_ms": {},
    }
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_keypair",
        lambda _w, _h: keypair,
    )
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_seed",
        lambda _w, _h, keypair=None: seed,
    )
    monkeypatch.setattr(
        PoolManager,
        "_auth_worker",
        lambda self, body, **kwargs: (str(body.get("worker_id", "")), ""),
    )
    return mgr, seed


def test_heartbeat_delivers_capacity_audit_context(tmp_path, monkeypatch):
    from verallm.chain.wallet import derive_evm_address

    from verallm.mesh.capacity_roster import (
        roster_cuda_gpus,
        verify_roster_signature,
    )

    mgr, seed = _capacity_context_manager(tmp_path, monkeypatch)
    beat = mgr.handle_heartbeat({"worker_id": "w-gpu", "status": "serving"})
    context = beat["capacity_audit_context"]
    evm_address = derive_evm_address(seed).lower()

    assert context["mesh_key"] == "m-cap"
    slot = context["slot"]
    assert slot["address"] == evm_address
    assert slot["model_index"] == 45
    assert slot["endpoint"] == "https://198.18.0.221:20001"
    assert slot["quant"] == "gguf_q4_k_xl"
    assert slot["max_context_len"] == 98304

    roster = context["roster"]
    assert verify_roster_signature(
        roster, context["roster_signature"], evm_address
    )
    assert roster["roster_epoch"] == 21550
    backends = {
        row["worker_id"]: row["backend"] for row in roster["workers"]
    }
    assert backends == {"w-gpu": "cuda", "w-mac": "metal"}
    gpus = roster_cuda_gpus(roster)
    assert [gpu.ordinal for gpu in gpus] == [0, 1]
    assert all(gpu.worker_id == "w-gpu" for gpu in gpus)

    # The CUDA worker gets exactly its own ordinal share; the Metal member
    # gets the context (it may drive) but zero local openings.
    assert [g["ordinal"] for g in context["local_gpus"]] == [0, 1]
    mac_beat = mgr.handle_heartbeat({"worker_id": "w-mac", "status": "serving"})
    assert mac_beat["capacity_audit_context"]["local_gpus"] == []


def test_heartbeat_context_requires_confirmed_chain_binding(
    tmp_path, monkeypatch
):
    mgr, _seed = _capacity_context_manager(tmp_path, monkeypatch)

    # Lease index mismatch: the stored registration is not this binding's
    # confirmation, so no audit context may be derived from it.
    mgr.state["mesh_registrations"]["glm"]["index"] = 44
    beat = mgr.handle_heartbeat({"worker_id": "w-gpu", "status": "serving"})
    assert "capacity_audit_context" not in beat

    mgr.state["mesh_registrations"]["glm"]["index"] = 45
    del mgr.state["meshes"]["m-cap"]["model_index"]
    beat = mgr.handle_heartbeat({"worker_id": "w-gpu", "status": "serving"})
    assert "capacity_audit_context" not in beat

    # An idle worker bound to no mesh never sees a context.
    mgr.state["workers"]["w-idle"] = {"capability": {}, "rtt_ms": {}}
    beat = mgr.handle_heartbeat({"worker_id": "w-idle", "status": "idle"})
    assert "capacity_audit_context" not in beat


def test_heartbeat_context_requires_registered_mesh_key(
    tmp_path, monkeypatch
):
    """A MEASUREMENT mesh for an already-registered model must not arm
    audits: it carries the registered model_index, but only the mesh the
    stored registration names is the chain contract .
    """

    mgr, _seed = _capacity_context_manager(tmp_path, monkeypatch)

    # Same model, same index, DIFFERENT mesh (a measurement mesh) — the
    # registration still names m-cap, so m-measure gets no context.
    mgr.state["meshes"]["m-measure"] = dict(
        mgr.state["meshes"]["m-cap"], mesh_key="m-measure"
    )
    mgr.state["workers"]["w-gpu"]["mesh"] = "m-measure"
    beat = mgr.handle_heartbeat({"worker_id": "w-gpu", "status": "serving"})
    assert "capacity_audit_context" not in beat

    # Back on the registered mesh the context returns.
    mgr.state["workers"]["w-gpu"]["mesh"] = "m-cap"
    beat = mgr.handle_heartbeat({"worker_id": "w-gpu", "status": "serving"})
    assert beat["capacity_audit_context"]["mesh_key"] == "m-cap"


def test_tls_listener_survives_a_silent_client(tmp_path):
    """A client that connects to the TLS API listener and sends NOTHING
    must cost one handler thread, never the accept loop: with the default
    handshake-in-accept it wedged the whole public manager port for 9+
    hours , killing every remote worker's heartbeats and
    delegated signing while local loopback traffic stayed healthy."""

    import socket as socket_module
    import ssl as ssl_module
    import urllib.request

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    probe = socket_module.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    server = serve_pool_manager(
        state_dir, host="127.0.0.1", port=0, api_tls_port=free_port
    )
    api_server = server.api_tls_server
    assert api_server is not None
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True),
        threading.Thread(target=api_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    tls_host, tls_port = api_server.server_address
    silent = socket_module.create_connection((tls_host, tls_port), timeout=5)
    try:
        # The silent connection is open and has sent zero TLS bytes. A
        # well-behaved TLS request must still complete promptly.
        context = ssl_module.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl_module.CERT_NONE
        # ANY completed HTTP response proves the listener is alive; this
        # unauthenticated body earns a 403, which is exactly enough.
        try:
            response = urllib.request.urlopen(
                f"https://{tls_host}:{tls_port}/v1/pool/status",
                data=b"{}",
                timeout=10,
                context=context,
            )
            status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status in (200, 403)
    finally:
        silent.close()
        server.shutdown()
        server.server_close()
        api_server.shutdown()
        api_server.server_close()


def test_management_client_trusts_local_pool_api_tls_cert(tmp_path):
    """Management calls run ON the coordinator box and must trust the pool's
    own self-signed api-tls cert automatically. Management tokens carry no CA
    pin, so before the local-cert trust every management call to the api-tls
    port died 'CERTIFICATE_VERIFY_FAILED: self-signed certificate' even though
    the trust root was a file in the pool dir ."""

    import argparse
    import json as json_module
    import socket as socket_module
    import urllib.request

    from verallm.mesh.worker import post_json

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    probe = socket_module.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    server = serve_pool_manager(
        state_dir, host="127.0.0.1", port=0, api_tls_port=free_port
    )
    api_server = server.api_tls_server
    assert api_server is not None
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True),
        threading.Thread(target=api_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    tls_host, tls_port = api_server.server_address
    endpoint = f"https://{tls_host}:{tls_port}"
    assert (state_dir / "api-tls-cert.pem").is_file()
    state = json_module.loads((state_dir / "pool-state.json").read_text())
    secret = state.get("management_secret") or state.get("pool_secret")

    # A management token points at the pool dir, and _resolve_pool_dir must
    # find it so the local cert can be trusted.
    args = argparse.Namespace(pool=str(state_dir))
    assert mesh_cli._resolve_pool_dir(args) == state_dir

    try:
        # Baseline: default PKI trust rejects the self-signed cert.
        urllib.request.install_opener(urllib.request.build_opener())
        with pytest.raises(RuntimeError) as baseline:
            post_json(
                endpoint + "/v1/pool/status",
                {"management_secret": secret},
                timeout=10,
            )
        assert "certificate verify" in str(baseline.value).lower()

        # After trusting the pool's local cert, the same call verifies and
        # authenticates end to end.
        mesh_cli._install_pool_api_tls_trust(endpoint, state_dir)
        response = post_json(
            endpoint + "/v1/pool/status",
            {"management_secret": secret},
            timeout=10,
        )
        assert response.get("status") == "ok"
        assert response.get("pool_id")
    finally:
        urllib.request.install_opener(urllib.request.build_opener())
        server.shutdown()
        server.server_close()
        api_server.shutdown()
        api_server.server_close()


def test_shipped_fetch_spec_carries_manifest_root_and_store_urls():
    """A shipped-catalogue launch spec must let the driver DOWNLOAD the
    owner-built tensor manifest: without the root + store URLs the driver
    silently rebuilt the manifest locally — always a broken fetch path and
    potentially a hard wedge when the native hasher cannot complete."""

    from verallm.mesh.manifest_store import all_default_store_urls
    from verallm.mesh.pool import _shipped_model_fetch_spec

    spec = _shipped_model_fetch_spec("qwen3.5-9b-q4-k-xl", chain_id=945)
    assert spec is not None
    assert spec["model_tensor_manifest_root"], "catalogue root must ship"
    assert spec["manifest_urls"] == ["https://verathos.ai/gleipnir/testnet"]

    # No chain binding (dev pool): root and the safe union of content-addressed
    # owner stores still ship, so the fetching box never rebuilds locally.
    dev = _shipped_model_fetch_spec("qwen3.5-9b-q4-k-xl", chain_id=None)
    assert dev is not None
    assert dev["model_tensor_manifest_root"]
    assert dev["manifest_urls"] == list(all_default_store_urls())

    assert _shipped_model_fetch_spec("no-such-model", chain_id=945) is None


def test_manager_epoch_blocks_follow_hosted_config(monkeypatch, tmp_path):
    """Snapshot epoch numbering must follow the AUTHORITATIVE hosted
    subnet-config, not the hardcoded 360 default: when testnet moved to
    180-block epochs the manager kept signing 360-numbered snapshots and
    every canary rejected the mesh ("snapshot still binds the previous
    epoch")."""

    import io
    import urllib.request as _urllib_request

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    manager = PoolManager(state_dir)

    # Dev pool (no chain binding): fallback stays 360, nothing fetched.
    assert manager._epoch_blocks() == 360

    # Chain-bound: the chain's gleipnir store serves epoch_blocks=180.
    manager.validator_binding = {"chain_id": 945}
    manager._epoch_blocks_cache = None
    fetched_urls = []

    def fake_urlopen(url, timeout=0):
        fetched_urls.append(url)
        return io.BytesIO(b'{"epoch": {"epoch_blocks": 180}}')

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    assert manager._epoch_blocks() == 180
    assert fetched_urls == [
        "https://verathos.ai/gleipnir/testnet/subnet-config.json"
    ]

    # Cached: no second fetch inside the TTL.
    assert manager._epoch_blocks() == 180
    assert len(fetched_urls) == 1

    # Fetch failure after cache expiry: falls back to 360, never raises.
    manager._epoch_blocks_cache = None

    def broken_urlopen(url, timeout=0):
        raise OSError("store down")

    monkeypatch.setattr(_urllib_request, "urlopen", broken_urlopen)
    assert manager._epoch_blocks() == 360


def test_subnet_launch_refuses_manifest_rebuild(monkeypatch, tmp_path):
    """A subnet launch whose driver must download the model REFUSES to
    start when no owner-built manifest root is known, and the driver
    refuses the local rebuild even if a stale command slips through.
    Manifests are owner-built and store-published; a rebuild on a miner
    box is always a broken fetch path ."""

    import pytest as _pytest

    manager = _subnet_manager(tmp_path)
    manager.state["workers"]["w-driver"] = {
        "status": "idle",
        "last_seen_unix": int(time.time()),
        "capability": {
            "vram_gb": 320,
            "free_disk_gb": 1000,
            "subnet_driver_ready": True,
        },
        "catalog": [],
        "endpoints": {"proof": "http://d:9402"},
        "rtt_ms": {},
        "peer_rtt_ms": {},
        "commands": [],
        "command_inflight": None,
    }
    # Rootless download source: refuse at launch, before any fetch starts.
    manager.state.setdefault("model_registry", {})["rootless-70b"] = {
        "hf_repo": "org/rootless-70b",
        "hf_files": ["m.gguf"],
        "layers": 80,
        "model_bytes": 1_000_000_000,
    }
    with _pytest.raises(ValueError, match="refusing subnet launch"):
        manager.handle_launch(
            {
                "management_secret": manager.state["management_secret"],
                "model_id": "rootless-70b",
                "workers": ["w-driver"],
                "driver": "w-driver",
                "pending_binding_ok": True,
            }
        )

    # With a root the same launch passes the gate and stamps the driver
    # requirement so the worker side can never fall back to a rebuild.
    manager.state["model_registry"]["rooted-70b"] = {
        "hf_repo": "org/rooted-70b",
        "hf_files": ["m.gguf"],
        "layers": 80,
        "model_bytes": 1_000_000_000,
        "model_tensor_manifest_root": "ab" * 32,
        "manifest_urls": ["https://store.example/testnet"],
    }
    result = manager.handle_launch(
        {
            "management_secret": manager.state["management_secret"],
            "model_id": "rooted-70b",
            "workers": ["w-driver"],
            "driver": "w-driver",
            "pending_binding_ok": True,
        }
    )
    assert result.get("mesh_key")
    queued = manager.state["workers"]["w-driver"]["command_inflight"] or (
        manager.state["workers"]["w-driver"]["commands"] or [{}]
    )[0]
    assert queued.get("action") == "fetch"
    assert queued["spec"]["require_published_manifest"] is True


def test_subnet_join_refuses_private_advertise(tmp_path):
    """PRODUCTION JOINS ONLY: a driver-capable worker joining a subnet
    pool with a container/LAN advertise (auto-detected 172.x on rented
    docker boxes) is refused AT JOIN with the fix
    in the message - never admitted to fail validators later. LAN slice
    workers stay legitimate via member_only."""

    import pytest as _pytest

    manager = _subnet_manager(tmp_path)
    # The gate under test sits AFTER worker auth; a real signed control
    # envelope is exercised elsewhere.
    manager._auth_worker = lambda body, **kw: (
        str(body.get("worker_id", "")),
        "",
    )

    def _join_body(worker_id, host, member_only=False):
        return {
            "worker_id": worker_id,
            "capability": {
                "vram_gb": 31,
                "stage_proof_key": "",
                "member_only": member_only,
            },
            "endpoints": {
                "mesh": f"http://{host}:9443",
                "proof": f"http://{host}:9402",
                "rpc": f"{host}:50052",
            },
        }

    with _pytest.raises(PermissionError, match="no validator can dial"):
        manager.handle_join(_join_body("w-docker", "172.20.0.3"))

    # member-only LAN slice and public advertise: pass the gate (any
    # later failure must not be the dial refusal).
    for wid, host, member in (
        ("w-slice", "192.168.1.50", True),
        ("w-public", "192.0.0.9", False),
    ):
        try:
            manager.handle_join(_join_body(wid, host, member))
        except PermissionError as exc:
            assert "no validator can dial" not in str(exc)


def test_self_audit_skippable_classification():
    """Contention never strikes the mesh; verification failures do. A
    busy 503 or an exclusive replay held by a real canary is normal
    concurrency - counting it toward mesh failure would relaunch healthy
    meshes under load."""
    from verallm.mesh.pool import self_audit_skippable

    for skip in (
        "slots busy",
        "HTTP 503 from http://127.0.0.1:20102/v1/chat/completions",
        "exclusive replay window held",
        "local coordinator snapshot no longer matches the manager-pinned mesh snapshot",
    ):
        assert self_audit_skippable(skip), skip
    for strike in (
        "decode audit verification failed: decode audit token is not the committed f32 argmax",
        "self-audit reply did not verify",
        "no GGML proof trace verified",
        "timed out",
    ):
        assert not self_audit_skippable(strike), strike


def test_reap_port_squatters_kills_own_port_leftovers(tmp_path):
    """A previous generation's grandchild (llama in its own session, PPID 1)
    survives the group reaper but holds this worker's port; at construction
    any listener on this worker's configured ports is a leftover and is
    reaped."""

    import socket
    import subprocess
    import sys as _sys
    import time as _time
    from pathlib import Path

    _state_dir, token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:9", serving_mode="dev"
    )
    # An ephemeral port we then configure as the worker's mesh port.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    squatter = subprocess.Popen(
        [
            _sys.executable,
            "-c",
            (
                "import socket, time\n"
                f"s = socket.socket()\n"
                f"s.bind((\"127.0.0.1\", {port}))\n"
                "s.listen()\n"
                "print(\"listening\", flush=True)\n"
                "time.sleep(60)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert squatter.stdout.readline().strip() == "listening"
        config = _worker_config(token, tmp_path, "reaper")
        config = PoolWorkerConfig(
            **{
                **{f: getattr(config, f) for f in config.__dataclass_fields__},
                "mesh_port": port,
            }
        )
        Path(config.workdir).mkdir(parents=True, exist_ok=True)
        runner = LocalMeshRunner(config)
        # The sweep runs ONLY on the daemon path now (construction-time
        # sweeping let TEST runners kill production listeners); invoke it
        # the way pool_worker_loop does.
        runner._reap_previous_generation()
        deadline = _time.time() + 10.0
        while _time.time() < deadline and squatter.poll() is None:
            _time.sleep(0.2)
        assert squatter.poll() is not None
    finally:
        try:
            squatter.kill()
        except Exception:
            pass


def test_runner_reap_never_kills_its_own_process_group(tmp_path):
    """pm2 fork-mode restarts can hand the fresh daemon the SAME process
    group the previous generation recorded; killpg on that group is then
    silent suicide .
    The reap must skip its own group and rely on the port sweep."""
    import os
    import subprocess
    from pathlib import Path

    from verallm.mesh.private_files import write_owner_only_json

    config = _local_runner_config(tmp_path)
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Same process group as this test (no start_new_session).
    peer = subprocess.Popen(["sleep", "300"])
    try:
        write_owner_only_json(
            workdir / "runner-procs.json",
            [{"pid": peer.pid, "pgid": os.getpgid(0)}],
        )
        runner = LocalMeshRunner(config)
        runner._reap_previous_generation()
        # If the guard is missing, killpg(own group) killed this very
        # test process - reaching these asserts IS the proof, and the
        # same-group peer must be untouched.
        assert peer.poll() is None, "own-group peer was killed"
        assert not (workdir / "runner-procs.json").exists()
    finally:
        try:
            peer.kill()
        except OSError:
            pass


def test_manager_startup_warns_about_orphaned_deploy(tmp_path, caplog):
    """A deploy marker surviving into a new manager process means the
    manager restarted mid-deploy — the deploy client is orphaned and the
    operator must re-run it ."""

    import json as _json
    import logging
    import time as _time

    state_dir, _tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    state_path = state_dir / "pool-state.json"
    state = _json.loads(state_path.read_text())
    state["deploys"] = {
        "glm-5.2-iq2-m": {
            "stage": "probe-gate",
            "updated_at_unix": int(_time.time()) - 30,
        },
        # Expired markers stay silent.
        "old-model": {
            "stage": "measure",
            "updated_at_unix": int(_time.time()) - 100_000,
        },
    }
    state_path.write_text(_json.dumps(state))

    with caplog.at_level(logging.WARNING, logger="verallm.mesh.pool"):
        PoolManager(state_dir)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "glm-5.2-iq2-m" in joined and "orphaned" in joined
    assert "old-model" not in joined


def test_operator_tuning_fields_survive_reregistration(tmp_path):
    """Per-model llama batching and proof tolerances are operator-pinned:
    they persist through post-measure re-registration and feed the launch
    command even when the registration payload omits them."""

    from verallm.mesh.pool import _llama_batch_args

    manager = _subnet_manager(tmp_path)
    secret = manager.state["management_secret"]
    registry = manager.state.setdefault("model_registry", {})
    registry.setdefault("glm-5.2-iq2-m", {})["measured_ctx_budget"] = 389_120
    base = {
        "management_secret": secret,
        "model_id": "glm-5.2-iq2-m",
        "hf_repo": "unsloth/GLM-5.2-GGUF",
        "hf_files": ["UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf"],
        "layers": 79,
        "model_index": 40,
        "quantization_scheme": "gguf_iq2_m",
        "model_package_hash": "aa" * 32,
        "model_tensor_manifest_root": "bb" * 32,
        "tokenizer_hash": "cc" * 32,
    }

    first = manager.handle_register_model(
        {
            **base,
            "llama_ubatch": 1024,
            "proof_tolerance_abs": 0.2,
            "proof_tolerance_rel": 0.08,
        }
    )["registry"]
    assert first["llama_ubatch"] == 1024
    assert first["proof_tolerance_abs"] == 0.2

    # The deploy re-registers after measurement with no tuning fields.
    second = manager.handle_register_model(dict(base))["registry"]
    assert second["llama_ubatch"] == 1024
    assert second["proof_tolerance_abs"] == 0.2
    assert second["proof_tolerance_rel"] == 0.08

    # An explicit new value still overrides.
    third = manager.handle_register_model(
        {**base, "llama_ubatch": 512}
    )["registry"]
    assert third["llama_ubatch"] == 512

    # The launch command derives -b/-ub from the stored fields.
    args = _llama_batch_args(third)
    assert args == [
        "--llama-extra-arg=-b",
        "--llama-extra-arg=1024",
        "--llama-extra-arg=-ub",
        "--llama-extra-arg=512",
    ]
    # Defaults when a model pins nothing.
    assert _llama_batch_args({}) == [
        "--llama-extra-arg=-b",
        "--llama-extra-arg=8192",
        "--llama-extra-arg=-ub",
        "--llama-extra-arg=4096",
    ]

    # Range validation refuses nonsense at registration time.
    try:
        manager.handle_register_model({**base, "llama_ubatch": 64})
        raise AssertionError("expected llama_ubatch below 128 to be refused")
    except ValueError as exc:
        assert "llama_ubatch" in str(exc)
    try:
        manager.handle_register_model(
            {**base, "llama_ubatch": 1024, "llama_batch": 512}
        )
        raise AssertionError("expected llama_batch below ubatch to be refused")
    except ValueError as exc:
        assert "llama_batch" in str(exc)


def test_first_batch_deadline_scales_with_model_bytes(monkeypatch):
    monkeypatch.delenv("VERATHOS_MESH_FIRST_BATCH_DEADLINE_S", raising=False)
    # Small models keep the historical floor.
    assert pool_module._first_batch_deadline_s(0) == 600.0
    assert pool_module._first_batch_deadline_s(6_000_000_000) == 600.0
    # A 120GB glm at the observed ~50 MB/s contended floor needs 2400s -
    # the value previously hand-injected per box via the env knob.
    assert pool_module._first_batch_deadline_s(120_000_000_000) == 2400.0
    # Pathological sizes clamp so a hung backend cannot hold the launch
    # slot for an unbounded time.
    assert pool_module._first_batch_deadline_s(10**13) == 7200.0
    # Env override is per-box break-glass and wins outright.
    monkeypatch.setenv("VERATHOS_MESH_FIRST_BATCH_DEADLINE_S", "900")
    assert pool_module._first_batch_deadline_s(120_000_000_000) == 900.0
    # Malformed env falls back to the derived value instead of crashing
    # the launch path.
    monkeypatch.setenv("VERATHOS_MESH_FIRST_BATCH_DEADLINE_S", "soon")
    assert pool_module._first_batch_deadline_s(120_000_000_000) == 2400.0


def test_report_for_deleted_mesh_orders_teardown(tmp_path):
    """A report whose mesh record no longer exists carries mesh_gone: the
    worker's spawn serves something no pool state references, and without
    the explicit teardown order it lingers unsupervised on the GPU."""

    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    with mgr.lock:
        del mgr.state["meshes"][mesh_key]
        mgr._save()

    response = mgr.handle_report(
        {
            **worker_body,
            **_command_report(
                delivered,
                worker_id="w1",
                event="drive_ready",
                mesh_id="mesh-gone",
                join_token="vtmesh_gone",
                coordinator_endpoint="http://w1:9500",
            ),
        }
    )
    assert response["status"] == "stale"
    assert response["mesh_gone"] is True


def test_report_for_terminal_mesh_orders_teardown_not_resurrection(tmp_path):
    """A drive that completes AFTER its mesh was failed (silent-driver reap)
    or torn down must not resurrect the record. Observed live: the drive
    finished into a serve holding ~20GB VRAM for a mesh the manager had
    already failed. The worker is ordered to tear down instead, and the
    record keeps its terminal status."""

    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    report = {
        **worker_body,
        **_command_report(
            delivered,
            worker_id="w1",
            event="drive_ready",
            mesh_id="mesh-terminal-status",
            join_token="vtmesh_terminal_status",
            coordinator_endpoint="http://w1:9500",
        ),
    }

    for terminal_status in ("error", "stopping", "stopped"):
        with mgr.lock:
            mgr.state["meshes"][mesh_key]["status"] = terminal_status
            mgr._save()
        response = mgr.handle_report(dict(report))
        assert response["status"] == "stale", terminal_status
        assert response["mesh_gone"] is True, terminal_status
        assert (
            mgr.state["meshes"][mesh_key]["status"] == terminal_status
        ), "terminal mesh status must never be resurrected by a late report"


def test_benign_stale_races_carry_no_teardown_order(tmp_path):
    """Command-identity races (cross-worker report, superseded command)
    return bare stale WITHOUT mesh_gone — tearing down a serve over a
    duplicate-delivery race would kill healthy meshes."""

    _state_dir, _tok, mgr, worker_body, mesh_key = (
        _direct_manager_with_drive_command(tmp_path)
    )
    delivered = mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})[
        "command"
    ]
    report = {
        **worker_body,
        **_command_report(
            delivered,
            worker_id="w1",
            event="drive_ready",
            mesh_id="mesh-benign-stale",
            join_token="vtmesh_benign_stale",
            coordinator_endpoint="http://w1:9500",
        ),
    }

    mgr.handle_join(
        {
            **worker_body,
            "worker_id": "w2",
            "capability": {"gpu_name": "t", "vram_gb": 48},
            "catalog": [{"model_id": "m1", "model_bytes": 1000}],
            "endpoints": {
                "rpc": "w2:50052",
                "proof": "http://w2:9402",
                "mesh": "http://w2:9500",
            },
        }
    )
    cross_worker = mgr.handle_report({**report, "worker_id": "w2"})
    assert cross_worker["status"] == "stale"
    assert "mesh_gone" not in cross_worker

    with mgr.lock:
        mgr._queue_worker_command(
            mgr.state["workers"]["w1"],
            {"action": "stop", "mesh_key": mesh_key},
        )
        mgr._save()
    superseded = mgr.handle_report(report)
    assert superseded["status"] == "stale"
    assert "mesh_gone" not in superseded


def _manager_with_registration_and_idle_worker(tmp_path):
    state_dir, tok = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    mgr = PoolManager(state_dir)
    mgr.AUTO_RELAUNCH_DELAY_S = 0.01
    worker_body = {"pool_secret": tok.pool_secret}
    mgr.handle_join({**worker_body, "worker_id": "w1",
                     "capability": {"gpu_name": "t", "vram_gb": 48},
                     "catalog": [{"model_id": "m1", "model_bytes": 1000}],
                     "endpoints": {"rpc": "w1:50052", "proof": "http://w1:9402",
                                   "mesh": "http://w1:9500"}})
    with mgr.lock:
        mgr.state.setdefault("mesh_registrations", {})["m1"] = {
            "model_id": "m1",
            "index": 7,
            "mesh_key": "m-dead",
            "endpoint": "https://mesh.example:9443",
            "quant": "gguf_mesh_q4_k_m",
            "max_context_len": 8192,
            "model_spec_ref": "aa" * 32,
            "expires_at": int(time.time()) + 86_400,
        }
        mgr._save()
    return mgr, worker_body


def _add_idle_worker(mgr, worker_body, worker_id):
    mgr.handle_join(
        {
            **worker_body,
            "worker_id": worker_id,
            "capability": {"gpu_name": "t", "vram_gb": 48},
            "catalog": [{"model_id": "m1", "model_bytes": 1000}],
            "endpoints": {
                "rpc": f"{worker_id}:50052",
                "proof": f"http://{worker_id}:9402",
                "mesh": f"http://{worker_id}:9500",
            },
        }
    )


def _add_detached_error_mesh(mgr):
    with mgr.lock:
        mgr.state["meshes"]["m-dead"] = {
            "mesh_key": "m-dead",
            "model_id": "m1",
            "model_index": 7,
            "members": ["w1"],
            "driver": "w1",
            "status": "error",
            "error": "worker runtime failed",
        }
        mgr._save()


def _wait_for_live_mesh(mgr, model_id, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with mgr.lock:
            live = [
                m for m in mgr.state["meshes"].values()
                if m.get("model_id") == model_id
                and m.get("status") in ("fetching", "driving")
            ]
        if live:
            return live[0]
        time.sleep(0.1)
    return None


class TestAutoRelaunch:
    """A registered model whose mesh died relaunches by itself when an
    idle worker holding the model heartbeats in. Operator stops suspend
    it; an explicit launch re-arms it; a cooldown bounds crash loops."""

    def test_idle_worker_triggers_relaunch(self, tmp_path):
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        mesh = _wait_for_live_mesh(mgr, "m1")
        assert mesh is not None, "auto-relaunch must launch the dead model"
        assert mesh["driver"] == "w1"

    def test_detached_error_tombstone_is_retired_before_relaunch(self, tmp_path):
        """The real crash path leaves an error mesh record behind after its
        workers detach. Auto-relaunch must retire that terminal record before
        creating the replacement, or subnet index-conflict checks reject the
        launch forever."""
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        _add_detached_error_mesh(mgr)

        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})

        mesh = _wait_for_live_mesh(mgr, "m1")
        assert mesh is not None
        with mgr.lock:
            assert "m-dead" not in mgr.state["meshes"]

    def test_relaunch_uses_recommended_multi_worker_placement(self, tmp_path):
        """An idle catalog-holding worker is only the recovery trigger. The
        model may need several machines, so placement must still pass through
        the manager's feasible-set recommender."""
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        _add_idle_worker(mgr, worker_body, "w2")
        calls = []

        def recommend(model_id):
            calls.append(model_id)
            return ([{"workers": ["w1", "w2"], "driver": "w1"}], [])

        mgr.recommend = recommend
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})

        mesh = _wait_for_live_mesh(mgr, "m1")
        assert mesh is not None
        assert calls == ["m1"]
        assert mesh["members"] == ["w1", "w2"]

    def test_relaunch_prefers_worker_behind_registered_endpoint(self, tmp_path):
        """A brief driver restart must not move a registration to another
        idle same-model worker merely because the recommender lists it first.

        The registered endpoint identifies the returning driver.  Placement
        still comes from the normal recommender, so a multi-worker mesh keeps
        every required member while selecting the returning driver.
        """

        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        _add_idle_worker(mgr, worker_body, "w2")
        with mgr.lock:
            mgr.state["mesh_registrations"]["m1"]["endpoint"] = (
                "http://w2:9500"
            )
            mgr._save()

        calls = []

        def recommend(model_id):
            calls.append(model_id)
            return (
                [
                    {"workers": ["w1"], "driver": "w1"},
                    {"workers": ["w2", "w1"], "driver": "w2"},
                ],
                [],
            )

        mgr.recommend = recommend
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})

        mesh = _wait_for_live_mesh(mgr, "m1")
        assert mesh is not None
        assert calls == ["m1"]
        assert mesh["driver"] == "w2"
        assert mesh["members"] == ["w2", "w1"]

    def test_delayed_relaunch_cannot_override_operator_stop(self, tmp_path):
        """Suspension may race the delayed relaunch thread. The launch must
        re-check operator intent under the same lock that creates the mesh."""
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        mgr.AUTO_RELAUNCH_DELAY_S = 0.2
        _add_detached_error_mesh(mgr)
        admin = {"management_secret": mgr.state["management_secret"]}

        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        mgr.handle_stop({**admin, "mesh_key": "m-dead"})
        time.sleep(0.4)

        assert _wait_for_live_mesh(mgr, "m1", timeout_s=0.3) is None
        with mgr.lock:
            assert mgr._registrations_locked()["m1"][
                "suspended_by_operator"
            ] is True

    def test_operator_stop_suspends_until_explicit_launch(self, tmp_path):
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        mesh = _wait_for_live_mesh(mgr, "m1")
        assert mesh is not None
        admin = {"management_secret": mgr.state["management_secret"]}
        mgr.handle_stop({**admin, "mesh_key": mesh["mesh_key"]})
        with mgr.lock:
            assert mgr._registrations_locked()["m1"][
                "suspended_by_operator"
            ] is True
        # Cooldown reset so ONLY the suspension can block the relaunch.
        mgr._auto_relaunch_last.clear()
        # Complete the stop so the worker idles again.
        stop_cmd = mgr.handle_heartbeat(
            {**worker_body, "worker_id": "w1"}
        )["command"]
        assert stop_cmd["action"] == "stop"
        mgr.handle_report(
            {
                **worker_body,
                **_command_report(stop_cmd, worker_id="w1", event="stopped"),
            }
        )
        mgr._auto_relaunch_last.clear()
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        assert _wait_for_live_mesh(mgr, "m1", timeout_s=0.3) is None, (
            "operator stop must suspend auto-relaunch"
        )
        # An explicit launch re-arms the guard.
        launched = mgr.handle_launch(
            {**admin, "model_id": "m1", "workers": ["w1"], "driver": "w1"}
        )
        with mgr.lock:
            assert "suspended_by_operator" not in mgr._registrations_locked()[
                "m1"
            ]
        assert launched["driver"] == "w1"

    def test_cooldown_bounds_crash_loops(self, tmp_path):
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        mgr._auto_relaunch_last = {"m1": time.monotonic()}
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        assert _wait_for_live_mesh(mgr, "m1", timeout_s=0.3) is None, (
            "a recent attempt must gate the next one"
        )

    def test_unbound_registration_is_ignored(self, tmp_path):
        mgr, worker_body = _manager_with_registration_and_idle_worker(tmp_path)
        with mgr.lock:
            mgr._registrations_locked()["m1"].pop("index")
        mgr.handle_heartbeat({**worker_body, "worker_id": "w1"})
        assert _wait_for_live_mesh(mgr, "m1", timeout_s=0.3) is None, (
            "auto-relaunch only guards chain-bound registrations"
        )
