"""The driver's coordinator self-tests must dial loopback.

The coordinator's internal-HMAC lane on validator routes only accepts
loopback clients (verallm/mesh/worker.py). The coordinator therefore binds
all interfaces, and every driver-local self-test (_verify_serving, operator
chats) dials 127.0.0.1 instead of the advertise host. Before this was fixed,
any pool whose worker advertised a real address failed drive() with
"backend self-test failed: HTTP 403".
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

import pytest

import verallm.mesh.pool as pool_module
from verallm.mesh.http_auth import sign_internal_http_request
from verallm.mesh.pool import LocalMeshRunner, MeshPoolToken, PoolWorkerConfig
from verallm.mesh.types import CapabilityAd, MeshSpec
from verallm.mesh.worker import serve_worker_in_thread


def _config(tmp_path, advertise_host: str = "203.0.113.9") -> PoolWorkerConfig:
    """A worker config whose advertise host is a non-loopback address."""
    return PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-test",
            manager_endpoint="http://manager.local:9500",
            pool_secret="test-secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "worker",
        advertise_host=advertise_host,
        rpc_port=50052,
        proof_port=9402,
        mesh_port=9443,
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
    )


def _first_non_loopback_ipv4() -> str:
    """A locally-assigned non-loopback IPv4, or empty when the host has none."""
    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(), None, family=socket.AF_INET
        )
    except OSError:
        addresses = []
    for info in addresses:
        candidate = info[4][0]
        if not candidate.startswith("127."):
            return candidate
    # Fall back to the routing trick: no packets are sent for UDP connect.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        candidate = probe.getsockname()[0]
    except OSError:
        return ""
    finally:
        probe.close()
    return "" if candidate.startswith("127.") else candidate


def _signed_post(url: str, path: str, secret: bytes, body: bytes):
    headers = sign_internal_http_request(
        secret=secret, method="POST", path=path, body=body
    )
    request = urllib.request.Request(
        url + path,
        data=body,
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_internal_hmac_lane_on_validator_routes_is_loopback_only():
    """The invariant the loopback fix depends on: an identical HMAC-signed
    request to a validator route passes auth from 127.0.0.1 and is refused
    with 403 from any non-loopback client address."""
    lan_ip = _first_non_loopback_ipv4()
    if not lan_ip:
        pytest.skip("host has no non-loopback IPv4 address")
    secret = b"pool-selftest-secret"
    server, thread = serve_worker_in_thread(
        capability=CapabilityAd(
            uid=1,
            hotkey="5MeshCoordinator",
            endpoint="http://127.0.0.1:9338",
            supported_backends=["gguf_stage"],
        ),
        host="0.0.0.0",
        server_role="coordinator",
        internal_auth_secret=secret,
    )
    port = server.server_address[1]
    body = json.dumps(
        {"messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}
    ).encode()
    try:
        loopback_status, loopback_payload = _signed_post(
            f"http://127.0.0.1:{port}", "/v1/chat/completions", secret, body
        )
        remote_status, remote_payload = _signed_post(
            f"http://{lan_ip}:{port}", "/v1/chat/completions", secret, body
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    # Loopback passes the auth gate: it fails later, on the missing backend,
    # never on validator authentication.
    assert loopback_status != 403
    assert "validator authentication" not in str(loopback_payload.get("error", ""))
    # The same signed request from a non-loopback address is refused.
    assert remote_status == 403
    assert remote_payload["error"] == "validator authentication is required"


def test_drive_self_test_dials_loopback_not_advertise_host(monkeypatch, tmp_path):
    """A single-box drive must run _verify_serving against 127.0.0.1 even
    when the advertise host is a routable address."""
    config = _config(tmp_path)
    runner = LocalMeshRunner(config)
    verified_endpoints: list[str] = []
    joined = MeshSpec.new_private_mesh(
        coordinator_uid=7,
        coordinator_hotkey="5Coordinator",
        endpoint=f"http://{config.advertise_host}:{config.mesh_port}",
        model_id="m1",
        model_package_hash="a" * 64,
        total_layers=4,
    )

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(
        pool_module, "join_mesh", lambda **_kwargs: (tmp_path / "joined", joined)
    )
    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_wait_http", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_llama_first_batch_probe", lambda *_a, **_k: None
    )
    monkeypatch.setattr(runner, "_wait_tcp", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_verify_serving",
        lambda endpoint, **_kwargs: verified_endpoints.append(endpoint),
    )
    monkeypatch.setattr(pool_module, "_prewarm_proof_cache", finish_prewarm)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": "a" * 64},
    )

    result = runner.drive(
        {
            "action": "drive",
            "serving_mode": "dev",
            "model_id": "m1",
            "member_count": 1,
            "member_vram": [24],
            "max_context_len": 32_768,
        }
    )
    assert result["event"] == "drive_ready"
    # The published endpoint stays the advertise address; only the local
    # self-test dial is loopback.
    assert result["coordinator_endpoint"] == (
        f"http://{config.advertise_host}:{config.mesh_port}"
    )
    assert verified_endpoints == [f"http://127.0.0.1:{config.mesh_port}"]


def test_verify_backend_ready_dials_loopback(monkeypatch, tmp_path):
    """The multi-box readiness gate must also self-test over loopback."""
    config = _config(tmp_path)
    runner = LocalMeshRunner(config)
    verified_endpoints: list[str] = []

    monkeypatch.setattr(runner, "_assert_command_active", lambda: None)
    monkeypatch.setattr(runner, "_wait_http", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_llama_first_batch_probe", lambda *_a, **_k: None
    )
    monkeypatch.setattr(runner, "_finalize_verification_snapshot", lambda: None)
    monkeypatch.setattr(
        runner,
        "_verify_serving",
        lambda endpoint, **_kwargs: verified_endpoints.append(endpoint),
    )
    runner._prewarm_result["converged"] = True
    runner._prewarm_done.set()

    runner.verify_backend_ready()
    assert verified_endpoints == [f"http://127.0.0.1:{config.mesh_port}"]


def test_verify_serving_retries_warmup_transient_template(monkeypatch, tmp_path):
    """A fresh mesh's first self-test request seeds the capture state, so a
    "no capture window holds a whole forward yet" 500 is warmup-transient:
    the retried request arms from the start of its own forward and the
    full-forward template materializes. _verify_serving must retry that
    class instead of erroring the mesh (observed on a 4-GPU glm serve:
    a 211-op CUDA3-only window against a 237-op challenge floor)."""
    config = _config(tmp_path)
    runner = LocalMeshRunner(config)
    monkeypatch.setattr(runner, "_assert_command_active", lambda: None)
    monkeypatch.setattr(runner, "_local_request_context", lambda: ("", {}))
    monkeypatch.setattr(pool_module.time, "sleep", lambda _s: None)

    calls: list[int] = []

    def flaky_post(url, payload, *, timeout, internal_auth_secret):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(
                "HTTP 500 from http://127.0.0.1:9443/v1/chat/completions: "
                '{"error": "slot view template covers 211 ops for 79 layers, '
                "below the 237-op challenge floor; no capture window holds a "
                'whole forward yet"}'
            )
        return {"verathos_mesh": {"proof_required": True, "verified": True}}

    monkeypatch.setattr(pool_module, "post_json", flaky_post)
    runner._verify_serving("http://127.0.0.1:9443")
    assert len(calls) == 3


def test_verify_serving_does_not_retry_non_transient_failures(
    monkeypatch, tmp_path
):
    """Every failure outside the warmup-transient class stays fatal on the
    first attempt - the honest-error contract of drive() is unchanged."""
    config = _config(tmp_path)
    runner = LocalMeshRunner(config)
    monkeypatch.setattr(runner, "_assert_command_active", lambda: None)
    monkeypatch.setattr(runner, "_local_request_context", lambda: ("", {}))

    calls: list[int] = []

    def failing_post(url, payload, *, timeout, internal_auth_secret):
        calls.append(1)
        raise RuntimeError("HTTP 403 from backend: validator authentication")

    monkeypatch.setattr(pool_module, "post_json", failing_post)
    with pytest.raises(RuntimeError, match="backend self-test failed"):
        runner._verify_serving("http://127.0.0.1:9443")
    assert len(calls) == 1


def test_verify_serving_generates_real_decode_steps(monkeypatch, tmp_path):
    """The self-test must request MORE than one token: a 1-token completion
    samples from the prefill graph's logits, so no decode-shaped graph runs
    and the slot-view template can never assemble its multi-device challenge
    universe on a fresh trace dir (observed: 211-op CUDA3-only template
    vs a 237-op floor)."""
    config = _config(tmp_path)
    runner = LocalMeshRunner(config)
    monkeypatch.setattr(runner, "_assert_command_active", lambda: None)
    monkeypatch.setattr(runner, "_local_request_context", lambda: ("", {}))

    payloads: list[dict] = []

    def capture_post(url, payload, *, timeout, internal_auth_secret):
        payloads.append(payload)
        return {"verathos_mesh": {"proof_required": True, "verified": True}}

    monkeypatch.setattr(pool_module, "post_json", capture_post)
    runner._verify_serving("http://127.0.0.1:9443")
    assert payloads and int(payloads[0]["max_tokens"]) > 1
