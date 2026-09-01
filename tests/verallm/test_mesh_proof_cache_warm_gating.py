"""The background proof-cache warm must not stack on the backend model load.

On the 24GB-GPU miner box class (30-32GB RAM) the warm's dequant
transients plus llama's model load exceed the memory budget when they run
concurrently, and the cgroup OOM-kills the worker mid-formation. The warm
therefore waits for the backend's load-complete signal (llama /health
turns 200 only once weights are resident) before touching the GGUF.
"""

from __future__ import annotations

import inspect
import io
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from verallm.mesh import CapabilityAd
from verallm.mesh import worker as worker_mod
from verallm.mesh.worker import serve_worker_in_thread, wait_for_backend_model_loaded


class _FakeResponse(io.BytesIO):
    def __init__(self, status: int) -> None:
        super().__init__(b"{}")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)


def test_empty_backend_url_returns_false_without_polling():
    calls = []

    def opener(url, timeout):  # pragma: no cover - must never run
        calls.append(url)
        raise AssertionError("opener must not be called for empty url")

    assert wait_for_backend_model_loaded("", opener=opener) is False
    assert calls == []


def test_immediate_200_returns_true():
    clock = _FakeClock()

    def opener(url, timeout):
        assert url.endswith("/health")
        return _FakeResponse(200)

    assert (
        wait_for_backend_model_loaded(
            "http://127.0.0.1:9999",
            opener=opener,
            clock=clock,
            sleeper=clock.sleep,
        )
        is True
    )


def test_loading_503_then_200_returns_true_after_polls():
    clock = _FakeClock()
    statuses = iter([503, 503, 200])

    def opener(url, timeout):
        return _FakeResponse(next(statuses))

    assert (
        wait_for_backend_model_loaded(
            "http://127.0.0.1:9999",
            deadline_s=600.0,
            poll_s=10.0,
            opener=opener,
            clock=clock,
            sleeper=clock.sleep,
        )
        is True
    )
    assert clock.now >= 20.0


def test_connection_refused_polls_until_deadline_then_false():
    clock = _FakeClock()
    attempts = []

    def opener(url, timeout):
        attempts.append(clock.now)
        raise OSError("connection refused")

    assert (
        wait_for_backend_model_loaded(
            "http://127.0.0.1:9999",
            deadline_s=60.0,
            poll_s=10.0,
            opener=opener,
            clock=clock,
            sleeper=clock.sleep,
        )
        is False
    )
    assert len(attempts) >= 6


def test_perpetual_503_times_out_false():
    clock = _FakeClock()

    def opener(url, timeout):
        return _FakeResponse(503)

    assert (
        wait_for_backend_model_loaded(
            "http://127.0.0.1:9999",
            deadline_s=30.0,
            poll_s=10.0,
            opener=opener,
            clock=clock,
            sleeper=clock.sleep,
        )
        is False
    )


def test_background_warm_waits_for_backend_before_building():
    """Source anchor: the warm closure gates on the load-complete wait.

    The warm is a nested closure inside the serve bring-up, so unit-driving
    it directly would need the whole server fixture; instead pin the
    invariant at source level, in the spirit of the fetch-never-builds
    suite: the wait call must appear inside ``_background_proof_cache_warm``
    before ``build_proof_weight_cache`` runs.
    """

    source = inspect.getsource(worker_mod)
    marker = "def _background_proof_cache_warm"
    start = source.index(marker)
    build_at = source.index(
        "prewarm_proof_weight_cache_to_convergence(manifest)", start
    )
    wait_at = source.index("wait_for_backend_model_loaded(backend_url)", start)
    assert start < wait_at < build_at


def _health_json(server):
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=2.0) as response:
            return int(response.status), json.loads(response.read())
    except HTTPError as exc:
        return int(exc.code), json.loads(exc.read())


def _capability() -> CapabilityAd:
    return CapabilityAd(
        uid=1,
        hotkey="5ProofWarm",
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[],
    )


def test_worker_health_waits_for_converged_proof_cache(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.gguf_manifest as manifest_module

    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        manifest_module,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensors": []},
    )

    def converge(_manifest):
        started.set()
        assert release.wait(5.0)
        return {"cached": 0, "merkle": 0, "convergence_passes": 2}

    monkeypatch.setattr(
        manifest_module,
        "prewarm_proof_weight_cache_to_convergence",
        converge,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="worker",
        require_proof=True,
        proof_gguf_manifest_path=manifest_path,
    )
    try:
        assert started.wait(2.0)
        status, body = _health_json(server)
        assert status == 503
        assert body["proof_cache_warm_state"] == "warming"
        release.set()
        deadline = time.monotonic() + 3.0
        while True:
            status, body = _health_json(server)
            if status == 200:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert body["proof_cache_warm_state"] == "ready"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_coordinator_health_does_not_wait_on_member_cache_gate(
    tmp_path,
    monkeypatch,
):
    """Multi-box formation must not deadlock before members can join."""

    import verallm.mesh.gguf_manifest as manifest_module

    release = threading.Event()
    monkeypatch.setattr(
        manifest_module,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensors": []},
    )

    def converge(_manifest):
        release.wait(5.0)
        return {"cached": 0, "merkle": 0, "convergence_passes": 2}

    monkeypatch.setattr(
        manifest_module,
        "prewarm_proof_weight_cache_to_convergence",
        converge,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="coordinator",
        require_proof=True,
        proof_gguf_manifest_path=manifest_path,
    )
    try:
        status, body = _health_json(server)
        assert status == 200
        assert body["proof_cache_warm_state"] in {"not_required", "ready"}
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_failed_worker_prewarm_remains_non_serving(tmp_path, monkeypatch):
    import verallm.mesh.gguf_manifest as manifest_module

    failed = threading.Event()
    monkeypatch.setattr(
        manifest_module,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensors": []},
    )

    def reject(_manifest):
        failed.set()
        raise RuntimeError("private cache path must not escape through health")

    monkeypatch.setattr(
        manifest_module,
        "prewarm_proof_weight_cache_to_convergence",
        reject,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="worker",
        require_proof=True,
        proof_gguf_manifest_path=manifest_path,
    )
    try:
        assert failed.wait(2.0)
        deadline = time.monotonic() + 2.0
        while True:
            status, body = _health_json(server)
            if body.get("proof_cache_warm_state") == "failed":
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert status == 503
        assert body["error"] == (
            "proof-weight cache prewarm failed; see worker logs"
        )
        assert "private cache path" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
