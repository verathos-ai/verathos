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

from verallm.mesh import worker as worker_mod
from verallm.mesh.worker import wait_for_backend_model_loaded


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
    build_at = source.index("build_proof_weight_cache(manifest)", start)
    wait_at = source.index("wait_for_backend_model_loaded(backend_url)", start)
    assert start < wait_at < build_at
