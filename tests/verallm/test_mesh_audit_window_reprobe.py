"""Bounded audit windows: strict-quiesce gate, in-flight registry, busy bytes.

The capture-skip removes the accidental decode quiescence organic joins used
to provide, so a hard audit whose probes keep colliding escalates ONCE: it
gates NEW capture-skipping organics, waits out the in-flight ones, and runs
a final solo probe pass. The gate's refusal must be byte-identical to the
admission ledger's saturation refusal (canary-oracle rule) and its wait must
keep streams alive with the ordinary keepalive ticks.
"""

from __future__ import annotations

import threading
import time

import pytest

from verallm.mesh import CapabilityAd
from verallm.mesh import worker as worker_module
from verallm.mesh.worker import (
    _CaptureWindowBusy,
    make_worker_server,
)


def _capability(endpoint: str = "http://127.0.0.1:9338") -> CapabilityAd:
    return CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=endpoint,
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[],
    )


def _make_server(tmp_path):
    return make_worker_server(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        backend_url="http://127.0.0.1:9",
        require_proof=True,
        proof_trace_enable_file=str(tmp_path / "enable"),
        proof_trace_dir=str(tmp_path / "traces"),
        proof_sample_bps=10000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=8,
        decode_audit_bps=1000,
        llama_n_parallel=4,
        llama_n_ubatch=512,
        proof_trace_manifest_format="compact-raw-v3",
    )


def test_quiesce_gate_released_serves_pass_through(tmp_path):
    server = _make_server(tmp_path)
    try:
        started = time.monotonic()
        server.verathos_wait_out_strict_quiesce(5.0, None)
        assert time.monotonic() - started < 0.5
    finally:
        server.server_close()


def test_quiesce_gate_times_out_into_capture_busy(tmp_path):
    server = _make_server(tmp_path)
    try:
        server.verathos_strict_quiesce.set()
        started = time.monotonic()
        with pytest.raises(_CaptureWindowBusy):
            server.verathos_wait_out_strict_quiesce(0.6, None)
        elapsed = time.monotonic() - started
        assert 0.5 <= elapsed < 3.0
    finally:
        server.verathos_strict_quiesce.clear()
        server.server_close()


def test_quiesce_gate_releases_waiters_on_clear(tmp_path):
    server = _make_server(tmp_path)
    try:
        server.verathos_strict_quiesce.set()
        released = threading.Event()

        def waiter() -> None:
            server.verathos_wait_out_strict_quiesce(10.0, None)
            released.set()

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.4)
        assert not released.is_set()
        server.verathos_strict_quiesce.clear()
        assert released.wait(timeout=3.0)
        thread.join(timeout=2.0)
    finally:
        server.verathos_strict_quiesce.clear()
        server.server_close()


def test_quiesce_gate_emits_keepalive_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module, "CAPTURE_WAIT_TICK_S", 0.2)
    server = _make_server(tmp_path)
    ticks: list[float] = []
    try:
        server.verathos_strict_quiesce.set()
        with pytest.raises(_CaptureWindowBusy):
            server.verathos_wait_out_strict_quiesce(
                1.0, lambda: ticks.append(time.monotonic())
            )
        assert len(ticks) >= 2
    finally:
        server.verathos_strict_quiesce.clear()
        server.server_close()


def test_inflight_registry_counts_and_notifies(tmp_path):
    server = _make_server(tmp_path)
    try:
        assert server.verathos_organic_inflight["count"] == 0
        with server.verathos_organic_inflight_tracked():
            assert server.verathos_organic_inflight["count"] == 1
            with server.verathos_organic_inflight_tracked():
                assert server.verathos_organic_inflight["count"] == 2
            assert server.verathos_organic_inflight["count"] == 1
        assert server.verathos_organic_inflight["count"] == 0

        # The escalation's drain wait is notified when the last serve
        # leaves - no polling latency at the moment that matters.
        with server.verathos_organic_inflight_tracked():
            drained = threading.Event()

            def drain_wait() -> None:
                with server.verathos_organic_inflight_cond:
                    while server.verathos_organic_inflight["count"] > 0:
                        server.verathos_organic_inflight_cond.wait(timeout=5.0)
                drained.set()

            waiter = threading.Thread(target=drain_wait, daemon=True)
            waiter.start()
            time.sleep(0.2)
            assert not drained.is_set()
        assert drained.wait(timeout=3.0)
    finally:
        server.server_close()


def test_quiesce_refusal_wording_is_byte_identical_to_saturation_busy():
    """Canary-oracle pin: every slots-saturation refusal in the worker -
    the _CaptureWindowBusy mapping (which the quiesce gate raises into),
    the capacity-drain refusal, and the admission ledger's BUSY - must
    construct the IDENTICAL body. An observer must not be able to tell an
    audit window from ordinary saturation by wording, fields, or flags.
    Source-level pin: editing any one literal breaks this test."""

    import inspect
    import re

    source = inspect.getsource(worker_module)
    # Normalize whitespace inside each busy-body construction.
    bodies = re.findall(
        r"\{\s*\"error\":\s*\(\s*f\"all \{int\(llama_n_parallel\)\} generation \"\s*"
        r"\"slots are busy; retry or fail over\"\s*\),\s*"
        r"\"type\":\s*\"slots_busy\",\s*"
        r"\"retryable\":\s*True,\s*\}",
        source,
    )
    # Capture-busy mapper + capacity-drain refusal + admission BUSY.
    assert len(bodies) >= 3, (
        "expected the identical slots_busy body at every refusal site; "
        f"found {len(bodies)} - a wording drift would hand callers an "
        "oracle for audit windows"
    )
    # And the capture-busy mapper is what _CaptureWindowBusy resolves to.
    capture_mapping = source.split("isinstance(exc, _CaptureWindowBusy)", 1)[1]
    assert "slots are busy; retry or fail over" in capture_mapping[:2000]
