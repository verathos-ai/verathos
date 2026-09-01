"""Probe deadline budget and driver timing telemetry."""
from __future__ import annotations

import time

import pytest

import verallm.mesh.pool as pool_module
from verallm.mesh.pool import (
    CHAT_COORDINATOR_TIMEOUT_S,
    CHAT_OPERATOR_DEADLINE_S,
    PROBE_DEADLINE_MAX_S,
    chat_deadline_seconds,
)


def test_chat_deadline_clamps():
    # Ordinary chats keep the interactive ceiling.
    assert chat_deadline_seconds({}) == CHAT_OPERATOR_DEADLINE_S
    assert chat_deadline_seconds({"timeout": 60}) == 60.0
    assert (
        chat_deadline_seconds({"timeout": 10_000}) == CHAT_OPERATOR_DEADLINE_S
    )
    # Probes raise the ceiling but stay bounded.
    assert chat_deadline_seconds({"probe": True}) == PROBE_DEADLINE_MAX_S
    assert chat_deadline_seconds({"probe": True, "timeout": 900}) == 900.0
    assert (
        chat_deadline_seconds({"probe": True, "timeout": 10_000})
        == PROBE_DEADLINE_MAX_S
    )
    # Garbage falls back to the applicable ceiling.
    assert chat_deadline_seconds({"timeout": "nan"}) == CHAT_OPERATOR_DEADLINE_S
    assert chat_deadline_seconds({"timeout": -5}) == CHAT_OPERATOR_DEADLINE_S
    assert (
        chat_deadline_seconds({"probe": True, "timeout": float("inf")})
        == PROBE_DEADLINE_MAX_S
    )
    # The probe ceiling covers the validator's 900 s full-context budget.
    assert PROBE_DEADLINE_MAX_S >= 900.0


def _manager(tmp_path):
    from verallm.mesh.pool import PoolManager, create_pool_state

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    return PoolManager(state_dir)


def test_guard_honours_active_probe_deadline(tmp_path, monkeypatch):
    """A probe legitimately holds the mesh slot past CHAT_ACTIVE_MAX_S; the
    guard must not reclaim it while its own deadline is still in the future."""
    mgr = _manager(tmp_path)
    worker = {"worker_id": "drv", "last_chat_poll_unix": time.time()}
    mgr.state["workers"]["drv"] = worker
    import verallm.mesh.pool as pool_module

    monkeypatch.setattr(pool_module, "_serve_parallel_slots", lambda: 1)
    # Older than CHAT_ACTIVE_MAX_S, which used to force a reclaim.
    mgr.chat_active["m-1"] = {"c-probe": time.monotonic() - 600.0}
    mgr.chat_contexts["c-probe"] = {
        "driver": "drv",
        "mesh_key": "m-1",
        "state": "running",
        "request_deadline_mono": time.monotonic() + 300.0,
    }
    with mgr.lock:
        with pytest.raises(ValueError, match="already running"):
            mgr._guard_chat_slot("m-1", worker)
    # Once the probe's own deadline (plus slack) passes, the slot is
    # reclaimable again.
    mgr.chat_contexts["c-probe"]["request_deadline_mono"] = (
        time.monotonic() - 60.0
    )
    with mgr.lock:
        mgr._guard_chat_slot("m-1", worker)
    assert "m-1" not in mgr.chat_active


def test_chat_result_passes_timing_fields_through(tmp_path):
    mgr = _manager(tmp_path)
    mgr.chat_contexts["c-1"] = {
        "driver": "drv",
        "mesh_key": "m-1",
        "expected_stage_count": 0,
        "state": "running",
        "request_deadline_mono": time.monotonic() + 60.0,
        "last_seq": 0,
        "stream": False,
        "client_disconnected": False,
    }
    waiter = {"event": __import__("threading").Event(), "result": None}
    mgr.chat_waiters["c-1"] = waiter
    response = mgr.handle_chat_result(
        {
            "pool_secret": mgr.state["pool_secret"],
            "worker_id": "drv",
            "chat_id": "c-1",
            "seq": 1,
            "content": "ok",
            "verified": True,
            "receipt_verified": True,
            "receipts": 1,
            "proof_stages": 1,
            "engine_tps": 41.5,
            "ttft_s": 0.82,
            "total_s": 4.31,
            "pickup_s": 0.05,
        }
    )
    assert response["status"] == "ok"
    result = waiter["result"]
    assert result["ttft_s"] == 0.82
    assert result["total_s"] == 4.31
    assert result["pickup_s"] == 0.05
    assert result["engine_tps"] == 41.5


def test_run_pool_chat_nonstream_reports_total(monkeypatch, tmp_path):
    from verallm.mesh.pool import MeshPoolToken, PoolWorkerConfig

    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-test",
            manager_endpoint="http://manager.local:9500",
            pool_secret="s",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "worker",
        advertise_host="203.0.113.9",
        rpc_port=50052,
        proof_port=9402,
        mesh_port=9443,
        llama_server_binary="llama-server",
        rpc_worker_binary="rpc-server",
        catalog=[],
    )
    sent = []

    def fake_post_json(endpoint, body, timeout=0, **_kwargs):
        if endpoint.endswith("/v1/chat/completions"):
            # The per-chat coordinator budget must reach the POST timeout.
            assert timeout <= 900.0
            assert timeout > CHAT_COORDINATOR_TIMEOUT_S
            return {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"completion_tokens": 1},
                "verathos_mesh": {"verified": True, "proof_receipt_count": 1},
            }
        sent.append((endpoint, dict(body)))
        return {"status": "ok"}

    monkeypatch.setattr(pool_module, "post_json", fake_post_json)
    pool_module._run_pool_chat(
        "https://manager.example",
        "secret",
        "drv",
        config,
        {
            "chat_id": "c-t",
            "model": "m1",
            "messages": [{"role": "user", "content": "hi"}],
            "coordinator_timeout_s": 900.0,
            "request_expires_at_unix_ms": int((time.time() + 900) * 1000),
        },
    )
    final = next(
        body for endpoint, body in sent if endpoint.endswith("/v1/pool/chat-result")
    )
    assert final["ttft_s"] == -1.0
    assert final["total_s"] >= 0.0
