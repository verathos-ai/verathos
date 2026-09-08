from __future__ import annotations

import stat
import os
import json
import pytest
import time
from types import SimpleNamespace

from neurons.mesh_repin_ipc import (
    acknowledge_mesh_repin_request,
    enqueue_mesh_repin_request,
    mesh_repin_spool_dir,
    pending_mesh_repin_requests,
)
from neurons.validator import ValidatorNeuron
from verallm.mesh.private_files import write_owner_only_text


ADDRESS = "0x" + "ab" * 20


def test_private_relay_preserves_consumer_and_debounce(tmp_path, monkeypatch):
    from neurons.mesh_repin_ipc import relay_mesh_repin_requests
    source, target = tmp_path / "source", tmp_path / "target"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(source))
    path = enqueue_mesh_repin_request(address=ADDRESS, model_index=3, reason="snapshot mismatch")
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(target))
    assert relay_mesh_repin_requests(source, source_uid=os.geteuid(), shared_state_path="unused") == 1
    assert not path.exists()
    pending = pending_mesh_repin_requests()
    assert len(pending) == 1
    assert pending[0][1]["model_index"] == 3
    assert stat.S_IMODE(source.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@pytest.mark.parametrize("problem", ["stale", "future", "nan", "public_file", "symlink"])
def test_private_relay_refuses_invalid_source(tmp_path, monkeypatch, problem):
    from neurons.mesh_repin_ipc import relay_mesh_repin_requests
    source, target = tmp_path / "source", tmp_path / "target"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(source))
    path = enqueue_mesh_repin_request(address=ADDRESS, model_index=3, reason="snapshot mismatch")
    payload = json.loads(path.read_text())
    if problem in {"stale", "future", "nan"}:
        payload["requested_at"] = {"stale": time.time()-300, "future": time.time()+300, "nan": float("nan")}[problem]
        write_owner_only_text(path, json.dumps(payload))
    elif problem == "public_file":
        path.chmod(0o644)
    else:
        other = source / "other"
        path.rename(other)
        path.symlink_to(other)
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(target))
    assert relay_mesh_repin_requests(source, source_uid=os.geteuid(), shared_state_path="unused") == 0
    assert not target.exists()


def test_private_relay_refuses_public_directory(tmp_path, monkeypatch):
    from neurons.mesh_repin_ipc import relay_mesh_repin_requests
    source = tmp_path / "source"
    source.mkdir(mode=0o755)
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(tmp_path / "target"))
    with pytest.raises(PermissionError):
        relay_mesh_repin_requests(source, source_uid=os.geteuid(), shared_state_path="unused")


def test_request_round_trips_owner_only(tmp_path, monkeypatch):
    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))

    path = enqueue_mesh_repin_request(
        address=ADDRESS,
        model_index=45,
        reason="mesh coordinator has no validator-pinned verification snapshot",
    )

    assert mesh_repin_spool_dir() == spool
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    pending = pending_mesh_repin_requests()
    assert len(pending) == 1
    assert pending[0][1]["address"] == ADDRESS
    assert pending[0][1]["model_index"] == 45

    acknowledge_mesh_repin_request(pending[0][0])
    assert pending_mesh_repin_requests() == []


def test_requests_debounce_per_slot(tmp_path, monkeypatch):
    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))

    first = enqueue_mesh_repin_request(
        address=ADDRESS, model_index=45, reason="gap"
    )
    assert first is not None
    # A hot routing loop must not rewrite the request every attempt.
    assert (
        enqueue_mesh_repin_request(
            address=ADDRESS, model_index=45, reason="gap again"
        )
        is None
    )
    # A different slot is independent.
    assert (
        enqueue_mesh_repin_request(
            address=ADDRESS, model_index=50, reason="gap"
        )
        is not None
    )
    assert len(pending_mesh_repin_requests()) == 2


def test_invalid_request_is_quarantined(tmp_path, monkeypatch):
    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))
    bad = spool / "repin-not-a-slot.json"
    write_owner_only_text(bad, "{broken")
    good = enqueue_mesh_repin_request(
        address=ADDRESS, model_index=45, reason="gap"
    )

    pending = pending_mesh_repin_requests()

    assert [p for p, _ in pending] == [good]
    assert not bad.exists()
    assert bad.with_suffix(".json.invalid").exists()


def test_validator_serves_requests_through_capped_core(tmp_path, monkeypatch):
    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))
    enqueue_mesh_repin_request(address=ADDRESS, model_index=45, reason="gap")

    calls: list[tuple] = []
    neuron = object.__new__(ValidatorNeuron)
    neuron.config = SimpleNamespace(shared_state_path=str(tmp_path / "state.json"))
    neuron._current_epoch = 43231
    neuron._repin_mesh_snapshot_slot = (
        lambda address, model_index, epoch_number, *, trigger: calls.append(
            (address, model_index, epoch_number, trigger)
        )
        or True
    )

    assert neuron._ingest_mesh_repin_requests() == 1
    assert calls == [(ADDRESS, 45, 43231, "proxy request")]
    # Spent either way - the proxy re-files after its debounce if needed.
    assert pending_mesh_repin_requests() == []


def test_validator_acknowledges_even_when_repin_raises(tmp_path, monkeypatch):
    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))
    enqueue_mesh_repin_request(address=ADDRESS, model_index=45, reason="gap")

    def boom(*_a, **_k):
        raise RuntimeError("fetch failed")

    neuron = object.__new__(ValidatorNeuron)
    neuron.config = SimpleNamespace(shared_state_path="")
    neuron._current_epoch = 1
    neuron._repin_mesh_snapshot_slot = boom

    neuron._ingest_mesh_repin_requests()
    assert pending_mesh_repin_requests() == []


def test_validator_consumes_and_reports_refused_requests(tmp_path, monkeypatch):
    """A request the capped core refuses is still spent — but the ingest
    must SAY so (silent consumption hid a dead self-heal path in prod)."""

    spool = tmp_path / "repin"
    monkeypatch.setenv("VERATHOS_MESH_REPIN_DIR", str(spool))
    enqueue_mesh_repin_request(address=ADDRESS, model_index=45, reason="gap")

    neuron = object.__new__(ValidatorNeuron)
    neuron.config = SimpleNamespace(shared_state_path="")
    neuron._current_epoch = 43306
    neuron._repin_mesh_snapshot_slot = lambda *a, **k: False

    assert neuron._ingest_mesh_repin_requests() == 1
    assert pending_mesh_repin_requests() == []
    # A consumed request can be re-filed immediately (debounce is keyed on
    # the file, which the ingest deleted) — the self-heal loop stays live.
    assert enqueue_mesh_repin_request(
        address=ADDRESS, model_index=45, reason="gap again"
    ) is not None
    assert len(pending_mesh_repin_requests()) == 1


def test_repin_slot_cap_refuses_without_side_effects():
    """Cap refusal happens before any network fetch and returns False."""

    neuron = object.__new__(ValidatorNeuron)
    neuron._mesh_snapshot_repins = {(ADDRESS, 45, 43306): 3}

    assert (
        neuron._repin_mesh_snapshot_slot(
            ADDRESS, 45, 43306, trigger="proxy request"
        )
        is False
    )
    # The counter is untouched by a refused attempt.
    assert neuron._mesh_snapshot_repins == {(ADDRESS, 45, 43306): 3}


def test_adopt_successor_attempts_are_budgeted_per_slot_epoch():
    """Successor rediscovery gets a strictly bounded attempt budget per
    (address, index, epoch): each attempt consumes budget even when it
    stops early, and once the budget is exhausted further attempts are
    refused loudly (returns False) without side effects. A budget instead
    of a single shot keeps transient races from burning the whole epoch
    on one pre-consumed attempt."""

    neuron = object.__new__(ValidatorNeuron)
    neuron._mesh_snapshot_repins = {}
    neuron._miner_client = None

    cap_key = ("adopt", ADDRESS, 45, 43306)
    for attempt in range(1, 4):
        # Each attempt consumes budget, then stops at the missing client.
        assert neuron._adopt_repin_successor(
            ADDRESS, 45, 43306, trigger="proxy request"
        ) is False
        assert neuron._mesh_snapshot_repins == {cap_key: attempt}

    # The budget is exhausted: refusal leaves the counter untouched.
    assert neuron._adopt_repin_successor(
        ADDRESS, 45, 43306, trigger="proxy request"
    ) is False
    assert neuron._mesh_snapshot_repins == {cap_key: 3}
