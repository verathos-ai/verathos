"""Pool manager lease renewer: window-driven, serving-gated, fail-lapse."""
from __future__ import annotations

import time

import pytest

import verallm.mesh.registration as registration_module
from verallm.chain.miner_lifecycle import LifecycleRefusal
from verallm.mesh.pool import PoolManager, create_pool_state
from verallm.mesh.registration import (
    LEASE_RENEW_WINDOW_S,
    RegistrationOutcome,
    renew_lease_if_due,
)

PRIVATE_KEY = "0x" + "11" * 32


def _manager(tmp_path) -> PoolManager:
    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    return PoolManager(state_dir)


def _registration(expires_in_s: float) -> dict:
    return {
        "model_id": "qwen2.5-7b-q4-k-m",
        "endpoint": "https://mesh.example:9443",
        "quant": "gguf_mesh_q4_k_m",
        "max_context_len": 32_768,
        "model_spec_ref": "aa" * 32,
        "index": 0,
        "mesh_key": "m-1",
        "expires_at": int(time.time() + expires_in_s),
    }


def _wire_renew(monkeypatch, outcomes: list):
    def fake_renew_once(chain_config, target, index, *, private_key):
        outcome = RegistrationOutcome(
            action="renew",
            index=index,
            tx_hash="0x" + "cd" * 32,
            expires_at=int(time.time()) + 86_400,
        )
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(registration_module, "renew_once", fake_renew_once)


def test_renews_only_inside_window(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["meshes"]["m-1"] = {"status": "serving"}

    # 20h remaining: outside the 12h window, nothing happens.
    manager.state["mesh_registrations"] = {
        "qwen2.5-7b-q4-k-m": _registration(20 * 3600)
    }
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    assert outcomes == []

    # 6h remaining: renew, and the stored expiry advances.
    manager.state["mesh_registrations"] = {
        "qwen2.5-7b-q4-k-m": _registration(6 * 3600)
    }
    outcome = renew_lease_if_due(
        manager, chain_config=None, private_key=PRIVATE_KEY
    )
    assert outcome is not None
    stored = manager.state["mesh_registrations"]["qwen2.5-7b-q4-k-m"]
    assert stored["expires_at"] == outcome.expires_at
    assert stored["expires_at"] > time.time() + LEASE_RENEW_WINDOW_S


def test_dead_mesh_lets_the_lease_lapse(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["mesh_registration"] = _registration(3600)

    # Mesh gone entirely.
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    # Mesh present but errored.
    manager.state["meshes"]["m-1"] = {"status": "error"}
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    assert outcomes == []


def test_no_registration_is_a_quiet_noop(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    assert outcomes == []


def test_eligibility_regression_propagates(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    manager.state["meshes"]["m-1"] = {"status": "serving"}
    manager.state["mesh_registration"] = _registration(3600)

    def refuse(chain_config, target, index, *, private_key):
        raise LifecycleRefusal("ModelRegistry weight_merkle_root must be a non-zero commitment")

    monkeypatch.setattr(registration_module, "renew_once", refuse)
    with pytest.raises(LifecycleRefusal):
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)


def test_idempotent_across_restart(monkeypatch, tmp_path):
    """The trigger is the persisted on-chain expires_at, so a fresh manager
    over the same state dir picks up exactly where the old one stopped."""
    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["meshes"]["m-1"] = {"status": "serving"}
    manager.state["mesh_registration"] = _registration(6 * 3600)
    first = renew_lease_if_due(
        manager, chain_config=None, private_key=PRIVATE_KEY
    )
    assert first is not None
    # Same tick again: freshly renewed, outside the window, no second tx.
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    assert len(outcomes) == 1


def test_registration_state_route_round_trip(tmp_path):
    manager = _manager(tmp_path)
    secret = manager.state["management_secret"]
    stored = manager.handle_registration_state(
        {"management_secret": secret, "registration": _registration(3600)}
    )
    assert stored["registration"]["mesh_key"] == "m-1"
    # Read-only call returns the stored value.
    read = manager.handle_registration_state({"management_secret": secret})
    assert read["registration"]["index"] == 0
    # Incomplete payloads are refused.
    with pytest.raises(ValueError, match="missing"):
        manager.handle_registration_state(
            {"management_secret": secret, "registration": {"model_id": "x"}}
        )
    # Worker tokens cannot touch it.
    with pytest.raises(PermissionError):
        manager.handle_registration_state(
            {"pool_secret": manager.state["pool_secret"]}
        )


def _registration_for(model_id: str, mesh_key: str, expires_in_s: float) -> dict:
    return {
        **_registration(expires_in_s),
        "model_id": model_id,
        "mesh_key": mesh_key,
        "index": {"glm-x": 40, "qwen-x": 41}.get(model_id, 7),
    }


def test_multi_model_registrations_round_trip_and_scoped_clear(tmp_path):
    """A pool holds one registration PER model: storing a second model must
    not overwrite the first (observed: a glm mesh and a qwen mesh on
    the same pool), and clear is model-scoped once several exist."""

    manager = _manager(tmp_path)
    secret = manager.state["management_secret"]
    manager.handle_registration_state(
        {
            "management_secret": secret,
            "registration": _registration_for("glm-x", "m-glm", 3600),
        }
    )
    stored = manager.handle_registration_state(
        {
            "management_secret": secret,
            "registration": _registration_for("qwen-x", "m-qwen", 3600),
        }
    )
    assert set(stored["registrations"]) == {"glm-x", "qwen-x"}
    # No single-slot compat field once several are stored.
    assert stored["registration"] is None
    # A bare clear is ambiguous on a multi-model pool.
    with pytest.raises(ValueError, match="model_id"):
        manager.handle_registration_state(
            {"management_secret": secret, "clear": True}
        )
    cleared = manager.handle_registration_state(
        {"management_secret": secret, "clear": True, "model_id": "glm-x"}
    )
    assert cleared["cleared"]["mesh_key"] == "m-glm"
    read = manager.handle_registration_state({"management_secret": secret})
    assert set(read["registrations"]) == {"qwen-x"}
    # Back to one: the single-slot compat field returns.
    assert read["registration"]["model_id"] == "qwen-x"


def test_legacy_single_slot_state_migrates_in_place(tmp_path):
    manager = _manager(tmp_path)
    secret = manager.state["management_secret"]
    manager.state["mesh_registration"] = _registration_for(
        "glm-x", "m-glm", 3600
    )
    read = manager.handle_registration_state({"management_secret": secret})
    assert set(read["registrations"]) == {"glm-x"}
    assert "mesh_registration" not in manager.state


def test_renewer_renews_every_due_model(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["meshes"]["m-glm"] = {"status": "serving"}
    manager.state["meshes"]["m-qwen"] = {"status": "serving"}
    manager.state["mesh_registrations"] = {
        "glm-x": _registration_for("glm-x", "m-glm", 3600),
        "qwen-x": _registration_for("qwen-x", "m-qwen", 3600),
    }
    outcome = renew_lease_if_due(
        manager, chain_config=None, private_key=PRIVATE_KEY
    )
    assert outcome is not None
    assert len(outcomes) == 2
    stored = manager.state["mesh_registrations"]
    assert all(
        int(entry["expires_at"]) > time.time() + 23 * 3600
        for entry in stored.values()
    )


def test_relaunched_mesh_is_adopted_and_stored_key_heals(monkeypatch, tmp_path):
    """A relaunch mints a NEW mesh_key while the stored registration keeps
    the old one. The registration's contract is the (model, index) chain
    slot, not one mesh incarnation: the renewer must adopt the serving
    relaunch and heal the stored key instead of letting a LIVE chain entry
    expire ."""

    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["mesh_registrations"] = {
        "qwen2.5-7b-q4-k-m": _registration(6 * 3600)
    }
    # The registered mesh_key m-1 no longer exists; the SAME model serves
    # chain-bound at the SAME index under a fresh key.
    manager.state["meshes"]["m-relaunched"] = {
        "status": "serving",
        "model_id": "qwen2.5-7b-q4-k-m",
        "model_index": 0,
    }

    outcome = renew_lease_if_due(
        manager, chain_config=None, private_key=PRIVATE_KEY
    )
    assert outcome is not None
    stored = manager.state["mesh_registrations"]["qwen2.5-7b-q4-k-m"]
    assert stored["mesh_key"] == "m-relaunched"
    assert stored["expires_at"] == outcome.expires_at


def test_adoption_requires_same_model_and_index(monkeypatch, tmp_path):
    """The dead-mesh rule survives adoption: a serving mesh with a
    different model or index is NOT this registration's slot, so the
    lease still lapses."""

    manager = _manager(tmp_path)
    outcomes: list = []
    _wire_renew(monkeypatch, outcomes)
    manager.state["mesh_registrations"] = {
        "qwen2.5-7b-q4-k-m": _registration(3600)
    }
    manager.state["meshes"]["m-other-model"] = {
        "status": "serving",
        "model_id": "some-other-model",
        "model_index": 0,
    }
    manager.state["meshes"]["m-other-index"] = {
        "status": "serving",
        "model_id": "qwen2.5-7b-q4-k-m",
        "model_index": 7,
    }
    manager.state["meshes"]["m-measurement"] = {
        # An unregistered measurement launch has no model_index at all.
        "status": "serving",
        "model_id": "qwen2.5-7b-q4-k-m",
    }
    assert (
        renew_lease_if_due(manager, chain_config=None, private_key=PRIVATE_KEY)
        is None
    )
    assert outcomes == []
    assert (
        manager.state["mesh_registrations"]["qwen2.5-7b-q4-k-m"]["mesh_key"]
        == "m-1"
    )
