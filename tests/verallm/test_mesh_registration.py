"""Mesh registration: fail-closed anchors, index guard, renewal guard."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import verallm.mesh.registration as registration
from verallm.chain.miner_lifecycle import LifecycleRefusal
from verallm.mesh.registration import (
    MeshChainAnchors,
    build_registration_target,
    ensure_evm_registered,
    load_mesh_registration_state,
    register_mesh_endpoint,
    registration_target_from_state,
    renew_once,
    resolve_mesh_chain_anchors,
    save_mesh_registration_state,
)

MODEL_ID = "qwen2.5-7b-q4-k-m"
PRIVATE_KEY = "0x" + "11" * 32


def _spec(**overrides):
    values = {
        "model_id": MODEL_ID,
        "quant_mode": "gguf_q4_k_m",
        "weight_file_hash": b"\x01" * 32,
        "weight_merkle_root": b"\x02" * 32,
        "tokenizer_hash": b"\x03" * 32,
        "weight_block_merkle_roots": [],
        "num_layers": 28,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _entry(target, *, active=True, expires_at=None):
    return SimpleNamespace(
        model_id=target.model_id,
        endpoint=target.endpoint,
        quant=target.quant,
        model_spec_ref=target.model_spec_ref,
        max_context_len=target.max_context_len,
        active=active,
        expires_at=(
            int(time.time()) + 86_400 if expires_at is None else expires_at
        ),
    )


class _FakeModelRegistry:
    def __init__(self, spec):
        self._spec = spec

    def get_model_spec(self, model_id):
        return self._spec if self._spec else None


class _FakeMinerRegistry:
    def __init__(self, entries):
        self.entries = entries
        self.register_calls = []
        self.renew_calls = []

    def get_miner_models(self, address):
        return list(self.entries)

    def register_model(self, model_id, endpoint, model_spec_ref, quant,
                       max_context_len, private_key=None):
        self.register_calls.append(model_id)
        return "0x" + "ab" * 32

    def renew_model(self, index, private_key=None):
        self.renew_calls.append(index)
        return "0x" + "cd" * 32


def _wire(monkeypatch, *, spec, miner):
    monkeypatch.setattr(
        registration, "_model_registry_client", lambda cfg: _FakeModelRegistry(spec)
    )
    monkeypatch.setattr(
        registration, "_miner_registry_client", lambda cfg: miner
    )


def test_anchor_resolution_derives_everything_from_spec(monkeypatch):
    _wire(monkeypatch, spec=_spec(), miner=_FakeMinerRegistry([]))
    anchors = resolve_mesh_chain_anchors(
        None, MODEL_ID, max_context_len=32_768
    )
    assert anchors.registry_quant == "gguf_mesh_q4_k_m"
    assert anchors.chain_quantization_scheme == "gguf_q4_k_m"
    assert anchors.model_package_hash == "01" * 32
    assert anchors.model_tensor_manifest_root == "02" * 32
    assert anchors.tokenizer_hash == "03" * 32
    assert anchors.total_layers == 28
    assert anchors.max_context_len == 32_768


@pytest.mark.parametrize(
    "mutation, match",
    [
        ({"__absent__": True}, "absent from ModelRegistry"),
        ({"quant_mode": "fp8"}, "not a GGUF scheme"),
        ({"model_id": "other"}, "does not match"),
        ({"weight_file_hash": b"\x00" * 32}, "non-zero"),
        ({"weight_merkle_root": b"\x02" * 16}, "32 bytes"),
        ({"weight_block_merkle_roots": [b"\x04" * 32]}, "must be empty"),
        ({"num_layers": 0}, "num_layers"),
    ],
)
def test_anchor_resolution_fails_closed(monkeypatch, mutation, match):
    spec = None if mutation.get("__absent__") else _spec(**mutation)
    _wire(monkeypatch, spec=spec, miner=_FakeMinerRegistry([]))
    with pytest.raises(LifecycleRefusal, match=match):
        resolve_mesh_chain_anchors(None, MODEL_ID, max_context_len=32_768)


def test_unapproved_model_fails_closed(monkeypatch):
    _wire(
        monkeypatch,
        spec=_spec(model_id="not-approved"),
        miner=_FakeMinerRegistry([]),
    )
    with pytest.raises(LifecycleRefusal, match="approved mesh scoring profile"):
        resolve_mesh_chain_anchors(None, "not-approved", max_context_len=32_768)


def _anchors(monkeypatch):
    _wire(monkeypatch, spec=_spec(), miner=_FakeMinerRegistry([]))
    return resolve_mesh_chain_anchors(None, MODEL_ID, max_context_len=32_768)


def test_register_appends_and_verifies_index(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    miner = _FakeMinerRegistry([])

    def register_model(*args, **kwargs):
        miner.register_calls.append(args[0])
        miner.entries.append(_entry(target))
        return "0x" + "ab" * 32

    miner.register_model = register_model
    _wire(monkeypatch, spec=_spec(), miner=miner)
    outcome = register_mesh_endpoint(
        None, target, private_key=PRIVATE_KEY, expected_index=0
    )
    assert outcome.action == "append"
    assert outcome.index == 0
    assert outcome.tx_hash.startswith("0x")
    assert miner.register_calls == [MODEL_ID]


def test_register_refuses_index_mismatch(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    other = build_registration_target(
        anchors=anchors, endpoint="https://other.example:9443"
    )
    miner = _FakeMinerRegistry([_entry(other)])

    def register_model(*args, **kwargs):
        miner.entries.append(_entry(target))
        return "0x" + "ab" * 32

    miner.register_model = register_model
    _wire(monkeypatch, spec=_spec(), miner=miner)
    # The mesh was launched against predicted index 0, but an unrelated
    # registration slipped in and the real index became 1.
    with pytest.raises(LifecycleRefusal, match="launched against index 0"):
        register_mesh_endpoint(
            None, target, private_key=PRIVATE_KEY, expected_index=0
        )


def test_register_refreshes_changed_contract_at_same_index(monkeypatch):
    """A re-measured max_context_len on an ACTIVE slot updates that slot
    in place (deactivate + register reactivates the same index), never
    appends a new index - score history is per (address, index)."""
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    stale = _entry(target)
    stale.max_context_len = 8192  # chain still carries the old contract
    miner = _FakeMinerRegistry([stale])
    calls: list[str] = []

    def deactivate_model(index, private_key=None):
        calls.append(f"deactivate:{index}")
        miner.entries[index].active = False
        return "0x" + "cd" * 32

    def register_model(*args, **kwargs):
        calls.append("register")
        # The contract reactivates the matching tuple IN PLACE.
        miner.entries[0] = _entry(target)
        return "0x" + "ab" * 32

    miner.deactivate_model = deactivate_model
    miner.register_model = register_model
    _wire(monkeypatch, spec=_spec(), miner=miner)
    outcome = register_mesh_endpoint(
        None, target, private_key=PRIVATE_KEY, expected_index=0
    )
    assert outcome.action == "refresh"
    assert outcome.index == 0
    assert calls == ["deactivate:0", "register"]


def test_refresh_register_failure_names_the_deactivated_slot(monkeypatch):
    """deactivate→register is non-atomic: when the register leg fails the
    slot is left OFF CHAIN, and the error must say so plus name the
    recovery (re-run reactivates the same slot in place)."""
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    stale = _entry(target)
    stale.max_context_len = 8192
    miner = _FakeMinerRegistry([stale])

    def deactivate_model(index, private_key=None):
        miner.entries[index].active = False
        return "0x" + "cd" * 32

    def register_model(*args, **kwargs):
        raise RuntimeError("rpc timeout")

    miner.deactivate_model = deactivate_model
    miner.register_model = register_model
    _wire(monkeypatch, spec=_spec(), miner=miner)
    with pytest.raises(LifecycleRefusal, match="OFF CHAIN.*same slot"):
        register_mesh_endpoint(
            None, target, private_key=PRIVATE_KEY, expected_index=0
        )
    assert miner.entries[0].active is False


def test_register_reuses_exact_active_entry_without_tx(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    miner = _FakeMinerRegistry([_entry(target)])
    _wire(monkeypatch, spec=_spec(), miner=miner)
    outcome = register_mesh_endpoint(None, target, private_key=PRIVATE_KEY)
    assert outcome.action == "reuse-active"
    assert outcome.tx_hash == ""
    assert miner.register_calls == []


def test_register_runs_eligibility_gate_at_write_time(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    # Chain state changed between preflight and write: spec vanished.
    miner = _FakeMinerRegistry([])
    _wire(monkeypatch, spec=None, miner=miner)
    with pytest.raises(LifecycleRefusal, match="absent from ModelRegistry"):
        register_mesh_endpoint(None, target, private_key=PRIVATE_KEY)
    assert miner.register_calls == []


def test_renew_refuses_expired_and_renews_live(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    expired = _FakeMinerRegistry(
        [_entry(target, expires_at=int(time.time()) - 10)]
    )
    _wire(monkeypatch, spec=_spec(), miner=expired)
    with pytest.raises(LifecycleRefusal, match="expired"):
        renew_once(None, target, 0, private_key=PRIVATE_KEY)
    assert expired.renew_calls == []

    live = _FakeMinerRegistry([_entry(target)])
    _wire(monkeypatch, spec=_spec(), miner=live)
    outcome = renew_once(None, target, 0, private_key=PRIVATE_KEY)
    assert outcome.action == "renew"
    assert live.renew_calls == [0]


def test_renew_refuses_when_eligibility_regresses(monkeypatch):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    live = _FakeMinerRegistry([_entry(target)])
    # The ModelSpec was altered on chain: the lease must lapse.
    _wire(monkeypatch, spec=_spec(weight_merkle_root=b"\x00" * 32), miner=live)
    with pytest.raises(LifecycleRefusal, match="non-zero"):
        renew_once(None, target, 0, private_key=PRIVATE_KEY)
    assert live.renew_calls == []


def test_ensure_evm_registered_reconciles(monkeypatch):
    calls = {"register": 0}
    state = {"bound": False}

    class _EvmClient:
        def get_associated_uid(self, addr, refresh=False):
            return 7 if state["bound"] else None

        def get_registered_uid_for_evm(self, addr, refresh=False):
            return 7 if state["bound"] else None

        def get_registered_evm_for_uid(self, uid, refresh=False):
            from eth_account import Account

            return Account.from_key(PRIVATE_KEY).address if state["bound"] else None

        def register_evm(self, uid, hotkey_seed, netuid, private_key=None):
            calls["register"] += 1
            state["bound"] = True
            return "0x" + "ee" * 32

    monkeypatch.setattr(
        registration, "_miner_registry_client", lambda cfg: _EvmClient()
    )
    assert ensure_evm_registered(
        None, uid=7, hotkey_seed=b"\x05" * 32, netuid=405, private_key=PRIVATE_KEY
    )
    assert calls["register"] == 1
    # Already bound: no transaction.
    assert not ensure_evm_registered(
        None, uid=7, hotkey_seed=b"\x05" * 32, netuid=405, private_key=PRIVATE_KEY
    )
    assert calls["register"] == 1


def test_registration_state_round_trip(monkeypatch, tmp_path):
    anchors = _anchors(monkeypatch)
    target = build_registration_target(
        anchors=anchors, endpoint="https://mesh.example:9443"
    )
    path = save_mesh_registration_state(
        tmp_path, target=target, index=3, mesh_key="m-abc", expires_at=123456
    )
    assert path.stat().st_mode & 0o777 == 0o600
    state = load_mesh_registration_state(tmp_path)
    assert state["index"] == 3
    assert state["mesh_key"] == "m-abc"
    restored = registration_target_from_state(state)
    assert restored == target
    assert load_mesh_registration_state(tmp_path / "nope") is None


def test_registration_state_clear_removes_the_stored_lease(tmp_path):
    """`mesh retire` clears the manager's stored registration; without
    this the lease renewer would resurrect a deliberately deactivated
    chain entry within 12h."""
    from tests.verallm.test_mesh_pool import _subnet_manager

    manager = _subnet_manager(tmp_path)
    secret = manager.state["management_secret"]
    stored = {
        "model_id": "glm-5.2-iq2-m",
        "endpoint": "http://198.19.0.4:20043",
        "quant": "IQ2_M",
        "max_context_len": 98304,
        "model_spec_ref": "0x" + "ab" * 32,
        "index": 40,
        "mesh_key": "m-1",
        "expires_at": 1_900_000_000,
    }
    manager.handle_registration_state(
        {"management_secret": secret, "registration": stored}
    )
    assert (
        manager.state["mesh_registrations"]["glm-5.2-iq2-m"]["index"] == 40
    )
    payload = manager.handle_registration_state(
        {"management_secret": secret, "clear": True}
    )
    assert payload["registration"] is None
    assert payload["cleared"]["index"] == 40
    assert not manager.state.get("mesh_registrations")
    # Idempotent: clearing again is fine.
    again = manager.handle_registration_state(
        {"management_secret": secret, "clear": True}
    )
    assert again["cleared"] is None


def test_cmd_retire_stops_deactivates_and_clears(monkeypatch, capsys):
    """Retire = stop mesh -> deactivateModel (releases the endpoint claim)
    -> verify inactive -> clear stored registration. If the chain still
    reads active after the tx, the stored registration is KEPT so the
    renewer and a re-run can recover."""
    import argparse

    import verallm.chain.miner_registry as miner_registry_module
    from verallm.chain.config import ChainConfig
    from verallm.mesh import cli as mesh_cli

    stored = {
        "model_id": "glm-5.2-iq2-m",
        "endpoint": "http://198.19.0.4:20043",
        "quant": "IQ2_M",
        "max_context_len": 98304,
        "model_spec_ref": "0x" + "ab" * 32,
        "index": 1,
        "mesh_key": "m-1",
        "expires_at": 1_900_000_000,
    }
    calls: list[tuple[str, dict]] = []
    pool_meshes = {"m-1": {"status": "serving"}}

    def fake_call(route, body):
        calls.append((route, dict(body)))
        if route == "/v1/pool/registration-state":
            if body.get("clear"):
                return {"status": "ok", "registration": None}
            return {"status": "ok", "registration": dict(stored)}
        if route == "/v1/pool/stop":
            pool_meshes.pop(body["mesh_key"], None)
            return {"status": "stopping"}
        if route == "/v1/pool/status":
            return {"status": "ok", "meshes": dict(pool_meshes)}
        raise AssertionError(f"unexpected route {route}")

    monkeypatch.setattr(mesh_cli, "_pool_client", lambda args: fake_call)
    monkeypatch.setattr(
        mesh_cli, "_deploy_credentials", lambda args: ("11" * 32, None, "")
    )
    monkeypatch.setattr(
        ChainConfig, "resolve_config_path", classmethod(
            lambda cls, explicit, network: "/tmp/fake_chain.json"
        )
    )
    monkeypatch.setattr(
        ChainConfig, "from_json", classmethod(lambda cls, path: object())
    )

    deactivated: list[int] = []
    active = {"value": True}

    class _FakeRegistry:
        def __init__(self, config):
            pass

        def deactivate_model(self, index, private_key=None):
            deactivated.append(int(index))
            active["value"] = False
            return "0xtx"

        def get_miner_models(self, address):
            import types

            entry = types.SimpleNamespace(active=active["value"])
            return [types.SimpleNamespace(active=True), entry]

    monkeypatch.setattr(
        miner_registry_module, "MinerRegistryClient", _FakeRegistry
    )

    args = argparse.Namespace(
        model_id="",
        chain_config="",
        subtensor_network="test",
        yes=True,
        keep_serving=False,
        timeout=5.0,
    )
    mesh_cli.cmd_retire(args)
    assert deactivated == [1]
    assert ("/v1/pool/stop", {"mesh_key": "m-1"}) in [
        (r, {k: v for k, v in b.items() if k == "mesh_key"})
        for r, b in calls
    ]
    assert any(
        r == "/v1/pool/registration-state" and b.get("clear")
        for r, b in calls
    )
    out = capsys.readouterr().out
    assert '"tx_hash": "0xtx"' in out

    # Chain still active after the tx: keep the stored registration.
    calls.clear()
    pool_meshes["m-1"] = {"status": "serving"}

    class _StubbornRegistry(_FakeRegistry):
        def deactivate_model(self, index, private_key=None):
            deactivated.append(int(index))
            return "0xtx2"

        def get_miner_models(self, address):
            import types

            return [
                types.SimpleNamespace(active=True),
                types.SimpleNamespace(active=True),
            ]

    monkeypatch.setattr(
        miner_registry_module, "MinerRegistryClient", _StubbornRegistry
    )
    with pytest.raises(SystemExit, match="still reads active"):
        mesh_cli.cmd_retire(args)
    assert not any(
        r == "/v1/pool/registration-state" and b.get("clear")
        for r, b in calls
    )


class TestSubstrateEndpoint:
    """One node URL serves both chain clients: http[s] flips to ws[s] for
    the Substrate side (mirror of ChainConfig.resolve_rpc_url's ws->http),
    network names and ws URLs pass through untouched."""

    def test_http_becomes_ws(self):
        assert (
            registration.substrate_endpoint("http://chain.example.org:9944")
            == "ws://chain.example.org:9944"
        )

    def test_https_becomes_wss(self):
        assert (
            registration.substrate_endpoint("https://node.example:9944")
            == "wss://node.example:9944"
        )

    def test_ws_and_names_pass_through(self):
        assert (
            registration.substrate_endpoint("ws://chain.example.org:9944")
            == "ws://chain.example.org:9944"
        )
        assert registration.substrate_endpoint("test") == "test"
        assert registration.substrate_endpoint("finney") == "finney"
        assert registration.substrate_endpoint("") == ""

    def test_resolve_uid_uses_normalized_endpoint(self, monkeypatch):
        seen = {}

        class _FakeSubtensor:
            def __init__(self, network):
                seen["network"] = network

            def metagraph(self, netuid):
                return SimpleNamespace(hotkeys=["5Hotkey"])

        import sys

        monkeypatch.setitem(
            sys.modules,
            "bittensor",
            SimpleNamespace(Subtensor=_FakeSubtensor),
        )
        uid = registration.resolve_uid_for_hotkey(
            "http://chain.example.org:9944", 405, "5Hotkey"
        )
        assert uid == 0
        assert seen["network"] == "ws://chain.example.org:9944"
