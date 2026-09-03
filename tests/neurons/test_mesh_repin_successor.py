"""Successor adoption when a re-pin request names a dead slot.

An owner roll deactivates the old index and registers a successor at a
NEW index mid-epoch. The re-pin core must adopt the successor instead of
leaving the model invisible until the boundary .
"""

from types import SimpleNamespace

from neurons.validator import ValidatorNeuron

ADDRESS = "0x" + "ab" * 20


def _neuron(monkeypatch, fresh_entries, epoch_miners=None):
    neuron = object.__new__(ValidatorNeuron)
    neuron._epoch_miners = list(epoch_miners or [])
    neuron._mesh_snapshot_cache = {
        neuron._mesh_snapshot_cache_key(ADDRESS, 52, 43286): SimpleNamespace(
            model=SimpleNamespace(model_id="glm-5.2-iq2-m")
        )
    }
    neuron._mesh_snapshot_repins = {}
    neuron._miner_client = object()
    neuron._model_client = None
    neuron._db = None
    neuron._validator_hotkey_ss58 = "5Validator"
    neuron._validator_private_key = b"\x01" * 32
    neuron._shared_state_writes = 0
    neuron._enrichment_calls = 0
    neuron._admission_calls = []

    def _write():
        neuron._shared_state_writes += 1

    def _enrich(miners):
        neuron._enrichment_calls += 1
        for miner in miners:
            miner.hotkey_ss58 = "5Miner"

    def _retain(miners, *, epoch_number):
        neuron._admission_calls.append(("hotkey", epoch_number))
        return [m for m in miners if getattr(m, "hotkey_ss58", "")]

    def _policy(name):
        def _apply(miners, *, epoch_number, preserve_existing=False):
            neuron._admission_calls.append(
                (name, epoch_number, preserve_existing)
            )
            return list(miners)

        return _apply

    neuron._write_shared_state = _write
    neuron._enrich_miners_from_metagraph = _enrich
    neuron._retain_epoch_miners_with_authenticated_hotkeys = _retain
    neuron._apply_authenticated_mesh_policy = _policy("mesh")
    neuron._apply_mainnet_endpoint_policy = _policy("endpoint")
    neuron._apply_mainnet_mesh_backend_policy = _policy("backend")

    def _anchors(miner, epoch):
        assert miner.hotkey_ss58 == "5Miner"
        return object()

    neuron._mesh_snapshot_trust_anchors = _anchors
    monkeypatch.setattr(
        "neurons.validator.discover_active_miners",
        lambda mc, mdl: list(fresh_entries),
    )
    monkeypatch.setattr(
        "neurons.validator.discover_and_pin_mesh_verification_snapshot",
        lambda **kw: {"pinned": kw["coordinator_endpoint"]},
    )
    return neuron


def _entry(
    model_index,
    endpoint="http://198.51.100.163:20002",
    *,
    model_id="glm-5.2-iq2-m",
    quant="gguf_mesh_iq2_m",
):
    return SimpleNamespace(
        address=ADDRESS,
        model_index=model_index,
        endpoint=endpoint,
        model_id=model_id,
        quant=quant,
        mesh_enabled=True,
    )


def test_dead_slot_adopts_successor(monkeypatch):
    successor = _entry(56)
    neuron = _neuron(monkeypatch, [successor])

    assert neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert successor in neuron._epoch_miners
    key = neuron._mesh_snapshot_cache_key(ADDRESS, 56, 43286)
    assert neuron._mesh_snapshot_cache[key] == {
        "pinned": successor.endpoint
    }
    assert neuron._shared_state_writes == 1
    assert neuron._enrichment_calls == 1
    assert neuron._admission_calls == [
        ("hotkey", 43286),
        ("mesh", 43286, True),
        ("endpoint", 43286, True),
        ("backend", 43286, True),
    ]


def test_adoption_is_capped_per_slot_epoch(monkeypatch):
    neuron = _neuron(monkeypatch, [])
    # The adoption budget is 3 attempts per (slot, epoch), not 1: a
    # successor mesh can transiently refuse its first pin seconds after
    # relaunch, and a single pre-consumed attempt turned that race into a
    # full-epoch outage (see _adopt_repin_successor).
    calls = []
    monkeypatch.setattr(
        "neurons.validator.discover_active_miners",
        lambda mc, mdl: calls.append(1) or [],
    )
    for _ in range(3):
        assert not neuron._repin_mesh_snapshot_slot(
            ADDRESS, 52, 43286, trigger="proxy request"
        )
    assert calls == [1, 1, 1]
    # The fourth attempt for the same dead slot must refuse without
    # re-running discovery.
    assert not neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert calls == [1, 1, 1]


def test_tracked_entries_are_not_readopted(monkeypatch):
    tracked = _entry(56)
    neuron = _neuron(monkeypatch, [tracked], epoch_miners=[tracked])
    # miner IS tracked for idx 56, but the request names dead idx 52 —
    # nothing new to adopt, so the core reports failure without
    # duplicating the tracked entry.
    assert not neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert neuron._epoch_miners == [tracked]


def test_non_mesh_entries_are_ignored(monkeypatch):
    vllm_entry = SimpleNamespace(
        address=ADDRESS,
        model_index=57,
        endpoint="http://x:1",
        model_id="qwen",
        quant="int4",
        mesh_enabled=False,
    )
    neuron = _neuron(monkeypatch, [vllm_entry])
    assert not neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert neuron._epoch_miners == []


def test_successor_from_unrelated_logical_model_is_ignored(monkeypatch):
    unrelated = _entry(
        57,
        model_id="qwen3.8-27b-q4-k-m",
        quant="gguf_mesh_q4_k_m",
    )
    neuron = _neuron(monkeypatch, [unrelated])

    assert not neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert neuron._epoch_miners == []


def test_successor_requant_of_same_logical_model_is_adopted(monkeypatch):
    requant = _entry(
        58,
        model_id="glm-5.2-q3-k-xl",
        quant="gguf_mesh_q3_k_xl",
    )
    neuron = _neuron(monkeypatch, [requant])

    assert neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert neuron._epoch_miners == [requant]


def test_successor_adoption_fails_closed_without_original_model(monkeypatch):
    successor = _entry(58)
    neuron = _neuron(monkeypatch, [successor])
    neuron._mesh_snapshot_cache = {}

    assert not neuron._repin_mesh_snapshot_slot(
        ADDRESS, 52, 43286, trigger="proxy request"
    )
    assert neuron._epoch_miners == []


def test_adoption_fetches_missing_chain_spec_on_demand(monkeypatch):
    """A model registered on chain after validator start has no cached
    spec; the trust-anchor path must fetch it instead of failing the
    adoption ."""

    from neurons.validator import get_mesh_model_scoring_profile

    neuron = object.__new__(ValidatorNeuron)
    neuron._on_chain_model_spec_cache = {}
    neuron._model_spec_cache = {}
    spec = SimpleNamespace(
        model_id="glm-5.2-iq2-m", quant_mode="gguf_iq2_m"
    )
    neuron._model_client = SimpleNamespace(
        get_on_chain_model_spec=lambda mid: spec
    )
    miner = _entry(58)
    # Only exercise the spec-resolution portion: stop right after by
    # feeding a spec whose quant matches the profile.
    try:
        neuron._mesh_snapshot_trust_anchors(miner, 43286)
    except ValueError as exc:
        # Downstream anchors (metagraph etc.) may legitimately fail in
        # this stub — but never on the spec.
        assert "ModelSpec is unavailable" not in str(exc)
    except AttributeError:
        pass
    assert neuron._on_chain_model_spec_cache["glm-5.2-iq2-m"] is spec
