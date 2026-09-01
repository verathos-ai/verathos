"""Focused validator scoring tests for exact GGUF-mesh model profiles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurons.runtime import (
    MESH_QUANT_PREFIX,
    get_mesh_model_scoring_profile,
)
from neurons.scoring import compute_model_base_utility
from neurons.validator import (
    ValidatorNeuron,
    _model_scoring_entry,
    _resolve_model_scoring_runtime,
)
from verallm.registry import get_model
from verallm.registry.models import MESH_GGUF_MODELS, gguf_quant_quality


MESH_MODEL_ID = "qwen2.5-7b-q4-k-m"
MESH_QUANT = "gguf_mesh_q4_k_m"


@pytest.fixture(autouse=True)
def _isolate_registry_reload(monkeypatch):
    """Keep miss-path tests from replacing process-global registry types.

    The dedicated reload test below installs its own spy over this stub. Real
    on-disk reload behavior belongs to an isolated subprocess test because a
    module reload deliberately replaces enum/dataclass identities for every
    test module already collected in this interpreter.
    """
    monkeypatch.setattr("importlib.reload", lambda module: module)


def _miner(**overrides):
    values = {
        "address": "0x00000000000000000000000000000000000000aa",
        "model_index": 0,
        "endpoint": "https://mesh.example.com",
        "model_id": MESH_MODEL_ID,
        "quant": MESH_QUANT,
        "max_context_len": 262_144,
        "mesh_enabled": True,
        "registered_at": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_every_catalogue_mesh_model_has_a_scorable_profile():
    """The owner-curated catalogue IS the approval list: every shipped mesh
    model id resolves to a profile the validator can score, including the
    mesh-only models that deliberately have no vLLM registry entry."""
    assert MESH_GGUF_MODELS, "catalogue must not be empty"
    for mesh_model_id, (entry, variant) in MESH_GGUF_MODELS.items():
        profile = get_mesh_model_scoring_profile(mesh_model_id)
        assert profile is not None, mesh_model_id
        assert profile.model_id == mesh_model_id
        assert profile.runtime_family == MESH_QUANT_PREFIX
        assert profile.registry_quant == (
            f"{MESH_QUANT_PREFIX}_{variant.gguf_scheme.lower()}"
        )
        assert profile.scoring_quant == f"gguf_{variant.gguf_scheme.lower()}"
        # The validator must be able to score it: a registry-linked logical
        # model OR the catalogue's own quality facts.
        facts = _model_scoring_entry(mesh_model_id)
        assert facts is not None, mesh_model_id
        assert facts.active_params_b > 0 or facts.moe_dense_equivalent > 0


def test_mesh_only_models_score_from_catalogue_facts():
    profile = get_mesh_model_scoring_profile("glm-5.2-iq2-m")
    assert profile is not None
    assert profile.logical_model_id == ""
    facts = profile.fallback_scoring_facts()
    assert facts is not None
    assert facts.moe_dense_equivalent == 120.0
    assert facts.generation_quality == 1.5
    # Quant quality resolves per GGUF scheme: an iq2_m must not score like
    # a q4_k_m of the same model.
    iq2 = compute_model_base_utility(
        active_params_b=facts.active_params_b,
        max_context_len=389_120,
        quant="gguf_iq2_m",
        moe_dense_equivalent=facts.moe_dense_equivalent,
        generation_quality=facts.generation_quality,
    )
    q4 = compute_model_base_utility(
        active_params_b=facts.active_params_b,
        max_context_len=389_120,
        quant="gguf_q4_k_xl",
        moe_dense_equivalent=facts.moe_dense_equivalent,
        generation_quality=facts.generation_quality,
    )
    assert iq2 < q4
    assert iq2 / q4 == (
        gguf_quant_quality("iq2_m") / gguf_quant_quality("q4_k_xl")
    )


def test_mesh_profile_lookup_is_exact_and_quant_checked():
    assert get_mesh_model_scoring_profile(MESH_MODEL_ID, MESH_QUANT) is not None
    assert get_mesh_model_scoring_profile(MESH_MODEL_ID, "GGUF_MESH_Q4_K_M") is not None
    assert get_mesh_model_scoring_profile(MESH_MODEL_ID, "gguf_mesh_q5_k_m") is None
    assert get_mesh_model_scoring_profile("QWEN2.5-7B-Q4-K-M", MESH_QUANT) is None
    assert get_mesh_model_scoring_profile(f" {MESH_MODEL_ID}", MESH_QUANT) is None
    assert get_mesh_model_scoring_profile("qwen2.5-7b-unapproved-q4-k-m", MESH_QUANT) is None


def test_mesh_scoring_runtime_scores_context_at_the_advertised_value():
    """The registered context IS the measured KV auto-fit, and validators
    canary at it, so scoring uses it unbounded (only HARD proof challenges
    are context-capped, by the proof calibration)."""
    resolved = _resolve_model_scoring_runtime(
        MESH_MODEL_ID,
        MESH_QUANT,
        262_144,
    )

    assert resolved is not None
    model_entry, context_len, quant = resolved
    assert model_entry.id == "qwen2.5-7b-instruct"
    assert context_len == 262_144
    assert quant == "gguf_q4_k_m"

    smaller = _resolve_model_scoring_runtime(MESH_MODEL_ID, MESH_QUANT, 8_192)
    assert smaller is not None and smaller[1:] == (8_192, "gguf_q4_k_m")


def test_unknown_or_mismatched_mesh_runtime_fails_closed():
    assert _resolve_model_scoring_runtime(
        "qwen2.5-7b-unapproved-q4-k-m",
        MESH_QUANT,
        32_768,
    ) is None
    assert _resolve_model_scoring_runtime(
        MESH_MODEL_ID,
        "gguf_mesh_q5_k_m",
        32_768,
    ) is None


def test_served_model_fallback_and_observed_runtime_accept_profiled_mesh():
    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    neuron._model_client = None
    neuron._epoch_miners = [
        _miner(),
        _miner(
            address="0x00000000000000000000000000000000000000bb",
            model_index=1,
            model_id="qwen2.5-7b-unapproved-q4-k-m",
        ),
    ]

    assert neuron._get_model_bucket_ids() == [MESH_MODEL_ID]
    assert neuron._observed_model_runtimes() == {
        MESH_MODEL_ID: (262_144, "gguf_q4_k_m")
    }


def test_mesh_and_vllm_variants_use_independent_runtime_family_buckets():
    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    neuron._model_client = SimpleNamespace(
        get_model_list=lambda: [
            MESH_MODEL_ID,
            "qwen2.5-7b-instruct",
            "unapproved-local-model",
        ]
    )
    neuron._epoch_miners = [_miner()]
    neuron.config = SimpleNamespace(
        demand_bonus_enabled=False,
        mesh_emission_bps=2500,
    )

    budgets = neuron._build_model_emission_budgets({})

    assert set(budgets) == {MESH_MODEL_ID, "qwen2.5-7b-instruct"}
    mesh_group = neuron._last_model_emission_groups[MESH_MODEL_ID]
    solo_group = neuron._last_model_emission_groups["qwen2.5-7b-instruct"]
    assert mesh_group == "mesh:Qwen/Qwen2.5-7B-Instruct"
    assert solo_group == "vllm:Qwen/Qwen2.5-7B-Instruct"
    assert neuron._last_model_group_budgets == {
        mesh_group: budgets[MESH_MODEL_ID],
        solo_group: budgets["qwen2.5-7b-instruct"],
    }
    assert neuron._last_model_group_shares == {
        mesh_group: pytest.approx(0.25),
        solo_group: pytest.approx(0.75),
    }


def test_zero_mesh_share_preserves_exact_vllm_family_allocation():
    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    neuron._model_client = SimpleNamespace(
        get_model_list=lambda: [
            MESH_MODEL_ID,
            "qwen2.5-7b-instruct",
            "qwen3.6-27b",
        ]
    )
    neuron._epoch_miners = []
    neuron.config = SimpleNamespace(
        demand_bonus_enabled=False,
        mesh_emission_bps=0,
    )

    neuron._build_model_emission_budgets({})

    vllm_shares = {
        group: share
        for group, share in neuron._last_model_group_shares.items()
        if group.startswith("vllm:")
    }
    assert all(not group.startswith("mesh:") for group in vllm_shares)
    assert sum(vllm_shares.values()) == pytest.approx(1.0)
    raw_vllm = {
        group: budget
        for group, budget in neuron._last_model_group_budgets.items()
        if group.startswith("vllm:")
    }
    raw_total = sum(raw_vllm.values())
    assert vllm_shares == {
        group: pytest.approx(budget / raw_total)
        for group, budget in raw_vllm.items()
    }


def test_rosterless_mesh_runtime_is_inactive_but_selectable():
    """A mesh entry now belongs to the capacity-audit universe: it joins
    the selection snapshot, but without a known signed roster it has no
    auditable obligation (not active) and the model gate stays silent."""
    from neurons.capacity_audit import CapacityAuditRuntimeConfig

    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    mesh = _miner()
    solo = _miner(
        address="0x00000000000000000000000000000000000000bb",
        endpoint="https://solo.example.com",
        model_id="qwen2.5-7b-instruct",
        quant="int4",
        max_context_len=32_768,
        mesh_enabled=False,
    )
    neuron.config = SimpleNamespace(
        chain_id=945,
        netuid=405,
        mesh_capacity_audit_enforcement_epoch=0,
    )
    neuron._capacity_audit_cfg = CapacityAuditRuntimeConfig(enabled=True)
    neuron._db = SimpleNamespace(get_uid=lambda address: 7)
    neuron._fetch_mesh_capacity_roster = lambda entry: None

    assert neuron._capacity_audit_active_slots([mesh]) == []
    selected = neuron._capacity_audit_selection_slots([mesh, solo])
    assert sorted(slot.model_id for slot in selected) == sorted(
        [mesh.model_id, solo.model_id]
    )
    neuron._subnet_runtime_config_authoritative = True
    assert neuron._capacity_audit_model_gate_reason(mesh, 12) == ""


def test_mesh_capacity_routing_also_uses_quant_when_flag_is_stale():
    """A stale mesh_enabled=False flag must not push a GGUF entry down the
    single-GPU /health path: the quant marker alone routes it through the
    roster-based mesh eligibility."""
    from neurons.capacity_audit import CapacityAuditRuntimeConfig

    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    mesh = _miner(mesh_enabled=False)
    neuron.config = SimpleNamespace(chain_id=945, netuid=405)
    neuron._capacity_audit_cfg = CapacityAuditRuntimeConfig(enabled=True)
    neuron._db = SimpleNamespace(get_uid=lambda address: 7)
    fetches = []

    def _fetch(entry):
        fetches.append(entry)
        return None

    neuron._fetch_mesh_capacity_roster = _fetch

    assert neuron._capacity_audit_active_slots([mesh]) == []
    assert fetches and fetches[0] is mesh


def test_mesh_only_quant_rungs_share_one_emission_bucket():
    """Every quant rung of a mesh-only model must group into ONE logical
    emission bucket, exactly like vLLM variants of one base model do.
    Without base_model on the fallback facts the three glm-5.2 rungs held
    three separate buckets summing to ~48% of emission ;
    which rung an operator serves is their choice, and the quant quality
    factor already prices that choice inside the shared bucket."""

    from neurons.validator import ValidatorNeuron, _model_scoring_entry

    def group(model_id: str) -> str:
        return ValidatorNeuron._model_emission_group(
            None, _model_scoring_entry(model_id)
        )

    assert (
        group("glm-5.2-iq2-m")
        == group("glm-5.2-q3-k-xl")
        == group("glm-5.2-q4-k-xl")
        == "zai-org/GLM-5.2"
    )
    # Different base snapshots are genuinely different models and must
    # keep separate buckets.
    assert group("deepseek-v4-flash-iq3-xxs") != group(
        "deepseek-v4-flash-0731-q3-k-xl"
    )
    # Mesh variants of a model WITH a vLLM registry entry keep grouping
    # through that entry's base_model (shared with the vLLM quants).
    assert group("qwen3.6-27b-q4-k-m") == "Qwen/Qwen3.6-27B"


def test_profile_miss_triggers_one_rate_limited_registry_reload(monkeypatch):
    """A registry sync landing after process start must become visible
    without a restart (process-predates-registry trap
    qwen3.8-27b-q4-k-m read "not approved" against an already-synced
    tree). A miss reloads the on-disk module once per interval; repeat
    misses inside the interval never reload again."""

    import neurons.runtime as runtime

    reloads: list[str] = []

    def fake_reload(module):
        reloads.append(module.__name__)
        return module

    monkeypatch.setattr("importlib.reload", fake_reload)
    monkeypatch.setattr(runtime, "_registry_reload_at", 0.0)

    # Unknown id: miss -> one reload attempt -> still None.
    assert runtime.get_mesh_model_scoring_profile("not-a-model") is None
    assert reloads == ["verallm.registry.models"]

    # Second miss inside the rate window: no second reload.
    assert runtime.get_mesh_model_scoring_profile("still-not-a-model") is None
    assert reloads == ["verallm.registry.models"]

    # A KNOWN id never touches the reload path.
    assert runtime.get_mesh_model_scoring_profile("glm-5.2-iq2-m") is not None
    assert reloads == ["verallm.registry.models"]
