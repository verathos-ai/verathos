"""Postcommit audit tier draw: light/hard resolution and the hard demand.

The tier is decided AFTER the origin receipt froze the response, from the
nonce-derived beacon against the signed hard-audit rate riding the receipt
(v3's postcommit audit-tier draw). These tests pin: the draw's determinism
and rate behavior, the stricter-only force_hard demand, the receipt stamping,
the verifier's rejection of a light answer to a hard demand, and the canary
scheduler's exactly-one-hard-slot-per-mesh-miner assignment.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.verallm.test_mesh_postcommit import (
    _CHALLENGE_NONCE,
    _origin_artifact,
)
from verallm.mesh.proof import (
    mesh_proof_gate_hash,
    mesh_receipt_hash,
    mesh_response_commitment_hash,
)
from verallm.mesh.verification_snapshot import MeshVerificationPolicy
from verallm.mesh.worker import (
    postcommit_audit_context_from_receipt,
    postcommit_audit_decision,
    verify_mesh_postcommit_artifact,
)


def _tiered_origin(*, hard_bps: int, proof_bps: int = 10_000) -> dict:
    origin = _origin_artifact()
    receipt = origin["receipt"]
    receipt["proof_sample_bps"] = int(proof_bps)
    receipt["proof_postcommit_hard_bps"] = int(hard_bps)
    # Field mutations must re-derive the committed hashes in the same order
    # the production receipt builder does.
    receipt["proof_gate_hash"] = mesh_proof_gate_hash(receipt)
    receipt["mesh_response_commitment_hash"] = mesh_response_commitment_hash(
        receipt
    )
    receipt["receipt_hash"] = mesh_receipt_hash(receipt)
    return origin


def _policy(**overrides) -> MeshVerificationPolicy:
    fields = dict(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1024,
        deferred_proof_enabled=False,
    )
    fields.update(overrides)
    return MeshVerificationPolicy(**fields)


def test_policy_round_trips_hard_audit_rate() -> None:
    policy = _policy(postcommit_hard_audit_bps=250)
    data = policy.to_dict()
    assert data["postcommit_hard_audit_bps"] == 250
    assert MeshVerificationPolicy.from_dict(data) == policy


def test_policy_defaults_absent_rate_to_full_hard() -> None:
    data = _policy().to_dict()
    del data["postcommit_hard_audit_bps"]
    parsed = MeshVerificationPolicy.from_dict(data)
    assert parsed.postcommit_hard_audit_bps == 10_000


def test_decision_rate_zero_resolves_light_with_no_obligation() -> None:
    origin = _tiered_origin(hard_bps=0)
    decision = postcommit_audit_decision(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    assert decision["audit_tier"] == "light"
    assert decision["proof_sampled"] is False
    # v3 shape: a light draw owes nothing at reveal time - the origin
    # receipt with its inline light proof IS the light tier.
    assert decision["proof_required"] is False


def test_decision_rate_full_resolves_hard() -> None:
    origin = _tiered_origin(hard_bps=10_000)
    decision = postcommit_audit_decision(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    assert decision["audit_tier"] == "hard"
    assert decision["proof_sampled"] is True


def test_decision_is_deterministic_for_a_fixed_reveal() -> None:
    origin = _tiered_origin(hard_bps=5_000)
    first = postcommit_audit_decision(
        origin["receipt"], origin["request"], challenge_nonce=_CHALLENGE_NONCE
    )
    second = postcommit_audit_decision(
        origin["receipt"], origin["request"], challenge_nonce=_CHALLENGE_NONCE
    )
    assert first == second


def test_force_hard_overrides_a_light_draw_never_the_reverse() -> None:
    origin = _tiered_origin(hard_bps=0)
    forced = postcommit_audit_decision(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
        force_hard=True,
    )
    assert forced["audit_tier"] == "hard"
    assert forced["proof_sampled"] is True
    # There is no light override parameter at all; the API admits only the
    # stricter direction.
    with pytest.raises(TypeError):
        postcommit_audit_decision(
            origin["receipt"],
            origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            force_light=True,
        )


def test_context_stamps_the_resolved_tier() -> None:
    origin = _tiered_origin(hard_bps=0)
    ctx = postcommit_audit_context_from_receipt(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    assert ctx["proof_audit_tier"] == "light"
    hard_ctx = postcommit_audit_context_from_receipt(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
        force_hard=True,
    )
    assert hard_ctx["proof_audit_tier"] == "hard"


def test_verifier_rejects_light_answer_to_a_hard_demand() -> None:
    # Miner resolves the draw (light); the validator demanded hard. The
    # recomputed expected context then disagrees on the tier fields before
    # any payload verification happens.
    origin = _tiered_origin(hard_bps=0)
    light_ctx = postcommit_audit_context_from_receipt(
        origin["receipt"],
        origin["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    final_artifact = {
        "response": deepcopy(origin["response"]),
        "receipt": light_ctx,
    }
    origin_artifact = {
        "response": origin["response"],
        "receipt": origin["receipt"],
    }
    with pytest.raises(RuntimeError, match="proof_sampled|proof_audit_tier"):
        verify_mesh_postcommit_artifact(
            final_artifact,
            origin_artifact,
            origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            audit_tier="hard",
        )


def test_verifier_rejects_malformed_tier_demand() -> None:
    origin = _tiered_origin(hard_bps=0)
    with pytest.raises(RuntimeError, match="audit tier demand"):
        verify_mesh_postcommit_artifact(
            {"response": {}, "receipt": {}},
            {"response": origin["response"], "receipt": origin["receipt"]},
            origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            audit_tier="light",
        )


class _Miner:
    def __init__(self, address: str, *, mesh: bool) -> None:
        self.address = address
        self.endpoint = f"https://{address}.example:9443"
        self.model_id = "mesh-model" if mesh else "vllm-model"
        self.model_index = 3
        self.max_context_len = 32_768
        self.quant = "gguf_mesh_q4_k_m" if mesh else "gptq_int4"
        self.mesh_enabled = mesh
        self.tee_enabled = False


def test_scheduler_flags_mesh_lanes_without_premarking_tiers() -> None:
    """The signed policy engine owns all canary scheduling.

    Mesh endpoints are planned exactly like every other endpoint; the
    planner only flags them for the mesh execution lane.  The postcommit
    hard demand is no longer a pre-marked slot: it derives from the policy
    engine's hidden hard draw (``verify_proof``) at dispatch time.
    """

    from neurons.canary import CanaryScheduler

    scheduler = CanaryScheduler(
        epoch_number=42,
        epoch_start_block=1000,
        epoch_blocks=360,
        validator_hotkey="validator-hotkey",
        validator_seed=b"\x07" * 32,
        small_count=2,
        full_context_count=1,
    )
    mesh_a = _Miner("mesh-a", mesh=True)
    mesh_b = _Miner("mesh-b", mesh=True)
    vllm = _Miner("vllm-c", mesh=False)
    tests = scheduler.plan_epoch([mesh_a, mesh_b, vllm])

    for miner in (mesh_a, mesh_b):
        flagged = [
            test
            for test in tests
            if test.miner_address == miner.address
        ]
        assert flagged and all(test.verify_mesh for test in flagged)
        assert all(test.mesh_audit_tier == "" for test in flagged)
    assert all(
        not test.verify_mesh and test.mesh_audit_tier == ""
        for test in tests
        if test.miner_address == vllm.address
    )


def test_mesh_dispatch_maps_policy_hard_draw_to_hard_audit_tier() -> None:
    """``verify_proof`` from the policy engine demands the mesh HARD tier."""

    from types import SimpleNamespace

    import pytest as _pytest

    import neurons.mesh_verify as mesh_verify
    from neurons.canary import CanaryTest
    from neurons.mesh_verify import MeshValidatorVerificationError
    from neurons.validator import ValidatorNeuron

    def _mesh_test(verify_proof: bool) -> CanaryTest:
        return CanaryTest(
            miner_address="0xabc",
            miner_endpoint="https://mesh-a.example:9443",
            model_id="mesh-model",
            model_index=3,
            max_context_len=32_768,
            target_block=1,
            test_index=0,
            test_type="small",
            prompt="hello",
            max_new_tokens=8,
            temperature=0.0,
            verify_proof=verify_proof,
            verify_mesh=True,
            obligation_id="11" * 16,
        )

    captured: list[str] = []

    def fake_run_mesh_canary(**kwargs):
        captured.append(kwargs["audit_tier"])
        return SimpleNamespace(
            transport_error=False,
            transport_phase="",
            validator_error=True,
            reason="stop after capturing the audit tier",
        )

    original = mesh_verify.run_mesh_canary
    mesh_verify.run_mesh_canary = fake_run_mesh_canary
    try:
        neuron = ValidatorNeuron.__new__(ValidatorNeuron)
        neuron._running = True
        neuron._current_epoch = 42
        neuron.config = SimpleNamespace(
            canary_inference_timeout=30,
            canary_full_context_inference_timeout=60,
        )
        neuron._validator_hotkey_ss58 = "validator-hotkey"
        neuron._validator_private_key = b"\x01" * 32
        key = neuron._mesh_snapshot_cache_key("0xabc", 3, 42)
        neuron._mesh_snapshot_cache = {key: object()}
        neuron._epoch_miners = [
            SimpleNamespace(
                address="0xabc",
                model_index=3,
                endpoint="https://mesh-a.example:9443",
                model_id="mesh-model",
            )
        ]
        neuron._mesh_snapshot_trust_anchors = (
            lambda _miner, _epoch: SimpleNamespace(coordinator=object())
        )

        for verify_proof, expected_tier in ((True, "hard"), (False, "")):
            with _pytest.raises(MeshValidatorVerificationError):
                neuron._execute_mesh_canary(
                    _mesh_test(verify_proof),
                    42,
                )
            assert captured[-1] == expected_tier
    finally:
        mesh_verify.run_mesh_canary = original


def _tiered_origin_with_decode(*, hard_bps: int) -> dict:
    origin = _tiered_origin(hard_bps=hard_bps)
    receipt = origin["receipt"]
    receipt["decode_audit_bps"] = 10_000
    receipt["proof_gate_hash"] = mesh_proof_gate_hash(receipt)
    receipt["mesh_response_commitment_hash"] = mesh_response_commitment_hash(
        receipt
    )
    receipt["receipt_hash"] = mesh_receipt_hash(receipt)
    return origin


def test_decode_audit_rides_the_hard_tier_only() -> None:
    #the protocol is light or hard, nothing
    # else. A light draw carries no decode-audit obligation - the hard
    # relation already proves the decode - so nothing on a light draw can
    # fail an honest serve (regression: a light-draw decode audit
    # false-failed an honest hybrid-model serve via replay-probe state
    # divergence and probated it from one public chat request).
    light = _tiered_origin_with_decode(hard_bps=0)
    decision = postcommit_audit_decision(
        light["receipt"],
        light["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    assert decision["audit_tier"] == "light"
    assert decision["decode_audit_sampled"] is False
    assert decision["proof_required"] is False

    hard = _tiered_origin_with_decode(hard_bps=10_000)
    decision = postcommit_audit_decision(
        hard["receipt"],
        hard["request"],
        challenge_nonce=_CHALLENGE_NONCE,
    )
    assert decision["audit_tier"] == "hard"
    assert decision["decode_audit_sampled"] is True
