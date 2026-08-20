"""LIGHT-tier sampling parity: sampler params are bound, not rejected.

The greedy pin on the mesh GGUF lane protected two things: the legacy
free-run replay (superseded by teacher-forced slot-view audits) and the
hard tier's exact-argmax binding. The light relation only verifies the
realized token against the captured top-k of the RAW logits, so a
sampler whose support stays inside that top-k and whose filters never
reorder raw logits is verifiable as-is; the applied controls are
committed in the receipt (the mesh analogue of the vLLM route's
sampler_config_hash substitution guard).
"""

from __future__ import annotations

import pytest

from verallm.mesh.proof import (
    VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
    derive_mesh_replay_seed,
)
from verallm.mesh.worker import (
    VERIFIED_GGUF_SAMPLED_LIGHT_MODE,
    VERIFIED_GGUF_SAMPLER_CONTROLS,
    VERIFIED_GGUF_SAMPLER_MODE,
    backend_openai_request,
    finalize_verified_sampler_policy,
    sampled_controls_from_context,
    sampled_light_controls_from_request,
    verified_gguf_sampler_controls_hash,
)


def _request(**overrides):
    return {
        "model": "glm-5.2-iq2-m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
        **overrides,
    }


def test_neutral_requests_stay_on_the_greedy_profile():
    assert sampled_light_controls_from_request(_request(), seed=7) is None
    assert (
        sampled_light_controls_from_request(
            _request(temperature=0, top_k=1, top_p=1, seed=0), seed=7
        )
        is None
    )
    # temperature 0 with an explicit seed is still greedy.
    assert (
        sampled_light_controls_from_request(
            _request(temperature=0, seed=1234), seed=7
        )
        is None
    )


def test_sampled_profile_is_bounded_ordered_and_seeded():
    controls = sampled_light_controls_from_request(
        _request(temperature=0.8, top_p=0.95), seed=42
    )
    assert controls is not None
    assert controls["temperature"] == 0.8
    assert controls["top_p"] == 0.95
    # No explicit top_k: default to the decode-audit width, never wider.
    assert controls["top_k"] == VERATHOS_GGUF_DECODE_AUDIT_TOP_K
    # Support restriction before temperature; no reordering samplers.
    assert controls["samplers"] == ["top_k", "top_p", "min_p", "temperature"]
    assert controls["seed"] == 42
    assert controls["repeat_penalty"] == 1.0
    assert controls["presence_penalty"] == 0.0
    # An over-wide request clamps (the committed profile is what ran).
    wide = sampled_light_controls_from_request(
        _request(temperature=1.0, top_k=4096), seed=1
    )
    assert wide["top_k"] == VERATHOS_GGUF_DECODE_AUDIT_TOP_K
    narrow = sampled_light_controls_from_request(
        _request(temperature=1.0, top_k=4), seed=1
    )
    assert narrow["top_k"] == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.8, "presence_penalty": 1.5},
        {"temperature": 0.8, "repeat_penalty": 1.2},
        {"temperature": 0.8, "logit_bias": {"5": 10}},
        {"temperature": 0.8, "grammar": "root ::= x"},
        {"temperature": 0.8, "response_format": {"type": "json_object"}},
        {"temperature": 0.8, "n": 2},
        {"temperature": 9.9},
    ],
)
def test_reordering_and_transform_controls_stay_rejected(overrides):
    """Penalties and post-logits transforms can push the realized token
    outside the captured top-k its own opening must prove; they stay
    pinned even on the sampled lane."""
    with pytest.raises(ValueError):
        sampled_light_controls_from_request(_request(**overrides), seed=1)


def test_backend_request_applies_the_committed_profile_exactly():
    controls = sampled_light_controls_from_request(
        _request(temperature=0.7), seed=99
    )
    request = backend_openai_request(
        _request(temperature=0.7),
        proof_capture_required=True,
        verified_sampler_required=True,
        sampled_profile=controls,
    )
    for key, value in controls.items():
        assert request[key] == value
    # Without a profile the greedy override still wins.
    greedy = backend_openai_request(
        _request(),
        proof_capture_required=True,
        verified_sampler_required=True,
    )
    for key, value in VERIFIED_GGUF_SAMPLER_CONTROLS.items():
        assert greedy[key] == value


def test_policy_finalizer_commits_mode_and_controls():
    policy = {
        "verified_sampler_required": True,
        "proof_postcommit_hard_bps": 0,
    }
    request_id = "req-123"
    controls = finalize_verified_sampler_policy(
        policy, _request(temperature=0.9), request_id
    )
    assert policy["verified_sampler_mode"] == VERIFIED_GGUF_SAMPLED_LIGHT_MODE
    assert policy["verified_sampler_controls"] == controls
    # No client seed: the derived replay seed pins the draw.
    assert controls["seed"] == derive_mesh_replay_seed(request_id) & 0x7FFFFFFF
    # The committed hash differs from the greedy constant and matches the
    # committed controls: the substitution guard.
    assert verified_gguf_sampler_controls_hash(controls) != (
        verified_gguf_sampler_controls_hash()
    )
    # Greedy requests keep the greedy mode.
    greedy_policy = {
        "verified_sampler_required": True,
        "proof_postcommit_hard_bps": 0,
    }
    assert (
        finalize_verified_sampler_policy(greedy_policy, _request(), "req-9")
        is None
    )
    assert greedy_policy["verified_sampler_mode"] == VERIFIED_GGUF_SAMPLER_MODE


def test_policy_finalizer_refuses_hard_lanes():
    """The exact-argmax hard relation has no sampled counterpart: an
    explicit hard demand and any lane that can DRAW hard audits both
    refuse sampled profiles instead of serving an unverifiable one."""
    hard_request = _request(temperature=0.9, verathos={"proof_tier": "hard"})
    with pytest.raises(ValueError, match="hard-tier"):
        finalize_verified_sampler_policy(
            {"verified_sampler_required": True, "proof_postcommit_hard_bps": 0},
            hard_request,
            "req-1",
        )
    with pytest.raises(ValueError, match="hard-audit rate"):
        finalize_verified_sampler_policy(
            {
                "verified_sampler_required": True,
                "proof_postcommit_hard_bps": 10_000,
            },
            _request(temperature=0.9),
            "req-1",
        )


def test_sampled_context_round_trip_and_missing_controls():
    policy = {
        "verified_sampler_required": True,
        "proof_postcommit_hard_bps": 0,
    }
    controls = finalize_verified_sampler_policy(
        policy, _request(temperature=0.6), "req-x"
    )
    assert sampled_controls_from_context(policy) == controls
    assert (
        sampled_controls_from_context(
            {"verified_sampler_mode": VERIFIED_GGUF_SAMPLER_MODE}
        )
        is None
    )
    with pytest.raises(RuntimeError, match="missing its committed"):
        sampled_controls_from_context(
            {"verified_sampler_mode": VERIFIED_GGUF_SAMPLED_LIGHT_MODE}
        )
