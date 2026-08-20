"""Fail-closed request semantics for the currently deterministic GGUF proof."""

from __future__ import annotations

import pytest

from verallm.mesh.worker import (
    VERIFIED_GGUF_SAMPLER_CONTROLS,
    backend_openai_request,
    validate_verified_gguf_sampler_request,
)


def _request(**changes):
    request = {
        "model": "qwen2.5-7b-q4-k-m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    request.update(changes)
    return request


def test_omitted_sampler_controls_select_the_declared_verified_profile():
    request = _request(verathos={"validator_nonce": "ab" * 32})

    backend = backend_openai_request(
        request,
        verified_sampler_required=True,
        proof_metadata_required=True,
    )

    assert "verathos" not in backend
    for name, value in VERIFIED_GGUF_SAMPLER_CONTROLS.items():
        assert backend[name] == value
    assert backend["return_tokens"] is True
    assert backend["verbose"] is True


def test_explicit_neutral_controls_are_accepted():
    request = _request(**VERIFIED_GGUF_SAMPLER_CONTROLS)

    validate_verified_gguf_sampler_request(request)


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("temperature", 0.7),
        ("top_k", 40),
        ("top_p", 0.9),
        ("presence_penalty", 0.2),
        ("do_sample", True),
        ("samplers", ["top_k", "temperature"]),
        ("logit_bias", {"42": 1.0}),
        ("grammar", "root ::= 'yes'"),
        ("response_format", {"type": "json_object"}),
        ("n", 2),
    ],
)
def test_noncanonical_sampling_controls_fail_before_backend_forward(
    control, value
):
    with pytest.raises(ValueError, match="verified GGUF sampler"):
        backend_openai_request(
            _request(**{control: value}),
            verified_sampler_required=True,
        )


def test_noncanonical_controls_remain_available_when_no_decode_proof_is_used():
    backend = backend_openai_request(
        _request(temperature=0.7, top_p=0.9),
        verified_sampler_required=False,
    )

    assert backend["temperature"] == 0.7
    assert backend["top_p"] == 0.9
