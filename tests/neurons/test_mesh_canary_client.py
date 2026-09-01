"""Trust-boundary tests for the active private-mesh canary client."""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace

import pytest
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons import request_signing
from neurons.mesh_verify import (
    MESH_POSTCOMMIT_FINALIZATION_POLL_MAX_SECONDS,
    MESH_POSTCOMMIT_MAX_ATTEMPTS,
    MESH_CHAT_COMPLETIONS_PATH,
    MESH_POSTCOMMIT_AUDIT_PATH,
    MeshCanaryTransportError,
    _default_mesh_canary_transport,
    run_mesh_canary,
)
from verallm.mesh.proof import mesh_validator_challenge_nonce_commitment
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    MeshVerificationStage,
    sign_mesh_verification_snapshot,
)


NOW = 1_800_000_000
COORDINATOR_SEED = bytes.fromhex("11" * 32)
VALIDATOR_SEED = bytes.fromhex("22" * 32)
COORDINATOR_KEYPAIR = Keypair.create_from_seed(COORDINATOR_SEED.hex())
VALIDATOR_KEYPAIR = Keypair.create_from_seed(VALIDATOR_SEED.hex())
STAGE_KEYPAIRS = (
    Keypair.create_from_seed(("44" * 32)),
    Keypair.create_from_seed(("55" * 32)),
)


def _coordinator(**changes) -> MeshCoordinatorIdentity:
    return replace(
        MeshCoordinatorIdentity(
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            coordinator_hotkey=COORDINATOR_KEYPAIR.ss58_address,
            coordinator_evm_address="0x" + "ab" * 20,
            model_index=7,
        ),
        **changes,
    )


def _model() -> MeshModelAnchors:
    return MeshModelAnchors(
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="aa" * 32,
        model_tensor_manifest_root="bb" * 32,
        tokenizer_hash="cc" * 32,
        total_layers=2,
        max_context_len=32_768,
        quantization_scheme="gguf_q4_k_m",
        activation_dtype="f16",
    )


def _policy(**changes) -> MeshVerificationPolicy:
    return replace(
        MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format="compact-raw-v3",
            base_proof_sample_bps=10_000,
            organic_decode_sample_bps=10_000,
            canary_decode_sample_bps=10_000,
            proof_ops_per_request=3,
            deferred_proof_enabled=False,
        ),
        **changes,
    )


def _snapshot(
    *,
    coordinator: MeshCoordinatorIdentity | None = None,
    policy: MeshVerificationPolicy | None = None,
) -> MeshVerificationSnapshot:
    unsigned = MeshVerificationSnapshot(
        mesh_id="mesh-test",
        generation=3,
        epoch=12,
        issued_at_unix=NOW - 60,
        expires_at_unix=NOW + 600,
        coordinator=coordinator or _coordinator(),
        model=_model(),
        policy=policy or _policy(),
        expected_stage_count=2,
        stages=(
            MeshVerificationStage(
                stage_id="stg_" + "01" * 16,
                layer_start=0,
                layer_end=1,
                proof_key_scheme="sr25519",
                proof_key=STAGE_KEYPAIRS[0].ss58_address,
                proof_commitment="01" * 32,
            ),
            MeshVerificationStage(
                stage_id="stg_" + "02" * 16,
                layer_start=1,
                layer_end=2,
                proof_key_scheme="sr25519",
                proof_key=STAGE_KEYPAIRS[1].ss58_address,
                proof_commitment="02" * 32,
            ),
        ),
    )
    return sign_mesh_verification_snapshot(unsigned, COORDINATOR_KEYPAIR)


def _artifact(snapshot: MeshVerificationSnapshot | None = None) -> dict:
    pinned = snapshot or _snapshot()
    return {
        "receipt": {
            "receipt_hash": "ab" * 32,
            "mesh_response_commitment_hash": "bc" * 32,
            "verification_snapshot_hash": pinned.snapshot_hash_hex(),
            "proof_validator_hotkey": VALIDATOR_KEYPAIR.ss58_address,
            "prompt_token_count": 12,
            "completion_token_count": 34,
        },
        "response": {
            "usage": {"prompt_tokens": 12, "completion_tokens": 34},
            "choices": [{"message": {"content": "hello mesh"}}],
        },
    }


def _transport_for(
    artifact: dict,
    captured: list[dict] | None = None,
    *,
    final_artifact: dict | None = None,
):
    def transport(url, body, headers, timeout):
        if captured is not None:
            captured.append(
                {
                    "url": url,
                    "body": body,
                    "headers": dict(headers),
                    "timeout": timeout,
                }
            )
        selected = (
            final_artifact
            if url.endswith(MESH_POSTCOMMIT_AUDIT_PATH)
            and final_artifact is not None
            else artifact
        )
        return json.dumps(selected, sort_keys=True).encode("utf-8")

    return transport


def _run(
    *,
    snapshot: MeshVerificationSnapshot | None = None,
    expected_coordinator: MeshCoordinatorIdentity | None = None,
    transport=None,
    verify=None,
    verify_origin=None,
    nonce_factory=None,
    request_id_factory=None,
    deferred=False,
    retry_sleep=lambda _delay: None,
    postcommit_clock=None,
    wall_clock=None,
):
    pinned = snapshot or _snapshot()
    return run_mesh_canary(
        endpoint="https://coordinator.example:9338",
        model_id=pinned.model.model_id,
        messages=[{"role": "user", "content": "hi"}],
        max_new_tokens=32,
        temperature=0.0,
        timeout=7.5,
        verify_artifact_fn=verify or (lambda *args, **kwargs: True),
        verify_origin_artifact_fn=(
            verify_origin or (lambda *args, **kwargs: True)
        ),
        deferred=deferred,
        verification_snapshot=pinned,
        expected_coordinator=expected_coordinator or pinned.coordinator,
        validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
        validator_hotkey_seed=VALIDATOR_SEED,
        transport=transport or _transport_for(_artifact(pinned)),
        nonce_factory=nonce_factory,
        request_id_factory=(
            request_id_factory or (lambda: "cd" * 32)
        ),
        clock=lambda: NOW,
        wall_clock=wall_clock,
        retry_sleep=retry_sleep,
        postcommit_clock=postcommit_clock,
    )


def test_signs_exact_post_body_and_binds_canary_policy_and_snapshot():
    snapshot = _snapshot()
    captured: list[dict] = []
    origin_verify_calls: list[tuple[dict, dict, dict]] = []
    verify_calls: list[tuple[dict, dict, dict, dict]] = []

    def verify_origin(artifact, openai_request, **kwargs):
        origin_verify_calls.append((artifact, openai_request, kwargs))
        return True

    def verify(final_artifact, origin_artifact, openai_request, **kwargs):
        verify_calls.append(
            (final_artifact, origin_artifact, openai_request, kwargs)
        )
        return True

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(_artifact(snapshot), captured),
        verify=verify,
        verify_origin=verify_origin,
        nonce_factory=lambda: "de" * 32,
    )

    assert result.ok is True
    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert result.full_text == "hello mesh"
    assert result.deferred_pending is False
    assert result.ttft_ms is None
    assert result.inference_ms <= result.total_ms

    assert len(captured) == 2
    inference = captured[0]
    reveal = captured[1]
    assert inference["url"] == (
        "https://coordinator.example:9338" + MESH_CHAT_COMPLETIONS_PATH
    )
    assert reveal["url"] == (
        "https://coordinator.example:9338" + MESH_POSTCOMMIT_AUDIT_PATH
    )
    assert inference["timeout"] == reveal["timeout"] == 7.5
    body = json.loads(inference["body"])
    # Identical in shape to what the proxy signs for organic streaming
    # traffic: the OpenAI request itself, no envelope, stream on.
    assert set(body) == {
        "model", "messages", "max_tokens", "temperature", "stream",
        "do_sample", "verathos",
    }
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    policy = body["verathos"]
    assert policy == {
        "challenge_nonce_commitment": (
            mesh_validator_challenge_nonce_commitment(
                "de" * 32,
                validator_request_id="cd" * 32,
                verification_snapshot_hash=snapshot.snapshot_hash_hex(),
            )
        ),
        "validator_request_id": "cd" * 32,
        # Read from the snapshot's single decode rate, not a canary-specific
        # constant. The proxy puts the same value on organic requests, so this
        # field cannot tell the coordinator which requests are canaries.
        "decode_audit_bps": snapshot.policy.decode_sample_bps,
        "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
    }
    assert snapshot.policy.decode_sample_bps == 10_000
    assert "mesh_spec_hash" not in body
    assert "validator_nonce" not in inference["body"].decode("utf-8")
    assert ("de" * 32) not in inference["body"].decode("utf-8")
    assert "worker" not in inference["body"].decode("utf-8").lower()

    reveal_body = json.loads(reveal["body"])
    assert reveal_body == {
        "validator_request_id": "cd" * 32,
        "origin_receipt_hash": "ab" * 32,
        "mesh_response_commitment_hash": "bc" * 32,
        "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
        "challenge_nonce": "de" * 32,
    }

    headers = inference["headers"]
    ok, reason = request_signing.verify_request(
        method="POST",
        path=MESH_CHAT_COMPLETIONS_PATH,
        body=inference["body"],
        hotkey_ss58=headers[request_signing.HDR_HOTKEY],
        signature_hex=headers[request_signing.HDR_SIGNATURE],
        timestamp_str=headers[request_signing.HDR_TIMESTAMP],
    )
    assert (ok, reason) == (True, "ok")
    mutated_ok, _ = request_signing.verify_request(
        method="POST",
        path=MESH_CHAT_COMPLETIONS_PATH,
        body=inference["body"] + b" ",
        hotkey_ss58=headers[request_signing.HDR_HOTKEY],
        signature_hex=headers[request_signing.HDR_SIGNATURE],
        timestamp_str=headers[request_signing.HDR_TIMESTAMP],
    )
    assert mutated_ok is False

    reveal_headers = reveal["headers"]
    reveal_ok, reveal_reason = request_signing.verify_request(
        method="POST",
        path=MESH_POSTCOMMIT_AUDIT_PATH,
        body=reveal["body"],
        hotkey_ss58=reveal_headers[request_signing.HDR_HOTKEY],
        signature_hex=reveal_headers[request_signing.HDR_SIGNATURE],
        timestamp_str=reveal_headers[request_signing.HDR_TIMESTAMP],
    )
    assert (reveal_ok, reveal_reason) == (True, "ok")

    _, origin_verified_request, origin_kwargs = origin_verify_calls[0]
    assert origin_verified_request == body
    assert origin_kwargs["require_configured_proof"] is True
    assert origin_kwargs["require_cryptographic_proof"] is False
    assert origin_kwargs["require_coordinator_signature"] is True
    assert origin_kwargs["require_validator_request_id"] is True

    _, _, verified_request, kwargs = verify_calls[0]
    assert verified_request == body
    assert kwargs["verification_snapshot"] is snapshot
    assert kwargs["require_coordinator_signature"] is True
    assert kwargs["challenge_nonce"] == "de" * 32
    assert kwargs["expected_validator_hotkey"] == VALIDATOR_KEYPAIR.ss58_address
    assert kwargs["expected_coordinator_hotkey"] == snapshot.coordinator.coordinator_hotkey
    assert kwargs["expected_coordinator_uid"] == snapshot.coordinator.coordinator_uid


def test_every_call_uses_a_fresh_hidden_challenge_and_separate_reveal_signature():
    captured: list[dict] = []
    nonces = iter(("01" * 32, "02" * 32))
    request_ids = iter(("03" * 32, "04" * 32))
    snapshot = _snapshot()
    for _ in range(2):
        result = _run(
            snapshot=snapshot,
            transport=_transport_for(_artifact(snapshot), captured),
            nonce_factory=lambda: next(nonces),
            request_id_factory=lambda: next(request_ids),
        )
        assert result.ok

    inference_calls = [
        item for item in captured if item["url"].endswith(MESH_CHAT_COMPLETIONS_PATH)
    ]
    reveal_calls = [
        item
        for item in captured
        if item["url"].endswith(MESH_POSTCOMMIT_AUDIT_PATH)
    ]
    assert len(inference_calls) == len(reveal_calls) == 2
    requests = [
        json.loads(item["body"])
        for item in inference_calls
    ]
    assert requests[0]["verathos"]["challenge_nonce_commitment"] != (
        requests[1]["verathos"]["challenge_nonce_commitment"]
    )
    assert requests[0]["verathos"]["validator_request_id"] != (
        requests[1]["verathos"]["validator_request_id"]
    )
    assert all("validator_nonce" not in request["verathos"] for request in requests)
    reveals = [json.loads(item["body"]) for item in reveal_calls]
    assert [item["challenge_nonce"] for item in reveals] == [
        "01" * 32,
        "02" * 32,
    ]
    assert [item["validator_request_id"] for item in reveals] == [
        "03" * 32,
        "04" * 32,
    ]
    assert len(
        {
            item["headers"][request_signing.HDR_SIGNATURE]
            for item in captured
        }
    ) == 4
    for nonce in ("01" * 32, "02" * 32):
        assert sum(
            json.loads(item["body"]).get("challenge_nonce") == nonce
            for item in captured
        ) == 1


def test_canary_uses_proof_bound_counts_when_response_usage_is_missing():
    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    del artifact["response"]["usage"]

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(artifact),
    )

    assert result.ok is True
    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert "usage" not in artifact["response"]


def test_full_text_counts_reasoning_content_as_real_output():
    """A thinking-only response (llama-server --reasoning-format deepseek
    routes the whole canary budget into message.reasoning_content, content
    is empty) must not read as empty output: full_text carries the
    reasoning and the output-sanity guard passes."""

    from neurons.output_sanity import check_output_sanity

    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    reasoning = (
        "First I consider the question carefully, weighing the options "
        "before answering."
    )
    artifact["response"]["choices"][0]["message"] = {
        "content": "",
        "reasoning_content": reasoning,
    }

    result = _run(snapshot=snapshot, transport=_transport_for(artifact))

    assert result.ok is True
    assert result.full_text == reasoning
    assert (
        check_output_sanity(result.output_tokens, result.full_text, 64)
        is None
    )


def test_full_text_appends_reasoning_after_visible_content():
    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["response"]["choices"][0]["message"] = {
        "content": "hello mesh",
        "reasoning_content": "quick think",
    }

    result = _run(snapshot=snapshot, transport=_transport_for(artifact))

    assert result.ok is True
    assert result.full_text == "hello mesh\nquick think"


def test_truly_empty_response_still_fails_output_sanity():
    """No content AND no reasoning: the guard must still fail the canary."""

    from neurons.output_sanity import check_output_sanity

    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["response"]["choices"][0]["message"] = {"content": ""}

    result = _run(snapshot=snapshot, transport=_transport_for(artifact))

    assert result.ok is True
    assert result.full_text == ""
    reason = check_output_sanity(result.output_tokens, result.full_text, 64)
    assert reason is not None
    assert reason.startswith("empty_visible_output")


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 999, "completion_tokens": 34},
        {"prompt_tokens": 12, "completion_tokens": 999},
        {"prompt_tokens": "12", "completion_tokens": 34},
        {"prompt_tokens": 12, "completion_tokens": 34.0},
        {"prompt_tokens": True, "completion_tokens": 34},
        {"prompt_tokens": 12},
        None,
        {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 99},
    ],
)
def test_canary_rejects_unbound_or_malformed_response_usage(usage):
    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["response"]["usage"] = usage

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(artifact),
    )

    assert result.ok is False
    assert result.transport_error is False
    assert result.validator_error is False
    assert "proof-bound token accounting invalid" in result.reason


def test_canary_rejects_noninteger_final_receipt_token_count():
    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["receipt"]["completion_token_count"] = True

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(artifact),
    )

    assert result.ok is False
    assert result.transport_error is False
    assert result.validator_error is False
    assert "completion_token_count must be a non-negative JSON integer" in (
        result.reason
    )


def test_origin_must_verify_before_the_nonce_is_revealed():
    snapshot = _snapshot()
    captured: list[dict] = []
    result = _run(
        snapshot=snapshot,
        transport=_transport_for(_artifact(snapshot), captured),
        verify_origin=lambda *args, **kwargs: False,
        nonce_factory=lambda: "de" * 32,
    )

    assert result.ok is False
    assert result.transport_error is False
    assert "origin verification failed" in result.reason
    assert len(captured) == 1
    assert captured[0]["url"].endswith(MESH_CHAT_COMPLETIONS_PATH)
    assert ("de" * 32) not in captured[0]["body"].decode("utf-8")


def test_origin_validator_identity_must_match_before_nonce_reveal():
    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["receipt"]["proof_validator_hotkey"] = "wrong-validator"
    captured: list[dict] = []

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(artifact, captured),
        verify_origin=lambda *args, **kwargs: True,
        nonce_factory=lambda: "de" * 32,
    )

    assert result.ok is False
    assert result.transport_error is False
    assert "validator hotkey mismatch" in result.reason
    assert len(captured) == 1
    assert captured[0]["url"].endswith(MESH_CHAT_COMPLETIONS_PATH)
    assert ("de" * 32) not in captured[0]["body"].decode("utf-8")


def test_phase_two_transport_retries_exact_reveal_then_fails_proof_obligation():
    snapshot = _snapshot()
    captured: list[dict] = []

    def transport(url, body, headers, timeout):
        captured.append({"url": url, "body": body, "headers": headers})
        if url.endswith(MESH_POSTCOMMIT_AUDIT_PATH):
            raise ConnectionError("proof endpoint unavailable")
        return json.dumps(_artifact(snapshot), sort_keys=True).encode("utf-8")

    result = _run(
        snapshot=snapshot,
        transport=transport,
        nonce_factory=lambda: "de" * 32,
    )

    assert result.ok is False
    assert result.transport_error is False
    assert result.validator_error is False
    assert "proof obligation unavailable" in result.reason
    assert result.receipt["receipt_hash"] == "ab" * 32
    assert len(captured) == 1 + MESH_POSTCOMMIT_MAX_ATTEMPTS
    reveal_calls = captured[1:]
    assert all(call["body"] == reveal_calls[0]["body"] for call in reveal_calls)
    assert json.loads(reveal_calls[0]["body"])["challenge_nonce"] == "de" * 32
    assert len(
        {
            call["headers"][request_signing.HDR_TIMESTAMP]
            for call in reveal_calls
        }
    ) == MESH_POSTCOMMIT_MAX_ATTEMPTS
    assert len(
        {
            call["headers"][request_signing.HDR_SIGNATURE]
            for call in reveal_calls
        }
    ) == MESH_POSTCOMMIT_MAX_ATTEMPTS


@pytest.mark.parametrize(
    "error_code",
    [
        "postcommit_finalization_in_progress",
        "postcommit_capacity_unavailable",
    ],
)
def test_phase_two_polls_exact_reveal_while_proof_work_is_pending(error_code):
    snapshot = _snapshot()
    captured: list[dict] = []
    sleeps: list[float] = []
    poll_times = iter((10.0, 12.0, 14.0, 16.0, 18.0, 20.0))

    def transport(url, body, headers, _timeout):
        captured.append({"url": url, "body": body, "headers": dict(headers)})
        if url.endswith(MESH_CHAT_COMPLETIONS_PATH):
            return json.dumps(_artifact(snapshot), sort_keys=True).encode()
        reveal_count = sum(
            item["url"].endswith(MESH_POSTCOMMIT_AUDIT_PATH)
            for item in captured
        )
        if reveal_count <= MESH_POSTCOMMIT_MAX_ATTEMPTS:
            raise MeshCanaryTransportError(
                "proof finalization still active",
                status_code=503,
                retryable=True,
                phase="postcommit",
                error_code=error_code,
                retry_after_seconds=999.0,
            )
        return json.dumps(_artifact(snapshot), sort_keys=True).encode()

    result = _run(
        snapshot=snapshot,
        transport=transport,
        retry_sleep=sleeps.append,
        postcommit_clock=lambda: next(poll_times),
    )

    assert result.ok is True
    reveal_calls = [
        item
        for item in captured
        if item["url"].endswith(MESH_POSTCOMMIT_AUDIT_PATH)
    ]
    assert len(reveal_calls) == MESH_POSTCOMMIT_MAX_ATTEMPTS + 1
    assert all(item["body"] == reveal_calls[0]["body"] for item in reveal_calls)
    assert len(
        {
            item["headers"][request_signing.HDR_TIMESTAMP]
            for item in reveal_calls
        }
    ) == len(reveal_calls)
    assert sleeps == [MESH_POSTCOMMIT_FINALIZATION_POLL_MAX_SECONDS] * 3


def test_phase_two_finalization_poll_window_exhaustion_is_punitive(
    monkeypatch,
):
    snapshot = _snapshot()
    reveal_calls: list[bytes] = []
    sleeps: list[float] = []
    poll_times = iter((10.0, 10.9, 11.1))

    def transport(url, body, _headers, _timeout):
        if url.endswith(MESH_CHAT_COMPLETIONS_PATH):
            return json.dumps(_artifact(snapshot), sort_keys=True).encode()
        reveal_calls.append(body)
        raise MeshCanaryTransportError(
            "proof finalization still active",
            status_code=503,
            retryable=True,
            phase="postcommit",
            error_code="postcommit_finalization_in_progress",
            retry_after_seconds=0.75,
        )

    monkeypatch.setattr(
        "neurons.mesh_verify.MESH_POSTCOMMIT_FINALIZATION_WINDOW_SECONDS",
        1.0,
    )
    result = _run(
        snapshot=snapshot,
        transport=transport,
        retry_sleep=sleeps.append,
        postcommit_clock=lambda: next(poll_times),
    )

    assert result.ok is False
    assert result.transport_error is False
    assert result.validator_error is False
    assert "proof obligation unavailable" in result.reason
    assert len(reveal_calls) == 2
    assert reveal_calls[0] == reveal_calls[1]
    assert sleeps == [1.0]


def test_nonzero_temperature_is_rejected_before_transport():
    snapshot = _snapshot()
    calls: list[dict] = []
    with pytest.raises(ValueError, match="temperature=0"):
        run_mesh_canary(
            endpoint="https://coordinator.example:9338",
            model_id=snapshot.model.model_id,
            messages=[{"role": "user", "content": "hi"}],
            max_new_tokens=32,
            temperature=0.25,
            verification_snapshot=snapshot,
            expected_coordinator=snapshot.coordinator,
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            transport=_transport_for(_artifact(snapshot), calls),
            clock=lambda: NOW,
        )
    assert calls == []


def test_expected_coordinator_and_snapshot_signature_are_hard_preflight_bindings():
    calls = []
    transport = _transport_for(_artifact(), calls)
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="expected coordinator"):
        _run(
            snapshot=snapshot,
            expected_coordinator=replace(snapshot.coordinator, coordinator_uid=2),
            transport=transport,
        )
    assert calls == []

    with pytest.raises(ValueError, match="signature is invalid"):
        _run(
            snapshot=replace(snapshot, signature="00" * 64),
            transport=transport,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (_policy(base_proof_sample_bps=9_999), "base proof"),
        # Decode is 0-or-full uniform on both paths: a partial rate is
        # invalid outright, and a
        # canary-only rate would make canaries identifiable at phase one.
        (
            _policy(
                canary_decode_sample_bps=9_999,
                organic_decode_sample_bps=9_999,
            ),
            "zero or full",
        ),
        (
            _policy(
                canary_decode_sample_bps=10_000,
                organic_decode_sample_bps=0,
            ),
            "must be equal",
        ),
        (_policy(deferred_proof_enabled=True), "deferred proof"),
    ],
)
def test_snapshot_cannot_downgrade_active_canary_proof_policy(policy, reason):
    with pytest.raises(ValueError, match=reason):
        _run(snapshot=_snapshot(policy=policy))


def test_zero_decode_rate_is_a_legitimate_light_tier_policy():
    """Decode 0 (uniform) is NOT a downgrade: the light tier carries no
    decode obligation — decode verification lives in hard draws."""
    result = _run(
        snapshot=_snapshot(
            policy=_policy(
                canary_decode_sample_bps=0,
                organic_decode_sample_bps=0,
            )
        )
    )
    assert result.ok is True


def test_served_malformed_or_unverifiable_artifacts_are_proof_failures():
    def malformed_transport(*args):
        return b'{"receipt":'

    malformed = _run(transport=malformed_transport)
    assert malformed.ok is False
    assert malformed.transport_error is False
    assert "response invalid" in malformed.reason

    missing = _run(transport=_transport_for({"response": {}}))
    assert missing.ok is False
    assert missing.transport_error is False
    assert "missing receipt" in missing.reason

    rejected = _run(verify_origin=lambda *args, **kwargs: False)
    assert rejected.ok is False
    assert rejected.transport_error is False
    assert "origin verification failed" in rejected.reason

    snapshot = _snapshot()

    def malformed_final_transport(url, *args):
        if url.endswith(MESH_POSTCOMMIT_AUDIT_PATH):
            return b'{"receipt":'
        return json.dumps(_artifact(snapshot)).encode("utf-8")

    malformed_final = _run(
        snapshot=snapshot,
        transport=malformed_final_transport,
    )
    assert malformed_final.ok is False
    assert malformed_final.transport_error is False
    assert "postcommit proof response invalid" in malformed_final.reason

    rejected_final = _run(verify=lambda *args, **kwargs: False)
    assert rejected_final.ok is False
    assert rejected_final.transport_error is False
    assert "postcommit artifact verification failed" in rejected_final.reason


def test_transport_failure_is_distinct_from_served_proof_failure():
    def fail(*args):
        raise ConnectionError("coordinator unavailable")

    result = _run(transport=fail)
    assert result.ok is False
    assert result.transport_error is True
    assert result.transport_retryable is True
    assert result.transport_phase == "inference"
    assert "request failed" in result.reason


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (401, False), (408, True), (429, True), (500, True), (503, True)],
)
def test_urllib_status_adapter_has_explicit_retry_semantics(
    monkeypatch,
    status_code,
    retryable,
):
    def reject(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://coordinator.example/v1/mesh/inference",
            status_code,
            "rejected",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)

    with pytest.raises(MeshCanaryTransportError) as exc_info:
        _default_mesh_canary_transport(
            "https://coordinator.example/v1/mesh/inference",
            b"{}",
            {"content-type": "application/json"},
            1.0,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is retryable


def test_urllib_status_adapter_preserves_bounded_finalization_metadata(
    monkeypatch,
):
    payload = json.dumps(
        {"error_code": "postcommit_finalization_in_progress"}
    ).encode()

    def reject(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://coordinator.example/v1/mesh/proof/postcommit-audit",
            503,
            "busy",
            {"Retry-After": "9999"},
            io.BytesIO(payload),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)

    with pytest.raises(MeshCanaryTransportError) as exc_info:
        _default_mesh_canary_transport(
            "https://coordinator.example/v1/mesh/proof/postcommit-audit",
            b"{}",
            {"content-type": "application/json"},
            1.0,
        )

    assert exc_info.value.error_code == "postcommit_finalization_in_progress"
    assert exc_info.value.retry_after_seconds == (
        MESH_POSTCOMMIT_FINALIZATION_POLL_MAX_SECONDS
    )


def test_only_explicit_local_verifier_fault_is_indeterminate():
    def local_failure(*_args, **_kwargs):
        from neurons.mesh_verify import MeshValidatorVerificationError

        raise MeshValidatorVerificationError(
            "trusted validator proof executor unavailable"
        )

    local = _run(verify=local_failure)
    assert local.ok is False
    assert local.validator_error is True
    assert local.transport_error is False

    for failure in (
        MemoryError("miner-triggered verifier allocation failure"),
        OSError("miner-triggered native verifier failure"),
        SystemError("miner-triggered native verifier failure"),
        RuntimeError("tampered stage proof"),
    ):
        invalid = _run(
            verify=lambda *_args, failure=failure, **_kwargs: (
                _ for _ in ()
            ).throw(failure)
        )
        assert invalid.ok is False
        assert invalid.validator_error is False
        assert invalid.transport_error is False


def test_nested_local_fault_label_cannot_make_outer_protocol_error_indeterminate():
    def chained_failure(*_args, **_kwargs):
        from neurons.mesh_verify import MeshValidatorVerificationError

        try:
            raise MeshValidatorVerificationError("spoofed nested local label")
        except MeshValidatorVerificationError as exc:
            raise RuntimeError("invalid miner proof") from exc

    invalid = _run(verify=chained_failure)

    assert invalid.ok is False
    assert invalid.validator_error is False
    assert invalid.transport_error is False
    assert "invalid miner proof" in invalid.reason


def test_origin_import_fault_stops_before_reveal_and_is_indeterminate():
    captured: list[dict] = []

    def local_failure(*_args, **_kwargs):
        from neurons.mesh_verify import MeshValidatorVerificationError

        raise MeshValidatorVerificationError(
            "trusted native verifier import unavailable"
        )

    result = _run(
        transport=_transport_for(_artifact(), captured),
        verify_origin=local_failure,
    )

    assert result.validator_error is True
    assert result.transport_error is False
    assert len(captured) == 1
    assert captured[0]["url"].endswith(MESH_CHAT_COMPLETIONS_PATH)


def test_phase_one_latency_excludes_postcommit_proof_time(monkeypatch):
    ticks = iter((10.0, 10.25, 11.5, 12.0))
    wall_ticks = iter((1_800_000_100.0, 1_800_000_100.25))
    monkeypatch.setattr("neurons.mesh_verify.time.monotonic", lambda: next(ticks))

    result = _run(wall_clock=lambda: next(wall_ticks))

    assert result.ok is True
    assert result.inference_ms == pytest.approx(250.0)
    assert result.phase_one_start_ts == pytest.approx(1_800_000_100.0)
    assert result.phase_one_end_ts == pytest.approx(1_800_000_100.25)
    assert result.total_ms == pytest.approx(2000.0)
    assert result.ttft_ms is None


def test_deferred_path_is_not_selectable_and_deferred_receipt_cannot_downgrade_verify():
    with pytest.raises(ValueError, match="disabled"):
        _run(deferred=True)

    snapshot = _snapshot()
    artifact = _artifact(snapshot)
    artifact["receipt"]["proof_deferred_obligation"] = True
    observed = {}

    def reject_deferred(
        _final_artifact,
        _origin_artifact,
        _request,
        **kwargs,
    ):
        observed.update(kwargs)
        raise RuntimeError("inline proof missing")

    result = _run(
        snapshot=snapshot,
        transport=_transport_for(artifact),
        verify=reject_deferred,
    )
    assert result.ok is False
    assert result.deferred_pending is False
    assert observed["challenge_nonce"]
    assert "inline proof missing" in result.reason


def test_client_has_no_mesh_spec_or_private_worker_endpoint_dependency():
    snapshot = _snapshot()
    captured: list[dict] = []
    result = _run(
        snapshot=snapshot,
        transport=_transport_for(_artifact(snapshot), captured),
    )
    assert result.ok
    assert len(captured) == 2
    assert all(
        item["url"].startswith("https://coordinator.example:9338/")
        for item in captured
    )
    serialized = "\n".join(
        item["body"].decode("utf-8") for item in captured
    )
    for forbidden in ("mesh_spec", "rpc_endpoints", "worker_endpoint", "health"):
        assert forbidden not in serialized
