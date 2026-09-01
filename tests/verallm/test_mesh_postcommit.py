"""Focused security tests for the two-phase mesh proof challenge."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from copy import deepcopy
from http.client import HTTPResponse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from eth_account import Account
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    build_signing_message,
)
from tests.verallm.test_mesh_control import (
    _fake_openai_backend,
    _gguf_manifest_for_traces,
    _shutdown_server,
    _write_decode_lm_head_trace,
    _write_ggml_mul_mat_trace,
)
from verallm.mesh import CapabilityAd, MeshMember, MeshSpec, StageRange
from verallm.mesh.http_auth import sign_internal_http_request
from verallm.mesh.proof import (
    PROOF_SAMPLE_BPS_DENOMINATOR,
    VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
    mesh_proof_gate_hash,
    mesh_receipt_hash,
    mesh_response_commitment_hash,
    mesh_validator_challenge_nonce_commitment,
)
from verallm.mesh.receipt_signing import (
    sign_receipt_hash,
    sign_stage_proof_receipt_body_hash,
)
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
)
from verallm.mesh.worker import (
    POSTCOMMIT_AUDIT_PATH,
    finalize_postcommit_audit_context,
    postcommit_audit_context_from_receipt,
    postcommit_audit_decision,
    semantic_openai_response_hash,
    serve_worker_in_thread,
    verify_mesh_postcommit_artifact,
    verify_mesh_proof_sampling_fields,
)


_REQUEST_ID = "11" * 32
_SNAPSHOT_HASH = "22" * 32
_CHALLENGE_NONCE = "33" * 32
_VALIDATOR_HOTKEY = "validator-hotkey-a"


def _request(
    *,
    challenge_nonce: str = _CHALLENGE_NONCE,
    request_id: str = _REQUEST_ID,
    snapshot_hash: str = _SNAPSHOT_HASH,
    stream: bool = False,
) -> dict:
    return {
        "model": "mesh-model",
        "messages": [{"role": "user", "content": "prove this response"}],
        "stream": bool(stream),
        "verathos": {
            "verification_snapshot_hash": snapshot_hash,
            "challenge_nonce_commitment": (
                mesh_validator_challenge_nonce_commitment(
                    challenge_nonce,
                    validator_request_id=request_id,
                    verification_snapshot_hash=snapshot_hash,
                )
            ),
            "validator_request_id": request_id,
        },
    }


def _response(*, content: str = "verified") -> dict:
    return {
        "id": "chatcmpl-postcommit",
        "object": "chat.completion",
        "created": 1,
        "model": "mesh-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
        },
    }


def _origin_artifact(
    *,
    request: dict | None = None,
    response: dict | None = None,
    validator_hotkey: str = _VALIDATOR_HOTKEY,
    frozen_marker: str = "44" * 32,
) -> dict:
    request = deepcopy(request or _request())
    response = deepcopy(response or _response())
    policy = request["verathos"]
    receipt = {
        "version": 1,
        "request_id": policy["validator_request_id"],
        "mesh_id": "mesh-postcommit-test",
        "mesh_spec_hash": "55" * 32,
        "stage_assignment_hash": "66" * 32,
        "rpc_plan_hash": "77" * 32,
        "model_package_hash": "88" * 32,
        "model_tensor_manifest_root": "99" * 32,
        "rpc_endpoints": [],
        "runtime": "llama_cpp_rpc",
        "uid": 1,
        "hotkey": "coordinator-hotkey",
        "endpoint": "https://coordinator.example",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 4,
        "request_hash": hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "response_hash": hashlib.sha256(
            json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "semantic_response_hash": semantic_openai_response_hash(response),
        "verification_snapshot_hash": policy["verification_snapshot_hash"],
        "proof_policy_version": 2,
        "proof_configured_required": True,
        "proof_capture_required": True,
        "proof_metadata_required": False,
        "proof_sample_bps": 0,
        "proof_sample_denominator": PROOF_SAMPLE_BPS_DENOMINATOR,
        "proof_ops_per_request": 1,
        "proof_trace_candidates_per_request": 1024,
        "proof_challenge_kind": VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
        "proof_challenge_nonce_commitment": policy[
            "challenge_nonce_commitment"
        ],
        "proof_validator_hotkey": validator_hotkey,
        "proof_postcommit": True,
        "proof_postcommit_origin_receipt_hash": "",
        "proof_postcommit_challenge_nonce": "",
        "proof_postcommit_finalized": False,
        "proof_deferred": False,
        "proof_required": False,
        "proof_sampled": False,
        "proof_sample_value": -1,
        "proof_beacon": "",
        "proof_receipt_root": "",
        "proof_receipt_count": 0,
        "proof_receipt_verified": False,
        "proof_op_manifest_root": frozen_marker,
        "proof_op_manifest_count": 1,
        "proof_trace_scope": "op_manifest_challenge_v1",
        "decode_audit_bps": 0,
        "decode_audit_required": False,
        "decode_audit_sampled": False,
        "decode_audit_sample_value": -1,
        "decode_audit_positions": [],
        "verified": False,
    }
    receipt["proof_gate_hash"] = mesh_proof_gate_hash(receipt)
    receipt["mesh_response_commitment_hash"] = mesh_response_commitment_hash(
        receipt
    )
    receipt["receipt_hash"] = mesh_receipt_hash(receipt)
    return {"request": request, "response": response, "receipt": receipt}


def _final_artifact(origin: dict, *, challenge_nonce: str = _CHALLENGE_NONCE) -> dict:
    context = postcommit_audit_context_from_receipt(
        origin["receipt"],
        origin["request"],
        challenge_nonce=challenge_nonce,
    )
    receipt = finalize_postcommit_audit_context(
        context,
        proof_receipts=[],
        proof_payloads=[],
    )
    return {"response": deepcopy(origin["response"]), "receipt": receipt}


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("verathos", "validator_nonce"),
        ("verathos", "nonce"),
        ("top", "validator_nonce"),
    ],
)
def test_phase_one_verifier_rejects_all_raw_nonce_aliases(
    location: str,
    field: str,
) -> None:
    origin = _origin_artifact()
    exposed = deepcopy(origin["request"])
    if location == "verathos":
        exposed["verathos"][field] = _CHALLENGE_NONCE
    else:
        exposed[field] = _CHALLENGE_NONCE

    with pytest.raises(
        RuntimeError,
        match="phase-one request exposed the validator nonce",
    ):
        verify_mesh_proof_sampling_fields(origin["receipt"], exposed)


def test_postcommit_reveal_is_bound_to_the_hidden_commitment() -> None:
    origin = _origin_artifact()

    with pytest.raises(
        RuntimeError,
        match="challenge nonce reveal mismatch",
    ):
        postcommit_audit_decision(
            origin["receipt"],
            origin["request"],
            challenge_nonce="aa" * 32,
        )


def test_postcommit_rejects_snapshot_substitution() -> None:
    origin = _origin_artifact()
    substituted = deepcopy(origin["request"])
    substituted["verathos"]["verification_snapshot_hash"] = "ab" * 32

    with pytest.raises(RuntimeError, match="verification snapshot mismatch"):
        postcommit_audit_decision(
            origin["receipt"],
            substituted,
            challenge_nonce=_CHALLENGE_NONCE,
        )


def test_postcommit_rejects_wrong_validator_hotkey(monkeypatch) -> None:
    import verallm.mesh.worker as mesh_worker

    origin = _origin_artifact()
    final = _final_artifact(origin)
    monkeypatch.setattr(
        mesh_worker,
        "verify_mesh_inference_artifact",
        lambda *args, **kwargs: True,
    )

    with pytest.raises(RuntimeError, match="origin validator hotkey mismatch"):
        verify_mesh_postcommit_artifact(
            final,
            origin,
            origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            expected_validator_hotkey="validator-hotkey-b",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("response", "final response changed the origin response"),
        ("origin_hash", "proof_postcommit_origin_receipt_hash mismatch"),
        ("frozen_receipt", "changed frozen origin state"),
    ],
)
def test_postcommit_rejects_response_origin_and_frozen_state_substitution(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    import verallm.mesh.worker as mesh_worker

    origin = _origin_artifact()
    final = _final_artifact(origin)
    monkeypatch.setattr(
        mesh_worker,
        "verify_mesh_inference_artifact",
        lambda *args, **kwargs: True,
    )
    if mutation == "response":
        final["response"] = _response(content="substituted")
    elif mutation == "origin_hash":
        final["receipt"]["proof_postcommit_origin_receipt_hash"] = "ff" * 32
    else:
        final["receipt"]["proof_op_manifest_root"] = "ee" * 32

    with pytest.raises(RuntimeError, match=message):
        verify_mesh_postcommit_artifact(
            final,
            origin,
            origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            expected_validator_hotkey=_VALIDATOR_HOTKEY,
        )


def test_postcommit_rejects_a_different_frozen_origin(monkeypatch) -> None:
    import verallm.mesh.worker as mesh_worker

    original = _origin_artifact()
    substituted_origin = _origin_artifact(frozen_marker="aa" * 32)
    final = _final_artifact(original)
    monkeypatch.setattr(
        mesh_worker,
        "verify_mesh_inference_artifact",
        lambda *args, **kwargs: True,
    )

    with pytest.raises(RuntimeError, match="changed frozen origin state"):
        verify_mesh_postcommit_artifact(
            final,
            substituted_origin,
            substituted_origin["request"],
            challenge_nonce=_CHALLENGE_NONCE,
            expected_validator_hotkey=_VALIDATOR_HOTKEY,
        )


def _signed_headers(
    keypair: Keypair,
    *,
    path: str,
    payload: dict,
    timestamp: int,
) -> dict[str, str]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    timestamp_text = str(int(timestamp))
    return {
        HDR_HOTKEY: keypair.ss58_address,
        HDR_SIGNATURE: keypair.sign(
            build_signing_message("POST", path, body, timestamp_text)
        ).hex(),
        HDR_TIMESTAMP: timestamp_text,
    }


def _post_json_status(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
) -> tuple[int, dict]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        response: HTTPResponse | HTTPError
        response = urlopen(request, timeout=15.0)
    except HTTPError as exc:
        response = exc
    with response:
        status = int(response.status)
        decoded = json.loads(response.read().decode("utf-8"))
    assert isinstance(decoded, dict)
    return status, decoded


def _post_json_wire(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
) -> tuple[int, bytes, int]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        response: HTTPResponse | HTTPError
        response = urlopen(request, timeout=15.0)
    except HTTPError as exc:
        response = exc
    with response:
        status = int(response.status)
        content_length = int(response.headers["Content-Length"])
        raw = response.read()
    return status, raw, content_length


def _post_sse_events(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
) -> list[tuple[str, object]]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    events: list[tuple[str, object]] = []
    with urlopen(request, timeout=15.0) as response:
        event = ""
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    events.append(
                        (
                            event,
                            data if data == "[DONE]" else json.loads(data),
                        )
                    )
                event = ""
                data_lines = []
                continue
            if line.startswith("event:"):
                event = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())
    return events


@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/v1/mesh/inference"],
)
def test_authenticated_phase_one_fails_closed_without_snapshot(
    tmp_path,
    path: str,
) -> None:
    coordinator_key = Keypair.create_from_uri(
        "//MeshPostcommitNoSnapshotCoordinator"
    )
    validator_key = Keypair.create_from_uri(
        "//MeshPostcommitNoSnapshotValidator"
    )
    coordinator_evm = Account.from_key(bytes.fromhex("72" * 32))
    allowlist_path = tmp_path / "validators.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()),
                "netuid": 405,
                "validators": [
                    {
                        "uid": 7,
                        "hotkey_ss58": validator_key.ss58_address,
                        "stake": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capability = CapabilityAd(
        uid=1,
        hotkey=coordinator_key.ss58_address,
        endpoint="http://coordinator.private:9338",
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=["a1" * 32],
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=capability,
        server_role="coordinator",
        receipt_signer=lambda receipt_hash: sign_receipt_hash(
            receipt_hash,
            coordinator_key,
        ),
        evm_address=coordinator_evm.address,
        evm_private_key=coordinator_evm.key.hex(),
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
    )
    host, port = coordinator.server_address
    request = _request()
    payload = (
        request
        if path == "/v1/chat/completions"
        else {
            "request_id": "ignored-validator-request-id",
            "openai_request": request,
            "require_proof": True,
        }
    )
    try:
        status, response = _post_json_status(
            f"http://{host}:{port}{path}",
            payload,
            headers=_signed_headers(
                validator_key,
                path=path,
                payload=payload,
                timestamp=int(time.time()),
            ),
        )
        assert status == 503
        assert response == {
            "error": (
                "authenticated mesh inference requires "
                "a verification snapshot"
            )
        }
    finally:
        _shutdown_server(coordinator, coordinator_thread)


@pytest.fixture
def authenticated_postcommit_coordinator(tmp_path, request):
    # Indirect param = the policy's postcommit hard-audit rate; the default
    # keeps the pre-tiering fixture behavior (every audit resolves hard).
    postcommit_hard_audit_bps = int(getattr(request, "param", 10_000))
    trace_root = tmp_path / "coordinator-traces"
    seed_trace = _write_ggml_mul_mat_trace(
        trace_root,
        created_unix_ns=1,
        name="seed",
    )
    # The decode audit rate is now identical for organic and canary traffic,
    # so every request in this fixture is decode-audited and the trace has to
    # carry a real final-projection op alongside the layer GEMM.
    seed_lm_head = _write_decode_lm_head_trace(
        trace_root,
        created_unix_ns=2,
        name="seed-lm-head",
        op_index=1,
    )
    manifest = _gguf_manifest_for_traces(seed_trace, seed_lm_head)
    manifest_path = trace_root / "gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seed_trace.path.unlink()
    seed_lm_head.path.unlink()
    capture_file = trace_root / ".capture-enabled"

    def on_request(_payload: dict) -> None:
        stamp = time.time_ns()
        _write_ggml_mul_mat_trace(
            trace_root,
            created_unix_ns=stamp,
            name=f"request-{stamp}",
        )
        _write_decode_lm_head_trace(
            trace_root,
            created_unix_ns=stamp + 1,
            name=f"request-lm-head-{stamp}",
            op_index=1,
        )

    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_key = Keypair.create_from_uri("//MeshPostcommitCoordinator")
    stage_key = Keypair.create_from_uri("//MeshPostcommitStage")
    validator_key = Keypair.create_from_uri("//MeshPostcommitValidator")
    coordinator_evm = Account.from_key(bytes.fromhex("71" * 32))
    coordinator_endpoint = "http://coordinator.private:9338"
    mesh_spec = MeshSpec(
        mesh_id="mesh-postcommit-http",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey=coordinator_key.ss58_address,
        model_id="mesh-model",
        model_package_hash="a1" * 32,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
        tokenizer_hash="b2" * 32,
        quantization_scheme="gguf_q4_k_m",
        activation_dtype="f16",
        total_layers=4,
        max_context_len=32_768,
        proof_trace_manifest_format="compact-raw-v3",
        members=[
            MeshMember(
                uid=1,
                hotkey=coordinator_key.ss58_address,
                endpoint=coordinator_endpoint,
                stage_index=0,
                layers=StageRange(0, 4),
                role="coordinator",
                backend="gguf_stage",
                proof_key=stage_key.ss58_address,
                payout_bps=10_000,
            )
        ],
    )
    mesh_spec.validate()
    policy = MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        # Both decode rates must be equal, so the value on the wire cannot
        # identify a canary, and an active mesh has to run the full rate.
        # This fixture therefore decode-audits every request, which is why its
        # trace carries a final-projection op.
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1024,
        deferred_proof_enabled=False,
        postcommit_hard_audit_bps=postcommit_hard_audit_bps,
    )
    now = int(time.time())

    def make_snapshot(
        spec: MeshSpec,
        *,
        generation: int,
        issued_at_unix: int,
        expires_at_unix: int,
        stage_identity_commitment: str,
        proof_commitment: str,
    ):
        return sign_mesh_verification_snapshot(
            build_mesh_verification_snapshot(
                spec,
                coordinator=MeshCoordinatorIdentity(
                    chain_id=945,
                    netuid=405,
                    coordinator_uid=spec.coordinator_uid,
                    coordinator_hotkey=coordinator_key.ss58_address,
                    coordinator_evm_address=coordinator_evm.address.lower(),
                    model_index=26,
                ),
                policy=policy,
                generation=generation,
                epoch=spec.epoch,
                issued_at_unix=issued_at_unix,
                expires_at_unix=expires_at_unix,
                stage_bindings=(
                    MeshVerificationStageBinding(
                        stage_index=0,
                        stage_identity_commitment=stage_identity_commitment,
                        proof_key_scheme="sr25519",
                        proof_key=stage_key.ss58_address,
                        proof_commitment=proof_commitment,
                    ),
                ),
            ),
            coordinator_key,
        )

    snapshot = make_snapshot(
        mesh_spec,
        generation=1,
        issued_at_unix=now - 5,
        expires_at_unix=now + 60,
        stage_identity_commitment="c3" * 32,
        proof_commitment="d4" * 32,
    )
    rotated_spec = deepcopy(mesh_spec)
    rotated_spec.epoch = mesh_spec.epoch + 1
    rotated_spec.validate()
    rotated_snapshot = make_snapshot(
        rotated_spec,
        generation=2,
        issued_at_unix=now,
        expires_at_unix=now + 600,
        stage_identity_commitment="e5" * 32,
        proof_commitment="f6" * 32,
    )
    runtime_state = {
        "spec": mesh_spec,
        "snapshot": snapshot,
        "snapshot_loads": 0,
        "rotate_after_snapshot_load": False,
    }

    def load_snapshot():
        runtime_state["snapshot_loads"] += 1
        loaded = runtime_state["snapshot"]
        if runtime_state["rotate_after_snapshot_load"]:
            runtime_state["rotate_after_snapshot_load"] = False
            runtime_state["spec"] = rotated_spec
            runtime_state["snapshot"] = rotated_snapshot
        return loaded.to_dict()

    allowlist_path = tmp_path / "validators.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()),
                "netuid": 405,
                "validators": [
                    {
                        "uid": 7,
                        "hotkey_ss58": validator_key.ss58_address,
                        "stake": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capability = CapabilityAd(
        uid=mesh_spec.coordinator_uid,
        hotkey=coordinator_key.ss58_address,
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[mesh_spec.model_package_hash],
    )
    internal_auth_secret = "mesh-postcommit-operator-secret"
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec_loader=lambda: runtime_state["spec"],
        server_role="coordinator",
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=capture_file,
        proof_trace_dir=seed_trace.path.parent,
        proof_gguf_manifest_path=manifest_path,
        proof_sample_bps=policy.base_proof_sample_bps,
        decode_audit_bps=policy.organic_decode_sample_bps,
        proof_ops_per_request=policy.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            policy.proof_trace_candidates_per_request
        ),
        stage_proof_key=stage_key.ss58_address,
        stage_receipt_signer=lambda body_hash: (
            sign_stage_proof_receipt_body_hash(
                body_hash,
                stage_key,
                expected_proof_key=stage_key.ss58_address,
            )
        ),
        evm_address=coordinator_evm.address,
        evm_private_key=coordinator_evm.key.hex(),
        receipt_signer=lambda receipt_hash: sign_receipt_hash(
            receipt_hash,
            coordinator_key,
        ),
        verification_snapshot_loader=load_snapshot,
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
        internal_auth_secret=internal_auth_secret,
    )
    host, port = coordinator.server_address
    try:
        yield {
            "base_url": f"http://{host}:{port}",
            "coordinator": coordinator,
            "snapshot": snapshot,
            "rotated_spec": rotated_spec,
            "rotated_snapshot": rotated_snapshot,
            "runtime_state": runtime_state,
            "validator_key": validator_key,
            "internal_auth_secret": internal_auth_secret,
        }
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(backend, backend_thread)


def test_internal_operator_chat_pins_one_snapshot_across_runtime_rotation(
    authenticated_postcommit_coordinator,
) -> None:
    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    rotated_snapshot = authenticated_postcommit_coordinator[
        "rotated_snapshot"
    ]
    runtime_state = authenticated_postcommit_coordinator["runtime_state"]
    internal_auth_secret = authenticated_postcommit_coordinator[
        "internal_auth_secret"
    ]
    snapshot_hash = snapshot.snapshot_hash_hex()
    request = _request(
        challenge_nonce="a7" * 32,
        request_id="a8" * 32,
        snapshot_hash=snapshot_hash,
    )
    request["verathos"].pop("challenge_nonce_commitment")
    request["verathos"]["validator_nonce"] = "a7" * 32
    path = "/v1/chat/completions"
    encoded = json.dumps(request, sort_keys=True).encode("utf-8")
    runtime_state["rotate_after_snapshot_load"] = True

    status, response = _post_json_status(
        base_url + path,
        request,
        headers=sign_internal_http_request(
            secret=internal_auth_secret,
            method="POST",
            path=path,
            body=encoded,
        ),
    )

    assert status == 200
    assert runtime_state["snapshot_loads"] == 1
    assert runtime_state["snapshot"] == rotated_snapshot
    metadata = response["verathos_mesh"]
    assert metadata["verification_snapshot_hash"] == snapshot_hash
    assert metadata["verification_snapshot_generation"] == snapshot.generation
    assert metadata["verification_snapshot_epoch"] == snapshot.epoch
    assert metadata["proof_receipt_verified"] is True
    assert metadata["verified"] is True


def test_streaming_internal_operator_chat_pins_snapshot_across_rotation(
    authenticated_postcommit_coordinator,
) -> None:
    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    rotated_snapshot = authenticated_postcommit_coordinator[
        "rotated_snapshot"
    ]
    runtime_state = authenticated_postcommit_coordinator["runtime_state"]
    internal_auth_secret = authenticated_postcommit_coordinator[
        "internal_auth_secret"
    ]
    snapshot_hash = snapshot.snapshot_hash_hex()
    request = _request(
        challenge_nonce="b7" * 32,
        request_id="b8" * 32,
        snapshot_hash=snapshot_hash,
        stream=True,
    )
    request["verathos"].pop("challenge_nonce_commitment")
    request["verathos"]["validator_nonce"] = "b7" * 32
    path = "/v1/chat/completions"
    encoded = json.dumps(request, sort_keys=True).encode("utf-8")
    runtime_state["rotate_after_snapshot_load"] = True

    events = _post_sse_events(
        base_url + path,
        request,
        headers=sign_internal_http_request(
            secret=internal_auth_secret,
            method="POST",
            path=path,
            body=encoded,
        ),
    )

    done_payload = next(
        payload for event, payload in events if event == "done"
    )
    assert isinstance(done_payload, dict)
    assert runtime_state["snapshot_loads"] == 1
    assert runtime_state["snapshot"] == rotated_snapshot
    metadata = done_payload["verathos_mesh"]
    assert metadata["verification_snapshot_hash"] == snapshot_hash
    assert metadata["verification_snapshot_generation"] == snapshot.generation
    assert metadata["verification_snapshot_epoch"] == snapshot.epoch
    assert metadata["proof_receipt_verified"] is True
    assert metadata["verified"] is True


def test_coordinator_rejects_wrong_reveal_then_replays_exact_final_artifact(
    authenticated_postcommit_coordinator,
    caplog,
) -> None:
    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())

    request_id = "e1" * 32
    challenge_nonce = "e2" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_path = "/v1/mesh/inference"
    inference_body = {
        "request_id": "ignored-validator-request-id",
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + inference_path,
        inference_body,
        headers=_signed_headers(
            validator_key,
            path=inference_path,
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    origin_receipt = origin["receipt"]
    assert origin_receipt["proof_postcommit"] is True
    assert origin_receipt["proof_postcommit_finalized"] is False

    wrong_reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": origin_receipt["receipt_hash"],
        "mesh_response_commitment_hash": origin_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": "ef" * 32,
    }
    with caplog.at_level(logging.ERROR, logger="verallm.mesh.worker"):
        wrong_status, wrong_payload = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            wrong_reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=wrong_reveal,
                timestamp=timestamp + 1,
            ),
        )
        correct_reveal = {**wrong_reveal, "challenge_nonce": challenge_nonce}
        retry_status, retry_payload = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            correct_reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=correct_reveal,
                timestamp=timestamp + 2,
            ),
        )
        replay_status, replay_payload = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            correct_reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=correct_reveal,
                timestamp=timestamp + 3,
            ),
        )
        conflicting_reveal = {
            **correct_reveal,
            "challenge_nonce": "ed" * 32,
        }
        conflict_status, conflict_payload = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            conflicting_reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=conflicting_reveal,
                timestamp=timestamp + 4,
            ),
        )

    assert (wrong_status, wrong_payload) == (
        500,
        {"error": "mesh inference failed"},
    )
    assert retry_status == 200
    assert retry_payload["receipt"]["proof_postcommit_finalized"] is True
    assert retry_payload["receipt"]["proof_receipt_verified"] is True
    assert replay_status == 200
    assert replay_payload == retry_payload
    assert json.dumps(
        replay_payload,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8") == json.dumps(
        retry_payload,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    assert (conflict_status, conflict_payload) == (
        500,
        {"error": "mesh inference failed"},
    )
    private_failures = [
        str(record.exc_info[1])
        for record in caplog.records
        if record.exc_info is not None
    ]
    assert any("challenge nonce reveal mismatch" in item for item in private_failures)
    assert any(
        "retry conflicts with finalized artifact" in item
        for item in private_failures
    )

    stream_request_id = "f1" * 32
    stream_nonce = "f2" * 32
    stream_request = _request(
        challenge_nonce=stream_nonce,
        request_id=stream_request_id,
        snapshot_hash=snapshot_hash,
        stream=True,
    )
    stream_path = "/v1/chat/completions"
    stream_events = _post_sse_events(
        base_url + stream_path,
        stream_request,
        headers=_signed_headers(
            validator_key,
            path=stream_path,
            payload=stream_request,
            timestamp=timestamp + 5,
        ),
    )
    done_index, done_payload = next(
        (index, payload)
        for index, (event, payload) in enumerate(stream_events)
        if event == "done"
    )
    assert any(
        index < done_index
        and isinstance(payload, dict)
        and payload.get("object") == "chat.completion.chunk"
        for index, (_event, payload) in enumerate(stream_events)
    )
    assert isinstance(done_payload, dict)
    metadata = done_payload["verathos_mesh"]
    receipt = metadata["receipt"]
    assert receipt["proof_postcommit"] is True
    assert receipt["proof_postcommit_finalized"] is False
    assert receipt["proof_beacon"] == ""
    assert receipt["proof_required"] is False
    assert receipt["proof_receipt_count"] == 0
    assert "proof_payloads" not in metadata


def test_postcommit_local_or_cache_failure_releases_claim_for_exact_retry(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    request_id = "c7" * 32
    challenge_nonce = "c8" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    origin_receipt = origin["receipt"]
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": origin_receipt["receipt_hash"],
        "mesh_response_commitment_hash": origin_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }
    coordinator = authenticated_postcommit_coordinator["coordinator"]
    collect_proof_receipts = coordinator.verathos_proof_receipt_collector
    finalized_artifact_limit = (
        mesh_worker.POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES
    )

    def fail_local_proof_generation(**_kwargs):
        raise RuntimeError("local proof generation failed")

    monkeypatch.setattr(
        coordinator,
        "verathos_proof_receipt_collector",
        fail_local_proof_generation,
    )
    failed_status, failed_payload = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 1,
        ),
    )
    assert (failed_status, failed_payload) == (
        500,
        {"error": "mesh inference failed"},
    )

    monkeypatch.setattr(
        coordinator,
        "verathos_proof_receipt_collector",
        collect_proof_receipts,
    )
    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES",
        1,
    )
    limited_status, limited_payload = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 2,
        ),
    )
    assert (limited_status, limited_payload) == (
        500,
        {"error": "mesh inference failed"},
    )

    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES",
        finalized_artifact_limit,
    )
    retry_status, final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 3,
        ),
    )
    assert retry_status == 200
    assert final["receipt"]["proof_postcommit_finalized"] is True
    assert final["receipt"]["proof_receipt_verified"] is True


def test_postcommit_final_and_replay_use_exact_bounded_canonical_wire_bytes(
    authenticated_postcommit_coordinator,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    request_id = "b3" * 32
    challenge_nonce = "b4" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    receipt = origin["receipt"]
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": receipt["receipt_hash"],
        "mesh_response_commitment_hash": receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }
    final_status, final_wire, final_length = _post_json_wire(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 1,
        ),
    )
    replay_status, replay_wire, replay_length = _post_json_wire(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 2,
        ),
    )

    assert final_status == replay_status == 200
    assert final_length == len(final_wire)
    assert replay_length == len(replay_wire)
    assert replay_wire == final_wire
    decoded = json.loads(final_wire)
    assert isinstance(decoded, dict)
    assert final_wire == json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert len(final_wire) <= mesh_worker.POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES


def test_oversized_postcommit_origin_is_refused_before_it_is_returned(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    original_limit = mesh_worker.POSTCOMMIT_ORIGIN_MAX_SINGLE_BYTES
    monkeypatch.setattr(mesh_worker, "POSTCOMMIT_ORIGIN_MAX_SINGLE_BYTES", 1)

    rejected_request = _request(
        challenge_nonce="91" * 32,
        request_id="92" * 32,
        snapshot_hash=snapshot_hash,
    )
    rejected_body = {
        "openai_request": rejected_request,
        "require_proof": True,
    }
    rejected_status, rejected_payload = _post_json_status(
        base_url + "/v1/mesh/inference",
        rejected_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=rejected_body,
            timestamp=timestamp,
        ),
    )
    assert (rejected_status, rejected_payload) == (
        500,
        {"error": "mesh inference failed"},
    )

    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_ORIGIN_MAX_SINGLE_BYTES",
        original_limit,
    )
    admitted_request = _request(
        challenge_nonce="93" * 32,
        request_id="94" * 32,
        snapshot_hash=snapshot_hash,
    )
    admitted_body = {
        "openai_request": admitted_request,
        "require_proof": True,
    }
    admitted_status, admitted_origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        admitted_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=admitted_body,
            timestamp=timestamp + 1,
        ),
    )
    assert admitted_status == 200
    assert admitted_origin["receipt"]["request_id"] == "94" * 32


@pytest.mark.parametrize(
    "budget_name",
    [
        "POSTCOMMIT_ORIGIN_MAX_BYTES_PER_PRINCIPAL",
        "POSTCOMMIT_ORIGIN_MAX_BYTES",
    ],
)
def test_postcommit_origin_byte_capacity_recovers_without_corrupting_retry(
    authenticated_postcommit_coordinator,
    monkeypatch,
    budget_name: str,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())

    def phase_one(request_id: str, nonce: str, offset: int):
        inference_request = _request(
            challenge_nonce=nonce,
            request_id=request_id,
            snapshot_hash=snapshot_hash,
        )
        inference_body = {
            "openai_request": inference_request,
            "require_proof": True,
        }
        return _post_json_status(
            base_url + "/v1/mesh/inference",
            inference_body,
            headers=_signed_headers(
                validator_key,
                path="/v1/mesh/inference",
                payload=inference_body,
                timestamp=timestamp + offset,
            ),
        )

    first_status, first_origin = phase_one("95" * 32, "96" * 32, 0)
    assert first_status == 200
    first_receipt = first_origin["receipt"]
    first_reveal = {
        "validator_request_id": "95" * 32,
        "origin_receipt_hash": first_receipt["receipt_hash"],
        "mesh_response_commitment_hash": first_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": "96" * 32,
    }

    original_budget = getattr(mesh_worker, budget_name)
    monkeypatch.setattr(mesh_worker, budget_name, 1)
    refused_status, refused_payload = phase_one("97" * 32, "98" * 32, 1)
    assert (refused_status, refused_payload) == (
        500,
        {"error": "mesh inference failed"},
    )

    final_status, final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        first_reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=first_reveal,
            timestamp=timestamp + 2,
        ),
    )
    assert final_status == 200
    assert final["receipt"]["proof_postcommit_finalized"] is True

    monkeypatch.setattr(mesh_worker, budget_name, original_budget)
    recovered_status, recovered_origin = phase_one("99" * 32, "9a" * 32, 3)
    assert recovered_status == 200
    assert recovered_origin["receipt"]["request_id"] == "99" * 32

    replay_status, replay = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        first_reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=first_reveal,
            timestamp=timestamp + 4,
        ),
    )
    assert replay_status == 200
    assert replay == final


def test_concurrent_exact_postcommit_retry_is_explicitly_retryable(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    request_id = "ca" * 32
    challenge_nonce = "cb" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    origin_receipt = origin["receipt"]
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": origin_receipt["receipt_hash"],
        "mesh_response_commitment_hash": origin_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }
    coordinator = authenticated_postcommit_coordinator["coordinator"]
    collect_proof_receipts = coordinator.verathos_proof_receipt_collector
    proof_started = threading.Event()
    allow_proof = threading.Event()
    proof_calls: list[object] = []

    def block_local_proof_generation(**kwargs):
        proof_calls.append(object())
        proof_started.set()
        if not allow_proof.wait(timeout=5):
            raise RuntimeError("timed out waiting to finish local proof")
        return collect_proof_receipts(**kwargs)

    monkeypatch.setattr(
        coordinator,
        "verathos_proof_receipt_collector",
        block_local_proof_generation,
    )
    first_result: dict[str, object] = {}

    def finalize_first_request() -> None:
        first_result["value"] = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=reveal,
                timestamp=timestamp + 1,
            ),
        )

    finalizer = threading.Thread(target=finalize_first_request, daemon=True)
    finalizer.start()
    if not proof_started.wait(timeout=5):
        allow_proof.set()
        finalizer.join(timeout=5)
        pytest.fail("postcommit finalization did not reach proof generation")
    in_progress_status, in_progress_payload = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 2,
        ),
    )
    assert (in_progress_status, in_progress_payload) == (
        503,
        {
            "error": "postcommit finalization is in progress",
            "error_code": "postcommit_finalization_in_progress",
            "retryable": True,
        },
    )
    allow_proof.set()
    finalizer.join(timeout=10)
    assert not finalizer.is_alive()
    first_value = first_result.get("value")
    assert isinstance(first_value, tuple)
    first_status, first_final = first_value
    assert first_status == 200

    replay_status, replay_final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 3,
        ),
    )
    assert replay_status == 200
    assert replay_final == first_final
    assert len(proof_calls) == 1


@pytest.mark.parametrize(
    "limit_name",
    [
        "POSTCOMMIT_PHASE2_MAX_ACTIVE",
        "POSTCOMMIT_FINALIZATION_MAX_ACTIVE",
    ],
)
def test_postcommit_capacity_backpressure_releases_slot_and_attempt(
    authenticated_postcommit_coordinator,
    monkeypatch,
    limit_name: str,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    request_id = "ad" * 32
    challenge_nonce = "ae" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    receipt = origin["receipt"]
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": receipt["receipt_hash"],
        "mesh_response_commitment_hash": receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }

    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_PHASE2_MAX_ACTIVE_PER_PRINCIPAL",
        1,
    )
    original_limit = getattr(mesh_worker, limit_name)
    monkeypatch.setattr(mesh_worker, limit_name, 0)
    limited_status, limited_payload = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 1,
        ),
    )
    assert (limited_status, limited_payload) == (
        503,
        {
            "error": "postcommit capacity is temporarily unavailable",
            "error_code": "postcommit_capacity_unavailable",
            "retryable": True,
        },
    )

    monkeypatch.setattr(mesh_worker, limit_name, original_limit)
    retry_status, final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 2,
        ),
    )
    assert retry_status == 200
    assert final["receipt"]["proof_postcommit_finalized"] is True


def test_finalized_postcommit_replay_expires_with_original_origin_deadline(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    monkeypatch.setattr(mesh_worker, "POSTCOMMIT_ORIGIN_TTL_SECONDS", 30.0)
    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())
    # Bind phase-one storage to the same origin instant used by this test.
    # Deriving expiry from an earlier integer timestamp is racy under the full
    # suite: if request preparation crosses the next wall-clock second, the
    # worker correctly stores a later deadline and timestamp + TTL + 1 has not
    # actually expired it yet.
    monkeypatch.setattr(mesh_worker.time, "time", lambda: float(timestamp))
    request_id = "cc" * 32
    challenge_nonce = "cd" * 32
    inference_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": inference_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=timestamp,
        ),
    )
    assert status == 200
    origin_receipt = origin["receipt"]
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": origin_receipt["receipt_hash"],
        "mesh_response_commitment_hash": origin_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }
    final_status, _final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=timestamp + 1,
        ),
    )
    assert final_status == 200

    expired_time = timestamp + 31
    monkeypatch.setattr(mesh_worker.time, "time", lambda: float(expired_time))
    replay_status, replay_payload = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=expired_time,
        ),
    )
    assert (replay_status, replay_payload) == (
        500,
        {"error": "mesh inference failed"},
    )


def test_replay_capacity_backpressures_phase_one_without_evicting_unexpired_finals(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    monkeypatch.setattr(mesh_worker, "POSTCOMMIT_ORIGIN_MAX_ENTRIES", 1)
    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_ORIGIN_MAX_ENTRIES_PER_PRINCIPAL",
        1,
    )
    monkeypatch.setattr(mesh_worker, "POSTCOMMIT_FINALIZED_MAX_ENTRIES", 2)
    monkeypatch.setattr(
        mesh_worker,
        "POSTCOMMIT_FINALIZED_MAX_ENTRIES_PER_PRINCIPAL",
        2,
    )
    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())

    def phase_one(request_id: str, challenge_nonce: str, offset: int):
        inference_request = _request(
            challenge_nonce=challenge_nonce,
            request_id=request_id,
            snapshot_hash=snapshot_hash,
        )
        inference_body = {
            "openai_request": inference_request,
            "require_proof": True,
        }
        status, origin = _post_json_status(
            base_url + "/v1/mesh/inference",
            inference_body,
            headers=_signed_headers(
                validator_key,
                path="/v1/mesh/inference",
                payload=inference_body,
                timestamp=timestamp + offset,
            ),
        )
        if status != 200:
            return status, origin
        receipt = origin["receipt"]
        return (
            status,
            {
                "validator_request_id": request_id,
                "origin_receipt_hash": receipt["receipt_hash"],
                "mesh_response_commitment_hash": receipt[
                    "mesh_response_commitment_hash"
                ],
                "verification_snapshot_hash": snapshot_hash,
                "challenge_nonce": challenge_nonce,
            },
        )

    def phase_two(reveal: dict, offset: int):
        return _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=reveal,
                timestamp=timestamp + offset,
            ),
        )

    first_origin_status, first_reveal = phase_one("ce" * 32, "cf" * 32, 0)
    assert first_origin_status == 200
    first_status, first_final = phase_two(first_reveal, 1)
    assert first_status == 200

    second_origin_status, second_reveal = phase_one("d0" * 32, "d1" * 32, 2)
    assert second_origin_status == 200
    second_status, second_final = phase_two(second_reveal, 3)
    assert second_status == 200

    first_replay_status, first_replay = phase_two(first_reveal, 4)
    assert first_replay_status == 200
    assert first_replay == first_final
    second_replay_status, second_replay = phase_two(second_reveal, 4)
    assert second_replay_status == 200
    assert second_replay == second_final

    # Both replay slots are committed until their original deadlines. A third
    # phase one is refused before exposing a signed origin; neither existing
    # final is sacrificed to admit it.
    refused_status, refused_payload = phase_one("d2" * 32, "d3" * 32, 5)
    assert (refused_status, refused_payload) == (
        500,
        {"error": "mesh inference failed"},
    )
    first_retry_status, first_retry = phase_two(first_reveal, 6)
    second_retry_status, second_retry = phase_two(second_reveal, 7)
    assert first_retry_status == second_retry_status == 200
    assert first_retry == first_final
    assert second_retry == second_final


def test_postcommit_uses_phase_one_pinned_spec_snapshot_and_validation_time(
    authenticated_postcommit_coordinator,
    monkeypatch,
) -> None:
    import verallm.mesh.worker as mesh_worker

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    rotated_spec = authenticated_postcommit_coordinator["rotated_spec"]
    rotated_snapshot = authenticated_postcommit_coordinator[
        "rotated_snapshot"
    ]
    runtime_state = authenticated_postcommit_coordinator["runtime_state"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    validation_time = max(
        int(snapshot.issued_at_unix),
        min(int(time.time()), int(snapshot.expires_at_unix) - 1),
    )
    # Decode audit now fires on every request, so there is no unsampled
    # variant to hunt for and a single phase one is enough.
    request_id = "d1" * 32
    challenge_nonce = "e1" * 32
    openai_request = _request(
        challenge_nonce=challenge_nonce,
        request_id=request_id,
        snapshot_hash=snapshot_hash,
    )
    inference_body = {
        "openai_request": openai_request,
        "require_proof": True,
    }
    status, origin = _post_json_status(
        base_url + "/v1/mesh/inference",
        inference_body,
        headers=_signed_headers(
            validator_key,
            path="/v1/mesh/inference",
            payload=inference_body,
            timestamp=validation_time,
        ),
    )
    assert status == 200
    assert postcommit_audit_decision(
        origin["receipt"],
        openai_request,
        challenge_nonce=challenge_nonce,
    )["decode_audit_sampled"]
    snapshot_loads_after_phase_one = runtime_state["snapshot_loads"]
    assert snapshot_loads_after_phase_one == 1
    origin_receipt = origin["receipt"]
    assert origin_receipt["mesh_spec_hash"] != rotated_spec.spec_hash_hex()
    assert origin_receipt["verification_snapshot_hash"] != (
        rotated_snapshot.snapshot_hash_hex()
    )
    runtime_state["spec"] = rotated_spec
    runtime_state["snapshot"] = rotated_snapshot
    postcommit_time = int(snapshot.expires_at_unix) + 1
    monkeypatch.setattr(
        mesh_worker.time,
        "time",
        lambda: float(postcommit_time),
    )
    reveal = {
        "validator_request_id": request_id,
        "origin_receipt_hash": origin_receipt["receipt_hash"],
        "mesh_response_commitment_hash": origin_receipt[
            "mesh_response_commitment_hash"
        ],
        "verification_snapshot_hash": snapshot_hash,
        "challenge_nonce": challenge_nonce,
    }
    final_status, final = _post_json_status(
        base_url + POSTCOMMIT_AUDIT_PATH,
        reveal,
        headers=_signed_headers(
            validator_key,
            path=POSTCOMMIT_AUDIT_PATH,
            payload=reveal,
            timestamp=postcommit_time,
        ),
    )
    assert final_status == 200
    assert runtime_state["snapshot_loads"] == snapshot_loads_after_phase_one
    final_receipt = final["receipt"]
    assert final_receipt["mesh_spec_hash"] == origin_receipt["mesh_spec_hash"]
    assert final_receipt["verification_snapshot_hash"] == snapshot_hash
    assert final_receipt["verification_snapshot_generation"] == (
        snapshot.generation
    )
    assert all(
        item["verification_snapshot_hash"] == snapshot_hash
        for item in final["proof_receipts"]
    )
    with pytest.raises(ValueError, match="expired"):
        verify_mesh_postcommit_artifact(
            final,
            origin,
            openai_request,
            challenge_nonce=challenge_nonce,
            expected_coordinator_hotkey=(
                snapshot.coordinator.coordinator_hotkey
            ),
            expected_coordinator_uid=(
                snapshot.coordinator.coordinator_uid
            ),
            expected_validator_hotkey=validator_key.ss58_address,
            require_coordinator_signature=True,
            verification_snapshot=snapshot,
        )
    assert verify_mesh_postcommit_artifact(
        final,
        origin,
        openai_request,
        challenge_nonce=challenge_nonce,
        expected_coordinator_hotkey=snapshot.coordinator.coordinator_hotkey,
        expected_coordinator_uid=snapshot.coordinator.coordinator_uid,
        expected_validator_hotkey=validator_key.ss58_address,
        require_coordinator_signature=True,
        verification_snapshot=snapshot,
        verification_snapshot_now_unix=validation_time,
    )


@pytest.mark.parametrize(
    "authenticated_postcommit_coordinator", [0], indirect=True
)
def test_postcommit_light_resolution_and_signed_hard_demand(
    authenticated_postcommit_coordinator,
) -> None:
    """With a zero hard-audit rate the reveal resolves LIGHT end to end,
    and the validator's signed audit_tier=hard demand forces the full hard
    relation on the same coordinator."""

    from verallm.mesh.proof import (
        VERATHOS_GGML_GEMM_PROOF_MODE,
        VERATHOS_GGML_LIGHT_PROOF_MODE,
    )

    base_url = authenticated_postcommit_coordinator["base_url"]
    snapshot = authenticated_postcommit_coordinator["snapshot"]
    validator_key = authenticated_postcommit_coordinator["validator_key"]
    snapshot_hash = snapshot.snapshot_hash_hex()
    timestamp = int(time.time())

    def run_flow(request_id: str, challenge_nonce: str, *, audit_tier: str):
        inference_request = _request(
            challenge_nonce=challenge_nonce,
            request_id=request_id,
            snapshot_hash=snapshot_hash,
        )
        inference_path = "/v1/mesh/inference"
        inference_body = {
            "request_id": "ignored-validator-request-id",
            "openai_request": inference_request,
            "require_proof": True,
        }
        nonlocal timestamp
        timestamp += 1
        status, origin = _post_json_status(
            base_url + inference_path,
            inference_body,
            headers=_signed_headers(
                validator_key,
                path=inference_path,
                payload=inference_body,
                timestamp=timestamp,
            ),
        )
        assert status == 200, origin
        origin_receipt = origin["receipt"]
        assert origin_receipt["proof_postcommit_finalized"] is False
        # The hard rate is part of the frozen origin, signed by the snapshot.
        assert origin_receipt["proof_postcommit_hard_bps"] == 0
        reveal = {
            "validator_request_id": request_id,
            "origin_receipt_hash": origin_receipt["receipt_hash"],
            "mesh_response_commitment_hash": origin_receipt[
                "mesh_response_commitment_hash"
            ],
            "verification_snapshot_hash": snapshot_hash,
            "challenge_nonce": challenge_nonce,
        }
        if audit_tier:
            reveal["audit_tier"] = audit_tier
        timestamp += 1
        status, final = _post_json_status(
            base_url + POSTCOMMIT_AUDIT_PATH,
            reveal,
            headers=_signed_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=reveal,
                timestamp=timestamp,
            ),
        )
        assert status == 200, final
        return inference_request, origin, final

    # Default reveal: the zero rate resolves LIGHT. v3 shape: a light draw
    # owes NOTHING at reveal time - the origin receipt with its inline
    # light proof is the light tier, so the finalized resolution carries
    # no revealed proof material and still passes the exact
    # validator-side verifier.
    request, origin, final = run_flow("a1" * 32, "a2" * 32, audit_tier="")
    receipt = final["receipt"]
    assert receipt["proof_audit_tier"] == "light"
    assert receipt["proof_sampled"] is False
    assert receipt["proof_required"] is False
    assert not final.get("proof_payloads"), "light resolutions owe nothing"
    assert verify_mesh_postcommit_artifact(
        final,
        origin,
        request,
        challenge_nonce="a2" * 32,
        expected_coordinator_hotkey=snapshot.coordinator.coordinator_hotkey,
        expected_coordinator_uid=snapshot.coordinator.coordinator_uid,
        expected_validator_hotkey=validator_key.ss58_address,
        require_coordinator_signature=True,
        verification_snapshot=snapshot,
    )
    # A light artifact must NOT satisfy a hard demand.
    with pytest.raises(RuntimeError, match="proof_sampled|proof_audit_tier"):
        verify_mesh_postcommit_artifact(
            final,
            origin,
            request,
            challenge_nonce="a2" * 32,
            expected_coordinator_hotkey=snapshot.coordinator.coordinator_hotkey,
            expected_coordinator_uid=snapshot.coordinator.coordinator_uid,
            expected_validator_hotkey=validator_key.ss58_address,
            require_coordinator_signature=True,
            verification_snapshot=snapshot,
            audit_tier="hard",
        )

    # Signed hard demand: same coordinator, same zero rate, full hard proof.
    request, origin, final = run_flow("b1" * 32, "b2" * 32, audit_tier="hard")
    receipt = final["receipt"]
    assert receipt["proof_audit_tier"] == "hard"
    assert receipt["proof_sampled"] is True
    assert receipt["proof_mode"] == VERATHOS_GGML_GEMM_PROOF_MODE
    assert receipt["proof_receipt_verified"] is True
    assert verify_mesh_postcommit_artifact(
        final,
        origin,
        request,
        challenge_nonce="b2" * 32,
        expected_coordinator_hotkey=snapshot.coordinator.coordinator_hotkey,
        expected_coordinator_uid=snapshot.coordinator.coordinator_uid,
        expected_validator_hotkey=validator_key.ss58_address,
        require_coordinator_signature=True,
        verification_snapshot=snapshot,
        audit_tier="hard",
    )
