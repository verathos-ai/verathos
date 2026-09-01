import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons.receipts import create_receipt, receipt_to_dict
from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    build_signing_message,
)
from verallm.mesh import (
    CapabilityAd,
    MeshMember,
    MeshSpec,
    StageRange,
    VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
    mesh_validator_challenge_nonce_commitment,
)
from verallm.mesh.http_auth import sign_internal_http_request
from verallm.mesh.receipt_signing import sign_receipt_hash
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
)
from verallm.mesh.worker import (
    _assert_public_proof_artifact_value,
    _sanitize_validator_proof_artifacts,
    make_worker_server,
    serve_worker_in_thread,
)


_MODEL_PACKAGE_HASH = "a" * 64


def _capability(hotkey: str = "5MeshCoordinator") -> CapabilityAd:
    return CapabilityAd(
        uid=1,
        hotkey=hotkey,
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[_MODEL_PACKAGE_HASH],
    )


def _secure_coordinator_kwargs(seed_hex: str) -> dict:
    hotkey = Keypair.create_from_seed(seed_hex * 32)
    evm = Account.create(extra_entropy=("mesh-evm-" + seed_hex).encode())
    return {
        "capability": _capability(str(hotkey.ss58_address)),
        "receipt_signer": lambda receipt_hash: sign_receipt_hash(
            receipt_hash,
            hotkey,
        ),
        "evm_address": evm.address,
        "evm_private_key": evm.key.hex(),
    }


def _write_validator_allowlist(
    path,
    *hotkeys: str,
    updated_at: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "updated_at": (
                    int(time.time()) if updated_at is None else updated_at
                ),
                "netuid": 405,
                "validators": [
                    {
                        "uid": index + 2,
                        "hotkey_ss58": hotkey,
                        "stake": 1.0,
                    }
                    for index, hotkey in enumerate(hotkeys)
                ],
            }
        ),
        encoding="utf-8",
    )


def _validator_headers(
    keypair: Keypair,
    *,
    method: str,
    path: str,
    body: bytes,
    timestamp: int | None = None,
) -> dict[str, str]:
    timestamp_text = str(int(time.time()) if timestamp is None else timestamp)
    signature = keypair.sign(
        build_signing_message(method, path, body, timestamp_text)
    )
    return {
        HDR_HOTKEY: str(keypair.ss58_address),
        HDR_SIGNATURE: signature.hex(),
        HDR_TIMESTAMP: timestamp_text,
    }


def _signed_service_receipt(
    keypair: Keypair,
    *,
    seed_hex: str,
    miner_address: str,
    epoch: int = 7,
    model_id: str = "mesh-model",
    model_index: int = 0,
) -> dict:
    observed_end = time.time()
    return receipt_to_dict(
        create_receipt(
            miner_address=miner_address,
            model_id=model_id,
            model_index=model_index,
            epoch_number=epoch,
            commitment_hash=b"\xab" * 32,
            ttft_ms=25.0,
            tokens_generated=8,
            generation_time_ms=200.0,
            tokens_per_sec=40.0,
            validator_hotkey=keypair.public_key,
            validator_private_key=bytes.fromhex(seed_hex * 32),
            prompt_tokens=4,
            proof_verified=True,
            proof_requested=True,
            is_canary=True,
            timestamp=int(observed_end),
            observed_start_ts=observed_end - 0.2,
            observed_end_ts=observed_end,
        )
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    data = body if method == "POST" else None
    if method == "POST":
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=5.0) as response:
            status = int(response.status)
            raw = response.read()
    except HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    payload = json.loads(raw.decode("utf-8"))
    assert isinstance(payload, dict)
    return status, payload


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def _fake_backend() -> tuple[
    ThreadingHTTPServer,
    threading.Thread,
    str,
    list[dict],
]:
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if self.path.rstrip("/") != "/v1/chat/completions":
                response = {"error": "not found"}
                status = 404
            else:
                calls.append(payload)
                response = {
                    "id": "chatcmpl-security-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "mesh-test"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                status = 200
            encoded = json.dumps(response, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}", calls


def _failing_backend(private_detail: str) -> tuple[
    ThreadingHTTPServer,
    threading.Thread,
    str,
]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            encoded = json.dumps({"error": private_detail}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _private_mesh(
    *,
    coordinator_hotkey: str,
    coordinator_endpoint: str,
    worker_hotkey: str = "private-worker-hotkey-alpha",
    worker_endpoint: str = "http://10.44.0.2:19101",
) -> MeshSpec:
    spec = MeshSpec(
        mesh_id="mesh_http_security",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey=coordinator_hotkey,
        model_id="mesh-security-model",
        model_package_hash=_MODEL_PACKAGE_HASH,
        total_layers=2,
        members=[
            MeshMember(
                uid=1,
                hotkey=coordinator_hotkey,
                endpoint=coordinator_endpoint,
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10_000,
            ),
            MeshMember(
                uid=91,
                hotkey=worker_hotkey,
                endpoint=worker_endpoint,
                proof_endpoint=f"{worker_endpoint}/proof",
                rpc_endpoint="10.44.0.2:50052",
                rpc_split_weight=1,
                stage_index=1,
                layers=StageRange(0, 2),
                role="worker",
                backend="gguf_stage_worker",
            ),
        ],
    )
    spec.validate()
    return spec


def test_coordinator_validator_auth_binds_raw_body_and_rejects_replay(
    tmp_path,
    monkeypatch,
):
    keypair = Keypair.create_from_seed("11" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(keypair.ss58_address))
    monkeypatch.setenv("VERALLM_DATA_DIR", str(tmp_path))
    coordinator = _secure_coordinator_kwargs("77")

    server, thread = serve_worker_in_thread(
        **coordinator,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
    )
    host, port = server.server_address
    path = "/epoch/receipt"
    url = f"http://{host}:{port}{path}"
    body = json.dumps(
        _signed_service_receipt(
            keypair,
            seed_hex="11",
            miner_address=coordinator["evm_address"],
        ),
        sort_keys=True,
    ).encode("utf-8")
    try:
        unsigned_status, unsigned = _request_json(
            url,
            method="POST",
            body=body,
        )
        assert unsigned_status == 401
        assert "validator auth header" in unsigned["error"]

        headers = _validator_headers(
            keypair,
            method="POST",
            path=path,
            body=body,
        )
        tampered_status, tampered = _request_json(
            url,
            method="POST",
            body=body + b" ",
            headers=headers,
        )
        assert tampered_status == 401
        assert tampered["error"] == "invalid validator request signature"

        accepted_status, accepted = _request_json(
            url,
            method="POST",
            body=body,
            headers=headers,
        )
        assert accepted_status == 200
        assert accepted["status"] == "accepted"

        duplicate_headers = _validator_headers(
            keypair,
            method="POST",
            path=path,
            body=body,
            timestamp=int(time.time()) + 1,
        )
        duplicate_status, duplicate = _request_json(
            url,
            method="POST",
            body=body,
            headers=duplicate_headers,
        )
        assert duplicate_status == 200
        assert duplicate == {
            "status": "duplicate",
            "epoch": 7,
            "count": 1,
        }

        replay_status, replay = _request_json(
            url,
            method="POST",
            body=body,
            headers=headers,
        )
        assert replay_status == 401
        assert replay["error"] == "validator request was already accepted"
    finally:
        _stop_server(server, thread)


def test_coordinator_validator_auth_rejects_stale_allowlist(tmp_path):
    keypair = Keypair.create_from_seed("12" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(
        allowlist_path,
        str(keypair.ss58_address),
        updated_at=int(time.time()) - 901,
    )
    server, thread = serve_worker_in_thread(
        **_secure_coordinator_kwargs("78"),
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        validator_allowlist_max_age_seconds=900,
    )
    host, port = server.server_address
    path = "/v1/mesh/verification-snapshot"
    try:
        status, payload = _request_json(
            f"http://{host}:{port}{path}",
            headers=_validator_headers(
                keypair,
                method="GET",
                path=path,
                body=b"",
            ),
        )
        assert status == 503
        assert payload["error"] == "validator allowlist unavailable"
    finally:
        _stop_server(server, thread)


def test_authenticated_receipt_binds_http_signer_and_coordinator_address(
    tmp_path,
    monkeypatch,
):
    first = Keypair.create_from_seed("21" * 32)
    second = Keypair.create_from_seed("22" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(
        allowlist_path,
        str(first.ss58_address),
        str(second.ss58_address),
    )
    monkeypatch.setenv("VERALLM_DATA_DIR", str(tmp_path))
    coordinator = _secure_coordinator_kwargs("79")
    server, thread = serve_worker_in_thread(
        **coordinator,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
    )
    host, port = server.server_address
    path = "/epoch/receipt"
    url = f"http://{host}:{port}{path}"
    try:
        second_receipt = json.dumps(
            _signed_service_receipt(
                second,
                seed_hex="22",
                miner_address=coordinator["evm_address"],
            ),
            sort_keys=True,
        ).encode("utf-8")
        wrong_signer_status, wrong_signer = _request_json(
            url,
            method="POST",
            body=second_receipt,
            headers=_validator_headers(
                first,
                method="POST",
                path=path,
                body=second_receipt,
            ),
        )
        assert wrong_signer_status == 403
        assert "signer does not match" in wrong_signer["error"]

        wrong_address_receipt = json.dumps(
            _signed_service_receipt(
                first,
                seed_hex="21",
                miner_address="0x" + "ff" * 20,
            ),
            sort_keys=True,
        ).encode("utf-8")
        wrong_address_status, wrong_address = _request_json(
            url,
            method="POST",
            body=wrong_address_receipt,
            headers=_validator_headers(
                first,
                method="POST",
                path=path,
                body=wrong_address_receipt,
            ),
        )
        assert wrong_address_status == 403
        assert "miner address" in wrong_address["error"]
    finally:
        _stop_server(server, thread)


def test_authenticated_inference_fails_closed_without_verification_snapshot(
    tmp_path,
):
    keypair = Keypair.create_from_seed("22" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(keypair.ss58_address))
    backend, backend_thread, backend_url, backend_calls = _fake_backend()
    server, thread = serve_worker_in_thread(
        **_secure_coordinator_kwargs("88"),
        backend_url=backend_url,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
    )
    host, port = server.server_address
    path = "/v1/chat/completions"
    url = f"http://{host}:{port}{path}"

    def send(payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        return _request_json(
            url,
            method="POST",
            body=body,
            headers=_validator_headers(
                keypair,
                method="POST",
                path=path,
                body=body,
            ),
        )

    request = {
        "model": "mesh-test",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {
            "validator_nonce": "33" * 32,
            "validator_request_id": "44" * 32,
        },
    }
    try:
        status, payload = send(request)
        assert status == 503
        assert payload == {
            "error": (
                "authenticated mesh inference requires a verification snapshot"
            )
        }
        assert backend_calls == []
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_worker_internal_hmac_requires_auth_and_allows_fresh_request_nonces():
    secret = b"ephemeral-coordinator-worker-secret"
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        server_role="worker",
        internal_auth_secret=secret,
    )
    host, port = server.server_address
    path = "/v1/stage/handshake"
    url = f"http://{host}:{port}{path}"
    body = b"{}"
    try:
        unsigned_status, unsigned = _request_json(url, method="POST", body=body)
        assert unsigned_status == 401
        assert "internal auth header" in unsigned["error"]

        first_headers = sign_internal_http_request(
            secret=secret,
            method="POST",
            path=path,
            body=body,
            nonce="11" * 16,
        )
        second_headers = sign_internal_http_request(
            secret=secret,
            method="POST",
            path=path,
            body=body,
            nonce="22" * 16,
        )
        first_status, first = _request_json(
            url,
            method="POST",
            body=body,
            headers=first_headers,
        )
        second_status, second = _request_json(
            url,
            method="POST",
            body=body,
            headers=second_headers,
        )
        assert first_status == second_status == 200
        assert first["status"] == second["status"] == "accepted"

        replay_status, replay = _request_json(
            url,
            method="POST",
            body=body,
            headers=first_headers,
        )
        assert replay_status == 401
        assert replay["error"] == "internal request was already accepted"
    finally:
        _stop_server(server, thread)


def test_health_and_evm_identity_challenge_remain_public_with_internal_auth():
    account = Account.create(extra_entropy=b"mesh-worker-http-security-test")
    secret = b"ephemeral-internal-secret"
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        server_role="worker",
        internal_auth_secret=secret,
        evm_address=account.address,
        evm_private_key=account.key.hex(),
        require_proof=True,
        proof_warmup=False,
    )
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    nonce = bytes.fromhex("44" * 32)
    challenge_body = json.dumps({"nonce": nonce.hex()}, sort_keys=True).encode(
        "utf-8"
    )
    try:
        health_status, health = _request_json(f"{base_url}/health")
        assert health_status == 200
        assert health["status"] == "ok"
        assert health["proof_challenge_kind"] == "inline_every_request_v1"

        identity_status, identity = _request_json(
            f"{base_url}/identity/challenge",
            method="POST",
            body=challenge_body,
        )
        assert identity_status == 200
        assert identity["address"] == account.address
        signed_message = encode_defunct(
            primitive=nonce + bytes.fromhex(account.address[2:])
        )
        recovered = Account.recover_message(
            signed_message,
            signature=identity["signature"],
        )
        assert recovered == account.address
    finally:
        _stop_server(server, thread)


def test_authenticated_validator_inference_hides_spec_and_requires_snapshot(
    tmp_path,
):
    validator = Keypair.create_from_seed("61" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(validator.ss58_address))
    coordinator = _secure_coordinator_kwargs("62")
    capability = coordinator["capability"]
    private_worker_hotkey = "private-worker-hotkey-alpha"
    private_worker_endpoint = "http://10.44.0.2:19101"
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
        worker_hotkey=private_worker_hotkey,
        worker_endpoint=private_worker_endpoint,
    )
    backend, backend_thread, backend_url, backend_calls = _fake_backend()
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
    )
    host, port = server.server_address
    path = "/v1/mesh/inference"
    url = f"http://{host}:{port}{path}"

    def encoded_request(*, nonce_byte: str, include_spec_hash: bool) -> bytes:
        payload = {
            "openai_request": {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "verathos": {
                    "validator_nonce": nonce_byte * 64,
                    "validator_request_id": ("a" + nonce_byte) * 32,
                },
            }
        }
        if include_spec_hash:
            payload["mesh_spec_hash"] = spec.spec_hash_hex()
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    try:
        unsigned_body = encoded_request(nonce_byte="1", include_spec_hash=False)
        unsigned_status, unsigned = _request_json(
            url,
            method="POST",
            body=unsigned_body,
        )
        assert unsigned_status == 401
        assert "validator auth header" in unsigned["error"]

        supplied_body = encoded_request(nonce_byte="2", include_spec_hash=True)
        supplied_status, supplied = _request_json(
            url,
            method="POST",
            body=supplied_body,
            headers=_validator_headers(
                validator,
                method="POST",
                path=path,
                body=supplied_body,
            ),
        )
        assert supplied_status == 400
        assert supplied == {"error": "mesh_spec_hash is internal-only"}
        assert backend_calls == []

        accepted_body = encoded_request(nonce_byte="3", include_spec_hash=False)
        accepted_status, accepted = _request_json(
            url,
            method="POST",
            body=accepted_body,
            headers=_validator_headers(
                validator,
                method="POST",
                path=path,
                body=accepted_body,
            ),
        )
        assert accepted_status == 503
        assert accepted == {
            "error": (
                "authenticated mesh inference requires a verification snapshot"
            )
        }
        assert backend_calls == []
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_internal_worker_inference_still_requires_exact_mesh_spec_hash():
    secret = b"coordinator-to-private-worker-secret"
    worker_capability = CapabilityAd(
        uid=91,
        hotkey="private-worker-hotkey-alpha",
        endpoint="http://10.44.0.2:19101",
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[_MODEL_PACKAGE_HASH],
    )
    spec = _private_mesh(
        coordinator_hotkey="5PublicCoordinator",
        coordinator_endpoint="https://coordinator.example",
        worker_hotkey=worker_capability.hotkey,
        worker_endpoint=worker_capability.endpoint,
    )
    backend, backend_thread, backend_url, backend_calls = _fake_backend()
    server, thread = serve_worker_in_thread(
        capability=worker_capability,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="worker",
        internal_auth_secret=secret,
    )
    host, port = server.server_address
    path = "/v1/mesh/inference"
    url = f"http://{host}:{port}{path}"

    def send(mesh_hash: str | None) -> tuple[int, dict]:
        payload = {
            "openai_request": {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "verathos": {"validator_request_id": "a4" * 32},
            }
        }
        if mesh_hash is not None:
            payload["mesh_spec_hash"] = mesh_hash
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        return _request_json(
            url,
            method="POST",
            body=body,
            headers=sign_internal_http_request(
                secret=secret,
                method="POST",
                path=path,
                body=body,
            ),
        )

    try:
        missing_status, missing = send(None)
        assert missing_status == 409
        assert missing == {"error": "mesh_spec_hash mismatch"}

        wrong_status, wrong = send("b" * 64)
        assert wrong_status == 409
        assert wrong == {"error": "mesh_spec_hash mismatch"}

        accepted_status, accepted = send(spec.spec_hash_hex())
        assert accepted_status == 200
        assert accepted["response"]["choices"][0]["message"]["content"] == "pong"
        assert len(backend_calls) == 1

        handshake_path = "/v1/stage/handshake"
        handshake_url = f"http://{host}:{port}{handshake_path}"
        missing_handshake_body = b"{}"
        missing_handshake_status, missing_handshake = _request_json(
            handshake_url,
            method="POST",
            body=missing_handshake_body,
            headers=sign_internal_http_request(
                secret=secret,
                method="POST",
                path=handshake_path,
                body=missing_handshake_body,
            ),
        )
        assert missing_handshake_status == 409
        assert missing_handshake == {"error": "mesh_spec_hash mismatch"}

        exact_handshake_body = json.dumps(
            {"mesh_spec_hash": spec.spec_hash_hex()},
            sort_keys=True,
        ).encode("utf-8")
        exact_handshake_status, exact_handshake = _request_json(
            handshake_url,
            method="POST",
            body=exact_handshake_body,
            headers=sign_internal_http_request(
                secret=secret,
                method="POST",
                path=handshake_path,
                body=exact_handshake_body,
            ),
        )
        assert exact_handshake_status == 200
        assert exact_handshake["status"] == "accepted"
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_authenticated_validator_error_hides_private_routing_detail(
    tmp_path,
    caplog,
):
    validator = Keypair.create_from_seed("63" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(validator.ss58_address))
    coordinator = _secure_coordinator_kwargs("64")
    capability = coordinator["capability"]
    coordinator_key = Keypair.create_from_seed("64" * 32)
    private_detail = (
        "worker http://10.44.0.2:19101 failed reading "
        "/home/miner/private/trace.json"
    )
    backend, backend_thread, backend_url = _failing_backend(private_detail)
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    stage_key = Keypair.create_from_seed("65" * 32)
    spec.members[1].proof_key = stage_key.ss58_address
    spec.model_tensor_manifest_root = "66" * 32
    spec.tokenizer_hash = "67" * 32
    spec.quantization_scheme = "gguf_q4_k_m"
    spec.proof_trace_manifest_format = "compact-raw-v3"
    spec.max_context_len = 32_768
    spec.validate()
    now = int(time.time())
    policy = MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1024,
        deferred_proof_enabled=False,
    )
    snapshot = sign_mesh_verification_snapshot(
        build_mesh_verification_snapshot(
            spec,
            coordinator=MeshCoordinatorIdentity(
                chain_id=945,
                netuid=405,
                coordinator_uid=spec.coordinator_uid,
                coordinator_hotkey=spec.coordinator_hotkey,
                coordinator_evm_address=coordinator["evm_address"].lower(),
                model_index=26,
            ),
            policy=policy,
            generation=1,
            epoch=spec.epoch,
            issued_at_unix=now - 5,
            expires_at_unix=now + 600,
            stage_bindings=(
                MeshVerificationStageBinding(
                    stage_index=1,
                    stage_identity_commitment="68" * 32,
                    proof_key_scheme="sr25519",
                    proof_key=stage_key.ss58_address,
                    proof_commitment="69" * 32,
                ),
            ),
        ),
        coordinator_key,
    )
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
        require_proof=True,
        proof_sample_bps=policy.base_proof_sample_bps,
        decode_audit_bps=policy.organic_decode_sample_bps,
        proof_ops_per_request=policy.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            policy.proof_trace_candidates_per_request
        ),
        verification_snapshot_loader=lambda: snapshot.to_dict(),
    )
    host, port = server.server_address
    path = "/v1/mesh/inference"
    url = f"http://{host}:{port}{path}"
    validator_request_id = "a4" * 32
    challenge_nonce = "70" * 32
    body = json.dumps(
        {
            "openai_request": {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "verathos": {
                    "validator_request_id": validator_request_id,
                    "verification_snapshot_hash": (
                        snapshot.snapshot_hash_hex()
                    ),
                    "challenge_nonce_commitment": (
                        mesh_validator_challenge_nonce_commitment(
                            challenge_nonce,
                            validator_request_id=validator_request_id,
                            verification_snapshot_hash=(
                                snapshot.snapshot_hash_hex()
                            ),
                        )
                    ),
                },
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    caplog.set_level("ERROR", logger="verallm.mesh.worker")
    try:
        status, response = _request_json(
            url,
            method="POST",
            body=body,
            headers=_validator_headers(
                validator,
                method="POST",
                path=path,
                body=body,
            ),
        )
        assert status == 500
        assert response == {"error": "mesh inference failed"}
        assert private_detail not in json.dumps(response)
        assert private_detail in caplog.text
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_validator_proof_artifact_sanitizer_strips_private_metadata_and_rebinds():
    from verallm.mesh.ggml_proof import ggml_proof_payload_commitment_hash

    private_endpoint = "http://10.44.0.2:19101"
    private_hotkey = "private-worker-hotkey-alpha"
    payload = {
        "version": 1,
        "proof_mode": "verathos_ggml_gemm_v1",
        "trace": {
            "path": "/home/miner/private/trace-17.json",
            "tensor_name": "blk.0.ffn_down.weight",
            "trace_membership_path": ["ab" * 32],
        },
        "transport": private_endpoint,
        "worker_uid": 91,
        "owner": private_hotkey,
    }
    payload["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(payload)
    original_commitment = payload["proof_commitment_hash"]
    receipt = {
        "proof_commitment_hash": original_commitment,
        "signature": "private-worker-signature",
    }

    receipts, payloads = _sanitize_validator_proof_artifacts(
        [receipt],
        [payload],
        private_tokens=(private_endpoint, private_hotkey),
    )

    sanitized = payloads[0]
    assert "path" not in sanitized["trace"]
    assert sanitized["trace"]["trace_membership_path"] == ["ab" * 32]
    assert "transport" not in sanitized
    assert "worker_uid" not in sanitized
    assert "owner" not in sanitized
    assert sanitized["proof_commitment_hash"] != original_commitment
    assert receipts[0]["proof_commitment_hash"] == sanitized[
        "proof_commitment_hash"
    ]
    assert receipts[0]["signature"] == ""


def test_validator_proof_artifact_sanitizer_rejects_private_list_value():
    from verallm.mesh.ggml_proof import ggml_proof_payload_commitment_hash

    payload = {
        "version": 1,
        "proof_mode": "verathos_ggml_gemm_v1",
        "opaque_values": ["safe", "http://10.44.0.2:19101"],
    }
    payload["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(payload)
    receipt = {"proof_commitment_hash": payload["proof_commitment_hash"]}
    with pytest.raises(RuntimeError, match="private routing or filesystem"):
        _sanitize_validator_proof_artifacts([receipt], [payload])


@pytest.mark.parametrize(
    "private_artifact",
    [
        {"trace": {"path": "/home/miner/private/trace.json"}},
        {"worker_uid": 91},
        {
            "owner": str(
                Keypair.create_from_seed("65" * 32).ss58_address
            )
        },
        {"transport": "http://10.44.0.2:19101"},
    ],
)
def test_validator_proof_artifact_guard_rejects_private_identity_and_routing(
    private_artifact,
):
    with pytest.raises(RuntimeError, match="private|routing|filesystem"):
        _assert_public_proof_artifact_value(private_artifact)


def test_secure_coordinator_startup_rejects_missing_receipt_signer(tmp_path):
    keypair = Keypair.create_from_seed("55" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(keypair.ss58_address))

    with pytest.raises(
        ValueError,
        match="validator-authenticated coordinators require receipt signing",
    ):
        make_worker_server(
            capability=_capability(),
            host="127.0.0.1",
            port=0,
            server_role="coordinator",
            validator_auth_enabled=True,
            validator_allowlist_path=allowlist_path,
        )


def test_startup_rejects_mismatched_evm_private_key():
    configured_identity = Account.create(extra_entropy=b"configured-identity")
    wrong_key = Account.create(extra_entropy=b"wrong-private-key")

    with pytest.raises(ValueError, match="evm_private_key does not match evm_address"):
        make_worker_server(
            capability=_capability(),
            host="127.0.0.1",
            port=0,
            server_role="worker",
            evm_address=configured_identity.address,
            evm_private_key=wrong_key.key.hex(),
        )


def test_snapshot_bound_coordinator_requires_validator_auth_and_fresh_nonce(
    tmp_path,
):
    kwargs = _secure_coordinator_kwargs("73")
    with pytest.raises(ValueError, match="validator auth and fresh nonces"):
        make_worker_server(
            **kwargs,
            host="127.0.0.1",
            port=0,
            server_role="coordinator",
            verification_snapshot_loader=lambda: {},
        )

    validator = Keypair.create_from_seed("74" * 32)
    allowlist = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist, str(validator.ss58_address))
    with pytest.raises(ValueError, match="validator auth and fresh nonces"):
        make_worker_server(
            **kwargs,
            host="127.0.0.1",
            port=0,
            server_role="coordinator",
            validator_auth_enabled=True,
            validator_allowlist_path=allowlist,
            verification_snapshot_loader=lambda: {},
        )


def test_validator_routes_deny_fallthrough_and_dev_bypass_is_loopback_explicit():
    denied, denied_thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="coordinator",
    )
    try:
        host, port = denied.server_address
        status, payload = _request_json(
            f"http://{host}:{port}/v1/mesh/verification-snapshot"
        )
        assert status == 403
        assert payload["error"] == "validator authentication is required"
    finally:
        _stop_server(denied, denied_thread)

    dev, dev_thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="coordinator",
        allow_loopback_dev_validator_routes=True,
        require_proof=True,
        proof_warmup=False,
    )
    try:
        host, port = dev.server_address
        health_status, health = _request_json(f"http://{host}:{port}/health")
        assert health_status == 200
        assert health["proof_challenge_kind"] == "inline_every_request_v1"
        status, payload = _request_json(
            f"http://{host}:{port}/v1/mesh/verification-snapshot"
        )
        assert status == 404
        assert payload["error"] == "verification snapshot unavailable"
    finally:
        _stop_server(dev, dev_thread)


def test_loopback_internal_hmac_can_run_operator_self_test_route():
    secret = "local-operator-secret"
    server, thread = serve_worker_in_thread(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        server_role="coordinator",
        internal_auth_secret=secret,
    )
    try:
        host, port = server.server_address
        path = "/v1/mesh/verification-snapshot"
        headers = sign_internal_http_request(
            secret=secret,
            method="GET",
            path=path,
            body=b"",
        )
        status, payload = _request_json(
            f"http://{host}:{port}{path}", headers=headers
        )
        assert status == 404
        assert payload["error"] == "verification snapshot unavailable"
    finally:
        _stop_server(server, thread)


def test_snapshot_bound_health_is_static_and_never_leaks_trace_path(tmp_path):
    validator = Keypair.create_from_seed("75" * 32)
    allowlist = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist, str(validator.ss58_address))
    private_trace = tmp_path / "operator" / "private-traces"

    def must_not_load_snapshot():
        raise AssertionError("health must not load request-bound snapshot state")

    server, thread = serve_worker_in_thread(
        **_secure_coordinator_kwargs("76"),
        host="127.0.0.1",
        port=0,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist,
        require_validator_nonce=True,
        verification_snapshot_loader=must_not_load_snapshot,
        require_proof=True,
        proof_trace_dir=private_trace,
        proof_warmup=False,
    )
    try:
        host, port = server.server_address
        status, payload = _request_json(f"http://{host}:{port}/health")
        assert status == 200
        assert payload["embedded_proof_configured"] is True
        assert payload["proof_challenge_kind"] == (
            VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
        )
        assert "proof_trace_dir" not in payload
        assert str(private_trace) not in json.dumps(payload)
    finally:
        _stop_server(server, thread)


def test_snapshot_mismatch_is_reported_as_a_distinct_non_retryable_class(
    tmp_path,
):
    """A pinned-snapshot refusal must be distinguishable from an outage.

    A coordinator that answers with a generic 500 lets snapshot rotation be
    scored as unavailability instead of as a refusal to be verified, which is
    an epoch of free canary evasion.
    """

    validator = Keypair.create_from_seed("73" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist_path, str(validator.ss58_address))
    coordinator = _secure_coordinator_kwargs("74")
    capability = coordinator["capability"]
    coordinator_key = Keypair.create_from_seed("74" * 32)
    backend, backend_thread, backend_url = _failing_backend("unused")
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    stage_key = Keypair.create_from_seed("75" * 32)
    spec.members[1].proof_key = stage_key.ss58_address
    spec.model_tensor_manifest_root = "76" * 32
    spec.tokenizer_hash = "77" * 32
    spec.quantization_scheme = "gguf_q4_k_m"
    spec.proof_trace_manifest_format = "compact-raw-v3"
    spec.max_context_len = 32_768
    spec.validate()
    now = int(time.time())
    policy = MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1024,
        deferred_proof_enabled=False,
    )
    snapshot = sign_mesh_verification_snapshot(
        build_mesh_verification_snapshot(
            spec,
            coordinator=MeshCoordinatorIdentity(
                chain_id=945,
                netuid=405,
                coordinator_uid=spec.coordinator_uid,
                coordinator_hotkey=spec.coordinator_hotkey,
                coordinator_evm_address=coordinator["evm_address"].lower(),
                model_index=26,
            ),
            policy=policy,
            generation=1,
            epoch=spec.epoch,
            issued_at_unix=now - 5,
            expires_at_unix=now + 600,
            stage_bindings=(
                MeshVerificationStageBinding(
                    stage_index=1,
                    stage_identity_commitment="78" * 32,
                    proof_key_scheme="sr25519",
                    proof_key=stage_key.ss58_address,
                    proof_commitment="79" * 32,
                ),
            ),
        ),
        coordinator_key,
    )
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
        require_proof=True,
        proof_sample_bps=policy.base_proof_sample_bps,
        decode_audit_bps=policy.organic_decode_sample_bps,
        proof_ops_per_request=policy.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            policy.proof_trace_candidates_per_request
        ),
        verification_snapshot_loader=lambda: snapshot.to_dict(),
    )
    host, port = server.server_address
    path = "/v1/mesh/inference"
    url = f"http://{host}:{port}{path}"
    validator_request_id = "a5" * 32
    challenge_nonce = "80" * 32
    # A snapshot hash the coordinator does not serve, as seen by a validator
    # that pinned generation N while the coordinator rotated to N+1.
    stale_snapshot_hash = "5c" * 32
    body = json.dumps(
        {
            "openai_request": {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "verathos": {
                    "validator_request_id": validator_request_id,
                    "verification_snapshot_hash": stale_snapshot_hash,
                    "challenge_nonce_commitment": (
                        mesh_validator_challenge_nonce_commitment(
                            challenge_nonce,
                            validator_request_id=validator_request_id,
                            verification_snapshot_hash=stale_snapshot_hash,
                        )
                    ),
                },
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        status, response = _request_json(
            url,
            method="POST",
            body=body,
            headers=_validator_headers(
                validator,
                method="POST",
                path=path,
                body=body,
            ),
        )
        assert status == 409
        assert response["error_code"] == "verification_snapshot_mismatch"
        assert response["retryable"] is False
        # The live snapshot hash must not leak back to the caller.
        assert snapshot.snapshot_hash_hex() not in json.dumps(response)
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_authenticated_validator_never_gets_the_grindable_inline_challenge():
    """inline_every_request_v1 must not serve a validator.

    Its beacon is SHA256(tag || gate_hash), which the coordinator computes
    itself and can grind by regenerating the response until it likes the draw.
    Reaching that branch for an authenticated validator means the postcommit
    requirement was lost upstream, which is the auth-regression case the
    branch has to fail closed on rather than silently downgrade.
    """

    import inspect

    from verallm.mesh import worker as mesh_worker

    source = inspect.getsource(mesh_worker.make_worker_server)
    marker = 'challenge_kind = "inline_every_request_v1"'
    assert marker in source
    guard = source[: source.index(marker)]
    assert "if bool(validator_authenticated):" in guard
    assert "grindable inline challenge" in guard


def test_private_value_detection_semantics_survive_the_fast_path():
    """The separator gate must not weaken detection: every network/path
    pattern requires one of ':/\\~', so gating the regexes on those
    characters is purely an optimization. SS58 and token-substring checks
    still run on separator-free strings (a container-id token IS pure hex
    and must still be caught inside a digest-like string)."""
    from verallm.mesh.worker import _looks_like_private_proof_artifact_value

    private = [
        "http://203.0.113.137:20043",
        "203.0.113.137:20043",
        "localhost:9402",
        "my-host.example.com:443",
        "/home/user/secret",
        "C:\\Users\\op\\wallet",
        "~/private/keys",
        "5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv",
    ]
    for value in private:
        assert _looks_like_private_proof_artifact_value(value), value
    public = [
        "ab" * 32,  # digest
        "deadbeef",
        "stage-1",
        "verathos_ggml_light_v1",
        "",
    ]
    for value in public:
        assert not _looks_like_private_proof_artifact_value(value), value
    # Separator-free token (container id, pure hex) inside a hex blob.
    assert _looks_like_private_proof_artifact_value(
        "00" * 8 + "4f02b214a23f" + "00" * 8,
        private_tokens=("4f02b214a23f",),
    )


def test_sanitizer_scalar_fast_paths_keep_structure_and_checks():
    """Huge numeric arrays pass through in one pass (the glm LM-head hard
    payload ground the per-element walk for 20+ minutes while holding the
    GIL, wedging the proof port); string lists still reject private
    values with the element index in the error."""
    import pytest as _pytest

    from verallm.mesh.worker import (
        _assert_public_proof_artifact_value,
        _sanitize_public_proof_artifact_value,
    )

    numbers = list(range(100_000))
    digests = ["ab" * 32] * 1000
    payload = {
        "logits_i32": numbers,
        "merkle_path": digests,
        "nested": [{"row": numbers}],
    }
    sanitized = _sanitize_public_proof_artifact_value(
        payload, private_tokens=("http://10.0.0.1:9402",), path="p"
    )
    assert sanitized["logits_i32"] == numbers
    assert sanitized["merkle_path"] == digests
    assert sanitized["nested"][0]["row"] == numbers
    _assert_public_proof_artifact_value(
        sanitized, private_tokens=("http://10.0.0.1:9402",)
    )
    with _pytest.raises(RuntimeError, match=r"\[1\] contains private"):
        _sanitize_public_proof_artifact_value(
            ["ab" * 32, "203.0.113.137:20043"],
            private_tokens=(),
            path="p",
        )
    with _pytest.raises(RuntimeError, match="contains private"):
        _assert_public_proof_artifact_value(
            {"paths": ["ok", "/home/user/x"]}, private_tokens=()
        )


def test_sanitizer_keeps_merkle_path_keys_but_strips_path_values():
    """Merkle openings carry sibling lists under the literal key "path"
    (pcs w-col manifest groups); the key-level strip silently deleted
    them and EVERY payload challenging a w-col-rooted tensor failed
    membership with path_len=0 (observed on glm router tensors).
    The key must survive; an actual filesystem-path VALUE under the same
    key must still go."""
    from verallm.mesh.worker import (
        _assert_public_proof_artifact_value,
        _is_private_proof_artifact_key,
        _sanitize_public_proof_artifact_value,
    )

    assert not _is_private_proof_artifact_key("path")
    # The metadata spellings stay private.
    for key in ("trace_path", "file_path", "local_path", "worker_trace_path"):
        assert _is_private_proof_artifact_key(key), key

    opening = {
        "blocks": [
            {
                "row_block": 0,
                "column_block": 0,
                "groups": [
                    {
                        "group_index": 0,
                        "commitments": ["ab" * 32] * 4,
                        "path": ["cd" * 32, "ef" * 32],
                    }
                ],
            }
        ]
    }
    sanitized = _sanitize_public_proof_artifact_value(
        opening, private_tokens=(), path="p"
    )
    assert sanitized["blocks"][0]["groups"][0]["path"] == [
        "cd" * 32,
        "ef" * 32,
    ]
    _assert_public_proof_artifact_value(sanitized, private_tokens=())
    # A filesystem path VALUE under the bare key is still removed.
    leaky = _sanitize_public_proof_artifact_value(
        {"path": "/home/user/secret"}, private_tokens=(), path="p"
    )
    assert "path" not in leaky


def _blocking_backend() -> tuple[
    ThreadingHTTPServer,
    threading.Thread,
    str,
    list[dict],
    threading.Event,
    threading.Event,
]:
    """A chat backend whose FIRST response blocks until released."""

    calls: list[dict] = []
    started = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.rstrip("/") != "/v1/chat/completions":
                # Tokenizer probes (admission demand) are not completions.
                body = json.dumps({"error": "not found"}).encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            calls.append(payload)
            if len(calls) == 1:
                started.set()
                release.wait(30)
            body = json.dumps(
                {
                    "id": "chatcmpl-admission-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "mesh-test"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "pong",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:  # noqa: N802
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}", calls, started, release


def test_chat_admission_busy_is_instant_uniform_and_burns_nothing(tmp_path):
    """Plan Part B route proof: a request that cannot fit RIGHT NOW gets
    an instant 503 slots_busy BEFORE the backend sees it, and the very
    same request succeeds once the pool drains - the refusal consumed
    nothing (no queue, owner decision; router failover is the UX)."""

    coordinator = _secure_coordinator_kwargs("83")
    capability = coordinator["capability"]
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    backend, backend_thread, backend_url, calls, started, release = (
        _blocking_backend()
    )
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        llama_n_parallel=1,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(
        {
            "model": "mesh-security-model",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "stream": False,
        },
        sort_keys=True,
    ).encode("utf-8")

    first: dict = {}

    def _first_request() -> None:
        status, payload = _request_json(url, method="POST", body=body)
        first["status"] = status
        first["payload"] = payload

    worker_thread = threading.Thread(target=_first_request, daemon=True)
    try:
        worker_thread.start()
        assert started.wait(10)

        busy_status, busy = _request_json(url, method="POST", body=body)
        assert busy_status == 503
        assert busy["type"] == "slots_busy"
        assert busy["retryable"] is True
        # The refusal never reached the backend: one call in flight only.
        assert len(calls) == 1

        release.set()
        worker_thread.join(20)
        assert first["status"] == 200

        retry_status, retry = _request_json(url, method="POST", body=body)
        assert retry_status == 200
        assert retry["choices"][0]["message"]["content"] == "pong"
        assert len(calls) == 2
    finally:
        release.set()
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_chat_admission_oversized_is_physical_budget_not_contract(tmp_path):
    """OVERSIZED means the demand can never fit the LAUNCHED unified KV
    pool. The registered contract must NOT gate admission: kv-unified
    lets one request use the full launched context, and reading the
    stale chain contract made a RE-deploy refuse its own measurement
    probes .
    Over-physical demand still 400s with zero backend traffic."""

    coordinator = _secure_coordinator_kwargs("84")
    capability = coordinator["capability"]
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    spec.max_context_len = 64  # stale registered contract, must not gate
    backend, backend_thread, backend_url, calls, _started, release = (
        _blocking_backend()
    )
    release.set()  # never block: admitted requests complete instantly
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        llama_n_parallel=4,
        llama_ctx_budget=4_096,  # the launched physical pool
        allow_loopback_dev_validator_routes=True,
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/v1/chat/completions"
    try:
        # Above the stale contract but inside the physical pool: FLOWS
        # (this is exactly a re-deploy measurement probe).
        over_contract = json.dumps(
            {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 100,
                "stream": False,
            },
            sort_keys=True,
        ).encode("utf-8")
        ok_status, ok_payload = _request_json(
            url, method="POST", body=over_contract
        )
        assert ok_status == 200
        assert ok_payload["choices"][0]["message"]["content"] == "pong"
        assert len(calls) == 1

        # Beyond the physical pool: can never fit, uniform 400, no
        # backend traffic for the refused request.
        over_physical = json.dumps(
            {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5_000,
                "stream": False,
            },
            sort_keys=True,
        ).encode("utf-8")
        status, payload = _request_json(
            url, method="POST", body=over_physical
        )
        assert status == 400
        assert payload["type"] == "exceed_context_size_error"
        assert len(calls) == 1

        # The refusal reserved nothing: a small request still flows.
        small = json.dumps(
            {
                "model": "mesh-security-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": False,
            },
            sort_keys=True,
        ).encode("utf-8")
        ok_status, ok_payload = _request_json(url, method="POST", body=small)
        assert ok_status == 200
        assert ok_payload["choices"][0]["message"]["content"] == "pong"
        assert len(calls) == 2
    finally:
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_capacity_drain_gate_503_is_byte_identical_to_ledger_busy(tmp_path):
    """During a capacity-audit drain window admission refuses with the
    ORDINARY busy 503. The payload must be byte-identical to the admission
    ledger's saturation rejection: any distinguishable detail (message,
    extra field, audit id) would hand validators an oracle for audit
    windows, breaking canary/organic indistinguishability."""

    coordinator = _secure_coordinator_kwargs("85")
    capability = coordinator["capability"]
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    backend, backend_thread, backend_url, calls, started, release = (
        _blocking_backend()
    )
    drain_file = tmp_path / "capacity-audit-drain.json"
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        backend_url=backend_url,
        server_role="coordinator",
        llama_n_parallel=1,
        allow_loopback_dev_validator_routes=True,
        capacity_drain_file=str(drain_file),
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(
        {
            "model": "mesh-security-model",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "stream": False,
        },
        sort_keys=True,
    ).encode("utf-8")

    first: dict = {}

    def _first_request() -> None:
        status, payload = _request_json(url, method="POST", body=body)
        first["status"] = status
        first["payload"] = payload

    try:
        # Reference busy payload from a genuinely saturated ledger.
        worker_thread = threading.Thread(target=_first_request, daemon=True)
        worker_thread.start()
        assert started.wait(10)
        busy_status, ledger_busy = _request_json(url, method="POST", body=body)
        assert busy_status == 503
        release.set()
        worker_thread.join(20)
        assert first["status"] == 200

        # Active drain: same 503, byte-identical payload, backend untouched.
        drain_file.write_text(
            json.dumps(
                {
                    "active": True,
                    "reason": "capacity_audit",
                    "audit_id": "aa" * 32,
                    "until_ts": time.time() + 120.0,
                }
            )
        )
        backend_calls_before = len(calls)
        drain_status, drain_busy = _request_json(url, method="POST", body=body)
        assert drain_status == 503
        assert drain_busy == ledger_busy
        assert len(calls) == backend_calls_before

        # Expired drain: serving resumes without any file cleanup.
        drain_file.write_text(
            json.dumps({"active": True, "until_ts": time.time() - 1.0})
        )
        ok_status, ok_payload = _request_json(url, method="POST", body=body)
        assert ok_status == 200
        assert ok_payload["choices"][0]["message"]["content"] == "pong"

        # Cleared drain file: serving stays up.
        drain_file.write_text(json.dumps({"active": False}))
        ok_status, _ = _request_json(url, method="POST", body=body)
        assert ok_status == 200
    finally:
        release.set()
        _stop_server(server, thread)
        _stop_server(backend, backend_thread)


def test_capacity_roster_route_serves_signed_roster(tmp_path):
    """GET /capacity/roster is the validator's PULL path for a mesh's GPU
    obligation: 404 until the daemon publishes a signed roster, 200 with
    the exact document afterwards, no auth wall.

    Internal auth is configured HERE exactly as in production: the route
    must stay public even then (a 401 on this route means no validator can
    ever learn the roster, silently exempting the mesh from capacity
    audits — the regression that shipped on 08-12)."""

    coordinator = _secure_coordinator_kwargs("86")
    capability = coordinator["capability"]
    spec = _private_mesh(
        coordinator_hotkey=capability.hotkey,
        coordinator_endpoint=capability.endpoint,
    )
    roster_file = tmp_path / "capacity-roster.json"
    server, thread = serve_worker_in_thread(
        **coordinator,
        mesh_spec=spec,
        server_role="coordinator",
        allow_loopback_dev_validator_routes=True,
        internal_auth_secret=b"ephemeral-internal-secret",
        capacity_roster_file=str(roster_file),
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/capacity/roster"
    try:
        status, payload = _request_json(url)
        assert status == 404

        roster_file.write_text(
            json.dumps(
                {
                    "roster": {"version": 1, "workers": []},
                    "roster_signature": "ab" * 65,
                }
            )
        )
        status, payload = _request_json(url)
        assert status == 200
        assert payload["roster"] == {"version": 1, "workers": []}
        assert payload["roster_signature"] == "ab" * 65

        # A truncated/corrupt file must fail closed as absent, not 500.
        roster_file.write_text("{not json")
        status, _payload = _request_json(url)
        assert status == 404
    finally:
        _stop_server(server, thread)
