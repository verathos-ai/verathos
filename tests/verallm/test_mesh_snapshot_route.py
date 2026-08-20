"""End-to-end coverage for the authenticated verification-snapshot route."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from eth_account import Account
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons.mesh_snapshot import (
    MESH_VERIFICATION_SNAPSHOT_PATH,
    MeshSnapshotExpectations,
    fetch_and_pin_mesh_verification_snapshot,
)
from neurons.request_signing import verify_request
from neurons.validator_db import ValidatorStateDB
from verallm.mesh import CapabilityAd, MeshMember, MeshSpec, StageRange
from verallm.mesh.receipt_signing import sign_receipt_hash
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
)
from verallm.mesh.worker import serve_worker_in_thread


COORDINATOR_SEED = bytes.fromhex("71" * 32)
VALIDATOR_SEED = bytes.fromhex("72" * 32)
COORDINATOR_KEYPAIR = Keypair.create_from_seed(COORDINATOR_SEED.hex())
VALIDATOR_KEYPAIR = Keypair.create_from_seed(VALIDATOR_SEED.hex())
COORDINATOR_EVM = Account.from_key(bytes.fromhex("73" * 32))


def _mesh_spec() -> MeshSpec:
    spec = MeshSpec(
        mesh_id="mesh_snapshot_route_e2e",
        mode="private",
        coordinator_uid=7,
        coordinator_hotkey=COORDINATOR_KEYPAIR.ss58_address,
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="a1" * 32,
        model_tensor_manifest_root="b2" * 32,
        tokenizer_hash="c3" * 32,
        quantization_scheme="gguf_q4_k_m",
        activation_dtype="f16",
        proof_trace_manifest_format="compact-raw-v3",
        total_layers=16,
        max_context_len=32_768,
        members=[
            MeshMember(
                uid=7,
                hotkey=COORDINATOR_KEYPAIR.ss58_address,
                endpoint="http://127.0.0.1:19100",
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
            ),
            MeshMember(
                uid=101,
                hotkey="private-worker-alpha",
                proof_key=Keypair.create_from_uri(
                    "//MeshSnapshotRouteStage1"
                ).ss58_address,
                endpoint="http://10.44.0.2:19101",
                proof_endpoint="http://10.44.0.2:19201/proof",
                rpc_endpoint="10.44.0.2:50052",
                stage_index=1,
                layers=StageRange(0, 8),
            ),
            MeshMember(
                uid=102,
                hotkey="private-worker-beta",
                proof_key=Keypair.create_from_uri(
                    "//MeshSnapshotRouteStage2"
                ).ss58_address,
                endpoint="http://10.44.0.3:19102",
                proof_endpoint="http://10.44.0.3:19202/proof",
                rpc_endpoint="10.44.0.3:50053",
                stage_index=2,
                layers=StageRange(8, 16),
            ),
        ],
        epoch=77,
    )
    spec.validate()
    return spec


def _policy() -> MeshVerificationPolicy:
    return MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=3,
        deferred_proof_enabled=False,
    )


def _binding(stage_index: int) -> MeshVerificationStageBinding:
    return MeshVerificationStageBinding(
        stage_index=stage_index,
        stage_identity_commitment=f"{stage_index + 1:02x}" * 32,
        proof_key_scheme="sr25519",
        proof_key=Keypair.create_from_uri(
            f"//MeshSnapshotRouteStage{stage_index}"
        ).ss58_address,
        proof_commitment=f"{stage_index + 16:02x}" * 32,
    )


def _write_allowlist(path) -> None:
    path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()),
                "netuid": 405,
                "validators": [
                    {
                        "uid": 2,
                        "hotkey_ss58": VALIDATOR_KEYPAIR.ss58_address,
                        "stake": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_authenticated_snapshot_route_round_trip_pins_endpoint_free_snapshot(
    tmp_path,
) -> None:
    now = int(time.time())
    spec = _mesh_spec()
    coordinator = MeshCoordinatorIdentity(
        chain_id=945,
        netuid=405,
        coordinator_uid=spec.coordinator_uid,
        coordinator_hotkey=spec.coordinator_hotkey,
        coordinator_evm_address=COORDINATOR_EVM.address.lower(),
        model_index=26,
    )
    policy = _policy()
    snapshot = sign_mesh_verification_snapshot(
        build_mesh_verification_snapshot(
            spec,
            coordinator=coordinator,
            policy=policy,
            generation=9,
            epoch=spec.epoch,
            issued_at_unix=now - 5,
            expires_at_unix=now + 600,
            stage_bindings=(_binding(1), _binding(2)),
        ),
        COORDINATOR_KEYPAIR,
    )
    assert snapshot.model.max_context_len == 32_768
    expectations = MeshSnapshotExpectations(
        mesh_id=spec.mesh_id,
        generation=9,
        epoch=spec.epoch,
        coordinator=coordinator,
        model=snapshot.model,
        policy=policy,
    )

    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path)
    state_db = ValidatorStateDB(db_path=str(tmp_path / "validator.db"))
    server, thread = serve_worker_in_thread(
        capability=CapabilityAd(
            uid=spec.coordinator_uid,
            hotkey=spec.coordinator_hotkey,
            endpoint="http://127.0.0.1:19100",
            supported_backends=["gguf_stage"],
            cached_model_package_hashes=[spec.model_package_hash],
        ),
        mesh_spec=spec,
        server_role="coordinator",
        validator_auth_enabled=True,
        validator_allowlist_path=allowlist_path,
        require_validator_nonce=True,
        receipt_signer=lambda receipt_hash: sign_receipt_hash(
            receipt_hash,
            COORDINATOR_KEYPAIR,
        ),
        evm_address=COORDINATOR_EVM.address,
        evm_private_key=COORDINATOR_EVM.key.hex(),
        proof_sample_bps=policy.base_proof_sample_bps,
        decode_audit_bps=policy.organic_decode_sample_bps,
        proof_ops_per_request=policy.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            policy.proof_trace_candidates_per_request
        ),
        defer_proof=policy.deferred_proof_enabled,
        verification_snapshot_loader=lambda: snapshot.to_dict(),
    )
    host, port = server.server_address
    coordinator_origin = f"http://{host}:{port}"
    snapshot_url = coordinator_origin + MESH_VERIFICATION_SNAPSHOT_PATH
    captured: dict[str, object] = {}

    def fetch(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = dict(headers)
        with urlopen(
            Request(url, headers=dict(headers), method="GET"),
            timeout=timeout,
        ) as response:
            raw = response.read()
        captured["raw"] = raw
        return raw

    try:
        try:
            urlopen(Request(snapshot_url, method="GET"), timeout=5.0)
        except HTTPError as exc:
            assert exc.code == 401
            assert "validator auth header" in exc.read().decode("utf-8")
        else:
            raise AssertionError("unsigned validator snapshot request was accepted")

        pinned = fetch_and_pin_mesh_verification_snapshot(
            coordinator_endpoint=coordinator_origin,
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            expectations=expectations,
            state_db=state_db,
            timeout=5.0,
            fetch=fetch,
            clock=lambda: float(now),
        )

        assert pinned == snapshot
        assert captured["url"] == snapshot_url
        headers = captured["headers"]
        assert isinstance(headers, dict)
        assert verify_request(
            method="GET",
            path=MESH_VERIFICATION_SNAPSHOT_PATH,
            body=b"",
            hotkey_ss58=headers["X-Validator-Hotkey"],
            signature_hex=headers["X-Validator-Signature"],
            timestamp_str=headers["X-Validator-Timestamp"],
        ) == (True, "ok")

        raw = captured["raw"]
        assert isinstance(raw, bytes)
        assert json.loads(raw) == snapshot.to_dict()
        public_text = raw.decode("utf-8")
        for private_value in (
            "private-worker-alpha",
            "private-worker-beta",
            "10.44.0.2",
            "10.44.0.3",
            "50052",
            "50053",
            "19101",
            "19102",
            "19201",
            "19202",
        ):
            assert private_value not in public_text
        assert "endpoint" not in public_text.lower()

        row = state_db.get_latest_mesh_verification_snapshot_for_epoch(
            coordinator_address=coordinator.coordinator_evm_address,
            model_index=coordinator.model_index,
            epoch=spec.epoch,
        )
        assert row is not None
        assert row["generation"] == snapshot.generation
        assert row["snapshot_hash"] == snapshot.snapshot_hash_hex()
        assert json.loads(row["snapshot_json"]) == snapshot.to_dict()
        assert "endpoint" not in row["snapshot_json"].lower()
    finally:
        state_db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
