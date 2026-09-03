"""Manager-delegated coordinator signing for token-only workers.

The pool manager KEEPS the coordinator wallet (subnet pools require it);
these tests pin down that a worker holding nothing but its join token and
stage key can obtain exactly the mesh signatures a driver owes a validator,
and nothing else.
"""

from __future__ import annotations

import hashlib
import json

import pytest

try:
    from bittensor_wallet import Keypair
except ImportError:  # pragma: no cover - legacy stack
    from substrateinterface import Keypair

from verallm.mesh.delegated_signing import (
    IDENTITY_CHALLENGE_PURPOSE,
    ManagerDelegateKeypair,
    coordinator_delegation_from_file,
    coordinator_sign_retry_profile,
    coordinator_sign_request_with_retry,
    coordinator_sign_message,
    delegate_keypair_from_worker_request,
    split_coordinator_sign_message,
)
from verallm.mesh.pool import PoolManager, create_pool_state
from verallm.mesh.receipt_signing import (
    RECEIPT_SIGNATURE_DOMAIN,
    ensure_stage_proof_key_file,
    sign_receipt_hash,
    verify_receipt_signature,
)
from verallm.mesh.verification_snapshot import SNAPSHOT_SIGNATURE_DOMAIN

BODY_HASH = hashlib.sha256(b"body").hexdigest()
COORDINATOR_SEED = b"\x11" * 32


def _coordinator_keypair() -> Keypair:
    return Keypair.create_from_seed(COORDINATOR_SEED.hex())


# ---------------------------------------------------------------------------
# Message discipline: only mesh snapshots and receipts are signable.
# ---------------------------------------------------------------------------


def test_sign_message_round_trip_both_purposes():
    for purpose, domain in (
        ("verification-snapshot", SNAPSHOT_SIGNATURE_DOMAIN),
        ("receipt", RECEIPT_SIGNATURE_DOMAIN),
    ):
        message = coordinator_sign_message(purpose, BODY_HASH)
        assert message == domain + BODY_HASH.encode("ascii")
        assert split_coordinator_sign_message(message) == (purpose, BODY_HASH)


def test_sign_message_rejects_unknown_purpose_and_bad_hash():
    with pytest.raises(ValueError):
        coordinator_sign_message("extrinsic", BODY_HASH)
    with pytest.raises(ValueError):
        coordinator_sign_message("receipt", "not-a-hash")
    with pytest.raises(ValueError):
        coordinator_sign_message("receipt", BODY_HASH.upper())


def test_split_rejects_non_mesh_messages():
    # A substrate extrinsic payload, or any free-form bytes, must never be
    # classified as delegatable.
    for message in (
        b"some arbitrary transaction bytes",
        b"verathos-mesh-receipt-v1:" + b"zz" * 32,
        SNAPSHOT_SIGNATURE_DOMAIN + b"deadbeef",  # short hash
        b"",
    ):
        with pytest.raises(ValueError):
            split_coordinator_sign_message(message)


def test_coordinator_sign_transport_retries_only_transient_failures():
    calls: list[float] = []
    delays: list[float] = []

    def request_once(timeout_s: float):
        calls.append(timeout_s)
        if len(calls) == 1:
            raise RuntimeError("HTTP 503 from manager: temporarily unavailable")
        if len(calls) == 2:
            raise RuntimeError("failed to connect to manager: timed out")
        return {"signature": "ab" * 64}

    response = coordinator_sign_request_with_retry(
        request_once,
        attempts=3,
        attempt_timeout_s=7.0,
        retry_delays_s=(0.1, 0.2),
        sleep=delays.append,
    )
    assert response == {"signature": "ab" * 64}
    assert calls == [7.0, 7.0, 7.0]
    assert delays == [0.1, 0.2]


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503, 599])
def test_coordinator_sign_transport_retries_transient_http_statuses(status):
    calls = 0

    def request_once(_timeout_s: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(f"HTTP {status} from manager: transient")
        return {"status": "ok"}

    assert coordinator_sign_request_with_retry(
        request_once,
        attempts=2,
        retry_delays_s=(0.0,),
        sleep=lambda _delay: None,
    ) == {"status": "ok"}
    assert calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 409, 422])
def test_coordinator_sign_transport_does_not_retry_semantic_failures(status):
    calls = 0

    def request_once(_timeout_s: float):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"HTTP {status} from manager: refused")

    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        coordinator_sign_request_with_retry(
            request_once,
            attempts=3,
            retry_delays_s=(0.0, 0.0),
            sleep=lambda _delay: None,
        )
    assert calls == 1


def test_deadline_sensitive_keepalive_failure_skips_duplicate_urlopen(
    monkeypatch,
):
    import http.client

    from verallm.mesh import worker as worker_mod

    connections = []

    class FailedConnection:
        def __init__(self, *_args, **_kwargs):
            connections.append(self)

        def request(self, *_args, **_kwargs):
            raise TimeoutError("route timed out")

        def close(self):
            pass

    monkeypatch.setattr(
        worker_mod,
        "_PINNED_TLS_CONTEXTS",
        {("manager.example", 443): object()},
    )
    monkeypatch.setattr(worker_mod, "_KEEPALIVE_CONNS", {})
    monkeypatch.setattr(http.client, "HTTPSConnection", FailedConnection)

    with pytest.raises(RuntimeError, match="failed to connect"):
        worker_mod._post_json_keepalive(
            "https://manager.example/v1/pool/coordinator-sign",
            b"{}",
            {"Content-Type": "application/json"},
            timeout=8.0,
            fallback_on_error=False,
        )
    assert len(connections) == 1


def test_capacity_sign_retry_profile_stays_inside_artifact_delivery_grace():
    attempts, timeout_s, delays = coordinator_sign_retry_profile(
        "capacity-audit-artifact"
    )
    assert attempts == 3
    assert attempts * timeout_s + sum(delays) < 10.0

    receipt_attempts, receipt_timeout_s, _ = coordinator_sign_retry_profile(
        "receipt"
    )
    assert receipt_attempts == 3
    assert receipt_timeout_s > timeout_s


# ---------------------------------------------------------------------------
# Delegate keypair: signs through the manager, verifies locally.
# ---------------------------------------------------------------------------


def _manager_backed_request(keypair: Keypair):
    def request(action, fields):
        assert action == "coordinator-sign"
        message = coordinator_sign_message(fields["purpose"], fields["body_hash"])
        return {
            "status": "ok",
            "signature": keypair.sign(message).hex(),
            "coordinator_hotkey": keypair.ss58_address,
        }

    return request


def test_delegate_keypair_signs_receipts_verifiably():
    coordinator = _coordinator_keypair()
    delegate = delegate_keypair_from_worker_request(
        coordinator_hotkey=coordinator.ss58_address,
        mesh_key="m-1",
        request=_manager_backed_request(coordinator),
    )
    assert delegate.ss58_address == coordinator.ss58_address
    signature = sign_receipt_hash(BODY_HASH, delegate)
    assert verify_receipt_signature(BODY_HASH, signature, coordinator.ss58_address)


def test_delegate_keypair_refuses_arbitrary_bytes():
    coordinator = _coordinator_keypair()
    delegate = delegate_keypair_from_worker_request(
        coordinator_hotkey=coordinator.ss58_address,
        mesh_key="m-1",
        request=_manager_backed_request(coordinator),
    )
    with pytest.raises(ValueError):
        delegate.sign(b"arbitrary transaction payload")


def test_delegate_keypair_rejects_wrong_key_signature():
    coordinator = _coordinator_keypair()
    imposter = Keypair.create_from_seed(("22" * 32))
    delegate = delegate_keypair_from_worker_request(
        coordinator_hotkey=coordinator.ss58_address,
        mesh_key="m-1",
        request=_manager_backed_request(imposter),
    )
    with pytest.raises(RuntimeError):
        delegate.sign(coordinator_sign_message("receipt", BODY_HASH))


# ---------------------------------------------------------------------------
# Manager route: wallet stays required, and only the assigned driver may ask.
# ---------------------------------------------------------------------------


def _subnet_manager(tmp_path, monkeypatch, *, driver="w-driver"):
    state_dir, _tok = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:0",
        serving_mode="subnet",
        owner_account=_coordinator_keypair().ss58_address,
        coordinator_address="0x" + "11" * 20,
        validator_shared_state_path=tmp_path / "shared_state.json",
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=1,
    )
    manager = PoolManager(state_dir)
    manager.state["wallet_name"] = "walletname"
    manager.state["wallet_hotkey"] = "hotkeyname"
    manager.state.setdefault("meshes", {})["m-1"] = {"driver": driver}
    workers = manager.state.setdefault("workers", {})
    workers["w-driver"] = {"mesh": "m-1"}
    workers["w-member"] = {"mesh": "m-1"}
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_keypair",
        lambda _w, _h: _coordinator_keypair(),
    )
    return manager


def _as_worker(manager, monkeypatch, worker_id):
    monkeypatch.setattr(
        PoolManager,
        "_auth_worker",
        lambda self, body, **kwargs: (worker_id, ""),
    )
    return manager


def test_route_signs_for_the_assigned_driver(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-driver")
    response = manager.handle_coordinator_sign(
        {"purpose": "receipt", "body_hash": BODY_HASH, "mesh_key": "m-1"}
    )
    coordinator = _coordinator_keypair()
    assert response["coordinator_hotkey"] == coordinator.ss58_address
    assert coordinator.verify(
        coordinator_sign_message("receipt", BODY_HASH),
        bytes.fromhex(response["signature"]),
    )


def test_route_refuses_non_driver_worker(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {"purpose": "receipt", "body_hash": BODY_HASH, "mesh_key": "m-1"}
        )


def test_route_refuses_unknown_mesh_and_purpose(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-driver")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {"purpose": "receipt", "body_hash": BODY_HASH, "mesh_key": "m-404"}
        )
    with pytest.raises(ValueError):
        manager.handle_coordinator_sign(
            {"purpose": "extrinsic", "body_hash": BODY_HASH, "mesh_key": "m-1"}
        )


def test_route_requires_the_manager_wallet(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    manager.state["wallet_name"] = ""
    manager.state["wallet_hotkey"] = ""
    _as_worker(manager, monkeypatch, "w-driver")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {"purpose": "receipt", "body_hash": BODY_HASH, "mesh_key": "m-1"}
        )


def test_route_refuses_dev_pools(tmp_path, monkeypatch):
    state_dir, _tok = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:0",
        serving_mode="dev",
    )
    manager = PoolManager(state_dir)
    _as_worker(manager, monkeypatch, "w-driver")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {"purpose": "receipt", "body_hash": BODY_HASH, "mesh_key": "m-1"}
        )


def test_route_identity_challenge_signs_evm_nonce(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-driver")
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_seed",
        lambda _w, _h, keypair=None: COORDINATOR_SEED,
    )
    nonce = b"\x07" * 32
    response = manager.handle_coordinator_sign(
        {
            "purpose": IDENTITY_CHALLENGE_PURPOSE,
            "nonce": nonce.hex(),
            "mesh_key": "m-1",
        }
    )
    from eth_account import Account
    from eth_account.messages import encode_defunct

    from verallm.chain.wallet import derive_evm_address

    expected_address = derive_evm_address(COORDINATOR_SEED)
    assert response["evm_address"].lower() == expected_address.lower()
    recovered = Account.recover_message(
        encode_defunct(
            primitive=nonce + bytes.fromhex(expected_address[2:])
        ),
        signature=bytes.fromhex(
            response["signature"].removeprefix("0x")
        ),
    )
    assert recovered.lower() == expected_address.lower()


# ---------------------------------------------------------------------------
# Delegation file: the serve subprocess rebuilds the same channel.
# ---------------------------------------------------------------------------


def test_delegation_file_round_trip(tmp_path, monkeypatch):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-driver")
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_seed",
        lambda _w, _h, keypair=None: COORDINATOR_SEED,
    )
    stage_key_file = tmp_path / "stage-proof-key.seed"
    ensure_stage_proof_key_file(stage_key_file)
    coordinator = _coordinator_keypair()
    from verallm.chain.wallet import derive_evm_address

    delegation_file = tmp_path / "coordinator-signing.json"
    delegation_file.write_text(
        json.dumps(
            {
                "manager_endpoint": "http://127.0.0.1:59999",
                "pool_secret": "secret",
                "worker_id": "w-driver",
                "worker_session_id": "a" * 64,
                "mesh_key": "m-1",
                "coordinator_hotkey": coordinator.ss58_address,
                "evm_address": derive_evm_address(COORDINATOR_SEED),
                "stage_proof_key_file": str(stage_key_file),
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_post_json(
        url,
        payload,
        timeout=0,
        keepalive=False,
        keepalive_fallback=True,
    ):
        calls.append(url)
        assert url.endswith("/v1/pool/coordinator-sign")
        # The subprocess signs its request with the stage key like any
        # worker call; auth itself is covered by the pool token tests.
        assert payload.get("worker_auth_signature")
        assert keepalive_fallback is False
        return manager.handle_coordinator_sign(dict(payload))

    monkeypatch.setattr("verallm.mesh.worker.post_json", fake_post_json)

    delegation = coordinator_delegation_from_file(str(delegation_file))
    assert delegation.coordinator_hotkey == coordinator.ss58_address

    signature = sign_receipt_hash(BODY_HASH, delegation.keypair)
    assert verify_receipt_signature(BODY_HASH, signature, coordinator.ss58_address)

    nonce = b"\x09" * 32
    challenge_signature = delegation.challenge_signer(nonce)
    from eth_account import Account
    from eth_account.messages import encode_defunct

    expected_address = derive_evm_address(COORDINATOR_SEED)
    recovered = Account.recover_message(
        encode_defunct(primitive=nonce + bytes.fromhex(expected_address[2:])),
        signature=bytes.fromhex(challenge_signature.removeprefix("0x")),
    )
    assert recovered.lower() == expected_address.lower()
    assert len(calls) >= 2


def test_delegation_file_rebuilds_worker_auth_for_transport_retry(
    tmp_path, monkeypatch
):
    manager = _subnet_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-driver")
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_seed",
        lambda _w, _h, keypair=None: COORDINATOR_SEED,
    )
    stage_key_file = tmp_path / "stage-proof-key.seed"
    ensure_stage_proof_key_file(stage_key_file)
    coordinator = _coordinator_keypair()
    delegation_file = tmp_path / "coordinator-signing.json"
    delegation_file.write_text(
        json.dumps(
            {
                "manager_endpoint": "http://127.0.0.1:59999",
                "pool_secret": "secret",
                "worker_id": "w-driver",
                "worker_session_id": "a" * 64,
                "mesh_key": "m-1",
                "coordinator_hotkey": coordinator.ss58_address,
                "stage_proof_key_file": str(stage_key_file),
            }
        ),
        encoding="utf-8",
    )

    payloads = []

    def fake_post_json(
        url,
        payload,
        timeout=0,
        keepalive=False,
        keepalive_fallback=True,
    ):
        payloads.append(dict(payload))
        assert timeout == 8.0
        assert keepalive is True
        assert keepalive_fallback is False
        if len(payloads) == 1:
            raise RuntimeError(f"HTTP 503 from {url}: retry")
        if len(payloads) == 2:
            raise RuntimeError(f"failed to connect to {url}: timed out")
        return manager.handle_coordinator_sign(dict(payload))

    monkeypatch.setattr("verallm.mesh.worker.post_json", fake_post_json)
    monkeypatch.setattr("verallm.mesh.delegated_signing.time.sleep", lambda _s: None)

    delegation = coordinator_delegation_from_file(str(delegation_file))
    signature = sign_receipt_hash(BODY_HASH, delegation.keypair)
    assert verify_receipt_signature(BODY_HASH, signature, coordinator.ss58_address)
    assert len(payloads) == 3
    assert len({body["worker_auth_nonce"] for body in payloads}) == 3
    assert all(body["purpose"] == "receipt" for body in payloads)
    assert all(body["body_hash"] == BODY_HASH for body in payloads)


# ---------------------------------------------------------------------------
# Capacity-audit artifact purpose: EVM-signed, member-open, tightly bounded.
# ---------------------------------------------------------------------------


def _capacity_manager(tmp_path, monkeypatch, *, model_index=45):
    manager = _subnet_manager(tmp_path, monkeypatch)
    if model_index is not None:
        manager.state["meshes"]["m-1"]["model_index"] = model_index
    monkeypatch.setattr(
        "verallm.mesh.receipt_signing.load_hotkey_seed",
        lambda _w, _h, keypair=None: COORDINATOR_SEED,
    )
    return manager


def _capacity_artifact(**overrides):
    from verallm.chain.wallet import derive_evm_address

    artifact = {
        "protocol_version": "verathos-capacity-audit-v1",
        "artifact_type": "capacity_audit_pass0_receipt",
        "audit_id": "aa" * 32,
        "slot_id": "bb" * 32,
        "address": derive_evm_address(COORDINATOR_SEED).lower(),
        "model_index": 45,
        "worker_id": "w-member",
        "gpu_index": 2,
        "local_gpu_index": 0,
        "pass0_root": "cc" * 32,
    }
    artifact.update(overrides)
    return artifact


def test_capacity_purpose_signs_for_any_mesh_bound_member(tmp_path, monkeypatch):
    # Openings run on EVERY member's GPUs, so unlike all other purposes the
    # authority is mesh-bound membership, not driver appointment.
    manager = _capacity_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    artifact = _capacity_artifact()
    response = manager.handle_coordinator_sign(
        {
            "purpose": "capacity-audit-artifact",
            "artifact": artifact,
            "mesh_key": "m-1",
        }
    )
    from neurons.capacity_audit import recover_artifact_signer

    signed = dict(artifact)
    signed["miner_signature"] = response["signature"]
    assert (
        recover_artifact_signer(signed).lower()
        == response["evm_address"].lower()
        == artifact["address"]
    )


def test_capacity_purpose_refuses_foreign_worker_id(tmp_path, monkeypatch):
    manager = _capacity_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(worker_id="w-driver"),
                "mesh_key": "m-1",
            }
        )


def test_capacity_purpose_refuses_unbound_worker(tmp_path, monkeypatch):
    manager = _capacity_manager(tmp_path, monkeypatch)
    manager.state["workers"]["w-stranger"] = {"mesh": "m-other"}
    _as_worker(manager, monkeypatch, "w-stranger")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(worker_id="w-stranger"),
                "mesh_key": "m-1",
            }
        )


def test_capacity_purpose_requires_chain_bound_mesh(tmp_path, monkeypatch):
    manager = _capacity_manager(tmp_path, monkeypatch, model_index=None)
    _as_worker(manager, monkeypatch, "w-member")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(),
                "mesh_key": "m-1",
            }
        )


def test_capacity_purpose_refuses_slot_mismatch(tmp_path, monkeypatch):
    manager = _capacity_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(model_index=44),
                "mesh_key": "m-1",
            }
        )
    with pytest.raises(PermissionError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(address="0x" + "99" * 20),
                "mesh_key": "m-1",
            }
        )


def test_capacity_purpose_bounds_shape_and_size(tmp_path, monkeypatch):
    manager = _capacity_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    with pytest.raises(ValueError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(artifact_type="receipt"),
                "mesh_key": "m-1",
            }
        )
    with pytest.raises(ValueError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(audit_id=""),
                "mesh_key": "m-1",
            }
        )
    with pytest.raises(ValueError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": _capacity_artifact(padding="z" * (65 * 1024)),
                "mesh_key": "m-1",
            }
        )
    with pytest.raises(ValueError):
        manager.handle_coordinator_sign(
            {
                "purpose": "capacity-audit-artifact",
                "artifact": "not-a-dict",
                "mesh_key": "m-1",
            }
        )


def test_capacity_signer_client_verifies_recovery(tmp_path, monkeypatch):
    from verallm.chain.wallet import derive_evm_address
    from verallm.mesh.delegated_signing import (
        delegate_capacity_signer_from_worker_request,
    )

    manager = _capacity_manager(tmp_path, monkeypatch)
    _as_worker(manager, monkeypatch, "w-member")
    evm_address = derive_evm_address(COORDINATOR_SEED)

    def request(action, fields):
        assert action == "coordinator-sign"
        return manager.handle_coordinator_sign(dict(fields))

    signer = delegate_capacity_signer_from_worker_request(
        evm_address=evm_address,
        mesh_key="m-1",
        request=request,
    )
    signed = signer.sign_artifact(_capacity_artifact())
    from neurons.capacity_audit import verify_artifact_signature

    assert verify_artifact_signature(signed, evm_address)

    # A manager answering with the WRONG key must fail locally, before any
    # artifact could reach a validator.
    def bad_request(action, fields):
        response = manager.handle_coordinator_sign(dict(fields))
        response["signature"] = "ab" * 65
        return response

    bad_signer = delegate_capacity_signer_from_worker_request(
        evm_address=evm_address,
        mesh_key="m-1",
        request=bad_request,
    )
    with pytest.raises(Exception):
        bad_signer.sign_artifact(_capacity_artifact())


def test_max_capacity_artifact_bytes_is_type_aware():
    """Receipts stay under the tight bound; sampled proof payloads measure
    ~254KiB for a single-GPU combined workload, and the flat 64KiB bound
    refused every one of them — each timing pass became
    hard_proof_miss/missing_payload ."""

    from verallm.mesh.delegated_signing import (
        MAX_CAPACITY_ARTIFACT_BYTES,
        MAX_CAPACITY_PROOF_ARTIFACT_BYTES,
        max_capacity_artifact_bytes,
    )

    assert (
        max_capacity_artifact_bytes("capacity_audit_pass0_receipt")
        == MAX_CAPACITY_ARTIFACT_BYTES
    )
    assert (
        max_capacity_artifact_bytes("capacity_audit_final_receipt")
        == MAX_CAPACITY_ARTIFACT_BYTES
    )
    proof_bound = max_capacity_artifact_bytes("capacity_audit_proof_payload")
    assert proof_bound == MAX_CAPACITY_PROOF_ARTIFACT_BYTES
    # A real single-GPU payload (~254KiB) must fit with generous headroom,
    # while staying far below the validator ingest's 32MiB cap.
    assert 260 * 1024 < proof_bound <= 32 * 1024 * 1024
