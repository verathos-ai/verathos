"""Validator-side trust-boundary tests for mesh verification snapshots."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons.mesh_snapshot import (
    MESH_VERIFICATION_SNAPSHOT_PATH,
    MeshSnapshotExpectations,
    MeshSnapshotTrustAnchors,
    discover_and_pin_mesh_verification_snapshot,
    fetch_and_pin_mesh_verification_snapshot,
)
from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    verify_request,
)
from neurons.validator_db import ValidatorStateDB
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    MeshVerificationStage,
    sign_mesh_verification_snapshot,
)


COORDINATOR_SEED = bytes.fromhex("11" * 32)
OTHER_COORDINATOR_SEED = bytes.fromhex("33" * 32)
VALIDATOR_SEED = bytes.fromhex("22" * 32)
COORDINATOR_KEYPAIR = Keypair.create_from_seed(COORDINATOR_SEED.hex())
OTHER_COORDINATOR_KEYPAIR = Keypair.create_from_seed(OTHER_COORDINATOR_SEED.hex())
VALIDATOR_KEYPAIR = Keypair.create_from_seed(VALIDATOR_SEED.hex())
STAGE_KEYPAIRS = (
    Keypair.create_from_seed(("44" * 32)),
    Keypair.create_from_seed(("55" * 32)),
)


@pytest.fixture
def state_db(tmp_path):
    db = ValidatorStateDB(db_path=str(tmp_path / "validator.db"))
    yield db
    db.close()


def _coordinator(**changes) -> MeshCoordinatorIdentity:
    identity = MeshCoordinatorIdentity(
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        coordinator_hotkey=COORDINATOR_KEYPAIR.ss58_address,
        coordinator_evm_address="0x" + "ab" * 20,
        model_index=0,
    )
    return replace(identity, **changes)


def _model(**changes) -> MeshModelAnchors:
    model = MeshModelAnchors(
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="aa" * 32,
        model_tensor_manifest_root="bb" * 32,
        tokenizer_hash="cc" * 32,
        total_layers=2,
        max_context_len=32_768,
        quantization_scheme="gguf_q4_k_m",
        activation_dtype="f16",
    )
    return replace(model, **changes)


def _policy(**changes) -> MeshVerificationPolicy:
    policy = MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=3,
        deferred_proof_enabled=False,
    )
    return replace(policy, **changes)


def _unsigned_snapshot(
    *,
    mesh_id: str = "mesh-test",
    generation: int = 1,
    epoch: int = 12,
    issued_at_unix: int = 900,
    expires_at_unix: int = 2_000,
    coordinator: MeshCoordinatorIdentity | None = None,
    model: MeshModelAnchors | None = None,
    policy: MeshVerificationPolicy | None = None,
) -> MeshVerificationSnapshot:
    return MeshVerificationSnapshot(
        mesh_id=mesh_id,
        generation=generation,
        epoch=epoch,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        coordinator=coordinator or _coordinator(),
        model=model or _model(),
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


def _signed_snapshot(**changes) -> MeshVerificationSnapshot:
    keypair = changes.pop("keypair", COORDINATOR_KEYPAIR)
    return sign_mesh_verification_snapshot(_unsigned_snapshot(**changes), keypair)


def _expectations(
    *,
    mesh_id: str = "mesh-test",
    generation: int = 1,
    epoch: int = 12,
    coordinator: MeshCoordinatorIdentity | None = None,
    model: MeshModelAnchors | None = None,
    policy: MeshVerificationPolicy | None = None,
) -> MeshSnapshotExpectations:
    return MeshSnapshotExpectations(
        mesh_id=mesh_id,
        generation=generation,
        epoch=epoch,
        coordinator=coordinator or _coordinator(),
        model=model or _model(),
        policy=policy or _policy(),
    )


def _raw(snapshot: MeshVerificationSnapshot) -> bytes:
    return json.dumps(snapshot.to_dict(), sort_keys=True).encode("utf-8")


def _trust_anchors(*, epoch: int = 12) -> MeshSnapshotTrustAnchors:
    return MeshSnapshotTrustAnchors(
        epoch=epoch,
        coordinator=_coordinator(),
        model=_model(),
        policy=_policy(),
    )


def _fetching(raw: bytes, captured: dict | None = None):
    def fetch(url, headers, timeout):
        if captured is not None:
            captured.update(url=url, headers=dict(headers), timeout=timeout)
        return raw

    return fetch


def _fetch_and_pin(
    state_db: ValidatorStateDB,
    snapshot: MeshVerificationSnapshot,
    *,
    expectations: MeshSnapshotExpectations | None = None,
    clock_value: float = 1_000.25,
    captured: dict | None = None,
) -> MeshVerificationSnapshot:
    return fetch_and_pin_mesh_verification_snapshot(
        coordinator_endpoint="http://coordinator.example:9000/",
        validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
        validator_hotkey_seed=VALIDATOR_SEED,
        expectations=expectations or _expectations(),
        state_db=state_db,
        timeout=7.5,
        fetch=_fetching(_raw(snapshot), captured),
        clock=lambda: clock_value,
    )


def test_happy_path_signs_exact_get_and_pins_snapshot(state_db):
    snapshot = _signed_snapshot()
    captured: dict = {}

    pinned = _fetch_and_pin(state_db, snapshot, captured=captured)

    assert pinned == snapshot
    assert captured["url"] == (
        "http://coordinator.example:9000" + MESH_VERIFICATION_SNAPSHOT_PATH
    )
    assert captured["timeout"] == 7.5
    headers = captured["headers"]
    assert set(headers) == {HDR_HOTKEY, HDR_SIGNATURE, HDR_TIMESTAMP}
    assert headers[HDR_HOTKEY] == VALIDATOR_KEYPAIR.ss58_address
    assert verify_request(
        method="GET",
        path=MESH_VERIFICATION_SNAPSHOT_PATH,
        body=b"",
        hotkey_ss58=headers[HDR_HOTKEY],
        signature_hex=headers[HDR_SIGNATURE],
        timestamp_str=headers[HDR_TIMESTAMP],
    ) == (True, "ok")
    assert verify_request(
        method="GET",
        path=MESH_VERIFICATION_SNAPSHOT_PATH + "/wrong",
        body=b"",
        hotkey_ss58=headers[HDR_HOTKEY],
        signature_hex=headers[HDR_SIGNATURE],
        timestamp_str=headers[HDR_TIMESTAMP],
    )[0] is False
    assert verify_request(
        method="GET",
        path=MESH_VERIFICATION_SNAPSHOT_PATH,
        body=b"{}",
        hotkey_ss58=headers[HDR_HOTKEY],
        signature_hex=headers[HDR_SIGNATURE],
        timestamp_str=headers[HDR_TIMESTAMP],
    )[0] is False

    row = state_db.get_latest_mesh_verification_snapshot_for_epoch(
        coordinator_address=_coordinator().coordinator_evm_address,
        model_index=0,
        epoch=12,
    )
    assert row is not None
    assert row["generation"] == 1
    assert row["snapshot_hash"] == snapshot.snapshot_hash_hex()


def test_discovery_accepts_coordinator_authored_mesh_id_and_generation(state_db):
    snapshot = _signed_snapshot(mesh_id="mesh-private-a", generation=7)

    pinned = discover_and_pin_mesh_verification_snapshot(
        coordinator_endpoint="http://coordinator.example:9000",
        validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
        validator_hotkey_seed=VALIDATOR_SEED,
        trust_anchors=_trust_anchors(),
        state_db=state_db,
        fetch=_fetching(_raw(snapshot)),
        clock=lambda: 1_000,
    )

    assert pinned.mesh_id == "mesh-private-a"
    assert pinned.generation == 7


def test_discovery_rejects_mesh_id_change_within_one_epoch(state_db):
    first = _signed_snapshot(mesh_id="mesh-private-a", generation=1)
    second = _signed_snapshot(mesh_id="mesh-private-b", generation=2)
    for snapshot in (first,):
        discover_and_pin_mesh_verification_snapshot(
            coordinator_endpoint="http://coordinator.example:9000",
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            trust_anchors=_trust_anchors(),
            state_db=state_db,
            fetch=_fetching(_raw(snapshot)),
            clock=lambda: 1_000,
        )

    with pytest.raises(ValueError, match="mesh_id changed within epoch"):
        discover_and_pin_mesh_verification_snapshot(
            coordinator_endpoint="http://coordinator.example:9000",
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            trust_anchors=_trust_anchors(),
            state_db=state_db,
            fetch=_fetching(_raw(second)),
            clock=lambda: 1_001,
        )


def test_validator_initiated_repin_adopts_new_mesh_lineage(state_db):
    """Bootstrap/refusal re-pins pass allow_mesh_change: a relaunched mesh
    (new mesh_id, generation restarting below the dead lineage's) supersedes
    the dead lineage instead of being excluded for the rest of the epoch."""
    first = _signed_snapshot(mesh_id="mesh-private-a", generation=5)
    relaunched = _signed_snapshot(mesh_id="mesh-private-b", generation=1)
    discover_and_pin_mesh_verification_snapshot(
        coordinator_endpoint="http://coordinator.example:9000",
        validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
        validator_hotkey_seed=VALIDATOR_SEED,
        trust_anchors=_trust_anchors(),
        state_db=state_db,
        fetch=_fetching(_raw(first)),
        clock=lambda: 1_000,
    )

    adopted = discover_and_pin_mesh_verification_snapshot(
        coordinator_endpoint="http://coordinator.example:9000",
        validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
        validator_hotkey_seed=VALIDATOR_SEED,
        trust_anchors=_trust_anchors(),
        state_db=state_db,
        fetch=_fetching(_raw(relaunched)),
        clock=lambda: 1_001,
        allow_mesh_change=True,
    )
    assert adopted.mesh_id == "mesh-private-b"

    anchors = _trust_anchors()
    latest = state_db.get_latest_mesh_verification_snapshot_for_epoch(
        coordinator_address=anchors.coordinator.coordinator_evm_address,
        model_index=anchors.coordinator.model_index,
        epoch=anchors.epoch,
    )
    payload = json.loads(str(latest["snapshot_json"]))
    assert payload["mesh_id"] == "mesh-private-b"
    assert int(latest["generation"]) == 1

    # The unsolicited path still rejects a further swap.
    third = _signed_snapshot(mesh_id="mesh-private-c", generation=2)
    with pytest.raises(ValueError, match="mesh_id changed within epoch"):
        discover_and_pin_mesh_verification_snapshot(
            coordinator_endpoint="http://coordinator.example:9000",
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            trust_anchors=_trust_anchors(),
            state_db=state_db,
            fetch=_fetching(_raw(third)),
            clock=lambda: 1_002,
        )


def test_unsigned_snapshot_is_rejected_before_cache(state_db):
    with pytest.raises(ValueError, match="signature is required"):
        _fetch_and_pin(state_db, _unsigned_snapshot())

    assert state_db.get_mesh_verification_snapshot_history(
        coordinator_address=_coordinator().coordinator_evm_address,
        model_index=0,
        epoch=12,
    ) == []


def test_tampered_signed_snapshot_is_rejected(state_db):
    signed = _signed_snapshot()
    payload = signed.to_dict()
    payload["expires_at_unix"] = 1_999

    with pytest.raises(ValueError, match="signature is invalid"):
        fetch_and_pin_mesh_verification_snapshot(
            coordinator_endpoint="http://coordinator.example:9000",
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            expectations=_expectations(),
            state_db=state_db,
            fetch=_fetching(json.dumps(payload).encode("utf-8")),
            clock=lambda: 1_000,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("chain_id", 946),
        ("netuid", 406),
        ("coordinator_uid", 2),
        ("coordinator_evm_address", "0x" + "cd" * 20),
        ("model_index", 1),
    ],
)
def test_every_non_hotkey_coordinator_identity_field_is_bound(
    state_db, field, wrong_value
):
    wrong = _coordinator(**{field: wrong_value})
    snapshot = _signed_snapshot(coordinator=wrong)

    with pytest.raises(ValueError, match="coordinator does not match"):
        _fetch_and_pin(state_db, snapshot)


def test_coordinator_hotkey_is_bound_to_chain_approved_identity(state_db):
    wrong = _coordinator(coordinator_hotkey=OTHER_COORDINATOR_KEYPAIR.ss58_address)
    snapshot = _signed_snapshot(
        coordinator=wrong,
        keypair=OTHER_COORDINATOR_KEYPAIR,
    )

    with pytest.raises(ValueError, match="coordinator does not match"):
        _fetch_and_pin(state_db, snapshot)


@pytest.mark.parametrize(
    ("snapshot_changes", "message"),
    [
        ({"mesh_id": "mesh-other"}, "mesh_id does not match"),
        ({"generation": 2}, "generation does not match"),
        ({"epoch": 13}, "epoch does not match"),
    ],
)
def test_mesh_generation_and_epoch_are_validator_owned(
    state_db, snapshot_changes, message
):
    snapshot = _signed_snapshot(**snapshot_changes)

    with pytest.raises(ValueError, match=message):
        _fetch_and_pin(state_db, snapshot)


def test_model_anchors_are_bound_exactly(state_db):
    wrong_model = _model(model_tensor_manifest_root="dd" * 32)
    snapshot = _signed_snapshot(model=wrong_model)

    with pytest.raises(ValueError, match="model does not match"):
        _fetch_and_pin(state_db, snapshot)


def test_policy_is_validator_owned_and_bound_exactly(state_db):
    # Both decode rates move together; they are required to be equal.
    wrong_policy = _policy(
        organic_decode_sample_bps=999, canary_decode_sample_bps=999,
    )
    snapshot = _signed_snapshot(policy=wrong_policy)

    with pytest.raises(ValueError, match="policy does not match"):
        _fetch_and_pin(state_db, snapshot)


def test_expired_snapshot_is_rejected(state_db):
    snapshot = _signed_snapshot(expires_at_unix=1_000)

    with pytest.raises(ValueError, match="expired"):
        _fetch_and_pin(state_db, snapshot, clock_value=1_000)


def test_endpoint_bearing_payload_is_rejected_before_signature_check(state_db):
    payload = _signed_snapshot().to_dict()
    payload["stages"][0]["worker_endpoint"] = "http://private-worker:9001"

    with pytest.raises(ValueError, match="forbidden network-location field"):
        fetch_and_pin_mesh_verification_snapshot(
            coordinator_endpoint="http://coordinator.example:9000",
            validator_hotkey_ss58=VALIDATOR_KEYPAIR.ss58_address,
            validator_hotkey_seed=VALIDATOR_SEED,
            expectations=_expectations(),
            state_db=state_db,
            fetch=_fetching(json.dumps(payload).encode("utf-8")),
            clock=lambda: 1_000,
        )


def test_db_generation_rollback_propagates_to_caller(state_db):
    generation_two = _signed_snapshot(generation=2)
    _fetch_and_pin(
        state_db,
        generation_two,
        expectations=_expectations(generation=2),
    )

    with pytest.raises(ValueError, match="generation rollback"):
        _fetch_and_pin(state_db, _signed_snapshot(generation=1))


def test_db_same_generation_equivocation_propagates_to_caller(state_db):
    original = _signed_snapshot()
    _fetch_and_pin(state_db, original)
    conflicting = _signed_snapshot(issued_at_unix=901)

    with pytest.raises(ValueError, match="conflicting.*hash"):
        _fetch_and_pin(state_db, conflicting)


def test_transient_fetch_error_classification():
    """Load/transport-shaped failures are retryable; every definitive
    verdict about the response is not. A busy coordinator stalls the
    snapshot route while /health stays sub-second , so timeouts must be
    distinguishable from signature/binding failures."""

    import http.client
    import socket
    import urllib.error

    from neurons.mesh_snapshot import is_transient_snapshot_fetch_error

    transient = [
        TimeoutError("timed out"),
        socket.timeout("timed out"),
        ConnectionRefusedError("refused"),
        ConnectionResetError("reset"),
        OSError("network unreachable"),
        urllib.error.URLError("timed out"),
        http.client.RemoteDisconnected("closed"),
        http.client.BadStatusLine("garbage"),
        urllib.error.HTTPError("http://x", 503, "busy", None, None),
        urllib.error.HTTPError("http://x", 429, "slow down", None, None),
    ]
    for exc in transient:
        assert is_transient_snapshot_fetch_error(exc), exc

    definitive = [
        ValueError("mesh verification snapshot signature is invalid"),
        ValueError("snapshot epoch does not match expected epoch"),
        urllib.error.HTTPError("http://x", 401, "unauthorized", None, None),
        urllib.error.HTTPError("http://x", 403, "forbidden", None, None),
        urllib.error.HTTPError("http://x", 404, "not found", None, None),
        RuntimeError("unrelated"),
    ]
    for exc in definitive:
        assert not is_transient_snapshot_fetch_error(exc), exc
