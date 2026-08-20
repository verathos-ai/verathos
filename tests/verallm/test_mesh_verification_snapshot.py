"""Endpoint-free signed verification snapshots for private GGUF meshes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from bittensor_wallet import Keypair
from scalecodec.utils.ss58 import ss58_encode

from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    MeshVerificationStage,
    assert_endpoint_free_payload,
    derive_opaque_stage_id,
    rotate_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
    snapshot_signature_message,
    verify_mesh_verification_snapshot_signature,
)


def _keypair(uri: str = "//VerathosMeshSnapshotTest") -> Keypair:
    return Keypair.create_from_uri(uri)


def _stage(
    index: int,
    start: int,
    end: int,
    *,
    mesh_id: str = "mesh_qwen25_7b_q4km",
    identity_byte: str | None = None,
) -> MeshVerificationStage:
    commitment = (identity_byte or f"{index + 1:x}") * 64
    proof_key = _keypair(
        f"//VerathosMeshSnapshotStage{index}_{start}_{end}_{identity_byte or 'default'}"
    ).ss58_address
    return MeshVerificationStage(
        stage_id=derive_opaque_stage_id(
            mesh_id=mesh_id,
            stage_index=index,
            layer_start=start,
            layer_end=end,
            stage_identity_commitment=commitment,
        ),
        layer_start=start,
        layer_end=end,
        proof_key_scheme="sr25519",
        proof_key=proof_key,
        proof_commitment=commitment,
    )


def _snapshot(*, reverse_stages: bool = False) -> MeshVerificationSnapshot:
    keypair = _keypair()
    stages = (_stage(0, 0, 16), _stage(1, 16, 32))
    if reverse_stages:
        stages = tuple(reversed(stages))
    return MeshVerificationSnapshot(
        mesh_id="mesh_qwen25_7b_q4km",
        generation=7,
        epoch=123,
        issued_at_unix=2_000_000_000,
        expires_at_unix=2_000_000_600,
        coordinator=MeshCoordinatorIdentity(
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            coordinator_hotkey=keypair.ss58_address,
            coordinator_evm_address="0x" + "ab" * 20,
            model_index=26,
        ),
        model=MeshModelAnchors(
            model_id="qwen2.5-7b-q4-k-m",
            model_package_hash="a1" * 32,
            model_tensor_manifest_root="b2" * 32,
            tokenizer_hash="c3" * 32,
            total_layers=32,
            max_context_len=32_768,
            quantization_scheme="gguf_q4_k_m",
            activation_dtype="f16",
        ),
        policy=MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format="compact-raw-v3",
            base_proof_sample_bps=10_000,
            organic_decode_sample_bps=10_000,
            canary_decode_sample_bps=10_000,
            proof_ops_per_request=3,
            deferred_proof_enabled=False,
        ),
        expected_stage_count=2,
        stages=stages,
    )


def test_snapshot_serialization_is_endpoint_free_and_closed() -> None:
    payload = _snapshot().to_dict()
    assert_endpoint_free_payload(payload)
    assert payload["model"]["max_context_len"] == 32_768

    missing_max_context = deepcopy(payload)
    del missing_max_context["model"]["max_context_len"]
    with pytest.raises(
        ValueError,
        match="model missing fields: max_context_len",
    ):
        MeshVerificationSnapshot.from_dict(missing_max_context)

    with_top_level_endpoint = deepcopy(payload)
    with_top_level_endpoint["endpoint"] = "https://worker.example:9000"
    with pytest.raises(ValueError, match="forbidden network-location field"):
        MeshVerificationSnapshot.from_dict(with_top_level_endpoint)

    with_nested_url = deepcopy(payload)
    with_nested_url["policy"]["audit"] = {
        "transport": {"url": "https://worker.internal:9443/proof"}
    }
    with pytest.raises(ValueError, match="forbidden network-location field"):
        MeshVerificationSnapshot.from_dict(with_nested_url)

    with_unknown_benign_field = deepcopy(payload)
    with_unknown_benign_field["model"]["notes"] = "opaque"
    with pytest.raises(ValueError, match="model contains unknown fields: notes"):
        MeshVerificationSnapshot.from_dict(with_unknown_benign_field)

    with_url_in_known_field = deepcopy(payload)
    with_url_in_known_field["stages"][0]["proof_key"] = "https://10.0.0.2:9001"
    with pytest.raises(ValueError, match="must not contain a network location"):
        MeshVerificationSnapshot.from_dict(with_url_in_known_field)


def test_endpoint_guard_rejects_nested_location_key_and_value() -> None:
    with pytest.raises(ValueError, match=r"snapshot\.outer\.worker_endpoint"):
        assert_endpoint_free_payload(
            {"outer": {"worker_endpoint": "opaque", "safe": "value"}}
        )
    with pytest.raises(ValueError, match=r"snapshot\.outer\.transport"):
        assert_endpoint_free_payload(
            {"outer": {"transport": "http://127.0.0.1:8080/private"}}
        )


def test_deterministic_stage_id_uses_only_committed_public_facts() -> None:
    kwargs = {
        "mesh_id": "mesh_qwen25_7b_q4km",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 16,
        "stage_identity_commitment": "11" * 32,
    }
    stage_id = derive_opaque_stage_id(**kwargs)
    assert stage_id == derive_opaque_stage_id(**kwargs)
    assert stage_id.startswith("stg_") and len(stage_id) == 36
    assert stage_id != derive_opaque_stage_id(**{**kwargs, "stage_index": 1})
    assert stage_id != derive_opaque_stage_id(
        **{**kwargs, "stage_identity_commitment": "22" * 32}
    )

    with pytest.raises(ValueError, match="32-byte hex digest"):
        derive_opaque_stage_id(
            **{**kwargs, "stage_identity_commitment": "https://worker:9000"}
        )
    with pytest.raises(ValueError, match="non-empty half-open range"):
        derive_opaque_stage_id(**{**kwargs, "layer_end": 0})


@pytest.mark.parametrize(
    "stages, expected_count, message",
    [
        ((_stage(0, 1, 16), _stage(1, 16, 32)), 2, "exactly once"),
        ((_stage(0, 0, 15), _stage(1, 16, 32)), 2, "exactly once"),
        ((_stage(0, 0, 17), _stage(1, 16, 32)), 2, "exactly once"),
        ((_stage(0, 0, 16), _stage(1, 16, 31)), 2, "exactly once"),
        ((_stage(0, 0, 16), _stage(1, 16, 33)), 2, "exceeds total_layers"),
        ((_stage(0, 0, 16), _stage(1, 16, 32)), 1, "does not match"),
        (
            (
                _stage(0, 0, 16),
                replace(_stage(1, 16, 32), stage_id=_stage(0, 0, 16).stage_id),
            ),
            2,
            "must be unique",
        ),
    ],
)
def test_snapshot_requires_exact_stage_tiling(
    stages: tuple[MeshVerificationStage, ...],
    expected_count: int,
    message: str,
) -> None:
    snapshot = replace(
        _snapshot(),
        stages=stages,
        expected_stage_count=expected_count,
    )
    with pytest.raises(ValueError, match=message):
        snapshot.validate()


def test_stage_order_is_canonical_but_stage_collection_is_immutable() -> None:
    ordered = _snapshot()
    reversed_snapshot = _snapshot(reverse_stages=True)
    reversed_snapshot.validate()
    assert reversed_snapshot.expected_stage_ids == ordered.expected_stage_ids
    assert reversed_snapshot.to_dict()["stages"] == ordered.to_dict()["stages"]

    mutable = replace(ordered, stages=list(ordered.stages))
    with pytest.raises(ValueError, match="immutable tuple"):
        mutable.validate()


def test_stage_keys_reject_alternate_ss58_prefix_aliases() -> None:
    snapshot = _snapshot()
    first_key = _keypair(
        "//VerathosMeshSnapshotStage0_0_16_default"
    ).public_key
    alternate_prefix = ss58_encode(bytes(first_key), ss58_format=0)
    aliased = replace(
        snapshot,
        stages=(
            snapshot.stages[0],
            replace(snapshot.stages[1], proof_key=alternate_prefix),
        ),
    )
    with pytest.raises(ValueError, match="Bittensor SS58 format 42"):
        aliased.validate()


def test_stage_key_must_be_cryptographically_distinct_from_coordinator() -> None:
    snapshot = _snapshot()
    stage_public_key = bytes(
        _keypair("//VerathosMeshSnapshotStage0_0_16_default").public_key
    )
    coordinator_alias = ss58_encode(stage_public_key, ss58_format=0)
    aliased = replace(
        snapshot,
        coordinator=replace(
            snapshot.coordinator,
            coordinator_hotkey=coordinator_alias,
        ),
    )
    with pytest.raises(ValueError, match="distinct from the coordinator"):
        aliased.validate()


def test_canonical_hashes_are_order_independent_and_have_golden_values() -> None:
    snapshot = _snapshot()
    reversed_snapshot = _snapshot(reverse_stages=True)

    assert snapshot.body_hash_hex() == reversed_snapshot.body_hash_hex()
    assert snapshot.snapshot_hash_hex() == reversed_snapshot.snapshot_hash_hex()
    assert snapshot.stage_coverage_hash_hex() == reversed_snapshot.stage_coverage_hash_hex()
    assert MeshVerificationSnapshot.from_dict(snapshot.to_dict()) == snapshot

    # The policy includes the signed postcommit_hard_audit_bps tier-draw
    # rate, so changing it intentionally changes these wire-format goldens.
    assert snapshot.body_hash_hex() == (
        "edebc20aff822ff09356be8f33ac27863a15e52a1846fcf67f21e09257c92b3e"
    )
    assert snapshot.snapshot_hash_hex() == (
        "ed4be0e892926697c2344cf35d597b3af38feac5587c581fa0df02b38792a511"
    )
    assert snapshot.stage_coverage_hash_hex() == (
        "5c87718389e8296aa7f91e343dec4e8671f396fd255991590f2a8ec24d68bebd"
    )
    assert len(
        {
            snapshot.body_hash_hex(),
            snapshot.snapshot_hash_hex(),
            snapshot.stage_coverage_hash_hex(),
        }
    ) == 3


def test_sign_verify_tamper_wrong_hotkey_and_expiry() -> None:
    keypair = _keypair()
    snapshot = _snapshot()
    signed = sign_mesh_verification_snapshot(snapshot, keypair)

    assert signed.signature
    assert signed.body_hash_hex() == snapshot.body_hash_hex()
    assert signed.snapshot_hash_hex() != snapshot.snapshot_hash_hex()
    assert verify_mesh_verification_snapshot_signature(
        signed,
        expected_hotkey=keypair.ss58_address,
        expected_epoch=snapshot.epoch,
        expected_mesh_id=snapshot.mesh_id,
        expected_generation=snapshot.generation,
        expected_coordinator=snapshot.coordinator,
        expected_model=snapshot.model,
        expected_policy=snapshot.policy,
        now_unix=snapshot.expires_at_unix - 1,
    )
    assert not verify_mesh_verification_snapshot_signature(
        signed,
        now_unix=snapshot.expires_at_unix,
    )

    tampered = replace(signed, generation=signed.generation + 1)
    assert not verify_mesh_verification_snapshot_signature(tampered)

    other = _keypair("//VerathosMeshSnapshotOther")
    assert not verify_mesh_verification_snapshot_signature(
        signed,
        expected_hotkey=other.ss58_address,
    )
    with pytest.raises(ValueError, match="does not match coordinator_hotkey"):
        sign_mesh_verification_snapshot(snapshot, other)

    wrong_signature = other.sign(snapshot_signature_message(snapshot)).hex()
    assert not verify_mesh_verification_snapshot_signature(
        replace(snapshot, signature=wrong_signature)
    )


def test_verifier_rejects_wrong_chain_model_and_policy_bindings() -> None:
    snapshot = _snapshot()
    signed = sign_mesh_verification_snapshot(snapshot, _keypair())

    wrong_coordinator = replace(snapshot.coordinator, model_index=27)
    wrong_model = replace(snapshot.model, max_context_len=65_536)
    wrong_policy = replace(snapshot.policy, organic_decode_sample_bps=999)

    assert not verify_mesh_verification_snapshot_signature(
        signed, expected_coordinator=wrong_coordinator
    )
    assert not verify_mesh_verification_snapshot_signature(signed, expected_model=wrong_model)
    assert not verify_mesh_verification_snapshot_signature(
        signed, expected_policy=wrong_policy
    )
    assert not verify_mesh_verification_snapshot_signature(
        signed, expected_mesh_id="another_mesh"
    )
    assert not verify_mesh_verification_snapshot_signature(
        signed, expected_generation=snapshot.generation + 1
    )


@pytest.mark.parametrize(
    "component, replacement, message",
    [
        ("model", {"model_package_hash": "AA" * 32}, "lowercase 32-byte"),
        ("model", {"model_id": "https://worker.example/model"}, "registry model"),
        ("model", {"total_layers": True}, "must be an integer"),
        ("model", {"max_context_len": True}, "must be an integer"),
        ("model", {"max_context_len": 0}, r"\[1, 4294967295\]"),
        ("model", {"max_context_len": 2**32}, r"\[1, 4294967295\]"),
        ("policy", {"base_proof_sample_bps": 10_001}, r"\[0, 10000\]"),
        ("policy", {"proof_ops_per_request": 0}, r"\[1, 4096\]"),
        (
            "policy",
            {"proof_trace_candidates_per_request": 1023},
            r"\[1024, 4096\]",
        ),
        ("policy", {"deferred_proof_enabled": 0}, "must be boolean"),
        (
            "policy",
            {"trace_manifest_format": "invented-v9"},
            "unsupported policy.trace_manifest_format",
        ),
        ("coordinator", {"chain_id": 0}, r"\[1, 2\^64\)"),
        ("coordinator", {"coordinator_uid": True}, "must be an integer"),
        (
            "coordinator",
            {"coordinator_evm_address": "0x" + "AB" * 20},
            "lowercase 20-byte",
        ),
    ],
)
def test_model_policy_and_coordinator_validation(
    component: str,
    replacement: dict[str, object],
    message: str,
) -> None:
    snapshot = _snapshot()
    changed_component = replace(getattr(snapshot, component), **replacement)
    invalid = replace(snapshot, **{component: changed_component})
    with pytest.raises(ValueError, match=message):
        invalid.validate()


@pytest.mark.parametrize(
    "path, invalid_value",
    [
        (("epoch",), "123"),
        (("model", "total_layers"), 32.0),
        (("model", "max_context_len"), 32_768.0),
        (("policy", "base_proof_sample_bps"), True),
        (("coordinator", "model_index"), "26"),
        (("stages", 0, "layers", "start"), False),
    ],
)
def test_deserialization_rejects_coercible_noncanonical_types(
    path: tuple[str | int, ...], invalid_value: object
) -> None:
    payload = deepcopy(_snapshot().to_dict())
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = invalid_value
    with pytest.raises(ValueError, match="must be an integer"):
        MeshVerificationSnapshot.from_dict(payload)


class TestDecodeRateCarriesNoCanarySignal:
    """The decode-audit rate is cleartext at phase one, so it must be constant.

    A coordinator that can read "this is a canary" off the request serves
    canaries honestly and cheats everything else, and the decode gate is the
    only live semantic guard on the organic path.
    """

    def test_unequal_decode_rates_are_rejected(self):
        from verallm.mesh.verification_snapshot import MeshVerificationPolicy

        policy = MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format="compact-raw-v3",
            base_proof_sample_bps=10_000,
            organic_decode_sample_bps=1_000,
            canary_decode_sample_bps=10_000,
            proof_ops_per_request=1,
            proof_trace_candidates_per_request=1024,
            deferred_proof_enabled=False,
        )

        with pytest.raises(ValueError, match="must be equal"):
            policy.validate()
        with pytest.raises(ValueError, match="decode rates disagree"):
            policy.decode_sample_bps

    def test_equal_decode_rates_expose_one_value(self):
        from verallm.mesh.verification_snapshot import MeshVerificationPolicy

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

        policy.validate()
        assert policy.decode_sample_bps == 10_000


def test_rotate_snapshot_rebinds_epoch_and_resigns() -> None:
    """Epoch rotation: same content, new epoch/generation/freshness, valid
    signature. Validators pin one snapshot per scoring epoch, so a
    long-lived mesh re-signs at every boundary without relaunching."""
    keypair = _keypair()
    signed = sign_mesh_verification_snapshot(_snapshot(), keypair)
    rotated = rotate_mesh_verification_snapshot(
        signed, epoch=124, keypair=keypair, now_unix=2_000_000_700
    )
    assert rotated.epoch == 124
    assert rotated.generation == signed.generation + 1
    assert rotated.issued_at_unix == 2_000_000_700
    # The freshness window length is preserved (600s in the fixture).
    assert rotated.expires_at_unix == 2_000_000_700 + 600
    before = signed.to_dict(include_signature=False)
    after = rotated.to_dict(include_signature=False)
    for moving in ("epoch", "generation", "issued_at_unix", "expires_at_unix"):
        before.pop(moving)
        after.pop(moving)
    assert before == after
    assert verify_mesh_verification_snapshot_signature(
        rotated,
        expected_hotkey=keypair.ss58_address,
        expected_epoch=124,
        now_unix=2_000_000_800,
    )
    # Redelivered command: rotating to the epoch it already binds is a
    # no-op and must not burn a generation.
    again = rotate_mesh_verification_snapshot(
        rotated, epoch=124, keypair=keypair
    )
    assert again is rotated


def test_rotate_snapshot_recovers_an_expired_snapshot() -> None:
    """Rotating an EXPIRED snapshot is the recovery case (e.g. the manager
    was down across several epochs) and must succeed."""
    keypair = _keypair()
    signed = sign_mesh_verification_snapshot(_snapshot(), keypair)
    long_after_expiry = 2_100_000_000
    rotated = rotate_mesh_verification_snapshot(
        signed, epoch=999, keypair=keypair, now_unix=long_after_expiry
    )
    assert verify_mesh_verification_snapshot_signature(
        rotated,
        expected_hotkey=keypair.ss58_address,
        expected_epoch=999,
        now_unix=long_after_expiry + 10,
    )


def test_rotate_snapshot_rejects_a_foreign_keypair() -> None:
    signed = sign_mesh_verification_snapshot(_snapshot(), _keypair())
    with pytest.raises(ValueError):
        rotate_mesh_verification_snapshot(
            signed, epoch=999, keypair=_keypair("//SomeoneElse")
        )
