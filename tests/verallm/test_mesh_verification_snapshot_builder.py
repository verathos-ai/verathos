"""Focused tests for building public snapshots from private runtime meshes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from bittensor_wallet import Keypair

from verallm.mesh.types import MeshMember, MeshSpec, StageRange
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    derive_opaque_stage_id,
)


def _runtime_spec() -> MeshSpec:
    hotkey = Keypair.create_from_uri("//MeshSnapshotBuilder").ss58_address
    spec = MeshSpec(
        mesh_id="mesh_qwen25_7b_q4km",
        mode="private",
        coordinator_uid=7,
        coordinator_hotkey=hotkey,
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
                hotkey=hotkey,
                endpoint="http://127.0.0.1:9100",
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
            ),
            MeshMember(
                uid=101,
                hotkey="private-worker-a",
                endpoint="http://10.44.0.2:9101",
                proof_endpoint="http://10.44.0.2:9201/proof",
                rpc_endpoint="10.44.0.2:50052",
                stage_index=1,
                layers=StageRange(0, 8),
            ),
            MeshMember(
                uid=102,
                hotkey="private-worker-b",
                endpoint="http://10.44.0.3:9102",
                proof_endpoint="http://10.44.0.3:9202/proof",
                rpc_endpoint="10.44.0.3:50052",
                stage_index=2,
                layers=StageRange(8, 16),
            ),
        ],
        epoch=123,
    )
    spec.validate()
    return spec


def _coordinator(spec: MeshSpec) -> MeshCoordinatorIdentity:
    return MeshCoordinatorIdentity(
        chain_id=945,
        netuid=405,
        coordinator_uid=spec.coordinator_uid,
        coordinator_hotkey=spec.coordinator_hotkey,
        coordinator_evm_address="0x" + "ab" * 20,
        model_index=26,
    )


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


def _binding(index: int) -> MeshVerificationStageBinding:
    return MeshVerificationStageBinding(
        stage_index=index,
        stage_identity_commitment=f"{index + 1:02x}" * 32,
        proof_key_scheme="sr25519",
        proof_key=Keypair.create_from_uri(
            f"//MeshSnapshotBuilderStage{index}"
        ).ss58_address,
        proof_commitment=f"{index + 16:02x}" * 32,
    )


def _build(
    spec: MeshSpec,
    *,
    coordinator: MeshCoordinatorIdentity | None = None,
    policy: MeshVerificationPolicy | None = None,
    stage_bindings: tuple[MeshVerificationStageBinding, ...] | None = None,
    epoch: int | None = None,
):
    return build_mesh_verification_snapshot(
        spec,
        coordinator=coordinator or _coordinator(spec),
        policy=policy or _policy(),
        generation=9,
        epoch=spec.epoch if epoch is None else epoch,
        issued_at_unix=2_000_000_000,
        expires_at_unix=2_000_000_600,
        stage_bindings=stage_bindings or (_binding(1), _binding(2)),
    )


def test_builder_copies_exact_anchors_and_never_hashes_private_locations() -> None:
    spec = _runtime_spec()
    snapshot = _build(spec)

    assert snapshot.signature == ""
    assert snapshot.model == MeshModelAnchors(
        model_id=spec.model_id,
        model_package_hash=spec.model_package_hash,
        model_tensor_manifest_root=spec.model_tensor_manifest_root,
        tokenizer_hash=spec.tokenizer_hash,
        total_layers=spec.total_layers,
        max_context_len=spec.max_context_len,
        quantization_scheme=spec.quantization_scheme,
        activation_dtype=spec.activation_dtype,
    )
    assert tuple((stage.layer_start, stage.layer_end) for stage in snapshot.stages) == (
        (0, 8),
        (8, 16),
    )
    assert snapshot.stages[0].stage_id == derive_opaque_stage_id(
        mesh_id=spec.mesh_id,
        stage_index=1,
        layer_start=0,
        layer_end=8,
        stage_identity_commitment=_binding(1).stage_identity_commitment,
    )

    public_text = str(snapshot.to_dict())
    assert "127.0.0.1" not in public_text
    assert "10.44.0." not in public_text
    assert "private-worker" not in public_text

    rerouted = deepcopy(spec)
    rerouted.members[0].endpoint = "http://192.168.90.1:9990"
    rerouted.members[1].endpoint = "http://192.168.90.2:9991"
    rerouted.members[1].proof_endpoint = "http://192.168.90.2:9992/proof"
    rerouted.members[1].rpc_endpoint = "192.168.90.2:50099"
    rerouted.members[2].endpoint = "http://192.168.90.3:9993"
    rerouted.validate()
    assert _build(rerouted).body_hash_hex() == snapshot.body_hash_hex()

    reversed_bindings = _build(
        spec,
        stage_bindings=(_binding(2), _binding(1)),
    )
    assert reversed_bindings.body_hash_hex() == snapshot.body_hash_hex()


@pytest.mark.parametrize(
    "field_name",
    ["model_package_hash", "model_tensor_manifest_root", "tokenizer_hash"],
)
def test_builder_requires_complete_model_anchors(field_name: str) -> None:
    spec = _runtime_spec()
    setattr(spec, field_name, "")
    with pytest.raises(ValueError, match=field_name):
        _build(spec)


@pytest.mark.parametrize(
    ("max_context_len", "message"),
    [
        (0, r"max_context_len must be in \[1, 4294967295\]"),
        (2**32, "max_context_len must fit uint32"),
    ],
)
def test_builder_requires_positive_uint32_context_limit(
    max_context_len: int,
    message: str,
) -> None:
    spec = _runtime_spec()
    spec.max_context_len = max_context_len
    with pytest.raises(ValueError, match=message):
        _build(spec)


def test_builder_requires_exact_policy_coordinator_and_epoch_bindings() -> None:
    spec = _runtime_spec()

    with pytest.raises(ValueError, match="trace_manifest_format does not match"):
        _build(spec, policy=replace(_policy(), trace_manifest_format="compact-raw-v2"))
    with pytest.raises(ValueError, match="coordinator uid does not match"):
        _build(spec, coordinator=replace(_coordinator(spec), coordinator_uid=8))
    with pytest.raises(ValueError, match="coordinator hotkey does not match"):
        _build(
            spec,
            coordinator=replace(
                _coordinator(spec),
                coordinator_hotkey=Keypair.create_from_uri("//Other").ss58_address,
            ),
        )
    with pytest.raises(ValueError, match="epoch does not match"):
        _build(spec, epoch=spec.epoch + 1)


def test_builder_requires_exact_compute_stage_binding_set() -> None:
    spec = _runtime_spec()

    with pytest.raises(ValueError, match="missing bindings.*2"):
        _build(spec, stage_bindings=(_binding(1),))
    with pytest.raises(ValueError, match="duplicate stage binding.*1"):
        _build(spec, stage_bindings=(_binding(1), _binding(1), _binding(2)))
    with pytest.raises(ValueError, match="unknown or non-compute.*9"):
        _build(spec, stage_bindings=(_binding(1), _binding(2), _binding(9)))
    with pytest.raises(ValueError, match="orchestration-only.*0"):
        _build(spec, stage_bindings=(_binding(0), _binding(1), _binding(2)))


def test_builder_rejects_endpoint_like_public_binding_values() -> None:
    spec = _runtime_spec()
    endpoint_key = replace(_binding(1), proof_key="http://10.44.0.2:9201")
    with pytest.raises(ValueError, match="valid Sr25519 SS58"):
        _build(spec, stage_bindings=(endpoint_key, _binding(2)))
