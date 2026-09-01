"""A relaunched mesh resumes its verification snapshot chain.

A validator pins one verification snapshot for a whole epoch, and the
coordinator refuses requests pinned to any other snapshot as a priced
proof failure. An honest relaunch therefore must not mint a new chain:
same epoch re-serves the persisted snapshot byte-identically, a later
epoch rotates it exactly like the boundary rotation command, and only a
changed assignment (or unusable file) mints fresh.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

from bittensor_wallet import Keypair

from verallm.mesh import pool as pool_mod
from verallm.mesh.pool import resume_mesh_verification_snapshot_chain
from verallm.mesh.types import MeshMember, MeshSpec, StageRange
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
)

_KEYPAIR = Keypair.create_from_uri("//MeshSnapshotResume")


def _spec() -> MeshSpec:
    spec = MeshSpec(
        mesh_id="mesh_resume_fixture",
        mode="private",
        coordinator_uid=7,
        coordinator_hotkey=_KEYPAIR.ss58_address,
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
                hotkey=_KEYPAIR.ss58_address,
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
                layers=StageRange(0, 16),
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
            f"//MeshSnapshotResumeStage{index}"
        ).ss58_address,
        proof_commitment=f"{index + 16:02x}" * 32,
    )


def _persist(tmp_path: Path, *, epoch: int, issued_at: int = 2_000_000_000):
    spec = _spec()
    snapshot = build_mesh_verification_snapshot(
        spec,
        coordinator=_coordinator(spec),
        policy=_policy(),
        generation=9,
        epoch=epoch,
        issued_at_unix=issued_at,
        expires_at_unix=issued_at + 600,
        stage_bindings=(_binding(1),),
    )
    signed = sign_mesh_verification_snapshot(snapshot, _KEYPAIR)
    path = tmp_path / "verification_snapshot.json"
    path.write_text(json.dumps(signed.to_dict()), encoding="utf-8")
    return path, spec, signed


def _resume(path: Path, spec: MeshSpec, *, epoch: int):
    return resume_mesh_verification_snapshot_chain(
        path,
        spec=spec,
        coordinator_identity=_coordinator(spec),
        policy=_policy(),
        stage_bindings=(_binding(1),),
        epoch=epoch,
        keypair=_KEYPAIR,
    )


def test_same_epoch_adopts_byte_identical_snapshot(tmp_path):
    path, spec, signed = _persist(tmp_path, epoch=123)
    resumed = _resume(path, spec, epoch=123)
    assert resumed is not None
    assert resumed.snapshot_hash_hex() == signed.snapshot_hash_hex()
    assert resumed.generation == 9
    assert resumed.signature == signed.signature


def test_later_epoch_rotates_like_the_boundary_command(tmp_path):
    path, spec, signed = _persist(tmp_path, epoch=123)
    resumed = _resume(path, spec, epoch=125)
    assert resumed is not None
    assert resumed.epoch == 125
    assert resumed.generation == 10
    assert resumed.model == signed.model
    assert resumed.mesh_id == signed.mesh_id
    assert resumed.snapshot_hash_hex() != signed.snapshot_hash_hex()


def test_expired_snapshot_still_rotates_recovery_case(tmp_path):
    path, spec, _ = _persist(tmp_path, epoch=123, issued_at=1_600_000_000)
    resumed = _resume(path, spec, epoch=200)
    assert resumed is not None
    assert resumed.epoch == 200
    assert resumed.generation == 10


def test_changed_assignment_mints_fresh(tmp_path):
    path, spec, _ = _persist(tmp_path, epoch=123)
    changed = replace(spec, model_package_hash="d4" * 32)
    assert _resume(path, changed, epoch=123) is None


def test_missing_or_corrupt_file_mints_fresh(tmp_path):
    spec = _spec()
    missing = tmp_path / "verification_snapshot.json"
    assert _resume(missing, spec, epoch=123) is None
    missing.write_text("{not json", encoding="utf-8")
    assert _resume(missing, spec, epoch=123) is None


def test_launch_reuses_registration_mesh_key():
    src = inspect.getsource(pool_mod)
    mint_at = src.index('mesh_key = "m-" + uuid.uuid4().hex[:10]')
    block = src[mint_at : mint_at + 1400]
    # Reuse is gated on a chain-bound registration and on no live mesh
    # holding the key; measurement launches keep fresh instance keys.
    assert "if chain_bound:" in block
    assert '"mesh_registrations"' in block
    assert "previous_key and previous_key not in" in block
    assert "mesh_key = previous_key" in block


def test_drive_resumes_before_minting_and_reports_generation():
    src = inspect.getsource(pool_mod)
    drive_at = src.index("signed = resume_mesh_verification_snapshot_chain(")
    mint_at = src.index("signed = sign_mesh_verification_snapshot(snapshot, keypair)")
    assert drive_at < mint_at
    ready_at = src.index('"event": "drive_ready"')
    ready_block = src[ready_at : ready_at + 2400]
    assert '"snapshot_generation"' in ready_block
    handler_at = src.index('if action == "drive" and event == "drive_ready"')
    handler_block = src[handler_at : src.index('elif action == "fetch"', handler_at)]
    assert 'mesh["snapshot_generation"] = reported_generation' in handler_block


def test_local_request_context_reports_the_persisted_generation(tmp_path):
    path, _spec_value, signed = _persist(tmp_path, epoch=123)
    (tmp_path / "mesh-state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "role": "coordinator",
                "join_secret": "snapshot-resume-test-secret",
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(pool_mod.LocalMeshRunner)
    runner.mesh_dir = path.parent

    _secret, binding = runner._local_request_context()

    assert binding["verification_snapshot_hash"] == signed.snapshot_hash_hex()
    assert binding["verification_snapshot_generation"] == signed.generation


def test_finalized_recheck_is_anchored_on_the_persisted_chain():
    src = inspect.getsource(pool_mod)
    at = src.index("persisted verification snapshot does not match the final mesh")
    block = src[at - 2200 : at]
    assert "generation=int(existing.generation)" in block
    assert "epoch=int(existing.epoch)" in block


def test_factory_honors_and_validates_resumed_mesh_id():
    import pytest

    from verallm.mesh.types import MeshSpec

    kwargs = dict(
        coordinator_uid=7,
        coordinator_hotkey=_KEYPAIR.ss58_address,
        endpoint="http://127.0.0.1:9100",
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="a1" * 32,
        total_layers=16,
    )
    resumed = MeshSpec.new_private_mesh(mesh_id="mesh-" + "ab" * 8, **kwargs)
    assert resumed.mesh_id == "mesh-" + "ab" * 8
    fresh_a = MeshSpec.new_private_mesh(**kwargs)
    fresh_b = MeshSpec.new_private_mesh(**kwargs)
    assert fresh_a.mesh_id != fresh_b.mesh_id
    with pytest.raises(ValueError):
        MeshSpec.new_private_mesh(mesh_id="mesh-notahexid", **kwargs)


def test_relaunch_threads_the_internal_mesh_identity():
    """mesh_key reuse alone is not enough: the snapshot chain and mesh
    state dir are keyed by the internal spec mesh_id. When that id is
    minted from a random nonce on every drive, a relaunch that reuses
    its m- key still lands in a fresh dir and serves a fresh snapshot.
    """

    src = inspect.getsource(pool_mod)
    # Launch reuse also captures the registration's stored mesh id.
    reuse_at = src.index("mesh_key = previous_key")
    assert 'previous_registration.get("mesh_id", "")' in src[reuse_at : reuse_at + 300]
    # The drive command carries it to the coordinator.
    drive_cmd_at = src.index('"action": "drive",')
    assert '"resume_mesh_id"' in src[drive_cmd_at : drive_cmd_at + 900]
    # The coordinator passes it into the spec factory.
    factory_at = src.index("spec = MeshSpec.new_private_mesh(")
    assert 'command.get("resume_mesh_id"' in src[factory_at : factory_at + 800]
    # drive_ready persists the identity on the durable registration.
    ready_at = src.index('mesh["mesh_id"] = mesh_id')
    assert 'ready_registration["mesh_id"] = mesh_id' in src[ready_at : ready_at + 700]


def test_recreated_mesh_state_keeps_the_internal_secret(tmp_path):
    """The stage bindings derive from the mesh state's join secret; a
    resumed mesh that minted a fresh secret could never adopt its own
    persisted snapshot (the anchors always differed), so every relaunch
    silently minted a new chain while reusing both mesh identities.
    """

    from verallm.mesh.state import (
        create_mesh_state,
        state_internal_auth_secret,
    )

    spec = MeshSpec.new_private_mesh(
        coordinator_uid=7,
        coordinator_hotkey=_KEYPAIR.ss58_address,
        endpoint="http://127.0.0.1:9100",
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="a1" * 32,
        total_layers=16,
        mesh_id="mesh-" + "cd" * 8,
    )
    _dir1, state1, token1 = create_mesh_state(spec=spec, root=tmp_path)
    _dir2, state2, token2 = create_mesh_state(spec=spec, root=tmp_path)
    assert state_internal_auth_secret(state1) == state_internal_auth_secret(
        state2
    )
    assert token1.join_secret == token2.join_secret

    fresh = MeshSpec.new_private_mesh(
        coordinator_uid=7,
        coordinator_hotkey=_KEYPAIR.ss58_address,
        endpoint="http://127.0.0.1:9100",
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="a1" * 32,
        total_layers=16,
    )
    _dir3, state3, _tok3 = create_mesh_state(spec=fresh, root=tmp_path)
    assert state_internal_auth_secret(state3) != state_internal_auth_secret(
        state1
    )
