"""Independent authentication for opaque mesh stage proof receipts."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from bittensor_wallet import Keypair

from verallm.mesh.proof import (
    LlamaGraphOpReceipt,
    MeshStageProofReceipt,
    mesh_stage_proof_receipt_root_hex,
    sign_mesh_stage_proof_receipt,
    verify_mesh_stage_proof_receipts_for_snapshot,
)
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    MeshVerificationStage,
    sign_mesh_verification_snapshot,
)


COORDINATOR = Keypair.create_from_uri("//MeshStageAuthCoordinator")
STAGE_KEYS = (
    Keypair.create_from_uri("//MeshStageAuthWorker0"),
    Keypair.create_from_uri("//MeshStageAuthWorker1"),
)


def _snapshot(
    *,
    generation: int = 1,
    commitments: tuple[str, str] = ("81" * 32, "82" * 32),
) -> MeshVerificationSnapshot:
    now = int(time.time())
    unsigned = MeshVerificationSnapshot(
        mesh_id="mesh_stage_auth",
        generation=generation,
        epoch=17,
        issued_at_unix=now - 10,
        expires_at_unix=now + 600,
        coordinator=MeshCoordinatorIdentity(
            chain_id=945,
            netuid=405,
            coordinator_uid=4,
            coordinator_hotkey=COORDINATOR.ss58_address,
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
            quantization_scheme="Q4_K_M",
            activation_dtype="f16",
        ),
        policy=MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format="compact-raw-v3",
            base_proof_sample_bps=10_000,
            organic_decode_sample_bps=10_000,
            canary_decode_sample_bps=10_000,
            proof_ops_per_request=2,
            deferred_proof_enabled=False,
        ),
        expected_stage_count=2,
        stages=(
            MeshVerificationStage(
                stage_id="stg_" + "11" * 16,
                layer_start=0,
                layer_end=16,
                proof_key_scheme="sr25519",
                proof_key=STAGE_KEYS[0].ss58_address,
                proof_commitment=commitments[0],
            ),
            MeshVerificationStage(
                stage_id="stg_" + "22" * 16,
                layer_start=16,
                layer_end=32,
                proof_key_scheme="sr25519",
                proof_key=STAGE_KEYS[1].ss58_address,
                proof_commitment=commitments[1],
            ),
        ),
    )
    return sign_mesh_verification_snapshot(unsigned, COORDINATOR)


def _unsigned_stage_receipt(
    snapshot: MeshVerificationSnapshot,
    stage_index: int,
    *,
    request_id: str = "req-stage-auth",
    request_hash: str = "d4" * 32,
    response_hash: str = "e5" * 32,
) -> MeshStageProofReceipt:
    stage = snapshot.stages[stage_index]
    return MeshStageProofReceipt(
        request_id=request_id,
        mesh_id=snapshot.mesh_id,
        mesh_spec_hash="31" * 32,
        stage_assignment_hash="32" * 32,
        rpc_plan_hash="33" * 32,
        model_package_hash=snapshot.model.model_package_hash,
        model_tensor_manifest_root=snapshot.model.model_tensor_manifest_root,
        model_index=snapshot.coordinator.model_index,
        model_total_layers=snapshot.model.total_layers,
        verification_snapshot_hash=snapshot.snapshot_hash_hex(),
        proof_gate_hash="c4" * 32,
        stage_proof_commitment=stage.proof_commitment,
        stage_id=stage.stage_id,
        stage_index=stage_index,
        layer_start=stage.layer_start,
        layer_end=stage.layer_end,
        request_hash=request_hash,
        response_hash=response_hash,
        graph_id="graph-stage-auth",
        op_index=stage_index + 3,
        op_type="MUL_MAT",
        layer_index=stage.layer_start,
        tensor_name=f"blk.{stage.layer_start}.attn_q.weight",
        input_root=f"{41 + stage_index:02x}" * 32,
        weight_root=f"{51 + stage_index:02x}" * 32,
        output_root=f"{61 + stage_index:02x}" * 32,
        # Chained handoffs: each stage's output boundary is the next stage's
        # input boundary. The first and last ends are open.
        input_boundary_root=_boundary(stage_index),
        output_boundary_root=(
            _boundary(stage_index + 1)
            if stage_index + 1 < len(snapshot.stages)
            else ""
        ),
        quantization="Q4_K_M",
        backend="ggml",
        device="cuda",
        proof_kind="gemm",
        proof_commitment_hash=f"{71 + stage_index:02x}" * 32,
    )


def _boundary(index: int) -> str:
    """Return the activation boundary handed into stage ``index``."""

    return "" if index <= 0 else f"{0xb0 + index:02x}" * 32


def _signed_receipts(
    snapshot: MeshVerificationSnapshot,
) -> list[MeshStageProofReceipt]:
    return [
        sign_mesh_stage_proof_receipt(
            _unsigned_stage_receipt(snapshot, index),
            STAGE_KEYS[index],
            expected_proof_key=snapshot.stages[index].proof_key,
            proof_key_scheme=snapshot.stages[index].proof_key_scheme,
        )
        for index in range(2)
    ]


def _top_receipt(
    snapshot: MeshVerificationSnapshot,
    receipts: list[MeshStageProofReceipt],
) -> dict[str, object]:
    first = receipts[0]
    return {
        "request_id": first.request_id,
        "mesh_id": snapshot.mesh_id,
        "mesh_spec_hash": first.mesh_spec_hash,
        "stage_assignment_hash": first.stage_assignment_hash,
        "rpc_plan_hash": first.rpc_plan_hash,
        "model_package_hash": snapshot.model.model_package_hash,
        "model_tensor_manifest_root": snapshot.model.model_tensor_manifest_root,
        "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
        "request_hash": first.request_hash,
        "response_hash": first.response_hash,
        "proof_gate_hash": first.proof_gate_hash,
        "proof_receipt_format": "opaque_stage_v2",
        "proof_receipt_count": len(receipts),
        "proof_receipt_root": mesh_stage_proof_receipt_root_hex(receipts),
    }


def _verify(
    snapshot: MeshVerificationSnapshot,
    receipts: list[MeshStageProofReceipt],
) -> list[MeshStageProofReceipt]:
    return verify_mesh_stage_proof_receipts_for_snapshot(
        _top_receipt(snapshot, receipts),
        receipts,
        snapshot,
    )


def test_every_stage_receipt_is_independently_authenticated() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)

    assert _verify(snapshot, receipts) == receipts
    assert receipts[0].signature != receipts[1].signature


def test_private_receipt_conversion_binds_and_signs_snapshot_context() -> None:
    snapshot = _snapshot()
    private = LlamaGraphOpReceipt(
        request_id="req-stage-auth",
        mesh_id=snapshot.mesh_id,
        mesh_spec_hash="31" * 32,
        stage_assignment_hash="32" * 32,
        rpc_plan_hash="33" * 32,
        model_package_hash=snapshot.model.model_package_hash,
        model_tensor_manifest_root=snapshot.model.model_tensor_manifest_root,
        uid=91,
        hotkey=STAGE_KEYS[0].ss58_address,
        endpoint="http://private-worker.internal:9338",
        stage_index=0,
        layer_start=0,
        layer_end=16,
        request_hash="d4" * 32,
        response_hash="e5" * 32,
        graph_id="graph-stage-auth",
        op_index=3,
        op_type="MUL_MAT",
        layer_index=0,
        tensor_name="blk.0.attn_q.weight",
        input_root="29" * 32,
        weight_root="33" * 32,
        output_root="3d" * 32,
        quantization="Q4_K_M",
        backend="ggml",
        device="cuda",
        proof_kind="gemm",
        proof_commitment_hash="47" * 32,
    )
    stage = snapshot.stages[0]
    public = MeshStageProofReceipt.from_private_receipt(
        private,
        stage_id=stage.stage_id,
        model_index=snapshot.coordinator.model_index,
        model_total_layers=snapshot.model.total_layers,
        verification_snapshot_hash=snapshot.snapshot_hash_hex(),
        proof_gate_hash="c4" * 32,
        stage_proof_commitment=stage.proof_commitment,
    )
    signed = sign_mesh_stage_proof_receipt(
        public,
        STAGE_KEYS[0],
        expected_proof_key=stage.proof_key,
        proof_key_scheme=stage.proof_key_scheme,
    )

    assert {"uid", "hotkey", "endpoint"}.isdisjoint(signed.to_dict())
    assert verify_mesh_stage_proof_receipts_for_snapshot(
        _top_receipt(snapshot, [signed]),
        [signed],
        snapshot,
        require_complete_coverage=False,
    ) == [signed]


def test_missing_stage_signature_fails_closed() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    receipts[0] = replace(receipts[0], signature="")

    with pytest.raises(ValueError, match="signature is required"):
        mesh_stage_proof_receipt_root_hex(receipts)


def test_wrong_stage_key_is_rejected() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    unsigned = _unsigned_stage_receipt(snapshot, 0)
    receipts[0] = sign_mesh_stage_proof_receipt(
        unsigned,
        STAGE_KEYS[1],
        expected_proof_key=STAGE_KEYS[1].ss58_address,
        proof_key_scheme="sr25519",
    )

    with pytest.raises(RuntimeError, match="stage signature invalid"):
        _verify(snapshot, receipts)


def test_cross_stage_signature_replay_is_rejected() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    stage = snapshot.stages[1]
    receipts[0] = replace(
        receipts[0],
        stage_id=stage.stage_id,
        stage_index=1,
        layer_start=stage.layer_start,
        layer_end=stage.layer_end,
        stage_proof_commitment=stage.proof_commitment,
    )

    with pytest.raises(RuntimeError, match="stage signature invalid"):
        _verify(snapshot, receipts)


def test_cross_request_signature_replay_is_rejected() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    receipts = [replace(item, request_id="req-other") for item in receipts]

    with pytest.raises(RuntimeError, match="stage signature invalid"):
        _verify(snapshot, receipts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("response_hash", "f6" * 32),
        ("model_package_hash", "f7" * 32),
        ("model_index", 27),
        ("model_total_layers", 33),
        ("proof_gate_hash", "fa" * 32),
        ("proof_commitment_hash", "f8" * 32),
        ("output_root", "f9" * 32),
        ("tensor_name", "blk.0.attn_k.weight"),
    ],
)
def test_signed_receipt_field_mutation_is_rejected(
    field: str,
    value: str | int,
) -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    receipts[0] = replace(receipts[0], **{field: value})
    if field in {"response_hash", "proof_gate_hash"}:
        receipts[1] = replace(receipts[1], **{field: value})

    expected = (
        f"proof receipt {field} mismatch"
        if field in {"model_package_hash", "model_index", "model_total_layers"}
        else "stage signature invalid"
    )
    with pytest.raises(RuntimeError, match=expected):
        _verify(snapshot, receipts)


def test_receipt_cannot_move_to_resigned_snapshot_or_commitment() -> None:
    original = _snapshot()
    receipts = _signed_receipts(original)
    replacement = _snapshot(
        generation=2,
        commitments=("91" * 32, "92" * 32),
    )

    with pytest.raises(RuntimeError, match="snapshot hash mismatch"):
        _verify(replacement, receipts)


def test_worker_must_acknowledge_snapshot_stage_proof_commitment() -> None:
    snapshot = _snapshot()
    receipts = _signed_receipts(snapshot)
    wrong_commitment = replace(
        _unsigned_stage_receipt(snapshot, 0),
        stage_proof_commitment="93" * 32,
    )
    receipts[0] = sign_mesh_stage_proof_receipt(
        wrong_commitment,
        STAGE_KEYS[0],
        expected_proof_key=snapshot.stages[0].proof_key,
        proof_key_scheme=snapshot.stages[0].proof_key_scheme,
    )

    with pytest.raises(RuntimeError, match="stage proof commitment mismatch"):
        _verify(snapshot, receipts)


def test_snapshot_rejects_invalid_or_non_dedicated_stage_keys() -> None:
    snapshot = _snapshot()
    invalid = replace(snapshot.stages[0], proof_key="not-a-public-key")
    with pytest.raises(ValueError, match="valid Sr25519 SS58"):
        replace(snapshot, stages=(invalid, snapshot.stages[1]), signature="").validate()

    coordinator_key = replace(
        snapshot.stages[0],
        proof_key=COORDINATOR.ss58_address,
    )
    with pytest.raises(ValueError, match="distinct from the coordinator"):
        replace(
            snapshot,
            stages=(coordinator_key, snapshot.stages[1]),
            signature="",
        ).validate()

    duplicate_key = replace(
        snapshot.stages[1],
        proof_key=snapshot.stages[0].proof_key,
    )
    with pytest.raises(ValueError, match="dedicated proof key"):
        replace(
            snapshot,
            stages=(snapshot.stages[0], duplicate_key),
            signature="",
        ).validate()

    unsupported_scheme = replace(
        snapshot.stages[0],
        proof_key_scheme="ed25519",
    )
    with pytest.raises(ValueError, match="unsupported stage proof key scheme"):
        replace(
            snapshot,
            stages=(unsupported_scheme, snapshot.stages[1]),
            signature="",
        ).validate()


def test_signer_refuses_key_not_declared_by_snapshot() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="does not match snapshot proof key"):
        sign_mesh_stage_proof_receipt(
            _unsigned_stage_receipt(snapshot, 0),
            STAGE_KEYS[1],
            expected_proof_key=snapshot.stages[0].proof_key,
            proof_key_scheme=snapshot.stages[0].proof_key_scheme,
        )
