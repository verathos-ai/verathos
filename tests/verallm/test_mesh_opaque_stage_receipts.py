"""Security boundaries for endpoint-free, snapshot-bound mesh proof receipts."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from bittensor_wallet import Keypair

from verallm.mesh.ggml_proof import (
    GgmlProofVerification,
)
from verallm.mesh.proof import (
    LlamaGraphOpReceipt,
    MESH_REQUIRE_BOUNDARY_CHAIN_ENV,
    _require_stage_boundary_chain,
    is_mesh_binding_violation,
    MeshStageProofReceipt,
    VERATHOS_GGML_GEMM_PROOF_MODE,
    VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
    VERATHOS_GGUF_DECODE_AUDIT_MODE,
    VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
    derive_mesh_decode_audit_positions,
    derive_mesh_postcommit_proof_beacon,
    mesh_decode_audit_commitment_hash,
    mesh_proof_gate_hash,
    mesh_receipt_hash,
    mesh_response_commitment_hash,
    mesh_stage_proof_receipt_root_hex,
    mesh_validator_challenge_nonce_commitment,
    sign_mesh_stage_proof_receipt,
    verify_mesh_stage_proof_receipts_for_snapshot,
)
from verallm.mesh.receipt_signing import sign_receipt_hash
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshModelAnchors,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    MeshVerificationStage,
    sign_mesh_verification_snapshot,
)
from verallm.mesh.worker import (
    completion_token_ids_hash,
    prompt_token_ids_hash,
    semantic_openai_response_hash,
    verified_gguf_sampler_controls_hash,
    verify_mesh_inference_artifact,
)


HEX_A = "a1" * 32
HEX_B = "b2" * 32
HEX_C = "c3" * 32
HEX_D = "d4" * 32
HEX_E = "e5" * 32
COORDINATOR = Keypair.create_from_uri("//OpaqueStageReceiptCoordinator")
BOUNDARY_ROOT = "b1" * 32
STAGE_KEYS = (
    Keypair.create_from_uri("//OpaqueStageReceiptWorker0"),
    Keypair.create_from_uri("//OpaqueStageReceiptWorker1"),
)
POSTCOMMIT_NONCE = "88" * 32
POSTCOMMIT_ORIGIN_HASH = "89" * 32
POSTCOMMIT_VALIDATOR_REQUEST_ID = "8a" * 32


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _snapshot(
    *,
    generation: int = 7,
    policy: MeshVerificationPolicy | None = None,
) -> MeshVerificationSnapshot:
    now = int(time.time())
    stages = (
        MeshVerificationStage(
            stage_id="stg_" + "11" * 16,
            layer_start=0,
            layer_end=16,
            proof_key_scheme="sr25519",
            proof_key=STAGE_KEYS[0].ss58_address,
            proof_commitment="91" * 32,
        ),
        MeshVerificationStage(
            stage_id="stg_" + "22" * 16,
            layer_start=16,
            layer_end=32,
            proof_key_scheme="sr25519",
            proof_key=STAGE_KEYS[1].ss58_address,
            proof_commitment="92" * 32,
        ),
    )
    unsigned = MeshVerificationSnapshot(
        mesh_id="mesh_qwen25_opaque_receipts",
        generation=generation,
        epoch=123,
        issued_at_unix=now - 60,
        expires_at_unix=now + 3600,
        coordinator=MeshCoordinatorIdentity(
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            coordinator_hotkey=COORDINATOR.ss58_address,
            coordinator_evm_address="0x" + "ab" * 20,
            model_index=26,
        ),
        model=MeshModelAnchors(
            model_id="qwen2.5-7b-q4-k-m",
            model_package_hash=HEX_A,
            model_tensor_manifest_root=HEX_B,
            tokenizer_hash=HEX_C,
            total_layers=32,
            max_context_len=32_768,
            quantization_scheme="Q4_K_M",
            activation_dtype="f16",
        ),
        policy=policy
        or MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format="compact-raw-v3",
            base_proof_sample_bps=10_000,
            organic_decode_sample_bps=10_000,
            canary_decode_sample_bps=10_000,
            proof_ops_per_request=2,
            deferred_proof_enabled=False,
        ),
        expected_stage_count=2,
        stages=stages,
    )
    return sign_mesh_verification_snapshot(unsigned, COORDINATOR)


def _stage_receipts(
    snapshot: MeshVerificationSnapshot,
    *,
    request_hash: str = HEX_D,
    response_hash: str = HEX_E,
    proof_gate_hash: str = "34" * 32,
) -> list[MeshStageProofReceipt]:
    common = {
        "request_id": POSTCOMMIT_VALIDATOR_REQUEST_ID,
        "mesh_id": snapshot.mesh_id,
        "mesh_spec_hash": "31" * 32,
        "stage_assignment_hash": "32" * 32,
        "rpc_plan_hash": "33" * 32,
        "model_package_hash": snapshot.model.model_package_hash,
        "model_tensor_manifest_root": snapshot.model.model_tensor_manifest_root,
        "model_index": snapshot.coordinator.model_index,
        "model_total_layers": snapshot.model.total_layers,
        "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
        "proof_gate_hash": proof_gate_hash,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "graph_id": "graph-opaque",
        "op_type": "MUL_MAT",
        "quantization": "Q4_K_M",
        "backend": "ggml",
        "device": "cuda",
        "proof_kind": "gemm",
    }
    unsigned = [
        MeshStageProofReceipt(
            **common,
            stage_proof_commitment=snapshot.stages[0].proof_commitment,
            stage_id=snapshot.stages[0].stage_id,
            stage_index=1,
            layer_start=0,
            layer_end=16,
            op_index=10,
            layer_index=7,
            tensor_name="blk.7.attn_q.weight",
            input_root="41" * 32,
            weight_root="42" * 32,
            output_root="43" * 32,
            # Stage 1 is the first present stage, so its input handoff is
            # open; its output handoff is what stage 2 must have received.
            input_boundary_root="",
            output_boundary_root=BOUNDARY_ROOT,
            proof_commitment_hash="44" * 32,
        ),
        MeshStageProofReceipt(
            **common,
            stage_proof_commitment=snapshot.stages[1].proof_commitment,
            stage_id=snapshot.stages[1].stage_id,
            stage_index=2,
            layer_start=16,
            layer_end=32,
            op_index=20,
            layer_index=31,
            tensor_name="output.weight",
            input_root="51" * 32,
            weight_root="52" * 32,
            output_root="53" * 32,
            input_boundary_root=BOUNDARY_ROOT,
            output_boundary_root="",
            proof_commitment_hash="54" * 32,
        ),
    ]
    return [
        sign_mesh_stage_proof_receipt(
            item,
            STAGE_KEYS[index],
            expected_proof_key=snapshot.stages[index].proof_key,
            proof_key_scheme=snapshot.stages[index].proof_key_scheme,
        )
        for index, item in enumerate(unsigned)
    ]


def _top_receipt_binding(
    snapshot: MeshVerificationSnapshot,
    receipts: list[MeshStageProofReceipt],
    *,
    count: int | None = None,
) -> dict[str, Any]:
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
        "proof_receipt_count": len(receipts) if count is None else count,
        "proof_receipt_root": mesh_stage_proof_receipt_root_hex(receipts),
    }


def test_two_stage_receipt_root_is_canonical_and_requires_complete_snapshot_coverage() -> None:
    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    top = _top_receipt_binding(snapshot, receipts)

    assert mesh_stage_proof_receipt_root_hex(reversed(receipts)) == top[
        "proof_receipt_root"
    ]
    assert verify_mesh_stage_proof_receipts_for_snapshot(
        top,
        [item.to_dict() for item in receipts],
        snapshot,
    ) == receipts
    assert verify_mesh_stage_proof_receipts_for_snapshot(
        top,
        [item.to_dict() for item in reversed(receipts)],
        snapshot,
    ) == list(reversed(receipts))

    missing = receipts[:1]
    with pytest.raises(RuntimeError, match="missing proof receipts"):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, missing),
            missing,
            snapshot,
        )


@pytest.mark.parametrize("forbidden", ["uid", "hotkey", "endpoint", "worker_url"])
def test_stage_receipt_schema_rejects_private_identity_routing_and_unknown_fields(
    forbidden: str,
) -> None:
    payload = _stage_receipts(_snapshot())[0].to_dict()
    payload[forbidden] = (
        7 if forbidden == "uid" else "https://private-worker.internal:9338"
    )
    with pytest.raises(ValueError, match=rf"unknown {forbidden}"):
        MeshStageProofReceipt.from_dict(payload)


def test_stage_receipt_schema_rejects_missing_fields() -> None:
    payload = _stage_receipts(_snapshot())[0].to_dict()
    payload.pop("stage_id")
    with pytest.raises(ValueError, match="missing stage_id"):
        MeshStageProofReceipt.from_dict(payload)


def test_top_receipt_count_must_equal_endpoint_free_receipt_count() -> None:
    """A signed root cannot make a false receipt count acceptable."""

    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    with pytest.raises(RuntimeError, match="proof_receipt_count"):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, receipts, count=len(receipts) + 1),
            receipts,
            snapshot,
        )


def test_exact_duplicate_stage_receipt_is_rejected() -> None:
    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    duplicated = [receipts[0], receipts[0], receipts[1]]

    with pytest.raises(RuntimeError, match="duplicate stage proof receipt"):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, duplicated),
            duplicated,
            snapshot,
        )


def test_duplicate_stage_substitution_cannot_satisfy_complete_coverage() -> None:
    snapshot = _snapshot()
    first = _stage_receipts(snapshot)[0]
    duplicate = replace(
        first,
        op_index=first.op_index + 1,
        proof_commitment_hash="61" * 32,
    )
    duplicate = sign_mesh_stage_proof_receipt(
        duplicate,
        STAGE_KEYS[0],
        expected_proof_key=snapshot.stages[0].proof_key,
        proof_key_scheme=snapshot.stages[0].proof_key_scheme,
    )
    receipts = [first, duplicate]
    with pytest.raises(RuntimeError, match="missing proof receipts"):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, receipts),
            receipts,
            snapshot,
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda items, snap: [
                replace(items[0], stage_id=snap.stages[1].stage_id),
                items[1],
            ],
            "opaque stage binding",
        ),
        (
            lambda items, _snap: [
                replace(items[0], layer_end=15),
                items[1],
            ],
            "opaque stage binding",
        ),
        (
            lambda items, _snap: [
                items[0],
                replace(items[1], stage_index=items[0].stage_index),
            ],
            "stage index maps to multiple stages",
        ),
    ],
)
def test_substituted_stage_id_range_or_colliding_runtime_index_fails(
    mutate,
    message: str,
) -> None:
    snapshot = _snapshot()
    receipts = mutate(_stage_receipts(snapshot), snapshot)
    with pytest.raises(RuntimeError, match=message):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, receipts),
            receipts,
            snapshot,
        )


def test_one_opaque_stage_cannot_claim_multiple_runtime_indexes() -> None:
    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    second_opening_for_first_stage = replace(
        receipts[0],
        stage_index=99,
        op_index=99,
        proof_commitment_hash="62" * 32,
    )
    inconsistent = [receipts[0], second_opening_for_first_stage, receipts[1]]
    with pytest.raises(RuntimeError, match="stage.*multiple.*indexes"):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, inconsistent),
            inconsistent,
            snapshot,
        )


def test_stage_receipt_root_mismatch_fails() -> None:
    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    top = _top_receipt_binding(snapshot, receipts)
    top["proof_receipt_root"] = "ff" * 32
    with pytest.raises(RuntimeError, match="proof_receipt_root"):
        verify_mesh_stage_proof_receipts_for_snapshot(top, receipts, snapshot)


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("request_id", "req-substituted", "request_id mismatch"),
        ("mesh_id", "mesh_substituted", "mesh_id mismatch"),
        ("mesh_spec_hash", "71" * 32, "mesh_spec_hash mismatch"),
        ("stage_assignment_hash", "72" * 32, "stage_assignment_hash mismatch"),
        ("rpc_plan_hash", "73" * 32, "rpc_plan_hash mismatch"),
        ("model_package_hash", "74" * 32, "model_package_hash mismatch"),
        (
            "model_tensor_manifest_root",
            "75" * 32,
            "model_tensor_manifest_root mismatch",
        ),
        ("request_hash", "76" * 32, "request_hash mismatch"),
        ("response_hash", "77" * 32, "response_hash mismatch"),
    ],
)
def test_stage_receipt_common_request_and_model_bindings_cannot_be_substituted(
    field: str,
    replacement: str,
    message: str,
) -> None:
    snapshot = _snapshot()
    receipts = _stage_receipts(snapshot)
    receipts[1] = replace(receipts[1], **{field: replacement})
    with pytest.raises(RuntimeError, match=message):
        verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, receipts),
            receipts,
            snapshot,
        )


def _finalize_sampled_receipt(
    receipt: dict[str, Any],
    *,
    challenge_nonce: str,
) -> None:
    """Apply the same post-inference commitments as the coordinator."""

    receipt.pop("signature", None)
    receipt.pop("receipt_hash", None)
    receipt["decode_audit_commitment_hash"] = mesh_decode_audit_commitment_hash(
        receipt
    )
    receipt["proof_gate_hash"] = mesh_proof_gate_hash(receipt)
    beacon = derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=POSTCOMMIT_ORIGIN_HASH,
        proof_gate_hash=receipt["proof_gate_hash"],
        challenge_nonce=challenge_nonce,
    )
    receipt.update(
        {
            "proof_postcommit_origin_receipt_hash": POSTCOMMIT_ORIGIN_HASH,
            "proof_postcommit_challenge_nonce": challenge_nonce,
            "proof_postcommit_finalized": True,
            "proof_beacon": beacon.hex(),
            "proof_sample_value": 0,
            "proof_sampled": True,
            "proof_required": True,
            "decode_audit_sample_value": 0,
            "decode_audit_sampled": True,
            "decode_audit_required": True,
            "decode_audit_positions": derive_mesh_decode_audit_positions(
                beacon=beacon,
                decode_commitment_hash=receipt["decode_audit_commitment_hash"],
                completion_token_count=receipt["completion_token_count"],
            ),
        }
    )
    receipt["mesh_response_commitment_hash"] = mesh_response_commitment_hash(receipt)
    receipt["receipt_hash"] = mesh_receipt_hash(receipt)
    receipt["signature"] = sign_receipt_hash(receipt["receipt_hash"], COORDINATOR)


def test_sampling_domains_exclude_coordinator_routing_and_trace_choices() -> None:
    receipt = {
        "request_id": "91" * 32,
        "request_hash": "92" * 32,
        "response_hash": "93" * 32,
        "semantic_response_hash": "94" * 32,
        "mesh_id": "mesh-stable-sampling",
        "verification_snapshot_hash": "95" * 32,
        "model_package_hash": "96" * 32,
        "model_tensor_manifest_root": "97" * 32,
        "prompt_token_ids_hash": "98" * 32,
        "prompt_token_count": 4,
        "completion_token_ids_hash": "99" * 32,
        "completion_token_count": 3,
        "proof_policy_profile": "gguf_mesh_v1",
        "proof_trace_manifest_format": "compact-raw-v3",
        "proof_sample_bps": 1_000,
        "proof_sample_denominator": 10_000,
        "proof_ops_per_request": 1,
        "proof_trace_candidates_per_request": 1_024,
        "proof_challenge_kind": "fiat_shamir_inline_v1",
        "proof_deferred": False,
        "verified_sampler_mode": "deterministic-v1",
        "verified_sampler_controls_hash": "9a" * 32,
        "decode_audit_mode": VERATHOS_GGUF_DECODE_AUDIT_MODE,
        "decode_audit_bps": 1_000,
        "decode_audit_top_k": VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
        # These bindings are still signed and independently verified, but a
        # coordinator can choose/serialize them and must not grind sampling.
        "mesh_spec_hash": "a1" * 32,
        "stage_assignment_hash": "a2" * 32,
        "rpc_plan_hash": "a3" * 32,
        "uid": 7,
        "hotkey": "5Coordinator",
        "endpoint": "http://private-route.invalid",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 32,
        "runtime": "llama_cpp_rpc",
        "proof_trace_commitment_root": "a4" * 32,
        "proof_trace_commitment_count": 8,
        "proof_trace_scope": "op_manifest_challenge_v1",
        "proof_op_manifest_root": "a5" * 32,
        "proof_op_manifest_count": 1_024,
        "proof_receipt_root": "a6" * 32,
        "inference_started_unix_ns": 1,
        "inference_ended_unix_ns": 2,
    }
    expected_gate = mesh_proof_gate_hash(receipt)
    expected_decode = mesh_decode_audit_commitment_hash(receipt)

    coordinator_malleable = {
        "response_hash": "b0" * 32,
        "mesh_spec_hash": "b1" * 32,
        "stage_assignment_hash": "b2" * 32,
        "rpc_plan_hash": "b3" * 32,
        "uid": 999,
        "hotkey": "5Relabelled",
        "endpoint": "http://another-private-route.invalid",
        "stage_index": 9,
        "layer_start": 99,
        "layer_end": 100,
        "runtime": "relabeled-runtime",
        "proof_trace_commitment_root": "b4" * 32,
        "proof_trace_commitment_count": 999,
        "proof_trace_scope": "trace_candidate_set_v1",
        "proof_op_manifest_root": "b5" * 32,
        "proof_op_manifest_count": 999,
        "proof_receipt_root": "b6" * 32,
        "inference_started_unix_ns": 999,
        "inference_ended_unix_ns": 1_000,
    }
    mutated = {**receipt, **coordinator_malleable}
    assert mesh_proof_gate_hash(mutated) == expected_gate
    assert mesh_decode_audit_commitment_hash(mutated) == expected_decode

    stable_bindings = {
        "request_id": "c1" * 32,
        "request_hash": "c2" * 32,
        "semantic_response_hash": "c3" * 32,
        "mesh_id": "mesh-other",
        "verification_snapshot_hash": "c4" * 32,
        "model_package_hash": "c5" * 32,
        "model_tensor_manifest_root": "c6" * 32,
        "prompt_token_ids_hash": "c7" * 32,
        "prompt_token_count": 5,
        "completion_token_ids_hash": "c8" * 32,
        "completion_token_count": 4,
        "proof_sample_bps": 10_000,
        "decode_audit_bps": 10_000,
    }
    for field, replacement in stable_bindings.items():
        changed = {**receipt, field: replacement}
        assert mesh_proof_gate_hash(changed) != expected_gate, field
        assert mesh_decode_audit_commitment_hash(changed) != expected_decode, field


def _sampled_v2_artifact() -> tuple[
    dict[str, Any], dict[str, Any], MeshVerificationSnapshot
]:
    snapshot = _snapshot()
    nonce = POSTCOMMIT_NONCE
    request = {
        "model": snapshot.model.model_id,
        "messages": [{"role": "user", "content": "prove this mesh"}],
        "stream": False,
        "temperature": 0,
        "verathos": {
            "challenge_nonce_commitment": (
                mesh_validator_challenge_nonce_commitment(
                    nonce,
                    validator_request_id=POSTCOMMIT_VALIDATOR_REQUEST_ID,
                    verification_snapshot_hash=snapshot.snapshot_hash_hex(),
                )
            ),
            "validator_request_id": POSTCOMMIT_VALIDATOR_REQUEST_ID,
            "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
            "decode_audit_bps": 10_000,
        },
    }
    response = {
        "id": "chatcmpl-opaque",
        "model": snapshot.model.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "verified"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    request_hash = _canonical_hash(request)
    response_hash = _canonical_hash(response)
    stage_receipts = _stage_receipts(
        snapshot,
        request_hash=request_hash,
        response_hash=response_hash,
    )
    prompt_ids = [101, 102]
    completion_ids = [42]
    receipt: dict[str, Any] = {
        "version": 1,
        "request_id": stage_receipts[0].request_id,
        "mesh_id": snapshot.mesh_id,
        "mesh_spec_hash": stage_receipts[0].mesh_spec_hash,
        "stage_assignment_hash": stage_receipts[0].stage_assignment_hash,
        "model_package_hash": snapshot.model.model_package_hash,
        "model_tensor_manifest_root": snapshot.model.model_tensor_manifest_root,
        "rpc_endpoints": [],
        "rpc_plan_hash": stage_receipts[0].rpc_plan_hash,
        "runtime": "llama_cpp_rpc",
        "uid": snapshot.coordinator.coordinator_uid,
        "hotkey": snapshot.coordinator.coordinator_hotkey,
        "endpoint": "https://coordinator.example",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": snapshot.model.total_layers,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "semantic_response_hash": semantic_openai_response_hash(response),
        "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
        "verification_snapshot_generation": snapshot.generation,
        "verification_snapshot_epoch": snapshot.epoch,
        "proof_policy_profile": snapshot.policy.profile,
        "proof_trace_manifest_format": snapshot.policy.trace_manifest_format,
        "proof_sample_bps": snapshot.policy.base_proof_sample_bps,
        "proof_sample_denominator": 10_000,
        "proof_ops_per_request": snapshot.policy.proof_ops_per_request,
        "proof_trace_candidates_per_request": (
            snapshot.policy.proof_trace_candidates_per_request
        ),
        "proof_deferred": False,
        "proof_configured_required": True,
        "proof_capture_required": True,
        "proof_metadata_required": True,
        "proof_policy_version": 2,
        "proof_challenge_kind": VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
        "proof_challenge_nonce_commitment": request["verathos"][
            "challenge_nonce_commitment"
        ],
        "proof_validator_hotkey": "5OpaqueStageValidator",
        "proof_postcommit": True,
        "proof_postcommit_origin_receipt_hash": POSTCOMMIT_ORIGIN_HASH,
        "proof_postcommit_challenge_nonce": nonce,
        "proof_postcommit_finalized": True,
        "proof_trace_commitment_root": "81" * 32,
        "proof_trace_commitment_count": 2,
        "proof_trace_scope": "op_manifest_challenge_v1",
        "proof_op_manifest_root": "82" * 32,
        "proof_op_manifest_count": 2,
        "proof_receipt_format": "opaque_stage_v2",
        "proof_receipt_root": mesh_stage_proof_receipt_root_hex(stage_receipts),
        "proof_receipt_count": len(stage_receipts),
        "proof_receipt_verified": True,
        "proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE,
        "verified": True,
        "verified_sampler_required": True,
        "verified_sampler_mode": (
            "deterministic_no_penalty_top_k_1_seed_0_prompt_cache_v2"
        ),
        "verified_sampler_controls_hash": verified_gguf_sampler_controls_hash(),
        "prompt_token_ids_hash": prompt_token_ids_hash(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_source": "llama_cpp_apply_template_tokenize_v1",
        "prompt_template_hash": "83" * 32,
        "completion_token_ids_hash": completion_token_ids_hash(completion_ids),
        "completion_token_count": len(completion_ids),
        "completion_token_source": "llama_cpp_tokens_v1",
        "decode_audit_mode": VERATHOS_GGUF_DECODE_AUDIT_MODE,
        "decode_audit_stage_index": stage_receipts[1].stage_index,
        "decode_audit_bps": snapshot.policy.canary_decode_sample_bps,
        "decode_audit_top_k": VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
        "decode_audit_completion_token_ids": completion_ids,
        "decode_audit_completion_token_ids_hash": completion_token_ids_hash(
            completion_ids
        ),
        "decode_audit_completion_token_count": len(completion_ids),
        "decode_audit_completion_token_source": "llama_cpp_tokens_v1",
        "decode_audit_receipt_root": "84" * 32,
        "decode_audit_verified": True,
    }
    _finalize_sampled_receipt(receipt, challenge_nonce=nonce)
    stage_receipts = _stage_receipts(
        snapshot,
        request_hash=request_hash,
        response_hash=response_hash,
        proof_gate_hash=receipt["proof_gate_hash"],
    )
    receipt["proof_receipt_root"] = mesh_stage_proof_receipt_root_hex(
        stage_receipts
    )
    receipt["proof_receipt_count"] = len(stage_receipts)
    _finalize_sampled_receipt(receipt, challenge_nonce=nonce)
    proof_payloads = [
        {
            "stage_index": item.stage_index,
            "proof_commitment_hash": item.proof_commitment_hash,
        }
        for item in stage_receipts
    ]
    artifact = {
        "receipt": receipt,
        "response": response,
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "proof_receipts": [item.to_dict() for item in stage_receipts],
        "proof_payloads": proof_payloads,
    }
    return artifact, request, snapshot


def _install_lightweight_math_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    import verallm.mesh.ggml_proof as ggml

    def verify_base(payloads, receipts, *, mesh_receipt=None):
        assert mesh_receipt is not None
        payload_pairs = {
            (int(item["stage_index"]), str(item["proof_commitment_hash"]))
            for item in payloads
        }
        receipt_pairs = {
            (int(item["stage_index"]), str(item["proof_commitment_hash"]))
            for item in receipts
        }
        assert payload_pairs == receipt_pairs
        return GgmlProofVerification(True, 0.0)

    def verify_decode(receipt, payloads, *, completion_token_ids, receipts):
        assert len(receipts) == len(payloads)
        assert receipt["decode_audit_stage_index"] in {
            int(item["stage_index"]) for item in payloads
        }
        assert completion_token_ids == [42]
        return GgmlProofVerification(True, 0.0)

    monkeypatch.setattr(ggml, "verify_ggml_gemm_proof_payloads", verify_base)
    monkeypatch.setattr(ggml, "verify_ggml_decode_audit_payloads", verify_decode)


def test_snapshot_bound_artifact_accepts_complete_consistent_v2_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, request, snapshot = _sampled_v2_artifact()
    _install_lightweight_math_verifiers(monkeypatch)

    assert verify_mesh_inference_artifact(
        artifact,
        request,
        require_configured_proof=True,
        require_cryptographic_proof=True,
        verification_snapshot=snapshot,
    )


def test_snapshot_bound_artifact_rejects_private_v1_receipts() -> None:
    artifact, request, snapshot = _sampled_v2_artifact()
    public = MeshStageProofReceipt.from_dict(artifact["proof_receipts"][0])
    artifact["proof_receipts"] = [
        LlamaGraphOpReceipt(
            request_id=public.request_id,
            mesh_id=public.mesh_id,
            mesh_spec_hash=public.mesh_spec_hash,
            stage_assignment_hash=public.stage_assignment_hash,
            rpc_plan_hash=public.rpc_plan_hash,
            model_package_hash=public.model_package_hash,
            model_tensor_manifest_root=public.model_tensor_manifest_root,
            uid=9,
            hotkey="5PrivateWorker",
            endpoint="http://10.0.0.9:9338",
            stage_index=public.stage_index,
            layer_start=public.layer_start,
            layer_end=public.layer_end,
            request_hash=public.request_hash,
            response_hash=public.response_hash,
            graph_id=public.graph_id,
            op_index=public.op_index,
            op_type=public.op_type,
            layer_index=public.layer_index,
            tensor_name=public.tensor_name,
            input_root=public.input_root,
            weight_root=public.weight_root,
            output_root=public.output_root,
            quantization=public.quantization,
            backend=public.backend,
            device=public.device,
            proof_kind=public.proof_kind,
            proof_commitment_hash=public.proof_commitment_hash,
        ).to_dict()
    ]

    with pytest.raises(ValueError, match="forbidden network-location field"):
        verify_mesh_inference_artifact(
            artifact,
            request,
            verification_snapshot=snapshot,
        )


def test_snapshot_bound_artifact_rejects_endpoint_leak_in_proof_payload() -> None:
    artifact, request, snapshot = _sampled_v2_artifact()
    artifact["proof_payloads"][0]["worker_endpoint"] = (
        "http://private-worker.internal:9338"
    )
    with pytest.raises(ValueError, match="forbidden network-location field"):
        verify_mesh_inference_artifact(
            artifact,
            request,
            verification_snapshot=snapshot,
        )


def test_snapshot_bound_artifact_rejects_wrong_signed_snapshot() -> None:
    artifact, request, _snapshot_used = _sampled_v2_artifact()
    other_snapshot = _snapshot(generation=8)
    with pytest.raises(RuntimeError, match="snapshot hash mismatch"):
        verify_mesh_inference_artifact(
            artifact,
            request,
            verification_snapshot=other_snapshot,
        )


def test_snapshot_bound_artifact_rejects_policy_substitution() -> None:
    artifact, request, snapshot = _sampled_v2_artifact()
    receipt = artifact["receipt"]
    receipt["proof_policy_profile"] = "gguf_mesh_weaker_v1"
    receipt["receipt_hash"] = mesh_receipt_hash(receipt)
    receipt["signature"] = sign_receipt_hash(receipt["receipt_hash"], COORDINATOR)
    with pytest.raises(RuntimeError, match="proof policy profile mismatch"):
        verify_mesh_inference_artifact(
            artifact,
            request,
            verification_snapshot=snapshot,
        )


def test_snapshot_bound_artifact_rejects_wrong_decode_stage_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, request, snapshot = _sampled_v2_artifact()
    artifact = deepcopy(artifact)
    artifact["receipt"]["decode_audit_stage_index"] = 1
    _finalize_sampled_receipt(
        artifact["receipt"],
        challenge_nonce=POSTCOMMIT_NONCE,
    )
    resigned_receipts = []
    for index, payload in enumerate(artifact["proof_receipts"]):
        stage_receipt = replace(
            MeshStageProofReceipt.from_dict(payload),
            proof_gate_hash=artifact["receipt"]["proof_gate_hash"],
        )
        resigned_receipts.append(
            sign_mesh_stage_proof_receipt(
                stage_receipt,
                STAGE_KEYS[index],
                expected_proof_key=snapshot.stages[index].proof_key,
                proof_key_scheme=snapshot.stages[index].proof_key_scheme,
            )
        )
    artifact["proof_receipts"] = [item.to_dict() for item in resigned_receipts]
    artifact["receipt"]["proof_receipt_root"] = (
        mesh_stage_proof_receipt_root_hex(resigned_receipts)
    )
    _finalize_sampled_receipt(
        artifact["receipt"],
        challenge_nonce=POSTCOMMIT_NONCE,
    )
    _install_lightweight_math_verifiers(monkeypatch)

    with pytest.raises(RuntimeError, match="decode stage owner mismatch"):
        verify_mesh_inference_artifact(
            artifact,
            request,
            verification_snapshot=snapshot,
        )


class TestStageBoundaryChain:
    """Stages must prove one continuous activation path, not isolated claims.

    Without the chain each stage proves correct arithmetic over an activation
    of its own choosing, so a coordinator can run one cheap stage, fabricate a
    plausible hidden state, and have the rest prove honest arithmetic on the
    fabrication.
    """

    def test_a_consistent_chain_verifies(self):
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)

        assert verify_mesh_stage_proof_receipts_for_snapshot(
            _top_receipt_binding(snapshot, receipts),
            [item.to_dict() for item in receipts],
            snapshot,
        ) == receipts

    def test_a_broken_handoff_is_rejected(self):
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        # The realistic attack signs a well-formed receipt whose handoff does
        # not chain, rather than editing bytes after signing, so the receipt
        # is re-signed and the top binding recomputed around it.
        forged = sign_mesh_stage_proof_receipt(
            replace(
                receipts[1], input_boundary_root="c2" * 32, signature="",
            ),
            STAGE_KEYS[1],
            expected_proof_key=snapshot.stages[1].proof_key,
            proof_key_scheme=snapshot.stages[1].proof_key_scheme,
        )
        chain = [receipts[0], forged]

        with pytest.raises(RuntimeError, match="do not match"):
            verify_mesh_stage_proof_receipts_for_snapshot(
                _top_receipt_binding(snapshot, chain),
                [item.to_dict() for item in chain],
                snapshot,
            )

    def test_a_broken_handoff_is_a_binding_violation(self):
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        broken = replace(receipts[1], input_boundary_root="c2" * 32)

        with pytest.raises(RuntimeError) as excinfo:
            _require_stage_boundary_chain([receipts[0], broken], stage_count=2)

        assert is_mesh_binding_violation(str(excinfo.value))

    def test_an_omitted_interior_boundary_is_rejected_once_required(
        self, monkeypatch,
    ):
        """Dropping the root must not be an escape from proving the handoff.

        Presence is opt-in until a live run confirms the capture side, so this
        asserts the behaviour the flag turns on.
        """

        monkeypatch.setenv(MESH_REQUIRE_BOUNDARY_CHAIN_ENV, "1")
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        stripped = replace(receipts[1], input_boundary_root="")

        with pytest.raises(RuntimeError, match="missing its input activation"):
            _require_stage_boundary_chain(
                [receipts[0], stripped], stage_count=2,
            )

    def test_a_mesh_that_reports_no_boundaries_is_rejected_by_default(
        self, monkeypatch,
    ):
        """Default-on since the full-stack e2e confirmed roots chain live.

        A capture-less build can still opt out explicitly, which is what the
        second half exercises.
        """

        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        bare = [
            replace(item, input_boundary_root="", output_boundary_root="")
            for item in receipts
        ]

        with pytest.raises(RuntimeError, match="missing its .* boundary"):
            _require_stage_boundary_chain(bare, stage_count=2)

        monkeypatch.setenv("VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN", "0")
        _require_stage_boundary_chain(bare, stage_count=2)

    def test_reported_boundaries_must_chain_even_before_they_are_required(self):
        """Tolerating absence must not mean tolerating a mismatch."""

        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        broken = replace(receipts[1], input_boundary_root="c2" * 32)

        with pytest.raises(RuntimeError, match="do not match"):
            _require_stage_boundary_chain([receipts[0], broken], stage_count=2)

    def test_a_single_stage_mesh_has_no_handoff_to_prove(self):
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)

        _require_stage_boundary_chain([receipts[0]], stage_count=1)

    def test_one_stage_cannot_report_two_different_boundaries(self):
        snapshot = _snapshot()
        receipts = _stage_receipts(snapshot)
        contradictory = replace(receipts[0], output_boundary_root="d3" * 32)

        with pytest.raises(RuntimeError, match="two different activation"):
            _require_stage_boundary_chain(
                [receipts[0], contradictory], stage_count=2,
            )


def test_decode_audit_positions_avoid_the_final_completion_position():
    """The final token's next-token distribution is the flattest of the
    completion (sentence-final punctuation, EOS-adjacent mass) and its
    top-k membership does not survive replay-vs-serve numerics on split
    meshes: anchoring EVERY audit at count-1 cycled an honest 4-GPU mesh
    through probation on false attributions all night .
    The anchor sits at count-2 and random candidates exclude count-1;
    a single-token completion still audits its only position."""
    beacon = b"\x11" * 32
    commitment = "ab" * 32
    for count in (2, 7, 200, 1024, 5000):
        positions = derive_mesh_decode_audit_positions(
            beacon=beacon,
            decode_commitment_hash=commitment,
            completion_token_count=count,
        )
        assert positions, count
        assert (count - 1) not in positions, (count, positions)
        assert (count - 2) in positions, (count, positions)
        assert all(0 <= p < count for p in positions)
    assert derive_mesh_decode_audit_positions(
        beacon=beacon,
        decode_commitment_hash=commitment,
        completion_token_count=1,
    ) == [0]
    # Deterministic across calls (both sides must derive identically).
    a = derive_mesh_decode_audit_positions(
        beacon=beacon, decode_commitment_hash=commitment,
        completion_token_count=5000,
    )
    b = derive_mesh_decode_audit_positions(
        beacon=beacon, decode_commitment_hash=commitment,
        completion_token_count=5000,
    )
    assert a == b
