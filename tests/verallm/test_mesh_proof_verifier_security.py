from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from verallm.mesh.ggml_proof import (
    GGML_OP_MUL_MAT,
    GgmlMulMatTrace,
    GgmlOpManifestEntry,
    GgmlProofVerification,
    MIN_PROOF_OPS_PER_LAYER,
    _pair_proof_payloads_and_receipts,
    _proof_membership_domains,
    _require_challenge_universe_floor,
    _require_fiat_shamir_base_challenges,
    _stage_layer_spans,
    _walk_to_provable_index,
    derive_ggml_output_block_challenge,
    ggml_op_manifest_root,
    ggml_proof_payload_commitment_hash,
    ggml_proof_selection_payload,
    ggml_trace_commitment_root,
    op_manifest_membership_payload,
    prove_ggml_mul_mat_trace,
    select_manifest_challenge_indexes,
    select_trace_challenge_indexes,
    trace_membership_payload,
    verify_ggml_decode_audit_payloads,
    verify_ggml_gemm_proof_payload,
    verify_ggml_gemm_proof_payloads,
)


DIGEST = "ab" * 32
BEACON = "13" * 32


def _entry(
    tmp_path: Path,
    *,
    index: int,
    shape: tuple[int, ...],
    tensor_name: str = "blk.0.attn_q.weight",
) -> GgmlOpManifestEntry:
    return GgmlOpManifestEntry(
        path=tmp_path / f"manifest-{index}.jsonl",
        created_unix_ns=index + 1,
        manifest_index=index,
        graph_id=f"graph-{index}",
        op_index=index,
        op_type=GGML_OP_MUL_MAT,
        tensor_name=tensor_name,
        src0_name=tensor_name,
        src1_name=f"x-{index}",
        dst_name=f"y-{index}",
        src0_shape=shape,
        src1_shape=(int(shape[0]), 1, 1, 1),
        dst_shape=(max(1, int(shape[1])), 1, 1, 1),
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu",
        device="CPU",
    )


def _trace(
    tmp_path: Path,
    *,
    index: int,
    shape: tuple[int, ...],
) -> GgmlMulMatTrace:
    src1 = tmp_path / f"trace-{index}-src1.f32"
    dst = tmp_path / f"trace-{index}-dst.f32"
    src1.write_bytes(b"src1" + bytes([index]))
    dst.write_bytes(b"dst" + bytes([index]))
    return GgmlMulMatTrace(
        path=tmp_path / f"trace-{index}.json",
        created_unix_ns=index + 1,
        graph_id=f"graph-{index}",
        op_index=index,
        tensor_name=f"tensor-{index}",
        src0_shape=shape,
        src1_shape=(int(shape[0]), 1, 1, 1),
        dst_shape=(max(1, int(shape[1])), 1, 1, 1),
        src0_f32_path=None,
        src1_f32_path=src1,
        dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu",
        device="CPU",
        manifest_index=index,
    )


def _declared_payload(raw: dict) -> dict:
    payload = dict(raw)
    payload["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(payload)
    return payload


def test_challenge_domains_filter_vectors_but_keep_moe_planes(tmp_path: Path) -> None:
    vector = _entry(tmp_path, index=0, shape=(16, 1, 1, 1))
    matrix = _entry(tmp_path, index=1, shape=(16, 8, 1, 1))
    moe = _entry(tmp_path, index=2, shape=(16, 8, 4, 1))

    assert ggml_op_manifest_root([vector, matrix, moe]) == ggml_op_manifest_root(
        [matrix, moe]
    )
    selection = ggml_proof_selection_payload(
        traces=[],
        manifest_entries=[vector, matrix, moe],
        receipt_context={
            "stage_index": 3,
            "proof_beacon": BEACON,
            "proof_sampled": True,
            "proof_ops_per_request": 2,
        },
    )
    assert selection["op_manifest_count"] == 2
    assert len(selection["selected_manifest_leaf_indexes"]) == 2

    vector_trace = _trace(tmp_path, index=0, shape=(16, 1, 1, 1))
    matrix_trace = _trace(tmp_path, index=1, shape=(16, 8, 1, 1))
    moe_trace = _trace(tmp_path, index=2, shape=(16, 8, 4, 1))
    assert ggml_trace_commitment_root(
        [vector_trace, matrix_trace, moe_trace]
    ) == ggml_trace_commitment_root([matrix_trace, moe_trace])
    membership = trace_membership_payload(
        [vector_trace, matrix_trace, moe_trace],
        0,
        stage_index=3,
    )
    assert membership["trace_set_count"] == 2


def test_legacy_walk_never_remaps_an_unprovable_draw(tmp_path: Path) -> None:
    vector = _entry(tmp_path, index=0, shape=(16, 1, 1, 1))
    matrix = _entry(tmp_path, index=1, shape=(16, 8, 1, 1))

    with pytest.raises(RuntimeError, match="canonical challenge domain"):
        _walk_to_provable_index(
            [vector, matrix],
            0,
            lambda item: item.src0_shape,
        )


@pytest.mark.parametrize(
    ("scope_name", "selector"),
    [
        ("trace", select_trace_challenge_indexes),
        ("op manifest", select_manifest_challenge_indexes),
        ("slot view", select_manifest_challenge_indexes),
    ],
)
def test_all_fiat_shamir_base_draws_are_required(scope_name, selector) -> None:
    count = 11
    if scope_name == "trace":
        expected = selector(
            beacon=BEACON,
            trace_set_root=DIGEST,
            trace_set_count=count,
            proof_ops_per_request=3,
            stage_index=2,
        )
    else:
        expected = selector(
            beacon=BEACON,
            op_manifest_root=DIGEST,
            op_manifest_count=count,
            proof_ops_per_request=3,
            stage_index=2,
        )
    domains = {
        2: {
            "root": DIGEST,
            "count": count,
            "indexes": set(expected[1:]),
            "payloads_by_index": {},
        }
    }

    with pytest.raises(RuntimeError, match="missing Fiat-Shamir base challenge"):
        _require_fiat_shamir_base_challenges(
            mesh_receipt={
                "proof_beacon": BEACON,
                "proof_sampled": True,
                "proof_ops_per_request": 3,
            },
            domains=domains,
            expected_stage_indexes={2},
            scope_name=scope_name,
        )


def test_decode_witness_may_be_additive_but_never_replace_base_draw() -> None:
    expected = select_manifest_challenge_indexes(
        beacon=BEACON,
        op_manifest_root=DIGEST,
        op_manifest_count=7,
        proof_ops_per_request=1,
        stage_index=4,
    )
    extra = next(index for index in range(7) if index not in expected)
    domains = {
        4: {
            "root": DIGEST,
            "count": 7,
            "indexes": {expected[0], extra},
            "payloads_by_index": {
                expected[0]: {},
                extra: {"decode_audit_openings": [{"position": 0}]},
            },
        }
    }
    _require_fiat_shamir_base_challenges(
        mesh_receipt={
            "proof_beacon": BEACON,
            "proof_sampled": True,
            "proof_ops_per_request": 1,
            "decode_audit_required": True,
            "decode_audit_stage_index": 4,
        },
        domains=domains,
        expected_stage_indexes={4},
        scope_name="op manifest",
    )

    domains[4]["indexes"] = {extra}
    with pytest.raises(RuntimeError, match="missing Fiat-Shamir base challenge"):
        _require_fiat_shamir_base_challenges(
            mesh_receipt={
                "proof_beacon": BEACON,
                "proof_sampled": True,
                "proof_ops_per_request": 1,
                "decode_audit_required": True,
                "decode_audit_stage_index": 4,
            },
            domains=domains,
            expected_stage_indexes={4},
            scope_name="op manifest",
        )


def test_ordinary_verifier_invokes_base_challenge_coverage(monkeypatch) -> None:
    import verallm.mesh.ggml_proof as proof_module

    root = "31" * 32
    expected = select_trace_challenge_indexes(
        beacon=BEACON,
        trace_set_root=root,
        trace_set_count=9,
        proof_ops_per_request=2,
        stage_index=1,
    )
    payload = _declared_payload(
        {
            "trace_membership": {
                "stage_index": 1,
                "trace_set_root": root,
                "trace_set_count": 9,
                "trace_leaf_index": expected[0],
            }
        }
    )
    receipt = {
        "proof_commitment_hash": payload["proof_commitment_hash"],
        "stage_index": 1,
    }
    monkeypatch.setattr(
        proof_module,
        "verify_ggml_gemm_proof_payload",
        lambda *_args, **_kwargs: GgmlProofVerification(True, 0.0),
    )

    result = verify_ggml_gemm_proof_payloads(
        [payload],
        [receipt],
        mesh_receipt={
            "proof_trace_scope": "trace_candidate_set_v1",
            "proof_beacon": BEACON,
            "proof_sampled": True,
            "proof_ops_per_request": 2,
        },
    )

    assert not result.verified
    assert "missing Fiat-Shamir base challenge" in result.message


def test_payload_receipt_pairing_is_unique_and_stage_bound() -> None:
    payload = _declared_payload(
        {"trace_membership": {"stage_index": 1}}
    )
    receipt = {
        "proof_commitment_hash": payload["proof_commitment_hash"],
        "stage_index": 2,
    }
    with pytest.raises(RuntimeError, match="graph receipt stage"):
        _pair_proof_payloads_and_receipts([payload], [receipt])

    receipt["stage_index"] = 1
    with pytest.raises(RuntimeError, match="duplicate proof payload commitment"):
        _pair_proof_payloads_and_receipts(
            [payload, payload],
            [receipt, receipt],
        )


def test_stage_domain_root_and_count_cannot_equivocate() -> None:
    paired = [
        (
            {
                "op_manifest_membership": {
                    "stage_index": 1,
                    "op_manifest_root": "01" * 32,
                    "op_manifest_count": 2,
                    "op_manifest_leaf_index": 0,
                }
            },
            {},
        ),
        (
            {
                "op_manifest_membership": {
                    "stage_index": 1,
                    "op_manifest_root": "02" * 32,
                    "op_manifest_count": 2,
                    "op_manifest_leaf_index": 1,
                }
            },
            {},
        ),
    ]
    with pytest.raises(RuntimeError, match="inconsistent.*root/count"):
        _proof_membership_domains(
            paired,
            membership_key="op_manifest_membership",
            root_key="op_manifest_root",
            count_key="op_manifest_count",
            index_key="op_manifest_leaf_index",
        )


def test_manifest_membership_does_not_synthesize_singleton_trace_tree(
    tmp_path: Path,
) -> None:
    k, n = 4, 3
    x = np.asarray([[1.0, -1.0, 0.5, 2.0]], dtype=np.float32)
    w = np.arange(k * n, dtype=np.float32).reshape(k, n) / 10.0
    y = x @ w
    src0 = tmp_path / "src0.f32"
    src1 = tmp_path / "src1.f32"
    dst = tmp_path / "dst.f32"
    np.ascontiguousarray(w.T).tofile(src0)
    np.ascontiguousarray(x).tofile(src1)
    np.ascontiguousarray(y).tofile(dst)
    trace = GgmlMulMatTrace(
        path=tmp_path / "trace.json",
        created_unix_ns=1,
        graph_id="graph-0",
        op_index=0,
        tensor_name="blk.0.attn_q.weight",
        src0_name="blk.0.attn_q.weight",
        src1_name="x-0",
        dst_name="y-0",
        src0_shape=(k, n, 1, 1),
        src1_shape=(k, 1, 1, 1),
        dst_shape=(n, 1, 1, 1),
        src0_f32_path=src0,
        src1_f32_path=src1,
        dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu",
        device="CPU",
        manifest_index=0,
    )
    entry = _entry(tmp_path, index=0, shape=(k, n, 1, 1))
    membership = op_manifest_membership_payload([entry], trace, stage_index=1)
    proof = prove_ggml_mul_mat_trace(
        trace,
        {
            "request_id": "request",
            "mesh_id": "mesh",
            "mesh_spec_hash": "10" * 32,
            "stage_assignment_hash": "11" * 32,
            "rpc_plan_hash": "12" * 32,
            "uid": 1,
            "hotkey": "hotkey",
            "endpoint": "http://worker.invalid",
            "stage_index": 1,
            "layer_start": 0,
            "layer_end": 1,
            "request_hash": "13" * 32,
            "response_hash": "14" * 32,
        },
        tolerance_abs=1e-6,
        include_proof=True,
        op_manifest_membership=membership,
        verify_before_return=False,
    )

    assert proof.proof_payload["trace_membership"] == {}
    assert proof.proof_payload["op_manifest_membership"] == membership


def test_decode_verifier_rejects_unexpected_position(tmp_path: Path) -> None:
    entry = _entry(
        tmp_path,
        index=0,
        shape=(8, 4, 1, 1),
        tensor_name="output.weight",
    )
    trace = _trace(tmp_path, index=0, shape=(8, 4, 1, 1))
    trace = GgmlMulMatTrace(
        **{
            **trace.__dict__,
            "tensor_name": "output.weight",
            "src0_name": "output.weight",
            "src1_name": "x-0",
            "dst_name": "y-0",
        }
    )
    membership = op_manifest_membership_payload([entry], trace, stage_index=1)
    payload = {
        "trace": {
            "graph_id": entry.graph_id,
            "op_index": entry.op_index,
            "tensor_name": entry.tensor_name,
            "src0_name": entry.src0_name,
            "src1_name": entry.src1_name,
            "dst_name": entry.dst_name,
            "src0_shape": list(entry.src0_shape),
            "src1_shape": list(entry.src1_shape),
            "dst_shape": list(entry.dst_shape),
            "source_types": dict(entry.source_types),
            "backend": entry.backend,
            "device": entry.device,
            "manifest_index": entry.manifest_index,
        },
        "op_manifest_membership": membership,
        "decode_audit_openings": [{"position": 1}],
    }
    result = verify_ggml_decode_audit_payloads(
        {
            "decode_audit_stage_index": 1,
            "decode_audit_positions": [0],
        },
        [payload],
        completion_token_ids=[7, 8],
    )

    assert not result.verified
    assert "unexpected position" in result.message


def test_decode_verifier_optionally_binds_payload_to_graph_receipt_stage() -> None:
    payload = _declared_payload(
        {
            "op_manifest_membership": {"stage_index": 1},
            "decode_audit_openings": [{"position": 0}],
        }
    )
    result = verify_ggml_decode_audit_payloads(
        {
            "decode_audit_stage_index": 1,
            "decode_audit_positions": [0],
        },
        [payload],
        completion_token_ids=[7],
        receipts=[
            {
                "proof_commitment_hash": payload["proof_commitment_hash"],
                "stage_index": 2,
            }
        ],
    )

    assert not result.verified
    assert "graph receipt stage" in result.message


def _secure_transcript_proof(tmp_path: Path):
    k, n = 4, 130
    x = np.asarray([[1.0, -1.0, 0.5, 2.0]], dtype=np.float32)
    w = np.arange(k * n, dtype=np.float32).reshape(k, n) / 10.0
    y = x @ w
    src0 = tmp_path / "secure-src0.f32"
    src1 = tmp_path / "secure-src1.f32"
    dst = tmp_path / "secure-dst.f32"
    np.ascontiguousarray(w.T).tofile(src0)
    np.ascontiguousarray(x).tofile(src1)
    np.ascontiguousarray(y).tofile(dst)
    trace = GgmlMulMatTrace(
        path=tmp_path / "secure-trace.json",
        created_unix_ns=7,
        graph_id="secure-graph",
        op_index=3,
        tensor_name="blk.0.attn_q.weight",
        src0_name="blk.0.attn_q.weight",
        src1_name="secure-x",
        dst_name="secure-y",
        src0_shape=(k, n, 1, 1),
        src1_shape=(k, 1, 1, 1),
        dst_shape=(n, 1, 1, 1),
        src0_f32_path=src0,
        src1_f32_path=src1,
        dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu",
        device="CPU",
        manifest_index=3,
    )
    entry = GgmlOpManifestEntry(
        path=tmp_path / "secure-manifest.jsonl",
        created_unix_ns=7,
        manifest_index=3,
        graph_id=trace.graph_id,
        op_index=trace.op_index,
        op_type=GGML_OP_MUL_MAT,
        tensor_name=trace.tensor_name,
        src0_name=trace.src0_name,
        src1_name=trace.src1_name,
        dst_name=trace.dst_name,
        src0_shape=trace.src0_shape,
        src1_shape=trace.src1_shape,
        dst_shape=trace.dst_shape,
        source_types=trace.source_types,
        backend=trace.backend,
        device=trace.device,
    )
    stage_id = "stg_" + "21" * 16
    context = {
        "request_id": "secure-request",
        "mesh_id": "secure-mesh",
        "mesh_spec_hash": "31" * 32,
        "stage_assignment_hash": "32" * 32,
        "rpc_plan_hash": "33" * 32,
        "model_package_hash": "34" * 32,
        "model_tensor_manifest_root": "",
        "uid": 1,
        "hotkey": "private-worker",
        "endpoint": "http://private.invalid",
        "stage_id": stage_id,
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "request_hash": "35" * 32,
        "response_hash": "36" * 32,
        "verification_snapshot_hash": "37" * 32,
        "model_index": 26,
        "model_total_layers": 1,
        "proof_gate_hash": "38" * 32,
        "proof_beacon": "05" * 32,
        "proof_policy_version": 1,
        "proof_policy_profile": "gguf_mesh_fs_v1",
        "proof_receipt_format": "opaque_stage_v2",
        "proof_trace_manifest_format": "compact-raw-v3",
        "proof_sample_bps": 10_000,
        "proof_sample_denominator": 10_000,
        "proof_ops_per_request": 1,
        "proof_trace_candidates_per_request": 1,
        "proof_challenge_kind": "inline_every_request_v1",
        "proof_deferred": False,
        "verified_sampler_required": True,
        "verified_sampler_mode": "deterministic_v1",
        "verified_sampler_controls_hash": "39" * 32,
        "decode_audit_mode": "gguf_decode_audit_v1",
        "decode_audit_bps": 1_000,
        "decode_audit_top_k": 8,
        "decode_audit_stage_index": 0,
        "proof_trace_scope": "op_manifest_challenge_v1",
        "proof_op_manifest_scope": "",
    }
    membership = op_manifest_membership_payload([entry], trace, stage_index=0)
    proof = prove_ggml_mul_mat_trace(
        trace,
        context,
        tolerance_abs=1e-6,
        include_proof=True,
        op_manifest_membership=membership,
    )
    public_receipt = {**proof.receipt.to_dict(), "stage_id": stage_id}
    mesh_receipt = dict(context)
    return proof, public_receipt, mesh_receipt, trace, entry


def _recommit_payload(payload: dict, receipt: dict) -> tuple[dict, dict]:
    payload["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(payload)
    receipt["proof_commitment_hash"] = payload["proof_commitment_hash"]
    return payload, receipt


def test_transcript_label_is_recomputed_not_read_from_payload(tmp_path: Path) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    payload = copy.deepcopy(proof.proof_payload)
    payload.pop("transcript_label_hex")
    payload, receipt = _recommit_payload(payload, dict(receipt))

    result = verify_ggml_gemm_proof_payload(
        payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert result.verified, result.message


def test_secure_output_block_challenge_is_beacon_bound(tmp_path: Path) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    payload = copy.deepcopy(proof.proof_payload)
    transcript_context = payload["transcript_context"]
    expected = derive_ggml_output_block_challenge(
        beacon=payload["proof_beacon"],
        transcript_context=transcript_context,
        layer_index=0,
        output_shape=payload["output_shape"],
        block_size=payload["proof_block_size"],
    )
    actual = [
        (int(item["bi"]), int(item["bj"]))
        for item in payload["proof"]["block_proofs"]
    ]

    assert expected == (0, 1)
    assert actual == [expected]

    payload["proof_beacon"] = "01" * 32
    mesh_receipt = {**mesh_receipt, "proof_beacon": payload["proof_beacon"]}
    payload, receipt = _recommit_payload(payload, dict(receipt))
    result = verify_ggml_gemm_proof_payload(
        payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert not result.verified
    assert "output block challenge mismatch" in result.message


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("receipt", "request_hash", "41" * 32, "request_hash"),
        ("receipt", "response_hash", "42" * 32, "response_hash"),
        ("receipt", "stage_id", "stg_" + "43" * 16, "stage_id"),
        ("mesh", "verification_snapshot_hash", "44" * 32, "snapshot"),
        ("mesh", "model_index", 27, "model_index"),
        ("mesh", "model_package_hash", "46" * 32, "model_package_hash"),
        ("mesh", "proof_gate_hash", "45" * 32, "proof_gate_hash"),
        ("mesh", "proof_sample_bps", 9_999, "proof_policy_context"),
    ],
)
def test_secure_transcript_rejects_cross_context_replay(
    tmp_path: Path,
    target: str,
    field: str,
    value,
    message: str,
) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    if target == "receipt":
        receipt[field] = value
    else:
        mesh_receipt[field] = value

    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert not result.verified
    assert message in result.message


def test_secure_transcript_rejects_membership_stage_relabel(tmp_path: Path) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    payload = copy.deepcopy(proof.proof_payload)
    payload["op_manifest_membership"]["stage_index"] = 1
    payload, receipt = _recommit_payload(payload, dict(receipt))

    result = verify_ggml_gemm_proof_payload(
        payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert not result.verified
    assert "stage" in result.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proof_block_size", 1),
        ("proof_spot_checks", 1),
    ],
)
def test_verifier_rejects_payload_controlled_proof_strength_downgrade(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    payload = copy.deepcopy(proof.proof_payload)
    payload[field] = value
    payload, receipt = _recommit_payload(payload, dict(receipt))

    result = verify_ggml_gemm_proof_payload(
        payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert not result.verified
    assert "canonical minimum" in result.message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("proof_trace_manifest_format", "compact", "compact-raw-v3"),
        ("proof_trace_scope", "trace_candidate_set_v1", "op-manifest"),
    ],
)
def test_secure_verifier_rejects_membership_scope_downgrade(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    proof, receipt, mesh_receipt, _trace_item, _entry_item = (
        _secure_transcript_proof(tmp_path)
    )
    mesh_receipt[field] = value

    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=receipt,
        mesh_receipt=mesh_receipt,
    )

    assert not result.verified
    assert message in result.message


def test_secure_prover_rejects_out_of_range_tensor_stage(tmp_path: Path) -> None:
    _proof, _receipt, context, trace, entry = _secure_transcript_proof(tmp_path)
    context = dict(context)
    context.update(
        {
            "stage_id": "stg_" + "51" * 16,
            "stage_index": 1,
            "layer_start": 1,
            "layer_end": 2,
            "model_total_layers": 2,
        }
    )
    membership = op_manifest_membership_payload([entry], trace, stage_index=1)

    with pytest.raises(RuntimeError, match="outside the signed stage range"):
        prove_ggml_mul_mat_trace(
            trace,
            context,
            tolerance_abs=1e-6,
            include_proof=True,
            op_manifest_membership=membership,
            verify_before_return=False,
        )


def test_secure_prover_fails_closed_on_unknown_tensor_owner(tmp_path: Path) -> None:
    _proof, _receipt, context, trace, _entry_item = _secure_transcript_proof(tmp_path)
    unknown = GgmlMulMatTrace(
        **{
            **trace.__dict__,
            "tensor_name": "mystery.weight",
            "src0_name": "mystery.weight",
        }
    )
    entry = GgmlOpManifestEntry(
        path=tmp_path / "unknown-manifest.jsonl",
        created_unix_ns=unknown.created_unix_ns,
        manifest_index=int(unknown.manifest_index or 0),
        graph_id=unknown.graph_id,
        op_index=unknown.op_index,
        op_type=GGML_OP_MUL_MAT,
        tensor_name=unknown.tensor_name,
        src0_name=unknown.src0_name,
        src1_name=unknown.src1_name,
        dst_name=unknown.dst_name,
        src0_shape=unknown.src0_shape,
        src1_shape=unknown.src1_shape,
        dst_shape=unknown.dst_shape,
        source_types=unknown.source_types,
        backend=unknown.backend,
        device=unknown.device,
    )
    membership = op_manifest_membership_payload([entry], unknown, stage_index=0)

    with pytest.raises(RuntimeError, match="ownership cannot be established"):
        prove_ggml_mul_mat_trace(
            unknown,
            context,
            tolerance_abs=1e-6,
            include_proof=True,
            op_manifest_membership=membership,
            verify_before_return=False,
        )


def _floor_domains(count: int) -> dict[int, dict[str, object]]:
    return {1: {"root": "a" * 64, "count": count, "indexes": {0}}}


def test_challenge_universe_floor_rejects_shrunken_slot_view_template():
    """A one-entry template makes every draw land on the same op."""

    # 8 layers over 16 completion tokens: a real template emits at least
    # MIN_PROOF_OPS_PER_LAYER ops per layer per token.
    minimum = 16 * 8 * MIN_PROOF_OPS_PER_LAYER
    with pytest.raises(RuntimeError, match="challenge universe is too small"):
        _require_challenge_universe_floor(
            mesh_receipt={"completion_token_count": 16},
            domains=_floor_domains(minimum - 1),
            spans={1: 8},
            scope_name="slot view",
        )


def test_challenge_universe_floor_accepts_a_real_template():
    _require_challenge_universe_floor(
        mesh_receipt={"completion_token_count": 16},
        domains=_floor_domains(16 * 8 * MIN_PROOF_OPS_PER_LAYER),
        spans={1: 8},
        scope_name="slot view",
    )


def test_challenge_universe_floor_skips_stages_that_own_no_layers():
    """The coordinator stage is orchestration-only and proves nothing."""

    _require_challenge_universe_floor(
        mesh_receipt={"completion_token_count": 16},
        domains=_floor_domains(1),
        spans={1: 0},
        scope_name="slot view",
    )


def test_stage_layer_spans_reads_the_claimed_range():
    paired = [({}, {"stage_index": 2, "layer_start": 14, "layer_end": 28})]
    assert _stage_layer_spans(paired) == {2: 14}


class TestBindingViolationTagging:
    """Broken commitments must be distinguishable from ordinary failures.

    The scorer prices them differently, and it can only do that if the
    verifier marks them, so this is a contract between the two.
    """

    def test_a_shrunken_challenge_universe_is_tagged(self):
        from verallm.mesh.proof import is_mesh_binding_violation

        with pytest.raises(RuntimeError) as excinfo:
            _require_challenge_universe_floor(
                mesh_receipt={"completion_token_count": 4},
                domains=_floor_domains(1),
                spans={1: 8},
                scope_name="slot view",
            )

        assert is_mesh_binding_violation(str(excinfo.value))

    def test_an_unrelated_failure_is_not_tagged(self):
        from verallm.mesh.proof import is_mesh_binding_violation

        assert not is_mesh_binding_violation("mesh verification failed")
        assert not is_mesh_binding_violation(None)


class TestSkippedFloatRecomputeMustBeEarned:
    """The float cross-check is what catches a fabricated hidden state.

    The prover may waive it on exactly one path, the final projection under
    decode audit, where the proved quantity is the quantized logit vector
    rather than the trace's float dst. That waiver used to be recorded and
    never read, which made it an unconditional opt-out on the one op where a
    fabricated input pays best.
    """

    @staticmethod
    def _check(payload, receipt=None):
        from verallm.mesh.ggml_proof import (
            _require_skipped_float_recompute_is_justified,
        )

        return _require_skipped_float_recompute_is_justified(
            payload, receipt if receipt is not None else {"layer_start": 0},
        )

    def test_a_payload_that_did_not_skip_is_untouched(self):
        self._check({"float_recompute_skipped": False})
        self._check({})

    def test_skipping_on_a_layer_gemm_is_rejected(self):
        with pytest.raises(RuntimeError, match="not the final projection"):
            self._check(
                {
                    "float_recompute_skipped": True,
                    "tensor_name": "blk.7.ffn_down.weight",
                }
            )

    def test_skipping_without_decode_openings_is_rejected(self):
        with pytest.raises(RuntimeError, match="without decode audit openings"):
            self._check(
                {
                    "float_recompute_skipped": True,
                    "tensor_name": "output.weight",
                }
            )

    def test_a_justified_skip_passes(self):
        self._check(
            {
                "float_recompute_skipped": True,
                "tensor_name": "output.weight",
                "decode_audit_openings": [{"position": 0, "token_id": 101}],
            }
        )

    def test_the_skip_is_a_binding_violation_not_a_plain_failure(self):
        from verallm.mesh.proof import is_mesh_binding_violation

        with pytest.raises(RuntimeError) as excinfo:
            self._check(
                {
                    "float_recompute_skipped": True,
                    "tensor_name": "blk.0.attn_q.weight",
                }
            )

        assert is_mesh_binding_violation(str(excinfo.value))

    def test_a_downstream_stage_must_commit_its_input_boundary(self, monkeypatch):
        from verallm.mesh.proof import MESH_REQUIRE_BOUNDARY_CHAIN_ENV

        monkeypatch.setenv(MESH_REQUIRE_BOUNDARY_CHAIN_ENV, "1")
        payload = {
            "float_recompute_skipped": True,
            "tensor_name": "output.weight",
            "decode_audit_openings": [{"position": 0, "token_id": 101}],
        }

        with pytest.raises(RuntimeError, match="no input activation boundary"):
            self._check(payload, {"layer_start": 16, "input_boundary_root": ""})

        self._check(
            payload, {"layer_start": 16, "input_boundary_root": "b1" * 32},
        )


class TestBindingViolationReasonTagging:
    """Tagging an existing reason must be idempotent.

    The postcommit abort path tags a reason that may already be tagged when a
    retry re-enters it, and double prefixes would make the reason unreadable
    in logs without changing behaviour, which is the kind of thing that
    survives for years.
    """

    def test_an_untagged_reason_is_tagged(self):
        from verallm.mesh.proof import (
            is_mesh_binding_violation,
            mesh_binding_violation_reason,
        )

        tagged = mesh_binding_violation_reason("obligation not fulfilled")

        assert is_mesh_binding_violation(tagged)
        assert tagged.endswith("obligation not fulfilled")

    def test_tagging_twice_changes_nothing(self):
        from verallm.mesh.proof import mesh_binding_violation_reason

        once = mesh_binding_violation_reason("obligation not fulfilled")

        assert mesh_binding_violation_reason(once) == once

    def test_an_empty_reason_is_still_tagged(self):
        from verallm.mesh.proof import (
            is_mesh_binding_violation,
            mesh_binding_violation_reason,
        )

        assert is_mesh_binding_violation(mesh_binding_violation_reason(""))


def test_decode_audit_near_tie_acceptance_reads_only_the_hash_bound_row():
    """A near-tied rank-2 flip is honest cross-path numerics; a distant
    substitution and unverifiable rows stay rejected ."""

    import hashlib as _hashlib

    import numpy as _np

    from verallm.mesh.ggml_proof import decode_audit_near_tie_accepts

    def _opening(row):
        data = _np.asarray(row, dtype="<f4").tobytes()
        return {
            "dst_f32_bytes_hex": data.hex(),
            "dst_f32_sha256": _hashlib.sha256(data).hexdigest(),
        }

    # Near-tie: served token logit within tolerance of the argmax.
    row = [0.0] * 16
    row[3] = 20.00
    row[7] = 19.95
    ok, gap = decode_audit_near_tie_accepts(
        _opening(row), token_id=7, f32_argmax=3, top_k=8
    )
    assert ok and gap is not None and 0.0 < gap < 0.1

    # Distant substitution: same rank-2 slot, gap far past tolerance.
    row_far = [0.0] * 16
    row_far[3] = 20.0
    row_far[7] = 12.0
    ok, gap = decode_audit_near_tie_accepts(
        _opening(row_far), token_id=7, f32_argmax=3, top_k=8
    )
    assert not ok and gap is not None and gap > 1.0

    # Served token outside the audited top-k: rejected even when near.
    row_out = list(range(16))
    ok, _ = decode_audit_near_tie_accepts(
        _opening(row_out), token_id=0, f32_argmax=15, top_k=4
    )
    assert not ok

    # Tampered row hash: never accepted.
    bad = _opening(row)
    bad["dst_f32_sha256"] = "00" * 32
    ok, gap = decode_audit_near_tie_accepts(
        bad, token_id=7, f32_argmax=3, top_k=8
    )
    assert not ok and gap is None

    # Missing row: acceptance impossible.
    ok, gap = decode_audit_near_tie_accepts(
        {}, token_id=7, f32_argmax=3, top_k=8
    )
    assert not ok and gap is None

    # Metadata argmax disagreeing with the actual row: rejected.
    ok, gap = decode_audit_near_tie_accepts(
        _opening(row), token_id=7, f32_argmax=5, top_k=8
    )
    assert not ok and gap is None
