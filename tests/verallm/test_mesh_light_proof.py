"""Light proof relation: round-trip and tamper resistance.

The light tier proves structural membership, activation chaining, the
consumed input activation, and (for the final projection) that the streamed
tokens are in the top-k of the captured logits. It makes no weight-execution
claim. These tests establish that the assembled payload is self-consistent
(verify accepts an honest one) and binding (verify rejects each tampered
field), which is how the construction's soundness is pinned rather than by
hand-reasoning the hashing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from verallm.mesh.ggml_proof import (
    GGML_OP_MUL_MAT,
    GgmlMulMatTrace,
    GgmlOpManifestEntry,
    op_manifest_membership_payload,
    prove_ggml_light_trace,
    verify_ggml_light_payload,
)
from verallm.mesh.proof import VERATHOS_GGML_LIGHT_PROOF_MODE


def _entry(tmp_path: Path, *, k: int, n: int, tensor_name: str) -> GgmlOpManifestEntry:
    return GgmlOpManifestEntry(
        path=tmp_path / "manifest-0.jsonl",
        created_unix_ns=1,
        manifest_index=0,
        graph_id="graph-0",
        op_index=0,
        op_type=GGML_OP_MUL_MAT,
        tensor_name=tensor_name,
        src0_name=tensor_name,
        src1_name="x-0",
        dst_name="y-0",
        src0_shape=(k, n, 1, 1),
        src1_shape=(k, 1, 1, 1),
        dst_shape=(n, 1, 1, 1),
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu",
        device="CPU",
        proof_eligible=True,
    )


def _trace_and_ctx(tmp_path: Path):
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
    tensor_name = "blk.0.attn_q.weight"
    trace = GgmlMulMatTrace(
        path=tmp_path / "trace.json",
        created_unix_ns=1,
        graph_id="graph-0",
        op_index=0,
        tensor_name=tensor_name,
        src0_name=tensor_name,
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
    entry = _entry(tmp_path, k=k, n=n, tensor_name=tensor_name)
    membership = op_manifest_membership_payload([entry], trace, stage_index=1)
    ctx = {
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
        "input_boundary_root": "aa" * 32,
        "output_boundary_root": "bb" * 32,
    }
    return trace, ctx, membership


def test_light_round_trip_verifies(tmp_path: Path) -> None:
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    assert payload["proof_mode"] == VERATHOS_GGML_LIGHT_PROOF_MODE
    # No weight execution material is present.
    assert "proof" not in payload
    assert "weight_root" not in payload
    assert verify_ggml_light_payload(payload, ctx) is True


def test_light_default_has_no_input_band_and_verifies(tmp_path: Path) -> None:
    # Default light is O(1): membership + boundary + decode-audit only.
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    assert "input_band_openings" not in payload
    assert "input_root" not in payload
    assert verify_ggml_light_payload(payload, ctx) is True


def test_light_rejects_tampered_input_band(tmp_path: Path) -> None:
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership, include_input_band=True
    )
    assert verify_ggml_light_payload(payload, ctx) is True
    tampered = copy.deepcopy(payload)
    band = tampered["input_band_openings"][0]
    # Flip the authenticated span bytes.
    import base64

    raw = bytearray(base64.b64decode(band["span_data_b64"]))
    raw[0] ^= 0xFF
    band["span_data_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(RuntimeError):
        verify_ggml_light_payload(tampered, ctx)


def test_light_rejects_tampered_transcript_context(tmp_path: Path) -> None:
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    tampered = copy.deepcopy(payload)
    tampered["transcript_context"]["stage_index"] = 99
    with pytest.raises(RuntimeError, match="transcript label"):
        verify_ggml_light_payload(tampered, ctx)


def test_light_rejects_boundary_root_mismatch(tmp_path: Path) -> None:
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    other_ctx = dict(ctx)
    other_ctx["input_boundary_root"] = "cc" * 32
    with pytest.raises(RuntimeError, match="boundary_root"):
        verify_ggml_light_payload(payload, other_ctx)


def test_light_requires_membership(tmp_path: Path) -> None:
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(trace, ctx, op_manifest_membership=None)
    with pytest.raises(RuntimeError, match="membership"):
        verify_ggml_light_payload(payload, ctx)


def test_tier_router_accepts_light_for_organic(tmp_path) -> None:
    from verallm.mesh.ggml_proof import verify_mesh_proof_payloads_any_tier

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    mesh_receipt = dict(ctx, proof_challenge_kind="inline_every_request_v1")
    result = verify_mesh_proof_payloads_any_tier(
        [payload], [], mesh_receipt=mesh_receipt
    )
    assert result.verified, result.message


def test_tier_router_rejects_light_for_validator_kinds(tmp_path) -> None:
    from verallm.mesh.ggml_proof import verify_mesh_proof_payloads_any_tier

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    for kind in (
        "fiat_shamir_inline_v1",
        "deferred_future_randomness_v1",
        "validator_postcommit_v1",
        "",
    ):
        mesh_receipt = dict(ctx, proof_challenge_kind=kind)
        result = verify_mesh_proof_payloads_any_tier(
            [payload], [], mesh_receipt=mesh_receipt
        )
        assert not result.verified
        assert "hard proof" in result.message


def test_tier_router_rejects_mixed_payloads(tmp_path) -> None:
    from verallm.mesh.ggml_proof import verify_mesh_proof_payloads_any_tier

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    light = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership
    )
    hard_shaped = dict(light)
    hard_shaped["proof_mode"] = "verathos_ggml_gemm_v1"
    mesh_receipt = dict(ctx, proof_challenge_kind="inline_every_request_v1")
    result = verify_mesh_proof_payloads_any_tier(
        [light, hard_shaped], [], mesh_receipt=mesh_receipt
    )
    assert not result.verified
    assert "mixed" in result.message


def _lmhead_trace_and_ctx(tmp_path: Path):
    # Final-projection trace for decode audit: single row, vocab-wide logits.
    vocab = 12
    x = np.asarray([[0.5, -0.5, 1.0, 0.25]], dtype=np.float32)
    logits = np.arange(vocab, dtype=np.float32)[None, :]  # argmax = vocab-1
    src1 = tmp_path / "lm-src1.f32"
    dst = tmp_path / "lm-dst.f32"
    np.ascontiguousarray(x).tofile(src1)
    np.ascontiguousarray(logits).tofile(dst)
    name = "output.weight"
    trace = GgmlMulMatTrace(
        path=tmp_path / "lm.json", created_unix_ns=1, graph_id="graph-0",
        op_index=0, tensor_name=name, src0_name=name, src1_name="x-0", dst_name="y-0",
        src0_shape=(4, vocab, 1, 1), src1_shape=(4, 1, 1, 1), dst_shape=(vocab, 1, 1, 1),
        src0_f32_path=None, src1_f32_path=src1, dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cpu", device="CPU", manifest_index=0,
    )
    entry = _entry(tmp_path, k=4, n=vocab, tensor_name=name)
    membership = op_manifest_membership_payload([entry], trace, stage_index=1)
    ctx = {
        "request_id": "r", "mesh_id": "m", "mesh_spec_hash": "10" * 32,
        "stage_assignment_hash": "11" * 32, "rpc_plan_hash": "12" * 32, "uid": 1,
        "hotkey": "h", "endpoint": "http://x", "stage_index": 1, "layer_start": 0,
        "layer_end": 1, "request_hash": "13" * 32, "response_hash": "14" * 32,
        "input_boundary_root": "aa" * 32, "output_boundary_root": "bb" * 32,
        "decode_audit_positions": [0], "decode_audit_stage_index": 1,
        "decode_audit_top_k": 8,
    }
    return trace, ctx, membership, vocab


def test_light_decode_audit_round_trip(tmp_path: Path) -> None:
    trace, ctx, membership, vocab = _lmhead_trace_and_ctx(tmp_path)
    argmax = vocab - 1
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership,
        decode_audit_positions=[0], decode_audit_token_ids=[argmax],
    )
    assert "decode_audit_openings" in payload
    assert verify_ggml_light_payload(
        payload, ctx, completion_token_ids=[argmax]
    ) is True


def test_light_verified_wrapper_round_trip_with_receipt(tmp_path: Path) -> None:
    from verallm.mesh.ggml_proof import (
        GGML_LIGHT_NO_ROOT,
        GGML_LIGHT_PROOF_KIND,
        prove_ggml_light_trace_verified,
        verify_mesh_proof_payloads_any_tier,
    )

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    proof = prove_ggml_light_trace_verified(
        trace, ctx, op_manifest_membership=membership
    )
    assert proof.proof_mode == VERATHOS_GGML_LIGHT_PROOF_MODE
    assert proof.receipt.proof_kind == GGML_LIGHT_PROOF_KIND
    assert proof.receipt.input_root == GGML_LIGHT_NO_ROOT
    assert proof.receipt.weight_root == GGML_LIGHT_NO_ROOT
    assert proof.receipt.output_root == GGML_LIGHT_NO_ROOT
    assert (
        proof.receipt.proof_commitment_hash
        == proof.proof_payload["proof_commitment_hash"]
    )
    # The receipt validates like any graph op receipt, so signing plumbing
    # (opaque_stage_v2 conversion) works unchanged.
    proof.receipt.validate()
    mesh_receipt = dict(ctx, proof_challenge_kind="inline_every_request_v1")
    result = verify_mesh_proof_payloads_any_tier(
        [proof.proof_payload],
        [proof.receipt.to_dict()],
        mesh_receipt=mesh_receipt,
    )
    assert result.verified, result.message


def test_light_verified_wrapper_rejects_foreign_receipt(tmp_path: Path) -> None:
    from verallm.mesh.ggml_proof import (
        prove_ggml_light_trace_verified,
        verify_mesh_proof_payloads_any_tier,
    )

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    proof = prove_ggml_light_trace_verified(
        trace, ctx, op_manifest_membership=membership
    )
    other_ctx = dict(ctx, request_hash="99" * 32, request_id="other-request")
    other = prove_ggml_light_trace_verified(
        trace, other_ctx, op_manifest_membership=membership
    )
    mesh_receipt = dict(ctx, proof_challenge_kind="inline_every_request_v1")
    result = verify_mesh_proof_payloads_any_tier(
        [proof.proof_payload],
        [other.receipt.to_dict()],
        mesh_receipt=mesh_receipt,
    )
    assert not result.verified


def test_light_transcript_must_match_receipt_identity(tmp_path: Path) -> None:
    # A fully self-consistent payload lifted from ANOTHER request must not
    # verify against this request's receipt context.
    trace, ctx, membership = _trace_and_ctx(tmp_path)
    other_ctx = dict(ctx, request_hash="99" * 32)
    payload = prove_ggml_light_trace(
        trace, other_ctx, op_manifest_membership=membership
    )
    with pytest.raises(RuntimeError, match="request_hash"):
        verify_ggml_light_payload(payload, ctx)


def test_light_membership_aggregate_binds_to_receipt_root(tmp_path: Path) -> None:
    from verallm.mesh.ggml_proof import (
        mesh_op_manifest_aggregate_root,
        prove_ggml_light_trace_verified,
        verify_mesh_proof_payloads_any_tier,
    )

    trace, ctx, membership = _trace_and_ctx(tmp_path)
    proof = prove_ggml_light_trace_verified(
        trace, ctx, op_manifest_membership=membership
    )
    aggregate = mesh_op_manifest_aggregate_root(
        [
            {
                "stage_index": int(membership["stage_index"]),
                "op_manifest_root": str(membership["op_manifest_root"]),
                "op_manifest_count": int(membership["op_manifest_count"]),
            }
        ]
    )
    good = dict(
        ctx,
        proof_challenge_kind="inline_every_request_v1",
        proof_op_manifest_root=aggregate,
    )
    result = verify_mesh_proof_payloads_any_tier(
        [proof.proof_payload], [proof.receipt.to_dict()], mesh_receipt=good
    )
    assert result.verified, result.message
    bad = dict(good, proof_op_manifest_root="77" * 32)
    result = verify_mesh_proof_payloads_any_tier(
        [proof.proof_payload], [proof.receipt.to_dict()], mesh_receipt=bad
    )
    assert not result.verified
    assert "aggregate" in result.message


def test_light_decode_audit_rejects_token_not_in_committed_topk(tmp_path: Path) -> None:
    trace, ctx, membership, vocab = _lmhead_trace_and_ctx(tmp_path)
    argmax = vocab - 1
    payload = prove_ggml_light_trace(
        trace, ctx, op_manifest_membership=membership,
        decode_audit_positions=[0], decode_audit_token_ids=[argmax],
    )
    # Claim a completion token that is NOT the committed argmax/top-k content.
    with pytest.raises(RuntimeError, match="token id does not match|not in the committed"):
        verify_ggml_light_payload(payload, ctx, completion_token_ids=[0])


def test_prune_stale_proof_traces_removes_only_expired_artifacts(tmp_path):
    """The trace janitor deletes artifacts past the postcommit audit
    window and nothing else: fresh windows, the sentinel warmup trace,
    and unrelated files all survive."""

    import time as _time

    from verallm.mesh.worker import prune_stale_proof_traces

    now_ns = int(_time.time() * 1e9)
    old_ns = now_ns - int(3600 * 1e9)
    keep = [
        tmp_path / f"trace-{now_ns}-cuda0-op7.json",
        tmp_path / f"trace-{now_ns}-cuda0-op7-src1.f32",
        tmp_path / f"manifest-{now_ns}.vmanifest",
        tmp_path / "trace-1-warmup-cuda0-op1.json",
        tmp_path / "token-journal.log",
    ]
    drop = [
        tmp_path / f"trace-{old_ns}-cuda0-op7.json",
        tmp_path / f"trace-{old_ns}-cuda0-op7-dst.f32",
        tmp_path / f"manifest-{old_ns}.vmanifest",
    ]
    for path in keep + drop:
        path.write_bytes(b"x")

    removed = prune_stale_proof_traces(tmp_path)

    assert removed == len(drop)
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {p.name for p in keep}


def test_decode_audit_opening_selects_row_from_multi_row_instance(
    tmp_path: Path,
) -> None:
    """A teacher-forced probe's final chunk computes logits for several
    positions in ONE small GEMM (glm-5.2 live: a 4-row output.weight
    instance). The opening must select the audited position's row by the
    committed-token argmax criterion instead of failing the audit."""
    from verallm.mesh.ggml_proof import make_decode_audit_openings_for_trace

    vocab, rows = 16, 4
    # Strictly decreasing baseline so top-k ranking is unambiguous (a
    # zero-filled row makes argpartition tie-break arbitrarily), then bump
    # row r's argmax to token r+3.
    logits = np.tile(
        -np.arange(vocab, dtype=np.float32), (rows, 1)
    ).astype(np.float32)
    for r in range(rows):
        logits[r, r + 3] = 5.0 + r
    dst = tmp_path / "multi-dst.f32"
    np.ascontiguousarray(logits).tofile(dst)
    trace = GgmlMulMatTrace(
        path=tmp_path / "multi-trace.json",
        created_unix_ns=1,
        graph_id="graph-4",
        op_index=0,
        tensor_name="output.weight",
        src0_name="output.weight",
        src1_name="x-0",
        dst_name="y-0",
        src0_shape=(8, vocab, 1, 1),
        src1_shape=(8, rows, 1, 1),
        dst_shape=(vocab, rows, 1, 1),
        src0_f32_path=None,
        src1_f32_path=None,
        dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cuda",
        device="CUDA3",
        manifest_index=0,
    )
    # Audited position 2 committed token 5 = row 2's argmax.
    openings = make_decode_audit_openings_for_trace(
        trace,
        decode_audit_positions=[2],
        decode_audit_token_ids=[3, 4, 5, 6],
        decode_audit_top_k=4,
    )
    assert len(openings) == 1
    opening = openings[0]
    assert opening["dst_row_index"] == 2
    assert opening["dst_row_count"] == rows
    assert opening["argmax_token_id"] == 5
    assert opening["token_id"] == 5
    assert opening["top_token_ids"][0] == 5

    # A token that is not the argmax but sits in a row's top-k is a valid
    # opening (non-greedy organic traffic): row 1 ranks token 4 first and
    # token 12 second, so auditing token 12 opens row 1.
    logits[1, 12] = 5.5
    np.ascontiguousarray(logits).tofile(dst)
    topk_openings = make_decode_audit_openings_for_trace(
        trace,
        decode_audit_positions=[1],
        decode_audit_token_ids=[3, 12, 5, 6],
        decode_audit_top_k=4,
    )
    assert topk_openings[0]["dst_row_index"] == 1
    assert 12 in topk_openings[0]["top_token_ids"]

    # A token no row carries at all -> the audit fails loudly instead of
    # opening a row the verifier must reject.
    try:
        make_decode_audit_openings_for_trace(
            trace,
            decode_audit_positions=[0],
            decode_audit_token_ids=[15, 4, 5, 6],
            decode_audit_top_k=2,
        )
        raise AssertionError("expected missing-token instance to raise")
    except RuntimeError as exc:
        assert "no row carrying the committed token" in str(exc)


def test_decode_audit_opening_carries_hash_bound_row_for_near_ties(
    tmp_path: Path,
) -> None:
    """An opening whose committed token is not the row argmax must embed the
    hash-bound dst bytes, or the verifier's near-tie tolerance can never
    engage and every serve-vs-replay argmax flip becomes probation. Exact-argmax openings must stay
    lean, and the near-tie check must slice the audited row from a
    multi-row instance."""
    from verallm.mesh.ggml_proof import (
        decode_audit_near_tie_accepts,
        make_decode_audit_openings_for_trace,
    )

    vocab, rows = 16, 4
    logits = np.tile(
        -np.arange(vocab, dtype=np.float32), (rows, 1)
    ).astype(np.float32)
    for r in range(rows):
        logits[r, r + 3] = 5.0 + r
    # Row 1: token 12 is a NEAR-TIED rank-2 (gap 0.02 vs argmax token 4).
    logits[1, 12] = 5.98
    dst = tmp_path / "near-tie-dst.f32"
    np.ascontiguousarray(logits).tofile(dst)
    trace = GgmlMulMatTrace(
        path=tmp_path / "near-tie-trace.json",
        created_unix_ns=1,
        graph_id="graph-5",
        op_index=0,
        tensor_name="output.weight",
        src0_name="output.weight",
        src1_name="x-0",
        dst_name="y-0",
        src0_shape=(8, vocab, 1, 1),
        src1_shape=(8, rows, 1, 1),
        dst_shape=(vocab, rows, 1, 1),
        src0_f32_path=None,
        src1_f32_path=None,
        dst_f32_path=dst,
        source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
        backend="llama_cpp_cuda",
        device="CUDA3",
        manifest_index=0,
    )
    openings = make_decode_audit_openings_for_trace(
        trace,
        decode_audit_positions=[1, 2],
        decode_audit_token_ids=[3, 12, 5, 6],
        decode_audit_top_k=4,
    )
    by_position = {opening["position"]: opening for opening in openings}

    # Position 2 commits row 2's exact argmax: no row bytes attached.
    assert "dst_f32_bytes_hex" not in by_position[2]

    # Position 1 commits a rank-2 token: the full dst dump rides along and
    # the near-tie acceptance verifies it end-to-end from the opening alone.
    near_tie = by_position[1]
    assert near_tie["dst_f32_bytes_hex"] == dst.read_bytes().hex()
    ok, gap = decode_audit_near_tie_accepts(
        near_tie,
        token_id=int(near_tie["token_id"]),
        f32_argmax=int(near_tie["argmax_token_id"]),
        top_k=4,
    )
    assert ok and gap is not None and 0.0 < gap < 0.05

    # A distant substitution in the same geometry stays rejected.
    ok, gap = decode_audit_near_tie_accepts(
        near_tie,
        token_id=8,
        f32_argmax=int(near_tie["argmax_token_id"]),
        top_k=4,
    )
    assert not ok

    # Tampered row geometry never verifies.
    tampered = dict(near_tie)
    tampered["dst_row_index"] = 0
    ok, _ = decode_audit_near_tie_accepts(
        tampered,
        token_id=int(near_tie["token_id"]),
        f32_argmax=int(near_tie["argmax_token_id"]),
        top_k=4,
    )
    assert not ok


def test_decode_audit_near_tie_accepts_flat_rows_by_value_rank() -> None:
    """A fully tied (flat) row is the token-0 degeneration signature: greedy
    picks index 0 while argpartition reports an arbitrary mid-vocab run as
    'top-k' . A faithful pick over a tied row must verify;
    NaN rows and distant substitutions must not."""
    import hashlib as _hashlib

    import numpy as _np

    from verallm.mesh.ggml_proof import decode_audit_near_tie_accepts

    def _opening(row):
        data = _np.asarray(row, dtype="<f4").tobytes()
        return {
            "dst_f32_bytes_hex": data.hex(),
            "dst_f32_sha256": _hashlib.sha256(data).hexdigest(),
        }

    # Flat row: every index is a tied argmax. Committed token 0 with the
    # opening's argpartition-arbitrary argmax index elsewhere is faithful.
    flat = [1.5] * 32
    ok, gap = decode_audit_near_tie_accepts(
        _opening(flat), token_id=0, f32_argmax=17, top_k=8
    )
    assert ok and gap == 0.0

    # The claimed argmax index must actually hold a maximal value.
    bumped = list(flat)
    bumped[3] = 2.0
    ok, _ = decode_audit_near_tie_accepts(
        _opening(bumped), token_id=0, f32_argmax=17, top_k=8
    )
    assert not ok

    # NaN corruption is never accepted as a near-tie.
    nan_row = list(flat)
    nan_row[17] = float("nan")
    ok, _ = decode_audit_near_tie_accepts(
        _opening(nan_row), token_id=0, f32_argmax=17, top_k=8
    )
    assert not ok

    # A token below a large tied plateau is outside the value rank.
    plateau = [5.0] * 16 + [0.0] * 16
    ok, _ = decode_audit_near_tie_accepts(
        _opening(plateau), token_id=20, f32_argmax=2, top_k=8
    )
    assert not ok


def test_scheduler_input_copies_are_not_committed_weights() -> None:
    """The foreign-model template guard may only judge COMMITTED WEIGHT rows.

    A split serve's template also carries scheduler input copies named
    "<backend>#<tensor>#<n>" (live: CUDA0#attn_inp_k_rot#3). They are never
    provable leaves and never appear in the model's tensor manifest, so a
    guard that treated them as foreign rejected an honest template and left
    the mesh in error through its launch self-test.
    """
    from verallm.mesh.ggml_proof import _slot_view_op_is_committed_weight

    for copy_name in (
        "CUDA0#attn_inp_k_rot#3",
        "CUDA3#ffn_moe_weighted#11",
    ):
        assert not _slot_view_op_is_committed_weight({"tensor_name": copy_name})
    for weight_name in ("blk.0.attn_q.weight", "output.weight"):
        assert _slot_view_op_is_committed_weight({"tensor_name": weight_name})
