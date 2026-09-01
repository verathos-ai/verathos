"""Tail-ring probe-free light tier: flush-group coverage semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from verallm.mesh.ggml_proof import (
    GgmlMulMatTrace,
    prove_ggml_light_trace,
)
from verallm.mesh.worker import (
    tail_group_covers_positions,
    tail_group_position_offset,
)


def test_synthetic_base_leaf_trace_builds_light_payload() -> None:
    # A base light leaf carries no witness files at all: the payload must
    # serialize from pure metadata (the live regression: manifest_index=None
    # crashed the serializer and 500'd every light receipt).
    k, n = 8, 6
    trace = GgmlMulMatTrace(
        path=Path("slot-view-leaf-41.synthetic"),
        created_unix_ns=1,
        graph_id="slot-view-leaf-41",
        op_index=41,
        tensor_name="blk.3.ffn_down.weight",
        src0_name="blk.3.ffn_down.weight",
        src1_name="",
        dst_name="",
        src0_shape=(k, n, 1, 1),
        src1_shape=(k, 1, 1, 1),
        dst_shape=(n, 1, 1, 1),
        src0_f32_path=None,
        src1_f32_path=None,
        dst_f32_path=None,
        source_types={"src0": "q4_K", "src1": "f32", "dst": "f32"},
        backend="llama_cpp_cuda",
        device="CUDA0",
        manifest_index=41,
        graph_seq=-1,
        intra_graph_index=5,
    )
    ctx = {
        "request_id": "request",
        "mesh_id": "mesh",
        "mesh_spec_hash": "10" * 32,
        "stage_assignment_hash": "11" * 32,
        "uid": 1,
        "hotkey": "hotkey",
        "endpoint": "http://worker.invalid",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "request_hash": "13" * 32,
        "response_hash": "14" * 32,
    }
    payload = prove_ggml_light_trace(trace, ctx)
    assert payload["trace"]["manifest_index"] == 41
    assert payload["trace"]["intra_graph_index"] == 5
    assert payload["proof_commitment_hash"]


def _write_row(directory: Path, name: str, argmax_token: int, vocab: int = 64) -> str:
    row = np.linspace(-4.0, -1.0, vocab, dtype=np.float32)
    row[argmax_token] = 5.0
    (directory / name).write_bytes(row.tobytes())
    return name


def _group(directory: Path, argmax_by_seq: list[int]) -> list[dict]:
    total = len(argmax_by_seq)
    group = []
    for seq, token in enumerate(argmax_by_seq):
        dst = _write_row(directory, f"tail-{seq}-dst.f32", token)
        group.append(
            {
                "tail_ring": 1,
                "tail_seq": seq,
                "tail_total": total,
                "dst_f32": dst,
                "_tail_path": str(directory / f"tail-{seq}.json"),
            }
        )
    return group


def test_positions_map_to_trailing_sequences(tmp_path: Path) -> None:
    # 10 committed tokens, ring kept the last 4 instances (tokens 6..9).
    committed = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    group = _group(tmp_path, [committed[6], committed[7], committed[8], committed[9]])
    assert tail_group_covers_positions(
        group, positions=[9, 7], committed_ids=committed, top_k=8
    )
    # A position older than the ring window is not covered.
    assert not tail_group_covers_positions(
        group, positions=[5], committed_ids=committed, top_k=8
    )


def test_committed_token_must_be_in_top_k(tmp_path: Path) -> None:
    committed = [3, 4, 5]
    group = _group(tmp_path, [committed[0], committed[1], committed[2]])
    assert tail_group_covers_positions(
        group, positions=[2], committed_ids=committed, top_k=8
    )
    # A row whose top-k excludes the committed token fails coverage: the
    # exact case an off-by-one instance mapping would produce.
    wrong = _group(tmp_path, [committed[0], committed[1], 60])
    assert not tail_group_covers_positions(
        wrong, positions=[2], committed_ids=committed, top_k=1
    )


def test_short_reply_fully_inside_ring(tmp_path: Path) -> None:
    committed = [7, 8]
    group = _group(tmp_path, [7, 8])
    assert tail_group_covers_positions(
        group, positions=[0, 1], committed_ids=committed, top_k=8
    )


def test_stop_draw_reply_offsets_by_one(tmp_path: Path) -> None:
    # A naturally ended reply has one extra trailing instance: the graph
    # that produced the stop token's logits. The live regression: every
    # EOS-terminated audit draw missed because the mapping assumed the
    # group's last instance was the last committed position, so it opened
    # the NEXT position's row for every audit.
    committed = [10, 20, 30]
    group = _group(tmp_path, [10, 20, 30, 63])
    assert tail_group_covers_positions(
        group, positions=[0, 2], committed_ids=committed, top_k=1
    )
    assert (
        tail_group_position_offset(
            group, positions=[0, 2], committed_ids=committed, top_k=1
        )
        == 0
    )


def test_length_capped_reply_uses_trailing_offset(tmp_path: Path) -> None:
    # A capped reply samples no stop token: the last instance IS the last
    # committed position and the trailing mapping must still resolve.
    committed = [10, 20, 30]
    group = _group(tmp_path, [10, 20, 30])
    assert (
        tail_group_position_offset(
            group, positions=[0, 2], committed_ids=committed, top_k=1
        )
        == 0
    )
    # Ring-capped: only the last two instances survive; the offset shifts.
    capped = _group(tmp_path, [20, 30])
    assert (
        tail_group_position_offset(
            capped, positions=[2], committed_ids=committed, top_k=1
        )
        == -1
    )


def test_prewarm_makes_first_seen_lengths_prefix_hits() -> None:
    # After a prewarm to the ceiling, a previously unseen shorter length
    # must not rebuild leaves - only the ragged right edge (at most one
    # node per level) may hash fresh.
    import hashlib

    from verallm.mesh import ggml_proof
    from verallm.mesh.ggml_proof import (
        GgmlOpManifestEntry,
        slot_view_template_from_manifest_entries,
    )
    from verallm.mesh.worker import prewarm_slot_view_caches

    entries = [
        GgmlOpManifestEntry(
            path=Path("manifest-0.jsonl"),
            created_unix_ns=idx + 1,
            manifest_index=idx,
            graph_id=f"graph-{idx}",
            op_index=idx,
            op_type="GGML_OP_MUL_MAT",
            tensor_name=f"blk.{idx}.attn_q.weight",
            src0_name=f"blk.{idx}.attn_q.weight",
            src1_name="x",
            dst_name="y",
            src0_shape=(8, 6, 1, 1),
            src1_shape=(8, 1, 1, 1),
            dst_shape=(6, 1, 1, 1),
            source_types={"src0": "F32", "src1": "F32", "dst": "F32"},
            backend="llama_cpp_cpu",
            device="CPU",
            proof_eligible=True,
            graph_seq=1,
            intra_graph_index=idx,
        )
        for idx in range(3)
    ]
    template = slot_view_template_from_manifest_entries(entries)
    with ggml_proof._slot_view_memo_lock:
        ggml_proof._slot_view_leaf_cache.clear()
        ggml_proof._slot_view_levels_cache.clear()
        ggml_proof._slot_view_root_cache.clear()

    journal_lines: list[str] = []
    prewarm_slot_view_caches(
        template,
        target_tokens=64,
        chunk_tokens=16,
        chunk_sleep_s=0.0,
        journal=journal_lines.append,
    )
    assert journal_lines and "prewarm tokens=64" in journal_lines[0]

    calls = {"n": 0}
    original = ggml_proof._manifest_node_hash

    def counting(left, right):
        calls["n"] += 1
        return original(left, right)

    ggml_proof._manifest_node_hash = counting
    try:
        root, count = ggml_proof.ggml_slot_view_root_from_template(
            template=template, completion_token_count=37
        )
    finally:
        ggml_proof._manifest_node_hash = original
    assert count == 37 * 3
    assert len(root) == 64 and int(root, 16) >= 0
    # log2(111 leaves) ~ 7 levels; one ragged node per level at most.
    assert calls["n"] <= 16, f"first-seen shorter length rehashed {calls['n']}"


def test_incomplete_or_inconsistent_groups_fail(tmp_path: Path) -> None:
    committed = [1, 2, 3]
    group = _group(tmp_path, [1, 2, 3])
    # Missing member for the mapped sequence.
    partial = [item for item in group if item["tail_seq"] != 2]
    assert not tail_group_covers_positions(
        partial, positions=[2], committed_ids=committed, top_k=8
    )
    # Mixed tail_total values mean a torn flush.
    group[1]["tail_total"] = 99
    assert not tail_group_covers_positions(
        group, positions=[2], committed_ids=committed, top_k=8
    )
    assert not tail_group_covers_positions(
        [], positions=[0], committed_ids=committed, top_k=8
    )
