"""Conformance tests for the mesh port of the streaming execution anchor.

Pins the properties the O(log context) hard-audit design rests on: the MMR
frontier is byte-identical to a plain MerkleTree over the same leaves at any
count, the two-level lane leaf is deterministic and tamper-sensitive, row and
lane openings verify against the frozen root and fail closed on any tamper,
the replay container rebuilds the identical root from retained leaf hashes
only, and the accumulator state stays logarithmic.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from verallm.mesh.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorReplayStageV3,
    ProofV3Error,
    ProofV3VerificationError,
    StreamingExecutionAnchorV3,
    build_execution_anchor_lane_opening_v3,
    build_execution_anchor_tree_v3,
    execution_anchor_inventory_digest_v3,
    execution_anchor_lane_bytes_v3,
    execution_anchor_row_leaf_hash_v3,
    verify_execution_anchor_lane_v3,
    verify_execution_anchor_row_v3,
)
from zkllm.crypto.merkle import MerkleTree


STAGE = "l0.mesh_gemm_output"
WIDTH = 4096


def _row(index: int, width: int = WIDTH) -> bytes:
    return bytes((index * 31 + offset) % 251 for offset in range(width))


def _leaf(index: int, width: int = WIDTH) -> bytes:
    return execution_anchor_row_leaf_hash_v3(
        stage_id=STAGE,
        row_index=index,
        row_width=width,
        row_bytes=_row(index, width),
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 100])
def test_streaming_frontier_matches_full_merkle_tree(count: int) -> None:
    leaves = [_leaf(i) for i in range(count)]
    anchor = StreamingExecutionAnchorV3(stage_id=STAGE, row_width=WIDTH)
    for i in range(count):
        anchor.append(_row(i))
    assert anchor.row_count == count
    assert anchor.root == MerkleTree.from_leaf_hashes(leaves).root
    # Logarithmic state: exactly one occupied peak per set bit of the count.
    occupied = [peak for peak in anchor.frontier_hashes if peak is not None]
    assert len(occupied) == bin(count).count("1")


def test_row_leaf_is_deterministic_and_tamper_sensitive() -> None:
    assert _leaf(0) == _leaf(0)
    tampered = bytearray(_row(0))
    tampered[-1] ^= 1
    assert _leaf(0) != execution_anchor_row_leaf_hash_v3(
        stage_id=STAGE,
        row_index=0,
        row_width=WIDTH,
        row_bytes=bytes(tampered),
    )


def test_row_opening_round_trip_and_tamper() -> None:
    rows = tuple(_row(i) for i in range(9))
    commitment, tree = build_execution_anchor_tree_v3(
        stage_id=STAGE, rows=rows
    )
    for index in (0, 4, 8):
        path = tree.get_path(index)
        verify_execution_anchor_row_v3(
            commitment=commitment,
            row_index=index,
            row_bytes=rows[index],
            path=path,
        )
        with pytest.raises(ProofV3VerificationError):
            verify_execution_anchor_row_v3(
                commitment=commitment,
                row_index=index,
                row_bytes=rows[(index + 1) % len(rows)],
                path=path,
            )


def test_lane_opening_round_trip_and_tamper() -> None:
    lane_bytes = execution_anchor_lane_bytes_v3(STAGE)
    width = lane_bytes * 4
    rows = tuple(_row(i, width) for i in range(6))
    commitment, tree = build_execution_anchor_tree_v3(
        stage_id=STAGE, rows=rows
    )
    opening = build_execution_anchor_lane_opening_v3(
        commitment=commitment,
        row_index=3,
        row_bytes=rows[3],
        row_tree=tree,
        lane_index=2,
    )
    lane = verify_execution_anchor_lane_v3(
        commitment=commitment, opening=opening
    )
    assert lane == rows[3][2 * lane_bytes : 3 * lane_bytes]
    bad = replace(
        opening,
        lane_bytes=bytes(
            value ^ 1 if index == 0 else value
            for index, value in enumerate(opening.lane_bytes)
        ),
    )
    with pytest.raises(ProofV3VerificationError):
        verify_execution_anchor_lane_v3(commitment=commitment, opening=bad)


def test_replay_stage_rebuilds_identical_root_from_leaf_hashes() -> None:
    rows = tuple(_row(i) for i in range(21))
    commitment, _tree = build_execution_anchor_tree_v3(
        stage_id=STAGE, rows=rows
    )
    leaf_stream = b"".join(_leaf(i) for i in range(21))
    selected = (2, 19)
    replay = ExecutionAnchorReplayStageV3(
        stage_id=STAGE,
        row_count=21,
        row_width=WIDTH,
        leaf_hashes=leaf_stream,
        selected_rows=tuple((index, rows[index]) for index in selected),
    )
    leaves = [
        replay.leaf_hashes[offset : offset + 32]
        for offset in range(0, len(replay.leaf_hashes), 32)
    ]
    assert MerkleTree.from_leaf_hashes(leaves).root == commitment.root
    corrupt = bytearray(leaf_stream)
    corrupt[0] ^= 1
    corrupt_leaves = [
        bytes(corrupt[offset : offset + 32])
        for offset in range(0, len(corrupt), 32)
    ]
    assert MerkleTree.from_leaf_hashes(corrupt_leaves).root != commitment.root


def test_replay_stage_rejects_malformed_selected_rows() -> None:
    leaf_stream = b"".join(_leaf(i) for i in range(4))
    with pytest.raises(ProofV3Error):
        ExecutionAnchorReplayStageV3(
            stage_id=STAGE,
            row_count=4,
            row_width=WIDTH,
            leaf_hashes=leaf_stream,
            selected_rows=((9, _row(0)),),
        )


def test_inventory_digest_is_order_and_content_bound() -> None:
    commitments = tuple(
        ExecutionAnchorCommitmentV3(
            stage_id=f"l{i}.mesh_gemm_output",
            row_count=8,
            row_width=WIDTH,
            root=bytes([i + 1]) * 32,
        )
        for i in range(3)
    )
    digest = execution_anchor_inventory_digest_v3(commitments)
    assert digest == execution_anchor_inventory_digest_v3(commitments)
    # Out-of-order or duplicated inventories are rejected outright, so a
    # reordering attack cannot even produce a digest.
    swapped = (commitments[1], commitments[0], commitments[2])
    with pytest.raises(ProofV3Error):
        execution_anchor_inventory_digest_v3(swapped)
    changed = (
        replace(commitments[0], root=bytes([9]) * 32),
    ) + commitments[1:]
    assert digest != execution_anchor_inventory_digest_v3(changed)


def test_frontier_replacement_validates_occupancy() -> None:
    anchor = StreamingExecutionAnchorV3(stage_id=STAGE, row_width=WIDTH)
    for i in range(4):
        anchor.append(_row(i))
    reference = StreamingExecutionAnchorV3(stage_id=STAGE, row_width=WIDTH)
    for i in range(5):
        reference.append(_row(i))
    frontier = list(reference.frontier_hashes)
    frontier += [None] * (32 - len(frontier))
    anchor.replace_frontier_after_append(
        added_rows=1, frontier=tuple(frontier)
    )
    assert anchor.row_count == 5
    assert anchor.root == reference.root
    with pytest.raises(ProofV3Error):
        anchor.replace_frontier_after_append(
            added_rows=1, frontier=tuple([None] * 32)
        )


def test_commitment_matches_built_tree() -> None:
    rows = tuple(_row(i) for i in range(7))
    commitment, _tree = build_execution_anchor_tree_v3(
        stage_id=STAGE, rows=rows
    )
    anchor = StreamingExecutionAnchorV3(stage_id=STAGE, row_width=WIDTH)
    for row in rows:
        anchor.append(row)
    assert anchor.commitment() == commitment
