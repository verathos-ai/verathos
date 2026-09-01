"""Bounded anchor row openings: determinism, binding, and tamper resistance.

These pin the properties that make the O(log context) audit sound: the
prover cannot choose which rows it answers with, every returned row is
Merkle-bound to the pre-nonce commitment, and the cost is bounded and
independent of context length.
"""

from __future__ import annotations

import pytest

from verallm.mesh.anchor_audit import (
    ANCHOR_AUDIT_ROWS_PER_STAGE,
    AnchorRowOpening,
    build_anchor_row_openings,
    select_anchor_audit_rows,
    verify_anchor_row_openings,
)
from verallm.mesh.anchor_streams import AnchorStageStream
from verallm.mesh.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ProofV3Error,
    ProofV3VerificationError,
    StreamingExecutionAnchorV3,
    execution_anchor_row_leaf_hash_v3,
)

STAGE = "blk.3.ffn_gate.weight:dst"
WIDTH = 2048
BEACON = "ab" * 32


def _row(index: int) -> bytes:
    return bytes((index * 41 + offset) % 251 for offset in range(WIDTH))


def _stream(rows: int) -> AnchorStageStream:
    anchor = StreamingExecutionAnchorV3(stage_id=STAGE, row_width=WIDTH)
    leaves = []
    for index in range(rows):
        leaf = execution_anchor_row_leaf_hash_v3(
            stage_id=STAGE,
            row_index=index,
            row_width=WIDTH,
            row_bytes=_row(index),
        )
        leaves.append(leaf)
        anchor.append_leaf_hash(leaf)
    return AnchorStageStream(
        stage_id=STAGE,
        row_width=WIDTH,
        leaf_hashes=tuple(leaves),
        commitment=anchor.commitment(),
    )


def _rows_for(stream: AnchorStageStream) -> dict[int, bytes]:
    selected = select_anchor_audit_rows(
        beacon=BEACON, commitment=stream.commitment
    )
    return {index: _row(index) for index in selected}


@pytest.mark.parametrize("rows", [1, 2, 5, 64, 4096, 26214])
def test_selection_is_bounded_deterministic_and_covers_the_ends(
    rows: int,
) -> None:
    stream = _stream(min(rows, 64)) if rows <= 64 else None
    commitment = (
        stream.commitment
        if stream is not None
        else ExecutionAnchorCommitmentV3(
            stage_id=STAGE,
            row_count=rows,
            row_width=WIDTH,
            root=bytes(range(32)),
        )
    )
    selected = select_anchor_audit_rows(beacon=BEACON, commitment=commitment)
    assert selected == select_anchor_audit_rows(
        beacon=BEACON, commitment=commitment
    )
    assert len(selected) == min(ANCHOR_AUDIT_ROWS_PER_STAGE, rows)
    assert len(set(selected)) == len(selected)
    assert all(0 <= index < rows for index in selected)
    assert 0 in selected
    if rows > 1:
        assert rows - 1 in selected


def test_selection_changes_with_the_beacon() -> None:
    commitment = _stream(1000).commitment
    first = select_anchor_audit_rows(beacon="ab" * 32, commitment=commitment)
    second = select_anchor_audit_rows(beacon="cd" * 32, commitment=commitment)
    assert first != second


def test_selection_is_bound_to_the_commitment() -> None:
    base = _stream(1000).commitment
    other = ExecutionAnchorCommitmentV3(
        stage_id=base.stage_id,
        row_count=base.row_count,
        row_width=base.row_width,
        root=bytes([7]) * 32,
    )
    assert select_anchor_audit_rows(
        beacon=BEACON, commitment=base
    ) != select_anchor_audit_rows(beacon=BEACON, commitment=other)


def test_round_trip_openings_verify() -> None:
    stream = _stream(500)
    openings = build_anchor_row_openings(
        beacon=BEACON, stream=stream, rows=_rows_for(stream)
    )
    verified = verify_anchor_row_openings(
        beacon=BEACON,
        commitment=stream.commitment,
        openings=[item.to_dict() for item in openings],
    )
    assert set(verified) == set(
        select_anchor_audit_rows(beacon=BEACON, commitment=stream.commitment)
    )
    for index, row in verified.items():
        assert row == _row(index)


def test_opening_path_is_logarithmic_in_context() -> None:
    small = _stream(16)
    large = _stream(4096)
    small_openings = build_anchor_row_openings(
        beacon=BEACON, stream=small, rows=_rows_for(small)
    )
    large_openings = build_anchor_row_openings(
        beacon=BEACON, stream=large, rows=_rows_for(large)
    )
    assert len(small_openings) == len(large_openings)
    assert len(small_openings[0].sibling_hashes) == 4
    assert len(large_openings[0].sibling_hashes) == 12


def test_tampered_row_fails_verification() -> None:
    stream = _stream(500)
    openings = [
        item.to_dict()
        for item in build_anchor_row_openings(
            beacon=BEACON, stream=stream, rows=_rows_for(stream)
        )
    ]
    raw = bytearray(bytes.fromhex(openings[0]["row_hex"]))
    raw[0] ^= 1
    openings[0]["row_hex"] = bytes(raw).hex()
    with pytest.raises(ProofV3VerificationError):
        verify_anchor_row_openings(
            beacon=BEACON, commitment=stream.commitment, openings=openings
        )


def test_prover_cannot_substitute_its_own_rows() -> None:
    stream = _stream(500)
    honest = build_anchor_row_openings(
        beacon=BEACON, stream=stream, rows=_rows_for(stream)
    )
    # Swap one selected row for a different, genuinely committed row with a
    # valid path: the verifier's own selection must reject it.
    from zkllm.crypto.merkle import MerkleTree

    tree = MerkleTree.from_leaf_hashes(list(stream.leaf_hashes))
    unselected = next(
        index
        for index in range(stream.row_count)
        if index not in {item.row_index for item in honest}
    )
    path = tree.get_path(unselected)
    forged = list(honest[:-1]) + [
        AnchorRowOpening(
            stage_id=STAGE,
            row_index=unselected,
            row_bytes=_row(unselected),
            sibling_hashes=tuple(
                sibling for sibling, _is_left in path.siblings
            ),
        )
    ]
    with pytest.raises(ProofV3VerificationError, match="beacon-selected"):
        verify_anchor_row_openings(
            beacon=BEACON,
            commitment=stream.commitment,
            openings=[item.to_dict() for item in forged],
        )


def test_missing_replayed_row_fails_closed() -> None:
    stream = _stream(500)
    rows = _rows_for(stream)
    rows.pop(next(iter(rows)))
    with pytest.raises(ProofV3Error, match="missing replayed row"):
        build_anchor_row_openings(beacon=BEACON, stream=stream, rows=rows)


def test_foreign_stage_opening_is_rejected() -> None:
    stream = _stream(64)
    openings = [
        item.to_dict()
        for item in build_anchor_row_openings(
            beacon=BEACON, stream=stream, rows=_rows_for(stream)
        )
    ]
    openings[0]["stage_id"] = "blk.9.other.weight:dst"
    with pytest.raises(ProofV3VerificationError, match="different stage"):
        verify_anchor_row_openings(
            beacon=BEACON, commitment=stream.commitment, openings=openings
        )


ENTRY_HASH = "3c" * 32


def test_entry_selection_is_deterministic_and_covers_the_ends() -> None:
    from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_entry

    for row_count in (1, 2, 5, 64, 4096, 26214):
        selected = select_anchor_audit_rows_for_entry(
            beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=row_count
        )
        assert selected == select_anchor_audit_rows_for_entry(
            beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=row_count
        )
        assert len(selected) == min(ANCHOR_AUDIT_ROWS_PER_STAGE, row_count)
        assert len(set(selected)) == len(selected)
        assert all(0 <= index < row_count for index in selected)
        assert 0 in selected
        if row_count > 1:
            assert row_count - 1 in selected


def test_entry_selection_does_not_depend_on_prover_produced_state() -> None:
    """The grinding defence: only pre-nonce state plus the nonce feed it.

    The mesh audit builds its anchor streams during the post-nonce replay,
    so if the draw keyed on those commitments the prover could re-run the
    replay, perturbing rows it does not intend to open, until the draw
    moved somewhere convenient. Keying on the frozen op-manifest entry
    makes every such attempt produce the identical row set.
    """

    from verallm.mesh.anchor_audit import (
        select_anchor_audit_rows_for_entry,
        select_anchor_audit_rows_for_op,
    )

    baseline = select_anchor_audit_rows_for_entry(
        beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=500
    )
    # Two different stream contents for the same op: the entry-seeded draw
    # is identical, while the commitment-seeded draw moves.
    first = _stream(500)
    tampered = AnchorStageStream(
        stage_id=first.stage_id,
        row_width=first.row_width,
        leaf_hashes=first.leaf_hashes,
        commitment=ExecutionAnchorCommitmentV3(
            stage_id=first.stage_id,
            row_count=first.commitment.row_count,
            row_width=first.commitment.row_width,
            root=bytes([9]) * 32,
        ),
    )
    assert baseline == select_anchor_audit_rows_for_entry(
        beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=500
    )
    assert select_anchor_audit_rows_for_op(
        beacon=BEACON,
        src1_commitment=first.commitment,
        dst_commitment=first.commitment,
    ) != select_anchor_audit_rows_for_op(
        beacon=BEACON,
        src1_commitment=tampered.commitment,
        dst_commitment=tampered.commitment,
    )


def test_entry_selection_binds_beacon_entry_and_row_count() -> None:
    from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_entry

    base = select_anchor_audit_rows_for_entry(
        beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=500
    )
    assert base != select_anchor_audit_rows_for_entry(
        beacon="cd" * 32, op_manifest_entry_hash=ENTRY_HASH, row_count=500
    )
    assert base != select_anchor_audit_rows_for_entry(
        beacon=BEACON, op_manifest_entry_hash="7f" * 32, row_count=500
    )
    # A truncated stream cannot reuse the full-length draw.
    assert base != select_anchor_audit_rows_for_entry(
        beacon=BEACON, op_manifest_entry_hash=ENTRY_HASH, row_count=499
    )


def test_entry_selection_rejects_malformed_inputs() -> None:
    from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_entry

    with pytest.raises(ProofV3Error, match="entry hash"):
        select_anchor_audit_rows_for_entry(
            beacon=BEACON, op_manifest_entry_hash="", row_count=8
        )
    with pytest.raises(ProofV3Error, match="must be hex"):
        select_anchor_audit_rows_for_entry(
            beacon=BEACON, op_manifest_entry_hash="zz", row_count=8
        )
    with pytest.raises(ProofV3Error, match="beacon must be 32 bytes"):
        select_anchor_audit_rows_for_entry(
            beacon="ab", op_manifest_entry_hash=ENTRY_HASH, row_count=8
        )
