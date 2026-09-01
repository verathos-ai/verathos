"""Loading runtime anchor artifacts fails closed on any disagreement.

The runtime writes its own leaf stream and its own committed root. These
tests pin that the loader never trusts either alone: a truncated stream, a
row-count disagreement, a tampered leaf, or a substituted root must all
raise rather than yield an anchor an audit would then open against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verallm.mesh.anchor_streams import (
    ANCHOR_SUMMARY_ROW,
    anchor_inventory_digest,
    load_anchor_row_dumps,
    load_anchor_streams,
)
from verallm.mesh.execution_anchor import (
    ProofV3Error,
    StreamingExecutionAnchorV3,
    execution_anchor_row_leaf_hash_v3,
)

STAGE = "blk.0.ffn_gate.weight:dst"
WIDTH = 4096
ROWS = 9
TOKEN = "capture-token-1"


def _row(index: int) -> bytes:
    return bytes((index * 37 + offset) % 251 for offset in range(WIDTH))


def _write_capture(
    directory: Path,
    *,
    rows: int = ROWS,
    stage: str = STAGE,
    token: str = TOKEN,
) -> tuple[Path, str]:
    anchor = StreamingExecutionAnchorV3(stage_id=stage, row_width=WIDTH)
    leaves = []
    for index in range(rows):
        leaf = execution_anchor_row_leaf_hash_v3(
            stage_id=stage,
            row_index=index,
            row_width=WIDTH,
            row_bytes=_row(index),
        )
        leaves.append(leaf)
        anchor.append_leaf_hash(leaf)
    root = anchor.root.hex()
    safe = "".join(
        character
        if character.isalnum() or character in {".", "_", "-"}
        else "_"
        for character in stage
    )
    stream_path = directory / f"anchor-1-{safe}.vleafs"
    stream_path.write_bytes(b"".join(leaves))
    summary_path = directory / "anchor-1.json"
    summary_path.write_text(
        json.dumps(
            {
                "row": ANCHOR_SUMMARY_ROW,
                "created_unix_ns": 1,
                "capture_token": token,
                "stages": [
                    {
                        "stage": stage,
                        "rows": rows,
                        "row_width": WIDTH,
                        "root": root,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return stream_path, root


def test_round_trip_loads_and_rebuilds(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    streams = load_anchor_streams(tmp_path)
    assert set(streams) == {STAGE}
    stream = streams[STAGE]
    assert stream.row_count == ROWS
    assert stream.row_width == WIDTH
    assert stream.commitment.row_count == ROWS
    assert anchor_inventory_digest(streams)


def test_capture_token_filter_selects_the_window(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    assert load_anchor_streams(tmp_path, capture_token=TOKEN)
    assert load_anchor_streams(tmp_path, capture_token="other") == {}


def test_missing_leaf_stream_fails_closed(tmp_path: Path) -> None:
    stream_path, _root = _write_capture(tmp_path)
    stream_path.unlink()
    with pytest.raises(ProofV3Error, match="missing"):
        load_anchor_streams(tmp_path)


def test_truncated_leaf_stream_fails_closed(tmp_path: Path) -> None:
    stream_path, _root = _write_capture(tmp_path)
    raw = stream_path.read_bytes()
    stream_path.write_bytes(raw[:-5])
    with pytest.raises(ProofV3Error, match="truncated"):
        load_anchor_streams(tmp_path)


def test_row_count_disagreement_fails_closed(tmp_path: Path) -> None:
    stream_path, _root = _write_capture(tmp_path)
    raw = stream_path.read_bytes()
    stream_path.write_bytes(raw[:-32])
    with pytest.raises(ProofV3Error, match="rows"):
        load_anchor_streams(tmp_path)


def test_tampered_leaf_fails_the_root_rebuild(tmp_path: Path) -> None:
    stream_path, _root = _write_capture(tmp_path)
    raw = bytearray(stream_path.read_bytes())
    raw[0] ^= 1
    stream_path.write_bytes(bytes(raw))
    with pytest.raises(ProofV3Error, match="rebuild"):
        load_anchor_streams(tmp_path)


def test_substituted_root_fails_the_rebuild(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    summary_path = tmp_path / "anchor-1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stages"][0]["root"] = "11" * 32
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ProofV3Error, match="rebuild"):
        load_anchor_streams(tmp_path)


def test_partial_summary_write_is_ignored(tmp_path: Path) -> None:
    # The runtime writes to anchor-N.json.tmp then renames; a reader that
    # picked up the temp file could observe half a JSON object.
    _write_capture(tmp_path)
    (tmp_path / "anchor-1.json.tmp").write_text("{\"row\":", encoding="utf-8")
    assert set(load_anchor_streams(tmp_path)) == {STAGE}


def test_no_anchor_artifacts_is_not_an_error(tmp_path: Path) -> None:
    assert load_anchor_streams(tmp_path) == {}
    assert anchor_inventory_digest({}) == ""


def _row_record(index: int, width: int = WIDTH) -> bytes:
    return (
        index.to_bytes(8, "little")
        + width.to_bytes(4, "little")
        + _row(index)
    )


def _write_row_dump(
    directory: Path, indexes: list[int], *, stage: str = STAGE
) -> Path:
    safe = "".join(
        character
        if character.isalnum() or character in {".", "_", "-"}
        else "_"
        for character in stage
    )
    path = directory / f"anchor-1-{safe}.vrows"
    path.write_bytes(b"".join(_row_record(index) for index in indexes))
    return path


def test_row_dump_round_trip_leaf_verifies(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    _write_row_dump(tmp_path, [0, 5, 8])
    streams = load_anchor_streams(tmp_path)
    dumps = load_anchor_row_dumps(tmp_path, streams)
    assert set(dumps) == {STAGE}
    assert sorted(dumps[STAGE]) == [0, 5, 8]
    for index, row_bytes in dumps[STAGE].items():
        assert row_bytes == _row(index)


def test_missing_row_dump_is_not_an_error(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    streams = load_anchor_streams(tmp_path)
    assert load_anchor_row_dumps(tmp_path, streams) == {}


def test_row_dump_capture_token_filter(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    _write_row_dump(tmp_path, [0])
    streams = load_anchor_streams(tmp_path, capture_token=TOKEN)
    assert load_anchor_row_dumps(
        tmp_path, streams, capture_token=TOKEN
    )
    assert load_anchor_row_dumps(
        tmp_path, streams, capture_token="other"
    ) == {}


def test_tampered_row_dump_bytes_fail_the_leaf(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    path = _write_row_dump(tmp_path, [0, 5])
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="committed leaf"):
        load_anchor_row_dumps(tmp_path, streams)


def test_row_dump_substituted_index_fails_the_leaf(tmp_path: Path) -> None:
    # Row 5's bytes filed under index 3: the leaf hash binds the index, so
    # the dump cannot relabel a genuinely computed row.
    _write_capture(tmp_path)
    record = (
        (3).to_bytes(8, "little")
        + WIDTH.to_bytes(4, "little")
        + _row(5)
    )
    safe = STAGE.replace(":", "_")
    (tmp_path / f"anchor-1-{safe}.vrows").write_bytes(record)
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="committed leaf"):
        load_anchor_row_dumps(tmp_path, streams)


def test_row_dump_truncated_record_fails_closed(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    path = _write_row_dump(tmp_path, [0])
    path.write_bytes(path.read_bytes()[:-7])
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="truncated"):
        load_anchor_row_dumps(tmp_path, streams)


def test_row_dump_wrong_width_fails_closed(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    record = (
        (0).to_bytes(8, "little")
        + (WIDTH - 4).to_bytes(4, "little")
        + _row(0)[: WIDTH - 4]
    )
    safe = STAGE.replace(":", "_")
    (tmp_path / f"anchor-1-{safe}.vrows").write_bytes(record)
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="width"):
        load_anchor_row_dumps(tmp_path, streams)


def test_row_dump_out_of_range_index_fails_closed(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    record = (
        ROWS.to_bytes(8, "little")
        + WIDTH.to_bytes(4, "little")
        + _row(0)
    )
    safe = STAGE.replace(":", "_")
    (tmp_path / f"anchor-1-{safe}.vrows").write_bytes(record)
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="out of range"):
        load_anchor_row_dumps(tmp_path, streams)


def test_row_dump_duplicate_index_fails_closed(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    _write_row_dump(tmp_path, [4, 4])
    streams = load_anchor_streams(tmp_path)
    with pytest.raises(ProofV3Error, match="duplicated"):
        load_anchor_row_dumps(tmp_path, streams)


def test_capture_token_prefix_scopes_one_window(tmp_path: Path) -> None:
    # Full tokens append per-member selection markers to a shared minted
    # base; prefix scoping must select the window without knowing them.
    _write_capture(tmp_path, token="1234|selected_v3=3:7|anchor_rows=0,8")
    assert load_anchor_streams(tmp_path, capture_token_prefix="1234")
    assert (
        load_anchor_streams(tmp_path, capture_token_prefix="9999") == {}
    )
    streams = load_anchor_streams(tmp_path, capture_token_prefix="1234")
    _write_row_dump(tmp_path, [0, 8])
    assert load_anchor_row_dumps(
        tmp_path, streams, capture_token_prefix="1234"
    )
    assert (
        load_anchor_row_dumps(
            tmp_path, streams, capture_token_prefix="9999"
        )
        == {}
    )
