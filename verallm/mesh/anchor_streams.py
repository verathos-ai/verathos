"""Read and validate the runtime's streaming execution-anchor artifacts.

The patched llama.cpp runtime writes, per capture window:

  anchor-<digits>-<stage>.vleafs   raw 32-byte row leaves, in row order
  anchor-<digits>.json             {"stages": [{stage, rows, row_width, root}]}
  anchor-<digits>-<stage>.vrows    bounded selected-row dump, only when the
                                   capture token carries anchor_rows=; each
                                   record is u64 LE row index + u32 LE row
                                   width + the raw row bytes

The summary root is the runtime's own O(log n) frontier result; the leaf
stream is the material a bounded audit replays against.  Both are produced
by the same process, so neither is trusted alone: every read rebuilds the
Merkle root from the leaf stream and requires it to equal the summary root,
and requires the row count and width to agree.  A miner that streams one
set of leaves and commits a different root fails here rather than later in
an opening.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from verallm.mesh.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ProofV3Error,
    execution_anchor_inventory_digest_v3,
    execution_anchor_row_leaf_hash_v3,
)
from zkllm.crypto.merkle import MerkleTree

ANCHOR_SUMMARY_ROW = "VERATHOS_GGML_ANCHOR_V1"
_LEAF_BYTES = 32


@dataclass(frozen=True)
class AnchorStageStream:
    """One stage's verified leaf stream plus its commitment."""

    stage_id: str
    row_width: int
    leaf_hashes: tuple[bytes, ...]
    commitment: ExecutionAnchorCommitmentV3

    @property
    def row_count(self) -> int:
        return len(self.leaf_hashes)


def _summary_paths(trace_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in trace_dir.glob("anchor*.json")
        if not path.name.endswith(".tmp")
    )


def _stage_file_path(
    summary_path: Path, stage_id: str, extension: str
) -> Path:
    safe = "".join(
        character
        if (
            character.isalnum()
            or character in {".", "_", "-"}
        )
        else "_"
        for character in stage_id
    )
    return summary_path.with_name(
        summary_path.stem + "-" + safe + extension
    )


def _leaf_stream_path(summary_path: Path, stage_id: str) -> Path:
    return _stage_file_path(summary_path, stage_id, ".vleafs")


def _capture_token_matches(
    summary_token: str, capture_token: str, capture_token_prefix: str
) -> bool:
    if capture_token and summary_token != capture_token:
        return False
    if capture_token_prefix and (
        summary_token.split("|", 1)[0] != capture_token_prefix
    ):
        # Full tokens append per-member selection markers to a shared
        # minted base; prefix scoping selects one capture window across
        # every member without knowing their local markers.
        return False
    return True


def load_anchor_streams(
    trace_dir: Path | str,
    *,
    capture_token: str = "",
    capture_token_prefix: str = "",
) -> dict[str, AnchorStageStream]:
    """Load every anchored stage for a capture window, fully re-verified.

    Raises ProofV3Error when a leaf stream is missing, truncated, or
    disagrees with the committed root: an anchor that cannot be rebuilt is
    unusable for an audit and must fail closed at load time.
    """

    directory = Path(trace_dir)
    summaries = _summary_paths(directory)
    if not summaries:
        return {}
    streams: dict[str, AnchorStageStream] = {}
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProofV3Error(
                f"anchor summary {summary_path.name} is unreadable"
            ) from exc
        if not isinstance(summary, Mapping):
            raise ProofV3Error("anchor summary must be an object")
        if str(summary.get("row", "")) != ANCHOR_SUMMARY_ROW:
            raise ProofV3Error("anchor summary row marker is unsupported")
        if not _capture_token_matches(
            str(summary.get("capture_token", "")),
            capture_token,
            capture_token_prefix,
        ):
            continue
        stages = summary.get("stages")
        if not isinstance(stages, list):
            raise ProofV3Error("anchor summary stages must be a list")
        for entry in stages:
            stream = _load_stage(summary_path, entry)
            if stream.stage_id in streams:
                raise ProofV3Error(
                    f"anchor stage {stream.stage_id} is duplicated"
                )
            streams[stream.stage_id] = stream
    return streams


def _load_stage(
    summary_path: Path, entry: Any
) -> AnchorStageStream:
    if not isinstance(entry, Mapping):
        raise ProofV3Error("anchor stage entry must be an object")
    stage_id = str(entry.get("stage", ""))
    rows = int(entry.get("rows", 0) or 0)
    row_width = int(entry.get("row_width", 0) or 0)
    root_hex = str(entry.get("root", ""))
    if not stage_id or rows <= 0 or row_width <= 0:
        raise ProofV3Error("anchor stage geometry is malformed")
    try:
        root = bytes.fromhex(root_hex)
    except ValueError as exc:
        raise ProofV3Error("anchor stage root must be hex") from exc
    if len(root) != 32:
        raise ProofV3Error("anchor stage root must be 32 bytes")

    path = _leaf_stream_path(summary_path, stage_id)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProofV3Error(
            f"anchor leaf stream for {stage_id} is missing"
        ) from exc
    if len(raw) % _LEAF_BYTES:
        raise ProofV3Error(
            f"anchor leaf stream for {stage_id} is truncated"
        )
    leaves = tuple(
        raw[offset : offset + _LEAF_BYTES]
        for offset in range(0, len(raw), _LEAF_BYTES)
    )
    if len(leaves) != rows:
        raise ProofV3Error(
            f"anchor leaf stream for {stage_id} has {len(leaves)} rows "
            f"but the summary committed {rows}"
        )
    rebuilt = MerkleTree.from_leaf_hashes(list(leaves)).root
    if rebuilt != root:
        raise ProofV3Error(
            f"anchor leaf stream for {stage_id} does not rebuild its "
            "committed root"
        )
    return AnchorStageStream(
        stage_id=stage_id,
        row_width=row_width,
        leaf_hashes=leaves,
        commitment=ExecutionAnchorCommitmentV3(
            stage_id=stage_id,
            row_count=rows,
            row_width=row_width,
            root=root,
        ),
    )


_ROW_DUMP_HEADER_BYTES = 12


def load_anchor_row_dumps(
    trace_dir: Path | str,
    streams: Mapping[str, AnchorStageStream],
    *,
    capture_token: str = "",
    capture_token_prefix: str = "",
) -> dict[str, dict[int, bytes]]:
    """Load the bounded selected-row dumps, leaf-verified against streams.

    Every dumped row must hash to the leaf the verified stream committed at
    that index, so the dump can never smuggle bytes the serve did not
    anchor.  Returns {stage_id: {row_index: row_bytes}}; stages without a
    dump file are simply absent (the runtime writes .vrows only when the
    capture token armed a row selection).  Raises ProofV3Error on any
    malformed, duplicated, out-of-range, or leaf-mismatched record.
    """

    directory = Path(trace_dir)
    dumps: dict[str, dict[int, bytes]] = {}
    for summary_path in _summary_paths(directory):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # load_anchor_streams already rejects unreadable ones
        if not isinstance(summary, Mapping):
            continue
        if not _capture_token_matches(
            str(summary.get("capture_token", "")),
            capture_token,
            capture_token_prefix,
        ):
            continue
        for stage_id, stream in streams.items():
            path = _stage_file_path(summary_path, stage_id, ".vrows")
            if not path.exists():
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise ProofV3Error(
                    f"anchor row dump for {stage_id} is unreadable"
                ) from exc
            rows = dumps.setdefault(stage_id, {})
            offset = 0
            while offset < len(raw):
                if offset + _ROW_DUMP_HEADER_BYTES > len(raw):
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} is truncated"
                    )
                index = int.from_bytes(raw[offset : offset + 8], "little")
                width = int.from_bytes(
                    raw[offset + 8 : offset + 12], "little"
                )
                offset += _ROW_DUMP_HEADER_BYTES
                if width != stream.row_width:
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} row {index} has "
                        f"width {width}, stream committed {stream.row_width}"
                    )
                if index >= stream.row_count:
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} row {index} is "
                        "out of range"
                    )
                if index in rows:
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} row {index} is "
                        "duplicated"
                    )
                if offset + width > len(raw):
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} is truncated"
                    )
                row_bytes = raw[offset : offset + width]
                offset += width
                leaf = execution_anchor_row_leaf_hash_v3(
                    stage_id=stage_id,
                    row_index=index,
                    row_width=width,
                    row_bytes=row_bytes,
                )
                if leaf != stream.leaf_hashes[index]:
                    raise ProofV3Error(
                        f"anchor row dump for {stage_id} row {index} does "
                        "not hash to the committed leaf"
                    )
                rows[index] = row_bytes
    return dumps


def anchor_inventory_digest(
    streams: Mapping[str, AnchorStageStream],
) -> str:
    """Digest the ordered anchor inventory for the precommit gate hash.

    Returns "" when nothing was anchored, so a runtime without the anchor
    build contributes no gate-hash material and stays compatible.
    """

    if not streams:
        return ""
    commitments = tuple(
        streams[stage_id].commitment for stage_id in sorted(streams)
    )
    return execution_anchor_inventory_digest_v3(commitments).hex()
