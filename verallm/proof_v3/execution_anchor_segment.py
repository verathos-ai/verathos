"""Compact Merkle summaries for skipped execution-anchor decode ranges.

The signed anchor root remains the security boundary.  These helpers retain
only canonical roots of complete dyadic intervals.  A prover can later combine
them with independently replayed prompt and suffix leaves to reconstruct the
same root and authentication paths without retaining every decode leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from verallm.proof_v3.errors import ProofV3Error
from zkllm.crypto.merkle import MerkleTree, hash_node
from zkllm.types import MerklePath


@dataclass(frozen=True, slots=True)
class ExecutionAnchorRangeNodeV3:
    """One complete dyadic subtree in level-local coordinates."""

    level: int
    index: int
    digest: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
            or self.level >= 32
            or isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
            or self.index >= 1 << 32
            or not isinstance(self.digest, bytes)
            or len(self.digest) != 32
        ):
            raise ProofV3Error("execution-anchor range node is malformed")


@dataclass(frozen=True, slots=True)
class ExecutionAnchorRangeSummaryV3:
    """Canonical retained nodes for one stage and absolute row interval."""

    stage_id: str
    start: int
    end: int
    nodes: tuple[ExecutionAnchorRangeNodeV3, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_.:/-]{0,95}", self.stage_id)
            is None
            or isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
            or not 0 <= self.start < self.end < 1 << 32
            or not isinstance(self.nodes, tuple)
            or any(
                not isinstance(node, ExecutionAnchorRangeNodeV3)
                for node in self.nodes
            )
            or tuple((node.level, node.index) for node in self.nodes)
            != canonical_execution_anchor_range_cover_v3(
                num_leaves=self.end,
                start=self.start,
                end=self.end,
            )
        ):
            raise ProofV3Error("execution-anchor range summary is malformed")

    @property
    def retained_bytes(self) -> int:
        return len(self.nodes) * 40


@dataclass(frozen=True, slots=True)
class ExecutionAnchorLeafRangeV3:
    """One contiguous replayed run of outer Merkle leaf hashes."""

    start: int
    leaf_hashes: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
            or self.start >= 1 << 32
            or not isinstance(self.leaf_hashes, bytes)
            or not self.leaf_hashes
            or len(self.leaf_hashes) % 32
            or self.start + self.row_count >= 1 << 32
        ):
            raise ProofV3Error(
                "execution-anchor replay leaf range is malformed"
            )

    @property
    def row_count(self) -> int:
        return len(self.leaf_hashes) // 32


@dataclass(frozen=True, slots=True)
class ExecutionAnchorRetainedLaneRangeV3:
    """Retained lane material for one independently replayed row range."""

    start: int
    replay_stage: object

    def __post_init__(self) -> None:
        from verallm.proof_v3.execution_anchor import (
            ExecutionAnchorReplayStageV3,
        )

        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
            or self.start >= 1 << 32
            or not isinstance(
                self.replay_stage,
                ExecutionAnchorReplayStageV3,
            )
            or not self.replay_stage.retained_lane_indices
            or self.start + self.row_count >= 1 << 32
        ):
            raise ProofV3Error(
                "execution-anchor retained lane range is malformed"
            )

    @property
    def row_count(self) -> int:
        return int(self.replay_stage.row_count)

    @property
    def retained_bytes(self) -> int:
        return len(self.replay_stage.retained_lane_hashes) + len(
            self.replay_stage.retained_lane_values
        )


@dataclass(frozen=True, slots=True)
class SparseExecutionAnchorReplayStageV3:
    """Prompt/suffix replay plus compact roots for skipped rows.

    This is prover-local rebuild material. It changes neither the frozen
    commitment nor the wire opening: the ordinary reveal builder reconstructs
    the exact original root and authentication paths before emitting anything.
    """

    stage_id: str
    row_count: int
    row_width: int
    leaf_ranges: tuple[ExecutionAnchorLeafRangeV3, ...]
    retained_nodes: tuple[ExecutionAnchorRangeNodeV3, ...]
    selected_rows: tuple[tuple[int, bytes], ...]
    retained_lane_ranges: tuple[ExecutionAnchorRetainedLaneRangeV3, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_.:/-]{0,95}", self.stage_id)
            is None
            or isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or not 0 < self.row_count < 1 << 32
            or isinstance(self.row_width, bool)
            or not isinstance(self.row_width, int)
            or not 0 < self.row_width < 1 << 24
            or not isinstance(self.leaf_ranges, tuple)
            or not self.leaf_ranges
            or any(
                not isinstance(item, ExecutionAnchorLeafRangeV3)
                for item in self.leaf_ranges
            )
            or not isinstance(self.retained_nodes, tuple)
            or any(
                not isinstance(item, ExecutionAnchorRangeNodeV3)
                for item in self.retained_nodes
            )
            or not isinstance(self.selected_rows, tuple)
            or not isinstance(self.retained_lane_ranges, tuple)
            or any(
                not isinstance(item, ExecutionAnchorRetainedLaneRangeV3)
                for item in self.retained_lane_ranges
            )
        ):
            raise ProofV3Error(
                "sparse execution-anchor replay stage is malformed"
            )
        ranges = tuple(
            (item.start, item.start + item.row_count)
            for item in self.leaf_ranges
        )
        if (
            ranges != tuple(sorted(ranges))
            or any(end > self.row_count for _start, end in ranges)
            or any(
                left_end > right_start
                for (_left_start, left_end), (right_start, _right_end)
                in zip(ranges, ranges[1:])
            )
        ):
            raise ProofV3Error(
                "sparse execution-anchor replay leaf ranges overlap"
            )
        rows = tuple(self.selected_rows)
        indices = tuple(index for index, _row in rows)
        if (
            indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.row_count
                or not isinstance(row, bytes)
                or len(row) != self.row_width
                for index, row in rows
            )
        ):
            raise ProofV3Error(
                "sparse execution-anchor selected rows are malformed"
            )
        lane_ranges = tuple(self.retained_lane_ranges)
        lane_intervals = tuple(
            (item.start, item.start + item.row_count)
            for item in lane_ranges
        )
        if (
            lane_intervals != tuple(sorted(lane_intervals))
            or any(end > self.row_count for _start, end in lane_intervals)
            or any(
                left_end > right_start
                for (_left_start, left_end), (right_start, _right_end)
                in zip(lane_intervals, lane_intervals[1:])
            )
            or any(
                item.replay_stage.stage_id != self.stage_id
                or item.replay_stage.row_width != self.row_width
                or (item.start, item.start + item.row_count) not in ranges
                for item in lane_ranges
            )
        ):
            raise ProofV3Error(
                "sparse execution-anchor retained lane ranges are malformed"
            )

    def visible_leaf_hashes(self) -> dict[int, bytes]:
        result: dict[int, bytes] = {}
        for item in self.leaf_ranges:
            for offset in range(item.row_count):
                index = item.start + offset
                result[index] = item.leaf_hashes[
                    offset * 32 : (offset + 1) * 32
                ]
        return result


def compose_segmented_execution_anchor_replay_stage_v3(
    *,
    stage_id: str,
    row_count: int,
    row_width: int,
    replay_segments: tuple[tuple[int, object], ...],
    retained_summaries: tuple[ExecutionAnchorRangeSummaryV3, ...],
) -> SparseExecutionAnchorReplayStageV3:
    """Compose independently replayed runs with pre-nonce range summaries."""

    from verallm.proof_v3.execution_anchor import (
        ExecutionAnchorReplayStageV3,
    )

    if (
        not isinstance(replay_segments, tuple)
        or not replay_segments
        or not isinstance(retained_summaries, tuple)
    ):
        raise ProofV3Error(
            "segmented execution-anchor replay inventory is malformed"
        )
    leaf_ranges = []
    lane_ranges = []
    selected_rows = []
    intervals = []
    for start, segment in replay_segments:
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or not isinstance(segment, ExecutionAnchorReplayStageV3)
            or segment.stage_id != stage_id
            or segment.row_width != row_width
            or start + segment.row_count > row_count
        ):
            raise ProofV3Error(
                "segmented execution-anchor replay segment changed"
            )
        leaf_ranges.append(
            ExecutionAnchorLeafRangeV3(
                start=start,
                leaf_hashes=segment.leaf_hashes,
            )
        )
        if segment.retained_lane_indices:
            lane_ranges.append(
                ExecutionAnchorRetainedLaneRangeV3(
                    start=start,
                    replay_stage=segment,
                )
            )
        selected_rows.extend(
            (start + index, row) for index, row in segment.selected_rows
        )
        intervals.append((start, start + segment.row_count))
    retained_nodes = []
    for summary in retained_summaries:
        if (
            not isinstance(summary, ExecutionAnchorRangeSummaryV3)
            or summary.stage_id != stage_id
            or summary.end > row_count
        ):
            raise ProofV3Error(
                "segmented execution-anchor retained range changed"
            )
        intervals.append((summary.start, summary.end))
        retained_nodes.extend(summary.nodes)
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise ProofV3Error(
                "segmented execution-anchor ranges are incomplete or overlap"
            )
        cursor = end
    if cursor != row_count:
        raise ProofV3Error(
            "segmented execution-anchor ranges are incomplete or overlap"
        )
    selected_rows.sort(key=lambda item: item[0])
    if len({index for index, _row in selected_rows}) != len(selected_rows):
        raise ProofV3Error(
            "segmented execution-anchor selected rows overlap"
        )
    return SparseExecutionAnchorReplayStageV3(
        stage_id=stage_id,
        row_count=row_count,
        row_width=row_width,
        leaf_ranges=tuple(sorted(leaf_ranges, key=lambda item: item.start)),
        retained_nodes=tuple(retained_nodes),
        selected_rows=tuple(selected_rows),
        retained_lane_ranges=tuple(lane_ranges),
    )


class StreamingExecutionAnchorRangeV3:
    """Reduce one absolute leaf interval into its canonical dyadic cover.

    Unlike a normal Merkle frontier, this accumulator never merges across the
    interval's left boundary.  Its retained nodes are therefore valid nodes of
    the original full-sequence tree even when ``start`` is not power-of-two
    aligned.  The state is O(log(end - start)) and can be mirrored by the
    isolated CUDA capture path without retaining per-row hashes.
    """

    __slots__ = ("_start", "_cursor", "_nodes")

    def __init__(self, *, start: int) -> None:
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or not 0 <= start < 1 << 32
        ):
            raise ProofV3Error("execution-anchor range start is malformed")
        self._start = start
        self._cursor = start
        self._nodes: list[ExecutionAnchorRangeNodeV3] = []

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._cursor

    def append_leaf_hash(self, leaf_hash: bytes) -> None:
        if (
            not isinstance(leaf_hash, bytes)
            or len(leaf_hash) != 32
            or self._cursor >= (1 << 32) - 1
        ):
            raise ProofV3Error("execution-anchor range leaf is malformed")
        node = ExecutionAnchorRangeNodeV3(
            level=0,
            index=self._cursor,
            digest=leaf_hash,
        )
        self._cursor += 1
        while self._nodes:
            left = self._nodes[-1]
            if (
                left.level != node.level
                or left.index % 2
                or node.index != left.index + 1
            ):
                break
            self._nodes.pop()
            node = ExecutionAnchorRangeNodeV3(
                level=left.level + 1,
                index=left.index // 2,
                digest=hash_node(left.digest, node.digest),
            )
        self._nodes.append(node)

    def summary(self) -> tuple[ExecutionAnchorRangeNodeV3, ...]:
        if self._cursor == self._start:
            raise ProofV3Error("execution-anchor range is empty")
        expected = canonical_execution_anchor_range_cover_v3(
            num_leaves=self._cursor,
            start=self._start,
            end=self._cursor,
        )
        observed = tuple((node.level, node.index) for node in self._nodes)
        if observed != expected:
            raise ProofV3Error(
                "execution-anchor streaming range cover is inconsistent"
            )
        return tuple(self._nodes)


def canonical_execution_anchor_range_cover_v3(
    *,
    num_leaves: int,
    start: int,
    end: int,
) -> tuple[tuple[int, int], ...]:
    """Return the unique greedy dyadic cover of ``[start, end)``."""

    if (
        isinstance(num_leaves, bool)
        or not isinstance(num_leaves, int)
        or num_leaves <= 0
        or num_leaves >= 1 << 32
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= num_leaves
    ):
        raise ProofV3Error("execution-anchor range is malformed")
    cursor = start
    cover = []
    while cursor < end:
        remaining = end - cursor
        if cursor == 0:
            size = 1 << (remaining.bit_length() - 1)
        else:
            size = cursor & -cursor
            while size > remaining:
                size >>= 1
        level = size.bit_length() - 1
        cover.append((level, cursor >> level))
        cursor += size
    return tuple(cover)


def build_execution_anchor_range_summary_v3(
    tree: MerkleTree,
    *,
    start: int,
    end: int,
) -> tuple[ExecutionAnchorRangeNodeV3, ...]:
    """Extract a compact canonical summary from an already-built tree."""

    if not isinstance(tree, MerkleTree):
        raise ProofV3Error("execution-anchor range tree is malformed")
    return tuple(
        ExecutionAnchorRangeNodeV3(
            level=level,
            index=index,
            digest=tree.get_node_hash(level, index),
        )
        for level, index in canonical_execution_anchor_range_cover_v3(
            num_leaves=tree.num_leaves,
            start=start,
            end=end,
        )
    )


def reconstruct_execution_anchor_sparse_paths_v3(
    *,
    num_leaves: int,
    leaf_hashes: Mapping[int, bytes],
    retained_nodes: tuple[ExecutionAnchorRangeNodeV3, ...],
    opening_indices: tuple[int, ...],
) -> tuple[bytes, tuple[MerklePath, ...]]:
    """Rebuild one root and selected paths from disjoint complete ranges."""

    if (
        isinstance(num_leaves, bool)
        or not isinstance(num_leaves, int)
        or num_leaves <= 0
        or num_leaves >= 1 << 32
        or not isinstance(leaf_hashes, Mapping)
        or not isinstance(retained_nodes, tuple)
        or not isinstance(opening_indices, tuple)
    ):
        raise ProofV3Error("sparse execution-anchor material is malformed")
    try:
        leaves = {int(index): bytes(digest) for index, digest in leaf_hashes.items()}
        openings = tuple(int(index) for index in opening_indices)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("sparse execution-anchor material is malformed") from exc
    if (
        any(not 0 <= index < num_leaves for index in leaves)
        or any(len(digest) != 32 for digest in leaves.values())
        or openings != tuple(sorted(set(openings)))
        or any(index not in leaves for index in openings)
        or any(not isinstance(node, ExecutionAnchorRangeNodeV3)
               for node in retained_nodes)
    ):
        raise ProofV3Error("sparse execution-anchor material is malformed")

    nodes = {(0, index): digest for index, digest in leaves.items()}
    intervals = [(index, index + 1) for index in leaves]
    for node in retained_nodes:
        size = 1 << node.level
        start = node.index * size
        end = start + size
        if end > num_leaves or (node.level, node.index) in nodes:
            raise ProofV3Error("sparse execution-anchor ranges overlap")
        nodes[(node.level, node.index)] = node.digest
        intervals.append((start, end))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise ProofV3Error(
                "sparse execution-anchor ranges are incomplete or overlap"
            )
        cursor = end
    if cursor != num_leaves:
        raise ProofV3Error("sparse execution-anchor ranges are incomplete")

    level = 0
    level_size = num_leaves
    while level_size > 1:
        parent_size = (level_size + 1) // 2
        for parent in range(parent_size):
            coordinate = (level + 1, parent)
            if coordinate in nodes:
                continue
            left = nodes.get((level, parent * 2))
            right_index = parent * 2 + 1
            right = (
                left
                if right_index >= level_size
                else nodes.get((level, right_index))
            )
            if left is not None and right is not None:
                nodes[coordinate] = hash_node(left, right)
        level += 1
        level_size = parent_size
    root = nodes.get((level, 0))
    if root is None:
        raise ProofV3Error("sparse execution-anchor root is incomplete")

    paths = []
    for opening in openings:
        siblings = []
        index = opening
        level_size = num_leaves
        for current_level in range(level):
            sibling_index = index - 1 if index % 2 else index + 1
            if sibling_index >= level_size:
                sibling_index = index
            sibling = nodes.get((current_level, sibling_index))
            if sibling is None:
                raise ProofV3Error(
                    "sparse execution-anchor opening path is incomplete"
                )
            siblings.append((sibling, bool(index % 2)))
            index //= 2
            level_size = (level_size + 1) // 2
        paths.append(MerklePath(leaf_index=opening, siblings=siblings))
    return root, tuple(paths)


__all__ = [
    "ExecutionAnchorLeafRangeV3",
    "ExecutionAnchorRangeNodeV3",
    "ExecutionAnchorRangeSummaryV3",
    "ExecutionAnchorRetainedLaneRangeV3",
    "SparseExecutionAnchorReplayStageV3",
    "StreamingExecutionAnchorRangeV3",
    "build_execution_anchor_range_summary_v3",
    "canonical_execution_anchor_range_cover_v3",
    "compose_segmented_execution_anchor_replay_stage_v3",
    "reconstruct_execution_anchor_sparse_paths_v3",
]
