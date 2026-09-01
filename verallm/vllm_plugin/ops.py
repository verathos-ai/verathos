"""
verallm custom ops for activation capture during CUDA graph execution.

verallm::capture — Graph split point op (runs in eager mode at split).
verallm::buffer_copy — Static-grid Triton copy for buffer mode on Ampere.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from verallm.miner.activation_tracker import RequestActivationTracker

logger = logging.getLogger(__name__)

# Module-level reference set by set_active_tracker().
# Read at each split point by the custom op.
_active_tracker: RequestActivationTracker | None = None


def set_active_tracker(tracker: RequestActivationTracker | None) -> None:
    """Set the tracker that receives captures at split points."""
    global _active_tracker
    _active_tracker = tracker


def get_active_tracker() -> RequestActivationTracker | None:
    """Get the currently active tracker (or None)."""
    return _active_tracker


# -- Triton static-grid buffer copy ------------------------------------------
#
# Replaces aten::copy_ for buffer-mode activation capture on Ampere GPUs
# (sm_80-86) where Inductor-generated dynamic-grid copy kernels cause
# CUDA illegal memory access during graph capture.
#
# The kernel always launches with a STATIC grid (sized for the full buffer)
# and uses a mask to copy only n_elements.  The dynamic element count is a
# scalar kernel argument captured by value in the CUDA graph — the grid
# never changes, so graph replay is safe on all architectures.

@triton.jit
def _buffer_copy_kernel(
    src_ptr, dst_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _buffer_gather_rows_kernel(
    src_ptr,
    dst_ptr,
    row_indices_ptr,
    src_rows,
    dst_rows,
    width,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_row = offsets // width
    column = offsets - dst_row * width
    row_in_range = dst_row < dst_rows
    src_row = tl.load(row_indices_ptr + dst_row, mask=row_in_range, other=-1)
    valid = row_in_range & (src_row >= 0) & (src_row < src_rows)
    values = tl.load(
        src_ptr + src_row * src_stride_row + column * src_stride_col,
        mask=valid,
        other=0.0,
    )
    tl.store(
        dst_ptr + dst_row * dst_stride_row + column * dst_stride_col,
        values,
        mask=valid,
    )


@triton.jit
def _buffer_scatter_root_history_kernel(
    src_ptr,
    dst_ptr,
    row_indices_ptr,
    active_rows,
    active_elements,
    src_stride_stage,
    src_stride_row,
    dst_stride_stage,
    dst_stride_row,
    dst_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    byte = offsets % 32
    stage_row = offsets // 32
    src_row = stage_row % active_rows
    stage = stage_row // active_rows
    in_range = offsets < active_elements
    dst_row = tl.load(row_indices_ptr + src_row, mask=in_range, other=-1)
    valid = in_range & (dst_row >= 0) & (dst_row < dst_rows)
    values = tl.load(
        src_ptr
        + stage * src_stride_stage
        + src_row * src_stride_row
        + byte,
        mask=valid,
        other=0,
    )
    tl.store(
        dst_ptr
        + stage * dst_stride_stage
        + dst_row * dst_stride_row
        + byte,
        values,
        mask=valid,
    )


@triton.jit
def _buffer_copy_rows_padded_kernel(
    src_ptr,
    dst_ptr,
    rows,
    width,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = rows * width
    row = offsets // width
    column = offsets - row * width
    valid = offsets < total
    values = tl.load(
        src_ptr + row * src_stride_row + column * src_stride_col,
        mask=valid,
        other=0.0,
    )
    tl.store(
        dst_ptr + row * dst_stride_row + column * dst_stride_col,
        values,
        mask=valid,
    )


@triton.jit
def _buffer_copy_three_rows_padded_kernel(
    src0_ptr,
    src1_ptr,
    src2_ptr,
    dst0_ptr,
    dst1_ptr,
    dst2_ptr,
    rows,
    width0,
    width1,
    width2,
    src0_stride_row,
    src1_stride_row,
    src2_stride_row,
    dst0_stride_row,
    dst1_stride_row,
    dst2_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row0 = offsets // width0
    column0 = offsets - row0 * width0
    valid0 = offsets < rows * width0
    values0 = tl.load(
        src0_ptr + row0 * src0_stride_row + column0,
        mask=valid0,
        other=0.0,
    )
    tl.store(
        dst0_ptr + row0 * dst0_stride_row + column0,
        values0,
        mask=valid0,
    )

    row1 = offsets // width1
    column1 = offsets - row1 * width1
    valid1 = offsets < rows * width1
    values1 = tl.load(
        src1_ptr + row1 * src1_stride_row + column1,
        mask=valid1,
        other=0.0,
    )
    tl.store(
        dst1_ptr + row1 * dst1_stride_row + column1,
        values1,
        mask=valid1,
    )

    row2 = offsets // width2
    column2 = offsets - row2 * width2
    valid2 = offsets < rows * width2
    values2 = tl.load(
        src2_ptr + row2 * src2_stride_row + column2,
        mask=valid2,
        other=0.0,
    )
    tl.store(
        dst2_ptr + row2 * dst2_stride_row + column2,
        values2,
        mask=valid2,
    )


@triton.jit
def _buffer_copy_input_router_sum_rows_padded_kernel(
    src0_ptr,
    src1_ptr,
    src2a_ptr,
    src2b_ptr,
    dst0_ptr,
    dst1_ptr,
    dst2_ptr,
    rows,
    width0,
    width1,
    width2,
    src0_stride_row,
    src1_stride_row,
    src2a_stride_row,
    src2b_stride_row,
    dst0_stride_row,
    dst1_stride_row,
    dst2_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage an input, router, and exact shared-plus-routed MoE output."""

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row0 = offsets // width0
    column0 = offsets - row0 * width0
    valid0 = offsets < rows * width0
    values0 = tl.load(
        src0_ptr + row0 * src0_stride_row + column0,
        mask=valid0,
        other=0.0,
    )
    tl.store(
        dst0_ptr + row0 * dst0_stride_row + column0,
        values0,
        mask=valid0,
    )

    row1 = offsets // width1
    column1 = offsets - row1 * width1
    valid1 = offsets < rows * width1
    values1 = tl.load(
        src1_ptr + row1 * src1_stride_row + column1,
        mask=valid1,
        other=0.0,
    )
    tl.store(
        dst1_ptr + row1 * dst1_stride_row + column1,
        values1,
        mask=valid1,
    )

    row2 = offsets // width2
    column2 = offsets - row2 * width2
    valid2 = offsets < rows * width2
    values2a = tl.load(
        src2a_ptr + row2 * src2a_stride_row + column2,
        mask=valid2,
        other=0.0,
    )
    values2b = tl.load(
        src2b_ptr + row2 * src2b_stride_row + column2,
        mask=valid2,
        other=0.0,
    )
    tl.store(
        dst2_ptr + row2 * dst2_stride_row + column2,
        values2a + values2b,
        mask=valid2,
    )


@triton.jit
def _buffer_copy_residual_boundaries_rows_padded_kernel(
    residual_ptr,
    hidden_ptr,
    output_residual_ptr,
    residual_dst_ptr,
    output_dst_ptr,
    rows,
    width,
    residual_stride_row,
    hidden_stride_row,
    output_residual_stride_row,
    residual_dst_stride_row,
    output_dst_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage the exact sparse residual boundaries in one launch."""

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = offsets // width
    column = offsets - row * width
    valid = offsets < rows * width
    residual = tl.load(
        residual_ptr + row * residual_stride_row + column,
        mask=valid,
        other=0.0,
    )
    hidden = tl.load(
        hidden_ptr + row * hidden_stride_row + column,
        mask=valid,
        other=0.0,
    )
    output_residual = tl.load(
        output_residual_ptr
        + row * output_residual_stride_row
        + column,
        mask=valid,
        other=0.0,
    )
    tl.store(
        residual_dst_ptr + row * residual_dst_stride_row + column,
        residual,
        mask=valid,
    )
    tl.store(
        output_dst_ptr + row * output_dst_stride_row + column,
        hidden + output_residual,
        mask=valid,
    )


@torch.library.triton_op("verallm::buffer_copy", mutates_args=("dst",))
def buffer_copy(dst: torch.Tensor, src: torch.Tensor, n_tokens: int) -> None:
    """Static-grid Triton copy: dst[:n_tokens] = src[:n_tokens].

    Grid is always sized for dst.numel() (constant per buffer).
    n_tokens is a scalar kernel arg captured by value in CUDA graphs.
    Safe on ALL GPU architectures including Ampere (sm_80-86).
    """
    hidden_dim = dst.shape[1]
    n_elements = n_tokens * hidden_dim
    BLOCK_SIZE = 1024
    grid = ((dst.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    torch.library.wrap_triton(_buffer_copy_kernel)[grid](
        src,
        dst,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@torch.library.triton_op(
    "verallm::activation_row_stage",
    mutates_args=("staging",),
)
def activation_row_stage(
    staging: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """Traceably stage bounded runtime rows for the shared root reducer."""

    block_size = 1024
    total = src.shape[0] * src.shape[1]
    grid = (triton.cdiv(total, block_size),)
    torch.library.wrap_triton(_buffer_copy_rows_padded_kernel)[grid](
        src,
        staging,
        src.shape[0],
        src.shape[1],
        src.stride(0),
        src.stride(1),
        staging.stride(0),
        staging.stride(1),
        BLOCK_SIZE=block_size,
    )


@torch.library.triton_op("verallm::buffer_gather_rows", mutates_args=("dst",))
def buffer_gather_rows(
    dst: torch.Tensor,
    src: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    """Gather one source row per scheduler slot into a fixed capture buffer."""

    if dst.ndim != 2 or src.ndim != 2:
        raise ValueError("capture row gather requires two-dimensional tensors")
    if dst.shape[1] != src.shape[1]:
        raise ValueError("capture row gather width mismatch")
    if row_indices.ndim != 1 or row_indices.shape[0] != dst.shape[0]:
        raise ValueError("capture row gather index shape mismatch")
    if row_indices.dtype != torch.int64:
        raise ValueError("capture row gather indices must be int64")
    # Gather indices are compacted at the beginning of the destination and
    # there cannot be more selected rows than scheduler source rows.  Decode
    # graphs commonly have 1-8 source rows while the bounded replay staging
    # buffer reserves 128 rows.  Sizing the grid from the full destination
    # made every projection scan all 128 slots on every generated token.
    active_rows = min(src.shape[0], dst.shape[0])
    block_size = 1024
    active_elements = active_rows * dst.shape[1]
    grid = ((active_elements + block_size - 1) // block_size,)
    torch.library.wrap_triton(_buffer_gather_rows_kernel)[grid](
        src,
        dst,
        row_indices,
        src.shape[0],
        active_rows,
        src.shape[1],
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK_SIZE=block_size,
    )


@torch.library.triton_op(
    "verallm::buffer_scatter_root_history",
    mutates_args=("dst",),
)
def buffer_scatter_root_history(
    dst: torch.Tensor,
    src: torch.Tensor,
    row_indices: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Scatter one graph-produced root row into each request-history row."""

    active_rows = int(reference.shape[0])
    if (
        dst.ndim != 3
        or src.ndim != 3
        or dst.dtype != torch.uint8
        or src.dtype != torch.uint8
        or int(dst.shape[2]) != 32
        or int(src.shape[2]) != 32
        or int(dst.shape[0]) != int(src.shape[0])
        or row_indices.ndim != 1
        or row_indices.dtype != torch.int64
        or int(row_indices.shape[0]) < active_rows
        or reference.ndim != 2
        or not dst.is_cuda
        or not src.is_cuda
        or not row_indices.is_cuda
        or not reference.is_cuda
        or any(
            tensor.device != src.device
            for tensor in (dst, row_indices, reference)
        )
        or int(dst.stride(2)) != 1
        or int(src.stride(2)) != 1
        or int(dst.stride(1)) <= 0
        or int(src.stride(1)) != 32
        or active_rows > int(src.shape[1])
    ):
        raise ValueError("execution-anchor root history geometry is malformed")
    if active_rows == 0:
        return
    block_size = 256
    active_elements = int(src.shape[0]) * active_rows * 32
    grid = ((active_elements + block_size - 1) // block_size,)
    torch.library.wrap_triton(_buffer_scatter_root_history_kernel)[grid](
        src,
        dst,
        row_indices,
        active_rows,
        active_elements,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        dst.shape[1],
        BLOCK_SIZE=block_size,
    )


@torch.library.triton_op(
    "verallm::buffer_copy_three_rows_padded",
    mutates_args=("dst0", "dst1", "dst2"),
)
def buffer_copy_three_rows_padded(
    dst0: torch.Tensor,
    dst1: torch.Tensor,
    dst2: torch.Tensor,
    src0: torch.Tensor,
    src1: torch.Tensor,
    src2: torch.Tensor,
) -> None:
    """Copy three same-row runtime tensors with one fixed CUDA launch."""

    destinations = (dst0, dst1, dst2)
    sources = (src0, src1, src2)
    row_count = int(src0.shape[0])
    if (
        any(tensor.ndim != 2 for tensor in (*destinations, *sources))
        or any(source.shape[0] != row_count for source in sources)
        or any(
            destination.shape[0] < row_count
            or destination.shape[1] < source.shape[1]
            or destination.dtype != source.dtype
            or destination.device != source.device
            for destination, source in zip(destinations, sources)
        )
        or any(source.device != src0.device for source in sources)
        or any(source.dtype != src0.dtype for source in sources)
        or not src0.is_cuda
        or any(source.stride(1) != 1 for source in sources)
        or any(destination.stride(1) != 1 for destination in destinations)
    ):
        raise ValueError("three-row runtime staging geometry is malformed")
    block_size = 1024
    active_elements = row_count * max(
        int(src0.shape[1]),
        int(src1.shape[1]),
        int(src2.shape[1]),
    )
    grid = ((active_elements + block_size - 1) // block_size,)
    torch.library.wrap_triton(_buffer_copy_three_rows_padded_kernel)[grid](
        src0,
        src1,
        src2,
        dst0,
        dst1,
        dst2,
        row_count,
        src0.shape[1],
        src1.shape[1],
        src2.shape[1],
        src0.stride(0),
        src1.stride(0),
        src2.stride(0),
        dst0.stride(0),
        dst1.stride(0),
        dst2.stride(0),
        BLOCK_SIZE=block_size,
    )


@torch.library.triton_op(
    "verallm::buffer_copy_input_router_sum_rows_padded",
    mutates_args=("dst0", "dst1", "dst2"),
)
def buffer_copy_input_router_sum_rows_padded(
    dst0: torch.Tensor,
    dst1: torch.Tensor,
    dst2: torch.Tensor,
    src0: torch.Tensor,
    src1: torch.Tensor,
    src2a: torch.Tensor,
    src2b: torch.Tensor,
) -> None:
    """Stage opaque MoE roots without materializing a duplicate sum."""

    destinations = (dst0, dst1, dst2)
    sources = (src0, src1, src2a, src2b)
    row_count = int(src0.shape[0])
    if (
        any(tensor.ndim != 2 for tensor in (*destinations, *sources))
        or any(source.shape[0] != row_count for source in sources)
        or src2a.shape[1] != src2b.shape[1]
        or any(
            destination.shape[0] < row_count
            or destination.shape[1] < source.shape[1]
            or destination.dtype != source.dtype
            or destination.device != source.device
            for destination, source in zip(
                destinations,
                (src0, src1, src2a),
            )
        )
        or any(source.device != src0.device for source in sources)
        or any(source.dtype != src0.dtype for source in sources)
        or not src0.is_cuda
        or any(source.stride(1) != 1 for source in sources)
        or any(destination.stride(1) != 1 for destination in destinations)
    ):
        raise ValueError("opaque MoE runtime staging geometry is malformed")
    block_size = 1024
    active_elements = row_count * max(
        int(src0.shape[1]),
        int(src1.shape[1]),
        int(src2a.shape[1]),
    )
    grid = ((active_elements + block_size - 1) // block_size,)
    torch.library.wrap_triton(
        _buffer_copy_input_router_sum_rows_padded_kernel
    )[grid](
        src0,
        src1,
        src2a,
        src2b,
        dst0,
        dst1,
        dst2,
        row_count,
        src0.shape[1],
        src1.shape[1],
        src2a.shape[1],
        src0.stride(0),
        src1.stride(0),
        src2a.stride(0),
        src2b.stride(0),
        dst0.stride(0),
        dst1.stride(0),
        dst2.stride(0),
        BLOCK_SIZE=block_size,
    )


@torch.library.triton_op(
    "verallm::buffer_copy_residual_boundaries_rows_padded",
    mutates_args=("residual_dst", "output_dst"),
)
def buffer_copy_residual_boundaries_rows_padded(
    residual_dst: torch.Tensor,
    output_dst: torch.Tensor,
    residual: torch.Tensor,
    hidden: torch.Tensor,
    output_residual: torch.Tensor,
) -> None:
    """Stage the exact sparse residual boundaries in one CUDA launch."""

    destinations = (residual_dst, output_dst)
    sources = (residual, hidden, output_residual)
    row_count = int(residual.shape[0])
    if (
        any(tensor.ndim != 2 for tensor in (*destinations, *sources))
        or any(source.shape[0] != row_count for source in sources)
        or any(source.shape[1] != residual.shape[1] for source in sources)
        or any(
            destination.shape[0] < row_count
            or destination.shape[1] < residual.shape[1]
            or destination.dtype != residual.dtype
            or destination.device != residual.device
            for destination in destinations
        )
        or any(source.device != residual.device for source in sources)
        or any(source.dtype != residual.dtype for source in sources)
        or not residual.is_cuda
        or any(source.stride(1) != 1 for source in sources)
        or any(destination.stride(1) != 1 for destination in destinations)
    ):
        raise ValueError("residual boundary staging geometry is malformed")
    block_size = 1024
    active_elements = row_count * int(residual.shape[1])
    grid = ((active_elements + block_size - 1) // block_size,)
    torch.library.wrap_triton(
        _buffer_copy_residual_boundaries_rows_padded_kernel
    )[grid](
        residual,
        hidden,
        output_residual,
        residual_dst,
        output_dst,
        row_count,
        residual.shape[1],
        residual.stride(0),
        hidden.stride(0),
        output_residual.stride(0),
        residual_dst.stride(0),
        output_dst.stride(0),
        BLOCK_SIZE=block_size,
    )


@torch.library.custom_op(
    "verallm::activation_row_roots",
    mutates_args=("dst",),
)
def activation_row_roots(
    dst: torch.Tensor,
    src: torch.Tensor,
    lane_bytes: int,
) -> None:
    """Write exact canonical execution-anchor roots for native runtime rows."""

    if (
        dst.ndim != 2
        or dst.dtype != torch.uint8
        or dst.shape[1] != 32
        or src.ndim != 2
        or dst.shape[0] < src.shape[0]
        or not dst.is_cuda
        or not src.is_cuda
        or dst.device != src.device
        or src.stride(1) != 1
        or src.stride(0) < src.shape[1]
        or lane_bytes not in (256, 2048)
    ):
        raise ValueError("runtime activation row-root geometry is malformed")
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_row_roots_into",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "native runtime activation row reducer is unavailable"
        )
    fn(dst, src, lane_bytes, src.device.index or 0)


@torch.library.register_fake("verallm::activation_row_roots")
def _activation_row_roots_fake(
    dst: torch.Tensor,
    src: torch.Tensor,
    lane_bytes: int,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_row_roots_or_stage",
    mutates_args=("dst", "staging"),
)
def activation_row_roots_or_stage(
    dst: torch.Tensor,
    staging: torch.Tensor,
    src: torch.Tensor,
    lane_bytes: int,
) -> None:
    """Stage bounded decode rows or directly hash a larger prefill batch."""

    if (
        dst.ndim != 2
        or dst.dtype != torch.uint8
        or dst.shape[1] != 32
        or staging.ndim != 2
        or src.ndim != 2
        or staging.dtype != src.dtype
        or staging.shape[1] < src.shape[1]
        or not dst.is_cuda
        or not staging.is_cuda
        or not src.is_cuda
        or dst.device != src.device
        or staging.device != src.device
        or src.stride(1) != 1
        or staging.stride(1) != 1
        or lane_bytes != 2048
    ):
        raise ValueError("staged runtime activation geometry is malformed")
    if src.shape[0] <= staging.shape[0]:
        block_size = 1024
        total = src.shape[0] * src.shape[1]
        grid = ((total + block_size - 1) // block_size,)
        _buffer_copy_rows_padded_kernel[grid](
            src,
            staging,
            src.shape[0],
            src.shape[1],
            src.stride(0),
            src.stride(1),
            staging.stride(0),
            staging.stride(1),
            BLOCK_SIZE=block_size,
        )
        return
    activation_row_roots(dst, src, lane_bytes)


@torch.library.register_fake("verallm::activation_row_roots_or_stage")
def _activation_row_roots_or_stage_fake(
    dst: torch.Tensor,
    staging: torch.Tensor,
    src: torch.Tensor,
    lane_bytes: int,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_row_roots_or_stage_retain",
    mutates_args=(
        "dst",
        "staging",
        "retained_values",
        "retained_hashes",
    ),
)
def activation_row_roots_or_stage_retain(
    dst: torch.Tensor,
    staging: torch.Tensor,
    retained_values: torch.Tensor,
    retained_hashes: torch.Tensor,
    retention_slots: torch.Tensor,
    src: torch.Tensor,
    retention_stage: int,
    lane_bytes: int,
) -> None:
    """Stage small rows or hash and retain one selected large stage."""

    if (
        dst.ndim != 2
        or dst.dtype != torch.uint8
        or dst.shape[1] != 32
        or staging.ndim != 2
        or src.ndim != 2
        or staging.dtype != src.dtype
        or staging.shape[1] < src.shape[1]
        or retained_values.ndim != 4
        or retained_values.dtype != torch.uint8
        or retained_values.shape[3] != 2048
        or retained_hashes.ndim != 4
        or retained_hashes.dtype != torch.uint8
        or retained_hashes.shape[:3] != retained_values.shape[:3]
        or retained_hashes.shape[3] != 32
        or retention_slots.ndim != 1
        or retention_slots.dtype != torch.int32
        or retention_stage < 0
        or retention_stage >= retention_slots.shape[0]
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                staging,
                retained_values,
                retained_hashes,
                retention_slots,
                src,
            )
        )
        or any(
            tensor.device != src.device
            for tensor in (
                dst,
                staging,
                retained_values,
                retained_hashes,
                retention_slots,
            )
        )
        or src.stride(1) != 1
        or staging.stride(1) != 1
        or lane_bytes != 2048
    ):
        raise ValueError("retained runtime activation geometry is malformed")
    if src.shape[0] <= staging.shape[0]:
        block_size = 1024
        total = src.shape[0] * src.shape[1]
        grid = ((total + block_size - 1) // block_size,)
        _buffer_copy_rows_padded_kernel[grid](
            src,
            staging,
            src.shape[0],
            src.shape[1],
            src.stride(0),
            src.stride(1),
            staging.stride(0),
            staging.stride(1),
            BLOCK_SIZE=block_size,
        )
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_row_roots_retain_into",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "native retained runtime activation reducer is unavailable"
        )
    fn(
        dst,
        src,
        retained_values,
        retained_hashes,
        retention_slots,
        retention_stage,
        lane_bytes,
        src.device.index or 0,
    )


@torch.library.register_fake(
    "verallm::activation_row_roots_or_stage_retain"
)
def _activation_row_roots_or_stage_retain_fake(
    dst: torch.Tensor,
    staging: torch.Tensor,
    retained_values: torch.Tensor,
    retained_hashes: torch.Tensor,
    retention_slots: torch.Tensor,
    src: torch.Tensor,
    retention_stage: int,
    lane_bytes: int,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_staged_row_roots",
    mutates_args=("dst", "scratch"),
)
def activation_staged_row_roots(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Finalize all staged decode roots in one native batch."""

    if (
        dst.ndim != 3
        or dst.dtype != torch.uint8
        or dst.shape[2] != 32
        or scratch.ndim != 4
        or scratch.dtype != torch.uint8
        or scratch.shape[3] != 32
        or staging.ndim != 3
        or reference.ndim != 2
        or row_widths.ndim != 1
        or row_widths.dtype != torch.int32
        or dst.shape[0] != staging.shape[0]
        or scratch.shape[0] != staging.shape[0]
        or scratch.shape[1] != staging.shape[1]
        or row_widths.shape[0] != staging.shape[0]
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                scratch,
                staging,
                row_widths,
                reference,
            )
        )
        or any(
            tensor.device != staging.device
            for tensor in (
                dst,
                scratch,
                row_widths,
                reference,
            )
        )
    ):
        raise ValueError("staged runtime root batch is malformed")
    row_count = reference.shape[0]
    if row_count > staging.shape[1]:
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_into",
        None,
    )
    if fn is None:
        raise RuntimeError("native staged runtime root reducer is unavailable")
    fn(
        dst,
        scratch,
        staging,
        row_widths,
        row_count,
        staging.device.index or 0,
    )


@torch.library.register_fake("verallm::activation_staged_row_roots")
def _activation_staged_row_roots_fake(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_staged_row_roots_lanes",
    mutates_args=("dst", "scratch"),
)
def activation_staged_row_roots_lanes(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Finalize mixed 256/2,048-byte staged roots in one native batch."""

    if (
        dst.ndim != 3
        or dst.dtype != torch.uint8
        or dst.shape[2] != 32
        or scratch.ndim != 4
        or scratch.dtype != torch.uint8
        or scratch.shape[3] != 32
        or staging.ndim != 3
        or reference.ndim != 2
        or row_widths.ndim != 1
        or row_widths.dtype != torch.int32
        or lane_widths.ndim != 1
        or lane_widths.dtype != torch.int32
        or dst.shape[0] != staging.shape[0]
        or scratch.shape[0] != staging.shape[0]
        or scratch.shape[1] != staging.shape[1]
        or row_widths.shape[0] != staging.shape[0]
        or lane_widths.shape[0] != staging.shape[0]
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                scratch,
                staging,
                row_widths,
                lane_widths,
                reference,
            )
        )
        or any(
            tensor.device != staging.device
            for tensor in (
                dst,
                scratch,
                row_widths,
                lane_widths,
                reference,
            )
        )
    ):
        raise ValueError("mixed-lane staged runtime root batch is malformed")
    row_count = reference.shape[0]
    if row_count > staging.shape[1]:
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_lanes_into",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "native mixed-lane staged runtime root reducer is unavailable"
        )
    fn(
        dst,
        scratch,
        staging,
        row_widths,
        lane_widths,
        row_count,
        staging.device.index or 0,
    )


@torch.library.register_fake(
    "verallm::activation_staged_row_roots_lanes"
)
def _activation_staged_row_roots_lanes_fake(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_staged_row_roots_lanes_history",
    mutates_args=("dst", "scratch"),
)
def activation_staged_row_roots_lanes_history(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    destination_indices: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Finalize mixed-lane roots directly into persistent row history."""

    if (
        dst.ndim != 3
        or dst.dtype != torch.uint8
        or dst.shape[2] != 32
        or scratch.ndim != 4
        or scratch.dtype != torch.uint8
        or scratch.shape[3] != 32
        or staging.ndim != 3
        or reference.ndim != 2
        or row_widths.ndim != 1
        or row_widths.dtype != torch.int32
        or lane_widths.ndim != 1
        or lane_widths.dtype != torch.int32
        or destination_indices.ndim != 1
        or destination_indices.dtype != torch.int64
        or dst.shape[0] != staging.shape[0]
        or scratch.shape[0] != staging.shape[0]
        or scratch.shape[1] != staging.shape[1]
        or row_widths.shape[0] != staging.shape[0]
        or lane_widths.shape[0] != staging.shape[0]
        or destination_indices.shape[0] < reference.shape[0]
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                scratch,
                staging,
                row_widths,
                lane_widths,
                destination_indices,
                reference,
            )
        )
        or any(
            tensor.device != staging.device
            for tensor in (
                dst,
                scratch,
                row_widths,
                lane_widths,
                destination_indices,
                reference,
            )
        )
    ):
        raise ValueError(
            "mixed-lane persistent runtime root batch is malformed"
        )
    row_count = reference.shape[0]
    if row_count > staging.shape[1]:
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_lanes_history_into",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "native persistent staged runtime root reducer is unavailable"
        )
    fn(
        dst,
        scratch,
        staging,
        row_widths,
        lane_widths,
        destination_indices,
        row_count,
        staging.device.index or 0,
    )


@torch.library.register_fake(
    "verallm::activation_staged_row_roots_lanes_history"
)
def _activation_staged_row_roots_lanes_history_fake(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    staging: torch.Tensor,
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    destination_indices: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_tensor_row_roots_lanes",
    mutates_args=("dst", "scratch"),
)
def activation_tensor_row_roots_lanes(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    sources_first: list[torch.Tensor],
    sources_second: list[torch.Tensor],
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Hash one decode step directly from its opaque runtime tensors."""

    stage_count = len(sources_first)
    if (
        stage_count <= 0
        or stage_count > 128
        or len(sources_second) != stage_count
        or dst.ndim != 3
        or dst.dtype != torch.uint8
        or dst.shape[0] != stage_count
        or dst.shape[2] != 32
        or scratch.ndim != 4
        or scratch.dtype != torch.uint8
        or scratch.shape[0] != stage_count
        or scratch.shape[3] != 32
        or row_widths.ndim != 1
        or row_widths.dtype != torch.int32
        or row_widths.shape[0] != stage_count
        or lane_widths.ndim != 1
        or lane_widths.dtype != torch.int32
        or lane_widths.shape[0] != stage_count
        or reference.ndim != 2
        or any(source.ndim != 2 for source in sources_first)
        or any(source.ndim != 2 for source in sources_second)
        or any(source.dtype != torch.bfloat16 for source in sources_first)
        or any(source.dtype != torch.bfloat16 for source in sources_second)
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                scratch,
                row_widths,
                lane_widths,
                reference,
                *sources_first,
                *sources_second,
            )
        )
        or any(
            tensor.device != dst.device
            for tensor in (
                scratch,
                row_widths,
                lane_widths,
                reference,
                *sources_first,
                *sources_second,
            )
        )
    ):
        raise ValueError("runtime tensor root batch is malformed")
    row_count = int(reference.shape[0])
    if row_count > int(scratch.shape[1]):
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_tensor_row_roots_lanes_into",
        None,
    )
    if fn is None:
        raise RuntimeError("native runtime tensor root reducer is unavailable")
    fn(
        dst,
        scratch,
        sources_first,
        sources_second,
        row_widths,
        lane_widths,
        row_count,
        dst.device.index or 0,
    )


@torch.library.register_fake(
    "verallm::activation_tensor_row_roots_lanes"
)
def _activation_tensor_row_roots_lanes_fake(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    sources_first: list[torch.Tensor],
    sources_second: list[torch.Tensor],
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "verallm::activation_indexed_tensor_row_roots_lanes_history",
    mutates_args=("dst", "scratch"),
)
def activation_indexed_tensor_row_roots_lanes_history(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    sources: list[torch.Tensor],
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    source_indices: list[torch.Tensor],
    destination_indices: torch.Tensor,
) -> None:
    """Hash scheduler-indexed opaque cache rows into persistent history."""

    stage_count = len(sources)
    if (
        stage_count <= 0
        or stage_count > 128
        or dst.ndim != 3
        or dst.dtype != torch.uint8
        or dst.shape[0] != stage_count
        or dst.shape[2] != 32
        or scratch.ndim != 4
        or scratch.dtype != torch.uint8
        or scratch.shape[0] != stage_count
        or scratch.shape[3] != 32
        or row_widths.ndim != 1
        or row_widths.dtype != torch.int32
        or row_widths.shape[0] != stage_count
        or lane_widths.ndim != 1
        or lane_widths.dtype != torch.int32
        or lane_widths.shape[0] != stage_count
        or len(source_indices) != stage_count
        or any(
            item.ndim != 1 or item.dtype != torch.int32
            for item in source_indices
        )
        or destination_indices.ndim != 1
        or destination_indices.dtype != torch.int64
        or any(
            item.shape[0] != destination_indices.shape[0]
            for item in source_indices
        )
        or destination_indices.shape[0] > scratch.shape[1]
        or any(not 2 <= source.ndim <= 4 for source in sources)
        or any(
            source.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            for source in sources
        )
        or not all(
            tensor.is_cuda
            for tensor in (
                dst,
                scratch,
                row_widths,
                lane_widths,
                destination_indices,
                *sources,
                *source_indices,
            )
        )
        or any(
            tensor.device != dst.device
            for tensor in (
                scratch,
                row_widths,
                lane_widths,
                destination_indices,
                *sources,
                *source_indices,
            )
        )
    ):
        raise ValueError("indexed runtime tensor root history is malformed")
    row_count = int(destination_indices.shape[0])
    if row_count <= 0:
        return
    from zkllm.cuda import zkllm_native

    fn = getattr(
        zkllm_native,
        "cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "native indexed runtime tensor root reducer is unavailable"
        )
    fn(
        dst,
        scratch,
        sources,
        row_widths,
        lane_widths,
        source_indices,
        destination_indices,
        row_count,
        dst.device.index or 0,
    )


@torch.library.register_fake(
    "verallm::activation_indexed_tensor_row_roots_lanes_history"
)
def _activation_indexed_tensor_row_roots_lanes_history_fake(
    dst: torch.Tensor,
    scratch: torch.Tensor,
    sources: list[torch.Tensor],
    row_widths: torch.Tensor,
    lane_widths: torch.Tensor,
    source_indices: list[torch.Tensor],
    destination_indices: torch.Tensor,
) -> None:
    return None


# -- Custom op registration (graph split point) ------------------------------

@torch.library.custom_op("verallm::capture", mutates_args=("x",))
def capture(x: torch.Tensor, layer_idx: int, tensor_kind: int) -> torch.Tensor:
    """Graph split point for activation capture.

    Runs in eager mode at the split.  Delegates to the active
    RequestActivationTracker for per-request row slicing.

    Returns a zero-sized sequencing token rather than cloning the complete
    activation.  The operation's mutation annotation keeps the split/capture
    side effect live; all call sites deliberately discard the token.  Runtime
    values that a proof needs are copied by ``capture_at_split`` with the
    request's bounded retention policy.

    Args:
        x: activation tensor [batch_tokens, hidden_dim]
        layer_idx: transformer layer index
        tensor_kind: verifier-owned witness kind.  The canonical mapping lives
            in ``RequestActivationTracker._KIND_TO_SUFFIX``.
    """
    tracker = _active_tracker
    if tracker is not None:
        return tracker.capture_at_split(x, layer_idx, tensor_kind)
    return x.new_empty((0,))


@torch.library.register_fake("verallm::capture")
def _capture_fake(x: torch.Tensor, layer_idx: int, tensor_kind: int) -> torch.Tensor:
    """Shape inference for the zero-sized capture sequencing token."""
    return x.new_empty((0,))
