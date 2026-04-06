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


@torch.library.custom_op("verallm::buffer_copy", mutates_args=("dst",))
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
    _buffer_copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)


@torch.library.register_fake("verallm::buffer_copy")
def _buffer_copy_fake(
    dst: torch.Tensor, src: torch.Tensor, n_tokens: int,
) -> None:
    return None


# -- Custom op registration (graph split point) ------------------------------

@torch.library.custom_op("verallm::capture", mutates_args=("x",))
def capture(x: torch.Tensor, layer_idx: int, tensor_kind: int) -> torch.Tensor:
    """Graph split point for activation capture.

    Runs in eager mode at the split.  Delegates to the active
    RequestActivationTracker for per-request row slicing.

    Returns x.clone() to provide a distinct output tensor.  The return
    value is used by MoE call-sites (router_logits capture) so the op
    is not dead-code-eliminated.  For dense gate_proj capture where the
    return is discarded, x still has downstream users (self.original(x))
    so DCE does not apply.

    Args:
        x: activation tensor [batch_tokens, hidden_dim]
        layer_idx: transformer layer index
        tensor_kind: 0 = mlp_gate_input, 1 = router_logits, 2 = mlp_gate_output
    """
    tracker = _active_tracker
    if tracker is not None:
        # capture_at_split returns the cloned snapshot — reuse it as our
        # output to avoid a redundant second x.clone().
        return tracker.capture_at_split(x, layer_idx, tensor_kind)
    return x.clone()


@torch.library.register_fake("verallm::capture")
def _capture_fake(x: torch.Tensor, layer_idx: int, tensor_kind: int) -> torch.Tensor:
    """Shape inference for torch.compile: output shape == input shape.
    """
    return x.clone()
