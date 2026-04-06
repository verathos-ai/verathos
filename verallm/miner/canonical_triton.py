"""Fused Triton kernel for canonical sampling.

Must remain as a plain .py source file — Triton's JIT compiler uses
inspect.getsource() at runtime to read the kernel source code.
"""
from __future__ import annotations

import torch

_triton_kernel = None


def canonical_triton_sample_and_mask(
    top_vals: torch.Tensor,
    top_idx: torch.Tensor,
    logits: torch.Tensor,
    row_idx: int,
    inv_temp: float,
    user_top_k: int,
    top_p: float,
    min_p: float,
    draw: float,
):
    """Single Triton kernel: canonical sample → -inf mask.

    Fuses softmax + top_k/min_p/top_p filters + cumsum + searchsorted
    into one GPU kernel launch.  Writes the chosen token index to a
    GPU output tensor (zero .item() syncs), then masks logits via
    gather/fill/scatter (2 fast torch ops).
    """
    global _triton_kernel
    if _triton_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _kernel(
            top_vals_ptr, top_idx_ptr, out_ptr,
            draw, inv_temp,
            user_top_k,
            top_p, min_p,
            BLOCK: tl.constexpr,
        ):
            offs = tl.arange(0, BLOCK)
            vals = tl.load(top_vals_ptr + offs).to(tl.float64)
            idxs = tl.load(top_idx_ptr + offs)

            # Temperature scale
            vals = vals * inv_temp

            # Stable softmax
            vals = vals - tl.max(vals, axis=0)
            probs = tl.exp(vals)
            probs = probs / tl.sum(probs, axis=0)

            # User top-k filter (no-op when user_top_k <= 0)
            probs = tl.where(
                (user_top_k <= 0) | (offs < user_top_k), probs, 0.0,
            )

            # Min-p filter (no-op when min_p <= 0)
            p0 = tl.sum(tl.where(offs == 0, probs, 0.0), axis=0)
            probs = tl.where(
                (min_p <= 0.0) | (probs >= p0 * min_p), probs, 0.0,
            )

            # Top-p filter (no-op when top_p >= 1 or top_p <= 0)
            cum = tl.cumsum(probs, axis=0)
            prev_cum = cum - probs
            do_top_p = (top_p > 0.0) & (top_p < 1.0)
            probs = tl.where(
                (~do_top_p) | (prev_cum <= top_p), probs, 0.0,
            )

            # Renormalize
            total = tl.sum(probs, axis=0)
            safe_total = tl.where(total > 0.0, total, 1.0)
            probs = probs / safe_total

            # Searchsorted: count positions where cdf <= draw
            cdf = tl.cumsum(probs, axis=0)
            below = tl.where(cdf <= draw, 1, 0)
            local_idx = tl.sum(below, axis=0)

            # Fallback: overflow → last nonzero; degenerate → 0
            nonzero_pos = tl.where(probs > 0.0, offs, -1)
            last_nz = tl.max(nonzero_pos, axis=0)
            last_nz = tl.where(last_nz >= 0, last_nz, 0)
            local_idx = tl.where(local_idx >= BLOCK, last_nz, local_idx)
            local_idx = tl.where(total > 0.0, local_idx, 0)

            # Look up chosen vocab token and write to output
            chosen_mask = (offs == local_idx)
            chosen_vocab_idx = tl.sum(tl.where(chosen_mask, idxs, 0), axis=0)
            tl.store(out_ptr, chosen_vocab_idx)

        _triton_kernel = _kernel

    K = top_vals.shape[0]
    out = torch.empty(1, dtype=torch.int64, device=top_vals.device)
    _triton_kernel[(1,)](
        top_vals, top_idx, out,
        draw, inv_temp,
        int(user_top_k) if user_top_k > 0 else 0,
        float(top_p), float(min_p),
        BLOCK=K,
    )

    # -inf mask: 2 torch ops, no .item() sync.
    # gather chosen value → fill row -inf → scatter value back
    orig_val = logits[row_idx].gather(0, out).clone()
    logits[row_idx].fill_(float('-inf'))
    logits[row_idx].scatter_(0, out, orig_val)
