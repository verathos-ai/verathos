"""Canonical sampling helper for the lm_head hook (CPU fallback path).

Must remain as a plain .py source file to guarantee bit-exact fp64
rounding behaviour with the validator's canonical_sample.
"""
from __future__ import annotations

import numpy as np
import torch


def run_canonical_sample_and_mask(
    top_vals_gpu: torch.Tensor,
    top_idx_gpu: torch.Tensor,
    logits: torch.Tensor,
    row_idx: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    seed: bytes,
    step: int,
) -> None:
    """Run canonical_sample on CPU numpy and mask logits in-place.

    Transfers top-K from GPU to CPU numpy, runs the exact same
    canonical_sample as the validator, then writes the -inf mask
    back to the GPU logits tensor.
    """
    from verallm.sampling import canonical_sample

    vals_np = top_vals_gpu.detach().float().cpu().numpy()
    idx_np = top_idx_gpu.detach().cpu().numpy().astype(np.int64)

    # Sort by (value DESC, index ASC) — same as the Merkle leaf
    # serialization and the validator's parse_top_k_leaf output.
    # Without this, torch.topk's arbitrary tie-breaking differs
    # from the lexsort-sorted order the validator expects.
    order = np.lexsort((idx_np, -vals_np))
    vals_np = vals_np[order]
    idx_np = idx_np[order]

    token = canonical_sample(
        vals_np, idx_np,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        seed=seed,
        step=step,
    )

    orig_val = logits[row_idx, token].clone()
    logits[row_idx].fill_(float("-inf"))
    logits[row_idx, token] = orig_val
