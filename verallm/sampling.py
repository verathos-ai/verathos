"""
Decode-integrity sampling utilities.

Shared helpers for:
- commitment-domain hidden-row encoding/Merkle roots
- deterministic challenge-rate normalization
- proof-domain hidden-row quantization
- compact int32 logits row serialization
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from verallm.crypto.merkle import MerkleTree


EMPTY_DECODE_HIDDEN_ROOT = hashlib.sha256(b"DECODE_HIDDEN_ROWS_EMPTY_V1").digest()


def clamp_sampling_bps(value: int | float) -> int:
    """Clamp sampling verification rate to integer basis points [0, 10000]."""
    try:
        bps = int(round(float(value)))
    except Exception:
        return 0
    return max(0, min(10_000, bps))


def temperature_to_milli(temperature: float) -> int:
    """Quantize temperature to milli-units for commitment binding."""
    try:
        t = float(temperature)
    except Exception:
        t = 0.0
    if t < 0.0:
        t = 0.0
    # Keep within uint16 range.
    return max(0, min(65_535, int(round(t * 1000.0))))


def canonicalize_hidden_row(row: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Canonical hidden row as CPU float32 1D tensor with NaN/Inf sanitized."""
    if isinstance(row, torch.Tensor):
        t = row
    else:
        t = torch.tensor(list(row), dtype=torch.float32)
    if t.dim() == 2 and t.shape[0] == 1:
        t = t[0]
    if t.dim() != 1:
        t = t.reshape(-1)
    t = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return t.contiguous().cpu()


def hidden_row_to_bytes(row: torch.Tensor | Sequence[float]) -> bytes:
    """Serialize hidden row as little-endian fp16 bytes."""
    r = canonicalize_hidden_row(row)
    return r.half().numpy().astype("<f2", copy=False).tobytes()


def hidden_row_from_bytes(data: bytes) -> np.ndarray:
    """Deserialize hidden row bytes (little-endian fp16) to float32 numpy."""
    return np.frombuffer(data, dtype="<f2").astype(np.float32, copy=False)


def build_hidden_row_merkle(
    hidden_steps: Iterable[torch.Tensor],
) -> Tuple[Optional[MerkleTree], List[bytes], bytes]:
    """Build Merkle root over per-step hidden rows."""
    row_bytes = [hidden_row_to_bytes(row) for row in hidden_steps]
    if not row_bytes:
        return None, [], EMPTY_DECODE_HIDDEN_ROOT
    tree = MerkleTree(row_bytes)
    return tree, row_bytes, tree.root


def quantize_hidden_row_int64(row: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Quantize one hidden row to proof-domain int64 shape [1, hidden_dim]."""
    r = canonicalize_hidden_row(row)
    absmax = r.abs().max().clamp(min=1e-8)
    q = (r / absmax * 127.0).round().clamp(-128, 127).to(torch.int64)
    return q.view(1, -1).contiguous()


def logits_i32_to_bytes(logits: torch.Tensor | Sequence[int]) -> bytes:
    """Serialize one proven logits row as little-endian int32 bytes."""
    if isinstance(logits, torch.Tensor):
        arr64 = logits.detach().cpu().to(torch.int64).reshape(-1).numpy()
    else:
        arr64 = np.asarray(list(logits), dtype=np.int64).reshape(-1)
    if arr64.size == 0:
        return b""
    if arr64.min() < np.iinfo(np.int32).min or arr64.max() > np.iinfo(np.int32).max:
        raise ValueError("proved logits exceed int32 range")
    arr32 = arr64.astype("<i4", copy=False)
    return arr32.tobytes()


def logits_i32_from_bytes(data: bytes) -> np.ndarray:
    """Deserialize little-endian int32 logits bytes."""
    return np.frombuffer(data, dtype="<i4")


def argmax_from_logits_i32_bytes(data: bytes) -> int:
    """Return argmax index from serialized int32 logits row."""
    arr = logits_i32_from_bytes(data)
    if arr.size == 0:
        return -1
    return int(np.argmax(arr))


# ============================================================================
# Quantization-stable argmax verification
# ============================================================================

# Committed token must be in top-k of proved int32 logits.
# The int8×int8→int32 GEMM approximates fp16 logits; quantization noise
# can flip the argmax when the top-2 gap is small.  A generous k handles
# this without false rejections.
SAMPLING_ARGMAX_TOP_K = 5

# Basis-point threshold at or above which high-assurance mode is activated
# (fp16 logits row opening + exact argmax check).
HIGH_ASSURANCE_BPS = 9000


def verify_quantized_argmax(
    logits_i32: np.ndarray,
    committed_token: int,
    top_k: int = SAMPLING_ARGMAX_TOP_K,
) -> Tuple[bool, str]:
    """Quantization-stable argmax: committed token must be in top-k of proved logits.

    Returns (passed, detail_message).
    """
    if logits_i32.size == 0:
        return False, "empty logits row"
    if committed_token < 0 or committed_token >= logits_i32.size:
        return False, f"token {committed_token} out of range [0, {logits_i32.size})"

    proved_argmax = int(np.argmax(logits_i32))

    # Exact argmax match is strictly stronger than top-k — always accept.
    # This also handles the all-zero logits case (e.g. EOS/pad tokens) where
    # argpartition returns arbitrary indices due to tied values.
    if committed_token == proved_argmax:
        return True, (
            f"token {committed_token} is proved argmax (logit={int(logits_i32[proved_argmax])})"
        )

    k = min(top_k, logits_i32.size)
    top_k_indices = set(np.argpartition(logits_i32, -k)[-k:].tolist())

    if committed_token in top_k_indices:
        rank = _rank_in_logits(logits_i32, committed_token)
        return True, (
            f"token {committed_token} in top-{k} "
            f"(proved_argmax={proved_argmax}, rank={rank})"
        )

    committed_logit = int(logits_i32[committed_token])
    top1_logit = int(logits_i32[proved_argmax])
    rank = _rank_in_logits(logits_i32, committed_token)
    return False, (
        f"token {committed_token} NOT in top-{k}: "
        f"proved_argmax={proved_argmax} (logit={top1_logit}), "
        f"committed_logit={committed_logit}, gap={top1_logit - committed_logit}, rank={rank}"
    )


def _rank_in_logits(logits_i32: np.ndarray, token: int) -> int:
    """1-based rank of token in logits (1 = highest)."""
    return int((logits_i32 > logits_i32[token]).sum()) + 1


def serialize_top_k_to_bytes(
    top_vals_sorted: np.ndarray,
    top_indices_sorted: np.ndarray,
) -> bytes:
    """Serialize a top-K subset (already sorted) to a single bytes object.

    Layout: ``top_vals (K * 4 bytes, fp32)`` || ``top_indices (K * 8 bytes, int64)``.
    K is implicit from the byte length: ``K = len(bytes) // 12``.

    Both miner and validator use this format so the leaf is bit-exactly
    identical on both sides.  The caller must ensure the inputs are
    already sorted by ``(value DESC, index ASC)`` — i.e. produced by
    ``extract_top_k_sorted``.
    """
    vals_b = np.asarray(top_vals_sorted, dtype="<f4").tobytes()
    idx_b = np.asarray(top_indices_sorted, dtype="<i8").tobytes()
    return vals_b + idx_b


def parse_top_k_leaf(
    leaf_bytes: bytes,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a top-K leaf back into ``(top_vals, top_indices)`` numpy arrays.

    Inverse of ``serialize_top_k_to_bytes``.  K is derived from the byte
    length so the same helper can handle any vocab/K combination.

    Returns ``(top_vals_fp32, top_indices_int64)`` — both length K, in
    the same sorted order they were serialized in.
    """
    n = len(leaf_bytes)
    if n == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    if n % 12 != 0:
        raise ValueError(
            f"top-K leaf length {n} is not a multiple of 12 (4 bytes fp32 + 8 bytes int64 per entry)"
        )
    k = n // 12
    vals = np.frombuffer(leaf_bytes[: 4 * k], dtype="<f4").astype(np.float32, copy=False)
    idx = np.frombuffer(leaf_bytes[4 * k :], dtype="<i8").astype(np.int64, copy=False)
    return vals, idx


def verify_fp16_argmax(leaf_bytes: bytes, committed_token: int) -> Tuple[bool, str]:
    """Exact argmax check against a top-K leaf.

    The captured leaf is a top-K subset sorted by ``(value DESC, index ASC)``,
    so the global vocab argmax is always at sorted position 0.  Verifying
    the argmax reduces to: ``top_indices[0] == committed_token``.

    Falls back to the legacy full-vocab fp32 layout if the leaf length
    isn't a multiple of 12 — handles old proofs produced before the
    top-K capture switch.
    """
    n = len(leaf_bytes)
    if n == 0:
        return False, "empty logits row"

    if n % 12 == 0:
        # Top-K leaf format: K × fp32 vals + K × int64 indices.
        try:
            vals, idx = parse_top_k_leaf(leaf_bytes)
        except ValueError as e:
            return False, str(e)
        if vals.size == 0:
            return False, "empty top-K leaf"
        # Top-K is sorted DESC by value, so position 0 is the global argmax.
        argmax = int(idx[0])
        if argmax == committed_token:
            return True, f"top-K argmax (idx[0]={argmax}) matches committed token"
        return False, (
            f"top-K argmax mismatch: idx[0]={argmax}, committed={committed_token}"
        )

    # Legacy full-vocab fallback (fp32 or fp16).
    if n % 4 == 0:
        logits = np.frombuffer(leaf_bytes, dtype="<f4")
    elif n % 2 == 0:
        logits = np.frombuffer(leaf_bytes, dtype="<f2")
    else:
        return False, f"invalid logits row length: {n}"
    if logits.size == 0:
        return False, "empty logits row"
    exact_argmax = int(np.argmax(logits))
    if exact_argmax == committed_token:
        return True, f"exact logits argmax matches token {committed_token}"
    return False, (
        f"logits argmax mismatch: argmax={exact_argmax}, committed={committed_token}"
    )


def build_logits_row_merkle(
    leaves: Iterable,
) -> Tuple[Optional[MerkleTree], List[bytes], bytes]:
    """Build Merkle tree over per-step logits leaves.

    Accepts either:
      - ``bytes`` items (already serialized — top-K leaves from the
        activation tracker, the common path) — used directly.
      - torch tensors (legacy/test path) — converted via the legacy
        full-vocab fp32 serialization for backwards compat with tests.

    Uses MerkleTree's internal ``hash_leaf`` for domain-separated hashing.
    The verifier uses ``verify_merkle_path(root, row_bytes, path)`` which
    re-hashes the raw bytes the same way.

    Returns (tree, row_bytes_list, root).
    """
    row_bytes_list: List[bytes] = []
    for item in leaves:
        if isinstance(item, (bytes, bytearray, memoryview)):
            row_bytes_list.append(bytes(item))
        elif isinstance(item, torch.Tensor):
            # Legacy fallback: full-vocab fp32 serialization.
            row_bytes_list.append(
                item.detach().squeeze().float().cpu().numpy().astype("<f4", copy=False).tobytes()
            )
        else:
            row_bytes_list.append(bytes(item))
    if not row_bytes_list:
        return None, [], b""
    tree = MerkleTree(row_bytes_list)
    return tree, row_bytes_list, tree.root


# ============================================================================
# Sampler config binding
# ============================================================================

import struct as _struct  # noqa: E402


def compute_sampler_config_hash(
    top_k: int = -1,
    top_p: float = 1.0,
    min_p: float = 0.0,
    chat_template_hash: bytes = b"",
    presence_penalty: float = 0.0,
) -> bytes:
    """Compute deterministic hash of the full sampler configuration.

    Bound into ``InferenceCommitment.sampler_config_hash`` so the validator
    can detect parameter substitution (e.g. miner ignoring top_p).

    V2 adds presence_penalty to the binding. Previously pp was committed in
    ``presence_penalty_milli`` but omitted from the expected-hash check,
    giving a one-way binding that the validator could not verify against
    a request. V2 closes that gap.

    Args:
        top_k: Top-k value (-1 = disabled).
        top_p: Nucleus sampling threshold [0, 1].
        min_p: Minimum probability threshold [0, 1].
        chat_template_hash: SHA256 of the chat template string used.
        presence_penalty: OpenAI-style presence penalty in [0, 2].

    Returns:
        32-byte SHA256 digest.
    """
    h = hashlib.sha256(b"SAMPLER_CONFIG_V2")
    h.update(_struct.pack("<i", int(top_k)))
    # Quantize floats to milli-units for deterministic cross-platform hashing.
    h.update(_struct.pack("<I", max(0, min(65535, int(round(float(top_p) * 1000))))))
    h.update(_struct.pack("<I", max(0, min(65535, int(round(float(min_p) * 1000))))))
    h.update(_struct.pack("<H", max(0, min(65535, int(round(float(presence_penalty) * 1000))))))
    if chat_template_hash:
        h.update(chat_template_hash)
    return h.digest()


# ============================================================================
# Canonical CPU sampler (do_sample=True verification)
# ============================================================================

import random as _random  # noqa: E402


def derive_token_seed(batch_seed: bytes, step: int) -> bytes:
    """Derive per-token deterministic seed from batch seed + decode step.

    Returns 32-byte SHA256 digest usable as RNG seed.
    """
    return hashlib.sha256(
        b"VERILLM_SAMPLE_V1" + batch_seed + _struct.pack("<I", step)
    ).digest()


# ---------------------------------------------------------------------------
# canonical_sample — operates on a pre-extracted top-K subset
# ---------------------------------------------------------------------------
#
# At Qwen3.5-9B vocab=152064 the original full-vocab canonical_sample took
# ~1.5ms per call, dominating the per-token LP overhead and dropping
# do_sample=True throughput from ~88 tok/s native to ~70 tok/s.
#
# Empirical profiling on the live RTX 4090 showed:
#   - GPU→CPU sync + transfer: 0.115 ms
#   - canonical_sample math:   1.485 ms  ← bottleneck (np.exp + cumsum on 152K float64)
#   - mask creation:           0.025 ms
#
# The fix is to operate on the top-K subset (K=4096) instead of the full
# vocab.  Sampled tokens are essentially always in the top-K of the logits
# distribution at temperature 0.7 — the bottom (vocab - K) probability mass
# is < 1e-15 in fp64.  This drops the math from O(vocab) to O(K), a ~37x
# reduction, while remaining bit-exact between miner and validator.
#
# Bit-exact agreement requires that both sides:
#   1. Extract the SAME top-K from the same fp32 logits
#   2. Sort the top-K in the SAME deterministic order: by (value DESC, index ASC)
#   3. Run the SAME math on the sorted top-K
#
# Miner: GPU torch.topk → CPU → sort → canonical_sample
# Validator: opens fp16_logits_row from proof, CPU np.argpartition → sort → canonical_sample
#
# Both implementations apply np.lexsort((indices, -values)) for tie-breaking,
# guaranteeing the same sorted order even when fp32 values are tied.
# ---------------------------------------------------------------------------

CANONICAL_TOP_K = 1024


def extract_top_k_sorted(
    logits: np.ndarray,
    k: int = CANONICAL_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the top-k logits and sort them deterministically.

    Returns ``(top_values_sorted, top_indices_sorted)`` where the sort
    order is by ``(value DESC, original_index ASC)``.  This tie-break
    rule is the same on miner and validator so both produce the same
    canonical sort even on the (extraordinarily rare) fp32 ties.

    Args:
        logits: Raw logits array [vocab_size] (float32 or float64).
        k: Number of top entries to keep.  If vocab < k, returns all entries.

    Returns:
        (sorted_values, sorted_original_indices) — both length min(k, vocab).
    """
    n = int(logits.shape[0])
    actual_k = min(k, n)
    if actual_k >= n:
        # Whole vocab — sort everything.
        idx = np.arange(n, dtype=np.int64)
        vals = np.asarray(logits, dtype=np.float32)
    else:
        # argpartition is O(n); sort the K-element result instead of the full vocab.
        partial = np.argpartition(logits, -actual_k)[-actual_k:]
        idx = partial.astype(np.int64, copy=False)
        vals = np.asarray(logits[idx], dtype=np.float32)
    # Deterministic order: primary key value DESC, secondary key index ASC.
    order = np.lexsort((idx, -vals))
    return vals[order], idx[order]


def canonical_sample(
    top_values_sorted: np.ndarray,
    top_indices_sorted: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    seed: bytes,
    step: int,
) -> int:
    """Deterministic CPU sampler over a pre-sorted top-K subset.

    Both miner and validator call this with bit-identical inputs and
    produce bit-identical chosen tokens.

    Args:
        top_values_sorted: top-K logit values, sorted by ``extract_top_k_sorted``.
        top_indices_sorted: original vocab indices in the same order.
        temperature: Sampling temperature (> 0).
        top_k: User top-k filter (-1 = disabled).  Applied AFTER the
            internal CANONICAL_TOP_K subset.
        top_p: Nucleus sampling threshold.
        min_p: Minimum probability threshold.
        seed: 32-byte batch sampling seed.
        step: Decode step index.

    Returns:
        Selected ORIGINAL vocab index.
    """
    # Cast to fp64 for numerical stability of the sampling math.
    fvals = top_values_sorted.astype(np.float64, copy=True)

    # Temperature scaling.
    if temperature > 0:
        fvals /= temperature

    # Stable softmax over the K-element subset.
    fvals -= fvals.max()
    probs = np.exp(fvals)
    probs /= probs.sum()

    # User top-k filter — keeps the first ``top_k`` entries (already sorted by value DESC).
    if top_k > 0 and top_k < len(probs):
        probs[top_k:] = 0.0

    # Min-p filtering — relative to the largest probability (which is probs[0]).
    if min_p > 0.0 and probs.size > 0:
        threshold = float(probs[0]) * min_p
        probs[probs < threshold] = 0.0

    # Top-p (nucleus) — cumulative sum is monotonic because probs are sorted DESC.
    if 0.0 < top_p < 1.0:
        cum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cum, top_p, side="right")) + 1
        if cutoff < len(probs):
            probs[cutoff:] = 0.0

    # Re-normalize after filtering.
    total = probs.sum()
    if total <= 0:
        # Degenerate: pick the largest token (probs[0] in sorted order).
        return int(top_indices_sorted[0])
    probs /= total

    # Seeded categorical sampling — vectorized via cumsum + searchsorted.
    token_seed = derive_token_seed(seed, step)
    rng = _random.Random(int.from_bytes(token_seed[:8], "little"))
    draw = rng.random()  # [0, 1)
    cdf = np.cumsum(probs)
    local_idx = int(np.searchsorted(cdf, draw, side="right"))
    if local_idx >= len(probs):
        # fp64 rounding fallback: pick the last non-zero entry.
        nonzero = np.flatnonzero(probs > 0.0)
        if nonzero.size == 0:
            return int(top_indices_sorted[0])
        local_idx = int(nonzero[-1])
    return int(top_indices_sorted[local_idx])
