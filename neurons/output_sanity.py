"""Decoded-output sanity checks for verified inference responses.

The ZK proof pipeline verifies that committed output token IDs are
consistent with the model computation (commitments, GEMM sumcheck,
sampled decode positions).  It does NOT verify that those tokens decode
into a plausible visible assistant message, nor that the miner-reported
``output_tokens`` respects the requested ``max_new_tokens``.  A miner can
therefore stream a near-empty reply (e.g. ``"hmm,The"``) while reporting
thousands of output tokens, and the response still counts as a
successful verified generation for TPS, receipts, billing, and routing.

This module is the policy layer that closes that gap.  It is
deliberately NOT semantic grading — it only answers "did the claimed
generated tokens produce a plausible visible assistant message?"

Two check tiers:

``check_output_sanity``
    O(len(text)) string checks — always-on for organic traffic in the
    proxy and for every canary in the validator.  No tokenizer needed.

``check_committed_output_sanity``
    Canary-only (validator side): re-decodes the committed
    ``output_token_ids`` from the proof bundle with the model tokenizer
    and cross-checks against the streamed text.  Catches special-token
    spam inside a proof-valid token stream and stream/commit divergence.

Failure attribution:

Reasons prefixed with a tag in ``MINER_FAULT_REASONS`` can only be
produced by miner-side misbehaviour (e.g. reporting more output tokens
than the request allowed) and are safe to strike/probation on organic
traffic.  The visible-text ratio checks can in principle be induced by
an adversarial *user* prompt ("reply with 2000 blank lines"), so on the
organic path they only disqualify the response from billing/TPS/receipt
credit; enforcement with probation happens on canaries, whose prompts
the validator controls.

Env overrides (operator escape hatches):

``VERATHOS_OUTPUT_SANITY``            set to ``0`` to disable all checks
``VERATHOS_OUTPUT_SANITY_MIN_CHARS``  default 32
``VERATHOS_OUTPUT_SANITY_MIN_RATIO``  default 0.10 (chars per token)
"""
from __future__ import annotations

import os
from typing import List, Optional

# A generation of at least this many tokens with ZERO visible characters is
# malformed regardless of length — a genuine assistant reply of even a few
# tokens always renders some non-whitespace text.  Catches short empty/
# all-special-token responses (e.g. 29 committed tokens, nothing visible)
# that the higher-count checks below would miss.
EMPTY_VISIBLE_MIN_TOKENS = int(os.environ.get("VERATHOS_OUTPUT_SANITY_EMPTY_MIN", "8"))

# A response claiming at least this many output tokens must have some
# visible text at all.
MIN_TOKENS_FOR_VISIBLE_CHECK = 64
# ... and at least this many non-whitespace characters.
MIN_VISIBLE_CHARS = int(os.environ.get("VERATHOS_OUTPUT_SANITY_MIN_CHARS", "32"))

# Above this many claimed output tokens, visible chars per claimed token
# must exceed MIN_VISIBLE_RATIO.  Natural text is ~3-4 chars/token; CJK
# is ~1 char/token; 0.10 leaves a 10x margin below the densest honest
# tokenizer output.
RATIO_TOKENS_THRESHOLD = 256
MIN_VISIBLE_RATIO = float(os.environ.get("VERATHOS_OUTPUT_SANITY_MIN_RATIO", "0.10"))

# Reported output_tokens may exceed the requested max_new_tokens only by
# this multiplicative factor plus additive slop (covers off-by-a-few
# counting differences around stop tokens).  Honest vLLM never exceeds
# the requested budget.
OVERSHOOT_FACTOR = 1.05
OVERSHOOT_SLOP_TOKENS = 16

# Stream/commit divergence band for the canary cross-check: streamed
# visible chars and committed-decode visible chars must agree within 2x
# once either side is non-trivial.  Incremental detokenization on the
# miner and a one-shot decode on the validator differ by at most a few
# characters, never 2x.
DIVERGENCE_MIN_CHARS = 64
DIVERGENCE_FACTOR = 2.0

# Reason tags that are attributable ONLY to miner-side misbehaviour
# (cannot be induced by a user prompt).  Safe for organic strikes.
MINER_FAULT_REASONS = ("token_overshoot", "count_mismatch")


def _enabled() -> bool:
    return os.environ.get("VERATHOS_OUTPUT_SANITY", "1") != "0"


def streaming_overshoot_budget(max_new_tokens: Optional[int]) -> Optional[int]:
    """Token count above which a live stream is definitely overshooting.

    Used by the proxy to cut a streaming response off mid-flight once the
    miner has emitted more tokens than the request could honestly produce,
    so the garbage/inflation tail never reaches the user.  Returns ``None``
    when the guard is disabled or no budget was requested (no cutoff).
    """
    if not _enabled() or not max_new_tokens or max_new_tokens <= 0:
        return None
    return int(max_new_tokens * OVERSHOOT_FACTOR + OVERSHOOT_SLOP_TOKENS)


def visible_chars(text: Optional[str]) -> int:
    """Number of non-whitespace characters in ``text``."""
    if not text:
        return 0
    return sum(1 for ch in text if not ch.isspace())


def is_miner_fault(reason: Optional[str]) -> bool:
    """True when ``reason`` can only be caused by the miner (never the user)."""
    return bool(reason) and reason.startswith(MINER_FAULT_REASONS)


def check_output_sanity(
    output_tokens: int,
    text: Optional[str],
    max_new_tokens: Optional[int],
) -> Optional[str]:
    """Cheap always-on plausibility check on a finished generation.

    Args:
        output_tokens: miner-reported completion token count.
        text: the decoded assistant text as served to the user.
        max_new_tokens: the token budget the request asked for (skip the
            overshoot check when falsy).

    Returns:
        ``None`` when the response is plausible, else a short reason
        string (``tag: detail``).
    """
    if not _enabled():
        return None

    output_tokens = int(output_tokens or 0)

    if max_new_tokens and output_tokens > max_new_tokens * OVERSHOOT_FACTOR + OVERSHOOT_SLOP_TOKENS:
        return (
            f"token_overshoot: reported output_tokens={output_tokens} exceeds "
            f"requested max_new_tokens={max_new_tokens}"
        )

    vis = visible_chars(text)
    if output_tokens >= EMPTY_VISIBLE_MIN_TOKENS and vis == 0:
        return (
            f"empty_visible_output: output_tokens={output_tokens} but zero "
            f"visible chars"
        )
    if output_tokens >= MIN_TOKENS_FOR_VISIBLE_CHECK and vis < MIN_VISIBLE_CHARS:
        return (
            f"tiny_visible_output: output_tokens={output_tokens} but only "
            f"{vis} visible chars"
        )
    if output_tokens >= RATIO_TOKENS_THRESHOLD and vis < output_tokens * MIN_VISIBLE_RATIO:
        return (
            f"low_visible_ratio: {vis} visible chars for "
            f"output_tokens={output_tokens} ({vis / output_tokens:.3f} chars/token)"
        )
    return None


def check_reported_vs_committed_count(
    reported_output_tokens: int,
    committed_token_ids,
) -> Optional[str]:
    """Cheap count-only cross-check for the proxy's sampled-verify path.

    When the proof bundle has been deserialized (the ~10% of organic
    requests the proxy already verifies), the committed ``output_token_ids``
    are in hand.  The billed ``output_tokens`` is a *separate*, miner-
    reported field; this asserts the two agree so a miner can't inflate the
    billed count above what the proof actually attests to.  No tokenizer,
    no decode — one ``len()`` and a subtraction.
    """
    if not _enabled() or not committed_token_ids:
        return None
    n_committed = len(committed_token_ids)
    reported = int(reported_output_tokens or 0)
    if reported and abs(n_committed - reported) > OVERSHOOT_SLOP_TOKENS:
        return (
            f"count_mismatch: reported output_tokens={reported} but proof "
            f"commits {n_committed} token IDs"
        )
    return None


def check_reported_vs_commitment_count(reported_output_tokens: int, commitment) -> Optional[str]:
    """Bind the billed/scored count to ``commitment.output_token_count``.

    Unlike the proof bundle (deserialized only on the ~10% sampled-verify
    slice), the commitment is deserialized on EVERY request, so
    ``output_token_count`` is available 100% of the time.  It is also the
    exact value the ZK proof pressures when a request IS sampled — so billing
    on it (instead of the free-form ``done``-event ``output_tokens``) forces a
    miner to keep the committed count honest: it cannot predict which requests
    get proof-verified, and inflating the commitment risks a proof failure.

    Returns a ``count_mismatch`` reason (miner-fault) when the billed count
    diverges from the committed count beyond the slop, else ``None``.
    """
    if not _enabled() or commitment is None:
        return None
    committed = int(getattr(commitment, "output_token_count", 0) or 0)
    reported = int(reported_output_tokens or 0)
    if committed and reported and abs(committed - reported) > OVERSHOOT_SLOP_TOKENS:
        return (
            f"count_mismatch: reported output_tokens={reported} but commitment "
            f"output_token_count={committed}"
        )
    return None


def check_committed_output_sanity(
    tokenizer,
    output_token_ids: List[int],
    streamed_text: Optional[str],
    reported_output_tokens: int,
    max_new_tokens: Optional[int],
) -> Optional[str]:
    """Canary-time cross-check of the committed token IDs.

    Re-decodes the proof bundle's ``output_token_ids`` (the sequence the
    proof actually attests to) and validates that (1) the committed
    count matches the miner-reported usage, (2) the committed sequence
    respects the requested budget, (3) it decodes to a plausible visible
    message, and (4) it agrees with what was streamed.

    Returns ``None`` when plausible, else a short reason string.
    """
    if not _enabled():
        return None
    if not output_token_ids:
        return None

    n_committed = len(output_token_ids)
    reported = int(reported_output_tokens or 0)

    # Miner-reported usage must match what the proof attests to.
    if reported and abs(n_committed - reported) > OVERSHOOT_SLOP_TOKENS:
        return (
            f"count_mismatch: reported output_tokens={reported} but proof "
            f"commits {n_committed} token IDs"
        )

    if max_new_tokens and n_committed > max_new_tokens * OVERSHOOT_FACTOR + OVERSHOOT_SLOP_TOKENS:
        return (
            f"token_overshoot: committed {n_committed} token IDs exceeds "
            f"requested max_new_tokens={max_new_tokens}"
        )

    try:
        decoded = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    except Exception:
        # Validator-side decode problem — never attribute to the miner.
        return None

    reason = check_output_sanity(n_committed, decoded, max_new_tokens=None)
    if reason is not None:
        return f"committed_{reason}"

    # Stream/commit divergence: the streamed text must be the
    # detokenization of the committed IDs (modulo incremental-decode
    # edge cases, which are a few characters at most).
    svis = visible_chars(streamed_text)
    dvis = visible_chars(decoded)
    if max(svis, dvis) >= DIVERGENCE_MIN_CHARS:
        lo, hi = min(svis, dvis), max(svis, dvis)
        if lo * DIVERGENCE_FACTOR < hi:
            return (
                f"stream_commit_divergence: streamed {svis} visible chars but "
                f"committed token IDs decode to {dvis} visible chars"
            )
    return None
