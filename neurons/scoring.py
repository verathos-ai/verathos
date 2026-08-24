"""Epoch-based composite scoring for the Verathos Bittensor subnet.

Weight composition:

    WEIGHT(uid) = logical_model_share × normalize_within_logical_model(ENTRY_EMA)

    LOGICAL_MODEL_SHARE = normalize(max(BASE_UTILITY × DEMAND_BONUS))
                  across approved logical models.

    Approved quantized variants are eligible inside their logical model's
    shared bucket, but they do not receive reserved sub-buckets. Empty logical
    model buckets are left unallocated so the validator can burn that share
    instead of redistributing it to crowded models.

    ENTRY_EMA   = exponential moving average of epoch_scores:
                  - No receipts at all          → None (neutral, keep EMA)
                  - Receipts present + integrity → UTILITY × WORK_FACTOR × TTFT × SPEED × DEMAND_BONUS
                  - Receipt integrity failure    → 0.0 (hard penalty, decays EMA)

    UTILITY     = log2(Q)^1.8 × log2(ctx/1K) × Qq × Gq
                  Matches verallm/registry/models.py utility formula.

    WORK_FACTOR  = 0.09 × min(3, 1 + (trusted_organic_tokens / 25K)²)
                  Every eligible endpoint gets the same nonzero base. Canary
                  token counts never affect score; they establish integrity
                  and provide performance samples. Only organic receipts from
                  the subnet scoring authority add throughput credit.
                  Output weighted 3× input — decode is sequential, prefill is parallel.
                  Sybil defense: N UIDs splitting fixed demand each get 1/N tokens,
                  score per UID = (1/N)², total = N × (1/N)² = 1/N of honest.
                  Per-receipt organic output is capped at ORGANIC_OUTPUT_CAP
                  (matches the proxy default). This is defense
                  in depth for legacy/imported receipts; live clients reject
                  counts that exceed the request or disagree with proof data.

    TTFT_FACTOR = min(1.3, sqrt(model_median_ttft / miner_median_ttft))
                  Peer-relative: compares this miner's median TTFT to the median
                  across all miners serving the same model.
                  Soft cap at 1.3 (lowered from 2.0 to bound fast-hardware uplift):
                  1.7× faster → 1.30× (cap), honest peer → 1.00,
                  2× slower → 0.71×, 4× slower → 0.50×.

    SPEED_FACTOR = min(1.3, sqrt(miner_median_tps / model_median_tps))
                  Peer-relative: compares this miner's median decode speed (tok/s)
                  to the median across all miners serving the same model.
                  Soft cap at 1.3 (lowered from 2.0).  Combined ttft × speed
                  uplift is capped at 1.69×, instead of the prior 4×.

    DEMAND_BONUS  = 1.0 + (demand_bps / 10000) × demand_bonus_max
                  Per-model demand signal from organic (non-canary) traffic.
                  Range: 1.0 (no demand data) to 1.0 + demand_bonus_max.

Performance samples use one median per permitted validator, then one median
per miner, then the model-level median across miners. This keeps TPS/TTFT
decentralized while preventing receipt-count flooding from creating influence.
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import bittensor as bt
import os
import struct
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

from neurons.receipts import ServiceReceipt
from verallm.proof_v3.canary_policy import canary_prompt_token_tolerance_v3

logger = logging.getLogger(__name__)

# Quant quality factors — aligned with verallm/registry/models.py QUANT_QUALITY.
# Imported at module level to avoid circular imports with the registry.
QUANT_QUALITY: dict[str, float] = {
    "bf16": 1.0, "fp16": 1.0,
    "fp8": 0.98, "int8": 0.95,
    "int4": 0.90, "nf4": 0.90, "nvfp4": 0.92,
}

# Performance is decentralized, but one permitted signer must never be able to
# move an endpoint's score alone. Three independent signer medians make the
# middle observation robust to one arbitrary outlier. With fewer, performance
# factors stay neutral while integrity and fixed-base eligibility still apply.
MIN_PERFORMANCE_VALIDATORS = 3


@dataclass
class EpochOutcome:
    """Result of an epoch for a single miner-model entry.

    Built from the receipt batch pulled from the miner at epoch boundary.

    Scoring rules:
        - Proof verification failure → 0.0 (hard penalty, triggers probation)
        - Receipt integrity failure (own receipts missing) → 0.0 (hard penalty)
        - No receipts at all (unreachable or new miner) → None (EMA unchanged)
        - Receipts present → scored from median throughput + latency
    """

    miner_address: str
    model_id: str
    model_index: int
    # Bittensor identity (filled at construction time so log messages can
    # use the operator-recognizable UID and SS58 hotkey instead of the
    # internal EVM address).  Defaults preserve backwards-compat for
    # callers that don't have this info.
    uid: int = -1
    hotkey_ss58: str = ""

    # Receipts from THIS validator for this miner-model
    own_receipts: List[ServiceReceipt] = field(default_factory=list)
    # Expected number of own receipts (canaries sent by this validator)
    expected_own_receipt_count: int = 0
    # Exact signed v4 obligations: id -> (kind, target prompt tokens).
    expected_canary_obligations: Dict[bytes, Tuple[str, int]] = field(
        default_factory=dict
    )
    # Obligations this validator ALSO scheduled for the same epoch/entry but
    # that are no longer part of the active plan (a mid-epoch validator
    # restart re-plans the epoch under a fresh secret salt; obligations of
    # the discarded plan may already have produced signed receipts). A
    # receipt matching one of these is validator-produced residue: it is
    # validated but never scored and never treated as a forgery. Receipt
    # count cannot inflate score — scoring uses the expected obligation
    # inventory, not the receipt count.
    superseded_canary_obligations: Dict[bytes, Tuple[str, int]] = field(
        default_factory=dict
    )

    # ALL receipts for this miner-model (from all validators)
    all_receipts: List[ServiceReceipt] = field(default_factory=list)
    # Epoch-latched scoring-authority receipts. ``None`` preserves the
    # standalone scoring API for historical/offline callers; production
    # validators always provide an explicitly filtered list (possibly empty).
    scoring_receipts: Optional[List[ServiceReceipt]] = None

    # Proof verification outcomes (from own receipts where proof_requested=True)
    proof_tests: int = 0  # number of receipts where proof was requested
    proof_failures: int = 0  # number where proof was requested but failed
    # Raw proof failures remain recorded above.  This bit is the separately
    # configured operational decision for the current source epoch.
    proof_failure_penalty_required: bool = True
    # Failed hard receipts granted a neutral first strike must not make the
    # exact obligation inventory malformed, and must not earn throughput.
    neutral_hard_obligation_ids: Set[bytes] = field(default_factory=set)

    # TEE attestation outcomes (from own receipts where tee_attestation_verified is set)
    tee_tests: int = 0
    tee_failures: int = 0
    tee_verified: bool = False  # at least one TEE attestation passed this epoch

    # Model entry metadata (for scoring)
    max_context_len: int = 0
    quant: str = ""
    quant_qualified: bool = True

    # 503 busy-skip count this epoch (audit trail for load-aware forgiveness)
    busy_skip_count: int = 0


@dataclass
class ModelEntryScore:
    """Per-entry score state tracked across epochs."""

    model_id: str
    model_index: int
    ema_score: float = 0.0
    total_epochs: int = 0
    scored_epochs: int = 0


@dataclass
class MinerScoreState:
    """Rolling score state for a single miner UID."""

    uid: int
    address: str
    entries: Dict[int, ModelEntryScore] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> float:
        """Sum of all entry EMAs — additive multi-model aggregation."""
        return sum(e.ema_score for e in self.entries.values())


@dataclass
class PeerMedians:
    """Per-model peer performance medians computed from all epoch receipts."""

    median_ttft_ms: float  # median TTFT across all miners serving this model
    median_tps: float  # median decode speed across all miners serving this model


def compute_model_base_utility(
    active_params_b: float,
    max_context_len: int,
    quant: str,
    moe_dense_equivalent: float = 0.0,
    generation_quality: float = 1.0,
) -> float:
    """Compute the model-level utility used by scoring and model buckets."""
    quality_params = (
        moe_dense_equivalent if moe_dense_equivalent > 0 else active_params_b
    )
    utility = math.log2(max(quality_params, 1.0)) ** 1.8
    ctx_value = math.log2(max(max_context_len / 1024, 1))
    quant_q = QUANT_QUALITY.get(quant)
    if quant_q is None and quant.startswith("gguf_"):
        # Mesh profiles carry the exact GGUF scheme (gguf_iq2_m, ...): a
        # q4_k_m and an iq2_m of the same model must not score the same
        # quality. The per-scheme table lives with the catalogue.
        from verallm.registry.models import gguf_quant_quality

        quant_q = gguf_quant_quality(quant[len("gguf_"):])
    if quant_q is None:
        quant_q = 0.80
    return utility * ctx_value * quant_q * generation_quality


def compute_ttft_factor(
    miner_median_ttft_ms: float,
    model_median_ttft_ms: float,
) -> float:
    """Peer-relative TTFT factor: reward fast miners, penalize slow ones.

    TTFT_FACTOR = sqrt(model_median / miner_median)

    Faster miners earn a bonus:
        1.7× faster than median: 1.30× (cap)
        2× faster:               1.30× (cap)
    Slower miners penalized:
        2× slower than median:   0.71×
        4× slower:               0.50×

    Soft cap at 1.3 (lowered from 2.0 to bound the fast-hardware uplift —
    a single B-class GPU otherwise dominates a 4090-tier peer pool by 2×
    on this factor alone, which compounds with speed_factor and throughput²).

    Returns 1.0 if either value is <= 0 (no data / single miner).
    """
    if miner_median_ttft_ms <= 0 or model_median_ttft_ms <= 0:
        return 1.0
    return min(1.3, math.sqrt(model_median_ttft_ms / miner_median_ttft_ms))


def compute_speed_factor(
    miner_median_tps: float,
    model_median_tps: float,
) -> float:
    """Peer-relative decode speed factor: reward fast miners, penalize slow ones.

    SPEED_FACTOR = sqrt(miner_tps / model_median)

    Faster miners earn a bonus:
        1.7× faster than median: 1.30× (cap)
        24× faster:              1.30× (cap)
    Slower miners penalized:
        2× slower than median:   0.71×
        4× slower:               0.50×

    Soft cap at 1.3 (lowered from 2.0).  Combined with ttft_factor (also
    capped at 1.3) the total latency-side uplift is bounded at 1.69×.

    Returns 1.0 if either value is <= 0 (no data / single miner).
    """
    if miner_median_tps <= 0 or model_median_tps <= 0:
        return 1.0
    return min(1.3, math.sqrt(miner_median_tps / model_median_tps))


def _median_of_validator_medians(
    receipts: List[ServiceReceipt],
    attribute: str,
    *,
    min_validators: int = MIN_PERFORMANCE_VALIDATORS,
) -> float:
    """Return one equally weighted performance sample per receipt signer."""

    per_validator: Dict[bytes, List[float]] = {}
    for receipt in receipts:
        if receipt.proof_requested and not receipt.proof_verified:
            continue
        value = float(getattr(receipt, attribute, 0.0) or 0.0)
        if value <= 0:
            continue
        signer = bytes(getattr(receipt, "validator_hotkey", b"") or b"")
        if len(signer) != 32:
            continue
        per_validator.setdefault(signer, []).append(value)
    if len(per_validator) < max(1, int(min_validators)):
        return 0.0
    return _median([_median(values) for values in per_validator.values()])


def select_scoring_authority_receipts(
    receipts: List[ServiceReceipt],
    authority_hotkey: bytes,
) -> List[ServiceReceipt]:
    """Select receipts signed by one exact 32-byte scoring authority."""

    authority = bytes(authority_hotkey or b"")
    if len(authority) != 32:
        return []
    return [
        receipt
        for receipt in receipts
        if bytes(getattr(receipt, "validator_hotkey", b"") or b"")
        == authority
    ]


def _receipt_matches_canary_obligation(
    receipt: ServiceReceipt,
    expected: Tuple[str, int],
) -> bool:
    """Check one signed canary receipt against one planned obligation.

    Kind, exact target and the materializer's prompt-token tolerance must
    all agree — a receipt claiming an obligation with mismatched geometry
    is misattributed or forged and must never be excused.
    """

    expected_kind, expected_target = expected
    actual_target = int(
        getattr(receipt, "canary_target_prompt_tokens", 0) or 0
    )
    prompt_tokens = int(receipt.prompt_tokens or 0)
    tolerance = canary_prompt_token_tolerance_v3(int(expected_target))
    return (
        str(getattr(receipt, "canary_kind", "") or "") == expected_kind
        and actual_target == int(expected_target)
        and 0 < prompt_tokens <= int(expected_target)
        and int(expected_target) - prompt_tokens <= tolerance
    )


def compute_epoch_entry_score(
    outcome: EpochOutcome,
    active_params_b: float,
    moe_dense_equivalent: float = 0.0,
    generation_quality: float = 1.0,
    throughput_power: float = 2.0,
    peer_medians: Optional[PeerMedians] = None,
    tee_bonus: float = 1.0,
    demand_bonus: float = 1.0,
    suppress_hard_failures: bool = False,
) -> Optional[float]:
    """Compute a single entry's score for one epoch from receipt data.

    Returns:
        float score, None if no receipts (EMA unchanged), or 0.0 for
        integrity failure (hard penalty).
    """
    # Receipt integrity check: all own receipts must be present
    # Build a stable, operator-recognizable miner identifier for logs.
    # Prefer UID + SS58 hotkey (Bittensor-native) and only fall back to
    # the internal EVM address when SS58 isn't available yet.
    if outcome.uid >= 0 and outcome.hotkey_ss58:
        _who = f"UID {outcome.uid} {outcome.hotkey_ss58}"
    elif outcome.uid >= 0:
        _who = f"UID {outcome.uid}"
    else:
        _who = outcome.miner_address[:10]

    if not outcome.quant_qualified:
        action = "score update suppressed" if suppress_hard_failures else "score=0"
        bt.logging.info(
            f"Unqualified proof-v3 quantization for {_who} "
            f"model_index={outcome.model_index} -> {action}"
        )
        return None if suppress_hard_failures else 0.0

    expected_obligations = dict(outcome.expected_canary_obligations)
    if expected_obligations:
        superseded_obligations = dict(
            getattr(outcome, "superseded_canary_obligations", {}) or {}
        )
        seen: Dict[bytes, ServiceReceipt] = {}
        invalid = False
        for receipt in outcome.own_receipts:
            if not receipt.is_canary:
                continue
            obligation_id = bytes(
                getattr(receipt, "canary_obligation_id", b"") or b""
            )
            if obligation_id in outcome.neutral_hard_obligation_ids:
                # Main's neutral-hard scoping: obligations the validator
                # itself neutralized must not zero the miner.
                continue
            if int(getattr(receipt, "receipt_version", 1) or 1) < 4:
                invalid = True
                continue
            expected_item = expected_obligations.get(obligation_id)
            if expected_item is None:
                superseded_item = superseded_obligations.get(obligation_id)
                if superseded_item is not None and (
                    _receipt_matches_canary_obligation(
                        receipt, superseded_item
                    )
                ):
                    # Benign validator-side residue: this validator scheduled
                    # and verified the test under an earlier plan of the same
                    # epoch (mid-epoch restart re-plan). Ignoring it cannot
                    # inflate score; zeroing it would fail an honest serve.
                    continue
                # A receipt for a test never scheduled in this epoch is a
                # forged or misattributed receipt -> hard integrity failure.
                invalid = True
                continue
            if not _receipt_matches_canary_obligation(receipt, expected_item):
                invalid = True
                continue
            if obligation_id in seen:
                # Benign duplicate of an already-satisfied obligation
                # (retry/dispatch duplication, e.g. across a mid-epoch mesh
                # roll). The obligation is fulfilled and count inflation
                # cannot increase score, so the duplicate is ignored. A
                # duplicate with mismatched geometry fails the check above
                # and still zeroes.
                continue
            seen[obligation_id] = receipt
        missing = set(expected_obligations).difference(seen)
        if invalid or missing:
            action = (
                "score update suppressed"
                if suppress_hard_failures
                else "score=0"
            )
            bt.logging.info(
                f"Canary receipt integrity failure for {_who} "
                f"model_index={outcome.model_index}: "
                f"expected={len(expected_obligations)} valid={len(seen)} "
                f"missing={len(missing)} invalid={invalid} -> {action}"
            )
            return None if suppress_hard_failures else 0.0
    elif outcome.expected_own_receipt_count > 0:
        # Compatibility for historical callers without an obligation
        # inventory: exact count, with no one-receipt slop.
        own_count = len(outcome.own_receipts)
        if own_count < outcome.expected_own_receipt_count:
            action = "score update suppressed" if suppress_hard_failures else "score=0"
            bt.logging.info(
                f"Receipt integrity failure for {_who} "
                f"model_index={outcome.model_index}: expected "
                f"{outcome.expected_own_receipt_count} own canary receipts, "
                f"found {own_count} -> {action}"
            )
            return None if suppress_hard_failures else 0.0

    # Proof verification failure: any failed proof -> hard penalty.
    # Use INFO not WARNING — the validator did its job correctly, the
    # miner is the one with the problem.  Operators don't need to be
    # paged for cheating / faulty miners.
    if (
        outcome.proof_tests > 0
        and outcome.proof_failures > 0
        and outcome.proof_failure_penalty_required
    ):
        action = "score update suppressed" if suppress_hard_failures else "score=0"
        bt.logging.info(f"Proof verification failure for {_who} model_index={outcome.model_index}: {outcome.proof_failures}/{outcome.proof_tests} proofs failed -> {action}")
        return None if suppress_hard_failures else 0.0

    # TEE attestation failure: any failed attestation -> hard penalty.
    # Same rationale: miner-side failure, not a validator issue.
    if outcome.tee_tests > 0 and outcome.tee_failures > 0:
        action = "score update suppressed" if suppress_hard_failures else "score=0"
        bt.logging.info(f"TEE attestation failure for {_who} model_index={outcome.model_index}: {outcome.tee_failures}/{outcome.tee_tests} attestations failed -> {action}")
        return None if suppress_hard_failures else 0.0

    # No receipts at all -> not tested / unreachable (EMA unchanged)
    if not outcome.all_receipts:
        return None

    # Organic work is additive to a fixed nonzero eligibility baseline.
    # Output tokens are weighted 3× input because decode is sequential while
    # prefill is parallel. Canary token counts never enter this calculation:
    # their hidden schedule and request geometry are security parameters, not
    # demand, and must not create random score variance.
    #
    # Per-receipt output cap is defense in depth for legacy/imported receipts.
    # Live validator clients reject counts that exceed max_new_tokens or do not
    # match the commitment and proof bundle. Organic receipts are capped at the
    # proxy default max_new_tokens.
    OUTPUT_WEIGHT = 3
    ORGANIC_OUTPUT_CAP = 4096     # matches PlaintextChatRequest.max_new_tokens default
    ORGANIC_PROMPT_CAP = 4096
    ORGANIC_PROMPT_TO_OUTPUT_CAP = 8
    IDLE_BASELINE_WORK_SCORE = 0.09
    ORGANIC_REFERENCE_WEIGHTED_TOKENS = 25_000
    MAX_WORK_FACTOR_MULTIPLIER = 3.0

    # Production validators explicitly provide only receipts signed by the
    # epoch-latched scoring authority. ``None`` is retained for standalone and
    # historical callers, where the supplied receipt set is already the full
    # scoring input.
    traffic_receipts = (
        outcome.all_receipts
        if outcome.scoring_receipts is None
        else outcome.scoring_receipts
    )
    organic_tokens = 0
    for r in traffic_receipts:
        if r.tokens_generated <= 0:
            continue
        if r.proof_requested and not r.proof_verified:
            continue
        if r.is_canary:
            continue
        output_tokens = min(max(0, r.tokens_generated), ORGANIC_OUTPUT_CAP)
        prompt_tokens = min(
            max(0, r.prompt_tokens),
            ORGANIC_PROMPT_CAP,
            ORGANIC_PROMPT_TO_OUTPUT_CAP * output_tokens,
        )
        weighted = OUTPUT_WEIGHT * output_tokens + prompt_tokens
        organic_tokens += weighted

    # Extract latency: split canary vs organic, take WORSE (max) of the two
    # validator-balanced medians when both are statistically meaningful. Each
    # permitted signer contributes one median regardless of its receipt count,
    # preventing one validator from flooding the performance sample. This neutralizes any
    # canary preferential treatment (prefill-cache, load-shedding, dedicated
    # GPU on canary requests, etc.) — a miner cannot benefit from making
    # canaries artificially faster than organic.  Falls back to whichever
    # population has three independent signers. Fewer signers are neutral.
    organic_median_ttft = _median_of_validator_medians(
        [r for r in outcome.all_receipts if not r.is_canary],
        "ttft_ms",
    )
    canary_median_ttft = _median_of_validator_medians(
        [r for r in outcome.all_receipts if r.is_canary],
        "ttft_ms",
    )
    if organic_median_ttft > 0 and canary_median_ttft > 0:
        median_ttft = max(
            organic_median_ttft,
            canary_median_ttft,
        )
    elif canary_median_ttft > 0:
        median_ttft = canary_median_ttft
    elif organic_median_ttft > 0:
        median_ttft = organic_median_ttft
    else:
        median_ttft = 0

    # UTILITY (unchanged formula)
    base_utility = compute_model_base_utility(
        active_params_b=active_params_b,
        max_context_len=outcome.max_context_len,
        quant=outcome.quant,
        moe_dense_equivalent=moe_dense_equivalent,
        generation_quality=generation_quality,
    )

    # WORK_SCORE = fixed eligible base + authenticated organic throughput².
    # The additive form means zero organic demand remains nonzero while the
    # quadratic organic term retains the existing split-demand Sybil defense.
    throughput_score = IDLE_BASELINE_WORK_SCORE * min(
        MAX_WORK_FACTOR_MULTIPLIER,
        1.0
        + (organic_tokens / ORGANIC_REFERENCE_WEIGHTED_TOKENS)
        ** throughput_power,
    )

    # TTFT_FACTOR (peer-relative, capped at 1.3)
    # Faster than peer median → bonus (up to 1.3×).
    # Slower than peer median → penalty (sqrt curve).
    # No peers → 1.0 (no comparison possible).
    ttft_factor = 1.0
    if peer_medians is not None and median_ttft > 0:
        ttft_factor = compute_ttft_factor(median_ttft, peer_medians.median_ttft_ms)

    # SPEED_FACTOR (peer-relative, capped at 1.3)
    # Split canary vs organic, take WORSE (min) of the two medians when both
    # are statistically meaningful.  Same defense as the TTFT split above —
    # any canary speed advantage is neutralized; honest hardware speed (the
    # min of the two) drives the score.
    speed_factor = 1.0
    if peer_medians is not None:
        organic_median_tps = _median_of_validator_medians(
            [r for r in outcome.all_receipts if not r.is_canary],
            "tokens_per_sec",
        )
        canary_median_tps = _median_of_validator_medians(
            [r for r in outcome.all_receipts if r.is_canary],
            "tokens_per_sec",
        )
        if organic_median_tps > 0 and canary_median_tps > 0:
            miner_median_tps = min(
                organic_median_tps,
                canary_median_tps,
            )
        elif canary_median_tps > 0:
            miner_median_tps = canary_median_tps
        elif organic_median_tps > 0:
            miner_median_tps = organic_median_tps
        else:
            miner_median_tps = 0
        if miner_median_tps > 0:
            speed_factor = compute_speed_factor(
                miner_median_tps, peer_medians.median_tps,
            )

    # TEE bonus: reward miners with verified TEE attestation
    tee_multiplier = tee_bonus if outcome.tee_verified else 1.0

    return base_utility * throughput_score * ttft_factor * speed_factor * demand_bonus * tee_multiplier


def compute_traffic_volume(
    all_receipts: List[ServiceReceipt],
    epoch_number: int,
) -> float:
    """Compute traffic volume multiplier from epoch receipts.

    Volume only — quality (tok/s) is already captured in throughput² scoring.
    Range: 1.0 (no traffic) to 1.5 (heavy traffic).
    """
    # Same proof-failure filter as compute_epoch_entry_score.
    valid = [
        r for r in all_receipts
        if r.epoch_number == epoch_number
        and not (r.proof_requested and not r.proof_verified)
    ]
    if not valid:
        return 1.0

    total_tokens = sum(r.tokens_generated + r.prompt_tokens for r in valid)

    # Log scale, diminishing returns (cap 0.5)
    # 1K tok → 1.0, 10K → 1.08, 100K → 1.17, 1M → 1.25, 10M → 1.33
    volume_bonus = min(math.log2(max(total_tokens / 1000, 1)) * 0.025, 0.5)

    return 1.0 + volume_bonus


def compute_model_demand(
    all_receipts: List[ServiceReceipt],
    epoch_number: int,
) -> Dict[str, int]:
    """Compute per-model demand scores from organic (non-canary) traffic.

    Demand metric: ``sqrt(request_count * generated_tokens)`` (geometric mean).
    Normalized to 0-10000 basis points (max model = 10000).

    Args:
        all_receipts: All receipts from all miners for the epoch.
        epoch_number: Current epoch number.

    Returns:
        Dict mapping model_id to demand score in basis points (0-10000).
    """
    # Filter to organic traffic for this epoch
    organic = [
        r for r in all_receipts
        if r.epoch_number == epoch_number
        and not r.is_canary
        and r.tokens_generated > 0
        and not (r.proof_requested and not r.proof_verified)
    ]
    if not organic:
        return {}

    # Aggregate per model: request count + total generated tokens
    model_stats: Dict[str, list] = {}  # model_id -> [count, tokens]
    for r in organic:
        if r.model_id not in model_stats:
            model_stats[r.model_id] = [0, 0]
        model_stats[r.model_id][0] += 1
        model_stats[r.model_id][1] += r.tokens_generated

    # Raw score = sqrt(count * tokens)
    raw_scores: Dict[str, float] = {}
    for model_id, (count, tokens) in model_stats.items():
        raw_scores[model_id] = math.sqrt(count * tokens)

    # Normalize: max model = 10000 bps
    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    if max_raw <= 0:
        return {mid: 0 for mid in raw_scores}

    return {
        mid: int(score / max_raw * 10000)
        for mid, score in raw_scores.items()
    }


def compute_demand_bonus(
    demand_score_bps: int,
    demand_bonus_max: float = 0.20,
) -> float:
    """Convert a demand score (basis points) to a scoring multiplier.

    Linear mapping: 0 bps → 1.0, 10000 bps → 1.0 + demand_bonus_max.

    Args:
        demand_score_bps: Demand score in basis points (0-10000).
        demand_bonus_max: Maximum bonus fraction (default 0.20 = 20%).

    Returns:
        Multiplier in range [1.0, 1.0 + demand_bonus_max].
    """
    clamped = max(0, min(demand_score_bps, 10000))
    return 1.0 + (clamped / 10000) * demand_bonus_max


def _median(values: List[float]) -> float:
    """Compute median of a list of floats. Returns 0.0 if empty."""
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def compute_peer_medians(
    all_receipts: List[ServiceReceipt],
    epoch_number: int,
) -> Dict[str, PeerMedians]:
    """Compute per-model median TTFT and decode speed from all epoch receipts.

    Uses median-of-validator-medians inside each miner, then a median across
    miners. Receipt count therefore cannot give one permitted validator or one
    high-traffic miner disproportionate influence over the reference.

    Args:
        all_receipts: All receipts from all miners for the epoch.
        epoch_number: Current epoch number.

    Returns:
        Dict mapping model_id to PeerMedians.
    """
    # Group by model -> miner -> validator -> ([ttft], [tps]).
    model_miner_stats: Dict[
        str,
        Dict[str, Dict[bytes, Tuple[List[float], List[float]]]],
    ] = {}

    for r in all_receipts:
        if r.epoch_number != epoch_number:
            continue
        if r.proof_requested and not r.proof_verified:
            continue
        signer = bytes(getattr(r, "validator_hotkey", b"") or b"")
        if len(signer) != 32:
            continue
        model_miners = model_miner_stats.setdefault(r.model_id, {})
        miner_validators = model_miners.setdefault(r.miner_address, {})
        ttft_list, tps_list = miner_validators.setdefault(
            signer,
            ([], []),
        )
        if r.ttft_ms > 0:
            ttft_list.append(r.ttft_ms)
        if r.tokens_per_sec > 0:
            tps_list.append(r.tokens_per_sec)

    result: Dict[str, PeerMedians] = {}
    for model_id, miners in model_miner_stats.items():
        # Per-miner medians
        miner_ttft_medians: List[float] = []
        miner_tps_medians: List[float] = []

        for _addr, validators in miners.items():
            ttft_signer_medians = [
                _median(ttft_vals)
                for ttft_vals, _tps_vals in validators.values()
                if ttft_vals
            ]
            ttft_med = (
                _median(ttft_signer_medians)
                if len(ttft_signer_medians) >= MIN_PERFORMANCE_VALIDATORS
                else 0.0
            )
            if ttft_med > 0:
                miner_ttft_medians.append(ttft_med)
            tps_signer_medians = [
                _median(tps_vals)
                for _ttft_vals, tps_vals in validators.values()
                if tps_vals
            ]
            tps_med = (
                _median(tps_signer_medians)
                if len(tps_signer_medians) >= MIN_PERFORMANCE_VALIDATORS
                else 0.0
            )
            if tps_med > 0:
                miner_tps_medians.append(tps_med)

        # Model-level median = median of miner medians
        result[model_id] = PeerMedians(
            median_ttft_ms=_median(miner_ttft_medians),
            median_tps=_median(miner_tps_medians),
        )

    return result


def compute_receipt_set_hash(
    all_receipts: List[ServiceReceipt],
    epoch_number: int,
) -> bytes:
    """Compute a deterministic hash over non-canary receipts for tamper evidence.

    All validators pulling the same receipt set will produce the same hash.
    Receipts are sorted by (miner_address, model_id, timestamp) for determinism.

    Args:
        all_receipts: All receipts from all miners for the epoch.
        epoch_number: Current epoch number.

    Returns:
        32-byte SHA256 digest.
    """
    organic = [
        r for r in all_receipts
        if r.epoch_number == epoch_number and not r.is_canary
    ]
    organic.sort(key=lambda r: (r.miner_address, r.model_id, r.timestamp))

    h = hashlib.sha256()
    h.update(b"VERATHOS_RECEIPT_SET_V1")
    h.update(struct.pack(">q", epoch_number))
    for r in organic:
        h.update(r.miner_address.encode("utf-8"))
        h.update(r.model_id.encode("utf-8"))
        h.update(struct.pack(">q", r.timestamp))
        h.update(struct.pack(">q", r.tokens_generated))
        h.update(struct.pack(">d", r.tokens_per_sec))
    return h.digest()


class CompositeScorer:
    """Aggregates epoch outcomes into per-UID scores for Bittensor weight-setting.

    Per-entry EMA tracking with additive multi-model aggregation. Organic work
    credit is already included in each epoch score.
    """

    def __init__(
        self,
        ema_alpha: float = 0.2,
        throughput_power: float = 2.0,
    ):
        self.ema_alpha = ema_alpha
        self.throughput_power = throughput_power
        self.states: Dict[int, MinerScoreState] = {}  # uid -> state

    def update(
        self,
        uid: int,
        address: str,
        model_index: int,
        outcome: EpochOutcome,
        active_params_b: float,
        moe_dense_equivalent: float = 0.0,
        generation_quality: float = 1.0,
        demand_bonus: float = 1.0,
        peer_medians: Optional[PeerMedians] = None,
        tee_bonus: float = 1.0,
        suppress_hard_failures: bool = False,
    ) -> Optional[float]:
        """Update score for a single entry based on an epoch outcome.

        Returns the entry's epoch score, or None if no data.
        """
        if uid not in self.states:
            self.states[uid] = MinerScoreState(uid=uid, address=address)

        state = self.states[uid]
        if state.address.lower() != address.lower():
            bt.logging.info(
                f"UID {uid} EVM address changed {state.address[:10]} -> "
                f"{address[:10]}; resetting stale score state"
            )
            state = MinerScoreState(uid=uid, address=address)
            self.states[uid] = state

        # Ensure entry exists
        if model_index not in state.entries:
            state.entries[model_index] = ModelEntryScore(
                model_id=outcome.model_id,
                model_index=model_index,
            )

        entry = state.entries[model_index]
        if entry.model_id != outcome.model_id:
            entry.model_id = outcome.model_id
            entry.ema_score = 0.0
            entry.total_epochs = 0
            entry.scored_epochs = 0

        epoch_score = compute_epoch_entry_score(
            outcome,
            active_params_b=active_params_b,
            moe_dense_equivalent=moe_dense_equivalent,
            generation_quality=generation_quality,
            throughput_power=self.throughput_power,
            peer_medians=peer_medians,
            tee_bonus=tee_bonus,
            demand_bonus=demand_bonus,
            suppress_hard_failures=suppress_hard_failures,
        )

        self._update_entry_ema(entry, epoch_score)
        return epoch_score

    def apply_traffic_volume(
        self,
        uid: int,
        all_receipts: List[ServiceReceipt],
        epoch_number: int,
    ) -> float:
        """Compute and store the traffic volume multiplier for a UID.

        Called once per epoch with all receipts from the miner.
        Returns the multiplier.
        """
        volume = compute_traffic_volume(all_receipts, epoch_number)
        if uid in self.states:
            self.states[uid]._traffic_volume = volume
        return volume

    @staticmethod
    def _excluded_entry_keys(
        excluded_entries: Optional[Set[Tuple[str, int]]],
    ) -> Set[Tuple[str, int]]:
        return {
            (str(address).lower(), int(model_index))
            for address, model_index in (excluded_entries or set())
        }

    def get_weights(
        self,
        *,
        excluded_entries: Optional[Set[Tuple[str, int]]] = None,
    ) -> Dict[int, float]:
        """Get normalized weights for all UIDs.

        WEIGHT(uid) = normalize( AGGREGATE )

        Authenticated organic work is already captured by the work factor.
        No separate volume multiplier is applied. Excluded
        entries retain their EMA history but contribute no emission while an
        external policy gate such as probation is active.
        """
        excluded = self._excluded_entry_keys(excluded_entries)
        raw = {}
        for uid, state in self.states.items():
            address = state.address.lower()
            score = sum(
                entry.ema_score
                for entry in state.entries.values()
                if (address, entry.model_index) not in excluded
            )
            # Floor dust-level scores to zero — prevents negligible weights
            # from persisting for miners that left the network long ago.
            if score < 1e-6:
                score = 0.0
            raw[uid] = score

        total = sum(raw.values())
        if total <= 0:
            return {uid: 0.0 for uid in raw}
        return {uid: s / total for uid, s in raw.items()}

    def get_model_bucket_weights(
        self,
        model_budgets: Dict[str, float],
        model_groups: Optional[Dict[str, str]] = None,
        group_budgets: Optional[Dict[str, float]] = None,
        *,
        excluded_entries: Optional[Set[Tuple[str, int]]] = None,
    ) -> Tuple[Dict[int, float], float]:
        """Get UID weights with approved logical-model buckets.

        ``model_budgets`` is a raw, positive budget per approved model. The
        method normalizes logical-model groups, then distributes each group
        among UIDs with positive EMA for any approved variant in that group.
        Approved quantized variants are eligible but do not get reserved
        sub-buckets. If a logical group has no positive-scoring endpoints,
        its share is returned as ``unallocated`` so the validator can burn it.
        """
        weights: Dict[int, float] = {uid: 0.0 for uid in self.states}
        budgets = {
            model_id: float(value)
            for model_id, value in model_budgets.items()
            if value > 0
        }
        if not budgets:
            return weights, 0.0
        groups = model_groups or {}

        if group_budgets is None:
            logical_budgets: Dict[str, float] = {}
            for model_id, budget in budgets.items():
                group_id = groups.get(model_id) or model_id
                logical_budgets[group_id] = max(
                    logical_budgets.get(group_id, 0.0),
                    budget,
                )
        else:
            logical_budgets = {
                group_id: float(value)
                for group_id, value in group_budgets.items()
                if value > 0
            }

        logical_total = sum(logical_budgets.values())
        if logical_total <= 0:
            return weights, 0.0

        approved_groups: Dict[str, Set[str]] = {}
        for model_id, budget in budgets.items():
            group_id = groups.get(model_id) or model_id
            if group_id not in logical_budgets:
                continue
            approved_groups.setdefault(group_id, set()).add(model_id)

        excluded = self._excluded_entry_keys(excluded_entries)
        entries_by_group: Dict[str, List[Tuple[int, float]]] = {}
        for uid, state in self.states.items():
            for entry in state.entries.values():
                if (state.address.lower(), entry.model_index) in excluded:
                    continue
                if entry.model_id not in budgets:
                    continue
                group_id = groups.get(entry.model_id) or entry.model_id
                if entry.model_id not in approved_groups.get(group_id, set()):
                    continue
                score = entry.ema_score
                if score < 1e-6:
                    score = 0.0
                if score <= 0:
                    continue
                entries_by_group.setdefault(group_id, []).append((uid, score))

        unallocated = 0.0
        for group_id, logical_budget in logical_budgets.items():
            logical_share = logical_budget / logical_total
            entries = entries_by_group.get(group_id, [])
            group_score_total = sum(score for _uid, score in entries)
            if group_score_total <= 0:
                unallocated += logical_share
                continue
            for uid, score in entries:
                weights[uid] = weights.get(uid, 0.0) + (
                    logical_share * score / group_score_total
                )

        return weights, min(max(unallocated, 0.0), 1.0)

    def halve_ema(self, address: str, model_index: int) -> None:
        """Halve the EMA score for a miner-model entry on proof failure.

        Called on every proof failure (entering probation or probation reset).
        Geometric decay: one failure = 50% penalty, two = 75%, three = 87.5%.
        Harsh enough to punish repeat offenders, fair enough that a single
        legitimate restart (missing one proof) is recoverable in a few epochs.
        """
        for _uid, mstate in self.states.items():
            if mstate.address.lower() != address.lower():
                continue
            entry = mstate.entries.get(model_index)
            if entry is not None:
                old = entry.ema_score
                entry.ema_score *= 0.5
                bt.logging.info(f"EMA halved for {address[:10]} model_index={model_index}: {old:.4f} -> {entry.ema_score:.4f}")
                return

    def zero_ema(self, address: str, model_index: int) -> None:
        """Zero the EMA score after a broken commitment.

        Halving assumes a single failure is recoverable, which is right for a
        miner having a bad epoch. A coordinator that broke a commitment it had
        already made keeps most of its accumulated score under geometric
        decay, so that case starts from zero instead.
        """
        for _uid, mstate in self.states.items():
            if mstate.address.lower() != address.lower():
                continue
            entry = mstate.entries.get(model_index)
            if entry is not None:
                old = entry.ema_score
                entry.ema_score = 0.0
                bt.logging.info(
                    f"EMA zeroed for {address[:10]} model_index={model_index} "
                    f"(binding violation): {old:.4f} -> 0.0000"
                )
                return

    def _update_entry_ema(
        self,
        entry: ModelEntryScore,
        epoch_score: Optional[float],
    ) -> None:
        """Update a single entry's EMA score from a zero cold start.

        The first scored epoch follows the same recurrence as every later
        epoch.  Copying the first observation directly into ``ema_score``
        allowed a newly registered model index to crystallize one favorable
        canary draw at full weight and made index churn bypass smoothing.
        """
        entry.total_epochs += 1

        if epoch_score is None:
            # No data this epoch — don't touch EMA
            return

        entry.scored_epochs += 1
        entry.ema_score = (
            self.ema_alpha * epoch_score
            + (1 - self.ema_alpha) * entry.ema_score
        )


# ── Probation tracker ─────────────────────────────────────────────────


@dataclass
class ProbationState:
    """Probation state for a single miner-model entry."""

    entered_at_epoch: int  # epoch when probation started
    consecutive_passes: int = 0  # consecutive epochs with all proofs passing
    required_passes: int = 3  # must pass N consecutive epochs to exit
    escalation_epochs: int = 5  # report offline after N epochs on probation
    endpoint: str = ""  # endpoint URL when probation started (for migration checks)
    # Why probation entered. "availability" = missed obligations only (a
    # dead or unreachable box, zero dishonesty evidence); "for_cause" =
    # any proof, integrity, or evasion failure. A fresh registration may
    # clear availability probation; for_cause always serves the full
    # consecutive-pass exit. Unknown/legacy state reads as for_cause.
    cause: str = "for_cause"


class ProbationTracker:
    """Tracks miners on probation after proof verification failures.

    Probation lifecycle:
        Normal → [proof failure] → Probation → [N consecutive passes] → Normal
                                       ↑              |
                                       └──────────────┘  (any failure resets)
                                              |
                                   [M epochs on probation] → reportOffline escalation

    During probation:
        - Epoch score forced to 0.0 (EMA decays)
        - Canary scheduler forces 100% proof verification
        - Proxy excludes miner from organic traffic (via shared state)
    """

    def __init__(
        self,
        required_passes: int = 3,
        escalation_epochs: int = 5,
        state_path: str = "/tmp/verathos_probation.json",
    ):
        self.required_passes = required_passes
        self.escalation_epochs = escalation_epochs
        self._state_path = state_path
        self._lock = threading.RLock()
        # Key: (miner_address, model_index) → ProbationState
        self._probation: Dict[Tuple[str, int], ProbationState] = {}
        self._load()

    def enter_probation(self, key: Tuple[str, int], epoch: int,
                        endpoint: str = "", cause: str = "for_cause") -> None:
        """Put a miner-model entry on probation (or reset if already on)."""
        with self._lock:
            if key in self._probation:
                # Already on probation — reset consecutive passes
                self._probation[key].consecutive_passes = 0
                if endpoint:
                    self._probation[key].endpoint = endpoint
                if cause == "for_cause":
                    # Severity only ratchets up: an availability entry that
                    # later fails a proof becomes for_cause; it never
                    # downgrades back.
                    self._probation[key].cause = "for_cause"
                bt.logging.info(f"Probation RESET for {key[0][:10]} model_index={key[1]} (new failure during probation)")
            else:
                self._probation[key] = ProbationState(
                    entered_at_epoch=epoch,
                    required_passes=self.required_passes,
                    escalation_epochs=self.escalation_epochs,
                    endpoint=endpoint,
                    cause=cause if cause in ("availability", "for_cause") else "for_cause",
                )
                bt.logging.info(f"Probation ENTERED for {key[0][:10]} model_index={key[1]} at epoch {epoch} endpoint={endpoint} (must pass {self.required_passes} consecutive epochs to exit)")
            self._save()

    def reconcile_probation(
        self,
        key: Tuple[str, int],
        *,
        entered_at_epoch: int,
        consecutive_passes: int,
        endpoint: str = "",
    ) -> None:
        """Replace one local probation row with authoritative DB state."""

        with self._lock:
            current = self._probation.get(key)
            self._probation[key] = ProbationState(
                entered_at_epoch=int(entered_at_epoch),
                consecutive_passes=max(0, int(consecutive_passes)),
                required_passes=(
                    current.required_passes if current else self.required_passes
                ),
                escalation_epochs=(
                    current.escalation_epochs
                    if current
                    else self.escalation_epochs
                ),
                endpoint=endpoint or (current.endpoint if current else ""),
            )
            self._save()

    def reconcile_not_on_probation(self, key: Tuple[str, int]) -> None:
        """Remove a stale local row after authoritative DB reconciliation."""

        with self._lock:
            if self._probation.pop(key, None) is not None:
                self._save()

    def record_pass(self, key: Tuple[str, int]) -> bool:
        """Record a clean epoch (all proofs passed) for a probation entry.

        Returns True if probation is lifted (enough consecutive passes).
        """
        with self._lock:
            if key not in self._probation:
                return False

            state = self._probation[key]
            state.consecutive_passes += 1

            if state.consecutive_passes >= state.required_passes:
                del self._probation[key]
                bt.logging.info(f"Probation LIFTED for {key[0][:10]} model_index={key[1]} after {state.consecutive_passes} consecutive passes")
                self._save()
                return True

            bt.logging.info(f"Probation pass {state.consecutive_passes}/{state.required_passes} for {key[0][:10]} model_index={key[1]}")
            self._save()
            return False

    def record_failure(self, key: Tuple[str, int]) -> None:
        """Record a proof failure during probation — resets consecutive passes."""
        with self._lock:
            if key in self._probation:
                self._probation[key].consecutive_passes = 0
                bt.logging.info(f"Probation pass counter RESET for {key[0][:10]} model_index={key[1]} (proof failure)")
                self._save()

    def clear_availability_probation(self, key: Tuple[str, int]) -> bool:
        """Drop a probation row whose cause is availability-only.

        Used when the authoritative database cleared the row after a
        changed registration. For_cause rows are never dropped here.
        """
        with self._lock:
            state = self._probation.get(key)
            if state is None or state.cause != "availability":
                return False
            del self._probation[key]
            self._save()
            bt.logging.info(
                f"Probation dropped for {key[0][:10]} model_index={key[1]}: "
                f"database cleared an availability-only record"
            )
            return True

    def is_on_probation(self, key: Tuple[str, int]) -> bool:
        """Check if a miner-model entry is on probation."""
        with self._lock:
            return key in self._probation

    def migrate_index(self, address: str, new_index: int,
                      new_endpoint: str = "") -> bool:
        """Migrate probation from an old model_index to a new one.

        When a miner re-registers (leaseModel), the contract array index
        changes but the probation entry still references the old index.
        This reconciles the mismatch by re-keying on the new index.

        Only migrates when the old probation entry's endpoint matches the
        new endpoint (same server re-registering with a new index).

        Does NOT migrate when endpoints differ — that would punish a
        healthy endpoint for a different server's failures.

        Returns True if a migration occurred.
        """
        with self._lock:
            new_key = (address, new_index)
            if new_key in self._probation:
                return False  # already correct

            # Find existing probation entry for this address with a different index
            old_key = None
            for k in self._probation:
                if k[0] == address and k[1] != new_index:
                    old_key = k
                    break

            if old_key is None:
                return False

            old_state = self._probation[old_key]
            old_endpoint = old_state.endpoint

            # Only migrate if same endpoint (same server, new index)
            if old_endpoint and new_endpoint and old_endpoint != new_endpoint:
                bt.logging.info(
                    f"Probation NOT migrated for {address[:10]}: "
                    f"index {old_key[1]} ({old_endpoint}) != "
                    f"index {new_index} ({new_endpoint}) — different endpoints"
                )
                return False

            # Same endpoint (or unknown endpoints for old probation entries
            # that predate the endpoint field) — migrate
            self._probation[new_key] = self._probation.pop(old_key)
            bt.logging.info(f"Probation migrated for {address[:10]}: model_index {old_key[1]} -> {new_index}")
            self._save()
            return True

    def should_escalate(self, key: Tuple[str, int], current_epoch: int) -> bool:
        """Check if probation has lasted long enough to escalate to reportOffline."""
        with self._lock:
            if key not in self._probation:
                return False
            state = self._probation[key]
            elapsed = current_epoch - state.entered_at_epoch
            if elapsed < 0 or elapsed > 10 * max(1, state.escalation_epochs):
                # Epoch NUMBERING changed under the stored entry (the owner
                # retimed epoch_blocks via the hosted config: 360 -> 180 doubled
                # every epoch number) or the clock ran
                # backwards. The stored age is meaningless either way — re-anchor
                # to now instead of instantly escalating a healthy rehab to
                # reportOffline.
                bt.logging.warning(
                    f"Probation age for {key[0][:10]} idx={key[1]} is implausible "
                    f"(entered_at_epoch={state.entered_at_epoch}, current="
                    f"{current_epoch}); re-anchoring to the current epoch"
                )
                self._probation[key] = replace(state, entered_at_epoch=int(current_epoch))
                self._save()
                return False
            return elapsed >= state.escalation_epochs

    def get_probation_entries(self) -> Set[Tuple[str, int]]:
        """Get all (miner_address, model_index) pairs currently on probation."""
        with self._lock:
            return set(self._probation.keys())

    def get_probation_addresses(self) -> Dict[str, List[int]]:
        """Get probation entries grouped by address (for shared state)."""
        with self._lock:
            result: Dict[str, List[int]] = {}
            for addr, model_index in self._probation:
                result.setdefault(addr, []).append(model_index)
            return result

    def clear_address(self, address: str) -> int:
        """Drop probation state for an identity that no longer owns its UID."""
        with self._lock:
            address_lower = str(address).lower()
            stale_keys = [key for key in self._probation if key[0].lower() == address_lower]
            for key in stale_keys:
                del self._probation[key]
            if stale_keys:
                self._save()
            return len(stale_keys)

    def clear_all(self) -> int:
        """Clear all operational probation entries and persist once."""

        with self._lock:
            count = len(self._probation)
            self._probation.clear()
            self._save()
            return count


    def _save(self) -> None:
        """Persist probation state to disk (atomic write)."""
        with self._lock:
            data = []
            for (addr, model_index), state in self._probation.items():
                data.append({
                    "address": addr,
                    "model_index": model_index,
                    "entered_at_epoch": state.entered_at_epoch,
                    "consecutive_passes": state.consecutive_passes,
                    "required_passes": state.required_passes,
                    "escalation_epochs": state.escalation_epochs,
                    "endpoint": state.endpoint,
                    "cause": state.cause,
                })
            tmp_path = self._state_path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(data, f)
                os.replace(tmp_path, self._state_path)
            except Exception as exc:
                bt.logging.warning(f"Failed to save probation state: {exc}")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _load(self) -> None:
        """Load probation state from disk (if it exists)."""
        with self._lock:
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                for entry in data:
                    key = (entry["address"], entry["model_index"])
                    self._probation[key] = ProbationState(
                        entered_at_epoch=entry["entered_at_epoch"],
                        consecutive_passes=entry.get("consecutive_passes", 0),
                        required_passes=entry.get("required_passes", self.required_passes),
                        escalation_epochs=entry.get("escalation_epochs", self.escalation_epochs),
                        endpoint=entry.get("endpoint", ""),
                        cause=str(entry.get("cause") or "for_cause"),
                    )
                if self._probation:
                    bt.logging.info(f"Loaded {len(self._probation)} probation entries from {self._state_path}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                bt.logging.warning(f"Failed to load probation state: {exc}")
