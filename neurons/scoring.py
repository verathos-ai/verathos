"""Epoch-based composite scoring for the Verathos Bittensor subnet.

Weight composition:

    WEIGHT(uid) = normalize( AGGREGATE )

    AGGREGATE   = Σ entry_ema(model_index)
                  Additive: more entries with good scores = more weight.

    ENTRY_EMA   = exponential moving average of epoch_scores:
                  - No receipts at all          → None (neutral, keep EMA)
                  - Receipts present + integrity → UTILITY × THROUGHPUT² × TTFT × SPEED × DEMAND_BONUS
                  - Receipt integrity failure    → 0.0 (hard penalty, decays EMA)

    UTILITY     = log2(Q)^1.8 × log2(ctx/1K) × Qq × Gq
                  Matches verallm/registry/models.py utility formula.

    THROUGHPUT²  = (weighted_tokens / 1M)²
                  Weighted tokens served per epoch (output × 3 + input), in megatokens.
                  Output weighted 3× input — decode is sequential, prefill is parallel.
                  Sybil defense: N UIDs splitting fixed demand each get 1/N tokens,
                  score per UID = (1/N)², total = N × (1/N)² = 1/N of honest.

    TTFT_FACTOR = min(2.0, sqrt(model_median_ttft / miner_median_ttft))
                  Peer-relative: compares this miner's median TTFT to the median
                  across all miners serving the same model.
                  Uncapped with soft cap at 2.0:
                  2× faster → 1.41×, 4× faster → 2.0× (capped).
                  2× slower → 0.71×, 4× slower → 0.50×.

    SPEED_FACTOR = min(2.0, sqrt(miner_median_tps / model_median_tps))
                  Peer-relative: compares this miner's median decode speed (tok/s)
                  to the median across all miners serving the same model.
                  Same uncapped curve with soft cap at 2.0.

    DEMAND_BONUS  = 1.0 + (demand_bps / 10000) × demand_bonus_max
                  Per-model demand signal from organic (non-canary) traffic.
                  Range: 1.0 (no demand data) to 1.0 + demand_bonus_max.

Peer medians use median-of-miner-medians: each miner's median TTFT/TPS is
computed from its receipts, then the model-level median is the median across
miner medians. This prevents high-traffic miners from skewing the reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import bittensor as bt
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from neurons.receipts import ServiceReceipt

logger = logging.getLogger(__name__)

# Quant quality factors — aligned with verallm/registry/models.py QUANT_QUALITY.
# Imported at module level to avoid circular imports with the registry.
QUANT_QUALITY: dict[str, float] = {
    "bf16": 1.0, "fp16": 1.0,
    "fp8": 0.98, "int8": 0.95,
    "int4": 0.90, "nf4": 0.90, "nvfp4": 0.92,
}


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

    # ALL receipts for this miner-model (from all validators)
    all_receipts: List[ServiceReceipt] = field(default_factory=list)

    # Proof verification outcomes (from own receipts where proof_requested=True)
    proof_tests: int = 0  # number of receipts where proof was requested
    proof_failures: int = 0  # number where proof was requested but failed

    # TEE attestation outcomes (from own receipts where tee_attestation_verified is set)
    tee_tests: int = 0
    tee_failures: int = 0
    tee_verified: bool = False  # at least one TEE attestation passed this epoch

    # Model entry metadata (for scoring)
    max_context_len: int = 0
    quant: str = ""

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


def compute_ttft_factor(
    miner_median_ttft_ms: float,
    model_median_ttft_ms: float,
) -> float:
    """Peer-relative TTFT factor: reward fast miners, penalize slow ones.

    TTFT_FACTOR = sqrt(model_median / miner_median)

    Uncapped — faster miners earn a bonus:
        2× faster than median: 1.41
        4× faster:             2.00
    Slower miners penalized:
        2× slower than median: 0.71
        4× slower:             0.50

    Soft cap at 2.0 to keep scores reasonable.

    Returns 1.0 if either value is <= 0 (no data / single miner).
    """
    if miner_median_ttft_ms <= 0 or model_median_ttft_ms <= 0:
        return 1.0
    return min(2.0, math.sqrt(model_median_ttft_ms / miner_median_ttft_ms))


def compute_speed_factor(
    miner_median_tps: float,
    model_median_tps: float,
) -> float:
    """Peer-relative decode speed factor: reward fast miners, penalize slow ones.

    SPEED_FACTOR = sqrt(miner_tps / model_median)

    Uncapped — faster miners earn a bonus:
        2× faster than median: 1.41
        4× faster:             2.00
    Slower miners penalized:
        2× slower than median: 0.71
        4× slower:             0.50

    Soft cap at 2.0 to keep scores reasonable.

    Returns 1.0 if either value is <= 0 (no data / single miner).
    """
    if miner_median_tps <= 0 or model_median_tps <= 0:
        return 1.0
    return min(2.0, math.sqrt(miner_median_tps / model_median_tps))


def compute_epoch_entry_score(
    outcome: EpochOutcome,
    active_params_b: float,
    moe_dense_equivalent: float = 0.0,
    generation_quality: float = 1.0,
    throughput_power: float = 2.0,
    peer_medians: Optional[PeerMedians] = None,
    tee_bonus: float = 1.0,
    demand_bonus: float = 1.0,
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

    if outcome.expected_own_receipt_count > 0:
        if len(outcome.own_receipts) < outcome.expected_own_receipt_count:
            bt.logging.info(f"Receipt integrity failure for {_who} model_index={outcome.model_index}: expected {outcome.expected_own_receipt_count} own receipts, found {len(outcome.own_receipts)} -> score=0")
            return 0.0

    # Proof verification failure: any failed proof -> hard penalty.
    # Use INFO not WARNING — the validator did its job correctly, the
    # miner is the one with the problem.  Operators don't need to be
    # paged for cheating / faulty miners.
    if outcome.proof_tests > 0 and outcome.proof_failures > 0:
        bt.logging.info(f"Proof verification failure for {_who} model_index={outcome.model_index}: {outcome.proof_failures}/{outcome.proof_tests} proofs failed -> score=0")
        return 0.0

    # TEE attestation failure: any failed attestation -> hard penalty.
    # Same rationale: miner-side failure, not a validator issue.
    if outcome.tee_tests > 0 and outcome.tee_failures > 0:
        bt.logging.info(f"TEE attestation failure for {_who} model_index={outcome.model_index}: {outcome.tee_failures}/{outcome.tee_tests} attestations failed -> score=0")
        return 0.0

    # No receipts at all -> not tested / unreachable (EMA unchanged)
    if not outcome.all_receipts:
        return None

    # Total weighted tokens served this epoch.
    # Output tokens weighted 3× input — decode is sequential (one forward pass
    # per token) while prefill is parallel. Reflects actual GPU cost ratio.
    OUTPUT_WEIGHT = 3
    total_tokens = sum(
        OUTPUT_WEIGHT * r.tokens_generated + r.prompt_tokens
        for r in outcome.all_receipts
        if r.tokens_generated > 0
    )
    if total_tokens <= 0:
        return 0.0

    # Extract latency: median TTFT from all receipts.
    # The caller (validator._close_epoch) filters outcome.all_receipts to
    # this model_id only, so no cross-model contamination.
    median_ttft = _median([
        r.ttft_ms for r in outcome.all_receipts if r.ttft_ms > 0
    ])

    # UTILITY (unchanged formula)
    quality_params = (
        moe_dense_equivalent if moe_dense_equivalent > 0 else active_params_b
    )
    utility = math.log2(max(quality_params, 1.0)) ** 1.8
    ctx_value = math.log2(max(outcome.max_context_len / 1024, 1))
    quant_q = QUANT_QUALITY.get(outcome.quant, 0.80)
    gen_q = generation_quality
    base_utility = utility * ctx_value * quant_q * gen_q

    # THROUGHPUT_SCORE (total tokens served, squared — sybil defense)
    # N sybil UIDs splitting fixed demand: each gets 1/N tokens,
    # score per UID = (1/N)², total = N × (1/N)² = 1/N of honest.
    # No normalization needed — weights are normalized across UIDs,
    # and utility already handles cross-model size differentiation.
    # Scale to megatokens before squaring for human-readable scores.
    # Divisor is cosmetic (cancels in weight normalization). At /1M:
    # canary-only ≈ 2 (idle baseline), 1K organic reqs ≈ 156 (real demand).
    throughput_score = (total_tokens / 1_000_000) ** throughput_power

    # TTFT_FACTOR (peer-relative, uncapped)
    # Faster than peer median → bonus (up to 2.0×).
    # Slower than peer median → penalty (sqrt curve).
    # No peers → 1.0 (no comparison possible).
    ttft_factor = 1.0
    if peer_medians is not None and median_ttft > 0:
        ttft_factor = compute_ttft_factor(median_ttft, peer_medians.median_ttft_ms)

    # SPEED_FACTOR (peer-relative, uncapped)
    speed_factor = 1.0
    if peer_medians is not None:
        miner_median_tps = _median([
            r.tokens_per_sec for r in outcome.all_receipts
            if r.tokens_per_sec > 0
        ])
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
    valid = [r for r in all_receipts if r.epoch_number == epoch_number]
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
        if r.epoch_number == epoch_number and not r.is_canary
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

    Uses median-of-miner-medians: each miner's median TTFT/TPS is computed
    from its receipts, then the model-level median is the median across miner
    medians. This prevents high-traffic miners from skewing the reference.

    Args:
        all_receipts: All receipts from all miners for the epoch.
        epoch_number: Current epoch number.

    Returns:
        Dict mapping model_id to PeerMedians.
    """
    # Group by (model_id, miner_address) → ([ttft], [tps])
    model_miner_stats: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}

    for r in all_receipts:
        if r.epoch_number != epoch_number:
            continue
        model_miners = model_miner_stats.setdefault(r.model_id, {})
        ttft_list, tps_list = model_miners.setdefault(
            r.miner_address, ([], []),
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

        for _addr, (ttft_vals, tps_vals) in miners.items():
            ttft_med = _median(ttft_vals)
            if ttft_med > 0:
                miner_ttft_medians.append(ttft_med)
            tps_med = _median(tps_vals)
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

    Per-entry EMA tracking with additive multi-model aggregation.
    Traffic volume multiplier from epoch service receipts.
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
    ) -> Optional[float]:
        """Update score for a single entry based on an epoch outcome.

        Returns the entry's epoch score, or None if no data.
        """
        if uid not in self.states:
            self.states[uid] = MinerScoreState(uid=uid, address=address)

        state = self.states[uid]

        # Ensure entry exists
        if model_index not in state.entries:
            state.entries[model_index] = ModelEntryScore(
                model_id=outcome.model_id,
                model_index=model_index,
            )

        entry = state.entries[model_index]
        epoch_score = compute_epoch_entry_score(
            outcome,
            active_params_b=active_params_b,
            moe_dense_equivalent=moe_dense_equivalent,
            generation_quality=generation_quality,
            throughput_power=self.throughput_power,
            peer_medians=peer_medians,
            tee_bonus=tee_bonus,
            demand_bonus=demand_bonus,
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

    def get_weights(self) -> Dict[int, float]:
        """Get normalized weights for all UIDs.

        WEIGHT(uid) = normalize( AGGREGATE )

        Traffic volume is already captured in throughput² (total tokens
        served per entry). No separate volume multiplier needed.
        """
        raw = {}
        for uid, state in self.states.items():
            score = state.aggregate_score
            # Floor dust-level scores to zero — prevents negligible weights
            # from persisting for miners that left the network long ago.
            if score < 1e-6:
                score = 0.0
            raw[uid] = score

        total = sum(raw.values())
        if total <= 0:
            return {uid: 0.0 for uid in raw}
        return {uid: s / total for uid, s in raw.items()}

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

    def _update_entry_ema(
        self,
        entry: ModelEntryScore,
        epoch_score: Optional[float],
    ) -> None:
        """Update a single entry's EMA score."""
        entry.total_epochs += 1

        if epoch_score is None:
            # No data this epoch — don't touch EMA
            return

        entry.scored_epochs += 1
        if entry.scored_epochs == 1:
            entry.ema_score = epoch_score
        else:
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
        # Key: (miner_address, model_index) → ProbationState
        self._probation: Dict[Tuple[str, int], ProbationState] = {}
        self._load()

    def enter_probation(self, key: Tuple[str, int], epoch: int,
                        endpoint: str = "") -> None:
        """Put a miner-model entry on probation (or reset if already on)."""
        if key in self._probation:
            # Already on probation — reset consecutive passes
            self._probation[key].consecutive_passes = 0
            if endpoint:
                self._probation[key].endpoint = endpoint
            bt.logging.info(f"Probation RESET for {key[0][:10]} model_index={key[1]} (new failure during probation)")
        else:
            self._probation[key] = ProbationState(
                entered_at_epoch=epoch,
                required_passes=self.required_passes,
                escalation_epochs=self.escalation_epochs,
                endpoint=endpoint,
            )
            bt.logging.info(f"Probation ENTERED for {key[0][:10]} model_index={key[1]} at epoch {epoch} endpoint={endpoint} (must pass {self.required_passes} consecutive epochs to exit)")
        self._save()

    def record_pass(self, key: Tuple[str, int]) -> bool:
        """Record a clean epoch (all proofs passed) for a probation entry.

        Returns True if probation is lifted (enough consecutive passes).
        """
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
        if key in self._probation:
            self._probation[key].consecutive_passes = 0
            bt.logging.info(f"Probation pass counter RESET for {key[0][:10]} model_index={key[1]} (proof failure)")
            self._save()

    def is_on_probation(self, key: Tuple[str, int]) -> bool:
        """Check if a miner-model entry is on probation."""
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
        if key not in self._probation:
            return False
        state = self._probation[key]
        return (current_epoch - state.entered_at_epoch) >= state.escalation_epochs

    def get_probation_entries(self) -> Set[Tuple[str, int]]:
        """Get all (miner_address, model_index) pairs currently on probation."""
        return set(self._probation.keys())

    def get_probation_addresses(self) -> Dict[str, List[int]]:
        """Get probation entries grouped by address (for shared state)."""
        result: Dict[str, List[int]] = {}
        for addr, model_index in self._probation:
            result.setdefault(addr, []).append(model_index)
        return result

    def _save(self) -> None:
        """Persist probation state to disk (atomic write)."""
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
                )
            if self._probation:
                bt.logging.info(f"Loaded {len(self._probation)} probation entries from {self._state_path}")
        except FileNotFoundError:
            pass
        except Exception as exc:
            bt.logging.warning(f"Failed to load probation state: {exc}")
