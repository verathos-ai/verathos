"""economic_recompute_v3: the complete wire-level HARD-tier audit verifier.

This is the current HARD audit of the economic proof system (the strong
``global_folded_execution_v3`` AIR path stays fail-closed until a genuine
native adapter exists).  It verifies one canonical
:class:`~verallm.proof_v3.economic_wire.EconomicRecomputeProofV3` against
exclusively validator-owned inputs:

* the exact signed profile + frozen commitment envelope (digest-bound);
* the validator-derived challenge (:mod:`economic_challenge`) -- the miner
  supplies NO selection; reveals for other coordinates are rejected;
* the authority-signed artifact manifest (:mod:`economic_artifacts`) -- the
  payload never supplies trusted weights or callbacks;
* the validator's own precommit context and OBSERVED prompt/output tokens.

Verified relations (all exact integer):

1. envelope.execution_root freezes the exact oracle inventory pre-nonce and
   the inventory matches the signed per-layer/global ABI exactly;
2. request/output binding (detached-trace / stale / cross-request cache);
3. per selected layer: capture-authenticated X + manifest-authenticated W +
   EXACT int8 recompute against the Merkle-committed surrogate
   (substitution, subtle-robust);
4. MANDATORY chain: bottom anchor residual[0] == signed embedding of the
   validator-observed prompt, complete per-layer boundary connectivity;
5. MANDATORY top anchor: capture-opened final hidden x manifest-opened
   LM-head rows == Merkle-committed surrogate logits at every sampled vocab
   row, and the validator-observed output token is their argmax.

Security model: probabilistic ECONOMIC security (sampled exact recompute at
unpredictable post-commitment coordinates), not a deterministic proof of
every operation.  K/V-cache oracles are committed and inventory-frozen;
their attention-internal recompute is the next audit extension.
"""

from __future__ import annotations

import math

from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_challenge import (
    CORRIDOR_QUANT_COEFF_DEN_V3,
    CORRIDOR_QUANT_COEFF_NUM_V3,
    CORRIDOR_REL_COEFF_DEN_V3,
    CORRIDOR_REL_COEFF_NUM_V3,
    ECONOMIC_SELECTION_ABI_V3,
    EconomicChallengeV3,
    audited_projections_for_layer_kind_v3,
    economic_selection_is_compact_v3,
    economic_selection_is_streaming_v3,
)
from verallm.proof_v3.economic_commitment import (
    expected_economic_inventory_v3,
    logits_block_geometry_v3,
    logits_block_oracle_id_v3,
    oracle_leaf_index_v3,
    oracle_leaf_width_v3,
    verify_economic_oracle_opening_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicRecomputeProofV3,
    bits_to_scale_v3,
    bounded_byte_width_v3,
    decode_int8_row_v3,
    encode_int8_row_v3,
    scale_to_bits_v3,
)
from verallm.proof_v3.economic_execution_anchor import (
    derive_economic_execution_anchor_oracle_binding_v3,
    economic_execution_anchor_encoding_v3,
    expected_economic_execution_anchor_inventory_v3,
    expected_economic_execution_anchor_reveals_v3,
    required_economic_transition_anchor_lane_keys_v3,
    verify_economic_transition_anchor_lanes_v3,
    verify_economic_execution_anchor_residual_chain_v3,
    verify_economic_execution_anchor_reveals_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.request import (
    PreExecutionRequestContextV3,
    execution_input_token_id_at_position_v3,
)

ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3 = "economic_recompute_v3"


_QUANT_COEFF = CORRIDOR_QUANT_COEFF_NUM_V3 / CORRIDOR_QUANT_COEFF_DEN_V3
_REL_COEFF = CORRIDOR_REL_COEFF_NUM_V3 / CORRIDOR_REL_COEFF_DEN_V3
# Safety factor over the per-cell int8-rounding std for the L2 corridor.
# Sound margin for honest concentration; a distributed substitution whose
# per-cell shift exceeds this many sigma is rejected.
import os as _os

_GL_MODULUS = (1 << 64) - (1 << 32) + 1

_CORRIDOR_SIGMA = float(_os.environ.get("VERATHOS_CORRIDOR_SIGMA", "8.0"))
# Aggregate acceptance: per LINK KIND (K-cache, V-cache, o/down/gate_up Y),
# the mean over that kind's corridor cells of the squared normalized
# deviation (delta / (sigma + floor + rel))^2, thresholded on the MAX over
# kinds.  A distributed low-amplitude substitution shifts every cell of the
# affected kinds a little; the per-kind mean then sits well outside honest
# concentration even when no single cell crosses the per-cell band, and
# unaffected kinds cannot dilute the signal.
#
# The cap is PER-MODEL calibration data: the signed manifest carries the
# owner-calibrated value (corridor_chi2_bits, measured on honest serving
# traces at model registration -- e.g. 0.04 for Qwen2.5-0.5B where honest
# kinds sit at 0.012-0.022 and a 0.1-sigma distributed substitute lifts
# its worst kind to 0.05-0.07).  Without a signed cap this conservative
# model-agnostic default applies.
_CORRIDOR_CHI2 = float(_os.environ.get("VERATHOS_CORRIDOR_CHI2", "0.2"))
# telemetry sink: when a list, corridor checks record
# (tag, aggregate_kind, delta, sigma, extra)
# instead of failing (calibration only; never set in production)
_CORRIDOR_REPORT = None
# Qualification-only sink for the exact fused-SwiGLU cells selected by the
# production verifier.  A model release report consumes these records to prove
# calibration and held-out coverage for every registered runtime layer.  It is
# never enabled by serving or validation.
_MLP_ACTIVATION_QUALIFICATION_REPORT = None
# Ordinary hard-trace replay under one authenticated serving geometry admits
# at most one quantized bin here. Prefix-cache replay crosses independently
# scheduled prefill geometries and uses its own signed cell-plus-row corridor.
_REPLAY_CAPTURE_MAX_LSB_DELTA_V3 = 1


def _execution_anchor_row_for_absolute_position_v3(
    *,
    absolute_position: int,
    prefix_cached_tokens: int,
) -> int | None:
    """Map a request position into the executed-suffix anchor domain.

    Prefix-cache execution-anchor trees contain only rows executed by the
    current request.  Positions inside the reused prefix therefore have no
    row in that tree; positions in the executed suffix are indexed from zero.
    """
    if (
        isinstance(absolute_position, bool)
        or not isinstance(absolute_position, int)
        or absolute_position < 0
        or isinstance(prefix_cached_tokens, bool)
        or not isinstance(prefix_cached_tokens, int)
        or prefix_cached_tokens < 0
    ):
        raise ValueError("execution-anchor position is malformed")
    if absolute_position < prefix_cached_tokens:
        return None
    return absolute_position - prefix_cached_tokens


def _replay_capture_cell_matches_v3(
    actual: int,
    expected: int,
    *,
    max_lsb_delta: int = _REPLAY_CAPTURE_MAX_LSB_DELTA_V3,
) -> bool:
    if (
        isinstance(max_lsb_delta, bool)
        or not isinstance(max_lsb_delta, int)
        or not 0 <= max_lsb_delta <= 8_190
    ):
        return False
    return abs(int(actual) - int(expected)) <= (
        max_lsb_delta
    )


def _replay_capture_row_matches_v3(
    actual,
    expected,
    *,
    max_lsb_delta: int = _REPLAY_CAPTURE_MAX_LSB_DELTA_V3,
    max_row_sq_delta: int | None = None,
) -> bool:
    if len(actual) != len(expected):
        return False
    if (
        isinstance(max_lsb_delta, bool)
        or not isinstance(max_lsb_delta, int)
        or not 0 <= max_lsb_delta <= 8_190
        or (
            max_row_sq_delta is not None
            and (
                isinstance(max_row_sq_delta, bool)
                or not isinstance(max_row_sq_delta, int)
                or not 0 <= max_row_sq_delta < 1 << 64
            )
        )
    ):
        return False
    try:
        import numpy as np

        actual_values = np.fromiter(
            actual,
            dtype=np.int64,
            count=len(actual),
        )
        expected_values = np.fromiter(
            expected,
            dtype=np.int64,
            count=len(expected),
        )
        delta = actual_values - expected_values
        if not bool(np.all(np.abs(delta) <= max_lsb_delta)):
            return False
        if max_row_sq_delta is None:
            return True
        row_sq_delta = sum(int(value) * int(value) for value in delta)
        return row_sq_delta <= max_row_sq_delta
    except ImportError:  # pragma: no cover - production dependency.
        deltas = tuple(
            int(actual_value) - int(expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
        return all(abs(delta) <= max_lsb_delta for delta in deltas) and (
            max_row_sq_delta is None
            or sum(delta * delta for delta in deltas) <= max_row_sq_delta
        )


def _prefix_cache_replay_row_matches_v3(
    actual,
    expected,
    *,
    max_lsb_delta: int,
    max_row_sq_delta: int,
    tag: str,
    layer: int,
    kv_head: int,
    position: int,
) -> bool:
    """Check one signed cache corridor and expose qualification telemetry."""

    try:
        deltas = tuple(
            int(actual_value) - int(expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    except ValueError:
        return False
    if len(deltas) != len(actual) or len(actual) != len(expected):
        return False
    max_abs = max((abs(delta) for delta in deltas), default=0)
    row_sq = sum(delta * delta for delta in deltas)
    if _CORRIDOR_REPORT is not None:
        _CORRIDOR_REPORT.append(
            (
                f"prefix-cache {tag.upper()} replay l{layer} h{kv_head} p{position}",
                f"prefix_cache_{tag}_row",
                float(max_abs),
                float(row_sq),
                0.0,
            )
        )
        return True
    return max_abs <= max_lsb_delta and row_sq <= max_row_sq_delta


if _os.environ.get("VERATHOS_CORRIDOR_REPORT"):
    _CORRIDOR_REPORT = []


def _requantize_economic_ox_head_v3(
    *,
    full_row,
    economic_scale: float,
    head: int,
    head_dim: int,
    attention_scale_num: int,
    attention_scale_exp: int,
) -> tuple[int, ...]:
    """Map an authenticated economic o_x row to the signed attention rail.

    Both scales are treated as exact rationals.  Rounding is nearest-even,
    matching ``torch.round`` used by the capture path, and the signed
    attention rail clamps to [-127, 127].  This deterministic conversion is
    what lets one runtime tensor feed differently-scaled commitment families
    without trusting a miner-provided bridge.
    """

    scale_num, scale_den = float(economic_scale).as_integer_ratio()
    if scale_num <= 0 or scale_den <= 0:
        raise _fail("economic o_x scale is not positive")
    if attention_scale_num <= 0 or attention_scale_exp < 0:
        raise _fail("signed attention o_x scale is malformed")
    start = int(head) * int(head_dim)
    stop = start + int(head_dim)
    if start < 0 or stop > len(full_row):
        raise _fail("economic o_x head exceeds the authenticated row")
    denominator = scale_den * int(attention_scale_num)
    multiplier = scale_num * (1 << int(attention_scale_exp))
    result = []
    for value in full_row[start:stop]:
        numerator = int(value) * multiplier
        sign = -1 if numerator < 0 else 1
        quotient, remainder = divmod(abs(numerator), denominator)
        doubled = remainder << 1
        if doubled > denominator or (
            doubled == denominator and quotient & 1
        ):
            quotient += 1
        rounded = sign * quotient
        result.append(max(-127, min(127, rounded)))
    return tuple(result)
def _corridor_check(
    *,
    surrogate_value: int,
    captured_value: int,
    x_row,
    w_row,
    x_scale: float,
    w_scale: float,
    y_scale: float,
    what: str,
    bias_value: float = 0.0,
    bias_quant: float = 0.0,
    stats: list | None = None,
    kind: str = "",
    x_sq: float | None = None,
    w_sq: float | None = None,
    sigma_cap: float = _CORRIDOR_SIGMA,
    output_quant_floor: float | None = None,
    captured_is_quantized: bool = True,
) -> float:
    """|dequant(surrogate) - dequant(captured)| within the signed corridor.

    The honest gap between the int8 surrogate ``x8.W8`` and the captured
    runtime output is a sum over the hidden dimension of independent int8
    rounding errors (``dx.W`` and ``x.dW``, each element in a half-step
    band) plus one output half-step and the bf16 accumulation term.  A sum
    of ``d`` independent zero-mean bounded terms concentrates as ``sqrt(d)``,
    NOT ``d`` -- so the sound corridor is an L2 (root-sum-of-squares) band,
    not an L1 worst-case one.  The L1 bound is ~``sqrt(d)`` too wide, which
    is exactly the slack a small distributed weight substitution hides in.

    ``_CORRIDOR_SIGMA`` is the safety factor over the per-cell standard
    deviation (calibrated so honest requests clear it with margin while a
    meaningfully-substituted model does not).  Returns the normalized
    margin for telemetry.
    """

    import math

    lhs = surrogate_value * x_scale * w_scale + bias_value
    # Most callers open an int8 output oracle, where +/- rails represent a
    # one-sided interval rather than a point.  The attention runtime bridge
    # instead supplies an exact decoded BF16/FP16 value.  Do not coerce that
    # value to int: doing so truncates the authenticated runtime row before
    # both its per-cell and aggregate corridor checks.
    if captured_is_quantized:
        captured_value = int(captured_value)
    else:
        captured_value = float(captured_value)
    rhs = captured_value * y_scale
    # per-cell std of the int8 quantization error, propagated:
    #   dx (<= x_scale/2 band, var (x_scale)^2/12) through W  -> sum W_i^2
    #   dW (<= w_scale/2 band, var (w_scale)^2/12) through x  -> sum x_i^2
    if w_sq is None:
        w_sq = sum(w * w for w in w_row)
    if x_sq is None:
        x_sq = sum(a * a for a in x_row)
    var = (
        (x_scale * x_scale / 12.0) * (w_scale * w_scale) * w_sq
        + (w_scale * w_scale / 12.0) * (x_scale * x_scale) * x_sq
    )
    sigma = math.sqrt(var)
    # output int8 half-step (+ the bias reveal's own half-step when the
    # projection carries a manifest-bound bias) + a small relative bf16
    # accumulation allowance
    captured_on_rail = captured_is_quantized and captured_value in (-128, 127)
    floor = (
        0.0
        if captured_on_rail
        else (
            0.5 * y_scale
            if output_quant_floor is None
            else float(output_quant_floor)
        )
    ) + bias_quant
    rel = _REL_COEFF * x_scale * w_scale * math.sqrt(x_sq * w_sq)
    extra = floor + rel
    bound = sigma_cap * sigma + extra
    if captured_is_quantized and captured_value == 127:
        delta = max(126.5 * y_scale - lhs, 0.0)
    elif captured_is_quantized and captured_value == -128:
        delta = max(lhs + 127.5 * y_scale, 0.0)
    else:
        delta = abs(lhs - rhs)
    if stats is not None:
        # Keep the exact relation label for failure diagnostics.  Acceptance
        # still aggregates by ``kind`` exactly as before; the extra field is
        # observational only and lets a retained failing bundle identify the
        # nonce-selected layer/relation that dominated the statistic.
        stats.append((kind, delta, sigma + extra, what))
    if _CORRIDOR_REPORT is not None:
        _CORRIDOR_REPORT.append((what, kind, delta, sigma, extra))
        return delta / bound if bound > 0 else 0.0
    if delta > bound:
        raise _fail(
            f"{what}: captured runtime output is outside the quantization "
            "corridor of the registered-weight surrogate (fabricated or "
            "substituted computation)"
        )
    return delta / bound if bound > 0 else 0.0


def _fixed_quantization_corridor_check(
    *,
    delta: float,
    quant: float,
    relative: float,
    what: str,
    kind: str,
    failure: str,
) -> float:
    """Apply one protocol-fixed quantization corridor.

    Calibration report mode records the exact coefficient required by honest
    runtime data but does not change the production coefficient. This keeps
    fixed-relation failures visible without silently folding them into the
    separately signed projection-sigma parameter.
    """

    bound = _QUANT_COEFF * quant + relative
    if _CORRIDOR_REPORT is not None:
        _CORRIDOR_REPORT.append(
            (what, f"fixed_quant:{kind}", delta, quant, relative)
        )
        return delta / bound if bound > 0.0 else 0.0
    if delta > bound:
        raise _fail(failure)
    return delta / bound if bound > 0.0 else 0.0


def _quantized_target_interval_delta(
    *,
    expected_lower: float,
    expected_upper: float,
    target_value: int,
    target_scale: float,
) -> tuple[float, float, float]:
    """Return distance to an authenticated int8 output cell.

    Interior values retain the historical center-plus-half-step treatment.
    The two rails are one-sided intervals and therefore contribute no finite
    output half-step.  The tuple is ``(delta, target_quant, center)``.
    """

    if (
        isinstance(target_value, bool)
        or not isinstance(target_value, int)
        or not -128 <= target_value <= 127
        or not math.isfinite(target_scale)
        or target_scale <= 0.0
        or not math.isfinite(expected_lower)
        or not math.isfinite(expected_upper)
        or expected_lower > expected_upper
    ):
        raise _fail("quantized target interval is malformed")
    center = target_value * target_scale
    if target_value == 127:
        return max(126.5 * target_scale - expected_upper, 0.0), 0.0, center
    if target_value == -128:
        return max(expected_lower + 127.5 * target_scale, 0.0), 0.0, center
    return (
        max(expected_lower - center, center - expected_upper, 0.0),
        0.5 * target_scale,
        center,
    )


def _quantized_sum_corridor_check(
    *,
    output_i8: int,
    output_scale: float,
    left_i8: int,
    left_scale: float,
    right_i8: int,
    right_scale: float,
    what: str,
    kind: str,
    failure: str,
) -> float:
    """Check ``output = left + right`` with correct one-sided rails."""

    def value_interval(value: int, scale: float) -> tuple[float, float]:
        value = int(value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise _fail("quantized sum scale is malformed")
        if value == 127:
            return 126.5 * scale, math.inf
        if value == -128:
            return -math.inf, -127.5 * scale
        center = value * scale
        return center - 0.5 * scale, center + 0.5 * scale

    output_interval = value_interval(output_i8, output_scale)
    left_interval = value_interval(left_i8, left_scale)
    right_interval = value_interval(right_i8, right_scale)
    summed_interval = (
        left_interval[0] + right_interval[0],
        left_interval[1] + right_interval[1],
    )
    delta = max(
        output_interval[0] - summed_interval[1],
        summed_interval[0] - output_interval[1],
        0.0,
    )
    center_output = int(output_i8) * output_scale
    center_sum = int(left_i8) * left_scale + int(right_i8) * right_scale
    return _fixed_quantization_corridor_check(
        delta=delta,
        quant=0.0,
        relative=_REL_COEFF * max(abs(center_output), abs(center_sum)),
        what=what,
        kind=kind,
        failure=failure,
    )


def _swiglu_output_is_forced_to_quantization_rail_v3(
    *,
    gate_i8: int,
    up_i8: int,
    gate_up_scale: float,
    output_i8: int,
    output_scale: float,
) -> bool:
    """Recognize an exact saturated SwiGLU output encoding.

    The ordinary fixed-quantization corridor assumes an unsaturated output
    bin.  For a rail value, prove the stronger statement instead: every
    real gate/up value represented by the authenticated input bins maps to
    that same output rail.  A saturated gate remains refused because SiLU
    over its half-infinite input interval needs a separate transcendental
    bound.  A saturated ``up`` value is safe to handle directly: it is a
    one-sided linear interval, and a strictly signed finite SiLU interval
    either proves the output rail or fails closed.
    """

    gate_i8 = int(gate_i8)
    up_i8 = int(up_i8)
    output_i8 = int(output_i8)
    if (
        output_i8 not in (-128, 127)
        or gate_i8 in (-128, 127)
        or not math.isfinite(gate_up_scale)
        or not math.isfinite(output_scale)
        or gate_up_scale <= 0.0
        or output_scale <= 0.0
    ):
        return False

    half = 0.5 * gate_up_scale
    gate_low = gate_i8 * gate_up_scale - half
    gate_high = gate_i8 * gate_up_scale + half
    if up_i8 == -128:
        up_low = -math.inf
        up_high = -127.5 * gate_up_scale
    elif up_i8 == 127:
        up_low = 126.5 * gate_up_scale
        up_high = math.inf
    else:
        up_low = up_i8 * gate_up_scale - half
        up_high = up_i8 * gate_up_scale + half
    gate_candidates = [gate_low, gate_high]
    # SiLU has one finite stationary point.  Include it when the input bin
    # spans it so the interval remains rigorous for negative gate values.
    silu_stationary = -1.2784645427610737
    if gate_low <= silu_stationary <= gate_high:
        gate_candidates.append(silu_stationary)
    silu_values = tuple(_silu(value) for value in gate_candidates)
    silu_low = min(silu_values)
    silu_high = max(silu_values)
    if silu_low > 0.0 and up_low >= 0.0:
        output_low = silu_low * up_low
        output_high = math.inf
    elif silu_high < 0.0 and up_high <= 0.0:
        output_low = silu_high * up_high
        output_high = math.inf
    elif silu_low > 0.0 and up_high <= 0.0:
        output_low = -math.inf
        output_high = silu_low * up_high
    elif silu_high < 0.0 and up_low >= 0.0:
        output_low = -math.inf
        output_high = silu_high * up_low
    elif math.isfinite(up_low) and math.isfinite(up_high):
        products = tuple(
            silu_value * up_value
            for silu_value in (silu_low, silu_high)
            for up_value in (up_low, up_high)
        )
        output_low = min(products)
        output_high = max(products)
    else:
        return False
    if output_i8 == -128:
        return output_high < -127.5 * output_scale
    return output_low > 126.5 * output_scale


def _rmsnorm_denominator_interval(
    *,
    source_row: tuple[int, ...] | list[int],
    source_scale: float,
    epsilon: float,
) -> tuple[float, float]:
    if len(source_row) == 0 or source_scale <= 0.0 or epsilon <= 0.0:
        raise _fail("RMSNorm quantization interval geometry is malformed")
    import numpy as np

    values = np.asarray(source_row, dtype=np.float64)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise _fail("RMSNorm quantization interval row is malformed")
    source_half = 0.5 * source_scale
    magnitudes = np.abs(values * source_scale)
    square_min_sum = float(
        np.square(np.maximum(magnitudes - source_half, 0.0)).sum()
    )
    square_max_sum = float(
        np.square(magnitudes + source_half).sum()
    )
    width = len(source_row)
    return (
        math.sqrt(square_min_sum / width + epsilon),
        math.sqrt(square_max_sum / width + epsilon),
    )


def _rmsnorm_saturation_aware_interval(
    *,
    source_row: tuple[int, ...] | list[int],
    source_scale: float,
    target_row: tuple[int, ...] | list[int],
    target_scale: float,
    norm_row: tuple[int, ...] | list[int],
    norm_scale: float,
    norm_gain_offset: float,
    epsilon: float,
) -> tuple[
    tuple[float, float],
    dict[int, tuple[float, float]],
    dict[int, tuple[float, float]],
]:
    """Bound RMSNorm when a committed int8 source row hits a rail.

    A rail value represents a half-infinite interval, not the ordinary
    half-step bin used by :func:`_rmsnorm_denominator_interval`.  The proof
    already authenticates the complete quantized RMSNorm source, output and
    norm-weight rows.  For each saturated source coordinate, ``output/gain``
    bounds ``source/rms``.  Substituting those ratios into the RMS equation
    gives a finite, conservative denominator interval without a fitted
    tolerance or an additional wire opening.

    A gain interval that crosses zero cannot be inverted.  For that case the
    exact RMSNorm identity still gives ``abs(source / rms) <= sqrt(width)``;
    use that model-independent interval directly instead of dividing by a
    near-zero gain.
    """

    import numpy as np

    source = np.asarray(source_row, dtype=np.int16)
    target = np.asarray(target_row, dtype=np.int16)
    weights = np.asarray(norm_row, dtype=np.int16)
    if (
        source.ndim != 1
        or target.shape != source.shape
        or weights.shape != source.shape
        or len(source) == 0
        or source_scale <= 0.0
        or target_scale <= 0.0
        or norm_scale <= 0.0
        or epsilon <= 0.0
    ):
        raise _fail("saturated RMSNorm interval geometry is malformed")
    saturated = np.flatnonzero((source == -128) | (source == 127))
    if not len(saturated):
        return (
            _rmsnorm_denominator_interval(
                source_row=source_row,
                source_scale=source_scale,
                epsilon=epsilon,
            ),
            {},
            {},
        )

    source_half = 0.5 * source_scale
    unsaturated = (source != -128) & (source != 127)
    magnitudes = np.abs(source[unsaturated].astype(np.float64) * source_scale)
    square_min_sum = float(
        np.square(np.maximum(magnitudes - source_half, 0.0)).sum()
    )
    square_max_sum = float(np.square(magnitudes + source_half).sum())

    ratio_intervals: dict[int, tuple[float, float]] = {}
    direct_output_intervals: dict[int, tuple[float, float]] = {}
    ratio_square_min = 0.0
    ratio_square_max = 0.0
    rail_square_min_sum = 0.0
    for raw_column in saturated:
        column = int(raw_column)
        target_value = int(target[column])
        gain_center = (
            int(weights[column]) * norm_scale + norm_gain_offset
        )
        gain_interval = (
            gain_center - 0.5 * norm_scale,
            gain_center + 0.5 * norm_scale,
        )
        rail_edge = (
            126.5 * source_scale
            if int(source[column]) == 127
            else -127.5 * source_scale
        )
        rail_square_min_sum += rail_edge * rail_edge
        if gain_interval[0] <= 0.0 <= gain_interval[1]:
            ratio_cap = math.sqrt(len(source))
            ratio_low, ratio_high = (
                (0.0, ratio_cap)
                if int(source[column]) == 127
                else (-ratio_cap, 0.0)
            )
            output_candidates = tuple(
                ratio * gain
                for ratio in (ratio_low, ratio_high)
                for gain in gain_interval
            )
            direct_output_intervals[column] = (
                min(output_candidates),
                max(output_candidates),
            )
        else:
            target_center = target_value * target_scale
            if target_value not in (-128, 127):
                target_interval = (
                    target_center - 0.5 * target_scale,
                    target_center + 0.5 * target_scale,
                )
                ratios = tuple(
                    numerator / denominator
                    for numerator in target_interval
                    for denominator in gain_interval
                )
                ratio_low, ratio_high = min(ratios), max(ratios)
            else:
                # RMSNorm itself gives the model-independent global bound
                # |source/rms| < sqrt(width).  Combine it with the finite edge
                # of the target rail; the open side remains conservative.
                ratio_cap = math.sqrt(len(source))
                if target_value == 127:
                    target_edge = 126.5 * target_scale
                    edge_ratios = tuple(
                        target_edge / denominator
                        for denominator in gain_interval
                    )
                    if gain_interval[0] > 0.0:
                        ratio_low, ratio_high = min(edge_ratios), ratio_cap
                    else:
                        ratio_low, ratio_high = -ratio_cap, max(edge_ratios)
                else:
                    target_edge = -127.5 * target_scale
                    edge_ratios = tuple(
                        target_edge / denominator
                        for denominator in gain_interval
                    )
                    if gain_interval[0] > 0.0:
                        ratio_low, ratio_high = -ratio_cap, max(edge_ratios)
                    else:
                        ratio_low, ratio_high = min(edge_ratios), ratio_cap
        if int(source[column]) == 127:
            ratio_low = max(ratio_low, 0.0)
            if ratio_high <= 0.0:
                raise _fail("saturated RMSNorm source has the wrong sign")
        else:
            ratio_high = min(ratio_high, 0.0)
            if ratio_low >= 0.0:
                raise _fail("saturated RMSNorm source has the wrong sign")
        ratio_intervals[column] = (ratio_low, ratio_high)
        ratio_magnitudes = (abs(ratio_low), abs(ratio_high))
        ratio_square_min += min(ratio_magnitudes) ** 2
        ratio_square_max += max(ratio_magnitudes) ** 2

    width = len(source)
    denominator_upper = (
        math.inf
        if ratio_square_max >= width
        else math.sqrt(
            (square_max_sum / width + epsilon)
            / (1.0 - ratio_square_max / width)
        )
    )
    fixed_point_lower = math.sqrt(
            (square_min_sum / width + epsilon)
            / (1.0 - ratio_square_min / width)
        )
    rail_lower = math.sqrt(
        (square_min_sum + rail_square_min_sum) / width + epsilon
    )
    denominator_interval = (
        max(fixed_point_lower, rail_lower),
        denominator_upper,
    )
    denominator_low, denominator_high = denominator_interval
    source_intervals: dict[int, tuple[float, float]] = {}
    for column, (ratio_low, ratio_high) in ratio_intervals.items():
        if not math.isfinite(denominator_high):
            source_intervals[column] = (
                (126.5 * source_scale, math.inf)
                if int(source[column]) == 127
                else (-math.inf, -127.5 * source_scale)
            )
            continue
        candidates = tuple(
            denominator * ratio
            for denominator in denominator_interval
            for ratio in (ratio_low, ratio_high)
        )
        source_low, source_high = min(candidates), max(candidates)
        if int(source[column]) == 127:
            source_low = max(source_low, 126.5 * source_scale)
        else:
            source_high = min(source_high, -127.5 * source_scale)
        if source_low > source_high:
            raise _fail(
                "saturated RMSNorm source is inconsistent with its rail"
            )
        source_intervals[column] = (source_low, source_high)
    return denominator_interval, source_intervals, direct_output_intervals


def _rmsnorm_quantization_interval(
    *,
    source_row: tuple[int, ...] | list[int],
    source_scale: float,
    norm_weight: int,
    norm_scale: float,
    norm_gain_offset: float,
    column: int,
    epsilon: float,
    denominator_interval: tuple[float, float] | None = None,
    selected_source_interval: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Enclose RMSNorm output for every value represented by the inputs.

    The prior first-order error propagation could under-estimate a rare
    quantization combination.  This interval evaluates the complete
    denominator and selected numerator over the authenticated source/weight
    quantization cells instead.  It is conservative without introducing a
    model-fitted tolerance.
    """

    if (
        len(source_row) == 0
        or column < 0
        or column >= len(source_row)
        or source_scale <= 0.0
        or norm_scale <= 0.0
        or epsilon <= 0.0
    ):
        raise _fail("RMSNorm quantization interval geometry is malformed")

    if selected_source_interval is None:
        source_half = 0.5 * source_scale
        selected_center = int(source_row[column]) * source_scale
        selected_interval = (
            selected_center - source_half,
            selected_center + source_half,
        )
    else:
        selected_interval = selected_source_interval
        if (
            len(selected_interval) != 2
            or selected_interval[0] > selected_interval[1]
            or any(math.isnan(value) for value in selected_interval)
        ):
            raise _fail("RMSNorm selected source interval is malformed")
    if denominator_interval is None:
        rms_lower, rms_upper = _rmsnorm_denominator_interval(
            source_row=source_row,
            source_scale=source_scale,
            epsilon=epsilon,
        )
    else:
        rms_lower, rms_upper = denominator_interval
        if (
            rms_lower <= 0.0
            or rms_upper < rms_lower
            or not math.isfinite(rms_lower)
            or math.isnan(rms_upper)
        ):
            raise _fail("RMSNorm denominator interval is malformed")
    gain_center = norm_weight * norm_scale + norm_gain_offset
    gain_interval = (
        gain_center - 0.5 * norm_scale,
        gain_center + 0.5 * norm_scale,
    )
    selected_lower, selected_upper = selected_interval
    if math.isinf(selected_upper):
        if (
            selected_upper < 0.0
            or not math.isfinite(selected_lower)
            or selected_lower < 0.0
        ):
            raise _fail("RMSNorm selected source interval is malformed")
        # A positive int8 source rail represents [edge, +inf).  Dividing
        # the two half-infinite source/RMS intervals independently loses the
        # exact RMSNorm invariant and produces an invalid infinite output
        # interval.  For every finite row and positive epsilon,
        #
        #     abs(source_i / rms(source)) < sqrt(width).
        #
        # Use that backend-independent identity as the finite ratio rail.
        # The lower endpoint remains zero because another saturated
        # coordinate may dominate the denominator.
        ratio_interval = (0.0, math.sqrt(len(source_row)))
        output_candidates = tuple(
            ratio * gain
            for ratio in ratio_interval
            for gain in gain_interval
        )
    elif math.isinf(selected_lower):
        if (
            selected_lower > 0.0
            or not math.isfinite(selected_upper)
            or selected_upper > 0.0
        ):
            raise _fail("RMSNorm selected source interval is malformed")
        ratio_interval = (-math.sqrt(len(source_row)), 0.0)
        output_candidates = tuple(
            ratio * gain
            for ratio in ratio_interval
            for gain in gain_interval
        )
    else:
        numerator_candidates = tuple(
            source * gain
            for source in selected_interval
            for gain in gain_interval
        )
        numerator_lower = min(numerator_candidates)
        numerator_upper = max(numerator_candidates)
        output_candidates = (
            numerator_lower / rms_lower,
            numerator_lower / rms_upper,
            numerator_upper / rms_lower,
            numerator_upper / rms_upper,
        )
    return min(output_candidates), max(output_candidates)


def _rmsnorm_corridor_check(
    *,
    target_value: int,
    target_scale: float,
    source_row: tuple[int, ...] | list[int],
    source_scale: float,
    norm_weight: int,
    norm_scale: float,
    norm_gain_offset: float,
    column: int,
    epsilon: float,
    denominator_interval: tuple[float, float] | None = None,
    selected_source_interval: tuple[float, float] | None = None,
    what: str,
    kind: str,
    failure: str,
) -> float:
    lower, upper = _rmsnorm_quantization_interval(
        source_row=source_row,
        source_scale=source_scale,
        norm_weight=norm_weight,
        norm_scale=norm_scale,
        norm_gain_offset=norm_gain_offset,
        column=column,
        epsilon=epsilon,
        denominator_interval=denominator_interval,
        selected_source_interval=selected_source_interval,
    )
    delta, target_quant, got = _quantized_target_interval_delta(
        expected_lower=lower,
        expected_upper=upper,
        target_value=target_value,
        target_scale=target_scale,
    )
    relative = _REL_COEFF * max(abs(lower), abs(upper), abs(got))
    return _fixed_quantization_corridor_check(
        delta=delta,
        quant=target_quant,
        relative=relative,
        what=what,
        kind=kind,
        failure=failure,
    )


def _rmsnorm_exact_source_corridor_check(
    *,
    target_value: int,
    target_scale: float,
    source_values,
    norm_weight: int,
    norm_scale: float,
    norm_gain_offset: float,
    column: int,
    epsilon: float,
    denominator: float | None = None,
    what: str,
    kind: str,
    failure: str,
) -> float:
    """Check RMSNorm from the exact replayed bf16/fp16 source row.

    Decode-window rows can exceed the absmax of the pre-nonce candidate pool.
    Treating a clipped int8 rail as a half-step quantization cell then gives a
    false, non-conservative denominator interval.  The compact GDN proof
    therefore carries the bounded replay source row already used by the
    geometric replay and binds it back to the opened int8 source before this
    check.  Only the authenticated norm-weight cell and target quantization
    remain interval-valued here.
    """

    import numpy as np

    values = np.asarray(source_values, dtype=np.float64)
    if (
        values.ndim != 1
        or not 0 <= column < len(values)
        or not bool(np.isfinite(values).all())
        or target_scale <= 0.0
        or norm_scale <= 0.0
        or epsilon <= 0.0
    ):
        raise _fail("exact RMSNorm source geometry is malformed")
    if denominator is None:
        denominator = math.sqrt(float(np.square(values).mean()) + epsilon)
    elif not math.isfinite(denominator) or denominator <= 0.0:
        raise _fail("exact RMSNorm denominator is malformed")
    gain_center = norm_weight * norm_scale + norm_gain_offset
    gain_low = gain_center - 0.5 * norm_scale
    gain_high = gain_center + 0.5 * norm_scale
    outputs = (
        float(values[column]) * gain_low / denominator,
        float(values[column]) * gain_high / denominator,
    )
    lower, upper = min(outputs), max(outputs)
    delta, target_quant, got = _quantized_target_interval_delta(
        expected_lower=lower,
        expected_upper=upper,
        target_value=target_value,
        target_scale=target_scale,
    )
    relative = _REL_COEFF * max(abs(lower), abs(upper), abs(got))
    return _fixed_quantization_corridor_check(
        delta=delta,
        quant=target_quant,
        relative=relative,
        what=what,
        kind=kind,
        failure=failure,
    )


def _silu(value: float) -> float:
    import math

    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def _runtime_precision_neighbors_v3(
    value: float,
    *,
    encoding_id: str,
    ulps: int = 2,
) -> tuple[float, ...]:
    """Canonical fp16/bf16 neighborhood around one fused-runtime result."""

    import numpy as np

    if not math.isfinite(value) or not 0 <= int(ulps) <= 2:
        raise _fail("fused MLP runtime precision input is malformed")
    if encoding_id == "fp16.v1":
        word = int(
            np.asarray([value], dtype="<f2").view("<u2")[0]
        )

        def decode(candidate: int) -> float:
            return float(
                np.asarray([candidate], dtype="<u2")
                .view("<f2")
                .astype(np.float64)[0]
            )

    elif encoding_id == "bf16.v1":
        value32 = np.asarray([value], dtype="<f4")
        bits = value32.view("<u4")
        rounded = bits + np.uint32(0x7FFF) + (
            (bits >> np.uint32(16)) & np.uint32(1)
        )
        word = int((rounded >> np.uint32(16))[0])

        def decode(candidate: int) -> float:
            bits = np.asarray(
                [np.uint32(candidate) << np.uint32(16)],
                dtype="<u4",
            )
            return float(bits.view("<f4")[0])

    else:
        raise _fail("fused MLP runtime encoding is not qualified")
    center = decode(word)
    if not math.isfinite(center):
        raise _fail("fused MLP runtime precision neighborhood is malformed")

    # IEEE fp16/bf16 words are sign-magnitude encodings, not numerically
    # ordered integers.  In particular, decrementing 0x8000 (-0) enters the
    # NaN range instead of crossing to small positive values.  Walk each
    # numerical direction explicitly and treat the two zero encodings as one
    # value so a finite near-zero fused result has a canonical finite ULP
    # neighborhood.
    def adjacent(candidate: int, *, upward: bool) -> int | None:
        if candidate in (0x0000, 0x8000):
            next_word = 0x0001 if upward else 0x8001
        elif upward:
            next_word = candidate - 1 if candidate & 0x8000 else candidate + 1
        else:
            next_word = candidate + 1 if candidate & 0x8000 else candidate - 1
        if not 0 <= next_word <= 0xFFFF:
            return None
        return next_word if math.isfinite(decode(next_word)) else None

    words = {word}
    for upward in (False, True):
        candidate = word
        for _ in range(int(ulps)):
            next_word = adjacent(candidate, upward=upward)
            if next_word is None:
                break
            words.add(next_word)
            candidate = next_word
    values = tuple(sorted({decode(candidate) for candidate in words}))
    if not values or not all(math.isfinite(candidate) for candidate in values):
        raise _fail("fused MLP runtime precision neighborhood is malformed")
    return values


def _verify_exact_runtime_mlp_cell_v3(
    *,
    gate: float,
    up: float,
    got: float,
    expected_gate_i8: int,
    expected_up_i8: int,
    expected_down_i8: int,
    gate_up_scale: float,
    down_x_scale: float,
    encoding_id: str,
) -> tuple[int, int, int]:
    """Bind one exact runtime SwiGLU cell to its quantized proof openings."""

    import numpy as np

    values = (gate, up, got, gate_up_scale, down_x_scale)
    if (
        not all(math.isfinite(float(value)) for value in values)
        or gate_up_scale <= 0.0
        or down_x_scale <= 0.0
        or any(
            isinstance(value, bool) or not -128 <= int(value) <= 127
            for value in (
                expected_gate_i8,
                expected_up_i8,
                expected_down_i8,
            )
        )
    ):
        raise _fail("lean MLP activation cell is malformed")

    def quantize(value: float, scale: float) -> int:
        rounded = int(np.rint(np.float64(value) / np.float64(scale)))
        return max(-128, min(127, rounded))

    actual_gate_i8 = quantize(gate, gate_up_scale)
    actual_up_i8 = quantize(up, gate_up_scale)
    actual_down_i8 = quantize(got, down_x_scale)
    if not all((
        _replay_capture_cell_matches_v3(actual_gate_i8, expected_gate_i8),
        _replay_capture_cell_matches_v3(actual_up_i8, expected_up_i8),
        _replay_capture_cell_matches_v3(actual_down_i8, expected_down_i8),
    )):
        raise _fail(
            "lean MLP activation cells are detached from the projection "
            "openings"
        )
    predicted = _silu(gate) * up
    if got not in _runtime_precision_neighbors_v3(
        predicted,
        encoding_id=encoding_id,
    ):
        raise _fail("lean MLP activation link is inconsistent")
    return actual_gate_i8, actual_up_i8, actual_down_i8


def quantization_stable_argmax_candidates_v3(
    logits, *, top_k: int
) -> tuple[int, ...]:
    """Return the canonical first ``top_k`` surrogate-logit candidates.

    Ordering is descending logit with the lower vocabulary index winning
    ties.  The bounded heap keeps this O(V log K) for production vocabularies
    and, unlike ``argpartition``, makes tie handling consensus-deterministic.
    """

    import heapq

    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise ProofV3VerificationError(
            "signed lm_head argmax top-k must be in [1, 32]"
        )
    if not logits:
        raise ProofV3VerificationError("top anchor has no committed logits")
    k = min(top_k, len(logits))
    return tuple(
        heapq.nsmallest(
            k,
            range(len(logits)),
            key=lambda row: (-int(logits[row]), row),
        )
    )


__all__ = [
    "ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3",
    "quantization_stable_argmax_candidates_v3",
    "verify_economic_recompute_v3",
]


def _fail(message: str) -> ProofV3VerificationError:
    return ProofV3VerificationError(message)


def _fail_with_row_mismatch(
    message: str,
    *,
    layer: int,
    position: int,
    token: int,
    column: int,
    actual: int,
    expected: int,
) -> ProofV3VerificationError:
    if _os.environ.get("VERATHOS_PROOF_V3_DIAGNOSTICS") != "1":
        return _fail(message)
    return _fail(
        f"{message} "
        f"(layer={layer}, position={position}, row={token}, column={column}, "
        f"actual={actual}, expected={expected})"
    )


def _opened_rows(
    *,
    oracle,
    base_binding: bytes,
    rows,
    opening,
    require_int8: bool,
    what: str,
    expect_mode: int = 1,
    packed_rows=None,
    expect_bounded_width: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Authenticate full oracle rows; return ``{row: signed value tuple}``.

    ``expect_mode`` is the SITE CONTRACT for the compact opening's value
    transport. ``packed_rows`` maps row -> packed int8 bytes for
    external-mode openings (the wire's single copy of the values);
    Merkle reconstruction against the committed root authenticates
    them."""

    rows = tuple(rows)
    col_count = oracle.col_count
    packed_decoded = None
    if packed_rows is not None:
        if expect_mode != 0:
            raise _fail(
                f"{what}: packed rows pair only with external-mode openings"
            )
        packed_decoded = {}
        packed_values = []
        for row in rows:
            row_bytes = packed_rows.get(row)
            if row_bytes is None:
                raise _fail(f"{what}: packed row is missing from the wire")
            entries = decode_int8_row_v3(row_bytes)
            if len(entries) != col_count:
                raise _fail(
                    f"{what}: packed row width does not match the oracle"
                )
            packed_decoded[row] = entries
            packed_values.append(row_bytes)
        if expect_bounded_width is None:
            from verallm.proof_v3.economic_commitment import (
                verify_external_oracle_rows_fast_v3,
            )

            try:
                fast = verify_external_oracle_rows_fast_v3(
                    oracle=oracle,
                    base_binding=base_binding,
                    row_indices=rows,
                    packed_rows=tuple(packed_values),
                    opening=opening,
                )
            except ProofV3VerificationError as exc:
                raise _fail(f"{what}: {exc}") from exc
            if fast:
                return packed_decoded
    leaves: list[int] = []
    row_base: dict[int, int] = {}
    for row in rows:
        if row >= oracle.row_count:
            raise _fail(f"{what}: sampled row exceeds the committed oracle")
        # leaf indices of one row are contiguous: base = index of (row, 0)
        base = oracle_leaf_index_v3(row, 0, col_count)
        row_base[row] = base
        leaves.extend(range(base, base + col_count))
    external = None
    if packed_rows is not None:
        external = []
        for row in rows:
            entries = packed_decoded[row]
            external.extend(v % _GL_MODULUS for v in entries)
    try:
        verify_economic_oracle_opening_v3(
            oracle=oracle,
            base_binding=base_binding,
            expected_indices=leaves,
            opening=opening,
            expected_mode=expect_mode,
            external_values=external,
            expected_bounded_width=expect_bounded_width,
            return_values=False,
        )
    except ProofV3VerificationError as exc:
        raise _fail(f"{what}: {exc}") from exc
    if packed_decoded is not None:
        result = packed_decoded
    else:
        if opening.values is None:
            raise _fail(f"{what}: authenticated row values are missing")
        width = oracle_leaf_width_v3(col_count)
        chunks = tuple(sorted({index // width for index in leaves}))
        chunk_slot = {chunk: slot for slot, chunk in enumerate(chunks)}
        result = {}
        for row in rows:
            base = row_base[row]
            slot = chunk_slot.get(base // width)
            if slot is None:
                raise _fail(f"{what}: authenticated row chunk is missing")
            start = slot * width
            entries = tuple(opening.values[start:start + col_count])
            if len(entries) != col_count:
                raise _fail(f"{what}: authenticated row is truncated")
            if expect_mode == 1:
                half = _GL_MODULUS // 2
                entries = tuple(
                    value - _GL_MODULUS if value > half else value
                    for value in entries
                )
            result[row] = entries
    for row in rows:
        entries = result[row]
        if require_int8 and (min(entries) < -128 or max(entries) > 127):
            raise _fail(f"{what}: committed capture value is outside int8 range")
    return result


def _opened_cells(
    *,
    oracle,
    base_binding: bytes,
    cells,
    opening,
    what: str,
    expect_mode: int = 1,
    expect_bounded_width: int | None = None,
    external_cells=None,
) -> dict[tuple[int, int], int]:
    col_count = oracle.col_count
    leaf_for: dict[int, tuple[int, int]] = {}
    for row, col in cells:
        if row >= oracle.row_count or col >= col_count:
            raise _fail(f"{what}: sampled cell exceeds the committed oracle")
        leaf_for[oracle_leaf_index_v3(row, col, col_count)] = (row, col)
    try:
        values = verify_economic_oracle_opening_v3(
            oracle=oracle,
            base_binding=base_binding,
            expected_indices=tuple(leaf_for),
            opening=opening,
            expected_mode=expect_mode,
            external_values=(
                None
                if external_cells is None
                else [
                    int(external_cells[leaf_for[leaf]]) % _GL_MODULUS
                    for leaf in leaf_for
                ]
            ),
            expected_bounded_width=expect_bounded_width,
        )
    except ProofV3VerificationError as exc:
        raise _fail(f"{what}: {exc}") from exc
    return {cell: values[leaf] for leaf, cell in leaf_for.items()}


def _anchor_packed_rows(
    *,
    anchor_binding,
    oracle_id: str,
    row_indices,
) -> dict[int, bytes] | None:
    """Return the validator-derived int8 rows already authenticated by the
    streaming execution anchors.

    Streaming transition-oracle openings use external-value mode so the same
    row is not serialized once as raw FP16/BF16 anchor data and again as an
    int8 oracle value.  Merkle reconstruction still authenticates the
    post-nonce compact oracle against these exact derived values.
    """

    if anchor_binding is None:
        return None
    expected = anchor_binding.expected_rows.get(oracle_id)
    rows = tuple(int(row) for row in row_indices)
    if expected is None or any(row not in expected for row in rows):
        raise _fail(
            f"execution anchor has no external row for oracle {oracle_id}"
        )
    return {
        row: encode_int8_row_v3(expected[row])
        for row in rows
    }


def _anchor_external_cells(
    *,
    anchor_binding,
    oracle_id: str,
    cells,
) -> dict[tuple[int, int], int] | None:
    if anchor_binding is None:
        return None
    coordinates = tuple((int(row), int(col)) for row, col in cells)
    sparse = anchor_binding.expected_cells.get(oracle_id, {})
    rows = anchor_binding.expected_rows.get(oracle_id, {})
    result = {}
    for coordinate in coordinates:
        value = sparse.get(coordinate)
        row, col = coordinate
        if value is None and row in rows and 0 <= col < len(rows[row]):
            value = rows[row][col]
        if value is None:
            raise _fail(
                f"execution anchor has no external cell for oracle "
                f"{oracle_id}"
            )
        result[coordinate] = int(value)
    return result


def _projection_surrogate_value(values, row: int, column: int) -> int:
    """Read a projection surrogate from row-packed or sparse-cell storage."""

    row_values = values.get(row)
    if row_values is not None:
        return int(row_values[column])
    return int(values[(row, column)])


def _streaming_attention_metadata_v3(
    *,
    profile,
    artifacts,
    challenge,
    envelope,
    capture_chain_digest: bytes,
    minimum_candidate_position: int = 0,
):
    """Derive the exact signed full-attention plan before anchor openings."""

    from verallm.proof_v3.attention_reduction_audit import (
        derive_reduction_bundle_v3,
    )
    from verallm.proof_v3.scored_calibration_set import (
        select_signed_calibration_v3,
    )

    calibration_set = getattr(artifacts, "attn_calibration_set", None)
    if calibration_set is None:
        raise _fail(
            "streaming attention requires the signed calibration set"
        )
    try:
        calibration = select_signed_calibration_v3(
            artifacts.manifest,
            calibration_set,
            int(envelope.context_token_count),
        )
    except ProofV3Error as exc:
        raise _fail(f"attention calibration set: {exc}")
    audits = {
        int(plan.layer_index): plan
        for plan in profile.relation_spec.layer_audits
    }
    if tuple(sorted(audits)) != tuple(audits):
        raise _fail("signed layer-audit inventory is not canonical")
    full_universe = tuple(
        layer for layer, plan in audits.items() if plan.is_full_attention
    )
    if not full_universe:
        raise _fail(
            "signed manifest requires attention but the profile has no "
            "full-attention layers"
        )
    selected = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer in audits and audits[layer].is_full_attention
    )
    minimum = min(
        int(profile.relation_spec.audit_policy.minimum_full_attention_layers),
        len(full_universe),
    )
    if len(selected) < minimum or not selected:
        raise _fail(
            "post-nonce hard selection does not cover the signed minimum "
            "full-attention layers"
        )
    head_counts = {
        (
            int(audits[layer].attention_query_head_count),
            int(audits[layer].attention_key_value_head_count),
            int(audits[layer].attention_head_dimension),
            str(audits[layer].attention_semantics_id),
        )
        for layer in full_universe
    }
    if len(head_counts) != 1:
        raise _fail(
            "economic attention adapter requires one qualified geometry "
            "across selectable full-attention layers"
        )
    nh, n_kv, hd, semantics_id = next(iter(head_counts))
    semantics = getattr(artifacts, "attention_runtime_semantics", None)
    signed_semantics = getattr(
        artifacts.manifest, "attn_runtime_semantics_digest", b""
    )
    if (
        semantics is None
        or not signed_semantics
        or semantics.digest() != signed_semantics
        or semantics.adapter_id != semantics_id
        or semantics.rotary_dimension > hd
    ):
        raise _fail(
            "attention runtime semantics are not authenticated by the "
            "signed profile/manifest"
        )
    policy = calibration_set.policy
    if int(policy.heads_per_layer) != int(
        profile.relation_spec.audit_policy.full_attention_heads_per_layer
    ):
        raise _fail(
            "attention calibration sampling policy disagrees with the "
            "signed hard-audit profile"
        )
    for layer in full_universe:
        try:
            heads = calibration.heads_for(layer)
        except (KeyError, IndexError, ProofV3Error) as exc:
            raise _fail(
                f"signed calibration does not qualify full-attention "
                f"layer {layer}"
            ) from exc
        if len(heads) != nh or any(
            int(params.head_dim) != hd for params, _bounds in heads
        ):
            raise _fail(
                f"signed calibration geometry disagrees at layer {layer}"
            )
    candidate_rows = tuple(
        position
        for position in challenge.attention_candidate_positions
        if position >= int(minimum_candidate_position)
    )
    if not candidate_rows:
        raise _fail("hard replay has no executed attention candidate")
    plans = derive_reduction_bundle_v3(
        validator_nonce=bytes(challenge.selection_seed),
        capture_chain_digest=bytes(capture_chain_digest),
        profile_digest=calibration.digest,
        selected_layers=selected,
        head_count=nh,
        candidate_rows=candidate_rows,
        chunk_count=1,
        heads_per_layer=int(policy.heads_per_layer),
        row_samples=int(policy.row_samples),
    )
    return {
        "audited": selected,
        "calibration": calibration,
        "calibration_set": calibration_set,
        "candidate_rows": candidate_rows,
        "plans": plans,
        "nh": nh,
        "n_kv": n_kv,
        "hd": hd,
        "semantics": semantics,
    }


def _verify_rational_attention_bundle_v3(*, artifacts, envelope,
                                         challenge, proof,
                                         base_binding: bytes,
                                         profile,
                                         anchor_rows,
                                         anchor_binding,
                                         anchor_encoding: str,
                                         opened_projections,
                                         streaming_metadata=None,
                                         gdn_lane_keys=(),
                                         auxiliary_lane_keys=(),
                                         prefix_cache_lanes=(),
                                         prefix_cache_projection_heads=()) -> None:
    """SCORED_SCHEME_RATIONAL_V2 attention audit INSIDE the economic
    proof (one proof satisfies the manifest's weight + attention
    mandates).

    The capture-kv rational bundle rides the PROOF's canonical
    attention request section together with its transport inputs
    (per-layer pre-nonce roots, tree binding, candidate pool, key
    count, base capture digest).  The verifier RECONSTRUCTS the
    pre-nonce binding itself: commitment = capture_kv_commitment_v3
    over the section's inputs, then fold(base, commitment) must equal
    the proof's capture_chain_digest -- which the envelope's execution
    root already froze before the nonce.  A stale, cross-request or
    post-hoc-substituted section fails that equality.  The bundle then
    verifies through the SAME adapter the canary audit uses --
    nonce-derived plans, kv equality vs the pre-nonce roots, wire row
    transport, and the signed rational bridge.  Geometry and head
    counts come only from signed data: scheme constants, the signed
    calibration band, and the manifest's own projection dims
    (gated-aware qkv/o algebra)."""

    from verallm.proof_v3.capture_kv_binding import (
        capture_kv_commitment_v3,
        fold_capture_kv_commitment_v3,
    )
    from verallm.proof_v3.rational_bundle_adapter import (
        apply_capture_kv_bundle_wire_v3,
        release_rational_geometry_v3,
    )
    from verallm.proof_v3.scored_calibration_set import (
        select_signed_calibration_v3,
    )

    section = getattr(proof, "attention", None)
    if section is None:
        raise _fail(
            "signed manifest requires the attention audit but the proof "
            "carries no attention request section (fail-closed)"
        )
    roots_by_layer = {
        int(layer): tuple(roots)
        for layer, roots in section.roots_by_layer
    }
    pool = tuple(int(p) for p in section.pool)
    key_count = int(section.key_count)
    binding = bytes(section.binding)
    wire = bytes(section.bundle_wire)
    from verallm.proof_v3.economic_profile import economic_profile_is_lean_v3

    lean = economic_profile_is_lean_v3(profile)
    streaming = economic_selection_is_streaming_v3(
        challenge.selection_abi_id
    )
    if streaming:
        import hashlib

        if streaming_metadata is None or anchor_rows is None:
            raise _fail(
                "anchor-backed attention verifier metadata is missing"
            )
        expected_binding = hashlib.sha256(
            b"VERATHOS/PROOF_V3/ANCHOR_ATTN_HELPER_BINDING/V1"
            + envelope.digest()
            + bytes(proof.capture_chain_digest)
        ).digest()
        if (
            binding != expected_binding
            or bytes(section.base_capture_digest)
            != bytes(proof.capture_chain_digest)
        ):
            raise _fail(
                "anchor-backed attention helper transport is stale or "
                "cross-request"
            )
        attention_digest = bytes(proof.capture_chain_digest)
    else:
        try:
            recomputed = capture_kv_commitment_v3(
                roots_by_layer=roots_by_layer, capture_binding=binding,
                candidate_rows=pool, key_count=key_count)
            folded = fold_capture_kv_commitment_v3(
                base_capture_digest=section.base_capture_digest,
                commitment=recomputed)
        except ProofV3Error as exc:
            raise _fail(f"capture-kv transport inputs are malformed: {exc}")
        if folded != bytes(proof.capture_chain_digest):
            raise _fail(
                "attention transport commitment is not folded into the "
                "authenticated capture chain (stale, cross-request, or "
                "post-nonce substituted section)"
            )
        attention_digest = recomputed
    if key_count != int(envelope.context_token_count):
        raise _fail(
            "capture-kv key count does not match the authenticated "
            "context length"
        )
    expected_pool = (
        tuple(streaming_metadata["candidate_rows"])
        if streaming
        else tuple(challenge.attention_candidate_positions)
    )
    if pool != expected_pool:
        raise _fail(
            "attention candidate pool does not match the canonical "
            "bounded transition pool"
        )
    # the V2 economic embedding requires the SIGNED calibration SET:
    # it carries the sampling policy (heads_per_layer, row_samples)
    # the bundle plans derive from -- protocol data, never a choice
    calib_set_digest = getattr(
        artifacts.manifest, "attn_calibration_set_digest", b"")
    if not calib_set_digest:
        raise _fail(
            "rational attention scheme requires a signed calibration "
            "SET digest in the manifest (fail-closed)"
        )
    calibration_set = getattr(artifacts, "attn_calibration_set", None)
    if calibration_set is None:
        raise _fail(
            "manifest pins an attention calibration SET digest but the "
            "validator loaded no calibration set (fail-closed)"
        )
    if streaming:
        calibration = streaming_metadata["calibration"]
        if calibration_set is not streaming_metadata["calibration_set"]:
            raise _fail("attention calibration set changed during verification")
    else:
        try:
            calibration = select_signed_calibration_v3(
                artifacts.manifest, calibration_set, int(key_count))
        except ProofV3Error as exc:
            raise _fail(f"attention calibration set: {exc}")
    policy = calibration_set.policy
    audited = (
        tuple(streaming_metadata["audited"])
        if streaming
        else tuple(sorted(int(x) for x in calibration.discriminative))
    )
    if not audited:
        raise _fail(
            "signed calibration pins no discriminative attention layers")
    heads0 = calibration.heads_for(audited[0])
    hd = (
        int(streaming_metadata["hd"])
        if streaming
        else int(heads0[0][0].head_dim)
    )
    gated = any(bool(getattr(b, "gated", False)) for _p, b in heads0)
    # head counts from the SIGNED manifest dims (gated-aware algebra:
    # the gated fused qkv packs [q_h | gate_h] per head, so its q
    # section is twice the o input width)
    qkv_entry = artifacts.manifest.entry_for(f"l{audited[0]}.qkv")
    o_entry = artifacts.manifest.entry_for(f"l{audited[0]}.o")
    if qkv_entry is None or o_entry is None:
        raise _fail(
            "signed manifest does not carry the audited attention "
            "layer's qkv/o projections"
        )
    o_in = int(o_entry.in_dim)
    qkv_out = int(qkv_entry.out_dim)
    if o_in <= 0 or o_in % hd:
        raise _fail(
            "signed o projection width is not head-aligned with the "
            "calibration head_dim"
        )
    nh = (
        int(streaming_metadata["nh"])
        if streaming
        else o_in // hd
    )
    if len(heads0) != nh:
        raise _fail(
            "signed calibration head count disagrees with the "
            "manifest projection dims"
        )
    kv_section = qkv_out - nh * hd * (2 if gated else 1)
    if kv_section <= 0 or kv_section % (2 * hd):
        raise _fail(
            "signed qkv/o dims are inconsistent with the calibrated "
            "attention layout"
        )
    n_kv = (
        int(streaming_metadata["n_kv"])
        if streaming
        else kv_section // (2 * hd)
    )
    if tuple(sorted(roots_by_layer)) != audited:
        raise _fail(
            "attention capture roots do not cover exactly the signed "
            "discriminative layer set"
        )

    # Derive the exact row/head plans before authenticating the economic
    # bridge openings.  The openings carry neither positions nor values
    # selected by the miner: both the row set and oracle identity come from
    # this nonce-bound plan and the signed inventory.
    from verallm.proof_v3.attention_reduction_audit import (
        derive_reduction_bundle_v3,
    )

    plans = (
        tuple(streaming_metadata["plans"])
        if streaming
        else derive_reduction_bundle_v3(
            validator_nonce=bytes(challenge.selection_seed),
            capture_chain_digest=attention_digest,
            profile_digest=calibration.digest,
            selected_layers=audited,
            head_count=nh,
            candidate_rows=pool,
            chunk_count=1,
            heads_per_layer=int(policy.heads_per_layer),
            row_samples=int(policy.row_samples),
        )
    )
    economic_openings = tuple(section.economic_ox_openings)
    if tuple(layer for layer, _index, _opening in economic_openings) != (
        audited
    ):
        raise _fail(
            "attention economic o_x openings do not cover exactly the "
            "signed discriminative layer set"
        )
    economic_rows: dict[int, dict[int, tuple[int, ...]]] = {}
    economic_scales: dict[int, float] = {}
    for plan, (layer, oracle_index, opening) in zip(
        plans, economic_openings, strict=True
    ):
        if int(layer) != int(plan.layer):
            raise _fail(
                "attention economic o_x opening references the wrong layer")
        if not 0 <= int(oracle_index) < len(proof.oracles):
            raise _fail(
                "attention economic o_x opening oracle index is out of range")
        oracle = proof.oracles[int(oracle_index)]
        expected_id = f"l{plan.layer}.attn_o_x"
        if oracle.oracle_id != expected_id:
            raise _fail(
                "attention economic o_x opening references the wrong oracle")
        if oracle.col_count != nh * hd:
            raise _fail(
                "attention economic o_x oracle has the wrong signed width")
        absolute_positions = tuple(int(p) for p in plan.row_positions)
        pool_rows = tuple(
            challenge.pool_row_for_sequence_position(position)
            for position in absolute_positions
        )
        if lean:
            projection = opened_projections.get((int(plan.layer), "o"))
            if (
                not isinstance(projection, tuple)
                or len(projection) < 1
                or not isinstance(projection[0], dict)
            ):
                raise _fail(
                    f"attention l{plan.layer} has no verified o-projection "
                    "input rows"
                )
            verified_projection_rows = projection[0]
            packed_rows = {}
            for row in pool_rows:
                values = verified_projection_rows.get(row)
                if values is None:
                    raise _fail(
                        f"attention l{plan.layer} o-projection input does "
                        "not cover the nonce-selected row"
                    )
                packed_rows[row] = encode_int8_row_v3(values)
        else:
            packed_rows = _anchor_packed_rows(
                anchor_binding=anchor_binding,
                oracle_id=expected_id,
                row_indices=pool_rows,
            )
        opened = _opened_rows(
            oracle=oracle,
            base_binding=base_binding,
            rows=pool_rows,
            opening=opening,
            require_int8=True,
            what=f"attention l{plan.layer} economic o_x bridge",
            expect_mode=(0 if streaming else 2),
            packed_rows=packed_rows,
        )
        economic_rows[int(plan.layer)] = {
            position: opened[row]
            for position, row in zip(
                absolute_positions, pool_rows, strict=True)
        }
        economic_scales[int(plan.layer)] = bits_to_scale_v3(
            oracle.scale_bits)

    def _economic_ox8_head_row(layer: int, head: int, position: int):
        try:
            full_row = economic_rows[int(layer)][int(position)]
            params = calibration.heads_for(int(layer))[int(head)][0]
            scale = economic_scales[int(layer)]
        except (KeyError, IndexError) as exc:
            raise _fail(
                "attention economic o_x bridge requested an unauthenticated "
                "coordinate"
            ) from exc
        return _requantize_economic_ox_head_v3(
            full_row=full_row,
            economic_scale=scale,
            head=int(head),
            head_dim=hd,
            attention_scale_num=int(params.ox_num),
            attention_scale_exp=int(params.ox_e),
        )

    anchor_roots_by_layer = None
    anchor_kv_value = None
    anchor_q13_head_row = None
    anchor_gate_fx_head_row = None
    used_lane_keys: set[tuple[int, int, int]] = set()
    lane_reveal_keys: set[tuple[int, int, int]] = set()
    gdn_lane_keys = set(tuple(key) for key in gdn_lane_keys)
    auxiliary_lane_keys = set(
        tuple(key) for key in auxiliary_lane_keys
    )
    if streaming:
        from verallm.proof_v3.attention_anchor_binding import (
            AttentionAnchorGeometryV3,
            attention_anchor_geometry_v3,
            attention_anchor_head_byte_range_v3,
            extract_execution_anchor_range_v3,
            required_execution_anchor_lanes_v3,
            runtime_attention_quantized_row_v3,
            runtime_kv_head_quantization_bounds_v3,
            runtime_kv_head_quantized_v3,
        )
        from verallm.proof_v3.scored_attention_reference import (
            GATE_FIXED_BITS,
        )

        semantics = streaming_metadata["semantics"]
        commitment_index_by_stage = {
            commitment.stage_id: index
            for index, commitment in enumerate(proof.execution_anchors)
        }
        commitment_by_layer = {}
        geometry_by_layer = {}
        lane_geometry_by_layer = {}
        params_by_layer = {}
        anchor_stage_suffix = (
            "attention_kv_output" if lean else "attention_qkv_output"
        )
        for layer in audited:
            stage = f"l{layer}.{anchor_stage_suffix}"
            try:
                commitment_index = commitment_index_by_stage[stage]
            except KeyError as exc:
                raise _fail(
                    f"attention anchor inventory has no {stage}"
                ) from exc
            commitment = proof.execution_anchors[commitment_index]
            qkv_in, qkv_out = artifacts.dims(f"l{layer}.qkv")
            o_in, _o_out = artifacts.dims(f"l{layer}.o")
            del qkv_in
            geometry_by_layer[layer] = attention_anchor_geometry_v3(
                qkv_width=qkv_out,
                o_input_width=o_in,
                query_heads=nh,
                kv_heads=n_kv,
                head_dim=hd,
                semantics=semantics,
            )
            geometry = geometry_by_layer[layer]
            lane_geometry_by_layer[layer] = (
                AttentionAnchorGeometryV3(
                    query_heads=geometry.query_heads,
                    kv_heads=geometry.kv_heads,
                    head_dim=geometry.head_dim,
                    qkv_width=2 * geometry.kv_heads * geometry.head_dim,
                    q_block_width=0,
                    k_block_offset=0,
                    v_block_offset=geometry.kv_heads * geometry.head_dim,
                    gated=geometry.gated,
                )
                if lean
                else geometry
            )
            commitment_by_layer[layer] = (
                commitment_index,
                commitment,
            )
            params_by_layer[layer] = tuple(
                params for params, _bounds in calibration.heads_for(layer)
            )
        anchor_roots_by_layer = {
            layer: commitment.root
            for layer, (_index, commitment) in commitment_by_layer.items()
        }
        prefix_cache_lane_map = dict(prefix_cache_lanes)
        prefix_commitment = getattr(proof, "prefix_cache", None)
        prefix_commitment = (
            None if prefix_commitment is None else prefix_commitment.commitment
        )
        if prefix_commitment is not None:
            from verallm.proof_v3.prefix_cache import (
                prefix_cache_attention_anchor_root_v3,
            )

            anchor_roots_by_layer = {
                layer: prefix_cache_attention_anchor_root_v3(
                    commitment=prefix_commitment,
                    layer_index=layer,
                )
                for layer in audited
            }
        lane_openings_by_layer = {layer: {} for layer in audited}
        for reveal in proof.execution_anchor_lane_reveals:
            key = (
                int(reveal.commitment_index),
                int(reveal.opening.row_index),
                int(reveal.opening.lane_index),
            )
            lane_reveal_keys.add(key)
            matched = False
            for layer, (commitment_index, _commitment) in (
                commitment_by_layer.items()
            ):
                if reveal.commitment_index == commitment_index:
                    lane_openings_by_layer[layer][
                        (
                            int(reveal.opening.row_index),
                            int(reveal.opening.lane_index),
                        )
                    ] = reveal.opening
                    matched = True
                    break
            if not matched:
                if (
                    key not in gdn_lane_keys
                    and key not in auxiliary_lane_keys
                ):
                    raise _fail(
                        "anchor lane reveal references a stage outside the "
                        "nonce-selected attention/GDN proof"
                    )
        quantized_rows = {}
        query_rows = {
            (int(layer), int(position)): bytes(row)
            for layer, position, row in section.query_rows
        }
        expected_query_rows = {
            (int(plan.layer), int(position))
            for plan in plans
            for position in plan.row_positions
        }
        expected_query_rows.update(
            (int(layer), int(position))
            for layer, _kv_head, position in prefix_cache_projection_heads
        )
        if lean:
            if set(query_rows) != expected_query_rows:
                raise _fail(
                    "lean attention query rows do not match the "
                    "nonce-derived plan"
                )
        elif query_rows:
            raise _fail(
                "non-lean attention proof carries post-nonce query rows"
            )

        def _runtime_row(layer: int, position: int):
            key = (int(layer), int(position))
            if key not in quantized_rows:
                try:
                    raw = (
                        query_rows[(int(layer), int(position))]
                        if lean
                        else anchor_rows[
                            f"l{layer}.attention_qkv_output"
                        ][int(position)]
                    )
                except KeyError as exc:
                    raise _fail(
                        "attention Q/gate runtime anchor row is missing"
                    ) from exc
                quantized_rows[key] = runtime_attention_quantized_row_v3(
                    row_bytes=raw,
                    layer=int(layer),
                    position=int(position),
                    geometry=geometry_by_layer[int(layer)],
                    semantics=semantics,
                    params_by_head=params_by_layer[int(layer)],
                    encoding_id=anchor_encoding,
                )
            return quantized_rows[key]

        def _anchor_q13_head_row(layer: int, head: int, position: int):
            return _runtime_row(
                layer, position
            ).q13_by_head[int(head)]

        def _anchor_gate_head_row(layer: int, head: int, position: int):
            gates = _runtime_row(layer, position).gate_by_head
            if gates is None:
                raise _fail(
                    "ungated attention runtime was asked for gate values"
                )
            import numpy as np

            return tuple(
                int(value)
                for value in np.rint(
                    np.asarray(gates[int(head)], dtype=np.float64)
                    * (1 << GATE_FIXED_BITS)
                ).tolist()
            )

        kv_head_cache = {}
        kv_head_bounds_cache = {}

        def _anchor_kv(
            layer: int,
            tag: str,
            native_leaf: int,
            sp: int,
            dim: int,
        ) -> int:
            kv_head, remainder = divmod(
                int(native_leaf), int(sp) * int(dim)
            )
            position, coordinate = divmod(remainder, int(dim))
            if kv_head >= n_kv or position >= key_count:
                return 0
            cache_key = (int(layer), tag, kv_head, position)
            if cache_key not in kv_head_cache:
                geometry = lane_geometry_by_layer[int(layer)]
                start, length = attention_anchor_head_byte_range_v3(
                    geometry=geometry,
                    tag=tag,
                    head=kv_head,
                )
                cached = (
                    0
                    if prefix_commitment is None
                    else prefix_commitment.cached_token_count
                )
                if position < cached:
                    from verallm.proof_v3.execution_anchor import (
                        execution_anchor_lane_bytes_v3,
                    )

                    stage_id = f"l{layer}.attention_{tag}_cache"
                    lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
                    block = position // prefix_commitment.block_token_count
                    row = position % prefix_commitment.block_token_count
                    lanes = required_execution_anchor_lanes_v3(
                        byte_start=kv_head * geometry.head_dim * 2,
                        byte_length=length,
                        lane_bytes=lane_bytes,
                    )
                    try:
                        joined = b"".join(
                            prefix_cache_lane_map[
                                (block, stage_id, row, lane)
                            ]
                            for lane in lanes
                        )
                    except KeyError as exc:
                        raise _fail(
                            "attention K/V equality lacks a nonce-selected "
                            "prefix-cache lane"
                        ) from exc
                    offset = (
                        kv_head * geometry.head_dim * 2
                        - lanes[0] * lane_bytes
                    )
                    raw_head = joined[offset:offset + length]
                    if len(raw_head) != length:
                        raise _fail(
                            "attention prefix-cache K/V head is truncated"
                        )
                else:
                    commitment_index, commitment = commitment_by_layer[
                        int(layer)
                    ]
                    lanes = required_execution_anchor_lanes_v3(
                        byte_start=start,
                        byte_length=length,
                    )
                    suffix_row = position - cached
                    used_lane_keys.update(
                        (commitment_index, suffix_row, lane)
                        for lane in lanes
                    )
                    raw_head = extract_execution_anchor_range_v3(
                        commitment=commitment,
                        row_index=suffix_row,
                        byte_start=start,
                        byte_length=length,
                        openings=lane_openings_by_layer[int(layer)],
                    )
                if position < cached:
                    from verallm.proof_v3.attention_anchor_binding import (
                        runtime_paged_cache_kv_head_quantized_v3,
                    )

                    kv_head_cache[cache_key] = (
                        runtime_paged_cache_kv_head_quantized_v3(
                            tag=tag,
                            raw_head_bytes=raw_head,
                            kv_head=kv_head,
                            geometry=geometry,
                            params_by_head=params_by_layer[int(layer)],
                            encoding_id=anchor_encoding,
                        )
                    )
                    kv_head_bounds_cache[cache_key] = (
                        kv_head_cache[cache_key],
                        kv_head_cache[cache_key],
                    )
                else:
                    kv_head_cache[cache_key] = runtime_kv_head_quantized_v3(
                        tag=tag,
                        raw_head_bytes=raw_head,
                        layer=int(layer),
                        position=position,
                        kv_head=kv_head,
                        geometry=geometry,
                        semantics=semantics,
                        params_by_head=params_by_layer[int(layer)],
                        encoding_id=anchor_encoding,
                    )
                    kv_head_bounds_cache[cache_key] = (
                        runtime_kv_head_quantization_bounds_v3(
                            tag=tag,
                            raw_head_bytes=raw_head,
                            layer=int(layer),
                            position=position,
                            kv_head=kv_head,
                            geometry=geometry,
                            semantics=semantics,
                            params_by_head=params_by_layer[int(layer)],
                            encoding_id=anchor_encoding,
                        )
                    )
            return int(kv_head_cache[cache_key][coordinate])

        def _anchor_kv_bounds(
            layer: int,
            tag: str,
            native_leaf: int,
            sp: int,
            dim: int,
        ) -> tuple[int, int]:
            kv_head, remainder = divmod(
                int(native_leaf), int(sp) * int(dim)
            )
            position, coordinate = divmod(remainder, int(dim))
            if kv_head >= n_kv or position >= key_count:
                return 0, 0
            _anchor_kv(layer, tag, native_leaf, sp, dim)
            lower, upper = kv_head_bounds_cache[
                (int(layer), tag, kv_head, position)
            ]
            return int(lower[coordinate]), int(upper[coordinate])

        anchor_kv_value = _anchor_kv
        anchor_q13_head_row = _anchor_q13_head_row
        anchor_gate_fx_head_row = (
            _anchor_gate_head_row if semantics.gated else None
        )

    try:
        apply_capture_kv_bundle_wire_v3(
            wire=bytes(wire),
            validator_nonce=bytes(challenge.selection_seed),
            capture_chain_digest=attention_digest,
            validator_binding_digest=envelope.digest(),
            selected_layers=audited, calibration=calibration,
            geometry=release_rational_geometry_v3(hd),
            head_count=nh, n_kv=n_kv,
            candidate_rows=tuple(int(p) for p in pool),
            key_count=int(key_count),
            capture_roots_by_layer={
                int(layer): tuple(roots)
                for layer, roots in roots_by_layer.items()},
            capture_binding=bytes(binding),
            economic_ox8_head_row=_economic_ox8_head_row,
            anchor_roots_by_layer=anchor_roots_by_layer,
            anchor_kv_value=anchor_kv_value,
            anchor_kv_bounds=(
                _anchor_kv_bounds if streaming else None
            ),
            anchor_q13_head_row=anchor_q13_head_row,
            pcs_query_count=challenge.pcs_query_count,
            anchor_gate_fx_head_row=anchor_gate_fx_head_row,
            anchor_integer_tolerance=(
                0
                if not streaming
                else int(streaming_metadata["semantics"].integer_tolerance)
            ),
            heads_per_layer=int(policy.heads_per_layer),
            row_samples=int(policy.row_samples))
    except (ProofV3Error, ProofV3VerificationError) as exc:
        raise _fail(f"attention bundle verification failed: {exc}")
    if streaming and lane_reveal_keys != (
        used_lane_keys | gdn_lane_keys | auxiliary_lane_keys
    ):
        raise _fail(
            "anchor lane reveals do not cover exactly the nonce-derived "
            "attention/GDN coordinates"
        )


def verify_economic_recompute_v3(
    *,
    proof_system_id: str,
    proof: EconomicRecomputeProofV3,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    challenge: EconomicChallengeV3,
    artifacts: EconomicVerifiedArtifactsV3,
    precommit_context: PreExecutionRequestContextV3,
    prompt_token_ids,
    observed_output_token_ids,
    observed_text_utf8: bytes,
    finish_reason: str,
) -> None:
    """Verify one canonical economic wire proof; raise on ANY broken relation."""

    import os as _os
    import time as _time

    _profile_timing = _os.environ.get("VERATHOS_PROOF_V3_PROFILE", "") == "1"
    _profile_sync = (
        _profile_timing
        and _os.environ.get("VERATHOS_PROOF_V3_PROFILE_SYNC", "") == "1"
    )
    _profile_started = _time.perf_counter()
    _profile_last = _profile_started

    def _profile_mark(name: str) -> None:
        nonlocal _profile_last
        if not _profile_timing:
            return
        if _profile_sync:
            try:
                import torch as _torch
            except ImportError:
                pass
            else:
                if _torch.cuda.is_available():
                    _torch.cuda.synchronize()
        now = _time.perf_counter()
        print(
            f"[PROOF-V3-HARD-VERIFY] phase={name} "
            f"seconds={now - _profile_last:.3f} "
            f"total={now - _profile_started:.3f}",
            flush=True,
        )
        _profile_last = now

    # ---- (0) proof-system routing: this adapter is only economic ----------
    from verallm.proof_v3.profile import GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3

    if proof_system_id == GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3:
        raise _fail(
            "economic adapter must not accept the strong global-folded profile"
        )
    if proof_system_id != ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
        raise _fail(
            f"economic adapter cannot verify proof-system {proof_system_id!r}"
        )
    if not isinstance(proof, EconomicRecomputeProofV3):
        raise _fail("economic proof has an unexpected type")
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise _fail("economic profile has an unexpected type")
    if profile.proof_system_id != ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
        raise _fail("signed profile is not an economic recompute profile")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise _fail("economic envelope has an unexpected type")
    if not isinstance(challenge, EconomicChallengeV3):
        raise _fail("economic challenge has an unexpected type")
    if not isinstance(artifacts, EconomicVerifiedArtifactsV3):
        raise _fail("economic artifacts have an unexpected type")
    if not isinstance(precommit_context, PreExecutionRequestContextV3):
        raise _fail("economic precommit context has an unexpected type")
    from verallm.proof_v3.economic_profile import (
        ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
        economic_profile_is_lean_v3,
        economic_profile_uses_selected_trace_v3,
    )
    if profile.adapter_version not in {
        ECONOMIC_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
        ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
    }:
        raise _fail("economic profile adapter version is unsupported")
    lean = economic_profile_is_lean_v3(profile)
    selected_trace_profile = economic_profile_uses_selected_trace_v3(profile)
    complete_projection = (
        lean and challenge.full_row_projection_audit
    )
    compact_terminal = (
        economic_selection_is_compact_v3(challenge.selection_abi_id)
    )
    succinct_projection = (
        lean
        and compact_terminal
        and not complete_projection
        and not selected_trace_profile
    )
    selected_trace = (
        selected_trace_profile and not complete_projection
    )
    if selected_trace != bool(proof.selected_trace_wire):
        raise _fail(
            "selected-trace presence disagrees with the signed "
            "post-commitment audit mode"
        )
    if complete_projection != bool(proof.lean_projection_batch_wire):
        raise _fail(
            "complete projection batch presence disagrees with the "
            "post-commitment audit mode"
        )
    if succinct_projection != bool(
        proof.succinct_projection_batch_wire
    ):
        raise _fail(
            "succinct projection batch presence disagrees with the "
            "post-commitment audit mode"
        )

    # SIGNED per-cell corridor sigma (calibrated with corridor_chi2_bits);
    # falls back to the model-agnostic default when the manifest omits it.
    _sigma_bits = getattr(artifacts.manifest, "corridor_sigma_bits", 0)
    corridor_sigma = (
        bits_to_scale_v3(_sigma_bits) if _sigma_bits else _CORRIDOR_SIGMA
    )
    norm_gain_offset, rmsnorm_epsilon = artifacts.rms_norm_parameters()

    # ---- (1) digest binding: proof <-> envelope <-> profile ---------------
    profile_digest = profile.digest()
    if envelope.execution_profile_digest != profile_digest:
        raise _fail("economic envelope is bound to a different profile")
    if proof.execution_profile_digest != profile_digest:
        raise _fail("economic proof is bound to a different execution profile")
    if proof.commitment_envelope_digest != envelope.digest():
        raise _fail("economic proof is bound to a different commitment envelope")
    if envelope.precommit_context_digest != precommit_context.digest():
        raise _fail(
            "economic envelope is bound to a different validator precommit context"
        )

    # ---- (2) the oracle inventory must be frozen in the envelope ----------
    if envelope.execution_root != proof.expected_execution_root():
        raise _fail(
            "oracle inventory is not frozen in the commitment envelope "
            "(execution_root mismatch)"
        )

    # ---- (3) request/output binding (validator-observed stream) -----------
    prompt_token_ids = tuple(int(token) for token in prompt_token_ids)
    observed_output_token_ids = tuple(
        int(token) for token in observed_output_token_ids
    )
    if len(prompt_token_ids) != envelope.context_token_count:
        raise _fail("validator prompt length does not match the envelope")
    if len(observed_output_token_ids) != envelope.decode_token_count:
        raise _fail("validator observed output length does not match the envelope")
    from verallm.proof_v3.request_binding import verify_request_bound_capture_v3

    verify_request_bound_capture_v3(
        signed_bound_digest=proof.signed_bound_digest,
        capture_chain_digest=proof.capture_chain_digest,
        precommit_context=precommit_context,
        observed_output_token_ids=observed_output_token_ids,
        observed_text_utf8=observed_text_utf8,
        finish_reason=finish_reason,
    )

    # ---- (4) EXACT signed inventory + validator-owned dimensions ----------
    layer_universe = tuple(
        sorted(plan.layer_index for plan in profile.relation_spec.layer_audits)
    )
    layer_kinds = {
        int(plan.layer_index): (
            "full_attention" if plan.is_full_attention else "gdn"
        )
        for plan in profile.relation_spec.layer_audits
    }
    if lean:
        from dataclasses import replace

        from verallm.proof_v3.lean_execution_anchor import (
            lean_selected_corridor_layers_v3,
        )

        challenge = replace(
            challenge,
            selected_layer_indices=lean_selected_corridor_layers_v3(
                selected_layer_indices=challenge.selected_layer_indices,
                layer_indices=layer_universe,
            ),
        )
    gdn_runtime_semantics = getattr(
        artifacts, "gdn_runtime_semantics", None
    )
    if "gdn" in layer_kinds.values():
        signed_gdn_semantics = getattr(
            artifacts.manifest,
            "gdn_runtime_semantics_digest",
            b"",
        )
        if (
            gdn_runtime_semantics is None
            or not signed_gdn_semantics
            or gdn_runtime_semantics.digest() != signed_gdn_semantics
        ):
            raise _fail(
                "GDN runtime semantics are not authenticated by the signed "
                "manifest"
            )
    global_layer_index = layer_universe[-1] + 1
    embed_hidden_dim, _embed_vocab_dim = artifacts.dims("embed_tokens")
    _lm_hidden_dim, lm_vocab_dim = artifacts.dims("lm_head")
    if compact_terminal:
        logits_block_cols, logits_block_count = lm_vocab_dim, 0
    else:
        logits_block_cols, logits_block_count = logits_block_geometry_v3(
            decode_rows=envelope.decode_token_count,
            vocab=lm_vocab_dim,
        )
    streaming_inventory = economic_selection_is_streaming_v3(
        challenge.selection_abi_id
    )
    selected_inventory_layers = tuple(
        sorted(challenge.selected_layer_indices)
    )
    _prefix_cache_lanes = ()
    prefix_cache_gdn_windows = ()
    prefix_cache_projection_heads = ()
    prefix_cached_tokens = 0
    if proof.prefix_cache is not None:
        if not profile.relation_spec.cache.allows_prefix_cache_sharing:
            raise _fail(
                "prefix-cache proof is not permitted by the signed profile"
            )
        from verallm.proof_v3.prefix_cache import (
            derive_prefix_cache_executed_suffix_digest_v3,
            verify_prefix_cache_postnonce_v3,
        )

        prefix_cached_tokens = int(
            proof.prefix_cache.commitment.cached_token_count
        )
        if not 0 < prefix_cached_tokens < envelope.context_token_count:
            raise _fail(
                "prefix-cache proof leaves no executed prompt suffix"
            )

        required_prefix_lanes = ()
        if (
            getattr(proof, "attention", None) is not None
            and bool(getattr(artifacts.manifest, "attn_audit_required", 0))
        ):
            from verallm.proof_v3.attention_anchor_binding import (
                AttentionAnchorGeometryV3,
                attention_anchor_geometry_v3,
                derive_prefix_cache_projection_heads_v3,
                required_prefix_cache_attention_lane_keys_v3,
            )
            from verallm.proof_v3.lean_execution_anchor import (
                expected_lean_execution_anchor_reveals_v3,
            )
            from verallm.proof_v3.prefix_cache import (
                derive_prefix_cache_projection_head_lane_keys_v3,
                derive_prefix_cache_projection_lane_keys_v3,
            )

            prefix_attention = _streaming_attention_metadata_v3(
                profile=profile,
                artifacts=artifacts,
                challenge=challenge,
                envelope=envelope,
                capture_chain_digest=proof.capture_chain_digest,
                minimum_candidate_position=prefix_cached_tokens,
            )
            prefix_geometries = {}
            for plan in prefix_attention["plans"]:
                layer = int(plan.layer)
                qkv_width = artifacts.dims(f"l{layer}.qkv")[1]
                o_width = artifacts.dims(f"l{layer}.o")[0]
                geometry = attention_anchor_geometry_v3(
                    qkv_width=qkv_width,
                    o_input_width=o_width,
                    query_heads=int(prefix_attention["nh"]),
                    kv_heads=int(prefix_attention["n_kv"]),
                    head_dim=int(prefix_attention["hd"]),
                    semantics=prefix_attention["semantics"],
                )
                prefix_geometries[layer] = AttentionAnchorGeometryV3(
                    query_heads=geometry.query_heads,
                    kv_heads=geometry.kv_heads,
                    head_dim=geometry.head_dim,
                    qkv_width=2 * geometry.kv_heads * geometry.head_dim,
                    q_block_width=0,
                    k_block_offset=0,
                    v_block_offset=geometry.kv_heads * geometry.head_dim,
                    gated=geometry.gated,
                )
            attention_prefix_lanes = (
                required_prefix_cache_attention_lane_keys_v3(
                    bundle_wire=proof.attention.bundle_wire,
                    plans=prefix_attention["plans"],
                    geometries_by_layer=prefix_geometries,
                    key_count=envelope.context_token_count,
                    cached_token_count=(
                        proof.prefix_cache.commitment.cached_token_count
                    ),
                    block_token_count=(
                        proof.prefix_cache.commitment.block_token_count
                    ),
                )
            )
            prefix_attention_rows = {
                int(plan.layer): tuple(int(row) for row in plan.row_positions)
                for plan in prefix_attention["plans"]
            }
            expected_prefix_reveals = (
                expected_lean_execution_anchor_reveals_v3(
                    challenge=challenge,
                    layer_indices=layer_universe,
                    layer_kinds=layer_kinds,
                    attention_rows_by_layer=prefix_attention_rows,
                    gdn_runtime_semantics=gdn_runtime_semantics,
                    complete_gdn_projection_window=False,
                )
            )
            prefix_projection_positions = tuple(sorted(
                (
                    int(stage_id.split(".", 1)[0][1:]),
                    tuple(int(position) for position in positions),
                )
                for stage_id, positions in expected_prefix_reveals
                if stage_id.endswith(".attention_kv_output")
            ))
            prefix_cache_projection_heads = (
                derive_prefix_cache_projection_heads_v3(
                    plans=prefix_attention["plans"],
                    positions_by_layer=prefix_projection_positions,
                    geometries_by_layer=prefix_geometries,
                    cached_token_count=(
                        proof.prefix_cache.commitment.cached_token_count
                    ),
                )
            )
            prefix_projection_dims = tuple(
                (
                    layer,
                    int(prefix_geometries[layer].kv_heads)
                    * int(prefix_geometries[layer].head_dim),
                )
                for layer, _positions in prefix_projection_positions
            )
            projection_prefix_lanes = (
                derive_prefix_cache_projection_lane_keys_v3(
                    challenge=challenge,
                    positions_by_layer=prefix_projection_positions,
                    kv_dims_by_layer=prefix_projection_dims,
                    cached_token_count=(
                        proof.prefix_cache.commitment.cached_token_count
                    ),
                    block_token_count=(
                        proof.prefix_cache.commitment.block_token_count
                    ),
                )
                if prefix_projection_positions
                else ()
            )
            required_prefix_lanes = tuple(sorted(
                set(attention_prefix_lanes)
                | set(projection_prefix_lanes)
                | set(
                    derive_prefix_cache_projection_head_lane_keys_v3(
                        projection_heads=prefix_cache_projection_heads,
                        head_dim=int(prefix_attention["hd"]),
                        cached_token_count=(
                            proof.prefix_cache.commitment.cached_token_count
                        ),
                        block_token_count=(
                            proof.prefix_cache.commitment.block_token_count
                        ),
                    )
                )
            ))
        try:
            _prefix_cache_lanes = verify_prefix_cache_postnonce_v3(
                section=proof.prefix_cache,
                capture_chain_digest=proof.capture_chain_digest,
                selection_seed=challenge.selection_seed,
                execution_profile_digest=profile_digest,
                prompt_token_root=precommit_context.prompt_token_root,
                prompt_token_ids=prompt_token_ids,
                executed_suffix_digest=(
                    derive_prefix_cache_executed_suffix_digest_v3(
                        proof.execution_anchors
                    )
                ),
                signed_block_token_count=(
                    profile.relation_spec.cache.page_token_count
                ),
                layer_caches=profile.relation_spec.cache.layer_caches,
                selected_layer_indices=selected_inventory_layers,
                required_lane_keys=required_prefix_lanes,
            )
        except (ProofV3Error, ProofV3VerificationError) as exc:
            raise _fail(str(exc)) from exc
    prefix_cache_lane_map = dict(_prefix_cache_lanes)
    _profile_mark("prefix-cache-structure")
    expected_inventory = expected_economic_inventory_v3(
        layer_indices=layer_universe,
        layer_kinds=layer_kinds,
        global_layer_index=global_layer_index,
        logits_block_count=logits_block_count,
        selected_layer_indices=(
            selected_inventory_layers if streaming_inventory else None
        ),
    )
    wire_inventory = tuple(
        (oracle.oracle_id, oracle.phase, oracle.layer_index, oracle.operation)
        for oracle in proof.oracles
    )
    if wire_inventory != expected_inventory:
        raise _fail(
            "oracle inventory does not match the signed profile inventory "
            "(missing, duplicate, reordered or extra oracles)"
        )
    oracle_by_id = {oracle.oracle_id: oracle for oracle in proof.oracles}
    candidate_rows = challenge.candidate_row_count
    embed_hidden = embed_hidden_dim
    lm_hidden, lm_vocab = _lm_hidden_dim, lm_vocab_dim
    if lm_hidden != embed_hidden:
        raise _fail("signed embed/LM-head hidden dimensions are inconsistent")
    for layer in layer_universe:
        layer_rows = (
            None
            if lean and layer in selected_inventory_layers
            else candidate_rows
        )
        for oracle_id in (
            f"l{layer}.residual_in",
            f"l{layer}.residual_out",
        ):
            oracle = oracle_by_id[oracle_id]
            if (
                (
                    layer_rows is not None
                    and oracle.row_count != layer_rows
                )
                or oracle.col_count != embed_hidden
            ):
                raise _fail(
                    f"oracle {oracle_id} dimensions do not match the signed "
                    "artifacts"
                )
        if streaming_inventory and layer not in selected_inventory_layers:
            continue
        gu_in, gu_out = artifacts.dims(f"l{layer}.gate_up")
        dn_in, dn_out = artifacts.dims(f"l{layer}.down")
        norm_in, norm_out = artifacts.dims(f"l{layer}.input_norm")
        post_in, post_out = artifacts.dims(f"l{layer}.post_norm")
        if (
            gu_in != embed_hidden
            or dn_out != embed_hidden
            or norm_in != embed_hidden
            or post_in != embed_hidden
            or norm_out != 1
            or post_out != 1
            or gu_out != 2 * dn_in
        ):
            raise _fail(
                f"layer {layer} signed projection dimensions are inconsistent"
            )
        common_checks = (
            (f"l{layer}.gate_up_x", layer_rows, gu_in),
            (f"l{layer}.gate_up_s", layer_rows, gu_out),
            (f"l{layer}.gate_up_y", layer_rows, gu_out),
            (f"l{layer}.down_x", layer_rows, dn_in),
            (f"l{layer}.down_s", layer_rows, dn_out),
            (f"l{layer}.down_y", layer_rows, dn_out),
            (f"l{layer}.mid_residual", layer_rows, embed_hidden),
        )
        if layer_kinds[layer] == "full_attention":
            qkv_in, qkv_out = artifacts.dims(f"l{layer}.qkv")
            o_in, o_out = artifacts.dims(f"l{layer}.o")
            from verallm.proof_v3.attention_runtime_semantics import (
                Q_GATE_INTERLEAVED_LAYOUT_V3,
            )

            attention_semantics = getattr(
                artifacts, "attention_runtime_semantics", None
            )
            q_block_width = o_in * (
                2
                if (
                    attention_semantics is not None
                    and attention_semantics.qkv_layout_id
                    == Q_GATE_INTERLEAVED_LAYOUT_V3
                )
                else 1
            )
            if (
                qkv_in != embed_hidden
                or o_out != embed_hidden
                or qkv_out <= q_block_width
                or (qkv_out - q_block_width) % 2 != 0
            ):
                raise _fail(
                    f"layer {layer} signed attention projection dimensions "
                    "are inconsistent"
                )
            kv_dim = (qkv_out - q_block_width) // 2
            architecture_checks = (
                (f"l{layer}.qkv_x", layer_rows, qkv_in),
                (f"l{layer}.qkv_s", layer_rows, qkv_out),
                (f"l{layer}.attn_o_x", layer_rows, o_in),
                (f"l{layer}.attn_o_s", layer_rows, o_out),
                (f"l{layer}.attn_o_y", layer_rows, o_out),
                (f"l{layer}.q_cache", layer_rows, q_block_width),
                (f"l{layer}.k_cache", layer_rows, kv_dim),
                (f"l{layer}.v_cache", layer_rows, kv_dim),
            )
        else:
            qkvz_in, qkvz_out = artifacts.dims(f"l{layer}.gdn_qkvz")
            ba_in, ba_out = artifacts.dims(f"l{layer}.gdn_ba")
            gdn_o_in, gdn_o_out = artifacts.dims(f"l{layer}.gdn_o")
            parameters = gdn_runtime_semantics.layer_for(
                layer
            ).parameters()
            conv_width = (
                2 * parameters.num_key_heads * parameters.key_head_dim
                + parameters.num_value_heads * parameters.value_head_dim
            )
            value_width = (
                parameters.num_value_heads * parameters.value_head_dim
            )
            if (
                qkvz_in != embed_hidden
                or ba_in != embed_hidden
                or qkvz_out != conv_width + value_width
                or ba_out != 2 * parameters.num_value_heads
                or gdn_o_in != value_width
                or gdn_o_out != embed_hidden
            ):
                raise _fail(
                    f"layer {layer} signed GDN projection dimensions "
                    "disagree with the authenticated runtime semantics"
                )
            architecture_checks = (
                (f"l{layer}.gdn_qkvz_x", layer_rows, qkvz_in),
                (f"l{layer}.gdn_qkvz_s", layer_rows, qkvz_out),
                (f"l{layer}.gdn_qkvz_y", layer_rows, qkvz_out),
                (f"l{layer}.gdn_ba_x", layer_rows, ba_in),
                (f"l{layer}.gdn_ba_s", layer_rows, ba_out),
                (f"l{layer}.gdn_ba_y", layer_rows, ba_out),
                (f"l{layer}.gdn_o_x", layer_rows, gdn_o_in),
                (f"l{layer}.gdn_o_s", layer_rows, gdn_o_out),
                (f"l{layer}.gdn_o_y", layer_rows, gdn_o_out),
            )
        checks = architecture_checks + common_checks
        for oracle_id, rows, cols in checks:
            oracle = oracle_by_id[oracle_id]
            if (
                (rows is not None and oracle.row_count != rows)
                or oracle.col_count != cols
            ):
                raise _fail(
                    f"oracle {oracle_id} dimensions do not match the signed "
                    "artifacts"
                )
    final_oracle = oracle_by_id["final_hidden"]
    if (
        final_oracle.row_count != envelope.decode_token_count
        or final_oracle.col_count != lm_hidden
    ):
        raise _fail("final_hidden oracle dimensions do not match the artifacts")
    for block in range(logits_block_count):
        block_oracle = oracle_by_id[logits_block_oracle_id_v3(block)]
        expected_cols = min(
            logits_block_cols, lm_vocab - block * logits_block_cols
        )
        if (
            block_oracle.row_count != envelope.decode_token_count
            or block_oracle.col_count != expected_cols
        ):
            raise _fail(
                "logits block oracle dimensions do not match the artifacts"
            )
    if streaming_inventory:
        from verallm.proof_v3.economic_challenge import (
            PROMPT_CANDIDATE_ROWS_V3,
        )

        stamp_input = oracle_by_id["response_stamp_input"]
        if (
            stamp_input.row_count
            != min(envelope.context_token_count, PROMPT_CANDIDATE_ROWS_V3)
            or stamp_input.col_count != embed_hidden
            or stamp_input.scale_bits
            != scale_to_bits_v3(artifacts.scale_for("embed_tokens"))
        ):
            raise _fail(
                "response-stamp input oracle dimensions or scale do not "
                "match the authenticated request and embedding artifact"
            )

    # ---- (4.5) streaming full-sequence membership -------------------------
    anchor_binding = None
    anchor_rows = None
    anchor_encoding = None
    streaming_attention_metadata = None
    gdn_lane_keys = ()
    transition_lane_keys = ()
    lean_row_layouts = {}
    lean_tokens_by_layer = {}
    lean_positions_by_layer = {}
    kv_positions_by_layer = {}
    if challenge.selection_abi_id == ECONOMIC_SELECTION_ABI_V3:
        if (
            proof.execution_anchors
            or proof.execution_anchor_reveals
            or proof.execution_anchor_lane_reveals
        ):
            raise _fail(
                "legacy economic selection must not carry execution anchors"
            )
    elif economic_selection_is_streaming_v3(challenge.selection_abi_id):
        if bool(getattr(artifacts.manifest, "attn_audit_required", 0)):
            streaming_attention_metadata = _streaming_attention_metadata_v3(
                profile=profile,
                artifacts=artifacts,
                challenge=challenge,
                envelope=envelope,
                capture_chain_digest=proof.capture_chain_digest,
                minimum_candidate_position=prefix_cached_tokens,
            )
        projection_dims = {}
        for layer in layer_universe:
            names = (
                ("qkv", "o")
                if layer_kinds[layer] == "full_attention"
                else ("gdn_qkvz", "gdn_ba", "gdn_o")
            ) + ("gate_up", "down")
            for name in names:
                projection_dims[f"l{layer}.{name}"] = artifacts.dims(
                    f"l{layer}.{name}"
                )
        if lean:
            from verallm.proof_v3.lean_execution_anchor import (
                expected_lean_execution_anchor_inventory_v3,
                expected_lean_execution_anchor_reveals_v3,
            )

            audits = {
                int(plan.layer_index): plan
                for plan in profile.relation_spec.layer_audits
            }
            attention_kv_widths = {
                layer: (
                    2
                    * int(audits[layer].attention_key_value_head_count)
                    * int(audits[layer].attention_head_dimension)
                )
                for layer in layer_universe
                if layer_kinds[layer] == "full_attention"
            }
            expected_anchors = expected_lean_execution_anchor_inventory_v3(
                layer_indices=layer_universe,
                layer_kinds=layer_kinds,
                sequence_token_count=challenge.sequence_token_count,
                hidden_dim=embed_hidden,
                projection_dims=projection_dims,
                attention_kv_widths=attention_kv_widths,
                gdn_runtime_semantics=getattr(
                    artifacts, "gdn_runtime_semantics", None
                ),
                context_token_count=challenge.context_token_count,
            )
            expected_anchor_reveals = (
                expected_lean_execution_anchor_reveals_v3(
                    challenge=challenge,
                    layer_indices=layer_universe,
                    layer_kinds=layer_kinds,
                    attention_rows_by_layer=(
                        None
                        if streaming_attention_metadata is None
                        else {
                            int(plan.layer): tuple(plan.row_positions)
                            for plan in streaming_attention_metadata["plans"]
                        }
                    ),
                    gdn_runtime_semantics=getattr(
                        artifacts, "gdn_runtime_semantics", None
                    ),
                    complete_gdn_projection_window=selected_trace_profile,
                )
            )
        else:
            expected_anchors = expected_economic_execution_anchor_inventory_v3(
                layer_indices=layer_universe,
                layer_kinds=layer_kinds,
                sequence_token_count=challenge.sequence_token_count,
                hidden_dim=embed_hidden,
                projection_dims=projection_dims,
                gdn_runtime_semantics=getattr(
                    artifacts, "gdn_runtime_semantics", None
                ),
                context_token_count=challenge.context_token_count,
            )
            expected_anchor_reveals = (
                expected_economic_execution_anchor_reveals_v3(
                    challenge=challenge,
                    layer_indices=layer_universe,
                    layer_kinds=layer_kinds,
                    attention_rows_by_layer=(
                        None
                        if streaming_attention_metadata is None
                        else {
                            int(plan.layer): tuple(plan.row_positions)
                            for plan in streaming_attention_metadata["plans"]
                        }
                    ),
                )
            )
        # Projection rows are indexed in the complete request, even when the
        # execution-anchor tree for a cache hit contains only the executed
        # suffix.  Preserve the validator-derived absolute K/V coordinates
        # before converting anchor-opening rows to suffix-local indices.
        absolute_attention_kv_positions = {
            int(stage_id.split(".", 1)[0][1:]): tuple(positions)
            for stage_id, positions in expected_anchor_reveals
            if stage_id.endswith(".attention_kv_output")
        }
        if prefix_cached_tokens:
            static_suffixes = (
                ".gdn_conv_prompt_boundary",
                ".gdn_recurrent_prompt_boundary",
                ".gdn_conv_decode_checkpoints",
                ".gdn_recurrent_decode_checkpoints",
            )
            suffix_rows = challenge.sequence_token_count - prefix_cached_tokens
            expected_anchors = tuple(
                (
                    stage_id,
                    (
                        row_count
                        if stage_id.endswith(static_suffixes)
                        else suffix_rows
                    ),
                    row_width,
                )
                for stage_id, row_count, row_width in expected_anchors
            )
            expected_anchor_reveals = tuple(
                (stage_id, mapped)
                for stage_id, positions in expected_anchor_reveals
                for mapped in (
                    (
                        tuple(int(position) for position in positions)
                        if stage_id.endswith(static_suffixes)
                        else tuple(
                            int(position) - prefix_cached_tokens
                            for position in positions
                            if int(position) >= prefix_cached_tokens
                        )
                    ),
                )
                if mapped
            )
        if selected_trace:
            actual_anchors = tuple(
                (item.stage_id, item.row_count, item.row_width)
                for item in proof.execution_anchors
            )
            if actual_anchors != tuple(expected_anchors):
                raise _fail(
                    "execution anchor inventory does not match the signed "
                    "geometry"
                )
            if (
                proof.execution_anchor_reveals
                or proof.execution_anchor_lane_reveals
            ):
                raise _fail(
                    "selected trace must not carry legacy outer anchor "
                    "openings"
                )
            anchor_rows = {}
        else:
            anchor_rows = verify_economic_execution_anchor_reveals_v3(
                commitments=proof.execution_anchors,
                reveals=proof.execution_anchor_reveals,
                expected_inventory=expected_anchors,
                expected_reveals=expected_anchor_reveals,
            )
        if (
            lean
            and not selected_trace
            and any(
                layer_kinds[layer] == "gdn"
                for layer in challenge.selected_layer_indices
            )
        ):
            from verallm.proof_v3.economic_gdn_replay import (
                required_lean_gdn_checkpoint_lane_keys_v3,
            )

            gdn_lane_keys = required_lean_gdn_checkpoint_lane_keys_v3(
                commitments=proof.execution_anchors,
                challenge=challenge,
                layer_kinds=layer_kinds,
                semantics=gdn_runtime_semantics,
            )
        if lean:
            from verallm.proof_v3.gdn_decode_corridor import (
                derive_gdn_decode_corridor_for_challenge_v3,
            )
            from verallm.proof_v3.lean_execution_anchor import (
                lean_projection_row_layouts_v3,
            )

            checkpointed_gdn = bool(
                artifacts.gdn_runtime_semantics is not None
                and artifacts.gdn_runtime_semantics
                .decode_checkpoint_stride
            )
            gdn_decode_layers = (
                tuple(
                    layer
                    for layer in challenge.selected_layer_indices
                    if layer_kinds[layer] == "gdn"
                )
                if selected_trace or checkpointed_gdn
                else ()
            )
            gdn_decode_positions = (
                {
                    layer: (
                        derive_gdn_decode_corridor_for_challenge_v3(
                            challenge=challenge,
                            semantics=(
                                artifacts.gdn_runtime_semantics.layer_for(
                                    layer
                                )
                            ),
                        ).sequence_positions
                    )
                    for layer in gdn_decode_layers
                }
                if (
                    gdn_decode_layers
                    and artifacts.gdn_runtime_semantics is not None
                    and artifacts.gdn_runtime_semantics
                    .decode_checkpoint_stride
                )
                else None
            )
            kv_positions_by_layer = absolute_attention_kv_positions
            lean_row_layouts = dict(
                lean_projection_row_layouts_v3(
                    challenge=challenge,
                    layer_indices=layer_universe,
                    attention_kv_positions_by_layer=kv_positions_by_layer,
                    gdn_decode_only_layers=(
                        gdn_decode_layers if selected_trace else ()
                    ),
                    gdn_decode_positions_by_layer=gdn_decode_positions,
                    gdn_prefix_positions_by_layer={
                        window.layer_index: window.sequence_positions
                        for window in prefix_cache_gdn_windows
                    },
                )
            )
            lean_tokens_by_layer = {
                layer: tuple(
                    row_index for _position, row_index in layout
                )
                for layer, layout in lean_row_layouts.items()
            }
            lean_positions_by_layer = {
                layer: {
                    row_index: position
                    for position, row_index in layout
                }
                for layer, layout in lean_row_layouts.items()
            }
            for layer, layout in lean_row_layouts.items():
                expected_rows = challenge.candidate_row_count + sum(
                    1
                    for position, _row_index in layout
                    if position
                    not in set(challenge.candidate_sequence_positions)
                )
                for oracle in proof.oracles:
                    if (
                        oracle.layer_index == layer
                        and oracle.row_count != expected_rows
                    ):
                        raise _fail(
                            f"lean layer {layer} oracle rows do not match "
                            "the nonce-derived corridor layout"
                        )
            transition_lane_keys = ()
            transition_anchor_cells = {}
        else:
            verify_economic_execution_anchor_residual_chain_v3(
                commitments=proof.execution_anchors,
                layer_indices=layer_universe,
            )
            transition_lane_keys = (
                required_economic_transition_anchor_lane_keys_v3(
                    commitments=proof.execution_anchors,
                    challenge=challenge,
                    layer_indices=layer_universe,
                    layer_kinds=layer_kinds,
                    projection_dims=projection_dims,
                    hidden_dim=embed_hidden,
                )
            )
            transition_anchor_cells = (
                verify_economic_transition_anchor_lanes_v3(
                    commitments=proof.execution_anchors,
                    lane_reveals=proof.execution_anchor_lane_reveals,
                    challenge=challenge,
                    layer_indices=layer_universe,
                    layer_kinds=layer_kinds,
                    projection_dims=projection_dims,
                    hidden_dim=embed_hidden,
                )
            )
        anchor_encoding = economic_execution_anchor_encoding_v3(profile)
        if not lean:
            anchor_binding = derive_economic_execution_anchor_oracle_binding_v3(
                opened_rows=anchor_rows,
                opened_cells=transition_anchor_cells,
                challenge=challenge,
                layer_indices=layer_universe,
                layer_kinds=layer_kinds,
                oracle_by_id=oracle_by_id,
                projection_dims=projection_dims,
                embedding_scale=artifacts.scale_for("embed_tokens"),
                encoding_id=anchor_encoding,
                attention_runtime_semantics=getattr(
                    artifacts, "attention_runtime_semantics", None
                ),
            )
        if not lean and any(
            layer_kinds[layer] == "gdn"
            for layer in challenge.selected_layer_indices
        ):
            from verallm.proof_v3.economic_gdn_replay import (
                required_economic_gdn_anchor_lane_keys_v3,
                verify_economic_gdn_replay_v3,
            )

            gdn_lane_keys = required_economic_gdn_anchor_lane_keys_v3(
                commitments=proof.execution_anchors,
                challenge=challenge,
                layer_kinds=layer_kinds,
                semantics=gdn_runtime_semantics,
            )
            verify_economic_gdn_replay_v3(
                opened_rows=anchor_rows,
                challenge=challenge,
                layer_kinds=layer_kinds,
                semantics=gdn_runtime_semantics,
                anchor_encoding_id=anchor_encoding,
                commitments=proof.execution_anchors,
                lane_reveals=proof.execution_anchor_lane_reveals,
            )
        if (
            streaming_attention_metadata is None
            and tuple(
                (
                    int(reveal.commitment_index),
                    int(reveal.opening.row_index),
                    int(reveal.opening.lane_index),
                )
                for reveal in proof.execution_anchor_lane_reveals
            )
            != tuple(
                sorted(
                    set(gdn_lane_keys) | set(transition_lane_keys)
                )
            )
        ):
            raise _fail(
                "streaming anchor lanes do not match the nonce-selected "
                "GDN coordinates"
            )
    else:
        raise _fail("economic selection ABI is not supported")

    base_binding = precommit_context.digest()

    _profile_mark("admission-and-inventory")

    if selected_trace:
        try:
            from verallm.proof_v3.goldilocks_selected_trace import (
                verify_goldilocks_selected_trace_v3,
            )
            from verallm.proof_v3.goldilocks_selected_trace_context import (
                build_goldilocks_selected_trace_context_v3,
            )
            from verallm.proof_v3.goldilocks_selected_trace_wire import (
                decode_goldilocks_selected_trace_v3,
            )

            selected_proof = decode_goldilocks_selected_trace_v3(
                proof.selected_trace_wire
            )
            selected_context = (
                build_goldilocks_selected_trace_context_v3(
                    attention_capture_roots_by_layer=(
                        selected_proof.attention_capture_roots_by_layer
                    ),
                    rmsnorm_weight_rows=(
                        selected_proof.rmsnorm_weight_rows
                    ),
                    projection_bias_rows=(
                        selected_proof.projection_bias_rows
                    ),
                    envelope_digest=envelope.digest(),
                    capture_base_binding_digest=base_binding,
                    capture_chain_digest=proof.capture_chain_digest,
                    challenge=challenge,
                    layer_universe=layer_universe,
                    layer_kinds=layer_kinds,
                    row_layouts=lean_row_layouts,
                    attention_metadata=streaming_attention_metadata,
                    attention_kv_positions_by_layer=kv_positions_by_layer,
                    artifacts=artifacts,
                    oracles=proof.oracles,
                    execution_anchors=proof.execution_anchors,
                    anchor_encoding_id=anchor_encoding,
                    prompt_token_ids=prompt_token_ids,
                    observed_output_token_ids=observed_output_token_ids,
                )
            )
            verify_goldilocks_selected_trace_v3(
                selected_proof,
                context=selected_context,
            )
        except ProofV3VerificationError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            ProofV3Error,
        ) as exc:
            raise _fail(
                f"selected-trace verification failed: {exc}"
            ) from exc
        _profile_mark("selected-trace")
        return None

    # ---- (5) architecture-specific projection audits ---------------------
    selected_layers = tuple(sorted(challenge.selected_layer_indices))
    from verallm.proof_v3.economic_profile import (
        economic_profile_uses_canonical_gdn_input_v3,
        economic_profile_uses_exact_mlp_activation_v3,
    )

    exact_mlp_activation = economic_profile_uses_exact_mlp_activation_v3(
        profile
    )
    canonical_gdn_projection_inputs = (
        economic_profile_uses_canonical_gdn_input_v3(profile)
    )
    expected_reveal_keys = tuple(
        (layer, x_suffix, s_suffix, manifest_suffix)
        for layer in selected_layers
        for (x_suffix, s_suffix, manifest_suffix) in (
            audited_projections_for_layer_kind_v3(layer_kinds[layer])
        )
    )
    if len(proof.projections) != len(expected_reveal_keys):
        raise _fail(
            "projection reveals do not cover exactly the selected layers "
            "and audited projections"
        )
    expected_tokens = tuple(challenge.sampled_token_rows)
    # opened material retained for the coupling checks:
    # (layer, manifest_suffix) -> (x_rows, surrogate_cells, weight_rows,
    #                              x_scale, w_scale, expected_outs)
    opened_projections: dict = {}
    projection_tokens_by_layer: dict[int, tuple[int, ...]] = {}
    gdn_projection_output_columns_by_key = {}
    gdn_recurrence_input_columns_by_layer = {}
    attention_query_output_columns_by_layer = {}

    def _projection_weight_row_sq(
        *,
        layer: int,
        suffix: str,
        out: int,
        weight_rows: dict[int, tuple[int, ...]],
    ) -> int:
        row = weight_rows.get(out)
        if row is not None:
            return sum(value * value for value in row)
        if succinct_projection or (
            complete_projection and suffix == "qkv"
        ):
            return artifacts.weight_row_sq(f"l{layer}.{suffix}", out)
        raise _fail(
            f"projection l{layer}.{suffix} has no authenticated weight "
            f"material for output row {out}"
        )

    projection_opening_profile = {
        "x": 0.0,
        "anchor": 0.0,
        "surrogate": 0.0,
        "weights": 0.0,
        "recompute": 0.0,
        "claim_build": 0.0,
    }

    if lean:
        from verallm.proof_v3.economic_gdn_replay import (
            economic_gdn_runtime_columns_v3,
        )

        for layer in selected_layers:
            if layer_kinds[layer] != "gdn":
                continue
            parameters = (
                gdn_runtime_semantics.layer_for(layer)
                .parameters()
                .replay_parameters()
            )
            heads = challenge.gdn_value_heads_for(
                layer_index=layer,
                num_key_heads=parameters.num_key_heads,
                num_value_heads=parameters.num_value_heads,
            )
            qkvz_columns, ba_columns, recurrence_input_columns = (
                economic_gdn_runtime_columns_v3(
                    parameters=parameters,
                    selected_value_heads=heads,
                )
            )
            gdn_projection_output_columns_by_key[
                (layer, "gdn_qkvz")
            ] = qkvz_columns
            gdn_projection_output_columns_by_key[
                (layer, "gdn_ba")
            ] = ba_columns
            gdn_recurrence_input_columns_by_layer[
                layer
            ] = recurrence_input_columns
    if lean and streaming_attention_metadata is not None:
        from verallm.proof_v3.attention_runtime_semantics import (
            Q_GATE_INTERLEAVED_LAYOUT_V3,
        )

        head_dim = int(streaming_attention_metadata["hd"])
        gated_q = (
            streaming_attention_metadata["semantics"].qkv_layout_id
            == Q_GATE_INTERLEAVED_LAYOUT_V3
        )
        head_stride = head_dim * (2 if gated_q else 1)
        attention_query_output_columns_by_layer = {
            int(plan.layer): tuple(
                sorted(
                    column
                    for head in plan.heads
                    for column in range(
                        int(head) * head_stride,
                        (int(head) + 1) * head_stride,
                    )
                )
            )
            for plan in streaming_attention_metadata["plans"]
        }
        prefix_heads_by_layer: dict[int, set[int]] = {}
        for layer, kv_head, _position in prefix_cache_projection_heads:
            prefix_heads_by_layer.setdefault(int(layer), set()).add(
                int(kv_head)
            )
        attention_prefix_output_columns_by_layer = {}
        for layer, heads in prefix_heads_by_layer.items():
            qkv_out = artifacts.dims(f"l{layer}.qkv")[1]
            kv_dim = (
                int(streaming_attention_metadata["n_kv"])
                * head_dim
            )
            q_width = qkv_out - 2 * kv_dim
            attention_prefix_output_columns_by_layer[layer] = tuple(
                sorted(
                    column
                    for head in heads
                    for base in (q_width, q_width + kv_dim)
                    for column in range(
                        base + head * head_dim,
                        base + (head + 1) * head_dim,
                    )
                )
            )
    else:
        attention_prefix_output_columns_by_layer = {}
    lean_claims = []
    succinct_claims = []
    for reveal, (layer, x_suffix, s_suffix, manifest_suffix) in zip(
        proof.projections, expected_reveal_keys, strict=True
    ):
        layer_tokens = (
            lean_tokens_by_layer[layer]
            if lean
            else expected_tokens
        )
        projection_tokens_by_layer[layer] = layer_tokens
        x_oracle = proof.oracles[reveal.x_oracle_index]
        s_oracle = proof.oracles[reveal.s_oracle_index]
        if (
            x_oracle.oracle_id != f"l{layer}.{x_suffix}"
            or s_oracle.oracle_id != f"l{layer}.{s_suffix}"
            or reveal.manifest_name != f"l{layer}.{manifest_suffix}"
        ):
            raise _fail("projection reveal references the wrong oracles")
        if reveal.token_indices != layer_tokens:
            raise _fail(
                "projection tokens are not the validator-derived selection"
            )
        required_runtime_columns = tuple(
            sorted(
                set(
                    gdn_projection_output_columns_by_key.get(
                        (layer, manifest_suffix), ()
                    )
                )
                | (
                    set(
                        attention_query_output_columns_by_layer.get(
                            layer, ()
                        )
                    )
                    if (
                        layer_kinds[layer] == "full_attention"
                        and manifest_suffix == "qkv"
                    )
                    else set()
                )
                | (
                    set(
                        attention_prefix_output_columns_by_layer.get(
                            layer, ()
                        )
                    )
                    if (
                        layer_kinds[layer] == "full_attention"
                        and manifest_suffix == "qkv"
                    )
                    else set()
                )
                | (
                    {
                        output
                        for column in challenge.mlp_cols_for(
                            layer_index=layer,
                            inter_dim=s_oracle.col_count // 2,
                        )
                        for output in (
                            column,
                            s_oracle.col_count // 2 + column,
                        )
                    }
                    if (
                        exact_mlp_activation
                        and manifest_suffix == "gate_up"
                        and s_oracle.col_count % 2 == 0
                    )
                    else set()
                )
            )
        )
        expected_outs = challenge.projection_binding_columns_for(
            layer_index=layer,
            layer_kind=layer_kinds[layer],
            projection=manifest_suffix,
            out_dim=s_oracle.col_count,
            kv_dim=(
                oracle_by_id[f"l{layer}.k_cache"].col_count
                if (
                    layer_kinds[layer] == "full_attention"
                    and manifest_suffix == "qkv"
                )
                else 0
            ),
            required_runtime_columns=required_runtime_columns,
        )
        opened_outs = (
            tuple(range(s_oracle.col_count))
            if complete_projection
            else expected_outs
        )
        if complete_projection:
            if (
                not reveal.complete_output
                or reveal.succinct_output
                or reveal.out_indices
            ):
                raise _fail(
                    "full-row escalation does not carry a complete output opening"
                )
        elif succinct_projection:
            if (
                reveal.complete_output
                or not reveal.succinct_output
                or reveal.out_indices != expected_outs
            ):
                raise _fail(
                    "succinct projection reveal does not match the "
                    "validator-derived selection"
                )
        elif (
            reveal.complete_output
            or reveal.succinct_output
            or reveal.out_indices != expected_outs
        ):
            raise _fail(
                "projection output cells are not the validator-derived selection"
            )
        packed_x = dict(zip(layer_tokens, reveal.x_rows, strict=True))
        section_started = _time.perf_counter()
        x_rows = _opened_rows(
            oracle=x_oracle,
            base_binding=base_binding,
            rows=layer_tokens,
            opening=reveal.x_opening,
            require_int8=True,
            what=f"projection l{layer}.{manifest_suffix} X",
            expect_mode=0,
            packed_rows=packed_x,
        )
        section_finished = _time.perf_counter()
        projection_opening_profile["x"] += section_finished - section_started
        section_started = section_finished
        if anchor_binding is not None:
            anchor_binding.verify_rows(
                oracle_id=x_oracle.oracle_id,
                actual_rows=x_rows,
                row_indices=layer_tokens,
            )
        section_finished = _time.perf_counter()
        projection_opening_profile["anchor"] += (
            section_finished - section_started
        )
        # Fabrication signature: EVERY nonce-sampled X row identically
        # zero (a lazy all-zero capture makes the surrogate check
        # vacuous).  A single zero row is legitimate under global int8
        # absmax scaling at long repetitive contexts, cannot be targeted
        # pre-nonce, and stays tied to its residual row by the norm-link
        # couplings.
        if all(
            all(value == 0 for value in x_rows[token])
            for token in layer_tokens
        ):
            raise _fail(
                f"projection l{layer}.{manifest_suffix} X rows are "
                "identically zero (fabricated capture)"
            )
        section_started = _time.perf_counter()
        if complete_projection:
            # Complete-output hard proofs open every cell of each selected
            # row. Keep them row-packed instead of constructing a second
            # multi-million-entry (row, column) dictionary.
            surrogate = _opened_rows(
                oracle=s_oracle,
                base_binding=base_binding,
                rows=layer_tokens,
                opening=reveal.s_opening,
                require_int8=False,
                what=f"projection l{layer}.{manifest_suffix} surrogate",
                expect_mode=3,
                expect_bounded_width=bounded_byte_width_v3(
                    artifacts.entry(reveal.manifest_name).in_dim
                ),
            )
        else:
            surrogate = _opened_cells(
                oracle=s_oracle,
                base_binding=base_binding,
                cells=tuple(
                    (token, out)
                    for token in layer_tokens
                    for out in opened_outs
                ),
                opening=reveal.s_opening,
                what=f"projection l{layer}.{manifest_suffix} surrogate",
                expect_mode=3,
                expect_bounded_width=bounded_byte_width_v3(
                    artifacts.entry(reveal.manifest_name).in_dim
                ),
            )
        section_finished = _time.perf_counter()
        projection_opening_profile["surrogate"] += (
            section_finished - section_started
        )
        weight_rows: dict[int, tuple[int, ...]] = {}
        section_started = section_finished
        for row_reveal in reveal.weight_rows:
            weight_rows[row_reveal.row_index] = artifacts.verify_weight_row(
                name=reveal.manifest_name, reveal=row_reveal
            )
        section_finished = _time.perf_counter()
        projection_opening_profile["weights"] += (
            section_finished - section_started
        )
        signed_norm_qkv = (
            complete_projection and manifest_suffix == "qkv"
        )
        batch_authenticated = signed_norm_qkv or succinct_projection
        expected_weight_rows = () if batch_authenticated else expected_outs
        if tuple(sorted(weight_rows)) != expected_weight_rows:
            raise _fail(
                "projection sampled weight rows do not match the "
                "validator-derived coupling coordinates"
            )
        # Exact int8 recompute at the challenged cells. int8 * int8 over the
        # projection width d maxes at 127*127*d (~1.4e7 at d=896), far inside
        # int64, so an int64 matmul is bit-identical to the per-pair Python
        # sum -- one C-level GEMM replaces ~d * |cells| Python mul-adds. The
        # per-pair loop stays as an exact fallback for ragged/degenerate rows.
        section_started = _time.perf_counter()
        if not batch_authenticated:
            toks = tuple(layer_tokens)
            outs = tuple(expected_outs)
            prod = None
            try:
                import numpy as _np

                x_mat = _np.array(
                    [x_rows[t] for t in toks], dtype=_np.int64
                )
                w_mat = _np.array(
                    [weight_rows[o] for o in outs], dtype=_np.int64
                )
                if x_mat.ndim == 2 and w_mat.ndim == 2 and (
                    x_mat.shape[1] == w_mat.shape[1]
                ):
                    prod = (x_mat @ w_mat.T).tolist()
            except (ValueError, TypeError):
                prod = None
            if prod is not None:
                for ti, token in enumerate(toks):
                    prow = prod[ti]
                    for oi, out in enumerate(outs):
                        if prow[oi] != _projection_surrogate_value(
                            surrogate, token, out
                        ):
                            raise _fail(
                                f"projection l{layer}.{manifest_suffix} exact "
                                "int8 recompute != committed surrogate -- "
                                "served weight is not the registered weight "
                                "(substitution)"
                            )
            else:
                for token in toks:
                    x_row = x_rows[token]
                    for out in outs:
                        w_row = weight_rows[out]
                        recomputed = sum(
                            a * b
                            for a, b in zip(x_row, w_row, strict=True)
                        )
                        if recomputed != _projection_surrogate_value(
                            surrogate, token, out
                        ):
                            raise _fail(
                                f"projection l{layer}.{manifest_suffix} exact "
                                "int8 recompute != committed surrogate -- "
                                "served weight is not the registered weight "
                                "(substitution)"
                            )
        section_finished = _time.perf_counter()
        projection_opening_profile["recompute"] += (
            section_finished - section_started
        )
        section_started = section_finished
        if complete_projection or succinct_projection:
            from verallm.proof_v3.lean_projection_batch import (
                LeanProjectionBatchClaimV3,
            )
            from verallm.proof_v3.lean_projection_fold import (
                LeanProjectionCatalogV3,
                lean_projection_operation_key_v3,
            )

            catalog = getattr(artifacts, "lean_projection_catalog", None)
            if not isinstance(catalog, LeanProjectionCatalogV3):
                raise _fail(
                    "lean profile has no authenticated complete projection catalog"
                )
            operation = catalog.operation(
                lean_projection_operation_key_v3(
                    layer_index=layer,
                    projection=manifest_suffix,
                )
            )
            for token in layer_tokens:
                if succinct_projection:
                    from verallm.proof_v3.succinct_projection_batch import (
                        SuccinctProjectionClaimV3,
                    )

                    succinct_claims.append(
                        SuccinctProjectionClaimV3(
                            operation=operation,
                            input_row_i8=x_rows[token],
                            surrogate_oracle=s_oracle,
                            row_index=token,
                            output_columns=expected_outs,
                        )
                    )
                    continue
                # Complete-output lean openings are already row-packed and
                # ordered across every output cell. Reuse that authenticated
                # tuple directly instead of rebuilding it one Python integer
                # at a time for every projection claim.
                surrogate_row = surrogate.get(token)
                if surrogate_row is None or len(surrogate_row) != len(
                    opened_outs
                ):
                    raise _fail(
                        f"projection l{layer}.{manifest_suffix} surrogate row "
                        "is incomplete"
                    )
                lean_claims.append(
                    LeanProjectionBatchClaimV3(
                        operation=operation,
                        input_row_i8=x_rows[token],
                        surrogate_output_i64=surrogate_row,
                    )
                )
        projection_opening_profile["claim_build"] += (
            _time.perf_counter() - section_started
        )
        opened_projections[(layer, manifest_suffix)] = (
            x_rows,
            surrogate,
            weight_rows,
            bits_to_scale_v3(x_oracle.scale_bits),
            artifacts.scale_for(reveal.manifest_name),
            expected_outs,
        )

    _profile_mark("projection-openings")
    if _profile_timing:
        print(
            "[PROOF-V3-HARD-VERIFY] projection_opening_detail "
            + " ".join(
                f"{name}={seconds:.3f}s"
                for name, seconds in projection_opening_profile.items()
            ),
            flush=True,
        )

    if complete_projection:
        from verallm.proof_v3.lean_projection_batch import (
            decode_lean_projection_batch_v3,
            verify_lean_projection_batch_v3,
        )

        try:
            verify_lean_projection_batch_v3(
                proof=decode_lean_projection_batch_v3(
                    proof.lean_projection_batch_wire
                ),
                validator_binding_digest=envelope.digest(),
                validator_nonce=bytes(challenge.selection_seed),
                claims=tuple(lean_claims),
            )
        except (ProofV3Error, ProofV3VerificationError) as exc:
            raise _fail(
                f"complete projection batch verification failed: {exc}"
            )
    elif succinct_projection:
        from verallm.proof_v3.succinct_projection_batch import (
            decode_succinct_projection_batch_v3,
            verify_succinct_projection_batch_v3,
        )

        try:
            verify_succinct_projection_batch_v3(
                proof=decode_succinct_projection_batch_v3(
                    proof.succinct_projection_batch_wire
                ),
                validator_binding_digest=envelope.digest(),
                capture_base_binding_digest=base_binding,
                validator_nonce=bytes(challenge.selection_seed),
                claims=tuple(succinct_claims),
            )
        except (ProofV3Error, ProofV3VerificationError) as exc:
            raise _fail(
                f"succinct projection batch verification failed: {exc}"
            )
    _profile_mark("projection-batch")

    # ---- (6) MANDATORY chain: bottom anchor + full connectivity -----------
    if proof.chain is None:
        raise _fail("economic proof requires the chain reveal (chain missing)")
    chain = proof.chain
    first_layer = layer_universe[0]
    residual0_oracle = proof.oracles[chain.residual0_oracle_index]
    if residual0_oracle.oracle_id != f"l{first_layer}.residual_in":
        raise _fail("chain bottom anchor references the wrong oracle")
    bottom_rows = tuple(
        sorted(
            set(challenge.bottom_anchor_rows)
            | (
                set(
                    lean_tokens_by_layer.get(
                        first_layer,
                        challenge.sampled_token_rows,
                    )
                )
                if lean and first_layer in selected_layers
                else set()
            )
        )
    )
    from verallm.proof_v3.lean_execution_anchor import (
        lean_bottom_sequence_positions_v3,
    )

    bottom_positions = lean_bottom_sequence_positions_v3(
        rows=bottom_rows,
        candidate_sequence_positions=(
            challenge.candidate_sequence_positions
        ),
        lean_positions=(
            lean_positions_by_layer[first_layer]
            if lean and first_layer in lean_positions_by_layer
            else {}
        ),
    )
    residual0_rows = _opened_rows(
        oracle=residual0_oracle,
        base_binding=base_binding,
        rows=bottom_rows,
        opening=chain.residual0_opening,
        require_int8=True,
        what="chain residual0",
        expect_mode=(
            2 if lean else (0 if anchor_binding is not None else 2)
        ),
        packed_rows=_anchor_packed_rows(
            anchor_binding=anchor_binding,
            oracle_id=residual0_oracle.oracle_id,
            row_indices=bottom_rows,
        ),
    )
    exact_final_source = None
    if anchor_binding is not None:
        anchor_binding.verify_rows(
            oracle_id=residual0_oracle.oracle_id,
            actual_rows=residual0_rows,
            row_indices=bottom_rows,
        )
    if len(chain.embedding_rows) != len(bottom_positions):
        raise _fail("chain embedding rows do not match the sampled positions")
    for row, position, embed_reveal in zip(
        bottom_rows, bottom_positions, chain.embedding_rows, strict=True
    ):
        expected_token = execution_input_token_id_at_position_v3(
            prompt_token_ids=prompt_token_ids,
            observed_output_token_ids=observed_output_token_ids,
            sequence_position=position,
        )
        if embed_reveal.row_index != expected_token:
            raise _fail(
                "chain embedding row is not the validator-observed execution "
                "input token"
            )
        embed_row = artifacts.verify_weight_row(
            name="embed_tokens", reveal=embed_reveal
        )
        if tuple(embed_row) != residual0_rows[row]:
            raise _fail(
                "bottom anchor: residual[0] != signed embedding of the real "
                "execution input (input substitution)"
            )
    boundary_layers = tuple(
        boundary.layer_index for boundary in chain.boundaries
    )
    expected_boundary_layers = (
        selected_layers
        if lean or anchor_binding is not None
        else layer_universe
    )
    if boundary_layers != expected_boundary_layers:
        raise _fail(
            "chain boundaries do not cover the exact required layer set"
        )
    opened_boundaries: dict[int, tuple[dict, dict]] = {}
    for boundary in chain.boundaries:
        layer = boundary.layer_index
        token_rows = (
            lean_tokens_by_layer[layer]
            if lean
            else tuple(challenge.sampled_token_rows)
        )
        in_oracle = proof.oracles[boundary.in_oracle_index]
        out_oracle = proof.oracles[boundary.out_oracle_index]
        if (
            in_oracle.oracle_id != f"l{layer}.residual_in"
            or out_oracle.oracle_id != f"l{layer}.residual_out"
        ):
            raise _fail("chain boundary references the wrong oracles")
        in_rows = _opened_rows(
            oracle=in_oracle,
            base_binding=base_binding,
            rows=token_rows,
            opening=boundary.in_opening,
            require_int8=True,
            what=f"chain l{layer} residual_in",
            expect_mode=(
                2 if lean else (0 if anchor_binding is not None else 2)
            ),
            packed_rows=_anchor_packed_rows(
                anchor_binding=anchor_binding,
                oracle_id=in_oracle.oracle_id,
                row_indices=token_rows,
            ),
        )
        residual_cells = tuple(
            (token, col)
            for token in token_rows
            for col in challenge.residual_cols_for(
                layer_index=layer,
                hidden_dim=embed_hidden,
            )
        )
        if anchor_binding is None:
            out_rows = _opened_rows(
                oracle=out_oracle,
                base_binding=base_binding,
                rows=token_rows,
                opening=boundary.out_opening,
                require_int8=True,
                what=f"chain l{layer} residual_out",
                expect_mode=2,
            )
        else:
            out_rows = _opened_cells(
                oracle=out_oracle,
                base_binding=base_binding,
                cells=residual_cells,
                opening=boundary.out_opening,
                what=f"chain l{layer} residual_out",
                expect_mode=2,
            )
        if anchor_binding is not None:
            anchor_binding.verify_rows(
                oracle_id=in_oracle.oracle_id,
                actual_rows=in_rows,
                row_indices=token_rows,
            )
            anchor_binding.verify_cells(
                oracle_id=out_oracle.oracle_id,
                actual_cells=out_rows,
                cells=residual_cells,
            )
        opened_boundaries[layer] = (in_rows, out_rows)
    if anchor_binding is None:
        if lean:
            from verallm.proof_v3.lean_execution_anchor import (
                lean_transition_positions_v3,
            )

            lean_required_chain_positions = (
                lean_transition_positions_v3(challenge)
            )
        else:
            lean_required_chain_positions = ()
        for previous_layer, next_layer in zip(
            expected_boundary_layers,
            expected_boundary_layers[1:],
            strict=False,
        ):
            if lean and next_layer != previous_layer + 1:
                continue
            out_scale = oracle_by_id[
                f"l{previous_layer}.residual_out"
            ].scale_bits
            in_scale = oracle_by_id[
                f"l{next_layer}.residual_in"
            ].scale_bits
            if out_scale != in_scale:
                raise _fail(
                    "chaining: adjacent residual boundary scales differ "
                    "(disconnected trace)"
                )
            _in_prev, out_prev = opened_boundaries[previous_layer]
            in_next, _out_next = opened_boundaries[next_layer]
            previous_rows = (
                lean_tokens_by_layer[previous_layer]
                if lean
                else tuple(challenge.sampled_token_rows)
            )
            next_rows = (
                lean_tokens_by_layer[next_layer]
                if lean
                else tuple(challenge.sampled_token_rows)
            )
            if lean:
                from verallm.proof_v3.lean_execution_anchor import (
                    lean_shared_boundary_rows_v3,
                )

                try:
                    shared_rows = lean_shared_boundary_rows_v3(
                        previous_layout=lean_row_layouts[previous_layer],
                        next_layout=lean_row_layouts[next_layer],
                        required_positions=lean_required_chain_positions,
                    )
                except ProofV3Error as exc:
                    raise _fail(f"chaining: {exc}") from exc
            else:
                if previous_rows != next_rows:
                    raise _fail(
                        "chaining: adjacent layers use different row layouts"
                    )
                shared_rows = tuple(
                    (
                        int(challenge.candidate_sequence_positions[row]),
                        row,
                        row,
                    )
                    for row in previous_rows
                )
            for position, previous_row, next_row in shared_rows:
                if out_prev[previous_row] != in_next[next_row]:
                    raise _fail(
                        f"chaining: layer {previous_layer} residual_out != "
                        f"layer {next_layer} residual_in at sequence position "
                        f"{position} (disconnected trace)"
                    )

    if lean:
        from verallm.proof_v3.economic_execution_anchor import (
            quantize_execution_anchor_row_v3,
        )
        from verallm.proof_v3.lean_execution_anchor import (
            LEAN_EXECUTION_CHECKPOINT_STRIDE_V3,
        )

        selected_set = set(selected_layers)
        corridor_starts = tuple(
            layer
            for layer in selected_layers
            if layer - 1 not in selected_set
        )
        for corridor_start in corridor_starts:
            corridor_end = min(
                corridor_start
                + LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
                - 1,
                layer_universe[-1],
            )
            while corridor_end not in selected_set:
                corridor_end -= 1
            start_rows = opened_boundaries[corridor_start][0]
            end_rows = opened_boundaries[corridor_end][1]
            corridor_rows = lean_tokens_by_layer[corridor_start]
            if corridor_rows != lean_tokens_by_layer[corridor_end]:
                raise _fail(
                    "lean corridor endpoints use different row layouts"
                )
            for pool_row in corridor_rows:
                position = lean_positions_by_layer[corridor_start][pool_row]
                if prefix_cached_tokens:
                    # A cache-hit execution and an independent full-prompt
                    # replay do not share residual arithmetic or scheduling.
                    # Bind them only through state that actually crosses the
                    # cache boundary (nonce-selected attention K/V or GDN
                    # boundary state) and the authenticated terminal path.
                    # The replay's registered-weight corridor and internal
                    # residual chain remain mandatory.
                    continue
                if corridor_start > 0:
                    stage_id = (
                        f"l{corridor_start - 1}.residual_out"
                    )
                    try:
                        raw = anchor_rows[stage_id][position]
                    except KeyError as exc:
                        raise _fail(
                            "lean corridor input has no authenticated "
                            "checkpoint row"
                        ) from exc
                    expected = quantize_execution_anchor_row_v3(
                        row_bytes=raw,
                        scale=bits_to_scale_v3(
                            oracle_by_id[
                                f"l{corridor_start}.residual_in"
                            ].scale_bits
                        ),
                        encoding_id=anchor_encoding,
                    )
                    if not _replay_capture_row_matches_v3(
                        start_rows[pool_row],
                        expected,
                    ):
                        raise _fail(
                            "lean corridor input is detached from its "
                            "pre-nonce checkpoint"
                        )
                stage_id = f"l{corridor_end}.residual_out"
                try:
                    raw = anchor_rows[stage_id][position]
                except KeyError as exc:
                    raise _fail(
                        "lean corridor output has no authenticated "
                        "checkpoint row"
                    ) from exc
                expected = quantize_execution_anchor_row_v3(
                    row_bytes=raw,
                    scale=bits_to_scale_v3(
                        oracle_by_id[
                            f"l{corridor_end}.residual_out"
                        ].scale_bits
                    ),
                    encoding_id=anchor_encoding,
                )
                actual = tuple(end_rows[pool_row])
                if not _replay_capture_row_matches_v3(actual, expected):
                    differences = tuple(
                        (column, actual_value, expected_value)
                        for column, (actual_value, expected_value) in enumerate(
                            zip(actual, expected, strict=True)
                        )
                        if actual_value != expected_value
                    )
                    first = differences[0]
                    max_delta = max(
                        abs(actual_value - expected_value)
                        for _column, actual_value, expected_value
                        in differences
                    )
                    raise _fail(
                        "lean corridor output is detached from its "
                        "pre-nonce checkpoint "
                        f"(layer={corridor_end}, position={position}, "
                        f"differences={len(differences)}, "
                        f"first_column={first[0]}, "
                        f"first_actual={first[1]}, "
                        f"first_expected={first[2]}, "
                        f"max_delta={max_delta})"
                    )

    _profile_mark("chain")

    # ---- (6.5) MANDATORY couplings: captured Y <-> surrogates, residual
    # compositions, elementwise MLP and both RMSNorm links -------------------
    selected_full_layers = tuple(
        layer for layer in selected_layers
        if layer_kinds[layer] == "full_attention"
    )
    selected_gdn_layers = tuple(
        layer for layer in selected_layers if layer_kinds[layer] == "gdn"
    )
    attention_plans_by_layer = (
        {}
        if streaming_attention_metadata is None
        else {
            int(plan.layer): plan
            for plan in streaming_attention_metadata["plans"]
        }
    )
    attention_query_rows = {}
    if lean and selected_full_layers:
        section = getattr(proof, "attention", None)
        if section is None:
            raise _fail("lean attention proof has no request section")
        attention_query_rows = {
            (int(layer), int(position)): bytes(row)
            for layer, position, row in section.query_rows
        }
        expected_query_rows = {
            (int(plan.layer), int(position))
            for plan in attention_plans_by_layer.values()
            for position in plan.row_positions
        }
        expected_query_rows.update(
            (int(layer), int(position))
            for layer, _kv_head, position in prefix_cache_projection_heads
        )
        if set(attention_query_rows) != expected_query_rows:
            raise _fail(
                "lean attention query rows do not match the nonce-derived plan"
            )
    elif getattr(getattr(proof, "attention", None), "query_rows", ()):
        raise _fail(
            "non-lean attention proof must not carry post-nonce query rows"
        )
    if (
        tuple(c.layer_index for c in proof.couplings)
        != selected_full_layers
    ):
        raise _fail(
            "attention coupling reveals do not cover exactly the selected "
            "full-attention layers"
        )
    import math

    corridor_stats: list[tuple[str, float, float, str]] = []
    mlp_activation_reveals = {
        int(reveal.layer_index): tuple(reveal.rows)
        for reveal in proof.mlp_activation_reveals
    }
    if exact_mlp_activation:
        if tuple(sorted(mlp_activation_reveals)) != selected_layers:
            raise _fail(
                "lean MLP activation reveals do not cover exactly the "
                "selected layers"
            )
    elif mlp_activation_reveals:
        raise _fail(
            "non-lean proof must not carry MLP activation replay cells"
        )

    def _verify_exact_mlp_activation_v3(
        *,
        layer: int,
        layer_tokens,
        mlp_cols,
        gate_up_values,
        inter_dim: int,
        gate_up_scale: float,
        down_x_rows,
        down_x_scale: float,
        gate_up_surrogate,
        gate_up_x_rows,
        gate_up_weight_rows,
        gate_up_x_scale: float,
        gate_up_w_scale: float,
        gate_up_bias_at,
        gate_up_x_sq,
        gate_up_w_sq,
        stats,
    ) -> None:
        from verallm.proof_v3.economic_execution_anchor import (
            _decode_row_v3,
        )

        token_for_position = {
            int(position): int(token)
            for token, position in lean_positions_by_layer[layer].items()
        }
        expected_positions = tuple(sorted(token_for_position))
        records = mlp_activation_reveals[layer]
        actual_positions = tuple(int(record[0]) for record in records)
        if actual_positions != expected_positions:
            raise _fail(
                f"lean MLP activation rows are incomplete for layer {layer}"
            )
        expected_width = 2 * len(mlp_cols)
        for position, gate_raw, up_raw, down_raw in records:
            if any(
                len(raw) != expected_width
                for raw in (gate_raw, up_raw, down_raw)
            ):
                raise _fail(
                    f"lean MLP activation cell width is malformed for "
                    f"layer {layer}"
                )
            gate_values = _decode_row_v3(gate_raw, anchor_encoding)
            up_values = _decode_row_v3(up_raw, anchor_encoding)
            down_values = _decode_row_v3(down_raw, anchor_encoding)
            token = token_for_position[int(position)]
            if token not in set(layer_tokens):
                raise _fail(
                    f"lean MLP activation row is detached for layer {layer}"
                )
            for slot, col in enumerate(mlp_cols):
                gate = float(gate_values[slot])
                up = float(up_values[slot])
                got = float(down_values[slot])
                expected_gate_i8 = int(gate_up_values[(token, col)])
                expected_up_i8 = int(
                    gate_up_values[(token, inter_dim + col)]
                )
                expected_down_i8 = int(down_x_rows[token][col])
                for output, value, label in (
                    (col, gate, "gate"),
                    (inter_dim + col, up, "up"),
                ):
                    bias_value, bias_quant = gate_up_bias_at(output)
                    _corridor_check(
                        surrogate_value=_projection_surrogate_value(
                            gate_up_surrogate,
                            token,
                            output,
                        ),
                        captured_value=value,
                        x_row=gate_up_x_rows[token],
                        w_row=gate_up_weight_rows.get(output, ()),
                        x_scale=gate_up_x_scale,
                        w_scale=gate_up_w_scale,
                        y_scale=1.0,
                        output_quant_floor=0.0,
                        captured_is_quantized=False,
                        what=(
                            f"coupling l{layer} exact MLP {label} projection"
                        ),
                        bias_value=bias_value,
                        bias_quant=bias_quant,
                        stats=stats,
                        kind="y_gate_up",
                        x_sq=gate_up_x_sq[token],
                        w_sq=gate_up_w_sq[output],
                        sigma_cap=corridor_sigma,
                    )
                try:
                    actual_i8 = _verify_exact_runtime_mlp_cell_v3(
                        gate=gate,
                        up=up,
                        got=got,
                        expected_gate_i8=expected_gate_i8,
                        expected_up_i8=expected_up_i8,
                        expected_down_i8=expected_down_i8,
                        gate_up_scale=gate_up_scale,
                        down_x_scale=down_x_scale,
                        encoding_id=anchor_encoding,
                    )
                except ProofV3VerificationError as exc:
                    raise _fail(
                        f"{exc} (layer={layer}, position={position}, "
                        f"column={col})"
                    ) from exc
                if _MLP_ACTIVATION_QUALIFICATION_REPORT is not None:
                    _MLP_ACTIVATION_QUALIFICATION_REPORT.append(
                        {
                            "layer": int(layer),
                            "position": int(position),
                            "column": int(col),
                            "gate_i8": int(actual_i8[0]),
                            "up_i8": int(actual_i8[1]),
                            "down_i8": int(actual_i8[2]),
                            "gate_near_zero": abs(int(actual_i8[0])) <= 1,
                            "quantized_rail": any(
                                value in (-128, 127) for value in actual_i8
                            ),
                        }
                    )

    for coupling in proof.couplings:
        layer = coupling.layer_index
        layer_tokens = projection_tokens_by_layer[layer]
        oracle_ids = {
            "attn_o_y": (coupling.attn_o_y_oracle_index, f"l{layer}.attn_o_y"),
            "down_y": (coupling.down_y_oracle_index, f"l{layer}.down_y"),
            "mid": (coupling.mid_oracle_index, f"l{layer}.mid_residual"),
            "gate_up_y": (
                coupling.gate_up_y_oracle_index,
                f"l{layer}.gate_up_y",
            ),
            "k": (coupling.k_oracle_index, f"l{layer}.k_cache"),
            "v": (coupling.v_oracle_index, f"l{layer}.v_cache"),
        }
        oracles = {}
        for key, (index, expected_id) in oracle_ids.items():
            oracle = proof.oracles[index]
            if oracle.oracle_id != expected_id:
                raise _fail("coupling reveal references the wrong oracles")
            oracles[key] = oracle

        # manifest-driven projection biases: WHICH projections carry a bias
        # is derived from the signed manifest alone -- a manifest bias entry
        # makes its reveal mandatory, and a reveal without a manifest entry
        # is rejected.  Model-agnostic: bias-free families register none.
        revealed_bias = dict(coupling.bias_rows)
        biases: dict[str, tuple[tuple[int, ...], float]] = {}
        for proj_index, (_x, _s, manifest_suffix) in enumerate(
            audited_projections_for_layer_kind_v3("full_attention")
        ):
            bias_name = f"l{layer}.{manifest_suffix}_bias"
            if not artifacts.has_entry(bias_name):
                continue
            reveal = revealed_bias.pop(proj_index, None)
            if reveal is None:
                raise _fail(
                    f"coupling l{layer} is missing the manifest-required "
                    f"{manifest_suffix} bias reveal"
                )
            biases[manifest_suffix] = (
                artifacts.verify_weight_row(name=bias_name, reveal=reveal),
                artifacts.scale_for(bias_name),
            )
        if revealed_bias:
            raise _fail(
                f"coupling l{layer} reveals a bias with no manifest entry"
            )

        def _bias_at(suffix: str, out_col: int) -> tuple[float, float]:
            entry = biases.get(suffix)
            if entry is None:
                return 0.0, 0.0
            values, scale = entry
            if out_col >= len(values):
                raise _fail(
                    f"coupling l{layer} {suffix} bias row is narrower than "
                    "the audited output column"
                )
            return values[out_col] * scale, 0.5 * scale

        residual_cols = challenge.residual_cols_for(
            layer_index=layer,
            hidden_dim=embed_hidden,
        )
        output_cells = {}
        for key, suffix in (("attn_o_y", "o"), ("down_y", "down")):
            outs = opened_projections[(layer, suffix)][5]
            cells = tuple(
                (token, col)
                for token in layer_tokens
                for col in sorted(set(outs) | set(residual_cols))
            )
            output_cells[key] = _opened_cells(
                oracle=oracles[key],
                base_binding=base_binding,
                cells=cells,
                opening=(
                    coupling.attn_o_y_opening
                    if key == "attn_o_y"
                    else coupling.down_y_opening
                ),
                what=f"coupling l{layer} {key}",
                expect_mode=2,
            )
            if anchor_binding is not None:
                anchor_binding.verify_cells(
                    oracle_id=oracles[key].oracle_id,
                    actual_cells=output_cells[key],
                    cells=cells,
                )
        attn_o_y_rows = output_cells["attn_o_y"]
        down_y_rows = output_cells["down_y"]
        mid_rows = _opened_rows(
            oracle=oracles["mid"],
            base_binding=base_binding,
            rows=layer_tokens,
            opening=coupling.mid_opening,
            require_int8=True,
            what=f"coupling l{layer} mid_residual",
            expect_mode=2,
        )
        if anchor_binding is not None:
            anchor_binding.verify_rows(
                oracle_id=oracles["mid"].oracle_id,
                actual_rows=mid_rows,
                row_indices=layer_tokens,
            )

        # --- (a) Y corridors: attn_o and down at the audited out cells ---
        for suffix, y_rows in (
            ("o", attn_o_y_rows),
            ("down", down_y_rows),
        ):
            x_rows, surrogate, weight_rows, x_scale, w_scale, outs = (
                opened_projections[(layer, suffix)]
            )
            y_scale = bits_to_scale_v3(
                oracles["attn_o_y" if suffix == "o" else "down_y"].scale_bits
            )
            x_sq_by_token = {
                token: sum(a * a for a in x_rows[token])
                for token in layer_tokens
            }
            w_sq_by_out = {
                out: _projection_weight_row_sq(
                    layer=layer,
                    suffix=suffix,
                    out=out,
                    weight_rows=weight_rows,
                )
                for out in outs
            }
            for token in layer_tokens:
                for out in outs:
                    bias_value, bias_quant = _bias_at(suffix, out)
                    _corridor_check(
                        surrogate_value=_projection_surrogate_value(
                            surrogate, token, out
                        ),
                        captured_value=y_rows[(token, out)],
                        x_row=x_rows[token],
                        w_row=weight_rows.get(out, ()),
                        x_scale=x_scale,
                        w_scale=w_scale,
                        y_scale=y_scale,
                        what=f"coupling l{layer}.{suffix} Y corridor",
                        bias_value=bias_value,
                        bias_quant=bias_quant,
                        stats=corridor_stats,
                        kind=f"y_{suffix}",
                        x_sq=x_sq_by_token[token],
                        w_sq=w_sq_by_out[out],
                        sigma_cap=corridor_sigma,
                    )

        # --- gate_up corridor + elementwise MLP cells ---
        (
            gu_x_rows,
            gu_surrogate,
            gu_weight_rows,
            gu_x_scale,
            gu_w_scale,
            gu_outs,
        ) = opened_projections[(layer, "gate_up")]
        gate_up_oracle = oracles["gate_up_y"]
        gu_y_scale = bits_to_scale_v3(gate_up_oracle.scale_bits)
        inter_dim = gate_up_oracle.col_count // 2
        mlp_cols = challenge.mlp_cols_for(
            layer_index=layer, inter_dim=inter_dim
        )
        gate_up_cells = set()
        for token in layer_tokens:
            for out in gu_outs:
                gate_up_cells.add((token, out))
            for col in mlp_cols:
                gate_up_cells.add((token, col))
                gate_up_cells.add((token, inter_dim + col))
        gate_up_values = _opened_cells(
            oracle=gate_up_oracle,
            base_binding=base_binding,
            cells=tuple(gate_up_cells),
            opening=coupling.gate_up_y_opening,
            what=f"coupling l{layer} gate_up_y",
            expect_mode=2,
        )
        if anchor_binding is not None:
            anchor_binding.verify_cells(
                oracle_id=gate_up_oracle.oracle_id,
                actual_cells=gate_up_values,
                cells=tuple(gate_up_cells),
            )
        gu_x_sq = {
            token: sum(a * a for a in gu_x_rows[token])
            for token in layer_tokens
        }
        gu_w_sq = {
            out: _projection_weight_row_sq(
                layer=layer,
                suffix="gate_up",
                out=out,
                weight_rows=gu_weight_rows,
            )
            for out in gu_outs
        }
        for token in layer_tokens:
            for out in gu_outs:
                bias_value, bias_quant = _bias_at("gate_up", out)
                _corridor_check(
                    surrogate_value=_projection_surrogate_value(
                        gu_surrogate, token, out
                    ),
                    captured_value=gate_up_values[(token, out)],
                    x_row=gu_x_rows[token],
                    w_row=gu_weight_rows.get(out, ()),
                    x_scale=gu_x_scale,
                    w_scale=gu_w_scale,
                    y_scale=gu_y_scale,
                    what=f"coupling l{layer}.gate_up Y corridor",
                    bias_value=bias_value,
                    bias_quant=bias_quant,
                    stats=corridor_stats,
                    kind="y_gate_up",
                    x_sq=gu_x_sq[token],
                    w_sq=gu_w_sq[out],
                    sigma_cap=corridor_sigma,
                )
        down_x_rows, _s, _w, down_x_scale, _ws, _o = opened_projections[
            (layer, "down")
        ]
        if exact_mlp_activation:
            _verify_exact_mlp_activation_v3(
                layer=layer,
                layer_tokens=layer_tokens,
                mlp_cols=mlp_cols,
                gate_up_values=gate_up_values,
                inter_dim=inter_dim,
                gate_up_scale=gu_y_scale,
                down_x_rows=down_x_rows,
                down_x_scale=down_x_scale,
                gate_up_surrogate=gu_surrogate,
                gate_up_x_rows=gu_x_rows,
                gate_up_weight_rows=gu_weight_rows,
                gate_up_x_scale=gu_x_scale,
                gate_up_w_scale=gu_w_scale,
                gate_up_bias_at=lambda output: _bias_at("gate_up", output),
                gate_up_x_sq=gu_x_sq,
                gate_up_w_sq=gu_w_sq,
                stats=corridor_stats,
            )
        else:
            for token in layer_tokens:
                for col in mlp_cols:
                    gate_i8 = gate_up_values[(token, col)]
                    up_i8 = gate_up_values[(token, inter_dim + col)]
                    down_i8 = down_x_rows[token][col]
                    gate = gate_i8 * gu_y_scale
                    up = up_i8 * gu_y_scale
                    predicted = _silu(gate) * up
                    got = down_i8 * down_x_scale
                    quant = 0.5 * down_x_scale + 0.5 * gu_y_scale * (
                        1.1 * abs(up)
                        + abs(_silu(gate))
                        + 0.5 * gu_y_scale
                    )
                    if _swiglu_output_is_forced_to_quantization_rail_v3(
                        gate_i8=gate_i8,
                        up_i8=up_i8,
                        gate_up_scale=gu_y_scale,
                        output_i8=down_i8,
                        output_scale=down_x_scale,
                    ):
                        predicted = got
                    _fixed_quantization_corridor_check(
                        delta=abs(got - predicted),
                        quant=quant,
                        relative=_REL_COEFF * abs(predicted),
                        what=f"coupling l{layer} elementwise MLP link",
                        kind="mlp_elementwise",
                        failure=(
                            f"coupling l{layer} elementwise MLP link is outside "
                            "the quantization corridor (fabricated MLP trace)"
                        ),
                    )

        # --- (b) qkv <-> K/V cache corridor at sampled kv columns ---
        (
            qkv_x_rows,
            _qkv_surrogate,
            _qkv_weight_rows,
            qkv_x_scale,
            qkv_w_scale,
            _qkv_outs,
        ) = opened_projections[(layer, "qkv")]
        kv_dim = oracles["k"].col_count
        qkv_s_oracle = oracle_by_id[f"l{layer}.qkv_s"]
        q_width = qkv_s_oracle.col_count - 2 * kv_dim
        kv_cols = challenge.kv_cols_for(layer_index=layer, kv_dim=kv_dim)
        prefix_bridge_heads = tuple(
            (int(kv_head), int(position))
            for bridge_layer, kv_head, position
            in prefix_cache_projection_heads
            if int(bridge_layer) == int(layer)
        )
        prefix_bridge_cols = tuple(
            sorted(
                column
                for kv_head, _position in prefix_bridge_heads
                for column in range(
                    kv_head * int(streaming_attention_metadata["hd"]),
                    (kv_head + 1)
                    * int(streaming_attention_metadata["hd"]),
                )
            )
        )
        kv_corridor_cols = tuple(
            sorted(set(kv_cols) | set(prefix_bridge_cols))
        )
        query_output_cols = ()
        plan = attention_plans_by_layer.get(layer)
        if lean and plan is not None:
            from verallm.proof_v3.attention_runtime_semantics import (
                Q_GATE_INTERLEAVED_LAYOUT_V3,
            )

            head_dim = int(streaming_attention_metadata["hd"])
            gated_q = (
                streaming_attention_metadata["semantics"].qkv_layout_id
                == Q_GATE_INTERLEAVED_LAYOUT_V3
            )
            head_stride = head_dim * (2 if gated_q else 1)
            query_output_cols = tuple(
                sorted(
                    column
                    for head in plan.heads
                    for column in range(
                        int(head) * head_stride,
                        (int(head) + 1) * head_stride,
                    )
                )
            )
        kv_global_outs = tuple(sorted(
            {q_width + col for col in kv_corridor_cols}
            | {q_width + kv_dim + col for col in kv_corridor_cols}
            | set(query_output_cols)
        ))
        expected_kv_rows = tuple(
            row.row_index for row in coupling.qkv_kv_weight_rows
        )
        required_kv_rows = () if lean else kv_global_outs
        if expected_kv_rows != required_kv_rows:
            raise _fail(
                "coupling kv weight rows are not the validator-derived "
                "selection"
            )
        kv_weight_rows = {
            row.row_index: artifacts.verify_weight_row(
                name=f"l{layer}.qkv", reveal=row
            )
            for row in coupling.qkv_kv_weight_rows
        }
        qkv_s_index = None
        for index, oracle in enumerate(proof.oracles):
            if oracle.oracle_id == f"l{layer}.qkv_s":
                qkv_s_index = index
                break
        surrogate_kv = _opened_cells(
            oracle=proof.oracles[qkv_s_index],
            base_binding=base_binding,
            cells=tuple(
                (token, out)
                for token in layer_tokens
                for out in kv_global_outs
            ),
            opening=coupling.qkv_s_kv_opening,
            what=f"coupling l{layer} qkv_s kv cells",
            expect_mode=3,
            expect_bounded_width=bounded_byte_width_v3(
                artifacts.entry(f"l{layer}.qkv").in_dim),
        )
        kv_cells = tuple(
            (token, col)
            for token in layer_tokens
            for col in kv_corridor_cols
        )
        k_values = _opened_cells(
            oracle=oracles["k"],
            base_binding=base_binding,
            cells=kv_cells,
            opening=coupling.k_opening,
            what=f"coupling l{layer} k_cache",
            expect_mode=1,
        )
        v_values = _opened_cells(
            oracle=oracles["v"],
            base_binding=base_binding,
            cells=kv_cells,
            opening=coupling.v_opening,
            what=f"coupling l{layer} v_cache",
            expect_mode=1,
        )
        if anchor_binding is not None:
            anchor_binding.verify_cells(
                oracle_id=oracles["k"].oracle_id,
                actual_cells=k_values,
                cells=kv_cells,
            )
            anchor_binding.verify_cells(
                oracle_id=oracles["v"].oracle_id,
                actual_cells=v_values,
                cells=kv_cells,
            )
        k_scale = bits_to_scale_v3(oracles["k"].scale_bits)
        v_scale = bits_to_scale_v3(oracles["v"].scale_bits)
        if lean:
            from verallm.proof_v3.economic_execution_anchor import (
                quantize_execution_anchor_row_v3,
            )

            stage_id = f"l{layer}.attention_kv_output"
            cache_relation = profile.relation_spec.cache
            authenticated_positions = set(
                kv_positions_by_layer.get(layer, ())
            )
            for token in layer_tokens:
                position = lean_positions_by_layer[layer][token]
                if position not in authenticated_positions:
                    continue
                if prefix_cached_tokens and position < prefix_cached_tokens:
                    from verallm.proof_v3.execution_anchor import (
                        execution_anchor_lane_bytes_v3,
                    )

                    block_index, row_index = divmod(
                        position,
                        proof.prefix_cache.commitment.block_token_count,
                    )
                    # Paged-cache V is the projection output and can be
                    # compared directly.  K is normalized and RoPE-rotated
                    # before entering vLLM's cache; its complete-head bridge
                    # is verified below after the registered QKV corridor.
                    for tag, expected_by_cell, scale in (
                        ("v", v_values, v_scale),
                    ):
                        cache_stage = f"l{layer}.attention_{tag}_cache"
                        lane_bytes = execution_anchor_lane_bytes_v3(
                            cache_stage
                        )
                        for col in kv_cols:
                            byte_offset = int(col) * 2
                            lane_index, in_lane = divmod(
                                byte_offset, lane_bytes
                            )
                            try:
                                lane = prefix_cache_lane_map[
                                    (
                                        block_index,
                                        cache_stage,
                                        row_index,
                                        lane_index,
                                    )
                                ]
                            except KeyError as exc:
                                raise _fail(
                                    "lean K/V projection row lacks its "
                                    "nonce-selected prefix-cache lane"
                                ) from exc
                            raw_cell = lane[in_lane:in_lane + 2]
                            if len(raw_cell) != 2:
                                raise _fail(
                                    "lean prefix-cache K/V cell is truncated"
                                )
                            quantized = quantize_execution_anchor_row_v3(
                                row_bytes=raw_cell,
                                scale=scale,
                                encoding_id=anchor_encoding,
                            )
                            if len(quantized) != 1 or (
                                _CORRIDOR_REPORT is None
                                and not _replay_capture_cell_matches_v3(
                                    expected_by_cell[(token, col)],
                                    quantized[0],
                                    max_lsb_delta=(
                                        cache_relation.prefix_cache_v_cell_delta_max
                                    ),
                                )
                            ):
                                raise _fail(
                                    "lean K/V projection is detached from "
                                    "the nonce-selected prefix-cache state "
                                    f"(layer={layer}, token={token}, "
                                    f"position={position}, tag={tag}, "
                                    f"column={col}, "
                                    f"actual={expected_by_cell[(token, col)]}, "
                                    f"expected={quantized[0]})"
                                )
                    continue
                try:
                    anchor_position = (
                        position - prefix_cached_tokens
                        if prefix_cached_tokens
                        else position
                    )
                    raw_kv = anchor_rows[stage_id][anchor_position]
                except KeyError as exc:
                    raise _fail(
                        "lean K/V projection row has no authenticated "
                        "pre-nonce cache row"
                    ) from exc
                k_expected = quantize_execution_anchor_row_v3(
                    row_bytes=raw_kv,
                    scale=k_scale,
                    encoding_id=anchor_encoding,
                )[:kv_dim]
                v_expected = quantize_execution_anchor_row_v3(
                    row_bytes=raw_kv,
                    scale=v_scale,
                    encoding_id=anchor_encoding,
                )[kv_dim:]
                if (
                    len(k_expected) != kv_dim
                    or len(v_expected) != kv_dim
                ):
                    raise _fail(
                        "lean authenticated K/V row has the wrong width"
                    )
                for col in kv_cols:
                    if (
                        not _replay_capture_cell_matches_v3(
                            k_values[(token, col)],
                            k_expected[col],
                        )
                        or not _replay_capture_cell_matches_v3(
                            v_values[(token, col)],
                            v_expected[col],
                        )
                    ):
                        raise _fail(
                            "lean K/V oracle is detached from the pre-nonce "
                            "runtime cache commitment "
                            f"(layer={layer}, token={token}, "
                            f"position={position}, column={col}, "
                            f"k_actual={k_values[(token, col)]}, "
                            f"k_expected={k_expected[col]}, "
                            f"v_actual={v_values[(token, col)]}, "
                            f"v_expected={v_expected[col]})"
                        )
        qkv_x_sq = {
            token: sum(a * a for a in qkv_x_rows[token])
            for token in layer_tokens
        }
        kv_w_sq = (
            {
                out: artifacts.weight_row_sq(f"l{layer}.qkv", out)
                for out in kv_global_outs
            }
            if lean
            else {
                out: sum(w * w for w in row)
                for out, row in kv_weight_rows.items()
            }
        )
        # Verify the exact K/V projection cells in one C-level int64 GEMM.
        # The former nested Python sum performed ~2M scalar lookups for the
        # release shape and was one of the largest non-crypto verifier costs.
        kv_recomputed = None
        if not lean:
            try:
                import numpy as _np

                _tokens = tuple(layer_tokens)
                _outs = tuple(kv_global_outs)
                _x_mat = _np.asarray(
                    [qkv_x_rows[token] for token in _tokens],
                    dtype=_np.int64,
                )
                _w_mat = _np.asarray(
                    [kv_weight_rows[out] for out in _outs],
                    dtype=_np.int64,
                )
                if (
                    _x_mat.ndim == 2
                    and _w_mat.ndim == 2
                    and _x_mat.shape[1] == _w_mat.shape[1]
                ):
                    _product = _x_mat @ _w_mat.T
                    kv_recomputed = {
                        (token, out): int(
                            _product[token_slot, out_slot]
                        )
                        for token_slot, token in enumerate(_tokens)
                        for out_slot, out in enumerate(_outs)
                    }
            except (ValueError, TypeError):
                kv_recomputed = None
        for token in layer_tokens:
            for col in kv_corridor_cols:
                k_out = q_width + col
                v_out = q_width + kv_dim + col
                # exact recompute of the two surrogate cells first
                for out in (k_out, v_out):
                    if not lean:
                        recomputed = (
                            kv_recomputed[(token, out)]
                            if kv_recomputed is not None
                            else sum(
                                a * b
                                for a, b in zip(
                                    qkv_x_rows[token],
                                    kv_weight_rows[out],
                                    strict=True,
                                )
                            )
                        )
                        if recomputed != surrogate_kv[(token, out)]:
                            raise _fail(
                                f"coupling l{layer} kv surrogate cell != "
                                "exact int8 recompute (substitution)"
                            )
                k_bias, k_bias_quant = _bias_at("qkv", k_out)
                _corridor_check(
                    surrogate_value=surrogate_kv[(token, k_out)],
                    captured_value=k_values[(token, col)],
                    x_row=qkv_x_rows[token],
                    w_row=kv_weight_rows.get(k_out, ()),
                    x_scale=qkv_x_scale,
                    w_scale=qkv_w_scale,
                    y_scale=k_scale,
                    what=f"coupling l{layer} K-cache corridor",
                    bias_value=k_bias,
                    bias_quant=k_bias_quant,
                    stats=corridor_stats,
                    kind="k_cache",
                    x_sq=qkv_x_sq[token],
                    w_sq=kv_w_sq[k_out],
                    sigma_cap=corridor_sigma,
                )
                v_bias, v_bias_quant = _bias_at("qkv", v_out)
                _corridor_check(
                    surrogate_value=surrogate_kv[(token, v_out)],
                    captured_value=v_values[(token, col)],
                    x_row=qkv_x_rows[token],
                    w_row=kv_weight_rows.get(v_out, ()),
                    x_scale=qkv_x_scale,
                    w_scale=qkv_w_scale,
                    y_scale=v_scale,
                    what=f"coupling l{layer} V-cache corridor",
                    bias_value=v_bias,
                    bias_quant=v_bias_quant,
                    stats=corridor_stats,
                    kind="v_cache",
                    x_sq=qkv_x_sq[token],
                    w_sq=kv_w_sq[v_out],
                    sigma_cap=corridor_sigma,
                )

        if prefix_bridge_heads:
            from verallm.proof_v3.attention_anchor_binding import (
                AttentionAnchorGeometryV3,
                decode_runtime_values_v3,
                required_execution_anchor_lanes_v3,
                runtime_kv_head_quantized_v3,
                runtime_paged_cache_kv_head_quantized_v3,
            )
            from verallm.proof_v3.execution_anchor import (
                execution_anchor_lane_bytes_v3,
            )

            hd = int(streaming_attention_metadata["hd"])
            n_kv = int(streaming_attention_metadata["n_kv"])
            semantics = streaming_attention_metadata["semantics"]
            params_by_head = tuple(
                params
                for params, _bounds in streaming_attention_metadata[
                    "calibration"
                ].heads_for(layer)
            )
            geometry = AttentionAnchorGeometryV3(
                query_heads=int(streaming_attention_metadata["nh"]),
                kv_heads=n_kv,
                head_dim=hd,
                qkv_width=2 * kv_dim,
                q_block_width=0,
                k_block_offset=0,
                v_block_offset=kv_dim,
                gated=False,
            )
            token_for_position = {
                int(position): int(token)
                for token, position in lean_positions_by_layer[layer].items()
            }
            relative_step = (
                2.0 ** -7
                if anchor_encoding == "bf16.v1"
                else 2.0 ** -10
            )

            def _target_cache_head(tag: str, kv_head: int, position: int):
                stage_id = f"l{layer}.attention_{tag}_cache"
                lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
                byte_start = int(kv_head) * hd * 2
                lanes = required_execution_anchor_lanes_v3(
                    byte_start=byte_start,
                    byte_length=hd * 2,
                    lane_bytes=lane_bytes,
                )
                block, row = divmod(
                    int(position),
                    proof.prefix_cache.commitment.block_token_count,
                )
                try:
                    joined = b"".join(
                        prefix_cache_lane_map[
                            (block, stage_id, row, lane)
                        ]
                        for lane in lanes
                    )
                except KeyError as exc:
                    raise _fail(
                        "prefix-cache QKV bridge lacks a nonce-selected "
                        "complete-head lane"
                    ) from exc
                offset = byte_start - lanes[0] * lane_bytes
                result = joined[offset:offset + hd * 2]
                if len(result) != hd * 2:
                    raise _fail(
                        "prefix-cache QKV bridge head is truncated"
                    )
                return result

            for kv_head, position in prefix_bridge_heads:
                try:
                    token = token_for_position[int(position)]
                    raw_qkv = attention_query_rows[(layer, int(position))]
                except KeyError as exc:
                    raise _fail(
                        "prefix-cache QKV bridge lacks its nonce-selected "
                        "replay row"
                    ) from exc
                raw_values = decode_runtime_values_v3(
                    raw_qkv, anchor_encoding
                )
                if len(raw_values) != qkv_s_oracle.col_count:
                    raise _fail(
                        "prefix-cache QKV bridge replay row has the wrong "
                        "signed width"
                    )
                local_start = int(kv_head) * hd
                k_start = q_width + local_start
                v_start = q_width + kv_dim + local_start
                k_raw = raw_qkv[k_start * 2:(k_start + hd) * 2]
                v_raw = raw_qkv[v_start * 2:(v_start + hd) * 2]
                for tag, start in (("k", k_start), ("v", v_start)):
                    for coordinate in range(hd):
                        out = start + coordinate
                        raw_value = float(raw_values[out])
                        bias_value, bias_quant = _bias_at("qkv", out)
                        _corridor_check(
                            surrogate_value=surrogate_kv[(token, out)],
                            captured_value=raw_value,
                            x_row=qkv_x_rows[token],
                            w_row=kv_weight_rows.get(out, ()),
                            x_scale=qkv_x_scale,
                            w_scale=qkv_w_scale,
                            y_scale=1.0,
                            what=(
                                f"coupling l{layer} prefix-cache {tag.upper()} "
                                "runtime row"
                            ),
                            bias_value=bias_value,
                            bias_quant=bias_quant,
                            stats=corridor_stats,
                            kind=f"prefix_{tag}_runtime",
                            x_sq=qkv_x_sq[token],
                            w_sq=kv_w_sq[out],
                            sigma_cap=corridor_sigma,
                            output_quant_floor=max(
                                abs(raw_value) * relative_step,
                                2.0 ** -24,
                            ),
                            captured_is_quantized=False,
                        )
                replay_k = runtime_kv_head_quantized_v3(
                    tag="k",
                    raw_head_bytes=k_raw,
                    layer=layer,
                    position=position,
                    kv_head=kv_head,
                    geometry=geometry,
                    semantics=semantics,
                    params_by_head=params_by_head,
                    encoding_id=anchor_encoding,
                )
                target_k = runtime_paged_cache_kv_head_quantized_v3(
                    tag="k",
                    raw_head_bytes=_target_cache_head(
                        "k", kv_head, position
                    ),
                    kv_head=kv_head,
                    geometry=geometry,
                    params_by_head=params_by_head,
                    encoding_id=anchor_encoding,
                )
                replay_v = runtime_kv_head_quantized_v3(
                    tag="v",
                    raw_head_bytes=v_raw,
                    layer=layer,
                    position=position,
                    kv_head=kv_head,
                    geometry=geometry,
                    semantics=semantics,
                    params_by_head=params_by_head,
                    encoding_id=anchor_encoding,
                )
                target_v = runtime_paged_cache_kv_head_quantized_v3(
                    tag="v",
                    raw_head_bytes=_target_cache_head(
                        "v", kv_head, position
                    ),
                    kv_head=kv_head,
                    geometry=geometry,
                    params_by_head=params_by_head,
                    encoding_id=anchor_encoding,
                )
                cache_relation = profile.relation_spec.cache
                if not _prefix_cache_replay_row_matches_v3(
                    replay_k,
                    target_k,
                    max_lsb_delta=cache_relation.prefix_cache_k_cell_delta_max,
                    max_row_sq_delta=(
                        cache_relation.prefix_cache_k_row_sq_delta_max
                    ),
                    tag="k",
                    layer=layer,
                    kv_head=kv_head,
                    position=position,
                ):
                    raise _fail(
                        "prefix-cache K state is detached from the "
                        "registered QKV projection and signed QK/RoPE "
                        "semantics"
                    )
                if not _prefix_cache_replay_row_matches_v3(
                    replay_v,
                    target_v,
                    max_lsb_delta=cache_relation.prefix_cache_v_cell_delta_max,
                    max_row_sq_delta=(
                        cache_relation.prefix_cache_v_row_sq_delta_max
                    ),
                    tag="v",
                    layer=layer,
                    kv_head=kv_head,
                    position=position,
                ):
                    raise _fail(
                        "prefix-cache V state is detached from the "
                        "registered QKV projection"
                    )

        if lean and query_output_cols:
            from verallm.proof_v3.attention_anchor_binding import (
                decode_runtime_values_v3,
            )

            token_for_position = {
                position: token
                for token, position in lean_positions_by_layer[layer].items()
            }
            relative_step = (
                2.0 ** -7
                if anchor_encoding == "bf16.v1"
                else 2.0 ** -10
            )
            for position in plan.row_positions:
                try:
                    token = token_for_position[int(position)]
                    raw_values = decode_runtime_values_v3(
                        attention_query_rows[(layer, int(position))],
                        anchor_encoding,
                    )
                except KeyError as exc:
                    raise _fail(
                        "lean attention query row is outside its proven "
                        "projection corridor"
                    ) from exc
                if len(raw_values) != qkv_s_oracle.col_count:
                    raise _fail(
                        "lean attention query row has the wrong signed width"
                    )
                for out in query_output_cols:
                    bias_value, bias_quant = _bias_at("qkv", out)
                    raw_value = float(raw_values[out])
                    _corridor_check(
                        surrogate_value=surrogate_kv[(token, out)],
                        captured_value=raw_value,
                        x_row=qkv_x_rows[token],
                        w_row=kv_weight_rows.get(out, ()),
                        x_scale=qkv_x_scale,
                        w_scale=qkv_w_scale,
                        y_scale=1.0,
                        what=(
                            f"coupling l{layer} attention query/gate "
                            "runtime row"
                        ),
                        bias_value=bias_value,
                        bias_quant=bias_quant,
                        stats=corridor_stats,
                        kind="q_gate_runtime",
                        x_sq=qkv_x_sq[token],
                        w_sq=kv_w_sq[out],
                        sigma_cap=corridor_sigma,
                        output_quant_floor=max(
                            abs(raw_value) * relative_step,
                            2.0 ** -24,
                        ),
                        captured_is_quantized=False,
                    )

        # --- (c) residual compositions over ALL hidden columns ---
        rin_rows, rout_rows = opened_boundaries[layer]
        rin_scale = bits_to_scale_v3(
            oracle_by_id[f"l{layer}.residual_in"].scale_bits
        )
        rout_scale = bits_to_scale_v3(
            oracle_by_id[f"l{layer}.residual_out"].scale_bits
        )
        mid_scale = bits_to_scale_v3(oracles["mid"].scale_bits)
        o_y_scale = bits_to_scale_v3(oracles["attn_o_y"].scale_bits)
        d_y_scale = bits_to_scale_v3(oracles["down_y"].scale_bits)
        for token in layer_tokens:
            for col in residual_cols:
                mid_i8 = mid_rows[token][col]
                rin_i8 = rin_rows[token][col]
                attn_o_i8 = attn_o_y_rows[(token, col)]
                mid_value = mid_i8 * mid_scale
                composed = (
                    rin_i8 * rin_scale
                    + attn_o_i8 * o_y_scale
                )
                quant = 0.5 * (mid_scale + rin_scale + o_y_scale)
                attention_failure = (
                    f"coupling l{layer} attention residual composition "
                    "broken: residual_in + o_proj output != "
                    "mid-residual (fabricated attention trace)"
                )
                if any(
                    value in (-128, 127)
                    for value in (mid_i8, rin_i8, attn_o_i8)
                ):
                    _quantized_sum_corridor_check(
                        output_i8=mid_i8,
                        output_scale=mid_scale,
                        left_i8=rin_i8,
                        left_scale=rin_scale,
                        right_i8=attn_o_i8,
                        right_scale=o_y_scale,
                        what=f"coupling l{layer} attention residual composition",
                        kind="attention_residual",
                        failure=attention_failure,
                    )
                else:
                    _fixed_quantization_corridor_check(
                        delta=abs(mid_value - composed),
                        quant=quant,
                        relative=_REL_COEFF * max(
                            abs(mid_value),
                            abs(composed),
                        ),
                        what=f"coupling l{layer} attention residual composition",
                        kind="attention_residual",
                        failure=attention_failure,
                    )
                out_quantized = (
                    rout_rows[(token, col)]
                    if anchor_binding is not None
                    else rout_rows[token][col]
                )
                down_i8 = down_y_rows[(token, col)]
                out_value = out_quantized * rout_scale
                composed_out = (
                    mid_i8 * mid_scale
                    + down_i8 * d_y_scale
                )
                quant_out = 0.5 * (rout_scale + mid_scale + d_y_scale)
                mlp_failure = (
                    f"coupling l{layer} MLP residual composition broken: "
                    "mid-residual + down output != residual_out "
                    "(fabricated MLP trace)"
                )
                if any(
                    value in (-128, 127)
                    for value in (out_quantized, mid_i8, down_i8)
                ):
                    _quantized_sum_corridor_check(
                        output_i8=out_quantized,
                        output_scale=rout_scale,
                        left_i8=mid_i8,
                        left_scale=mid_scale,
                        right_i8=down_i8,
                        right_scale=d_y_scale,
                        what=f"coupling l{layer} MLP residual composition",
                        kind="mlp_residual",
                        failure=mlp_failure,
                    )
                else:
                    _fixed_quantization_corridor_check(
                        delta=abs(out_value - composed_out),
                        quant=quant_out,
                        relative=_REL_COEFF * max(
                            abs(out_value),
                            abs(composed_out),
                        ),
                        what=f"coupling l{layer} MLP residual composition",
                        kind="mlp_residual",
                        failure=mlp_failure,
                    )

        # --- (d) both RMSNorm links at sampled columns ---
        input_norm_row = artifacts.verify_weight_row(
            name=f"l{layer}.input_norm", reveal=coupling.input_norm_row
        )
        post_norm_row = artifacts.verify_weight_row(
            name=f"l{layer}.post_norm", reveal=coupling.post_norm_row
        )
        input_norm_scale = artifacts.scale_for(f"l{layer}.input_norm")
        post_norm_scale = artifacts.scale_for(f"l{layer}.post_norm")
        exact_input_norm_sources = {}
        if lean and layer != layer_universe[0]:
            # The compact proof already authenticates the exact previous-layer
            # residual row.  Use it for the input RMSNorm denominator instead
            # of deriving that denominator from a possibly clipped int8 rail.
            from verallm.proof_v3.economic_execution_anchor import (
                _decode_row_v3,
            )

            layer_slot = layer_universe.index(layer)
            previous_layer = layer_universe[layer_slot - 1]
            input_stage_id = f"l{previous_layer}.residual_out"
            for token in layer_tokens:
                position = lean_positions_by_layer[layer][token]
                anchor_position = (
                    _execution_anchor_row_for_absolute_position_v3(
                        absolute_position=position,
                        prefix_cached_tokens=prefix_cached_tokens,
                    )
                )
                if anchor_position is None:
                    # The exact BF16/FP16 predecessor was reused from the
                    # authenticated cache and is not part of this request's
                    # executed-suffix anchor tree.  Retain the existing
                    # conservative quantized RMSNorm check for that row.
                    continue
                try:
                    input_raw = anchor_rows[input_stage_id][anchor_position]
                except KeyError:
                    # Some all-attention corridors open only their signed
                    # endpoints.  They retain the pre-existing conservative
                    # int8 interval until the exact predecessor is available.
                    continue
                input_values = _decode_row_v3(input_raw, anchor_encoding)
                if len(input_values) != embed_hidden:
                    raise _fail(
                        f"coupling l{layer} exact input norm source has the "
                        "wrong width"
                    )
                expected_input = quantize_execution_anchor_row_v3(
                    row_bytes=input_raw,
                    scale=rin_scale,
                    encoding_id=anchor_encoding,
                )
                if not _replay_capture_row_matches_v3(
                    rin_rows[token], expected_input
                ):
                    raise _fail(
                        f"coupling l{layer} exact input norm source is "
                        "detached from its authenticated residual row"
                    )
                exact_input_norm_sources[token] = input_values
        for which, norm_row, norm_scale, source_rows, source_scale,                 target_key, target_scale_name in (
            (
                "input",
                input_norm_row,
                input_norm_scale,
                rin_rows,
                rin_scale,
                "qkv",
                f"l{layer}.qkv_x",
            ),
            (
                "post",
                post_norm_row,
                post_norm_scale,
                mid_rows,
                mid_scale,
                "gate_up",
                f"l{layer}.gate_up_x",
            ),
        ):
            target_rows = opened_projections[(layer, target_key)][0]
            target_scale = bits_to_scale_v3(
                oracle_by_id[target_scale_name].scale_bits
            )
            norm_cols = challenge.norm_cols_for(
                layer_index=layer, hidden_dim=embed_hidden, which=which
            )
            for token in layer_tokens:
                source_row = source_rows[token]
                exact_source = (
                    exact_input_norm_sources[token]
                    if which == "input" and token in exact_input_norm_sources
                    else None
                )
                saturated_source_intervals = {}
                direct_saturated_outputs = {}
                if exact_source is not None:
                    denominator_interval = None
                elif any(value in (-128, 127) for value in source_row):
                    (
                        denominator_interval,
                        saturated_source_intervals,
                        direct_saturated_outputs,
                    ) = _rmsnorm_saturation_aware_interval(
                        source_row=source_row,
                        source_scale=source_scale,
                        target_row=target_rows[token],
                        target_scale=target_scale,
                        norm_row=norm_row,
                        norm_scale=norm_scale,
                        norm_gain_offset=norm_gain_offset,
                        epsilon=rmsnorm_epsilon,
                    )
                else:
                    denominator_interval = _rmsnorm_denominator_interval(
                        source_row=source_row,
                        source_scale=source_scale,
                        epsilon=rmsnorm_epsilon,
                    )
                for col in norm_cols:
                    target_value = target_rows[token][col]
                    check_kwargs = dict(
                        target_value=target_value,
                        target_scale=target_scale,
                        norm_weight=norm_row[col],
                        norm_scale=norm_scale,
                        norm_gain_offset=norm_gain_offset,
                        column=col,
                        epsilon=rmsnorm_epsilon,
                        what=f"coupling l{layer} {which} RMSNorm link",
                        kind=f"rmsnorm_{which}",
                        failure=(
                            f"coupling l{layer} {which} RMSNorm link is "
                            "outside the quantization corridor (fabricated "
                            "norm trace)"
                        ),
                    )
                    if exact_source is not None:
                        _rmsnorm_exact_source_corridor_check(
                            source_values=exact_source,
                            **check_kwargs,
                        )
                    elif col in direct_saturated_outputs:
                        lower, upper = direct_saturated_outputs[col]
                        delta, target_quant, target_center = (
                            _quantized_target_interval_delta(
                                expected_lower=lower,
                                expected_upper=upper,
                                target_value=target_value,
                                target_scale=target_scale,
                            )
                        )
                        _fixed_quantization_corridor_check(
                            delta=delta,
                            quant=target_quant,
                            relative=_REL_COEFF
                            * max(
                                abs(lower),
                                abs(upper),
                                abs(target_center),
                            ),
                            what=check_kwargs["what"],
                            kind=check_kwargs["kind"],
                            failure=check_kwargs["failure"],
                        )
                    else:
                        if (
                            col in saturated_source_intervals
                            and target_rows[token][col] in (-128, 127)
                        ):
                            # This coordinate's one-sided target rail was
                            # already incorporated into the global RMS
                            # equation above.  Re-expanding the correlated
                            # source/rms ratio as independent infinite
                            # intervals would lose that relation.
                            continue
                        _rmsnorm_corridor_check(
                            source_row=source_row,
                            source_scale=source_scale,
                            denominator_interval=denominator_interval,
                            selected_source_interval=(
                                saturated_source_intervals.get(col)
                            ),
                            **check_kwargs,
                        )

    if (
        tuple(c.layer_index for c in proof.gdn_couplings)
        != selected_gdn_layers
    ):
        raise _fail(
            "GDN coupling reveals do not cover exactly the selected GDN "
            "layers"
        )
    lean_gdn_runtime_rows = {}
    for coupling in proof.gdn_couplings:
        layer = coupling.layer_index
        layer_tokens = projection_tokens_by_layer[layer]
        norm_source_rows_by_token = {}
        if lean:
            if not coupling.runtime_rows:
                raise _fail(
                    f"lean GDN coupling l{layer} has no runtime replay rows"
                )
            lean_gdn_runtime_rows[layer] = tuple(coupling.runtime_rows)
            token_for_position = {
                position: token
                for token, position in lean_positions_by_layer[layer].items()
            }
            expected_norm_positions = tuple(
                sorted(token_for_position)
            )
            actual_norm_positions = tuple(
                int(record[0]) for record in coupling.norm_source_rows
            )
            if actual_norm_positions != expected_norm_positions:
                raise _fail(
                    f"lean GDN coupling l{layer} norm-source rows do not "
                    "match the nonce-derived projection layout"
                )
            from verallm.proof_v3.economic_execution_anchor import (
                _decode_row_v3,
            )

            for position, input_raw, post_raw in coupling.norm_source_rows:
                input_values = _decode_row_v3(input_raw, anchor_encoding)
                post_values = _decode_row_v3(post_raw, anchor_encoding)
                if (
                    len(input_values) != embed_hidden
                    or len(post_values) != embed_hidden
                ):
                    raise _fail(
                        f"lean GDN coupling l{layer} norm-source width is "
                        "malformed"
                    )
                norm_source_rows_by_token[token_for_position[position]] = (
                    input_values,
                    post_values,
                    input_raw,
                    post_raw,
                )
        elif coupling.runtime_rows:
            raise _fail(
                "non-lean GDN coupling carries post-nonce runtime rows"
            )
        elif coupling.norm_source_rows:
            raise _fail(
                "non-lean GDN coupling carries post-nonce norm-source rows"
            )
        oracle_ids = {
            "qkvz_y": (
                coupling.qkvz_y_oracle_index,
                f"l{layer}.gdn_qkvz_y",
            ),
            "ba_y": (
                coupling.ba_y_oracle_index,
                f"l{layer}.gdn_ba_y",
            ),
            "gdn_o_y": (
                coupling.gdn_o_y_oracle_index,
                f"l{layer}.gdn_o_y",
            ),
            "down_y": (
                coupling.down_y_oracle_index,
                f"l{layer}.down_y",
            ),
            "mid": (
                coupling.mid_oracle_index,
                f"l{layer}.mid_residual",
            ),
            "gate_up_y": (
                coupling.gate_up_y_oracle_index,
                f"l{layer}.gate_up_y",
            ),
        }
        oracles = {}
        for key, (index, expected_id) in oracle_ids.items():
            oracle = proof.oracles[index]
            if oracle.oracle_id != expected_id:
                raise _fail("GDN coupling reveal references the wrong oracles")
            oracles[key] = oracle

        projection_specs = audited_projections_for_layer_kind_v3("gdn")
        revealed_bias = dict(coupling.bias_rows)
        biases: dict[str, tuple[tuple[int, ...], float]] = {}
        for proj_index, (_x, _s, manifest_suffix) in enumerate(
            projection_specs
        ):
            bias_name = f"l{layer}.{manifest_suffix}_bias"
            if not artifacts.has_entry(bias_name):
                continue
            reveal = revealed_bias.pop(proj_index, None)
            if reveal is None:
                raise _fail(
                    f"GDN coupling l{layer} is missing the manifest-required "
                    f"{manifest_suffix} bias reveal"
                )
            biases[manifest_suffix] = (
                artifacts.verify_weight_row(name=bias_name, reveal=reveal),
                artifacts.scale_for(bias_name),
            )
        if revealed_bias:
            raise _fail(
                f"GDN coupling l{layer} reveals a bias with no manifest entry"
            )

        def _gdn_bias_at(
            suffix: str, out_col: int
        ) -> tuple[float, float]:
            entry = biases.get(suffix)
            if entry is None:
                return 0.0, 0.0
            values, scale = entry
            if out_col >= len(values):
                raise _fail(
                    f"GDN coupling l{layer} {suffix} bias row is narrower "
                    "than the audited output column"
                )
            return values[out_col] * scale, 0.5 * scale

        row_openings = (
            ("qkvz_y", coupling.qkvz_y_opening),
            ("ba_y", coupling.ba_y_opening),
            ("mid", coupling.mid_opening),
        )
        opened_y_rows = {}
        for key, opening in row_openings:
            oracle = oracles[key]
            rows = _opened_rows(
                oracle=oracle,
                base_binding=base_binding,
                rows=layer_tokens,
                opening=opening,
                require_int8=True,
                what=f"GDN coupling l{layer} {key}",
                expect_mode=2,
            )
            if anchor_binding is not None:
                anchor_binding.verify_rows(
                    oracle_id=oracle.oracle_id,
                    actual_rows=rows,
                    row_indices=layer_tokens,
                )
            opened_y_rows[key] = rows
        residual_cols = challenge.residual_cols_for(
            layer_index=layer,
            hidden_dim=embed_hidden,
        )
        for key, suffix, opening in (
            ("gdn_o_y", "gdn_o", coupling.gdn_o_y_opening),
            ("down_y", "down", coupling.down_y_opening),
        ):
            outs = opened_projections[(layer, suffix)][5]
            cells = tuple(
                (token, col)
                for token in layer_tokens
                for col in sorted(set(outs) | set(residual_cols))
            )
            values = _opened_cells(
                oracle=oracles[key],
                base_binding=base_binding,
                cells=cells,
                opening=opening,
                what=f"GDN coupling l{layer} {key}",
                expect_mode=2,
            )
            if anchor_binding is not None:
                anchor_binding.verify_cells(
                    oracle_id=oracles[key].oracle_id,
                    actual_cells=values,
                    cells=cells,
                )
            opened_y_rows[key] = values

        if lean:
            from verallm.proof_v3.economic_execution_anchor import (
                quantize_execution_anchor_row_v3,
            )

            token_for_position = {
                position: token
                for token, position in lean_positions_by_layer[layer].items()
            }
            qkvz_scale = bits_to_scale_v3(
                oracles["qkvz_y"].scale_bits
            )
            ba_scale = bits_to_scale_v3(oracles["ba_y"].scale_bits)
            gdn_o_x_rows = opened_projections[(layer, "gdn_o")][0]
            gdn_o_x_scale = opened_projections[(layer, "gdn_o")][3]
            qkvz_columns = gdn_projection_output_columns_by_key[
                (layer, "gdn_qkvz")
            ]
            ba_columns = gdn_projection_output_columns_by_key[
                (layer, "gdn_ba")
            ]
            gdn_o_columns = gdn_recurrence_input_columns_by_layer[layer]
            for position, qkvz_raw, ba_raw, output_raw in (
                coupling.runtime_rows
            ):
                token = token_for_position.get(int(position))
                if token is None:
                    # Every forwarded row is consumed by the signed recurrence
                    # replay below. Projection coupling intentionally remains
                    # nonce-sampled at the signed corridor positions.
                    continue
                qkvz_expected = quantize_execution_anchor_row_v3(
                    row_bytes=qkvz_raw,
                    scale=qkvz_scale,
                    encoding_id=anchor_encoding,
                )
                ba_expected = quantize_execution_anchor_row_v3(
                    row_bytes=ba_raw,
                    scale=ba_scale,
                    encoding_id=anchor_encoding,
                )
                output_expected = quantize_execution_anchor_row_v3(
                    row_bytes=output_raw,
                    scale=gdn_o_x_scale,
                    encoding_id=anchor_encoding,
                )
                if (
                    len(qkvz_expected) != len(qkvz_columns)
                    or len(ba_expected) != len(ba_columns)
                    or len(output_expected) != len(gdn_o_columns)
                ):
                    raise _fail(
                        "lean GDN compact runtime row has the wrong geometry"
                    )
                for column, expected in zip(
                    qkvz_columns, qkvz_expected, strict=True
                ):
                    actual = opened_y_rows["qkvz_y"][token][column]
                    if not _replay_capture_cell_matches_v3(actual, expected):
                        raise _fail_with_row_mismatch(
                            "lean GDN qkvz replay row is detached from its "
                            "registered projection",
                            layer=layer,
                            position=int(position),
                            token=token,
                            column=column,
                            actual=actual,
                            expected=expected,
                        )
                for column, expected in zip(
                    ba_columns, ba_expected, strict=True
                ):
                    actual = opened_y_rows["ba_y"][token][column]
                    if not _replay_capture_cell_matches_v3(actual, expected):
                        raise _fail_with_row_mismatch(
                            "lean GDN BA replay row is detached from its "
                            "registered projection",
                            layer=layer,
                            position=int(position),
                            token=token,
                            column=column,
                            actual=actual,
                            expected=expected,
                        )
                for column, expected in zip(
                    gdn_o_columns, output_expected, strict=True
                ):
                    actual = gdn_o_x_rows[token][column]
                    if not _replay_capture_cell_matches_v3(actual, expected):
                        raise _fail_with_row_mismatch(
                            "lean GDN recurrence output is detached from its "
                            "registered output projection",
                            layer=layer,
                            position=int(position),
                            token=token,
                            column=column,
                            actual=actual,
                            expected=expected,
                        )

        # qkvz and BA are parallel registered projections of the identical
        # normalized runtime input.  This is checked in the raw authenticated
        # domain as well as through the RMSNorm relation below.
        if lean:
            qkvz_inputs = opened_projections[(layer, "gdn_qkvz")][0]
            ba_inputs = opened_projections[(layer, "gdn_ba")][0]
            if any(
                qkvz_inputs[token] != ba_inputs[token]
                for token in layer_tokens
            ):
                raise _fail(
                    f"GDN coupling l{layer} qkvz/BA inputs disagree"
                )
        else:
            if anchor_rows is None:
                raise _fail(
                    "GDN coupling requires streaming execution anchors"
                )
            for token in layer_tokens:
                position = challenge.candidate_sequence_positions[token]
                if (
                    anchor_rows[f"l{layer}.gdn_qkvz_input"][position]
                    != anchor_rows[f"l{layer}.gdn_ba_input"][position]
                ):
                    raise _fail(
                        f"GDN coupling l{layer} qkvz/BA runtime inputs "
                        "disagree"
                    )

        # Every architecture-specific registered projection is tied to its
        # authenticated runtime output at the nonce-selected output cells.
        if lean and canonical_gdn_projection_inputs:
            from verallm.proof_v3.gdn_projection_input import (
                canonical_gdn_projection_inputs_v3,
            )

            canonical_input_norm_row = artifacts.verify_weight_row(
                name=f"l{layer}.input_norm",
                reveal=coupling.input_norm_row,
            )
            canonical_input_norm_scale = artifacts.scale_for(
                f"l{layer}.input_norm"
            )
            canonical_scale_bits, canonical_scale, canonical_x_rows = (
                canonical_gdn_projection_inputs_v3(
                    source_rows_by_token={
                        token: norm_source_rows_by_token[token][0]
                        for token in layer_tokens
                    },
                    norm_row=canonical_input_norm_row,
                    norm_scale=canonical_input_norm_scale,
                    norm_gain_offset=norm_gain_offset,
                    epsilon=rmsnorm_epsilon,
                )
            )
            for projection in ("gdn_qkvz", "gdn_ba"):
                projection_rows = opened_projections[(layer, projection)]
                x_rows = projection_rows[0]
                x_scale_bits = oracle_by_id[
                    f"l{layer}.{projection}_x"
                ].scale_bits
                if x_scale_bits != canonical_scale_bits:
                    raise _fail(
                        f"GDN coupling l{layer} {projection} input scale is "
                        "detached from the authenticated norm source"
                    )
                if any(
                    tuple(x_rows[token]) != canonical_x_rows[token]
                    for token in layer_tokens
                ):
                    raise _fail(
                        f"GDN coupling l{layer} {projection} input row is "
                        "detached from the authenticated norm source"
                    )
                opened_projections[(layer, projection)] = (
                    projection_rows[0],
                    projection_rows[1],
                    projection_rows[2],
                    canonical_scale,
                    projection_rows[4],
                    projection_rows[5],
                )
        for suffix, row_key in (
            ("gdn_qkvz", "qkvz_y"),
            ("gdn_ba", "ba_y"),
            ("gdn_o", "gdn_o_y"),
            ("down", "down_y"),
        ):
            (
                x_rows,
                surrogate,
                weight_rows,
                x_scale,
                w_scale,
                outs,
            ) = opened_projections[(layer, suffix)]
            y_rows = opened_y_rows[row_key]
            y_scale = bits_to_scale_v3(oracles[row_key].scale_bits)
            x_sq_by_token = {
                token: sum(value * value for value in x_rows[token])
                for token in layer_tokens
            }
            w_sq_by_out = {
                out: _projection_weight_row_sq(
                    layer=layer,
                    suffix=suffix,
                    out=out,
                    weight_rows=weight_rows,
                )
                for out in outs
            }
            for token in layer_tokens:
                for out in outs:
                    bias_value, bias_quant = _gdn_bias_at(suffix, out)
                    _corridor_check(
                        surrogate_value=_projection_surrogate_value(
                            surrogate, token, out
                        ),
                        captured_value=(
                            y_rows[(token, out)]
                            if row_key in {"gdn_o_y", "down_y"}
                            else y_rows[token][out]
                        ),
                        x_row=x_rows[token],
                        w_row=weight_rows.get(out, ()),
                        x_scale=x_scale,
                        w_scale=w_scale,
                        y_scale=y_scale,
                        what=(
                            f"GDN coupling l{layer}.{suffix} Y corridor"
                        ),
                        bias_value=bias_value,
                        bias_quant=bias_quant,
                        stats=corridor_stats,
                        kind=f"y_{suffix}",
                        x_sq=x_sq_by_token[token],
                        w_sq=w_sq_by_out[out],
                        sigma_cap=corridor_sigma,
                    )

        # Common gated MLP coupling.
        (
            gu_x_rows,
            gu_surrogate,
            gu_weight_rows,
            gu_x_scale,
            gu_w_scale,
            gu_outs,
        ) = opened_projections[(layer, "gate_up")]
        gate_up_oracle = oracles["gate_up_y"]
        gu_y_scale = bits_to_scale_v3(gate_up_oracle.scale_bits)
        inter_dim = gate_up_oracle.col_count // 2
        mlp_cols = challenge.mlp_cols_for(
            layer_index=layer, inter_dim=inter_dim
        )
        gate_up_cells = {
            (token, out)
            for token in layer_tokens
            for out in gu_outs
        }
        for token in layer_tokens:
            for col in mlp_cols:
                gate_up_cells.add((token, col))
                gate_up_cells.add((token, inter_dim + col))
        gate_up_cells = tuple(sorted(gate_up_cells))
        gate_up_values = _opened_cells(
            oracle=gate_up_oracle,
            base_binding=base_binding,
            cells=gate_up_cells,
            opening=coupling.gate_up_y_opening,
            what=f"GDN coupling l{layer} gate_up_y",
            expect_mode=2,
        )
        if anchor_binding is not None:
            anchor_binding.verify_cells(
                oracle_id=gate_up_oracle.oracle_id,
                actual_cells=gate_up_values,
                cells=gate_up_cells,
            )
        gu_x_sq = {
            token: sum(value * value for value in gu_x_rows[token])
            for token in layer_tokens
        }
        gu_w_sq = {
            out: _projection_weight_row_sq(
                layer=layer,
                suffix="gate_up",
                out=out,
                weight_rows=gu_weight_rows,
            )
            for out in gu_outs
        }
        for token in layer_tokens:
            for out in gu_outs:
                bias_value, bias_quant = _gdn_bias_at("gate_up", out)
                _corridor_check(
                    surrogate_value=_projection_surrogate_value(
                        gu_surrogate, token, out
                    ),
                    captured_value=gate_up_values[(token, out)],
                    x_row=gu_x_rows[token],
                    w_row=gu_weight_rows.get(out, ()),
                    x_scale=gu_x_scale,
                    w_scale=gu_w_scale,
                    y_scale=gu_y_scale,
                    what=f"GDN coupling l{layer}.gate_up Y corridor",
                    bias_value=bias_value,
                    bias_quant=bias_quant,
                    stats=corridor_stats,
                    kind="y_gate_up",
                    x_sq=gu_x_sq[token],
                    w_sq=gu_w_sq[out],
                    sigma_cap=corridor_sigma,
                )
        down_x_rows = opened_projections[(layer, "down")][0]
        down_x_scale = opened_projections[(layer, "down")][3]
        if exact_mlp_activation:
            _verify_exact_mlp_activation_v3(
                layer=layer,
                layer_tokens=layer_tokens,
                mlp_cols=mlp_cols,
                gate_up_values=gate_up_values,
                inter_dim=inter_dim,
                gate_up_scale=gu_y_scale,
                down_x_rows=down_x_rows,
                down_x_scale=down_x_scale,
                gate_up_surrogate=gu_surrogate,
                gate_up_x_rows=gu_x_rows,
                gate_up_weight_rows=gu_weight_rows,
                gate_up_x_scale=gu_x_scale,
                gate_up_w_scale=gu_w_scale,
                gate_up_bias_at=lambda output: _gdn_bias_at(
                    "gate_up", output
                ),
                gate_up_x_sq=gu_x_sq,
                gate_up_w_sq=gu_w_sq,
                stats=corridor_stats,
            )
        else:
            for token in layer_tokens:
                for col in mlp_cols:
                    gate_i8 = gate_up_values[(token, col)]
                    up_i8 = gate_up_values[(token, inter_dim + col)]
                    down_i8 = down_x_rows[token][col]
                    gate = gate_i8 * gu_y_scale
                    up = up_i8 * gu_y_scale
                    predicted = _silu(gate) * up
                    got = down_i8 * down_x_scale
                    quant = 0.5 * down_x_scale + 0.5 * gu_y_scale * (
                        1.1 * abs(up)
                        + abs(_silu(gate))
                        + 0.5 * gu_y_scale
                    )
                    if _swiglu_output_is_forced_to_quantization_rail_v3(
                        gate_i8=gate_i8,
                        up_i8=up_i8,
                        gate_up_scale=gu_y_scale,
                        output_i8=down_i8,
                        output_scale=down_x_scale,
                    ):
                        predicted = got
                    _fixed_quantization_corridor_check(
                        delta=abs(got - predicted),
                        quant=quant,
                        relative=_REL_COEFF * abs(predicted),
                        what=f"GDN coupling l{layer} elementwise MLP link",
                        kind="gdn_mlp_elementwise",
                        failure=(
                            f"GDN coupling l{layer} elementwise MLP link is "
                            "outside the quantization corridor"
                        ),
                    )

        # Residual chain: GDN output projection, then common MLP.
        rin_rows, rout_rows = opened_boundaries[layer]
        rin_scale = bits_to_scale_v3(
            oracle_by_id[f"l{layer}.residual_in"].scale_bits
        )
        rout_scale = bits_to_scale_v3(
            oracle_by_id[f"l{layer}.residual_out"].scale_bits
        )
        mid_rows = opened_y_rows["mid"]
        gdn_o_y_rows = opened_y_rows["gdn_o_y"]
        down_y_rows = opened_y_rows["down_y"]
        mid_scale = bits_to_scale_v3(oracles["mid"].scale_bits)
        gdn_o_y_scale = bits_to_scale_v3(
            oracles["gdn_o_y"].scale_bits
        )
        down_y_scale = bits_to_scale_v3(oracles["down_y"].scale_bits)
        if lean:
            from verallm.proof_v3.economic_execution_anchor import (
                quantize_execution_anchor_row_v3,
            )

            for token in layer_tokens:
                input_values, post_values, input_raw, post_raw = (
                    norm_source_rows_by_token[token]
                )
                if not _replay_capture_row_matches_v3(
                    tuple(rin_rows[token]),
                    quantize_execution_anchor_row_v3(
                        row_bytes=input_raw,
                        scale=rin_scale,
                        encoding_id=anchor_encoding,
                    ),
                ):
                    raise _fail(
                        f"lean GDN coupling l{layer} input norm source is "
                        "detached from its projection row"
                    )
                if not _replay_capture_row_matches_v3(
                    tuple(mid_rows[token]),
                    quantize_execution_anchor_row_v3(
                        row_bytes=post_raw,
                        scale=mid_scale,
                        encoding_id=anchor_encoding,
                    ),
                ):
                    raise _fail(
                        f"lean GDN coupling l{layer} post norm source is "
                        "detached from its projection row"
                    )
        for token in layer_tokens:
            for col in residual_cols:
                mid_i8 = mid_rows[token][col]
                rin_i8 = rin_rows[token][col]
                gdn_o_i8 = gdn_o_y_rows[(token, col)]
                mid_value = mid_i8 * mid_scale
                composed_mid = (
                    rin_i8 * rin_scale
                    + gdn_o_i8 * gdn_o_y_scale
                )
                quant_mid = 0.5 * (
                    mid_scale + rin_scale + gdn_o_y_scale
                )
                gdn_input_failure = (
                    f"GDN coupling l{layer} residual_in + output "
                    "projection != mid-residual"
                )
                if any(
                    value in (-128, 127)
                    for value in (mid_i8, rin_i8, gdn_o_i8)
                ):
                    _quantized_sum_corridor_check(
                        output_i8=mid_i8,
                        output_scale=mid_scale,
                        left_i8=rin_i8,
                        left_scale=rin_scale,
                        right_i8=gdn_o_i8,
                        right_scale=gdn_o_y_scale,
                        what=f"GDN coupling l{layer} input residual composition",
                        kind="gdn_attention_residual",
                        failure=gdn_input_failure,
                    )
                else:
                    _fixed_quantization_corridor_check(
                        delta=abs(mid_value - composed_mid),
                        quant=quant_mid,
                        relative=_REL_COEFF * max(
                            abs(mid_value),
                            abs(composed_mid),
                        ),
                        what=f"GDN coupling l{layer} input residual composition",
                        kind="gdn_attention_residual",
                        failure=gdn_input_failure,
                    )
                out_quantized = (
                    rout_rows[(token, col)]
                    if anchor_binding is not None
                    else rout_rows[token][col]
                )
                down_i8 = down_y_rows[(token, col)]
                out_value = out_quantized * rout_scale
                composed_out = (
                    mid_i8 * mid_scale
                    + down_i8 * down_y_scale
                )
                quant_out = 0.5 * (
                    rout_scale + mid_scale + down_y_scale
                )
                gdn_mlp_failure = (
                    f"GDN coupling l{layer} mid-residual + MLP output "
                    "!= residual_out"
                )
                if any(
                    value in (-128, 127)
                    for value in (out_quantized, mid_i8, down_i8)
                ):
                    _quantized_sum_corridor_check(
                        output_i8=out_quantized,
                        output_scale=rout_scale,
                        left_i8=mid_i8,
                        left_scale=mid_scale,
                        right_i8=down_i8,
                        right_scale=down_y_scale,
                        what=f"GDN coupling l{layer} MLP residual composition",
                        kind="gdn_mlp_residual",
                        failure=gdn_mlp_failure,
                    )
                else:
                    _fixed_quantization_corridor_check(
                        delta=abs(out_value - composed_out),
                        quant=quant_out,
                        relative=_REL_COEFF * max(
                            abs(out_value),
                            abs(composed_out),
                        ),
                        what=f"GDN coupling l{layer} MLP residual composition",
                        kind="gdn_mlp_residual",
                        failure=gdn_mlp_failure,
                    )

        # Both parallel GDN projections must be the input RMSNorm output;
        # the common MLP projection must be the post-GDN RMSNorm output.
        input_norm_row = artifacts.verify_weight_row(
            name=f"l{layer}.input_norm",
            reveal=coupling.input_norm_row,
        )
        post_norm_row = artifacts.verify_weight_row(
            name=f"l{layer}.post_norm",
            reveal=coupling.post_norm_row,
        )
        input_norm_scale = artifacts.scale_for(f"l{layer}.input_norm")
        post_norm_scale = artifacts.scale_for(f"l{layer}.post_norm")
        norm_relations = (
            (
                "input-qkvz",
                input_norm_row,
                input_norm_scale,
                rin_rows,
                rin_scale,
                "gdn_qkvz",
                "input",
            ),
            (
                "input-ba",
                input_norm_row,
                input_norm_scale,
                rin_rows,
                rin_scale,
                "gdn_ba",
                "input",
            ),
            (
                "post",
                post_norm_row,
                post_norm_scale,
                mid_rows,
                mid_scale,
                "gate_up",
                "post",
            ),
        )
        for (
            relation_name,
            norm_row,
            norm_scale,
            source_rows,
            source_scale,
            target_key,
            challenge_kind,
        ) in norm_relations:
            target_rows = opened_projections[(layer, target_key)][0]
            target_scale = opened_projections[(layer, target_key)][3]
            norm_cols = challenge.norm_cols_for(
                layer_index=layer,
                hidden_dim=embed_hidden,
                which=challenge_kind,
            )
            for token in layer_tokens:
                source_row = source_rows[token]
                exact_source = (
                    norm_source_rows_by_token[token][
                        0 if challenge_kind == "input" else 1
                    ]
                    if lean
                    else None
                )
                denominator_interval = (
                    None
                    if exact_source is not None
                    else _rmsnorm_denominator_interval(
                        source_row=source_row,
                        source_scale=source_scale,
                        epsilon=rmsnorm_epsilon,
                    )
                )
                for col in norm_cols:
                    target_value = target_rows[token][col]
                    check_kwargs = dict(
                        target_value=target_value,
                        target_scale=target_scale,
                        norm_weight=norm_row[col],
                        norm_scale=norm_scale,
                        norm_gain_offset=norm_gain_offset,
                        column=col,
                        epsilon=rmsnorm_epsilon,
                        what=(
                            f"GDN coupling l{layer} {relation_name} "
                            "RMSNorm link"
                        ),
                        kind=f"gdn_rmsnorm_{challenge_kind}",
                        failure=(
                            f"GDN coupling l{layer} {relation_name} RMSNorm "
                            "link is outside the quantization corridor"
                        ),
                    )
                    if exact_source is not None:
                        _rmsnorm_exact_source_corridor_check(
                            source_values=exact_source,
                            **check_kwargs,
                        )
                    else:
                        _rmsnorm_corridor_check(
                            source_row=source_row,
                            source_scale=source_scale,
                            denominator_interval=denominator_interval,
                            **check_kwargs,
                        )

    if lean and selected_gdn_layers:
        from verallm.proof_v3.economic_gdn_replay import (
            verify_lean_economic_gdn_replay_v3,
        )

        try:
            verify_lean_economic_gdn_replay_v3(
                runtime_rows_by_layer=lean_gdn_runtime_rows,
                opened_rows=anchor_rows,
                challenge=challenge,
                layer_kinds=layer_kinds,
                semantics=gdn_runtime_semantics,
                anchor_encoding_id=anchor_encoding,
                commitments=proof.execution_anchors,
                lane_reveals=proof.execution_anchor_lane_reveals,
            )
        except ProofV3VerificationError as exc:
            raise _fail(f"lean GDN recurrence verification failed: {exc}")

    # ---- (6.6) AGGREGATE corridor acceptance: a distributed low-amplitude
    # substitution shifts every cell of the affected link kinds a little;
    # each kind's mean squared normalized deviation then sits far outside
    # honest concentration even when no single cell crosses the per-cell
    # band.  Thresholding the MAX over kinds keeps unaffected kinds from
    # diluting the signal.
    if corridor_stats and _CORRIDOR_REPORT is None:
        by_kind: dict[str, list[float]] = {}
        by_relation: dict[tuple[str, str], list[float]] = {}
        for kind, delta, spread, relation in corridor_stats:
            normalized_sq = (delta / spread) ** 2
            by_kind.setdefault(kind, []).append(normalized_sq)
            by_relation.setdefault((kind, relation), []).append(
                normalized_sq
            )
        worst_kind, worst = max(
            (
                (kind, sum(values) / len(values))
                for kind, values in by_kind.items()
            ),
            key=lambda item: item[1],
        )
        chi2_cap = _CORRIDOR_CHI2
        manifest_bits = getattr(artifacts.manifest, "corridor_chi2_bits", 0)
        if manifest_bits:
            chi2_cap = bits_to_scale_v3(manifest_bits)
        if worst > chi2_cap:
            (_relation_kind, worst_relation), relation_values = max(
                by_relation.items(),
                key=lambda item: sum(item[1]) / len(item[1]),
            )
            relation_mean = sum(relation_values) / len(relation_values)
            raise _fail(
                "aggregate corridor statistic is outside the honest "
                "quantization envelope (distributed weight substitution); "
                f"worst_kind={worst_kind} mean_sq={worst:.9g} "
                f"cap={chi2_cap:.9g}; worst_relation={worst_relation} "
                f"relation_mean_sq={relation_mean:.9g} "
                f"cells={len(relation_values)}"
            )

    _profile_mark("transition-couplings")

    # ---- (7) MANDATORY top anchor: final hidden -> LM head -> token -------
    if proof.final is None:
        raise _fail("economic proof requires the final reveal (final missing)")
    final = proof.final
    if proof.oracles[final.final_oracle_index].oracle_id != "final_hidden":
        raise _fail("final reveal references the wrong oracles")
    if not challenge.audited_decode_positions:
        raise _fail("economic challenge has no audited decode position")
    audited_position = challenge.audited_decode_positions[0]
    if final.audited_position != audited_position:
        raise _fail(
            "final reveal is not for the validator-derived decode position"
        )
    observed_token = observed_output_token_ids[audited_position]
    if not 0 <= observed_token < lm_vocab:
        raise _fail("observed output token exceeds the committed vocabulary")
    final_rows = _opened_rows(
        oracle=final_oracle,
        base_binding=base_binding,
        rows=(audited_position,),
        opening=final.final_opening,
        require_int8=True,
        what="final hidden",
            expect_mode=2,
    )
    final_row = final_rows[audited_position]

    # ---- (7a) FINAL NORM LINK: the audited final hidden row must be the
    # RMSNorm of the last layer's residual_out row that produced the token
    last_layer = layer_universe[-1]
    last_rout_oracle = proof.oracles[final.last_residual_oracle_index]
    if last_rout_oracle.oracle_id != f"l{last_layer}.residual_out":
        raise _fail("final-norm link references the wrong residual oracle")
    producing_row = challenge.pool_row_for_decode_position(audited_position)
    last_rows = _opened_rows(
        oracle=last_rout_oracle,
        base_binding=base_binding,
        rows=(producing_row,),
        opening=final.last_residual_opening,
        require_int8=True,
        what="final-norm residual",
        expect_mode=(
            2 if lean else (0 if anchor_binding is not None else 2)
        ),
        packed_rows=_anchor_packed_rows(
            anchor_binding=anchor_binding,
            oracle_id=last_rout_oracle.oracle_id,
            row_indices=(producing_row,),
        ),
    )
    if anchor_binding is not None:
        anchor_binding.verify_rows(
            oracle_id=last_rout_oracle.oracle_id,
            actual_rows=last_rows,
            row_indices=(producing_row,),
        )
    elif lean:
        from verallm.proof_v3.economic_execution_anchor import (
            _decode_row_v3,
            quantize_execution_anchor_row_v3,
        )

        producing_position = (
            challenge.context_token_count - 1 + audited_position
        )
        if prefix_cached_tokens:
            producing_position -= prefix_cached_tokens
            if producing_position < 0:
                raise _fail(
                    "lean final residual precedes the authenticated cache "
                    "suffix"
                )
        try:
            raw_last = anchor_rows[
                f"l{last_layer}.residual_out"
            ][producing_position]
        except KeyError as exc:
            raise _fail(
                "lean final residual has no authenticated checkpoint row"
            ) from exc
        expected_last = quantize_execution_anchor_row_v3(
            row_bytes=raw_last,
            scale=bits_to_scale_v3(last_rout_oracle.scale_bits),
            encoding_id=anchor_encoding,
        )
        if not _replay_capture_row_matches_v3(
            last_rows[producing_row],
            expected_last,
        ):
            raise _fail(
                "lean final residual is detached from its pre-nonce checkpoint"
            )
        exact_final_source = _decode_row_v3(raw_last, anchor_encoding)
    final_norm_gains = artifacts.verify_weight_row(
        name="final_norm", reveal=final.final_norm_row
    )
    final_norm_scale = artifacts.scale_for("final_norm")
    last_scale = bits_to_scale_v3(last_rout_oracle.scale_bits)
    final_scale = bits_to_scale_v3(final_oracle.scale_bits)
    import math as _math

    source_real = (
        exact_final_source
        if exact_final_source is not None
        else [value * last_scale for value in last_rows[producing_row]]
    )
    mean_square = (
        sum(value * value for value in source_real) / embed_hidden
        + rmsnorm_epsilon
    )
    rms = _math.sqrt(mean_square)
    sum_abs = sum(abs(value) for value in source_real)
    for col in range(embed_hidden):
        if exact_final_source is not None:
            _rmsnorm_exact_source_corridor_check(
                target_value=final_row[col],
                target_scale=final_scale,
                source_values=exact_final_source,
                norm_weight=final_norm_gains[col],
                norm_scale=final_norm_scale,
                norm_gain_offset=norm_gain_offset,
                column=col,
                epsilon=rmsnorm_epsilon,
                denominator=rms,
                what="final RMSNorm link",
                kind="final_rmsnorm",
                failure=(
                    "final-norm link broken: audited final hidden row is not "
                    "the RMSNorm of the last residual_out (detached top anchor)"
                ),
            )
            continue
        gain = (
            final_norm_gains[col] * final_norm_scale
            + norm_gain_offset
        )
        predicted = source_real[col] / rms * gain
        delta, target_quant, got = _quantized_target_interval_delta(
            expected_lower=predicted,
            expected_upper=predicted,
            target_value=final_row[col],
            target_scale=final_scale,
        )
        quant = (
            target_quant
            + (abs(gain) / rms)
            * 0.5
            * last_scale
            * (
                1.0
                + abs(source_real[col]) * sum_abs / (embed_hidden * mean_square)
            )
            + abs(source_real[col] / rms) * 0.5 * final_norm_scale
        )
        _fixed_quantization_corridor_check(
            delta=delta,
            quant=quant,
            relative=_REL_COEFF * abs(predicted),
            what="final RMSNorm link",
            kind="final_rmsnorm",
            failure=(
                "final-norm link broken: audited final hidden row is not the "
                "RMSNorm of the last residual_out (detached top anchor)"
            ),
        )

    top_k = getattr(artifacts.manifest, "lm_head_argmax_top_k", 1)
    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise _fail("signed LM-head top-k policy is malformed")
    sampled_vocab = challenge.vocab_rows_for(vocab_size=lm_vocab)
    terminal_escalation = (
        compact_terminal and challenge.full_row_projection_audit
    )
    if compact_terminal:
        expected_candidate_count = min(top_k, lm_vocab)
        candidate_tokens = tuple(final.candidate_token_rows)
        if (
            len(candidate_tokens) != expected_candidate_count
            or observed_token not in candidate_tokens
            or any(not 0 <= row < lm_vocab for row in candidate_tokens)
        ):
            raise _fail(
                "compact terminal candidate rows do not match the signed "
                "top-k policy"
            )
        if final.logits_openings:
            raise _fail(
                "compact terminal proof carries legacy logits openings"
            )
        if terminal_escalation:
            if len(final.revealed_logits) != lm_vocab:
                raise _fail(
                    "terminal escalation does not reveal the exact vocabulary"
                )
            full_logits: list[int] = list(final.revealed_logits)
        else:
            if final.revealed_logits:
                raise _fail(
                    "sampled terminal proof carries an unsolicited full "
                    "vocabulary"
                )
            full_logits = []
        expected_vocab = tuple(
            sorted(set(sampled_vocab) | set(candidate_tokens))
        )
    else:
        if final.candidate_token_rows or final.revealed_logits:
            raise _fail(
                "legacy terminal proof carries compact-v9 fields"
            )
        expected_vocab = tuple(
            sorted(set(sampled_vocab) | {observed_token})
        )
    if tuple(reveal.row_index for reveal in final.lm_head_rows) != expected_vocab:
        raise _fail("LM-head rows are not the validator-derived selection")
    if not compact_terminal:
        # Legacy profiles authenticate every pre-nonce logits block.
        if len(final.logits_openings) != logits_block_count:
            raise _fail(
                "final logits openings do not cover every committed block"
            )
        full_logits = [0] * lm_vocab
        for block, (oracle_index, opening) in enumerate(final.logits_openings):
            block_oracle = proof.oracles[oracle_index]
            if block_oracle.oracle_id != logits_block_oracle_id_v3(block):
                raise _fail("final logits opening references the wrong block")
            block_rows = _opened_rows(
                oracle=block_oracle,
                base_binding=base_binding,
                rows=(audited_position,),
                opening=opening,
                require_int8=False,
                what=f"surrogate logits block {block}",
                expect_mode=3,
                expect_bounded_width=bounded_byte_width_v3(
                    artifacts.entry("lm_head").in_dim
                ),
            )
            base_col = block * logits_block_cols
            for offset, value in enumerate(block_rows[audited_position]):
                full_logits[base_col + offset] = value
    committed_logits: dict[int, int] = {}
    for reveal in final.lm_head_rows:
        lm_row = artifacts.verify_weight_row(name="lm_head", reveal=reveal)
        recomputed = sum(
            a * b for a, b in zip(final_row, lm_row, strict=True)
        )
        if full_logits and recomputed != full_logits[reveal.row_index]:
            raise _fail(
                "top anchor: exact logit recompute disagrees with the "
                "revealed full-vocabulary row"
            )
        committed_logits[reveal.row_index] = recomputed
    _profile_mark("top-anchor-openings")

    # Sampled-chunk attention audit (bounded, context-length-independent): when
    # the signed manifest mandates it, the request is rejected fail-closed unless
    # the validator ran the audit and set artifacts.attention_audit_passed. The
    # audit itself (dispatch_economic_attention_audit_v3) runs off the reveal +
    # committed KV oracles; wiring the reveal into the proof is the plumbing step.
    if bool(getattr(artifacts.manifest, "attn_audit_required", 0)):
        from verallm.proof_v3.economic_attention_section import (
            attention_layer_commitment_v3,
            verify_attention_section_v3,
        )
        from verallm.proof_v3.scored_attention_reference import (
            SCORED_SCHEME_RATIONAL_V2,
            SCORED_SCHEME_V1,
        )

        # SIGNED scheme gate: the legacy uniform product-domain scheme is
        # proven UNSOUND for runtime-output binding (bucket width
        # 2*sqrt(D)*max|q|*max|k|/2^sb nats, independent of qk_bits) -- no
        # qualified hard profile may select it.  The verifier accepts
        # EXACTLY the scheme the authority signed: V1 (scored prob-path)
        # or V2 (rational, SCORED_SCHEME_RATIONAL_V2); everything the
        # section verifies is constructed for that one scheme, so a
        # V1<->V2 downgrade or cross-scheme proof fails closed (the
        # rational flag and V2 ABI are transcript-bound in the tile
        # statement digest).
        signed_scheme = int(getattr(artifacts.manifest, "attn_scheme", 0))
        if signed_scheme not in (SCORED_SCHEME_V1, SCORED_SCHEME_RATIONAL_V2):
            raise _fail(
                "signed manifest requires the attention audit but does not "
                "pin a qualified scored attention scheme (the product-domain "
                "scheme is not qualifiable for hard audits)"
            )

        if signed_scheme == SCORED_SCHEME_RATIONAL_V2:
            # V2 (rational): the capture-kv rational bundle rides INSIDE
            # the economic proof and verifies through the same adapter
            # the canary audit uses -- kv equality + row transport vs
            # the pre-nonce roots, signed bridge bounds, all inputs
            # authenticated via the pre-nonce capture-kv commitment.
            _verify_rational_attention_bundle_v3(
                artifacts=artifacts, envelope=envelope,
                challenge=challenge, proof=proof,
                base_binding=base_binding,
                profile=profile,
                anchor_rows=anchor_rows,
                anchor_binding=anchor_binding,
                anchor_encoding=anchor_encoding,
                opened_projections=opened_projections,
                streaming_metadata=streaming_attention_metadata,
                gdn_lane_keys=gdn_lane_keys,
                auxiliary_lane_keys=transition_lane_keys,
                prefix_cache_lanes=_prefix_cache_lanes,
                prefix_cache_projection_heads=(
                    prefix_cache_projection_heads
                ),
            )
        else:
            if getattr(proof, "attention", None) is not None:
                raise _fail(
                    "proof carries a rational attention section against "
                    "a scheme-1 manifest (cross-scheme section rejected)"
                )
            # INLINE verification (no external/mutable verdict): the adapter
            # itself authenticates the revealed Q/K/V/attn_o slices against the
            # pre-nonce committed capture roots, recomputes the sampled chunks,
            # and checks composition against the OPENED output. Roots are
            # request-bound via request_binding; the claims digest is the
            # pre-nonce committed one.
            section = getattr(artifacts, "attention_section", None)
            roots = getattr(artifacts, "attention_roots", None)
            if section is None or roots is None:
                raise _fail(
                    "signed manifest requires the attention audit but the proof "
                    "carries no attention section/roots (fail-closed)"
                )
            # capture_binding is the PRE-NONCE envelope digest (the roots were
            # capture-committed before the nonce); everything is authenticated
            # against those roots inline -- no trust boolean, no external
            # verdict.
            # PRE-NONCE binding (#5): the attention roots + claims digest are
            # folded into capture_chain_digest -> execution_root -> the nonce,
            # so the miner cannot fabricate roots post-nonce. The validator
            # supplies the expected per-layer commitment (recovered from the
            # authenticated pre-nonce capture_chain_digest); the adapter
            # requires the section's roots to hash to it. Fail-closed when the
            # manifest mandates the audit but no authenticated commitment is
            # present.
            expected_commitment = getattr(
                artifacts, "attention_root_commitment", None)
            if expected_commitment is None:
                raise _fail(
                    "attention audit required but no pre-nonce root commitment "
                    "is bound for this request (fail-closed)"
                )
            if attention_layer_commitment_v3(
                    layer=section.layer, roots=roots) != expected_commitment:
                raise _fail(
                    "attention capture roots do not match the pre-nonce "
                    "committed root commitment (post-nonce root fabrication)"
                )
            # CROSS-SCHEME gate: the section's statement scheme must equal
            # the SIGNED scheme exactly -- a V1 proof against a signed V2
            # manifest (downgrade) or vice versa fails closed here, before
            # any section verification runs.
            if int(getattr(section.statement, "rational", 0)) != int(
                    signed_scheme == SCORED_SCHEME_RATIONAL_V2):
                raise _fail(
                    "attention section scheme does not match the signed "
                    "attn_scheme (cross-scheme proof rejected)"
                )
            try:
                section_o_rows = verify_attention_section_v3(
                    capture_binding=envelope.digest(), roots=roots,
                    section=section,
                    expected_claims_digest=roots.claims_digest)
            except ProofV3VerificationError as exc:
                raise _fail(f"attention section verification failed: {exc}")

            # RUNTIME-OUTPUT BRIDGE (release blocker): the composed attention
            # output above is only int8 claims math until it is bound to what
            # the model ACTUALLY consumed -- the captured o_proj input.  When
            # the signed manifest pins a calibration digest, this is mandatory.
            calib_digest = getattr(
                artifacts.manifest, "attn_calibration_digest", b"")
            calib_set_digest = getattr(
                artifacts.manifest, "attn_calibration_set_digest", b"")
            if calib_digest or calib_set_digest:
                from verallm.proof_v3.economic_attention_section import (
                    verify_attention_output_bridge_v3,
                )

                if calib_set_digest:
                    # Multi-band release path: the SIGNED set digest selects
                    # the band for this section's AUTHENTICATED key count (the
                    # statement was already verified against the pre-nonce
                    # capture roots above).  Single/set exclusivity, digest
                    # match, and out-of-domain contexts all fail closed inside
                    # the binder.
                    from verallm.proof_v3.scored_calibration_set import (
                        select_signed_calibration_v3,
                    )

                    calibration_set = getattr(
                        artifacts, "attn_calibration_set", None)
                    if calibration_set is None:
                        raise _fail(
                            "manifest pins an attention calibration SET "
                            "digest but the validator loaded no calibration "
                            "set (fail-closed)"
                        )
                    try:
                        calibration = select_signed_calibration_v3(
                            artifacts.manifest, calibration_set,
                            section.statement.key_count)
                    except ProofV3Error as exc:
                        raise _fail(f"attention calibration set: {exc}")
                else:
                    calibration = getattr(artifacts, "attn_calibration", None)
                    if calibration is None:
                        raise _fail(
                            "manifest pins an attention calibration digest "
                            "but the validator loaded no calibration blob "
                            "(fail-closed)"
                        )
                    if calibration.digest != calib_digest:
                        raise _fail(
                            "attention calibration blob digest does not "
                            "match the signed manifest (wrong or tampered "
                            "blob)"
                        )
                bridge_opening = getattr(
                    artifacts, "attention_bridge_opening", None)
                if bridge_opening is None:
                    raise _fail(
                        "attention audit requires the o_proj bridge opening "
                        "but none was provided (fail-closed)"
                    )
                bridge_oracle = oracle_by_id.get(
                    f"l{section.layer}.attn_o_x")
                if bridge_oracle is None:
                    raise _fail(
                        "audited attention layer has no committed o_proj "
                        "input oracle to bridge against"
                    )
                bridge_tokens = tuple(
                    section.statement.query_positions[r]
                    for r in section.rows)
                ox8_rows = _opened_rows(
                    oracle=bridge_oracle,
                    base_binding=base_binding,
                    rows=bridge_tokens,
                    opening=bridge_opening,
                    require_int8=True,
                    what=f"attention l{section.layer} o_proj bridge",
            expect_mode=2,
                )
                try:
                    verify_attention_output_bridge_v3(
                        section=section, o_rows=section_o_rows,
                        ox8_rows_by_token=ox8_rows,
                        calibration_heads=calibration.heads_for(
                            section.layer))
                except ProofV3VerificationError as exc:
                    raise _fail(f"attention output bridge failed: {exc}")

    if not bool(getattr(artifacts.manifest, "attn_audit_required", 0)):
        # an unsolicited attention section is pure adversarial parse
        # surface -- the signed manifest did not mandate the audit
        if getattr(proof, "attention", None) is not None:
            raise _fail(
                "proof carries an attention section the signed manifest "
                "does not mandate (unsolicited section rejected)"
            )
    _profile_mark("attention")

    from verallm.proof_v3.projection_manifest import (
        LM_HEAD_CATALOG_BINDING_V3,
    )

    lm_head_mode = int(
        getattr(artifacts.manifest, "lm_head_binding", 0)
    )
    lm_head_int8 = getattr(artifacts, "lm_head_int8", None)
    lm_head_catalog = getattr(
        artifacts,
        "lm_head_catalog_binding",
        None,
    )
    if lm_head_mode == LM_HEAD_CATALOG_BINDING_V3:
        if lm_head_catalog is None:
            raise _fail(
                "top anchor: signed manifest requires the weightless "
                "LM-head catalog but the validator has no authenticated "
                "catalog"
            )
        from verallm.proof_v3.economic_lm_head_catalog_fold import (
            verify_lm_head_catalog_folds_v3,
        )

        if not compact_terminal or terminal_escalation:
            verify_lm_head_catalog_folds_v3(
                binding=lm_head_catalog,
                folded_weights=final.lm_head_catalog_folds,
                hidden_row_int8=final_row,
                revealed_logits=full_logits,
                selection_seed=bytes(challenge.selection_seed),
                envelope_digest=envelope.digest(),
                manifest_digest=artifacts.manifest.digest(),
                audited_position=final.audited_position,
            )
        elif final.lm_head_catalog_folds:
            raise _fail(
                "sampled terminal proof carries unsolicited catalog folds"
            )
    elif final.lm_head_catalog_folds:
        raise _fail(
            "top anchor carries unsolicited LM-head catalog folds"
        )
    if lm_head_mode == 1 and lm_head_int8 is None:
        raise _fail(
            "top anchor: signed legacy manifest requires validator-resident "
            "authenticated LM-head weights"
        )
    if (
        lm_head_mode == 1
        and lm_head_int8 is not None
        and (not compact_terminal or terminal_escalation)
    ):
        from verallm.proof_v3.economic_lm_head_binding import (
            verify_full_vocab_lm_head_recompute_v3,
        )

        verify_full_vocab_lm_head_recompute_v3(
            int8_lm_head=lm_head_int8,
            hidden_row_int8=final_row,
            committed_logits=full_logits,
        )
    if top_k > 1 and not (
        (lm_head_mode == 1 and lm_head_int8 is not None)
        or (
            lm_head_mode == LM_HEAD_CATALOG_BINDING_V3
            and lm_head_catalog is not None
        )
    ):
        raise _fail(
            "top anchor: signed quantization-stable top-k requires the "
            "mandatory authenticated full-vocabulary lm_head binding"
        )
    if compact_terminal and not terminal_escalation:
        candidate_set = set(candidate_tokens)
        worst_candidate = max(
            candidate_tokens,
            key=lambda row: (-committed_logits[row], row),
        )
        worst_key = (
            -committed_logits[worst_candidate],
            worst_candidate,
        )
        for row in sampled_vocab:
            if row in candidate_set:
                continue
            if (-committed_logits[row], row) < worst_key:
                raise _fail(
                    "sampled terminal row outranks the claimed top-k set"
                )
    else:
        candidates = quantization_stable_argmax_candidates_v3(
            full_logits,
            top_k=top_k,
        )
        if compact_terminal and tuple(sorted(candidates)) != candidate_tokens:
            raise _fail(
                "terminal escalation candidate set disagrees with the exact "
                "full-vocabulary ranking"
            )
        if observed_token not in candidates:
            raise _fail(
                "top anchor: validator-observed token is outside the signed "
                f"FULL-VOCABULARY quantization-stable argmax top-{top_k} of "
                "the authenticated logits"
            )
    _profile_mark("lm-head-and-final")
