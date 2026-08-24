"""Validator-derived challenge selection for economic_recompute_v3.

Every audited coordinate -- which layers, which token rows, which projection
output cells, which prompt positions, which decode positions, which vocab
rows -- is derived SOLELY from the nonce-bound transcript digest, the exact
signed profile, and the frozen commitment envelope.  The miner supplies no
selection: reveals that do not match these derived coordinates are rejected
by the adapter.  Because the transcript digest folds the hidden validator
nonce and the envelope (which itself folds the exact pre-nonce oracle
inventory through ``execution_root``), no coordinate is predictable while
the miner can still choose what to commit.

The selection shape is pinned by the signed profile's audit policy
(``selected_layer_count`` layers, ``transition_query_rows`` token rows) plus
the fixed constants of the versioned economic selection ABI below.  A
profile signed for a different selection ABI is rejected.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3

ECONOMIC_SELECTION_ABI_V3 = "economic.recompute.selection.v2.bounded_pool"
ECONOMIC_STREAMING_SELECTION_ABI_V3 = (
    "economic.recompute.selection.v7.streaming_stratified_transition_lanes"
)
ECONOMIC_COMPACT_SELECTION_ABI_V3 = (
    "economic.recompute.selection.v9.compact_projection_terminal_escalation"
)
ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3 = (
    "economic.recompute.selection.v10.compact_projection_terminal"
)

# The optional escalation ABI audits a nonce-selected succinct complete-output
# projection relation and a sampled terminal top-k claim on nine out of ten
# hard requests. It unpredictably escalates the remaining requests to both the
# complete projection-row reveal and an exact full-vocabulary terminal check.
# The compact-only ABI never selects that full-row mode. This is signed protocol
# data: changing either policy requires a new selection ABI and profile.
FULL_ROW_PROJECTION_ESCALATION_BPS_V3 = 1_000
# Compact hard proofs trade a bounded FRI proximity margin for materially
# smaller wire; the nonce-selected complete-row escalation retains the shipped
# 16-query posture. Both counts are validator-owned protocol constants.
ECONOMIC_COMPACT_PCS_QUERY_COUNT_V3 = 8
ECONOMIC_ESCALATION_PCS_QUERY_COUNT_V3 = 16

# Fixed sample widths and corridor coefficients of the economic selection ABI.
# Changing any of these is a new selection ABI (and therefore a new signed
# profile).
PROMPT_CANDIDATE_ROWS_V3 = 32
DECODE_CANDIDATE_ROWS_V3 = 32
OUT_SAMPLES_PER_PROJECTION_V3 = 16
VOCAB_SAMPLES_V3 = 16
AUDITED_DECODE_POSITIONS_V3 = 1
BOTTOM_ANCHOR_ROWS_V3 = 2
KV_COL_SAMPLES_V3 = 8
MLP_CELL_SAMPLES_V3 = 16
NORM_COL_SAMPLES_V3 = 16
GDN_VALUE_HEAD_SAMPLES_V3 = 7
ANCHOR_LANE_ELEMENTS_V3 = 1024
OUTPUT_LANE_SAMPLES_V3 = 2
RESIDUAL_COL_SAMPLES_V3 = 16
# The per-layer projections the economic proof audits, in wire order:
# (oracle x id suffix, oracle s id suffix, manifest suffix)
FULL_ATTENTION_AUDITED_PROJECTIONS_V3 = (
    ("attn_o_x", "attn_o_s", "o"),
    ("down_x", "down_s", "down"),
    ("gate_up_x", "gate_up_s", "gate_up"),
    ("qkv_x", "qkv_s", "qkv"),
)
GDN_AUDITED_PROJECTIONS_V3 = (
    ("down_x", "down_s", "down"),
    ("gate_up_x", "gate_up_s", "gate_up"),
    ("gdn_ba_x", "gdn_ba_s", "gdn_ba"),
    ("gdn_o_x", "gdn_o_s", "gdn_o"),
    ("gdn_qkvz_x", "gdn_qkvz_s", "gdn_qkvz"),
)
# Compatibility name for the already-shipped full-attention wire/prover
# call sites. New architecture-aware code must use the helper below.
AUDITED_PROJECTIONS_V3 = FULL_ATTENTION_AUDITED_PROJECTIONS_V3
# Corridor bound coefficients (numerator/denominator of a rational
# multiplier): |dequant(surrogate) - dequant(captured)| <=
#   QUANT_COEFF * (input half-step propagated through |W| + output
#   half-step) + REL_COEFF * (absolute-magnitude accumulation term).
# QUANT covers int8 rounding of the projection input/output; REL covers
# bf16/fp16 runtime accumulation error vs the exact real-valued product.
CORRIDOR_QUANT_COEFF_NUM_V3 = 3
CORRIDOR_QUANT_COEFF_DEN_V3 = 2
CORRIDOR_REL_COEFF_NUM_V3 = 1
CORRIDOR_REL_COEFF_DEN_V3 = 64

_SEED_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_SELECTION/SEED/SHA256"
_DRAW_DOMAIN = b"VERATHOS/PROOF_V3/ECONOMIC_SELECTION/DRAW/SHA256"

__all__ = [
    "AUDITED_DECODE_POSITIONS_V3",
    "AUDITED_PROJECTIONS_V3",
    "ANCHOR_LANE_ELEMENTS_V3",
    "FULL_ATTENTION_AUDITED_PROJECTIONS_V3",
    "GDN_VALUE_HEAD_SAMPLES_V3",
    "GDN_AUDITED_PROJECTIONS_V3",
    "BOTTOM_ANCHOR_ROWS_V3",
    "CORRIDOR_QUANT_COEFF_DEN_V3",
    "CORRIDOR_QUANT_COEFF_NUM_V3",
    "CORRIDOR_REL_COEFF_DEN_V3",
    "CORRIDOR_REL_COEFF_NUM_V3",
    "ECONOMIC_SELECTION_ABI_V3",
    "ECONOMIC_STREAMING_SELECTION_ABI_V3",
    "ECONOMIC_COMPACT_SELECTION_ABI_V3",
    "ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3",
    "ECONOMIC_COMPACT_PCS_QUERY_COUNT_V3",
    "ECONOMIC_ESCALATION_PCS_QUERY_COUNT_V3",
    "economic_selection_is_compact_v3",
    "FULL_ROW_PROJECTION_ESCALATION_BPS_V3",
    "KV_COL_SAMPLES_V3",
    "MLP_CELL_SAMPLES_V3",
    "NORM_COL_SAMPLES_V3",
    "OUT_SAMPLES_PER_PROJECTION_V3",
    "OUTPUT_LANE_SAMPLES_V3",
    "PROMPT_CANDIDATE_ROWS_V3",
    "RESIDUAL_COL_SAMPLES_V3",
    "DECODE_CANDIDATE_ROWS_V3",
    "VOCAB_SAMPLES_V3",
    "EconomicChallengeV3",
    "economic_attention_candidate_positions_v3",
    "economic_candidate_positions_v3",
    "economic_streaming_attention_candidate_positions_v3",
    "economic_streaming_candidate_positions_v3",
    "economic_selection_is_streaming_v3",
    "derive_economic_challenge_v3",
    "audited_projections_for_layer_kind_v3",
]


def economic_selection_is_compact_v3(selection_abi_id: object) -> bool:
    """Whether an exact signed selection ABI uses compact PCS openings."""

    return selection_abi_id in {
        ECONOMIC_COMPACT_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    }


def economic_selection_is_streaming_v3(selection_abi_id: str) -> bool:
    """Return whether an economic ABI uses the production streaming pool."""

    return selection_abi_id in {
        ECONOMIC_STREAMING_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    }


def audited_projections_for_layer_kind_v3(
    transition_kind: str,
) -> tuple[tuple[str, str, str], ...]:
    if transition_kind == "full_attention":
        return FULL_ATTENTION_AUDITED_PROJECTIONS_V3
    if transition_kind == "gdn":
        return GDN_AUDITED_PROJECTIONS_V3
    raise ProofV3Error("economic transition kind is not supported")


class _Sampler:
    """Unbiased deterministic rejection sampler over one labeled domain."""

    def __init__(self, seed: bytes, label: bytes) -> None:
        self._seed = seed
        self._label = label
        self._counter = 0

    def _word(self) -> int:
        if self._counter >= 1 << 64:
            raise ProofV3Error("economic selection counter overflowed")
        digest = hashlib.sha256(
            _DRAW_DOMAIN
            + self._seed
            + struct.pack("<H", len(self._label))
            + self._label
            + struct.pack("<Q", self._counter)
        ).digest()
        self._counter += 1
        return int.from_bytes(digest, "big")

    def below(self, bound: int) -> int:
        if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
            raise ProofV3Error("economic selection bound must be positive")
        ceiling = (1 << 256) - ((1 << 256) % bound)
        while True:
            candidate = self._word()
            if candidate < ceiling:
                return candidate % bound

    def distinct(self, population: int, count: int) -> tuple[int, ...]:
        if count > population:
            count = population
        remaining = list(range(population))
        chosen: list[int] = []
        for _ in range(count):
            chosen.append(remaining.pop(self.below(len(remaining))))
        return tuple(sorted(chosen))


def economic_attention_candidate_positions_v3(
    *, context_token_count: int
) -> tuple[int, ...]:
    """Canonical prompt-row pool shared with the attention audit.

    The pool is fixed before the validator nonce, bounded independently of
    context length, and derived from the authenticated request dimensions.
    Its entries are absolute positions in the forwarded prompt.
    """

    if (
        isinstance(context_token_count, bool)
        or not isinstance(context_token_count, int)
        or not 0 < context_token_count < 1 << 32
    ):
        raise ProofV3Error("economic context token count is invalid")
    start = max(0, context_token_count - PROMPT_CANDIDATE_ROWS_V3)
    return tuple(range(start, context_token_count))


def economic_candidate_positions_v3(
    *, context_token_count: int, decode_token_count: int
) -> tuple[int, ...]:
    """Canonical bounded rows retained by the transition proof.

    Prompt rows are exactly the attention candidate pool.  A second bounded
    tail covers generated tokens that entered a later forward pass, allowing
    the final-hidden/LM-head/token anchor to connect back to a real residual
    row without retaining the full sequence.
    """

    prompt = economic_attention_candidate_positions_v3(
        context_token_count=context_token_count
    )
    if (
        isinstance(decode_token_count, bool)
        or not isinstance(decode_token_count, int)
        or not 0 < decode_token_count < 1 << 32
    ):
        raise ProofV3Error("economic decode token count is invalid")
    # Output token p is produced by sequence row context-1+p.  Token zero is
    # therefore already covered by the final prompt row.  Only the following
    # decode-1 rows need a separate bounded tail.
    sequence_token_count = context_token_count + decode_token_count - 1
    decode_start = max(
        context_token_count,
        sequence_token_count - DECODE_CANDIDATE_ROWS_V3,
    )
    return prompt + tuple(range(decode_start, sequence_token_count))


def economic_streaming_attention_candidate_positions_v3(
    *, selection_seed: bytes, context_token_count: int
) -> tuple[int, ...]:
    """Post-commitment prompt positions for the streaming execution anchor.

    Unlike the legacy tail pool, these absolute positions cover the complete
    prompt and are unpredictable until the validator nonce is revealed.  The
    miner therefore cannot prepare registered-model witnesses only for a
    known fixed tail before freezing the full-sequence roots.
    """

    if not isinstance(selection_seed, bytes) or len(selection_seed) != 32:
        raise ProofV3Error("economic selection seed must be exactly 32 bytes")
    if (
        isinstance(context_token_count, bool)
        or not isinstance(context_token_count, int)
        or not 0 < context_token_count < 1 << 32
    ):
        raise ProofV3Error("economic context token count is invalid")
    # The final prompt row produces output token zero and is therefore
    # mandatory for the final-hidden -> LM-head -> returned-token anchor.
    # Sample the remaining rows post-nonce across the complete preceding
    # prompt.  This keeps the pool bounded and unpredictable without leaving
    # one-token responses with no auditable output producer.
    if context_token_count == 1:
        return (0,)
    sampled = _Sampler(
        selection_seed, b"streaming-prompt-pool-v5"
    ).distinct(
        context_token_count - 1,
        PROMPT_CANDIDATE_ROWS_V3 - 1,
    )
    return tuple(sorted(sampled + (context_token_count - 1,)))


def economic_streaming_candidate_positions_v3(
    *,
    selection_seed: bytes,
    context_token_count: int,
    decode_token_count: int,
) -> tuple[int, ...]:
    """Bounded post-nonce rows over full prompt plus forwarded decode tail."""

    prompt = economic_streaming_attention_candidate_positions_v3(
        selection_seed=selection_seed,
        context_token_count=context_token_count,
    )
    if (
        isinstance(decode_token_count, bool)
        or not isinstance(decode_token_count, int)
        or not 0 < decode_token_count < 1 << 32
    ):
        raise ProofV3Error("economic decode token count is invalid")
    sequence_token_count = context_token_count + decode_token_count - 1
    decode_start = max(
        context_token_count,
        sequence_token_count - DECODE_CANDIDATE_ROWS_V3,
    )
    return prompt + tuple(range(decode_start, sequence_token_count))


@dataclass(frozen=True, slots=True)
class EconomicChallengeV3:
    """The complete validator-derived economic audit selection."""

    selection_abi_id: str
    selection_seed: bytes
    sequence_token_count: int
    context_token_count: int
    decode_token_count: int
    candidate_sequence_positions: tuple[int, ...]
    attention_candidate_positions: tuple[int, ...]
    selected_layer_indices: tuple[int, ...]
    # Merkle row slots in candidate_sequence_positions, never absolute token
    # positions.  This distinction is part of the v2 bounded-pool ABI.
    sampled_token_rows: tuple[int, ...]
    bottom_anchor_rows: tuple[int, ...]
    bottom_anchor_positions: tuple[int, ...]
    audited_decode_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.selection_abi_id not in (
            ECONOMIC_SELECTION_ABI_V3,
            ECONOMIC_STREAMING_SELECTION_ABI_V3,
            ECONOMIC_COMPACT_SELECTION_ABI_V3,
            ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
        ):
            raise ProofV3Error("economic selection ABI is not supported")
        if (
            not isinstance(self.selection_seed, bytes)
            or len(self.selection_seed) != 32
        ):
            raise ProofV3Error("economic selection seed must be exactly 32 bytes")
        expected_sequence_count = (
            self.context_token_count + self.decode_token_count - 1
        )
        if self.sequence_token_count != expected_sequence_count:
            raise ProofV3Error(
                "economic sequence token count is inconsistent"
            )
        if not economic_selection_is_streaming_v3(self.selection_abi_id):
            expected_candidates = economic_candidate_positions_v3(
                context_token_count=self.context_token_count,
                decode_token_count=self.decode_token_count,
            )
            expected_attention = economic_attention_candidate_positions_v3(
                context_token_count=self.context_token_count
            )
        else:
            expected_candidates = economic_streaming_candidate_positions_v3(
                selection_seed=self.selection_seed,
                context_token_count=self.context_token_count,
                decode_token_count=self.decode_token_count,
            )
            expected_attention = (
                economic_streaming_attention_candidate_positions_v3(
                    selection_seed=self.selection_seed,
                    context_token_count=self.context_token_count,
                )
            )
        if tuple(self.candidate_sequence_positions) != expected_candidates:
            raise ProofV3Error(
                "economic candidate positions are not canonical"
            )
        if tuple(self.attention_candidate_positions) != expected_attention:
            raise ProofV3Error(
                "economic attention candidate positions are not canonical"
            )
        rows = tuple(self.sampled_token_rows)
        if (
            tuple(sorted(set(rows))) != rows
            or any(not 0 <= row < len(expected_candidates) for row in rows)
        ):
            raise ProofV3Error("economic sampled pool rows are malformed")
        bottom_rows = tuple(self.bottom_anchor_rows)
        if (
            tuple(sorted(set(bottom_rows))) != bottom_rows
            or any(not 0 <= row < len(expected_attention) for row in bottom_rows)
        ):
            raise ProofV3Error("economic bottom-anchor pool rows are malformed")
        if tuple(self.bottom_anchor_positions) != tuple(
            expected_candidates[row] for row in bottom_rows
        ):
            raise ProofV3Error(
                "economic bottom-anchor rows and positions disagree"
            )
        available_decode_positions = self.candidate_decode_positions
        decode_positions = tuple(self.audited_decode_positions)
        if (
            tuple(sorted(set(decode_positions))) != decode_positions
            or any(position not in available_decode_positions
                   for position in decode_positions)
        ):
            raise ProofV3Error(
                "economic audited decode positions are outside the pool"
            )

    @property
    def candidate_row_count(self) -> int:
        return len(self.candidate_sequence_positions)

    @property
    def full_row_projection_audit(self) -> bool:
        """Whether this nonce selected the complete-row escalation.

        The v7 streaming ABI remains complete-row-only.  The v9 compact ABI
        draws an unbiased basis-point value from the post-commitment seed.
        The legacy bounded-pool ABI keeps its original sampled-cell behavior.
        """

        if self.selection_abi_id == ECONOMIC_STREAMING_SELECTION_ABI_V3:
            return True
        if self.selection_abi_id == ECONOMIC_COMPACT_SELECTION_ABI_V3:
            return (
                _Sampler(
                    self.selection_seed,
                    b"projection-audit-mode/v1",
                ).below(10_000)
                < FULL_ROW_PROJECTION_ESCALATION_BPS_V3
            )
        if self.selection_abi_id == ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3:
            return False
        return False

    @property
    def pcs_query_count(self) -> int:
        """Exact dynamic-PCS query budget selected after commitment."""

        if (
            economic_selection_is_compact_v3(self.selection_abi_id)
            and not self.full_row_projection_audit
        ):
            return ECONOMIC_COMPACT_PCS_QUERY_COUNT_V3
        return ECONOMIC_ESCALATION_PCS_QUERY_COUNT_V3

    @property
    def sampled_sequence_positions(self) -> tuple[int, ...]:
        return tuple(
            self.candidate_sequence_positions[row]
            for row in self.sampled_token_rows
        )

    @property
    def candidate_decode_positions(self) -> tuple[int, ...]:
        slots = {
            position: row
            for row, position in enumerate(self.candidate_sequence_positions)
        }
        return tuple(
            output_position
            for output_position in range(self.decode_token_count)
            if self.context_token_count - 1 + output_position in slots
        )

    def pool_row_for_sequence_position(self, position: int) -> int:
        try:
            return self.candidate_sequence_positions.index(int(position))
        except ValueError as exc:
            raise ProofV3Error(
                "sequence position is outside the economic candidate pool"
            ) from exc

    def pool_row_for_decode_position(self, position: int) -> int:
        if int(position) not in self.candidate_decode_positions:
            raise ProofV3Error(
                "decode position is outside the economic candidate pool"
            )
        return self.pool_row_for_sequence_position(
            self.context_token_count - 1 + int(position)
        )


    def out_cells_for(
        self,
        *,
        layer_index: int,
        out_dim: int,
        projection: str = "qkv",
    ) -> tuple[int, ...]:
        """Sampled output columns for one selected projection.

        ``out_dim`` comes from the pre-nonce frozen oracle inventory (bound
        into the envelope's execution root), never from a post-nonce claim.
        The draw is domain-separated per projection name.
        """

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("projection layer was not selected")
        label = (
            b"proj-out/"
            + projection.encode("ascii")
            + b"/"
            + struct.pack("<I", layer_index)
        )
        return self._lane_coherent_cells(
            population=out_dim,
            count=OUT_SAMPLES_PER_PROJECTION_V3,
            label=label,
        )

    def projection_binding_columns_for(
        self,
        *,
        layer_index: int,
        layer_kind: str,
        projection: str,
        out_dim: int,
        kv_dim: int = 0,
        required_runtime_columns=(),
    ) -> tuple[int, ...]:
        """Exact nonce-selected outputs bound to registered projection rows.

        The base sample covers every projection. Full-attention QKV also
        includes the selected K/V cache coordinates, and architecture adapters
        may add validator-derived runtime columns (selected query/gate heads or
        GDN recurrence lanes). The miner never supplies this set.
        """

        columns = set(
            self.out_cells_for(
                layer_index=layer_index,
                out_dim=out_dim,
                projection=projection,
            )
        )
        try:
            runtime_columns = tuple(
                int(column) for column in required_runtime_columns
            )
        except (TypeError, ValueError) as exc:
            raise ProofV3Error(
                "projection runtime columns are malformed"
            ) from exc
        if (
            runtime_columns != tuple(sorted(set(runtime_columns)))
            or any(column < 0 or column >= out_dim for column in runtime_columns)
        ):
            raise ProofV3Error(
                "projection runtime columns are malformed"
            )
        columns.update(runtime_columns)
        if layer_kind == "full_attention" and projection == "qkv":
            if (
                isinstance(kv_dim, bool)
                or not isinstance(kv_dim, int)
                or kv_dim <= 0
                or out_dim <= 2 * kv_dim
            ):
                raise ProofV3Error(
                    "projection QKV binding geometry is malformed"
                )
            q_width = out_dim - 2 * kv_dim
            kv_columns = self.kv_cols_for(
                layer_index=layer_index,
                kv_dim=kv_dim,
            )
            columns.update(q_width + column for column in kv_columns)
            columns.update(
                q_width + kv_dim + column for column in kv_columns
            )
        elif layer_kind not in ("full_attention", "gdn"):
            raise ProofV3Error(
                "projection binding layer kind is unsupported"
            )
        return tuple(sorted(columns))

    def kv_cols_for(self, *, layer_index: int, kv_dim: int) -> tuple[int, ...]:
        """Sampled K/V slice columns for the qkv<->cache corridor."""

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("kv corridor layer was not selected")
        sampler = _Sampler(
            self.selection_seed, b"kv-cols/" + struct.pack("<I", layer_index)
        )
        return sampler.distinct(kv_dim, KV_COL_SAMPLES_V3)

    def mlp_cols_for(self, *, layer_index: int, inter_dim: int) -> tuple[int, ...]:
        """Sampled intermediate columns for the elementwise MLP check."""

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("mlp cell layer was not selected")
        return self._lane_coherent_cells(
            population=inter_dim,
            count=MLP_CELL_SAMPLES_V3,
            label=b"mlp-cols/" + struct.pack("<I", layer_index),
        )

    def residual_cols_for(
        self, *, layer_index: int, hidden_dim: int
    ) -> tuple[int, ...]:
        """Nonce-selected hidden cells for both residual compositions."""

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("residual transition layer was not selected")
        return self._lane_coherent_cells(
            population=hidden_dim,
            count=RESIDUAL_COL_SAMPLES_V3,
            label=b"residual-cols/" + struct.pack("<I", layer_index),
        )

    def moe_token_position_for(
        self,
        *,
        layer_index: int,
        minimum_position: int = 0,
    ) -> int:
        """Select one executed absolute row for bounded sparse-MoE replay.

        ``minimum_position`` excludes rows served entirely from an
        authenticated prefix cache.  The default preserves the original
        no-cache selection ABI exactly.
        """

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("MoE layer was not selected")
        if (
            isinstance(minimum_position, bool)
            or not isinstance(minimum_position, int)
            or minimum_position < 0
            or minimum_position >= self.sequence_token_count
        ):
            raise ProofV3Error("MoE minimum sequence position is malformed")
        candidates = tuple(
            position
            for position in self.candidate_sequence_positions
            if position >= minimum_position
        )
        if not candidates:
            raise ProofV3Error("MoE executed candidate sequence pool is empty")
        slot = _Sampler(
            self.selection_seed,
            b"moe-token/v1/" + struct.pack("<I", layer_index),
        ).below(len(candidates))
        return candidates[slot]

    def _lane_coherent_cells(
        self,
        *,
        population: int,
        count: int,
        label: bytes,
    ) -> tuple[int, ...]:
        """Sample bounded cells from at most two authenticated 2-KiB lanes.

        The lane choices remain post-commitment and transcript-derived.  The
        bounded grouping changes only transport geometry: a selected lane is
        opened once and supplies several verifier-selected cells.
        """

        if (
            isinstance(population, bool)
            or not isinstance(population, int)
            or population <= 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(label, bytes)
            or not label
        ):
            raise ProofV3Error("lane-coherent selection geometry is malformed")
        target = min(population, count)
        lane_count = (
            population + ANCHOR_LANE_ELEMENTS_V3 - 1
        ) // ANCHOR_LANE_ELEMENTS_V3
        selected_lanes = _Sampler(
            self.selection_seed,
            label + b"/lanes",
        ).distinct(
            lane_count,
            min(OUTPUT_LANE_SAMPLES_V3, lane_count),
        )
        remaining = target
        selected: list[int] = []
        for slot, lane in enumerate(selected_lanes):
            lanes_left = len(selected_lanes) - slot
            take = (remaining + lanes_left - 1) // lanes_left
            start = lane * ANCHOR_LANE_ELEMENTS_V3
            width = min(
                ANCHOR_LANE_ELEMENTS_V3,
                population - start,
            )
            offsets = _Sampler(
                self.selection_seed,
                label + b"/cells/" + struct.pack("<I", lane),
            ).distinct(width, min(take, width))
            selected.extend(start + offset for offset in offsets)
            remaining -= len(offsets)
        if remaining:
            # This is reachable only when the final short lane cannot supply
            # its even share. Fill from the complete selected-lane union with
            # a separate draw while preserving distinctness.
            selected_set = set(selected)
            available = tuple(
                cell
                for lane in selected_lanes
                for cell in range(
                    lane * ANCHOR_LANE_ELEMENTS_V3,
                    min(
                        population,
                        (lane + 1) * ANCHOR_LANE_ELEMENTS_V3,
                    ),
                )
                if cell not in selected_set
            )
            fill = _Sampler(
                self.selection_seed,
                label + b"/fill",
            ).distinct(len(available), remaining)
            selected.extend(available[index] for index in fill)
        return tuple(sorted(selected))

    def norm_cols_for(
        self, *, layer_index: int, hidden_dim: int, which: str
    ) -> tuple[int, ...]:
        """Sampled hidden columns for one RMSNorm link check."""

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("norm link layer was not selected")
        if which not in ("input", "post"):
            raise ProofV3Error("norm link selector is invalid")
        label = (
            b"norm-cols/"
            + which.encode("ascii")
            + b"/"
            + struct.pack("<I", layer_index)
        )
        sampler = _Sampler(self.selection_seed, label)
        return sampler.distinct(hidden_dim, NORM_COL_SAMPLES_V3)

    def gdn_value_heads_for(
        self,
        *,
        layer_index: int,
        num_key_heads: int,
        num_value_heads: int,
    ) -> tuple[int, ...]:
        """Select value heads across distinct GDN key-head groups.

        Qwen GDN maps a fixed-size group of value heads to each key head.
        Selecting key-head groups first avoids spending both samples on the
        same Q/K recurrence.  The choice is derived only after the pre-nonce
        execution-anchor roots are frozen.
        """

        if layer_index not in self.selected_layer_indices:
            raise ProofV3Error("GDN head layer was not selected")
        if (
            type(num_key_heads) is not int
            or type(num_value_heads) is not int
            or num_key_heads <= 0
            or num_value_heads <= 0
            or num_value_heads % num_key_heads
        ):
            raise ProofV3Error("GDN head geometry is malformed")
        group_size = num_value_heads // num_key_heads
        selected_key_heads = _Sampler(
            self.selection_seed,
            b"gdn-key-heads/" + struct.pack("<I", layer_index),
        ).distinct(num_key_heads, GDN_VALUE_HEAD_SAMPLES_V3)
        value_heads = []
        for key_head in selected_key_heads:
            offset = _Sampler(
                self.selection_seed,
                b"gdn-value-head/"
                + struct.pack("<II", layer_index, key_head),
            ).below(group_size)
            value_heads.append(key_head * group_size + offset)
        return tuple(sorted(value_heads))

    def vocab_rows_for(self, *, vocab_size: int) -> tuple[int, ...]:
        """Sampled vocab rows for the top-anchor logit audit."""

        sampler = _Sampler(self.selection_seed, b"vocab")
        return sampler.distinct(vocab_size, VOCAB_SAMPLES_V3)


def derive_economic_challenge_v3(
    *,
    transcript_digest: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> EconomicChallengeV3:
    """Derive the full economic selection from the nonce-bound transcript.

    ``transcript_digest`` is the Stack-A execution transcript root (hidden
    validator nonce + exact profile + complete envelope) -- available only
    after the envelope is frozen, so no committed oracle can adapt to it.
    """

    if not isinstance(transcript_digest, bytes) or len(transcript_digest) != 32:
        raise ProofV3Error("economic transcript_digest must be exactly 32 bytes")
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("envelope has an unexpected type")
    if envelope.execution_profile_digest != profile.digest():
        raise ProofV3Error("economic selection envelope has an unexpected profile")
    policy = profile.relation_spec.audit_policy
    if policy.selection_abi_id not in (
        ECONOMIC_SELECTION_ABI_V3,
        ECONOMIC_STREAMING_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_SELECTION_ABI_V3,
        ECONOMIC_COMPACT_ONLY_SELECTION_ABI_V3,
    ):
        raise ProofV3Error(
            "signed profile does not carry the economic selection ABI"
        )
    # FORWARDED rows: the final generated token is emitted by the last
    # forward pass but never enters one itself, so runtime captures hold
    # context + decode - 1 sequence rows (context alone when decode == 0).
    sequence_token_count = envelope.context_token_count + (
        envelope.decode_token_count - 1 if envelope.decode_token_count else 0
    )
    if not 0 < sequence_token_count < 1 << 32:
        raise ProofV3Error("economic sequence token count is invalid")
    layer_universe = tuple(
        plan.layer_index for plan in profile.relation_spec.layer_audits
    )
    if len(set(layer_universe)) != len(layer_universe) or not layer_universe:
        raise ProofV3Error("economic layer universe is malformed")
    if policy.selected_layer_count > len(layer_universe):
        raise ProofV3Error("signed layer count exceeds the relation graph")

    seed = hashlib.sha256(
        _SEED_DOMAIN
        + transcript_digest
        + profile.digest()
        + envelope.digest()
        + struct.pack(
            "<III",
            policy.selected_layer_count,
            policy.transition_query_rows,
            sequence_token_count,
        )
    ).digest()

    ordered_universe = tuple(sorted(layer_universe))
    if economic_selection_is_streaming_v3(policy.selection_abi_id):
        plans = {
            int(plan.layer_index): plan
            for plan in profile.relation_spec.layer_audits
        }
        full_attention = tuple(
            layer
            for layer in ordered_universe
            if plans[layer].is_full_attention
        )
        gdn = tuple(
            layer
            for layer in ordered_universe
            if not plans[layer].is_full_attention
        )
        if (
            policy.minimum_full_attention_layers > len(full_attention)
            or policy.minimum_gdn_layers > len(gdn)
        ):
            raise ProofV3Error(
                "signed stratified layer minimum exceeds the qualified "
                "architecture inventory"
            )

        def _sample_layers(population, count, label):
            slots = _Sampler(seed, label).distinct(len(population), count)
            return tuple(population[slot] for slot in slots)

        required = (
            _sample_layers(
                full_attention,
                policy.minimum_full_attention_layers,
                b"layers/full-attention",
            )
            + _sample_layers(
                gdn,
                policy.minimum_gdn_layers,
                b"layers/gdn",
            )
        )
        remaining = tuple(
            layer for layer in ordered_universe if layer not in set(required)
        )
        fill_count = policy.selected_layer_count - len(required)
        if fill_count > len(remaining):
            raise ProofV3Error(
                "signed stratified layer count exceeds the remaining "
                "architecture inventory"
            )
        selected_layers = tuple(
            sorted(
                required
                + _sample_layers(remaining, fill_count, b"layers/fill")
            )
        )
    else:
        layer_positions = _Sampler(seed, b"layers").distinct(
            len(ordered_universe), policy.selected_layer_count
        )
        selected_layers = tuple(
            ordered_universe[i] for i in layer_positions
        )
    if policy.selection_abi_id == ECONOMIC_SELECTION_ABI_V3:
        attention_candidates = economic_attention_candidate_positions_v3(
            context_token_count=envelope.context_token_count
        )
        candidates = economic_candidate_positions_v3(
            context_token_count=envelope.context_token_count,
            decode_token_count=envelope.decode_token_count,
        )
    else:
        attention_candidates = (
            economic_streaming_attention_candidate_positions_v3(
                selection_seed=seed,
                context_token_count=envelope.context_token_count,
            )
        )
        candidates = economic_streaming_candidate_positions_v3(
            selection_seed=seed,
            context_token_count=envelope.context_token_count,
            decode_token_count=envelope.decode_token_count,
        )
    token_rows = _Sampler(seed, b"token-rows").distinct(
        len(candidates), policy.transition_query_rows
    )
    bottom_rows = _Sampler(seed, b"bottom").distinct(
        len(attention_candidates), BOTTOM_ANCHOR_ROWS_V3
    )
    bottom_positions = tuple(candidates[row] for row in bottom_rows)
    available_decode_positions = tuple(
        position
        for position in range(envelope.decode_token_count)
        if envelope.context_token_count - 1 + position in candidates
    )
    decode_slots = _Sampler(seed, b"decode-pos").distinct(
        len(available_decode_positions), AUDITED_DECODE_POSITIONS_V3
    )
    decode_positions = tuple(
        available_decode_positions[slot] for slot in decode_slots
    )
    return EconomicChallengeV3(
        selection_abi_id=policy.selection_abi_id,
        selection_seed=seed,
        sequence_token_count=sequence_token_count,
        context_token_count=envelope.context_token_count,
        decode_token_count=envelope.decode_token_count,
        candidate_sequence_positions=candidates,
        attention_candidate_positions=attention_candidates,
        selected_layer_indices=selected_layers,
        sampled_token_rows=token_rows,
        bottom_anchor_rows=bottom_rows,
        bottom_anchor_positions=bottom_positions,
        audited_decode_positions=decode_positions,
    )
