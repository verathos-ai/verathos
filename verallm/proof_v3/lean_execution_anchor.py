"""Lean pre-nonce execution-anchor inventory for proof-v3.

The full streaming inventory commits every intermediate activation on every
request.  That is sound, but its cost scales with the sum of all intermediate
tensor widths.  The lean inventory instead freezes only values that cannot be
reconstructed by a selected hard transition proof:

* one residual checkpoint after every bounded layer corridor;
* the fused K/V portion of every full-attention QKV output; and
* the prompt-boundary state of every GDN layer.

After the validator nonce, a selected transition proof must reconstruct all
omitted intermediates from these boundaries and the authenticated static
weights.  This module only defines the exact pre-nonce inventory and the exact
nonce-selected rows to open.  It deliberately does not treat omitted
intermediates as authenticated until the transition verifier accepts them.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Final

from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.errors import ProofV3Error

LEAN_EXECUTION_ANCHOR_ABI_V3: Final = (
    "execution_anchor.per_layer_kv.v5"
)
LEAN_EXECUTION_CHECKPOINT_STRIDE_V3: Final = 1
LEAN_KV_PROJECTION_ROWS_PER_LAYER_V3: Final = 16

_KV_ROW_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LEAN_EXECUTION_ANCHOR/V2/KV_LAYER_ROWS/SHA256"
)

__all__ = [
    "LEAN_EXECUTION_ANCHOR_ABI_V3",
    "LEAN_EXECUTION_CHECKPOINT_STRIDE_V3",
    "LEAN_KV_PROJECTION_ROWS_PER_LAYER_V3",
    "derive_lean_kv_projection_rows_v3",
    "expected_lean_execution_anchor_inventory_v3",
    "expected_lean_execution_anchor_reveals_v3",
    "lean_execution_checkpoint_layers_v3",
    "lean_bottom_sequence_positions_v3",
    "lean_projection_row_layouts_v3",
    "lean_qkv_projection_rows_v3",
    "lean_shared_boundary_rows_v3",
    "lean_selected_corridor_layers_v3",
    "lean_transition_positions_v3",
]


def lean_bottom_sequence_positions_v3(
    *,
    rows,
    candidate_sequence_positions,
    lean_positions,
) -> tuple[int, ...]:
    """Map compact bottom rows to authenticated absolute sequence positions."""

    positions = []
    candidates = tuple(candidate_sequence_positions)
    lean = dict(lean_positions)
    for raw_row in rows:
        row = int(raw_row)
        if row in lean:
            positions.append(int(lean[row]))
            continue
        if not 0 <= row < len(candidates):
            raise ProofV3Error(
                "chain bottom row has no authenticated sequence position"
            )
        positions.append(int(candidates[row]))
    return tuple(positions)


def _layers_and_kinds(
    layer_indices,
    layer_kinds: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    layers = tuple(int(layer) for layer in layer_indices)
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or any(layer < 0 or layer >= 1 << 32 for layer in layers)
        or not isinstance(layer_kinds, Mapping)
    ):
        raise ProofV3Error("lean execution-anchor layer inventory is malformed")
    kinds = {int(layer): str(kind) for layer, kind in layer_kinds.items()}
    if set(kinds) != set(layers) or any(
        kind not in {"full_attention", "gdn"} for kind in kinds.values()
    ):
        raise ProofV3Error(
            "lean execution-anchor layer kinds do not match the inventory"
        )
    return tuple((layer, kinds[layer]) for layer in layers)


def _positive_u32(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value >= 1 << 32
    ):
        raise ProofV3Error(f"{name} is invalid")
    return value


def lean_execution_checkpoint_layers_v3(
    layer_indices,
) -> tuple[int, ...]:
    """Return the exact residual endpoints committed before the nonce."""

    layers = tuple(int(layer) for layer in layer_indices)
    if (
        not layers
        or layers != tuple(range(len(layers)))
    ):
        raise ProofV3Error(
            "lean execution-anchor layers must be contiguous from zero"
        )
    return tuple(
        layer
        for layer in layers
        if (
            (layer + 1) % LEAN_EXECUTION_CHECKPOINT_STRIDE_V3 == 0
            or layer == layers[-1]
        )
    )


def lean_selected_corridor_layers_v3(
    *,
    selected_layer_indices,
    layer_indices,
) -> tuple[int, ...]:
    """Expand nonce-selected layers to complete checkpointed corridors."""

    layers = tuple(int(layer) for layer in layer_indices)
    checkpoints = lean_execution_checkpoint_layers_v3(layers)
    del checkpoints
    selected = tuple(sorted(int(layer) for layer in selected_layer_indices))
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(layer not in layers for layer in selected)
    ):
        raise ProofV3Error(
            "lean selected layers are outside the execution inventory"
        )
    audited: set[int] = set()
    stride = LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
    for layer in selected:
        position = layers.index(layer)
        start = (position // stride) * stride
        stop = min(start + stride, len(layers))
        audited.update(layers[start:stop])
    return tuple(sorted(audited))


def expected_lean_execution_anchor_inventory_v3(
    *,
    layer_indices,
    layer_kinds: Mapping[int, str],
    sequence_token_count: int,
    hidden_dim: int,
    projection_dims: Mapping[str, tuple[int, int]],
    attention_kv_widths: Mapping[int, int],
    gdn_runtime_semantics=None,
    context_token_count: int | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """Return the exact signed lean pre-nonce stage inventory.

    Rows are raw FP16/BF16 runtime rows and widths are encoded bytes.  Static
    GDN prompt boundaries retain their authenticated model-specific widths.
    """

    layers_and_kinds = _layers_and_kinds(layer_indices, layer_kinds)
    sequence = _positive_u32(
        sequence_token_count, "lean execution-anchor sequence length"
    )
    hidden = _positive_u32(hidden_dim, "lean execution-anchor hidden width")
    if not isinstance(projection_dims, Mapping):
        raise ProofV3Error("lean projection geometry is malformed")
    if not isinstance(attention_kv_widths, Mapping):
        raise ProofV3Error("lean K/V geometry is malformed")

    records: list[tuple[str, int, int]] = []
    checkpoints = set(lean_execution_checkpoint_layers_v3(
        layer for layer, _kind in layers_and_kinds
    ))
    for layer, kind in layers_and_kinds:
        if layer in checkpoints:
            records.append(
                (f"l{layer}.residual_out", sequence, hidden * 2)
            )
        if kind == "full_attention":
            dims = projection_dims.get(f"l{layer}.qkv")
            if (
                not isinstance(dims, tuple)
                or len(dims) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in dims
                )
            ):
                raise ProofV3Error(
                    f"lean execution-anchor geometry is missing l{layer}.qkv"
                )
            kv_width = attention_kv_widths.get(layer)
            if (
                isinstance(kv_width, bool)
                or not isinstance(kv_width, int)
                or kv_width <= 0
                or kv_width >= dims[1]
            ):
                raise ProofV3Error(
                    f"lean execution-anchor K/V geometry is missing layer {layer}"
                )
            records.append(
                (f"l{layer}.attention_kv_output", sequence, kv_width * 2)
            )
            continue

        if gdn_runtime_semantics is None:
            raise ProofV3Error(
                "lean GDN execution anchors require authenticated semantics"
            )
        try:
            semantics = gdn_runtime_semantics.layer_for(layer)
            conv_bytes = _positive_u32(
                semantics.conv_state_bytes,
                "lean GDN convolution-state width",
            )
            recurrent_bytes = _positive_u32(
                semantics.recurrent_state_bytes,
                "lean GDN recurrent-state width",
            )
        except (AttributeError, ProofV3Error) as exc:
            raise ProofV3Error(
                f"lean GDN execution-anchor semantics are missing layer {layer}"
            ) from exc
        records.extend(
            (
                (f"l{layer}.gdn_conv_prompt_boundary", 1, conv_bytes),
                (
                    f"l{layer}.gdn_recurrent_prompt_boundary",
                    1,
                    recurrent_bytes,
                ),
            )
        )
        checkpoint_stride = int(
            getattr(semantics, "decode_checkpoint_stride", 0)
        )
        if checkpoint_stride:
            from verallm.proof_v3.gdn_decode_corridor import (
                gdn_decode_checkpoint_offsets_v3,
            )

            context = _positive_u32(
                context_token_count,
                "lean execution-anchor context length",
            )
            if context > sequence:
                raise ProofV3Error(
                    "lean execution-anchor context exceeds the sequence"
                )
            checkpoint_count = len(
                gdn_decode_checkpoint_offsets_v3(
                    decode_token_count=sequence - context + 1,
                    checkpoint_stride=checkpoint_stride,
                )
            )
            records.extend(
                (
                    (
                        f"l{layer}.gdn_conv_decode_checkpoints",
                        checkpoint_count,
                        conv_bytes,
                    ),
                    (
                        f"l{layer}.gdn_recurrent_decode_checkpoints",
                        checkpoint_count,
                        recurrent_bytes,
                    ),
                )
            )
    return tuple(sorted(records))


def derive_lean_kv_projection_rows_v3(
    *,
    selection_seed: bytes,
    layer_index: int,
    query_position: int,
    count: int = LEAN_KV_PROJECTION_ROWS_PER_LAYER_V3,
) -> tuple[int, ...]:
    """Derive distinct prefix rows whose QKV projection must also be proven."""

    if not isinstance(selection_seed, bytes) or len(selection_seed) != 32:
        raise ProofV3Error("lean K/V selection seed must be 32 bytes")
    layer = _positive_u32(layer_index + 1, "lean K/V layer") - 1
    query = _positive_u32(query_position + 1, "lean K/V query") - 1
    samples = _positive_u32(count, "lean K/V sample count")
    population = query + 1
    samples = min(samples, population)
    chosen: set[int] = set()
    counter = 0
    while len(chosen) < samples:
        digest = hashlib.sha256(
            _KV_ROW_DOMAIN
            + selection_seed
            + struct.pack("<III", layer, query, counter)
        ).digest()
        for offset in range(0, len(digest), 4):
            candidate = (
                int.from_bytes(digest[offset : offset + 4], "little")
                % population
            )
            chosen.add(candidate)
            if len(chosen) == samples:
                break
        counter += 1
    return tuple(sorted(chosen))


def lean_qkv_projection_rows_v3(
    *,
    challenge: EconomicChallengeV3,
    kv_positions,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map authenticated absolute K/V rows into one QKV oracle row layout.

    Existing transition-pool positions retain their canonical pool slots.
    Full-prefix samples outside that bounded pool are appended in absolute
    order. The result is ``(projection_row_slots, appended_positions)`` and is
    derived identically by prover and verifier.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("lean QKV row challenge has a wrong type")
    positions = tuple(int(position) for position in kv_positions)
    if (
        not positions
        or positions != tuple(sorted(set(positions)))
        or any(
            position < 0 or position >= challenge.context_token_count
            for position in positions
        )
    ):
        raise ProofV3Error("lean QKV projection positions are malformed")
    pool_slot = {
        int(position): slot
        for slot, position in enumerate(
            challenge.candidate_sequence_positions
        )
    }
    appended = tuple(
        position for position in positions if position not in pool_slot
    )
    appended_slot = {
        position: challenge.candidate_row_count + index
        for index, position in enumerate(appended)
    }
    required_positions = set(
        int(challenge.candidate_sequence_positions[row])
        for row in challenge.sampled_token_rows
    )
    required_positions.update(positions)
    rows = tuple(
        sorted(
            (
                pool_slot[position]
                if position in pool_slot
                else appended_slot[position]
            )
            for position in required_positions
        )
    )
    return rows, appended


def lean_projection_row_layouts_v3(
    *,
    challenge: EconomicChallengeV3,
    layer_indices,
    attention_kv_positions_by_layer: Mapping[int, tuple[int, ...]],
    gdn_decode_only_layers=(),
    gdn_decode_positions_by_layer: Mapping[
        int, tuple[int, ...]
    ] | None = None,
    gdn_prefix_positions_by_layer: Mapping[
        int, tuple[int, ...]
    ] | None = None,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """Derive the exact post-nonce row layout for every audited corridor.

    The ordinary transition samples remain in their canonical candidate-pool
    slots.  Each full-attention K/V prefix sample is propagated through the
    complete checkpoint corridor containing that attention layer, so the
    selected QKV input is not a detached post-nonce witness.  Positions outside
    the bounded candidate pool are appended in absolute order independently
    for each corridor.  A v9 GDN checkpoint window is replayed in full, but
    only an independent nonce-derived row sample extends the projection
    layout. This keeps the registered-weight binding unpredictable without
    making proof cost linear in checkpoint stride. Only the selected-trace
    caller explicitly names layers whose layout is replaced by the complete
    bounded recurrence window.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("lean projection-row challenge has a wrong type")
    layers = tuple(int(layer) for layer in layer_indices)
    audited = lean_selected_corridor_layers_v3(
        selected_layer_indices=challenge.selected_layer_indices,
        layer_indices=layers,
    )
    if not isinstance(attention_kv_positions_by_layer, Mapping):
        raise ProofV3Error("lean attention K/V row map is malformed")
    gdn_layers = tuple(int(layer) for layer in gdn_decode_only_layers)
    gdn_positions = (
        {}
        if gdn_decode_positions_by_layer is None
        else {
            int(layer): tuple(int(position) for position in positions)
            for layer, positions in gdn_decode_positions_by_layer.items()
        }
    )
    gdn_prefix_positions = (
        {}
        if gdn_prefix_positions_by_layer is None
        else {
            int(layer): tuple(int(position) for position in positions)
            for layer, positions in gdn_prefix_positions_by_layer.items()
        }
    )
    if (
        gdn_layers != tuple(sorted(set(gdn_layers)))
        or any(layer not in audited for layer in gdn_layers)
        or any(layer not in audited for layer in gdn_positions)
        or any(layer not in audited for layer in gdn_prefix_positions)
    ):
        raise ProofV3Error("lean GDN decode-row inventory is malformed")

    # The signed policy selects transition rows and one returned-token path
    # after the nonce.  Committing the complete runtime trace pre-nonce is
    # mandatory; opening every decode row is not, and would make proof size
    # depend on the requested output length instead of the signed audit count.
    transition_positions = lean_transition_positions_v3(challenge)
    positions_by_layer = {
        layer: set(transition_positions) for layer in audited
    }
    if gdn_layers or gdn_positions:
        for layer in sorted(set(gdn_layers) | set(gdn_positions)):
            decode_positions = gdn_positions.get(layer)
            if decode_positions is None:
                decode_positions = tuple(
                    range(
                        challenge.context_token_count,
                        challenge.sequence_token_count,
                    )
                )
            if (
                not decode_positions
                or decode_positions
                != tuple(sorted(set(decode_positions)))
                or any(
                    position < challenge.context_token_count
                    or position >= challenge.sequence_token_count
                    for position in decode_positions
                )
            ):
                raise ProofV3Error(
                    "lean GDN transition rows are malformed"
                )
            if layer in gdn_layers:
                positions_by_layer[layer] = set(decode_positions)
            else:
                from verallm.proof_v3.gdn_decode_corridor import (
                    derive_gdn_projection_binding_positions_v3,
                )

                positions_by_layer[layer].update(
                    derive_gdn_projection_binding_positions_v3(
                        selection_seed=challenge.selection_seed,
                        layer_index=layer,
                        sequence_positions=decode_positions,
                    )
                )
    stride = LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
    for layer, positions in gdn_prefix_positions.items():
        if (
            not positions
            or positions != tuple(sorted(set(positions)))
            or any(
                position < 0
                or position >= challenge.context_token_count
                for position in positions
            )
        ):
            raise ProofV3Error(
                "lean GDN prefix-cache transition rows are malformed"
            )
        layer_slot = layers.index(layer)
        start = (layer_slot // stride) * stride
        stop = min(start + stride, len(layers))
        for corridor_layer in layers[start:stop]:
            if corridor_layer in positions_by_layer:
                positions_by_layer[corridor_layer].update(positions)
    for raw_layer, raw_positions in attention_kv_positions_by_layer.items():
        layer = int(raw_layer)
        if layer not in audited:
            raise ProofV3Error(
                "lean attention K/V rows reference an unaudited layer"
            )
        positions = tuple(int(position) for position in raw_positions)
        if (
            not positions
            or positions != tuple(sorted(set(positions)))
            or any(
                position < 0
                or position >= challenge.context_token_count
                for position in positions
            )
        ):
            raise ProofV3Error("lean attention K/V rows are malformed")
        layer_slot = layers.index(layer)
        start = (layer_slot // stride) * stride
        stop = min(start + stride, len(layers))
        for corridor_layer in layers[start:stop]:
            if corridor_layer in positions_by_layer:
                positions_by_layer[corridor_layer].update(positions)

    pool_slot = {
        int(position): slot
        for slot, position in enumerate(
            challenge.candidate_sequence_positions
        )
    }
    result = []
    for layer in audited:
        positions = tuple(sorted(positions_by_layer[layer]))
        appended = tuple(
            position for position in positions if position not in pool_slot
        )
        appended_slot = {
            position: challenge.candidate_row_count + index
            for index, position in enumerate(appended)
        }
        layout = tuple(
            sorted(
                (
                    (
                        position,
                        (
                            pool_slot[position]
                            if position in pool_slot
                            else appended_slot[position]
                        ),
                    )
                    for position in positions
                ),
                key=lambda pair: pair[1],
            )
        )
        result.append((layer, layout))
    return tuple(result)


def lean_transition_positions_v3(
    challenge: EconomicChallengeV3,
) -> tuple[int, ...]:
    """Return the exact nonce-selected positions used for residual chaining."""

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("lean transition challenge has a wrong type")
    positions = set(int(position) for position in challenge.sampled_sequence_positions)
    positions.update(
        challenge.context_token_count - 1 + int(output_position)
        for output_position in challenge.audited_decode_positions
    )
    result = tuple(sorted(positions))
    if (
        not result
        or any(
            position < 0 or position >= challenge.sequence_token_count
            for position in result
        )
    ):
        raise ProofV3Error("lean transition positions are malformed")
    return result


def lean_shared_boundary_rows_v3(
    *,
    previous_layout,
    next_layout,
    required_positions,
) -> tuple[tuple[int, int, int], ...]:
    """Match adjacent compact row layouts by authenticated sequence position.

    Full-attention layers append nonce-selected K/V provenance positions to
    their ordinary transition rows.  Those layer-specific positions need not
    occupy the same compact slots in an adjacent layer.  Residual continuity
    is therefore compared by absolute sequence position, while the canonical
    transition positions must be present on both sides.
    """

    def _position_to_row(layout, name: str) -> dict[int, int]:
        pairs = tuple((int(position), int(row)) for position, row in layout)
        if (
            not pairs
            or len({position for position, _row in pairs}) != len(pairs)
            or len({row for _position, row in pairs}) != len(pairs)
            or any(position < 0 or row < 0 for position, row in pairs)
        ):
            raise ProofV3Error(f"{name} lean row layout is malformed")
        return {position: row for position, row in pairs}

    previous = _position_to_row(previous_layout, "previous")
    next_ = _position_to_row(next_layout, "next")
    required = tuple(sorted(set(int(position) for position in required_positions)))
    if not required or any(position < 0 for position in required):
        raise ProofV3Error("required lean boundary positions are malformed")
    shared = tuple(sorted(set(previous) & set(next_)))
    if not set(required).issubset(shared):
        raise ProofV3Error(
            "adjacent lean boundaries omit a required transition position"
        )
    return tuple(
        (position, previous[position], next_[position])
        for position in shared
    )


def expected_lean_execution_anchor_reveals_v3(
    *,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    attention_rows_by_layer: Mapping[int, tuple[int, ...]] | None = None,
    gdn_runtime_semantics=None,
    complete_gdn_projection_window: bool = False,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return exact precommitted boundary rows needed by one hard audit.

    For a selected layer, its input is either the signed embedding (layer 0)
    or the previous layer's residual-output commitment.  Full-attention QKV
    openings cover each selected query plus independent prefix rows used to
    bind the K/V cache to registered QKV projection weights. GDN convolution
    checkpoints remain row openings; the much wider recurrent checkpoints
    are authenticated through nonce-selected head-lane openings.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("lean execution-anchor challenge has a wrong type")
    if not isinstance(complete_gdn_projection_window, bool):
        raise ProofV3Error(
            "lean GDN projection-window selector must be boolean"
        )
    layers_and_kinds = _layers_and_kinds(layer_indices, layer_kinds)
    layers = tuple(layer for layer, _kind in layers_and_kinds)
    kinds = dict(layers_and_kinds)
    selected = tuple(sorted(int(x) for x in challenge.selected_layer_indices))
    audited = lean_selected_corridor_layers_v3(
        selected_layer_indices=selected,
        layer_indices=layers,
    )
    checkpoints = set(lean_execution_checkpoint_layers_v3(layers))

    sampled_positions = {
        int(challenge.candidate_sequence_positions[row])
        for row in challenge.sampled_token_rows
    }
    # Bind the nonce-selected returned-token path in every selected corridor.
    # Other decode rows remain covered by the pre-nonce runtime commitments
    # and are selected probabilistically across independent canaries.
    sampled_positions.update(
        challenge.context_token_count - 1 + output_position
        for output_position in challenge.audited_decode_positions
    )
    expected: dict[str, set[int]] = {}
    stride = LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
    for layer in selected:
        position = layers.index(layer)
        corridor_start = (position // stride) * stride
        corridor_end = min(corridor_start + stride, len(layers)) - 1
        endpoint = layers[corridor_end]
        if endpoint not in checkpoints:
            raise ProofV3Error(
                "lean execution corridor has no signed endpoint"
            )
        expected.setdefault(f"l{endpoint}.residual_out", set()).update(
            sampled_positions
        )
        if corridor_start:
            previous = layers[corridor_start - 1]
            expected.setdefault(
                f"l{previous}.residual_out", set()
            ).update(sampled_positions)
    for layer in audited:
        if kinds[layer] == "gdn":
            semantics = (
                gdn_runtime_semantics.layer_for(layer)
                if gdn_runtime_semantics is not None
                else None
            )
            from verallm.proof_v3.gdn_decode_corridor import (
                derive_gdn_decode_corridor_for_challenge_v3,
            )

            plan = derive_gdn_decode_corridor_for_challenge_v3(
                challenge=challenge,
                semantics=semantics,
            )
            if plan is None:
                expected[f"l{layer}.gdn_conv_prompt_boundary"] = {0}
                if complete_gdn_projection_window:
                    expected[
                        f"l{layer}.gdn_recurrent_prompt_boundary"
                    ] = {0}
                decode_positions = set(
                    range(
                        challenge.context_token_count,
                        challenge.sequence_token_count,
                    )
                )
            else:
                checkpoint_rows = {
                    plan.start_checkpoint_row,
                    plan.end_checkpoint_row,
                }
                expected[
                    f"l{layer}.gdn_conv_decode_checkpoints"
                ] = set(checkpoint_rows)
                if complete_gdn_projection_window:
                    expected[
                        f"l{layer}.gdn_recurrent_decode_checkpoints"
                    ] = set(checkpoint_rows)
                if complete_gdn_projection_window:
                    decode_positions = set(plan.sequence_positions)
                else:
                    from verallm.proof_v3.gdn_decode_corridor import (
                        derive_gdn_projection_binding_positions_v3,
                    )

                    decode_positions = set(
                        derive_gdn_projection_binding_positions_v3(
                            selection_seed=challenge.selection_seed,
                            layer_index=layer,
                            sequence_positions=plan.sequence_positions,
                        )
                    )
            if not decode_positions:
                raise ProofV3Error(
                    "lean GDN replay requires a forwarded decode row"
                )
            expected.setdefault(
                f"l{layer}.residual_out", set()
            ).update(decode_positions)
            layer_slot = layers.index(layer)
            if layer_slot:
                expected.setdefault(
                    f"l{layers[layer_slot - 1]}.residual_out",
                    set(),
                ).update(decode_positions)

    last = layers[-1]
    expected.setdefault(f"l{last}.residual_out", set()).update(
        challenge.context_token_count - 1 + output_position
        for output_position in challenge.audited_decode_positions
    )

    attention_rows_by_layer = attention_rows_by_layer or {}
    kv_positions_by_layer: dict[int, tuple[int, ...]] = {}
    for raw_layer, raw_positions in attention_rows_by_layer.items():
        layer = int(raw_layer)
        if layer not in audited or kinds.get(layer) != "full_attention":
            raise ProofV3Error(
                "lean attention rows reference an unselected or GDN layer"
            )
        positions = tuple(int(position) for position in raw_positions)
        if (
            not positions
            or positions != tuple(sorted(set(positions)))
            or any(
                position not in challenge.attention_candidate_positions
                for position in positions
            )
        ):
            raise ProofV3Error(
                "lean attention rows are outside the nonce-derived pool"
            )
        kv_rows = expected.setdefault(
            f"l{layer}.attention_kv_output", set()
        )
        kv_rows.update(positions)
        kv_rows.update(
            derive_lean_kv_projection_rows_v3(
                selection_seed=challenge.selection_seed,
                layer_index=layer,
                query_position=max(positions),
            )
        )
        kv_positions_by_layer[layer] = tuple(sorted(kv_rows))

    missing_attention = {
        layer
        for layer in audited
        if kinds[layer] == "full_attention"
    } - {int(layer) for layer in attention_rows_by_layer}
    if missing_attention:
        raise ProofV3Error(
            "lean full-attention selection has no query-row plan"
        )

    # A selected K/V row is useful only if its QKV input is itself linked to
    # an authenticated execution state.  Open the surrounding residual
    # checkpoints at those same absolute positions; the post-nonce proof then
    # replays and proves every registered operation inside that corridor.
    stride = LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
    for layer, positions in kv_positions_by_layer.items():
        layer_slot = layers.index(layer)
        corridor_start = (layer_slot // stride) * stride
        corridor_end = min(corridor_start + stride, len(layers)) - 1
        endpoint = layers[corridor_end]
        expected.setdefault(f"l{endpoint}.residual_out", set()).update(
            positions
        )
        if corridor_start:
            previous = layers[corridor_start - 1]
            expected.setdefault(
                f"l{previous}.residual_out", set()
            ).update(positions)

    return tuple(
        (stage_id, tuple(sorted(rows)))
        for stage_id, rows in sorted(expected.items())
    )
