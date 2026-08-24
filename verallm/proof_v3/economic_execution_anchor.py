"""Fail-closed verifier rules for streaming economic execution anchors.

The full-sequence roots are frozen before the validator nonce.  This module
derives the exact signed stage inventory and the exact absolute runtime rows
that a hard audit must open after the nonce, then authenticates every raw row
against its frozen root.  It deliberately does not interpret or quantize the
raw FP16/BF16 values; that cross-binding is a separate verifier step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.economic_wire import (
    EconomicExecutionAnchorRevealV3,
    EconomicOracleCommitmentV3,
    bits_to_scale_v3,
    scale_to_bits_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    verify_execution_anchor_lane_v3,
    verify_execution_anchor_row_v3,
)
from zkllm.types import MerklePath

_COMMON_STAGE_DIMS = (
    ("mlp_gate_up_input", "gate_up", 0),
    ("mlp_gate_up_output", "gate_up", 1),
    ("mlp_down_input", "down", 0),
    ("mlp_down_output", "down", 1),
)
_FULL_ATTENTION_STAGE_DIMS = (
    ("attention_qkv_input", "qkv", 0),
    ("attention_qkv_output", "qkv", 1),
    ("attention_o_input", "o", 0),
    ("attention_o_output", "o", 1),
)
_GDN_STAGE_DIMS = (
    ("gdn_qkvz_input", "gdn_qkvz", 0),
    ("gdn_qkvz_output", "gdn_qkvz", 1),
    ("gdn_ba_input", "gdn_ba", 0),
    ("gdn_ba_output", "gdn_ba", 1),
    ("gdn_o_input", "gdn_o", 0),
    ("gdn_o_output", "gdn_o", 1),
)
_RESIDUAL_STAGES = (
    "residual_in",
    "residual_after_attention",
    "residual_out",
)
_GDN_BOUNDARY_SUFFIXES = (
    "gdn_conv_prompt_boundary",
)
_LANE_ONLY_OUTPUT_SUFFIXES = frozenset(
    (
        "attention_o_output",
        "gdn_o_output",
        "mlp_gate_up_output",
        "mlp_down_output",
        "residual_out",
    )
)

__all__ = [
    "expected_economic_execution_anchor_inventory_v3",
    "expected_economic_execution_anchor_reveals_v3",
    "expected_economic_transition_anchor_cells_v3",
    "required_economic_transition_anchor_lane_keys_v3",
    "verify_economic_transition_anchor_lanes_v3",
    "economic_execution_anchor_encoding_v3",
    "derive_economic_execution_anchor_quantized_v3",
    "derive_economic_execution_anchor_oracle_binding_v3",
    "EconomicExecutionAnchorOracleBindingV3",
    "verify_economic_execution_anchor_residual_chain_v3",
    "verify_economic_execution_anchor_reveals_v3",
]


def _layers(values, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or result != tuple(sorted(set(result)))
        or any(value < 0 or value >= 1 << 32 for value in result)
    ):
        raise ProofV3Error(f"{name} is not ordered and distinct")
    return result


def _layer_kinds(
    layer_indices,
    layer_kinds: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    layers = _layers(layer_indices, "execution anchor layer inventory")
    if not isinstance(layer_kinds, Mapping):
        raise ProofV3Error("execution anchor layer kinds are malformed")
    normalized = {
        int(layer): str(kind)
        for layer, kind in layer_kinds.items()
    }
    if set(normalized) != set(layers) or any(
        kind not in {"full_attention", "gdn"}
        for kind in normalized.values()
    ):
        raise ProofV3Error(
            "execution anchor layer kinds do not match the signed inventory"
        )
    return tuple((layer, normalized[layer]) for layer in layers)


def expected_economic_execution_anchor_inventory_v3(
    *,
    layer_indices,
    layer_kinds: Mapping[int, str],
    sequence_token_count: int,
    hidden_dim: int,
    projection_dims: Mapping[str, tuple[int, int]],
    gdn_runtime_semantics=None,
    context_token_count: int | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """Return exact ``(stage_id, row_count, row_width_bytes)`` inventory."""

    layers_and_kinds = _layer_kinds(layer_indices, layer_kinds)
    if (
        isinstance(sequence_token_count, bool)
        or not isinstance(sequence_token_count, int)
        or not 0 < sequence_token_count < 1 << 32
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or not 0 < hidden_dim < 1 << 23
        or not isinstance(projection_dims, Mapping)
    ):
        raise ProofV3Error("execution anchor geometry is malformed")
    records: list[tuple[str, int, int]] = []
    for layer, kind in layers_and_kinds:
        stage_dims = (
            _FULL_ATTENTION_STAGE_DIMS
            if kind == "full_attention"
            else _GDN_STAGE_DIMS
        ) + _COMMON_STAGE_DIMS
        for suffix, projection, side in stage_dims:
            key = f"l{layer}.{projection}"
            dims = projection_dims.get(key)
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
                    f"execution anchor projection geometry is missing {key}"
                )
            records.append(
                (f"l{layer}.{suffix}", sequence_token_count, dims[side] * 2)
            )
        if kind == "gdn":
            if gdn_runtime_semantics is None:
                raise ProofV3Error(
                    "GDN execution anchors require authenticated runtime "
                    "semantics"
                )
            try:
                semantics = gdn_runtime_semantics.layer_for(layer)
            except (AttributeError, ProofV3VerificationError) as exc:
                raise ProofV3Error(
                    f"GDN execution anchor semantics are missing layer {layer}"
                ) from exc
            records.extend(
                (
                    (
                        f"l{layer}.gdn_conv_prompt_boundary",
                        1,
                        semantics.conv_state_bytes,
                    ),
                    (
                        f"l{layer}.gdn_recurrent_prompt_boundary",
                        1,
                        semantics.recurrent_state_bytes,
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

                if (
                    isinstance(context_token_count, bool)
                    or not isinstance(context_token_count, int)
                    or not 0 < context_token_count <= sequence_token_count
                ):
                    raise ProofV3Error(
                        "execution anchor context geometry is malformed"
                    )
                checkpoint_count = len(
                    gdn_decode_checkpoint_offsets_v3(
                        decode_token_count=(
                            sequence_token_count
                            - context_token_count
                            + 1
                        ),
                        checkpoint_stride=checkpoint_stride,
                    )
                )
                records.extend(
                    (
                        (
                            f"l{layer}.gdn_conv_decode_checkpoints",
                            checkpoint_count,
                            semantics.conv_state_bytes,
                        ),
                        (
                            f"l{layer}.gdn_recurrent_decode_checkpoints",
                            checkpoint_count,
                            semantics.recurrent_state_bytes,
                        ),
                    )
                )
        records.extend(
            (
                f"l{layer}.{suffix}",
                sequence_token_count,
                hidden_dim * 2,
            )
            for suffix in _RESIDUAL_STAGES
        )
    return tuple(sorted(records))


def expected_economic_execution_anchor_reveals_v3(
    *,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    attention_rows_by_layer: Mapping[int, tuple[int, ...]] | None = None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return exact stage/absolute-row openings for the transition proof.

    All layer boundaries open at the transition-selected positions.  Selected
    layers additionally open every registered full-attention/MLP stage.  The
    first input includes the prompt bottom-anchor rows and the final layer
    output includes the forwarded row that produced the audited token.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("execution anchor challenge has an unexpected type")
    layers_and_kinds = _layer_kinds(layer_indices, layer_kinds)
    layers = tuple(layer for layer, _kind in layers_and_kinds)
    kinds = dict(layers_and_kinds)
    selected = _layers(
        challenge.selected_layer_indices,
        "execution anchor selected layers",
    )
    if any(layer not in layers for layer in selected):
        raise ProofV3Error(
            "execution anchor selected layer is outside the signed inventory"
        )
    sampled = set(challenge.sampled_sequence_positions)
    expected: dict[str, set[int]] = {}
    for layer in selected:
        expected[f"l{layer}.residual_in"] = set(sampled)
    expected.setdefault(f"l{layers[0]}.residual_in", set()).update(
        challenge.bottom_anchor_positions
    )
    expected.setdefault(f"l{layers[-1]}.residual_out", set()).update(
        challenge.context_token_count - 1 + output_position
        for output_position in challenge.audited_decode_positions
    )
    for layer in selected:
        transition_stages = (
            _FULL_ATTENTION_STAGE_DIMS
            if kinds[layer] == "full_attention"
            else _GDN_STAGE_DIMS
        )
        for suffix, _projection, _side in (
            transition_stages + _COMMON_STAGE_DIMS
        ):
            if suffix in _LANE_ONLY_OUTPUT_SUFFIXES:
                continue
            expected[f"l{layer}.{suffix}"] = set(sampled)
        expected[f"l{layer}.residual_after_attention"] = set(sampled)
        if kinds[layer] == "gdn":
            for suffix in _GDN_BOUNDARY_SUFFIXES:
                expected[f"l{layer}.{suffix}"] = {0}
    if attention_rows_by_layer is not None:
        for layer, positions in attention_rows_by_layer.items():
            layer = int(layer)
            if layer not in selected or kinds[layer] != "full_attention":
                raise ProofV3Error(
                    "attention anchor rows reference an unselected or GDN layer"
                )
            rows = {int(position) for position in positions}
            if (
                not rows
                or any(
                    position not in challenge.attention_candidate_positions
                    for position in rows
                )
            ):
                raise ProofV3Error(
                    "attention anchor rows are outside the nonce-derived pool"
                )
            expected[f"l{layer}.attention_qkv_output"].update(rows)
            expected[f"l{layer}.attention_o_input"].update(rows)
    return tuple(
        (stage_id, tuple(sorted(positions)))
        for stage_id, positions in sorted(expected.items())
    )


def expected_economic_transition_anchor_cells_v3(
    *,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    projection_dims: Mapping[str, tuple[int, int]],
    hidden_dim: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Derive the exact output cells carried by authenticated anchor lanes.

    Projection inputs and norm sources remain complete rows. Wide projection
    outputs and residual destinations need only the nonce-selected cells used
    by the exact projection corridors and probabilistic transition checks.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("execution anchor challenge has an unexpected type")
    layers_and_kinds = _layer_kinds(layer_indices, layer_kinds)
    layers = tuple(layer for layer, _kind in layers_and_kinds)
    kinds = dict(layers_and_kinds)
    if (
        not isinstance(projection_dims, Mapping)
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
    ):
        raise ProofV3Error("transition anchor cell geometry is malformed")
    positions = tuple(challenge.sampled_sequence_positions)
    result: dict[str, set[tuple[int, int]]] = {}

    def add(stage_id: str, columns) -> None:
        cells = result.setdefault(stage_id, set())
        cells.update(
            (int(position), int(column))
            for position in positions
            for column in columns
        )

    for layer in challenge.selected_layer_indices:
        if layer not in kinds:
            raise ProofV3Error(
                "transition anchor cells reference an unknown layer"
            )
        residual_cols = challenge.residual_cols_for(
            layer_index=layer,
            hidden_dim=hidden_dim,
        )
        gate_out = projection_dims[f"l{layer}.gate_up"][1]
        if gate_out % 2:
            raise ProofV3Error(
                f"layer {layer} gate/up output width is not even"
            )
        inter_dim = gate_out // 2
        mlp_cols = challenge.mlp_cols_for(
            layer_index=layer,
            inter_dim=inter_dim,
        )
        gate_cells = set(
            challenge.out_cells_for(
                layer_index=layer,
                out_dim=gate_out,
                projection="gate_up",
            )
        )
        gate_cells.update(mlp_cols)
        gate_cells.update(inter_dim + column for column in mlp_cols)
        add(f"l{layer}.mlp_gate_up_output", gate_cells)

        down_out = projection_dims[f"l{layer}.down"][1]
        down_cells = set(
            challenge.out_cells_for(
                layer_index=layer,
                out_dim=down_out,
                projection="down",
            )
        )
        down_cells.update(residual_cols)
        add(f"l{layer}.mlp_down_output", down_cells)
        add(f"l{layer}.residual_out", residual_cols)

        if kinds[layer] == "full_attention":
            output_name = "o"
            stage_suffix = "attention_o_output"
        else:
            output_name = "gdn_o"
            stage_suffix = "gdn_o_output"
        output_dim = projection_dims[f"l{layer}.{output_name}"][1]
        output_cells = set(
            challenge.out_cells_for(
                layer_index=layer,
                out_dim=output_dim,
                projection=output_name,
            )
        )
        output_cells.update(residual_cols)
        add(f"l{layer}.{stage_suffix}", output_cells)

    return {
        stage_id: tuple(sorted(cells))
        for stage_id, cells in sorted(result.items())
    }


def required_economic_transition_anchor_lane_keys_v3(
    *,
    commitments,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    projection_dims: Mapping[str, tuple[int, int]],
    hidden_dim: int,
) -> tuple[tuple[int, int, int], ...]:
    """Map the exact transition-output cells to canonical 2-KiB lanes."""

    commitments = tuple(commitments)
    if not all(
        isinstance(item, ExecutionAnchorCommitmentV3)
        for item in commitments
    ):
        raise ProofV3VerificationError(
            "transition anchor commitment inventory is malformed"
        )
    index_by_stage = {
        item.stage_id: index for index, item in enumerate(commitments)
    }
    if len(index_by_stage) != len(commitments):
        raise ProofV3VerificationError(
            "transition anchor commitment inventory is duplicated"
        )
    cells_by_stage = expected_economic_transition_anchor_cells_v3(
        challenge=challenge,
        layer_indices=layer_indices,
        layer_kinds=layer_kinds,
        projection_dims=projection_dims,
        hidden_dim=hidden_dim,
    )
    keys: set[tuple[int, int, int]] = set()
    for stage_id, cells in cells_by_stage.items():
        try:
            commitment_index = index_by_stage[stage_id]
            commitment = commitments[commitment_index]
        except KeyError as exc:
            raise ProofV3VerificationError(
                f"transition anchor inventory has no {stage_id}"
            ) from exc
        if commitment.row_width % 2:
            raise ProofV3VerificationError(
                f"transition anchor {stage_id} has a non-half row width"
            )
        width = commitment.row_width // 2
        for position, column in cells:
            if position >= commitment.row_count or column >= width:
                raise ProofV3VerificationError(
                    f"transition anchor {stage_id} cell is out of range"
                )
            keys.add(
                (
                    commitment_index,
                    position,
                    (column * 2) // 2048,
                )
            )
    return tuple(sorted(keys))


def verify_economic_transition_anchor_lanes_v3(
    *,
    commitments,
    lane_reveals,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    projection_dims: Mapping[str, tuple[int, int]],
    hidden_dim: int,
) -> dict[str, dict[tuple[int, int], bytes]]:
    """Authenticate and extract the exact lane-backed transition cells."""

    from verallm.proof_v3.attention_anchor_binding import (
        extract_execution_anchor_range_v3,
    )

    commitments = tuple(commitments)
    required = required_economic_transition_anchor_lane_keys_v3(
        commitments=commitments,
        challenge=challenge,
        layer_indices=layer_indices,
        layer_kinds=layer_kinds,
        projection_dims=projection_dims,
        hidden_dim=hidden_dim,
    )
    required_set = set(required)
    relevant = tuple(
        reveal
        for reveal in lane_reveals
        if (
            int(reveal.commitment_index),
            int(reveal.opening.row_index),
            int(reveal.opening.lane_index),
        )
        in required_set
    )
    actual = tuple(
        (
            int(reveal.commitment_index),
            int(reveal.opening.row_index),
            int(reveal.opening.lane_index),
        )
        for reveal in relevant
    )
    if actual != required:
        raise ProofV3VerificationError(
            "transition anchor lanes do not match the nonce-selected cells"
        )
    openings: dict[
        int,
        dict[tuple[int, int], object],
    ] = {}
    for reveal in relevant:
        commitment = commitments[reveal.commitment_index]
        verify_execution_anchor_lane_v3(
            commitment=commitment,
            opening=reveal.opening,
        )
        openings.setdefault(reveal.commitment_index, {})[
            (
                int(reveal.opening.row_index),
                int(reveal.opening.lane_index),
            )
        ] = reveal.opening
    cells_by_stage = expected_economic_transition_anchor_cells_v3(
        challenge=challenge,
        layer_indices=layer_indices,
        layer_kinds=layer_kinds,
        projection_dims=projection_dims,
        hidden_dim=hidden_dim,
    )
    index_by_stage = {
        item.stage_id: index for index, item in enumerate(commitments)
    }
    result: dict[str, dict[tuple[int, int], bytes]] = {}
    for stage_id, cells in cells_by_stage.items():
        commitment_index = index_by_stage[stage_id]
        commitment = commitments[commitment_index]
        stage_cells = {}
        for position, column in cells:
            stage_cells[(position, column)] = (
                extract_execution_anchor_range_v3(
                    commitment=commitment,
                    row_index=position,
                    byte_start=column * 2,
                    byte_length=2,
                    openings=openings[commitment_index],
                )
            )
        result[stage_id] = stage_cells
    return result


def verify_economic_execution_anchor_residual_chain_v3(
    *,
    commitments: tuple[ExecutionAnchorCommitmentV3, ...],
    layer_indices,
) -> None:
    """Verify complete residual continuity from the pre-nonce commitments.

    Runtime rows use the same canonical lane/row Merkle construction for every
    stage. Adjacent residual stages have identical signed geometry, so equal
    roots bind every ordered row byte-for-byte. Selected-layer arithmetic,
    the prompt bottom anchor, and the final-token anchor are still opened
    separately; retransmitting sampled rows at every unselected boundary adds
    no stronger continuity statement.
    """

    layers = _layers(layer_indices, "execution anchor residual chain")
    commitments = tuple(commitments)
    by_stage = {
        commitment.stage_id: commitment
        for commitment in commitments
        if isinstance(commitment, ExecutionAnchorCommitmentV3)
    }
    if len(by_stage) != len(commitments):
        raise ProofV3VerificationError(
            "execution anchor residual inventory is malformed"
        )
    for previous, following in zip(layers, layers[1:], strict=False):
        try:
            output = by_stage[f"l{previous}.residual_out"]
            input_ = by_stage[f"l{following}.residual_in"]
        except KeyError as exc:
            raise ProofV3VerificationError(
                "execution anchor residual chain is incomplete"
            ) from exc
        if (
            output.row_count != input_.row_count
            or output.row_width != input_.row_width
            or output.root != input_.root
        ):
            raise ProofV3VerificationError(
                f"execution anchor residual chain disconnects layers "
                f"{previous} and {following}"
            )


def verify_economic_execution_anchor_reveals_v3(
    *,
    commitments: tuple[ExecutionAnchorCommitmentV3, ...],
    reveals: tuple[EconomicExecutionAnchorRevealV3, ...],
    expected_inventory: tuple[tuple[str, int, int], ...],
    expected_reveals: tuple[tuple[str, tuple[int, ...]], ...],
) -> dict[str, dict[int, bytes]]:
    """Authenticate the exact expected raw rows; reject any shape deviation."""

    commitments = tuple(commitments)
    reveals = tuple(reveals)
    actual_inventory = tuple(
        (item.stage_id, item.row_count, item.row_width)
        for item in commitments
    )
    if actual_inventory != tuple(expected_inventory):
        raise ProofV3VerificationError(
            "execution anchor inventory does not match the signed geometry"
        )
    expected_by_stage = dict(expected_reveals)
    expected_indices = tuple(
        index
        for index, commitment in enumerate(commitments)
        if commitment.stage_id in expected_by_stage
    )
    if tuple(reveal.commitment_index for reveal in reveals) != expected_indices:
        raise ProofV3VerificationError(
            "execution anchor reveals do not cover the exact selected stages"
        )

    opened: dict[str, dict[int, bytes]] = {}
    for reveal in reveals:
        commitment = commitments[reveal.commitment_index]
        expected_rows = expected_by_stage[commitment.stage_id]
        if tuple(row.row_index for row in reveal.rows) != expected_rows:
            raise ProofV3VerificationError(
                f"execution anchor {commitment.stage_id} rows do not match "
                "the validator-derived selection"
            )
        path_depth = (commitment.row_count - 1).bit_length()
        stage_rows: dict[int, bytes] = {}
        for row in reveal.rows:
            if (
                len(row.row_bytes) != commitment.row_width
                or len(row.sibling_hashes) != path_depth
            ):
                raise ProofV3VerificationError(
                    f"execution anchor {commitment.stage_id} opening geometry "
                    "is malformed"
                )
            path = MerklePath(
                leaf_index=row.row_index,
                siblings=[
                    (
                        sibling,
                        bool((row.row_index >> level) & 1),
                    )
                    for level, sibling in enumerate(row.sibling_hashes)
                ],
            )
            verify_execution_anchor_row_v3(
                commitment=commitment,
                row_index=row.row_index,
                row_bytes=row.row_bytes,
                path=path,
            )
            stage_rows[row.row_index] = row.row_bytes
        opened[commitment.stage_id] = stage_rows
    return opened


def economic_execution_anchor_encoding_v3(profile) -> str:
    """Return the one signed runtime encoding used by execution anchors."""

    try:
        tensors = tuple(profile.relation_spec.tensors)
    except AttributeError as exc:
        raise ProofV3VerificationError(
            "execution anchor profile has no tensor inventory"
        ) from exc
    # Cache tensors have their own authenticated attention/GDN semantics and
    # may legitimately use a different storage dtype (for example an FP16
    # runtime with FP32 recurrent GDN state).  Generic execution-anchor rows
    # cover the serving-state chain, final hidden state and logits only.
    runtime_roles = {"runtime_state", "final_hidden", "logits"}
    encodings = {
        tensor.encoding_id
        for tensor in tensors
        if tensor.commitment_role in runtime_roles
    }
    if len(encodings) != 1:
        raise ProofV3VerificationError(
            "execution anchor runtime tensors do not share one signed encoding"
        )
    encoding = next(iter(encodings))
    if encoding not in {"fp16.v1", "bf16.v1"}:
        raise ProofV3VerificationError(
            "execution anchor runtime encoding is not qualified"
        )
    return encoding


def _decode_row_v3(row_bytes: bytes, encoding_id: str):
    try:
        import numpy as np

        if encoding_id == "fp16.v1":
            values = np.frombuffer(row_bytes, dtype="<f2").astype(
                np.float64
            )
        elif encoding_id == "bf16.v1":
            words = np.frombuffer(row_bytes, dtype="<u2").astype(np.uint32)
            values = (words << 16).view("<f4").astype(np.float64)
        else:
            raise ProofV3VerificationError(
                "execution anchor runtime encoding is not qualified"
            )
        if not bool(np.isfinite(values).all()):
            raise ProofV3VerificationError(
                "execution anchor runtime row contains a non-finite value"
            )
        return values
    except (ValueError, TypeError) as exc:
        raise ProofV3VerificationError(
            "execution anchor runtime row encoding is malformed"
        ) from exc


def _absmax_scale_v3(rows) -> float:
    try:
        import numpy as np

        maximum = max(
            (float(np.max(np.abs(row), initial=0.0)) for row in rows),
            default=0.0,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProofV3VerificationError(
            "execution anchor scale derivation failed"
        ) from exc
    return max(maximum, 1e-8) / 127.0


def _quantize_row_v3(row, scale: float) -> tuple[int, ...]:
    try:
        import numpy as np

        quantized = np.rint(row / float(scale))
        quantized = np.clip(quantized, -128, 127).astype(np.int16)
        return tuple(quantized.tolist())
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ProofV3VerificationError(
            "execution anchor row quantization failed"
        ) from exc


def quantize_execution_anchor_matrix_v3(
    *,
    row_bytes: bytes,
    row_count: int,
    scale: float,
    encoding_id: str,
):
    """Decode and canonically quantize authenticated runtime rows.

    The binary64 NumPy operation is the protocol's shared rounding domain.
    In particular, the prover must not substitute a CUDA division here:
    values exactly on a half-integer boundary can otherwise round to the
    adjacent integer on different hardware.
    """

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count <= 0
    ):
        raise ProofV3VerificationError(
            "execution anchor matrix row count is malformed"
        )
    try:
        import numpy as np

        values = _decode_row_v3(bytes(row_bytes), encoding_id)
        if values.size == 0 or values.size % row_count:
            raise ProofV3VerificationError(
                "execution anchor matrix geometry is malformed"
            )
        matrix = values.reshape(row_count, values.size // row_count)
        quantized = np.rint(matrix / float(scale))
        return np.clip(quantized, -128, 127).astype(np.int8)
    except ProofV3VerificationError:
        raise
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ProofV3VerificationError(
            "execution anchor matrix quantization failed"
        ) from exc


def quantize_execution_anchor_row_v3(
    *,
    row_bytes: bytes,
    scale: float,
    encoding_id: str,
) -> tuple[int, ...]:
    """Decode and canonically quantize one authenticated runtime row."""

    return tuple(
        int(value)
        for value in quantize_execution_anchor_matrix_v3(
            row_bytes=row_bytes,
            row_count=1,
            scale=scale,
            encoding_id=encoding_id,
        )[0]
    )


@dataclass(frozen=True, slots=True)
class EconomicExecutionAnchorOracleBindingV3:
    """Validator-derived int8 rows and scales for authenticated raw anchors."""

    expected_rows: Mapping[str, Mapping[int, object]]
    expected_cells: Mapping[str, Mapping[tuple[int, int], int]]
    expected_scale_bits: Mapping[str, int]

    def verify_rows(
        self,
        *,
        oracle_id: str,
        actual_rows: Mapping[int, tuple[int, ...]],
        row_indices,
    ) -> None:
        expected = self.expected_rows.get(oracle_id)
        rows = tuple(int(row) for row in row_indices)
        if expected is None or any(row not in expected for row in rows):
            raise ProofV3VerificationError(
                f"execution anchor has no binding for oracle {oracle_id}"
            )
        for row in rows:
            if tuple(actual_rows[row]) != tuple(expected[row]):
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} row is detached from the "
                    "pre-nonce runtime anchor"
                )

    def verify_cells(
        self,
        *,
        oracle_id: str,
        actual_cells: Mapping[tuple[int, int], int],
        cells,
    ) -> None:
        expected = self.expected_rows.get(oracle_id)
        sparse = self.expected_cells.get(oracle_id, {})
        coordinates = tuple((int(row), int(col)) for row, col in cells)
        if expected is None and not sparse:
            raise ProofV3VerificationError(
                f"execution anchor has no binding for oracle {oracle_id}"
            )
        for row, col in coordinates:
            value = sparse.get((row, col))
            values = expected.get(row) if expected is not None else None
            if value is None and values is not None and 0 <= col < len(values):
                value = values[col]
            if value is None:
                raise ProofV3VerificationError(
                    f"execution anchor has no binding for oracle {oracle_id} "
                    "cell"
                )
            if int(actual_cells[(row, col)]) != value:
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} cell is detached from the "
                    "pre-nonce runtime anchor"
                )


def derive_economic_execution_anchor_oracle_binding_v3(
    *,
    opened_rows: Mapping[str, Mapping[int, bytes]],
    opened_cells: Mapping[
        str, Mapping[tuple[int, int], bytes]
    ] | None = None,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    oracle_by_id: Mapping[str, EconomicOracleCommitmentV3] | None,
    projection_dims: Mapping[str, tuple[int, int]],
    embedding_scale: float,
    encoding_id: str,
    attention_runtime_semantics=None,
    moe_layer_indices=(),
) -> EconomicExecutionAnchorOracleBindingV3:
    """Interpret raw anchors and derive every selected transition oracle row.

    Layer oracles are constructed only after the nonce.  Their scale is
    therefore canonical over exactly the authenticated rows that the nonce
    selected, rather than over a predictable candidate pool.  Residual
    boundaries share one scale so equality remains exact; the first residual
    uses the signed embedding scale for the bottom anchor.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3VerificationError(
            "execution anchor challenge has an unexpected type"
        )
    layers_and_kinds = _layer_kinds(layer_indices, layer_kinds)
    layers = tuple(layer for layer, _kind in layers_and_kinds)
    kinds = dict(layers_and_kinds)
    moe_layers = tuple(int(layer) for layer in moe_layer_indices)
    if (
        not isinstance(opened_rows, Mapping)
        or (
            opened_cells is not None
            and not isinstance(opened_cells, Mapping)
        )
        or (
            oracle_by_id is not None
            and not isinstance(oracle_by_id, Mapping)
        )
        or not isinstance(projection_dims, Mapping)
        or isinstance(embedding_scale, bool)
        or not isinstance(embedding_scale, (int, float))
        or embedding_scale <= 0
        or encoding_id not in {"fp16.v1", "bf16.v1"}
        or moe_layers != tuple(sorted(set(moe_layers)))
        or any(layer not in layers for layer in moe_layers)
    ):
        raise ProofV3VerificationError(
            "execution anchor oracle-binding inputs are malformed"
        )
    pool_slot = {
        position: slot
        for slot, position in enumerate(challenge.candidate_sequence_positions)
    }
    decoded: dict[str, dict[int, object]] = {}
    for stage_id, rows in opened_rows.items():
        if not isinstance(rows, Mapping):
            raise ProofV3VerificationError(
                "execution anchor opened rows are malformed"
            )
        # GDN prompt-boundary states use their own signed f16/f32 layouts and
        # feed the recurrence verifier directly.  They are authenticated by
        # the same exact-set anchor verifier but are not int8 oracle rows.
        if any(stage_id.endswith(suffix) for suffix in _GDN_BOUNDARY_SUFFIXES):
            continue
        decoded[stage_id] = {
            int(position): _decode_row_v3(bytes(row), encoding_id)
            for position, row in rows.items()
        }
    decoded_cells: dict[str, dict[tuple[int, int], float]] = {}
    for stage_id, cells in (opened_cells or {}).items():
        if not isinstance(cells, Mapping):
            raise ProofV3VerificationError(
                "execution anchor opened cells are malformed"
            )
        decoded_cells[stage_id] = {}
        for coordinate, raw in cells.items():
            try:
                position, column = coordinate
            except (TypeError, ValueError) as exc:
                raise ProofV3VerificationError(
                    "execution anchor opened cell coordinate is malformed"
                ) from exc
            value = _decode_row_v3(bytes(raw), encoding_id)
            if len(value) != 1:
                raise ProofV3VerificationError(
                    "execution anchor opened cell width is malformed"
                )
            decoded_cells[stage_id][
                (int(position), int(column))
            ] = float(value[0])

    expected: dict[str, dict[int, object]] = {}
    expected_cells: dict[str, dict[tuple[int, int], int]] = {}
    expected_scale_bits: dict[str, int] = {}

    def _oracle(oracle_id: str) -> EconomicOracleCommitmentV3:
        if oracle_by_id is None:
            raise ProofV3VerificationError(
                "execution anchor oracle validation was not requested"
            )
        oracle = oracle_by_id.get(oracle_id)
        if not isinstance(oracle, EconomicOracleCommitmentV3):
            raise ProofV3VerificationError(
                f"execution anchor oracle {oracle_id} is missing"
            )
        return oracle

    def _bind(
        *,
        stage_id: str,
        oracle_id: str,
        scale: float | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> None:
        import numpy as np

        stage = decoded.get(stage_id)
        if not stage:
            raise ProofV3VerificationError(
                f"execution anchor stage {stage_id} is missing"
            )
        slices = {
            position: row[start:stop]
            for position, row in stage.items()
            if position in pool_slot
        }
        if not slices:
            raise ProofV3VerificationError(
                f"execution anchor stage {stage_id} has no selected pool row"
            )
        positions = tuple(slices)
        try:
            matrix = np.stack(
                tuple(slices[position] for position in positions),
                axis=0,
            )
        except (TypeError, ValueError) as exc:
            raise ProofV3VerificationError(
                f"execution anchor stage {stage_id} rows are malformed"
            ) from exc
        canonical_scale = (
            max(
                float(np.max(np.abs(matrix), initial=0.0)),
                1e-8,
            )
            / 127.0
            if scale is None
            else float(scale)
        )
        scale_bits = scale_to_bits_v3(canonical_scale)
        canonical_scale = bits_to_scale_v3(scale_bits)
        try:
            quantized_matrix = np.clip(
                np.rint(matrix / canonical_scale),
                -128,
                127,
            ).astype(np.int16)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            raise ProofV3VerificationError(
                "execution anchor row quantization failed"
            ) from exc
        rows = {
            pool_slot[position]: quantized_matrix[index]
            for index, position in enumerate(positions)
        }
        if oracle_by_id is not None:
            oracle = _oracle(oracle_id)
            if oracle.scale_bits != scale_bits:
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} scale is not derived from the "
                    "authenticated runtime rows"
                )
            if any(len(row) != oracle.col_count for row in rows.values()):
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} width disagrees with its runtime anchor"
                )
        expected[oracle_id] = rows
        expected_scale_bits[oracle_id] = scale_bits

    def _bind_cells(
        *,
        stage_id: str,
        oracle_id: str,
        scale: float | None = None,
    ) -> None:
        import numpy as np

        stage = decoded_cells.get(stage_id)
        if not stage:
            complete_rows = {
                position: row
                for position, row in decoded.get(stage_id, {}).items()
                if position in pool_slot
            }
            if not complete_rows:
                raise ProofV3VerificationError(
                    f"execution anchor stage {stage_id} has no selected cells"
                )
            if oracle_id in expected:
                # A preceding complete-row binding already authenticated and
                # quantized this exact stage under the same signed scale.
                return
            positions = tuple(complete_rows)
            try:
                matrix = np.stack(
                    tuple(complete_rows[position] for position in positions),
                    axis=0,
                )
            except (TypeError, ValueError) as exc:
                raise ProofV3VerificationError(
                    f"execution anchor stage {stage_id} rows are malformed"
                ) from exc
            canonical_scale = (
                max(
                    float(np.max(np.abs(matrix), initial=0.0)),
                    1e-8,
                )
                / 127.0
                if scale is None
                else float(scale)
            )
            scale_bits = scale_to_bits_v3(canonical_scale)
            canonical_scale = bits_to_scale_v3(scale_bits)
            try:
                quantized_matrix = np.clip(
                    np.rint(matrix / canonical_scale),
                    -128,
                    127,
                ).astype(np.int16)
            except (
                ValueError,
                TypeError,
                OverflowError,
                ZeroDivisionError,
            ) as exc:
                raise ProofV3VerificationError(
                    "execution anchor row quantization failed"
                ) from exc
            rows = {
                pool_slot[position]: quantized_matrix[index]
                for index, position in enumerate(positions)
            }
            if oracle_by_id is not None:
                oracle = _oracle(oracle_id)
                if oracle.scale_bits != scale_bits:
                    raise ProofV3VerificationError(
                        f"oracle {oracle_id} scale is not derived from the "
                        "authenticated runtime rows"
                    )
                if any(
                    len(row) != oracle.col_count for row in rows.values()
                ):
                    raise ProofV3VerificationError(
                        f"oracle {oracle_id} width disagrees with its "
                        "runtime anchor"
                    )
            expected[oracle_id] = rows
            expected_scale_bits[oracle_id] = scale_bits
            return
        if not stage:
            raise ProofV3VerificationError(
                f"execution anchor stage {stage_id} has no selected cells"
            )
        selected = {
            (pool_slot[position], column): value
            for (position, column), value in stage.items()
            if position in pool_slot
        }
        if not selected:
            raise ProofV3VerificationError(
                f"execution anchor stage {stage_id} has no selected pool cell"
            )
        canonical_scale = (
            _absmax_scale_v3(
                np.asarray((value,), dtype=np.float64)
                for value in selected.values()
            )
            if scale is None
            else float(scale)
        )
        scale_bits = scale_to_bits_v3(canonical_scale)
        canonical_scale = bits_to_scale_v3(scale_bits)
        quantized = {
            coordinate: _quantize_row_v3(
                np.asarray((value,), dtype=np.float64),
                canonical_scale,
            )[0]
            for coordinate, value in selected.items()
        }
        if oracle_by_id is not None:
            oracle = _oracle(oracle_id)
            if oracle.scale_bits != scale_bits:
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} scale is not derived from the "
                    "authenticated runtime cells"
                )
            if any(
                column >= oracle.col_count
                for _row, column in quantized
            ):
                raise ProofV3VerificationError(
                    f"oracle {oracle_id} cell exceeds its runtime width"
                )
        expected_cells.setdefault(oracle_id, {}).update(quantized)
        previous_scale = expected_scale_bits.get(oracle_id)
        if previous_scale is not None and previous_scale != scale_bits:
            raise ProofV3VerificationError(
                f"oracle {oracle_id} runtime bindings disagree on scale"
            )
        expected_scale_bits[oracle_id] = scale_bits

    # Only selected computation boundaries plus the prompt/final anchors need
    # post-nonce rows. Complete unselected-layer continuity is authenticated
    # by equality of the pre-nonce full-sequence residual roots.
    selected = tuple(challenge.selected_layer_indices)
    first = layers[0]
    last = layers[-1]
    residual_stage_ids = {
        *(f"l{layer}.residual_in" for layer in selected),
        *(f"l{layer}.residual_out" for layer in selected),
        f"l{first}.residual_in",
        f"l{last}.residual_out",
    }
    residual_rows = []
    for stage_id in sorted(residual_stage_ids):
        if stage_id == f"l{first}.residual_in":
            continue
        residual_rows.extend(decoded.get(stage_id, {}).values())
        residual_rows.extend(
            (value,)
            for value in decoded_cells.get(stage_id, {}).values()
        )
    residual_scale = _absmax_scale_v3(residual_rows)

    _bind(
        stage_id=f"l{first}.residual_in",
        oracle_id=f"l{first}.residual_in",
        scale=float(embedding_scale),
    )
    for layer in selected:
        if layer != first:
            _bind(
                stage_id=f"l{layer}.residual_in",
                oracle_id=f"l{layer}.residual_in",
                scale=residual_scale,
            )
        residual_out_stage = f"l{layer}.residual_out"
        if decoded.get(residual_out_stage):
            _bind(
                stage_id=residual_out_stage,
                oracle_id=residual_out_stage,
                scale=residual_scale,
            )
        _bind_cells(
            stage_id=residual_out_stage,
            oracle_id=residual_out_stage,
            scale=residual_scale,
        )
    if last not in selected:
        _bind(
            stage_id=f"l{last}.residual_out",
            oracle_id=f"l{last}.residual_out",
            scale=residual_scale,
        )

    moe_layer_set = frozenset(moe_layers)
    for layer in challenge.selected_layer_indices:
        input_stages = [("residual_after_attention", "mid_residual")]
        output_stages = []
        if layer not in moe_layer_set:
            input_stages[:0] = [
                ("mlp_gate_up_input", "gate_up_x"),
                ("mlp_down_input", "down_x"),
            ]
            output_stages.extend(
                (
                    ("mlp_gate_up_output", "gate_up_y"),
                    ("mlp_down_output", "down_y"),
                )
            )
        for stage_suffix, oracle_suffix in input_stages:
            _bind(
                stage_id=f"l{layer}.{stage_suffix}",
                oracle_id=f"l{layer}.{oracle_suffix}",
            )
        for stage_suffix, oracle_suffix in output_stages:
            _bind_cells(
                stage_id=f"l{layer}.{stage_suffix}",
                oracle_id=f"l{layer}.{oracle_suffix}",
            )
        if kinds[layer] == "full_attention":
            dims = {
                name: projection_dims[f"l{layer}.{name}"]
                for name in ("qkv", "o")
            }
            qkv_out = dims["qkv"][1]
            from verallm.proof_v3.attention_runtime_semantics import (
                Q_GATE_INTERLEAVED_LAYOUT_V3,
            )

            q_width = dims["o"][0] * (
                2
                if (
                    attention_runtime_semantics is not None
                    and attention_runtime_semantics.qkv_layout_id
                    == Q_GATE_INTERLEAVED_LAYOUT_V3
                )
                else 1
            )
            if qkv_out <= q_width or (qkv_out - q_width) % 2:
                raise ProofV3VerificationError(
                    f"layer {layer} QKV geometry is inconsistent"
                )
            kv_dim = (qkv_out - q_width) // 2
            for stage_suffix, oracle_suffix in (
                ("attention_qkv_input", "qkv_x"),
                ("attention_o_input", "attn_o_x"),
            ):
                _bind(
                    stage_id=f"l{layer}.{stage_suffix}",
                    oracle_id=f"l{layer}.{oracle_suffix}",
                )
            _bind_cells(
                stage_id=f"l{layer}.attention_o_output",
                oracle_id=f"l{layer}.attn_o_y",
            )
            qkv_stage = f"l{layer}.attention_qkv_output"
            _bind(
                stage_id=qkv_stage,
                oracle_id=f"l{layer}.q_cache",
                stop=q_width,
            )
            _bind(
                stage_id=qkv_stage,
                oracle_id=f"l{layer}.k_cache",
                start=q_width,
                stop=q_width + kv_dim,
            )
            _bind(
                stage_id=qkv_stage,
                oracle_id=f"l{layer}.v_cache",
                start=q_width + kv_dim,
            )
        else:
            for stage_suffix, oracle_suffix in (
                ("gdn_qkvz_input", "gdn_qkvz_x"),
                ("gdn_qkvz_output", "gdn_qkvz_y"),
                ("gdn_ba_input", "gdn_ba_x"),
                ("gdn_ba_output", "gdn_ba_y"),
                ("gdn_o_input", "gdn_o_x"),
            ):
                _bind(
                    stage_id=f"l{layer}.{stage_suffix}",
                    oracle_id=f"l{layer}.{oracle_suffix}",
                )
            _bind_cells(
                stage_id=f"l{layer}.gdn_o_output",
                oracle_id=f"l{layer}.gdn_o_y",
            )
    return EconomicExecutionAnchorOracleBindingV3(
        expected_rows=expected,
        expected_cells=expected_cells,
        expected_scale_bits=expected_scale_bits,
    )


def derive_economic_execution_anchor_quantized_v3(
    *,
    opened_rows: Mapping[str, Mapping[int, bytes]],
    opened_cells: Mapping[
        str, Mapping[tuple[int, int], bytes]
    ] | None = None,
    challenge: EconomicChallengeV3,
    layer_indices,
    layer_kinds: Mapping[int, str],
    projection_dims: Mapping[str, tuple[int, int]],
    embedding_scale: float,
    encoding_id: str,
    attention_runtime_semantics=None,
    moe_layer_indices=(),
) -> EconomicExecutionAnchorOracleBindingV3:
    """Miner/verifier-shared canonical post-nonce raw-row quantization."""

    return derive_economic_execution_anchor_oracle_binding_v3(
        opened_rows=opened_rows,
        opened_cells=opened_cells,
        challenge=challenge,
        layer_indices=layer_indices,
        layer_kinds=layer_kinds,
        oracle_by_id=None,
        projection_dims=projection_dims,
        embedding_scale=embedding_scale,
        encoding_id=encoding_id,
        attention_runtime_semantics=attention_runtime_semantics,
        moe_layer_indices=moe_layer_indices,
    )
