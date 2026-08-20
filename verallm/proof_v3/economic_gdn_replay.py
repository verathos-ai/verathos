"""Anchor-backed Qwen GDN recurrence verification for economic hard audits.

The prompt-boundary cache and every forwarded decode QKVZ/BA/output row are
frozen before the validator nonce.  After the nonce selects a GDN layer, this
module replays the canonical recurrence from that boundary and compares its
out-projection input to the authenticated runtime row.  The signed semantics
artifact owns runtime/cache encodings and the qualified numerical tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from verallm.proof_v2.transition import (
    GDNReplayParametersV2,
    GDNReplayResultV2,
    ProofV2TransitionError,
    _sigmoid_f32,
    _softplus_f32,
)
from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    execution_anchor_lane_bytes_v3,
    verify_execution_anchor_lane_v3,
)
from verallm.proof_v3.gdn_runtime_semantics import GdnRuntimeSemanticsV3

__all__ = [
    "EconomicGdnReplayStatsV3",
    "economic_gdn_runtime_columns_v3",
    "required_economic_gdn_anchor_lane_keys_v3",
    "required_lean_gdn_checkpoint_lane_keys_v3",
    "verify_economic_gdn_replay_v3",
    "verify_lean_economic_gdn_replay_v3",
    "verify_prefix_cache_gdn_replay_v3",
]


def _decode(blob: bytes, encoding_id: str) -> np.ndarray:
    try:
        if encoding_id == "fp16.v1":
            result = np.frombuffer(blob, dtype="<f2").astype(np.float32)
        elif encoding_id == "fp32.v1":
            result = np.frombuffer(blob, dtype="<f4").copy()
        elif encoding_id == "bf16.v1":
            words = np.frombuffer(blob, dtype="<u2").astype(np.uint32)
            result = (words << 16).view("<f4")
        else:
            raise ProofV3VerificationError(
                "GDN runtime encoding is not qualified"
            )
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "GDN runtime value encoding is malformed"
        ) from exc
    if not bool(np.isfinite(result).all()):
        raise ProofV3VerificationError(
            "GDN runtime value contains a non-finite number"
        )
    return result


def _round(values: np.ndarray, encoding_id: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if encoding_id == "fp32.v1":
        return values.copy()
    if encoding_id == "fp16.v1":
        return values.astype("<f2").astype(np.float32)
    if encoding_id == "bf16.v1":
        # IEEE round-to-nearest-even to BF16, retained as float32 for NumPy
        # arithmetic. This is bit-identical to a BF16 store/load boundary.
        words = values.view(np.uint32)
        rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    raise ProofV3VerificationError("GDN runtime encoding is not qualified")


def economic_gdn_runtime_columns_v3(
    *,
    parameters: GDNReplayParametersV2,
    selected_value_heads,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the canonical compact QKVZ, BA, and output coordinates.

    A selected GDN value head depends only on its own V/Z/BA/output lanes and
    on the Q/K lanes of the key-head group that owns it.  The verifier still
    replays every decode transition, but the wire need not duplicate unrelated
    heads from the authenticated runtime tensors.
    """

    if not isinstance(parameters, GDNReplayParametersV2):
        raise ProofV3VerificationError(
            "GDN compact runtime parameters are malformed"
        )
    selected = tuple(int(head) for head in selected_value_heads)
    nk = parameters.num_key_heads
    nv = parameters.num_value_heads
    dk = parameters.key_head_dim
    dv = parameters.value_head_dim
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(head < 0 or head >= nv for head in selected)
        or nk <= 0
        or nv <= 0
        or nv % nk
        or dk <= 0
        or dv <= 0
    ):
        raise ProofV3VerificationError(
            "GDN compact runtime head selection is malformed"
        )
    key_width = nk * dk
    value_width = nv * dv
    conv_width = 2 * key_width + value_width
    group_size = nv // nk
    qkvz_columns: set[int] = set()
    ba_columns: set[int] = set()
    output_columns: set[int] = set()
    for head in selected:
        key_head = head // group_size
        for start, width in (
            (key_head * dk, dk),
            (key_width + key_head * dk, dk),
            (2 * key_width + head * dv, dv),
            (conv_width + head * dv, dv),
        ):
            qkvz_columns.update(range(start, start + width))
        ba_columns.update((head, nv + head))
        output_columns.update(range(head * dv, (head + 1) * dv))
    return (
        tuple(sorted(qkvz_columns)),
        tuple(sorted(ba_columns)),
        tuple(sorted(output_columns)),
    )


def _replay(
    *,
    qkvz: np.ndarray,
    ba: np.ndarray,
    conv_state: np.ndarray,
    recurrent_state: np.ndarray,
    parameters: GDNReplayParametersV2,
    runtime_encoding_id: str,
    conv_state_encoding_id: str,
    recurrent_state_encoding_id: str,
    selected_value_heads: tuple[int, ...] | None = None,
) -> GDNReplayResultV2:
    rows = int(qkvz.shape[0])
    nk = parameters.num_key_heads
    nv = parameters.num_value_heads
    dk = parameters.key_head_dim
    dv = parameters.value_head_dim
    key_width = nk * dk
    value_width = nv * dv
    conv_width = 2 * key_width + value_width
    selected = (
        tuple(range(nv))
        if selected_value_heads is None
        else tuple(selected_value_heads)
    )
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(head < 0 or head >= nv for head in selected)
        or qkvz.shape != (rows, conv_width + value_width)
        or ba.shape != (rows, 2 * nv)
        or conv_state.shape
        != (parameters.conv_kernel_size - 1, conv_width)
        or recurrent_state.shape != (len(selected), dv, dk)
    ):
        raise ProofV3VerificationError("GDN replay geometry is inconsistent")

    qkvz = qkvz.astype(np.float32)
    ba = ba.astype(np.float32)
    conv_state = _round(conv_state, conv_state_encoding_id)
    recurrent_state = _round(
        recurrent_state, recurrent_state_encoding_id
    )
    conv_weight = np.asarray(parameters.conv_weight, dtype=np.float32)
    a_log = np.asarray(parameters.a_log, dtype=np.float32)
    dt_bias = np.asarray(parameters.dt_bias, dtype=np.float32)
    norm_weight = np.asarray(parameters.norm_weight, dtype=np.float32)
    group_size = nv // nk
    scale = np.float32(dk**-0.5)
    selected_array = np.asarray(selected, dtype=np.intp)
    key_heads = selected_array // group_size
    core_rows = []
    projection_rows = []

    for row_index in range(rows):
        current = qkvz[row_index, :conv_width]
        window = np.concatenate(
            (conv_state, current.reshape(1, -1)), axis=0
        )
        convolved = np.sum(
            window.T * conv_weight,
            axis=1,
            dtype=np.float32,
        )
        convolved = _round(
            convolved * _sigmoid_f32(convolved),
            runtime_encoding_id,
        )
        conv_state = _round(window[1:], conv_state_encoding_id)

        q = convolved[:key_width].reshape(nk, dk)
        k = convolved[key_width:2 * key_width].reshape(nk, dk)
        v = convolved[2 * key_width:].reshape(nv, dv)
        z = qkvz[row_index, conv_width:].reshape(nv, dv)[
            list(selected)
        ]
        beta_raw = ba[row_index, :nv]
        decay_raw = ba[row_index, nv:]
        q *= np.reciprocal(
            np.sqrt(
                np.sum(q * q, axis=1, keepdims=True, dtype=np.float32)
                + np.float32(1e-6),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        k *= np.reciprocal(
            np.sqrt(
                np.sum(k * k, axis=1, keepdims=True, dtype=np.float32)
                + np.float32(1e-6),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        decay = np.exp(
            -np.exp(a_log, dtype=np.float32)
            * _softplus_f32(decay_raw + dt_bias),
            dtype=np.float32,
        )
        beta = _round(_sigmoid_f32(beta_raw), runtime_encoding_id)
        states = recurrent_state.copy()
        states *= decay[selected_array, None, None]
        selected_k = k[key_heads]
        predicted = np.sum(
            states * selected_k[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        delta = v[selected_array] - predicted
        states += (
            beta[selected_array, None, None]
            * delta[:, :, None]
            * selected_k[:, None, :]
        )
        recurrent_state = _round(
            states,
            recurrent_state_encoding_id,
        )
        output = np.sum(
            recurrent_state
            * (q[key_heads] * scale)[:, None, :],
            axis=2,
            dtype=np.float32,
        )

        runtime_output = _round(output, runtime_encoding_id)
        variance = np.mean(
            runtime_output * runtime_output,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        normalized = runtime_output * np.reciprocal(
            np.sqrt(
                variance + np.float32(parameters.rms_epsilon),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        normalized *= norm_weight.reshape(1, -1)
        gated = _round(
            normalized * (z * _sigmoid_f32(z)),
            runtime_encoding_id,
        )
        core_rows.append(runtime_output.copy())
        projection_rows.append(gated.reshape(-1).copy())

    return GDNReplayResultV2(
        conv_state_after=np.ascontiguousarray(conv_state, dtype=np.float32),
        recurrent_state_after=np.ascontiguousarray(
            recurrent_state, dtype=np.float32
        ),
        core_output=np.ascontiguousarray(
            np.stack(core_rows), dtype=np.float32
        ),
        out_projection_input=np.ascontiguousarray(
            np.stack(projection_rows), dtype=np.float32
        ),
    )


@dataclass(frozen=True, slots=True)
class EconomicGdnReplayStatsV3:
    layer_index: int
    row_count: int
    maximum_absolute_error: float


def required_economic_gdn_anchor_lane_keys_v3(
    *,
    commitments,
    challenge: EconomicChallengeV3,
    layer_kinds: Mapping[int, str],
    semantics: GdnRuntimeSemanticsV3,
) -> tuple[tuple[int, int, int], ...]:
    """Derive exact nonce-selected runtime and state lane openings."""

    commitments = tuple(commitments)
    if (
        not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(layer_kinds, Mapping)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
        or not all(
            isinstance(item, ExecutionAnchorCommitmentV3)
            for item in commitments
        )
    ):
        raise ProofV3VerificationError(
            "GDN recurrent anchor metadata is malformed"
        )
    index_by_stage = {
        item.stage_id: index for index, item in enumerate(commitments)
    }
    if len(index_by_stage) != len(commitments):
        raise ProofV3VerificationError(
            "GDN recurrent anchor inventory is duplicated"
        )
    keys: set[tuple[int, int, int]] = set()

    def add_range(
        *,
        commitment_index: int,
        row_index: int,
        byte_start: int,
        byte_length: int,
    ) -> None:
        commitment = commitments[commitment_index]
        lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
        if (
            byte_start < 0
            or byte_length <= 0
            or byte_start + byte_length > commitment.row_width
        ):
            raise ProofV3VerificationError(
                "GDN runtime lane range is outside the committed row"
            )
        keys.update(
            (commitment_index, row_index, lane)
            for lane in range(
                byte_start // lane_bytes,
                (byte_start + byte_length - 1) // lane_bytes + 1,
            )
        )

    for layer in challenge.selected_layer_indices:
        if layer_kinds.get(layer) != "gdn":
            continue
        signed = semantics.layer_for(layer)
        parameters = signed.parameters().replay_parameters()
        stage_id = f"l{layer}.gdn_recurrent_prompt_boundary"
        try:
            commitment_index = index_by_stage[stage_id]
        except KeyError as exc:
            raise ProofV3VerificationError(
                f"GDN recurrent anchor is missing layer {layer}"
            ) from exc
        commitment = commitments[commitment_index]
        if (
            commitment.row_count != 1
            or commitment.row_width != signed.recurrent_state_bytes
        ):
            raise ProofV3VerificationError(
                f"GDN recurrent anchor geometry changed at layer {layer}"
            )
        recurrent_cells = (
            parameters.num_value_heads
            * parameters.value_head_dim
            * parameters.key_head_dim
        )
        if signed.recurrent_state_bytes % recurrent_cells:
            raise ProofV3VerificationError(
                f"lean GDN recurrent checkpoint encoding changed at "
                f"layer {layer}"
            )
        element_bytes = signed.recurrent_state_bytes // recurrent_cells
        if element_bytes not in (2, 4):
            raise ProofV3VerificationError(
                f"lean GDN recurrent checkpoint encoding is unsupported at "
                f"layer {layer}"
            )
        head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * element_bytes
        )
        heads = challenge.gdn_value_heads_for(
            layer_index=layer,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        key_width = parameters.num_key_heads * parameters.key_head_dim
        value_width = parameters.num_value_heads * parameters.value_head_dim
        conv_width = 2 * key_width + value_width
        stage_indices = {}
        for suffix, expected_values in (
            ("gdn_qkvz_output", conv_width + value_width),
            ("gdn_ba_output", 2 * parameters.num_value_heads),
            ("gdn_o_input", value_width),
        ):
            stage_id = f"l{layer}.{suffix}"
            try:
                index = index_by_stage[stage_id]
            except KeyError as exc:
                raise ProofV3VerificationError(
                    f"GDN runtime anchor is missing {stage_id}"
                ) from exc
            commitment = commitments[index]
            if commitment.row_count != challenge.sequence_token_count:
                raise ProofV3VerificationError(
                    f"GDN runtime anchor row count changed at {stage_id}"
                )
            if commitment.row_width % expected_values:
                raise ProofV3VerificationError(
                    f"GDN runtime anchor encoding changed at {stage_id}"
                )
            element_bytes = commitment.row_width // expected_values
            if element_bytes not in (2, 4):
                raise ProofV3VerificationError(
                    f"GDN runtime anchor encoding is unsupported at {stage_id}"
                )
            stage_indices[suffix] = (index, element_bytes)

        positions = range(
            challenge.context_token_count,
            challenge.sequence_token_count,
        )
        group_size = (
            parameters.num_value_heads // parameters.num_key_heads
        )
        for position in positions:
            qkvz_index, element_bytes = stage_indices["gdn_qkvz_output"]
            ba_index, ba_element_bytes = stage_indices["gdn_ba_output"]
            output_index, output_element_bytes = stage_indices["gdn_o_input"]
            for head in heads:
                key_head = head // group_size
                for value_start, width in (
                    (
                        key_head * parameters.key_head_dim,
                        parameters.key_head_dim,
                    ),
                    (
                        key_width
                        + key_head * parameters.key_head_dim,
                        parameters.key_head_dim,
                    ),
                    (
                        2 * key_width
                        + head * parameters.value_head_dim,
                        parameters.value_head_dim,
                    ),
                    (
                        conv_width + head * parameters.value_head_dim,
                        parameters.value_head_dim,
                    ),
                ):
                    add_range(
                        commitment_index=qkvz_index,
                        row_index=position,
                        byte_start=value_start * element_bytes,
                        byte_length=width * element_bytes,
                    )
                for value_start in (
                    head,
                    parameters.num_value_heads + head,
                ):
                    add_range(
                        commitment_index=ba_index,
                        row_index=position,
                        byte_start=value_start * ba_element_bytes,
                        byte_length=ba_element_bytes,
                    )
                add_range(
                    commitment_index=output_index,
                    row_index=position,
                    byte_start=(
                        head
                        * parameters.value_head_dim
                        * output_element_bytes
                    ),
                    byte_length=(
                        parameters.value_head_dim * output_element_bytes
                    ),
                )
        for head in heads:
            start = head * head_bytes
            add_range(
                commitment_index=commitment_index,
                row_index=0,
                byte_start=start,
                byte_length=head_bytes,
            )
    return tuple(sorted(keys))


def required_lean_gdn_checkpoint_lane_keys_v3(
    *,
    commitments,
    challenge: EconomicChallengeV3,
    layer_kinds: Mapping[int, str],
    semantics: GdnRuntimeSemanticsV3,
) -> tuple[tuple[int, int, int], ...]:
    """Select only the recurrent-state heads consumed by bounded replay."""

    commitments = tuple(commitments)
    if (
        not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(layer_kinds, Mapping)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
    ):
        raise ProofV3VerificationError(
            "lean GDN checkpoint metadata is malformed"
        )
    selected = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer_kinds.get(layer) == "gdn"
    )
    if not selected:
        return ()
    if not all(
        isinstance(item, ExecutionAnchorCommitmentV3)
        for item in commitments
    ):
        raise ProofV3VerificationError(
            "lean GDN checkpoint metadata is malformed"
        )
    index_by_stage = {
        item.stage_id: index for index, item in enumerate(commitments)
    }
    if len(index_by_stage) != len(commitments):
        raise ProofV3VerificationError(
            "lean GDN checkpoint inventory is duplicated"
        )
    keys: set[tuple[int, int, int]] = set()
    from verallm.proof_v3.gdn_decode_corridor import (
        derive_gdn_decode_corridor_for_challenge_v3,
    )

    for layer in selected:
        signed = semantics.layer_for(layer)
        parameters = signed.parameters().replay_parameters()
        corridor = derive_gdn_decode_corridor_for_challenge_v3(
            challenge=challenge,
            semantics=signed,
        )
        if corridor is None:
            stage_id = f"l{layer}.gdn_recurrent_prompt_boundary"
            rows = (0,)
        else:
            stage_id = f"l{layer}.gdn_recurrent_decode_checkpoints"
            rows = tuple(
                sorted(
                    {
                        int(corridor.start_checkpoint_row),
                        int(corridor.end_checkpoint_row),
                    }
                )
            )
        try:
            commitment_index = index_by_stage[stage_id]
        except KeyError as exc:
            raise ProofV3VerificationError(
                f"lean GDN recurrent checkpoint is missing {stage_id}"
            ) from exc
        commitment = commitments[commitment_index]
        if (
            commitment.row_width != signed.recurrent_state_bytes
            or any(row < 0 or row >= commitment.row_count for row in rows)
        ):
            raise ProofV3VerificationError(
                f"lean GDN recurrent checkpoint geometry changed at "
                f"layer {layer}"
            )
        element_bytes = (
            signed.recurrent_state_bytes
            // (
                parameters.num_value_heads
                * parameters.value_head_dim
                * parameters.key_head_dim
            )
        )
        head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * element_bytes
        )
        lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
        heads = challenge.gdn_value_heads_for(
            layer_index=layer,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        for row in rows:
            for head in heads:
                start = head * head_bytes
                keys.update(
                    (commitment_index, row, lane)
                    for lane in range(
                        start // lane_bytes,
                        (start + head_bytes - 1) // lane_bytes + 1,
                    )
                )
    return tuple(sorted(keys))


def verify_economic_gdn_replay_v3(
    *,
    opened_rows: Mapping[str, Mapping[int, bytes]],
    challenge: EconomicChallengeV3,
    layer_kinds: Mapping[int, str],
    semantics: GdnRuntimeSemanticsV3,
    anchor_encoding_id: str,
    commitments,
    lane_reveals,
) -> tuple[EconomicGdnReplayStatsV3, ...]:
    """Replay nonce-selected heads over every forwarded decode row."""

    if (
        not isinstance(opened_rows, Mapping)
        or not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(layer_kinds, Mapping)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
    ):
        raise ProofV3VerificationError("GDN replay inputs are malformed")
    selected = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer_kinds.get(layer) == "gdn"
    )
    if not selected:
        return ()
    positions = tuple(
        range(
            challenge.context_token_count,
            challenge.sequence_token_count,
        )
    )
    if not positions:
        raise ProofV3VerificationError(
            "GDN hard audit requires at least one forwarded decode row"
        )
    commitments = tuple(commitments)
    lane_reveals = tuple(lane_reveals)
    required_lane_keys = required_economic_gdn_anchor_lane_keys_v3(
        commitments=commitments,
        challenge=challenge,
        layer_kinds=layer_kinds,
        semantics=semantics,
    )
    required_commitment_indices = {
        commitment_index
        for commitment_index, _row, _lane in required_lane_keys
    }
    relevant_reveals = tuple(
        reveal
        for reveal in lane_reveals
        if getattr(reveal, "commitment_index", -1)
        in required_commitment_indices
    )
    actual_lane_keys = tuple(
        (
            int(reveal.commitment_index),
            int(reveal.opening.row_index),
            int(reveal.opening.lane_index),
        )
        for reveal in relevant_reveals
    )
    if actual_lane_keys != required_lane_keys:
        raise ProofV3VerificationError(
            "GDN recurrent lanes do not match the nonce-selected heads"
        )
    openings_by_commitment: dict[int, dict[tuple[int, int], object]] = {}
    for reveal in relevant_reveals:
        commitment = commitments[reveal.commitment_index]
        verify_execution_anchor_lane_v3(
            commitment=commitment,
            opening=reveal.opening,
        )
        openings_by_commitment.setdefault(
            reveal.commitment_index, {}
        )[(reveal.opening.row_index, reveal.opening.lane_index)] = (
            reveal.opening
        )

    def stage(layer: int, suffix: str) -> Mapping[int, bytes]:
        value = opened_rows.get(f"l{layer}.{suffix}")
        if not isinstance(value, Mapping):
            raise ProofV3VerificationError(
                f"GDN runtime anchor {suffix} is missing"
            )
        return value

    results = []
    for layer in selected:
        signed = semantics.layer_for(layer)
        if signed.runtime_encoding_id != anchor_encoding_id:
            raise ProofV3VerificationError(
                "GDN runtime encoding disagrees with execution anchors"
            )
        if len(positions) > signed.max_decode_replay_rows:
            raise ProofV3VerificationError(
                "GDN decode replay exceeds the signed row bound"
            )
        parameters = signed.parameters().replay_parameters()
        selected_heads = challenge.gdn_value_heads_for(
            layer_index=layer,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        conv_width = (
            2 * parameters.num_key_heads * parameters.key_head_dim
            + parameters.num_value_heads * parameters.value_head_dim
        )
        key_width = (
            parameters.num_key_heads * parameters.key_head_dim
        )
        value_width = (
            parameters.num_value_heads * parameters.value_head_dim
        )
        group_size = (
            parameters.num_value_heads // parameters.num_key_heads
        )
        element_bytes = {
            "fp16.v1": 2,
            "bf16.v1": 2,
            "fp32.v1": 4,
        }.get(signed.runtime_encoding_id)
        if element_bytes is None:
            raise ProofV3VerificationError(
                "GDN runtime encoding is not qualified"
            )
        index_by_stage = {
            commitment.stage_id: index
            for index, commitment in enumerate(commitments)
        }
        try:
            qkvz_index = index_by_stage[f"l{layer}.gdn_qkvz_output"]
            ba_index = index_by_stage[f"l{layer}.gdn_ba_output"]
            output_index = index_by_stage[f"l{layer}.gdn_o_input"]
            conv_raw = stage(layer, "gdn_conv_prompt_boundary")[0]
        except (KeyError, ValueError) as exc:
            raise ProofV3VerificationError(
                "GDN runtime anchors do not cover the exact decode suffix"
            ) from exc
        from verallm.proof_v3.attention_anchor_binding import (
            extract_execution_anchor_range_v3,
        )

        def raw_range(
            commitment_index: int,
            position: int,
            value_start: int,
            value_count: int,
        ) -> np.ndarray:
            return _decode(
                extract_execution_anchor_range_v3(
                    commitment=commitments[commitment_index],
                    row_index=position,
                    byte_start=value_start * element_bytes,
                    byte_length=value_count * element_bytes,
                    openings=openings_by_commitment[commitment_index],
                ),
                signed.runtime_encoding_id,
            )

        qkvz = np.zeros(
            (len(positions), conv_width + value_width),
            dtype=np.float32,
        )
        ba = np.zeros(
            (len(positions), 2 * parameters.num_value_heads),
            dtype=np.float32,
        )
        actual = np.zeros(
            (len(positions), value_width),
            dtype=np.float32,
        )
        try:
            for row_slot, position in enumerate(positions):
                for head in selected_heads:
                    key_head = head // group_size
                    for value_start, width in (
                        (
                            key_head * parameters.key_head_dim,
                            parameters.key_head_dim,
                        ),
                        (
                            key_width
                            + key_head * parameters.key_head_dim,
                            parameters.key_head_dim,
                        ),
                        (
                            2 * key_width
                            + head * parameters.value_head_dim,
                            parameters.value_head_dim,
                        ),
                        (
                            conv_width
                            + head * parameters.value_head_dim,
                            parameters.value_head_dim,
                        ),
                    ):
                        qkvz[
                            row_slot, value_start:value_start + width
                        ] = raw_range(
                            qkvz_index,
                            position,
                            value_start,
                            width,
                        )
                    for value_start in (
                        head,
                        parameters.num_value_heads + head,
                    ):
                        ba[row_slot, value_start] = raw_range(
                            ba_index,
                            position,
                            value_start,
                            1,
                        )[0]
                    output_start = head * parameters.value_head_dim
                    actual[
                        row_slot,
                        output_start:
                        output_start + parameters.value_head_dim,
                    ] = raw_range(
                        output_index,
                        position,
                        output_start,
                        parameters.value_head_dim,
                    )
        except (KeyError, ValueError) as exc:
            raise ProofV3VerificationError(
                "GDN runtime anchor lanes do not cover the decode suffix"
            ) from exc
        conv = _decode(
            conv_raw, signed.conv_state_encoding_id
        ).reshape(parameters.conv_kernel_size - 1, conv_width)
        commitment_index = next(
            index
            for index, commitment in enumerate(commitments)
            if commitment.stage_id
            == f"l{layer}.gdn_recurrent_prompt_boundary"
        )
        commitment = commitments[commitment_index]
        element_bytes = (
            signed.recurrent_state_bytes
            // (
                parameters.num_value_heads
                * parameters.value_head_dim
                * parameters.key_head_dim
            )
        )
        head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * element_bytes
        )
        recurrent = np.stack(
            [
                _decode(
                    extract_execution_anchor_range_v3(
                        commitment=commitment,
                        row_index=0,
                        byte_start=head * head_bytes,
                        byte_length=head_bytes,
                        openings=openings_by_commitment[commitment_index],
                    ),
                    signed.recurrent_state_encoding_id,
                ).reshape(
                    parameters.value_head_dim,
                    parameters.key_head_dim,
                )
                for head in selected_heads
            ]
        )
        try:
            replay = _replay(
                qkvz=qkvz,
                ba=ba,
                conv_state=conv,
                recurrent_state=recurrent,
                parameters=parameters,
                runtime_encoding_id=signed.runtime_encoding_id,
                conv_state_encoding_id=signed.conv_state_encoding_id,
                recurrent_state_encoding_id=(
                    signed.recurrent_state_encoding_id
                ),
                selected_value_heads=selected_heads,
            )
        except (ProofV2TransitionError, ValueError, FloatingPointError) as exc:
            raise ProofV3VerificationError("GDN recurrence replay failed") from exc
        atol = signed.output_atol_q24 / float(1 << 24)
        rtol = signed.output_rtol_q24 / float(1 << 24)
        selected_actual = np.concatenate(
            [
                actual[
                    :,
                    head
                    * parameters.value_head_dim : (head + 1)
                    * parameters.value_head_dim,
                ]
                for head in selected_heads
            ],
            axis=1,
        )
        if selected_actual.shape != replay.out_projection_input.shape or not bool(
            np.allclose(
                selected_actual,
                replay.out_projection_input,
                atol=atol,
                rtol=rtol,
            )
        ):
            raise ProofV3VerificationError(
                f"GDN layer {layer} replay does not match the committed "
                "out-projection input"
            )
        results.append(
            EconomicGdnReplayStatsV3(
                layer_index=layer,
                row_count=len(positions),
                maximum_absolute_error=float(
                    np.max(
                        np.abs(
                            selected_actual - replay.out_projection_input
                        ),
                        initial=0.0,
                    )
                ),
            )
        )
    return tuple(results)


def verify_lean_economic_gdn_replay_v3(
    *,
    runtime_rows_by_layer: Mapping[
        int, tuple[tuple[int, bytes, bytes, bytes], ...]
    ],
    opened_rows: Mapping[str, Mapping[int, bytes]],
    challenge: EconomicChallengeV3,
    layer_kinds: Mapping[int, str],
    semantics: GdnRuntimeSemanticsV3,
    anchor_encoding_id: str,
    commitments=(),
    lane_reveals=(),
) -> tuple[EconomicGdnReplayStatsV3, ...]:
    """Replay GDN from pre-nonce state using bounded post-nonce raw rows."""

    if (
        not isinstance(runtime_rows_by_layer, Mapping)
        or not isinstance(opened_rows, Mapping)
        or not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(layer_kinds, Mapping)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
    ):
        raise ProofV3VerificationError("lean GDN replay inputs are malformed")
    selected = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer_kinds.get(layer) == "gdn"
    )
    if tuple(sorted(runtime_rows_by_layer)) != selected:
        raise ProofV3VerificationError(
            "lean GDN replay rows do not cover the selected GDN layers"
        )
    commitments = tuple(commitments)
    lane_reveals = tuple(lane_reveals)
    openings_by_commitment: dict[int, dict[tuple[int, int], object]] = {}
    if commitments:
        required_lane_keys = required_lean_gdn_checkpoint_lane_keys_v3(
            commitments=commitments,
            challenge=challenge,
            layer_kinds=layer_kinds,
            semantics=semantics,
        )
        required_indices = {
            commitment_index
            for commitment_index, _row, _lane in required_lane_keys
        }
        relevant = tuple(
            reveal
            for reveal in lane_reveals
            if int(getattr(reveal, "commitment_index", -1))
            in required_indices
        )
        actual_lane_keys = tuple(
            (
                int(reveal.commitment_index),
                int(reveal.opening.row_index),
                int(reveal.opening.lane_index),
            )
            for reveal in relevant
        )
        if actual_lane_keys != required_lane_keys:
            raise ProofV3VerificationError(
                "lean GDN recurrent lanes do not match the nonce-selected "
                "checkpoint heads"
            )
        for reveal in relevant:
            commitment = commitments[reveal.commitment_index]
            verify_execution_anchor_lane_v3(
                commitment=commitment,
                opening=reveal.opening,
            )
            openings_by_commitment.setdefault(
                int(reveal.commitment_index), {}
            )[
                (
                    int(reveal.opening.row_index),
                    int(reveal.opening.lane_index),
                )
            ] = reveal.opening
    elif lane_reveals:
        raise ProofV3VerificationError(
            "lean GDN checkpoint lanes have no commitment inventory"
        )
    results = []
    for layer in selected:
        signed = semantics.layer_for(layer)
        if signed.runtime_encoding_id != anchor_encoding_id:
            raise ProofV3VerificationError(
                "lean GDN runtime encoding disagrees with the profile"
            )
        from verallm.proof_v3.gdn_decode_corridor import (
            derive_gdn_decode_corridor_for_challenge_v3,
        )

        try:
            corridor = derive_gdn_decode_corridor_for_challenge_v3(
                challenge=challenge,
                semantics=signed,
            )
        except ProofV3Error as exc:
            raise ProofV3VerificationError(
                "lean GDN decode corridor is malformed"
            ) from exc
        positions = (
            tuple(
                range(
                    challenge.context_token_count,
                    challenge.sequence_token_count,
                )
            )
            if corridor is None
            else tuple(corridor.sequence_positions)
        )
        if not positions:
            raise ProofV3VerificationError(
                "lean GDN hard audit requires a decode window"
            )
        if len(positions) > signed.max_decode_replay_rows:
            raise ProofV3VerificationError(
                "lean GDN decode replay exceeds the signed row bound"
            )
        records = tuple(runtime_rows_by_layer[layer])
        if tuple(record[0] for record in records) != positions:
            raise ProofV3VerificationError(
                "lean GDN runtime positions are not canonical"
            )
        parameters = signed.parameters().replay_parameters()
        selected_heads = challenge.gdn_value_heads_for(
            layer_index=layer,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        key_width = parameters.num_key_heads * parameters.key_head_dim
        value_width = parameters.num_value_heads * parameters.value_head_dim
        conv_width = 2 * key_width + value_width
        qkvz_columns, ba_columns, output_columns = (
            economic_gdn_runtime_columns_v3(
                parameters=parameters,
                selected_value_heads=selected_heads,
            )
        )
        element_bytes = {
            "fp16.v1": 2,
            "bf16.v1": 2,
            "fp32.v1": 4,
        }.get(signed.runtime_encoding_id)
        if element_bytes is None or any(
            len(record[slot])
            != len(columns) * element_bytes
            for record in records
            for slot, columns in (
                (1, qkvz_columns),
                (2, ba_columns),
                (3, output_columns),
            )
        ):
            raise ProofV3VerificationError(
                "lean GDN compact runtime row geometry is inconsistent"
            )
        qkvz_compact = np.stack(
            [
                _decode(record[1], signed.runtime_encoding_id)
                for record in records
            ]
        )
        ba_compact = np.stack(
            [
                _decode(record[2], signed.runtime_encoding_id)
                for record in records
            ]
        )
        actual_compact = np.stack(
            [
                _decode(record[3], signed.runtime_encoding_id)
                for record in records
            ]
        )
        if (
            qkvz_compact.shape != (len(positions), len(qkvz_columns))
            or ba_compact.shape != (len(positions), len(ba_columns))
            or actual_compact.shape
            != (len(positions), len(output_columns))
        ):
            raise ProofV3VerificationError(
                "lean GDN compact runtime row geometry is inconsistent"
            )
        qkvz = np.zeros(
            (len(positions), conv_width + value_width),
            dtype=np.float32,
        )
        ba = np.zeros(
            (len(positions), 2 * parameters.num_value_heads),
            dtype=np.float32,
        )
        qkvz[:, qkvz_columns] = qkvz_compact
        ba[:, ba_columns] = ba_compact
        if corridor is None:
            conv_stage = f"l{layer}.gdn_conv_prompt_boundary"
            recurrent_stage = (
                f"l{layer}.gdn_recurrent_prompt_boundary"
            )
            start_state_row = 0
            end_state_row = None
        else:
            conv_stage = f"l{layer}.gdn_conv_decode_checkpoints"
            recurrent_stage = (
                f"l{layer}.gdn_recurrent_decode_checkpoints"
            )
            start_state_row = corridor.start_checkpoint_row
            end_state_row = corridor.end_checkpoint_row
        try:
            conv_raw = opened_rows[conv_stage][start_state_row]
        except KeyError as exc:
            raise ProofV3VerificationError(
                "lean GDN authenticated start state is missing"
            ) from exc
        conv = _decode(
            conv_raw, signed.conv_state_encoding_id
        ).reshape(parameters.conv_kernel_size - 1, conv_width)
        recurrent_element_bytes = (
            signed.recurrent_state_bytes
            // (
                parameters.num_value_heads
                * parameters.value_head_dim
                * parameters.key_head_dim
            )
        )
        recurrent_head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * recurrent_element_bytes
        )

        def recurrent_heads(row_index: int) -> np.ndarray:
            if not commitments:
                try:
                    recurrent_raw = opened_rows[recurrent_stage][row_index]
                except KeyError as exc:
                    raise ProofV3VerificationError(
                        "lean GDN authenticated recurrent state is missing"
                    ) from exc
                recurrent_all = _decode(
                    recurrent_raw,
                    signed.recurrent_state_encoding_id,
                ).reshape(
                    parameters.num_value_heads,
                    parameters.value_head_dim,
                    parameters.key_head_dim,
                )
                return recurrent_all[list(selected_heads)].copy()
            try:
                commitment_index = next(
                    index
                    for index, commitment in enumerate(commitments)
                    if commitment.stage_id == recurrent_stage
                )
            except StopIteration as exc:
                raise ProofV3VerificationError(
                    "lean GDN recurrent checkpoint commitment is missing"
                ) from exc
            from verallm.proof_v3.attention_anchor_binding import (
                extract_execution_anchor_range_v3,
            )

            return np.stack(
                [
                    _decode(
                        extract_execution_anchor_range_v3(
                            commitment=commitments[commitment_index],
                            row_index=row_index,
                            byte_start=head * recurrent_head_bytes,
                            byte_length=recurrent_head_bytes,
                            openings=openings_by_commitment[
                                commitment_index
                            ],
                        ),
                        signed.recurrent_state_encoding_id,
                    ).reshape(
                        parameters.value_head_dim,
                        parameters.key_head_dim,
                    )
                    for head in selected_heads
                ]
            )

        recurrent = recurrent_heads(start_state_row)
        try:
            replay = _replay(
                qkvz=qkvz,
                ba=ba,
                conv_state=conv,
                recurrent_state=recurrent,
                parameters=parameters,
                runtime_encoding_id=signed.runtime_encoding_id,
                conv_state_encoding_id=signed.conv_state_encoding_id,
                recurrent_state_encoding_id=(
                    signed.recurrent_state_encoding_id
                ),
                selected_value_heads=selected_heads,
            )
        except (ProofV2TransitionError, ValueError, FloatingPointError) as exc:
            raise ProofV3VerificationError(
                "lean GDN recurrence replay failed"
            ) from exc
        selected_actual = actual_compact
        atol = signed.output_atol_q24 / float(1 << 24)
        rtol = signed.output_rtol_q24 / float(1 << 24)
        output_mismatch = not bool(
            np.allclose(
                selected_actual,
                replay.out_projection_input,
                atol=atol,
                rtol=rtol,
            )
        )
        output_mismatch_detail: str | None = None
        if output_mismatch:
            delta = np.abs(selected_actual - replay.out_projection_input)
            worst = np.unravel_index(int(np.argmax(delta)), delta.shape)
            output_mismatch_detail = (
                f"lean GDN layer {layer} replay does not match its "
                "out-projection input "
                f"(max_delta={float(delta[worst]):.9g}, "
                f"actual={float(selected_actual[worst]):.9g}, "
                f"expected={float(replay.out_projection_input[worst]):.9g}, "
                f"atol={atol:.9g}, rtol={rtol:.9g}, index={worst})"
            )
        if end_state_row is not None:
            try:
                end_conv_raw = opened_rows[conv_stage][end_state_row]
            except KeyError as exc:
                raise ProofV3VerificationError(
                    "lean GDN authenticated end state is missing"
                ) from exc
            end_conv = _decode(
                end_conv_raw,
                signed.conv_state_encoding_id,
            ).reshape(parameters.conv_kernel_size - 1, conv_width)
            end_recurrent = recurrent_heads(end_state_row)
            conv_columns = tuple(
                column
                for column in qkvz_columns
                if column < conv_width
            )
            conv_error = np.abs(
                replay.conv_state_after[:, conv_columns]
                - end_conv[:, conv_columns]
            )
            recurrent_error = np.abs(
                replay.recurrent_state_after - end_recurrent
            )
            conv_allowed = (
                signed.conv_state_atol_q24 / float(1 << 24)
            )
            recurrent_allowed = (
                signed.recurrent_state_atol_q24 / float(1 << 24)
            )
            if bool(
                np.any(conv_error > conv_allowed)
                or np.any(recurrent_error > recurrent_allowed)
            ):
                conv_max = float(np.max(conv_error, initial=0.0))
                recurrent_max = float(
                    np.max(recurrent_error, initial=0.0)
                )
                raise ProofV3VerificationError(
                    f"lean GDN layer {layer} replay does not reach its "
                    "authenticated end checkpoint "
                    f"(conv_max_delta={conv_max:.9g}, "
                    f"conv_atol={conv_allowed:.9g}, "
                    f"recurrent_max_delta={recurrent_max:.9g}, "
                    f"recurrent_atol={recurrent_allowed:.9g})"
                )
        if output_mismatch_detail is not None:
            raise ProofV3VerificationError(output_mismatch_detail)
        results.append(
            EconomicGdnReplayStatsV3(
                layer_index=layer,
                row_count=len(positions),
                maximum_absolute_error=float(
                    np.max(
                        np.abs(
                            selected_actual - replay.out_projection_input
                        ),
                        initial=0.0,
                    )
                ),
            )
        )
    return tuple(results)


def verify_prefix_cache_gdn_replay_v3(
    *,
    runtime_rows_by_layer: Mapping[
        int, tuple[tuple[int, bytes, bytes, bytes], ...]
    ],
    windows,
    prefix_cache_lanes,
    challenge: EconomicChallengeV3,
    semantics: GdnRuntimeSemanticsV3,
    anchor_encoding_id: str,
) -> tuple[EconomicGdnReplayStatsV3, ...]:
    """Replay nonce-selected cached GDN blocks between committed states."""

    from verallm.proof_v3.prefix_cache_gdn_binding import (
        PrefixCacheGdnWindowV3,
    )

    plans = tuple(windows)
    lane_map = dict(prefix_cache_lanes)
    if (
        not isinstance(runtime_rows_by_layer, Mapping)
        or not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
        or not plans
        or not all(isinstance(plan, PrefixCacheGdnWindowV3) for plan in plans)
        or tuple(sorted(runtime_rows_by_layer))
        != tuple(plan.layer_index for plan in plans)
    ):
        raise ProofV3VerificationError(
            "prefix-cache GDN replay inputs are malformed"
        )

    def raw_range(
        *, block: int, stage_id: str, byte_start: int, byte_length: int
    ) -> bytes:
        lane_width = execution_anchor_lane_bytes_v3(stage_id)
        first = byte_start // lane_width
        last = (byte_start + byte_length - 1) // lane_width
        try:
            joined = b"".join(
                lane_map[(block, stage_id, 0, lane)]
                for lane in range(first, last + 1)
            )
        except KeyError as exc:
            raise ProofV3VerificationError(
                "prefix-cache GDN state lacks a nonce-selected lane"
            ) from exc
        offset = byte_start - first * lane_width
        result = joined[offset:offset + byte_length]
        if len(result) != byte_length:
            raise ProofV3VerificationError(
                "prefix-cache GDN state lane is truncated"
            )
        return result

    results = []
    for plan in plans:
        layer = plan.layer_index
        if plan.cached_token_count > challenge.context_token_count:
            raise ProofV3VerificationError(
                "prefix-cache GDN window exceeds the request context"
            )
        signed = semantics.layer_for(layer)
        if signed.runtime_encoding_id != anchor_encoding_id:
            raise ProofV3VerificationError(
                "prefix-cache GDN runtime encoding disagrees with the profile"
            )
        if plan.block_token_count != signed.max_prefix_cache_replay_rows:
            raise ProofV3VerificationError(
                "prefix-cache GDN replay disagrees with the signed page width"
            )
        parameters = signed.parameters().replay_parameters()
        selected_heads = challenge.gdn_value_heads_for(
            layer_index=layer,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        qkvz_columns, ba_columns, output_columns = (
            economic_gdn_runtime_columns_v3(
                parameters=parameters,
                selected_value_heads=selected_heads,
            )
        )
        records = tuple(runtime_rows_by_layer[layer])
        positions = plan.sequence_positions
        if tuple(record[0] for record in records) != positions:
            raise ProofV3VerificationError(
                "prefix-cache GDN runtime positions are not canonical"
            )
        element_bytes = {
            "fp16.v1": 2,
            "bf16.v1": 2,
            "fp32.v1": 4,
        }.get(signed.runtime_encoding_id)
        if element_bytes is None or any(
            len(record[slot]) != len(columns) * element_bytes
            for record in records
            for slot, columns in (
                (1, qkvz_columns),
                (2, ba_columns),
                (3, output_columns),
            )
        ):
            raise ProofV3VerificationError(
                "prefix-cache GDN runtime row geometry is inconsistent"
            )
        key_width = parameters.num_key_heads * parameters.key_head_dim
        value_width = parameters.num_value_heads * parameters.value_head_dim
        conv_width = 2 * key_width + value_width
        qkvz = np.zeros(
            (len(positions), conv_width + value_width), dtype=np.float32
        )
        ba = np.zeros(
            (len(positions), 2 * parameters.num_value_heads), dtype=np.float32
        )
        qkvz[:, qkvz_columns] = np.stack([
            _decode(record[1], signed.runtime_encoding_id)
            for record in records
        ])
        ba[:, ba_columns] = np.stack([
            _decode(record[2], signed.runtime_encoding_id)
            for record in records
        ])
        actual = np.stack([
            _decode(record[3], signed.runtime_encoding_id)
            for record in records
        ])

        recurrent_element_bytes = signed.recurrent_state_bytes // (
            parameters.num_value_heads
            * parameters.value_head_dim
            * parameters.key_head_dim
        )
        recurrent_head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * recurrent_element_bytes
        )
        if plan.start_state_block is None:
            conv = np.zeros(
                (parameters.conv_kernel_size - 1, conv_width),
                dtype=np.float32,
            )
            recurrent = np.zeros(
                (
                    len(selected_heads),
                    parameters.value_head_dim,
                    parameters.key_head_dim,
                ),
                dtype=np.float32,
            )
        else:
            conv = _decode(
                raw_range(
                    block=plan.start_state_block,
                    stage_id=f"l{layer}.gdn_conv_boundary",
                    byte_start=0,
                    byte_length=signed.conv_state_bytes,
                ),
                signed.conv_state_encoding_id,
            ).reshape(parameters.conv_kernel_size - 1, conv_width)
            recurrent = np.stack([
                _decode(
                    raw_range(
                        block=plan.start_state_block,
                        stage_id=f"l{layer}.gdn_recurrent_boundary",
                        byte_start=head * recurrent_head_bytes,
                        byte_length=recurrent_head_bytes,
                    ),
                    signed.recurrent_state_encoding_id,
                ).reshape(
                    parameters.value_head_dim,
                    parameters.key_head_dim,
                )
                for head in selected_heads
            ])
        try:
            replay = _replay(
                qkvz=qkvz,
                ba=ba,
                conv_state=conv,
                recurrent_state=recurrent,
                parameters=parameters,
                runtime_encoding_id=signed.runtime_encoding_id,
                conv_state_encoding_id=signed.conv_state_encoding_id,
                recurrent_state_encoding_id=signed.recurrent_state_encoding_id,
                selected_value_heads=selected_heads,
            )
        except (ProofV2TransitionError, ValueError, FloatingPointError) as exc:
            raise ProofV3VerificationError(
                "prefix-cache GDN recurrence replay failed"
            ) from exc
        output_atol = (
            signed.prefix_cache_output_atol_q24 / float(1 << 24)
        )
        output_rtol = (
            signed.prefix_cache_output_rtol_q24 / float(1 << 24)
        )
        if not bool(np.allclose(
            actual,
            replay.out_projection_input,
            atol=output_atol,
            rtol=output_rtol,
        )):
            raise ProofV3VerificationError(
                f"prefix-cache GDN layer {layer} replay does not match its "
                "out-projection input"
            )
        end_conv = _decode(
            raw_range(
                block=plan.end_state_block,
                stage_id=f"l{layer}.gdn_conv_boundary",
                byte_start=0,
                byte_length=signed.conv_state_bytes,
            ),
            signed.conv_state_encoding_id,
        ).reshape(parameters.conv_kernel_size - 1, conv_width)
        end_recurrent = np.stack([
            _decode(
                raw_range(
                    block=plan.end_state_block,
                    stage_id=f"l{layer}.gdn_recurrent_boundary",
                    byte_start=head * recurrent_head_bytes,
                    byte_length=recurrent_head_bytes,
                ),
                signed.recurrent_state_encoding_id,
            ).reshape(
                parameters.value_head_dim,
                parameters.key_head_dim,
            )
            for head in selected_heads
        ])
        conv_columns = tuple(
            column for column in qkvz_columns if column < conv_width
        )
        if (
            bool(np.any(
                np.abs(
                    replay.conv_state_after[:, conv_columns]
                    - end_conv[:, conv_columns]
                )
                > signed.prefix_cache_conv_state_atol_q24 / float(1 << 24)
            ))
            or bool(np.any(
                np.abs(replay.recurrent_state_after - end_recurrent)
                > signed.prefix_cache_recurrent_state_atol_q24
                / float(1 << 24)
            ))
        ):
            raise ProofV3VerificationError(
                f"prefix-cache GDN layer {layer} replay does not reach its "
                "authenticated block boundary"
            )
        results.append(EconomicGdnReplayStatsV3(
            layer_index=layer,
            row_count=len(positions),
            maximum_absolute_error=float(np.max(
                np.abs(actual - replay.out_projection_input),
                initial=0.0,
            )),
        ))
    return tuple(results)
