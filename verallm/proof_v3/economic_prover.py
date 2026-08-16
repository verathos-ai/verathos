"""Miner-side assembly of a canonical EconomicRecomputeProofV3 wire proof.

The miner committed every oracle pre-nonce (:mod:`economic_commitment`);
after the validator reveals its nonce (and the challenge becomes derivable),
this module opens exactly the validator-derived coordinates and assembles
the bounded wire proof.  Nothing here is trusted by the validator: the
verifier re-derives every expectation itself.
"""

from __future__ import annotations

from verallm.proof_v3.economic_artifacts import open_manifest_weight_row_v3
from verallm.proof_v3.economic_challenge import (
    EconomicChallengeV3,
    economic_selection_is_compact_v3,
    economic_selection_is_streaming_v3,
)
from verallm.proof_v3.economic_commitment import (
    EconomicOracleSetV3,
    oracle_leaf_index_v3,
)
from verallm.proof_v3.economic_challenge import (
    audited_projections_for_layer_kind_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicBoundaryOpeningV3,
    EconomicChainRevealV3,
    EconomicFinalRevealV3,
    EconomicExecutionAnchorRevealV3,
    EconomicExecutionAnchorRowV3,
    EconomicGdnLayerCouplingRevealV3,
    EconomicLayerCouplingRevealV3,
    EconomicMlpActivationRevealV3,
    EconomicProjectionRevealV3,
    EconomicRecomputeProofV3,
    bounded_byte_width_v3,
    encode_int8_row_v3,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.lean_execution_anchor import (
    lean_bottom_sequence_positions_v3,
)
from verallm.proof_v3.request import (
    execution_input_token_id_at_position_v3,
)

__all__ = [
    "build_attention_bridge_opening_v3",
    "build_economic_execution_anchor_reveals_v3",
    "build_economic_execution_anchor_reveals_from_replay_v3",
    "build_economic_execution_anchor_lane_reveals_from_replay_v3",
    "build_economic_recompute_proof_v3",
]


def build_attention_bridge_opening_v3(*, oracle_set, layer: int, token_rows):
    """Open the audited layer's o_proj INPUT oracle (``l{layer}.attn_o_x``)
    at the attention plan's sampled token positions -- the RUNTIME side of
    the output bridge.  The validator authenticates it against the same
    pre-nonce oracle commitment the projection audit uses and sets it as
    ``artifacts.attention_bridge_opening``."""

    oracle = oracle_set.get(f"l{layer}.attn_o_x")
    _indices, opening = oracle.open_rows(
        tuple(int(t) for t in token_rows), value_mode=2)
    return opening


def _row_bytes(oracle, row: int) -> bytes:
    rows_cpu = getattr(oracle, "int_rows_cpu", None)
    if rows_cpu is not None:
        # bulk row extraction; encode_int8_row_v3 still range-checks
        return encode_int8_row_v3(rows_cpu[row].tolist())
    return encode_int8_row_v3(
        tuple(
            oracle.signed_value(row, col)
            for col in range(oracle.commitment.col_count)
        )
    )


def _chain_bottom_sequence_positions_v3(
    *,
    rows,
    candidate_sequence_positions,
    lean_positions,
) -> tuple[int, ...]:
    """Compatibility wrapper for the shared canonical row mapping."""

    return lean_bottom_sequence_positions_v3(
        rows=rows,
        candidate_sequence_positions=candidate_sequence_positions,
        lean_positions=lean_positions,
    )


def build_economic_execution_anchor_reveals_v3(
    *,
    commitments,
    trees_by_stage,
    rows_by_stage,
    expected_reveals,
) -> tuple[EconomicExecutionAnchorRevealV3, ...]:
    """Assemble canonical selected-row paths from post-nonce rebuilt trees."""

    commitments = tuple(commitments)
    expected = dict(expected_reveals)
    reveals = []
    for commitment_index, commitment in enumerate(commitments):
        positions = expected.get(commitment.stage_id)
        if positions is None:
            continue
        tree = trees_by_stage.get(commitment.stage_id)
        rows = rows_by_stage.get(commitment.stage_id)
        if tree is None or rows is None:
            raise ProofV3Error(
                f"execution anchor rebuild is missing {commitment.stage_id}"
            )
        reveals.append(
            EconomicExecutionAnchorRevealV3(
                commitment_index=commitment_index,
                rows=tuple(
                    EconomicExecutionAnchorRowV3(
                        row_index=position,
                        row_bytes=bytes(rows[position]),
                        sibling_hashes=tuple(
                            digest
                            for digest, _is_left in tree.get_path(
                                position
                            ).siblings
                        ),
                    )
                    for position in positions
                ),
            )
        )
    if tuple(
        commitments[reveal.commitment_index].stage_id for reveal in reveals
    ) != tuple(stage_id for stage_id, _positions in expected_reveals):
        raise ProofV3Error(
            "execution anchor rebuild does not cover the exact selected stages"
        )
    return tuple(reveals)


def build_economic_execution_anchor_reveals_from_replay_v3(
    *,
    commitments,
    replay_stages,
    expected_reveals,
) -> tuple[EconomicExecutionAnchorRevealV3, ...]:
    """Build selected-row paths from context-bounded replay material.

    The tracker retains leaf hashes only for selected stages and raw values
    only for selected rows.  This function builds and releases one temporary
    outer tree per stage, verifies it reconstructs the already-frozen
    pre-nonce root, and emits the canonical wire openings.
    """

    from verallm.proof_v3.execution_anchor import (
        ExecutionAnchorReplayStageV3,
    )
    from verallm.proof_v3.execution_anchor_segment import (
        SparseExecutionAnchorReplayStageV3,
        reconstruct_execution_anchor_sparse_paths_v3,
    )
    from zkllm.crypto.merkle import MerkleTree

    commitments = tuple(commitments)
    replay_stages = tuple(replay_stages)
    expected_reveals = tuple(expected_reveals)
    expected = dict(expected_reveals)
    replay_by_stage = {}
    for stage in replay_stages:
        if (
            not isinstance(
                stage,
                (
                    ExecutionAnchorReplayStageV3,
                    SparseExecutionAnchorReplayStageV3,
                ),
            )
            or stage.stage_id in replay_by_stage
        ):
            raise ProofV3Error(
                "execution anchor replay stage inventory is malformed"
            )
        replay_by_stage[stage.stage_id] = stage
    expected_stage_ids = tuple(
        stage_id for stage_id, _positions in expected_reveals
    )
    if any(stage_id not in replay_by_stage for stage_id in expected_stage_ids):
        raise ProofV3Error(
            "execution anchor replay does not cover every selected stage"
        )

    reveals = []
    for commitment_index, commitment in enumerate(commitments):
        positions = expected.get(commitment.stage_id)
        if positions is None:
            continue
        stage = replay_by_stage[commitment.stage_id]
        selected_rows = dict(stage.selected_rows)
        if (
            len(selected_rows) != len(stage.selected_rows)
            or stage.row_count != commitment.row_count
            or stage.row_width != commitment.row_width
            or any(position not in selected_rows for position in positions)
        ):
            raise ProofV3Error(
                f"execution anchor replay geometry changed for "
                f"{commitment.stage_id}"
            )
        if isinstance(stage, SparseExecutionAnchorReplayStageV3):
            root, paths = reconstruct_execution_anchor_sparse_paths_v3(
                num_leaves=stage.row_count,
                leaf_hashes=stage.visible_leaf_hashes(),
                retained_nodes=stage.retained_nodes,
                opening_indices=tuple(positions),
            )
            paths_by_position = dict(zip(positions, paths, strict=True))
        else:
            leaf_hashes = [
                stage.leaf_hashes[offset:offset + 32]
                for offset in range(0, len(stage.leaf_hashes), 32)
            ]
            tree = MerkleTree.from_leaf_hashes(leaf_hashes)
            root = tree.root
            paths_by_position = {
                position: tree.get_path(position) for position in positions
            }
        if root != commitment.root:
            raise ProofV3Error(
                f"execution anchor replay root changed for "
                f"{commitment.stage_id}"
            )
        reveals.append(
            EconomicExecutionAnchorRevealV3(
                commitment_index=commitment_index,
                rows=tuple(
                    EconomicExecutionAnchorRowV3(
                        row_index=position,
                        row_bytes=selected_rows[position],
                        sibling_hashes=tuple(
                            digest
                            for digest, _is_left in (
                                paths_by_position[position].siblings
                            )
                        ),
                    )
                    for position in positions
                ),
            )
        )
    if tuple(
        commitments[reveal.commitment_index].stage_id for reveal in reveals
    ) != tuple(stage_id for stage_id, _positions in expected_reveals):
        raise ProofV3Error(
            "execution anchor replay paths do not cover the exact stages"
        )
    return tuple(reveals)


def build_economic_execution_anchor_lane_reveals_from_replay_v3(
    *,
    commitments,
    replay_stages,
    lane_keys,
):
    """Build canonical nested lane openings from a bounded second replay.

    ``lane_keys`` is the validator-derivable set returned by
    :func:`required_attention_anchor_lane_keys_v3`.  Replay material must
    contain exactly its stages and rows.  Every temporary row tree must
    reconstruct the original pre-nonce root before an opening is emitted.
    """

    from verallm.proof_v3.economic_wire import (
        EconomicExecutionAnchorLaneRevealV3,
    )
    from verallm.proof_v3.execution_anchor import (
        ExecutionAnchorLaneOpeningV3,
        ExecutionAnchorReplayStageV3,
        build_execution_anchor_lane_opening_v3,
        execution_anchor_lane_bytes_v3,
    )
    from verallm.proof_v3.execution_anchor_segment import (
        SparseExecutionAnchorReplayStageV3,
        reconstruct_execution_anchor_sparse_paths_v3,
    )
    from zkllm.crypto.merkle import MerkleTree, hash_leaf

    def _retained_lane_source(stage, row_index, lane_index):
        if isinstance(stage, SparseExecutionAnchorReplayStageV3):
            for retained_range in stage.retained_lane_ranges:
                local_index = int(row_index) - int(retained_range.start)
                source = retained_range.replay_stage
                if (
                    0 <= local_index < source.row_count
                    and int(lane_index) in source.retained_lane_indices
                ):
                    return source, local_index
            return None
        if int(lane_index) in stage.retained_lane_indices:
            return stage, int(row_index)
        return None

    commitments = tuple(commitments)
    keys = tuple(lane_keys)
    if not keys:
        return ()
    if (
        keys != tuple(sorted(set(keys)))
        or any(
            type(commitment_index) is not int
            or type(row_index) is not int
            or type(lane_index) is not int
            or commitment_index < 0
            or row_index < 0
            or lane_index < 0
            or commitment_index >= len(commitments)
            for commitment_index, row_index, lane_index in keys
        )
    ):
        raise ProofV3Error(
            "execution anchor lane replay keys are malformed"
        )
    positions_by_stage: dict[str, set[int]] = {}
    for commitment_index, row_index, _lane_index in keys:
        positions_by_stage.setdefault(
            commitments[commitment_index].stage_id,
            set(),
        ).add(row_index)
    replay_by_stage = {}
    for stage in tuple(replay_stages):
        if (
            not isinstance(
                stage,
                (
                    ExecutionAnchorReplayStageV3,
                    SparseExecutionAnchorReplayStageV3,
                ),
            )
            or stage.stage_id in replay_by_stage
        ):
            raise ProofV3Error(
                "execution anchor lane replay inventory is malformed"
            )
        replay_by_stage[stage.stage_id] = stage
    if any(stage_id not in replay_by_stage for stage_id in positions_by_stage):
        raise ProofV3Error(
            "execution anchor lane replay does not cover every selected stage"
        )

    sources = {}
    for stage_id, positions in positions_by_stage.items():
        stage = replay_by_stage[stage_id]
        commitment = next(
            item for item in commitments if item.stage_id == stage_id
        )
        selected_rows = dict(stage.selected_rows)
        sparse = isinstance(stage, SparseExecutionAnchorReplayStageV3)
        requested = tuple(
            (row_index, lane_index)
            for commitment_index, row_index, lane_index in keys
            if commitments[commitment_index].stage_id == stage_id
        )
        if (
            len(selected_rows) != len(stage.selected_rows)
            or stage.row_count != commitment.row_count
            or stage.row_width != commitment.row_width
            or any(
                row_index not in selected_rows
                and _retained_lane_source(
                    stage,
                    row_index,
                    lane_index,
                )
                is None
                for row_index, lane_index in requested
            )
        ):
            raise ProofV3Error(
                f"execution anchor lane replay geometry changed for "
                f"{stage_id}"
            )
        if sparse:
            root, paths = reconstruct_execution_anchor_sparse_paths_v3(
                num_leaves=stage.row_count,
                leaf_hashes=stage.visible_leaf_hashes(),
                retained_nodes=stage.retained_nodes,
                opening_indices=tuple(sorted(positions)),
            )
            paths_by_position = dict(
                zip(sorted(positions), paths, strict=True)
            )
            tree = None
        else:
            tree = MerkleTree.from_leaf_hashes(
                [
                    stage.leaf_hashes[offset:offset + 32]
                    for offset in range(0, len(stage.leaf_hashes), 32)
                ]
            )
            root = tree.root
            paths_by_position = {
                position: tree.get_path(position) for position in positions
            }
        if root != commitment.root:
            raise ProofV3Error(
                f"execution anchor lane replay root changed for {stage_id}"
            )
        sources[stage_id] = (
            tree,
            paths_by_position,
            selected_rows,
            stage,
        )

    reveals = []
    for commitment_index, row_index, lane_index in keys:
        commitment = commitments[commitment_index]
        tree, paths_by_position, rows, stage = sources[
            commitment.stage_id
        ]
        if row_index in rows:
            if tree is not None:
                opening = build_execution_anchor_lane_opening_v3(
                    commitment=commitment,
                    row_index=row_index,
                    row_bytes=rows[row_index],
                    row_tree=tree,
                    lane_index=lane_index,
                )
            else:
                lane_bytes = execution_anchor_lane_bytes_v3(
                    commitment.stage_id
                )
                padded = rows[row_index] + bytes(
                    (-len(rows[row_index])) % lane_bytes
                )
                lane_values = tuple(
                    padded[offset : offset + lane_bytes]
                    for offset in range(0, len(padded), lane_bytes)
                )
                lane_tree = MerkleTree.from_leaf_hashes(
                    [hash_leaf(value) for value in lane_values]
                )
                visible = stage.visible_leaf_hashes()
                if (
                    lane_index >= len(lane_values)
                    or lane_tree.root != visible.get(row_index)
                ):
                    raise ProofV3Error(
                        "sparse execution anchor selected lane changed "
                        "after replay"
                    )
                opening = ExecutionAnchorLaneOpeningV3(
                    row_index=row_index,
                    lane_index=lane_index,
                    lane_bytes=lane_values[lane_index],
                    lane_sibling_hashes=tuple(
                        sibling
                        for sibling, _is_left in lane_tree.get_path(
                            lane_index
                        ).siblings
                    ),
                    row_sibling_hashes=tuple(
                        sibling
                        for sibling, _is_left in paths_by_position[
                            row_index
                        ].siblings
                    ),
                )
        else:
            lane_bytes = execution_anchor_lane_bytes_v3(
                commitment.stage_id
            )
            lane_count = (
                commitment.row_width + lane_bytes - 1
            ) // lane_bytes
            source_record = _retained_lane_source(
                stage,
                row_index,
                lane_index,
            )
            if source_record is None:
                raise ProofV3Error(
                    "execution anchor retained lane is unavailable"
                )
            retained_stage, retained_row_index = source_record
            try:
                retained_slot = retained_stage.retained_lane_indices.index(
                    lane_index
                )
            except ValueError as exc:
                raise ProofV3Error(
                    "execution anchor retained lane is unavailable"
                ) from exc
            hash_base = retained_row_index * lane_count * 32
            lane_hashes = [
                retained_stage.retained_lane_hashes[
                    hash_base + index * 32:
                    hash_base + (index + 1) * 32
                ]
                for index in range(lane_count)
            ]
            value_base = (
                (
                    retained_row_index
                    * len(retained_stage.retained_lane_indices)
                    + retained_slot
                )
                * lane_bytes
            )
            lane_value = retained_stage.retained_lane_values[
                value_base:value_base + lane_bytes
            ]
            if (
                len(lane_value) != lane_bytes
                or hash_leaf(lane_value) != lane_hashes[lane_index]
            ):
                raise ProofV3Error(
                    "execution anchor retained lane changed after capture"
                )
            lane_tree = MerkleTree.from_leaf_hashes(lane_hashes)
            expected_row_leaf = (
                stage.visible_leaf_hashes().get(row_index)
                if sparse
                else stage.leaf_hashes[
                    row_index * 32:(row_index + 1) * 32
                ]
            )
            if lane_tree.root != expected_row_leaf:
                raise ProofV3Error(
                    "execution anchor retained lane tree changed after capture"
                )
            lane_path = lane_tree.get_path(lane_index)
            row_path = (
                paths_by_position[row_index]
                if sparse
                else tree.get_path(row_index)
            )
            opening = ExecutionAnchorLaneOpeningV3(
                row_index=row_index,
                lane_index=lane_index,
                lane_bytes=lane_value,
                lane_sibling_hashes=tuple(
                    sibling
                    for sibling, _is_left in lane_path.siblings
                ),
                row_sibling_hashes=tuple(
                    sibling
                    for sibling, _is_left in row_path.siblings
                ),
            )
        reveals.append(
            EconomicExecutionAnchorLaneRevealV3(
                commitment_index=commitment_index,
                opening=opening,
            )
        )
    return tuple(reveals)


def build_economic_recompute_proof_v3(
    *,
    oracle_set: EconomicOracleSetV3,
    weight_trees: dict,
    manifest_chunk_size: int,
    challenge: EconomicChallengeV3,
    layer_universe,
    layer_kinds,
    envelope_digest: bytes,
    profile_digest: bytes,
    signed_bound_digest: bytes,
    capture_chain_digest: bytes,
    static_manifest_digest: bytes | None = None,
    lm_head_catalog_binding=None,
    prompt_token_ids,
    observed_output_token_ids,
    attention_section=None,
    execution_anchors=(),
    execution_anchor_reveals=(),
    execution_anchor_lane_reveals=(),
    lean_projection_catalog=None,
    lm_head_argmax_top_k: int = 1,
    attention_qkv_columns_by_layer=None,
    attention_kv_positions_by_layer=None,
    gdn_runtime_rows_by_layer=None,
    gdn_norm_source_rows_by_layer=None,
    mlp_activation_rows_by_layer=None,
    require_exact_mlp_activation: bool = False,
    gdn_output_columns_by_key=None,
    gdn_decode_positions_by_layer=None,
    gdn_prefix_positions_by_layer=None,
    selected_projection_rows=None,
    prefix_cache=None,
) -> EconomicRecomputeProofV3:
    """Open the validator-derived coordinates and assemble the wire proof.

    ``weight_trees[name]`` is ``(tree, int8_rows, in_dim)`` for every signed
    manifest entry the miner must open (``l{L}.qkv``, ``embed_tokens``,
    ``lm_head``) -- built from the weights the miner claims to serve.
    """

    import os
    import time

    trace = os.environ.get("VERATHOS_ATTN_TRACE") == "1"
    phase_started = time.perf_counter()
    total_started = phase_started

    def _mark_phase(name: str) -> None:
        nonlocal phase_started
        now = time.perf_counter()
        if trace:
            print(
                f"[PROOF-V3-ASSEMBLY] phase={name} "
                f"seconds={now - phase_started:.3f}",
                flush=True,
            )
        phase_started = now

    layer_universe = tuple(sorted(int(layer) for layer in layer_universe))
    if not layer_universe:
        raise ProofV3Error("economic prover needs a layer universe")
    layer_kinds = {
        int(layer): str(kind) for layer, kind in dict(layer_kinds).items()
    }
    if tuple(sorted(layer_kinds)) != layer_universe:
        raise ProofV3Error(
            "economic prover layer kinds do not cover the exact layer universe"
        )
    for kind in layer_kinds.values():
        audited_projections_for_layer_kind_v3(kind)

    def _tree(name: str):
        if name not in weight_trees:
            raise ProofV3Error(f"weight tree {name!r} is missing")
        return weight_trees[name]

    def _weight_row(name: str, row_index: int):
        tree, int8_rows, in_dim = _tree(name)
        return open_manifest_weight_row_v3(
            tree=tree,
            row_index=row_index,
            in_dim=in_dim,
            chunk_size=manifest_chunk_size,
            weight_tensor=int8_rows,
        )

    selected_projection_rows_supplied = selected_projection_rows is not None
    try:
        selected_projection_rows = (
            {}
            if selected_projection_rows is None
            else dict(selected_projection_rows)
        )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "economic prover selected projection rows are malformed"
        ) from exc

    def _projection_rows(name: str):
        if selected_projection_rows_supplied:
            try:
                return selected_projection_rows[name]
            except KeyError as exc:
                raise ProofV3Error(
                    "economic prover selected projection inventory is "
                    "incomplete"
                ) from exc
        _weight_tree, rows, _weight_in_dim = _tree(name)
        return rows

    # -- projection reveals for the challenge-selected layers ---------------
    # Audit the exact architecture-specific projection set for every selected
    # layer. Full-attention and GDN coupling sections are deliberately
    # separate so neither can be decoded or verified under the other's
    # semantics.
    projections = []
    lean_claims = []
    lean_claim_weights = []
    succinct_witnesses = []
    succinct_claim_weights = []
    couplings = []
    gdn_couplings = []
    sampled_tokens = tuple(challenge.sampled_token_rows)
    streaming = economic_selection_is_streaming_v3(
        challenge.selection_abi_id
    )
    lean_mode = lean_projection_catalog is not None
    complete_projection = (
        lean_mode and challenge.full_row_projection_audit
    )
    succinct_projection = (
        lean_mode
        and economic_selection_is_compact_v3(
            challenge.selection_abi_id
        )
        and not complete_projection
    )
    if selected_projection_rows_supplied:
        expected_projection_names = {
            f"l{layer}.{manifest_suffix}"
            for layer in challenge.selected_layer_indices
            for _x_suffix, _s_suffix, manifest_suffix in (
                audited_projections_for_layer_kind_v3(layer_kinds[layer])
            )
        }
        if set(selected_projection_rows) != expected_projection_names:
            raise ProofV3Error(
                "economic prover selected projection inventory is not exact"
            )
    attention_qkv_columns_by_layer = {
        int(layer): tuple(int(column) for column in columns)
        for layer, columns in (
            () if attention_qkv_columns_by_layer is None
            else attention_qkv_columns_by_layer
        )
    }
    gdn_runtime_rows_by_layer = {
        int(layer): tuple(rows)
        for layer, rows in (
            () if gdn_runtime_rows_by_layer is None
            else gdn_runtime_rows_by_layer
        )
    }
    gdn_norm_source_rows_by_layer = {
        int(layer): tuple(rows)
        for layer, rows in (
            ()
            if gdn_norm_source_rows_by_layer is None
            else gdn_norm_source_rows_by_layer
        )
    }
    mlp_activation_rows_by_layer = {
        int(layer): tuple(rows)
        for layer, rows in (
            ()
            if mlp_activation_rows_by_layer is None
            else mlp_activation_rows_by_layer
        )
    }
    gdn_output_columns_by_key = {
        (int(layer), str(projection)): tuple(
            int(column) for column in columns
        )
        for layer, projection, columns in (
            () if gdn_output_columns_by_key is None
            else gdn_output_columns_by_key
        )
    }
    gdn_decode_positions_by_layer = {
        int(layer): tuple(int(position) for position in positions)
        for layer, positions in (
            ()
            if gdn_decode_positions_by_layer is None
            else gdn_decode_positions_by_layer
        )
    }
    gdn_prefix_positions_by_layer = {
        int(layer): tuple(int(position) for position in positions)
        for layer, positions in (
            ()
            if gdn_prefix_positions_by_layer is None
            else gdn_prefix_positions_by_layer
        )
    }
    if any(
        layer not in challenge.selected_layer_indices
        or layer_kinds.get(layer) != "gdn"
        or not positions
        or positions != tuple(sorted(set(positions)))
        or tuple(
            int(record[0])
            for record in gdn_runtime_rows_by_layer.get(layer, ())
        )
        != positions
        for layer, positions in gdn_decode_positions_by_layer.items()
    ):
        raise ProofV3Error(
            "checkpointed GDN projection rows are malformed"
        )
    if any(
        layer not in challenge.selected_layer_indices
        or layer_kinds.get(layer) != "gdn"
        or not positions
        or positions != tuple(sorted(set(positions)))
        or any(
            position < 0 or position >= challenge.context_token_count
            for position in positions
        )
        or not set(positions).issubset({
            int(record[0])
            for record in gdn_runtime_rows_by_layer.get(layer, ())
        })
        for layer, positions in gdn_prefix_positions_by_layer.items()
    ):
        raise ProofV3Error(
            "prefix-cache GDN projection rows are malformed"
        )
    lean_tokens_by_layer = {}
    lean_positions_by_layer = {}
    if lean_mode:
        from verallm.proof_v3.lean_projection_fold import (
            LeanProjectionCatalogV3,
        )

        if not isinstance(lean_projection_catalog, LeanProjectionCatalogV3):
            raise ProofV3Error(
                "lean projection catalog has an unexpected type"
            )
        from verallm.proof_v3.lean_execution_anchor import (
            lean_projection_row_layouts_v3,
        )

        anchors = tuple(execution_anchors)
        if attention_kv_positions_by_layer is None:
            kv_positions_by_layer = {}
            for reveal in execution_anchor_reveals:
                try:
                    stage_id = anchors[reveal.commitment_index].stage_id
                except (IndexError, AttributeError) as exc:
                    raise ProofV3Error(
                        "lean execution-anchor reveal is malformed"
                    ) from exc
                if stage_id.endswith(".attention_kv_output"):
                    layer_text = stage_id.split(".", 1)[0]
                    kv_positions_by_layer[int(layer_text[1:])] = tuple(
                        int(row.row_index) for row in reveal.rows
                    )
        else:
            try:
                kv_positions_by_layer = {
                    int(layer): tuple(int(position) for position in positions)
                    for layer, positions in attention_kv_positions_by_layer
                }
            except (TypeError, ValueError) as exc:
                raise ProofV3Error(
                    "lean attention K/V position map is malformed"
                ) from exc
        lean_layouts = lean_projection_row_layouts_v3(
            challenge=challenge,
            layer_indices=layer_universe,
            attention_kv_positions_by_layer=kv_positions_by_layer,
            gdn_decode_positions_by_layer=(
                gdn_decode_positions_by_layer or None
            ),
            gdn_prefix_positions_by_layer=(
                gdn_prefix_positions_by_layer or None
            ),
        )
        lean_tokens_by_layer = {
            layer: tuple(row_index for _position, row_index in layout)
            for layer, layout in lean_layouts
        }
        lean_positions_by_layer = {
            layer: {
                row_index: position for position, row_index in layout
            }
            for layer, layout in lean_layouts
        }
        if require_exact_mlp_activation:
            expected_mlp_layers = tuple(
                sorted(int(layer) for layer in challenge.selected_layer_indices)
            )
            if tuple(sorted(mlp_activation_rows_by_layer)) != expected_mlp_layers:
                raise ProofV3Error(
                    "lean MLP activation replay does not cover the exact "
                    "selected layer inventory"
                )
            for layer in expected_mlp_layers:
                expected_positions = tuple(
                    position
                    for position, _row_index in dict(lean_layouts)[layer]
                )
                actual_positions = tuple(
                    int(record[0])
                    for record in mlp_activation_rows_by_layer[layer]
                )
                if actual_positions != tuple(sorted(expected_positions)):
                    raise ProofV3Error(
                        "lean MLP activation replay positions are incomplete "
                        f"for layer {layer}"
                    )
        elif mlp_activation_rows_by_layer:
            raise ProofV3Error(
                "legacy lean proof must not carry exact MLP activation cells"
            )
    elif mlp_activation_rows_by_layer:
        raise ProofV3Error(
            "non-lean proof must not carry MLP activation replay cells"
        )
    for layer in sorted(challenge.selected_layer_indices):
        tokens = (
            lean_tokens_by_layer[layer]
            if lean_mode
            else sampled_tokens
        )
        projection_specs = audited_projections_for_layer_kind_v3(
            layer_kinds[layer]
        )
        for x_suffix, s_suffix, manifest_suffix in projection_specs:
            x_oracle = oracle_set.get(f"l{layer}.{x_suffix}")
            s_oracle = oracle_set.get(f"l{layer}.{s_suffix}")
            complete_output = complete_projection
            required_runtime_columns = tuple(
                sorted(
                    set(
                        gdn_output_columns_by_key.get(
                            (layer, manifest_suffix), ()
                        )
                    )
                    | (
                        set(attention_qkv_columns_by_layer.get(layer, ()))
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
                                inter_dim=s_oracle.commitment.col_count // 2,
                            )
                            for output in (
                                column,
                                s_oracle.commitment.col_count // 2 + column,
                            )
                        }
                        if (
                            require_exact_mlp_activation
                            and manifest_suffix == "gate_up"
                            and s_oracle.commitment.col_count % 2 == 0
                        )
                        else set()
                    )
                )
            )
            kv_dim = (
                oracle_set.get(f"l{layer}.k_cache").commitment.col_count
                if (
                    layer_kinds[layer] == "full_attention"
                    and manifest_suffix == "qkv"
                )
                else 0
            )
            projection_binding_columns = getattr(
                challenge,
                "projection_binding_columns_for",
                None,
            )
            if callable(projection_binding_columns):
                sampled_outs = projection_binding_columns(
                    layer_index=layer,
                    layer_kind=layer_kinds[layer],
                    projection=manifest_suffix,
                    out_dim=s_oracle.commitment.col_count,
                    kv_dim=kv_dim,
                    required_runtime_columns=required_runtime_columns,
                )
            else:
                # Small reference fixtures may provide only the historical
                # output-cell selector. Production challenges always use the
                # canonical method above; preserve the same exact union here.
                sampled_outs = tuple(
                    sorted(
                        set(
                            challenge.out_cells_for(
                                layer_index=layer,
                                out_dim=s_oracle.commitment.col_count,
                                projection=manifest_suffix,
                            )
                        )
                        | set(required_runtime_columns)
                    )
                )
            if any(
                out < 0 or out >= s_oracle.commitment.col_count
                for out in sampled_outs
            ):
                raise ProofV3Error(
                    f"l{layer}.{manifest_suffix} projection output "
                    "coordinates exceed the registered output dimension"
                )
            outs = (
                tuple(range(s_oracle.commitment.col_count))
                if complete_output
                else sampled_outs
            )
            _x_indices, x_opening = x_oracle.open_rows(tokens, value_mode=0)
            if complete_output:
                _s_indices, s_opening = s_oracle.open_rows(
                    tokens,
                    value_mode=3,
                    bounded_width=bounded_byte_width_v3(
                        x_oracle.commitment.col_count
                    ),
                )
            else:
                _s_indices, s_opening = s_oracle.open_cells(
                    tuple((token, out) for token in tokens for out in outs),
                    value_mode=3,
                    bounded_width=bounded_byte_width_v3(
                        x_oracle.commitment.col_count
                    ),
                )
            manifest_name = f"l{layer}.{manifest_suffix}"
            if complete_output or succinct_projection:
                from verallm.proof_v3.lean_projection_batch import (
                    LeanProjectionBatchClaimV3,
                )
                from verallm.proof_v3.lean_projection_fold import (
                    LeanProjectionCatalogV3,
                    lean_projection_operation_key_v3,
                )

                if not isinstance(
                    lean_projection_catalog,
                    LeanProjectionCatalogV3,
                ):
                    raise ProofV3Error(
                        "lean projection catalog has an unexpected type"
                    )
                operation = lean_projection_catalog.operation(
                    lean_projection_operation_key_v3(
                        layer_index=layer,
                        projection=manifest_suffix,
                    )
                )
                weight_rows_i8 = _projection_rows(manifest_name)
                for token in tokens:
                    x_row = tuple(
                        int(value)
                        for value in (
                            x_oracle.int_rows_cpu[token].tolist()
                            if x_oracle.int_rows_cpu is not None
                            else (
                                x_oracle.signed_value(token, column)
                                for column in range(
                                    x_oracle.commitment.col_count
                                )
                            )
                        )
                    )
                    s_row = tuple(
                        int(value)
                        for value in (
                            s_oracle.int_rows_cpu[token].tolist()
                            if s_oracle.int_rows_cpu is not None
                            else (
                                s_oracle.signed_value(token, column)
                                for column in range(
                                    s_oracle.commitment.col_count
                                )
                            )
                        )
                    )
                    if complete_output:
                        lean_claims.append(
                            LeanProjectionBatchClaimV3(
                                operation=operation,
                                input_row_i8=x_row,
                                surrogate_output_i64=s_row,
                            )
                        )
                        lean_claim_weights.append(weight_rows_i8)
                    else:
                        from verallm.proof_v3.succinct_projection_batch import (
                            SuccinctProjectionClaimV3,
                            SuccinctProjectionWitnessV3,
                        )

                        claim = SuccinctProjectionClaimV3(
                            operation=operation,
                            input_row_i8=x_row,
                            surrogate_oracle=s_oracle.commitment,
                            row_index=token,
                            output_columns=sampled_outs,
                        )
                        succinct_witnesses.append(
                            SuccinctProjectionWitnessV3(
                                claim=claim,
                                surrogate_output_i64=s_row,
                                committed_surrogate=s_oracle,
                            )
                        )
                        succinct_claim_weights.append(weight_rows_i8)
            projections.append(
                EconomicProjectionRevealV3(
                    x_oracle_index=oracle_set.index_of(f"l{layer}.{x_suffix}"),
                    s_oracle_index=oracle_set.index_of(f"l{layer}.{s_suffix}"),
                    manifest_name=manifest_name,
                    token_indices=tokens,
                    x_rows=tuple(
                        _row_bytes(x_oracle, token) for token in tokens
                    ),
                    x_opening=x_opening,
                    out_indices=(() if complete_output else outs),
                    s_opening=s_opening,
                    weight_rows=(
                        ()
                        if succinct_projection
                        or (
                            complete_output
                            and manifest_suffix == "qkv"
                        )
                        else (
                            tuple(
                                _weight_row(manifest_name, out)
                                for out in (
                                    sampled_outs if complete_output else outs
                                )
                            )
                        )
                    ),
                    complete_output=complete_output,
                    succinct_output=succinct_projection,
                )
            )

        # -- common MLP, residual and norm coupling material ----------------
        down_y = oracle_set.get(f"l{layer}.down_y")
        mid = oracle_set.get(f"l{layer}.mid_residual")
        gate_up_y = oracle_set.get(f"l{layer}.gate_up_y")

        residual_cols = challenge.residual_cols_for(
            layer_index=layer,
            hidden_dim=down_y.commitment.col_count,
        )
        down_outs = challenge.out_cells_for(
            layer_index=layer,
            out_dim=down_y.commitment.col_count,
            projection="down",
        )
        down_y_cells = tuple(
            (token, col)
            for token in tokens
            for col in sorted(set(down_outs) | set(residual_cols))
        )
        _i, down_y_opening = down_y.open_cells(
            down_y_cells,
            value_mode=2,
        )
        _i, mid_opening = mid.open_rows(tokens, value_mode=2)

        gate_up_dim = gate_up_y.commitment.col_count
        inter_dim = gate_up_dim // 2
        gu_outs = challenge.out_cells_for(
            layer_index=layer, out_dim=gate_up_dim, projection="gate_up"
        )
        mlp_cols = challenge.mlp_cols_for(
            layer_index=layer, inter_dim=inter_dim
        )
        gate_up_cells = set()
        for token in tokens:
            for out in gu_outs:
                gate_up_cells.add((token, out))
            for col in mlp_cols:
                gate_up_cells.add((token, col))
                gate_up_cells.add((token, inter_dim + col))
        # int8 capture: 1-byte transport (the chunk-dragged neighbour
        # cells make 8-byte field transport the single largest wire term)
        _i, gate_up_y_opening = gate_up_y.open_cells(
            tuple(gate_up_cells),
            value_mode=2,
        )

        # manifest-driven bias reveals: one row per audited projection that
        # has a signed "l{L}.{proj}_bias" entry (model families without
        # biased projections register none)
        bias_rows = []
        for proj_index, (_x, _s, manifest_suffix) in enumerate(
            projection_specs
        ):
            bias_name = f"l{layer}.{manifest_suffix}_bias"
            if bias_name not in weight_trees:
                continue
            bias_rows.append((
                proj_index,
                _weight_row(bias_name, 0),
            ))
        input_norm_row = _weight_row(f"l{layer}.input_norm", 0)
        post_norm_row = _weight_row(f"l{layer}.post_norm", 0)
        if layer_kinds[layer] == "full_attention":
            attn_o_y = oracle_set.get(f"l{layer}.attn_o_y")
            k_oracle = oracle_set.get(f"l{layer}.k_cache")
            v_oracle = oracle_set.get(f"l{layer}.v_cache")
            qkv_s = oracle_set.get(f"l{layer}.qkv_s")
            o_outs = challenge.out_cells_for(
                layer_index=layer,
                out_dim=attn_o_y.commitment.col_count,
                projection="o",
            )
            attn_o_cells = tuple(
                (token, col)
                for token in tokens
                for col in sorted(set(o_outs) | set(residual_cols))
            )
            _i, attn_o_y_opening = attn_o_y.open_cells(
                attn_o_cells,
                value_mode=2,
            )
            kv_dim = k_oracle.commitment.col_count
            qkv_out_dim = qkv_s.commitment.col_count
            q_width = qkv_out_dim - 2 * kv_dim
            kv_cols = challenge.kv_cols_for(
                layer_index=layer, kv_dim=kv_dim
            )
            runtime_qkv_columns = set(
                attention_qkv_columns_by_layer.get(layer, ())
            )
            prefix_kv_cols = {
                out - q_width
                for out in runtime_qkv_columns
                if q_width <= out < q_width + kv_dim
            } | {
                out - q_width - kv_dim
                for out in runtime_qkv_columns
                if q_width + kv_dim <= out < q_width + 2 * kv_dim
            }
            kv_corridor_cols = tuple(
                sorted(set(kv_cols) | prefix_kv_cols)
            )
            kv_cells = tuple(
                (token, col)
                for token in tokens
                for col in kv_corridor_cols
            )
            _i, k_opening = k_oracle.open_cells(kv_cells, value_mode=1)
            _i, v_opening = v_oracle.open_cells(kv_cells, value_mode=1)
            kv_global_outs = tuple(sorted(
                {q_width + col for col in kv_corridor_cols}
                | {q_width + kv_dim + col for col in kv_corridor_cols}
                | (
                    set(attention_qkv_columns_by_layer.get(layer, ()))
                    if lean_mode
                    else set()
                )
            ))
            _i, qkv_s_kv_opening = qkv_s.open_cells(
                tuple(
                    (token, out)
                    for token in tokens
                    for out in kv_global_outs
                ),
                value_mode=3,
                bounded_width=bounded_byte_width_v3(
                    oracle_set.get(
                        f"l{layer}.qkv_x"
                    ).commitment.col_count
                ),
            )
            qkv_kv_weight_rows = (
                ()
                if lean_mode
                else tuple(
                    _weight_row(f"l{layer}.qkv", out)
                    for out in kv_global_outs
                )
            )
            couplings.append(
                EconomicLayerCouplingRevealV3(
                    layer_index=layer,
                    attn_o_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.attn_o_y"
                    ),
                    attn_o_y_opening=attn_o_y_opening,
                    down_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.down_y"
                    ),
                    down_y_opening=down_y_opening,
                    mid_oracle_index=oracle_set.index_of(
                        f"l{layer}.mid_residual"
                    ),
                    mid_opening=mid_opening,
                    gate_up_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.gate_up_y"
                    ),
                    gate_up_y_opening=gate_up_y_opening,
                    k_oracle_index=oracle_set.index_of(
                        f"l{layer}.k_cache"
                    ),
                    k_opening=k_opening,
                    v_oracle_index=oracle_set.index_of(
                        f"l{layer}.v_cache"
                    ),
                    v_opening=v_opening,
                    qkv_s_kv_opening=qkv_s_kv_opening,
                    qkv_kv_weight_rows=qkv_kv_weight_rows,
                    input_norm_row=input_norm_row,
                    post_norm_row=post_norm_row,
                    bias_rows=tuple(bias_rows),
                )
            )
        else:
            qkvz_y = oracle_set.get(f"l{layer}.gdn_qkvz_y")
            ba_y = oracle_set.get(f"l{layer}.gdn_ba_y")
            gdn_o_y = oracle_set.get(f"l{layer}.gdn_o_y")
            _i, qkvz_y_opening = qkvz_y.open_rows(tokens, value_mode=2)
            _i, ba_y_opening = ba_y.open_rows(tokens, value_mode=2)
            gdn_o_outs = challenge.out_cells_for(
                layer_index=layer,
                out_dim=gdn_o_y.commitment.col_count,
                projection="gdn_o",
            )
            gdn_o_cells = tuple(
                (token, col)
                for token in tokens
                for col in sorted(set(gdn_o_outs) | set(residual_cols))
            )
            _i, gdn_o_y_opening = gdn_o_y.open_cells(
                gdn_o_cells,
                value_mode=2,
            )
            gdn_couplings.append(
                EconomicGdnLayerCouplingRevealV3(
                    layer_index=layer,
                    qkvz_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.gdn_qkvz_y"
                    ),
                    qkvz_y_opening=qkvz_y_opening,
                    ba_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.gdn_ba_y"
                    ),
                    ba_y_opening=ba_y_opening,
                    gdn_o_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.gdn_o_y"
                    ),
                    gdn_o_y_opening=gdn_o_y_opening,
                    down_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.down_y"
                    ),
                    down_y_opening=down_y_opening,
                    mid_oracle_index=oracle_set.index_of(
                        f"l{layer}.mid_residual"
                    ),
                    mid_opening=mid_opening,
                    gate_up_y_oracle_index=oracle_set.index_of(
                        f"l{layer}.gate_up_y"
                    ),
                    gate_up_y_opening=gate_up_y_opening,
                    input_norm_row=input_norm_row,
                    post_norm_row=post_norm_row,
                    bias_rows=tuple(bias_rows),
                    runtime_rows=gdn_runtime_rows_by_layer.get(layer, ()),
                    norm_source_rows=gdn_norm_source_rows_by_layer.get(
                        layer, ()
                    ),
                )
            )
    _mark_phase("projection-and-coupling-openings")
    lean_projection_batch_wire = b""
    succinct_projection_batch_wire = b""
    if complete_projection:
        from verallm.proof_v3.lean_projection_batch import (
            build_lean_projection_batch_reference_v3,
            encode_lean_projection_batch_v3,
        )
        from verallm.proof_v3.lean_projection_native import (
            build_lean_projection_batch_cuda_v3,
        )

        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if trace:
            print(
                "[PROOF-V3-ASSEMBLY] "
                f"projection_claims={len(lean_claims)} "
                f"distinct_operations={len({id(weight) for weight in lean_claim_weights})}",
                flush=True,
            )
        if has_cuda:
            lean_batch = build_lean_projection_batch_cuda_v3(
                validator_binding_digest=envelope_digest,
                validator_nonce=bytes(challenge.selection_seed),
                claims=tuple(lean_claims),
                weight_rows_i8=tuple(lean_claim_weights),
            )
        else:
            witnesses = []
            for claim, weights in zip(
                lean_claims,
                lean_claim_weights,
                strict=True,
            ):
                rows = (
                    weights.tolist()
                    if hasattr(weights, "tolist")
                    else weights
                )
                pad = claim.operation.padded_input_dim
                witnesses.append(
                    tuple(
                        tuple(int(value) for value in row)
                        + (0,) * (pad - len(row))
                        for row in rows
                    )
                )
            lean_batch = build_lean_projection_batch_reference_v3(
                validator_binding_digest=envelope_digest,
                validator_nonce=bytes(challenge.selection_seed),
                claims=tuple(lean_claims),
                weight_columns_i8=tuple(witnesses),
            )
        lean_projection_batch_wire = encode_lean_projection_batch_v3(
            lean_batch
        )
    elif succinct_projection:
        from verallm.proof_v3.succinct_projection_batch import (
            SuccinctProjectionWeightRowsV3,
            build_succinct_projection_batch_reference_v3,
            encode_succinct_projection_batch_v3,
        )
        from verallm.proof_v3.lean_projection_native import (
            build_succinct_projection_batch_cuda_v3,
        )

        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if trace:
            print(
                "[PROOF-V3-ASSEMBLY] "
                f"succinct_projection_claims={len(succinct_witnesses)} "
                "distinct_operations="
                f"{len({id(weight) for weight in succinct_claim_weights})}",
                flush=True,
            )
        if has_cuda:
            succinct_batch = build_succinct_projection_batch_cuda_v3(
                validator_binding_digest=envelope_digest,
                validator_nonce=bytes(challenge.selection_seed),
                witnesses=tuple(succinct_witnesses),
                weight_rows_i8=tuple(succinct_claim_weights),
            )
        else:
            padded_weight_rows = []
            for witness, weights in zip(
                succinct_witnesses,
                succinct_claim_weights,
                strict=True,
            ):
                if isinstance(weights, SuccinctProjectionWeightRowsV3):
                    padded_weight_rows.append(weights)
                    continue
                rows = (
                    weights.tolist()
                    if hasattr(weights, "tolist")
                    else weights
                )
                pad = witness.claim.operation.padded_input_dim
                padded_weight_rows.append(
                    tuple(
                        tuple(int(value) for value in row)
                        + (0,) * (pad - len(row))
                        for row in rows
                    )
                )
            succinct_batch = build_succinct_projection_batch_reference_v3(
                validator_binding_digest=envelope_digest,
                validator_nonce=bytes(challenge.selection_seed),
                witnesses=tuple(succinct_witnesses),
                weight_columns_i8=tuple(padded_weight_rows),
            )
        succinct_projection_batch_wire = (
            encode_succinct_projection_batch_v3(succinct_batch)
        )
    _mark_phase("complete-projection-batch")

    # -- chain: bottom anchor + complete boundary coverage ------------------
    first_layer = layer_universe[0]
    residual0 = oracle_set.get(f"l{first_layer}.residual_in")
    chain_bottom_rows = tuple(
        sorted(
            set(challenge.bottom_anchor_rows)
            | (
                set(
                    lean_tokens_by_layer.get(
                        first_layer,
                        challenge.sampled_token_rows,
                    )
                )
                if lean_mode and first_layer in challenge.selected_layer_indices
                else set()
            )
        )
    )
    _r0_indices, residual0_opening = residual0.open_rows(
        chain_bottom_rows,
        value_mode=(2 if lean_mode else (0 if streaming else 2)),
    )
    lean_bottom_positions = (
        lean_positions_by_layer[first_layer]
        if lean_mode and first_layer in lean_positions_by_layer
        else {}
    )
    chain_bottom_positions = lean_bottom_sequence_positions_v3(
        rows=chain_bottom_rows,
        candidate_sequence_positions=(
            challenge.candidate_sequence_positions
        ),
        lean_positions=lean_bottom_positions,
    )
    embedding_rows = tuple(
        _weight_row(
            "embed_tokens",
            execution_input_token_id_at_position_v3(
                prompt_token_ids=prompt_token_ids,
                observed_output_token_ids=observed_output_token_ids,
                sequence_position=position,
            ),
        )
        for position in chain_bottom_positions
    )
    boundary_layers = (
        tuple(sorted(challenge.selected_layer_indices))
        if streaming
        else layer_universe
    )
    boundaries = []
    for layer in boundary_layers:
        boundary_token_rows = (
            lean_tokens_by_layer[layer]
            if lean_mode
            else tuple(challenge.sampled_token_rows)
        )
        in_oracle = oracle_set.get(f"l{layer}.residual_in")
        out_oracle = oracle_set.get(f"l{layer}.residual_out")
        _in_idx, in_opening = in_oracle.open_rows(
            boundary_token_rows,
            value_mode=(2 if lean_mode else (0 if streaming else 2)),
        )
        residual_cells = tuple(
            (token, col)
            for token in boundary_token_rows
            for col in challenge.residual_cols_for(
                layer_index=layer,
                hidden_dim=out_oracle.commitment.col_count,
            )
        )
        if streaming and not lean_mode:
            _out_idx, out_opening = out_oracle.open_cells(
                residual_cells,
                value_mode=2,
            )
        else:
            _out_idx, out_opening = out_oracle.open_rows(
                boundary_token_rows,
                value_mode=2,
            )
        boundaries.append(
            EconomicBoundaryOpeningV3(
                layer_index=layer,
                in_oracle_index=oracle_set.index_of(f"l{layer}.residual_in"),
                out_oracle_index=oracle_set.index_of(f"l{layer}.residual_out"),
                in_opening=in_opening,
                out_opening=out_opening,
            )
        )
    chain = EconomicChainRevealV3(
        residual0_oracle_index=oracle_set.index_of(
            f"l{first_layer}.residual_in"
        ),
        residual0_opening=residual0_opening,
        embedding_rows=embedding_rows,
        boundaries=tuple(boundaries),
    )
    _mark_phase("chain-openings")

    # -- final: final hidden -> LM head -> observed token -------------------
    from verallm.proof_v3.economic_commitment import (
        logits_block_geometry_v3,
        logits_block_oracle_id_v3,
    )

    if not challenge.audited_decode_positions:
        raise ProofV3Error("economic prover needs an audited decode position")
    if (
        isinstance(lm_head_argmax_top_k, bool)
        or not isinstance(lm_head_argmax_top_k, int)
        or not 1 <= lm_head_argmax_top_k <= 32
    ):
        raise ProofV3Error("economic LM-head top-k policy is malformed")
    position = challenge.audited_decode_positions[0]
    final_oracle = oracle_set.get("final_hidden")
    observed_token = int(observed_output_token_ids[position])
    _lm_tree, lm_rows, _lm_in_dim = _tree("lm_head")
    vocab = len(lm_rows)
    compact_terminal = (
        economic_selection_is_compact_v3(challenge.selection_abi_id)
    )
    hidden_row = tuple(
        final_oracle.signed_value(position, column)
        for column in range(final_oracle.commitment.col_count)
    )
    full_logits: tuple[int, ...] = ()
    candidate_token_rows: tuple[int, ...] = ()
    revealed_logits: tuple[int, ...] = ()
    if compact_terminal or lm_head_catalog_binding is not None:
        import heapq

        from verallm.proof_v3.economic_lm_head_catalog_fold import (
            build_lm_head_logits_v3,
        )

        compute_rows = getattr(weight_trees, "lm_head_int8", lm_rows)
        full_logits = build_lm_head_logits_v3(
            lm_head_rows=compute_rows,
            hidden_row_int8=hidden_row,
        )
        if len(full_logits) != vocab:
            raise ProofV3Error(
                "economic LM-head compute returned the wrong vocabulary"
            )
        if compact_terminal:
            candidate_token_rows = tuple(
                sorted(
                    heapq.nsmallest(
                        min(lm_head_argmax_top_k, vocab),
                        range(vocab),
                        key=lambda row: (-full_logits[row], row),
                    )
                )
            )
            if observed_token not in candidate_token_rows:
                raise ProofV3Error(
                    "observed token is outside the signed compact top-k policy"
                )
            if challenge.full_row_projection_audit:
                revealed_logits = full_logits
    sampled_vocab_rows = challenge.vocab_rows_for(vocab_size=vocab)
    vocab_rows = tuple(
        sorted(
            set(sampled_vocab_rows)
            | {observed_token}
            | set(candidate_token_rows)
        )
    )

    # Legacy profiles open every pre-nonce logits block. Compact-v9 profiles
    # deliberately carry no such oracle: their terminal relation is selected
    # after the nonce and is represented by the sampled candidate certificate
    # or the complete escalation vector above.
    logits_openings = []
    if not compact_terminal:
        _block_cols, _block_count = logits_block_geometry_v3(
            decode_rows=final_oracle.commitment.row_count, vocab=vocab
        )
        for block in range(_block_count):
            block_oracle = oracle_set.get(logits_block_oracle_id_v3(block))
            _idx, opening = block_oracle.open_rows(
                (position,),
                value_mode=3,
                bounded_width=bounded_byte_width_v3(
                    final_oracle.commitment.col_count,
                ),
            )
            logits_openings.append(
                (
                    oracle_set.index_of(logits_block_oracle_id_v3(block)),
                    opening,
                )
            )
    _f_idx, final_opening = final_oracle.open_rows(
        (position,), value_mode=2)
    _mark_phase("final-logit-openings")
    lm_head_catalog_folds = ()
    if lm_head_catalog_binding is not None:
        from verallm.proof_v3.economic_lm_head_catalog_fold import (
            EconomicLmHeadCatalogBindingV3,
            build_lm_head_catalog_folds_cuda_v3,
            build_lm_head_catalog_folds_reference_v3,
            derive_lm_head_catalog_coefficients_v3,
            lm_head_logits_digest_v3,
        )

        if (
            not isinstance(
                lm_head_catalog_binding,
                EconomicLmHeadCatalogBindingV3,
            )
            or not isinstance(static_manifest_digest, bytes)
            or len(static_manifest_digest) != 32
        ):
            raise ProofV3Error(
                "LM-head catalog fold inputs are malformed"
            )
        if not compact_terminal or challenge.full_row_projection_audit:
            if not full_logits:
                raise ProofV3Error(
                    "LM-head catalog folds require complete revealed logits"
                )
            coefficients = derive_lm_head_catalog_coefficients_v3(
                selection_seed=bytes(challenge.selection_seed),
                envelope_digest=envelope_digest,
                manifest_digest=static_manifest_digest,
                operation_root=lm_head_catalog_binding.operation_root,
                revealed_logits_digest=lm_head_logits_digest_v3(full_logits),
                audited_position=position,
                vocab=vocab,
            )
            compute_rows = getattr(
                weight_trees,
                "lm_head_int8",
                lm_rows,
            )
            try:
                is_cuda = bool(compute_rows.is_cuda)
            except AttributeError:
                is_cuda = False
            lm_head_catalog_folds = (
                build_lm_head_catalog_folds_cuda_v3(
                    lm_head_rows=compute_rows,
                    packed_coefficients=coefficients,
                )
                if is_cuda
                else build_lm_head_catalog_folds_reference_v3(
                    lm_head_rows=compute_rows,
                    packed_coefficients=coefficients,
                )
            )
    _mark_phase("lm-head-catalog-folds")
    # final-norm link: the forwarded row that produced the audited token
    last_layer = layer_universe[-1]
    last_rout = oracle_set.get(f"l{last_layer}.residual_out")
    last_row = challenge.pool_row_for_decode_position(position)
    _l_idx, last_residual_opening = last_rout.open_rows(
        (last_row,),
        value_mode=(2 if lean_mode else (0 if streaming else 2)),
    )
    final = EconomicFinalRevealV3(
        final_oracle_index=oracle_set.index_of("final_hidden"),
        audited_position=position,
        final_opening=final_opening,
        last_residual_oracle_index=oracle_set.index_of(
            f"l{last_layer}.residual_out"
        ),
        last_residual_opening=last_residual_opening,
        final_norm_row=_weight_row("final_norm", 0),
        lm_head_rows=tuple(
            _weight_row("lm_head", row)
            for row in vocab_rows
        ),
        logits_openings=tuple(logits_openings),
        candidate_token_rows=candidate_token_rows,
        revealed_logits=revealed_logits,
        lm_head_catalog_folds=lm_head_catalog_folds,
    )
    _mark_phase("final-anchor-and-weight-openings")

    proof = EconomicRecomputeProofV3(
        commitment_envelope_digest=envelope_digest,
        execution_profile_digest=profile_digest,
        signed_bound_digest=signed_bound_digest,
        capture_chain_digest=capture_chain_digest,
        execution_anchors=tuple(execution_anchors),
        execution_anchor_reveals=tuple(execution_anchor_reveals),
        execution_anchor_lane_reveals=tuple(
            execution_anchor_lane_reveals
        ),
        oracles=oracle_set.wire_oracles(),
        projections=tuple(projections),
        couplings=tuple(couplings),
        gdn_couplings=tuple(gdn_couplings),
        mlp_activation_reveals=tuple(
            EconomicMlpActivationRevealV3(
                layer_index=layer,
                rows=mlp_activation_rows_by_layer[layer],
            )
            for layer in sorted(mlp_activation_rows_by_layer)
        ),
        chain=chain,
        final=final,
        attention=attention_section,
        lean_projection_batch_wire=lean_projection_batch_wire,
        succinct_projection_batch_wire=succinct_projection_batch_wire,
        prefix_cache=prefix_cache,
    )
    if trace:
        import zlib
        from verallm.proof_v3.economic_wire import (
            _Writer,
            _encode_prefix_cache_section_v3,
        )

        def _section_bytes(records) -> int:
            writer = _Writer()
            for record in records:
                record.encode(writer)
            return len(writer.finish())

        def _prefix_cache_bytes(section) -> int:
            if section is None:
                return 0
            writer = _Writer()
            _encode_prefix_cache_section_v3(writer, section)
            return len(writer.finish())

        raw_wire = proof.canonical_bytes()
        compressed_wire = zlib.compress(raw_wire, level=1)
        sections = {
            "anchor_reveals": _section_bytes(proof.execution_anchor_reveals),
            "anchor_lanes": _section_bytes(
                proof.execution_anchor_lane_reveals
            ),
            "projections": _section_bytes(proof.projections),
            "couplings": _section_bytes(proof.couplings),
            "gdn_couplings": _section_bytes(proof.gdn_couplings),
            "lean_batch": len(proof.lean_projection_batch_wire),
            "succinct_batch": len(
                proof.succinct_projection_batch_wire
            ),
            "chain": _section_bytes(
                () if proof.chain is None else (proof.chain,)
            ),
            "final": _section_bytes(
                () if proof.final is None else (proof.final,)
            ),
            "attention": _section_bytes(
                () if proof.attention is None else (proof.attention,)
            ),
            "prefix_cache": _prefix_cache_bytes(proof.prefix_cache),
        }
        print(
            "[PROOF-V3-ASSEMBLY] "
            f"wire_raw={len(raw_wire)} wire_compressed={len(compressed_wire)} "
            + " ".join(f"{name}={size}" for name, size in sections.items()),
            flush=True,
        )
        for reveal in proof.execution_anchor_reveals:
            commitment = proof.execution_anchors[
                reveal.commitment_index
            ]
            row_bytes = sum(len(row.row_bytes) for row in reveal.rows)
            path_bytes = sum(
                32 * len(row.sibling_hashes) for row in reveal.rows
            )
            print(
                "[PROOF-V3-ASSEMBLY] "
                f"anchor_stage={commitment.stage_id} "
                f"rows={len(reveal.rows)} row_bytes={row_bytes} "
                f"path_bytes={path_bytes}",
                flush=True,
            )
        print(
            f"[PROOF-V3-ASSEMBLY] total="
            f"{time.perf_counter() - total_started:.3f}s",
            flush=True,
        )
    return proof
