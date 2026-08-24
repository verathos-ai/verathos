"""Production replay adapters for the succinct proof-v3 selected trace.

The graph-integrated replay retains one execution-anchor leaf hash per
sequence row and raw bytes only for validator-selected rows.  Succinct
composition consumes the same material through projection/residual anchor
witnesses; it must never rebuild a second, unauthenticated row tree.
"""

from __future__ import annotations

from functools import wraps

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorReplayStageV3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionAnchorClaimV3,
    GoldilocksProjectionAnchorWitnessV3,
)
from zkllm.crypto.merkle import MerkleTree

__all__ = [
    "projection_anchor_from_replay_v3",
    "prove_goldilocks_selected_trace_from_runtime_v3",
]


def _chain_pcs_scoped(function):
    """Build every selected-trace column under one signed PCS profile."""

    @wraps(function)
    def scoped(*args, **kwargs):
        context = kwargs.get("context")
        if context is None:
            raise ProofV3Error(
                "selected-trace runtime lacks its validator-owned context"
            )
        from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
            pcs_query_count_v3,
        )
        from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
            pcs_coset_profile_v3,
        )

        with (
            pcs_query_count_v3(context.pcs_query_count),
            pcs_coset_profile_v3("chain"),
        ):
            return function(*args, **kwargs)

    return scoped


def projection_anchor_from_replay_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    replay_stage: ExecutionAnchorReplayStageV3,
    anchor_rows,
    source_column_offset: int,
    encoding_id: str,
) -> tuple[
    GoldilocksProjectionAnchorClaimV3,
    GoldilocksProjectionAnchorWitnessV3,
]:
    """Rebuild one authenticated anchor witness from bounded replay material."""

    if (
        not isinstance(commitment, ExecutionAnchorCommitmentV3)
        or not isinstance(replay_stage, ExecutionAnchorReplayStageV3)
        or replay_stage.stage_id != commitment.stage_id
        or replay_stage.row_count != commitment.row_count
        or replay_stage.row_width != commitment.row_width
    ):
        raise ProofV3Error(
            "selected-trace replay anchor geometry changed"
        )
    rows = tuple(anchor_rows)
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < commitment.row_count
            for row in rows
        )
    ):
        raise ProofV3Error(
            "selected-trace replay anchor rows are malformed"
        )
    retained = dict(replay_stage.selected_rows)
    if (
        len(retained) != len(replay_stage.selected_rows)
        or any(row not in retained for row in rows)
    ):
        raise ProofV3Error(
            "selected-trace replay lacks an authenticated anchor row"
        )
    leaf_hashes = tuple(
        replay_stage.leaf_hashes[offset : offset + 32]
        for offset in range(0, len(replay_stage.leaf_hashes), 32)
    )
    tree = MerkleTree.from_leaf_hashes(list(leaf_hashes))
    if tree.root != commitment.root:
        raise ProofV3Error(
            "selected-trace replay anchor root changed"
        )
    claim = GoldilocksProjectionAnchorClaimV3(
        commitment=commitment,
        anchor_rows=rows,
        source_column_offset=source_column_offset,
        encoding_id=encoding_id,
    )
    witness = GoldilocksProjectionAnchorWitnessV3(
        row_bytes_by_index=tuple((row, retained[row]) for row in rows),
        row_tree=tree,
    )
    return claim, witness


class _ReplayAnchorRegistryV3:
    """Build selected replay trees once and serve bounded witnesses.

    Pre-nonce commitments cover the complete registered execution inventory.
    Post-nonce replay deliberately retains only the validator-selected stages.
    Every stage referenced by the selected statement is still resolved through
    :meth:`_material`, which fails closed if its replay material is absent.
    """

    def __init__(self, *, commitments, replay_stages) -> None:
        commitments_t = tuple(commitments)
        stages_t = tuple(replay_stages)
        self._commitments = {
            item.stage_id: item for item in commitments_t
        }
        self._stages = {item.stage_id: item for item in stages_t}
        if (
            not commitments_t
            or len(self._commitments) != len(commitments_t)
            or not stages_t
            or len(self._stages) != len(stages_t)
            or not set(self._stages).issubset(self._commitments)
        ):
            raise ProofV3Error(
                "selected-trace replay anchor inventory changed"
            )
        self._trees: dict[str, MerkleTree] = {}

    def _material(
        self,
        commitment: ExecutionAnchorCommitmentV3,
        rows,
    ) -> tuple[tuple[tuple[int, bytes], ...], MerkleTree]:
        try:
            expected = self._commitments[commitment.stage_id]
            stage = self._stages[commitment.stage_id]
        except (AttributeError, KeyError) as exc:
            raise ProofV3Error(
                "selected-trace replay anchor is missing"
            ) from exc
        rows_t = tuple(int(row) for row in rows)
        if (
            expected != commitment
            or not rows_t
            or len(set(rows_t)) != len(rows_t)
            or stage.row_count != commitment.row_count
            or stage.row_width != commitment.row_width
            or len(stage.leaf_hashes) != 32 * commitment.row_count
        ):
            raise ProofV3Error(
                "selected-trace replay anchor geometry changed"
            )
        retained = dict(stage.selected_rows)
        if (
            len(retained) != len(stage.selected_rows)
            or any(
                row not in retained
                or len(retained[row]) != commitment.row_width
                for row in rows_t
            )
        ):
            raise ProofV3Error(
                "selected-trace replay lacks an authenticated anchor row"
            )
        tree = self._trees.get(commitment.stage_id)
        if tree is None:
            tree = MerkleTree.from_leaf_hashes(
                [
                    stage.leaf_hashes[offset : offset + 32]
                    for offset in range(
                        0,
                        len(stage.leaf_hashes),
                        32,
                    )
                ]
            )
            if tree.root != commitment.root:
                raise ProofV3Error(
                    "selected-trace replay anchor root changed"
                )
            self._trees[commitment.stage_id] = tree
        return (
            tuple((row, retained[row]) for row in rows_t),
            tree,
        )

    def projection_witness(self, anchor):
        from verallm.proof_v3.goldilocks_projection_composition import (
            GoldilocksProjectionAnchorWitnessV3,
        )

        rows, tree = self._material(
            anchor.commitment,
            anchor.anchor_rows,
        )
        return GoldilocksProjectionAnchorWitnessV3(
            row_bytes_by_index=rows,
            row_tree=tree,
        )

    def boundary_witness(self, commitment, row_index: int = 0):
        from verallm.proof_v3.goldilocks_gdn_composition import (
            GoldilocksGdnBoundaryWitnessV3,
        )

        rows, tree = self._material(commitment, (row_index,))
        return GoldilocksGdnBoundaryWitnessV3(
            row_bytes=rows[0][1],
            row_tree=tree,
            row_index=row_index,
        )

    def final_witness(self, claim):
        from verallm.proof_v3.goldilocks_final_rmsnorm import (
            GoldilocksFinalRmsnormWitnessV3,
        )

        rows, tree = self._material(
            claim.residual_anchor.commitment,
            claim.residual_anchor.anchor_rows,
        )
        return GoldilocksFinalRmsnormWitnessV3(
            claim=claim,
            residual_row_bytes=rows[0][1],
            residual_row_tree=tree,
        )


@_chain_pcs_scoped
def prove_goldilocks_selected_trace_from_runtime_v3(
    *,
    context,
    oracle_set,
    execution_anchors,
    replay_stages,
    projection_weight_rows_i8,
    projection_bias_rows=(),
    rmsnorm_weight_rows,
    embedding_rows,
    terminal_weight_rows_i8,
    terminal_logits,
    collector,
    attention_sections=(),
    attention_anchor_lane_reveals=(),
    attention_query_heads=(),
    gdn_runtime_rows_by_layer=None,
    fused=None,
):
    """Compose the complete selected trace from authenticated replay state."""

    from verallm.proof_v3.economic_commitment import EconomicOracleSetV3
    from verallm.proof_v3.goldilocks_bottom_anchor import (
        prove_goldilocks_bottom_oracle_anchor_v3,
    )
    from verallm.proof_v3.goldilocks_final_rmsnorm import (
        prove_goldilocks_final_rmsnorm_v3,
    )
    from verallm.proof_v3.goldilocks_gdn_composition import (
        GoldilocksGdnWitnessV3,
        prove_goldilocks_gdn_composition_v3,
    )
    from verallm.proof_v3.goldilocks_mlp_composition import (
        prove_goldilocks_mlp_composition_v3,
    )
    from verallm.proof_v3.goldilocks_projection_composition import (
        GoldilocksProjectionWitnessV3,
        prove_goldilocks_projection_composition_v3,
    )
    from verallm.proof_v3.goldilocks_residual_composition import (
        GoldilocksResidualStageWitnessV3,
        GoldilocksResidualWitnessV3,
        prove_goldilocks_residual_composition_v3,
    )
    from verallm.proof_v3.goldilocks_rmsnorm_composition import (
        prove_goldilocks_rmsnorm_composition_v3,
    )
    from verallm.proof_v3.goldilocks_selected_trace import (
        GoldilocksSelectedTraceContextV3,
        finalize_goldilocks_selected_trace_v3,
    )
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )
    from verallm.proof_v3.goldilocks_terminal_path import (
        prove_goldilocks_terminal_path_v3,
    )

    if (
        not isinstance(context, GoldilocksSelectedTraceContextV3)
        or not isinstance(oracle_set, EconomicOracleSetV3)
        or not isinstance(collector, BatchOpeningCollectorV3)
    ):
        raise ProofV3Error(
            "selected-trace runtime inputs have unexpected types"
        )
    projection_weights = tuple(projection_weight_rows_i8)
    if len(projection_weights) != len(context.projection_claims):
        raise ProofV3Error(
            "selected-trace projection weights are not exact"
        )
    anchors = _ReplayAnchorRegistryV3(
        commitments=execution_anchors,
        replay_stages=replay_stages,
    )
    runtime_rows_by_layer = {
        int(layer): tuple(rows)
        for layer, rows in (
            ()
            if gdn_runtime_rows_by_layer is None
            else gdn_runtime_rows_by_layer
        )
    }
    if set(runtime_rows_by_layer) != {
        claim.layer_index
        for claim in context.gdn_claims
        if claim.end_state_row is not None
    }:
        raise ProofV3Error(
            "selected-trace GDN native runtime rows are not exact"
        )

    projection_witnesses = []
    for claim in context.projection_claims:
        if (
            claim.x_anchor is not None
            or (
                claim.runtime is not None
                and claim.runtime.y_anchor is not None
            )
        ):
            raise ProofV3Error(
                "selected-trace runtime projection retained a raw anchor"
            )
        projection_witnesses.append(
            GoldilocksProjectionWitnessV3(
                claim=claim,
                committed_x=oracle_set.get(claim.x_oracle.oracle_id),
                committed_s=oracle_set.get(claim.s_oracle.oracle_id),
                committed_y=(
                    None
                    if claim.runtime is None
                    else oracle_set.get(claim.runtime.y_oracle.oracle_id)
                ),
            )
        )
    projection = prove_goldilocks_projection_composition_v3(
        validator_binding_digest=context.validator_binding_digest,
        capture_base_binding_digest=(
            context.capture_base_binding_digest
        ),
        validator_nonce=context.validator_nonce,
        witnesses=tuple(projection_witnesses),
        weight_rows_i8=projection_weights,
        fused=fused,
        external_collector=collector,
        collector_ns="projection/",
    )

    def _residual_stage(stage):
        return GoldilocksResidualStageWitnessV3(
            committed=oracle_set.get(stage.oracle.oracle_id),
            anchor=(
                None
                if stage.anchor is None
                else anchors.projection_witness(stage.anchor)
            ),
        )

    residual_witnesses = tuple(
        GoldilocksResidualWitnessV3(
            claim=claim,
            residual_in=_residual_stage(claim.residual_in),
            mid_residual=_residual_stage(claim.mid_residual),
            residual_out=_residual_stage(claim.residual_out),
        )
        for claim in context.residual_claims
    )
    residual = prove_goldilocks_residual_composition_v3(
        validator_binding_digest=context.validator_binding_digest,
        validator_nonce=context.validator_nonce,
        witnesses=residual_witnesses,
        projection_proof=projection,
        projection_claims=context.projection_claims,
        fused=fused,
        external_collector=collector,
        collector_ns="residual/",
    )

    gdn = None
    if context.gdn_claims:
        gdn = prove_goldilocks_gdn_composition_v3(
            witnesses=tuple(
                GoldilocksGdnWitnessV3(
                    claim=claim,
                    conv_state=anchors.boundary_witness(
                        claim.conv_state_anchor,
                        claim.start_state_row,
                    ),
                    recurrent_state=anchors.boundary_witness(
                        claim.recurrent_state_anchor,
                        claim.start_state_row,
                    ),
                    end_conv_state=(
                        None
                        if claim.end_state_row is None
                        else anchors.boundary_witness(
                            claim.conv_state_anchor,
                            claim.end_state_row,
                        )
                    ),
                    end_recurrent_state=(
                        None
                        if claim.end_state_row is None
                        else anchors.boundary_witness(
                            claim.recurrent_state_anchor,
                            claim.end_state_row,
                        )
                    ),
                    runtime_rows=runtime_rows_by_layer.get(
                        claim.layer_index,
                        (),
                    ),
                )
                for claim in context.gdn_claims
            ),
            projection_proof=projection,
            projection_claims=context.projection_claims,
            semantics=context.gdn_semantics,
        )
    rmsnorm = prove_goldilocks_rmsnorm_composition_v3(
        claims=context.rmsnorm_claims,
        artifacts=context.rmsnorm_artifacts,
        residual_proof=residual,
        residual_claims=context.residual_claims,
        projection_proof=projection,
        projection_claims=context.projection_claims,
    )
    mlp = prove_goldilocks_mlp_composition_v3(
        claims=context.mlp_claims,
        projection_proof=projection,
        projection_claims=context.projection_claims,
    )
    if context.bottom_claim.residual_oracle is None:
        raise ProofV3Error(
            "selected-trace runtime bottom input is not an oracle"
        )
    bottom = prove_goldilocks_bottom_oracle_anchor_v3(
        claim=context.bottom_claim,
        committed_residual=oracle_set.get(
            context.bottom_claim.residual_oracle.oracle_id
        ),
        embedding_rows=tuple(embedding_rows),
        expected_token_ids=context.bottom_token_ids,
        artifacts=context.signed_artifacts,
        capture_base_binding_digest=(
            context.capture_base_binding_digest
        ),
    )
    committed_final = oracle_set.get(
        context.final_hidden_oracle.oracle_id
    )
    final_hidden = tuple(
        committed_final.signed_value(
            context.final_hidden_row,
            column,
        )
        for column in range(context.final_hidden_oracle.col_count)
    )
    terminal = prove_goldilocks_terminal_path_v3(
        validator_binding_digest=context.validator_binding_digest,
        validator_nonce=context.validator_nonce,
        binding=context.terminal_binding,
        committed_final_hidden=committed_final,
        final_hidden_row=context.final_hidden_row,
        final_hidden_i8=final_hidden,
        logits=terminal_logits,
        observed_token=context.observed_token,
        weight_rows_i8=terminal_weight_rows_i8,
        collector=collector,
        fused=fused,
    )
    final_rmsnorm = prove_goldilocks_final_rmsnorm_v3(
        witness=anchors.final_witness(context.final_rmsnorm_claim),
        artifact=context.final_rmsnorm_artifact,
        final_hidden_i8=final_hidden,
    )
    return finalize_goldilocks_selected_trace_v3(
        bottom=bottom,
        projection=projection,
        residual=residual,
        attention_sections=tuple(attention_sections),
        attention_anchor_lane_reveals=tuple(
            attention_anchor_lane_reveals
        ),
        attention_query_heads=tuple(attention_query_heads),
        gdn=gdn,
        rmsnorm=rmsnorm,
        mlp=mlp,
        terminal=terminal,
        final_rmsnorm=final_rmsnorm,
        collector=collector,
        validator_nonce=context.validator_nonce,
        fused=fused,
        attention_capture_roots_by_layer=(
            ()
            if context.attention is None
            else tuple(
                sorted(context.attention.capture_roots_by_layer.items())
            )
        ),
        rmsnorm_weight_rows=tuple(rmsnorm_weight_rows),
        projection_bias_rows=tuple(projection_bias_rows),
    )
