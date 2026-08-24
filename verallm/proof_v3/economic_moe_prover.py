"""Miner-side construction of bounded authenticated sparse-MoE openings."""

from __future__ import annotations

from collections.abc import Mapping

from verallm.proof_v3.economic_artifacts import (
    open_manifest_weight_range_v3,
    open_manifest_weight_row_v3,
)
from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.economic_moe_wire import (
    EconomicMoeCapacityRevealV3,
    EconomicMoeExpertRevealV3,
    EconomicMoeLayerRevealV3,
    EconomicMoeWireV3,
    encode_economic_moe_wire_v3,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.moe_projection_inventory import (
    MOE_EXPERT_DOWN_OPERATION_V3,
    MOE_EXPERT_GATE_UP_OPERATION_V3,
    MOE_ROUTER_OPERATION_V3,
    MOE_SHARED_DOWN_OPERATION_V3,
    MOE_SHARED_GATE_OPERATION_V3,
    MOE_SHARED_GATE_UP_OPERATION_V3,
    moe_projection_name_v3,
    select_capacity_experts_v3,
    select_capacity_rows_v3,
)
from verallm.proof_v3.moe_relation import canonical_qwen_router_v3
from verallm.proof_v3.moe_runtime_semantics import MoeRuntimeSemanticsV3


__all__ = ["build_economic_moe_wire_v3"]


def _runtime_row(
    *,
    economic_rows: Mapping,
    layer_index: int,
    suffix: str,
    token_position: int,
):
    """Return one replay row from the production ``economic_rows`` ABI."""

    key = (layer_index, suffix)
    try:
        records = economic_rows[key]
    except (KeyError, TypeError) as exc:
        raise ProofV3Error(
            f"economic MoE replay lacks l{layer_index}.{suffix}"
        ) from exc
    matches = tuple(
        value
        for position, value in records
        if int(position) == token_position
    )
    if len(matches) != 1:
        raise ProofV3Error(
            f"economic MoE replay row l{layer_index}.{suffix} is not unique"
        )
    return matches[0]


def _router_tensor(value, *, expert_count: int, encoding_id: str):
    try:
        import torch

        if isinstance(value, bytes):
            import numpy as np

            if encoding_id == "bf16.v1":
                words = np.frombuffer(value, dtype="<u2").astype(np.uint32)
                decoded = (words << 16).view("<f4").copy()
            elif encoding_id == "fp32.v1":
                decoded = np.frombuffer(value, dtype="<f4").copy()
            else:
                raise ValueError("unsupported router encoding")
            result = torch.from_numpy(decoded)
        else:
            result = torch.as_tensor(value).detach().reshape(-1).to("cpu")
            expected_dtype = {
                "bf16.v1": torch.bfloat16,
                "fp32.v1": torch.float32,
            }.get(encoding_id)
            if expected_dtype is None or result.dtype != expected_dtype:
                raise ValueError("router-logit tensor dtype mismatch")
        result = result.to(dtype=torch.float64).reshape(1, -1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProofV3Error("economic MoE router-logit row is malformed") from exc
    if (
        tuple(result.shape) != (1, expert_count)
        or not bool(torch.isfinite(result).all())
    ):
        raise ProofV3Error("economic MoE router-logit row is malformed")
    return result


def build_economic_moe_wire_v3(
    *,
    challenge: EconomicChallengeV3,
    semantics: MoeRuntimeSemanticsV3,
    economic_rows: Mapping,
    weight_trees: Mapping,
    manifest_chunk_size: int,
    minimum_token_position: int = 0,
) -> bytes:
    """Construct the exact nonce-selected nested MoE wire.

    Expert choices come only from the replay-captured router logits. Capacity
    experts, token positions, hidden rows and all Merkle openings are selected
    from the authenticated hard challenge and semantics.
    """

    if not isinstance(challenge, EconomicChallengeV3):
        raise ProofV3Error("economic MoE challenge has an unexpected type")
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("economic MoE semantics have an unexpected type")
    if not isinstance(economic_rows, Mapping) or not isinstance(
        weight_trees, Mapping
    ):
        raise ProofV3Error("economic MoE prover material is malformed")
    if (
        isinstance(manifest_chunk_size, bool)
        or not isinstance(manifest_chunk_size, int)
        or manifest_chunk_size <= 0
        or isinstance(minimum_token_position, bool)
        or not isinstance(minimum_token_position, int)
        or minimum_token_position < 0
    ):
        raise ProofV3Error("economic MoE prover geometry is malformed")

    semantic_layers = {item.layer_index for item in semantics.layers}
    selected_layers = tuple(
        layer
        for layer in challenge.selected_layer_indices
        if layer in semantic_layers
    )
    if not selected_layers:
        raise ProofV3Error("economic MoE challenge selected no sparse layer")

    def material(name: str, *, in_dim: int, out_dim: int):
        try:
            tree, rows, observed_in_dim = weight_trees[name]
            observed_out_dim = len(rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                f"economic MoE weight tree {name!r} is missing"
            ) from exc
        if (
            int(observed_in_dim) != in_dim
            or int(observed_out_dim) != out_dim
        ):
            raise ProofV3Error(
                f"economic MoE weight tree {name!r} has wrong geometry"
            )
        return tree, rows

    def full(name: str, *, in_dim: int, out_dim: int):
        tree, rows = material(name, in_dim=in_dim, out_dim=out_dim)
        return open_manifest_weight_range_v3(
            tree=tree,
            first_row=0,
            row_count=out_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            chunk_size=manifest_chunk_size,
            weight_tensor=rows,
        )

    def row(name: str, row_index: int, *, in_dim: int, out_dim: int):
        tree, rows = material(name, in_dim=in_dim, out_dim=out_dim)
        return open_manifest_weight_row_v3(
            tree=tree,
            row_index=row_index,
            in_dim=in_dim,
            chunk_size=manifest_chunk_size,
            weight_tensor=rows,
        )

    hidden_size = semantics.hidden_size
    expert_intermediate = semantics.expert_intermediate_size
    shared_intermediate = semantics.shared_expert_intermediate_size
    layers = []
    for layer_index in selected_layers:
        token_position = challenge.moe_token_position_for(
            layer_index=layer_index,
            minimum_position=minimum_token_position,
        )
        logits = _router_tensor(
            _runtime_row(
                economic_rows=economic_rows,
                layer_index=layer_index,
                suffix="moe_router_logits",
                token_position=token_position,
            ),
            expert_count=semantics.num_experts,
            encoding_id=semantics.router_encoding_id,
        )
        selected_ids, _routing_weights = canonical_qwen_router_v3(
            logits,
            top_k=semantics.top_k,
        )
        selected_ids = tuple(int(value) for value in selected_ids[0])
        hidden_indices = challenge.residual_cols_for(
            layer_index=layer_index,
            hidden_dim=hidden_size,
        )

        def name(operation_id: str, expert_id: int | None = None) -> str:
            return moe_projection_name_v3(
                layer_index=layer_index,
                operation_id=operation_id,
                expert_id=expert_id,
            )

        experts = tuple(
            EconomicMoeExpertRevealV3(
                expert_id=expert_id,
                gate_up_rows=full(
                    name(MOE_EXPERT_GATE_UP_OPERATION_V3, expert_id),
                    in_dim=hidden_size,
                    out_dim=2 * expert_intermediate,
                ),
                down_rows=tuple(
                    row(
                        name(MOE_EXPERT_DOWN_OPERATION_V3, expert_id),
                        hidden_index,
                        in_dim=expert_intermediate,
                        out_dim=hidden_size,
                    )
                    for hidden_index in hidden_indices
                ),
            )
            for expert_id in selected_ids
        )
        capacity = []
        for expert_id in sorted(
            select_capacity_experts_v3(
                nonce=challenge.selection_seed,
                semantics=semantics,
                layer_index=layer_index,
            )
        ):
            gate_up_index, down_index = select_capacity_rows_v3(
                nonce=challenge.selection_seed,
                semantics=semantics,
                layer_index=layer_index,
                expert_id=expert_id,
            )
            capacity.append(
                EconomicMoeCapacityRevealV3(
                    expert_id=expert_id,
                    gate_up_row=row(
                        name(MOE_EXPERT_GATE_UP_OPERATION_V3, expert_id),
                        gate_up_index,
                        in_dim=hidden_size,
                        out_dim=2 * expert_intermediate,
                    ),
                    down_row=row(
                        name(MOE_EXPERT_DOWN_OPERATION_V3, expert_id),
                        down_index,
                        in_dim=expert_intermediate,
                        out_dim=hidden_size,
                    ),
                )
            )
        layers.append(
            EconomicMoeLayerRevealV3(
                layer_index=layer_index,
                token_position=token_position,
                padding_multiple=semantics.padding_multiple,
                hidden_indices=hidden_indices,
                router_rows=full(
                    name(MOE_ROUTER_OPERATION_V3),
                    in_dim=hidden_size,
                    out_dim=semantics.num_experts,
                ),
                selected_experts=experts,
                shared_gate_rows=full(
                    name(MOE_SHARED_GATE_OPERATION_V3),
                    in_dim=hidden_size,
                    out_dim=1,
                ),
                shared_gate_up_rows=full(
                    name(MOE_SHARED_GATE_UP_OPERATION_V3),
                    in_dim=hidden_size,
                    out_dim=2 * shared_intermediate,
                ),
                shared_down_rows=tuple(
                    row(
                        name(MOE_SHARED_DOWN_OPERATION_V3),
                        hidden_index,
                        in_dim=shared_intermediate,
                        out_dim=hidden_size,
                    )
                    for hidden_index in hidden_indices
                ),
                capacity_experts=tuple(capacity),
            )
        )
    return encode_economic_moe_wire_v3(
        EconomicMoeWireV3(
            semantics_digest=semantics.digest(),
            layers=tuple(layers),
        )
    )
