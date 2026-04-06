"""
Protocol helpers: proof size estimation and auto-k computation.

These functions are used by both demo scripts and the API layer
for configuring challenge parameters and reporting proof sizes.
"""

from typing import Optional

from verallm.config import get_config
from verallm.types import InferenceProofBundle


def estimate_proof_size(proof_bundle: InferenceProofBundle) -> dict:
    """Estimate the size of the proof bundle.

    Deduplicates leaf_data for W Merkle proofs: multiple spot checks
    in the same block share identical leaf_data, so we only count each
    unique block once.
    """
    # Header size
    header_size = len(proof_bundle.commitment.to_bytes()) + 32  # beacon

    # Layer proofs
    layer_sizes = []
    total_block_proofs = 0

    # Track unique leaf_data objects across the entire proof bundle
    seen_leaf_data = set()

    for layer_proof in proof_bundle.layer_proofs:
        layer_size = 4  # layer_idx

        for gemm_proof in layer_proof.gemm_proofs:
            gemm_size = 32  # output_root

            for block_proof in gemm_proof.block_proofs:
                block_size = (
                    8 +  # bi, bj
                    32 +  # leaf_hash
                    len(block_proof.merkle_path.siblings) * 33 +  # merkle_path
                    len(block_proof.sumcheck_proof.rounds) * 24 +  # sumcheck rounds (3 * 8 bytes)
                    len(block_proof.spot_X) * 12 +  # spot_X
                    len(block_proof.spot_W) * 12    # spot_W
                )
                # W Merkle proofs - deduplicate shared leaf_data
                for spot_wp in block_proof.spot_W_with_proofs:
                    leaf_id = id(spot_wp.leaf_data)
                    if leaf_id not in seen_leaf_data:
                        # First occurrence: count full leaf_data + merkle path
                        seen_leaf_data.add(leaf_id)
                        block_size += (
                            12 +  # row, col, value
                            len(spot_wp.leaf_data) +  # raw block bytes (unique)
                            len(spot_wp.merkle_path.siblings) * 33  # merkle path
                        )
                    else:
                        # Duplicate block: only count the spot-check metadata
                        # (references same leaf_data, would be deduplicated in serialization)
                        block_size += 12  # row, col, value (reference to shared block)

                gemm_size += block_size
                total_block_proofs += 1

            layer_size += gemm_size

        layer_sizes.append(layer_size)

    total_size = header_size + sum(layer_sizes)

    return {
        "header_bytes": header_size,
        "layer_sizes": layer_sizes,
        "total_bytes": total_size,
        "total_kb": total_size / 1024,
        "num_block_proofs": total_block_proofs,
    }


def compute_auto_k(num_layers: int, target_detection: Optional[float] = None,
                   is_moe: bool = False) -> int:
    """
    Automatically compute k (layers to challenge) based on model size.

    The goal is to maintain a consistent per-inference detection probability
    regardless of model size. Default target is from config (6.25%).

    Formula: k = max(1, round(num_layers * target_detection))

    For MoE models, k=1 is acceptable because each MoE layer generates multiple
    expert proofs (k_experts * expert_gemms), providing sufficient coverage.

    Args:
        num_layers: Total number of layers in the model
        target_detection: Target per-inference detection probability (default from config)
        is_moe: Whether this is an MoE model (allows k=1)

    Returns:
        Recommended k value

    Examples:
        Dense models:
        - 12 layers (GPT-2):     k=2  (16.7% detection)
        - 32 layers (7B):        k=4  (12.5% detection)
        - 80 layers (70B):       k=10 (12.5% detection)

        MoE models (k=1 acceptable due to multi-expert proofs):
        - 16 layers (OLMoE):     k=2  (12.5% detection)
        - 24 layers (Qwen-MoE):  k=3  (12.5% detection)
    """
    if target_detection is None:
        target_detection = get_config().target_detection
    k = max(1, round(num_layers * target_detection))
    # Cap at half the layers (diminishing returns beyond that)
    k = min(k, num_layers // 2)
    return k


def compute_auto_k_experts(num_experts: int,
                           target_detection: Optional[float] = None) -> int:
    """
    Automatically compute k_experts (total unique experts to challenge per layer).

    Formula: k = max(1, round(num_experts * target_detection / k_tokens))

    k_tokens division keeps proof volume practical: candidates come from k_tokens
    sampled tokens' routing, then we cap to k_experts total per layer.

    Examples (target=6.25%, k_tokens=4):
        - 16 experts (Phi-MoE):     k=1  →  3 GEMMs/layer
        - 64 experts (OLMoE):       k=1  →  3 GEMMs/layer
        - 128 experts (Qwen3-MoE):  k=2  →  6 GEMMs/layer
        - 256 experts (MiniMax):    k=4  → 12 GEMMs/layer
    """
    cfg = get_config()
    if target_detection is None:
        target_detection = cfg.target_expert_detection
    k_tokens = cfg.k_tokens_per_expert
    k = max(1, round(num_experts * target_detection / k_tokens))
    # Cap at half the experts (diminishing returns beyond that)
    k = min(k, num_experts // 2)
    return k
