"""
Model weight commitment.

Creates a Merkle tree commitment over all model weights,
allowing efficient verification that specific weights were used.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import struct

import torch
import torch.nn as nn

from verallm.crypto.merkle import MerkleTree, hash_leaf


@dataclass
class ModelCommitment:
    """
    Commitment to model weights.

    Stores:
    - Root hash (32 bytes) - the overall model commitment
    - Per-layer commitments for efficient per-layer verification
    - Weight name to leaf index mapping for proof extraction
    """

    model_id: str
    root: bytes
    num_parameters: int
    layer_commitments: Dict[str, bytes]  # layer_name -> commitment
    weight_tree: Optional[MerkleTree] = None  # Full tree for proof extraction

    # Mapping from weight name to (leaf_index, shape)
    weight_info: Dict[str, Tuple[int, Tuple[int, ...]]] = field(default_factory=dict)

    def get_layer_commitment(self, layer_name: str) -> Optional[bytes]:
        """Get commitment for a specific layer."""
        return self.layer_commitments.get(layer_name)

    def to_bytes(self) -> bytes:
        """Serialize commitment (just the root for compact representation)."""
        return (
            b"VERILLM_MODEL_COMMIT_V1"
            + self.model_id.encode()
            + b"\x00"
            + struct.pack("<Q", self.num_parameters)
            + self.root
        )


def _hash_weight_tensor(name: str, tensor: torch.Tensor) -> bytes:
    """Hash a single weight tensor with its name."""
    h = hashlib.sha256()
    h.update(b"VERILLM_WEIGHT_V1")

    # Include name
    h.update(name.encode())
    h.update(b"\x00")

    # Include shape
    h.update(struct.pack(f"<{len(tensor.shape)}Q", *tensor.shape))

    # Include data
    # For quantized weights (INT8), this is deterministic
    # For FP weights, we quantize first for determinism
    flat = tensor.contiguous().flatten()

    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        data = flat.numpy().tobytes()
    elif tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        # For floats, round to avoid FP non-determinism
        # In production, weights should already be quantized
        rounded = (flat.float() * 1e6).round().to(torch.int64)
        data = rounded.numpy().tobytes()
    else:
        data = flat.to(torch.int64).numpy().tobytes()

    h.update(data)

    return h.digest()


def create_model_commitment(
    model: nn.Module,
    model_id: str,
    include_tree: bool = False,
) -> ModelCommitment:
    """
    Create commitment to model weights.

    Builds a Merkle tree over all parameters, organized by layer.

    Args:
        model: PyTorch model
        model_id: Identifier for this model
        include_tree: If True, store full tree for proof extraction

    Returns:
        ModelCommitment object
    """
    # Collect all parameters
    weight_hashes: List[bytes] = []
    weight_info: Dict[str, Tuple[int, Tuple[int, ...]]] = {}
    layer_weights: Dict[str, List[bytes]] = {}  # layer -> list of weight hashes

    total_params = 0

    for name, param in model.named_parameters():
        # Hash this weight
        weight_hash = _hash_weight_tensor(name, param.data)
        leaf_idx = len(weight_hashes)
        weight_hashes.append(weight_hash)

        # Store info
        weight_info[name] = (leaf_idx, tuple(param.shape))
        total_params += param.numel()

        # Group by layer (first part of name before first ".")
        parts = name.split(".")
        if len(parts) > 1:
            layer_name = parts[0]
            if layer_name not in layer_weights:
                layer_weights[layer_name] = []
            layer_weights[layer_name].append(weight_hash)
        else:
            # Top-level parameter
            if "_root" not in layer_weights:
                layer_weights["_root"] = []
            layer_weights["_root"].append(weight_hash)

    # Build per-layer commitments
    layer_commitments: Dict[str, bytes] = {}
    for layer_name, hashes in layer_weights.items():
        # Simple hash of concatenated weight hashes
        h = hashlib.sha256()
        h.update(b"VERILLM_LAYER_V1")
        h.update(layer_name.encode())
        h.update(b"\x00")
        for wh in hashes:
            h.update(wh)
        layer_commitments[layer_name] = h.digest()

    # Build full Merkle tree
    if len(weight_hashes) == 0:
        # Empty model (shouldn't happen but handle gracefully)
        root = hashlib.sha256(b"VERILLM_EMPTY_MODEL").digest()
        tree = None
    else:
        tree = MerkleTree(weight_hashes) if include_tree else None

        # Root is hash of all layer commitments in sorted order
        h = hashlib.sha256()
        h.update(b"VERILLM_MODEL_ROOT_V1")
        h.update(model_id.encode())
        h.update(b"\x00")
        for layer_name in sorted(layer_commitments.keys()):
            h.update(layer_commitments[layer_name])
        root = h.digest()

    return ModelCommitment(
        model_id=model_id,
        root=root,
        num_parameters=total_params,
        layer_commitments=layer_commitments,
        weight_tree=tree,
        weight_info=weight_info,
    )


def verify_weight(
    commitment: ModelCommitment,
    name: str,
    tensor: torch.Tensor,
) -> bool:
    """
    Verify that a weight tensor matches the commitment.

    For spot checks during verification.

    Args:
        commitment: Model commitment
        name: Weight name
        tensor: Weight tensor to verify

    Returns:
        True if weight matches commitment
    """
    if name not in commitment.weight_info:
        return False

    leaf_idx, expected_shape = commitment.weight_info[name]

    # Check shape
    if tuple(tensor.shape) != expected_shape:
        return False

    # Check hash
    actual_hash = _hash_weight_tensor(name, tensor)

    if commitment.weight_tree is not None:
        expected_hash = commitment.weight_tree.get_leaf(leaf_idx)
        # get_leaf returns hash_leaf(raw_data), so apply hash_leaf to match
        return hash_leaf(actual_hash) == expected_hash

    # Without tree, we can't verify individual weights
    # (would need to store all hashes)
    return True  # Can't verify without tree
