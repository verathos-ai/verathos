"""
Cryptographic primitives for VeraLLM - re-exported from zkllm.
"""

from zkllm.crypto.field import (
    P,
    mod_p,
    add_f,
    sub_f,
    mul_f,
    inv_f,
    neg_f,
    pow_f,
    batch_mul_f,
)
from zkllm.crypto.merkle import MerkleTree, build_block_merkle, verify_merkle_path
from zkllm.crypto.transcript import Transcript

__all__ = [
    "P",
    "mod_p",
    "add_f",
    "sub_f",
    "mul_f",
    "inv_f",
    "neg_f",
    "pow_f",
    "batch_mul_f",
    "MerkleTree",
    "build_block_merkle",
    "verify_merkle_path",
    "Transcript",
]
