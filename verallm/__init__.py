"""
VeraLLM: Probabilistic LLM Inference Verification

Extends zkgemm sumcheck + spot-check approach to verify LLM inference.

Subpackages:
    miner/       - Core miner logic (Miner, VllmMiner)
    validator/   - Validator logic (Validator)
    registry/    - Model root computation + caching
    api/         - HTTP layer (server, client, serialization)
    crypto/      - Merkle trees, Fiat-Shamir transcripts, field ops
    prover/      - Low-level GEMM prover
    verifier/    - Low-level GEMM verifier
    challenge/   - Beacon + challenge derivation
    moe/         - Mixture-of-Experts support
    quantization/- Auto-detection of model quantization
    commitment/  - Witness commitment logic
    runtime/     - Runtime utilities
"""

__version__ = "0.1.0"

from verallm.config import Config, ToleranceSpec
from verallm.types import (
    InferenceCommitment,
    InferenceProofBundle,
    InferenceRequest,
    PhaseTimings,
    ChallengeSet,
    ModelSpec,
    VerificationResult,
)

__all__ = [
    "Config",
    "ToleranceSpec",
    "InferenceCommitment",
    "InferenceProofBundle",
    "InferenceRequest",
    "PhaseTimings",
    "ChallengeSet",
    "ModelSpec",
    "VerificationResult",
]
