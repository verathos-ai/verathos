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

from importlib import import_module
from typing import TYPE_CHECKING, Any

from verallm.runtime_dependency_migration import ensure_bittensor_codec_compatibility

ensure_bittensor_codec_compatibility()

__version__ = "0.1.0"

if TYPE_CHECKING:
    from verallm.config import Config, ToleranceSpec
    from verallm.types import (
        ChallengeSet,
        InferenceCommitment,
        InferenceProofBundle,
        InferenceRequest,
        ModelSpec,
        PhaseTimings,
        VerificationResult,
    )


_LAZY_EXPORTS = {
    "Config": ("verallm.config", "Config"),
    "ToleranceSpec": ("verallm.config", "ToleranceSpec"),
    "InferenceCommitment": ("verallm.types", "InferenceCommitment"),
    "InferenceProofBundle": ("verallm.types", "InferenceProofBundle"),
    "InferenceRequest": ("verallm.types", "InferenceRequest"),
    "PhaseTimings": ("verallm.types", "PhaseTimings"),
    "ChallengeSet": ("verallm.types", "ChallengeSet"),
    "ModelSpec": ("verallm.types", "ModelSpec"),
    "VerificationResult": ("verallm.types", "VerificationResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
