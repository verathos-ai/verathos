"""
JSON serialization for VeraLLM protocol types.

Converts dataclass instances to/from JSON-safe dicts. Bytes fields are
encoded as hex strings. Supports nested dataclasses and all collection types.

Usage:
    from verallm.api.serialization import to_dict, from_dict

    d = to_dict(model_spec)           # ModelSpec -> dict
    spec = from_dict(ModelSpec, d)     # dict -> ModelSpec
"""

import dataclasses
from typing import Any, Dict, Type, get_type_hints

from zkllm.types import (
    SpotCheck,
    SpotCheckWithProof,
    SumcheckRound,
    SumcheckProof,
    MerklePath,
    GEMMBlockProof,
    GEMMProof,
    VerificationResult,
)
from verallm.types import (
    InferenceCommitment,
    LayerProof,
    GEMMChallenge,
    LayerChallenge,
    EmbeddingChallenge,
    EmbeddingProof,
    EmbeddingRowOpening,
    EmbeddingOutputOpening,
    SamplingChallenge,
    ChallengeSet,
    InferenceProofBundle,
    ModelSpec,
    ExpertChallenge,
    MoELayerChallenge,
    RouterCommitment,
    RouterLogitsOpening,
    RouterOutputRow,
    RouterLayerProof,
    SamplingProof,
)


# All known protocol dataclasses, keyed by name for deserialization
_TYPE_REGISTRY: Dict[str, Type] = {}


def _register(*classes):
    for cls in classes:
        _TYPE_REGISTRY[cls.__name__] = cls


_register(
    SpotCheck, SpotCheckWithProof, SumcheckRound, SumcheckProof,
    MerklePath, GEMMBlockProof, GEMMProof, VerificationResult,
    InferenceCommitment, LayerProof, GEMMChallenge, LayerChallenge,
    EmbeddingChallenge, EmbeddingProof, EmbeddingRowOpening, EmbeddingOutputOpening,
    ChallengeSet, InferenceProofBundle, ModelSpec,
    ExpertChallenge, MoELayerChallenge, RouterCommitment,
    RouterLogitsOpening, RouterOutputRow, RouterLayerProof,
    SamplingChallenge, SamplingProof,
)


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or primitive) to a JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _serialize_key(k): to_dict(v)
            for k, v in obj.items()
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {"__type__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            result[f.name] = to_dict(getattr(obj, f.name))
        return result
    return str(obj)


def _serialize_key(key: Any) -> str:
    """Dict keys must be strings in JSON."""
    if isinstance(key, int):
        return str(key)
    if isinstance(key, tuple):
        return ",".join(str(k) for k in key)
    return str(key)


def from_dict(cls: Type, data: Any) -> Any:
    """Reconstruct a dataclass instance from a serialized dict."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(data).__name__}")

    hints = get_type_hints(cls)
    kwargs = {}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            if f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            continue

        raw = data[f.name]
        hint = hints.get(f.name, f.type)
        kwargs[f.name] = _deserialize_field(raw, hint)

    return cls(**kwargs)


def _deserialize_field(raw: Any, hint) -> Any:
    """Deserialize a single field value given its type hint."""
    if raw is None:
        return None

    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", ())

    if origin is type(None):
        return None

    if _is_optional(hint):
        inner = args[0] if args[0] is not type(None) else args[1]
        return _deserialize_field(raw, inner)

    if hint is bytes:
        if isinstance(raw, str):
            return bytes.fromhex(raw)
        return raw

    if hint in (int, float, str, bool):
        return hint(raw)

    if origin is list:
        item_hint = args[0] if args else Any
        return [_deserialize_field(item, item_hint) for item in raw]

    if origin is tuple:
        if isinstance(raw, (list, tuple)):
            return tuple(
                _deserialize_field(item, args[i] if i < len(args) else Any)
                for i, item in enumerate(raw)
            )
        return raw

    if origin is dict:
        key_hint = args[0] if args else str
        val_hint = args[1] if len(args) > 1 else Any
        result = {}
        for k, v in raw.items():
            dk = _deserialize_dict_key(k, key_hint)
            result[dk] = _deserialize_field(v, val_hint)
        return result

    if isinstance(raw, dict):
        type_name = raw.get("__type__")
        if type_name and type_name in _TYPE_REGISTRY:
            target_cls = _TYPE_REGISTRY[type_name]
            return from_dict(target_cls, raw)
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return from_dict(hint, raw)
        if isinstance(hint, str) and hint in _TYPE_REGISTRY:
            return from_dict(_TYPE_REGISTRY[hint], raw)

    return raw


def _is_optional(hint) -> bool:
    import typing
    origin = getattr(hint, "__origin__", None)
    if origin is typing.Union:
        args = getattr(hint, "__args__", ())
        return type(None) in args and len(args) == 2
    return False


def _deserialize_dict_key(key_str: str, hint) -> Any:
    if hint is int:
        return int(key_str)
    if hint is tuple or (getattr(hint, "__origin__", None) is tuple):
        parts = key_str.split(",")
        return tuple(int(p) for p in parts)
    return key_str


# ============================================================================
# Convenience functions for specific types
# ============================================================================

def model_spec_to_dict(spec: ModelSpec) -> dict:
    d = to_dict(spec)
    d.pop("__type__", None)
    return d


def dict_to_model_spec(d: dict) -> ModelSpec:
    spec = from_dict(ModelSpec, d)
    # Infer num_experts from expert roots if not explicitly set
    if spec.num_experts == 0 and spec.expert_weight_merkle_roots:
        first_roots = next(iter(spec.expert_weight_merkle_roots.values()), [])
        spec.num_experts = len(first_roots)
    return spec


def commitment_to_dict(c: InferenceCommitment) -> dict:
    d = to_dict(c)
    d.pop("__type__", None)
    return d


def dict_to_commitment(d: dict) -> InferenceCommitment:
    return from_dict(InferenceCommitment, d)


def challenge_set_to_dict(cs: ChallengeSet) -> dict:
    d = to_dict(cs)
    d.pop("__type__", None)
    return d


def dict_to_challenge_set(d: dict) -> ChallengeSet:
    return from_dict(ChallengeSet, d)


def proof_bundle_to_dict(pb: InferenceProofBundle) -> dict:
    d = to_dict(pb)
    d.pop("__type__", None)
    return d


def dict_to_proof_bundle(d: dict) -> InferenceProofBundle:
    return from_dict(InferenceProofBundle, d)
