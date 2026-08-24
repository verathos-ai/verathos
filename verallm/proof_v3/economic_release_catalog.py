"""Local descriptors for authority-authenticated economic proof-v3 releases."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from verallm.proof_v3.economic_release import (
    QualifiedEconomicProofV3Release,
    load_qualified_economic_proof_v3_release,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES = 64 << 10
logger = logging.getLogger(__name__)
_REQUIRED = frozenset(
    {
        "version",
        "model_id",
        "manifest",
        "execution_profile",
        "calibration_set",
        "attention_runtime_semantics",
        "runtime_encoding_id",
        "max_decode_tokens",
    }
)
_OPTIONAL = frozenset(
    {
        "gdn_runtime_semantics",
        "moe_runtime_semantics",
        "lm_head_catalog",
        "projection_manifest",
        "projection_catalog",
    }
)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3Error(
                "proof-v3 release descriptor has duplicate keys"
            )
        result[key] = value
    return result


def load_proof_v3_release_descriptor(
    path,
) -> tuple[Path, dict[str, object]]:
    """Load one bounded local release descriptor without qualifying it."""

    source = Path(path).expanduser().resolve()
    try:
        size = source.stat().st_size
        if not 0 < size <= MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES:
            raise ProofV3Error(
                "proof-v3 release descriptor size is out of range"
            )
        encoded = source.read_bytes()
    except ProofV3Error:
        raise
    except OSError as exc:
        raise ProofV3Error(
            "proof-v3 release descriptor could not be loaded"
        ) from exc
    if len(encoded) != size:
        raise ProofV3Error(
            "proof-v3 release descriptor changed while loading"
        )
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
        )
    except ProofV3Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProofV3Error(
            "proof-v3 release descriptor is not valid JSON"
        ) from exc
    if not isinstance(value, dict) or frozenset(value) != _REQUIRED | (
        frozenset(value) & _OPTIONAL
    ):
        raise ProofV3Error(
            "proof-v3 release descriptor has an unexpected inventory"
        )
    if type(value.get("version")) is not int or value["version"] != 1:
        raise ProofV3Error(
            "proof-v3 release descriptor version is unsupported"
        )
    return source, value


def proof_v3_release_artifact_paths(
    path,
) -> tuple[Path, dict[str, Path]]:
    """Return the descriptor and its exact bounded artifact inventory."""

    source, value = load_proof_v3_release_descriptor(path)
    root = source.parent
    result = {
        name: _relative_file(root, value[name], name.replace("_", " "))
        for name in sorted((_REQUIRED | _OPTIONAL) - {"version", "model_id"})
        if name in value
        and name
        not in {"runtime_encoding_id", "max_decode_tokens"}
    }
    return source, result


def _relative_file(root: Path, value: object, name: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).is_absolute()
    ):
        raise ProofV3Error(f"proof-v3 {name} path is malformed")
    result = (root / value).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ProofV3Error(
            f"proof-v3 {name} path escapes its release directory"
        ) from exc
    if not result.is_file():
        raise ProofV3Error(f"proof-v3 {name} file is unavailable")
    return result


def load_qualified_proof_v3_catalog(
    descriptor_paths: Sequence[str | Path],
    *,
    model_registry_client,
    tokenizer_digest_resolver: Callable[[str], bytes] | None = None,
    validation_cache_dir=None,
) -> dict[str, QualifiedEconomicProofV3Release]:
    """Load exact model releases against current chain-selected authorities."""

    paths = tuple(descriptor_paths)
    if not paths:
        return {}
    if model_registry_client is None:
        raise ProofV3Error("proof-v3 catalog requires ModelRegistry")
    if tokenizer_digest_resolver is None:
        from verallm.registry.tokenizer_hash import compute_tokenizer_hash

        tokenizer_digest_resolver = compute_tokenizer_hash
    authority = model_registry_client.get_manifest_authority()
    signers = tuple(authority.signers)
    threshold = int(authority.threshold)
    result: dict[str, QualifiedEconomicProofV3Release] = {}
    for descriptor_path in paths:
        source, value = load_proof_v3_release_descriptor(descriptor_path)
        model_id = value.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ProofV3Error("proof-v3 descriptor model_id is malformed")
        if model_id in result:
            raise ProofV3Error(
                "proof-v3 catalog contains a duplicate model"
            )
        spec = model_registry_client.get_on_chain_model_spec(model_id)
        if spec is None:
            raise ProofV3VerificationError(
                "proof-v3 release model is not registered"
            )
        chain_tokenizer = bytes(getattr(spec, "tokenizer_hash", b"") or b"")
        if len(chain_tokenizer) != 32 or chain_tokenizer == bytes(32):
            raise ProofV3VerificationError(
                "proof-v3 release requires an on-chain tokenizer binding"
            )
        local_tokenizer = tokenizer_digest_resolver(model_id)
        if (
            not isinstance(local_tokenizer, bytes)
            or len(local_tokenizer) != 32
            or local_tokenizer != chain_tokenizer
        ):
            raise ProofV3VerificationError(
                "proof-v3 release tokenizer does not match ModelRegistry"
            )
        encoding = value.get("runtime_encoding_id")
        max_decode = value.get("max_decode_tokens")
        if not isinstance(encoding, str) or not encoding:
            raise ProofV3Error(
                "proof-v3 runtime encoding is malformed"
            )
        if (
            isinstance(max_decode, bool)
            or not isinstance(max_decode, int)
            or not 0 < max_decode < 1 << 32
        ):
            raise ProofV3Error(
                "proof-v3 maximum decode bound is malformed"
            )
        root = source.parent
        gdn_name = value.get("gdn_runtime_semantics")
        moe_name = value.get("moe_runtime_semantics")
        lm_head_catalog_name = value.get("lm_head_catalog")
        projection_manifest_name = value.get("projection_manifest")
        projection_catalog_name = value.get("projection_catalog")
        if bool(projection_manifest_name) != bool(projection_catalog_name):
            raise ProofV3Error(
                "proof-v3 projection manifest and catalog must be "
                "configured together"
            )
        verified_projection_manifest = None
        projection_catalog = None
        projection_catalog_validation_context = None
        if projection_manifest_name is not None:
            chain_config = getattr(model_registry_client, "_config", None)
            if chain_config is None:
                raise ProofV3Error(
                    "proof-v3 projection catalog requires the authenticated "
                    "ModelRegistry chain configuration"
                )
            from verallm.proof_v3.catalog import (
                load_verified_projection_manifest_v3,
            )
            from verallm.proof_v3.catalog_validation_cache import (
                load_weight_catalog_with_validation_cache_v3,
            )

            verified_projection_manifest = (
                load_verified_projection_manifest_v3(
                    _relative_file(
                        root,
                        projection_manifest_name,
                        "projection manifest",
                    ),
                    chain_config=chain_config,
                    model_registry_client=model_registry_client,
                    expected_model_id=model_id,
                )
            )
            (
                projection_catalog,
                projection_catalog_validation_context,
            ) = load_weight_catalog_with_validation_cache_v3(
                catalog_path=_relative_file(
                    root,
                    projection_catalog_name,
                    "projection catalog",
                ),
                verified_manifest=verified_projection_manifest,
                cache_dir=validation_cache_dir,
            )
            logger.info(
                "Proof-v3 projection catalog validation receipt for %s: %s",
                model_id,
                (
                    "hit"
                    if projection_catalog_validation_context.cache_hit
                    else "miss; deep qualification required once"
                ),
            )
        result[model_id] = load_qualified_economic_proof_v3_release(
            signed_profile_path=_relative_file(
                root,
                value["execution_profile"],
                "execution profile",
            ),
            manifest_artifact_path=_relative_file(
                root,
                value["manifest"],
                "manifest",
            ),
            calibration_set_path=_relative_file(
                root,
                value["calibration_set"],
                "calibration set",
            ),
            attention_runtime_semantics_path=_relative_file(
                root,
                value["attention_runtime_semantics"],
                "attention semantics",
            ),
            gdn_runtime_semantics_path=(
                _relative_file(root, gdn_name, "GDN semantics")
                if gdn_name is not None
                else None
            ),
            moe_runtime_semantics_path=(
                _relative_file(root, moe_name, "MoE semantics")
                if moe_name is not None
                else None
            ),
            lm_head_catalog_path=(
                _relative_file(
                    root,
                    lm_head_catalog_name,
                    "LM-head catalog",
                )
                if lm_head_catalog_name is not None
                else None
            ),
            expected_model_id=model_id,
            expected_authorities=signers,
            authority_threshold=threshold,
            layer_kinds=None,
            tokenizer_binding_digest=local_tokenizer,
            runtime_encoding_id=encoding,
            max_decode_tokens=max_decode,
            verified_projection_manifest=verified_projection_manifest,
            weight_catalog=projection_catalog,
            projection_catalog_validation_context=(
                projection_catalog_validation_context
            ),
        )
    return result


__all__ = [
    "MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES",
    "load_proof_v3_release_descriptor",
    "load_qualified_proof_v3_catalog",
    "proof_v3_release_artifact_paths",
]
