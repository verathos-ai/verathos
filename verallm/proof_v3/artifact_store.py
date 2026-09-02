"""Content-addressed distribution for authenticated proof-v3 releases.

The remote index is discovery metadata, not a trust anchor.  Every downloaded
release is still authenticated against the current ModelRegistry authority,
on-chain ModelSpec, signed execution profile, and exact catalog commitments by
``load_qualified_proof_v3_catalog``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from verallm.proof_v2.artifact_store import (
    ProofV2ArtifactStoreError,
    _atomic_write,
    _cached_or_downloaded_object,
    _fetch_small_bytes,
    _read_bounded_file,
    _sha256_file,
    normalize_proof_v2_artifact_base_urls,
)
from verallm.proof_v3.canary_policy import MAX_CANARY_POLICY_DOCUMENT_BYTES
from verallm.proof_v3.catalog import (
    MAX_PALLAS_CATALOG_BYTES_V3,
    MAX_QUALIFIED_ARTIFACT_BYTES_V3,
)
from verallm.proof_v3.document import MAX_EXECUTION_PROFILE_DOCUMENT_BYTES
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3,
)
from verallm.proof_v3.economic_release import QualifiedEconomicProofV3Release
from verallm.proof_v3.economic_release_catalog import (
    MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES,
    load_proof_v3_release_descriptor,
    load_qualified_proof_v3_catalog,
    proof_v3_release_artifact_paths,
)

PROOF_V3_ARTIFACT_INDEX_SCHEMA_V1 = "verathos-proof-v3-artifact-index-v1"
PROOF_V3_ARTIFACT_INDEX_SCHEMA = "verathos-proof-v3-artifact-index-v2"
PROOF_V3_ARTIFACT_INDEX_FILENAME = "proof-v3-index.json"
PROOF_V3_ARTIFACT_BASE_URLS_ENV = "VERATHOS_PROOF_V3_ARTIFACT_BASE_URLS"
PROOF_V3_ARTIFACT_CACHE_DIR_ENV = "VERATHOS_PROOF_V3_ARTIFACT_CACHE_DIR"
MAX_PROOF_V3_ARTIFACT_INDEX_BYTES = 8 << 20
MAX_PROOF_V3_ARTIFACT_INDEX_MODELS = 10_000
MAX_PROOF_V3_RELEASE_ARTIFACT_BYTES = 512 << 20
DEFAULT_PROOF_V3_ARTIFACT_TIMEOUT_SECONDS = 300.0

_ARTIFACT_MAXIMUMS = MappingProxyType(
    {
        "manifest": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "execution_profile": MAX_EXECUTION_PROFILE_DOCUMENT_BYTES,
        "calibration_set": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "attention_runtime_semantics": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "gdn_runtime_semantics": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "moe_runtime_semantics": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "lm_head_catalog": MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3,
        "projection_manifest": MAX_QUALIFIED_ARTIFACT_BYTES_V3,
        "projection_catalog": MAX_PALLAS_CATALOG_BYTES_V3,
        "canary_policy": MAX_CANARY_POLICY_DOCUMENT_BYTES,
    }
)


class ProofV3ArtifactStoreError(RuntimeError):
    """Remote proof-v3 artifact discovery or validation failed."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3ArtifactStoreError(
                f"proof-v3 artifact index contains duplicate field: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ProofV3ArtifactStoreError(
        f"proof-v3 artifact index contains a non-finite number: {value}"
    )


def _hex32(value: object, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
    ):
        raise ProofV3ArtifactStoreError(
            f"{name} must be canonical lowercase hexadecimal"
        )
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ProofV3ArtifactStoreError(f"{name} is not hexadecimal") from exc
    if len(result) != 32:
        raise ProofV3ArtifactStoreError(f"{name} must be exactly 32 bytes")
    return result


def _size(value: object, maximum: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ProofV3ArtifactStoreError(f"{name} is out of range")
    return value


def _model_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != unicodedata.normalize("NFC", value)
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ProofV3ArtifactStoreError("proof-v3 artifact model_id is malformed")
    return value


@dataclass(frozen=True, slots=True)
class ProofV3ArtifactObject:
    role: str
    sha256: bytes
    size: int

    def __post_init__(self) -> None:
        maximum = _ARTIFACT_MAXIMUMS.get(self.role)
        if maximum is None:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact role is unsupported"
            )
        if type(self.sha256) is not bytes or len(self.sha256) != 32:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact SHA-256 must be exactly 32 bytes"
            )
        _size(self.size, maximum, f"{self.role} size")

    @property
    def filename(self) -> str:
        return f"{self.sha256.hex()}.object"

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sha256": self.sha256.hex(),
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ProofV3ArtifactObject":
        if not isinstance(value, dict) or set(value) != {
            "role",
            "sha256",
            "size",
        }:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact object inventory is malformed"
            )
        role = value["role"]
        if not isinstance(role, str) or role not in _ARTIFACT_MAXIMUMS:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact object role is malformed"
            )
        return cls(
            role=role,
            sha256=_hex32(value["sha256"], f"{role} SHA-256"),
            size=_size(value["size"], _ARTIFACT_MAXIMUMS[role], f"{role} size"),
        )


@dataclass(frozen=True, slots=True)
class ProofV3ArtifactIndexEntry:
    model_id: str
    descriptor_sha256: bytes
    descriptor_size: int
    artifacts: tuple[ProofV3ArtifactObject, ...]
    _by_role: Mapping[str, ProofV3ArtifactObject] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _model_id(self.model_id)
        if (
            type(self.descriptor_sha256) is not bytes
            or len(self.descriptor_sha256) != 32
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 descriptor SHA-256 must be exactly 32 bytes"
            )
        _size(
            self.descriptor_size,
            MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES,
            "descriptor size",
        )
        artifacts = tuple(self.artifacts)
        roles = tuple(item.role for item in artifacts)
        if (
            not artifacts
            or roles != tuple(sorted(roles))
            or len(roles) != len(set(roles))
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact roles must be sorted and unique"
            )
        if sum(item.size for item in artifacts) > MAX_PROOF_V3_RELEASE_ARTIFACT_BYTES:
            raise ProofV3ArtifactStoreError(
                "proof-v3 release artifact inventory is too large"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "_by_role",
            MappingProxyType({item.role: item for item in artifacts}),
        )

    @property
    def descriptor_filename(self) -> str:
        return f"{self.descriptor_sha256.hex()}.release.json"

    @property
    def by_role(self) -> Mapping[str, ProofV3ArtifactObject]:
        return self._by_role

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [item.to_dict() for item in self.artifacts],
            "descriptor_sha256": self.descriptor_sha256.hex(),
            "descriptor_size": self.descriptor_size,
            "model_id": self.model_id,
        }

    def release_sha256(self) -> bytes:
        """Return the complete model-entry identity used for hot refresh."""

        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(
            b"verathos.proof_v3.release-index-entry.v1\0" + encoded
        ).digest()

    @classmethod
    def from_dict(cls, value: object) -> "ProofV3ArtifactIndexEntry":
        if not isinstance(value, dict) or set(value) != {
            "artifacts",
            "descriptor_sha256",
            "descriptor_size",
            "model_id",
        }:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index entry is malformed"
            )
        raw_artifacts = value["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact inventory must be a list"
            )
        return cls(
            model_id=_model_id(value["model_id"]),
            descriptor_sha256=_hex32(
                value["descriptor_sha256"],
                "descriptor SHA-256",
            ),
            descriptor_size=_size(
                value["descriptor_size"],
                MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES,
                "descriptor size",
            ),
            artifacts=tuple(
                ProofV3ArtifactObject.from_dict(item)
                for item in raw_artifacts
            ),
        )


@dataclass(frozen=True, slots=True)
class ProofV3ArtifactIndex:
    chain_id: int
    netuid: int
    registry_address: bytes
    models: tuple[ProofV3ArtifactIndexEntry, ...]
    canary_policy: ProofV3ArtifactObject | None = None
    schema: str = PROOF_V3_ARTIFACT_INDEX_SCHEMA
    _by_model: Mapping[str, ProofV3ArtifactIndexEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema not in {
            PROOF_V3_ARTIFACT_INDEX_SCHEMA_V1,
            PROOF_V3_ARTIFACT_INDEX_SCHEMA,
        }:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index schema is unsupported"
            )
        if self.schema == PROOF_V3_ARTIFACT_INDEX_SCHEMA_V1:
            if self.canary_policy is not None:
                raise ProofV3ArtifactStoreError(
                    "proof-v3 v1 artifact index cannot contain a canary policy"
                )
        elif (
            not isinstance(self.canary_policy, ProofV3ArtifactObject)
            or self.canary_policy.role != "canary_policy"
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 v2 artifact index requires a canary policy"
            )
        if type(self.chain_id) is not int or self.chain_id <= 0:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index chain_id is invalid"
            )
        if type(self.netuid) is not int or self.netuid < 0:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index netuid is invalid"
            )
        if (
            type(self.registry_address) is not bytes
            or len(self.registry_address) != 20
            or self.registry_address == bytes(20)
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index registry address is invalid"
            )
        models = tuple(self.models)
        model_ids = tuple(item.model_id for item in models)
        if (
            not models
            or len(models) > MAX_PROOF_V3_ARTIFACT_INDEX_MODELS
            or model_ids != tuple(sorted(model_ids))
            or len(model_ids) != len(set(model_ids))
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index models must be sorted and unique"
            )
        object.__setattr__(self, "models", models)
        object.__setattr__(
            self,
            "_by_model",
            MappingProxyType({item.model_id: item for item in models}),
        )

    @property
    def by_model(self) -> Mapping[str, ProofV3ArtifactIndexEntry]:
        return self._by_model

    def to_dict(self) -> dict[str, object]:
        result = {
            "chain_id": self.chain_id,
            "models": [item.to_dict() for item in self.models],
            "netuid": self.netuid,
            "registry_address": self.registry_address.hex(),
            "schema": self.schema,
        }
        if self.canary_policy is not None:
            result["canary_policy"] = self.canary_policy.to_dict()
        return result

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_PROOF_V3_ARTIFACT_INDEX_BYTES:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index exceeds the size limit"
            )
        return encoded

    def validate_context(self, chain_config) -> None:
        try:
            address = bytes.fromhex(
                str(chain_config.model_registry_address).removeprefix("0x")
            )
        except ValueError as exc:
            raise ProofV3ArtifactStoreError(
                "configured ModelRegistry address is invalid"
            ) from exc
        if (
            self.chain_id != chain_config.chain_id
            or self.netuid != chain_config.netuid
            or self.registry_address != address
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index does not match the chain context"
            )

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "ProofV3ArtifactIndex":
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_PROOF_V3_ARTIFACT_INDEX_BYTES
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index length is out of range"
            )
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_json_constant,
            )
        except ProofV3ArtifactStoreError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index fields are malformed"
            )
        schema = value.get("schema")
        expected_fields = {
            "chain_id",
            "models",
            "netuid",
            "registry_address",
            "schema",
        }
        if schema == PROOF_V3_ARTIFACT_INDEX_SCHEMA:
            expected_fields.add("canary_policy")
        if set(value) != expected_fields:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index fields are malformed"
            )
        raw_models = value["models"]
        if not isinstance(raw_models, list):
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index models must be a list"
            )
        address = _hex32(
            "0" * 24 + str(value["registry_address"]),
            "registry address",
        )[12:]
        result = cls(
            chain_id=value["chain_id"],
            netuid=value["netuid"],
            registry_address=address,
            models=tuple(
                ProofV3ArtifactIndexEntry.from_dict(item)
                for item in raw_models
            ),
            canary_policy=(
                ProofV3ArtifactObject.from_dict(value["canary_policy"])
                if "canary_policy" in value
                else None
            ),
            schema=schema,
        )
        if result.canonical_json_bytes() != encoded:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index is not canonical JSON"
            )
        return result


@dataclass(frozen=True)
class ResolvedProofV3Release:
    model_id: str
    descriptor_path: Path
    release: QualifiedEconomicProofV3Release
    release_sha256: bytes
    index_source_url: str


@dataclass(frozen=True)
class RemoteProofV3ReleaseResolution:
    releases: Mapping[str, ResolvedProofV3Release]
    missing_model_ids: tuple[str, ...]
    failures: Mapping[str, str]
    index_source_url: str
    canary_policy_path: Path | None = None
    index_sha256: bytes = b""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "releases",
            MappingProxyType(dict(self.releases)),
        )
        object.__setattr__(
            self,
            "failures",
            MappingProxyType(dict(self.failures)),
        )
        if type(self.index_sha256) is not bytes or len(self.index_sha256) != 32:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index SHA-256 must be exactly 32 bytes"
            )


@dataclass(frozen=True, slots=True)
class ProofV3ReleaseIndexProbe:
    """Authenticated-context discovery metadata for one published release.

    The index is not a trust anchor.  Callers may use this lightweight probe
    only to detect a possible descriptor change; they must still resolve and
    authenticate the complete release before adopting it.
    """

    model_id: str
    release_sha256: bytes
    descriptor_sha256: bytes
    index_source_url: str


@dataclass(frozen=True, slots=True)
class ProofV3ArtifactIndexProbe:
    """Bounded change detector for one context-validated remote index."""

    index_sha256: bytes
    index_source_url: str

    def __post_init__(self) -> None:
        if type(self.index_sha256) is not bytes or len(self.index_sha256) != 32:
            raise ProofV3ArtifactStoreError(
                "proof-v3 artifact index SHA-256 must be exactly 32 bytes"
            )


def normalize_proof_v3_artifact_base_urls(
    values: Iterable[str],
) -> tuple[str, ...]:
    try:
        return normalize_proof_v2_artifact_base_urls(values)
    except ProofV2ArtifactStoreError as exc:
        raise ProofV3ArtifactStoreError(str(exc).replace("proof-v2", "proof-v3")) from exc


def configured_proof_v3_artifact_base_urls(
    cli_values: Sequence[str] | None,
    *,
    default_values: Sequence[str] = (),
) -> tuple[str, ...]:
    if cli_values is not None:
        values = tuple(item for item in cli_values if item)
        return normalize_proof_v3_artifact_base_urls(values) if values else ()
    raw = os.environ.get(PROOF_V3_ARTIFACT_BASE_URLS_ENV, "")
    values = tuple(
        item.strip()
        for line in raw.splitlines()
        for item in line.split(",")
        if item.strip()
    )
    if values:
        return normalize_proof_v3_artifact_base_urls(values)
    defaults = tuple(item for item in default_values if item)
    return normalize_proof_v3_artifact_base_urls(defaults) if defaults else ()


def proof_v3_artifact_cache_directory(
    configured_path: str | Path | None,
    *,
    chain_id: int,
    netuid: int,
) -> Path:
    raw = str(
        configured_path
        or os.environ.get(PROOF_V3_ARTIFACT_CACHE_DIR_ENV, "")
    ).strip()
    if raw:
        root = Path(raw).expanduser()
    else:
        data_dir = os.environ.get("VERALLM_DATA_DIR", "").strip()
        root = (
            Path(data_dir).expanduser()
            if data_dir
            else Path.home() / ".verathos"
        ) / "proof_v3_artifacts"
    return root.absolute() / f"{chain_id}-{netuid}"


def _url(base_url: str, filename: str) -> str:
    return f"{base_url}/{filename}"


def _load_remote_index(
    base_urls: tuple[str, ...],
    *,
    cache_directory: Path,
    chain_config,
    timeout_seconds: float,
) -> tuple[ProofV3ArtifactIndex, str]:
    failures = []
    for base_url in base_urls:
        url = _url(base_url, PROOF_V3_ARTIFACT_INDEX_FILENAME)
        cache_path = (
            cache_directory
            / "indexes"
            / f"{hashlib.sha256(base_url.encode()).hexdigest()}.json"
        )
        try:
            encoded = _fetch_small_bytes(
                url,
                maximum=MAX_PROOF_V3_ARTIFACT_INDEX_BYTES,
                timeout_seconds=timeout_seconds,
            )
            index = ProofV3ArtifactIndex.from_json_bytes(encoded)
            index.validate_context(chain_config)
            _atomic_write(cache_path, encoded)
            return index, url
        except Exception as remote_exc:
            try:
                encoded = _read_bounded_file(
                    cache_path,
                    maximum=MAX_PROOF_V3_ARTIFACT_INDEX_BYTES,
                    name="cached proof-v3 artifact index",
                )
                index = ProofV3ArtifactIndex.from_json_bytes(encoded)
                index.validate_context(chain_config)
                return index, f"{url} (cached)"
            except Exception:
                failures.append(str(remote_exc))
    raise ProofV3ArtifactStoreError(
        "no valid proof-v3 artifact index is available: "
        + "; ".join(failures)
    )


def _download(
    entry,
    *,
    base_urls: tuple[str, ...],
    cache: Path,
    timeout_seconds: float,
    descriptor: bool = False,
) -> Path:
    try:
        return _cached_or_downloaded_object(
            base_urls=base_urls,
            cache_directory=cache,
            filename=(
                entry.descriptor_filename if descriptor else entry.filename
            ),
            expected_sha256=(
                entry.descriptor_sha256 if descriptor else entry.sha256
            ),
            expected_size=(
                entry.descriptor_size if descriptor else entry.size
            ),
            maximum=(
                MAX_PROOF_V3_RELEASE_DESCRIPTOR_BYTES
                if descriptor
                else _ARTIFACT_MAXIMUMS[entry.role]
            ),
            timeout_seconds=timeout_seconds,
        )
    except ProofV2ArtifactStoreError as exc:
        raise ProofV3ArtifactStoreError(str(exc).replace("proof-v2", "proof-v3")) from exc


def _safe_relative(root: Path, value: object, name: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).is_absolute()
    ):
        raise ProofV3ArtifactStoreError(
            f"proof-v3 {name} path is malformed"
        )
    result = (root / value).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ProofV3ArtifactStoreError(
            f"proof-v3 {name} path escapes the release directory"
        ) from exc
    return result


def _link_verified(source: Path, destination: Path, *, digest: bytes, size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            actual_digest, actual_size = _sha256_file(
                destination,
                maximum=size,
            )
            if actual_digest == digest and actual_size == size:
                return
        except Exception:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
    except FileExistsError:
        return
    except OSError:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=str(destination.parent),
        )
        try:
            with source.open("rb") as input_handle, os.fdopen(
                descriptor,
                "wb",
            ) as output_handle:
                while chunk := input_handle.read(1 << 20):
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary_name).unlink(missing_ok=True)
            raise


def _materialize_release(
    entry: ProofV3ArtifactIndexEntry,
    descriptor_object: Path,
    artifact_objects: Mapping[str, Path],
    *,
    cache: Path,
) -> Path:
    _source, descriptor = load_proof_v3_release_descriptor(descriptor_object)
    if descriptor.get("model_id") != entry.model_id:
        raise ProofV3ArtifactStoreError(
            "proof-v3 descriptor model does not match the artifact index"
        )
    path_roles = {
        role
        for role in _ARTIFACT_MAXIMUMS
        if role in descriptor
    }
    if path_roles != set(entry.by_role):
        raise ProofV3ArtifactStoreError(
            "proof-v3 descriptor artifact inventory does not match the index"
        )
    root = cache / "releases" / entry.descriptor_sha256.hex()
    destinations = {}
    local_descriptor = root / "release.json"
    for role in sorted(path_roles):
        destination = _safe_relative(root, descriptor[role], role)
        if (
            destination == local_descriptor
            or destination in destinations.values()
        ):
            raise ProofV3ArtifactStoreError(
                "proof-v3 descriptor aliases two artifact paths"
            )
        destinations[role] = destination
    for role, destination in destinations.items():
        item = entry.by_role[role]
        _link_verified(
            artifact_objects[role],
            destination,
            digest=item.sha256,
            size=item.size,
        )
    _link_verified(
        descriptor_object,
        local_descriptor,
        digest=entry.descriptor_sha256,
        size=entry.descriptor_size,
    )
    return local_descriptor


def resolve_remote_proof_v3_releases(
    base_urls: Iterable[str],
    *,
    chain_config,
    model_registry_client,
    cache_directory: str | Path | None = None,
    model_ids: Iterable[str] | None = None,
    tokenizer_digest_resolver=None,
    timeout_seconds: float = DEFAULT_PROOF_V3_ARTIFACT_TIMEOUT_SECONDS,
) -> RemoteProofV3ReleaseResolution:
    """Download, cache, and authenticate exact proof-v3 model releases."""

    sources = normalize_proof_v3_artifact_base_urls(base_urls)
    cache = proof_v3_artifact_cache_directory(
        cache_directory,
        chain_id=chain_config.chain_id,
        netuid=chain_config.netuid,
    )
    index, source_url = _load_remote_index(
        sources,
        cache_directory=cache,
        chain_config=chain_config,
        timeout_seconds=timeout_seconds,
    )
    canary_policy_path = None
    if index.canary_policy is not None:
        policy_object = _download(
            index.canary_policy,
            base_urls=sources,
            cache=cache,
            timeout_seconds=timeout_seconds,
        )
        canary_policy_path = (
            cache
            / "policies"
            / f"{index.canary_policy.sha256.hex()}.json"
        )
        _link_verified(
            policy_object,
            canary_policy_path,
            digest=index.canary_policy.sha256,
            size=index.canary_policy.size,
        )
    requested = (
        tuple(model_registry_client.get_model_list())
        if model_ids is None
        else tuple(model_ids)
    )
    requested = tuple(sorted(set(_model_id(value) for value in requested)))
    releases = {}
    missing = []
    failures = {}
    for model_id in requested:
        entry = index.by_model.get(model_id)
        if entry is None:
            missing.append(model_id)
            continue
        try:
            descriptor_object = _download(
                entry,
                base_urls=sources,
                cache=cache,
                timeout_seconds=timeout_seconds,
                descriptor=True,
            )
            objects = {
                item.role: _download(
                    item,
                    base_urls=sources,
                    cache=cache,
                    timeout_seconds=timeout_seconds,
                )
                for item in entry.artifacts
            }
            descriptor_path = _materialize_release(
                entry,
                descriptor_object,
                objects,
                cache=cache,
            )
            release = load_qualified_proof_v3_catalog(
                (descriptor_path,),
                model_registry_client=model_registry_client,
                tokenizer_digest_resolver=tokenizer_digest_resolver,
                validation_cache_fallback_dirs=(
                    cache / "catalog_validation",
                ),
            )[model_id]
            releases[model_id] = ResolvedProofV3Release(
                model_id=model_id,
                descriptor_path=descriptor_path,
                release=release,
                release_sha256=entry.release_sha256(),
                index_source_url=source_url,
            )
        except Exception as exc:
            failures[model_id] = str(exc)
    return RemoteProofV3ReleaseResolution(
        releases=releases,
        missing_model_ids=tuple(missing),
        failures=failures,
        index_source_url=source_url,
        canary_policy_path=canary_policy_path,
        index_sha256=hashlib.sha256(index.canonical_json_bytes()).digest(),
    )


def probe_remote_proof_v3_index(
    base_urls: Iterable[str],
    *,
    chain_config,
    cache_directory: str | Path | None = None,
    timeout_seconds: float = DEFAULT_PROOF_V3_ARTIFACT_TIMEOUT_SECONDS,
) -> ProofV3ArtifactIndexProbe:
    """Fetch and context-check only the bounded release index.

    Equality with a previously fully authenticated index lets a running
    validator retain its in-memory release snapshot without reloading large
    catalogs. A changed digest is only a signal to run the complete resolver;
    the index itself remains discovery metadata, never a trust anchor.
    """

    sources = normalize_proof_v3_artifact_base_urls(base_urls)
    cache = proof_v3_artifact_cache_directory(
        cache_directory,
        chain_id=chain_config.chain_id,
        netuid=chain_config.netuid,
    )
    index, source_url = _load_remote_index(
        sources,
        cache_directory=cache,
        chain_config=chain_config,
        timeout_seconds=timeout_seconds,
    )
    return ProofV3ArtifactIndexProbe(
        index_sha256=hashlib.sha256(index.canonical_json_bytes()).digest(),
        index_source_url=source_url,
    )


def probe_remote_proof_v3_release(
    model_id: str,
    base_urls: Iterable[str],
    *,
    chain_config,
    cache_directory: str | Path | None = None,
    timeout_seconds: float = DEFAULT_PROOF_V3_ARTIFACT_TIMEOUT_SECONDS,
) -> ProofV3ReleaseIndexProbe:
    """Fetch only the bounded index entry for one model.

    This avoids repeatedly loading large model catalogs on a running miner.
    A changed probe is deliberately insufficient for adoption; the caller
    must follow it with :func:`resolve_remote_proof_v3_release`.
    """

    selected_model = _model_id(model_id)
    sources = normalize_proof_v3_artifact_base_urls(base_urls)
    cache = proof_v3_artifact_cache_directory(
        cache_directory,
        chain_id=chain_config.chain_id,
        netuid=chain_config.netuid,
    )
    index, source_url = _load_remote_index(
        sources,
        cache_directory=cache,
        chain_config=chain_config,
        timeout_seconds=timeout_seconds,
    )
    entry = index.by_model.get(selected_model)
    if entry is None:
        raise ProofV3ArtifactStoreError(
            f"no remote proof-v3 release is published for model: {selected_model}"
        )
    return ProofV3ReleaseIndexProbe(
        model_id=selected_model,
        release_sha256=entry.release_sha256(),
        descriptor_sha256=entry.descriptor_sha256,
        index_source_url=source_url,
    )


def resolve_remote_proof_v3_release(
    model_id: str,
    base_urls: Iterable[str],
    **kwargs,
) -> ResolvedProofV3Release:
    """Resolve one miner model, failing closed on missing or invalid data."""

    result = resolve_remote_proof_v3_releases(
        base_urls,
        model_ids=(model_id,),
        **kwargs,
    )
    release = result.releases.get(model_id)
    if release is not None:
        return release
    failure = result.failures.get(model_id)
    if failure:
        raise ProofV3ArtifactStoreError(failure)
    raise ProofV3ArtifactStoreError(
        f"no remote proof-v3 release is published for model: {model_id}"
    )


def artifact_index_entry_from_release_descriptor(
    descriptor_path: str | Path,
) -> ProofV3ArtifactIndexEntry:
    """Build one publishable index entry from a complete local release."""

    source, value = load_proof_v3_release_descriptor(descriptor_path)
    _source, paths = proof_v3_release_artifact_paths(source)
    descriptor_bytes = source.read_bytes()
    artifacts = []
    for role, path in sorted(paths.items()):
        maximum = _ARTIFACT_MAXIMUMS[role]
        digest, size = _sha256_file(path, maximum=maximum)
        artifacts.append(ProofV3ArtifactObject(role, digest, size))
    return ProofV3ArtifactIndexEntry(
        model_id=_model_id(value["model_id"]),
        descriptor_sha256=hashlib.sha256(descriptor_bytes).digest(),
        descriptor_size=len(descriptor_bytes),
        artifacts=tuple(artifacts),
    )


__all__ = [
    "DEFAULT_PROOF_V3_ARTIFACT_TIMEOUT_SECONDS",
    "MAX_PROOF_V3_ARTIFACT_INDEX_BYTES",
    "PROOF_V3_ARTIFACT_BASE_URLS_ENV",
    "PROOF_V3_ARTIFACT_CACHE_DIR_ENV",
    "PROOF_V3_ARTIFACT_INDEX_FILENAME",
    "ProofV3ArtifactIndexProbe",
    "PROOF_V3_ARTIFACT_INDEX_SCHEMA",
    "PROOF_V3_ARTIFACT_INDEX_SCHEMA_V1",
    "ProofV3ArtifactIndex",
    "ProofV3ArtifactIndexEntry",
    "ProofV3ReleaseIndexProbe",
    "ProofV3ArtifactObject",
    "ProofV3ArtifactStoreError",
    "RemoteProofV3ReleaseResolution",
    "ResolvedProofV3Release",
    "artifact_index_entry_from_release_descriptor",
    "configured_proof_v3_artifact_base_urls",
    "normalize_proof_v3_artifact_base_urls",
    "proof_v3_artifact_cache_directory",
    "probe_remote_proof_v3_release",
    "probe_remote_proof_v3_index",
    "resolve_remote_proof_v3_release",
    "resolve_remote_proof_v3_releases",
]
