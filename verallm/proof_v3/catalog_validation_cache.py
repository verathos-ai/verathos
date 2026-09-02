"""Persistent receipts for one-time static projection-catalog qualification."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from verallm.proof_v3.errors import ProofV3Error

CATALOG_VALIDATION_RECEIPT_ABI_V3 = (
    "verathos.proof_v3.catalog_validation_receipt.v1"
)
CATALOG_DEEP_VALIDATOR_REVISION_V3 = (
    "pallas_points_and_signed_operation_trees.v1"
)
MAX_CATALOG_VALIDATION_RECEIPT_BYTES_V3 = 8 << 10
_CONTEXT_TOKEN = object()
_RECEIPT_FIELDS = frozenset(
    {
        "abi",
        "catalog_sha256",
        "catalog_size",
        "manifest_digest",
        "model_id",
        "validator_revision",
    }
)


def default_catalog_validation_cache_dir_v3() -> Path:
    """Return the role-independent protected cache for deep-validation receipts.

    Validators, proxies, and miners may intentionally use separate
    ``VERALLM_DATA_DIR`` trees.  Deep projection-catalog validation is bound to
    the exact catalog SHA-256, signed manifest digest, and validator revision,
    so repeating it once per role provides no additional security.  Keep the
    receipts under the account-wide Verathos state directory instead.  The
    explicit environment override remains available for installations whose
    roles do not share a home directory.
    """

    configured = os.environ.get(
        "VERATHOS_PROOF_V3_VALIDATION_CACHE_DIR"
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".verathos" / "proof_v3_catalog_validation"


def _sha256_file(path: Path) -> tuple[int, bytes]:
    try:
        before = path.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > 256 << 20
        ):
            raise ProofV3Error(
                "projection catalog validation source is malformed"
            )
        digest = hashlib.sha256()
        length = 0
        with path.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(4 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                length += len(chunk)
        after = path.stat()
    except ProofV3Error:
        raise
    except OSError as exc:
        raise ProofV3Error(
            "projection catalog validation source could not be read"
        ) from exc
    if (
        length != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
    ):
        raise ProofV3Error(
            "projection catalog changed while checking its validation receipt"
        )
    return length, digest.digest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate receipt key")
        result[key] = value
    return result


def _receipt_value(
    *,
    model_id: str,
    manifest_digest: bytes,
    catalog_size: int,
    catalog_sha256: bytes,
) -> dict[str, object]:
    return {
        "abi": CATALOG_VALIDATION_RECEIPT_ABI_V3,
        "catalog_sha256": catalog_sha256.hex(),
        "catalog_size": catalog_size,
        "manifest_digest": manifest_digest.hex(),
        "model_id": model_id,
        "validator_revision": CATALOG_DEEP_VALIDATOR_REVISION_V3,
    }


def _receipt_matches(path: Path, expected: dict[str, object]) -> bool:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size
            > MAX_CATALOG_VALIDATION_RECEIPT_BYTES_V3
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or metadata.st_mode & 0o022
        ):
            return False
        encoded = path.read_bytes()
        if len(encoded) != metadata.st_size:
            return False
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and frozenset(value) == _RECEIPT_FIELDS
        and value == expected
    )


def _publish_receipt(
    path: Path,
    expected: dict[str, object],
) -> bool:
    directory = path.parent
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if hasattr(os, "geteuid"):
            directory_stat = directory.stat()
            if (
                directory_stat.st_uid != os.geteuid()
                or directory_stat.st_mode & 0o022
            ):
                return False
        descriptor, raw = tempfile.mkstemp(
            prefix=".validating-",
            suffix=".json",
            dir=directory,
        )
        temporary = Path(raw)
        os.fchmod(descriptor, 0o600)
        encoded = (
            json.dumps(
                expected,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _receipt_path(
    cache_dir: Path,
    *,
    manifest_digest: bytes,
    catalog_sha256: bytes,
) -> Path:
    key = hashlib.sha256(
        CATALOG_VALIDATION_RECEIPT_ABI_V3.encode("ascii")
        + b"\x00"
        + CATALOG_DEEP_VALIDATOR_REVISION_V3.encode("ascii")
        + manifest_digest
        + catalog_sha256
    ).hexdigest()
    return cache_dir / f"{key}.json"


@dataclass(frozen=True, slots=True, init=False)
class CatalogValidationContextV3:
    """Opaque exact-file context used by the catalog binding factory."""

    model_id: str
    manifest_digest: bytes
    catalog_size: int
    catalog_sha256: bytes
    receipt_path: Path
    cache_hit: bool
    _token: object

    @classmethod
    def _create(
        cls,
        *,
        model_id: str,
        manifest_digest: bytes,
        catalog_size: int,
        catalog_sha256: bytes,
        receipt_path: Path,
        cache_hit: bool,
    ) -> "CatalogValidationContextV3":
        result = object.__new__(cls)
        object.__setattr__(result, "model_id", model_id)
        object.__setattr__(result, "manifest_digest", manifest_digest)
        object.__setattr__(result, "catalog_size", catalog_size)
        object.__setattr__(result, "catalog_sha256", catalog_sha256)
        object.__setattr__(result, "receipt_path", receipt_path)
        object.__setattr__(result, "cache_hit", cache_hit)
        object.__setattr__(result, "_token", _CONTEXT_TOKEN)
        return result

    def require_exact_source(
        self,
        *,
        verified_manifest,
        catalog_bytes: bytes,
    ) -> None:
        if self._token is not _CONTEXT_TOKEN:
            raise ProofV3Error(
                "projection catalog validation context is unauthenticated"
            )
        verified_manifest.require_loader_provenance()
        manifest = verified_manifest.manifest
        if (
            manifest.model_spec.model_id != self.model_id
            or manifest.digest() != self.manifest_digest
            or type(catalog_bytes) is not bytes
            or len(catalog_bytes) != self.catalog_size
            or hashlib.sha256(catalog_bytes).digest()
            != self.catalog_sha256
        ):
            raise ProofV3Error(
                "projection catalog validation context changed"
            )

    def publish_after_deep_validation(self) -> None:
        if self._token is not _CONTEXT_TOKEN or self.cache_hit:
            return
        expected = _receipt_value(
            model_id=self.model_id,
            manifest_digest=self.manifest_digest,
            catalog_size=self.catalog_size,
            catalog_sha256=self.catalog_sha256,
        )
        # The receipt is an optimization only. Deep validation has already
        # succeeded, so a read-only cache must not break startup.
        _publish_receipt(self.receipt_path, expected)


def load_weight_catalog_with_validation_cache_v3(
    *,
    catalog_path,
    verified_manifest,
    cache_dir=None,
    fallback_cache_dirs=(),
):
    """Load a catalog, reusing only an exact protected validation receipt."""

    verified_manifest.require_loader_provenance()
    manifest = verified_manifest.manifest
    model_id = manifest.model_spec.model_id
    manifest_digest = manifest.digest()
    path = Path(catalog_path).expanduser().resolve()
    catalog_size, catalog_sha256 = _sha256_file(path)
    directory = (
        default_catalog_validation_cache_dir_v3()
        if cache_dir is None
        else Path(cache_dir).expanduser()
    )
    receipt_path = _receipt_path(
        directory,
        manifest_digest=manifest_digest,
        catalog_sha256=catalog_sha256,
    )
    expected = _receipt_value(
        model_id=model_id,
        manifest_digest=manifest_digest,
        catalog_size=catalog_size,
        catalog_sha256=catalog_sha256,
    )
    cache_hit = _receipt_matches(receipt_path, expected)
    if not cache_hit:
        for raw_fallback in fallback_cache_dirs:
            try:
                fallback_directory = Path(raw_fallback).expanduser()
                same_directory = (
                    fallback_directory.resolve() == directory.resolve()
                )
            except (OSError, RuntimeError, TypeError):
                # A legacy cache is an optional optimization and must never
                # prevent full validation from the primary artifact source.
                continue
            if same_directory:
                continue
            fallback_path = _receipt_path(
                fallback_directory,
                manifest_digest=manifest_digest,
                catalog_sha256=catalog_sha256,
            )
            if not _receipt_matches(fallback_path, expected):
                continue
            # Promote a previously authenticated per-role receipt into the
            # shared cache. If the shared location is read-only, using the
            # exact protected fallback receipt is still safe and fast.
            if not _publish_receipt(receipt_path, expected):
                receipt_path = fallback_path
            cache_hit = True
            break
    context = CatalogValidationContextV3._create(
        model_id=model_id,
        manifest_digest=manifest_digest,
        catalog_size=catalog_size,
        catalog_sha256=catalog_sha256,
        receipt_path=receipt_path,
        cache_hit=cache_hit,
    )
    from verallm.proof_v2.catalog import WeightCommitmentCatalogV2

    catalog = (
        WeightCommitmentCatalogV2._load_prevalidated(path)
        if cache_hit
        else WeightCommitmentCatalogV2.load(path)
    )
    if catalog.manifest_digest != manifest_digest:
        raise ProofV3Error(
            "projection catalog does not match its signed manifest"
        )
    context.require_exact_source(
        verified_manifest=verified_manifest,
        catalog_bytes=catalog.canonical_bytes(),
    )
    return catalog, context


__all__ = [
    "CATALOG_DEEP_VALIDATOR_REVISION_V3",
    "CATALOG_VALIDATION_RECEIPT_ABI_V3",
    "CatalogValidationContextV3",
    "default_catalog_validation_cache_dir_v3",
    "load_weight_catalog_with_validation_cache_v3",
]
