"""Content-addressed distribution for mesh tensor manifests.

Same style as the GLEIPNIR artifact store: one or more base URLs (chain
config `mesh_manifest_base_urls`, overridable with
`VERATHOS_MESH_MANIFEST_BASE_URLS`) each host a small JSON index plus one
manifest file per model. The remote index is discovery metadata, NOT a
trust anchor: a downloaded manifest is accepted only when its recomputed
`tensor_manifest_root` equals the root the caller already trusts (the
mesh spec's committed root, which registration binds on-chain), and the
package hash matches when the caller knows it. A wrong or tampered
download is therefore self-defeating; the only thing distribution saves
is the local manifest rebuild (minutes for a 7B, over an hour for the
largest models).

Index shape at `<base>/mesh-manifest-index.json`:

    {
      "schema": "verathos-mesh-manifest-index-v1",
      "models": {
        "<model_id>": {
          "filename": "<model_id>.tensor-manifest.json",
          "sha256": "<hex sha256 of the file bytes>",
          "bytes": <int>,
          "tensor_manifest_root": "<hex root, advisory>"
        }
      }
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

MESH_MANIFEST_INDEX_SCHEMA = "verathos-mesh-manifest-index-v1"
MESH_MANIFEST_INDEX_FILENAME = "mesh-manifest-index.json"
MESH_MANIFEST_BASE_URLS_ENV = "VERATHOS_MESH_MANIFEST_BASE_URLS"

# Owner-operated gleipnir stores, keyed by EVM chain id. The launch path
# falls back to these when a model's registry entry names no manifest
# URLs and the env override is unset, so a shipped-catalogue launch on a
# subnet pool always finds the owner-built manifest instead of rebuilding
# it on the driver (a rebuild on any miner is always a broken fetch path).
GLEIPNIR_STORE_URLS_BY_CHAIN: dict[int, tuple[str, ...]] = {
    945: ("https://verathos.ai/gleipnir/testnet",),
    964: ("https://verathos.ai/gleipnir/mainnet",),
}


def default_store_urls_for_chain(chain_id: int | None) -> tuple[str, ...]:
    if chain_id is None:
        return ()
    return GLEIPNIR_STORE_URLS_BY_CHAIN.get(int(chain_id), ())
MESH_MANIFEST_CACHE_DIR_ENV = "VERATHOS_MESH_MANIFEST_CACHE_DIR"
MAX_MESH_MANIFEST_INDEX_BYTES = 8 << 20
MAX_MESH_MANIFEST_BYTES = 64 << 20
DEFAULT_MESH_MANIFEST_TIMEOUT_SECONDS = 120.0

_FILENAME_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class MeshManifestStoreError(RuntimeError):
    pass


def normalize_mesh_manifest_base_urls(values) -> tuple[str, ...]:
    urls: list[str] = []
    for value in values or ():
        url = str(value or "").strip().rstrip("/")
        if not url:
            continue
        if not url.startswith(("https://", "http://")):
            raise MeshManifestStoreError(
                f"mesh manifest base URL must be http(s): {url!r}"
            )
        if url not in urls:
            urls.append(url)
    return tuple(urls)


def configured_mesh_manifest_base_urls(defaults=()) -> tuple[str, ...]:
    raw = os.environ.get(MESH_MANIFEST_BASE_URLS_ENV, "")
    if raw.strip():
        return normalize_mesh_manifest_base_urls(
            part for part in raw.split(",")
        )
    return normalize_mesh_manifest_base_urls(defaults)


def _default_cache_dir() -> Path:
    raw = os.environ.get(MESH_MANIFEST_CACHE_DIR_ENV, "")
    if raw.strip():
        return Path(raw).expanduser()
    data_dir = os.environ.get("VERALLM_DATA_DIR", "~/.verathos")
    return Path(data_dir).expanduser() / "mesh-manifest-cache"


def _fetch_bounded(url: str, *, max_bytes: int, timeout: float) -> bytes:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise MeshManifestStoreError(
            f"remote object exceeds {max_bytes} bytes: {url}"
        )
    return data


def _parse_index(data: bytes, *, source_url: str) -> dict:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshManifestStoreError(
            f"mesh manifest index is not valid JSON: {source_url}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema") != (
        MESH_MANIFEST_INDEX_SCHEMA
    ):
        raise MeshManifestStoreError(
            f"mesh manifest index has an unsupported schema: {source_url}"
        )
    models = document.get("models")
    if not isinstance(models, dict):
        raise MeshManifestStoreError(
            f"mesh manifest index has no models mapping: {source_url}"
        )
    return models


def fetch_mesh_tensor_manifest(
    model_id: str,
    *,
    expected_tensor_manifest_root: str,
    expected_package_hash: str = "",
    base_urls,
    cache_dir: str | Path | None = None,
    timeout: float = DEFAULT_MESH_MANIFEST_TIMEOUT_SECONDS,
) -> Path:
    """Fetch and verify one model's tensor manifest; return its cache path.

    `expected_tensor_manifest_root` is REQUIRED and must come from a source
    the caller already trusts (the mesh spec, whose root registration binds
    on-chain). The cache is content-addressed by that root, so a cache hit
    is re-verified only by cheap root equality at load.
    """

    from verallm.mesh.gguf_manifest import load_gguf_tensor_manifest
    from verallm.mesh.model_spec import gguf_package_hash

    expected_root = str(expected_tensor_manifest_root or "").strip().lower()
    if not expected_root:
        raise MeshManifestStoreError(
            "fetching a mesh manifest requires the committed tensor "
            "manifest root; refusing an unanchored download"
        )
    urls = normalize_mesh_manifest_base_urls(base_urls)
    if not urls:
        raise MeshManifestStoreError("no mesh manifest base URLs configured")

    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_path = cache_root / f"{expected_root}.tensor-manifest.json"
    if cache_path.is_file():
        cached = load_gguf_tensor_manifest(cache_path)
        if str(cached.get("tensor_manifest_root", "")).lower() == expected_root:
            return cache_path
        # A colliding filename with the wrong content is corruption, never
        # an acceptable answer; fall through and re-download.
        cache_path.unlink(missing_ok=True)

    errors: list[str] = []
    for base_url in urls:
        index_url = f"{base_url}/{MESH_MANIFEST_INDEX_FILENAME}"
        try:
            models = _parse_index(
                _fetch_bounded(
                    index_url,
                    max_bytes=MAX_MESH_MANIFEST_INDEX_BYTES,
                    timeout=timeout,
                ),
                source_url=index_url,
            )
            entry = models.get(str(model_id))
            if not isinstance(entry, dict):
                raise MeshManifestStoreError(
                    f"model {model_id!r} is not in the index: {index_url}"
                )
            filename = str(entry.get("filename", "") or "")
            if not _FILENAME_SAFE.match(filename):
                raise MeshManifestStoreError(
                    f"index filename for {model_id!r} is not a plain "
                    f"basename: {filename!r}"
                )
            expected_sha = str(entry.get("sha256", "") or "").lower()
            if len(expected_sha) != 64:
                raise MeshManifestStoreError(
                    f"index entry for {model_id!r} has no sha256: {index_url}"
                )
            data = _fetch_bounded(
                f"{base_url}/{filename}",
                max_bytes=MAX_MESH_MANIFEST_BYTES,
                timeout=timeout,
            )
            if hashlib.sha256(data).hexdigest() != expected_sha:
                raise MeshManifestStoreError(
                    f"manifest bytes do not match the index sha256: "
                    f"{base_url}/{filename}"
                )
            cache_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=cache_root, delete=False, suffix=".part"
            ) as handle:
                handle.write(data)
                temp_path = Path(handle.name)
            try:
                manifest = load_gguf_tensor_manifest(temp_path)
                got_root = str(
                    manifest.get("tensor_manifest_root", "")
                ).lower()
                if got_root != expected_root:
                    raise MeshManifestStoreError(
                        f"downloaded manifest root {got_root[:16]}... does "
                        f"not match the committed root "
                        f"{expected_root[:16]}...; refusing it"
                    )
                if expected_package_hash:
                    got_package = gguf_package_hash(manifest).hex()
                    if got_package != str(expected_package_hash).lower():
                        raise MeshManifestStoreError(
                            "downloaded manifest package hash does not "
                            "match the committed package hash; refusing it"
                        )
                os.replace(temp_path, cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
            return cache_path
        except MeshManifestStoreError as exc:
            errors.append(str(exc))
        except OSError as exc:
            errors.append(f"{base_url}: {exc}")
    raise MeshManifestStoreError(
        "no configured mesh manifest source produced a verified manifest "
        f"for {model_id!r}: " + " | ".join(errors)
    )
