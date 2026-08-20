"""Content-addressed mesh manifest store: fetch, verify, refuse."""
from __future__ import annotations

import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from verallm.mesh.gguf_manifest import (
    build_gguf_tensor_manifest,
    save_gguf_tensor_manifest,
)
from verallm.mesh.manifest_store import (
    MESH_MANIFEST_INDEX_FILENAME,
    MESH_MANIFEST_INDEX_SCHEMA,
    MeshManifestStoreError,
    configured_mesh_manifest_base_urls,
    fetch_mesh_tensor_manifest,
    normalize_mesh_manifest_base_urls,
)
from verallm.mesh.model_spec import gguf_package_hash

MODEL_ID = "tiny-test-model"


def _write_test_gguf(path: Path) -> None:
    import gguf

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tensor(
        "blk.0.attn_q.weight",
        np.random.default_rng(7).standard_normal((4, 8), dtype=np.float32),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    model = root / "model.gguf"
    _write_test_gguf(model)
    manifest = build_gguf_tensor_manifest(model)
    serve_dir = root / "serve"
    serve_dir.mkdir()
    filename = f"{MODEL_ID}.tensor-manifest.json"
    save_gguf_tensor_manifest(manifest, serve_dir / filename)
    data = (serve_dir / filename).read_bytes()
    index = {
        "schema": MESH_MANIFEST_INDEX_SCHEMA,
        "models": {
            MODEL_ID: {
                "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "tensor_manifest_root": manifest["tensor_manifest_root"],
            }
        },
    }
    (serve_dir / MESH_MANIFEST_INDEX_FILENAME).write_text(json.dumps(index))
    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "serve_dir": serve_dir,
            "root": str(manifest["tensor_manifest_root"]),
            "package_hash": gguf_package_hash(manifest).hex(),
        }
    finally:
        server.shutdown()


def test_fetch_verifies_and_caches(store, tmp_path):
    cache = tmp_path / "cache"
    path = fetch_mesh_tensor_manifest(
        MODEL_ID,
        expected_tensor_manifest_root=store["root"],
        expected_package_hash=store["package_hash"],
        base_urls=[store["base_url"]],
        cache_dir=cache,
    )
    assert path.is_file()
    assert store["root"] in path.name
    # Cache hit: no server needed the second time.
    again = fetch_mesh_tensor_manifest(
        MODEL_ID,
        expected_tensor_manifest_root=store["root"],
        base_urls=["http://127.0.0.1:1"],
        cache_dir=cache,
    )
    assert again == path


def test_fetch_refuses_wrong_committed_root(store, tmp_path):
    with pytest.raises(MeshManifestStoreError, match="committed root"):
        fetch_mesh_tensor_manifest(
            MODEL_ID,
            expected_tensor_manifest_root="ab" * 32,
            base_urls=[store["base_url"]],
            cache_dir=tmp_path / "cache",
        )
    assert not list((tmp_path / "cache").glob("*.json"))


def test_fetch_refuses_wrong_package_hash(store, tmp_path):
    with pytest.raises(MeshManifestStoreError, match="package hash"):
        fetch_mesh_tensor_manifest(
            MODEL_ID,
            expected_tensor_manifest_root=store["root"],
            expected_package_hash="cd" * 32,
            base_urls=[store["base_url"]],
            cache_dir=tmp_path / "cache",
        )


def test_fetch_refuses_tampered_bytes(store, tmp_path):
    index_path = store["serve_dir"] / MESH_MANIFEST_INDEX_FILENAME
    original = index_path.read_text()
    document = json.loads(original)
    document["models"][MODEL_ID]["sha256"] = "ef" * 32
    index_path.write_text(json.dumps(document))
    try:
        with pytest.raises(MeshManifestStoreError, match="sha256"):
            fetch_mesh_tensor_manifest(
                MODEL_ID,
                expected_tensor_manifest_root=store["root"],
                base_urls=[store["base_url"]],
                cache_dir=tmp_path / "cache",
            )
    finally:
        index_path.write_text(original)


def test_fetch_refuses_path_escaping_filename(store, tmp_path):
    index_path = store["serve_dir"] / MESH_MANIFEST_INDEX_FILENAME
    original = index_path.read_text()
    document = json.loads(original)
    document["models"][MODEL_ID]["filename"] = "../../etc/passwd"
    index_path.write_text(json.dumps(document))
    try:
        with pytest.raises(MeshManifestStoreError, match="basename"):
            fetch_mesh_tensor_manifest(
                MODEL_ID,
                expected_tensor_manifest_root=store["root"],
                base_urls=[store["base_url"]],
                cache_dir=tmp_path / "cache",
            )
    finally:
        index_path.write_text(original)


def test_fetch_requires_committed_root(store, tmp_path):
    with pytest.raises(MeshManifestStoreError, match="unanchored"):
        fetch_mesh_tensor_manifest(
            MODEL_ID,
            expected_tensor_manifest_root="",
            base_urls=[store["base_url"]],
            cache_dir=tmp_path / "cache",
        )


def test_unknown_model_reports_every_source(store, tmp_path):
    with pytest.raises(MeshManifestStoreError, match="not in the index"):
        fetch_mesh_tensor_manifest(
            "no-such-model",
            expected_tensor_manifest_root=store["root"],
            base_urls=[store["base_url"]],
            cache_dir=tmp_path / "cache",
        )


def test_normalize_rejects_non_http():
    with pytest.raises(MeshManifestStoreError):
        normalize_mesh_manifest_base_urls(["ftp://example.com/x"])
    assert normalize_mesh_manifest_base_urls(
        ["https://a.example/x/", "https://a.example/x", ""]
    ) == ("https://a.example/x",)


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(
        "VERATHOS_MESH_MANIFEST_BASE_URLS",
        "https://env.example/mesh/, https://env2.example/mesh",
    )
    assert configured_mesh_manifest_base_urls(["https://default.example"]) == (
        "https://env.example/mesh",
        "https://env2.example/mesh",
    )
    monkeypatch.delenv("VERATHOS_MESH_MANIFEST_BASE_URLS")
    assert configured_mesh_manifest_base_urls(["https://default.example"]) == (
        "https://default.example",
    )
