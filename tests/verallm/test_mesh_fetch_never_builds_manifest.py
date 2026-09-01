"""A fetching box NEVER builds a tensor manifest.

Manifests are owner-built and distributed (store download or a
pre-staged owner file); a rebuild on the fetching side is always a
broken distribution path, dev pools included. These tests ban the
builder from the fetch path at the source level and pin every accepted
and refused acquisition behavior, so the enforcement cannot silently
regress to a per-mode carve-out again.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from verallm.mesh import pool as pool_mod
from verallm.mesh.manifest_store import (
    MeshManifestStoreError,
    all_default_store_urls,
    default_store_urls_for_chain,
)
from verallm.mesh.pool import LocalMeshRunner, _acquire_owner_manifest


def test_fetch_path_never_references_the_manifest_builder():
    """Source-level ban: neither the fetch method nor the acquisition
    helper may reference the builder. Owner tooling keeps its own
    builders elsewhere; the fetch path is download-or-refuse."""
    for fn in (LocalMeshRunner.fetch, _acquire_owner_manifest):
        source = inspect.getsource(fn)
        assert "build_gguf_tensor_manifest" not in source, fn.__qualname__


def test_store_fetch_failure_refuses_instead_of_building(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise MeshManifestStoreError("store down")

    monkeypatch.setattr(
        "verallm.mesh.manifest_store.fetch_mesh_tensor_manifest", boom
    )
    spec = {
        "model_tensor_manifest_root": "ab" * 32,
        "manifest_urls": ["https://example.invalid/store"],
    }
    with pytest.raises(ValueError, match="never rebuilds a manifest"):
        _acquire_owner_manifest(
            model_id="m", spec=spec, dest_dir=tmp_path, progress=None
        )


def test_store_fetch_failure_uses_matching_prestaged_manifest(
    tmp_path, monkeypatch
):
    root = "ab" * 32

    def boom(*_a, **_k):
        raise MeshManifestStoreError("store down")

    monkeypatch.setattr(
        "verallm.mesh.manifest_store.fetch_mesh_tensor_manifest", boom
    )
    monkeypatch.setattr(
        "verallm.mesh.gguf_manifest.load_gguf_tensor_manifest",
        lambda _p: {"tensor_manifest_root": root},
    )
    staged = tmp_path / "tensor-manifest.json"
    staged.write_text("{}")

    got = _acquire_owner_manifest(
        model_id="m",
        spec={
            "model_tensor_manifest_root": root,
            "manifest_urls": ["https://example.invalid/store"],
        },
        dest_dir=tmp_path,
        progress=None,
    )

    assert got == str(staged)


def test_nothing_available_refuses_instead_of_building(tmp_path):
    with pytest.raises(ValueError, match="never builds manifests"):
        _acquire_owner_manifest(
            model_id="m", spec={}, dest_dir=tmp_path, progress=None
        )


def test_prestaged_owner_manifest_accepted_when_root_matches(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "verallm.mesh.gguf_manifest.load_gguf_tensor_manifest",
        lambda _p: {"tensor_manifest_root": "cd" * 32},
    )
    staged = tmp_path / "tensor-manifest.json"
    staged.write_text("{}")
    got = _acquire_owner_manifest(
        model_id="m",
        spec={"model_tensor_manifest_root": "cd" * 32},
        dest_dir=tmp_path,
        progress=None,
    )
    assert got == str(staged)


def test_prestaged_manifest_with_wrong_root_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "verallm.mesh.gguf_manifest.load_gguf_tensor_manifest",
        lambda _p: {"tensor_manifest_root": "ee" * 32},
    )
    (tmp_path / "tensor-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing it"):
        _acquire_owner_manifest(
            model_id="m",
            spec={"model_tensor_manifest_root": "cd" * 32},
            dest_dir=tmp_path,
            progress=None,
        )


def test_prestaged_manifest_accepted_for_unrooted_dev_model(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "verallm.mesh.gguf_manifest.load_gguf_tensor_manifest",
        lambda _p: {"tensor_manifest_root": "any" * 16},
    )
    staged = tmp_path / "tensor-manifest.json"
    staged.write_text("{}")
    got = _acquire_owner_manifest(
        model_id="m", spec={}, dest_dir=tmp_path, progress=None
    )
    assert got == str(staged)


def test_chainless_pools_fall_back_to_the_union_of_stores():
    """A pool with no chain binding must still reach the owner's stores:
    the root is content-addressed, so querying every known store is
    safe, and without this fallback a dev pool silently loses manifest
    distribution entirely."""
    assert default_store_urls_for_chain(None) == ()
    union = all_default_store_urls()
    assert union, "no default stores known"
    for chain_id in (945, 964):
        for url in default_store_urls_for_chain(chain_id):
            assert url in union


def test_shipped_fetch_spec_carries_store_urls_without_a_chain():
    spec = pool_mod._shipped_model_fetch_spec(
        "qwen3.8-27b-q4-k-m", chain_id=None
    )
    assert spec is not None
    assert spec.get("model_tensor_manifest_root")
    assert spec.get("manifest_urls"), (
        "chainless spec lost its store URLs; dev pools would rebuild"
    )
