"""Subnet model catalog: the disk cache layer under the in-memory one.

A fresh CLI process (manage reopens many times a day) used to re-read
the chain on every start; the disk layer makes restarts instant and an
unreachable chain degrade to the last known catalog instead of an empty
one.
"""

from __future__ import annotations

import time

from verallm.mesh import model_catalog


ROWS = [
    {
        "model_id": "qwen-test-7b",
        "layers": 28,
        "quant": "q4_k_m",
        "manifest_urls": ["https://store.example/gleipnir/"],
        "tensor_manifest_root": "ab" * 32,
    }
]


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_catalog, "_subnet_catalog_cache", {})


def test_fresh_disk_cache_answers_without_the_chain(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    model_catalog._write_subnet_catalog_disk("testnet", ROWS)

    def _never(*a, **k):
        raise AssertionError("chain resolution ran despite a fresh cache")

    import verallm.mesh.onboarding as onboarding

    monkeypatch.setattr(onboarding, "resolve_network", _never)
    rows = model_catalog.subnet_model_catalog("testnet")
    assert [r["model_id"] for r in rows] == ["qwen-test-7b"]


def test_stale_disk_cache_beats_an_unreachable_chain(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    model_catalog._write_subnet_catalog_disk("testnet", ROWS)
    # Age the copy past the disk TTL so only the stale-fallback path can
    # serve it.
    import json

    path = model_catalog._subnet_catalog_cache_path("testnet")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fetched_at_unix"] = time.time() - 10 * 24 * 3600
    path.write_text(json.dumps(data), encoding="utf-8")

    import verallm.mesh.onboarding as onboarding

    def _unreachable(*a, **k):
        raise SystemExit("no chain config")

    monkeypatch.setattr(onboarding, "resolve_network", _unreachable)
    rows = model_catalog.subnet_model_catalog("testnet")
    assert [r["model_id"] for r in rows] == ["qwen-test-7b"]


def test_refresh_bypasses_the_disk_copy(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    model_catalog._write_subnet_catalog_disk("testnet", ROWS)

    import verallm.mesh.onboarding as onboarding

    calls: list[str] = []

    def _tracked(*a, **k):
        calls.append("resolve")
        raise SystemExit("no chain config")

    monkeypatch.setattr(onboarding, "resolve_network", _tracked)
    rows = model_catalog.subnet_model_catalog("testnet", refresh=True)
    # Refresh reached for the chain; with it unreachable the disk copy
    # still serves as the fallback.
    assert calls == ["resolve"]
    assert [r["model_id"] for r in rows] == ["qwen-test-7b"]


def test_no_cache_and_no_chain_yields_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    import verallm.mesh.onboarding as onboarding

    monkeypatch.setattr(
        onboarding,
        "resolve_network",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("nope")),
    )
    assert model_catalog.subnet_model_catalog("testnet") == []
