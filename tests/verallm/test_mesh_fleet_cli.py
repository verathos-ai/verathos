"""`verathos mesh fleet`: rendering and token discovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verallm.mesh import cli as mesh_cli
from verallm.mesh.cli import render_fleet
from verallm.mesh.pool import (
    POOL_ADMIN_TOKEN_FILE,
    MeshPoolToken,
    PoolManager,
    create_pool_state,
    load_pool_token_file,
)


def _manager(tmp_path) -> PoolManager:
    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    return PoolManager(state_dir)


def _join(manager: PoolManager, worker_id: str, *, vram: int = 24) -> None:
    manager.handle_join(
        {
            "pool_secret": manager.state["pool_secret"],
            "worker_id": worker_id,
            "capability": {
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "vram_gb": vram,
                "rpc_device": "CUDA0",
                "free_disk_gb": 90.0,
            },
            "catalog": [
                {
                    "model_id": "m1",
                    "model_bytes": 4_000_000_000,
                    "layers": 28,
                    "hf_repo": "org/model",
                    "hf_files": ["model.gguf"],
                }
            ],
            "endpoints": {
                "rpc": "198.51.100.7:50052",
                "proof": "http://198.51.100.7:9402",
                "mesh": "http://198.51.100.7:9443",
            },
        }
    )


def test_render_fleet_against_real_overview_payload(tmp_path):
    """The renderer consumes the actual handle_operator_overview schema, so
    it cannot drift from the pool's response shape."""
    manager = _manager(tmp_path)
    _join(manager, "pod-gpu0")
    _join(manager, "pod-gpu1")
    overview = manager.handle_operator_overview(
        {"management_secret": manager.state["management_secret"]}
    )
    suggestions, reasons = manager.recommend("m1")
    text = render_fleet(
        overview,
        suggestions=suggestions,
        reasons=reasons,
        model_id="m1",
        managed=True,
    )
    # Machine grouping: the shared host prefix is shown once and the
    # per-GPU rows use the short names.
    assert "host pod" in text
    assert "gpu0" in text and "gpu1" in text
    assert "NVIDIA GeForce RTX 4090" in text
    # Placement advice appears; limit reasons print verbatim, while
    # "available (...)" reasons are filtered — a healthy worker's row in
    # the table above already says it, and repeating it as a caveat
    # buries the real limits.
    assert "placement for m1:" in text
    for reason in reasons.values():
        if str(reason).startswith("available"):
            assert reason not in text
        else:
            assert reason in text
    # No em-dashes anywhere in operator-facing output.
    assert "\u2014" not in text


def test_render_fleet_public_view_banner():
    text = render_fleet(
        {"pool_id": "pool-x", "serving_mode": "dev", "workers": {}, "meshes": {}},
        managed=False,
    )
    assert "public view" in text
    assert "none joined yet" in text
    assert "\u2014" not in text


def test_render_fleet_mesh_rows_and_warns():
    overview = {
        "pool_id": "pool-x",
        "serving_mode": "validator",
        "owner_account": "5Owner",
        "workers": {
            "solo": {
                "status": "driving",
                "mesh": "m-abc",
                "rtt_ms": {"manager": 1.2},
                "capability": {"gpu_name": "A100", "vram_gb": 80},
            }
        },
        "meshes": {
            "m-abc": {
                "model_id": "m1",
                "driver": "solo",
                "status": "serving",
                "routing_ready": True,
                "validator_score": 0.42,
                "error": "",
            }
        },
        "models": {"m1": {"layers": 28, "model_bytes": 4e9, "launch_ready": True}},
    }
    warn = (
        "95 ms link: every decode token crosses it (~5 tok/s ceiling); "
        "co-located workers run at full speed"
    )
    text = render_fleet(
        overview,
        suggestions=[
            {
                "workers": ["solo", "far"],
                "driver": "solo",
                "max_rtt_ms": 95.0,
                "link_class": "far",
                "driver_vram_gb": 80.0,
                "warn": warn,
            }
        ],
        reasons={
            "far": "available (file-less member slice)",
            "slowbox": "member only: downloading needs ~5 GB free disk",
        },
        model_id="m1",
    )
    assert "serving" in text
    assert "score 0.42" in text
    assert "launch-ready" in text
    assert warn in text
    # Limits render under "worker limits:"; "available (...)" reasons are
    # filtered because the suggestion table above already answers them.
    assert "worker limits:" in text
    assert "slowbox: member only: downloading needs ~5 GB free disk" in text
    assert "available (file-less member slice)" not in text


_STYLE_OVERVIEW = {
    "pool_id": "pool-x",
    "serving_mode": "dev",
    "workers": {
        "pod-gpu0": {
            "status": "online",
            "mesh": "m-abc",
            "rtt_ms": {"manager": 1.0},
            "capability": {"gpu_name": "RTX 5090", "vram_gb": 31},
        }
    },
    "meshes": {
        "m-abc": {
            "model_id": "m1",
            "driver": "pod-gpu0",
            "status": "serving",
            "error": "boom",
        }
    },
    "models": {"m1": {"layers": 28, "launch_ready": False}},
}


def test_render_fleet_unstyled_output_has_no_ansi():
    """Agents and pipes must keep getting the plain text they parse."""

    text = render_fleet(_STYLE_OVERVIEW, managed=True, styled=False)
    assert "\033[" not in text


def test_render_fleet_styled_only_adds_color_never_changes_text():
    """Styling is color-only: strip the ANSI and the bytes are identical."""

    import re

    plain = render_fleet(_STYLE_OVERVIEW, managed=True, styled=False)
    styled = render_fleet(_STYLE_OVERVIEW, managed=True, styled=True)
    assert "\033[" in styled
    assert re.sub(r"\033\[[0-9;]*m", "", styled) == plain


def test_resolve_pool_context_prefers_explicit_then_pool_dir(tmp_path, monkeypatch):
    import argparse

    state_dir, worker_token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    args = argparse.Namespace(
        pool_token_file="", pool_token="", pool=str(state_dir), manager_ca_file=""
    )
    token = mesh_cli._resolve_pool_context(args)
    # The pool dir resolves the ADMIN token, not the worker token.
    assert token.scope == "management"
    admin = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    assert token.pool_secret == admin.pool_secret

    inline = worker_token.encode()
    args = argparse.Namespace(
        pool_token_file="", pool_token=inline, pool=str(state_dir), manager_ca_file=""
    )
    assert mesh_cli._resolve_pool_context(args).scope == "worker"


def test_resolve_pool_context_discovers_single_pool(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    pools_root = tmp_path / ".verathos" / "pools"
    create_pool_state(
        pools_root, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    args = argparse.Namespace(
        pool_token_file="", pool_token="", pool="", manager_ca_file=""
    )
    monkeypatch.delenv("VERATHOS_POOL_TOKEN_FILE", raising=False)
    assert mesh_cli._resolve_pool_context(args).scope == "management"

    # A second pool makes discovery ambiguous.
    create_pool_state(
        pools_root, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    with pytest.raises(SystemExit, match="several pools"):
        mesh_cli._resolve_pool_context(args)


def test_resolve_pool_context_worker_box_falls_back_to_unit_registry(
    tmp_path, monkeypatch
):
    import argparse

    from verallm.mesh import units as units_module
    from verallm.mesh.private_files import write_owner_only_text

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("VERATHOS_POOL_TOKEN_FILE", raising=False)
    token = MeshPoolToken(
        pool_id="pool-w",
        manager_endpoint="https://pool.example:9500",
        pool_secret="s3cret",
    )
    token_file = tmp_path / "pool-token.txt"
    write_owner_only_text(token_file, token.encode())
    units = units_module.plan_worker_units(
        [units_module.GpuInfo(index=0, name="RTX 4090", vram_gb=24)],
        worker_id_base="pod",
        home=tmp_path,
    )
    units_module.save_unit_registry(
        units,
        manager_endpoint=token.manager_endpoint,
        pool_id=token.pool_id,
        token_file=str(token_file),
        path=tmp_path / ".verathos" / "mesh-units.json",
    )
    monkeypatch.setattr(
        units_module, "UNIT_REGISTRY_PATH", tmp_path / ".verathos" / "mesh-units.json"
    )
    args = argparse.Namespace(
        pool_token_file="", pool_token="", pool="", manager_ca_file=""
    )
    resolved = mesh_cli._resolve_pool_context(args)
    assert resolved.scope == "worker"
    assert resolved.manager_endpoint == "https://pool.example:9500"
