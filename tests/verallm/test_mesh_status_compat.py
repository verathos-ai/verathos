"""`verathos mesh status` grew a no-argument worker view; the original
mesh-spec form must stay byte-identical for existing scripts."""
from __future__ import annotations

from pathlib import Path

import pytest

from verallm.mesh import cli as mesh_cli


def test_status_with_mesh_path_parses_exactly_as_before():
    parser = mesh_cli.build_parser()
    args = parser.parse_args(["status", "/some/mesh"])
    assert args.mesh == "/some/mesh"
    assert args.func is mesh_cli.cmd_status
    assert args.probe is False
    args = parser.parse_args(["status", "/some/mesh", "--probe"])
    assert args.mesh == "/some/mesh"
    assert args.probe is True


def test_status_without_mesh_selects_worker_view(monkeypatch):
    parser = mesh_cli.build_parser()
    args = parser.parse_args(["status"])
    assert args.mesh == ""
    seen = []
    monkeypatch.setattr(
        mesh_cli, "cmd_worker_status", lambda a: seen.append(a)
    )
    mesh_cli.cmd_status(args)
    assert len(seen) == 1


def test_probe_without_mesh_path_exits():
    parser = mesh_cli.build_parser()
    args = parser.parse_args(["status", "--probe"])
    with pytest.raises(SystemExit, match="--probe requires"):
        mesh_cli.cmd_status(args)


def test_worker_status_without_any_state_prints_setup_hint(
    monkeypatch, tmp_path, capsys
):
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module, "UNIT_REGISTRY_PATH", tmp_path / "absent.json"
    )
    monkeypatch.setattr(mesh_cli, "_local_pool_views", lambda: [])
    monkeypatch.setattr(mesh_cli, "_local_mesh_processes", lambda: [])
    parser = mesh_cli.build_parser()
    mesh_cli.cmd_status(parser.parse_args(["status"]))
    out = capsys.readouterr().out
    assert "has not joined" in out
    assert "verathos mesh setup" in out


def test_worker_status_sees_a_pool_without_a_unit_registry(
    monkeypatch, tmp_path, capsys
):
    """The coordinator-box regression: manually launched `pool worker`
    processes have no unit registry, and status once claimed the machine
    had never joined a pool while a manager and two workers served on it."""

    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module, "UNIT_REGISTRY_PATH", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        mesh_cli,
        "_local_pool_views",
        lambda: [
            {
                "pool_dir": str(tmp_path / "pool-x"),
                "pool_id": "pool-x",
                "manager_reachable": True,
                "workers": {
                    "w-a": {
                        "status": "serving",
                        "stale": False,
                        "mesh": "m-1",
                        "capability": {"gpu_name": "RTX 5090"},
                    }
                },
                "meshes": {
                    "m-1": {
                        "status": "serving",
                        "routing_ready": True,
                        "model_id": "q7b",
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(
        mesh_cli,
        "_local_mesh_processes",
        lambda: [
            {"pid": 42, "kind": "manager", "pm2": False, "args": "pool serve"},
            {"pid": 43, "kind": "worker", "pm2": False, "args": "pool worker"},
        ],
    )
    parser = mesh_cli.build_parser()
    mesh_cli.cmd_status(parser.parse_args(["status"]))
    out = capsys.readouterr().out
    assert "has not joined" not in out
    assert "pool-x" in out
    assert "manager healthy" in out
    assert "w-a" in out and "m-1" in out
    assert "[manual]" in out and "pid" in out


def test_worker_status_marks_unroutable_serving_meshes(
    monkeypatch, tmp_path, capsys
):
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module, "UNIT_REGISTRY_PATH", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        mesh_cli,
        "_local_pool_views",
        lambda: [
            {
                "pool_dir": str(tmp_path / "pool-x"),
                "pool_id": "pool-x",
                "manager_reachable": True,
                "workers": {},
                "meshes": {
                    "m-dead": {
                        "status": "serving",
                        "routing_ready": False,
                        "model_id": "q7b",
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(mesh_cli, "_local_mesh_processes", lambda: [])
    parser = mesh_cli.build_parser()
    mesh_cli.cmd_status(parser.parse_args(["status"]))
    out = capsys.readouterr().out
    assert "NOT routable" in out


def test_worker_status_json_bundles_all_sources(monkeypatch, tmp_path, capsys):
    import json as json_module

    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module, "UNIT_REGISTRY_PATH", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        mesh_cli,
        "_local_pool_views",
        lambda: [
            {
                "pool_dir": "/p",
                "pool_id": "pool-x",
                "manager_reachable": False,
                "workers": {},
                "meshes": {},
            }
        ],
    )
    monkeypatch.setattr(
        mesh_cli,
        "_local_mesh_processes",
        lambda: [{"pid": 7, "kind": "worker", "pm2": True, "args": "x"}],
    )
    parser = mesh_cli.build_parser()
    mesh_cli.cmd_status(parser.parse_args(["status", "--json"]))
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["units"] is None
    assert payload["pools"][0]["pool_id"] == "pool-x"
    assert payload["processes"][0]["pid"] == 7


def test_worker_status_payload_joins_pm2_and_pool_views(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    units = units_module.plan_worker_units(
        [
            units_module.GpuInfo(index=0, name="RTX 4090", vram_gb=24),
            units_module.GpuInfo(index=1, name="RTX 4090", vram_gb=24),
        ],
        worker_id_base="pod",
        home=tmp_path,
        groups=[(0,), (1,)],
    )
    registry = {
        "version": 1,
        "manager_endpoint": "http://pool.example:9500",
        "pool_id": "pool-x",
        "token_file": str(tmp_path / "token"),
        "units": [unit.__dict__ for unit in units],
    }
    monkeypatch.setattr(
        mesh_cli,
        "_pm2_jlist",
        lambda: [
            {"name": "verathos-mesh-pod-gpu0", "pm2_env": {"status": "online"}},
            {"name": "verathos-mesh-pod-gpu1", "pm2_env": {"status": "stopped"}},
        ],
    )

    def fake_post_json(url, body, timeout=0):
        assert url.endswith("/v1/operator/overview")
        return {
            "workers": {
                "pod-gpu0": {"status": "driving", "mesh": "m-1", "stale": False},
                "pod-gpu1": {"status": "idle", "mesh": "", "stale": False},
            },
            "meshes": {
                "m-1": {"driver": "pod-gpu0", "model_id": "m1", "status": "serving"}
            },
        }

    import verallm.mesh.worker as worker_module

    monkeypatch.setattr(worker_module, "post_json", fake_post_json)
    payload = mesh_cli._worker_status_payload(registry)
    rows = {row["worker_id"]: row for row in payload["units"]}
    assert rows["pod-gpu0"]["pm2_status"] == "online"
    assert rows["pod-gpu0"]["role"] == "driver"
    assert rows["pod-gpu0"]["model_id"] == "m1"
    assert rows["pod-gpu1"]["pm2_status"] == "stopped"
    assert rows["pod-gpu1"]["role"] == ""


def test_resolve_pm2_targets(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    units = units_module.plan_worker_units(
        [
            units_module.GpuInfo(index=0, name="a", vram_gb=24),
            units_module.GpuInfo(index=1, name="b", vram_gb=24),
        ],
        worker_id_base="pod",
        home=tmp_path,
        groups=[(0,), (1,)],
    )
    registry = {
        "manager_endpoint": "",
        "pool_id": "",
        "token_file": "",
        "units": [unit.__dict__ for unit in units],
    }
    monkeypatch.setattr(
        units_module, "load_unit_registry", lambda path=None: registry
    )
    import argparse

    args = argparse.Namespace(target="pod-gpu1", manager=False, all=False)
    assert mesh_cli._resolve_pm2_targets(args) == ["verathos-mesh-pod-gpu1"]
    args = argparse.Namespace(target="", manager=True, all=False)
    assert mesh_cli._resolve_pm2_targets(args) == ["verathos-pool-manager"]
    # Two units and no target: refuse rather than fan out implicitly.
    args = argparse.Namespace(target="", manager=False, all=False)
    with pytest.raises(SystemExit, match="several mesh units"):
        mesh_cli._resolve_pm2_targets(args)
    # --all is an explicit fan-out where the verb allows it.
    args = argparse.Namespace(target="", manager=False, all=True)
    assert mesh_cli._resolve_pm2_targets(args, allow_all=True) == [
        "verathos-mesh-pod-gpu0",
        "verathos-mesh-pod-gpu1",
    ]
