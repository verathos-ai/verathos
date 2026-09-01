"""Per-GPU worker unit planning: ids, ports, registry round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verallm.mesh import units as units_module
from verallm.mesh.units import (
    DEFAULT_MESH_PORT_BASE,
    DEFAULT_PROOF_PORT_BASE,
    DEFAULT_RPC_PORT_BASE,
    GpuInfo,
    MAX_UNITS_PER_HOST,
    WorkerUnit,
    detect_gpus,
    load_unit_registry,
    plan_worker_units,
    sanitize_worker_id_base,
    save_unit_registry,
    units_from_registry,
)

_WORKER_ID_RE = units_module._WORKER_ID_RE


def _gpus(count: int) -> list[GpuInfo]:
    return [
        GpuInfo(index=i, name=f"NVIDIA GeForce RTX 4090 #{i}", vram_gb=24)
        for i in range(count)
    ]


def test_unit_zero_reproduces_single_gpu_defaults(tmp_path):
    unit = plan_worker_units(
        _gpus(1), worker_id_base="pod", home=tmp_path
    )[0]
    assert unit.worker_id == "pod-gpu0"
    assert unit.pm2_name == "verathos-mesh-pod-gpu0"
    assert unit.rpc_port == DEFAULT_RPC_PORT_BASE
    assert unit.proof_port == DEFAULT_PROOF_PORT_BASE
    assert unit.mesh_port == DEFAULT_MESH_PORT_BASE
    assert unit.cuda_visible_devices == "0"
    # Inside the visibility mask the unit's GPU is always device 0.
    assert unit.rpc_device == "CUDA0"
    assert unit.workdir == str(tmp_path / ".verathos" / "poolwork-gpu0")


def test_eight_units_have_no_port_or_path_collisions(tmp_path):
    units = plan_worker_units(
        _gpus(8),
        worker_id_base="pod",
        home=tmp_path,
        groups=[(i,) for i in range(8)],
    )
    assert len(units) == 8
    ports: set[int] = set()
    for unit in units:
        # The driver's llama-server binds mesh_port + 1 as well.
        unit_ports = {
            unit.rpc_port,
            unit.proof_port,
            unit.mesh_port,
            unit.mesh_port + 1,
        }
        assert len(unit_ports) == 4
        assert not ports.intersection(unit_ports)
        ports.update(unit_ports)
        assert _WORKER_ID_RE.fullmatch(unit.worker_id)
        assert unit.cuda_visible_devices == str(unit.gpu_index)
        assert unit.rpc_device == "CUDA0"
    assert len({unit.workdir for unit in units}) == 8
    assert len({unit.catalog for unit in units}) == 8
    assert max(unit.mesh_port + 1 for unit in units) < 9500


def test_gpu_subset_packs_sequential_port_slots(tmp_path):
    subset = [GpuInfo(index=0, name="a", vram_gb=24), GpuInfo(index=3, name="b", vram_gb=24)]
    units = plan_worker_units(
        subset, worker_id_base="pod", home=tmp_path, groups=[(0,), (3,)]
    )
    assert [unit.worker_id for unit in units] == ["pod-gpu0", "pod-gpu3"]
    # Port slots are sequential, not GPU-index-derived.
    assert [unit.rpc_port for unit in units] == [
        DEFAULT_RPC_PORT_BASE,
        DEFAULT_RPC_PORT_BASE + 1,
    ]
    # CUDA masking still follows the real GPU index.
    assert [unit.cuda_visible_devices for unit in units] == ["0", "3"]


def test_hostname_sanitization():
    assert sanitize_worker_id_base("My Host.local") == "my-host.local"
    assert sanitize_worker_id_base("pod: gpu box!") == "pod-gpu-box"
    assert sanitize_worker_id_base("") == "worker"
    assert sanitize_worker_id_base("---") == "worker"
    long = sanitize_worker_id_base("x" * 200)
    assert _WORKER_ID_RE.fullmatch(long + "-gpu27")


def test_too_many_gpus_refused(tmp_path):
    count = MAX_UNITS_PER_HOST + 1
    with pytest.raises(ValueError, match="port budget"):
        plan_worker_units(
            _gpus(count),
            worker_id_base="pod",
            home=tmp_path,
            groups=[(i,) for i in range(count)],
        )
    with pytest.raises(ValueError, match="no GPUs"):
        plan_worker_units([], worker_id_base="pod", home=tmp_path)


def test_registry_round_trip(tmp_path):
    units = plan_worker_units(
        _gpus(2), worker_id_base="pod", home=tmp_path, groups=[(0,), (1,)]
    )
    path = tmp_path / "mesh-units.json"
    save_unit_registry(
        units,
        manager_endpoint="https://pool.example:9500",
        pool_id="pool-abc",
        token_file=str(tmp_path / "token.txt"),
        path=path,
    )
    assert path.stat().st_mode & 0o777 == 0o600
    registry = load_unit_registry(path)
    assert registry["manager_endpoint"] == "https://pool.example:9500"
    assert registry["pool_id"] == "pool-abc"
    loaded = units_from_registry(registry)
    assert loaded == units
    assert load_unit_registry(tmp_path / "absent.json") is None


def test_detect_gpus_parses_csv_with_commas(monkeypatch):
    class _Result:
        stdout = (
            "0, NVIDIA A100-SXM4-80GB, 81920\n"
            "1, Odd, Name, With Commas, 24564\n"
        )

    monkeypatch.setattr(
        units_module.subprocess,
        "run",
        lambda *args, **kwargs: _Result(),
    )
    gpus = detect_gpus()
    assert gpus[0] == GpuInfo(index=0, name="NVIDIA A100-SXM4-80GB", vram_gb=80)
    assert gpus[1].index == 1
    assert gpus[1].name == "Odd, Name, With Commas"
    assert gpus[1].vram_gb == 23


def test_detect_gpus_without_nvidia_smi(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(units_module.subprocess, "run", _raise)
    assert detect_gpus() == []


def test_plan_units_cli_json(tmp_path, monkeypatch, capsys):
    from verallm.mesh import cli as mesh_cli

    monkeypatch.setattr(
        units_module, "detect_gpus", lambda: _gpus(2)
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    mesh_cli.main(
        ["plan-units", "--worker-id-base", "My Pod", "--gpus", "1"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["units"]) == 1
    unit = payload["units"][0]
    assert unit["worker_id"] == "my-pod-gpu1"
    assert unit["cuda_visible_devices"] == "1"
    assert unit["rpc_port"] == DEFAULT_RPC_PORT_BASE


def test_default_plan_is_one_worker_owning_all_gpus(tmp_path):
    gpus = [
        GpuInfo(index=i, name="NVIDIA A100-SXM4-80GB", vram_gb=80)
        for i in range(4)
    ]
    units = plan_worker_units(gpus, worker_id_base="pod", home=tmp_path)
    assert len(units) == 1
    unit = units[0]
    assert unit.worker_id == "pod-gpu0-3"
    assert unit.cuda_visible_devices == "0,1,2,3"
    assert unit.rpc_device == "CUDA0,CUDA1,CUDA2,CUDA3"
    assert unit.gpu_indices == [0, 1, 2, 3]
    assert unit.per_gpu_vram_gb == [80, 80, 80, 80]
    assert unit.vram_gb == 320
    assert unit.gpu_name == "NVIDIA A100-SXM4-80GB x4"


def test_grouped_plan_mixes_single_and_multi_gpu_units(tmp_path):
    groups = units_module.parse_gpu_groups("0,1,2+3", [0, 1, 2, 3])
    assert groups == [(0,), (1,), (2, 3)]
    units = plan_worker_units(
        _gpus(4), worker_id_base="pod", home=tmp_path, groups=groups
    )
    assert [u.worker_id for u in units] == ["pod-gpu0", "pod-gpu1", "pod-gpu2-3"]
    pair = units[2]
    assert pair.cuda_visible_devices == "2,3"
    assert pair.rpc_device == "CUDA0,CUDA1"
    assert pair.per_gpu_vram_gb == [24, 24]
    assert pair.vram_gb == 48
    assert pair.workdir.endswith("poolwork-gpu2-3")
    # Port slots stay sequential per unit, not per GPU.
    assert [u.rpc_port for u in units] == [
        DEFAULT_RPC_PORT_BASE,
        DEFAULT_RPC_PORT_BASE + 1,
        DEFAULT_RPC_PORT_BASE + 2,
    ]


def test_parse_gpu_groups_grammar():
    assert units_module.parse_gpu_groups("", [0, 1]) == [(0, 1)]
    assert units_module.parse_gpu_groups("all", [0, 1]) == [(0, 1)]
    assert units_module.parse_gpu_groups("0,1", [0, 1]) == [(0,), (1,)]
    with pytest.raises(ValueError, match="more than one group"):
        units_module.parse_gpu_groups("0,0+1", [0, 1])
    with pytest.raises(ValueError, match="not present"):
        units_module.parse_gpu_groups("0,5", [0, 1])
    with pytest.raises(ValueError, match="repeats"):
        units_module.parse_gpu_groups("1+1", [0, 1])


def test_gpu_group_label_forms():
    assert units_module.gpu_group_label([2]) == "2"
    assert units_module.gpu_group_label([2, 3]) == "2-3"
    assert units_module.gpu_group_label([0, 1, 2, 3]) == "0-3"
    assert units_module.gpu_group_label([0, 2]) == "0.2"


def test_legacy_registry_without_group_fields_loads(tmp_path):
    unit = {
        "worker_id": "pod-gpu0",
        "pm2_name": "verathos-mesh-pod-gpu0",
        "gpu_index": 0,
        "gpu_name": "A100",
        "vram_gb": 80,
        "cuda_visible_devices": "0",
        "rpc_device": "CUDA0",
        "rpc_port": 50052,
        "proof_port": 9402,
        "mesh_port": 9443,
        "workdir": "/w",
        "catalog": "/c",
    }
    loaded = units_from_registry({"units": [unit]})
    assert loaded[0].gpu_indices == []
    assert loaded[0].per_gpu_vram_gb == []
