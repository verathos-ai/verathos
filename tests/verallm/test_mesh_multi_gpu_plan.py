"""Multi-GPU workers: per-device tensor split, plan expansion, binding.

A worker owning several GPUs runs ONE rpc-server exposing one llama device
per GPU. llama numbers remote devices RPC0..RPCn-1 globally in --rpc
endpoint (= committed stage) order, so a two-member mesh whose first member
owns 2 GPUs binds --device RPC0,RPC1,RPC2 with a per-DEVICE tensor split.
The member's committed layer range is the union of its devices' contiguous
ranges; stage proofs stay per WORKER.
"""

from __future__ import annotations

import pytest

from verallm.mesh.cli import validate_all_rpc_runtime_binding
from verallm.mesh.llama_cpp import (
    MeshRpcPlan,
    llama_tensor_split_layer_ranges,
    rpc_plan_from_mesh,
)
from verallm.mesh.state import assign_mesh_members
from verallm.mesh.types import CapabilityAd, MeshMember, MeshSpec, StageRange

TOTAL_LAYERS = 40


def _spec(members: list[MeshMember]) -> MeshSpec:
    coordinator = MeshMember(
        uid=0,
        hotkey="coordinator-hotkey",
        endpoint="http://coordinator:9443",
        stage_index=0,
        layers=StageRange(0, 0),
        role="coordinator",
        backend="gguf_stage",
    )
    return MeshSpec(
        mesh_id="mesh-multi-gpu-test",
        mode="private",
        coordinator_uid=0,
        coordinator_hotkey="coordinator-hotkey",
        model_id="test/model",
        model_package_hash="ab" * 32,
        total_layers=TOTAL_LAYERS,
        members=[coordinator, *members],
    )


def _capability(
    idx: int, *, vram_gb: int, per_gpu: list[int] | None = None
) -> CapabilityAd:
    return CapabilityAd(
        uid=idx,
        hotkey=f"worker-{idx}",
        endpoint=f"http://worker-{idx}:9443",
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=vram_gb,
        per_gpu_vram_gb=per_gpu or [],
        rpc_endpoint=f"10.0.0.{idx}:50052",
        proof_endpoint=f"http://10.0.0.{idx}:9402",
    )


def _coordinator_capability() -> CapabilityAd:
    return CapabilityAd(
        uid=0,
        hotkey="coordinator-hotkey",
        endpoint="http://coordinator:9443",
        supported_backends=["gguf_stage"],
    )


def test_assign_splits_over_devices_and_unions_member_ranges() -> None:
    spec = _spec([])
    assigned = assign_mesh_members(
        spec,
        [
            _coordinator_capability(),
            _capability(1, vram_gb=160, per_gpu=[80, 80]),
            _capability(2, vram_gb=80),
        ],
        coordinator_computes=False,
    )
    device_ranges = llama_tensor_split_layer_ranges(TOTAL_LAYERS, [80, 80, 80])
    workers = sorted(
        (m for m in assigned.members if m.role != "coordinator"),
        key=lambda m: m.stage_index,
    )
    pair, single = workers
    # The 2-GPU member's range is the union of its two device ranges.
    assert pair.layers == StageRange(
        device_ranges[0].start, device_ranges[1].end
    )
    assert single.layers == device_ranges[2]
    assert pair.rpc_split_weight == 160
    assert pair.rpc_device_weights == [80, 80]
    assert single.rpc_split_weight == 80
    assert single.rpc_device_weights == []


def test_rpc_plan_expands_devices_and_validates_union_ranges() -> None:
    device_ranges = llama_tensor_split_layer_ranges(TOTAL_LAYERS, [80, 80, 80])
    pair = MeshMember(
        uid=1,
        hotkey="worker-1",
        endpoint="http://worker-1:9443",
        stage_index=1,
        layers=StageRange(device_ranges[0].start, device_ranges[1].end),
        rpc_endpoint="10.0.0.1:50052",
        rpc_split_weight=160,
        rpc_device_weights=[80, 80],
    )
    single = MeshMember(
        uid=2,
        hotkey="worker-2",
        endpoint="http://worker-2:9443",
        stage_index=2,
        layers=device_ranges[2],
        rpc_endpoint="10.0.0.2:50052",
        rpc_split_weight=80,
    )
    plan = rpc_plan_from_mesh(_spec([pair, single]))
    assert plan.rpc_endpoints == ["10.0.0.1:50052", "10.0.0.2:50052"]
    assert plan.total_devices == 3
    assert plan.tensor_split_arg == "80,80,80"
    # A member range that is NOT the union of its device ranges is rejected.
    wrong = MeshMember(
        uid=1,
        hotkey="worker-1",
        endpoint="http://worker-1:9443",
        stage_index=1,
        layers=StageRange(0, TOTAL_LAYERS),
        rpc_endpoint="10.0.0.1:50052",
        rpc_split_weight=160,
        rpc_device_weights=[80, 80],
    )
    with pytest.raises(ValueError, match="do not match committed tensor split"):
        rpc_plan_from_mesh(_spec([wrong, single]))


def test_binding_expects_one_rpc_device_per_gpu() -> None:
    device_ranges = llama_tensor_split_layer_ranges(TOTAL_LAYERS, [80, 80, 80])
    pair = MeshMember(
        uid=1,
        hotkey="worker-1",
        endpoint="http://worker-1:9443",
        stage_index=1,
        layers=StageRange(device_ranges[0].start, device_ranges[1].end),
        rpc_endpoint="10.0.0.1:50052",
        rpc_split_weight=160,
        rpc_device_weights=[80, 80],
    )
    single = MeshMember(
        uid=2,
        hotkey="worker-2",
        endpoint="http://worker-2:9443",
        stage_index=2,
        layers=device_ranges[2],
        rpc_endpoint="10.0.0.2:50052",
        rpc_split_weight=80,
    )
    spec = _spec([pair, single])
    plan = rpc_plan_from_mesh(spec)
    validate_all_rpc_runtime_binding(
        spec, plan, device="RPC0,RPC1,RPC2", n_gpu_layers="all"
    )
    with pytest.raises(ValueError, match="expected RPC0,RPC1,RPC2"):
        validate_all_rpc_runtime_binding(
            spec, plan, device="RPC0,RPC1", n_gpu_layers="all"
        )


def test_single_device_plan_hash_is_unchanged_by_device_weight_default() -> None:
    explicit = MeshRpcPlan(
        mesh_id="m",
        rpc_endpoints=["10.0.0.1:50052"],
        rpc_split_weights=[80],
        rpc_layer_ranges=[{"start": 0, "end": TOTAL_LAYERS}],
        rpc_arg="10.0.0.1:50052",
        rpc_device_weights=[[80]],
    )
    defaulted = MeshRpcPlan(
        mesh_id="m",
        rpc_endpoints=["10.0.0.1:50052"],
        rpc_split_weights=[80],
        rpc_layer_ranges=[{"start": 0, "end": TOTAL_LAYERS}],
        rpc_arg="10.0.0.1:50052",
    )
    assert explicit.plan_hash_hex() == defaulted.plan_hash_hex()
    assert defaulted.tensor_split_arg == "80"
    assert defaulted.total_devices == 1


def test_mismatched_device_weight_sum_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to the member split weight"):
        MeshRpcPlan(
            mesh_id="m",
            rpc_endpoints=["10.0.0.1:50052"],
            rpc_split_weights=[160],
            rpc_layer_ranges=[{"start": 0, "end": TOTAL_LAYERS}],
            rpc_arg="10.0.0.1:50052",
            rpc_device_weights=[[80, 70]],
        )
