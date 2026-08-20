"""Serve-supervisor join wait state vs the strict all-RPC binding validation.

Regression for a cross-box launch failure: while a multi-box mesh is still
admitting members, the joined subset forms a valid SMALLER all-RPC plan (one
member covering the whole model), so validating the launch configuration
against it raises "llama device order does not match the committed all-RPC
plan: expected RPC0" even though nothing is wrong. The supervisor must decide
the min-worker wait state from ``committed_rpc_stage_count`` BEFORE running
``validate_all_rpc_runtime_binding``; validating first turned every join tick
into that error and became a terminal drive failure when the last member
committed between the raise and the monitor's assignment-completeness
re-read.
"""

from __future__ import annotations

import pytest

from verallm.mesh.cli import (
    committed_rpc_stage_count,
    validate_all_rpc_runtime_binding,
)
from verallm.mesh.llama_cpp import (
    llama_tensor_split_layer_ranges,
    rpc_plan_from_mesh,
)
from verallm.mesh.types import MeshMember, MeshSpec, StageRange

TOTAL_LAYERS = 32


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
        mesh_id="mesh-join-wait-test",
        mode="private",
        coordinator_uid=0,
        coordinator_hotkey="coordinator-hotkey",
        model_id="test/model",
        model_package_hash="ab" * 32,
        total_layers=TOTAL_LAYERS,
        members=[coordinator, *members],
    )


def _worker(stage_index: int, layers: StageRange, *, rpc_port: int) -> MeshMember:
    return MeshMember(
        uid=stage_index,
        hotkey=f"worker-{stage_index}",
        endpoint=f"http://worker-{stage_index}:9443",
        stage_index=stage_index,
        layers=layers,
        rpc_endpoint=f"10.0.0.{stage_index}:{rpc_port}",
        rpc_split_weight=1,
    )


def _mid_join_spec() -> MeshSpec:
    # Only the first worker has joined; it covers the whole model, which is a
    # VALID one-stage all-RPC plan on its own.
    return _spec([_worker(1, StageRange(0, TOTAL_LAYERS), rpc_port=50052)])


def _complete_spec() -> MeshSpec:
    # Layer ranges must reproduce llama.cpp's weight-derived placement or
    # rpc_plan_from_mesh rejects the mesh outright.
    ranges = llama_tensor_split_layer_ranges(TOTAL_LAYERS, [1, 1])
    return _spec(
        [
            _worker(1, ranges[0], rpc_port=50052),
            _worker(2, ranges[1], rpc_port=50053),
        ]
    )


def test_mid_join_state_is_below_min_workers() -> None:
    assert committed_rpc_stage_count(_mid_join_spec()) == 1
    assert committed_rpc_stage_count(_complete_spec()) == 2


def test_mid_join_validation_raises_device_order_without_the_wait() -> None:
    # Documents WHY the wait check must run first: the strict validation on
    # the mid-join spec sees a committed one-stage plan and rejects the
    # two-device launch configuration.
    spec = _mid_join_spec()
    plan = rpc_plan_from_mesh(spec)
    with pytest.raises(ValueError, match="expected RPC0$"):
        validate_all_rpc_runtime_binding(
            spec,
            plan,
            device="RPC0,RPC1",
            n_gpu_layers="all",
        )


def test_complete_assignment_passes_strict_validation() -> None:
    spec = _complete_spec()
    plan = rpc_plan_from_mesh(spec)
    validate_all_rpc_runtime_binding(
        spec,
        plan,
        device="RPC0,RPC1",
        n_gpu_layers="all",
    )


def test_complete_assignment_still_rejects_wrong_device_order() -> None:
    # The fix must not weaken the guard once the mesh is fully admitted.
    spec = _complete_spec()
    plan = rpc_plan_from_mesh(spec)
    with pytest.raises(ValueError, match="expected RPC0,RPC1"):
        validate_all_rpc_runtime_binding(
            spec,
            plan,
            device="CUDA0",
            n_gpu_layers="all",
        )
