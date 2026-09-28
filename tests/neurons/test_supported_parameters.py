"""Miner-advertised ``supported_parameters`` survive discovery and shared state.

Miners publish the OpenAI request fields they can serve on ``/health``
(``verallm/api/server.py``).  The validator is the only component that polls
``/health``, so if it drops the field the proxy has no way to learn that a
model can serve tool calls, and ``GET /v1/models`` advertises nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from neurons.discovery import ActiveMiner
from neurons.shared_state import (
    MinerEntry,
    ValidatorSharedState,
    read_shared_state,
    write_shared_state,
)
from neurons.validator import ValidatorNeuron


ADDRESS = "0x" + "ab" * 20
ENDPOINT = "https://miner.invalid:9082"
ADVERTISED = ["tools", "tool_choice", "parallel_tool_calls"]
MINER_KWARGS = {
    "address": ADDRESS,
    "endpoint": ENDPOINT,
    "model_id": "qwen3.8-27b",
    "model_index": 0,
    "quant": "fp8",
    "max_context_len": 262144,
}


def _health_response(monkeypatch, payload):
    """Stage a ``/health`` body for ``_fetch_miner_hardware_status``."""
    response = SimpleNamespace(
        status_code=200,
        json=lambda: payload,
    )
    monkeypatch.setattr(
        "neurons.validator.httpx.get",
        lambda *args, **kwargs: response,
    )


def _hardware():
    return {
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "vram_gb": 80,
        "compute_capability": "9.0",
        "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
    }


def test_health_supported_parameters_are_recorded(monkeypatch):
    miner = ActiveMiner(**MINER_KWARGS)
    _health_response(
        monkeypatch,
        {"hardware": _hardware(), "supported_parameters": ADVERTISED},
    )

    status = ValidatorNeuron._fetch_miner_hardware_status(
        SimpleNamespace(
            _ADVERTISABLE_SUPPORTED_PARAMETERS=(
                ValidatorNeuron._ADVERTISABLE_SUPPORTED_PARAMETERS
            )
        ),
        miner,
    )

    assert status == "fetched"
    assert miner.supported_parameters == ADVERTISED


def test_unknown_supported_parameters_are_discarded(monkeypatch):
    """A miner cannot inject arbitrary strings into the public listing."""
    miner = ActiveMiner(**MINER_KWARGS)
    _health_response(
        monkeypatch,
        {
            "hardware": _hardware(),
            "supported_parameters": [
                "tools",
                "logprobs",
                {"not": "a string"},
                "'; DROP TABLE miners; --",
            ],
        },
    )

    ValidatorNeuron._fetch_miner_hardware_status(
        SimpleNamespace(
            _ADVERTISABLE_SUPPORTED_PARAMETERS=(
                ValidatorNeuron._ADVERTISABLE_SUPPORTED_PARAMETERS
            )
        ),
        miner,
    )

    assert miner.supported_parameters == ["tools"]


def test_missing_or_malformed_supported_parameters_default_empty(monkeypatch):
    for payload in ({"hardware": _hardware()},
                    {"hardware": _hardware(), "supported_parameters": "tools"}):
        miner = ActiveMiner(**MINER_KWARGS)
        _health_response(monkeypatch, payload)

        ValidatorNeuron._fetch_miner_hardware_status(
            SimpleNamespace(
                _ADVERTISABLE_SUPPORTED_PARAMETERS=(
                    ValidatorNeuron._ADVERTISABLE_SUPPORTED_PARAMETERS
                )
            ),
            miner,
        )

        assert miner.supported_parameters == []


def test_shared_state_round_trip_preserves_supported_parameters(tmp_path):
    path = str(tmp_path / "shared_state.json")
    state = ValidatorSharedState(
        miner_endpoints=[
            MinerEntry(**MINER_KWARGS, supported_parameters=list(ADVERTISED))
        ],
    )

    write_shared_state(state, path=path)

    on_disk = json.loads(open(path).read())
    assert on_disk["miner_endpoints"][0]["supported_parameters"] == ADVERTISED

    restored = read_shared_state(path=path)
    assert restored is not None
    assert restored.miner_endpoints[0].supported_parameters == ADVERTISED

    # A state file written before this field existed still loads.
    on_disk["miner_endpoints"][0].pop("supported_parameters")
    with open(path, "w") as handle:
        json.dump(on_disk, handle)

    legacy = read_shared_state(path=path)
    assert legacy is not None
    assert legacy.miner_endpoints[0].supported_parameters == []
