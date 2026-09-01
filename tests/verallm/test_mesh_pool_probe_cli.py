"""verathos mesh pool probe: operator self-test of a serving mesh."""

from __future__ import annotations

import argparse
import json

import pytest

import verallm.mesh.cli as mesh_cli
import verallm.mesh.worker as mesh_worker


class _FakeToken:
    manager_endpoint = "http://manager.test:9500"
    pool_secret = "secret"
    scope = "management"


def _args(**overrides) -> argparse.Namespace:
    base = {
        "mesh_key": "",
        "prompt": "test prompt",
        "max_tokens": 8,
        "timeout": 30.0,
        "gate": False,
        "samples": 1,
        "hard_samples": 0,
        "min_tok_s": None,
        "no_full_context": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch(monkeypatch, responses):
    calls: list[tuple[str, dict]] = []

    def fake_post_json(url, body, timeout=10.0):
        calls.append((url, body))
        for suffix, payload in responses:
            if url.endswith(suffix):
                return payload
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(mesh_worker, "post_json", fake_post_json)
    return calls


def test_probe_rejects_worker_scoped_token(monkeypatch) -> None:
    token = _FakeToken()
    token.scope = "worker"
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: token
    )
    with pytest.raises(SystemExit, match="management token"):
        mesh_cli.cmd_pool_probe(_args())


STATUS_ONE_SERVING = {
    "status": "ok",
    "meshes": {
        "m-aaa": {"status": "serving", "members": ["w0", "w1"]},
        "m-old": {"status": "stopped", "members": ["w0"]},
    },
}

CHAT_VERIFIED = {
    "status": "ok",
    "content": "A Merkle tree ...",
    "verified": True,
    "receipt_verified": True,
    "receipts": 2,
    "proof_stages": 2,
    "expected_stage_count": 2,
    "ttft_s": 0.1,
    "total_s": 2.0,
    "engine_tps": 100.0,
    "usage": {"completion_tokens": 8, "prompt_tokens": 4},
    "error": "",
}


def test_probe_auto_selects_single_serving_mesh(monkeypatch, capsys) -> None:
    calls = _patch(
        monkeypatch,
        [("/v1/pool/status", STATUS_ONE_SERVING), ("/v1/pool/chat", CHAT_VERIFIED)],
    )
    mesh_cli.cmd_pool_probe(_args())
    out = json.loads(capsys.readouterr().out)
    assert out["mesh_key"] == "m-aaa"
    assert out["verified"] is True
    assert out["receipts"] == 2
    chat_body = calls[-1][1]
    assert chat_body["probe"] is True
    assert chat_body["stream"] is True
    assert chat_body["mesh_key"] == "m-aaa"


def test_probe_requires_mesh_key_when_ambiguous(monkeypatch) -> None:
    status = {
        "status": "ok",
        "meshes": {
            "m-aaa": {"status": "serving"},
            "m-bbb": {"status": "serving"},
        },
    }
    _patch(monkeypatch, [("/v1/pool/status", status)])
    with pytest.raises(SystemExit, match="m-aaa, m-bbb"):
        mesh_cli.cmd_pool_probe(_args())


def test_probe_exits_nonzero_on_unverified(monkeypatch, capsys) -> None:
    failed = dict(CHAT_VERIFIED, verified=False, receipts=0, proof_stages=0,
                  error="proof endpoint failed")
    _patch(
        monkeypatch,
        [("/v1/pool/status", STATUS_ONE_SERVING), ("/v1/pool/chat", failed)],
    )
    with pytest.raises(SystemExit) as excinfo:
        mesh_cli.cmd_pool_probe(_args())
    assert excinfo.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["verified"] is False
    assert "proof endpoint failed" in out["error"]


def test_probe_gate_mode_runs_gate_and_exits_on_fail(monkeypatch, capsys) -> None:
    slow = dict(CHAT_VERIFIED, total_s=60.0, engine_tps=1.0,
                usage={"completion_tokens": 8, "prompt_tokens": 4})
    _patch(
        monkeypatch,
        [("/v1/pool/status", STATUS_ONE_SERVING), ("/v1/pool/chat", slow)],
    )
    with pytest.raises(SystemExit) as excinfo:
        mesh_cli.cmd_pool_probe(_args(gate=True))
    assert excinfo.value.code == 1
    assert "NOT READY" in capsys.readouterr().out


def test_probe_gate_prints_ready_verdict_with_measured_context(
    monkeypatch, capsys
) -> None:
    """--gate is the operator's ready-to-register confirmation: light and
    hard proofs, tok/s, TTFT, and the measured context in one verdict."""
    status = {
        "status": "ok",
        "meshes": {
            "m-aaa": {
                "status": "serving",
                "members": ["w0", "w1"],
                "model_id": "glm-5.2-iq2-m",
            },
        },
        "models": {
            "glm-5.2-iq2-m": {"measured_ctx_budget": 389_120},
        },
    }
    fast = dict(
        CHAT_VERIFIED,
        total_s=2.0,
        ttft_s=0.4,
        usage={"completion_tokens": 120, "prompt_tokens": 30},
    )
    _patch(
        monkeypatch,
        [("/v1/pool/status", status), ("/v1/pool/chat", fast)],
    )
    with pytest.raises(SystemExit) as excinfo:
        mesh_cli.cmd_pool_probe(_args(gate=True, hard_samples=1))
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "READY TO REGISTER" in out
    assert "hard 1/1 verified" in out
    assert "measured context 389120" in out


def test_probe_parser_wires_subcommand() -> None:
    parser = mesh_cli.build_parser()
    args = parser.parse_args(
        ["pool", "probe", "--pool-token-file", "/tmp/x", "--mesh-key", "m-abc"]
    )
    assert args.func is mesh_cli.cmd_pool_probe
    assert args.mesh_key == "m-abc"
