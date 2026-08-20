"""verathos mesh chat: interactive streaming REPL, and the guided launch."""

from __future__ import annotations

import argparse
import io
import json

import pytest

import verallm.mesh.cli as mesh_cli
import verallm.mesh.worker as mesh_worker


class _FakeToken:
    manager_endpoint = "http://manager.test:9500"
    pool_secret = "secret"
    scope = "management"


STATUS = {
    "status": "ok",
    "meshes": {
        "m-serving": {
            "status": "serving",
            "model_id": "qwen2.5-7b-q4-k-m",
            "members": ["w0", "w1"],
            "routing_ready": True,
        },
    },
    "workers": {
        "w0": {
            "worker_id": "w0",
            "catalog": [
                {
                    "model_id": "qwen2.5-7b-q4-k-m",
                    "layers": 28,
                    "model_bytes": 4_683_073_632,
                }
            ],
        },
        "w1": {"worker_id": "w1", "catalog": []},
    },
}


def _sse(events) -> io.BytesIO:
    payload = b""
    for event in events:
        payload += b"data: " + json.dumps(event).encode() + b"\n\n"
    return io.BytesIO(payload)


class _FakeStream(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_select_serving_mesh_auto_and_errors() -> None:
    key, mesh = mesh_cli._select_serving_mesh(
        STATUS["meshes"], "", interactive=False
    )
    assert key == "m-serving"
    assert mesh["model_id"] == "qwen2.5-7b-q4-k-m"
    with pytest.raises(SystemExit, match="unknown mesh"):
        mesh_cli._select_serving_mesh(STATUS["meshes"], "m-nope", interactive=False)
    with pytest.raises(SystemExit, match="no mesh is serving"):
        mesh_cli._select_serving_mesh({}, "", interactive=False)


def test_chat_streams_deltas_and_verifies(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(
        mesh_worker, "post_json", lambda url, body, timeout=10.0: STATUS
    )
    posted: list[dict] = []

    def fake_urlopen(request, timeout=0):
        posted.append(json.loads(request.data.decode()))
        return _FakeStream(
            _sse(
                [
                    {"type": "phase", "phase": "prefill"},
                    {"type": "delta", "delta": "Hello "},
                    {"type": "delta", "delta": "world."},
                    {
                        "type": "done",
                        "verified": True,
                        "receipts": 2,
                        "proof_stages": 2,
                        "expected_stage_count": 2,
                        "ttft_s": 0.1,
                        "engine_tps": 100.0,
                        "content": "Hello world.",
                        "error": "",
                    },
                ]
            ).getvalue()
        )

    monkeypatch.setattr(mesh_cli, "urlopen", fake_urlopen)
    prompts = iter(["hi there", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(prompts))

    args = argparse.Namespace(
        mesh_key="", max_tokens=64, timeout=30.0, thinking=False
    )
    mesh_cli.cmd_mesh_chat(args)
    out = capsys.readouterr().out
    assert "Hello world." in out
    assert "verified (2/2 stages, 2 receipts)" in out
    body = posted[0]
    assert body["mesh_key"] == "m-serving"
    assert body["messages"][-1] == {"role": "user", "content": "hi there"}
    assert body["thinking"] is False


def test_chat_verified_reply_joins_history(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(
        mesh_worker, "post_json", lambda url, body, timeout=10.0: STATUS
    )
    posted: list[dict] = []

    def fake_urlopen(request, timeout=0):
        posted.append(json.loads(request.data.decode()))
        return _FakeStream(
            _sse(
                [
                    {"type": "delta", "delta": "Paris."},
                    {"type": "done", "verified": True, "error": ""},
                ]
            ).getvalue()
        )

    monkeypatch.setattr(mesh_cli, "urlopen", fake_urlopen)
    prompts = iter(["capital of France?", "are you sure?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(prompts))

    args = argparse.Namespace(
        mesh_key="", max_tokens=64, timeout=30.0, thinking=False
    )
    mesh_cli.cmd_mesh_chat(args)
    # The second request must carry the verified assistant reply, or the
    # model answers every follow-up without seeing what it already said.
    assert [(m["role"], m["content"]) for m in posted[1]["messages"]] == [
        ("user", "capital of France?"),
        ("assistant", "Paris."),
        ("user", "are you sure?"),
    ]


def test_chat_unverified_reply_is_flagged_and_dropped(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(
        mesh_worker, "post_json", lambda url, body, timeout=10.0: STATUS
    )
    posted: list[dict] = []

    def fake_urlopen(request, timeout=0):
        posted.append(json.loads(request.data.decode()))
        return _FakeStream(
            _sse(
                [
                    {"type": "delta", "delta": "bad"},
                    {
                        "type": "done",
                        "verified": False,
                        "error": "proof endpoint failed",
                    },
                ]
            ).getvalue()
        )

    monkeypatch.setattr(mesh_cli, "urlopen", fake_urlopen)
    prompts = iter(["first", "second", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(prompts))

    args = argparse.Namespace(
        mesh_key="", max_tokens=64, timeout=30.0, thinking=False
    )
    mesh_cli.cmd_mesh_chat(args)
    out = capsys.readouterr().out
    assert "unverified" in out
    # The failed exchange must not poison later history: the second request
    # carries only its own user message.
    assert [m["content"] for m in posted[1]["messages"]] == ["second"]


def test_launch_non_interactive_still_requires_model_id(monkeypatch) -> None:
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(mesh_cli, "_interactive_terminal", lambda: False)
    monkeypatch.setattr(
        mesh_worker, "post_json", lambda url, body, timeout=10.0: STATUS
    )
    args = argparse.Namespace(
        model_id="", workers="", driver="", timeout=10.0
    )
    with pytest.raises(SystemExit, match="--model-id is required"):
        mesh_cli.cmd_pool_launch(args)


def test_launch_non_interactive_passes_through(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(mesh_cli, "_interactive_terminal", lambda: False)
    calls: list[tuple[str, dict]] = []

    def fake_post_json(url, body, timeout=10.0):
        calls.append((url, body))
        return {"status": "ok", "mesh_key": "m-new"}

    monkeypatch.setattr(mesh_worker, "post_json", fake_post_json)
    args = argparse.Namespace(
        model_id="qwen2.5-7b-q4-k-m",
        workers="w0,w1",
        driver="w0",
        timeout=10.0,
        no_wait=True,
    )
    mesh_cli.cmd_pool_launch(args)
    out = json.loads(capsys.readouterr().out)
    assert out["mesh_key"] == "m-new"
    url, body = calls[-1]
    assert url.endswith("/v1/pool/launch")
    assert body["workers"] == ["w0", "w1"]
    assert body["driver"] == "w0"


def test_launch_non_interactive_waits_for_routing_ready(
    monkeypatch, capsys
) -> None:
    # Scripted launches share the guided flow's contract: when the command
    # returns, the mesh routes (no more launch-then-chat 400 races).
    monkeypatch.setattr(
        mesh_cli, "_resolve_pool_context", lambda *a, **k: _FakeToken()
    )
    monkeypatch.setattr(mesh_cli, "_interactive_terminal", lambda: False)
    monkeypatch.setattr(mesh_cli.time, "sleep", lambda s: None)
    status_calls = {"n": 0}

    def fake_post_json(url, body, timeout=10.0):
        if url.endswith("/v1/pool/launch"):
            return {"status": "launching", "mesh_key": "m-new"}
        status_calls["n"] += 1
        if status_calls["n"] < 3:
            return {"meshes": {"m-new": {"status": "launching"}}}
        return {
            "meshes": {
                "m-new": {"status": "serving", "routing_ready": True}
            }
        }

    monkeypatch.setattr(mesh_worker, "post_json", fake_post_json)
    args = argparse.Namespace(
        model_id="qwen2.5-7b-q4-k-m",
        workers="w0",
        driver="w0",
        timeout=60.0,
        no_wait=False,
    )
    mesh_cli.cmd_pool_launch(args)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "serving"
    assert out["routing_ready"] is True
    assert status_calls["n"] >= 3


def test_chat_parser_wires_subcommand() -> None:
    parser = mesh_cli.build_parser()
    args = parser.parse_args(["chat", "--mesh-key", "m-abc"])
    assert args.func is mesh_cli.cmd_mesh_chat
    assert args.mesh_key == "m-abc"
    # A usable reply budget by default: 1024 truncated ordinary answers.
    assert args.max_tokens == 4096
    launch_args = parser.parse_args(["pool", "launch"])
    assert launch_args.func is mesh_cli.cmd_pool_launch
    assert launch_args.model_id == ""
