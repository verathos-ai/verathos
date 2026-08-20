"""Private pool OpenAI API: operator keys front the pool's own meshes.

Distinct from the validator-hosted subnet API: these keys serve ONE
pool's meshes to the operator's own tools, independent of subnet
registration, over the pool manager's existing queue/driver plumbing
with proofs always on (organic light tier).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.verallm.test_mesh_pool import (
    _call,
    _command_report,
    _join,
    pool,  # noqa: F401 (fixture)
)
from verallm.mesh import apikeys
from verallm.mesh.openai_api import (
    normalize_tool_decision,
    openai_models_payload,
    pool_chat_body_from_openai,
    resolve_mesh_for_model,
)


def _http(endpoint, path, *, method="GET", body=None, key=""):
    request = urllib.request.Request(
        endpoint + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _serving_mesh(endpoint, token):
    _join(endpoint, token, "solo")
    launched = _call(
        endpoint, token, "/v1/pool/launch",
        {"model_id": "m1", "workers": ["solo"], "driver": "solo"},
    )
    mesh_key = launched["mesh_key"]
    drive = _call(
        endpoint, token, "/v1/pool/heartbeat", {"worker_id": "solo"}
    )["command"]
    _call(
        endpoint, token, "/v1/pool/report",
        _command_report(
            drive,
            worker_id="solo",
            event="drive_ready",
            mesh_id="mesh-x",
            join_token="vtmesh_tok",
            coordinator_endpoint="http://127.0.0.1:9500",
        ),
    )
    _call(endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 0})
    return mesh_key


def _mint_key(endpoint, token, name="test"):
    payload = _http(
        endpoint,
        "/v1/pool/api-keys",
        method="POST",
        body={
            "management_secret": token.management.pool_secret,
            "action": "create",
            "name": name,
        },
    )[1]
    return payload["api_key"]


def test_api_key_lifecycle_hash_at_rest(pool):
    endpoint, token = pool
    key = _mint_key(endpoint, token, name="laptop")
    assert key.startswith(apikeys.POOL_API_KEY_PREFIX)
    listing = _http(
        endpoint,
        "/v1/pool/api-keys",
        method="POST",
        body={
            "management_secret": token.management.pool_secret,
            "action": "list",
        },
    )[1]
    rows = listing["keys"]
    assert rows and rows[0]["name"] == "laptop"
    # The cleartext key never appears in the listing (hash-at-rest).
    assert key not in json.dumps(listing)
    revoked = _http(
        endpoint,
        "/v1/pool/api-keys",
        method="POST",
        body={
            "management_secret": token.management.pool_secret,
            "action": "revoke",
            "key_id": rows[0]["id"],
        },
    )[1]
    assert revoked["revoked"] is True
    # A revoked key stops authenticating.
    status, _payload = _http(endpoint, "/v1/models", key=key)
    assert status == 401


def test_models_requires_key_and_lists_pool_models(pool):
    endpoint, token = pool
    _serving_mesh(endpoint, token)
    status, payload = _http(endpoint, "/v1/models")
    assert status == 401
    assert "API key" in payload["error"]["message"]
    key = _mint_key(endpoint, token)
    status, payload = _http(endpoint, "/v1/models", key=key)
    assert status == 200
    assert payload["object"] == "list"
    ids = [row["id"] for row in payload["data"]]
    assert "m1" in ids
    row = payload["data"][ids.index("m1")]
    assert row["verathos"]["serving"] is True


def test_chat_completion_end_to_end_with_sampler_passthrough(pool):
    """OpenAI request -> pool queue -> driver long-poll -> OpenAI response.

    The model resolves to the serving mesh, sampler params ride the chat
    to the driver (committed light profile downstream), usage passes
    through from the driver's result, and proof facts land in the
    verathos extension."""

    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)

    result: dict = {}
    chatter = threading.Thread(
        target=lambda: result.update(
            dict(
                zip(
                    ("status", "payload"),
                    _http(
                        endpoint,
                        "/v1/chat/completions",
                        method="POST",
                        key=key,
                        body={
                            "model": "m1",
                            "messages": [
                                {"role": "user", "content": "ping"}
                            ],
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 16,
                        },
                    ),
                )
            )
        ),
        daemon=True,
    )
    chatter.start()
    polled = _call(
        endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 5}
    )
    assert len(polled["chat"]) == 1
    chat = polled["chat"][0]
    assert chat["messages"][0]["content"] == "ping"
    assert chat["sampler"] == {"temperature": 0.7, "top_p": 0.9}
    # Explicit light: the chain-bound operator lane would otherwise
    # upgrade to hard, which refuses sampled profiles.
    assert chat["proof_tier"] == "light"
    _call(
        endpoint,
        token,
        "/v1/pool/chat-pickup",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "delivery_token": chat["delivery_token"],
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-result",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "content": "pong",
            "verified": True,
            "receipts": 1,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        },
    )
    chatter.join(timeout=10)
    assert result["status"] == 200
    payload = result["payload"]
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "m1"
    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "pong"}
    assert choice["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 4
    assert payload["verathos"]["verified"] is True


def test_chat_completion_passes_reasoning_content_through(pool):
    """A thinking-heavy result: the driver's final carries the reasoning
    text and the OpenAI response surfaces it as message.reasoning_content,
    even when the visible content is empty."""

    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)

    result: dict = {}
    chatter = threading.Thread(
        target=lambda: result.update(
            dict(
                zip(
                    ("status", "payload"),
                    _http(
                        endpoint,
                        "/v1/chat/completions",
                        method="POST",
                        key=key,
                        body={
                            "model": "m1",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 16,
                        },
                    ),
                )
            )
        ),
        daemon=True,
    )
    chatter.start()
    polled = _call(
        endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 5}
    )
    chat = polled["chat"][0]
    _call(
        endpoint,
        token,
        "/v1/pool/chat-pickup",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "delivery_token": chat["delivery_token"],
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-result",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "content": "",
            "reasoning_content": "all sixteen tokens went to thinking",
            "verified": True,
            "receipts": 1,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 16,
                "total_tokens": 19,
            },
        },
    )
    chatter.join(timeout=10)
    assert result["status"] == 200
    message = result["payload"]["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "all sixteen tokens went to thinking",
    }


def test_chat_completion_stream_relays_reasoning_chunks(pool):
    """SSE: thinking chunks ride delta.reasoning_content, content rides
    delta.content, and the terminal chunk still closes the stream."""

    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)

    events: list = []
    errors: list = []

    def read_sse():
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "m1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "stream": True,
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as resp:
                assert resp.headers["Content-Type"].startswith(
                    "text/event-stream"
                )
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    events.append(data)
                    if data == "[DONE]":
                        break
        except Exception as exc:  # surfaced by the main thread's asserts
            errors.append(exc)

    reader = threading.Thread(target=read_sse, daemon=True)
    reader.start()
    polled = _call(
        endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 5}
    )
    chat = polled["chat"][0]
    _call(
        endpoint,
        token,
        "/v1/pool/chat-pickup",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "delivery_token": chat["delivery_token"],
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-chunk",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "thinking": "mulling it over",
            "seq": 1,
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-chunk",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "delta": "pong",
            "seq": 2,
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-result",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "content": "pong",
            "reasoning_content": "mulling it over",
            "verified": True,
            "receipts": 1,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            },
            "seq": 3,
        },
    )
    reader.join(timeout=15)
    assert not errors
    assert events[-1] == "[DONE]"
    chunks = [json.loads(data) for data in events[:-1]]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert {"role": "assistant", "reasoning_content": "mulling it over"} in deltas
    assert {"content": "pong"} in deltas
    # Ordering: reasoning precedes the first visible content token.
    reasoning_index = next(
        index
        for index, delta in enumerate(deltas)
        if "reasoning_content" in delta
    )
    content_index = next(
        index for index, delta in enumerate(deltas) if "content" in delta
    )
    assert reasoning_index < content_index
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_rejects_unsupported_and_unauthed(pool):
    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)
    status, _ = _http(
        endpoint,
        "/v1/chat/completions",
        method="POST",
        body={"model": "m1", "messages": [{"role": "user", "content": "x"}]},
    )
    assert status == 401
    status, payload = _http(
        endpoint,
        "/v1/chat/completions",
        method="POST",
        key=key,
        body={
            "model": "m1",
            "messages": [{"role": "user", "content": "x"}],
            "logit_bias": {"5": 10},
        },
    )
    assert status == 400
    assert "logit_bias" in payload["error"]["message"]
    status, payload = _http(
        endpoint,
        "/v1/chat/completions",
        method="POST",
        key=key,
        body={
            "model": "nope",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert status == 400
    assert "not serving" in payload["error"]["message"]
    # Streaming with tools stays a clean 400, like the validator proxy.
    status, payload = _http(
        endpoint,
        "/v1/chat/completions",
        method="POST",
        key=key,
        body={
            "model": "m1",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {}},
                }
            ],
        },
    )
    assert status == 400
    assert "tools" in payload["error"]["message"]


def test_tool_decision_pass_end_to_end(pool):
    """Tools ride the prompt-engineered decision pass (no native parser):
    the driver sees the tool instruction prompt; its JSON reply
    normalizes into OpenAI tool_calls with finish_reason tool_calls."""

    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    result: dict = {}
    chatter = threading.Thread(
        target=lambda: result.update(
            dict(
                zip(
                    ("status", "payload"),
                    _http(
                        endpoint,
                        "/v1/chat/completions",
                        method="POST",
                        key=key,
                        body={
                            "model": "m1",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "weather in Berlin?",
                                }
                            ],
                            "tools": tools,
                        },
                    ),
                )
            )
        ),
        daemon=True,
    )
    chatter.start()
    polled = _call(
        endpoint, token, "/v1/pool/chat-poll", {"worker_id": "solo", "wait": 5}
    )
    chat = polled["chat"][0]
    # The decision instruction leads the delivered conversation.
    assert chat["messages"][0]["role"] == "system"
    assert "tool_calls" in chat["messages"][0]["content"]
    assert chat["messages"][-1]["content"] == "weather in Berlin?"
    _call(
        endpoint,
        token,
        "/v1/pool/chat-pickup",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "delivery_token": chat["delivery_token"],
        },
    )
    _call(
        endpoint, token, "/v1/pool/chat-result",
        {
            "worker_id": "solo",
            "chat_id": chat["chat_id"],
            "content": (
                '{"tool_calls":[{"name":"get_weather",'
                '"arguments":{"city":"Berlin"}}]}'
            ),
            "verified": True,
            "receipts": 1,
        },
    )
    chatter.join(timeout=10)
    assert result["status"] == 200
    choice = result["payload"]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Berlin"}


def test_adapters_pure_shapes():
    status = {
        "models": {"m1": {"model_bytes": 5, "quantization_scheme": "Q4_K_M"}},
        "meshes": {
            "mk": {
                "model_id": "m1",
                "status": "serving",
                "routing_ready": True,
            }
        },
    }
    payload = openai_models_payload(status)
    assert payload["data"][0]["id"] == "m1"
    assert payload["data"][0]["verathos"]["serving"] is True
    assert resolve_mesh_for_model(status, "auto") == ("mk", "m1")
    with pytest.raises(ValueError, match="not serving"):
        resolve_mesh_for_model(status, "m2")
    body = pool_chat_body_from_openai(
        {
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.5,
            "seed": 7,
            "max_tokens": 8,
        },
        mesh_key="mk",
    )
    assert body["mesh_key"] == "mk"
    assert body["sampler"] == {"temperature": 0.5, "seed": 7}
    assert body["max_tokens"] == 8
    message, finish = normalize_tool_decision(
        "no tools needed, plain text", [], None, None
    )
    assert finish == "stop" and message["content"]


def test_rate_limits_protect_the_exposed_surface(pool, monkeypatch):
    """Internet exposure rules: per-key request limit and per-IP failed
    auth backoff, both fixed one-minute windows with OpenAI-style 429s."""
    import verallm.mesh.pool as pool_module

    endpoint, token = pool
    _serving_mesh(endpoint, token)
    key = _mint_key(endpoint, token)
    monkeypatch.setattr(pool_module, "POOL_API_RATE_LIMIT_PER_MIN", 3)
    monkeypatch.setattr(pool_module, "POOL_API_AUTH_FAILURES_PER_MIN", 2)
    statuses = [_http(endpoint, "/v1/models", key=key)[0] for _ in range(5)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3:] == [429, 429]
    status, payload = _http(endpoint, "/v1/models", key=key)
    assert status == 429 and "rate limit" in payload["error"]["message"]
    # Failed auth backs off per source address after the allowance.
    bad = [
        _http(endpoint, "/v1/models", key="vrt_pk_wrong")[0]
        for _ in range(4)
    ]
    assert bad[:2] == [401, 401]
    assert bad[2:] == [429, 429]


def test_api_request_allowed_window_rolls_over(tmp_path, monkeypatch):
    from tests.verallm.test_mesh_pool import _subnet_manager

    manager = _subnet_manager(tmp_path)
    now = {"value": 1_000_000.0}
    import verallm.mesh.pool as pool_module

    monkeypatch.setattr(pool_module.time, "time", lambda: now["value"])
    assert manager.api_request_allowed("key:x", limit=2)
    assert manager.api_request_allowed("key:x", limit=2)
    assert not manager.api_request_allowed("key:x", limit=2)
    now["value"] += 60
    assert manager.api_request_allowed("key:x", limit=2)


def test_api_tls_listener_serves_the_same_surface(tmp_path):
    """--api-tls-port opens a second https listener with the pool's
    auto-minted self-signed cert: same handler, same key auth, workers
    keep the plain-http port. This is the CLI-operable internet exposure
    (no nginx, no sudo, container-friendly)."""
    import shutil
    import ssl as ssl_module
    import threading as threading_module

    if shutil.which("openssl") is None:
        pytest.skip("openssl binary unavailable")

    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        create_pool_state,
        load_pool_token_file,
        serve_pool_manager,
    )

    state_dir, _token = create_pool_state(
        tmp_path, manager_endpoint="http://127.0.0.1:0", serving_mode="dev"
    )
    admin = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    server = serve_pool_manager(
        state_dir, host="127.0.0.1", port=0, api_tls_port=0
    )
    assert getattr(server, "api_tls_server") is None
    server.server_close()
    import socket as socket_module

    with socket_module.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        free_port = probe_socket.getsockname()[1]
    server = serve_pool_manager(
        state_dir, host="127.0.0.1", port=0, api_tls_port=free_port
    )
    api_server = getattr(server, "api_tls_server")
    assert api_server is not None
    # Cert minted once into the pool dir, owner-only key.
    assert (state_dir / "api-tls-cert.pem").is_file()
    assert (state_dir / "api-tls-key.pem").stat().st_mode & 0o077 == 0
    threads = [
        threading_module.Thread(target=server.serve_forever, daemon=True),
        threading_module.Thread(target=api_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        http_port = server.server_address[1]
        tls_port = api_server.server_address[1]
        # State records the CONFIGURED port for the board/CLI URL hints.
        state = json.loads((state_dir / "pool-state.json").read_text())
        assert state["api_tls_port"] == free_port
        # Key minted over plain http (loopback management), used over TLS.
        status, payload = _http(
            f"http://127.0.0.1:{http_port}",
            "/v1/pool/api-keys",
            method="POST",
            body={
                "management_secret": admin.pool_secret,
                "action": "create",
                "name": "tls",
            },
        )
        assert status == 200
        key = payload["api_key"]
        context = ssl_module._create_unverified_context()
        request = urllib.request.Request(
            f"https://127.0.0.1:{tls_port}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            method="GET",
        )
        with urllib.request.urlopen(
            request, timeout=10.0, context=context
        ) as response:
            assert response.status == 200
            body = json.loads(response.read())
            assert body["object"] == "list"
        # No key over TLS: the same 401 posture.
        bare = urllib.request.Request(
            f"https://127.0.0.1:{tls_port}/v1/models", method="GET"
        )
        try:
            urllib.request.urlopen(bare, timeout=10.0, context=context)
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
    finally:
        server.shutdown()
        api_server.shutdown()
        server.server_close()
        api_server.server_close()
