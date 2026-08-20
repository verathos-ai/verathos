"""OpenAI wire adapters for the private pool API.

The pool manager exposes ``GET /v1/models`` and
``POST /v1/chat/completions`` for operator-issued keys
(verallm/mesh/apikeys.py). This module is the pure translation layer:
OpenAI request -> pool chat body, pool chat result / SSE events ->
OpenAI response / chunks, plus the engine-agnostic prompt-engineered
tool-calling decision pass ported from the validator proxy (which does
not depend on a native tool parser in the serving engine).

Proof semantics are unchanged: every private-API chat runs the same
organic light-tier verification as subnet traffic (greedy or the
committed sampled profile), and the proof facts ride in a ``verathos``
extension object on the response.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Mapping

# Post-logits transforms are rejected at the API edge with a clean 400:
# they modify logits after the proved LM-head GEMM, so no verified lane
# (greedy or sampled) can honor them.
_UNSUPPORTED_REQUEST_CONTROLS = ("logit_bias", "grammar", "json_schema")

_SAMPLER_PASSTHROUGH = ("temperature", "top_k", "top_p", "min_p", "seed")


def openai_models_payload(status: Mapping[str, Any]) -> dict[str, Any]:
    """The pool's models as an OpenAI list (quant-selectable by id).

    Every pool model id names a specific quant (e.g. ``glm-5.2-iq2-m``),
    so "pick a quant" IS "pick a model". ``verathos.serving`` says
    whether a mesh currently answers for it.
    """

    serving_models = {
        str(mesh.get("model_id", "") or "")
        for mesh in (status.get("meshes") or {}).values()
        if str(mesh.get("status", "")) == "serving"
        and mesh.get("routing_ready", True)
    }
    # The registry only knows models with a download source; a model a
    # worker holds locally (catalog entry) or that is serving right now
    # is just as real to an API client.
    models: dict[str, Mapping[str, Any]] = {
        str(model_id): entry if isinstance(entry, Mapping) else {}
        for model_id, entry in (status.get("models") or {}).items()
    }
    for worker in (status.get("workers") or {}).values():
        for item in worker.get("catalog") or []:
            model_id = str(item.get("model_id", "") or "")
            if model_id and model_id not in models:
                models[model_id] = (
                    item if isinstance(item, Mapping) else {}
                )
    for model_id in serving_models:
        models.setdefault(model_id, {})
    data = []
    for model_id, entry in sorted(models.items()):
        data.append(
            {
                "id": str(model_id),
                "object": "model",
                "created": int(entry.get("created_at_unix", 0) or 0),
                "owned_by": "pool",
                "verathos": {
                    "serving": str(model_id) in serving_models,
                    "quantization_scheme": str(
                        entry.get("quantization_scheme", "") or ""
                    ),
                    "max_context_len": int(
                        entry.get("max_context_len", 0) or 0
                    ),
                    "model_bytes": int(entry.get("model_bytes", 0) or 0),
                },
            }
        )
    return {"object": "list", "data": data}


def resolve_mesh_for_model(
    status: Mapping[str, Any], model: str
) -> tuple[str, str]:
    """Map an OpenAI ``model`` to a serving mesh: (mesh_key, model_id).

    ``auto`` (or empty) picks the single serving mesh when unambiguous.
    Raises ValueError with the available model ids otherwise.
    """

    requested = str(model or "").strip()
    serving: list[tuple[str, str]] = []
    for mesh_key, mesh in sorted((status.get("meshes") or {}).items()):
        if str(mesh.get("status", "")) != "serving":
            continue
        if not mesh.get("routing_ready", True):
            continue
        serving.append((str(mesh_key), str(mesh.get("model_id", "") or "")))
    if not serving:
        raise ValueError("no mesh is serving in this pool right now")
    if requested in ("", "auto"):
        if len({model_id for _key, model_id in serving}) == 1:
            return serving[0]
        raise ValueError(
            "several models are serving; pick one: "
            + ", ".join(sorted({m for _k, m in serving}))
        )
    for mesh_key, model_id in serving:
        if model_id == requested:
            return mesh_key, model_id
    raise ValueError(
        f"model {requested!r} is not serving; available: "
        + ", ".join(sorted({m for _k, m in serving}))
    )


def pool_chat_body_from_openai(
    request: Mapping[str, Any], *, mesh_key: str
) -> dict[str, Any]:
    """OpenAI chat request -> the manager's pool chat body.

    Raises ValueError for shapes the verified lane cannot honor, so the
    route can answer a clean OpenAI-style 400 instead of a mid-stream
    coordinator error.
    """

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for name in _UNSUPPORTED_REQUEST_CONTROLS:
        if request.get(name) not in (None, "", {}, []):
            raise ValueError(
                f"{name} is not supported: it transforms logits after the "
                "proved LM-head computation"
            )
    response_format = request.get("response_format")
    if response_format not in (None, {}, {"type": "text"}):
        raise ValueError("only text response_format is supported")
    if request.get("n") not in (None, 1):
        raise ValueError("n must be 1")
    raw_max = request.get(
        "max_tokens", request.get("max_completion_tokens")
    )
    max_tokens = (
        int(raw_max) if isinstance(raw_max, int) and raw_max > 0 else 1024
    )
    body: dict[str, Any] = {
        "mesh_key": mesh_key,
        "messages": [dict(m) for m in messages if isinstance(m, Mapping)],
        "max_tokens": max_tokens,
        "stream": bool(request.get("stream", False)),
        # EXPLICIT light tier: on a chain-bound mesh the manager-pinned
        # operator lane upgrades anything else to the hard relation
        # (seconds of latency, and it refuses sampled profiles). The
        # private API is user traffic: organic light, proofs always on,
        # exactly like subnet serving traffic. Hard stays reachable via
        # `mesh probe`.
        "proof_tier": "light",
    }
    sampler = {
        name: request[name]
        for name in _SAMPLER_PASSTHROUGH
        if isinstance(request.get(name), (int, float))
        and not isinstance(request.get(name), bool)
    }
    if sampler:
        body["sampler"] = sampler
    return body


def _verathos_extension(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "verified",
            "receipt_verified",
            "proof_mode",
            "proof_receipt_root",
            "verification_snapshot_hash",
            "engine_tps",
            "prompt_tps",
            "ttft_s",
            "total_s",
            "proof_wall_s",
        )
        if result.get(key) is not None
    }


def openai_response_from_result(
    result: Mapping[str, Any],
    *,
    model: str,
    request_id: str,
    message: Mapping[str, Any] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Pool chat result -> OpenAI chat completion body.

    ``message`` overrides the assistant message (tool decision pass);
    default is the result content. ``usage`` passes through from the
    coordinator's real OpenAI body when present.
    """

    usage = result.get("usage")
    if message is not None:
        response_message: dict[str, Any] = dict(message)
    else:
        response_message = {
            "role": "assistant",
            "content": str(result.get("content", "") or ""),
        }
        # Reasoning models: thinking text is real generated output and
        # rides the OpenAI-standard message.reasoning_content field.
        reasoning = result.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            response_message["reasoning_content"] = reasoning
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }
        ],
        **({"usage": dict(usage)} if isinstance(usage, Mapping) else {}),
        "verathos": _verathos_extension(result),
    }


def openai_chunk(
    *,
    request_id: str,
    model: str,
    delta: Mapping[str, Any] | None = None,
    finish_reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One OpenAI SSE chunk body (mirrors the validator proxy's shape)."""

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": dict(delta or {}),
                "finish_reason": finish_reason,
            }
        ],
        **(dict(extra) if extra else {}),
    }


# ---------------------------------------------------------------------------
# Tool calling: prompt-engineered decision pass (no native parser needed)
# ---------------------------------------------------------------------------


def tool_choice_mode(tool_choice: Any) -> str:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice.lower()
    if isinstance(tool_choice, Mapping):
        return "forced"
    return "auto"


def _forced_tool_name(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, Mapping):
        return None
    fn = tool_choice.get("function") or {}
    if isinstance(fn, Mapping) and fn.get("name"):
        return str(fn["name"])
    return None


def _allowed_tool_names(tools: list[Any]) -> set[str]:
    names = set()
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            continue
        fn = tool.get("function") or {}
        if isinstance(fn, Mapping) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def build_tool_decision_messages(
    messages: list[dict],
    tools: list[dict],
    tool_choice: Any,
    parallel_tool_calls: bool | None,
) -> list[dict]:
    """Prompt the model into emitting an OpenAI-shaped tool decision.

    Ported from the validator proxy: it is still a model decision (the
    model chooses ``content`` or ``tool_calls``); the route normalizes
    the reply to the OpenAI wire format.
    """

    mode = tool_choice_mode(tool_choice)
    forced_name = _forced_tool_name(tool_choice)
    max_calls = 1 if parallel_tool_calls is False else 4
    instruction = {
        "role": "system",
        "content": (
            "You are deciding whether to call tools for an OpenAI-"
            "compatible chat completion. Return only one JSON object and "
            "no prose.\n\n"
            "If no tool is needed, return:\n"
            '{"content":"your answer"}\n\n'
            "If a tool is needed, return:\n"
            '{"tool_calls":[{"name":"tool_name","arguments":{...}}]}\n\n'
            f"tool_choice: {json.dumps(tool_choice if tool_choice is not None else 'auto')}\n"
            f"forced_tool_name: {json.dumps(forced_name)}\n"
            f"max_tool_calls: {max_calls}\n"
            "Use tools for current, external, source-backed, or "
            "unavailable knowledge. Do not invent tool results. Do not "
            "execute tools.\n"
            f"Available tools:\n{json.dumps(tools, ensure_ascii=False)}"
        ),
    }
    if mode in {"required", "forced"}:
        instruction["content"] += "\nA tool call is required for this request."
    return [instruction, *messages]


def _extract_first_json_object(text: str) -> dict | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _coerce_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps({"input": arguments}, ensure_ascii=False)
    if isinstance(arguments, Mapping):
        return json.dumps(dict(arguments), ensure_ascii=False)
    return json.dumps({}, ensure_ascii=False)


def normalize_tool_decision(
    text: str,
    tools: list[dict],
    tool_choice: Any,
    parallel_tool_calls: bool | None,
) -> tuple[dict, str]:
    """Model reply text -> (assistant message, finish_reason)."""

    data = _extract_first_json_object(text)
    allowed = _allowed_tool_names(tools)
    max_calls = 1 if parallel_tool_calls is False else 4
    tool_calls: list[dict] = []
    raw_calls: list[dict] = []
    if data and isinstance(data.get("tool_calls"), list):
        raw_calls.extend(
            call for call in data["tool_calls"] if isinstance(call, Mapping)
        )
    elif data and (
        data.get("name") or isinstance(data.get("function"), Mapping)
    ):
        raw_calls.append(data)
    seen: set[tuple[str, str]] = set()
    for raw_call in raw_calls:
        if len(tool_calls) >= max_calls:
            break
        fn = (
            raw_call.get("function")
            if isinstance(raw_call.get("function"), Mapping)
            else {}
        )
        name = raw_call.get("name") or fn.get("name")
        if not name or str(name) not in allowed:
            continue
        arguments_json = _coerce_tool_arguments(
            raw_call.get("arguments", fn.get("arguments", {}))
        )
        call_key = (str(name), arguments_json)
        if call_key in seen:
            continue
        seen.add(call_key)
        tool_calls.append(
            {
                "id": raw_call.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": arguments_json,
                },
            }
        )
    mode = tool_choice_mode(tool_choice)
    forced = _forced_tool_name(tool_choice)
    if not tool_calls and mode in {"required", "forced"}:
        name = forced or next(iter(sorted(allowed)), "")
        if name:
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            )
    if tool_calls:
        return (
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            "tool_calls",
        )
    content = (
        data["content"]
        if data and isinstance(data.get("content"), str)
        else text
    )
    return {"role": "assistant", "content": content}, "stop"
