"""
Verathos custom LLM provider for LiteLLM.

Verathos (https://verathos.ai) provides verified LLM inference on the
Bittensor network.  Every response can be cryptographically proven to come
from the declared model -- no output substitution is possible.

The API is fully OpenAI-compatible, so this provider is a thin routing
layer that:
  1. Registers "verathos" as a LiteLLM custom provider.
  2. Strips the ``verathos/`` prefix and forwards to the Verathos API via
     the ``openai`` Python SDK.
  3. Supports ``verathos/auto`` (Verathos picks the best available model)
     and ``verathos/<model-name>`` (user picks explicitly).
"""

from __future__ import annotations

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Union,
)

import httpx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERATHOS_API_BASE = "https://api.verathos.ai/v1"
VERATHOS_ENV_API_KEY = "VERATHOS_API_KEY"
VERATHOS_ENV_API_BASE = "VERATHOS_API_BASE"
PROVIDER_NAME = "verathos"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Return an API key from the argument, env-var, or raise."""
    key = api_key or os.environ.get(VERATHOS_ENV_API_KEY)
    if not key:
        raise CustomLLMError(
            status_code=401,
            message=(
                "Verathos API key not provided. Pass api_key= or set "
                f"the {VERATHOS_ENV_API_KEY} environment variable."
            ),
        )
    return key


def _resolve_api_base(api_base: Optional[str] = None) -> str:
    """Return the Verathos API base URL."""
    return api_base or os.environ.get(VERATHOS_ENV_API_BASE, VERATHOS_API_BASE)


def _strip_provider_prefix(model: str) -> str:
    """``verathos/auto`` -> ``auto``, ``verathos/Qwen/Qwen3-30B`` -> ``Qwen/Qwen3-30B``."""
    prefix = f"{PROVIDER_NAME}/"
    if model.startswith(prefix):
        return model[len(prefix):]
    return model


def _build_openai_params(
    messages: list,
    model: str,
    optional_params: dict,
    stream: bool = False,
) -> dict:
    """Build the kwargs dict for ``client.chat.completions.create()``."""
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    # Forward all OpenAI-compatible params the caller passed.
    _forwarded = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "n",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "user",
        "logit_bias",
    }
    for key in _forwarded:
        if key in optional_params:
            params[key] = optional_params[key]
    # Also forward stream_options if streaming.
    if stream and "stream_options" in optional_params:
        params["stream_options"] = optional_params["stream_options"]
    return params


def _openai_response_to_model_response(
    openai_resp: Any,
    model_response: ModelResponse,
) -> ModelResponse:
    """Copy fields from an ``openai`` SDK response into a LiteLLM ModelResponse."""
    resp_dict = openai_resp.model_dump()
    model_response.id = resp_dict.get("id", model_response.id)
    model_response.created = resp_dict.get("created", model_response.created)
    model_response.model = resp_dict.get("model", model_response.model)
    model_response.system_fingerprint = resp_dict.get("system_fingerprint")
    if resp_dict.get("choices"):
        model_response.choices = []  # type: ignore[assignment]
        for choice_data in resp_dict["choices"]:
            from litellm.types.utils import Choices, Message

            msg_data = choice_data.get("message", {})
            message = Message(
                content=msg_data.get("content"),
                role=msg_data.get("role", "assistant"),
                tool_calls=msg_data.get("tool_calls"),
                function_call=msg_data.get("function_call"),
            )
            choice = Choices(
                finish_reason=choice_data.get("finish_reason", "stop"),
                index=choice_data.get("index", 0),
                message=message,
            )
            model_response.choices.append(choice)  # type: ignore[union-attr]
    if resp_dict.get("usage"):
        from litellm.types.utils import Usage

        u = resp_dict["usage"]
        model_response.usage = Usage(  # type: ignore[assignment]
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
    return model_response


def _iter_streaming_chunks(
    openai_stream: Any,
) -> Iterator[GenericStreamingChunk]:
    """Yield ``GenericStreamingChunk`` dicts from an ``openai`` streaming response."""
    for chunk in openai_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason
        text = delta.content or ""
        yield GenericStreamingChunk(
            text=text,
            is_finished=finish_reason is not None,
            finish_reason=finish_reason or "",
            usage=None,
        )


async def _aiter_streaming_chunks(
    openai_stream: Any,
) -> AsyncIterator[GenericStreamingChunk]:
    """Yield ``GenericStreamingChunk`` dicts from an async ``openai`` streaming response."""
    async for chunk in openai_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason
        text = delta.content or ""
        yield GenericStreamingChunk(
            text=text,
            is_finished=finish_reason is not None,
            finish_reason=finish_reason or "",
            usage=None,
        )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class VerathosProvider(CustomLLM):
    """LiteLLM custom provider for Verathos verified LLM inference.

    Register once, then use ``model="verathos/auto"`` (or any specific model
    name) with ``litellm.completion()`` / ``litellm.acompletion()``.
    """

    # ------------------------------------------------------------------
    # Registration helper
    # ------------------------------------------------------------------

    @staticmethod
    def register() -> None:
        """Register the Verathos provider with LiteLLM.

        Safe to call multiple times -- subsequent calls are no-ops.
        """
        if PROVIDER_NAME in litellm._custom_providers:
            return
        litellm.custom_provider_map.append(
            {"provider": PROVIDER_NAME, "custom_handler": VerathosProvider()}
        )
        # Also update the internal tracking list and provider_list.
        if PROVIDER_NAME not in litellm._custom_providers:
            litellm._custom_providers.append(PROVIDER_NAME)
        if PROVIDER_NAME not in litellm.provider_list:
            litellm.provider_list.append(PROVIDER_NAME)

    # ------------------------------------------------------------------
    # Sync completion
    # ------------------------------------------------------------------

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        base_url = _resolve_api_base(api_base if api_base else None)
        key = _resolve_api_key(api_key)
        verathos_model = _strip_provider_prefix(model)

        timeout_val = (
            timeout if isinstance(timeout, (int, float)) else 600.0
        )

        openai_client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_val,
        )

        stream = optional_params.pop("stream", False)

        params = _build_openai_params(
            messages=messages,
            model=verathos_model,
            optional_params=optional_params,
            stream=stream,
        )

        # Logging: record the request.
        logging_obj.pre_call(
            input=messages,
            api_key=key,
            additional_args={
                "complete_input_dict": params,
                "api_base": base_url,
                "headers": {"Authorization": f"Bearer {key[:8]}..."},
            },
        )

        if stream:
            response = openai_client.chat.completions.create(**params)
            return _iter_streaming_chunks(response)  # type: ignore[return-value]

        response = openai_client.chat.completions.create(**params)

        result = _openai_response_to_model_response(response, model_response)

        # Logging: record the response.
        logging_obj.post_call(
            input=messages,
            api_key=key,
            original_response=response,
            additional_args={"complete_input_dict": params},
        )

        return result

    # ------------------------------------------------------------------
    # Sync streaming
    # ------------------------------------------------------------------

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        base_url = _resolve_api_base(api_base if api_base else None)
        key = _resolve_api_key(api_key)
        verathos_model = _strip_provider_prefix(model)

        timeout_val = (
            timeout if isinstance(timeout, (int, float)) else 600.0
        )

        openai_client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_val,
        )

        params = _build_openai_params(
            messages=messages,
            model=verathos_model,
            optional_params=optional_params,
            stream=True,
        )

        logging_obj.pre_call(
            input=messages,
            api_key=key,
            additional_args={
                "complete_input_dict": params,
                "api_base": base_url,
            },
        )

        response = openai_client.chat.completions.create(**params)
        return _iter_streaming_chunks(response)

    # ------------------------------------------------------------------
    # Async completion
    # ------------------------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        base_url = _resolve_api_base(api_base if api_base else None)
        key = _resolve_api_key(api_key)
        verathos_model = _strip_provider_prefix(model)

        timeout_val = (
            timeout if isinstance(timeout, (int, float)) else 600.0
        )

        openai_client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_val,
        )

        stream = optional_params.pop("stream", False)

        params = _build_openai_params(
            messages=messages,
            model=verathos_model,
            optional_params=optional_params,
            stream=stream,
        )

        logging_obj.pre_call(
            input=messages,
            api_key=key,
            additional_args={
                "complete_input_dict": params,
                "api_base": base_url,
            },
        )

        if stream:
            response = await openai_client.chat.completions.create(**params)
            return _aiter_streaming_chunks(response)  # type: ignore[return-value]

        response = await openai_client.chat.completions.create(**params)

        result = _openai_response_to_model_response(response, model_response)

        logging_obj.post_call(
            input=messages,
            api_key=key,
            original_response=response,
            additional_args={"complete_input_dict": params},
        )

        return result

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        base_url = _resolve_api_base(api_base if api_base else None)
        key = _resolve_api_key(api_key)
        verathos_model = _strip_provider_prefix(model)

        timeout_val = (
            timeout if isinstance(timeout, (int, float)) else 600.0
        )

        openai_client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_val,
        )

        params = _build_openai_params(
            messages=messages,
            model=verathos_model,
            optional_params=optional_params,
            stream=True,
        )

        logging_obj.pre_call(
            input=messages,
            api_key=key,
            additional_args={
                "complete_input_dict": params,
                "api_base": base_url,
            },
        )

        response = await openai_client.chat.completions.create(**params)
        async for chunk in _aiter_streaming_chunks(response):
            yield chunk
