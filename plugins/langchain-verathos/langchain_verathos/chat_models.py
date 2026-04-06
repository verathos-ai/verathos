"""Verathos chat model for LangChain.

Extends ``ChatOpenAI`` from ``langchain-openai`` with proof-verification
metadata, model discovery, and automatic model selection.

Every inference response from Verathos includes a cryptographic proof that the
model was executed honestly (ZK sumcheck + Merkle commitments).  This module
surfaces that proof metadata on the LangChain ``AIMessage`` so downstream
chains can inspect and act on verification results.

Example::

    from langchain_verathos import ChatVerathos

    llm = ChatVerathos(model="auto")                     # automatic best-model
    msg = llm.invoke("Explain zero-knowledge proofs.")
    print(msg.content)
    print(msg.response_metadata["proof_verified"])        # True / False
    print(msg.response_metadata["proof_details"])         # layer / beacon details
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, model_validator

logger = logging.getLogger(__name__)

_VERATHOS_BASE_URL = "https://api.verathos.ai/v1"

# Keys that Verathos adds to the standard OpenAI response payload.
_PROOF_KEYS = ("proof_verified", "proof_details", "timing")


class ChatVerathos(ChatOpenAI):
    r"""Chat model for `Verathos <https://verathos.ai>`_ — verified LLM
    inference on `Bittensor <https://bittensor.com>`_.

    Verathos is an OpenAI-compatible API where every response is backed by a
    cryptographic proof (ZK sumcheck + Merkle commitments) that the declared
    model was executed faithfully — no output substitution, no bait-and-switch.

    ``ChatVerathos`` extends ``ChatOpenAI`` so all standard LangChain chat-model
    features (streaming, tool calling, structured output, async, batching) work
    out of the box.  The key value-add:

    * Proof-verification metadata on every ``AIMessage`` via
      ``response_metadata``  (``proof_verified``, ``proof_details``, ``timing``)
    * ``model="auto"`` for automatic best-model selection
    * Dedicated ``list_models()`` helper for model discovery
    * Sensible defaults (``base_url``, ``api_key`` env var)

    Setup:
        Install the package and set your API key::

            pip install langchain-verathos

            export VERATHOS_API_KEY="your-api-key"

        Or pass the key directly::

            llm = ChatVerathos(api_key="vrt_sk_...")

    Instantiate:
        .. code-block:: python

            from langchain_verathos import ChatVerathos

            # Automatic model selection (recommended)
            llm = ChatVerathos(model="auto")

            # Specific model
            llm = ChatVerathos(model="Qwen/Qwen3-30B-A3B")

            # All standard ChatOpenAI params work
            llm = ChatVerathos(
                model="auto",
                temperature=0.7,
                max_tokens=1024,
                streaming=True,
            )

    Invoke:
        .. code-block:: python

            msg = llm.invoke("Explain zero-knowledge proofs in one paragraph.")
            print(msg.content)

            # Proof metadata is available on every response
            print(msg.response_metadata["proof_verified"])   # True
            print(msg.response_metadata["timing"])           # inference/proof timing

    Stream:
        .. code-block:: python

            full = None
            for chunk in llm.stream("Write a haiku about cryptography."):
                print(chunk.text, end="", flush=True)
                full = chunk if full is None else full + chunk

            # Proof metadata is on the final accumulated message
            print(full.response_metadata.get("proof_verified"))

    Model discovery:
        .. code-block:: python

            models = ChatVerathos.list_models()
            for m in models:
                print(f"{m['id']:40s}  owned_by={m.get('owned_by', '?')}")

    Args:
        model: Model name or ``"auto"`` for automatic selection.
        api_key: Verathos API key.  Falls back to ``VERATHOS_API_KEY`` env var.
        base_url: Override the API base URL (default: ``https://api.verathos.ai/v1``).
        include_proof: Request detailed proof metadata in responses.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        streaming: Whether to stream responses.
    """

    # -- Field overrides -------------------------------------------------------

    model_name: str = Field(default="auto", alias="model")
    """Model to use.  ``"auto"`` lets Verathos pick the best available model."""

    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("VERATHOS_API_KEY", default=None),
    )
    """Verathos API key.  Falls back to the ``VERATHOS_API_KEY`` environment variable."""

    openai_api_base: Optional[str] = Field(
        default=_VERATHOS_BASE_URL,
        alias="base_url",
    )
    """Verathos API base URL.  Defaults to ``https://api.verathos.ai/v1``."""

    include_proof: bool = Field(default=True)
    """Include detailed proof metadata (challenged layers, detection probability,
    beacon validity, etc.) in responses.  Disable to save bandwidth."""

    # Verathos models aren't in tiktoken — skip token counting to avoid warnings.
    tiktoken_model_name: Optional[str] = Field(default="gpt-4o")

    # Disable stream_usage by default — Verathos sends usage in the final chunk
    # regardless, and the OpenAI SDK's stream_options logic can interfere.
    stream_usage: Optional[bool] = None

    # -- Validators ------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_verathos(self) -> "ChatVerathos":
        """Set defaults that make sense for the Verathos API."""
        # Ensure base_url always has a value.
        if not self.openai_api_base:
            self.openai_api_base = _VERATHOS_BASE_URL
        return self

    # -- LangSmith / tracing ---------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "verathos-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "VERATHOS_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> dict[str, Any]:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "verathos"
        return params

    # -- Request payload -------------------------------------------------------

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Inject ``include_proof`` into the request via ``extra_body``."""
        # Merge include_proof into extra_body so the Verathos proxy sees it.
        if self.include_proof:
            extra = dict(self.extra_body) if self.extra_body else {}
            extra.setdefault("include_proof", True)
            kwargs["extra_body"] = extra
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    # -- Non-streaming: extract proof metadata ---------------------------------

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        """Override to extract Verathos proof fields from the API response."""
        # Let the parent do all the heavy lifting (message parsing, usage, etc.)
        result = super()._create_chat_result(response, generation_info)

        # Extract proof metadata from the raw response.
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )

        verathos_meta: dict[str, Any] = {}
        for key in _PROOF_KEYS:
            if key in response_dict:
                verathos_meta[key] = response_dict[key]

        if verathos_meta:
            # Attach to llm_output so it lands in response_metadata on the message.
            if result.llm_output is None:
                result.llm_output = {}
            result.llm_output.update(verathos_meta)
            result.llm_output["model_provider"] = "verathos"

            # Also attach directly to the AIMessage.additional_kwargs for
            # convenient programmatic access.
            for gen in result.generations:
                msg = gen.message
                for key in _PROOF_KEYS:
                    if key in verathos_meta:
                        msg.additional_kwargs[key] = verathos_meta[key]

        return result

    # -- Streaming: extract proof metadata from final chunk --------------------

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict] = None,
    ) -> Optional[ChatGenerationChunk]:
        """Override to capture proof metadata from Verathos streaming chunks.

        Verathos sends proof metadata in the final SSE chunk (the usage chunk
        with ``choices: []``).  We extract ``proof_verified``, ``timing``, and
        ``proof_details`` and attach them to the chunk's ``response_metadata``.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None

        # Check if this chunk carries Verathos proof fields.
        verathos_meta: dict[str, Any] = {}
        for key in _PROOF_KEYS:
            if key in chunk:
                verathos_meta[key] = chunk[key]

        if verathos_meta:
            msg = generation_chunk.message
            # Update response_metadata on the message chunk.
            if hasattr(msg, "response_metadata"):
                msg.response_metadata.update(verathos_meta)
            msg.response_metadata["model_provider"] = "verathos"

            # Also put in additional_kwargs for easy access after accumulation.
            if isinstance(msg, AIMessageChunk):
                for key, value in verathos_meta.items():
                    msg.additional_kwargs[key] = value

        else:
            # Always tag provider even on non-proof chunks.
            if hasattr(generation_chunk.message, "response_metadata"):
                generation_chunk.message.response_metadata["model_provider"] = "verathos"

        return generation_chunk

    # -- Model discovery -------------------------------------------------------

    @staticmethod
    def list_models(
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Fetch available models from the Verathos API.

        Returns the ``data`` array from ``GET /v1/models`` — each entry has at
        minimum ``id`` and ``object`` fields, plus Verathos-specific fields like
        ``pricing`` and ``context_length``.

        Args:
            api_key: Override API key (defaults to ``VERATHOS_API_KEY`` env var).
            base_url: Override base URL.
            timeout: HTTP timeout in seconds.

        Returns:
            List of model dicts.

        Example::

            models = ChatVerathos.list_models()
            for m in models:
                print(m["id"])
        """
        import os

        key = api_key or os.environ.get("VERATHOS_API_KEY", "")
        url = (base_url or _VERATHOS_BASE_URL).rstrip("/") + "/models"

        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        resp = httpx.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", [])

    @staticmethod
    def list_model_ids(
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[str]:
        """Return just the model ID strings from the Verathos API.

        Convenience wrapper around :meth:`list_models`.

        Example::

            ids = ChatVerathos.list_model_ids()
            # ['Qwen/Qwen3-30B-A3B', 'meta-llama/Llama-3.3-70B-Instruct', ...]
        """
        return [m["id"] for m in ChatVerathos.list_models(api_key=api_key, base_url=base_url)]
