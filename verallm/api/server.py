#!/usr/bin/env python3
"""
VeraLLM Miner Server — FastAPI server wrapping VllmMiner.

Runs a vLLM-powered miner that serves inference requests and generates
cryptographic proofs over a REST API.  The protocol is non-interactive
(Fiat-Shamir): a single POST /inference streams tokens and returns
commitment + proofs.  No separate proof request is needed — the miner
derives beacon and challenges internally from the commitment and the
validator nonce it received with the request.

Endpoints:
    GET  /health      — Server health check
    GET  /model_spec  — Model specification (weight Merkle roots)
    POST /inference   — SSE-streamed inference + commitment + proofs

Usage:
    python -m verallm.api.server --model Qwen/Qwen3-8B --port 8000
    python -m verallm.api.server --model allenai/OLMoE-1B-7B-0125-Instruct --quant int8
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import sys
import time
import uuid
import warnings
from typing import Optional

# ── Suppress noisy third-party output BEFORE any imports trigger it ──
# vLLM reconfigures logging on import; prevent that.
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
# PyTorch C++ glog noise (FakeTensor, TorchDynamo, inductor)
os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("TORCHDYNAMO_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
# Suppress tqdm progress bars from vLLM/safetensors/transformers
os.environ.setdefault("TQDM_DISABLE", "1")
# Suppress transformers internal warnings (trust_remote_code, rope_parameters)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Suppress ALL Python warnings from third-party libs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# FlashInfer TRTLLM MoE FP4 backend produces NaN/garbage activations on
# Blackwell (tested up to FlashInfer 0.6.3).  Force the CUTLASS backend
# before any vLLM module is imported so the env var is visible when vLLM
# reads it during model loading.
# See: https://github.com/flashinfer-ai/flashinfer/issues/1049
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")

# Cap CPU worker threads used by OpenMP/BLAS kernels inside proof workers.
# This prevents N proof threads × M CPU-library threads from exhausting
# process thread limits at high concurrency.
#
# Override with VERALLM_CPU_THREADS_PER_WORKER=<n> if needed.
try:
    _CPU_THREADS_PER_WORKER = max(1, int(os.environ.get("VERALLM_CPU_THREADS_PER_WORKER", "1")))
except ValueError:
    _CPU_THREADS_PER_WORKER = 1
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = str(_CPU_THREADS_PER_WORKER)

# Enable expandable segments in the CUDA memory allocator to reduce
# fragmentation.  Without this, models that leave only ~1.5 GiB free GPU
# memory after loading fail to allocate temporary workspace for weight
# extraction (AWQ int4 unpacking, int8 quantization) during prewarm.
# expandable_segments lets the allocator reuse freed blocks more efficiently.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bittensor as bt
import torch

try:
    torch.set_num_threads(_CPU_THREADS_PER_WORKER)
    torch.set_num_interop_threads(_CPU_THREADS_PER_WORKER)
except Exception:
    # Best-effort hardening: if a backend disallows runtime thread changes,
    # keep startup running with env-level caps above.
    pass

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from verallm.config import Config, set_config
from verallm.challenge.beacon import derive_beacon_from_nonce, derive_challenges, derive_sampling_challenge, derive_embedding_challenge
from verallm.sampling import (
    clamp_sampling_bps, temperature_to_milli, build_hidden_row_merkle,
    build_logits_row_merkle, HIGH_ASSURANCE_BPS,
)
from verallm.crypto.merkle import MerkleTree
from verallm.helpers import compute_auto_k, compute_auto_k_experts
from verallm.miner import VllmMiner
from verallm.miner.batch_engine import BatchAwareEngine
from verallm.miner.activation_tracker import RequestActivationTracker
from verallm.miner.admission import TokenBudgetAdmission
from verallm.miner.memory_budget import (
    auto_detect_max_requests_with_ram,
    auto_detect_proof_concurrency,
    estimate_per_request_ram_mb,
)
from verallm.miner.proof_pipeline import ProofPipeline
from verallm.moe import is_moe_model, is_moe_layer, get_moe_config, derive_moe_challenges, MoEHookManager, BatchMoEHookManager
from verallm.quantization import detect_quantization
from verallm.registry import compute_model_roots, load_cached_model_spec, save_model_spec_to_cache
from verallm.types import InferenceCommitment, ModelSpec

from verallm.api.serialization import (
    model_spec_to_dict,
    commitment_to_dict,
    proof_bundle_to_dict,
)


# ============================================================================
# Pydantic models for request validation
# ============================================================================

class InferenceRequestBody(BaseModel):
    prompt: str
    validator_nonce: str  # hex-encoded 32 bytes
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    sampling_verification_bps: int = 0
    enable_thinking: bool = True  # chain-of-thought for models that support it
    # Sampling parameters (post-logits processors).  Server applies
    # model-specific defaults when these are None (e.g. Qwen3 presence_penalty).
    presence_penalty: Optional[float] = None   # vLLM default 0.0
    top_k: Optional[int] = None                # vLLM default -1 (disabled)
    top_p: Optional[float] = None              # vLLM default 1.0
    min_p: Optional[float] = None              # vLLM default 0.0


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequestBody(BaseModel):
    """Chat-style inference with OpenAI-compatible messages array.

    The miner applies the model's chat template server-side (it already
    has the tokenizer loaded), so clients don't need the tokenizer.
    """
    messages: list[ChatMessage]
    validator_nonce: str  # hex-encoded 32 bytes
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    sampling_verification_bps: int = 0
    enable_thinking: bool = True  # chain-of-thought for models that support it
    # Sampling parameters — see InferenceRequestBody for docs.
    presence_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None


# ============================================================================
# App state (populated during startup)
# ============================================================================

logger = logging.getLogger(__name__)


class MinerState:
    def __init__(self):
        self.miner: Optional[VllmMiner] = None
        self.model_spec: Optional[ModelSpec] = None
        self.config: Optional[Config] = None
        self.moe_config = None
        self.model_name: str = ""
        # Epoch receipt storage (SQLite-backed, survives restarts)
        from verallm.api.receipt_store import ReceiptStore
        self.receipt_store = ReceiptStore()
        # Batch mode state (None when batch mode is off)
        self.batch_mode: bool = False
        self.batch_engine: Optional[BatchAwareEngine] = None
        self.activation_tracker: Optional[RequestActivationTracker] = None
        self.moe_hook_mgr: Optional[BatchMoEHookManager] = None
        self.proof_pipeline: Optional[ProofPipeline] = None
        self.admission: Optional[TokenBudgetAdmission] = None
        self._step_loop_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._last_request_time: float = 0.0  # monotonic clock
        # EVM identity (populated via --evm-address / --evm-private-key from miner.py)
        self.evm_address: Optional[str] = None
        self.evm_private_key: Optional[str] = None  # hex, for identity challenge signing
        # TEE state (populated when --tee-enabled is set)
        self.tee_enabled: bool = False
        self.tee_platform: str = ""
        self.tee_skip_proofs: bool = False
        self.tee_private_key: Optional[bytes] = None
        self.tee_public_key: Optional[bytes] = None
        self.tee_attestation = None  # TEEAttestation instance
        # Hardware metadata (populated at startup from torch.cuda)
        self.gpu_name: str = ""
        self.gpu_count: int = 0
        self.vram_gb: int = 0
        self.compute_capability: str = ""


state = MinerState()
app = FastAPI(title="VeraLLM Miner", version="0.1.0")


@app.on_event("startup")
async def _on_startup():
    """Start the background engine step loop if batch mode is enabled."""
    if state.batch_mode and state.batch_engine is not None:
        state._step_loop_task = asyncio.create_task(_engine_step_loop())
        bt.logging.info("Started background engine step loop (batch mode)")

        # Batch-mode warmup: the synchronous LLM.generate() warmup in startup()
        # compiles Triton kernels for the sync code path, but batch mode uses
        # engine.step() via run_in_executor + the async step loop.  This can
        # trigger additional JIT compilation on the first real request (9-15s
        # penalty).  Sending a dummy request through the actual batch path
        # ensures all kernels are compiled before real traffic arrives.
        await _batch_warmup()

        # Start periodic keepalive to prevent CUDA graph cache eviction
        # during long idle periods.
        state._keepalive_task = asyncio.create_task(_keepalive_loop())
        bt.logging.info("Started GPU keepalive loop")


async def _batch_warmup():
    """Send a dummy request through the batch engine to compile remaining kernels."""
    from vllm import SamplingParams

    miner = state.miner
    batch_engine = state.batch_engine
    if miner is None or batch_engine is None:
        return

    bt.logging.info("Running batch-mode warmup...")
    t0 = time.perf_counter()

    try:
        tokenizer = miner.tokenizer
        template_kwargs = _chat_template_kwargs(tokenizer, enable_thinking=True)

        # Use enable_thinking=True template — this is the webapp's default
        # and may produce different prompt token counts than thinking=False,
        # triggering different FLA/Triton kernel specializations.
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Briefly explain quantum computing."}],
                tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:
            prompt = "warmup"

        loop = asyncio.get_event_loop()

        # Generate enough tokens to trigger ALL lazy kernel compilations.
        # vLLM V1 + torch.compile/Triton lazily compile kernels during the
        # first N decode steps (CUDA graph captures, attention kernel
        # specializations, KV cache page management paths).  8 tokens is
        # NOT enough — user's first real request still pays 9+ seconds of
        # compilation spikes spread across many steps.  256 tokens covers
        # all observed compilation events (empirically verified).
        params = SamplingParams(max_tokens=256, temperature=0)

        # Register with activation tracker like a real request — exercises
        # the full code path including hidden state capture hooks.
        tracker = state.activation_tracker
        req_id = "warmup-batch-0"
        session_id = "warmup-session"
        if tracker is not None:
            tracker.register_request(req_id, session_id, capture_logits=False)

        q = batch_engine.add_request(req_id, prompt, params)
        while True:
            output = await q.get()
            if output.finished:
                break
        batch_engine.clear_finished(req_id)

        # Clean up tracker state
        if tracker is not None:
            tracker.unregister_request(req_id)

        bt.logging.info(f"Batch warmup done ({(time.perf_counter() - t0) * 1000:.0f}ms)")
    except Exception as e:
        bt.logging.warning(f"Batch warmup failed: {e}")


# Default keepalive interval: 10 minutes.  CUDA graph caches survive hours of
# idle, but driver power-state transitions (P0→P2→P0) combined with memory
# pressure from other processes can cause eviction sooner.  10 min is
# conservative — a single 1-token inference is enough to keep graphs warm.
_KEEPALIVE_INTERVAL = int(os.environ.get("VERALLM_KEEPALIVE_INTERVAL", "600"))


async def _keepalive_loop():
    """Periodically send a tiny inference to keep CUDA graphs warm.

    After long idle periods (6+ hours observed), the first real request can
    take 9+ seconds due to CUDA graph cache re-warmup.  This loop sends a
    single 1-token generation through the batch engine every 10 minutes
    (configurable via VERALLM_KEEPALIVE_INTERVAL) when there's no real
    traffic, preventing the driver from evicting cached graph state.

    The keepalive is skipped when real requests are in-flight (no point —
    the graphs are already being exercised).
    """
    from vllm import SamplingParams

    batch_engine = state.batch_engine
    if batch_engine is None:
        return

    # Mark initial warmup as the first "request"
    state._last_request_time = time.monotonic()

    while True:
        try:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)

            # Skip if there are active requests — graphs are already warm
            if batch_engine.has_active_requests():
                continue

            # Skip if a real request ran recently
            idle_secs = time.monotonic() - state._last_request_time
            if idle_secs < _KEEPALIVE_INTERVAL * 0.9:
                continue

            t0 = time.perf_counter()
            params = SamplingParams(max_tokens=1, temperature=0)
            req_id = f"keepalive-{uuid.uuid4().hex[:8]}"
            q = batch_engine.add_request(req_id, "keepalive", params)
            while True:
                output = await q.get()
                if output.finished:
                    break
            batch_engine.clear_finished(req_id)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            bt.logging.debug(f"GPU keepalive done ({elapsed_ms:.0f}ms)")

        except asyncio.CancelledError:
            break
        except Exception:
            bt.logging.debug("GPU keepalive failed")
            await asyncio.sleep(60)  # back off on errors


@app.on_event("shutdown")
async def _on_shutdown():
    """Clean up batch mode resources."""
    if state._keepalive_task is not None:
        state._keepalive_task.cancel()
    if state._step_loop_task is not None:
        state._step_loop_task.cancel()
    if state.proof_pipeline is not None:
        state.proof_pipeline.shutdown(wait=False)
    # Shutdown batched proof matmul service.
    try:
        from verallm.miner.matmul import shutdown_proof_matmul_batcher
        shutdown_proof_matmul_batcher()
    except ImportError:
        pass
    if state.activation_tracker is not None:
        state.activation_tracker.remove_hooks()
    if state.moe_hook_mgr is not None:
        state.moe_hook_mgr.remove_hooks()
    # Clear the active tracker reference for the capture custom op
    try:
        from verallm.vllm_plugin.ops import set_active_tracker
        set_active_tracker(None)
    except ImportError:
        pass

from verallm.api.auth import APIKeyMiddleware  # noqa: E402
from verallm.api.validator_auth import ValidatorAuthMiddleware  # noqa: E402

# Validator auth: verifies Sr25519 signature against metagraph allowlist.
# Blocks non-public requests when no validators file exists (deny by default).
app.add_middleware(ValidatorAuthMiddleware)
# API key auth: optional secondary layer (VERATHOS_API_KEY env var).
app.add_middleware(APIKeyMiddleware)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    # Health must always respond quickly, even under heavy load.
    # No CUDA calls here — KV pool stats from admission control are
    # the real saturation metric; torch.cuda.memory_allocated() syncs
    # the GPU and can block the event loop for seconds under load.
    result = {
        "status": "ok",
        "model": state.model_name,
        "moe": state.moe_config is not None,
        "batch_mode": state.batch_mode,
        "capture_backend": (
            state.activation_tracker.backend
            if state.activation_tracker is not None
            else ("splitting_ops" if state.miner and getattr(state.miner, "_use_cuda_graphs", False) else "hooks")
        ),
        "max_model_len": state.miner.llm.llm_engine.model_config.max_model_len
        if state.miner and state.miner.llm else None,
    }
    if state.gpu_name:
        result["hardware"] = {
            "gpu_name": state.gpu_name,
            "gpu_count": state.gpu_count,
            "vram_gb": state.vram_gb,
            "compute_capability": state.compute_capability,
            "gpu_uuids": getattr(state, "gpu_uuids", []),
        }
    if state.batch_mode and state.admission is not None:
        s = state.admission.status()
        result["active_requests"] = s.active_requests
        result["max_requests"] = state.admission.max_requests
        result["kv_pool_tokens"] = s.total_kv_tokens
        result["kv_used_tokens"] = s.used_tokens
        result["kv_free_tokens"] = s.free_tokens
        result["kv_utilization_pct"] = round(s.used_tokens / s.total_kv_tokens * 100, 1) if s.total_kv_tokens > 0 else 0
        result["can_accept_max_context"] = s.can_accept_max_context
        result["max_context"] = s.max_context
        if state.proof_pipeline is not None:
            result["proof_pending"] = state.proof_pipeline.num_pending
            result["proof_max_pending"] = state.proof_pipeline.max_pending
    if state.tee_enabled:
        result["tee"] = {
            "enabled": True,
            "platform": state.tee_platform,
            "proof_mode": "attestation" if state.tee_skip_proofs else "verallm",
        }
    return result


class IdentityChallengeBody(BaseModel):
    nonce: str  # 64 hex chars (32 bytes)


@app.post("/identity/challenge")
async def identity_challenge(body: IdentityChallengeBody):
    """Prove this endpoint is controlled by its registered EVM address.

    Validators send a random nonce; the miner signs (nonce || evm_address)
    with its EVM private key.  The validator recovers the signer and compares
    against the on-chain registered address — rejecting hijacked endpoints.
    """
    if not state.evm_private_key or not state.evm_address:
        return JSONResponse(
            status_code=501,
            content={"error": "Identity challenge not available (no EVM key configured)"},
        )

    try:
        nonce_bytes = bytes.fromhex(body.nonce)
        if len(nonce_bytes) != 32:
            return JSONResponse(status_code=400, content={"error": "Nonce must be 32 bytes (64 hex chars)"})
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid hex nonce"})

    # Sign: nonce (32 bytes) || evm_address (20 bytes) = 52 bytes
    address_bytes = bytes.fromhex(state.evm_address[2:])  # strip 0x
    message = nonce_bytes + address_bytes

    from eth_account import Account
    from eth_account.messages import encode_defunct
    signable = encode_defunct(primitive=message)
    signed = Account.sign_message(signable, private_key=state.evm_private_key)

    return {
        "address": state.evm_address,
        "signature": signed.signature.hex(),
    }


@app.get("/model_spec")
async def get_model_spec():
    """Return the ModelSpec with weight Merkle roots.

    NOTE: In production, ModelSpec is published on-chain by a trusted
    registrant (subnet owner / DAO).  The validator reads roots from
    chain, not from the miner.  The miner computes its own roots and
    compares against on-chain roots as a self-diagnostic — catching
    wrong model versions, corrupt downloads, or quantization mismatches.
    This direct-from-miner serving is a development simplification;
    chain integration is a TODO.
    """
    if state.model_spec is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    return model_spec_to_dict(state.model_spec)


def _resolve_sampling_params(
    body,
    model_name: str,
) -> dict:
    """Resolve final vLLM SamplingParams from request body + model defaults.

    When the caller sends None for a param, we apply sensible defaults:
    - ``enable_thinking=True``  → ``presence_penalty=1.5`` (prevents infinite
      ``<think>`` loops in *any* thinking-capable model, not just Qwen3).
    - ``enable_thinking=False`` → ``presence_penalty=1.2`` (prevents
      degenerate repetition loops that plague many models, especially
      Qwen3.5, when no penalty is applied).
    When the caller sends an explicit value (including 0.0), we respect it
    unconditionally.  Validator canary tests send explicit ``0.0`` for strict
    argmax binding — that path is unaffected.

    Returns dict of kwargs for ``SamplingParams(...)``.
    """
    temperature = 0.0 if not body.do_sample else body.temperature
    enable_thinking = getattr(body, "enable_thinking", True)

    # Resolve presence_penalty: model-agnostic, tied to thinking mode.
    # Both modes now get a non-zero default to prevent repetition loops.
    # Canary tests send explicit 0.0 for strict argmax verification.
    pp = body.presence_penalty
    if pp is None:
        pp = 1.5 if enable_thinking else 1.2

    return {
        "max_tokens": body.max_new_tokens,
        "temperature": temperature,
        "presence_penalty": pp,
        "top_k": body.top_k if body.top_k is not None else -1,
        "top_p": body.top_p if body.top_p is not None else 1.0,
        "min_p": body.min_p if body.min_p is not None else 0.0,
    }


def _chat_template_kwargs(tokenizer, enable_thinking: bool = True) -> dict:
    """Extra kwargs for apply_chat_template based on model capabilities.

    Models like Qwen3/3.5 accept ``enable_thinking`` — when True the template
    injects ``<think>\\n`` into the prompt so the model reasons before
    answering.  The caller controls this per-request.
    """
    import inspect
    try:
        src = inspect.getsource(tokenizer.apply_chat_template)
    except (TypeError, OSError):
        src = ""
    # Also check the Jinja template string itself (HF fast tokenizers store it)
    tpl = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in src or "enable_thinking" in tpl:
        return {"enable_thinking": enable_thinking}
    # GPT-oss uses reasoning_effort instead of enable_thinking
    if "reasoning_effort" in tpl:
        return {"reasoning_effort": "none" if not enable_thinking else "medium"}
    return {}


def _apply_chat_template(tokenizer, raw_prompt: str, enable_thinking: bool = True):
    """Wrap a raw prompt in the model's chat template.

    Instruction-tuned models (Phi-4, Mistral, etc.) expect prompts formatted
    with their chat template.  Without it, some models emit EOS immediately.

    Returns:
        (formatted_prompt, prompt_token_ids): One will be set, the other None.
        - Mistral tokenizers return token IDs directly from apply_chat_template.
        - Other tokenizers return a formatted string.
        If the tokenizer has no chat template, returns (raw_prompt, None).
    """
    messages = [{"role": "user", "content": raw_prompt}]
    _is_mistral_tok = "MistralTokenizer" in type(tokenizer).__name__
    try:
        if _is_mistral_tok:
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return None, token_ids
        else:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **_chat_template_kwargs(tokenizer, enable_thinking),
            )
            return formatted, None
    except Exception:
        # Tokenizer has no chat template — pass through raw prompt.
        return raw_prompt, None


@app.post("/inference")
async def run_inference(body: InferenceRequestBody, request: Request = None):
    """Run inference, stream tokens, then return commitment + proofs.

    The full non-interactive protocol runs within this single request:
    1. Stream tokens via SSE as they are generated
    2. Build commitment (Merkle roots of activations)
    3. Derive beacon + challenges (Fiat-Shamir, from commitment + nonce)
    4. Generate proofs for challenged layers
    5. Return commitment + proof bundle in the final SSE event

    In batch mode, multiple requests run concurrently via vLLM's
    continuous batching.  A semaphore limits concurrency to prevent OOM.
    Returns 503 when all slots are occupied.
    """
    if state.miner is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    nonce = bytes.fromhex(body.validator_nonce)

    # Compute prompt_hash from the raw prompt (for /inference endpoint).
    # Stored as local var (not on state) to avoid race conditions between
    # concurrent requests — state._current_prompt_hash was shared/mutable.
    _prompt_hash = hashlib.sha256(body.prompt.encode()).digest()

    # Apply chat template — instruction-tuned models need proper formatting
    tokenizer = state.miner.tokenizer
    formatted_prompt, prompt_token_ids = _apply_chat_template(
        tokenizer, body.prompt, enable_thinking=body.enable_thinking)
    if formatted_prompt is not None:
        body = InferenceRequestBody(
            prompt=formatted_prompt,
            validator_nonce=body.validator_nonce,
            max_new_tokens=body.max_new_tokens,
            do_sample=body.do_sample,
            temperature=body.temperature,
            sampling_verification_bps=body.sampling_verification_bps,
        )

    if state.batch_mode:
        if state.proof_pipeline is not None:
            pending = state.proof_pipeline.num_pending
            max_pending = state.proof_pipeline.max_pending
            if pending >= max_pending:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Miner busy: proof queue full",
                        "proof_pending": pending,
                        "proof_max_pending": max_pending,
                        "retry_after_ms": 5000,
                    },
                    headers={"Retry-After": "5"},
                )

        # Batch mode: dynamic token-budget admission
        # Estimate prompt tokens from formatted prompt (includes chat template overhead)
        if prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        else:
            prompt_tokens = len(tokenizer.encode(body.prompt))
        token_budget = prompt_tokens + body.max_new_tokens

        # Admission check — reject with 503 before streaming starts
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        admitted = await state.admission.try_admit(request_id, token_budget)
        if not admitted:
            s = state.admission.status()
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Miner busy: KV cache full",
                    "free_tokens": s.free_tokens,
                    "requested_tokens": token_budget,
                    "active_requests": s.active_requests,
                    "retry_after_ms": 5000,
                },
                headers={"Retry-After": "5"},
            )

        _vhk = getattr(getattr(request, "state", None), "validator_hotkey", "") if request else ""
        return StreamingResponse(
            _stream_inference_batched(body, nonce, prompt_token_ids=prompt_token_ids,
                                     token_budget=token_budget,
                                     admitted_request_id=request_id,
                                     prompt_hash=_prompt_hash,
                                     validator_hotkey=_vhk),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-batch mode: original single-request path
    return StreamingResponse(
        _stream_inference(body, nonce, prompt_token_ids=prompt_token_ids,
                          prompt_hash=_prompt_hash),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def run_chat(body: ChatRequestBody, request: Request = None):
    """Chat-style inference: accepts messages array, applies chat template.

    Same protocol as /inference but accepts OpenAI-style messages instead of
    a raw prompt string. The miner applies the model's chat template using
    the loaded tokenizer, so clients don't need it.

    Used by the chain-discovery webapp and any client that wants chat-native
    inference without managing tokenizers.
    """
    if state.miner is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    # Extract validator hotkey for logging (set by ValidatorAuthMiddleware)
    _vali_hotkey = getattr(getattr(request, "state", None), "validator_hotkey", "") if request else ""

    nonce = bytes.fromhex(body.validator_nonce)

    # Apply chat template using the miner's tokenizer
    tokenizer = state.miner.tokenizer
    messages_dicts = [{"role": m.role, "content": m.content} for m in body.messages]
    # Compute prompt_hash from the canonical messages JSON for input integrity.
    # Stored as local var (not on state) to avoid race conditions between
    # concurrent requests — state._current_prompt_hash was shared/mutable.
    import json as _json
    _prompt_hash_input = _json.dumps(messages_dicts, sort_keys=True, ensure_ascii=False).encode()
    _prompt_hash = hashlib.sha256(_prompt_hash_input).digest()
    bt.logging.debug(
        f"prompt_hash: {_prompt_hash.hex()[:16]} (len={len(_prompt_hash_input)})"
    )

    _is_mistral_tok = "MistralTokenizer" in type(tokenizer).__name__
    _extra_kw = _chat_template_kwargs(tokenizer, body.enable_thinking)
    try:
        if _is_mistral_tok:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages_dicts, tokenize=True, add_generation_prompt=True
            )
            formatted_prompt = None
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                messages_dicts, tokenize=False, add_generation_prompt=True,
                **_extra_kw,
            )
            prompt_token_ids = None
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to apply chat template: {e}"},
        )

    # Create a synthetic InferenceRequestBody so _stream_inference works
    # For Mistral tokenizers, pass token_ids separately
    synth_body = InferenceRequestBody(
        prompt=formatted_prompt or "",
        validator_nonce=body.validator_nonce,
        max_new_tokens=body.max_new_tokens,
        do_sample=body.do_sample,
        temperature=body.temperature,
        sampling_verification_bps=body.sampling_verification_bps,
        enable_thinking=body.enable_thinking,
        presence_penalty=body.presence_penalty,
        top_k=body.top_k,
        top_p=body.top_p,
        min_p=body.min_p,
    )

    if state.batch_mode:
        if state.proof_pipeline is not None:
            pending = state.proof_pipeline.num_pending
            max_pending = state.proof_pipeline.max_pending
            if pending >= max_pending:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Miner busy: proof queue full",
                        "proof_pending": pending,
                        "proof_max_pending": max_pending,
                        "retry_after_ms": 5000,
                    },
                    headers={"Retry-After": "5"},
                )

        # Estimate token budget from pre-tokenized prompt
        if prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        else:
            prompt_tokens = len(tokenizer.encode(synth_body.prompt))
        token_budget = prompt_tokens + body.max_new_tokens

        # Admission check — reject with 503 before streaming starts
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        admitted = await state.admission.try_admit(request_id, token_budget)
        if not admitted:
            s = state.admission.status()
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Miner busy: KV cache full",
                    "free_tokens": s.free_tokens,
                    "requested_tokens": token_budget,
                    "active_requests": s.active_requests,
                    "retry_after_ms": 5000,
                },
                headers={"Retry-After": "5"},
            )

        return StreamingResponse(
            _stream_inference_batched(synth_body, nonce,
                                     prompt_token_ids=prompt_token_ids,
                                     token_budget=token_budget,
                                     admitted_request_id=request_id,
                                     prompt_hash=_prompt_hash,
                                     validator_hotkey=_vali_hotkey),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_inference(synth_body, nonce, prompt_token_ids=prompt_token_ids,
                          prompt_hash=_prompt_hash),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# TEE endpoints — confidential GPU mode (E2E encryption + attestation)
# ============================================================================


class TEEChatRequestBody(BaseModel):
    """Encrypted chat request for TEE mode."""
    envelope: dict  # {session_id, sender_public_key, nonce, ciphertext} (all hex)
    validator_nonce: str  # hex-encoded 32 bytes


@app.get("/tee/info")
async def tee_info():
    """Return enclave public key and attestation report.

    Clients use this to:
    1. Get the enclave's X25519 public key for E2E encryption
    2. Verify the attestation report (proves correct code + model in TEE)
    3. Cross-check the weight Merkle root against on-chain ModelRegistry
    """
    if not state.tee_enabled:
        return JSONResponse(status_code=404, content={"error": "TEE not enabled"})

    from verallm.tee.serialization import attestation_to_dict

    result = {
        "enclave_public_key": state.tee_public_key.hex(),
        "attestation": attestation_to_dict(state.tee_attestation),
        "proof_mode": "attestation" if state.tee_skip_proofs else "verallm",
        "model": state.model_name,
    }
    # Include weight Merkle root so validators can cross-check against chain
    if state.model_spec is not None:
        result["weight_merkle_root"] = state.model_spec.weight_merkle_root.hex()

    # Include weight file hash for TEE model identity verification
    if hasattr(state, "tee_weight_file_hash") and state.tee_weight_file_hash:
        result["model_weight_hash"] = state.tee_weight_file_hash.hex()

    return JSONResponse(content=result)


@app.post("/tee/reattest")
async def tee_reattest(request: Request):
    """Fresh attestation with validator-provided nonce.

    Proves the TEE is still live — old attestations don't contain the
    new nonce, so replay is impossible.

    Request body: ``{"nonce": "<hex-encoded 32+ bytes>"}``
    """
    if not state.tee_enabled:
        return JSONResponse(status_code=404, content={"error": "TEE not enabled"})

    from verallm.tee.attestation import get_attestation_provider
    from verallm.tee.serialization import attestation_to_dict

    body = await request.json()
    nonce_hex = body.get("nonce", "")
    if not nonce_hex or len(nonce_hex) < 16:  # 16 hex chars = 8 bytes
        return JSONResponse(status_code=400, content={"error": "nonce required (hex-encoded, minimum 8 bytes / 16 hex chars)"})

    nonce = bytes.fromhex(nonce_hex)
    provider = get_attestation_provider(state.tee_platform)
    model_hash = getattr(state, "tee_weight_file_hash", b"") or b""

    fresh = provider.generate_reattestation(
        state.tee_public_key,
        model_hash,
        nonce,
    )
    return JSONResponse(content=attestation_to_dict(fresh))


@app.post("/tee/chat")
async def tee_chat(body: TEEChatRequestBody, request: Request):
    """Encrypted chat inference for TEE mode.

    The client encrypts an OpenAI-style chat request to the enclave's public
    key.  The server decrypts inside the TEE, runs inference, and returns
    encrypted token chunks + an optional proof bundle (if proofs are enabled).

    SSE event types:
      - encrypted_token: {seq, ciphertext} — per-token encrypted chunk
      - done: {encrypted_output, commitment?, proof_bundle?, timing}
      - error: {error}
    """
    if not state.tee_enabled:
        return JSONResponse(status_code=404, content={"error": "TEE not enabled"})
    if state.miner is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    from verallm.tee.crypto import decrypt_payload, encrypt_payload
    from verallm.tee.types import EncryptedEnvelope

    # Parse the encrypted envelope
    try:
        env = body.envelope
        envelope = EncryptedEnvelope(
            session_id=env["session_id"],
            sender_public_key=bytes.fromhex(env["sender_public_key"]),
            nonce=bytes.fromhex(env["nonce"]),
            ciphertext=bytes.fromhex(env["ciphertext"]),
        )
    except (KeyError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid envelope: {e}"},
        )

    # Decrypt the chat request inside the TEE
    try:
        plaintext = decrypt_payload(envelope, state.tee_private_key)
        chat_request = json.loads(plaintext)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Decryption failed: {e}"},
        )

    # Parse the decrypted chat request (OpenAI-style messages)
    messages = chat_request.get("messages", [])
    max_new_tokens = chat_request.get("max_new_tokens", 4096)
    do_sample = chat_request.get("do_sample", False)
    temperature = chat_request.get("temperature", 1.0)
    enable_thinking = chat_request.get("enable_thinking", True)

    # Apply chat template
    tokenizer = state.miner.tokenizer
    messages_dicts = [{"role": m["role"], "content": m["content"]} for m in messages]

    _is_mistral_tok = "MistralTokenizer" in type(tokenizer).__name__
    _extra_kw = _chat_template_kwargs(tokenizer, enable_thinking)
    try:
        if _is_mistral_tok:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages_dicts, tokenize=True, add_generation_prompt=True
            )
            formatted_prompt = None
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                messages_dicts, tokenize=False, add_generation_prompt=True,
                **_extra_kw,
            )
            prompt_token_ids = None
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Chat template failed: {e}"},
        )

    nonce = bytes.fromhex(body.validator_nonce)
    sender_pk = envelope.sender_public_key

    # Compute prompt_hash for TEE chat (same canonical JSON format).
    import json as _json
    _prompt_hash = hashlib.sha256(
        _json.dumps(messages_dicts, sort_keys=True, ensure_ascii=False).encode()
    ).digest()

    _vhk = getattr(getattr(request, "state", None), "validator_hotkey", "") if request else ""

    async def _stream_tee_inference():
        """Run inference and stream encrypted results."""
        t0 = time.time()
        seq = 0
        full_output_tokens = []

        # Build a synthetic request for the existing inference pipeline
        synth_body = InferenceRequestBody(
            prompt=formatted_prompt or "",
            validator_nonce=body.validator_nonce,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            enable_thinking=getattr(body, "enable_thinking", True),
        )

        try:
            if state.batch_mode and state.batch_engine is not None:
                # Use batch inference path
                gen = _stream_inference_batched(
                    synth_body, nonce,
                    prompt_token_ids=prompt_token_ids,
                    token_budget=(len(prompt_token_ids or []) or 0) + max_new_tokens,
                    prompt_hash=_prompt_hash,
                    validator_hotkey=_vhk,
                )
            else:
                gen = _stream_inference(
                    synth_body, nonce,
                    prompt_token_ids=prompt_token_ids,
                    prompt_hash=_prompt_hash,
                    validator_hotkey=_vhk,
                )

            # Intercept SSE events from the underlying inference generator,
            # encrypt tokens, and pass through proof/commitment data.
            commitment_data = None
            proof_data = None
            timing_data = None

            async for chunk in gen:
                # SSE chunks may contain multiple lines:
                # "event: token\ndata: {...}\n\n"
                # Extract the data line.
                raw = ""
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        raw = line.removeprefix("data: ").strip()
                        break
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "text" in event and "done" not in event and "commitment" not in event:
                    # Token event — encrypt and forward
                    token_text = event["text"]
                    full_output_tokens.append(token_text)
                    encrypted_chunk = encrypt_payload(
                        token_text.encode("utf-8"),
                        state.tee_private_key,
                        sender_pk,
                        envelope.session_id,
                    )
                    yield (
                        f"data: {json.dumps({'event': 'encrypted_token', 'seq': seq, 'ciphertext': encrypted_chunk.ciphertext.hex(), 'nonce': encrypted_chunk.nonce.hex()})}\n\n"
                    )
                    seq += 1
                elif "done" in event or "commitment" in event:
                    # Final event — extract proof data and output text
                    commitment_data = event.get("commitment")
                    proof_data = event.get("proof_bundle")
                    # Timing may be nested under "timing" or flat in the event
                    timing_data = event.get("timing") or {
                        k: event[k] for k in (
                            "input_tokens", "output_tokens", "inference_ms",
                            "ttft_ms", "commitment_ms", "prove_ms",
                            "beacon_ms", "challenge_ms", "model_id",
                        ) if k in event
                    }
                    # In skip-proofs/TEE mode, output_text comes in the
                    # done event (no per-token streaming).
                    if "output_text" in event and not full_output_tokens:
                        full_output_tokens.append(event["output_text"])

            # Encrypt the full output
            full_output = "".join(full_output_tokens)
            encrypted_output = encrypt_payload(
                full_output.encode("utf-8"),
                state.tee_private_key,
                sender_pk,
                envelope.session_id,
            )

            if timing_data and "model_id" not in timing_data and state.miner:
                timing_data["model_id"] = getattr(state.miner, "model_id", "")
            done_event = {
                "event": "done",
                "encrypted_output": encrypted_output.ciphertext.hex(),
                "encrypted_output_nonce": encrypted_output.nonce.hex(),
                "timing": timing_data or {"total_s": time.time() - t0},
            }
            # Include proof data if proofs are enabled (non-TEE-skip mode)
            if commitment_data is not None:
                done_event["commitment"] = commitment_data
            if proof_data is not None:
                done_event["proof_bundle"] = proof_data

            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        _stream_tee_inference(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Epoch receipt endpoints — validators push signed receipts, pull at epoch end
# ============================================================================

class EpochReceiptBody(BaseModel):
    miner_address: str
    model_id: str
    model_index: int
    epoch_number: int
    commitment_hash: str  # hex
    timestamp: int
    ttft_ms: float
    tokens_generated: int
    generation_time_ms: float
    tokens_per_sec: float
    prompt_tokens: int = 0
    proof_verified: bool = False
    proof_requested: bool = False
    tee_attestation_verified: Optional[bool] = None  # None=not tested, True=passed, False=failed
    is_canary: bool = False
    validator_hotkey: str  # hex
    validator_signature: str  # hex


@app.post("/epoch/receipt")
async def receive_epoch_receipt(body: EpochReceiptBody):
    """Accept a validator-signed service receipt for the current epoch.

    After verified inference, the validator pushes a signed receipt to the
    miner.  The miner accumulates receipts from ALL validators throughout
    the epoch.  At epoch boundary, validators pull the complete batch via
    GET /epoch/{n}/receipts — every validator sees the same set.

    Receipts are persisted to SQLite so they survive server restarts.
    """
    epoch = body.epoch_number

    # Anti-hijacking: reject receipts addressed to a different miner
    if state.evm_address and body.miner_address.lower() != state.evm_address.lower():
        return JSONResponse(
            status_code=403,
            content={"error": "Receipt address mismatch — this endpoint belongs to a different miner"},
        )

    receipt_dict = body.model_dump()

    # No artificial receipt cap — throughput is naturally bounded by inference
    # rate, and validator auth + epoch GC prevent abuse.
    count = state.receipt_store.add(epoch, receipt_dict)

    # Auto-GC: remove old epochs from memory + disk
    state.receipt_store.gc(epoch)

    return {"status": "accepted", "epoch": epoch, "count": count}


@app.get("/epoch/{epoch_number}/receipts")
async def get_epoch_receipts(epoch_number: int):
    """Return all accumulated receipts for the given epoch.

    Validators pull this at epoch boundary.  Every validator receives the
    SAME receipt set, computes the SAME scores, and produces IDENTICAL
    weights (Yuma consensus).
    """
    receipts = state.receipt_store.get(epoch_number)
    return {
        "epoch": epoch_number,
        "receipt_count": len(receipts),
        "receipts": receipts,
    }


# ============================================================================
# SSE inference streaming + proof generation
# ============================================================================

async def _stream_inference(body: InferenceRequestBody, nonce: bytes,
                            prompt_token_ids: list[int] | None = None,
                            prompt_hash: bytes = b""):
    """Generator that yields SSE events: token deltas, then commitment + proofs.

    Args:
        body: Request body with prompt, nonce, and generation params.
        nonce: Validator nonce bytes.
        prompt_token_ids: If provided (e.g. from /chat with Mistral tokenizer),
            use these token IDs directly instead of body.prompt string.
    """
    from vllm import SamplingParams

    state._last_request_time = time.monotonic()
    miner = state.miner
    set_config(state.config)
    session_id = str(uuid.uuid4())

    # Register activation hooks
    activations = {}
    hook_handles = []

    def make_hook(layer_idx, name):
        def hook(module, inp, output):
            key = f"layer_{layer_idx}_{name}"
            if isinstance(inp, tuple) and len(inp) > 0:
                activations[f"{key}_input"] = inp[0].detach().float().cpu()
            if isinstance(output, torch.Tensor):
                activations[f"{key}_output"] = output.detach().float().cpu()
        return hook

    for idx, layer in enumerate(miner._get_layers()):
        mlp = miner._get_mlp(layer)
        if mlp is not None:
            if miner.is_moe and is_moe_layer(layer):
                hook_handles.append(mlp.register_forward_hook(make_hook(idx, "mlp_gate")))
            else:
                gate_proj = miner._get_gate_proj(mlp)
                if gate_proj is not None:
                    hook_handles.append(gate_proj.register_forward_hook(make_hook(idx, "mlp_gate")))

    # Embedding output hook DISABLED — see verallm/api/client.py
    # verify_proof() for rationale.  Hook preserved here for re-enablement.
    # from verallm.introspection import get_embedding_module
    # embed_mod = get_embedding_module(miner.model)
    # if embed_mod is not None:
    #     def _embed_hook(module, inp, output):
    #         if isinstance(output, torch.Tensor):
    #             activations["embedding_output"] = output.detach().float().cpu()
    #     hook_handles.append(embed_mod.register_forward_hook(_embed_hook))

    # Install MoE router hooks (captures routing decisions for commitment)
    moe_hook_mgr = None
    if miner.is_moe and miner.model is not None:
        moe_hook_mgr = MoEHookManager(
            miner.model,
            router_top_k=getattr(miner.model_spec, "router_top_k", 0),
            router_scoring=getattr(miner.model_spec, "router_scoring", "softmax"),
        )
        moe_hook_mgr.install_hooks()

    sampling_params = SamplingParams(
        **_resolve_sampling_params(body, state.model_name),
    )

    engine = miner.llm.llm_engine
    t_infer = time.perf_counter()
    if prompt_token_ids is not None:
        engine.add_request("stream-0", {"prompt_token_ids": prompt_token_ids}, sampling_params)
    else:
        engine.add_request("stream-0", body.prompt, sampling_params)

    prev_text = ""
    final_output = None
    t_first_token = None

    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            cur_text = output.outputs[0].text if output.outputs else ""
            delta = cur_text[len(prev_text):]
            if delta:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"
            prev_text = cur_text
            if output.finished:
                final_output = output

    inference_ms = (time.perf_counter() - t_infer) * 1000
    ttft_ms = ((t_first_token - t_infer) * 1000) if t_first_token else 0

    for h in hook_handles:
        h.remove()

    # Capture router commitments from MoE hooks
    session_router_commitments = {}
    session_router_logits = {}
    if moe_hook_mgr is not None:
        from verallm.crypto.field import P as FIELD_PRIME
        for layer_idx in moe_hook_mgr._challenged_layers:
            rc = moe_hook_mgr.build_router_commitment(layer_idx, FIELD_PRIME)
            if rc is not None:
                session_router_commitments[layer_idx] = rc
            decision = moe_hook_mgr.get_router_decision(layer_idx)
            if decision is not None:
                session_router_logits[layer_idx] = decision.router_logits.detach().float().cpu()
        moe_hook_mgr.remove_hooks()

    if final_output is None:
        yield f"event: error\ndata: {json.dumps({'error': 'No output generated'})}\n\n"
        return

    input_token_ids = final_output.prompt_token_ids
    output_token_ids = final_output.outputs[0].token_ids

    # Bind router GEMM verification to a quantization-agnostic committed path.
    _attach_proof_domain_router_topk(miner, activations, session_router_commitments)

    # ── Build commitment ──────────────────────────────────────────────
    _resolved = _resolve_sampling_params(body, state.model_name)
    resolved_pp = _resolved["presence_penalty"]
    _pending = getattr(miner, "_pending_sampling_seeds", {}) or {}
    _seed_for_commit = _pending.pop(session_id, b"")
    commitment, commitment_ms = _build_commitment(
        miner, activations, input_token_ids, output_token_ids,
        session_id, inference_ms,
        router_commitments=session_router_commitments,
        do_sample=body.do_sample,
        temperature=body.temperature,
        sampling_verification_bps=body.sampling_verification_bps,
        presence_penalty=resolved_pp,
        prompt_hash=prompt_hash,
        top_k=int(_resolved.get("top_k", -1) or -1),
        top_p=float(_resolved.get("top_p", 1.0) or 1.0),
        min_p=float(_resolved.get("min_p", 0.0) or 0.0),
        sampling_seed=_seed_for_commit,
    )
    # Store router commitments for proof bundle
    if session_router_commitments:
        miner.router_commitments[session_id] = session_router_commitments
    if session_router_logits:
        miner.router_logits[session_id] = session_router_logits
    # Store input token IDs for embedding proof generation.
    miner.input_token_ids[session_id] = list(input_token_ids)

    # ── Phase 4: Derive beacon (Fiat-Shamir) ────────────────────────
    t0 = time.perf_counter()
    beacon = derive_beacon_from_nonce(
        commitment_hash=commitment.commitment_hash(),
        validator_nonce=nonce,
    )
    beacon_ms = (time.perf_counter() - t0) * 1000

    # ── Phase 5: Derive challenges (Fiat-Shamir) ────────────────────
    t0 = time.perf_counter()
    if state.moe_config is not None:
        challenges = derive_moe_challenges(
            beacon=beacon,
            commitment=commitment,
            moe_config=state.moe_config,
            router_commitments=session_router_commitments,
            k_layers=state.config.k_layers,
            k_tokens_per_layer=state.config.k_tokens_per_expert,
            k_experts_per_layer=state.config.k_experts_per_layer,
        )
    else:
        challenges = derive_challenges(
            beacon=beacon,
            commitment=commitment,
            k_layers=state.config.k_layers,
            k_gemms_per_layer=2,
            k_blocks_per_gemm=state.config.k_blocks,
        )
    challenges.sampling_challenge = derive_sampling_challenge(
        beacon=beacon,
        commitment=commitment,
        vocab_size=int(getattr(miner.model_spec, "vocab_size", 0) or 0),
    )
    # Derive embedding challenge for input binding verification.
    emb_root = getattr(miner.model_spec, "embedding_weight_merkle_root", b"")
    if emb_root:
        challenges.embedding_challenge = derive_embedding_challenge(
            beacon=beacon,
            commitment=commitment,
            num_input_tokens=len(input_token_ids),
        )
    challenge_ms = (time.perf_counter() - t0) * 1000

    # ── Phase 6: Generate proofs ────────────────────────────────────
    t0 = time.perf_counter()
    proof_bundle, timing_details, _ = miner.generate_proofs(
        commitment, challenges, validator_nonce=nonce,
    )
    prove_ms = (time.perf_counter() - t0) * 1000

    # Clean up session state
    # TODO(concurrency): With the concurrency guard (§3.1), only one request
    #   runs at a time, so cleanup races are impossible.  Without it, this
    #   pop is still safe (session_id is unique) but overlapping requests
    #   accumulate memory until each finishes.
    miner.witnesses.pop(session_id, None)
    miner.activation_merkle_trees.pop(session_id, None)
    miner.router_commitments.pop(session_id, None)
    miner.router_logits.pop(session_id, None)
    miner.decode_hidden_row_trees.pop(session_id, None)
    miner.decode_hidden_rows.pop(session_id, None)
    miner.decode_logits_row_trees.pop(session_id, None)
    miner.decode_logits_rows.pop(session_id, None)
    miner.input_token_ids.pop(session_id, None)
    miner.embedding_output_trees.pop(session_id, None)
    miner.output_token_ids.pop(session_id, None)

    # ── Emit final SSE event with commitment + proofs ───────────────
    done_data = {
        "commitment": commitment_to_dict(commitment),
        "proof_bundle": proof_bundle_to_dict(proof_bundle),
        "output_text": prev_text,
        "input_tokens": len(input_token_ids),
        "output_tokens": len(output_token_ids),
        "inference_ms": round(inference_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "commitment_ms": round(commitment_ms, 1),
        "beacon_ms": round(beacon_ms, 3),
        "challenge_ms": round(challenge_ms, 3),
        "prove_ms": round(prove_ms, 1),
        "prove_timing_details": timing_details,
    }
    _n_out = len(output_token_ids)
    _tps = _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
    bt.logging.info(f"Served {session_id[:12]} | {len(input_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms")
    yield f"event: done\ndata: {json.dumps(done_data)}\n\n"


def _build_commitment(miner, activations, input_token_ids, output_token_ids,
                      session_id, inference_ms, router_commitments=None,
                      *, do_sample: bool = False, temperature: float = 0.0,
                      sampling_verification_bps: int = 0,
                      presence_penalty: float = 0.0,
                      prompt_hash: bytes = b"",
                      top_k: int = -1, top_p: float = 1.0, min_p: float = 0.0,
                      sampling_seed: bytes = b""):
    """Build InferenceCommitment from captured activations."""
    t0 = time.perf_counter()

    witnesses = {}
    for key, tensor in activations.items():
        if not key.startswith("layer_"):
            continue  # Skip non-layer keys (e.g. lm_head_hidden_steps)
        parts = key.split("_")
        layer_idx = int(parts[1])
        if layer_idx not in witnesses:
            witnesses[layer_idx] = {}
        witnesses[layer_idx][key] = tensor

    miner.witnesses[session_id] = witnesses
    miner.activation_merkle_trees[session_id] = {}

    # Try CUDA-accelerated BLAKE3 for activation Merkle leaves.
    _has_cuda_activation_blake3 = False
    try:
        from zkllm.cuda import zkllm_native as _native
        _has_cuda_activation_blake3 = (
            getattr(_native, 'HAS_CUDA', False)
            and hasattr(_native, 'cuda_blake3_activation_merkle_leaves')
        )
    except ImportError:
        pass

    def build_activation_merkle_tree(tensor, block_size=256):
        flat = tensor.flatten()
        if flat.isnan().any():
            flat = flat.nan_to_num(0.0)
        absmax = flat.abs().max().clamp(min=1e-8)
        # Vectorized: single quantization op + single numpy() call.
        quantized = (flat / absmax * 127).round().clamp(-128, 127).to(torch.int64)

        n = len(quantized)
        if n == 0:
            return MerkleTree([b"empty"]), MerkleTree([b"empty"]).root

        # GPU fast path: hash all leaves on GPU in parallel.
        # Only used when tensor is already on GPU (e.g. buffer-mode GPUs
        # where finalize keeps data on GPU).
        if _has_cuda_activation_blake3 and quantized.is_cuda:
            leaf_hashes_tensor = _native.cuda_blake3_activation_merkle_leaves(
                quantized.contiguous(), block_size,
                quantized.device.index or 0,
            )
            num_leaves = leaf_hashes_tensor.shape[0]
            leaf_hash_list = [
                bytes(leaf_hashes_tensor[i].numpy())
                for i in range(num_leaves)
            ]
            tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
            return tree, tree.root

        # CPU path: vectorized quantization + numpy slicing
        if quantized.is_cuda:
            quantized = quantized.cpu()
        arr = quantized.numpy()
        leaves = [arr[i:i + block_size].tobytes() for i in range(0, n, block_size)]
        tree = MerkleTree(leaves)
        return tree, tree.root

    layer_commitments = []
    num_layers = miner._get_num_layers()
    for i in range(num_layers):
        if i in witnesses:
            input_key = f"layer_{i}_mlp_gate_input"
            if input_key in witnesses[i]:
                X = witnesses[i][input_key]
                if X.dim() == 3:
                    X = X.view(-1, X.shape[-1])
                # Marlin MXFP4 pads FusedMoE input to 256-aligned boundary
                # (e.g. hidden_size 2880 → 3072).  Truncate to hidden_dim
                # so the commitment matches proof generation (which also truncates).
                _hidden_dim = miner._get_hidden_dim()
                if X.shape[-1] > _hidden_dim:
                    X = X[..., :_hidden_dim]

                # Use pre-computed leaf hashes from CUDA BLAKE3 if available
                # (computed on GPU during capture_at_split, ~50× faster).
                _pre_hash_key = f"_leaf_hashes_{input_key}"
                if _pre_hash_key in activations:
                    leaf_hashes_tensor = activations[_pre_hash_key]
                    num_leaves = leaf_hashes_tensor.shape[0]
                    leaf_hash_list = [
                        bytes(leaf_hashes_tensor[j].numpy())
                        for j in range(num_leaves)
                    ]
                    tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
                    root = tree.root
                else:
                    tree, root = build_activation_merkle_tree(X, block_size=256)

                miner.activation_merkle_trees[session_id][i] = (tree, tuple(X.shape))
                layer_commitments.append(root)
            else:
                layer_commitments.append(hashlib.sha256(f"layer_{i}_no_input".encode()).digest())
        else:
            layer_commitments.append(hashlib.sha256(f"layer_{i}".encode()).digest())

    # Decode-integrity: build Merkle trees over hidden rows and logits rows.
    # Built for greedy mode (temp=0) AND for canonical-sampler mode
    # (do_sample=True with sampling_seed committed).  In canonical mode the
    # logits processor masks the GPU logits to force the CPU-chosen token,
    # so the post-mask greedy verification path is also valid.
    decode_hidden_row_root = b""
    hidden_steps = activations.get("lm_head_hidden_steps", [])
    _canonical_active = bool(do_sample) and bool(sampling_seed)
    is_greedy = ((not do_sample) and temperature_to_milli(temperature) == 0) or _canonical_active
    bt.logging.debug(
        f"[CANON-COMMIT] do_sample={do_sample} sampling_seed_len={len(sampling_seed) if sampling_seed else 0} "
        f"canonical_active={_canonical_active} is_greedy={is_greedy} "
        f"hidden_steps_len={len(hidden_steps) if isinstance(hidden_steps, list) else 'N/A'}"
    )

    # Self-check: verify logits-to-output-token alignment before committing.
    # The capture hook can sometimes produce misaligned logits rows (e.g., when
    # concurrent requests cause batch composition changes mid-sequence).
    # Committing misaligned data would cause false proof failures.
    #
    # IMPORTANT: when canonical sampler is active (do_sample=True with seed),
    # the captured logits are the PRE-mask logits (compute_logits hook fires
    # before the LogitsProcessor runs).  The output token is the canonically
    # chosen token, which is NOT the argmax of pre-mask logits.  Skip the
    # argmax-alignment check in this case — the validator's canonical replay
    # branch handles correctness verification end-to-end.
    logits_steps = activations.get("lm_head_logits_steps", [])
    # Materialize deferred GPU tensors.  The capture hook stores
    # (vals_gpu, idx_gpu) tuples to avoid per-step D2H syncs;
    # bulk-convert to serialized leaf bytes at commitment time.
    if logits_steps and isinstance(logits_steps[0], tuple):
        from verallm.sampling import serialize_top_k_to_bytes as _ser_topk
        import numpy as _np_mat
        _materialized = []
        for _item in logits_steps:
            if isinstance(_item, tuple):
                _v_gpu, _i_gpu = _item
                _vn = _v_gpu.float().cpu().numpy()
                _in = _i_gpu.cpu().numpy().astype(_np_mat.int64)
                _ord = _np_mat.lexsort((_in, -_vn))
                _materialized.append(_ser_topk(_vn[_ord], _in[_ord]))
            else:
                _materialized.append(_item)
        logits_steps = _materialized
    _logits_aligned = True
    if _canonical_active:
        # When the model finishes via stop token (finish_reason=stop),
        # the last decode step produces a token that is NOT counted in
        # output_token_ids, but the compute_logits hook still captures
        # its logits.  Strip the LAST entry (the uncounted stop-token
        # logits) to keep per_step aligned 1:1 with output_token_ids.
        if isinstance(logits_steps, list) and len(logits_steps) == len(output_token_ids) + 1:
            logits_steps = logits_steps[:-1]  # drop the stop-token entry
        if isinstance(logits_steps, list) and len(logits_steps) != len(output_token_ids):
            _logits_aligned = False
            bt.logging.warning(
                f"Logits/output length mismatch (canonical mode): "
                f"logits_steps={len(logits_steps)}, output_tokens={len(output_token_ids)}. "
                f"Skipping decode commitment."
            )
    elif is_greedy and isinstance(logits_steps, list) and logits_steps:
        # Captured leaves are top-K bytes: K × fp32 vals + K × int64 indices,
        # sorted by (value DESC, index ASC).  The global argmax is at
        # sorted position 0, so we just check `top_idx[0] == output_token`.
        from verallm.sampling import parse_top_k_leaf as _parse_top_k_leaf
        _n_check = min(len(logits_steps), len(output_token_ids))
        _mismatches = []
        for _si in range(_n_check):
            _step_leaf = logits_steps[_si]
            _argmax = None
            if isinstance(_step_leaf, (bytes, bytearray, memoryview)):
                _step_leaf = bytes(_step_leaf)
                if len(_step_leaf) > 0 and len(_step_leaf) % 12 == 0:
                    try:
                        _vals, _idx = _parse_top_k_leaf(_step_leaf)
                        if _idx.size > 0:
                            _argmax = int(_idx[0])
                    except ValueError:
                        _argmax = None
            elif isinstance(_step_leaf, torch.Tensor) and _step_leaf.numel() > 0:
                # Legacy full-vocab tensor capture (kept for compatibility).
                _argmax = int(_step_leaf.float().squeeze().argmax().item())
            if _argmax is None:
                continue
            _expected = int(output_token_ids[_si])
            if _argmax != _expected:
                _mismatches.append((_si, _argmax, _expected))
        if _mismatches:
            _logits_aligned = False
            bt.logging.debug(
                f"Sampling divergence: {len(_mismatches)}/{_n_check} steps (skip decode commitment)"
            )
        elif len(logits_steps) != len(output_token_ids):
            _logits_aligned = False
            bt.logging.warning(
                f"Logits/output length mismatch: logits_steps={len(logits_steps)}, output_tokens={len(output_token_ids)}. "
                f"Skipping decode commitment."
            )

    if isinstance(hidden_steps, list) and hidden_steps and is_greedy and _logits_aligned:
        _t_hm = time.perf_counter()
        hidden_tree, hidden_rows, decode_hidden_row_root = build_hidden_row_merkle(hidden_steps)
        _t_hm = (time.perf_counter() - _t_hm) * 1000
        if hidden_tree is not None:
            miner.decode_hidden_row_trees[session_id] = hidden_tree
            miner.decode_hidden_rows[session_id] = hidden_rows
            bt.logging.debug(f"Hidden row Merkle: {_t_hm:.1f}ms, {len(hidden_steps)} steps")

    decode_logits_row_root = b""
    bps = clamp_sampling_bps(sampling_verification_bps)
    # Build the fp32 logits Merkle tree when:
    #   - high-assurance bps (>=9000) — full canary verification, OR
    #   - canonical sampler is active (do_sample=True with seed) — the
    #     validator's canonical replay needs the captured fp32 logits
    #     row at any bps where the sampling challenge might fire,
    #     including organic at bps=1000.  Without this, organic
    #     stochastic verification was only theoretical (the LP fires
    #     on the miner side, but the validator could not actually
    #     replay because the fp32 row was missing from the proof).
    _need_fp32_tree = bps >= HIGH_ASSURANCE_BPS or _canonical_active
    if isinstance(logits_steps, list) and logits_steps and is_greedy and _need_fp32_tree and _logits_aligned:
        _t_lm = time.perf_counter()
        logits_tree, logits_rows, decode_logits_row_root = build_logits_row_merkle(logits_steps)
        _t_lm = (time.perf_counter() - _t_lm) * 1000
        if logits_tree is not None:
            miner.decode_logits_row_trees[session_id] = logits_tree
            miner.decode_logits_rows[session_id] = logits_rows
            bt.logging.debug(
                f"Logits row Merkle: {_t_lm:.1f}ms, {len(logits_steps)} steps, ~{len(logits_steps) * logits_rows[0].__len__() / 1024 / 1024:.1f}MB data"
            )

    input_ids_tensor = torch.tensor(input_token_ids, dtype=torch.int64)
    output_ids_tensor = torch.tensor(output_token_ids, dtype=torch.int64)
    input_commitment = hashlib.sha256(input_ids_tensor.numpy().tobytes()).digest()
    output_commitment = hashlib.sha256(output_ids_tensor.numpy().tobytes()).digest()
    temp_milli = temperature_to_milli(temperature)

    # Embedding output tree DISABLED — see comment in
    # verallm/api/client.py verify_proof() for rationale.

    # Layer transition hash chain (anchored to input_commitment).
    from verallm.miner.base import Miner as _MinerBase
    transition_hashes = _MinerBase.compute_layer_transition_hashes(
        layer_commitments, input_commitment,
    )

    # Compute sampler config hash from actual sampling params used.
    # NOTE: chat_template_hash is intentionally NOT included here — until the
    # on-chain registry stores it, the validator cannot compute a matching
    # expected hash.  Template binding is a future enhancement.
    from verallm.sampling import compute_sampler_config_hash as _compute_scfg
    sampler_cfg_hash = _compute_scfg(
        top_k=int(top_k), top_p=float(top_p), min_p=float(min_p),
        presence_penalty=float(presence_penalty),
    )

    # Sampling seed commitment for do_sample=True canonical replay.
    sampling_seed_commitment = b""
    if bool(do_sample) and sampling_seed:
        sampling_seed_commitment = hashlib.sha256(sampling_seed).digest()
        # Stash for proof bundle.
        if not hasattr(miner, "_sampling_seeds"):
            miner._sampling_seeds = {}
        miner._sampling_seeds[session_id] = sampling_seed

    commitment = InferenceCommitment(
        session_id=session_id,
        model_id=miner.model_name,
        model_commitment=miner.model_commitment,
        input_commitment=input_commitment,
        output_commitment=output_commitment,
        layer_commitments=layer_commitments,
        router_commitment_hash=InferenceCommitment.compute_router_hash(router_commitments or {}),
        decode_hidden_row_root=decode_hidden_row_root,
        decode_logits_row_root=decode_logits_row_root,
        sampling_verification_bps=bps,
        output_token_count=int(len(output_token_ids)),
        do_sample=bool(do_sample),
        temperature_milli=temp_milli,
        presence_penalty_milli=int(round(float(presence_penalty) * 1000.0)),
        layer_transition_hashes=transition_hashes,
        prompt_hash=prompt_hash,
        sampler_config_hash=sampler_cfg_hash,
        sampling_seed_commitment=sampling_seed_commitment,
        timestamp=time.time(),
    )
    miner.output_token_ids[session_id] = [int(t) for t in output_token_ids]
    commitment_ms = (time.perf_counter() - t0) * 1000
    return commitment, commitment_ms


def _attach_proof_domain_router_topk(miner, activations, router_commitments) -> None:
    """Populate quantized proof-domain router top-k for all committed MoE layers."""
    if not router_commitments:
        return

    updated = 0
    for layer_idx, router_commitment in router_commitments.items():
        if router_commitment is None:
            continue
        top_k = int(getattr(router_commitment, "top_k", 0) or 0)
        if top_k <= 0:
            continue
        x_key = f"layer_{int(layer_idx)}_mlp_gate_input"
        x_activation = activations.get(x_key)
        if not isinstance(x_activation, torch.Tensor):
            continue

        proof_selected = miner.compute_router_proof_selected_experts(
            int(layer_idx),
            x_activation,
            top_k,
        )
        if proof_selected:
            router_commitment.proof_selected_experts = [
                [int(expert_idx) for expert_idx in row]
                for row in proof_selected
            ]
            updated += 1

    if updated and updated != len(router_commitments):
        bt.logging.warning(
            f"Router proof-domain top-k attached for {updated}/{len(router_commitments)} layers"
        )


# ============================================================================
# Batch mode: step loop + batched inference stream
# ============================================================================

async def _engine_step_loop():
    """Background task: run engine.step_and_distribute() in a loop.

    Runs continuously while batch mode is active.  When no requests are
    in-flight, sleeps briefly to avoid busy-spinning.  Each step() call
    runs synchronously in a thread executor so the async event loop stays
    responsive for SSE streaming and new request acceptance.
    """
    loop = asyncio.get_event_loop()
    batch_engine = state.batch_engine

    # Per-request step timing instrumentation
    _step_times: list[float] = []
    _gpu_times: list[float] = []  # thread execution only
    _step_count = 0
    _max_active = 0


    def _timed_step():
        """Run step_and_distribute and return GPU-side wall time."""
        t = time.perf_counter()
        batch_engine.step_and_distribute()
        return time.perf_counter() - t

    while True:
        if not batch_engine.has_active_requests():
            if _step_count > 0:
                avg_ms = sum(_step_times) / len(_step_times) * 1000
                max_ms = max(_step_times) * 1000
                avg_gpu = sum(_gpu_times) / len(_gpu_times) * 1000
                max_gpu = max(_gpu_times) * 1000
                overhead_ms = avg_ms - avg_gpu
                bt.logging.debug(
                    f"STEP LOOP: {_step_count} steps, wall avg={avg_ms:.1f}ms max={max_ms:.1f}ms, "
                    f"gpu avg={avg_gpu:.1f}ms max={max_gpu:.1f}ms, event_loop_overhead={overhead_ms:.1f}ms, "
                    f"max_batch={_max_active}, total={sum(_step_times) * 1000:.0f}ms"
                )
                _step_times.clear()
                _gpu_times.clear()
                _step_count = 0
                _max_active = 0
            await asyncio.sleep(0.002)
            continue
        try:
            na = batch_engine.num_active
            if na > _max_active:
                _max_active = na
            t_step = time.perf_counter()
            gpu_time = await loop.run_in_executor(None, _timed_step)
            _step_times.append(time.perf_counter() - t_step)
            _gpu_times.append(gpu_time)
            _step_count += 1
        except Exception:
            logger.exception("Error in engine step loop")
            await asyncio.sleep(0.01)
        # Yield to event loop so SSE generators can send queued outputs
        await asyncio.sleep(0)


async def _stream_inference_batched(body: "InferenceRequestBody", nonce: bytes,
                                     prompt_token_ids: list[int] | None = None,
                                     token_budget: int = 0,
                                     admitted_request_id: str | None = None,
                                     prompt_hash: bytes = b"",
                                     validator_hotkey: str = ""):
    """Batched inference generator: uses BatchAwareEngine + activation tracker.

    Same SSE protocol as _stream_inference but supports concurrent requests
    via vLLM's continuous batching.  Hooks are persistent (installed once at
    startup) and demux activations per-request using query_start_loc.

    Admission control is dynamic and token-based: each request reserves
    ``token_budget`` (prompt + max_new_tokens) from the shared KV cache pool.
    This allows many small requests to batch concurrently while guaranteeing
    that large requests (up to advertised max context) are always possible
    when budget is available.

    Args:
        body: Request body with prompt, nonce, and generation params.
        nonce: Validator nonce bytes.
        prompt_token_ids: If provided, use these token IDs directly.
        token_budget: KV cache tokens to reserve (prompt + max_new_tokens).
        admitted_request_id: If provided, admission was already done at
            endpoint level — skip try_admit and use this request_id.
    """
    from vllm import SamplingParams

    # Update keepalive timestamp — real traffic keeps CUDA graphs warm
    state._last_request_time = time.monotonic()

    miner = state.miner
    batch_engine = state.batch_engine
    tracker = state.activation_tracker
    moe_mgr = state.moe_hook_mgr
    proof_pipeline = state.proof_pipeline
    admission = state.admission
    set_config(state.config)

    if admitted_request_id is not None:
        # Admission already done at endpoint level (proper 503 returned there)
        request_id = admitted_request_id
        session_id = str(uuid.uuid4())
    else:
        # Fallback: admit here (SSE error event if rejected)
        session_id = str(uuid.uuid4())
        request_id = f"req-{session_id[:8]}"
        admitted = await admission.try_admit(request_id, token_budget)
        if not admitted:
            s = admission.status()
            error_data = json.dumps({
                "error": "Miner busy: KV cache full",
                "free_tokens": s.free_tokens,
                "requested_tokens": token_budget,
                "active_requests": s.active_requests,
                "retry_after_ms": 5000,
            })
            yield f"event: error\ndata: {error_data}\n\n"
            return

    try:
        # Register with activation tracker (before first engine step).
        # Capture logits when:
        #   - high-assurance bps (>=9000) — full canary verification, OR
        #   - the request will run canonical sampler mode (do_sample=True
        #     with bps>0) — the validator's canonical replay needs the
        #     captured fp32 logits row at any bps where the sampling
        #     challenge might fire, including organic at bps=1000.
        # Skipping the capture would leave decode_logits_row_root empty
        # and the canonical replay verification would fail with
        # "Canonical replay requires fp16_logits_row".
        bps_for_request = clamp_sampling_bps(body.sampling_verification_bps)
        need_logits = bps_for_request >= HIGH_ASSURANCE_BPS or (
            body.do_sample and bps_for_request > 0
        )

        # Pre-generate sampling seed.  The canonical sampler now
        # runs inside patched_compute_logits (the lm_head hook) instead of
        # a separate AdapterLogitsProcessor — bypasses vLLM's per-step LP
        # dispatch overhead (~0.7-1.0 ms/step saved) and deduplicates the
        # topk+D2H+sort already done for the Merkle leaf capture.
        #
        # Path 1: fire for ALL do_sample requests regardless of bps.
        # Single security model — every sampled token is canonically bound.
        _seed_bytes = None
        if body.do_sample and bps_for_request > 0:
            import os as _os
            _seed_bytes = _os.urandom(32)
            if not hasattr(miner, "_pending_sampling_seeds"):
                miner._pending_sampling_seeds = {}
            miner._pending_sampling_seeds[session_id] = _seed_bytes
        bt.logging.debug(
            f"[CANON] do_sample={body.do_sample} bps={bps_for_request} session={session_id[:8]}"
            + (f" seed={_seed_bytes[:8].hex()}" if _seed_bytes else "")
        )

        if tracker is not None:
            tracker.register_request(
                request_id, session_id,
                capture_logits=need_logits or (_seed_bytes is not None),
                canonical_seed=_seed_bytes,
                canonical_temperature=max(0.001, float(body.temperature) if body.temperature else 1.0) if _seed_bytes else 1.0,
                canonical_top_k=int(body.top_k) if body.top_k is not None else -1,
                canonical_top_p=float(body.top_p) if body.top_p is not None else 1.0,
                canonical_min_p=float(body.min_p) if body.min_p is not None else 0.0,
            )

        # Resolve sampling params and build canonical sampler if do_sample=True.
        _resolved_sp = _resolve_sampling_params(body, state.model_name)

        sampling_params = SamplingParams(**_resolved_sp)
        if prompt_token_ids is not None:
            prompt = {"prompt_token_ids": prompt_token_ids}
        else:
            prompt = body.prompt

        t_infer = time.perf_counter()
        output_queue = batch_engine.add_request(request_id, prompt, sampling_params)

        # Stream tokens from per-request queue
        prev_text = ""
        final_output = None
        t_first_token = None

        while True:
            try:
                output = await asyncio.wait_for(output_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield f"event: error\ndata: {json.dumps({'error': 'Inference timeout'})}\n\n"
                batch_engine.abort_request(request_id)
                if tracker is not None:
                    tracker.unregister_request(request_id)
                if moe_mgr:
                    moe_mgr.clear_request(request_id)
                return

            cur_text = output.outputs[0].text if output.outputs else ""
            delta = cur_text[len(prev_text):]
            if delta:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"
            prev_text = cur_text

            if output.finished:
                final_output = output
                break

        inference_ms = (time.perf_counter() - t_infer) * 1000
        ttft_ms = ((t_first_token - t_infer) * 1000) if t_first_token else 0

        if final_output is None:
            yield f"event: error\ndata: {json.dumps({'error': 'No output generated'})}\n\n"
            tracker.unregister_request(request_id)
            if moe_mgr:
                moe_mgr.clear_request(request_id)
            return

        input_token_ids = final_output.prompt_token_ids
        output_token_ids = final_output.outputs[0].token_ids

        # Yield so the engine-step loop can schedule the next decode step
        # before this request enters post-inference processing.
        #
        # Buffer-mode dense capture snapshots activations from per-layer
        # device buffers at finalize-time; yielding here can let the next
        # decode step overwrite those buffers first, causing intermittent
        # verification mismatches. Keep finalize immediate in buffer mode.
        if tracker is not None and not tracker.has_capture_buffers:
            await asyncio.sleep(0)

        # TEE mode or SKIP_CAPTURE: skip all post-inference processing.
        # In TEE mode, hardware attestation replaces proofs entirely.
        _skip_proofs = state.tee_skip_proofs or os.environ.get("VERALLM_SKIP_CAPTURE", "0") == "1"
        if _skip_proofs:
            batch_engine.clear_finished(request_id)
            await admission.release(request_id)
            if tracker is not None:
                tracker.unregister_request(request_id)
            done_data = {
                "output_text": prev_text,
                "input_tokens": len(final_output.prompt_token_ids),
                "output_tokens": len(final_output.outputs[0].token_ids),
                "inference_ms": round(inference_ms, 1),
                "ttft_ms": round(ttft_ms, 1),
                "commitment_ms": 0,
                "beacon_ms": 0,
                "challenge_ms": 0,
                "prove_ms": 0,
                "prove_timing_details": {},
                "commitment": {},
                "proof_bundle": {"layer_proofs": [], "sampling_proofs": []},
            }
            _n_out = len(final_output.outputs[0].token_ids)
            _tps = _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
            bt.logging.info(f"Served {request_id} | {len(final_output.prompt_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms")
            if validator_hotkey:
                bt.logging.debug(f"  └─ validator: {validator_hotkey}")
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"
            return

        # Release KV token budget and batch engine slot immediately — the
        # proof pipeline only reads captured activations (CPU) and model
        # weights, not KV cache.  Early release lets new requests start
        # decoding while post-inference work runs in the executor.
        batch_engine.clear_finished(request_id)
        await admission.release(request_id)

        try:
            # Buffer-mode finalization MUST happen on the event loop thread
            # (synchronously) before yielding.  The capture buffers are live
            # GPU tensors overwritten on every forward() step.  If we defer
            # finalize to an executor thread, the next engine step can run
            # forward() and overwrite the buffers before the fallback readout
            # in finalize_activations() completes — causing the fallback to
            # read the NEXT request's activation data instead of this one's.
            #
            # After finalize, all activation data is on CPU and the executor
            # can safely process the rest (commitment, beacon, challenges).
            if tracker.has_capture_buffers:
                activations = tracker.finalize_activations(request_id)
            else:
                activations = None  # finalize in executor (no buffer race)

            # Post-inference work (router → commitment → beacon → challenges)
            # runs off the event loop in one executor call.  This keeps the
            # event loop free for engine stepping and token delivery.
            def _postprocess_sync():
                nonlocal activations
                if activations is None:
                    activations = tracker.finalize_activations(request_id)

                if tracker.backend == "splitting_ops":
                    has_mlp_capture = any(
                        k.endswith("_mlp_gate_input") for k in activations.keys()
                    )
                    if not has_mlp_capture:
                        raise RuntimeError(
                            "splitting_ops capture backend produced no MLP activations; "
                            "capture plugin is inactive"
                        )

                # Capture per-request router commitments (MoE)
                session_router_commitments = {}
                session_router_logits = {}
                if moe_mgr is not None:
                    # Process pre-captured router logits (splitting_ops backend only).
                    if tracker.backend == "splitting_ops":
                        for layer_idx in moe_mgr.get_challenged_layers():
                            moe_mgr.process_captured_router_logits(request_id, layer_idx)

                    from verallm.crypto.field import P as FIELD_PRIME
                    for layer_idx in moe_mgr.get_challenged_layers():
                        rc = moe_mgr.build_router_commitment_for_request(
                            request_id, layer_idx, FIELD_PRIME
                        )
                        if rc is not None:
                            session_router_commitments[layer_idx] = rc
                        decision = moe_mgr.get_router_decision_for_request(request_id, layer_idx)
                        if decision is not None:
                            session_router_logits[layer_idx] = decision.router_logits.detach().float().cpu()
                    moe_mgr.clear_request(request_id)
                    bt.logging.debug(
                        f"Built {len(session_router_commitments)} router commitments for {request_id} (backend={tracker.backend})"
                    )
                    if not session_router_commitments:
                        if tracker.backend == "splitting_ops":
                            router_keys = sorted(
                                k for k in activations.keys()
                                if k.endswith("_router_logits")
                            )
                            raise RuntimeError(
                                "splitting_ops router capture backend produced no router "
                                f"logits for request {request_id}; available router keys="
                                f"{router_keys[:8]}"
                            )
                        else:
                            mlp_keys = sorted(
                                k for k in activations.keys()
                                if k.endswith("_mlp_gate_input")
                            )
                            raise RuntimeError(
                                "hooks router recompute produced no router commitments "
                                f"for request {request_id}; available mlp keys={mlp_keys[:8]}"
                            )

                _attach_proof_domain_router_topk(miner, activations, session_router_commitments)
                if tracker is not None:
                    tracker.unregister_request(request_id)

                _resolved = _resolve_sampling_params(body, state.model_name)
                resolved_pp = _resolved["presence_penalty"]
                _pending = getattr(miner, "_pending_sampling_seeds", {}) or {}
                _seed_for_commit = _pending.pop(session_id, b"")
                commitment, commitment_ms = _build_commitment(
                    miner, activations, input_token_ids, output_token_ids,
                    session_id, inference_ms,
                    router_commitments=session_router_commitments,
                    do_sample=body.do_sample,
                    temperature=body.temperature,
                    sampling_verification_bps=body.sampling_verification_bps,
                    presence_penalty=resolved_pp,
                    prompt_hash=prompt_hash,
                    top_k=int(_resolved.get("top_k", -1) or -1),
                    top_p=float(_resolved.get("top_p", 1.0) or 1.0),
                    min_p=float(_resolved.get("min_p", 0.0) or 0.0),
                    sampling_seed=_seed_for_commit,
                )
                if session_router_commitments:
                    miner.router_commitments[session_id] = session_router_commitments
                if session_router_logits:
                    miner.router_logits[session_id] = session_router_logits
                miner.input_token_ids[session_id] = list(input_token_ids)

                t0 = time.perf_counter()
                beacon = derive_beacon_from_nonce(
                    commitment_hash=commitment.commitment_hash(),
                    validator_nonce=nonce,
                )
                beacon_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                if state.moe_config is not None:
                    challenges = derive_moe_challenges(
                        beacon=beacon,
                        commitment=commitment,
                        moe_config=state.moe_config,
                        router_commitments=session_router_commitments,
                        k_layers=state.config.k_layers,
                        k_tokens_per_layer=state.config.k_tokens_per_expert,
                        k_experts_per_layer=state.config.k_experts_per_layer,
                    )
                else:
                    challenges = derive_challenges(
                        beacon=beacon,
                        commitment=commitment,
                        k_layers=state.config.k_layers,
                        k_gemms_per_layer=2,
                        k_blocks_per_gemm=state.config.k_blocks,
                    )
                challenges.sampling_challenge = derive_sampling_challenge(
                    beacon=beacon,
                    commitment=commitment,
                    vocab_size=int(getattr(miner.model_spec, "vocab_size", 0) or 0),
                )
                emb_root = getattr(miner.model_spec, "embedding_weight_merkle_root", b"")
                if emb_root:
                    challenges.embedding_challenge = derive_embedding_challenge(
                        beacon=beacon,
                        commitment=commitment,
                        num_input_tokens=len(input_token_ids),
                    )
                challenge_ms = (time.perf_counter() - t0) * 1000
                return commitment, commitment_ms, beacon_ms, challenges, challenge_ms

            loop = asyncio.get_event_loop()
            commitment, commitment_ms, beacon_ms, challenges, challenge_ms = (
                await loop.run_in_executor(None, _postprocess_sync)
            )

            # Submit proof to background pipeline.

            t0 = time.perf_counter()
            if os.environ.get("VERALLM_SKIP_PROOFS"):
                from verallm.crypto.proof import ProofBundle
                proof_bundle = ProofBundle(layer_proofs=[], sampling_proofs=[])
                timing_details = {}
            else:
                await proof_pipeline.submit_proof(
                    session_id, miner, commitment, challenges, nonce,
                )
                proof_bundle, timing_details, _ = await proof_pipeline.await_proof(session_id, miner=miner)
            prove_ms = (time.perf_counter() - t0) * 1000

            # Clean up session state
            miner.witnesses.pop(session_id, None)
            miner.activation_merkle_trees.pop(session_id, None)
            miner.router_commitments.pop(session_id, None)
            miner.router_logits.pop(session_id, None)
            miner.decode_hidden_row_trees.pop(session_id, None)
            miner.decode_hidden_rows.pop(session_id, None)
            miner.decode_logits_row_trees.pop(session_id, None)
            miner.decode_logits_rows.pop(session_id, None)
            miner.input_token_ids.pop(session_id, None)
            miner.embedding_output_trees.pop(session_id, None)
            miner.output_token_ids.pop(session_id, None)

            # Emit final SSE event
            done_data = {
                "commitment": commitment_to_dict(commitment),
                "proof_bundle": proof_bundle_to_dict(proof_bundle),
                "output_text": prev_text,
                "input_tokens": len(input_token_ids),
                "output_tokens": len(output_token_ids),
                "inference_ms": round(inference_ms, 1),
                "commitment_ms": round(commitment_ms, 1),
                "beacon_ms": round(beacon_ms, 3),
                "challenge_ms": round(challenge_ms, 3),
                "prove_ms": round(prove_ms, 1),
                "prove_timing_details": timing_details,
            }
            _n_out = len(output_token_ids)
            _tps = _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
            bt.logging.info(f"Served {request_id} | {len(input_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms")
            if validator_hotkey:
                bt.logging.debug(f"  └─ validator: {validator_hotkey}")
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            bt.logging.error(f"Post-inference error for {request_id}: {e}\n{tb}")
            # Always clean up captured session artifacts on failure to avoid
            # memory growth under bursty proof backlogs.
            miner.witnesses.pop(session_id, None)
            miner.activation_merkle_trees.pop(session_id, None)
            miner.router_commitments.pop(session_id, None)
            miner.router_logits.pop(session_id, None)
            miner.decode_hidden_row_trees.pop(session_id, None)
            miner.decode_hidden_rows.pop(session_id, None)
            miner.decode_logits_row_trees.pop(session_id, None)
            miner.decode_logits_rows.pop(session_id, None)
            miner.input_token_ids.pop(session_id, None)
            miner.embedding_output_trees.pop(session_id, None)
            miner.output_token_ids.pop(session_id, None)

            if "Proof pipeline saturated" in str(e):
                err = {
                    "error": "Miner busy: proof queue full",
                    "retry_after_ms": 5000,
                    "details": str(e),
                }
            else:
                err = {"error": f"Post-inference error: {e}"}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
        finally:
            # Ensure tracker state is always cleaned up.
            tracker.unregister_request(request_id)

    finally:
        # Always release token budget, even on error/timeout
        await admission.release(request_id)


# ============================================================================
# Startup: load model, compute roots, build Merkle trees
# ============================================================================

def _preflight_gpu_check(skip: bool = False) -> None:
    """Check GPU is available and not occupied by other processes.

    Inspects ``nvidia-smi`` for processes holding GPU memory.  If foreign
    processes are found, prints their PIDs and command lines so the operator
    can kill them, then aborts.  Skipped when ``--skip-gpu-check`` is set.
    """
    if skip:
        return

    if not torch.cuda.is_available():
        bt.logging.error("No CUDA GPU detected. VeraLLM requires a GPU.")
        raise SystemExit(1)

    # Check for other processes using the GPU via nvidia-smi.
    # Filter to GPUs actually visible to THIS process (CUDA_VISIBLE_DEVICES)
    # so a miner on GPU 1 doesn't trip on a sibling miner running on GPU 0.
    import subprocess
    my_pid = os.getpid()

    def _norm_uuid(u: str) -> str:
        s = u.strip().lower()
        return s[4:] if s.startswith("gpu-") else s

    my_gpu_uuids: set[str] = set()
    try:
        for i in range(torch.cuda.device_count()):
            try:
                my_gpu_uuids.add(_norm_uuid(str(torch.cuda.get_device_properties(i).uuid)))
            except Exception:
                pass
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            # nvidia-smi failed — skip check, don't block startup
            return

        foreign = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",", 3)]
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            gpu_uuid, mem_mb, name = parts[1], parts[2], parts[3]
            if pid == my_pid:
                continue
            # Skip processes on GPUs not visible to this miner.
            if my_gpu_uuids and _norm_uuid(gpu_uuid) not in my_gpu_uuids:
                continue
            # Ignore small consumers like Xorg (display server, typically <50 MB)
            try:
                if int(mem_mb) < 50:
                    continue
            except ValueError:
                pass
            foreign.append((pid, mem_mb, name))

        if foreign:
            pids = " ".join(str(p) for p, _, _ in foreign)
            proc_lines = "\n".join(
                "  PID %7d  %6s MiB  %s" % (pid, mem_mb, name)
                for pid, mem_mb, name in foreign
            )
            bt.logging.error(
                f"GPU is in use by other processes!\n{proc_lines}\n"
                f"Kill them before starting VeraLLM:\n  kill {pids}\n"
                f"Or use --skip-gpu-check to bypass this check."
            )
            raise SystemExit(1)

    except FileNotFoundError:
        # nvidia-smi not found — skip check
        pass
    except subprocess.TimeoutExpired:
        pass

    # Report free VRAM
    try:
        free_mem = torch.cuda.mem_get_info(0)  # (free, total)
        free_gb = free_mem[0] / (1024 ** 3)
        total_gb = free_mem[1] / (1024 ** 3)
        used_gb = total_gb - free_gb
        if used_gb > 4.0:
            bt.logging.warning(
                f"GPU has {used_gb:.1f} GB already in use ({free_gb:.1f} / {total_gb:.1f} GB free). "
                f"This may cause OOM during model loading."
            )
    except Exception:
        pass


def _init_tee(args, model_spec: ModelSpec):
    """Initialize TEE: generate keypair, produce attestation, set state."""
    from verallm.tee.crypto import generate_keypair
    from verallm.tee.attestation import get_attestation_provider

    platform = getattr(args, 'tee_platform', 'mock')
    skip_proofs = getattr(args, 'tee_skip_proofs', None)
    if skip_proofs is None:
        # Default: skip proofs when TEE is enabled (hardware attestation replaces them)
        skip_proofs = True

    bt.logging.info(f"TEE mode: {platform}")
    bt.logging.info(f"Proof mode: {'attestation (proofs disabled)' if skip_proofs else 'verallm (proofs enabled)'}")

    # Generate enclave keypair
    private_key, public_key = generate_keypair()
    bt.logging.info(f"Enclave public key: {public_key.hex()[:16]}...")

    # Compute weight file hash (flat SHA256 of safetensors) for TEE binding
    weight_file_hash = b""
    try:
        from verallm.tee.weight_hash import compute_weight_file_hash
        weight_file_hash = compute_weight_file_hash(model_spec.model_id)
        bt.logging.info(f"TEE weight_file_hash: {weight_file_hash.hex()[:16]}...")
    except Exception as e:
        bt.logging.warning(f"TEE: could not compute weight_file_hash ({e}), using empty")

    # Generate attestation binding the public key + model identity
    provider = get_attestation_provider(platform)
    attestation = provider.generate_attestation(public_key, weight_file_hash)
    bt.logging.info(f"Attestation generated (platform={attestation.platform})")

    # Store in state
    state.tee_enabled = True
    state.tee_platform = platform
    state.tee_skip_proofs = skip_proofs
    state.tee_private_key = private_key
    state.tee_public_key = public_key
    state.tee_attestation = attestation
    state.tee_weight_file_hash = weight_file_hash

    bt.logging.info(f"Weight Merkle root: {model_spec.weight_merkle_root.hex()[:16]}...")
    bt.logging.info("TEE ready -- /tee/info, /tee/chat, /tee/reattest endpoints active")


def _resolve_model_gpu_uuids(state) -> list[str]:
    """Return UUIDs of the GPU(s) the loaded model actually resides on.

    Walks ``model.parameters()`` to find the unique CUDA device indices the
    weights are loaded on, then maps each index back to its NVIDIA UUID via
    ``torch.cuda.get_device_properties(idx).uuid``.

    For tensor-parallel models, returns the UUID of every device in the TP
    group. For single-GPU loads (the common case), returns a single UUID.

    Falls back to enumerating all visible CUDA devices if vLLM internals
    can't be accessed (e.g. version drift) or the model isn't loaded yet.
    The caller is responsible for handling the fallback list — this
    function never raises.
    """
    try:
        miner = state.miner
        if miner is None or miner.llm is None:
            raise AttributeError("miner.llm not available")
        engine = getattr(miner.llm, "llm_engine", None)
        if engine is None:
            raise AttributeError("llm_engine not present")
        executor = getattr(engine, "model_executor", None)
        worker = getattr(executor, "driver_worker", None) if executor else None
        runner = getattr(worker, "model_runner", None) if worker else None
        model = getattr(runner, "model", None) if runner else None
        if model is None:
            raise AttributeError(
                "vLLM model not accessible — internal API may have changed"
            )
        indices = sorted({
            p.device.index
            for p in model.parameters()
            if p.is_cuda and p.device.index is not None
        })
        if indices:
            return [str(torch.cuda.get_device_properties(i).uuid) for i in indices]
        raise RuntimeError("model has no CUDA parameters")
    except Exception as e:
        bt.logging.warning(
            f"Could not introspect model GPU device(s); falling back to "
            f"all-visible enumeration: {type(e).__name__}: {e}"
        )
    return [
        str(torch.cuda.get_device_properties(i).uuid)
        for i in range(torch.cuda.device_count())
    ]


def startup(args):
    """Initialize the miner: load model, compute roots, build trees.

    NOTE: In production, the miner fetches the on-chain ModelSpec and
    compares it against its locally computed roots as a self-diagnostic.
    If the roots don't match (wrong model version, corrupt download,
    quantization mismatch), the miner can abort early rather than serve
    proofs that will inevitably fail verification.  The on-chain registry
    and comparison logic are a TODO — currently the miner computes roots
    and serves them directly via GET /model_spec.
    """
    _preflight_gpu_check(skip=getattr(args, 'skip_gpu_check', False))

    # Check CUDA extension is available — CPU fallback is 10-50x slower
    from zkllm.crypto.merkle import _HAS_CUDA_BLAKE3
    if not _HAS_CUDA_BLAKE3:
        msg = (
            "\n"
            "ERROR: CUDA extension not available — blake3 GPU kernel missing.\n"
            "  Merkle tree computation will be 10-50x slower on CPU fallback.\n"
            "  Build the extension: cd zkllm/cuda && python build.py\n"
            "  To force CPU-only mode: set VERATHOS_ALLOW_CPU_FALLBACK=1\n"
        )
        if os.environ.get("VERATHOS_ALLOW_CPU_FALLBACK") != "1":
            bt.logging.error(f"{msg}")
            sys.exit(1)
        else:
            bt.logging.warning("CUDA extension missing -- using slow CPU fallback (VERATHOS_ALLOW_CPU_FALLBACK=1)")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    quant = args.quant

    # Resolve model: either from registry or raw checkpoint
    if args.model_id:
        model_name, quant, registry_max_model_len = _resolve_model_from_registry(
            args.model_id, quant, args.max_model_len,
        )
        if args.max_model_len is None:
            args.max_model_len = registry_max_model_len
    else:
        model_name = args.model

    state.model_name = model_name

    is_gptq = "gptq" in model_name.lower() or "int4" in model_name.lower()
    is_awq = "awq" in model_name.lower()
    is_fp8 = "fp8" in model_name.lower()
    if quant == "auto":
        if is_gptq or is_awq:
            quant = "int4"
        elif is_fp8:
            quant = "fp8"
        else:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                quant = "fp16" if vram_gb >= 16 else "int8"
            except Exception:
                quant = "fp16"

    _config_overrides: dict = dict(
        block_size=256,
        k_layers=args.k_layers or 0,
        k_experts_per_layer=args.k_experts or 0,
        k_tokens_per_expert=args.k_tokens,
        proof_matmul_backend=args.proof_matmul_backend,
        proof_gpu_matmul_limit=args.proof_gpu_matmul_limit,
    )
    if args.spot_checks is not None:
        _config_overrides["spot_checks"] = args.spot_checks
    if args.k_blocks is not None:
        _config_overrides["k_blocks"] = args.k_blocks
    if args.target_detection is not None:
        _config_overrides["target_detection"] = args.target_detection
    config = Config(**_config_overrides)
    set_config(config)
    state.config = config

    # EVM identity (for anti-hijacking: receipt validation + identity challenge)
    if getattr(args, "evm_address", None):
        state.evm_address = args.evm_address
    if getattr(args, "evm_private_key", None):
        state.evm_private_key = args.evm_private_key

    # Wire proof matmul semaphore limit
    if args.proof_gpu_matmul_limit > 0:
        from verallm.miner.matmul import set_gpu_matmul_limit
        set_gpu_matmul_limit(args.proof_gpu_matmul_limit)
    # When 0 (auto), matmul module auto-detects from SM count at import time.

    # Auto-detect attention backend based on GPU compute capability.
    # Blackwell (sm_100+) lacks flash_attn PTX — use TRITON_ATTN instead.
    # B200/GB200 = sm_100 (cc 10.0), RTX 5090 = sm_120 (cc 12.0).
    attention_backend = args.attention_backend
    if attention_backend is None:
        try:
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 10:
                attention_backend = "TRITON_ATTN"
        except Exception:
            pass

    # ── Startup banner ──
    from verallm.log import print_server_banner
    gpu_name = ""
    vram_gb = 0.0
    sm = ""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = torch.cuda.get_device_capability(0)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024**3)
        sm = f"sm_{cc[0]}{cc[1]}0"
        # Populate hardware metadata in state for /health endpoint
        from verallm.registry.gpu import detect_vram_gb
        state.gpu_name = gpu_name
        state.gpu_count = torch.cuda.device_count()
        try:
            state.vram_gb = detect_vram_gb()
        except Exception:
            state.vram_gb = round(vram_gb)
        state.compute_capability = f"{cc[0]}.{cc[1]}"
        # Collect GPU UUIDs for all visible devices (sybil detection)
        state.gpu_uuids = []
        for i in range(torch.cuda.device_count()):
            try:
                state.gpu_uuids.append(str(torch.cuda.get_device_properties(i).uuid))
            except Exception:
                pass
    tee = ""
    if getattr(args, 'tee_enabled', False):
        tee = f"{args.tee_platform} (proofs {'disabled' if getattr(args, 'tee_skip_proofs', True) else 'enabled'})"
    print_server_banner(
        model=model_name,
        quant=quant,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        sm=sm,
        attention=attention_backend or "",
        tee=tee,
        batch_mode=getattr(args, 'batch_mode', True) and not getattr(args, 'no_batch_mode', False),
        port=args.port,
    )

    bt.logging.info("Phase 0: Loading model and computing weight roots...")

    temp_spec = ModelSpec(
        model_id=model_name,
        weight_merkle_root=b"\x00" * 32,
        num_layers=0, hidden_dim=0, num_heads=0, head_dim=0,
        intermediate_dim=0, vocab_size=0,
        activation="silu", norm_type="rmsnorm", attention_type="gqa",
    )
    miner = VllmMiner(model_name, temp_spec, config)

    vllm_kwargs = {}
    if args.max_model_len:
        vllm_kwargs["max_model_len"] = args.max_model_len
    if attention_backend:
        vllm_kwargs["attention_config"] = {"backend": attention_backend}
    miner.setup_vllm(
        quant=quant,
        gpu_memory_utilization=args.gpu_memory_utilization,
        is_gptq=is_gptq,
        is_awq=is_awq,
        **vllm_kwargs,
    )

    # NOTE: Triton scratch-memory allocator is set per-thread in
    # batch_engine.step_and_distribute().  Triton's _allocator is a ContextVar,
    # so setting it here (main thread) does NOT propagate to executor threads.

    moe_config = None
    if is_moe_model(miner.model):
        moe_config = get_moe_config(miner.model)
        miner.moe_config = moe_config
        miner.is_moe = True
        bt.logging.info(f"Detected MoE: {moe_config.num_routed_experts} experts, top-{moe_config.top_k}")
    state.moe_config = moe_config

    detected_quant = detect_quantization(miner.model)
    detected_mode = detected_quant.quant_mode

    # TEE-only mode: skip Merkle root computation entirely — attestation
    # replaces proofs, so weight trees are not needed.  We still need a
    # minimal ModelSpec for the model_id and architecture fields.
    _tee_only = getattr(args, 'tee_enabled', False) and getattr(args, 'tee_skip_proofs', None) is not False

    if _tee_only:
        bt.logging.info("TEE mode: skipping Merkle root computation (attestation replaces proofs)")
        model_spec = _build_minimal_model_spec(miner.model, model_name, detected_mode)
        # Still need on-chain spec for model_id, vocab_size, etc.
        if hasattr(args, 'chain_config') and args.chain_config:
            try:
                from verallm.chain.config import ChainConfig
                from verallm.chain.model_registry import ModelRegistryClient
                chain_config = ChainConfig.from_json(args.chain_config)
                chain_client = ModelRegistryClient(chain_config)
                chain_spec = chain_client.get_model_spec(model_name)
                if chain_spec:
                    model_spec = chain_spec
                    bt.logging.info(f"TEE mode: using on-chain ModelSpec for {model_name}")
            except Exception as e:
                bt.logging.warning(f"TEE mode: could not fetch chain ModelSpec: {e}")
    else:
        # NOTE: In production, ModelSpec comes from on-chain registry.
        # Miner computes its own roots and compares as a self-check.
        model_spec = None if args.no_cache else load_cached_model_spec(
            model_name, config.w_merkle_chunk_size, detected_mode)
        if model_spec is None:
            bt.logging.info(
                "Computing weight Merkle roots (no cache found, may take a few minutes). "
                "Cache at .model_root_cache/ is reusable — copy it to skip this on other instances."
            )
            model_spec = compute_model_roots(
                miner.model, model_name, chunk_size=config.w_merkle_chunk_size,
            )
            save_model_spec_to_cache(model_spec, config.w_merkle_chunk_size, detected_mode)
        else:
            bt.logging.info("Using cached ModelSpec (from .model_root_cache/)")

        # Populate tokenizer_hash from the tokenizer.  Computed fresh on
        # every startup so it always reflects the local tokenizer state —
        # the validator will compare its own local hash to the on-chain
        # anchor at epoch start to detect drift.
        try:
            from verallm.registry.tokenizer_hash import compute_tokenizer_hash
            model_spec.tokenizer_hash = compute_tokenizer_hash(model_name)
            bt.logging.info(
                f"Tokenizer hash: {model_spec.tokenizer_hash[:8].hex()}..."
            )
        except Exception as e:
            bt.logging.warning(f"Could not compute tokenizer_hash: {e}")

        # On-chain self-check: compare local roots against chain registry
        if hasattr(args, 'chain_config') and args.chain_config:
            _chain_self_check(args, model_spec)

    miner.model_spec = model_spec
    miner.model_commitment = model_spec.weight_merkle_root
    state.model_spec = model_spec

    if config.k_layers == 0 and model_spec.num_layers > 0:
        config.k_layers = compute_auto_k(model_spec.num_layers)
        bt.logging.info(f"Auto k_layers: {config.k_layers}/{model_spec.num_layers}")

    if moe_config and config.k_experts_per_layer == 0:
        config.k_experts_per_layer = compute_auto_k_experts(moe_config.num_routed_experts)
        bt.logging.info(f"Auto k_experts: {config.k_experts_per_layer}/{moe_config.num_routed_experts}")

    if not _tee_only:
        bt.logging.info("Precomputing weight Merkle trees...")
        tree_ms = miner.precompute_weight_merkles()
        bt.logging.info(f"Merkle trees ready ({tree_ms:.0f}ms)")
    else:
        bt.logging.info("TEE mode: skipping weight Merkle tree precomputation")

    gc.collect()
    torch.cuda.empty_cache()

    state.miner = miner

    # Refresh gpu_uuids to reflect ONLY the device(s) the model actually
    # loaded onto. The earlier startup-time population enumerated every
    # visible CUDA device, which over-reports on multi-GPU hosts that
    # didn't isolate via CUDA_VISIBLE_DEVICES.
    try:
        _resolved = _resolve_model_gpu_uuids(state)
        if _resolved:
            if state.gpu_uuids != _resolved:
                bt.logging.info(
                    f"GPU UUIDs refined to model-resident set: "
                    f"{[u[:8] + '...' for u in _resolved]} "
                    f"(was {[u[:8] + '...' for u in state.gpu_uuids]})"
                )
            state.gpu_uuids = _resolved
            state.gpu_count = len(_resolved)
    except Exception as e:
        bt.logging.warning(
            f"GPU UUID post-load refresh failed; keeping startup values: {e}"
        )

    # ── TEE setup (confidential GPU mode) ───────────────────────────
    if getattr(args, 'tee_enabled', False):
        _init_tee(args, model_spec)

    # ── Batch mode setup ────────────────────────────────────────────
    if getattr(args, 'batch_mode', True) and not getattr(args, 'no_batch_mode', False):
        state.batch_mode = True

        # Create batch engine
        state.batch_engine = BatchAwareEngine(miner.llm)

        # When TEE skip_proofs is active, skip all activation capture and proof
        # pipeline setup. Hardware attestation replaces VeraLLM proofs entirely.
        _tee_skip = state.tee_skip_proofs

        if _tee_skip:
            bt.logging.info("TEE mode: skipping activation capture and proof pipeline (attestation replaces proofs)")

        # Determine capture backend: splitting_ops (CUDA graphs) or hooks (eager)
        layers = miner._get_layers() if not _tee_skip else []
        use_cuda_graphs = getattr(miner, '_use_cuda_graphs', False)
        _skip_capture = os.environ.get("VERALLM_SKIP_CAPTURE", "0") == "1"

        if _tee_skip:
            pass  # Skip all capture/proof setup below
        elif _skip_capture:
            # Profiling mode: no activation capture, no hooks, no buffers.
            # The CUDA graph runs identically to raw vLLM (gate_proj not wrapped).
            state.activation_tracker = RequestActivationTracker(
                state.batch_engine.model_runner, backend="splitting_ops"
            )
            # Still need prepare_inputs patch for request tracking (register/unregister).
            state.activation_tracker.install_hooks(
                layers=layers,
                is_moe_layer_fn=is_moe_layer if miner.is_moe else (lambda _: False),
                get_mlp_fn=miner._get_mlp,
                get_gate_proj_fn=miner._get_gate_proj,
            )
            bt.logging.warning("VERALLM_SKIP_CAPTURE: ALL activation capture DISABLED")
        elif use_cuda_graphs:
            # Phase 4 production path: splitting_ops capture with CUDA graphs.
            # This keeps eager disabled and captures per-request activations
            # via verallm::capture custom ops.
            state.activation_tracker = RequestActivationTracker(
                state.batch_engine.model_runner, backend="splitting_ops"
            )
            n_hooks = state.activation_tracker.install_hooks(
                layers=layers,
                is_moe_layer_fn=is_moe_layer if miner.is_moe else (lambda _: False),
                get_mlp_fn=miner._get_mlp,
                get_gate_proj_fn=miner._get_gate_proj,
            )

            # Attach capture ops:
            # - MoE: set _layer_idx on FusedMoE/CaptureFusedMoE modules
            # - Dense: wrap gate_proj with CaptureLinearWrapper
            from verallm.vllm_plugin.capture_linear import attach_capture_ops
            # Dense buffer-mode models don't need lm_head wrapping —
            # that would force piecewise CUDA graphs via verallm::capture.
            # NOTE: has_capture_buffers is populated LATER, so detect mode
            # directly from already-wrapped gate_proj modules here.
            _use_buffer = False
            if not miner.is_moe and miner._use_cuda_graphs:
                from verallm.vllm_plugin.capture_linear import CaptureLinearWrapper
                for layer in layers:
                    mlp = miner._get_mlp(layer)
                    if mlp is None:
                        continue
                    gate = miner._get_gate_proj(mlp)
                    if isinstance(gate, CaptureLinearWrapper):
                        _use_buffer = bool(getattr(gate, "_use_buffer", False))
                        break
            elif miner.is_moe and getattr(miner, '_moe_buffer_mode', False):
                _use_buffer = True
            n_instrumented = attach_capture_ops(
                model=miner.model,
                layers=layers,
                is_moe=miner.is_moe,
                get_mlp_fn=miner._get_mlp,
                get_gate_proj_fn=miner._get_gate_proj,
                is_moe_layer_fn=is_moe_layer if miner.is_moe else (lambda _: False),
                wrap_lm_head=not _use_buffer,
            )

            # Also instrument runtime model if it's a different wrapper instance.
            n_runtime_instrumented = 0
            runtime_model = getattr(state.batch_engine.model_runner, "model", None)
            if runtime_model is not None and hasattr(runtime_model, "unwrap"):
                try:
                    runtime_model = runtime_model.unwrap()
                except Exception:
                    pass
            if runtime_model is not None and runtime_model is not miner.model:
                try:
                    from verallm.miner.vllm_utils import _find_layers as _find_runtime_layers
                    runtime_layers = _find_runtime_layers(runtime_model)
                    n_runtime_instrumented = attach_capture_ops(
                        model=runtime_model,
                        layers=runtime_layers,
                        is_moe=miner.is_moe,
                        get_mlp_fn=miner._get_mlp,
                        get_gate_proj_fn=miner._get_gate_proj,
                        is_moe_layer_fn=is_moe_layer if miner.is_moe else (lambda _: False),
                        wrap_lm_head=not _use_buffer,
                    )
                except Exception as e:
                    bt.logging.warning(
                        f"Failed to instrument runtime batch model for capture ops: {e}"
                    )

            # Wire active tracker for verallm::capture custom op.
            from verallm.vllm_plugin.ops import set_active_tracker
            set_active_tracker(state.activation_tracker)

            bt.logging.info(
                f"Batch mode: splitting_ops backend "
                f"({n_instrumented} layers instrumented, {n_hooks} forward hooks, runtime instrumented={n_runtime_instrumented})"
            )

            # splitting_ops path for MoE: router logits are captured by custom op,
            # then converted to RouterDecision at request finalization.
            if miner.is_moe and miner.model is not None:
                state.moe_hook_mgr = BatchMoEHookManager(
                    miner.model,
                    state.activation_tracker,
                    router_top_k=getattr(miner.model_spec, "router_top_k", 0),
                    router_scoring=getattr(miner.model_spec, "router_scoring", "softmax"),
                )
                state.moe_hook_mgr._challenged_layers = [
                    i for i, layer in enumerate(layers) if is_moe_layer(layer)
                ]
                bt.logging.info(
                    f"Batch mode: MoE router capture via splitting_ops ({len(state.moe_hook_mgr.get_challenged_layers())} challenged layers)"
                )
        else:
            # Hook-based capture (enforce_eager=True path)
            state.activation_tracker = RequestActivationTracker(state.batch_engine.model_runner)
            n_hooks = state.activation_tracker.install_hooks(
                layers=layers,
                is_moe_layer_fn=is_moe_layer if miner.is_moe else lambda _: False,
                get_mlp_fn=miner._get_mlp,
                get_gate_proj_fn=miner._get_gate_proj,
            )
            bt.logging.info(f"Batch mode: installed {n_hooks} persistent activation hooks")

            # Create batch MoE hook manager (persistent)
            if miner.is_moe and miner.model is not None:
                state.moe_hook_mgr = BatchMoEHookManager(
                    miner.model,
                    state.activation_tracker,
                    router_top_k=getattr(miner.model_spec, "router_top_k", 0),
                    router_scoring=getattr(miner.model_spec, "router_scoring", "softmax"),
                )
                state.moe_hook_mgr.install_hooks()
                bt.logging.info(
                    f"Batch mode: installed MoE router hooks for {len(state.moe_hook_mgr.get_challenged_layers())} layers"
                )

        if not _tee_skip:
            # Register MoE buffer-mode capture layers with the activation tracker.
            _moe_capture_buffers = getattr(miner, '_moe_capture_buffers', [])
            if miner.is_moe and _moe_capture_buffers:
                state.activation_tracker.register_capture_buffers(_moe_capture_buffers)
                bt.logging.info(
                    f"Batch mode: MoE buffer-mode capture ({len(_moe_capture_buffers)} layers registered)"
                )

            # Pre-extract MoE router weights to CPU for zero-GPU recomputation.
            if getattr(state, 'moe_hook_mgr', None) is not None:
                n_prewarmed = state.moe_hook_mgr.prewarm_router_weights()
                if n_prewarmed:
                    bt.logging.info(f"Batch mode: pre-warmed {n_prewarmed} router weights to CPU")

            # Register buffer-mode capture layers for dense models.
            if not miner.is_moe and not _skip_capture:
                from verallm.vllm_plugin.capture_linear import CaptureLinearWrapper
                buf_wrappers = []
                for layer in layers:
                    mlp = miner._get_mlp(layer)
                    if mlp is None:
                        continue
                    gate = miner._get_gate_proj(mlp)
                    if isinstance(gate, CaptureLinearWrapper) and gate._use_buffer:
                        buf_wrappers.append(gate)
                if buf_wrappers:
                    state.activation_tracker.register_capture_buffers(buf_wrappers)
                    bt.logging.info(f"Batch mode: registered {len(buf_wrappers)} buffer-mode capture layers")

            # Install lm_head hook for decode-integrity capture.
            if _skip_capture or os.environ.get("VERALLM_SKIP_LM_HEAD_HOOK"):
                if not _skip_capture:
                    bt.logging.warning("VERALLM_SKIP_LM_HEAD_HOOK: lm_head decode-integrity hook DISABLED")
            elif miner.model is not None and hasattr(miner.model, "compute_logits"):
                state.activation_tracker.install_lm_head_hook(miner.model, capture_logits=False)

            # Embedding output capture DISABLED — per-request Merkle
            # tree is too expensive for large contexts.  See comment in
            # verallm/api/client.py verify_proof() for full rationale.
            # The embedding hook and install_embedding_hook() are preserved
            # in activation_tracker.py for future re-enablement.
            # from verallm.introspection import get_embedding_module as _get_emb
            # _emb_mod = _get_emb(miner.model)
            # if _emb_mod is not None:
            #     state.activation_tracker.install_embedding_hook(_emb_mod)

            # Create proof pipeline.
            proof_threads = getattr(args, 'proof_threads', None) or auto_detect_proof_concurrency()
            proof_max_pending_override = getattr(args, 'proof_max_pending', None)
            state.proof_pipeline = ProofPipeline(
                max_concurrent_proofs=proof_threads,
                max_pending=proof_max_pending_override,
            )

            # Initialize batched proof matmul service when backend is "batched".
            _matmul_backend = getattr(args, 'proof_matmul_backend', 'batched')
            if _matmul_backend == "batched":
                from verallm.miner.matmul import init_proof_matmul_batcher
                init_proof_matmul_batcher()

        # Dynamic token-budget admission control (needed in both proof and TEE mode)
        # vLLM >=0.16.1 moved cache_config under vllm_config
        engine = miner.llm.llm_engine
        cache_config = getattr(engine, 'cache_config', None)
        if cache_config is None:
            cache_config = engine.vllm_config.cache_config
        total_kv_tokens = (cache_config.num_gpu_blocks or 0) * (cache_config.block_size or 16)
        # Read the actual fitted max_model_len from vLLM (handles auto-fit)
        max_context = miner.llm.llm_engine.model_config.max_model_len
        max_requests = getattr(args, 'max_concurrent', None) or auto_detect_max_requests_with_ram(
            hidden_dim=model_spec.hidden_dim,
            intermediate_dim=model_spec.intermediate_dim,
        )

        estimated_mb = estimate_per_request_ram_mb(
            model_spec.hidden_dim, model_spec.intermediate_dim,
        )
        state.admission = TokenBudgetAdmission(
            total_kv_tokens=total_kv_tokens,
            max_context=max_context,
            max_requests=max_requests,
            ram_headroom_gb=2.0,
        )

        if not _tee_skip and state.proof_pipeline is not None:
            # Now that max_requests is known, set proof pipeline pending cap.
            if proof_max_pending_override is None:
                state.proof_pipeline._max_pending = max_requests

            # Auto-tune admission after first proof using measured RSS delta.
            def _on_rss_measured(measured_mb: int) -> None:
                if measured_mb <= 0:
                    return
                safe_mb = max(measured_mb * 2, 50)
                new_max = auto_detect_max_requests_with_ram(
                    hidden_dim=model_spec.hidden_dim,
                    intermediate_dim=model_spec.intermediate_dim,
                    per_request_ram_mb=safe_mb,
                )
                current = state.admission.max_requests
                if new_max != current:
                    direction = "increasing" if new_max > current else "reducing"
                    bt.logging.debug(
                        f"Admission: {direction} max_requests from measured RSS: "
                        f"estimated={estimated_mb} MB, measured={measured_mb} MB, safe={safe_mb} MB, max_requests {current} -> {new_max}"
                    )
                    state.admission.update_max_requests(new_max)
                    state.proof_pipeline._max_pending = max(
                        state.proof_pipeline._max_pending, new_max)
                else:
                    bt.logging.debug(
                        f"Admission: measured RSS ({measured_mb} MB, safe={safe_mb} MB) confirms max_requests={current}"
                    )

            state.proof_pipeline.on_rss_measured = _on_rss_measured

        if _tee_skip:
            bt.logging.info(
                f"Batch mode (TEE): KV pool={total_kv_tokens} tokens, max_context={max_context}, max_requests={max_requests}"
            )
        else:
            bt.logging.info(
                f"Batch mode: KV pool={total_kv_tokens} tokens, max_context={max_context}, max_requests={max_requests}, "
                f"proof_threads={state.proof_pipeline.max_concurrent}, proof_max_pending={state.proof_pipeline.max_pending}"
            )

    # VRAM headroom guard: if free VRAM is below 1 GB after model + KV cache
    # + CUDA graphs, GPU proof matmul will contend severely with inference
    # (especially at moderate batch sizes where adaptive threshold allows GPU).
    # Proactively disable to avoid the B8-type slowdown where GPU proof matmul
    # at the threshold boundary causes 50%+ overhead.
    _VRAM_HEADROOM_MIN_MB = 1024  # 1 GB
    try:
        free_vram = torch.cuda.mem_get_info()[0]
        free_mb = free_vram / 1e6
        if free_mb < _VRAM_HEADROOM_MIN_MB:
            from verallm.miner.matmul import disable_gpu_proof_matmul
            disable_gpu_proof_matmul(
                f"VRAM headroom too low ({free_mb:.0f} MB < {_VRAM_HEADROOM_MIN_MB} MB)"
            )
        else:
            bt.logging.info(f"VRAM headroom at startup: {free_mb:.0f} MB free")
    except Exception:
        pass

    # Warmup: trigger Triton JIT + torch.compile for all kernel variants.
    # Without this, first user requests pay 5-24s compilation penalty.
    #
    # FLA/GDN attention (Qwen3.5, etc.) has TWO code paths:
    #   - Recurrent mode: used for short sequences / decode (seq_len < ~64)
    #   - Chunk mode: used for long prefills (seq_len >= ~64)
    # Both paths use different Triton kernels that must be JIT-compiled.
    # A short warmup only compiles recurrent mode; the first long-prompt
    # request then pays ~5s for chunk-mode Triton compilation.
    #
    # Solution: warmup with BOTH a short prompt (recurrent + decode kernels)
    # AND a long prompt (chunk-mode prefill kernels).
    bt.logging.info("Running warmup inference...")
    t_warmup = time.perf_counter()
    try:
        from vllm import SamplingParams

        def _apply_template(messages):
            try:
                return miner.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return " ".join(m.get("content", "") for m in messages)

        _warmup_params = SamplingParams(max_tokens=4, temperature=0)

        # 1) Short prompt — compiles recurrent-mode FLA kernels + decode
        _short = _apply_template([{"role": "user", "content": "Hi"}])
        miner.llm.generate([_short], sampling_params=_warmup_params)
        bt.logging.info(f"Warmup short prompt done ({(time.perf_counter() - t_warmup) * 1000:.0f}ms)")

        # 2) Long prompt — compiles chunk-mode FLA kernels for prefill.
        #    ~1000 tokens is enough to trigger chunked prefill path.
        _long_content = "Hello, this is warmup. " * 80  # ~320 tokens raw
        _long_msgs = [
            {"role": "user", "content": "Tell me about AI."},
            {"role": "assistant", "content": _long_content},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": _long_content},
            {"role": "user", "content": "Ok."},
        ]
        _long = _apply_template(_long_msgs)
        miner.llm.generate([_long], sampling_params=_warmup_params)
        bt.logging.info(f"Warmup long prompt done ({(time.perf_counter() - t_warmup) * 1000:.0f}ms)")

    except Exception as e:
        bt.logging.warning(f"Warmup inference failed: {e}")
    bt.logging.info(f"Warmup complete ({(time.perf_counter() - t_warmup) * 1000:.0f}ms)")

    bt.logging.info(f"ModelSpec: {model_spec.num_layers} layers, hidden={model_spec.hidden_dim}")
    bt.logging.info(f"Roots: {len(model_spec.weight_block_merkle_roots)} layer roots")
    if state.tee_enabled:
        bt.logging.info(
            f"Proof pipeline: TEE attestation ({state.tee_platform}) — "
            f"ZK proofs disabled, hardware attestation active"
        )
    elif not _tee_only:
        _cap_backend = getattr(state, "activation_tracker", None)
        _cap_str = getattr(_cap_backend, "backend", "unknown") if _cap_backend else "none"
        bt.logging.success(
            f"Proof pipeline: Cryptographic verification active — "
            f"k={config.k_layers}/{model_spec.num_layers} layers, "
            f"capture={_cap_str}, "
            f"root={model_spec.weight_merkle_root.hex()[:16]}..."
        )
    bt.logging.info(f"Miner ready. Serving on port {args.port}")
    if state.batch_mode:
        s = state.admission.status()
        bt.logging.info(
            f"Mode: BATCH (dynamic admission, {s.total_kv_tokens} KV tokens, max {state.admission.max_requests} requests)"
        )
    else:
        bt.logging.info("Mode: SINGLE (one request at a time)")


# ============================================================================
# CLI
# ============================================================================

def _resolve_model_from_registry(model_id: str, quant: str, max_model_len: int | None):
    """Resolve --model-id to checkpoint + quant + max_model_len from the registry.

    Returns (model_name, quant, max_model_len) with registry defaults
    filled in where the user didn't override.

    Context-length strategy:
      - **Native match** (model has a TierConfig for this exact GPU tier):
        use the registry cap — it's a tested safe value for this VRAM budget.
      - **Inherited match** (model configured for a lower tier, running on
        a bigger GPU): skip the cap and let vLLM auto-size the KV cache.
        The GPU has more VRAM than needed, so the model's full native
        context can usually be used.
      - User ``--max-model-len`` always overrides both.
    """
    from verallm.registry.models import resolve_model_for_tier
    from verallm.registry.gpu import detect_vram_tier

    tier = detect_vram_tier()
    tm = resolve_model_for_tier(model_id, tier)
    model_name = tm.config.checkpoint

    native_str = "native" if tm.native else f"inherited from {tm.config.tier.name}"
    bt.logging.info(f"Registry: {tm.model.name} -> {model_name}")
    bt.logging.info(f"Tier match: {tier.name} ({tier.value} GB) [{native_str}]")

    # Use first quant mode from config if user didn't specify
    if quant == "auto" and tm.config.quant_configs:
        quant = tm.config.quant_configs[0].quant
        bt.logging.info(f"Registry quant: {quant}")

    # Look up the per-quant max_model_len for the selected quant
    selected_qo = next(
        (qo for qo in tm.config.quant_configs if qo.quant == quant),
        tm.config.quant_configs[0] if tm.config.quant_configs else None,
    )

    # Context-length resolution: let vLLM auto-size unless user overrides.
    # vLLM defaults to the model's native context length.  If that exceeds
    # available KV cache memory, setup_vllm() retry logic catches the error
    # and restarts with vLLM's estimated maximum — always giving the TRUE
    # maximum context the GPU can support without wasting VRAM.
    if max_model_len is None:
        bt.logging.info("max_model_len: auto (vLLM will determine from available KV cache)")

    return model_name, quant, max_model_len


def _build_minimal_model_spec(model, model_name, quant_mode):
    """Build a ModelSpec with architecture info but no Merkle roots.

    Used in TEE-only mode where attestation replaces proofs and weight
    trees are not needed.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    return ModelSpec(
        model_id=model_name,
        weight_merkle_root=b"\x00" * 32,
        num_layers=getattr(text_cfg, "num_hidden_layers", 0),
        hidden_dim=getattr(text_cfg, "hidden_size", 0),
        num_heads=getattr(text_cfg, "num_attention_heads", 0),
        head_dim=getattr(text_cfg, "head_dim", 0) or (
            getattr(text_cfg, "hidden_size", 0) // max(getattr(text_cfg, "num_attention_heads", 1), 1)
        ),
        intermediate_dim=getattr(text_cfg, "intermediate_size", 0),
        vocab_size=getattr(text_cfg, "vocab_size", 0) or getattr(cfg, "vocab_size", 0),
        activation="silu",
        norm_type="rmsnorm",
        attention_type="gqa",
        quant_mode=quant_mode,
    )


def _chain_self_check(args, local_spec):
    """Compare locally computed ModelSpec roots against the on-chain registry.

    If the roots don't match, the miner's proofs will inevitably fail
    verification. This catches wrong model versions, corrupt downloads,
    or quantization mismatches at startup rather than at verification time.

    RPC failures (429, timeout, connection errors) are treated as warnings
    rather than fatal — the chain may be temporarily unreachable but the
    model itself is fine. The miner will be verified at canary time anyway.
    """
    from verallm.chain.config import ChainConfig
    from verallm.chain.mock import create_clients

    rpc_override = getattr(args, "evm_rpc_url", None)
    chain_config = ChainConfig.from_json(
        args.chain_config,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )

    # On public RPCs, pause before self-check to let the parent miner
    # process's startup burst (registerEvm, allowlist) clear the rate limit.
    if any(h in chain_config.rpc_url for h in ("opentensor.ai", "finney")):
        time.sleep(5)

    model_client, _, _ = create_clients(chain_config)

    try:
        chain_spec = model_client.get_model_spec(local_spec.model_id)
    except Exception as e:
        bt.logging.warning(
            f"On-chain self-check skipped: RPC error querying ModelRegistry: {e}. "
            f"The miner will continue with local ModelSpec. If the model is not "
            f"registered on-chain, proof verification will fail at canary time."
        )
        return

    if chain_spec is None:
        msg = (
            f"Model '{local_spec.model_id}' is not registered on the on-chain "
            f"ModelRegistry. This miner cannot pass proof verification until "
            f"the subnet owner registers this model. "
            f"Registered models can be queried via the ModelRegistry contract."
        )
        if args.force:
            bt.logging.warning(f"{msg} Continuing anyway (--force).")
            return
        bt.logging.error(f"{msg} Exiting.")
        sys.exit(1)

    # Compare layer roots
    local_roots = local_spec.weight_block_merkle_roots
    chain_roots = chain_spec.weight_block_merkle_roots

    if len(local_roots) != len(chain_roots):
        msg = f"ROOT MISMATCH: local has {len(local_roots)} layer roots, chain has {len(chain_roots)}"
        if args.force:
            bt.logging.warning(f"{msg} Continuing (--force).")
            return
        bt.logging.error(f"{msg} Aborting. Use --force to override.")
        sys.exit(1)

    mismatches = []
    for i, (lr, cr) in enumerate(zip(local_roots, chain_roots)):
        if lr != cr:
            mismatches.append(i)

    if mismatches:
        msg = f"ROOT MISMATCH: {len(mismatches)} layers differ (first: layer {mismatches[0]})"
        if args.force:
            bt.logging.warning(f"{msg} Continuing (--force).")
            return
        bt.logging.error(f"{msg} Aborting. Use --force to override.")
        sys.exit(1)

    bt.logging.info(f"On-chain self-check PASSED: {len(chain_roots)} layer roots match.")

    # Sync architectural fields from chain spec so the miner derives
    # challenges identically to the validator (e.g. vocab_size enters
    # the Fiat-Shamir seed in derive_sampling_challenge).
    mismatches = []
    for field in ("vocab_size", "hidden_dim", "num_heads", "head_dim",
                  "intermediate_dim", "num_layers"):
        chain_val = getattr(chain_spec, field, None)
        local_val = getattr(local_spec, field, None)
        if chain_val and chain_val != local_val:
            bt.logging.warning(
                f"Syncing {field} from chain: local={local_val}, chain={chain_val}"
            )
            mismatches.append(f"{field}: local={local_val}, chain={chain_val}")
            setattr(local_spec, field, chain_val)
    if mismatches:
        bt.logging.error(
            f"ARCHITECTURE MISMATCH — {len(mismatches)} fields differ between "
            f"local model and on-chain registration. This means the model was "
            f"registered with a different environment (e.g. wrong transformers "
            f"version). Fields auto-synced from chain, but the on-chain "
            f"registration should be updated: {', '.join(mismatches)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="VeraLLM Miner Server")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="HuggingFace model name (raw checkpoint)")
    model_group.add_argument(
        "--model-id",
        help="Registry model ID (auto-resolves checkpoint, quant, context length "
             "for detected GPU). Run 'python -m verallm.registry' to list available models.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--quant", default="auto",
                        choices=["auto", "fp16", "fp8", "int8", "int4"],
                        help="Quantization mode")
    parser.add_argument("--spot-checks", type=int, default=None,
                        help="Spot checks per block (default: from config)")
    parser.add_argument("--k-blocks", type=int, default=None,
                        help="Blocks per GEMM to verify (default: from config)")
    parser.add_argument("--k-layers", type=int, default=None,
                        help="Layers to challenge (None = auto)")
    parser.add_argument("--target-detection", type=float, default=None,
                        help="Per-inference detection target for auto-k (default: 0.0625)")
    parser.add_argument("--k-experts", type=int, default=None,
                        help="Experts to challenge per layer (None = auto)")
    parser.add_argument("--k-tokens", type=int, default=4,
                        help="Tokens to sample for expert challenges")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="Fraction of GPU memory for vLLM KV cache (default: 0.80).")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass ModelSpec cache")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS key file")
    parser.add_argument("--ssl-certfile", default=None, help="TLS cert file")
    parser.add_argument("--api-key", default=None,
                        help="API key for auth (or set VERATHOS_API_KEY env var)")
    parser.add_argument("--attention-backend", default=None,
                        help="vLLM attention backend (e.g. TRITON_ATTN, FLASH_ATTN)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print environment diagnostics and exit")
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON (compares local roots against chain)")
    parser.add_argument("--evm-rpc-url", default=None,
                        help="EVM RPC URL (overrides rpc_url in chain config)")
    parser.add_argument("--force", action="store_true",
                        help="Start even if on-chain root comparison fails")
    # Batch mode (continuous batching)
    parser.add_argument("--batch-mode", action="store_true", default=True,
                        help="Enable continuous batching for concurrent requests (default: on)")
    parser.add_argument("--no-batch-mode", action="store_true",
                        help="Disable continuous batching (legacy single-request mode)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max concurrent requests in batch mode (None = auto-detect)")
    parser.add_argument("--proof-threads", type=int, default=None,
                        help="Max concurrent proof threads (None = auto-detect from CPU/VRAM)")
    parser.add_argument("--proof-max-pending", type=int, default=None,
                        help="Max pending proofs (running + queued) before returning 503 (None = 2x proof-threads)")
    parser.add_argument("--proof-matmul-backend", default="batched",
                        choices=["gpu", "cpu", "adaptive", "batched"],
                        help="Matmul backend for proof generation (default: batched). "
                             "'batched' collects matmuls, groups by layer, single-stream dispatch. "
                             "'gpu' uses non-blocking GPU-first with CPU f32 spillover. "
                             "'cpu' forces CPU-only f32 SGEMM. "
                             "'adaptive' is an alias for 'gpu'.")
    parser.add_argument("--proof-gpu-matmul-limit", type=int, default=0,
                        help="Max concurrent GPU matmul allocations (0 = auto from SM count)")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="Skip pre-flight GPU occupancy check")
    # EVM identity (passed by neurons/miner.py for anti-hijacking)
    parser.add_argument("--evm-address", default=None,
                        help="Miner's EVM address (for receipt validation + identity challenge)")
    parser.add_argument("--evm-private-key", default=None,
                        help="Miner's EVM private key hex (for identity challenge signing)")
    # TEE (Trusted Execution Environment) — confidential GPU mode
    parser.add_argument("--tee-enabled", action="store_true",
                        help="Enable TEE mode (E2E encryption + attestation)")
    parser.add_argument("--tee-platform", default="mock",
                        choices=["mock", "tdx", "sev-snp", "gpu"],
                        help="TEE attestation platform (default: mock)")
    parser.add_argument("--tee-skip-proofs", action="store_true", default=None,
                        help="Skip VeraLLM proof generation (use hardware attestation instead). "
                             "Default: True when --tee-enabled is set.")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning"],
                        help="Logging level (default: info)")
    return parser.parse_args()


def _print_diagnostics():
    """Print environment info for debugging remote deployments."""
    import sys
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")

    print("\n--- torch ---")
    try:
        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            props = torch.cuda.get_device_properties(0)
            cc = torch.cuda.get_device_capability(0)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"GPU: {props.name}")
            print(f"VRAM: {vram_gb:.1f} GB")
            print(f"Compute: sm_{cc[0]}{cc[1]}0")
            attn = "TRITON_ATTN" if cc[0] >= 10 else "FLASH_ATTN"
            quant = "fp16" if vram_gb >= 16 else "int8"
            print(f"Auto attention: {attn}")
            print(f"Auto quant: {quant}")
    except Exception as e:
        print(f"torch error: {e}")

    print("\n--- vLLM ---")
    try:
        import vllm
        print(f"vllm: {vllm.__version__}")
    except Exception as e:
        print(f"vllm: FAILED ({e})")

    print("\n--- bitsandbytes ---")
    try:
        import bitsandbytes
        print(f"bitsandbytes: {bitsandbytes.__version__}")
    except Exception as e:
        print(f"bitsandbytes: not installed ({e})")

    print("\n--- CUDA extension ---")
    try:
        print(f"zkllm_native: loaded OK")
    except Exception as e:
        print(f"zkllm_native: {e}")

    print("\n--- NVIDIA libs ---")
    import glob as _glob
    for lib_name in ["libcusparseLt.so", "libnvrtc.so", "libcudnn.so"]:
        found = _glob.glob(f"/usr/local/cuda*/lib64/{lib_name}*")
        try:
            import site
            for sp in site.getsitepackages():
                found += _glob.glob(f"{sp}/nvidia/**/{lib_name}*", recursive=True)
        except Exception:
            pass
        if found:
            print(f"  {lib_name}: {found[0]}")
        else:
            print(f"  {lib_name}: NOT FOUND")


def main():
    args = parse_args()

    # Configure logging before anything else (so import-time logs are captured).
    from verallm.log import setup_server_logging
    setup_server_logging(args.log_level)

    if args.diagnose:
        _print_diagnostics()
        return
    if args.api_key:
        os.environ["VERATHOS_API_KEY"] = args.api_key
    startup(args)

    import uvicorn
    uvicorn_kwargs = dict(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile

    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
