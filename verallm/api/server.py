#!/usr/bin/env python3
"""
VeraLLM Miner Server — FastAPI server wrapping VllmMiner.

Runs a vLLM-powered miner that serves inference requests and generates
cryptographic proofs over a REST API. Legacy v1 derives its Fiat-Shamir
challenges from the initial request. Proof v2 streams the frozen commitment
digest, accepts one authenticated nonce-reveal POST, and returns the proof on
the still-open inference stream.

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
import hmac
import json
import logging
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

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
    _CPU_THREADS_PER_WORKER = max(
        1, int(os.environ.get("VERALLM_CPU_THREADS_PER_WORKER", "1"))
    )
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
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, StrictInt, root_validator, validator

from neurons.version import (
    miner_version_str,
    version_str as spec_version_str,
)
from verallm.config import Config, set_config
from verallm.challenge.beacon import (
    derive_hard_audit_sampling_challenge,
    derive_beacon_from_nonce,
    derive_challenges,
    derive_embedding_challenge,
    derive_sampling_challenge,
    hard_audit_required,
    validate_proof_v2_decode_commitment,
)
from verallm.sampling import (
    clamp_sampling_bps,
    temperature_to_milli,
    build_hidden_row_merkle,
    build_logits_row_merkle,
    HIGH_ASSURANCE_BPS,
)
from verallm.crypto.merkle import MerkleTree
from verallm.helpers import compute_auto_k, compute_auto_k_experts
from verallm.miner import VllmMiner
from verallm.miner.batch_engine import BatchAwareEngine, BatchEngineRequestError
from verallm.miner.activation_tracker import RequestActivationTracker
from verallm.miner.admission import TokenBudgetAdmission
from verallm.miner.memory_budget import (
    auto_detect_max_requests_with_ram,
    auto_detect_proof_concurrency,
    estimate_per_request_ram_mb,
)
from verallm.miner.proof_pipeline import ProofPipeline
from verallm.moe import (
    is_moe_model,
    is_moe_layer,
    get_moe_config,
    derive_moe_challenges,
    MoEHookManager,
    BatchMoEHookManager,
)
from verallm.quantization import detect_quantization
from verallm.registry import (
    compute_model_roots,
    load_cached_model_spec,
    save_model_spec_to_cache,
)
from verallm.types import ChallengeSet, InferenceCommitment, ModelSpec

from verallm.api.serialization import (
    model_spec_to_dict,
    commitment_to_dict,
    proof_bundle_to_dict,
)
from verallm.api.proof_protocol import (
    LEGACY_PROOF_PROTOCOL_VERSION,
    PROOF_PROTOCOL_V2,
    PROOF_PROTOCOL_V3,
    commit_validator_nonce_v2,
    decode_proof_challenge_id,
    decode_proof_commitment_hash,
    decode_validator_nonce,
    encode_proof_challenge_id,
    encode_proof_commitment_hash,
    resolve_proof_protocol_version,
    validate_proof_protocol_version,
    validator_nonce_matches_commitment_v2,
)


def _engine_scheduler_max_num_seqs(engine: object) -> int:
    """Return vLLM's real request-width ceiling across supported layouts."""

    candidates = (
        getattr(engine, "scheduler_config", None),
        getattr(getattr(engine, "vllm_config", None), "scheduler_config", None),
    )
    for scheduler_config in candidates:
        raw = getattr(scheduler_config, "max_num_seqs", None)
        if isinstance(raw, bool):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    raise RuntimeError("vLLM scheduler max_num_seqs is unavailable")


def _bounded_server_request_admission(
    requested_max_requests: int,
    scheduler_max_num_seqs: int,
) -> int:
    """Keep HTTP admission inside the engine/capture concurrency envelope."""

    if (
        isinstance(requested_max_requests, bool)
        or isinstance(scheduler_max_num_seqs, bool)
        or int(requested_max_requests) <= 0
        or int(scheduler_max_num_seqs) <= 0
    ):
        raise ValueError("request admission limits must be positive integers")
    return min(int(requested_max_requests), int(scheduler_max_num_seqs))


# ============================================================================
# Pydantic models for request validation
# ============================================================================


class ProofProtocolRequestBody(BaseModel):
    """Common proof request fields validated at the miner API boundary."""

    validator_nonce: Optional[str] = None
    validator_nonce_commitment: Optional[str] = None
    proof_challenge_id: Optional[str] = None
    proof_v3_preexecution_context: Optional[str] = None
    proof_v3_hard_proof_arrival_budget_ns: Optional[StrictInt] = None
    proof_protocol_version: Optional[StrictInt] = None
    _inline_proof_v2: ClassVar[bool] = True

    @validator("validator_nonce")
    def _validate_validator_nonce(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            decode_validator_nonce(value)
        return value

    @validator("validator_nonce_commitment")
    def _validate_validator_nonce_commitment(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is not None:
            decode_proof_commitment_hash(value)
        return value

    @validator("proof_challenge_id")
    def _validate_proof_challenge_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            decode_proof_challenge_id(value)
        return value

    @validator("proof_v3_preexecution_context")
    def _validate_proof_v3_preexecution_context(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        from verallm.proof_v3.request import (
            PREEXECUTION_CONTEXT_BYTES_V3,
            PreExecutionRequestContextV3,
        )

        if (
            not isinstance(value, str)
            or len(value) != 2 * PREEXECUTION_CONTEXT_BYTES_V3
        ):
            raise ValueError(
                "proof_v3_preexecution_context has an invalid hexadecimal "
                "length"
            )
        try:
            encoded = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(
                "proof_v3_preexecution_context must be hexadecimal"
            ) from exc
        if encoded.hex() != value:
            raise ValueError(
                "proof_v3_preexecution_context must use canonical lowercase "
                "hexadecimal"
            )
        PreExecutionRequestContextV3.from_canonical_bytes(encoded)
        return value

    @validator("proof_protocol_version")
    def _validate_proof_protocol_version(cls, value: Optional[int]) -> Optional[int]:
        return validate_proof_protocol_version(value)

    @validator("proof_v3_hard_proof_arrival_budget_ns")
    def _validate_proof_v3_hard_proof_arrival_budget_ns(
        cls,
        value: Optional[int],
    ) -> Optional[int]:
        if value is None:
            return None
        from verallm.proof_v3.session import (
            MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3,
        )

        if not 0 < value <= MAX_HARD_PROOF_ARRIVAL_BUDGET_NS_V3:
            raise ValueError(
                "proof_v3_hard_proof_arrival_budget_ns is out of range"
            )
        return value

    @root_validator(skip_on_failure=True)
    def _validate_nonce_exchange(cls, values: dict) -> dict:
        version = resolve_proof_protocol_version(values.get("proof_protocol_version"))
        nonce = values.get("validator_nonce")
        nonce_commitment = values.get("validator_nonce_commitment")
        challenge_id = values.get("proof_challenge_id")
        v3_context = values.get("proof_v3_preexecution_context")
        v3_hard_budget = values.get(
            "proof_v3_hard_proof_arrival_budget_ns"
        )
        if version == PROOF_PROTOCOL_V2 and cls._inline_proof_v2:
            if nonce is not None:
                raise ValueError("proof-v2 request must not reveal validator_nonce")
            if nonce_commitment is None or challenge_id is None:
                raise ValueError(
                    "proof-v2 request requires validator_nonce_commitment and "
                    "proof_challenge_id"
                )
            if v3_context is not None:
                raise ValueError(
                    "proof-v2 request must not include proof-v3 context"
                )
            if v3_hard_budget is not None:
                raise ValueError(
                    "proof-v2 request must not include proof-v3 hard budget"
                )
        elif version == PROOF_PROTOCOL_V3:
            if (
                nonce is not None
                or nonce_commitment is not None
                or challenge_id is not None
            ):
                raise ValueError(
                    "proof-v3 request must not include legacy or proof-v2 "
                    "nonce fields"
                )
            if v3_context is None:
                raise ValueError(
                    "proof-v3 request requires proof_v3_preexecution_context"
                )
        else:
            if nonce is None:
                raise ValueError("legacy proof request requires validator_nonce")
            if nonce_commitment is not None or challenge_id is not None:
                raise ValueError(
                    "legacy proof request must not include proof-v2 reveal fields"
                )
            if v3_context is not None:
                raise ValueError(
                    "legacy proof request must not include proof-v3 context"
                )
            if v3_hard_budget is not None:
                raise ValueError(
                    "legacy proof request must not include proof-v3 hard budget"
                )
        return values

    @property
    def validator_nonce_bytes(self) -> bytes:
        if self.validator_nonce is None:
            raise ValueError("validator_nonce has not been revealed")
        return decode_validator_nonce(self.validator_nonce)

    @property
    def validator_nonce_commitment_bytes(self) -> bytes:
        if self.validator_nonce_commitment is None:
            raise ValueError("validator_nonce_commitment is unavailable")
        return decode_proof_commitment_hash(self.validator_nonce_commitment)

    @property
    def proof_challenge_id_bytes(self) -> bytes:
        if self.proof_challenge_id is None:
            raise ValueError("proof_challenge_id is unavailable")
        return decode_proof_challenge_id(self.proof_challenge_id)

    @property
    def proof_v3_preexecution_context_value(self):
        if self.proof_v3_preexecution_context is None:
            raise ValueError("proof-v3 preexecution context is unavailable")
        from verallm.proof_v3.request import PreExecutionRequestContextV3

        return PreExecutionRequestContextV3.from_canonical_bytes(
            bytes.fromhex(self.proof_v3_preexecution_context)
        )

    @property
    def resolved_proof_protocol_version(self) -> int:
        return resolve_proof_protocol_version(self.proof_protocol_version)


class InferenceRequestBody(ProofProtocolRequestBody):
    prompt: str
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    sampling_verification_bps: int = 0
    enable_thinking: bool = True  # chain-of-thought for models that support it
    # Sampling parameters (post-logits processors).  Server applies
    # model-specific defaults when these are None (e.g. Qwen3 presence_penalty).
    presence_penalty: Optional[float] = None  # vLLM default 0.0
    top_k: Optional[int] = None  # vLLM default -1 (disabled)
    top_p: Optional[float] = None  # vLLM default 1.0
    min_p: Optional[float] = None  # vLLM default 0.0


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ChatRequestBody(ProofProtocolRequestBody):
    """Chat-style inference with OpenAI-compatible messages array.

    The miner applies the model's chat template server-side (it already
    has the tokenizer loaded), so clients don't need the tokenizer.
    """

    messages: list[ChatMessage]
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
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None


class ProofV2ChallengeRevealBody(BaseModel):
    """Authenticated validator reveal for one frozen v2 commitment."""

    proof_challenge_id: str
    session_id: str
    commitment_hash: str
    validator_nonce: str

    @validator("proof_challenge_id")
    def _validate_challenge_id(cls, value: str) -> str:
        decode_proof_challenge_id(value)
        return value

    @validator("commitment_hash")
    def _validate_commitment_hash(cls, value: str) -> str:
        decode_proof_commitment_hash(value)
        return value

    @validator("validator_nonce")
    def _validate_revealed_nonce(cls, value: str) -> str:
        decode_validator_nonce(value)
        return value

    @validator("session_id")
    def _validate_session_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 128:
            raise ValueError("session_id must be a non-empty string")
        return value


def _chat_prompt_hash_payload(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    parallel_tool_calls: Optional[bool] = None,
) -> Any:
    """Return the chat prompt-hash payload used for input binding."""
    if tools:
        return {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
    # Preserve the legacy no-tool hash contract for rolling miner/proxy upgrades.
    return messages


# ============================================================================
# App state (populated during startup)
# ============================================================================

def _proof_v3_capture_buffer_is_whole_step(wrapper: object) -> bool:
    """Classify one capture buffer by its own row layout.

    A root-enabled model can mix a whole-step QKV buffer with a compact
    selected-row o-projection buffer.  The presence of the shared gather
    index elsewhere in the tracker therefore says nothing about this
    particular buffer's layout.
    """

    return not isinstance(
        getattr(wrapper, "_capture_row_indices", None),
        torch.Tensor,
    )


def _proof_v3_reduction_wrapper_is_dedicated_buffer(
    wrappers: Mapping[str, object],
) -> bool:
    """Keep nested buffer-mode reduction wrappers out of the base inventory.

    Root-row capture gives a buffer-mode reduction wrapper a shared gather
    index.  That changes its row layout, but it remains the outer dedicated
    reduction wrapper around the canonical projection capture.  Registering
    both wrappers under the same logical stage makes the witness ambiguous.
    Split-mode gather wrappers are canonical and remain in the base inventory.
    """

    row_indices = wrappers.get("row_indices")
    qkv = wrappers.get("qkv")
    output = wrappers.get("o")
    return (
        not isinstance(row_indices, torch.Tensor)
        or bool(getattr(qkv, "_use_buffer", False))
        or bool(getattr(output, "_use_buffer", False))
    )


def _proof_v3_capture_wrapper_has_buffer(wrapper: object) -> bool:
    """Return whether a wrapper exposes graph-readable raw-row storage."""

    # Linear wrappers expose their input/output tensors directly, while
    # decoder-layer wrappers expose a canonical inventory method.  Inspect the
    # concrete class for that method: CaptureLinearWrapper deliberately
    # delegates unknown attributes to its wrapped vLLM module, so an instance
    # getattr would incorrectly probe the underlying projection.
    if (
        getattr(wrapper, "_capture_buf", None) is not None
        or getattr(wrapper, "_capture_output_buf", None) is not None
    ):
        return True
    inventory = getattr(type(wrapper), "proof_capture_buffers", None)
    return bool(inventory is not None and inventory(wrapper))


def _register_proof_v3_economic_pool_capture(
    *,
    tracker: RequestActivationTracker,
    capture_wrappers: list[object],
    root_row_aliases: tuple[tuple[int, str, torch.Tensor], ...],
    reduction_root_row_aliases: list[tuple[int, str, torch.Tensor]],
) -> None:
    """Converge buffer and graph-split witnesses into one pool inventory.

    Split-mode wrappers emit raw rows through ``capture_at_split`` rather
    than a graph-static tensor.  They must still be registered with the same
    tracker inventory as buffer-mode witnesses before any request is served.
    """

    economic_pool_buffers = list(
        tuple(tracker._capture_buffers) + root_row_aliases
    )
    economic_pool_keys = tuple(
        (int(item[0]), str(item[1]))
        for item in economic_pool_buffers
    )
    for reduction_alias in reduction_root_row_aliases:
        alias_key = (
            int(reduction_alias[0]),
            str(reduction_alias[1]),
        )
        if alias_key not in economic_pool_keys:
            economic_pool_buffers.append(reduction_alias)
            economic_pool_keys += (alias_key,)
    for whole_step_alias in reduction_root_row_aliases:
        alias_key = (
            int(whole_step_alias[0]),
            str(whole_step_alias[1]),
        )
        if economic_pool_keys.count(alias_key) != 1:
            raise RuntimeError(
                "proof-v3 K/V compact raw-row alias is missing or ambiguous"
            )

    # A split wrapper with a graph-native gather buffer must be read through
    # the post-step buffer path: CUDA-graph replay updates that tensor but does
    # not re-enter the Python split callback.  Register the callback-only stage
    # solely when no canonical buffer backs it, avoiding duplicate prompt rows
    # on backends where the eager split does execute.
    buffer_key_set = set(economic_pool_keys)
    split_stages = tuple(
        stage
        for wrapper in capture_wrappers
        for stage in wrapper.proof_capture_split_stages()
        if stage not in buffer_key_set
    )
    registered_keys = set(economic_pool_keys).union(split_stages)
    if (0, "residual_in") not in registered_keys:
        raise RuntimeError(
            "proof-v3 initial economic-pool residual witness is unavailable"
        )

    tracker.register_economic_pool_buffers(tuple(economic_pool_buffers))
    tracker.register_split_economic_pool_stages(split_stages)


logger = logging.getLogger(__name__)

PROOF_V2_RESPONSE_TARGET_SECONDS = 1.0
PROOF_V2_REVEAL_TIMEOUT_SECONDS = 2.0
PROOF_V2_MAX_PENDING_REVEALS = 256


class _PendingProofV2Reveal:
    """One in-memory reveal gate bound to a transmitted commitment."""

    __slots__ = (
        "challenge_id",
        "commitment_hash",
        "created_at",
        "deadline_at",
        "event",
        "nonce_commitment",
        "revealed_nonce",
        "session_id",
        "validator_hotkey",
    )

    def __init__(
        self,
        *,
        challenge_id: bytes,
        commitment_hash: bytes,
        deadline_at: float,
        nonce_commitment: bytes,
        session_id: str,
        validator_hotkey: str,
    ):
        self.challenge_id = challenge_id
        self.commitment_hash = commitment_hash
        self.created_at = time.perf_counter()
        self.deadline_at = deadline_at
        self.event = asyncio.Event()
        self.nonce_commitment = nonce_commitment
        self.revealed_nonce: Optional[bytes] = None
        self.session_id = session_id
        self.validator_hotkey = validator_hotkey


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
        self.evm_private_key: Optional[
            str
        ] = None  # hex, for identity challenge signing
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
        self.capacity_audit_state_file: str = ""
        self.proof_v2_pending_reveals: dict[bytes, _PendingProofV2Reveal] = {}
        # Proof-v3 remains unavailable until all authenticated artifacts,
        # graph-integrated capture and the hard-opening coordinator are ready.
        self.proof_v3_runtime = None
        self.proof_v3_coordinator = None
        self.allowed_proof_protocol_versions = (1, 3)


state = MinerState()
_PROOF_PROTOCOL_POLICY_RELOAD_SECONDS = 60.0
_proof_protocol_policy_last_load = 0.0
# A long prefill can legitimately produce no SSE output for several minutes,
# especially when it is continuously batched with another large request.  The
# validator's signed full-context request budget is 900 seconds, so the miner
# must not abort a healthy request earlier with its own queue-idle deadline.
_BATCH_OUTPUT_IDLE_TIMEOUT_SECONDS = 900.0


def _current_allowed_proof_protocol_versions() -> tuple[int, ...]:
    """Reload the owner policy written atomically by the miner wrapper."""

    global _proof_protocol_policy_last_load
    now = time.monotonic()
    if (
        now - _proof_protocol_policy_last_load
        < _PROOF_PROTOCOL_POLICY_RELOAD_SECONDS
    ):
        return state.allowed_proof_protocol_versions
    _proof_protocol_policy_last_load = now
    try:
        from verallm.api.validator_auth import resolve_validators_path

        path = resolve_validators_path()
        payload = json.loads(path.read_text())
        raw = payload.get("allowed_proof_protocol_versions")
        if not isinstance(raw, list) or not raw:
            return state.allowed_proof_protocol_versions
        versions = tuple(int(version) for version in raw)
        if (
            list(versions) != sorted(set(versions))
            or any(version < 1 or version > 255 for version in versions)
            or 2 in versions
        ):
            raise ValueError("invalid allowed proof protocol versions")
        state.allowed_proof_protocol_versions = versions
    except FileNotFoundError:
        pass
    except Exception as exc:
        bt.logging.warning(
            "Ignoring invalid refreshed proof-protocol allowlist: "
            f"{type(exc).__name__}: {exc}"
        )
    return state.allowed_proof_protocol_versions


def _remove_expired_proof_v2_reveals(now: Optional[float] = None) -> None:
    current = time.perf_counter() if now is None else now
    expired = [
        challenge_id
        for challenge_id, pending in state.proof_v2_pending_reveals.items()
        if pending.deadline_at <= current
    ]
    for challenge_id in expired:
        state.proof_v2_pending_reveals.pop(challenge_id, None)


def _register_proof_v2_reveal(
    *,
    challenge_id: bytes,
    commitment_hash: bytes,
    deadline_at: float,
    nonce_commitment: bytes,
    session_id: str,
    validator_hotkey: str,
) -> _PendingProofV2Reveal:
    _remove_expired_proof_v2_reveals()
    pending_reveals = state.proof_v2_pending_reveals
    if len(pending_reveals) >= PROOF_V2_MAX_PENDING_REVEALS:
        raise RuntimeError("proof-v2 reveal queue is full")
    if challenge_id in pending_reveals:
        raise RuntimeError("proof_challenge_id is already pending")
    pending = _PendingProofV2Reveal(
        challenge_id=challenge_id,
        commitment_hash=commitment_hash,
        deadline_at=deadline_at,
        nonce_commitment=nonce_commitment,
        session_id=session_id,
        validator_hotkey=validator_hotkey,
    )
    pending_reveals[challenge_id] = pending
    return pending


async def _await_proof_v2_reveal(pending: _PendingProofV2Reveal) -> bytes:
    remaining = pending.deadline_at - time.perf_counter()
    timeout = min(PROOF_V2_REVEAL_TIMEOUT_SECONDS, remaining)
    if timeout <= 0:
        raise TimeoutError("proof-v2 response deadline expired before nonce reveal")
    try:
        await asyncio.wait_for(pending.event.wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("proof-v2 validator nonce reveal timed out") from exc
    if pending.revealed_nonce is None:
        raise RuntimeError("proof-v2 nonce reveal completed without a nonce")
    return pending.revealed_nonce


def _proof_v2_precommit_data(
    *,
    challenge_id: bytes,
    commitment: InferenceCommitment,
    last_token_at: float,
) -> dict:
    return {
        "proof_challenge_id": encode_proof_challenge_id(challenge_id),
        "session_id": commitment.session_id,
        "commitment_hash": encode_proof_commitment_hash(commitment.commitment_hash()),
        "last_token_to_precommit_ms": round(
            max(0.0, (time.perf_counter() - last_token_at) * 1000),
            3,
        ),
    }


def _proof_v2_batch_capture_available() -> bool:
    return bool(
        state.batch_mode
        and state.batch_engine is not None
        and state.activation_tracker is not None
    )


def _advertised_proof_protocol_versions() -> list[int]:
    """Advertise ready protocols intersected with the owner allowlist."""

    v3_ready = bool(
        state.proof_v3_runtime is not None
        and state.proof_v3_coordinator is not None
        and _proof_v2_batch_capture_available()
    )
    ready = {LEGACY_PROOF_PROTOCOL_VERSION}
    if v3_ready:
        ready.add(PROOF_PROTOCOL_V3)
    allowed = set(_current_allowed_proof_protocol_versions())
    return sorted(ready.intersection(allowed))


def _proof_protocol_rollout_gate(
    proof_protocol_version: int,
) -> Optional[JSONResponse]:
    """Reject owner-disallowed proof protocols before inference."""

    allowed = _current_allowed_proof_protocol_versions()
    if proof_protocol_version not in allowed:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Proof protocol version is not currently allowed",
                "allowed_proof_protocol_versions": list(
                    allowed
                ),
            },
        )
    return None


def _proof_v2_hard_audit_capture_required(miner) -> bool:
    """Return whether every v2 request must retain hard-audit decode state.

    The decision itself is made only after the precommitment.  This helper
    intentionally reads just the authenticated manifest policy; request-level
    ``sampling_verification_bps`` must not make a request ineligible.
    """

    manifest = getattr(miner, "proof_v2_manifest", None)
    policy = getattr(
        getattr(manifest, "model_execution", None),
        "audit_policy",
        None,
    )
    rate = getattr(policy, "hard_audit_bps", None)
    return type(rate) is int and 1 <= rate <= 10_000


def _request_needs_logits_capture(
    *,
    do_sample: bool,
    sampling_bps: int,
    proof_protocol_version: int,
    require_hard_audit_capture: bool = False,
) -> bool:
    bps = clamp_sampling_bps(sampling_bps)
    return bool(
        bps >= HIGH_ASSURANCE_BPS
        or (do_sample and bps > 0)
        or (proof_protocol_version == PROOF_PROTOCOL_V2 and bps > 0)
        or (
            proof_protocol_version == PROOF_PROTOCOL_V2
            and require_hard_audit_capture
        )
    )


app = FastAPI(title="VeraLLM Miner", version="0.1.0")


def _capacity_audit_gate() -> Optional[JSONResponse]:
    """Reject normal inference while this miner is inside a local audit drain."""
    path = state.capacity_audit_state_file
    if not path:
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        bt.logging.warning(f"Capacity audit state file unreadable: {exc}")
        return None
    if not isinstance(payload, dict) or not payload.get("active"):
        return None
    now = time.time()
    try:
        until_ts = float(payload.get("until_ts") or 0.0)
    except Exception:
        until_ts = 0.0
    if until_ts > 0.0 and until_ts <= now:
        return None
    retry_after = 5
    if until_ts > now:
        retry_after = max(1, min(120, int(until_ts - now)))
    audit_id = str(payload.get("audit_id") or "")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Miner temporarily unavailable: capacity audit in progress",
            "audit_id": audit_id,
            "retry_after_ms": retry_after * 1000,
        },
        headers={"Retry-After": str(retry_after)},
    )


@app.on_event("startup")
async def _on_startup():
    """Start the background engine step loop if batch mode is enabled."""
    if state.batch_mode and state.batch_engine is not None:
        state._step_loop_task = asyncio.create_task(_engine_step_loop())
        bt.logging.info("Started background engine step loop (batch mode)")
        if state.proof_v3_runtime is not None:
            state.proof_v3_runtime.bind_serving_loop()

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
    """Warm decode and long-prefill kernels through the real batch engine."""
    from vllm import SamplingParams

    miner = state.miner
    batch_engine = state.batch_engine
    if miner is None or batch_engine is None:
        return

    bt.logging.info("Running batch-mode warmup...")
    t0 = time.perf_counter()

    tokenizer = miner.tokenizer
    template_kwargs = _chat_template_kwargs(tokenizer, enable_thinking=True)

    def _apply_template(content: str) -> str:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:
            return content

    async def _run(
        *,
        request_id: str,
        prompt: str,
        max_tokens: int,
    ) -> None:
        tracker = state.activation_tracker
        registered = False
        added = False
        try:
            if tracker is not None:
                tracker.register_request(
                    request_id,
                    "warmup-session",
                    capture_logits=False,
                )
                registered = True
            q = batch_engine.add_request(
                request_id,
                prompt,
                SamplingParams(max_tokens=max_tokens, temperature=0),
            )
            added = True
            while True:
                output = await q.get()
                if isinstance(output, BatchEngineRequestError):
                    raise output
                if output.finished:
                    break
        finally:
            if added:
                batch_engine.clear_finished(request_id)
            if registered:
                tracker.unregister_request(request_id)

    # Decode warmup: enough steps to compile all observed lazy decode kernels.
    await _run(
        request_id="warmup-batch-decode",
        prompt=_apply_template("Briefly explain quantum computing."),
        max_tokens=256,
    )
    decode_ms = (time.perf_counter() - t0) * 1000
    bt.logging.info(f"Batch decode warmup done ({decode_ms:.0f}ms)")

    # Long-prefill warmup: exercise the chunked prefill path used by full
    # canaries. The actual batch engine chunks this prompt according to the
    # production max_num_batched_tokens setting.
    long_content = (
        "Warmup context records a concise technical observation followed by "
        "one supporting detail. "
    ) * 256
    await _run(
        request_id="warmup-batch-prefill",
        prompt=_apply_template(long_content),
        max_tokens=4,
    )
    bt.logging.info(
        f"Batch warmup done ({(time.perf_counter() - t0) * 1000:.0f}ms)"
    )


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

_AWQ_GEMM_HINT_PATH = "/tmp/verathos_awq_gemm_fallback"
_AWQ_GEMM_HINT_EXIT = 43


def _quant_method_from_config(qcfg) -> str:
    if qcfg is None:
        return ""
    keys = (
        "quant_method",
        "quantization_method",
        "quantization",
        "format",
        "load_format",
    )
    if isinstance(qcfg, dict):
        for key in keys:
            val = qcfg.get(key)
            if isinstance(val, str) and val:
                return val.lower().replace("_", "-")
        for nested in ("config_groups", "quantization_config"):
            val = qcfg.get(nested)
            if isinstance(val, dict):
                nested_method = _quant_method_from_config(next(iter(val.values()), val))
                if nested_method:
                    return nested_method
        return ""
    for key in keys:
        val = getattr(qcfg, key, None)
        if isinstance(val, str) and val:
            return val.lower().replace("_", "-")
    return ""


def _model_quant_method(model_name: str) -> str:
    if not model_name:
        return ""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        method = _quant_method_from_config(getattr(cfg, "quantization_config", None))
        if method:
            return method
    except Exception:
        pass
    try:
        from transformers import PretrainedConfig

        cfg_dict, _ = PretrainedConfig.get_config_dict(
            model_name,
            trust_remote_code=True,
        )
        if isinstance(cfg_dict, dict):
            return _quant_method_from_config(cfg_dict.get("quantization_config"))
    except Exception:
        pass
    return ""


# Validator auth: verifies Sr25519 signature against metagraph allowlist.
# Blocks non-public requests when no validators file exists (deny by default).
app.add_middleware(ValidatorAuthMiddleware)
# API key auth: optional secondary layer (VERATHOS_API_KEY env var).
app.add_middleware(APIKeyMiddleware)

from verallm.api.economic_proof_v3 import (
    register_economic_proof_v3_routes,
)

register_economic_proof_v3_routes(
    app,
    get_coordinator=lambda: state.proof_v3_coordinator,
    retain_completed_bundle=(
        state.receipt_store.stage_proof_v3_hard_bundle
    ),
)


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
        # Report the versions imported by this running process.  These are
        # operational diagnostics only; validators never trust /health for
        # proof or release admission.
        "miner_version": miner_version_str,
        "spec_version": spec_version_str,
        "model": state.model_name,
        "moe": state.moe_config is not None,
        "batch_mode": state.batch_mode,
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ],
        "capture_backend": (
            state.activation_tracker.backend
            if state.activation_tracker is not None
            else (
                "splitting_ops"
                if state.miner and getattr(state.miner, "_use_cuda_graphs", False)
                else "hooks"
            )
        ),
        "max_model_len": state.miner.llm.llm_engine.model_config.max_model_len
        if state.miner and state.miner.llm
        else None,
        "proof_protocol_versions": _advertised_proof_protocol_versions(),
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
        # Admission reservations and live vLLM requests must agree after a
        # disconnected stream has been cleaned up.  Expose the engine-side
        # count separately so routing and operators can detect an orphaned
        # generation instead of treating the endpoint as idle.
        result["engine_active_requests"] = (
            state.batch_engine.num_active
            if state.batch_engine is not None
            else 0
        )
        result["max_requests"] = state.admission.max_requests
        result["kv_pool_tokens"] = s.total_kv_tokens
        result["kv_used_tokens"] = s.used_tokens
        result["kv_free_tokens"] = s.free_tokens
        result["kv_utilization_pct"] = (
            round(s.used_tokens / s.total_kv_tokens * 100, 1)
            if s.total_kv_tokens > 0
            else 0
        )
        result["can_accept_max_context"] = s.can_accept_max_context
        result["hard_proof_exclusive"] = s.hard_proof_exclusive
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
            content={
                "error": "Identity challenge not available (no EVM key configured)"
            },
        )

    try:
        nonce_bytes = bytes.fromhex(body.nonce)
        if len(nonce_bytes) != 32:
            return JSONResponse(
                status_code=400,
                content={"error": "Nonce must be 32 bytes (64 hex chars)"},
            )
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


@app.post("/proof/v2/challenge")
async def reveal_proof_v2_challenge(
    body: ProofV2ChallengeRevealBody,
    request: Request,
):
    """Reveal one committed validator nonce after the miner freezes C."""
    challenge_id = decode_proof_challenge_id(body.proof_challenge_id)
    commitment_hash = decode_proof_commitment_hash(body.commitment_hash)
    validator_nonce = decode_validator_nonce(body.validator_nonce)
    pending = state.proof_v2_pending_reveals.get(challenge_id)
    if pending is None:
        return JSONResponse(
            status_code=404,
            content={"error": "proof-v2 challenge is not pending"},
        )

    now = time.perf_counter()
    if pending.deadline_at <= now:
        state.proof_v2_pending_reveals.pop(challenge_id, None)
        return JSONResponse(
            status_code=410,
            content={"error": "proof-v2 challenge has expired"},
        )

    validator_hotkey = getattr(request.state, "validator_hotkey", "")
    if pending.validator_hotkey != validator_hotkey:
        return JSONResponse(
            status_code=403,
            content={"error": "proof-v2 challenge belongs to another validator"},
        )
    if pending.session_id != body.session_id:
        return JSONResponse(
            status_code=409,
            content={"error": "proof-v2 session_id does not match"},
        )
    if not hmac.compare_digest(pending.commitment_hash, commitment_hash):
        return JSONResponse(
            status_code=409,
            content={"error": "proof-v2 commitment hash does not match"},
        )
    if not validator_nonce_matches_commitment_v2(
        validator_nonce=validator_nonce,
        proof_challenge_id=challenge_id,
        expected_commitment=pending.nonce_commitment,
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "validator nonce does not match its commitment"},
        )

    if pending.revealed_nonce is not None:
        if not hmac.compare_digest(pending.revealed_nonce, validator_nonce):
            return JSONResponse(
                status_code=409,
                content={"error": "proof-v2 challenge was already revealed"},
            )
        return {
            "status": "accepted",
            "proof_challenge_id": body.proof_challenge_id,
            "idempotent": True,
        }

    pending.revealed_nonce = validator_nonce
    pending.event.set()
    return {
        "status": "accepted",
        "proof_challenge_id": body.proof_challenge_id,
        "idempotent": False,
    }


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
    # The proof transcript commits milli-units, so execute that exact value.
    pp = round(float(pp) * 1000.0) / 1000.0

    return {
        "max_tokens": body.max_new_tokens,
        "temperature": temperature,
        "presence_penalty": pp,
        "top_k": body.top_k if body.top_k is not None else -1,
        "top_p": body.top_p if body.top_p is not None else 1.0,
        "min_p": body.min_p if body.min_p is not None else 0.0,
    }


def _private_vllm_sampling_seed(
    canonical_seed: bytes | None = None,
) -> int:
    """Return a private per-request seed for vLLM's CUDA sampler.

    vLLM uses the process-global CUDA generator whenever any sampled request
    has no seed. A failed CUDA-graph capture can leave that generator in
    PyTorch's capture state, causing every later unseeded sample to fail even
    though the engine itself remains usable. Giving every stochastic request
    its own generator avoids that process-global failure mode. The canonical
    proof seed remains unchanged and is only domain-separated for vLLM's
    internal draw; unaudited stochastic requests receive fresh private
    entropy without adding proof capture work.
    """

    if canonical_seed is None:
        material = os.urandom(32)
    else:
        if not isinstance(canonical_seed, bytes) or len(canonical_seed) != 32:
            raise ValueError("canonical sampling seed must contain 32 bytes")
        material = canonical_seed
    digest = hashlib.sha256(
        b"verathos.vllm.private_sampling_seed.v1\x00" + material
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _chat_template_kwargs(tokenizer, enable_thinking: bool = True) -> dict:
    """Extra kwargs for apply_chat_template based on model capabilities.

    Models like Qwen3/3.5 accept ``enable_thinking`` — when True the template
    injects ``<think>\\n`` into the prompt so the model reasons before
    answering.  The caller controls this per-request.
    """
    import inspect

    try:
        src = inspect.getsource(tokenizer.apply_chat_template)
    except (AttributeError, TypeError, OSError):
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
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
    audit_gate = _capacity_audit_gate()
    if audit_gate is not None:
        return audit_gate

    proof_protocol_version = body.resolved_proof_protocol_version
    rollout_gate = _proof_protocol_rollout_gate(proof_protocol_version)
    if rollout_gate is not None:
        return rollout_gate
    nonce = (
        body.validator_nonce_bytes
        if proof_protocol_version == LEGACY_PROOF_PROTOCOL_VERSION
        else None
    )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and getattr(state.miner, "proof_v2_manifest", None) is None
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Requested proof protocol is unavailable"},
        )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and not _proof_v2_batch_capture_available()
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Requested proof protocol requires batch capture"},
        )
    if proof_protocol_version == PROOF_PROTOCOL_V3:
        if (
            state.proof_v3_runtime is None
            or state.proof_v3_coordinator is None
        ):
            return JSONResponse(
                status_code=503,
                content={"error": "Requested proof protocol is unavailable"},
            )
        if not _proof_v2_batch_capture_available():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Requested proof protocol requires batch capture"
                },
            )

    # Compute prompt_hash from the raw prompt (for /inference endpoint).
    # Stored as local var (not on state) to avoid race conditions between
    # concurrent requests — state._current_prompt_hash was shared/mutable.
    _prompt_hash = hashlib.sha256(body.prompt.encode()).digest()

    # Apply chat template — instruction-tuned models need proper formatting
    tokenizer = state.miner.tokenizer
    formatted_prompt, prompt_token_ids = _apply_chat_template(
        tokenizer, body.prompt, enable_thinking=body.enable_thinking
    )
    if formatted_prompt is not None:
        body = InferenceRequestBody(
            prompt=formatted_prompt,
            validator_nonce=body.validator_nonce,
            validator_nonce_commitment=body.validator_nonce_commitment,
            proof_challenge_id=body.proof_challenge_id,
            proof_v3_preexecution_context=(
                body.proof_v3_preexecution_context
            ),
            proof_v3_hard_proof_arrival_budget_ns=(
                body.proof_v3_hard_proof_arrival_budget_ns
            ),
            proof_protocol_version=body.proof_protocol_version,
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

    proof_v3_context = None
    proof_v3_tracker_options = None
    if proof_protocol_version == PROOF_PROTOCOL_V3:
        if prompt_token_ids is None:
            try:
                prompt_token_ids = tokenizer.encode(
                    body.prompt,
                    add_special_tokens=False,
                )
            except TypeError:
                prompt_token_ids = tokenizer.encode(body.prompt)
        _vhk = (
            getattr(getattr(request, "state", None), "validator_hotkey", "")
            if request
            else ""
        )
        try:
            proof_v3_context = body.proof_v3_preexecution_context_value
            proof_v3_tracker_options = (
                state.proof_v3_runtime.validate_initial_request(
                    precommit_context=proof_v3_context,
                    authenticated_validator_hotkey=_vhk,
                    prompt_token_ids=prompt_token_ids,
                    do_sample=body.do_sample,
                    resolved_sampling_params=_resolve_sampling_params(
                        body,
                        state.model_name,
                    ),
                )
            )
        except Exception as exc:
            from verallm.proof_v3.errors import ProofV3Error

            if isinstance(exc, ProofV3Error):
                return JSONResponse(
                    status_code=409,
                    content={"error": str(exc)},
                )
            raise

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

        _vhk = (
            getattr(getattr(request, "state", None), "validator_hotkey", "")
            if request
            else ""
        )
        return StreamingResponse(
            _stream_inference_batched(
                body,
                nonce,
                prompt_token_ids=prompt_token_ids,
                token_budget=token_budget,
                admitted_request_id=request_id,
                prompt_hash=_prompt_hash,
                validator_hotkey=_vhk,
                proof_protocol_version=proof_protocol_version,
                proof_v3_precommit_context=proof_v3_context,
                proof_v3_tracker_options=proof_v3_tracker_options,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-batch mode: original single-request path
    _vhk = (
        getattr(getattr(request, "state", None), "validator_hotkey", "")
        if request
        else ""
    )
    return StreamingResponse(
        _stream_inference(
            body,
            nonce,
            prompt_token_ids=prompt_token_ids,
            prompt_hash=_prompt_hash,
            validator_hotkey=_vhk,
            proof_protocol_version=proof_protocol_version,
        ),
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
    audit_gate = _capacity_audit_gate()
    if audit_gate is not None:
        return audit_gate

    # Extract validator hotkey for logging (set by ValidatorAuthMiddleware)
    _vali_hotkey = (
        getattr(getattr(request, "state", None), "validator_hotkey", "")
        if request
        else ""
    )

    proof_protocol_version = body.resolved_proof_protocol_version
    rollout_gate = _proof_protocol_rollout_gate(proof_protocol_version)
    if rollout_gate is not None:
        return rollout_gate
    nonce = (
        body.validator_nonce_bytes
        if proof_protocol_version == LEGACY_PROOF_PROTOCOL_VERSION
        else None
    )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and getattr(state.miner, "proof_v2_manifest", None) is None
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Requested proof protocol is unavailable"},
        )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and not _proof_v2_batch_capture_available()
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Requested proof protocol requires batch capture"},
        )
    if proof_protocol_version == PROOF_PROTOCOL_V3:
        if (
            state.proof_v3_runtime is None
            or state.proof_v3_coordinator is None
        ):
            return JSONResponse(
                status_code=503,
                content={"error": "Requested proof protocol is unavailable"},
            )
        if not _proof_v2_batch_capture_available():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Requested proof protocol requires batch capture"
                },
            )

    # Apply chat template using the miner's tokenizer
    tokenizer = state.miner.tokenizer

    def _msg_to_dict(m: ChatMessage) -> dict:
        if hasattr(m, "model_dump"):
            d = m.model_dump(exclude_none=True)
        else:
            d = m.dict(exclude_none=True)
        if m.content is None and m.tool_calls:
            d["content"] = None
        if "content" not in d and "tool_calls" not in d:
            d["content"] = ""
        return d

    messages_dicts = [_msg_to_dict(m) for m in body.messages]
    # Compute prompt_hash from the canonical messages JSON for input integrity.
    # Stored as local var (not on state) to avoid race conditions between
    # concurrent requests — state._current_prompt_hash was shared/mutable.
    import json as _json

    _prompt_hash_obj = _chat_prompt_hash_payload(
        messages_dicts,
        body.tools,
        body.tool_choice,
        body.parallel_tool_calls,
    )
    _prompt_hash_input = _json.dumps(
        _prompt_hash_obj, sort_keys=True, ensure_ascii=False
    ).encode()
    _prompt_hash = hashlib.sha256(_prompt_hash_input).digest()
    bt.logging.debug(
        f"prompt_hash: {_prompt_hash.hex()[:16]} (len={len(_prompt_hash_input)})"
    )

    _is_mistral_tok = "MistralTokenizer" in type(tokenizer).__name__
    _extra_kw = _chat_template_kwargs(tokenizer, body.enable_thinking)
    _template_kw = dict(_extra_kw)
    if body.tools:
        _template_kw["tools"] = body.tools
    try:
        if proof_protocol_version == PROOF_PROTOCOL_V3:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages_dicts,
                tokenize=True,
                add_generation_prompt=True,
                **_template_kw,
            )
            from verallm.proof_v3.request import (
                canonical_tokenizer_token_ids_v3,
            )

            prompt_token_ids = canonical_tokenizer_token_ids_v3(
                prompt_token_ids
            )
            formatted_prompt = None
        elif _is_mistral_tok:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages_dicts,
                tokenize=True,
                add_generation_prompt=True,
            )
            formatted_prompt = None
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                messages_dicts,
                tokenize=False,
                add_generation_prompt=True,
                **_template_kw,
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
        validator_nonce_commitment=body.validator_nonce_commitment,
        proof_challenge_id=body.proof_challenge_id,
        proof_v3_preexecution_context=body.proof_v3_preexecution_context,
        proof_v3_hard_proof_arrival_budget_ns=(
            body.proof_v3_hard_proof_arrival_budget_ns
        ),
        proof_protocol_version=body.proof_protocol_version,
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

        proof_v3_context = None
        proof_v3_tracker_options = None
        if proof_protocol_version == PROOF_PROTOCOL_V3:
            try:
                proof_v3_context = (
                    body.proof_v3_preexecution_context_value
                )
                proof_v3_tracker_options = (
                    state.proof_v3_runtime.validate_initial_request(
                        precommit_context=proof_v3_context,
                        authenticated_validator_hotkey=_vali_hotkey,
                        prompt_token_ids=prompt_token_ids,
                        do_sample=synth_body.do_sample,
                        resolved_sampling_params=_resolve_sampling_params(
                            synth_body,
                            state.model_name,
                        ),
                    )
                )
            except Exception as exc:
                from verallm.proof_v3.errors import ProofV3Error

                if isinstance(exc, ProofV3Error):
                    await state.admission.release(request_id)
                    return JSONResponse(
                        status_code=409,
                        content={"error": str(exc)},
                    )
                raise

        return StreamingResponse(
            _stream_inference_batched(
                synth_body,
                nonce,
                prompt_token_ids=prompt_token_ids,
                token_budget=token_budget,
                admitted_request_id=request_id,
                prompt_hash=_prompt_hash,
                validator_hotkey=_vali_hotkey,
                proof_protocol_version=proof_protocol_version,
                proof_v3_precommit_context=proof_v3_context,
                proof_v3_tracker_options=proof_v3_tracker_options,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_inference(
            synth_body,
            nonce,
            prompt_token_ids=prompt_token_ids,
            prompt_hash=_prompt_hash,
            validator_hotkey=_vali_hotkey,
            proof_protocol_version=proof_protocol_version,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# TEE endpoints — confidential GPU mode (E2E encryption + attestation)
# ============================================================================


class TEEChatRequestBody(ProofProtocolRequestBody):
    """Encrypted chat request for TEE mode."""

    _inline_proof_v2: ClassVar[bool] = False
    envelope: dict  # {session_id, sender_public_key, nonce, ciphertext} (all hex)


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
        return JSONResponse(
            status_code=400,
            content={
                "error": "nonce required (hex-encoded, minimum 8 bytes / 16 hex chars)"
            },
        )

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
    audit_gate = _capacity_audit_gate()
    if audit_gate is not None:
        return audit_gate

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
                messages_dicts,
                tokenize=False,
                add_generation_prompt=True,
                **_extra_kw,
            )
            prompt_token_ids = None
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Chat template failed: {e}"},
        )

    nonce = body.validator_nonce_bytes
    proof_protocol_version = body.resolved_proof_protocol_version
    if proof_protocol_version == PROOF_PROTOCOL_V2:
        return JSONResponse(
            status_code=400,
            content={"error": "proof-v2 is unavailable for encrypted TEE chat"},
        )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and not _proof_v2_batch_capture_available()
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Requested proof protocol requires batch capture"},
        )
    sender_pk = envelope.sender_public_key

    # Compute prompt_hash for TEE chat (same canonical JSON format).
    import json as _json

    _prompt_hash = hashlib.sha256(
        _json.dumps(messages_dicts, sort_keys=True, ensure_ascii=False).encode()
    ).digest()

    _vhk = (
        getattr(getattr(request, "state", None), "validator_hotkey", "")
        if request
        else ""
    )

    async def _stream_tee_inference():
        """Run inference and stream encrypted results."""
        t0 = time.time()
        seq = 0
        full_output_tokens = []

        # Build a synthetic request for the existing inference pipeline
        synth_body = InferenceRequestBody(
            prompt=formatted_prompt or "",
            validator_nonce=body.validator_nonce,
            proof_protocol_version=body.proof_protocol_version,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            enable_thinking=getattr(body, "enable_thinking", True),
        )

        try:
            if state.batch_mode and state.batch_engine is not None:
                # Use batch inference path
                gen = _stream_inference_batched(
                    synth_body,
                    nonce,
                    prompt_token_ids=prompt_token_ids,
                    token_budget=(len(prompt_token_ids or []) or 0) + max_new_tokens,
                    prompt_hash=_prompt_hash,
                    validator_hotkey=_vhk,
                    proof_protocol_version=proof_protocol_version,
                )
            else:
                gen = _stream_inference(
                    synth_body,
                    nonce,
                    prompt_token_ids=prompt_token_ids,
                    prompt_hash=_prompt_hash,
                    proof_protocol_version=proof_protocol_version,
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

                if (
                    "text" in event
                    and "done" not in event
                    and "commitment" not in event
                ):
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
                        k: event[k]
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "inference_ms",
                            "ttft_ms",
                            "commitment_ms",
                            "prove_ms",
                            "beacon_ms",
                            "challenge_ms",
                            "model_id",
                        )
                        if k in event
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
    tee_attestation_verified: Optional[
        bool
    ] = None  # None=not tested, True=passed, False=failed
    is_canary: bool = False
    receipt_version: int = 1
    timing_source: str = "legacy"
    observed_start_ts: float = 0.0
    observed_end_ts: float = 0.0
    capture_chain_digest: str = ""
    canary_obligation_id: str = ""
    canary_kind: str = ""
    canary_target_prompt_tokens: int = 0
    timing_signature: str = ""
    validator_hotkey: str  # hex
    validator_signature: str  # hex


def _require_authorized_hard_receipt_identity(
    body: EpochReceiptBody,
    request: Request,
) -> JSONResponse | None:
    if not bool(
        getattr(
            request.state,
            "proof_v3_hard_auditor_authorized",
            False,
        )
    ):
        return JSONResponse(
            status_code=403,
            content={"error": "Proof-v3 hard auditor is not authorized"},
        )
    try:
        from verallm.chain.wallet import ss58_encode

        receipt_hotkey = bytes.fromhex(body.validator_hotkey)
    except ValueError:
        receipt_hotkey = b""
    if (
        len(receipt_hotkey) != 32
        or ss58_encode(receipt_hotkey)
        != getattr(request.state, "validator_hotkey", "")
    ):
        return JSONResponse(
            status_code=403,
            content={
                "error": (
                    "Receipt validator identity does not match "
                    "the authenticated hard auditor"
                )
            },
        )
    return None


@app.post("/epoch/receipt")
async def receive_epoch_receipt(body: EpochReceiptBody, request: Request):
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
            content={
                "error": "Receipt address mismatch — this endpoint belongs to a different miner"
            },
        )

    receipt_dict = body.model_dump()

    if (
        body.proof_requested
        and body.proof_verified
        and body.is_canary
        and bool(
            getattr(
                request.state,
                "proof_v3_hard_auditor_authorized",
                False,
            )
        )
    ):
        try:
            identity_error = _require_authorized_hard_receipt_identity(
                body,
                request,
            )
            if identity_error is not None:
                return identity_error
            state.receipt_store.promote_proof_v3_hard_bundle(
                epoch=epoch,
                receipt_dict=receipt_dict,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Proof-v3 hard bundle retention failed: %s", exc)
            return JSONResponse(
                status_code=409,
                content={"error": "Proof-v3 hard bundle retention failed"},
            )

    # No artificial receipt cap — throughput is naturally bounded by inference
    # rate, and validator auth + epoch GC prevent abuse.
    count = state.receipt_store.add(epoch, receipt_dict)

    # Auto-GC: remove old epochs from memory + disk
    state.receipt_store.gc(epoch)

    return {"status": "accepted", "epoch": epoch, "count": count}


@app.post("/proof/v3/audit-receipt")
async def receive_proof_v3_security_receipt(
    body: EpochReceiptBody,
    request: Request,
):
    """Promote one late verified hard bundle without throughput credit."""

    if (
        not body.proof_requested
        or not body.proof_verified
        or not body.is_canary
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "A verified hard-audit receipt is required"},
        )
    if (
        state.evm_address
        and body.miner_address.lower() != state.evm_address.lower()
    ):
        return JSONResponse(
            status_code=403,
            content={"error": "Receipt address mismatch"},
        )
    identity_error = _require_authorized_hard_receipt_identity(body, request)
    if identity_error is not None:
        return identity_error
    receipt_dict = body.model_dump()
    try:
        promoted = state.receipt_store.promote_proof_v3_hard_bundle(
            epoch=body.epoch_number,
            receipt_dict=receipt_dict,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Late proof-v3 hard bundle retention failed: %s", exc)
        return JSONResponse(
            status_code=409,
            content={"error": "Proof-v3 hard bundle retention failed"},
        )
    if promoted is None:
        return JSONResponse(
            status_code=409,
            content={"error": "Completed proof-v3 hard bundle is unavailable"},
        )
    state.receipt_store.gc(body.epoch_number)
    return {
        "status": "accepted",
        "epoch": body.epoch_number,
        "commitment_hash": promoted["commitment_hash"],
    }


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


@app.get("/proof/v3/bundles/{epoch_number}")
async def get_proof_v3_hard_bundle_index(epoch_number: int):
    """Return receipt-matched completed hard bundles in canonical order."""

    if not 0 <= epoch_number < 1 << 63:
        return JSONResponse(
            status_code=400,
            content={"error": "Proof-v3 bundle epoch is out of range"},
        )
    entries = state.receipt_store.get_proof_v3_hard_bundle_index(
        epoch_number
    )
    return JSONResponse(
        content={
            "epoch": epoch_number,
            "bundle_count": len(entries),
            "bundles": entries,
        },
        headers={"Cache-Control": "private, max-age=30"},
    )


@app.get("/proof/v3/bundles/{epoch_number}/{commitment_hash}")
async def get_proof_v3_hard_bundle(
    epoch_number: int,
    commitment_hash: str,
):
    """Return one immutable retained bundle for deterministic re-verification."""

    if not 0 <= epoch_number < 1 << 63:
        return JSONResponse(
            status_code=400,
            content={"error": "Proof-v3 bundle epoch is out of range"},
        )
    try:
        parsed = bytes.fromhex(commitment_hash)
    except ValueError:
        parsed = b""
    if len(parsed) != 32 or parsed.hex() != commitment_hash:
        return JSONResponse(
            status_code=400,
            content={"error": "Proof-v3 commitment hash is malformed"},
        )
    try:
        encoded = state.receipt_store.get_proof_v3_hard_bundle(
            epoch=epoch_number,
            commitment_hash=parsed,
        )
    except RuntimeError:
        logger.exception("Retained proof-v3 hard bundle checksum failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Retained proof-v3 hard bundle is unavailable"},
        )
    if encoded is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Proof-v3 hard bundle was not found"},
        )
    from verallm.proof_v3.hard_bundle import HARD_BUNDLE_MEDIA_TYPE_V3

    return Response(
        content=encoded,
        media_type=HARD_BUNDLE_MEDIA_TYPE_V3,
        headers={
            "Cache-Control": "private, max-age=3600, immutable",
            "ETag": f'"{hashlib.sha256(encoded).hexdigest()}"',
        },
    )


# ============================================================================
# SSE inference streaming + proof generation
# ============================================================================


def _derive_inference_challenges(
    *,
    miner,
    commitment: InferenceCommitment,
    beacon: bytes,
    session_id: str,
    proof_protocol_version: int,
    validator_nonce: bytes,
    router_commitments,
    num_input_tokens: int,
) -> ChallengeSet:
    """Derive the negotiated proof challenges plus common output challenges."""

    if proof_protocol_version == PROOF_PROTOCOL_V2:
        from verallm.challenge.v2 import (
            MAX_BLOCKS_PER_OPERATION,
            derive_block_challenges_v2,
            derive_inference_transcript_state_v2,
            derive_stratified_execution_layers_v2,
        )
        from verallm.proof_v2.layout import registered_operations_from_manifest

        manifest = getattr(miner, "proof_v2_manifest", None)
        if manifest is None:
            raise RuntimeError("proof-v2 manifest is not configured")
        x_state = getattr(miner, "proof_v2_x_states", {}).get(session_id)
        if x_state is None:
            raise RuntimeError("proof-v2 pre-challenge X state is missing")
        envelope = x_state.envelope
        trace_commitment = getattr(envelope, "execution_trace_commitment", None)
        execution_profile = getattr(manifest, "execution_profile", None)
        if execution_profile is None:
            raise RuntimeError("proof-v2 causal execution profile is required")
        if trace_commitment is None:
            raise RuntimeError("proof-v2 causal execution trace commitment is missing")
        if (
            trace_commitment.profile != execution_profile
            or trace_commitment.num_layers != manifest.model_spec.num_layers
            or trace_commitment.token_count != commitment.output_token_count
            or any(
                item.row_count != trace_commitment.token_count
                for item in envelope.x_commitments
            )
        ):
            raise RuntimeError("proof-v2 causal execution trace context is not exact")
        if envelope.manifest_digest != manifest.digest():
            raise RuntimeError(
                "proof-v2 X state does not match the configured manifest"
            )
        if commitment.proof_v2_commitment != envelope.canonical_bytes():
            raise RuntimeError(
                "proof-v2 X state does not match the inference commitment"
            )
        validate_proof_v2_decode_commitment(commitment)
        policy = getattr(
            getattr(manifest, "model_execution", None),
            "audit_policy",
            None,
        )
        if policy is None:
            raise RuntimeError("proof-v2 signed hard-audit policy is missing")
        hard_audit_bps = getattr(policy, "hard_audit_bps", None)
        if type(hard_audit_bps) is not int or not 1 <= hard_audit_bps <= 10_000:
            raise RuntimeError("proof-v2 signed hard-audit policy rate is invalid")
        transcript_state = derive_inference_transcript_state_v2(
            validator_nonce=validator_nonce,
            manifest_digest=manifest.digest(),
            commitment_envelope=envelope.canonical_bytes(),
            model_id=commitment.model_id,
            model_commitment=commitment.model_commitment,
            input_commitment=commitment.input_commitment,
            prompt_hash=commitment.prompt_hash,
            sampler_config_hash=commitment.sampler_config_hash,
            sampling_verification_bps=commitment.sampling_verification_bps,
            do_sample=commitment.do_sample,
            temperature_milli=commitment.temperature_milli,
            presence_penalty_milli=commitment.presence_penalty_milli,
        )
        hard_audit = hard_audit_required(
            beacon,
            commitment,
            hard_audit_bps,
        )
        hard_audit_layers = None
        hard_audit_layer_count = state.config.k_layers
        hard_audit_blocks_per_operation = state.config.k_blocks
        hard_audit_row = None
        if hard_audit:
            hard_audit_layer_count = policy.hard_layer_count
            hard_audit_blocks_per_operation = getattr(
                policy,
                "hard_blocks_per_operation",
                None,
            )
            if (
                type(hard_audit_blocks_per_operation) is not int
                or not 1
                <= hard_audit_blocks_per_operation
                <= MAX_BLOCKS_PER_OPERATION
            ):
                raise RuntimeError(
                    "proof-v2 signed hard-audit block coverage is invalid"
                )
            hard_audit_layers = derive_stratified_execution_layers_v2(
                transcript_state=transcript_state,
                layer_attention_profiles=tuple(
                    item.attention_profile for item in manifest.layer_execution
                ),
                hard_layer_count=policy.hard_layer_count,
                min_full_attention_layers=policy.min_full_attention_layers,
                min_gdn_layers=policy.min_gdn_layers,
            )
            sampling_challenge = derive_hard_audit_sampling_challenge(
                beacon=beacon,
                commitment=commitment,
                vocab_size=int(getattr(miner.model_spec, "vocab_size", 0) or 0),
                k_positions=2,
            )
            if sampling_challenge is None:
                raise RuntimeError("proof-v2 decode challenge could not be derived")
            from verallm.proof_v2.hardening import (
                select_lm_head_audit_decode_step_v2,
            )

            hard_audit_row = select_lm_head_audit_decode_step_v2(
                transcript_state=transcript_state,
                commitment_hash=commitment.commitment_hash(),
                decode_positions=tuple(sampling_challenge.decode_positions),
                minimum_decode_step=1,
            )

        block_challenge_kwargs = {}
        if hard_audit_layers is not None:
            block_challenge_kwargs["selected_layer_indices"] = hard_audit_layers
        block_challenges = derive_block_challenges_v2(
            transcript_state=transcript_state,
            num_layers=manifest.model_spec.num_layers,
            operations=registered_operations_from_manifest(manifest),
            x_commitments=envelope.x_commitments,
            runtime_y_commitments=envelope.runtime_y_commitments,
            k_layers=(hard_audit_layer_count if hard_audit else state.config.k_layers),
            k_operations_per_layer=1,
            k_blocks_per_operation=(
                hard_audit_blocks_per_operation
                if hard_audit
                else state.config.k_blocks
            ),
            all_operations_per_selected_layer=hard_audit,
            required_row_index=hard_audit_row,
            **block_challenge_kwargs,
        )
        challenges = ChallengeSet(
            beacon=beacon,
            layer_challenges=[],
            proof_v2_challenges=block_challenges,
            proof_v2_transcript_state=transcript_state,
            proof_v2_hard_audit=hard_audit,
        )
    elif proof_protocol_version == LEGACY_PROOF_PROTOCOL_VERSION:
        if state.moe_config is not None:
            challenges = derive_moe_challenges(
                beacon=beacon,
                commitment=commitment,
                moe_config=state.moe_config,
                router_commitments=router_commitments or {},
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
    else:
        raise ValueError(
            f"unsupported proof protocol version: {proof_protocol_version}"
        )

    hard_decode_audit = (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and bool(challenges.proof_v2_hard_audit)
    )
    sampling_challenge = (
        derive_hard_audit_sampling_challenge(
            beacon=beacon,
            commitment=commitment,
            vocab_size=int(getattr(miner.model_spec, "vocab_size", 0) or 0),
            k_positions=2,
        )
        if hard_decode_audit
        else derive_sampling_challenge(
            beacon=beacon,
            commitment=commitment,
            vocab_size=int(getattr(miner.model_spec, "vocab_size", 0) or 0),
        )
    )
    challenges.sampling_challenge = (
        sampling_challenge
        if proof_protocol_version != PROOF_PROTOCOL_V2 or hard_decode_audit
        else None
    )
    if (
        hard_decode_audit
        and challenges.sampling_challenge is None
    ):
        raise RuntimeError("proof-v2 decode challenge could not be derived")
    emb_root = getattr(miner.model_spec, "embedding_weight_merkle_root", b"")
    if emb_root:
        challenges.embedding_challenge = derive_embedding_challenge(
            beacon=beacon,
            commitment=commitment,
            num_input_tokens=num_input_tokens,
            include_last_position=(proof_protocol_version == PROOF_PROTOCOL_V2),
        )
    return challenges


def _cleanup_inference_session(miner, session_id: str) -> None:
    """Release proof artifacts retained for one completed inference."""
    miner.witnesses.pop(session_id, None)
    miner.activation_merkle_trees.pop(session_id, None)
    getattr(miner, "proof_v2_x_states", {}).pop(session_id, None)
    getattr(miner, "proof_v2_trace_tokens", {}).pop(session_id, None)
    getattr(miner, "proof_v2_trace_tails", {}).pop(session_id, None)
    getattr(miner, "proof_v2_trace_layer_contexts", {}).pop(session_id, None)
    getattr(miner, "proof_v2_gdn_prompt_boundaries", {}).pop(session_id, None)
    getattr(miner, "proof_v2_full_attention_prompt_boundaries", {}).pop(
        session_id,
        None,
    )
    miner.router_commitments.pop(session_id, None)
    miner.router_logits.pop(session_id, None)
    miner.decode_hidden_row_trees.pop(session_id, None)
    miner.decode_hidden_rows.pop(session_id, None)
    miner.decode_logits_row_trees.pop(session_id, None)
    miner.decode_logits_rows.pop(session_id, None)
    getattr(miner, "_sampling_seeds", {}).pop(session_id, None)
    miner.input_token_ids.pop(session_id, None)
    miner.embedding_output_trees.pop(session_id, None)
    miner.output_token_ids.pop(session_id, None)


async def _stream_inference(
    body: InferenceRequestBody,
    nonce: Optional[bytes],
    prompt_token_ids: list[int] | None = None,
    prompt_hash: bytes = b"",
    validator_hotkey: str = "",
    proof_protocol_version: int = LEGACY_PROOF_PROTOCOL_VERSION,
):
    """Generator that yields SSE events: token deltas, then commitment + proofs.

    Args:
        body: Request body with prompt, nonce, and generation params.
        nonce: Validator nonce bytes for v1. V2 reveals it after precommit.
        prompt_token_ids: If provided (e.g. from /chat with Mistral tokenizer),
            use these token IDs directly instead of body.prompt string.
        proof_protocol_version: Resolved proof version (omitted requests are v1).
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
                hook_handles.append(
                    mlp.register_forward_hook(make_hook(idx, "mlp_gate"))
                )
            else:
                gate_proj = miner._get_gate_proj(mlp)
                if gate_proj is not None:
                    hook_handles.append(
                        gate_proj.register_forward_hook(make_hook(idx, "mlp_gate"))
                    )

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

    _resolved_sp = _resolve_sampling_params(body, state.model_name)
    if body.do_sample:
        _resolved_sp["seed"] = _private_vllm_sampling_seed()
    sampling_params = SamplingParams(**_resolved_sp)

    engine = miner.llm.llm_engine
    t_infer = time.perf_counter()
    if prompt_token_ids is not None:
        engine.add_request(
            "stream-0", {"prompt_token_ids": prompt_token_ids}, sampling_params
        )
    else:
        engine.add_request("stream-0", body.prompt, sampling_params)

    prev_text = ""
    final_output = None
    t_first_token = None
    t_last_token = None

    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            cur_text = output.outputs[0].text if output.outputs else ""
            delta = cur_text[len(prev_text) :]
            if delta:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                t_last_token = time.perf_counter()
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
                session_router_logits[layer_idx] = (
                    decision.router_logits.detach().float().cpu()
                )
        moe_hook_mgr.remove_hooks()

    if final_output is None:
        yield f"event: error\ndata: {json.dumps({'error': 'No output generated'})}\n\n"
        return

    if t_last_token is None:
        t_last_token = time.perf_counter()

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
        miner,
        activations,
        input_token_ids,
        output_token_ids,
        session_id,
        inference_ms,
        router_commitments=session_router_commitments,
        do_sample=body.do_sample,
        temperature=float(_resolved["temperature"]),
        sampling_verification_bps=body.sampling_verification_bps,
        presence_penalty=resolved_pp,
        prompt_hash=prompt_hash,
        top_k=int(_resolved.get("top_k", -1) or -1),
        top_p=float(_resolved.get("top_p", 1.0) or 1.0),
        min_p=float(_resolved.get("min_p", 0.0) or 0.0),
        sampling_seed=_seed_for_commit,
        finish_reason=str(final_output.outputs[0].finish_reason or ""),
        proof_protocol_version=proof_protocol_version,
    )
    # Store router commitments for proof bundle
    if session_router_commitments:
        miner.router_commitments[session_id] = session_router_commitments
    if session_router_logits:
        miner.router_logits[session_id] = session_router_logits
    # Store input token IDs for embedding proof generation.
    miner.input_token_ids[session_id] = list(input_token_ids)

    pending_reveal = None
    reveal_ms = 0.0
    if proof_protocol_version == PROOF_PROTOCOL_V2:
        challenge_id = body.proof_challenge_id_bytes
        pending_reveal = _register_proof_v2_reveal(
            challenge_id=challenge_id,
            commitment_hash=commitment.commitment_hash(),
            deadline_at=time.perf_counter() + PROOF_V2_REVEAL_TIMEOUT_SECONDS,
            nonce_commitment=body.validator_nonce_commitment_bytes,
            session_id=session_id,
            validator_hotkey=validator_hotkey,
        )
        precommit_data = _proof_v2_precommit_data(
            challenge_id=challenge_id,
            commitment=commitment,
            last_token_at=t_last_token or time.perf_counter(),
        )
        yield ("event: proof_precommit\n" f"data: {json.dumps(precommit_data)}\n\n")
        yield (
            "event: proof_commitment\n"
            f"data: {json.dumps({'commitment': commitment_to_dict(commitment)})}\n\n"
        )
        reveal_started = time.perf_counter()
        try:
            nonce = await _await_proof_v2_reveal(pending_reveal)
        except Exception as exc:
            state.proof_v2_pending_reveals.pop(challenge_id, None)
            _cleanup_inference_session(miner, session_id)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return
        reveal_ms = (time.perf_counter() - reveal_started) * 1000

    if nonce is None:
        _cleanup_inference_session(miner, session_id)
        yield (
            "event: error\n"
            f"data: {json.dumps({'error': 'validator nonce is unavailable'})}\n\n"
        )
        return

    # ── Phase 4: Derive beacon (Fiat-Shamir) ────────────────────────
    t0 = time.perf_counter()
    beacon = derive_beacon_from_nonce(
        commitment_hash=commitment.commitment_hash(),
        validator_nonce=nonce,
    )
    beacon_ms = (time.perf_counter() - t0) * 1000

    # ── Phase 5: Derive challenges (Fiat-Shamir) ────────────────────
    t0 = time.perf_counter()
    challenges = _derive_inference_challenges(
        miner=miner,
        commitment=commitment,
        beacon=beacon,
        session_id=session_id,
        proof_protocol_version=proof_protocol_version,
        validator_nonce=nonce,
        router_commitments=session_router_commitments,
        num_input_tokens=len(input_token_ids),
    )
    challenge_ms = (time.perf_counter() - t0) * 1000

    # ── Phase 6: Generate proofs ────────────────────────────────────
    t0 = time.perf_counter()
    proof_bundle, timing_details, _ = miner.generate_proofs(
        commitment,
        challenges,
        validator_nonce=nonce,
        proof_protocol_version=proof_protocol_version,
    )
    prove_ms = (time.perf_counter() - t0) * 1000

    # Clean up session state
    # TODO(concurrency): With the concurrency guard (§3.1), only one request
    #   runs at a time, so cleanup races are impossible.  Without it, this
    #   pop is still safe (session_id is unique) but overlapping requests
    #   accumulate memory until each finishes.
    _cleanup_inference_session(miner, session_id)
    if pending_reveal is not None:
        state.proof_v2_pending_reveals.pop(pending_reveal.challenge_id, None)

    last_token_to_proof_ms = max(
        0.0,
        (time.perf_counter() - t_last_token) * 1000,
    )
    if (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and last_token_to_proof_ms >= PROOF_V2_RESPONSE_TARGET_SECONDS * 1000
    ):
        bt.logging.warning(
            "Proof-v2 missed the post-token latency target: "
            f"total={last_token_to_proof_ms:.3f}ms "
            f"commitment={commitment_ms:.3f}ms reveal={reveal_ms:.3f}ms "
            f"challenge={challenge_ms:.3f}ms prove={prove_ms:.3f}ms"
        )

    # ── Emit final SSE event with commitment + proofs ───────────────
    compact_v2_response = proof_protocol_version == PROOF_PROTOCOL_V2
    done_data = {
        "commitment": ({} if compact_v2_response else commitment_to_dict(commitment)),
        "proof_bundle": proof_bundle_to_dict(
            proof_bundle,
            include_commitment=not compact_v2_response,
        ),
        "output_text": prev_text,
        "input_tokens": len(input_token_ids),
        "output_tokens": len(output_token_ids),
        "inference_ms": round(inference_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "commitment_ms": round(commitment_ms, 1),
        "beacon_ms": round(beacon_ms, 3),
        "challenge_ms": round(challenge_ms, 3),
        "reveal_ms": round(reveal_ms, 3),
        "prove_ms": round(prove_ms, 1),
        "last_token_to_proof_ms": round(last_token_to_proof_ms, 3),
        "prove_timing_details": timing_details,
    }
    _n_out = len(output_token_ids)
    _tps = _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
    bt.logging.info(
        f"Served {session_id[:12]} | {len(input_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms"
    )
    yield f"event: done\ndata: {json.dumps(done_data)}\n\n"


def _normalize_terminal_stop_logits(
    captured_steps,
    *,
    output_token_count: int,
    finish_reason: str,
):
    """Remove one unreturned terminal-stop row from a decode capture."""

    if (
        finish_reason == "stop"
        and isinstance(captured_steps, list)
        and len(captured_steps) == int(output_token_count) + 1
    ):
        return captured_steps[:-1]
    return captured_steps


def _has_runtime_mlp_capture(activations) -> bool:
    """Return whether the active capture profile produced an MLP input."""

    return any(
        key.endswith(("_mlp_gate_input", "_mlp_gate_up_input")) for key in activations
    )


def _proof_v2_gdn_transition_capture_enabled(miner) -> bool:
    """Return whether the authenticated manifest enables the live GDN ABI."""

    manifest = getattr(miner, "proof_v2_manifest", None)
    try:
        from verallm.proof_v2.trace import TRACE_ATTENTION_GDN_TRANSITION_V1

        return any(
            item.attention_profile == TRACE_ATTENTION_GDN_TRANSITION_V1
            for item in getattr(manifest, "layer_execution", ())
        )
    except Exception:
        return False


def _proof_v2_full_attention_transition_capture_enabled(miner) -> bool:
    """Return whether the authenticated manifest enables the live full ABI."""

    manifest = getattr(miner, "proof_v2_manifest", None)
    try:
        from verallm.proof_v2.trace import TRACE_ATTENTION_FULL_TRANSITION_V1

        return any(
            item.attention_profile == TRACE_ATTENTION_FULL_TRANSITION_V1
            for item in getattr(manifest, "layer_execution", ())
        )
    except Exception:
        return False


def _store_inference_witnesses(miner, activations, session_id):
    """Install the request-local layer witnesses used by proof generation."""

    witnesses = {}
    trace_tail = {}
    gdn_boundaries = {}
    full_attention_boundaries = {}
    for key, tensor in activations.items():
        if key.startswith("lm_head_"):
            trace_tail[key] = tensor
            continue
        if key.startswith("proof_v2_gdn_boundary_"):
            encoded = key.removeprefix("proof_v2_gdn_boundary_")
            try:
                layer_text, component = encoded.rsplit("_", 1)
            except ValueError as exc:
                raise RuntimeError(
                    "proof-v2 GDN boundary capture key is malformed"
                ) from exc
            if component not in ("conv", "recurrent"):
                raise RuntimeError("proof-v2 GDN boundary capture key is malformed")
            try:
                layer_idx = int(layer_text)
            except ValueError as exc:
                raise RuntimeError("proof-v2 GDN boundary layer is malformed") from exc
            gdn_boundaries.setdefault(layer_idx, {})[component] = tensor
            continue
        if key.startswith("proof_v2_full_attention_boundary_"):
            encoded = key.removeprefix("proof_v2_full_attention_boundary_")
            try:
                layer_text, component = encoded.rsplit("_", 1)
            except ValueError as exc:
                raise RuntimeError(
                    "proof-v2 full-attention boundary capture key is malformed"
                ) from exc
            if component not in ("keys", "values"):
                raise RuntimeError(
                    "proof-v2 full-attention boundary capture key is malformed"
                )
            try:
                layer_idx = int(layer_text)
            except ValueError as exc:
                raise RuntimeError(
                    "proof-v2 full-attention boundary layer is malformed"
                ) from exc
            full_attention_boundaries.setdefault(layer_idx, {})[component] = tensor
            continue
        if not key.startswith("layer_"):
            continue
        parts = key.split("_")
        layer_idx = int(parts[1])
        witnesses.setdefault(layer_idx, {})[key] = tensor
    miner.witnesses[session_id] = witnesses
    if hasattr(miner, "proof_v2_trace_tails"):
        miner.proof_v2_trace_tails[session_id] = trace_tail
    if hasattr(miner, "proof_v2_gdn_prompt_boundaries"):
        miner.proof_v2_gdn_prompt_boundaries[session_id] = gdn_boundaries
    if hasattr(miner, "proof_v2_full_attention_prompt_boundaries"):
        miner.proof_v2_full_attention_prompt_boundaries[session_id] = (
            full_attention_boundaries
        )
    miner.activation_merkle_trees[session_id] = {}
    return witnesses


def _prepare_proof_v2_x_state_for_commitment(miner, activations, session_id):
    """Freeze the v2 runtime witnesses before the validator nonce is revealed."""

    _store_inference_witnesses(miner, activations, session_id)
    return miner.prepare_proof_v2_x_state(session_id)


def _build_commitment(
    miner,
    activations,
    input_token_ids,
    output_token_ids,
    session_id,
    inference_ms,
    router_commitments=None,
    *,
    do_sample: bool = False,
    temperature: float = 0.0,
    sampling_verification_bps: int = 0,
    presence_penalty: float = 0.0,
    prompt_hash: bytes = b"",
    top_k: int = -1,
    top_p: float = 1.0,
    min_p: float = 0.0,
    sampling_seed: bytes = b"",
    finish_reason: str = "",
    proof_protocol_version: int = LEGACY_PROOF_PROTOCOL_VERSION,
):
    """Build InferenceCommitment from captured activations."""
    t0 = time.perf_counter()
    hard_audit_capture_required = bool(
        proof_protocol_version == PROOF_PROTOCOL_V2
        and _proof_v2_hard_audit_capture_required(miner)
    )

    x_state = None
    if proof_protocol_version == PROOF_PROTOCOL_V2:
        x_state = getattr(miner, "proof_v2_x_states", {}).get(session_id)
        witnesses = miner.witnesses.get(session_id) if x_state is not None else None
        if x_state is not None and witnesses is None:
            raise RuntimeError("proof-v2 prepared state is missing its witnesses")
    else:
        witnesses = None
    if witnesses is None:
        witnesses = _store_inference_witnesses(miner, activations, session_id)

    # Try CUDA-accelerated BLAKE3 for activation Merkle leaves.
    _has_cuda_activation_blake3 = False
    try:
        from zkllm.cuda import zkllm_native as _native

        _has_cuda_activation_blake3 = getattr(_native, "HAS_CUDA", False) and hasattr(
            _native, "cuda_blake3_activation_merkle_leaves"
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
                quantized.contiguous(),
                block_size,
                quantized.device.index or 0,
            )
            num_leaves = leaf_hashes_tensor.shape[0]
            leaf_hash_list = [
                bytes(leaf_hashes_tensor[i].numpy()) for i in range(num_leaves)
            ]
            tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
            return tree, tree.root

        # CPU path: vectorized quantization + numpy slicing
        if quantized.is_cuda:
            quantized = quantized.cpu()
        arr = quantized.numpy()
        leaves = [arr[i : i + block_size].tobytes() for i in range(0, n, block_size)]
        tree = MerkleTree(leaves)
        return tree, tree.root

    num_layers = miner._get_num_layers()
    if proof_protocol_version == PROOF_PROTOCOL_V2:
        if x_state is None:
            x_state = miner.prepare_proof_v2_x_state(session_id)
        if (
            getattr(
                getattr(miner, "proof_v2_manifest", None), "execution_profile", None
            )
            is None
        ):
            raise RuntimeError("proof-v2 causal execution profile is required")
        # The canonical v2 envelope is the sole X commitment set. Legacy layer
        # roots are intentionally empty so there is no second representation.
        layer_commitments = []
    else:
        layer_commitments = []
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

                    miner.activation_merkle_trees[session_id][i] = (
                        tree,
                        tuple(X.shape),
                    )
                    layer_commitments.append(root)
                else:
                    layer_commitments.append(
                        hashlib.sha256(f"layer_{i}_no_input".encode()).digest()
                    )
            else:
                layer_commitments.append(hashlib.sha256(f"layer_{i}".encode()).digest())

    # Decode-integrity: build Merkle trees over hidden rows and logits rows.
    # Built for greedy mode (temp=0) AND for canonical-sampler mode
    # (do_sample=True with sampling_seed committed).  In canonical mode the
    # logits processor masks the GPU logits to force the CPU-chosen token,
    # so the post-mask greedy verification path is also valid.
    decode_hidden_row_root = b""
    hidden_steps = activations.get("lm_head_hidden_steps", [])
    hidden_steps = _normalize_terminal_stop_logits(
        hidden_steps,
        output_token_count=len(output_token_ids),
        finish_reason=finish_reason,
    )
    _canonical_active = bool(do_sample) and bool(sampling_seed)
    is_greedy = (
        (not do_sample) and temperature_to_milli(temperature) == 0
    ) or _canonical_active
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
    logits_steps = _normalize_terminal_stop_logits(
        logits_steps,
        output_token_count=len(output_token_ids),
        finish_reason=finish_reason,
    )
    _v2_decode_required = (
        proof_protocol_version == PROOF_PROTOCOL_V2
        and (
            clamp_sampling_bps(sampling_verification_bps) > 0
            or hard_audit_capture_required
        )
        and len(output_token_ids) > 0
    )
    if _v2_decode_required and (
        not isinstance(hidden_steps, list)
        or len(hidden_steps) != len(output_token_ids)
        or not isinstance(logits_steps, list)
        or len(logits_steps) != len(output_token_ids)
    ):
        raise RuntimeError("proof-v2 decode capture does not match the output length")
    if _canonical_active:
        if isinstance(logits_steps, list) and len(logits_steps) != len(
            output_token_ids
        ):
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
            _logits_aligned = proof_protocol_version == PROOF_PROTOCOL_V2
            bt.logging.debug(
                f"Sampling divergence: {len(_mismatches)}/{_n_check} steps"
            )
        elif len(logits_steps) != len(output_token_ids):
            _logits_aligned = False
            bt.logging.warning(
                f"Logits/output length mismatch: logits_steps={len(logits_steps)}, output_tokens={len(output_token_ids)}. "
                f"Skipping decode commitment."
            )

    if (
        isinstance(hidden_steps, list)
        and hidden_steps
        and is_greedy
        and _logits_aligned
    ):
        _t_hm = time.perf_counter()
        hidden_tree, hidden_rows, decode_hidden_row_root = build_hidden_row_merkle(
            hidden_steps
        )
        _t_hm = (time.perf_counter() - _t_hm) * 1000
        if hidden_tree is not None:
            miner.decode_hidden_row_trees[session_id] = hidden_tree
            miner.decode_hidden_rows[session_id] = hidden_rows
            bt.logging.debug(
                f"Hidden row Merkle: {_t_hm:.1f}ms, {len(hidden_steps)} steps"
            )

    decode_logits_row_root = b""
    bps = clamp_sampling_bps(sampling_verification_bps)
    # V2 uses the compact committed top-k row for both greedy and sampled
    # response consistency, so it must be available whenever the request can
    # draw a sampling challenge. V1 keeps its existing high-assurance gate.
    _need_fp32_tree = (
        bps >= HIGH_ASSURANCE_BPS
        or _canonical_active
        or (proof_protocol_version == PROOF_PROTOCOL_V2 and bps > 0)
        or hard_audit_capture_required
    )
    if (
        isinstance(logits_steps, list)
        and logits_steps
        and is_greedy
        and _need_fp32_tree
        and _logits_aligned
    ):
        _t_lm = time.perf_counter()
        logits_tree, logits_rows, decode_logits_row_root = build_logits_row_merkle(
            logits_steps
        )
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

    transition_hashes = (
        []
        if proof_protocol_version == PROOF_PROTOCOL_V2
        else _MinerBase.compute_layer_transition_hashes(
            layer_commitments,
            input_commitment,
        )
    )

    # Compute sampler config hash from actual sampling params used.
    # NOTE: chat_template_hash is intentionally NOT included here — until the
    # on-chain registry stores it, the validator cannot compute a matching
    # expected hash.  Template binding is a future enhancement.
    from verallm.sampling import compute_sampler_config_hash as _compute_scfg

    sampler_cfg_hash = _compute_scfg(
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
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
        router_commitment_hash=InferenceCommitment.compute_router_hash(
            router_commitments or {}
        ),
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
        proof_v2_commitment=(
            miner.proof_v2_x_states[session_id].envelope.canonical_bytes()
            if proof_protocol_version == PROOF_PROTOCOL_V2
            else b""
        ),
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
                [int(expert_idx) for expert_idx in row] for row in proof_selected
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
            failed_ids = batch_engine.fail_all_requests(
                BatchEngineRequestError(
                    "Inference engine step failed; request was aborted"
                )
            )
            if failed_ids:
                bt.logging.error(
                    f"Aborted {len(failed_ids)} request(s) after shared "
                    "engine step failure"
                )
            await asyncio.sleep(0.01)
        # Yield to event loop so SSE generators can send queued outputs
        await asyncio.sleep(0)


async def _stream_inference_batched(
    body: "InferenceRequestBody",
    nonce: Optional[bytes],
    prompt_token_ids: list[int] | None = None,
    token_budget: int = 0,
    admitted_request_id: str | None = None,
    prompt_hash: bytes = b"",
    validator_hotkey: str = "",
    proof_protocol_version: int = LEGACY_PROOF_PROTOCOL_VERSION,
    proof_v3_precommit_context=None,
    proof_v3_tracker_options: Optional[dict[str, Any]] = None,
):
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
        nonce: Validator nonce bytes for v1. V2 reveals it after precommit.
        prompt_token_ids: If provided, use these token IDs directly.
        token_budget: KV cache tokens to reserve (prompt + max_new_tokens).
        admitted_request_id: If provided, admission was already done at
            endpoint level — skip try_admit and use this request_id.
        proof_protocol_version: Resolved proof version (omitted requests are v1).
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
            error_data = json.dumps(
                {
                    "error": "Miner busy: KV cache full",
                    "free_tokens": s.free_tokens,
                    "requested_tokens": token_budget,
                    "active_requests": s.active_requests,
                    "retry_after_ms": 5000,
                }
            )
            yield f"event: error\ndata: {error_data}\n\n"
            return

    engine_request_active = False
    try:
        # Register with activation tracker (before first engine step).
        # Capture logits for high-assurance v1 checks, canonical sampled
        # decoding, and every v2 request eligible for a manifest-signed hard
        # execution audit.  The latter cannot depend on caller-visible BPS.
        bps_for_request = clamp_sampling_bps(body.sampling_verification_bps)
        require_hard_audit_capture = (
            proof_protocol_version == PROOF_PROTOCOL_V2
            and _proof_v2_hard_audit_capture_required(miner)
        )
        need_logits = _request_needs_logits_capture(
            do_sample=body.do_sample,
            sampling_bps=bps_for_request,
            proof_protocol_version=proof_protocol_version,
            require_hard_audit_capture=require_hard_audit_capture,
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
        if body.do_sample and (
            bps_for_request > 0 or require_hard_audit_capture
        ):
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
            if proof_protocol_version == PROOF_PROTOCOL_V3:
                if (
                    state.proof_v3_runtime is None
                    or proof_v3_precommit_context is None
                    or not isinstance(proof_v3_tracker_options, dict)
                ):
                    raise RuntimeError(
                        "proof-v3 request reached serving without a validated "
                        "capture plan"
                    )
                tracker.register_request(
                    request_id,
                    session_id,
                    **proof_v3_tracker_options,
                )
            else:
                tracker.register_request(
                    request_id,
                    session_id,
                    capture_logits=need_logits or (_seed_bytes is not None),
                    capture_full_trace=(
                        proof_protocol_version == PROOF_PROTOCOL_V2
                        and getattr(
                            getattr(miner, "proof_v2_manifest", None),
                            "execution_profile",
                            None,
                        )
                        is not None
                    ),
                    capture_gdn_transition=(
                        proof_protocol_version == PROOF_PROTOCOL_V2
                        and _proof_v2_gdn_transition_capture_enabled(miner)
                    ),
                    capture_full_attention_transition=(
                        proof_protocol_version == PROOF_PROTOCOL_V2
                        and _proof_v2_full_attention_transition_capture_enabled(
                            miner
                        )
                    ),
                    canonical_seed=_seed_bytes,
                    canonical_temperature=max(
                        0.001,
                        float(body.temperature) if body.temperature else 1.0,
                    )
                    if _seed_bytes
                    else 1.0,
                    canonical_top_k=(
                        int(body.top_k) if body.top_k is not None else -1
                    ),
                    canonical_top_p=(
                        float(body.top_p) if body.top_p is not None else 1.0
                    ),
                    canonical_min_p=(
                        float(body.min_p) if body.min_p is not None else 0.0
                    ),
                )

        # Resolve sampling params and build canonical sampler if do_sample=True.
        _resolved_sp = _resolve_sampling_params(body, state.model_name)
        if body.do_sample:
            _resolved_sp["seed"] = _private_vllm_sampling_seed(_seed_bytes)

        sampling_params = SamplingParams(**_resolved_sp)
        # vLLM owns the object passed to add_request.  Keep an independent,
        # pristine copy for a possible post-commit hard replay.
        proof_v3_replay_sampling_params = (
            sampling_params.clone()
            if proof_protocol_version == PROOF_PROTOCOL_V3
            else None
        )
        if prompt_token_ids is not None:
            # Keep the proof transcript's canonical immutable sequence at the
            # protocol boundary, but satisfy vLLM's TokensPrompt ABI here.
            prompt = {
                "prompt_token_ids": [
                    int(token_id) for token_id in prompt_token_ids
                ]
            }
        else:
            prompt = body.prompt

        t_infer = time.perf_counter()
        retain_finished_cache_for_proof = bool(
            proof_protocol_version == PROOF_PROTOCOL_V3
            and isinstance(proof_v3_tracker_options, dict)
            and (
                proof_v3_tracker_options.get("capture_prefix_cache", False)
                or proof_v3_tracker_options.get(
                    "capture_gdn_execution_checkpoints",
                    False,
                )
            )
        )
        output_queue = batch_engine.add_request(
            request_id,
            prompt,
            sampling_params,
            retain_finished_cache_for_proof=(
                retain_finished_cache_for_proof
            ),
            proof_cache_provenance_token_limit=(
                int(proof_v3_tracker_options[
                    "prefix_cache_provenance_token_limit"
                ])
                if proof_v3_tracker_options.get(
                    "capture_prefix_cache",
                    False,
                )
                else 0
                if retain_finished_cache_for_proof
                else None
            ),
        )
        engine_request_active = True

        # Stream tokens from per-request queue
        prev_text = ""
        final_output = None
        t_first_token = None
        t_last_token = None
        t_inference_finished = None
        pending_reveal = None
        activation_finalize_future = None
        prepared_v3_precommit = None
        proof_v2_prepare_ms = 0.0
        prev_token_count = 0

        def _finalize_activations_for_proof():
            captured = tracker.finalize_activations(request_id)
            prepare_ms = 0.0
            if proof_protocol_version == PROOF_PROTOCOL_V2:
                prepare_started = time.perf_counter()
                _prepare_proof_v2_x_state_for_commitment(
                    miner,
                    captured,
                    session_id,
                )
                prepare_ms = (time.perf_counter() - prepare_started) * 1000
            return captured, prepare_ms

        while True:
            try:
                output = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=_BATCH_OUTPUT_IDLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield f"event: error\ndata: {json.dumps({'error': 'Inference timeout'})}\n\n"
                batch_engine.abort_request(request_id)
                engine_request_active = False
                if tracker is not None:
                    tracker.unregister_request(request_id)
                if moe_mgr:
                    moe_mgr.clear_request(request_id)
                return

            if isinstance(output, BatchEngineRequestError):
                # The shared step loop already aborted every affected engine
                # request before publishing this bounded error.
                engine_request_active = False
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': str(output)})}\n\n"
                )
                if tracker is not None:
                    tracker.unregister_request(request_id)
                if moe_mgr:
                    moe_mgr.clear_request(request_id)
                _cleanup_inference_session(miner, session_id)
                return

            cur_text = output.outputs[0].text if output.outputs else ""
            delta = cur_text[len(prev_text) :]
            current_token_ids = (
                list(output.outputs[0].token_ids)
                if output.outputs
                else []
            )
            token_delta = current_token_ids[prev_token_count:]
            if delta or token_delta:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                t_last_token = time.perf_counter()
            if output.finished:
                # BatchAwareEngine removes a finished request before putting
                # this terminal output on the request queue.
                engine_request_active = False
                t_inference_finished = time.perf_counter()
                final_output = output
                miner.input_token_ids[session_id] = [
                    int(value) for value in output.prompt_token_ids
                ]
                miner.output_token_ids[session_id] = [
                    int(value) for value in output.outputs[0].token_ids
                ]
                if tracker is not None and tracker.can_finalize_in_background(
                    request_id
                ) and proof_protocol_version != PROOF_PROTOCOL_V3:
                    activation_finalize_future = (
                        asyncio.get_running_loop().run_in_executor(
                            None,
                            _finalize_activations_for_proof,
                        )
                    )
                if proof_protocol_version == PROOF_PROTOCOL_V3:
                    # The authenticated output and graph-native capture are
                    # frozen before this final token is exposed. Finish the
                    # nonce-free precommit against that immutable state before
                    # handing off the final token. The wire still publishes
                    # token, precommit, done in that order, while a wide
                    # co-batch cannot consume the signed post-token deadline
                    # in Python worker scheduling.
                    try:
                        prepared_v3_precommit = await (
                            state.proof_v3_runtime.finalize_initial_request(
                                request_id=request_id,
                                precommit_context=proof_v3_precommit_context,
                                prompt_token_ids=output.prompt_token_ids,
                                output_token_ids=output.outputs[0].token_ids,
                                emitted_text_utf8=cur_text.encode("utf-8"),
                                finish_reason=str(
                                    output.outputs[0].finish_reason or ""
                                ),
                                sampling_params=proof_v3_replay_sampling_params,
                                requested_decode_tokens=body.max_new_tokens,
                                hard_proof_arrival_budget_ns=(
                                    body.proof_v3_hard_proof_arrival_budget_ns
                                ),
                            )
                        )
                    except Exception as exc:
                        bt.logging.error(
                            "Proof-v3 precommit failed before final-token "
                            f"delivery for {request_id}: {exc}"
                        )
                        batch_engine.clear_finished(request_id)
                        if tracker is not None:
                            tracker.unregister_request(request_id)
                        if moe_mgr:
                            moe_mgr.clear_request(request_id)
                        _cleanup_inference_session(miner, session_id)
                        yield (
                            "event: error\n"
                            f"data: {json.dumps({'error': f'Post-inference error: {exc}'})}\n\n"
                        )
                        return
            if delta or (
                proof_protocol_version == PROOF_PROTOCOL_V3 and token_delta
            ):
                token_data = {"text": delta}
                if proof_protocol_version == PROOF_PROTOCOL_V3:
                    token_data["token_ids"] = [
                        int(value) for value in token_delta
                    ]
                yield f"event: token\ndata: {json.dumps(token_data)}\n\n"
            prev_text = cur_text
            prev_token_count = len(current_token_ids)

            if output.finished:
                break

        inference_ms = (
            (t_inference_finished or time.perf_counter()) - t_infer
        ) * 1000
        ttft_ms = ((t_first_token - t_infer) * 1000) if t_first_token else 0

        if final_output is None:
            yield f"event: error\ndata: {json.dumps({'error': 'No output generated'})}\n\n"
            tracker.unregister_request(request_id)
            if moe_mgr:
                moe_mgr.clear_request(request_id)
            return

        if t_last_token is None:
            t_last_token = time.perf_counter()

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
        _skip_proofs = (
            state.tee_skip_proofs or os.environ.get("VERALLM_SKIP_CAPTURE", "0") == "1"
        )
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
            _tps = (
                _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
            )
            bt.logging.info(
                f"Served {request_id} | {len(final_output.prompt_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms"
            )
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
            if proof_protocol_version == PROOF_PROTOCOL_V3:
                if prepared_v3_precommit is None:
                    raise RuntimeError(
                        "proof-v3 capture finalization did not complete"
                    )
                prepared = prepared_v3_precommit
                last_token_to_precommit_ms = max(
                    0.0,
                    (time.perf_counter() - t_last_token) * 1000,
                )
                precommit_data = {
                    "proof_protocol_version": PROOF_PROTOCOL_V3,
                    "proof_challenge_id": (
                        proof_v3_precommit_context.proof_challenge_id.hex()
                    ),
                    "commitment_envelope": prepared.envelope_bytes.hex(),
                    "last_token_to_precommit_ms": round(
                        last_token_to_precommit_ms,
                        3,
                    ),
                }
                yield (
                    "event: proof_precommit\n"
                    f"data: {json.dumps(precommit_data)}\n\n"
                )
                done_data = {
                    "proof_protocol_version": PROOF_PROTOCOL_V3,
                    "output_text": prev_text,
                    "finish_reason": str(
                        final_output.outputs[0].finish_reason or ""
                    ),
                    "input_tokens": len(input_token_ids),
                    "output_tokens": len(output_token_ids),
                    "inference_ms": round(inference_ms, 1),
                    "ttft_ms": round(ttft_ms, 1),
                    "last_token_to_precommit_ms": round(
                        last_token_to_precommit_ms,
                        3,
                    ),
                }
                yield f"event: done\ndata: {json.dumps(done_data)}\n\n"
                return

            # Finished buffer-mode requests are snapshotted on the GPU by the
            # engine callback before their final output enters this queue. The
            # CPU transfer starts before the final text delta is yielded, so it
            # can overlap normal token delivery without delaying or weakening
            # the transcript. Other backends already own request-local clones.
            if activation_finalize_future is not None:
                activations, proof_v2_prepare_ms = await activation_finalize_future
            elif tracker.has_capture_buffers:
                # Safety fallback if the engine callback could not identify the
                # request. Preserve the old immediate readout behavior.
                activations = tracker.finalize_activations(request_id)
                if proof_protocol_version == PROOF_PROTOCOL_V2:
                    prepare_started = time.perf_counter()
                    _prepare_proof_v2_x_state_for_commitment(
                        miner,
                        activations,
                        session_id,
                    )
                    proof_v2_prepare_ms = (time.perf_counter() - prepare_started) * 1000
            else:
                activations = None  # finalize in executor (no buffer race)

            # Post-inference work (router → commitment → beacon → challenges)
            # runs off the event loop in one executor call.  This keeps the
            # event loop free for engine stepping and token delivery.
            def _postprocess_sync():
                nonlocal activations, proof_v2_prepare_ms
                if activations is None:
                    activations = tracker.finalize_activations(request_id)
                    if proof_protocol_version == PROOF_PROTOCOL_V2:
                        prepare_started = time.perf_counter()
                        _prepare_proof_v2_x_state_for_commitment(
                            miner,
                            activations,
                            session_id,
                        )
                        proof_v2_prepare_ms = (
                            time.perf_counter() - prepare_started
                        ) * 1000

                if tracker.backend == "splitting_ops":
                    if not _has_runtime_mlp_capture(activations):
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
                            moe_mgr.process_captured_router_logits(
                                request_id, layer_idx
                            )

                    from verallm.crypto.field import P as FIELD_PRIME

                    for layer_idx in moe_mgr.get_challenged_layers():
                        rc = moe_mgr.build_router_commitment_for_request(
                            request_id, layer_idx, FIELD_PRIME
                        )
                        if rc is not None:
                            session_router_commitments[layer_idx] = rc
                        decision = moe_mgr.get_router_decision_for_request(
                            request_id, layer_idx
                        )
                        if decision is not None:
                            session_router_logits[layer_idx] = (
                                decision.router_logits.detach().float().cpu()
                            )
                    moe_mgr.clear_request(request_id)
                    bt.logging.debug(
                        f"Built {len(session_router_commitments)} router commitments for {request_id} (backend={tracker.backend})"
                    )
                    if not session_router_commitments:
                        if tracker.backend == "splitting_ops":
                            router_keys = sorted(
                                k
                                for k in activations.keys()
                                if k.endswith("_router_logits")
                            )
                            raise RuntimeError(
                                "splitting_ops router capture backend produced no router "
                                f"logits for request {request_id}; available router keys="
                                f"{router_keys[:8]}"
                            )
                        else:
                            mlp_keys = sorted(
                                k
                                for k in activations.keys()
                                if k.endswith("_mlp_gate_input")
                            )
                            raise RuntimeError(
                                "hooks router recompute produced no router commitments "
                                f"for request {request_id}; available mlp keys={mlp_keys[:8]}"
                            )

                _attach_proof_domain_router_topk(
                    miner, activations, session_router_commitments
                )
                if tracker is not None:
                    tracker.unregister_request(request_id)

                _resolved = _resolve_sampling_params(body, state.model_name)
                resolved_pp = _resolved["presence_penalty"]
                _pending = getattr(miner, "_pending_sampling_seeds", {}) or {}
                _seed_for_commit = _pending.pop(session_id, b"")
                commitment, commitment_tail_ms = _build_commitment(
                    miner,
                    activations,
                    input_token_ids,
                    output_token_ids,
                    session_id,
                    inference_ms,
                    router_commitments=session_router_commitments,
                    do_sample=body.do_sample,
                    temperature=float(_resolved["temperature"]),
                    sampling_verification_bps=body.sampling_verification_bps,
                    presence_penalty=resolved_pp,
                    prompt_hash=prompt_hash,
                    top_k=int(_resolved.get("top_k", -1) or -1),
                    top_p=float(_resolved.get("top_p", 1.0) or 1.0),
                    min_p=float(_resolved.get("min_p", 0.0) or 0.0),
                    sampling_seed=_seed_for_commit,
                    finish_reason=str(final_output.outputs[0].finish_reason or ""),
                    proof_protocol_version=proof_protocol_version,
                )
                commitment_ms = commitment_tail_ms + proof_v2_prepare_ms
                if session_router_commitments:
                    miner.router_commitments[session_id] = session_router_commitments
                if session_router_logits:
                    miner.router_logits[session_id] = session_router_logits
                miner.input_token_ids[session_id] = list(input_token_ids)

                return commitment, commitment_ms, session_router_commitments

            loop = asyncio.get_event_loop()
            (
                commitment,
                commitment_ms,
                session_router_commitments,
            ) = await loop.run_in_executor(None, _postprocess_sync)

            reveal_ms = 0.0
            if proof_protocol_version == PROOF_PROTOCOL_V2:
                challenge_id = body.proof_challenge_id_bytes
                pending_reveal = _register_proof_v2_reveal(
                    challenge_id=challenge_id,
                    commitment_hash=commitment.commitment_hash(),
                    deadline_at=(time.perf_counter() + PROOF_V2_REVEAL_TIMEOUT_SECONDS),
                    nonce_commitment=body.validator_nonce_commitment_bytes,
                    session_id=session_id,
                    validator_hotkey=validator_hotkey,
                )
                precommit_data = _proof_v2_precommit_data(
                    challenge_id=challenge_id,
                    commitment=commitment,
                    last_token_at=t_last_token or time.perf_counter(),
                )
                yield (
                    "event: proof_precommit\n" f"data: {json.dumps(precommit_data)}\n\n"
                )
                yield (
                    "event: proof_commitment\n"
                    f"data: {json.dumps({'commitment': commitment_to_dict(commitment)})}\n\n"
                )
                reveal_started = time.perf_counter()
                nonce = await _await_proof_v2_reveal(pending_reveal)
                reveal_ms = (time.perf_counter() - reveal_started) * 1000

            if nonce is None:
                raise RuntimeError("validator nonce is unavailable")

            t0 = time.perf_counter()
            beacon = derive_beacon_from_nonce(
                commitment_hash=commitment.commitment_hash(),
                validator_nonce=nonce,
            )
            beacon_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            challenges = _derive_inference_challenges(
                miner=miner,
                commitment=commitment,
                beacon=beacon,
                session_id=session_id,
                proof_protocol_version=proof_protocol_version,
                validator_nonce=nonce,
                router_commitments=session_router_commitments,
                num_input_tokens=len(input_token_ids),
            )
            challenge_ms = (time.perf_counter() - t0) * 1000

            # Submit proof to background pipeline.

            t0 = time.perf_counter()
            if os.environ.get("VERALLM_SKIP_PROOFS"):
                from verallm.crypto.proof import ProofBundle

                proof_bundle = ProofBundle(layer_proofs=[], sampling_proofs=[])
                timing_details = {}
            else:
                await proof_pipeline.submit_proof(
                    session_id,
                    miner,
                    commitment,
                    challenges,
                    nonce,
                    proof_protocol_version=proof_protocol_version,
                )
                proof_bundle, timing_details, _ = await proof_pipeline.await_proof(
                    session_id, miner=miner
                )
            prove_ms = (time.perf_counter() - t0) * 1000

            # Clean up session state
            _cleanup_inference_session(miner, session_id)

            last_token_to_proof_ms = max(
                0.0,
                (time.perf_counter() - t_last_token) * 1000,
            )
            if (
                proof_protocol_version == PROOF_PROTOCOL_V2
                and last_token_to_proof_ms >= PROOF_V2_RESPONSE_TARGET_SECONDS * 1000
            ):
                bt.logging.warning(
                    "Proof-v2 missed the post-token latency target: "
                    f"total={last_token_to_proof_ms:.3f}ms "
                    f"commitment={commitment_ms:.3f}ms reveal={reveal_ms:.3f}ms "
                    f"challenge={challenge_ms:.3f}ms prove={prove_ms:.3f}ms"
                )

            # Emit final SSE event
            compact_v2_response = proof_protocol_version == PROOF_PROTOCOL_V2
            done_data = {
                "commitment": (
                    {} if compact_v2_response else commitment_to_dict(commitment)
                ),
                "proof_bundle": proof_bundle_to_dict(
                    proof_bundle,
                    include_commitment=not compact_v2_response,
                ),
                "output_text": prev_text,
                "input_tokens": len(input_token_ids),
                "output_tokens": len(output_token_ids),
                "inference_ms": round(inference_ms, 1),
                "commitment_ms": round(commitment_ms, 1),
                "beacon_ms": round(beacon_ms, 3),
                "challenge_ms": round(challenge_ms, 3),
                "reveal_ms": round(reveal_ms, 3),
                "prove_ms": round(prove_ms, 1),
                "last_token_to_proof_ms": round(last_token_to_proof_ms, 3),
                "prove_timing_details": timing_details,
            }
            _n_out = len(output_token_ids)
            _tps = (
                _n_out / (inference_ms / 1000) if inference_ms > 0 and _n_out > 0 else 0
            )
            bt.logging.info(
                f"Served {request_id} | {len(input_token_ids)}→{_n_out} tokens | {_tps:.1f} tok/s | {inference_ms:.0f}ms"
            )
            if validator_hotkey:
                bt.logging.debug(f"  └─ validator: {validator_hotkey}")
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            bt.logging.error(
                f"Post-inference error for {request_id}: {e}\n{tb}"
            )
            # Always clean up captured session artifacts on failure to avoid
            # memory growth under bursty proof backlogs.
            _cleanup_inference_session(miner, session_id)

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
            if pending_reveal is not None:
                state.proof_v2_pending_reveals.pop(
                    pending_reveal.challenge_id,
                    None,
                )
            # Ensure tracker state is always cleaned up.
            tracker.unregister_request(request_id)

    finally:
        # Closing an SSE response cancels this generator.  Admission cleanup
        # alone is insufficient: without an explicit abort, vLLM can continue
        # an invisible generation after the client has gone away, consuming
        # scheduler/KV capacity while /health reports the reservation free.
        if engine_request_active:
            try:
                batch_engine.abort_request(request_id)
            except Exception:
                bt.logging.exception(
                    f"Failed to abort disconnected request {request_id}"
                )
            if tracker is not None:
                tracker.unregister_request(request_id)
            if moe_mgr:
                moe_mgr.clear_request(request_id)
            _cleanup_inference_session(miner, session_id)
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
                my_gpu_uuids.add(
                    _norm_uuid(str(torch.cuda.get_device_properties(i).uuid))
                )
            except Exception:
                pass
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
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
        free_gb = free_mem[0] / (1024**3)
        total_gb = free_mem[1] / (1024**3)
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

    platform = getattr(args, "tee_platform", "mock")
    skip_proofs = getattr(args, "tee_skip_proofs", None)
    if skip_proofs is None:
        # Default: skip proofs when TEE is enabled (hardware attestation replaces them)
        skip_proofs = True

    bt.logging.info(f"TEE mode: {platform}")
    bt.logging.info(
        f"Proof mode: {'attestation (proofs disabled)' if skip_proofs else 'verallm (proofs enabled)'}"
    )

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
        bt.logging.warning(
            f"TEE: could not compute weight_file_hash ({e}), using empty"
        )

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

    bt.logging.info(
        f"Weight Merkle root: {model_spec.weight_merkle_root.hex()[:16]}..."
    )
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
        indices = sorted(
            {
                p.device.index
                for p in model.parameters()
                if p.is_cuda and p.device.index is not None
            }
        )
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


def _proof_v2_execution_capture_requested(args) -> bool:
    """Detect configured v2 artifact discovery before vLLM graph creation.

    This is only an instrumentation decision.  Artifacts are authenticated by
    ``_configure_proof_v2_artifacts`` after the local ModelSpec is available.
    """

    if str(
        getattr(args, "proof_v2_manifest", None)
        or os.environ.get("VERATHOS_PROOF_V2_MANIFEST", "")
    ).strip():
        return True

    from verallm.proof_v2.artifact_store import (
        configured_proof_v2_artifact_base_urls,
    )

    if configured_proof_v2_artifact_base_urls(
        getattr(args, "proof_v2_artifact_base_url", None)
    ):
        return True
    chain_config_path = getattr(args, "chain_config", None)
    if not chain_config_path:
        return False
    try:
        from verallm.chain.config import ChainConfig
        from verallm.proof_v2.runtime import bundled_proof_v2_manifest_paths

        chain_config = ChainConfig.from_json(
            chain_config_path,
            **(
                {"rpc_url": getattr(args, "evm_rpc_url", None)}
                if getattr(args, "evm_rpc_url", None)
                else {}
            ),
        )
        if configured_proof_v2_artifact_base_urls(
            getattr(args, "proof_v2_artifact_base_url", None),
            default_values=chain_config.proof_v2_artifact_base_urls,
        ):
            return True
        return bool(
            bundled_proof_v2_manifest_paths(
                chain_id=chain_config.chain_id,
                netuid=chain_config.netuid,
            )
        )
    except Exception as exc:
        bt.logging.warning(
            "Could not preflight proof-v2 execution capture sources: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def _proof_v3_execution_capture_requested(args) -> bool:
    """Return whether an explicit v3 release needs graph-time capture."""

    return bool(
        str(
            getattr(args, "proof_v3_manifest", None)
            or os.environ.get("VERATHOS_PROOF_V3_MANIFEST", "")
        ).strip()
    )


def _proof_execution_capture_modes(
    args,
    *,
    allowed_protocol_versions: tuple[int, ...],
) -> tuple[bool, bool]:
    """Return (complete trace capture, legacy full-attention state capture)."""

    proof_v2 = _proof_v2_execution_capture_requested(args)
    proof_v3 = _proof_v3_execution_capture_requested(args)
    # Artifact discovery is not protocol activation.  V2 artifacts may still
    # be listed in a chain config while a v3-only miner runs with protocol 2
    # explicitly disabled.  Only an allowed v2 wire may install its Python
    # paged-K/V wrapper into the model compiled by TorchDynamo.
    return (
        proof_v2 or proof_v3,
        proof_v2 and PROOF_PROTOCOL_V2 in allowed_protocol_versions,
    )


def _legacy_weight_merkle_startup_required(
    *,
    proof_v3_configured: bool,
    allowed_protocol_versions: tuple[int, ...],
) -> bool:
    """Return whether this server can negotiate a legacy v1 proof."""

    return not (
        proof_v3_configured
        and LEGACY_PROOF_PROTOCOL_VERSION not in allowed_protocol_versions
    )


def _load_v3_required_onchain_model_spec(args, model_name: str):
    """Load the chain ModelSpec when authenticated v3 is the sole protocol.

    Compact v3 authenticates the exact static projection catalog before
    serving, then binds challenge-selected runtime rows to that catalog in
    each hard proof. Rebuilding the backend-coupled legacy ModelSpec roots is
    unnecessary when v1 cannot be negotiated and is incorrect for compressed
    checkpoints whose canonical v3 source differs from vLLM's historical
    compatibility view.
    """

    chain_config_path = str(
        getattr(args, "chain_config", None) or ""
    ).strip()
    if not chain_config_path:
        raise RuntimeError(
            "v3_required startup needs an authenticated chain config"
        )

    from verallm.chain.config import ChainConfig
    from verallm.chain.model_registry import ModelRegistryClient

    rpc_override = getattr(args, "evm_rpc_url", None)
    chain_config = ChainConfig.from_json(
        chain_config_path,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )
    model_spec = ModelRegistryClient(chain_config).get_model_spec(model_name)
    if model_spec is None:
        raise RuntimeError(
            f"proof-v3 model {model_name!r} is not registered on-chain"
        )
    return model_spec


MAX_PROOF_V3_MANIFEST_ARTIFACT_BYTES = 64 << 20


def _prepare_proof_v3_graph_capture(args, *, expected_model_id: str):
    """Configure the exact model-build capture layout for a v3 release.

    This preflight deliberately does not establish trust in the artifact; the
    complete signature/ModelRegistry qualification still runs after model
    load.  It only derives bounded allocation geometry and the exact
    full-attention layer inventory needed before vLLM constructs CUDA graphs.
    """

    manifest_path = str(
        getattr(args, "proof_v3_manifest", None)
        or os.environ.get("VERATHOS_PROOF_V3_MANIFEST", "")
    ).strip()
    if not manifest_path:
        return None

    from pathlib import Path

    from verallm.proof_v3.capture_staging import (
        recommended_dense_capture_staging_rows_v3,
    )
    from verallm.proof_v3.economic_profile import (
        infer_economic_manifest_layer_kinds_v3,
    )
    from verallm.proof_v3.errors import ProofV3Error
    from verallm.proof_v3.projection_manifest import ProjectionManifestV3

    path = Path(manifest_path)
    try:
        if path.stat().st_size > MAX_PROOF_V3_MANIFEST_ARTIFACT_BYTES:
            raise ProofV3Error("proof-v3 manifest artifact is too large")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        manifest = ProjectionManifestV3.from_json(
            json.dumps(
                artifact["manifest"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            "proof-v3 graph-capture manifest could not be loaded"
        ) from exc
    if manifest.model_id != expected_model_id:
        raise RuntimeError(
            "proof-v3 graph-capture manifest does not match the model"
        )

    profile_path = str(
        getattr(args, "proof_v3_execution_profile", None)
        or os.environ.get("VERATHOS_PROOF_V3_EXECUTION_PROFILE", "")
    ).strip()
    lean_capture = False
    prefix_cache_sharing = False
    if profile_path:
        from verallm.proof_v3.document import (
            load_signed_execution_profile_document_v3,
        )
        from verallm.proof_v3.economic_profile import (
            ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
            economic_profile_is_lean_v3,
        )

        profile = load_signed_execution_profile_document_v3(
            profile_path
        ).profile
        if profile.static_manifest_digest != manifest.digest():
            raise RuntimeError(
                "proof-v3 graph-capture profile does not match the manifest"
            )
        if profile.adapter_version not in {
            ECONOMIC_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_LEAN_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_COMPACT_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_COMPACT_ONLY_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_SELECTED_TRACE_PROFILE_ADAPTER_VERSION_V3,
            ECONOMIC_SELECTED_TRACE_ESCALATION_PROFILE_ADAPTER_VERSION_V3,
        }:
            raise RuntimeError(
                "proof-v3 graph-capture profile adapter is unsupported"
            )
        lean_capture = economic_profile_is_lean_v3(profile)
        relation_spec = getattr(profile, "relation_spec", None)
        prefix_cache_sharing = bool(
            getattr(
                getattr(relation_spec, "cache", None),
                "allows_prefix_cache_sharing",
                False,
            )
        )

    layer_kinds = infer_economic_manifest_layer_kinds_v3(manifest)
    full_attention_layers = tuple(
        index
        for index, kind in enumerate(layer_kinds)
        if kind == "full_attention"
    )
    staging_rows = recommended_dense_capture_staging_rows_v3(manifest)
    legacy_full_rows = str(
        os.environ.get("VERALLM_CAPTURE_FULL_ROWS", "")
    ).strip()
    if legacy_full_rows and legacy_full_rows != "0":
        raise RuntimeError(
            "VERALLM_CAPTURE_FULL_ROWS conflicts with compact proof-v3 capture"
        )
    expected_environment = {
        "VERALLM_CAPTURE_ROOT_ROWS": "1",
        "VERALLM_CAPTURE_GATHER_ROWS": str(staging_rows),
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    }
    if lean_capture:
        from verallm.proof_v3.lean_execution_anchor import (
            LEAN_EXECUTION_CHECKPOINT_STRIDE_V3,
        )

        expected_environment.update(
            {
                "VERALLM_CAPTURE_ROOT_SUFFIXES": (
                    "attention_kv_output,residual_out"
                ),
                "VERALLM_CAPTURE_ROOT_CHECKPOINT_STRIDE": str(
                    LEAN_EXECUTION_CHECKPOINT_STRIDE_V3
                ),
            }
        )
    else:
        for name in (
            "VERALLM_CAPTURE_ROOT_SUFFIXES",
            "VERALLM_CAPTURE_ROOT_CHECKPOINT_STRIDE",
        ):
            if str(os.environ.get(name, "")).strip():
                raise RuntimeError(
                    f"{name} conflicts with the qualified proof-v3 "
                    "capture plan"
                )
    if full_attention_layers:
        expected_environment["VERALLM_REDUCTION_AUDIT_LAYERS"] = ",".join(
            str(layer) for layer in full_attention_layers
        )
    for name, expected in expected_environment.items():
        existing = str(os.environ.get(name, "")).strip()
        if existing and existing != expected:
            raise RuntimeError(
                f"{name} conflicts with the qualified proof-v3 capture plan"
            )
        os.environ[name] = expected

    setattr(args, "_proof_v3_capture_staging_rows", staging_rows)
    setattr(args, "_proof_v3_full_attention_layers", full_attention_layers)
    setattr(args, "_proof_v3_layer_kinds", layer_kinds)
    setattr(args, "_proof_v3_lean_capture", lean_capture)
    setattr(
        args,
        "_proof_v3_prefix_cache_sharing",
        prefix_cache_sharing,
    )
    return manifest


def _proof_v3_coordinator_record_capacity(
    *,
    configured_max_records: int,
    max_admitted_requests: int,
) -> int:
    """Reserve one completed-proof retry slot beyond live admission.

    A successfully returned hard proof remains cached for the bounded HTTP
    retry window.  Counting that record against the same ceiling as live
    precommits otherwise makes the next full-width scheduler batch fail after
    inference, even though capture and request admission both had capacity.
    """

    configured = int(configured_max_records)
    admitted = int(max_admitted_requests)
    if configured <= 0 or admitted <= 0:
        raise ValueError("proof-v3 record capacities must be positive")
    return max(configured, admitted + 1)


def _proof_v3_gdn_checkpoint_runtime(args):
    """Construct the bounded, prover-local decode checkpoint accelerator."""

    from verallm.miner.gdn_checkpoint_store import BoundedGdnCheckpointStore

    global_bytes = int(
        getattr(args, "proof_v3_gdn_checkpoint_max_bytes", 8 << 30)
    )
    request_bytes = int(
        getattr(
            args,
            "proof_v3_gdn_checkpoint_request_max_bytes",
            2 << 30,
        )
    )
    interval_tokens = int(
        getattr(args, "proof_v3_gdn_checkpoint_interval_tokens", 1024)
    )
    if global_bytes == 0 and request_bytes == 0:
        if interval_tokens <= 0:
            raise ValueError("proof-v3 GDN checkpoint interval must be positive")
        return None, interval_tokens
    if global_bytes <= 0 or request_bytes <= 0 or interval_tokens <= 0:
        raise ValueError(
            "proof-v3 GDN checkpoint budgets and interval must be positive"
        )
    return (
        BoundedGdnCheckpointStore(
            max_bytes=global_bytes,
            max_bytes_per_request=request_bytes,
        ),
        interval_tokens,
    )


def _configure_proof_v3_runtime(args, miner, model_spec) -> None:
    """Authenticate and install one explicit economic proof-v3 release."""

    names = (
        "proof_v3_manifest",
        "proof_v3_execution_profile",
        "proof_v3_calibration_set",
        "proof_v3_attention_semantics",
    )
    configured = {
        name: str(getattr(args, name, None) or "").strip()
        for name in names
    }
    lm_head_catalog_path = str(
        getattr(args, "proof_v3_lm_head_catalog", None) or ""
    ).strip() or None
    projection_manifest_path = str(
        getattr(args, "proof_v3_projection_manifest", None)
        or os.environ.get("VERATHOS_PROOF_V3_PROJECTION_MANIFEST", "")
    ).strip()
    projection_catalog_path = str(
        getattr(args, "proof_v3_projection_catalog", None)
        or os.environ.get("VERATHOS_PROOF_V3_PROJECTION_CATALOG", "")
    ).strip()
    if bool(projection_manifest_path) != bool(projection_catalog_path):
        raise RuntimeError(
            "proof-v3 projection manifest and catalog must be configured "
            "together"
        )
    if not any(configured.values()):
        return
    if not all(configured.values()):
        raise RuntimeError(
            "proof-v3 manifest, signed execution profile, calibration set "
            "and attention semantics must be configured together"
        )
    if (
        not state.batch_mode
        or state.batch_engine is None
        or state.activation_tracker is None
    ):
        raise RuntimeError("proof-v3 requires graph-integrated batch capture")
    if state.tee_skip_proofs:
        raise RuntimeError("proof-v3 cannot be enabled in TEE-only mode")
    chain_config_path = str(getattr(args, "chain_config", None) or "").strip()
    miner_hotkey = str(
        getattr(args, "miner_hotkey_ss58", None) or ""
    ).strip()
    runtime_encoding = str(
        getattr(args, "proof_v3_runtime_encoding", None) or ""
    ).strip()
    if not chain_config_path:
        raise RuntimeError(
            "proof-v3 artifacts require an authenticated chain config"
        )
    if not miner_hotkey:
        raise RuntimeError(
            "proof-v3 requires the serving miner hotkey SS58 identity"
        )
    if not runtime_encoding:
        raise RuntimeError(
            "proof-v3 requires an explicit qualified runtime encoding"
        )

    from verallm.chain.config import ChainConfig
    from verallm.chain.mock import create_clients
    from verallm.miner.economic_proof_v3_live import (
        EconomicProofV3LiveRuntime,
    )
    from verallm.miner.economic_proof_v3_serving import (
        DEFAULT_ECONOMIC_PROOF_V3_TTL_SECONDS,
        EconomicProofV3ServingCoordinator,
    )
    from verallm.miner.economic_proof_v3_weights import (
        EconomicProofV3WeightStore,
        prepare_economic_proof_v3_weight_startup_v3,
        validate_compact_projection_catalog_runtime_v3,
    )
    from verallm.proof_v3.document import (
        load_signed_execution_profile_document_v3,
    )
    from verallm.proof_v3.economic_release import (
        load_qualified_economic_proof_v3_release,
    )
    from verallm.proof_v3.economic_challenge import (
        audited_projections_for_layer_kind_v3,
    )
    from verallm.proof_v3.native_reference_tree_accelerator import (
        install_fused_reference_acceleration,
    )
    from verallm.proof_v3.runtime_architecture import (
        qualify_runtime_layer_kinds_v3,
    )
    from verallm.registry.tokenizer_hash import compute_tokenizer_hash

    rpc_override = getattr(args, "evm_rpc_url", None)
    chain_config = ChainConfig.from_json(
        chain_config_path,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )
    model_client, *_ = create_clients(chain_config)
    if not hasattr(model_client, "get_manifest_authority"):
        raise RuntimeError(
            "proof-v3 requires a live ModelRegistry manifest authority"
        )
    authority = model_client.get_manifest_authority()
    layers = tuple(miner._get_layers())
    model_config = miner._get_text_config()
    layer_kinds = qualify_runtime_layer_kinds_v3(
        config=model_config,
        layers=layers,
    )
    tokenizer_digest = compute_tokenizer_hash(model_spec.model_id)
    signed_profile = load_signed_execution_profile_document_v3(
        configured["proof_v3_execution_profile"]
    )
    from verallm.proof_v3.economic_profile import economic_profile_is_lean_v3

    projection_source = None
    lean_profile = economic_profile_is_lean_v3(signed_profile.profile)
    if lean_profile != bool(projection_manifest_path):
        raise RuntimeError(
            "proof-v3 projection catalog availability does not match the "
            "signed execution profile"
        )
    if lean_profile:
        from verallm.proof_v3.catalog import (
            load_verified_projection_manifest_v3,
        )
        from verallm.proof_v3.catalog_validation_cache import (
            load_weight_catalog_with_validation_cache_v3,
        )

        verified_projection_manifest = (
            load_verified_projection_manifest_v3(
                projection_manifest_path,
                chain_config=chain_config,
                model_registry_client=model_client,
                expected_model_id=model_spec.model_id,
            )
        )
        weight_cache_dir = (
            getattr(args, "proof_v3_weight_cache_dir", None) or None
        )
        (
            projection_catalog,
            projection_validation_context,
        ) = load_weight_catalog_with_validation_cache_v3(
            catalog_path=projection_catalog_path,
            verified_manifest=verified_projection_manifest,
            cache_dir=(
                Path(weight_cache_dir) / "catalog_validation"
                if weight_cache_dir
                else None
            ),
        )
        bt.logging.info(
            "Proof-v3 projection catalog validation receipt: "
            + (
                "hit"
                if projection_validation_context.cache_hit
                else "miss; deep qualification required once"
            )
        )
        projection_source = (
            verified_projection_manifest,
            projection_catalog,
            projection_validation_context,
        )
    qualified_release = load_qualified_economic_proof_v3_release(
        signed_profile_path=configured["proof_v3_execution_profile"],
        manifest_artifact_path=configured["proof_v3_manifest"],
        calibration_set_path=configured["proof_v3_calibration_set"],
        attention_runtime_semantics_path=(
            configured["proof_v3_attention_semantics"]
        ),
        gdn_runtime_semantics_path=(
            str(getattr(args, "proof_v3_gdn_semantics", None) or "").strip()
            or None
        ),
        lm_head_catalog_path=lm_head_catalog_path,
        expected_model_id=model_spec.model_id,
        expected_authorities=authority.signers,
        authority_threshold=authority.threshold,
        layer_kinds=layer_kinds,
        tokenizer_binding_digest=tokenizer_digest,
        runtime_encoding_id=runtime_encoding,
        max_decode_tokens=signed_profile.profile.max_verified_decode_tokens,
        verified_projection_manifest=(
            projection_source[0]
            if projection_source is not None else None
        ),
        weight_catalog=(
            projection_source[1]
            if projection_source is not None else None
        ),
        projection_catalog_validation_context=(
            projection_source[2]
            if projection_source is not None else None
        ),
    )
    release = qualified_release.runtime
    if not install_fused_reference_acceleration():
        raise RuntimeError(
            "proof-v3 requires the native CUDA commitment backend"
        )
    from verallm.proof_v3.economic_profile import (
        economic_profile_is_compact_v3,
    )

    compact_static_weights = economic_profile_is_compact_v3(
        release.profile
    )
    weight_store = EconomicProofV3WeightStore(
        runtime_model=miner.model,
        decoder_layers=layers,
        model_config=model_config,
        runtime_release=release,
        cache_dir=(
            getattr(args, "proof_v3_weight_cache_dir", None) or None
        ),
        compact_selected_trace=compact_static_weights,
    )
    base_started = time.perf_counter()
    challenge_names = {
        f"l{layer}.{manifest_suffix}"
        for layer, kind in enumerate(layer_kinds)
        for _x_suffix, _s_suffix, manifest_suffix in (
            audited_projections_for_layer_kind_v3(kind)
        )
    }
    (
        lm_head_compute_bytes,
        reclaimed_weight_cache_bytes,
        challenge_material_count,
    ) = prepare_economic_proof_v3_weight_startup_v3(
        profile=release.profile,
        weight_store=weight_store,
        projection_names=challenge_names,
    )
    compatibility_operation_count = 0
    if compact_static_weights:
        compatibility_operation_count = (
            validate_compact_projection_catalog_runtime_v3(
                weight_store=weight_store,
                projection_catalog=release.artifacts.lean_projection_catalog,
                projection_names=challenge_names,
            )
        )
    base_seconds = time.perf_counter() - base_started
    coordinator_record_capacity = _proof_v3_coordinator_record_capacity(
        configured_max_records=int(
            getattr(args, "proof_v3_max_records", 64)
        ),
        max_admitted_requests=int(state.admission.max_requests),
    )
    coordinator = EconomicProofV3ServingCoordinator(
        max_records=coordinator_record_capacity,
        max_retained_bytes=int(
            getattr(args, "proof_v3_max_retained_bytes", 8 << 30)
        ),
        ttl_seconds=float(
            getattr(
                args,
                "proof_v3_ttl_seconds",
                DEFAULT_ECONOMIC_PROOF_V3_TTL_SECONDS,
            )
        ),
    )
    checkpoint_store, checkpoint_interval = _proof_v3_gdn_checkpoint_runtime(
        args
    )
    runtime = EconomicProofV3LiveRuntime(
        runtime_release=release,
        weight_store=weight_store,
        coordinator=coordinator,
        batch_engine=state.batch_engine,
        tracker=state.activation_tracker,
        miner_hotkey_ss58=miner_hotkey,
        admission=state.admission,
        gdn_execution_checkpoint_store=checkpoint_store,
        gdn_execution_checkpoint_target_interval_tokens=checkpoint_interval,
    )
    state.proof_v3_coordinator = coordinator
    state.proof_v3_runtime = runtime
    bt.logging.info(
        "Proof-v3 authenticated runtime ready: "
        f"profile={release.profile.digest().hex()[:16]}... "
        f"layers={len(layer_kinds)} base_setup={base_seconds:.2f}s "
        f"challenge_weights={challenge_material_count} "
        f"catalog_runtime_rows={compatibility_operation_count} "
        f"retained_records={coordinator_record_capacity} "
        f"bounded_checkpoints={'on' if checkpoint_store is not None else 'off'} "
        f"checkpoint_interval={checkpoint_interval} "
        f"cache_reclaimed="
        f"{reclaimed_weight_cache_bytes / (1 << 30):.2f}GiB "
        f"lm_head_compute={lm_head_compute_bytes / (1 << 20):.1f}MiB"
    )


def _configure_proof_v2_artifacts(args, miner, model_spec, *, tee_only: bool) -> None:
    """Authenticate explicit or release-bundled proof-v2 static artifacts."""

    explicit_manifest = str(
        getattr(args, "proof_v2_manifest", None)
        or os.environ.get("VERATHOS_PROOF_V2_MANIFEST", "")
    ).strip()
    explicit_catalog = str(
        getattr(args, "proof_v2_weight_catalog", None)
        or os.environ.get("VERATHOS_PROOF_V2_WEIGHT_CATALOG", "")
    ).strip()
    if bool(explicit_manifest) != bool(explicit_catalog):
        raise RuntimeError(
            "proof-v2 manifest and weight catalog must be configured together"
        )
    if explicit_manifest and tee_only:
        raise RuntimeError("proof-v2 artifacts cannot be used in TEE-only mode")

    from verallm.proof_v2.artifact_store import (
        configured_proof_v2_artifact_base_urls,
    )

    remote_base_urls = configured_proof_v2_artifact_base_urls(
        getattr(args, "proof_v2_artifact_base_url", None)
    )
    remote_cache_directory = getattr(
        args,
        "proof_v2_artifact_cache_dir",
        None,
    )

    chain_config_path = getattr(args, "chain_config", None)
    if not chain_config_path:
        if explicit_manifest or remote_base_urls:
            raise RuntimeError("proof-v2 artifacts require --chain-config")
        return

    from verallm.chain.config import ChainConfig
    from verallm.chain.mock import create_clients
    from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
    from verallm.proof_v2.runtime import (
        BUNDLED_CATALOG_SUFFIX,
        bundled_proof_v2_manifest_paths,
        load_verified_proof_v2_manifest,
        load_verified_proof_v2_manifests,
    )

    rpc_override = getattr(args, "evm_rpc_url", None)
    proof_v2_chain_config = ChainConfig.from_json(
        chain_config_path,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )
    remote_base_urls = configured_proof_v2_artifact_base_urls(
        getattr(args, "proof_v2_artifact_base_url", None),
        default_values=getattr(
            proof_v2_chain_config,
            "proof_v2_artifact_base_urls",
            (),
        ),
    )
    remote_cache_directory = (
        remote_cache_directory
        or getattr(
            proof_v2_chain_config,
            "proof_v2_artifact_cache_dir",
            "",
        )
        or None
    )

    bundled_manifests = ()
    if not explicit_manifest:
        if tee_only:
            return
        bundled_manifests = bundled_proof_v2_manifest_paths(
            chain_id=proof_v2_chain_config.chain_id,
            netuid=proof_v2_chain_config.netuid,
        )
        if not bundled_manifests and not remote_base_urls:
            return

    model_client, *_ = create_clients(proof_v2_chain_config)
    weight_catalog = None

    if explicit_manifest:
        verified_manifest = load_verified_proof_v2_manifest(
            explicit_manifest,
            chain_config=proof_v2_chain_config,
            model_registry_client=model_client,
            expected_model_id=model_spec.model_id,
        )
        catalog_path = explicit_catalog
    else:
        verified_manifest = None
        if bundled_manifests:
            verified = load_verified_proof_v2_manifests(
                bundled_manifests,
                chain_config=proof_v2_chain_config,
                model_registry_client=model_client,
            )
            verified_manifest = verified.get(model_spec.model_id)
        if verified_manifest is not None:
            catalog_path = str(
                verified_manifest.source_path.parent
                / (verified_manifest.manifest.digest().hex() + BUNDLED_CATALOG_SUFFIX)
            )
        elif remote_base_urls:
            from verallm.proof_v2.artifact_store import (
                resolve_remote_proof_v2_artifacts,
            )

            resolved = resolve_remote_proof_v2_artifacts(
                model_spec.model_id,
                remote_base_urls,
                chain_config=proof_v2_chain_config,
                model_registry_client=model_client,
                cache_directory=remote_cache_directory,
            )
            verified_manifest = resolved.verified_manifest
            weight_catalog = resolved.weight_catalog
            catalog_path = str(resolved.catalog_path)
            bt.logging.info(
                "Resolved proof-v2 artifacts from " f"{resolved.index_source_url}"
            )
        else:
            bt.logging.info(
                f"No bundled proof-v2 artifacts registered for {model_spec.model_id}"
            )
            return

    if weight_catalog is None:
        weight_catalog = WeightCommitmentCatalogV2.load(catalog_path)
    miner.configure_proof_v2(verified_manifest.manifest, weight_catalog)
    bt.logging.info(
        f"Proof-v2 static artifacts authenticated for {model_spec.model_id}"
    )


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
    _preflight_gpu_check(skip=getattr(args, "skip_gpu_check", False))
    state.capacity_audit_state_file = str(
        getattr(args, "capacity_audit_state_file", "") or ""
    )

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
            bt.logging.warning(
                "CUDA extension missing -- using slow CPU fallback (VERATHOS_ALLOW_CPU_FALLBACK=1)"
            )

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    quant = args.quant

    # Resolve model: either from registry or raw checkpoint
    if args.model_id:
        model_name, quant, registry_max_model_len = _resolve_model_from_registry(
            args.model_id,
            quant,
            args.max_model_len,
        )
        if args.max_model_len is None:
            args.max_model_len = registry_max_model_len
    else:
        model_name = args.model

    state.model_name = model_name
    proof_v3_capture_manifest = _prepare_proof_v3_graph_capture(
        args,
        expected_model_id=model_name,
    )

    configured_quant_method = _model_quant_method(model_name)
    model_name_l = model_name.lower()
    is_gptq = configured_quant_method == "gptq" or (
        not configured_quant_method and "gptq" in model_name_l
    )
    is_awq = configured_quant_method == "awq" or (
        not configured_quant_method and "awq" in model_name_l
    )
    is_fp8 = "fp8" in model_name.lower()
    setattr(args, "_configured_quant_method", configured_quant_method)
    if quant == "auto":
        if is_gptq or is_awq:
            quant = "int4"
        elif is_fp8:
            quant = "fp8"
        else:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
    if getattr(args, "tee_enabled", False):
        tee = f"{args.tee_platform} (proofs {'disabled' if getattr(args, 'tee_skip_proofs', True) else 'enabled'})"
    print_server_banner(
        model=model_name,
        quant=quant,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        sm=sm,
        attention=attention_backend or "",
        tee=tee,
        batch_mode=getattr(args, "batch_mode", True)
        and not getattr(args, "no_batch_mode", False),
        port=args.port,
    )

    bt.logging.info("Phase 0: Loading model and computing weight roots...")

    temp_spec = ModelSpec(
        model_id=model_name,
        weight_merkle_root=b"\x00" * 32,
        num_layers=0,
        hidden_dim=0,
        num_heads=0,
        head_dim=0,
        intermediate_dim=0,
        vocab_size=0,
        activation="silu",
        norm_type="rmsnorm",
        attention_type="gqa",
    )
    miner = VllmMiner(model_name, temp_spec, config)

    vllm_kwargs = {}
    if args.max_model_len:
        vllm_kwargs["max_model_len"] = args.max_model_len
    if attention_backend:
        vllm_kwargs["attention_config"] = {"backend": attention_backend}
    if getattr(args, "max_num_seqs", None):
        vllm_kwargs["max_num_seqs"] = args.max_num_seqs
    if proof_v3_capture_manifest is not None:
        prefix_cache_sharing = bool(
            getattr(args, "_proof_v3_prefix_cache_sharing", False)
        )
        vllm_kwargs["enable_prefix_caching"] = prefix_cache_sharing
        if prefix_cache_sharing:
            vllm_kwargs["scheduler_cls"] = (
                "verallm.miner.proof_cache_scheduler.ProofCacheScheduler"
            )
    (
        full_execution_trace_capture,
        full_attention_state_capture,
    ) = _proof_execution_capture_modes(
        args,
        allowed_protocol_versions=state.allowed_proof_protocol_versions,
    )
    miner.setup_vllm(
        quant=quant,
        gpu_memory_utilization=args.gpu_memory_utilization,
        is_gptq=is_gptq,
        is_awq=is_awq,
        force_awq_gemm_fallback=getattr(args, "awq_gemm_fallback", False),
        proof_v2_full_trace_capture=full_execution_trace_capture,
        # Logical paged-K/V boundary capture belongs to the dormant v2
        # transition statement.  V3 still installs the complete projection,
        # residual, GDN, attention-reduction and terminal capture path above,
        # but must not place this Python v2 wrapper in its compiled model.
        proof_v2_full_attention_state_capture=full_attention_state_capture,
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
        bt.logging.info(
            f"Detected MoE: {moe_config.num_routed_experts} experts, top-{moe_config.top_k}"
        )
    state.moe_config = moe_config

    detected_quant = detect_quantization(miner.model)
    detected_mode = detected_quant.quant_mode
    root_cache_mode = detected_mode
    if detected_mode == "int4" and getattr(args, "awq_gemm_fallback", False):
        root_cache_mode = "int4-awq-gemm"

    # TEE-only mode: skip Merkle root computation entirely — attestation
    # replaces proofs, so weight trees are not needed.  We still need a
    # minimal ModelSpec for the model_id and architecture fields.
    _tee_only = (
        getattr(args, "tee_enabled", False)
        and getattr(args, "tee_skip_proofs", None) is not False
    )

    if _tee_only:
        bt.logging.info(
            "TEE mode: skipping Merkle root computation (attestation replaces proofs)"
        )
        model_spec = _build_minimal_model_spec(miner.model, model_name, detected_mode)
        # Still need on-chain spec for model_id, vocab_size, etc.
        if hasattr(args, "chain_config") and args.chain_config:
            try:
                from verallm.chain.config import ChainConfig
                from verallm.chain.model_registry import ModelRegistryClient

                chain_config = ChainConfig.from_json(args.chain_config)
                chain_client = ModelRegistryClient(chain_config)
                chain_spec = chain_client.get_model_spec(model_name)
                if chain_spec:
                    model_spec = chain_spec
                    bt.logging.info(
                        f"TEE mode: using on-chain ModelSpec for {model_name}"
                    )
            except Exception as e:
                bt.logging.warning(f"TEE mode: could not fetch chain ModelSpec: {e}")
    elif not _legacy_weight_merkle_startup_required(
        proof_v3_configured=proof_v3_capture_manifest is not None,
        allowed_protocol_versions=state.allowed_proof_protocol_versions,
    ):
        model_spec = _load_v3_required_onchain_model_spec(
            args,
            model_name,
        )
        bt.logging.info(
            "Proof-v3 required: using the on-chain ModelSpec; selected "
            "runtime weights are bound to the authenticated v3 catalog by "
            "hard proofs"
        )
        try:
            from verallm.registry.tokenizer_hash import compute_tokenizer_hash

            model_spec.tokenizer_hash = compute_tokenizer_hash(model_name)
            bt.logging.info(
                f"Tokenizer hash: {model_spec.tokenizer_hash[:8].hex()}..."
            )
        except Exception as e:
            bt.logging.warning(f"Could not compute tokenizer_hash: {e}")
    else:
        # NOTE: In production, ModelSpec comes from on-chain registry.
        # Miner computes its own roots and compares as a self-check.
        model_spec = (
            None
            if args.no_cache
            else load_cached_model_spec(
                model_name, config.w_merkle_chunk_size, root_cache_mode
            )
        )
        if model_spec is None:
            bt.logging.info(
                "Computing weight Merkle roots (no cache found, may take a few minutes). "
                "Cache at .model_root_cache/ is reusable — copy it to skip this on other instances."
            )
            model_spec = compute_model_roots(
                miner.model,
                model_name,
                chunk_size=config.w_merkle_chunk_size,
            )
            save_model_spec_to_cache(
                model_spec, config.w_merkle_chunk_size, root_cache_mode
            )
        else:
            bt.logging.info("Using cached ModelSpec (from .model_root_cache/)")

        # Populate tokenizer_hash from the tokenizer.  Computed fresh on
        # every startup so it always reflects the local tokenizer state —
        # the validator will compare its own local hash to the on-chain
        # anchor at epoch start to detect drift.
        try:
            from verallm.registry.tokenizer_hash import compute_tokenizer_hash

            model_spec.tokenizer_hash = compute_tokenizer_hash(model_name)
            bt.logging.info(f"Tokenizer hash: {model_spec.tokenizer_hash[:8].hex()}...")
        except Exception as e:
            bt.logging.warning(f"Could not compute tokenizer_hash: {e}")

        # On-chain self-check: compare local roots against chain registry
        if hasattr(args, "chain_config") and args.chain_config:
            _chain_self_check(args, model_spec)

    miner.model_spec = model_spec
    miner.model_commitment = model_spec.weight_merkle_root
    state.model_spec = model_spec

    _configure_proof_v2_artifacts(
        args,
        miner,
        model_spec,
        tee_only=_tee_only,
    )

    if config.k_layers == 0 and model_spec.num_layers > 0:
        config.k_layers = compute_auto_k(model_spec.num_layers)
        bt.logging.info(f"Auto k_layers: {config.k_layers}/{model_spec.num_layers}")

    if moe_config and config.k_experts_per_layer == 0:
        config.k_experts_per_layer = compute_auto_k_experts(
            moe_config.num_routed_experts
        )
        bt.logging.info(
            f"Auto k_experts: {config.k_experts_per_layer}/{moe_config.num_routed_experts}"
        )

    if not _tee_only and _legacy_weight_merkle_startup_required(
        proof_v3_configured=proof_v3_capture_manifest is not None,
        allowed_protocol_versions=state.allowed_proof_protocol_versions,
    ):
        bt.logging.info("Precomputing weight Merkle trees...")
        tree_ms = miner.precompute_weight_merkles()
        bt.logging.info(f"Merkle trees ready ({tree_ms:.0f}ms)")
    elif not _tee_only:
        bt.logging.info(
            "Proof-v3 required: skipping legacy v1 weight Merkle tree startup"
        )
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
    if getattr(args, "tee_enabled", False):
        _init_tee(args, model_spec)

    # ── Batch mode setup ────────────────────────────────────────────
    if getattr(args, "batch_mode", True) and not getattr(args, "no_batch_mode", False):
        state.batch_mode = True

        # Create batch engine
        state.batch_engine = BatchAwareEngine(miner.llm)

        # When TEE skip_proofs is active, skip all activation capture and proof
        # pipeline setup. Hardware attestation replaces VeraLLM proofs entirely.
        _tee_skip = state.tee_skip_proofs

        if _tee_skip:
            bt.logging.info(
                "TEE mode: skipping activation capture and proof pipeline (attestation replaces proofs)"
            )

        # Determine capture backend: splitting_ops (CUDA graphs) or hooks (eager)
        layers = miner._get_layers() if not _tee_skip else []
        use_cuda_graphs = getattr(miner, "_use_cuda_graphs", False)
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
                from verallm.vllm_plugin.capture_linear import (
                    CaptureDecoderLayerWrapper,
                    CaptureLinearWrapper,
                )

                for layer in layers:
                    mlp = miner._get_mlp(layer)
                    if mlp is None:
                        continue
                    gate = miner._get_gate_proj(mlp)
                    if isinstance(gate, CaptureLinearWrapper):
                        _use_buffer = bool(getattr(gate, "_use_buffer", False))
                        break
            elif miner.is_moe and getattr(miner, "_moe_buffer_mode", False):
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
                    from verallm.miner.vllm_utils import (
                        _find_layers as _find_runtime_layers,
                    )

                    runtime_layers = _find_runtime_layers(runtime_model)
                    n_runtime_instrumented = attach_capture_ops(
                        model=runtime_model,
                        layers=runtime_layers,
                        is_moe=miner.is_moe,
                        get_mlp_fn=miner._get_mlp,
                        get_gate_proj_fn=miner._get_gate_proj,
                        is_moe_layer_fn=is_moe_layer
                        if miner.is_moe
                        else (lambda _: False),
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
                    router_scoring=getattr(
                        miner.model_spec, "router_scoring", "softmax"
                    ),
                )
                state.moe_hook_mgr._challenged_layers = [
                    i for i, layer in enumerate(layers) if is_moe_layer(layer)
                ]
                bt.logging.info(
                    f"Batch mode: MoE router capture via splitting_ops ({len(state.moe_hook_mgr.get_challenged_layers())} challenged layers)"
                )
        else:
            # Hook-based capture (enforce_eager=True path)
            state.activation_tracker = RequestActivationTracker(
                state.batch_engine.model_runner
            )
            n_hooks = state.activation_tracker.install_hooks(
                layers=layers,
                is_moe_layer_fn=is_moe_layer if miner.is_moe else lambda _: False,
                get_mlp_fn=miner._get_mlp,
                get_gate_proj_fn=miner._get_gate_proj,
            )
            bt.logging.info(
                f"Batch mode: installed {n_hooks} persistent activation hooks"
            )

            # Create batch MoE hook manager (persistent)
            if miner.is_moe and miner.model is not None:
                state.moe_hook_mgr = BatchMoEHookManager(
                    miner.model,
                    state.activation_tracker,
                    router_top_k=getattr(miner.model_spec, "router_top_k", 0),
                    router_scoring=getattr(
                        miner.model_spec, "router_scoring", "softmax"
                    ),
                )
                state.moe_hook_mgr.install_hooks()
                bt.logging.info(
                    f"Batch mode: installed MoE router hooks for {len(state.moe_hook_mgr.get_challenged_layers())} layers"
                )

        if not _tee_skip:
            # The live GDN cache wrapper uses the same request tracker as
            # split-point capture.  Register it for eager mode as well; this
            # is inert unless an authenticated transition profile opts in.
            from verallm.vllm_plugin.ops import set_active_tracker

            set_active_tracker(state.activation_tracker)

            # Register MoE buffer-mode capture layers with the activation tracker.
            _moe_capture_buffers = getattr(miner, "_moe_capture_buffers", [])
            if miner.is_moe and _moe_capture_buffers:
                state.activation_tracker.register_capture_buffers(_moe_capture_buffers)
                bt.logging.info(
                    f"Batch mode: MoE buffer-mode capture ({len(_moe_capture_buffers)} layers registered)"
                )

            # Pre-extract MoE router weights to CPU for zero-GPU recomputation.
            if getattr(state, "moe_hook_mgr", None) is not None:
                n_prewarmed = state.moe_hook_mgr.prewarm_router_weights()
                if n_prewarmed:
                    bt.logging.info(
                        f"Batch mode: pre-warmed {n_prewarmed} router weights to CPU"
                    )

            # Register buffer-mode capture layers for dense models.
            if not miner.is_moe and not _skip_capture:
                from verallm.vllm_plugin.capture_linear import (
                    CaptureDecoderLayerWrapper,
                    CaptureLinearWrapper,
                )

                reduction_wrappers = getattr(
                    state.batch_engine.model_runner,
                    "_verathos_reduction_wrappers",
                    None,
                ) or {}
                reduction_wrapper_ids = {
                    id(wrapper)
                    for wrappers in reduction_wrappers.values()
                    if _proof_v3_reduction_wrapper_is_dedicated_buffer(
                        wrappers
                    )
                    for wrapper in (wrappers.get("qkv"), wrappers.get("o"))
                    if wrapper is not None
                }
                capture_wrappers = []
                seen_wrappers = set()
                for layer in layers:
                    for module in layer.modules():
                        if (
                            isinstance(
                                module,
                                (CaptureLinearWrapper, CaptureDecoderLayerWrapper),
                            )
                            and id(module) not in reduction_wrapper_ids
                            and id(module) not in seen_wrappers
                        ):
                            seen_wrappers.add(id(module))
                            capture_wrappers.append(module)
                buf_wrappers = [
                    wrapper
                    for wrapper in capture_wrappers
                    if _proof_v3_capture_wrapper_has_buffer(wrapper)
                ]
                if buf_wrappers:
                    state.activation_tracker.register_capture_buffers(buf_wrappers)
                    bt.logging.info(
                        "Batch mode: registered "
                        f"{len(buf_wrappers)} buffer-mode proof projections"
                    )

                if proof_v3_capture_manifest is not None:
                    tracker = state.activation_tracker
                    root_wrappers = [
                        wrapper
                        for wrapper in capture_wrappers
                        if wrapper.proof_capture_root_buffers()
                    ]
                    root_buffers = tuple(
                        item
                        for wrapper in root_wrappers
                        for item in wrapper.proof_capture_root_buffers()
                    )
                    split_root_row_aliases = tuple(
                        item
                        for wrapper in root_wrappers
                        if isinstance(wrapper, CaptureLinearWrapper)
                        for item in wrapper.proof_capture_split_row_aliases()
                    )
                    if root_buffers:
                        tracker.register_execution_anchor_root_buffers(
                            root_buffers
                        )
                        root_staging_buffers = tuple(
                            (
                                int(binding.stage_id.split(".", 1)[0][1:]),
                                binding.stage_id.split(".", 1)[1],
                                staging,
                                binding.row_width,
                            )
                            for wrapper in root_wrappers
                            for binding in wrapper._runtime_root_bindings()
                            for staging in (
                                getattr(
                                    binding.owner,
                                    binding.staging_attribute,
                                ),
                            )
                            if staging is not None
                        )
                        tracker.register_execution_anchor_root_staging_buffers(
                            root_staging_buffers
                        )
                        root_retention_records = tuple(
                            item
                            for wrapper in root_wrappers
                            for item in wrapper.proof_capture_root_retention()
                        )
                        tracker.register_execution_anchor_root_retention(
                            root_retention_records
                        )
                        tracker.register_split_execution_anchor_aliases(
                            split_root_row_aliases
                        )
                    elif tracker._capture_row_indices is not None:
                        raise RuntimeError(
                            "proof-v3 requires whole-step graph-integrated "
                            "capture buffers or compact graph roots"
                        )
                    root_row_aliases = tuple(
                        item
                        for wrapper in buf_wrappers
                        for item in (
                            wrapper.proof_capture_root_row_aliases()
                            if isinstance(wrapper, CaptureLinearWrapper)
                            else ()
                        )
                    )
                    reduction_root_row_aliases = []
                    expected_attention = tuple(
                        args._proof_v3_full_attention_layers
                    )
                    actual_attention = tuple(sorted(reduction_wrappers))
                    if actual_attention != expected_attention:
                        raise RuntimeError(
                            "proof-v3 reduction buffers do not match the "
                            "signed full-attention inventory"
                    )
                    if reduction_wrappers:
                        reduction_buffers = []
                        split_reduction_stages = []
                        for layer, wrappers in sorted(
                            reduction_wrappers.items()
                        ):
                            qkv = wrappers.get("qkv")
                            output = wrappers.get(
                                "qkv_output_buffer",
                                getattr(
                                    qkv,
                                    "_capture_output_buf",
                                    None,
                                ),
                            )
                            o_projection = wrappers.get("o")
                            input_ = wrappers.get(
                                "o_input_buffer",
                                getattr(
                                    o_projection,
                                    "_capture_buf",
                                    None,
                                ),
                            )
                            selected_row_buffers = []
                            registered_row_indices = wrappers.get(
                                "row_indices"
                            )
                            if isinstance(
                                registered_row_indices,
                                torch.Tensor,
                            ):
                                tracker.register_capture_row_indices(
                                    registered_row_indices
                                )
                                selected_row_buffers.append(
                                    registered_row_indices
                                )
                            for wrapper in (qkv, o_projection):
                                row_indices = getattr(
                                    wrapper,
                                    "_capture_row_indices",
                                    None,
                                )
                                if isinstance(row_indices, torch.Tensor):
                                    tracker.register_capture_row_indices(
                                        row_indices
                                    )
                                    selected_row_buffers.append(row_indices)
                            split_mode = (
                                not getattr(qkv, "_use_buffer", True)
                                and not getattr(
                                    o_projection,
                                    "_use_buffer",
                                    True,
                                )
                                and output is None
                                and input_ is None
                                and not selected_row_buffers
                            )
                            if split_mode:
                                if args._proof_v3_lean_capture:
                                    raise RuntimeError(
                                        "proof-v3 lean split attention K/V "
                                        "row alias is unavailable"
                                    )
                                split_reduction_stages.extend(
                                    (
                                        (
                                            layer,
                                            "attention_qkv_output",
                                        ),
                                        (
                                            layer,
                                            "attention_o_input",
                                        ),
                                    )
                                )
                                continue
                            if output is None or input_ is None:
                                raise RuntimeError(
                                    "proof-v3 reduction capture mixes split "
                                    "and buffer modes"
                                )
                            qkv_whole_step = (
                                _proof_v3_capture_buffer_is_whole_step(qkv)
                            )
                            o_input_whole_step = (
                                _proof_v3_capture_buffer_is_whole_step(
                                    o_projection
                                )
                            )
                            kv_root_rows = ()
                            if args._proof_v3_lean_capture:
                                root_owner = qkv
                                while isinstance(
                                    root_owner,
                                    CaptureLinearWrapper,
                                ):
                                    kv_root_rows = (
                                        root_owner
                                        .proof_capture_root_row_aliases(
                                            output
                                        )
                                    )
                                    if kv_root_rows:
                                        break
                                    root_owner = getattr(
                                        root_owner,
                                        "original",
                                        None,
                                    )
                                if (
                                    len(kv_root_rows) != 1
                                    or kv_root_rows[0][0] != layer
                                    or kv_root_rows[0][1]
                                    != "attention_kv_output"
                                ):
                                    raise RuntimeError(
                                        "proof-v3 lean attention K/V "
                                        "whole-step capture is unavailable"
                                    )
                            reduction_buffers.extend(
                                (
                                    (
                                        layer,
                                        "attention_qkv_output",
                                        output,
                                        qkv_whole_step,
                                    ),
                                    (
                                        layer,
                                        "attention_o_input",
                                        input_,
                                        o_input_whole_step,
                                    ),
                                )
                            )
                            reduction_buffers.extend(
                                (
                                    alias_layer,
                                    alias_suffix,
                                    alias_buffer,
                                    qkv_whole_step,
                                )
                                for (
                                    alias_layer,
                                    alias_suffix,
                                    alias_buffer,
                                ) in kv_root_rows
                            )
                            reduction_root_row_aliases.extend(kv_root_rows)
                        if reduction_buffers and split_reduction_stages:
                            raise RuntimeError(
                                "proof-v3 reduction capture cannot mix "
                                "split and buffer modes"
                            )
                        if reduction_buffers:
                            tracker.register_reduction_buffers(
                                reduction_buffers
                            )
                        else:
                            tracker.register_split_reduction_stages(
                                split_reduction_stages
                            )

                    _register_proof_v3_economic_pool_capture(
                        tracker=tracker,
                        capture_wrappers=capture_wrappers,
                        root_row_aliases=root_row_aliases,
                        reduction_root_row_aliases=(
                            reduction_root_row_aliases
                        ),
                    )

                    gdn_modules = []
                    for layer_index, layer in enumerate(layers):
                        owner = layer
                        while isinstance(
                            getattr(owner, "original", None),
                            torch.nn.Module,
                        ):
                            owner = owner.original
                        module = getattr(owner, "linear_attn", None)
                        while isinstance(
                            getattr(module, "original", None),
                            torch.nn.Module,
                        ):
                            module = module.original
                        if (
                            isinstance(module, torch.nn.Module)
                            and hasattr(module, "kv_cache")
                        ):
                            gdn_modules.append((layer_index, module))
                    expected_gdn = tuple(
                        index
                        for index, kind in enumerate(
                            args._proof_v3_layer_kinds
                        )
                        if kind == "gdn"
                    )
                    if tuple(
                        layer for layer, _module in gdn_modules
                    ) != expected_gdn:
                        raise RuntimeError(
                            "proof-v3 GDN state modules do not match the "
                            "signed layer inventory"
                        )
                    if gdn_modules:
                        tracker.register_gdn_state_modules(gdn_modules)
                    if bool(
                        getattr(
                            args,
                            "_proof_v3_prefix_cache_sharing",
                            False,
                        )
                    ):
                        expected_attention = tuple(
                            index
                            for index, kind in enumerate(
                                args._proof_v3_layer_kinds
                            )
                            if kind == "full_attention"
                        )
                        if not expected_attention:
                            raise RuntimeError(
                                "proof-v3 prefix-cache profile has no "
                                "registered attention layers"
                            )
                        tracker.register_prefix_cache_attention_layers(
                            expected_attention
                        )

            if state.activation_tracker.has_capture_buffers:
                state.batch_engine.set_step_output_callback(
                    state.activation_tracker.snapshot_trace_step_buffers
                )
                state.batch_engine.set_finished_outputs_callback(
                    state.activation_tracker.snapshot_capture_buffers_batch
                )

            # Install lm_head hook for decode-integrity capture.
            if _skip_capture or os.environ.get("VERALLM_SKIP_LM_HEAD_HOOK"):
                if not _skip_capture:
                    bt.logging.warning(
                        "VERALLM_SKIP_LM_HEAD_HOOK: lm_head decode-integrity hook DISABLED"
                    )
            elif miner.model is not None and hasattr(miner.model, "compute_logits"):
                state.activation_tracker.install_lm_head_hook(
                    miner.model, capture_logits=False
                )

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
            proof_threads = (
                getattr(args, "proof_threads", None) or auto_detect_proof_concurrency()
            )
            proof_max_pending_override = getattr(args, "proof_max_pending", None)
            state.proof_pipeline = ProofPipeline(
                max_concurrent_proofs=proof_threads,
                max_pending=proof_max_pending_override,
            )

            # Initialize batched proof matmul service when backend is "batched".
            _matmul_backend = getattr(args, "proof_matmul_backend", "batched")
            if _matmul_backend == "batched":
                from verallm.miner.matmul import init_proof_matmul_batcher

                init_proof_matmul_batcher()

        # Dynamic token-budget admission control (needed in both proof and TEE mode)
        # vLLM >=0.16.1 moved cache_config under vllm_config
        engine = miner.llm.llm_engine
        cache_config = getattr(engine, "cache_config", None)
        if cache_config is None:
            cache_config = engine.vllm_config.cache_config
        total_kv_tokens = (cache_config.num_gpu_blocks or 0) * (
            cache_config.block_size or 16
        )
        # Read the actual fitted max_model_len from vLLM (handles auto-fit)
        max_context = miner.llm.llm_engine.model_config.max_model_len
        scheduler_max_requests = _engine_scheduler_max_num_seqs(engine)
        requested_max_requests = getattr(
            args, "max_concurrent", None
        ) or auto_detect_max_requests_with_ram(
            hidden_dim=model_spec.hidden_dim,
            intermediate_dim=model_spec.intermediate_dim,
        )
        max_requests = _bounded_server_request_admission(
            int(requested_max_requests),
            scheduler_max_requests,
        )

        estimated_mb = estimate_per_request_ram_mb(
            model_spec.hidden_dim,
            model_spec.intermediate_dim,
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
                new_max = _bounded_server_request_admission(
                    new_max,
                    scheduler_max_requests,
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
                        state.proof_pipeline._max_pending, new_max
                    )
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
                f"scheduler_max_num_seqs={scheduler_max_requests}, "
                f"proof_threads={state.proof_pipeline.max_concurrent}, proof_max_pending={state.proof_pipeline.max_pending}"
            )

    _configure_proof_v3_runtime(args, miner, model_spec)

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
    if state.batch_mode:
        bt.logging.info(
            "Deferring warmup to the production batch serving path"
        )
    else:
        bt.logging.info("Running warmup inference...")
        t_warmup = time.perf_counter()
        try:
            from vllm import SamplingParams

            def _apply_template(messages):
                try:
                    return miner.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    return " ".join(m.get("content", "") for m in messages)

            _warmup_params = SamplingParams(max_tokens=4, temperature=0)

            # 1) Short prompt — compiles recurrent-mode FLA kernels + decode
            _short = _apply_template([{"role": "user", "content": "Hi"}])
            miner.llm.generate([_short], sampling_params=_warmup_params)
            bt.logging.info(
                f"Warmup short prompt done "
                f"({(time.perf_counter() - t_warmup) * 1000:.0f}ms)"
            )

            # 2) Long prompt — compiles chunk-mode FLA kernels for prefill.
            _long_content = "Hello, this is warmup. " * 80
            _long_msgs = [
                {"role": "user", "content": "Tell me about AI."},
                {"role": "assistant", "content": _long_content},
                {"role": "user", "content": "Continue."},
                {"role": "assistant", "content": _long_content},
                {"role": "user", "content": "Ok."},
            ]
            _long = _apply_template(_long_msgs)
            miner.llm.generate([_long], sampling_params=_warmup_params)
            bt.logging.info(
                f"Warmup long prompt done "
                f"({(time.perf_counter() - t_warmup) * 1000:.0f}ms)"
            )

        except Exception as e:
            bt.logging.warning(f"Warmup inference failed: {e}")
        bt.logging.info(
            f"Warmup complete "
            f"({(time.perf_counter() - t_warmup) * 1000:.0f}ms)"
        )

    bt.logging.info(
        f"ModelSpec: {model_spec.num_layers} layers, hidden={model_spec.hidden_dim}"
    )
    bt.logging.info(f"Roots: {len(model_spec.weight_block_merkle_roots)} layer roots")
    if state.tee_enabled:
        bt.logging.info(
            f"Proof pipeline: TEE attestation ({state.tee_platform}) — "
            f"ZK proofs disabled, hardware attestation active"
        )
    elif not _tee_only:
        _cap_backend = getattr(state, "activation_tracker", None)
        _cap_str = (
            getattr(_cap_backend, "backend", "unknown") if _cap_backend else "none"
        )
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
        bt.logging.info(
            "max_model_len: auto (vLLM will determine from available KV cache)"
        )

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
        head_dim=getattr(text_cfg, "head_dim", 0)
        or (
            getattr(text_cfg, "hidden_size", 0)
            // max(getattr(text_cfg, "num_attention_heads", 1), 1)
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

    def _request_awq_gemm_fallback(reason: str) -> None:
        if getattr(args, "awq_gemm_fallback", False):
            return
        model_arg = str(
            getattr(args, "model", "") or getattr(args, "model_id", "") or ""
        )
        quant_arg = str(getattr(args, "quant", "") or "").lower()
        spec_quant = str(getattr(local_spec, "quant_mode", "") or "").lower()
        configured_quant = str(
            getattr(args, "_configured_quant_method", "") or ""
        ).lower()
        if spec_quant != "int4":
            return
        if configured_quant and configured_quant != "awq":
            return
        if "awq" not in model_arg.lower() and quant_arg != "int4":
            return
        try:
            with open(_AWQ_GEMM_HINT_PATH, "w") as f:
                f.write("1")
        except Exception as e:
            bt.logging.warning(f"Could not write AWQ GEMM fallback hint: {e}")
        bt.logging.warning(
            f"{reason}. Requesting restart with --awq-gemm-fallback before failing hard."
        )
        sys.exit(_AWQ_GEMM_HINT_EXIT)

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
        _request_awq_gemm_fallback(msg)
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
        _request_awq_gemm_fallback(msg)
        bt.logging.error(f"{msg} Aborting. Use --force to override.")
        sys.exit(1)

    bt.logging.info(
        f"On-chain self-check PASSED: {len(chain_roots)} layer roots match."
    )

    # Sync architectural fields from chain spec so the miner derives
    # challenges identically to the validator (e.g. vocab_size enters
    # the Fiat-Shamir seed in derive_sampling_challenge).
    mismatches = []
    for field in (
        "vocab_size",
        "hidden_dim",
        "num_heads",
        "head_dim",
        "intermediate_dim",
        "num_layers",
    ):
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
    parser.add_argument(
        "--quant",
        default="auto",
        choices=["auto", "fp16", "fp8", "int8", "int4"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--spot-checks",
        type=int,
        default=None,
        help="Spot checks per block (default: from config)",
    )
    parser.add_argument(
        "--k-blocks",
        type=int,
        default=None,
        help="Blocks per GEMM to verify (default: from config)",
    )
    parser.add_argument(
        "--k-layers", type=int, default=None, help="Layers to challenge (None = auto)"
    )
    parser.add_argument(
        "--target-detection",
        type=float,
        default=None,
        help="Per-inference detection target for auto-k (default: 0.0625)",
    )
    parser.add_argument(
        "--k-experts",
        type=int,
        default=None,
        help="Experts to challenge per layer (None = auto)",
    )
    parser.add_argument(
        "--k-tokens", type=int, default=4, help="Tokens to sample for expert challenges"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
        help="Fraction of GPU memory for vLLM KV cache (default: 0.80).",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass ModelSpec cache"
    )
    parser.add_argument("--ssl-keyfile", default=None, help="TLS key file")
    parser.add_argument("--ssl-certfile", default=None, help="TLS cert file")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for auth (or set VERATHOS_API_KEY env var)",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="vLLM attention backend (e.g. TRITON_ATTN, FLASH_ATTN)",
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Print environment diagnostics and exit"
    )
    parser.add_argument(
        "--chain-config",
        default=None,
        help="Path to chain config JSON (compares local roots against chain)",
    )
    parser.add_argument(
        "--proof-v2-manifest",
        default=os.environ.get("VERATHOS_PROOF_V2_MANIFEST") or None,
        help="Path to the signed proof-v2 manifest document",
    )
    parser.add_argument(
        "--proof-v2-weight-catalog",
        default=os.environ.get("VERATHOS_PROOF_V2_WEIGHT_CATALOG") or None,
        help="Path to the manifest-bound static weight commitment catalog",
    )
    parser.add_argument(
        "--proof-v2-artifact-base-url",
        action="append",
        default=None,
        help=(
            "HTTPS base URL for content-addressed proof-v2 artifacts. "
            "Repeat to configure fallback mirrors."
        ),
    )
    parser.add_argument(
        "--proof-v2-artifact-cache-dir",
        default=os.environ.get("VERATHOS_PROOF_V2_ARTIFACT_CACHE_DIR") or None,
        help="Local cache directory for downloaded proof-v2 artifacts",
    )
    parser.add_argument(
        "--proof-v3-manifest",
        default=os.environ.get("VERATHOS_PROOF_V3_MANIFEST") or None,
        help="Path to the authority-signed proof-v3 projection manifest",
    )
    parser.add_argument(
        "--proof-v3-execution-profile",
        default=os.environ.get("VERATHOS_PROOF_V3_EXECUTION_PROFILE") or None,
        help="Path to the authority-signed proof-v3 execution profile",
    )
    parser.add_argument(
        "--proof-v3-calibration-set",
        default=os.environ.get("VERATHOS_PROOF_V3_CALIBRATION_SET") or None,
        help="Path to the manifest-bound proof-v3 calibration set",
    )
    parser.add_argument(
        "--proof-v3-attention-semantics",
        default=os.environ.get("VERATHOS_PROOF_V3_ATTENTION_SEMANTICS")
        or None,
        help="Path to the manifest-bound attention runtime semantics",
    )
    parser.add_argument(
        "--proof-v3-gdn-semantics",
        default=os.environ.get("VERATHOS_PROOF_V3_GDN_SEMANTICS") or None,
        help="Path to manifest-bound GDN runtime semantics when required",
    )
    parser.add_argument(
        "--proof-v3-lm-head-catalog",
        default=os.environ.get("VERATHOS_PROOF_V3_LM_HEAD_CATALOG") or None,
        help="Path to the manifest-bound LM-head commitment catalog",
    )
    parser.add_argument(
        "--proof-v3-projection-manifest",
        default=os.environ.get("VERATHOS_PROOF_V3_PROJECTION_MANIFEST")
        or None,
        help=(
            "Path to the authority-signed complete projection commitment "
            "manifest required by the lean v3 hard profile"
        ),
    )
    parser.add_argument(
        "--proof-v3-projection-catalog",
        default=os.environ.get("VERATHOS_PROOF_V3_PROJECTION_CATALOG")
        or None,
        help=(
            "Path to the manifest-bound complete projection commitment "
            "catalog required by the lean v3 hard profile"
        ),
    )
    parser.add_argument(
        "--proof-v3-runtime-encoding",
        default=os.environ.get("VERATHOS_PROOF_V3_RUNTIME_ENCODING") or None,
        help="Qualified activation encoding ID for this release",
    )
    parser.add_argument(
        "--proof-v3-weight-cache-dir",
        default=os.environ.get("VERATHOS_PROOF_V3_WEIGHT_CACHE_DIR") or None,
        help="Persistent cache for authenticated static proof-v3 weights",
    )
    parser.add_argument(
        "--proof-v3-max-decode-tokens",
        type=int,
        default=4096,
        help="Maximum decode tokens covered by the v3 execution profile",
    )
    parser.add_argument(
        "--proof-v3-max-records",
        type=int,
        default=64,
        help="Maximum pending postcommit v3 request records",
    )
    parser.add_argument(
        "--proof-v3-max-retained-bytes",
        type=int,
        default=8 << 30,
        help="Maximum retained v3 precommit witness bytes",
    )
    parser.add_argument(
        "--proof-v3-gdn-checkpoint-max-bytes",
        type=int,
        default=int(
            os.environ.get(
                "VERATHOS_PROOF_V3_GDN_CHECKPOINT_MAX_BYTES",
                8 << 30,
            )
        ),
        help=(
            "Maximum pageable host bytes for prover-local GDN decode "
            "checkpoints; set this and the per-request cap to zero to disable"
        ),
    )
    parser.add_argument(
        "--proof-v3-gdn-checkpoint-request-max-bytes",
        type=int,
        default=int(
            os.environ.get(
                "VERATHOS_PROOF_V3_GDN_CHECKPOINT_REQUEST_MAX_BYTES",
                2 << 30,
            )
        ),
        help="Maximum pageable GDN checkpoint bytes retained per request",
    )
    parser.add_argument(
        "--proof-v3-gdn-checkpoint-interval-tokens",
        type=int,
        default=int(
            os.environ.get(
                "VERATHOS_PROOF_V3_GDN_CHECKPOINT_INTERVAL_TOKENS",
                1024,
            )
        ),
        help="Target decode-token interval for prover-local GDN checkpoints",
    )
    parser.add_argument(
        "--proof-v3-ttl-seconds",
        type=float,
        default=300.0,
        help="Maximum hard-reveal delay after a v3 precommit",
    )
    parser.add_argument(
        "--allowed-proof-protocol-versions",
        default=os.environ.get(
            "VERATHOS_PROOF_PROTOCOL_ALLOWED_VERSIONS",
            "1,3",
        ),
        help="Comma-separated owner-allowed inference proof versions",
    )
    parser.add_argument(
        "--miner-hotkey-ss58",
        default=os.environ.get("VERATHOS_MINER_HOTKEY_SS58") or None,
        help="Serving miner hotkey identity bound into proof-v3 requests",
    )
    parser.add_argument(
        "--evm-rpc-url",
        default=None,
        help="EVM RPC URL (overrides rpc_url in chain config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start even if on-chain root comparison fails",
    )
    # Batch mode (continuous batching)
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        default=True,
        help="Enable continuous batching for concurrent requests (default: on)",
    )
    parser.add_argument(
        "--no-batch-mode",
        action="store_true",
        help="Disable continuous batching (legacy single-request mode)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent requests in batch mode (None = auto-detect)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM max_num_seqs (max concurrent decode sequences). "
        "None = vLLM default (1024). On Mamba/GDN hybrid models "
        "(Qwen3.5, Qwen3.6) the default exceeds the Mamba state "
        "block budget; the first launch attempt writes the auto-"
        "tuned value to /tmp/verathos_mamba_max_num_seqs and exits "
        "with code 42, then the launcher restarts with this flag.",
    )
    parser.add_argument(
        "--awq-gemm-fallback",
        action="store_true",
        help="Force vLLM plain AWQ GEMM instead of AWQ-Marlin. "
        "Used automatically when the first AWQ-Marlin model "
        "load fails with a known backend error.",
    )
    parser.add_argument(
        "--proof-threads",
        type=int,
        default=None,
        help="Max concurrent proof threads (None = auto-detect from CPU/VRAM)",
    )
    parser.add_argument(
        "--proof-max-pending",
        type=int,
        default=None,
        help="Max pending proofs (running + queued) before returning 503 (None = 2x proof-threads)",
    )
    parser.add_argument(
        "--proof-matmul-backend",
        default="batched",
        choices=["gpu", "cpu", "adaptive", "batched"],
        help="Matmul backend for proof generation (default: batched). "
        "'batched' collects matmuls, groups by layer, single-stream dispatch. "
        "'gpu' uses non-blocking GPU-first with CPU f32 spillover. "
        "'cpu' forces CPU-only f32 SGEMM. "
        "'adaptive' is an alias for 'gpu'.",
    )
    parser.add_argument(
        "--proof-gpu-matmul-limit",
        type=int,
        default=0,
        help="Max concurrent GPU matmul allocations (0 = auto from SM count)",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip pre-flight GPU occupancy check",
    )
    # EVM identity (passed by neurons/miner.py for anti-hijacking)
    parser.add_argument(
        "--evm-address",
        default=None,
        help="Miner's EVM address (for receipt validation + identity challenge)",
    )
    parser.add_argument(
        "--evm-private-key",
        default=None,
        help="Miner's EVM private key hex (for identity challenge signing)",
    )
    # TEE (Trusted Execution Environment) — confidential GPU mode
    parser.add_argument(
        "--tee-enabled",
        action="store_true",
        help="Enable TEE mode (E2E encryption + attestation)",
    )
    parser.add_argument(
        "--tee-platform",
        default="mock",
        choices=["mock", "tdx", "sev-snp", "gpu"],
        help="TEE attestation platform (default: mock)",
    )
    parser.add_argument(
        "--tee-skip-proofs",
        action="store_true",
        default=None,
        help="Skip VeraLLM proof generation (use hardware attestation instead). "
        "Default: True when --tee-enabled is set.",
    )
    parser.add_argument(
        "--capacity-audit-state-file", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning"],
        help="Logging level (default: info)",
    )
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
            vram_gb = props.total_memory / (1024**3)
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
    try:
        state.allowed_proof_protocol_versions = tuple(
            sorted(
                {
                    int(part.strip())
                    for part in args.allowed_proof_protocol_versions.split(",")
                    if part.strip()
                }
            )
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "--allowed-proof-protocol-versions must be comma-separated integers"
        ) from exc
    if not state.allowed_proof_protocol_versions:
        raise SystemExit("--allowed-proof-protocol-versions must not be empty")
    if (
        list(state.allowed_proof_protocol_versions)
        != sorted(set(state.allowed_proof_protocol_versions))
        or 2 in state.allowed_proof_protocol_versions
        or any(
            version < 1 or version > 255
            for version in state.allowed_proof_protocol_versions
        )
    ):
        raise SystemExit(
            "--allowed-proof-protocol-versions is invalid or enables reserved v2"
        )
    if not set(state.allowed_proof_protocol_versions).intersection({1, 3}):
        raise SystemExit(
            "subnet allows no inference proof protocol supported by this "
            "miner binary"
        )
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
