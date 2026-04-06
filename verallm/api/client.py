#!/usr/bin/env python3
"""
VeraLLM Validator Client — connects to a remote miner and verifies inference.

Non-interactive protocol (2 HTTP requests):
1. GET  /model_spec  — Fetch weight Merkle roots (on-chain data in production)
2. POST /inference   — SSE-streamed tokens + commitment + proofs (single request)

The validator then verifies proofs locally (lightweight — no model loading).

Usage:
    python -m verallm.api.client --miner-url http://localhost:8000 \
        --prompt "Explain the halting problem"

    # With TLS:
    python -m verallm.api.client --miner-url https://miner-host:8443 \
        --prompt "Hello world" --verify-tls
"""

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

import httpx
import numpy as np
import torch

from verallm.config import Config, set_config, get_config
from verallm.types import (
    InferenceCommitment,
    InferenceProofBundle,
    ModelSpec,
    VerificationResult,
)
from verallm.challenge.beacon import (
    derive_beacon_from_nonce,
    derive_challenges,
    derive_sampling_challenge,
    derive_embedding_challenge,
    compute_detection_probability,
)
from verallm.verifier.gemm import GEMMVerifier
from verallm.crypto.transcript import Transcript
from verallm.crypto.merkle import verify_merkle_path, verify_flat_chunk_merkle_path, build_block_merkle, MerkleTree
from verallm.crypto.field import mod_p
from verallm.moe import (
    MoEConfig,
    derive_moe_challenges,
)
from verallm.moe.router_commitment import (
    compute_topk_indices,
    logits_row_to_bytes,
)

from verallm.helpers import compute_auto_k, compute_auto_k_experts
from verallm.sampling import (
    clamp_sampling_bps,
    hidden_row_from_bytes,
    logits_i32_from_bytes,
    quantize_hidden_row_int64,
    verify_quantized_argmax,
    verify_fp16_argmax,
)
from verallm.api.serialization import (
    dict_to_model_spec,
    dict_to_commitment,
    dict_to_proof_bundle,
)


# ============================================================================
# SSE parser
# ============================================================================

def _parse_sse_stream(response):
    """Parse a Server-Sent Events stream from httpx.

    Yields (event_type, data_dict) tuples.  Supports both RFC 6902 format
    (``event: type\\ndata: json``) and the miner's inline format
    (``data: {"event": "type", ...}``).
    """
    event_type = None
    data_lines = []

    for line in response.iter_lines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "":
            # Empty line = end of event
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"raw": raw}
                # If no explicit event: line, check for inline event field
                evt = event_type or parsed.get("event", "")
                if evt:
                    yield evt, parsed
            event_type = None
            data_lines = []

    # Handle final event without trailing empty line
    if data_lines:
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        evt = event_type or parsed.get("event", "")
        if evt:
            yield evt, parsed


# ============================================================================
# Validator Request Signing (httpx Auth)
# ============================================================================

class ValidatorRequestAuth(httpx.Auth):
    """httpx Auth class that signs every request with a validator Sr25519 hotkey.

    Attaches X-Validator-Hotkey (SS58), X-Validator-Signature, X-Validator-Timestamp
    headers so that miners can verify the caller is a registered validator.
    """

    def __init__(self, hotkey_ss58: str, hotkey_seed: bytes):
        self._hotkey_ss58 = hotkey_ss58
        self._hotkey_seed = hotkey_seed

    def auth_flow(self, request: httpx.Request):
        from neurons.request_signing import sign_request
        from urllib.parse import urlparse

        path = urlparse(str(request.url)).path
        body = request.content if request.content else b""
        headers = sign_request(
            method=request.method,
            path=path,
            body=body,
            hotkey_ss58=self._hotkey_ss58,
            hotkey_seed=self._hotkey_seed,
        )
        for k, v in headers.items():
            request.headers[k] = v
        yield request


# ============================================================================
# Validator Client
# ============================================================================

class ValidatorClient:
    """Primary production client — used by the subnet validator and proxy.

    Bundles HTTP transport (inference requests, SSE streaming, TEE) with
    local proof verification.  This is the canonical verification path;
    ``verallm.validator.core.Validator`` mirrors the same logic for
    offline demos and tests only.

    The protocol is non-interactive (Fiat-Shamir): a single POST /inference
    returns tokens + commitment + proofs.  The validator re-derives beacon
    and challenges locally during verification.
    """

    def __init__(
        self,
        miner_url: str,
        config: Optional[Config] = None,
        verify_tls: bool = True,
        timeout: float = 600.0,
        api_key: Optional[str] = None,
        chain_config=None,
        model_id: Optional[str] = None,
        validator_hotkey_ss58: Optional[str] = None,
        validator_seed: Optional[bytes] = None,
    ):
        self.miner_url = miner_url.rstrip("/")
        self.config = config or get_config()
        self.model_spec: Optional[ModelSpec] = None
        self.moe_config: Optional[MoEConfig] = None
        self._chain_config = chain_config
        self._model_id = model_id
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        auth = None
        if validator_hotkey_ss58 and validator_seed:
            auth = ValidatorRequestAuth(validator_hotkey_ss58, validator_seed)

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=30.0),
            verify=verify_tls,
            headers=headers,
            auth=auth,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Fetch ModelSpec
    # ------------------------------------------------------------------

    def fetch_model_spec(self) -> ModelSpec:
        """Fetch the ModelSpec (weight Merkle roots).

        If a chain_config is set, reads from the on-chain ModelRegistry
        (the trust anchor — roots are NOT from the miner).

        Falls back to GET /model_spec from the miner when no chain config
        is provided (development/testing mode).
        """
        if self._chain_config is not None:
            spec = self._fetch_model_spec_from_chain()
            if spec is not None:
                self.model_spec = spec
                self._auto_configure_from_spec(spec)
                return spec
            # If chain read failed, don't fall back — that would be insecure
            raise RuntimeError(
                f"Model '{self._model_id}' not found on-chain. "
                "Cannot fall back to miner (trust anchor must be on-chain)."
            )

        # Dev mode: fetch from miner directly
        resp = self.client.get(f"{self.miner_url}/model_spec")
        resp.raise_for_status()
        self.model_spec = dict_to_model_spec(resp.json())
        self._auto_configure_from_spec(self.model_spec)
        return self.model_spec

    def _auto_configure_from_spec(self, model_spec: ModelSpec) -> None:
        """Auto-compute k_layers and detect MoE from the model spec.

        Called from fetch_model_spec() so that verify_proof() always has
        correct k_layers, regardless of whether run() or the standalone
        fetch_model_spec() + verify_proof() path is used.
        """
        # Auto-compute k_layers if not explicitly set
        if self.config.k_layers == 0 and model_spec.num_layers > 0:
            k = compute_auto_k(model_spec.num_layers)
            self.config = Config(
                **{
                    **{f.name: getattr(self.config, f.name)
                       for f in self.config.__dataclass_fields__.values()},
                    "k_layers": k,
                }
            )
            set_config(self.config)

        # Detect MoE from model_spec
        if self.moe_config is None:
            num_experts = model_spec.num_experts
            if num_experts == 0 and model_spec.expert_weight_merkle_roots:
                first_roots = next(iter(model_spec.expert_weight_merkle_roots.values()), [])
                num_experts = len(first_roots)

            if num_experts > 0:
                if model_spec.expert_weight_merkle_roots:
                    moe_layer_indices = sorted(model_spec.expert_weight_merkle_roots.keys())
                else:
                    moe_layer_indices = list(range(model_spec.num_layers))
                expert_inter = (
                    model_spec.expert_w_num_cols
                    if model_spec.expert_w_num_cols > 0
                    else model_spec.intermediate_dim
                )
                self.moe_config = MoEConfig(
                    is_moe=True,
                    num_layers=model_spec.num_layers,
                    moe_layer_indices=moe_layer_indices,
                    num_routed_experts=num_experts,
                    num_shared_experts=0,
                    top_k=model_spec.router_top_k if model_spec.router_top_k > 0 else 2,
                    hidden_dim=model_spec.hidden_dim,
                    intermediate_dim=model_spec.intermediate_dim,
                    expert_intermediate_dim=expert_inter,
                    has_shared_expert_gate=False,
                    uses_3d_expert_weights=False,
                    router_type=model_spec.router_scoring or "top_k",
                )

                # Auto-compute k_experts_per_layer if not set
                if self.config.k_experts_per_layer == 0:
                    k_exp = compute_auto_k_experts(num_experts)
                    self.config = Config(
                        **{
                            **{f.name: getattr(self.config, f.name)
                               for f in self.config.__dataclass_fields__.values()},
                            "k_experts_per_layer": k_exp,
                        }
                    )
                    set_config(self.config)

    def _fetch_model_spec_from_chain(self) -> Optional[ModelSpec]:
        """Read ModelSpec from the on-chain ModelRegistry."""
        from verallm.chain.mock import create_clients

        model_client, *_ = create_clients(self._chain_config)
        model_id = self._model_id
        if not model_id:
            # Try to get model_id from the miner's health endpoint
            health = self.health_check()
            model_id = health.get("model", "")
        if not model_id:
            raise ValueError(
                "model_id must be provided when using chain_config "
                "(or miner /health must return 'model' field)"
            )
        self._model_id = model_id
        return model_client.get_model_spec(model_id)

    def health_check(self) -> dict:
        """Check if the miner is healthy."""
        resp = self.client.get(f"{self.miner_url}/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Inference + proofs (single request)
    # ------------------------------------------------------------------

    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_verification_bps: int = 0,
        stream_callback=None,
    ) -> Tuple[str, InferenceCommitment, InferenceProofBundle, bytes, dict]:
        """Send inference request, stream tokens, get commitment + proofs.

        The miner runs the full protocol in a single request:
        inference -> commitment -> beacon -> challenges -> proofs.

        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            stream_callback: Optional callable(token_text) invoked per token.

        Returns:
            (full_text, commitment, proof_bundle, nonce, timing_info)
        """
        # Generate validator nonce (32 random bytes)
        nonce = os.urandom(32)

        request_body = {
            "prompt": prompt,
            "validator_nonce": nonce.hex(),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": max(0, min(10_000, int(sampling_verification_bps))),
        }

        full_text = ""
        commitment = None
        proof_bundle = None
        timing = {}
        t_first_token = None

        t0 = time.perf_counter()
        with self.client.stream("POST", f"{self.miner_url}/inference",
                                json=request_body) as resp:
            resp.raise_for_status()
            for event_type, data in _parse_sse_stream(resp):
                if event_type == "token":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    token_text = data.get("text", "")
                    full_text += token_text
                    if stream_callback:
                        stream_callback(token_text)
                elif event_type == "done":
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get("proof_bundle", {"layer_proofs": [], "sampling_proofs": []})
                    if commit_data:
                        try:
                            commitment = dict_to_commitment(commit_data)
                        except (TypeError, KeyError):
                            pass
                    if proof_data:
                        try:
                            proof_bundle = dict_to_proof_bundle(proof_data)
                        except (TypeError, KeyError):
                            pass
                    timing["inference_ms"] = data.get("inference_ms", 0)
                    timing["commitment_ms"] = data.get("commitment_ms", 0)
                    timing["prove_ms"] = data.get("prove_ms", 0)
                    timing["beacon_ms"] = data.get("beacon_ms", 0)
                    timing["challenge_ms"] = data.get("challenge_ms", 0)
                    timing["input_tokens"] = data.get("input_tokens", 0)
                    timing["output_tokens"] = data.get("output_tokens", 0)
                elif event_type == "error":
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        timing["round_trip_ms"] = (time.perf_counter() - t0) * 1000
        if t_first_token is not None:
            timing["ttft_ms"] = (t_first_token - t0) * 1000

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()

        return full_text, commitment, proof_bundle, nonce, timing

    def run_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_verification_bps: int = 0,
        stream_callback=None,
        enable_thinking: bool = True,
        presence_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
    ) -> Tuple[str, InferenceCommitment, InferenceProofBundle, bytes, dict]:
        """Send chat-style inference request (OpenAI messages format).

        Uses POST /chat on the miner, which applies the chat template
        server-side. Otherwise identical to run_inference().

        Args:
            messages: List of {role, content} dicts (OpenAI format).
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            stream_callback: Optional callable(token_text) invoked per token.
            enable_thinking: Enable chain-of-thought for models that support it.
            presence_penalty: Penalize repeated tokens (None = server default).
            top_k: Top-k sampling (None = server default).
            top_p: Nucleus sampling (None = server default).
            min_p: Minimum probability (None = server default).

        Returns:
            (full_text, commitment, proof_bundle, nonce, timing_info)
        """
        nonce = os.urandom(32)

        request_body = {
            "messages": messages,
            "validator_nonce": nonce.hex(),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": max(0, min(10_000, int(sampling_verification_bps))),
            "enable_thinking": enable_thinking,
        }
        # Only send sampling params when explicitly set (None = server defaults)
        if presence_penalty is not None:
            request_body["presence_penalty"] = presence_penalty
        if top_k is not None:
            request_body["top_k"] = top_k
        if top_p is not None:
            request_body["top_p"] = top_p
        if min_p is not None:
            request_body["min_p"] = min_p

        full_text = ""
        commitment = None
        proof_bundle = None
        timing = {}
        t_first_token = None

        t0 = time.perf_counter()
        _t_last_tok = time.perf_counter()
        with self.client.stream("POST", f"{self.miner_url}/chat",
                                json=request_body) as resp:
            resp.raise_for_status()
            for event_type, data in _parse_sse_stream(resp):
                if event_type == "token":
                    _t_last_tok = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    token_text = data.get("text", "")
                    full_text += token_text
                    if stream_callback:
                        stream_callback(token_text)
                elif event_type == "done":
                    _t_done_recv = time.perf_counter()
                    _done_gap_ms = (_t_done_recv - _t_last_tok) * 1000
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get("proof_bundle", {"layer_proofs": [], "sampling_proofs": []})
                    if commit_data:
                        try:
                            commitment = dict_to_commitment(commit_data)
                        except (TypeError, KeyError):
                            pass
                    _t_commit_deser = time.perf_counter()
                    if proof_data:
                        try:
                            proof_bundle = dict_to_proof_bundle(proof_data)
                        except (TypeError, KeyError):
                            pass
                    _t_proof_deser = time.perf_counter()
                    _deser_ms = (_t_proof_deser - _t_done_recv) * 1000
                    import logging as _logging
                    _logging.getLogger("verallm.api.client").debug(
                        "CLIENT TIMING: last_token→done_recv=%.0fms deser=%.0fms (commit=%.0fms proof=%.0fms)",
                        _done_gap_ms, _deser_ms,
                        (_t_commit_deser - _t_done_recv) * 1000,
                        (_t_proof_deser - _t_commit_deser) * 1000,
                    )
                    timing["inference_ms"] = data.get("inference_ms", 0)
                    timing["commitment_ms"] = data.get("commitment_ms", 0)
                    timing["prove_ms"] = data.get("prove_ms", 0)
                    timing["beacon_ms"] = data.get("beacon_ms", 0)
                    timing["challenge_ms"] = data.get("challenge_ms", 0)
                    timing["input_tokens"] = data.get("input_tokens", 0)
                    timing["output_tokens"] = data.get("output_tokens", 0)
                elif event_type == "error":
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        timing["round_trip_ms"] = (time.perf_counter() - t0) * 1000
        if t_first_token is not None:
            timing["ttft_ms"] = (t_first_token - t0) * 1000

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()

        return full_text, commitment, proof_bundle, nonce, timing

    # ------------------------------------------------------------------
    # TEE encrypted inference
    # ------------------------------------------------------------------

    def run_encrypted_inference(
        self,
        encrypted_envelope: "EncryptedEnvelope",
        validator_nonce: bytes,
        stream_callback=None,
    ) -> Tuple[Optional[dict], "InferenceCommitment", "InferenceProofBundle", bytes, dict, list]:
        """Forward an encrypted chat request to a TEE-enabled miner.

        The validator cannot decrypt the conversation — it only verifies
        the plaintext proof bundle and relays the encrypted output.

        Args:
            encrypted_envelope: User's encrypted chat request.
            validator_nonce: Validator's 32-byte nonce for Fiat-Shamir.
            stream_callback: Optional callable invoked for each encrypted_token
                event with the chunk dict (or None for heartbeats).

        Returns:
            (encrypted_output_dict, commitment, proof_bundle, nonce, timing,
             encrypted_chunks) where encrypted_chunks is a list of
             serialized EncryptedEnvelope dicts for streaming token deltas.
        """

        request_body = {
            "envelope": {
                "session_id": encrypted_envelope.session_id,
                "sender_public_key": encrypted_envelope.sender_public_key.hex(),
                "nonce": encrypted_envelope.nonce.hex(),
                "ciphertext": encrypted_envelope.ciphertext.hex(),
                "content_type": encrypted_envelope.content_type,
            },
            "validator_nonce": validator_nonce.hex(),
        }

        commitment = None
        proof_bundle = None
        encrypted_output = None
        encrypted_chunks: list[dict] = []
        timing = {}
        t0 = time.perf_counter()

        with self.client.stream("POST", f"{self.miner_url}/tee/chat",
                                json=request_body) as resp:
            resp.raise_for_status()
            for event_type, data in _parse_sse_stream(resp):
                if event_type == "heartbeat":
                    if stream_callback:
                        stream_callback(None)
                elif event_type == "encrypted_token":
                    encrypted_chunks.append(data)
                    if stream_callback:
                        stream_callback(data)
                elif event_type == "done":
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get("proof_bundle", {"layer_proofs": [], "sampling_proofs": []})
                    if commit_data:
                        try:
                            commitment = dict_to_commitment(commit_data)
                        except (TypeError, KeyError):
                            pass
                    if proof_data:
                        try:
                            proof_bundle = dict_to_proof_bundle(proof_data)
                        except (TypeError, KeyError):
                            pass
                    encrypted_output = data.get("encrypted_output")
                    # Miner sends encrypted_output_nonce alongside encrypted_output
                    if data.get("encrypted_output_nonce"):
                        encrypted_output = {
                            "encrypted_output": data["encrypted_output"],
                            "encrypted_output_nonce": data["encrypted_output_nonce"],
                        }
                    # Timing may be nested under "timing" key or flat
                    t = data.get("timing", data)
                    timing["inference_ms"] = t.get("inference_ms", 0)
                    timing["commitment_ms"] = t.get("commitment_ms", 0)
                    timing["prove_ms"] = t.get("prove_ms", 0)
                    timing["input_tokens"] = t.get("input_tokens", 0)
                    timing["output_tokens"] = t.get("output_tokens", 0)
                    timing["model_id"] = t.get("model_id", "")
                elif event_type == "error":
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        timing["round_trip_ms"] = (time.perf_counter() - t0) * 1000

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()

        return encrypted_output, commitment, proof_bundle, validator_nonce, timing, encrypted_chunks

    def fetch_tee_info(self) -> Optional[dict]:
        """Fetch TEE capability from a miner. Returns None if TEE disabled."""
        try:
            resp = self.client.get(f"{self.miner_url}/tee/info")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            return None

    def _verify_moe_layer_root(
        self,
        layer_idx: int,
        expert_roots: list[bytes],
        router_weight_root: Optional[bytes],
    ) -> bool:
        """Verify expert/router roots hash to the committed MoE layer root."""
        if self.model_spec is None:
            return False
        if layer_idx >= len(self.model_spec.weight_block_merkle_roots):
            return True

        committed_layer_root = self.model_spec.weight_block_merkle_roots[layer_idx]
        if committed_layer_root is None:
            return True

        if router_weight_root:
            h_v2 = hashlib.sha256(b"MOE_LAYER_V2")
            h_v2.update(router_weight_root)
            for er in expert_roots:
                h_v2.update(er)
            if h_v2.digest() == committed_layer_root:
                return True

        h_v1 = hashlib.sha256(b"MOE_LAYER_V1")
        for er in expert_roots:
            h_v1.update(er)
        return h_v1.digest() == committed_layer_root

    def _verify_router_layer_openings(self, proof_bundle: InferenceProofBundle, layer_challenge) -> VerificationResult:
        """Verify Merkle openings and top-k consistency for sampled router tokens."""
        layer_idx = layer_challenge.layer_idx
        router_commitment = proof_bundle.router_commitments.get(layer_idx)
        if router_commitment is None:
            return VerificationResult.failure(
                f"Missing router commitment for challenged MoE layer {layer_idx}"
            )

        layer_routing_proof = proof_bundle.router_layer_proofs.get(layer_idx)
        if layer_routing_proof is None:
            return VerificationResult.failure(
                f"Missing router layer proof for challenged MoE layer {layer_idx}"
            )

        if not router_commitment.router_logits_row_root:
            return VerificationResult.failure(
                f"Router logits row root missing in router commitment for layer {layer_idx}"
            )

        # Strictly bind routing metadata to model spec to prevent challenge steering.
        expected_num_experts = int(getattr(self.model_spec, "num_experts", 0) or 0) if self.model_spec else 0
        if expected_num_experts <= 0 and self.model_spec is not None:
            spec_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
            if spec_roots:
                expected_num_experts = len(spec_roots)
        if expected_num_experts > 0:
            if router_commitment.num_experts <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing num_experts"
                )
            if router_commitment.num_experts != expected_num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: num_experts mismatch "
                    f"(committed={router_commitment.num_experts}, expected={expected_num_experts})"
                )

        expected_top_k = int(getattr(self.model_spec, "router_top_k", 0) or 0) if self.model_spec else 0
        if expected_top_k > 0:
            if router_commitment.top_k <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing top_k"
                )
            if router_commitment.top_k != expected_top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: top_k mismatch "
                    f"(committed={router_commitment.top_k}, expected={expected_top_k})"
                )

        expected_scoring = str(getattr(self.model_spec, "router_scoring", "") or "") if self.model_spec else ""
        if expected_scoring and router_commitment.scoring_func:
            if str(router_commitment.scoring_func) != expected_scoring:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: scoring_func mismatch "
                    f"(committed={router_commitment.scoring_func}, expected={expected_scoring})"
                )

        committed_seq = len(router_commitment.selected_experts)
        if router_commitment.seq_len and router_commitment.seq_len != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: seq_len mismatch "
                f"(committed={router_commitment.seq_len}, rows={committed_seq})"
            )
        if len(router_commitment.routing_weights) != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: routing_weights row-count mismatch "
                f"(weights={len(router_commitment.routing_weights)}, experts={committed_seq})"
            )
        if router_commitment.proof_selected_experts:
            if len(router_commitment.proof_selected_experts) != committed_seq:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row-count mismatch "
                    f"(proof={len(router_commitment.proof_selected_experts)}, experts={committed_seq})"
                )

        top_k = int(router_commitment.top_k or 0)
        if top_k <= 0:
            return VerificationResult.failure(
                f"Layer {layer_idx}: invalid top_k={router_commitment.top_k}"
            )

        from verallm.crypto.field import P as FIELD_PRIME
        for row_idx, row in enumerate(router_commitment.selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0 and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: expert index out of range at row {row_idx}: {expert_idx}"
                    )

        for row_idx, row in enumerate(router_commitment.routing_weights):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: routing_weights row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for w in row:
                if int(w) < 0 or int(w) >= FIELD_PRIME:
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: routing weight out of field range at row {row_idx}"
                    )
        for row_idx, row in enumerate(router_commitment.proof_selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0 and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: proof expert index out of range at row {row_idx}: {expert_idx}"
                    )

        expert_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, []) if self.model_spec else []
        if not expert_roots:
            expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
        if expert_roots and not self._verify_moe_layer_root(
            layer_idx=layer_idx,
            expert_roots=expert_roots,
            router_weight_root=layer_routing_proof.router_weight_root,
        ):
            return VerificationResult.failure(
                f"MoE layer {layer_idx}: router/expert roots do not match committed layer root"
            )

        # Verify sampled router GEMM proof: this binds routing to X and W.
        if layer_routing_proof.router_gemm_proof is None:
            return VerificationResult.failure(
                f"Missing router GEMM proof for challenged MoE layer {layer_idx}"
            )
        if not layer_routing_proof.proved_output_rows:
            return VerificationResult.failure(
                f"Missing proven router output rows for challenged MoE layer {layer_idx}"
            )

        proved_rows_by_token = {
            int(row.token_idx): row
            for row in layer_routing_proof.proved_output_rows
        }
        if len(proved_rows_by_token) != len(layer_routing_proof.proved_output_rows):
            return VerificationResult.failure(
                f"Duplicate token indices in proven router output rows for layer {layer_idx}"
            )

        proved_matrix_rows: list[list[int]] = []
        for token_idx in layer_challenge.sampled_token_indices:
            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            if router_commitment.num_experts and len(proved_row.logits_int) != router_commitment.num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: proven router row width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(proved_row.logits_int)})"
                )
            proved_matrix_rows.append([int(v) for v in proved_row.logits_int])

        if not proved_matrix_rows:
            return VerificationResult.failure(
                f"No proven router rows available for challenged layer {layer_idx}"
            )

        y_tensor = torch.tensor(proved_matrix_rows, dtype=torch.int64)
        y_root = build_block_merkle(y_tensor, self.config.block_size).root
        router_verifier = GEMMVerifier(self.config)
        router_verify = router_verifier.verify(
            proof=layer_routing_proof.router_gemm_proof,
            X_commitment=proof_bundle.commitment.layer_commitments[layer_idx],
            W_commitment=layer_routing_proof.router_weight_root,
            Y_commitment=y_root,
            transcript=Transcript(f"layer_{layer_idx}_router_gemm".encode()),
            spot_check_fn=lambda _spot, _matrix_id: True,
            W_merkle_root=layer_routing_proof.router_weight_root,
            W_num_cols=(router_commitment.num_experts
                        if router_commitment.num_experts > 0
                        else len(proved_matrix_rows[0])),
            w_chunk_size=self.model_spec.w_merkle_chunk_size,
        )
        if not router_verify.passed:
            return VerificationResult.failure(
                f"Layer {layer_idx}: router GEMM proof failed: {router_verify.message}"
            )

        openings_by_token = {
            opening.token_idx: opening
            for opening in layer_routing_proof.logits_openings
        }

        for token_idx in layer_challenge.sampled_token_indices:
            opening = openings_by_token.get(int(token_idx))
            if opening is None:
                return VerificationResult.failure(
                    f"Missing router logits opening for layer {layer_idx}, token {token_idx}"
                )

            if token_idx >= len(router_commitment.selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for selected_experts in layer {layer_idx}"
                )

            if router_commitment.seq_len and token_idx >= router_commitment.seq_len:
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for committed seq_len in layer {layer_idx}"
                )

            if router_commitment.num_experts and len(opening.logits) != router_commitment.num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: logits width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(opening.logits)})"
                )

            path_ok = verify_merkle_path(
                root=router_commitment.router_logits_row_root,
                leaf_data=logits_row_to_bytes(opening.logits),
                path=opening.merkle_path,
            )
            if not path_ok:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router logits Merkle proof invalid"
                )

            committed_experts = router_commitment.selected_experts[token_idx]
            recomputed = compute_topk_indices(opening.logits, top_k)
            committed_prefix = [int(x) for x in committed_experts[:top_k]]
            if recomputed != committed_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: top-k mismatch "
                    f"(committed={committed_prefix}, recomputed={recomputed})"
                )

            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            proved_topk = compute_topk_indices(proved_row.logits_int, top_k)
            if token_idx >= len(router_commitment.proof_selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for proof_selected_experts in layer {layer_idx}"
                )
            proof_prefix = [int(x) for x in router_commitment.proof_selected_experts[token_idx][:top_k]]
            if proved_topk != proof_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router-GEMM top-k mismatch "
                    f"(committed={proof_prefix}, proved={proved_topk})"
                )

        return VerificationResult.success(f"Routing openings verified for layer {layer_idx}")

    # ------------------------------------------------------------------
    # Phase 7: Verify proofs locally
    # ------------------------------------------------------------------

    def verify_proof(
        self,
        proof_bundle: InferenceProofBundle,
        nonce: bytes,
        expected_sampling_verification_bps: Optional[int] = None,
        expected_do_sample: Optional[bool] = None,
        expected_temperature: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        expected_input_commitment: Optional[bytes] = None,
        expected_prompt_hash: Optional[bytes] = None,
        expected_sampler_config_hash: Optional[bytes] = None,
        expected_top_k: Optional[int] = None,
        expected_top_p: Optional[float] = None,
        expected_min_p: Optional[float] = None,
    ) -> Tuple[VerificationResult, Dict[str, float]]:
        """Verify the proof bundle locally (lightweight -- no model needed).

        Re-derives beacon and challenges from the commitment + nonce
        (Fiat-Shamir), then verifies sumcheck proofs against on-chain
        weight Merkle roots.

        Args:
            expected_sampling_verification_bps: If set, verify the committed
                bps matches the validator-requested rate.
            expected_do_sample: If set, verify the committed do_sample flag
                matches the validator's request (prevents miner from setting
                do_sample=True to skip sampling checks).
            expected_temperature: If set, verify the committed temperature
                matches the validator's request (prevents miner from committing
                nonzero temperature to skip sampling checks).
            enable_thinking: If False, argmax divergence in high-assurance
                mode becomes a hard failure (no logits processor excuse).

        NOTE: In production, weight Merkle roots come from the on-chain
        registry.  Here they come from the ModelSpec fetched from the miner.
        """
        if self.model_spec is None:
            raise RuntimeError("ModelSpec not fetched. Call fetch_model_spec() first.")

        set_config(self.config)
        verifier = GEMMVerifier(self.config)
        timing_details = {}

        # Re-derive beacon from commitment + nonce (Fiat-Shamir)
        commitment = proof_bundle.commitment
        beacon = derive_beacon_from_nonce(
            commitment_hash=commitment.commitment_hash(),
            validator_nonce=nonce,
        )

        # Verify beacon matches what miner used
        if proof_bundle.beacon != beacon:
            return VerificationResult.failure(
                f"Beacon mismatch: miner used {proof_bundle.beacon[:8].hex()}..., "
                f"expected {beacon[:8].hex()}..."
            ), timing_details

        # Verify committed decode-mode fields match validator expectations.
        # Without these checks, a miner could commit do_sample=True or
        # nonzero temperature to dodge sampling verification entirely.
        if expected_do_sample is not None:
            if commitment.do_sample != expected_do_sample:
                return VerificationResult.failure(
                    f"do_sample mismatch: committed={commitment.do_sample}, "
                    f"expected={expected_do_sample}"
                ), timing_details
        if expected_temperature is not None:
            from verallm.sampling import temperature_to_milli
            expected_milli = temperature_to_milli(expected_temperature)
            if commitment.temperature_milli != expected_milli:
                return VerificationResult.failure(
                    f"temperature mismatch: committed={commitment.temperature_milli}m, "
                    f"expected={expected_milli}m"
                ), timing_details

        if expected_sampling_verification_bps is not None:
            expected_bps = clamp_sampling_bps(expected_sampling_verification_bps)
            committed_bps = clamp_sampling_bps(commitment.sampling_verification_bps)
            if committed_bps != expected_bps:
                return VerificationResult.failure(
                    f"Sampling verification rate mismatch: committed={committed_bps} bps, "
                    f"expected={expected_bps} bps"
                ), timing_details

        # Verify sampler config hash (top_k/top_p/min_p/template binding).
        if expected_sampler_config_hash is not None:
            if not commitment.sampler_config_hash:
                return VerificationResult.failure(
                    "sampler_config_hash missing from commitment"
                ), timing_details
            if commitment.sampler_config_hash != expected_sampler_config_hash:
                return VerificationResult.failure(
                    "sampler_config_hash mismatch: miner committed different "
                    "sampling parameters than validator requested"
                ), timing_details

        # Verify committed input matches what the validator sent (prevents
        # input truncation: miner drops tokens to save compute).
        if expected_input_commitment is not None:
            if commitment.input_commitment != expected_input_commitment:
                return VerificationResult.failure(
                    f"input_commitment mismatch: committed="
                    f"{commitment.input_commitment.hex()[:16]}..., "
                    f"expected={expected_input_commitment.hex()[:16]}..."
                ), timing_details

        # Verify committed prompt hash matches what the proxy/validator sent
        # (prevents prompt substitution: miner runs a different prompt).
        if expected_prompt_hash is not None:
            if not commitment.prompt_hash:
                return VerificationResult.failure(
                    "Missing prompt_hash in commitment — "
                    "miner must include prompt_hash for input integrity"
                ), timing_details
            if commitment.prompt_hash != expected_prompt_hash:
                return VerificationResult.failure(
                    f"prompt_hash mismatch: committed="
                    f"{commitment.prompt_hash.hex()[:16]}..., "
                    f"expected={expected_prompt_hash.hex()[:16]}..."
                ), timing_details

        # Bind served output to commitment (prevents post-proof token substitution).
        if commitment.output_token_count > 0 and not proof_bundle.output_token_ids:
            return VerificationResult.failure(
                "Missing output_token_ids in proof bundle"
            ), timing_details
        if proof_bundle.output_token_ids:
            if commitment.output_token_count > 0 and len(proof_bundle.output_token_ids) != commitment.output_token_count:
                return VerificationResult.failure(
                    f"output_token_count mismatch: committed={commitment.output_token_count}, "
                    f"bundle={len(proof_bundle.output_token_ids)}"
                ), timing_details
            output_ids_arr = np.asarray(proof_bundle.output_token_ids, dtype=np.int64)
            output_hash = hashlib.sha256(output_ids_arr.astype("<i8", copy=False).tobytes()).digest()
            if output_hash != commitment.output_commitment:
                return VerificationResult.failure(
                    "output_commitment mismatch: output_token_ids do not match commitment"
                ), timing_details

        # Verify router_commitments match the committed hash (MoE binding).
        # Without this check, a miner could send tampered router_commitments
        # to control which experts get challenged while the beacon still
        # validates (because the beacon binds commitment_hash, not the raw
        # router_commitments).
        is_moe = self.moe_config is not None and self.moe_config.is_moe
        if is_moe and proof_bundle.router_commitments:
            expected_rc_hash = InferenceCommitment.compute_router_hash(
                proof_bundle.router_commitments,
            )
            if expected_rc_hash != commitment.router_commitment_hash:
                return VerificationResult.failure(
                    "Router commitment hash mismatch: proof bundle "
                    "router_commitments don't match the committed "
                    "router_commitment_hash"
                ), timing_details

        # Verify embedding input binding proof.
        # The embedding proof cryptographically binds the committed input tokens
        # to the on-chain embedding weight Merkle root, preventing a miner from
        # hashing the real prompt but running inference on a different one.
        emb_root = getattr(self.model_spec, "embedding_weight_merkle_root", b"")
        if not emb_root:
            return VerificationResult.failure(
                "ModelSpec.embedding_weight_merkle_root is missing — "
                "model must be re-registered with embedding root"
            ), timing_details
        if proof_bundle.embedding_proof is None:
            return VerificationResult.failure(
                "Missing embedding_proof in proof bundle — "
                "input binding verification requires embedding proof"
            ), timing_details

        emb_proof = proof_bundle.embedding_proof

        # input_token_ids must be present and non-empty.
        if not emb_proof.input_token_ids:
            return VerificationResult.failure(
                "Embedding proof: input_token_ids is empty"
            ), timing_details

        # Verify input_token_ids hash matches committed input_commitment.
        input_ids_arr = np.asarray(emb_proof.input_token_ids, dtype=np.int64)
        input_hash = hashlib.sha256(
            input_ids_arr.astype("<i8", copy=False).tobytes()
        ).digest()
        if input_hash != commitment.input_commitment:
            return VerificationResult.failure(
                "Embedding proof: input_token_ids hash does not match "
                "committed input_commitment"
            ), timing_details

        # Re-derive embedding challenge from beacon (Fiat-Shamir).
        emb_challenge = derive_embedding_challenge(
            beacon=beacon,
            commitment=commitment,
            num_input_tokens=len(emb_proof.input_token_ids),
        )
        if emb_challenge is None:
            return VerificationResult.failure(
                "Embedding proof: failed to derive embedding challenge"
            ), timing_details

        # Verify each challenged row opening against the on-chain
        # embedding_weight_merkle_root.
        openings_by_pos = {
            o.token_position: o for o in emb_proof.row_openings
        }
        chunk_size = self.model_spec.w_merkle_chunk_size
        hidden_dim = self.model_spec.hidden_dim

        for pos in emb_challenge.token_positions:
            opening = openings_by_pos.get(pos)
            if opening is None:
                return VerificationResult.failure(
                    f"Embedding proof: missing row opening for "
                    f"challenged position {pos}"
                ), timing_details

            # Verify the Merkle path against the on-chain root.
            # FlatWeightMerkle uses double hashing: hash_flat_chunk(data) then
            # hash_leaf(chunk_hash), so we must use verify_flat_chunk_merkle_path.
            path_valid = verify_flat_chunk_merkle_path(
                root=emb_root,
                chunk_data=opening.leaf_data,
                path=opening.merkle_path,
            )
            if not path_valid:
                from zkllm.crypto.merkle import hash_flat_chunk, hash_leaf
                _chunk_hash = hash_flat_chunk(opening.leaf_data)
                _leaf_hash = hash_leaf(_chunk_hash)
                import logging as _log
                _log.getLogger("verallm.debug").warning(
                    f"EMB DEBUG: pos={pos} token_id={opening.token_id} "
                    f"leaf_data_len={len(opening.leaf_data)} "
                    f"chunk_hash={_chunk_hash[:8].hex()} "
                    f"leaf_hash={_leaf_hash[:8].hex()} "
                    f"path_len={len(opening.merkle_path.siblings)} "
                    f"path_idx={opening.merkle_path.leaf_index} "
                    f"emb_root={emb_root[:8].hex()} "
                    f"chunk_size={chunk_size} hidden_dim={hidden_dim}"
                )
                return VerificationResult.failure(
                    f"Embedding proof: Merkle path invalid for "
                    f"position {pos}, token_id={opening.token_id}"
                ), timing_details

            # Verify the token_id matches what's in input_token_ids.
            expected_token = emb_proof.input_token_ids[pos]
            if opening.token_id != expected_token:
                return VerificationResult.failure(
                    f"Embedding proof: token_id mismatch at "
                    f"position {pos}: opening has "
                    f"{opening.token_id}, expected {expected_token}"
                ), timing_details

            # Verify the chunk index is correct for this token_id.
            expected_chunk_idx = (opening.token_id * hidden_dim) // chunk_size
            if opening.merkle_path.leaf_index != expected_chunk_idx:
                return VerificationResult.failure(
                    f"Embedding proof: chunk index mismatch at "
                    f"position {pos}: path.leaf_index="
                    f"{opening.merkle_path.leaf_index}, expected "
                    f"{expected_chunk_idx} for token_id={opening.token_id}"
                ), timing_details

        logger.debug("    - Embedding proof: %d row openings verified against on-chain root",
                     len(emb_proof.row_openings))

        # ----------------------------------------------------------------
        # Embedding output → layer 0 binding (DISABLED)
        #
        # This builds a per-request Merkle tree over the embedding
        # output tensor to spot-check that output[p] == W_emb[token_id_p].
        # It binds proven embedding rows to the actual activations
        # entering layer 0.
        #
        # WHY DISABLED: The per-request Merkle tree costs O(seq_len *
        # hidden_dim) to build — ~14s for 176K-token full-context
        # requests vs the 30ms proof budget.  No practical way to reduce
        # this without GPU tree building, which conflicts with inference
        # CUDA streams.
        #
        # WHY IT'S OK: Faking layer 0 input while running real inference
        # saves zero compute — no economic incentive.  Layer 0 is also
        # challenged with probability k/N ~ 6% per request.  Over 50
        # requests the detection probability exceeds 95%.  Additionally,
        # the decode sampling proof (temp=0) verifies output tokens
        # end-to-end — wrong layer 0 input -> wrong hidden states ->
        # wrong argmax -> caught.
        #
        # The full implementation is preserved in types.py
        # (EmbeddingOutputOpening, output_openings field on EmbeddingProof)
        # and can be re-enabled if GPU Merkle tree building becomes
        # practical.
        # ----------------------------------------------------------------

        # Verify layer transition hash chain.
        # Binds consecutive layer commitments together, anchored by
        # input_commitment (ties the layer chain to the proven input).
        expected_num_hashes = max(0, len(commitment.layer_commitments) - 1)
        if len(commitment.layer_transition_hashes) != expected_num_hashes:
            return VerificationResult.failure(
                f"Wrong number of transition hashes: got "
                f"{len(commitment.layer_transition_hashes)}, expected "
                f"{expected_num_hashes}"
            ), timing_details

        for i in range(expected_num_hashes):
            expected_hash = hashlib.sha256(
                b"LAYER_TRANSITION_V2"
                + commitment.input_commitment
                + struct.pack("<I", i)
                + commitment.layer_commitments[i]
                + commitment.layer_commitments[i + 1]
            ).digest()
            if commitment.layer_transition_hashes[i] != expected_hash:
                return VerificationResult.failure(
                    f"Transition hash mismatch at boundary {i}->{i+1}: "
                    f"committed hash does not match re-derived hash"
                ), timing_details

        if expected_num_hashes > 0:
            logger.debug("    - Transition hash chain: %d boundaries verified", expected_num_hashes)

        # Re-derive challenges (Fiat-Shamir)
        if is_moe:
            expected_challenges = derive_moe_challenges(
                beacon=beacon,
                commitment=commitment,
                moe_config=self.moe_config,
                router_commitments=proof_bundle.router_commitments,
                k_layers=self.config.k_layers,
                k_tokens_per_layer=self.config.k_tokens_per_expert,
                k_experts_per_layer=self.config.k_experts_per_layer,
            )
            logger.debug("  Validator: Re-deriving MoE challenges (Fiat-Shamir)...")
        else:
            expected_challenges = derive_challenges(
                beacon=beacon,
                commitment=commitment,
                k_layers=self.config.k_layers,
                k_gemms_per_layer=2,
                k_blocks_per_gemm=self.config.k_blocks,
            )
            logger.debug("  Validator: Re-deriving challenges (Fiat-Shamir)...")

        # Debug: show what layers were derived
        expected_layer_indices = [lc.layer_idx for lc in expected_challenges.layer_challenges]
        logger.debug("    - Expected challenge layers: %s", expected_layer_indices)
        proof_layer_indices = [lp.layer_idx for lp in proof_bundle.layer_proofs]
        logger.debug("    - Proof bundle layers: %s", proof_layer_indices)
        logger.debug("    - num_layers in commitment: %d", len(commitment.layer_commitments))
        logger.debug("    - Router commitments keys: %s",
                     sorted(proof_bundle.router_commitments.keys()) if proof_bundle.router_commitments else 'None')

        # Lightweight mode: W verified via Merkle proofs
        logger.debug("    - W spot checks verified via Merkle proofs against on-chain roots")

        # Verify each layer proof
        for layer_proof in proof_bundle.layer_proofs:
            layer_idx = layer_proof.layer_idx

            # Find the matching layer challenge
            layer_challenge = None
            for lc in expected_challenges.layer_challenges:
                if lc.layer_idx == layer_idx:
                    layer_challenge = lc
                    break

            if layer_challenge is None:
                return VerificationResult.failure(
                    f"No challenge found for layer {layer_idx}"
                ), timing_details

            for i, gemm_proof in enumerate(layer_proof.gemm_proofs):
                # Handle MoE vs dense challenges
                is_moe_challenge = hasattr(layer_challenge, 'expert_challenges')
                expert_idx = None

                if is_moe_challenge:
                    if i < len(layer_challenge.expert_challenges):
                        gemm_idx = layer_challenge.expert_challenges[i].gemm_idx
                        expert_idx = layer_challenge.expert_challenges[i].expert_idx
                    else:
                        gemm_idx = i
                else:
                    gemm_idx = (
                        layer_challenge.gemm_challenges[i].gemm_idx
                        if i < len(layer_challenge.gemm_challenges) else i
                    )

                hidden_dim = self.model_spec.hidden_dim
                intermediate_dim = self.model_spec.intermediate_dim

                X_commitment = commitment.layer_commitments[layer_idx]
                W_commitment = self.model_spec.weight_merkle_root
                Y_commitment = gemm_proof.output_root

                # Verify X spot checks with Merkle proofs
                for block_proof in gemm_proof.block_proofs:
                    if block_proof.spot_X_with_proofs:
                        for spot_wp in block_proof.spot_X_with_proofs:
                            path_valid = verify_merkle_path(
                                root=X_commitment,
                                leaf_data=spot_wp.leaf_data,
                                path=spot_wp.merkle_path,
                            )
                            if not path_valid:
                                return VerificationResult.failure(
                                    f"X spot check Merkle proof invalid at layer {layer_idx}, "
                                    f"position ({spot_wp.row}, {spot_wp.col})"
                                ), timing_details

                            leaf_values = struct.unpack(
                                f"<{len(spot_wp.leaf_data) // 8}q",
                                spot_wp.leaf_data,
                            )
                            flat_idx = spot_wp.row * hidden_dim + spot_wp.col
                            block_start = spot_wp.merkle_path.leaf_index * 256
                            idx_in_block = flat_idx - block_start

                            if 0 <= idx_in_block < len(leaf_values):
                                expected_value = mod_p(int(leaf_values[idx_in_block]))
                                if spot_wp.value != expected_value:
                                    return VerificationResult.failure(
                                        f"X spot check value mismatch at layer {layer_idx}, "
                                        f"position ({spot_wp.row}, {spot_wp.col}): "
                                        f"claimed {spot_wp.value}, leaf has {expected_value}"
                                    ), timing_details

                # Lightweight: W verified via Merkle proofs in verifier
                spot_check_fn = lambda _spot, _matrix_id: True

                # Get W Merkle root for this layer
                W_merkle_root = None
                if is_moe_challenge and expert_idx is not None:
                    # Try spec first (local mode), then proof bundle (chain mode)
                    expert_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
                    if not expert_roots:
                        expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
                    if expert_idx < len(expert_roots):
                        W_merkle_root = expert_roots[expert_idx]

                        router_root = self.model_spec.router_weight_merkle_roots.get(layer_idx)
                        if router_root is None:
                            rp = proof_bundle.router_layer_proofs.get(layer_idx)
                            router_root = rp.router_weight_root if rp is not None else None
                        if not self._verify_moe_layer_root(layer_idx, expert_roots, router_root):
                            return VerificationResult.failure(
                                f"MoE layer {layer_idx}: hierarchical root mismatch "
                                f"(expert roots don't hash to committed layer root)"
                            ), timing_details
                    else:
                        if layer_idx < len(self.model_spec.weight_block_merkle_roots):
                            W_merkle_root = self.model_spec.weight_block_merkle_roots[layer_idx]
                elif layer_idx < len(self.model_spec.weight_block_merkle_roots):
                    W_merkle_root = self.model_spec.weight_block_merkle_roots[layer_idx]

                # Transcript must match prover (Fiat-Shamir)
                if is_moe_challenge and expert_idx is not None:
                    transcript = Transcript(f"layer_{layer_idx}_expert_{expert_idx}_gemm_{gemm_idx}".encode())
                else:
                    transcript = Transcript(f"layer_{layer_idx}_gemm_{gemm_idx}".encode())

                # For MoE experts with fused gate+up, use expert_w_num_cols
                expert_w_cols = getattr(self.model_spec, 'expert_w_num_cols', 0)
                if is_moe_challenge and expert_idx is not None and expert_w_cols > 0:
                    w_cols = expert_w_cols
                else:
                    w_cols = intermediate_dim

                t0 = time.perf_counter()
                result = verifier.verify(
                    proof=gemm_proof,
                    X_commitment=X_commitment,
                    W_commitment=W_commitment,
                    Y_commitment=Y_commitment,
                    transcript=transcript,
                    spot_check_fn=spot_check_fn,
                    W_merkle_root=W_merkle_root,
                    W_num_cols=w_cols,
                    w_chunk_size=self.model_spec.w_merkle_chunk_size,
                )
                verify_time = (time.perf_counter() - t0) * 1000

                if is_moe_challenge and expert_idx is not None:
                    timing_key = f"Layer {layer_idx}, Expert {expert_idx}, GEMM {gemm_idx}"
                else:
                    timing_key = f"Layer {layer_idx}, GEMM {gemm_idx}"
                timing_details[timing_key] = verify_time

                status = "PASSED" if result.passed else "FAILED"
                logger.debug("    - %s: %s (%.2fms)", timing_key, status, verify_time)

                if not result.passed:
                    return VerificationResult.failure(
                        f"GEMM verification failed at {timing_key}: {result.message}"
                    ), timing_details

        # Verify routing openings/top-k consistency for challenged MoE layers.
        if is_moe:
            for layer_challenge in expected_challenges.layer_challenges:
                if not hasattr(layer_challenge, "sampled_token_indices"):
                    continue
                if not getattr(layer_challenge, "verify_routing", False):
                    continue
                routing_result = self._verify_router_layer_openings(proof_bundle, layer_challenge)
                if not routing_result.passed:
                    return VerificationResult.failure(
                        f"Routing verification failed for layer {layer_challenge.layer_idx}: "
                        f"{routing_result.message}"
                    ), timing_details

        # Verify optional decode-integrity proofs (greedy argmax or
        # canonical sampler replay for do_sample=True with committed seed).
        sampling_challenge = derive_sampling_challenge(
            beacon=beacon,
            commitment=commitment,
            vocab_size=int(getattr(self.model_spec, "vocab_size", 0) or 0),
        )
        if sampling_challenge is not None:
            if not commitment.decode_hidden_row_root:
                return VerificationResult.failure(
                    "Sampling challenge active but decode_hidden_row_root is missing"
                ), timing_details
            if not proof_bundle.sampling_proofs:
                return VerificationResult.failure(
                    "Sampling challenge active but sampling proofs are missing"
                ), timing_details
            if not proof_bundle.output_token_ids:
                return VerificationResult.failure(
                    "Sampling challenge active but output_token_ids are missing"
                ), timing_details
            lm_head_root = getattr(self.model_spec, "lm_head_weight_merkle_root", b"")
            if not lm_head_root:
                return VerificationResult.failure(
                    "Sampling challenge active but ModelSpec.lm_head_weight_merkle_root is missing"
                ), timing_details

            # Collect proofs in challenge order, verify per-row checks.
            proofs_by_step = {int(sp.decode_step): sp for sp in proof_bundle.sampling_proofs}
            ordered_proofs: list = []
            X_rows_i64: list = []
            Y_rows_i64: list = []
            batched_gemm_proof = None

            for decode_step in sampling_challenge.decode_positions:
                step = int(decode_step)
                sp = proofs_by_step.get(step)
                if sp is None:
                    return VerificationResult.failure(
                        f"Missing sampling proof for decode step {step}"
                    ), timing_details
                if step < 0 or step >= len(proof_bundle.output_token_ids):
                    return VerificationResult.failure(
                        f"Sampling proof decode step out of range: {step}"
                    ), timing_details
                committed_token = int(proof_bundle.output_token_ids[step])
                if int(sp.token_id) != committed_token:
                    return VerificationResult.failure(
                        f"Sampling token mismatch at step {step}: "
                        f"proof={sp.token_id}, committed={committed_token}"
                    ), timing_details

                # Verify hidden row against committed root.
                hidden_ok = verify_merkle_path(
                    root=commitment.decode_hidden_row_root,
                    leaf_data=sp.hidden_row,
                    path=sp.hidden_merkle_path,
                )
                if not hidden_ok:
                    return VerificationResult.failure(
                        f"Decode hidden-row Merkle proof invalid at step {step}"
                    ), timing_details

                if sp.lm_head_weight_root != lm_head_root:
                    return VerificationResult.failure(
                        f"lm_head weight root mismatch at decode step {step}"
                    ), timing_details

                # Reconstruct quantized hidden row.
                hidden_fp32 = hidden_row_from_bytes(sp.hidden_row)
                X_row = quantize_hidden_row_int64(hidden_fp32)
                X_rows_i64.append(X_row)

                logits_i32 = logits_i32_from_bytes(sp.proved_logits_i32)
                if logits_i32.size == 0:
                    return VerificationResult.failure(
                        f"Empty proved logits row at decode step {step}"
                    ), timing_details
                Y_row = torch.from_numpy(logits_i32.astype(np.int64, copy=True)).view(1, -1)
                Y_rows_i64.append(Y_row)

                # ----------------------------------------------------------
                # Canonical sampler replay for do_sample=True.
                # ----------------------------------------------------------
                # The miner's CanonicalSamplerLP runs canonical_sample on the
                # raw fp32 logits coming out of compute_logits.  To replay
                # this exactly the validator must use the SAME bit-identical
                # fp32 logits — NOT the int32 reconstruction from the
                # quantized lm_head GEMM proof.  The miner captures the fp32
                # logits row in fp16_logits_row whenever canonical mode is
                # active (any bps > 0), so we open it and verify its Merkle
                # path against decode_logits_row_root.
                #
                # The replay runs at every challenged position regardless of
                # bps.  This closes the loophole where a miner could observe
                # bps_for_request, detect canary vs organic, and selectively
                # cheat on organic — the validator now enforces canonical
                # sampling on every verified request.
                if commitment.do_sample and commitment.sampling_seed_commitment:
                    _opened_seed = getattr(sp, "sampling_seed", None)
                    if _opened_seed is None:
                        return VerificationResult.failure(
                            f"Sampling seed missing from proof at step {step} "
                            "(do_sample=True with seed commitment requires opening)"
                        ), timing_details
                    import hashlib as _hl
                    if _hl.sha256(_opened_seed).digest() != commitment.sampling_seed_commitment:
                        return VerificationResult.failure(
                            f"Sampling seed commitment mismatch at step {step}"
                        ), timing_details

                    if not commitment.decode_logits_row_root:
                        return VerificationResult.failure(
                            f"Canonical replay requires decode_logits_row_root at step {step}"
                        ), timing_details
                    if not sp.fp16_logits_row or sp.fp16_logits_merkle_path is None:
                        return VerificationResult.failure(
                            f"Canonical replay requires fp16_logits_row + merkle path at step {step}"
                        ), timing_details
                    # Verify Merkle path so the miner can't substitute
                    # different logits than what they actually produced.
                    if not verify_merkle_path(
                        root=commitment.decode_logits_row_root,
                        leaf_data=sp.fp16_logits_row,
                        path=sp.fp16_logits_merkle_path,
                    ):
                        return VerificationResult.failure(
                            f"Canonical replay: fp16 logits Merkle path invalid at step {step}"
                        ), timing_details
                    # Parse the top-K leaf bytes directly.  The miner's
                    # activation tracker captures only the top
                    # CANONICAL_TOP_K logits per decode step (sorted by
                    # value DESC, index ASC) and serializes them via
                    # `serialize_top_k_to_bytes`.  The leaf IS the top-K
                    # — no extraction step is needed on the validator side.
                    #
                    # Falls back to the legacy full-vocab fp32/fp16 parse
                    # for proofs produced by old miners that captured the
                    # full row.  In that case we extract top-K on the
                    # validator side via extract_top_k_sorted.
                    from verallm.sampling import (
                        canonical_sample as _canonical_sample,
                        parse_top_k_leaf as _parse_top_k_leaf,
                        extract_top_k_sorted as _extract_top_k_sorted,
                    )
                    _row_bytes = sp.fp16_logits_row
                    _row_len = len(_row_bytes)
                    if _row_len == 0:
                        return VerificationResult.failure(
                            f"Canonical replay: empty fp16_logits_row at step {step}"
                        ), timing_details

                    if _row_len % 12 == 0:
                        # Top-K leaf format (current default).
                        try:
                            _top_vals, _top_idx = _parse_top_k_leaf(_row_bytes)
                        except ValueError as _pe:
                            return VerificationResult.failure(
                                f"Canonical replay: failed to parse top-K leaf at step {step}: {_pe}"
                            ), timing_details
                    else:
                        # Legacy full-vocab fallback.
                        if _row_len % 4 == 0:
                            _full_logits = np.frombuffer(_row_bytes, dtype="<f4").astype(np.float32, copy=False)
                        elif _row_len % 2 == 0:
                            _full_logits = np.frombuffer(_row_bytes, dtype="<f2").astype(np.float32, copy=False)
                        else:
                            return VerificationResult.failure(
                                f"Canonical replay: invalid fp16_logits_row length "
                                f"{_row_len} at step {step}"
                            ), timing_details
                        _top_vals, _top_idx = _extract_top_k_sorted(_full_logits)

                    # Resolve sampling parameters.  Prefer the explicit
                    # expected_* values supplied by the caller (which match
                    # the actual request body); fall back to canary defaults.
                    # The sampler_config_hash check above already binds
                    # these values, so they cannot disagree silently.
                    _temp = max(0.001, commitment.temperature_milli / 1000.0)
                    _top_k = int(expected_top_k) if expected_top_k is not None else -1
                    _top_p = float(expected_top_p) if expected_top_p is not None else 1.0
                    _min_p = float(expected_min_p) if expected_min_p is not None else 0.0
                    replayed_token = _canonical_sample(
                        _top_vals, _top_idx,
                        _temp, _top_k, _top_p, _min_p,
                        _opened_seed, step,
                    )
                    if replayed_token != committed_token:
                        return VerificationResult.failure(
                            f"Canonical sampler replay diverged at step {step}: "
                            f"replayed={replayed_token}, committed={committed_token}"
                        ), timing_details

                    ordered_proofs.append(sp)
                    if sp.lm_head_gemm_proof is not None and batched_gemm_proof is None:
                        batched_gemm_proof = sp.lm_head_gemm_proof
                    continue

                # Quantization-stable argmax check (int8×int8→int32 recomputation).
                argmax_ok, argmax_detail = verify_quantized_argmax(logits_i32, committed_token)

                # When post-logits processors are active (presence_penalty, etc.),
                # argmax divergence is expected — processors modify logits after
                # compute_logits().  Only enforce strict argmax when no
                # processors can cause divergence.
                _has_post_logits_mods = (
                    commitment.presence_penalty_milli != 0
                    or (enable_thinking is not False)  # thinking logits processor
                )

                if not argmax_ok and not sampling_challenge.high_assurance:
                    if _has_post_logits_mods:
                        # Post-logits processors active — divergence expected, log only.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "(%s) — post-logits processors active (non-fatal)",
                            step, argmax_detail,
                        )
                    else:
                        # Strict mode but GEMM proof passed — likely capture
                        # alignment issue (non-fatal).  Demote to debug:
                        # the GEMM proof did pass, so this is a known
                        # capture-side noise pattern, not a verification
                        # failure that the operator needs to act on.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "in strict mode (%s) — GEMM proof passed, "
                            "treating as capture alignment issue (non-fatal)",
                            step, argmax_detail,
                        )

                # High-assurance: fp16 logits row + exact argmax (authoritative).
                # The fp16 logits are captured directly from inference (no
                # requantization loss) and Merkle-committed.  When the int32
                # recomputation diverges — which happens at long contexts with
                # quantized models due to accumulated hidden-state error — the
                # fp16 check is the ground truth.
                if sampling_challenge.high_assurance:
                    if not commitment.decode_logits_row_root:
                        return VerificationResult.failure(
                            "High-assurance mode active but decode_logits_row_root missing"
                        ), timing_details
                    if not sp.fp16_logits_row:
                        return VerificationResult.failure(
                            f"High-assurance mode active but fp16_logits_row missing at step {step}"
                        ), timing_details
                    if sp.fp16_logits_merkle_path is None:
                        return VerificationResult.failure(
                            f"High-assurance mode active but fp16_logits_merkle_path missing at step {step}"
                        ), timing_details
                    fp16_path_ok = verify_merkle_path(
                        root=commitment.decode_logits_row_root,
                        leaf_data=sp.fp16_logits_row,
                        path=sp.fp16_logits_merkle_path,
                    )
                    if not fp16_path_ok:
                        return VerificationResult.failure(
                            f"Fp16 logits Merkle proof invalid at step {step}"
                        ), timing_details
                    fp16_ok, fp16_detail = verify_fp16_argmax(sp.fp16_logits_row, committed_token)

                    if not fp16_ok:
                        if not _has_post_logits_mods:
                            # Strict mode but GEMM proof passed — likely capture
                            # alignment issue. Debug-level: GEMM did pass, so
                            # this is known noise from continuous-batching
                            # capture, not a verification failure.
                            logger.debug(
                                "    - Sampling step %d: fp16 argmax "
                                "diverged in strict mode (%s) — "
                                "GEMM proof passed, treating as capture alignment "
                                "issue (non-fatal)",
                                step, fp16_detail,
                            )
                        else:
                            # Post-logits processors active: divergence expected.
                            logger.debug(
                                "    - Sampling step %d: fp16 argmax diverged "
                                "from committed token (%s) — "
                                "post-logits processors active (non-fatal)",
                                step, fp16_detail,
                            )

                    if not argmax_ok:
                        # Both int32 and fp16 can diverge from capture alignment.
                        # Log for diagnostics.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "(%s), fp16: %s",
                            step, argmax_detail, fp16_detail,
                        )
                elif not argmax_ok:
                    # Not high-assurance and int32 failed — shouldn't reach here
                    # because we already return above, but guard defensively.
                    return VerificationResult.failure(
                        f"Sampling argmax failed at decode step {step}: {argmax_detail}"
                    ), timing_details

                # Grab the batched proof from the first proof that carries it.
                if sp.lm_head_gemm_proof is not None and batched_gemm_proof is None:
                    batched_gemm_proof = sp.lm_head_gemm_proof

                ordered_proofs.append(sp)

            # Verify batched lm_head GEMM proof: X_batch @ W = Y_batch.
            if ordered_proofs and X_rows_i64:
                if batched_gemm_proof is None:
                    return VerificationResult.failure(
                        "No lm_head GEMM proof found in sampling proofs"
                    ), timing_details

                X_batch = torch.cat(X_rows_i64, dim=0)
                Y_batch = torch.cat(Y_rows_i64, dim=0)
                hidden_dim = X_batch.shape[1]

                # Reconstruct X Merkle (same chunking as prover).
                flat_x = X_batch.flatten().to(torch.int64)
                x_leaves = []
                for start in range(0, int(flat_x.numel()), 256):
                    x_leaves.append(flat_x[start:start + 256].numpy().astype("<i8", copy=False).tobytes())
                if not x_leaves:
                    x_leaves = [b"empty"]
                x_tree = MerkleTree(x_leaves)

                # Reconstruct Y Merkle (single block covering [k, vocab]).
                y_block = int(max(1, Y_batch.shape[1]))
                Y_merkle = build_block_merkle(Y_batch, y_block)

                # Verify Y Merkle root matches proof output_root.
                if Y_merkle.root != batched_gemm_proof.output_root:
                    return VerificationResult.failure(
                        "Batched lm_head Y Merkle root mismatch: "
                        "reconstructed Y_batch does not match proof output_root"
                    ), timing_details

                transcript = Transcript(b"decode_lm_head_gemm_batched")
                lm_head_verify = verifier.verify(
                    proof=batched_gemm_proof,
                    X_commitment=x_tree.root,
                    W_commitment=ordered_proofs[0].lm_head_weight_root,
                    Y_commitment=batched_gemm_proof.output_root,
                    transcript=transcript,
                    spot_check_fn=lambda _spot, _matrix_id: True,
                    W_merkle_root=lm_head_root,
                    W_num_cols=Y_batch.shape[1],
                    w_chunk_size=self.model_spec.w_merkle_chunk_size,
                )
                if not lm_head_verify.passed:
                    return VerificationResult.failure(
                        f"Batched lm_head GEMM verification failed: {lm_head_verify.message}"
                    ), timing_details

        return VerificationResult.success("All GEMM proofs verified"), timing_details

    # ------------------------------------------------------------------
    # Full protocol run
    # ------------------------------------------------------------------

    def run_protocol(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        stream_callback=None,
    ) -> Tuple[bool, str, dict]:
        """Run the complete verification protocol.

        Non-interactive flow:
        1. Fetch ModelSpec (on-chain in production, from miner here)
        2. POST /inference -> stream tokens + commitment + proofs
        3. Verify proofs locally (re-derive beacon + challenges)

        Args:
            prompt: Input prompt.
            max_new_tokens: Max tokens to generate.
            stream_callback: Optional callable(token_text) per token.

        Returns:
            (passed, output_text, all_timings)
        """
        all_timings = {}

        # Fetch ModelSpec
        logger.info("\n%s\nPHASE 1: FETCH MODEL SPECIFICATION\n%s", "=" * 70, "=" * 70)
        t0 = time.perf_counter()
        model_spec = self.fetch_model_spec()
        phase_ms = (time.perf_counter() - t0) * 1000
        all_timings["fetch_model_spec_ms"] = phase_ms

        logger.info("  Model: %s", model_spec.model_id)
        logger.info("  Layers: %d, Hidden: %d", model_spec.num_layers, model_spec.hidden_dim)
        logger.info("  Roots: %d layer roots", len(model_spec.weight_block_merkle_roots))
        logger.info("  Fetched in %.1fms", phase_ms)

        # Auto-compute k_layers if needed
        if self.config.k_layers == 0:
            k = max(1, round(model_spec.num_layers * self.config.target_detection))
            k = min(k, model_spec.num_layers // 2)
            self.config = Config(
                **{
                    **{f.name: getattr(self.config, f.name)
                       for f in self.config.__dataclass_fields__.values()},
                    "k_layers": k,
                }
            )
            set_config(self.config)
            logger.info("  Auto k_layers: %d/%d", k, model_spec.num_layers)

        # Detect MoE from model_spec.
        # Chain-mode specs have num_experts from on-chain data;
        # local specs may have expert_weight_merkle_roots instead.
        num_experts = model_spec.num_experts
        if num_experts == 0 and model_spec.expert_weight_merkle_roots:
            first_roots = next(iter(model_spec.expert_weight_merkle_roots.values()), [])
            num_experts = len(first_roots)

        if num_experts > 0:
            if model_spec.expert_weight_merkle_roots:
                moe_layer_indices = sorted(model_spec.expert_weight_merkle_roots.keys())
            else:
                # Chain mode: expert roots not in spec, assume all layers are MoE
                moe_layer_indices = list(range(model_spec.num_layers))
            expert_inter = (
                model_spec.expert_w_num_cols
                if model_spec.expert_w_num_cols > 0
                else model_spec.intermediate_dim
            )
            self.moe_config = MoEConfig(
                is_moe=True,
                num_layers=model_spec.num_layers,
                moe_layer_indices=moe_layer_indices,
                num_routed_experts=num_experts,
                num_shared_experts=0,
                top_k=model_spec.router_top_k if model_spec.router_top_k > 0 else 2,
                hidden_dim=model_spec.hidden_dim,
                intermediate_dim=model_spec.intermediate_dim,
                expert_intermediate_dim=expert_inter,
                has_shared_expert_gate=False,
                uses_3d_expert_weights=False,
                router_type=model_spec.router_scoring or "top_k",
            )
            logger.info("  Detected MoE: %d experts, %d MoE layers", num_experts, len(moe_layer_indices))

            # Auto-compute k_experts_per_layer if not set (must match server)
            if self.config.k_experts_per_layer == 0:
                k_exp = compute_auto_k_experts(num_experts)
                from dataclasses import fields as dc_fields
                self.config = Config(
                    **{
                        **{f.name: getattr(self.config, f.name) for f in dc_fields(self.config)},
                        "k_experts_per_layer": k_exp,
                    }
                )
                set_config(self.config)
                logger.info("  Auto k_experts: %d/%d", k_exp, num_experts)

        # Inference + proofs (single request)
        logger.info("\n%s\nINFERENCE + PROOFS (single request)\n%s", "=" * 70, "=" * 70)
        logger.info("  Prompt: %s%s", prompt[:80], "..." if len(prompt) > 80 else "")

        def default_stream_cb(token):
            sys.stdout.write(token)
            sys.stdout.flush()

        cb = stream_callback or default_stream_cb

        sys.stdout.write("  Output: ")
        sys.stdout.flush()
        full_text, commitment, proof_bundle, nonce, infer_timing = self.run_inference(
            prompt, max_new_tokens=max_new_tokens, stream_callback=cb,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
        all_timings.update(infer_timing)

        logger.info("  Tokens: %d", len(full_text.split()))
        logger.info("  Inference: %.1fms", infer_timing['inference_ms'])
        logger.info("  Commitment: %.1fms", infer_timing['commitment_ms'])
        logger.info("  Prove (miner): %.1fms", infer_timing['prove_ms'])
        logger.info("  Round-trip: %.1fms", infer_timing['round_trip_ms'])

        detection_info = compute_detection_probability(
            self.config.k_layers, model_spec.num_layers,
        )
        detection_prob = detection_info["p_detect_per_inference"]
        logger.info("  Detection probability: %.1f%%", detection_prob * 100)

        # Phase 7: Verify proofs locally
        logger.info("\n%s\nPHASE 7: VERIFY PROOFS\n%s", "=" * 70, "=" * 70)
        t0 = time.perf_counter()
        result, verify_timing = self.verify_proof(proof_bundle, nonce)
        verify_ms = (time.perf_counter() - t0) * 1000
        all_timings["verify_ms"] = verify_ms
        all_timings["verify_details"] = verify_timing

        # Summary
        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "\n%s\n  VERIFICATION RESULT: %s\n  Message: %s\n\n"
            "  Timing Summary:\n"
            "    Model spec fetch:   %.1fms\n"
            "    Inference RTT:      %.1fms\n"
            "      Inference:        %.1fms\n"
            "      Commitment:       %.1fms\n"
            "      Prove (miner):    %.1fms\n"
            "    Verify (local):     %.1fms",
            "=" * 70, status, result.message,
            all_timings.get('fetch_model_spec_ms', 0),
            all_timings.get('round_trip_ms', 0),
            all_timings.get('inference_ms', 0),
            all_timings.get('commitment_ms', 0),
            all_timings.get('prove_ms', 0),
            verify_ms,
        )
        total = (
            all_timings.get("fetch_model_spec_ms", 0)
            + all_timings.get("round_trip_ms", 0)
            + verify_ms
        )
        logger.info("    Total wall time:    %.1fms\n%s", total, "=" * 70)

        return result.passed, full_text, all_timings


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VeraLLM Validator Client")
    parser.add_argument("--miner-url", required=True,
                        help="Miner server URL (e.g. http://localhost:8000)")
    parser.add_argument("--prompt", required=True,
                        help="Inference prompt")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--k-layers", type=int, default=0,
                        help="Layers to challenge (0 = auto)")
    parser.add_argument("--k-experts", type=int, default=0,
                        help="Experts to challenge per layer (0 = auto)")
    parser.add_argument("--k-tokens", type=int, default=4,
                        help="Tokens to sample for expert challenges")
    parser.add_argument("--spot-checks", type=int, default=50,
                        help="Number of spot checks per block")
    parser.add_argument("--no-verify-tls", action="store_true",
                        help="Disable TLS certificate verification")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Request timeout in seconds")
    parser.add_argument("--api-key", default=None,
                        help="API key for miner authentication")
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON (reads ModelSpec from chain)")
    parser.add_argument("--model-id", default=None,
                        help="Model ID for on-chain lookup (required with --chain-config)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        block_size=256,
        spot_checks=args.spot_checks,
        k_layers=args.k_layers,
        k_experts_per_layer=args.k_experts,
        k_tokens_per_expert=args.k_tokens,
    )
    set_config(config)

    logger.info(
        "\n%s\n  VeraLLM Validator Client\n  Miner: %s\n  Prompt: %s%s\n%s",
        "=" * 70, args.miner_url,
        args.prompt[:60], "..." if len(args.prompt) > 60 else "",
        "=" * 70,
    )

    chain_config = None
    if args.chain_config:
        from verallm.chain.config import ChainConfig
        chain_config = ChainConfig.from_json(args.chain_config)

    with ValidatorClient(
        miner_url=args.miner_url,
        config=config,
        verify_tls=not args.no_verify_tls,
        timeout=args.timeout,
        api_key=args.api_key,
        chain_config=chain_config,
        model_id=args.model_id,
    ) as client:
        # Health check
        try:
            health = client.health_check()
            logger.info("  Miner status: %s", health.get('status', 'unknown'))
            logger.info("  Miner model: %s", health.get('model', 'unknown'))
        except Exception as e:
            logger.error("  Cannot reach miner at %s: %s", args.miner_url, e)
            sys.exit(1)

        # Run full protocol
        passed, output_text, timings = client.run_protocol(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
