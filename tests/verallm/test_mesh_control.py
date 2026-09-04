import json
import hashlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
from bittensor_wallet import Keypair
from eth_account import Account

from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    build_signing_message,
)
from verallm.mesh import (
    CapabilityAd,
    GgmlMulMatTrace,
    GgmlOpManifestEntry,
    LlamaGraphOpReceipt,
    MeshMember,
    MeshSpec,
    POSTCOMMIT_AUDIT_PATH,
    StageRange,
    StageReceipt,
    VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
    VERATHOS_GGUF_DECODE_AUDIT_MODE,
    VERATHOS_GGML_TRACE_PROOF_MODE,
    VERATHOS_GGML_GEMM_PROOF_MODE,
    find_op_manifest_entries_for_window,
    ggml_decode_audit_receipt_root,
    ggml_op_manifest_root,
    ggml_op_manifest_summary_for_window,
    llama_graph_receipt_root,
    load_mesh_spec,
    make_ggml_proof_server,
    mesh_deferred_audit_bundle_hash,
    mesh_deferred_audit_commitment_hash,
    mesh_deferred_audit_sample_commitment_hash,
    mesh_validator_challenge_nonce_commitment,
    mesh_op_manifest_aggregate_root,
    op_manifest_membership_payload,
    mesh_receipt_hash,
    mesh_proof_gate_hash,
    mesh_response_commitment_hash,
    derive_mesh_postcommit_proof_beacon,
    prove_ggml_mul_mat_trace,
    rpc_plan_from_mesh,
    save_json,
    stage_receipt_root,
    suggest_ggml_trace_max_elems_from_tensor_records,
    verify_ggml_gemm_proof_payload,
    verify_ggml_gemm_proof_payloads,
    verify_ggml_decode_audit_payloads,
    verify_deferred_mesh_audit_bundle,
    verify_mesh_postcommit_artifact,
    verify_op_manifest_membership,
)
from verallm.mesh.ggml_proof import (
    ggml_proof_payload_commitment_hash,
    ggml_proof_selection_payload,
    is_decode_candidate_manifest_entry,
    select_decode_manifest_entries,
    select_manifest_challenge_indexes,
)
from verallm.mesh.cli import main as mesh_cli_main
from verallm.mesh.state import (
    admit_mesh_worker,
    assign_mesh_members,
    create_mesh_state,
    join_mesh,
    load_mesh_state,
    refresh_worker_mesh_state,
    save_mesh_state,
    state_admitted_compute_stage_count,
    state_capabilities,
    state_mesh_spec,
    update_worker_mesh_state,
)
from verallm.mesh.proof import MeshStageProofReceipt
from verallm.mesh.receipt_signing import (
    sign_receipt_hash,
    sign_stage_proof_receipt_body_hash,
    verify_stage_proof_receipt_signature,
)
from verallm.mesh.verification_snapshot import (
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationStageBinding,
    build_mesh_verification_snapshot,
    sign_mesh_verification_snapshot,
)
from verallm.mesh.gguf_manifest import (
    build_gguf_tensor_manifest,
    gguf_tensor_opening,
    proof_i8_weight_matrix_from_gguf_f32,
    tensor_leaf_bytes,
    verify_gguf_tensor_opening,
)
from verallm.crypto.merkle import FlatWeightMerkle, MerkleTree
from zkllm.config import DEFAULT_W_MERKLE_CHUNK_SIZE
from verallm.mesh.worker import (
    _accumulate_openai_stream_chunk,
    _stream_aggregate_response,
    completion_token_ids_from_response,
    deferred_audit_context_from_receipt,
    deferred_audit_decision,
    post_json,
    probe_worker,
    proof_payload_stage_index,
    semantic_openai_response_hash,
    serve_worker_in_thread,
    strip_backend_verification_fields,
    validate_mesh_stage_context_commitments,
    verify_mesh_inference_artifact,
    verify_mesh_proof_sampling_fields,
)
from verallm.types import InferenceCommitment


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


_NATIVE_STACK_SKIP_REASON = (
    "zkllm native proof stack unavailable for this torch/CUDA environment; "
    "verified serving refuses these paths fail-closed without it (the "
    "shipped wheels cover torch 2.10/2.11)"
)


def _native_proof_stack_available() -> bool:
    # Delegates to the SAME probe production serving gates on: a real
    # kernel launch, not symbol presence. A wheel built without this
    # GPU's arch imports fine and then fails every launch; these tests
    # must skip exactly where serving refuses.
    try:
        from verallm.mesh.gguf_manifest import _gpu_merkle_hash_available

        return bool(_gpu_merkle_hash_available())
    except Exception:
        return False


requires_native_proof_stack = pytest.mark.skipif(
    not _native_proof_stack_available(), reason=_NATIVE_STACK_SKIP_REASON
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _receipt_hash(receipt: dict) -> str:
    return mesh_receipt_hash(receipt)


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validator_post_headers(
    keypair: Keypair,
    *,
    path: str,
    payload: dict,
) -> dict[str, str]:
    # Sign the EXACT wire bytes post_json sends: it serializes with compact
    # separators (worker.post_json), and the server
    # verifies the signature over the raw received body.
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    timestamp = str(int(time.time()))
    signature = keypair.sign(
        build_signing_message("POST", path, body, timestamp)
    )
    return {
        HDR_HOTKEY: keypair.ss58_address,
        HDR_SIGNATURE: signature.hex(),
        HDR_TIMESTAMP: timestamp,
    }


def _tagged_digest_hex(tag: str, *parts: str) -> str:
    h = hashlib.sha256(tag.encode("utf-8"))
    for part in parts:
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def _deferred_randomness_for_sample(receipt: dict, *, sampled: bool) -> str:
    for counter in range(100_000):
        randomness = hashlib.sha256(
            b"VERATHOS_TEST_DEFERRED_RANDOMNESS" + counter.to_bytes(4, "little")
        ).digest()
        decision = deferred_audit_decision(receipt, randomness=randomness)
        if bool(decision.get("sampled", False)) == sampled:
            return randomness.hex()
    raise AssertionError("could not find deferred randomness sample bucket")


def _mesh_response_commitment_hash(receipt: dict) -> str:
    return mesh_response_commitment_hash(receipt)


def _verified_sampler_controls_request(request: dict) -> dict:
    payload = dict(request)
    # Requests without their own completion bound get the serve default cap.
    if "max_tokens" not in payload and "max_completion_tokens" not in payload:
        payload["max_tokens"] = 4096
    payload["temperature"] = 0
    payload["samplers"] = ["top_k"]
    payload["top_k"] = 1
    payload["top_p"] = 1
    payload["min_p"] = 0
    payload["repeat_last_n"] = 0
    payload["repeat_penalty"] = 1.0
    payload["presence_penalty"] = 0.0
    payload["frequency_penalty"] = 0.0
    payload["dry_multiplier"] = 0.0
    payload["mirostat"] = 0
    payload["ignore_eos"] = False
    payload["seed"] = 0
    payload["cache_prompt"] = True
    return payload


def _verified_sampler_backend_request(request: dict) -> dict:
    return _verified_sampler_controls_request(request)


def _verified_backend_request(request: dict) -> dict:
    payload = _verified_sampler_backend_request(request)
    payload["return_tokens"] = True
    payload["verbose"] = True
    return payload


def test_semantic_openai_response_hash_ignores_volatile_replay_fields():
    response = {
        "id": "chatcmpl-a",
        "object": "chat.completion",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "same"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    replay = {**response, "id": "chatcmpl-b", "created": 2}
    assert semantic_openai_response_hash(response) == semantic_openai_response_hash(replay)

    changed = json.loads(json.dumps(response))
    changed["choices"][0]["message"]["content"] = "different"
    assert semantic_openai_response_hash(response) != semantic_openai_response_hash(changed)


def test_verathos_generated_tokens_are_extracted_and_stripped():
    response = {
        "id": "chatcmpl-a",
        "object": "chat.completion",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "same"},
                "finish_reason": "length",
                "verathos_generated_tokens": [11, 22, 33],
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }
    token_ids, token_source = completion_token_ids_from_response(response)
    assert token_ids == [11, 22, 33]
    assert token_source == "choices[0].verathos_generated_tokens"

    cleaned = strip_backend_verification_fields(response)
    assert "verathos_generated_tokens" not in cleaned["choices"][0]
    assert semantic_openai_response_hash(cleaned) == semantic_openai_response_hash(response)


def test_stream_aggregate_uses_llama_cpp_timings_as_usage():
    state: dict = {}
    _accumulate_openai_stream_chunk(
        state,
        {
            "id": "chatcmpl-stream",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "pong"},
                    "finish_reason": None,
                }
            ],
        },
    )
    _accumulate_openai_stream_chunk(
        state,
        {
            "id": "chatcmpl-stream",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length",
                    "verathos_slot_id": 1,
                }
            ],
            "timings": {"prompt_n": 3, "predicted_n": 5},
        },
    )

    response = _stream_aggregate_response(
        request_id="req-stream",
        openai_request={"model": "model", "stream": True},
        state=state,
    )

    assert response["choices"][0]["message"]["content"] == "pong"
    assert response["choices"][0]["verathos_slot_id"] == 1
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    # No reasoning deltas -> no reasoning_content key (absent, not empty).
    assert "reasoning_content" not in response["choices"][0]["message"]


def test_stream_aggregate_collects_reasoning_deltas():
    """Thinking deltas (llama-server --reasoning-format deepseek) must
    survive aggregation into message.reasoning_content: the aggregate is
    the response the receipt hash commits to."""

    state: dict = {}
    for delta in (
        {"role": "assistant", "reasoning_content": "let me think"},
        {"reasoning_content": " harder"},
        {"content": "the answer"},
    ):
        _accumulate_openai_stream_chunk(
            state,
            {
                "id": "chatcmpl-think",
                "created": 1,
                "model": "model",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}
                ],
            },
        )

    response = _stream_aggregate_response(
        request_id="req-think",
        openai_request={"model": "model", "stream": True},
        state=state,
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "the answer"
    assert message["reasoning_content"] == "let me think harder"


def test_stream_aggregate_thinking_only_response_keeps_reasoning():
    """A canary whose whole budget went to thinking still carries the
    reasoning text in the committed response."""

    state: dict = {}
    _accumulate_openai_stream_chunk(
        state,
        {
            "id": "chatcmpl-think-only",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "all thinking, no answer",
                    },
                    "finish_reason": "length",
                }
            ],
        },
    )

    response = _stream_aggregate_response(
        request_id="req-think-only",
        openai_request={"model": "model", "stream": True},
        state=state,
    )

    message = response["choices"][0]["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "all thinking, no answer"


def test_llama_server_command_pins_reasoning_format_deepseek():
    """The serve argv always requests reasoning_content extraction, and an
    operator-supplied override in extra args wins (no duplicate flag)."""

    from verallm.mesh.llama_cpp import build_llama_server_command

    cmd = build_llama_server_command(model="/models/m.gguf")
    fmt_index = cmd.index("--reasoning-format")
    assert cmd[fmt_index + 1] == "deepseek"

    overridden = build_llama_server_command(
        model="/models/m.gguf",
        extra_args=["--reasoning-format", "none"],
    )
    assert overridden.count("--reasoning-format") == 1
    fmt_index = overridden.index("--reasoning-format")
    assert overridden[fmt_index + 1] == "none"


def _fake_openai_backend(content: str = "pong", on_request=None, stream_failure=None):
    """Fake llama-server OpenAI backend.

    stream_failure selects a mid-stream failure shape for streaming
    requests, mirroring the patched llama-server's contract that the
    [DONE] sentinel is sent only after a completed serve:
      - "truncate_mid_stream": relay the first chunk, then close the
        connection with no final chunk and no [DONE] (a backend crash,
        observed as a CUDA abort mid-serve).
      - "error_event": relay the first chunk, then an {"error": ...}
        data payload, then close with no [DONE] (llama-server's
        mid-stream error path).
    """

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, payloads: list[dict], done: bool = True) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for payload in payloads:
                self.wfile.write(
                    f"data: {json.dumps(payload, sort_keys=True)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            if done:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = self.path.rstrip("/")
            if path == "/apply-template":
                payload = self._read_json_body()
                self._send_json(
                    200,
                    {"prompt": json.dumps(payload.get("messages", []), sort_keys=True)},
                )
                return
            if path == "/tokenize":
                self._read_json_body()
                self._send_json(200, {"tokens": [201, 202, 203]})
                return
            if path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            payload = self._read_json_body()
            calls.append(payload)
            if on_request is not None:
                on_request(payload)
            if payload.get("stream"):
                midpoint = max(1, len(content) // 2)
                # llama.cpp emits per-chunk token ids when return_tokens is
                # set, and the mesh receipt path needs them for the decode
                # audit, which cannot fall back to re-tokenizing the text.
                # One token id total, matching the usage block below, and
                # matching the single final-projection op the trace fixtures
                # write per request.
                # The patched llama-server honors return_tokens on the
                # streaming OpenAI route and emits the chunk's sampled ids
                # as a top-level "tokens" extension field (same family as
                # "timings"). Verified against the live patched server; the
                # fake backend mirrors that shape so the tests exercise the
                # carrier the code actually has to read. The old
                # logprobs.content[].id carrier stays supported reader-side
                # but is no longer requested: llama-server sorts the full
                # vocab per generated token for logprobs (measured -30%
                # aggregate at 8-way concurrency).
                first_logprobs = None
                second_logprobs = (
                    {"content": [{"id": 101, "token": "pong"}]}
                    if payload.get("logprobs")
                    else None
                )
                second_chunk_tokens = (
                    [101]
                    if payload.get("return_tokens") and not payload.get("logprobs")
                    else None
                )
                if stream_failure is not None:
                    first_chunk = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": payload.get("model", "mesh-test"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content[:midpoint],
                                },
                                "logprobs": first_logprobs,
                                "finish_reason": None,
                            }
                        ],
                    }
                    if stream_failure == "truncate_mid_stream":
                        self._send_sse([first_chunk], done=False)
                        return
                    if stream_failure == "error_event":
                        self._send_sse(
                            [
                                first_chunk,
                                {
                                    "error": {
                                        "message": (
                                            "CUDA error: an illegal memory "
                                            "access was encountered"
                                        ),
                                        "type": "server_error",
                                        "code": 500,
                                    }
                                },
                            ],
                            done=False,
                        )
                        return
                    raise ValueError(
                        f"unknown stream_failure mode: {stream_failure!r}"
                    )
                self._send_sse(
                    [
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload.get("model", "mesh-test"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content[:midpoint]},
                                    "logprobs": first_logprobs,
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload.get("model", "mesh-test"),
                            **(
                                {"tokens": second_chunk_tokens}
                                if second_chunk_tokens
                                else {}
                            ),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content[midpoint:]},
                                    "logprobs": second_logprobs,
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload.get("model", "mesh-test"),
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                    ]
                )
                return
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": payload.get("model", "mesh-test"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            if payload.get("verbose"):
                response["__verbose"] = {
                    "content": content,
                    "tokens": [101] if payload.get("return_tokens") else [],
                }
            self._send_json(200, response)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}", calls


def _post_sse_events(url: str, payload: dict) -> list[tuple[str, object]]:
    req = Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    events = []
    with urlopen(req, timeout=10.0) as resp:
        event = ""
        data_lines = []
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            elif line == "":
                if data_lines:
                    data = "\n".join(data_lines)
                    if data == "[DONE]":
                        events.append((event or "message", "[DONE]"))
                        break
                    else:
                        parsed = json.loads(data)
                        events.append((event or parsed.get("event") or "message", parsed))
                event = ""
                data_lines = []
    return events


def _fake_proof_adapter():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.rstrip("/") == "/v1/mesh/proof/commitment":
                payload = self._read_json_body()
                ctx = payload["receipt_context"]
                self._send_json(
                    200,
                    {
                        "version": 1,
                        "stage_index": int(ctx.get("stage_index", 0)),
                        "trace_commitment_root": _tagged_digest_hex(
                            "trace-root",
                            ctx["request_id"],
                            ctx["request_hash"],
                            ctx["response_hash"],
                            str(ctx.get("stage_index", 0)),
                        ),
                        "trace_commitment_count": int(
                            ctx.get("proof_trace_candidates_per_request")
                            or ctx.get("proof_ops_per_request")
                            or 1
                        ),
                    },
                )
                return
            if self.path.rstrip("/") != "/v1/mesh/proof/receipt":
                self._send_json(404, {"error": "not found"})
                return
            payload = self._read_json_body()
            calls.append(payload)
            ctx = payload["receipt_context"]
            proof = LlamaGraphOpReceipt(
                request_id=ctx["request_id"],
                mesh_id=ctx["mesh_id"],
                mesh_spec_hash=ctx["mesh_spec_hash"],
                stage_assignment_hash=ctx["stage_assignment_hash"],
                rpc_plan_hash=ctx["rpc_plan_hash"],
                model_package_hash=ctx.get("model_package_hash", ""),
                model_tensor_manifest_root=ctx.get("model_tensor_manifest_root", ""),
                uid=ctx["uid"],
                hotkey=ctx["hotkey"],
                endpoint=ctx["endpoint"],
                stage_index=ctx["stage_index"],
                layer_start=ctx["layer_start"],
                layer_end=ctx["layer_end"],
                request_hash=ctx["request_hash"],
                response_hash=ctx["response_hash"],
                graph_id=_tagged_digest_hex(
                    "graph",
                    ctx["request_id"],
                    ctx["request_hash"],
                    ctx["response_hash"],
                ),
                op_index=0,
                op_type="GGML_OP_MUL_MAT",
                layer_index=int(ctx["layer_start"]),
                tensor_name="blk.0.ffn_up.weight",
                input_root=_tagged_digest_hex("input", ctx["request_hash"]),
                weight_root=_tagged_digest_hex("weight", ctx["mesh_spec_hash"]),
                output_root=_tagged_digest_hex("output", ctx["response_hash"]),
                quantization="Q8_0",
                backend=ctx["runtime"],
                device="test-device",
                proof_kind="trace",
                proof_commitment_hash=_tagged_digest_hex(
                    "proof",
                    ctx["request_id"],
                    ctx["stage_assignment_hash"],
                    ctx["response_hash"],
                ),
            )
            self._send_json(
                200,
                {
                    "proof_mode": VERATHOS_GGML_TRACE_PROOF_MODE,
                    "proof_receipts": [proof.to_dict()],
                },
            )

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}", calls


def test_proof_payload_stage_index_prefers_membership_metadata():
    assert proof_payload_stage_index({"stage_index": 3}) == 3
    assert proof_payload_stage_index({"op_manifest_membership": {"stage_index": 2}}) == 2
    assert proof_payload_stage_index({"trace_membership": {"stage_index": 1}}) == 1
    assert proof_payload_stage_index({"trace": {"stage_index": 4}}) == 4
    assert proof_payload_stage_index({}) == 0


def _write_ggml_mul_mat_trace(
    tmp_path,
    *,
    created_unix_ns=None,
    name: str = "trace",
    op_index: int = 0,
    manifest_index: int | None = None,
    write_trace: bool = True,
) -> GgmlMulMatTrace:
    trace_dir = tmp_path / "ggml-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    created_unix_ns = int(created_unix_ns or time.time_ns())
    x = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32)
    w = np.array(
        [
            [2.0, -1.0, 0.5, 3.0],
            [0.0, 4.0, -2.0, 1.0],
            [-3.0, 1.5, 2.0, 0.25],
        ],
        dtype=np.float32,
    )
    y = x @ w
    # GGML MUL_MAT layout: src0 [K,N] stored as [N,K], src1 [K,M]
    # stored as [M,K], dst [N,M] stored as [M,N].
    src0 = np.ascontiguousarray(w.T)
    src1 = np.ascontiguousarray(x)
    dst = np.ascontiguousarray(y)
    src0_path = trace_dir / f"{name}-src0.f32"
    src0_raw_path = trace_dir / f"{name}-src0.raw"
    src1_path = trace_dir / f"{name}-src1.f32"
    dst_path = trace_dir / f"{name}-dst.f32"
    src0.tofile(src0_path)
    src0.tofile(src0_raw_path)
    src1.tofile(src1_path)
    dst.tofile(dst_path)
    trace_path = trace_dir / f"{name}.json"
    manifest_index = int(op_index if manifest_index is None else manifest_index)
    manifest_entry = {
        "version": 1,
        "created_unix_ns": created_unix_ns,
        "manifest_index": int(manifest_index),
        "graph_id": "graph-real-gemm",
        "op_index": int(op_index),
        "op_type": "GGML_OP_MUL_MAT",
        "tensor_name": "blk.0.test.weight",
        "src0_name": "blk.0.test.weight",
        "src1_name": "blk.0.test.activation",
        "dst_name": "blk.0.test.output",
        "src0_shape": [3, 4, 1, 1],
        "src1_shape": [3, 2, 1, 1],
        "dst_shape": [4, 2, 1, 1],
        "source_types": {"src0": "F32", "src1": "F32", "dst": "F32"},
        "backend": "llama_cpp_rpc",
        "device": "test-device",
        "proof_eligible": True,
    }
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry, sort_keys=True) + "\n")
    if not write_trace:
        return GgmlMulMatTrace(
            path=trace_path,
            created_unix_ns=created_unix_ns,
            graph_id=str(manifest_entry["graph_id"]),
            op_index=int(op_index),
            tensor_name="blk.0.test.weight",
            src0_name="blk.0.test.weight",
            src1_name="blk.0.test.activation",
            dst_name="blk.0.test.output",
            src0_shape=tuple(manifest_entry["src0_shape"]),
            src1_shape=tuple(manifest_entry["src1_shape"]),
            dst_shape=tuple(manifest_entry["dst_shape"]),
            src0_f32_path=src0_path,
            src1_f32_path=src1_path,
            dst_f32_path=dst_path,
            source_types=dict(manifest_entry["source_types"]),
            src0_raw_path=src0_raw_path,
            src0_raw_type="F32",
            backend="llama_cpp_rpc",
            device="test-device",
            manifest_index=manifest_index,
        )
    trace_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_unix_ns": created_unix_ns,
                "manifest_index": int(manifest_index),
                "graph_id": "graph-real-gemm",
                "op_index": int(op_index),
                "op_type": "GGML_OP_MUL_MAT",
                "tensor_name": "blk.0.test.weight",
                "src0_name": "blk.0.test.weight",
                "src1_name": "blk.0.test.activation",
                "dst_name": "blk.0.test.output",
                "src0_shape": [3, 4, 1, 1],
                "src1_shape": [3, 2, 1, 1],
                "dst_shape": [4, 2, 1, 1],
                "src0_f32": src0_path.name,
                "src0_raw": src0_raw_path.name,
                "src0_raw_type": "F32",
                "src1_f32": src1_path.name,
                "dst_f32": dst_path.name,
                "source_types": {"src0": "F32", "src1": "F32", "dst": "F32"},
                "backend": "llama_cpp_rpc",
                "device": "test-device",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return GgmlMulMatTrace.from_json(trace_path)


def _write_decode_lm_head_trace(
    tmp_path,
    *,
    token_id: int = 101,
    created_unix_ns=None,
    name: str = "decode-lm-head",
    op_index: int = 0,
    manifest_index: int | None = None,
    write_trace: bool = True,
) -> GgmlMulMatTrace:
    trace_dir = tmp_path / "ggml-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    created_unix_ns = int(created_unix_ns or time.time_ns())
    vocab = max(128, int(token_id) + 8)
    hidden = 4
    x = np.array([[1.0, 0.25, -0.5, 0.125]], dtype=np.float32)
    w = np.zeros((hidden, vocab), dtype=np.float32)
    w[0, :] = np.linspace(-0.5, 0.5, vocab, dtype=np.float32)
    w[:, int(token_id)] = np.array([6.0, 1.0, -1.0, 0.5], dtype=np.float32)
    y = x @ w
    assert int(np.argmax(y[0])) == int(token_id)

    src0 = np.ascontiguousarray(w.T)
    src1 = np.ascontiguousarray(x)
    dst = np.ascontiguousarray(y)
    src0_path = trace_dir / f"{name}-src0.f32"
    src0_raw_path = trace_dir / f"{name}-src0.raw"
    src1_path = trace_dir / f"{name}-src1.f32"
    dst_path = trace_dir / f"{name}-dst.f32"
    src0.tofile(src0_path)
    src0.tofile(src0_raw_path)
    src1.tofile(src1_path)
    dst.tofile(dst_path)
    trace_path = trace_dir / f"{name}.json"
    manifest_index = int(op_index if manifest_index is None else manifest_index)
    meta = {
        "version": 1,
        "created_unix_ns": created_unix_ns,
        "manifest_index": int(manifest_index),
        "graph_id": f"decode-graph-{manifest_index}",
        "op_index": int(op_index),
        "op_type": "GGML_OP_MUL_MAT",
        "tensor_name": "output.weight",
        "src0_name": "output.weight",
        "src1_name": "decode.hidden",
        "dst_name": "logits",
        "src0_shape": [hidden, vocab, 1, 1],
        "src1_shape": [hidden, 1, 1, 1],
        "dst_shape": [vocab, 1, 1, 1],
        "source_types": {"src0": "F32", "src1": "F32", "dst": "F32"},
        "backend": "llama_cpp_rpc",
        "device": "test-device",
        "proof_eligible": True,
    }
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, sort_keys=True) + "\n")
    if not write_trace:
        return GgmlMulMatTrace(
            path=trace_path,
            created_unix_ns=created_unix_ns,
            graph_id=str(meta["graph_id"]),
            op_index=int(op_index),
            tensor_name="output.weight",
            src0_name="output.weight",
            src1_name="decode.hidden",
            dst_name="logits",
            src0_shape=tuple(meta["src0_shape"]),
            src1_shape=tuple(meta["src1_shape"]),
            dst_shape=tuple(meta["dst_shape"]),
            src0_f32_path=src0_path,
            src1_f32_path=src1_path,
            dst_f32_path=dst_path,
            source_types=dict(meta["source_types"]),
            src0_raw_path=src0_raw_path,
            src0_raw_type="F32",
            backend="llama_cpp_rpc",
            device="test-device",
            manifest_index=manifest_index,
        )
    trace_path.write_text(
        json.dumps(
            {
                **meta,
                "src0_f32": src0_path.name,
                "src0_raw": src0_raw_path.name,
                "src0_raw_type": "F32",
                "src1_f32": src1_path.name,
                "dst_f32": dst_path.name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return GgmlMulMatTrace.from_json(trace_path)


def _gguf_manifest_record(trace: GgmlMulMatTrace) -> dict:
    raw = trace.src0_raw_path.read_bytes()
    f32 = trace.src0_f32_path.read_bytes()
    f32_data = np.frombuffer(f32, dtype=np.float32)
    proof_i8 = proof_i8_weight_matrix_from_gguf_f32(f32_data, list(trace.src0_shape))
    proof_i8_bytes = proof_i8.tobytes(order="C")
    proof_i8_merkle = FlatWeightMerkle(
        torch.from_numpy(proof_i8),
        DEFAULT_W_MERKLE_CHUNK_SIZE,
        store_raw=False,
    )
    record = {
        "name": trace.src0_name or trace.tensor_name,
        "tensor_type": trace.src0_raw_type or "F32",
        "shape": list(trace.src0_shape),
        "n_elements": int(np.prod(trace.src0_shape)),
        "n_bytes": len(raw),
        "data_offset": 0,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_path": str(trace.src0_raw_path),
        "f32_nbytes": len(f32),
        "f32_sha256": hashlib.sha256(f32).hexdigest(),
        "f32_path": str(trace.src0_f32_path),
        "proof_i8_nbytes": len(proof_i8_bytes),
        "proof_i8_sha256": hashlib.sha256(proof_i8_bytes).hexdigest(),
        "proof_i8_merkle_root": proof_i8_merkle.root.hex(),
        "proof_i8_chunk_size": int(DEFAULT_W_MERKLE_CHUNK_SIZE),
    }
    return record


def _gguf_manifest_for_traces(*traces: GgmlMulMatTrace) -> dict:
    """Build a manifest covering every tensor the traces reference.

    Decode audit needs a final-projection tensor in the same manifest as the
    layer GEMMs, so fixtures that exercise the full policy pass both traces.
    """

    records = [_gguf_manifest_record(trace) for trace in traces]
    root = MerkleTree(
        [tensor_leaf_bytes(record) for record in records]
    ).root.hex()
    return {
        "version": 1,
        "model_file": "test.gguf",
        "model_file_sha256": DIGEST_C,
        "tensor_count": len(records),
        "tensor_manifest_root": root,
        "tensors": records,
    }


def _gguf_manifest_for_trace(trace: GgmlMulMatTrace) -> dict:
    return _gguf_manifest_for_traces(trace)


def test_ggml_trace_max_elems_scales_from_repeated_matrix_sizes():
    records = []
    for layer in range(24):
        records.append(
            {
                "name": f"blk.{layer}.attn_k.weight",
                "shape": [4096, 1024],
                "n_elements": 4096 * 1024,
            }
        )
        records.append(
            {
                "name": f"blk.{layer}.attn_q.weight",
                "shape": [4096, 4096],
                "n_elements": 4096 * 4096,
            }
        )
        records.append(
            {
                "name": f"blk.{layer}.ffn_gate.weight",
                "shape": [4096, 14336],
                "n_elements": 4096 * 14336,
            }
        )
    assert suggest_ggml_trace_max_elems_from_tensor_records(records) == 8_388_608


def test_gguf_tensor_manifest_hashes_real_tensor_bytes(tmp_path):
    import gguf

    model_path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(model_path, "llama")
    writer.add_tensor(
        "blk.0.test.weight",
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    writer.add_tensor(
        "blk.1.test.weight",
        np.arange(4, dtype=np.float16).reshape(2, 2),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = build_gguf_tensor_manifest(model_path)
    reader = gguf.GGUFReader(model_path)
    reader_tensors = {tensor.name: tensor for tensor in reader.tensors}

    assert manifest["tensor_count"] == 2
    for record in manifest["tensors"]:
        raw = memoryview(reader_tensors[record["name"]].data).cast("B")
        assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]
        f32 = gguf.dequantize(
            reader_tensors[record["name"]].data,
            reader_tensors[record["name"]].tensor_type,
        ).astype("float32", copy=False)
        assert hashlib.sha256(f32.tobytes(order="C")).hexdigest() == record["f32_sha256"]
        proof_i8 = proof_i8_weight_matrix_from_gguf_f32(f32, record["shape"])
        proof_i8_merkle = FlatWeightMerkle(
            torch.from_numpy(proof_i8),
            DEFAULT_W_MERKLE_CHUNK_SIZE,
            store_raw=False,
        )
        assert record["proof_i8_merkle_root"] == proof_i8_merkle.root.hex()
        opening = gguf_tensor_opening(manifest, record["name"])
        assert verify_gguf_tensor_opening(
            opening,
            expected_root=manifest["tensor_manifest_root"],
            expected_tensor_name=record["name"],
            expected_raw_sha256=record["raw_sha256"],
            expected_f32_sha256=record["f32_sha256"],
            expected_proof_i8_merkle_root=record["proof_i8_merkle_root"],
            expected_proof_i8_chunk_size=record["proof_i8_chunk_size"],
            expected_n_bytes=record["n_bytes"],
            expected_f32_n_bytes=record["f32_nbytes"],
        )


def test_gguf_tensor_manifest_auto_expands_split_shards(tmp_path):
    import gguf

    first = tmp_path / "tiny-q4_k_m-00001-of-00002.gguf"
    second = tmp_path / "tiny-q4_k_m-00002-of-00002.gguf"
    writer = gguf.GGUFWriter(first, "llama")
    writer.add_tensor("blk.0.test.weight", np.arange(6, dtype=np.float32).reshape(2, 3))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    writer = gguf.GGUFWriter(second, "llama")
    writer.add_tensor("blk.1.test.weight", np.arange(8, dtype=np.float32).reshape(2, 4))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = build_gguf_tensor_manifest(first)

    assert manifest["tensor_count"] == 2
    assert len(manifest["model_files"]) == 2
    assert {record["name"] for record in manifest["tensors"]} == {
        "blk.0.test.weight",
        "blk.1.test.weight",
    }
    assert {record["model_file_index"] for record in manifest["tensors"]} == {0, 1}


def _receipt_context_for_trace(mesh_spec: MeshSpec, worker_endpoint: str) -> dict:
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    response = {"choices": [{"message": {"content": "pong"}}]}
    return {
        "request_id": "req-real-gemm",
        "mesh_id": mesh_spec.mesh_id,
        "mesh_spec_hash": mesh_spec.spec_hash_hex(),
        "stage_assignment_hash": mesh_spec.stage_assignment_hash_hex(),
        "model_package_hash": mesh_spec.model_package_hash,
        "model_tensor_manifest_root": mesh_spec.model_tensor_manifest_root,
        "model_total_layers": mesh_spec.total_layers,
        "rpc_plan_hash": rpc_plan_from_mesh(mesh_spec).plan_hash_hex(),
        "runtime": "llama_cpp_rpc",
        "uid": 1,
        "hotkey": "5Coord",
        "endpoint": worker_endpoint,
        "stage_index": 1,
        "layer_start": 2,
        "layer_end": 4,
        "request_hash": _payload_hash(request),
        "response_hash": _payload_hash(response),
        "inference_started_unix_ns": 1,
        "inference_ended_unix_ns": 2,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mesh_id", "mesh-coordinator-relabel"),
        ("mesh_spec_hash", DIGEST_B),
        ("stage_assignment_hash", DIGEST_C),
        ("rpc_plan_hash", DIGEST_D),
        ("model_package_hash", DIGEST_B),
        ("model_tensor_manifest_root", DIGEST_C),
        ("model_total_layers", 999),
    ],
)
def test_stage_worker_recomputes_common_context_before_signing(field, value):
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    context = _receipt_context_for_trace(
        mesh_spec,
        "http://worker.local:9338",
    )
    validate_mesh_stage_context_commitments(context, mesh_spec)

    tampered = {**context, field: value}
    with pytest.raises(RuntimeError, match=rf"proof context {field} mismatch"):
        validate_mesh_stage_context_commitments(tampered, mesh_spec)


def _shutdown_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _wait_for_http(endpoint: str, *, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            post_json(
                endpoint.rstrip("/") + "/v1/chat/completions",
                {
                    "model": "probe",
                    "messages": [{"role": "user", "content": "probe"}],
                    "stream": False,
                },
                timeout=0.5,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"{endpoint} did not become ready: {last_error}")


def _wait_for_probe(
    endpoint: str,
    *,
    timeout: float = 15.0,
    internal_auth_secret: str | bytes = "",
) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            probe_worker(
                endpoint,
                timeout=0.5,
                internal_auth_secret=internal_auth_secret,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"{endpoint} did not become probeable: {last_error}")


def _wait_for_file(path, *, timeout: float = 8.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return path.read_text(encoding="utf-8")
        time.sleep(0.1)
    raise RuntimeError(f"{path} was not written")


def _wait_for_mesh_member_count(state_dir, count: int, *, timeout: float = 8.0) -> MeshSpec:
    deadline = time.time() + timeout
    last_spec = None
    while time.time() < deadline:
        last_spec = state_mesh_spec(load_mesh_state(state_dir))
        if len(last_spec.members) == count:
            return last_spec
        time.sleep(0.1)
    raise RuntimeError(
        f"{state_dir} did not reach {count} mesh members; "
        f"last={len(last_spec.members) if last_spec else 'missing'}"
    )


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _write_fake_llama_cpp_binaries(tmp_path):
    llama_args = tmp_path / "llama-args.json"
    rpc_args = tmp_path / "rpc-args.json"
    fake_llama = tmp_path / "llama-server"
    fake_rpc = tmp_path / "rpc-server"
    fake_llama.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--rpc", default="")
parser.add_argument("--device", default="")
parser.add_argument("--n-gpu-layers", default="")
parser.add_argument("--ctx-size", default="")
parser.add_argument("--tensor-split", default="")
parser.add_argument("--alias", default="")
args, extra = parser.parse_known_args()
with open(os.environ["FAKE_LLAMA_ARGS_FILE"], "w", encoding="utf-8") as f:
    json.dump({"args": vars(args), "extra": extra}, f, sort_keys=True)

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode() if length else "{}")
        self._send_json(200, {
            "id": "fake-llama",
            "object": "chat.completion",
            "created": 1,
            "model": request.get("model", args.alias or args.model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "fake mesh ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        })
    def log_message(self, fmt, *values):
        return

server = ThreadingHTTPServer((args.host, args.port), Handler)
server.daemon_threads = True
server.serve_forever()
""",
        encoding="utf-8",
    )
    fake_rpc.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
import select
import signal
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", default="0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=50052)
parser.add_argument("--device", default="")
parser.add_argument("-c", "--cache", action="store_true")
args, extra = parser.parse_known_args()
with open(os.environ["FAKE_RPC_ARGS_FILE"], "w", encoding="utf-8") as f:
    json.dump({"args": vars(args), "extra": extra}, f, sort_keys=True)

running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((args.host, args.port))
listener.listen()
listener.setblocking(False)
while running:
    ready, _, _ = select.select([listener], [], [], 0.1)
    if ready:
        conn, _ = listener.accept()
        conn.close()
listener.close()
""",
        encoding="utf-8",
    )
    fake_llama.chmod(0o755)
    fake_rpc.chmod(0o755)
    return fake_llama, fake_rpc, llama_args, rpc_args


def _two_member_mesh(
    *,
    coordinator_endpoint: str,
    worker_endpoint: str,
    model_tensor_manifest_root: str = "",
) -> MeshSpec:
    return MeshSpec(
        mesh_id="mesh-test",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        model_tensor_manifest_root=model_tensor_manifest_root,
        total_layers=4,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=coordinator_endpoint,
                stage_index=0,
                layers=StageRange(0, 2),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=worker_endpoint,
                rpc_endpoint="worker.local:50052",
                rpc_split_weight=1,
                proof_endpoint=worker_endpoint,
                stage_index=1,
                layers=StageRange(2, 4),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
        ],
    )


def _make_coordinator_orchestration_only(spec: MeshSpec) -> MeshSpec:
    """Move all compute layers to the worker for coordinator fan-in tests."""

    coordinator = next(member for member in spec.members if member.role == "coordinator")
    workers = [member for member in spec.members if member.role != "coordinator"]
    assert len(workers) == 1
    coordinator.layers = StageRange(0, 0)
    workers[0].layers = StageRange(0, spec.total_layers)
    spec.validate()
    return spec


def test_private_mesh_hashes_round_trip(tmp_path):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=42,
        coordinator_hotkey="5Hotkey",
        endpoint="https://miner.example.com",
        model_id="qwen3-32b",
        model_package_ref="hf://org/repo@abc",
        model_package_hash=DIGEST_A,
        tokenizer_hash=DIGEST_B,
        quantization_scheme="Q8_0",
        activation_dtype="f16",
        max_context_len=32_768,
        total_layers=64,
    )

    path = save_json(tmp_path / "mesh.json", spec.to_dict())
    loaded = load_mesh_spec(path)

    assert loaded.mesh_id == spec.mesh_id
    assert loaded.spec_hash() == spec.spec_hash()
    assert loaded.body_hash() == spec.body_hash()
    assert loaded.stage_assignment_hash() == spec.stage_assignment_hash()
    assert loaded.members[0].payout_bps == 10000
    assert loaded.max_context_len == 32_768


def test_mesh_context_limit_is_strict_and_binds_spec_and_assignment_hashes():
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=42,
        coordinator_hotkey="5Hotkey",
        endpoint="https://miner.example.com",
        model_id="qwen3-32b",
        model_package_hash=DIGEST_A,
        total_layers=64,
    )
    legacy_spec_hash = spec.spec_hash()
    legacy_assignment_hash = spec.stage_assignment_hash()
    assert "max_context_len" not in spec.to_dict()
    assert MeshSpec.from_dict(spec.to_dict()).max_context_len == 0

    spec.max_context_len = 32_768
    spec.validate()
    assert spec.to_dict()["max_context_len"] == 32_768
    assert spec.spec_hash() != legacy_spec_hash
    assert spec.stage_assignment_hash() != legacy_assignment_hash

    for invalid in (None, True, "32768", -1, 2**32):
        payload = spec.to_dict()
        payload["max_context_len"] = invalid
        with pytest.raises(ValueError, match="max_context_len"):
            MeshSpec.from_dict(payload)


def test_mesh_member_proof_endpoint_round_trips_and_binds_assignment(tmp_path):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    base_hash = spec.stage_assignment_hash()
    spec.members[1].proof_endpoint = "http://worker.local:9339"

    path = save_json(tmp_path / "mesh.json", spec.to_dict())
    loaded = load_mesh_spec(path)

    assert loaded.members[1].proof_endpoint == "http://worker.local:9339"
    assert loaded.stage_assignment_hash() != base_hash


def test_mesh_member_rpc_split_weight_round_trips_and_binds_assignment(tmp_path):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    base_hash = spec.stage_assignment_hash()
    spec.members[1].rpc_split_weight = 24

    path = save_json(tmp_path / "mesh.json", spec.to_dict())
    loaded = load_mesh_spec(path)

    assert loaded.members[0].to_dict()["rpc_split_weight"] == 0
    assert loaded.members[1].rpc_split_weight == 24
    assert loaded.members[1].to_dict()["rpc_split_weight"] == 24
    assert loaded.stage_assignment_hash() != base_hash


def test_mesh_member_rpc_split_weight_legacy_zero_and_validation():
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    legacy = spec.to_dict()
    legacy["members"][1].pop("rpc_split_weight")

    loaded = MeshSpec.from_dict(legacy)
    assert loaded.members[1].rpc_split_weight == 0
    with pytest.raises(ValueError, match="split weights must be positive"):
        rpc_plan_from_mesh(loaded)

    loaded.members[1].rpc_split_weight = -1
    with pytest.raises(ValueError, match="rpc_split_weight must be >= 0"):
        loaded.validate()

    loaded.members[1].rpc_split_weight = 1
    loaded.members[1].rpc_endpoint = ""
    with pytest.raises(ValueError, match="must be zero without an rpc_endpoint"):
        loaded.validate()

    loaded.members[1].rpc_split_weight = 0
    loaded.members[0].rpc_split_weight = 1
    with pytest.raises(ValueError, match="coordinator rpc_split_weight must be zero"):
        loaded.validate()

    loaded.members[0].role = "worker"
    loaded.members[0].rpc_endpoint = "coord.local:50052"
    with pytest.raises(ValueError, match="coordinator rpc_split_weight must be zero"):
        loaded.validate()


def test_mesh_requires_contiguous_layer_ranges():
    spec = MeshSpec(
        mesh_id="mesh-test",
        mode="declared",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="https://a.example.com",
                stage_index=0,
                layers=StageRange(0, 2),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=5000,
            ),
            MeshMember(
                uid=2,
                hotkey="5Worker",
                endpoint="https://b.example.com",
                stage_index=1,
                layers=StageRange(3, 4),
                payout_bps=5000,
            ),
        ],
    )

    with pytest.raises(ValueError, match="without gaps"):
        spec.validate()


def test_capability_ad_hash_changes_with_signature():
    ad = CapabilityAd(
        uid=7,
        hotkey="5Worker",
        endpoint="https://worker.example.com",
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    body_hash = ad.body_hash()
    unsigned_hash = ad.ad_hash()

    ad.signatures["5Worker"] = "0x1234"

    assert ad.body_hash() == body_hash
    assert ad.ad_hash() != unsigned_hash


def test_stage_receipt_root_is_order_independent():
    r1 = StageReceipt(
        request_id="req",
        commitment_hash=DIGEST_A,
        uid=1,
        stage_index=0,
        layer_start=0,
        layer_end=2,
        input_activation_root=DIGEST_B,
        output_activation_root=DIGEST_C,
        proof_commitment_hash=DIGEST_D,
        inference_ms=12.5,
        bytes_in=10,
        bytes_out=20,
        signature="sig-a",
    )
    r2 = StageReceipt(
        request_id="req",
        commitment_hash=DIGEST_A,
        uid=2,
        stage_index=1,
        layer_start=2,
        layer_end=4,
        input_activation_root=DIGEST_C,
        output_activation_root=DIGEST_B,
        proof_commitment_hash=DIGEST_D,
        inference_ms=13.5,
        bytes_in=20,
        bytes_out=30,
        signature="sig-b",
    )

    assert stage_receipt_root([r1, r2]) == stage_receipt_root([r2, r1])


def test_llama_graph_receipt_root_binds_graph_and_mesh_fields():
    r1 = LlamaGraphOpReceipt(
        request_id="req",
        mesh_id="mesh-test",
        mesh_spec_hash=DIGEST_A,
        stage_assignment_hash=DIGEST_B,
        rpc_plan_hash=DIGEST_C,
        model_package_hash=DIGEST_A,
        model_tensor_manifest_root="",
        uid=1,
        hotkey="5Worker",
        endpoint="http://worker.local:9338",
        stage_index=0,
        layer_start=0,
        layer_end=2,
        request_hash=DIGEST_A,
        response_hash=DIGEST_B,
        graph_id="graph-a",
        op_index=0,
        op_type="GGML_OP_MUL_MAT",
        layer_index=0,
        tensor_name="blk.0.attn_q.weight",
        input_root=DIGEST_A,
        weight_root=DIGEST_B,
        output_root=DIGEST_C,
        quantization="Q8_0",
        backend="llama_cpp_rpc",
        device="CUDA0",
        proof_kind="trace",
        proof_commitment_hash=DIGEST_D,
        signature="sig-a",
    )
    r2 = LlamaGraphOpReceipt(
        request_id="req",
        mesh_id="mesh-test",
        mesh_spec_hash=DIGEST_A,
        stage_assignment_hash=DIGEST_B,
        rpc_plan_hash=DIGEST_C,
        model_package_hash=DIGEST_A,
        model_tensor_manifest_root="",
        uid=1,
        hotkey="5Worker",
        endpoint="http://worker.local:9338",
        stage_index=0,
        layer_start=0,
        layer_end=2,
        request_hash=DIGEST_A,
        response_hash=DIGEST_B,
        graph_id="graph-a",
        op_index=1,
        op_type="GGML_OP_MUL_MAT",
        layer_index=0,
        tensor_name="blk.0.attn_k.weight",
        input_root=DIGEST_A,
        weight_root=DIGEST_C,
        output_root=DIGEST_D,
        quantization="Q8_0",
        backend="llama_cpp_rpc",
        device="CUDA0",
        proof_kind="trace",
        proof_commitment_hash=DIGEST_D,
        signature="sig-b",
    )
    tampered = LlamaGraphOpReceipt.from_dict(
        {**r2.to_dict(), "output_root": DIGEST_A}
    )

    assert llama_graph_receipt_root([r1, r2]) == llama_graph_receipt_root([r2, r1])
    assert llama_graph_receipt_root([r1, r2]) != llama_graph_receipt_root([r1, tampered])


def test_ggml_proof_adapter_generates_and_verifies_real_gemm_proof(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )

    assert proof.verified is True
    assert proof.proof_mode == VERATHOS_GGML_GEMM_PROOF_MODE
    assert proof.float_max_abs_error == 0.0
    assert proof.receipt.proof_kind == "gemm"
    assert proof.receipt.op_type == "GGML_OP_MUL_MAT"
    assert proof.receipt.proof_commitment_hash == proof.proof_commitment_hash
    independent = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    )
    assert independent.verified is True
    assert independent.verifier_ms >= 0.0


def test_ggml_proof_adapter_challenges_full_manifest_beyond_trace_prefix(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    base_ns = time.time_ns()
    trace_a = _write_ggml_mul_mat_trace(
        tmp_path,
        created_unix_ns=base_ns,
        name="trace-a",
        op_index=0,
    )
    _write_ggml_mul_mat_trace(
        tmp_path,
        created_unix_ns=base_ns + 1,
        name="trace-b",
        op_index=1,
    )
    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    proof_server = make_ggml_proof_server(
        trace_dir=trace_a.path.parent,
        tolerance_abs=1e-6,
    )
    proof_thread = threading.Thread(target=proof_server.serve_forever, daemon=True)
    proof_thread.start()
    proof_host, proof_port = proof_server.server_address
    proof_url = f"http://{proof_host}:{proof_port}"
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx.update(
        {
            "request_id": "req-proof-adapter-manifest-wide",
            "inference_started_unix_ns": base_ns - 1,
            "inference_ended_unix_ns": base_ns + 2,
            "proof_ops_per_request": 1,
            "proof_trace_candidates_per_request": 1,
        }
    )
    try:
        commitment = post_json(
            f"{proof_url}/v1/mesh/proof/commitment",
            {"receipt_context": ctx},
        )
        payload = post_json(
            f"{proof_url}/v1/mesh/proof/receipt",
            {"receipt_context": ctx, "include_proof": True},
        )
        proof_payload = payload["proof_payloads"][0]

        assert commitment["trace_commitment_count"] == 1
        assert commitment["op_manifest_count"] == 2
        assert payload["proof_receipts"][0]["op_index"] == 1
        assert proof_payload["trace"]["op_index"] == 1
        assert proof_payload["op_manifest_membership"]["op_manifest_leaf_index"] == 1
        # A manifest-selected trace outside the committed trace prefix must not
        # manufacture a synthetic singleton trace tree.  Its independently
        # committed op-manifest membership is the canonical proof domain.
        assert proof_payload.get("trace_membership") in (None, {})
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt={
                "model_package_hash": mesh_spec.model_package_hash,
                "proof_trace_scope": "op_manifest_challenge_v1",
                "proof_op_manifest_root": mesh_op_manifest_aggregate_root(
                    [commitment]
                ),
            },
        ).verified
    finally:
        _shutdown_server(proof_server, proof_thread)


def test_ggml_proof_adapter_selection_reports_missing_manifest_witness(tmp_path):
    base_ns = time.time_ns()
    _write_decode_lm_head_trace(
        tmp_path,
        token_id=101,
        created_unix_ns=base_ns,
        manifest_index=7,
        write_trace=False,
    )
    trace_dir = tmp_path / "ggml-traces"
    proof_server = make_ggml_proof_server(trace_dir=trace_dir)
    proof_thread = threading.Thread(target=proof_server.serve_forever, daemon=True)
    proof_thread.start()
    proof_host, proof_port = proof_server.server_address
    proof_url = f"http://{proof_host}:{proof_port}"
    ctx = {
        "request_id": "req-proof-adapter-selection",
        "stage_index": 1,
        "inference_started_unix_ns": base_ns - 1,
        "inference_ended_unix_ns": base_ns + 1,
        "proof_sampled": False,
        "proof_ops_per_request": 1,
        "decode_audit_required": True,
        "decode_audit_positions": [0],
        "decode_audit_completion_token_ids": [101],
    }
    try:
        commitment = post_json(
            f"{proof_url}/v1/mesh/proof/commitment",
            {"receipt_context": ctx},
        )
        selection = post_json(
            f"{proof_url}/v1/mesh/proof/selection",
            {"receipt_context": ctx},
        )

        assert commitment["trace_commitment_count"] == 0
        assert commitment["trace_commitment_root"] == ""
        assert commitment["op_manifest_count"] == 1
        assert commitment["op_manifest_root"]
        assert selection["stage_index"] == 1
        assert selection["op_manifest_root"] == commitment["op_manifest_root"]
        assert selection["op_manifest_count"] == 1
        assert selection["selected_manifest_indexes"] == [7]
        assert selection["missing_manifest_indexes"] == [7]
    finally:
        _shutdown_server(proof_server, proof_thread)


def test_ggml_manifest_bound_decode_proof_without_src0_dump(tmp_path, monkeypatch):
    trace = _write_decode_lm_head_trace(tmp_path, token_id=101)
    manifest = _gguf_manifest_for_trace(trace)
    proof_f32 = np.ascontiguousarray(
        np.fromfile(trace.src0_f32_path, dtype=np.float32).reshape(
            trace.src0_shape[1],
            trace.src0_shape[0],
        ).T
    )
    proof_i8 = proof_i8_weight_matrix_from_gguf_f32(
        np.fromfile(trace.src0_f32_path, dtype=np.float32),
        list(trace.src0_shape),
    )
    import verallm.mesh.gguf_manifest as gguf_manifest_mod

    monkeypatch.setattr(
        gguf_manifest_mod,
        "_proof_i8_weight_matrix_from_model_file",
        lambda _model_file, _tensor_name, _expected_root, _chunk_size, expected_sha256="": (
            proof_i8,
            1.0,
        ),
    )
    monkeypatch.setattr(
        gguf_manifest_mod,
        "_proof_f32_weight_matrix_from_model_file",
        lambda _model_file, _tensor_name, _expected_f32_sha256: proof_f32,
    )
    data = json.loads(trace.path.read_text(encoding="utf-8"))
    data.pop("src0_f32", None)
    data.pop("src0_raw", None)
    trace.path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    trace.src0_f32_path.unlink()
    trace.src0_raw_path.unlink()
    compact_trace = GgmlMulMatTrace.from_json(trace.path)
    manifest_entries = find_op_manifest_entries_for_window(
        tmp_path / "ggml-traces",
        start_unix_ns=0,
        end_unix_ns=time.time_ns() + 1,
    )

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx.update(
        {
            "decode_audit_required": True,
            "decode_audit_positions": [0],
            "decode_audit_completion_token_ids": [101],
            "decode_audit_top_k": 8,
        }
    )

    proof = prove_ggml_mul_mat_trace(
        compact_trace,
        ctx,
        include_proof=True,
        gguf_manifest=manifest,
        op_manifest_membership=op_manifest_membership_payload(
            manifest_entries,
            compact_trace,
            stage_index=1,
        ),
        decode_audit_positions=[0],
        decode_audit_token_ids=[101],
        decode_audit_top_k=8,
    )
    payload = proof.proof_payload
    assert "src0_f32_sha256" not in payload["trace"]
    assert "src0_raw_sha256" not in payload["trace"]
    assert verify_ggml_gemm_proof_payload(
        payload,
        receipt=proof.receipt.to_dict(),
    ).verified
    assert verify_ggml_decode_audit_payloads(
        {
            "decode_audit_positions": [0],
            "decode_audit_stage_index": 1,
            "decode_audit_top_k": 8,
        },
        [payload],
        completion_token_ids=[101],
    ).verified
    wrong_stage_payload = json.loads(json.dumps(payload))
    wrong_stage_payload["op_manifest_membership"]["stage_index"] = 0
    wrong_stage = verify_ggml_decode_audit_payloads(
        {
            "decode_audit_positions": [0],
            "decode_audit_stage_index": 1,
            "decode_audit_top_k": 8,
        },
        [wrong_stage_payload],
        completion_token_ids=[101],
    )
    assert not wrong_stage.verified
    assert "wrong mesh stage" in wrong_stage.message
    duplicate = verify_ggml_decode_audit_payloads(
        {
            "decode_audit_positions": [0],
            "decode_audit_stage_index": 1,
            "decode_audit_top_k": 8,
        },
        [payload, payload],
        completion_token_ids=[101],
    )
    assert not duplicate.verified
    assert "duplicated position" in duplicate.message


def test_ggml_proof_payload_rejects_tampered_proof(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["proof"]["block_proofs"][0]["sumcheck_proof"]["final_A"] += 1

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=proof.receipt.to_dict(),
    )

    assert result.verified is False


def test_ggml_proof_payload_rejects_empty_block_proof(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["proof"]["block_proofs"] = []
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "block proofs" in result.message


def test_ggml_proof_payload_rejects_wrong_transcript_spot_position(tmp_path):
    import struct

    from zkllm.crypto.field import mod_p

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    block = tampered["proof"]["block_proofs"][0]
    spot = block["spot_X"][0]
    opening = block["spot_X_with_proofs"][0]
    input_rows, input_cols = tampered["input_shape"]
    chunk_size = int(tampered["input_chunk_size"])
    leaf_data = bytes.fromhex(opening["leaf_data"])
    leaf_start = int(opening["merkle_path"]["leaf_index"]) * chunk_size
    old_position = (int(spot["row"]), int(spot["col"]))
    new_position = None
    for flat_idx in range(
        leaf_start,
        min(leaf_start + len(leaf_data), int(input_rows) * int(input_cols)),
    ):
        row = flat_idx // int(input_cols)
        col = flat_idx % int(input_cols)
        if (row, col) != old_position:
            value = mod_p(struct.unpack_from("<b", leaf_data, flat_idx - leaf_start)[0])
            new_position = (row, col, value)
            break
    assert new_position is not None
    for target in (spot, opening):
        target["row"] = new_position[0]
        target["col"] = new_position[1]
        target["value"] = new_position[2]
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    # A moved spot position is rejected either by the transcript replay or,
    # when a gemm-v2 sidecar rides along, by its earlier witness-binding
    # check; both mean the same forged position never verifies.
    assert (
        "transcript position" in result.message
        or "spot opening position mismatch" in result.message
    )


def test_ggml_proof_payload_rejects_missing_output_block_opening(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered.pop("output_block_openings", None)
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "output block openings" in result.message


def test_ggml_proof_payload_rejects_tampered_output_block_opening(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    opening = tampered["output_block_openings"][0]
    leaf = bytearray.fromhex(opening["leaf_data"])
    leaf[0] ^= 1
    opening["leaf_data"] = leaf.hex()
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "output block opening" in result.message


def test_ggml_proof_payload_rejects_tampered_trace_membership(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["trace_membership"]["trace_commitment_hash"] = DIGEST_A

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=proof.receipt.to_dict(),
    )

    assert result.verified is False


def test_ggml_proof_payload_rejects_tampered_trace_metadata_commitment(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["trace"]["src1_f32_sha256"] = DIGEST_B
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "trace commitment" in result.message


def test_ggml_proof_payload_rejects_model_package_hash_mismatch(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "model_package_hash": DIGEST_B,
    }

    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "model_package_hash" in result.message


def test_ggml_proof_payload_verifies_gguf_tensor_manifest_binding(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )

    assert proof.proof_payload["gguf_tensor_membership"]["tensor_manifest_root"] == (
        manifest["tensor_manifest_root"]
    )
    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    )
    assert result.verified is True


def test_ggml_manifest_prover_retries_once_after_native_failure(
    tmp_path, monkeypatch
):
    # The live intermittent: native_prove_block explodes on a draw the same
    # inputs served fine before and after. The prover must drop the weight
    # caches, rebuild verified, and retry once instead of 500ing the receipt.
    from zkllm.prover.gemm_fast import GEMMProverFast

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]

    calls = {"n": 0}
    original = GEMMProverFast.prove

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(
                "cannot create std::vector larger than max_size()"
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GEMMProverFast, "prove", flaky)
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )
    assert calls["n"] == 2
    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    )
    assert result.verified is True


def test_ggml_manifest_bound_proof_falls_back_to_exact_f32(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod
    from verallm.mesh.gguf_manifest import quantize_proof_i8

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    x_f32, exact_w_f32, y_f32 = trace.load_matrices()
    proof_i8, scale = quantize_proof_i8(exact_w_f32)
    fast_w_f32 = np.ascontiguousarray(proof_i8.astype(np.float32) * scale)
    assert float(np.max(np.abs((x_f32 @ fast_w_f32) - y_f32))) > 1e-6
    calls = []

    def load_weight(_manifest, _tensor_name, *, exact=None):
        calls.append(exact)
        return exact_w_f32 if exact is True else fast_w_f32

    monkeypatch.setattr(
        ggml_proof_mod,
        "proof_f32_weight_matrix_from_manifest",
        load_weight,
    )
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]

    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        tolerance_rel=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )

    assert calls == [None, True]
    assert proof.proof_payload["float_exact_fallback_used"] is True
    assert proof.float_max_abs_error == 0.0
    assert verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    ).verified


def test_ggml_manifest_bound_proof_accepts_logical_metal_io_layout(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    x_f32, _w_f32, y_f32 = trace.load_matrices()
    np.ascontiguousarray(x_f32.T).tofile(trace.src1_f32_path)
    np.ascontiguousarray(y_f32.T).tofile(trace.dst_f32_path)
    meta = json.loads(trace.path.read_text(encoding="utf-8"))
    meta["backend"] = "llama_cpp_metal"
    trace.path.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    trace = GgmlMulMatTrace.from_json(trace.path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]

    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )

    assert proof.proof_payload["trace_io_layout"] == "logical_ne0_ne1"
    assert verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    ).verified


def test_ggml_manifest_bound_proof_accepts_bias_output_transform(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    _x_f32, _w_f32, y_f32 = trace.load_matrices()
    bias = np.array([0.25, -0.5, 1.0, -1.5], dtype=np.float32)
    np.ascontiguousarray(y_f32 + bias).tofile(trace.dst_f32_path)
    manifest = _gguf_manifest_for_trace(trace)
    bias_path = trace.path.parent / "blk.0.test.bias.f32"
    bias.tofile(bias_path)
    bias_record = {
        "name": "blk.0.test.bias",
        "tensor_type": "F32",
        "shape": [4],
        "n_elements": 4,
        "n_bytes": bias_path.stat().st_size,
        "data_offset": 0,
        "raw_sha256": hashlib.sha256(bias_path.read_bytes()).hexdigest(),
        "raw_path": str(bias_path),
        "f32_nbytes": bias_path.stat().st_size,
        "f32_sha256": hashlib.sha256(bias_path.read_bytes()).hexdigest(),
        "f32_path": str(bias_path),
    }
    manifest["tensors"].append(bias_record)
    ordered = sorted(
        manifest["tensors"],
        key=lambda item: (
            str(item.get("name", "")),
            int(item.get("model_file_index", 0)),
            int(item.get("data_offset", 0)),
        ),
    )
    manifest["tensors"] = ordered
    manifest["tensor_count"] = len(ordered)
    manifest["tensor_manifest_root"] = MerkleTree(
        [tensor_leaf_bytes(record) for record in ordered]
    ).root.hex()
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]

    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )

    assert proof.proof_payload["trace_output_transform"] == "matmul_plus_bias"
    assert proof.proof_payload["trace_output_bias_tensor_name"] == "blk.0.test.bias"
    assert verify_ggml_gemm_proof_payload(
        proof.proof_payload,
        receipt=proof.receipt.to_dict(),
    ).verified


def test_ggml_proof_generation_requires_gguf_manifest_for_tensor_root(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]

    with pytest.raises(RuntimeError, match="requires gguf_manifest"):
        prove_ggml_mul_mat_trace(
            trace,
            ctx,
            tolerance_abs=1e-6,
            include_proof=True,
        )


def test_ggml_proof_generation_rejects_gguf_manifest_root_mismatch(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = DIGEST_B

    with pytest.raises(RuntimeError, match="root does not match"):
        prove_ggml_mul_mat_trace(
            trace,
            ctx,
            tolerance_abs=1e-6,
            include_proof=True,
            gguf_manifest=manifest,
        )


def test_ggml_proof_payload_rejects_missing_gguf_tensor_membership(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered.pop("gguf_tensor_membership", None)
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "GGUF tensor membership" in result.message


def test_ggml_proof_payload_rejects_tampered_gguf_f32_witness_hash(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx["model_tensor_manifest_root"] = manifest["tensor_manifest_root"]
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["trace"]["src0_f32_sha256"] = DIGEST_B
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)
    tampered_receipt = {
        **proof.receipt.to_dict(),
        "proof_commitment_hash": tampered["proof_commitment_hash"],
    }

    result = verify_ggml_gemm_proof_payload(
        tampered,
        receipt=tampered_receipt,
    )

    assert result.verified is False
    assert "trace commitment" in result.message


def test_ggml_proof_payload_rejects_missing_gguf_tensor_membership_without_receipt(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        tolerance_abs=1e-6,
        include_proof=True,
        gguf_manifest=manifest,
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered.pop("gguf_tensor_membership", None)
    tampered["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(tampered)

    result = verify_ggml_gemm_proof_payload(tampered)

    assert result.verified is False
    assert "GGUF tensor membership" in result.message


def test_ggml_op_manifest_membership_verifies_selected_trace(tmp_path):
    trace = _write_ggml_mul_mat_trace(tmp_path, op_index=7, manifest_index=7)
    entries = find_op_manifest_entries_for_window(trace.path.parent)
    membership = op_manifest_membership_payload(entries, trace, stage_index=1)

    assert isinstance(entries[0], GgmlOpManifestEntry)
    assert ggml_op_manifest_root(entries) == membership["op_manifest_root"]
    assert verify_op_manifest_membership(
        entry_hash=membership["op_manifest_entry_hash"],
        op_manifest_root=membership["op_manifest_root"],
        op_manifest_count=membership["op_manifest_count"],
        leaf_index=membership["op_manifest_leaf_index"],
        path=membership["op_manifest_membership_path"],
    )


def test_ggml_compact_op_manifest_matches_json_commitment(tmp_path):
    trace = _write_ggml_mul_mat_trace(tmp_path, op_index=7, manifest_index=7)
    json_entries = find_op_manifest_entries_for_window(trace.path.parent)
    assert len(json_entries) == 1
    entry = json_entries[0]

    def as_hex(value: str) -> str:
        return value.encode("utf-8").hex()

    compact_line = "\t".join(
        [
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_V1",
            str(entry.created_unix_ns),
            str(entry.manifest_index),
            as_hex(entry.graph_id),
            str(entry.op_index),
            as_hex(entry.tensor_name),
            as_hex(entry.src0_name),
            as_hex(entry.src1_name),
            as_hex(entry.dst_name),
            ",".join(str(item) for item in entry.src0_shape),
            ",".join(str(item) for item in entry.src1_shape),
            ",".join(str(item) for item in entry.dst_shape),
            entry.source_types["src0"],
            entry.source_types["src1"],
            entry.source_types["dst"],
            as_hex(entry.backend),
            as_hex(entry.device),
            "1",
        ]
    )
    (trace.path.parent / "manifest.jsonl").unlink()
    (trace.path.parent / f"manifest-{entry.created_unix_ns}.vmanifest").write_text(
        compact_line + "\n",
        encoding="utf-8",
    )

    compact_entries = find_op_manifest_entries_for_window(trace.path.parent)
    assert [item.to_commitment_body() for item in compact_entries] == [
        item.to_commitment_body() for item in json_entries
    ]
    assert ggml_op_manifest_root(compact_entries) == ggml_op_manifest_root(json_entries)
    assert ggml_op_manifest_summary_for_window(trace.path.parent) == (
        ggml_op_manifest_root(json_entries),
        1,
    )
    assert op_manifest_membership_payload(compact_entries, trace, stage_index=1)

    raw_line = "\t".join(
        [
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V1",
            str(entry.created_unix_ns),
            str(entry.manifest_index),
            entry.graph_id,
            str(entry.op_index),
            entry.tensor_name,
            entry.src0_name,
            entry.src1_name,
            entry.dst_name,
            ",".join(str(item) for item in entry.src0_shape),
            ",".join(str(item) for item in entry.src1_shape),
            ",".join(str(item) for item in entry.dst_shape),
            entry.source_types["src0"],
            entry.source_types["src1"],
            entry.source_types["dst"],
            entry.backend,
            entry.device,
            "1",
        ]
    )
    (trace.path.parent / f"manifest-{entry.created_unix_ns}.vmanifest").write_text(
        raw_line + "\n",
        encoding="utf-8",
    )
    raw_entries = find_op_manifest_entries_for_window(trace.path.parent)
    assert [item.to_commitment_body() for item in raw_entries] == [
        item.to_commitment_body() for item in json_entries
    ]
    assert ggml_op_manifest_root(raw_entries) == ggml_op_manifest_root(json_entries)
    assert ggml_op_manifest_summary_for_window(trace.path.parent) == (
        ggml_op_manifest_root(json_entries),
        1,
    )

    v2_expected = GgmlOpManifestEntry(
        path=trace.path.parent / f"manifest-{entry.created_unix_ns}.vmanifest",
        created_unix_ns=entry.created_unix_ns,
        manifest_index=entry.manifest_index,
        graph_id=f"metal-mul-mat-{entry.manifest_index}",
        op_index=entry.manifest_index,
        op_type=entry.op_type,
        tensor_name=entry.tensor_name,
        src0_name=entry.tensor_name,
        src1_name=entry.src1_name,
        dst_name=entry.dst_name,
        src0_shape=entry.src0_shape,
        src1_shape=entry.src1_shape,
        dst_shape=entry.dst_shape,
        source_types=entry.source_types,
        backend="llama_cpp_metal",
        device="MTL0",
        proof_eligible=True,
    )
    v2_line = "\t".join(
        [
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V2",
            str(entry.created_unix_ns),
            str(entry.manifest_index),
            entry.tensor_name,
            entry.src1_name,
            entry.dst_name,
            ",".join(str(item) for item in entry.src0_shape),
            ",".join(str(item) for item in entry.src1_shape),
            ",".join(str(item) for item in entry.dst_shape),
            entry.source_types["src0"],
            entry.source_types["src1"],
            entry.source_types["dst"],
            "llama_cpp_metal",
            "MTL0",
            "1",
        ]
    )
    (trace.path.parent / f"manifest-{entry.created_unix_ns}.vmanifest").write_text(
        v2_line + "\n",
        encoding="utf-8",
    )
    v2_entries = find_op_manifest_entries_for_window(trace.path.parent)
    assert [item.to_commitment_body() for item in v2_entries] == [
        v2_expected.to_commitment_body()
    ]
    assert ggml_op_manifest_summary_for_window(trace.path.parent) == (
        ggml_op_manifest_root([v2_expected]),
        1,
    )


def test_ggml_proof_payload_rejects_tampered_op_manifest_membership(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    entries = find_op_manifest_entries_for_window(trace.path.parent)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
        op_manifest_membership=op_manifest_membership_payload(
            entries,
            trace,
            stage_index=1,
        ),
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["op_manifest_membership"]["entry"]["tensor_name"] = "other.weight"

    result = verify_ggml_gemm_proof_payload(tampered)

    assert result.verified is False
    assert "op manifest" in result.message


def test_ggml_proof_payload_rejects_tampered_source_tensor_name(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    entries = find_op_manifest_entries_for_window(trace.path.parent)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
        op_manifest_membership=op_manifest_membership_payload(
            entries,
            trace,
            stage_index=1,
        ),
    )
    tampered = json.loads(json.dumps(proof.proof_payload))
    tampered["trace"]["src0_name"] = "blk.0.other.weight"

    result = verify_ggml_gemm_proof_payload(tampered)

    assert result.verified is False
    assert "trace commitment" in result.message


def test_ggml_proof_payloads_reject_tampered_op_manifest_aggregate(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    entries = find_op_manifest_entries_for_window(trace.path.parent)
    membership = op_manifest_membership_payload(entries, trace, stage_index=1)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
        op_manifest_membership=membership,
    )
    aggregate = mesh_op_manifest_aggregate_root(
        [
            {
                "stage_index": 1,
                "op_manifest_root": membership["op_manifest_root"],
                "op_manifest_count": membership["op_manifest_count"],
            }
        ]
    )

    assert verify_ggml_gemm_proof_payloads(
        [proof.proof_payload],
        [proof.receipt.to_dict()],
        mesh_receipt={"proof_op_manifest_root": aggregate},
    ).verified

    result = verify_ggml_gemm_proof_payloads(
        [proof.proof_payload],
        [proof.receipt.to_dict()],
        mesh_receipt={"proof_op_manifest_root": DIGEST_A},
    )

    assert result.verified is False
    assert "op manifest aggregate mismatch" in result.message


def test_ggml_proof_payloads_reject_model_package_hash_mesh_mismatch(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )

    result = verify_ggml_gemm_proof_payloads(
        [proof.proof_payload],
        [proof.receipt.to_dict()],
        mesh_receipt={"model_package_hash": DIGEST_B},
    )

    assert result.verified is False
    assert "model_package_hash" in result.message


def test_ggml_proof_payloads_reject_tampered_trace_aggregate(tmp_path):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )

    result = verify_ggml_gemm_proof_payloads(
        [proof.proof_payload],
        [proof.receipt.to_dict()],
        mesh_receipt={"proof_trace_commitment_root": DIGEST_A},
    )

    assert result.verified is False
    assert "aggregate mismatch" in result.message


def test_inference_commitment_binds_mesh_fields():
    base = InferenceCommitment(
        session_id="s",
        model_id="m",
        model_commitment=bytes.fromhex(DIGEST_A),
        input_commitment=bytes.fromhex(DIGEST_B),
        output_commitment=bytes.fromhex(DIGEST_C),
        layer_commitments=[],
        timestamp=1.0,
    )
    staged = InferenceCommitment(
        session_id="s",
        model_id="m",
        model_commitment=bytes.fromhex(DIGEST_A),
        input_commitment=bytes.fromhex(DIGEST_B),
        output_commitment=bytes.fromhex(DIGEST_C),
        layer_commitments=[],
        mesh_spec_hash=bytes.fromhex(DIGEST_A),
        stage_assignment_hash=bytes.fromhex(DIGEST_B),
        stage_boundary_roots=[bytes.fromhex(DIGEST_C)],
        stage_receipt_root=bytes.fromhex(DIGEST_D),
        timestamp=1.0,
    )

    assert base.commitment_hash() != staged.commitment_hash()
    assert b"MESH_V1" in staged.to_bytes()


def test_inference_commitment_binds_mesh_proof_receipt_root():
    base = InferenceCommitment(
        session_id="s",
        model_id="m",
        model_commitment=bytes.fromhex(DIGEST_A),
        input_commitment=bytes.fromhex(DIGEST_B),
        output_commitment=bytes.fromhex(DIGEST_C),
        layer_commitments=[],
        mesh_spec_hash=bytes.fromhex(DIGEST_A),
        stage_assignment_hash=bytes.fromhex(DIGEST_B),
        stage_receipt_root=bytes.fromhex(DIGEST_C),
        timestamp=1.0,
    )
    with_proof_root = InferenceCommitment(
        session_id="s",
        model_id="m",
        model_commitment=bytes.fromhex(DIGEST_A),
        input_commitment=bytes.fromhex(DIGEST_B),
        output_commitment=bytes.fromhex(DIGEST_C),
        layer_commitments=[],
        mesh_spec_hash=bytes.fromhex(DIGEST_A),
        stage_assignment_hash=bytes.fromhex(DIGEST_B),
        stage_receipt_root=bytes.fromhex(DIGEST_C),
        mesh_proof_receipt_root=bytes.fromhex(DIGEST_D),
        timestamp=1.0,
    )

    assert base.commitment_hash() != with_proof_root.commitment_hash()
    assert b"MESH_PROOF_RECEIPT_ROOT_V1" in with_proof_root.to_bytes()


def test_cli_style_mesh_json_is_stable(tmp_path):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="https://coord.example.com",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=2,
    )
    path = save_json(tmp_path / "mesh.json", spec.to_dict())
    data = json.loads(path.read_text())

    assert data["members"][0]["layers"] == {"end": 2, "start": 0}
    assert data["activation_dtype"] == "f16"


def test_cli_status_includes_rpc_plan(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    path = save_json(tmp_path / "mesh.json", spec.to_dict())

    mesh_cli_main(["status", str(path)])
    payload = json.loads(capsys.readouterr().out)

    assert payload["rpc_plan"]["rpc_arg"] == "worker.local:50052"
    assert payload["rpc_plan"]["rpc_plan_hash"] == rpc_plan_from_mesh(spec).plan_hash_hex()


def test_cli_assign_rewrites_stage_members(tmp_path):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://rtx.local:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        max_context_len=32_768,
        total_layers=4,
    )
    src = save_json(tmp_path / "mesh.json", spec.to_dict())
    out = tmp_path / "mesh-split.json"

    mesh_cli_main(
        [
            "assign",
            str(src),
            "--member",
            "1,5Coord,http://rtx.local:9338,0:2,gguf_stage,10000,coordinator",
            "--member",
            "2,5Mac,http://mac.local:9338,2:4,gguf_stage_worker,0,worker",
            "--output",
            str(out),
        ]
    )

    split = load_mesh_spec(out)
    assert len(split.members) == 2
    assert split.members[0].layers == StageRange(0, 2)
    assert split.members[1].layers == StageRange(2, 4)
    assert split.max_context_len == 32_768
    assert split.stage_assignment_hash() != spec.stage_assignment_hash()


def test_worker_probe_round_trip():
    capability = CapabilityAd(
        uid=11,
        hotkey="5Worker",
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    server, thread = serve_worker_in_thread(capability=capability)
    host, port = server.server_address
    try:
        probe = probe_worker(f"http://{host}:{port}")
        assert probe.status == "ok"
        assert probe.service == "verathos-mesh-worker"
        assert probe.capability.uid == 11
        assert probe.capability.ad_hash_hex() == capability.ad_hash_hex()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_handshake_binds_mesh_hash():
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=2,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[DIGEST_A],
    )
    server, thread = serve_worker_in_thread(capability=capability, mesh_spec=mesh_spec)
    host, port = server.server_address
    handshake_url = f"http://{host}:{port}/v1/stage/handshake"
    try:
        response = post_json(handshake_url, {"mesh_spec_hash": mesh_spec.spec_hash_hex()})
        assert response["status"] == "accepted"
        assert response["mesh_spec_hash"] == mesh_spec.spec_hash_hex()

        with pytest.raises(RuntimeError, match="409"):
            post_json(handshake_url, {"mesh_spec_hash": DIGEST_B})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_join_token_admits_worker_and_reassigns_layers(tmp_path):
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        model_tensor_manifest_root=DIGEST_C,
        max_context_len=32_768,
        total_layers=4,
    )
    state_dir, state, token = create_mesh_state(spec=mesh_spec, root=tmp_path / "coord")
    coordinator_capability = CapabilityAd.from_dict(state["capabilities"][0])
    join_secret = state["join_secret"]

    def join_handler(body):
        if body.get("join_secret") != join_secret:
            raise PermissionError("invalid join token")
        updated = admit_mesh_worker(state_dir, CapabilityAd.from_dict(body["capability"]))
        return {
            "status": "joined",
            "mesh": updated.to_dict(),
            "mesh_spec_hash": updated.spec_hash_hex(),
            "stage_assignment_hash": updated.stage_assignment_hash_hex(),
        }

    server, thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        join_handler=join_handler,
    )
    host, port = server.server_address
    runtime_token = token.__class__(
        mesh_id=token.mesh_id,
        coordinator_endpoint=f"http://{host}:{port}",
        join_secret=token.join_secret,
        coordinator_uid=token.coordinator_uid,
        coordinator_hotkey=token.coordinator_hotkey,
        model_id=token.model_id,
    ).encode()
    try:
        worker_dir, joined = join_mesh(
            token=runtime_token,
            endpoint="http://worker.local:9338",
            root=tmp_path / "worker",
            package_hash=DIGEST_A,
            gpu_name="Test GPU",
            vram_gb=24,
            rpc_endpoint="worker.local:50052",
        )
        assert len(joined.members) == 2
        assert joined.model_tensor_manifest_root == DIGEST_C
        assert joined.max_context_len == 32_768
        assert joined.members[0].layers == StageRange(0, 3)
        assert joined.members[1].layers == StageRange(3, 4)
        assert joined.members[1].rpc_endpoint == "worker.local:50052"
        assert joined.members[1].proof_endpoint == "http://worker.local:9338"
        assert "llama_cpp_rpc" in CapabilityAd.from_dict(
            load_mesh_state(worker_dir)["capabilities"][0]
        ).supported_backends
        persisted_capability = CapabilityAd.from_dict(
            load_mesh_state(worker_dir)["capabilities"][0]
        )
        assert persisted_capability.proof_endpoint == "http://worker.local:9338"
        assert persisted_capability.gpu_name == "Test GPU"
        assert persisted_capability.vram_gb == 24
        assert state_mesh_spec(load_mesh_state(state_dir)).mesh_id == joined.mesh_id
        assert state_mesh_spec(load_mesh_state(worker_dir)).mesh_id == joined.mesh_id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_join_mesh_retries_and_fails_closed_on_member_update_errors(
    tmp_path,
    monkeypatch,
):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://coordinator.local:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
    )
    _state_dir, _state, token = create_mesh_state(
        spec=spec,
        root=tmp_path / "coord",
    )
    calls = []

    def failed_join(*_args, **_kwargs):
        calls.append(1)
        return {
            "mesh": spec.to_dict(),
            "mesh_update_errors": [
                {
                    "endpoint": "http://existing.private:9338",
                    "error": "temporarily unavailable",
                }
            ],
        }

    monkeypatch.setattr("verallm.mesh.state.post_json", failed_join)
    with pytest.raises(
        RuntimeError,
        match="could not update 1 existing mesh member",
    ):
        join_mesh(
            token=token.encode(),
            endpoint="http://new.private:9338",
            root=tmp_path / "worker",
            rpc_endpoint="new.private:50052",
        )
    assert len(calls) == 3
    assert not (tmp_path / "worker" / spec.mesh_id).exists()


def test_join_mesh_dials_loopback_when_coordinator_is_this_host(
    tmp_path,
    monkeypatch,
):
    """A driver joining its own coordinator must dial loopback: many
    provider NATs cannot hairpin a box to its own public IP, so the dial
    to the advertised endpoint times out while every other machine
    reaches it fine . The spec keeps the
    advertised address; only the dial URL changes, and only when the
    coordinator host IS this worker's advertise host."""
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://192.0.0.9:20003",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
    )
    _state_dir, _state, token = create_mesh_state(
        spec=spec,
        root=tmp_path / "coord",
    )
    dialed = []

    def record_join(url, *_args, **_kwargs):
        dialed.append(url)
        return {"mesh": spec.to_dict(), "mesh_update_errors": []}

    monkeypatch.setattr("verallm.mesh.state.post_json", record_join)
    join_mesh(
        token=token.encode(),
        endpoint="http://192.0.0.9:20002",
        root=tmp_path / "worker",
        self_advertise_host="192.0.0.9",
    )
    assert dialed == ["http://127.0.0.1:20003/v1/mesh/join"]

    # A remote member (different advertise host) keeps the advertised dial.
    dialed.clear()
    join_mesh(
        token=token.encode(),
        endpoint="http://10.9.9.9:20002",
        root=tmp_path / "worker2",
        self_advertise_host="10.9.9.9",
    )
    assert dialed == ["http://192.0.0.9:20003/v1/mesh/join"]


def test_assign_mesh_members_stageless_coordinator(tmp_path):
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        max_context_len=32_768,
        total_layers=4,
    )
    state_dir, state, _token = create_mesh_state(spec=mesh_spec, root=tmp_path / "coord")
    capabilities = [CapabilityAd.from_dict(item) for item in state["capabilities"]]
    capabilities.append(
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://w1.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            rpc_endpoint="w1.local:50052",
            proof_endpoint="http://w1.local:9338",
        )
    )
    capabilities.append(
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://w2.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            rpc_endpoint="w2.local:50052",
            proof_endpoint="http://w2.local:9338",
        )
    )

    # Orchestration-only coordinator: empty range, workers tile every layer.
    updated = assign_mesh_members(mesh_spec, capabilities, coordinator_computes=False)
    assert updated.members[0].role == "coordinator"
    assert updated.members[0].layers == StageRange(0, 0)
    assert updated.members[1].layers == StageRange(0, 3)
    assert updated.members[2].layers == StageRange(3, 4)
    assert updated.max_context_len == 32_768
    round_tripped = MeshSpec.from_dict(updated.to_dict())
    assert round_tripped.members[0].layers == StageRange(0, 0)
    assert round_tripped.max_context_len == 32_768

    # Including the output layer in llama placement leaves the last device
    # with no repeating block for this tiny 4-layer/3-device topology, so a
    # proof-stage assignment must fail closed.
    with pytest.raises(ValueError, match="zero transformer layers"):
        assign_mesh_members(mesh_spec, capabilities)

    # A stage-less non-coordinator is rejected.
    broken = updated.to_dict()
    broken["members"][1]["layers"] = {"start": 0, "end": 0}
    broken["members"][2]["layers"] = {"start": 0, "end": 4}
    with pytest.raises(ValueError):
        MeshSpec.from_dict(broken)


def test_assign_mesh_members_uses_ordered_vram_weights_and_persists_updates(
    tmp_path,
):
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=40,
    )
    state_dir, state, _token = create_mesh_state(
        spec=mesh_spec,
        root=tmp_path / "coord",
    )
    state["coordinator_computes"] = False
    save_mesh_state(state_dir, state)

    first = admit_mesh_worker(
        state_dir,
        CapabilityAd(
            uid=1,
            hotkey="5Worker1",
            endpoint="http://w1.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            gpu_name="RTX 4090",
            vram_gb=24,
            rpc_endpoint="w1.local:50052",
            proof_endpoint="http://w1.local:9338",
        ),
    )
    assert first.members[0].layers == StageRange(0, 0)
    assert first.members[1].layers == StageRange(0, 40)

    weighted = admit_mesh_worker(
        state_dir,
        CapabilityAd(
            uid=1,
            hotkey="5Worker2",
            endpoint="http://w2.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            gpu_name="M1 Max",
            vram_gb=49,
            rpc_endpoint="w2.local:50052",
            proof_endpoint="http://w2.local:9338",
        ),
    )
    assert [member.layers for member in weighted.members] == [
        StageRange(0, 0),
        StageRange(0, 14),
        StageRange(14, 40),
    ]

    persisted = load_mesh_state(state_dir)
    assert [
        capability.vram_gb
        for capability in state_capabilities(persisted)
    ] == [0, 24, 49]
    assert MeshSpec.from_dict(persisted["mesh"]).to_dict() == weighted.to_dict()


def test_assign_mesh_members_falls_back_to_equal_ranges_for_missing_vram():
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=40,
    )
    capabilities = [
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://127.0.0.1:9338",
            supported_backends=["gguf_stage"],
        ),
        CapabilityAd(
            uid=1,
            hotkey="5Worker1",
            endpoint="http://w1.local:9338",
            supported_backends=["gguf_stage_worker"],
            vram_gb=24,
        ),
        CapabilityAd(
            uid=1,
            hotkey="5Worker2",
            endpoint="http://w2.local:9338",
            supported_backends=["gguf_stage_worker"],
            vram_gb=0,
        ),
    ]

    fallback = assign_mesh_members(
        mesh_spec,
        capabilities,
        coordinator_computes=False,
    )
    assert [member.layers for member in fallback.members] == [
        StageRange(0, 0),
        StageRange(0, 21),
        StageRange(21, 40),
    ]


def test_weighted_assignment_rejects_zero_repeating_layer_stages():
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=3,
    )
    capabilities = [
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://127.0.0.1:9338",
            supported_backends=["gguf_stage"],
        )
    ] + [
        CapabilityAd(
            uid=1,
            hotkey=f"5Worker{index}",
            endpoint=f"http://w{index}.local:9338",
            supported_backends=["gguf_stage_worker"],
            vram_gb=weight,
        )
        for index, weight in enumerate((1000, 1, 1), start=1)
    ]

    with pytest.raises(ValueError, match="zero transformer layers"):
        assign_mesh_members(
            mesh_spec,
            capabilities,
            coordinator_computes=False,
        )


def test_admit_mesh_worker_honors_coordinator_computes_flag(tmp_path):
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
    )
    state_dir, state, _token = create_mesh_state(spec=mesh_spec, root=tmp_path / "coord")
    state["coordinator_computes"] = False
    save_mesh_state(state_dir, state)
    assert state_admitted_compute_stage_count(load_mesh_state(state_dir)) == 0
    updated = admit_mesh_worker(
        state_dir,
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://w1.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            rpc_endpoint="w1.local:50052",
            proof_endpoint="http://w1.local:9338",
        ),
    )
    assert updated.members[0].layers == StageRange(0, 0)
    assert updated.members[1].layers == StageRange(0, 4)
    assert state_admitted_compute_stage_count(load_mesh_state(state_dir)) == 1


def test_mesh_spec_pins_proof_trace_manifest_format(tmp_path):
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
    )
    # Unpinned specs serialize without the key so their hashes stay stable.
    assert "proof_trace_manifest_format" not in mesh_spec.to_dict()

    pinned = MeshSpec.from_dict(
        {**mesh_spec.to_dict(), "proof_trace_manifest_format": "compact-raw-v2"}
    )
    assert pinned.proof_trace_manifest_format == "compact-raw-v2"
    assert pinned.to_dict()["proof_trace_manifest_format"] == "compact-raw-v2"

    # The pin survives worker admission reassignment.
    state_dir, state, _token = create_mesh_state(spec=pinned, root=tmp_path / "coord")
    updated = admit_mesh_worker(
        state_dir,
        CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://w1.local:9338",
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            proof_modes=["verathos-gemv1"],
            rpc_endpoint="w1.local:50052",
            proof_endpoint="http://w1.local:9338",
        ),
    )
    assert updated.proof_trace_manifest_format == "compact-raw-v2"

    with pytest.raises(ValueError):
        MeshSpec.from_dict(
            {**mesh_spec.to_dict(), "proof_trace_manifest_format": "bogus"}
        )


def test_worker_mesh_state_refresh_and_update_endpoint_persist_latest_spec(tmp_path):
    coord_port = _free_port()
    worker_port = _free_port()
    coordinator_endpoint = f"http://127.0.0.1:{coord_port}"
    worker_endpoint = f"http://127.0.0.1:{worker_port}"
    mesh_spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint=coordinator_endpoint,
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=8,
    )
    state_dir, state, token = create_mesh_state(spec=mesh_spec, root=tmp_path / "coord")
    coordinator_capability = CapabilityAd.from_dict(state["capabilities"][0])

    def mesh_spec_handler(body):
        if body.get("join_secret") != token.join_secret:
            raise PermissionError("invalid join token")
        current = state_mesh_spec(load_mesh_state(state_dir))
        return {
            "status": "ok",
            "mesh": current.to_dict(),
            "mesh_spec_hash": current.spec_hash_hex(),
            "stage_assignment_hash": current.stage_assignment_hash_hex(),
        }

    def join_handler(body):
        if body.get("join_secret") != token.join_secret:
            raise PermissionError("invalid join token")
        updated = admit_mesh_worker(state_dir, CapabilityAd.from_dict(body["capability"]))
        return {
            "status": "joined",
            "mesh": updated.to_dict(),
            "mesh_spec_hash": updated.spec_hash_hex(),
            "stage_assignment_hash": updated.stage_assignment_hash_hex(),
        }

    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        host="127.0.0.1",
        port=coord_port,
        mesh_spec_loader=lambda: state_mesh_spec(load_mesh_state(state_dir)),
        join_handler=join_handler,
        mesh_spec_handler=mesh_spec_handler,
    )
    worker = None
    worker_thread = None
    try:
        worker_dir, joined = join_mesh(
            token=token.encode(),
            endpoint=worker_endpoint,
            root=tmp_path / "worker-a",
            package_hash=DIGEST_A,
            rpc_endpoint="127.0.0.1:50052",
        )
        assert len(joined.members) == 2

        updated = admit_mesh_worker(
            state_dir,
            CapabilityAd(
                uid=1,
                hotkey="5Coord",
                endpoint="http://worker-b.local:9338",
                supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
                cached_model_package_hashes=[DIGEST_A],
                rpc_endpoint="worker-b.local:50052",
                proof_endpoint="http://worker-b.local:9338",
            ),
        )
        assert len(state_mesh_spec(load_mesh_state(worker_dir)).members) == 2

        refreshed = refresh_worker_mesh_state(worker_dir)
        assert refreshed.stage_assignment_hash() == updated.stage_assignment_hash()
        assert len(state_mesh_spec(load_mesh_state(worker_dir)).members) == 3

        worker_capability = CapabilityAd.from_dict(
            load_mesh_state(worker_dir)["capabilities"][0]
        )

        def mesh_update_handler(body):
            if body.get("join_secret") != token.join_secret:
                raise PermissionError("invalid join token")
            spec = MeshSpec.from_dict(body["mesh"])
            saved = update_worker_mesh_state(worker_dir, spec)
            return {
                "status": "updated",
                "mesh_id": saved.mesh_id,
                "mesh_spec_hash": saved.spec_hash_hex(),
                "stage_assignment_hash": saved.stage_assignment_hash_hex(),
            }

        worker, worker_thread = serve_worker_in_thread(
            capability=worker_capability,
            host="127.0.0.1",
            port=worker_port,
            mesh_spec_loader=lambda: state_mesh_spec(load_mesh_state(worker_dir)),
            mesh_update_handler=mesh_update_handler,
        )

        pushed = admit_mesh_worker(
            state_dir,
            CapabilityAd(
                uid=1,
                hotkey="5Coord",
                endpoint="http://worker-c.local:9338",
                supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
                cached_model_package_hashes=[DIGEST_A],
                rpc_endpoint="worker-c.local:50052",
                proof_endpoint="http://worker-c.local:9338",
            ),
        )
        response = post_json(
            f"{worker_endpoint}/v1/mesh/update",
            {
                "join_secret": token.join_secret,
                "mesh_id": pushed.mesh_id,
                "mesh_spec_hash": pushed.spec_hash_hex(),
                "stage_assignment_hash": pushed.stage_assignment_hash_hex(),
                "mesh": pushed.to_dict(),
            },
        )
        assert response["status"] == "updated"
        assert _wait_for_mesh_member_count(worker_dir, 4).stage_assignment_hash() == (
            pushed.stage_assignment_hash()
        )
    finally:
        if worker is not None and worker_thread is not None:
            _shutdown_server(worker, worker_thread)
        _shutdown_server(coordinator, coordinator_thread)


def test_mesh_worker_forwards_openai_request_and_returns_receipt():
    backend, backend_thread, backend_url, calls = _fake_openai_backend()
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-test",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
            },
        )
        response = payload["response"]
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [request])
        assert response["choices"][0]["message"]["content"] == "pong"
        assert receipt["request_id"] == "req-test"
        assert receipt["mesh_id"] == mesh_spec.mesh_id
        assert receipt["mesh_spec_hash"] == mesh_spec.spec_hash_hex()
        assert receipt["stage_assignment_hash"] == mesh_spec.stage_assignment_hash_hex()
        assert receipt["stage_index"] == 1
        assert receipt["layer_start"] == 2
        assert receipt["layer_end"] == 4
        assert receipt["runtime"] == "llama_cpp_rpc"
        assert receipt["rpc_endpoints"] == []
        assert "worker.local:50052" not in json.dumps(payload)
        assert receipt["rpc_plan_hash"] == rpc_plan_from_mesh(mesh_spec).plan_hash_hex()
        assert receipt["request_hash"] == _payload_hash(request)
        assert receipt["response_hash"] == _payload_hash(response)
        assert (
            receipt["mesh_response_commitment_hash"]
            == _mesh_response_commitment_hash(receipt)
        )
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
        assert receipt["proof_mode"] == "llama_cpp_rpc_receipt_v1"
        assert receipt["verified"] is False
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
        )
        with pytest.raises(RuntimeError, match="cryptographic proof"):
            verify_mesh_inference_artifact(
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
                require_cryptographic_proof=True,
            )

        with pytest.raises(RuntimeError, match="409"):
            post_json(
                f"http://{host}:{port}/v1/mesh/inference",
                {
                    "request_id": "req-test-bad",
                    "mesh_spec_hash": DIGEST_B,
                    "openai_request": request,
                },
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_requires_bound_proof_receipts_when_requested():
    backend, backend_thread, backend_url, calls = _fake_openai_backend()
    proof, proof_thread, proof_url, proof_calls = _fake_proof_adapter()
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        proof_url=proof_url,
        require_proof=True,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-proof",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_receipts = [
            LlamaGraphOpReceipt.from_dict(item) for item in payload["proof_receipts"]
        ]

        assert _calls_match_with_derived_seed(calls, [request])
        assert "__verbose" not in payload["response"]
        assert len(proof_calls) == 1
        assert receipt["request_id"] == "req-proof"
        assert "verified_sampler_mode" not in receipt
        assert "verified_sampler_controls_hash" not in receipt
        assert "prompt_token_count" not in receipt
        assert "prompt_token_source" not in receipt
        assert "prompt_token_ids" not in payload
        assert receipt["completion_token_count"] == 1
        assert "completion_token_source" not in receipt
        assert payload["completion_token_ids"] == []
        assert receipt["proof_required"] is True
        assert receipt["proof_mode"] == VERATHOS_GGML_TRACE_PROOF_MODE
        assert receipt["proof_receipt_verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert receipt["proof_receipt_root"] == llama_graph_receipt_root(
            proof_receipts
        ).hex()
        assert (
            receipt["mesh_response_commitment_hash"]
            == _mesh_response_commitment_hash(receipt)
        )
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
        assert receipt["verified"] is False
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(proof, proof_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_streams_chunks_and_final_proof_metadata():
    backend, backend_thread, backend_url, calls = _fake_openai_backend(content="mesh stream pong")
    proof, proof_thread, proof_url, proof_calls = _fake_proof_adapter()
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        proof_url=proof_url,
        require_proof=True,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    try:
        events = _post_sse_events(f"http://{host}:{port}/v1/chat/completions", request)
        chunk_events = [
            payload
            for event, payload in events
            if event == "message" and isinstance(payload, dict)
        ]
        done_events = [
            payload
            for event, payload in events
            if event == "done" and isinstance(payload, dict)
        ]

        # Proof-capturing requests without a client seed get a deterministic
        # replay seed derived from the request id so deferred-audit replays
        # can reproduce the completion.
        assert len(calls) == 1
        backend_call = dict(calls[0])
        derived_seed = backend_call.pop("seed", None)
        # Client sent no completion bound: the serve-side default cap applies.
        assert backend_call.pop("max_tokens", None) == 4096
        assert backend_call == request
        assert isinstance(derived_seed, int) and derived_seed >= 1
        assert len(proof_calls) == 1
        assert len(chunk_events) >= 2
        assert done_events
        done = done_events[-1]
        mesh = done["verathos_mesh"]
        response = done["response"]

        assert response["choices"][0]["message"]["content"] == "mesh stream pong"
        assert "prompt_token_ids" not in mesh
        assert "prompt_token_count" not in mesh["receipt"]
        assert mesh["verified_sampler_mode"] == ""
        assert mesh["proof_required"] is True
        assert mesh["proof_receipt_verified"] is True
        assert mesh["proof_receipt_count"] == 1
        assert "proof_payloads" not in mesh
        assert mesh["receipt"]["request_hash"] == _payload_hash(request)
        assert mesh["receipt"]["response_hash"] == _payload_hash(response)
        assert mesh["receipt"]["receipt_hash"] == _receipt_hash(mesh["receipt"])
        assert events[-1] == ("message", "[DONE]")
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(proof, proof_thread)
        _shutdown_server(backend, backend_thread)


def _serve_stream_failure_worker(stream_failure: str):
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        content="mesh stream pong",
        stream_failure=stream_failure,
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
    )
    return backend, backend_thread, worker, worker_thread, calls


def test_mesh_worker_stream_truncated_backend_fails_closed():
    """A backend stream that dies without the [DONE] sentinel (llama-server
    crash, observed as a CUDA abort mid-serve) must surface a clear
    truncation error, never a receipt over the partial chunks and never the
    misleading missing-slot-id "runtime must be built with the Verathos
    server patches" error out of the receipt path."""

    backend, backend_thread, worker, worker_thread, _calls = (
        _serve_stream_failure_worker("truncate_mid_stream")
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    try:
        events = _post_sse_events(f"http://{host}:{port}/v1/chat/completions", request)
        error_events = [
            payload
            for event, payload in events
            if event == "error" and isinstance(payload, dict)
        ]
        done_events = [
            payload
            for event, payload in events
            if event == "done" and isinstance(payload, dict)
        ]

        assert not done_events
        assert len(error_events) == 1
        error_text = str(error_events[0].get("error", ""))
        assert "[DONE] sentinel" in error_text
        assert "cannot be receipted" in error_text
        assert "Verathos server patches" not in error_text
        # The chunks relayed before the cut still reached the client raw.
        relayed_chunks = [
            payload
            for event, payload in events
            if event == "message" and isinstance(payload, dict)
        ]
        assert relayed_chunks
        assert events[-1] == ("message", "[DONE]")
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_stream_backend_error_event_surfaces_backend_detail():
    """llama-server reports a mid-stream failure as an {"error": ...} data
    payload and closes without [DONE]; the worker must fail the serve with
    the backend's own error detail instead of aggregating the partial
    stream into a receipt."""

    backend, backend_thread, worker, worker_thread, _calls = (
        _serve_stream_failure_worker("error_event")
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    try:
        events = _post_sse_events(f"http://{host}:{port}/v1/chat/completions", request)
        error_events = [
            payload
            for event, payload in events
            if event == "error" and isinstance(payload, dict)
        ]
        done_events = [
            payload
            for event, payload in events
            if event == "done" and isinstance(payload, dict)
        ]

        assert not done_events
        assert len(error_events) == 1
        error_text = str(error_events[0].get("error", ""))
        assert "backend stream failed mid-serve" in error_text
        assert "illegal memory access" in error_text
        assert "Verathos server patches" not in error_text
        assert events[-1] == ("message", "[DONE]")
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_stream_deferred_audit_replays_non_stream_transport(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        content="mesh stream pong",
        on_request=lambda _payload: _write_ggml_mul_mat_trace(tmp_path),
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        defer_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream deferred audit"}],
        "stream": True,
    }
    try:
        events = _post_sse_events(f"http://{host}:{port}/v1/chat/completions", request)
        done = [
            payload
            for event, payload in events
            if event == "done" and isinstance(payload, dict)
        ][-1]
        artifact = done["verathos_mesh"]
        response = done["response"]
        receipt = artifact["receipt"]
        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)

        # The deferred audit proves a candidate witness captured during the
        # original serve (trace_candidate_set_v1) — no regeneration of the
        # completion. The streamed request is the only backend call.
        bundle = post_json(
            f"http://{host}:{port}/v1/mesh/proof/deferred-audit",
            {
                "artifact": artifact,
                "openai_request": request,
                "openai_response": response,
                "deferred_randomness": sampled_randomness,
            },
        )

        assert calls[0]["stream"] is True
        assert receipt["request_hash"] == _payload_hash(request)
        assert bundle["audit_receipt"]["proof_receipt_verified"] is True
        assert bundle["audit_receipt"]["verified"] is True
        assert len(bundle["proof_receipts"]) == 1
        assert verify_deferred_mesh_audit_bundle(
            bundle,
            artifact,
            request,
            openai_response=response,
            spec=mesh_spec,
            member_index=1,
        )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_aggregates_multiple_member_proof_endpoints():
    backend, backend_thread, backend_url, calls = _fake_openai_backend()
    proof_a, proof_a_thread, proof_a_url, proof_a_calls = _fake_proof_adapter()
    proof_b, proof_b_thread, proof_b_url, proof_b_calls = _fake_proof_adapter()
    coordinator_endpoint = "http://coord.local:9338"
    mesh_spec = MeshSpec(
        mesh_id="mesh-test",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=6,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=coordinator_endpoint,
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="http://worker-a.local:9338",
                proof_endpoint=proof_a_url,
                stage_index=1,
                layers=StageRange(0, 3),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="http://worker-b.local:9338",
                proof_endpoint=proof_b_url,
                stage_index=2,
                layers=StageRange(3, 6),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
        ],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[DIGEST_A],
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "aggregate proofs"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-multi-proof",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_receipts = [
            LlamaGraphOpReceipt.from_dict(item) for item in payload["proof_receipts"]
        ]

        assert _calls_match_with_derived_seed(calls, [request])
        assert len(proof_a_calls) == 1
        assert len(proof_b_calls) == 1
        assert payload.get("completion_token_ids", []) == []
        assert receipt["completion_token_count"] == 1
        assert receipt["verified_sampler_required"] is False
        assert receipt["proof_metadata_required"] is False
        assert receipt["proof_required"] is True
        assert receipt["proof_mode"] == VERATHOS_GGML_TRACE_PROOF_MODE
        assert receipt["proof_receipt_verified"] is True
        assert receipt["proof_receipt_count"] == 2
        assert {item.stage_index for item in proof_receipts} == {1, 2}
        assert receipt["proof_receipt_root"] == llama_graph_receipt_root(
            proof_receipts
        ).hex()
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            require_configured_proof=True,
        )

        tampered = json.loads(json.dumps(payload))
        tampered["proof_receipts"] = [tampered["proof_receipts"][0]]
        tampered_receipts = [
            LlamaGraphOpReceipt.from_dict(item) for item in tampered["proof_receipts"]
        ]
        tampered["receipt"]["proof_receipt_count"] = 1
        tampered["receipt"]["proof_receipt_root"] = llama_graph_receipt_root(
            tampered_receipts
        ).hex()
        tampered["receipt"]["mesh_response_commitment_hash"] = (
            _mesh_response_commitment_hash(tampered["receipt"])
        )
        tampered["receipt"]["receipt_hash"] = _receipt_hash(tampered["receipt"])
        with pytest.raises(RuntimeError, match="missing proof receipts"):
            verify_mesh_inference_artifact(
                tampered,
                request,
                spec=mesh_spec,
                require_configured_proof=True,
            )
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(proof_a, proof_a_thread)
        _shutdown_server(proof_b, proof_b_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_requires_real_ggml_gemm_proof_when_requested(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    capture_enabled_during_backend = []
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=lambda _payload: capture_enabled_during_backend.append(
            trace_enable_file.exists()
        )
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof_server = make_ggml_proof_server(
        trace_dir=trace.path.parent,
        trace_finder=lambda _ctx: trace,
        tolerance_abs=1e-6,
    )
    proof_thread = threading.Thread(target=proof_server.serve_forever, daemon=True)
    proof_thread.start()
    proof_host, proof_port = proof_server.server_address
    proof_url = f"http://{proof_host}:{proof_port}"
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        proof_url=proof_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-real-gemm",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [request])
        assert payload.get("completion_token_ids", []) == []
        assert receipt["completion_token_count"] == 1
        assert capture_enabled_during_backend == [True]
        assert not trace_enable_file.exists()
        assert receipt["proof_required"] is True
        assert receipt["proof_mode"] == VERATHOS_GGML_GEMM_PROOF_MODE
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert receipt["proof_verifier_ms"] >= 0.0
        assert payload["proof_receipts"][0]["proof_kind"] == "gemm"
        assert payload["proof_receipts"][0]["op_type"] == "GGML_OP_MUL_MAT"
        assert len(payload["proof_payloads"]) == 1
        assert (
            payload["proof_payloads"][0]["proof_commitment_hash"]
            == payload["proof_receipts"][0]["proof_commitment_hash"]
        )
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
        ).verified
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(proof_server, proof_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_embeds_real_ggml_gemm_proof_when_trace_dir_is_local(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    traces = []
    capture_enabled_during_backend = []

    def on_request(_payload):
        capture_enabled_during_backend.append(trace_enable_file.exists())
        traces.append(_write_ggml_mul_mat_trace(tmp_path))

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-embedded-gemm",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [request])
        assert payload.get("completion_token_ids", []) == []
        assert receipt["completion_token_count"] == 1
        assert len(traces) == 1
        assert capture_enabled_during_backend == [True]
        assert not trace_enable_file.exists()
        assert receipt["proof_required"] is True
        assert receipt["proof_mode"] == VERATHOS_GGML_GEMM_PROOF_MODE
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert receipt["proof_verifier_ms"] >= 0.0
        assert payload["proof_receipts"][0]["proof_kind"] == "gemm"
        assert payload["proof_receipts"][0]["op_type"] == "GGML_OP_MUL_MAT"
        assert payload["proof_receipts"][0]["request_id"] == "req-embedded-gemm"
        assert len(payload["proof_payloads"]) == 1
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
        ).verified
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_organic_default_emits_light_proof(tmp_path):
    # The organic inline lane (no proof_tier request) rides the LIGHT tier:
    # openings-only payloads with signed-able receipts, verified through the
    # any-tier router. The hard relation stays reachable via the upgrade
    # knob, which the test above pins.
    from verallm.mesh.ggml_proof import (
        GGML_LIGHT_PROOF_KIND,
        verify_mesh_proof_payloads_any_tier,
    )
    from verallm.mesh.proof import VERATHOS_GGML_LIGHT_PROOF_MODE

    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        _write_ggml_mul_mat_trace(tmp_path)

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-organic-light",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert receipt["proof_challenge_kind"] == "inline_every_request_v1"
        assert receipt["proof_mode"] == VERATHOS_GGML_LIGHT_PROOF_MODE
        assert receipt["verified"] is True
        assert receipt["proof_receipt_verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert payload["proof_receipts"][0]["proof_kind"] == GGML_LIGHT_PROOF_KIND
        assert len(payload["proof_payloads"]) == 1
        light_payload = payload["proof_payloads"][0]
        assert light_payload["proof_mode"] == VERATHOS_GGML_LIGHT_PROOF_MODE
        assert "proof" not in light_payload
        assert "weight_root" not in light_payload
        mesh_receipt = dict(receipt)
        result = verify_mesh_proof_payloads_any_tier(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=mesh_receipt,
        )
        assert result.verified, result.message
        # The same light payload must NOT satisfy a validator challenge kind.
        result = verify_mesh_proof_payloads_any_tier(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=dict(mesh_receipt, proof_challenge_kind="fiat_shamir_inline_v1"),
        )
        assert not result.verified
        assert "hard proof" in result.message
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_proves_multiple_ops_per_request(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        base_ns = time.time_ns()
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="trace-a",
            op_index=0,
        )
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="trace-b",
            op_index=1,
        )

    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=2,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-two-ops",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "verathos": {"proof_tier": "hard"},
                },
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        assert receipt["proof_required"] is True
        assert receipt["proof_ops_per_request"] == 2
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert receipt["proof_receipt_count"] == 2
        assert sorted(item["op_index"] for item in payload["proof_receipts"]) == [0, 1]
        assert len(payload["proof_payloads"]) == 2
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
        ).verified
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_proves_beacon_selected_trace_from_candidate_set(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        base_ns = time.time_ns()
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="trace-a",
            op_index=0,
        )
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="trace-b",
            op_index=1,
        )

    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=2,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-candidate-set",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "verathos": {"proof_tier": "hard"},
                },
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        membership = payload["proof_payloads"][0]["trace_membership"]

        assert receipt["proof_receipt_count"] == 1
        assert receipt["proof_trace_commitment_count"] == 2
        assert receipt["proof_trace_candidates_per_request"] == 2
        assert receipt["proof_op_manifest_count"] == 2
        assert receipt["proof_op_manifest_root"]
        assert payload["proof_payloads"][0]["op_manifest_membership"]["op_manifest_count"] == 2
        assert membership["trace_set_count"] == 2
        assert membership["trace_leaf_index"] in (0, 1)
        assert payload["proof_receipts"][0]["op_index"] in (0, 1)
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_keeps_organic_proof_on_trace_candidate_set_with_manifest(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        base_ns = time.time_ns()
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="trace-a",
            op_index=0,
        )
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="trace-b",
            op_index=1,
        )

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-manifest-wide",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "verathos": {"proof_tier": "hard"},
                },
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_payload = payload["proof_payloads"][0]

        assert receipt["proof_trace_commitment_count"] == 1
        assert receipt["proof_op_manifest_count"] == 2
        assert receipt["proof_trace_scope"] == "trace_candidate_set_v1"
        assert proof_payload["trace"]["op_index"] == 0
        assert proof_payload["op_manifest_membership"]["op_manifest_leaf_index"] == 0
        assert proof_payload["trace_membership"]["trace_set_count"] == 1
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_trace_capture_accepts_selected_manifest_indexes(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Worker",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=trace_enable_file,
    )
    host, port = worker.server_address
    try:
        response = post_json(
            f"http://{host}:{port}/v1/mesh/trace-capture",
            {
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "request_id": "req-selected-capture",
                "enabled": True,
                "selected_manifest_indexes": [7, 11],
            },
        )

        assert response["capture_enabled"] is True
        assert response["selected_manifest_indexes"] == [7, 11]
        token = trace_enable_file.read_text(encoding="utf-8")
        assert "|selected=7,11" in token

        response = post_json(
            f"http://{host}:{port}/v1/mesh/trace-capture",
            {
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "request_id": "req-selected-capture",
                "enabled": False,
            },
        )
        assert response["capture_enabled"] is False
        assert not trace_enable_file.exists()
    finally:
        _shutdown_server(worker, worker_thread)


def test_mesh_worker_organic_proof_ignores_missing_manifest_witness(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        base_ns = time.time_ns()
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="trace-a",
            op_index=0,
        )
        missing = _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="trace-b",
            op_index=1,
        )
        missing.path.unlink()

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-manifest-missing",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                },
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_payload = payload["proof_payloads"][0]

        assert receipt["proof_trace_scope"] == "trace_candidate_set_v1"
        assert proof_payload["trace"]["op_index"] == 0
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_organic_proof_does_not_replay_missing_manifest_witness(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        token = (
            trace_enable_file.read_text(encoding="utf-8")
            if trace_enable_file.exists()
            else ""
        )
        base_ns = time.time_ns()
        if "selected=1" in token:
            _write_ggml_mul_mat_trace(
                tmp_path,
                created_unix_ns=base_ns,
                name="trace-b-replay",
                op_index=1,
                manifest_index=1,
            )
            return
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="trace-a",
            op_index=0,
            manifest_index=0,
        )
        missing = _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="trace-b",
            op_index=1,
            manifest_index=1,
        )
        missing.path.unlink()

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-manifest-replay",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": backend_request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_payload = payload["proof_payloads"][0]

        assert _calls_match_with_derived_seed(calls, [backend_request])
        assert receipt["proof_trace_commitment_count"] == 1
        assert receipt["proof_op_manifest_count"] == 2
        assert receipt["proof_trace_scope"] == "trace_candidate_set_v1"
        assert proof_payload["trace"]["op_index"] == 0
        assert "trace-a" in proof_payload["trace"]["path"]
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_manifest_only_organic_proof_replays_selected_witness(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    trace_enable_file = tmp_path / ".capture-enabled"
    capture_tokens = []

    def on_request(_payload):
        token = (
            trace_enable_file.read_text(encoding="utf-8")
            if trace_enable_file.exists()
            else ""
        )
        capture_tokens.append(token)
        base_ns = time.time_ns()
        if "selected=1" in token:
            _write_ggml_mul_mat_trace(
                tmp_path,
                created_unix_ns=base_ns,
                name="trace-b-replay",
                op_index=1,
                manifest_index=1,
            )
            return
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns,
            name="manifest-a",
            op_index=0,
            manifest_index=0,
            write_trace=False,
        )
        _write_ggml_mul_mat_trace(
            tmp_path,
            created_unix_ns=base_ns + 1,
            name="manifest-b",
            op_index=1,
            manifest_index=1,
            write_trace=False,
        )

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "manifest only"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-manifest-only-replay",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": backend_request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_payload = payload["proof_payloads"][0]

        assert _calls_match_with_derived_seed(calls, [backend_request, backend_request])
        assert capture_tokens[0]
        assert "selected=1" in capture_tokens[1]
        assert receipt["proof_trace_commitment_count"] == 0
        assert receipt["proof_op_manifest_count"] == 2
        assert receipt["proof_trace_scope"] == "op_manifest_challenge_v1"
        assert receipt["proof_trace_commitment_root"] == ""
        assert receipt.get("proof_replay_started_unix_ns", 0) > 0
        assert proof_payload["trace"]["op_index"] == 1
        assert "trace-b-replay" in proof_payload["trace"]["path"]
        assert proof_payload["op_manifest_membership"]["op_manifest_leaf_index"] == 1
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_coordinator_keeps_remote_organic_proof_on_trace_candidates(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    remote_root = tmp_path / "remote"
    remote_capture_file = remote_root / ".capture-enabled"
    trace_dir = remote_root / "ggml-traces"

    def on_request(_payload):
        token = (
            remote_capture_file.read_text(encoding="utf-8")
            if remote_capture_file.exists()
            else ""
        )
        base_ns = time.time_ns()
        if "selected=1" in token:
            _write_ggml_mul_mat_trace(
                remote_root,
                created_unix_ns=base_ns,
                name="remote-trace-b-replay",
                op_index=1,
                manifest_index=1,
            )
            return
        _write_ggml_mul_mat_trace(
            remote_root,
            created_unix_ns=base_ns,
            name="remote-trace-a",
            op_index=0,
            manifest_index=0,
        )
        missing = _write_ggml_mul_mat_trace(
            remote_root,
            created_unix_ns=base_ns + 1,
            name="remote-trace-b",
            op_index=1,
            manifest_index=1,
        )
        missing.path.unlink()

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "remote replay"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_endpoint = "http://coord.local:9338"
    remote_port = _free_port()
    remote_endpoint = f"http://127.0.0.1:{remote_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=remote_endpoint,
    )
    _make_coordinator_orchestration_only(mesh_spec)
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=remote_endpoint,
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
        rpc_endpoint="worker.local:50052",
        proof_endpoint=remote_endpoint,
    )
    remote_worker, remote_thread = serve_worker_in_thread(
        capability=remote_capability,
        host="127.0.0.1",
        port=remote_port,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=remote_capture_file,
        proof_trace_dir=trace_dir,
        proof_tolerance_abs=1e-6,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-remote-organic-candidate-proof",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": backend_request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof_payload = payload["proof_payloads"][0]

        assert _calls_match_with_derived_seed(calls, [backend_request])
        assert receipt["proof_trace_scope"] == "trace_candidate_set_v1"
        assert receipt.get("proof_replay_started_unix_ns", 0) == 0
        assert receipt.get("proof_replay_ended_unix_ns", 0) == 0
        assert proof_payload["trace"]["op_index"] == 0
        assert "remote-trace-a" in proof_payload["trace"]["path"]
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
        assert not remote_capture_file.exists()
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_stageless_coordinator_replays_remote_selected_witnesses_without_local_capture(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.ggml_proof as ggml_proof_mod

    remote_roots = [
        tmp_path / "remote-selected-replay-one",
        tmp_path / "remote-selected-replay-two",
    ]
    remote_capture_files = [root / ".capture-enabled" for root in remote_roots]
    capture_tokens = [[], []]

    def on_request(_payload):
        base_ns = time.time_ns()
        for stage_offset, (remote_root, capture_file, tokens) in enumerate(
            zip(remote_roots, remote_capture_files, capture_tokens),
            start=1,
        ):
            token = (
                capture_file.read_text(encoding="utf-8")
                if capture_file.exists()
                else ""
            )
            tokens.append(token)
            if "selected=1" in token:
                _write_ggml_mul_mat_trace(
                    remote_root,
                    created_unix_ns=base_ns + stage_offset,
                    name=f"remote-{stage_offset}-trace-b-replay",
                    op_index=1,
                    manifest_index=1,
                )
                continue
            _write_ggml_mul_mat_trace(
                remote_root,
                created_unix_ns=base_ns + stage_offset,
                name=f"remote-{stage_offset}-manifest-a",
                op_index=0,
                manifest_index=0,
                write_trace=False,
            )
            _write_ggml_mul_mat_trace(
                remote_root,
                created_unix_ns=base_ns + stage_offset + 2,
                name=f"remote-{stage_offset}-manifest-b",
                op_index=1,
                manifest_index=1,
                write_trace=False,
            )

    monkeypatch.setattr(
        ggml_proof_mod,
        "select_manifest_challenge_indexes",
        lambda **_kwargs: [1],
    )
    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "remote selected replay"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_endpoint = "http://coord.local:9338"
    remote_ports = [_free_port(), _free_port()]
    remote_endpoints = [
        f"http://127.0.0.1:{port}" for port in remote_ports
    ]
    mesh_spec = MeshSpec(
        mesh_id="mesh-all-rpc-selected-replay",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=coordinator_endpoint,
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=remote_endpoints[0],
                rpc_endpoint="worker-one.local:50052",
                rpc_split_weight=1,
                proof_endpoint=remote_endpoints[0],
                stage_index=1,
                layers=StageRange(0, 3),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=remote_endpoints[1],
                rpc_endpoint="worker-two.local:50052",
                rpc_split_weight=1,
                proof_endpoint=remote_endpoints[1],
                stage_index=2,
                layers=StageRange(3, 4),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
        ],
    )
    mesh_spec.validate()
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_servers = []
    for index, (remote_root, capture_file, port, endpoint) in enumerate(
        zip(remote_roots, remote_capture_files, remote_ports, remote_endpoints),
        start=1,
    ):
        remote_capability = CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint=endpoint,
            supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
            cached_model_package_hashes=[DIGEST_A],
            rpc_endpoint=f"worker-{index}.local:50052",
            proof_endpoint=endpoint,
        )
        remote_servers.append(
            serve_worker_in_thread(
                capability=remote_capability,
                host="127.0.0.1",
                port=port,
                mesh_spec=mesh_spec,
                proof_trace_enable_file=capture_file,
                proof_trace_dir=remote_root / "ggml-traces",
                proof_tolerance_abs=1e-6,
            )
        )
    # Pool-launched all-RPC coordinators are orchestration-only and therefore
    # intentionally have no local proof trace directory or capture file.
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-remote-selected-replay",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": backend_request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(
            calls,
            [backend_request, backend_request],
        )
        for tokens in capture_tokens:
            assert len(tokens) == 2
            assert tokens[0]
            assert "selected=" not in tokens[0]
            assert "selected=1" in tokens[1]
        selected_window_tokens = {
            tokens[1].split("|", 1)[0] for tokens in capture_tokens
        }
        assert len(selected_window_tokens) == 1
        assert next(iter(selected_window_tokens))
        assert receipt["proof_trace_scope"] == "op_manifest_challenge_v1"
        assert receipt["proof_trace_commitment_count"] == 0
        assert receipt["proof_op_manifest_count"] == 4
        assert receipt.get("proof_replay_started_unix_ns", 0) > 0
        assert receipt.get("proof_replay_ended_unix_ns", 0) >= receipt[
            "proof_replay_started_unix_ns"
        ]
        assert receipt["proof_receipt_verified"] is True
        assert receipt["proof_receipt_count"] == 2
        assert {item["stage_index"] for item in payload["proof_receipts"]} == {
            1,
            2,
        }
        assert len(payload["proof_payloads"]) == 2
        assert {
            proof_payload_stage_index(item) for item in payload["proof_payloads"]
        } == {1, 2}
        assert all(
            item["trace"]["op_index"] == 1
            for item in payload["proof_payloads"]
        )
        assert {
            Path(item["trace"]["path"]).stem
            for item in payload["proof_payloads"]
        } == {
            "remote-1-trace-b-replay",
            "remote-2-trace-b-replay",
        }
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
            mesh_receipt=receipt,
        ).verified
        assert all(not path.exists() for path in remote_capture_files)
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        for remote_worker, remote_thread in remote_servers:
            _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_coordinator_resolves_remote_deferred_audit_bundle(tmp_path):
    remote_root = tmp_path / "remote-deferred"
    remote_capture_file = remote_root / ".capture-enabled"
    trace_dir = remote_root / "ggml-traces"

    def on_request(_payload):
        if remote_capture_file.exists():
            _write_ggml_mul_mat_trace(remote_root)

    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "remote deferred audit"}],
        "stream": False,
    }
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_endpoint = "http://coord.local:9338"
    remote_port = _free_port()
    remote_endpoint = f"http://127.0.0.1:{remote_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=remote_endpoint,
    )
    _make_coordinator_orchestration_only(mesh_spec)
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=remote_endpoint,
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
        rpc_endpoint="worker.local:50052",
        proof_endpoint=remote_endpoint,
    )
    remote_worker, remote_thread = serve_worker_in_thread(
        capability=remote_capability,
        host="127.0.0.1",
        port=remote_port,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=remote_capture_file,
        proof_trace_dir=trace_dir,
        proof_tolerance_abs=1e-6,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_sample_bps=1000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-remote-deferred-bundle",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": backend_request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        assert _calls_match_with_derived_seed(calls, [backend_request])
        assert receipt["proof_required"] is False
        assert receipt["proof_deferred"] is True
        assert receipt["proof_op_manifest_count"] == 1
        assert "proof_receipts" not in payload

        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)
        bundle = post_json(
            f"http://{host}:{port}/v1/mesh/proof/deferred-audit",
            {
                "artifact": payload,
                "openai_request": backend_request,
                "deferred_randomness": sampled_randomness,
            },
        )
        proof_receipts = [
            LlamaGraphOpReceipt.from_dict(item) for item in bundle["proof_receipts"]
        ]
        assert bundle["origin_receipt_hash"] == receipt["receipt_hash"]
        assert bundle["audit_receipt"]["proof_required"] is True
        assert bundle["audit_receipt"]["proof_receipt_verified"] is True
        assert {item.stage_index for item in proof_receipts} == {1}
        assert len(bundle["proof_payloads"]) == 1
        assert verify_deferred_mesh_audit_bundle(
            bundle,
            payload,
            backend_request,
            spec=mesh_spec,
        )
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_partial_bps_binds_trace_without_unsampled_proof(tmp_path, monkeypatch):
    import verallm.mesh.worker as mesh_worker_mod

    trace_enable_file = tmp_path / ".capture-enabled"
    capture_enabled_during_backend = []

    def on_request(_payload):
        capture_enabled_during_backend.append(trace_enable_file.exists())
        _write_ggml_mul_mat_trace(tmp_path)

    monkeypatch.setattr(mesh_worker_mod, "mesh_proof_sample_value", lambda _beacon: 9000)
    monkeypatch.setattr(
        mesh_worker_mod,
        "should_sample_mesh_proof",
        lambda *, beacon, sample_bps: False,
    )
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=1000,
    )
    host, port = worker.server_address
    backend_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    request = {
        **backend_request,
        "verathos": {"validator_nonce": "11" * 32},
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-sampled-out",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [backend_request])
        assert capture_enabled_during_backend == [True]
        assert payload["completion_token_ids"] == []
        assert receipt["completion_token_count"] == 1
        assert receipt["proof_configured_required"] is True
        assert receipt["proof_capture_required"] is True
        assert receipt["verified_sampler_required"] is False
        assert receipt["proof_metadata_required"] is False
        assert receipt["proof_required"] is False
        assert receipt["proof_sampled"] is False
        assert receipt["proof_sample_bps"] == 1000
        assert receipt["proof_sample_value"] == 9000
        assert receipt["proof_challenge_kind"] == "fiat_shamir_inline_v1"
        assert receipt["proof_deferred"] is True
        assert receipt["proof_deferred_mode"] == "future_randomness_v1"
        assert receipt["proof_deferred_audit_bps"] == 1000
        assert receipt["proof_deferred_sample_commitment_hash"] == (
            mesh_deferred_audit_sample_commitment_hash(receipt)
        )
        assert receipt["proof_deferred_commitment_hash"] == (
            mesh_deferred_audit_commitment_hash(receipt)
        )
        assert receipt["proof_trace_commitment_count"] == 1
        assert receipt["proof_trace_commitment_root"]
        assert receipt["proof_op_manifest_count"] == 1
        assert receipt["proof_op_manifest_root"]
        assert receipt["proof_gate_hash"]
        assert len(receipt["proof_beacon"]) == 64
        assert "proof_receipts" not in payload
        assert mesh_proof_gate_hash(receipt) == receipt["proof_gate_hash"]
        post_gate_mutation = {
            **receipt,
            "proof_required": True,
            "proof_sampled": True,
            "proof_sample_value": 0,
            "proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE,
            "inference_started_unix_ns": 1,
            "inference_ended_unix_ns": 2,
        }
        assert mesh_proof_gate_hash(post_gate_mutation) == receipt["proof_gate_hash"]
        trace_root_mutation = {
            **receipt,
            "proof_trace_commitment_root": DIGEST_B,
            "proof_op_manifest_root": DIGEST_C,
        }
        assert mesh_proof_gate_hash(trace_root_mutation) == receipt["proof_gate_hash"]
        verify_mesh_proof_sampling_fields(receipt, request)
        from verallm.mesh.proof import (
            mesh_proof_sample_value as real_mesh_proof_sample_value,
            should_sample_mesh_proof as real_should_sample_mesh_proof,
        )

        monkeypatch.setattr(
            mesh_worker_mod,
            "mesh_proof_sample_value",
            real_mesh_proof_sample_value,
        )
        monkeypatch.setattr(
            mesh_worker_mod,
            "should_sample_mesh_proof",
            real_should_sample_mesh_proof,
        )
        unsampled_randomness = _deferred_randomness_for_sample(
            receipt,
            sampled=False,
        )
        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)
        unsampled_decision = deferred_audit_decision(
            receipt,
            randomness=unsampled_randomness,
        )
        assert unsampled_decision["sampled"] is False
        sampled_decision = deferred_audit_decision(
            receipt,
            randomness=sampled_randomness,
        )
        assert sampled_decision["sampled"] is True
        assert sampled_decision["proof_sample_value"] == sampled_decision["sample_value"]
        monkeypatch.setattr(mesh_worker_mod, "mesh_proof_sample_value", lambda _beacon: 9000)
        monkeypatch.setattr(
            mesh_worker_mod,
            "should_sample_mesh_proof",
            lambda *, beacon, sample_bps: False,
        )
        tampered = {**receipt, "proof_sampled": True, "proof_required": True}
        with pytest.raises(RuntimeError, match="proof_sampled"):
            verify_mesh_proof_sampling_fields(tampered, request)
        deferred_tampered = {**receipt, "proof_deferred_audit_bps": 999}
        with pytest.raises(
            RuntimeError,
            match="proof_deferred_sample_commitment_hash",
        ):
            deferred_audit_decision(deferred_tampered)
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_partial_bps_samples_and_attaches_proof(tmp_path, monkeypatch):
    import verallm.mesh.worker as mesh_worker_mod

    trace_enable_file = tmp_path / ".capture-enabled"
    reference_trace = _write_ggml_mul_mat_trace(tmp_path / "manifest-source")
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def on_request(_payload):
        _write_ggml_mul_mat_trace(tmp_path)

    monkeypatch.setattr(mesh_worker_mod, "mesh_proof_sample_value", lambda _beacon: 0)
    monkeypatch.setattr(
        mesh_worker_mod,
        "should_sample_mesh_proof",
        lambda *, beacon, sample_bps: True,
    )
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=1000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {"validator_nonce": "22" * 32},
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-sampled-in",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(
            calls,
            [
                {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                }
            ],
        )
        assert payload["completion_token_ids"] == []
        assert receipt["completion_token_count"] == 1
        assert receipt["verified_sampler_required"] is False
        assert receipt["proof_metadata_required"] is False
        assert receipt["proof_required"] is True
        assert receipt["proof_sampled"] is True
        assert receipt["proof_sample_value"] == 0
        assert receipt["proof_trace_commitment_count"] == 1
        assert receipt["proof_op_manifest_count"] == 1
        assert receipt["proof_op_manifest_root"]
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert payload["proof_receipts"][0]["proof_kind"] == "gemm"
        assert len(payload["proof_payloads"]) == 1
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            require_configured_proof=True,
            require_cryptographic_proof=True,
        )
        unverified_payload = json.loads(json.dumps(payload))
        unverified_payload["receipt"]["verified"] = False
        unverified_payload["receipt"]["receipt_hash"] = _receipt_hash(
            unverified_payload["receipt"]
        )
        with pytest.raises(RuntimeError, match="verified flag"):
            verify_mesh_inference_artifact(
                unverified_payload,
                request,
                spec=mesh_spec,
                member_index=1,
                require_configured_proof=True,
                require_cryptographic_proof=True,
            )
        verify_mesh_proof_sampling_fields(receipt, request)
        assert verify_ggml_gemm_proof_payloads(
            payload["proof_payloads"],
            payload["proof_receipts"],
        ).verified
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_request_canary_raises_proof_bps(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    reference_trace = _write_ggml_mul_mat_trace(tmp_path / "manifest-source")
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def on_request(_payload):
        _write_ggml_mul_mat_trace(tmp_path)

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=1000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {"proof_tier": "hard",
            "validator_nonce": "33" * 32,
            "proof_sample_bps": 10_000,
        },
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-canary-proof",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(
            calls,
            [
                {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                }
            ],
        )
        assert payload["completion_token_ids"] == []
        assert receipt["proof_sample_bps"] == 10_000
        assert receipt["proof_challenge_kind"] == "inline_every_request_v1"
        assert receipt["verified_sampler_required"] is False
        assert receipt["proof_metadata_required"] is False
        assert receipt["proof_required"] is True
        assert receipt["proof_sampled"] is True
        assert receipt["proof_sample_value"] == 0
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert len(payload["proof_payloads"]) == 1
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            require_configured_proof=True,
            require_cryptographic_proof=True,
        )
        verify_mesh_proof_sampling_fields(receipt, request)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_request_canary_raises_decode_audit_bps(tmp_path, monkeypatch):
    # These fixtures model a mesh whose llama.cpp build has no boundary
    # capture; the chain requirement is default-on in production, so the
    # capture-less flow needs the explicit opt-out.
    monkeypatch.setenv("VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN", "0")

    trace_enable_file = tmp_path / ".capture-enabled"
    reference_trace = _write_decode_lm_head_trace(
        tmp_path / "manifest-source",
        token_id=101,
    )
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def on_request(_payload):
        _write_decode_lm_head_trace(tmp_path, token_id=101)

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=0,
        decode_audit_bps=0,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "verathos": {"proof_tier": "hard",
            "decode_audit_bps": 10_000,
            "decode_audit_top_k": 8,
        },
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-canary-decode",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert calls == [
            _verified_sampler_controls_request(
                {
                    "model": "model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "return_tokens": True,
                    "verbose": True,
                }
            )
        ]
        assert payload["completion_token_ids"] == [101]
        assert receipt["proof_sample_bps"] == 0
        assert receipt["proof_sampled"] is False
        assert receipt["decode_audit_bps"] == 10_000
        assert receipt["decode_audit_required"] is True
        assert receipt["decode_audit_sampled"] is True
        assert receipt["decode_audit_positions"] == [0]
        assert receipt["decode_audit_verified"] is True
        assert receipt["proof_required"] is True
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert len(payload["proof_payloads"]) == 1
        assert payload["proof_payloads"][0]["decode_audit_openings"][0]["token_id"] == 101
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            require_configured_proof=True,
            require_cryptographic_proof=True,
        )
        verify_mesh_proof_sampling_fields(receipt, request)
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_decode_audit_samples_and_verifies_argmax(tmp_path, monkeypatch):
    # These fixtures model a mesh whose llama.cpp build has no boundary
    # capture; the chain requirement is default-on in production, so the
    # capture-less flow needs the explicit opt-out.
    monkeypatch.setenv("VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN", "0")

    trace_enable_file = tmp_path / ".capture-enabled"
    capture_enabled_during_backend = []
    reference_trace = _write_decode_lm_head_trace(tmp_path / "manifest-source", token_id=101)
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def on_request(_payload):
        capture_enabled_during_backend.append(trace_enable_file.exists())
        _write_decode_lm_head_trace(tmp_path, token_id=101)

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=0,
        decode_audit_bps=10_000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "decode audit"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-decode-audit",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [_verified_backend_request(request)])
        assert capture_enabled_during_backend == [True]
        assert payload["completion_token_ids"] == [101]
        assert receipt["proof_sampled"] is False
        assert receipt["proof_required"] is True
        assert receipt["decode_audit_mode"] == VERATHOS_GGUF_DECODE_AUDIT_MODE
        assert receipt["decode_audit_sampled"] is True
        assert receipt["decode_audit_required"] is True
        assert receipt["proof_trace_scope"] == "op_manifest_challenge_v1"
        assert receipt["decode_audit_positions"] == [0]
        assert receipt["decode_audit_verified"] is True
        assert len(payload["proof_payloads"]) == 1
        assert len(payload["proof_payloads"][0]["decode_audit_openings"]) == 1
        assert receipt["decode_audit_receipt_root"] == ggml_decode_audit_receipt_root(
            payload["proof_payloads"]
        )
        opening = payload["proof_payloads"][0]["decode_audit_openings"][0]
        assert opening["position"] == 0
        assert opening["token_id"] == 101
        assert opening["argmax_token_id"] == 101
        assert opening["top_token_ids"][0] == 101
        assert verify_ggml_decode_audit_payloads(
            receipt,
            payload["proof_payloads"],
            completion_token_ids=payload["completion_token_ids"],
        ).verified
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            require_configured_proof=True,
            require_cryptographic_proof=True,
        )

        tampered = json.loads(json.dumps(payload))
        tampered["proof_payloads"][0]["decode_audit_openings"][0]["argmax_token_id"] = 100
        with pytest.raises(RuntimeError, match="proof verification|decode audit"):
            verify_mesh_inference_artifact(
                tampered,
                request,
                spec=mesh_spec,
                member_index=1,
                require_configured_proof=True,
                require_cryptographic_proof=True,
            )
        tampered_top_k = json.loads(json.dumps(payload))
        tampered_top_k["proof_payloads"][0]["decode_audit_openings"][0][
            "top_token_ids"
        ][0] = 100
        with pytest.raises(RuntimeError, match="proof verification|decode audit"):
            verify_mesh_inference_artifact(
                tampered_top_k,
                request,
                spec=mesh_spec,
                member_index=1,
                require_configured_proof=True,
                require_cryptographic_proof=True,
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_decode_audit_candidate_selection_ignores_block_attention_output(tmp_path):
    trace = _write_decode_lm_head_trace(
        tmp_path,
        token_id=101,
        op_index=99,
        manifest_index=99,
        write_trace=False,
    )
    final_entry = GgmlOpManifestEntry(
        path=tmp_path / "manifest.jsonl",
        created_unix_ns=trace.created_unix_ns,
        manifest_index=99,
        graph_id=trace.graph_id,
        op_index=99,
        op_type="GGML_OP_MUL_MAT",
        tensor_name="output.weight",
        src0_name="output.weight",
        src1_name="result_norm",
        dst_name="result_output",
        src0_shape=trace.src0_shape,
        src1_shape=trace.src1_shape,
        dst_shape=trace.dst_shape,
        source_types=trace.source_types,
        backend=trace.backend,
        device=trace.device,
    )
    block_entry = GgmlOpManifestEntry(
        path=tmp_path / "manifest.jsonl",
        created_unix_ns=trace.created_unix_ns - 1,
        manifest_index=98,
        graph_id="attn-block",
        op_index=98,
        op_type="GGML_OP_MUL_MAT",
        tensor_name="blk.13.attn_output.weight",
        src0_name="blk.13.attn_output.weight",
        src1_name="kqv_out-13",
        dst_name="node_468",
        src0_shape=(896, 896, 1, 1),
        src1_shape=(896, 1, 1, 1),
        dst_shape=(896, 1, 1, 1),
        source_types={"src0": "q5_0", "src1": "f32", "dst": "f32"},
        backend="llama_cpp_cuda",
        device="CUDA0",
    )

    assert not is_decode_candidate_manifest_entry(block_entry)
    assert is_decode_candidate_manifest_entry(final_entry)
    assert select_decode_manifest_entries(
        [block_entry, final_entry],
        completion_token_ids=[101],
        positions=[0],
    ) == {0: final_entry}


def test_mesh_worker_sampled_decode_audit_binds_original_token_metadata(
    tmp_path,
    monkeypatch,
):
    # Capture-less fixture mesh; see the sibling decode-audit tests.
    monkeypatch.setenv("VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN", "0")
    import verallm.mesh.worker as mesh_worker_mod

    trace_enable_file = tmp_path / ".capture-enabled"
    reference_trace = _write_decode_lm_head_trace(tmp_path / "manifest-source", token_id=101)
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    request_count = 0

    def on_request(_payload):
        nonlocal request_count
        request_count += 1
        _write_decode_lm_head_trace(
            tmp_path,
            token_id=101,
            write_trace=request_count > 1,
        )

    monkeypatch.setattr(
        mesh_worker_mod,
        "mesh_decode_audit_sample_value",
        lambda _beacon: 0,
    )
    monkeypatch.setattr(
        mesh_worker_mod,
        "should_sample_mesh_decode_audit",
        lambda *, beacon, sample_bps: True,
    )
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=0,
        decode_audit_bps=100,
    )
    host, port = worker.server_address
    base_request = {
        "model": "model",
        "messages": [{"role": "user", "content": "sampled decode audit"}],
        "stream": False,
    }
    request = {**base_request, "verathos": {"validator_nonce": "44" * 32}}
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-sampled-decode-audit",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        # The live serve keeps the prompt cache; the witness-regenerating
        # decode-audit replay disables it so a cache hit cannot starve it.
        assert calls == [
            _verified_backend_request(base_request),
            {**_verified_backend_request(base_request), "cache_prompt": False},
        ]
        assert payload["completion_token_ids"] == [101]
        assert receipt["proof_metadata_required"] is True
        assert receipt["proof_deferred"] is True
        assert receipt["proof_sampled"] is False
        assert receipt["decode_audit_sampled"] is True
        assert receipt["decode_audit_required"] is True
        assert receipt["decode_audit_completion_token_ids"] == [101]
        assert receipt["decode_audit_completion_token_count"] == 1
        assert receipt["decode_audit_completion_token_ids_hash"]
        assert receipt["decode_audit_verified"] is True
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            require_configured_proof=True,
            require_cryptographic_proof=True,
        )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_decode_audit_rejects_non_argmax_output(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"

    def on_request(_payload):
        _write_decode_lm_head_trace(tmp_path, token_id=100)

    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=0,
        decode_audit_bps=10_000,
    )
    host, port = worker.server_address
    try:
        with pytest.raises(RuntimeError, match="decode audit|HTTP 500"):
            post_json(
                f"http://{host}:{port}/v1/mesh/inference",
                {
                    "request_id": "req-decode-audit-bad",
                    "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                    "openai_request": {
                        "model": "model",
                        "messages": [{"role": "user", "content": "decode audit bad"}],
                        "stream": False,
                    },
                    "require_proof": True,
                },
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_partial_bps_without_nonce_uses_deferred_audit(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=lambda _payload: _write_ggml_mul_mat_trace(tmp_path)
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_sample_bps=1000,
    )
    host, port = worker.server_address
    try:
        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-no-nonce",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        assert receipt["proof_challenge_kind"] == "deferred_future_randomness_v1"
        assert receipt["proof_required"] is False
        assert receipt["proof_sampled"] is False
        assert receipt["proof_sample_value"] == -1
        assert receipt["proof_beacon"] == ""
        assert receipt["proof_deferred"] is True
        assert receipt["proof_deferred_audit_bps"] == 1000
        assert receipt["proof_deferred_sample_commitment_hash"] == (
            mesh_deferred_audit_sample_commitment_hash(receipt)
        )
        assert receipt["proof_deferred_commitment_hash"] == (
            mesh_deferred_audit_commitment_hash(receipt)
        )
        verify_mesh_proof_sampling_fields(receipt, request)
        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)
        decision = deferred_audit_decision(receipt, randomness=sampled_randomness)
        assert decision["sampled"] is True
        with pytest.raises(RuntimeError, match="deferred sampled proof"):
            verify_mesh_inference_artifact(
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
                deferred_randomness=sampled_randomness,
                require_deferred_proof_if_sampled=True,
            )
        assert verify_mesh_inference_artifact(
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
            deferred_randomness=_deferred_randomness_for_sample(
                receipt,
                sampled=False,
            ),
            require_deferred_proof_if_sampled=True,
        )
        deferred_mutation = {
            **receipt,
            "proof_deferred_randomness_round": "different-round",
        }
        with pytest.raises(
            RuntimeError,
            match="proof_deferred_sample_commitment_hash",
        ):
            deferred_audit_decision(
                deferred_mutation,
                randomness=sampled_randomness,
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_full_bps_defer_proof_requires_deferred_audit_bundle(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=lambda _payload: _write_ggml_mul_mat_trace(tmp_path)
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        defer_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=10_000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "full bps deferred proof"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-full-bps-deferred-gemm-bundle",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        assert receipt["proof_challenge_kind"] == "deferred_future_randomness_v1"
        assert receipt["proof_capture_required"] is True
        assert receipt["proof_required"] is False
        assert receipt["proof_sampled"] is False
        assert receipt["proof_sample_value"] == -1
        assert receipt["proof_beacon"] == ""
        assert receipt["proof_deferred"] is True
        assert receipt["proof_deferred_obligation"] is True
        assert receipt["proof_deferred_required"] is True
        assert receipt["proof_deferred_audit_bps"] == 10_000
        assert receipt["proof_trace_scope"] == "trace_candidate_set_v1"
        assert receipt["proof_deferred_sample_commitment_hash"] == (
            mesh_deferred_audit_sample_commitment_hash(receipt)
        )
        assert receipt["proof_deferred_commitment_hash"] == (
            mesh_deferred_audit_commitment_hash(receipt)
        )
        assert "proof_receipts" not in payload
        assert "proof_payloads" not in payload
        verify_mesh_proof_sampling_fields(receipt, request)

        sampled_randomness = hashlib.sha256(
            b"VERATHOS_TEST_FULL_BPS_DEFERRED_RANDOMNESS"
        ).hexdigest()
        decision = deferred_audit_decision(receipt, randomness=sampled_randomness)
        assert decision["sampled"] is True
        assert decision["proof_sampled"] is True
        audit_context = deferred_audit_context_from_receipt(
            receipt,
            randomness=sampled_randomness,
        )
        assert audit_context["proof_trace_scope"] == "trace_candidate_set_v1"
        with pytest.raises(RuntimeError, match="deferred sampled proof"):
            verify_mesh_inference_artifact(
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
                deferred_randomness=sampled_randomness,
                require_deferred_proof_if_sampled=True,
            )

        bundle = post_json(
            f"http://{host}:{port}/v1/mesh/proof/deferred-audit",
            {
                "artifact": payload,
                "openai_request": request,
                "deferred_randomness": sampled_randomness,
            },
        )

        assert bundle["origin_receipt_hash"] == receipt["receipt_hash"]
        assert bundle["decision"]["sampled"] is True
        assert bundle["decision"]["proof_sampled"] is True
        assert bundle["audit_receipt"]["proof_required"] is True
        assert bundle["audit_receipt"]["proof_sampled"] is True
        assert bundle["audit_receipt"]["proof_receipt_verified"] is True
        assert bundle["audit_receipt"]["verified"] is True
        assert len(bundle["proof_receipts"]) == 1
        assert len(bundle["proof_payloads"]) == 1
        assert verify_deferred_mesh_audit_bundle(
            bundle,
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
        )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_resolves_deferred_gemm_audit_bundle(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=lambda _payload: _write_ggml_mul_mat_trace(tmp_path)
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=1000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "post hoc gemm audit"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-deferred-gemm-bundle",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)
        with pytest.raises(RuntimeError, match="deferred sampled proof"):
            verify_mesh_inference_artifact(
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
                deferred_randomness=sampled_randomness,
                require_deferred_proof_if_sampled=True,
            )

        bundle = post_json(
            f"http://{host}:{port}/v1/mesh/proof/deferred-audit",
            {
                "artifact": payload,
                "openai_request": request,
                "deferred_randomness": sampled_randomness,
            },
        )

        assert bundle["origin_receipt_hash"] == receipt["receipt_hash"]
        assert bundle["deferred_commitment_hash"] == receipt[
            "proof_deferred_commitment_hash"
        ]
        assert bundle["deferred_sample_commitment_hash"] == receipt[
            "proof_deferred_sample_commitment_hash"
        ]
        assert bundle["decision"]["sampled"] is True
        assert bundle["decision"]["proof_sampled"] is True
        assert bundle["audit_receipt"]["proof_required"] is True
        assert bundle["audit_receipt"]["proof_sampled"] is True
        assert bundle["audit_receipt"]["proof_receipt_verified"] is True
        assert bundle["audit_receipt"]["verified"] is True
        assert len(bundle["proof_receipts"]) == 1
        assert len(bundle["proof_payloads"]) == 1
        assert verify_deferred_mesh_audit_bundle(
            bundle,
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
        )

        missing_payloads = json.loads(json.dumps(bundle))
        missing_payloads["proof_payloads"] = []
        missing_payloads["bundle_hash"] = mesh_deferred_audit_bundle_hash(missing_payloads)
        with pytest.raises(RuntimeError, match="proof payloads"):
            verify_deferred_mesh_audit_bundle(
                missing_payloads,
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
            )

        tampered_origin = json.loads(json.dumps(bundle))
        tampered_origin["origin_receipt_hash"] = DIGEST_B
        tampered_origin["bundle_hash"] = mesh_deferred_audit_bundle_hash(tampered_origin)
        with pytest.raises(RuntimeError, match="origin_receipt_hash"):
            verify_deferred_mesh_audit_bundle(
                tampered_origin,
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_resolves_deferred_decode_audit_bundle(tmp_path, monkeypatch):
    # These fixtures model a mesh whose llama.cpp build has no boundary
    # capture; the chain requirement is default-on in production, so the
    # capture-less flow needs the explicit opt-out.
    monkeypatch.setenv("VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN", "0")

    trace_enable_file = tmp_path / ".capture-enabled"
    reference_trace = _write_decode_lm_head_trace(
        tmp_path / "manifest-source",
        token_id=101,
    )
    manifest = _gguf_manifest_for_trace(reference_trace)
    manifest_path = tmp_path / "model.gguf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=lambda _payload: _write_decode_lm_head_trace(tmp_path, token_id=101)
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_sample_bps=0,
        decode_audit_bps=1000,
    )
    host, port = worker.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "post hoc decode audit"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-deferred-decode-bundle",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        sampled_randomness = _deferred_randomness_for_sample(receipt, sampled=True)
        decision = deferred_audit_decision(receipt, randomness=sampled_randomness)
        assert decision["proof_sampled"] is False
        assert decision["decode_audit_sampled"] is True
        assert receipt["proof_required"] is False
        assert receipt["decode_audit_required"] is False
        assert payload["completion_token_ids"] == [101]

        bundle = post_json(
            f"http://{host}:{port}/v1/mesh/proof/deferred-audit",
            {
                "artifact": payload,
                "openai_request": request,
                "deferred_randomness": sampled_randomness,
            },
        )

        assert _calls_match_with_derived_seed(calls, [_verified_backend_request(request)])
        assert bundle["decision"]["decode_audit_sampled"] is True
        assert bundle["audit_receipt"]["proof_required"] is True
        assert bundle["audit_receipt"]["proof_sampled"] is False
        assert bundle["audit_receipt"]["decode_audit_required"] is True
        assert bundle["audit_receipt"]["decode_audit_positions"] == [0]
        assert bundle["audit_receipt"]["decode_audit_verified"] is True
        assert bundle["audit_receipt"]["decode_audit_completion_token_ids"] == [101]
        assert bundle["proof_payloads"][0]["decode_audit_openings"][0]["token_id"] == 101
        assert verify_deferred_mesh_audit_bundle(
            bundle,
            payload,
            request,
            spec=mesh_spec,
            member_index=1,
        )

        tampered_decode = json.loads(json.dumps(bundle))
        tampered_decode["proof_payloads"][0]["decode_audit_openings"][0][
            "argmax_token_id"
        ] = 100
        tampered_decode["bundle_hash"] = mesh_deferred_audit_bundle_hash(tampered_decode)
        with pytest.raises(RuntimeError, match="proof verification|decode audit"):
            verify_deferred_mesh_audit_bundle(
                tampered_decode,
                payload,
                request,
                spec=mesh_spec,
                member_index=1,
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_exposes_embedded_ggml_proof_endpoint(tmp_path):
    trace = _write_ggml_mul_mat_trace(tmp_path, created_unix_ns=1)
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        proof_trace_dir=trace.path.parent,
        proof_tolerance_abs=1e-6,
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/proof/receipt",
            {
                "receipt_context": _receipt_context_for_trace(
                    mesh_spec,
                    worker_endpoint,
                )
            },
        )

        assert payload["proof_mode"] == VERATHOS_GGML_GEMM_PROOF_MODE
        assert payload["verified"] is False
        assert payload["proof_receipts"][0]["proof_kind"] == "gemm"
        assert payload["proof_receipts"][0]["op_type"] == "GGML_OP_MUL_MAT"
    finally:
        _shutdown_server(worker, worker_thread)


def test_mesh_worker_sanitizes_and_signs_its_own_opaque_stage_receipt(tmp_path):
    trace = _write_ggml_mul_mat_trace(tmp_path, created_unix_ns=1)
    gguf_manifest = _gguf_manifest_for_trace(trace)
    gguf_manifest_path = tmp_path / "gguf-manifest.json"
    gguf_manifest_path.write_text(json.dumps(gguf_manifest), encoding="utf-8")
    worker_endpoint = "http://private-worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    _make_coordinator_orchestration_only(mesh_spec)
    stage_key = Keypair.create_from_uri("//MeshControlOpaqueStageWorker")
    member = mesh_spec.members[1]
    member.hotkey = stage_key.ss58_address
    member.proof_key = stage_key.ss58_address
    mesh_spec.model_tensor_manifest_root = gguf_manifest["tensor_manifest_root"]
    mesh_spec.proof_trace_manifest_format = "compact-raw-v3"
    mesh_spec.validate()
    capability = CapabilityAd(
        uid=member.uid,
        hotkey=member.hotkey,
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        server_role="worker",
        proof_trace_dir=trace.path.parent,
        proof_gguf_manifest_path=gguf_manifest_path,
        proof_tolerance_abs=1e-6,
        proof_trace_candidates_per_request=1024,
        stage_proof_key=stage_key.ss58_address,
        stage_receipt_signer=lambda body_hash: (
            sign_stage_proof_receipt_body_hash(
                body_hash,
                stage_key,
                expected_proof_key=stage_key.ss58_address,
            )
        ),
    )
    context = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    context.update(
        {
            "hotkey": member.hotkey,
            "layer_start": 0,
            "layer_end": mesh_spec.total_layers,
            "model_tensor_manifest_root": gguf_manifest["tensor_manifest_root"],
            "verification_snapshot_hash": "11" * 32,
            "stage_id": "stg_" + "22" * 16,
            "stage_proof_key": stage_key.ss58_address,
            "stage_proof_key_scheme": "sr25519",
            "stage_proof_commitment": "33" * 32,
            "model_index": 26,
            "model_total_layers": mesh_spec.total_layers,
            "proof_gate_hash": "44" * 32,
            "proof_beacon": "45" * 32,
            "proof_policy_version": 1,
            "proof_policy_profile": "gguf_mesh_v1",
            "proof_receipt_format": "opaque_stage_v2",
            "proof_trace_manifest_format": "compact-raw-v3",
            "proof_sample_bps": 10_000,
            "proof_sample_denominator": 10_000,
            "proof_ops_per_request": 1,
            "proof_trace_candidates_per_request": 1024,
            "proof_challenge_kind": "inline_every_request_v1",
            "proof_deferred": False,
            "verified_sampler_required": True,
            "verified_sampler_mode": (
                "deterministic_no_penalty_top_k_1_seed_0_prompt_cache_v2"
            ),
            "verified_sampler_controls_hash": "55" * 32,
            "decode_audit_mode": VERATHOS_GGUF_DECODE_AUDIT_MODE,
            "decode_audit_bps": 1_000,
            "decode_audit_top_k": 8,
            "decode_audit_stage_index": member.stage_index,
            "proof_trace_scope": "op_manifest_challenge_v1",
        }
    )
    host, port = worker.server_address
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/proof/receipt",
            {"receipt_context": context, "include_proof": True},
        )
        public = MeshStageProofReceipt.from_dict(payload["proof_receipts"][0])

        assert public.stage_id == context["stage_id"]
        assert public.verification_snapshot_hash == context[
            "verification_snapshot_hash"
        ]
        assert verify_stage_proof_receipt_signature(
            public.body_hash().hex(),
            public.signature,
            stage_key.ss58_address,
            "sr25519",
        )
        serialized = json.dumps(payload, sort_keys=True)
        assert "trace_paths" not in payload
        assert worker_endpoint not in serialized
        assert stage_key.ss58_address not in serialized
        assert "uid" not in payload["proof_receipts"][0]
        assert "hotkey" not in payload["proof_receipts"][0]
        assert "endpoint" not in payload["proof_receipts"][0]
    finally:
        _shutdown_server(worker, worker_thread)


def test_snapshot_bound_coordinator_only_aggregates_worker_signed_receipts(
    tmp_path,
    monkeypatch,
):
    import verallm.mesh.worker as mesh_worker_mod

    monkeypatch.setattr(
        mesh_worker_mod,
        "mesh_decode_audit_sample_value",
        lambda _beacon: 9_000,
    )
    # Decode audit is pinned at the full rate so its wire value cannot
    # identify a canary, which short-circuits the sampling gate entirely.
    # This fixture therefore carries a real final-projection op instead of
    # stubbing the gate off.
    remote_root = tmp_path / "remote-snapshot"
    seed_trace = _write_ggml_mul_mat_trace(remote_root, created_unix_ns=1)
    seed_lm_head = _write_decode_lm_head_trace(
        remote_root, created_unix_ns=2, name="seed-lm-head", op_index=1,
    )
    gguf_manifest = _gguf_manifest_for_traces(seed_trace, seed_lm_head)
    manifest_path = remote_root / "gguf-manifest.json"
    manifest_path.write_text(json.dumps(gguf_manifest), encoding="utf-8")
    seed_trace.path.unlink()
    seed_lm_head.path.unlink()

    remote_capture_file = remote_root / ".capture-enabled"

    def on_request(_payload):
        stamp = time.time_ns()
        _write_ggml_mul_mat_trace(remote_root, created_unix_ns=stamp)
        _write_decode_lm_head_trace(
            remote_root,
            created_unix_ns=stamp + 1,
            name=f"request-lm-head-{stamp}",
            op_index=1,
        )

    backend, backend_thread, backend_url, _calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_key = Keypair.create_from_uri("//MeshControlSnapshotCoordinator")
    stage_key = Keypair.create_from_uri("//MeshControlSnapshotStage")
    validator_key = Keypair.create_from_uri("//MeshControlSnapshotValidator")
    validator_allowlist_path = tmp_path / "validators.json"
    validator_allowlist_path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()),
                "netuid": 405,
                "validators": [
                    {
                        "uid": 7,
                        "hotkey_ss58": validator_key.ss58_address,
                        "stake": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    coordinator_evm = Account.from_key(bytes.fromhex("61" * 32))
    coordinator_endpoint = "http://coord.private:9338"
    remote_port = _free_port()
    remote_endpoint = f"http://127.0.0.1:{remote_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=remote_endpoint,
        model_tensor_manifest_root=gguf_manifest["tensor_manifest_root"],
    )
    _make_coordinator_orchestration_only(mesh_spec)
    mesh_spec.coordinator_hotkey = coordinator_key.ss58_address
    mesh_spec.members[0].hotkey = coordinator_key.ss58_address
    mesh_spec.members[1].hotkey = stage_key.ss58_address
    mesh_spec.members[1].proof_key = stage_key.ss58_address
    mesh_spec.tokenizer_hash = "62" * 32
    mesh_spec.proof_trace_manifest_format = "compact-raw-v3"
    mesh_spec.max_context_len = 32_768
    mesh_spec.validate()
    now = int(time.time())
    policy = MeshVerificationPolicy(
        profile="gguf_mesh_v1",
        trace_manifest_format="compact-raw-v3",
        base_proof_sample_bps=10_000,
        organic_decode_sample_bps=10_000,
        canary_decode_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1024,
        deferred_proof_enabled=False,
    )
    snapshot = sign_mesh_verification_snapshot(
        build_mesh_verification_snapshot(
            mesh_spec,
            coordinator=MeshCoordinatorIdentity(
                chain_id=945,
                netuid=405,
                coordinator_uid=mesh_spec.coordinator_uid,
                coordinator_hotkey=coordinator_key.ss58_address,
                coordinator_evm_address=coordinator_evm.address.lower(),
                model_index=26,
            ),
            policy=policy,
            generation=1,
            epoch=mesh_spec.epoch,
            issued_at_unix=now - 5,
            expires_at_unix=now + 600,
            stage_bindings=(
                MeshVerificationStageBinding(
                    stage_index=mesh_spec.members[1].stage_index,
                    stage_identity_commitment="63" * 32,
                    proof_key_scheme="sr25519",
                    proof_key=stage_key.ss58_address,
                    proof_commitment="64" * 32,
                ),
            ),
        ),
        coordinator_key,
    )
    coordinator_capability = CapabilityAd(
        uid=mesh_spec.coordinator_uid,
        hotkey=coordinator_key.ss58_address,
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_capability = CapabilityAd(
        uid=mesh_spec.members[1].uid,
        hotkey=stage_key.ss58_address,
        endpoint=remote_endpoint,
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
        rpc_endpoint=mesh_spec.members[1].rpc_endpoint,
        proof_endpoint=remote_endpoint,
    )
    remote_worker, remote_thread = serve_worker_in_thread(
        capability=remote_capability,
        host="127.0.0.1",
        port=remote_port,
        mesh_spec=mesh_spec,
        server_role="worker",
        proof_trace_enable_file=remote_capture_file,
        proof_trace_dir=seed_trace.path.parent,
        proof_gguf_manifest_path=manifest_path,
        proof_tolerance_abs=1e-6,
        proof_trace_candidates_per_request=1024,
        stage_proof_key=stage_key.ss58_address,
        stage_receipt_signer=lambda body_hash: (
            sign_stage_proof_receipt_body_hash(
                body_hash,
                stage_key,
                expected_proof_key=stage_key.ss58_address,
            )
        ),
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        server_role="coordinator",
        backend_url=backend_url,
        require_proof=True,
        proof_sample_bps=policy.base_proof_sample_bps,
        decode_audit_bps=policy.organic_decode_sample_bps,
        proof_ops_per_request=policy.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            policy.proof_trace_candidates_per_request
        ),
        evm_address=coordinator_evm.address,
        evm_private_key=coordinator_evm.key.hex(),
        receipt_signer=lambda receipt_hash: sign_receipt_hash(
            receipt_hash,
            coordinator_key,
        ),
        verification_snapshot_loader=lambda: snapshot.to_dict(),
        validator_auth_enabled=True,
        validator_allowlist_path=validator_allowlist_path,
        require_validator_nonce=True,
    )
    host, port = coordinator.server_address
    challenge_nonce = "65" * 32
    validator_request_id = "66" * 32
    request = {
        "model": mesh_spec.model_id,
        "messages": [{"role": "user", "content": "prove privately"}],
        "stream": False,
        "verathos": {
            "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
            "challenge_nonce_commitment": (
                mesh_validator_challenge_nonce_commitment(
                    challenge_nonce,
                    validator_request_id=validator_request_id,
                    verification_snapshot_hash=snapshot.snapshot_hash_hex(),
                )
            ),
            "validator_request_id": validator_request_id,
        },
    }
    inference_path = "/v1/mesh/inference"
    inference_body = {
        "request_id": "req-snapshot-worker-signed",
        "openai_request": request,
        "require_proof": True,
    }
    try:
        origin = post_json(
            f"http://{host}:{port}{inference_path}",
            inference_body,
            headers=_validator_post_headers(
                validator_key,
                path=inference_path,
                payload=inference_body,
            ),
        )
        origin_receipt = origin["receipt"]
        assert origin_receipt["proof_postcommit"] is True
        assert origin_receipt["proof_postcommit_finalized"] is False
        assert origin_receipt["proof_beacon"] == ""
        assert origin_receipt["proof_receipt_count"] == 0
        assert "proof_receipts" not in origin
        postcommit_body = {
            "validator_request_id": validator_request_id,
            "origin_receipt_hash": origin_receipt["receipt_hash"],
            "mesh_response_commitment_hash": origin_receipt[
                "mesh_response_commitment_hash"
            ],
            "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
            "challenge_nonce": challenge_nonce,
        }
        artifact = post_json(
            f"http://{host}:{port}{POSTCOMMIT_AUDIT_PATH}",
            postcommit_body,
            headers=_validator_post_headers(
                validator_key,
                path=POSTCOMMIT_AUDIT_PATH,
                payload=postcommit_body,
            ),
        )
        assert verify_mesh_postcommit_artifact(
            artifact,
            origin,
            request,
            challenge_nonce=challenge_nonce,
            spec=mesh_spec,
            expected_coordinator_hotkey=coordinator_key.ss58_address,
            expected_coordinator_uid=mesh_spec.coordinator_uid,
            expected_validator_hotkey=validator_key.ss58_address,
            require_coordinator_signature=True,
            verification_snapshot=snapshot,
        )
        receipt = artifact["receipt"]
        public_receipts = [
            MeshStageProofReceipt.from_dict(entry)
            for entry in artifact["proof_receipts"]
        ]

        assert receipt["proof_receipt_format"] == "opaque_stage_v2"
        assert receipt["proof_trace_scope"] == "op_manifest_challenge_v1"
        # One receipt for the sampled proof op, plus one for the decode audit
        # when the sampled op is not already the final projection. Which of
        # the two happens is beacon-driven, so assert the aggregate is
        # self-consistent rather than pinning the count.
        assert receipt["proof_receipt_count"] == len(public_receipts)
        assert 1 <= len(public_receipts) <= 2
        assert receipt["proof_receipt_verified"] is True
        for public in public_receipts:
            assert public.stage_id == snapshot.stages[0].stage_id
            assert verify_stage_proof_receipt_signature(
                public.body_hash().hex(),
                public.signature,
                stage_key.ss58_address,
                "sr25519",
            )
        serialized = json.dumps(artifact, sort_keys=True)
        assert remote_endpoint not in serialized
        assert stage_key.ss58_address not in serialized
        assert "trace_paths" not in serialized
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_arms_remote_rpc_trace_capture_during_backend_request(tmp_path):
    proof, proof_thread, proof_url, _ = _fake_proof_adapter()
    remote_capture_file = tmp_path / "remote" / ".capture-enabled"
    capture_enabled_during_backend = []
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=lambda _payload: capture_enabled_during_backend.append(
            remote_capture_file.exists()
        )
    )
    coordinator_endpoint = "http://coord.local:9338"
    remote_port = _free_port()
    remote_endpoint = f"http://127.0.0.1:{remote_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=remote_endpoint,
    )
    _make_coordinator_orchestration_only(mesh_spec)
    mesh_spec.members[1].proof_endpoint = proof_url
    mesh_spec.validate()
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=remote_endpoint,
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
        rpc_endpoint="worker.local:50052",
        proof_endpoint=proof_url,
    )
    remote_worker, remote_thread = serve_worker_in_thread(
        capability=remote_capability,
        host="127.0.0.1",
        port=remote_port,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=remote_capture_file,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "remote capture"}],
        "stream": False,
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-remote-capture",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]

        assert _calls_match_with_derived_seed(calls, [request])
        assert payload.get("completion_token_ids", []) == []
        assert receipt["completion_token_count"] == 1
        assert capture_enabled_during_backend == [True]
        assert not remote_capture_file.exists()
        assert receipt["proof_required"] is True
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is False
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(proof, proof_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_coordinator_collects_remote_member_proof_endpoint(tmp_path):
    remote_root = tmp_path / "remote"
    remote_capture_file = remote_root / ".capture-enabled"
    trace_dir = remote_root / "ggml-traces"
    capture_enabled_during_backend = []
    traces = []

    def on_request(_payload):
        capture_enabled_during_backend.append(remote_capture_file.exists())
        traces.append(_write_ggml_mul_mat_trace(remote_root))

    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        on_request=on_request
    )
    coordinator_endpoint = "http://coord.local:9338"
    remote_port = _free_port()
    remote_endpoint = f"http://127.0.0.1:{remote_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=remote_endpoint,
    )
    _make_coordinator_orchestration_only(mesh_spec)
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
    )
    remote_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=remote_endpoint,
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        cached_model_package_hashes=[DIGEST_A],
        rpc_endpoint="worker.local:50052",
        proof_endpoint=remote_endpoint,
    )
    remote_worker, remote_thread = serve_worker_in_thread(
        capability=remote_capability,
        host="127.0.0.1",
        port=remote_port,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=remote_capture_file,
        proof_trace_dir=trace_dir,
        proof_tolerance_abs=1e-6,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        allow_loopback_dev_validator_routes=True,
    )
    host, port = coordinator.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "remote proof"}],
        "stream": False,
        "verathos": {"proof_tier": "hard"},
    }
    try:
        payload = post_json(
            f"http://{host}:{port}/v1/mesh/inference",
            {
                "request_id": "req-remote-member-proof",
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "openai_request": request,
                "require_proof": True,
            },
        )
        receipt = payload["receipt"]
        proof = payload["proof_receipts"][0]

        assert _calls_match_with_derived_seed(calls, [request])
        assert receipt["completion_token_count"] == 1
        assert len(traces) == 1
        assert capture_enabled_during_backend == [True]
        assert not remote_capture_file.exists()
        assert receipt["proof_required"] is True
        assert receipt["proof_mode"] == VERATHOS_GGML_GEMM_PROOF_MODE
        assert receipt["proof_receipt_verified"] is True
        assert receipt["verified"] is True
        assert receipt["proof_receipt_count"] == 1
        assert proof["endpoint"] == remote_endpoint
        assert proof["stage_index"] == 1
        assert proof["layer_start"] == 0
        assert proof["layer_end"] == 4
        assert proof["proof_kind"] == "gemm"
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(remote_worker, remote_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_worker_rejects_receipt_only_when_proof_is_required():
    backend, backend_thread, backend_url, _ = _fake_openai_backend()
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
    )
    host, port = worker.server_address
    try:
        with pytest.raises(RuntimeError, match="proof endpoint is configured"):
            post_json(
                f"http://{host}:{port}/v1/mesh/inference",
                {
                    "request_id": "req-proof-missing",
                    "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                    "openai_request": {
                        "model": "model",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": False,
                    },
                    "require_proof": True,
                },
            )
    finally:
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_chat_completions_busy_returns_retryable_503():
    """vLLM-parity admission: with every generation slot busy (default
    llama_n_parallel=1 here) the mesh answers an immediate 503 with type
    slots_busy instead of queueing behind an unbounded wait, and admits
    again once the slot frees. Over-limit prompts are the OTHER failure
    (llama's own 400 exceed_context_size_error); this gate is about
    concurrency, so a router can fail over instead of stalling."""

    release = threading.Event()

    def hold(_payload) -> None:
        release.wait(10.0)

    backend, backend_thread, backend_url, backend_calls = _fake_openai_backend(
        content="held pong", on_request=hold
    )
    coordinator_port = _free_port()
    worker_port = _free_port()
    coordinator_endpoint = f"http://127.0.0.1:{coordinator_port}"
    worker_endpoint = f"http://127.0.0.1:{worker_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=worker_endpoint,
    )
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=worker_capability,
        host="127.0.0.1",
        port=worker_port,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        host="127.0.0.1",
        port=coordinator_port,
        mesh_spec=mesh_spec,
        allow_loopback_dev_validator_routes=True,
    )
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "hold the slot"}],
        "stream": False,
    }
    try:
        results: dict[str, Any] = {}

        def occupy() -> None:
            results["first"] = post_json(
                f"{coordinator_endpoint}/v1/chat/completions", request
            )

        holder = threading.Thread(target=occupy)
        holder.start()
        # Deterministic ordering: only fire the second request once the
        # holder's call demonstrably reached the backend (and is parked
        # in hold()), so the slot is provably occupied.
        deadline = time.monotonic() + 5.0
        while not backend_calls and time.monotonic() < deadline:
            time.sleep(0.05)
        assert backend_calls, "holder request never reached the backend"
        with pytest.raises(RuntimeError) as rejected:
            post_json(
                f"{coordinator_endpoint}/v1/chat/completions", request
            )
        assert "HTTP 503" in str(rejected.value)
        assert "slots_busy" in str(rejected.value)
        assert "busy" in str(rejected.value)
        release.set()
        holder.join(timeout=15)
        assert (
            results["first"]["choices"][0]["message"]["content"]
            == "held pong"
        )
        # The slot freed: the same request is admitted and served again.
        after = post_json(
            f"{coordinator_endpoint}/v1/chat/completions", request
        )
        assert after["choices"][0]["message"]["content"] == "held pong"
    finally:
        release.set()
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_mesh_coordinator_routes_to_worker_and_verifies_receipt():
    backend, backend_thread, backend_url, _ = _fake_openai_backend(content="mesh pong")
    coordinator_port = _free_port()
    worker_port = _free_port()
    coordinator_endpoint = f"http://127.0.0.1:{coordinator_port}"
    worker_endpoint = f"http://127.0.0.1:{worker_port}"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint=coordinator_endpoint,
        worker_endpoint=worker_endpoint,
    )
    coordinator_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=coordinator_endpoint,
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker_capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=worker_capability,
        host="127.0.0.1",
        port=worker_port,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
    )
    coordinator, coordinator_thread = serve_worker_in_thread(
        capability=coordinator_capability,
        host="127.0.0.1",
        port=coordinator_port,
        mesh_spec=mesh_spec,
        allow_loopback_dev_validator_routes=True,
    )
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "route"}],
        "stream": False,
    }
    try:
        response = post_json(f"{coordinator_endpoint}/v1/chat/completions", request)
        mesh = response["verathos_mesh"]
        receipt = mesh["receipt"]

        assert response["choices"][0]["message"]["content"] == "mesh pong"
        assert mesh["receipt_verified"] is True
        assert mesh["runtime"] == "llama_cpp_rpc"
        assert mesh["rpc_endpoints"] == []
        assert "worker.local:50052" not in json.dumps(mesh)
        assert mesh["rpc_plan_hash"] == rpc_plan_from_mesh(mesh_spec).plan_hash_hex()
        assert (
            mesh["mesh_response_commitment_hash"]
            == _mesh_response_commitment_hash(receipt)
        )
        assert mesh["verified"] is False
        assert mesh["proof_mode"] == "llama_cpp_rpc_receipt_v1"
        assert receipt["mesh_id"] == mesh_spec.mesh_id
        assert receipt["mesh_spec_hash"] == mesh_spec.spec_hash_hex()
        assert receipt["stage_assignment_hash"] == mesh_spec.stage_assignment_hash_hex()
        assert receipt["rpc_plan_hash"] == rpc_plan_from_mesh(mesh_spec).plan_hash_hex()
        assert receipt["endpoint"] == worker_endpoint
        assert receipt["stage_index"] == 1
        assert receipt["layer_start"] == 2
        assert receipt["layer_end"] == 4
        assert receipt["request_hash"] == _payload_hash(request)

        response_without_mesh = dict(response)
        response_without_mesh.pop("verathos_mesh")
        assert receipt["response_hash"] == _payload_hash(response_without_mesh)
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
        assert verify_mesh_inference_artifact(
            mesh,
            request,
            openai_response=response_without_mesh,
            spec=mesh_spec,
            member_index=1,
        )
        tampered_response = json.loads(json.dumps(response_without_mesh))
        tampered_response["choices"][0]["message"]["content"] = "tampered"
        with pytest.raises(RuntimeError, match="response_hash"):
            verify_mesh_inference_artifact(
                mesh,
                request,
                openai_response=tampered_response,
                spec=mesh_spec,
                member_index=1,
            )
    finally:
        _shutdown_server(coordinator, coordinator_thread)
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_cli_infer_posts_non_streaming_chat_request(capsys):
    backend, backend_thread, backend_url, calls = _fake_openai_backend(content="cli pong")
    try:
        mesh_cli_main(
            [
                "infer",
                backend_url,
                "--model",
                "model",
                "--prompt",
                "hello",
            ]
        )
        response = json.loads(capsys.readouterr().out)
        assert response["choices"][0]["message"]["content"] == "cli pong"
        assert calls[0] == {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }

        mesh_cli_main(
            [
                "infer",
                backend_url,
                "--model",
                "model",
                "--prompt",
                "hello again",
                "--content",
            ]
        )
        assert capsys.readouterr().out == "cli pong\n"
    finally:
        _shutdown_server(backend, backend_thread)


def test_cli_infer_can_attach_validator_nonce(capsys):
    backend, backend_thread, backend_url, calls = _fake_openai_backend()
    try:
        mesh_cli_main(
            [
                "infer",
                backend_url,
                "--model",
                "model",
                "--prompt",
                "hello",
                "--validator-nonce",
                "33" * 32,
            ]
        )
        capsys.readouterr()
        assert calls[0] == {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "verathos": {"validator_nonce": "33" * 32},
        }
    finally:
        _shutdown_server(backend, backend_thread)


def test_cli_verify_artifact_checks_saved_mesh_payload(tmp_path, capsys):
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "verify artifact"}],
        "stream": False,
    }
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "verified"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    member = mesh_spec.members[1]
    plan = rpc_plan_from_mesh(mesh_spec)
    receipt = {
        "request_id": "req-cli-verify",
        "runtime": "openai_backend",
        "mesh_id": mesh_spec.mesh_id,
        "mesh_spec_hash": mesh_spec.spec_hash_hex(),
        "stage_assignment_hash": mesh_spec.stage_assignment_hash_hex(),
        "rpc_endpoints": [],
        "rpc_plan_hash": plan.plan_hash_hex(),
        "model_package_hash": mesh_spec.model_package_hash,
        "model_tensor_manifest_root": mesh_spec.model_tensor_manifest_root,
        "uid": member.uid,
        "hotkey": member.hotkey,
        "endpoint": member.endpoint,
        "stage_index": member.stage_index,
        "layer_start": member.layers.start,
        "layer_end": member.layers.end,
        "request_hash": _payload_hash(request),
        "response_hash": _payload_hash(response),
        "proof_configured_required": False,
        "proof_capture_required": False,
        "proof_required": False,
        "proof_sampled": False,
        "proof_receipt_root": "",
        "proof_receipt_count": 0,
        "proof_receipt_verified": False,
        "verified": False,
        "proof_mode": "receipt_only",
    }
    receipt["mesh_response_commitment_hash"] = _mesh_response_commitment_hash(receipt)
    receipt["receipt_hash"] = _receipt_hash(receipt)
    request_path = save_json(tmp_path / "request.json", request)
    artifact_path = save_json(tmp_path / "artifact.json", {"response": response, "receipt": receipt})
    mesh_path = save_json(tmp_path / "mesh.json", mesh_spec.to_dict())

    mesh_cli_main(
        [
            "verify-artifact",
            str(artifact_path),
            "--request",
            str(request_path),
            "--mesh",
            str(mesh_path),
            "--member-index",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["request_id"] == "req-cli-verify"
    assert payload["proof_required"] is False


def test_cli_infer_stream_prints_content_chunks(capsys):
    backend, backend_thread, backend_url, calls = _fake_openai_backend(content="cli stream pong")
    try:
        mesh_cli_main(
            [
                "infer",
                backend_url,
                "--model",
                "model",
                "--prompt",
                "hello",
                "--stream",
                "--content",
            ]
        )
        assert capsys.readouterr().out == "cli stream pong\n"
        assert calls[0] == {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    finally:
        _shutdown_server(backend, backend_thread)


def test_llama_cpp_rpc_plan_binds_worker_rpc_endpoint(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    path = save_json(tmp_path / "mesh.json", spec.to_dict())

    plan = rpc_plan_from_mesh(spec)
    assert plan.rpc_endpoints == ["worker.local:50052"]
    assert plan.rpc_split_weights == [1]
    assert plan.rpc_layer_ranges == [{"start": 2, "end": 4}]
    assert plan.rpc_arg == "worker.local:50052"
    assert plan.tensor_split_arg == "1"
    assert plan.to_dict()["rpc_members"] == [
        {
            "rpc_endpoint": "worker.local:50052",
            "rpc_split_weight": 1,
            "layers": {"start": 2, "end": 4},
        }
    ]

    mesh_cli_main(
        [
            "rpc-plan",
            str(path),
            "--model",
            "/models/model.gguf",
            "--device",
            "RPC0",
            "--n-gpu-layers",
            "99",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_arg"] == "worker.local:50052"
    assert payload["tensor_split_arg"] == "1"
    assert payload["rpc_split_weights"] == [1]
    assert payload["rpc_layer_ranges"] == [{"start": 2, "end": 4}]
    assert payload["llama_server_command"] == [
        "llama-server",
        "--model",
        "/models/model.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--reasoning-format",
        "deepseek",
        "--rpc",
        "worker.local:50052",
        "--device",
        "RPC0",
        "--n-gpu-layers",
        "99",
        "--tensor-split",
        "1",
        "--alias",
        "model",
    ]
    with pytest.raises(SystemExit, match="committed RPC plan"):
        mesh_cli_main(
            [
                "rpc-plan",
                str(path),
                "--model",
                "/models/model.gguf",
                "--tensor-split",
                "2",
            ]
        )


def test_llama_cpp_rpc_plan_binds_weight_and_range_and_rejects_duplicates():
    spec = MeshSpec(
        mesh_id="mesh-rpc-plan",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=40,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="http://coord.local:9338",
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5WorkerA",
                endpoint="http://worker-a.local:9338",
                rpc_endpoint="worker-a.local:50052",
                rpc_split_weight=24,
                stage_index=1,
                layers=StageRange(0, 14),
            ),
            MeshMember(
                uid=1,
                hotkey="5WorkerB",
                endpoint="http://worker-b.local:9338",
                rpc_endpoint="worker-b.local:50052",
                rpc_split_weight=49,
                stage_index=2,
                layers=StageRange(14, 40),
            ),
        ],
    )
    spec.validate()

    plan = rpc_plan_from_mesh(spec)
    assert plan.rpc_endpoints == [
        "worker-a.local:50052",
        "worker-b.local:50052",
    ]
    assert plan.rpc_split_weights == [24, 49]
    assert plan.rpc_layer_ranges == [
        {"start": 0, "end": 14},
        {"start": 14, "end": 40},
    ]
    assert plan.tensor_split_arg == "24,49"

    weight_hash = plan.plan_hash_hex()
    spec.members[1].rpc_split_weight = 25
    assert rpc_plan_from_mesh(spec).plan_hash_hex() != weight_hash

    spec.members[1].rpc_split_weight = 24
    spec.members[1].layers = StageRange(0, 20)
    spec.members[2].layers = StageRange(20, 40)
    with pytest.raises(ValueError, match="do not match committed tensor split"):
        rpc_plan_from_mesh(spec)

    spec.members[1].layers = StageRange(0, 14)
    spec.members[2].layers = StageRange(14, 40)
    spec.members[2].rpc_endpoint = "tcp://worker-a.local:50052"
    with pytest.raises(ValueError, match="duplicate RPC endpoint"):
        rpc_plan_from_mesh(spec)


def test_all_rpc_cli_binds_device_order_and_full_offload(tmp_path, capsys):
    spec = MeshSpec(
        mesh_id="mesh-all-rpc-runtime",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=40,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="http://coord.local:9338",
                stage_index=0,
                layers=StageRange(0, 0),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5WorkerA",
                endpoint="http://worker-a.local:9338",
                rpc_endpoint="worker-a.local:50052",
                rpc_split_weight=24,
                stage_index=1,
                layers=StageRange(0, 14),
            ),
            MeshMember(
                uid=1,
                hotkey="5WorkerB",
                endpoint="http://worker-b.local:9338",
                rpc_endpoint="worker-b.local:50052",
                rpc_split_weight=49,
                stage_index=2,
                layers=StageRange(14, 40),
            ),
        ],
    )
    spec.validate()
    path = save_json(tmp_path / "mesh.json", spec.to_dict())

    mesh_cli_main(
        [
            "rpc-plan",
            str(path),
            "--model",
            "/models/model.gguf",
            "--device",
            "RPC0,RPC1",
            "--n-gpu-layers",
            "all",
        ]
    )
    command = json.loads(capsys.readouterr().out)["llama_server_command"]
    assert command[command.index("--device") + 1] == "RPC0,RPC1"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--tensor-split") + 1] == "24,49"

    for device, layers, message in (
        ("RPC1,RPC0", "all", "device order"),
        ("RPC0", "all", "device order"),
        ("CUDA0,RPC0,RPC1", "all", "device order"),
        ("RPC0,RPC1", "40", "n-gpu-layers all"),
    ):
        with pytest.raises(SystemExit, match=message):
            mesh_cli_main(
                [
                    "rpc-plan",
                    str(path),
                    "--model",
                    "/models/model.gguf",
                    "--device",
                    device,
                    "--n-gpu-layers",
                    layers,
                ]
            )


def test_cli_rpc_worker_dry_run(capsys):
    mesh_cli_main(
        [
            "rpc-worker",
            "--host",
            "0.0.0.0",
            "--advertise-host",
            "worker.local",
            "--port",
            "50052",
            "--device",
            "CUDA0",
            "--cache",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_endpoint"] == "worker.local:50052"
    assert payload["rpc_worker_command"] == [
        "rpc-server",
        "-H",
        "0.0.0.0",
        "-p",
        "50052",
        "--device",
        "CUDA0",
        "-c",
    ]


def test_cli_rpc_worker_dry_run_normalizes_metal_alias(capsys):
    mesh_cli_main(
        [
            "rpc-worker",
            "--host",
            "0.0.0.0",
            "--port",
            "50052",
            "--device",
            "Metal0",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_worker_command"] == [
        "rpc-server",
        "-H",
        "0.0.0.0",
        "-p",
        "50052",
        "--device",
        "MTL0",
    ]


def test_cli_llama_server_dry_run_uses_mesh_rpc_endpoints(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    path = save_json(tmp_path / "mesh.json", spec.to_dict())

    mesh_cli_main(
        [
            "llama-server",
            "--mesh",
            str(path),
            "--model",
            "/models/model.gguf",
            "--device",
            "RPC0",
            "--n-gpu-layers",
            "99",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_plan"]["rpc_endpoints"] == ["worker.local:50052"]
    assert payload["backend_url"] == "http://127.0.0.1:8080"
    assert "--rpc" in payload["llama_server_command"]
    assert "worker.local:50052" in payload["llama_server_command"]
    assert "--device" in payload["llama_server_command"]
    assert "RPC0" in payload["llama_server_command"]
    split_index = payload["llama_server_command"].index("--tensor-split")
    assert payload["llama_server_command"][split_index + 1] == "1"
    with pytest.raises(SystemExit, match="committed RPC plan"):
        mesh_cli_main(
            [
                "llama-server",
                "--mesh",
                str(path),
                "--model",
                "/models/model.gguf",
                "--tensor-split",
                "2",
                "--dry-run",
            ]
        )


def test_cli_llama_server_dry_run_supports_hf_alias(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    path = save_json(tmp_path / "mesh.json", spec.to_dict())

    mesh_cli_main(
        [
            "llama-server",
            "--mesh",
            str(path),
            "--hf",
            "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
            "--device",
            "RPC0",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["llama_server_command"][:3] == [
        "llama-server",
        "-hf",
        "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    ]
    assert "--model" not in payload["llama_server_command"]
    assert "worker.local:50052" in payload["llama_server_command"]
    split_index = payload["llama_server_command"].index("--tensor-split")
    assert payload["llama_server_command"][split_index + 1] == "1"


def test_cli_serve_llama_dry_run_uses_joined_mesh_state(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-model",
            "/models/model.gguf",
            "--llama-device",
            "RPC0",
            "--llama-n-gpu-layers",
            "all",
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mesh_id"] == spec.mesh_id
    assert payload["rpc_plan"]["rpc_arg"] == "worker.local:50052"
    assert payload["backend_url"] == "http://127.0.0.1:8080"
    device_index = payload["llama_server_command"].index("--device")
    assert payload["llama_server_command"][device_index + 1] == "RPC0"
    ngl_index = payload["llama_server_command"].index("--n-gpu-layers")
    assert payload["llama_server_command"][ngl_index + 1] == "all"
    split_index = payload["llama_server_command"].index("--tensor-split")
    assert payload["llama_server_command"][split_index + 1] == "1"
    assert payload["llama_server_command"][-2:] == ["--alias", "model"]


def test_cli_serve_llama_context_is_derived_from_mesh_and_cannot_be_overridden(
    tmp_path,
    capsys,
):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    spec.max_context_len = 32_768
    spec.validate()
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    base_args = [
        "serve",
        "--mesh",
        str(state_dir),
        "--llama-model",
        "/models/model.gguf",
        "--llama-device",
        "RPC0",
        "--llama-n-gpu-layers",
        "all",
        "--llama-dry-run",
    ]
    mesh_cli_main(base_args)
    command = json.loads(capsys.readouterr().out)["llama_server_command"]
    context_index = command.index("--ctx-size")
    assert command[context_index + 1] == "32768"

    # ABOVE the contract is allowed now: the measured-fit budget serves
    # concurrency headroom (the admission ledger enforces the per-request
    # contract). BELOW the contract stays refused - a smaller unified KV
    # cannot serve the advertised per-request maximum.
    mesh_cli_main(
        [
            *base_args[:-1],
            "--llama-ctx-size",
            "65536",
            "--llama-dry-run",
        ]
    )
    command = json.loads(capsys.readouterr().out)["llama_server_command"]
    context_index = command.index("--ctx-size")
    assert command[context_index + 1] == "65536"

    with pytest.raises(SystemExit, match="at least the mesh max_context_len"):
        mesh_cli_main(
            [
                *base_args[:-1],
                "--llama-ctx-size",
                "16384",
                "--llama-dry-run",
            ]
        )


def test_cli_serve_llama_hf_dry_run_uses_joined_mesh_state(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-hf",
            "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
            "--llama-device",
            "RPC0",
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["llama_server_command"][:3] == [
        "llama-server",
        "-hf",
        "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    ]
    assert "--model" not in payload["llama_server_command"]
    assert payload["rpc_plan"]["rpc_arg"] == "worker.local:50052"


def test_cli_serve_rpc_worker_dry_run(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--rpc-worker",
            "--rpc-device",
            "CUDA0",
            "--rpc-cache",
            "--rpc-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_worker_command"] == [
        "rpc-server",
        "-H",
        "0.0.0.0",
        "-p",
        "50052",
        "--device",
        "CUDA0",
        "-c",
    ]


@requires_native_proof_stack
def test_cli_serve_require_proof_rejects_raw_rpc_worker(tmp_path):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    with pytest.raises(SystemExit, match="raw llama.cpp rpc-server is receipt-only"):
        mesh_cli_main(
            [
                "serve",
                "--mesh",
                str(state_dir),
                "--rpc-worker",
                "--require-proof",
                "--rpc-dry-run",
            ]
        )


@requires_native_proof_stack
def test_cli_serve_require_proof_allows_verathos_rpc_worker_dry_run(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--rpc-worker",
            "--rpc-worker-binary",
            "verathos-rpc-server",
            "--require-proof",
            "--rpc-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rpc_worker_command"][0] == "verathos-rpc-server"
    assert payload["proof_required"] is True
    assert payload["proof_capable_rpc_worker"] is True
    # The rpc graph-end mirror must NOT be armed: it dumps after the whole
    # graph ran, when ggml has reused intermediate buffers, so its mid-graph
    # witnesses fail float recomputation. The rpc-server's own at-execution
    # CUDA/CPU hooks carry every witness.
    assert "VERATHOS_GGML_RPC_GRAPH_TRACE" not in payload["proof_runtime_env"]
    # An rpc-worker is a leaf compute stage: it captures its own candidate
    # witnesses during serve and proves a stored candidate at audit. Base
    # light/hard proofs also need the compact v3 op manifest: slot-view
    # leaves verify against it even at a zero decode-audit rate; decode_audit_bps
    # no longer gates the format).
    assert (
        payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MANIFEST_FORMAT"]
        == "compact-raw-v3"
    )
    # Not a slot-view serve (no decode audit, single slot), so the 8
    # serve-time candidate dumps stay armed.
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] == "8"
    assert (
        payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_ELEMS"] == "4194304"
    )


@requires_native_proof_stack
def test_cli_serve_tail_ring_idle_flush_default(tmp_path, capsys, monkeypatch):
    # The 60ms C-side idle flush splits one generation's tail ring across
    # drains whenever the pipeline stalls longer than a token gap, and a
    # split group cannot cover an audit draw (only a reply-final group maps
    # tail_seq to positions). The serve spawn env must raise it so the
    # reply-end signal is the drain path and idle stays a safety net.
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)
    serve_args = [
        "serve",
        "--mesh",
        str(state_dir),
        "--rpc-worker",
        "--rpc-worker-binary",
        "verathos-rpc-server",
        "--require-proof",
        "--rpc-dry-run",
    ]

    mesh_cli_main(serve_args)
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["proof_runtime_env"]["VERATHOS_GGML_TAIL_RING_FLUSH_MS"]
        == "750"
    )

    # An explicit operator value wins over the default.
    monkeypatch.setenv("VERATHOS_GGML_TAIL_RING_FLUSH_MS", "90")
    mesh_cli_main(serve_args)
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["proof_runtime_env"]["VERATHOS_GGML_TAIL_RING_FLUSH_MS"]
        == "90"
    )
    monkeypatch.delenv("VERATHOS_GGML_TAIL_RING_FLUSH_MS")

    # The tail-capture kill switch reverts the whole tier: ring off, and no
    # flush tuning is injected.
    monkeypatch.setenv("VERATHOS_MESH_LIGHT_TAIL_CAPTURE", "0")
    mesh_cli_main(serve_args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TAIL_RING"] == "0"
    assert (
        "VERATHOS_GGML_TAIL_RING_FLUSH_MS"
        not in payload["proof_runtime_env"]
    )
    monkeypatch.delenv("VERATHOS_MESH_LIGHT_TAIL_CAPTURE")

    # Concurrent serving (--parallel > 1): the ring's instance-order
    # position binding is void across interleaved slots, so the ring goes
    # OFF deterministically and audit-drawn lights take the certified
    # probe path instead.
    mesh_cli_main(
        serve_args
        + [
            "--llama-extra-arg=--parallel",
            "--llama-extra-arg=8",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TAIL_RING"] == "0"
    assert (
        "VERATHOS_GGML_TAIL_RING_FLUSH_MS"
        not in payload["proof_runtime_env"]
    )


@requires_native_proof_stack
def test_cli_serve_require_proof_with_llama_requires_proof_source(tmp_path):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    for member in spec.members:
        member.proof_endpoint = ""
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    with pytest.raises(
        SystemExit,
        match="requires --proof-url, --proof-trace-dir, or member proof_endpoint",
    ):
        mesh_cli_main(
            [
                "serve",
                "--mesh",
                str(state_dir),
                "--llama-model",
                "/models/model.gguf",
                "--require-proof",
                "--llama-dry-run",
            ]
        )


@requires_native_proof_stack
def test_cli_serve_require_proof_allows_pending_joined_proof_endpoints(tmp_path, capsys):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://coord.local:9338",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-model",
            "/models/model.gguf",
            "--require-proof",
            "--llama-min-rpc-workers",
            "1",
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["proof_required"] is True
    assert payload["proof_collection"] == "none"


@requires_native_proof_stack
def test_cli_serve_require_proof_with_llama_allows_member_proof_endpoints(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-model",
            "/models/model.gguf",
            "--require-proof",
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["proof_required"] is True
    assert payload["proof_collection"] == "member_endpoints"


@requires_native_proof_stack
def test_cli_serve_require_proof_with_llama_allows_embedded_trace_dir(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)
    trace_dir = tmp_path / "traces"

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-model",
            "/models/model.gguf",
            "--require-proof",
            "--proof-trace-dir",
            str(trace_dir),
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["proof_required"] is True
    assert payload["proof_trace_dir"] == str(trace_dir)
    assert payload["proof_collection"] == "embedded"


@requires_native_proof_stack
def test_cli_serve_decode_audit_keeps_generic_trace_element_cap(tmp_path, capsys):
    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)
    trace_dir = tmp_path / "traces"

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-model",
            "/models/model.gguf",
            "--require-proof",
            "--proof-trace-dir",
            str(trace_dir),
            "--proof-sample-bps",
            "0",
            "--decode-audit-bps",
            "1000",
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["decode_audit_bps"] == 1000
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_ELEMS"] == "262144"
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] == "0"


@requires_native_proof_stack
def test_cli_serve_llama_hf_scales_trace_element_cap_from_cached_gguf(
    tmp_path, capsys, monkeypatch
):
    import verallm.mesh.cli as cli_mod

    spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint="http://worker.local:9338",
    )
    state_dir, _, _ = create_mesh_state(spec=spec, root=tmp_path)
    trace_dir = tmp_path / "traces"
    manifest = tmp_path / "model.gguf-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    snapshot = (
        tmp_path
        / "hf"
        / "hub"
        / "models--Qwen--Qwen2.5-7B-Instruct-GGUF"
        / "snapshots"
        / "abc"
    )
    snapshot.mkdir(parents=True)
    gguf = snapshot / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    gguf.write_bytes(b"not a real gguf")
    (snapshot / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf").write_bytes(b"")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    def fake_suggest(path):
        assert Path(path) == gguf
        return 3_670_016

    # The CLI imports this lazily from its home module (heavy imports
    # left the interactive startup path), so patch the source.
    import verallm.mesh.gguf_manifest as gguf_manifest_mod

    monkeypatch.setattr(
        gguf_manifest_mod,
        "suggest_ggml_trace_max_elems_from_gguf_model",
        fake_suggest,
    )

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--llama-hf",
            "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
            "--require-proof",
            "--proof-trace-dir",
            str(trace_dir),
            "--proof-gguf-manifest",
            str(manifest),
            "--llama-dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    # Every proof-capturing compute stage (including a coordinator that
    # computes some layers alongside remote workers) keeps its 8 candidate
    # witnesses captured during serve and proved as a stored candidate at
    # audit, and base proofs additionally carry the compact v3 op
    # manifest.
    assert (
        payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MANIFEST_FORMAT"]
        == "compact-raw-v3"
    )
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] == "8"
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_MAX_ELEMS"] == "3670016"
    assert payload["proof_runtime_env"]["VERATHOS_GGML_TRACE_SKIP_SRC0_DUMP"] == "1"


def test_llama_rpc_ready_probe_skips_raw_tcp_by_default(monkeypatch):
    import verallm.mesh.cli as cli_mod

    def fail_connect(*args, **kwargs):
        raise AssertionError("raw TCP probe should be opt-in")

    monkeypatch.delenv("VERATHOS_LLAMA_RPC_TCP_PROBE", raising=False)
    monkeypatch.setattr(cli_mod.socket, "create_connection", fail_connect)

    assert cli_mod._rpc_endpoint_is_ready("worker.local:50052")


def test_llama_cpp_runtime_patches_include_gpu_backends_and_per_capture_gating():
    root = Path(__file__).resolve().parents[2]
    script = (root / "runtime/llama_cpp/build_verathos_rpc_server.sh").read_text(
        encoding="utf-8"
    )
    cpu_patch = (root / "runtime/llama_cpp/verathos_ggml_cpu_trace.patch").read_text(
        encoding="utf-8"
    )
    cuda_patch = (root / "runtime/llama_cpp/verathos_ggml_cuda_trace.patch").read_text(
        encoding="utf-8"
    )
    metal_patch = (root / "runtime/llama_cpp/verathos_ggml_metal_trace.patch").read_text(
        encoding="utf-8"
    )
    vulkan_patch = (root / "runtime/llama_cpp/verathos_ggml_vulkan_trace.patch").read_text(
        encoding="utf-8"
    )

    assert "VERATHOS_BUILD_CUDA" in script
    assert "VERATHOS_BUILD_METAL" in script
    assert "VERATHOS_BUILD_VULKAN" in script
    assert "verathos_ggml_cuda_trace.patch" in script
    assert "verathos_ggml_metal_trace.patch" in script
    assert "verathos_ggml_vulkan_trace.patch" in script
    assert "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE" in cpu_patch
    assert "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE" in cuda_patch
    assert "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE" in metal_patch
    assert "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE" in vulkan_patch
    assert "%s/manifest%s.%s" in cpu_patch
    assert "verathos_cpu_manifest_suffix" in cpu_patch
    assert "verathos_cuda_manifest_suffix" in cuda_patch
    assert "/manifest\" + verathos_cuda_manifest_suffix()" in cuda_patch
    assert "verathos_metal_manifest_suffix" in metal_patch
    assert "/manifest\" + verathos_metal_manifest_suffix()" in metal_patch
    assert "verathos_vk_manifest_suffix" in vulkan_patch
    assert "/manifest\" + verathos_vk_manifest_suffix()" in vulkan_patch
    assert "selected=" in cpu_patch
    assert "selected=" in cuda_patch
    assert "selected=" in metal_patch
    assert "selected=" in vulkan_patch
    assert "should_dump_manifest_index" in cpu_patch
    assert "should_dump_manifest_index" in cuda_patch
    assert "should_dump_manifest_index" in metal_patch
    assert "should_dump_manifest_index" in vulkan_patch
    assert "proof_eligible" in cuda_patch
    assert "llama_cpp_cuda" in cuda_patch
    assert "llama_cpp_metal" in metal_patch
    assert "verathos_metal_is_stable_after_graph" in metal_patch
    assert "ggml_node_get_use_count(cgraph, node_idx) == 0" in metal_patch
    assert "llama_cpp_vulkan" in vulkan_patch
    assert "verathos_metal_try_dump_graph_trace" in metal_patch
    assert "!verathos_vk_trace_enabled()" in vulkan_patch


def test_cli_build_runtime_dry_run_exposes_backend_flags(capsys):
    mesh_cli_main(
        [
            "build-runtime",
            "/opt/llama.cpp",
            "--build-dir",
            "/tmp/build-verathos",
            "--cuda",
            "--metal",
            "--vulkan",
            "--llama-server",
            "--jobs",
            "8",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"][-2:] == ["/opt/llama.cpp", "/tmp/build-verathos"]
    assert payload["enabled_backends"] == ["cuda", "metal", "vulkan"]
    assert payload["build_llama_server"] is True
    assert payload["env"]["VERATHOS_BUILD_CUDA"] == "1"
    assert payload["env"]["VERATHOS_BUILD_METAL"] == "1"
    assert payload["env"]["VERATHOS_BUILD_VULKAN"] == "1"
    assert payload["env"]["VERATHOS_BUILD_LLAMA_SERVER"] == "1"
    assert payload["env"]["VERATHOS_BUILD_JOBS"] == "8"


def test_cli_processes_launch_join_reload_and_infer_with_fake_llama_cpp(tmp_path):
    fake_llama, fake_rpc, llama_args_file, rpc_args_file = _write_fake_llama_cpp_binaries(
        tmp_path
    )
    coord_port = _free_port()
    worker_port = _free_port()
    rpc_port = _free_port()
    llama_port = _free_port()
    coordinator_endpoint = f"http://127.0.0.1:{coord_port}"
    worker_endpoint = f"http://127.0.0.1:{worker_port}"
    rpc_endpoint = f"127.0.0.1:{rpc_port}"
    env = os.environ.copy()
    env["FAKE_LLAMA_ARGS_FILE"] = str(llama_args_file)
    env["FAKE_RPC_ARGS_FILE"] = str(rpc_args_file)

    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint=coordinator_endpoint,
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=8,
    )
    state_dir, _, token = create_mesh_state(spec=spec, root=tmp_path / "coord")

    coordinator = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "neurons.cli",
            "mesh",
            "serve",
            "--mesh",
            str(state_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(coord_port),
            "--allow-loopback-dev-validator-routes",
            "--llama-model",
            "/models/model.gguf",
            "--llama-server-binary",
            str(fake_llama),
            "--llama-host",
            "127.0.0.1",
            "--llama-port",
            str(llama_port),
            "--llama-min-rpc-workers",
            "1",
            "--llama-reload-interval",
            "0.2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    worker = None
    try:
        _wait_for_probe(
            coordinator_endpoint,
            internal_auth_secret=token.join_secret,
        )
        worker_dir, joined = join_mesh(
            token=token.encode(),
            endpoint=worker_endpoint,
            root=tmp_path / "worker",
            package_hash=DIGEST_A,
            rpc_endpoint=rpc_endpoint,
        )
        assert joined.members[1].rpc_endpoint == rpc_endpoint

        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "neurons.cli",
                "mesh",
                "serve",
                "--mesh",
                str(worker_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(worker_port),
                "--rpc-worker",
                "--rpc-worker-binary",
                str(fake_rpc),
                "--rpc-host",
                "127.0.0.1",
                "--rpc-port",
                str(rpc_port),
                "--rpc-device",
                "CUDA0",
                "--rpc-cache",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_probe(
            worker_endpoint,
            internal_auth_secret=token.join_secret,
        )
        rpc_args = json.loads(_wait_for_file(rpc_args_file))
        assert rpc_args["args"]["port"] == rpc_port
        assert rpc_args["args"]["device"] == "CUDA0"
        assert rpc_args["args"]["cache"] is True

        _wait_for_http(coordinator_endpoint)
        llama_args = json.loads(_wait_for_file(llama_args_file))
        assert llama_args["args"]["model"] == "/models/model.gguf"
        assert llama_args["args"]["rpc"] == rpc_endpoint

        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "process route"}],
            "stream": False,
        }
        response = post_json(f"{coordinator_endpoint}/v1/chat/completions", request)
        mesh = response["verathos_mesh"]
        receipt = mesh["receipt"]
        current = state_mesh_spec(load_mesh_state(state_dir))

        assert response["choices"][0]["message"]["content"] == "fake mesh ok"
        assert mesh["receipt_verified"] is True
        assert mesh["runtime"] == "llama_cpp_rpc"
        assert mesh["rpc_endpoints"] == []
        assert rpc_endpoint not in json.dumps(mesh)
        assert mesh["rpc_plan_hash"] == rpc_plan_from_mesh(current).plan_hash_hex()
        assert mesh["mesh_response_commitment_hash"] == _mesh_response_commitment_hash(receipt)
        assert receipt["request_hash"] == _payload_hash(request)
        response_without_mesh = dict(response)
        response_without_mesh.pop("verathos_mesh")
        assert receipt["response_hash"] == _payload_hash(response_without_mesh)
        assert receipt["receipt_hash"] == _receipt_hash(receipt)

        worker_two_port = _free_port()
        worker_two_rpc_port = _free_port()
        worker_two_endpoint = f"http://127.0.0.1:{worker_two_port}"
        _, joined_two = join_mesh(
            token=token.encode(),
            endpoint=worker_two_endpoint,
            root=tmp_path / "worker-two",
            package_hash=DIGEST_A,
            rpc_endpoint=f"127.0.0.1:{worker_two_rpc_port}",
        )
        assert len(joined_two.members) == 3
        synced = _wait_for_mesh_member_count(worker_dir, 3)
        assert synced.stage_assignment_hash() == joined_two.stage_assignment_hash()
    finally:
        if worker is not None:
            _terminate_process(worker)
        _terminate_process(coordinator)


def _calls_match_with_derived_seed(calls, expected):
    """Compare backend calls tolerating the derived replay seed injection."""

    if len(calls) != len(expected):
        return False
    for got, want in zip(calls, expected):
        got = dict(got)
        # The verathos envelope (proof tier, snapshot binding) is consumed by
        # the mesh control plane and never forwarded to the backend.
        want = {key: value for key, value in want.items() if key != "verathos"}
        if "seed" not in want and isinstance(got.get("seed"), int) and got["seed"] >= 1:
            got.pop("seed")
        # Witness-regenerating replay serves disable the llama-server prompt
        # cache so a cache hit cannot leave the capture window witness-less.
        if "cache_prompt" not in want and got.get("cache_prompt") is False:
            got.pop("cache_prompt")
        # Requests without their own completion bound get the serve-side
        # default cap (llama-server would otherwise decode without limit).
        if (
            "max_tokens" not in want
            and "max_completion_tokens" not in want
            and isinstance(got.get("max_tokens"), int)
        ):
            got.pop("max_tokens")
        if got != want:
            return False
    return True


# --- slot-view concurrency attribution (verified --parallel N support) ---


def _slot_view_imports():
    from verallm.mesh.ggml_proof import (
        GgmlOpManifestEntry,
        SlotViewLeaf,
        _slot_view_leaf_body_from_template_op,
        build_slot_view_leaves,
        derive_every_request_trace_beacon,
        derive_every_request_trace_beacon_v2,
        find_slot_view_template_for_window,
        ggml_slot_view_root,
        ggml_slot_view_root_from_template,
        ggml_slot_view_selection_payload,
        ggml_slot_view_selection_payload_from_template,
        select_slot_view_challenges,
        slot_view_membership_payload,
        slot_view_membership_payload_from_template,
        slot_view_template_from_manifest_entries,
        solo_prefill_graph_count,
        solo_replay_chunk_for_position,
        verify_slot_view_membership,
        verify_slot_view_proof_payload,
        window_capture_file_token,
    )

    return {
        "GgmlOpManifestEntry": GgmlOpManifestEntry,
        "SlotViewLeaf": SlotViewLeaf,
        "_slot_view_leaf_body_from_template_op": _slot_view_leaf_body_from_template_op,
        "build_slot_view_leaves": build_slot_view_leaves,
        "derive_every_request_trace_beacon": derive_every_request_trace_beacon,
        "derive_every_request_trace_beacon_v2": derive_every_request_trace_beacon_v2,
        "find_slot_view_template_for_window": find_slot_view_template_for_window,
        "ggml_slot_view_root": ggml_slot_view_root,
        "ggml_slot_view_root_from_template": ggml_slot_view_root_from_template,
        "ggml_slot_view_selection_payload": ggml_slot_view_selection_payload,
        "ggml_slot_view_selection_payload_from_template": (
            ggml_slot_view_selection_payload_from_template
        ),
        "select_slot_view_challenges": select_slot_view_challenges,
        "slot_view_membership_payload": slot_view_membership_payload,
        "slot_view_membership_payload_from_template": (
            slot_view_membership_payload_from_template
        ),
        "slot_view_template_from_manifest_entries": slot_view_template_from_manifest_entries,
        "solo_prefill_graph_count": solo_prefill_graph_count,
        "solo_replay_chunk_for_position": solo_replay_chunk_for_position,
        "verify_slot_view_membership": verify_slot_view_membership,
        "verify_slot_view_proof_payload": verify_slot_view_proof_payload,
        "window_capture_file_token": window_capture_file_token,
    }


def _v3_row(
    now_ns,
    manifest_index,
    name,
    src1_rows,
    graph_seq,
    intra,
    *,
    k=64,
    n=32,
    backend="llama_cpp_cpu",
    device="CPU",
):
    return (
        "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3"
        f"\t{now_ns}\t{manifest_index}\t{name}\tsrc1-{name}\tdst-{name}"
        f"\t{k},{n},1,1\t{k},{src1_rows},1,1\t{n},{src1_rows},1,1"
        f"\tq4_K\tf32\tf32\t{backend}\t{device}\t1"
        f"\t{graph_seq}\t{intra}"
    )


def test_v3_manifest_row_parses_and_keeps_v2_entry_hash():
    mods = _slot_view_imports()
    entry_cls = mods["GgmlOpManifestEntry"]
    v3 = entry_cls.from_compact_line(_v3_row(1000, 1, "blk.0.attn_q.weight", 4, 7, 3))
    assert v3.graph_seq == 7
    assert v3.intra_graph_index == 3
    v2_line = "\t".join(_v3_row(1000, 1, "blk.0.attn_q.weight", 4, 7, 3).split("\t")[:15])
    v2 = entry_cls.from_compact_line(
        v2_line.replace(
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3",
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V2",
        )
    )
    assert v2.graph_seq == -1
    # graph tags are bookkeeping only: committed entry hashes must not change
    assert v2.entry_hash() == v3.entry_hash()


def test_solo_prefill_graph_count():
    mods = _slot_view_imports()
    fn = mods["solo_prefill_graph_count"]
    assert fn(prompt_token_count=1, n_ubatch=128) == 1
    assert fn(prompt_token_count=128, n_ubatch=128) == 1
    assert fn(prompt_token_count=129, n_ubatch=128) == 2
    assert fn(prompt_token_count=100, n_ubatch=64) == 2


def test_solo_replay_chunk_arithmetic():
    mods = _slot_view_imports()
    fn = mods["solo_replay_chunk_for_position"]
    # prefill chunks of 64 over a 100-token prompt: graphs 1..2
    assert fn(position=0, prompt_token_count=100, n_ubatch=64) == (1, 0, 64)
    assert fn(position=63, prompt_token_count=100, n_ubatch=64) == (1, 63, 64)
    assert fn(position=64, prompt_token_count=100, n_ubatch=64) == (2, 0, 36)
    assert fn(position=99, prompt_token_count=100, n_ubatch=64) == (2, 35, 36)
    # decode positions: one graph per token
    assert fn(position=100, prompt_token_count=100, n_ubatch=64) == (3, 0, 1)
    assert fn(position=130, prompt_token_count=100, n_ubatch=64) == (33, 0, 1)


def _layer_template_entries(now_ns=1000):
    # A tiny two-op model template: one attention proj + the LM head.
    mods = _slot_view_imports()
    entry_cls = mods["GgmlOpManifestEntry"]
    return [
        entry_cls.from_compact_line(_v3_row(now_ns, 1, "blk.0.attn_q.weight", 16, 1, 0)),
        entry_cls.from_compact_line(_v3_row(now_ns + 1, 2, "output.weight", 1, 1, 1)),
    ]


def test_build_slot_view_leaves_is_deterministic_decode_layout():
    mods = _slot_view_imports()
    # Template entries can come from co-batched graphs of any batch size; the
    # leaves depend only on committed counts + the per-intra-index template.
    entries = _layer_template_entries()
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=entries,
        completion_token_count=3,
    )
    # 2 template ops x 3 decoded tokens
    assert len(leaves) == 6
    # v3 leaves carry no graph ordinal: the layout is (token_index, op) only.
    assert sorted({leaf.token_index for leaf in leaves}) == [0, 1, 2]
    token_zero = [leaf for leaf in leaves if leaf.token_index == 0]
    assert {leaf.intra_graph_index for leaf in token_zero} == {0, 1}

    # Co-batched template with bigger src1 rows yields identical leaves: the
    # commitment excludes batch-dependent activation row counts.
    mods2 = _slot_view_imports()
    entry_cls = mods2["GgmlOpManifestEntry"]
    batched = [
        entry_cls.from_compact_line(_v3_row(2000, 5, "blk.0.attn_q.weight", 384, 9, 0)),
        entry_cls.from_compact_line(_v3_row(2001, 6, "output.weight", 12, 9, 1)),
    ]
    leaves_batched = mods2["build_slot_view_leaves"](
        manifest_entries=batched,
        completion_token_count=3,
    )
    assert mods["ggml_slot_view_root"](leaves) == mods2["ggml_slot_view_root"](
        leaves_batched
    )


def test_slot_view_membership_and_selection_roundtrip():
    mods = _slot_view_imports()
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=_layer_template_entries(),
        completion_token_count=4,
    )
    beacon = hashlib.sha256(b"test-beacon").digest()
    root = mods["ggml_slot_view_root"](leaves)
    selected = mods["select_slot_view_challenges"](
        beacon=beacon,
        slot_view_root=root,
        slot_view_count=len(leaves),
        proof_ops_per_request=1,
        stage_index=0,
    )
    assert len(selected) == 1
    leaf_index = selected[0]
    membership = mods["slot_view_membership_payload"](
        leaves,
        leaf_index,
        stage_index=0,
    )
    assert membership["slot_view_root"] == root
    assert membership["scope"] == "slot_view_v3"
    assert mods["verify_slot_view_membership"](
        leaf_hash=membership["slot_view_leaf_hash"],
        slot_view_root=membership["slot_view_root"],
        slot_view_count=membership["slot_view_count"],
        leaf_index=membership["slot_view_leaf_index"],
        path=membership["slot_view_membership_path"],
    )


def test_light_slot_view_cost_is_flat_in_prompt_context():
    """The light tier must not grow with context, up to 250k-token prompts.

    v3 leaves are fully prompt-independent: the slot view commits (token
    index, template op) only, so the leaf set, root, and every cache built
    over them are identical across prompt lengths and session turns. The
    prompt itself stays bound at the receipt level (prompt_token_ids_hash),
    not in the slot view.
    """

    from verallm.mesh.ggml_proof import slot_view_leaf_hash_bytes_from_template

    mods = _slot_view_imports()
    template = mods["slot_view_template_from_manifest_entries"](
        _layer_template_entries()
    )
    leaves = slot_view_leaf_hash_bytes_from_template(
        template=template,
        completion_token_count=8,
    )
    assert len(leaves) == 2 * 8
    root, count = mods["ggml_slot_view_root_from_template"](
        template=template,
        completion_token_count=8,
    )
    assert count == len(leaves)
    # Same template + same completion = same root, no prompt axis at all.
    assert (root, count) == mods["ggml_slot_view_root_from_template"](
        template=template,
        completion_token_count=8,
    )


def test_slot_view_proof_path_shares_one_tree_build():
    """Selection (root) and membership (path) must share ONE levels build.

    The incremental levels cache hashes each interior node at most once
    per template prefix; a second pass over the same leaf set (selection
    then membership) must hash nothing new.
    """

    from verallm.mesh import ggml_proof

    mods = _slot_view_imports()
    template = mods["slot_view_template_from_manifest_entries"](
        _layer_template_entries()
    )
    with ggml_proof._slot_view_memo_lock:
        ggml_proof._slot_view_leaf_cache.clear()
        ggml_proof._slot_view_levels_cache.clear()

    calls = {"n": 0}
    original = ggml_proof._manifest_node_hash

    def counting(left, right):
        calls["n"] += 1
        return original(left, right)

    ggml_proof._manifest_node_hash = counting
    try:
        ctx = {
            "stage_index": 0,
            "proof_ops_per_request": 1,
            "completion_token_count": 4,
            "prompt_token_count": 20,
            "proof_runtime_ubatch_size": 64,
            "proof_beacon": hashlib.sha256(b"levels-share").hexdigest(),
        }
        selection = mods["ggml_slot_view_selection_payload_from_template"](
            template=template, receipt_context=ctx
        )
        after_selection = calls["n"]
        membership = mods["slot_view_membership_payload_from_template"](
            template=template,
            leaf_index=int(selection["selected"][0]["leaf_index"]),
            completion_token_count=4,
            stage_index=0,
        )
    finally:
        ggml_proof._manifest_node_hash = original
    assert after_selection > 0
    # Membership reassembles from cached complete nodes; only the ragged
    # right edge (at most one node per level) may be rehashed.
    assert calls["n"] - after_selection <= 8, (
        f"membership rehashed {calls['n'] - after_selection} nodes"
    )
    assert membership["slot_view_root"] == selection["slot_view_root"]


def test_slot_view_proof_path_reuses_memoized_leaf_hashes():
    """Selection and the membership opening must not each rebuild the leaves.

    Both need the identical leaf set, which is O(decoded tokens x template
    ops); recomputing it twice measured 21.6 s against 8.2 s for one pass at
    4096 decoded tokens. Pin the reuse so the second pass cannot creep back.
    """

    from verallm.mesh import ggml_proof

    mods = _slot_view_imports()
    template = mods["slot_view_template_from_manifest_entries"](
        _layer_template_entries()
    )
    with ggml_proof._slot_view_memo_lock:
        ggml_proof._slot_view_leaf_cache.clear()

    calls = {"n": 0}
    original = ggml_proof._slot_view_leaf_hashes_uncached

    def counting(**kwargs):
        calls["n"] += 1
        return original(**kwargs)

    ggml_proof._slot_view_leaf_hashes_uncached = counting
    try:
        ctx = {
            "stage_index": 0,
            "proof_ops_per_request": 1,
            "completion_token_count": 4,
            "prompt_token_count": 20,
            "proof_runtime_ubatch_size": 64,
            "proof_beacon": hashlib.sha256(b"memo-reuse").hexdigest(),
        }
        selection = mods["ggml_slot_view_selection_payload_from_template"](
            template=template, receipt_context=ctx
        )
        mods["slot_view_membership_payload_from_template"](
            template=template,
            leaf_index=int(selection["selected"][0]["leaf_index"]),
            completion_token_count=4,
            stage_index=0,
        )
    finally:
        ggml_proof._slot_view_leaf_hashes_uncached = original
    assert calls["n"] == 1, (
        f"leaf hashes rebuilt {calls['n']} times across selection + membership"
    )


def test_fast_slot_view_template_root_matches_materialized_leaves():
    mods = _slot_view_imports()
    entries = _layer_template_entries()
    template = mods["slot_view_template_from_manifest_entries"](entries)
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=entries,
        completion_token_count=4,
    )
    fast_root, fast_count = mods["ggml_slot_view_root_from_template"](
        template=template,
        completion_token_count=4,
    )
    assert fast_count == len(leaves)
    assert fast_root == mods["ggml_slot_view_root"](leaves)

    beacon_hex = hashlib.sha256(b"fast-template-selection").hexdigest()
    ctx = {
        "stage_index": 0,
        "proof_ops_per_request": 1,
        "completion_token_count": 4,
        "prompt_token_count": 20,
        "proof_runtime_ubatch_size": 64,
        "proof_beacon": beacon_hex,
    }
    materialized = mods["ggml_slot_view_selection_payload"](
        leaves=leaves,
        receipt_context=ctx,
    )
    fast = mods["ggml_slot_view_selection_payload_from_template"](
        template=template,
        receipt_context=ctx,
    )
    assert fast["slot_view_root"] == materialized["slot_view_root"]
    assert fast["slot_view_count"] == materialized["slot_view_count"]
    assert fast["selected"] == materialized["selected"]
    assert fast["selected_ops"] == materialized["selected_ops"]

    leaf_index = materialized["selected"][0]["leaf_index"]
    materialized_membership = mods["slot_view_membership_payload"](
        leaves,
        leaf_index,
        stage_index=0,
    )
    fast_membership = mods["slot_view_membership_payload_from_template"](
        template=template,
        leaf_index=leaf_index,
        completion_token_count=4,
        stage_index=0,
    )
    assert fast_membership == materialized_membership


def test_slot_view_selection_payload_force_includes_decode_leaf():
    mods = _slot_view_imports()
    entries = _layer_template_entries()
    template = mods["slot_view_template_from_manifest_entries"](entries)
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=entries,
        completion_token_count=4,
    )
    ctx = {
        "stage_index": 0,
        "proof_ops_per_request": 1,
        "completion_token_count": 4,
        "prompt_token_count": 20,
        "proof_runtime_ubatch_size": 64,
        "proof_beacon": hashlib.sha256(b"decode-slot-view-selection").hexdigest(),
        "decode_audit_required": True,
        "decode_audit_positions": [1],
    }
    materialized = mods["ggml_slot_view_selection_payload"](
        leaves=leaves,
        receipt_context=ctx,
    )
    fast = mods["ggml_slot_view_selection_payload_from_template"](
        template=template,
        receipt_context=ctx,
    )
    assert fast["slot_view_root"] == materialized["slot_view_root"]
    assert fast["slot_view_count"] == materialized["slot_view_count"]
    assert fast["selected"] == materialized["selected"]
    assert fast["selected_ops"] == materialized["selected_ops"]
    assert len(fast["selected"]) == 1
    selected = fast["selected"][0]
    assert selected["decode_audit_positions"] == [1]
    assert selected["replay_token_index"] == 1
    assert selected["leaf"]["tensor_name"] == "output.weight"
    assert len(fast["selected_ops"]) <= 64
    ordinal_ops = [op for op in fast["selected_ops"] if not op.startswith("n:")]
    name_ops = [op for op in fast["selected_ops"] if op.startswith("n:")]
    assert all(
        op.endswith(f":{selected['replay_intra_index']}") for op in ordinal_ops
    )
    # Name-armed entries identify the op across graph kinds (length-dependent
    # op streams shift intras); they come LAST so parsers without name
    # support keep their exact pre-name behavior.
    assert name_ops == ["n:output.weight"]
    assert fast["selected_ops"][-1] == "n:output.weight"


def test_slot_view_selected_ops_fit_native_capture_token():
    mods = _slot_view_imports()
    entries = _layer_template_entries()
    template = mods["slot_view_template_from_manifest_entries"](entries)
    ctx = {
        "stage_index": 0,
        "proof_ops_per_request": 1,
        "completion_token_count": 149,
        "prompt_token_count": 37,
        "proof_runtime_ubatch_size": 512,
        "proof_beacon": hashlib.sha256(b"decode-slot-view-token-budget").hexdigest(),
        "decode_audit_required": True,
        "decode_audit_positions": [148],
    }
    payload = mods["ggml_slot_view_selection_payload_from_template"](
        template=template,
        receipt_context=ctx,
    )
    selected = payload["selected"][0]
    # v3 leaves carry no graph ordinal; arming derives the committed solo
    # ordinal arithmetically: prefill graphs (ceil(37/512) = 1) + token + 1.
    expected_graph = 1 + int(selected["replay_token_index"]) + 1
    expected_op = f"{expected_graph}:{selected['replay_intra_index']}"
    assert expected_op in payload["selected_ops"]
    # Ord-agnostic wildcard entries lead: replay graph ordinals drift from
    # the committed layout when the scheduler splits a pass into extra
    # subgraphs, so every armed intra is also dumped ord-independently.
    assert payload["selected_ops"][0] == f"*:{selected['replay_intra_index']}"
    assert len(",".join(payload["selected_ops"])) <= 240


def test_slot_view_decode_selection_falls_back_without_final_stage_op():
    mods = _slot_view_imports()
    entry_cls = mods["GgmlOpManifestEntry"]
    entries = [
        entry_cls.from_compact_line(_v3_row(1000, 1, "blk.0.attn_q.weight", 16, 1, 0)),
        entry_cls.from_compact_line(_v3_row(1001, 2, "blk.0.attn_k.weight", 16, 1, 1)),
    ]
    template = mods["slot_view_template_from_manifest_entries"](entries)
    ctx = {
        "stage_index": 0,
        "proof_ops_per_request": 1,
        "completion_token_count": 4,
        "prompt_token_count": 20,
        "proof_runtime_ubatch_size": 64,
        "proof_beacon": hashlib.sha256(b"decode-no-final-stage").hexdigest(),
        "proof_sampled": True,
        "decode_audit_required": True,
        "decode_audit_stage_index": 1,
        "decode_audit_positions": [1],
    }
    payload = mods["ggml_slot_view_selection_payload_from_template"](
        template=template,
        receipt_context=ctx,
    )
    assert len(payload["selected"]) == 1
    assert payload["selected"][0]["decode_audit_positions"] == []
    assert payload["selected"][0]["leaf"]["tensor_name"] in {
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
    }


def test_fast_slot_view_template_loader_stops_after_first_v3_graph(tmp_path):
    mods = _slot_view_imports()
    path = tmp_path / "manifest-1000.vmanifest"
    rows = [
        _v3_row(1000, 1, "blk.0.attn_q.weight", 16, 1, 0),
        _v3_row(1001, 2, "output.weight", 1, 1, 1),
        _v3_row(1002, 3, "blk.0.attn_q.weight", 16, 2, 0),
        _v3_row(1003, 4, "output.weight", 1, 2, 1),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    template = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="1000",
    )
    assert [item["intra_graph_index"] for item in template] == [0, 1]
    assert [item["tensor_name"] for item in template] == [
        "blk.0.attn_q.weight",
        "output.weight",
    ]


def test_fast_slot_view_template_loader_skips_incomplete_first_graph(tmp_path):
    mods = _slot_view_imports()
    path = tmp_path / "manifest-1000.vmanifest"
    rows = [
        _v3_row(1000, 1, "blk.0.attn_q.weight", 16, 1, 0),
        _v3_row(1001, 2, "blk.0.attn_q.weight", 1, 2, 0),
        _v3_row(1002, 3, "output.weight", 1, 2, 1),
        _v3_row(1003, 4, "blk.0.attn_q.weight", 1, 3, 0),
        _v3_row(1004, 5, "output.weight", 1, 3, 1),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    template = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="1000",
    )
    assert [item["intra_graph_index"] for item in template] == [0, 1]
    assert [item["tensor_name"] for item in template] == [
        "blk.0.attn_q.weight",
        "output.weight",
    ]


def test_fast_slot_view_template_loader_merges_device_subgraphs(tmp_path):
    """A multi-GPU forward runs one sub-graph per device; the logit graph
    spans only the logit device's layers. The template must merge one
    decode-shaped sub-graph per device or the slot-view challenge universe
    covers a fraction of the model and the per-layer floor rejects it
    (observed on glm-5.2 over 4 GPUs: 211 of the needed 237 ops)."""
    mods = _slot_view_imports()
    path = tmp_path / "manifest-1000.vmanifest"
    cuda0 = {"backend": "llama_cpp_cuda", "device": "CUDA0"}
    cuda1 = {"backend": "llama_cpp_cuda", "device": "CUDA1"}
    rows = [
        # Prefill forward: chunk-width layer GEMMs on both devices, the
        # logit device also computes last-token logits (single row).
        _v3_row(1000, 1, "blk.0.attn_q.weight", 16, 1, 0, **cuda0),
        _v3_row(1001, 2, "blk.1.attn_q.weight", 16, 2, 0, **cuda1),
        _v3_row(1002, 3, "output.weight", 1, 2, 1, **cuda1),
        # Decode forward: single-row everywhere, one sub-graph per device.
        _v3_row(1003, 4, "blk.0.attn_q.weight", 1, 3, 0, **cuda0),
        _v3_row(1004, 5, "blk.1.attn_q.weight", 1, 4, 0, **cuda1),
        _v3_row(1005, 6, "output.weight", 1, 4, 1, **cuda1),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    template = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="1000",
    )
    assert [
        (item["device"], item["tensor_name"], item["intra_graph_index"])
        for item in template
    ] == [
        ("CUDA0", "blk.0.attn_q.weight", 0),
        ("CUDA1", "blk.1.attn_q.weight", 0),
        ("CUDA1", "output.weight", 1),
    ]


def test_fast_slot_view_template_loader_scans_past_long_multigpu_prefill(
    tmp_path,
):
    """Prompt length must not cap structural discovery.

    A real 28k-token GLM request produced 220 prefill sub-graphs across four
    GPUs before its first decode forward.  The loader must compact that
    history and still merge the later per-device decode sub-graphs.
    """

    mods = _slot_view_imports()
    path = tmp_path / "manifest-1000.vmanifest"
    devices = [
        {"backend": "llama_cpp_cuda", "device": f"CUDA{idx}"}
        for idx in range(4)
    ]
    rows = []
    timestamp = 1000
    sequence = 1
    graph = 1
    for _forward in range(20):
        for device_index, device in enumerate(devices):
            rows.append(
                _v3_row(
                    timestamp,
                    sequence,
                    f"blk.{device_index}.attn_q.weight",
                    512,
                    graph,
                    0,
                    **device,
                )
            )
            timestamp += 1
            sequence += 1
            graph += 1
    for device_index, device in enumerate(devices):
        rows.append(
            _v3_row(
                timestamp,
                sequence,
                f"blk.{device_index}.attn_q.weight",
                1,
                graph,
                0,
                **device,
            )
        )
        timestamp += 1
        sequence += 1
        if device_index == 3:
            rows.append(
                _v3_row(
                    timestamp,
                    sequence,
                    "output.weight",
                    1,
                    graph,
                    1,
                    **device,
                )
            )
            timestamp += 1
            sequence += 1
        graph += 1
    # The repeated CUDA0 graph proves that the complete four-device decode
    # forward has ended and gives the streaming loader its stop boundary.
    rows.append(
        _v3_row(
            timestamp,
            sequence,
            "blk.0.attn_q.weight",
            1,
            graph,
            0,
            **devices[0],
        )
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    template = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="1000",
    )
    assert [(item["device"], item["tensor_name"]) for item in template] == [
        ("CUDA0", "blk.0.attn_q.weight"),
        ("CUDA1", "blk.1.attn_q.weight"),
        ("CUDA2", "blk.2.attn_q.weight"),
        ("CUDA3", "blk.3.attn_q.weight"),
        ("CUDA3", "output.weight"),
    ]


def test_fast_slot_view_template_loader_skips_prefill_only_probe_window(tmp_path):
    """The newest manifest at request start can be a closed certification-
    probe window whose graphs are ALL prefill-shaped (observed: a
    13-token probe file pinned a single-device prefill-fallback template
    for the whole serve). With a decode-shaped window on disk, the loader
    must fall back to it instead of settling for the probe file."""
    mods = _slot_view_imports()
    cuda0 = {"backend": "llama_cpp_cuda", "device": "CUDA0"}
    cuda1 = {"backend": "llama_cpp_cuda", "device": "CUDA1"}
    # Probe window (newer token): one 13-row forward on both devices.
    probe = tmp_path / "manifest-2000.vmanifest"
    probe.write_text(
        "\n".join(
            [
                _v3_row(2000, 1, "blk.0.attn_q.weight", 13, 1, 0, **cuda0),
                _v3_row(2001, 2, "blk.1.attn_q.weight", 13, 2, 0, **cuda1),
                _v3_row(2002, 3, "output.weight", 1, 2, 1, **cuda1),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Serve window (older token): a real decode forward.
    serve = tmp_path / "manifest-1000.vmanifest"
    serve.write_text(
        "\n".join(
            [
                _v3_row(1000, 1, "blk.0.attn_q.weight", 1, 1, 0, **cuda0),
                _v3_row(1001, 2, "blk.1.attn_q.weight", 1, 2, 0, **cuda1),
                _v3_row(1002, 3, "output.weight", 1, 2, 1, **cuda1),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    template = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="2000",
    )
    assert [(op["device"], op["tensor_name"]) for op in template] == [
        ("CUDA0", "blk.0.attn_q.weight"),
        ("CUDA1", "blk.1.attn_q.weight"),
        ("CUDA1", "output.weight"),
    ]
    # With no decode-shaped window anywhere, the prefill fallback survives.
    serve.unlink()
    fallback = mods["find_slot_view_template_for_window"](
        tmp_path,
        file_token="2000",
    )
    assert [op["tensor_name"] for op in fallback] == [
        "blk.1.attn_q.weight",
        "output.weight",
    ]


def test_slot_view_selection_payload_targets_solo_graph_ords():
    mods = _slot_view_imports()
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=_layer_template_entries(),
        completion_token_count=2,
    )
    payload = mods["ggml_slot_view_selection_payload"](
        leaves=leaves,
        receipt_context={
            "stage_index": 0,
            "proof_ops_per_request": 1,
            "completion_token_count": 2,
            "prompt_token_count": 20,
            "proof_runtime_ubatch_size": 64,
            "proof_beacon": hashlib.sha256(b"selection").hexdigest(),
        },
    )
    assert payload["scope"] == "slot_view_v3"
    assert len(payload["selected"]) == 1
    item = payload["selected"][0]
    assert item["replay_token_index"] in (0, 1)
    # selected_ops cover candidate graph ordinals for the challenged intra so
    # the replay dumps the decode instance regardless of warmup/prefill
    # offset, plus one trailing name-armed entry that identifies the op
    # across graph kinds when intras drift.
    intra = item["replay_intra_index"]
    ordinal_ops = [op for op in payload["selected_ops"] if not op.startswith("n:")]
    assert all(op.endswith(f":{intra}") for op in ordinal_ops)
    assert len(ordinal_ops) >= 2
    leaf_name = str(item["leaf"]["tensor_name"])
    assert payload["selected_ops"][-1] == f"n:{leaf_name}"


def test_slot_view_proof_payload_verifies_replay_binding():
    mods = _slot_view_imports()
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=_layer_template_entries(),
        completion_token_count=2,
    )
    beacon_hex = hashlib.sha256(b"bind").hexdigest()
    root = mods["ggml_slot_view_root"](leaves)
    selected = mods["select_slot_view_challenges"](
        beacon=bytes.fromhex(beacon_hex),
        slot_view_root=root,
        slot_view_count=len(leaves),
        proof_ops_per_request=1,
        stage_index=0,
    )
    leaf_index = selected[0]
    leaf = leaves[leaf_index]
    membership = mods["slot_view_membership_payload"](leaves, leaf_index, stage_index=0)
    # A well-formed solo-replay trace for the selected op (one activation row).
    payload = {
        "slot_view_membership": membership,
        "trace": {
            "graph_seq": 3,
            "intra_graph_index": leaf.intra_graph_index,
            "tensor_name": leaf.tensor_name,
            "src0_shape": [leaf.k_dim, leaf.n_dim, 1, 1],
            "src1_shape": [leaf.k_dim, 1, 1, 1],
            "source_types": dict(leaf.source_types),
        },
    }
    receipt = {
        "proof_beacon": beacon_hex,
        "proof_ops_per_request": 1,
        "prompt_token_count": 20,
        "completion_token_count": 2,
        "proof_runtime_ubatch_size": 64,
    }
    # Valid payload verifies.
    mods["verify_slot_view_proof_payload"](payload, mesh_receipt=receipt)

    # A drifted intra ordinal alone is ACCEPTED: both values are
    # prover-supplied so their equality never added binding, and
    # architectures with length-dependent op streams (glm-dsa) place the
    # same committed op at different intras per graph kind.
    drifted = json.loads(json.dumps(payload))
    drifted["trace"]["intra_graph_index"] = leaf.intra_graph_index + 1
    mods["verify_slot_view_proof_payload"](drifted, mesh_receipt=receipt)

    # A different WEIGHT than the committed leaf is rejected: the tensor
    # name is the op's identity (Merkle-bound against the signed manifest
    # by the payload verifier).
    bad = json.loads(json.dumps(payload))
    bad["trace"]["tensor_name"] = "blk.0.ffn_up.weight"
    try:
        mods["verify_slot_view_proof_payload"](bad, mesh_receipt=receipt)
        raise AssertionError("expected tensor name mismatch to raise")
    except RuntimeError:
        pass

    # A hard-tier teacher-forced probe can expose the runtime's bounded eager
    # prompt tail as one GEMM on recurrent models. It is accepted through the
    # shared 32-row ceiling, while anything larger remains fail-closed.
    hard = json.loads(json.dumps(payload))
    hard["trace"]["src1_shape"] = [leaf.k_dim, 14, 1, 1]
    hard_receipt = {**receipt, "proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE}
    hard["proof_mode"] = VERATHOS_GGML_GEMM_PROOF_MODE
    mods["verify_slot_view_proof_payload"](hard, mesh_receipt=hard_receipt)

    too_wide = json.loads(json.dumps(hard))
    too_wide["trace"]["src1_shape"] = [leaf.k_dim, 33, 1, 1]
    try:
        mods["verify_slot_view_proof_payload"](
            too_wide, mesh_receipt=hard_receipt
        )
        raise AssertionError("expected over-wide hard probe to raise")
    except RuntimeError:
        pass

    # A multi-row activation (co-batched leak) is rejected.
    bad2 = json.loads(json.dumps(payload))
    bad2["trace"]["src1_shape"] = [leaf.k_dim, 2, 1, 1]
    try:
        mods["verify_slot_view_proof_payload"](bad2, mesh_receipt=receipt)
        raise AssertionError("expected multi-row activation to raise")
    except RuntimeError:
        pass


def test_slot_view_proof_payload_allows_decode_selected_leaf():
    mods = _slot_view_imports()
    leaves = mods["build_slot_view_leaves"](
        manifest_entries=_layer_template_entries(),
        completion_token_count=4,
    )
    root = mods["ggml_slot_view_root"](leaves)
    decode_leaf_index = next(
        idx
        for idx, leaf in enumerate(leaves)
        if leaf.token_index == 1 and leaf.tensor_name == "output.weight"
    )
    beacon_hex = ""
    for attempt in range(64):
        candidate = hashlib.sha256(f"decode-not-base-{attempt}".encode()).hexdigest()
        selected = mods["select_slot_view_challenges"](
            beacon=bytes.fromhex(candidate),
            slot_view_root=root,
            slot_view_count=len(leaves),
            proof_ops_per_request=1,
            stage_index=0,
        )
        if decode_leaf_index not in selected:
            beacon_hex = candidate
            break
    assert beacon_hex
    leaf = leaves[decode_leaf_index]
    membership = mods["slot_view_membership_payload"](
        leaves,
        decode_leaf_index,
        stage_index=0,
    )
    payload = {
        "slot_view_membership": membership,
        "decode_audit_openings": [{"position": 1}],
        "trace": {
            "graph_seq": 3,
            "intra_graph_index": leaf.intra_graph_index,
            "tensor_name": leaf.tensor_name,
            "src0_shape": [leaf.k_dim, leaf.n_dim, 1, 1],
            "src1_shape": [leaf.k_dim, 1, 1, 1],
            "source_types": dict(leaf.source_types),
        },
    }
    receipt = {
        "proof_beacon": beacon_hex,
        "proof_ops_per_request": 1,
        "prompt_token_count": 20,
        "completion_token_count": 4,
        "proof_runtime_ubatch_size": 64,
        "decode_audit_required": True,
        "decode_audit_positions": [1],
    }
    mods["verify_slot_view_proof_payload"](payload, mesh_receipt=receipt)

    bad_leaf_index = next(
        idx
        for idx, item in enumerate(leaves)
        if item.token_index == 1 and item.tensor_name != "output.weight"
    )
    bad_leaf = leaves[bad_leaf_index]
    bad_payload = {
        "slot_view_membership": mods["slot_view_membership_payload"](
            leaves,
            bad_leaf_index,
            stage_index=0,
        ),
        "decode_audit_openings": [{"position": 1}],
        "trace": {
            "graph_seq": 3,
            "intra_graph_index": bad_leaf.intra_graph_index,
            "tensor_name": bad_leaf.tensor_name,
            "src0_shape": [bad_leaf.k_dim, bad_leaf.n_dim, 1, 1],
            "src1_shape": [bad_leaf.k_dim, 1, 1, 1],
            "source_types": dict(bad_leaf.source_types),
        },
    }
    try:
        mods["verify_slot_view_proof_payload"](bad_payload, mesh_receipt=receipt)
        raise AssertionError("expected non-logit decode leaf to raise")
    except RuntimeError:
        pass


def test_window_capture_file_token(tmp_path):
    mods = _slot_view_imports()
    (tmp_path / "manifest-1000.vmanifest").write_text("", encoding="utf-8")
    (tmp_path / "manifest-2000.vmanifest").write_text("", encoding="utf-8")
    token = mods["window_capture_file_token"](tmp_path, started_unix_ns=1500)
    assert token == "1000"
    token = mods["window_capture_file_token"](tmp_path, started_unix_ns=2500)
    assert token == "2000"


def test_every_request_beacon_v2_mixes_nonce():
    mods = _slot_view_imports()
    gate_hash = hashlib.sha256(b"gate").hexdigest()
    nonce_a = bytes(range(32))
    nonce_b = bytes(range(1, 33))
    v1 = mods["derive_every_request_trace_beacon"](gate_hash)
    v2_a = mods["derive_every_request_trace_beacon_v2"](gate_hash, nonce_a)
    v2_b = mods["derive_every_request_trace_beacon_v2"](gate_hash, nonce_b)
    assert v1 != v2_a
    assert v2_a != v2_b


def test_validator_postcommit_challenge_binds_request_snapshot_and_origin():
    nonce_a = bytes(range(32))
    nonce_b = bytes(range(1, 33))
    request_id = "11" * 32
    snapshot_hash = "22" * 32
    gate_hash = "33" * 32
    origin_hash = "44" * 32

    commitment = mesh_validator_challenge_nonce_commitment(
        nonce_a,
        validator_request_id=request_id,
        verification_snapshot_hash=snapshot_hash,
    )
    assert len(commitment) == 64
    assert bytes.fromhex(commitment) != nonce_a
    assert commitment == mesh_validator_challenge_nonce_commitment(
        nonce_a.hex(),
        validator_request_id=request_id,
        verification_snapshot_hash=snapshot_hash,
    )
    assert commitment != mesh_validator_challenge_nonce_commitment(
        nonce_b,
        validator_request_id=request_id,
        verification_snapshot_hash=snapshot_hash,
    )
    assert commitment != mesh_validator_challenge_nonce_commitment(
        nonce_a,
        validator_request_id="55" * 32,
        verification_snapshot_hash=snapshot_hash,
    )
    assert commitment != mesh_validator_challenge_nonce_commitment(
        nonce_a,
        validator_request_id=request_id,
        verification_snapshot_hash="66" * 32,
    )

    beacon = derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=origin_hash,
        proof_gate_hash=gate_hash,
        challenge_nonce=nonce_a,
    )
    assert len(beacon) == 32
    assert beacon == derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=origin_hash,
        proof_gate_hash=gate_hash,
        challenge_nonce=nonce_a.hex(),
    )
    assert beacon != derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash="77" * 32,
        proof_gate_hash=gate_hash,
        challenge_nonce=nonce_a,
    )
    assert beacon != derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=origin_hash,
        proof_gate_hash="88" * 32,
        challenge_nonce=nonce_a,
    )
    assert beacon != derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=origin_hash,
        proof_gate_hash=gate_hash,
        challenge_nonce=nonce_b,
    )
    assert VALIDATOR_POSTCOMMIT_CHALLENGE_KIND == "validator_postcommit_v1"


def test_replay_seed_derivation_is_deterministic_and_bounded():
    from verallm.mesh.proof import derive_mesh_replay_seed

    seed = derive_mesh_replay_seed("req-1")
    assert seed == derive_mesh_replay_seed("req-1")
    assert seed != derive_mesh_replay_seed("req-2")
    assert 1 <= seed <= 2**31 - 1


def test_deferred_commitment_version_gating():
    from verallm.mesh.proof import mesh_deferred_audit_commitment_hash

    base = {
        "request_id": "req",
        "mesh_id": "mesh",
        "request_hash": "aa" * 32,
        "response_hash": "bb" * 32,
        "proof_gate_hash": "cc" * 32,
        "proof_sample_bps": 10000,
    }
    legacy = mesh_deferred_audit_commitment_hash(base)
    # Receipts without slot-view replay state remain deterministic.
    assert legacy == mesh_deferred_audit_commitment_hash(dict(base))
    slot_view = mesh_deferred_audit_commitment_hash(
        {
            **base,
            "proof_op_manifest_scope": "slot_view_v3",
            "proof_runtime_ubatch_size": 128,
        }
    )
    # Deferred future randomness arrives after this commitment, so the exact
    # replay layout must be frozen even though inline sampling deliberately
    # excludes coordinator-controlled layout choices.
    assert slot_view != legacy


def test_manifest_commits_expert_planes_and_loader_matches(tmp_path):
    """A 3-D MoE expert tensor gets per-plane proof commitments, and the
    plane loader returns exactly the plane a full dequant would produce."""
    import gguf
    import numpy as np
    from verallm.mesh.gguf_manifest import (
        build_gguf_tensor_manifest,
        proof_i8_expert_plane_from_manifest,
        proof_f32_weight_matrix_from_gguf_f32,
        quantize_proof_i8,
    )

    model_path = str(tmp_path / "moe.gguf")
    writer = gguf.GGUFWriter(model_path, "llama")
    rng = np.random.default_rng(7)
    experts = rng.standard_normal((4, 3, 8), dtype=np.float32)  # [E=4, n=3, k=8] numpy
    writer.add_tensor("blk.0.ffn_gate_exps.weight", experts)
    writer.add_tensor("blk.0.attn_q.weight", rng.standard_normal((3, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = build_gguf_tensor_manifest(model_path)
    by_name = {r["name"]: r for r in manifest["tensors"]}
    moe = by_name["blk.0.ffn_gate_exps.weight"]
    dense = by_name["blk.0.attn_q.weight"]

    # dense record: unchanged single-matrix commitment, no expert fields
    assert "proof_i8_sha256" in dense and "proof_i8_expert_planes" not in dense
    # moe record: per-plane commitments, one per expert
    planes = int(moe["proof_i8_expert_planes"])
    assert planes == 4
    assert len(moe["proof_i8_expert_sha256"]) == planes
    assert len(moe["proof_i8_expert_merkle_roots"]) == planes
    assert "proof_i8_sha256" not in moe  # 3-D tensors have no whole-tensor matrix

    # loader plane == slice of a full dequant, for every expert
    reader = gguf.GGUFReader(model_path)
    t = {x.name: x for x in reader.tensors}["blk.0.ffn_gate_exps.weight"]
    f32 = gguf.dequantize(t.data, t.tensor_type).astype("float32", copy=False).reshape(-1)
    shape = [int(d) for d in t.shape.tolist()]
    k_dim, n_dim = shape[0], shape[1]
    for e in range(planes):
        got, _scale = proof_i8_expert_plane_from_manifest(manifest, "blk.0.ffn_gate_exps.weight", e)
        plane = f32[e * k_dim * n_dim : (e + 1) * k_dim * n_dim]
        want, _ = quantize_proof_i8(
            proof_f32_weight_matrix_from_gguf_f32(plane, [k_dim, n_dim])
        )
        assert np.array_equal(got, want), f"expert {e} plane mismatch"


def test_prove_mul_mat_id_expert_witness(tmp_path):
    """A MUL_MAT_ID trace (one routed expert's vector GEMM) proves and
    verifies against the manifest's per-plane commitment."""
    import gguf
    import json as _json
    import numpy as np
    from verallm.mesh.gguf_manifest import (
        build_gguf_tensor_manifest,
        proof_i8_expert_plane_from_manifest,
    )
    from verallm.mesh.ggml_proof import GgmlMulMatTrace, prove_ggml_mul_mat_trace

    model_path = str(tmp_path / "moe.gguf")
    writer = gguf.GGUFWriter(model_path, "llama")
    rng = np.random.default_rng(11)
    k_dim, n_dim, n_exp = 16, 6, 4
    experts = rng.standard_normal((n_exp, n_dim, k_dim), dtype=np.float32)
    writer.add_tensor("blk.0.ffn_down_exps.weight", experts)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    manifest = build_gguf_tensor_manifest(model_path)

    expert = 2
    w_i8, w_scale = proof_i8_expert_plane_from_manifest(
        manifest, "blk.0.ffn_down_exps.weight", expert
    )
    w_f32 = w_i8.astype(np.float32) * w_scale  # [k, n]
    x = rng.standard_normal(k_dim).astype(np.float32)
    y = (w_f32.T @ x).astype(np.float32)  # [n]

    tdir = tmp_path / "traces"
    tdir.mkdir()
    (tdir / "t-src1.f32").write_bytes(x.tobytes())
    (tdir / "t-dst.f32").write_bytes(y.tobytes())
    meta = {
        "version": 1,
        "created_unix_ns": 123,
        "manifest_index": 1,
        "graph_id": "cuda0-mul-mat-1",
        "op_index": 1,
        "op_type": "GGML_OP_MUL_MAT_ID",
        "expert_index": expert,
        "tensor_name": "blk.0.ffn_down_exps.weight",
        "src0_name": "blk.0.ffn_down_exps.weight",
        "src1_name": "ffn_moe_in",
        "dst_name": "ffn_moe_out",
        "src0_shape": [k_dim, n_dim, 1, 1],
        "src1_shape": [k_dim, 1, 1, 1],
        "dst_shape": [n_dim, 1, 1, 1],
        "src1_f32": "t-src1.f32",
        "dst_f32": "t-dst.f32",
        "src0_raw_type": "f32",
        "source_types": {"src0": "f32", "src1": "f32", "dst": "f32"},
        "backend": "llama_cpp_cuda",
        "device": "CUDA0",
    }
    (tdir / "t.json").write_text(_json.dumps(meta))
    trace = GgmlMulMatTrace.from_json(tdir / "t.json")
    assert trace.expert_index == expert

    ctx = {
        "request_id": "moe-test",
        "mesh_id": "moe-test",
        "mesh_spec_hash": "00" * 32,
        "stage_assignment_hash": "11" * 32,
        "rpc_plan_hash": "22" * 32,
        "uid": 0,
        "hotkey": "moe-test",
        "endpoint": "http://127.0.0.1:0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "request_hash": "33" * 32,
        "response_hash": "44" * 32,
    }
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        gguf_manifest=manifest,
        verify_before_return=True,
    )
    assert proof.verified, getattr(proof, "receipt", None)


def test_gemm_v2_sidecar_rides_the_payload_and_is_enforced(tmp_path):
    """The v2 sidecar must be attached, committed, and actually checked."""
    import copy

    from verallm.mesh.gemm_v2_sidecar import gemm_v2_available

    if not gemm_v2_available():
        pytest.skip("native PCS library is not built on this host")

    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    trace = _write_ggml_mul_mat_trace(tmp_path)
    proof = prove_ggml_mul_mat_trace(
        trace,
        _receipt_context_for_trace(mesh_spec, worker_endpoint),
        tolerance_abs=1e-6,
        include_proof=True,
    )
    payload = proof.proof_payload
    assert isinstance(payload.get("gemm_v2"), dict)
    assert payload["gemm_v2"]["blocks"], "sidecar proved no blocks"

    # A tampered sumcheck wire proof must fail independent verification.
    tampered = copy.deepcopy(payload)
    wire = bytearray.fromhex(tampered["gemm_v2"]["sumcheck_wire"])
    wire[len(wire) // 2] ^= 1
    tampered["gemm_v2"]["sumcheck_wire"] = bytes(wire).hex()
    result = verify_ggml_gemm_proof_payload(tampered)
    assert result.verified is False
    assert "gemm-v2" in result.message

    # Stripping the sidecar breaks the payload commitment the receipt binds.
    stripped = copy.deepcopy(payload)
    del stripped["gemm_v2"]
    result = verify_ggml_gemm_proof_payload(
        stripped, receipt=proof.receipt.to_dict()
    )
    assert result.verified is False


def test_read_boundary_roots_for_window(tmp_path):
    from verallm.mesh.ggml_proof import read_boundary_roots_for_window

    path = tmp_path / "boundary.jsonl"

    assert read_boundary_roots_for_window(tmp_path) is None

    row = {
        "row": "VERATHOS_GGML_BOUNDARY_V1",
        "created_unix_ns": 2_000_000_000_000_000_000,
        "capture_token": "req-1",
        "input_boundary_root": "aa" * 32,
        "output_boundary_root": "bb" * 32,
        "input_bytes": 4096,
        "output_bytes": 4096,
    }
    path.write_text(json.dumps(row) + "\n")
    roots = read_boundary_roots_for_window(
        tmp_path, start_unix_ns=row["created_unix_ns"] - 1_000
    )
    assert roots == {
        "input_boundary_root": "aa" * 32,
        "output_boundary_root": "bb" * 32,
    }

    # A row older than the window belongs to a previous request.
    assert (
        read_boundary_roots_for_window(
            tmp_path, start_unix_ns=row["created_unix_ns"] + 10_000_000_000
        )
        is None
    )

    # Zero absorbed bytes means the chain end is open, not vacuously hashed.
    row["input_bytes"] = 0
    path.write_text(json.dumps(row) + "\n")
    roots = read_boundary_roots_for_window(tmp_path)
    assert roots == {
        "input_boundary_root": "",
        "output_boundary_root": "bb" * 32,
    }


def test_prompt_cache_stays_on_except_explicit_replay_bypass():
    """The cache is ON for every serve; only replays disable it explicitly.

    Canary and organic traffic arrive in the same signed postcommit shape,
    so any cache rule keyed on request contents would be applied to both --
    which is exactly why the old blanket cache_prompt=false made every
    multi-turn conversation re-prefill its whole history each message. The
    cache now stays on uniformly; the only bypass is prompt_cache_disabled,
    set by witness-regenerating replays and the zero-witness fallback serve,
    both keyed on observable state rather than on anything in the request.
    """
    from verallm.mesh.worker import backend_openai_request

    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    # Sampler-bound serves (canary and organic alike) keep the cache on.
    sampler_bound = backend_openai_request(
        request,
        proof_capture_required=True,
        verified_sampler_required=True,
    )
    assert sampler_bound["cache_prompt"] is True

    # Metadata-only traffic (no verified sampler) sends no cache field.
    organic = backend_openai_request(
        request,
        proof_capture_required=True,
        verified_sampler_required=False,
    )
    assert "cache_prompt" not in organic

    # Witness-regenerating replays bypass the cache explicitly, and the
    # bypass overrides the sampler controls.
    replay = backend_openai_request(
        request,
        proof_capture_required=True,
        verified_sampler_required=True,
        prompt_cache_disabled=True,
    )
    assert replay["cache_prompt"] is False


def _anchored_audit_fixture(tmp_path, *, rows_total: int | None = None):
    """A layer GEMM anchored for exactly ONE selected op execution.

    Anchoring is armed per request through the capture token and gated on
    the selected-instance predicate, so each stream holds exactly that
    execution's rows. Its row count must therefore equal the op-manifest
    entry's own src1_shape[1]; rows_total exists only so tests can build a
    deliberately mis-scoped stream and prove it is rejected.
    """

    from verallm.mesh.anchor_streams import AnchorStageStream
    from verallm.mesh.execution_anchor import (
        StreamingExecutionAnchorV3,
        execution_anchor_row_leaf_hash_v3,
    )
    from verallm.mesh.gguf_manifest import proof_f32_weight_matrix_from_manifest

    trace = _write_ggml_mul_mat_trace(tmp_path)
    manifest = _gguf_manifest_for_trace(trace)
    k = int(trace.src0_shape[0])
    n = int(trace.src0_shape[1])
    if rows_total is None:
        # The honest shape: exactly the selected execution's rows.
        rows_total = int(trace.src1_shape[1])
    w_kn = proof_f32_weight_matrix_from_manifest(manifest, trace.src0_name)
    rng = np.random.default_rng(11)
    x_full = np.ascontiguousarray(
        rng.uniform(-1.0, 1.0, size=(rows_total, k)).astype(np.float32)
    )
    y_full = np.ascontiguousarray((x_full @ w_kn).astype(np.float32))
    streams = {}
    row_dumps = {}
    for side, matrix, width in (("src1", x_full, k), ("dst", y_full, n)):
        stage_id = f"{trace.src0_name}:{side}"
        anchor = StreamingExecutionAnchorV3(
            stage_id=stage_id, row_width=width * 4
        )
        leaves = []
        rows = {}
        for index in range(rows_total):
            row_bytes = matrix[index].tobytes()
            leaf = execution_anchor_row_leaf_hash_v3(
                stage_id=stage_id,
                row_index=index,
                row_width=width * 4,
                row_bytes=row_bytes,
            )
            leaves.append(leaf)
            anchor.append_leaf_hash(leaf)
            rows[index] = row_bytes
        streams[stage_id] = AnchorStageStream(
            stage_id=stage_id,
            row_width=width * 4,
            leaf_hashes=tuple(leaves),
            commitment=anchor.commitment(),
        )
        row_dumps[stage_id] = rows
    return trace, manifest, streams, row_dumps


def _anchored_receipt_context(tmp_path, manifest):
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
        model_tensor_manifest_root=manifest["tensor_manifest_root"],
    )
    ctx = _receipt_context_for_trace(mesh_spec, worker_endpoint)
    ctx.update(
        {
            "layer_start": 0,
            "layer_end": mesh_spec.total_layers,
            "verification_snapshot_hash": "11" * 32,
            "stage_id": "stg_" + "22" * 16,
            "stage_proof_commitment": "33" * 32,
            "model_index": 3,
            "proof_gate_hash": "44" * 32,
            "proof_beacon": "13" * 32,
            "proof_policy_version": 1,
            "proof_policy_profile": "gguf_mesh_v1",
            "proof_receipt_format": "opaque_stage_v2",
            "proof_trace_manifest_format": "compact-raw-v3",
            "proof_sample_bps": 10_000,
            "proof_sample_denominator": 10_000,
            "proof_ops_per_request": 1,
            "proof_trace_candidates_per_request": 1024,
            "proof_challenge_kind": "fiat_shamir_inline_v1",
            "proof_deferred": False,
            "verified_sampler_required": True,
            "verified_sampler_mode": (
                "deterministic_no_penalty_top_k_1_seed_0_prompt_cache_v2"
            ),
            "verified_sampler_controls_hash": "55" * 32,
            "proof_trace_scope": "op_manifest_challenge_v1",
            "decode_audit_mode": VERATHOS_GGUF_DECODE_AUDIT_MODE,
            "decode_audit_bps": 0,
            "decode_audit_top_k": 8,
            "decode_audit_stage_index": 1,
        }
    )
    return ctx


def _prove_anchored(tmp_path, trace, manifest, streams, row_dumps):
    """Prove one anchored op; raises when the anchored binding refuses."""

    ctx = _anchored_receipt_context(tmp_path, manifest)
    manifest_entries = find_op_manifest_entries_for_window(
        tmp_path / "ggml-traces",
        start_unix_ns=0,
        end_unix_ns=time.time_ns() + 1,
    )
    proof = prove_ggml_mul_mat_trace(
        trace,
        ctx,
        include_proof=True,
        gguf_manifest=manifest,
        op_manifest_membership=op_manifest_membership_payload(
            manifest_entries,
            trace,
            stage_index=int(ctx["stage_index"]),
        ),
        anchored_rows={"streams": streams, "row_dumps": row_dumps},
    )
    return proof, ctx


def _anchored_proof(tmp_path, *, rows_total: int | None = None):
    from verallm.mesh.anchor_streams import anchor_inventory_digest

    trace, manifest, streams, row_dumps = _anchored_audit_fixture(
        tmp_path, rows_total=rows_total
    )
    proof, ctx = _prove_anchored(tmp_path, trace, manifest, streams, row_dumps)
    receipt = proof.receipt.to_dict()
    receipt["proof_anchor_inventory_digest"] = anchor_inventory_digest(streams)
    return proof, receipt, streams, ctx


def _verify_anchored(proof_payload, receipt, ctx):
    return verify_ggml_gemm_proof_payload(
        proof_payload, receipt=receipt, mesh_receipt=ctx
    )


def test_anchored_row_audit_proves_and_verifies_reduced_gemm(tmp_path):
    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    payload = proof.proof_payload
    assert proof.verified is True
    assert payload["trace_io_layout"] == "anchored_rows_v1"
    block = payload["anchor_row_openings"]
    indexes = block["row_indexes"]
    # One selected op execution, so the stream is that execution's rows and
    # the draw comes from the frozen op-manifest entry, not the stream.
    entry_rows = int(proof.proof_payload["trace"]["src1_shape"][1])
    assert len(indexes) == min(4, entry_rows)
    assert 0 in indexes and entry_rows - 1 in indexes
    # The reduced GEMM has exactly the opened rows, not the full context.
    assert payload["input_shape"][0] == len(indexes)
    assert payload["output_shape"][0] == len(indexes)
    assert len(block["inventory"]) == 2
    assert _verify_anchored(payload, receipt, ctx).verified


def test_anchored_row_audit_rejects_a_mis_scoped_stream(tmp_path):
    """A stream spanning more than the selected execution must be rejected.

    Without this, row_count is prover-chosen: anchor every execution of the
    tensor, then truncate to exactly the rows you intend to open. The frozen
    op-manifest entry pins the true row count.
    """

    trace, manifest, streams, row_dumps = _anchored_audit_fixture(
        tmp_path, rows_total=20
    )
    entry_rows = int(trace.src1_shape[1])
    assert entry_rows != 20
    with pytest.raises(RuntimeError, match="was not scoped"):
        _prove_anchored(tmp_path, trace, manifest, streams, row_dumps)


def test_anchored_row_audit_rejects_entry_hash_substitution(tmp_path):
    """The row draw is bound to the frozen entry, not to payload content."""

    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    payload = json.loads(json.dumps(proof.proof_payload))
    # Same geometry, different committed entry: the verifier re-derives the
    # row set from the entry hash, so the declared rows no longer match.
    payload["op_manifest_membership"]["entry"]["graph_id"] = "graph-other"
    result = verify_ggml_gemm_proof_payload(payload, mesh_receipt=ctx)
    assert not result.verified


def test_anchored_row_audit_still_enforces_a_committed_inventory(tmp_path):
    """When a receipt DOES commit an inventory, equality is still required.

    The mesh audit anchors during the post-nonce replay, so no origin digest
    normally exists. This keeps the pre-nonce lane honest for the future.
    """

    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    wrong = {**receipt, "proof_anchor_inventory_digest": "9a" * 16}
    result = verify_ggml_gemm_proof_payload(
        proof.proof_payload, receipt=wrong, mesh_receipt=ctx
    )
    assert not result.verified
    assert "frozen receipt digest" in result.message


def test_anchored_row_audit_rejects_tampered_row_bytes(tmp_path):
    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    payload = json.loads(json.dumps(proof.proof_payload))
    opening = payload["anchor_row_openings"]["src1"]["openings"][0]
    raw = bytearray(bytes.fromhex(opening["row_hex"]))
    raw[0] ^= 1
    opening["row_hex"] = bytes(raw).hex()
    result = _verify_anchored(payload, receipt, ctx)
    assert not result.verified


def test_anchored_row_audit_rejects_substituted_row_indexes(tmp_path):
    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    payload = json.loads(json.dumps(proof.proof_payload))
    indexes = list(payload["anchor_row_openings"]["row_indexes"])
    swapped = sorted(set(indexes[:-1] + [indexes[-1] - 1]))
    payload["anchor_row_openings"]["row_indexes"] = swapped
    # No receipt: the payload-commitment gate would fire first and mask the
    # anchored check this test pins.
    result = verify_ggml_gemm_proof_payload(payload, mesh_receipt=ctx)
    assert not result.verified
    assert "beacon" in result.message


def test_anchored_layout_without_openings_is_rejected(tmp_path):
    proof, receipt, _streams, ctx = _anchored_proof(tmp_path)
    payload = json.loads(json.dumps(proof.proof_payload))
    del payload["anchor_row_openings"]
    result = verify_ggml_gemm_proof_payload(payload, mesh_receipt=ctx)
    assert not result.verified
    assert "missing its row openings" in result.message


def test_mesh_trace_capture_accepts_selected_anchor_rows(tmp_path):
    trace_enable_file = tmp_path / ".capture-enabled"
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Worker",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        proof_trace_enable_file=trace_enable_file,
    )
    host, port = worker.server_address
    try:
        response = post_json(
            f"http://{host}:{port}/v1/mesh/trace-capture",
            {
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "request_id": "req-anchor-rows",
                "enabled": True,
                "selected_ops": ["3:7"],
                "selected_anchor_rows": [0, 5, 44, 990],
            },
        )
        assert response["capture_enabled"] is True
        token = trace_enable_file.read_text(encoding="utf-8")
        assert "|selected_v3=3:7" in token
        assert "|anchor_rows=0,5,44,990" in token
    finally:
        post_json(
            f"http://{host}:{port}/v1/mesh/trace-capture",
            {
                "mesh_spec_hash": mesh_spec.spec_hash_hex(),
                "request_id": "req-anchor-rows",
                "enabled": False,
            },
        )
        _shutdown_server(worker, worker_thread)


def _template_entry(
    tmp_path, *, index, name, src1_cols, src0_planes=1, intra
):
    return GgmlOpManifestEntry(
        path=tmp_path / "manifest.vmanifest",
        created_unix_ns=1000 + index,
        manifest_index=index,
        graph_id=f"cuda3-mul-mat-{index}",
        op_index=index,
        op_type="GGML_OP_MUL_MAT",
        tensor_name=name,
        src0_name=name,
        src1_name=f"src1-{index}",
        dst_name=f"dst-{index}",
        src0_shape=(6144, 2048, src0_planes, 1),
        src1_shape=(6144, src1_cols, 1, 1),
        dst_shape=(2048, src1_cols, 1, 1),
        source_types={"src0": "q5_K", "src1": "f32", "dst": "f32"},
        backend="llama_cpp_cuda",
        device="CUDA3",
        graph_seq=1,
        intra_graph_index=intra,
    )


def test_decode_shaped_majority_rule_tolerates_fixed_width_tile_ops(tmp_path):
    """glm-dsa decode graphs mix single-row GEMMs with 32-wide indexer tile
    ops. The old all-rows rule classified every such decode graph as prefill,
    so the template fell back to a REAL prefill graph and every slot-view or
    audit draw failed with intras matching no live decode instance."""
    from verallm.mesh.ggml_proof import (
        _slot_view_graph_is_decode_shaped,
        slot_view_template_from_manifest_entries,
    )

    decode_graph = [
        _template_entry(tmp_path, index=0, name="blk.60.attn_q_a.weight", src1_cols=1, intra=0),
        _template_entry(tmp_path, index=1, name="blk.60.attn_q_b.weight", src1_cols=1, intra=1),
        _template_entry(tmp_path, index=2, name="blk.60.indexer.attn_k.weight", src1_cols=32, intra=2),
        _template_entry(tmp_path, index=3, name="blk.60.ffn_down_shexp.weight", src1_cols=1, intra=3),
        # 3D expert tensor: batched slots even at decode, exempt from the rule.
        _template_entry(tmp_path, index=4, name="blk.60.ffn_gate_exps.weight", src1_cols=8, src0_planes=256, intra=4),
        _template_entry(tmp_path, index=5, name="output.weight", src1_cols=1, intra=5),
    ]
    assert _slot_view_graph_is_decode_shaped(decode_graph)

    # A prefill graph (every layer GEMM chunk-width, logit row single) must
    # stay on the other side of the majority rule.
    prefill_graph = [
        _template_entry(tmp_path, index=0, name="blk.60.attn_q_a.weight", src1_cols=32, intra=0),
        _template_entry(tmp_path, index=1, name="blk.60.attn_q_b.weight", src1_cols=32, intra=1),
        _template_entry(tmp_path, index=2, name="blk.60.ffn_down_shexp.weight", src1_cols=32, intra=3),
        _template_entry(tmp_path, index=3, name="output.weight", src1_cols=1, intra=5),
    ]
    assert not _slot_view_graph_is_decode_shaped(prefill_graph)

    # Template from the decode graph drops the tile op (it has no single-row
    # decode instance to open) and keeps everything else, experts included.
    template = slot_view_template_from_manifest_entries(decode_graph)
    names = [op["tensor_name"] for op in template]
    assert "blk.60.indexer.attn_k.weight" not in names
    assert "blk.60.attn_q_b.weight" in names
    assert "blk.60.ffn_gate_exps.weight" in names
    assert "output.weight" in names

    # The prefill fallback is kept whole: filtering there would drop every
    # chunk-width row and empty the template.
    fallback = slot_view_template_from_manifest_entries(prefill_graph)
    assert [op["tensor_name"] for op in fallback] == [
        "blk.60.attn_q_a.weight",
        "blk.60.attn_q_b.weight",
        "blk.60.ffn_down_shexp.weight",
        "output.weight",
    ]


def test_postcommit_pending_reservation_permits_honest_canary_overlap():
    """Worst-case pending reservations (128MB each against a 256MB
    per-principal budget) allowed only TWO overlapping deferred-audit
    origins per validator; an honest fast-epoch cadence (several
    full-context canaries plus retries inside one 15-minute origin TTL)
    aborted the third canary mid-stream and sent a freshly registered
    index straight to probation  - and
    probation's 100% canary rate makes the overlap WORSE, so rehab could
    never succeed. The pending reservation must stay realistic: at least
    six overlapping pendings per principal, while the per-artifact hard
    cap still bounds any single artifact."""
    from verallm.mesh import worker as worker_module

    per_principal = worker_module.POSTCOMMIT_FINALIZED_MAX_BYTES_PER_PRINCIPAL
    pending = worker_module.POSTCOMMIT_PENDING_RESERVATION_BYTES
    assert per_principal // pending >= 6
    assert (
        worker_module.POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES > pending
    ), "the hard per-artifact cap is enforced at finalize time, not here"


# ── Capture-window liveness during exclusive replays ─────────────────────
#
# A hard audit's teacher-forced replay holds the EXCLUSIVE capture window
# (witness dumps must not contain another request's activations; graph
# ordinals restart at 1), so an overlapping organic chat must WAIT for the
# window to drain before its shared join can arm. Product rule: that wait
# may make the chat slower but must never make it look dead, and when the
# wait budget is exhausted the refusal must be the SAME generic retryable
# busy the admission ledger sends (canary-oracle rule: an audit-caused
# refusal must be indistinguishable from a saturation refusal).


def _post_sse_raw_lines(
    url: str,
    payload: dict,
    timeout: float = 15.0,
) -> list[str]:
    """Raw SSE lines INCLUDING comment frames (_post_sse_events drops them)."""

    req = Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    lines: list[str] = []
    with urlopen(req, timeout=timeout) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            lines.append(raw.decode("utf-8").rstrip("\r\n"))
    return lines


def _capture_liveness_worker(tmp_path):
    """Deferred-audit streaming coordinator whose serves join the capture
    window (same recipe as the deferred-audit stream test above)."""

    trace_enable_file = tmp_path / ".capture-enabled"
    backend, backend_thread, backend_url, calls = _fake_openai_backend(
        content="mesh stream pong",
        on_request=lambda _payload: _write_ggml_mul_mat_trace(tmp_path),
    )
    worker_endpoint = "http://worker.local:9338"
    mesh_spec = _two_member_mesh(
        coordinator_endpoint="http://coord.local:9338",
        worker_endpoint=worker_endpoint,
    )
    capability = CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    worker, worker_thread = serve_worker_in_thread(
        capability=capability,
        mesh_spec=mesh_spec,
        backend_url=backend_url,
        require_proof=True,
        defer_proof=True,
        proof_trace_enable_file=trace_enable_file,
        proof_trace_dir=tmp_path / "ggml-traces",
        proof_tolerance_abs=1e-6,
        proof_sample_bps=10_000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=1,
    )
    return worker, worker_thread, backend, backend_thread, mesh_spec, calls


def _set_exclusive_window(base_url: str, spec, *, enabled: bool) -> None:
    post_json(
        f"{base_url}/v1/mesh/trace-capture",
        {
            "mesh_spec_hash": spec.spec_hash_hex(),
            "enabled": enabled,
            "mode": "exclusive",
            "request_id": "hard-audit-window",
        },
    )


def test_organic_stream_stays_alive_and_completes_after_exclusive_window(
    tmp_path,
    monkeypatch,
):
    from verallm.mesh import worker as worker_module

    monkeypatch.setattr(worker_module, "CAPTURE_WAIT_TICK_S", 0.05)
    monkeypatch.setattr(worker_module, "ORGANIC_CAPTURE_WAIT_S", 10.0)
    worker, worker_thread, backend, backend_thread, mesh_spec, calls = (
        _capture_liveness_worker(tmp_path)
    )
    host, port = worker.server_address
    base_url = f"http://{host}:{port}"
    release = threading.Timer(
        0.7,
        _set_exclusive_window,
        args=(base_url, mesh_spec),
        kwargs={"enabled": False},
    )
    try:
        _set_exclusive_window(base_url, mesh_spec, enabled=True)
        release.start()
        request = {
            "model": "model",
            "messages": [
                {"role": "user", "content": "stream behind a hard replay"}
            ],
            "stream": True,
        }
        lines = _post_sse_raw_lines(
            f"{base_url}/v1/chat/completions",
            request,
        )
        keepalive_indexes = [
            index for index, line in enumerate(lines) if line.startswith(":")
        ]
        first_data_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("data:")
        )
        # The wait was covered by comment frames BEFORE the first data byte:
        # the stream was provably alive while the exclusive window drained.
        assert keepalive_indexes, "waiting stream emitted no keepalive frames"
        assert keepalive_indexes[0] < first_data_index
        # Comment frames are pure SSE comments: generic, unlabeled bytes.
        for index in keepalive_indexes:
            assert lines[index] == ": keepalive"
        # Once the window drained the chat completed normally end-to-end.
        assert "event: done" in lines
        assert any("[DONE]" in line for line in lines)
        # The backend ran exactly once, and only after the window drained -
        # the wait never leaked organic execution into the audit window.
        assert len(calls) == 1
    finally:
        release.cancel()
        _set_exclusive_window(base_url, mesh_spec, enabled=False)
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_stream_capture_wait_timeout_is_generic_retryable_busy(
    tmp_path,
    monkeypatch,
):
    from verallm.mesh import worker as worker_module

    monkeypatch.setattr(worker_module, "CAPTURE_WAIT_TICK_S", 0.05)
    monkeypatch.setattr(worker_module, "ORGANIC_CAPTURE_WAIT_S", 0.3)
    worker, worker_thread, backend, backend_thread, mesh_spec, calls = (
        _capture_liveness_worker(tmp_path)
    )
    host, port = worker.server_address
    base_url = f"http://{host}:{port}"
    try:
        _set_exclusive_window(base_url, mesh_spec, enabled=True)
        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "stream into a wall"}],
            "stream": True,
        }
        events = _post_sse_events(
            f"{base_url}/v1/chat/completions",
            request,
        )
        error_events = [
            payload
            for event, payload in events
            if event == "error" and isinstance(payload, dict)
        ]
        assert error_events, f"expected an SSE error event, got {events!r}"
        error = error_events[0]
        assert error["type"] == "slots_busy"
        assert error["retryable"] is True
        assert error["error"] == (
            "all 1 generation slots are busy; retry or fail over"
        )
        # Canary-oracle rule: the refusal must not name its cause.
        lowered = json.dumps(error, sort_keys=True).lower()
        for oracle_word in ("audit", "replay", "capture", "canary", "exclusive"):
            assert oracle_word not in lowered
        assert events[-1] == ("message", "[DONE]")
        # The backend was never consulted: the request died at the window.
        assert calls == []
    finally:
        _set_exclusive_window(base_url, mesh_spec, enabled=False)
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_non_stream_capture_wait_timeout_matches_admission_busy_503(
    tmp_path,
    monkeypatch,
):
    from urllib.error import HTTPError as _HTTPError

    from verallm.mesh import worker as worker_module

    monkeypatch.setattr(worker_module, "ORGANIC_CAPTURE_WAIT_S", 0.3)
    worker, worker_thread, backend, backend_thread, mesh_spec, calls = (
        _capture_liveness_worker(tmp_path)
    )
    host, port = worker.server_address
    base_url = f"http://{host}:{port}"
    try:
        _set_exclusive_window(base_url, mesh_spec, enabled=True)
        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "json into a wall"}],
        }
        req = Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(request, sort_keys=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(_HTTPError) as raised:
            urlopen(req, timeout=10.0)
        assert raised.value.code == 503
        body = json.loads(raised.value.read().decode("utf-8"))
        # BYTE-IDENTICAL to the admission ledger's saturation refusal: an
        # audit-caused refusal must not be distinguishable from a busy one.
        assert body == {
            "error": "all 1 generation slots are busy; retry or fail over",
            "type": "slots_busy",
            "retryable": True,
        }
        assert calls == []
    finally:
        _set_exclusive_window(base_url, mesh_spec, enabled=False)
        _shutdown_server(worker, worker_thread)
        _shutdown_server(backend, backend_thread)


def test_validator_lane_keeps_the_short_capture_wait_budget():
    """The long wait is an ORGANIC concession; validator canaries keep the
    short budget and resolve a held window into their retryable busy
    machinery instead of squatting on a claimed nonce."""

    from verallm.mesh import worker as worker_module

    assert (
        worker_module.ORGANIC_CAPTURE_WAIT_S
        > worker_module.SHARED_CAPTURE_WAIT_S
    )
