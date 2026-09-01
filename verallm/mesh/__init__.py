"""Distributed GGUF mesh primitives for Verathos.

Every public name lazy-imports its home module on first attribute access
(PEP 562). The eager form pulled the whole proof stack — including torch
via ggml_proof — into every ``import verallm.mesh``, which made *every*
CLI start (chat, manage, fleet) pay seconds of imports on a quiet box and
minutes on one whose disk a 200 GB model load is saturating. Interactive
commands need none of that at startup.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {}


def _module(module: str, names: tuple[str, ...]) -> None:
    for name in names:
        _EXPORTS[name] = module


_module(
    "verallm.mesh.types",
    (
        "CapabilityAd",
        "MeshMember",
        "MeshSpec",
        "StageRange",
        "StageReceipt",
        "canonical_json_bytes",
        "load_capability_ad",
        "load_mesh_spec",
        "save_json",
        "stage_receipt_root",
    ),
)
_module(
    "verallm.mesh.proof",
    (
        "LLAMA_CPP_RPC_RECEIPT_PROOF_MODE",
        "RECEIPT_ONLY_PROOF_MODE",
        "VERATHOS_GGML_GEMM_PROOF_MODE",
        "VERATHOS_GGML_TRACE_PROOF_MODE",
        "VERATHOS_GGUF_DECODE_AUDIT_MODE",
        "VERATHOS_GGUF_DECODE_AUDIT_TOP_K",
        "VALIDATOR_POSTCOMMIT_CHALLENGE_KIND",
        "LlamaGraphOpReceipt",
        "MeshStageProofReceipt",
        "derive_mesh_deferred_audit_beacon",
        "derive_mesh_decode_audit_positions",
        "derive_mesh_postcommit_proof_beacon",
        "derive_mesh_proof_beacon",
        "is_proof_capable_rpc_worker_binary",
        "llama_graph_receipt_root",
        "llama_graph_receipt_root_hex",
        "mesh_stage_proof_receipt_root",
        "mesh_stage_proof_receipt_root_hex",
        "sign_mesh_stage_proof_receipt",
        "mesh_decode_audit_commitment_hash",
        "mesh_decode_audit_sample_value",
        "mesh_deferred_audit_commitment_hash",
        "mesh_deferred_audit_sample_commitment_hash",
        "mesh_deferred_audit_sample_value",
        "mesh_validator_challenge_nonce_commitment",
        "mesh_proof_gate_hash",
        "mesh_proof_sample_value",
        "mesh_receipt_hash",
        "mesh_response_commitment_hash",
        "normalize_deferred_randomness",
        "normalize_proof_sample_bps",
        "normalize_validator_nonce",
        "normalize_validator_challenge_nonce",
        "should_sample_mesh_deferred_audit",
        "should_sample_mesh_decode_audit",
        "should_sample_mesh_proof",
        "verify_llama_graph_proof_receipts",
        "verify_llama_graph_proof_receipts_for_mesh",
        "verify_mesh_stage_proof_receipts_for_snapshot",
    ),
)
_module(
    "verallm.mesh.worker",
    (
        "DEFAULT_WORKER_PORT",
        "POSTCOMMIT_AUDIT_PATH",
        "WorkerProbe",
        "build_deferred_audit_bundle",
        "deferred_audit_decision",
        "deferred_audit_context_from_receipt",
        "finalize_deferred_audit_context",
        "finalize_postcommit_audit_context",
        "mesh_deferred_audit_bundle_hash",
        "postcommit_audit_context_from_receipt",
        "postcommit_audit_decision",
        "probe_worker",
        "serve_worker",
        "serve_worker_in_thread",
        "verify_deferred_mesh_audit_bundle",
        "verify_mesh_inference_artifact",
        "verify_mesh_postcommit_artifact",
    ),
)
_module(
    "verallm.mesh.llama_cpp",
    (
        "DEFAULT_LLAMA_RPC_PORT",
        "DEFAULT_LLAMA_SERVER_PORT",
        "MeshRpcPlan",
        "build_llama_server_command",
        "build_rpc_worker_command",
        "command_preview",
        "normalize_rpc_endpoint",
        "rpc_plan_from_mesh",
    ),
)
_module(
    "verallm.mesh.ggml_proof",
    (
        "GgmlOpManifestEntry",
        "GgmlMulMatTrace",
        "VerifiedGgmlProof",
        "find_op_manifest_entries_for_window",
        "find_trace_for_window",
        "find_traces_for_window",
        "ggml_op_manifest_root",
        "ggml_op_manifest_summary_for_window",
        "ggml_decode_audit_receipt_root",
        "ggml_proof_payload_commitment_hash",
        "mesh_op_manifest_aggregate_root",
        "make_ggml_proof_server",
        "op_manifest_membership_payload",
        "prove_ggml_mul_mat_trace",
        "select_manifest_challenge_indexes",
        "select_decode_manifest_entries",
        "serve_ggml_proof_adapter",
        "verify_ggml_decode_audit_payloads",
        "verify_op_manifest_membership",
        "verify_ggml_gemm_proof_payload",
        "verify_ggml_gemm_proof_payloads",
        "warm_ggml_proof_adapter",
    ),
)
_module(
    "verallm.mesh.gguf_manifest",
    (
        "GGML_TRACE_MAX_ELEMS_FLOOR",
        "proof_i8_weight_matrix_from_manifest",
        "proof_i8_weight_matrix_from_gguf_f32",
        "quantize_proof_i8",
        "suggest_ggml_trace_max_elems_from_tensor_records",
    ),
)

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
