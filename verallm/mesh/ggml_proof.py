"""Verified GGML proof adapter for mesh-side llama.cpp traces."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import struct
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from verallm.api.serialization import from_dict, to_dict
from verallm.crypto.merkle import (
    FlatWeightMerkle,
    build_block_merkle,
    hash_block,
    hash_flat_chunk,
    hash_leaf,
    hash_node,
    verify_flat_chunk_merkle_path,
)
from verallm.crypto.transcript import Transcript
from verallm.mesh.proof import (
    VERATHOS_GGML_GEMM_PROOF_MODE,
    VERATHOS_GGML_LIGHT_PROOF_MODE,
    VERATHOS_GGUF_DECODE_AUDIT_MODE,
    VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_ABS,
    VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_REL,
    VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
    LlamaGraphOpReceipt,
    mesh_binding_violation,
    mesh_boundary_chain_required,
)
from verallm.mesh.gguf_manifest import (
    f32_tensor_from_manifest,
    gguf_tensor_opening,
    load_gguf_tensor_manifest,
    proof_f32_weight_matrix_from_manifest,
    proof_i8_expert_plane_from_manifest,
    proof_i8_weight_matrix_from_manifest,
    quantize_proof_i8,
    verify_gguf_tensor_opening,
)
from verallm.mesh.types import canonical_json_bytes
from verallm.prover.gemm_fast import GEMMProverFast
from verallm.sampling import logits_i32_from_bytes, logits_i32_to_bytes, verify_quantized_argmax
from verallm.verifier.gemm import GEMMVerifier
from zkllm.config import get_config
from zkllm.crypto.field import mod_p
from zkllm.crypto.field_fast import tensor_to_field_vec
from zkllm.crypto.mle_fast import build_mle_from_matrix_fast

GGML_TRACE_WINDOW_SLACK_NS = 100_000_000
from zkllm.crypto.sumcheck import sumcheck_verify
from zkllm.types import GEMMBlockProof, GEMMProof, SpotCheck, SpotCheckWithProof

logger = logging.getLogger(__name__)


GGML_OP_MUL_MAT = "GGML_OP_MUL_MAT"
_GGUF_WEIGHT_MERKLE_CACHE_LOCK = threading.Lock()
_GGUF_WEIGHT_MERKLE_CACHE: dict[
    tuple[str, int, tuple[int, int], str],
    FlatWeightMerkle,
] = {}
# Sized to hold EVERY drawable tensor of a stage (~100 for a 7B), not a
# working set: draws are beacon-random, so a small LRU evicts constantly
# and under the compact disk profile each eviction is a 1-4 s
# derive+rebuild on the next draw of that tensor. Trees cost roughly half
# the tensor's i8 bytes in RAM (~1.5 GB for a full 7B stage), which is
# the deliberate trade for a 1.2 GB disk cache: hot in RAM, cold on GGUF.
_GGUF_WEIGHT_MERKLE_CACHE_MAX = int(
    os.environ.get("VERATHOS_WEIGHT_MERKLE_RAM_CACHE_ENTRIES", "256") or 256
)
GGML_CANONICAL_MIN_PROOF_BLOCK_SIZE = 64
GGML_CANONICAL_MIN_SPOT_CHECKS = 8
# Lower bound on proof-eligible MUL_MAT ops a transformer block contributes
# to the per-token challenge universe. Held well under what any real block
# emits so no legitimate architecture trips it; it exists only to reject a
# coordinator that commits a shrunken template to steer Fiat-Shamir draws.
MIN_PROOF_OPS_PER_LAYER = 3

_GGML_TRANSCRIPT_DOMAIN_V1 = b"VERATHOS_GGML_GEMM_TRANSCRIPT_V1"
_GGML_TRANSCRIPT_DOMAIN_V2 = b"VERATHOS_GGML_GEMM_TRANSCRIPT_V2"
_GGML_TRANSCRIPT_CONTEXT_DOMAIN = b"VERATHOS_GGML_TRANSCRIPT_CONTEXT_V1"
_GGML_MEMBERSHIP_CONTEXT_DOMAIN = b"VERATHOS_GGML_MEMBERSHIP_CONTEXT_V1"
_GGML_POLICY_CONTEXT_DOMAIN = b"VERATHOS_GGML_PROOF_POLICY_CONTEXT_V1"
_OPAQUE_STAGE_ID_RE = re.compile(r"^stg_[0-9a-f]{32}$")
_LAYER_TENSOR_PATTERNS = (
    re.compile(r"(?:^|[./_])blk[./_](\d+)(?:[./_]|$)"),
    re.compile(r"(?:^|[./_])layers?[./_](\d+)(?:[./_]|$)"),
    re.compile(r"(?:^|[./_])blocks?[./_](\d+)(?:[./_]|$)"),
    re.compile(r"(?:^|[./_])h[./_](\d+)(?:[./_]|$)"),
)
_FIRST_STAGE_TENSOR_NAMES = frozenset(
    {
        "token_embd.weight",
        "tok_embeddings.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    }
)
_FINAL_STAGE_TENSOR_NAMES = frozenset(
    {
        "output.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "output_norm.weight",
        "model.norm.weight",
        "transformer.ln_f.weight",
    }
)
_GGML_POLICY_CONTEXT_FIELDS = (
    "proof_policy_version",
    "proof_policy_profile",
    "proof_receipt_format",
    "proof_trace_manifest_format",
    "proof_sample_bps",
    "proof_sample_denominator",
    "proof_ops_per_request",
    "proof_trace_candidates_per_request",
    "proof_challenge_kind",
    "proof_deferred",
    "verified_sampler_required",
    "verified_sampler_mode",
    "verified_sampler_controls_hash",
    "decode_audit_mode",
    "decode_audit_bps",
    "decode_audit_top_k",
    "decode_audit_stage_index",
)
_GGML_POLICY_INT_FIELDS = frozenset(
    {
        "proof_policy_version",
        "proof_sample_bps",
        "proof_sample_denominator",
        "proof_ops_per_request",
        "proof_trace_candidates_per_request",
        "decode_audit_bps",
        "decode_audit_top_k",
        "decode_audit_stage_index",
    }
)
_GGML_POLICY_BOOL_FIELDS = frozenset(
    {"proof_deferred", "verified_sampler_required"}
)


def _require_lower_hex_digest(value: Any, *, field_name: str) -> str:
    raw = str(value)
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(f"{field_name} must be a lowercase SHA-256 digest")
    return raw


def _normalized_ggml_policy_context(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact proof-policy fields bound into a secure transcript."""

    body: dict[str, Any] = {"version": 1}
    for field in _GGML_POLICY_CONTEXT_FIELDS:
        if field not in source:
            raise RuntimeError(f"secure GGML transcript is missing {field}")
        value = source[field]
        if field in _GGML_POLICY_INT_FIELDS:
            if type(value) is not int:
                raise RuntimeError(f"secure GGML transcript {field} must be an integer")
            body[field] = int(value)
        elif field in _GGML_POLICY_BOOL_FIELDS:
            if type(value) is not bool:
                raise RuntimeError(f"secure GGML transcript {field} must be boolean")
            body[field] = bool(value)
        else:
            if not isinstance(value, str):
                raise RuntimeError(f"secure GGML transcript {field} must be a string")
            body[field] = value
    return body


def ggml_proof_policy_context_hash(source: Mapping[str, Any]) -> str:
    """Hash the canonical policy knobs that govern one GGML proof request."""

    return hashlib.sha256(
        _GGML_POLICY_CONTEXT_DOMAIN
        + canonical_json_bytes(_normalized_ggml_policy_context(source))
    ).hexdigest()


def _membership_binding_material(
    membership_key: str,
    membership: Mapping[str, Any],
) -> dict[str, Any]:
    if membership_key == "trace_membership":
        root_key, count_key, index_key, leaf_key = (
            "trace_set_root",
            "trace_set_count",
            "trace_leaf_index",
            "trace_commitment_hash",
        )
    elif membership_key == "op_manifest_membership":
        root_key, count_key, index_key, leaf_key = (
            "op_manifest_root",
            "op_manifest_count",
            "op_manifest_leaf_index",
            "op_manifest_entry_hash",
        )
    elif membership_key == "slot_view_membership":
        root_key, count_key, index_key, leaf_key = (
            "slot_view_root",
            "slot_view_count",
            "slot_view_leaf_index",
            "slot_view_leaf_hash",
        )
    else:
        raise RuntimeError("unsupported GGML membership context")
    root = _require_lower_hex_digest(
        membership.get(root_key, ""), field_name=f"{membership_key}.{root_key}"
    )
    leaf = _require_lower_hex_digest(
        membership.get(leaf_key, ""), field_name=f"{membership_key}.{leaf_key}"
    )
    count = membership.get(count_key)
    index = membership.get(index_key)
    stage_index = membership.get("stage_index")
    if type(count) is not int or count <= 0:
        raise RuntimeError(f"{membership_key}.{count_key} must be positive")
    if type(index) is not int or index < 0 or index >= count:
        raise RuntimeError(f"{membership_key}.{index_key} is invalid")
    if type(stage_index) is not int or stage_index < 0:
        raise RuntimeError(f"{membership_key}.stage_index is invalid")
    return {
        "membership_key": membership_key,
        "stage_index": int(stage_index),
        "root": root,
        "count": int(count),
        "leaf_index": int(index),
        "leaf_hash": leaf,
    }


def _membership_context_binding(
    membership_key: str,
    membership: Mapping[str, Any],
    *,
    base_context_hash: str,
) -> str:
    material = _membership_binding_material(membership_key, membership)
    material["base_context_hash"] = _require_lower_hex_digest(
        base_context_hash, field_name="base_context_hash"
    )
    return hashlib.sha256(
        _GGML_MEMBERSHIP_CONTEXT_DOMAIN + canonical_json_bytes(material)
    ).hexdigest()


def _ggml_transcript_context_base(
    source: Mapping[str, Any],
    trace_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical request/stage context independently of payload labels."""

    secure = bool(
        source.get("verification_snapshot_hash")
        or source.get("stage_id")
        or source.get("model_index") is not None
    )
    common = {
        "version": 2 if secure else 1,
        "request_id": str(source.get("request_id", "")),
        "mesh_id": str(source.get("mesh_id", "")),
        "mesh_spec_hash": str(source.get("mesh_spec_hash", "")),
        "stage_assignment_hash": str(source.get("stage_assignment_hash", "")),
        "stage_index": int(source.get("stage_index", -1)),
        "layer_start": int(source.get("layer_start", -1)),
        "layer_end": int(source.get("layer_end", -1)),
        "model_package_hash": str(source.get("model_package_hash", "")),
        "model_tensor_manifest_root": str(
            source.get("model_tensor_manifest_root", "")
        ),
        "request_hash": str(source.get("request_hash", "")),
        "response_hash": str(source.get("response_hash", "")),
        "graph_id": str(trace_meta.get("graph_id", "")),
        "op_index": int(trace_meta.get("op_index", -1)),
    }
    if not common["request_id"] or not common["mesh_id"]:
        raise RuntimeError("GGML transcript request_id and mesh_id are required")
    if common["stage_index"] < 0 or not (
        0 <= common["layer_start"] < common["layer_end"]
    ):
        raise RuntimeError("GGML transcript stage range is invalid")
    if not common["graph_id"] or common["op_index"] < 0:
        raise RuntimeError("GGML transcript graph/op identity is invalid")
    for field in (
        "mesh_spec_hash",
        "stage_assignment_hash",
        "request_hash",
        "response_hash",
    ):
        _require_lower_hex_digest(common[field], field_name=field)
    for field in ("model_package_hash", "model_tensor_manifest_root"):
        if common[field]:
            _require_lower_hex_digest(common[field], field_name=field)
    if not secure:
        return common

    stage_id = str(source.get("stage_id", ""))
    if not _OPAQUE_STAGE_ID_RE.fullmatch(stage_id):
        raise RuntimeError("secure GGML transcript stage_id is invalid")
    snapshot_hash = _require_lower_hex_digest(
        source.get("verification_snapshot_hash", ""),
        field_name="verification_snapshot_hash",
    )
    gate_hash = _require_lower_hex_digest(
        source.get("proof_gate_hash", ""), field_name="proof_gate_hash"
    )
    model_index = source.get("model_index")
    model_total_layers = source.get("model_total_layers")
    if type(model_index) is not int or not 0 <= model_index < 2**32:
        raise RuntimeError("secure GGML transcript model_index is invalid")
    if type(model_total_layers) is not int or model_total_layers <= 0:
        raise RuntimeError("secure GGML transcript model_total_layers is invalid")
    if common["layer_end"] > model_total_layers:
        raise RuntimeError("secure GGML transcript stage exceeds model layers")
    if not common["model_package_hash"]:
        raise RuntimeError("secure GGML transcript model package commitment is required")
    policy_source = source.get("proof_policy_context")
    if not isinstance(policy_source, Mapping):
        policy_source = source
    policy_context = _normalized_ggml_policy_context(policy_source)
    common.update(
        {
            "verification_snapshot_hash": snapshot_hash,
            "stage_id": stage_id,
            "model_index": int(model_index),
            "model_total_layers": int(model_total_layers),
            "proof_gate_hash": gate_hash,
            "proof_policy_context": policy_context,
            "proof_policy_context_hash": hashlib.sha256(
                _GGML_POLICY_CONTEXT_DOMAIN + canonical_json_bytes(policy_context)
            ).hexdigest(),
        }
    )
    return common


def _ggml_transcript_context(
    source: Mapping[str, Any],
    trace_meta: Mapping[str, Any],
    memberships: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    context = _ggml_transcript_context_base(source, trace_meta)
    base_hash = hashlib.sha256(
        _GGML_TRANSCRIPT_CONTEXT_DOMAIN + canonical_json_bytes(context)
    ).hexdigest()
    bindings: dict[str, str] = {}
    for membership_key in (
        "trace_membership",
        "op_manifest_membership",
        "slot_view_membership",
    ):
        membership = memberships.get(membership_key)
        if not membership:
            continue
        if int(membership.get("stage_index", -1)) != int(context["stage_index"]):
            raise RuntimeError(
                f"{membership_key} stage does not match transcript stage"
            )
        bindings[membership_key] = _membership_context_binding(
            membership_key,
            membership,
            base_context_hash=base_hash,
        )
    context["membership_context_bindings"] = bindings
    return context


def _ggml_transcript_label(context: Mapping[str, Any]) -> bytes:
    version = int(context.get("version", 0))
    if version not in (1, 2):
        raise RuntimeError("unsupported GGML transcript context version")
    domain = _GGML_TRANSCRIPT_DOMAIN_V2 if version == 2 else _GGML_TRANSCRIPT_DOMAIN_V1
    return domain + hashlib.sha256(canonical_json_bytes(dict(context))).digest()


def derive_ggml_output_block_challenge(
    *,
    beacon: str | bytes,
    transcript_context: Mapping[str, Any],
    layer_index: int,
    output_shape: list[int] | tuple[int, int],
    block_size: int,
) -> tuple[int, int]:
    """Derive the snapshot-bound output block that must be proved.

    The beacon and signed snapshot/stage/model bindings select an output
    element independently of the operator-configured block size.  The selected
    block is whichever block contains that element, so increasing block size
    cannot steer the challenge away from it.
    """

    if int(transcript_context.get("version", 0)) != 2:
        raise RuntimeError("output block challenge requires transcript context v2")
    beacon_bytes = (
        bytes.fromhex(
            _require_lower_hex_digest(beacon, field_name="proof_beacon")
        )
        if isinstance(beacon, str)
        else bytes(beacon)
    )
    if len(beacon_bytes) != 32:
        raise RuntimeError("proof_beacon must be 32 bytes")
    if len(output_shape) != 2:
        raise RuntimeError("output block challenge requires a 2D output shape")
    rows, cols = int(output_shape[0]), int(output_shape[1])
    if rows <= 0 or cols <= 0:
        raise RuntimeError("output block challenge shape must be positive")
    size = int(block_size)
    if size <= 0:
        raise RuntimeError("output block challenge block_size must be positive")
    layer = int(layer_index)
    if layer < 0:
        raise RuntimeError("output block challenge layer_index is invalid")

    body = {
        "version": 1,
        "verification_snapshot_hash": _require_lower_hex_digest(
            transcript_context.get("verification_snapshot_hash", ""),
            field_name="verification_snapshot_hash",
        ),
        "proof_gate_hash": _require_lower_hex_digest(
            transcript_context.get("proof_gate_hash", ""),
            field_name="proof_gate_hash",
        ),
        "stage_id": str(transcript_context.get("stage_id", "")),
        "model_index": int(transcript_context.get("model_index", -1)),
        "layer_index": layer,
        "output_rows": rows,
        "output_cols": cols,
    }
    if not _OPAQUE_STAGE_ID_RE.fullmatch(body["stage_id"]):
        raise RuntimeError("output block challenge stage_id is invalid")
    if body["model_index"] < 0:
        raise RuntimeError("output block challenge model_index is invalid")
    material = canonical_json_bytes(body)
    target_row = int.from_bytes(
        hashlib.sha256(
            b"VERATHOS_GGML_OUTPUT_ELEMENT_ROW_V1"
            + beacon_bytes
            + material
        ).digest()[:8],
        "little",
    ) % rows
    target_col = int.from_bytes(
        hashlib.sha256(
            b"VERATHOS_GGML_OUTPUT_ELEMENT_COL_V1"
            + beacon_bytes
            + material
        ).digest()[:8],
        "little",
    ) % cols
    return target_row // size, target_col // size


def _trace_layer_index(
    trace_meta: Mapping[str, Any],
    *,
    layer_start: int,
    layer_end: int,
    model_total_layers: int | None = None,
) -> int:
    """Resolve and validate the signed stage that owns a proved tensor."""

    names = {
        str(trace_meta.get("tensor_name", "")).strip(),
        str(trace_meta.get("src0_name", "")).strip(),
    }
    names.discard("")
    indexes: set[int] = set()
    for name in names:
        for pattern in _LAYER_TENSOR_PATTERNS:
            match = pattern.search(name)
            if match:
                indexes.add(int(match.group(1)))
    if len(indexes) > 1:
        raise RuntimeError("GGML trace tensor names disagree on layer ownership")
    if indexes:
        layer_index = next(iter(indexes))
        if not layer_start <= layer_index < layer_end:
            raise RuntimeError("GGML trace tensor layer is outside the signed stage range")
        return layer_index

    normalized = {name.lower() for name in names}
    if normalized & _FIRST_STAGE_TENSOR_NAMES:
        if layer_start != 0:
            raise RuntimeError("GGML embedding tensor is outside the first stage")
        return 0
    if normalized & _FINAL_STAGE_TENSOR_NAMES:
        if model_total_layers is None or model_total_layers <= 0:
            raise RuntimeError("GGML final tensor ownership needs model_total_layers")
        if layer_end != model_total_layers:
            raise RuntimeError("GGML final tensor is outside the final stage")
        return model_total_layers - 1
    raise RuntimeError("GGML trace tensor layer ownership cannot be established")


def _sha256_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _array_root(array: np.ndarray, *, domain: bytes) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(
        domain,
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode(),
        str(contiguous.dtype).encode(),
        contiguous.tobytes(),
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Largest k-tile whose int8 GEMM partial sums stay exactly representable in
# float32: 1024 * 127 * 127 = 16,516,096 < 2^24. Every product and every
# running sum inside a tile is an integer below the mantissa bound, so the
# float GEMM result is bit-exact, not approximate.
_PROOF_I8_F32_TILE_K = 1024


def _proof_i8_matmul_i64_torch(x: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Exact int8 GEMM via k-tiled float32 torch matmul with int64 accumulation.

    This venv's numpy has no optimized BLAS, so ``x_i8 @ w_i8`` runs as a
    naive loop (~124 ms per LM-head row measured), which made the hard prove
    O(context) with a huge constant. Float tensor cores are exact for int8
    products when the k extent is tiled below the f32 mantissa bound, so this
    computes the identical int64 result at GEMM speed. Rows are chunked to
    bound peak memory. Returns None on any failure so the caller can fall
    back to the reference integer path.
    """

    if os.environ.get("VERATHOS_PROOF_GEMM_TORCH", "1") == "0":
        return None
    rows, k = int(x.shape[0]), int(x.shape[1])
    n = int(w.shape[1])
    try:
        forced = os.environ.get("VERATHOS_PROOF_GEMM_DEVICE", "").strip()
        if forced:
            device = torch.device(forced)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        wt = torch.from_numpy(np.ascontiguousarray(w)).to(device)
        # Cap the per-chunk f32 partial (n*4 bytes/row) plus i64 accumulator
        # (n*8 bytes/row) at ~1 GiB of transient device memory.
        chunk_rows = max(1, (1 << 30) // max(1, n * 12))
        tile = _PROOF_I8_F32_TILE_K
        out = np.empty((rows, n), dtype=np.int64)
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            acc = torch.zeros((r1 - r0, n), dtype=torch.int64, device=device)
            for k0 in range(0, k, tile):
                k1 = min(k, k0 + tile)
                part = xt[r0:r1, k0:k1].to(torch.float32) @ wt[k0:k1, :].to(
                    torch.float32
                )
                acc += part.to(torch.int64)
            out[r0:r1] = acc.cpu().numpy()
        return np.ascontiguousarray(out)
    except Exception:
        logger.warning(
            "proof int8 GEMM torch path failed; using integer fallback",
            exc_info=True,
        )
        return None


def _proof_i8_matmul_i64(x_i8: np.ndarray, w_i8: np.ndarray) -> np.ndarray:
    """Multiply proof-domain int8 matrices and return int64 outputs."""

    x = np.asarray(x_i8, dtype=np.int8)
    w = np.asarray(w_i8, dtype=np.int8)
    if x.ndim != 2 or w.ndim != 2 or int(x.shape[1]) != int(w.shape[0]):
        raise ValueError("proof-domain GEMM shapes do not align")
    fast = _proof_i8_matmul_i64_torch(x, w)
    if fast is not None:
        return fast
    k = int(x.shape[1])
    max_abs_sum = 127 * 127 * max(1, k)
    if max_abs_sum <= int(np.iinfo(np.int32).max):
        y_i32 = x.astype(np.int32, copy=False) @ w.astype(np.int32, copy=False)
        return np.ascontiguousarray(y_i32.astype(np.int64, copy=False))
    return np.ascontiguousarray(
        x.astype(np.int64, copy=False) @ w.astype(np.int64, copy=False)
    )


def _cached_gguf_weight_merkle(
    w_tensor: torch.Tensor,
    *,
    expected_root: str,
    chunk_size: int,
    i8_sha256: str = "",
    tensor_name: str = "",
) -> FlatWeightMerkle:
    """Return a cached manifest-bound FlatWeightMerkle for a GGUF weight tensor.

    Three tiers, cheapest first: a small in-process LRU, a byte-bounded
    on-disk store (survives restart, keyed by the weight's committed
    proof_i8_sha256), then a rebuild (GPU leaf hashing when the kernel is
    available: 0.8-3 s for layer tensors). Whether a rebuild is persisted
    follows the cache profile via should_persist_weight_merkle: compact
    persists only the final projection (its 4M+ leaves still build in
    ~22 s), full persists everything (CPU-only boxes, where any rebuild
    costs 6-200 s). The disk entry is always re-verified against the
    signed manifest root, so a stale or corrupt blob fails closed to a
    rebuild, never into a proof.
    """

    root = str(expected_root)
    key = (
        root,
        int(chunk_size),
        (int(w_tensor.shape[0]), int(w_tensor.shape[1])),
        str(w_tensor.dtype),
    )
    with _GGUF_WEIGHT_MERKLE_CACHE_LOCK:
        cached = _GGUF_WEIGHT_MERKLE_CACHE.get(key)
    if cached is not None:
        return cached

    merkle = None
    sha = str(i8_sha256 or "")
    if sha:
        from verallm.mesh.gguf_manifest import load_cached_weight_merkle

        disk = load_cached_weight_merkle(sha)
        if disk is not None and (not root or disk.root.hex() == root):
            merkle = disk

    built_now = False
    if merkle is None:
        merkle = FlatWeightMerkle(w_tensor, int(chunk_size), store_raw=False)
        if root and merkle.root.hex() != root:
            raise RuntimeError("GGUF proof weight Merkle root mismatch")
        built_now = True

    if built_now and sha:
        from verallm.mesh.gguf_manifest import (
            should_persist_weight_merkle,
            store_cached_weight_merkle,
        )

        if should_persist_weight_merkle(tensor_name):
            store_cached_weight_merkle(sha, merkle)

    with _GGUF_WEIGHT_MERKLE_CACHE_LOCK:
        if key not in _GGUF_WEIGHT_MERKLE_CACHE:
            if len(_GGUF_WEIGHT_MERKLE_CACHE) >= _GGUF_WEIGHT_MERKLE_CACHE_MAX:
                _GGUF_WEIGHT_MERKLE_CACHE.pop(next(iter(_GGUF_WEIGHT_MERKLE_CACHE)))
            _GGUF_WEIGHT_MERKLE_CACHE[key] = merkle
        return _GGUF_WEIGHT_MERKLE_CACHE[key]


def _is_final_projection_tensor_name(name: str) -> bool:
    lower = str(name).lower()
    return (
        lower == "output.weight"
        or lower == "lm_head.weight"
        or lower.endswith(".output.weight")
        or lower.endswith(".lm_head.weight")
    )


def _final_projection_tensor_names_from_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    records = manifest.get("tensors", [])
    if not isinstance(records, list):
        return []
    names: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("name", ""))
        if _is_final_projection_tensor_name(name):
            names.append(name)
    return sorted(set(names))


def warm_gguf_decode_projection_cache(
    manifest: Mapping[str, Any],
    *,
    max_tensors: int = 1,
) -> dict[str, Any]:
    """Preload final-projection proof weights used by decode canary audits."""

    started = time.perf_counter()
    warmed: list[dict[str, Any]] = []
    names = _final_projection_tensor_names_from_manifest(manifest)
    if max_tensors > 0:
        names = names[: int(max_tensors)]
    for tensor_name in names:
        opening = gguf_tensor_opening(manifest, tensor_name)
        tensor_leaf = opening.get("tensor_leaf", {})
        if not isinstance(tensor_leaf, Mapping):
            continue
        weight_root = str(tensor_leaf.get("proof_i8_merkle_root", ""))
        chunk_size = int(tensor_leaf.get("proof_i8_chunk_size", 0) or 0)
        if not weight_root or chunk_size <= 0:
            continue
        w_i8 = proof_i8_weight_matrix_from_manifest(manifest, tensor_name)
        # i8_sha256 makes the warmed tree PERSIST (store_cached_weight_merkle
        # keys on it); without it this warmed only the in-process LRU and the
        # next process rebuilt the LM-head tree from scratch.
        merkle = _cached_gguf_weight_merkle(
            torch.from_numpy(np.ascontiguousarray(w_i8)),
            expected_root=weight_root,
            chunk_size=chunk_size,
            i8_sha256=str(tensor_leaf.get("proof_i8_sha256", "")),
            tensor_name=tensor_name,
        )
        warmed.append(
            {
                "tensor_name": tensor_name,
                "shape": [int(item) for item in w_i8.shape],
                "proof_i8_merkle_root": merkle.root.hex(),
            }
        )
    return {
        "warmed": warmed,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def _trace_leaf_hash(index: int, trace_commitment_hash: str) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGML_TRACE_SET_LEAF_V1"
        + int(index).to_bytes(4, "little")
        + bytes.fromhex(trace_commitment_hash)
    ).digest()


def _trace_node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGML_TRACE_SET_NODE_V1" + left + right
    ).digest()


def _is_provable_weight_op(src0_shape) -> bool:
    """True when a traced MUL_MAT's weight is a real matrix we can prove.

    MoE graphs also run MUL_MAT over 1-D gate VECTORS (e.g. the shared-expert
    gate ffn_gate_inp_shexp, GGUF shape [2048], padded by ggml to
    [2048,1,1,1]). Those tensors have no committed proof matrix in the GGUF
    manifest (matrix records need two non-trivial dims), so selecting one can
    only ever fail verification. Selection walks forward past them
    deterministically; the committed candidate/manifest sets are unchanged.

    A 3-D src0 [k, n, n_experts] is a MUL_MAT_ID expert weight and IS
    provable: the manifest commits each expert plane separately and the
    prover binds the routed plane (expert_index) to its committed root. Only
    the two dump hooks feed this population, and the dense hook rejects
    ne[2] != 1 at capture, so dims[2] > 1 here always means an expert
    tensor. (Requiring dims[2] <= 1 was what silently excluded every expert
    GEMM from beacon selection.)
    """
    dims = [int(d) for d in src0_shape]
    return (
        len(dims) >= 2
        and all(d > 0 for d in dims)
        and dims[0] > 1
        and dims[1] > 1
        and all(d <= 1 for d in dims[3:])
    )


def _walk_to_provable_index(items, start: int, shape_of) -> int:
    """Compatibility guard for callers that used to remap challenged rows.

    Proof populations are now filtered before committing their root.  Moving
    a challenged index afterwards would change the FS statement, so this
    helper only accepts the already-provable row at the exact drawn index.
    """

    index = int(start)
    if index < 0 or index >= len(items):
        raise IndexError("proof challenge index out of range")
    if not _is_provable_weight_op(shape_of(items[index])):
        raise RuntimeError("unprovable row remained in canonical challenge domain")
    return index


def _witness_traces(traces: list["GgmlMulMatTrace"]) -> list["GgmlMulMatTrace"]:
    """Drop rpc graph-end mirror traces from witness candidacy.

    The rpc hook dumps its op AFTER the whole graph ran, but ggml reuses
    intermediate activation buffers within a graph, so a mid-graph mirror's
    src1/dst bytes are stale garbage and its float recomputation fails with
    ~1.0 relative error. Every compute stage's own at-execution hook
    (llama_cpp_cpu / llama_cpp_cuda / llama_cpp_metal) dumps the same ops
    consistently, so mirrors add no coverage — they only inject unprovable
    candidates that fail chats when the beacon draws one. Old mirror files
    still on disk make this filter necessary even after the serve stops
    arming VERATHOS_GGML_RPC_GRAPH_TRACE. Keyed on the mirror hook's own
    graph_id prefix ("rpc-dev<N>-graph<M>"), which no at-execution hook
    writes; the parsed ``backend`` field can't distinguish, because absent
    fields default to llama_cpp_rpc for legacy traces.
    """
    return [
        t for t in traces if not str(getattr(t, "graph_id", "")).startswith("rpc-dev")
    ]


def _ordered_traces(traces: list["GgmlMulMatTrace"]) -> list["GgmlMulMatTrace"]:
    """Return the canonical, proof-capable trace challenge population.

    Challenge indexes are positions in this population.  Filtering after an
    index is drawn changes the Fiat--Shamir domain and lets an unprovable row
    redirect a challenge to a different witness.  Keep the filter here so
    roots, counts, selection, and membership all see exactly the same rows.
    Three-dimensional MoE expert weights remain eligible; padded one-
    dimensional gate vectors do not.
    """

    return sorted(
        (
            item
            for item in traces
            if _is_provable_weight_op(item.src0_shape)
            and not str(getattr(item, "graph_id", "")).startswith("rpc-dev")
        ),
        key=lambda item: (
            int(item.created_unix_ns),
            int(item.manifest_index) if item.manifest_index is not None else -1,
            str(item.graph_id),
            int(item.op_index),
            str(item.tensor_name),
            item.path.name,
        ),
    )


def _ordered_manifest_entries(
    entries: list["GgmlOpManifestEntry"],
) -> list["GgmlOpManifestEntry"]:
    """Return the canonical, proof-capable manifest challenge population."""

    return sorted(
        (
            item
            for item in entries
            if bool(item.proof_eligible)
            and _is_provable_weight_op(item.src0_shape)
        ),
        key=lambda item: (
            int(item.created_unix_ns),
            int(item.manifest_index),
            str(item.backend),
            str(item.device),
            str(item.graph_id),
            int(item.op_index),
            str(item.tensor_name),
            item.path.name,
        ),
    )


def _trace_set_levels(commitments: list[str]) -> list[list[bytes]]:
    leaves = [_trace_leaf_hash(i, commitment) for i, commitment in enumerate(commitments)]
    if not leaves:
        return []
    levels = [leaves]
    current = leaves
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else left
            next_level.append(_trace_node_hash(left, right))
        levels.append(next_level)
        current = next_level
    return levels


def _trace_membership_path(commitments: list[str], leaf_index: int) -> list[dict[str, Any]]:
    if leaf_index < 0 or leaf_index >= len(commitments):
        raise IndexError("trace leaf index out of range")
    levels = _trace_set_levels(commitments)
    path: list[dict[str, Any]] = []
    idx = int(leaf_index)
    for level in levels[:-1]:
        if idx % 2 == 0:
            sibling_idx = idx + 1 if idx + 1 < len(level) else idx
            sibling_is_left = False
        else:
            sibling_idx = idx - 1
            sibling_is_left = True
        path.append(
            {
                "sibling": level[sibling_idx].hex(),
                "sibling_is_left": bool(sibling_is_left),
            }
        )
        idx //= 2
    return path


def _trace_set_root_from_commitments(commitments: list[str]) -> str:
    levels = _trace_set_levels(commitments)
    return levels[-1][0].hex() if levels else ""


def _manifest_leaf_hash(index: int, entry_hash: str) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGML_OP_MANIFEST_LEAF_V1"
        + int(index).to_bytes(8, "little", signed=False)
        + bytes.fromhex(entry_hash)
    ).digest()


def _manifest_leaf_hash_bytes(index: int, entry_hash: bytes) -> bytes:
    if len(entry_hash) != 32:
        raise ValueError("manifest entry hash must be 32 bytes")
    return hashlib.sha256(
        b"VERATHOS_GGML_OP_MANIFEST_LEAF_V1"
        + int(index).to_bytes(8, "little", signed=False)
        + entry_hash
    ).digest()


def _manifest_node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGML_OP_MANIFEST_NODE_V1" + left + right
    ).digest()


def _manifest_levels(entry_hashes: list[str]) -> list[list[bytes]]:
    leaves = [
        _manifest_leaf_hash(i, entry_hash)
        for i, entry_hash in enumerate(entry_hashes)
    ]
    if not leaves:
        return []
    levels = [leaves]
    current = leaves
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else left
            next_level.append(_manifest_node_hash(left, right))
        levels.append(next_level)
        current = next_level
    return levels


def _manifest_membership_path(
    entry_hashes: list[str],
    leaf_index: int,
) -> list[dict[str, Any]]:
    if leaf_index < 0 or leaf_index >= len(entry_hashes):
        raise IndexError("manifest leaf index out of range")
    levels = _manifest_levels(entry_hashes)
    path: list[dict[str, Any]] = []
    idx = int(leaf_index)
    for level in levels[:-1]:
        if idx % 2 == 0:
            sibling_idx = idx + 1 if idx + 1 < len(level) else idx
            sibling_is_left = False
        else:
            sibling_idx = idx - 1
            sibling_is_left = True
        path.append(
            {
                "sibling": level[sibling_idx].hex(),
                "sibling_is_left": bool(sibling_is_left),
            }
        )
        idx //= 2
    return path


def _manifest_root_from_hashes(entry_hashes: list[str]) -> str:
    levels = _manifest_levels(entry_hashes)
    return levels[-1][0].hex() if levels else ""


def _manifest_levels_from_hash_bytes(entry_hashes: list[bytes]) -> list[list[bytes]]:
    """``_manifest_levels`` over raw 32-byte digests.

    Identical output to the hex form: ``_manifest_leaf_hash`` only differs
    from ``_manifest_leaf_hash_bytes`` by a ``bytes.fromhex`` on its input.
    """

    leaves = [
        _manifest_leaf_hash_bytes(i, entry_hash)
        for i, entry_hash in enumerate(entry_hashes)
    ]
    if not leaves:
        return []
    levels = [leaves]
    current = leaves
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else left
            next_level.append(_manifest_node_hash(left, right))
        levels.append(next_level)
        current = next_level
    return levels


def _manifest_membership_path_from_levels(
    levels: list[list[bytes]],
    leaf_index: int,
) -> list[dict[str, Any]]:
    """Membership path walk over prebuilt levels (no re-hashing)."""

    path: list[dict[str, Any]] = []
    idx = int(leaf_index)
    for level in levels[:-1]:
        if idx % 2 == 0:
            sibling_idx = idx + 1 if idx + 1 < len(level) else idx
            sibling_is_left = False
        else:
            sibling_idx = idx - 1
            sibling_is_left = True
        path.append(
            {
                "sibling": level[sibling_idx].hex(),
                "sibling_is_left": bool(sibling_is_left),
            }
        )
        idx //= 2
    return path


def _manifest_membership_path_from_hash_bytes(
    entry_hashes: list[bytes],
    leaf_index: int,
) -> list[dict[str, Any]]:
    """``_manifest_membership_path`` over raw 32-byte digests."""

    if leaf_index < 0 or leaf_index >= len(entry_hashes):
        raise IndexError("manifest leaf index out of range")
    return _manifest_membership_path_from_levels(
        _manifest_levels_from_hash_bytes(entry_hashes), leaf_index
    )


def _manifest_root_from_hash_bytes(entry_hashes: list[bytes]) -> str:
    current = [
        _manifest_leaf_hash_bytes(i, entry_hash)
        for i, entry_hash in enumerate(entry_hashes)
    ]
    if not current:
        return ""
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else left
            next_level.append(_manifest_node_hash(left, right))
        current = next_level
    return current[0].hex()


def _shape_tuple(raw: Any) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw)
    if not values:
        raise ValueError("shape must be non-empty")
    return values


def verify_trace_membership(
    *,
    trace_commitment_hash: str,
    trace_set_root: str,
    trace_set_count: int,
    leaf_index: int,
    path: list[Mapping[str, Any]],
) -> bool:
    """Verify a trace leaf membership path inside a committed trace set."""

    if trace_set_count <= 0:
        return False
    if leaf_index < 0 or leaf_index >= trace_set_count:
        return False
    if len(trace_set_root) != 64:
        return False
    try:
        node = _trace_leaf_hash(int(leaf_index), trace_commitment_hash)
        for item in path:
            sibling = bytes.fromhex(str(item["sibling"]))
            if bool(item.get("sibling_is_left", False)):
                node = _trace_node_hash(sibling, node)
            else:
                node = _trace_node_hash(node, sibling)
    except Exception:
        return False
    return node.hex() == trace_set_root


def verify_op_manifest_membership(
    *,
    entry_hash: str,
    op_manifest_root: str,
    op_manifest_count: int,
    leaf_index: int,
    path: list[Mapping[str, Any]],
) -> bool:
    """Verify an op-manifest leaf membership path."""

    if op_manifest_count <= 0:
        return False
    if leaf_index < 0 or leaf_index >= op_manifest_count:
        return False
    if len(op_manifest_root) != 64:
        return False
    try:
        node = _manifest_leaf_hash(int(leaf_index), entry_hash)
        for item in path:
            sibling = bytes.fromhex(str(item["sibling"]))
            if bool(item.get("sibling_is_left", False)):
                node = _manifest_node_hash(sibling, node)
            else:
                node = _manifest_node_hash(node, sibling)
    except Exception:
        return False
    return node.hex() == op_manifest_root


def _verify_payload_op_manifest_membership(
    payload: Mapping[str, Any],
    membership: Mapping[str, Any],
) -> None:
    entry_raw = membership.get("entry")
    if not isinstance(entry_raw, Mapping):
        raise RuntimeError("GGML op manifest membership is missing entry metadata")
    entry = GgmlOpManifestEntry.from_mapping(entry_raw)
    if not entry.proof_eligible or not _is_provable_weight_op(entry.src0_shape):
        raise RuntimeError("GGML op manifest membership is not proof-capable")
    entry_hash = entry.entry_hash()
    if entry_hash != str(membership.get("op_manifest_entry_hash", "")):
        raise RuntimeError("GGML op manifest entry hash mismatch")
    if not verify_op_manifest_membership(
        entry_hash=entry_hash,
        op_manifest_root=str(membership.get("op_manifest_root", "")),
        op_manifest_count=int(membership.get("op_manifest_count", 0)),
        leaf_index=int(membership.get("op_manifest_leaf_index", -1)),
        path=list(membership.get("op_manifest_membership_path", [])),
    ):
        raise RuntimeError("GGML op manifest membership proof failed")
    trace_meta = payload.get("trace", {})
    if not isinstance(trace_meta, Mapping):
        raise RuntimeError("GGML proof payload is missing trace metadata")
    checks = {
        "graph_id": entry.graph_id,
        "op_index": int(entry.op_index),
        "tensor_name": entry.tensor_name,
        "src0_name": entry.src0_name,
        "src1_name": entry.src1_name,
        "dst_name": entry.dst_name,
        "src0_shape": list(entry.src0_shape),
        "src1_shape": list(entry.src1_shape),
        "dst_shape": list(entry.dst_shape),
        "source_types": dict(sorted(entry.source_types.items())),
        "backend": entry.backend,
        "device": entry.device,
        "manifest_index": int(entry.manifest_index),
    }
    for field, expected in checks.items():
        if expected == "" and field not in trace_meta:
            continue
        if field not in trace_meta:
            raise RuntimeError(f"proof payload trace {field} missing")
        actual = trace_meta.get(field)
        if isinstance(expected, list):
            actual_value = [int(item) for item in actual]
        elif isinstance(expected, dict):
            actual_value = {str(k): str(v) for k, v in actual.items()}
        elif isinstance(expected, int):
            actual_value = int(actual)
        else:
            actual_value = str(actual)
        if actual_value != expected:
            raise RuntimeError(
                f"GGML trace commitment field {field} does not match op manifest"
            )


def trace_membership_payload(
    traces: list["GgmlMulMatTrace"],
    selected_index: int,
    *,
    stage_index: int,
) -> dict[str, Any]:
    """Return the trace-set membership opening for a selected trace."""

    ordered = _ordered_traces(traces)
    if not ordered:
        raise RuntimeError("no proof-capable GGML traces found for membership")
    commitments = [item.commitment_hash() for item in ordered]
    return {
        "version": 1,
        "stage_index": int(stage_index),
        "trace_set_root": _trace_set_root_from_commitments(commitments),
        "trace_set_count": len(commitments),
        "trace_leaf_index": int(selected_index),
        "trace_commitment_hash": commitments[int(selected_index)],
        "trace_membership_path": _trace_membership_path(commitments, int(selected_index)),
    }


def op_manifest_membership_payload(
    entries: list["GgmlOpManifestEntry"],
    trace: "GgmlMulMatTrace",
    *,
    stage_index: int,
) -> dict[str, Any]:
    """Return a manifest membership opening for the op represented by ``trace``."""

    ordered = _ordered_manifest_entries(entries)
    if not ordered:
        raise RuntimeError("no GGML op manifest entries found for inference window")
    entry_hashes = [item.entry_hash() for item in ordered]
    selected_index = -1
    selected_entry: GgmlOpManifestEntry | None = None
    for idx, entry in enumerate(ordered):
        if entry.matches_trace(trace):
            selected_index = idx
            selected_entry = entry
            break
    if selected_entry is None:
        raise RuntimeError("selected GGML trace is not present in the op manifest")
    entry_hash = entry_hashes[selected_index]
    return {
        "version": 1,
        "stage_index": int(stage_index),
        "op_manifest_root": _manifest_root_from_hashes(entry_hashes),
        "op_manifest_count": len(entry_hashes),
        "op_manifest_leaf_index": int(selected_index),
        "op_manifest_entry_hash": entry_hash,
        "op_manifest_membership_path": _manifest_membership_path(
            entry_hashes,
            int(selected_index),
        ),
        "entry": selected_entry.to_commitment_body(),
    }


def select_trace_challenge_indexes(
    *,
    beacon: bytes | str,
    trace_set_root: str,
    trace_set_count: int,
    proof_ops_per_request: int,
    stage_index: int = 0,
) -> list[int]:
    """Select trace indexes from a committed trace set using a beacon."""

    count = int(trace_set_count)
    limit = min(max(1, int(proof_ops_per_request)), count)
    if count <= 0:
        return []
    beacon_bytes = bytes.fromhex(beacon) if isinstance(beacon, str) else beacon
    if len(beacon_bytes) != 32:
        raise ValueError("trace challenge beacon must be 32 bytes")
    if len(trace_set_root) != 64:
        raise ValueError("trace_set_root must be a hex SHA-256 digest")
    selected: list[int] = []
    counter = 0
    while len(selected) < limit:
        digest = hashlib.sha256(
            b"VERATHOS_GGML_TRACE_CHALLENGE_INDEX_V1"
            + beacon_bytes
            + bytes.fromhex(trace_set_root)
            + int(stage_index).to_bytes(4, "little", signed=False)
            + counter.to_bytes(4, "little", signed=False)
        ).digest()
        candidate = int.from_bytes(digest[:8], "little") % count
        if candidate not in selected:
            selected.append(candidate)
        counter += 1
    return selected


def select_manifest_challenge_indexes(
    *,
    beacon: bytes | str,
    op_manifest_root: str,
    op_manifest_count: int,
    proof_ops_per_request: int,
    stage_index: int = 0,
) -> list[int]:
    """Select op-manifest indexes from the full committed GGML manifest."""

    count = int(op_manifest_count)
    limit = min(max(1, int(proof_ops_per_request)), count)
    if count <= 0:
        return []
    beacon_bytes = bytes.fromhex(beacon) if isinstance(beacon, str) else beacon
    if len(beacon_bytes) != 32:
        raise ValueError("manifest challenge beacon must be 32 bytes")
    if len(op_manifest_root) != 64:
        raise ValueError("op_manifest_root must be a hex SHA-256 digest")
    selected: list[int] = []
    counter = 0
    while len(selected) < limit:
        digest = hashlib.sha256(
            b"VERATHOS_GGML_OP_MANIFEST_CHALLENGE_INDEX_V1"
            + beacon_bytes
            + bytes.fromhex(op_manifest_root)
            + int(stage_index).to_bytes(4, "little", signed=False)
            + counter.to_bytes(4, "little", signed=False)
        ).digest()
        candidate = int.from_bytes(digest[:8], "little") % count
        if candidate not in selected:
            selected.append(candidate)
        counter += 1
    return selected


def derive_every_request_trace_beacon(gate_hash: str) -> bytes:
    """Derive a deterministic trace-selection beacon for proof-every-request mode."""

    if len(gate_hash) != 64:
        raise ValueError("gate_hash must be a hex SHA-256 digest")
    return hashlib.sha256(
        b"VERATHOS_MESH_EVERY_REQUEST_TRACE_BEACON_V1"
        + bytes.fromhex(gate_hash)
    ).digest()


def derive_standalone_trace_beacon(trace_set_root: str) -> bytes:
    """Derive a trace-selection beacon for standalone proof endpoint tests/tools."""

    if len(trace_set_root) != 64:
        raise ValueError("trace_set_root must be a hex SHA-256 digest")
    return hashlib.sha256(
        b"VERATHOS_GGML_STANDALONE_TRACE_BEACON_V1"
        + bytes.fromhex(trace_set_root)
    ).digest()


def derive_every_request_trace_beacon_v2(
    gate_hash: str,
    validator_nonce: bytes,
) -> bytes:
    """Mix a validator nonce into the proof-every-request beacon.

    The V1 beacon is derived from the gate hash alone, which leaves which-op
    selection fully under miner-controlled input at 10000 bps. When the
    request carries a validator nonce, V2 mixes it in so op selection is no
    longer grindable over commitment malleability. V1 stays in use when no
    nonce is present (backward compatible).
    """

    if len(gate_hash) != 64:
        raise ValueError("gate_hash must be a hex SHA-256 digest")
    if len(validator_nonce) != 32:
        raise ValueError("validator_nonce must be 32 bytes")
    return hashlib.sha256(
        b"VERATHOS_MESH_EVERY_REQUEST_TRACE_BEACON_V2"
        + bytes.fromhex(gate_hash)
        + validator_nonce
    ).digest()


# --- per-request slot views for verified concurrency (--parallel N) ---
#
# With llama.cpp continuous batching, one backend graph mixes token rows from
# multiple requests, so the co-batched op shapes (src1 row counts) and graph
# boundaries are not per-request. Rather than reconstruct ubatch boundaries
# across two clocks and a live shared window (fragile), the slot view is the
# request's DETERMINISTIC solo decode layout:
#
#   - The per-op structure template (tensor name, weight dims, types) is read
#     from the shared v3 manifest's decode-shaped graphs, one sub-graph per
#     device on multi-GPU serves. The op's IDENTITY across graph kinds is its
#     committed weight name: intra ordinals are informational only, because
#     architectures with length-dependent op streams (glm-dsa) place the same
#     op at different intras per graph kind. Arming, replay matching and
#     verification all bind by name plus committed dims/types.
#   - A solo replay of the request runs ceil(prompt/ubatch) prefill graphs and
#     then exactly one graph per decoded token. The slot view commits, per
#     decode step and per template op, a leaf addressing (solo graph ordinal,
#     intra index, weight structure). Decode graphs are clean single-forward
#     passes that reproduce identically under solo replay.
#
# Attribution to the request comes from the committed request/response/token
# hashes plus solo-replay output equality; witnesses are only ever taken from
# the solo replay, so openings never contain another request's activations.
# Prefill-op coverage and per-position decode audits are out of scope under
# --parallel N (decode audits are rejected for batched meshes); the challenged
# op set is the per-token decode-step layer stack plus the LM head.

SLOT_VIEW_SCOPE = "slot_view_v3"
SLOT_VIEW_SELECTED_OP_LIMIT = 64
SLOT_VIEW_SELECTED_OP_TOKEN_BUDGET = 220


def window_capture_file_token(
    trace_dir: str | Path,
    *,
    started_unix_ns: int,
) -> str:
    """Return the digits token of the capture window active at a timestamp.

    Capture windows write one manifest file per window token (the token is
    the window-open time in ns), so the window serving a request is the
    newest manifest file opened at or before the request start.
    """

    root = Path(trace_dir)
    best_token = ""
    best_ts = -1
    bound = int(started_unix_ns)
    for path in root.glob("manifest*.vmanifest"):
        ts = _manifest_file_unix_ns(path)
        if ts <= 0 or ts > bound:
            continue
        if ts > best_ts:
            best_ts = ts
            best_token = str(ts)
    return best_token


def _op_template_from_manifest(
    manifest_entries: list[GgmlOpManifestEntry],
) -> list[dict[str, Any]]:
    """Build the model's per-graph op template.

    Rows are keyed by (backend, device, intra_graph_index): on a multi-GPU
    serve one forward pass is several DEVICE SUB-GRAPHS whose intra
    counters each restart at zero, so intra alone would collapse distinct
    ops from different devices onto one template row. Weight dims (src0)
    and types are committed; batch-dependent src1/dst row counts are
    deliberately excluded. The intra index is informational only: matching
    against live instances is by tensor name, never by intra ordinal
    (architectures with length-dependent op streams shift intras between
    graph kinds).
    """

    template: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry in _ordered_manifest_entries(manifest_entries):
        intra = int(entry.intra_graph_index)
        if intra < 0:
            raise ValueError(
                "slot view requires v3 manifest rows with intra-graph indexes"
            )
        key = (str(entry.backend), str(entry.device), intra)
        if key in template:
            continue
        template[key] = {
            "intra_graph_index": intra,
            "tensor_name": entry.tensor_name,
            "src0_name": entry.src0_name,
            "k_dim": int(entry.src0_shape[0]),
            "n_dim": int(entry.src0_shape[1]),
            "source_types": dict(entry.source_types),
            "backend": entry.backend,
            "device": entry.device,
        }
    return [template[key] for key in sorted(template)]


def _slot_view_op_is_committed_weight(op: Mapping[str, Any]) -> bool:
    """Only GEMMs against a committed model weight are provable leaves.

    Split serving introduces dim-valid activation x activation GEMMs whose
    src0 is a scheduler input copy (named ``<backend>#<tensor>#<n>``) or
    another runtime activation. Those cannot be checked against the signed
    weight manifest, and their instances do not reproduce across graph
    kinds, so a beacon that draws one fails the audit spuriously.
    """

    name = str(op.get("tensor_name", "") or "")
    return name.endswith(".weight") and "#" not in name


def _ordered_slot_template_ops(
    template: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            op
            for op in template
            if int(op.get("k_dim", 0)) > 1
            and int(op.get("n_dim", 0)) > 1
            and _slot_view_op_is_committed_weight(op)
        ),
        key=lambda op: (
            str(op.get("backend", "")),
            str(op.get("device", "")),
            int(op.get("intra_graph_index", -1)),
        ),
    )


# A decode step processes ONE token, but architectures with residual
# stream fan-out (deepseek4 hyper-connections mix 6 streams per token)
# still run small multi-row dense GEMMs each step. Anything at or below
# this bound is per-token structure; prefill chunks and teacher-forced
# audit chunks batch whole token ranges (dozens to hundreds of rows).
_SLOT_VIEW_DECODE_SHAPED_MAX_DENSE_ROWS = 8


def _slot_view_graph_is_decode_shaped(
    entries: list["GgmlOpManifestEntry"],
) -> bool:
    """True when a graph's DENSE GEMMs are decode-sized (a decode step).

    Expert (MUL_MAT_ID) rows are exempt: their src1 batches the routed
    expert slots even for a single token. Prefill graphs carry chunk-row
    dense GEMMs, so this separates the two graph kinds on every
    architecture, including asymmetric ones (deepseek4 CSA).

    MAJORITY rule, not all-rows: some architectures run fixed-width tile
    ops inside every decode step (glm-dsa's sparse-attention indexer
    scores a 32-token window per decoded token), so a decode graph
    legitimately mixes single-row GEMMs with tile-width ones. Requiring
    every dense row to be decode-sized classified every such decode graph
    as prefill, the template fell back to a REAL prefill graph, and every
    slot-view/audit draw failed with intra ordinals that matched no live
    decode instance. A prefill graph stays cleanly on the other side of
    the rule: ALL its layer GEMMs are chunk-width, so decode-sized rows
    never reach half.
    """

    if not entries:
        return False
    small = 0
    large = 0
    for item in entries:
        src0 = tuple(item.src0_shape) + (1, 1, 1, 1)
        if src0[2] > 1:
            continue
        # The final-logit row is single-column in BOTH graph kinds (prefill
        # computes logits only for its last token), so it carries no signal
        # and must not tip the tally on small graphs.
        if _slot_view_op_is_decode_candidate(item.to_commitment_body()):
            continue
        src1 = tuple(item.src1_shape) + (1, 1, 1, 1)
        if src1[1] > _SLOT_VIEW_DECODE_SHAPED_MAX_DENSE_ROWS:
            large += 1
        else:
            small += 1
    if small == 0:
        # A decode step has at least one decode-sized layer GEMM; a prefill
        # graph has none (all its layer GEMMs are chunk-width).
        return False
    return small >= large


def _slot_view_template_rows_for_decode_graph(
    entries: list["GgmlOpManifestEntry"],
) -> list["GgmlOpManifestEntry"]:
    """Drop ops that are never single-row at decode from a template graph.

    In a decode-shaped graph, a dense GEMM still carrying a tile-width
    src1 (the glm-dsa indexer window) has no single-row instance for the
    selection to open; leaving it in the template lets the beacon draw an
    unprovable leaf and fail an honest node's audit.
    """

    kept: list[GgmlOpManifestEntry] = []
    for item in entries:
        src0 = tuple(item.src0_shape) + (1, 1, 1, 1)
        src1 = tuple(item.src1_shape) + (1, 1, 1, 1)
        if (
            src0[2] <= 1
            and src1[1] > _SLOT_VIEW_DECODE_SHAPED_MAX_DENSE_ROWS
        ):
            continue
        kept.append(item)
    return kept


def _slot_view_op_is_decode_candidate(op: Mapping[str, Any]) -> bool:
    tensor_name = str(op.get("tensor_name", "")).lower()
    src0_name = str(op.get("src0_name", "")).lower()
    projection_names = {tensor_name, src0_name}
    return (
        "output.weight" in projection_names
        or "lm_head.weight" in projection_names
        or any(name.endswith(".output.weight") for name in projection_names)
        or any(name.endswith(".lm_head.weight") for name in projection_names)
    )


def _slot_view_decode_template_offset(
    ordered_template: list[Mapping[str, Any]],
) -> int:
    candidates = [
        (idx, int(op.get("intra_graph_index", -1)))
        for idx, op in enumerate(ordered_template)
        if _slot_view_op_is_decode_candidate(op)
    ]
    if not candidates:
        raise RuntimeError("decode audit requires a final-logit slot-view op")
    return max(candidates, key=lambda item: item[1])[0]


def _slot_view_decode_leaf_indexes_from_template(
    *,
    ordered_template: list[Mapping[str, Any]],
    completion_token_count: int,
    positions: list[int],
) -> dict[int, list[int]]:
    completion = int(completion_token_count)
    if completion <= 0:
        raise RuntimeError("decode audit requires completion tokens")
    template_count = len(ordered_template)
    if template_count <= 0:
        raise RuntimeError("decode audit requires a slot-view template")
    decode_offset = _slot_view_decode_template_offset(ordered_template)
    selected: dict[int, list[int]] = {}
    for position in sorted({int(item) for item in positions}):
        if position < 0 or position >= completion:
            raise RuntimeError("decode audit position out of completion range")
        leaf_index = position * template_count + decode_offset
        selected.setdefault(leaf_index, []).append(position)
    return selected


def _slot_view_decode_leaf_indexes(
    *,
    leaves: list["SlotViewLeaf"],
    completion_token_count: int,
    positions: list[int],
) -> dict[int, list[int]]:
    completion = int(completion_token_count)
    if completion <= 0:
        raise RuntimeError("decode audit requires completion tokens")
    selected: dict[int, list[int]] = {}
    for position in sorted({int(item) for item in positions}):
        if position < 0 or position >= completion:
            raise RuntimeError("decode audit position out of completion range")
        candidates = [
            (idx, leaf)
            for idx, leaf in enumerate(leaves)
            if int(leaf.token_index) == position
            and _slot_view_op_is_decode_candidate(leaf.to_commitment_body())
        ]
        if not candidates:
            raise RuntimeError("decode audit requires a final-logit slot-view op")
        leaf_index, _leaf = max(
            candidates,
            key=lambda item: int(item[1].intra_graph_index),
        )
        selected.setdefault(int(leaf_index), []).append(position)
    return selected


def _slot_view_graph_ord_candidates(
    *,
    expected_ords: list[int],
    max_graph_ord: int,
    budget: int,
) -> list[int]:
    if not expected_ords or budget <= 0:
        return []
    upper = max(1, int(max_graph_ord), max(int(item) for item in expected_ords))
    selected: list[int] = []
    seen: set[int] = set()

    def add(candidate: int) -> bool:
        value = int(candidate)
        if value < 1 or value > upper or value in seen:
            return False
        seen.add(value)
        selected.append(value)
        return True

    for expected in sorted({max(1, int(item)) for item in expected_ords}):
        if len(selected) >= budget:
            break
        add(min(expected, upper))
    radius = 1
    while len(selected) < budget and radius <= upper + budget:
        progressed = False
        for expected in sorted({max(1, int(item)) for item in expected_ords}):
            for candidate in (expected - radius, expected + radius):
                if len(selected) >= budget:
                    break
                progressed = add(candidate) or progressed
            if len(selected) >= budget:
                break
        if not progressed and radius > upper:
            break
        radius += 1
    return selected


def _slot_view_selected_ops(
    selected: list[Mapping[str, Any]],
    *,
    prompt_token_count: int,
    completion_token_count: int,
    n_ubatch: int,
    limit: int = SLOT_VIEW_SELECTED_OP_LIMIT,
    token_budget: int = SLOT_VIEW_SELECTED_OP_TOKEN_BUDGET,
) -> list[str]:
    if not selected:
        return []
    prompt = int(prompt_token_count)
    completion = max(1, int(completion_token_count) or 1)
    ubatch = int(n_ubatch)
    prefill_graphs = (
        solo_prefill_graph_count(prompt_token_count=prompt, n_ubatch=ubatch)
        if prompt > 0 and ubatch > 0
        else 1
    )
    # v3 leaves carry no graph ordinal; the committed solo layout's ordinal
    # for a leaf is still pure arithmetic over its token index, which is all
    # the C-side dump filter arming needs.
    grouped: dict[int, set[int]] = {}
    for item in selected:
        leaf_raw = item.get("leaf", {})
        if not isinstance(leaf_raw, Mapping):
            continue
        leaf = SlotViewLeaf.from_mapping(leaf_raw)
        grouped.setdefault(int(leaf.intra_graph_index), set()).add(
            prefill_graphs + int(leaf.token_index) + 1
        )
    if not grouped:
        return []
    # A small upper slack covers llama.cpp warmup/control graphs while keeping
    # the C-side selected-op parser's fixed 64-entry budget intact.
    max_graph_ord = prefill_graphs + completion + 8
    per_group = max(1, int(limit) // max(1, len(grouped)))
    # Per-op candidate ordinals, committed layout first. The teacher-forced
    # audit probes run at the START of the exclusive window (prefill chunks
    # plus one decode graph), so the low ordinals must be candidates too or
    # the C-side dump filter skips exactly the instances the audit needs.
    per_intra: dict[int, list[int]] = {}
    for intra in sorted(grouped):
        committed_candidates = _slot_view_graph_ord_candidates(
            expected_ords=sorted(grouped[intra]),
            max_graph_ord=max_graph_ord,
            budget=per_group,
        )
        probe_candidates = [
            ord_value
            for ord_value in range(1, 9)
            if ord_value not in set(committed_candidates)
        ]
        per_intra[intra] = [*committed_candidates, *probe_candidates]
    # Name-armed entries (n:<weight name>): the C side dumps ANY small-row
    # instance whose src0 weight matches the name, regardless of graph or
    # intra ordinal. This is the entry that survives architectures whose op
    # stream is length-dependent (glm-dsa's sparse-attention indexer path
    # switches with KV state, shifting every intra between graph kinds), so
    # the ordinal-based entries below can never be relied on alone. Names
    # are emitted LAST: older builds' parsers stop at the first
    # non-numeric entry, so appending keeps their behavior byte-identical.
    names: list[str] = []
    for item in selected:
        leaf_raw = item.get("leaf", {})
        if not isinstance(leaf_raw, Mapping):
            continue
        name = str(leaf_raw.get("tensor_name", "") or "")
        if name and name not in names:
            names.append(name)
    name_ops = [f"n:{name}" for name in names[: max(1, int(limit) // 4)]]
    name_chars = sum(len(op) + 1 for op in name_ops)
    ordinal_limit = max(1, int(limit) - len(name_ops))
    ordinal_budget = max(1, int(token_budget) - name_chars)

    selected_ops: list[str] = []
    selected_ops_chars = 0
    # Ord-agnostic wildcard entries come FIRST: real replay graph ordinals
    # drift from the committed layout whenever the scheduler splits a pass
    # into extra subgraphs (rpc view-boundary splits, deepseek4
    # hyper-connection islands), so exact ord:intra pairs alone miss the
    # decode instances the audit needs. The C side dumps wildcard matches
    # only for small-row instances, so this stays bounded.
    for intra in sorted(per_intra):
        op = f"*:{intra}"
        extra_chars = len(op) + (1 if selected_ops else 0)
        if selected_ops and selected_ops_chars + extra_chars > ordinal_budget:
            break
        selected_ops.append(op)
        selected_ops_chars += extra_chars
    # Round-robin across ops so the char budget truncates the TAIL of every
    # op's candidate list rather than starving whichever op sorts last —
    # sequential emission dropped the LM head's ordinals entirely and the
    # decode audit could never find its instance (observed).
    for rank in range(max(len(items) for items in per_intra.values())):
        if len(selected_ops) >= ordinal_limit:
            break
        for intra in sorted(per_intra):
            candidates = per_intra[intra]
            if rank >= len(candidates):
                continue
            op = f"{candidates[rank]}:{intra}"
            extra_chars = len(op) + (1 if selected_ops else 0)
            if selected_ops and selected_ops_chars + extra_chars > ordinal_budget:
                return [*selected_ops, *name_ops]
            selected_ops.append(op)
            selected_ops_chars += extra_chars
            if len(selected_ops) >= ordinal_limit:
                break
    return [*selected_ops, *name_ops]


def _slot_view_leaf_body_from_template_op(
    *,
    op: Mapping[str, Any],
    token_index: int,
) -> dict[str, Any]:
    # v3 leaves carry no graph ordinal: the replay layout is runtime-dependent
    # (warmup graphs, cache-bounded probe windows) and the verifier never
    # checked it. Leaving it out makes the leaf set - and every cache built
    # over it - independent of the prompt, so sessions reuse leaf hashes,
    # levels and roots across turns.
    return {
        "version": 3,
        "token_index": int(token_index),
        "intra_graph_index": int(op["intra_graph_index"]),
        "tensor_name": str(op.get("tensor_name", "")),
        "src0_name": str(op.get("src0_name", "")),
        "k_dim": int(op["k_dim"]),
        "n_dim": int(op["n_dim"]),
        "source_types": dict(
            sorted(
                (str(k), str(v))
                for k, v in dict(op.get("source_types", {}) or {}).items()
            )
        ),
        "backend": str(op.get("backend", "")),
        "device": str(op.get("device", "")),
    }


def _slot_view_leaf_hash_from_body(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"VERATHOS_GGML_SLOT_VIEW_LEAF_V3" + canonical_json_bytes(body)
    ).hexdigest()


def _slot_view_leaf_hash_bytes_from_body(body: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGML_SLOT_VIEW_LEAF_V3" + canonical_json_bytes(body)
    ).digest()


def _slot_view_leaf_json_fragments_from_template_op(
    op: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    """Precompute static canonical-JSON fragments for one template op.

    token_index is the only per-leaf dynamic field in a v3 body, so the
    canonical JSON is prefix + str(token_index) + suffix. Must stay
    byte-identical to canonical_json_bytes of the leaf body.
    """

    source_types = dict(
        sorted(
            (str(k), str(v))
            for k, v in dict(op.get("source_types", {}) or {}).items()
        )
    )
    source_json = (
        "{"
        + ",".join(
            f"{_json_string(key)}:{_json_string(value)}"
            for key, value in source_types.items()
        )
        + "}"
    )
    prefix = (
        "{"
        f"\"backend\":{_json_string(str(op.get('backend', '')))},"
        f"\"device\":{_json_string(str(op.get('device', '')))},"
        f"\"intra_graph_index\":{int(op['intra_graph_index'])},"
        f"\"k_dim\":{int(op['k_dim'])},"
        f"\"n_dim\":{int(op['n_dim'])},"
        f"\"source_types\":{source_json},"
        f"\"src0_name\":{_json_string(str(op.get('src0_name', '')))},"
        f"\"tensor_name\":{_json_string(str(op.get('tensor_name', '')))},"
        "\"token_index\":"
    ).encode("utf-8")
    suffix = ",\"version\":3}".encode("utf-8")
    return prefix, suffix


def _slot_view_leaf_from_template_index(
    *,
    template: list[Mapping[str, Any]],
    leaf_index: int,
    completion_token_count: int,
) -> SlotViewLeaf:
    ordered_template = _ordered_slot_template_ops(template)
    if not ordered_template:
        raise ValueError("slot view requires at least one template op")
    count = int(completion_token_count) * len(ordered_template)
    index = int(leaf_index)
    if index < 0 or index >= count:
        raise IndexError("slot view leaf index out of range")
    token_index = index // len(ordered_template)
    op = ordered_template[index % len(ordered_template)]
    body = _slot_view_leaf_body_from_template_op(
        op=op,
        token_index=token_index,
    )
    return SlotViewLeaf.from_mapping(body)


def slot_view_template_from_manifest_entries(
    manifest_entries: list[GgmlOpManifestEntry],
) -> list[dict[str, Any]]:
    """Return the request-independent slot-view op template.

    When the source graph is decode-shaped, tile-width dense ops (never
    single-row at decode, e.g. the glm-dsa indexer window) are excluded:
    they have no openable decode instance, so a beacon draw landing on one
    would fail an honest node. A non-decode-shaped source (the legacy
    prefill fallback on symmetric architectures) is kept whole, where the
    filter would wrongly drop every chunk-width row.
    """

    entries = manifest_entries
    if _slot_view_graph_is_decode_shaped(entries):
        entries = _slot_view_template_rows_for_decode_graph(entries)
    return _op_template_from_manifest(entries)


def slot_view_leaf_hashes_from_template(
    *,
    template: list[Mapping[str, Any]],
    completion_token_count: int,
) -> list[str]:
    """Return slot-view leaf hashes without materializing leaf objects."""

    ordered_template = _ordered_slot_template_ops(template)
    if not ordered_template:
        raise ValueError("no proof-eligible op template rows for the slot view")
    completion = int(completion_token_count)
    if completion <= 0:
        raise ValueError("slot view requires at least one decoded token")
    hashes: list[str] = []
    for token_index in range(completion):
        for op in ordered_template:
            hashes.append(
                _slot_view_leaf_hash_from_body(
                    _slot_view_leaf_body_from_template_op(
                        op=op,
                        token_index=token_index,
                    )
                )
            )
    return hashes


# Serve-path leaf-hash memo. A v3 leaf's hash depends only on the template
# op and its own token index, NEVER on the prompt or on how many tokens the
# request ends up producing, so the hashes for one template form an
# extensible PREFIX shared by every request and every session turn: a
# 512-token request reuses every hash a 256-token request already computed.
# Without this, each serve recomputed completion x template_ops SHA-256
# leaves plus a full Merkle root, which measured as the whole ~59 ms/request
# receipt-path residual at 8-way concurrency (66 ms for 512 tokens x 200 ops
# on a 4090 box).
_SLOT_VIEW_LEAF_CACHE_MAX_BYTES = int(
    os.environ.get("VERATHOS_SLOT_VIEW_LEAF_CACHE_BYTES", str(64 << 20)) or (64 << 20)
)
_SLOT_VIEW_ROOT_CACHE_MAX_ENTRIES = 512
# Full Merkle LEVELS for recent request shapes. One proof needs the tree
# twice (selection derives the root, the membership opening walks a path);
# without this each pass rebuilt every level, which at 4096 decoded tokens
# is ~1.6M SHA256 calls per pass. Bounded in bytes: levels total ~2N x 32B.
_SLOT_VIEW_LEVELS_CACHE_MAX_BYTES = int(
    os.environ.get("VERATHOS_SLOT_VIEW_LEVELS_CACHE_BYTES", str(256 << 20))
    or (256 << 20)
)
_slot_view_memo_lock = threading.Lock()
_slot_view_leaf_cache: "OrderedDict[bytes, list[bytes]]" = OrderedDict()
_slot_view_root_cache: "OrderedDict[tuple[bytes, int], tuple[str, int]]" = (
    OrderedDict()
)
_slot_view_levels_cache: "OrderedDict[bytes, dict[str, Any]]" = OrderedDict()


def _slot_view_levels_cached(leaf_hashes: list[bytes]) -> list[list[bytes]]:
    """Merkle levels for a leaf set, built incrementally per leaf prefix.

    v3 leaf sets for one template form an extensible prefix, and in the
    odd-duplicate tree every node covering a COMPLETE leaf block
    [i*2^L, (i+1)*2^L) is identical for every set length that contains
    it; only the ragged right edge (at most one node per level) depends
    on the exact length. Caching the complete nodes per prefix makes a
    request cost O(new leaves + log n) instead of a full O(n) rebuild
    for every previously unseen completion length, which measured 2-4 s
    per novel reply length at long decodes.
    """

    if not leaf_hashes:
        return []
    n = len(leaf_hashes)
    # Prefix identity: the first leaf pins the template universe; the
    # element at the shared boundary is re-checked below before reuse.
    key = leaf_hashes[0]
    with _slot_view_memo_lock:
        entry = _slot_view_levels_cache.get(key)
        if entry is not None:
            _slot_view_levels_cache.move_to_end(key)
        else:
            entry = {"bound": [], "levels": []}
            _slot_view_levels_cache[key] = entry
        bound: list[bytes] = entry["bound"]
        complete: list[list[bytes]] = entry["levels"]
        # The first leaf does not uniquely pin the universe (two templates
        # sharing their first op collide), so verify the last SHARED leaf in
        # the position-bound domain before reusing or extending; on any
        # disagreement rebuild this entry from the input.
        shared = min(len(bound), n)
        if shared > 0 and bound[shared - 1] != _manifest_leaf_hash_bytes(
            shared - 1, leaf_hashes[shared - 1]
        ):
            entry = {"bound": [], "levels": []}
            _slot_view_levels_cache[key] = entry
            bound, complete = entry["bound"], entry["levels"]
        # Extend the position-bound leaf array to n.
        for i in range(len(bound), n):
            bound.append(_manifest_leaf_hash_bytes(i, leaf_hashes[i]))
        # Extend each level's complete nodes, then assemble the ragged view.
        # A node is cacheable only when it covers a fully populated leaf
        # block ((i+1)*2^level <= n): a node with a ragged/duplicated child
        # is length-specific and caching it would poison larger lengths.
        levels: list[list[bytes]] = [bound[:n]]
        current = levels[0]
        level_index = 0
        while len(current) > 1:
            if level_index >= len(complete):
                complete.append([])
            nodes = complete[level_index]
            cacheable = n >> (level_index + 1)
            for idx in range(len(nodes), cacheable):
                nodes.append(
                    _manifest_node_hash(current[2 * idx], current[2 * idx + 1])
                )
            next_level = nodes[:cacheable]
            tail = current[2 * cacheable :]
            if len(tail) == 2:
                next_level = next_level + [_manifest_node_hash(tail[0], tail[1])]
            elif len(tail) == 1:
                next_level = next_level + [_manifest_node_hash(tail[0], tail[0])]
            levels.append(next_level)
            current = next_level
            level_index += 1
        running = sum(
            (len(item["bound"]) + sum(len(lv) for lv in item["levels"])) * 32
            for item in _slot_view_levels_cache.values()
        )
        while (
            running > _SLOT_VIEW_LEVELS_CACHE_MAX_BYTES
            and len(_slot_view_levels_cache) > 1
        ):
            _dropped_key, dropped = _slot_view_levels_cache.popitem(last=False)
            running -= (
                len(dropped["bound"]) + sum(len(lv) for lv in dropped["levels"])
            ) * 32
        return levels


def _slot_view_template_digest(fragments: list[tuple[bytes, bytes]]) -> bytes:
    """Identity of an ordered template, for memo keys."""

    digest = hashlib.sha256(b"VERATHOS_SLOT_VIEW_TEMPLATE_ID_V1")
    for prefix, suffix in fragments:
        digest.update(struct.pack("<II", len(prefix), len(suffix)))
        digest.update(prefix)
        digest.update(suffix)
    return digest.digest()


def _slot_view_leaf_hashes_uncached(
    *,
    fragments: list[tuple[bytes, bytes]],
    completion: int,
    start_token: int = 0,
) -> list[bytes]:
    """Leaf hashes for [start_token, completion), no cache involvement."""

    domain = b"VERATHOS_GGML_SLOT_VIEW_LEAF_V3"
    hashes: list[bytes] = []
    for token_index in range(int(start_token), int(completion)):
        token_bytes = str(token_index).encode("ascii")
        for prefix, suffix in fragments:
            hashes.append(
                hashlib.sha256(domain + prefix + token_bytes + suffix).digest()
            )
    return hashes


def _slot_view_leaf_cache_evict_locked() -> None:
    total = sum(len(item) for item in _slot_view_leaf_cache.values()) * 32
    while total > _SLOT_VIEW_LEAF_CACHE_MAX_BYTES and len(_slot_view_leaf_cache) > 1:
        _key, dropped = _slot_view_leaf_cache.popitem(last=False)
        total -= len(dropped) * 32


def slot_view_leaf_hash_bytes_from_template(
    *,
    template: list[Mapping[str, Any]],
    completion_token_count: int,
) -> list[bytes]:
    """Return slot-view leaf hash bytes for root-only hot paths."""

    ordered_template = _ordered_slot_template_ops(template)
    if not ordered_template:
        raise ValueError("no proof-eligible op template rows for the slot view")
    completion = int(completion_token_count)
    if completion <= 0:
        raise ValueError("slot view requires at least one decoded token")
    fragments = [
        _slot_view_leaf_json_fragments_from_template_op(op)
        for op in ordered_template
    ]
    ops_per_token = len(fragments)
    needed = completion * ops_per_token
    cache_key = _slot_view_template_digest(fragments)
    with _slot_view_memo_lock:
        cached = _slot_view_leaf_cache.get(cache_key)
        if cached is not None and len(cached) >= needed:
            _slot_view_leaf_cache.move_to_end(cache_key)
            return cached[:needed]
        start_token = len(cached) // ops_per_token if cached else 0
    fresh = _slot_view_leaf_hashes_uncached(
        fragments=fragments,
        completion=completion,
        start_token=start_token,
    )
    with _slot_view_memo_lock:
        current = _slot_view_leaf_cache.get(cache_key)
        if current is not None and len(current) >= needed:
            # A concurrent extender already covered this request.
            _slot_view_leaf_cache.move_to_end(cache_key)
            return current[:needed]
        if current is None and start_token == 0:
            # Cold key: `fresh` already covers [0, completion), so it IS the
            # full set. A template's key goes cold once per process, so this
            # is the first-request path per mesh, not the common case.
            hashes = fresh
            _slot_view_leaf_cache[cache_key] = hashes
        elif current is not None and len(current) == start_token * ops_per_token:
            current.extend(fresh)
            hashes = current
        else:
            # Raced against an extender that landed a different prefix
            # length (or an eviction). `fresh` only covers [start_token,
            # completion), so it cannot be joined onto a prefix of unknown
            # length; recompute this key from zero. Rare and always correct.
            hashes = _slot_view_leaf_hashes_uncached(
                fragments=fragments,
                completion=completion,
            )
            _slot_view_leaf_cache[cache_key] = hashes
        _slot_view_leaf_cache.move_to_end(cache_key)
        _slot_view_leaf_cache_evict_locked()
        return hashes[:needed]


def ggml_slot_view_root_from_template(
    *,
    template: list[Mapping[str, Any]],
    completion_token_count: int,
) -> tuple[str, int]:
    """Return the slot-view Merkle root/count from a cached op template."""

    ordered_template = _ordered_slot_template_ops(template)
    if not ordered_template:
        raise ValueError("no proof-eligible op template rows for the slot view")
    fragments = [
        _slot_view_leaf_json_fragments_from_template_op(op)
        for op in ordered_template
    ]
    completion = int(completion_token_count)
    if completion > 0:
        # Same-length repeat (the common case under batching: same model,
        # same max_tokens) skips the Merkle pass too.
        root_key = (_slot_view_template_digest(fragments), completion)
        with _slot_view_memo_lock:
            cached_root = _slot_view_root_cache.get(root_key)
            if cached_root is not None:
                _slot_view_root_cache.move_to_end(root_key)
                return cached_root
        leaf_hashes = slot_view_leaf_hash_bytes_from_template(
            template=template,
            completion_token_count=completion_token_count,
        )
        levels = _slot_view_levels_cached(leaf_hashes)
        result = (levels[-1][0].hex(), len(leaf_hashes))
        with _slot_view_memo_lock:
            _slot_view_root_cache[root_key] = result
            _slot_view_root_cache.move_to_end(root_key)
            while len(_slot_view_root_cache) > _SLOT_VIEW_ROOT_CACHE_MAX_ENTRIES:
                _slot_view_root_cache.popitem(last=False)
        return result
    leaf_hashes = slot_view_leaf_hash_bytes_from_template(
        template=template,
        completion_token_count=completion_token_count,
    )
    levels = _slot_view_levels_cached(leaf_hashes)
    return levels[-1][0].hex(), len(leaf_hashes)


def slot_view_membership_payload_from_template(
    *,
    template: list[Mapping[str, Any]],
    leaf_index: int,
    completion_token_count: int,
    stage_index: int,
) -> dict[str, Any]:
    """Return a slot-view membership opening using a cached op template."""

    # Memoized (extensible-prefix) leaf hashes, not the plain hex loop: the
    # selection payload has almost always just built this exact leaf set, so
    # the second full pass is pure waste. Measured on a 201-op template at
    # 4096 decoded tokens (823k leaves), selection+membership cost 21.6 s
    # against 8.2 s for one leaf pass. Roots and paths are unchanged; the
    # bytes and hex leaf hashers differ only by a bytes.fromhex.
    leaf_hashes = slot_view_leaf_hash_bytes_from_template(
        template=template,
        completion_token_count=completion_token_count,
    )
    index = int(leaf_index)
    if index < 0 or index >= len(leaf_hashes):
        raise IndexError("slot view leaf index out of range")
    levels = _slot_view_levels_cached(leaf_hashes)
    leaf = _slot_view_leaf_from_template_index(
        template=template,
        leaf_index=index,
        completion_token_count=completion_token_count,
    )
    return {
        "version": 3,
        "scope": SLOT_VIEW_SCOPE,
        "stage_index": int(stage_index),
        "slot_view_root": levels[-1][0].hex() if levels else "",
        "slot_view_count": len(leaf_hashes),
        "slot_view_leaf_index": index,
        "slot_view_leaf_hash": leaf_hashes[index].hex(),
        "slot_view_membership_path": _manifest_membership_path_from_levels(
            levels,
            index,
        ),
        "leaf": leaf.to_commitment_body(),
    }


def ggml_slot_view_selection_payload_from_template(
    *,
    template: list[Mapping[str, Any]],
    receipt_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Select challenged slot-view leaves without materializing all leaves."""

    ctx = receipt_context
    stage_index = int(ctx.get("stage_index", 0))
    proof_limit = max(1, int(ctx.get("proof_ops_per_request") or 1))
    prompt = int(ctx.get("prompt_token_count", 0) or 0)
    completion = int(ctx.get("completion_token_count", 0) or 0)
    n_ubatch = int(ctx.get("proof_runtime_ubatch_size", 0) or 0)
    # Memoized leaf hashes (see slot_view_membership_payload_from_template):
    # this set is O(decoded tokens x template ops) and the membership opening
    # that follows needs the very same one.
    leaf_hashes = slot_view_leaf_hash_bytes_from_template(
        template=template,
        completion_token_count=completion,
    )
    if not leaf_hashes:
        raise ValueError("slot view selection needs at least one leaf")
    beacon = str(ctx.get("proof_beacon", ""))
    if not beacon:
        raise ValueError("slot view selection needs a proof beacon")
    levels = _slot_view_levels_cached(leaf_hashes)
    view_root = levels[-1][0].hex() if levels else ""
    decode_required = bool(ctx.get("decode_audit_required", False))
    decode_stage_index = int(ctx.get("decode_audit_stage_index", stage_index))
    decode_active = bool(decode_required and stage_index == decode_stage_index)
    # ``proof_sampled`` and ``decode_audit_required`` are independent gates.
    # Older decode-only callers omitted ``proof_sampled`` entirely, so retain
    # that behavior while making an explicit true value additive.
    base_sampled = bool(ctx.get("proof_sampled", not decode_required))
    selected_indexes: list[int] = []
    decode_positions_by_leaf: dict[int, list[int]] = {}
    if base_sampled:
        selected_indexes = select_slot_view_challenges(
            beacon=beacon,
            slot_view_root=view_root,
            slot_view_count=len(leaf_hashes),
            proof_ops_per_request=proof_limit,
            stage_index=stage_index,
        )
    if decode_active:
        decode_positions = [
            int(item) for item in ctx.get("decode_audit_positions", []) or []
        ]
        if not decode_positions:
            raise RuntimeError("decode audit is required but no positions were selected")
        ordered_template = _ordered_slot_template_ops(list(template))
        try:
            decode_positions_by_leaf = _slot_view_decode_leaf_indexes_from_template(
                ordered_template=ordered_template,
                completion_token_count=completion,
                positions=decode_positions,
            )
            selected_indexes = sorted(
                set(selected_indexes) | set(decode_positions_by_leaf)
            )
        except RuntimeError as exc:
            if "final-logit slot-view op" in str(exc):
                raise RuntimeError(
                    "decode owner stage has no final-logit slot-view op"
                ) from exc
            raise

    selected_by_leaf: dict[int, dict[str, Any]] = {}
    for leaf_index in selected_indexes:
        leaf = _slot_view_leaf_from_template_index(
            template=template,
            leaf_index=int(leaf_index),
            completion_token_count=completion,
        )
        selected_by_leaf[int(leaf_index)] = {
            "leaf_index": int(leaf_index),
            "leaf": leaf.to_commitment_body(),
            "replay_intra_index": int(leaf.intra_graph_index),
            "replay_token_index": int(leaf.token_index),
            "decode_audit_positions": [
                int(item) for item in decode_positions_by_leaf.get(int(leaf_index), [])
            ],
        }
    selected = [selected_by_leaf[index] for index in sorted(selected_by_leaf)]
    selected_ops = _slot_view_selected_ops(
        selected,
        prompt_token_count=prompt,
        completion_token_count=completion,
        n_ubatch=n_ubatch,
    )
    return {
        "version": 3,
        "scope": SLOT_VIEW_SCOPE,
        "stage_index": stage_index,
        "slot_view_root": view_root,
        "slot_view_count": len(leaf_hashes),
        "selected": selected,
        "selected_ops": selected_ops,
        "selected_manifest_indexes": [int(item["leaf_index"]) for item in selected],
        "missing_manifest_indexes": [int(item["leaf_index"]) for item in selected],
        "op_manifest_root": view_root,
        "op_manifest_count": len(leaf_hashes),
    }


@dataclass(frozen=True)
class SlotViewLeaf:
    """One committed decode-step op in a request's deterministic solo layout."""

    token_index: int
    intra_graph_index: int
    tensor_name: str
    src0_name: str
    k_dim: int
    n_dim: int
    source_types: dict[str, str]
    backend: str
    device: str

    def to_commitment_body(self) -> dict[str, Any]:
        return {
            "version": 3,
            "token_index": int(self.token_index),
            "intra_graph_index": int(self.intra_graph_index),
            "tensor_name": self.tensor_name,
            "src0_name": self.src0_name,
            "k_dim": int(self.k_dim),
            "n_dim": int(self.n_dim),
            "source_types": dict(sorted(self.source_types.items())),
            "backend": self.backend,
            "device": self.device,
        }

    def leaf_hash(self) -> str:
        return hashlib.sha256(
            b"VERATHOS_GGML_SLOT_VIEW_LEAF_V3"
            + canonical_json_bytes(self.to_commitment_body())
        ).hexdigest()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SlotViewLeaf":
        if int(data.get("version", 0)) != 3:
            raise ValueError("unsupported slot view leaf version")
        return cls(
            token_index=int(data["token_index"]),
            intra_graph_index=int(data["intra_graph_index"]),
            tensor_name=str(data.get("tensor_name", "")),
            src0_name=str(data.get("src0_name", "")),
            k_dim=int(data["k_dim"]),
            n_dim=int(data["n_dim"]),
            source_types={
                str(k): str(v) for k, v in data.get("source_types", {}).items()
            },
            backend=str(data.get("backend", "")),
            device=str(data.get("device", "")),
        )


def _ordered_slot_view_leaves(leaves: list[SlotViewLeaf]) -> list[SlotViewLeaf]:
    """Return the canonical proof-capable slot-view challenge population."""

    return sorted(
        (leaf for leaf in leaves if leaf.k_dim > 1 and leaf.n_dim > 1),
        key=lambda leaf: (
            int(leaf.token_index),
            str(leaf.backend),
            str(leaf.device),
            int(leaf.intra_graph_index),
        ),
    )


def solo_prefill_graph_count(*, prompt_token_count: int, n_ubatch: int) -> int:
    """Number of prefill graphs a solo replay runs before decoding."""

    prompt = int(prompt_token_count)
    ubatch = int(n_ubatch)
    if prompt <= 0:
        raise ValueError("prompt_token_count must be positive")
    if ubatch <= 0:
        raise ValueError("n_ubatch must be positive")
    return (prompt + ubatch - 1) // ubatch


def build_slot_view_leaves(
    *,
    manifest_entries: list[GgmlOpManifestEntry],
    completion_token_count: int,
) -> list[SlotViewLeaf]:
    """Commit the request's deterministic decode-step op layout.

    One leaf per (decoded token, template op); v3 leaves are independent of
    the prompt and of the runtime graph layout.
    """

    template = _op_template_from_manifest(manifest_entries)
    if not template:
        raise ValueError("no proof-eligible op template rows for the slot view")
    completion = int(completion_token_count)
    if completion <= 0:
        raise ValueError("slot view requires at least one decoded token")
    leaves: list[SlotViewLeaf] = []
    for token_index in range(completion):
        for op in template:
            leaves.append(
                SlotViewLeaf(
                    token_index=int(token_index),
                    intra_graph_index=int(op["intra_graph_index"]),
                    tensor_name=str(op["tensor_name"]),
                    src0_name=str(op["src0_name"]),
                    k_dim=int(op["k_dim"]),
                    n_dim=int(op["n_dim"]),
                    source_types=dict(op["source_types"]),
                    backend=str(op["backend"]),
                    device=str(op["device"]),
                )
            )
    leaves.sort(
        key=lambda leaf: (
            int(leaf.token_index),
            str(leaf.backend),
            str(leaf.device),
            int(leaf.intra_graph_index),
        )
    )
    return leaves


def ggml_slot_view_root(leaves: list[SlotViewLeaf]) -> str:
    """Merkle root over slot view leaves (shared manifest tree shape)."""

    ordered = _ordered_slot_view_leaves(leaves)
    return _manifest_root_from_hashes([leaf.leaf_hash() for leaf in ordered])


def slot_view_membership_payload(
    leaves: list[SlotViewLeaf],
    leaf_index: int,
    *,
    stage_index: int,
) -> dict[str, Any]:
    """Return the slot-view membership opening for a selected leaf."""

    ordered = _ordered_slot_view_leaves(leaves)
    if leaf_index < 0 or leaf_index >= len(ordered):
        raise IndexError("slot view leaf index out of range")
    leaf_hashes = [leaf.leaf_hash() for leaf in ordered]
    return {
        "version": 3,
        "scope": SLOT_VIEW_SCOPE,
        "stage_index": int(stage_index),
        "slot_view_root": _manifest_root_from_hashes(leaf_hashes),
        "slot_view_count": len(leaf_hashes),
        "slot_view_leaf_index": int(leaf_index),
        "slot_view_leaf_hash": leaf_hashes[int(leaf_index)],
        "slot_view_membership_path": _manifest_membership_path(
            leaf_hashes,
            int(leaf_index),
        ),
        "leaf": ordered[int(leaf_index)].to_commitment_body(),
    }


def verify_slot_view_membership(
    *,
    leaf_hash: str,
    slot_view_root: str,
    slot_view_count: int,
    leaf_index: int,
    path: list[Mapping[str, Any]],
) -> bool:
    """Verify a slot-view leaf membership path (manifest tree semantics)."""

    return verify_op_manifest_membership(
        entry_hash=leaf_hash,
        op_manifest_root=slot_view_root,
        op_manifest_count=int(slot_view_count),
        leaf_index=int(leaf_index),
        path=path,
    )


def select_slot_view_challenges(
    *,
    beacon: bytes | str,
    slot_view_root: str,
    slot_view_count: int,
    proof_ops_per_request: int,
    stage_index: int = 0,
) -> list[int]:
    """Select slot-view leaf indexes from the committed view (manifest gate)."""

    return select_manifest_challenge_indexes(
        beacon=beacon,
        op_manifest_root=slot_view_root,
        op_manifest_count=int(slot_view_count),
        proof_ops_per_request=int(proof_ops_per_request),
        stage_index=int(stage_index),
    )


def solo_replay_chunk_for_position(
    *,
    position: int,
    prompt_token_count: int,
    n_ubatch: int,
) -> tuple[int, int, int]:
    """Locate an absolute position inside a solo replay's graph layout.

    Returns (graph_ord, row_in_chunk, chunk_len) with 1-based graph_ord:
    a solo replay prefills the prompt in contiguous n_ubatch chunks and then
    decodes one token per graph.
    """

    pos = int(position)
    prompt = int(prompt_token_count)
    ubatch = int(n_ubatch)
    if pos < 0:
        raise ValueError("position must be non-negative")
    prefill_chunks = solo_prefill_graph_count(
        prompt_token_count=prompt,
        n_ubatch=ubatch,
    )
    if pos < prompt:
        chunk = pos // ubatch
        chunk_start = chunk * ubatch
        chunk_len = min(ubatch, prompt - chunk_start)
        return chunk + 1, pos - chunk_start, chunk_len
    return prefill_chunks + (pos - prompt) + 1, 0, 1


def ggml_slot_view_selection_payload(
    *,
    leaves: list[SlotViewLeaf],
    receipt_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Select challenged slot-view leaves and their solo-replay targets."""

    ctx = receipt_context
    stage_index = int(ctx.get("stage_index", 0))
    proof_limit = max(1, int(ctx.get("proof_ops_per_request") or 1))
    ordered_leaves = _ordered_slot_view_leaves(leaves)
    if not ordered_leaves:
        raise ValueError("slot view selection needs at least one leaf")
    beacon = str(ctx.get("proof_beacon", ""))
    if not beacon:
        raise ValueError("slot view selection needs a proof beacon")
    view_root = ggml_slot_view_root(ordered_leaves)
    completion = max(1, int(ctx.get("completion_token_count", 0)) or 1)
    n_ubatch = int(ctx.get("proof_runtime_ubatch_size", 0) or 0)
    prompt = int(ctx.get("prompt_token_count", 0) or 0)
    decode_required = bool(ctx.get("decode_audit_required", False))
    decode_stage_index = int(ctx.get("decode_audit_stage_index", stage_index))
    decode_active = bool(decode_required and stage_index == decode_stage_index)
    # The base GGML challenge remains selected when a response is also chosen
    # for the more expensive decode audit.
    base_sampled = bool(ctx.get("proof_sampled", not decode_required))
    selected_indexes: list[int] = []
    decode_positions_by_leaf: dict[int, list[int]] = {}
    if base_sampled:
        selected_indexes = select_slot_view_challenges(
            beacon=beacon,
            slot_view_root=view_root,
            slot_view_count=len(ordered_leaves),
            proof_ops_per_request=proof_limit,
            stage_index=stage_index,
        )
    if decode_active:
        decode_positions = [
            int(item) for item in ctx.get("decode_audit_positions", []) or []
        ]
        if not decode_positions:
            raise RuntimeError("decode audit is required but no positions were selected")
        try:
            decode_positions_by_leaf = _slot_view_decode_leaf_indexes(
                leaves=ordered_leaves,
                completion_token_count=completion,
                positions=decode_positions,
            )
            selected_indexes = sorted(
                set(selected_indexes) | set(decode_positions_by_leaf)
            )
        except RuntimeError as exc:
            if "final-logit slot-view op" in str(exc):
                raise RuntimeError(
                    "decode owner stage has no final-logit slot-view op"
                ) from exc
            raise

    selected: list[dict[str, Any]] = []
    for leaf_index in selected_indexes:
        leaf = ordered_leaves[int(leaf_index)]
        selected.append(
            {
                "leaf_index": int(leaf_index),
                "leaf": leaf.to_commitment_body(),
                "replay_intra_index": int(leaf.intra_graph_index),
                "replay_token_index": int(leaf.token_index),
                "decode_audit_positions": [
                    int(item) for item in decode_positions_by_leaf.get(int(leaf_index), [])
                ],
            }
        )
    selected_ops = _slot_view_selected_ops(
        selected,
        prompt_token_count=prompt,
        completion_token_count=completion,
        n_ubatch=n_ubatch,
    )
    return {
        "version": 3,
        "scope": SLOT_VIEW_SCOPE,
        "stage_index": stage_index,
        "slot_view_root": view_root,
        "slot_view_count": len(ordered_leaves),
        "selected": selected,
        "selected_ops": selected_ops,
        # Legacy field names so existing replay plumbing can detect work:
        "selected_manifest_indexes": [int(item["leaf_index"]) for item in selected],
        "missing_manifest_indexes": [int(item["leaf_index"]) for item in selected],
        "op_manifest_root": view_root,
        "op_manifest_count": len(ordered_leaves),
    }


def verify_slot_view_proof_payload(
    payload: Mapping[str, Any],
    *,
    mesh_receipt: Mapping[str, Any],
) -> None:
    """Verify a slot-view membership block and its replay binding.

    Raises RuntimeError on any mismatch. GEMM/weight verification is done by
    the regular payload verifier; this binds the proven op to the audited
    request: leaf membership in the committed per-request view, beacon-driven
    leaf selection, and the deterministic solo-replay location.
    """

    membership = payload.get("slot_view_membership", {})
    if not isinstance(membership, Mapping) or not membership:
        raise RuntimeError("slot view proof payload is missing membership")
    leaf_raw = membership.get("leaf")
    if not isinstance(leaf_raw, Mapping):
        raise RuntimeError("slot view membership is missing leaf metadata")
    leaf = SlotViewLeaf.from_mapping(leaf_raw)
    if leaf.k_dim <= 1 or leaf.n_dim <= 1:
        raise RuntimeError("slot view membership is not proof-capable")
    leaf_hash = leaf.leaf_hash()
    if leaf_hash != str(membership.get("slot_view_leaf_hash", "")):
        raise RuntimeError("slot view leaf hash mismatch")
    view_root = str(membership.get("slot_view_root", ""))
    view_count = int(membership.get("slot_view_count", 0))
    leaf_index = int(membership.get("slot_view_leaf_index", -1))
    if not verify_slot_view_membership(
        leaf_hash=leaf_hash,
        slot_view_root=view_root,
        slot_view_count=view_count,
        leaf_index=leaf_index,
        path=list(membership.get("slot_view_membership_path", [])),
    ):
        raise RuntimeError("slot view membership proof failed")

    stage_index = int(membership.get("stage_index", 0))
    beacon = str(mesh_receipt.get("proof_beacon", ""))
    if not beacon:
        raise RuntimeError("slot view verification needs the receipt proof beacon")
    proof_limit = max(1, int(mesh_receipt.get("proof_ops_per_request") or 1))
    expected_indexes = (
        select_slot_view_challenges(
            beacon=beacon,
            slot_view_root=view_root,
            slot_view_count=view_count,
            proof_ops_per_request=proof_limit,
            stage_index=stage_index,
        )
        if bool(mesh_receipt.get("proof_sampled", True))
        else []
    )
    decode_positions = [
        int(item) for item in mesh_receipt.get("decode_audit_positions", []) or []
    ]
    decode_openings = _decode_audit_openings(payload)
    opening_positions = {
        int(item.get("position", -1)) for item in decode_openings if isinstance(item, Mapping)
    }
    decode_stage_index = int(
        mesh_receipt.get("decode_audit_stage_index", stage_index)
    )
    decode_selected = bool(
        mesh_receipt.get("decode_audit_required", False)
        and stage_index == decode_stage_index
        and leaf.token_index in set(decode_positions)
        and _slot_view_op_is_decode_candidate(leaf.to_commitment_body())
    )
    if decode_openings:
        if not decode_selected:
            raise RuntimeError("slot view decode opening is not on a selected decode leaf")
        if opening_positions != {int(leaf.token_index)}:
            raise RuntimeError("slot view decode opening position mismatch")
    if leaf_index not in expected_indexes and not decode_selected:
        raise RuntimeError("slot view leaf was not selected by the beacon")

    completion_token_count = int(mesh_receipt.get("completion_token_count", 0))
    if leaf.token_index < 0 or leaf.token_index >= completion_token_count:
        raise RuntimeError("slot view leaf token index out of range")

    # Bind the proven op to the request's model and a decode step. Absolute
    # replay graph ordinals are NOT checked (a warmup graph offsets them and
    # the count is runtime-dependent); soundness rests on: the op is the
    # committed layer op (tensor + weight dims, Merkle-bound), it ran on a
    # single-token decode step (one activation row, so no co-batched rows
    # leak), and the solo replay reproduced the committed response/token
    # hashes (checked by the caller).
    trace_meta = payload.get("trace", {})
    if not isinstance(trace_meta, Mapping):
        raise RuntimeError("slot view proof payload is missing trace metadata")
    # The trace's intra ordinal is deliberately NOT compared to the leaf's:
    # both values are prover-supplied, so the equality never added binding,
    # and architectures with length-dependent op streams (glm-dsa) place
    # the same committed op at different intras per graph kind, failing
    # honest nodes. The binding is the WEIGHT NAME (checked below and
    # Merkle-bound against the signed manifest by the payload verifier),
    # the committed dims/types, and the single-row decode shape.
    if str(trace_meta.get("tensor_name", "")) != leaf.tensor_name:
        raise RuntimeError(
            "slot view trace tensor name mismatch: trace "
            f"{str(trace_meta.get('tensor_name', ''))!r} vs leaf "
            f"{leaf.tensor_name!r} (intra {int(leaf.intra_graph_index)})"
        )
    src0_shape = [int(item) for item in trace_meta.get("src0_shape", [])]
    if len(src0_shape) < 2 or src0_shape[0] != leaf.k_dim or src0_shape[1] != leaf.n_dim:
        raise RuntimeError("slot view trace weight shape mismatch")
    src1_shape = [int(item) for item in trace_meta.get("src1_shape", [])]
    if len(src1_shape) < 2 or src1_shape[0] != leaf.k_dim:
        raise RuntimeError("slot view trace activation shape mismatch")
    # Tail-backed organic witnesses must be clean single-token decode
    # instances. Witnesses from a teacher-forced probe window (any request
    # with a decode audit) may instead come from the probe's final chunk,
    # which computes a few positions in ONE small GEMM (glm-5.2 live: a
    # 4-row output.weight instance, 2-row layer GEMMs). A decode-audit
    # opening then selects the audited position's row and the token/top-k
    # acceptance checks bind it; base light witnesses open no values at
    # all, so the small row bound adds no prover freedom in either case.
    probe_backed = bool(
        decode_openings
        or mesh_receipt.get("decode_audit_required", False)
        # Hard-tier payloads always regenerate witnesses in the exclusive
        # solo replay window, never from the shared serve.
        or str(payload.get("proof_mode", "")) == VERATHOS_GGML_GEMM_PROOF_MODE
    )
    max_rows = 8 if probe_backed else 1
    if not 1 <= int(src1_shape[1]) <= max_rows:
        raise RuntimeError(
            "slot view decode replay activation rows out of bounds: "
            f"{int(src1_shape[1])} (max {max_rows})"
        )
    types = {
        str(k): str(v) for k, v in (trace_meta.get("source_types", {}) or {}).items()
    }
    if types and dict(sorted(types.items())) != dict(sorted(leaf.source_types.items())):
        raise RuntimeError("slot view trace source types mismatch")


def mesh_trace_commitment_aggregate_root(entries: list[Mapping[str, Any]]) -> str:
    """Aggregate per-stage trace-set roots into one mesh receipt root."""

    normalized = []
    for item in entries:
        root = str(item.get("trace_commitment_root") or item.get("trace_set_root") or "")
        if not root:
            continue
        normalized.append(
            {
                "stage_index": int(item.get("stage_index", 0)),
                "trace_commitment_root": root,
                "trace_commitment_count": int(
                    item.get("trace_commitment_count")
                    or item.get("trace_set_count")
                    or 0
                ),
            }
        )
    if not normalized:
        return ""
    ordered = sorted(normalized, key=lambda item: item["stage_index"])
    h = hashlib.sha256(b"VERATHOS_MESH_TRACE_COMMITMENT_ROOT_V1")
    h.update(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def mesh_op_manifest_aggregate_root(entries: list[Mapping[str, Any]]) -> str:
    """Aggregate per-stage full GGML op-manifest roots into one mesh receipt root."""

    normalized = []
    for item in entries:
        root = str(item.get("op_manifest_root") or item.get("proof_op_manifest_root") or "")
        if not root:
            continue
        normalized.append(
            {
                "stage_index": int(item.get("stage_index", 0)),
                "op_manifest_root": root,
                "op_manifest_count": int(
                    item.get("op_manifest_count")
                    or item.get("proof_op_manifest_count")
                    or 0
                ),
            }
        )
    if not normalized:
        return ""
    ordered = sorted(normalized, key=lambda item: item["stage_index"])
    h = hashlib.sha256(b"VERATHOS_MESH_GGML_OP_MANIFEST_ROOT_V1")
    h.update(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def is_decode_candidate_manifest_entry(entry: GgmlOpManifestEntry) -> bool:
    """Return true for final projection/logit MUL_MAT candidates."""

    if entry.op_type != GGML_OP_MUL_MAT:
        return False
    tensor_name = entry.tensor_name.lower()
    src0_name = entry.src0_name.lower()
    src1_name = entry.src1_name.lower()
    dst_name = entry.dst_name.lower()
    projection_names = {tensor_name, src0_name}
    if (
        "output.weight" in projection_names
        or "lm_head.weight" in projection_names
        or any(name.endswith(".lm_head.weight") for name in projection_names)
    ):
        return True
    if "attn_output" in tensor_name or "attn_output" in src0_name:
        return False
    return (
        ("logit" in dst_name or "logits" in dst_name)
        or dst_name in {"result_output", "output"}
        or (
            "result_output" in dst_name
            and ("result_norm" in src1_name or "final" in src1_name)
        )
    )


def select_decode_manifest_entries(
    entries: list[GgmlOpManifestEntry],
    *,
    completion_token_ids: list[int],
    positions: list[int],
) -> dict[int, GgmlOpManifestEntry]:
    """Select final-logit manifest entries for sampled completion positions."""

    token_count = len(completion_token_ids)
    if token_count <= 0:
        raise RuntimeError("decode audit requires completion tokens")
    ordered_positions = sorted({int(position) for position in positions})
    if any(position < 0 or position >= token_count for position in ordered_positions):
        raise RuntimeError("decode audit position out of completion range")
    candidates = [
        entry
        for entry in _ordered_manifest_entries(entries)
        if is_decode_candidate_manifest_entry(entry)
    ]
    if len(candidates) < token_count:
        raise RuntimeError("decode audit could not find enough final-logit ops")
    completion_candidates = candidates[-token_count:]
    return {position: completion_candidates[position] for position in ordered_positions}


def ggml_proof_selection_payload(
    *,
    traces: list[GgmlMulMatTrace],
    manifest_entries: list[GgmlOpManifestEntry],
    receipt_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return manifest indexes that must exist as proof witnesses for a receipt."""

    ctx = receipt_context
    stage_index = int(ctx.get("stage_index", 0))
    proof_limit = max(1, int(ctx.get("proof_ops_per_request") or 1))
    if not manifest_entries:
        return {
            "version": 1,
            "stage_index": stage_index,
            "selected_manifest_indexes": [],
            "missing_manifest_indexes": [],
            "op_manifest_root": "",
            "op_manifest_count": 0,
        }

    manifest_ordered = _ordered_manifest_entries(manifest_entries)
    manifest_root = ggml_op_manifest_root(manifest_ordered)
    beacon = str(ctx.get("proof_beacon", ""))
    if not beacon:
        gate_hash = str(ctx.get("proof_gate_hash", ""))
        trace_root = ggml_trace_commitment_root(traces)
        beacon = (
            derive_every_request_trace_beacon(gate_hash).hex()
            if gate_hash
            else derive_standalone_trace_beacon(trace_root).hex()
            if trace_root
            else hashlib.sha256(
                b"VERATHOS_GGML_STANDALONE_MANIFEST_CHALLENGE_V1"
                + bytes.fromhex(manifest_root)
            ).hexdigest()
        )

    selected_leaf_indexes: list[int] = []
    selected_entries: list[GgmlOpManifestEntry] = []
    decode_required = bool(ctx.get("decode_audit_required", False))
    decode_stage_index = int(ctx.get("decode_audit_stage_index", stage_index))
    decode_active = bool(decode_required and stage_index == decode_stage_index)
    base_sampled = bool(ctx.get("proof_sampled", not decode_required))
    if base_sampled:
        selected_leaf_indexes = select_manifest_challenge_indexes(
            beacon=beacon,
            op_manifest_root=manifest_root,
            op_manifest_count=len(manifest_ordered),
            proof_ops_per_request=proof_limit,
            stage_index=stage_index,
        )
        selected_entries = [manifest_ordered[index] for index in selected_leaf_indexes]

    selected_by_manifest_index = {
        int(entry.manifest_index): entry for entry in selected_entries
    }
    if decode_active:
        decode_positions = [
            int(item) for item in ctx.get("decode_audit_positions", []) or []
        ]
        decode_token_ids = [
            int(item)
            for item in ctx.get("decode_audit_completion_token_ids", []) or []
        ]
        if not decode_token_ids:
            completion_count = int(ctx.get("completion_token_count", 0) or 0)
            if completion_count > 0:
                decode_token_ids = [0] * completion_count
        if decode_positions and decode_token_ids:
            decode_entries = select_decode_manifest_entries(
                manifest_ordered,
                completion_token_ids=decode_token_ids,
                positions=decode_positions,
            )
            for entry in decode_entries.values():
                selected_by_manifest_index[int(entry.manifest_index)] = entry
            leaf_by_manifest_index = {
                int(entry.manifest_index): index
                for index, entry in enumerate(manifest_ordered)
            }
            selected_leaf_indexes = [
                leaf_by_manifest_index[int(entry.manifest_index)]
                for entry in selected_by_manifest_index.values()
            ]

    selected_entries = list(selected_by_manifest_index.values())
    selected_manifest_indexes = [int(entry.manifest_index) for entry in selected_entries]
    ordered_traces = _ordered_traces(traces)
    missing_manifest_indexes = [
        int(entry.manifest_index)
        for entry in selected_entries
        if not any(entry.matches_trace(trace) for trace in ordered_traces)
    ]
    return {
        "version": 1,
        "stage_index": stage_index,
        "selected_manifest_indexes": selected_manifest_indexes,
        "selected_manifest_leaf_indexes": [int(index) for index in selected_leaf_indexes],
        "missing_manifest_indexes": missing_manifest_indexes,
        "op_manifest_root": manifest_root,
        "op_manifest_count": len(manifest_ordered),
    }


def make_decode_audit_openings_for_trace(
    trace: "GgmlMulMatTrace",
    *,
    decode_audit_positions: list[int],
    decode_audit_token_ids: list[int],
    decode_audit_top_k: int = 8,
    proved_logits_i32: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Open final logits for sampled decode positions from a selected trace."""

    if not decode_audit_positions:
        return []
    flat = np.fromfile(trace.dst_f32_path, dtype=np.float32).reshape(-1)
    if flat.size <= 0:
        raise RuntimeError("decode audit trace has empty logits")
    # A teacher-forced probe can compute logits for several positions of the
    # final chunk in ONE small multi-row GEMM (observed on glm-5.2: a
    # 4-row output.weight instance). The audited position's row is the one
    # whose argmax equals the committed token at that position, which is
    # exactly the acceptance criterion the verifier enforces on the opened
    # top-k, so this selection adds no prover freedom.
    vocab = int(trace.dst_shape[0]) if trace.dst_shape else flat.size
    if vocab <= 0 or flat.size % vocab:
        raise RuntimeError("decode audit trace logits shape mismatch")
    row_count = flat.size // vocab
    if row_count > 1 and proved_logits_i32 is not None:
        raise RuntimeError(
            "decode audit proved logits require a single-row instance"
        )
    matrix = flat.reshape(row_count, vocab)
    top_k = max(1, min(int(decode_audit_top_k), int(vocab)))
    row_argmax = [int(np.argmax(matrix[r])) for r in range(row_count)]

    def _row_for_token(token_id: int) -> int:
        for r in range(row_count):
            if row_argmax[r] == int(token_id):
                return r
        # Non-greedy organic traffic can commit a token that is not the
        # argmax; the verifier accepts it inside the committed top-k, so a
        # row containing it is a valid opening. Rows that do not carry the
        # token at all belong to another decode step and must not be opened.
        k = min(int(top_k), int(vocab))
        for r in range(row_count):
            top = np.argpartition(matrix[r], -k)[-k:]
            if int(token_id) in {int(idx) for idx in top}:
                return r
        raise RuntimeError(
            "decode audit instance has no row carrying the committed token "
            f"{int(token_id)} in its top-{k} "
            f"(rows {row_count}, argmaxes {row_argmax})"
        )


    def _top_indexes_for_row(row: int) -> np.ndarray:
        logits = matrix[row]
        # argpartition + sort of the k slice: O(vocab) instead of a full
        # O(vocab log vocab) argsort, same ordered result (250k-entry vocabs
        # make the full sort a measurable slice of every light proof).
        if top_k < logits.size:
            part = np.argpartition(logits, -top_k)[-top_k:]
            return part[np.argsort(logits[part])][::-1]
        return np.argsort(logits)[-top_k:][::-1]

    top_indexes_by_row: dict[int, np.ndarray] = {}
    dst_bytes = trace.dst_f32_path.read_bytes()
    dst_hash = hashlib.sha256(dst_bytes).hexdigest()
    proved_logits = None
    proved_logits_bytes = b""
    proved_argmax_token_id = -1
    proved_top_indexes: np.ndarray | None = None
    if proved_logits_i32 is not None:
        proved_logits = np.asarray(proved_logits_i32, dtype=np.int64).reshape(-1)
        if proved_logits.size != int(vocab):
            raise RuntimeError("decode audit proved logits size does not match f32 logits")
        proved_logits_bytes = logits_i32_to_bytes(proved_logits)
        proved_argmax_token_id = int(np.argmax(proved_logits))
        proved_top_indexes = np.argsort(proved_logits)[-top_k:][::-1]
    openings = []
    for position in sorted({int(item) for item in decode_audit_positions}):
        if position < 0 or position >= len(decode_audit_token_ids):
            raise RuntimeError("decode audit position out of completion range")
        token_id = int(decode_audit_token_ids[position])
        dst_row_index = _row_for_token(token_id)
        if dst_row_index not in top_indexes_by_row:
            top_indexes_by_row[dst_row_index] = _top_indexes_for_row(dst_row_index)
        top_indexes = top_indexes_by_row[dst_row_index]
        logits = matrix[dst_row_index]
        argmax_token_id = int(top_indexes[0])
        opening = {
            "version": 1,
            "mode": VERATHOS_GGUF_DECODE_AUDIT_MODE,
            "position": int(position),
            "token_id": token_id,
            "argmax_token_id": argmax_token_id,
            "argmax_logit": float(logits[argmax_token_id]),
            "top_token_ids": [int(item) for item in top_indexes.tolist()],
            "top_logits": [float(logits[int(item)]) for item in top_indexes.tolist()],
            "dst_row_index": int(dst_row_index),
            "dst_row_count": int(row_count),
            "dst_f32_sha256": dst_hash,
            "dst_f32_nbytes": len(dst_bytes),
            # The near-tie acceptance path reads ONLY hash-bound row bytes,
            # so an opening whose committed token is not the row argmax must
            # CARRY those bytes or the sanctioned tolerance can never engage
            # and every replay-vs-serve argmax flip lands in probation.
            # Exact-argmax
            # openings stay lean: the verifier never needs the row there.
            **(
                {"dst_f32_bytes_hex": dst_bytes.hex()}
                if token_id != argmax_token_id
                else {}
            ),
            "trace_graph_id": trace.graph_id,
            "trace_op_index": int(trace.op_index),
            "trace_manifest_index": (
                int(trace.manifest_index) if trace.manifest_index is not None else -1
            ),
        }
        if proved_logits is not None and proved_top_indexes is not None:
            opening.update(
                {
                    "proved_logits_i32_sha256": hashlib.sha256(
                        proved_logits_bytes
                    ).hexdigest(),
                    "proved_logits_i32_nbytes": len(proved_logits_bytes),
                    "proved_logits_i32_bytes_hex": proved_logits_bytes.hex(),
                    "proved_argmax_token_id": proved_argmax_token_id,
                    "proved_argmax_logit_i32": int(proved_logits[proved_argmax_token_id]),
                    "proved_top_token_ids": [
                        int(item) for item in proved_top_indexes.tolist()
                    ],
                    "proved_top_logits_i32": [
                        int(proved_logits[int(item)])
                        for item in proved_top_indexes.tolist()
                    ],
                }
            )
        openings.append(opening)
    return openings


def _decode_opening_commitment(opening: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(
        b"VERATHOS_GGUF_DECODE_AUDIT_OPENING_V1"
        + canonical_json_bytes(dict(opening))
    ).digest()


def ggml_decode_audit_receipt_root(payloads: list[Mapping[str, Any]]) -> str:
    """Commit all decode-audit openings returned for a mesh receipt."""

    openings: list[Mapping[str, Any]] = []
    for payload in payloads:
        raw = payload.get("decode_audit_openings", [])
        if isinstance(raw, list):
            openings.extend(item for item in raw if isinstance(item, Mapping))
    h = hashlib.sha256(b"VERATHOS_GGUF_DECODE_AUDIT_RECEIPT_ROOT_V1")
    for opening in sorted(openings, key=lambda item: int(item.get("position", -1))):
        h.update(_decode_opening_commitment(opening))
    return h.hexdigest()


def decode_audit_near_tie_accepts(
    opening: Mapping[str, Any],
    *,
    token_id: int,
    f32_argmax: int,
    top_k: int,
) -> tuple[bool, float | None]:
    """Accept a served token the replay row itself places in a near-tie.

    The decision reads ONLY the hash-bound dst_f32 row (never
    prover-supplied top-k metadata): the served token must sit inside the
    audited top-k of that row with a logit gap to the row argmax within
    the shipped tolerance. Returns (accepted, gap); gap is None when no
    hash-bound row is available (acceptance is then impossible).
    """

    dst_hex = str(opening.get("dst_f32_bytes_hex", ""))
    if not dst_hex:
        return False, None
    try:
        dst_bytes = bytes.fromhex(dst_hex)
    except ValueError:
        return False, None
    if (
        not dst_bytes
        or hashlib.sha256(dst_bytes).hexdigest()
        != str(opening.get("dst_f32_sha256", ""))
    ):
        return False, None
    # The hex carries the FULL hash-bound dst dump (that is what
    # dst_f32_sha256 commits); a teacher-forced probe instance may hold
    # several decode rows, so slice the audited one via the committed
    # row geometry before any argmax is taken.
    flat = np.frombuffer(dst_bytes, dtype="<f4")
    try:
        row_count = int(opening.get("dst_row_count", 1) or 1)
        row_index = int(opening.get("dst_row_index", 0) or 0)
    except (TypeError, ValueError):
        return False, None
    if row_count <= 0 or flat.size % row_count or not 0 <= row_index < row_count:
        return False, None
    row_f32 = flat.reshape(row_count, -1)[row_index]
    if not (0 <= token_id < row_f32.size and 0 <= f32_argmax < row_f32.size):
        return False, None
    # Tie-tolerant argmax consistency: the committed argmax index must hold
    # a MAXIMAL value. Requiring the exact np.argmax index rejects every
    # tied row outright, because the opening's argmax comes from an
    # argpartition whose index choice among ties is arbitrary (a fully
    # flat row - the token-0 degeneration signature - picks a mid-vocab
    # run like [82770..82774]).
    top1 = float(row_f32[f32_argmax])
    row_max = float(np.max(row_f32))
    if not (np.isfinite(top1) and top1 == row_max):
        return False, None
    gap = top1 - float(row_f32[token_id])
    # Value-based rank, ties inclusive - the same tie-tolerant pattern as
    # the sumcheck-backed quantized check (verallm/sampling.py): index
    # membership in an argpartition top-k is arbitrary under ties, so a
    # faithful greedy pick over a tied row would never be "in" it.
    k = min(int(top_k), int(row_f32.size))
    strictly_greater = int(np.sum(row_f32 > row_f32[token_id]))
    tolerance = (
        VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_ABS
        + VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_REL * abs(top1)
    )
    return (strictly_greater < k and 0.0 <= gap <= tolerance), gap


def verify_ggml_decode_audit_payloads(
    receipt: Mapping[str, Any],
    payloads: list[Mapping[str, Any]],
    *,
    completion_token_ids: list[int],
    receipts: list[Mapping[str, Any]] | None = None,
) -> GgmlProofVerification:
    """Verify sampled final-logit/sampler openings."""

    started = time.perf_counter()
    try:
        if receipts is not None:
            _pair_proof_payloads_and_receipts(payloads, receipts)
        decode_stage_index = int(receipt.get("decode_audit_stage_index", -1))
        if decode_stage_index < 0:
            return GgmlProofVerification(
                False,
                0.0,
                "decode audit stage index missing",
            )
        expected_root = str(receipt.get("decode_audit_receipt_root", ""))
        actual_root = ggml_decode_audit_receipt_root(payloads)
        if expected_root and expected_root != actual_root:
            return GgmlProofVerification(False, 0.0, "decode audit receipt root mismatch")
        expected_positions = [int(item) for item in receipt.get("decode_audit_positions", [])]
        if not expected_positions:
            return GgmlProofVerification(False, 0.0, "decode audit positions missing")
        if len(expected_positions) != len(set(expected_positions)):
            return GgmlProofVerification(
                False,
                0.0,
                "decode audit positions contain duplicates",
            )
        expected_position_set = set(expected_positions)
        openings: dict[int, Mapping[str, Any]] = {}
        light_positions: set[int] = set()
        for payload in payloads:
            raw = payload.get("decode_audit_openings", [])
            if not isinstance(raw, list):
                continue
            payload_openings = [item for item in raw if isinstance(item, Mapping)]
            if len(payload_openings) != len(raw):
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit opening must be an object",
                )
            if not payload_openings:
                continue
            membership = payload.get("op_manifest_membership", {})
            slot_membership = payload.get("slot_view_membership", {})
            if isinstance(slot_membership, Mapping) and slot_membership:
                leaf_raw = slot_membership.get("leaf")
                if not isinstance(leaf_raw, Mapping) or not _slot_view_op_is_decode_candidate(
                    leaf_raw
                ):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit opening is not bound to a final-logit slot-view op",
                    )
                try:
                    verify_slot_view_proof_payload(payload, mesh_receipt=receipt)
                except RuntimeError as exc:
                    return GgmlProofVerification(False, 0.0, str(exc))
                payload_stage_index = int(slot_membership.get("stage_index", -1))
            elif isinstance(membership, Mapping) and membership:
                entry_raw = membership.get("entry")
                if not isinstance(entry_raw, Mapping):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit opening missing op-manifest entry",
                    )
                entry = GgmlOpManifestEntry.from_mapping(entry_raw)
                if not is_decode_candidate_manifest_entry(entry):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit opening is not bound to a final-logit op",
                    )
                try:
                    _verify_payload_op_manifest_membership(payload, membership)
                except RuntimeError as exc:
                    return GgmlProofVerification(False, 0.0, str(exc))
                payload_stage_index = int(membership.get("stage_index", -1))
            else:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit opening missing committed op membership",
                )
            if payload_stage_index != decode_stage_index:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit opening came from the wrong mesh stage",
                )
            trace_meta = payload.get("trace", {})
            if not isinstance(trace_meta, Mapping):
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proof trace metadata missing",
                )
            for item in payload_openings:
                position = int(item.get("position", -1))
                if position not in expected_position_set:
                    return GgmlProofVerification(
                        False,
                        0.0,
                        f"decode audit opening has unexpected position: {position}",
                    )
                if position in openings:
                    return GgmlProofVerification(
                        False,
                        0.0,
                        f"decode audit opening duplicated position: {position}",
                    )
                if str(item.get("trace_graph_id", "")) != str(
                    trace_meta.get("graph_id", "")
                ) or int(item.get("trace_op_index", -1)) != int(
                    trace_meta.get("op_index", -1)
                ):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit opening trace binding mismatch",
                    )
                openings[position] = item
                if (
                    str(payload.get("proof_mode", ""))
                    == VERATHOS_GGML_LIGHT_PROOF_MODE
                ):
                    light_positions.add(position)
        missing = sorted(set(expected_positions) - set(openings))
        if missing:
            return GgmlProofVerification(
                False,
                0.0,
                "decode audit openings missing positions: "
                + ",".join(str(item) for item in missing),
            )
        unexpected = sorted(set(openings) - expected_position_set)
        if unexpected:
            return GgmlProofVerification(
                False,
                0.0,
                "decode audit openings contain unexpected positions: "
                + ",".join(str(item) for item in unexpected),
            )
        for position in expected_positions:
            if position < 0 or position >= len(completion_token_ids):
                return GgmlProofVerification(False, 0.0, "decode audit position out of range")
            opening = openings[position]
            token_id = int(completion_token_ids[position])
            if int(opening.get("token_id", -1)) != token_id:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit token id mismatch: "
                    f"position={position} expected={token_id} "
                        f"opening={opening.get('token_id', -1)}",
                )
            if position in light_positions:
                # LIGHT tier: no proved (sumcheck-backed) logits exist by
                # design. The streamed token must sit in the committed top-k
                # the miner opened and signed; correctness of those logits is
                # what the hard canary lane enforces.
                top_ids = [int(item) for item in opening.get("top_token_ids", [])]
                if not top_ids or int(opening.get("argmax_token_id", -1)) != top_ids[0]:
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit light top-k metadata malformed",
                    )
                if token_id not in top_ids:
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit light token is not in the committed "
                        f"top-{len(top_ids)}",
                    )
                continue
            proved_hex = str(opening.get("proved_logits_i32_bytes_hex", ""))
            if not proved_hex:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved logits row missing",
                )
            try:
                proved_bytes = bytes.fromhex(proved_hex)
            except ValueError:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved logits bytes invalid",
                )
            if hashlib.sha256(proved_bytes).hexdigest() != str(
                opening.get("proved_logits_i32_sha256", "")
            ):
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved logits hash mismatch",
                )
            if len(proved_bytes) != int(opening.get("proved_logits_i32_nbytes", -1)):
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved logits size mismatch",
                )
            logits_i32 = logits_i32_from_bytes(proved_bytes)
            if logits_i32.size <= 0:
                return GgmlProofVerification(False, 0.0, "decode audit empty logits row")
            proved_argmax = int(np.argmax(logits_i32))
            if int(opening.get("proved_argmax_token_id", -1)) != proved_argmax:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved argmax mismatch",
                )
            top_k = int(
                receipt.get(
                    "decode_audit_top_k",
                    VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
                )
                or VERATHOS_GGUF_DECODE_AUDIT_TOP_K
            )
            if not 1 <= top_k <= VERATHOS_GGUF_DECODE_AUDIT_TOP_K:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit top-k exceeds the verified profile",
                )
            argmax_ok, argmax_detail = verify_quantized_argmax(
                logits_i32,
                token_id,
                top_k=top_k,
            )
            if not argmax_ok:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit proved logits token mismatch: " + argmax_detail,
                )
            top_ids = [int(item) for item in opening.get("top_token_ids", [])]
            f32_argmax = int(opening.get("argmax_token_id", -1))
            if not top_ids or f32_argmax != top_ids[0]:
                return GgmlProofVerification(
                    False,
                    0.0,
                    "decode audit f32 top-k metadata mismatch",
                )
            if token_id != f32_argmax:
                # Split meshes legitimately flip near-tied argmaxes between
                # the batched serve path and the single-row f32 replay
                # (different reduction orders); the quantized layer already
                # accepts top-k membership for exactly this reason
                # (verallm/sampling.py). Demanding exact f32 equality here
                # re-tightened past the sanctioned tolerance and failed
                # honest serves. Accept a served
                # token that the REPLAY ROW ITSELF (hash-bound dst_f32
                # bytes, never prover metadata) places inside the audited
                # top-k within the runtime's shipped logit tolerance;
                # degenerate substitutions sit far outside it.
                near_tie_ok, near_tie_gap = decode_audit_near_tie_accepts(
                    opening,
                    token_id=token_id,
                    f32_argmax=f32_argmax,
                    top_k=top_k,
                )
                if not near_tie_ok:
                    # Diagnosable failure detail: WHICH ids disagree and
                    # whether the committed argmax matches a NEIGHBORING
                    # completion token (the signature of an off-by-one in
                    # the capture row) - a bare mismatch message was
                    # undebuggable from the gate.
                    neighbors = []
                    if position > 0 and f32_argmax == int(
                        completion_token_ids[position - 1]
                    ):
                        neighbors.append("f32_argmax == completion[position-1]")
                    if position + 1 < len(completion_token_ids) and f32_argmax == int(
                        completion_token_ids[position + 1]
                    ):
                        neighbors.append("f32_argmax == completion[position+1]")
                    detail = (
                        f"position={position} served_token={token_id} "
                        f"f32_argmax={f32_argmax} top_ids={top_ids[:5]}"
                        + (
                            f" near_tie_gap={near_tie_gap:.6f}"
                            if near_tie_gap is not None
                            else " (no hash-bound f32 row for near-tie check)"
                        )
                        + (" (" + "; ".join(neighbors) + ")" if neighbors else "")
                    )
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit token is not the committed f32 argmax: "
                        + detail,
                    )
            dst_hex = str(opening.get("dst_f32_bytes_hex", ""))
            if dst_hex:
                try:
                    dst_bytes = bytes.fromhex(dst_hex)
                except ValueError:
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit logits bytes invalid",
                    )
                if hashlib.sha256(dst_bytes).hexdigest() != str(
                    opening.get("dst_f32_sha256", "")
                ):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit logits hash mismatch",
                    )
                if len(dst_bytes) != int(opening.get("dst_f32_nbytes", -1)):
                    return GgmlProofVerification(
                        False,
                        0.0,
                        "decode audit logits size mismatch",
                    )
        return GgmlProofVerification(True, (time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        return GgmlProofVerification(False, (time.perf_counter() - started) * 1000.0, str(exc))


def _quantize_int8(array: np.ndarray) -> tuple[np.ndarray, float]:
    return quantize_proof_i8(array)


@dataclass(frozen=True)
class GgmlOpManifestEntry:
    """Cheap metadata commitment for one proof-eligible GGML op execution."""

    path: Path
    created_unix_ns: int
    manifest_index: int
    graph_id: str
    op_index: int
    op_type: str
    tensor_name: str
    src0_shape: tuple[int, ...]
    src1_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    source_types: dict[str, str]
    src0_name: str = ""
    src1_name: str = ""
    dst_name: str = ""
    src0_raw_path: Path | None = None
    src0_raw_type: str = ""
    backend: str = "llama_cpp_rpc"
    device: str = "unknown"
    proof_eligible: bool = True
    expert_index: int | None = None
    # Backend-local graph ordinal within a capture window (v3 rows only).
    # Bookkeeping for slot-view attribution; intentionally excluded from
    # entry_hash so v2/v3 runtimes commit identical entries at --parallel 1.
    graph_seq: int = -1
    intra_graph_index: int = -1

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path = "",
    ) -> "GgmlOpManifestEntry":
        if int(data.get("version", 0)) != 1:
            raise ValueError("unsupported GGML op manifest version")
        op_type = str(data.get("op_type", ""))
        if op_type != GGML_OP_MUL_MAT:
            raise ValueError("manifest entry is not a GGML MUL_MAT op")
        return cls(
            path=Path(path),
            created_unix_ns=int(data["created_unix_ns"]),
            manifest_index=int(data.get("manifest_index", data.get("op_index", 0))),
            graph_id=str(data["graph_id"]),
            op_index=int(data["op_index"]),
            op_type=op_type,
            tensor_name=str(data.get("tensor_name", "")),
            src0_name=str(data.get("src0_name", "")),
            src1_name=str(data.get("src1_name", "")),
            dst_name=str(data.get("dst_name", "")),
            src0_shape=_shape_tuple(data["src0_shape"]),
            src1_shape=_shape_tuple(data["src1_shape"]),
            dst_shape=_shape_tuple(data["dst_shape"]),
            source_types={str(k): str(v) for k, v in data.get("source_types", {}).items()},
            backend=str(data.get("backend", "llama_cpp_rpc")),
            device=str(data.get("device", "unknown")),
            proof_eligible=bool(data.get("proof_eligible", True)),
            graph_seq=int(data.get("graph_seq", -1)),
            intra_graph_index=int(data.get("intra_graph_index", -1)),
            expert_index=(
                int(data["expert_index"]) if data.get("expert_index") is not None else None
            ),
        )

    @classmethod
    def from_compact_line(
        cls,
        line: str,
        *,
        path: str | Path = "",
    ) -> "GgmlOpManifestEntry":
        fields = line.rstrip("\n\r").split("\t")
        if fields[0] in (
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V2",
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3",
        ):
            is_v3 = fields[0] == "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3"
            if len(fields) != (17 if is_v3 else 15):
                raise ValueError("unsupported compact GGML op manifest line")
            manifest_index = int(fields[2])
            tensor_name = fields[3]
            backend = fields[12]
            device = fields[13]
            return cls(
                path=Path(path),
                created_unix_ns=int(fields[1]),
                manifest_index=manifest_index,
                graph_id=_compact_graph_id(
                    backend=backend,
                    device=device,
                    manifest_index=manifest_index,
                ),
                op_index=manifest_index,
                op_type=GGML_OP_MUL_MAT,
                tensor_name=tensor_name,
                src0_name=tensor_name,
                src1_name=fields[4],
                dst_name=fields[5],
                src0_shape=_compact_shape_tuple(fields[6]),
                src1_shape=_compact_shape_tuple(fields[7]),
                dst_shape=_compact_shape_tuple(fields[8]),
                source_types={
                    "src0": fields[9],
                    "src1": fields[10],
                    "dst": fields[11],
                },
                backend=backend,
                device=device,
                proof_eligible=fields[14] == "1",
                graph_seq=int(fields[15]) if is_v3 else -1,
                intra_graph_index=int(fields[16]) if is_v3 else -1,
            )
        if len(fields) != 18 or fields[0] not in (
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_V1",
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V1",
        ):
            raise ValueError("unsupported compact GGML op manifest line")
        raw_strings = fields[0] == "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V1"

        def decode_hex(raw: str) -> str:
            return bytes.fromhex(raw).decode("utf-8")

        def decode_field(raw: str) -> str:
            return raw if raw_strings else decode_hex(raw)

        def shape(raw: str) -> tuple[int, ...]:
            values = tuple(int(item) for item in raw.split(",") if item)
            if len(values) != 4:
                raise ValueError("compact GGML op manifest shape must have 4 dims")
            return values

        return cls(
            path=Path(path),
            created_unix_ns=int(fields[1]),
            manifest_index=int(fields[2]),
            graph_id=decode_field(fields[3]),
            op_index=int(fields[4]),
            op_type=GGML_OP_MUL_MAT,
            tensor_name=decode_field(fields[5]),
            src0_name=decode_field(fields[6]),
            src1_name=decode_field(fields[7]),
            dst_name=decode_field(fields[8]),
            src0_shape=shape(fields[9]),
            src1_shape=shape(fields[10]),
            dst_shape=shape(fields[11]),
            source_types={
                "src0": fields[12],
                "src1": fields[13],
                "dst": fields[14],
            },
            backend=decode_field(fields[15]),
            device=decode_field(fields[16]),
            proof_eligible=fields[17] == "1",
        )

    def to_commitment_body(self) -> dict[str, Any]:
        body = {
            "version": 1,
            "created_unix_ns": int(self.created_unix_ns),
            "manifest_index": int(self.manifest_index),
            "graph_id": self.graph_id,
            "op_index": int(self.op_index),
            "op_type": self.op_type,
            "tensor_name": self.tensor_name,
            "src0_shape": list(self.src0_shape),
            "src1_shape": list(self.src1_shape),
            "dst_shape": list(self.dst_shape),
            "source_types": dict(sorted(self.source_types.items())),
            "backend": self.backend,
            "device": self.device,
            "proof_eligible": bool(self.proof_eligible),
        }
        if self.src0_name or self.src1_name or self.dst_name:
            body["src0_name"] = self.src0_name
            body["src1_name"] = self.src1_name
            body["dst_name"] = self.dst_name
        return body

    def entry_hash(self) -> str:
        return hashlib.sha256(
            b"VERATHOS_GGML_OP_MANIFEST_ENTRY_V1"
            + canonical_json_bytes(self.to_commitment_body())
        ).hexdigest()

    def matches_trace(self, trace: "GgmlMulMatTrace") -> bool:
        if trace.manifest_index is not None and int(trace.manifest_index) != int(
            self.manifest_index
        ):
            return False
        return (
            self.op_type == GGML_OP_MUL_MAT
            and trace.graph_id == self.graph_id
            and int(trace.op_index) == int(self.op_index)
            and trace.tensor_name == self.tensor_name
            and (
                not self.src0_name
                or not trace.src0_name
                or trace.src0_name == self.src0_name
            )
            and (
                not self.src1_name
                or not trace.src1_name
                or trace.src1_name == self.src1_name
            )
            and (
                not self.dst_name
                or not trace.dst_name
                or trace.dst_name == self.dst_name
            )
            and tuple(trace.src0_shape) == tuple(self.src0_shape)
            and tuple(trace.src1_shape) == tuple(self.src1_shape)
            and tuple(trace.dst_shape) == tuple(self.dst_shape)
            and dict(sorted(trace.source_types.items()))
            == dict(sorted(self.source_types.items()))
            and trace.backend == self.backend
            and trace.device == self.device
        )


@dataclass(frozen=True)
class GgmlMulMatTrace:
    """A dumped GGML MUL_MAT witness from a proof-capable RPC worker."""

    path: Path
    created_unix_ns: int
    graph_id: str
    op_index: int
    tensor_name: str
    src0_shape: tuple[int, ...]
    src1_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    src0_f32_path: Path | None
    src1_f32_path: Path | None
    dst_f32_path: Path | None
    source_types: dict[str, str]
    src0_name: str = ""
    src1_name: str = ""
    dst_name: str = ""
    src0_raw_path: Path | None = None
    src0_raw_type: str = ""
    backend: str = "llama_cpp_rpc"
    device: str = "unknown"
    manifest_index: int | None = None
    graph_seq: int = -1
    intra_graph_index: int = -1
    expert_index: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "GgmlMulMatTrace":
        trace_path = Path(path)
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != 1:
            raise ValueError("unsupported GGML trace version")
        op_type = str(data.get("op_type", ""))
        if op_type not in (GGML_OP_MUL_MAT, "GGML_OP_MUL_MAT_ID"):
            raise ValueError("trace is not a GGML MUL_MAT witness")

        def resolve(name: str) -> Path:
            raw = Path(str(data[name]))
            return raw if raw.is_absolute() else trace_path.parent / raw

        return cls(
            path=trace_path,
            created_unix_ns=int(data["created_unix_ns"]),
            graph_id=str(data["graph_id"]),
            op_index=int(data["op_index"]),
            tensor_name=str(data.get("tensor_name", "")),
            src0_name=str(data.get("src0_name", "")),
            src1_name=str(data.get("src1_name", "")),
            dst_name=str(data.get("dst_name", "")),
            src0_shape=tuple(int(item) for item in data["src0_shape"]),
            src1_shape=tuple(int(item) for item in data["src1_shape"]),
            dst_shape=tuple(int(item) for item in data["dst_shape"]),
            src0_f32_path=resolve("src0_f32") if data.get("src0_f32") else None,
            src1_f32_path=resolve("src1_f32"),
            dst_f32_path=resolve("dst_f32"),
            source_types={str(k): str(v) for k, v in data.get("source_types", {}).items()},
            src0_raw_path=resolve("src0_raw") if data.get("src0_raw") else None,
            src0_raw_type=str(data.get("src0_raw_type", "")),
            backend=str(data.get("backend", "llama_cpp_rpc")),
            device=str(data.get("device", "unknown")),
            manifest_index=(
                int(data["manifest_index"])
                if data.get("manifest_index") is not None
                else None
            ),
            graph_seq=int(data.get("graph_seq", -1)),
            intra_graph_index=int(data.get("intra_graph_index", -1)),
            expert_index=(
                int(data["expert_index"]) if data.get("expert_index") is not None else None
            ),
        )

    def load_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the GGML witness as X, W, Y where Y ~= X @ W."""

        if len(self.src0_shape) < 2 or len(self.src1_shape) < 2 or len(self.dst_shape) < 2:
            raise ValueError("trace shapes must have at least two dimensions")
        if any(dim != 1 for dim in self.src0_shape[2:]):
            raise ValueError("batched src0 MUL_MAT traces are not supported yet")
        if any(dim != 1 for dim in self.src1_shape[2:]):
            raise ValueError("batched src1 MUL_MAT traces are not supported yet")
        if any(dim != 1 for dim in self.dst_shape[2:]):
            raise ValueError("batched dst MUL_MAT traces are not supported yet")

        if self.src0_f32_path is None:
            raise ValueError("trace does not include src0_f32 witness bytes")
        src0 = np.fromfile(self.src0_f32_path, dtype=np.float32)
        src1 = np.fromfile(self.src1_f32_path, dtype=np.float32)
        dst = np.fromfile(self.dst_f32_path, dtype=np.float32)

        k = self.src0_shape[0]
        n = self.src0_shape[1]
        m = self.src1_shape[1]
        if self.src1_shape[0] != k:
            raise ValueError("src0/src1 inner dimensions do not match")
        if self.dst_shape[0] != n or self.dst_shape[1] != m:
            raise ValueError("dst shape does not match GGML MUL_MAT dimensions")
        if src0.size != k * n or src1.size != k * m or dst.size != n * m:
            raise ValueError("trace binary size does not match metadata")

        # GGML stores ne0 as the fastest dimension. For MUL_MAT:
        # src0: [K, N], src1: [K, M], dst: [N, M].
        # The conventional proof relation is X[M,K] @ W[K,N] = Y[M,N].
        weight_rows = src0.reshape(n, k)
        x = src1.reshape(m, k)
        y = dst.reshape(m, n)
        return x, weight_rows.T, y

    @property
    def total_elements(self) -> int:
        return (
            int(np.prod(self.src0_shape))
            + int(np.prod(self.src1_shape))
            + int(np.prod(self.dst_shape))
        )

    def commitment_hash(self) -> str:
        """Cheap pre-proof commitment to the captured GGML witness files."""

        body = {
            "version": 1,
            "created_unix_ns": int(self.created_unix_ns),
            "graph_id": self.graph_id,
            "op_index": int(self.op_index),
            "tensor_name": self.tensor_name,
            "src0_shape": list(self.src0_shape),
            "src1_shape": list(self.src1_shape),
            "dst_shape": list(self.dst_shape),
            "src1_f32_sha256": _file_sha256(self.src1_f32_path),
            "dst_f32_sha256": _file_sha256(self.dst_f32_path),
            "source_types": dict(sorted(self.source_types.items())),
            "backend": self.backend,
            "device": self.device,
        }
        if self.src0_f32_path is not None:
            body["src0_f32_sha256"] = _file_sha256(self.src0_f32_path)
        if self.src0_raw_path is not None:
            body["src0_raw_sha256"] = _file_sha256(self.src0_raw_path)
            body["src0_raw_type"] = self.src0_raw_type
        if self.src0_name or self.src1_name or self.dst_name:
            body["src0_name"] = self.src0_name
            body["src1_name"] = self.src1_name
            body["dst_name"] = self.dst_name
        if self.manifest_index is not None:
            body["manifest_index"] = int(self.manifest_index)
        if self.expert_index is not None:
            body["expert_index"] = int(self.expert_index)
        return hashlib.sha256(
            b"VERATHOS_GGML_TRACE_COMMITMENT_V1" + canonical_json_bytes(body)
        ).hexdigest()


def _manifest_trace_io_candidates(
    trace: GgmlMulMatTrace,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if len(trace.src0_shape) < 2 or len(trace.src1_shape) < 2 or len(trace.dst_shape) < 2:
        raise ValueError("trace shapes must have at least two dimensions")
    if any(dim != 1 for dim in trace.src1_shape[2:]):
        raise ValueError("batched src1 MUL_MAT traces are not supported yet")
    if any(dim != 1 for dim in trace.dst_shape[2:]):
        raise ValueError("batched dst MUL_MAT traces are not supported yet")
    k = int(trace.src0_shape[0])
    n = int(trace.src0_shape[1])
    m = int(trace.src1_shape[1])
    if int(trace.src1_shape[0]) != k:
        raise ValueError("src0/src1 inner dimensions do not match")
    if int(trace.dst_shape[0]) != n or int(trace.dst_shape[1]) != m:
        raise ValueError("dst shape does not match GGML MUL_MAT dimensions")
    src1 = np.fromfile(trace.src1_f32_path, dtype=np.float32)
    dst = np.fromfile(trace.dst_f32_path, dtype=np.float32)
    if src1.size != k * m or dst.size != n * m:
        raise ValueError("trace activation/output binary size does not match metadata")

    # CUDA and RPC traces have historically dumped GGML's ne0-fastest memory
    # order, which is consumed as [M,K] activations and [M,N] outputs. Metal
    # can materialize contiguous host reads in logical [K,M]/[N,M] order for
    # some tensors, so manifest-backed verification tries both deterministic
    # interpretations and records the one that matches the committed output.
    src1_ggml = src1.reshape(m, k)
    dst_ggml = dst.reshape(m, n)
    src1_logical = src1.reshape(k, m).T
    dst_logical = dst.reshape(n, m).T
    candidates = [
        ("ggml_ne0_fast", src1_ggml, dst_ggml),
        ("logical_ne0_ne1", src1_logical, dst_logical),
        ("src1_logical_dst_ggml", src1_logical, dst_ggml),
        ("src1_ggml_dst_logical", src1_ggml, dst_logical),
    ]
    deduped: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for name, x, y in candidates:
        key = (np.ascontiguousarray(x).tobytes(), np.ascontiguousarray(y).tobytes())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, np.ascontiguousarray(x), np.ascontiguousarray(y)))
    return deduped


# The float recompute is a TOLERANCE GATE, not the cryptographic proof: it
# rejects a trace whose captured output wildly disagrees with x @ w. The
# actual attestation is the beacon-challenged sumcheck block plus the full
# y Merkle commitment, both O(1) in context. Recomputing x @ w over EVERY
# context row made the whole hard prove O(context) (measured 1.97s at ctx=1,
# 17.7s at ctx=128), because it is an [ctx, K] @ [K, N] matmul. v3 Gleipnir
# never recomputes the full sequence; it replays a bounded nonce-selected row
# set. Adopting that here: the gate recomputes at most this many rows. The
# sumcheck and the full-y commitment are unchanged, so per-row soundness is
# unaffected and, as in v3, beacon selection across requests covers the rows.
# Sized to the canonical proof block: the sumcheck cryptographically covers a
# block_size-row block, so the heuristic gate covering the same order of rows
# is the right scope and makes the gate O(1) in context (a fixed
# block_size x K x N recompute at any sequence length), a small multiple of
# the block for tolerance robustness.
_HARD_RECOMPUTE_ROW_SAMPLE = 2 * GGML_CANONICAL_MIN_PROOF_BLOCK_SIZE


def _recompute_row_slice(rows: int) -> int:
    if _HARD_RECOMPUTE_ROW_SAMPLE <= 0 or rows <= _HARD_RECOMPUTE_ROW_SAMPLE:
        return rows
    return _HARD_RECOMPUTE_ROW_SAMPLE


def _reference_matmul(xs: np.ndarray, w_f32: np.ndarray) -> np.ndarray:
    """xs @ w via torch, on GPU when available.

    This venv's numpy has no optimized BLAS: a 128x3584x18944 reference matmul
    took ~18 s, which was the whole context-scaling cost even after bounding
    the rows. torch.mm uses GPU or an optimized CPU kernel, so the bounded
    reference is milliseconds. Falls back to numpy if torch matmul raises.
    """
    try:
        xt = torch.from_numpy(np.ascontiguousarray(xs, dtype=np.float32))
        wt = torch.from_numpy(np.ascontiguousarray(w_f32, dtype=np.float32))
        if torch.cuda.is_available():
            out = (xt.cuda() @ wt.cuda()).cpu().numpy()
        else:
            out = (xt @ wt).numpy()
        return out
    except Exception:
        return xs @ w_f32


def _float_recompute_error(
    x_f32: np.ndarray,
    w_f32: np.ndarray,
    y_f32: np.ndarray,
) -> tuple[float, float, float]:
    s = _recompute_row_slice(int(x_f32.shape[0])) if x_f32.ndim == 2 else int(x_f32.shape[0])
    xs = x_f32[:s]
    ys = y_f32[:s]
    y_reference = _reference_matmul(xs, w_f32)
    diff = np.abs(y_reference - ys)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    y_scale = float(max(np.max(np.abs(ys)) if ys.size else 0.0, 1.0))
    max_rel = float(max_abs / y_scale)
    return max_abs, mean_abs, max_rel


def _float_recompute_ok(
    *,
    max_abs: float,
    mean_abs: float,
    max_rel: float,
    y_f32: np.ndarray,
    tolerance_abs: float,
    tolerance_rel: float,
) -> bool:
    y_scale = float(max(np.max(np.abs(y_f32)) if y_f32.size else 0.0, 1.0))
    abs_ok = np.isfinite(max_abs) and max_abs <= tolerance_abs
    rel_ok = (
        np.isfinite(max_rel)
        and max_rel <= float(tolerance_rel)
        and np.isfinite(mean_abs)
        and mean_abs <= max(float(tolerance_abs), float(tolerance_rel) * y_scale)
    )
    return bool(abs_ok or rel_ok)


def ggml_trace_commitment_root(traces: list[GgmlMulMatTrace]) -> str:
    """Deterministic root over captured GGML witness commitments."""

    if not traces:
        return ""
    commitments = [trace.commitment_hash() for trace in _ordered_traces(traces)]
    return _trace_set_root_from_commitments(commitments)


def ggml_op_manifest_root(entries: list[GgmlOpManifestEntry]) -> str:
    """Deterministic Merkle root over full proof-eligible GGML op metadata."""

    if not entries:
        return ""
    hashes = [entry.entry_hash() for entry in _ordered_manifest_entries(entries)]
    return _manifest_root_from_hashes(hashes)


def _json_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True, separators=(",", ":"))


def _compact_shape_json(raw: str) -> str:
    values = tuple(int(item) for item in raw.split(",") if item)
    if len(values) != 4:
        raise ValueError("compact GGML op manifest shape must have 4 dims")
    return "[" + ",".join(str(item) for item in values) + "]"


def _compact_shape_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item)
    if len(values) != 4:
        raise ValueError("compact GGML op manifest shape must have 4 dims")
    return values


def _compact_graph_id(*, backend: str, device: str, manifest_index: int) -> str:
    if backend == "llama_cpp_metal":
        return f"metal-mul-mat-{manifest_index}"
    if backend == "llama_cpp_cuda":
        raw_device = str(device)
        suffix = raw_device[4:] if raw_device.startswith("CUDA") else raw_device
        return f"cuda{suffix}-mul-mat-{manifest_index}"
    if backend == "llama_cpp_cpu":
        return f"cpu-mul-mat-{manifest_index}"
    if backend == "llama_cpp_vulkan":
        return f"vulkan-mul-mat-{manifest_index}"
    return f"{backend}-mul-mat-{manifest_index}"


def _compact_manifest_line_hash_and_order(
    line: str,
    *,
    path: Path,
) -> tuple[int, tuple[Any, ...], tuple[Any, ...], str] | None:
    fields = line.rstrip("\n\r").split("\t")
    if fields[0] in (
        "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V2",
        "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3",
    ):
        is_v3 = fields[0] == "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V3"
        if len(fields) != (17 if is_v3 else 15):
            raise ValueError("unsupported compact GGML op manifest line")
        if fields[14] != "1":
            return None
        created_unix_ns = int(fields[1])
        manifest_index = int(fields[2])
        tensor_name = fields[3]
        src0_name = tensor_name
        src1_name = fields[4]
        dst_name = fields[5]
        src0_shape_raw = fields[6]
        src1_shape_raw = fields[7]
        dst_shape_raw = fields[8]
        src0_shape = _compact_shape_json(src0_shape_raw)
        src1_shape = _compact_shape_json(src1_shape_raw)
        dst_shape = _compact_shape_json(dst_shape_raw)
        src0_type = fields[9]
        src1_type = fields[10]
        dst_type = fields[11]
        backend = fields[12]
        device = fields[13]
        graph_id = _compact_graph_id(
            backend=backend,
            device=device,
            manifest_index=manifest_index,
        )
        op_index = manifest_index
    else:
        if len(fields) != 18 or fields[0] not in (
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_V1",
            "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V1",
        ):
            raise ValueError("unsupported compact GGML op manifest line")
        if fields[17] != "1":
            return None
        raw_strings = fields[0] == "VERATHOS_GGML_OP_MANIFEST_COMPACT_RAW_V1"

        def decode_hex(raw: str) -> str:
            return bytes.fromhex(raw).decode("utf-8")

        def decode_field(raw: str) -> str:
            return raw if raw_strings else decode_hex(raw)

        created_unix_ns = int(fields[1])
        manifest_index = int(fields[2])
        graph_id = decode_field(fields[3])
        op_index = int(fields[4])
        tensor_name = decode_field(fields[5])
        src0_name = decode_field(fields[6])
        src1_name = decode_field(fields[7])
        dst_name = decode_field(fields[8])
        backend = decode_field(fields[15])
        device = decode_field(fields[16])
        src0_shape_raw = fields[9]
        src1_shape_raw = fields[10]
        dst_shape_raw = fields[11]
        src0_shape = _compact_shape_json(src0_shape_raw)
        src1_shape = _compact_shape_json(src1_shape_raw)
        dst_shape = _compact_shape_json(dst_shape_raw)
        src0_type = fields[12]
        src1_type = fields[13]
        dst_type = fields[14]
    if not _is_provable_weight_op(_compact_shape_tuple(src0_shape_raw)):
        return None
    has_names = bool(src0_name or src1_name or dst_name)

    body = [
        '{"backend":',
        _json_string(backend),
        ',"created_unix_ns":',
        str(created_unix_ns),
        ',"device":',
        _json_string(device),
    ]
    if has_names:
        body.extend(
            [
                ',"dst_name":',
                _json_string(dst_name),
            ]
        )
    body.extend(
        [
            ',"dst_shape":',
            dst_shape,
            ',"graph_id":',
            _json_string(graph_id),
            ',"manifest_index":',
            str(manifest_index),
            ',"op_index":',
            str(op_index),
            ',"op_type":"',
            GGML_OP_MUL_MAT,
            '","proof_eligible":true,"source_types":{"dst":',
            _json_string(dst_type),
            ',"src0":',
            _json_string(src0_type),
            ',"src1":',
            _json_string(src1_type),
            "}",
        ]
    )
    if has_names:
        body.extend(
            [
                ',"src0_name":',
                _json_string(src0_name),
            ]
        )
    body.extend(
        [
            ',"src0_shape":',
            src0_shape,
        ]
    )
    if has_names:
        body.extend(
            [
                ',"src1_name":',
                _json_string(src1_name),
            ]
        )
    body.extend(
        [
            ',"src1_shape":',
            src1_shape,
            ',"tensor_name":',
            _json_string(tensor_name),
            ',"version":1}',
        ]
    )
    entry_hash = hashlib.sha256(
        b"VERATHOS_GGML_OP_MANIFEST_ENTRY_V1" + "".join(body).encode("utf-8")
    ).hexdigest()
    order_key = (
        created_unix_ns,
        manifest_index,
        backend,
        device,
        graph_id,
        op_index,
        tensor_name,
        path.name,
    )
    dedupe_key = (
        created_unix_ns,
        manifest_index,
        graph_id,
        op_index,
        GGML_OP_MUL_MAT,
        tensor_name,
        src0_name,
        src1_name,
        dst_name,
        src0_shape_raw,
        src1_shape_raw,
        dst_shape_raw,
        dst_type,
        src0_type,
        src1_type,
        backend,
        device,
        True,
    )
    return created_unix_ns, order_key, dedupe_key, entry_hash


_MANIFEST_PARSE_LOCK = threading.Lock()
_MANIFEST_PARSE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_MANIFEST_PARSE_CACHE_MAX_FILES = 16


def _cached_manifest_rows(path: Path, tag: str, parse_line: Any) -> list[Any]:
    """Incrementally parse an append-only manifest file, sharing the result.

    Every proof commitment, slot-view rebuild, and manifest-root computation
    used to re-read and re-parse the WHOLE window manifest per request; with
    a sha256 per row that made coordinator-side receipt work scale with
    (concurrent requests x window rows) under the GIL. The runtime appends
    rows and rotates to a new token-suffixed file per window, so the file is
    append-only for its lifetime: parse only the new tail, under one lock,
    and let every concurrent caller share the accumulated items. A shrink
    (rotation reusing a name) resets the state; a partially written last
    line is left for the next call.

    ``parse_line`` maps a stripped non-empty line to an item or ``None``
    (skipped); it must not raise for the cache to stay coherent, so wrap
    fallible parsers.
    """

    try:
        stat_result = path.stat()
    except OSError:
        return []
    key = (str(path), str(tag))
    with _MANIFEST_PARSE_LOCK:
        state = _MANIFEST_PARSE_CACHE.get(key)
        rewritten_in_place = (
            state is not None
            and int(stat_result.st_size) == int(state["offset"])
            and int(stat_result.st_mtime_ns) != int(state["mtime_ns"])
        )
        if (
            state is None
            or int(stat_result.st_size) < int(state["offset"])
            or rewritten_in_place
        ):
            state = {"offset": 0, "mtime_ns": 0, "items": []}
            _MANIFEST_PARSE_CACHE[key] = state
            while len(_MANIFEST_PARSE_CACHE) > _MANIFEST_PARSE_CACHE_MAX_FILES:
                oldest = next(iter(_MANIFEST_PARSE_CACHE))
                if oldest == key:
                    break
                _MANIFEST_PARSE_CACHE.pop(oldest)
        if int(stat_result.st_size) == int(state["offset"]):
            return state["items"]
        try:
            with open(path, "rb") as handle:
                handle.seek(int(state["offset"]))
                chunk = handle.read()
        except OSError:
            return state["items"]
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return state["items"]
        consumed = chunk[: cut + 1]
        state["offset"] = int(state["offset"]) + cut + 1
        state["mtime_ns"] = int(stat_result.st_mtime_ns)
        for raw_line in consumed.decode("utf-8", errors="replace").splitlines():
            raw = raw_line.strip()
            if not raw:
                continue
            item = parse_line(raw)
            if item is not None:
                state["items"].append(item)
        return state["items"]


def ggml_op_manifest_summary_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
    end_unix_ns: int = 0,
) -> tuple[str, int]:
    """Return the op-manifest root/count without materializing manifest entries."""

    root = Path(trace_dir)
    rows: list[tuple[tuple[Any, ...], str]] = []
    seen: set[tuple[Any, ...]] = set()
    start_bound = (
        max(0, int(start_unix_ns) - GGML_TRACE_WINDOW_SLACK_NS)
        if start_unix_ns
        else 0
    )
    end_bound = int(end_unix_ns) + GGML_TRACE_WINDOW_SLACK_NS if end_unix_ns else 0
    manifest_paths = [
        *sorted(root.glob("manifest*.vmanifest")),
        *sorted(root.glob("manifest*.jsonl")),
    ]
    for path in manifest_paths:
        file_unix_ns = _manifest_file_unix_ns(path)
        if start_bound and file_unix_ns and file_unix_ns < start_bound:
            continue
        if end_bound and file_unix_ns and file_unix_ns > end_bound:
            continue

        if path.suffix == ".vmanifest":

            def _parse_hash_order(raw: str, _path: Path = path) -> Any:
                try:
                    return _compact_manifest_line_hash_and_order(raw, path=_path)
                except Exception:
                    return None

            parsed_rows = _cached_manifest_rows(path, "hash_order", _parse_hash_order)
        else:

            def _parse_json_hash_order(raw: str, _path: Path = path) -> Any:
                try:
                    data = json.loads(raw)
                    if not isinstance(data, Mapping):
                        return None
                    entry = GgmlOpManifestEntry.from_mapping(data, path=_path)
                    if (
                        not entry.proof_eligible
                        or not _is_provable_weight_op(entry.src0_shape)
                    ):
                        return None
                    order_key = (
                        int(entry.created_unix_ns),
                        int(entry.manifest_index),
                        str(entry.backend),
                        str(entry.device),
                        str(entry.graph_id),
                        int(entry.op_index),
                        str(entry.tensor_name),
                        _path.name,
                    )
                    return (
                        int(entry.created_unix_ns),
                        order_key,
                        _manifest_entry_dedupe_key(entry),
                        entry.entry_hash(),
                    )
                except Exception:
                    return None

            parsed_rows = _cached_manifest_rows(
                path, "hash_order", _parse_json_hash_order
            )

        for created_unix_ns, order_key, dedupe_key, entry_hash in parsed_rows:
            if start_unix_ns and created_unix_ns < start_unix_ns:
                continue
            if end_unix_ns and created_unix_ns > end_unix_ns:
                continue
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append((order_key, entry_hash))
    if not rows:
        return "", 0
    hashes = [entry_hash for _order_key, entry_hash in sorted(rows, key=lambda item: item[0])]
    return _manifest_root_from_hashes(hashes), len(hashes)


def _manifest_file_unix_ns(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("manifest-"):
        return 0
    raw = stem[len("manifest-"):]
    digits = []
    for char in raw:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return 0
    try:
        return int("".join(digits))
    except ValueError:
        return 0


def _manifest_entry_dedupe_key(entry: GgmlOpManifestEntry) -> tuple[Any, ...]:
    return (
        int(entry.created_unix_ns),
        int(entry.manifest_index),
        str(entry.graph_id),
        int(entry.op_index),
        str(entry.op_type),
        str(entry.tensor_name),
        str(entry.src0_name),
        str(entry.src1_name),
        str(entry.dst_name),
        tuple(int(item) for item in entry.src0_shape),
        tuple(int(item) for item in entry.src1_shape),
        tuple(int(item) for item in entry.dst_shape),
        tuple(sorted((str(k), str(v)) for k, v in entry.source_types.items())),
        str(entry.backend),
        str(entry.device),
        bool(entry.proof_eligible),
    )


def find_op_manifest_entries_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
    end_unix_ns: int = 0,
    file_token: str = "",
) -> list[GgmlOpManifestEntry]:
    """Load full GGML op manifest entries that belong to an inference window."""

    root = Path(trace_dir)
    entries: list[GgmlOpManifestEntry] = []
    seen: set[tuple[Any, ...]] = set()
    start_bound = (
        max(0, int(start_unix_ns) - GGML_TRACE_WINDOW_SLACK_NS)
        if start_unix_ns
        else 0
    )
    end_bound = int(end_unix_ns) + GGML_TRACE_WINDOW_SLACK_NS if end_unix_ns else 0
    manifest_paths = (
        [*sorted(root.glob(f"manifest-{file_token}.vmanifest"))]
        if file_token
        else [
            *sorted(root.glob("manifest*.vmanifest")),
            *sorted(root.glob("manifest*.jsonl")),
        ]
    )
    for path in manifest_paths:
        file_unix_ns = _manifest_file_unix_ns(path)
        if start_bound and file_unix_ns and file_unix_ns < start_bound:
            continue
        if end_bound and file_unix_ns and file_unix_ns > end_bound:
            continue
        def _parse_entry(raw: str, _path: Path = path) -> Any:
            try:
                if _path.suffix == ".vmanifest":
                    entry = GgmlOpManifestEntry.from_compact_line(raw, path=_path)
                else:
                    data = json.loads(raw)
                    if not isinstance(data, Mapping):
                        return None
                    entry = GgmlOpManifestEntry.from_mapping(data, path=_path)
            except Exception:
                return None
            return entry if entry.proof_eligible else None

        for entry in _cached_manifest_rows(path, "entries", _parse_entry):
            if start_unix_ns and entry.created_unix_ns < start_unix_ns:
                continue
            if end_unix_ns and entry.created_unix_ns > end_unix_ns:
                continue
            dedupe_key = _manifest_entry_dedupe_key(entry)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entries.append(entry)
    return _ordered_manifest_entries(entries)


def find_slot_view_template_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
    end_unix_ns: int = 0,
    file_token: str = "",
) -> list[dict[str, Any]]:
    """Load the model-invariant per-graph op template for slot views.

    Organic slot-view receipts do not need every request op row; they need the
    per-graph structural template. Some llama.cpp runs emit a first graph that
    is one op short of the decode graph, so compact v3 loading reads graph by
    graph until it sees a final-logit candidate, then uses that graph's unique
    intra-index rows. This keeps the hot path bounded without dropping the LM
    head from decode-audit slot views.
    """

    root = Path(trace_dir)
    start_bound = (
        max(0, int(start_unix_ns) - GGML_TRACE_WINDOW_SLACK_NS)
        if start_unix_ns
        else 0
    )
    end_bound = int(end_unix_ns) + GGML_TRACE_WINDOW_SLACK_NS if end_unix_ns else 0
    # The token-selected file comes FIRST, but it is not always usable: the
    # newest file at request start can be a closed certification-probe window
    # whose graphs are all prefill-shaped (observed: a 13-token probe
    # file pinned a single-device prefill-fallback template for the whole
    # serve). Every other manifest file, newest first, serves as fallback so
    # a decode-shaped template is found whenever ANY window recorded one.
    manifest_paths = (
        [
            *sorted(root.glob(f"manifest-{file_token}.vmanifest")),
            *sorted(
                (
                    p
                    for p in root.glob("manifest*.vmanifest")
                    if p.name != f"manifest-{file_token}.vmanifest"
                ),
                key=_manifest_file_unix_ns,
                reverse=True,
            ),
        ]
        if file_token
        else [
            *sorted(root.glob("manifest*.vmanifest")),
            *sorted(root.glob("manifest*.jsonl")),
        ]
    )
    fallback_template: list[dict[str, Any]] = []
    for path in manifest_paths:
        file_unix_ns = _manifest_file_unix_ns(path)
        if not file_token:
            if start_bound and file_unix_ns and file_unix_ns < start_bound:
                continue
            if end_bound and file_unix_ns and file_unix_ns > end_bound:
                continue
        if path.suffix != ".vmanifest":
            entries = find_op_manifest_entries_for_window(
                trace_dir,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
                file_token=file_token,
            )
            entries = [item for item in entries if item.graph_seq >= 0]
            return slot_view_template_from_manifest_entries(entries)
        entries_by_graph: dict[int, list[GgmlOpManifestEntry]] = {}
        seen_by_graph: dict[int, set[int]] = {}
        graph_order: list[int] = []
        decode_shaped_memo: dict[int, bool] = {}
        logit_memo: dict[int, bool] = {}
        scan_stopped_early = False

        def _finalize_graph(graph: int) -> None:
            items = entries_by_graph.get(graph, [])
            decode_shaped_memo[graph] = _slot_view_graph_is_decode_shaped(items)
            logit_memo[graph] = any(
                _slot_view_op_is_decode_candidate(item.to_commitment_body())
                for item in items
            )

        def _graph_device(graph: int) -> tuple[str, str]:
            items = entries_by_graph.get(graph, [])
            if not items:
                return ("", "")
            return (str(items[0].backend), str(items[0].device))

        def _scan_complete(incoming_device: tuple[str, str]) -> bool:
            # Stop once a DECODE-SHAPED graph with a final-logit op has been
            # seen AND the incoming graph's device has already contributed a
            # decode-shaped sub-graph. Architectures with asymmetric
            # prefill/decode graphs (deepseek4 CSA runs extra compression
            # GEMMs during prefill) place the same op at different intra
            # indexes per graph kind; every slot-view leaf is a decode-step
            # leaf, so the template MUST come from decode-shaped graphs. A
            # prefill graph also computes last-token logits, so the logit
            # test alone stopped the scan one graph too early. The
            # repeat-device condition matters on multi-GPU serves: one
            # forward is several device sub-graphs and the LOGIT sub-graph
            # need not come last (glm-5.2 live order: CUDA1, CUDA2,
            # CUDA3+logit, CUDA0), so the scan must run until the next
            # forward starts repeating devices or the template misses whole
            # layer spans.
            if not any(
                decode_shaped_memo.get(graph, False) and logit_memo.get(graph, False)
                for graph in graph_order
            ):
                return False
            return any(
                decode_shaped_memo.get(graph, False)
                and _graph_device(graph) == incoming_device
                for graph in graph_order
            )

        def _scan_once() -> bool:
            """One pass over the file head; True when the scan stopped at the
            repeat-device boundary (a complete forward group was seen)."""

            entries_by_graph.clear()
            seen_by_graph.clear()
            graph_order.clear()
            decode_shaped_memo.clear()
            logit_memo.clear()
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        entry = GgmlOpManifestEntry.from_compact_line(raw, path=path)
                    except Exception:
                        continue
                    if not entry.proof_eligible or entry.graph_seq < 0:
                        continue
                    if start_unix_ns and entry.created_unix_ns < start_unix_ns:
                        continue
                    if end_unix_ns and entry.created_unix_ns > end_unix_ns:
                        continue
                    graph_seq = int(entry.graph_seq)
                    if graph_seq not in entries_by_graph:
                        if graph_order:
                            _finalize_graph(graph_order[-1])
                            if _scan_complete(
                                (str(entry.backend), str(entry.device))
                            ):
                                return True
                            if len(graph_order) >= 64:
                                return True
                        graph_order.append(graph_seq)
                        entries_by_graph[graph_seq] = []
                        seen_by_graph[graph_seq] = set()
                    intra = int(entry.intra_graph_index)
                    if intra in seen_by_graph[graph_seq]:
                        continue
                    seen_by_graph[graph_seq].add(intra)
                    entries_by_graph[graph_seq].append(entry)
                if graph_order:
                    _finalize_graph(graph_order[-1])
            return False

        try:
            # A LIVE manifest flushes in batches: at the first request of a
            # fresh capture window the file can end mid-forward, and a
            # template built from that prefix misses whole device sub-graphs
            # (observed: a 4-GPU serve pinned a CUDA3-only template, so
            # every draw landed on the last 18 layers and the hard tier's
            # per-layer floor rejected honest proofs). Rescan while the file
            # is still growing; a stable size means the window is quiesced
            # and whatever it holds is the real graph structure.
            last_size = -1
            for _attempt in range(25):
                scan_stopped_early = _scan_once()
                if scan_stopped_early:
                    break
                try:
                    size = path.stat().st_size
                except OSError:
                    break
                if size == last_size:
                    break
                last_size = size
                time.sleep(0.2)
        except OSError:
            continue
        if entries_by_graph:
            order_index = {graph: idx for idx, graph in enumerate(graph_order)}

            decode_graphs = [
                graph
                for graph in graph_order
                if logit_memo.get(graph, False)
                and decode_shaped_memo.get(graph, False)
            ]
            if decode_graphs:
                # The smallest logit-bearing decode-shaped graph is the pure
                # decode step; larger decode-shaped graphs only appear for
                # very short prompts, where a prefill chunk passes the row
                # bound but still carries the extra prefill-only ops.
                best_graph = min(
                    decode_graphs,
                    key=lambda graph: (
                        len(entries_by_graph.get(graph, [])),
                        int(order_index[graph]),
                    ),
                )
                # A multi-GPU forward runs one sub-graph per device and the
                # logit sub-graph spans only the logit device's layers.
                # Merge the smallest decode-shaped sub-graph of every OTHER
                # device so the template - and with it the slot-view
                # challenge universe - covers the model's full layer span.
                # Single-device serves have no other devices and keep the
                # exact single-graph template.
                best_device = (
                    str(entries_by_graph[best_graph][0].backend),
                    str(entries_by_graph[best_graph][0].device),
                )
                per_device: dict[tuple[str, str], int] = {}
                for graph in graph_order:
                    if not decode_shaped_memo.get(graph, False):
                        continue
                    items = entries_by_graph.get(graph, [])
                    if not items:
                        continue
                    device = (str(items[0].backend), str(items[0].device))
                    if device == best_device:
                        continue
                    current = per_device.get(device)
                    if current is None or (
                        len(items),
                        int(order_index[graph]),
                    ) < (
                        len(entries_by_graph.get(current, [])),
                        int(order_index[current]),
                    ):
                        per_device[device] = graph
                merged = list(entries_by_graph.get(best_graph, []))
                for graph in sorted(
                    per_device.values(), key=lambda item: order_index[item]
                ):
                    merged.extend(entries_by_graph.get(graph, []))
                if merged:
                    return slot_view_template_from_manifest_entries(merged)
            else:
                best_graph = max(
                    graph_order,
                    key=lambda graph: (
                        bool(logit_memo.get(graph, False)),
                        len(entries_by_graph.get(graph, [])),
                        -int(order_index[graph]),
                    ),
                )
            entries = entries_by_graph.get(best_graph, [])
            if entries and not fallback_template:
                # Prefill-shaped file: keep as last resort and try the other
                # manifest files for a real decode-shaped template first.
                fallback_template = slot_view_template_from_manifest_entries(
                    entries
                )
    return fallback_template


@dataclass(frozen=True)
class VerifiedGgmlProof:
    receipt: LlamaGraphOpReceipt
    proof_mode: str
    verified: bool
    float_max_abs_error: float
    float_mean_abs_error: float
    proof_commitment_hash: str
    input_root: str
    weight_root: str
    output_root: str
    trace_path: str
    proof_payload: dict[str, Any]
    verifier_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_mode": self.proof_mode,
            "verified": self.verified,
            "proof_receipts": [self.receipt.to_dict()],
            "proof_payloads": [self.proof_payload],
            "float_max_abs_error": self.float_max_abs_error,
            "float_mean_abs_error": self.float_mean_abs_error,
            "proof_commitment_hash": self.proof_commitment_hash,
            "input_root": self.input_root,
            "weight_root": self.weight_root,
            "output_root": self.output_root,
            "trace_path": self.trace_path,
            "verifier_ms": self.verifier_ms,
        }


@dataclass(frozen=True)
class GgmlProofVerification:
    """Result of independently verifying a GGML proof payload."""

    verified: bool
    verifier_ms: float
    message: str = "verified"


def ggml_proof_payload_commitment_hash(payload: Mapping[str, Any]) -> str:
    """Commit to a verifier-checkable GGML proof payload.

    The commitment deliberately excludes fields that are either derived from the
    payload or produced by a local verifier run. A validator recomputes this
    before trusting the receipt's ``proof_commitment_hash``.
    """

    body = dict(payload)
    body.pop("proof_commitment_hash", None)
    body.pop("verification", None)
    body.pop("verifier_ms", None)
    return hashlib.sha256(
        b"VERATHOS_GGML_GEMM_PROOF_COMMITMENT_V2"
        + canonical_json_bytes(body)
    ).hexdigest()


def _payload_trace_commitment_hash(trace_meta: Mapping[str, Any]) -> str:
    required = (
        "created_unix_ns",
        "graph_id",
        "op_index",
        "tensor_name",
        "src0_shape",
        "src1_shape",
        "dst_shape",
        "src1_f32_sha256",
        "dst_f32_sha256",
        "source_types",
        "backend",
        "device",
    )
    missing = [field for field in required if field not in trace_meta]
    if missing:
        raise RuntimeError("proof payload trace missing commitment fields")
    body = {
        "version": 1,
        "created_unix_ns": int(trace_meta["created_unix_ns"]),
        "graph_id": str(trace_meta["graph_id"]),
        "op_index": int(trace_meta["op_index"]),
        "tensor_name": str(trace_meta["tensor_name"]),
        "src0_shape": [int(item) for item in trace_meta["src0_shape"]],
        "src1_shape": [int(item) for item in trace_meta["src1_shape"]],
        "dst_shape": [int(item) for item in trace_meta["dst_shape"]],
        "src1_f32_sha256": str(trace_meta["src1_f32_sha256"]),
        "dst_f32_sha256": str(trace_meta["dst_f32_sha256"]),
        "source_types": {
            str(k): str(v) for k, v in trace_meta.get("source_types", {}).items()
        },
        "backend": str(trace_meta["backend"]),
        "device": str(trace_meta["device"]),
    }
    if trace_meta.get("src0_f32_sha256"):
        body["src0_f32_sha256"] = str(trace_meta["src0_f32_sha256"])
    if trace_meta.get("src0_raw_sha256"):
        body["src0_raw_sha256"] = str(trace_meta["src0_raw_sha256"])
        body["src0_raw_type"] = str(trace_meta.get("src0_raw_type", ""))
    if trace_meta.get("src0_name") or trace_meta.get("src1_name") or trace_meta.get("dst_name"):
        body["src0_name"] = str(trace_meta.get("src0_name", ""))
        body["src1_name"] = str(trace_meta.get("src1_name", ""))
        body["dst_name"] = str(trace_meta.get("dst_name", ""))
    if int(trace_meta.get("manifest_index", -1)) >= 0:
        body["manifest_index"] = int(trace_meta["manifest_index"])
    if trace_meta.get("expert_index") is not None:
        body["expert_index"] = int(trace_meta["expert_index"])
    return hashlib.sha256(
        b"VERATHOS_GGML_TRACE_COMMITMENT_V1" + canonical_json_bytes(body)
    ).hexdigest()


def _attach_flat_spot_openings(
    proof: GEMMProof,
    *,
    matrix: torch.Tensor,
    merkle: FlatWeightMerkle,
    spot_attr: str,
    proof_attr: str,
) -> None:
    """Attach Merkle openings for existing spot checks in a flat int tensor."""

    flat = matrix.contiguous().view(-1)
    for block_proof in proof.block_proofs:
        spots = list(getattr(block_proof, spot_attr, []) or [])
        if not spots:
            setattr(block_proof, proof_attr, [])
            continue
        chunk_indexes = []
        for spot in spots:
            chunk_indexes.append(merkle.get_chunk_for_element(int(spot.row), int(spot.col)))
        unique_chunks = list(dict.fromkeys(chunk_indexes))
        batch = merkle.get_proof_batch(unique_chunks, W_tensor=flat)
        chunk_cache = {
            chunk_idx: (path, leaf_data)
            for chunk_idx, (path, leaf_data) in zip(unique_chunks, batch)
        }
        openings: list[SpotCheckWithProof] = []
        for spot, chunk_idx in zip(spots, chunk_indexes):
            path, leaf_data = chunk_cache[chunk_idx]
            openings.append(
                SpotCheckWithProof(
                    row=int(spot.row),
                    col=int(spot.col),
                    value=int(spot.value),
                    merkle_path=path,
                    leaf_data=leaf_data,
                )
            )
        setattr(block_proof, proof_attr, openings)


def _flat_chunk_range_multiproof(
    merkle: Any,
    *,
    first_chunk: int,
    last_chunk: int,
    flat: Any = None,
) -> dict[str, Any]:
    """Authentication material for a CONTIGUOUS run of flat chunks.

    Per-chunk paths are almost entirely redundant across a contiguous run:
    every interior sibling is another chunk the verifier already holds. Only
    the two frontier siblings per level are external, so this carries
    ~2*depth hashes instead of one full path per chunk (3.02 MB -> 1.3 KB of
    proof material for a 64-row band, which is what made the first live run
    time out the stage proof endpoint).
    """

    low_path, _low_data = merkle.get_proof(int(first_chunk), W_tensor=flat)
    high_path, _high_data = merkle.get_proof(int(last_chunk), W_tensor=flat)
    depth = max(len(low_path.siblings), len(high_path.siblings))
    left: list[str | None] = []
    right: list[str | None] = []
    for level in range(depth):
        low_sib = (
            low_path.siblings[level] if level < len(low_path.siblings) else None
        )
        high_sib = (
            high_path.siblings[level] if level < len(high_path.siblings) else None
        )
        # is_left=True means the sibling sits to the LEFT of the node, i.e.
        # it is the frontier the low end needs; the mirror holds for high.
        left.append(
            bytes(low_sib[0]).hex() if low_sib is not None and low_sib[1] else None
        )
        right.append(
            bytes(high_sib[0]).hex()
            if high_sib is not None and not high_sib[1]
            else None
        )
    return {"depth": depth, "left": left, "right": right}


def _input_band_openings(
    proof: GEMMProof,
    *,
    input_matrix: torch.Tensor,
    merkle: Any,
    block_size: int,
) -> list[dict[str, Any]]:
    """Return Merkle-authenticated X band bytes per challenged block.

    The sidecar's X row commitments were spot-bound only: the sampled
    positions had to agree, but the rest of the band was the prover's to
    choose. The band's flat element range is contiguous in the row-major X
    matrix, so the chunks of the SAME tree that ``input_root`` commits
    already cover it exactly. Shipping those chunk bytes plus one
    contiguous-range multiproof lets the verifier rebuild the band from
    authenticated bytes and recompute the sidecar's X commitments, which is
    the X analogue of the exact Y anchor.

    Payload tracks the band, not the matrix: a decode step challenges a
    single row (a few KB), and a full 64-row prefill band is the raw band
    bytes plus a couple of dozen hashes.
    """

    return _input_band_openings_for_blocks(
        input_matrix=input_matrix,
        merkle=merkle,
        blocks=[(int(bp.bi), int(bp.bj)) for bp in proof.block_proofs],
        block_size=block_size,
    )


def _input_band_openings_for_blocks(
    *,
    input_matrix: torch.Tensor,
    merkle: Any,
    blocks: list[tuple[int, int]],
    block_size: int,
) -> list[dict[str, Any]]:
    """X band openings for a challenged block list (hard and light share it).

    Uses only the block row index; there is no sumcheck cross-check on the X
    side (the band is Merkle-authenticated against input_root), so the light
    relation reuses this unchanged.
    """

    if input_matrix.dim() != 2:
        raise RuntimeError("GGML input proof tensor must be 2D")
    rows, cols = int(input_matrix.shape[0]), int(input_matrix.shape[1])
    flat = input_matrix.contiguous().view(-1)
    total = int(flat.numel())
    chunk_size = int(getattr(merkle, "chunk_size", 0) or 0)
    if chunk_size <= 0:
        raise RuntimeError("X Merkle tree has no usable chunk size")
    raw = flat.cpu().numpy().astype(np.int8, copy=False).tobytes()
    openings: list[dict[str, Any]] = []
    for bi, bj in blocks:
        bi = int(bi)
        row_start = bi * int(block_size)
        row_end = min(row_start + int(block_size), rows)
        if row_start >= row_end:
            raise RuntimeError("GGML proof block is outside input tensor bounds")
        first_chunk = (row_start * cols) // chunk_size
        last_chunk = ((row_end * cols) - 1) // chunk_size
        span_start = first_chunk * chunk_size
        span_end = min((last_chunk + 1) * chunk_size, total)
        openings.append(
            {
                "version": 2,
                "bi": bi,
                "bj": int(bj),
                "dtype": "int8",
                "shape": [row_end - row_start, cols],
                "row_start": row_start,
                "num_cols": cols,
                "chunk_size": chunk_size,
                "first_chunk": first_chunk,
                "last_chunk": last_chunk,
                "span_start": span_start,
                "span_data_b64": base64.b64encode(
                    raw[span_start:span_end]
                ).decode("ascii"),
                "range_proof": _flat_chunk_range_multiproof(
                    merkle,
                    first_chunk=first_chunk,
                    last_chunk=last_chunk,
                    flat=flat,
                ),
            }
        )
    return openings


def _authenticated_input_band(
    opening: Mapping[str, Any],
    *,
    root: bytes,
    input_shape: tuple[int, int],
    block_size: int,
) -> np.ndarray:
    """Rebuild one challenged X band from its Merkle-authenticated span.

    The span bytes are folded into the flat chunk tree with the supplied
    frontier siblings and the result must equal ``input_root`` (the same
    root the v1 engine binds into the transcript), so the returned band is
    the audited op's real input or this raises.
    """

    rows_total, cols = int(input_shape[0]), int(input_shape[1])
    if int(opening.get("version", 0)) != 2:
        raise RuntimeError("X band opening has an unsupported version")
    if str(opening.get("dtype", "")) != "int8":
        raise RuntimeError("X band opening must be int8")
    if int(opening.get("num_cols", cols)) != cols:
        raise RuntimeError("X band opening column count does not match the proof")
    chunk_size = int(opening.get("chunk_size", 0))
    if chunk_size <= 0:
        raise RuntimeError("X band opening has an invalid chunk size")
    bi = int(opening.get("bi", -1))
    row_start = bi * int(block_size)
    row_end = min(row_start + int(block_size), rows_total)
    if row_start < 0 or row_start >= row_end:
        raise RuntimeError("X band opening is outside the input tensor")
    if int(opening.get("row_start", row_start)) != row_start:
        raise RuntimeError("X band opening row_start does not match its block")

    first_chunk = int(opening.get("first_chunk", -1))
    last_chunk = int(opening.get("last_chunk", -1))
    span_start = int(opening.get("span_start", -1))
    if (
        first_chunk < 0
        or last_chunk < first_chunk
        or span_start != first_chunk * chunk_size
    ):
        raise RuntimeError("X band opening chunk range is inconsistent")
    band_start = row_start * cols
    band_end = row_end * cols
    if first_chunk * chunk_size > band_start or (
        last_chunk + 1
    ) * chunk_size < band_end:
        raise RuntimeError("X band chunk range does not cover the challenged band")

    span = base64.b64decode(str(opening.get("span_data_b64", "")), validate=True)
    expected_chunks = last_chunk - first_chunk + 1
    if not span or len(span) > expected_chunks * chunk_size:
        raise RuntimeError("X band span byte length is invalid")
    if len(span) < band_end - span_start:
        raise RuntimeError("X band span does not cover the challenged band")

    # Fold the span's chunk hashes up the flat tree with the frontier
    # siblings; anything else in the range is derived from the span itself.
    level_hashes = [
        hash_leaf(hash_flat_chunk(span[offset : offset + chunk_size]))
        for offset in range(0, len(span), chunk_size)
    ]
    range_proof = opening.get("range_proof")
    if not isinstance(range_proof, Mapping):
        raise RuntimeError("X band opening has no range proof")
    left = list(range_proof.get("left") or [])
    right = list(range_proof.get("right") or [])
    depth = int(range_proof.get("depth", 0))
    if depth != len(left) or depth != len(right) or depth > 64:
        raise RuntimeError("X band range proof is malformed")

    low, high = first_chunk, last_chunk
    for level in range(depth):
        nodes = list(level_hashes)
        index = low
        if index % 2 == 1:
            sibling = left[level]
            if sibling is None:
                raise RuntimeError("X band range proof is missing a left sibling")
            nodes.insert(0, bytes.fromhex(str(sibling)))
            index -= 1
        end_index = index + len(nodes) - 1
        if end_index % 2 == 0:
            sibling = right[level]
            if sibling is None:
                # Odd-dup: the last node of an odd level pairs with itself.
                nodes.append(nodes[-1])
            else:
                nodes.append(bytes.fromhex(str(sibling)))
        level_hashes = [
            hash_node(nodes[position], nodes[position + 1])
            for position in range(0, len(nodes), 2)
        ]
        low = index // 2
        high = (index + len(nodes) - 1) // 2
    if len(level_hashes) != 1 or level_hashes[0] != root:
        raise RuntimeError("X band range proof does not reproduce the input root")

    band = np.frombuffer(
        span[band_start - span_start : band_end - span_start], dtype=np.int8
    )
    return band.reshape(row_end - row_start, cols)


def _output_block_openings_for_blocks(
    *,
    output: torch.Tensor,
    blocks: list[tuple[int, int]],
    block_size: int,
    expected_leaf_hashes: list[bytes] | None = None,
) -> list[dict[str, Any]]:
    """Return verifier-facing Y block bytes for a challenged block list.

    Shared by the hard GEMM proof (which passes the sumcheck's block leaf
    hashes to cross-check) and the light relation (which has no sumcheck and
    passes ``expected_leaf_hashes=None``, emitting the computed leaf hash).
    The emitted opening is byte-identical either way, so one verifier accepts
    both tiers.
    """

    if output.dim() != 2:
        raise RuntimeError("GGML output proof tensor must be 2D")
    rows, cols = int(output.shape[0]), int(output.shape[1])
    openings: list[dict[str, Any]] = []
    for index, (bi, bj) in enumerate(blocks):
        bi = int(bi)
        bj = int(bj)
        row_start = bi * int(block_size)
        row_end = min(row_start + int(block_size), rows)
        col_start = bj * int(block_size)
        col_end = min(col_start + int(block_size), cols)
        if row_start >= row_end or col_start >= col_end:
            raise RuntimeError("GGML proof block is outside output tensor bounds")
        block = output[row_start:row_end, col_start:col_end]
        if block.is_cuda:
            block = block.cpu()
        block_bytes = (
            np.ascontiguousarray(block.numpy(), dtype="<i8")
            .tobytes(order="C")
        )
        leaf_hash = hash_leaf(hash_block(block_bytes))
        if (
            expected_leaf_hashes is not None
            and leaf_hash != expected_leaf_hashes[index]
        ):
            raise RuntimeError("GGML output block opening leaf hash mismatch")
        openings.append(
            {
                "version": 1,
                "bi": bi,
                "bj": bj,
                "dtype": "int64",
                "shape": [row_end - row_start, col_end - col_start],
                "leaf_hash": leaf_hash.hex(),
                "leaf_data": block_bytes.hex(),
            }
        )
    return openings


def _output_block_openings(
    proof: GEMMProof,
    *,
    output: torch.Tensor,
    block_size: int,
) -> list[dict[str, Any]]:
    """Return verifier-facing Y block bytes for each challenged proof block."""

    return _output_block_openings_for_blocks(
        output=output,
        blocks=[(int(bp.bi), int(bp.bj)) for bp in proof.block_proofs],
        block_size=block_size,
        expected_leaf_hashes=[bp.leaf_hash for bp in proof.block_proofs],
    )


def _light_transcript(
    trace: "GgmlMulMatTrace",
    receipt_context: Mapping[str, Any],
    op_manifest_membership: Mapping[str, Any] | None,
    slot_view_membership: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, str]:
    """Beacon-bound transcript for the light relation.

    Identical construction to the hard path so a light and a hard proof of
    the same op bind to the same context; only the openings differ.
    """
    effective_tensor_manifest_root = str(
        receipt_context.get("model_tensor_manifest_root", "")
    )
    source = dict(receipt_context)
    source["model_tensor_manifest_root"] = effective_tensor_manifest_root
    memberships = {
        key: value
        for key, value in (
            ("op_manifest_membership", op_manifest_membership),
            ("slot_view_membership", slot_view_membership),
        )
        if isinstance(value, Mapping) and bool(value)
    }
    transcript_context = _ggml_transcript_context(
        source,
        {"graph_id": trace.graph_id, "op_index": int(trace.op_index)},
        memberships,
    )
    transcript_label = _ggml_transcript_label(transcript_context)
    return transcript_context, transcript_label, effective_tensor_manifest_root


def _light_input_activation(trace: "GgmlMulMatTrace") -> np.ndarray:
    """Load only the input activation (src1) in ne0-fast [m, k] order.

    The x-only analogue of _manifest_trace_io_candidates: no output read, no
    four-way layout materialization. Light records trace_io_layout so the
    verifier interprets the band identically.
    """
    if len(trace.src0_shape) < 2 or len(trace.src1_shape) < 2:
        raise ValueError("trace shapes must have at least two dimensions")
    if any(dim != 1 for dim in trace.src1_shape[2:]):
        raise ValueError("batched src1 MUL_MAT traces are not supported yet")
    k = int(trace.src0_shape[0])
    m = int(trace.src1_shape[1])
    if int(trace.src1_shape[0]) != k:
        raise ValueError("src0/src1 inner dimensions do not match")
    src1 = np.fromfile(trace.src1_f32_path, dtype=np.float32)
    if src1.size != k * m:
        raise ValueError("trace activation binary size does not match metadata")
    return np.ascontiguousarray(src1.reshape(m, k))


def prove_ggml_light_trace(
    trace: "GgmlMulMatTrace",
    receipt_context: Mapping[str, Any],
    *,
    proof_block_size: int = 64,
    op_manifest_membership: Mapping[str, Any] | None = None,
    slot_view_membership: Mapping[str, Any] | None = None,
    decode_audit_positions: list[int] | None = None,
    decode_audit_token_ids: list[int] | None = None,
    decode_audit_top_k: int = 8,
    include_input_band: bool = False,
) -> dict[str, Any]:
    """Build a LIGHT proof payload: openings only, no weights, no sumcheck.

    Proves structural membership (the op belongs to the registered model's
    signed op manifest), activation chaining (the stage boundary roots), and,
    for the final projection, that the streamed tokens are in the top-k of the
    CAPTURED logits (decode-audit openings). It makes NO weight execution
    claim; that is the hard tier.

    This default set is O(1) in context and output length, so light stays
    milliseconds at any scale (measured: input-activation Merkle over a
    prefill grows to seconds at 32k tokens, and it binds an input a tier that
    proves no computation gains little from). ``include_input_band=True``
    additionally commits and opens the int8 input activation; it is off by
    default and intended only for small ops where the extra binding is cheap.
    Mirrors v3, whose light tier is output-focused and makes no execution
    claim.
    """

    transcript_context, transcript_label, tensor_manifest_root = (
        _light_transcript(
            trace,
            receipt_context,
            op_manifest_membership,
            slot_view_membership,
        )
    )
    block_size = max(1, int(proof_block_size))
    trace_io_layout = "ggml_ne0_fast"

    input_fields: dict[str, Any] = {}
    if include_input_band:
        x_f32 = _light_input_activation(trace)
        x_i8, _x_scale = _quantize_int8(x_f32)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_i8))
        if x_tensor.dim() != 2:
            raise RuntimeError("light proof input activation must be 2D")
        base_config = get_config()
        x_merkle = FlatWeightMerkle(
            x_tensor, base_config.w_merkle_chunk_size, store_raw=False
        )
        beacon = str(receipt_context.get("proof_beacon", ""))
        input_rows = int(x_tensor.shape[0])
        if int(transcript_context.get("version", 0)) == 2 and beacon:
            receipt_layer_index = _trace_layer_index(
                {"tensor_name": trace.tensor_name, "src0_name": trace.src0_name},
                layer_start=int(transcript_context["layer_start"]),
                layer_end=int(transcript_context["layer_end"]),
                model_total_layers=int(transcript_context["model_total_layers"]),
            )
            challenged = derive_ggml_output_block_challenge(
                beacon=beacon,
                transcript_context=transcript_context,
                layer_index=receipt_layer_index,
                output_shape=[input_rows, int(x_tensor.shape[1])],
                block_size=block_size,
            )
        else:
            challenged = (0, 0)
        input_fields = {
            "input_root": x_merkle.root.hex(),
            "input_shape": [int(x_tensor.shape[0]), int(x_tensor.shape[1])],
            "input_dtype": "int8",
            "input_chunk_size": int(x_merkle.chunk_size),
            "input_band_openings": _input_band_openings_for_blocks(
                input_matrix=x_tensor,
                merkle=x_merkle,
                blocks=[challenged],
                block_size=block_size,
            ),
        }

    decode_audit_openings: list[dict[str, Any]] = []
    if decode_audit_positions:
        decode_audit_openings = make_decode_audit_openings_for_trace(
            trace,
            decode_audit_positions=list(decode_audit_positions),
            decode_audit_token_ids=list(decode_audit_token_ids or []),
            decode_audit_top_k=int(decode_audit_top_k),
        )

    payload: dict[str, Any] = {
        "version": 1,
        "proof_mode": VERATHOS_GGML_LIGHT_PROOF_MODE,
        "model_package_hash": str(receipt_context.get("model_package_hash", "")),
        "model_tensor_manifest_root": tensor_manifest_root,
        "proof_block_size": block_size,
        "trace_io_layout": trace_io_layout,
        "transcript_context": transcript_context,
        "transcript_label_hex": transcript_label.hex(),
        **input_fields,
        "input_boundary_root": str(
            receipt_context.get("input_boundary_root", "") or ""
        ),
        "output_boundary_root": str(
            receipt_context.get("output_boundary_root", "") or ""
        ),
        "trace": {
            "graph_id": trace.graph_id,
            "op_index": int(trace.op_index),
            "tensor_name": trace.tensor_name,
            "src0_name": trace.src0_name,
            "src1_name": trace.src1_name,
            "dst_name": trace.dst_name,
            "src0_shape": list(trace.src0_shape),
            "src1_shape": list(trace.src1_shape),
            "dst_shape": list(trace.dst_shape),
            "source_types": dict(sorted(trace.source_types.items())),
            "backend": trace.backend,
            "device": trace.device,
            "manifest_index": int(trace.manifest_index),
            "intra_graph_index": int(getattr(trace, "intra_graph_index", -1) or -1),
        },
    }
    if op_manifest_membership:
        payload["op_manifest_membership"] = dict(op_manifest_membership)
    if slot_view_membership:
        payload["slot_view_membership"] = dict(slot_view_membership)
    if decode_audit_openings:
        payload["decode_audit_openings"] = decode_audit_openings
    payload["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(payload)
    return payload


# Placeholder digest for the light receipt's root fields: light opens no
# input/weight/output commitments, but LlamaGraphOpReceipt requires valid
# digests, and a recognizable constant keeps light receipts wire-compatible
# with every existing receipt consumer while staying unmistakably non-hard.
GGML_LIGHT_NO_ROOT = hashlib.sha256(b"VERATHOS_GGML_LIGHT_NO_ROOT_V1").hexdigest()

GGML_LIGHT_PROOF_KIND = "ggml_light_v1"


def prove_ggml_light_trace_verified(
    trace: "GgmlMulMatTrace",
    receipt_context: Mapping[str, Any],
    *,
    proof_block_size: int = 64,
    op_manifest_membership: Mapping[str, Any] | None = None,
    slot_view_membership: Mapping[str, Any] | None = None,
    decode_audit_positions: list[int] | None = None,
    decode_audit_token_ids: list[int] | None = None,
    decode_audit_top_k: int = 8,
    include_input_band: bool = False,
) -> VerifiedGgmlProof:
    """Assemble a light payload plus its per-stage signed-able receipt.

    Mirrors the hard prover's return shape so the worker's payload assembly,
    receipt sanitization/signing, and payload-receipt pairing all work
    unchanged for the light tier. The receipt binds the payload bytes via
    proof_commitment_hash; the root fields carry the light placeholder digest
    because light opens no execution commitments.
    """

    payload = prove_ggml_light_trace(
        trace,
        receipt_context,
        proof_block_size=proof_block_size,
        op_manifest_membership=op_manifest_membership,
        slot_view_membership=slot_view_membership,
        decode_audit_positions=decode_audit_positions,
        decode_audit_token_ids=decode_audit_token_ids,
        decode_audit_top_k=decode_audit_top_k,
        include_input_band=include_input_band,
    )
    transcript_context = payload["transcript_context"]
    receipt_layer_index = int(receipt_context.get("layer_start", 0))
    if int(transcript_context.get("version", 0)) == 2:
        receipt_layer_index = _trace_layer_index(
            {
                "tensor_name": trace.tensor_name,
                "src0_name": trace.src0_name,
            },
            layer_start=int(transcript_context["layer_start"]),
            layer_end=int(transcript_context["layer_end"]),
            model_total_layers=int(transcript_context["model_total_layers"]),
        )
    proof_commitment_hash = str(payload["proof_commitment_hash"])
    receipt = LlamaGraphOpReceipt(
        request_id=str(receipt_context["request_id"]),
        mesh_id=str(receipt_context["mesh_id"]),
        mesh_spec_hash=str(receipt_context["mesh_spec_hash"]),
        stage_assignment_hash=str(receipt_context["stage_assignment_hash"]),
        rpc_plan_hash=str(receipt_context.get("rpc_plan_hash", "")),
        model_package_hash=str(receipt_context.get("model_package_hash", "")),
        model_tensor_manifest_root=str(payload["model_tensor_manifest_root"]),
        uid=int(receipt_context["uid"]),
        hotkey=str(receipt_context["hotkey"]),
        endpoint=str(receipt_context["endpoint"]),
        stage_index=int(receipt_context["stage_index"]),
        layer_start=int(receipt_context["layer_start"]),
        layer_end=int(receipt_context["layer_end"]),
        request_hash=str(receipt_context["request_hash"]),
        response_hash=str(receipt_context["response_hash"]),
        graph_id=trace.graph_id,
        op_index=trace.op_index,
        op_type=GGML_OP_MUL_MAT,
        layer_index=receipt_layer_index,
        tensor_name=trace.tensor_name,
        input_root=GGML_LIGHT_NO_ROOT,
        weight_root=GGML_LIGHT_NO_ROOT,
        output_root=GGML_LIGHT_NO_ROOT,
        quantization="light_openings_v1",
        backend=trace.backend,
        device=trace.device,
        proof_kind=GGML_LIGHT_PROOF_KIND,
        proof_commitment_hash=proof_commitment_hash,
        input_boundary_root=str(
            receipt_context.get("input_boundary_root", "") or ""
        ),
        output_boundary_root=str(
            receipt_context.get("output_boundary_root", "") or ""
        ),
    )
    return VerifiedGgmlProof(
        receipt=receipt,
        proof_mode=VERATHOS_GGML_LIGHT_PROOF_MODE,
        verified=False,
        float_max_abs_error=0.0,
        float_mean_abs_error=0.0,
        proof_commitment_hash=proof_commitment_hash,
        input_root=GGML_LIGHT_NO_ROOT,
        weight_root=GGML_LIGHT_NO_ROOT,
        output_root=GGML_LIGHT_NO_ROOT,
        trace_path=str(trace.path),
        proof_payload=payload,
        verifier_ms=0.0,
    )


ORGANIC_LIGHT_CHALLENGE_KIND = "inline_every_request_v1"


def verify_mesh_proof_payloads_any_tier(
    payloads: list[Mapping[str, Any]],
    receipts: list[Mapping[str, Any]],
    *,
    mesh_receipt: Mapping[str, Any],
    completion_token_ids: list[int] | None = None,
    postcommit_light_ok: bool = False,
) -> GgmlProofVerification:
    """Route proof payloads to the hard or light verifier by proof_mode.

    Security rule enforced here, keyed on the already-signed challenge kind:
    a LIGHT payload is acceptable ONLY for the organic inline challenge
    (inline_every_request_v1), or on the postcommit lane when the CALLER
    passes ``postcommit_light_ok=True`` after recomputing the nonce-derived
    tier draw itself (never from miner-supplied data). Any other
    validator-nonce kind MUST carry a full hard proof. This is what keeps
    the tiering sound: a miner cannot serve a cheap light proof to a canary,
    because the hard tier draw or the validator's explicit hard demand
    forces the hard verifier here regardless of what the miner sent. Mixed
    light/hard payloads in one request are rejected.
    """

    started = time.perf_counter()
    modes = {str(p.get("proof_mode", "")) for p in payloads}
    challenge_kind = str(mesh_receipt.get("proof_challenge_kind", ""))
    if VERATHOS_GGML_LIGHT_PROOF_MODE in modes:
        if modes != {VERATHOS_GGML_LIGHT_PROOF_MODE}:
            return GgmlProofVerification(
                verified=False,
                verifier_ms=(time.perf_counter() - started) * 1000.0,
                message="mixed light and hard proof payloads are not allowed",
            )
        if challenge_kind != ORGANIC_LIGHT_CHALLENGE_KIND and not (
            postcommit_light_ok
            and challenge_kind == "validator_postcommit_v1"
        ):
            return GgmlProofVerification(
                verified=False,
                verifier_ms=(time.perf_counter() - started) * 1000.0,
                message=(
                    "light proof is not allowed for challenge kind "
                    f"{challenge_kind!r}; validator audits require a hard proof"
                ),
            )
        try:
            if receipts:
                paired = _pair_proof_payloads_and_receipts(payloads, receipts)
            else:
                # Receipt-less callers (unit contexts, single-stage self
                # checks) verify each payload against the mesh receipt, whose
                # context is the stage context in that shape.
                paired = [(payload, mesh_receipt) for payload in payloads]
            effective_completion_ids = completion_token_ids
            if effective_completion_ids is None:
                effective_completion_ids = [
                    int(item)
                    for item in mesh_receipt.get(
                        "decode_audit_completion_token_ids", []
                    )
                    or []
                ] or None
            for payload, receipt in paired:
                verify_ggml_light_payload(
                    payload,
                    receipt,
                    completion_token_ids=effective_completion_ids,
                )
                if isinstance(payload.get("slot_view_membership"), Mapping):
                    verify_slot_view_proof_payload(
                        payload, mesh_receipt=mesh_receipt
                    )
            # Anchor the membership roots the payloads opened against to the
            # commitment the mesh receipt carries; without this an emitter
            # could open against a fabricated manifest tree.
            manifest_scope = str(
                mesh_receipt.get("proof_op_manifest_scope", "") or ""
            )
            op_aggregate = str(mesh_receipt.get("proof_op_manifest_root", ""))
            if op_aggregate:
                if manifest_scope == SLOT_VIEW_SCOPE:
                    slot_domains = _proof_membership_domains(
                        paired,
                        membership_key="slot_view_membership",
                        root_key="slot_view_root",
                        count_key="slot_view_count",
                        index_key="slot_view_leaf_index",
                    )
                    if not slot_domains:
                        raise RuntimeError(
                            "light payload is missing slot view membership"
                        )
                    aggregate = mesh_op_manifest_aggregate_root(
                        [
                            {
                                "stage_index": stage,
                                "op_manifest_root": domain["root"],
                                "op_manifest_count": domain["count"],
                            }
                            for stage, domain in slot_domains.items()
                        ]
                    )
                else:
                    manifest_domains = _proof_membership_domains(
                        paired,
                        membership_key="op_manifest_membership",
                        root_key="op_manifest_root",
                        count_key="op_manifest_count",
                        index_key="op_manifest_leaf_index",
                    )
                    if not manifest_domains:
                        raise RuntimeError(
                            "light payload is missing op manifest membership"
                        )
                    aggregate = mesh_op_manifest_aggregate_root(
                        [
                            {
                                "stage_index": stage,
                                "op_manifest_root": domain["root"],
                                "op_manifest_count": domain["count"],
                            }
                            for stage, domain in manifest_domains.items()
                        ]
                    )
                if aggregate != op_aggregate:
                    raise RuntimeError(
                        "light membership aggregate does not match the "
                        "receipt op manifest root"
                    )
        except RuntimeError as exc:
            return GgmlProofVerification(
                verified=False,
                verifier_ms=(time.perf_counter() - started) * 1000.0,
                message=f"light proof verification failed: {exc}",
            )
        return GgmlProofVerification(
            verified=True,
            verifier_ms=(time.perf_counter() - started) * 1000.0,
            message="light proof verified",
        )
    return verify_ggml_gemm_proof_payloads(
        payloads, receipts, mesh_receipt=mesh_receipt
    )


def verify_ggml_light_payload(
    payload: Mapping[str, Any],
    receipt_context: Mapping[str, Any],
    *,
    completion_token_ids: list[int] | None = None,
) -> bool:
    """Verify a LIGHT payload. Raises on any inconsistency, else returns True.

    Checks exactly what light claims: the transcript binds to the same
    context the prover committed, the challenged input band authenticates
    against input_root, op-manifest membership authenticates against its
    signed root, the stage boundary roots match the receipt, and any
    decode-audit openings put the streamed tokens in the top-k of the
    committed logits. It does NOT check weight execution.
    """

    if str(payload.get("proof_mode", "")) != VERATHOS_GGML_LIGHT_PROOF_MODE:
        raise RuntimeError("light verify called on a non-light payload")

    trace_meta = payload.get("trace")
    if not isinstance(trace_meta, Mapping):
        raise RuntimeError("light payload is missing trace metadata")

    op_membership = payload.get("op_manifest_membership")
    slot_membership = payload.get("slot_view_membership")
    has_op = isinstance(op_membership, Mapping) and bool(op_membership)
    has_slot = isinstance(slot_membership, Mapping) and bool(slot_membership)
    if not has_op and not has_slot:
        raise RuntimeError(
            "light payload requires op manifest or slot view membership"
        )
    if has_op:
        _verify_payload_op_manifest_membership(payload, op_membership)
    # Slot-view membership is verified by the caller through
    # verify_slot_view_proof_payload, which needs the request-level mesh
    # receipt (beacon, decode positions); this function checks the
    # stage-level bindings only.

    # Recompute the transcript from the payload's own context and confirm the
    # committed label matches: a tampered context changes the label.
    context = payload.get("transcript_context")
    if not isinstance(context, Mapping):
        raise RuntimeError("light payload is missing transcript context")
    expected_label = _ggml_transcript_label(context).hex()
    if str(payload.get("transcript_label_hex", "")) != expected_label:
        raise RuntimeError("light transcript label does not match its context")

    # The transcript must describe THIS receipt's request and stage, not
    # merely be self-consistent: every identity field the receipt carries has
    # to match, otherwise a payload lifted from another request would pass.
    for key in (
        "request_id",
        "mesh_id",
        "mesh_spec_hash",
        "stage_assignment_hash",
        "rpc_plan_hash",
        "model_package_hash",
        "model_tensor_manifest_root",
        "request_hash",
        "response_hash",
        "uid",
        "hotkey",
        "endpoint",
        "stage_index",
        "layer_start",
        "layer_end",
    ):
        expected = receipt_context.get(key)
        if expected is None or expected == "":
            continue
        if key in context and str(context.get(key)) != str(expected):
            raise RuntimeError(
                f"light transcript {key} does not match receipt context"
            )

    # Input band is OPTIONAL (off by default). When present it must
    # authenticate against input_root; its absence is not a failure, because
    # light's default bindings are membership + boundary + decode-audit.
    openings = payload.get("input_band_openings")
    if openings:
        try:
            input_root = bytes.fromhex(str(payload.get("input_root", "")))
        except ValueError as exc:
            raise RuntimeError("light input_root must be hex") from exc
        if len(input_root) != 32:
            raise RuntimeError("light input_root must be 32 bytes")
        input_shape = [int(v) for v in payload.get("input_shape", [])]
        if len(input_shape) != 2:
            raise RuntimeError("light input_shape must be 2D")
        if not isinstance(openings, list):
            raise RuntimeError("light input_band_openings must be a list")
        block_size = int(payload.get("proof_block_size", 0) or 0)
        if block_size < 1:
            raise RuntimeError("light proof_block_size must be positive")
        for opening in openings:
            # Reconstructs the band from Merkle-authenticated bytes; raises if
            # the range multiproof does not verify against input_root.
            _authenticated_input_band(
                opening,
                root=input_root,
                input_shape=(input_shape[0], input_shape[1]),
                block_size=block_size,
            )

    # Stage boundary roots must equal the receipt's (chain binding).
    for key in ("input_boundary_root", "output_boundary_root"):
        if str(payload.get(key, "") or "") != str(
            receipt_context.get(key, "") or ""
        ):
            raise RuntimeError(f"light {key} does not match receipt context")

    # Decode-audit openings, when present, bind streamed tokens to the
    # committed logits exactly as the hard path checks them.
    decode_openings = payload.get("decode_audit_openings")
    if decode_openings:
        # Light checks the streamed token against the COMMITTED logits the
        # miner opened, not against sumcheck-proved logits (light proves no
        # execution). So: the token at each sampled position must equal the
        # streamed completion token AND be in the top-k of the committed
        # logits row for that position. Correctness of those logits is the
        # hard tier's job; light is the consistency layer backed by it.
        tokens = list(completion_token_ids or [])
        for opening in decode_openings:
            position = int(opening.get("position", -1))
            if position < 0 or position >= len(tokens):
                raise RuntimeError("light decode audit position out of range")
            token_id = int(tokens[position])
            if int(opening.get("token_id", -1)) != token_id:
                raise RuntimeError(
                    "light decode audit token id does not match completion"
                )
            top = [int(t) for t in opening.get("top_token_ids", [])]
            if not top or int(opening.get("argmax_token_id", -1)) != top[0]:
                raise RuntimeError("light decode audit top-k is malformed")
            if token_id not in top:
                raise RuntimeError(
                    "light decode audit streamed token is not in the committed "
                    f"top-{len(top)} at position {position}"
                )
    return True


def _verify_flat_spot_openings(
    *,
    root: bytes,
    spots: list[SpotCheckWithProof],
    num_cols: int,
    chunk_size: int,
    dtype: str,
    tensor_name: str,
) -> None:
    """Verify flat-chunk Merkle openings and spot values."""

    if num_cols <= 0:
        raise RuntimeError(f"{tensor_name} num_cols must be positive")
    if chunk_size <= 0:
        raise RuntimeError(f"{tensor_name} chunk_size must be positive")
    if dtype == "int8":
        bytes_per_elem = 1
        struct_fmt = "b"
    elif dtype == "int64":
        bytes_per_elem = 8
        struct_fmt = "q"
    else:
        raise RuntimeError(f"unsupported {tensor_name} proof dtype: {dtype}")

    for spot in spots:
        if not verify_flat_chunk_merkle_path(root, spot.leaf_data, spot.merkle_path):
            raise RuntimeError(
                f"{tensor_name} spot Merkle proof invalid at "
                f"({spot.row}, {spot.col})"
            )
        if len(spot.leaf_data) % bytes_per_elem:
            raise RuntimeError(f"{tensor_name} spot leaf has invalid byte length")
        values = struct.unpack(
            f"<{len(spot.leaf_data) // bytes_per_elem}{struct_fmt}",
            spot.leaf_data,
        )
        flat_idx = int(spot.row) * int(num_cols) + int(spot.col)
        chunk_start = int(spot.merkle_path.leaf_index) * int(chunk_size)
        offset = flat_idx - chunk_start
        if offset < 0 or offset >= len(values):
            raise RuntimeError(
                f"{tensor_name} spot offset out of leaf bounds at "
                f"({spot.row}, {spot.col})"
            )
        expected = mod_p(int(values[offset]))
        if int(spot.value) != expected:
            raise RuntimeError(
                f"{tensor_name} spot value mismatch at ({spot.row}, {spot.col})"
            )


def _spot_tuple(spot: SpotCheck | SpotCheckWithProof) -> tuple[int, int, int]:
    return int(spot.row), int(spot.col), int(spot.value)


def _verify_spot_opening_bindings(block_proof: GEMMBlockProof) -> None:
    """Ensure verifier-backed openings bind the exact bare spot-check claims."""

    x_spots = [_spot_tuple(spot) for spot in block_proof.spot_X]
    x_openings = [_spot_tuple(spot) for spot in block_proof.spot_X_with_proofs]
    if x_spots != x_openings:
        raise RuntimeError("X spot openings do not match proof spot claims")
    w_spots = [_spot_tuple(spot) for spot in block_proof.spot_W]
    w_openings = [_spot_tuple(spot) for spot in block_proof.spot_W_with_proofs]
    if w_spots != w_openings:
        raise RuntimeError("W spot openings do not match proof spot claims")


def _output_opening_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    openings = payload.get("output_block_openings")
    if not isinstance(openings, list) or not openings:
        raise RuntimeError("GGML proof payload is missing output block openings")
    mapped: dict[tuple[int, int], Mapping[str, Any]] = {}
    for opening in openings:
        if not isinstance(opening, Mapping):
            raise RuntimeError("GGML output block opening must be an object")
        key = (int(opening.get("bi", -1)), int(opening.get("bj", -1)))
        if key in mapped:
            raise RuntimeError("GGML output block opening duplicated")
        mapped[key] = opening
    return mapped


def _decode_audit_openings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    openings = payload.get("decode_audit_openings", [])
    if not openings:
        return []
    if not isinstance(openings, list):
        raise RuntimeError("GGML decode audit openings must be a list")
    return [item for item in openings if isinstance(item, Mapping)]


def _verify_decode_openings_against_output_block(
    payload: Mapping[str, Any],
    output_openings: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    output_shape: list[int],
) -> None:
    """Bind decode-audit logits to the GEMM proof's opened output row."""

    openings = _decode_audit_openings(payload)
    if not openings:
        return
    if len(output_shape) != 2 or int(output_shape[0]) != 1:
        raise RuntimeError("decode audit requires a single-row LM-head output")
    output_opening = output_openings.get((0, 0))
    if output_opening is None:
        raise RuntimeError("decode audit requires output block opening (0,0)")
    shape = [int(item) for item in output_opening.get("shape", [])]
    expected_shape = [1, int(output_shape[1])]
    if shape != expected_shape:
        raise RuntimeError("decode audit output opening must cover the full logits row")
    try:
        leaf_data = bytes.fromhex(str(output_opening.get("leaf_data", "")))
    except ValueError as exc:
        raise RuntimeError("decode audit output opening leaf_data must be hex") from exc
    expected_bytes = int(output_shape[1]) * 8
    if len(leaf_data) != expected_bytes:
        raise RuntimeError("decode audit output opening byte length mismatch")
    y_i64 = np.frombuffer(leaf_data, dtype="<i8").reshape(1, int(output_shape[1]))
    proved_bytes = logits_i32_to_bytes(y_i64.reshape(-1))
    proved_hash = hashlib.sha256(proved_bytes).hexdigest()
    for opening in openings:
        if str(opening.get("proved_logits_i32_sha256", "")) != proved_hash:
            raise RuntimeError("decode audit proved logits row does not match GEMM output")
        if int(opening.get("proved_logits_i32_nbytes", -1)) != len(proved_bytes):
            raise RuntimeError("decode audit proved logits byte length mismatch")
        if str(opening.get("proved_logits_i32_bytes_hex", "")) != proved_bytes.hex():
            raise RuntimeError("decode audit proved logits bytes mismatch")


def _verify_output_block_opening(
    block_proof: GEMMBlockProof,
    opening: Mapping[str, Any],
    *,
    output_shape: list[int],
    block_size: int,
    r_i: list[int],
    r_j: list[int],
) -> None:
    """Verify a challenged Y block and its sumcheck claim against the output root."""

    if int(opening.get("version", 0)) != 1:
        raise RuntimeError("unsupported GGML output block opening version")
    if str(opening.get("dtype", "")) != "int64":
        raise RuntimeError("GGML output block opening dtype must be int64")
    bi = int(block_proof.bi)
    bj = int(block_proof.bj)
    if int(opening.get("bi", -1)) != bi or int(opening.get("bj", -1)) != bj:
        raise RuntimeError("GGML output block opening index mismatch")

    output_rows, output_cols = int(output_shape[0]), int(output_shape[1])
    bs = max(1, int(block_size))
    row_start = bi * bs
    row_end = min(row_start + bs, output_rows)
    col_start = bj * bs
    col_end = min(col_start + bs, output_cols)
    rows = row_end - row_start
    cols = col_end - col_start
    if rows <= 0 or cols <= 0:
        raise RuntimeError("GGML output block opening is outside matrix bounds")
    shape = [int(item) for item in opening.get("shape", [])]
    if shape != [rows, cols]:
        raise RuntimeError("GGML output block opening shape mismatch")

    try:
        leaf_data = bytes.fromhex(str(opening.get("leaf_data", "")))
    except ValueError as exc:
        raise RuntimeError("GGML output block opening leaf_data must be hex") from exc
    if len(leaf_data) != rows * cols * 8:
        raise RuntimeError("GGML output block opening byte length mismatch")
    expected_leaf_hash = hash_leaf(hash_block(leaf_data))
    if expected_leaf_hash != block_proof.leaf_hash:
        raise RuntimeError("GGML output block opening leaf hash mismatch")
    if str(opening.get("leaf_hash", "")) not in ("", expected_leaf_hash.hex()):
        raise RuntimeError("GGML output block opening leaf_hash mismatch")

    y_block = np.frombuffer(leaf_data, dtype="<i8").reshape(rows, cols)
    y_field = tensor_to_field_vec(
        y_block,
        1 << int(block_proof.m_bits),
        1 << int(block_proof.n_bits),
    )
    claimed_sum = build_mle_from_matrix_fast(y_field).evaluate(r_i + r_j)
    if int(block_proof.sumcheck_proof.claimed_sum) != int(claimed_sum):
        raise RuntimeError("GGML output block opening claimed_sum mismatch")


def _verify_transcript_spot_positions(
    block_proof: GEMMBlockProof,
    *,
    transcript_label: bytes,
    output_root: bytes,
    input_shape: list[int],
    weight_shape: list[int],
    output_shape: list[int],
    output_opening: Mapping[str, Any],
    block_size: int,
    spot_checks: int,
) -> None:
    """Replay Fiat-Shamir transcript state and validate sampled spot positions."""

    m, k = int(input_shape[0]), int(input_shape[1])
    k2, n = int(weight_shape[0]), int(weight_shape[1])
    if k != k2:
        raise RuntimeError("GGML proof input/weight shape mismatch")
    bs = max(1, int(block_size))
    row_start = int(block_proof.bi) * bs
    row_end = min(row_start + bs, m)
    col_start = int(block_proof.bj) * bs
    col_end = min(col_start + bs, n)
    rows = row_end - row_start
    cols = col_end - col_start
    if rows <= 0 or cols <= 0:
        raise RuntimeError("GGML proof block is outside matrix bounds")

    transcript = Transcript(transcript_label)
    transcript.absorb_bytes(b"Y_root", output_root)
    block_transcript = transcript.fork(
        f"block_{block_proof.bi}_{block_proof.bj}".encode()
    )
    block_transcript.absorb(b"bi", int(block_proof.bi))
    block_transcript.absorb(b"bj", int(block_proof.bj))
    block_transcript.absorb_bytes(b"leaf_hash", block_proof.leaf_hash)
    r_i = block_transcript.squeeze_n(b"r_i", int(block_proof.m_bits))
    r_j = block_transcript.squeeze_n(b"r_j", int(block_proof.n_bits))
    valid, _final_claim, _challenges = sumcheck_verify(
        block_proof.sumcheck_proof,
        block_transcript,
    )
    if not valid:
        raise RuntimeError("GGML proof sumcheck transcript replay failed")
    _verify_output_block_opening(
        block_proof,
        output_opening,
        output_shape=output_shape,
        block_size=block_size,
        r_i=r_i,
        r_j=r_j,
    )

    expected_count = max(1, int(spot_checks))
    if len(block_proof.spot_X) != expected_count:
        raise RuntimeError("GGML proof X spot count mismatch")
    if len(block_proof.spot_W) != expected_count:
        raise RuntimeError("GGML proof W spot count mismatch")

    expected_x = []
    for index in range(expected_count):
        local_row = (
            block_transcript.squeeze(b"spot_X:row:" + str(index).encode()) % rows
        )
        local_col = (
            block_transcript.squeeze(b"spot_X:col:" + str(index).encode()) % k
        )
        expected_x.append((row_start + local_row, local_col))
    actual_x = [(int(spot.row), int(spot.col)) for spot in block_proof.spot_X]
    if actual_x != expected_x:
        raise RuntimeError("GGML proof X spot transcript position mismatch")

    expected_w = []
    for index in range(expected_count):
        local_row = (
            block_transcript.squeeze(b"spot_W:row:" + str(index).encode()) % k
        )
        local_col = (
            block_transcript.squeeze(b"spot_W:col:" + str(index).encode()) % cols
        )
        expected_w.append((local_row, col_start + local_col))
    actual_w = [(int(spot.row), int(spot.col)) for spot in block_proof.spot_W]
    if actual_w != expected_w:
        raise RuntimeError("GGML proof W spot transcript position mismatch")


def _verified_ggml_transcript_context(
    payload: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
    mesh_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute transcript context from authenticated receipts and payload roots.

    The serialized context is a declaration only.  Validator-facing calls bind
    every externally meaningful field to the stage receipt and signed mesh
    receipt before using the recomputed label for proof verification.
    """

    declared = payload.get("transcript_context")
    if not isinstance(declared, Mapping):
        raise RuntimeError("GGML proof payload is missing transcript context")
    trace_meta = payload.get("trace")
    if not isinstance(trace_meta, Mapping):
        raise RuntimeError("GGML proof payload is missing trace metadata")
    memberships = {
        key: value
        for key in (
            "trace_membership",
            "op_manifest_membership",
            "slot_view_membership",
        )
        if isinstance((value := payload.get(key)), Mapping) and bool(value)
    }

    if receipt is None:
        # Standalone/local verification still derives the label rather than
        # accepting transcript_label_hex.  Authenticated artifact verification
        # always supplies both receipts below.
        source: dict[str, Any] = dict(declared)
    else:
        source = dict(mesh_receipt or {})
        source.update(receipt)
        # Private worker receipts intentionally lack the opaque public ID.  It
        # is checked after coordinator conversion by validator-facing v2
        # receipts; during coordinator preflight retain the generated value.
        if "stage_id" not in source and declared.get("stage_id"):
            source["stage_id"] = declared.get("stage_id")
        for field in (
            "verification_snapshot_hash",
            "model_index",
            "model_total_layers",
            "proof_gate_hash",
        ):
            if field not in source and field in declared:
                source[field] = declared[field]

    secure_required = bool(
        (receipt or {}).get("stage_id")
        or (mesh_receipt or {}).get("verification_snapshot_hash")
    )
    if secure_required and mesh_receipt is not None:
        if str(mesh_receipt.get("proof_trace_manifest_format", "")) != (
            "compact-raw-v3"
        ):
            raise RuntimeError(
                "secure GGML proof requires compact-raw-v3 trace manifests"
            )
        if str(mesh_receipt.get("proof_trace_scope", "")) != (
            "op_manifest_challenge_v1"
        ):
            raise RuntimeError(
                "secure GGML proof requires the op-manifest challenge domain"
            )

    expected = _ggml_transcript_context(source, trace_meta, memberships)
    if dict(declared) != expected:
        for field, expected_value in expected.items():
            if declared.get(field) != expected_value:
                raise RuntimeError(f"GGML transcript {field} mismatch")
        raise RuntimeError("GGML transcript context does not match proof receipts")

    if secure_required and int(expected.get("version", 0)) != 2:
        raise RuntimeError("validator-facing GGML proof requires transcript context v2")
    if secure_required:
        if receipt is None or mesh_receipt is None:
            raise RuntimeError("secure GGML transcript requires stage and mesh receipts")
        for field in (
            "request_id",
            "mesh_id",
            "mesh_spec_hash",
            "stage_assignment_hash",
            "stage_index",
            "layer_start",
            "layer_end",
            "model_package_hash",
            "model_tensor_manifest_root",
            "request_hash",
            "response_hash",
            "graph_id",
            "op_index",
        ):
            if expected.get(field) != receipt.get(field):
                raise RuntimeError(f"GGML transcript {field} mismatch")
        for field in (
            "verification_snapshot_hash",
            "model_index",
            "model_total_layers",
            "proof_gate_hash",
        ):
            if expected.get(field) != mesh_receipt.get(field):
                raise RuntimeError(f"GGML transcript {field} mismatch")
        if expected.get("proof_policy_context_hash") != (
            ggml_proof_policy_context_hash(mesh_receipt)
        ):
            raise RuntimeError("GGML transcript proof policy context mismatch")
        for field in (
            "request_id",
            "mesh_id",
            "mesh_spec_hash",
            "stage_assignment_hash",
            "model_package_hash",
            "model_tensor_manifest_root",
            "request_hash",
            "response_hash",
        ):
            if mesh_receipt.get(field) != receipt.get(field):
                raise RuntimeError(f"GGML mesh/stage receipt {field} mismatch")
        if receipt.get("stage_id") and expected.get("stage_id") != receipt.get(
            "stage_id"
        ):
            raise RuntimeError("GGML transcript opaque stage_id mismatch")

        if str(mesh_receipt.get("proof_trace_manifest_format", "")) != (
            "compact-raw-v3"
        ):
            raise RuntimeError(
                "secure GGML proof requires compact-raw-v3 trace manifests"
            )
        if str(mesh_receipt.get("proof_trace_scope", "")) != (
            "op_manifest_challenge_v1"
        ):
            raise RuntimeError(
                "secure GGML proof requires the op-manifest challenge domain"
            )
        membership_scope = str(mesh_receipt.get("proof_op_manifest_scope", ""))
        if membership_scope == SLOT_VIEW_SCOPE:
            if "slot_view_membership" not in memberships:
                raise RuntimeError(
                    "secure GGML proof is missing slot-view membership"
                )
        elif "op_manifest_membership" not in memberships:
            raise RuntimeError("secure GGML proof is missing op-manifest membership")

        layer_index = _trace_layer_index(
            trace_meta,
            layer_start=int(expected["layer_start"]),
            layer_end=int(expected["layer_end"]),
            model_total_layers=int(expected["model_total_layers"]),
        )
        if int(receipt.get("layer_index", -1)) != layer_index:
            raise RuntimeError("GGML proof receipt layer ownership mismatch")

    computed_label = _ggml_transcript_label(expected)
    declared_label = str(payload.get("transcript_label_hex", ""))
    if declared_label and declared_label != computed_label.hex():
        raise RuntimeError("GGML proof declared transcript label mismatch")
    return expected


def _verify_anchor_row_bindings(
    payload: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
    input_shape: list[int],
    input_root: bytes,
    x_chunk_size: int,
) -> None:
    """Bind a reduced anchored GEMM payload to its pre-nonce commitments.

    Soundness chain: the origin receipt froze the anchor inventory digest
    before the validator nonce existed; the beacon (nonce-derived) picks
    the audited rows from BOTH stage commitments; every opened row is
    Merkle-bound to its commitment; and the proved input matrix must
    quantize and re-hash to the payload's own input_root, so the sumcheck
    ran over exactly the rows the serve committed.  The dst openings bind
    the runtime output rows the same way, giving the float gate committed
    material on both sides.
    """

    from verallm.mesh.anchor_audit import (
        anchor_commitment_from_dict,
        select_anchor_audit_rows_for_entry,
        verify_anchor_row_openings,
    )
    from verallm.mesh.execution_anchor import (
        execution_anchor_inventory_digest_v3,
    )

    if int(block.get("version", 0)) != 1:
        raise RuntimeError("unsupported anchor row openings version")
    if str(payload.get("trace_io_layout", "")) != GGML_ANCHORED_ROWS_LAYOUT:
        raise RuntimeError(
            "anchor row openings require the anchored trace layout"
        )
    trace_meta = payload.get("trace", {})
    if not isinstance(trace_meta, Mapping):
        raise RuntimeError("anchored payload is missing trace metadata")
    tensor_name = str(
        trace_meta.get("src0_name") or trace_meta.get("tensor_name", "")
    )
    sides: dict[str, Any] = {}
    for side in ("src1", "dst"):
        side_block = block.get(side)
        if not isinstance(side_block, Mapping):
            raise RuntimeError(f"anchored payload is missing the {side} side")
        commitment = anchor_commitment_from_dict(
            side_block.get("commitment", {})
        )
        if commitment.stage_id != f"{tensor_name}:{side}":
            raise RuntimeError(
                f"anchored {side} commitment belongs to a different stage"
            )
        sides[side] = (commitment, side_block)
    # Bind geometry to the op-manifest entry, which the origin receipt
    # Merkle-committed into proof_op_manifest_root BEFORE the nonce existed.
    # The membership itself is verified separately by
    # _verify_payload_op_manifest_membership, so by the time we read the
    # entry here it is already tied to the frozen root.
    op_membership = payload.get("op_manifest_membership")
    if not isinstance(op_membership, Mapping) or not op_membership:
        raise RuntimeError(
            "anchored proof requires op manifest membership to bind its "
            "row geometry to pre-nonce state"
        )
    entry_raw = op_membership.get("entry")
    if not isinstance(entry_raw, Mapping):
        raise RuntimeError("anchored proof op manifest entry is missing")
    entry = GgmlOpManifestEntry.from_mapping(entry_raw)
    entry_hash = entry.entry_hash()
    entry_rows = int(entry.src1_shape[1]) if len(entry.src1_shape) > 1 else 0
    if entry_rows <= 0:
        raise RuntimeError("anchored proof op manifest entry has no rows")
    entry_tensor = str(entry.src0_name or entry.tensor_name)
    if entry_tensor != tensor_name:
        raise RuntimeError(
            "anchored proof tensor does not match the op manifest entry"
        )
    for side in ("src1", "dst"):
        commitment = sides[side][0]
        # Without this the stream could span every execution of the tensor
        # in the window, letting the prover choose row_count by truncating
        # to exactly the rows it means to open.
        if int(commitment.row_count) != entry_rows:
            raise RuntimeError(
                f"anchored {side} stream has {int(commitment.row_count)} rows "
                f"but the frozen op manifest entry committed {entry_rows}"
            )

    beacon = _require_lower_hex_digest(
        payload.get("proof_beacon", ""), field_name="proof_beacon"
    )
    # Seeded ONLY from pre-nonce state plus the nonce, so re-running the
    # replay cannot move the draw. See select_anchor_audit_rows_for_entry.
    expected = select_anchor_audit_rows_for_entry(
        beacon=beacon,
        op_manifest_entry_hash=entry_hash,
        row_count=entry_rows,
    )

    # The inventory is optional: the mesh audit anchors during the post-nonce
    # replay, so no origin digest exists to compare against. When a receipt
    # DOES carry one (a future origin-anchored lane) keep enforcing equality.
    inventory_raw = block.get("inventory")
    receipt_digest = (
        str(receipt.get("proof_anchor_inventory_digest", ""))
        if receipt is not None
        else ""
    )
    if receipt_digest:
        if not isinstance(inventory_raw, list) or not inventory_raw:
            raise RuntimeError(
                "receipt committed an anchor inventory but the payload "
                "carries none"
            )
        inventory = tuple(
            anchor_commitment_from_dict(item) for item in inventory_raw
        )
        if execution_anchor_inventory_digest_v3(inventory).hex() != receipt_digest:
            raise RuntimeError(
                "anchored inventory does not match the frozen receipt digest"
            )
        committed = {item.canonical_bytes() for item in inventory}
        for side in ("src1", "dst"):
            if sides[side][0].canonical_bytes() not in committed:
                raise RuntimeError(
                    f"anchored {side} commitment is not in the frozen inventory"
                )
    declared = tuple(int(item) for item in block.get("row_indexes", []))
    if declared != expected:
        raise RuntimeError(
            "anchored row indexes do not match the beacon selection"
        )
    verified_rows: dict[str, dict[int, bytes]] = {}
    for side in ("src1", "dst"):
        commitment, side_block = sides[side]
        openings = side_block.get("openings")
        if not isinstance(openings, list):
            raise RuntimeError(f"anchored {side} openings must be a list")
        verified_rows[side] = verify_anchor_row_openings(
            beacon=beacon,
            commitment=commitment,
            openings=openings,
            expected_rows=expected,
        )
    x_f32 = np.ascontiguousarray(
        np.stack(
            [
                np.frombuffer(verified_rows["src1"][index], dtype="<f4")
                for index in expected
            ]
        )
    )
    if list(input_shape) != [int(x_f32.shape[0]), int(x_f32.shape[1])]:
        raise RuntimeError(
            "anchored input shape does not match the opened rows"
        )
    x_i8, x_scale = _quantize_int8(x_f32)
    if float(payload.get("x_scale", 0.0)) != float(x_scale):
        raise RuntimeError(
            "anchored input scale does not match the payload x_scale"
        )
    rebuilt = FlatWeightMerkle(
        torch.from_numpy(np.ascontiguousarray(x_i8)),
        int(x_chunk_size),
        store_raw=False,
    ).root
    if rebuilt != input_root:
        raise RuntimeError(
            "anchored rows do not rebuild the proved input root"
        )


def verify_ggml_gemm_proof_payload(
    payload: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None = None,
    mesh_receipt: Mapping[str, Any] | None = None,
) -> GgmlProofVerification:
    """Independently verify a verifier-facing GGML GEMM proof payload.

    This is the function a coordinator may run for early rejection and an
    external validator must run before accepting a sampled mesh proof.
    """

    started = time.perf_counter()
    try:
        if int(payload.get("version", 0)) != 1:
            raise RuntimeError("unsupported GGML proof payload version")
        proof_raw = payload.get("proof")
        if not isinstance(proof_raw, dict):
            raise RuntimeError("GGML proof payload is missing proof")
        proof = from_dict(GEMMProof, proof_raw)

        input_root = bytes.fromhex(str(payload["input_root"]))
        weight_root = bytes.fromhex(str(payload["weight_root"]))
        output_root = bytes.fromhex(str(payload["output_root"]))
        if proof.output_root != output_root:
            raise RuntimeError("proof output root does not match payload")

        trace_meta = payload.get("trace", {})
        if not isinstance(trace_meta, Mapping):
            trace_meta = {}

        if receipt is not None:
            expected_commitment = ggml_proof_payload_commitment_hash(payload)
            if str(receipt.get("proof_commitment_hash", "")) != expected_commitment:
                raise RuntimeError("proof payload commitment mismatch")
            for field in ("input_root", "weight_root", "output_root"):
                if str(receipt.get(field, "")) != str(payload.get(field, "")):
                    raise RuntimeError(f"proof payload {field} mismatch")
            if str(receipt.get("model_package_hash", "")):
                if str(receipt.get("model_package_hash", "")) != str(
                    payload.get("model_package_hash", "")
                ):
                    raise RuntimeError("proof payload model_package_hash mismatch")
            if isinstance(trace_meta, Mapping):
                for field in ("graph_id", "op_index", "tensor_name"):
                    if str(receipt.get(field, "")) != str(trace_meta.get(field, "")):
                        raise RuntimeError(f"proof payload trace {field} mismatch")
            if str(receipt.get("model_tensor_manifest_root", "")) != str(
                payload.get("model_tensor_manifest_root", "")
            ):
                raise mesh_binding_violation(
                    "proof payload model_tensor_manifest_root mismatch"
                )

        manifest_root = str(payload.get("model_tensor_manifest_root", ""))
        if manifest_root:
            tensor_membership = payload.get("gguf_tensor_membership")
            if not isinstance(tensor_membership, Mapping):
                raise RuntimeError("GGML proof payload is missing GGUF tensor membership")
            weight_source = str(payload.get("weight_source", "captured_trace"))
            bind_trace_weight = weight_source != "gguf_manifest"
            raw_hash = str(trace_meta.get("src0_raw_sha256", ""))
            f32_hash = str(trace_meta.get("src0_f32_sha256", ""))
            tensor_name = str(trace_meta.get("src0_name") or trace_meta.get("tensor_name", ""))
            raw_nbytes = (
                int(trace_meta["src0_raw_nbytes"])
                if str(trace_meta.get("src0_raw_nbytes", "")) != ""
                else None
            )
            f32_nbytes = (
                int(trace_meta["src0_f32_nbytes"])
                if str(trace_meta.get("src0_f32_nbytes", "")) != ""
                else None
            )
            if not verify_gguf_tensor_opening(
                tensor_membership,
                expected_root=manifest_root,
                expected_tensor_name=tensor_name,
                expected_raw_sha256=(raw_hash or None) if bind_trace_weight else None,
                expected_f32_sha256=(f32_hash or None) if bind_trace_weight else None,
                expected_proof_i8_merkle_root=str(payload.get("weight_root", "")),
                expected_proof_i8_chunk_size=int(payload.get("weight_chunk_size", 0)),
                expected_n_bytes=raw_nbytes if bind_trace_weight else None,
                expected_f32_n_bytes=f32_nbytes if bind_trace_weight else None,
                expected_expert_index=(
                    int(trace_meta["expert_index"])
                    if trace_meta.get("expert_index") is not None
                    else None
                ),
            ):
                raise RuntimeError("GGUF tensor membership proof failed")
            bias_membership = payload.get("gguf_bias_tensor_membership")
            if bias_membership:
                if not isinstance(bias_membership, Mapping):
                    raise RuntimeError("GGUF bias tensor membership must be an object")
                bias_name = str(payload.get("trace_output_bias_tensor_name", ""))
                if not bias_name or not verify_gguf_tensor_opening(
                    bias_membership,
                    expected_root=manifest_root,
                    expected_tensor_name=bias_name,
                ):
                    raise RuntimeError("GGUF bias tensor membership proof failed")

        membership = payload.get("trace_membership", {})
        op_membership = payload.get("op_manifest_membership", {})
        slot_membership = payload.get("slot_view_membership", {})
        if membership:
            if not _is_provable_weight_op(trace_meta.get("src0_shape", [])):
                raise RuntimeError("GGML trace membership is not proof-capable")
            trace_commitment_hash = str(membership.get("trace_commitment_hash", ""))
            if _payload_trace_commitment_hash(trace_meta) != trace_commitment_hash:
                raise RuntimeError("GGML trace commitment does not match proof payload")
            if not verify_trace_membership(
                trace_commitment_hash=trace_commitment_hash,
                trace_set_root=str(membership.get("trace_set_root", "")),
                trace_set_count=int(membership.get("trace_set_count", 0)),
                leaf_index=int(membership.get("trace_leaf_index", -1)),
                path=list(membership.get("trace_membership_path", [])),
            ):
                raise RuntimeError("GGML trace membership proof failed")
        elif receipt is not None and not op_membership and not slot_membership:
            raise RuntimeError("GGML proof payload is missing trace membership")

        if op_membership:
            if not isinstance(op_membership, Mapping):
                raise RuntimeError("GGML op manifest membership must be an object")
            _verify_payload_op_manifest_membership(payload, op_membership)

        input_shape = [int(item) for item in payload.get("input_shape", [])]
        weight_shape = [int(item) for item in payload.get("weight_shape", [])]
        output_shape = [int(item) for item in payload.get("output_shape", [])]
        if len(input_shape) != 2 or len(weight_shape) != 2 or len(output_shape) != 2:
            raise RuntimeError("GGML proof payload must include 2D input/weight/output shapes")
        if input_shape[1] != weight_shape[0]:
            raise RuntimeError("GGML proof input/weight shape mismatch")
        if output_shape != [input_shape[0], weight_shape[1]]:
            raise RuntimeError("GGML proof output shape mismatch")

        x_chunk_size = int(payload.get("input_chunk_size", 0))
        w_chunk_size = int(payload.get("weight_chunk_size", 0))
        x_dtype = str(payload.get("input_dtype", ""))
        w_dtype = str(payload.get("weight_dtype", ""))

        anchor_row_block = payload.get("anchor_row_openings")
        if anchor_row_block is not None:
            if not isinstance(anchor_row_block, Mapping):
                raise RuntimeError("anchor_row_openings must be an object")
            _verify_anchor_row_bindings(
                payload,
                anchor_row_block,
                receipt=receipt,
                input_shape=input_shape,
                input_root=input_root,
                x_chunk_size=x_chunk_size,
            )
        elif (
            str(payload.get("trace_io_layout", ""))
            == GGML_ANCHORED_ROWS_LAYOUT
        ):
            raise RuntimeError(
                "anchored layout payload is missing its row openings"
            )

        if not proof.block_proofs:
            raise RuntimeError("GGML proof is missing block proofs")
        proof_block_size_raw = payload.get("proof_block_size")
        proof_spot_checks_raw = payload.get("proof_spot_checks")
        if type(proof_block_size_raw) is not int:
            raise RuntimeError("GGML proof_block_size must be an integer")
        if type(proof_spot_checks_raw) is not int:
            raise RuntimeError("GGML proof_spot_checks must be an integer")
        proof_block_size = int(proof_block_size_raw)
        proof_spot_checks = int(proof_spot_checks_raw)
        if proof_block_size < GGML_CANONICAL_MIN_PROOF_BLOCK_SIZE:
            raise RuntimeError("GGML proof_block_size is below the canonical minimum")
        if proof_spot_checks < GGML_CANONICAL_MIN_SPOT_CHECKS:
            raise RuntimeError("GGML proof_spot_checks is below the canonical minimum")
        transcript_context = _verified_ggml_transcript_context(
            payload,
            receipt=receipt,
            mesh_receipt=mesh_receipt,
        )
        transcript_label = _ggml_transcript_label(transcript_context)
        if int(transcript_context.get("version", 0)) == 2:
            payload_beacon = _require_lower_hex_digest(
                payload.get("proof_beacon", ""),
                field_name="proof_beacon",
            )
            if mesh_receipt is not None and payload_beacon != str(
                mesh_receipt.get("proof_beacon", "")
            ):
                raise RuntimeError("GGML proof beacon mismatch")
            challenged_layer_index = _trace_layer_index(
                trace_meta,
                layer_start=int(transcript_context["layer_start"]),
                layer_end=int(transcript_context["layer_end"]),
                model_total_layers=int(
                    transcript_context["model_total_layers"]
                ),
            )
            expected_block = derive_ggml_output_block_challenge(
                beacon=payload_beacon,
                transcript_context=transcript_context,
                layer_index=challenged_layer_index,
                output_shape=output_shape,
                block_size=proof_block_size,
            )
            actual_blocks = [
                (int(item.bi), int(item.bj)) for item in proof.block_proofs
            ]
            if actual_blocks != [expected_block]:
                raise RuntimeError(
                    "GGML proof output block challenge mismatch"
                )

        from verallm.mesh.gemm_v2_sidecar import (
            GemmV2SidecarError,
            gemm_v2_available,
            gemm_v2_required,
            verify_gemm_v2_sidecar,
            verify_w_col_manifest_binding,
        )

        output_openings = _output_opening_map(payload)
        raw_input_band_openings = payload.get("input_band_openings")
        input_band_openings: dict[int, Mapping[str, Any]] = {}
        if isinstance(raw_input_band_openings, list):
            for item in raw_input_band_openings:
                if isinstance(item, Mapping):
                    input_band_openings[int(item.get("bi", -1))] = item

        def _anchor_x_band(bi: int) -> np.ndarray:
            # Exact X anchor, mandatory. The mesh has never shipped, so
            # there is no older worker to tolerate: a payload without the
            # band opening is simply not a valid mesh proof, and accepting
            # one would silently fall back to the spot-only X binding this
            # replaced.
            opening = input_band_openings.get(int(bi))
            if opening is None:
                raise RuntimeError(
                    "GGML proof payload is missing the X band opening"
                )
            return _authenticated_input_band(
                opening,
                root=input_root,
                input_shape=(int(input_shape[0]), int(input_shape[1])),
                block_size=proof_block_size,
            )

        def _anchor_y_block(bi: int, bj: int) -> np.ndarray:
            # The authenticated Y anchor for the sidecar: the same leaf_data
            # the block loop below verifies against leaf_hash and, through
            # the v1 engine, against output_root. Any tampering fails the
            # payload there, so these bytes are a sound anchor here.
            opening = output_openings.get((bi, bj))
            if opening is None:
                raise RuntimeError(
                    "GGML proof payload is missing output block opening"
                )
            bs = max(1, proof_block_size)
            rows = min(bs, int(output_shape[0]) - bi * bs)
            cols = min(bs, int(output_shape[1]) - bj * bs)
            try:
                leaf_data = bytes.fromhex(str(opening.get("leaf_data", "")))
            except ValueError as exc:
                raise RuntimeError(
                    "GGML output block opening leaf_data must be hex"
                ) from exc
            if rows <= 0 or cols <= 0 or len(leaf_data) != rows * cols * 8:
                raise RuntimeError(
                    "GGML output block opening byte length mismatch"
                )
            return np.frombuffer(leaf_data, dtype="<i8").reshape(rows, cols)

        gemm_v2_sidecar = payload.get("gemm_v2")
        # Mirror of the prover's decode-audit exemption: the LM head is
        # challenged as ONE full-row block whose Y is opened in full, which
        # cannot fit the per-column sidecar. Only a genuine final-projection
        # decode-audit payload qualifies — the tensor identity is bound via
        # trace/slot-view membership and the GGUF manifest, so a layer GEMM
        # cannot be dressed up as one to dodge the sidecar.
        _exemption_slot = payload.get("slot_view_membership", {})
        _exemption_leaf_ok = False
        if isinstance(_exemption_slot, Mapping) and _exemption_slot:
            _leaf_raw = _exemption_slot.get("leaf")
            _exemption_leaf_ok = isinstance(
                _leaf_raw, Mapping
            ) and _slot_view_op_is_decode_candidate(_leaf_raw)
        _exemption_name_ok = _is_final_projection_tensor_name(
            str(trace_meta.get("src0_name") or trace_meta.get("tensor_name", ""))
        )
        decode_audit_full_row_block = bool(
            _decode_audit_openings(payload)
            and (_exemption_leaf_ok or _exemption_name_ok)
            and len(output_shape) == 2
            and int(output_shape[0]) == 1
            and int(proof_block_size) >= int(output_shape[1])
        )
        if gemm_v2_sidecar is None:
            if gemm_v2_required() and not decode_audit_full_row_block:
                raise RuntimeError(
                    "gemm-v2 sidecar is required but the payload has none"
                )
        elif not gemm_v2_available():
            # A sidecar this verifier cannot check adds nothing but must not
            # fail an honest payload; every v1 obligation still applies.
            # Requiring the sidecar also requires being able to verify it.
            if gemm_v2_required():
                raise RuntimeError(
                    "VERATHOS_MESH_REQUIRE_GEMM_V2 is set but the native "
                    "PCS library is not available on this verifier"
                )
            logger.warning(
                "payload carries a gemm-v2 sidecar but the native PCS "
                "library is missing; sidecar left unverified"
            )
        else:
            if not isinstance(gemm_v2_sidecar, Mapping):
                raise RuntimeError("gemm-v2 sidecar must be an object")
            # For version-2 transcript contexts the block list was just
            # checked against the beacon derivation, so binding the sidecar
            # to the proof's own blocks is binding it to the challenge.
            sidecar_layer_index = (
                challenged_layer_index
                if int(transcript_context.get("version", 0)) == 2
                else int(transcript_context.get("layer_start", 0))
            )
            sidecar_tensor_name = str(
                trace_meta.get("src0_name") or trace_meta.get("tensor_name", "")
            )
            def _spot_signed(value: int) -> int:
                # v1 spot values live in F_{2^61-1}; recover the signed int8.
                p61 = (1 << 61) - 1
                value = int(value) % p61
                return value - p61 if value > p61 // 2 else value

            try:
                verify_gemm_v2_sidecar(
                    gemm_v2_sidecar,
                    transcript_label=transcript_label,
                    challenged_blocks=[
                        (int(item.bi), int(item.bj))
                        for item in proof.block_proofs
                    ],
                    block_size=proof_block_size,
                    tensor_name=sidecar_tensor_name,
                    layer_index=sidecar_layer_index,
                    op_index=int(trace_meta.get("op_index", 0)),
                    input_shape=(int(input_shape[0]), int(input_shape[1])),
                    weight_shape=(int(weight_shape[0]), int(weight_shape[1])),
                    # The spot positions are transcript-checked and their
                    # values Merkle-verified in the block loop below; any
                    # disagreement there fails the payload, so these tuples
                    # are trustworthy anchors for the sidecar binding.
                    spot_x_values=[
                        [
                            (int(sp.row), int(sp.col), _spot_signed(sp.value))
                            for sp in bp.spot_X
                        ]
                        for bp in proof.block_proofs
                    ],
                    spot_w_values=[
                        [
                            (int(sp.row), int(sp.col), _spot_signed(sp.value))
                            for sp in bp.spot_W
                        ]
                        for bp in proof.block_proofs
                    ],
                    y_block_values=[
                        _anchor_y_block(int(bp.bi), int(bp.bj))
                        for bp in proof.block_proofs
                    ],
                    x_band_values=[
                        _anchor_x_band(int(bp.bi))
                        for bp in proof.block_proofs
                    ],
                )
            except GemmV2SidecarError as exc:
                raise RuntimeError(f"gemm-v2 sidecar rejected: {exc}") from exc
            if manifest_root:
                w_col_membership = payload.get("gguf_tensor_membership")
                w_col_tensor_leaf = (
                    w_col_membership.get("tensor_leaf", {})
                    if isinstance(w_col_membership, Mapping)
                    else {}
                )
                if str(w_col_tensor_leaf.get("pcs_w_col_root", "") or ""):
                    # The signed manifest carries a per-column commitment
                    # root for this tensor, so the binding is REQUIRED: a
                    # miner cannot strip it without breaking the (already
                    # verified) tensor membership path.
                    try:
                        verify_w_col_manifest_binding(
                            gemm_v2_sidecar,
                            tensor_leaf=w_col_tensor_leaf,
                            w_col_manifest=payload.get(
                                "gemm_v2_w_col_manifest"
                            ),
                            challenged_blocks=[
                                (int(item.bi), int(item.bj))
                                for item in proof.block_proofs
                            ],
                            block_size=proof_block_size,
                            weight_shape=(
                                int(weight_shape[0]),
                                int(weight_shape[1]),
                            ),
                        )
                    except GemmV2SidecarError as exc:
                        # Name the exact op: the probe-overlap collision
                        # investigation needed this identity and it is
                        # generally the first question on any binding
                        # failure.
                        raise RuntimeError(
                            "gemm-v2 w-col manifest binding rejected for "
                            f"tensor={sidecar_tensor_name!r} "
                            f"layer={sidecar_layer_index} "
                            f"op_index={int(trace_meta.get('op_index', 0))} "
                            f"x={tuple(int(v) for v in input_shape)} "
                            f"w={tuple(int(v) for v in weight_shape)} "
                            f"blocks={[(int(b.bi), int(b.bj)) for b in proof.block_proofs]}: "
                            f"{exc}"
                        ) from exc
        _verify_decode_openings_against_output_block(
            payload,
            output_openings,
            output_shape=output_shape,
        )
        for block_proof in proof.block_proofs:
            output_opening = output_openings.get(
                (int(block_proof.bi), int(block_proof.bj))
            )
            if output_opening is None:
                raise RuntimeError("GGML proof payload is missing output block opening")
            if not block_proof.spot_X_with_proofs:
                raise RuntimeError("GGML proof is missing X Merkle spot openings")
            if not block_proof.spot_W_with_proofs:
                raise RuntimeError("GGML proof is missing W Merkle spot openings")
            _verify_spot_opening_bindings(block_proof)
            _verify_transcript_spot_positions(
                block_proof,
                transcript_label=transcript_label,
                output_root=output_root,
                input_shape=input_shape,
                weight_shape=weight_shape,
                output_shape=output_shape,
                output_opening=output_opening,
                block_size=proof_block_size,
                spot_checks=proof_spot_checks,
            )
            _verify_flat_spot_openings(
                root=input_root,
                spots=block_proof.spot_X_with_proofs,
                num_cols=input_shape[1],
                chunk_size=x_chunk_size,
                dtype=x_dtype,
                tensor_name="X",
            )

        config = replace(
            get_config(),
            block_size=proof_block_size,
            spot_checks=proof_spot_checks,
            parallel_proofs=False,
        )
        result = GEMMVerifier(config).verify(
            proof,
            input_root,
            weight_root,
            output_root,
            Transcript(transcript_label),
            spot_check_fn=lambda _spot, _matrix_id: True,
            W_merkle_root=weight_root,
            W_num_cols=weight_shape[1],
            w_chunk_size=w_chunk_size,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not result.passed:
            return GgmlProofVerification(False, elapsed_ms, result.message)
        return GgmlProofVerification(True, elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return GgmlProofVerification(False, elapsed_ms, str(exc))


def _pair_proof_payloads_and_receipts(
    payloads: list[Mapping[str, Any]],
    receipts: list[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pair payloads to graph receipts by unique, recomputed commitments."""

    if len(payloads) != len(receipts):
        raise RuntimeError("proof payload count does not match proof receipt count")
    payload_by_commitment: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        commitment = ggml_proof_payload_commitment_hash(payload)
        if str(payload.get("proof_commitment_hash", "")) != commitment:
            raise RuntimeError("proof payload declared commitment mismatch")
        if commitment in payload_by_commitment:
            raise RuntimeError("duplicate proof payload commitment")
        payload_by_commitment[commitment] = payload

    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen_receipts: set[str] = set()
    for receipt in receipts:
        commitment = str(receipt.get("proof_commitment_hash", ""))
        if not commitment:
            raise RuntimeError("proof receipt commitment is missing")
        if commitment in seen_receipts:
            raise RuntimeError("duplicate proof receipt commitment")
        seen_receipts.add(commitment)
        payload = payload_by_commitment.get(commitment)
        if payload is None:
            raise RuntimeError("missing proof payload for receipt commitment")
        receipt_stage = int(receipt.get("stage_index", -1))
        if receipt_stage < 0:
            raise RuntimeError("proof receipt stage index is missing")
        for membership_key in (
            "trace_membership",
            "op_manifest_membership",
            "slot_view_membership",
        ):
            membership = payload.get(membership_key, {})
            if not isinstance(membership, Mapping) or not membership:
                continue
            membership_stage = int(membership.get("stage_index", -1))
            if membership_stage != receipt_stage:
                raise RuntimeError(
                    f"{membership_key} stage does not match graph receipt stage"
                )
        paired.append((payload, receipt))
    if len(seen_receipts) != len(payload_by_commitment):
        raise RuntimeError("unpaired proof payload commitment")
    return paired


def _proof_membership_domains(
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    membership_key: str,
    root_key: str,
    count_key: str,
    index_key: str,
) -> dict[int, dict[str, Any]]:
    """Collect one consistent Merkle challenge domain for each mesh stage."""

    domains: dict[int, dict[str, Any]] = {}
    for payload, _receipt in paired:
        membership = payload.get(membership_key, {})
        if not isinstance(membership, Mapping) or not membership:
            continue
        stage_index = int(membership.get("stage_index", -1))
        root = str(membership.get(root_key, ""))
        count = int(membership.get(count_key, 0))
        leaf_index = int(membership.get(index_key, -1))
        if stage_index < 0 or len(root) != 64 or count <= 0:
            raise RuntimeError(f"invalid {membership_key} challenge domain")
        if leaf_index < 0 or leaf_index >= count:
            raise RuntimeError(f"invalid {membership_key} challenge index")
        domain = domains.setdefault(
            stage_index,
            {
                "root": root,
                "count": count,
                "indexes": set(),
                "payloads_by_index": {},
            },
        )
        if domain["root"] != root or int(domain["count"]) != count:
            raise RuntimeError(
                f"inconsistent {membership_key} root/count for mesh stage"
            )
        indexes = domain["indexes"]
        if leaf_index in indexes:
            raise RuntimeError(f"duplicate {membership_key} challenge index")
        indexes.add(leaf_index)
        domain["payloads_by_index"][leaf_index] = payload
    return domains


def _stage_layer_spans(
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[int, int]:
    """Return the layer count each stage receipt claims to own."""

    spans: dict[int, int] = {}
    for _payload, receipt in paired:
        stage_index = int(receipt.get("stage_index", -1))
        start = int(receipt.get("layer_start", 0))
        end = int(receipt.get("layer_end", 0))
        spans[stage_index] = max(0, end - start)
    return spans


def _require_challenge_universe_floor(
    *,
    mesh_receipt: Mapping[str, Any] | None,
    domains: dict[int, dict[str, Any]],
    spans: dict[int, int],
    scope_name: str,
) -> None:
    """Reject a challenge universe the miner shrank to steer sampling.

    The slot-view and op-manifest templates are built by the coordinator
    from its own op manifest, so a template committed with a single entry
    makes every Fiat-Shamir draw land on the same op and leaves every other
    GEMM in the stage permanently unchallengeable. The committed leaf count
    must therefore stay consistent with the layer span the stage claims.

    The per-layer floor is deliberately far below what any attention plus
    MLP block emits (separate or fused QKV, output projection, gate, up and
    down projections) so a legitimate architecture is never rejected.
    """

    if mesh_receipt is None:
        return
    tokens = int(mesh_receipt.get("completion_token_count", 0) or 0)
    if tokens <= 0:
        return
    for stage_index, domain in domains.items():
        layers = int(spans.get(stage_index, 0))
        if layers <= 0:
            continue
        minimum = tokens * layers * MIN_PROOF_OPS_PER_LAYER
        if int(domain["count"]) < minimum:
            raise mesh_binding_violation(
                f"{scope_name} challenge universe is too small for stage "
                f"{stage_index}: {int(domain['count'])} leaves for {layers} "
                f"layers over {tokens} tokens (minimum {minimum})"
            )


def _require_fiat_shamir_base_challenges(
    *,
    mesh_receipt: Mapping[str, Any],
    domains: dict[int, dict[str, Any]],
    scope_name: str,
    expected_stage_indexes: set[int],
) -> None:
    """Require every base FS draw; decode proofs may only add witnesses."""

    if not bool(mesh_receipt.get("proof_sampled", True)):
        return
    beacon = str(mesh_receipt.get("proof_beacon", ""))
    if not beacon:
        raise RuntimeError(f"{scope_name} verification needs the receipt proof beacon")
    if not domains:
        raise RuntimeError(f"proof payloads are missing {scope_name} membership")
    domain_stage_indexes = set(domains)
    missing_stages = sorted(expected_stage_indexes - domain_stage_indexes)
    extra_stages = sorted(domain_stage_indexes - expected_stage_indexes)
    if missing_stages:
        raise RuntimeError(
            f"{scope_name} is missing mesh stage domains: "
            + ",".join(str(item) for item in missing_stages)
        )
    if extra_stages:
        raise RuntimeError(
            f"{scope_name} contains unexpected mesh stage domains: "
            + ",".join(str(item) for item in extra_stages)
        )
    proof_limit = max(1, int(mesh_receipt.get("proof_ops_per_request") or 1))
    decode_required = bool(mesh_receipt.get("decode_audit_required", False))
    decode_stage = int(mesh_receipt.get("decode_audit_stage_index", -1))
    for stage_index, domain in domains.items():
        expected = set(
            select_manifest_challenge_indexes(
                beacon=beacon,
                op_manifest_root=str(domain["root"]),
                op_manifest_count=int(domain["count"]),
                proof_ops_per_request=proof_limit,
                stage_index=stage_index,
            )
            if scope_name in ("op manifest", "slot view")
            else select_trace_challenge_indexes(
                beacon=beacon,
                trace_set_root=str(domain["root"]),
                trace_set_count=int(domain["count"]),
                proof_ops_per_request=proof_limit,
                stage_index=stage_index,
            )
        )
        actual = set(domain["indexes"])
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(
                f"{scope_name} is missing Fiat-Shamir base challenge indexes: "
                + ",".join(str(item) for item in missing)
            )
        extras = actual - expected
        if not extras:
            continue
        if not decode_required or stage_index != decode_stage:
            raise RuntimeError(f"unexpected extra {scope_name} challenge indexes")
        for leaf_index in extras:
            payload = domain["payloads_by_index"][leaf_index]
            if not _decode_audit_openings(payload):
                raise RuntimeError(
                    f"extra {scope_name} challenge is not an additive decode proof"
                )


def _require_skipped_float_recompute_is_justified(
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Reject a proof that waives the float cross-check without earning it.

    The float recompute is the check that would catch a hidden state no
    transformer layer ever produced.  The prover is allowed to skip it on one
    path only: the final projection under decode audit, where the quantity
    being proved is the quantized logit vector rather than the trace's float
    dst, so a float comparison would be against a different quantity.

    That waiver used to be recorded and never read, which made it an
    unconditional opt-out for exactly the op where a fabricated input is most
    profitable.  It now has to be earned:

    - the op really is the final projection,
    - the payload really carries decode-audit openings, and
    - the stage committed the activations it was handed, whenever it was
      handed any, so the fabricated hidden state would have to be consistent
      with the boundary chain.

    The third condition is what ties the waiver to the stage boundary chain.
    It applies only to a stage that owns layers above the first, because that
    stage was necessarily handed activations by an upstream stage.  A stage
    starting at layer 0 has no upstream handoff to commit: either it is the
    whole model on one worker, or it is the head of the pipeline.
    """

    if not bool(payload.get("float_recompute_skipped", False)):
        return
    tensor_name = str(payload.get("tensor_name", "") or "")
    if not tensor_name:
        trace = payload.get("trace")
        if isinstance(trace, Mapping):
            tensor_name = str(trace.get("tensor_name", "") or "")
    if not _is_final_projection_tensor_name(tensor_name):
        raise mesh_binding_violation(
            "float recompute was skipped for a tensor that is not the final "
            f"projection: {tensor_name or 'unnamed'}"
        )
    if not _decode_audit_openings(payload):
        raise mesh_binding_violation(
            "float recompute was skipped without decode audit openings"
        )
    downstream_stage = int(receipt.get("layer_start", 0) or 0) > 0
    if (
        mesh_boundary_chain_required()
        and downstream_stage
        and not str(receipt.get("input_boundary_root", "") or "")
    ):
        raise mesh_binding_violation(
            "float recompute was skipped by a stage that committed no input "
            "activation boundary"
        )


def verify_ggml_gemm_proof_payloads(
    payloads: list[Mapping[str, Any]],
    receipts: list[Mapping[str, Any]],
    *,
    mesh_receipt: Mapping[str, Any] | None = None,
) -> GgmlProofVerification:
    """Verify proof payloads against their corresponding graph receipts."""

    started = time.perf_counter()
    try:
        paired = _pair_proof_payloads_and_receipts(payloads, receipts)
        receipt_stage_indexes = {
            int(receipt.get("stage_index", -1)) for _payload, receipt in paired
        }
        if -1 in receipt_stage_indexes:
            raise RuntimeError("proof receipt stage index is missing")
        for payload, receipt in paired:
            if mesh_receipt is not None and str(
                mesh_receipt.get("model_package_hash", "")
            ):
                expected_package = str(mesh_receipt.get("model_package_hash", ""))
                if str(receipt.get("model_package_hash", "")) != expected_package:
                    raise RuntimeError("proof receipt model_package_hash mismatch")
                if str(payload.get("model_package_hash", "")) != expected_package:
                    raise RuntimeError("proof payload model_package_hash mismatch")
            if mesh_receipt is not None and str(
                mesh_receipt.get("model_tensor_manifest_root", "")
            ):
                expected_root = str(mesh_receipt.get("model_tensor_manifest_root", ""))
                if str(receipt.get("model_tensor_manifest_root", "")) != expected_root:
                    raise mesh_binding_violation(
                        "proof receipt model_tensor_manifest_root mismatch"
                    )
                if str(payload.get("model_tensor_manifest_root", "")) != expected_root:
                    raise mesh_binding_violation(
                        "proof payload model_tensor_manifest_root mismatch"
                    )
            _require_skipped_float_recompute_is_justified(payload, receipt)
            result = verify_ggml_gemm_proof_payload(
                payload,
                receipt=receipt,
                mesh_receipt=mesh_receipt,
            )
            if not result.verified:
                raise RuntimeError(result.message)

        trace_domains = _proof_membership_domains(
            paired,
            membership_key="trace_membership",
            root_key="trace_set_root",
            count_key="trace_set_count",
            index_key="trace_leaf_index",
        )
        manifest_domains = _proof_membership_domains(
            paired,
            membership_key="op_manifest_membership",
            root_key="op_manifest_root",
            count_key="op_manifest_count",
            index_key="op_manifest_leaf_index",
        )
        slot_domains = _proof_membership_domains(
            paired,
            membership_key="slot_view_membership",
            root_key="slot_view_root",
            count_key="slot_view_count",
            index_key="slot_view_leaf_index",
        )

        trace_scope = (
            str(mesh_receipt.get("proof_trace_scope", "trace_candidate_set_v1"))
            if mesh_receipt is not None
            else "trace_candidate_set_v1"
        )
        manifest_scope = (
            str(mesh_receipt.get("proof_op_manifest_scope", "") or "")
            if mesh_receipt is not None
            else ""
        )
        if mesh_receipt is not None:
            trace_aggregate = str(
                mesh_receipt.get("proof_trace_commitment_root", "")
            )
            if trace_aggregate and trace_scope != "op_manifest_challenge_v1":
                if not trace_domains:
                    raise RuntimeError("proof payload is missing trace membership")
                aggregate = mesh_trace_commitment_aggregate_root(
                    [
                        {
                            "stage_index": stage,
                            "trace_set_root": domain["root"],
                            "trace_set_count": domain["count"],
                        }
                        for stage, domain in trace_domains.items()
                    ]
                )
                if aggregate != trace_aggregate:
                    raise RuntimeError("proof trace commitment aggregate mismatch")

            op_aggregate = str(mesh_receipt.get("proof_op_manifest_root", ""))
            if trace_scope == "op_manifest_challenge_v1" and not op_aggregate:
                raise RuntimeError(
                    "op-manifest challenge scope requires proof_op_manifest_root"
                )
            if op_aggregate and manifest_scope == SLOT_VIEW_SCOPE:
                if not slot_domains:
                    raise RuntimeError("proof payload is missing slot view membership")
                for payload, _receipt in paired:
                    verify_slot_view_proof_payload(payload, mesh_receipt=mesh_receipt)
                _require_challenge_universe_floor(
                    mesh_receipt=mesh_receipt,
                    domains=slot_domains,
                    spans=_stage_layer_spans(paired),
                    scope_name="slot view",
                )
                aggregate = mesh_op_manifest_aggregate_root(
                    [
                        {
                            "stage_index": stage,
                            "op_manifest_root": domain["root"],
                            "op_manifest_count": domain["count"],
                        }
                        for stage, domain in slot_domains.items()
                    ]
                )
                if aggregate != op_aggregate:
                    raise RuntimeError("slot view aggregate root mismatch")
            elif op_aggregate:
                if not manifest_domains:
                    raise RuntimeError("proof payload is missing op manifest membership")
                aggregate = mesh_op_manifest_aggregate_root(
                    [
                        {
                            "stage_index": stage,
                            "op_manifest_root": domain["root"],
                            "op_manifest_count": domain["count"],
                        }
                        for stage, domain in manifest_domains.items()
                    ]
                )
                if aggregate != op_aggregate:
                    raise RuntimeError("proof op manifest aggregate mismatch")

            strict_fs = any(
                key in mesh_receipt
                for key in ("proof_beacon", "proof_sampled", "proof_required")
            )
            if strict_fs:
                if manifest_scope == SLOT_VIEW_SCOPE:
                    _require_fiat_shamir_base_challenges(
                        mesh_receipt=mesh_receipt,
                        domains=slot_domains,
                        scope_name="slot view",
                        expected_stage_indexes=receipt_stage_indexes,
                    )
                elif trace_scope == "op_manifest_challenge_v1":
                    _require_fiat_shamir_base_challenges(
                        mesh_receipt=mesh_receipt,
                        domains=manifest_domains,
                        scope_name="op manifest",
                        expected_stage_indexes=receipt_stage_indexes,
                    )
                else:
                    _require_fiat_shamir_base_challenges(
                        mesh_receipt=mesh_receipt,
                        domains=trace_domains,
                        scope_name="trace",
                        expected_stage_indexes=receipt_stage_indexes,
                    )
        return GgmlProofVerification(
            True,
            (time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return GgmlProofVerification(
            False,
            (time.perf_counter() - started) * 1000.0,
            str(exc),
        )


def find_trace_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
    end_unix_ns: int = 0,
) -> GgmlMulMatTrace:
    """Find the smallest recent trace in an inference window."""

    candidates = find_traces_for_window(
        trace_dir,
        start_unix_ns=start_unix_ns,
        end_unix_ns=end_unix_ns,
    )
    if not candidates:
        raise RuntimeError("no GGML proof trace found for inference window")
    return candidates[0]


def read_boundary_roots_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
) -> dict[str, str] | None:
    """Read the stage's activation boundary roots captured for this window.

    ``boundary.jsonl`` is written by the patched RPC server and always holds
    the cumulative state of the current capture token, so the last line is
    the current window. A row older than the window start (minus the same
    slack the trace matcher uses) belongs to a previous request and is
    discarded rather than silently attributed.
    """

    path = Path(trace_dir) / "boundary.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    try:
        row = json.loads(lines[-1])
    except ValueError:
        return None
    if not isinstance(row, Mapping) or row.get("row") != "VERATHOS_GGML_BOUNDARY_V1":
        return None
    created = int(row.get("created_unix_ns", 0) or 0)
    if start_unix_ns and created < int(start_unix_ns) - GGML_TRACE_WINDOW_SLACK_NS:
        return None
    # A direction that absorbed zero bytes carries the hash of the empty
    # string; report it as absent so the verifier's open-chain-end rule
    # applies instead of chaining against a vacuous digest.
    input_root = str(row.get("input_boundary_root", "") or "")
    output_root = str(row.get("output_boundary_root", "") or "")
    if int(row.get("input_bytes", 0) or 0) <= 0:
        input_root = ""
    if int(row.get("output_bytes", 0) or 0) <= 0:
        output_root = ""
    if not input_root and not output_root:
        return None
    return {
        "input_boundary_root": input_root,
        "output_boundary_root": output_root,
    }


def find_traces_for_window(
    trace_dir: str | Path,
    *,
    start_unix_ns: int = 0,
    end_unix_ns: int = 0,
) -> list[GgmlMulMatTrace]:
    """Return proof candidates ordered by low proof cost, then recency."""

    root = Path(trace_dir)
    candidates: list[GgmlMulMatTrace] = []
    start_bound = (
        max(0, int(start_unix_ns) - GGML_TRACE_WINDOW_SLACK_NS)
        if start_unix_ns
        else 0
    )
    end_bound = int(end_unix_ns) + GGML_TRACE_WINDOW_SLACK_NS if end_unix_ns else 0
    for path in root.glob("*.json"):
        name_parts = path.stem.split("-")
        if len(name_parts) >= 2 and name_parts[0] == "trace":
            try:
                filename_unix_ns = int(name_parts[1])
            except ValueError:
                filename_unix_ns = 0
            if start_bound and filename_unix_ns and filename_unix_ns < start_bound:
                continue
            if end_bound and filename_unix_ns and filename_unix_ns > end_bound:
                continue
        try:
            trace = GgmlMulMatTrace.from_json(path)
        except Exception:
            continue
        if start_unix_ns and trace.created_unix_ns < start_unix_ns:
            continue
        if end_unix_ns and trace.created_unix_ns > end_unix_ns:
            continue
        candidates.append(trace)
    return sorted(candidates, key=lambda item: (item.total_elements, -item.created_unix_ns))


GGML_ANCHORED_ROWS_LAYOUT = "anchored_rows_v1"


def _prepare_anchored_rows(
    anchored_rows: Mapping[str, Any],
    *,
    tensor_name: str,
    k: int,
    n: int,
    beacon: str,
    op_manifest_entry_hash: str,
    entry_rows: int,
) -> dict[str, Any]:
    """Resolve, select, and load the bounded anchored witness for one op.

    Anchoring is armed for ONE selected op execution, so each stream holds
    exactly that execution's rows and its row count must equal the frozen
    op-manifest entry's own ``src1_shape[1]``. The audited rows derive from
    the beacon plus that frozen entry, never from the streams themselves,
    so a prover re-running the replay cannot move the draw. Fails closed
    when a stream, width, row count, or replayed row is missing: an
    anchored audit that cannot bind its rows must not silently fall back
    to a weaker shape.
    """

    from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_entry

    streams = anchored_rows.get("streams")
    row_dumps = anchored_rows.get("row_dumps")
    if not isinstance(streams, Mapping) or not isinstance(row_dumps, Mapping):
        raise RuntimeError(
            "anchored rows require streams and row_dumps mappings"
        )
    src1_stage = f"{tensor_name}:src1"
    dst_stage = f"{tensor_name}:dst"
    src1_stream = streams.get(src1_stage)
    dst_stream = streams.get(dst_stage)
    if src1_stream is None or dst_stream is None:
        raise RuntimeError(
            f"anchored audit is missing streams for {tensor_name}"
        )
    if int(src1_stream.row_width) != int(k) * 4:
        raise RuntimeError(
            "anchored src1 stream width does not match the trace K dim"
        )
    if int(dst_stream.row_width) != int(n) * 4:
        raise RuntimeError(
            "anchored dst stream width does not match the trace N dim"
        )
    for side, stream in (("src1", src1_stream), ("dst", dst_stream)):
        if int(stream.commitment.row_count) != int(entry_rows):
            raise RuntimeError(
                f"anchored {side} stream has "
                f"{int(stream.commitment.row_count)} rows but the op "
                f"manifest entry committed {int(entry_rows)}: the capture "
                "was not scoped to the selected op execution"
            )
    selected = select_anchor_audit_rows_for_entry(
        beacon=beacon,
        op_manifest_entry_hash=op_manifest_entry_hash,
        row_count=int(entry_rows),
    )
    matrices: dict[str, np.ndarray] = {}
    for stage_id, stream, width_elems in (
        (src1_stage, src1_stream, int(k)),
        (dst_stage, dst_stream, int(n)),
    ):
        stage_rows = row_dumps.get(stage_id)
        rows: list[np.ndarray] = []
        for index in selected:
            row_bytes = (
                stage_rows.get(index)
                if isinstance(stage_rows, Mapping)
                else None
            )
            if row_bytes is None:
                raise RuntimeError(
                    f"anchored audit is missing replayed row {index} for "
                    f"{stage_id}"
                )
            rows.append(np.frombuffer(row_bytes, dtype="<f4"))
        matrices[stage_id] = np.ascontiguousarray(
            np.stack(rows)
        ).reshape(len(selected), width_elems)
    return {
        "row_indexes": selected,
        "x_f32": matrices[src1_stage],
        "y_f32": matrices[dst_stage],
        "src1_stream": src1_stream,
        "dst_stream": dst_stream,
        "streams": streams,
        "row_dumps": row_dumps,
        "beacon": beacon,
    }


def _anchored_openings_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the payload block binding the reduced GEMM to the anchors."""

    from verallm.mesh.anchor_audit import (
        anchor_commitment_to_dict,
        build_anchor_row_openings,
    )

    selected = tuple(int(item) for item in state["row_indexes"])
    block: dict[str, Any] = {
        "version": 1,
        "row_indexes": list(selected),
        "inventory": [
            anchor_commitment_to_dict(state["streams"][stage_id].commitment)
            for stage_id in sorted(state["streams"])
        ],
    }
    for side, stream_key in (("src1", "src1_stream"), ("dst", "dst_stream")):
        stream = state[stream_key]
        rows = {
            index: state["row_dumps"][stream.stage_id][index]
            for index in selected
        }
        openings = build_anchor_row_openings(
            beacon=state["beacon"],
            stream=stream,
            rows=rows,
            expected_rows=selected,
        )
        block[side] = {
            "commitment": anchor_commitment_to_dict(stream.commitment),
            "openings": [item.to_dict() for item in openings],
        }
    return block


def prove_ggml_mul_mat_trace(
    trace: GgmlMulMatTrace,
    receipt_context: Mapping[str, Any],
    *,
    tolerance_abs: float = 8e-2,
    tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    spot_checks: int = 8,
    include_proof: bool = False,
    trace_membership: Mapping[str, Any] | None = None,
    op_manifest_membership: Mapping[str, Any] | None = None,
    slot_view_membership: Mapping[str, Any] | None = None,
    gguf_manifest: Mapping[str, Any] | None = None,
    decode_audit_positions: list[int] | None = None,
    decode_audit_token_ids: list[int] | None = None,
    decode_audit_top_k: int = 8,
    anchored_rows: Mapping[str, Any] | None = None,
    verify_before_return: bool = True,
) -> VerifiedGgmlProof:
    """Generate and verify a real Verathos GEMM proof for a GGML trace."""

    if anchored_rows is not None:
        if decode_audit_positions:
            raise RuntimeError(
                "anchored row audits and decode audits are separate lanes"
            )
        if gguf_manifest is None:
            raise RuntimeError(
                "anchored row audits require the GGUF manifest weight path"
            )

    # Manifest and slot memberships already bind the witness to their full
    # challenge domains.  A synthetic one-leaf trace tree adds no security and
    # can misleadingly look like independent trace-domain coverage.
    if (
        trace_membership is None
        and op_manifest_membership is None
        and slot_view_membership is None
    ):
        trace_membership = trace_membership_payload(
            [trace],
            0,
            stage_index=int(receipt_context.get("stage_index", 0)),
        )

    used_manifest_weight = False
    anchored_state: dict[str, Any] | None = None
    base_config = get_config()
    tensor_name = trace.src0_name or trace.tensor_name
    gguf_tensor_membership: dict[str, Any] | None = None
    gguf_bias_tensor_membership: dict[str, Any] | None = None
    trace_output_transform = "matmul"
    trace_output_bias_tensor_name = ""
    manifest_weight_root = ""
    manifest_weight_i8_sha256 = ""
    manifest_weight_chunk_size = int(base_config.w_merkle_chunk_size)
    float_recompute_skipped = False
    if gguf_manifest is not None:
        gguf_tensor_membership = gguf_tensor_opening(gguf_manifest, tensor_name)
        tensor_leaf = gguf_tensor_membership.get("tensor_leaf", {})
        if isinstance(tensor_leaf, Mapping):
            manifest_weight_root = str(tensor_leaf.get("proof_i8_merkle_root", ""))
            manifest_weight_i8_sha256 = str(tensor_leaf.get("proof_i8_sha256", ""))
            if str(tensor_leaf.get("proof_i8_chunk_size", "")) != "":
                manifest_weight_chunk_size = int(tensor_leaf["proof_i8_chunk_size"])
            # MoE expert witness: the committed artifact is THIS expert's
            # plane, not a whole-tensor matrix (3-D tensors have none).
            if trace.expert_index is not None:
                plane_shas = tensor_leaf.get("proof_i8_expert_sha256") or []
                plane_roots = tensor_leaf.get("proof_i8_expert_merkle_roots") or []
                if not (0 <= int(trace.expert_index) < len(plane_shas)):
                    raise RuntimeError(
                        f"expert witness {tensor_name!r} index {trace.expert_index} "
                        "not committed in the manifest (rebuild MoE manifests)"
                    )
                manifest_weight_i8_sha256 = str(plane_shas[int(trace.expert_index)])
                manifest_weight_root = str(plane_roots[int(trace.expert_index)])
    use_manifest_weight = bool(gguf_manifest is not None)
    max_abs = 0.0
    mean_abs = 0.0
    max_rel = 0.0
    trace_io_layout = "ggml_ne0_fast"
    float_exact_fallback_used = False
    if use_manifest_weight:
        if gguf_manifest is None:
            raise RuntimeError("GGUF manifest unavailable")
        if len(trace.src0_shape) < 2 or len(trace.src1_shape) < 2 or len(trace.dst_shape) < 2:
            raise ValueError("trace shapes must have at least two dimensions")
        if any(dim != 1 for dim in trace.src0_shape[2:]):
            raise ValueError("batched src0 MUL_MAT traces are not supported yet")
        if any(dim != 1 for dim in trace.src1_shape[2:]):
            raise ValueError("batched src1 MUL_MAT traces are not supported yet")
        if any(dim != 1 for dim in trace.dst_shape[2:]):
            raise ValueError("batched dst MUL_MAT traces are not supported yet")
        k = int(trace.src0_shape[0])
        n = int(trace.src0_shape[1])
        m = int(trace.src1_shape[1])
        if int(trace.src1_shape[0]) != k or int(trace.dst_shape[0]) != n:
            raise ValueError("trace dimensions do not match GGML MUL_MAT")
        if int(trace.dst_shape[1]) != m:
            raise ValueError("trace dst shape does not match GGML MUL_MAT")
        if anchored_rows is not None:
            anchored_beacon = str(receipt_context.get("proof_beacon", ""))
            if len(anchored_beacon) != 64:
                raise RuntimeError(
                    "anchored row audits require a v2 proof beacon"
                )
            # Row geometry and the row draw both come from the op-manifest
            # entry the origin receipt froze before the nonce existed, never
            # from the post-nonce streams.
            if not isinstance(op_manifest_membership, Mapping) or not (
                op_manifest_membership
            ):
                raise RuntimeError(
                    "anchored audits require op manifest membership"
                )
            anchored_entry_raw = op_manifest_membership.get("entry")
            if not isinstance(anchored_entry_raw, Mapping):
                raise RuntimeError(
                    "anchored audit op manifest entry is missing"
                )
            anchored_entry = GgmlOpManifestEntry.from_mapping(anchored_entry_raw)
            anchored_entry_rows = (
                int(anchored_entry.src1_shape[1])
                if len(anchored_entry.src1_shape) > 1
                else 0
            )
            anchored_state = _prepare_anchored_rows(
                anchored_rows,
                tensor_name=tensor_name,
                k=k,
                n=n,
                beacon=anchored_beacon,
                op_manifest_entry_hash=anchored_entry.entry_hash(),
                entry_rows=anchored_entry_rows,
            )
        skip_float_recompute = bool(decode_audit_positions) and _is_final_projection_tensor_name(
            tensor_name
        )
        if skip_float_recompute:
            candidates = _manifest_trace_io_candidates(trace)
            if not candidates:
                raise RuntimeError("decode audit final projection trace has no IO candidate")
            trace_io_layout, x_f32, y_f32 = candidates[0]
            trace_output_transform = "quantized_decode_audit"
            float_recompute_skipped = True
            if trace.expert_index is not None:
                w_i8, _plane_scale = proof_i8_expert_plane_from_manifest(
                    gguf_manifest, tensor_name, int(trace.expert_index)
                )
            else:
                w_i8 = proof_i8_weight_matrix_from_manifest(
                    gguf_manifest,
                    tensor_name,
                )
            if tuple(w_i8.shape) != (k, n):
                raise ValueError("GGUF proof weight shape does not match trace")
        else:
            if trace.expert_index is not None:
                # Reconstruct the float reference from the committed plane
                # (W ~= i8 * scale), mirroring the cached-i8 dense path.
                _plane_i8, _plane_scale = proof_i8_expert_plane_from_manifest(
                    gguf_manifest, tensor_name, int(trace.expert_index)
                )
                w_f32 = np.ascontiguousarray(
                    _plane_i8.astype(np.float32) * _plane_scale
                )
            else:
                w_f32 = proof_f32_weight_matrix_from_manifest(
                    gguf_manifest,
                    tensor_name,
                )
            if tuple(w_f32.shape) != (k, n):
                raise ValueError("GGUF f32 weight shape does not match trace")
            bias_f32: np.ndarray | None = None
            bias_name = ""
            if tensor_name.endswith(".weight"):
                candidate_bias_name = tensor_name[: -len(".weight")] + ".bias"
                try:
                    loaded_bias = f32_tensor_from_manifest(gguf_manifest, candidate_bias_name)
                    loaded_bias = np.asarray(loaded_bias, dtype=np.float32).reshape(-1)
                    if loaded_bias.size == n:
                        bias_f32 = np.ascontiguousarray(loaded_bias)
                        bias_name = candidate_bias_name
                except Exception:
                    bias_f32 = None
                    bias_name = ""

            def select_io_candidate(
                weight_f32: np.ndarray,
            ) -> tuple[
                tuple[
                    str,
                    np.ndarray,
                    np.ndarray,
                    float,
                    float,
                    float,
                    str,
                ]
                | None,
                list[tuple[str, float, float, float]],
            ]:
                candidate_errors: list[tuple[str, float, float, float]] = []
                if anchored_state is not None:
                    # Anchored rows ARE the runtime memory layout (the leaf
                    # bytes were hashed straight from the ne0-fast buffer),
                    # so there is no layout ambiguity and the float gate
                    # runs over every audited row directly.
                    candidate_x = anchored_state["x_f32"]
                    candidate_y = anchored_state["y_f32"]
                    xs_ref = _reference_matmul(candidate_x, weight_f32)
                    anchored_transforms: list[tuple[str, np.ndarray]] = [
                        ("matmul", xs_ref)
                    ]
                    if bias_f32 is not None:
                        anchored_transforms.append(
                            ("matmul_plus_bias", xs_ref + bias_f32)
                        )
                    for transform_name, y_reference in anchored_transforms:
                        diff = np.abs(y_reference - candidate_y)
                        cand_max_abs = float(np.max(diff)) if diff.size else 0.0
                        cand_mean_abs = float(np.mean(diff)) if diff.size else 0.0
                        y_scale = float(
                            max(
                                np.max(np.abs(candidate_y))
                                if candidate_y.size
                                else 0.0,
                                1.0,
                            )
                        )
                        cand_max_rel = float(cand_max_abs / y_scale)
                        candidate_name = (
                            GGML_ANCHORED_ROWS_LAYOUT
                            if transform_name == "matmul"
                            else f"{GGML_ANCHORED_ROWS_LAYOUT}+bias"
                        )
                        candidate_errors.append(
                            (
                                candidate_name,
                                cand_max_abs,
                                cand_mean_abs,
                                cand_max_rel,
                            )
                        )
                        if _float_recompute_ok(
                            max_abs=cand_max_abs,
                            mean_abs=cand_mean_abs,
                            max_rel=cand_max_rel,
                            y_f32=candidate_y,
                            tolerance_abs=tolerance_abs,
                            tolerance_rel=tolerance_rel,
                        ):
                            return (
                                (
                                    GGML_ANCHORED_ROWS_LAYOUT,
                                    candidate_x,
                                    candidate_y,
                                    cand_max_abs,
                                    cand_mean_abs,
                                    cand_max_rel,
                                    transform_name,
                                ),
                                candidate_errors,
                            )
                    return None, candidate_errors
                for layout_name, candidate_x, candidate_y in (
                    _manifest_trace_io_candidates(trace)
                ):
                    # Bounded-row recompute (see _float_recompute_error): the
                    # layout is a global property and the gate is heuristic, so
                    # a row sample disambiguates and tolerance-checks in O(1)
                    # rather than an O(context) matmul over every row. The
                    # returned candidate_x / candidate_y stay full for the
                    # commitment and sumcheck.
                    s = _recompute_row_slice(int(candidate_x.shape[0]))
                    xs = candidate_x[:s]
                    ys = candidate_y[:s]
                    xs_ref = _reference_matmul(xs, weight_f32)
                    transform_candidates: list[tuple[str, np.ndarray]] = [
                        ("matmul", xs_ref)
                    ]
                    if bias_f32 is not None:
                        transform_candidates.append(
                            ("matmul_plus_bias", xs_ref + bias_f32)
                        )
                    for transform_name, y_reference in transform_candidates:
                        diff = np.abs(y_reference - ys)
                        cand_max_abs = float(np.max(diff)) if diff.size else 0.0
                        cand_mean_abs = float(np.mean(diff)) if diff.size else 0.0
                        y_scale = float(
                            max(
                                np.max(np.abs(candidate_y))
                                if candidate_y.size
                                else 0.0,
                                1.0,
                            )
                        )
                        cand_max_rel = float(cand_max_abs / y_scale)
                        candidate_name = (
                            layout_name
                            if transform_name == "matmul"
                            else f"{layout_name}+bias"
                        )
                        candidate_errors.append(
                            (
                                candidate_name,
                                cand_max_abs,
                                cand_mean_abs,
                                cand_max_rel,
                            )
                        )
                        if _float_recompute_ok(
                            max_abs=cand_max_abs,
                            mean_abs=cand_mean_abs,
                            max_rel=cand_max_rel,
                            y_f32=candidate_y,
                            tolerance_abs=tolerance_abs,
                            tolerance_rel=tolerance_rel,
                        ):
                            return (
                                (
                                    layout_name,
                                    candidate_x,
                                    candidate_y,
                                    cand_max_abs,
                                    cand_mean_abs,
                                    cand_max_rel,
                                    transform_name,
                                ),
                                candidate_errors,
                            )
                return None, candidate_errors

            selected_io, candidate_errors = select_io_candidate(w_f32)
            exact_fallback_error = ""
            if selected_io is None and trace.expert_index is None:
                # The fast proof-i8 reconstruction is intentionally lossy.
                # It can occasionally reject a valid Q8_0 (or similarly
                # sensitive) execution even though the exact manifest-bound
                # GGUF weights reproduce the captured output. Retry from the
                # committed f32 artifact before considering any tolerance
                # relaxation. File-less members fetch that small/medium blob
                # from an authenticated peer and verify its SHA-256 locally.
                try:
                    exact_w_f32 = proof_f32_weight_matrix_from_manifest(
                        gguf_manifest,
                        tensor_name,
                        exact=True,
                    )
                    if tuple(exact_w_f32.shape) != (k, n):
                        raise ValueError(
                            "exact GGUF f32 weight shape does not match trace"
                        )
                    exact_selected, exact_errors = select_io_candidate(exact_w_f32)
                    if exact_selected is not None:
                        w_f32 = exact_w_f32
                        selected_io = exact_selected
                        candidate_errors = exact_errors
                        float_exact_fallback_used = True
                    else:
                        candidate_errors = [
                            (f"fast:{name}", err_abs, err_mean, err_rel)
                            for name, err_abs, err_mean, err_rel in candidate_errors
                        ] + [
                            (f"exact:{name}", err_abs, err_mean, err_rel)
                            for name, err_abs, err_mean, err_rel in exact_errors
                        ]
                except Exception as exc:
                    exact_fallback_error = str(exc)
            if selected_io is None:
                best = min(candidate_errors, key=lambda item: item[1]) if candidate_errors else None
                details = ", ".join(
                    f"{name}:max_abs={err_abs:.6f},max_rel={err_rel:.6f}"
                    for name, err_abs, _err_mean, err_rel in candidate_errors
                )
                raise RuntimeError(
                    "GGML MUL_MAT witness failed GGUF manifest float recomputation: "
                    f"tensor={tensor_name} backend={trace.backend} "
                    f"src0_shape={list(trace.src0_shape)} src1_shape={list(trace.src1_shape)} "
                    f"dst_shape={list(trace.dst_shape)} "
                    f"best_layout={best[0] if best else ''} "
                    f"max_abs_error={best[1] if best else 0.0:.6f}, "
                    f"max_rel_error={best[3] if best else 0.0:.6f}; "
                    f"layout_errors=[{details}]"
                    + (
                        f"; exact_fallback_error={exact_fallback_error}"
                        if exact_fallback_error
                        else ""
                    )
                )
            (
                trace_io_layout,
                x_f32,
                y_f32,
                max_abs,
                mean_abs,
                max_rel,
                trace_output_transform,
            ) = selected_io
            if trace_output_transform == "matmul_plus_bias":
                trace_output_bias_tensor_name = bias_name
                gguf_bias_tensor_membership = gguf_tensor_opening(
                    gguf_manifest,
                    bias_name,
                )
            # Ordinary layer tensors reuse the f32 matrix already loaded for output
            # binding. The final projection stays on the prewarmed proof-i8 cache to
            # avoid re-quantizing the huge LM head on every decode canary.
            if _is_final_projection_tensor_name(tensor_name):
                w_i8 = proof_i8_weight_matrix_from_manifest(
                    gguf_manifest,
                    tensor_name,
                )
            else:
                w_i8, _w_i8_scale = quantize_proof_i8(w_f32)
                if manifest_weight_i8_sha256:
                    actual_i8_hash = hashlib.sha256(w_i8.tobytes(order="C")).hexdigest()
                    if actual_i8_hash != manifest_weight_i8_sha256:
                        raise RuntimeError("GGUF proof weight i8 hash mismatch")
        x_i8, x_scale = _quantize_int8(x_f32)
        w_scale = 1.0
        used_manifest_weight = True
    else:
        if trace.src0_f32_path is None:
            raise ValueError("trace does not include src0_f32 witness bytes")
        x_f32, w_f32, y_f32 = trace.load_matrices()
        max_abs, mean_abs, max_rel = _float_recompute_error(x_f32, w_f32, y_f32)
        if not _float_recompute_ok(
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=max_rel,
            y_f32=y_f32,
            tolerance_abs=tolerance_abs,
            tolerance_rel=tolerance_rel,
        ):
            raise RuntimeError(
                "GGML MUL_MAT witness failed float recomputation: "
                f"tensor={tensor_name} backend={trace.backend} "
                f"src0_shape={list(trace.src0_shape)} src1_shape={list(trace.src1_shape)} "
                f"dst_shape={list(trace.dst_shape)} "
                f"max_abs_error={max_abs:.6f}, max_rel_error={max_rel:.6f}"
            )
        x_i8, x_scale = _quantize_int8(x_f32)
        w_i8, w_scale = _quantize_int8(w_f32)
    y_i64 = _proof_i8_matmul_i64(x_i8, w_i8)

    x_tensor = torch.from_numpy(np.ascontiguousarray(x_i8))
    w_tensor = torch.from_numpy(np.ascontiguousarray(w_i8))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_i64))

    effective_block_size = max(1, int(proof_block_size))
    if decode_audit_positions:
        if y_tensor.dim() != 2 or int(y_tensor.shape[0]) != 1:
            raise RuntimeError("decode audit requires a single-row LM-head GEMM trace")
        effective_block_size = max(effective_block_size, int(y_tensor.shape[1]))
    config = replace(
        base_config,
        block_size=effective_block_size,
        spot_checks=max(1, int(spot_checks)),
        parallel_proofs=False,
    )
    x_merkle = FlatWeightMerkle(x_tensor, config.w_merkle_chunk_size, store_raw=False)
    if used_manifest_weight and manifest_weight_root:
        w_merkle = _cached_gguf_weight_merkle(
            w_tensor,
            expected_root=manifest_weight_root,
            chunk_size=manifest_weight_chunk_size,
            i8_sha256=manifest_weight_i8_sha256,
            tensor_name=tensor_name,
        )
    else:
        w_merkle = FlatWeightMerkle(w_tensor, config.w_merkle_chunk_size, store_raw=False)
    y_merkle = build_block_merkle(y_tensor, config.block_size)
    expected_tensor_manifest_root = str(
        receipt_context.get("model_tensor_manifest_root", "")
    )
    if gguf_manifest is None and expected_tensor_manifest_root:
        raise RuntimeError(
            "GGUF tensor manifest binding requires gguf_manifest when "
            "model_tensor_manifest_root is set"
        )
    manifest_root = (
        str(gguf_manifest.get("tensor_manifest_root", ""))
        if gguf_manifest
        else ""
    )
    if expected_tensor_manifest_root and manifest_root != expected_tensor_manifest_root:
        raise RuntimeError("GGUF tensor manifest root does not match receipt context")
    effective_tensor_manifest_root = expected_tensor_manifest_root or manifest_root
    transcript_source = dict(receipt_context)
    transcript_source["model_tensor_manifest_root"] = effective_tensor_manifest_root
    transcript_context = _ggml_transcript_context(
        transcript_source,
        {"graph_id": trace.graph_id, "op_index": int(trace.op_index)},
        {
            key: value
            for key, value in (
                ("trace_membership", trace_membership),
                ("op_manifest_membership", op_manifest_membership),
                ("slot_view_membership", slot_view_membership),
            )
            if isinstance(value, Mapping) and bool(value)
        },
    )
    transcript_label = _ggml_transcript_label(transcript_context)
    receipt_layer_index = int(receipt_context.get("layer_start", 0))
    if int(transcript_context.get("version", 0)) == 2:
        receipt_layer_index = _trace_layer_index(
            {
                "tensor_name": trace.tensor_name,
                "src0_name": trace.src0_name,
            },
            layer_start=int(transcript_context["layer_start"]),
            layer_end=int(transcript_context["layer_end"]),
            model_total_layers=int(transcript_context["model_total_layers"]),
        )
        challenged_blocks = [
            derive_ggml_output_block_challenge(
                beacon=str(receipt_context.get("proof_beacon", "")),
                transcript_context=transcript_context,
                layer_index=receipt_layer_index,
                output_shape=list(y_tensor.shape),
                block_size=int(config.block_size),
            )
        ]
    else:
        challenged_blocks = [(0, 0)]
    transcript = Transcript(transcript_label)
    try:
        proof = GEMMProverFast(config).prove(
            x_tensor,
            w_tensor,
            y_tensor,
            challenged_blocks,
            transcript,
            W_merkle=w_merkle,
            Y_merkle=y_merkle,
        )
    except Exception:
        if not (used_manifest_weight and manifest_weight_root):
            raise
        # Intermittent live failure class: native_prove_block exploding on a
        # draw the same inputs served fine before and after ("cannot create
        # std::vector larger than max_size()"). Every cached layer that fed W
        # is dropped, W is re-derived and re-verified against the committed
        # manifest hashes, and the proof retries ONCE from a fresh
        # transcript. A second failure raises the enriched native
        # diagnostics; an honest miner no longer 500s a receipt over a
        # poisoned or torn cache entry.
        logger.exception(
            "hard-tier prover failed for %s (shapes x=%s w=%s y=%s, block=%s);"
            " dropping weight caches and retrying once",
            tensor_name,
            tuple(x_tensor.shape),
            tuple(w_tensor.shape),
            tuple(y_tensor.shape),
            int(config.block_size),
        )
        from verallm.mesh.gguf_manifest import (
            drop_cached_proof_i8,
            drop_cached_weight_merkle,
        )

        lru_key = (
            manifest_weight_root,
            int(manifest_weight_chunk_size),
            (int(w_tensor.shape[0]), int(w_tensor.shape[1])),
            str(w_tensor.dtype),
        )
        with _GGUF_WEIGHT_MERKLE_CACHE_LOCK:
            _GGUF_WEIGHT_MERKLE_CACHE.pop(lru_key, None)
        drop_cached_weight_merkle(manifest_weight_i8_sha256)
        if _is_final_projection_tensor_name(tensor_name):
            drop_cached_proof_i8({"proof_i8_sha256": manifest_weight_i8_sha256})
            w_i8 = proof_i8_weight_matrix_from_manifest(
                gguf_manifest,
                tensor_name,
            )
        else:
            w_i8, _w_i8_scale = quantize_proof_i8(w_f32)
        if manifest_weight_i8_sha256:
            rebuilt_sha = hashlib.sha256(w_i8.tobytes(order="C")).hexdigest()
            if rebuilt_sha != manifest_weight_i8_sha256:
                raise RuntimeError(
                    "GGUF proof weight i8 hash mismatch after prover-retry "
                    f"rebuild for {tensor_name}"
                )
        y_i64 = _proof_i8_matmul_i64(x_i8, w_i8)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_i8))
        w_tensor = torch.from_numpy(np.ascontiguousarray(w_i8))
        y_tensor = torch.from_numpy(np.ascontiguousarray(y_i64))
        w_merkle = _cached_gguf_weight_merkle(
            w_tensor,
            expected_root=manifest_weight_root,
            chunk_size=manifest_weight_chunk_size,
            i8_sha256=manifest_weight_i8_sha256,
            tensor_name=tensor_name,
        )
        x_merkle = FlatWeightMerkle(
            x_tensor, config.w_merkle_chunk_size, store_raw=False
        )
        y_merkle = build_block_merkle(y_tensor, config.block_size)
        transcript = Transcript(transcript_label)
        proof = GEMMProverFast(config).prove(
            x_tensor,
            w_tensor,
            y_tensor,
            challenged_blocks,
            transcript,
            W_merkle=w_merkle,
            Y_merkle=y_merkle,
        )
        logger.warning(
            "hard-tier prover retry succeeded for %s after cache rebuild",
            tensor_name,
        )
    _attach_flat_spot_openings(
        proof,
        matrix=x_tensor,
        merkle=x_merkle,
        spot_attr="spot_X",
        proof_attr="spot_X_with_proofs",
    )

    input_root = x_merkle.root.hex()
    weight_root = w_merkle.root.hex()
    output_root = proof.output_root.hex()

    proof_payload = {
        "version": 1,
        "proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE,
        "proof": to_dict(proof),
        "proof_output_root": output_root,
        "model_package_hash": str(receipt_context.get("model_package_hash", "")),
        "model_tensor_manifest_root": effective_tensor_manifest_root,
        "input_root": input_root,
        "weight_root": weight_root,
        "output_root": output_root,
        "input_shape": list(x_tensor.shape),
        "weight_shape": list(w_tensor.shape),
        "output_shape": list(y_tensor.shape),
        "input_dtype": "int8",
        "weight_dtype": "int8",
        "output_dtype": "int64",
        "input_chunk_size": int(x_merkle.chunk_size),
        "weight_chunk_size": int(w_merkle.chunk_size),
        "output_block_openings": _output_block_openings(
            proof,
            output=y_tensor,
            block_size=int(config.block_size),
        ),
        # Exact X anchor for the sidecar: the challenged band's own chunks
        # of the SAME tree input_root commits, so the verifier can rebuild
        # the band from authenticated bytes instead of trusting the
        # prover's X commitments between spot positions.
        "input_band_openings": _input_band_openings(
            proof,
            input_matrix=x_tensor,
            merkle=x_merkle,
            block_size=int(config.block_size),
        ),
        "proof_block_size": int(config.block_size),
        "proof_spot_checks": int(config.spot_checks),
        "transcript_context": transcript_context,
        "transcript_label_hex": transcript_label.hex(),
        "weight_source": "gguf_manifest" if used_manifest_weight else "captured_trace",
        "trace_io_layout": trace_io_layout,
        "float_max_abs_error": max_abs,
        "float_mean_abs_error": mean_abs,
        "float_max_rel_error": max_rel,
        "float_tolerance_abs": float(tolerance_abs),
        "float_tolerance_rel": float(tolerance_rel),
        "float_recompute_skipped": bool(float_recompute_skipped),
        "float_exact_fallback_used": bool(float_exact_fallback_used),
        "x_scale": x_scale,
        "w_scale": w_scale,
        "trace": {
            "path": str(trace.path),
            "created_unix_ns": int(trace.created_unix_ns),
            "graph_id": trace.graph_id,
            "op_index": trace.op_index,
            "tensor_name": trace.tensor_name,
            "src0_name": trace.src0_name,
            "src1_name": trace.src1_name,
            "dst_name": trace.dst_name,
            "src0_shape": list(trace.src0_shape),
            "src1_shape": list(trace.src1_shape),
            "dst_shape": list(trace.dst_shape),
            "source_types": dict(sorted(trace.source_types.items())),
            "backend": trace.backend,
            "device": trace.device,
            "manifest_index": (
                int(trace.manifest_index)
                if trace.manifest_index is not None
                else -1
            ),
            "graph_seq": int(trace.graph_seq),
            "intra_graph_index": int(trace.intra_graph_index),
            **(
                {"expert_index": int(trace.expert_index)}
                if trace.expert_index is not None
                else {}
            ),
        },
        "trace_membership": dict(trace_membership or {}),
    }
    if int(transcript_context.get("version", 0)) == 2:
        proof_payload["proof_beacon"] = str(
            receipt_context.get("proof_beacon", "")
        )
    if trace.src0_f32_path is not None:
        proof_payload["trace"]["src0_f32_sha256"] = _file_sha256(trace.src0_f32_path)
        proof_payload["trace"]["src0_f32_nbytes"] = trace.src0_f32_path.stat().st_size
    if trace.src1_f32_path is not None:
        proof_payload["trace"]["src1_f32_sha256"] = _file_sha256(trace.src1_f32_path)
        proof_payload["trace"]["src1_f32_nbytes"] = trace.src1_f32_path.stat().st_size
    if trace.dst_f32_path is not None:
        proof_payload["trace"]["dst_f32_sha256"] = _file_sha256(trace.dst_f32_path)
        proof_payload["trace"]["dst_f32_nbytes"] = trace.dst_f32_path.stat().st_size
    elif trace.src1_f32_path is None and anchored_rows is None:
        raise RuntimeError(
            "a trace without f32 witness files is only provable as an "
            "anchored row audit"
        )
    if trace.src0_raw_path is not None:
        proof_payload["trace"]["src0_raw_sha256"] = _file_sha256(trace.src0_raw_path)
        proof_payload["trace"]["src0_raw_nbytes"] = trace.src0_raw_path.stat().st_size
        proof_payload["trace"]["src0_raw_type"] = trace.src0_raw_type
    if gguf_tensor_membership is not None:
        proof_payload["gguf_tensor_membership"] = gguf_tensor_membership
    if gguf_bias_tensor_membership is not None:
        proof_payload["gguf_bias_tensor_membership"] = gguf_bias_tensor_membership
        proof_payload["trace_output_bias_tensor_name"] = trace_output_bias_tensor_name
    proof_payload["trace_output_transform"] = trace_output_transform
    if anchored_state is not None:
        proof_payload["anchor_row_openings"] = _anchored_openings_payload(
            anchored_state
        )
    if op_manifest_membership:
        proof_payload["op_manifest_membership"] = dict(op_manifest_membership)
    if slot_view_membership:
        proof_payload["slot_view_membership"] = dict(slot_view_membership)
    if decode_audit_positions:
        proof_payload["decode_audit_openings"] = make_decode_audit_openings_for_trace(
            trace,
            decode_audit_positions=[int(item) for item in decode_audit_positions],
            decode_audit_token_ids=[
                int(item) for item in (decode_audit_token_ids or [])
            ],
            decode_audit_top_k=int(decode_audit_top_k),
            proved_logits_i32=y_i64.reshape(-1),
        )
    from verallm.mesh.gemm_v2_sidecar import (
        GemmV2SidecarError,
        build_gemm_v2_sidecar,
        gemm_v2_available,
        gemm_v2_block_supported,
        gemm_v2_required,
    )

    gemm_v2_shape_ok = gemm_v2_block_supported(
        rows=int(x_tensor.shape[0]),
        inner=int(x_tensor.shape[1]),
        columns=int(w_tensor.shape[1]),
        block_size=int(config.block_size),
    )
    # Decode audits challenge the LM head as ONE full-row block so their
    # logit openings bind against the fully-opened output row. A row-wide
    # band cannot fit the per-column sidecar (combine-term cap), and it does
    # not need it: Y is opened in full, X is the single fully-dumped input
    # row, W is bound to the signed manifest through weight_root, and the v1
    # block sumcheck covers the arithmetic. The verifier mirrors this exact
    # exemption; every other op keeps the sidecar requirement.
    # TODO(chunked-commitment path): sidecar the wide row in 64-column
    # chunks so even the LM head gets Pallas-exact arithmetic.
    decode_audit_full_row_block = bool(
        decode_audit_positions
        and int(x_tensor.shape[0]) == 1
        and int(config.block_size) >= int(w_tensor.shape[1])
    )
    if gemm_v2_available() and not gemm_v2_shape_ok:
        if gemm_v2_required() and not decode_audit_full_row_block:
            raise RuntimeError(
                "VERATHOS_MESH_REQUIRE_GEMM_V2 is set but this operation's "
                "bands exceed the native PCS single-vector capacity "
                f"(x={tuple(int(i) for i in x_tensor.shape)}, "
                f"w={tuple(int(i) for i in w_tensor.shape)}, "
                f"block={int(config.block_size)})"
            )
        logger.info(
            "gemm-v2 sidecar skipped: band exceeds the PCS vector cap "
            "(needs the chunked commitment path)"
        )
    if gemm_v2_available() and gemm_v2_shape_ok:
        try:
            # The v1 proof's transcript-derived spot positions double as
            # witness-binding anchors: the sidecar opens its commitments at
            # the same cells the v1 spots Merkle-verify.
            spots_by_block = {
                (int(bp.bi), int(bp.bj)): {
                    "x": [(int(sp.row), int(sp.col)) for sp in bp.spot_X],
                    "w": [(int(sp.row), int(sp.col)) for sp in bp.spot_W],
                }
                for bp in proof.block_proofs
            }
            proof_payload["gemm_v2"] = build_gemm_v2_sidecar(
                x_matrix=x_tensor.detach().cpu().numpy(),
                w_matrix=w_tensor.detach().cpu().numpy(),
                y_matrix=y_tensor.detach().cpu().numpy().astype("int64"),
                challenged_blocks=[
                    (int(bi), int(bj)) for bi, bj in challenged_blocks
                ],
                block_size=int(config.block_size),
                transcript_label=transcript_label,
                tensor_name=tensor_name,
                layer_index=receipt_layer_index,
                op_index=int(trace.op_index),
                spot_positions=[
                    spots_by_block.get(
                        (int(bi), int(bj)), {"x": [], "w": []}
                    )
                    for bi, bj in challenged_blocks
                ],
            )
        except GemmV2SidecarError:
            # A witness that fails its own sumcheck is a real defect, never
            # something to ship without.
            raise
        except Exception:
            if gemm_v2_required():
                raise
            # Native PCS present but unusable (ABI drift, resource limits):
            # staged feature, ship the v1 payload and leave the sidecar off.
            logger.warning(
                "gemm-v2 sidecar generation failed; payload ships without it",
                exc_info=True,
            )
    elif gemm_v2_required() and not decode_audit_full_row_block:
        raise RuntimeError(
            "VERATHOS_MESH_REQUIRE_GEMM_V2 is set but the native PCS "
            "library is not available on this worker"
        )
    if (
        proof_payload.get("gemm_v2") is not None
        and gguf_manifest is not None
        and trace.expert_index is None
    ):
        # Full-band W binding to the signed manifest: challenge-time only,
        # fails closed when the witness columns diverge from the registered
        # model (shipping that payload would only earn a binding strike).
        from verallm.mesh.gguf_manifest import (
            _manifest_record_for_tensor,
            build_pcs_w_col_openings,
        )

        w_col_manifest = build_pcs_w_col_openings(
            _manifest_record_for_tensor(gguf_manifest, tensor_name),
            w_tensor.detach().cpu().numpy(),
            [(int(bi), int(bj)) for bi, bj in challenged_blocks],
            block_size=int(config.block_size),
        )
        if w_col_manifest is not None:
            proof_payload["gemm_v2_w_col_manifest"] = w_col_manifest
    proof_commitment_hash = ggml_proof_payload_commitment_hash(proof_payload)
    proof_payload["proof_commitment_hash"] = proof_commitment_hash
    verifier_ms = 0.0
    locally_verified = False
    if verify_before_return:
        independent_verification = verify_ggml_gemm_proof_payload(proof_payload)
        locally_verified = bool(independent_verification.verified)
        verifier_ms = float(independent_verification.verifier_ms)
        proof_payload["verification"] = {
            "verified": independent_verification.verified,
            "message": independent_verification.message,
        }
        proof_payload["verifier_ms"] = independent_verification.verifier_ms
        if not independent_verification.verified:
            raise RuntimeError(
                "GGML proof payload failed independent verification: "
                + independent_verification.message
            )
    else:
        proof_payload["verification"] = {
            "verified": False,
            "message": "skipped_before_return",
        }
        proof_payload["verifier_ms"] = 0.0

    receipt = LlamaGraphOpReceipt(
        request_id=str(receipt_context["request_id"]),
        mesh_id=str(receipt_context["mesh_id"]),
        mesh_spec_hash=str(receipt_context["mesh_spec_hash"]),
        stage_assignment_hash=str(receipt_context["stage_assignment_hash"]),
        rpc_plan_hash=str(receipt_context.get("rpc_plan_hash", "")),
        model_package_hash=str(receipt_context.get("model_package_hash", "")),
        model_tensor_manifest_root=effective_tensor_manifest_root,
        uid=int(receipt_context["uid"]),
        hotkey=str(receipt_context["hotkey"]),
        endpoint=str(receipt_context["endpoint"]),
        stage_index=int(receipt_context["stage_index"]),
        layer_start=int(receipt_context["layer_start"]),
        layer_end=int(receipt_context["layer_end"]),
        request_hash=str(receipt_context["request_hash"]),
        response_hash=str(receipt_context["response_hash"]),
        graph_id=trace.graph_id,
        op_index=trace.op_index,
        op_type=GGML_OP_MUL_MAT,
        layer_index=receipt_layer_index,
        tensor_name=trace.tensor_name,
        input_root=input_root,
        weight_root=weight_root,
        output_root=output_root,
        quantization="int8_proof_from_ggml_f32_trace",
        backend=trace.backend,
        device=trace.device,
        proof_kind="gemm",
        proof_commitment_hash=proof_commitment_hash,
        # Supplied by the caller from the stage's own activation handoff, not
        # derived from this op: the sampled op is one GEMM inside the stage,
        # while the boundary commits everything that crossed the RPC edge.
        input_boundary_root=str(
            receipt_context.get("input_boundary_root", "") or ""
        ),
        output_boundary_root=str(
            receipt_context.get("output_boundary_root", "") or ""
        ),
    )
    return VerifiedGgmlProof(
        receipt=receipt,
        proof_mode=VERATHOS_GGML_GEMM_PROOF_MODE,
        verified=locally_verified,
        float_max_abs_error=max_abs,
        float_mean_abs_error=mean_abs,
        proof_commitment_hash=proof_commitment_hash,
        input_root=input_root,
        weight_root=weight_root,
        output_root=output_root,
        trace_path=str(trace.path),
        proof_payload=proof_payload if include_proof else {
            "version": 1,
            "proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE,
            "proof_commitment_hash": proof_commitment_hash,
            "input_root": input_root,
            "weight_root": weight_root,
            "output_root": output_root,
        },
        verifier_ms=verifier_ms,
    )


def _write_warmup_trace(trace_dir: Path) -> GgmlMulMatTrace:
    """Write a tiny valid GGML MUL_MAT witness for prover warmup."""

    k = 32
    n = 8
    x = (np.arange(k, dtype=np.float32).reshape(1, k) - 16) / 16.0
    w = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 13 - 6) / 8.0
    y = x @ w

    prefix = trace_dir / "trace-1-warmup-cuda0-op1"
    np.ascontiguousarray(w.T, dtype=np.float32).tofile(str(prefix) + "-src0.f32")
    np.ascontiguousarray(x, dtype=np.float32).tofile(str(prefix) + "-src1.f32")
    np.ascontiguousarray(y, dtype=np.float32).tofile(str(prefix) + "-dst.f32")
    path = Path(str(prefix) + ".json")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_unix_ns": 1,
                "graph_id": "warmup-ggml-mul-mat",
                "op_index": 1,
                "op_type": GGML_OP_MUL_MAT,
                "tensor_name": "warmup",
                "src0_shape": [k, n, 1, 1],
                "src1_shape": [k, 1, 1, 1],
                "dst_shape": [n, 1, 1, 1],
                "src0_f32": prefix.name + "-src0.f32",
                "src1_f32": prefix.name + "-src1.f32",
                "dst_f32": prefix.name + "-dst.f32",
                "source_types": {"src0": "F32", "src1": "F32", "dst": "F32"},
                "backend": "llama_cpp_cuda",
                "device": "CUDA0",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return GgmlMulMatTrace.from_json(path)


def warm_ggml_proof_adapter(
    *,
    proof_block_size: int = 64,
    spot_checks: int = 8,
) -> float:
    """Warm the GEMM prover/verifier path and return elapsed milliseconds."""

    ctx = {
        "request_id": "warmup",
        "mesh_id": "warmup",
        "mesh_spec_hash": "00" * 32,
        "stage_assignment_hash": "11" * 32,
        "rpc_plan_hash": "22" * 32,
        "uid": 0,
        "hotkey": "warmup",
        "endpoint": "http://127.0.0.1:0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "request_hash": "33" * 32,
        "response_hash": "44" * 32,
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="verathos-ggml-proof-warmup-") as raw:
        trace = _write_warmup_trace(Path(raw))
        prove_ggml_mul_mat_trace(
            trace,
            ctx,
            tolerance_abs=1e-4,
            proof_block_size=proof_block_size,
            spot_checks=spot_checks,
        )
    return (time.perf_counter() - started) * 1000.0


def make_ggml_proof_server(
    *,
    trace_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 9349,
    tolerance_abs: float = 8e-2,
    tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    spot_checks: int = 8,
    trace_finder: Callable[[Mapping[str, Any]], GgmlMulMatTrace] | None = None,
    gguf_manifest_path: str | Path = "",
    warmup: bool = False,
) -> ThreadingHTTPServer:
    root = Path(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    gguf_manifest = (
        load_gguf_tensor_manifest(gguf_manifest_path)
        if gguf_manifest_path
        else None
    )
    warmup_ms = (
        warm_ggml_proof_adapter(
            proof_block_size=proof_block_size,
            spot_checks=spot_checks,
        )
        if warmup
        else 0.0
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "VerathosGgmlProofAdapter/0.1"

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "verathos-ggml-proof-adapter",
                        "version": 1,
                        "trace_dir": str(root),
                        "warmup_ms": round(warmup_ms, 3),
                        "time_unix": int(time.time()),
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/mesh/proof/commitment":
                try:
                    body = self._read_json_body()
                    ctx = body.get("receipt_context", {})
                    if not isinstance(ctx, dict):
                        raise ValueError("receipt_context must be an object")
                    if trace_finder is not None:
                        traces = [trace_finder(ctx)]
                    else:
                        traces = find_traces_for_window(
                            root,
                            start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                            end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                        )
                    manifest_root, manifest_count = ggml_op_manifest_summary_for_window(
                        root,
                        start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                        end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                    )
                    candidate_limit = max(
                        1,
                        int(
                            ctx.get("proof_trace_candidates_per_request")
                            or ctx.get("proof_ops_per_request")
                            or 1
                        ),
                    )
                    candidates = _ordered_traces(_witness_traces(traces))[:candidate_limit]
                    if not candidates and not manifest_root:
                        raise RuntimeError("no GGML proof trace or op manifest found for inference window")
                    payload = {
                        "version": 1,
                        "stage_index": int(ctx.get("stage_index", 0)),
                        "trace_commitment_root": ggml_trace_commitment_root(candidates),
                        "trace_commitment_count": len(candidates),
                    }
                    if manifest_root:
                        payload["op_manifest_root"] = manifest_root
                        payload["op_manifest_count"] = int(manifest_count)
                    self._send_json(200, payload)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/proof/selection":
                try:
                    body = self._read_json_body()
                    ctx = body.get("receipt_context", {})
                    if not isinstance(ctx, dict):
                        raise ValueError("receipt_context must be an object")
                    if trace_finder is not None:
                        traces = [trace_finder(ctx)]
                    else:
                        traces = find_traces_for_window(
                            root,
                            start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                            end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                        )
                    manifest_entries = find_op_manifest_entries_for_window(
                        root,
                        start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                        end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                    )
                    self._send_json(
                        200,
                        ggml_proof_selection_payload(
                            traces=traces,
                            manifest_entries=manifest_entries,
                            receipt_context=ctx,
                        ),
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") != "/v1/mesh/proof/receipt":
                self._send_json(404, {"error": "not found"})
                return
            try:
                body = self._read_json_body()
                ctx = body.get("receipt_context", {})
                if not isinstance(ctx, dict):
                    raise ValueError("receipt_context must be an object")
                if trace_finder is not None:
                    traces = [trace_finder(ctx)]
                else:
                    traces = find_traces_for_window(
                        root,
                        start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                        end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                    )
                manifest_entries = find_op_manifest_entries_for_window(
                    root,
                    start_unix_ns=int(ctx.get("inference_started_unix_ns", 0)),
                    end_unix_ns=int(ctx.get("inference_ended_unix_ns", 0)),
                )
                if not traces:
                    raise RuntimeError("no GGML proof trace found for inference window")
                errors: list[str] = []
                proofs = []
                proof_limit = max(1, int(ctx.get("proof_ops_per_request") or 1))
                candidate_limit = max(
                    proof_limit,
                    int(ctx.get("proof_trace_candidates_per_request") or proof_limit),
                )
                candidates = _ordered_traces(_witness_traces(traces))[:candidate_limit]
                trace_root = ggml_trace_commitment_root(candidates)
                beacon = str(ctx.get("proof_beacon", ""))
                if not beacon:
                    gate_hash = str(ctx.get("proof_gate_hash", ""))
                    beacon = (
                        derive_every_request_trace_beacon(gate_hash).hex()
                        if gate_hash
                        else derive_standalone_trace_beacon(trace_root).hex()
                    )
                manifest_ordered = []
                manifest_root = ""
                if manifest_entries:
                    manifest_ordered = _ordered_manifest_entries(manifest_entries)
                    manifest_root = ggml_op_manifest_root(manifest_ordered)
                if manifest_ordered and manifest_root:
                    selected_manifest_indexes = select_manifest_challenge_indexes(
                        beacon=beacon,
                        op_manifest_root=manifest_root,
                        op_manifest_count=len(manifest_ordered),
                        proof_ops_per_request=proof_limit,
                        stage_index=int(ctx.get("stage_index", 0)),
                    )
                    trace_pool = _ordered_traces(traces)
                    selected_items = []
                    for selected_manifest_index in selected_manifest_indexes:
                        entry = manifest_ordered[selected_manifest_index]
                        candidate_index = next(
                            (
                                idx
                                for idx, item in enumerate(candidates)
                                if entry.matches_trace(item)
                            ),
                            -1,
                        )
                        trace = next(
                            (
                                item
                                for item in trace_pool
                                if entry.matches_trace(item)
                            ),
                            None,
                        )
                        if trace is None:
                            raise RuntimeError(
                                "manifest-selected GGML op was not captured as a proof witness: "
                                f"manifest_index={entry.manifest_index}"
                            )
                        selected_items.append((trace, candidate_index))
                else:
                    selected_indexes = select_trace_challenge_indexes(
                        beacon=beacon,
                        trace_set_root=trace_root,
                        trace_set_count=len(candidates),
                        proof_ops_per_request=proof_limit,
                        stage_index=int(ctx.get("stage_index", 0)),
                    )
                    selected_items = [
                        (candidates[index], index) for index in selected_indexes
                    ]
                for trace, selected_index in selected_items:
                    try:
                        manifest_membership = (
                            op_manifest_membership_payload(
                                manifest_entries,
                                trace,
                                stage_index=int(ctx.get("stage_index", 0)),
                            )
                            if manifest_entries
                            else None
                        )
                        proofs.append(
                            prove_ggml_mul_mat_trace(
                                trace,
                                ctx,
                                tolerance_abs=tolerance_abs,
                                tolerance_rel=tolerance_rel,
                                proof_block_size=proof_block_size,
                                spot_checks=spot_checks,
                                include_proof=bool(body.get("include_proof")),
                                trace_membership=trace_membership_payload(
                                    candidates,
                                    selected_index,
                                    stage_index=int(ctx.get("stage_index", 0)),
                                )
                                if selected_index >= 0
                                else None,
                                op_manifest_membership=manifest_membership,
                                gguf_manifest=gguf_manifest,
                                verify_before_return=False,
                            )
                        )
                    except Exception as exc:
                        errors.append(f"{trace.path.name}: {exc}")
                if not proofs:
                    raise RuntimeError(
                        "no GGML proof trace verified: " + "; ".join(errors[:3])
                    )
                self._send_json(
                    200,
                    {
                        "proof_mode": proofs[0].proof_mode,
                        "verified": all(item.verified for item in proofs),
                        "proof_receipts": [item.receipt.to_dict() for item in proofs],
                        "proof_payloads": [
                            item.proof_payload for item in proofs if body.get("include_proof")
                        ],
                        "proof_verifier_ms": sum(item.verifier_ms for item in proofs),
                        "trace_paths": [item.trace_path for item in proofs],
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server


def serve_ggml_proof_adapter(
    *,
    trace_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 9349,
    tolerance_abs: float = 8e-2,
    tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    spot_checks: int = 8,
    gguf_manifest_path: str | Path = "",
    warmup: bool = True,
) -> None:
    server = make_ggml_proof_server(
        trace_dir=trace_dir,
        host=host,
        port=port,
        tolerance_abs=tolerance_abs,
        tolerance_rel=tolerance_rel,
        proof_block_size=proof_block_size,
        spot_checks=spot_checks,
        gguf_manifest_path=gguf_manifest_path,
        warmup=warmup,
    )
    print(f"verathos GGML proof adapter listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
