"""GGUF tensor manifest commitments for verified mesh proofs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from verallm.crypto.merkle import MerklePath, MerkleTree, verify_merkle_path
from verallm.mesh.types import canonical_json_bytes
from zkllm.config import DEFAULT_W_MERKLE_CHUNK_SIZE
from zkllm.crypto.merkle import FlatWeightMerkle


GGUF_TENSOR_MANIFEST_VERSION = 1
GGML_TRACE_MAX_ELEMS_FLOOR = 262_144

# Per-column PCS commitments for the gemm-v2 sidecar composition. Each dense
# provable tensor's record can carry a Merkle root over per-column PCS
# commitment hashes (grouped GGUF_PCS_W_COL_GROUP columns per leaf, aligned
# with the canonical proof block size). The root rides the tensor leaf body,
# so it is bound into the signed manifest root; the leaf-hash list is an
# UNCOMMITTED auxiliary field the prover uses to build membership paths.
# Commitments are deterministic functions of the proof_i8 bytes, so anyone
# holding the model can rebuild and check the whole structure.
GGUF_PCS_W_COL_VERSION = 1
GGUF_PCS_W_COL_GROUP = 64
_PCS_W_COL_LEAF_DOMAIN = b"VERATHOS_GGUF_PCS_W_COL_LEAF_V1"
_PCS_W_COL_NODE_DOMAIN = b"VERATHOS_GGUF_PCS_W_COL_NODE_V1"
_PCS_W_COL_ENV = "VERATHOS_MANIFEST_PCS_W_COLS"
_PCS_W_COL_ZERO_CACHE: dict[int, bytes] = {}
_PCS_W_COL_ZERO_LOCK = threading.Lock()


def _pcs_w_col_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _pcs_w_col_native_available() -> bool:
    try:
        from zkllm.crypto.pcs_v2 import native_library_path

        native_library_path()
        return True
    except Exception:
        return False


def _pcs_w_col_build_enabled() -> bool:
    raw = os.environ.get(_PCS_W_COL_ENV, "1")
    return raw.strip() != "0" and _pcs_w_col_native_available()


def pcs_w_col_leaf_hash(group_index: int, commitments: Sequence[bytes]) -> bytes:
    """Leaf hash for one group of consecutive per-column PCS commitments."""

    points = [bytes(item) for item in commitments]
    if not points or any(len(item) != 32 for item in points):
        raise ValueError("pcs w-col group commitments must be 32-byte points")
    return hashlib.sha256(
        _PCS_W_COL_LEAF_DOMAIN
        + int(group_index).to_bytes(4, "little")
        + len(points).to_bytes(2, "little")
        + b"".join(points)
    ).digest()


def _pcs_w_col_node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_PCS_W_COL_NODE_DOMAIN + left + right).digest()


def pcs_w_col_root_from_leaf_hashes(leaf_hashes: Sequence[bytes]) -> str:
    """Root over group-leaf hashes (odd nodes pair with themselves)."""

    current = [bytes(item) for item in leaf_hashes]
    if not current:
        return ""
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else left
            next_level.append(_pcs_w_col_node_hash(left, right))
        current = next_level
    return current[0].hex()


def pcs_w_col_membership_path(
    leaf_hashes: Sequence[bytes],
    index: int,
) -> list[dict[str, Any]]:
    """Membership path for one group leaf, mirroring the root construction."""

    level = [bytes(item) for item in leaf_hashes]
    if not 0 <= int(index) < len(level):
        raise ValueError("pcs w-col group index outside the leaf list")
    idx = int(index)
    path: list[dict[str, Any]] = []
    while len(level) > 1:
        sibling_idx = idx ^ 1
        if sibling_idx >= len(level):
            sibling_idx = idx
        path.append(
            {
                "sibling": level[sibling_idx].hex(),
                "sibling_is_left": bool(idx % 2 == 1),
            }
        )
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_pcs_w_col_node_hash(left, right))
        level = next_level
        idx //= 2
    return path


def verify_pcs_w_col_membership(
    *,
    leaf_hash: bytes,
    root: str,
    leaf_count: int,
    index: int,
    path: Sequence[Mapping[str, Any]],
) -> bool:
    """Verify one group leaf against a committed pcs w-col root."""

    if leaf_count <= 0 or not 0 <= int(index) < int(leaf_count):
        return False
    if len(str(root)) != 64:
        return False
    node = bytes(leaf_hash)
    try:
        for item in path:
            sibling = bytes.fromhex(str(item["sibling"]))
            if bool(item.get("sibling_is_left", False)):
                node = _pcs_w_col_node_hash(sibling, node)
            else:
                node = _pcs_w_col_node_hash(node, sibling)
    except Exception:
        return False
    return node.hex() == str(root)


def pcs_w_col_zero_commitment(vector_length: int) -> bytes:
    """Commitment of the all-zero padded column (pad columns in edge blocks)."""

    length = int(vector_length)
    with _PCS_W_COL_ZERO_LOCK:
        cached = _PCS_W_COL_ZERO_CACHE.get(length)
    if cached is not None:
        return cached
    from zkllm.crypto import pcs_v2

    value = bytes(
        pcs_v2.commit_i8_batch(bytes(length), vector_length=length)[0]
    )
    with _PCS_W_COL_ZERO_LOCK:
        _PCS_W_COL_ZERO_CACHE[length] = value
    return value


def compute_pcs_w_col_fields(
    proof_i8: np.ndarray,
    *,
    column_chunk: int = 4096,
) -> dict[str, Any]:
    """Compute the per-column PCS commitment fields for one dense tensor.

    Registration-time only. Columns are padded to the pow2 inner length with
    the exact call shape the gemm-v2 sidecar prover uses, so an honest
    sidecar's ``w_col_commitments`` equal these bytes verbatim.
    """

    from zkllm.crypto import pcs_v2

    if proof_i8.ndim != 2:
        raise ValueError("pcs w-col fields need a 2-D proof_i8 matrix")
    k, n = int(proof_i8.shape[0]), int(proof_i8.shape[1])
    k_p2 = _pcs_w_col_pow2(k)
    if k_p2 > int(pcs_v2.MAX_VECTOR_LEN):
        raise ValueError("tensor inner dimension exceeds the PCS vector cap")
    # One native call packs at most MAX_BATCH_I8_BYTES; size the column
    # chunk from the padded vector length (ffn_down-class tensors have
    # k_p2 = 32768, far past 4096 columns per call).
    column_chunk = max(
        1,
        min(
            int(column_chunk),
            int(pcs_v2.MAX_BATCH_COMMITMENTS),
            int(pcs_v2.MAX_BATCH_I8_BYTES) // k_p2,
        ),
    )
    commitments: list[bytes] = []
    for start in range(0, n, int(column_chunk)):
        stop = min(start + int(column_chunk), n)
        block = np.zeros((stop - start, k_p2), dtype=np.int8)
        block[:, :k] = proof_i8[:, start:stop].T
        commitments.extend(
            bytes(item)
            for item in pcs_v2.commit_i8_batch(
                block.tobytes(), vector_length=k_p2
            )
        )
    leaf_hashes = [
        pcs_w_col_leaf_hash(
            gi // GGUF_PCS_W_COL_GROUP,
            commitments[gi : gi + GGUF_PCS_W_COL_GROUP],
        )
        for gi in range(0, n, GGUF_PCS_W_COL_GROUP)
    ]
    return {
        "pcs_w_col_version": GGUF_PCS_W_COL_VERSION,
        "pcs_w_col_group": GGUF_PCS_W_COL_GROUP,
        "pcs_w_col_count": n,
        "pcs_w_col_vector_length": k_p2,
        "pcs_w_col_root": pcs_w_col_root_from_leaf_hashes(leaf_hashes),
        "pcs_w_col_leaf_hashes": [item.hex() for item in leaf_hashes],
    }


def pcs_w_col_groups_for_band(
    *,
    col_start: int,
    real_cols: int,
    group: int,
) -> list[int]:
    """Group indexes covering one challenged column band."""

    if real_cols <= 0 or group <= 0:
        raise ValueError("pcs w-col band is empty")
    first = int(col_start) // int(group)
    last = (int(col_start) + int(real_cols) - 1) // int(group)
    return list(range(first, last + 1))


def build_pcs_w_col_openings(
    record: Mapping[str, Any],
    w_matrix: np.ndarray,
    challenged_blocks: Sequence[tuple[int, int]],
    *,
    block_size: int,
) -> dict[str, Any] | None:
    """Build the challenged blocks' column-commitment manifest openings.

    Challenge-time only. The prover recommits the covered column groups from
    its own weight matrix and FAILS CLOSED if the resulting leaf hash does
    not match the manifest: shipping such a payload would only surface as a
    validator binding strike.
    """

    root = str(record.get("pcs_w_col_root", "") or "")
    if not root:
        return None
    from zkllm.crypto import pcs_v2

    group = int(record.get("pcs_w_col_group", 0))
    count = int(record.get("pcs_w_col_count", 0))
    vector_length = int(record.get("pcs_w_col_vector_length", 0))
    raw_hashes = record.get("pcs_w_col_leaf_hashes", [])
    if group <= 0 or count <= 0 or vector_length <= 0:
        raise RuntimeError("pcs w-col manifest record is malformed")
    leaf_hashes = [bytes.fromhex(str(item)) for item in raw_hashes]
    expected_leaves = (count + group - 1) // group
    if len(leaf_hashes) != expected_leaves:
        raise RuntimeError(
            "pcs w-col leaf hash list does not match the manifest record"
        )
    if pcs_w_col_root_from_leaf_hashes(leaf_hashes) != root:
        raise RuntimeError("pcs w-col leaf hashes do not match the signed root")
    k, n = int(w_matrix.shape[0]), int(w_matrix.shape[1])
    if n != count or _pcs_w_col_pow2(k) != vector_length:
        raise RuntimeError("pcs w-col record does not match the witness shape")

    group_cache: dict[int, list[bytes]] = {}

    def _group_commitments(g: int) -> list[bytes]:
        cached = group_cache.get(g)
        if cached is not None:
            return cached
        g0 = g * group
        g1 = min(g0 + group, n)
        block = np.zeros((g1 - g0, vector_length), dtype=np.int8)
        block[:, :k] = w_matrix[:, g0:g1].T
        commits = [
            bytes(item)
            for item in pcs_v2.commit_i8_batch(
                block.tobytes(), vector_length=vector_length
            )
        ]
        if pcs_w_col_leaf_hash(g, commits) != leaf_hashes[g]:
            raise RuntimeError(
                "witness weight columns diverge from the manifest "
                "pcs w-col commitments"
            )
        group_cache[g] = commits
        return commits

    blocks_out: list[dict[str, Any]] = []
    for bi, bj in challenged_blocks:
        col_start = int(bj) * int(block_size)
        real_cols = min(int(block_size), n - col_start)
        if real_cols <= 0:
            raise RuntimeError("pcs w-col challenged block outside the tensor")
        entries = []
        for g in pcs_w_col_groups_for_band(
            col_start=col_start, real_cols=real_cols, group=group
        ):
            commits = _group_commitments(g)
            entries.append(
                {
                    "group_index": int(g),
                    "commitments": [item.hex() for item in commits],
                    "path": pcs_w_col_membership_path(leaf_hashes, g),
                }
            )
        blocks_out.append(
            {
                "row_block": int(bi),
                "column_block": int(bj),
                "groups": entries,
            }
        )
    return {"version": GGUF_PCS_W_COL_VERSION, "blocks": blocks_out}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file_range(path: Path, *, offset: int, n_bytes: int) -> str:
    h = hashlib.sha256()
    remaining = int(n_bytes)
    with path.open("rb") as handle:
        handle.seek(int(offset))
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    if remaining != 0:
        raise RuntimeError("GGUF tensor range ended before expected byte length")
    return h.hexdigest()


def _path_to_dict(path: MerklePath) -> list[dict[str, Any]]:
    return [
        {"sibling": sibling.hex(), "sibling_is_left": bool(is_left)}
        for sibling, is_left in path.siblings
    ]


def _path_from_payload(items: list[Mapping[str, Any]], *, leaf_index: int) -> MerklePath:
    return MerklePath(
        leaf_index=int(leaf_index),
        siblings=[
            (
                bytes.fromhex(str(item["sibling"])),
                bool(item.get("sibling_is_left", False)),
            )
            for item in items
        ],
    )


def tensor_leaf_body(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable leaf body for one GGUF tensor record."""

    body = {
        "version": GGUF_TENSOR_MANIFEST_VERSION,
        "name": str(record["name"]),
        "tensor_type": str(record["tensor_type"]),
        "shape": [int(item) for item in record["shape"]],
        "n_elements": int(record["n_elements"]),
        "n_bytes": int(record["n_bytes"]),
        "data_offset": int(record["data_offset"]),
        "raw_sha256": str(record["raw_sha256"]),
        "f32_nbytes": int(record["f32_nbytes"]),
        "f32_sha256": str(record["f32_sha256"]),
    }
    if "model_file_index" in record:
        body["model_file_index"] = int(record["model_file_index"])
    if "model_file_sha256" in record:
        body["model_file_sha256"] = str(record["model_file_sha256"])
    for key in (
        "proof_i8_nbytes",
        "proof_i8_sha256",
        "proof_i8_merkle_root",
        "proof_i8_chunk_size",
    ):
        if key in record:
            value = record[key]
            body[key] = int(value) if key.endswith("_nbytes") or key.endswith("_size") else str(value)
    # Per-column PCS commitment root (gemm-v2 composition): bound into the
    # leaf so the signed manifest fixes it and a miner cannot strip it to
    # dodge the sidecar's full-band W binding. The auxiliary leaf-hash list
    # (pcs_w_col_leaf_hashes) deliberately stays OUT of the leaf body.
    if "pcs_w_col_root" in record:
        body["pcs_w_col_version"] = int(record["pcs_w_col_version"])
        body["pcs_w_col_group"] = int(record["pcs_w_col_group"])
        body["pcs_w_col_count"] = int(record["pcs_w_col_count"])
        body["pcs_w_col_vector_length"] = int(record["pcs_w_col_vector_length"])
        body["pcs_w_col_root"] = str(record["pcs_w_col_root"])
    # Per-expert plane commitments (MoE): these MUST be part of the leaf or
    # the planes are not bound to the manifest root at all. Absent on dense
    # tensors, so pre-MoE manifests keep their roots byte-identical.
    if "proof_i8_expert_planes" in record:
        body["proof_i8_expert_planes"] = int(record["proof_i8_expert_planes"])
        body["proof_i8_expert_nbytes"] = int(record["proof_i8_expert_nbytes"])
        body["proof_i8_expert_sha256"] = [str(x) for x in record["proof_i8_expert_sha256"]]
        body["proof_i8_expert_merkle_roots"] = [
            str(x) for x in record["proof_i8_expert_merkle_roots"]
        ]
    return body


def tensor_leaf_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        b"VERATHOS_GGUF_TENSOR_MANIFEST_LEAF_V1"
        + canonical_json_bytes(tensor_leaf_body(record))
    )


def _manifest_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("tensors", [])
    if not isinstance(records, list) or not records:
        raise ValueError("GGUF tensor manifest must contain tensors")
    return sorted(
        [dict(item) for item in records],
        key=lambda item: (
            str(item.get("name", "")),
            int(item.get("model_file_index", 0)),
            int(item.get("data_offset", 0)),
        ),
    )


def _expand_split_gguf_path(path: Path) -> list[Path]:
    match = re.match(
        r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.gguf)$",
        path.name,
        flags=re.IGNORECASE,
    )
    if match is None or match.group("index") != "00001":
        return [path]
    count = int(match.group("count"))
    paths = [
        path.with_name(
            f"{match.group('prefix')}-{index:05d}-of-{count:05d}{match.group('suffix')}"
        )
        for index in range(1, count + 1)
    ]
    return paths if all(item.exists() for item in paths) else [path]


def _normalize_manifest_model_paths(model_path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(model_path, (str, Path)):
        paths = [Path(model_path)]
    else:
        paths = [Path(item) for item in model_path]
    if not paths:
        raise ValueError("at least one GGUF model path is required")
    if len(paths) == 1:
        paths = _expand_split_gguf_path(paths[0])
    return [path.expanduser().resolve() for path in paths]


def _aggregate_model_file_sha256(model_files: list[Mapping[str, Any]]) -> str:
    if len(model_files) == 1:
        return str(model_files[0]["sha256"])
    body = [
        {
            "index": int(item["index"]),
            "sha256": str(item["sha256"]),
            "n_bytes": int(item["n_bytes"]),
        }
        for item in model_files
    ]
    return hashlib.sha256(
        b"VERATHOS_GGUF_MODEL_FILE_SET_V1" + canonical_json_bytes(body)
    ).hexdigest()


def _tensor_f32_data(gguf_mod: Any, tensor: Any) -> np.ndarray:
    """Deterministic f32 view of a GGUF tensor, integer types included.

    ``gguf.dequantize`` raises NotImplementedError for integer tensor types
    (I8/I16/I32/I64); DeepSeek V4's sparse-attention indexer ships I32
    tensors, which broke manifest builds. Integer tensors are never GEMM
    weights (MUL_MAT src0 is float-quantized; MUL_MAT_ID takes ids as a
    separate input, not a provable weight), so an exact astype conversion
    keeps them in the committed manifest without a float-dequant path.
    """
    type_name = str(getattr(tensor.tensor_type, "name", tensor.tensor_type))
    if type_name in ("I8", "I16", "I32", "I64"):
        return tensor.data.astype("float32")
    return gguf_mod.dequantize(tensor.data, tensor.tensor_type).astype(
        "float32", copy=False
    )


_MANIFEST_WORKER_READERS: dict[str, Any] = {}


def _manifest_worker_tensor(path_str: str, tensor_name: str) -> Any:
    """Fetch one tensor via a per-process cached GGUFReader (workers only)."""

    import gguf

    reader = _MANIFEST_WORKER_READERS.get(path_str)
    if reader is None:
        reader = gguf.GGUFReader(path_str)
        _MANIFEST_WORKER_READERS[path_str] = reader
    for tensor in reader.tensors:
        if str(tensor.name) == tensor_name:
            return tensor
    raise RuntimeError(f"tensor {tensor_name!r} not found in {path_str}")


def _expert_plane_byte_view(gguf_mod: Any, tensor: Any, planes: int) -> Any | None:
    """Flat per-plane view of an expert tensor's quantized bytes, or None.

    Returns the flat data array and per-plane item count when the layout is
    the contiguous block-aligned one every K-quant uses; the caller then
    dequantizes ONE plane at a time instead of materialising the full f32
    tensor (15GB-class for a 256-expert layer). The per-plane f32 streams
    concatenate to exactly the full-tensor dequant, so hashes are unchanged.
    """
    try:
        quant_sizes = getattr(gguf_mod, "GGML_QUANT_SIZES", None)
        if quant_sizes is None:
            quant_sizes = gguf_mod.constants.GGML_QUANT_SIZES
        block_size, type_size = quant_sizes[tensor.tensor_type]
        shape = [int(d) for d in tensor.shape.tolist()]
        k_dim, n_dim = shape[0], shape[1]
        if k_dim % int(block_size) != 0:
            return None
        plane_bytes = (k_dim // int(block_size)) * int(type_size) * n_dim
        flat = tensor.data.reshape(-1)
        itemsize = int(flat.dtype.itemsize)
        if plane_bytes % itemsize != 0:
            return None
        if int(flat.size) * itemsize != plane_bytes * planes:
            return None
        return flat, plane_bytes // itemsize
    except Exception:
        return None


def _build_tensor_manifest_record(
    path_str: str,
    file_index: int,
    model_file_sha256: str,
    tensor_name: str,
) -> dict[str, Any]:
    """Build one tensor's manifest record (process-pool worker body).

    Byte-identical to the old sequential builder: same record fields, same
    hashes, same per-plane commitments. Expert tensors stream plane by plane
    (their per-plane f32 bytes concatenate to the full dequant) so a worker
    never materialises a full 15GB expert tensor in f32.
    """

    import gguf

    path = Path(path_str)
    tensor = _manifest_worker_tensor(path_str, tensor_name)
    tensor_type = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
    tensor_shape = [int(item) for item in tensor.shape.tolist()]
    is_expert = (
        len(tensor_shape) >= 3
        and tensor_shape[2] > 1
        and int(tensor.n_elements)
        == tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    )
    record = {
        "name": str(tensor.name),
        "tensor_type": str(tensor_type),
        "shape": tensor_shape,
        "n_elements": int(tensor.n_elements),
        "n_bytes": int(tensor.n_bytes),
        "data_offset": int(tensor.data_offset),
        "raw_sha256": _sha256_file_range(
            path,
            offset=int(tensor.data_offset),
            n_bytes=int(tensor.n_bytes),
        ),
        "f32_nbytes": int(tensor.n_elements) * 4,
        "model_file": str(path),
        "model_file_index": int(file_index),
        "model_file_sha256": model_file_sha256,
    }
    plane_view = (
        _expert_plane_byte_view(gguf, tensor, tensor_shape[2]) if is_expert else None
    )
    if is_expert and plane_view is not None:
        flat, per_plane = plane_view
        k_dim, n_dim, n_experts = tensor_shape[0], tensor_shape[1], tensor_shape[2]
        f32_hash = hashlib.sha256()
        expert_hashes: list[str] = []
        expert_roots: list[str] = []
        for expert in range(n_experts):
            chunk = np.ascontiguousarray(
                flat[expert * per_plane : (expert + 1) * per_plane]
            )
            plane_f32 = gguf.dequantize(chunk, tensor.tensor_type).astype(
                "float32", copy=False
            ).reshape(-1)
            f32_hash.update(plane_f32.tobytes(order="C"))
            w_f32 = proof_f32_weight_matrix_from_gguf_f32(plane_f32, [k_dim, n_dim])
            proof_i8, _proof_scale = quantize_proof_i8(w_f32)
            expert_hashes.append(
                hashlib.sha256(proof_i8.tobytes(order="C")).hexdigest()
            )
            expert_roots.append(
                FlatWeightMerkle(
                    torch.from_numpy(proof_i8),
                    DEFAULT_W_MERKLE_CHUNK_SIZE,
                    store_raw=False,
                ).root.hex()
            )
            # Deliberately NOT banked: planes bank lazily at proof time,
            # governed by the LRU cap (all planes of all expert tensors are
            # a 30GB-class footprint the challenges barely touch).
        record["f32_sha256"] = f32_hash.hexdigest()
        record.update(
            {
                "proof_i8_expert_planes": int(n_experts),
                "proof_i8_expert_nbytes": int(k_dim * n_dim),
                "proof_i8_expert_sha256": expert_hashes,
                "proof_i8_expert_merkle_roots": expert_roots,
                "proof_i8_chunk_size": int(DEFAULT_W_MERKLE_CHUNK_SIZE),
            }
        )
        return record
    f32_data = _tensor_f32_data(gguf, tensor)
    record["f32_sha256"] = hashlib.sha256(f32_data.tobytes(order="C")).hexdigest()
    if (
        len(tensor_shape) >= 2
        and int(tensor.n_elements) == tensor_shape[0] * tensor_shape[1]
    ):
        w_f32 = proof_f32_weight_matrix_from_gguf_f32(f32_data, tensor_shape)
        proof_i8, proof_scale = quantize_proof_i8(w_f32)
        proof_i8_bytes = proof_i8.tobytes(order="C")
        proof_i8_merkle = FlatWeightMerkle(
            torch.from_numpy(proof_i8),
            DEFAULT_W_MERKLE_CHUNK_SIZE,
            store_raw=False,
        )
        record.update(
            {
                "proof_i8_nbytes": len(proof_i8_bytes),
                "proof_i8_sha256": hashlib.sha256(proof_i8_bytes).hexdigest(),
                "proof_i8_merkle_root": proof_i8_merkle.root.hex(),
                "proof_i8_chunk_size": int(DEFAULT_W_MERKLE_CHUNK_SIZE),
            }
        )
        if _pcs_w_col_build_enabled():
            try:
                record.update(compute_pcs_w_col_fields(proof_i8))
            except Exception:
                # Staged: a tensor outside the PCS caps (or a native-lib
                # hiccup) ships without the column root; the sidecar keeps
                # its spot binding for that tensor. The authority build
                # should treat a warning here as a review item.
                logging.getLogger(__name__).warning(
                    "pcs w-col commitments skipped for tensor %s",
                    record.get("name", "?"),
                    exc_info=True,
                )
        # Bank the blob NOW: it is already computed, and every future proof
        # on this box would otherwise re-pay this dequant during a live
        # request (the 20s "cold proof" stall). Best-effort, profile-gated:
        # under compact the re-derive is sha-verified and fast, so banking
        # every tensor would only rebuild the 2.5x-model-size cache.
        if _should_persist_i8(record):
            try:
                store_cached_proof_i8(record, proof_i8, proof_scale)
            except Exception:
                pass
    elif is_expert:
        # Fallback for unexpected byte layouts: full dequant, plane slicing
        # over the f32 stream (the original path).
        k_dim, n_dim, n_experts = tensor_shape[0], tensor_shape[1], tensor_shape[2]
        plane_elems = k_dim * n_dim
        expert_hashes = []
        expert_roots = []
        for expert in range(n_experts):
            plane_f32 = f32_data.reshape(-1)[
                expert * plane_elems : (expert + 1) * plane_elems
            ]
            w_f32 = proof_f32_weight_matrix_from_gguf_f32(plane_f32, [k_dim, n_dim])
            proof_i8, _proof_scale = quantize_proof_i8(w_f32)
            expert_hashes.append(
                hashlib.sha256(proof_i8.tobytes(order="C")).hexdigest()
            )
            expert_roots.append(
                FlatWeightMerkle(
                    torch.from_numpy(proof_i8),
                    DEFAULT_W_MERKLE_CHUNK_SIZE,
                    store_raw=False,
                ).root.hex()
            )
        record.update(
            {
                "proof_i8_expert_planes": int(n_experts),
                "proof_i8_expert_nbytes": int(plane_elems),
                "proof_i8_expert_sha256": expert_hashes,
                "proof_i8_expert_merkle_roots": expert_roots,
                "proof_i8_chunk_size": int(DEFAULT_W_MERKLE_CHUNK_SIZE),
            }
        )
    return record


def build_gguf_tensor_manifest(model_path: str | Path | Sequence[str | Path]) -> dict[str, Any]:
    """Build a deterministic Merkle manifest over raw tensor bytes in a GGUF file.

    Records are built per tensor in a process pool (32-core boxes finish a
    100GB MoE in minutes instead of the ~19 hours the sequential builder
    took); every record is byte-identical to the sequential output, and the
    final ordering is re-sorted, so manifest roots are unchanged.

    The pool uses the SPAWN start method (fork-started children inheriting a
    loaded native PCS library deadlock), so any script calling this from its
    main module MUST guard with ``if __name__ == "__main__"`` or spawn's
    __main__ re-import re-runs the build recursively and breaks the pool.
    Cap workers with ``VERATHOS_MANIFEST_BUILD_WORKERS`` on memory-tight
    boxes.
    """

    try:
        import gguf
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("GGUF tensor manifest generation requires the gguf package") from exc

    paths = _normalize_manifest_model_paths(model_path)
    model_files = [
        {
            "index": index,
            "path": str(path),
            "sha256": _sha256_file(path),
            "n_bytes": path.stat().st_size,
        }
        for index, path in enumerate(paths)
    ]
    work: list[tuple[str, int, str, str]] = []
    for file_index, path in enumerate(paths):
        reader = gguf.GGUFReader(path)
        model_file_sha256 = str(model_files[file_index]["sha256"])
        for tensor in sorted(reader.tensors, key=lambda item: item.name):
            work.append(
                (str(path), file_index, model_file_sha256, str(tensor.name))
            )
        del reader
    # Peak RSS per worker is a few GB on large tensors (f32 dequant + proof
    # i8 + padded PCS commit buffers); cap workers on memory-tight boxes.
    worker_cap = int(os.environ.get("VERATHOS_MANIFEST_BUILD_WORKERS", "16") or 16)
    max_workers = min(max(1, worker_cap), os.cpu_count() or 1, max(1, len(work)))
    if max_workers <= 1:
        records = [_build_tensor_manifest_record(*item) for item in work]
    else:
        import concurrent.futures
        import multiprocessing

        # Spawn, never fork: workers call the native PCS library for the
        # per-column commitments, and a fork-started child that inherits a
        # parent which already loaded that library (rayon thread pool state,
        # no threads in the child) deadlocks inside the first commit.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            records = list(
                pool.map(
                    _build_tensor_manifest_record,
                    *zip(*work),
                    chunksize=1,
                )
            )

    if not records:
        raise RuntimeError("GGUF tensor manifest cannot be built for a model without tensors")
    ordered_records = _manifest_records({"tensors": records})
    leaf_bytes = [tensor_leaf_bytes(record) for record in ordered_records]
    tree = MerkleTree(leaf_bytes)
    aggregate_sha256 = _aggregate_model_file_sha256(model_files)
    return {
        "version": GGUF_TENSOR_MANIFEST_VERSION,
        "model_file": str(paths[0]),
        "model_files": model_files,
        "model_file_sha256": aggregate_sha256,
        "tensor_count": len(ordered_records),
        "tensor_manifest_root": tree.root.hex(),
        "tensors": ordered_records,
    }


def bind_gguf_manifest_to_local_model(
    manifest: Mapping[str, Any],
    model_path: str | Path | Sequence[str | Path],
    *,
    expected_package_hash: str | None = None,
) -> dict[str, Any]:
    """Return a path-rebased manifest after verifying the local GGUF package.

    Tensor manifests are path-independent commitments: ``model_file`` and the
    ``path`` members of ``model_files`` are runtime locations and are excluded
    from tensor leaves.  A manifest copied from another host therefore keeps
    valid commitments but cannot service a proof-blob cache miss until those
    locations point at the driver's GGUF.

    Rebinding is safe only after every local file matches the manifest's
    committed size and SHA256 (and, when supplied, the expected package hash).
    Per-tensor file hashes/indexes and the tensor Merkle root are checked again
    before the rebased copy is returned.  The input mapping is never mutated.
    """

    raw_files = manifest.get("model_files")
    committed_files: list[dict[str, Any]] = []
    if isinstance(raw_files, Sequence) and not isinstance(
        raw_files, (str, bytes, bytearray)
    ) and raw_files:
        for position, raw in enumerate(raw_files):
            if not isinstance(raw, Mapping):
                raise ValueError("GGUF manifest model_files entries must be objects")
            index = raw.get("index", position)
            n_bytes = raw.get("n_bytes")
            sha256_hex = str(raw.get("sha256", ""))
            if type(index) is not int or index < 0:
                raise ValueError("GGUF manifest model file index must be non-negative")
            if type(n_bytes) is not int or n_bytes <= 0:
                raise ValueError("GGUF manifest model file n_bytes must be positive")
            if not re.fullmatch(r"[0-9a-f]{64}", sha256_hex):
                raise ValueError("GGUF manifest model file sha256 is invalid")
            committed_files.append(
                {
                    "index": index,
                    "n_bytes": n_bytes,
                    "sha256": sha256_hex,
                    "source": dict(raw),
                }
            )
        committed_files.sort(key=lambda item: int(item["index"]))
        if [int(item["index"]) for item in committed_files] != list(
            range(len(committed_files))
        ):
            raise ValueError(
                "GGUF manifest model file indexes must be contiguous from zero"
            )
    else:
        sha256_hex = str(manifest.get("model_file_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha256_hex):
            raise ValueError("GGUF manifest model_file_sha256 is invalid")
        raw_n_bytes = manifest.get("model_file_n_bytes")
        if raw_n_bytes is not None and (
            type(raw_n_bytes) is not int or raw_n_bytes <= 0
        ):
            raise ValueError("GGUF manifest model_file_n_bytes must be positive")
        committed_files.append(
            {
                "index": 0,
                "n_bytes": raw_n_bytes,
                "sha256": sha256_hex,
                "source": None,
            }
        )

    package_hash = _aggregate_model_file_sha256(committed_files)
    declared_package_hash = str(manifest.get("model_file_sha256", ""))
    if declared_package_hash != package_hash:
        raise ValueError("GGUF manifest package hash does not match its file set")
    if expected_package_hash is not None:
        expected = str(expected_package_hash).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("expected GGUF package hash is invalid")
        if expected != package_hash:
            raise ValueError("local GGUF package hash does not match the expected package")

    local_paths = _normalize_manifest_model_paths(model_path)
    if len(local_paths) != len(committed_files):
        raise ValueError(
            "local GGUF file count does not match the committed package"
        )
    for local_path, committed in zip(local_paths, committed_files):
        if not local_path.is_file():
            raise ValueError(f"local GGUF model file is unavailable: {local_path}")
        expected_size = committed.get("n_bytes")
        if expected_size is not None and local_path.stat().st_size != int(expected_size):
            raise ValueError(f"local GGUF model file size mismatch: {local_path}")
        if _sha256_file(local_path) != str(committed["sha256"]):
            raise ValueError(f"local GGUF model file hash mismatch: {local_path}")

    records = _manifest_records(manifest)
    expected_root = str(manifest.get("tensor_manifest_root", ""))
    computed_root = MerkleTree([tensor_leaf_bytes(record) for record in records]).root.hex()
    if expected_root != computed_root:
        raise ValueError("GGUF tensor manifest root mismatch")
    for record in records:
        file_index = int(record.get("model_file_index", 0))
        if not (0 <= file_index < len(committed_files)):
            raise ValueError("GGUF tensor references an invalid model file index")
        record_file_hash = str(record.get("model_file_sha256", ""))
        if record_file_hash and record_file_hash != str(
            committed_files[file_index]["sha256"]
        ):
            raise ValueError("GGUF tensor model file hash disagrees with package")

    rebound = deepcopy(dict(manifest))
    rebound["model_file"] = str(local_paths[0])
    if isinstance(raw_files, Sequence) and not isinstance(
        raw_files, (str, bytes, bytearray)
    ) and raw_files:
        rebound["model_files"] = []
        for local_path, committed in zip(local_paths, committed_files):
            item = dict(committed["source"])
            item["path"] = str(local_path)
            rebound["model_files"].append(item)
    for raw_record in rebound.get("tensors", []):
        if not isinstance(raw_record, dict):
            raise ValueError("GGUF tensor manifest entries must be objects")
        file_index = int(raw_record.get("model_file_index", 0))
        raw_record["model_file"] = str(local_paths[file_index])
        # A copied host-local f32 cache is not part of the package and cannot
        # be rebased to a GGUF shard. Drop it so resolution proceeds through
        # the content-addressed cache or the verified local model file.
        raw_record.pop("f32_path", None)

    rebound_records = _manifest_records(rebound)
    rebound_root = MerkleTree(
        [tensor_leaf_bytes(record) for record in rebound_records]
    ).root.hex()
    if rebound_root != expected_root:
        raise RuntimeError("GGUF runtime path binding changed tensor commitments")
    return rebound


def strip_gguf_manifest_runtime_paths(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a file-less manifest without host-local path disclosures.

    Package hashes, file indexes/sizes, and all tensor commitments remain in
    the result. Only uncommitted runtime locators are removed. Rechecking the
    Merkle root makes the boundary explicit and prevents a future committed
    field from being stripped accidentally.
    """

    records = _manifest_records(manifest)
    expected_root = str(manifest.get("tensor_manifest_root", ""))
    if MerkleTree([tensor_leaf_bytes(record) for record in records]).root.hex() != (
        expected_root
    ):
        raise ValueError("GGUF tensor manifest root mismatch")
    stripped = deepcopy(dict(manifest))
    stripped.pop("model_file", None)
    raw_files = stripped.get("model_files")
    if isinstance(raw_files, list):
        for raw in raw_files:
            if not isinstance(raw, dict):
                raise ValueError("GGUF manifest model_files entries must be objects")
            raw.pop("path", None)
    raw_tensors = stripped.get("tensors")
    if not isinstance(raw_tensors, list):
        raise ValueError("GGUF tensor manifest must contain tensors")
    for raw in raw_tensors:
        if not isinstance(raw, dict):
            raise ValueError("GGUF tensor manifest entries must be objects")
        raw.pop("model_file", None)
        raw.pop("f32_path", None)
    stripped_records = _manifest_records(stripped)
    if MerkleTree(
        [tensor_leaf_bytes(record) for record in stripped_records]
    ).root.hex() != expected_root:
        raise RuntimeError("stripping runtime paths changed tensor commitments")
    return stripped


def gguf_tensor_opening(manifest: Mapping[str, Any], tensor_name: str) -> dict[str, Any]:
    """Return a Merkle opening for ``tensor_name`` inside a manifest."""

    records = _manifest_records(manifest)
    leaves = [tensor_leaf_bytes(record) for record in records]
    tree = MerkleTree(leaves)
    expected_root = str(manifest.get("tensor_manifest_root", ""))
    if expected_root and tree.root.hex() != expected_root:
        raise RuntimeError("GGUF tensor manifest root mismatch")
    selected_index = -1
    selected_record: dict[str, Any] | None = None
    for index, record in enumerate(records):
        if str(record["name"]) == str(tensor_name):
            selected_index = index
            selected_record = record
            break
    if selected_record is None:
        raise KeyError(f"GGUF tensor manifest does not contain tensor {tensor_name!r}")
    return {
        "version": GGUF_TENSOR_MANIFEST_VERSION,
        "tensor_manifest_root": tree.root.hex(),
        "tensor_manifest_count": len(records),
        "tensor_leaf_index": selected_index,
        "tensor_leaf": tensor_leaf_body(selected_record),
        "tensor_membership_path": _path_to_dict(tree.get_path(selected_index)),
    }


def verify_gguf_tensor_opening(
    opening: Mapping[str, Any],
    *,
    expected_root: str,
    expected_tensor_name: str,
    expected_raw_sha256: str | None = None,
    expected_f32_sha256: str | None = None,
    expected_proof_i8_merkle_root: str | None = None,
    expected_proof_i8_chunk_size: int | None = None,
    expected_n_bytes: int | None = None,
    expected_f32_n_bytes: int | None = None,
    expected_expert_index: int | None = None,
) -> bool:
    """Verify a GGUF tensor opening against a selected trace raw hash."""

    if int(opening.get("version", 0)) != GGUF_TENSOR_MANIFEST_VERSION:
        return False
    if len(expected_root) != 64:
        return False
    if str(opening.get("tensor_manifest_root", "")) not in ("", str(expected_root)):
        return False
    leaf = opening.get("tensor_leaf")
    if not isinstance(leaf, Mapping):
        return False
    body = tensor_leaf_body(leaf)
    if str(body["name"]) != str(expected_tensor_name):
        return False
    if expected_raw_sha256 is not None and str(expected_raw_sha256):
        if str(body["raw_sha256"]) != str(expected_raw_sha256):
            return False
    elif expected_raw_sha256 == "":
        return False
    if expected_f32_sha256 is not None and str(body["f32_sha256"]) != str(
        expected_f32_sha256
    ):
        return False
    if expected_proof_i8_merkle_root is not None:
        if expected_expert_index is not None:
            # MoE expert witness: the receipt's weight root must be the
            # committed root of exactly the routed expert's plane.
            plane_roots = body.get("proof_i8_expert_merkle_roots") or []
            if not (0 <= int(expected_expert_index) < len(plane_roots)):
                return False
            if str(plane_roots[int(expected_expert_index)]) != str(
                expected_proof_i8_merkle_root
            ):
                return False
        elif str(body.get("proof_i8_merkle_root", "")) != str(
            expected_proof_i8_merkle_root
        ):
            return False
    if expected_proof_i8_chunk_size is not None and int(
        body.get("proof_i8_chunk_size", -1)
    ) != int(expected_proof_i8_chunk_size):
        return False
    if expected_n_bytes is not None and int(body["n_bytes"]) != int(expected_n_bytes):
        return False
    if expected_f32_n_bytes is not None and int(body["f32_nbytes"]) != int(
        expected_f32_n_bytes
    ):
        return False
    try:
        path = _path_from_payload(
            list(opening.get("tensor_membership_path", [])),
            leaf_index=int(opening.get("tensor_leaf_index", -1)),
        )
    except Exception:
        return False
    return verify_merkle_path(
        bytes.fromhex(expected_root),
        tensor_leaf_bytes(body),
        path,
    )


def quantize_proof_i8(array: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic proof-domain int8 quantization used by GGML proofs."""

    f32 = np.asarray(array, dtype=np.float32)
    max_abs = float(np.max(np.abs(f32))) if f32.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    quantized = np.clip(np.rint(f32 / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), scale


# -- proof-weight cache ------------------------------------------------------
#
# Content-addressed cache of proof-domain weights, keyed by the hashes the
# manifest leaves already commit to (proof_i8_sha256 / f32_sha256). Proving no
# longer needs the GGUF on disk once a tensor is cached: the per-request
# "open the model file and dequantize the sampled layer" cost (measured
# 3.5-60s) becomes a hash-verified blob read. Because blobs are addressed by
# committed content hashes, they are also safely distributable between mesh
# members — a worker can fetch its slice from any peer and verify it locally.
# Verifier semantics are untouched: the sumcheck still opens against the same
# committed proof_i8 Merkle roots.

PROOF_WEIGHT_CACHE_DIR_ENV = "VERALLM_PROOF_WEIGHT_CACHE_DIR"
PROOF_WEIGHT_CACHE_OFF_ENV = "VERALLM_PROOF_WEIGHT_CACHE_OFF"
PROOF_F32_EXACT_ENV = "VERALLM_PROOF_F32_EXACT"
_F32_BLOB_CACHE_MAX_BYTES = 64 * 1024 * 1024  # biases and norms, not matrices
_PROOF_I8_META_MAX_BYTES = 16 * 1024

logger = logging.getLogger(__name__)


def _is_final_projection_name(name: str) -> bool:
    lower = str(name).lower()
    return (
        lower == "output.weight"
        or lower == "lm_head.weight"
        or lower.endswith(".output.weight")
        or lower.endswith(".lm_head.weight")
    )


@lru_cache(maxsize=1)
def _gpu_merkle_hash_available() -> bool:
    """True when weight-Merkle trees build on GPU (cheap per prove).

    Probes with a real kernel launch, not just symbol presence: a wheel
    built without this GPU's arch imports fine and then fails every launch
    with "no kernel image", silently falling back to the 6-200 s CPU
    build. That exact failure produced an 11 GB cache on a 2x5090 box.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from zkllm.crypto import merkle as _merkle

        if not getattr(_merkle, "_HAS_CUDA_BLAKE3", False):
            return False
        from zkllm.cuda import zkllm_native as _native

        probe = torch.zeros(1, 256, dtype=torch.int8, device="cuda")
        _native.cuda_blake3_merkle_leaves(probe.contiguous(), 128, 0)
        return True
    except Exception:
        return False


PROOF_WEIGHT_CACHE_PROFILE_ENV = "VERALLM_PROOF_WEIGHT_CACHE_PROFILE"


def proof_weight_cache_profile() -> str:
    """"compact" or "full": how much the proof-weight cache persists.

    compact persists ONLY the final-projection i8 bytes (their CPU derive
    is ~10 s and every decode audit draws them); everything else derives
    from the GGUF on demand (0.2-1 s, sha-verified) and trees build on
    GPU per prove. full additionally persists every i8 blob and every
    weight-Merkle tree — the right trade ONLY where trees must build on
    CPU (6-200 s per tensor), i.e. no working GPU hash kernel.

    Default is auto: compact when the GPU hash actually works, full
    otherwise. Rationale: the full profile measured 11 GB for a 4.4 GB
    model (2.5x), which does not scale to multi-model boxes.
    """

    configured = (
        os.environ.get(PROOF_WEIGHT_CACHE_PROFILE_ENV, "").strip().lower()
    )
    if configured in ("compact", "full"):
        return configured
    if _gpu_merkle_hash_available():
        return "compact"
    _warn_full_profile_fallback_once()
    return "full"


@lru_cache(maxsize=1)
def _warn_full_profile_fallback_once() -> None:
    # The full profile persists a near-model-sized i8 cache and pays
    # 6-200 s CPU tree builds per tensor. On a CUDA box that is almost
    # always a zkllm extension built without this GPU's arch (the kernel
    # probe fails with "no kernel image"), which a per-box rebuild fixes.
    logger.warning(
        "GPU BLAKE3 Merkle kernel unavailable; proof-weight cache falls "
        "back to the FULL profile (near-model-sized cache, slow CPU tree "
        "builds). If this box has a CUDA GPU, rebuild the zkllm extension "
        "for its architecture: python zkllm/cuda/build.py"
    )


def _should_persist_i8(record: Mapping[str, Any]) -> bool:
    if proof_weight_cache_profile() == "full":
        return True
    return _is_final_projection_name(str(record.get("name", "")))


def should_persist_weight_merkle(tensor_name: str) -> bool:
    """Whether a built tree earns a .wmerkle blob under the active profile.

    compact: only the final projection. Its 4M+ leaves make even the
    GPU-leaf build ~22 s (Python-side tree assembly dominates), and every
    decode audit draws it. All other tensors rebuild per draw in 0.8-3 s
    with GPU leaf hashing, which is not worth 2.9 GB of blobs.
    """

    if proof_weight_cache_profile() == "full":
        return True
    return _is_final_projection_name(tensor_name)


def proof_weight_cache_dir() -> Path | None:
    """Cache root, or None when disabled/uncreatable."""

    if os.environ.get(PROOF_WEIGHT_CACHE_OFF_ENV, "").strip() == "1":
        return None
    configured = os.environ.get(PROOF_WEIGHT_CACHE_DIR_ENV, "").strip()
    root = Path(configured) if configured else Path.home() / ".verathos" / "proof-weight-cache"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return root


def _cache_blob_path(root: Path, sha256_hex: str, suffix: str) -> Path:
    return root / sha256_hex[:2] / (sha256_hex + suffix)


def _read_verified_blob(root: Path, sha256_hex: str, suffix: str) -> bytes | None:
    path = _cache_blob_path(root, sha256_hex, suffix)
    if not path.is_file():
        return None
    # Refresh mtime so the LRU cap keeps blobs that are actually being used
    # (proofs sample the same hot tensors repeatedly) and evicts cold ones.
    try:
        os.utime(path, None)
    except OSError:
        pass
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256_hex:
        # Corrupt cache entry: drop it and fall back to the source of truth.
        try:
            path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"proof-weight cache blob corrupt for {sha256_hex}")
    return raw


PROOF_WEIGHT_CACHE_MAX_GB_ENV = "VERALLM_PROOF_WEIGHT_CACHE_MAX_GB"
_cache_evict_lock = threading.Lock()


PROOF_WEIGHT_CACHE_DEFAULT_MAX_GB = 16.0


def _enforce_cache_cap(root: Path) -> None:
    """Evict least-recently-used blobs when the cache exceeds its cap.

    Blobs bank per proof-sampled tensor; on a big model that set is bounded by
    usage but unbounded in absolute size (a 70B's i8 blobs alone exceed a
    laptop's free disk if every tensor is eventually sampled). LRU by mtime
    keeps the hot working set and drops cold blobs — a re-fetch/rebuild just
    re-banks them. BOUNDED BY DEFAULT (16 GB): an unbounded cache silently
    grew toward model size wherever the GPU hash kernel was missing and the
    profile fell back to "full".
    The compact profile needs a fraction of the default; operators who
    deliberately pre-build full caches raise the cap (or set 0 for
    unbounded) via VERALLM_PROOF_WEIGHT_CACHE_MAX_GB.
    """
    raw_cap = os.environ.get(PROOF_WEIGHT_CACHE_MAX_GB_ENV, "").strip()
    try:
        cap_gb = (
            float(raw_cap) if raw_cap else PROOF_WEIGHT_CACHE_DEFAULT_MAX_GB
        )
    except ValueError:
        cap_gb = PROOF_WEIGHT_CACHE_DEFAULT_MAX_GB
    if cap_gb <= 0:
        # Explicit opt-out to unbounded stays available for owner boxes
        # that pre-build full caches deliberately.
        return
    cap_bytes = int(cap_gb * (1024 ** 3))
    with _cache_evict_lock:
        files = []
        total = 0
        for p in root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
                total += st.st_size
        if total <= cap_bytes:
            return
        files.sort(key=lambda x: x[0])  # oldest first
        for _mtime, size, p in files:
            if total <= cap_bytes:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass


def _write_blob(root: Path, sha256_hex: str, suffix: str, raw: bytes) -> None:
    path = _cache_blob_path(root, sha256_hex, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)  # atomic per-file publish
    _enforce_cache_cap(root)


PROOF_BLOB_PEERS_ENV = "VERALLM_PROOF_BLOB_PEERS"
PROOF_BLOB_AUTH_STATE_ENV = "VERALLM_PROOF_BLOB_AUTH_STATE"


def _proof_blob_peers() -> list[str]:
    raw = os.environ.get(PROOF_BLOB_PEERS_ENV, "").strip()
    return [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]


def _fetch_blob_from_peers(
    sha256_hex: str,
    suffix: str,
    *,
    max_bytes: int | None = None,
) -> bytes | None:
    """Fetch a content-addressed blob from mesh peers; sha-verified locally.

    This is what makes a GGUF-less member possible: its proof weights arrive
    as committed-hash-addressed blobs from any peer that has them (typically
    the driver), verified here — the peer is untrusted transport.
    """

    import urllib.request as _urllib_request

    internal_auth_secret = ""
    auth_state = os.environ.get(PROOF_BLOB_AUTH_STATE_ENV, "").strip()
    if auth_state:
        from verallm.mesh.state import load_mesh_state, state_internal_auth_secret

        internal_auth_secret = state_internal_auth_secret(load_mesh_state(auth_state))

    # Generous default: a peer that builds blobs on demand (dequantizing one
    # 72B-class tensor from its GGUF) can take a minute before first byte.
    timeout_s = float(os.environ.get("VERALLM_PROOF_BLOB_FETCH_TIMEOUT", "180"))
    for peer in _proof_blob_peers():
        url = f"{peer}/v1/mesh/proof-blob/{sha256_hex}{suffix}"
        try:
            request: str | _urllib_request.Request = url
            if internal_auth_secret:
                from verallm.mesh.http_auth import sign_internal_http_request

                path = f"/v1/mesh/proof-blob/{sha256_hex}{suffix}"
                request = _urllib_request.Request(
                    url,
                    headers=sign_internal_http_request(
                        secret=internal_auth_secret,
                        method="GET",
                        path=path,
                        body=b"",
                    ),
                )
            with _urllib_request.urlopen(request, timeout=timeout_s) as resp:
                raw = (
                    resp.read(int(max_bytes) + 1)
                    if max_bytes is not None
                    else resp.read()
                )
        except Exception:
            continue
        if max_bytes is not None and len(raw) > int(max_bytes):
            logger.warning("peer %s served an oversized blob for %s", peer, sha256_hex)
            continue
        if suffix in (".i8", ".f32") and hashlib.sha256(raw).hexdigest() != sha256_hex:
            logger.warning("peer %s served a non-matching blob for %s", peer, sha256_hex)
            continue
        return raw
    return None


def load_cached_proof_i8(record: Mapping[str, Any]) -> tuple[np.ndarray, float] | None:
    """Return (proof_i8, scale) from the cache (or peers), or None on miss."""

    sha = str(record.get("proof_i8_sha256", ""))
    shape_values = [int(dim) for dim in (record.get("shape") or [])]
    expected_nbytes = int(record.get("proof_i8_nbytes", 0) or 0)
    if expected_nbytes <= 0 and shape_values:
        expected_nbytes = int(np.prod(shape_values))
    root = proof_weight_cache_dir()
    if not sha or root is None:
        return None
    try:
        raw = _read_verified_blob(root, sha, ".i8")
    except RuntimeError:
        raw = None
    meta_path = _cache_blob_path(root, sha, ".i8.json")
    meta_raw = None
    if meta_path.is_file():
        try:
            if meta_path.stat().st_size <= _PROOF_I8_META_MAX_BYTES:
                meta_raw = meta_path.read_text()
        except OSError:
            meta_raw = None
    if raw is None or meta_raw is None:
        fetched = _fetch_blob_from_peers(
            sha,
            ".i8",
            max_bytes=expected_nbytes or None,
        )
        fetched_meta = _fetch_blob_from_peers(
            sha,
            ".i8.json",
            max_bytes=_PROOF_I8_META_MAX_BYTES,
        )
        if fetched is None or fetched_meta is None:
            return None
        raw, meta_raw = fetched, fetched_meta.decode("utf-8")
        try:
            _write_blob(root, sha, ".i8", raw)
            _write_blob_text(root, sha, ".i8.json", meta_raw)
        except OSError:
            pass
    # The .i8 bytes are hash-verified against the committed sha, but the
    # .i8.json sidecar cannot be (its own hash isn't in the manifest): treat
    # it as UNTRUSTED. Shape comes from the committed manifest record when
    # present, the JSON parse is guarded, and the scale is bounds-checked —
    # a garbage sidecar must degrade to a cache miss, not a crash or a
    # silently wrong recompute reference.
    try:
        meta = json.loads(meta_raw)
        rec_shape = record.get("shape")
        if rec_shape is not None and len(rec_shape) >= 2:
            # Proof-domain W is [dims[0], dims[1]] of the committed tensor
            # shape (see proof_f32_weight_matrix_from_gguf_f32).
            shape = (int(rec_shape[0]), int(rec_shape[1]))
        else:
            shape = tuple(int(d) for d in meta["shape"])
        scale = float(meta["scale"])
    except (ValueError, KeyError, TypeError):
        return None
    if not (math.isfinite(scale) and scale > 0.0):
        return None
    if expected_nbytes and len(raw) != expected_nbytes:
        return None
    arr = np.frombuffer(raw, dtype=np.int8)
    if arr.size != int(np.prod(shape)):
        return None
    return np.ascontiguousarray(arr.reshape(shape)), scale


_WEIGHT_MERKLE_BLOB_SUFFIX = ".wmerkle"


def _serialize_weight_merkle_cache(cache_data: Mapping[str, Any]) -> bytes:
    """Pack FlatWeightMerkle.get_cache_data() as header-json + raw tree bytes.

    No pickle: a length-prefixed JSON header (all ints / int lists) followed
    by the tree's raw node bytes. Deserialization rebuilds via from_cached and
    the caller re-verifies the root against the signed manifest, so a corrupt
    or hostile blob cannot be trusted into a proof.
    """
    tree_data = bytes(cache_data["tree_data"])
    header = {
        "num_rows": int(cache_data["num_rows"]),
        "num_cols": int(cache_data["num_cols"]),
        "chunk_size": int(cache_data["chunk_size"]),
        "total_elements": int(cache_data["total_elements"]),
        "num_chunks": int(cache_data["num_chunks"]),
        "bytes_per_element": int(cache_data["bytes_per_element"]),
        "dtype_code": str(cache_data["dtype_code"]),
        "bytes_per_chunk": int(cache_data["bytes_per_chunk"]),
        "tree_num_leaves": int(cache_data["tree_num_leaves"]),
        "tree_level_offsets": [int(v) for v in cache_data["tree_level_offsets"]],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(header_bytes).to_bytes(4, "big") + header_bytes + tree_data


def _deserialize_weight_merkle_cache(raw: bytes) -> dict[str, Any]:
    if len(raw) < 4:
        raise ValueError("weight merkle blob truncated")
    header_len = int.from_bytes(raw[:4], "big")
    header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
    tree_data = raw[4 + header_len :]
    # from_cached stores these fields without materializing anything, so a
    # blob whose header disagrees with its tree bytes only fails LATER,
    # inside native proof code. Reject inconsistent blobs here where the
    # caller treats the error as a cache miss and rebuilds.
    for key in (
        "num_rows",
        "num_cols",
        "chunk_size",
        "total_elements",
        "num_chunks",
        "bytes_per_element",
        "bytes_per_chunk",
        "tree_num_leaves",
    ):
        if type(header.get(key)) is not int or header[key] <= 0:
            raise ValueError(f"weight merkle blob header {key} is invalid")
    if header["num_rows"] * header["num_cols"] != header["total_elements"]:
        raise ValueError("weight merkle blob shape disagrees with element count")
    expected_chunks = -(-header["total_elements"] // header["chunk_size"])
    if header["num_chunks"] != expected_chunks:
        raise ValueError("weight merkle blob chunk count disagrees with geometry")
    if header["tree_num_leaves"] != header["num_chunks"]:
        raise ValueError("weight merkle blob leaf count disagrees with chunks")
    if len(tree_data) % 32 != 0:
        raise ValueError("weight merkle blob tree bytes are not whole nodes")
    node_count = len(tree_data) // 32
    # Level offsets list each level's start node plus a final end sentinel
    # equal to the node count; the first level is the leaves.
    offsets = header.get("tree_level_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) < 2
        or any(type(v) is not int for v in offsets)
        or offsets[0] != 0
        or any(b <= a for a, b in zip(offsets, offsets[1:]))
        or offsets[-1] != node_count
        or offsets[1] != header["tree_num_leaves"]
    ):
        raise ValueError("weight merkle blob level offsets disagree with tree bytes")
    return {
        "num_rows": header["num_rows"],
        "num_cols": header["num_cols"],
        "chunk_size": header["chunk_size"],
        "total_elements": header["total_elements"],
        "num_chunks": header["num_chunks"],
        "bytes_per_element": header["bytes_per_element"],
        "dtype_code": str(header["dtype_code"]),
        "bytes_per_chunk": header["bytes_per_chunk"],
        "tree_data": tree_data,
        "tree_num_leaves": header["tree_num_leaves"],
        "tree_level_offsets": header["tree_level_offsets"],
    }


def load_cached_weight_merkle(sha256_hex: str):
    """Load a persisted FlatWeightMerkle by its weight proof_i8_sha256.

    Returns the reconstructed tree (raw bytes not resident; get_proof needs a
    W_tensor) or None. The caller MUST re-verify merkle.root against the
    signed manifest root before use.
    """
    from zkllm.crypto.merkle import FlatWeightMerkle

    root = proof_weight_cache_dir()
    if not sha256_hex or root is None:
        return None
    path = _cache_blob_path(root, sha256_hex, _WEIGHT_MERKLE_BLOB_SUFFIX)
    if not path.is_file():
        return None
    try:
        os.utime(path, None)  # LRU touch
    except OSError:
        pass
    try:
        cache_data = _deserialize_weight_merkle_cache(path.read_bytes())
        return FlatWeightMerkle.from_cached(cache_data)
    except Exception as exc:
        # Broad on purpose: a truncated or foreign blob can carry a valid
        # JSON header whose counts disagree with the tree bytes, and
        # from_cached then fails in native code (pybind RuntimeError, e.g.
        # "cannot create std::vector larger than max_size()"). Any blob
        # that does not reconstruct is a cache miss that rebuilds and
        # re-persists; letting the exception escape failed the audit as a
        # receipt 500 instead (observed on the LM-head tensor).
        logger.warning("weight merkle cache load failed (%s); rebuilding", exc)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def drop_cached_weight_merkle(sha256_hex: str) -> None:
    """Remove a persisted weight-merkle blob (native-prover retry path)."""

    root = proof_weight_cache_dir()
    if not sha256_hex or root is None:
        return
    try:
        _cache_blob_path(root, sha256_hex, _WEIGHT_MERKLE_BLOB_SUFFIX).unlink(
            missing_ok=True
        )
    except OSError as exc:
        logger.warning("weight merkle cache drop failed: %s", exc)


def drop_cached_proof_i8(record: Mapping[str, Any]) -> None:
    """Remove a persisted proof-i8 blob + sidecar (native-prover retry path)."""

    sha = str(record.get("proof_i8_sha256", ""))
    root = proof_weight_cache_dir()
    if not sha or root is None:
        return
    for suffix in (".i8", ".i8.json"):
        try:
            _cache_blob_path(root, sha, suffix).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("proof_i8 cache drop failed: %s", exc)


def store_cached_weight_merkle(sha256_hex: str, merkle) -> None:
    """Persist a built FlatWeightMerkle keyed by its weight proof_i8_sha256."""

    root = proof_weight_cache_dir()
    if not sha256_hex or root is None:
        return
    try:
        raw = _serialize_weight_merkle_cache(merkle.get_cache_data())
        _write_blob(root, sha256_hex, _WEIGHT_MERKLE_BLOB_SUFFIX, raw)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("weight merkle cache write failed: %s", exc)


def store_cached_proof_i8(
    record: Mapping[str, Any], proof_i8: np.ndarray, scale: float
) -> None:
    """Persist a freshly built proof-domain matrix; refuses non-canonical bytes."""

    sha = str(record.get("proof_i8_sha256", ""))
    root = proof_weight_cache_dir()
    if not sha or root is None:
        return
    raw = np.ascontiguousarray(proof_i8, dtype=np.int8).tobytes(order="C")
    if hashlib.sha256(raw).hexdigest() != sha:
        # Never poison the cache: the manifest's committed hash is the truth.
        logger.warning("proof_i8 bytes do not match manifest hash; not caching")
        return
    try:
        _write_blob(root, sha, ".i8", raw)
        meta = {"scale": float(scale), "shape": [int(d) for d in proof_i8.shape]}
        _write_blob_text(root, sha, ".i8.json", json.dumps(meta))
    except OSError as exc:
        logger.warning("proof-weight cache write failed: %s", exc)


def _write_blob_text(root: Path, sha256_hex: str, suffix: str, text: str) -> None:
    path = _cache_blob_path(root, sha256_hex, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_cached_f32_blob(record: Mapping[str, Any]) -> np.ndarray | None:
    """Bounded f32 cache, keyed by the committed f32_sha256."""

    sha = str(record.get("f32_sha256", ""))
    expected_nbytes = int(record.get("f32_nbytes", 0) or 0)
    if expected_nbytes > _F32_BLOB_CACHE_MAX_BYTES:
        return None
    root = proof_weight_cache_dir()
    if not sha or root is None:
        return None
    try:
        raw = _read_verified_blob(root, sha, ".f32")
    except RuntimeError:
        raw = None
    if raw is None:
        raw = _fetch_blob_from_peers(
            sha,
            ".f32",
            max_bytes=expected_nbytes or _F32_BLOB_CACHE_MAX_BYTES,
        )
        if raw is None:
            return None
        try:
            _write_blob(root, sha, ".f32", raw)
        except OSError:
            pass
    if expected_nbytes and len(raw) != expected_nbytes:
        return None
    shape = tuple(int(d) for d in record.get("shape", []))
    arr = np.frombuffer(raw, dtype=np.float32)
    if shape and arr.size != int(np.prod(shape)):
        return None
    return np.ascontiguousarray(arr.reshape(shape) if shape else arr)


def store_cached_f32_blob(record: Mapping[str, Any], f32: np.ndarray) -> None:
    sha = str(record.get("f32_sha256", ""))
    root = proof_weight_cache_dir()
    if not sha or root is None:
        return
    raw = np.ascontiguousarray(f32, dtype=np.float32).tobytes(order="C")
    if len(raw) > _F32_BLOB_CACHE_MAX_BYTES:
        return  # matrices live in the i8 cache; f32 blobs are for small tensors
    if hashlib.sha256(raw).hexdigest() != sha:
        return
    try:
        _write_blob(root, sha, ".f32", raw)
    except OSError as exc:
        logger.warning("proof-weight cache write failed: %s", exc)


def build_proof_weight_cache(
    manifest: Mapping[str, Any],
    *,
    tensor_names: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Populate the cache for every (selected) tensor in one pass.

    This is the one-time per-machine cost (the same dequant work manifest
    generation already does); afterwards proving never opens the GGUF.

    Both blob kinds are built here: the i8 weight bytes AND the persisted
    weight Merkle tree. The tree matters as much as the bytes: a hard audit
    whose beacon draws a tensor with no ``.wmerkle`` blob rebuilds the tree
    in-process, which measured 20-30 s per draw on a 2x5090 box and made
    hard-audit latency a lottery (5 s on a persisted draw, 30-40 s
    otherwise) until every tensor had been drawn once.
    """

    from contextlib import ExitStack

    from zkllm.crypto.merkle import FlatWeightMerkle, bulk_chunk_hashing

    stats = {
        "cached": 0,
        "skipped": 0,
        "small_f32": 0,
        "merkle": 0,
        "pruned": 0,
        "profile": proof_weight_cache_profile(),
    }
    wanted = set(tensor_names or [])
    cache_root = proof_weight_cache_dir()
    # Bulk marking: this warm builds hundreds of trees and may run in the
    # background beside serving traffic; its hashing must go through the
    # shared bulk pool so per-draw proof trees never queue behind it. The
    # flag is thread-local and MUST clear on exit: a per-draw fallback
    # calls this same function for a single tensor from a serving thread.
    bulk_stack = ExitStack()
    bulk_stack.enter_context(bulk_chunk_hashing())
    try:
        return _build_proof_weight_cache_inner(
            manifest,
            stats=stats,
            wanted=wanted,
            cache_root=cache_root,
            progress=progress,
            flat_weight_merkle=FlatWeightMerkle,
        )
    finally:
        bulk_stack.close()


def _build_proof_weight_cache_inner(
    manifest: Mapping[str, Any],
    *,
    stats: dict[str, int],
    wanted: set,
    cache_root,
    progress,
    flat_weight_merkle,
) -> dict[str, int]:
    FlatWeightMerkle = flat_weight_merkle
    for record in _manifest_records(manifest):
        name = str(record.get("name", ""))
        if wanted and name not in wanted:
            continue
        if record.get("proof_i8_sha256"):
            sha = str(record.get("proof_i8_sha256", ""))
            root_hex = str(record.get("proof_i8_merkle_root", ""))
            chunk_size = int(record.get("proof_i8_chunk_size", 0) or 0)
            persist_i8 = _should_persist_i8(record)
            persist_tree = should_persist_weight_merkle(name)
            if not (persist_i8 or persist_tree):
                # Compact profile: this tensor derives (sha-verified) and
                # tree-builds (GPU leaves) per draw. Reclaim any blobs a
                # full-profile run banked for it.
                if cache_root is not None:
                    for suffix in (".i8", ".i8.json", _WEIGHT_MERKLE_BLOB_SUFFIX):
                        blob = _cache_blob_path(cache_root, sha, suffix)
                        if blob.is_file():
                            try:
                                blob.unlink()
                                stats["pruned"] += 1
                            except OSError:
                                pass
                continue
            loaded = load_cached_proof_i8(record)
            if loaded is not None:
                stats["skipped"] += 1
            elif persist_i8:
                # Route through the loader so build+verify+store share one path.
                proof_i8_weight_matrix_from_manifest(manifest, name)
                stats["cached"] += 1
                if progress is not None:
                    progress(f"cached {name}")
            if (
                persist_tree
                and sha
                and root_hex
                and chunk_size > 0
                and load_cached_weight_merkle(sha) is None
            ):
                if loaded is not None:
                    w_i8 = loaded[0]
                else:
                    w_i8 = proof_i8_weight_matrix_from_manifest(manifest, name)
                merkle = FlatWeightMerkle(
                    torch.from_numpy(np.ascontiguousarray(w_i8)),
                    chunk_size,
                    store_raw=False,
                )
                if merkle.root.hex() != root_hex:
                    raise RuntimeError(
                        f"weight merkle root mismatch for {name}: cache "
                        "build disagrees with the signed manifest"
                    )
                store_cached_weight_merkle(sha, merkle)
                stats["merkle"] += 1
                if progress is not None:
                    progress(f"merkle {name}")
        elif int(record.get("f32_nbytes", 0)) <= _F32_BLOB_CACHE_MAX_BYTES:
            if load_cached_f32_blob(record) is None:
                f32_tensor_from_manifest(manifest, name)
                stats["small_f32"] += 1
    return stats


def build_proof_blob_for_committed_sha(
    manifest: Mapping[str, Any],
    sha256_hex: str,
    suffix: str,
) -> bool:
    """Materialize one manifest-committed proof blob into the local cache.

    This is the server-side half of file-less proof-weight distribution.  A
    member requests a content hash from the model-owning peer; the peer finds
    the corresponding manifest record and derives only that committed tensor
    (or routed MoE expert plane) from its local GGUF.  The normal loaders
    verify the committed Merkle root/hash before publishing the cache blob.

    ``True`` means the digest is present in the manifest and a build was
    attempted.  Callers must still require the requested cache path to exist:
    for example, large exact-f32 tensors intentionally exceed the bounded f32
    cache and therefore remain unavailable to file-less members.
    """

    digest = str(sha256_hex).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return False
    if suffix not in {".i8", ".i8.json", ".f32"}:
        return False

    for record in _manifest_records(manifest):
        name = str(record.get("name", ""))
        if suffix in {".i8", ".i8.json"}:
            if str(record.get("proof_i8_sha256", "")).lower() == digest:
                proof_i8_weight_matrix_from_manifest(manifest, name)
                return True

            # MUL_MAT_ID commits one proof-domain matrix per routed expert.
            # These plane hashes deliberately replace the whole-tensor hash
            # on 3-D MoE records, so they must be indexed explicitly here.
            expert_hashes = [
                str(item).lower()
                for item in (record.get("proof_i8_expert_sha256") or [])
            ]
            try:
                expert = expert_hashes.index(digest)
            except ValueError:
                expert = -1
            if expert >= 0:
                proof_i8_expert_plane_from_manifest(manifest, name, expert)
                return True
        elif str(record.get("f32_sha256", "")).lower() == digest:
            # Exact f32 transport is intentionally limited to small tensors.
            # Do not dequantize a multi-GB matrix (or a whole MoE tensor) only
            # for store_cached_f32_blob() to reject it after the allocation.
            f32_nbytes = int(record.get("f32_nbytes", 0) or 0)
            if not (0 < f32_nbytes <= _F32_BLOB_CACHE_MAX_BYTES):
                return False
            f32_tensor_from_manifest(manifest, name)
            return True
    return False


def proof_i8_weight_matrix_from_gguf_f32(
    f32_data: np.ndarray,
    shape: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return the proof-domain W matrix for a GGUF tensor.

    GGML MUL_MAT traces expose src0 as [K, N] metadata while the dumped bytes
    are consumed as [N, K] rows before transposing into proof-domain W[K, N].
    The manifest root must use the same deterministic transform as
    ``GgmlMulMatTrace.load_matrices``.
    """

    w_f32 = proof_f32_weight_matrix_from_gguf_f32(f32_data, shape)
    proof_i8, _scale = quantize_proof_i8(w_f32)
    return proof_i8


def proof_f32_weight_matrix_from_gguf_f32(
    f32_data: np.ndarray,
    shape: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return the proof-domain f32 W matrix for a GGUF tensor."""

    dims = [int(item) for item in shape]
    if len(dims) < 2:
        raise ValueError("GGUF proof weight root requires a matrix tensor")
    k, n = int(dims[0]), int(dims[1])
    expected = k * n
    flat = np.asarray(f32_data, dtype=np.float32).reshape(-1)
    if flat.size != expected:
        raise ValueError("GGUF tensor f32 data size does not match first two shape dims")
    weight_rows = flat.reshape(n, k)
    return np.ascontiguousarray(weight_rows.T)


def _manifest_record_for_tensor(
    manifest: Mapping[str, Any],
    tensor_name: str,
) -> dict[str, Any]:
    records = _manifest_records(manifest)
    selected = next(
        (record for record in records if str(record.get("name", "")) == str(tensor_name)),
        None,
    )
    if selected is None:
        raise KeyError(f"GGUF tensor manifest does not contain tensor {tensor_name!r}")
    return selected


def _resolve_local_model_file(candidate: str) -> str:
    """Resolve a manifest-recorded model path on THIS machine.

    Manifests record the absolute GGUF path of the machine that BUILT them,
    and manifests travel between machines (owner onboarding, cache
    distribution). A foreign absolute path silently breaks every local
    weight derive - the warmer then fails all blobs and cold audits run on
    fragile fallbacks. Falling back to the canonical local layout by
    model-dir/basename is safe: every derived byte is verified against the
    committed sha/Merkle root downstream, so a wrong file fails closed.
    """

    if not candidate:
        return candidate
    path = Path(candidate)
    if path.is_file():
        return candidate
    local = Path.home() / ".verathos" / "mesh-models"
    tail = Path(*path.parts[-2:]) if len(path.parts) >= 2 else Path(path.name)
    for resolved in (local / tail, *sorted(local.glob(f"*/{path.name}"))):
        if resolved.is_file():
            return str(resolved)
    return candidate


def _model_file_for_manifest_record(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> str:
    model_file = str(record.get("model_file", ""))
    if model_file:
        return _resolve_local_model_file(model_file)
    model_files = manifest.get("model_files", [])
    if isinstance(model_files, list) and model_files:
        index = int(record.get("model_file_index", 0))
        if 0 <= index < len(model_files):
            item = model_files[index]
            if isinstance(item, Mapping) and str(item.get("path", "")):
                return _resolve_local_model_file(str(item["path"]))
    return _resolve_local_model_file(str(manifest.get("model_file", "")))


def _proof_f32_weight_matrix_from_record_cache(
    record: Mapping[str, Any],
) -> np.ndarray | None:
    f32_path = str(record.get("f32_path", ""))
    if not f32_path:
        return None
    path = Path(f32_path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    expected_hash = str(record.get("f32_sha256", ""))
    if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise RuntimeError("GGUF tensor f32 cache hash mismatch")
    f32_data = np.frombuffer(raw, dtype=np.float32)
    if f32_data.nbytes != int(record.get("f32_nbytes", f32_data.nbytes)):
        raise RuntimeError("GGUF tensor f32 cache byte length mismatch")
    return proof_f32_weight_matrix_from_gguf_f32(
        f32_data,
        [int(item) for item in record.get("shape", [])],
    )


def proof_i8_weight_matrix_from_manifest(
    manifest: Mapping[str, Any],
    tensor_name: str,
) -> np.ndarray:
    """Load and cache a tensor's deterministic proof-domain W matrix."""

    selected = _manifest_record_for_tensor(manifest, tensor_name)
    # Content-addressed cache first: sha match against the committed
    # proof_i8_sha256 makes the bytes canonical without re-reading the GGUF
    # (and without a redundant Merkle rebuild — the leaf binds both hashes to
    # the same bytes).
    cached = load_cached_proof_i8(selected)
    if cached is not None:
        return cached[0]
    cached_f32 = _proof_f32_weight_matrix_from_record_cache(selected)
    if cached_f32 is not None:
        proof_i8, scale = quantize_proof_i8(cached_f32)
        expected_sha = str(selected.get("proof_i8_sha256", ""))
        if expected_sha:
            derived_sha = hashlib.sha256(
                proof_i8.tobytes(order="C")
            ).hexdigest()
            if derived_sha != expected_sha:
                raise RuntimeError("GGUF proof weight sha mismatch")
        else:
            merkle = FlatWeightMerkle(
                torch.from_numpy(proof_i8),
                int(
                    selected.get(
                        "proof_i8_chunk_size", DEFAULT_W_MERKLE_CHUNK_SIZE
                    )
                ),
                store_raw=False,
            )
            if merkle.root.hex() != str(
                selected.get("proof_i8_merkle_root", "")
            ):
                raise RuntimeError("GGUF proof weight root mismatch")
        if _should_persist_i8(selected):
            store_cached_proof_i8(selected, proof_i8, scale)
        return proof_i8
    model_file = str(manifest.get("model_file", ""))
    model_file = _model_file_for_manifest_record(manifest, selected)
    if not model_file:
        raise RuntimeError("GGUF tensor manifest does not include model_file")
    root = str(selected.get("proof_i8_merkle_root", ""))
    if not root:
        raise RuntimeError(f"GGUF tensor {tensor_name!r} missing proof_i8_merkle_root")
    chunk_size = int(selected.get("proof_i8_chunk_size", DEFAULT_W_MERKLE_CHUNK_SIZE))
    proof_i8, scale = _proof_i8_weight_matrix_from_model_file(
        model_file,
        str(tensor_name),
        root,
        chunk_size,
        expected_sha256=str(selected.get("proof_i8_sha256", "")),
    )
    if _should_persist_i8(selected):
        store_cached_proof_i8(selected, proof_i8, scale)
    return proof_i8


def _dequant_single_expert_plane(
    gguf_mod: Any,
    tensor: Any,
    k_dim: int,
    n_dim: int,
    planes: int,
    expert: int,
) -> np.ndarray | None:
    """Dequantize ONLY one expert plane's contiguous byte slice.

    GGUF stores the 3-D expert tensor contiguously with ne[0]=k innermost,
    so plane ``expert`` is the block-aligned byte range
    ``[expert*plane_bytes, (expert+1)*plane_bytes)`` (K-quants require
    k % block_size == 0, so rows never straddle blocks). Slicing the
    memmap materializes ~1/N of the tensor instead of all of it — the
    difference between a multi-minute cold stall and tens of ms on the
    serving path. Returns None when the layout doesn't match expectations
    (caller falls back to the full-tensor path).
    """
    try:
        quant_sizes = getattr(gguf_mod, "GGML_QUANT_SIZES", None)
        if quant_sizes is None:
            quant_sizes = gguf_mod.constants.GGML_QUANT_SIZES
        block_size, type_size = quant_sizes[tensor.tensor_type]
        if k_dim % int(block_size) != 0:
            return None
        plane_bytes = (k_dim // int(block_size)) * int(type_size) * n_dim
        flat = tensor.data.reshape(-1)
        itemsize = int(flat.dtype.itemsize)
        if plane_bytes % itemsize != 0:
            return None
        if int(flat.size) * itemsize != plane_bytes * planes:
            return None
        per_plane = plane_bytes // itemsize
        # np.array() materializes just this slice of the memmap.
        chunk = np.array(flat[expert * per_plane : (expert + 1) * per_plane])
        out = gguf_mod.dequantize(chunk, tensor.tensor_type).astype(
            "float32", copy=False
        ).reshape(-1)
        if int(out.size) != k_dim * n_dim:
            return None
        return out
    except Exception:
        return None


def proof_i8_expert_plane_from_manifest(
    manifest: Mapping[str, Any],
    tensor_name: str,
    expert: int,
) -> tuple[np.ndarray, float]:
    """Load one MoE expert's committed proof-domain W plane.

    A MUL_MAT_ID trace proves a single routed expert's GEMM; its weight is
    plane ``expert`` of the 3-D expert tensor, committed per-plane in the
    manifest (proof_i8_expert_sha256 / _merkle_roots). Cache misses dequant
    ONLY the routed expert's contiguous byte slice (~1/N of the tensor) and
    verify it against that plane's committed Merkle root — mirroring the
    vLLM backend, where the per-request path never derives more than one
    expert (base.py's single-expert checkpoint readers). Falls back to a
    full-tensor dequant that banks every plane if the byte layout doesn't
    match expectations.
    """

    selected = _manifest_record_for_tensor(manifest, tensor_name)
    planes = int(selected.get("proof_i8_expert_planes", 0) or 0)
    if planes <= 0:
        raise RuntimeError(
            f"GGUF tensor {tensor_name!r} has no per-expert proof commitments "
            "(manifest predates MoE support; rebuild it)"
        )
    if not (0 <= int(expert) < planes):
        raise RuntimeError(f"expert index {expert} out of range (planes={planes})")
    shas = [str(x) for x in selected.get("proof_i8_expert_sha256", [])]
    roots = [str(x) for x in selected.get("proof_i8_expert_merkle_roots", [])]
    if len(shas) != planes or len(roots) != planes:
        raise RuntimeError(f"GGUF tensor {tensor_name!r} expert commitments malformed")
    shape = [int(d) for d in selected.get("shape", [])]
    plane_record = {
        "proof_i8_sha256": shas[int(expert)],
        "proof_i8_nbytes": int(selected.get("proof_i8_expert_nbytes", 0) or 0),
        "shape": shape[:2],
    }
    cached = load_cached_proof_i8(plane_record)
    if cached is not None:
        return cached
    model_file = _model_file_for_manifest_record(manifest, selected)
    if not model_file:
        raise RuntimeError("GGUF tensor manifest does not include model_file")
    try:
        import gguf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("GGUF proof weight loading requires the gguf package") from exc
    tensor = _gguf_tensor_from_model_file(model_file, tensor_name)
    k_dim, n_dim = shape[0], shape[1]
    plane_elems = k_dim * n_dim
    chunk_size = int(selected.get("proof_i8_chunk_size", DEFAULT_W_MERKLE_CHUNK_SIZE))
    plane_f32 = _dequant_single_expert_plane(
        gguf, tensor, k_dim, n_dim, planes, int(expert)
    )
    if plane_f32 is not None:
        w_f32 = proof_f32_weight_matrix_from_gguf_f32(plane_f32, [k_dim, n_dim])
        proof_i8, scale = quantize_proof_i8(w_f32)
        merkle = FlatWeightMerkle(
            torch.from_numpy(proof_i8), chunk_size, store_raw=False
        )
        if merkle.root.hex() != roots[int(expert)]:
            raise RuntimeError(
                f"GGUF expert plane root mismatch: {tensor_name!r} expert {expert}"
            )
        if proof_weight_cache_profile() == "full":
            try:
                store_cached_proof_i8(
                    {"proof_i8_sha256": shas[int(expert)]}, proof_i8, scale
                )
            except Exception:
                pass
        return proof_i8, float(scale)
    f32_data = gguf.dequantize(tensor.data, tensor.tensor_type).astype(
        "float32", copy=False
    ).reshape(-1)
    wanted: tuple[np.ndarray, float] | None = None
    for plane in range(planes):
        plane_f32 = f32_data[plane * plane_elems : (plane + 1) * plane_elems]
        w_f32 = proof_f32_weight_matrix_from_gguf_f32(plane_f32, [k_dim, n_dim])
        proof_i8, scale = quantize_proof_i8(w_f32)
        merkle = FlatWeightMerkle(
            torch.from_numpy(proof_i8), chunk_size, store_raw=False
        )
        if merkle.root.hex() != roots[plane]:
            raise RuntimeError(
                f"GGUF expert plane root mismatch: {tensor_name!r} expert {plane}"
            )
        if proof_weight_cache_profile() == "full":
            try:
                store_cached_proof_i8(
                    {"proof_i8_sha256": shas[plane]}, proof_i8, scale
                )
            except Exception:
                pass
        if plane == int(expert):
            wanted = (proof_i8, float(scale))
    assert wanted is not None
    return wanted


def proof_f32_weight_matrix_from_manifest(
    manifest: Mapping[str, Any],
    tensor_name: str,
    *,
    exact: bool | None = None,
) -> np.ndarray:
    """Load and cache a tensor's deterministic proof-domain f32 W matrix."""

    selected = _manifest_record_for_tensor(manifest, tensor_name)
    cached_f32 = _proof_f32_weight_matrix_from_record_cache(selected)
    if cached_f32 is not None:
        return cached_f32
    # Reconstruct from the cached proof-domain int8 (W ≈ i8 * scale) instead
    # of dequantizing the GGUF. This f32 only feeds the prover's local witness
    # layout/tolerance check (loose 0.08/0.04 windows, already absorbing Q4
    # noise); the verified object remains the committed int8 domain, so proof
    # semantics are unchanged. VERALLM_PROOF_F32_EXACT=1 forces the old path.
    use_exact = (
        os.environ.get(PROOF_F32_EXACT_ENV, "").strip() == "1"
        if exact is None
        else bool(exact)
    )
    if not use_exact:
        cached = load_cached_proof_i8(selected)
        if cached is not None:
            proof_i8, scale = cached
            return np.ascontiguousarray(proof_i8.astype(np.float32) * scale)
    else:
        # A file-less mesh member can still perform an exact fallback for
        # small/medium tensors: fetch the content-addressed f32 blob from the
        # model-owning peer and verify it against the manifest hash locally.
        f32_nbytes = int(selected.get("f32_nbytes", 0) or 0)
        if 0 < f32_nbytes <= _F32_BLOB_CACHE_MAX_BYTES:
            exact_tensor = f32_tensor_from_manifest(manifest, tensor_name)
            return proof_f32_weight_matrix_from_gguf_f32(
                exact_tensor,
                [int(item) for item in selected.get("shape", [])],
            )
    model_file = _model_file_for_manifest_record(manifest, selected)
    if not model_file:
        raise RuntimeError("GGUF tensor manifest does not include model_file")
    f32_sha256 = str(selected.get("f32_sha256", ""))
    if not f32_sha256:
        raise RuntimeError(f"GGUF tensor {tensor_name!r} missing f32_sha256")
    w_f32 = _proof_f32_weight_matrix_from_model_file(
        model_file,
        str(tensor_name),
        f32_sha256,
    )
    # The expensive dequant just happened — bank the i8 form so it never
    # happens again for this tensor on this machine (full profile only:
    # compact re-derives per draw by design, and banking here would regrow
    # the cache one hard audit at a time).
    proof_i8, scale = quantize_proof_i8(w_f32)
    if _should_persist_i8(selected):
        store_cached_proof_i8(selected, proof_i8, scale)
    return w_f32


def f32_tensor_from_manifest(
    manifest: Mapping[str, Any],
    tensor_name: str,
) -> np.ndarray:
    """Load and verify a canonical f32 tensor from a GGUF manifest."""

    selected = _manifest_record_for_tensor(manifest, tensor_name)
    cached_path = str(selected.get("f32_path", ""))
    expected_hash = str(selected.get("f32_sha256", ""))
    shape = tuple(int(item) for item in selected.get("shape", []))
    if cached_path:
        path = Path(cached_path)
        if path.exists():
            raw = path.read_bytes()
            if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
                raise RuntimeError("GGUF tensor f32 cache hash mismatch")
            array = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            return np.ascontiguousarray(array)
    cached_blob = load_cached_f32_blob(selected)
    if cached_blob is not None:
        return cached_blob
    model_file = _model_file_for_manifest_record(manifest, selected)
    if not model_file:
        raise RuntimeError("GGUF tensor manifest does not include model_file")
    array = _f32_tensor_from_model_file(
        model_file,
        str(tensor_name),
        expected_hash,
        shape,
    )
    store_cached_f32_blob(selected, array)
    return array


@lru_cache(maxsize=4)
def _gguf_reader_and_tensor_map(model_file: str) -> tuple[Any, dict[str, Any]]:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("GGUF tensor loading requires the gguf package") from exc

    reader = gguf.GGUFReader(Path(model_file))
    return reader, {str(tensor.name): tensor for tensor in reader.tensors}


def _gguf_tensor_from_model_file(model_file: str, tensor_name: str) -> Any:
    _reader, tensors = _gguf_reader_and_tensor_map(str(model_file))
    try:
        return tensors[str(tensor_name)]
    except KeyError as exc:
        raise KeyError(f"GGUF model file does not contain tensor {tensor_name!r}") from exc


@lru_cache(maxsize=16)
def _proof_i8_weight_matrix_from_model_file(
    model_file: str,
    tensor_name: str,
    expected_root: str,
    chunk_size: int,
    expected_sha256: str = "",
) -> tuple[np.ndarray, float]:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("GGUF proof weight loading requires the gguf package") from exc

    tensor = _gguf_tensor_from_model_file(model_file, tensor_name)
    f32_data = gguf.dequantize(tensor.data, tensor.tensor_type).astype(
        "float32",
        copy=False,
    )
    shaped_f32 = proof_f32_weight_matrix_from_gguf_f32(
        f32_data,
        [int(item) for item in tensor.shape.tolist()],
    )
    proof_i8, scale = quantize_proof_i8(shaped_f32)
    # Verify against the committed proof_i8_sha256 when the manifest leaf
    # carries one: the leaf binds the sha AND the Merkle root to the SAME
    # bytes, so the sha is an equally binding self-check while costing one
    # linear hash instead of a full tree build at the 128 B chunk size.
    # Measured on the 2x5090 box, the tree-build verification made a
    # 10.5 s LM-head derive cost 207 s and a 0.2 s attn derive cost 15 s.
    # The root-check branch stays for manifests without a committed sha.
    if expected_sha256:
        derived_sha = hashlib.sha256(proof_i8.tobytes(order="C")).hexdigest()
        if derived_sha != str(expected_sha256):
            raise RuntimeError("GGUF proof weight sha mismatch")
    else:
        merkle = FlatWeightMerkle(
            torch.from_numpy(proof_i8),
            int(chunk_size),
            store_raw=False,
        )
        if merkle.root.hex() != str(expected_root):
            raise RuntimeError("GGUF proof weight root mismatch")
    return proof_i8, scale


@lru_cache(maxsize=32)
def _f32_tensor_from_model_file(
    model_file: str,
    tensor_name: str,
    expected_f32_sha256: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("GGUF proof tensor loading requires the gguf package") from exc

    tensor = _gguf_tensor_from_model_file(model_file, tensor_name)
    f32_data = gguf.dequantize(tensor.data, tensor.tensor_type).astype(
        "float32",
        copy=False,
    )
    f32_bytes = f32_data.tobytes(order="C")
    if hashlib.sha256(f32_bytes).hexdigest() != str(expected_f32_sha256):
        raise RuntimeError("GGUF tensor f32 hash mismatch")
    array = np.asarray(f32_data, dtype=np.float32).reshape(expected_shape)
    return np.ascontiguousarray(array)


@lru_cache(maxsize=16)
def _proof_f32_weight_matrix_from_model_file(
    model_file: str,
    tensor_name: str,
    expected_f32_sha256: str,
) -> np.ndarray:
    try:
        import gguf
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("GGUF proof weight loading requires the gguf package") from exc

    tensor = _gguf_tensor_from_model_file(model_file, tensor_name)
    f32_data = gguf.dequantize(tensor.data, tensor.tensor_type).astype(
        "float32",
        copy=False,
    )
    f32_bytes = f32_data.tobytes(order="C")
    if hashlib.sha256(f32_bytes).hexdigest() != str(expected_f32_sha256):
        raise RuntimeError("GGUF tensor f32 hash mismatch")
    return proof_f32_weight_matrix_from_gguf_f32(
        f32_data,
        [int(item) for item in tensor.shape.tolist()],
    )


def save_gguf_tensor_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_gguf_tensor_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GGUF tensor manifest must be a JSON object")
    if int(data.get("version", 0)) != GGUF_TENSOR_MANIFEST_VERSION:
        raise ValueError("unsupported GGUF tensor manifest version")
    records = _manifest_records(data)
    root = MerkleTree([tensor_leaf_bytes(record) for record in records]).root.hex()
    if str(data.get("tensor_manifest_root", "")) != root:
        raise ValueError("GGUF tensor manifest root mismatch")
    return data


def _record_matrix_size(record: Mapping[str, Any]) -> int:
    name = str(record.get("name", "")).lower()
    if "bias" in name or "norm" in name:
        return 0
    try:
        shape = [int(item) for item in record.get("shape", [])]
        n_elements = int(record.get("n_elements", 0))
    except Exception:
        return 0
    if len(shape) < 2 or n_elements < 4096:
        return 0
    if min(shape[:2]) < 16:
        return 0
    return n_elements


def suggest_ggml_trace_max_elems_from_tensor_records(
    records: list[Mapping[str, Any]],
    *,
    floor: int = GGML_TRACE_MAX_ELEMS_FLOOR,
    multiplier: float = 2.0,
    full_range: bool = False,
) -> int:
    """Suggest a model-scaled GGML proof-capture element cap.

    The cheap organic proof path wants a real repeated model GEMM without
    routinely selecting the largest FFN/output projections. A lower-quartile
    repeated matrix size gives that behavior across GQA, MQA, and conventional
    attention layouts without requiring architecture-specific tensor names.

    ``full_range`` covers EVERY provable tensor instead. Slot-view meshes
    (``--parallel > 1``) need it: they have no serve-time dumps to keep
    cheap, and the quartile cap only starves the challenge universe below
    the anti-shrink floor (2 eligible ops per layer on GQA models) and
    blocks the final-projection replay dump the decode audit opens.
    """

    sizes = sorted(size for record in records if (size := _record_matrix_size(record)) > 0)
    if not sizes:
        return int(floor)
    if full_range:
        suggested = max(int(floor), int(math.ceil(float(sizes[-1]) * 1.25)))
        return ((suggested + 4095) // 4096) * 4096
    counts: dict[int, int] = {}
    for size in sizes:
        counts[size] = counts.get(size, 0) + 1
    repeated = sorted(size for size in sizes if counts.get(size, 0) >= 2)
    pool = repeated or sizes
    index = int((len(pool) - 1) * 0.25)
    selected = int(pool[max(0, min(index, len(pool) - 1))])
    suggested = max(int(floor), int(math.ceil(float(selected) * float(multiplier))))
    return ((suggested + 4095) // 4096) * 4096


def suggest_ggml_trace_max_elems_from_manifest(
    manifest: Mapping[str, Any],
    *,
    floor: int = GGML_TRACE_MAX_ELEMS_FLOOR,
    full_range: bool = False,
) -> int:
    records = manifest.get("tensors", [])
    if not isinstance(records, list):
        return int(floor)
    return suggest_ggml_trace_max_elems_from_tensor_records(
        records, floor=floor, full_range=full_range
    )


def suggest_ggml_trace_max_elems_from_gguf_model(
    model_path: str | Path,
    *,
    floor: int = GGML_TRACE_MAX_ELEMS_FLOOR,
    full_range: bool = False,
) -> int:
    """Suggest the proof-capture cap from GGUF tensor metadata only."""

    try:
        import gguf
    except Exception:
        return int(floor)
    try:
        reader = gguf.GGUFReader(Path(model_path))
    except Exception:
        return int(floor)
    records: list[dict[str, Any]] = []
    for tensor in getattr(reader, "tensors", []) or []:
        try:
            shape = [int(item) for item in tensor.shape.tolist()]
        except Exception:
            continue
        records.append(
            {
                "name": str(tensor.name),
                "shape": shape,
                "n_elements": int(tensor.n_elements),
            }
        )
    return suggest_ggml_trace_max_elems_from_tensor_records(
        records, floor=floor, full_range=full_range
    )
