"""Verifier-openable proof helpers for the fixed-workspace capacity audit."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from neurons.capacity_audit import (
    canonical_json,
    derive_sampled_pass_index,
    root_words_digest,
    transcript_root,
)


WORKSPACE_PROOF_FORMAT = "fixed_workspace_merkle_gemm_v1"
WORKSPACE_MIXED_PROOF_FORMAT = "fixed_workspace_mixed_merkle_gemm_v2"
WORKSPACE_TRANSITION_PROOF_FORMAT = "fixed_workspace_transition_merkle_gemm_v3"
WORKSPACE_COMPACT_PROOF_FORMAT = "fixed_workspace_compact_gemm_v1"
WORKSPACE_SEED_PREFIX = b"VERATHOS_FIXED_WORKSPACE_V1"
WORKSPACE_PASS_ROOT_PREFIX = b"VERATHOS_FIXED_WORKSPACE_PASS_ROOT_V1"
WORKSPACE_TRANSCRIPT_PREFIX = b"VERATHOS_FIXED_WORKSPACE_TRANSCRIPT_V1"
WORKSPACE_PROOF_ROUND_PREFIX = b"VERATHOS_FIXED_WORKSPACE_PROOF_ROUND_V1"
WORKSPACE_PROOF_BLOCK_PREFIX = b"VERATHOS_FIXED_WORKSPACE_PROOF_BLOCK_V1"
WORKSPACE_Q_BLOCK_TAG = b"VERATHOS_Q_BLOCK_V1"
WORKSPACE_Q_LEAF_TAG = b"VERATHOS_Q_LEAF_V1"
WORKSPACE_Q_NODE_TAG = b"VERATHOS_Q_NODE_V1"
WORKSPACE_STATE_BLOCK_TAG = b"VERATHOS_AB_STATE_BLOCK_V1"
WORKSPACE_STATE_LEAF_TAG = b"VERATHOS_AB_STATE_LEAF_V1"
WORKSPACE_STATE_NODE_TAG = b"VERATHOS_AB_STATE_NODE_V1"
WORKSPACE_TRANSITION_PROOF_TAG = b"VERATHOS_AB_TRANSITION_PROOF_V1"

_PASS_MIX = 0xD1B54A32D192ED03
_ROUND_MIX = 0xABC98388FB8FAC03
_A_MIX = 0x8CB92BA72F3D8DD7
_B_MIX = 0x195D7C4F8E2D2D13
_MIX_ROUND_MIX = 0x9E3779B97F4A7C15
_MIX_A_OFFSET = 0xA5A5A5A5A5A5A5A5
_MIX_B_OFFSET = 0x5A5A5A5A5A5A5A5A
_TRANSITION_A_OFFSET = 0xA5A5A5A5A5A5A5A5
_TRANSITION_B_OFFSET = 0x5A5A5A5A5A5A5A5A
_TRANSITION_A_READ = 0xBF58476D1CE4E5B9
_TRANSITION_B_READ = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1


def preload_workspace_proof_verifier() -> dict[str, Any]:
    """Load verifier dependencies before an audit window.

    The first verifier call may otherwise pay torch/native-extension import or
    JIT-loading cost. That cost is not part of proof verification and should be
    paid during validator startup or warm-up.
    """
    from zkllm.crypto.field import mod_p as _mod_p
    from zkllm.crypto.sumcheck import sumcheck_verify as _sumcheck_verify
    from zkllm.crypto.transcript import Transcript as _Transcript

    try:
        from zkllm.cuda import zkllm_native as _native
    except Exception:
        _native = None
    return {
        "mod_p": _mod_p is not None,
        "sumcheck_verify": _sumcheck_verify is not None,
        "transcript": _Transcript is not None,
        "zkllm_native_has_cuda": bool(getattr(_native, "HAS_CUDA", False)),
    }


def _u64(value: int) -> bytes:
    return struct.pack(">Q", int(value))


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def workspace_seed_for(lease_id: str, gpu_index: int, seed_hex: str = "") -> int:
    """Match ``bench_workspace.seed_for`` exactly."""
    h = hashlib.sha256()
    h.update(WORKSPACE_SEED_PREFIX)
    if seed_hex:
        raw = seed_hex[2:] if str(seed_hex).startswith("0x") else str(seed_hex)
        h.update(bytes.fromhex(raw))
    h.update(str(lease_id).encode())
    h.update(int(gpu_index).to_bytes(4, "little"))
    return int.from_bytes(h.digest()[:8], "little")


def _splitmix64(x: int) -> int:
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def workspace_i8_value(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    matrix_id: int,
    row: int,
    col: int,
    matrix_dim: int,
) -> int:
    """Recompute one deterministic int8 A/B entry from the CUDA fill kernel."""
    idx = int(row) * int(matrix_dim) + int(col)
    x = int(workspace_seed) & _MASK64
    x ^= (int(pass_index) * _PASS_MIX) & _MASK64
    x ^= (int(round_index) * _ROUND_MIX) & _MASK64
    x ^= int(idx) & _MASK64
    mix = _A_MIX if int(matrix_id) == 0 else _B_MIX
    return int((_splitmix64(x ^ mix) & 0xFF) - 128)


def _u8_from_i8(value: int) -> int:
    return int(value) & 0xFF


def _i8_from_u8(value: int) -> int:
    value = int(value) & 0xFF
    return value - 256 if value >= 128 else value


def workspace_i8_value_after_mix(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    matrix_id: int,
    row: int,
    col: int,
    matrix_dim: int,
    memory_mix_rounds: int,
) -> int:
    """Recompute one A/B entry after the opt-in v2 streaming mix.

    The mix is intentionally per-element: it adds deterministic streaming
    read/write pressure without cross-thread data dependencies, so the verifier
    can cheaply recompute sampled A/B values for any configured mix count.
    """
    if int(memory_mix_rounds) == 0:
        return workspace_i8_value(
            workspace_seed=workspace_seed,
            pass_index=pass_index,
            round_index=round_index,
            matrix_id=matrix_id,
            row=row,
            col=col,
            matrix_dim=matrix_dim,
        )
    if int(memory_mix_rounds) < 0:
        raise ValueError("unsupported_memory_mix_rounds")

    idx = int(row) * int(matrix_dim) + int(col)
    ai = _u8_from_i8(workspace_i8_value(
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        matrix_id=0,
        row=idx // int(matrix_dim),
        col=idx % int(matrix_dim),
        matrix_dim=matrix_dim,
    ))
    bi = _u8_from_i8(workspace_i8_value(
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        matrix_id=1,
        row=idx // int(matrix_dim),
        col=idx % int(matrix_dim),
        matrix_dim=matrix_dim,
    ))

    for mix_index in range(int(memory_mix_rounds)):
        basis = int(workspace_seed) & _MASK64
        basis ^= (int(pass_index) * _PASS_MIX) & _MASK64
        basis ^= (int(round_index) * _ROUND_MIX) & _MASK64
        basis ^= (int(mix_index) * _MIX_ROUND_MIX) & _MASK64
        ra = _splitmix64(basis ^ ((idx << 1) & _MASK64) ^ _MIX_A_OFFSET)
        rb = _splitmix64(basis ^ ((idx << 2) & _MASK64) ^ _MIX_B_OFFSET)
        ai ^= ra & 0xFF
        bi ^= rb & 0xFF

    return _i8_from_u8(ai if int(matrix_id) == 0 else bi)


def transition_basis(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    mix_index: int,
) -> int:
    basis = int(workspace_seed) & _MASK64
    basis ^= (int(pass_index) * _PASS_MIX) & _MASK64
    basis ^= (int(round_index) * _ROUND_MIX) & _MASK64
    basis ^= (int(mix_index) * _MIX_ROUND_MIX) & _MASK64
    return basis & _MASK64


def transition_source_indices(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    mix_index: int,
    element_index: int,
    matrix_dim: int,
    fanout: int = 1,
) -> tuple[tuple[int, int], ...]:
    total = int(matrix_dim) * int(matrix_dim)
    if total <= 0:
        raise ValueError("invalid_matrix_dim")
    fanout = int(fanout)
    if fanout <= 0:
        raise ValueError("invalid_transition_fanout")
    idx = int(element_index)
    if idx < 0 or idx >= total:
        raise ValueError("element_index_out_of_range")
    basis = transition_basis(
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        mix_index=mix_index,
    )
    out: list[tuple[int, int]] = []
    for lane in range(fanout):
        lane_basis = (basis ^ ((lane + 1) * 0xD6E8FEB86659FD93)) & _MASK64
        j = _splitmix64(lane_basis ^ ((idx * _TRANSITION_A_READ) & _MASK64)) % total
        k = _splitmix64(lane_basis ^ ((idx * _TRANSITION_B_READ) & _MASK64)) % total
        out.append((int(j), int(k)))
    return tuple(out)


def transition_expected_values(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    mix_index: int,
    element_index: int,
    ai: int,
    bi: int,
    sources: Sequence[tuple[int, int, int]],
) -> tuple[int, int]:
    idx = int(element_index)
    basis = transition_basis(
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        mix_index=mix_index,
    )
    out_a = _u8_from_i8(ai)
    out_b = _u8_from_i8(bi)
    for lane, aj, bk in sources:
        lane_basis = (basis ^ ((int(lane) + 1) * 0xD6E8FEB86659FD93)) & _MASK64
        ma = _splitmix64(lane_basis ^ ((idx << 1) & _MASK64) ^ _TRANSITION_A_OFFSET) & 0xFF
        mb = _splitmix64(lane_basis ^ ((idx << 2) & _MASK64) ^ _TRANSITION_B_OFFSET) & 0xFF
        out_a = (out_a + (_u8_from_i8(bk) ^ ma)) & 0xFF
        out_b = (out_b + (_u8_from_i8(aj) ^ mb)) & 0xFF
    return _i8_from_u8(out_a), _i8_from_u8(out_b)


def derive_transition_element_index(
    *,
    before_root_hex: str,
    after_root_hex: str,
    lease_id: str,
    gpu_index: int,
    pass_index: int,
    round_index: int,
    mix_index: int,
    matrix_dim: int,
    challenge_seed_hex: str = "",
) -> int:
    total = int(matrix_dim) * int(matrix_dim)
    if total <= 0:
        return 0
    digest = _sha256(
        WORKSPACE_TRANSITION_PROOF_TAG,
        bytes.fromhex(str(challenge_seed_hex or "").removeprefix("0x")) if challenge_seed_hex else b"",
        bytes.fromhex(str(before_root_hex).removeprefix("0x")),
        bytes.fromhex(str(after_root_hex).removeprefix("0x")),
        str(lease_id).encode(),
        _u64(int(gpu_index)),
        _u64(int(pass_index)),
        _u64(int(round_index)),
        _u64(int(mix_index)),
    )
    return int.from_bytes(digest[:8], "big") % total


def workspace_fill_state_root_i8(
    *,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    matrix_dim: int,
    block_size: int,
) -> str:
    """Compute the deterministic filled A/B state root before transition mix 0.

    This helper is CPU-side and intended for tests/proof assembly, not for the
    timed miner path.
    """
    n = int(matrix_dim)
    a_matrix = [
        [
            workspace_i8_value(
                workspace_seed=workspace_seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=0,
                row=i,
                col=j,
                matrix_dim=n,
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    b_matrix = [
        [
            workspace_i8_value(
                workspace_seed=workspace_seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=1,
                row=i,
                col=j,
                matrix_dim=n,
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    return WorkspaceStateMerkleTree.from_state_i8_matrices(
        a_matrix,
        b_matrix,
        int(block_size),
    ).root.hex()


def pass_root_from_round_roots(pass_index: int, round_roots: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(WORKSPACE_PASS_ROOT_PREFIX)
    h.update(_u64(int(pass_index)))
    h.update(_u64(len(round_roots)))
    for round_index, root_hex in enumerate(round_roots):
        h.update(_u64(round_index))
        h.update(bytes.fromhex(str(root_hex).removeprefix("0x")))
    return h.hexdigest()


def workspace_transcript_root(pass_entries: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(WORKSPACE_TRANSCRIPT_PREFIX)
    h.update(_u64(len(pass_entries)))
    for entry in pass_entries:
        pass_index = int(entry.get("pass_index", 0))
        pass_root = str(entry.get("pass_root") or "")
        round_roots = entry.get("round_roots") or []
        h.update(_u64(pass_index))
        h.update(bytes.fromhex(pass_root.removeprefix("0x")))
        h.update(_u64(len(round_roots)))
        for round_index, root_hex in enumerate(round_roots):
            h.update(_u64(round_index))
            h.update(bytes.fromhex(str(root_hex).removeprefix("0x")))
    return h.hexdigest()


def derive_sampled_round_index(
    transcript_hex: str,
    lease_id: str,
    gpu_index: int,
    pass_index: int,
    rounds: int,
) -> int:
    if rounds <= 0:
        return 0
    digest = _sha256(
        WORKSPACE_PROOF_ROUND_PREFIX,
        bytes.fromhex(str(transcript_hex).removeprefix("0x")),
        str(lease_id).encode(),
        _u64(int(gpu_index)),
        _u64(int(pass_index)),
    )
    return int.from_bytes(digest[:8], "big") % int(rounds)


def derive_sampled_transition_mix_index(
    transcript_hex: str,
    lease_id: str,
    gpu_index: int,
    pass_index: int,
    round_index: int,
    transition_mix_rounds: int,
) -> int:
    if transition_mix_rounds <= 0:
        return 0
    digest = _sha256(
        WORKSPACE_TRANSITION_PROOF_TAG,
        bytes.fromhex(str(transcript_hex).removeprefix("0x")),
        str(lease_id).encode(),
        _u64(int(gpu_index)),
        _u64(int(pass_index)),
        _u64(int(round_index)),
    )
    return int.from_bytes(digest[:8], "big") % int(transition_mix_rounds)


def derive_challenge_block_index(
    *,
    round_root_hex: str,
    lease_id: str,
    gpu_index: int,
    pass_index: int,
    round_index: int,
    blocks_per_row: int,
    challenge_seed_hex: str = "",
) -> int:
    total_blocks = int(blocks_per_row) * int(blocks_per_row)
    if total_blocks <= 0:
        return 0
    digest = _sha256(
        WORKSPACE_PROOF_BLOCK_PREFIX,
        bytes.fromhex(str(challenge_seed_hex or "").removeprefix("0x")) if challenge_seed_hex else b"",
        bytes.fromhex(str(round_root_hex).removeprefix("0x")),
        str(lease_id).encode(),
        _u64(int(gpu_index)),
        _u64(int(pass_index)),
        _u64(int(round_index)),
    )
    return int.from_bytes(digest[:8], "big") % total_blocks


def _level_sizes(num_leaves: int) -> list[int]:
    sizes: list[int] = []
    n = int(num_leaves)
    while True:
        sizes.append(n)
        if n <= 1:
            break
        n = (n + 1) >> 1
    return sizes


def _level_offsets(num_leaves: int) -> list[int]:
    offsets = [0]
    for size in _level_sizes(num_leaves)[:-1]:
        offsets.append(offsets[-1] + size)
    return offsets


def workspace_q_block_hash_i32(block_rows: Sequence[Sequence[int]]) -> bytes:
    h = hashlib.sha256()
    h.update(WORKSPACE_Q_BLOCK_TAG)
    for row in block_rows:
        for value in row:
            h.update(struct.pack("<i", int(value)))
    block_hash = h.digest()
    return hashlib.sha256(WORKSPACE_Q_LEAF_TAG + block_hash).digest()


def _fixed_tag(tag: bytes) -> bytes:
    if len(tag) > 32:
        raise ValueError("workspace tag too long")
    return tag + (b"\x00" * (32 - len(tag)))


def _state_block_arrays_i8(
    a_block_rows: Sequence[Sequence[int]],
    b_block_rows: Sequence[Sequence[int]],
) -> tuple[Any, Any]:
    import numpy as np

    a_arr = np.asarray(a_block_rows, dtype=np.int16)
    b_arr = np.asarray(b_block_rows, dtype=np.int16)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape != b_arr.shape:
        raise ValueError("invalid state block shape")
    if a_arr.shape[0] <= 0 or a_arr.shape[1] <= 0:
        raise ValueError("state block must not be empty")
    return a_arr, b_arr


def workspace_state_block_hash_i8(
    a_block_rows: Sequence[Sequence[int]],
    b_block_rows: Sequence[Sequence[int]],
) -> bytes:
    """Hash an A/B int8 state block exactly like the CUDA state-root path."""
    h = hashlib.sha256()
    h.update(_fixed_tag(WORKSPACE_STATE_BLOCK_TAG))
    try:
        import numpy as np

        if isinstance(a_block_rows, np.ndarray) and isinstance(b_block_rows, np.ndarray):
            a_arr = a_block_rows
            b_arr = b_block_rows
            if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape != b_arr.shape:
                raise ValueError("invalid state block shape")
            if a_arr.shape[0] <= 0 or a_arr.shape[1] <= 0:
                raise ValueError("state block must not be empty")
        else:
            a_arr, b_arr = _state_block_arrays_i8(a_block_rows, b_block_rows)
        packed = np.empty((a_arr.shape[0] * 2, a_arr.shape[1]), dtype=np.uint8)
        packed[0::2, :] = (a_arr & 0xFF).astype(np.uint8, copy=False)
        packed[1::2, :] = (b_arr & 0xFF).astype(np.uint8, copy=False)
        h.update(packed.tobytes(order="C"))
    except ImportError:
        if len(a_block_rows) != len(b_block_rows):
            raise ValueError("A/B block row count mismatch")
        if not a_block_rows:
            raise ValueError("state block must not be empty")
        width = len(a_block_rows[0])
        if width <= 0:
            raise ValueError("state block must not be empty")
        if any(len(row) != width for row in a_block_rows):
            raise ValueError("invalid A block shape")
        if any(len(row) != width for row in b_block_rows):
            raise ValueError("invalid B block shape")
        payload = bytearray()
        for a_row, b_row in zip(a_block_rows, b_block_rows):
            payload.extend(int(v) & 0xFF for v in a_row)
            payload.extend(int(v) & 0xFF for v in b_row)
        h.update(payload)
    block_hash = h.digest()
    return hashlib.sha256(_fixed_tag(WORKSPACE_STATE_LEAF_TAG) + block_hash).digest()


def workspace_merkle_parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(WORKSPACE_Q_NODE_TAG + left + right).digest()


def workspace_state_merkle_parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_fixed_tag(WORKSPACE_STATE_NODE_TAG) + left + right).digest()


def workspace_merkle_root_from_leaves(leaf_hashes: Sequence[bytes]) -> bytes:
    if not leaf_hashes:
        raise ValueError("cannot build workspace Merkle root from empty leaves")
    level = list(leaf_hashes)
    while len(level) > 1:
        next_level: list[bytes] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else left
            next_level.append(workspace_merkle_parent(left, right))
        level = next_level
    return level[0]


def workspace_state_merkle_root_from_leaves(leaf_hashes: Sequence[bytes]) -> bytes:
    if not leaf_hashes:
        raise ValueError("cannot build workspace state Merkle root from empty leaves")
    level = list(leaf_hashes)
    while len(level) > 1:
        next_level: list[bytes] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else left
            next_level.append(workspace_state_merkle_parent(left, right))
        level = next_level
    return level[0]


class WorkspaceMerkleTree:
    """Small CPU tree used by tests and post-timing sampled proof assembly."""

    def __init__(self, leaf_hashes: Sequence[bytes]):
        if not leaf_hashes:
            raise ValueError("leaf_hashes must not be empty")
        self.num_leaves = len(leaf_hashes)
        self._nodes: list[bytes] = []
        self._offsets: list[int] = []
        level = list(leaf_hashes)
        self._offsets.append(0)
        self._nodes.extend(level)
        while len(level) > 1:
            self._offsets.append(len(self._nodes))
            next_level: list[bytes] = []
            for idx in range(0, len(level), 2):
                left = level[idx]
                right = level[idx + 1] if idx + 1 < len(level) else left
                next_level.append(workspace_merkle_parent(left, right))
            self._nodes.extend(next_level)
            level = next_level
        self._offsets.append(len(self._nodes))

    @classmethod
    def from_flat_tree(cls, flat_tree: bytes, num_leaves: int) -> "WorkspaceMerkleTree":
        node_count = len(flat_tree) // 32
        expected = sum(_level_sizes(num_leaves))
        if len(flat_tree) % 32 != 0 or node_count != expected:
            raise ValueError("invalid flat workspace Merkle tree length")
        obj = cls.__new__(cls)
        obj.num_leaves = int(num_leaves)
        obj._nodes = [flat_tree[i * 32:(i + 1) * 32] for i in range(node_count)]
        obj._offsets = _level_offsets(num_leaves)
        obj._offsets.append(node_count)
        return obj

    @classmethod
    def from_matrix_i32(cls, matrix: Sequence[Sequence[int]], block_size: int) -> "WorkspaceMerkleTree":
        n = len(matrix)
        if n <= 0 or any(len(row) != n for row in matrix):
            raise ValueError("matrix must be non-empty and square")
        if n % int(block_size) != 0:
            raise ValueError("matrix dimension must be divisible by block_size")
        blocks_per_row = n // int(block_size)
        leaves: list[bytes] = []
        for bi in range(blocks_per_row):
            for bj in range(blocks_per_row):
                rows = [
                    row[bj * block_size:(bj + 1) * block_size]
                    for row in matrix[bi * block_size:(bi + 1) * block_size]
                ]
                leaves.append(workspace_q_block_hash_i32(rows))
        return cls(leaves)

    @property
    def root(self) -> bytes:
        return self._nodes[-1]

    def get_leaf(self, bi: int, bj: int, blocks_per_row: int | None = None) -> bytes:
        bpr = int(blocks_per_row or math.isqrt(self.num_leaves))
        return self._nodes[int(bi) * bpr + int(bj)]

    def get_path(self, bi: int, bj: int, blocks_per_row: int | None = None):
        from zkllm.types import MerklePath

        bpr = int(blocks_per_row or math.isqrt(self.num_leaves))
        leaf_index = int(bi) * bpr + int(bj)
        siblings: list[tuple[bytes, bool]] = []
        idx = leaf_index
        sizes = _level_sizes(self.num_leaves)
        offsets = _level_offsets(self.num_leaves)
        for level, level_size in enumerate(sizes[:-1]):
            if idx % 2 == 0:
                sib_idx = idx + 1
                is_left = False
                if sib_idx >= level_size:
                    sib_idx = idx
            else:
                sib_idx = idx - 1
                is_left = True
            siblings.append((self._nodes[offsets[level] + sib_idx], is_left))
            idx >>= 1
        return MerklePath(leaf_index=leaf_index, siblings=siblings)


class WorkspaceStateMerkleTree:
    """CPU Merkle tree for sampled A/B transition-state openings."""

    def __init__(self, leaf_hashes: Sequence[bytes]):
        if not leaf_hashes:
            raise ValueError("leaf_hashes must not be empty")
        self.num_leaves = len(leaf_hashes)
        self._nodes: list[bytes] = []
        self._offsets: list[int] = []
        level = list(leaf_hashes)
        self._offsets.append(0)
        self._nodes.extend(level)
        while len(level) > 1:
            self._offsets.append(len(self._nodes))
            next_level: list[bytes] = []
            for idx in range(0, len(level), 2):
                left = level[idx]
                right = level[idx + 1] if idx + 1 < len(level) else left
                next_level.append(workspace_state_merkle_parent(left, right))
            self._nodes.extend(next_level)
            level = next_level
        self._offsets.append(len(self._nodes))

    @classmethod
    def from_flat_tree(cls, flat_tree: bytes, num_leaves: int) -> "WorkspaceStateMerkleTree":
        node_count = len(flat_tree) // 32
        expected = sum(_level_sizes(num_leaves))
        if len(flat_tree) % 32 != 0 or node_count != expected:
            raise ValueError("invalid flat workspace state Merkle tree length")
        obj = cls.__new__(cls)
        obj.num_leaves = int(num_leaves)
        obj._nodes = [flat_tree[i * 32:(i + 1) * 32] for i in range(node_count)]
        obj._offsets = _level_offsets(num_leaves)
        obj._offsets.append(node_count)
        return obj

    @classmethod
    def from_state_i8_matrices(
        cls,
        a_matrix: Sequence[Sequence[int]],
        b_matrix: Sequence[Sequence[int]],
        block_size: int,
    ) -> "WorkspaceStateMerkleTree":
        n = len(a_matrix)
        if n <= 0 or len(b_matrix) != n:
            raise ValueError("state matrices must be non-empty and same-sized")
        if any(len(row) != n for row in a_matrix) or any(len(row) != n for row in b_matrix):
            raise ValueError("state matrices must be square")
        if n % int(block_size) != 0:
            raise ValueError("state matrix dimension must be divisible by block_size")
        leaves: list[bytes] = []
        blocks_per_row = n // int(block_size)
        for bi in range(blocks_per_row):
            for bj in range(blocks_per_row):
                a_rows = [
                    row[bj * block_size:(bj + 1) * block_size]
                    for row in a_matrix[bi * block_size:(bi + 1) * block_size]
                ]
                b_rows = [
                    row[bj * block_size:(bj + 1) * block_size]
                    for row in b_matrix[bi * block_size:(bi + 1) * block_size]
                ]
                leaves.append(workspace_state_block_hash_i8(a_rows, b_rows))
        return cls(leaves)

    @property
    def root(self) -> bytes:
        return self._nodes[-1]

    def get_path(self, block_index: int):
        from zkllm.types import MerklePath

        if int(block_index) < 0 or int(block_index) >= self.num_leaves:
            raise ValueError("block_index out of range")
        siblings: list[tuple[bytes, bool]] = []
        idx = int(block_index)
        sizes = _level_sizes(self.num_leaves)
        offsets = _level_offsets(self.num_leaves)
        for level, level_size in enumerate(sizes[:-1]):
            if idx % 2 == 0:
                sib_idx = idx + 1
                is_left = False
                if sib_idx >= level_size:
                    sib_idx = idx
            else:
                sib_idx = idx - 1
                is_left = True
            siblings.append((self._nodes[offsets[level] + sib_idx], is_left))
            idx >>= 1
        return MerklePath(leaf_index=int(block_index), siblings=siblings)


def verify_workspace_merkle_path(root: bytes, leaf_hash: bytes, path: Any) -> bool:
    current = bytes(leaf_hash)
    siblings = getattr(path, "siblings", path)
    for sibling, is_left in siblings:
        sibling_b = bytes.fromhex(sibling) if isinstance(sibling, str) else bytes(sibling)
        if bool(is_left):
            current = workspace_merkle_parent(sibling_b, current)
        else:
            current = workspace_merkle_parent(current, sibling_b)
    return current == bytes(root)


def verify_workspace_state_merkle_path(root: bytes, leaf_hash: bytes, path: Any) -> bool:
    current = bytes(leaf_hash)
    siblings = getattr(path, "siblings", path)
    for sibling, is_left in siblings:
        sibling_b = bytes.fromhex(sibling) if isinstance(sibling, str) else bytes(sibling)
        if bool(is_left):
            current = workspace_state_merkle_parent(sibling_b, current)
        else:
            current = workspace_state_merkle_parent(current, sibling_b)
    return current == bytes(root)


def serialize_merkle_path(path: Any) -> dict[str, Any]:
    return {
        "leaf_index": int(path.leaf_index),
        "siblings": [
            [bytes(sibling).hex(), bool(is_left)]
            for sibling, is_left in path.siblings
        ],
    }


def deserialize_merkle_path(data: Mapping[str, Any]):
    from zkllm.types import MerklePath

    return MerklePath(
        leaf_index=int(data.get("leaf_index", 0)),
        siblings=[
            (bytes.fromhex(str(item[0])), bool(item[1]))
            for item in data.get("siblings", [])
        ],
    )


def serialize_gemm_block_proof(proof: Any, *, q_block_i32: Sequence[Sequence[int]]) -> dict[str, Any]:
    sp = proof.sumcheck_proof
    return {
        "bi": int(proof.bi),
        "bj": int(proof.bj),
        "leaf_hash": bytes(proof.leaf_hash).hex(),
        "merkle_path": serialize_merkle_path(proof.merkle_path),
        "m_bits": int(proof.m_bits),
        "n_bits": int(proof.n_bits),
        "k_bits": int(proof.k_bits),
        "q_block_i32": [[int(v) for v in row] for row in q_block_i32],
        "sumcheck_proof": {
            "claimed_sum": int(sp.claimed_sum),
            "rounds": [[int(r.e0), int(r.e1), int(r.e2)] for r in sp.rounds],
            "final_A": int(sp.final_A),
            "final_B": int(sp.final_B),
        },
        "spot_A": [asdict(spot) for spot in proof.spot_X],
        "spot_B": [asdict(spot) for spot in proof.spot_W],
    }


def deserialize_gemm_block_proof(data: Mapping[str, Any]):
    from zkllm.types import GEMMBlockProof, SpotCheck, SumcheckProof, SumcheckRound

    proof_data = data.get("sumcheck_proof") or {}
    return GEMMBlockProof(
        bi=int(data.get("bi", 0)),
        bj=int(data.get("bj", 0)),
        leaf_hash=bytes.fromhex(str(data.get("leaf_hash") or "")),
        merkle_path=deserialize_merkle_path(data.get("merkle_path") or {}),
        sumcheck_proof=SumcheckProof(
            claimed_sum=int(proof_data.get("claimed_sum", 0)),
            rounds=[
                SumcheckRound(e0=int(r[0]), e1=int(r[1]), e2=int(r[2]))
                for r in proof_data.get("rounds", [])
            ],
            final_A=int(proof_data.get("final_A", 0)),
            final_B=int(proof_data.get("final_B", 0)),
        ),
        spot_X=[
            SpotCheck(row=int(s.get("row", 0)), col=int(s.get("col", 0)), value=int(s.get("value", 0)))
            for s in data.get("spot_A", [])
        ],
        spot_W=[
            SpotCheck(row=int(s.get("row", 0)), col=int(s.get("col", 0)), value=int(s.get("value", 0)))
            for s in data.get("spot_B", [])
        ],
        m_bits=int(data.get("m_bits", 0)),
        n_bits=int(data.get("n_bits", 0)),
        k_bits=int(data.get("k_bits", 0)),
    )


def _field_matrix_from_i32_block(block: Sequence[Sequence[int]], padded_rows: int, padded_cols: int) -> list[list[int]]:
    from zkllm.crypto.field import mod_p

    rows = len(block)
    cols = len(block[0]) if rows else 0
    return [
        [
            mod_p(int(block[i][j])) if i < rows and j < cols else 0
            for j in range(padded_cols)
        ]
        for i in range(padded_rows)
    ]


def _evaluate_q_block_claim(block: Sequence[Sequence[int]], r_i: Sequence[int], r_j: Sequence[int]) -> int:
    try:
        import numpy as np
        from zkllm.crypto.field import P
        from zkllm.crypto.field_fast import (
            build_eq_table_vec,
            mersenne_sum_axis0_fast,
            mersenne_vec_sum,
            mul_f_vec_mersenne,
        )

        matrix = np.asarray(block, dtype=np.int64)
        matrix = np.mod(matrix, P).astype(np.uint64, copy=False)
        rows, cols = matrix.shape
        eq_row = build_eq_table_vec(list(r_i))[:rows]
        eq_col = build_eq_table_vec(list(r_j))[:cols]
        tmp = mersenne_sum_axis0_fast(mul_f_vec_mersenne(matrix, eq_row[:, np.newaxis]))[:cols]
        return int(mersenne_vec_sum(mul_f_vec_mersenne(tmp, eq_col)))
    except Exception:
        from zkllm.crypto.mle import build_mle_from_matrix

        rows = len(block)
        cols = len(block[0]) if rows else 0
        m_bits = max(1, (rows - 1).bit_length())
        n_bits = max(1, (cols - 1).bit_length())
        field = _field_matrix_from_i32_block(block, 1 << m_bits, 1 << n_bits)
        return int(build_mle_from_matrix(field).evaluate(list(r_i) + list(r_j)))


def verify_workspace_gemm_block_proof(
    *,
    block_proof_data: Mapping[str, Any],
    round_root_hex: str,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    matrix_dim: int,
    block_size: int,
    spot_checks: int,
    memory_mix_rounds: int = 0,
    state_root_hex: str = "",
    state_spot_openings: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Verify the sampled block proof against the committed workspace round root."""
    from zkllm.crypto.field import mod_p
    from zkllm.crypto.sumcheck import sumcheck_verify
    from zkllm.crypto.transcript import Transcript

    q_block = block_proof_data.get("q_block_i32")
    if not isinstance(q_block, list) or not q_block:
        return False, "missing_q_block"
    if any(not isinstance(row, list) or len(row) != len(q_block[0]) for row in q_block):
        return False, "invalid_q_block_shape"
    if len(q_block) != int(block_size) or len(q_block[0]) != int(block_size):
        return False, "q_block_size_mismatch"

    block_proof = deserialize_gemm_block_proof(block_proof_data)
    blocks_per_row = int(matrix_dim) // int(block_size)
    expected_leaf_index = int(block_proof.bi) * blocks_per_row + int(block_proof.bj)
    if int(block_proof.merkle_path.leaf_index) != expected_leaf_index:
        return False, "merkle_leaf_index_mismatch"

    leaf_hash = workspace_q_block_hash_i32(q_block)
    if leaf_hash != block_proof.leaf_hash:
        return False, "q_block_leaf_hash_mismatch"
    round_root = bytes.fromhex(str(round_root_hex).removeprefix("0x"))
    if not verify_workspace_merkle_path(round_root, leaf_hash, block_proof.merkle_path):
        return False, "q_block_merkle_path_invalid"

    transcript = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
    transcript.absorb(b"pass_index", int(pass_index))
    transcript.absorb(b"round_index", int(round_index))
    transcript.absorb_bytes(b"round_root", round_root)
    transcript.absorb(b"bi", int(block_proof.bi))
    transcript.absorb(b"bj", int(block_proof.bj))
    transcript.absorb_bytes(b"leaf_hash", leaf_hash)

    r_i = transcript.squeeze_n(b"r_i", int(block_proof.m_bits))
    r_j = transcript.squeeze_n(b"r_j", int(block_proof.n_bits))
    expected_claim = _evaluate_q_block_claim(q_block, r_i, r_j)
    if int(block_proof.sumcheck_proof.claimed_sum) != expected_claim:
        return False, "q_block_claim_mismatch"

    ok, _final_claim, _challenges = sumcheck_verify(block_proof.sumcheck_proof, transcript)
    if not ok:
        return False, "sumcheck_invalid"

    if len(block_proof.spot_X) != int(spot_checks) or len(block_proof.spot_W) != int(spot_checks):
        return False, "spot_count_mismatch"
    state_blocks: dict[int, tuple[list[list[int]], list[list[int]]]] = {}
    if state_root_hex:
        for opening in state_spot_openings or []:
            ok, reason, block_index, a_rows, b_rows = _verify_state_block_opening(
                opening=opening,
                root_hex=state_root_hex,
                matrix_dim=matrix_dim,
                block_size=block_size,
            )
            if not ok:
                return False, reason
            state_blocks[block_index] = (a_rows, b_rows)

    def expected_state_value(matrix_id: int, row: int, col: int) -> tuple[bool, int | str]:
        block_index = _state_block_index_for_element(
            int(row) * int(matrix_dim) + int(col),
            matrix_dim,
            block_size,
        )
        if block_index not in state_blocks:
            return False, "missing_required_gemm_state_block"
        a_rows, b_rows = state_blocks[block_index]
        matrix = a_rows if int(matrix_id) == 0 else b_rows
        return True, _state_block_value(
            matrix,
            int(row) * int(matrix_dim) + int(col),
            matrix_dim,
            block_size,
        )

    row_start = int(block_proof.bi) * int(block_size)
    col_start = int(block_proof.bj) * int(block_size)
    for idx in range(int(spot_checks)):
        expected_x_row = row_start + int(transcript.squeeze(b"spot_X:row:" + str(idx).encode()) % int(block_size))
        expected_x_col = int(transcript.squeeze(b"spot_X:col:" + str(idx).encode()) % int(matrix_dim))
        spot_x = block_proof.spot_X[idx]
        if int(spot_x.row) != expected_x_row or int(spot_x.col) != expected_x_col:
            return False, "spot_a_position_mismatch"
        if state_root_hex:
            ok, value = expected_state_value(0, spot_x.row, spot_x.col)
            if not ok:
                return False, str(value)
            expected_x = int(value)
        else:
            try:
                expected_x = workspace_i8_value_after_mix(
                    workspace_seed=workspace_seed,
                    pass_index=pass_index,
                    round_index=round_index,
                    matrix_id=0,
                    row=spot_x.row,
                    col=spot_x.col,
                    matrix_dim=matrix_dim,
                    memory_mix_rounds=memory_mix_rounds,
                )
            except ValueError:
                return False, "unsupported_memory_mix_rounds"
        if int(spot_x.value) != mod_p(expected_x):
            return False, "spot_a_value_mismatch"

    for idx in range(int(spot_checks)):
        expected_w_row = int(transcript.squeeze(b"spot_W:row:" + str(idx).encode()) % int(matrix_dim))
        expected_w_col = col_start + int(transcript.squeeze(b"spot_W:col:" + str(idx).encode()) % int(block_size))
        spot_w = block_proof.spot_W[idx]
        if int(spot_w.row) != expected_w_row or int(spot_w.col) != expected_w_col:
            return False, "spot_b_position_mismatch"
        if state_root_hex:
            ok, value = expected_state_value(1, spot_w.row, spot_w.col)
            if not ok:
                return False, str(value)
            expected_w = int(value)
        else:
            try:
                expected_w = workspace_i8_value_after_mix(
                    workspace_seed=workspace_seed,
                    pass_index=pass_index,
                    round_index=round_index,
                    matrix_id=1,
                    row=spot_w.row,
                    col=spot_w.col,
                    matrix_dim=matrix_dim,
                    memory_mix_rounds=memory_mix_rounds,
                )
            except ValueError:
                return False, "unsupported_memory_mix_rounds"
        if int(spot_w.value) != mod_p(expected_w):
            return False, "spot_b_value_mismatch"

    return True, ""


def _state_block_index_for_element(element_index: int, matrix_dim: int, block_size: int) -> int:
    row = int(element_index) // int(matrix_dim)
    col = int(element_index) % int(matrix_dim)
    blocks_per_row = int(matrix_dim) // int(block_size)
    return (row // int(block_size)) * blocks_per_row + (col // int(block_size))


def _state_block_value(
    block_rows: Sequence[Sequence[int]],
    element_index: int,
    matrix_dim: int,
    block_size: int,
) -> int:
    row = int(element_index) // int(matrix_dim)
    col = int(element_index) % int(matrix_dim)
    local_row = row % int(block_size)
    local_col = col % int(block_size)
    return int(block_rows[local_row][local_col])


def _verify_state_block_opening(
    *,
    opening: Mapping[str, Any],
    root_hex: str,
    matrix_dim: int,
    block_size: int,
) -> tuple[bool, str, int, list[list[int]], list[list[int]]]:
    try:
        block_index = int(opening.get("block_index"))
        a_block = opening.get("a_block_i8")
        b_block = opening.get("b_block_i8")
        a_rows, b_rows = _state_block_arrays_i8(a_block, b_block)
    except Exception:
        return False, "invalid_state_block_opening", 0, [], []
    if tuple(a_rows.shape) != (int(block_size), int(block_size)):
        return False, "invalid_state_block_shape", 0, [], []
    blocks_per_row = int(matrix_dim) // int(block_size)
    total_blocks = blocks_per_row * blocks_per_row
    if block_index < 0 or block_index >= total_blocks:
        return False, "state_block_index_out_of_range", 0, [], []
    try:
        path = deserialize_merkle_path(opening.get("merkle_path") or {})
        path.leaf_index = int(path.leaf_index)
    except Exception:
        return False, "invalid_state_merkle_path", 0, [], []
    if int(path.leaf_index) != block_index:
        return False, "state_merkle_leaf_index_mismatch", 0, [], []
    leaf = workspace_state_block_hash_i8(a_rows, b_rows)
    root = bytes.fromhex(str(root_hex).removeprefix("0x"))
    if not verify_workspace_state_merkle_path(root, leaf, path):
        return False, "state_merkle_path_invalid", 0, [], []
    return True, "", block_index, a_rows, b_rows


def verify_workspace_transition_opening(
    *,
    opening: Mapping[str, Any],
    before_root_hex: str,
    after_root_hex: str,
    workspace_seed: int,
    pass_index: int,
    round_index: int,
    mix_index: int,
    matrix_dim: int,
    block_size: int,
    lease_id: str,
    gpu_index: int,
    challenge_seed_hex: str = "",
    transition_fanout: int = 1,
) -> tuple[bool, str]:
    """Verify one Fiat-Shamir sampled A/B transition opening.

    This verifies the memory-transition part of the v2 candidate relation. It
    intentionally samples one element and its required source blocks; production
    payload wiring decides how many such openings to request.
    """
    if int(matrix_dim) <= 0 or int(block_size) <= 0 or int(matrix_dim) % int(block_size) != 0:
        return False, "invalid_transition_parameters"
    expected_idx = derive_transition_element_index(
        before_root_hex=before_root_hex,
        after_root_hex=after_root_hex,
        lease_id=lease_id,
        gpu_index=int(gpu_index),
        pass_index=int(pass_index),
        round_index=int(round_index),
        mix_index=int(mix_index),
        matrix_dim=int(matrix_dim),
        challenge_seed_hex=challenge_seed_hex,
    )
    try:
        element_index = int(opening.get("element_index"))
    except Exception:
        return False, "invalid_transition_element_index"
    if element_index != expected_idx:
        return False, "transition_element_index_mismatch"

    try:
        source_pairs = transition_source_indices(
            workspace_seed=int(workspace_seed),
            pass_index=int(pass_index),
            round_index=int(round_index),
            mix_index=int(mix_index),
            element_index=element_index,
            matrix_dim=int(matrix_dim),
            fanout=int(transition_fanout),
        )
    except ValueError as exc:
        return False, str(exc)
    required_before = {
        _state_block_index_for_element(element_index, matrix_dim, block_size),
    }
    for j, k in source_pairs:
        required_before.add(_state_block_index_for_element(j, matrix_dim, block_size))
        required_before.add(_state_block_index_for_element(k, matrix_dim, block_size))
    required_after = {
        _state_block_index_for_element(element_index, matrix_dim, block_size),
    }

    before_block_payloads = opening.get("before_blocks") or []
    if len(before_block_payloads) != len(required_before):
        return False, "unexpected_before_state_block_count"
    before_blocks: dict[int, tuple[list[list[int]], list[list[int]]]] = {}
    for block in before_block_payloads:
        ok, reason, block_index, a_rows, b_rows = _verify_state_block_opening(
            opening=block,
            root_hex=before_root_hex,
            matrix_dim=int(matrix_dim),
            block_size=int(block_size),
        )
        if not ok:
            return False, reason
        before_blocks[block_index] = (a_rows, b_rows)
    if not required_before.issubset(before_blocks):
        return False, "missing_required_before_state_blocks"

    after_block_payloads = opening.get("after_blocks") or []
    if len(after_block_payloads) != len(required_after):
        return False, "unexpected_after_state_block_count"
    after_blocks: dict[int, tuple[list[list[int]], list[list[int]]]] = {}
    for block in after_block_payloads:
        ok, reason, block_index, a_rows, b_rows = _verify_state_block_opening(
            opening=block,
            root_hex=after_root_hex,
            matrix_dim=int(matrix_dim),
            block_size=int(block_size),
        )
        if not ok:
            return False, reason
        after_blocks[block_index] = (a_rows, b_rows)
    if not required_after.issubset(after_blocks):
        return False, "missing_required_after_state_blocks"

    idx_block = _state_block_index_for_element(element_index, matrix_dim, block_size)
    before_idx_a, before_idx_b = before_blocks[idx_block]
    after_idx_a, after_idx_b = after_blocks[idx_block]
    if int(mix_index) == 0:
        expected_before_values = [
            (before_idx_a, 0, element_index),
            (before_idx_b, 1, element_index),
        ]
        for j, k in source_pairs:
            j_block = _state_block_index_for_element(j, matrix_dim, block_size)
            k_block = _state_block_index_for_element(k, matrix_dim, block_size)
            before_j_a, _before_j_b = before_blocks[j_block]
            _before_k_a, before_k_b = before_blocks[k_block]
            expected_before_values.append((before_j_a, 0, j))
            expected_before_values.append((before_k_b, 1, k))
        for rows, matrix_id, element in expected_before_values:
            expected = workspace_i8_value(
                workspace_seed=int(workspace_seed),
                pass_index=int(pass_index),
                round_index=int(round_index),
                matrix_id=matrix_id,
                row=int(element) // int(matrix_dim),
                col=int(element) % int(matrix_dim),
                matrix_dim=int(matrix_dim),
            )
            actual = _state_block_value(rows, int(element), int(matrix_dim), int(block_size))
            if int(actual) != int(expected):
                return False, "transition_initial_state_value_mismatch"
    source_values = []
    for lane, (j, k) in enumerate(source_pairs):
        j_block = _state_block_index_for_element(j, matrix_dim, block_size)
        k_block = _state_block_index_for_element(k, matrix_dim, block_size)
        before_j_a, _before_j_b = before_blocks[j_block]
        _before_k_a, before_k_b = before_blocks[k_block]
        source_values.append((
            lane,
            _state_block_value(before_j_a, j, matrix_dim, block_size),
            _state_block_value(before_k_b, k, matrix_dim, block_size),
        ))
    expected_a, expected_b = transition_expected_values(
        workspace_seed=int(workspace_seed),
        pass_index=int(pass_index),
        round_index=int(round_index),
        mix_index=int(mix_index),
        element_index=element_index,
        ai=_state_block_value(before_idx_a, element_index, matrix_dim, block_size),
        bi=_state_block_value(before_idx_b, element_index, matrix_dim, block_size),
        sources=source_values,
    )
    actual_a = _state_block_value(after_idx_a, element_index, matrix_dim, block_size)
    actual_b = _state_block_value(after_idx_b, element_index, matrix_dim, block_size)
    if int(actual_a) != int(expected_a) or int(actual_b) != int(expected_b):
        return False, "transition_value_mismatch"
    return True, ""


def verify_workspace_proof_payload(
    *,
    proof: Mapping[str, Any],
    expected_transcript_root: str,
    expected_pass0_root: str,
    expected_final_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    pass_count: int,
    proof_challenge_seed_hex: str = "",
) -> tuple[bool, str]:
    proof_format = str(proof.get("format") or "")
    if proof_format not in {WORKSPACE_PROOF_FORMAT, WORKSPACE_MIXED_PROOF_FORMAT, WORKSPACE_TRANSITION_PROOF_FORMAT}:
        return False, "unsupported_proof_payload_format"
    try:
        matrix_dim = int(proof.get("matrix_dim"))
        block_size = int(proof.get("block_size"))
        rounds = int(proof.get("rounds"))
        spot_checks = int(proof.get("spot_checks", 1))
        workspace_seed = int(proof.get("workspace_seed"))
        memory_mix_rounds = int(proof.get("memory_mix_rounds", 0))
        transition_mix_rounds = int(proof.get("transition_mix_rounds", 0))
        transition_fanout = int(proof.get("transition_fanout", 1))
    except Exception:
        return False, "invalid_workspace_proof_parameters"
    if matrix_dim <= 0 or block_size <= 0 or rounds <= 0 or matrix_dim % block_size != 0:
        return False, "invalid_workspace_proof_parameters"
    if memory_mix_rounds < 0 or transition_mix_rounds < 0 or transition_fanout <= 0:
        return False, "invalid_workspace_proof_parameters"
    if proof_format == WORKSPACE_MIXED_PROOF_FORMAT and memory_mix_rounds <= 0:
        return False, "invalid_workspace_proof_parameters"
    if proof_format == WORKSPACE_PROOF_FORMAT and memory_mix_rounds != 0:
        return False, "invalid_workspace_proof_parameters"
    if proof_format == WORKSPACE_TRANSITION_PROOF_FORMAT:
        if memory_mix_rounds != 0 or transition_mix_rounds <= 0:
            return False, "invalid_workspace_proof_parameters"
    elif transition_mix_rounds != 0:
        return False, "invalid_workspace_proof_parameters"
    root_chain = proof.get("root_chain")
    if not isinstance(root_chain, list) or len(root_chain) != int(pass_count):
        return False, "root_chain_length_mismatch"
    for idx, entry in enumerate(root_chain):
        if int(entry.get("pass_index", -1)) != idx:
            return False, "root_chain_pass_index_mismatch"
        round_roots = entry.get("round_roots")
        if not isinstance(round_roots, list) or len(round_roots) != rounds:
            return False, "round_root_chain_length_mismatch"
        if pass_root_from_round_roots(idx, round_roots) != str(entry.get("pass_root") or ""):
            return False, "pass_root_mismatch"

    if workspace_transcript_root(root_chain) != str(expected_transcript_root):
        return False, "transcript_root_mismatch"
    if str(root_chain[0].get("pass_root")) != str(expected_pass0_root):
        return False, "pass0_root_mismatch"
    if str(root_chain[-1].get("pass_root")) != str(expected_final_root):
        return False, "final_root_mismatch"

    expected_seed = workspace_seed_for(lease_id, int(gpu_index), str(proof_seed_hex or ""))
    if int(workspace_seed) != int(expected_seed):
        return False, "workspace_seed_mismatch"
    challenge_seed = str(proof_challenge_seed_hex or expected_transcript_root)
    proof_challenge_seed = str(proof.get("proof_challenge_seed") or "")
    if proof_challenge_seed_hex and proof_challenge_seed != str(proof_challenge_seed_hex):
        return False, "proof_challenge_seed_mismatch"

    sampled = proof.get("sampled")
    block_proof = proof.get("block_proof")
    if not isinstance(sampled, dict) or not isinstance(block_proof, dict):
        return False, "missing_sampled_block_proof"
    try:
        pass_index = int(sampled.get("pass_index"))
        round_index = int(sampled.get("round_index"))
        block_index = int(sampled.get("block_index"))
    except Exception:
        return False, "invalid_sampled_indices"
    if pass_index < 0 or pass_index >= int(pass_count):
        return False, "sampled_pass_out_of_range"
    expected_pass = derive_sampled_pass_index(
        challenge_seed,
        lease_id,
        int(gpu_index),
        int(pass_count),
    )
    if pass_index != expected_pass:
        return False, "sampled_pass_index_mismatch"
    expected_round = derive_sampled_round_index(
        challenge_seed,
        lease_id,
        int(gpu_index),
        pass_index,
        rounds,
    )
    if round_index != expected_round:
        return False, "sampled_round_index_mismatch"
    round_root = str(root_chain[pass_index]["round_roots"][round_index])
    state_root_chain = proof.get("state_root_chain")
    sampled_transition_mix: int | None = None
    sampled_transition_before_root = ""
    sampled_transition_after_root = ""
    sampled_gemm_state_root = ""
    if proof_format == WORKSPACE_TRANSITION_PROOF_FORMAT:
        if not isinstance(state_root_chain, list) or len(state_root_chain) != int(pass_count):
            return False, "state_root_chain_length_mismatch"
        for idx, entry in enumerate(state_root_chain):
            if int(entry.get("pass_index", -1)) != idx:
                return False, "state_root_chain_pass_index_mismatch"
            state_rounds = entry.get("round_state_roots")
            if not isinstance(state_rounds, list) or len(state_rounds) != rounds:
                return False, "state_round_root_chain_length_mismatch"
            for state_roots in state_rounds:
                if not isinstance(state_roots, list) or len(state_roots) != transition_mix_rounds + 1:
                    return False, "state_mix_root_chain_length_mismatch"
        sampled_transition_mix = derive_sampled_transition_mix_index(
            challenge_seed,
            lease_id,
            int(gpu_index),
            pass_index,
            round_index,
            transition_mix_rounds,
        )
        sampled_state_roots = state_root_chain[pass_index]["round_state_roots"][round_index]
        sampled_transition_before_root = str(sampled_state_roots[sampled_transition_mix])
        sampled_transition_after_root = str(sampled_state_roots[sampled_transition_mix + 1])
        sampled_gemm_state_root = str(sampled_state_roots[-1])
    blocks_per_row = matrix_dim // block_size
    expected_block = derive_challenge_block_index(
        round_root_hex=round_root,
        lease_id=lease_id,
        gpu_index=int(gpu_index),
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=challenge_seed,
    )
    if block_index != expected_block:
        return False, "sampled_block_index_mismatch"

    ok, reason = verify_workspace_gemm_block_proof(
        block_proof_data=block_proof,
        round_root_hex=round_root,
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        spot_checks=spot_checks,
        memory_mix_rounds=memory_mix_rounds,
        state_root_hex=sampled_gemm_state_root,
        state_spot_openings=proof.get("gemm_state_openings") or [],
    )
    if not ok:
        return False, reason

    if proof_format == WORKSPACE_TRANSITION_PROOF_FORMAT:
        sampled_transition = proof.get("sampled_transition")
        if not isinstance(sampled_transition, dict):
            return False, "missing_sampled_transition_proof"
        try:
            transition_pass = int(sampled_transition.get("pass_index"))
            transition_round = int(sampled_transition.get("round_index"))
            transition_mix = int(sampled_transition.get("mix_index"))
        except Exception:
            return False, "invalid_sampled_transition_indices"
        if (
            sampled_transition_mix is None
            or transition_pass != pass_index
            or transition_round != round_index
            or transition_mix != sampled_transition_mix
        ):
            return False, "sampled_transition_index_mismatch"
        if str(sampled_transition.get("before_root") or "") != sampled_transition_before_root:
            return False, "sampled_transition_before_root_mismatch"
        if str(sampled_transition.get("after_root") or "") != sampled_transition_after_root:
            return False, "sampled_transition_after_root_mismatch"
        ok, reason = verify_workspace_transition_opening(
            opening=sampled_transition.get("opening") or {},
            before_root_hex=sampled_transition_before_root,
            after_root_hex=sampled_transition_after_root,
            workspace_seed=workspace_seed,
            pass_index=pass_index,
            round_index=round_index,
            mix_index=sampled_transition_mix,
            matrix_dim=matrix_dim,
            block_size=block_size,
            lease_id=lease_id,
            gpu_index=int(gpu_index),
            challenge_seed_hex=challenge_seed,
            transition_fanout=transition_fanout,
        )
        if not ok:
            return False, reason
    return True, ""


def verify_workspace_compact_proof_payload(
    *,
    proof: Mapping[str, Any],
    expected_transcript_root: str,
    expected_pass0_root: str,
    expected_final_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    pass_count: int,
) -> tuple[bool, str]:
    """Verify a deferred proof built around the calibrated compact timing path.

    The timed benchmark remains the original compact root chain. After the
    miner emits final_timing, it recomputes the sampled pass and opens a
    verifier-checkable GEMM block from the final round of that pass.
    """
    if str(proof.get("format") or "") != WORKSPACE_COMPACT_PROOF_FORMAT:
        return False, "unsupported_proof_payload_format"
    try:
        matrix_dim = int(proof.get("matrix_dim"))
        block_size = int(proof.get("block_size"))
        rounds = int(proof.get("rounds"))
        spot_checks = int(proof.get("spot_checks", 1))
        workspace_seed = int(proof.get("workspace_seed"))
    except Exception:
        return False, "invalid_workspace_proof_parameters"
    if matrix_dim <= 0 or block_size <= 0 or rounds <= 0 or matrix_dim % block_size != 0:
        return False, "invalid_workspace_proof_parameters"

    root_chain = proof.get("root_chain")
    if not isinstance(root_chain, list) or len(root_chain) != int(pass_count):
        return False, "root_chain_length_mismatch"
    for root in root_chain:
        if (
            not isinstance(root, list)
            or len(root) != 4
            or not all(isinstance(word, int) for word in root)
        ):
            return False, "invalid_root_chain"

    if transcript_root(root_chain) != str(expected_transcript_root):
        return False, "transcript_root_mismatch"
    if root_words_digest(root_chain[0]) != str(expected_pass0_root):
        return False, "pass0_root_mismatch"
    if root_words_digest(root_chain[-1]) != str(expected_final_root):
        return False, "final_root_mismatch"

    expected_seed = workspace_seed_for(lease_id, int(gpu_index), str(proof_seed_hex or ""))
    if int(workspace_seed) != int(expected_seed):
        return False, "workspace_seed_mismatch"

    sampled = proof.get("sampled")
    block_proof = proof.get("block_proof")
    if not isinstance(sampled, dict) or not isinstance(block_proof, dict):
        return False, "missing_sampled_block_proof"
    try:
        pass_index = int(sampled.get("pass_index"))
        round_index = int(sampled.get("round_index"))
        block_index = int(sampled.get("block_index"))
        sampled_root = sampled.get("compact_root")
        sampled_digest = str(sampled.get("compact_root_digest") or "")
        merkle_root = str(sampled.get("merkle_root") or "")
    except Exception:
        return False, "invalid_sampled_indices"
    if pass_index < 0 or pass_index >= int(pass_count):
        return False, "sampled_pass_out_of_range"
    expected_pass = derive_sampled_pass_index(
        str(expected_transcript_root),
        lease_id,
        int(gpu_index),
        int(pass_count),
    )
    if pass_index != expected_pass:
        return False, "sampled_pass_index_mismatch"
    if round_index != rounds - 1:
        return False, "sampled_round_index_mismatch"
    if sampled_root != root_chain[pass_index]:
        return False, "sampled_root_mismatch"
    if sampled_digest != root_words_digest(root_chain[pass_index]):
        return False, "sampled_root_digest_mismatch"
    try:
        bytes.fromhex(merkle_root.removeprefix("0x"))
    except ValueError:
        return False, "invalid_sampled_merkle_root"

    blocks_per_row = matrix_dim // block_size
    expected_block = derive_challenge_block_index(
        round_root_hex=merkle_root,
        lease_id=lease_id,
        gpu_index=int(gpu_index),
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
    )
    if block_index != expected_block:
        return False, "sampled_block_index_mismatch"

    ok, reason = verify_workspace_gemm_block_proof(
        block_proof_data=block_proof,
        round_root_hex=merkle_root,
        workspace_seed=workspace_seed,
        pass_index=pass_index,
        round_index=round_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        spot_checks=spot_checks,
    )
    return (True, "") if ok else (False, reason)
