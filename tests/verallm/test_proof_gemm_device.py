"""Proof GEMM device selection is VRAM-aware.

The i8 GEMM torch path shares the GPU with whatever else the box runs
(typically the serving backend). Device selection must keep CUDA a
best-effort acceleration: an explicit override always wins, and CUDA is
chosen only when the transient working set plus a safety margin fits in
currently free VRAM, so proof generation can never starve a cohabitant
process into a device-wide OOM. The exact integer result is identical on
every device; these tests lock the selection logic only.
"""

from __future__ import annotations

import numpy as np
import torch

from verallm.mesh.ggml_proof import (
    _PROOF_GEMM_CUDA_MARGIN_BYTES,
    _PROOF_I8_F32_TILE_K,
    _proof_gemm_pick_device,
    _proof_i8_gemm_transient_bytes,
    _proof_i8_matmul_i64,
    _proof_i8_matmul_i64_torch,
)


def test_transient_bytes_counts_operands_and_chunk_buffers() -> None:
    rows, k, n = 4, 16, 8
    # Small shapes fit one chunk: i8 operands + int64 acc + f32 partial
    # + f32 casts of one k-tile.
    tile = min(k, _PROOF_I8_F32_TILE_K)
    expected = (rows * k + k * n) + rows * n * 8 + rows * n * 4 + (rows * tile + tile * n) * 4
    assert _proof_i8_gemm_transient_bytes(rows, k, n) == expected


def test_transient_bytes_chunking_caps_row_buffers() -> None:
    # Row counts beyond the chunk cap must not scale the accumulator
    # estimate: rows are chunked, so only the i8 operands grow with rows.
    n = 1 << 20
    chunk_cap = (1 << 30) // (n * 12)
    small = _proof_i8_gemm_transient_bytes(chunk_cap, 1, n)
    huge = _proof_i8_gemm_transient_bytes(chunk_cap * 100, 1, n)
    operands_delta = (chunk_cap * 100 - chunk_cap) * 1
    assert huge - small == operands_delta


def test_forced_device_always_wins(monkeypatch) -> None:
    monkeypatch.setenv("VERATHOS_PROOF_GEMM_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (1 << 40, 1 << 40), raising=False
    )
    assert _proof_gemm_pick_device(8, 8, 8).type == "cpu"


def test_no_cuda_picks_cpu(monkeypatch) -> None:
    monkeypatch.delenv("VERATHOS_PROOF_GEMM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _proof_gemm_pick_device(8, 8, 8).type == "cpu"


def test_cuda_picked_when_working_set_fits(monkeypatch) -> None:
    monkeypatch.delenv("VERATHOS_PROOF_GEMM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rows, k, n = 8, 16, 8
    free = _proof_i8_gemm_transient_bytes(rows, k, n) + _PROOF_GEMM_CUDA_MARGIN_BYTES
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (free, free), raising=False
    )
    assert _proof_gemm_pick_device(rows, k, n).type == "cuda"


def test_cpu_picked_when_free_vram_too_small(monkeypatch) -> None:
    monkeypatch.delenv("VERATHOS_PROOF_GEMM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rows, k, n = 8, 16, 8
    free = _proof_i8_gemm_transient_bytes(rows, k, n) + _PROOF_GEMM_CUDA_MARGIN_BYTES - 1
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (free, free), raising=False
    )
    assert _proof_gemm_pick_device(rows, k, n).type == "cpu"


def test_mem_get_info_failure_keeps_cuda(monkeypatch) -> None:
    # The probe is an optimization guard, not a correctness gate: if the
    # allocator cannot report free memory, behavior stays what it was.
    monkeypatch.delenv("VERATHOS_PROOF_GEMM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom() -> tuple[int, int]:
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom, raising=False)
    assert _proof_gemm_pick_device(8, 8, 8).type == "cuda"


def test_torch_path_matches_reference_on_cpu(monkeypatch) -> None:
    monkeypatch.setenv("VERATHOS_PROOF_GEMM_DEVICE", "cpu")
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, size=(5, 2100), dtype=np.int8)
    w = rng.integers(-128, 128, size=(2100, 33), dtype=np.int8)
    got = _proof_i8_matmul_i64_torch(x, w)
    assert got is not None
    np.testing.assert_array_equal(got, _proof_i8_matmul_i64(x, w))
