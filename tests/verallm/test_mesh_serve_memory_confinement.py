"""Serve children are confined under memory.high on memory-capped boxes.

A GGUF load streams the model through page cache faster than a capped
container reclaims; hitting memory.max OOM-kills the worker tree
mid-formation. The spawn path places serve children in a sub-cgroup with
memory.high below the limit so the kernel throttles and reclaims instead
of killing. Everything fails open: no limit, no cgroup v2, or read-only
cgroupfs means the child runs unconfined exactly as before.
"""

from __future__ import annotations

import inspect

from verallm.mesh import pool as pool_mod
from verallm.mesh.pool import (
    _confine_serve_process_memory,
    _serve_memory_high_bytes,
)

GIB = 1 << 30


def _base(tmp_path, limit: str):
    (tmp_path / "memory.max").write_text(limit)
    (tmp_path / "cgroup.subtree_control").write_text("")
    return tmp_path


def test_high_is_limit_minus_margin(tmp_path):
    base = _base(tmp_path, str(30 * GIB))
    assert _serve_memory_high_bytes(base) == 26 * GIB


def test_unlimited_box_disables_confinement(tmp_path):
    base = _base(tmp_path, "max")
    assert _serve_memory_high_bytes(base) == 0


def test_small_limit_falls_back_to_half(tmp_path):
    base = _base(tmp_path, str(10 * GIB))
    assert _serve_memory_high_bytes(base) == 5 * GIB


def test_missing_cgroupfs_disables(tmp_path):
    assert _serve_memory_high_bytes(tmp_path / "nope") == 0


def test_env_override_wins(tmp_path, monkeypatch):
    base = _base(tmp_path, str(30 * GIB))
    monkeypatch.setenv("VERATHOS_SERVE_MEMORY_HIGH_BYTES", str(7 * GIB))
    assert _serve_memory_high_bytes(base) == 7 * GIB
    monkeypatch.setenv("VERATHOS_SERVE_MEMORY_HIGH_BYTES", "0")
    assert _serve_memory_high_bytes(base) == 0


def test_confine_writes_group_and_procs(tmp_path):
    base = _base(tmp_path, str(30 * GIB))
    got = _confine_serve_process_memory(4242, base)
    cg = tmp_path / "verathos-serve-4242"
    assert got == str(cg)
    assert (cg / "memory.high").read_text() == str(26 * GIB)
    assert (cg / "cgroup.procs").read_text() == "4242"


def test_confine_fails_open_without_limit(tmp_path):
    base = _base(tmp_path, "max")
    assert _confine_serve_process_memory(4242, base) == ""
    assert not (tmp_path / "verathos-serve-4242").exists()


def test_confine_sweeps_stale_empty_groups(tmp_path):
    base = _base(tmp_path, str(30 * GIB))
    stale = tmp_path / "verathos-serve-99"
    stale.mkdir()
    _confine_serve_process_memory(4242, base)
    assert not stale.exists()


def test_spawn_wires_confinement():
    src = inspect.getsource(pool_mod)
    marker = src.index("def _spawn(")
    block = src[marker : marker + 3000]
    assert "_confine_serve_process_memory(proc.pid)" in block
