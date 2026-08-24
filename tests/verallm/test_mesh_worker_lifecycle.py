"""Worker-lifecycle hardening: the untracked-serve failure class.

A worker daemon killed mid-drive (e.g. cgroup OOM) leaves its
coordinator + llama serving a mesh the manager has already failed: the
model's full VRAM held by processes no pool state references, silently
shadowing every later launch on that box. These tests pin the four
defenses:

1. the manager tells workers to TEAR DOWN serves for gone/terminal meshes
   (mesh_gone — covered in test_mesh_pool.py) and the worker loop wires
   that order into every report-delivery path;
2. the formation silent-driver budget scales with model size so heavy
   loads are not falsely reaped into exactly that state;
3. the daemon's post-fetch heap trim exists (download bloat made the
   daemon the OOM victim in the first place);
4. an orderly termination signal names the in-flight command instead of
   dying silently.
"""

from __future__ import annotations

import inspect

import verallm.mesh.pool as pool_module
from verallm.mesh.pool import (
    WORKER_STALE_S,
    _formation_silence_budget_s,
    _release_fetch_memory,
)


def test_formation_silence_budget_scales_with_model_bytes():
    # Small models keep the historical floor.
    assert _formation_silence_budget_s(0) == 120.0
    assert _formation_silence_budget_s(1_000_000_000) == 120.0
    # A mid-size GGUF can starve the driver's poll loop past the flat
    # floor while the load is healthy.
    assert _formation_silence_budget_s(21_713_462_848) > 300.0
    # Huge models clamp: a genuinely dead driver may not wedge a launch
    # for more than 15 minutes.
    assert _formation_silence_budget_s(120_000_000_000) == 900.0
    # The floor never drops below the worker-staleness multiple.
    assert _formation_silence_budget_s(1) >= 4 * WORKER_STALE_S


def test_release_fetch_memory_is_safe_everywhere():
    # Best-effort by contract: must never raise, on any platform (glibc
    # trim and /proc are both optional probes).
    _release_fetch_memory("test-model")


def test_fetch_releases_heap_before_the_drive_phase():
    src = inspect.getsource(pool_module.LocalMeshRunner.fetch)
    assert "_release_fetch_memory(" in src, (
        "fetch() must trim the download-inflated heap before the drive "
        "phase — the daemon being the fattest task is what made it the "
        "cgroup OOM victim mid-drive"
    )


def test_worker_loop_wires_mesh_gone_teardown_into_every_delivery_path():
    src = inspect.getsource(pool_module)
    # The teardown helper exists...
    assert "def _mesh_gone_teardown(" in src
    # ...and every delivery path consults it: the synchronous path and the
    # background retry redelivery. (Two call sites + the definition.)
    assert src.count("_mesh_gone_teardown(report_fields, response)") >= 2, (
        "both report-delivery paths must order the local teardown when the "
        "manager answers mesh_gone; a serve for an untracked mesh must "
        "never outlive that answer"
    )


def test_worker_daemon_logs_termination_signals():
    src = inspect.getsource(pool_module)
    assert "_log_termination_signal" in src
    assert "signal.signal(signal.SIGTERM, _log_termination_signal)" in src, (
        "an orderly stop must name the in-flight command before the daemon "
        "dies; a silent death mid-drive is undiagnosable from the manager"
    )


def test_manager_report_paths_mark_gone_meshes():
    src = inspect.getsource(pool_module.PoolManager.handle_report)
    assert src.count('"mesh_gone": True') >= 2, (
        "handle_report must tag BOTH the deleted-record and the "
        "terminal-status returns with mesh_gone"
    )
