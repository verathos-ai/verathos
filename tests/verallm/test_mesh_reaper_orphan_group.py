"""A dead registry anchor pid must not shield surviving group members.

An OOM kill can take the serve child (the recorded anchor) while its
llama-server survives in the same process group, squatting the GPU and
poisoning every later launch on the box. The boot reaper must kill the
group whenever anything in it is still alive, anchor dead or not.
"""

from __future__ import annotations

import os
import signal
import subprocess

from verallm.mesh import pool as pool_mod


def _spawn_group() -> tuple[subprocess.Popen, int]:
    """Start a throwaway sleeper in its own process group."""

    proc = subprocess.Popen(
        ["sleep", "300"], preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc, os.getpgid(proc.pid)


def test_dead_anchor_with_surviving_group_is_reaped(tmp_path, monkeypatch):
    sleeper, pgid = _spawn_group()
    try:
        # Anchor pid: something guaranteed dead. Use a child we reap fully.
        dead = subprocess.Popen(["true"])
        dead.wait()
        anchor = dead.pid
        # The registry names the dead anchor with the LIVE group's pgid.
        registry = [{"pid": anchor, "pgid": pgid}]

        killed: list[tuple[int, int]] = []
        real_killpg = os.killpg

        def fake_killpg(g, sig):
            killed.append((g, sig))
            return real_killpg(g, sig)

        monkeypatch.setattr(os, "killpg", fake_killpg)

        # Drive the same decision logic the reaper runs per record.
        own_pgid = os.getpgid(0)
        stale = []
        for record in registry:
            pid = int(record["pid"])
            g = int(record["pgid"])
            if g == own_pgid:
                continue
            try:
                if os.getpgid(pid) != g:
                    continue
            except OSError:
                if not pool_mod._process_group_alive(g):
                    continue
            stale.append(g)
            os.killpg(g, signal.SIGTERM)

        assert stale == [pgid], "surviving group must be selected for reaping"
        assert (pgid, signal.SIGTERM) in killed
        # Reap the zombie, or the group reads alive forever (the test is
        # the sleeper's parent; a real orphan is reparented and reaped by
        # the subreaper instead).
        sleeper.wait(timeout=5.0)
        assert not pool_mod._process_group_alive(pgid)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass


def test_dead_anchor_with_dead_group_is_skipped():
    dead = subprocess.Popen(["true"])
    dead.wait()
    anchor = dead.pid
    # A pgid that cannot exist: use the dead child's own pid as pgid; the
    # process is reaped, so the group query fails and liveness is false.
    assert not pool_mod._process_group_alive(anchor)


def test_reaper_source_uses_group_liveness_on_dead_anchor():
    import inspect

    src = inspect.getsource(pool_mod)
    marker = src.index("def _reap_previous_generation")
    block = src[marker : marker + 4000]
    assert "_process_group_alive(pgid)" in block.split("stale.append")[0], (
        "dead-anchor branch must consult group liveness before skipping"
    )
