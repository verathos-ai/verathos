"""Zombie hygiene for the pool worker daemon.

Rented mesh containers run ``sleep infinity`` as PID 1, which never reaps:
a serve process killed mid-roll leaves its llama-server/rpc children as
permanent ``<defunct>`` entries. The daemon registers itself as a child
subreaper (adopting those orphans) and sweeps them each beat with
``reap_untracked_zombies`` - WITHOUT ever stealing the exit status of a
child a live ``subprocess.Popen`` still owns.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from verallm.mesh.pool import enable_child_subreaper, reap_untracked_zombies

pytestmark = pytest.mark.skipif(
    not hasattr(os, "waitid"), reason="requires POSIX waitid"
)

def _peek_flags() -> int:
    return os.WEXITED | os.WNOHANG | os.WNOWAIT


class _ScriptedWait:
    """Scripted ``os.waitid``/``os.waitpid`` pair for the sweep unit tests.

    ``zombies`` is the kernel's queue of dead-but-unreaped children in peek
    order; ``waitid(P_ALL, WNOWAIT)`` re-reports the head until it is
    reaped. ``live_children=True`` models children that exist but have not
    exited (waitid returns None instead of raising ChildProcessError).
    """

    def __init__(self, zombies: list[int], *, live_children: bool = False):
        self.zombies = list(zombies)
        self.live_children = live_children
        self.waitpid_calls: list[int] = []

    def waitid(self, idtype, ident, options):
        assert idtype == os.P_ALL
        assert ident == 0
        assert options == _peek_flags()
        if self.zombies:
            return SimpleNamespace(si_pid=self.zombies[0])
        if self.live_children:
            return None
        raise ChildProcessError

    def waitpid(self, pid, options):
        assert options == 0
        assert self.zombies and self.zombies[0] == pid, (
            "sweep must only reap the pid it just peeked"
        )
        self.zombies.pop(0)
        self.waitpid_calls.append(pid)
        return (pid, 0)


def _patch(monkeypatch, fake: _ScriptedWait) -> None:
    monkeypatch.setattr(os, "waitid", fake.waitid)
    monkeypatch.setattr(os, "waitpid", fake.waitpid)


def test_untracked_zombie_pends_then_reaps(monkeypatch):
    fake = _ScriptedWait([4242])
    _patch(monkeypatch, fake)
    # First sighting: never reaped immediately - an untracked in-flight
    # waiter (subprocess.run in a helper thread) may be microseconds from
    # collecting it itself.
    reaped, pending = reap_untracked_zombies(lambda: set(), frozenset())
    assert reaped == []
    assert pending == frozenset({4242})
    assert fake.waitpid_calls == []
    # Still there a sweep later: no waiter exists, reap it.
    reaped, pending = reap_untracked_zombies(lambda: set(), pending)
    assert reaped == [4242]
    assert pending == frozenset()
    assert fake.waitpid_calls == [4242]


def test_tracked_zombie_left_for_its_popen_owner(monkeypatch):
    fake = _ScriptedWait([555])
    _patch(monkeypatch, fake)
    for _ in range(3):
        reaped, pending = reap_untracked_zombies(
            lambda: {555}, frozenset()
        )
        assert reaped == []
        assert pending == frozenset()
    assert fake.waitpid_calls == []
    assert fake.zombies == [555]  # status stays observable by its owner


def test_no_children_stops_quietly(monkeypatch):
    fake = _ScriptedWait([])
    _patch(monkeypatch, fake)
    reaped, pending = reap_untracked_zombies(lambda: set(), frozenset())
    assert reaped == []
    assert pending == frozenset()
    assert fake.waitpid_calls == []


def test_live_children_none_dead_stops(monkeypatch):
    fake = _ScriptedWait([], live_children=True)
    _patch(monkeypatch, fake)
    reaped, pending = reap_untracked_zombies(lambda: set(), frozenset())
    assert reaped == []
    assert pending == frozenset()


def test_drains_multiple_confirmed_zombies_in_one_sweep(monkeypatch):
    fake = _ScriptedWait([11, 22])
    _patch(monkeypatch, fake)
    reaped, pending = reap_untracked_zombies(
        lambda: set(), frozenset({11, 22})
    )
    assert reaped == [11, 22]
    assert pending == frozenset()
    assert fake.waitpid_calls == [11, 22]


def test_tracked_head_blocks_queue_without_stealing(monkeypatch):
    # waitid re-reports the same head until it is reaped: a tracked dead
    # child at the head ends the sweep, and whatever queues behind it is
    # drained on a later sweep after the owner polls.
    fake = _ScriptedWait([555, 666])
    _patch(monkeypatch, fake)
    reaped, pending = reap_untracked_zombies(
        lambda: {555}, frozenset({666})
    )
    assert reaped == []
    assert pending == frozenset()
    assert fake.waitpid_calls == []


def test_tracked_pids_recollected_each_iteration(monkeypatch):
    # A Popen registered between reaps must be honoured mid-sweep.
    fake = _ScriptedWait([11, 22])
    _patch(monkeypatch, fake)
    calls = {"n": 0}

    def tracked() -> set[int]:
        calls["n"] += 1
        return {22}

    reaped, pending = reap_untracked_zombies(tracked, frozenset({11, 22}))
    assert reaped == [11]
    assert fake.waitpid_calls == [11]
    assert calls["n"] >= 2


# ---------------------------------------------------------------------------
# Real-process integration (Linux only: PR_SET_CHILD_SUBREAPER)
# ---------------------------------------------------------------------------


def _proc_state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return ""


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper is Linux-only")
def test_subreaper_adopts_and_sweep_reaps_orphaned_grandchild():
    assert enable_child_subreaper() is True

    # Middle process (stands in for a mesh serve): spawns a grandchild
    # (stands in for llama-server) that exits once the middle dies, then
    # parks. Killing the middle re-parents the grandchild to THIS process
    # (the subreaper), where it dies untracked - exactly the live zombie
    # shape observed on the rented boxes.
    middle_src = textwrap.dedent(
        """
        import os, subprocess, sys, time
        code = (
            "import os, sys, time\\n"
            + f"while os.getppid() == {os.getpid()}:\\n"
            + "    time.sleep(0.02)\\n"
        )
        grand = subprocess.Popen([sys.executable, "-c", code])
        print(grand.pid, flush=True)
        time.sleep(600)
        """
    )
    middle = subprocess.Popen(
        [sys.executable, "-c", middle_src],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        grand_pid = int(middle.stdout.readline().strip())
    except Exception:
        middle.kill()
        middle.wait()
        raise
    middle.kill()
    middle.wait()  # the tracked owner reaps its own child

    reaped: list[int] = []
    pending: frozenset[int] = frozenset()
    deadline = time.time() + 30.0
    while grand_pid not in reaped and time.time() < deadline:
        got, pending = reap_untracked_zombies(
            lambda: {middle.pid}, pending
        )
        reaped.extend(got)
        if grand_pid in reaped:
            break
        time.sleep(0.05)
    assert grand_pid in reaped, (
        f"orphaned grandchild {grand_pid} was never reaped "
        f"(state={_proc_state(grand_pid)!r})"
    )
    # Fully gone: reaping again must find no such child.
    with pytest.raises(ChildProcessError):
        os.waitpid(grand_pid, os.WNOHANG)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc + waitid semantics")
def test_sweep_never_steals_a_tracked_popen_exit_status():
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    deadline = time.time() + 30.0
    while _proc_state(proc.pid) != "Z" and time.time() < deadline:
        time.sleep(0.02)
    assert _proc_state(proc.pid) == "Z"

    pending: frozenset[int] = frozenset()
    for _ in range(3):
        reaped, pending = reap_untracked_zombies(
            lambda: {proc.pid}, pending
        )
        assert proc.pid not in reaped
    # poll()/wait() still observe the real status - nothing was stolen.
    assert proc.wait(timeout=10) == 7
