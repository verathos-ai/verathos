"""Relaunch bind races must not spawn doomed backends or trip the breaker.

Killing a backend releases its process before the kernel releases its
listener, so a respawn racing the teardown binds against the dying
instance, exits immediately, and used to feed the crash-loop breaker
with failures that said nothing about the backend.
"""

from __future__ import annotations

import inspect
import socket

from verallm.mesh import cli as cli_mod
from verallm.mesh.cli import _wait_port_free


def _grab_port() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_free_port_returns_true_immediately():
    s, port = _grab_port()
    s.close()
    assert _wait_port_free("127.0.0.1", port, timeout_s=0.0) is True


def test_held_port_returns_false_at_zero_timeout():
    s, port = _grab_port()
    try:
        assert _wait_port_free("127.0.0.1", port, timeout_s=0.0) is False
    finally:
        s.close()


def test_wait_returns_true_once_holder_releases():
    s, port = _grab_port()
    ticks = {"n": 0}

    def clock() -> float:
        return float(ticks["n"])

    def sleeper(_dt: float) -> None:
        ticks["n"] += 1
        if ticks["n"] == 2:
            s.close()

    assert (
        _wait_port_free(
            "127.0.0.1", port, timeout_s=10.0, clock=clock, sleeper=sleeper
        )
        is True
    )


def test_held_port_times_out_false():
    s, port = _grab_port()
    ticks = {"n": 0}

    def clock() -> float:
        return float(ticks["n"])

    def sleeper(_dt: float) -> None:
        ticks["n"] += 1

    try:
        assert (
            _wait_port_free(
                "127.0.0.1", port, timeout_s=3.0, clock=clock, sleeper=sleeper
            )
            is False
        )
    finally:
        s.close()


def test_spawn_path_waits_for_port_and_excludes_bind_race():
    src = inspect.getsource(cli_mod)
    spawn_at = src.index('cmd = list(payload["llama_server_command"])')
    block = src[spawn_at - 3500 : spawn_at]
    # The respawn must gate on the port being free after terminating the
    # previous instance, and near-instant deaths with the port still held
    # must not count toward the crash breaker.
    assert "_wait_port_free(" in block
    assert "bind_race" in block
    assert "not counting toward" in block
