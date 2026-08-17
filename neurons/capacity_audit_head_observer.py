#!/usr/bin/env python3
"""Process-isolated current-head observer for capacity receipt chronology."""

from __future__ import annotations

import argparse
import mmap
import os
import select
import signal
import struct
import threading
import time


STATE_STRUCT = struct.Struct("<QQQQI4xQ")
STATE_SIZE = STATE_STRUCT.size


def _block_number(block_header: object) -> int:
    value = getattr(block_header, "value", block_header)
    if not isinstance(value, dict):
        return 0
    header = value.get("header") if isinstance(value.get("header"), dict) else value
    number = header.get("number")
    if number is None:
        return 0
    try:
        return int(number, 0) if isinstance(number, str) else int(number)
    except (TypeError, ValueError):
        return 0


class _StateWriter:
    def __init__(self, path: str) -> None:
        self._fd = os.open(path, os.O_RDWR)
        self._map = mmap.mmap(self._fd, STATE_SIZE, access=mmap.ACCESS_WRITE)
        self._sequence = 0
        self._block = 0

    def publish(self, block: int, *, connected: bool) -> None:
        self._block = max(self._block, int(block))
        odd = self._sequence + 1
        even = self._sequence + 2
        # A seqlock makes a concurrent parent read either complete or
        # retryable without any process-shared Python lock.
        struct.pack_into("<Q", self._map, 0, odd)
        struct.pack_into(
            "<QQQI4xQ",
            self._map,
            8,
            self._block,
            time.time_ns(),
            time.monotonic_ns(),
            1 if connected else 0,
            odd,
        )
        struct.pack_into("<Q", self._map, STATE_SIZE - 8, even)
        struct.pack_into("<Q", self._map, 0, even)
        self._sequence = even

    def close(self) -> None:
        self._map.close()
        os.close(self._fd)


def _close_subtensor(subtensor_obj) -> None:
    for obj in (subtensor_obj, getattr(subtensor_obj, "substrate", None)):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def run(
    *,
    network: str,
    state_path: str,
    parent_pid: int,
    parent_fd: int,
    poll_seconds: float,
    watchdog_seconds: float,
) -> int:
    import bittensor as bt

    stopping = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    writer = _StateWriter(state_path)
    reconnect_delay = min(1.0, max(0.1, float(poll_seconds)))
    observer_generation = 0
    os.set_blocking(int(parent_fd), False)

    def parent_alive() -> bool:
        if os.getppid() != int(parent_pid):
            return False
        readable, _writable, _errors = select.select([int(parent_fd)], [], [], 0)
        if not readable:
            return True
        try:
            return os.read(int(parent_fd), 1) != b""
        except BlockingIOError:
            return True

    try:
        while not stopping.is_set() and parent_alive():
            observer_generation += 1
            generation = observer_generation
            subtensor_obj = None
            subscription_thread = None
            stream_error: list[BaseException] = []
            last_header_mono_ns = 0
            state_lock = threading.Lock()
            try:
                SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
                subtensor_obj = SubtensorCls(network=str(network))
                substrate = getattr(subtensor_obj, "substrate", None)
                subscribe = getattr(substrate, "subscribe_block_headers", None)
                current_method = getattr(subtensor_obj, "get_current_block", None)
                if callable(current_method):
                    try:
                        writer.publish(int(current_method()), connected=True)
                    except Exception:
                        pass

                if not callable(subscribe):
                    while (
                        not stopping.wait(max(0.1, float(poll_seconds)))
                        and parent_alive()
                    ):
                        try:
                            if callable(current_method):
                                writer.publish(int(current_method()), connected=True)
                        except Exception:
                            writer.publish(0, connected=False)
                            break
                    continue

                def callback(block_header):
                    nonlocal last_header_mono_ns
                    if (
                        stopping.is_set()
                        or not parent_alive()
                        or generation != observer_generation
                    ):
                        raise StopIteration("capacity head observer stopping")
                    block = _block_number(block_header)
                    if block > 0:
                        writer.publish(block, connected=True)
                        with state_lock:
                            last_header_mono_ns = time.monotonic_ns()
                    return None

                def subscribe_forever() -> None:
                    try:
                        subscribe(callback, finalized_only=False)
                    except BaseException as exc:
                        stream_error.append(exc)

                subscription_thread = threading.Thread(
                    target=subscribe_forever,
                    name="capacity-head-subscription",
                    daemon=True,
                )
                subscription_thread.start()
                started_ns = time.monotonic_ns()
                while not stopping.wait(min(1.0, reconnect_delay)):
                    if not parent_alive():
                        stopping.set()
                        break
                    if not subscription_thread.is_alive() or stream_error:
                        break
                    with state_lock:
                        reference_ns = last_header_mono_ns or started_ns
                    if (
                        time.monotonic_ns() - reference_ns
                        > int(max(5.0, float(watchdog_seconds)) * 1_000_000_000)
                    ):
                        break
            except BaseException:
                pass
            finally:
                writer.publish(0, connected=False)
                _close_subtensor(subtensor_obj)
            if not stopping.is_set():
                stopping.wait(reconnect_delay)
    finally:
        writer.publish(0, connected=False)
        writer.close()
        os.close(int(parent_fd))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--parent-fd", required=True, type=int)
    parser.add_argument("--poll-seconds", required=True, type=float)
    parser.add_argument("--watchdog-seconds", required=True, type=float)
    args = parser.parse_args()
    return run(
        network=args.network,
        state_path=args.state_path,
        parent_pid=args.parent_pid,
        parent_fd=args.parent_fd,
        poll_seconds=args.poll_seconds,
        watchdog_seconds=args.watchdog_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
