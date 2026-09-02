"""Owned Subtensor connection cleanup helpers."""

from __future__ import annotations


def close_owned_subtensor(subtensor_obj: object | None) -> None:
    """Close an owned Subtensor connection without leaking its TCP socket.

    ``websockets`` performs a graceful close handshake by default.  A stale
    peer may time that handshake out before the socket is closed, leaving an
    established connection behind even though the caller attempted cleanup.
    Capacity workers create short-lived independent connections, so they must
    force-close the owned socket first and then run the normal cleanup path.
    """

    if subtensor_obj is None:
        return
    substrate = getattr(subtensor_obj, "substrate", None)
    websocket = getattr(substrate, "ws", None)
    close_socket = getattr(websocket, "close_socket", None)
    if callable(close_socket):
        try:
            close_socket()
        except Exception:
            pass

    seen: set[int] = set()
    for obj in (subtensor_obj, substrate):
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        close = getattr(obj, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            pass
