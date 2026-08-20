"""Small helpers for local files that contain mesh control-plane secrets."""

from __future__ import annotations

import errno
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping


def read_owner_only_text(path: str | Path, *, label: str) -> str:
    """Read a regular, current-user-owned file with no group/world access.

    Opening the descriptor with ``O_NOFOLLOW`` (where supported) keeps the
    metadata checks and the read bound to the same file.  Callers deliberately
    receive no content in errors so credentials cannot leak through logs.
    """

    source = Path(path).expanduser()
    try:
        path_metadata = os.lstat(source)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {source}") from exc
    if stat.S_ISLNK(path_metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {source}") from exc
    except OSError as exc:
        if source.is_symlink():
            raise ValueError(f"{label} must be a regular non-symlink file") from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if (metadata.st_dev, metadata.st_ino) != (
            path_metadata.st_dev,
            path_metadata.st_ino,
        ):
            raise ValueError(f"{label} changed while it was being opened")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise PermissionError(f"{label} must be owned by the current user")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o077:
            # Deliberately fail-closed rather than auto-chmod: silently
            # blessing a secret that sat group/world-readable would hide
            # the exposure. Name the remedy; a bare refusal left operators
            # (and a hand-copied pool token) stuck on a
            # "stopped" worker with no stated fix.
            raise PermissionError(
                f"{label} must not be accessible by group or world "
                f"(fix: chmod 600 the file; rotate it if it was exposed)"
            )
        if not mode & stat.S_IRUSR:
            raise PermissionError(f"{label} must be readable by its owner")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_owner_only_text(path: str | Path, value: str) -> Path:
    """Atomically write text with mode 0600 without following the target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = value.encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
        # The file contents and rename are not fully power-loss durable until
        # the containing directory entry is synced as well. Pool command state
        # relies on this helper to survive an abrupt host restart.
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(target.parent, directory_flags)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(directory_fd)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def write_owner_only_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write the canonical on-disk JSON form used by mesh state files."""

    return write_owner_only_text(
        path,
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
    )
