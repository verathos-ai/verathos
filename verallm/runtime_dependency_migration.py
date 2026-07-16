"""One-time runtime migrations for incompatible Python dependency upgrades."""

from __future__ import annotations

import fcntl
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path


_LOCK_PATH = Path(tempfile.gettempdir()) / "verathos-bittensor-codec-migration.lock"
_CY_SCALE_VERSION = "0.5.0"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _major(version: str | None) -> int:
    try:
        return int(str(version or "").split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _run_pip(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "pip command failed").strip()
        raise RuntimeError(detail)


def ensure_bittensor_codec_compatibility() -> None:
    """Remove the legacy SCALE codec before importing Bittensor 10.3+.

    Bittensor's async substrate client and the retired ``scalecodec`` package
    install the same Python namespace. Existing environments therefore need a
    one-time cleanup after pip resolves the new dependencies.
    """
    if _major(_distribution_version("async-substrate-interface")) < 2:
        return
    if not (
        _distribution_version("scalecodec")
        or _distribution_version("substrate-interface")
    ):
        return

    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if not (
            _distribution_version("scalecodec")
            or _distribution_version("substrate-interface")
        ):
            return

        print(
            "Verathos: migrating the Bittensor SCALE codec runtime...",
            file=sys.stderr,
        )
        _run_pip(
            [
                "uninstall",
                "-y",
                "substrate-interface",
                "scalecodec",
                "cyscale",
            ]
        )
        _run_pip(
            [
                "install",
                "--no-deps",
                "--force-reinstall",
                f"cyscale=={_CY_SCALE_VERSION}",
            ]
        )

        if _distribution_version("scalecodec") is not None:
            raise RuntimeError("legacy scalecodec package remains installed")
        if _distribution_version("cyscale") != _CY_SCALE_VERSION:
            raise RuntimeError("required cyscale runtime was not installed")
