"""Safe EVM signing-key resolution for administrative command-line tools."""

from __future__ import annotations

import re
import sys
from pathlib import Path


_PRIVATE_KEY_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_private_key(value: str, *, source: str) -> str:
    """Return one normalized 32-byte hex key without echoing it in errors."""

    candidate = str(value or "").strip()
    if candidate.startswith(("0x", "0X")):
        candidate = candidate[2:]
    if not _PRIVATE_KEY_RE.fullmatch(candidate):
        raise ValueError(f"{source} must contain exactly one 32-byte hex private key")
    return candidate.lower()


def resolve_cli_evm_private_key(
    *,
    wallet_name: str | None = None,
    hotkey_name: str = "default",
    private_key_file: str | Path | None = None,
    inline_private_key: str | None = None,
    required: bool = False,
) -> str | None:
    """Resolve an EVM private key from exactly one CLI authentication source.

    Wallet mode deliberately uses the shared cross-version hotkey seed loader:
    bittensor v9 exposes ``private_key`` on the loaded keypair, while v10 may
    expose the seed only through the hotkey keyfile.  Private-key files must be
    current-user-owned regular files with no group/world permissions.

    ``--private-key`` remains accepted for compatibility, but callers receive
    an explicit warning because process arguments are commonly visible through
    shell history and process inspection.  The key itself is never logged.
    """

    sources = sum(
        bool(value)
        for value in (wallet_name, private_key_file, inline_private_key)
    )
    if sources > 1:
        raise ValueError(
            "choose exactly one of --wallet, --private-key-file, or --private-key"
        )

    if wallet_name:
        from verallm.chain.wallet import derive_evm_private_key
        from verallm.mesh.receipt_signing import (
            load_hotkey_keypair,
            load_hotkey_seed,
        )

        keypair = load_hotkey_keypair(str(wallet_name), str(hotkey_name))
        seed = load_hotkey_seed(
            str(wallet_name),
            str(hotkey_name),
            keypair=keypair,
        )
        return _validate_private_key(
            derive_evm_private_key(seed),
            source="wallet-derived EVM key",
        )

    if private_key_file:
        from verallm.mesh.private_files import read_owner_only_text

        value = read_owner_only_text(
            private_key_file,
            label="EVM private key file",
        )
        return _validate_private_key(value, source="EVM private key file")

    if inline_private_key:
        print(
            "WARNING: --private-key is deprecated and exposes the signing key "
            "through process arguments; use --wallet or --private-key-file.",
            file=sys.stderr,
        )
        return _validate_private_key(
            inline_private_key,
            source="--private-key",
        )

    if required:
        raise ValueError(
            "one of --wallet, --private-key-file, or --private-key is required"
        )
    return None


__all__ = ["resolve_cli_evm_private_key"]
