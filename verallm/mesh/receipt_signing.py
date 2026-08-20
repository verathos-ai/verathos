"""Hotkey signing for mesh proof receipts.

A pool/mesh coordinator serving as a Verathos miner signs each receipt's
canonical hash with its Bittensor hotkey (Sr25519). The validator verifies the
signature against the hotkey's SS58 from the metagraph — the same trust model
as the vLLM miner path. Only the hotkey keypair is needed (no coldkey), so a
serving box never holds spend authority.

bittensor is imported lazily so the crypto-only import path (zkllm/verallm)
stays dependency-light; only a worker that actually signs pulls it in.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any

PLACEHOLDER_HOTKEY = "5PoolDriver"  # unsigned/operator-run coordinator
STAGE_PROOF_KEY_SCHEME = "sr25519"
STAGE_PROOF_RECEIPT_SIGNATURE_DOMAIN = b"verathos-mesh-stage-proof-receipt-v2:"
WORKER_CONTROL_SIGNATURE_DOMAIN = b"verathos-mesh-worker-control-v1:"
RECEIPT_SIGNATURE_DOMAIN = b"verathos-mesh-receipt-v1:"

_HEX_32_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_HEX_SEED_RE = re.compile(r"^[0-9a-f]{64}$")


def _keypair_from_ss58(ss58_address: str) -> Any:
    """A verify-only Keypair from an SS58, across bittensor/substrate versions."""
    try:
        from bittensor_wallet import Keypair  # modern bittensor
    except ImportError:
        from substrateinterface import Keypair  # legacy
    return Keypair(ss58_address=ss58_address)


def load_hotkey_keypair(wallet_name: str, hotkey_name: str) -> Any:
    """Load a Bittensor wallet's HOTKEY keypair (signing only, no coldkey).

    Returns a substrate Keypair (has .ss58_address, .sign, .verify).
    """
    import bittensor as bt

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    return wallet.hotkey  # raises if the hotkey file is absent


def _stage_keypair_from_seed(seed_hex: str) -> Any:
    """Build the one supported stage signer from a 32-byte secret seed."""

    if not isinstance(seed_hex, str) or not _HEX_SEED_RE.fullmatch(seed_hex):
        raise ValueError("stage proof key file must contain one 32-byte hex seed")
    try:
        from bittensor_wallet import Keypair
    except ImportError:  # pragma: no cover - legacy dependency layout
        from substrateinterface import Keypair
    keypair = Keypair.create_from_seed(seed_hex)
    if int(getattr(keypair, "crypto_type", 1)) != 1:
        raise ValueError("stage proof key must use Sr25519")
    validate_stage_proof_public_key(keypair.ss58_address, STAGE_PROOF_KEY_SCHEME)
    return keypair


def load_stage_proof_keypair_file(path: str | Path) -> Any:
    """Load an owner-only Sr25519 stage seed without accepting unsafe files.

    Stage keys authenticate proof-producing workers and are deliberately
    separate from the coordinator/miner hotkey.  Refuse symlinks, non-regular
    files, files owned by another local user, and any group/world access.
    """

    key_path = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(key_path, flags)
    except FileNotFoundError as exc:
        raise ValueError(f"stage proof key file does not exist: {key_path}") from exc
    except OSError as exc:
        # O_NOFOLLOW rejects a final symlink before it can be swapped between
        # a metadata check and a second path-based read.
        if key_path.is_symlink():
            raise ValueError(
                "stage proof key path must be a regular non-symlink file"
            ) from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("stage proof key path must be a regular non-symlink file")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise PermissionError("stage proof key file must be owned by the current user")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o077:
            raise PermissionError(
                "stage proof key file must not be accessible by group or world"
            )
        if not mode & stat.S_IRUSR:
            raise PermissionError("stage proof key file must be readable by its owner")
        with os.fdopen(descriptor, "r", encoding="ascii") as key_stream:
            descriptor = -1
            seed_hex = key_stream.read().strip()
    except UnicodeError as exc:
        raise ValueError("stage proof key file must contain ASCII hex") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _stage_keypair_from_seed(seed_hex)


def ensure_stage_proof_key_file(path: str | Path) -> Any:
    """Atomically create or load one persistent owner-only stage key."""

    key_path = Path(path).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(key_path, flags, 0o600)
    except FileExistsError:
        return load_stage_proof_keypair_file(key_path)

    try:
        seed = secrets.token_hex(32).encode("ascii") + b"\n"
        written = 0
        while written < len(seed):
            written += os.write(descriptor, seed[written:])
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        descriptor = -1
        try:
            key_path.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return load_stage_proof_keypair_file(key_path)


def load_hotkey_seed(
    wallet_name: str,
    hotkey_name: str,
    *,
    keypair: Any | None = None,
) -> bytes:
    """Load the 32-byte hotkey seed without ever touching the coldkey.

    Bittensor versions expose the decrypted seed either through the loaded
    keypair or only in the hotkey keyfile JSON.  Mesh coordinators need this
    seed to derive the same EVM identity used by ``neurons.miner`` for the
    public endpoint challenge.
    """

    kp = (
        keypair
        if keypair is not None
        else load_hotkey_keypair(
            wallet_name,
            hotkey_name,
        )
    )
    private_key = getattr(kp, "private_key", None)
    if private_key:
        seed = bytes(private_key[:32])
        if len(seed) == 32:
            return seed

    path = (
        Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = str(data.get("secretSeed", ""))
    if raw.startswith("0x"):
        raw = raw[2:]
    try:
        seed = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError("hotkey keyfile contains an invalid secretSeed") from exc
    if len(seed) != 32:
        raise ValueError("hotkey keyfile secretSeed must be 32 bytes")
    return seed


def sign_receipt_hash(receipt_hash_hex: str, keypair: Any) -> str:
    """Sign a receipt's canonical hash; return a hex signature."""
    if not receipt_hash_hex:
        return ""
    sig = keypair.sign(_message(receipt_hash_hex))
    return sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)


def verify_receipt_signature(
    receipt_hash_hex: str, signature_hex: str, ss58_address: str
) -> bool:
    """Verify a receipt signature against a hotkey SS58. False on any error.

    An empty signature from the placeholder hotkey is NOT valid — callers that
    require signed receipts must reject it; callers in permissive/operator mode
    may treat PLACEHOLDER_HOTKEY specially.
    """
    if not (receipt_hash_hex and signature_hex and ss58_address):
        return False
    try:
        kp = _keypair_from_ss58(ss58_address)
        return bool(kp.verify(_message(receipt_hash_hex), bytes.fromhex(signature_hex)))
    except Exception:
        return False


def validate_stage_proof_public_key(
    proof_key: str,
    proof_key_scheme: str,
) -> None:
    """Validate a public key supported by stage-receipt authentication.

    Snapshots previously accepted arbitrary opaque strings and advertised key
    schemes for which there was no verifier.  Fail closed until a concrete
    verifier is implemented for another scheme: the v2 stage-receipt protocol
    supports canonical Sr25519 SS58 public keys only.
    """

    if proof_key_scheme != STAGE_PROOF_KEY_SCHEME:
        raise ValueError(f"unsupported stage proof key scheme: {proof_key_scheme}")
    if not isinstance(proof_key, str) or not proof_key:
        raise ValueError("stage proof key must be a non-empty SS58 address")
    try:
        keypair = _keypair_from_ss58(proof_key)
    except Exception as exc:
        raise ValueError(
            "stage proof key must be a valid Sr25519 SS58 address"
        ) from exc
    canonical = getattr(keypair, "ss58_address", "")
    public_key = getattr(keypair, "public_key", b"")
    if canonical != proof_key or len(bytes(public_key)) != 32:
        raise ValueError("stage proof key must be a canonical Sr25519 SS58 address")
    try:
        from scalecodec.utils.ss58 import ss58_encode

        bittensor_address = ss58_encode(bytes(public_key), ss58_format=42)
    except Exception as exc:  # pragma: no cover - dependency/runtime corruption
        raise ValueError("stage proof key SS58 encoding is unavailable") from exc
    if proof_key != bittensor_address:
        raise ValueError("stage proof key must use the Bittensor SS58 format 42")


def stage_proof_receipt_signature_message(body_hash_hex: str) -> bytes:
    """Return the domain-separated message signed by an opaque stage key."""

    if not isinstance(body_hash_hex, str) or not _HEX_32_RE.fullmatch(body_hash_hex):
        raise ValueError("stage proof receipt body hash must be lowercase 32-byte hex")
    return STAGE_PROOF_RECEIPT_SIGNATURE_DOMAIN + body_hash_hex.encode("ascii")


def sign_stage_proof_receipt_body_hash(
    body_hash_hex: str,
    keypair: Any,
    *,
    expected_proof_key: str,
    proof_key_scheme: str = STAGE_PROOF_KEY_SCHEME,
) -> str:
    """Sign one public stage-receipt body with its dedicated stage key."""

    validate_stage_proof_public_key(expected_proof_key, proof_key_scheme)
    signer = getattr(keypair, "ss58_address", "")
    if signer != expected_proof_key:
        raise ValueError("stage receipt signer does not match snapshot proof key")
    crypto_type = getattr(keypair, "crypto_type", 1)
    if int(crypto_type) != 1:
        raise ValueError("stage receipt signer must use Sr25519")
    signature = keypair.sign(stage_proof_receipt_signature_message(body_hash_hex))
    if isinstance(signature, (bytes, bytearray)):
        signature_hex = bytes(signature).hex()
    else:
        signature_hex = str(signature)
        if signature_hex.startswith("0x"):
            signature_hex = signature_hex[2:]
    signature_hex = signature_hex.lower()
    if not _HEX_SIGNATURE_RE.fullmatch(signature_hex):
        raise ValueError("stage receipt signer returned an invalid Sr25519 signature")
    return signature_hex


def verify_stage_proof_receipt_signature(
    body_hash_hex: str,
    signature_hex: str,
    proof_key: str,
    proof_key_scheme: str,
) -> bool:
    """Verify a stage-receipt signature. Return false on every invalid input."""

    try:
        validate_stage_proof_public_key(proof_key, proof_key_scheme)
        if not isinstance(signature_hex, str) or not _HEX_SIGNATURE_RE.fullmatch(
            signature_hex
        ):
            return False
        keypair = _keypair_from_ss58(proof_key)
        return bool(
            keypair.verify(
                stage_proof_receipt_signature_message(body_hash_hex),
                bytes.fromhex(signature_hex),
            )
        )
    except Exception:
        return False


def _worker_control_signature_message(body_hash_hex: str) -> bytes:
    """Return the domain-separated message for one worker-control command."""

    if not isinstance(body_hash_hex, str) or not _HEX_32_RE.fullmatch(body_hash_hex):
        raise ValueError("worker control body hash must be lowercase 32-byte hex")
    return WORKER_CONTROL_SIGNATURE_DOMAIN + body_hash_hex.encode("ascii")


def sign_worker_control_body_hash(
    body_hash_hex: str,
    keypair: Any,
    *,
    expected_proof_key: str,
) -> str:
    """Sign a canonical worker-control body hash with its stage proof key."""

    validate_stage_proof_public_key(expected_proof_key, STAGE_PROOF_KEY_SCHEME)
    signer = getattr(keypair, "ss58_address", "")
    if signer != expected_proof_key:
        raise ValueError("worker control signer does not match expected stage proof key")
    if int(getattr(keypair, "crypto_type", 1)) != 1:
        raise ValueError("worker control signer must use Sr25519")
    signature = keypair.sign(_worker_control_signature_message(body_hash_hex))
    signature_hex = (
        bytes(signature).hex()
        if isinstance(signature, (bytes, bytearray))
        else str(signature)
    )
    if not _HEX_SIGNATURE_RE.fullmatch(signature_hex):
        raise ValueError("worker control signer returned an invalid Sr25519 signature")
    return signature_hex


def verify_worker_control_body_hash(
    body_hash_hex: str,
    signature_hex: str,
    expected_proof_key: str,
) -> bool:
    """Verify worker-control authentication, returning false on invalid input."""

    try:
        validate_stage_proof_public_key(
            expected_proof_key,
            STAGE_PROOF_KEY_SCHEME,
        )
        if not isinstance(signature_hex, str) or not _HEX_SIGNATURE_RE.fullmatch(
            signature_hex
        ):
            return False
        keypair = _keypair_from_ss58(expected_proof_key)
        return bool(
            keypair.verify(
                _worker_control_signature_message(body_hash_hex),
                bytes.fromhex(signature_hex),
            )
        )
    except Exception:
        return False


def _message(receipt_hash_hex: str) -> bytes:
    # Domain-separated so a receipt signature can never be replayed as some
    # other Verathos message signed by the same hotkey.
    return RECEIPT_SIGNATURE_DOMAIN + receipt_hash_hex.encode("ascii")
