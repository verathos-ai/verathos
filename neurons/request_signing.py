"""Request signing and verification for validator → miner authentication.

Validators sign every HTTP request to miners with their Sr25519 hotkey
(the native Substrate key type). Miners verify the signature and check
the signer against a cached metagraph validator list.

Uses Sr25519 (NOT Ed25519) because the metagraph provides Sr25519 public
keys. This lets the miner verify that the signer matches a known validator
hotkey without any key-mapping ambiguity.

Signing scheme:
    message = f"{method}:{path}:{sha256(body).hex()}:{timestamp}"
    signature = Sr25519.sign(message, hotkey_keypair)

Headers:
    X-Validator-Hotkey:    SS58-encoded hotkey address (string)
    X-Validator-Signature: 64-byte Sr25519 signature (hex)
    X-Validator-Timestamp: Unix seconds (string)

Replay protection: timestamp must be within MAX_CLOCK_SKEW of receiver's clock.
"""

from __future__ import annotations

import hashlib
import time

# Header names
HDR_HOTKEY = "X-Validator-Hotkey"
HDR_SIGNATURE = "X-Validator-Signature"
HDR_TIMESTAMP = "X-Validator-Timestamp"

# Allow 60 seconds of clock skew between validator and miner
MAX_CLOCK_SKEW = 60


def build_signing_message(
    method: str,
    path: str,
    body: bytes,
    timestamp: str,
) -> bytes:
    """Build the canonical message to sign/verify.

    Args:
        method: HTTP method (uppercase, e.g. "POST").
        path: URL path (e.g. "/inference").
        body: Raw request body bytes (empty bytes for GET).
        timestamp: Unix seconds as string.

    Returns:
        UTF-8 encoded message bytes.
    """
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{method}:{path}:{body_hash}:{timestamp}"
    return message.encode("utf-8")


def sign_request(
    method: str,
    path: str,
    body: bytes,
    hotkey_ss58: str,
    hotkey_seed: bytes,
) -> dict[str, str]:
    """Sign an HTTP request with Sr25519 and return auth headers.

    Args:
        method: HTTP method (e.g. "POST").
        path: URL path (e.g. "/inference").
        body: Raw request body bytes.
        hotkey_ss58: SS58-encoded hotkey address.
        hotkey_seed: 32-byte seed (Mini secret key).

    Returns:
        Dict of headers to add to the request.
    """
    from substrateinterface import Keypair

    timestamp = str(int(time.time()))
    message = build_signing_message(method, path, body, timestamp)

    keypair = Keypair.create_from_seed(hotkey_seed[:32].hex())
    signature = keypair.sign(message)

    return {
        HDR_HOTKEY: hotkey_ss58,
        HDR_SIGNATURE: signature.hex() if isinstance(signature, bytes) else signature,
        HDR_TIMESTAMP: timestamp,
    }


def verify_request(
    method: str,
    path: str,
    body: bytes,
    hotkey_ss58: str,
    signature_hex: str,
    timestamp_str: str,
) -> tuple[bool, str]:
    """Verify a signed request using Sr25519.

    Args:
        method: HTTP method.
        path: URL path.
        body: Raw request body bytes.
        hotkey_ss58: SS58-encoded hotkey address from header.
        signature_hex: Hex-encoded 64-byte Sr25519 signature from header.
        timestamp_str: Unix seconds string from header.

    Returns:
        (ok, reason) — True if valid, False with reason string.
    """
    from substrateinterface import Keypair

    # Check timestamp freshness
    try:
        ts = int(timestamp_str)
    except (ValueError, TypeError):
        return False, "invalid timestamp"

    now = int(time.time())
    if abs(now - ts) > MAX_CLOCK_SKEW:
        return False, f"timestamp too old ({abs(now - ts)}s skew, max {MAX_CLOCK_SKEW}s)"

    # Parse signature
    try:
        sig_hex = signature_hex
        if sig_hex.startswith("0x"):
            sig_hex = sig_hex[2:]
        signature_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        return False, "malformed hex in signature"

    if len(signature_bytes) != 64:
        return False, f"signature must be 64 bytes, got {len(signature_bytes)}"

    message = build_signing_message(method, path, body, timestamp_str)

    try:
        keypair = Keypair(ss58_address=hotkey_ss58)
        valid = keypair.verify(message, signature_bytes)
        if not valid:
            return False, "bad signature"
    except Exception as e:
        return False, f"verification error: {e}"

    return True, "ok"
