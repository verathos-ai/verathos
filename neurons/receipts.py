"""Service receipts — validator-signed proof of verified inference.

After a validator verifies miner inference (POST /inference + proof check),
it pushes a signed receipt to the miner via POST /epoch/receipt.  The receipt
includes performance metrics (ttft, tok/s, tokens generated) measured by the
validator and signed with its Ed25519 private key — unforgeable by the miner.

Miners accumulate receipts from ALL validators throughout an epoch.  At epoch
boundary, validators pull the complete receipt batch from each miner via
GET /epoch/{n}/receipts.  Every validator receives the SAME receipt set,
computes the SAME scores, and produces IDENTICAL weights (Yuma consensus).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass


@dataclass
class ServiceReceipt:
    """A validator-signed receipt proving verified inference occurred."""

    miner_address: str
    model_id: str
    model_index: int
    epoch_number: int  # epoch this receipt belongs to
    commitment_hash: bytes  # SHA256 of inference commitment (proves real work)
    timestamp: int

    # Performance metrics (measured by validator, signed -> unforgeable)
    ttft_ms: float  # time to first token
    tokens_generated: int  # output tokens produced
    generation_time_ms: float  # total generation wall time
    tokens_per_sec: float  # measured tok/s

    # Additional metrics
    prompt_tokens: int = 0  # input token count (for throughput budget accounting)
    proof_verified: bool = False  # did validator verify ZK proof for this request?
    proof_requested: bool = False  # was proof verification ATTEMPTED? (distinguishes "not tested" from "failed")
    tee_attestation_verified: object = None  # None = not tested, True = passed, False = failed
    is_canary: bool = False  # was this a canary test (vs organic user traffic)?

    # Validator identity
    validator_hotkey: bytes = b""  # 32-byte Ed25519 public key
    validator_signature: bytes = b""  # Ed25519 over all fields above


def encode_receipt_message(receipt: ServiceReceipt) -> bytes:
    """Encode receipt fields into a canonical byte string for signing/verification.

    Field order is fixed and unambiguous.  All strings are length-prefixed,
    all integers are big-endian 8-byte, all floats are big-endian double.
    """
    parts = []

    # String fields: length-prefixed UTF-8
    for s in (receipt.miner_address, receipt.model_id):
        encoded = s.encode("utf-8")
        parts.append(struct.pack(">I", len(encoded)))
        parts.append(encoded)

    # Integer fields
    parts.append(struct.pack(">q", receipt.model_index))
    parts.append(struct.pack(">q", receipt.epoch_number))

    # Fixed-length bytes
    parts.append(receipt.commitment_hash)

    # Timestamp
    parts.append(struct.pack(">q", receipt.timestamp))

    # Float metrics
    parts.append(struct.pack(">d", receipt.ttft_ms))
    parts.append(struct.pack(">q", receipt.tokens_generated))
    parts.append(struct.pack(">d", receipt.generation_time_ms))
    parts.append(struct.pack(">d", receipt.tokens_per_sec))

    # Additional metrics
    parts.append(struct.pack(">q", receipt.prompt_tokens))
    parts.append(struct.pack(">?", receipt.proof_verified))
    parts.append(struct.pack(">?", receipt.proof_requested))
    parts.append(struct.pack(">?", receipt.is_canary))

    # TEE attestation: encode as -1 (None/not tested), 0 (False), 1 (True)
    _tee_val = -1 if receipt.tee_attestation_verified is None else (1 if receipt.tee_attestation_verified else 0)
    parts.append(struct.pack(">b", _tee_val))

    # Validator hotkey
    parts.append(receipt.validator_hotkey)

    return b"".join(parts)


def sign_receipt(receipt: ServiceReceipt, private_key: bytes) -> ServiceReceipt:
    """Sign a receipt with an Ed25519 private key.

    Args:
        receipt: Receipt with all fields populated except ``validator_signature``.
        private_key: 32-byte Ed25519 seed (or 64-byte expanded key).

    Returns:
        A new ServiceReceipt with ``validator_signature`` populated.
    """
    from nacl.signing import SigningKey

    message = encode_receipt_message(receipt)

    # nacl expects 32-byte seed
    signing_key = SigningKey(private_key[:32])
    signed = signing_key.sign(message)

    return ServiceReceipt(
        miner_address=receipt.miner_address,
        model_id=receipt.model_id,
        model_index=receipt.model_index,
        epoch_number=receipt.epoch_number,
        commitment_hash=receipt.commitment_hash,
        timestamp=receipt.timestamp,
        ttft_ms=receipt.ttft_ms,
        tokens_generated=receipt.tokens_generated,
        generation_time_ms=receipt.generation_time_ms,
        tokens_per_sec=receipt.tokens_per_sec,
        prompt_tokens=receipt.prompt_tokens,
        proof_verified=receipt.proof_verified,
        proof_requested=receipt.proof_requested,
        tee_attestation_verified=receipt.tee_attestation_verified,
        is_canary=receipt.is_canary,
        validator_hotkey=receipt.validator_hotkey,
        validator_signature=signed.signature,
    )


def verify_service_receipt(
    receipt: ServiceReceipt,
    epoch_number: int,
    receipt_window_sec: float = 4500.0,
) -> bool:
    """Verify a service receipt's signature and freshness.

    Checks:
    1. Ed25519 signature is valid for the encoded message.
    2. ``epoch_number`` matches the expected epoch (prevents hoarding).
    3. Timestamp is within ``receipt_window_sec`` of now (~75 min for epoch).

    Returns:
        True if the receipt is valid and fresh.
    """
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError

    # Check epoch matches
    if receipt.epoch_number != epoch_number:
        return False

    # Check freshness
    now = int(time.time())
    if abs(now - receipt.timestamp) > receipt_window_sec:
        return False

    # Verify signature
    message = encode_receipt_message(receipt)
    try:
        verify_key = VerifyKey(receipt.validator_hotkey)
        verify_key.verify(message, receipt.validator_signature)
        return True
    except (BadSignatureError, Exception):
        return False


def create_receipt(
    miner_address: str,
    model_id: str,
    model_index: int,
    epoch_number: int,
    commitment_hash: bytes,
    ttft_ms: float,
    tokens_generated: int,
    generation_time_ms: float,
    tokens_per_sec: float,
    validator_hotkey: bytes,
    validator_private_key: bytes,
    prompt_tokens: int = 0,
    proof_verified: bool = False,
    proof_requested: bool = False,
    tee_attestation_verified: object = None,
    is_canary: bool = False,
) -> ServiceReceipt:
    """Convenience: build and sign a receipt in one call."""
    receipt = ServiceReceipt(
        miner_address=miner_address,
        model_id=model_id,
        model_index=model_index,
        epoch_number=epoch_number,
        commitment_hash=commitment_hash,
        timestamp=int(time.time()),
        ttft_ms=ttft_ms,
        tokens_generated=tokens_generated,
        generation_time_ms=generation_time_ms,
        tokens_per_sec=tokens_per_sec,
        prompt_tokens=prompt_tokens,
        proof_verified=proof_verified,
        proof_requested=proof_requested,
        tee_attestation_verified=tee_attestation_verified,
        is_canary=is_canary,
        validator_hotkey=validator_hotkey,
    )
    return sign_receipt(receipt, validator_private_key)


def receipt_to_dict(receipt: ServiceReceipt) -> dict:
    """Serialize a receipt to a JSON-safe dict (for HTTP transport)."""
    return {
        "miner_address": receipt.miner_address,
        "model_id": receipt.model_id,
        "model_index": receipt.model_index,
        "epoch_number": receipt.epoch_number,
        "commitment_hash": receipt.commitment_hash.hex(),
        "timestamp": receipt.timestamp,
        "ttft_ms": receipt.ttft_ms,
        "tokens_generated": receipt.tokens_generated,
        "generation_time_ms": receipt.generation_time_ms,
        "tokens_per_sec": receipt.tokens_per_sec,
        "prompt_tokens": receipt.prompt_tokens,
        "proof_verified": receipt.proof_verified,
        "proof_requested": receipt.proof_requested,
        "tee_attestation_verified": receipt.tee_attestation_verified,
        "is_canary": receipt.is_canary,
        "validator_hotkey": receipt.validator_hotkey.hex(),
        "validator_signature": receipt.validator_signature.hex(),
    }


def receipt_from_dict(d: dict) -> ServiceReceipt:
    """Deserialize a receipt from a JSON dict."""
    return ServiceReceipt(
        miner_address=d["miner_address"],
        model_id=d["model_id"],
        model_index=d["model_index"],
        epoch_number=d.get("epoch_number", d.get("poi_block", 0)),
        commitment_hash=bytes.fromhex(d["commitment_hash"]),
        timestamp=d["timestamp"],
        ttft_ms=d["ttft_ms"],
        tokens_generated=d["tokens_generated"],
        generation_time_ms=d["generation_time_ms"],
        tokens_per_sec=d["tokens_per_sec"],
        prompt_tokens=d.get("prompt_tokens", 0),
        proof_verified=d.get("proof_verified", False),
        proof_requested=d.get("proof_requested", False),
        tee_attestation_verified=d.get("tee_attestation_verified"),
        is_canary=d.get("is_canary", False),
        validator_hotkey=bytes.fromhex(d["validator_hotkey"]),
        validator_signature=bytes.fromhex(d["validator_signature"]),
    )
