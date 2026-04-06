"""TEE-related data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TEEAttestation:
    """Platform attestation from a TEE enclave.

    Generated inside the enclave and includes a hardware-signed report
    binding ``enclave_public_key`` to the enclave measurements.
    """

    platform: str  # "tdx", "sev-snp", "mock"
    enclave_public_key: bytes  # 32-byte X25519 public key
    attestation_report: bytes  # Platform-specific attestation blob
    report_data: bytes  # SHA256(enclave_public_key || model_weight_hash) bound in the report
    timestamp: int  # Unix timestamp when attestation was generated
    model_weight_hash: bytes = b""  # SHA256(safetensors) bound in report_data
    pcr_digest: bytes = b""  # Optional: PCR/RTMR measurement chain


@dataclass
class TEECapability:
    """On-chain TEE capability for a miner (mirrors Solidity struct)."""

    enabled: bool = False
    platform: str = ""
    enclave_public_key: bytes = b""  # 32-byte X25519 public key
    attestation_hash: bytes = b""  # keccak256(attestation_report)
    attested_at_block: int = 0
    model_weight_hash: bytes = b""  # SHA256(safetensors) for on-chain cross-check


@dataclass
class EncryptedEnvelope:
    """Encrypted request/response for TEE E2E encryption.

    Uses nacl.public.Box (X25519 ECDH + XSalsa20-Poly1305).
    """

    session_id: str
    sender_public_key: bytes  # 32-byte X25519 ephemeral public key
    nonce: bytes  # 24-byte nacl nonce
    ciphertext: bytes  # Encrypted payload
    content_type: str = "application/json"
