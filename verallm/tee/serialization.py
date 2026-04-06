"""JSON-safe serialization for TEE envelope types."""

from __future__ import annotations

from typing import Any, Dict

from verallm.tee.types import EncryptedEnvelope, TEEAttestation, TEECapability


def envelope_to_dict(envelope: EncryptedEnvelope) -> Dict[str, Any]:
    """Serialize an :class:`EncryptedEnvelope` to a JSON-safe dict."""
    return {
        "session_id": envelope.session_id,
        "sender_public_key": envelope.sender_public_key.hex(),
        "nonce": envelope.nonce.hex(),
        "ciphertext": envelope.ciphertext.hex(),
        "content_type": envelope.content_type,
    }


def dict_to_envelope(d: Dict[str, Any]) -> EncryptedEnvelope:
    """Deserialize an :class:`EncryptedEnvelope` from a JSON-safe dict."""
    return EncryptedEnvelope(
        session_id=str(d["session_id"]),
        sender_public_key=bytes.fromhex(d["sender_public_key"]),
        nonce=bytes.fromhex(d["nonce"]),
        ciphertext=bytes.fromhex(d["ciphertext"]),
        content_type=str(d.get("content_type", "application/json")),
    )


def attestation_to_dict(att: TEEAttestation) -> Dict[str, Any]:
    """Serialize a :class:`TEEAttestation` to a JSON-safe dict."""
    return {
        "platform": att.platform,
        "enclave_public_key": att.enclave_public_key.hex(),
        "attestation_report": att.attestation_report.hex(),
        "report_data": att.report_data.hex(),
        "timestamp": att.timestamp,
        "model_weight_hash": att.model_weight_hash.hex() if att.model_weight_hash else "",
        "pcr_digest": att.pcr_digest.hex() if att.pcr_digest else "",
    }


def dict_to_attestation(d: Dict[str, Any]) -> TEEAttestation:
    """Deserialize a :class:`TEEAttestation` from a JSON-safe dict."""
    pcr = d.get("pcr_digest", "")
    mwh = d.get("model_weight_hash", "")
    return TEEAttestation(
        platform=str(d["platform"]),
        enclave_public_key=bytes.fromhex(d["enclave_public_key"]),
        attestation_report=bytes.fromhex(d["attestation_report"]),
        report_data=bytes.fromhex(d["report_data"]),
        timestamp=int(d["timestamp"]),
        model_weight_hash=bytes.fromhex(mwh) if mwh else b"",
        pcr_digest=bytes.fromhex(pcr) if pcr else b"",
    )


def capability_to_dict(cap: TEECapability) -> Dict[str, Any]:
    """Serialize a :class:`TEECapability` to a JSON-safe dict."""
    return {
        "enabled": cap.enabled,
        "platform": cap.platform,
        "enclave_public_key": cap.enclave_public_key.hex(),
        "attestation_hash": cap.attestation_hash.hex(),
        "attested_at_block": cap.attested_at_block,
        "model_weight_hash": cap.model_weight_hash.hex() if cap.model_weight_hash else "",
    }


def dict_to_capability(d: Dict[str, Any]) -> TEECapability:
    """Deserialize a :class:`TEECapability` from a JSON-safe dict."""
    return TEECapability(
        enabled=bool(d.get("enabled", False)),
        platform=str(d.get("platform", "")),
        enclave_public_key=bytes.fromhex(d.get("enclave_public_key", "")),
        attestation_hash=bytes.fromhex(d.get("attestation_hash", "")),
        attested_at_block=int(d.get("attested_at_block", 0)),
        model_weight_hash=bytes.fromhex(d["model_weight_hash"]) if d.get("model_weight_hash") else b"",
    )
