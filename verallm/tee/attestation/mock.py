"""Mock TEE attestation provider for development and testing.

Always generates valid attestations.  NOT suitable for production.
"""

from __future__ import annotations

import hashlib
import time

from verallm.tee.attestation import AttestationProvider
from verallm.tee.types import TEEAttestation


def _compute_report_data(pubkey: bytes, model_weight_hash: bytes = b"") -> bytes:
    """SHA256(pubkey || model_weight_hash) — shared binding logic."""
    payload = pubkey + model_weight_hash if model_weight_hash else pubkey
    return hashlib.sha256(payload).digest()


class MockAttestationProvider(AttestationProvider):
    """Mock provider: report_data = SHA256(pubkey || model_hash), always verifies."""

    def platform_name(self) -> str:
        return "mock"

    def generate_attestation(
        self,
        enclave_public_key: bytes,
        model_weight_hash: bytes = b"",
    ) -> TEEAttestation:
        report_data = _compute_report_data(enclave_public_key, model_weight_hash)
        report = b"MOCK_ATTESTATION_V2:" + report_data
        return TEEAttestation(
            platform="mock",
            enclave_public_key=enclave_public_key,
            attestation_report=report,
            report_data=report_data,
            timestamp=int(time.time()),
            model_weight_hash=model_weight_hash,
        )

    def verify_attestation(
        self,
        attestation: TEEAttestation,
        expected_model_weight_hash: bytes = b"",
    ) -> bool:
        if attestation.platform != "mock":
            return False
        mwh = expected_model_weight_hash or attestation.model_weight_hash
        expected = _compute_report_data(attestation.enclave_public_key, mwh)
        return attestation.report_data == expected
