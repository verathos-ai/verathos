"""TEE attestation provider interface and factory.

Each supported TEE platform implements :class:`AttestationProvider`.
Use :func:`get_attestation_provider` to get the right implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from verallm.tee.types import TEEAttestation


class AttestationProvider(ABC):
    """Abstract base for TEE attestation generation and verification."""

    @abstractmethod
    def platform_name(self) -> str:
        """Return platform identifier (e.g. ``'tdx'``, ``'sev-snp'``)."""
        ...

    @abstractmethod
    def generate_attestation(
        self,
        enclave_public_key: bytes,
        model_weight_hash: bytes = b"",
    ) -> TEEAttestation:
        """Generate a platform attestation binding *enclave_public_key*.

        When *model_weight_hash* is provided, the resulting ``report_data``
        equals ``SHA256(enclave_public_key || model_weight_hash)``, binding
        both the enclave key and the model identity.  When empty, falls back
        to ``SHA256(enclave_public_key)`` for backward compatibility.
        """
        ...

    @abstractmethod
    def verify_attestation(
        self,
        attestation: TEEAttestation,
        expected_model_weight_hash: bytes = b"",
    ) -> bool:
        """Verify an attestation report is valid and binds the claimed key.

        Checks:
        1. Platform signature is valid (hardware root of trust)
        2. ``report_data`` matches ``SHA256(pubkey || model_weight_hash)``
        3. Measurements match expected values (optional, platform-specific)
        """
        ...

    def generate_reattestation(
        self,
        enclave_public_key: bytes,
        model_weight_hash: bytes,
        nonce: bytes,
    ) -> TEEAttestation:
        """Fresh attestation with validator-provided *nonce*.

        ``report_data = SHA256(enclave_public_key || model_weight_hash || nonce)``

        Proves the TEE is still live — old attestations don't contain the
        nonce, so replay is impossible.
        """
        # Default: delegates to generate_attestation with combined hash.
        # Subclasses may override for platform-specific nonce binding.
        import hashlib

        combined = hashlib.sha256(
            enclave_public_key + model_weight_hash + nonce
        ).digest()
        att = self.generate_attestation(enclave_public_key, combined)
        # Override report_data to include the nonce binding
        att.report_data = combined
        att.model_weight_hash = model_weight_hash
        return att


def get_attestation_provider(platform: str) -> AttestationProvider:
    """Factory: return the provider for *platform*.

    Raises :class:`ValueError` for unknown platforms.
    """
    platform = platform.strip().lower()
    if platform == "mock":
        from verallm.tee.attestation.mock import MockAttestationProvider

        return MockAttestationProvider()
    elif platform == "tdx":
        from verallm.tee.attestation.tdx import TDXAttestationProvider

        return TDXAttestationProvider()
    elif platform in {"sev-snp", "sev_snp"}:
        from verallm.tee.attestation.sev_snp import SEVSNPAttestationProvider

        return SEVSNPAttestationProvider()
    elif platform == "gpu":
        from verallm.tee.attestation.gpu import GpuAttestationProvider

        return GpuAttestationProvider()
    else:
        raise ValueError(
            f"Unknown TEE platform: {platform!r}. "
            "Supported: mock, tdx, sev-snp, gpu"
        )
