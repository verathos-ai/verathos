"""NVIDIA GPU attestation provider using nv_attestation_sdk.

Collects GPU attestation evidence proving the GPU is running in
Confidential Computing (CC) mode with authentic NVIDIA firmware.
This extends the TEE trust boundary from CPU (TDX/SEV-SNP) to GPU VRAM.

Requires:
- CC-capable GPU (H100, H200, B200) in CC mode
- ``nv_attestation_sdk`` package (``pip install nv-attestation-sdk``)
"""

from __future__ import annotations

import hashlib
import time

from verallm.tee.attestation import AttestationProvider
from verallm.tee.types import TEEAttestation

try:
    from nv_attestation_sdk import attestation as nv_attestation

    _HAS_NV_SDK = True
except ImportError:
    _HAS_NV_SDK = False


class GpuAttestationProvider(AttestationProvider):
    """NVIDIA GPU attestation via nv_attestation_sdk.

    Generates GPU attestation evidence that proves:
    1. The GPU is genuine NVIDIA hardware
    2. The GPU is running in CC mode (VRAM encryption active)
    3. The GPU firmware is authentic and untampered

    The enclave public key is bound into the attestation nonce so verifiers
    can confirm the key belongs to this specific GPU attestation session.
    """

    def __init__(self, ppcie_mode: bool = False):
        """Initialize GPU attestation provider.

        Args:
            ppcie_mode: If True, collect evidence in PPCIe mode (used for
                multi-GPU setups like H200 x8 with NVSwitch). If False (default),
                collect evidence in standard CC mode.
        """
        self._ppcie_mode = ppcie_mode

    def platform_name(self) -> str:
        return "gpu"

    def generate_attestation(
        self,
        enclave_public_key: bytes,
        model_weight_hash: bytes = b"",
    ) -> TEEAttestation:
        """Generate GPU attestation evidence.

        The attestation binds the enclave public key and model identity by
        using ``SHA256(pubkey || model_weight_hash)`` as the nonce.
        """
        if not _HAS_NV_SDK:
            raise ImportError(
                "nv_attestation_sdk is required for GPU attestation. "
                "Install with: pip install nv-attestation-sdk"
            )

        payload = enclave_public_key + model_weight_hash if model_weight_hash else enclave_public_key
        report_data = hashlib.sha256(payload).digest()
        nonce_hex = report_data.hex()

        evidence = _gather_gpu_evidence(
            name="verallm-gpu-attestation",
            nonce=nonce_hex,
            ppcie_mode=self._ppcie_mode,
        )

        return TEEAttestation(
            platform="gpu",
            enclave_public_key=enclave_public_key,
            attestation_report=evidence,
            report_data=report_data,
            timestamp=int(time.time()),
            model_weight_hash=model_weight_hash,
        )

    def verify_attestation(
        self,
        attestation: TEEAttestation,
        expected_model_weight_hash: bytes = b"",
    ) -> bool:
        """Verify GPU attestation evidence.

        Checks:
        1. report_data == SHA256(pubkey || model_weight_hash)
        2. GPU evidence is valid (delegated to nv_attestation_sdk remote verifier)
        """
        if attestation.platform != "gpu":
            return False

        mwh = expected_model_weight_hash or attestation.model_weight_hash
        payload = attestation.enclave_public_key + mwh if mwh else attestation.enclave_public_key
        expected = hashlib.sha256(payload).digest()
        if attestation.report_data != expected:
            return False

        if not _HAS_NV_SDK:
            raise ImportError(
                "nv_attestation_sdk is required to verify GPU attestation."
            )

        return _verify_gpu_evidence(attestation.attestation_report)


def _gather_gpu_evidence(
    name: str,
    nonce: str,
    ppcie_mode: bool = False,
) -> bytes:
    """Collect GPU attestation evidence via nv_attestation_sdk.

    This calls the NVIDIA attestation SDK to produce a hardware-signed
    evidence blob proving the GPU is in CC mode with authentic firmware.

    Args:
        name: Attestation client name (for identification).
        nonce: Hex-encoded nonce (typically SHA256 of enclave pubkey).
        ppcie_mode: Whether to use PPCIe mode evidence collection.

    Returns:
        Raw evidence bytes from the GPU.
    """
    client = nv_attestation.Attestation()
    client.set_name(name)
    client.set_nonce(nonce)
    client.set_claims_version("3.0")

    client.add_verifier(
        nv_attestation.Devices.GPU,
        nv_attestation.Environment.REMOTE,
        "",
        "",
    )

    evidence = client.get_evidence(options={"ppcie_mode": ppcie_mode})

    if evidence is None:
        raise RuntimeError(
            "GPU attestation evidence collection failed. "
            "Ensure GPU is in CC mode and nvidia-attestation-sdk is configured."
        )

    if isinstance(evidence, str):
        evidence = evidence.encode("utf-8")

    return evidence


def _verify_gpu_evidence(evidence: bytes) -> bool:
    """Verify GPU attestation evidence via nv_attestation_sdk.

    Uses NVIDIA's remote verification service to validate the evidence.

    Args:
        evidence: Raw evidence bytes from :func:`_gather_gpu_evidence`.

    Returns:
        True if evidence is valid, False otherwise.
    """
    try:
        client = nv_attestation.Attestation()
        client.set_name("verallm-gpu-verify")
        client.set_claims_version("3.0")

        client.add_verifier(
            nv_attestation.Devices.GPU,
            nv_attestation.Environment.REMOTE,
            "",
            "",
        )

        result = client.validate_evidence(evidence)
        return bool(result)
    except Exception:
        return False
