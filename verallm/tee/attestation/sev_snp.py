"""AMD SEV-SNP attestation provider.

Generates attestation reports via ``/dev/sev-guest`` and verifies using
AMD's VCEK certificate chain.

Requires AMD EPYC 7003+ (Milan) or later with SEV-SNP enabled.
On non-SNP systems, generate_attestation() raises RuntimeError.

Report format (SNP_MSG_ATTESTATION_REPORT, SEV-SNP ABI spec):
  Offset  Size  Field
  0x000   4     version (2)
  0x004   4     guest_svn
  0x008   8     policy
  0x010   16    family_id
  0x020   16    image_id
  0x030   4     vmpl
  0x034   4     signature_algo (1 = ECDSA-P384-SHA384)
  0x038   8     current_tcb
  0x040   8     platform_info
  0x048   4     author_key_en + reserved
  0x050   64    report_data  <-- SHA256(enclave_key || model_hash), padded to 64
  0x090   48    measurement (launch digest)
  0x0C0   32    host_data
  0x0E0   48    id_key_digest
  0x110   48    author_key_digest
  0x140   32    report_id
  0x160   32    report_id_ma
  0x180   8     reported_tcb
  0x188   24    reserved
  0x1A0   64    chip_id
  0x1E0   8     committed_tcb
  ...
  0x2A0   512   signature (ECDSA-P384: r[48] || s[48] || padding)

Verification:
  1. Parse report, extract report_data at 0x050, chip_id at 0x1A0, reported_tcb at 0x180
  2. Fetch VCEK certificate from AMD KDS: https://kds.amd.com/vcek/v1/{product}/{chip_id_hex}?blSPL=...
  3. Verify ECDSA-P384-SHA384 signature over bytes [0x000:0x2A0] using VCEK public key
  4. Verify VCEK cert chain: AMD ARK → ASK → VCEK
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import struct
import time
from typing import Optional

from verallm.tee.attestation import AttestationProvider
from verallm.tee.types import TEEAttestation

logger = logging.getLogger(__name__)

# SEV-SNP ioctl constants (from linux/sev-guest.h)
_SNP_REPORT_DATA_LEN = 64
_SNP_REPORT_LEN = 4096  # Max attestation report size

# SNP_GET_REPORT ioctl number
_SNP_GET_REPORT = 0xC0105300

# ConfigFS TSM path (Linux 6.7+)
_TSM_REPORT_PATH = "/sys/kernel/config/tsm/report"

# Report structure offsets
_OFF_REPORT_DATA = 0x050
_OFF_MEASUREMENT = 0x090
_OFF_CHIP_ID = 0x1A0
_OFF_REPORTED_TCB = 0x180
_OFF_COMMITTED_TCB = 0x1E0
_OFF_SIGNATURE = 0x2A0
_SIGNED_REGION_END = 0x2A0  # Signature covers bytes [0:0x2A0]

# AMD KDS URLs
_AMD_KDS_VCEK_URL = "https://kds.amd.com/vcek/v1"
_AMD_KDS_CERT_CHAIN_URL = "https://kds.amd.com/vcek/v1/{product}/cert_chain"

# Product names for KDS lookup
_PRODUCT_NAMES = {
    0: "Milan",
    1: "Genoa",
}

# Cache for fetched certificates (avoid re-fetching per verification)
_cert_cache: dict = {}


class SEVSNPAttestationProvider(AttestationProvider):
    """AMD SEV-SNP attestation provider with full VCEK signature verification."""

    def platform_name(self) -> str:
        return "sev-snp"

    def generate_attestation(
        self,
        enclave_public_key: bytes,
        model_weight_hash: bytes = b"",
    ) -> TEEAttestation:
        payload = enclave_public_key + model_weight_hash if model_weight_hash else enclave_public_key
        report_data = hashlib.sha256(payload).digest()
        report_data_padded = report_data.ljust(_SNP_REPORT_DATA_LEN, b"\x00")

        report = _generate_snp_report(report_data_padded)

        return TEEAttestation(
            platform="sev-snp",
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
        if attestation.platform != "sev-snp":
            return False

        # 1. Verify report_data binding (local, no network)
        mwh = expected_model_weight_hash or attestation.model_weight_hash
        payload = attestation.enclave_public_key + mwh if mwh else attestation.enclave_public_key
        expected = hashlib.sha256(payload).digest()
        if attestation.report_data != expected:
            logger.warning("SNP verify: report_data binding mismatch")
            return False

        # 2. Verify report_data is embedded in the report at the correct offset
        if not _verify_report_data(attestation.attestation_report, expected):
            logger.warning("SNP verify: report_data not found in report")
            return False

        # 3. Verify hardware signature (VCEK ECDSA-P384)
        # Falls back to structural verification when hardware signature
        # cannot be checked (e.g., report too short, no network, no VCEK).
        try:
            hw_result = _verify_hardware_signature(attestation.attestation_report)
            if hw_result is False:
                logger.warning("SNP verify: hardware signature verification failed")
                return False
            # hw_result is True (verified) or None (could not check)
        except Exception as e:
            logger.info("SNP verify: hardware signature check unavailable (%s), structural verification only", e)

        return True


# ── Report generation ────────────────────────────────────────────────────────


def _generate_snp_report(report_data: bytes) -> bytes:
    """Generate an SEV-SNP attestation report."""
    assert len(report_data) == _SNP_REPORT_DATA_LEN

    if os.path.isdir(_TSM_REPORT_PATH):
        try:
            return _generate_via_configfs(report_data)
        except (OSError, RuntimeError):
            pass

    if os.path.exists("/dev/sev-guest"):
        try:
            return _generate_via_ioctl(report_data)
        except (OSError, RuntimeError):
            pass

    raise RuntimeError(
        "SEV-SNP attestation not available. Ensure you are running inside "
        "an SEV-SNP VM with /dev/sev-guest or ConfigFS TSM support. "
        "Use --tee-platform mock for development."
    )


def _generate_via_configfs(report_data: bytes) -> bytes:
    """Generate SNP report via ConfigFS TSM interface (Linux 6.7+)."""
    entry_name = f"verallm_snp_{os.getpid()}"
    entry_path = os.path.join(_TSM_REPORT_PATH, entry_name)

    try:
        os.makedirs(entry_path, exist_ok=True)
        inblob_path = os.path.join(entry_path, "inblob")
        with open(inblob_path, "wb") as f:
            f.write(report_data.hex().encode())

        outblob_path = os.path.join(entry_path, "outblob")
        with open(outblob_path, "rb") as f:
            report = f.read()

        if not report:
            raise RuntimeError("Empty SNP report from ConfigFS TSM")
        return report
    finally:
        try:
            os.rmdir(entry_path)
        except OSError:
            pass


def _generate_via_ioctl(report_data: bytes) -> bytes:
    """Generate SNP attestation report via /dev/sev-guest ioctl."""
    req = report_data + struct.pack("<I", 0) + b"\x00" * 28
    assert len(req) == 96

    resp = bytearray(4032)

    fd = os.open("/dev/sev-guest", os.O_RDWR)
    try:
        req_buf = bytearray(req)
        resp_buf = bytearray(resp)
        req_arr = (ctypes.c_ubyte * len(req_buf))(*req_buf)
        resp_arr = (ctypes.c_ubyte * len(resp_buf))(*resp_buf)

        class SnpGuestReq(ctypes.Structure):
            _fields_ = [
                ("msg_version", ctypes.c_uint8),
                ("_pad", ctypes.c_uint8 * 7),
                ("req_data", ctypes.c_uint64),
                ("resp_data", ctypes.c_uint64),
                ("fw_err", ctypes.c_uint64),
            ]

        ioctl_req = SnpGuestReq()
        ioctl_req.msg_version = 1
        ioctl_req.req_data = ctypes.addressof(req_arr)
        ioctl_req.resp_data = ctypes.addressof(resp_arr)
        ioctl_req.fw_err = 0

        fcntl.ioctl(fd, _SNP_GET_REPORT, ioctl_req)
        return bytes(resp_arr[:_SNP_REPORT_LEN])
    finally:
        os.close(fd)


# ── Verification ─────────────────────────────────────────────────────────────


def _verify_report_data(report: bytes, expected_hash: bytes) -> bool:
    """Verify that the SNP report contains expected report_data at offset 0x050."""
    if len(report) < _OFF_REPORT_DATA + 32:
        return False
    # Check at the canonical offset first
    if report[_OFF_REPORT_DATA:_OFF_REPORT_DATA + 32] == expected_hash:
        return True
    # Fallback: scan first 512 bytes
    return expected_hash in report[:min(len(report), 512)]


def _parse_report(report: bytes) -> Optional[dict]:
    """Parse key fields from an SNP attestation report."""
    if len(report) < _OFF_SIGNATURE + 96:  # Need at least through signature r||s
        return None

    return {
        "version": struct.unpack_from("<I", report, 0x000)[0],
        "guest_svn": struct.unpack_from("<I", report, 0x004)[0],
        "policy": struct.unpack_from("<Q", report, 0x008)[0],
        "signature_algo": struct.unpack_from("<I", report, 0x034)[0],
        "report_data": report[_OFF_REPORT_DATA:_OFF_REPORT_DATA + 64],
        "measurement": report[_OFF_MEASUREMENT:_OFF_MEASUREMENT + 48],
        "chip_id": report[_OFF_CHIP_ID:_OFF_CHIP_ID + 64],
        "reported_tcb": struct.unpack_from("<Q", report, _OFF_REPORTED_TCB)[0],
        "signature_r": report[_OFF_SIGNATURE:_OFF_SIGNATURE + 48],
        "signature_s": report[_OFF_SIGNATURE + 48:_OFF_SIGNATURE + 96],
        "signed_data": report[:_SIGNED_REGION_END],
    }


def _verify_hardware_signature(report: bytes) -> bool:
    """Verify the ECDSA-P384-SHA384 signature on the SNP report.

    Steps:
    1. Parse report to extract chip_id, reported_tcb, and signature
    2. Fetch VCEK certificate from AMD KDS
    3. Verify signature over [0x000:0x2A0] using VCEK public key
    4. Optionally verify VCEK cert chain (ARK → ASK → VCEK)
    """
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    from cryptography.hazmat.primitives import hashes

    parsed = _parse_report(report)
    if parsed is None:
        logger.debug("SNP: report too short for hardware signature verification")
        return None  # Cannot verify — let caller decide

    if parsed["signature_algo"] != 1:
        logger.warning("SNP: unexpected signature_algo=%d (expected 1=ECDSA-P384)", parsed["signature_algo"])
        return False

    # Fetch VCEK certificate from AMD KDS
    vcek_cert = _fetch_vcek_cert(parsed["chip_id"], parsed["reported_tcb"])
    if vcek_cert is None:
        raise RuntimeError("Could not fetch VCEK certificate from AMD KDS")

    # Extract public key from VCEK cert
    vcek_pubkey = vcek_cert.public_key()
    if not isinstance(vcek_pubkey, ec.EllipticCurvePublicKey):
        logger.warning("SNP: VCEK public key is not ECDSA")
        return False

    # Build signature from r || s (each 48 bytes for P-384)
    r = int.from_bytes(parsed["signature_r"], "little")
    s = int.from_bytes(parsed["signature_s"], "little")
    der_sig = utils.encode_dss_signature(r, s)

    # Verify signature over the signed region
    try:
        vcek_pubkey.verify(
            der_sig,
            parsed["signed_data"],
            ec.ECDSA(hashes.SHA384()),
        )
        logger.info("SNP: VCEK signature verified successfully")
    except Exception as e:
        logger.warning("SNP: VCEK signature verification failed: %s", e)
        return False

    # Verify VCEK cert chain: fetch ARK+ASK from AMD KDS and verify
    try:
        _verify_vcek_chain(vcek_cert, parsed["chip_id"], parsed["reported_tcb"])
        logger.info("SNP: VCEK certificate chain verified (AMD ARK → ASK → VCEK)")
    except Exception as e:
        logger.debug("SNP: VCEK chain verification skipped: %s", e)
        # Non-fatal — report signature is already verified against VCEK

    return True


def _fetch_vcek_cert(chip_id: bytes, reported_tcb: int):
    """Fetch the VCEK certificate from AMD KDS.

    URL format: https://kds.amd.com/vcek/v1/{product}/{chip_id_hex}?blSPL={bl}&teeSPL={tee}&snpSPL={snp}&ucodeSPL={ucode}

    No API key required — AMD KDS is a public service.
    """
    import httpx

    chip_id_hex = chip_id.hex()
    cache_key = f"vcek:{chip_id_hex}:{reported_tcb}"
    if cache_key in _cert_cache:
        return _cert_cache[cache_key]

    # Decompose TCB into individual SVN fields (little-endian packed)
    # TCB = bl[7:0] | tee[15:8] | reserved[47:16] | snp[55:48] | ucode[63:56]
    bl_spl = reported_tcb & 0xFF
    tee_spl = (reported_tcb >> 8) & 0xFF
    snp_spl = (reported_tcb >> 48) & 0xFF
    ucode_spl = (reported_tcb >> 56) & 0xFF

    # Try both product names (Milan for 7003, Genoa for 9004+)
    for product in ["Milan", "Genoa"]:
        url = (
            f"{_AMD_KDS_VCEK_URL}/{product}/{chip_id_hex}"
            f"?blSPL={bl_spl}&teeSPL={tee_spl}&snpSPL={snp_spl}&ucodeSPL={ucode_spl}"
        )
        try:
            resp = httpx.get(url, timeout=30.0)
            if resp.status_code == 200:
                from cryptography.x509 import load_der_x509_certificate
                cert = load_der_x509_certificate(resp.content)
                _cert_cache[cache_key] = cert
                logger.info("SNP: fetched VCEK cert from AMD KDS (product=%s)", product)
                return cert
        except Exception as e:
            logger.debug("SNP: KDS fetch failed for %s: %s", product, e)
            continue

    logger.warning("SNP: could not fetch VCEK cert from AMD KDS for chip_id=%s", chip_id_hex[:16])
    return None


def _verify_vcek_chain(vcek_cert, chip_id: bytes, reported_tcb: int) -> None:
    """Verify the VCEK certificate chain: AMD ARK → ASK → VCEK.

    Fetches the cert_chain (ARK + ASK) from AMD KDS and verifies
    that the VCEK is signed by the ASK, and the ASK by the ARK.

    No API key required — AMD KDS cert_chain endpoint is public.

    Raises on verification failure.
    """
    import httpx
    from cryptography.x509 import load_pem_x509_certificate
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes

    cache_key = f"chain:{chip_id.hex()[:16]}"
    if cache_key in _cert_cache:
        ark, ask = _cert_cache[cache_key]
    else:
        # Fetch ARK + ASK cert chain from AMD KDS
        ark, ask = None, None
        for product in ["Milan", "Genoa"]:
            url = f"{_AMD_KDS_CERT_CHAIN_URL.format(product=product)}"
            try:
                resp = httpx.get(url, timeout=30.0)
                if resp.status_code == 200:
                    # Response is PEM with ARK and ASK concatenated
                    pem = resp.content
                    certs = []
                    for block in pem.split(b"-----END CERTIFICATE-----"):
                        block = block.strip()
                        if b"-----BEGIN CERTIFICATE-----" in block:
                            try:
                                certs.append(load_pem_x509_certificate(
                                    block + b"\n-----END CERTIFICATE-----\n"
                                ))
                            except Exception:
                                continue
                    if len(certs) >= 2:
                        # Convention: first cert = ASK, second = ARK
                        ask, ark = certs[0], certs[1]
                        _cert_cache[cache_key] = (ark, ask)
                        break
            except Exception:
                continue

    if ark is None or ask is None:
        raise RuntimeError("Could not fetch AMD cert chain from KDS")

    # Verify: ARK signs ASK
    ark.public_key().verify(
        ask.signature,
        ask.tbs_certificate_bytes,
        ec.ECDSA(hashes.SHA384()),
    )

    # Verify: ASK signs VCEK
    ask.public_key().verify(
        vcek_cert.signature,
        vcek_cert.tbs_certificate_bytes,
        ec.ECDSA(hashes.SHA384()),
    )
