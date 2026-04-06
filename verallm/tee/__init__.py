"""Optional TEE (Trusted Execution Environment) support for Verathos.

TEE adds end-to-end encryption between users and miner enclaves so neither
the miner operator nor the validator can read prompt/output content.  The
validator still verifies GEMM proofs (which don't contain plaintext).

TEE is fully opt-in:
  - Miners enable with ``--tee-enabled --tee-platform <platform>``
  - Users choose TEE-enabled miners via the validator proxy
  - The network works identically without TEE
"""

from verallm.tee.types import (
    EncryptedEnvelope,
    TEEAttestation,
    TEECapability,
)

__all__ = [
    "EncryptedEnvelope",
    "TEEAttestation",
    "TEECapability",
]
