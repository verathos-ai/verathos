"""Authentication for stage-worker control commands."""

from __future__ import annotations

import pytest
from bittensor_wallet import Keypair

from verallm.mesh.receipt_signing import (
    sign_stage_proof_receipt_body_hash,
    sign_worker_control_body_hash,
    verify_stage_proof_receipt_signature,
    verify_worker_control_body_hash,
)


STAGE_KEY = Keypair.create_from_uri("//MeshWorkerControlStage")
OTHER_STAGE_KEY = Keypair.create_from_uri("//MeshWorkerControlOtherStage")
BODY_HASH = "2a" * 32


def _sign(body_hash: str = BODY_HASH) -> str:
    return sign_worker_control_body_hash(
        body_hash,
        STAGE_KEY,
        expected_proof_key=STAGE_KEY.ss58_address,
    )


def test_worker_control_signature_round_trip_is_canonical() -> None:
    signature = _sign()

    assert len(signature) == 128
    assert signature == signature.lower()
    assert verify_worker_control_body_hash(
        BODY_HASH,
        signature,
        STAGE_KEY.ss58_address,
    )


@pytest.mark.parametrize(
    "body_hash",
    (
        "",
        "2A" * 32,
        "2a" * 31,
        "2a" * 33,
        "gg" * 32,
    ),
)
def test_worker_control_signer_rejects_noncanonical_body_hash(
    body_hash: str,
) -> None:
    with pytest.raises(ValueError, match="lowercase 32-byte hex"):
        _sign(body_hash)


def test_worker_control_signer_must_match_expected_stage_proof_key() -> None:
    with pytest.raises(ValueError, match="does not match expected stage proof key"):
        sign_worker_control_body_hash(
            BODY_HASH,
            STAGE_KEY,
            expected_proof_key=OTHER_STAGE_KEY.ss58_address,
        )


def test_worker_control_signer_rejects_noncanonical_signature_output() -> None:
    class UppercaseSignatureKey:
        ss58_address = STAGE_KEY.ss58_address
        crypto_type = STAGE_KEY.crypto_type

        @staticmethod
        def sign(message: bytes) -> str:
            return STAGE_KEY.sign(message).hex().upper()

    with pytest.raises(ValueError, match="invalid Sr25519 signature"):
        sign_worker_control_body_hash(
            BODY_HASH,
            UppercaseSignatureKey(),
            expected_proof_key=STAGE_KEY.ss58_address,
        )


def test_worker_control_verifier_rejects_wrong_key_and_hash() -> None:
    signature = _sign()

    assert not verify_worker_control_body_hash(
        BODY_HASH,
        signature,
        OTHER_STAGE_KEY.ss58_address,
    )
    assert not verify_worker_control_body_hash(
        "2b" * 32,
        signature,
        STAGE_KEY.ss58_address,
    )


@pytest.mark.parametrize(
    "body_hash",
    (
        "",
        "2A" * 32,
        "2a" * 31,
        "2a" * 33,
        "gg" * 32,
    ),
)
def test_worker_control_verifier_rejects_noncanonical_body_hash(
    body_hash: str,
) -> None:
    assert not verify_worker_control_body_hash(
        body_hash,
        _sign(),
        STAGE_KEY.ss58_address,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda signature: "",
        lambda signature: signature.upper(),
        lambda signature: "0x" + signature,
        lambda signature: signature[:-2],
        lambda signature: signature + "00",
        lambda signature: "g" + signature[1:],
        lambda signature: ("0" if signature[0] != "0" else "1") + signature[1:],
    ),
)
def test_worker_control_verifier_strictly_rejects_invalid_signatures(mutate) -> None:
    signature = _sign()

    assert not verify_worker_control_body_hash(
        BODY_HASH,
        mutate(signature),
        STAGE_KEY.ss58_address,
    )


def test_worker_control_and_stage_receipt_domains_are_not_interchangeable() -> None:
    worker_control_signature = _sign()
    stage_receipt_signature = sign_stage_proof_receipt_body_hash(
        BODY_HASH,
        STAGE_KEY,
        expected_proof_key=STAGE_KEY.ss58_address,
    )

    assert not verify_worker_control_body_hash(
        BODY_HASH,
        stage_receipt_signature,
        STAGE_KEY.ss58_address,
    )
    assert not verify_stage_proof_receipt_signature(
        BODY_HASH,
        worker_control_signature,
        STAGE_KEY.ss58_address,
        "sr25519",
    )
