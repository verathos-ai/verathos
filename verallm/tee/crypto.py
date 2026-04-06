"""X25519 ECDH key exchange + XSalsa20-Poly1305 authenticated encryption.

Uses ``nacl.public`` (already a dependency via pynacl).  No new packages.

Typical flow::

    # Enclave (miner) — one-time at startup
    enclave_sk, enclave_pk = generate_keypair()

    # User — per request
    user_sk, user_pk = generate_keypair()
    envelope = encrypt_payload(plaintext, user_sk, enclave_pk, session_id)

    # Enclave — decrypt
    plaintext = decrypt_payload(envelope, enclave_sk)

    # Enclave — encrypt response
    response_envelope = encrypt_payload(response, enclave_sk, user_pk, session_id)

    # User — decrypt response
    response = decrypt_payload(response_envelope, user_sk)
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from nacl.public import Box, PrivateKey, PublicKey
from nacl.utils import random as nacl_random

from verallm.tee.types import EncryptedEnvelope


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate an X25519 keypair.

    Returns:
        (private_key_bytes, public_key_bytes) — both 32 bytes.
    """
    sk = PrivateKey.generate()
    return bytes(sk), bytes(sk.public_key)


def encrypt_payload(
    plaintext: bytes,
    sender_private_key: bytes,
    recipient_public_key: bytes,
    session_id: str | None = None,
) -> EncryptedEnvelope:
    """Encrypt *plaintext* for *recipient_public_key*.

    Uses nacl.public.Box (X25519 shared secret → XSalsa20-Poly1305).
    """
    sk = PrivateKey(sender_private_key)
    pk = PublicKey(recipient_public_key)
    box = Box(sk, pk)
    nonce = nacl_random(Box.NONCE_SIZE)  # 24 bytes
    ciphertext = box.encrypt(plaintext, nonce).ciphertext
    return EncryptedEnvelope(
        session_id=session_id or str(uuid.uuid4()),
        sender_public_key=bytes(sk.public_key),
        nonce=nonce,
        ciphertext=ciphertext,
    )


def decrypt_payload(
    envelope: EncryptedEnvelope,
    recipient_private_key: bytes,
) -> bytes:
    """Decrypt an :class:`EncryptedEnvelope`.

    Raises ``nacl.exceptions.CryptoError`` on bad key / corrupted data.
    """
    sk = PrivateKey(recipient_private_key)
    pk = PublicKey(envelope.sender_public_key)
    box = Box(sk, pk)
    return box.decrypt(envelope.ciphertext, envelope.nonce)


# -- JSON convenience wrappers ------------------------------------------------


def encrypt_json(
    data: Dict[str, Any],
    sender_private_key: bytes,
    recipient_public_key: bytes,
    session_id: str | None = None,
) -> EncryptedEnvelope:
    """Encrypt a JSON-serializable dict."""
    plaintext = json.dumps(data, separators=(",", ":")).encode("utf-8")
    env = encrypt_payload(plaintext, sender_private_key, recipient_public_key, session_id)
    env.content_type = "application/json"
    return env


def decrypt_json(
    envelope: EncryptedEnvelope,
    recipient_private_key: bytes,
) -> Dict[str, Any]:
    """Decrypt an envelope whose plaintext is JSON."""
    plaintext = decrypt_payload(envelope, recipient_private_key)
    return json.loads(plaintext.decode("utf-8"))
