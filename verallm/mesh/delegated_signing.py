"""Coordinator signatures delegated from a pool worker to its pool manager.

A worker that DRIVES a subnet mesh has to produce two signatures under the
coordinator hotkey: the per-epoch verification snapshot, and each serving
receipt.  Requiring that wallet on every worker machine made "join with one
token" untrue -- an operator adding a second box had to copy hotkey material
onto it before the box could do anything but host stages.

The manager already holds the coordinator wallet (it renews the on-chain
model lease with it), so the worker asks the manager to sign instead of
holding a key it has no other use for.  Nothing on the wire changes: the
validator still receives exactly the same snapshot and receipt signatures
from exactly the same hotkey.

This is deliberately not a general signing oracle.  Two properties bound it:

* **Shape.** Only a message that is exactly ``<mesh domain><64 lowercase
  hex>`` is signable, and the domain must be one of the mesh domains below.
  A substrate extrinsic payload, a receipt for some other subnet, or any
  free-form text cannot be smuggled through, because the worker never sends
  bytes -- it sends a purpose name and a hash, and the manager rebuilds the
  message itself.
* **Authority.** The manager signs only for the worker it has itself
  assigned as the driver of the named mesh.  A member-only worker, or a
  worker naming a mesh it does not drive, is refused.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
from typing import Any, Callable, Sequence

from verallm.mesh.receipt_signing import RECEIPT_SIGNATURE_DOMAIN
from verallm.mesh.verification_snapshot import SNAPSHOT_SIGNATURE_DOMAIN

logger = logging.getLogger(__name__)

# Coordinator signing is cheap and normally completes in well under one
# second. Split the old single 20-second attempt into three bounded attempts:
# this survives a stale connection or short routing flap without lengthening
# the worst-case wait materially for a request whose manager is unreachable.
COORDINATOR_SIGN_ATTEMPTS = 3
COORDINATOR_SIGN_ATTEMPT_TIMEOUT_S = 8.0
COORDINATOR_SIGN_RETRY_DELAYS_S = (0.25, 0.75)
# Capacity final receipts live inside a much tighter chain-derived delivery
# window than inference receipts. Three short attempts stay below ten seconds
# including backoff, leaving the validator's transport grace available for the
# actual artifact POST.
CAPACITY_COORDINATOR_SIGN_ATTEMPT_TIMEOUT_S = 3.0
CAPACITY_COORDINATOR_SIGN_RETRY_DELAYS_S = (0.25, 0.5)

# Purpose name -> the domain prefix the manager will rebuild the message
# with. Purposes are named on the wire; raw domains are never sent, so a
# future domain cannot be reached by a worker running older code.
COORDINATOR_SIGN_DOMAINS: dict[str, bytes] = {
    "verification-snapshot": SNAPSHOT_SIGNATURE_DOMAIN,
    "receipt": RECEIPT_SIGNATURE_DOMAIN,
}

#: A validator proving-of-control challenge: the manager signs the 32-byte
#: nonce (EIP-191, nonce || address) with the coordinator's EVM key. This is
#: not in the table above because it is secp256k1 over a different message
#: construction, but it is bounded the same way: the caller supplies a nonce,
#: never bytes, so an Ethereum TRANSACTION (RLP, not an EIP-191 personal
#: message) can never be produced through it.
IDENTITY_CHALLENGE_PURPOSE = "identity-challenge"

#: A capacity-audit artifact (pass0/final receipt or proof payload) signed
#: with the coordinator's EVM key -- the same address the mesh's chain entry
#: registered, so validator-side ``recover_artifact_signer`` verification is
#: byte-identical to a vLLM miner's. Like the identity challenge this is
#: secp256k1 and therefore not in COORDINATOR_SIGN_DOMAINS, and it is bounded
#: the same way: the worker sends the full artifact DICT, never message
#: bytes, and the manager rebuilds the EIP-191 text itself from the fixed
#: ``VERATHOS_CAPACITY_AUDIT_ARTIFACT_V1`` framing -- an Ethereum transaction
#: (RLP) can never be produced through it. Unlike every other purpose it is
#: open to ANY worker bound to the mesh, not just the driver: capacity
#: openings run on every member's GPUs simultaneously, and each member
#: pushes its own signed artifacts.
CAPACITY_AUDIT_ARTIFACT_PURPOSE = "capacity-audit-artifact"

#: Upper bound on one artifact's canonical JSON, BY TYPE. Receipts are
#: ~1 KiB and stay under the tight bound; the sampled proof payload
#: measures ~254 KiB for a 27b-class GPU and ~11.2 MiB for an A100-class
#: opening. The ceiling stays below the
#: validator ingest's 32 MiB cap.
MAX_CAPACITY_ARTIFACT_BYTES = 64 * 1024
MAX_CAPACITY_PROOF_ARTIFACT_BYTES = 20 * 1024 * 1024


def max_capacity_artifact_bytes(artifact_type: str) -> int:
    """Signing-size ceiling for one capacity artifact type."""

    if str(artifact_type) == "capacity_audit_proof_payload":
        return MAX_CAPACITY_PROOF_ARTIFACT_BYTES
    return MAX_CAPACITY_ARTIFACT_BYTES

#: Artifact types the manager will sign. Mirrors what the validator ingest
#: routes accept; anything else is refused before a key is touched.
CAPACITY_ARTIFACT_TYPES = (
    "capacity_audit_pass0_receipt",
    "capacity_audit_final_receipt",
    "capacity_audit_proof_payload",
)

_HEX_32_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)


def _coordinator_sign_transport_retryable(exc: BaseException) -> bool:
    """Return whether a coordinator-sign failure is safe to retry.

    Only transport failures and explicitly transient HTTP statuses qualify.
    Authentication, authority, request-shape and signature failures remain
    fail-fast: retry must never turn the manager channel into a softer signing
    policy.
    """

    match = _HTTP_STATUS_RE.search(str(exc))
    if match is not None:
        status = int(match.group(1))
        return status in {408, 425, 429} or 500 <= status <= 599

    current: BaseException | None = exc
    while current is not None:
        if isinstance(
            current,
            (
                TimeoutError,
                ConnectionError,
                urllib.error.URLError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "failed to connect",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "remote end closed",
            "broken pipe",
        )
    )


def coordinator_sign_request_with_retry(
    request_once: Callable[[float], dict[str, Any]],
    *,
    attempts: int = COORDINATOR_SIGN_ATTEMPTS,
    attempt_timeout_s: float = COORDINATOR_SIGN_ATTEMPT_TIMEOUT_S,
    retry_delays_s: Sequence[float] = COORDINATOR_SIGN_RETRY_DELAYS_S,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Perform a bounded, fail-closed delegated signing request.

    ``request_once`` receives the per-attempt timeout and is deliberately
    invoked anew for every attempt. Callers therefore rebuild the authenticated
    worker-control body (fresh timestamp and nonce) while retaining the same
    signing purpose and content digest. A lost response is safe to retry
    because coordinator signing is idempotent for that digest.
    """

    attempts = max(1, int(attempts))
    timeout_s = max(0.1, float(attempt_timeout_s))
    delays = tuple(max(0.0, float(value)) for value in retry_delays_s)
    sleep_fn = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            return request_once(timeout_s)
        except Exception as exc:
            if attempt >= attempts or not _coordinator_sign_transport_retryable(exc):
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 0.0
            logger.warning(
                "coordinator-sign transport interrupted; retrying attempt %d/%d "
                "in %.2fs",
                attempt + 1,
                attempts,
                delay,
            )
            sleep_fn(delay)
    raise AssertionError("unreachable coordinator-sign retry state")


def coordinator_sign_retry_profile(
    purpose: object,
) -> tuple[int, float, tuple[float, ...]]:
    """Return the bounded retry profile for one delegated signing purpose."""

    if str(purpose or "") == CAPACITY_AUDIT_ARTIFACT_PURPOSE:
        return (
            COORDINATOR_SIGN_ATTEMPTS,
            CAPACITY_COORDINATOR_SIGN_ATTEMPT_TIMEOUT_S,
            CAPACITY_COORDINATOR_SIGN_RETRY_DELAYS_S,
        )
    return (
        COORDINATOR_SIGN_ATTEMPTS,
        COORDINATOR_SIGN_ATTEMPT_TIMEOUT_S,
        COORDINATOR_SIGN_RETRY_DELAYS_S,
    )


def coordinator_sign_message(purpose: str, body_hash_hex: str) -> bytes:
    """Rebuild the exact bytes a delegated coordinator signature covers."""

    domain = COORDINATOR_SIGN_DOMAINS.get(str(purpose))
    if domain is None:
        raise ValueError(f"unknown coordinator signing purpose: {purpose!r}")
    if not isinstance(body_hash_hex, str) or not _HEX_32_RE.fullmatch(body_hash_hex):
        raise ValueError("coordinator signing body hash must be lowercase 32-byte hex")
    return domain + body_hash_hex.encode("ascii")


def split_coordinator_sign_message(message: bytes) -> tuple[str, str]:
    """Classify signable bytes back into ``(purpose, body_hash_hex)``.

    The delegate keypair uses this so it can keep the ordinary
    ``keypair.sign(message)`` interface that the snapshot and receipt
    signers already call, while sending only a purpose and a hash.
    """

    if not isinstance(message, (bytes, bytearray)):
        raise ValueError("coordinator signing message must be bytes")
    raw = bytes(message)
    for purpose, domain in COORDINATOR_SIGN_DOMAINS.items():
        if not raw.startswith(domain):
            continue
        body = raw[len(domain) :]
        try:
            body_hash_hex = body.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("coordinator signing body hash is not ascii") from exc
        if not _HEX_32_RE.fullmatch(body_hash_hex):
            raise ValueError(
                "coordinator signing body hash must be lowercase 32-byte hex"
            )
        return purpose, body_hash_hex
    raise ValueError(
        "message is not delegatable: the pool manager signs only mesh "
        "verification snapshots and mesh receipts"
    )


class ManagerDelegateKeypair:
    """A keypair-shaped view of the coordinator hotkey held by the manager.

    Slots in wherever the driver used ``load_hotkey_keypair(...)``: exposes
    ``ss58_address`` and ``sign(message) -> bytes``.  Verification is not
    delegated -- anything that verifies a signature does so locally against
    the public ss58, which needs no secret.
    """

    #: Sr25519, matching the Bittensor hotkeys this stands in for. Callers
    #: assert on this before signing worker-control bodies.
    crypto_type = 1

    def __init__(
        self,
        ss58_address: str,
        sign_remote: Callable[[str, str], str],
        *,
        mesh_key: str = "",
    ) -> None:
        if not isinstance(ss58_address, str) or not ss58_address:
            raise ValueError("delegate keypair needs the coordinator ss58 address")
        self._ss58_address = ss58_address
        self._sign_remote = sign_remote
        self.mesh_key = str(mesh_key or "")

    @property
    def ss58_address(self) -> str:
        return self._ss58_address

    @property
    def public_key(self) -> bytes:
        """The coordinator public key, decoded from its ss58 address."""

        from verallm.mesh.receipt_signing import _keypair_from_ss58

        return bytes(_keypair_from_ss58(self._ss58_address).public_key)

    def sign(self, message: bytes) -> bytes:
        purpose, body_hash_hex = split_coordinator_sign_message(message)
        signature_hex = str(self._sign_remote(purpose, body_hash_hex) or "").lower()
        signature_hex = signature_hex.removeprefix("0x")
        if not _HEX_SIGNATURE_RE.fullmatch(signature_hex):
            raise RuntimeError(
                "pool manager returned a malformed coordinator signature"
            )
        signature = bytes.fromhex(signature_hex)
        # Verify locally before handing it on. A manager that signed with the
        # wrong hotkey, or corrupted the message, must fail here rather than
        # at the validator, where it would score as a forged snapshot.
        if not self.verify(message, signature):
            raise RuntimeError(
                "pool manager signature does not verify against coordinator "
                f"hotkey {self._ss58_address}"
            )
        return signature

    def verify(self, message: bytes, signature: bytes) -> bool:
        from verallm.mesh.receipt_signing import _keypair_from_ss58

        try:
            return bool(
                _keypair_from_ss58(self._ss58_address).verify(
                    bytes(message), bytes(signature)
                )
            )
        except Exception:
            return False


def identity_challenge_message(nonce: bytes, evm_address: str) -> bytes:
    """The EIP-191 payload a coordinator signs to prove EVM control."""

    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != 32:
        raise ValueError("identity challenge nonce must be 32 bytes")
    address = str(evm_address or "").strip().lower()
    if not address.startswith("0x") or len(address) != 42:
        raise ValueError("identity challenge needs a 0x EVM address")
    return bytes(nonce) + bytes.fromhex(address[2:])


class ManagerDelegateEvmSigner:
    """Signs validator identity challenges through the pool manager."""

    def __init__(
        self,
        evm_address: str,
        sign_remote: Callable[[str], str],
    ) -> None:
        address = str(evm_address or "").strip()
        if not address.lower().startswith("0x") or len(address) != 42:
            raise ValueError("delegate EVM signer needs a 0x EVM address")
        self.address = address
        self._sign_remote = sign_remote

    def sign_nonce(self, nonce: bytes) -> str:
        """Return the hex signature over ``nonce || address``."""

        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != 32:
            raise ValueError("identity challenge nonce must be 32 bytes")
        signature = str(self._sign_remote(bytes(nonce).hex()) or "")
        if not signature:
            raise RuntimeError("pool manager returned no identity signature")
        return signature


def delegate_evm_signer_from_worker_request(
    *,
    evm_address: str,
    mesh_key: str,
    request: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> ManagerDelegateEvmSigner:
    """Build the identity-challenge signer for a wallet-less driver."""

    def _sign_remote(nonce_hex: str) -> str:
        response = request(
            "coordinator-sign",
            {
                "purpose": IDENTITY_CHALLENGE_PURPOSE,
                "nonce": nonce_hex,
                "mesh_key": mesh_key,
            },
        )
        signer_address = str(response.get("evm_address", "") or "")
        if signer_address and signer_address.lower() != evm_address.lower():
            raise RuntimeError(
                "pool manager signed as a different coordinator EVM address"
            )
        return str(response.get("signature", "") or "")

    return ManagerDelegateEvmSigner(evm_address, _sign_remote)


class CoordinatorDelegation:
    """Everything a wallet-less serve process needs to sign as coordinator."""

    def __init__(
        self,
        *,
        coordinator_hotkey: str,
        keypair: "ManagerDelegateKeypair",
        evm_address: str,
        challenge_signer: Callable[[bytes], str] | None,
    ) -> None:
        self.coordinator_hotkey = coordinator_hotkey
        self.keypair = keypair
        self.evm_address = evm_address
        self.challenge_signer = challenge_signer


def coordinator_delegation_from_file(path: str) -> CoordinatorDelegation:
    """Rebuild the manager signing channel from a driver's delegation file.

    The pool worker writes this file (owner-only) next to the mesh state
    before spawning the coordinator serve process; argv would leak the pool
    secret to every local process. Requests are signed with the worker's
    persistent stage key, exactly like the parent worker's own manager
    calls, so the manager applies the same driver-only authority checks.
    """

    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    manager = str(data["manager_endpoint"]).rstrip("/")
    pool_secret = str(data["pool_secret"])
    worker_id = str(data["worker_id"])
    worker_session_id = str(data["worker_session_id"])
    mesh_key = str(data.get("mesh_key", "") or "")
    coordinator_hotkey = str(data["coordinator_hotkey"])
    evm_address = str(data.get("evm_address", "") or "")
    stage_key_file = str(data.get("stage_proof_key_file", "") or "")

    stage_keypair = None
    stage_proof_key = ""
    if stage_key_file:
        from verallm.mesh.receipt_signing import load_stage_proof_keypair_file

        stage_keypair = load_stage_proof_keypair_file(stage_key_file)
        stage_proof_key = str(stage_keypair.ss58_address)

    def request(action: str, fields: dict[str, Any]) -> dict[str, Any]:
        # Lazy imports: pool imports this module at load time.
        from verallm.mesh.pool import _signed_worker_control_body
        from verallm.mesh.worker import post_json

        def _attempt(timeout_s: float) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "pool_secret": pool_secret,
                "worker_id": worker_id,
                "worker_session_id": worker_session_id,
                **dict(fields),
            }
            if stage_keypair is not None:
                payload = _signed_worker_control_body(
                    payload,
                    action=action,
                    keypair=stage_keypair,
                    proof_key=stage_proof_key,
                )
            return post_json(
                f"{manager}/v1/pool/{action}",
                payload,
                timeout=timeout_s,
                keepalive=True,
                keepalive_fallback=False,
            )

        attempts, timeout_s, delays = coordinator_sign_retry_profile(
            fields.get("purpose")
        )
        return coordinator_sign_request_with_retry(
            _attempt,
            attempts=attempts,
            attempt_timeout_s=timeout_s,
            retry_delays_s=delays,
        )

    keypair = delegate_keypair_from_worker_request(
        coordinator_hotkey=coordinator_hotkey,
        mesh_key=mesh_key,
        request=request,
    )
    challenge_signer: Callable[[bytes], str] | None = None
    if evm_address:
        challenge_signer = delegate_evm_signer_from_worker_request(
            evm_address=evm_address,
            mesh_key=mesh_key,
            request=request,
        ).sign_nonce
    return CoordinatorDelegation(
        coordinator_hotkey=coordinator_hotkey,
        keypair=keypair,
        evm_address=evm_address,
        challenge_signer=challenge_signer,
    )


class ManagerDelegateCapacitySigner:
    """Signs capacity-audit artifacts through the pool manager.

    The worker builds the complete unsigned artifact, the manager validates
    its shape and authority, signs the canonical EIP-191 text with the
    coordinator EVM key, and the worker verifies recovery locally before
    ever pushing the artifact to a validator — a manager that signed with
    the wrong key must fail here, not score as a forged artifact.
    """

    def __init__(
        self,
        evm_address: str,
        sign_remote: Callable[[dict[str, Any]], str],
    ) -> None:
        address = str(evm_address or "").strip()
        if not address.lower().startswith("0x") or len(address) != 42:
            raise ValueError("capacity artifact signer needs a 0x EVM address")
        self.address = address
        self._sign_remote = sign_remote

    def sign_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``artifact`` carrying a verified miner_signature."""

        from neurons.capacity_audit import recover_artifact_signer

        unsigned = dict(artifact)
        unsigned.pop("miner_signature", None)
        unsigned.pop("signature", None)
        signature = str(self._sign_remote(unsigned) or "")
        if not signature:
            raise RuntimeError("pool manager returned no capacity signature")
        signed = dict(unsigned)
        signed["miner_signature"] = signature
        recovered = recover_artifact_signer(signed)
        if recovered.lower() != self.address.lower():
            raise RuntimeError(
                "pool manager capacity signature does not recover to the "
                f"coordinator EVM address {self.address}"
            )
        return signed


def delegate_capacity_signer_from_worker_request(
    *,
    evm_address: str,
    mesh_key: str,
    request: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> ManagerDelegateCapacitySigner:
    """Build the capacity-artifact signer for a wallet-less mesh worker."""

    def _sign_remote(artifact: dict[str, Any]) -> str:
        response = request(
            "coordinator-sign",
            {
                "purpose": CAPACITY_AUDIT_ARTIFACT_PURPOSE,
                "artifact": artifact,
                "mesh_key": mesh_key,
            },
        )
        signer_address = str(response.get("evm_address", "") or "")
        if signer_address and signer_address.lower() != evm_address.lower():
            raise RuntimeError(
                "pool manager signed as a different coordinator EVM address"
            )
        return str(response.get("signature", "") or "")

    return ManagerDelegateCapacitySigner(evm_address, _sign_remote)


def delegate_keypair_from_worker_request(
    *,
    coordinator_hotkey: str,
    mesh_key: str,
    request: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> ManagerDelegateKeypair:
    """Build the delegate a driver uses when it holds no coordinator wallet.

    ``request`` performs one authenticated worker->manager call and returns
    the decoded response body.
    """

    def _sign_remote(purpose: str, body_hash_hex: str) -> str:
        response = request(
            "coordinator-sign",
            {
                "purpose": purpose,
                "body_hash": body_hash_hex,
                "mesh_key": mesh_key,
            },
        )
        signer = str(response.get("coordinator_hotkey", "") or "")
        if signer and signer != coordinator_hotkey:
            raise RuntimeError(
                "pool manager signed as a different coordinator hotkey"
            )
        return str(response.get("signature", "") or "")

    return ManagerDelegateKeypair(
        coordinator_hotkey, _sign_remote, mesh_key=mesh_key
    )
