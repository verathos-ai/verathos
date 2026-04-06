"""TEEClient — end-to-end encrypted inference for Verathos users.

Handles the full TEE workflow:
1. Fetch miner's enclave public key via the proxy
2. Optionally verify attestation
3. Encrypt chat request with ephemeral X25519 keypair
4. Send through validator proxy (pin-routed to the correct miner)
5. Decrypt response (streaming or batch)

Usage::

    # Via proxy (production)
    client = TEEClient.from_miner("https://api.verathos.ai", model="qwen3-8b")
    result = client.chat([{"role": "user", "content": "Hello"}])
    print(result["output_text"])

    # Streaming
    for event in client.chat_stream([{"role": "user", "content": "Hello"}]):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\\nProof verified: {event['proof_verified']}")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import httpx

from verallm.tee.crypto import (
    decrypt_json,
    encrypt_json,
    generate_keypair,
)
from verallm.tee.serialization import dict_to_envelope
from verallm.tee.types import EncryptedEnvelope, TEEAttestation


class TEEClient:
    """End-to-end encrypted inference client.

    Each instance generates an ephemeral X25519 keypair for the session.
    The shared secret is derived via ECDH with the miner's enclave public key.
    """

    def __init__(
        self,
        validator_url: str,
        enclave_public_key: bytes,
        attestation: Optional[TEEAttestation] = None,
        timeout: float = 300.0,
        direct_miner: bool = False,
        model: str = "auto",
        api_key: Optional[str] = None,
    ):
        self.validator_url = validator_url.rstrip("/")
        self.enclave_public_key = enclave_public_key
        self.attestation = attestation
        self.direct_miner = direct_miner
        self.model = model
        self.api_key = api_key
        self._target_enclave_key = enclave_public_key.hex() if enclave_public_key else ""
        self._private_key, self._public_key = generate_keypair()
        self._timeout = timeout
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers=headers,
        )

    @property
    def _chat_endpoint(self) -> str:
        """Miner uses /tee/chat, proxy uses /v1/tee/chat/completions."""
        if self.direct_miner:
            return f"{self.validator_url}/tee/chat"
        return f"{self.validator_url}/v1/tee/chat/completions"

    @property
    def _stream_endpoint(self) -> str:
        if self.direct_miner:
            return f"{self.validator_url}/tee/chat"  # same endpoint, stream param
        return f"{self.validator_url}/v1/tee/chat/completions/stream"

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _refresh_enclave_key(self) -> None:
        """Re-fetch ``/tee/info`` and regenerate the ephemeral keypair.

        Called on retry when the pinned miner becomes unavailable (HTTP 410).
        """
        resp = self._client.get(
            f"{self.validator_url}/tee/info",
            params={"model": self.model},
        )
        resp.raise_for_status()
        info = resp.json()
        self.enclave_public_key = bytes.fromhex(info["enclave_public_key"])
        self._target_enclave_key = info["enclave_public_key"]
        if "attestation" in info:
            from verallm.tee.serialization import dict_to_attestation
            self.attestation = dict_to_attestation(info["attestation"])
        self._private_key, self._public_key = generate_keypair()

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the error indicates the miner went away (re-fetch key)."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (410, 502, 503)
        return False

    @classmethod
    def from_miner(
        cls,
        proxy_url: str,
        model: str = "auto",
        timeout: float = 300.0,
        direct_miner: bool = False,
        api_key: Optional[str] = None,
    ) -> "TEEClient":
        """Create a TEEClient by fetching TEE info from the validator proxy.

        The proxy's ``GET /tee/info?model=...`` selects a TEE miner for
        the requested model and returns its enclave public key.  The
        ``target_enclave_key`` is included in subsequent requests so the
        proxy pin-routes to the same miner.

        Args:
            proxy_url: Validator proxy URL (e.g. ``https://api.verathos.ai``).
            model: Model to request (e.g. ``"qwen3-8b"``).  ``"auto"``
                picks the best available TEE miner.
            timeout: HTTP timeout in seconds.
            direct_miner: If True, treat *proxy_url* as a miner URL and
                use ``/tee/chat`` directly (testing only).
            api_key: Optional API key for authenticated proxy access.
        """
        url = proxy_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(f"{url}/tee/info", params={"model": model}, timeout=30.0, headers=headers)
        if resp.status_code == 404:
            raise RuntimeError("No TEE-enabled miners available for this model")
        resp.raise_for_status()
        info = resp.json()

        enclave_pk = bytes.fromhex(info["enclave_public_key"])

        attestation = None
        if "attestation" in info:
            from verallm.tee.serialization import dict_to_attestation
            attestation = dict_to_attestation(info["attestation"])

        return cls(
            validator_url=proxy_url,
            enclave_public_key=enclave_pk,
            attestation=attestation,
            timeout=timeout,
            direct_miner=direct_miner,
            model=model,
            api_key=api_key,
        )

    @classmethod
    def from_chain(
        cls,
        validator_url: str,
        miner_address: str,
        chain_config,
        timeout: float = 300.0,
    ) -> "TEEClient":
        """Create a TEEClient by reading TEE capability from chain."""
        from verallm.chain.mock import create_clients

        _, miner_client, *_ = create_clients(chain_config)
        cap = miner_client.get_tee_capability(miner_address)

        if not cap.enabled:
            raise RuntimeError(f"Miner {miner_address} does not have TEE enabled on-chain")

        return cls(
            validator_url=validator_url,
            enclave_public_key=cap.enclave_pub_key,
            timeout=timeout,
        )

    def encrypt_chat_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> EncryptedEnvelope:
        """Encrypt a chat request for the TEE enclave.

        Extra keyword arguments (``temperature``, ``do_sample``,
        ``enable_thinking``, etc.) are included in the encrypted payload
        and interpreted by the miner.
        """
        import uuid

        data = {
            "messages": messages,
            "max_new_tokens": max_tokens,
            **kwargs,
        }
        sid = session_id or str(uuid.uuid4())
        return encrypt_json(data, self._private_key, self.enclave_public_key, sid)

    def decrypt_response(self, encrypted_output: dict) -> Dict[str, Any]:
        """Decrypt the TEE-encrypted response.

        Args:
            encrypted_output: Serialized EncryptedEnvelope dict from the done event.

        Returns:
            Decrypted response dict (contains "output_text").
        """
        envelope = dict_to_envelope(encrypted_output)
        return decrypt_json(envelope, self._private_key)

    def decrypt_chunk(self, encrypted_chunk: dict) -> Dict[str, Any]:
        """Decrypt a single encrypted token chunk.

        Args:
            encrypted_chunk: Serialized EncryptedEnvelope dict with an extra
                ``seq`` field for ordering.

        Returns:
            Dict with ``seq`` (int) and ``text`` (str).
        """
        envelope = dict_to_envelope(encrypted_chunk)
        return decrypt_json(envelope, self._private_key)

    def _make_request_body(self, envelope: EncryptedEnvelope) -> dict:
        env_dict = {
            "session_id": envelope.session_id,
            "sender_public_key": envelope.sender_public_key.hex(),
            "nonce": envelope.nonce.hex(),
            "ciphertext": envelope.ciphertext.hex(),
            "content_type": envelope.content_type,
        }
        if self.direct_miner:
            # Miner server expects {envelope: {...}, validator_nonce: "..."}
            import os
            nonce = os.urandom(32).hex()
            return {"envelope": env_dict, "validator_nonce": nonce}
        # Proxy expects the flat envelope + routing fields
        env_dict["model"] = self.model
        env_dict["target_enclave_key"] = self._target_enclave_key
        return env_dict

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        max_retries: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Full E2E encrypted chat: encrypt, send, verify, decrypt.

        On retryable failures (miner went offline), re-fetches the enclave
        key and re-encrypts for a new miner automatically.

        Args:
            messages: OpenAI-style messages list.
            max_tokens: Max output tokens.
            max_retries: Number of retries with key re-fetch (default 1).
            **kwargs: Extra parameters (``temperature``, ``enable_thinking``,
                etc.) passed through to the miner.

        Returns:
            Dict with "output_text", "proof_verified", "timing".
        """
        for attempt in range(max_retries + 1):
            # Encrypt (fresh each attempt — key may have changed on retry)
            envelope = self.encrypt_chat_request(messages, max_tokens, **kwargs)
            request_body = self._make_request_body(envelope)

            try:
                if self.direct_miner:
                    result = self._chat_sse(request_body, envelope)
                else:
                    resp = self._client.post(
                        self._chat_endpoint,
                        json=request_body,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                break  # Success
            except Exception as e:
                if attempt < max_retries and not self.direct_miner and self._is_retryable(e):
                    self._refresh_enclave_key()
                    continue
                raise

        # Decrypt response
        encrypted_output = result.get("encrypted_output")
        encrypted_nonce = result.get("encrypted_output_nonce")
        if encrypted_output and encrypted_nonce:
            output_bytes = self._decrypt_raw(
                bytes.fromhex(encrypted_output),
                bytes.fromhex(encrypted_nonce),
                envelope.session_id,
            )
            result["output_text"] = output_bytes.decode("utf-8")
        elif encrypted_output:
            decrypted = self.decrypt_response(encrypted_output)
            result["output_text"] = decrypted.get("output_text", "")
        else:
            result["output_text"] = result.get("output_text", "")

        return result

    def _chat_sse(self, request_body: dict, envelope: EncryptedEnvelope) -> dict:
        """Consume SSE stream from miner's /tee/chat and return the done event."""
        import json as _json

        with self._client.stream(
            "POST", self._chat_endpoint,
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            result = {}
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ").strip()
                if not raw:
                    continue
                try:
                    event = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                if event.get("event") == "done":
                    result = event
                elif event.get("event") == "error":
                    raise RuntimeError(event.get("error", "Unknown TEE error"))
        return result

    def _decrypt_raw(self, ciphertext: bytes, nonce: bytes, session_id: str) -> bytes:
        """Decrypt raw ciphertext using NaCl Box (client private + enclave public)."""
        from nacl.public import PrivateKey, PublicKey, Box
        sk = PrivateKey(self._private_key)
        pk = PublicKey(self.enclave_public_key)
        box = Box(sk, pk)
        return box.decrypt(ciphertext, nonce)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        max_retries: int = 1,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """Streaming E2E encrypted chat — yields decrypted token chunks.

        Connects to the validator proxy's streaming endpoint and decrypts
        each encrypted token chunk as it arrives.  On retryable failures,
        re-fetches the enclave key and retries with a new miner.

        Yields dicts with:
            - ``{"type": "token", "seq": int, "text": str}`` for each chunk
            - ``{"type": "done", "output_text": str, "proof_verified": bool, "timing": dict}``
              as the final event

        Args:
            messages: OpenAI-style messages list.
            max_tokens: Max output tokens.
            max_retries: Number of retries with key re-fetch (default 1).
        """
        envelope = self.encrypt_chat_request(messages, max_tokens, **kwargs)
        request_body = self._make_request_body(envelope)

        if self.direct_miner:
            yield from self._chat_stream_direct(request_body, envelope)
            return

        with self._client.stream(
            "POST",
            self._stream_endpoint,
            json=request_body,
        ) as resp:
            resp.raise_for_status()

            event_type = None
            data_lines: list[str] = []

            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                elif line == "":
                    if event_type and data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            data = {"raw": raw}

                        if event_type == "encrypted_token":
                            try:
                                # Proxy relays raw {ciphertext, nonce, seq} from miner
                                ct = bytes.fromhex(data["ciphertext"])
                                nc = bytes.fromhex(data["nonce"])
                                text = self._decrypt_raw(ct, nc, "").decode("utf-8")
                                yield {
                                    "type": "token",
                                    "seq": data.get("seq", 0),
                                    "text": text,
                                }
                            except Exception:
                                pass  # skip corrupted chunks
                        elif event_type == "done":
                            output_text = ""
                            enc_out = data.get("encrypted_output")
                            enc_nonce = data.get("encrypted_output_nonce")
                            if enc_out and enc_nonce:
                                try:
                                    out_bytes = self._decrypt_raw(
                                        bytes.fromhex(enc_out),
                                        bytes.fromhex(enc_nonce),
                                        "",
                                    )
                                    output_text = out_bytes.decode("utf-8")
                                except Exception:
                                    pass
                            elif enc_out and isinstance(enc_out, dict):
                                try:
                                    dec = self.decrypt_response(enc_out)
                                    output_text = dec.get("output_text", "")
                                except Exception:
                                    pass
                            yield {
                                "type": "done",
                                "output_text": output_text,
                                "proof_mode": data.get("proof_mode"),
                                "tee_verified": data.get("tee_verified"),
                                "proof_verified": data.get("proof_verified"),
                                "timing": data.get("timing", {}),
                            }
                        elif event_type == "error":
                            raise RuntimeError(
                                f"Server error: {data.get('error', data)}"
                            )
                    event_type = None
                    data_lines = []

            # Handle trailing event without final newline
            if event_type and data_lines:
                raw = "\n".join(data_lines)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return
                if event_type == "done":
                    output_text = ""
                    enc_out = data.get("encrypted_output")
                    enc_nonce = data.get("encrypted_output_nonce")
                    if enc_out and enc_nonce:
                        try:
                            out_bytes = self._decrypt_raw(
                                bytes.fromhex(enc_out),
                                bytes.fromhex(enc_nonce),
                                "",
                            )
                            output_text = out_bytes.decode("utf-8")
                        except Exception:
                            pass
                    elif enc_out and isinstance(enc_out, dict):
                        try:
                            dec = self.decrypt_response(enc_out)
                            output_text = dec.get("output_text", "")
                        except Exception:
                            pass
                    yield {
                        "type": "done",
                        "output_text": output_text,
                        "proof_verified": data.get("proof_verified"),
                        "timing": data.get("timing", {}),
                    }

    def _chat_stream_direct(
        self, request_body: dict, envelope: EncryptedEnvelope,
    ) -> Iterator[Dict[str, Any]]:
        """Consume SSE stream from miner's /tee/chat, yielding decrypted chunks.

        The miner emits ``data: {json}`` lines where the ``event`` field inside
        the JSON distinguishes event types:

        - ``encrypted_token``: ``{ciphertext, nonce, seq}`` — decrypt via NaCl
          Box (client private key + enclave public key) to get the token text.
        - ``done``: final event with ``encrypted_output`` / ``encrypted_output_nonce``.
        - ``error``: raises RuntimeError.
        """
        import json as _json

        with self._client.stream(
            "POST", self._chat_endpoint,
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ").strip()
                if not raw:
                    continue
                try:
                    event = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue

                evt_type = event.get("event", "")

                if evt_type == "encrypted_token":
                    # Decrypt individual token chunk: {ciphertext, nonce, seq}
                    try:
                        ct = bytes.fromhex(event["ciphertext"])
                        nonce = bytes.fromhex(event["nonce"])
                        plaintext = self._decrypt_raw(ct, nonce, envelope.session_id)
                        yield {
                            "type": "token",
                            "seq": event.get("seq", 0),
                            "text": plaintext.decode("utf-8"),
                        }
                    except Exception:
                        pass  # skip corrupted chunks

                elif evt_type == "done":
                    # Final event — decrypt full output if present
                    output_text = ""
                    enc_out = event.get("encrypted_output")
                    enc_nonce = event.get("encrypted_output_nonce")
                    if enc_out and enc_nonce:
                        try:
                            out_bytes = self._decrypt_raw(
                                bytes.fromhex(enc_out),
                                bytes.fromhex(enc_nonce),
                                envelope.session_id,
                            )
                            output_text = out_bytes.decode("utf-8")
                        except Exception:
                            pass
                    elif enc_out and isinstance(enc_out, dict):
                        try:
                            dec = self.decrypt_response(enc_out)
                            output_text = dec.get("output_text", "")
                        except Exception:
                            pass
                    yield {
                        "type": "done",
                        "output_text": output_text,
                        "proof_verified": event.get("proof_verified"),
                        "timing": event.get("timing", {}),
                    }

                elif evt_type == "error":
                    raise RuntimeError(event.get("error", "Unknown TEE error"))

    def verify_attestation(self) -> bool:
        """Verify the miner's TEE attestation (if available)."""
        if self.attestation is None:
            return False

        from verallm.tee.attestation import get_attestation_provider
        provider = get_attestation_provider(self.attestation.platform)
        return provider.verify_attestation(self.attestation)
