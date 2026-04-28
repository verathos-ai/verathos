#!/usr/bin/env python3
"""MinerNeuron — Bittensor miner wrapper for Verathos.

Lifecycle:
1. Resolve model config (auto or explicit, with cascading fallback).
2. Auto-associate EVM key if needed (one-time, via Substrate extrinsic).
3. Start the existing verallm.api.server as a subprocess.
4. Wait for /health to return OK.
5. Register model on MinerRegistry (skip if already registered with same params).
6. Enter heartbeat loop: renewModel() every 12 hours.

Usage:
    # Testnet — auto-select best model for GPU:
    python -m neurons.miner \
        --wallet miner --hotkey default --netuid 42 \
        --chain-config chain_config_testnet.json \
        --auto --endpoint https://miner.example.com:8000

    # Anvil — same thing but with --private-key instead of --wallet:
    python -m neurons.miner \
        --private-key 0x59c6...690d --netuid 42 \
        --chain-config chain_config_anvil.json \
        --auto --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import bittensor as bt
import httpx

from neurons.config import NeuronConfig
from neurons.model_resolve import add_model_args, resolve_model_config
from neurons.version import spec_version, version_str, miner_version, miner_version_str
from verallm.chain.config import ChainConfig
from verallm.chain.miner_registry import MinerRegistryClient
from verallm.chain.provider import Web3Provider
from verallm.chain.wallet import derive_evm_private_key, derive_evm_address

logger = logging.getLogger(__name__)


def _check_external_port(endpoint: str) -> None:
    """Verify the miner's endpoint port is reachable from the internet.

    Starts a temporary TCP listener on the port, asks external services
    to probe it, then shuts it down — all before vLLM loads. If a service
    confirms the port is closed, abort with a clear error. If all services
    are unreachable, log a warning and continue (skip).
    """
    from urllib.parse import urlparse
    import socket as _socket
    import threading as _threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if not host:
        bt.logging.warning("Cannot parse endpoint host — skipping external port check")
        return

    bt.logging.info(f"Checking if port {port} on {host} is reachable from the internet...")

    # Start a temporary HTTP server so external services have something to connect to.
    # Runs in a background thread, shut down after the check.
    class _SilentHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *args):
            pass  # silence

    tmp_server = None
    tmp_thread = None
    try:
        tmp_server = HTTPServer(("0.0.0.0", port), _SilentHandler)
        tmp_thread = _threading.Thread(target=tmp_server.serve_forever, daemon=True)
        tmp_thread.start()
        time.sleep(0.5)  # let it bind
    except OSError as e:
        # Port already in use (maybe server already running) — skip temp server,
        # the real server will handle it
        bt.logging.debug(f"Could not start temp listener on port {port}: {e} — port may already be in use")
        tmp_server = None

    # Try multiple external port-check services for reliability.
    # Each returns (responded: bool, port_open: bool).
    checks: list[tuple[str, bool, bool]] = []

    # Service 1: yougetsignal.com (returns HTML with "is open" or "is closed")
    try:
        resp = httpx.post(
            "https://ports.yougetsignal.com/check-port.php",
            data={"remoteAddress": host, "portNumber": str(port)},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            body = resp.text.lower()
            is_open = "is open" in body
            checks.append(("yougetsignal", True, is_open))
        else:
            checks.append(("yougetsignal", False, False))
    except Exception:
        checks.append(("yougetsignal", False, False))

    # Service 2: portchecker.io
    try:
        resp = httpx.get(
            f"https://portchecker.io/api/v1/query?host={host}&ports={port}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            # Response: {"host": ..., "ports": [{"port": N, "status": "open"|"closed"}]}
            ports_list = body.get("ports", [])
            is_open = any(p.get("status") == "open" for p in ports_list)
            checks.append(("portchecker.io", True, is_open))
        else:
            checks.append(("portchecker.io", False, False))
    except Exception:
        checks.append(("portchecker.io", False, False))

    # Shut down temp server before returning
    def _cleanup():
        if tmp_server is not None:
            tmp_server.shutdown()

    responded = [(name, is_open) for name, ok, is_open in checks if ok]
    if not responded:
        _cleanup()
        bt.logging.warning(
            "External port check: no check service reachable. "
            "Skipping — port may or may not be open."
        )
        return

    if any(is_open for _, is_open in responded):
        _cleanup()
        bt.logging.success(f"External port check passed: {host}:{port} is reachable")
        return

    # All responding services say port is closed
    _cleanup()
    services_str = ", ".join(name for name, _ in responded)
    bt.logging.error(
        f"\n{'=' * 60}\n"
        f"  EXTERNAL PORT CHECK FAILED\n"
        f"  Port {port} on {host} is NOT reachable from the internet.\n"
        f"  (Checked via: {services_str})\n\n"
        f"  Your miner server is running but nobody can connect to it.\n"
        f"  Common causes:\n"
        f"    - Firewall blocking the port (check: sudo ufw allow {port}/tcp)\n"
        f"    - Cloud provider security group missing inbound rule\n"
        f"    - NAT/router not forwarding port {port} to this machine\n\n"
        f"  Registration aborted — fix the port and restart.\n"
        f"{'=' * 60}"
    )
    sys.exit(1)


def _extract_hotkey_seed(wallet_name: str, hotkey_name: str, wallet) -> bytes:
    """Extract the 32-byte Ed25519 seed from a Bittensor hotkey.

    Works across bittensor v9 (wallet.hotkey.private_key) and v10+ (keyfile JSON).
    """
    # v9: direct attribute access
    if hasattr(wallet.hotkey, "private_key"):
        return bytes(wallet.hotkey.private_key[:32])

    # v10+: read from keyfile JSON
    import json
    from pathlib import Path
    hk_path = Path.home() / f".bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}"
    hk_data = json.loads(hk_path.read_text())
    return bytes.fromhex(hk_data["secretSeed"].replace("0x", ""))


class MinerNeuron:
    """Wraps the existing VeraLLM miner server with Bittensor chain registration."""

    def __init__(self, config: NeuronConfig):
        self.config = config
        self._provider: Optional[Web3Provider] = None
        self._miner_client: Optional[MinerRegistryClient] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._running = True

        self.evm_pk = ""
        self.evm_addr = ""
        self.uid: Optional[int] = None  # resolved from metagraph during setup()

    def setup(self, private_key: Optional[str] = None):
        """Initialize chain connection and derive EVM credentials.

        Args:
            private_key: Raw EVM private key (hex). If provided, skips
                bittensor wallet derivation (useful for Anvil testing).
                If None, derives from the bittensor wallet.
        """
        if private_key:
            # Direct private key mode (Anvil / testing)
            from eth_account import Account
            pk = private_key if private_key.startswith("0x") else f"0x{private_key}"
            self.evm_pk = pk
            self.evm_addr = Account.from_key(pk).address
            bt.logging.info(f"EVM address (from --private-key): {self.evm_addr}")
        else:
            # Wallet mode (real testnet / mainnet).  bittensor is already
            # imported at module level (line 39); the previous local
            # `import bittensor as bt` shadowed it, and the corresponding
            # try/except was unreachable AND broken — `bt.logging.error`
            # in the except branch ran before `bt` was bound, raising
            # NameError instead of the intended graceful exit.
            WalletCls = getattr(bt, "Wallet", None) or bt.wallet
            wallet = WalletCls(name=self.config.wallet_name, hotkey=self.config.hotkey_name)
            hotkey_seed = _extract_hotkey_seed(
                self.config.wallet_name, self.config.hotkey_name, wallet,
            )
            self.hotkey_seed = hotkey_seed

            self.evm_pk = derive_evm_private_key(hotkey_seed)
            self.evm_addr = derive_evm_address(hotkey_seed)
            bt.logging.info(f"EVM address (from wallet): {self.evm_addr}")

        self._provider = Web3Provider(self.config)
        self._miner_client = MinerRegistryClient(self.config, provider=self._provider)

        if not private_key:
            # Resolve UID from Substrate metagraph — needed for registerEvm().
            self.uid = self._resolve_uid_with_retry()
            bt.logging.info(f"Resolved UID: {self.uid}")
        else:
            bt.logging.info("Anvil mode — skipping EVM association check")

    def _resolve_uid_with_retry(self, max_attempts: int = 10) -> Optional[int]:
        """Resolve our UID from the Substrate metagraph with retry.

        Retries with exponential backoff on transient errors (429, timeouts).
        Returns None only in Anvil mode (no wallet).
        """
        import random

        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        wallet = WalletCls(
            name=self.config.wallet_name, hotkey=self.config.hotkey_name,
        )
        hk_ss58 = wallet.hotkey.ss58_address

        for attempt in range(1, max_attempts + 1):
            try:
                sub = bt.Subtensor(network=self.config.subtensor_network)
                mg = sub.metagraph(self.config.netuid)
                for uid_val in range(len(mg.hotkeys)):
                    if mg.hotkeys[uid_val] == hk_ss58:
                        bt.logging.info(f"UID {uid_val} resolved from metagraph (SS58={hk_ss58})")
                        return uid_val
                # Hotkey not found in metagraph — not registered on subnet
                bt.logging.error(f"Hotkey {hk_ss58} not found in metagraph for netuid {self.config.netuid}. Is this miner registered on the subnet?")
                return None
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.error(f"Failed to resolve UID after {max_attempts} attempts: {e}. Cannot proceed without UID.")
                    raise RuntimeError(
                        f"Cannot resolve UID from metagraph after {max_attempts} "
                        f"attempts: {e}"
                    ) from e
                delay = min(2 ** attempt * 3, 60) + random.uniform(0, 5)
                bt.logging.warning(f"Metagraph query failed (attempt {attempt}/{max_attempts}): {e} — retrying in {delay:.0f}s")
                time.sleep(delay)

    def _ensure_evm_registered(self):
        """Ensure registerEvm(uid) has been called on the current MinerRegistry contract.

        Uses self.uid (resolved at startup from Substrate metagraph).
        The contract's _resolveUid() falls back to the evmToUid mapping set
        by registerEvm(), so this must succeed before registerModel() will work.

        Retries with backoff — never silently gives up.
        """
        import random

        if self.uid is None:
            # Anvil mode or UID not resolvable — skip
            return

        for attempt in range(1, 11):
            try:
                if self._miner_client.is_evm_registered(self.evm_addr):
                    bt.logging.info(f"EVM already registered on MinerRegistry (UID={self.uid})")
                    return
            except Exception as e:
                if attempt == 10:
                    raise RuntimeError(
                        f"Cannot check EVM registration after 10 attempts: {e}"
                    ) from e
                delay = min(2 ** attempt * 3, 60) + random.uniform(0, 5)
                bt.logging.warning(f"is_evm_registered check failed (attempt {attempt}/10): {e} — retrying in {delay:.0f}s")
                time.sleep(delay)
                continue

            # Not registered — call registerEvm(uid)
            try:
                bt.logging.info(f"Registering EVM → UID {self.uid} on MinerRegistry")
                self._miner_client.register_evm(
                    self.uid,
                    hotkey_seed=self.hotkey_seed,
                    netuid=self.config.netuid,
                    private_key=self.evm_pk,
                )
                bt.logging.info(f"registerEvm({self.uid}) succeeded")
                return
            except Exception as e:
                if attempt == 10:
                    raise RuntimeError(
                        f"registerEvm({self.uid}) failed after 10 attempts: {e}"
                    ) from e
                delay = min(2 ** attempt * 3, 60) + random.uniform(0, 5)
                bt.logging.warning(f"registerEvm({self.uid}) failed (attempt {attempt}/10): {e} — retrying in {delay:.0f}s")
                time.sleep(delay)

    def start_server(self, server_args: list[str]):
        """Start the VeraLLM miner server as a subprocess."""
        cmd = [sys.executable, "-m", "verallm.api.server"] + server_args
        # Redact secrets from the logged command line
        _safe = []
        _skip = False
        for a in cmd:
            if _skip:
                _safe.append("***")
                _skip = False
            elif a in ("--evm-private-key", "--private-key"):
                _safe.append(a)
                _skip = True
            else:
                _safe.append(a)
        bt.logging.info(f"Starting miner server: {' '.join(_safe)}")
        self._server_process = subprocess.Popen(cmd)

    def wait_for_health(self, endpoint: str):
        """Poll /health until the server is ready. No timeout — model load + Merkle tree
        can take 30+ minutes for large models."""
        url = f"{endpoint}/health"
        start = time.monotonic()
        last_log = start
        while True:
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    elapsed = time.monotonic() - start
                    bt.logging.info(f"Miner server healthy after {elapsed:.0f}s")
                    return
            except Exception:
                pass
            now = time.monotonic()
            if now - last_log >= 60.0:
                bt.logging.info(f"Waiting for miner server... ({now - start:.0f}s elapsed)")
                last_log = now
            time.sleep(2.0)

    def query_actual_max_context(self, endpoint: str) -> Optional[int]:
        """Query the running server for the actual max single-request context.

        After vLLM loads the model, the real KV cache pool size is known.
        The actual max context a single request can use is::

            min(kv_pool_tokens, max_model_len)

        Returns None if the server doesn't expose batch-mode health fields
        (e.g. running in single-request mode).
        """
        try:
            resp = httpx.get(f"{endpoint}/health", timeout=5.0)
            if resp.status_code != 200:
                return None
            health = resp.json()
            kv_pool = health.get("kv_pool_tokens")
            max_context = health.get("max_context")  # = max_model_len
            if kv_pool is None or max_context is None:
                return None
            return min(kv_pool, max_context)
        except Exception as e:
            bt.logging.warning(f"Could not query server for actual context: {e}")
            return None

    @staticmethod
    def _ctx_close_enough(old: int, new: int, tolerance: float = 0.10) -> bool:
        """Return True if two context lengths are within tolerance (default 10%).

        KV pool size varies between restarts due to VRAM fragmentation,
        CUDA graph memory overhead, etc.  A 10% tolerance absorbs the
        typical variation (e.g. 225k → 215k = 4.7%) without creating
        unnecessary new on-chain entries that cause index drift.
        """
        if old == 0:
            return False
        return abs(old - new) / old <= tolerance

    def check_existing_registration(
        self,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
    ) -> Optional[int]:
        """Check existing registrations and clean up stale entries.

        Returns the matching model index if already registered with the same
        params (caller should skip re-registration), or None if a new
        registration is needed.

        Only deactivates entries that match on model_id + endpoint + quant but
        have a context length that changed beyond tolerance (restart noise with
        a significantly different KV pool).  Entries with a different quant or
        endpoint are left alone — a miner may legitimately serve the same model
        in multiple quants (int4 + fp16) or on different endpoints.

        Context length comparison uses a 2% tolerance — small KV pool
        variations between restarts are noise (VRAM fragmentation, CUDA graph
        memory, etc.) and don't warrant a new on-chain entry.
        """
        try:
            models = self._miner_client.get_miner_models(self.evm_addr)
        except Exception as e:
            # If we can't query existing entries, raise instead of silently
            # returning None — otherwise the caller registers a duplicate.
            raise RuntimeError(
                f"Cannot check existing registrations (chain RPC error): {e}"
            ) from e

        matching_index: Optional[int] = None

        for i, m in enumerate(models):
            if not m.active:
                continue
            # Different model but same endpoint — stale entry from a model change.
            # Deactivate it so the validator doesn't route to an endpoint serving
            # a different model than what's registered.
            if m.model_id != model_id and m.endpoint == endpoint:
                bt.logging.info(f"Deactivating stale entry at index {i}: model changed from {m.model_id} to {model_id} at same endpoint {endpoint}")
                self._deactivate_with_retry(i)
                continue
            # Different model, different endpoint — leave it alone
            if m.model_id != model_id:
                continue
            # Same model + quant — check if this is "our" slot
            if m.quant != quant:
                continue  # different quant for same model — leave it alone

            if m.endpoint != endpoint:
                # Different endpoint for same model+quant. If --update-endpoint
                # is set, only update if the host matches (port/scheme change on
                # the same machine). Never touch entries from other IPs — those
                # belong to other GPU instances.
                if getattr(self.config, "update_endpoint", False) and matching_index is None:
                    from urllib.parse import urlparse
                    old_host = urlparse(m.endpoint).hostname or ""
                    new_host = urlparse(endpoint).hostname or ""
                    if old_host == new_host:
                        bt.logging.info(f"Endpoint changed at index {i}: {m.endpoint} → {endpoint} — updating on-chain")
                        try:
                            self._miner_client.update_endpoint(i, endpoint, private_key=self.evm_pk)
                            bt.logging.info(f"Endpoint updated at index {i}")
                            matching_index = i
                        except Exception as e:
                            bt.logging.warning(f"updateEndpoint failed: {e} — will register new entry")
                continue

            # Same model, same endpoint, same quant — this is "our" slot
            if self._ctx_close_enough(m.max_context_len, max_context_len):
                if matching_index is not None:
                    # Duplicate active entry — deactivate the older one
                    bt.logging.info(f"Deactivating duplicate entry at index {matching_index} (keeping newer index {i})")
                    self._deactivate_with_retry(matching_index)
                # Context within tolerance — reuse this entry
                # Check if the lease expired (active flag is True in
                # struct but isModelActive() returns False when
                # expiresAt < now).  Renew immediately if so.
                if m.expires_at <= int(time.time()):
                    bt.logging.info(f"Lease expired at index {i} (expiresAt={m.expires_at}, {(time.time() - m.expires_at) / 3600:.1f}h ago) — deactivating and re-registering")
                    # Contract rejects renewModel() on fully expired
                    # leases ("Already expired, re-register").
                    # Deactivate the stale entry so register_on_chain()
                    # creates a fresh one.  Wait for tx confirmation
                    # before returning — the contract checks the raw
                    # active flag, not isModelActive().
                    self._deactivate_with_retry(i)
                    # Wait for deactivation to be mined
                    for _ in range(10):
                        time.sleep(3)
                        try:
                            updated = self._miner_client.get_miner_models(
                                self.evm_addr
                            )
                            if i < len(updated) and not updated[i].active:
                                bt.logging.info(f"Deactivation confirmed at index {i}")
                                break
                        except Exception:
                            pass
                    return None  # force new registration
                else:
                    bt.logging.info(f"Already registered with same config at index {i} (model={model_id}, quant={quant}, ctx={m.max_context_len} vs {max_context_len} — within tolerance) — skipping")
                matching_index = i
            else:
                # Same slot but context changed significantly — stale entry
                bt.logging.info(f"Deactivating stale entry at index {i} (same model/endpoint/quant, ctx changed: {m.max_context_len} → {max_context_len})")
                self._deactivate_with_retry(i)

        return matching_index

    def _renew_with_retry(self, index: int, max_attempts: int = 3) -> None:
        """Renew an expired model lease with retry on transient errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                tx = self._miner_client.renew_model(index, private_key=self.evm_pk)
                bt.logging.info(f"Renewed expired lease at index {index}: {tx}")
                return
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.warning(f"Failed to renew entry {index} after {max_attempts} attempts: {e}")
                else:
                    time.sleep(2 ** attempt * 5)

    def _deactivate_with_retry(self, index: int, max_attempts: int = 3) -> None:
        """Deactivate a model entry with retry on transient errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                self._miner_client.deactivate_model(index, private_key=self.evm_pk)
                return
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.warning(f"Failed to deactivate entry {index} after {max_attempts} attempts: {e}")
                else:
                    time.sleep(2 ** attempt * 5)

    def _parse_model_index_from_receipt(self, tx_hash: str) -> Optional[int]:
        """Parse ModelRegistered/ModelReactivated event from tx receipt.

        Returns the on-chain array index of the registered/reactivated entry.
        The contract emits exactly one of:
          - ModelReactivated(miner, index, modelId) — reused existing slot
          - ModelRegistered(miner, modelId, endpoint) — pushed new entry

        ModelReactivated contains the index directly.  ModelRegistered means
        a new push, so index = array_length - 1.
        """
        try:
            w3 = self._miner_client._provider.w3
            contract = self._miner_client._contract
            receipt = w3.eth.get_transaction_receipt(tx_hash)

            for log in receipt.get("logs", []):
                # Try ModelReactivated first (has explicit index)
                try:
                    event = contract.events.ModelReactivated().process_log(log)
                    index = event["args"]["index"]
                    bt.logging.info(f"Contract reactivated existing slot at index {index}")
                    return index
                except Exception:
                    pass
                # Try ModelRegistered (new push → count - 1)
                try:
                    contract.events.ModelRegistered().process_log(log)
                    count = self._miner_client.get_miner_model_count(self.evm_addr)
                    bt.logging.info(f"Contract created new entry at index {count - 1}")
                    return count - 1
                except Exception:
                    pass
        except Exception as e:
            bt.logging.warning(f"Failed to parse model index from receipt: {e}")
        return None

    def register_on_chain(
        self,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
        max_retries: int = 5,
    ) -> int:
        """Register this miner's model on MinerRegistry.

        Returns the on-chain model index for use in heartbeat renewals.

        The entire check→register flow is wrapped in a single retry loop so
        that a transient RPC failure during the pre-check never causes a
        blind registration (which would create duplicates).
        """
        from web3 import Web3

        for attempt in range(1, max_retries + 1):
            try:
                # Step 1: Check for existing registration (must succeed)
                existing_index = self.check_existing_registration(
                    model_id, endpoint, quant, max_context_len,
                )
                if existing_index is not None:
                    return existing_index

                # Step 2: Ensure EVM→UID mapping
                self._ensure_evm_registered()

                # Step 3: Register
                spec_ref = Web3.solidity_keccak(["string"], [model_id])
                bt.logging.info(f"Registering on-chain: model={model_id} endpoint={endpoint} quant={quant} ctx={max_context_len}")
                tx = self._miner_client.register_model(
                    model_id=model_id,
                    endpoint=endpoint,
                    model_spec_ref=spec_ref,
                    quant=quant,
                    max_context_len=max_context_len,
                    private_key=self.evm_pk,
                )
                bt.logging.info(f"Registered: {tx}")

                # Step 4: Get the index from the tx receipt events.
                # The contract emits ModelReactivated(miner, index, modelId)
                # when reactivating an existing slot, or ModelRegistered(
                # miner, modelId, endpoint) when pushing a new entry.
                # Parsing the event is always correct — no stale-state or
                # side-effect issues that plagued check_existing_registration.
                event_index = self._parse_model_index_from_receipt(tx)
                if event_index is not None:
                    bt.logging.info(f"Model index from tx receipt: {event_index}")
                    return event_index
                # Fallback: if event parsing fails (shouldn't happen),
                # use count - 1 as last resort.
                bt.logging.warning("Could not parse model index from tx receipt — using count-1 fallback")
                count = self._miner_client.get_miner_model_count(self.evm_addr)
                return count - 1
            except Exception as e:
                err_str = str(e)
                # "Duplicate active entry" means the model was already registered
                # (e.g. previous TX succeeded but receipt confirmation was 429'd).
                # Don't retry — loop back to check_existing_registration which
                # will find it and return the index.
                if "Duplicate active entry" in err_str:
                    bt.logging.info("Model already registered on-chain (duplicate detected), verifying...")
                    continue
                if attempt == max_retries:
                    raise
                delay = min(2 ** attempt * 5, 60)
                bt.logging.warning(f"Registration attempt {attempt}/{max_retries} failed ({e}), retrying in {delay}s")
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Validator allowlist refresh (metagraph → JSON file for middleware)
    # ------------------------------------------------------------------

    def _refresh_validator_allowlist(self) -> None:
        """Query the metagraph for validators and write their SS58 hotkeys to disk.

        The JSON file is read by ValidatorAuthMiddleware in the server subprocess.
        Validators must have a permit AND meet the on-chain minValidatorStake
        threshold (read from ValidatorRegistry every refresh cycle).
        """
        from verallm.api.validator_auth import DEFAULT_VALIDATORS_PATH
        try:
            sub = self._get_subtensor()
            metagraph = sub.metagraph(netuid=self.config.netuid)

            # Read minValidatorStake from ValidatorRegistry (0 if unavailable)
            min_validator_stake = 0
            if getattr(self.config, 'validator_registry_address', None):
                try:
                    from verallm.chain.validator_registry import ValidatorRegistryClient
                    vr = ValidatorRegistryClient(self.config)
                    min_validator_stake_raw = vr.get_min_validator_stake()
                    # Convert RAO to TAO for comparison with metagraph.S
                    min_validator_stake = min_validator_stake_raw / 1e9
                    bt.logging.debug(f"minValidatorStake from contract: {min_validator_stake:.2f} alpha")
                except Exception as e:
                    bt.logging.debug(f"Could not fetch minValidatorStake: {e}")

            validators = []
            n = metagraph.n.item()
            for uid in range(n):
                has_permit = (
                    hasattr(metagraph, 'validator_permit')
                    and bool(metagraph.validator_permit[uid])
                )
                stake = float(metagraph.S[uid]) if hasattr(metagraph, 'S') else float(metagraph.stake[uid])

                if has_permit and stake >= min_validator_stake:
                    ss58 = metagraph.hotkeys[uid]
                    validators.append({
                        "uid": uid,
                        "hotkey_ss58": ss58,
                        "stake": stake,
                    })

            # Inject manually allowed validators (--allow-validators)
            allow_extra = getattr(self.config, "allow_validators", None) or []
            existing_ss58 = {v["hotkey_ss58"] for v in validators}
            for ss58 in allow_extra:
                if ss58 not in existing_ss58:
                    validators.append({"uid": -1, "hotkey_ss58": ss58, "stake": 0})
                    bt.logging.info(f"Manually allowed validator: {ss58}")

            out_path = os.environ.get("VERATHOS_VALIDATORS_PATH", DEFAULT_VALIDATORS_PATH)
            data = {
                "updated_at": int(time.time()),
                "netuid": self.config.netuid,
                "validators": validators,
            }
            # Atomic write: write to temp file then rename
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, out_path)

            bt.logging.info(f"Validator allowlist updated: {len(validators)} validators written to {out_path}")

        except Exception as e:
            bt.logging.warning(f"Failed to refresh validator allowlist: {e}")
            raise

    _cached_metagraph_line: str = ""

    _subtensor_cache = None

    def _get_subtensor(self):
        if self._subtensor_cache is None:
            self._subtensor_cache = bt.Subtensor(network=self.config.subtensor_network)
        return self._subtensor_cache

    def _refresh_metagraph_stats(self) -> None:
        """Fetch metagraph from RPC and cache the stats line. Called every ~5 min."""
        try:
            sub = self._get_subtensor()
            block = sub.get_current_block()
            mg = sub.metagraph(netuid=self.config.netuid)
            uid = self.uid
            n = mg.n.item()
            if uid is not None and uid < n:
                _get = lambda attr: float(getattr(mg, attr)[uid]) if hasattr(mg, attr) else 0.0
                self._cached_metagraph_parts = (
                    f"UID {uid}",
                    f"incentive={_get('incentive'):.4f}",
                    f"emission={_get('emission'):.2f}α/tempo",
                    f"trust={_get('trust'):.2f}",
                    f"stake={_get('stake'):.2f}α",
                )
                self._cached_metagraph_line = f"Metagraph | block={block} | {' | '.join(self._cached_metagraph_parts)}"
        except Exception as e:
            bt.logging.debug(f"Metagraph refresh failed: {e}")

    def _validator_refresh_loop(self, interval: float = 300.0) -> None:
        """Background thread: refresh validator allowlist every `interval` seconds."""
        backoff = interval
        while self._running:
            success = False
            for attempt in range(1, 4):  # 3 retries
                try:
                    self._refresh_validator_allowlist()
                    backoff = interval  # Reset on success
                    success = True
                    break
                except Exception:
                    if attempt < 3:
                        wait = 10 * attempt  # 10s, 20s
                        bt.logging.debug(f"Allowlist refresh retry {attempt}/3 in {wait}s")
                        time.sleep(wait)
            if not success:
                backoff = min(backoff + 60, 600)  # grow by 1 min, max 10 min
                bt.logging.warning(f"Validator allowlist refresh failed after 3 retries, next attempt in {backoff:.0f}s")

            # Refresh metagraph stats from RPC (independent of allowlist success)
            self._refresh_metagraph_stats()
            if self._cached_metagraph_line:
                bt.logging.info(self._cached_metagraph_line)

            # Sleep in small increments; re-log cached stats every ~60s with fresh block
            deadline = time.monotonic() + backoff
            _last_stats_log = time.monotonic()
            while self._running and time.monotonic() < deadline:
                time.sleep(5.0)
                if hasattr(self, '_cached_metagraph_parts') and time.monotonic() - _last_stats_log >= 60:
                    # Refresh block number cheaply (single RPC, no metagraph sync)
                    try:
                        _block = self._get_subtensor().get_current_block()
                        self._cached_metagraph_line = f"Metagraph | block={_block} | {' | '.join(self._cached_metagraph_parts)}"
                    except Exception:
                        pass  # use stale line
                    if self._cached_metagraph_line:
                        bt.logging.info(self._cached_metagraph_line)
                    _last_stats_log = time.monotonic()

    def start_validator_refresh(self, interval: float = 300.0) -> None:
        """Start the background validator allowlist refresh thread."""
        t = threading.Thread(
            target=self._validator_refresh_loop,
            args=(interval,),
            daemon=True,
            name="validator-refresh",
        )
        t.start()
        bt.logging.info(f"Validator allowlist refresh started (interval={interval}s)")

    def _get_lease_remaining_sec(self, model_index: int) -> Optional[float]:
        """Read remaining lease time from chain (one read RPC, no gas)."""
        try:
            models = self._miner_client.get_miner_models(self.evm_addr)
            if model_index < len(models):
                remaining = models[model_index].expires_at - int(time.time())
                return max(remaining, 0.0)
        except Exception as e:
            bt.logging.warning(f"Could not read lease expiry: {e}")
        return None

    def heartbeat_loop(self, model_index: int = 0):
        """Periodically renew the model lease.

        On startup, checks remaining lease time via a read-only RPC call
        (no gas).  Only renews immediately if the lease won't survive
        until the next scheduled heartbeat.  On failure, retries with
        exponential backoff (30s → 60s → 120s … capped at 10 min) so
        transient RPC rate limits don't cause the 24h lease to expire.
        """
        interval = self.config.heartbeat_interval_sec

        # Check remaining lease — only renew now if it won't last until
        # the next heartbeat, otherwise just wait.
        remaining = self._get_lease_remaining_sec(model_index)
        if remaining is not None and remaining > interval + 3600:
            # Lease has plenty of headroom — sleep until renewal is due.
            # Schedule first renewal at remaining - interval (so the lease
            # gets extended well before it expires).
            sleep_time = remaining - interval
            bt.logging.info(f"Starting heartbeat loop (interval={interval}s, lease has {remaining / 3600:.1f}h remaining — first renewal in {sleep_time / 3600:.1f}h)")
        else:
            # Lease is short or unknown — renew immediately.
            sleep_time = 0
            _lease_info = f"has {remaining / 3600:.1f}h remaining" if remaining is not None else "unknown"
            bt.logging.info(f"Starting heartbeat loop (interval={interval}s, lease {_lease_info} — renewing immediately)")

        last_success = time.monotonic()
        consecutive_failures = 0

        while self._running:
            if sleep_time > 0:
                time.sleep(sleep_time)
            if not self._running:
                break
            try:
                tx = self._miner_client.renew_model(model_index, private_key=self.evm_pk)
                bt.logging.info(f"Renewed model at index {model_index}: {tx}")
                last_success = time.monotonic()
                consecutive_failures = 0
                sleep_time = interval
            except Exception as e:
                err_str = str(e)
                # "Already expired, re-register" means the entry we're
                # renewing has expired — either because we were renewing
                # the wrong index (register_on_chain bug) or because RPC
                # failures exceeded the 24h lease.  Re-register to get a
                # fresh entry and update model_index.
                if "Already expired" in err_str or "re-register" in err_str.lower():
                    bt.logging.warning(f"Lease expired at index {model_index} — re-registering")
                    try:
                        model_index = self.register_on_chain(
                            model_id=self._model_id,
                            endpoint=self._endpoint,
                            quant=self._quant,
                            max_context_len=self._max_context_len,
                        )
                        bt.logging.info(f"Re-registered at index {model_index} — heartbeat continues")
                        last_success = time.monotonic()
                        consecutive_failures = 0
                        sleep_time = interval
                        continue
                    except Exception as re_err:
                        bt.logging.error(f"Re-registration failed: {re_err}")
                        # Fall through to normal backoff

                consecutive_failures += 1
                hours_since_success = (time.monotonic() - last_success) / 3600
                # Exponential backoff: 30s, 60s, 120s, ... capped at 10 min
                sleep_time = min(30 * (2 ** (consecutive_failures - 1)), 600)
                bt.logging.warning(f"Renew failed (attempt {consecutive_failures}, {hours_since_success:.1f}h since last success, retry in {sleep_time}s): {e}")
                if hours_since_success > 20:
                    bt.logging.critical(f"MODEL LEASE EXPIRES IN <4 HOURS! Renewal has failed for {hours_since_success:.1f}h. Check EVM wallet balance and RPC connectivity.")

    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        if self._server_process:
            self._server_process.terminate()
            self._server_process.wait(timeout=10)
            bt.logging.info("Server process terminated")


def parse_args():
    parser = argparse.ArgumentParser(description="Verathos Miner Neuron")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON. If omitted, derived from --subtensor-network.")
    parser.add_argument("--subtensor-network", default=None,
                        help="Bittensor network name (test or finney). Selects chain config (contracts) and default RPC URL.")
    parser.add_argument("--subtensor-chain-endpoint", default=None,
                        help="Explicit Substrate+EVM RPC endpoint (e.g. http://localhost:9944 for local subtensor).")
    parser.add_argument("--endpoint", required=True, help="Public miner endpoint URL")
    parser.add_argument("--no-autorestart", action="store_true",
                        help="Don't auto-restart the server subprocess on crash")
    parser.add_argument("--auto-update", action="store_true",
                        help="Enable automatic code updates from git remote. "
                             "Checks every 30 min, pulls and restarts on new commits.")
    parser.add_argument("--auto-update-interval", type=int, default=1800,
                        help="Auto-update check interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--analytics", action="store_true",
                        help="Enable analytics reporting (request timing, proof stats)")

    # TEE (Trusted Execution Environment)
    tee_group = parser.add_argument_group("tee")
    parser.add_argument("--allow-validators", nargs="+", default=None,
                        metavar="SS58",
                        help="Extra validator SS58 hotkeys to allow (bypass permit check)")
    parser.add_argument("--update-endpoint", action="store_true",
                        help="Update existing on-chain endpoint if it changed (e.g. IP/port change). "
                             "Without this flag, a new entry is created for each unique endpoint.")

    tee_group.add_argument("--tee-enabled", action="store_true",
                           help="Enable TEE mode — register attestation on-chain after model registration")
    tee_group.add_argument("--tee-platform", default="mock",
                           choices=["mock", "tdx", "sev-snp", "gpu"],
                           help="TEE attestation platform (default: mock for dev)")

    # Authentication: wallet (testnet) or private-key (Anvil)
    auth = parser.add_argument_group("authentication")
    auth.add_argument("--wallet", default=None,
                      help="Bittensor wallet name (testnet — derives EVM key)")
    auth.add_argument("--hotkey", default="default",
                      help="Bittensor hotkey name (used with --wallet)")
    auth.add_argument("--private-key", default=None,
                      help="EVM private key (Anvil — skips bittensor wallet)")

    # Model selection (auto or explicit, with cascading fallback)
    add_model_args(parser)

    # Bittensor logging flags (--logging.debug, --logging.trace, --logging.info)
    bt.logging.add_args(parser)

    # Everything after -- is passed to the miner server
    args, server_args = parser.parse_known_args()
    return args, server_args


def _extract_code_measurement(platform: str, attestation_report: bytes, Web3) -> bytes:
    """Extract and hash the code measurement from an attestation report.

    Returns keccak256(raw_measurement) for on-chain storage as bytes32.
    - TDX: keccak256(mr_td) — 48-byte measurement at report body offset 0x0B8
    - SEV-SNP: keccak256(measurement) — 48-byte field at offset 0x090
    - mock/gpu: keccak256(b"mock") — deterministic placeholder
    """
    if platform == "tdx":
        try:
            # TDX DCAP quote: header(48) + body — mr_td is at body offset 0x0B8 (184), 48 bytes
            body_offset = 48  # after quote header
            mr_td_offset = body_offset + 0x0B8
            mr_td = attestation_report[mr_td_offset : mr_td_offset + 48]
            if len(mr_td) == 48:
                return bytes(Web3.keccak(mr_td))
        except Exception:
            pass
    elif platform == "sev-snp":
        try:
            # SEV-SNP report: measurement at offset 0x090, 48 bytes
            measurement = attestation_report[0x090 : 0x090 + 48]
            if len(measurement) == 48:
                return bytes(Web3.keccak(measurement))
        except Exception:
            pass
    # mock, gpu, or extraction failed — use deterministic placeholder
    return bytes(Web3.keccak(b"mock"))


def _register_tee(neuron, args, model_id: str):
    """Register TEE attestation on-chain after model is live.

    1. Fetch enclave pubkey + attestation from the running server's /tee/info
    2. Compute weight_file_hash (flat SHA256 of safetensors files)
    3. Call registerTEEAttestation() on MinerRegistry
    """
    import httpx

    bt.logging.info(f"TEE registration: fetching attestation from server (platform={args.tee_platform})")

    try:
        # Get TEE info from our own running server
        port = _extract_port(args.endpoint)
        resp = httpx.get(f"http://localhost:{port}/tee/info", timeout=10.0)
        resp.raise_for_status()
        tee_info = resp.json()
    except Exception as e:
        bt.logging.error(f"TEE registration failed — cannot reach /tee/info: {e}")
        return

    enclave_pubkey = bytes.fromhex(tee_info["enclave_public_key"])
    attestation = tee_info.get("attestation", {})
    attestation_report_hex = attestation.get("attestation_report", "")
    if not attestation_report_hex:
        bt.logging.error("TEE registration failed — /tee/info returned no attestation_report")
        return
    attestation_report = bytes.fromhex(attestation_report_hex)

    # keccak256 for on-chain storage
    from web3 import Web3
    attestation_hash = Web3.keccak(attestation_report)

    # Compute weight file hash (cheap single-pass SHA256 over safetensors files)
    try:
        from verallm.tee.weight_hash import compute_weight_file_hash
        weight_file_hash = compute_weight_file_hash(model_id)
    except Exception as e:
        bt.logging.warning(f"TEE: could not compute weight_file_hash ({e}), using zero hash")
        weight_file_hash = b"\x00" * 32

    # Extract code measurement from attestation report
    # TDX: keccak256(mr_td), SEV-SNP: keccak256(launch_digest), mock: keccak256("mock")
    code_measurement = _extract_code_measurement(args.tee_platform, attestation_report, Web3)

    bt.logging.info(
        f"TEE registration: platform={args.tee_platform} "
        f"pubkey={enclave_pubkey.hex()[:16]}... "
        f"weight_hash={weight_file_hash.hex()[:16]}... "
        f"code_measurement={code_measurement.hex()[:16]}..."
    )

    try:
        tx = neuron._miner_client.register_tee_attestation(
            platform=args.tee_platform,
            enclave_pub_key=enclave_pubkey,
            attestation_hash=attestation_hash,
            model_weight_hash=weight_file_hash,
            code_measurement=code_measurement,
            private_key=neuron.evm_pk,
        )
        bt.logging.success(f"TEE attestation registered on-chain: {tx}")
    except Exception as e:
        bt.logging.error(f"TEE registration tx failed: {e}")


def _extract_port(endpoint_url: str) -> int:
    """Extract port from endpoint URL, defaulting to 8000."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint_url)
    return parsed.port or 8000


def _extract_server_port(server_args: list[str]) -> int:
    """Extract --port from server subprocess args, defaulting to 8000."""
    for i, arg in enumerate(server_args):
        if arg == "--port" and i + 1 < len(server_args):
            try:
                return int(server_args[i + 1])
            except ValueError:
                break
    return 8000  # matches verallm.api.server default


def _clear_stale_compile_caches() -> None:
    """Clear torch.compile / Triton caches to prevent stale kernels.

    Stale caches from previous code versions can cause CUDA illegal memory
    access crashes or silent proof performance regressions.  Clearing on
    every startup is safe — recompilation adds ~30s to first startup only.
    """
    import shutil
    import pathlib
    home = pathlib.Path.home()
    cache_dirs = [
        home / ".cache" / "vllm" / "torch_compile_cache",
        home / ".triton" / "cache",
    ]
    # /tmp/torchinductor_<user> — match any user
    for p in pathlib.Path("/tmp").glob("torchinductor_*"):
        cache_dirs.append(p)
    cleared = 0
    for d in cache_dirs:
        if d.is_dir():
            try:
                shutil.rmtree(d)
                cleared += 1
            except Exception:
                pass
    if cleared:
        bt.logging.info(f"Cleared {cleared} stale torch/triton compile caches")


def main():
    from neurons.log import setup_neuron_logging, print_banner

    args, server_args = parse_args()
    setup_neuron_logging(args)
    _clear_stale_compile_caches()

    if not args.wallet and not args.private_key:
        bt.logging.error("Either --wallet or --private-key is required")
        sys.exit(1)

    # Resolve chain config FIRST so model selection can filter by on-chain models
    resolved_chain_path = ChainConfig.resolve_config_path(
        args.chain_config, args.subtensor_network,
    )
    if resolved_chain_path is None:
        bt.logging.error("Provide --chain-config or --subtensor-network (test/finney)")
        sys.exit(1)
    args.chain_config = resolved_chain_path  # update for downstream use

    # Resolve model configuration (auto or explicit)
    resolved = resolve_model_config(
        model_id=args.model_id,
        quant=args.quant,
        max_context_len=args.max_context_len,
        auto=args.auto,
        category=args.category,
        chain_config=resolved_chain_path,
        subtensor_network=args.subtensor_network,
    )
    bt.logging.info(f"Model config: {resolved.model_id} quant={resolved.quant} ctx={resolved.max_context_len}")

    config = NeuronConfig.from_env(
        wallet_name=args.wallet or "default",
        hotkey_name=args.hotkey,
        netuid=args.netuid,
    )

    # Resolve EVM RPC URL: explicit endpoint > network default > JSON default
    rpc_override = ChainConfig.resolve_rpc_url(
        getattr(args, "subtensor_chain_endpoint", None),
        args.subtensor_network,
    )
    chain_config = ChainConfig.from_json(
        resolved_chain_path,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )
    for k in ChainConfig.__dataclass_fields__:
        if getattr(chain_config, k) != ChainConfig.__dataclass_fields__[k].default:
            setattr(config, k, getattr(chain_config, k))

    # Set Substrate network from CLI args
    if getattr(args, "subtensor_chain_endpoint", None):
        ep = args.subtensor_chain_endpoint
        ws_ep = ep.replace("http://", "ws://").replace("https://", "wss://")
        config.subtensor_network = ws_ep
    elif args.subtensor_network:
        config.subtensor_network = args.subtensor_network

    # Transfer --allow-validators to config for the refresh loop
    if getattr(args, "allow_validators", None):
        config.allow_validators = args.allow_validators

    # Transfer --update-endpoint to config for registration logic
    if getattr(args, "update_endpoint", False):
        config.update_endpoint = True

    # ── Early on-chain model check ───────────────────────────────
    # Verify the resolved model is registered on-chain BEFORE loading
    # it into GPU. This avoids wasting ~60s on model load + root
    # computation only to fail at _chain_self_check in the server.
    if chain_config.model_registry_address:
        try:
            from verallm.chain.model_registry import ModelRegistryClient
            model_client = ModelRegistryClient(chain_config)
            on_chain = model_client.get_model_list()
            on_chain_lower = {m.lower() for m in (on_chain or [])}
            if on_chain is not None and resolved.model_id.lower() not in on_chain_lower:
                bt.logging.error(
                    f"Model '{resolved.model_id}' is not registered on-chain. "
                    f"Miners can only serve models registered on the ModelRegistry contract. "
                    f"ModelRegistry: {chain_config.model_registry_address} | "
                    f"Registered models: {sorted(on_chain) if on_chain else '(none)'}"
                )
                sys.exit(1)
            elif on_chain is not None:
                bt.logging.info(f"On-chain model check passed: '{resolved.model_id}' is registered")
        except Exception as e:
            bt.logging.warning(
                f"On-chain model check skipped (RPC error: {e}). "
                f"The server will re-check after model load."
            )

    neuron = MinerNeuron(config)

    def signal_handler(sig, _frame):
        bt.logging.info(f"Received signal {sig}, shutting down")
        neuron.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    neuron.setup(private_key=args.private_key)

    # ── Startup banner ──
    network = args.subtensor_network or ("testnet" if chain_config.chain_id == 945 else "mainnet")
    print_banner(
        "Miner",
        network=network,
        netuid=config.netuid,
        wallet=args.wallet or "(private-key)",
        hotkey=args.hotkey if args.wallet else "",
        evm=neuron.evm_addr or "",
        spec_ver=f"{version_str} ({spec_version})",
        miner_ver=f"{miner_version_str} ({miner_version})",
        model=resolved.model_id,
        quantization=resolved.quant,
        max_context=resolved.max_context_len,
        endpoint=args.endpoint,
        auto_update="enabled" if args.auto_update else "disabled",
    )

    # Build server args: always ensure --model and --quant are present
    if "--model" not in server_args and "--model-id" not in server_args:
        server_args = [
            "--model", resolved.model_id,
            "--quant", resolved.quant,
        ] + server_args

    # Pass EVM identity to server for anti-hijacking (receipt validation + identity challenge)
    if neuron.evm_addr:
        server_args.extend(["--evm-address", neuron.evm_addr])
    if neuron.evm_pk:
        server_args.extend(["--evm-private-key", neuron.evm_pk])

    # Forward log level to server subprocess
    if getattr(args, "logging.trace", False):
        server_args.extend(["--log-level", "debug"])  # server has no trace, use debug
    elif getattr(args, "logging.debug", False):
        server_args.extend(["--log-level", "debug"])

    # Forward chain config and resolved RPC URL so server can self-check roots
    if args.chain_config and "--chain-config" not in server_args:
        server_args.extend(["--chain-config", args.chain_config])
    if chain_config.rpc_url and "--evm-rpc-url" not in server_args:
        server_args.extend(["--evm-rpc-url", chain_config.rpc_url])

    # Forward TEE args to server subprocess
    if args.tee_enabled and "--tee-enabled" not in server_args:
        server_args.append("--tee-enabled")
        server_args.extend(["--tee-platform", args.tee_platform])

    # Write validator allowlist before starting server to avoid open-access window.
    # Only in wallet mode — Anvil mode has no metagraph.
    if args.wallet:
        try:
            neuron._refresh_validator_allowlist()
        except Exception as e:
            bt.logging.warning(f"Initial validator allowlist write failed: {e} — server will block until next refresh succeeds")

    # ── External port reachability check ──
    # Verify the endpoint port is reachable from outside BEFORE loading vLLM
    # (which takes 10+ min). Starts a temporary TCP listener on the port,
    # asks external services to probe it, then shuts it down.
    _check_external_port(args.endpoint)

    neuron.start_server(server_args)

    # Health-check on localhost. The --endpoint may be behind a reverse proxy
    # that isn't reachable from inside the container. Parse the server's actual
    # port from server_args (mirrors the server's own --port default of 8000).
    local_health_url = f"http://localhost:{_extract_server_port(server_args)}"
    neuron.wait_for_health(local_health_url)

    # Start background refresh loop (periodic updates)
    if args.wallet:
        neuron.start_validator_refresh(interval=300.0)

    # Use actual KV pool from the running server instead of the registry estimate.
    # After vLLM loads: real capacity = min(kv_pool_tokens, max_model_len).
    actual_context = neuron.query_actual_max_context(local_health_url)
    if actual_context is not None:
        if actual_context != resolved.max_context_len:
            bt.logging.info(f"On-chain max_context: {actual_context} (actual from vLLM, was {resolved.max_context_len} from registry)")
        else:
            bt.logging.info(f"On-chain max_context: {actual_context} (matches registry)")
        reg_context = actual_context
    else:
        bt.logging.warning(f"Could not query actual context from server — using registry value {resolved.max_context_len}")
        reg_context = resolved.max_context_len

    # Store registration params so heartbeat_loop can re-register on lease expiry
    neuron._model_id = resolved.model_id
    neuron._endpoint = args.endpoint
    neuron._quant = resolved.quant
    neuron._max_context_len = reg_context

    model_index = neuron.register_on_chain(
        model_id=resolved.model_id,
        endpoint=args.endpoint,
        quant=resolved.quant,
        max_context_len=reg_context,
    )

    # ── TEE registration / revocation ──
    if args.tee_enabled:
        _register_tee(neuron, args, resolved.model_id)
    else:
        # If TEE is not enabled but on-chain registration exists, revoke it.
        # This keeps on-chain state in sync with the miner's actual mode —
        # stale TEE registrations cause the proxy to route TEE traffic and
        # skip ZK proof verification for this miner.
        try:
            cap = neuron._miner_client.get_tee_capability(neuron.evm_addr)
            if cap.enabled:
                bt.logging.info("TEE not enabled but on-chain registration found — revoking...")
                neuron._miner_client.revoke_tee_attestation(private_key=neuron.evm_pk)
                bt.logging.success("TEE attestation revoked on-chain")
        except Exception as e:
            bt.logging.warning(f"Could not check/revoke TEE registration: {e}")

    # ── Auto-updater ──
    if args.auto_update:
        from neurons.auto_update import AutoUpdater

        updater = AutoUpdater(
            role="miner",
            check_interval=args.auto_update_interval,
        )
        updater.start()

    bt.logging.success(
        f"Miner ready — serving {resolved.model_id} ({resolved.quant}) "
        f"on {args.endpoint}",
    )

    neuron.heartbeat_loop(model_index=model_index)


if __name__ == "__main__":
    main()
