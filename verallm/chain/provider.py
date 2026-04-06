"""Web3 provider with retry, nonce management, and transaction helpers."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from verallm.chain.config import ChainConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ABI directory relative to this file
_ABI_DIR = Path(__file__).resolve().parent.parent.parent / "contracts" / "abi"

# Gas price ceiling (100 Gwei) — warn but don't block
_GAS_PRICE_WARN_THRESHOLD = 100 * 10**9


def _load_abi(name: str) -> list:
    """Load a contract ABI JSON from the abi/ directory."""
    import json

    abi_path = _ABI_DIR / f"{name}.json"
    if not abi_path.exists():
        raise FileNotFoundError(
            f"ABI file not found: {abi_path}\n"
            f"Run 'cd contracts && forge build' to generate ABIs, then copy "
            f"the JSON from contracts/out/<Name>.sol/<Name>.json → contracts/abi/<Name>.json"
        )
    try:
        with open(abi_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed ABI JSON in {abi_path}: {e}") from e


def _normalize_private_key(pk: str) -> str:
    """Validate and normalize a hex private key to 0x-prefixed 64-char form.

    Raises ValueError on invalid format.
    """
    cleaned = pk.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if len(cleaned) != 64 or not all(c in "0123456789abcdef" for c in cleaned):
        raise ValueError(
            f"Invalid EVM private key format. "
            f"Expected 64 hex characters (with optional 0x prefix), "
            f"got {len(cleaned)} chars."
        )
    return "0x" + cleaned


class Web3Provider:
    """Wraps web3.Web3 with retry logic, nonce tracking, and transaction helpers.

    Auto-detects WSS vs HTTP from the RPC URL.
    """

    def __init__(self, config: ChainConfig):
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        self.config = config

        if config.rpc_url.startswith("wss://") or config.rpc_url.startswith("ws://"):
            self.w3 = Web3(Web3.WebsocketProvider(config.rpc_url))
        else:
            # Limit connection pool to prevent CLOSE-WAIT socket accumulation
            # during RPC rate limiting (429s leave dangling connections).
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry as _Retry
            provider = Web3.HTTPProvider(
                config.rpc_url,
                request_kwargs={"timeout": 15},
            )
            adapter = HTTPAdapter(
                pool_connections=2,
                pool_maxsize=5,
                max_retries=_Retry(total=0),  # we handle retries ourselves
            )
            provider._session = __import__("requests").Session()
            provider._session.mount("https://", adapter)
            provider._session.mount("http://", adapter)
            self.w3 = Web3(provider)

        # Bittensor EVM is a PoA chain — inject middleware for extraData
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Nonce tracking to avoid race conditions on rapid transactions
        self._nonce_lock = threading.Lock()
        self._pending_nonce: Dict[str, int] = {}

        # Public RPCs have aggressive rate limits (~1 req/s).
        # Add inter-call delay and patient retries.
        self._is_public_rpc = any(
            h in config.rpc_url
            for h in ("opentensor.ai", "chain.opentensor", "finney")
        )
        self._last_rpc_time = 0.0
        self._rpc_min_interval = 1.1 if self._is_public_rpc else 0.0  # seconds
        if self._is_public_rpc and config.max_retries < 5:
            config.max_retries = 5
            config.retry_delay = 3.0

    def get_contract(self, address: str, abi_name: str):
        """Load a contract instance by address and ABI name."""
        from web3 import Web3
        abi = _load_abi(abi_name)
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi,
        )

    def call_with_retry(self, fn: Callable[[], T]) -> T:
        """Call a view function with retry and exponential backoff.

        Automatically detects HTTP 429 (rate limit) and uses longer backoff
        with jitter to avoid thundering-herd retries.

        Rate-limit retries use 10/20/40/60s backoff (up to 10 attempts)
        to handle the OTF testnet 1 req/s limit gracefully.
        """
        import random

        delay = self.config.retry_delay
        max_attempts = self.config.max_retries
        for attempt in range(max_attempts):
            # Rate limit for public RPCs: wait between calls
            if self._rpc_min_interval > 0:
                elapsed = time.time() - self._last_rpc_time
                if elapsed < self._rpc_min_interval:
                    time.sleep(self._rpc_min_interval - elapsed)
            try:
                self._last_rpc_time = time.time()
                return fn()
            except Exception as e:
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                if is_rate_limit:
                    # On first 429, switch to aggressive retry: 10 attempts
                    # with exponential backoff starting at 10s.
                    if attempt == 0:
                        max_attempts = max(max_attempts, 10)
                        delay = 10.0
                    wait = min(delay, 60) + random.uniform(1, 5)
                    if attempt == max_attempts - 1:
                        raise
                    # First few retries are debug (common on public RPCs),
                    # only warn after 3+ consecutive failures.
                    _log = logger.debug if attempt < 3 else logger.warning
                    _log(
                        "Rate limited (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, max_attempts, wait, str(e)[:200],
                    )
                else:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay
                    logger.warning(
                        "Chain call failed (attempt %d/%d): %s",
                        attempt + 1, max_attempts, e,
                    )
                time.sleep(wait)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def send_transaction(
        self,
        contract_fn,
        private_key: Optional[str] = None,
        value: int = 0,
    ) -> str:
        """Build, sign, and send a contract write transaction. Returns tx hash hex.

        Args:
            contract_fn: Prepared contract function call.
            private_key: Hex EVM private key (uses config default if not provided).
            value: TAO value in wei to send with the transaction (for payable functions).
        """
        from eth_account import Account

        pk = private_key or self.config.evm_private_key
        if not pk:
            raise ValueError("No private key configured for signing transactions")

        pk = _normalize_private_key(pk)
        account = Account.from_key(pk)
        address = account.address

        # Check balance before sending (with retry for 429)
        balance = self.call_with_retry(lambda: self.w3.eth.get_balance(address))
        if balance == 0:
            raise ValueError(
                f"Account {address} has zero balance. "
                f"Fund it first by transferring TAO to its SS58 mirror address. "
                f"See: python -c \"from verallm.chain.wallet import h160_to_ss58_mirror; "
                f"print(h160_to_ss58_mirror('{address}'))\""
            )
        if balance < 10**16:  # < 0.01 TAO
            logger.warning(
                "Account %s has low balance: %.6f TAO — transaction may fail.",
                address, balance / 10**18,
            )

        # Gas price with sanity check (with retry for 429)
        gas_price = self.call_with_retry(lambda: self.w3.eth.gas_price)
        if gas_price > _GAS_PRICE_WARN_THRESHOLD:
            logger.warning(
                "Gas price %.1f Gwei is unusually high (threshold: %.1f Gwei).",
                gas_price / 10**9, _GAS_PRICE_WARN_THRESHOLD / 10**9,
            )

        # Get nonce with lock to prevent race conditions (retry for 429)
        with self._nonce_lock:
            on_chain_nonce = self.call_with_retry(
                lambda: self.w3.eth.get_transaction_count(address)
            )
            pending = self._pending_nonce.get(address, -1)
            nonce = max(on_chain_nonce, pending + 1)
            self._pending_nonce[address] = nonce

        # Build transaction
        tx_params: Dict[str, Any] = {
            "from": address,
            "nonce": nonce,
            "chainId": self.config.chain_id,
            "gasPrice": gas_price,
        }
        if value > 0:
            tx_params["value"] = value
        tx = contract_fn.build_transaction(tx_params)

        # Estimate gas with margin (retry for 429)
        estimated = self.call_with_retry(lambda: self.w3.eth.estimate_gas(tx))
        tx["gas"] = int(estimated * 1.3)

        # Sign and send (retry for 429)
        signed = self.w3.eth.account.sign_transaction(tx, pk)
        try:
            tx_hash = self.call_with_retry(
                lambda: self.w3.eth.send_raw_transaction(signed.raw_transaction)
            )
        except Exception:
            # Send failed — reset pending nonce so next call re-fetches
            # from chain.  Without this, a dropped tx leaves a stale
            # pending nonce that blocks all subsequent renewals.
            with self._nonce_lock:
                self._pending_nonce.pop(address, None)
            raise

        logger.info("Sent tx %s from %s (nonce=%d)", tx_hash.hex(), address, nonce)

        # Wait for receipt (retry for 429 during polling)
        try:
            receipt = self.call_with_retry(
                lambda: self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            )
        except Exception:
            # Receipt timeout or RPC error — tx may be dropped or stuck in
            # mempool (e.g. behind a nonce gap from a dropped earlier tx).
            # Reset pending nonce so next call re-fetches from chain and
            # picks up a clean nonce.
            with self._nonce_lock:
                self._pending_nonce.pop(address, None)
            raise
        if receipt["status"] != 1:
            # Reset pending nonce so next tx re-fetches from chain
            with self._nonce_lock:
                self._pending_nonce.pop(address, None)
            raise RuntimeError(f"Transaction reverted: {tx_hash.hex()}")

        return tx_hash.hex()
