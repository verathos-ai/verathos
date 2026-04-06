"""Python client for the PaymentGateway Solidity contract."""

from __future__ import annotations

import logging
from typing import List, Optional

from verallm.chain.cache import TTLCache
from verallm.chain.config import ChainConfig
from verallm.chain.payment_types import DepositEvent
from verallm.chain.provider import Web3Provider

logger = logging.getLogger(__name__)


class PaymentGatewayClient:
    """Read/write interface to the on-chain PaymentGateway.

    Read calls are free (view functions) and cached with a configurable TTL.
    Write calls (deposit) require a funded EVM private key.
    """

    def __init__(self, config: ChainConfig, provider: Optional[Web3Provider] = None):
        self._config = config
        if not config.payment_gateway_address:
            raise ValueError(
                "payment_gateway_address not set. "
                "Deploy PaymentGateway first and provide "
                "--chain-config or set VERATHOS_PAYMENT_GATEWAY."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.payment_gateway_address, "PaymentGateway"
        )
        self._cache = TTLCache(default_ttl=60)

    # ── Free reads ───────────────────────────────────────────────

    def get_total_deposited(self, user_address: str) -> int:
        """Get total TAO deposited by a user (in wei). Cached 60s."""
        from web3 import Web3
        addr = Web3.to_checksum_address(user_address)

        cache_key = f"deposited:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.getDeposited(addr).call()
        )
        self._cache.set(cache_key, result, ttl=60)
        return result

    def get_owner_cut_bps(self) -> int:
        """Get the current owner cut in basis points (e.g. 1000 = 10%)."""
        cache_key = "owner_cut_bps"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.ownerCutBps().call()
        )
        self._cache.set(cache_key, result, ttl=300)
        return result

    def get_netuid(self) -> int:
        """Get the subnet UID this gateway targets."""
        cache_key = "netuid"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.netuid().call()
        )
        self._cache.set(cache_key, result, ttl=3600)
        return result

    # ── Paid writes ──────────────────────────────────────────────

    def deposit(
        self,
        validator_hotkey: bytes,
        amount_wei: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Deposit TAO for inference credits.

        Args:
            validator_hotkey: 32-byte hotkey of the validator to stake to.
            amount_wei: Amount of TAO in wei.
            private_key: Hex EVM private key (uses config default if not provided).

        Returns:
            Transaction hash hex string.
        """
        assert len(validator_hotkey) == 32, "Validator hotkey must be 32 bytes"
        assert amount_wei > 0, "Deposit amount must be > 0"

        # For payable functions we need to build the tx with value
        pk = private_key or self._config.evm_private_key
        if not pk:
            raise ValueError("No private key configured for signing transactions")

        tx_hash = self._provider.send_transaction(
            self._contract.functions.deposit(validator_hotkey),
            private_key=pk,
            value=amount_wei,
        )

        logger.info(
            "Deposited %d wei (%.6f TAO) to validator 0x%s: %s",
            amount_wei, amount_wei / 10**18,
            validator_hotkey.hex()[:16], tx_hash,
        )
        return tx_hash

    # ── Event queries ────────────────────────────────────────────

    def get_deposit_events(
        self,
        from_block: int = 0,
        to_block: str | int = "latest",
        user: Optional[str] = None,
    ) -> List[DepositEvent]:
        """Query Deposit events from the contract.

        Args:
            from_block: Start block (inclusive).
            to_block: End block (inclusive) or "latest".
            user: Optional filter by depositor address.

        Returns:
            List of DepositEvent sorted by block number ascending.
        """
        from web3 import Web3

        # Build filter arguments (web3 7.x uses snake_case)
        kwargs: dict = {
            "from_block": from_block,
            "to_block": to_block,
        }
        if user:
            kwargs["argument_filters"] = {
                "user": Web3.to_checksum_address(user)
            }

        logs = self._provider.call_with_retry(
            lambda: self._contract.events.Deposit.get_logs(**kwargs)
        )

        events = []
        for log in logs:
            events.append(DepositEvent(
                user=log.args["user"],
                validator_hotkey=bytes(log.args["validatorHotkey"]),
                tao_amount=log.args["taoAmount"],
                block_number=log.blockNumber,
                tx_hash=log.transactionHash.hex(),
            ))

        events.sort(key=lambda e: e.block_number)
        return events
