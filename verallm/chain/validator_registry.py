"""Python client for the ValidatorRegistry Solidity contract.

Validators register their proxy endpoints on-chain so users can discover
available proxies.  The subnet owner sets a ``defaultValidator`` — the
recommended proxy for new users.

Read calls are free view functions.  Write calls (register, update, etc.)
require a funded EVM private key.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from verallm.chain.cache import TTLCache
from verallm.chain.config import ChainConfig
from verallm.chain.provider import Web3Provider
from verallm.chain.types import OnChainValidatorInfo

logger = logging.getLogger(__name__)


class ValidatorRegistryClient:
    """Read/write interface to the on-chain ValidatorRegistry."""

    def __init__(self, config: ChainConfig, provider: Optional[Web3Provider] = None):
        self._config = config
        if not config.validator_registry_address:
            raise ValueError(
                "validator_registry_address not set. "
                "Deploy ValidatorRegistry and add to chain config."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.validator_registry_address, "ValidatorRegistry"
        )
        self._cache = TTLCache(default_ttl=config.miner_list_cache_ttl)

    # ── Free reads ───────────────────────────────────────────────

    def get_validator(self, address: str) -> OnChainValidatorInfo:
        """Get info for a specific validator (cached 60s)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)

        cache_key = f"validator:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self._provider.call_with_retry(
            lambda: self._contract.functions.getValidator(addr).call()
        )
        result = _parse_validator_info(raw)
        self._cache.set(cache_key, result, ttl=60)
        return result

    def get_active_validators(self) -> List[Tuple[str, OnChainValidatorInfo]]:
        """Get all active validators (including those without a proxy endpoint).

        Use this for participation counts. Returns list of (address, info) tuples.
        """
        cache_key = "active_validators"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw_addrs, raw_infos = self._provider.call_with_retry(
            lambda: self._contract.functions.getActiveValidators().call()
        )
        result = [
            (addr, _parse_validator_info(info))
            for addr, info in zip(raw_addrs, raw_infos)
        ]
        self._cache.set(cache_key, result)
        return result

    def get_proxy_validators(self) -> List[Tuple[str, OnChainValidatorInfo]]:
        """Get active validators that have a non-empty proxy endpoint (cached 5min).

        Use this for proxy discovery. Excludes validators registered without a proxy.
        Returns list of (address, info) tuples.
        """
        cache_key = "proxy_validators"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw_addrs, raw_infos = self._provider.call_with_retry(
            lambda: self._contract.functions.getProxyValidators().call()
        )
        result = [
            (addr, _parse_validator_info(info))
            for addr, info in zip(raw_addrs, raw_infos)
        ]
        self._cache.set(cache_key, result)
        return result

    def get_default(self) -> Tuple[str, OnChainValidatorInfo]:
        """Get the default validator (set by SN owner).

        Returns (address, info). Raises if no default is set.
        """
        cache_key = "default_validator"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        addr, raw_info = self._provider.call_with_retry(
            lambda: self._contract.functions.getDefault().call()
        )
        result = (addr, _parse_validator_info(raw_info))
        self._cache.set(cache_key, result)
        return result

    def is_evm_registered(self, address: str) -> bool:
        """Check if an EVM address has called registerEvm (UID mapping exists)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.evmRegistered(addr).call()
        )

    def is_active_validator(self, address: str) -> bool:
        """Check if an address is a registered, active validator."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.isActiveValidator(addr).call()
        )

    def get_validator_count(self) -> int:
        """Get total number of registered validators (including inactive)."""
        return self._provider.call_with_retry(
            lambda: self._contract.functions.getValidatorCount().call()
        )

    # ── Paid writes ──────────────────────────────────────────────

    def register_evm(
        self,
        uid: int,
        hotkey_seed: bytes,
        netuid: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Register EVM address → UID mapping with SR25519 hotkey proof."""
        from verallm.chain.wallet import sign_evm_registration
        from eth_account import Account

        contract_addr = self._contract.address
        pk = private_key or self._provider._default_private_key
        evm_address = Account.from_key(pk).address

        sig_r, sig_s = sign_evm_registration(
            hotkey_seed, evm_address, uid, netuid, contract_addr,
        )
        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerEvm(uid, sig_r, sig_s),
            private_key=private_key,
        )
        logger.info("Registered EVM → UID %d on ValidatorRegistry (SR25519 verified): %s", uid, tx_hash)
        return tx_hash

    def register(self, proxy_endpoint: str, private_key: Optional[str] = None) -> str:
        """Register as a validator. Pass empty string if not running a proxy yet."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.register(proxy_endpoint),
            private_key=private_key,
        )
        self._cache.invalidate("active_validators")
        self._cache.invalidate("proxy_validators")
        self._cache.invalidate("default_validator")
        logger.info("Registered validator endpoint %r: %s", proxy_endpoint, tx_hash)
        return tx_hash

    def update_endpoint(
        self, new_endpoint: str, private_key: Optional[str] = None
    ) -> str:
        """Update proxy endpoint URL. Pass empty string to clear it."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.updateEndpoint(new_endpoint),
            private_key=private_key,
        )
        self._cache.invalidate("active_validators")
        self._cache.invalidate("proxy_validators")
        self._cache.invalidate("default_validator")
        logger.info("Updated validator endpoint to %r: %s", new_endpoint, tx_hash)
        return tx_hash

    def deactivate(self, private_key: Optional[str] = None) -> str:
        """Voluntarily deactivate (e.g. going offline for maintenance)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.deactivate(),
            private_key=private_key,
        )
        self._cache.invalidate("active_validators")
        self._cache.invalidate("proxy_validators")
        logger.info("Deactivated validator: %s", tx_hash)
        return tx_hash

    def reactivate(self, private_key: Optional[str] = None) -> str:
        """Reactivate after deactivation."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.reactivate(),
            private_key=private_key,
        )
        self._cache.invalidate("active_validators")
        self._cache.invalidate("proxy_validators")
        logger.info("Reactivated validator: %s", tx_hash)
        return tx_hash

    # ── Stake thresholds (v3) ────────────────────────────────────

    def get_min_proxy_stake(self) -> int:
        """Minimum alpha stake (RAO) to register with a proxy endpoint."""
        return self._provider.call_with_retry(
            lambda: self._contract.functions.minProxyStake().call()
        )

    def get_min_validator_stake(self) -> int:
        """Minimum alpha stake (RAO) to register as a validator."""
        return self._provider.call_with_retry(
            lambda: self._contract.functions.minValidatorStake().call()
        )

    def is_whitelisted(self, address: str) -> bool:
        """Check if an address is on the stake-check whitelist."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.whitelisted(addr).call()
        )

    # ── Owner-only ───────────────────────────────────────────────

    def set_default_validator(
        self, validator_address: str, private_key: Optional[str] = None
    ) -> str:
        """Set the default validator (SN owner only)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(validator_address)
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setDefaultValidator(addr),
            private_key=private_key,
        )
        self._cache.invalidate("default_validator")
        logger.info("Set default validator to %s: %s", validator_address, tx_hash)
        return tx_hash

    def set_min_proxy_stake(self, amount: int, private_key: Optional[str] = None) -> str:
        """Set minimum alpha stake for proxy registration (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setMinProxyStake(amount),
            private_key=private_key,
        )
        logger.info("Set minProxyStake to %d: %s", amount, tx_hash)
        return tx_hash

    def set_min_validator_stake(self, amount: int, private_key: Optional[str] = None) -> str:
        """Set minimum alpha stake for validator registration (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setMinValidatorStake(amount),
            private_key=private_key,
        )
        logger.info("Set minValidatorStake to %d: %s", amount, tx_hash)
        return tx_hash

    def set_whitelisted(self, address: str, status: bool, private_key: Optional[str] = None) -> str:
        """Add/remove address from stake-check whitelist (owner only)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setWhitelisted(addr, status),
            private_key=private_key,
        )
        logger.info("Set whitelisted(%s) = %s: %s", address, status, tx_hash)
        return tx_hash


def _parse_validator_info(raw) -> OnChainValidatorInfo:
    """Parse a tuple from getValidator()/getActiveValidators() into OnChainValidatorInfo."""
    return OnChainValidatorInfo(
        proxy_endpoint=raw[0],
        uid=raw[1],
        registered_at=raw[2],
        updated_at=raw[3],
        active=raw[4],
    )
