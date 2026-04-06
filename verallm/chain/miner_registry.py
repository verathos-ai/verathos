"""Python client for the MinerRegistry Solidity contract."""

from __future__ import annotations

import logging
from typing import List, Optional

from verallm.chain.cache import TTLCache
from verallm.chain.config import ChainConfig
from verallm.chain.provider import Web3Provider
from verallm.chain.types import OnChainMinerModel, OnChainTEECapability

logger = logging.getLogger(__name__)

# Sentinel for caching None results (to distinguish "not cached" from "cached as None")
_NONE_SENTINEL = object()


class MinerRegistryClient:
    """Read/write interface to the on-chain MinerRegistry.

    Read calls are free (view functions) and cached with a configurable TTL.
    Write calls require a funded EVM private key.
    """

    def __init__(self, config: ChainConfig, provider: Optional[Web3Provider] = None):
        self._config = config
        if not config.miner_registry_address:
            raise ValueError(
                "miner_registry_address not set. "
                "Deploy contracts first (scripts/deploy_testnet.py) and provide "
                "--chain-config or set VERATHOS_MINER_REGISTRY."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.miner_registry_address, "MinerRegistry"
        )
        self._cache = TTLCache(default_ttl=config.miner_list_cache_ttl)

    # ── Free reads ───────────────────────────────────────────────

    def get_miner_models(self, address: str) -> List[OnChainMinerModel]:
        """Get all model entries for a miner address (cached 60s)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)

        cache_key = f"miner_models:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self._provider.call_with_retry(
            lambda: self._contract.functions.getMinerModels(addr).call()
        )
        result = [_parse_miner_model(entry) for entry in raw]
        self._cache.set(cache_key, result, ttl=60)
        return result

    def get_miner_model_count(self, address: str) -> int:
        """Get number of model entries for a miner."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.getMinerModelCount(addr).call()
        )

    def get_providers_for_model(self, model_id: str) -> List[str]:
        """Get all miner addresses serving a specific model."""
        cache_key = f"providers:{model_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.getProvidersForModel(model_id).call()
        )
        self._cache.set(cache_key, result)
        return result

    def is_model_active(self, address: str, index: int) -> bool:
        """Check if a specific miner-model entry is active and not expired."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.isModelActive(addr, index).call()
        )

    def get_associated_uid(self, evm_address: str) -> Optional[int]:
        """Get the UID associated with an EVM address via native associate_evm_key (cached 5min).

        Returns None if no association exists on-chain.
        Uses the contract's getAssociatedUid() which queries the UidLookup precompile (0x806).
        """
        from web3 import Web3
        addr = Web3.to_checksum_address(evm_address)

        cache_key = f"associated_uid:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached if cached != _NONE_SENTINEL else None

        uid, exists = self._provider.call_with_retry(
            lambda: self._contract.functions.getAssociatedUid(addr).call()
        )
        if not exists:
            self._cache.set(cache_key, _NONE_SENTINEL, ttl=300)
            return None
        self._cache.set(cache_key, uid, ttl=300)
        return uid

    # ── Paid writes ──────────────────────────────────────────────

    def register_evm(
        self,
        uid: int,
        hotkey_seed: bytes,
        netuid: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Register EVM address → UID mapping with SR25519 hotkey proof.

        The contract verifies that the caller owns the hotkey for the
        claimed UID via the Sr25519Verify precompile (0x403).

        Args:
            uid: UID to claim on the subnet.
            hotkey_seed: 32-byte SR25519 hotkey seed for signing.
            netuid: Subnet UID.
            private_key: EVM private key for the transaction.
        """
        from verallm.chain.wallet import sign_evm_registration

        contract_addr = self._contract.address
        # Derive the EVM address that will send the tx (= msg.sender)
        from eth_account import Account
        pk = private_key or self._provider._default_private_key
        evm_address = Account.from_key(pk).address

        sig_r, sig_s = sign_evm_registration(
            hotkey_seed, evm_address, uid, netuid, contract_addr,
        )
        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerEvm(uid, sig_r, sig_s),
            private_key=private_key,
        )
        logger.info("Registered EVM → UID %d (SR25519 verified): %s", uid, tx_hash)
        return tx_hash

    def is_evm_registered(self, address: str) -> bool:
        """Check if an EVM address has called registerEvm()."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)
        return self._provider.call_with_retry(
            lambda: self._contract.functions.evmRegistered(addr).call()
        )

    def register_model(
        self,
        model_id: str,
        endpoint: str,
        model_spec_ref: bytes,
        quant: str,
        max_context_len: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Register a new model endpoint (24h lease)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerModel(
                model_id, endpoint, model_spec_ref, quant, max_context_len
            ),
            private_key=private_key,
        )
        self._cache.invalidate(f"providers:{model_id}")
        logger.info("Registered miner model %s at %s: %s", model_id, endpoint, tx_hash)
        return tx_hash

    def renew_model(self, index: int, private_key: Optional[str] = None) -> str:
        """Renew a model lease (call periodically)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.renewModel(index),
            private_key=private_key,
        )
        logger.info("Renewed model at index %d: %s", index, tx_hash)
        return tx_hash

    def deactivate_model(self, index: int, private_key: Optional[str] = None) -> str:
        """Voluntarily deactivate a model."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.deactivateModel(index),
            private_key=private_key,
        )
        logger.info("Deactivated model at index %d: %s", index, tx_hash)
        return tx_hash

    def update_endpoint(
        self, index: int, new_endpoint: str, private_key: Optional[str] = None
    ) -> str:
        """Update endpoint URL for a model entry."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.updateEndpoint(index, new_endpoint),
            private_key=private_key,
        )
        logger.info("Updated endpoint at index %d: %s", index, tx_hash)
        return tx_hash

    def report_offline(
        self,
        miner_address: str,
        model_index: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Report a miner-model as unreachable (validator only)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(miner_address)
        tx_hash = self._provider.send_transaction(
            self._contract.functions.reportOffline(addr, model_index),
            private_key=private_key,
        )
        logger.info("Reported offline: miner=%s index=%d: %s", miner_address, model_index, tx_hash)
        return tx_hash

    def cleanup(
        self, miner_address: str, index: int, private_key: Optional[str] = None
    ) -> str:
        """Remove an expired entry (anyone can call for gas refund)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(miner_address)
        tx_hash = self._provider.send_transaction(
            self._contract.functions.cleanup(addr, index),
            private_key=private_key,
        )
        logger.info("Cleaned up miner=%s index=%d: %s", miner_address, index, tx_hash)
        return tx_hash

    # ── TEE attestation ─────────────────────────────────────────────

    def has_tee(self, address: str) -> bool:
        """Check if a miner has active TEE attestation (cached 60s)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)

        cache_key = f"has_tee:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.hasTEE(addr).call()
        )
        self._cache.set(cache_key, result, ttl=60)
        return result

    def get_tee_capability(self, address: str) -> OnChainTEECapability:
        """Get full TEE capability for a miner (cached 60s)."""
        from web3 import Web3
        addr = Web3.to_checksum_address(address)

        cache_key = f"tee_cap:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self._provider.call_with_retry(
            lambda: self._contract.functions.getTEECapability(addr).call()
        )
        result = _parse_tee_capability(raw)
        self._cache.set(cache_key, result, ttl=60)
        return result

    def register_tee_attestation(
        self,
        platform: str,
        enclave_pub_key: bytes,
        attestation_hash: bytes,
        model_weight_hash: bytes = b"\x00" * 32,
        code_measurement: bytes = b"\x00" * 32,
        private_key: Optional[str] = None,
    ) -> str:
        """Register or update TEE attestation for this miner."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerTEEAttestation(
                platform, enclave_pub_key, attestation_hash,
                model_weight_hash, code_measurement,
            ),
            private_key=private_key,
        )
        logger.info("Registered TEE attestation (platform=%s): %s", platform, tx_hash)
        return tx_hash

    def revoke_tee_attestation(self, private_key: Optional[str] = None) -> str:
        """Revoke TEE attestation (e.g. key rotation, enclave restart)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.revokeTEEAttestation(),
            private_key=private_key,
        )
        logger.info("Revoked TEE attestation: %s", tx_hash)
        return tx_hash

    # ── Demand scores ─────────────────────────────────────────────

    def get_model_demand_score(self, model_id: str) -> int:
        """Get demand score for a model (0-10000 bps). Returns 0 if not set."""
        cache_key = f"demand:{model_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = self._provider.call_with_retry(
                lambda: self._contract.functions.getModelDemandScore(model_id).call()
            )
        except Exception:
            logger.warning("Failed to read demand score for %s, defaulting to 0", model_id)
            return 0

        self._cache.set(cache_key, result)
        return result

    def get_model_demand_scores(self, model_ids: List[str]) -> List[int]:
        """Batch-read demand scores for multiple models. Returns 0 for unset models."""
        if not model_ids:
            return []

        try:
            result = self._provider.call_with_retry(
                lambda: self._contract.functions.getModelDemandScores(model_ids).call()
            )
            return list(result)
        except Exception:
            logger.warning("Failed to batch-read demand scores, defaulting to 0")
            return [0] * len(model_ids)

    def update_demand_scores(
        self,
        model_ids: List[str],
        scores: List[int],
        receipt_hash: bytes,
        epoch_number: int,
        private_key: Optional[str] = None,
    ) -> str:
        """Post demand scores on-chain (validator only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.updateDemandScores(
                model_ids, scores, receipt_hash, epoch_number
            ),
            private_key=private_key,
        )
        # Invalidate cached scores
        for mid in model_ids:
            self._cache.invalidate(f"demand:{mid}")
        logger.info(
            "Updated demand scores for %d models, epoch=%d: %s",
            len(model_ids), epoch_number, tx_hash,
        )
        return tx_hash


def _parse_miner_model(raw) -> OnChainMinerModel:
    """Parse a tuple from getMinerModels() into OnChainMinerModel."""
    return OnChainMinerModel(
        model_id=raw[0],
        endpoint=raw[1],
        model_spec_ref=bytes(raw[2]),
        quant=raw[3],
        max_context_len=raw[4],
        expires_at=raw[5],
        active=raw[6],
    )


def _parse_tee_capability(raw) -> OnChainTEECapability:
    """Parse a tuple from getTEECapability() into OnChainTEECapability."""
    return OnChainTEECapability(
        enabled=raw[0],
        platform=raw[1],
        enclave_pub_key=bytes(raw[2]),
        attestation_hash=bytes(raw[3]),
        attested_at=raw[4],
        model_weight_hash=bytes(raw[5]) if len(raw) > 5 else b"",
        code_measurement=bytes(raw[6]) if len(raw) > 6 else b"",
    )
