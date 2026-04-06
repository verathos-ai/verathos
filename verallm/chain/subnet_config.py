"""Python client for the SubnetConfig Solidity contract."""

from __future__ import annotations

import logging
from typing import List, Optional

from verallm.chain.cache import TTLCache
from verallm.chain.config import ChainConfig
from verallm.chain.provider import Web3Provider
from verallm.chain.types import ScoringParams

logger = logging.getLogger(__name__)


class SubnetConfigClient:
    """Read/write interface to the on-chain SubnetConfig.

    Read calls are free (view functions) and cached with a 5-minute TTL.
    Write calls require a funded EVM private key (subnet owner only).
    """

    def __init__(self, config: ChainConfig, provider: Optional[Web3Provider] = None):
        self._config = config
        if not config.subnet_config_address:
            raise ValueError(
                "subnet_config_address not set. "
                "Deploy SubnetConfig contract and provide "
                "--chain-config or set VERATHOS_SUBNET_CONFIG."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.subnet_config_address, "SubnetConfig"
        )
        self._cache = TTLCache(default_ttl=300.0)

    # ── TEE measurement allowlist reads ─────────────────────────

    def is_accepted_measurement(self, measurement_hash: bytes) -> bool:
        """Check if a measurement hash is in the on-chain allowlist (cached 5min)."""
        cache_key = f"measurement:{measurement_hash.hex()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._provider.call_with_retry(
            lambda: self._contract.functions.isAcceptedMeasurement(
                measurement_hash
            ).call()
        )
        self._cache.set(cache_key, result, ttl=300)
        return result

    # ── Miner blacklist reads ───────────────────────────────────

    def is_miner_blacklisted(self, address: str) -> bool:
        """Check if a miner is blacklisted (cached 5min)."""
        from web3 import Web3

        addr = Web3.to_checksum_address(address)
        cache_key = f"blacklist:{addr}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._provider.call_with_retry(
            lambda: self._contract.functions.isMinerBlacklisted(addr).call()
        )
        self._cache.set(cache_key, result, ttl=300)
        return result

    # ── Scoring parameters (single RPC call) ─────────────────────

    def get_scoring_params(self) -> ScoringParams:
        """Get all scoring parameters in a single RPC call (cached 5min)."""
        cache_key = "scoring_params"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        raw = self._provider.call_with_retry(
            lambda: self._contract.functions.getScoringParams().call()
        )
        result = ScoringParams(
            tee_bonus_bps=raw[0],
            ema_alpha_bps=raw[1],
            throughput_power_bps=raw[2],
            proof_sample_rate_bps=raw[3],
            probation_required_passes=raw[4],
            demand_bonus_max_bps=raw[5],
            emission_burn_bps=raw[6],
        )
        self._cache.set(cache_key, result, ttl=300)
        return result

    # ── Well-known feature flags ────────────────────────────────

    # Pre-computed key: keccak256("tee_enabled") — controls whether TEE
    # verification is active on the subnet. Default false = TEE disabled.
    TEE_ENABLED_KEY = bytes.fromhex(
        "16d3898c5e5da85792e8f0229f6544dd41a53713cbfc84c4f84fc3b1050d9b64"
    )

    def is_tee_enabled_on_subnet(self) -> bool:
        """Check if TEE is enabled on this subnet (cached 5min).

        Uses featureFlags[keccak256("tee_enabled")]. Default: false (disabled).
        The subnet owner sets this to true when TEE is ready for production.
        """
        return self.get_feature_flag(self.TEE_ENABLED_KEY)

    # ── Feature flags ───────────────────────────────────────────

    def get_feature_flag(self, key: bytes) -> bool:
        """Get a feature flag value (cached 5min)."""
        cache_key = f"flag:{key.hex()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._provider.call_with_retry(
            lambda: self._contract.functions.featureFlags(key).call()
        )
        self._cache.set(cache_key, result, ttl=300)
        return result

    # ── Generic KV ──────────────────────────────────────────────

    def get_config_value(self, key: bytes) -> bytes:
        """Get a generic config value (cached 5min)."""
        cache_key = f"config:{key.hex()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._provider.call_with_retry(
            lambda: self._contract.functions.configValues(key).call()
        )
        result_bytes = bytes(result)
        self._cache.set(cache_key, result_bytes, ttl=300)
        return result_bytes

    # ── Write functions (onlyOwner) ─────────────────────────────

    def set_accepted_measurement(
        self,
        measurement_hash: bytes,
        accepted: bool = True,
        private_key: Optional[str] = None,
    ) -> str:
        """Add or remove a measurement from the allowlist (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setAcceptedMeasurement(
                measurement_hash, accepted
            ),
            private_key=private_key,
        )
        self._cache.invalidate(f"measurement:{measurement_hash.hex()}")
        logger.info(
            "Set measurement %s accepted=%s: %s",
            measurement_hash.hex()[:16],
            accepted,
            tx_hash,
        )
        return tx_hash

    def set_accepted_measurements(
        self,
        hashes: List[bytes],
        values: List[bool],
        private_key: Optional[str] = None,
    ) -> str:
        """Batch update measurement allowlist (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setAcceptedMeasurements(hashes, values),
            private_key=private_key,
        )
        for h in hashes:
            self._cache.invalidate(f"measurement:{h.hex()}")
        logger.info("Batch-updated %d measurements: %s", len(hashes), tx_hash)
        return tx_hash

    def set_miner_blacklisted(
        self,
        address: str,
        blacklisted: bool = True,
        private_key: Optional[str] = None,
    ) -> str:
        """Blacklist or un-blacklist a miner (owner only)."""
        from web3 import Web3

        addr = Web3.to_checksum_address(address)
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setMinerBlacklisted(addr, blacklisted),
            private_key=private_key,
        )
        self._cache.invalidate(f"blacklist:{addr}")
        logger.info(
            "Set miner %s blacklisted=%s: %s", addr, blacklisted, tx_hash
        )
        return tx_hash

    def _set_scoring_param(
        self, fn_name: str, value: int, private_key: Optional[str] = None
    ) -> str:
        """Helper: call a scoring param setter and invalidate the batch cache."""
        fn = getattr(self._contract.functions, fn_name)
        tx_hash = self._provider.send_transaction(
            fn(value), private_key=private_key,
        )
        self._cache.invalidate("scoring_params")
        logger.info("Set %s=%d: %s", fn_name, value, tx_hash)
        return tx_hash

    def set_tee_bonus_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setTeeBonusBps", bps, private_key)

    def set_ema_alpha_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setEmaAlphaBps", bps, private_key)

    def set_throughput_power_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setThroughputPowerBps", bps, private_key)

    def set_proof_sample_rate_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setProofSampleRateBps", bps, private_key)

    def set_probation_required_passes(self, passes: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setProbationRequiredPasses", passes, private_key)

    def set_demand_bonus_max_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setDemandBonusMaxBps", bps, private_key)

    def set_emission_burn_bps(self, bps: int, private_key: Optional[str] = None) -> str:
        return self._set_scoring_param("setEmissionBurnBps", bps, private_key)

    def set_feature_flag(
        self, key: bytes, value: bool, private_key: Optional[str] = None
    ) -> str:
        """Set a feature flag (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setFeatureFlag(key, value),
            private_key=private_key,
        )
        self._cache.invalidate(f"flag:{key.hex()}")
        logger.info("Set feature flag %s=%s: %s", key.hex()[:16], value, tx_hash)
        return tx_hash

    def set_config_value(
        self, key: bytes, value: bytes, private_key: Optional[str] = None
    ) -> str:
        """Set a generic config value (owner only)."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.setConfigValue(key, value),
            private_key=private_key,
        )
        self._cache.invalidate(f"config:{key.hex()}")
        logger.info("Set config %s: %s", key.hex()[:16], tx_hash)
        return tx_hash
