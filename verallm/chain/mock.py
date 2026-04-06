"""Mock implementations for dev/testing without a live chain.

Activated when ``ChainConfig.mock=True`` or when ``web3`` is not installed.
Ensures all existing tests and dev workflows keep working.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from verallm.chain.config import ChainConfig
from verallm.chain.checkpoint import UsageCheckpointEvent
from verallm.chain.payment_types import DepositEvent
from verallm.chain.types import OnChainMinerModel, OnChainTEECapability, OnChainValidatorInfo
from verallm.types import ModelSpec

logger = logging.getLogger(__name__)


class MockModelRegistryClient:
    """Mock ModelRegistry that returns specs from local cache or pre-loaded data.

    If ``cache_dir`` is provided, loads ModelSpecs from ``.model_root_cache/*.pkl``
    files, matching the format used by ``verallm.registry.roots``.
    """

    def __init__(
        self,
        config: Optional[ChainConfig] = None,
        specs: Optional[Dict[str, ModelSpec]] = None,
        cache_dir: Optional[str] = None,
    ):
        self._specs: Dict[str, ModelSpec] = specs or {}

        if cache_dir:
            self._load_from_cache(cache_dir)

    def _load_from_cache(self, cache_dir: str) -> None:
        """Load ModelSpecs from pickle cache files."""
        import pickle

        cache_path = Path(cache_dir)
        for pkl_file in cache_path.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    spec = pickle.load(f)
                if isinstance(spec, ModelSpec):
                    self._specs[spec.model_id] = spec
                    logger.debug("Loaded mock spec from %s", pkl_file)
            except Exception as e:
                logger.warning("Failed to load %s: %s", pkl_file, e)

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        return self._specs.get(model_id)

    def get_model_list(self) -> List[str]:
        return list(self._specs.keys())

    def get_model_count(self) -> int:
        return len(self._specs)

    def register_model(self, spec: ModelSpec, private_key: Optional[str] = None) -> str:
        self._specs[spec.model_id] = spec
        return "0x" + "00" * 32  # fake tx hash

    def remove_model(self, model_id: str, private_key: Optional[str] = None) -> str:
        self._specs.pop(model_id, None)
        return "0x" + "00" * 32


class MockMinerRegistryClient:
    """Mock MinerRegistry with configurable miner entries."""

    def __init__(
        self,
        config: Optional[ChainConfig] = None,
        miners: Optional[Dict[str, List[OnChainMinerModel]]] = None,
        demand_scores: Optional[Dict[str, int]] = None,
    ):
        self._miners: Dict[str, List[OnChainMinerModel]] = miners or {}
        self._associated_uids: Dict[str, int] = {}  # evm_address -> uid
        self._demand_scores: Dict[str, int] = demand_scores or {}
        self._tee: Dict[str, OnChainTEECapability] = {}

    def get_miner_models(self, address: str) -> List[OnChainMinerModel]:
        return self._miners.get(address.lower(), [])

    def get_miner_model_count(self, address: str) -> int:
        return len(self._miners.get(address.lower(), []))

    def get_providers_for_model(self, model_id: str) -> List[str]:
        providers = []
        for addr, models in self._miners.items():
            if any(m.model_id == model_id and m.active for m in models):
                providers.append(addr)
        return providers

    def is_model_active(self, address: str, index: int) -> bool:
        models = self._miners.get(address.lower(), [])
        if index >= len(models):
            return False
        m = models[index]
        return m.active and m.expires_at > int(time.time())

    def get_associated_uid(self, evm_address: str) -> Optional[int]:
        return self._associated_uids.get(evm_address.lower())

    def set_associated_uid(self, evm_address: str, uid: int) -> None:
        """Test helper: simulate an associate_evm_key association."""
        self._associated_uids[evm_address.lower()] = uid

    def register_model(
        self, model_id, endpoint, model_spec_ref, quant, max_context_len, private_key=None
    ) -> str:
        return "0x" + "00" * 32

    def renew_model(self, index: int, private_key=None) -> str:
        return "0x" + "00" * 32

    def deactivate_model(self, index: int, private_key=None) -> str:
        return "0x" + "00" * 32

    def report_offline(self, miner_address, model_index, private_key=None) -> str:
        return "0x" + "00" * 32

    def cleanup(self, miner_address, index, private_key=None) -> str:
        return "0x" + "00" * 32

    def get_model_demand_score(self, model_id: str) -> int:
        return self._demand_scores.get(model_id, 0)

    def get_model_demand_scores(self, model_ids: List[str]) -> List[int]:
        return [self._demand_scores.get(mid, 0) for mid in model_ids]

    def update_demand_scores(
        self, model_ids, scores, receipt_hash, epoch_number, private_key=None
    ) -> str:
        for mid, score in zip(model_ids, scores):
            self._demand_scores[mid] = score
        return "0x" + "00" * 32

    # ── TEE attestation ───────────────────────────────────────────

    def has_tee(self, address: str) -> bool:
        cap = self._tee.get(address.lower())
        return cap is not None and cap.enabled

    def get_tee_capability(self, address: str) -> OnChainTEECapability:
        return self._tee.get(address.lower(), OnChainTEECapability())

    def register_tee_attestation(
        self, platform, enclave_pub_key, attestation_hash,
        model_weight_hash=b"\x00" * 32, private_key=None,
    ) -> str:
        addr = "mock_sender"
        self._tee[addr] = OnChainTEECapability(
            enabled=True,
            platform=platform,
            enclave_pub_key=enclave_pub_key,
            attestation_hash=attestation_hash,
            attested_at=0,
            model_weight_hash=model_weight_hash,
        )
        return "0x" + "00" * 32

    def revoke_tee_attestation(self, private_key=None) -> str:
        self._tee.pop("mock_sender", None)
        return "0x" + "00" * 32


class MockPaymentGatewayClient:
    """Mock PaymentGateway with in-memory deposit tracking."""

    def __init__(
        self,
        config: Optional[ChainConfig] = None,
        owner_cut_bps: int = 1000,
        netuid: int = 42,
    ):
        self._deposits: Dict[str, int] = {}  # address -> total deposited (wei)
        self._events: List[DepositEvent] = []
        self._owner_cut_bps = owner_cut_bps
        self._netuid = netuid
        self._block = 0

    def get_total_deposited(self, user_address: str) -> int:
        return self._deposits.get(user_address.lower(), 0)

    def get_owner_cut_bps(self) -> int:
        return self._owner_cut_bps

    def get_netuid(self) -> int:
        return self._netuid

    def deposit(
        self, validator_hotkey: bytes, amount_wei: int, private_key: Optional[str] = None
    ) -> str:
        # Simulate deposit — in mock we don't have a real sender address,
        # so we derive one from the private key if available
        sender = "0x" + "00" * 20
        if private_key:
            try:
                from eth_account import Account
                sender = Account.from_key(private_key).address.lower()
            except ImportError:
                pass

        self._deposits[sender] = self._deposits.get(sender, 0) + amount_wei
        self._block += 1
        self._events.append(DepositEvent(
            user=sender,
            validator_hotkey=validator_hotkey,
            tao_amount=amount_wei,
            block_number=self._block,
            tx_hash="0x" + "00" * 32,
        ))
        return "0x" + "00" * 32

    def get_deposit_events(
        self,
        from_block: int = 0,
        to_block: str | int = "latest",
        user: Optional[str] = None,
    ) -> List[DepositEvent]:
        events = self._events
        if from_block > 0:
            events = [e for e in events if e.block_number >= from_block]
        if isinstance(to_block, int):
            events = [e for e in events if e.block_number <= to_block]
        if user:
            addr = user.lower()
            events = [e for e in events if e.user.lower() == addr]
        return sorted(events, key=lambda e: e.block_number)

    # ── Test helpers ─────────────────────────────────────────────

    def _add_deposit(
        self,
        user_address: str,
        validator_hotkey: bytes,
        amount_wei: int,
        block_number: int = 0,
    ) -> None:
        """Directly inject a deposit for testing (bypasses tx logic)."""
        addr = user_address.lower()
        self._deposits[addr] = self._deposits.get(addr, 0) + amount_wei
        if block_number == 0:
            self._block += 1
            block_number = self._block
        self._events.append(DepositEvent(
            user=addr,
            validator_hotkey=validator_hotkey,
            tao_amount=amount_wei,
            block_number=block_number,
            tx_hash="0x" + "00" * 32,
        ))


class MockUsageCheckpointClient:
    """Mock UsageCheckpointRegistry with in-memory event tracking."""

    def __init__(self, config: Optional[ChainConfig] = None):
        self._events: List[UsageCheckpointEvent] = []
        self._block = 0

    def record_usage_checkpoints(
        self, users: list[str], consumed_amounts: list[int]
    ) -> str:
        assert len(users) == len(consumed_amounts), "Length mismatch"
        self._block += 1
        ts = int(time.time())
        for user, consumed in zip(users, consumed_amounts):
            self._events.append(UsageCheckpointEvent(
                user=user.lower(),
                total_consumed_wei=consumed,
                timestamp=ts,
                block_number=self._block,
                tx_hash="0x" + "00" * 32,
            ))
        return "0x" + "00" * 32

    def get_checkpoint_events(
        self,
        from_block: int = 0,
        to_block: str | int = "latest",
        user: Optional[str] = None,
    ) -> List[UsageCheckpointEvent]:
        events = self._events
        if from_block > 0:
            events = [e for e in events if e.block_number >= from_block]
        if isinstance(to_block, int):
            events = [e for e in events if e.block_number <= to_block]
        if user:
            addr = user.lower()
            events = [e for e in events if e.user.lower() == addr]
        return sorted(events, key=lambda e: e.block_number)

    def get_latest_checkpoints(self, from_block: int = 0) -> dict[str, int]:
        events = self.get_checkpoint_events(from_block=from_block)
        latest: dict[str, int] = {}
        for e in events:
            latest[e.user.lower()] = e.total_consumed_wei
        return latest


class MockValidatorRegistryClient:
    """Mock ValidatorRegistry with in-memory validator entries."""

    def __init__(self, config: Optional[ChainConfig] = None):
        self._validators: Dict[str, OnChainValidatorInfo] = {}
        self._default: Optional[str] = None

    def get_validator(self, address: str) -> OnChainValidatorInfo:
        addr = address.lower()
        return self._validators.get(addr, OnChainValidatorInfo(
            proxy_endpoint="", uid=0, registered_at=0, updated_at=0, active=False,
        ))

    def get_active_validators(self) -> list[tuple[str, OnChainValidatorInfo]]:
        return [
            (addr, info) for addr, info in self._validators.items() if info.active
        ]

    def get_proxy_validators(self) -> list[tuple[str, OnChainValidatorInfo]]:
        return [
            (addr, info) for addr, info in self._validators.items()
            if info.active and info.proxy_endpoint
        ]

    def is_evm_registered(self, address: str) -> bool:
        return address.lower() in self._validators

    def get_default(self) -> tuple[str, OnChainValidatorInfo]:
        if not self._default or self._default not in self._validators:
            raise ValueError("No default validator set")
        return (self._default, self._validators[self._default])

    def is_active_validator(self, address: str) -> bool:
        info = self._validators.get(address.lower())
        return info is not None and info.active

    def get_validator_count(self) -> int:
        return len(self._validators)

    def register_evm(self, uid: int, hotkey_seed: bytes = b"", netuid: int = 0, private_key=None) -> str:
        return "0x" + "00" * 32

    def register(self, proxy_endpoint: str, private_key=None) -> str:
        addr = "mock_validator"
        self._validators[addr] = OnChainValidatorInfo(
            proxy_endpoint=proxy_endpoint, uid=0,
            registered_at=int(time.time()), updated_at=int(time.time()),
            active=True,
        )
        return "0x" + "00" * 32

    def update_endpoint(self, new_endpoint: str, private_key=None) -> str:
        for info in self._validators.values():
            info.proxy_endpoint = new_endpoint
            info.updated_at = int(time.time())
            break
        return "0x" + "00" * 32

    def deactivate(self, private_key=None) -> str:
        return "0x" + "00" * 32

    def reactivate(self, private_key=None) -> str:
        return "0x" + "00" * 32

    def set_default_validator(self, validator_address: str, private_key=None) -> str:
        self._default = validator_address.lower()
        return "0x" + "00" * 32


def create_clients(config: ChainConfig):
    """Factory: return (model_client, miner_client, payment_client) — mock or real.

    If ``config.mock`` is True or ``web3`` is not installed, returns mock clients.
    The payment_client is None if no payment_gateway_address is configured.
    """
    if config.mock:
        logger.info("Using mock chain clients (config.mock=True)")
        payment = MockPaymentGatewayClient(config) if config.payment_gateway_address else None
        return (
            MockModelRegistryClient(config),
            MockMinerRegistryClient(config),
            payment,
        )

    try:
        from verallm.chain.model_registry import ModelRegistryClient
        from verallm.chain.miner_registry import MinerRegistryClient

        payment = None
        if config.payment_gateway_address:
            from verallm.chain.payment import PaymentGatewayClient
            from verallm.chain.provider import Web3Provider
            provider = Web3Provider(config)
            payment = PaymentGatewayClient(config, provider=provider)

        return ModelRegistryClient(config), MinerRegistryClient(config), payment
    except ImportError:
        logger.warning("web3 not installed, falling back to mock chain clients")
        payment = MockPaymentGatewayClient(config) if config.payment_gateway_address else None
        return (
            MockModelRegistryClient(config),
            MockMinerRegistryClient(config),
            payment,
        )
