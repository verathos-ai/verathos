"""Python client for the ModelRegistry Solidity contract."""

from __future__ import annotations

import logging
from typing import List, Optional

from verallm.chain.cache import TTLCache
from verallm.chain.config import ChainConfig
from verallm.chain.provider import Web3Provider
from verallm.chain.types import OnChainModelSpec, on_chain_to_model_spec
from verallm.types import ModelSpec

logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """Read/write interface to the on-chain ModelRegistry.

    Read calls are free (view functions) and cached with a configurable TTL.
    Write calls require a funded EVM private key.
    """

    def __init__(self, config: ChainConfig, provider: Optional[Web3Provider] = None):
        self._config = config
        if not config.model_registry_address:
            raise ValueError(
                "model_registry_address not set. "
                "Deploy contracts first (scripts/deploy_testnet.py) and provide "
                "--chain-config or set VERATHOS_MODEL_REGISTRY."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.model_registry_address, "ModelRegistry"
        )
        self._cache = TTLCache(default_ttl=config.model_spec_cache_ttl)

    # ── Free reads ───────────────────────────────────────────────

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Fetch a ModelSpec from chain (cached).

        Returns None if the model is not registered.
        """
        cache_key = f"spec:{self._config.netuid}:{model_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self._provider.call_with_retry(
            lambda: self._contract.functions.getModelSpec(
                self._config.netuid, model_id
            ).call()
        )

        # Empty model_id means not registered
        if not raw[0]:
            return None

        oc = _parse_model_spec(raw)
        spec = on_chain_to_model_spec(oc)
        self._cache.set(cache_key, spec)
        return spec

    def get_model_list(self) -> List[str]:
        """Get all registered model IDs for this subnet."""
        cache_key = f"model_list:{self._config.netuid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._provider.call_with_retry(
            lambda: self._contract.functions.getModelList(
                self._config.netuid
            ).call()
        )
        self._cache.set(cache_key, result)
        return result

    def get_model_count(self) -> int:
        """Get number of registered models for this subnet."""
        return self._provider.call_with_retry(
            lambda: self._contract.functions.getModelCount(
                self._config.netuid
            ).call()
        )

    # ── Paid writes ──────────────────────────────────────────────

    def register_model(
        self, spec: ModelSpec, private_key: Optional[str] = None
    ) -> str:
        """Register or update a model on-chain. Returns tx hash."""
        from verallm.chain.types import model_spec_to_on_chain

        oc = model_spec_to_on_chain(spec)
        struct_tuple = _spec_to_tuple(oc)

        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerModel(self._config.netuid, struct_tuple),
            private_key=private_key,
        )

        # Invalidate cache
        self._cache.invalidate(f"spec:{self._config.netuid}:{spec.model_id}")
        self._cache.invalidate(f"model_list:{self._config.netuid}")

        logger.info("Registered model %s on-chain: %s", spec.model_id, tx_hash)
        return tx_hash

    def remove_model(self, model_id: str, private_key: Optional[str] = None) -> str:
        """Remove a model from the registry. Returns tx hash."""
        tx_hash = self._provider.send_transaction(
            self._contract.functions.removeModel(self._config.netuid, model_id),
            private_key=private_key,
        )

        self._cache.invalidate(f"spec:{self._config.netuid}:{model_id}")
        self._cache.invalidate(f"model_list:{self._config.netuid}")

        logger.info("Removed model %s from chain: %s", model_id, tx_hash)
        return tx_hash


# ── Internal helpers ─────────────────────────────────────────────


def _parse_model_spec(raw) -> OnChainModelSpec:
    """Parse the tuple returned by getModelSpec() into OnChainModelSpec."""
    return OnChainModelSpec(
        model_id=raw[0],
        weight_merkle_root=raw[1],
        layer_roots=[bytes(r) for r in raw[2]],
        num_layers=raw[3],
        hidden_dim=raw[4],
        intermediate_dim=raw[5],
        num_heads=raw[6],
        head_dim=raw[7],
        vocab_size=raw[8],
        quant_mode=raw[9],
        merkle_chunk_size=raw[10],
        activation=raw[11],
        norm_type=raw[12],
        attention_type=raw[13],
        num_experts=raw[14],
        expert_w_num_cols=raw[15],
        lm_head_root=bytes(raw[16]) if len(raw) > 16 else b"",
        embedding_root=bytes(raw[17]) if len(raw) > 17 else b"",
        weight_file_hash=bytes(raw[18]) if len(raw) > 18 else b"",
        tokenizer_hash=bytes(raw[19]) if len(raw) > 19 else b"",
    )


def _spec_to_tuple(oc: OnChainModelSpec) -> tuple:
    """Convert OnChainModelSpec to the tuple expected by registerModel()."""
    return (
        oc.model_id,
        oc.weight_merkle_root,
        oc.layer_roots,
        oc.num_layers,
        oc.hidden_dim,
        oc.intermediate_dim,
        oc.num_heads,
        oc.head_dim,
        oc.vocab_size,
        oc.quant_mode,
        oc.merkle_chunk_size,
        oc.activation,
        oc.norm_type,
        oc.attention_type,
        oc.num_experts,
        oc.expert_w_num_cols,
        oc.lm_head_root if oc.lm_head_root else b"\x00" * 32,
        oc.embedding_root if oc.embedding_root else b"\x00" * 32,
        oc.weight_file_hash if oc.weight_file_hash else b"\x00" * 32,
        oc.tokenizer_hash if oc.tokenizer_hash else b"\x00" * 32,
    )
