"""Neuron configuration — extends ChainConfig with Bittensor-specific settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from verallm.chain.config import ChainConfig


@dataclass
class NeuronConfig(ChainConfig):
    """Configuration for Bittensor neuron wrappers.

    Inherits all chain config fields and adds subnet/wallet/scoring settings.
    """

    # Bittensor wallet
    wallet_name: str = "default"
    hotkey_name: str = "default"
    subtensor_network: str = "test"  # "finney", "test", "local"

    # Scoring parameters
    ema_alpha: float = 0.2  # EMA smoothing for scores
    throughput_power: float = 2.0  # superlinear exponent (sybil resistance)

    # Epoch timing (block-based)
    epoch_blocks: int = 360  # ~72 min at 12s/block
    epoch_grace_blocks: int = 10  # blocks after epoch boundary before pulling receipts
    set_weights_epoch_blocks: int = 360  # weight-setting interval (= epoch by default)

    # Canary testing
    canary_small_count: int = 12  # small canary tests per miner per epoch
    canary_full_context_count: int = 1  # full-context canary tests per miner per epoch
    canary_proof_sample_rate: float = 0.30  # probability of ZK proof verification on small canaries
    canary_inference_timeout: float = 300.0  # per-test inference timeout (seconds)
    epoch_receipt_pull_timeout: float = 30.0  # GET /epoch/{n}/receipts per-miner timeout (seconds)
    epoch_receipt_pull_overall_timeout: float = 60.0  # overall budget for all receipt pulls (seconds)

    # Identity challenge (anti-hijacking)
    identity_challenge_required: bool = True  # miners without /identity/challenge are excluded
    identity_challenge_timeout: float = 10.0  # seconds for POST /identity/challenge

    # Probation (proof failure penalty)
    probation_required_passes: int = 3  # consecutive clean epochs to exit probation
    probation_escalation_epochs: int = 5  # epochs on probation before reportOffline

    # Demand bonus
    demand_bonus_enabled: bool = True  # enable per-model demand bonus
    demand_bonus_max: float = 0.20  # 20% max bonus for highest-demand model

    # Miner heartbeat
    heartbeat_interval_sec: float = 43200.0  # 12 hours (half of 24h lease)

    # Discovery
    miner_endpoint_timeout: float = 10.0  # seconds for health check
    max_concurrent_verifications: int = 64  # I/O-bound (network), not CPU-bound

    # Pricing
    tao_usd_fallback: float = 500.0  # fallback TAO/USD price when CoinGecko is unreachable
    price_feed_cache_ttl: int = 300  # seconds to cache CoinGecko price
    verified_multiplier: float = 1.25  # 25% premium for proof-verified inference

    # x402 pay-per-request (USDC on Base L2)
    x402_recipient: str = ""  # Validator's Base USDC address (empty = x402 disabled)
    x402_testnet: bool = False  # Use Base Sepolia instead of mainnet
    x402_facilitator_url: str = ""  # Override facilitator URL (empty = auto)

    # X402Gateway on-chain USDC collection + TaoFi bridge
    x402_gateway_address: str = ""  # X402Gateway contract on Base (empty = direct-to-EOA)
    x402_base_rpc_url: str = ""  # Base RPC URL (empty = https://mainnet.base.org)

    # Shared state between validator and proxy processes
    shared_state_path: str = "/tmp/verathos_validator_state.json"

    @classmethod
    def from_env(cls, **overrides) -> NeuronConfig:
        """Build config from environment variables."""
        # Get base chain config fields
        base = ChainConfig.from_env()
        kwargs = {k: getattr(base, k) for k in ChainConfig.__dataclass_fields__}

        # Neuron-specific env vars
        neuron_env = {
            "wallet_name": "BT_WALLET_NAME",
            "hotkey_name": "BT_HOTKEY_NAME",
            "subtensor_network": "BT_SUBTENSOR_NETWORK",
            "ema_alpha": "VERATHOS_EMA_ALPHA",
            "heartbeat_interval_sec": "VERATHOS_HEARTBEAT_INTERVAL",
            "throughput_power": "VERATHOS_THROUGHPUT_POWER",
            "epoch_blocks": "VERATHOS_EPOCH_BLOCKS",
            "canary_small_count": "VERATHOS_CANARY_SMALL_COUNT",
            "canary_full_context_count": "VERATHOS_CANARY_FULL_CONTEXT_COUNT",
            "canary_proof_sample_rate": "VERATHOS_CANARY_PROOF_SAMPLE_RATE",
            "probation_required_passes": "VERATHOS_PROBATION_PASSES",
            "probation_escalation_epochs": "VERATHOS_PROBATION_ESCALATION",
            "set_weights_epoch_blocks": "VERATHOS_SET_WEIGHTS_EPOCH",
            "demand_bonus_max": "VERATHOS_DEMAND_BONUS_MAX",
            "demand_bonus_enabled": "VERATHOS_DEMAND_BONUS_ENABLED",
            "tao_usd_fallback": "VERATHOS_TAO_USD_FALLBACK",
            "price_feed_cache_ttl": "VERATHOS_PRICE_FEED_CACHE_TTL",
            "verified_multiplier": "VERATHOS_VERIFIED_MULTIPLIER",
            "x402_recipient": "VERATHOS_X402_RECIPIENT",
            "x402_testnet": "VERATHOS_X402_TESTNET",
            "x402_facilitator_url": "VERATHOS_X402_FACILITATOR_URL",
            "x402_gateway_address": "VERATHOS_X402_GATEWAY",
            "x402_base_rpc_url": "VERATHOS_X402_BASE_RPC",
            "shared_state_path": "VERATHOS_SHARED_STATE_PATH",
        }

        _float_fields = {
            "ema_alpha", "heartbeat_interval_sec",
            "throughput_power", "canary_proof_sample_rate",
            "demand_bonus_max", "tao_usd_fallback", "verified_multiplier",
        }
        _int_fields = {
            "epoch_blocks", "canary_small_count",
            "canary_full_context_count", "set_weights_epoch_blocks",
            "price_feed_cache_ttl",
            "probation_required_passes", "probation_escalation_epochs",
        }
        _bool_fields = {
            "demand_bonus_enabled", "x402_testnet",
        }

        for attr, env_var in neuron_env.items():
            val = os.environ.get(env_var)
            if val is not None:
                if attr in _float_fields:
                    kwargs[attr] = float(val)
                elif attr in _int_fields:
                    kwargs[attr] = int(val)
                elif attr in _bool_fields:
                    kwargs[attr] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[attr] = val

        kwargs.update(overrides)
        return cls(**kwargs)
