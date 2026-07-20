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
    canary_inference_timeout: float = 300.0  # per-small-test inference timeout (seconds)
    canary_full_context_inference_timeout: float = 900.0
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
    miner_debug_enabled: bool = False
    miner_debug_state_path: str = "/tmp/verathos_miner_debug.json"
    miner_debug_refresh_seconds: float = 60.0

    # Public subnet runtime config. This becomes the ground truth for runtime
    # tuning params once deployed; chain scoring remains fallback only.
    subnet_config_url: str = "https://api.verathos.ai/v1/subnet-config"
    subnet_config_refresh_seconds: float = 120.0
    subnet_config_timeout_seconds: float = 5.0
    subnet_config_cache_path: str = ""
    subnet_config_disable: bool = False

    # Operator-controlled maintenance grace. Intended for coordinated releases
    # where validators should keep measuring but temporarily avoid penalties.
    maintenance_grace_enabled: bool = False
    maintenance_grace_open_ended: bool = False
    maintenance_grace_until_epoch: int | None = None
    maintenance_grace_until_unix_ts: int | None = None
    maintenance_grace_reason: str = ""
    maintenance_grace_suppress_score_zeroing: bool = True
    maintenance_grace_suppress_probation: bool = True
    maintenance_grace_suppress_capacity_score_gate: bool = True
    maintenance_grace_suppress_report_offline: bool = True
    maintenance_grace_suppress_proxy_proof_strikes: bool = True

    # Hot-capacity audit windows. Enabled by default; observe mode records
    # receipts/verdicts without affecting endpoint score.
    capacity_audit_enabled: bool = True
    capacity_audit_mode: str = "observe"  # observe | score_gate | soft_gate | enforce
    capacity_audit_ingest_host: str = "0.0.0.0"
    capacity_audit_ingest_port: int = 8091
    capacity_audit_public_url: str = ""  # validator-owned public ingest URL published through axon metadata
    capacity_audit_serve_axon: bool = True  # publish direct ingest IP:port via Bittensor axon metadata
    capacity_audit_validator_urls: str = ""  # emergency manual override for miner artifact targets
    capacity_audit_worker_poll_s: float = 2.0
    capacity_audit_cohort_min: int = 100
    capacity_audit_cohort_fraction: float = 0.025
    capacity_audit_cohort_max: int = 250
    capacity_audit_windows_per_epoch: int = 30
    capacity_audit_max_drain_fraction: float = 0.05
    capacity_audit_group_stress_fraction: float = 0.35
    capacity_audit_group_stress_min_size: int = 2
    capacity_audit_group_stress_slots_per_group: int = 2
    capacity_audit_beacon_hash_count: int = 1
    capacity_audit_min_registration_age_s: float = 0.0
    capacity_audit_lead_blocks: int = 5
    capacity_audit_proof_challenge_delay_blocks: int = 4
    capacity_audit_drain_seconds: float = 30.0
    capacity_audit_deadline_s: float = 30.0
    capacity_audit_transport_grace_s: float = 3.0
    capacity_audit_payload_deadline_s: float = 60.0
    capacity_audit_max_proof_payload_bytes: int = 32 * 1024 * 1024
    capacity_audit_require_proof_payload: bool = True
    capacity_audit_repeat_window_epochs: int = 20
    capacity_audit_timing_misses_for_zero_score: int = 2
    capacity_audit_hard_proof_misses_for_zero_score: int = 2
    capacity_audit_invalid_proof_misses_for_zero_score: int = 1
    capacity_audit_allow_timing_only_score_gate: bool = True
    capacity_audit_uid_escalation_min_entries: int = 2
    capacity_audit_uid_escalation_fraction: float = 0.10
    capacity_audit_uid_escalation_max_entries: int = 10
    capacity_audit_slot_refresh_blocks: int = 0
    capacity_audit_slot_snapshot_stale_blocks: int = 0
    capacity_audit_proof_verify_workers: int = 4

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
            "miner_debug_enabled": "VERATHOS_MINER_DEBUG_ENABLED",
            "miner_debug_state_path": "VERATHOS_MINER_DEBUG_STATE_PATH",
            "miner_debug_refresh_seconds": "VERATHOS_MINER_DEBUG_REFRESH_SECONDS",
            "subnet_config_url": "VERATHOS_SUBNET_CONFIG_URL",
            "subnet_config_refresh_seconds": "VERATHOS_SUBNET_CONFIG_REFRESH_SECONDS",
            "subnet_config_timeout_seconds": "VERATHOS_SUBNET_CONFIG_TIMEOUT_SECONDS",
            "subnet_config_cache_path": "VERATHOS_SUBNET_CONFIG_CACHE_PATH",
            "subnet_config_disable": "VERATHOS_SUBNET_CONFIG_DISABLE",
            "maintenance_grace_enabled": "VERATHOS_MAINTENANCE_GRACE_ENABLED",
            "maintenance_grace_open_ended": "VERATHOS_MAINTENANCE_GRACE_OPEN_ENDED",
            "maintenance_grace_until_epoch": "VERATHOS_MAINTENANCE_GRACE_UNTIL_EPOCH",
            "maintenance_grace_until_unix_ts": "VERATHOS_MAINTENANCE_GRACE_UNTIL_UNIX_TS",
            "maintenance_grace_reason": "VERATHOS_MAINTENANCE_GRACE_REASON",
            "maintenance_grace_suppress_score_zeroing": "VERATHOS_MAINTENANCE_GRACE_SUPPRESS_SCORE_ZEROING",
            "maintenance_grace_suppress_probation": "VERATHOS_MAINTENANCE_GRACE_SUPPRESS_PROBATION",
            "maintenance_grace_suppress_capacity_score_gate": "VERATHOS_MAINTENANCE_GRACE_SUPPRESS_CAPACITY_SCORE_GATE",
            "maintenance_grace_suppress_report_offline": "VERATHOS_MAINTENANCE_GRACE_SUPPRESS_REPORT_OFFLINE",
            "maintenance_grace_suppress_proxy_proof_strikes": "VERATHOS_MAINTENANCE_GRACE_SUPPRESS_PROXY_PROOF_STRIKES",
            "capacity_audit_enabled": "VERATHOS_CAPACITY_AUDIT_ENABLED",
            "capacity_audit_mode": "VERATHOS_CAPACITY_AUDIT_MODE",
            "capacity_audit_ingest_host": "VERATHOS_CAPACITY_AUDIT_INGEST_HOST",
            "capacity_audit_ingest_port": "VERATHOS_CAPACITY_AUDIT_INGEST_PORT",
            "capacity_audit_public_url": "VERATHOS_CAPACITY_AUDIT_PUBLIC_URL",
            "capacity_audit_serve_axon": "VERATHOS_CAPACITY_AUDIT_SERVE_AXON",
            "capacity_audit_validator_urls": "VERATHOS_CAPACITY_AUDIT_VALIDATOR_URLS",
            "capacity_audit_worker_poll_s": "VERATHOS_CAPACITY_AUDIT_WORKER_POLL_S",
            "capacity_audit_cohort_min": "VERATHOS_CAPACITY_AUDIT_COHORT_MIN",
            "capacity_audit_cohort_fraction": "VERATHOS_CAPACITY_AUDIT_COHORT_FRACTION",
            "capacity_audit_cohort_max": "VERATHOS_CAPACITY_AUDIT_COHORT_MAX",
            "capacity_audit_windows_per_epoch": "VERATHOS_CAPACITY_AUDIT_WINDOWS_PER_EPOCH",
            "capacity_audit_max_drain_fraction": "VERATHOS_CAPACITY_AUDIT_MAX_DRAIN_FRACTION",
            "capacity_audit_group_stress_fraction": "VERATHOS_CAPACITY_AUDIT_GROUP_STRESS_FRACTION",
            "capacity_audit_group_stress_min_size": "VERATHOS_CAPACITY_AUDIT_GROUP_STRESS_MIN_SIZE",
            "capacity_audit_group_stress_slots_per_group": "VERATHOS_CAPACITY_AUDIT_GROUP_STRESS_SLOTS_PER_GROUP",
            "capacity_audit_beacon_hash_count": "VERATHOS_CAPACITY_AUDIT_BEACON_HASH_COUNT",
            "capacity_audit_min_registration_age_s": "VERATHOS_CAPACITY_AUDIT_MIN_REGISTRATION_AGE_S",
            "capacity_audit_lead_blocks": "VERATHOS_CAPACITY_AUDIT_LEAD_BLOCKS",
            "capacity_audit_proof_challenge_delay_blocks": "VERATHOS_CAPACITY_AUDIT_PROOF_CHALLENGE_DELAY_BLOCKS",
            "capacity_audit_drain_seconds": "VERATHOS_CAPACITY_AUDIT_DRAIN_SECONDS",
            "capacity_audit_deadline_s": "VERATHOS_CAPACITY_AUDIT_DEADLINE_S",
            "capacity_audit_transport_grace_s": "VERATHOS_CAPACITY_AUDIT_TRANSPORT_GRACE_S",
            "capacity_audit_payload_deadline_s": "VERATHOS_CAPACITY_AUDIT_PAYLOAD_DEADLINE_S",
            "capacity_audit_max_proof_payload_bytes": "VERATHOS_CAPACITY_AUDIT_MAX_PROOF_PAYLOAD_BYTES",
            "capacity_audit_require_proof_payload": "VERATHOS_CAPACITY_AUDIT_REQUIRE_PROOF_PAYLOAD",
            "capacity_audit_repeat_window_epochs": "VERATHOS_CAPACITY_AUDIT_REPEAT_WINDOW_EPOCHS",
            "capacity_audit_timing_misses_for_zero_score": "VERATHOS_CAPACITY_AUDIT_TIMING_MISSES_FOR_ZERO_SCORE",
            "capacity_audit_hard_proof_misses_for_zero_score": "VERATHOS_CAPACITY_AUDIT_HARD_PROOF_MISSES_FOR_ZERO_SCORE",
            "capacity_audit_invalid_proof_misses_for_zero_score": "VERATHOS_CAPACITY_AUDIT_INVALID_PROOF_MISSES_FOR_ZERO_SCORE",
            "capacity_audit_allow_timing_only_score_gate": "VERATHOS_CAPACITY_AUDIT_ALLOW_TIMING_ONLY_SCORE_GATE",
            "capacity_audit_uid_escalation_min_entries": "VERATHOS_CAPACITY_AUDIT_UID_ESCALATION_MIN_ENTRIES",
            "capacity_audit_uid_escalation_fraction": "VERATHOS_CAPACITY_AUDIT_UID_ESCALATION_FRACTION",
            "capacity_audit_uid_escalation_max_entries": "VERATHOS_CAPACITY_AUDIT_UID_ESCALATION_MAX_ENTRIES",
            "capacity_audit_slot_refresh_blocks": "VERATHOS_CAPACITY_AUDIT_SLOT_REFRESH_BLOCKS",
            "capacity_audit_slot_snapshot_stale_blocks": "VERATHOS_CAPACITY_AUDIT_SLOT_SNAPSHOT_STALE_BLOCKS",
            "capacity_audit_proof_verify_workers": "VERATHOS_CAPACITY_AUDIT_PROOF_VERIFY_WORKERS",
        }

        _float_fields = {
            "ema_alpha", "heartbeat_interval_sec",
            "throughput_power", "canary_proof_sample_rate",
            "demand_bonus_max", "tao_usd_fallback", "verified_multiplier",
            "capacity_audit_cohort_fraction", "capacity_audit_drain_seconds",
            "capacity_audit_deadline_s", "capacity_audit_transport_grace_s",
            "capacity_audit_payload_deadline_s",
            "capacity_audit_max_drain_fraction",
            "capacity_audit_group_stress_fraction",
            "capacity_audit_uid_escalation_fraction",
            "capacity_audit_min_registration_age_s",
            "capacity_audit_worker_poll_s",
            "miner_debug_refresh_seconds",
            "subnet_config_refresh_seconds", "subnet_config_timeout_seconds",
        }
        _int_fields = {
            "epoch_blocks", "canary_small_count",
            "canary_full_context_count", "set_weights_epoch_blocks",
            "price_feed_cache_ttl",
            "probation_required_passes", "probation_escalation_epochs",
            "capacity_audit_ingest_port", "capacity_audit_cohort_min",
            "capacity_audit_cohort_max", "capacity_audit_windows_per_epoch",
            "capacity_audit_group_stress_min_size",
            "capacity_audit_group_stress_slots_per_group",
            "capacity_audit_beacon_hash_count",
            "capacity_audit_lead_blocks",
            "capacity_audit_proof_challenge_delay_blocks",
            "capacity_audit_repeat_window_epochs",
            "capacity_audit_timing_misses_for_zero_score",
            "capacity_audit_hard_proof_misses_for_zero_score",
            "capacity_audit_invalid_proof_misses_for_zero_score",
            "capacity_audit_uid_escalation_min_entries",
            "capacity_audit_uid_escalation_max_entries",
            "capacity_audit_slot_refresh_blocks",
            "capacity_audit_slot_snapshot_stale_blocks",
            "capacity_audit_proof_verify_workers",
            "capacity_audit_max_proof_payload_bytes",
            "maintenance_grace_until_epoch", "maintenance_grace_until_unix_ts",
        }
        _bool_fields = {
            "demand_bonus_enabled", "x402_testnet",
            "capacity_audit_enabled", "capacity_audit_require_proof_payload",
            "capacity_audit_allow_timing_only_score_gate",
            "capacity_audit_serve_axon",
            "subnet_config_disable",
            "miner_debug_enabled",
            "maintenance_grace_enabled",
            "maintenance_grace_open_ended",
            "maintenance_grace_suppress_score_zeroing",
            "maintenance_grace_suppress_probation",
            "maintenance_grace_suppress_capacity_score_gate",
            "maintenance_grace_suppress_report_offline",
            "maintenance_grace_suppress_proxy_proof_strikes",
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
