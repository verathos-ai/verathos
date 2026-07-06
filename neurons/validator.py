#!/usr/bin/env python3
"""ValidatorNeuron — epoch-based canary testing for Verathos.

Lifecycle:
1. Ensure hotkey is linked (for reportOffline access).
2. Subscribe to current-head block headers via WebSocket.
3. On epoch boundary (every ``epoch_blocks``):
   a. Discover ALL active miners from MinerRegistry.
   b. Plan canary tests: ``canary_small_count`` small + ``canary_full_context_count``
      full-context tests per miner, spread across the epoch.
4. Each block: dispatch pending canary tests (target_block <= current_block).
   a. Send test through POST /chat (same endpoint as organic proxy traffic).
   b. Optionally verify ZK proof.
   c. Create signed receipt with metrics (TTFT, tok/s, tokens generated).
   d. Push receipt to miner via POST /epoch/receipt.
5. After epoch + grace window:
   a. Pull all receipts from each miner: GET /epoch/{n}/receipts.
   b. Build EpochOutcome per miner-model entry.
   c. Score entries: utility × throughput² × latency, update EMAs.
   d. Apply traffic volume multiplier.
6. At weight-setting boundary:
   a. Compute per-UID weights (additive aggregation × traffic volume).
   b. ``set_weights()`` on Substrate.

Usage:
    python -m neurons.validator \\
        --wallet default --hotkey default --netuid 42 \\
        --chain-config chain_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import signal
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import ipaddress

import bittensor as bt
import httpx

from neurons.canary import FULL_CONTEXT_TOKEN_CAP, CanaryScheduler, CanaryTest
from neurons.capacity_audit import (
    CapacityAuditRuntimeConfig,
    CapacitySlot,
    PROTOCOL_VERSION,
    build_capacity_slot_group_key,
    capacity_audit_window_fits_epoch,
    capacity_audit_window_triggered,
    capacity_gpu_pass_count,
    derive_audit_id,
    derive_audit_seed,
    derive_audit_seed_from_hashes,
    deterministic_sample_slots,
    derive_proof_challenge_seed,
    derive_proof_seed,
    lease_id,
    match_gpu_class,
    select_capacity_audit_slots,
    slot_id,
    verify_artifact_signature,
    window_cohort_budget,
)
from neurons.capacity_audit_combined import (
    COMBINED_PROOF_FORMAT,
    verify_combined_proof_payload,
)
from neurons.config import NeuronConfig
from neurons.discovery import ActiveMiner, discover_active_miners
from neurons.subnet_runtime_config import (
    RuntimeSubnetConfigClient,
    apply_runtime_config_to_neuron_config,
    capacity_audit_config_from_neuron_config,
)
from neurons.model_resolve import validate_capacity_recommended_model
from neurons.version import spec_version, version_str, validator_version, validator_version_str
from neurons.receipts import (
    ServiceReceipt,
    ValidatorAuthority,
    create_receipt,
    receipt_from_dict,
    receipt_observed_interval,
    receipt_has_validator_observed_timing,
    receipt_to_dict,
    validator_observed_timing,
    verify_service_receipt,
)
from neurons.scoring import (
    CompositeScorer,
    EpochOutcome,
    ModelEntryScore,
    MinerScoreState,
    ProbationTracker,
    compute_demand_bonus,
    compute_model_base_utility,
    compute_model_demand,
    compute_peer_medians,
)
from neurons.validator_db import ValidatorStateDB

from verallm.chain.config import ChainConfig
from verallm.chain.mock import create_clients
from verallm.chain.types import ScoringParams
from verallm.chain.wallet import derive_evm_private_key, derive_evm_address
from verallm.api.client import ValidatorClient
from verallm.config import Config
from verallm.registry import get_model, MODELS_BY_ID

logger = logging.getLogger(__name__)


def _validator_probation_state_path() -> str:
    explicit = os.environ.get("VERATHOS_PROBATION_STATE_PATH", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    data_dir = os.environ.get("VERALLM_DATA_DIR", "").strip()
    if data_dir:
        return os.path.join(os.path.expanduser(data_dir), "verathos_probation.json")
    return "/tmp/verathos_probation.json"


def _coerce_nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_health_hardware(hw: object) -> tuple[bool, dict[str, object]]:
    """Validate and normalize optional miner /health hardware metadata."""
    defaults: dict[str, object] = {
        "gpu_name": "",
        "gpu_count": 0,
        "vram_gb": 0,
        "compute_capability": "",
        "gpu_uuids": [],
    }
    if hw is None:
        return True, defaults
    if not isinstance(hw, dict):
        return False, defaults

    gpu_count = _coerce_nonnegative_int(hw.get("gpu_count"))
    vram_gb = _coerce_nonnegative_int(hw.get("vram_gb"))
    gpu_uuids = hw.get("gpu_uuids") or []
    if not isinstance(gpu_uuids, list):
        gpu_uuids = []

    normalized: dict[str, object] = {
        "gpu_name": hw.get("gpu_name") or "",
        "gpu_count": gpu_count,
        "vram_gb": vram_gb,
        "compute_capability": hw.get("compute_capability") or "",
        "gpu_uuids": gpu_uuids,
    }
    claims_gpu = bool(
        normalized["gpu_name"]
        or normalized["compute_capability"]
        or normalized["gpu_uuids"]
        or gpu_count > 0
        or vram_gb > 0
    )
    if claims_gpu and (gpu_count <= 0 or vram_gb <= 0):
        return False, normalized
    return True, normalized


def _identity_verification_key(miner: ActiveMiner) -> tuple[str, str]:
    """Stable key for one identity challenge across a miner's model entries."""
    return (miner.address.lower(), miner.endpoint.rstrip("/"))


def _group_miners_for_identity(miners: List[ActiveMiner]) -> Dict[tuple[str, str], List[ActiveMiner]]:
    groups: Dict[tuple[str, str], List[ActiveMiner]] = {}
    for miner in miners:
        groups.setdefault(_identity_verification_key(miner), []).append(miner)
    return groups


# ---------------------------------------------------------------------------
# Tokenizer cache for input commitment verification
# ---------------------------------------------------------------------------

# Imported at module scope so transformers' _LazyModule resolves once,
# single-threaded at startup — concurrent canary worker threads racing
# `from transformers import AutoTokenizer` previously hit a half-initialized
# module and raised ImportError ~30% of the time on first canary burst.
from transformers import AutoTokenizer as _AutoTokenizer

# Silence library logs that fire on every tokenizer cache miss
# (HEAD requests to HuggingFace + verbose config dumps).  Validator
# operator only cares about actionable events; transformers and hf_hub
# at INFO/DEBUG produce many lines per canary on cold cache.
import transformers as _transformers
_transformers.logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

_tokenizer_cache: Dict[str, object] = {}
_tokenizer_lock = threading.Lock()


def _get_tokenizer(model_id: str):
    """Load and cache a tokenizer for input commitment verification.

    Tokenizers are lightweight (~1-5 MB each, CPU only, no model weights).
    Cached after first load so repeated canary tests for the same model
    are instant.
    """
    with _tokenizer_lock:
        if model_id in _tokenizer_cache:
            return _tokenizer_cache[model_id]
        bt.logging.debug(f"Loading tokenizer for input commitment: {model_id}")
        tokenizer = _AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _tokenizer_cache[model_id] = tokenizer
        return tokenizer


def _compute_expected_input_commitment(
    model_id: str,
    prompt: str,
    enable_thinking: bool = True,
) -> Optional[bytes]:
    """Compute the expected input_commitment for a prompt.

    Replicates the miner's tokenization path:
    1. Load the model's tokenizer
    2. Apply chat template (same logic as verallm.api.server._apply_chat_template)
    3. Tokenize to get input_token_ids
    4. Return SHA256(int64 bytes)

    Returns None if tokenizer loading fails (non-fatal — verification
    proceeds without input commitment check).
    """
    import numpy as np
    try:
        tokenizer = _get_tokenizer(model_id)
    except Exception as e:
        bt.logging.warning(f"Cannot load tokenizer for {model_id}: {e}")
        return None

    messages = [{"role": "user", "content": prompt}]

    # Replicate _apply_chat_template logic from verallm/api/server.py
    _is_mistral = "MistralTokenizer" in type(tokenizer).__name__
    try:
        if _is_mistral:
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        else:
            # Replicate _chat_template_kwargs
            import inspect
            try:
                src = inspect.getsource(tokenizer.apply_chat_template)
            except (TypeError, OSError):
                src = ""
            tpl = getattr(tokenizer, "chat_template", "") or ""
            kwargs = {}
            if "enable_thinking" in src or "enable_thinking" in tpl:
                kwargs["enable_thinking"] = enable_thinking
            elif "reasoning_effort" in tpl:
                kwargs["reasoning_effort"] = "none" if not enable_thinking else "medium"

            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **kwargs,
            )
            token_ids = tokenizer.encode(formatted)
    except Exception:
        # No chat template — encode raw prompt
        token_ids = tokenizer.encode(prompt)

    token_arr = np.asarray(token_ids, dtype=np.int64)
    return hashlib.sha256(token_arr.astype("<i8", copy=False).tobytes()).digest()


def _verify_code_measurement(
    platform: str,
    on_chain_cap,
    fresh_attestation_report: bytes,
    subnet_config_client,
    miner_label: str,
) -> None:
    """Verify code measurement from attestation against on-chain allowlist.

    Always enforced — no skips. Raises RuntimeError on failure.

    Steps:
      1. Extract raw measurement from fresh attestation report
      2. Compute keccak256(raw_measurement) — same normalization as miner registration
      3. Compare with on-chain codeMeasurement (miner must not have re-registered a lie)
      4. Check SubnetConfig allowlist (subnet owner must have published the hash)
    """
    from web3 import Web3

    # Extract actual measurement from the fresh attestation report
    if platform == "tdx":
        body_offset = 48  # after quote header
        mr_td_offset = body_offset + 0x0B8
        raw_measurement = fresh_attestation_report[mr_td_offset : mr_td_offset + 48]
        if len(raw_measurement) != 48:
            raise RuntimeError("TDX: cannot extract mr_td from attestation report")
        actual_measurement = bytes(Web3.keccak(raw_measurement))
    elif platform == "sev-snp":
        raw_measurement = fresh_attestation_report[0x090 : 0x090 + 48]
        if len(raw_measurement) != 48:
            raise RuntimeError("SEV-SNP: cannot extract launch_digest from attestation report")
        actual_measurement = bytes(Web3.keccak(raw_measurement))
    else:
        # mock / gpu — deterministic placeholder
        actual_measurement = bytes(Web3.keccak(b"mock"))

    # Cross-check: actual measurement from hardware must match what miner registered on-chain
    registered = on_chain_cap.code_measurement
    if registered and registered != actual_measurement:
        bt.logging.error(
            f"TEE verify: code measurement mismatch for {miner_label} — "
            f"on-chain={registered.hex()[:16]}... actual={actual_measurement.hex()[:16]}..."
        )
        raise RuntimeError("Code measurement mismatch between attestation and on-chain registration")

    # Check SubnetConfig allowlist
    if subnet_config_client is None:
        raise RuntimeError("SubnetConfig client not available — cannot verify code measurement")

    if not subnet_config_client.is_accepted_measurement(actual_measurement):
        bt.logging.error(
            f"TEE verify: code measurement not in allowlist for {miner_label}: "
            f"{actual_measurement.hex()[:16]}..."
        )
        raise RuntimeError("Code measurement not in on-chain allowlist")

    bt.logging.debug(
        f"TEE verify: code measurement OK for {miner_label} ({actual_measurement.hex()[:16]}...)"
    )


def _extract_hotkey_seed(wallet_name: str, hotkey_name: str, wallet) -> bytes:
    """Extract the 32-byte Ed25519 seed from a Bittensor hotkey.

    Works across bittensor v9 (wallet.hotkey.private_key) and v10+ (keyfile JSON).
    """
    if hasattr(wallet.hotkey, "private_key"):
        return bytes(wallet.hotkey.private_key[:32])
    import json
    from pathlib import Path
    hk_path = Path.home() / f".bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}"
    hk_data = json.loads(hk_path.read_text())
    return bytes.fromhex(hk_data["secretSeed"].replace("0x", ""))


class ValidatorNeuron:
    """Epoch-based canary testing validator.

    Tests miners through the normal inference pipeline, making canary tests
    indistinguishable from real traffic.  Receipt correlation across the epoch
    catches cheaters.
    """

    def __init__(self, config: NeuronConfig):
        self.config = config
        self.scorer = CompositeScorer(
            ema_alpha=config.ema_alpha,
            throughput_power=config.throughput_power,
        )
        self._running = True
        self._probation_tracker = ProbationTracker(
            required_passes=config.probation_required_passes,
            escalation_epochs=config.probation_escalation_epochs,
            state_path=_validator_probation_state_path(),
        )

        # SQLite-backed validator state database
        db_path = os.path.join(
            os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
            "verathos_validator.db",
        )
        self._db = ValidatorStateDB(db_path=db_path)

        self.evm_pk = ""
        self.evm_addr = ""
        # Set True when validator runs without EVM registration (no on-chain
        # reportOffline / updateDemandScores). Triggered explicitly by
        # config.no_evm or implicitly when registerEvm fails (e.g. low TAO).
        self._evm_disabled: bool = bool(getattr(config, "no_evm", False))
        self._model_client = None
        self._miner_client = None
        self._subnet_config_client = None
        self._blacklisted_uids: set = set()
        self._blacklisted_addresses: set = set()  # lowercase EVM addrs
        self._burn_uid: int = 0
        self._scoring = ScoringParams()
        self._last_model_emission_budgets: Dict[str, float] = {}
        self._last_model_emission_groups: Dict[str, str] = {}
        self._last_model_group_budgets: Dict[str, float] = {}
        self._last_model_bucket_burn: float = 0.0
        self._bt_module = None
        self.__subtensor = None

        # SS58 cache: EVM address (lowercase) → {hotkey_ss58, coldkey_ss58}
        self._ss58_cache: Dict[str, Dict[str, str]] = {}

        # Epoch state
        self._current_epoch: int = 0
        self._epoch_start_block: int = 0
        self._canary_scheduler: Optional[CanaryScheduler] = None
        self._canary_scheduler_lock = threading.Lock()
        self._epoch_miners: List[ActiveMiner] = []
        self._epoch_miners_discovery_valid: bool = False
        # {(lowercase miner_address, model_index): expected_receipt_count}
        self._expected_receipts: Dict[Tuple[str, int], int] = {}
        # {epoch_number: {(miner_address, model_index): in_flight_count}}
        self._inflight_canaries: Dict[int, Dict[Tuple[str, int], int]] = {}
        self._closing_inflight_canaries: Dict[int, Dict[Tuple[str, int], int]] = {}
        # {(lowercase miner_address, model_index): 503_skip_count} — reset each epoch
        self._busy_skips: Dict[Tuple[str, int], int] = {}
        # Miners that entered probation via busy-skips (not real proof failure)
        # — maps to list of unix timestamps when 503s occurred, used to verify
        # organic receipts overlap temporally (miner was genuinely busy then)
        self._busy_skip_probations: Dict[Tuple[str, int], List[int]] = {}
        # {model_id: ModelSpec} — cached per epoch, avoids RPC per canary
        self._model_spec_cache: Dict[str, object] = {}
        # Remote miner_version tracking — opens a forgiveness window after a
        # release lands in the public repo, so miners restarting to pull the
        # new code don't get probation for "canary errors" that are really
        # vLLM-reload downtime.  Window is per-miner one-shot, lasts 2 epochs.
        self._miner_version_last_seen: int = 0
        self._miner_version_bump_at: float = 0.0
        self._miner_version_last_check: float = 0.0
        self._restart_forgiven: Set[Tuple[str, int]] = set()
        # Per-key set of (address, model_index) that had zero full-context
        # canary successes in the previous epoch.  Capability FAILURE only
        # probates when BOTH the previous and current epoch hit zero, so
        # one transient validator-side blip doesn't probate the network.
        self._zero_fc_last_epoch: Set[Tuple[str, int]] = set()
        # Epoch close state
        self._pending_epoch_close: Optional[int] = None
        self._auto_updater = None  # Set by main() if --auto-update
        self._epoch_close_block: int = 0
        self._epoch_close_retry_after: float = 0.0  # monotonic time
        self._epoch_close_backoff: float = 30.0  # seconds, doubles on failure
        self._last_known_block: int = 0  # fallback for _get_current_block
        self._last_block_hash_warning_at: float = 0.0
        self._capacity_audit_server = None
        self._capacity_audit_server_thread = None
        self._capacity_audit_schedule_lock = threading.Lock()
        self._capacity_audit_slot_snapshot_lock = threading.Lock()
        self._capacity_audit_slot_snapshot: list[tuple[CapacitySlot, object]] = []
        self._capacity_audit_slot_snapshot_block: int = 0
        self._capacity_audit_slot_snapshot_updated_at: float = 0.0
        self._capacity_audit_slot_snapshot_refreshing = False
        self._capacity_audit_slot_snapshot_last_error = ""
        self._capacity_audit_verifier_unhealthy = False
        self._capacity_audit_verifier_last_error = ""
        self._capacity_audit_cfg = capacity_audit_config_from_neuron_config(config)
        self._subnet_runtime_config_client = RuntimeSubnetConfigClient.from_config(
            config,
            log=bt.logging,
        )
        self._subnet_runtime_config_key: tuple[int, Optional[int], str] | None = None
        self._subnet_runtime_config_authoritative = False

        # Thread pool for concurrent canary tests. This pool is reset at epoch
        # rollover so queued stale canaries cannot occupy the next epoch.
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_verifications,
        )
        # Control-plane work must not wait behind canary inference. Epoch setup,
        # chain reports, blacklist checks, and weight setting use this pool.
        self._control_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_verifications,
        )
        # Capacity-audit scheduling must not queue behind epoch identity checks:
        # short drain windows are intentionally only a few blocks long. Keep
        # scheduling serialized so adjacent hash-triggered windows observe each
        # other's freshly written drains before selecting endpoint slots.
        self._capacity_audit_executor = ThreadPoolExecutor(max_workers=1)
        self._capacity_audit_discovery_executor = ThreadPoolExecutor(max_workers=1)
        proof_workers = max(
            1,
            int(getattr(config, "capacity_audit_proof_verify_workers", 4) or 4),
        )
        self._capacity_audit_proof_executor = ThreadPoolExecutor(max_workers=proof_workers)

    def _refresh_subnet_runtime_config(
        self,
        *,
        current_epoch: int | None = None,
        force: bool = False,
    ) -> bool:
        client = getattr(self, "_subnet_runtime_config_client", None)
        if client is None:
            return False
        runtime = client.get(current_epoch=current_epoch, force=force)
        if runtime is None:
            self._subnet_runtime_config_authoritative = False
            return False
        authoritative = bool(getattr(client, "last_authoritative", False))
        key = runtime.cache_key
        if key == self._subnet_runtime_config_key:
            self._subnet_runtime_config_authoritative = authoritative
            return True

        previous_proof_workers = int(
            getattr(self.config, "capacity_audit_proof_verify_workers", 4) or 4
        )
        apply_runtime_config_to_neuron_config(runtime, self.config)
        self._scoring = runtime.scoring
        self._last_good_scoring = runtime.scoring
        self._capacity_audit_cfg = runtime.capacity_audit
        self._probation_tracker.required_passes = runtime.scoring.probation_required_passes
        self._probation_tracker.escalation_epochs = runtime.probation_escalation_epochs
        for state in getattr(self._probation_tracker, "_probation", {}).values():
            state.required_passes = runtime.scoring.probation_required_passes
            state.escalation_epochs = runtime.probation_escalation_epochs

        proof_workers = max(
            1,
            int(getattr(self.config, "capacity_audit_proof_verify_workers", 4) or 4),
        )
        if proof_workers != previous_proof_workers:
            old_executor = self._capacity_audit_proof_executor
            self._capacity_audit_proof_executor = ThreadPoolExecutor(max_workers=proof_workers)
            old_executor.shutdown(wait=False, cancel_futures=True)

        self.scorer.ema_alpha = self._scoring.ema_alpha
        self.scorer.throughput_power = self._scoring.throughput_power
        self._subnet_runtime_config_key = key
        self._subnet_runtime_config_authoritative = authoritative
        bt.logging.info(
            f"Applied runtime subnet config version={runtime.version} "
            f"effective_epoch={runtime.effective_epoch} source={runtime.source or 'server'} "
            f"authoritative={authoritative}"
        )
        return True

    @property
    def _subtensor(self):
        """Lazy Subtensor connection — only connects when actually needed.

        Retries on transient errors (rate limits, network blips). Local
        subtensor connects on first try; public RPC may need a few retries
        when the per-IP quota is saturated.
        """
        if self.__subtensor is None:
            import time as _time
            bt_log = __import__("bittensor").logging
            bt_log.info("Connecting to Subtensor...")
            bt = self._bt_module
            SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            attempt = 0
            while True:
                try:
                    self.__subtensor = SubtensorCls(network=self.config.subtensor_network)
                    break
                except Exception as e:
                    attempt += 1
                    wait = min(60, 2 ** min(attempt, 6))  # 2,4,8,16,32,60,60,...
                    bt_log.warning(
                        f"Subtensor connect failed (attempt {attempt}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    _time.sleep(wait)
            bt_log.info("Subtensor connected")
        return self.__subtensor

    def setup(self):
        """Initialize chain clients and Bittensor wallet."""
        try:
            import bittensor as bt
        except ImportError:
            bt.logging.error("bittensor not installed")
            sys.exit(1)

        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        wallet = WalletCls(name=self.config.wallet_name, hotkey=self.config.hotkey_name)
        hotkey_seed = _extract_hotkey_seed(
            self.config.wallet_name, self.config.hotkey_name, wallet,
        )

        self.evm_pk = derive_evm_private_key(hotkey_seed)
        self.evm_addr = derive_evm_address(hotkey_seed)
        bt.logging.info(f"Validator EVM address: {self.evm_addr}")

        bt.logging.info("Creating chain clients...")
        self._model_client, self._miner_client, self._payment_client = create_clients(self.config)
        runtime_config_loaded = self._refresh_subnet_runtime_config(force=True)

        # SubnetConfig client for TEE measurement verification, blacklists, scoring params
        _sn_config_addr = getattr(self.config, "subnet_config_address", "")
        if _sn_config_addr:
            try:
                from verallm.chain.subnet_config import SubnetConfigClient
                _sn_chain_config = ChainConfig(
                    rpc_url=getattr(self.config, "rpc_url", ""),
                    chain_id=getattr(self.config, "chain_id", 945),
                    subnet_config_address=_sn_config_addr,
                )
                self._subnet_config_client = SubnetConfigClient(_sn_chain_config)
                bt.logging.info(f"SubnetConfig client initialized: {_sn_config_addr}")
                if not runtime_config_loaded:
                    # Chain scoring is fallback only when the public runtime
                    # subnet config and its local cache are unavailable.
                    try:
                        self._scoring = self._subnet_config_client.get_scoring_params()
                        self._last_good_scoring = self._scoring
                        bt.logging.info(
                            f"SubnetConfig fallback boot read: burn={self._scoring.emission_burn:.0%} "
                            f"ema={self._scoring.ema_alpha:.2f} tp={self._scoring.throughput_power:.1f}"
                        )
                    except Exception as e:
                        bt.logging.warning(f"SubnetConfig fallback boot read failed, using defaults: {e}")
            except Exception as e:
                bt.logging.warning(f"SubnetConfig client failed to initialize: {e}")
                self._subnet_config_client = None
        else:
            bt.logging.info("SubnetConfig not configured (no subnet_config_address in chain config)")

        # Store wallet for set_weights + EVM registration.
        # Subtensor + metagraph are lazy-loaded on first use (testnet WS is slow).
        self._wallet = wallet
        self.__subtensor = None
        self._metagraph = None
        self._bt_module = bt

        # Validator signing identity — Sr25519 (the same key the metagraph
        # publishes).  Receipts are anchored to this pubkey so that
        # verify_service_receipt can resolve SS58 → UID against a fresh
        # metagraph snapshot and reject anything not signed by a registered,
        # permitted validator.  The 32-byte ``_validator_hotkey_bytes`` is
        # the raw Sr25519 public key — equivalently, the bytes underlying
        # ``_validator_hotkey_ss58``.
        from substrateinterface import Keypair as _Keypair
        _kp = _Keypair.create_from_seed(hotkey_seed[:32].hex())
        self._validator_hotkey_bytes = _kp.public_key
        self._validator_private_key = hotkey_seed

        # SS58 hotkey address for Sr25519 request signing (miner auth)
        self._validator_hotkey_ss58 = wallet.hotkey.ss58_address

        # Cached at the top of every _close_epoch (one eth_call/epoch, ~72 min).
        # Used by verify_service_receipt's total-stake gate.  Broader than
        # the ValidatorRegistry contract's alpha-only register gate by
        # design — see ValidatorAuthority docstring.
        self._cached_min_validator_stake: float = 0.0

        # Ensure contract-level EVM → UID mapping exists (needs wallet + subtensor + hotkey_seed above)
        self._ensure_evm_registered()
        self._ensure_validator_registry_registered()

        # Resolve subnet owner UID (burn target) — never changes at runtime
        self._burn_uid = self._resolve_burn_uid()

        # Load persisted scores from DB into in-memory scorer
        self._load_scores_from_db()

        # Block from which to start processing.  Set in main_loop() to the
        # current chain head so we never replay historical blocks after a chain
        # reset / fast-sync.  All blocks before this are silently skipped.
        self._sync_block: int = 0
        self._capacity_audit_last_finalized_confirmed: int = 0

    def _ensure_evm_registered(self):
        """Ensure registerEvm(uid) has been called on the current MinerRegistry."""
        if self._evm_disabled:
            bt.logging.info("EVM disabled (--no-evm), skipping MinerRegistry registration")
            return
        try:
            if self._miner_client.is_evm_registered(self.evm_addr):
                return
        except Exception:
            return

        # Resolve UID from Substrate metagraph (not from contract — UidLookup is broken)
        uid = None
        try:
            hk_ss58 = self._wallet.hotkey.ss58_address
            mg = self._subtensor.metagraph(self.config.netuid)
            for uid_val in range(len(mg.hotkeys)):
                if mg.hotkeys[uid_val] == hk_ss58:
                    uid = uid_val
                    break
        except Exception as e:
            bt.logging.warning(f"Cannot resolve UID from metagraph: {e}")

        if uid is None:
            bt.logging.warning(f"Cannot auto-register EVM: UID not found in metagraph for {self.evm_addr}")
            return

        bt.logging.info(f"Auto-registering validator EVM → UID {uid} on MinerRegistry")
        try:
            self._miner_client.register_evm(
                uid,
                hotkey_seed=self._validator_private_key,
                netuid=self.config.netuid,
                private_key=self.evm_pk,
            )
        except Exception as e:
            self._evm_disabled = True
            bt.logging.warning(
                f"registerEvm({uid}) failed: {e} "
                f"Continuing without EVM (no on-chain reportOffline). "
                f"Pass --no-evm to silence."
            )

    def _ensure_validator_registry_registered(self):
        """Register on ValidatorRegistry for validator participation.

        The registry endpoint is the user/API proxy endpoint. Capacity-audit
        proof ingest is published through native axon metadata by the validator
        process itself and must not overwrite the proxy endpoint.
        """
        if self._evm_disabled:
            bt.logging.info("EVM disabled, skipping ValidatorRegistry registration")
            return
        try:
            from verallm.chain.validator_registry import ValidatorRegistryClient
            vr = ValidatorRegistryClient(self.config)
        except Exception as e:
            bt.logging.debug(f"ValidatorRegistry not configured, skipping registration: {e}")
            return

        try:
            if not vr.is_evm_registered(self.evm_addr):
                # Resolve UID from Substrate metagraph
                uid = None
                try:
                    hk_ss58 = self._wallet.hotkey.ss58_address
                    mg = self._subtensor.metagraph(self.config.netuid)
                    for uid_val in range(len(mg.hotkeys)):
                        if mg.hotkeys[uid_val] == hk_ss58:
                            uid = uid_val
                            break
                except Exception as e:
                    bt.logging.warning(f"ValidatorRegistry: UID lookup failed: {e}")

                if uid is None:
                    # Expected when this validator has no permit on the subnet.
                    bt.logging.info("ValidatorRegistry: UID not found, skipping")
                    return

                bt.logging.info(f"ValidatorRegistry: registerEvm → UID {uid}")
                vr.register_evm(
                    uid,
                    hotkey_seed=self._validator_private_key,
                    netuid=self.config.netuid,
                    private_key=self.evm_pk,
                )

            if not vr.is_active_validator(self.evm_addr):
                bt.logging.info(
                    "ValidatorRegistry: registering with empty endpoint "
                    "(no public proxy endpoint)"
                )
                vr.register("", private_key=self.evm_pk)
            else:
                bt.logging.debug("ValidatorRegistry: already registered")
        except Exception as e:
            # Expected when there's no validator permit yet — message says
            # "will continue", so this is informational, not a degraded state.
            bt.logging.info(
                f"ValidatorRegistry registration failed: {e}. "
                f"Validator will continue without ValidatorRegistry — "
                f"proxy endpoint discovery disabled until you have a validator permit."
            )

    @staticmethod
    def _is_public_audit_axon_ip(host: str) -> bool:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )

    @staticmethod
    def _is_loopback_audit_bind_host(host: str) -> bool:
        text = str(host or "").strip().lower()
        if text in {"localhost", "ip6-localhost"}:
            return True
        try:
            return ipaddress.ip_address(text).is_loopback
        except ValueError:
            return False

    def _capacity_audit_axon_external_endpoint(self) -> tuple[Optional[str], int]:
        port = int(getattr(self.config, "capacity_audit_ingest_port", 8091) or 8091)
        raw_endpoint = str(getattr(self.config, "capacity_audit_public_url", "") or "").strip()
        if not raw_endpoint:
            return None, port

        from neurons.capacity_audit_discovery import normalize_audit_endpoint

        endpoint = normalize_audit_endpoint(raw_endpoint)
        if not endpoint:
            bt.logging.warning(
                f"Capacity audit axon serve ignored invalid public URL {raw_endpoint!r}; "
                "falling back to auto-detected external IP"
            )
            return None, port

        parsed = urlparse(endpoint)
        if parsed.scheme != "http":
            bt.logging.warning(
                f"Capacity audit axon serve ignored non-HTTP public URL {raw_endpoint!r}; "
                "Bittensor axon metadata carries host/port only"
            )
            return None, port
        host = parsed.hostname or ""
        public_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return None, public_port
        if ValidatorNeuron._is_public_audit_axon_ip(host):
            return host, public_port
        bt.logging.warning(
            f"Capacity audit axon serve ignored non-public IP/domain {host!r}; "
            "falling back to auto-detected external IP"
        )
        return None, public_port

    def _ensure_capacity_audit_axon_served(self) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        if not bool(getattr(self.config, "capacity_audit_serve_axon", True)):
            return
        try:
            bind_host = str(
                getattr(self.config, "capacity_audit_ingest_host", "0.0.0.0") or "0.0.0.0"
            )
            public_url = str(getattr(self.config, "capacity_audit_public_url", "") or "").strip()
            if ValidatorNeuron._is_loopback_audit_bind_host(bind_host) and not public_url:
                bt.logging.warning(
                    "Capacity audit axon metadata publish skipped: ingest is bound to "
                    f"{bind_host!r} and no public URL/front door is configured"
                )
                return
            external_ip, external_port = self._capacity_audit_axon_external_endpoint()
            AxonCls = getattr(bt, "Axon", None) or getattr(bt, "axon")
            axon = AxonCls(
                wallet=self._wallet,
                port=int(getattr(self.config, "capacity_audit_ingest_port", 8091) or 8091),
                ip=bind_host,
                external_ip=external_ip,
                external_port=external_port,
            )
            response = self._subtensor.serve_axon(
                netuid=self.config.netuid,
                axon=axon,
                raise_error=False,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if getattr(response, "success", True):
                served_ip = external_ip or getattr(response, "data", {}).get("external_ip", None)
                bt.logging.info(
                    "Capacity audit ingest published via Bittensor axon metadata: "
                    f"{served_ip or 'auto'}:{external_port}"
                )
            else:
                bt.logging.warning(
                    "Capacity audit axon metadata publish failed: "
                    f"{getattr(response, 'message', response)}"
                )
        except Exception as exc:
            bt.logging.warning(f"Capacity audit axon metadata publish skipped: {exc}")

    def _resolve_burn_uid(self) -> int:
        """Resolve subnet owner hotkey to UID on this subnet (= burn target).

        Called once at startup — the subnet owner is always registered and
        the owner UID never changes at runtime.
        """
        owner_hotkey = self._subtensor.get_subnet_owner_hotkey(self.config.netuid)
        burn_uid = self._subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=owner_hotkey, netuid=self.config.netuid,
        )
        bt.logging.info(f"Burn UID resolved: {burn_uid} (subnet owner: {owner_hotkey})")
        return burn_uid

    @staticmethod
    def _coerce_block_hash(raw: object) -> Optional[bytes]:
        """Normalize a substrate block hash to 32 raw bytes."""
        if isinstance(raw, bytes):
            return raw if len(raw) == 32 else None
        if raw is None:
            return None
        if hasattr(raw, "hex") and not isinstance(raw, str):
            try:
                raw = raw.hex()
            except Exception:
                return None
        text = str(raw).strip()
        if text.startswith("0x"):
            text = text[2:]
        if len(text) != 64:
            return None
        try:
            return bytes.fromhex(text)
        except ValueError:
            return None

    @staticmethod
    def _synthetic_block_hash(block_number: int) -> bytes:
        """Compatibility hash for legacy block processing when RPC hash lookup fails."""
        return hashlib.sha256(f"block_{int(block_number)}".encode()).digest()

    def _get_chain_block_hash(
        self,
        block_number: int,
        subtensor_obj: object | None = None,
    ) -> Tuple[bytes, bool]:
        """Return (block_hash, is_real_hash) for an explicit chain block number."""
        targets: List[object] = []
        if subtensor_obj is not None:
            targets.append(subtensor_obj)
        else:
            targets.append(self._subtensor)

        substrate = getattr(targets[0], "substrate", None) if targets else None
        if substrate is not None:
            try:
                response = substrate.rpc_request("chain_getBlockHash", [int(block_number)])
                raw_hash = response.get("result") if isinstance(response, dict) else response
                normalized = ValidatorNeuron._coerce_block_hash(raw_hash)
                if normalized is not None:
                    return normalized, True
            except Exception:
                pass
            targets.append(substrate)

        for target in targets:
            method = getattr(target, "get_block_hash", None)
            if not callable(method):
                continue
            call_specs = (
                ((), {"block": int(block_number)}),
                ((), {"block_id": int(block_number)}),
                ((int(block_number),), {}),
            )
            for args, kwargs in call_specs:
                try:
                    normalized = ValidatorNeuron._coerce_block_hash(method(*args, **kwargs))
                except TypeError:
                    continue
                except Exception:
                    continue
                if normalized is not None:
                    return normalized, True

        now = time.time()
        if now - self._last_block_hash_warning_at > 300:
            self._last_block_hash_warning_at = now
            msg = (
                f"Could not resolve chain hash for block {block_number}; "
                "using synthetic compatibility hash for legacy canary processing"
            )
            if getattr(self.config, "capacity_audit_enabled", False):
                msg += " and skipping capacity-audit scheduling from this block"
            bt.logging.warning(msg)
        return ValidatorNeuron._synthetic_block_hash(block_number), False

    @staticmethod
    def _block_hash_hex(block_hash: bytes) -> str:
        return bytes(block_hash).hex()

    def _capacity_slot_group_key(self, miner: ActiveMiner) -> str:
        return build_capacity_slot_group_key(
            address=miner.address,
            endpoint=getattr(miner, "endpoint", "") or "",
            model_id=getattr(miner, "model_id", "") or "",
            gpu_name=getattr(miner, "gpu_name", "") or "",
            miner_uid=self._db.get_uid(miner.address),
        )

    def _capacity_audit_seed_hashes(self, selection_block: int, selection_block_hash: bytes) -> Optional[list[bytes]]:
        count = max(1, int(self._capacity_audit_cfg.beacon_hash_count or 1))
        hashes = [selection_block_hash]
        if count <= 1:
            return hashes
        for offset in range(1, count):
            block = int(selection_block) - offset
            if block < 0:
                bt.logging.info(
                    f"Capacity audit: skipping B_select={selection_block} because "
                    f"beacon hash offset={offset}/{count - 1} is before genesis"
                )
                return None
            block_hash, real = self._get_chain_block_hash(block)
            if not real:
                bt.logging.info(
                    f"Capacity audit: skipping B_select={selection_block} because "
                    f"beacon hash block={block} offset={offset}/{count - 1} is unavailable"
                )
                return None
            hashes.append(block_hash)
        return hashes

    def _discover_capacity_audit_miners(self) -> list[ActiveMiner]:
        try:
            miners = discover_active_miners(self._miner_client, self._model_client)
        except Exception as exc:
            bt.logging.warning(f"Capacity audit discovery failed, using cached epoch miners: {exc}")
            return list(getattr(self, "_epoch_miners", []) or [])
        if not miners:
            return list(getattr(self, "_epoch_miners", []) or [])
        max_workers = min(32, max(1, len(miners)))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(self._fetch_miner_hardware, m) for m in miners]
        try:
            try:
                for future in as_completed(futures, timeout=20):
                    try:
                        future.result()
                    except Exception:
                        pass
            except _FuturesTimeout:
                stalled = sum(1 for future in futures if not future.done())
                if stalled:
                    bt.logging.warning(
                        f"Capacity audit hardware refresh timed out with {stalled} pending endpoint(s)"
                    )
            except Exception:
                pass
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        return miners

    def _store_capacity_audit_slot_snapshot(
        self,
        active: list[tuple[CapacitySlot, object]],
        *,
        block_number: int,
        source: str,
    ) -> None:
        block = int(block_number or self._last_known_block or self._epoch_start_block or 0)
        with self._capacity_audit_slot_snapshot_lock:
            self._capacity_audit_slot_snapshot = list(active)
            self._capacity_audit_slot_snapshot_block = block
            self._capacity_audit_slot_snapshot_updated_at = time.time()
            self._capacity_audit_slot_snapshot_last_error = ""
        bt.logging.info(
            f"Capacity audit slot snapshot refreshed: slots={len(active)} "
            f"block={block} source={source}"
        )

    def _refresh_capacity_audit_slot_snapshot_from_miners(
        self,
        miners: list[ActiveMiner],
        *,
        block_number: int,
        source: str,
    ) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        self._hydrate_capacity_audit_hardware_from_cache(miners)
        self._store_capacity_audit_slot_snapshot(
            [(slot, None) for slot in self._capacity_audit_selection_slots(miners)],
            block_number=block_number,
            source=source,
        )

    def _hydrate_capacity_audit_hardware_from_cache(
        self,
        miners: list[ActiveMiner],
    ) -> None:
        """Fill missing transient /health metadata from matching cached rows."""
        missing = [
            miner for miner in miners
            if not (getattr(miner, "gpu_name", "") or "")
            or int(getattr(miner, "gpu_count", 0) or 0) <= 0
            or int(getattr(miner, "vram_gb", 0) or 0) <= 0
        ]
        if not missing:
            return
        try:
            rows = self._db.get_active_entries()
        except Exception:
            return
        by_key = {
            (str(row.get("address", "")).lower(), int(row.get("model_index", 0) or 0)): row
            for row in rows
        }
        for miner in missing:
            key = (str(getattr(miner, "address", "") or "").lower(), int(getattr(miner, "model_index", 0) or 0))
            row = by_key.get(key)
            if not row:
                continue
            if str(row.get("endpoint") or "") != str(getattr(miner, "endpoint", "") or ""):
                continue
            if str(row.get("model_id") or "") != str(getattr(miner, "model_id", "") or ""):
                continue
            gpu_name = str(row.get("gpu_name") or "")
            gpu_count = int(row.get("gpu_count") or 0)
            vram_gb = int(row.get("vram_gb") or 0)
            if not gpu_name or gpu_count <= 0 or vram_gb <= 0:
                continue
            miner.gpu_name = gpu_name
            miner.gpu_count = gpu_count
            miner.vram_gb = vram_gb
            miner.compute_capability = str(row.get("compute_capability") or "")
            try:
                uuids = json.loads(row.get("gpu_uuids") or "[]")
            except Exception:
                uuids = []
            miner.gpu_uuids = uuids if isinstance(uuids, list) else []

    def _request_capacity_audit_slot_snapshot_refresh(
        self,
        *,
        block_number: int,
        force: bool = False,
        reason: str = "periodic",
    ) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        block = int(block_number or self._last_known_block or 0)
        refresh_blocks = int(getattr(self.config, "capacity_audit_slot_refresh_blocks", 60) or 0)
        if refresh_blocks <= 0 and not force:
            return
        with self._capacity_audit_slot_snapshot_lock:
            if self._capacity_audit_slot_snapshot_refreshing:
                return
            last = int(self._capacity_audit_slot_snapshot_block or 0)
            if not force and refresh_blocks > 0 and last > 0 and block - last < refresh_blocks:
                return
            self._capacity_audit_slot_snapshot_refreshing = True

        def _run() -> None:
            try:
                miners = self._discover_capacity_audit_miners()
                active = self._capacity_audit_active_slots(miners)
                self._store_capacity_audit_slot_snapshot(
                    active,
                    block_number=block,
                    source=f"async:{reason}",
                )
            except Exception as exc:
                with self._capacity_audit_slot_snapshot_lock:
                    self._capacity_audit_slot_snapshot_last_error = str(exc)
                bt.logging.warning(f"Capacity audit slot snapshot refresh failed: {exc}")
            finally:
                with self._capacity_audit_slot_snapshot_lock:
                    self._capacity_audit_slot_snapshot_refreshing = False

        self._capacity_audit_discovery_executor.submit(_run)

    def _capacity_audit_slot_snapshot_for_selection(
        self,
        selection_block: int,
    ) -> list[tuple[CapacitySlot, object]]:
        block = int(selection_block)
        stale_blocks = int(getattr(self.config, "capacity_audit_slot_snapshot_stale_blocks", 120) or 0)
        with self._capacity_audit_slot_snapshot_lock:
            active = list(self._capacity_audit_slot_snapshot)
            snapshot_block = int(self._capacity_audit_slot_snapshot_block or 0)
            refreshing = bool(self._capacity_audit_slot_snapshot_refreshing)
            last_error = self._capacity_audit_slot_snapshot_last_error
        if not active:
            self._request_capacity_audit_slot_snapshot_refresh(
                block_number=block,
                force=False,
                reason="empty",
            )
            suffix = f" error={last_error}" if last_error else ""
            bt.logging.info(
                f"Capacity audit: no cached eligible endpoint slots at block {block}; "
                f"refreshing={refreshing}{suffix}"
            )
            return []
        if stale_blocks > 0 and snapshot_block > 0 and block - snapshot_block > stale_blocks:
            self._request_capacity_audit_slot_snapshot_refresh(
                block_number=block,
                force=False,
                reason="stale",
            )
            bt.logging.info(
                f"Capacity audit: cached eligible slot snapshot stale at block {block} "
                f"(snapshot_block={snapshot_block}); skipping this window"
            )
            return []
        return active

    def _capacity_audit_active_slots(
        self,
        miners: Optional[list[ActiveMiner]] = None,
    ) -> list[tuple[CapacitySlot, object]]:
        slots: list[tuple[CapacitySlot, object]] = []
        now = time.time()
        for miner in list(miners if miners is not None else (getattr(self, "_epoch_miners", []) or [])):
            registered_at = int(getattr(miner, "registered_at", 0) or 0)
            min_age = float(self._capacity_audit_cfg.min_registration_age_s or 0.0)
            if registered_at > 0 and min_age > 0 and now - registered_at < min_age:
                continue
            gpu_row = match_gpu_class(
                getattr(miner, "gpu_name", "") or "",
                int(getattr(miner, "vram_gb", 0) or 0),
                self._capacity_audit_cfg,
            )
            if gpu_row is None or not gpu_row.calibrated or capacity_gpu_pass_count(gpu_row) <= 0:
                continue
            slots.append((
                CapacitySlot(
                    chain_id=int(getattr(self.config, "chain_id", 0) or 0),
                    netuid=int(self.config.netuid),
                    address=miner.address,
                    model_index=int(miner.model_index),
                    endpoint=miner.endpoint,
                    model_id=miner.model_id,
                    quant=miner.quant,
                    max_context_len=int(miner.max_context_len or 0),
                    miner_uid=self._db.get_uid(miner.address),
                    gpu_name=getattr(miner, "gpu_name", "") or "",
                    gpu_count=int(getattr(miner, "gpu_count", 0) or 0),
                    vram_gb=int(getattr(miner, "vram_gb", 0) or 0),
                    group_key=self._capacity_slot_group_key(miner),
                ),
                gpu_row,
            ))
        return slots

    def _capacity_audit_selection_slots(
        self,
        miners: Optional[list[ActiveMiner]] = None,
    ) -> list[CapacitySlot]:
        slots: list[CapacitySlot] = []
        now = time.time()
        for miner in list(miners if miners is not None else (getattr(self, "_epoch_miners", []) or [])):
            registered_at = int(getattr(miner, "registered_at", 0) or 0)
            min_age = float(self._capacity_audit_cfg.min_registration_age_s or 0.0)
            if registered_at > 0 and min_age > 0 and now - registered_at < min_age:
                continue
            slots.append(CapacitySlot(
                chain_id=int(getattr(self.config, "chain_id", 0) or 0),
                netuid=int(self.config.netuid),
                address=miner.address,
                model_index=int(miner.model_index),
                endpoint=miner.endpoint,
                model_id=miner.model_id,
                quant=miner.quant,
                max_context_len=int(miner.max_context_len or 0),
                group_key=build_capacity_slot_group_key(
                    address=miner.address,
                    endpoint=getattr(miner, "endpoint", "") or "",
                    model_id=getattr(miner, "model_id", "") or "",
                ),
            ))
        return slots

    def _capacity_audit_supported_slots_by_id(
        self,
        selected_slots: list[CapacitySlot],
        active_snapshot: Optional[list[tuple[CapacitySlot, object]]] = None,
    ) -> dict[str, tuple[CapacitySlot, object]]:
        if not selected_slots:
            return {}
        selected_ids = {slot_id(slot) for slot in selected_slots}
        supported: dict[str, tuple[CapacitySlot, object]] = {}
        for slot, gpu_row in list(active_snapshot or []):
            sid = slot_id(slot)
            if sid not in selected_ids or gpu_row is None:
                continue
            if capacity_gpu_pass_count(gpu_row) <= 0:
                continue
            supported[sid] = (slot, gpu_row)
        try:
            rows = self._db.get_active_entries()
        except Exception:
            rows = []
        by_key = {
            (str(row.get("address", "")).lower(), int(row.get("model_index", 0) or 0)): row
            for row in rows
        }
        for slot in selected_slots:
            sid = slot_id(slot)
            if sid in supported:
                continue
            row = by_key.get((slot.address_lower, int(slot.model_index)))
            if not row:
                continue
            if str(row.get("endpoint") or "") != str(slot.endpoint or ""):
                continue
            if str(row.get("model_id") or "") != str(slot.model_id or ""):
                continue
            if str(row.get("quant") or "") != str(slot.quant or ""):
                continue
            if int(row.get("max_context_len") or 0) != int(slot.max_context_len or 0):
                continue
            gpu_name = str(row.get("gpu_name") or "")
            gpu_count = int(row.get("gpu_count") or 0)
            vram_gb = int(row.get("vram_gb") or 0)
            gpu_row = match_gpu_class(gpu_name, vram_gb, self._capacity_audit_cfg)
            if gpu_row is None or not gpu_row.calibrated or capacity_gpu_pass_count(gpu_row) <= 0:
                resolved = self._capacity_audit_resolve_selected_slot_hardware(slot)
                if resolved is not None:
                    supported[sid] = resolved
                continue
            get_uid = getattr(self._db, "get_uid", None)
            try:
                miner_uid = get_uid(slot.address) if callable(get_uid) else slot.miner_uid
            except Exception:
                miner_uid = slot.miner_uid
            supported_slot = CapacitySlot(
                chain_id=slot.chain_id,
                netuid=slot.netuid,
                address=slot.address,
                model_index=slot.model_index,
                endpoint=slot.endpoint,
                model_id=slot.model_id,
                quant=slot.quant,
                max_context_len=slot.max_context_len,
                miner_uid=miner_uid,
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_gb=vram_gb,
                group_key=build_capacity_slot_group_key(
                    address=slot.address,
                    endpoint=slot.endpoint,
                    model_id=slot.model_id,
                    gpu_name=gpu_name,
                    miner_uid=miner_uid,
                ),
            )
            supported[sid] = (supported_slot, gpu_row)
        return supported

    def _capacity_audit_resolve_selected_slot_hardware(
        self,
        slot: CapacitySlot,
    ) -> Optional[tuple[CapacitySlot, object]]:
        """Resolve missing hardware for one already-selected slot.

        Selection remains public and scalable. This fallback only touches slots
        that already passed the deterministic B_select predicate and per-window
        budget, so it does not reintroduce network-wide health fanout.
        """
        miner = ActiveMiner(
            address=slot.address,
            model_id=slot.model_id,
            endpoint=slot.endpoint,
            quant=slot.quant,
            max_context_len=int(slot.max_context_len or 0),
            model_index=int(slot.model_index),
        )
        try:
            valid = self._fetch_miner_hardware(miner)
        except Exception as exc:
            bt.logging.debug(
                f"Capacity audit selected-slot hardware fetch failed for "
                f"{slot.address_lower[:10]} idx={slot.model_index}: {exc}"
            )
            return None
        if valid is False:
            bt.logging.info(
                f"Capacity audit selected slot has invalid hardware metadata: "
                f"{slot.address_lower[:10]} idx={slot.model_index}"
            )
            return None

        gpu_name = str(getattr(miner, "gpu_name", "") or "")
        gpu_count = int(getattr(miner, "gpu_count", 0) or 0)
        vram_gb = int(getattr(miner, "vram_gb", 0) or 0)
        gpu_row = match_gpu_class(gpu_name, vram_gb, self._capacity_audit_cfg)
        if gpu_row is None or not gpu_row.calibrated or capacity_gpu_pass_count(gpu_row) <= 0:
            return None

        upsert = getattr(self._db, "upsert_entry", None)
        if callable(upsert):
            try:
                upsert(
                    address=slot.address,
                    model_index=int(slot.model_index),
                    model_id=slot.model_id,
                    endpoint=slot.endpoint,
                    quant=slot.quant,
                    max_context_len=int(slot.max_context_len or 0),
                    epoch=int(getattr(self, "_current_epoch", 0) or 0),
                    gpu_name=gpu_name,
                    gpu_count=gpu_count,
                    vram_gb=vram_gb,
                    compute_capability=str(getattr(miner, "compute_capability", "") or ""),
                    gpu_uuids=list(getattr(miner, "gpu_uuids", []) or []),
                )
            except Exception as exc:
                bt.logging.debug(
                    f"Capacity audit selected-slot hardware cache update failed for "
                    f"{slot.address_lower[:10]} idx={slot.model_index}: {exc}"
                )

        get_uid = getattr(self._db, "get_uid", None)
        try:
            miner_uid = get_uid(slot.address) if callable(get_uid) else slot.miner_uid
        except Exception:
            miner_uid = slot.miner_uid
        supported_slot = CapacitySlot(
            chain_id=slot.chain_id,
            netuid=slot.netuid,
            address=slot.address,
            model_index=slot.model_index,
            endpoint=slot.endpoint,
            model_id=slot.model_id,
            quant=slot.quant,
            max_context_len=slot.max_context_len,
            miner_uid=miner_uid,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_gb=vram_gb,
            group_key=build_capacity_slot_group_key(
                address=slot.address,
                endpoint=slot.endpoint,
                model_id=slot.model_id,
                gpu_name=gpu_name,
                miner_uid=miner_uid,
            ),
        )
        bt.logging.info(
            f"Capacity audit resolved selected-slot hardware: "
            f"{slot.address_lower[:10]} idx={slot.model_index} gpu={gpu_name} vram_gb={vram_gb}"
        )
        return supported_slot, gpu_row

    def _capacity_audit_start_recoverable(self, audit_block: int, current_block: int) -> bool:
        """Return True while a late scheduler can still fairly judge timing."""
        if int(current_block or 0) <= 0:
            return True
        cfg = self._capacity_audit_cfg
        timing_window_s = float(cfg.deadline_s) + float(cfg.transport_grace_s)
        recoverable_blocks = max(1, int((timing_window_s + 11.999) // 12.0))
        return int(current_block) <= int(audit_block) + recoverable_blocks

    def _capacity_audit_selection_recoverable(self, audit_block: int) -> bool:
        last_seen_block = int(getattr(self, "_last_known_block", 0) or 0)
        return ValidatorNeuron._capacity_audit_start_recoverable(self, audit_block, last_seen_block)

    def _schedule_capacity_audit_window(
        self,
        *,
        selection_block: int,
        selection_block_hash: bytes,
    ) -> None:
        schedule_lock = getattr(self, "_capacity_audit_schedule_lock", None)
        if schedule_lock is None:
            return self._schedule_capacity_audit_window_locked(
                selection_block=selection_block,
                selection_block_hash=selection_block_hash,
            )
        with schedule_lock:
            return self._schedule_capacity_audit_window_locked(
                selection_block=selection_block,
                selection_block_hash=selection_block_hash,
            )

    def _schedule_capacity_audit_window_locked(
        self,
        *,
        selection_block: int,
        selection_block_hash: bytes,
    ) -> None:
        cfg = self._capacity_audit_cfg
        if not cfg.enabled:
            return

        epoch_number = int(selection_block // self.config.epoch_blocks)
        audit_block = int(selection_block + cfg.lead_blocks)
        proof_challenge_block = int(
            audit_block + max(1, int(cfg.proof_challenge_delay_blocks or 1))
        )
        if not capacity_audit_window_fits_epoch(
            selection_block,
            int(self.config.epoch_blocks),
            cfg,
        ):
            epoch_end = (epoch_number + 1) * int(self.config.epoch_blocks)
            bt.logging.info(
                f"Capacity audit: skipping late window B_select={selection_block} "
                f"B_start={audit_block} B_proof={proof_challenge_block} "
                f"epoch_end={epoch_end}"
            )
            return
        now = time.time()
        active = self._capacity_audit_slot_snapshot_for_selection(selection_block)
        drained_slots = {
            (drain.address.lower(), int(drain.model_index))
            for drain in self._db.get_capacity_drains(now=now)
        }
        if drained_slots:
            before = len(active)
            active = [
                (slot, row)
                for slot, row in active
                if (slot.address_lower, int(slot.model_index)) not in drained_slots
            ]
            if before != len(active):
                bt.logging.info(
                    f"Capacity audit: skipped {before - len(active)} already-drained "
                    f"slot(s) at block {selection_block}"
                )
        try:
            busy_slots = set(
                self._db.get_capacity_audit_selection_busy_slots(
                    selection_block=int(selection_block),
                    cooldown_blocks=1,
                )
            )
        except Exception as exc:
            bt.logging.debug(f"Capacity audit busy-slot lookup failed: {exc}")
            busy_slots = set()
        if busy_slots:
            before = len(active)
            active = [
                (slot, row)
                for slot, row in active
                if (slot.address_lower, int(slot.model_index)) not in busy_slots
            ]
            if before != len(active):
                bt.logging.info(
                    f"Capacity audit: skipped {before - len(active)} busy-overlap "
                    f"slot(s) at block {selection_block}"
                )
        if not active:
            bt.logging.info(
                f"Capacity audit: no active endpoint slots at block {selection_block}"
            )
            return

        seed_hashes = self._capacity_audit_seed_hashes(selection_block, selection_block_hash)
        if seed_hashes is None:
            return
        if len(seed_hashes) <= 1:
            cohort_seed = derive_audit_seed(selection_block_hash, epoch_number)
        else:
            cohort_seed = derive_audit_seed_from_hashes(seed_hashes, epoch_number)
        selected = select_capacity_audit_slots(
            [slot for slot, _row in active],
            cohort_seed,
            cfg,
        )
        if not selected:
            return
        budget = window_cohort_budget(len(active), cfg)
        if budget > 0 and len(selected) > budget:
            before = len(selected)
            selected = deterministic_sample_slots(selected, cohort_seed, budget)
            bt.logging.info(
                f"Capacity audit: truncated selected slots {before}->{len(selected)} "
                f"by per-window drain budget at block {selection_block}"
            )
        supported_fn = getattr(self, "_capacity_audit_supported_slots_by_id", None)
        if not callable(supported_fn):
            supported_fn = ValidatorNeuron._capacity_audit_supported_slots_by_id.__get__(self)
        supported_selected = supported_fn(selected, active)

        if not self._capacity_audit_selection_recoverable(audit_block):
            last_seen_block = int(getattr(self, "_last_known_block", 0) or 0)
            bt.logging.info(
                f"Capacity audit: skipping unrecoverably late window B_select={selection_block} "
                f"B_start={audit_block} current_block={last_seen_block}"
            )
            return
        audit_id = derive_audit_id(
            chain_id=int(getattr(self.config, "chain_id", 0) or 0),
            netuid=int(self.config.netuid),
            epoch_number=epoch_number,
            selection_block=int(selection_block),
            audit_block=audit_block,
            cohort_seed=cohort_seed,
        )
        drain_until_ts = (
            now
            + cfg.drain_seconds
            + cfg.deadline_s
            + cfg.transport_grace_s
            + cfg.payload_deadline_s
        )
        rows: list[dict] = []
        unsupported_selected = 0
        for selected_slot in selected:
            sid = slot_id(selected_slot)
            supported = supported_selected.get(sid)
            if supported is None:
                unsupported_selected += 1
                continue
            slot, gpu_row = supported
            rows.append({
                "miner_address": slot.address_lower,
                "model_index": slot.model_index,
                "miner_uid": slot.miner_uid,
                "endpoint": slot.endpoint,
                "model_id": slot.model_id,
                "quant": slot.quant,
                "max_context_len": slot.max_context_len,
                "gpu_name": slot.gpu_name,
                "gpu_count": slot.gpu_count,
                "vram_gb": slot.vram_gb,
                "group_key": slot.group_key,
                "slot_id": sid,
                "lease_id": lease_id(slot, epoch_number),
                "claimed_gpu_class": gpu_row.match_gpu_name,
                "pass_count": capacity_gpu_pass_count(gpu_row),
                "deadline_s": gpu_row.deadline_s or cfg.deadline_s,
                "transport_grace_s": cfg.transport_grace_s,
                "payload_deadline_s": cfg.payload_deadline_s,
                "drain_until_ts": drain_until_ts,
            })
        if not rows:
            bt.logging.info(
                f"Capacity audit: public cohort selected no calibrated slots at block {selection_block} "
                f"(selected={len(selected)} unsupported_or_unknown={unsupported_selected})"
            )
            return

        self._db.create_capacity_audit_window(
            audit_id=audit_id,
            epoch_number=epoch_number,
            selection_block=int(selection_block),
            audit_block=audit_block,
            proof_challenge_block=proof_challenge_block,
            selection_block_hash=self._block_hash_hex(selection_block_hash),
            cohort_seed=cohort_seed,
            slots=rows,
        )
        self._write_shared_state()
        bt.logging.info(
            f"Capacity audit scheduled: audit_id={audit_id[:12]} "
            f"epoch={epoch_number} B_select={selection_block} B_start={audit_block} "
            f"B_proof={proof_challenge_block} "
            f"selected={len(rows)}/{len(active)} "
            f"unsupported_or_unknown={unsupported_selected} mode={cfg.mode}"
        )

    def _start_capacity_audit_windows(
        self,
        *,
        audit_block: int,
        audit_block_hash: bytes | None = None,
        audit_block_hash_real: bool = True,
    ) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        windows = self._db.get_capacity_audit_windows_for_start(audit_block)
        if not windows:
            return
        observed_at = time.time()
        for window in windows:
            window_audit_block = int(window["audit_block"])
            current_head = max(
                int(audit_block),
                int(getattr(self, "_last_known_block", 0) or 0),
            )
            strict_timing_mode = str(getattr(self._capacity_audit_cfg, "mode", "observe") or "observe") in {
                "score_gate",
                "enforce",
            }
            if strict_timing_mode and current_head > window_audit_block:
                reason = (
                    "validator_start_missed"
                    if not self._capacity_audit_start_recoverable(window_audit_block, current_head)
                    else "validator_start_replayed"
                )
                stale = self._db.mark_capacity_audit_window_stale(
                    window["audit_id"],
                    reason=reason,
                    released_at=observed_at,
                )
                if stale:
                    self._write_shared_state()
                bt.logging.warning(
                    f"Capacity audit stale window skipped: audit_id={window['audit_id'][:12]} "
                    f"B_start={window_audit_block} current_head={current_head} "
                    f"released_slots={stale}"
                )
                continue
            if not self._capacity_audit_start_recoverable(window_audit_block, current_head):
                stale = self._db.mark_capacity_audit_window_stale(
                    window["audit_id"],
                    reason="validator_start_missed",
                    released_at=observed_at,
                )
                if stale:
                    self._write_shared_state()
                bt.logging.warning(
                    f"Capacity audit stale window skipped: audit_id={window['audit_id'][:12]} "
                    f"B_start={window_audit_block} current_block={audit_block} "
                    f"released_slots={stale}"
                )
                continue
            if (
                window_audit_block == int(audit_block)
                and audit_block_hash is not None
                and audit_block_hash_real
            ):
                start_block_hash = audit_block_hash
            else:
                start_block_hash, real = self._get_chain_block_hash(window_audit_block)
                if not real:
                    bt.logging.info(
                        f"Capacity audit start catch-up waiting for real B_start hash: "
                        f"audit_id={window['audit_id'][:12]} B_start={window_audit_block} "
                        f"current_block={audit_block}"
                    )
                    continue
            self._db.set_capacity_audit_block_hash(
                window["audit_id"],
                self._block_hash_hex(start_block_hash),
                observed_at=observed_at,
            )
            resolved = self._db.resolve_capacity_audit_pending_start(
                window["audit_id"],
                observed_at=observed_at,
            )
            released = self._db.release_capacity_audit_completed_drains(
                window["audit_id"],
                require_proof_payload=self._capacity_audit_cfg.require_proof_payload,
                released_at=observed_at,
            )
            if resolved or released:
                self._write_shared_state()
            bt.logging.info(
                f"Capacity audit started: audit_id={window['audit_id'][:12]} "
                f"B_start={window_audit_block} current_block={audit_block} "
                f"pending_start_resolved={resolved} drains_released={released}"
            )

    def _record_capacity_audit_proof_challenges(
        self,
        *,
        block_number: int,
        block_hash: bytes | None = None,
        block_hash_real: bool = True,
    ) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        windows = self._db.get_capacity_audit_windows_for_proof_challenge(block_number)
        if not windows:
            return
        observed_at = time.time()
        updated = 0
        for window in windows:
            challenge_block = int(window.get("proof_challenge_block") or 0)
            if challenge_block <= 0:
                continue
            if (
                challenge_block == int(block_number)
                and block_hash is not None
                and block_hash_real
            ):
                challenge_hash = block_hash
            else:
                challenge_hash, real = self._get_chain_block_hash(challenge_block)
                if not real:
                    bt.logging.info(
                        f"Capacity audit proof challenge waiting for real block hash: "
                        f"audit_id={window['audit_id'][:12]} B_proof={challenge_block} "
                        f"current_block={block_number}"
                    )
                    continue
            self._db.set_capacity_audit_proof_challenge_hash(
                window["audit_id"],
                self._block_hash_hex(challenge_hash),
                observed_at=observed_at,
            )
            updated += 1
        if updated:
            bt.logging.info(f"Capacity audit: recorded {updated} proof challenge hashes")

    def _confirm_capacity_audit_finalized_blocks(self, subtensor_obj: object | None = None) -> None:
        """Confirm current-head audit hashes after they become finalized."""
        if not self._capacity_audit_cfg.enabled:
            return
        finalized_block, _head_hash, real = self._get_current_finalized_block_and_hash(subtensor_obj)
        if not real or finalized_block <= 0:
            return
        if finalized_block <= int(getattr(self, "_capacity_audit_last_finalized_confirmed", 0) or 0):
            return
        self._capacity_audit_last_finalized_confirmed = int(finalized_block)
        try:
            windows = self._db.get_capacity_audit_windows_for_finalization(finalized_block)
        except Exception as exc:
            bt.logging.debug(f"Capacity audit finalization lookup failed: {exc}")
            return
        if not windows:
            return

        confirmed = 0
        reorged = 0
        for window in windows:
            audit_id = str(window.get("audit_id") or "")
            if not audit_id:
                continue
            selection_ok: Optional[bool] = None
            audit_ok: Optional[bool] = None
            proof_ok: Optional[bool] = None

            if window.get("selection_finalized_at") is None:
                selection_block = int(window.get("selection_block") or 0)
                expected = str(window.get("selection_block_hash") or "")
                if selection_block > 0 and expected and selection_block <= finalized_block:
                    block_hash, hash_real = self._get_chain_block_hash(selection_block, subtensor_obj)
                    if hash_real:
                        selection_ok = self._block_hash_hex(block_hash) == expected

            if window.get("audit_finalized_at") is None:
                audit_block = int(window.get("audit_block") or 0)
                expected = str(window.get("audit_block_hash") or "")
                if audit_block > 0 and expected and audit_block <= finalized_block:
                    block_hash, hash_real = self._get_chain_block_hash(audit_block, subtensor_obj)
                    if hash_real:
                        audit_ok = self._block_hash_hex(block_hash) == expected

            if window.get("proof_challenge_finalized_at") is None:
                proof_block = int(window.get("proof_challenge_block") or 0)
                expected = str(window.get("proof_challenge_block_hash") or "")
                if proof_block > 0 and expected and proof_block <= finalized_block:
                    block_hash, hash_real = self._get_chain_block_hash(proof_block, subtensor_obj)
                    if hash_real:
                        proof_ok = self._block_hash_hex(block_hash) == expected

            if selection_ok is None and audit_ok is None and proof_ok is None:
                continue
            self._db.record_capacity_audit_finalization(
                audit_id,
                selection_confirmed=selection_ok,
                audit_confirmed=audit_ok,
                proof_confirmed=proof_ok,
            )
            if any(v is False for v in (selection_ok, audit_ok, proof_ok) if v is not None):
                reorged += 1
                bt.logging.warning(
                    f"Capacity audit finalized hash mismatch: audit_id={audit_id[:12]} "
                    f"B_finalized={finalized_block}"
                )
            else:
                confirmed += 1
        if confirmed or reorged:
            self._write_shared_state()
            bt.logging.info(
                f"Capacity audit finalized confirmation: confirmed_updates={confirmed} "
                f"reorged={reorged} finalized_block={finalized_block}"
            )

    def _recover_capacity_audit_hashes_for_artifact(self, row: dict, artifact: dict) -> dict:
        """Best-effort B_start/B_proof hash recovery for late audit artifacts."""
        if not self._capacity_audit_cfg.enabled:
            return row
        if self._capacity_audit_cfg.mode in ("score_gate", "enforce"):
            return row

        refreshed = False
        if not str(row.get("audit_block_hash") or ""):
            try:
                audit_block = int(row.get("audit_block") or artifact.get("B_start") or 0)
            except Exception:
                audit_block = 0
            if audit_block > 0:
                try:
                    audit_hash, real = self._get_chain_block_hash(audit_block)
                except Exception:
                    audit_hash, real = b"", False
                if real:
                    self._start_capacity_audit_windows(
                        audit_block=audit_block,
                        audit_block_hash=audit_hash,
                    )
                    refreshed = True

        if not str(row.get("proof_challenge_block_hash") or ""):
            try:
                proof_challenge_block = int(
                    row.get("proof_challenge_block") or artifact.get("B_proof") or 0
                )
            except Exception:
                proof_challenge_block = 0
            if proof_challenge_block > 0:
                try:
                    challenge_hash, real = self._get_chain_block_hash(proof_challenge_block)
                except Exception:
                    challenge_hash, real = b"", False
                if real:
                    self._record_capacity_audit_proof_challenges(
                        block_number=proof_challenge_block,
                        block_hash=challenge_hash,
                    )
                    refreshed = True

        if not refreshed:
            return row

        address = str(row.get("miner_address") or artifact.get("address") or "").lower()
        try:
            model_index = int(row.get("model_index") or artifact.get("model_index"))
        except Exception:
            model_index = -1
        audit_id = str(row.get("audit_id") or artifact.get("audit_id") or "")
        if not audit_id or not address or model_index < 0:
            return row
        updated = self._db.get_capacity_audit_slot(audit_id, address, model_index)
        return updated if updated is not None else row

    def _handle_capacity_audit_block(
        self,
        block_number: int,
        block_hash: bytes,
        *,
        block_hash_real: bool,
    ) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        expired = self._db.expire_capacity_audit_misses(
            require_proof_payload=self._capacity_audit_cfg.require_proof_payload,
            return_slots=True,
        )
        expired_slots = expired if isinstance(expired, list) else []
        expired_count = len(expired_slots) if isinstance(expired, list) else int(expired or 0)
        if expired_count:
            bt.logging.info(f"Capacity audit: expired {expired_count} pending slots")
            for slot in expired_slots:
                verdict = str(slot.get("verdict") or "")
                if verdict == "hard_proof_miss":
                    self._on_capacity_audit_failure(
                        str(slot.get("miner_address") or ""),
                        int(slot.get("model_index") or 0),
                        endpoint=str(slot.get("endpoint") or ""),
                        verdict=verdict,
                        failure_reason=str(slot.get("failure_reason") or ""),
                    )
                else:
                    bt.logging.info(
                        f"Capacity audit timing miss recorded pending receipt reconciliation: "
                        f"{str(slot.get('miner_address') or '')[:10]} "
                        f"model_index={int(slot.get('model_index') or 0)} "
                        f"verdict={verdict or 'unknown'}"
                    )
            self._write_shared_state()
        self._start_capacity_audit_windows(
            audit_block=block_number,
            audit_block_hash=block_hash,
            audit_block_hash_real=block_hash_real,
        )
        self._record_capacity_audit_proof_challenges(
            block_number=block_number,
            block_hash=block_hash,
            block_hash_real=block_hash_real,
        )
        if not block_hash_real:
            if block_number % self.config.epoch_blocks == 0:
                bt.logging.warning(
                    f"Capacity audit: skipping B_select={block_number} because block hash is synthetic"
                )
            return
        if capacity_audit_window_triggered(
            block_number,
            block_hash,
            int(self.config.epoch_blocks),
            self._capacity_audit_cfg,
        ):
            self._capacity_audit_executor.submit(
                self._schedule_capacity_audit_window,
                selection_block=block_number,
                selection_block_hash=block_hash,
            )

    def _capacity_audit_artifact_dir(self, audit_id: str) -> str:
        root = os.path.join(
            os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
            "capacity_audit",
            str(audit_id),
        )
        os.makedirs(root, exist_ok=True)
        return root

    def _recover_capacity_audit_window_from_artifact(self, artifact: dict) -> None:
        """Recover a deterministic audit window missed by the scheduler.

        A validator can transiently fail to read a chain block hash or fall
        behind the polling loop while miners continue deriving the same
        non-interactive window. On an otherwise unknown signed artifact, derive
        the window synchronously and let normal slot/audit_id validation decide
        whether the artifact belongs to the selected cohort.
        """
        if not self._capacity_audit_cfg.enabled:
            return
        if self._capacity_audit_cfg.mode in ("score_gate", "enforce"):
            return
        try:
            selection_block = int(artifact.get("B_select"))
            audit_block = int(artifact.get("B_start"))
        except Exception:
            return
        cfg = self._capacity_audit_cfg
        if audit_block != selection_block + int(cfg.lead_blocks):
            return
        if not capacity_audit_window_fits_epoch(
            selection_block,
            int(self.config.epoch_blocks),
            cfg,
        ):
            return
        last_seen_block = int(getattr(self, "_last_known_block", 0) or 0)
        if last_seen_block > 0 and selection_block > last_seen_block:
            return
        if not self._capacity_audit_selection_recoverable(audit_block):
            return
        selection_hash, real = self._get_chain_block_hash(selection_block)
        if not real:
            bt.logging.info(
                f"Capacity audit recovery skipped: real B_select hash unavailable "
                f"B_select={selection_block}"
            )
            return
        if not capacity_audit_window_triggered(
            selection_block,
            selection_hash,
            int(self.config.epoch_blocks),
            cfg,
        ):
            return
        self._schedule_capacity_audit_window(
            selection_block=selection_block,
            selection_block_hash=selection_hash,
        )
        if last_seen_block >= audit_block:
            audit_hash, audit_real = self._get_chain_block_hash(audit_block)
            if audit_real:
                self._start_capacity_audit_windows(
                    audit_block=audit_block,
                    audit_block_hash=audit_hash,
                )
        proof_challenge_block = int(
            audit_block + max(1, int(cfg.proof_challenge_delay_blocks or 1))
        )
        if last_seen_block >= proof_challenge_block:
            challenge_hash, challenge_real = self._get_chain_block_hash(proof_challenge_block)
            if challenge_real:
                self._record_capacity_audit_proof_challenges(
                    block_number=proof_challenge_block,
                    block_hash=challenge_hash,
                )

    def _validate_capacity_audit_artifact(self, artifact: dict) -> tuple[dict, Optional[str]]:
        audit_id = str(artifact.get("audit_id") or "")
        address = str(artifact.get("address") or artifact.get("miner_address") or "").lower()
        try:
            model_index = int(artifact.get("model_index"))
        except Exception:
            return {}, "invalid model_index"
        if not audit_id:
            return {}, "missing audit_id"
        if not address:
            return {}, "missing address"
        if artifact.get("protocol_version") not in (None, PROTOCOL_VERSION):
            return {}, "unsupported protocol_version"
        if not verify_artifact_signature(artifact, address):
            return {}, "invalid miner_signature"

        row = self._db.get_capacity_audit_slot(audit_id, address, model_index)
        if row is None:
            self._recover_capacity_audit_window_from_artifact(artifact)
            row = self._db.get_capacity_audit_slot(audit_id, address, model_index)
        if row is None:
            return {}, "unknown audit slot"
        if str(artifact.get("slot_id") or "") != str(row["slot_id"]):
            return {}, "slot_id mismatch"
        if int(artifact.get("B_select", row["selection_block"])) != int(row["selection_block"]):
            return {}, "B_select mismatch"
        if int(artifact.get("B_start", row["audit_block"])) != int(row["audit_block"]):
            return {}, "B_start mismatch"
        if int(row.get("proof_challenge_block") or 0) > 0:
            try:
                artifact_b_proof = artifact.get("B_proof", row["proof_challenge_block"])
                if int(artifact_b_proof) != int(row["proof_challenge_block"]):
                    return {}, "B_proof mismatch"
            except Exception:
                return {}, "B_proof mismatch"
        if int(artifact.get("pass_count", row["pass_count"]) or 0) != int(row["pass_count"]):
            return {}, "pass_count mismatch"
        claimed = str(artifact.get("claimed_gpu_class") or row["claimed_gpu_class"] or "")
        if claimed and str(row["claimed_gpu_class"]) and claimed != str(row["claimed_gpu_class"]):
            return {}, "claimed_gpu_class mismatch"
        row = self._recover_capacity_audit_hashes_for_artifact(row, artifact)
        return row, None

    def ingest_capacity_audit_artifact(
        self,
        artifact: dict,
        *,
        received_at: Optional[float] = None,
    ) -> tuple[int, dict]:
        """Ingest a miner-published capacity audit artifact."""
        if not self._capacity_audit_cfg.enabled:
            return 404, {"ok": False, "error": "capacity audit disabled"}
        if not isinstance(artifact, dict):
            return 400, {"ok": False, "error": "artifact must be an object"}
        artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "")
        row, error = self._validate_capacity_audit_artifact(artifact)
        if error:
            return 400, {"ok": False, "error": error}
        ts = time.time() if received_at is None else float(received_at)
        audit_id = str(row["audit_id"])
        address = str(row["miner_address"])
        model_index = int(row["model_index"])

        if artifact_type == "capacity_audit_pass0_receipt":
            pass0_root = str(artifact.get("pass0_root") or "")
            if not pass0_root:
                return 400, {"ok": False, "error": "missing pass0_root"}
            self._db.record_capacity_audit_pass0(
                audit_id=audit_id,
                address=address,
                model_index=model_index,
                pass0_root=pass0_root,
                artifact=artifact,
                received_at=ts,
            )
            return 200, {"ok": True, "verdict": "pass0_seen"}

        if artifact_type == "capacity_audit_final_receipt":
            final_root = str(artifact.get("final_root") or "")
            transcript = str(
                artifact.get("final_transcript_commit")
                or artifact.get("transcript_root")
                or ""
            )
            pass0_root = str(artifact.get("pass0_root") or "")
            if not final_root:
                return 400, {"ok": False, "error": "missing final_root"}
            if not transcript:
                return 400, {"ok": False, "error": "missing final_transcript_commit"}
            if row.get("pass0_root") and pass0_root and pass0_root != row.get("pass0_root"):
                self._db.record_capacity_audit_final(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    final_root=final_root,
                    transcript_root=transcript,
                    artifact=artifact,
                    timing_status="invalid_transcript",
                    verdict="hard_proof_miss",
                    failure_reason="pass0_root_mismatch",
                    received_at=ts,
                )
                self._on_capacity_audit_failure(
                    address,
                    model_index,
                    endpoint=str(row.get("endpoint") or ""),
                    verdict="hard_proof_miss",
                    failure_reason="pass0_root_mismatch",
                )
                return 200, {"ok": True, "verdict": "hard_proof_miss"}

            start_at = row.get("audit_start_observed_at")
            if start_at is None:
                timing_status = "pending_start"
                verdict = "pass0_seen"
                failure_reason = ""
            else:
                deadline = (
                    float(start_at)
                    + float(row.get("deadline_s") or self._capacity_audit_cfg.deadline_s)
                    + float(row.get("transport_grace_s") or self._capacity_audit_cfg.transport_grace_s)
                )
                if ts <= deadline:
                    timing_status = "pass"
                    verdict = "timing_pass"
                    failure_reason = ""
                else:
                    timing_status = "miss"
                    verdict = "timing_miss"
                    failure_reason = "deadline_exceeded"
            self._db.record_capacity_audit_final(
                audit_id=audit_id,
                address=address,
                model_index=model_index,
                final_root=final_root,
                transcript_root=transcript,
                artifact=artifact,
                timing_status=timing_status,
                verdict=verdict,
                failure_reason=failure_reason,
                received_at=ts,
            )
            if verdict == "timing_miss":
                bt.logging.info(
                    f"Capacity audit timing miss recorded pending receipt reconciliation: "
                    f"{address[:10]} model_index={model_index} audit_id={audit_id[:12]}"
                )
            if verdict == "timing_pass" and not self._capacity_audit_cfg.require_proof_payload:
                self._db.release_capacity_audit_drain(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    released_at=ts,
                )
            self._write_shared_state()
            return 200, {"ok": True, "verdict": verdict, "timing_status": timing_status}

        if artifact_type == "capacity_audit_proof_payload":
            if not row.get("transcript_root"):
                return 409, {"ok": False, "error": "final receipt required before proof payload"}
            self._db.record_capacity_audit_proof_received(
                audit_id=audit_id,
                address=address,
                model_index=model_index,
                received_at=ts,
            )

            def _record_hard_payload_miss(
                reason: str,
                proof_verify_ms: Optional[float] = None,
            ) -> tuple[int, dict]:
                self._db.record_capacity_audit_proof_verdict(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    proof_status="invalid_payload",
                    verdict="hard_proof_miss",
                    failure_reason=reason,
                    proof_verify_ms=proof_verify_ms,
                    received_at=ts,
                )
                self._on_capacity_audit_failure(
                    address,
                    model_index,
                    endpoint=str(row.get("endpoint") or ""),
                    verdict="hard_proof_miss",
                    failure_reason=reason,
                )
                self._write_shared_state()
                body = {"ok": True, "verdict": "hard_proof_miss", "proof_status": "invalid_payload"}
                if proof_verify_ms is not None:
                    body["proof_verify_ms"] = proof_verify_ms
                return 200, body

            proof = artifact.get("sampled_pass_proof")
            if isinstance(proof, dict) and str(proof.get("format") or "") == COMBINED_PROOF_FORMAT:
                proof_verify_ms: Optional[float] = None
                try:
                    gpu_index = int(artifact.get("gpu_index") or 0)
                except Exception:
                    gpu_index = 0
                proof_challenge_block_hash = str(row.get("proof_challenge_block_hash") or "")
                if not proof_challenge_block_hash:
                    proof_challenge_block = int(row.get("proof_challenge_block") or 0)
                    if proof_challenge_block <= 0:
                        return 409, {"ok": False, "error": "proof challenge block required"}
                    challenge_hash, real = self._get_chain_block_hash(proof_challenge_block)
                    if not real:
                        return 409, {"ok": False, "error": "proof challenge block hash required"}
                    proof_challenge_block_hash = self._block_hash_hex(challenge_hash)
                    self._db.set_capacity_audit_proof_challenge_hash(
                        audit_id,
                        proof_challenge_block_hash,
                        observed_at=ts,
                    )
                proof_challenge_seed = derive_proof_challenge_seed(
                    str(row["transcript_root"]),
                    proof_challenge_block_hash,
                    str(row["lease_id"]),
                    str(row["slot_id"]),
                    gpu_index,
                )
                if not str(row.get("audit_block_hash") or ""):
                    return 409, {"ok": False, "error": "audit block hash required"}
                proof_seed = derive_proof_seed(
                    str(row.get("audit_block_hash") or ""),
                    str(row.get("slot_id") or ""),
                    gpu_index,
                )
                final_artifact = {}
                try:
                    final_artifact = json.loads(str(row.get("final_artifact") or "{}"))
                except Exception:
                    final_artifact = {}
                verify_start = time.perf_counter()
                try:
                    ok, reason = verify_combined_proof_payload(
                        proof=proof,
                        final_artifact=final_artifact,
                        expected_combined_transcript_root=str(row["transcript_root"]),
                        lease_id=str(row["lease_id"]),
                        gpu_index=gpu_index,
                        proof_seed_hex=proof_seed,
                        proof_challenge_seed_hex=proof_challenge_seed,
                    )
                except Exception as exc:
                    proof_verify_ms = (time.perf_counter() - verify_start) * 1000.0
                    current_verdict = str(row.get("verdict") or "pending")
                    self._mark_capacity_audit_verifier_unhealthy(exc)
                    self._db.record_capacity_audit_proof_verdict(
                        audit_id=audit_id,
                        address=address,
                        model_index=model_index,
                        proof_status="verify_error",
                        verdict=current_verdict,
                        failure_reason="validator_verify_error",
                        proof_verify_ms=proof_verify_ms,
                        received_at=ts,
                    )
                    if current_verdict == "timing_pass":
                        self._db.release_capacity_audit_drain(
                            audit_id=audit_id,
                            address=address,
                            model_index=model_index,
                            released_at=ts,
                        )
                        self._write_shared_state()
                    bt.logging.warning(
                        f"Capacity audit proof verifier error: audit_id={audit_id[:12]} "
                        f"miner={address[:10]} model_index={model_index}: {exc}"
                    )
                    return 200, {
                        "ok": True,
                        "verdict": current_verdict,
                        "proof_status": "verify_error",
                        "proof_verify_ms": proof_verify_ms,
                    }
                proof_verify_ms = (time.perf_counter() - verify_start) * 1000.0
                if not ok:
                    return _record_hard_payload_miss(reason, proof_verify_ms)
                self._mark_capacity_audit_verifier_healthy()

                proof_path = os.path.join(
                    self._capacity_audit_artifact_dir(audit_id),
                    f"{address}_{model_index}_proof_payload.json",
                )
                with open(proof_path, "w") as f:
                    json.dump(artifact, f, sort_keys=True)
                current_verdict = str(row.get("verdict") or "pending")
                proof_status = "combined_proof_verified"
                self._db.record_capacity_audit_proof_verdict(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    proof_status=proof_status,
                    verdict=current_verdict,
                    proof_artifact_path=proof_path,
                    proof_verify_ms=proof_verify_ms,
                    received_at=ts,
                )
                if current_verdict == "timing_pass":
                    self._db.release_capacity_audit_drain(
                        audit_id=audit_id,
                        address=address,
                        model_index=model_index,
                        released_at=ts,
                    )
                    self._write_shared_state()
                return 200, {
                    "ok": True,
                    "verdict": current_verdict,
                    "proof_status": proof_status,
                    "proof_verify_ms": proof_verify_ms,
                }

            return _record_hard_payload_miss("unsupported_proof_payload_format")

        return 400, {"ok": False, "error": "unknown artifact_type"}

    def _prepare_capacity_audit_proof_enqueue(
        self,
        artifact: dict,
        *,
        received_at: float,
    ) -> tuple[int, dict, Optional[dict]]:
        if not self._capacity_audit_cfg.enabled:
            return 404, {"ok": False, "error": "capacity audit disabled"}, None
        if not isinstance(artifact, dict):
            return 400, {"ok": False, "error": "artifact must be an object"}, None
        artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "")
        if artifact_type != "capacity_audit_proof_payload":
            return 400, {"ok": False, "error": "unknown artifact_type"}, None

        row, error = self._validate_capacity_audit_artifact(artifact)
        if error:
            return 400, {"ok": False, "error": error}, None
        if not row or not row.get("transcript_root"):
            return 409, {"ok": False, "error": "final receipt required before proof payload"}, None

        audit_id = str(row["audit_id"])
        address = str(row["miner_address"])
        model_index = int(row["model_index"])
        if not str(row.get("proof_challenge_block_hash") or ""):
            proof_challenge_block = int(row.get("proof_challenge_block") or 0)
            if proof_challenge_block <= 0:
                return 409, {"ok": False, "error": "proof challenge block required"}, None
            challenge_hash, real = self._get_chain_block_hash(proof_challenge_block)
            if not real:
                return 409, {"ok": False, "error": "proof challenge block hash required"}, None
            self._db.set_capacity_audit_proof_challenge_hash(
                audit_id,
                self._block_hash_hex(challenge_hash),
                observed_at=received_at,
            )
        if not str(row.get("audit_block_hash") or ""):
            return 409, {"ok": False, "error": "audit block hash required"}, None

        self._db.record_capacity_audit_proof_received(
            audit_id=audit_id,
            address=address,
            model_index=model_index,
            received_at=received_at,
        )
        return 202, {"ok": True, "proof_status": "verify_pending"}, row

    def _run_capacity_audit_proof_verification(self, artifact: dict, received_at: float) -> None:
        try:
            self.ingest_capacity_audit_artifact(artifact, received_at=received_at)
        except Exception as exc:
            bt.logging.warning(f"Capacity audit proof verification worker failed: {exc}")
            try:
                row, error = self._validate_capacity_audit_artifact(artifact)
                if error or not row:
                    return
                current_verdict = str(row.get("verdict") or "pending")
                audit_id = str(row["audit_id"])
                address = str(row["miner_address"])
                model_index = int(row["model_index"])
                self._mark_capacity_audit_verifier_unhealthy(exc)
                self._db.record_capacity_audit_proof_verdict(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    proof_status="verify_error",
                    verdict=current_verdict,
                    failure_reason="validator_verify_error",
                    received_at=received_at,
                )
                if current_verdict == "timing_pass":
                    self._db.release_capacity_audit_drain(
                        audit_id=audit_id,
                        address=address,
                        model_index=model_index,
                        released_at=received_at,
                    )
                    self._write_shared_state()
            except Exception:
                pass

    def _mark_capacity_audit_verifier_unhealthy(self, exc: BaseException) -> None:
        self._capacity_audit_verifier_unhealthy = True
        self._capacity_audit_verifier_last_error = str(exc)

    def _mark_capacity_audit_verifier_healthy(self) -> None:
        if getattr(self, "_capacity_audit_verifier_unhealthy", False):
            bt.logging.info("Capacity audit proof verifier recovered after a successful verification")
        self._capacity_audit_verifier_unhealthy = False
        self._capacity_audit_verifier_last_error = ""

    def submit_capacity_audit_proof_artifact(
        self,
        artifact: dict,
        *,
        received_at: Optional[float] = None,
    ) -> tuple[int, dict]:
        ts = time.time() if received_at is None else float(received_at)
        status, body, _row = self._prepare_capacity_audit_proof_enqueue(
            artifact,
            received_at=ts,
        )
        if status >= 300:
            return status, body
        self._capacity_audit_proof_executor.submit(
            self._run_capacity_audit_proof_verification,
            artifact,
            ts,
        )
        return status, body

    def _build_capacity_audit_ingest_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        app = FastAPI(title="Verathos Capacity Audit Ingest")

        async def _read_payload(request: Request) -> tuple[int, dict]:
            max_bytes = int(
                getattr(
                    self._capacity_audit_cfg,
                    "max_proof_payload_bytes",
                    32 * 1024 * 1024,
                )
                or 32 * 1024 * 1024
            )
            body = await request.body()
            if len(body) > max_bytes:
                return 413, {
                    "error": "payload_too_large",
                    "max_bytes": max_bytes,
                }
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return 400, {"error": "invalid_json"}
            if not isinstance(payload, dict):
                return 400, {"error": "payload_must_be_object"}
            return 200, payload

        @app.get("/capacity/audit/v1/health")
        async def _health():
            return {
                "status": "ok",
                "service": "verathos-capacity-audit-ingest",
                "capacity_audit": True,
                "protocol_version": PROTOCOL_VERSION,
            }

        async def _receipt(request):
            read_status, payload = await _read_payload(request)
            if read_status != 200:
                return JSONResponse(status_code=read_status, content=payload)
            status, body = self.ingest_capacity_audit_artifact(payload)
            return JSONResponse(status_code=status, content=body)

        async def _proof(request):
            read_status, payload = await _read_payload(request)
            if read_status != 200:
                return JSONResponse(status_code=read_status, content=payload)
            status, body = self.submit_capacity_audit_proof_artifact(payload)
            return JSONResponse(status_code=status, content=body)

        _receipt.__annotations__["request"] = Request
        _proof.__annotations__["request"] = Request
        app.post("/capacity/audit/v1/receipt")(_receipt)
        app.post("/capacity/audit/v1/proof")(_proof)

        return app

    def _start_capacity_audit_ingest_server(self) -> None:
        if not self._capacity_audit_cfg.enabled:
            return
        try:
            import uvicorn
            app = self._build_capacity_audit_ingest_app()
        except Exception as e:
            bt.logging.warning(f"Capacity audit ingest disabled: FastAPI/uvicorn unavailable: {e}")
            return

        host = str(getattr(self.config, "capacity_audit_ingest_host", "127.0.0.1") or "127.0.0.1")
        port = int(getattr(self.config, "capacity_audit_ingest_port", 8091) or 8091)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._capacity_audit_server = server
        self._capacity_audit_server_thread = threading.Thread(
            target=server.run,
            name="capacity-audit-ingest",
            daemon=True,
        )
        self._capacity_audit_server_thread.start()
        bt.logging.info(f"Capacity audit ingest listening on {host}:{port}")

    def _capacity_audit_enforcement_enabled(
        self,
        epoch_number: Optional[int] = None,
    ) -> bool:
        cfg = self._capacity_audit_cfg
        if not cfg.enabled or cfg.mode not in ("score_gate", "enforce"):
            return False
        if bool(getattr(self, "_subnet_runtime_config_authoritative", False)):
            return True
        if epoch_number is not None:
            logged_epoch = getattr(
                self,
                "_capacity_audit_non_authoritative_log_epoch",
                None,
            )
            if logged_epoch != int(epoch_number):
                self._capacity_audit_non_authoritative_log_epoch = int(epoch_number)
                bt.logging.warning(
                    "Capacity audit enforcement disabled for this epoch: "
                    "hosted subnet config is unavailable or invalid"
                )
        return False

    def _capacity_audit_on_chain_model_ids(self, epoch_number: int) -> Optional[List[str]]:
        cache_epoch = getattr(self, "_capacity_audit_model_gate_cache_epoch", None)
        if cache_epoch == int(epoch_number):
            return getattr(self, "_capacity_audit_model_gate_model_ids", None)
        client = getattr(self, "_model_client", None)
        if client is None:
            return None
        try:
            model_ids = client.get_model_list()
            self._capacity_audit_model_gate_cache_epoch = int(epoch_number)
            self._capacity_audit_model_gate_model_ids = model_ids
            return model_ids
        except Exception as exc:
            bt.logging.warning(f"Capacity audit model gate: model list unavailable: {exc}")
            return None

    def _capacity_audit_model_gate_reason(
        self,
        miner: ActiveMiner,
        epoch_number: int,
    ) -> str:
        if not self._capacity_audit_enforcement_enabled(epoch_number):
            return ""
        gpu_name = str(getattr(miner, "gpu_name", "") or "")
        vram_gb = int(getattr(miner, "vram_gb", 0) or 0)
        if not gpu_name or vram_gb <= 0:
            return "capacity-audit model gate: missing hardware metadata"
        gpu_row = match_gpu_class(gpu_name, vram_gb, self._capacity_audit_cfg)
        if gpu_row is None:
            return f"capacity-audit model gate: unsupported GPU class {gpu_name} {vram_gb}GB"
        if not gpu_row.calibrated:
            return f"capacity-audit model gate: uncalibrated GPU class {gpu_row.match_gpu_name}"
        on_chain_models = self._capacity_audit_on_chain_model_ids(epoch_number)
        if on_chain_models is None:
            return ""
        ok, reason, expected = validate_capacity_recommended_model(
            model_id=str(getattr(miner, "model_id", "") or ""),
            quant=str(getattr(miner, "quant", "") or ""),
            max_context_len=int(getattr(miner, "max_context_len", 0) or 0),
            vram_gb=vram_gb,
            on_chain_models=on_chain_models,
        )
        if ok:
            return ""
        if expected is None:
            return f"capacity-audit model gate: {reason}"
        return (
            "capacity-audit model gate: "
            f"{reason}; expected model={expected.model_id} "
            f"quant={expected.quant}"
        )

    def _capacity_audit_score_gate_reason(
        self,
        address: str,
        model_index: int,
        epoch_number: int,
        uid: Optional[int] = None,
    ) -> str:
        cfg = self._capacity_audit_cfg
        if not cfg.enabled or cfg.mode != "score_gate":
            return ""
        if not self._capacity_audit_enforcement_enabled(epoch_number):
            return ""
        if getattr(self, "_capacity_audit_verifier_unhealthy", False):
            reason = str(getattr(self, "_capacity_audit_verifier_last_error", "") or "")
            suffix = f": {reason[:160]}" if reason else ""
            bt.logging.warning(
                f"Capacity audit score gate disabled while proof verifier is unhealthy{suffix}"
            )
            return ""
        since_epoch = max(0, int(epoch_number) - int(cfg.repeat_window_epochs) + 1)
        uid_int: Optional[int]
        try:
            uid_int = int(uid) if uid is not None else None
        except (TypeError, ValueError):
            uid_int = None
        if uid_int is None:
            try:
                resolved = self._db.get_uid(address)
                uid_int = int(resolved) if resolved is not None else None
            except Exception:
                uid_int = None
        if uid_int is not None:
            hard_failures = self._db.recent_capacity_failures_for_uid(
                uid_int,
                since_epoch=since_epoch,
                verdicts=("hard_proof_miss", "no_show"),
                require_chain_confirmed=True,
            )
        else:
            hard_failures = self._db.recent_capacity_failures(
                address,
                model_index,
                since_epoch=since_epoch,
                verdicts=("hard_proof_miss", "no_show"),
                require_chain_confirmed=True,
            )
        if hard_failures >= int(cfg.hard_proof_misses_for_zero_score):
            return f"{hard_failures} hard capacity-audit failures"
        if cfg.allow_timing_only_score_gate:
            if uid_int is not None:
                timing_failures = self._db.recent_capacity_failures_for_uid(
                    uid_int,
                    since_epoch=since_epoch,
                    verdicts=("timing_miss",),
                    require_chain_confirmed=True,
                )
            else:
                timing_failures = self._db.recent_capacity_failures(
                    address,
                    model_index,
                    since_epoch=since_epoch,
                    verdicts=("timing_miss",),
                    require_chain_confirmed=True,
                )
            if timing_failures >= int(cfg.timing_misses_for_zero_score):
                return f"{timing_failures} timing capacity-audit misses"
        return ""

    def _apply_capacity_audit_score_gate(
        self,
        address: str,
        model_index: int,
        uid: int,
        reason: str,
    ) -> bool:
        if not reason:
            return False
        if not self._capacity_audit_enforcement_enabled():
            return False
        if uid is None:
            return False
        state = self.scorer.states.get(uid)
        if state is None or not state.entries:
            bt.logging.info(
                f"Capacity audit score gate: {address[:10]} model_index={model_index} "
                f"matched ({reason}) but UID {uid} has no score state to zero"
            )
            return False
        for entry_model_index, entry in state.entries.items():
            if entry.ema_score != 0.0:
                entry.ema_score = 0.0
            self._db.save_score(
                state.address,
                entry_model_index,
                entry.ema_score,
                entry.total_epochs,
                entry.scored_epochs,
            )
        bt.logging.info(
            f"Capacity audit score gate: zeroed UID {uid} "
            f"entries={len(state.entries)} trigger={address[:10]} model_index={model_index} "
            f"({reason})"
        )
        return True

    def _apply_capacity_audit_model_gate(
        self,
        address: str,
        model_index: int,
        uid: int,
        reason: str,
    ) -> bool:
        if not reason:
            return False
        if not self._capacity_audit_enforcement_enabled():
            return False
        if uid is None:
            return False
        state = self.scorer.states.get(uid)
        if state is None or not state.entries:
            bt.logging.info(
                f"Capacity audit model gate: {address[:10]} model_index={model_index} "
                f"matched ({reason}) but UID {uid} has no score state to zero"
            )
            return False
        entry = state.entries.get(model_index)
        if entry is None:
            bt.logging.info(
                f"Capacity audit model gate: {address[:10]} model_index={model_index} "
                f"matched ({reason}) but UID {uid} has no matching score entry"
            )
            return False
        if entry.ema_score != 0.0:
            entry.ema_score = 0.0
        self._db.save_score(
            address,
            model_index,
            entry.ema_score,
            entry.total_epochs,
            entry.scored_epochs,
        )
        bt.logging.info(
            f"Capacity audit model gate: zeroed UID {uid} "
            f"model_index={model_index} address={address[:10]} ({reason})"
        )
        return True

    @staticmethod
    def _capacity_audit_receipt_overlap_s(
        receipt: ServiceReceipt,
        *,
        window_start: float,
        window_end: float,
        slack_s: float = 2.0,
    ) -> float:
        if int(getattr(receipt, "tokens_generated", 0) or 0) <= 0:
            return 0.0
        if not receipt_has_validator_observed_timing(receipt):
            return 0.0
        proofed = (
            bool(getattr(receipt, "proof_requested", False) and getattr(receipt, "proof_verified", False))
            or getattr(receipt, "tee_attestation_verified", None) is True
        )
        if not proofed:
            return 0.0
        interval = receipt_observed_interval(receipt)
        if interval is None:
            return 0.0
        start, end = interval
        start -= slack_s
        end += slack_s
        return max(0.0, min(end, window_end) - max(start, window_start))

    def _reconcile_capacity_audit_timing_excuses(
        self,
        miner: ActiveMiner,
        all_receipts: List[ServiceReceipt],
        epoch_number: int,
    ) -> int:
        """Neutralize timing/no-show misses explained by signed work overlap.

        This is intentionally not a generic busy excuse. ``all_receipts`` is
        expected to come from the verified receipt pull path. Only v2 receipts
        whose validator-observed timing is separately signed and whose work is
        proof/TEE verified can excuse a hot-capacity timing miss. The receipt
        may be a canary or organic request; miner-reported timing is ignored.
        """
        cfg = self._capacity_audit_cfg
        if not cfg.enabled or cfg.mode not in ("score_gate", "enforce"):
            return 0
        if not self._capacity_audit_enforcement_enabled(epoch_number):
            return 0
        try:
            failed_slots = self._db.get_capacity_audit_slots_for_epoch(
                int(epoch_number),
                address=miner.address,
                model_index=int(miner.model_index),
                verdicts=("timing_miss", "no_show"),
            )
        except Exception as exc:
            bt.logging.debug(f"Capacity audit overlap reconciliation lookup failed: {exc}")
            return 0
        if not failed_slots or not all_receipts:
            return 0

        relevant_receipts = [
            r for r in all_receipts
            if r.miner_address.lower() == miner.address.lower()
            and int(r.model_index) == int(miner.model_index)
            and r.model_id == miner.model_id
        ]
        if not relevant_receipts:
            return 0

        updated = 0
        for row in failed_slots:
            audit_start = row.get("audit_start_observed_at")
            if audit_start is None:
                continue
            audit_start = float(audit_start)
            deadline_s = float(row.get("deadline_s") or cfg.deadline_s)
            transport_grace_s = float(row.get("transport_grace_s") or cfg.transport_grace_s)
            window_end = audit_start + deadline_s + transport_grace_s
            if row.get("final_received_at") is not None:
                late_s = max(0.0, float(row["final_received_at"]) - window_end)
                required_overlap_s = max(1.0, min(5.0, late_s if late_s > 0 else 1.0))
            else:
                required_overlap_s = 5.0

            overlaps: list[tuple[ServiceReceipt, float]] = []
            total_overlap_s = 0.0
            for receipt in relevant_receipts:
                overlap_s = self._capacity_audit_receipt_overlap_s(
                    receipt,
                    window_start=audit_start,
                    window_end=window_end,
                )
                if overlap_s <= 0:
                    continue
                overlaps.append((receipt, overlap_s))
                total_overlap_s += overlap_s

            if total_overlap_s < required_overlap_s:
                continue
            validators = {
                getattr(r, "validator_hotkey", b"").hex()
                for r, _overlap in overlaps
                if getattr(r, "validator_hotkey", b"")
            }
            canary_count = sum(1 for r, _overlap in overlaps if getattr(r, "is_canary", False))
            organic_count = max(0, len(overlaps) - canary_count)
            reason = (
                "verified_work_overlap_receipt:"
                f"overlap_s={total_overlap_s:.1f},"
                f"required_s={required_overlap_s:.1f},"
                f"receipts={len(overlaps)},"
                f"canary={canary_count},"
                f"organic={organic_count},"
                f"validators={len(validators)},"
                f"prior_verdict={row.get('verdict') or ''}"
            )
            changed = self._db.mark_capacity_audit_timing_excused(
                audit_id=str(row["audit_id"]),
                address=miner.address,
                model_index=int(miner.model_index),
                reason=reason,
            )
            if changed:
                updated += changed
                bt.logging.info(
                    f"Capacity audit timing excused for {miner.address[:10]} "
                    f"model_index={miner.model_index} audit_id={str(row['audit_id'])[:12]} "
                    f"({reason})"
                )
        if updated:
            self._write_shared_state()
        return updated

    def _on_capacity_audit_failure(
        self,
        miner_address: str,
        model_index: int,
        *,
        endpoint: str = "",
        verdict: str = "",
        failure_reason: str = "",
    ) -> None:
        cfg = self._capacity_audit_cfg
        if not cfg.enabled or cfg.mode not in ("score_gate", "enforce"):
            return
        if not self._capacity_audit_enforcement_enabled():
            return
        if not miner_address:
            return
        bt.logging.info(
            f"Capacity audit failure -> probation: {miner_address[:10]} "
            f"model_index={model_index} verdict={verdict or 'unknown'} "
            f"reason={failure_reason or 'unknown'}"
        )
        self._on_proof_failure(miner_address, int(model_index), endpoint=endpoint)

    def on_finalized_block(
        self,
        block_number: int,
        block_hash: bytes,
        *,
        block_hash_real: bool = True,
    ):
        """Called by WebSocket subscription on each processed chain-head block.

        Drives the epoch lifecycle: start epoch, dispatch canary tests,
        close epoch (pull receipts + score), set weights.
        """
        # Skip historical blocks — only process from sync point onward.
        if block_number < self._sync_block:
            return

        _wd_t0 = time.monotonic()

        epoch_blocks = self.config.epoch_blocks
        blocks_into_epoch = block_number % epoch_blocks
        blocks_until_next = epoch_blocks - blocks_into_epoch
        sched_count = len(self._canary_scheduler.tests) if self._canary_scheduler else 0
        bt.logging.info(
            f"Block {block_number} | epoch {blocks_into_epoch}/{epoch_blocks} "
            f"| next_epoch_in={blocks_until_next} | pending_tests={sched_count}",
        )

        # Refresh metagraph from RPC every 60 blocks (~12 min),
        # re-log cached stats every 5 blocks (~1 min)
        if block_number % 20 == 0:
            self._refresh_metagraph_stats()
            # Re-enrich miners with SS58 + update shared state (~4 min cycle).
            # Catches new miners within ~4 min instead of waiting for epoch boundary.
            if self._epoch_miners:
                self._enrich_miners_from_metagraph(self._epoch_miners)
                self._write_shared_state()
        refresh_blocks = int(getattr(self.config, "capacity_audit_slot_refresh_blocks", 60) or 0)
        if (
            self._capacity_audit_cfg.enabled
            and refresh_blocks > 0
            and block_number % refresh_blocks == 0
        ):
            self._request_capacity_audit_slot_snapshot_refresh(
                block_number=block_number,
                reason="periodic",
            )
        if block_number % 5 == 0 and hasattr(self, '_cached_metagraph_parts'):
            bt.logging.info(f"Metagraph | block={block_number} | {' | '.join(self._cached_metagraph_parts)}")

        self._handle_capacity_audit_block(
            block_number,
            block_hash,
            block_hash_real=block_hash_real,
        )

        # 1. Epoch boundary → start new epoch
        if block_number % epoch_blocks == 0:
            # If there's a pending epoch close, close it first
            if self._pending_epoch_close is not None:
                self._try_close_epoch(self._pending_epoch_close)

            self._start_new_epoch(block_number)

        # 1b. Retry failed epoch start — skip while background setup is in flight.
        elif (
            blocks_into_epoch <= 30  # only retry in first ~6 min
            and self._pending_epoch_close is not None
            and not getattr(self, "_epoch_setup_in_progress", False)
            and (
                self._canary_scheduler is None
                or self._canary_scheduler.epoch_number != block_number // epoch_blocks
            )
        ):
            epoch_start = block_number - blocks_into_epoch
            bt.logging.info(f"Retrying epoch start at block {epoch_start} (offset {blocks_into_epoch})")
            self._start_new_epoch(epoch_start)

        # 2. Dispatch pending canary tests
        if self._canary_scheduler is not None:
            with self._canary_scheduler_lock:
                pending = self._canary_scheduler.get_pending_tests(block_number)
            pending = self._defer_capacity_audit_drained_canaries(pending, block_number)
            if pending:
                for t in pending:
                    _uid = self._db.get_uid(t.miner_address)
                    _uid_str = f"UID {_uid}" if _uid is not None else "UID ?"
                    bt.logging.debug(
                        f"Dispatching canary: {_uid_str} {t.miner_address[:10]} "
                        f"model={t.model_id} type={t.test_type} "
                        f"proof={t.verify_proof} tokens={t.max_new_tokens}",
                    )
                self._dispatch_canary_tests(pending)

        # 3. Epoch close: grace blocks after epoch boundary
        if (self._pending_epoch_close is not None
                and block_number >= self._epoch_close_block):
            self._try_close_epoch(self._pending_epoch_close)

        # 4. Weight-setting boundary
        if block_number % self.config.set_weights_epoch_blocks == 0:
            model_budgets = getattr(self, "_last_model_emission_budgets", {})
            model_bucket_mode = bool(model_budgets)
            model_unallocated = 0.0
            if model_bucket_mode:
                weights, model_unallocated = self.scorer.get_model_bucket_weights(
                    model_budgets,
                    model_groups=getattr(self, "_last_model_emission_groups", {}),
                    group_budgets=getattr(self, "_last_model_group_budgets", {}),
                )
            else:
                weights = self.scorer.get_weights()

            if weights or (model_bucket_mode and model_unallocated > 0):
                emission_burn = max(0.0, min(1.0, self._scoring.emission_burn))
                if model_bucket_mode:
                    # In bucket mode, blacklisted and unserved model shares are
                    # burned. Renormalizing them would leak that budget back to
                    # already-served models.
                    if self._blacklisted_uids:
                        for uid in self._blacklisted_uids:
                            removed = weights.get(uid, 0.0)
                            if removed > 0:
                                model_unallocated += removed
                                weights[uid] = 0.0

                    miner_share = sum(weights.values())
                    model_unallocated = min(
                        1.0,
                        max(model_unallocated, 1.0 - miner_share),
                    )
                    miner_scale = 1.0 - emission_burn
                    if miner_scale < 1.0:
                        for uid in list(weights.keys()):
                            weights[uid] *= miner_scale
                    burn_weight = emission_burn + miner_scale * model_unallocated
                    if burn_weight > 0:
                        weights[self._burn_uid] = (
                            weights.get(self._burn_uid, 0.0) + burn_weight
                        )
                    self._last_model_bucket_burn = burn_weight
                    if burn_weight > 0:
                        bt.logging.info(
                            f"Emission burn: {emission_burn:.0%} global + "
                            f"{model_unallocated:.1%} unallocated model buckets "
                            f"to UID {self._burn_uid}"
                        )
                else:
                    # Zero out blacklisted miners before weight normalization
                    if self._blacklisted_uids:
                        for uid in self._blacklisted_uids:
                            if uid in weights:
                                weights[uid] = 0.0
                        # Re-normalize after zeroing
                        total = sum(weights.values())
                        if total > 0:
                            weights = {uid: s / total for uid, s in weights.items()}
                    # Apply emission burn: redirect a fraction of weight to burn UID
                    if emission_burn > 0:
                        for uid in list(weights.keys()):
                            weights[uid] *= (1.0 - emission_burn)
                        weights[self._burn_uid] = weights.get(self._burn_uid, 0.0) + emission_burn
                        bt.logging.info(f"Emission burn: {emission_burn:.0%} to UID {self._burn_uid}")
                # Run set_weights + retries on executor so this callback returns fast.
                def _set_weights_with_retry(_w=dict(weights)):
                    for _sw_attempt in range(1, 4):
                        try:
                            self._set_weights(_w)
                            self._last_weights = _w
                            return
                        except Exception as _sw_err:
                            if _sw_attempt == 3:
                                bt.logging.error(f"set_weights failed after 3 attempts: {_sw_err}")
                            else:
                                _sw_delay = 30 * (2 ** (_sw_attempt - 1))
                                bt.logging.warning(f"set_weights attempt {_sw_attempt}/3 failed: {_sw_err} — retrying in {_sw_delay}s")
                                time.sleep(_sw_delay)
                self._control_executor.submit(_set_weights_with_retry)

        _wd_elapsed = time.monotonic() - _wd_t0
        if _wd_elapsed > 12.0:
            bt.logging.warning(
                f"on_finalized_block took {_wd_elapsed:.1f}s for block {block_number} "
                f"(>12s; main loop is falling behind chain)"
            )

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def _start_new_epoch(self, epoch_start_block: int):
        """Start a new epoch — non-blocking; heavy setup runs on executor."""
        epoch_number = epoch_start_block // self.config.epoch_blocks
        self._refresh_subnet_runtime_config(current_epoch=epoch_number, force=True)
        epoch_number = epoch_start_block // self.config.epoch_blocks
        if epoch_number != self._current_epoch:
            self._reset_canary_executor()
        self._current_epoch = epoch_number
        self._epoch_start_block = epoch_start_block

        # Schedule epoch close for after grace window
        self._pending_epoch_close = epoch_number
        self._epoch_close_block = (
            epoch_start_block + self.config.epoch_blocks + self.config.epoch_grace_blocks
        )

        # Dispatch no-ops until background sets the scheduler.
        self._canary_scheduler = None

        if getattr(self, "_epoch_setup_in_progress", False):
            bt.logging.debug(
                f"Epoch {epoch_number}: setup already in progress, skipping duplicate _start_new_epoch"
            )
            return
        self._epoch_setup_in_progress = True

        def _setup_and_clear_flag():
            try:
                self._do_epoch_setup(epoch_start_block, epoch_number)
            except Exception as e:
                bt.logging.error(f"Epoch {epoch_number} setup failed: {e}")
            finally:
                self._epoch_setup_in_progress = False

        self._control_executor.submit(_setup_and_clear_flag)

    def _reset_canary_executor(self):
        """Cancel queued canaries from prior epochs and create a fresh pool."""
        old_executor = self._executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_verifications,
        )
        old_executor.shutdown(wait=False, cancel_futures=True)

    def _canary_epoch_active(self, epoch_number: int) -> bool:
        """Return True while a canary still belongs to the active epoch."""
        return self._running and epoch_number == self._current_epoch

    @staticmethod
    def _miner_model_key(address: str, model_index: int) -> Tuple[str, int]:
        """Canonical in-memory key for per-epoch miner/model accounting."""
        return (str(address).lower(), int(model_index))

    def _mark_canary_started(self, epoch_number: int, key: Tuple[str, int]) -> None:
        """Count a canary only once it actually starts running."""
        key = self._miner_model_key(key[0], key[1])
        self._expected_receipts[key] = self._expected_receipts.get(key, 0) + 1
        epoch_inflight = self._inflight_canaries.setdefault(epoch_number, {})
        epoch_inflight[key] = epoch_inflight.get(key, 0) + 1

    def _mark_canary_finished(self, epoch_number: int, key: Tuple[str, int]) -> None:
        """Clear in-flight accounting when a canary finishes or is abandoned."""
        key = self._miner_model_key(key[0], key[1])
        epoch_inflight = self._inflight_canaries.get(epoch_number)
        if not epoch_inflight:
            return
        current = epoch_inflight.get(key, 0)
        if current <= 1:
            epoch_inflight.pop(key, None)
        else:
            epoch_inflight[key] = current - 1
        if not epoch_inflight:
            self._inflight_canaries.pop(epoch_number, None)

    def _decrement_expected_receipt(self, epoch_number: int, key: Tuple[str, int]) -> None:
        """Drop expected receipt count for active-epoch validator-side misses."""
        if epoch_number != self._current_epoch:
            return
        key = self._miner_model_key(key[0], key[1])
        if self._expected_receipts.get(key, 0) > 0:
            self._expected_receipts[key] -= 1

    def _effective_expected_receipts(self, epoch_number: int, key: Tuple[str, int]) -> int:
        """Expected receipts minus canaries still in flight at close time."""
        key = self._miner_model_key(key[0], key[1])
        expected = self._expected_receipts.get(key, 0)
        inflight_epoch = self._closing_inflight_canaries.get(
            epoch_number,
            self._inflight_canaries.get(epoch_number, {}),
        )
        inflight = inflight_epoch.get(key, 0)
        return max(0, expected - inflight)

    def _capacity_audit_drained_keys(self) -> Set[Tuple[str, int]]:
        """Return endpoint slots currently drained for a capacity audit window."""
        cfg = getattr(self, "_capacity_audit_cfg", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return set()
        try:
            drains = self._db.get_capacity_drains()
        except Exception as exc:
            bt.logging.debug(f"Capacity audit drain lookup failed: {exc}")
            return set()
        out: Set[Tuple[str, int]] = set()
        for drain in drains:
            try:
                out.add((str(drain.address).lower(), int(drain.model_index)))
            except Exception:
                continue
        return out

    def _capacity_audit_key_drained(self, key: Tuple[str, int]) -> bool:
        address, model_index = key
        return self._miner_model_key(address, model_index) in self._capacity_audit_drained_keys()

    def _requeue_capacity_audit_canary(
        self,
        test: CanaryTest,
        epoch_number: int,
        *,
        block_number: Optional[int] = None,
        phase: str = "dispatch",
    ) -> bool:
        """Requeue one canary when its target endpoint is in an audit drain."""
        if not self._canary_epoch_active(epoch_number):
            return False
        key = self._miner_model_key(test.miner_address, test.model_index)
        if not self._capacity_audit_key_drained(key):
            return False

        if block_number is None:
            block_number = int(self._last_known_block or self._epoch_start_block or 0)
        test.target_block = int(block_number) + 1
        with self._canary_scheduler_lock:
            if self._canary_scheduler is None or self._canary_scheduler.epoch_number != epoch_number:
                return False
            self._canary_scheduler.tests.append(test)
            self._canary_scheduler.tests.sort(key=lambda t: (t.target_block, t.miner_address))
        bt.logging.info(
            f"Capacity audit drain: requeued canary for {test.miner_address[:10]} "
            f"model_index={test.model_index} phase={phase} block={block_number}"
        )
        return True

    @staticmethod
    def _is_capacity_audit_http_503(exc: BaseException) -> bool:
        if not ValidatorNeuron._is_http_503(exc):
            return False
        response = getattr(exc, "response", None)
        try:
            body = str(response.text or "").lower()
        except Exception:
            body = ""
        return "capacity audit" in body or "audit in progress" in body

    @staticmethod
    def _is_http_503(exc: BaseException) -> bool:
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        response = getattr(exc, "response", None)
        return response is not None and int(getattr(response, "status_code", 0) or 0) == 503

    def _requeue_capacity_audit_gate_canary(
        self,
        test: CanaryTest,
        epoch_number: int,
        *,
        block_number: Optional[int] = None,
        phase: str = "audit_gate_503",
    ) -> bool:
        """Requeue one canary after the miner's local audit gate rejected it."""
        if not self._canary_epoch_active(epoch_number):
            return False
        if block_number is None:
            block_number = int(self._last_known_block or self._epoch_start_block or 0)
        test.target_block = int(block_number) + 1
        with self._canary_scheduler_lock:
            if self._canary_scheduler is None or self._canary_scheduler.epoch_number != epoch_number:
                return False
            self._canary_scheduler.tests.append(test)
            self._canary_scheduler.tests.sort(key=lambda t: (t.target_block, t.miner_address))
        bt.logging.info(
            f"Capacity audit gate: requeued canary for {test.miner_address[:10]} "
            f"model_index={test.model_index} phase={phase} block={block_number}"
        )
        return True

    def _defer_capacity_audit_drained_canaries(
        self,
        tests: List[CanaryTest],
        block_number: int,
    ) -> List[CanaryTest]:
        """Requeue canaries whose endpoint is temporarily drained for audit.

        Audit drains are not miner failures; they reserve a short deterministic
        window where organic traffic and validator canaries must stay off the
        selected slot so the hot-capacity timing signal is clean.  Requeueing to
        the next block preserves coverage when the drain clears and avoids
        counting a skipped audit-window canary as an expected receipt.
        """
        if not tests:
            return tests
        drained = self._capacity_audit_drained_keys()
        if not drained:
            return tests

        runnable: List[CanaryTest] = []
        deferred: List[CanaryTest] = []
        for test in tests:
            key = self._miner_model_key(test.miner_address, test.model_index)
            if key in drained:
                test.target_block = int(block_number) + 1
                deferred.append(test)
            else:
                runnable.append(test)

        if deferred and self._canary_scheduler is not None:
            with self._canary_scheduler_lock:
                if self._canary_scheduler is not None:
                    self._canary_scheduler.tests.extend(deferred)
                    self._canary_scheduler.tests.sort(key=lambda t: (t.target_block, t.miner_address))
            bt.logging.info(
                f"Capacity audit drain: deferred {len(deferred)} canary test(s) "
                f"at block {block_number}"
            )
        return runnable

    def _do_epoch_setup(self, epoch_start_block: int, epoch_number: int):
        """Heavy epoch setup — runs on a background executor thread."""
        t0 = time.monotonic()

        # Discover ALL active miners
        previous_miners = list(self._epoch_miners)  # cache for fallback
        discovery_failed = False
        try:
            self._epoch_miners = discover_active_miners(
                self._miner_client, self._model_client,
            )
            self._epoch_miners_discovery_valid = True
        except Exception as e:
            bt.logging.warning(f"Discovery RPC failed: {e} — will fall back to previous miners")
            discovery_failed = True
            self._epoch_miners = []  # triggers fallback below
            self._epoch_miners_discovery_valid = False
        bt.logging.info(f"Epoch {epoch_number} (block {epoch_start_block}): discovered {len(self._epoch_miners)} miner entries")

        # Enrich miners with SS58 keys from metagraph (for analytics + shared state)
        self._enrich_miners_from_metagraph(self._epoch_miners)

        if not self._epoch_miners:
            if not discovery_failed:
                deactivated = self._db.mark_unseen_inactive(epoch_number)
                if deactivated > 0:
                    bt.logging.info(f"Deactivated {deactivated} stale miner entries")
                self._refresh_capacity_audit_slot_snapshot_from_miners(
                    [],
                    block_number=epoch_start_block,
                    source="epoch_setup_empty",
                )
                self._canary_scheduler = None
                self._write_shared_state()
                return
            # RPC failure (e.g. 429 rate limit) — fall back to previous
            # epoch's miners so canary testing continues uninterrupted.
            # Miners don't change between epochs in practice.
            if previous_miners:
                self._epoch_miners = previous_miners
                self._epoch_miners_discovery_valid = False
                bt.logging.warning(f"Discovery returned 0 miners — falling back to {len(previous_miners)} miners from previous epoch")
            else:
                # Fresh start with no previous miners — try shared state
                from neurons.shared_state import read_shared_state
                try:
                    shared = read_shared_state(self.config.shared_state_path)
                    if shared and shared.miner_endpoints:
                        self._epoch_miners = [
                            ActiveMiner(
                                address=m.address, endpoint=m.endpoint,
                                model_id=m.model_id, model_index=m.model_index,
                                quant=m.quant, max_context_len=m.max_context_len,
                            )
                            for m in shared.miner_endpoints
                        ]
                        self._epoch_miners_discovery_valid = False
                        bt.logging.warning(f"Discovery returned 0 miners, no previous epoch — falling back to {len(self._epoch_miners)} miners from shared state")
                except Exception:
                    pass

                if not self._epoch_miners:
                    self._canary_scheduler = None
                    return

        # ── Fast TCP pre-filter: skip dead endpoints entirely ─────────
        # Before spending resources on TLS/HTTP identity challenges, do a
        # cheap 1-second TCP connect to each endpoint.  Dead endpoints
        # (stale on-chain entries, stopped miners) are filtered out without
        # spawning threads or opening TLS sessions.  This makes discovery
        # O(1) per dead entry instead of O(15s) — critical at scale where
        # there may be hundreds of stale entries.
        #
        # Cache: endpoints that fail the TCP check are remembered for
        # the next `tcp_prefilter_ttl_epochs` epochs so we don't re-test
        # them every epoch (further reducing wasted effort).
        import socket
        from urllib.parse import urlparse

        def _tcp_alive(endpoint: str, timeout: float = 2.0, retries: int = 2) -> bool:
            """TCP connect check with retry. Worst case per endpoint: retries × timeout."""
            try:
                p = urlparse(endpoint)
                host = p.hostname
                port = p.port or (443 if p.scheme == "https" else 80)
                if not host:
                    return False
            except Exception:
                return False
            for attempt in range(retries):
                try:
                    with socket.create_connection((host, port), timeout=timeout):
                        return True
                except Exception:
                    pass  # no sleep — next attempt starts immediately
            return False

        # Parallel TCP check — each probe is 1s, total wall time scales with
        # miner_count / pool_workers.  Budget: enough batches for all miners
        # plus a 5s grace.  At 500 entries and 32 workers = ~16 batches ~16s + 5s = 21s.
        alive_miners = []
        tcp_futures = {
            self._control_executor.submit(_tcp_alive, m.endpoint): m
            for m in self._epoch_miners
        }
        prefilter_dead = 0
        # Deadline: scales with miner count / pool workers. Dead endpoints
        # take up to 2×2s=4s each (2 retries × 2s timeout), but they run
        # in parallel across the thread pool.
        _tcp_timeout = min(30, max(10, len(self._epoch_miners) // self.config.max_concurrent_verifications * 4 + 5))
        try:
            for fut in as_completed(tcp_futures, timeout=_tcp_timeout):
                miner = tcp_futures[fut]
                try:
                    if fut.result():
                        alive_miners.append(miner)
                    else:
                        prefilter_dead += 1
                except Exception:
                    prefilter_dead += 1
        except _FuturesTimeout:
            # Any TCP probe still running after 5s is treated as dead
            for f in tcp_futures:
                if not f.done():
                    f.cancel()
                    prefilter_dead += 1

        if prefilter_dead:
            bt.logging.info(
                f"TCP pre-filter: {len(alive_miners)}/{len(self._epoch_miners)} alive, "
                f"{prefilter_dead} unreachable"
            )
            # Create DB entries for unreachable miners so they appear in
            # miner_scores with ema=0 (not absent → proxy defaults to 1.0).
            dead_miners = [m for m in self._epoch_miners if m not in alive_miners]
            for m in dead_miners:
                self._db.upsert_entry(
                    address=m.address, model_index=m.model_index,
                    model_id=m.model_id, endpoint=m.endpoint,
                    quant=m.quant, max_context_len=m.max_context_len,
                    epoch=epoch_number,
                    hotkey_ss58=getattr(m, "hotkey_ss58", ""),
                    coldkey_ss58=getattr(m, "coldkey_ss58", ""),
                )
                self._db.mark_entry_inactive(m.address, m.model_index)
        self._epoch_miners = alive_miners
        if not self._epoch_miners:
            bt.logging.warning(f"Epoch {epoch_number}: no reachable miners after TCP pre-filter")
            self._refresh_capacity_audit_slot_snapshot_from_miners(
                [],
                block_number=epoch_start_block,
                source="epoch_setup_no_reachable",
            )
            self._canary_scheduler = None
            self._write_shared_state()
            return

        # ── Identity verification: filter out hijacked endpoints ────
        # CRITICAL: one failing miner must NOT block others.  The per-miner
        # budget is bounded (deadline inside _verify_miner_identity), and the
        # overall as_completed loop is wrapped in try/except so a single
        # unresponsive miner can never stall epoch start for others.
        verified_miners = []
        identity_groups = _group_miners_for_identity(self._epoch_miners)
        if len(identity_groups) < len(self._epoch_miners):
            bt.logging.info(
                f"Identity verification grouped {len(self._epoch_miners)} model entries "
                f"into {len(identity_groups)} unique miner endpoints"
            )
        id_futures = {}
        for miners in identity_groups.values():
            representative = miners[0]
            future = self._control_executor.submit(self._verify_miner_identity, representative)
            id_futures[future] = (representative, miners)

        # Overall deadline: scales with unique endpoint groups / pool workers,
        # but stays capped. Allow enough per-miner budget for one cold HTTP
        # timeout plus a retry; beyond that, proceed with miners that verified.
        _id_batches = len(identity_groups) // self.config.max_concurrent_verifications + 1
        _id_per_miner_budget = min(
            45,
            max(
                self.config.identity_challenge_timeout + 10,
                self.config.identity_challenge_timeout * 2 + 5,
            ),
        )
        overall_deadline = min(120, max(
            _id_per_miner_budget,
            _id_batches * 3 + 10,  # ~3s per batch (most pass fast) + 10s grace
        ))
        completed_futures = set()
        try:
            for future in as_completed(id_futures, timeout=overall_deadline):
                completed_futures.add(future)
                miner, grouped_miners = id_futures[future]
                try:
                    result = future.result()
                    if result is False:
                        bt.logging.info(
                            f"Identity FAILED for {miner.address[:10]} at {miner.endpoint} — "
                            f"excluding {len(grouped_miners)} model entr{'y' if len(grouped_miners) == 1 else 'ies'} from epoch"
                        )
                        # Dispatch chain call to executor — wait_for_transaction_receipt
                        # blocks up to 360s per call (3×120s retries) and would stall the
                        # main loop while we wait.  Background task logs its own outcome.
                        self._control_executor.submit(self._report_offline, miner)
                        continue
                    if result is None and self.config.identity_challenge_required:
                        bt.logging.info(
                            f"Identity UNSUPPORTED for {miner.address[:10]} at {miner.endpoint} — "
                            f"excluding {len(grouped_miners)} model entr{'y' if len(grouped_miners) == 1 else 'ies'} (required mode)"
                        )
                        continue
                    verified_miners.extend(grouped_miners)
                except Exception as e:
                    bt.logging.info(
                        f"Identity check error for {miner.address[:10]}: {e} — "
                        f"including {len(grouped_miners)} model entr{'y' if len(grouped_miners) == 1 else 'ies'}"
                    )
                    verified_miners.extend(grouped_miners)
        except _FuturesTimeout:
            # One or more futures didn't finish in time — exclude them from
            # this epoch but continue with the verified ones.  This prevents
            # a single stalled miner from blocking canary dispatch for all.
            stalled = [id_futures[f][0].address[:10] for f in id_futures if f not in completed_futures]
            stalled_entries = sum(len(id_futures[f][1]) for f in id_futures if f not in completed_futures)
            bt.logging.warning(
                f"Identity verification timeout after {overall_deadline}s — "
                f"{len(stalled)} unique miner endpoint(s) stalled ({stalled_entries} model entries): {stalled}. "
                f"Proceeding with {len(verified_miners)} verified model entries."
            )
            # Cancel stalled futures so threads can be reused
            for f in id_futures:
                if f not in completed_futures:
                    f.cancel()

        excluded = len(self._epoch_miners) - len(verified_miners)
        if excluded > 0:
            bt.logging.info(f"Identity verification: {len(verified_miners)}/{len(self._epoch_miners)} miners passed, {excluded} excluded")
        self._epoch_miners = verified_miners

        if not self._epoch_miners:
            self._canary_scheduler = None
            return

        # ── Fetch hardware metadata from miner /health (best-effort) ──
        hw_futures = {}
        for miner in self._epoch_miners:
            future = self._control_executor.submit(self._fetch_miner_hardware, miner)
            hw_futures[future] = miner
        hardware_failed = []
        try:
            for future in as_completed(hw_futures, timeout=10):
                miner = hw_futures[future]
                try:
                    if future.result() is False:
                        hardware_failed.append(miner)
                except Exception:
                    pass  # Non-fatal — hardware metadata is optional
        except _FuturesTimeout:
            for f in hw_futures:
                if not f.done():
                    f.cancel()

        if hardware_failed:
            failed_ids = {id(m) for m in hardware_failed}
            for miner in hardware_failed:
                bt.logging.info(
                    f"Hardware health FAILED for {miner.address[:10]} idx={miner.model_index} "
                    f"at {miner.endpoint}: gpu_count={miner.gpu_count}, vram_gb={miner.vram_gb} — excluding from epoch"
                )
                self._db.upsert_entry(
                    address=miner.address, model_index=miner.model_index,
                    model_id=miner.model_id, endpoint=miner.endpoint,
                    quant=miner.quant, max_context_len=miner.max_context_len,
                    epoch=epoch_number,
                    hotkey_ss58=getattr(miner, "hotkey_ss58", ""),
                    coldkey_ss58=getattr(miner, "coldkey_ss58", ""),
                    gpu_name=getattr(miner, "gpu_name", ""),
                    gpu_count=getattr(miner, "gpu_count", 0),
                    vram_gb=getattr(miner, "vram_gb", 0),
                    compute_capability=getattr(miner, "compute_capability", ""),
                    gpu_uuids=getattr(miner, "gpu_uuids", []),
                )
                self._db.save_score(miner.address, miner.model_index, 0.0, 0, 0)
                self._control_executor.submit(self._report_offline, miner)
            self._epoch_miners = [m for m in self._epoch_miners if id(m) not in failed_ids]
            bt.logging.info(
                f"Hardware health: {len(self._epoch_miners)}/{len(self._epoch_miners) + len(hardware_failed)} "
                f"miners passed, {len(hardware_failed)} excluded"
            )

        if not self._epoch_miners:
            self._canary_scheduler = None
            return

        self._refresh_capacity_audit_slot_snapshot_from_miners(
            self._epoch_miners,
            block_number=epoch_start_block,
            source="epoch_setup",
        )

        # Persist discovered miners to DB
        for miner in self._epoch_miners:
            self._db.upsert_entry(
                address=miner.address, model_index=miner.model_index,
                model_id=miner.model_id, endpoint=miner.endpoint,
                quant=miner.quant, max_context_len=miner.max_context_len,
                epoch=epoch_number,
                hotkey_ss58=miner.hotkey_ss58,
                coldkey_ss58=miner.coldkey_ss58,
                tee_enabled=getattr(miner, "tee_enabled", False),
                tee_platform=getattr(miner, "tee_platform", ""),
                gpu_name=miner.gpu_name,
                gpu_count=miner.gpu_count,
                vram_gb=miner.vram_gb,
                compute_capability=miner.compute_capability,
                gpu_uuids=miner.gpu_uuids,
            )
        # Mark entries not seen this epoch as inactive
        deactivated = self._db.mark_unseen_inactive(epoch_number)
        if deactivated > 0:
            bt.logging.info(f"Deactivated {deactivated} stale miner entries")

        # Pre-warm UID cache: resolve each miner's EVM address → Bittensor UID
        # at epoch start when RPC budget is freshest. The cache persists in the
        # DB, so even if the RPC is 429'd during epoch close, the cached UID
        # allows scoring to proceed without an on-chain lookup.
        # Bounded by a 15s deadline so slow RPC doesn't block epoch start.
        _uid_deadline = time.monotonic() + 15
        _uid_cached = 0
        for miner in self._epoch_miners:
            if time.monotonic() > _uid_deadline:
                _remaining = len(self._epoch_miners) - _uid_cached
                bt.logging.warning(f"UID cache warm-up hit 15s deadline — {_remaining} miners uncached (will use DB cache at epoch close)")
                break
            try:
                uid = self._miner_client.get_associated_uid(miner.address)
                if uid is not None:
                    self._db.set_uid(miner.address, uid)
                    _uid_cached += 1
            except Exception:
                pass  # Non-fatal — close falls back to DB cache

        # Pre-fetch ModelSpecs for all unique models this epoch (one RPC call
        # per model, not per canary). Canary verification uses the cached spec
        # instead of calling client.fetch_model_spec() each time.
        # IMPORTANT: Do NOT clear the cache — previous epoch's specs are still
        # valid (same model, same on-chain Merkle roots). Only overwrite on
        # successful fetch. This way, RPC 429 at epoch start doesn't leave
        # the entire epoch without specs.
        unique_models = {m.model_id for m in self._epoch_miners}
        for model_id in unique_models:
            try:
                spec = self._model_client.get_model_spec(model_id)
                if spec is not None:
                    self._model_spec_cache[model_id] = spec
                    bt.logging.info(f"Cached ModelSpec for {model_id}")
                    # Tokenizer drift check: compare local tokenizer hash to
                    # the on-chain anchor.  On mismatch, mark as drifted —
                    # canary path will short-circuit and attribute correctly
                    # without penalizing the miner.
                    self._check_tokenizer_drift(model_id, spec)
            except Exception as e:
                if model_id in self._model_spec_cache:
                    bt.logging.warning(f"Failed to refresh ModelSpec for {model_id}: {e} — using previous epoch's cache")
                else:
                    bt.logging.warning(f"Failed to fetch ModelSpec for {model_id}: {e} — no cache available")

        # Reset expected receipt counts, busy-skip tracker, and canary error tracker
        self._expected_receipts = {}
        self._busy_skips = {}
        self._busy_skip_probations = {}
        self._canary_errors: Dict[Tuple[str, int], int] = {}
        self._canary_error_times: Dict[Tuple[str, int], List[int]] = {}

        # Check if TEE is enabled on the subnet (feature flag from SubnetConfig)
        _subnet_tee_enabled = False
        if self._subnet_config_client is not None:
            try:
                _subnet_tee_enabled = self._subnet_config_client.is_tee_enabled_on_subnet()
            except Exception:
                pass
        if not _subnet_tee_enabled:
            # TEE disabled on subnet — treat all miners as non-TEE (use ZK proofs)
            for miner in self._epoch_miners:
                if getattr(miner, "tee_enabled", False):
                    bt.logging.info(
                        f"TEE disabled on subnet — forcing ZK mode for {miner.address[:10]}"
                    )
                    miner.tee_enabled = False
                    miner.tee_platform = ""

        # Plan canary tests (probation entries get 100% proof verification)
        self._canary_scheduler = CanaryScheduler(
            epoch_number=epoch_number,
            epoch_start_block=epoch_start_block,
            epoch_blocks=self.config.epoch_blocks,
            validator_hotkey=self._wallet.hotkey.ss58_address,
            validator_seed=self._validator_private_key,
            small_count=self.config.canary_small_count,
            full_context_count=self.config.canary_full_context_count,
            proof_sample_rate=self.config.canary_proof_sample_rate,
            probation_entries={
                (addr, idx)
                for addr, indices in self._db.get_probation_addresses().items()
                for idx in indices
            },
        )
        tests = self._canary_scheduler.plan_epoch(self._epoch_miners)

        elapsed = time.monotonic() - t0
        _unique_miners = len({m.address for m in self._epoch_miners})
        _unique_endpoints = len(self._epoch_miners)
        bt.logging.info(f"Epoch {epoch_number}: planned {len(tests)} canary tests for {_unique_miners} miners ({_unique_endpoints} endpoints) ({elapsed:.1f}s)")
        # Per-miner-endpoint summary (much more readable than per-test lines)
        _summary: dict[tuple, dict] = {}
        for t in tests:
            key = (t.miner_address[:10], t.model_id, t.model_index, t.miner_endpoint)
            if key not in _summary:
                _summary[key] = {"small": 0, "full_context": 0, "proof": 0}
            _summary[key][t.test_type] = _summary[key].get(t.test_type, 0) + 1
            if t.verify_proof:
                _summary[key]["proof"] += 1
        for (addr, model, idx, ep), counts in _summary.items():
            bt.logging.debug(
                f"  {addr} idx={idx} model={model} endpoint={ep} "
                f"— {counts.get('small', 0)} small + {counts.get('full_context', 0)} full, "
                f"{counts['proof']} with proof"
            )

        # Update shared state so the proxy knows the current epoch immediately.
        # Without this, the proxy uses the stale epoch from the previous close
        # and organic receipts get tagged with the wrong epoch number.
        self._write_shared_state()

    def _dispatch_canary_tests(self, tests: List[CanaryTest]):
        """Dispatch canary tests to executor — non-blocking; errors via done-callback."""
        def _on_done(test, future):
            try:
                future.result()
            except Exception as e:
                bt.logging.info(
                    f"Canary test failed for {test.miner_address[:10]} model={test.model_id}: {e}"
                )

        for test in tests:
            future = self._executor.submit(
                self._execute_canary_test, test, self._current_epoch,
            )
            future.add_done_callback(lambda f, t=test: _on_done(t, f))

    def _execute_canary_test(self, test: CanaryTest, epoch_number: int):
        """Execute a single canary test: inference + optional proof verify + push receipt.

        Runs in a thread pool worker.  Uses POST /chat (same endpoint as
        organic proxy traffic) with randomized enable_thinking so miners
        cannot fingerprint canaries.

        Retries up to 3 times on HTTP 503 (miner busy) with exponential
        backoff.  A legitimately busy miner frees up in seconds; a fake-503
        evader would have to refuse every retry across the entire epoch.

        Additionally retries once on transport errors (RemoteProtocolError,
        ReadError/Timeout, ConnectError/Timeout, WriteError) — a single TCP
        glitch mid-stream on the full_context canary otherwise loses a
        heavy receipt and craters the per-epoch throughput² score.
        """
        import httpx as _httpx

        transport_retry_exc = (
            _httpx.RemoteProtocolError,
            _httpx.ReadError,
            _httpx.ReadTimeout,
            _httpx.WriteError,
            _httpx.ConnectError,
            _httpx.ConnectTimeout,
        )

        if not self._canary_epoch_active(epoch_number):
            bt.logging.debug(
                f"Skipping stale canary for {test.miner_address[:10]} "
                f"model_index={test.model_index}: test_epoch={epoch_number}, "
                f"current_epoch={self._current_epoch}"
            )
            return

        key = self._miner_model_key(test.miner_address, test.model_index)
        if self._requeue_capacity_audit_canary(
            test,
            epoch_number,
            phase="pre_start",
        ):
            return

        self._mark_canary_started(epoch_number, key)

        try:
            max_retries = 3
            last_exc = None
            for attempt in range(max_retries + 1):
                if not self._canary_epoch_active(epoch_number):
                    bt.logging.debug(
                        f"Aborting stale canary retry for {test.miner_address[:10]} "
                        f"model_index={test.model_index}: test_epoch={epoch_number}, "
                        f"current_epoch={self._current_epoch}"
                    )
                    return
                if self._requeue_capacity_audit_canary(
                    test,
                    epoch_number,
                    phase=f"retry_{attempt}",
                ):
                    self._decrement_expected_receipt(epoch_number, key)
                    return
                try:
                    return self._execute_canary_test_once(
                        test, epoch_number, _transport_retry_allowed=True,
                    )
                except _httpx.HTTPStatusError as e:
                    if e.response.status_code == 503 and attempt < max_retries:
                        wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                        bt.logging.info(f"Canary retry {attempt + 1}/{max_retries} (miner busy) for {test.miner_address[:10]}, waiting {wait}s")
                        import time as _time
                        _time.sleep(wait)
                        last_exc = e
                        continue
                    raise  # non-503 — let _execute_canary_test_once handle it
                except transport_retry_exc as e:
                    # One-shot transport retry — call inner again with
                    # _transport_retry_allowed=False so a second failure falls
                    # through to its normal error-bookkeeping path.
                    _uid_tr = self._db.get_uid(test.miner_address)
                    _uid_tr_s = f"UID {_uid_tr}" if _uid_tr is not None else "UID ?"
                    bt.logging.info(
                        f"Canary transport retry for {_uid_tr_s} {test.miner_address[:10]} "
                        f"(type={test.test_type}): {type(e).__name__} — waiting 5s"
                    )
                    import time as _time
                    _time.sleep(5)
                    return self._execute_canary_test_once(
                        test, epoch_number, _transport_retry_allowed=False,
                    )
                except Exception:
                    raise  # non-HTTP errors — let inner handler deal with it

            # All retries exhausted on 503 — handle as busy skip
            if last_exc is not None:
                if not self._canary_epoch_active(epoch_number):
                    return
                if self._is_capacity_audit_http_503(last_exc):
                    if self._requeue_capacity_audit_gate_canary(
                        test,
                        epoch_number,
                        phase="retry_exhausted",
                    ):
                        self._decrement_expected_receipt(epoch_number, key)
                        return
                bt.logging.info(f"Canary failed after {max_retries} retries (miner busy) for {test.miner_address[:10]} model={test.model_id}")
                reject_ts = int(time.time())
                self._decrement_expected_receipt(epoch_number, key)
                self._busy_skips[key] = self._busy_skips.get(key, 0) + 1
                # Record 503 timestamp for temporal overlap check at epoch close
                self._busy_skip_probations.setdefault(key, []).append(reject_ts)
                if self._busy_skips[key] > 3:
                    # Logged only — penalty is deferred to epoch close where
                    # organic receipts can prove the miner was genuinely busy.
                    bt.logging.info(f"Miner {test.miner_address[:10]} model_index={test.model_index} returned 503 on {self._busy_skips[key]} canaries (>3 after retries) — will evaluate at epoch close")
        finally:
            self._mark_canary_finished(epoch_number, key)

    def _execute_canary_test_once(
        self,
        test: CanaryTest,
        epoch_number: int,
        _transport_retry_allowed: bool = False,
    ):
        """Single attempt at a canary test (called by _execute_canary_test).

        If ``_transport_retry_allowed`` is True, transport-level exceptions
        (connection reset, incomplete read, connect timeout, etc.) are
        re-raised to the outer wrapper so it can retry once.  Otherwise
        all exceptions are caught and recorded as canary errors.
        """
        import httpx as _httpx
        _transport_exc = (
            _httpx.RemoteProtocolError,
            _httpx.ReadError,
            _httpx.ReadTimeout,
            _httpx.WriteError,
            _httpx.ConnectError,
            _httpx.ConnectTimeout,
        )
        verification_config = Config(block_size=256, spot_checks=25)

        try:
            if not self._canary_epoch_active(epoch_number):
                bt.logging.debug(
                    f"Skipping stale canary before inference for {test.miner_address[:10]} "
                    f"model_index={test.model_index}: test_epoch={epoch_number}, "
                    f"current_epoch={self._current_epoch}"
                )
                return
            key = (test.miner_address.lower(), int(test.model_index))
            if self._requeue_capacity_audit_canary(
                test,
                epoch_number,
                phase="pre_http",
            ):
                self._decrement_expected_receipt(epoch_number, key)
                return

            with ValidatorClient(
                miner_url=test.miner_endpoint,
                config=verification_config,
                timeout=(
                    self.config.canary_full_context_inference_timeout
                    if test.test_type == "full_context"
                    else self.config.canary_inference_timeout
                ),
                verify_tls=False,
                chain_config=self.config if not self.config.mock else None,
                model_id=test.model_id,
                validator_hotkey_ss58=self._validator_hotkey_ss58,
                validator_seed=self._validator_private_key,
            ) as client:
                # Run inference through /chat (same endpoint as organic traffic)
                do_sample = test.temperature > 0
                temperature = test.temperature
                # Sampling proof verification rate for canaries:
                #   - greedy (temp=0):  100% — existing argmax sampling proof
                #   - sampled (temp>0): 100% — canonical replay path
                # Both paths are end-to-end verified at canary time so the
                # validator catches mode-conditional cheating where a miner
                # serves honestly for greedy but biases the distribution for
                # sampled requests.
                sampling_bps = 10_000
                enable_thinking = test.enable_thinking
                messages = [{"role": "user", "content": test.prompt}]
                full_text, commitment, proof_bundle, nonce, timing = client.run_chat(
                    messages=messages,
                    max_new_tokens=test.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    sampling_verification_bps=sampling_bps,
                    enable_thinking=enable_thinking,
                    presence_penalty=test.presence_penalty,
                    top_k=test.top_k,
                    top_p=test.top_p,
                )

                if not self._canary_epoch_active(epoch_number):
                    bt.logging.debug(
                        f"Dropping stale canary result for {test.miner_address[:10]} "
                        f"model_index={test.model_index}: test_epoch={epoch_number}, "
                        f"current_epoch={self._current_epoch}"
                    )
                    return

                # Optional proof verification
                proof_verified = False
                proof_failure_reason = None
                verify_timing = {}
                if test.verify_proof:
                    try:
                        # Use epoch-cached ModelSpec (fetched once at epoch start)
                        # instead of client.fetch_model_spec() which makes an RPC
                        # call per canary. All miners serving the same model_id
                        # share the same spec (weight Merkle roots from chain).
                        cached_spec = self._model_spec_cache.get(test.model_id)
                        if cached_spec is not None:
                            # Tokenizer drift short-circuit: validator-side
                            # tokenizer doesn't match the on-chain anchor.
                            # Skip verification and DO NOT penalize the miner.
                            if getattr(cached_spec, "_tokenizer_drift", False):
                                bt.logging.warning(
                                    f"Skipping proof verification for {test.model_id} "
                                    "due to tokenizer drift (validator-side issue, "
                                    "miner not penalized)"
                                )
                                test.verify_proof = False
                                raise RuntimeError("tokenizer drift, validator-side")
                            client.model_spec = cached_spec
                            client._auto_configure_from_spec(cached_spec)
                        else:
                            # No cached spec — skip proof verification entirely.
                            # NEVER make an RPC call here; it 429s and then gets
                            # logged as [PROOF ERROR] which looks like a miner issue.
                            bt.logging.warning(f"No cached ModelSpec for {test.model_id} — skipping proof verification (validator-side issue, not miner fault)")
                            test.verify_proof = False
                            raise RuntimeError("ModelSpec cache miss — skipping verification")
                        # Compute expected input commitment from the prompt
                        # we sent — prevents miner from truncating input.
                        expected_ic = _compute_expected_input_commitment(
                            test.model_id, test.prompt,
                            enable_thinking=enable_thinking,
                        )
                        # Compute expected prompt_hash from the messages we sent.
                        import json as _json
                        _canary_messages = [{"role": "user", "content": test.prompt}]
                        _canary_hash_input = _json.dumps(
                            _canary_messages, sort_keys=True, ensure_ascii=False,
                        ).encode()
                        expected_ph = hashlib.sha256(_canary_hash_input).digest()
                        _committed_ph = commitment.prompt_hash.hex()[:16] if commitment.prompt_hash else "None"
                        bt.logging.debug(
                            f"Canary expected_ph: {expected_ph.hex()[:16]} (len={len(_canary_hash_input)}, first_100={_canary_hash_input[:100].decode(errors='replace')}), "
                            f"committed_ph: {_committed_ph}"
                        )
                        # Compute expected sampler config hash from the
                        # canary's sampling params.  Bound separately from
                        # tokenizer_hash (which is on-chain, validator-startup).
                        from verallm.sampling import compute_sampler_config_hash as _scfg
                        # Canaries send explicit presence_penalty (see canary.py);
                        # bind it into the expected hash so the miner can't swap it.
                        _expected_pp = float(getattr(test, "presence_penalty", 0.0) or 0.0)
                        _expected_scfg = _scfg(
                            top_k=int(test.top_k or -1),
                            top_p=float(test.top_p or 1.0),
                            min_p=float(getattr(test, 'min_p', 0.0) or 0.0),
                            presence_penalty=_expected_pp,
                        )
                        result, verify_timing = client.verify_proof(
                            proof_bundle, nonce,
                            expected_sampling_verification_bps=sampling_bps,
                            expected_do_sample=do_sample,
                            expected_temperature=temperature,
                            enable_thinking=enable_thinking,
                            expected_input_commitment=expected_ic,
                            expected_prompt_hash=expected_ph,
                            expected_sampler_config_hash=_expected_scfg,
                            # Pass raw sampling params so the canonical
                            # replay (high_assurance) can use
                            # them.  Bound separately by sampler_config_hash.
                            expected_top_k=int(test.top_k or -1),
                            expected_top_p=float(test.top_p or 1.0),
                            expected_min_p=float(getattr(test, "min_p", 0.0) or 0.0),
                        )
                        proof_verified = result.passed
                        if not proof_verified:
                            proof_failure_reason = result.message
                            # Miner-side fault, not a validator problem — the
                            # detection + probation flow is the success path.
                            # DEBUG so prod dashboards don't page on miner faults.
                            # UID resolved via local SQLite (indexed lookup,
                            # microseconds) — no RPC.
                            _uid = self._db.get_uid(test.miner_address)
                            _uid_str = f"uid={_uid}" if _uid is not None else "uid=?"
                            bt.logging.debug(
                                f"Proof failure | {_uid_str} {test.miner_address[:10]} "
                                f"| model={test.model_id}: {result.message}",
                            )
                            # Mid-epoch cutoff: immediately put on probation
                            # and update shared state so proxy stops routing
                            self._on_proof_failure(
                                test.miner_address, test.model_index,
                                endpoint=test.miner_endpoint,
                            )
                    except Exception as e:
                        bt.logging.warning(f"Proof verification skipped (validator-side error, NOT miner fault): {test.miner_address[:10]} — {e}")
                        # Verification errored (e.g. RPC 429, cache miss).
                        # This is NOT the miner's fault — mark as "not tested"
                        # so the receipt doesn't count as a proof failure.
                        test.verify_proof = False

                # ── TEE attestation verification ──────────────────────────
                tee_attestation_verified = None  # None = not tested (non-TEE miner)
                if test.verify_tee:
                    try:
                        # 1. Fetch miner's /tee/info endpoint
                        tee_info_url = f"{test.miner_endpoint.rstrip('/')}/tee/info"
                        tee_resp = httpx.get(tee_info_url, timeout=15.0, verify=False)
                        tee_resp.raise_for_status()
                        tee_info = tee_resp.json()

                        from verallm.tee.serialization import dict_to_attestation
                        from verallm.tee.attestation import get_attestation_provider

                        miner_attestation = dict_to_attestation(tee_info["attestation"])
                        enclave_pubkey = bytes.fromhex(tee_info["enclave_public_key"])
                        platform = tee_info.get("platform", miner_attestation.platform)

                        # 2. Get on-chain TEE capability from MinerRegistry
                        on_chain_cap = self._miner_client.get_tee_capability(test.miner_address)
                        if not on_chain_cap.enabled:
                            bt.logging.info(f"TEE verify: miner {test.miner_address[:10]} not TEE-enabled on-chain")
                            raise RuntimeError("Miner not TEE-enabled on-chain")

                        # 3. Verify report_data binding: SHA256(pubkey || weight_hash)
                        weight_hash = on_chain_cap.model_weight_hash or miner_attestation.model_weight_hash
                        expected_report_data = hashlib.sha256(enclave_pubkey + weight_hash).digest()
                        if miner_attestation.report_data != expected_report_data:
                            bt.logging.error(f"TEE verify: report_data binding mismatch for {test.miner_address[:10]}")
                            raise RuntimeError("report_data binding mismatch")

                        # 4. Verify keccak256(report) matches on-chain attestation_hash
                        from web3 import Web3
                        report_hash = Web3.keccak(miner_attestation.attestation_report)
                        if on_chain_cap.attestation_hash and report_hash != on_chain_cap.attestation_hash:
                            bt.logging.error(f"TEE verify: attestation_hash mismatch for {test.miner_address[:10]}")
                            raise RuntimeError("attestation_hash mismatch with on-chain value")

                        # 5. Hardware signature check via platform-specific provider
                        if platform == "mock":
                            _allow_mock = (
                                getattr(self.config, "subtensor_network", "") == "test"
                                or getattr(self.config, "mock", False)
                                or getattr(self.config, "allow_mock_tee", False)
                            )
                            if not _allow_mock:
                                bt.logging.error(f"TEE verify: mock platform rejected on mainnet for {test.miner_address[:10]}")
                                raise RuntimeError("Mock TEE platform not allowed on mainnet (use --allow-mock-tee to override)")
                            bt.logging.debug(f"TEE verify: mock platform — auto-pass for {test.miner_address[:10]}")
                        else:
                            provider = get_attestation_provider(platform)
                            if not provider.verify_attestation(miner_attestation, expected_model_weight_hash=weight_hash):
                                bt.logging.error(f"TEE verify: hardware attestation failed for {test.miner_address[:10]}")
                                raise RuntimeError("Hardware attestation verification failed")

                        # 6. Re-attestation challenge with random nonce (signed request)
                        import os as _os
                        import json as _json_tee
                        from neurons.request_signing import sign_request as _sign_req
                        challenge_nonce = _os.urandom(32)
                        reattest_url = f"{test.miner_endpoint.rstrip('/')}/tee/reattest"
                        _reattest_body = _json_tee.dumps({"nonce": challenge_nonce.hex()}).encode()
                        _reattest_headers = _sign_req(
                            method="POST", path="/tee/reattest", body=_reattest_body,
                            hotkey_ss58=self._validator_hotkey_ss58,
                            hotkey_seed=self._validator_private_key,
                        )
                        _reattest_headers["Content-Type"] = "application/json"
                        reattest_resp = httpx.post(
                            reattest_url,
                            content=_reattest_body,
                            headers=_reattest_headers,
                            timeout=30.0,
                            verify=False,
                        )
                        reattest_resp.raise_for_status()
                        reattest_data = reattest_resp.json()
                        # /tee/reattest returns attestation_to_dict() directly (not wrapped)
                        fresh_att = dict_to_attestation(reattest_data)

                        # Verify nonce binding: report_data == SHA256(pubkey || weight_hash || nonce)
                        expected_nonce_rd = hashlib.sha256(enclave_pubkey + weight_hash + challenge_nonce).digest()
                        if fresh_att.report_data != expected_nonce_rd:
                            bt.logging.error(f"TEE verify: re-attestation nonce binding failed for {test.miner_address[:10]}")
                            raise RuntimeError("Re-attestation nonce binding mismatch")

                        # 7. Code measurement verification (MRTD allowlist)
                        _verify_code_measurement(
                            platform=platform,
                            on_chain_cap=on_chain_cap,
                            fresh_attestation_report=fresh_att.attestation_report,
                            subnet_config_client=self._subnet_config_client,
                            miner_label=test.miner_address[:10],
                        )

                        tee_attestation_verified = True
                        bt.logging.info(f"TEE attestation verified for {test.miner_address[:10]} (platform={platform})")

                    except Exception as e:
                        bt.logging.info(f"TEE attestation verification failed for {test.miner_address[:10]}: {e}")
                        tee_attestation_verified = False

                if not self._canary_epoch_active(epoch_number):
                    bt.logging.debug(
                        f"Not pushing stale canary receipt for {test.miner_address[:10]} "
                        f"model_index={test.model_index}: test_epoch={epoch_number}, "
                        f"current_epoch={self._current_epoch}"
                    )
                    return

                # Extract metrics from timing
                ttft_ms = timing.get("ttft_ms", 0.0)
                output_tokens = timing.get("output_tokens", 0)
                input_tokens = timing.get("input_tokens", 0)
                observed_start_ts, observed_end_ts, inference_ms = validator_observed_timing(timing)
                tokens_per_sec = (
                    output_tokens / (inference_ms / 1000)
                    if inference_ms > 0 and output_tokens > 0
                    else 0.0
                )

                # On push failure, drop the expected count so integrity isn't penalized.
                pushed_ok = self._push_receipt_to_miner(
                    miner_address=test.miner_address,
                    miner_endpoint=test.miner_endpoint,
                    model_id=test.model_id,
                    model_index=test.model_index,
                    epoch_number=epoch_number,
                    commitment_hash=commitment.commitment_hash(),
                    ttft_ms=ttft_ms,
                    tokens_generated=output_tokens,
                    generation_time_ms=inference_ms,
                    tokens_per_sec=tokens_per_sec,
                    prompt_tokens=input_tokens,
                    proof_verified=proof_verified,
                    proof_requested=test.verify_proof,
                    tee_attestation_verified=tee_attestation_verified,
                    is_canary=True,
                    timestamp=int(observed_end_ts),
                    observed_start_ts=observed_start_ts,
                    observed_end_ts=observed_end_ts,
                )
                if not pushed_ok:
                    _key_pf = (test.miner_address, test.model_index)
                    self._decrement_expected_receipt(epoch_number, _key_pf)

                _uid_c = self._db.get_uid(test.miner_address)
                _uid_cs = f"UID {_uid_c}" if _uid_c is not None else "UID ?"
                bt.logging.debug(f"Canary {_uid_cs} {test.miner_address[:10]} {test.model_id}/{test.model_index}: type={test.test_type} ttft={ttft_ms:.0f}ms tps={tokens_per_sec:.1f} proof={proof_verified}")

                # Log to analytics DB
                try:
                    _uid = self._db.get_uid(test.miner_address)
                    self._db.log_canary_result(
                        network=self.config.subtensor_network or "unknown",
                        chain_id=getattr(self.config, "chain_id", 0),
                        netuid=self.config.netuid,
                        epoch_number=epoch_number,
                        block_number=self._last_known_block or 0,
                        miner_address=test.miner_address,
                        miner_uid=_uid,
                        miner_hotkey_ss58=self._get_miner_ss58(test.miner_address, "hotkey"),
                        miner_coldkey_ss58=self._get_miner_ss58(test.miner_address, "coldkey"),
                        model_id=test.model_id,
                        model_index=test.model_index,
                        endpoint=test.miner_endpoint,
                        test_type=test.test_type,
                        test_index=test.test_index,
                        proof_requested=1 if test.verify_proof else 0,
                        tee_requested=1 if test.verify_tee else 0,
                        tee_verified=1 if tee_attestation_verified else (0 if test.verify_tee else None),
                        enable_thinking=1 if test.enable_thinking else 0,
                        temperature=test.temperature,
                        max_new_tokens=test.max_new_tokens,
                        status="ok",
                        ttft_ms=ttft_ms,
                        tokens_generated=output_tokens,
                        inference_ms=inference_ms,
                        tokens_per_sec=tokens_per_sec,
                        prompt_tokens=input_tokens,
                        proof_verified=1 if proof_verified else (0 if test.verify_proof else None),
                        proof_failure_reason=proof_failure_reason,
                        prove_ms=timing.get("prove_ms"),
                        commitment_ms=timing.get("commitment_ms"),
                        verify_ms=sum(verify_timing.values()) if verify_timing else None,
                        commitment_hash=commitment.commitment_hash().hex() if commitment else None,
                        receipt_pushed=1 if pushed_ok else 0,
                    )
                except Exception as _db_err:
                    bt.logging.debug(f"Failed to log canary result: {_db_err}")

        except Exception as e:
            if not self._canary_epoch_active(epoch_number):
                bt.logging.debug(
                    f"Ignoring stale canary error for {test.miner_address[:10]} "
                    f"model_index={test.model_index}: test_epoch={epoch_number}, "
                    f"current_epoch={self._current_epoch}: {e}"
                )
                return
            # HTTP 503 (miner busy) is handled by the retry wrapper in
            # _execute_canary_test — if we get here it's a real error
            # (connection refused, timeout, non-503 HTTP error, etc.).
            # If this is a transport-level error AND the outer wrapper
            # hasn't retried yet, re-raise so it can retry once.
            if _transport_retry_allowed and isinstance(e, _transport_exc):
                raise
            if _transport_retry_allowed and self._is_http_503(e):
                raise
            # Validator-side sqlite3 errors must NEVER be attributed to the
            # miner.  A cross-thread Connection race ("bad parameter or
            # other API misuse") inside any of our DB helpers would
            # otherwise be logged as a miner canary failure and count
            # toward the >3-errors probation threshold.
            import sqlite3 as _sqlite3
            if isinstance(e, _sqlite3.Error):
                bt.logging.warning(
                    f"Validator-side DB error during canary execution "
                    f"(NOT attributed to miner {test.miner_address[:10]} "
                    f"model={test.model_id}): {type(e).__name__}: {e}"
                )
                return
            _uid_err = self._db.get_uid(test.miner_address)
            _uid_err_s = f"UID {_uid_err}" if _uid_err is not None else "UID ?"
            _err_msg = str(e).split("\nFor more information")[0]
            bt.logging.info(f"Canary test error for {_uid_err_s} {test.miner_address[:10]} model={test.model_id}: {_err_msg}")

            # Log error to analytics DB
            try:
                self._db.log_canary_result(
                    network=self.config.subtensor_network or "unknown",
                    chain_id=getattr(self.config, "chain_id", 0),
                    netuid=self.config.netuid,
                    epoch_number=epoch_number,
                    block_number=self._last_known_block or 0,
                    miner_address=test.miner_address,
                    miner_hotkey_ss58=self._get_miner_ss58(test.miner_address, "hotkey"),
                    miner_coldkey_ss58=self._get_miner_ss58(test.miner_address, "coldkey"),
                    model_id=test.model_id,
                    model_index=test.model_index,
                    endpoint=test.miner_endpoint,
                    test_type=test.test_type,
                    test_index=test.test_index,
                    proof_requested=1 if test.verify_proof else 0,
                    enable_thinking=1 if test.enable_thinking else 0,
                    temperature=test.temperature,
                    max_new_tokens=test.max_new_tokens,
                    status="error",
                    error_message=str(e)[:500],
                )
            except Exception as _db_err:
                bt.logging.debug(f"Failed to log canary error: {_db_err}")
                pass  # Non-fatal

            # Track transient error per (miner, model_index) — do NOT
            # report offline on a single failure.  Decrement expected receipt
            # count so the receipt integrity check at epoch close tolerates
            # the missing receipt (same treatment as 503 busy-skips).
            # Repeated failures (>3 per epoch) are evaluated at epoch close,
            # with the error timestamps used to find overlapping organic
            # receipts (same forgiveness mechanism as the 503 busy-skip path).
            key = self._miner_model_key(test.miner_address, test.model_index)
            self._canary_errors[key] = self._canary_errors.get(key, 0) + 1
            self._canary_error_times.setdefault(key, []).append(int(time.time()))
            self._decrement_expected_receipt(epoch_number, key)
            if self._canary_errors[key] > 3:
                bt.logging.info(
                    f"Miner {test.miner_address[:10]} model_index={test.model_index} "
                    f"has {self._canary_errors[key]} canary errors (>3) — will evaluate at epoch close"
                )

    def _try_close_epoch(self, epoch_number: int):
        """Attempt epoch close with exponential backoff on failure.

        On success, clears pending state and resets backoff.
        On failure (e.g. 429 rate limit), schedules retry with increasing delay
        to avoid hammering the RPC.
        """
        # Guard: never close the same epoch twice — each re-close blends
        # another score into the EMA, destroying it.
        if not hasattr(self, '_last_closed_epoch'):
            self._last_closed_epoch = -1
        if epoch_number <= self._last_closed_epoch:
            self._pending_epoch_close = None
            return

        now = time.monotonic()
        if now < self._epoch_close_retry_after:
            return  # Still in cooldown from a previous failure

        try:
            self._closing_inflight_canaries[epoch_number] = dict(
                self._inflight_canaries.get(epoch_number, {})
            )
            try:
                self._close_epoch(epoch_number)
            finally:
                self._closing_inflight_canaries.pop(epoch_number, None)
            self._pending_epoch_close = None
            self._last_closed_epoch = epoch_number
            self._epoch_close_backoff = 30.0  # Reset on success
            # If auto-update was deferred, apply it now (between epochs)
            if self._auto_updater is not None:
                self._auto_updater.notify_not_busy()
        except Exception as e:
            self._epoch_close_retry_after = now + self._epoch_close_backoff
            bt.logging.warning(f"Epoch {epoch_number} close failed, retrying in {self._epoch_close_backoff:.0f}s: {e}")
            self._epoch_close_backoff = min(self._epoch_close_backoff * 2, 300)

    # Epoch duration in seconds — 360 blocks × 12s = 4320s ≈ 72 min.
    # Used by the restart-forgiveness window (2 epochs).
    _EPOCH_DURATION_SEC = 360 * 12

    def _poll_remote_miner_version(self) -> None:
        """Detect a remote ``miner_version`` bump and open a forgiveness window.

        Called once per ``_close_epoch``.  When the remote (public-repo)
        ``miner_version`` is higher than the value we last saw, record the
        timestamp and clear the per-miner forgiveness ledger.  For the next
        ``2 * _EPOCH_DURATION_SEC`` seconds, miners that would otherwise be
        sent to probation for canary errors / capability failure get a
        one-shot pass — exactly the window during which legit auto-update
        restarts cause `ConnectError`/`ConnectTimeout` canaries.

        Failures (no git remote, fetch error, parse error) are silent —
        same conservative behaviour as the existing auto-update path.
        """
        from neurons.auto_update import fetch_origin, get_remote_version
        from neurons.version import miner_version as _local_miner_version

        # Rate-limit the git fetch: epoch close is already a hot path
        # (receipt pulls, scoring, weight set).  The forgiveness window is
        # 2 epochs = ~144 min — 10-min staleness on the bump signal is
        # invisible to miners.
        now = time.time()
        if now - self._miner_version_last_check < 600:
            return
        self._miner_version_last_check = now

        if not fetch_origin():
            return
        remote = get_remote_version("miner")
        if remote is None:
            return
        # Seed the baseline lazily — on first call after start-up, treat the
        # local installed version as "already seen" so a validator that boots
        # AFTER the bump landed in public doesn't grant blanket forgiveness.
        if self._miner_version_last_seen == 0:
            self._miner_version_last_seen = max(remote, _local_miner_version)
            return
        if remote > self._miner_version_last_seen:
            bt.logging.info(
                f"Remote miner_version bumped {self._miner_version_last_seen} "
                f"→ {remote}; opening restart-forgiveness window for "
                f"{2 * self._EPOCH_DURATION_SEC}s"
            )
            self._miner_version_last_seen = remote
            self._miner_version_bump_at = time.time()
            self._restart_forgiven.clear()

    def _restart_window_grants_pass(self, key: Tuple[str, int]) -> bool:
        """Return True iff the miner gets a one-shot pass right now.

        Combines the two predicates: are we still inside the 2-epoch window
        after the most recent miner_version bump, AND has this miner not
        already used its single forgiveness ticket?  Caller is responsible
        for adding ``key`` to ``self._restart_forgiven`` after granting.
        """
        if self._miner_version_bump_at <= 0:
            return False
        if time.time() - self._miner_version_bump_at >= 2 * self._EPOCH_DURATION_SEC:
            return False
        return key not in self._restart_forgiven

    def _close_epoch(self, epoch_number: int):
        """Close an epoch: pull receipts from all miners, score, update EMAs.

        Two-pass approach:
        1. Pull all receipts from all miners.
        2. Compute per-model demand from organic traffic.
        3. Score each miner-model entry with demand bonus applied.
        4. Post demand scores on-chain.
        """
        t0 = time.monotonic()
        bt.logging.info(
            f"Epoch {epoch_number} closing: pulling receipts from {len(self._epoch_miners)} miners",
        )

        # Detect a remote miner_version bump and (re-)open the restart-
        # forgiveness window before scoring this epoch's canary errors.
        try:
            self._poll_remote_miner_version()
        except Exception as e:
            bt.logging.debug(f"miner_version poll failed: {e}")

        runtime_config_loaded = self._refresh_subnet_runtime_config(
            current_epoch=epoch_number,
            force=True,
        )

        # ── Build the validator authority snapshot for receipt verification.
        # Done ONCE per epoch close so the per-receipt loop is pure dict +
        # array access (no RPC, no metagraph rebuild).
        #
        # 1. Force a fresh metagraph fetch at the epoch boundary so we don't
        #    miss validators that registered in the last 0–4 minutes.
        #    Falls back to the last cached metagraph if Substrate is down.
        # 2. Build ss58→uid in O(N), replacing per-receipt mg.hotkeys.index().
        # 3. Read minValidatorStake() once (cheap eth_call), cache for next
        #    epoch's fallback if RPC is briefly down.
        try:
            mg = self._subtensor.metagraph(self.config.netuid)
            self._metagraph = mg
        except Exception as e:
            bt.logging.warning(
                f"Epoch {epoch_number}: metagraph refresh failed, using last cached: {e}"
            )
            mg = self._metagraph

        receipt_authority: ValidatorAuthority | None = None
        if mg is not None:
            try:
                ss58_to_uid = {hk: i for i, hk in enumerate(mg.hotkeys)}
                permits = list(mg.validator_permit) if hasattr(mg, "validator_permit") else []

                # mg.S = chain's effective subnet stake (tao_weight * tao_stake
                # + alpha_stake).  See ValidatorAuthority docstring for why
                # we use total here instead of alpha_stake.
                #
                # TODO: minValidatorStake on chain stays at 0 (must — otherwise
                # root validators with no alpha can't register).  That makes
                # this stake check a functional no-op, leaving validator_permit
                # as the actual gate.  If a stricter receipt-side filter is
                # ever needed, add a dedicated SubnetConfig field rather than
                # raising minValidatorStake (which would break root-only valis).
                stakes_src = mg.S if hasattr(mg, "S") else getattr(mg, "stake", [])
                stakes = [float(s) for s in stakes_src]

                try:
                    from verallm.chain.validator_registry import ValidatorRegistryClient
                    vr = ValidatorRegistryClient(self.config)
                    min_stake_rao = vr.get_min_validator_stake()
                    self._cached_min_validator_stake = min_stake_rao / 1e9
                except Exception as e:
                    bt.logging.debug(
                        f"minValidatorStake read failed, using last cached "
                        f"({self._cached_min_validator_stake}): {e}"
                    )

                receipt_authority = ValidatorAuthority(
                    ss58_to_uid=ss58_to_uid,
                    validator_permit=permits,
                    stakes=stakes,
                    min_stake=self._cached_min_validator_stake,
                )
                bt.logging.debug(
                    f"Epoch {epoch_number} authority: {len(ss58_to_uid)} hotkeys, "
                    f"min_stake={self._cached_min_validator_stake:.2f}"
                )
            except Exception as e:
                bt.logging.warning(
                    f"Epoch {epoch_number}: failed to build receipt authority: {e}. "
                    f"All receipts will be rejected this epoch (existing EMAs decay)."
                )

        # ── Pass 1: collect all receipts ──────────────────────────
        self._receipt_pull_failed_keys: Set[Tuple[str, int]] = set()
        miner_receipts, all_epoch_receipts = self._collect_epoch_receipts(
            epoch_number, receipt_authority,
        )

        # ── Store all receipts (full network view) ────────────────
        try:
            stored = self._db.log_network_receipts(
                all_epoch_receipts,
                own_hotkey=self._validator_hotkey_bytes,
                network=self.config.subtensor_network or "unknown",
                netuid=self.config.netuid,
                ss58_lookup=self._ss58_cache,
            )
            bt.logging.info(f"Epoch {epoch_number}: stored {stored} network receipts ({len(all_epoch_receipts)} total)")
        except Exception as e:
            bt.logging.debug(f"Failed to store network receipts: {e}")

        # ── Compute per-model demand ──────────────────────────────
        demand_scores: Dict[str, int] = {}
        if self.config.demand_bonus_enabled:
            demand_scores = compute_model_demand(all_epoch_receipts, epoch_number)
            if demand_scores:
                bt.logging.info(f"Epoch {epoch_number} demand scores: {{k: v for k, v in sorted(demand_scores.items(), key=lambda x: -x[1])[:5]}}")
        # Stash for shared state (proxy serves these via /v1/network/stats)
        self._last_demand_scores = demand_scores

        # ── Compute per-model peer medians (TTFT + decode speed) ─
        peer_medians_by_model = compute_peer_medians(
            all_epoch_receipts, epoch_number,
        )
        if peer_medians_by_model:
            _pm_summary = {k: f"ttft={v.median_ttft_ms:.0f}ms tps={v.median_tps:.1f}"
                 for k, v in sorted(peer_medians_by_model.items())[:5]}
            bt.logging.info(f"Epoch {epoch_number} peer medians: {_pm_summary}")

        # ── Read scoring params ───────────────────────────────────
        # Hosted subnet config is authoritative. Chain scoring is fallback
        # only when no hosted config/cache is usable.
        if runtime_config_loaded:
            bt.logging.debug(
                f"Runtime subnet config scoring: tee={self._scoring.tee_bonus:.2f} "
                f"ema={self._scoring.ema_alpha:.2f} tp={self._scoring.throughput_power:.1f} "
                f"proof_rate={self._scoring.proof_sample_rate:.2f} "
                f"prob_passes={self._scoring.probation_required_passes} "
                f"demand_max={self._scoring.demand_bonus_max:.2f} "
                f"burn={self._scoring.emission_burn:.0%}"
            )
        elif self._subnet_config_client is not None:
            try:
                self._scoring = self._subnet_config_client.get_scoring_params()
                self._last_good_scoring = self._scoring  # cache for fallback
                bt.logging.debug(
                    f"SubnetConfig fallback scoring: tee={self._scoring.tee_bonus:.2f} "
                    f"ema={self._scoring.ema_alpha:.2f} tp={self._scoring.throughput_power:.1f} "
                    f"proof_rate={self._scoring.proof_sample_rate:.2f} "
                    f"prob_passes={self._scoring.probation_required_passes} "
                    f"demand_max={self._scoring.demand_bonus_max:.2f} "
                    f"burn={self._scoring.emission_burn:.0%}"
                )
            except Exception as e:
                if hasattr(self, "_last_good_scoring"):
                    self._scoring = self._last_good_scoring
                    bt.logging.info(f"SubnetConfig read failed, using last-known values (burn={self._scoring.emission_burn:.0%}): {e}")
                else:
                    self._scoring = ScoringParams()  # hardcoded defaults on first-ever failure
                    bt.logging.info(f"SubnetConfig read failed, no cache, using defaults: {e}")
        else:
            if not hasattr(self, "_last_good_scoring"):
                self._scoring = ScoringParams()

        # Update scorer EMA alpha + throughput power from chain
        self.scorer.ema_alpha = self._scoring.ema_alpha
        self.scorer.throughput_power = self._scoring.throughput_power

        self._last_model_emission_budgets = self._build_model_emission_budgets(
            demand_scores,
        )

        # Refresh blacklist from SubnetConfig (parallel RPC per address, cached 5min).
        self._refresh_blacklist({m.address for m in self._epoch_miners})

        # ── GPU UUID dedup: one GPU = one endpoint ─────────────────
        # Build map: gpu_uuid -> list of (address, model_index, ema_score).
        # If any UUID appears on more than one endpoint, keep the highest-
        # scored, skip the rest.  One physical GPU can only serve one endpoint.
        _uuid_endpoints: Dict[str, List[tuple]] = {}
        for m in self._epoch_miners:
            for _uuid in getattr(m, "gpu_uuids", []):
                if not _uuid:
                    continue
                _uid = self._resolve_uid(m.address)
                _ema = 0.0
                if _uid is not None and _uid in self.scorer.states:
                    _st = self.scorer.states[_uid]
                    if m.model_index in _st.entries:
                        _ema = _st.entries[m.model_index].ema_score
                _uuid_endpoints.setdefault(_uuid.lower(), []).append(
                    (m.address.lower(), m.model_index, _ema)
                )

        _sybil_skip: Set[tuple] = set()  # (address, model_index) pairs to skip
        for _uuid, _eps in _uuid_endpoints.items():
            if len(_eps) <= 1:
                continue
            bt.logging.debug(
                f"GPU UUID {_uuid[:16]}... used by {len(_eps)} endpoints — keeping best"
            )
            _eps.sort(key=lambda x: -x[2])  # highest EMA first
            for _addr, _midx, _ema in _eps[1:]:
                _sybil_skip.add((_addr, _midx))
                # Zero stale EMA on the skipped slot. Without this, the
                # slot's score from before the dup was observed stays
                # frozen (probation can't decay it because the scoring
                # pass below `continue`s past the probation block).
                _zuid = self._resolve_uid(_addr)
                if _zuid is not None and _zuid in self.scorer.states:
                    _zentry = self.scorer.states[_zuid].entries.get(_midx)
                    if _zentry is not None and _zentry.ema_score != 0.0:
                        _zold = _zentry.ema_score
                        _zentry.ema_score = 0.0
                        bt.logging.debug(
                            f"  GPU dedup: skipping {_addr[:10]} model_index={_midx} "
                            f"(zeroed stale ema {_zold:.4f} → 0.0)"
                        )
                    else:
                        bt.logging.debug(
                            f"  GPU dedup: skipping {_addr[:10]} model_index={_midx} (ema=0)"
                        )
                else:
                    bt.logging.debug(
                        f"  GPU dedup: skipping {_addr[:10]} model_index={_midx} (ema={_ema:.4f}, uid unresolved)"
                    )

        # ── Pass 2: score each miner-model entry ─────────────────
        for miner in self._epoch_miners:
            if not self._running:
                break

            # GPU UUID dedup — skip endpoints sharing a GPU with a higher-scored one
            if (miner.address.lower(), miner.model_index) in _sybil_skip:
                bt.logging.info(f"Skipping {miner.address[:10]} model_index={miner.model_index} (GPU UUID duplicate)")
                continue

            uid = self._resolve_uid(miner.address)
            if uid is None:
                # Miner-side issue (not registered in metagraph), validator is fine.
                bt.logging.info(f"Cannot resolve UID for {miner.address[:10]}, skipping")
                continue

            model_entry = get_model(miner.model_id)
            if model_entry is None:
                # Miner registered for an unknown model — miner-side config issue.
                bt.logging.info(f"Model {miner.model_id} not in registry, skipping")
                continue

            # Filter receipts to THIS specific model entry (address + model_id
            # + model_index).  A miner can register the same model on multiple
            # endpoints (e.g. multi-GPU), each with its own model_index.
            # Each entry must be scored on its OWN receipts only — not the
            # combined pool from all endpoints.
            all_receipts = [
                r for r in miner_receipts.get(miner.address, [])
                if r.model_id == miner.model_id and r.model_index == miner.model_index
            ]
            own_receipts = [
                r for r in all_receipts
                if r.validator_hotkey == self._validator_hotkey_bytes
            ]

            key = self._miner_model_key(miner.address, miner.model_index)
            # Reconcile stale probation keys: if the miner re-registered
            # (new leaseModel call), the contract array index changes but
            # probation still references the old index.
            self._probation_tracker.migrate_index(
                miner.address, miner.model_index,
                new_endpoint=getattr(miner, 'endpoint', ''),
            )
            # Also migrate in DB (old_index is found automatically inside)
            # DB migrate_probation needs explicit old index; use tracker's side-effect
            # to keep them in sync.
            expected = self._effective_expected_receipts(epoch_number, key)
            self._reconcile_capacity_audit_timing_excuses(
                miner,
                all_receipts,
                epoch_number,
            )
            model_gate_reason = self._capacity_audit_model_gate_reason(
                miner,
                epoch_number,
            )
            audit_score_gate_reason = self._capacity_audit_score_gate_reason(
                miner.address,
                miner.model_index,
                epoch_number,
                uid=uid,
            )

            if key in getattr(self, "_receipt_pull_failed_keys", set()):
                gated = self._apply_capacity_audit_model_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    model_gate_reason,
                )
                gated = self._apply_capacity_audit_score_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    audit_score_gate_reason,
                ) or gated
                if not gated:
                    bt.logging.warning(
                        f"Skipping receipt-based score for {miner.address[:10]} "
                        f"model_index={miner.model_index} at epoch {epoch_number} — "
                        f"receipt pull failed after retries"
                    )
                continue

            # Skip scoring if no canaries were dispatched AND no busy-skips.
            # The expected count is decremented on each 503, so expected==0
            # can mean either "nothing scheduled" or "all canaries got 503".
            # In the latter case we must still run the busy-skip evaluation.
            busy_skips_this_epoch = self._busy_skips.get(key, 0)
            if expected == 0 and busy_skips_this_epoch == 0:
                gated = self._apply_capacity_audit_model_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    model_gate_reason,
                )
                gated = self._apply_capacity_audit_score_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    audit_score_gate_reason,
                ) or gated
                if not gated:
                    bt.logging.info(f"Skipping score for {miner.address[:10]} model_index={miner.model_index} — 0 canaries dispatched")
                continue

            # Count proof verification outcomes from own receipts
            proof_tested = [
                r for r in own_receipts if r.proof_requested
            ]
            proof_failed = [
                r for r in proof_tested if not r.proof_verified
            ]

            # TEE attestation outcomes from own receipts
            tee_tested = [r for r in own_receipts if getattr(r, "tee_attestation_verified", None) is not None]
            tee_failed = [r for r in tee_tested if not r.tee_attestation_verified]

            outcome = EpochOutcome(
                miner_address=miner.address,
                model_id=miner.model_id,
                model_index=miner.model_index,
                uid=uid if uid is not None else -1,
                hotkey_ss58=getattr(miner, "hotkey_ss58", "") or "",
                own_receipts=own_receipts,
                expected_own_receipt_count=expected,
                all_receipts=all_receipts,
                proof_tests=len(proof_tested),
                proof_failures=len(proof_failed),
                tee_tests=len(tee_tested),
                tee_failures=len(tee_failed),
                tee_verified=len(tee_tested) > 0 and len(tee_failed) == 0,
                max_context_len=miner.max_context_len,
                quant=miner.quant,
                busy_skip_count=self._busy_skips.get(key, 0),
            )

            # Demand bonus for this model
            demand_bonus = 1.0
            if self.config.demand_bonus_enabled:
                model_bps = demand_scores.get(miner.model_id, 0)
                demand_bonus = compute_demand_bonus(
                    model_bps, self._scoring.demand_bonus_max,
                )

            epoch_score = self.scorer.update(
                uid=uid,
                address=miner.address,
                model_index=miner.model_index,
                outcome=outcome,
                active_params_b=model_entry.active_params_b,
                moe_dense_equivalent=model_entry.moe_dense_equivalent,
                generation_quality=model_entry.generation_quality,
                demand_bonus=demand_bonus,
                peer_medians=peer_medians_by_model.get(miner.model_id),
                tee_bonus=self._scoring.tee_bonus,
            )
            gated = self._apply_capacity_audit_model_gate(
                miner.address,
                miner.model_index,
                uid,
                model_gate_reason,
            )
            gated = self._apply_capacity_audit_score_gate(
                miner.address,
                miner.model_index,
                uid,
                audit_score_gate_reason,
            ) or gated
            if gated:
                epoch_score = 0.0

            # Persist score to DB (write-through)
            if epoch_score is not None:
                entry = self.scorer.states[uid].entries.get(miner.model_index)
                if entry:
                    self._db.save_score(
                        miner.address, miner.model_index,
                        entry.ema_score, entry.total_epochs, entry.scored_epochs,
                    )

            # Collect for summary table (printed after loop)
            if not hasattr(self, "_epoch_score_rows"):
                self._epoch_score_rows = []
            # Store references for late EMA lookup — the actual ema_score
            # may be modified by penalty handlers (halve_ema) that run
            # after scoring but before the table is printed.
            # Short GPU label for the score table (e.g. "A100", "RTX 4090")
            _gpu = getattr(miner, "gpu_name", "") or ""
            _gpu_short = _gpu.replace("NVIDIA ", "").replace("GeForce ", "").strip()
            # Shorten SXM/PCIe variants but keep memory size (A100 40GB vs 80GB matters)
            _gpu_short = _gpu_short.replace("-SXM4-", " ").replace("-SXM5-", " ").replace("-PCIe-", " ")
            self._epoch_score_rows.append({
                "uid": uid, "entry": miner.model_index,
                "model": miner.model_id, "quant": miner.quant,
                "gpu": _gpu_short,
                "score": epoch_score if epoch_score is not None else 0.0,
                "_scorer_ref": (uid, miner.model_index),  # resolved at print time
                "demand": demand_bonus,
                "own": len(own_receipts), "expected": expected,
                "failed": epoch_score is None and expected > 0 and len(own_receipts) < expected,
            })

            # Log epoch score to analytics DB
            try:
                _ema = entry.ema_score if (epoch_score is not None and entry) else 0.0
                _peer = peer_medians_by_model.get(miner.model_id)
                self._db.log_epoch_score(
                    epoch_number=epoch_number,
                    miner_address=miner.address,
                    model_index=miner.model_index,
                    model_id=miner.model_id,
                    miner_uid=uid,
                    miner_hotkey_ss58=miner.hotkey_ss58,
                    miner_coldkey_ss58=miner.coldkey_ss58,
                    own_receipts=len(own_receipts),
                    all_receipts=len(all_receipts),
                    expected_receipts=expected,
                    proof_tests=outcome.proof_tests,
                    proof_failures=outcome.proof_failures,
                    epoch_score=epoch_score,
                    demand_bonus=demand_bonus,
                    ema_score=_ema,
                    peer_median_ttft_ms=getattr(_peer, "median_ttft_ms", None),
                    peer_median_tps=getattr(_peer, "median_tps", None),
                    network=self.config.subtensor_network or "unknown",
                    netuid=self.config.netuid,
                )
            except Exception as _db_err:
                bt.logging.debug(f"Failed to log epoch score: {_db_err}")

            # ── Probation lifecycle (DB-backed + in-memory tracker) ──
            key = self._miner_model_key(miner.address, miner.model_index)
            had_proof_failure = (
                outcome.proof_tests > 0 and outcome.proof_failures > 0
            )

            if had_proof_failure:
                # Enter or reset probation (mid-epoch may have already entered)
                self._probation_tracker.enter_probation(
                    key, epoch_number, endpoint=getattr(miner, 'endpoint', ''))
                self._db.enter_probation(
                    miner.address, miner.model_index, epoch_number,
                    uid=uid if uid is not None else -1,
                    hotkey_ss58=getattr(miner, "hotkey_ss58", "") or "",
                )
            elif self._probation_tracker.is_on_probation(key):
                if outcome.proof_tests > 0 and outcome.proof_failures == 0:
                    # All proofs passed this epoch — record clean pass
                    self._probation_tracker.record_pass(key)
                    self._db.record_pass(miner.address, miner.model_index)
                elif outcome.proof_tests == 0 and len(own_receipts) == 0:
                    # No receipts at all (RPC delay lost them) — don't penalize
                    # the miner for infrastructure issues. Count as neutral pass
                    # so probation can still clear during RPC outages.
                    self._probation_tracker.record_pass(key)
                    self._db.record_pass(miner.address, miner.model_index)
                    bt.logging.info(f"Probation neutral pass for {miner.address[:10]} model_index={miner.model_index} (0 receipts — RPC delay, not miner fault)")

            # ── Busy-skip evaluation ─────────────────────────────────
            # If >3 canaries got 503, check for organic receipts within
            # ±120s of each rejection. ≥3 overlapping → forgiven (busy).
            # Otherwise → evasion (probation + EMA halve).
            # Only evaluate when threshold exceeded (≤3 is tolerated).
            reject_times = self._busy_skip_probations.get(key)
            if reject_times and busy_skips_this_epoch > 3 and not had_proof_failure:
                overlap_window = 120  # seconds
                organic_near_reject = [
                    r for r in all_receipts
                    if not r.is_canary
                    and any(
                        abs(r.timestamp - rt) <= overlap_window
                        for rt in reject_times
                    )
                ]
                if len(organic_near_reject) >= 3:
                    # Genuinely busy — forgive the 503s (no new probation).
                    # Existing probation lifecycle is handled above; don't
                    # double-count as an extra pass here.
                    bt.logging.info(f"Busy-skip FORGIVEN for {miner.address[:10]} model_index={miner.model_index} ({len(organic_near_reject)} organic receipts overlapping 503 windows)")
                elif self._restart_window_grants_pass(key):
                    # Inside the post-version-bump window — assume this miner
                    # is reloading vLLM after auto-update.  One-shot per
                    # miner per bump, see _poll_remote_miner_version.
                    bt.logging.info(
                        f"Busy-skip FORGIVEN (restart-window) for {miner.address[:10]} "
                        f"model_index={miner.model_index} ({len(reject_times)} 503s, "
                        f"within 2 epochs of miner_version bump)"
                    )
                    self._restart_forgiven.add(key)
                else:
                    # No evidence of legitimate busyness — treat as evasion
                    bt.logging.info(f"Busy-skip EVASION for {miner.address[:10]} model_index={miner.model_index} ({len(reject_times)} 503s, only {len(organic_near_reject)} overlapping organic receipts)")
                    self._on_proof_failure(
                        miner.address, miner.model_index,
                        endpoint=getattr(miner, 'endpoint', ''))

            # ── Canary error evaluation ────────────────────────────
            # Tolerate ≤3 transient errors (network glitches, timeouts).
            # Mirror the 503 busy-skip forgiveness — if organic traffic was
            # flowing during the canary failures, the miner was genuinely busy
            # (full-context canaries can still trip validator-side timeouts on
            # overloaded GPUs).
            canary_errors = self._canary_errors.get(key, 0)
            if canary_errors > 3 and not had_proof_failure:
                error_times = self._canary_error_times.get(key, [])
                overlap_window = 120  # seconds — same as 503 busy-skip path
                organic_near_error = [
                    r for r in all_receipts
                    if not r.is_canary
                    and any(
                        abs(r.timestamp - et) <= overlap_window
                        for et in error_times
                    )
                ]
                if len(organic_near_error) >= 3:
                    bt.logging.info(
                        f"Canary errors FORGIVEN for {miner.address[:10]} "
                        f"model_index={miner.model_index} ({canary_errors} errors but "
                        f"{len(organic_near_error)} organic receipts overlapping ±120s windows)"
                    )
                elif self._restart_window_grants_pass(key):
                    bt.logging.info(
                        f"Canary errors FORGIVEN (restart-window) for {miner.address[:10]} "
                        f"model_index={miner.model_index} ({canary_errors} errors, "
                        f"within 2 epochs of miner_version bump)"
                    )
                    self._restart_forgiven.add(key)
                else:
                    bt.logging.info(
                        f"Canary error PENALTY for {miner.address[:10]} "
                        f"model_index={miner.model_index} ({canary_errors} errors, "
                        f"only {len(organic_near_error)} overlapping organic — no busy excuse)"
                    )
                    self._on_proof_failure(
                        miner.address, miner.model_index,
                        endpoint=getattr(miner, 'endpoint', ''))

            # ── Capability presence check ──────────────────────────
            # Catches the gap where canary_errors didn't trip 3 (e.g. only
            # 1-2 timed out per epoch but the same full-context kept failing
            # silently) yet the miner clearly cannot serve full-context.
            # Two-strike rule: only probate if zero full-context succeeded
            # in BOTH the previous and current epoch — one validator-side
            # blip (e.g. RPC slowdown causing FC canary timeouts) must not
            # cascade-probate the whole network.
            # Derived from the canary cap so the two can't drift apart again.
            FC_PROMPT_THRESHOLD = FULL_CONTEXT_TOKEN_CAP // 2
            fc_succeeded = sum(
                1 for r in all_receipts
                if r.is_canary
                and r.prompt_tokens > FC_PROMPT_THRESHOLD
                and r.tokens_generated > 0
                and not (r.proof_requested and not r.proof_verified)
            )
            had_strike_last_epoch = key in self._zero_fc_last_epoch
            if (
                fc_succeeded == 0
                and not had_proof_failure
                and not self._probation_tracker.is_on_probation(key)
            ):
                organic_count = sum(1 for r in all_receipts if not r.is_canary)
                if organic_count >= 3:
                    bt.logging.info(
                        f"Full-context capability DEFERRED for {miner.address[:10]} "
                        f"model_index={miner.model_index} ({organic_count} organic "
                        f"receipts — miner busy with real load)"
                    )
                elif self._restart_window_grants_pass(key):
                    bt.logging.info(
                        f"Capability FAILURE FORGIVEN (restart-window) for {miner.address[:10]} "
                        f"model_index={miner.model_index} (within 2 epochs of miner_version bump)"
                    )
                    self._restart_forgiven.add(key)
                elif not had_strike_last_epoch:
                    bt.logging.info(
                        f"Capability FAILURE WARNING for {miner.address[:10]} "
                        f"model_index={miner.model_index}: 0 full-context canaries "
                        f"this epoch (first strike — will probate if zero again next epoch)"
                    )
                else:
                    bt.logging.info(
                        f"Capability FAILURE for {miner.address[:10]} "
                        f"model_index={miner.model_index}: 0 full-context canaries "
                        f"succeeded across 2 consecutive epochs, no organic excuse"
                    )
                    self._on_proof_failure(
                        miner.address, miner.model_index,
                        endpoint=getattr(miner, 'endpoint', ''))
            # Update the two-strike tracker for next epoch's decision.
            if fc_succeeded == 0:
                self._zero_fc_last_epoch.add(key)
            else:
                self._zero_fc_last_epoch.discard(key)

            # Escalation: too long on probation → report offline on-chain
            if self._db.should_escalate(miner.address, miner.model_index, epoch_number):
                bt.logging.info(f"Probation ESCALATION for {miner.address[:10]} model_index={miner.model_index} -> reportOffline")
                # Background dispatch — chain wait must not block epoch close
                self._control_executor.submit(self._report_offline, miner)

        # ── Zero undiscovered miners ───────────────────────────────
        # If a miner's lease expired or it wasn't discovered this epoch,
        # it's not serving — zero its EMA immediately.  This prevents
        # stale scores from persisting in get_weights() after miners
        # leave the network.  Transient issues (unreachable but still
        # discovered) are handled by the canary error penalty path above.
        _discovered_keys = {
            (m.address.lower(), m.model_index) for m in self._epoch_miners
        }
        # Prune the two-strike tracker to currently-active miners so
        # entries for deregistered miners don't accumulate forever.
        # Keys in _zero_fc_last_epoch are stored with the original-case
        # address (matching ``key = (miner.address, miner.model_index)``
        # used in the capability check above), so case-fold for comparison.
        self._zero_fc_last_epoch = {
            k for k in self._zero_fc_last_epoch
            if (k[0].lower(), k[1]) in _discovered_keys
        }
        _zeroed = 0
        for _uid, _mstate in self.scorer.states.items():
            for _midx, _entry in _mstate.entries.items():
                if (_mstate.address.lower(), _midx) not in _discovered_keys:
                    if _entry.ema_score > 0:
                        bt.logging.info(
                            f"Zeroing undiscovered UID {_uid} entry {_midx}: "
                            f"{_entry.ema_score:.6f} -> 0"
                        )
                        _entry.ema_score = 0.0
                        self._db.save_score(
                            _mstate.address, _midx, 0.0,
                            _entry.total_epochs, _entry.scored_epochs,
                        )
                        _zeroed += 1
        if _zeroed:
            bt.logging.info(f"Zeroed {_zeroed} undiscovered miner entries")

        # ── Print score summary table ─────────────────────────────
        # Resolve EMA at print time (not scoring time) so penalties
        # applied between scoring and printing are reflected.
        rows = getattr(self, "_epoch_score_rows", [])
        if rows:
            for r in rows:
                _ref = r.pop("_scorer_ref", None)
                if _ref:
                    _uid_r, _midx = _ref
                    _st = self.scorer.states.get(_uid_r)
                    _ent = _st.entries.get(_midx) if _st else None
                    r["ema"] = _ent.ema_score if _ent else 0.0
                else:
                    r.setdefault("ema", 0.0)

            rows.sort(key=lambda r: (-r["ema"], r["uid"], r["entry"]))
            _unique_uids = len({r["uid"] for r in rows})
            bt.logging.success(f"Epoch {epoch_number} scores ({len(rows)} entries, {_unique_uids} miners):")
            bt.logging.info("")
            bt.logging.info(f"{'UID':<5} {'Entry':<5}  {'Model':<30}  {'Quant':<5}  {'GPU':<20}  {'Score':>8}  {'EMA':>8}  {'Demand':>6}  {'Receipts':>9}")
            bt.logging.info(f"{'─'*5} {'─'*5}  {'─'*30}  {'─'*5}  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}")
            # Group by UID for total rows
            from collections import defaultdict
            _uid_scores = defaultdict(list)
            for r in rows:
                _uid_scores[r["uid"]].append(r)
            for uid_val in sorted(_uid_scores, key=lambda u: -sum(r["ema"] for r in _uid_scores[u])):
                uid_rows = sorted(_uid_scores[uid_val], key=lambda r: -r["ema"])
                for r in uid_rows:
                    status = "FAIL" if r["failed"] else ""
                    receipts = f"{r['own']}/{r['expected']}"
                    gpu = r.get("gpu", "")[:20]
                    bt.logging.info(
                        f"{r['uid']:<5} {r['entry']:<5}  {r['model']:<30}  {r['quant']:<5}  {gpu:<20}  {r['score']:>8.4f}  {r['ema']:>8.4f}  {r['demand']:>5.2f}x  {receipts:>9} {status}"
                    )
                if len(uid_rows) > 1:
                    total_score = sum(r["score"] for r in uid_rows)
                    total_ema = sum(r["ema"] for r in uid_rows)
                    bt.logging.info(f"{uid_val:<5} {'':<5}  {'── total ──':<30}  {'':<5}  {'':<20}  {total_score:>8.4f}  {total_ema:>8.4f}")
            bt.logging.info("")
        self._epoch_score_rows = []

        # ── Write shared state for proxy (BEFORE on-chain posts) ──
        # Shared state must be written even if on-chain calls fail (429).
        self._write_shared_state()

        # ── Epoch audit log ─────────────────────────────────────
        weight_set = False  # weights are set on a separate boundary
        self._db.log_epoch(
            epoch=epoch_number,
            start_block=self._epoch_start_block,
            miner_count=len(self._epoch_miners),
            receipt_count=len(all_epoch_receipts),
            weight_set=weight_set,
        )
        self._db.set_meta("current_epoch", str(epoch_number))

        try:
            cleanup = self._db.compact_capacity_audit_storage(
                current_epoch=int(epoch_number),
                retain_failure_epochs=int(self._capacity_audit_cfg.repeat_window_epochs) + 2,
                retain_artifacts=os.environ.get(
                    "VERATHOS_CAPACITY_AUDIT_RETAIN_ARTIFACTS",
                    "",
                ).lower() in {"1", "true", "yes", "on"},
            )
            if any(cleanup.values()):
                bt.logging.info(f"Capacity audit storage cleanup: {cleanup}")
        except Exception as exc:
            bt.logging.warning(f"Capacity audit storage cleanup failed: {exc}")

        # Periodic analytics backup+cleanup (every ~7 days ≈ 140 epochs)
        if epoch_number % 140 == 0:
            for table, fn in [
                ("canary results", self._db.backup_and_cleanup_canary_results),
                ("network receipts", self._db.backup_and_cleanup_network_receipts),
            ]:
                archived = fn(retain_days=7)
                if archived > 0:
                    bt.logging.info(f"Analytics: archived {archived} {table}")

            # Auto-delete old backup files unless --retain-backups is set
            if not getattr(self.config, "retain_backups", False):
                _backup_dir = os.path.join(
                    os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
                    "backups",
                )
                if os.path.isdir(_backup_dir):
                    import glob as _glob
                    _cutoff = time.time() - (7 * 86400)
                    for _pattern in ("canary_results_*.jsonl.gz", "network_receipts_*.jsonl.gz"):
                        for _f in _glob.glob(os.path.join(_backup_dir, _pattern)):
                            if os.path.getmtime(_f) < _cutoff:
                                os.remove(_f)
                                bt.logging.info(f"Deleted old backup: {os.path.basename(_f)}")

        elapsed = time.monotonic() - t0
        bt.logging.info(f"Epoch {epoch_number} closed in {elapsed:.1f}s")

    def _collect_epoch_receipts(
        self,
        epoch_number: int,
        receipt_authority: ValidatorAuthority | None,
    ) -> Tuple[Dict[str, List[ServiceReceipt]], List[ServiceReceipt]]:
        """Pull epoch receipts without sharing workers with canary execution."""
        miner_receipts: Dict[str, List[ServiceReceipt]] = {}
        all_epoch_receipts: List[ServiceReceipt] = []
        pull_failed_keys: Set[Tuple[str, int]] = set()
        if not self._epoch_miners:
            self._receipt_pull_failed_keys = pull_failed_keys
            return miner_receipts, all_epoch_receipts

        # Receipt close is a scoring-critical phase. Do not submit these jobs
        # to self._executor: long canaries can occupy every canary worker and
        # leave all receipt pulls queued until the overall timeout fires.
        receipt_workers = max(
            1,
            min(self.config.max_concurrent_verifications, len(self._epoch_miners)),
        )
        receipt_executor = ThreadPoolExecutor(
            max_workers=receipt_workers,
            thread_name_prefix="receipt-pull",
        )
        receipt_futures = {}
        try:
            for miner in self._epoch_miners:
                if not self._running:
                    break
                receipt_futures[
                    receipt_executor.submit(
                        self._pull_epoch_receipts, miner, epoch_number, receipt_authority
                    )
                ] = miner

            # Overall budget for all receipt pulls — scales with miner count,
            # floored by config, capped at 120s.
            _rp_timeout = min(120, max(
                self.config.epoch_receipt_pull_overall_timeout,
                len(self._epoch_miners) // self.config.max_concurrent_verifications * 3 + 10,
            ))
            # Cross-pull signature dedup: when a miner registers multiple
            # model_indices on a single physical server, every endpoint returns
            # the same receipt buffer. Filter by validator_signature so each
            # signed receipt counts once across all pulls for the same address.
            # Multi-server legit operators see no filtering (signatures differ).
            seen_sigs_by_addr: Dict[str, set] = {}
            cross_pull_dups = 0
            try:
                for fut in as_completed(receipt_futures, timeout=_rp_timeout):
                    miner = receipt_futures[fut]
                    try:
                        receipts = fut.result()
                        if receipts is None:
                            pull_failed_keys.add(
                                self._miner_model_key(miner.address, miner.model_index)
                            )
                            continue
                        addr_key = miner.address.lower()
                        seen = seen_sigs_by_addr.setdefault(addr_key, set())
                        new_receipts = []
                        for r in receipts:
                            if r.validator_signature in seen:
                                cross_pull_dups += 1
                                continue
                            seen.add(r.validator_signature)
                            new_receipts.append(r)
                        miner_receipts.setdefault(miner.address, []).extend(new_receipts)
                        all_epoch_receipts.extend(new_receipts)
                    except Exception as e:
                        pull_failed_keys.add(
                            self._miner_model_key(miner.address, miner.model_index)
                        )
                        bt.logging.warning(f"Receipt pull exception for {miner.address[:10]}: {e}")
            except _FuturesTimeout:
                stalled = [
                    receipt_futures[f].address[:10]
                    for f in receipt_futures if not f.done()
                ]
                bt.logging.warning(
                    f"Receipt pull timeout after {_rp_timeout:.0f}s — {len(stalled)} miner(s) stalled: {stalled}. "
                    f"Proceeding with {sum(len(r) for r in miner_receipts.values())} receipts."
                )
                for f in receipt_futures:
                    if not f.done():
                        miner = receipt_futures[f]
                        pull_failed_keys.add(
                            self._miner_model_key(miner.address, miner.model_index)
                        )
                        f.cancel()

            if cross_pull_dups > 0:
                bt.logging.info(
                    f"Receipt cross-pull dedup: dropped {cross_pull_dups} duplicate signatures"
                )
        finally:
            receipt_executor.shutdown(wait=False, cancel_futures=True)

        self._receipt_pull_failed_keys = pull_failed_keys
        return miner_receipts, all_epoch_receipts

    def _pull_epoch_receipts(
        self,
        miner: ActiveMiner,
        epoch_number: int,
        authority: ValidatorAuthority | None = None,
    ) -> Optional[List[ServiceReceipt]]:
        """Pull all receipts from a miner for the given epoch.

        GET /epoch/{epoch_number}/receipts — returns all accumulated receipts.

        Each receipt is verified against ``authority`` (built once per epoch
        close from a fresh metagraph + ValidatorRegistry read).  Receipts
        whose embedded Sr25519 pubkey does not resolve to a registered
        validator with permit and stake >= ``minValidatorStake`` are
        rejected — closes the lone-miner forgery vector entirely.

        Duplicates (same signature appearing more than once in the response)
        are also dropped — the original anti-replay guard.
        """
        url = f"{miner.endpoint.rstrip('/')}/epoch/{epoch_number}/receipts"
        path = f"/epoch/{epoch_number}/receipts"

        from neurons.request_signing import sign_request
        auth_headers = sign_request(
            method="GET", path=path, body=b"",
            hotkey_ss58=self._validator_hotkey_ss58,
            hotkey_seed=self._validator_private_key,
        )
        transient_status = {408, 425, 429, 500, 502, 503, 504}
        transient_exc = (
            httpx.TimeoutException, httpx.ReadError, httpx.ConnectError,
            httpx.RemoteProtocolError, httpx.WriteError,
        )
        attempts = 3
        last_err = None
        resp = None
        for attempt in range(1, attempts + 1):
            try:
                resp = httpx.get(
                    url,
                    timeout=self.config.epoch_receipt_pull_timeout,
                    headers=auth_headers,
                    verify=False,
                )
                if resp.status_code == 200:
                    break
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code in transient_status and attempt < attempts:
                    bt.logging.info(
                        f"Receipt pull retry {attempt}/{attempts} for {miner.address[:10]} "
                        f"model_index={miner.model_index}: {last_err}"
                    )
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                bt.logging.warning(
                    f"Receipt pull from {miner.address[:10]} model_index={miner.model_index} "
                    f"failed: {last_err}"
                )
                return None
            except transient_exc as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < attempts:
                    bt.logging.info(
                        f"Receipt pull retry {attempt}/{attempts} for {miner.address[:10]} "
                        f"model_index={miner.model_index}: {last_err}"
                    )
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                bt.logging.warning(
                    f"Receipt pull failed for {miner.address[:10]} model_index={miner.model_index} "
                    f"after {attempts} attempts: {last_err}"
                )
                return None
            except Exception as e:
                bt.logging.warning(
                    f"Receipt pull failed for {miner.address[:10]} model_index={miner.model_index}: {e}"
                )
                return None

        try:
            data = resp.json() if resp is not None else {}
            receipt_dicts = data.get("receipts", [])

            verified = []
            seen_sigs: set[bytes] = set()
            duplicates = 0
            for r_dict in receipt_dicts:
                try:
                    receipt = receipt_from_dict(r_dict)
                    if not verify_service_receipt(receipt, epoch_number, authority=authority):
                        continue
                    if receipt.validator_signature in seen_sigs:
                        duplicates += 1
                        continue
                    seen_sigs.add(receipt.validator_signature)
                    verified.append(receipt)
                except Exception as e:
                    bt.logging.debug(f"Invalid receipt from {miner.address[:10]}: {e}")

            if duplicates > 0:
                # Demoted from warning — duplicate signatures during pulls are
                # expected when a miner is polled across multiple paths.  The
                # dedup itself is the safety net, not a problem.
                bt.logging.debug(
                    f"Receipt dedup from {miner.address[:10]}: dropped {duplicates} duplicate "
                    f"signature(s) out of {len(receipt_dicts)} pulled"
                )

            bt.logging.info(
                f"Pulled {len(verified)}/{len(receipt_dicts)} valid receipt(s) "
                f"from {miner.address[:10]} model_index={miner.model_index}"
            )
            return verified

        except Exception as e:
            bt.logging.warning(
                f"Receipt pull decode/verify failed for {miner.address[:10]} "
                f"model_index={miner.model_index}: {e}"
            )
            return None

    def _push_receipt_to_miner(
        self,
        miner_address: str,
        miner_endpoint: str,
        model_id: str,
        model_index: int,
        epoch_number: int,
        commitment_hash: bytes,
        ttft_ms: float,
        tokens_generated: int,
        generation_time_ms: float,
        tokens_per_sec: float,
        prompt_tokens: int = 0,
        proof_verified: bool = False,
        proof_requested: bool = False,
        tee_attestation_verified: object = None,  # None=not tested, True=passed, False=failed
        is_canary: bool = False,
        timestamp: Optional[int] = None,
        observed_start_ts: Optional[float] = None,
        observed_end_ts: Optional[float] = None,
    ) -> bool:
        """Push signed receipt. Returns True on 200; retries transient transport/5xx."""
        receipt = create_receipt(
            miner_address=miner_address,
            model_id=model_id,
            model_index=model_index,
            epoch_number=epoch_number,
            commitment_hash=commitment_hash,
            ttft_ms=ttft_ms,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time_ms,
            tokens_per_sec=tokens_per_sec,
            validator_hotkey=self._validator_hotkey_bytes,
            validator_private_key=self._validator_private_key,
            prompt_tokens=prompt_tokens,
            proof_verified=proof_verified,
            proof_requested=proof_requested,
            tee_attestation_verified=tee_attestation_verified,
            is_canary=is_canary,
            timestamp=timestamp,
            observed_start_ts=observed_start_ts,
            observed_end_ts=observed_end_ts,
        )

        url = f"{miner_endpoint.rstrip('/')}/epoch/receipt"
        import json as _json
        import time as _time
        from neurons.request_signing import sign_request as _sign
        receipt_body = _json.dumps(receipt_to_dict(receipt)).encode("utf-8")
        auth_headers = _sign(
            method="POST", path="/epoch/receipt", body=receipt_body,
            hotkey_ss58=self._validator_hotkey_ss58,
            hotkey_seed=self._validator_private_key,
        )
        # 5s per attempt × 3 attempts; transient retry only.
        per_attempt_timeout = min(5.0, self.config.miner_endpoint_timeout)
        backoffs = [0.5, 1.5]
        transient_exc = (
            httpx.TimeoutException, httpx.ReadError, httpx.ConnectError,
            httpx.RemoteProtocolError, httpx.WriteError,
        )
        last_status = None
        last_err = None
        for attempt in range(3):
            try:
                resp = httpx.post(
                    url,
                    content=receipt_body,
                    headers={**auth_headers, "content-type": "application/json"},
                    timeout=per_attempt_timeout,
                    verify=False,
                )
                if resp.status_code == 200:
                    bt.logging.debug(
                        f"Pushed receipt to {miner_address[:10]} model_index={model_index} "
                        f"epoch={epoch_number} (attempt {attempt + 1})"
                    )
                    return True
                last_status = resp.status_code
                if resp.status_code >= 500 and attempt < len(backoffs):
                    _time.sleep(backoffs[attempt])
                    continue
                break  # 4xx — don't retry
            except transient_exc as e:
                last_err = e
                if attempt < len(backoffs):
                    _time.sleep(backoffs[attempt])
                    continue
                break
            except Exception as e:
                last_err = e
                break
        reason = (
            f"HTTP {last_status}" if last_status is not None
            else f"{type(last_err).__name__}: {last_err}"
        )
        bt.logging.info(
            f"Receipt push FAILED to {miner_address[:10]} model_index={model_index} "
            f"epoch={epoch_number} after 3 attempts: {reason}"
        )
        return False

    def _check_tokenizer_drift(self, model_id: str, spec) -> None:
        """Compare local tokenizer hash to the on-chain anchor.

        On mismatch, sets ``spec._tokenizer_drift = True`` so the canary
        dispatch path can short-circuit verification with a validator-side
        attribution (does NOT penalize the miner).

        Failure modes handled:
        - On-chain hash empty: feature not enforced for this model (legacy
          spec from before the upgrade) → drift flag stays False.
        - Local tokenizer cannot be loaded: marked as drifted (validator
          can't compute the local hash, so it can't safely verify proofs).
        - Hash mismatch: marked as drifted, error logged with both hashes.
        """
        on_chain = getattr(spec, "tokenizer_hash", b"") or b""
        if not on_chain:
            spec._tokenizer_drift = False
            return
        try:
            from verallm.registry.tokenizer_hash import compute_tokenizer_hash
            local = compute_tokenizer_hash(model_id)
        except Exception as e:
            bt.logging.warning(
                f"Could not compute local tokenizer hash for {model_id}: {e} — "
                "marking as drifted (validator-side issue, miner not penalized)"
            )
            spec._tokenizer_drift = True
            return
        if local != on_chain:
            bt.logging.error(
                f"TOKENIZER DRIFT for {model_id}: "
                f"on-chain={on_chain[:8].hex()} local={local[:8].hex()} — "
                "this is a validator-side issue, not a miner failure. "
                "Refusing to verify proofs for this model until resolved."
            )
            spec._tokenizer_drift = True
        else:
            spec._tokenizer_drift = False
            bt.logging.info(
                f"Tokenizer hash verified for {model_id}: {local[:8].hex()}"
            )

    def _on_proof_failure(self, miner_address: str, model_index: int,
                          endpoint: str = ""):
        """Mid-epoch cutoff: immediately put miner on probation and notify proxy.

        Called as soon as a proof verification fails (not waiting for epoch close).
        Updates shared state so the proxy stops routing organic traffic to this miner.
        Halves EMA score on every failure — geometric decay punishes repeat
        offenders while keeping single failures recoverable.
        """
        key = self._miner_model_key(miner_address, model_index)
        # Look up UID + SS58 for human-readable logging.  These are only
        # used for log messages — the DB row itself is keyed on
        # (address, model_index).
        _uid = -1
        _ss58 = ""
        try:
            _u = self._db.get_uid(miner_address)
            if _u is not None:
                _uid = int(_u)
        except Exception:
            pass
        try:
            _ss58 = self._get_miner_ss58(miner_address, "hotkey") or ""
        except Exception:
            pass
        if not self._probation_tracker.is_on_probation(key):
            self._probation_tracker.enter_probation(key, self._current_epoch, endpoint=endpoint)
            self._db.enter_probation(
                miner_address, model_index, self._current_epoch,
                uid=_uid, hotkey_ss58=_ss58,
            )
        else:
            self._probation_tracker.record_failure(key)
            self._db.record_failure(miner_address, model_index)

        # Halve EMA score immediately — don't wait for epoch close
        self.scorer.halve_ema(miner_address, model_index)
        self._db.halve_ema(miner_address, model_index)

        # Immediately update shared state so proxy cuts off this miner
        self._write_shared_state()
        bt.logging.info(f"Mid-epoch cutoff: {miner_address[:10]} model_index={model_index} on probation, shared state updated for proxy")

    # ------------------------------------------------------------------
    # Identity verification (anti-hijacking)
    # ------------------------------------------------------------------

    def _verify_miner_identity(self, miner: ActiveMiner) -> Optional[bool]:
        """Verify a miner controls the endpoint it registered.

        Sends a random nonce to POST /identity/challenge.  The miner signs
        (nonce || evm_address) with its EVM key.  We recover the signer and
        compare against the on-chain registered address.

        Returns:
            True  — identity confirmed (signer matches registered address).
            False — identity FAILED (signer mismatch — likely hijacking).
            None  — endpoint doesn't support challenge (404/501/timeout).
        """
        import os

        url = f"{miner.endpoint.rstrip('/')}/identity/challenge"
        bt.logging.debug(f"Identity challenge for {miner.address[:10]} at {url}")
        resp = None
        last_error = None
        last_status = None
        max_attempts = 4
        timeout_s = max(1.0, float(self.config.identity_challenge_timeout or 10.0))
        deadline = time.monotonic() + min(45.0, max(timeout_s + 10.0, timeout_s * 2.0 + 5.0))
        for attempt in range(1, max_attempts + 1):
            if time.monotonic() >= deadline:
                break
            nonce = os.urandom(32)
            try:
                remaining = max(1.0, deadline - time.monotonic())
                resp = httpx.post(
                    url,
                    json={"nonce": nonce.hex()},
                    timeout=min(timeout_s, remaining),
                    verify=False,
                )
                last_status = int(resp.status_code)
                if resp.status_code == 200:
                    break
                if resp.status_code in (404, 405, 501):
                    return None  # endpoint doesn't support challenges
            except Exception as _ide:
                last_error = _ide
                bt.logging.debug(f"Identity challenge exception for {miner.address[:10]}: {_ide}")
            if attempt < max_attempts and time.monotonic() < deadline:
                wait = min(2.0, 0.25 * (2 ** (attempt - 1)), deadline - time.monotonic())
                if wait <= 0:
                    break
                bt.logging.debug(
                    f"Identity challenge retry {attempt}/{max_attempts} for "
                    f"{miner.address[:10]}, next in {wait:.2f}s"
                )
                time.sleep(wait)

        if resp is None or resp.status_code != 200:
            detail = (
                f"status={last_status}"
                if last_status is not None
                else f"error={last_error}"
            )
            bt.logging.debug(
                f"Identity challenge failed for {miner.address[:10]} after "
                f"{attempt} attempts ({detail})"
            )
            return None

        try:
            data = resp.json()
            sig_hex = data["signature"]
            claimed_address = data["address"]
        except (KeyError, ValueError) as e:
            bt.logging.debug(f"Invalid identity response from {miner.address[:10]}: {e}")
            return False

        # Recover signer from EIP-191 personal sign
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct

            address_bytes = bytes.fromhex(miner.address[2:] if miner.address.startswith("0x") else miner.address)
            message = nonce + address_bytes
            signable = encode_defunct(primitive=message)
            recovered = Account.recover_message(signable, signature=bytes.fromhex(sig_hex))
        except Exception as e:
            bt.logging.debug(f"Signature recovery failed for {miner.address[:10]}: {e}")
            return False

        if recovered.lower() == miner.address.lower():
            return True

        bt.logging.info(f"Identity mismatch for {miner.endpoint}: expected {miner.address[:10]}, recovered {recovered[:10]}")
        return False

    def _fetch_miner_hardware(self, miner: ActiveMiner) -> bool:
        """Fetch hardware metadata from a miner's /health endpoint (best-effort).

        Populates gpu_name, gpu_count, vram_gb, compute_capability on the
        ActiveMiner object.  Non-fatal — old miners without the hardware
        block simply keep empty defaults.
        """
        try:
            resp = httpx.get(
                f"{miner.endpoint.rstrip('/')}/health",
                timeout=5.0, verify=False,
            )
            if resp.status_code != 200:
                return True
            data = resp.json()
            if "hardware" not in data:
                return True
            valid, hw = _normalize_health_hardware(data.get("hardware"))
            miner.gpu_name = hw["gpu_name"]
            miner.gpu_count = hw["gpu_count"]
            miner.vram_gb = hw["vram_gb"]
            miner.compute_capability = hw["compute_capability"]
            miner.gpu_uuids = hw["gpu_uuids"]
            return valid
        except Exception:
            return True  # Non-fatal

    # ------------------------------------------------------------------
    # Discovery + resolution
    # ------------------------------------------------------------------

    def _get_all_model_ids(self) -> List[str]:
        """Get all registered model IDs from the on-chain ModelRegistry."""
        if self._model_client is not None:
            try:
                return self._model_client.get_model_list()
            except Exception as e:
                bt.logging.warning(f"Failed to get model list: {e}")

        # Fallback: use local registry
        return list(MODELS_BY_ID.keys())

    def _get_model_bucket_ids(self) -> List[str]:
        """Get approved model IDs for emission buckets.

        Bucket burn is based on on-chain approval. If the registry read fails,
        fall back to currently served known models only; using the full local
        catalogue would burn against models that may not be approved on chain.
        """
        if self._model_client is not None:
            try:
                return self._model_client.get_model_list()
            except Exception as e:
                bt.logging.warning(
                    f"Failed to read ModelRegistry for emission buckets: {e}"
                )
        served = {
            m.model_id
            for m in getattr(self, "_epoch_miners", [])
            if get_model(m.model_id) is not None
        }
        if served:
            bt.logging.info(
                "Model emission buckets: using served-model fallback "
                "(ModelRegistry unavailable)"
            )
        return sorted(served)

    def _best_registry_model_runtime(self, model_entry) -> Tuple[int, str]:
        """Return the registry quant/context pair with highest bucket utility."""
        best_score = 0.0
        best_ctx = 0
        best_quant = ""
        for tier_config in getattr(model_entry, "tier_configs", ()):
            for quant_option in getattr(tier_config, "quant_configs", ()):
                ctx = int(
                    getattr(quant_option, "max_model_len", 0)
                    or getattr(model_entry, "native_context_len", 0)
                    or 0
                )
                quant = getattr(quant_option, "quant", "") or ""
                if ctx <= 0 or not quant:
                    continue
                score = compute_model_base_utility(
                    active_params_b=model_entry.active_params_b,
                    max_context_len=ctx,
                    quant=quant,
                    moe_dense_equivalent=model_entry.moe_dense_equivalent,
                    generation_quality=model_entry.generation_quality,
                )
                if score > best_score:
                    best_score = score
                    best_ctx = ctx
                    best_quant = quant
        return best_ctx, best_quant

    def _observed_model_runtimes(self) -> Dict[str, Tuple[int, str]]:
        """Return best observed quant/context pair per model this epoch."""
        observed: Dict[str, Tuple[int, str]] = {}
        observed_score: Dict[str, float] = {}
        for miner in getattr(self, "_epoch_miners", []):
            model_entry = get_model(miner.model_id)
            if model_entry is None:
                continue
            ctx = int(getattr(miner, "max_context_len", 0) or 0)
            quant = getattr(miner, "quant", "") or ""
            if ctx <= 0 or not quant:
                continue
            score = compute_model_base_utility(
                active_params_b=model_entry.active_params_b,
                max_context_len=ctx,
                quant=quant,
                moe_dense_equivalent=model_entry.moe_dense_equivalent,
                generation_quality=model_entry.generation_quality,
            )
            if score > observed_score.get(miner.model_id, 0.0):
                observed_score[miner.model_id] = score
                observed[miner.model_id] = (ctx, quant)
        return observed

    def _model_emission_group(self, model_entry) -> str:
        """Return logical model bucket key for a registry entry."""
        return (
            getattr(model_entry, "base_model", "")
            or getattr(model_entry, "id", "")
        )

    def _build_model_emission_budgets(
        self,
        demand_scores: Dict[str, int],
    ) -> Dict[str, float]:
        """Build raw model-level emission budgets for approved models."""
        model_ids = self._get_model_bucket_ids()
        observed = self._observed_model_runtimes()
        budgets: Dict[str, float] = {}
        groups: Dict[str, str] = {}
        group_budgets: Dict[str, float] = {}
        skipped_unknown = 0

        for model_id in model_ids:
            model_entry = get_model(model_id)
            if model_entry is None:
                skipped_unknown += 1
                continue

            ctx, quant = observed.get(model_id, (0, ""))
            if ctx <= 0 or not quant:
                ctx, quant = self._best_registry_model_runtime(model_entry)
            if ctx <= 0 or not quant:
                continue

            base_utility = compute_model_base_utility(
                active_params_b=model_entry.active_params_b,
                max_context_len=ctx,
                quant=quant,
                moe_dense_equivalent=model_entry.moe_dense_equivalent,
                generation_quality=model_entry.generation_quality,
            )
            if base_utility <= 0:
                continue

            demand_bonus = 1.0
            if self.config.demand_bonus_enabled:
                demand_bonus = compute_demand_bonus(
                    demand_scores.get(model_id, 0),
                    self._scoring.demand_bonus_max,
                )
            variant_budget = base_utility * demand_bonus
            budgets[model_id] = variant_budget
            group_id = self._model_emission_group(model_entry) or model_id
            groups[model_id] = group_id
            group_budgets[group_id] = max(
                group_budgets.get(group_id, 0.0),
                variant_budget,
            )

        if skipped_unknown:
            bt.logging.info(
                f"Model emission buckets: skipped {skipped_unknown} approved "
                "model(s) missing from local registry"
            )
        if budgets:
            group_total = sum(group_budgets.values())
            group_top = {
                group_id: f"{value / group_total:.1%}"
                for group_id, value in sorted(
                    group_budgets.items(), key=lambda item: -item[1]
                )[:5]
            } if group_total > 0 else {}
            variant_total = sum(budgets.values())
            top = {
                model_id: f"{value / variant_total:.1%}"
                for model_id, value in sorted(
                    budgets.items(), key=lambda item: -item[1]
                )[:5]
            } if variant_total > 0 else {}
            bt.logging.info(f"Logical model emission bucket shares: {group_top}")
            bt.logging.info(f"Approved variant budget weights: {top}")
        self._last_model_emission_groups = groups
        self._last_model_group_budgets = group_budgets
        return budgets

    def _refresh_blacklist(self, addresses) -> None:
        """Populate ``self._blacklisted_uids`` from SubnetConfig for the given addresses.

        Called at boot (after startup discovery) AND at every epoch close so
        weight-setting always sees an up-to-date blacklist.  Without the boot
        call, the first weight-set after restart fires before the first
        ``_close_epoch`` and the empty default lets blacklisted miners through.
        """
        self._blacklisted_uids = set()
        self._blacklisted_addresses = set()
        if self._subnet_config_client is None or not addresses:
            return
        _bl_futures = {
            self._control_executor.submit(
                self._subnet_config_client.is_miner_blacklisted, addr
            ): addr
            for addr in addresses
        }
        try:
            for fut in as_completed(_bl_futures, timeout=15):
                addr = _bl_futures[fut]
                try:
                    if fut.result():
                        self._blacklisted_addresses.add(addr.lower())
                        uid = self._resolve_uid(addr)
                        if uid is not None:
                            self._blacklisted_uids.add(uid)
                            bt.logging.info(f"Miner {addr[:10]} (UID {uid}) is BLACKLISTED — score will be zeroed")
                except Exception:
                    pass
        except _FuturesTimeout:
            bt.logging.warning("Blacklist check timeout (15s) — proceeding with partial results")

    def _resolve_uid(self, evm_address: str) -> Optional[int]:
        """Resolve EVM address to Bittensor UID.

        Returns DB-cached UID immediately if available (no RPC call).
        RPC is only used during the epoch-start pre-warm to refresh the
        cache — never on the epoch-close hot path.
        """
        # DB cache first — instant, no RPC
        cached = self._db.get_uid(evm_address)
        if cached is not None:
            return cached

        # No cache — must try RPC (first time seeing this address)
        try:
            uid = self._miner_client.get_associated_uid(evm_address)
            if uid is not None:
                self._db.set_uid(evm_address, uid)
            return uid
        except Exception as e:
            bt.logging.warning(f"UID lookup failed for {evm_address[:10]}, no cache: {e}")
            return None

    def _get_miner_ss58(self, miner_address: str, key_type: str = "hotkey") -> str:
        """O(1) lookup of miner SS58 from cache."""
        entry = self._ss58_cache.get(miner_address.lower())
        if entry:
            return entry.get(f"{key_type}_ss58", "")
        return ""

    def _enrich_miners_from_metagraph(self, miners: List[ActiveMiner]) -> None:
        """Enrich miners with SS58 keys from metagraph. Updates _ss58_cache.

        Called at startup and at each epoch start. Uses chain UID lookup
        first (authoritative), falls back to DB cache.
        """
        try:
            mg = self._subtensor.metagraph(self.config.netuid)
            self._metagraph = mg
            n = mg.n.item()
            for miner in miners:
                uid_val = None
                try:
                    uid_val = self._miner_client.get_associated_uid(miner.address)
                except Exception:
                    uid_val = self._db.get_uid(miner.address)
                if uid_val is not None and uid_val < n:
                    miner.hotkey_ss58 = mg.hotkeys[uid_val]
                    miner.coldkey_ss58 = mg.coldkeys[uid_val] if hasattr(mg, 'coldkeys') else ""
                    self._db.set_uid(miner.address, uid_val)
                    # Update cache
                    self._ss58_cache[miner.address.lower()] = {
                        "hotkey_ss58": miner.hotkey_ss58,
                        "coldkey_ss58": miner.coldkey_ss58,
                    }
        except Exception as e:
            bt.logging.debug(f"Metagraph enrichment failed: {e}")

    def _report_offline(self, miner: ActiveMiner):
        """Report a miner-model entry as offline."""
        if self._evm_disabled:
            bt.logging.debug(
                f"EVM disabled — skipping reportOffline for {miner.address[:10]} "
                f"model_index={miner.model_index} (other validators + 24h lease handle it)"
            )
            return
        try:
            self._miner_client.report_offline(
                miner.address, miner.model_index, private_key=self.evm_pk,
            )
            bt.logging.info(f"Reported {miner.address[:10]} model_index={miner.model_index} as offline")
        except Exception as e:
            _msg = str(e)
            if "Already reported" in _msg:
                bt.logging.debug(f"Miner already reported offline (idempotent)")
            else:
                bt.logging.warning(f"Failed to report offline: {e}")

    def _write_shared_state(self):
        """Write shared state file for the proxy process.

        Derives scores and probation from the validator state DB, then
        overlays the current epoch's miner endpoints (live from discovery).
        """
        from neurons.shared_state import write_shared_state, MinerEntry

        shared = self._db.derive_shared_state(self._current_epoch)
        shared.epoch_start_block = self._epoch_start_block
        shared.last_weights = getattr(self, "_last_weights", {})
        shared.demand_scores = getattr(self, "_last_demand_scores", {})
        shared.blacklisted_addresses = sorted(self._blacklisted_addresses)
        # Build ss58_map with UIDs so the proxy can resolve UIDs for
        # miners not in miner_endpoints (e.g. inactive/unreachable).
        uid_map_all = self._db.get_all_uids()
        ss58_with_uid: Dict[str, Dict[str, str]] = {}
        for addr, info in self._ss58_cache.items():
            entry = dict(info)  # copy {hotkey_ss58, coldkey_ss58}
            uid_val = uid_map_all.get(addr)
            if uid_val is not None:
                entry["uid"] = str(uid_val)
            ss58_with_uid[addr] = entry
        shared.ss58_map = ss58_with_uid

        # Build miner endpoints from live miners (epoch miners).
        # Only reachable miners go here — the proxy uses its own on-chain
        # discovery for the full set and falls back to ss58_map (above)
        # for UID/SS58 of miners the validator can't TCP-reach.
        uid_map = uid_map_all
        miners = getattr(self, "_epoch_miners", [])
        if not miners and not getattr(self, "_epoch_miners_discovery_valid", False):
            # No live miners (startup before first epoch) — reconstruct from DB
            db_entries = self._db.get_active_entries()
            from neurons.discovery import ActiveMiner
            miners = [
                ActiveMiner(
                    address=e["address"], endpoint=e["endpoint"],
                    model_id=e["model_id"], model_index=e["model_index"],
                    quant=e.get("quant", ""), max_context_len=e.get("max_context_len", 0),
                    hotkey_ss58=e.get("hotkey_ss58") or "",
                    coldkey_ss58=e.get("coldkey_ss58") or "",
                    gpu_name=e.get("gpu_name") or "",
                    gpu_count=e.get("gpu_count") or 0,
                    vram_gb=e.get("vram_gb") or 0,
                    compute_capability=e.get("compute_capability") or "",
                    gpu_uuids=json.loads(e.get("gpu_uuids") or "[]"),
                )
                for e in db_entries
            ]
        # Check if TEE is enabled on the subnet (cached, no extra RPC)
        _subnet_tee = False
        if self._subnet_config_client is not None:
            try:
                _subnet_tee = self._subnet_config_client.is_tee_enabled_on_subnet()
            except Exception:
                pass

        shared.miner_endpoints = [
            MinerEntry(
                address=m.address, endpoint=m.endpoint,
                model_id=m.model_id, model_index=m.model_index,
                quant=m.quant, max_context_len=m.max_context_len,
                uid=uid_map.get(m.address.lower()),
                hotkey_ss58=m.hotkey_ss58 or self._get_miner_ss58(m.address, "hotkey"),
                coldkey_ss58=m.coldkey_ss58 or self._get_miner_ss58(m.address, "coldkey"),
                tee_enabled=getattr(m, "tee_enabled", False) and _subnet_tee,
                tee_platform=getattr(m, "tee_platform", "") if _subnet_tee else "",
                enclave_public_key=getattr(m, "enclave_public_key", "") if _subnet_tee else "",
                gpu_name=getattr(m, "gpu_name", ""),
                gpu_count=getattr(m, "gpu_count", 0),
                vram_gb=getattr(m, "vram_gb", 0),
                compute_capability=getattr(m, "compute_capability", ""),
                gpu_uuids=getattr(m, "gpu_uuids", []),
            )
            for m in miners
        ]

        write_shared_state(shared, self.config.shared_state_path)
        bt.logging.info(f"Shared state written: epoch={self._current_epoch}, {len(shared.miner_scores)} miner scores")

    def _load_scores_from_db(self):
        """Load persisted EMA scores from the validator state DB into the in-memory scorer.

        Called once during setup() so scores survive validator restarts without
        the 10-epoch EMA recovery period.  Requires UID mappings to be present
        in the DB (populated during previous epochs' UID pre-warm).
        """
        saved = self._db.load_all_scores()
        if not saved:
            bt.logging.info("No saved scores in validator DB")
            return

        uid_map = self._db.get_all_uids()
        loaded = 0
        for (address, model_index), data in saved.items():
            uid = uid_map.get(address)
            if uid is None:
                continue

            if uid not in self.scorer.states:
                self.scorer.states[uid] = MinerScoreState(uid=uid, address=address)

            state = self.scorer.states[uid]
            if model_index not in state.entries:
                state.entries[model_index] = ModelEntryScore(
                    model_id=data["model_id"],
                    model_index=model_index,
                )

            entry = state.entries[model_index]
            entry.ema_score = data["ema_score"]
            entry.total_epochs = data["total_epochs"]
            entry.scored_epochs = data["scored_epochs"]
            loaded += 1

        if loaded:
            bt.logging.info(f"Loaded {loaded} score entries from validator DB")

    def _set_weights(self, weights: Dict[int, float]):
        """Set weights on Bittensor substrate."""
        if not weights:
            return

        uids = list(weights.keys())
        vals = [weights[uid] for uid in uids]

        max_val = max(vals) if vals else 1.0
        if max_val <= 0:
            return

        import torch
        uid_tensor = torch.tensor(uids, dtype=torch.long)
        weight_tensor = torch.tensor(
            [int(v / max_val * 65535) for v in vals],
            dtype=torch.long,
        )

        subtensor_obj = None
        try:
            SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            subtensor_obj = SubtensorCls(network=self.config.subtensor_network)
            subtensor_obj.set_weights(
                wallet=self._wallet,
                netuid=self.config.netuid,
                uids=uid_tensor,
                weights=weight_tensor,
                version_key=spec_version,
            )
            bt.logging.success(
                f"Weights set for {len(uids)} UIDs (version_key={spec_version}): {dict(zip(uids, vals))}",
            )
        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")
            raise
        finally:
            if subtensor_obj is not None:
                self._close_subtensor(subtensor_obj)

    def _get_current_block(self) -> int:
        """Get the current best/head block number."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        try:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self._get_current_head_block_and_hash)
                block, _hash, real = future.result(timeout=15)
                if real and block > 0:
                    return block
                return self._last_known_block
        except _TE:
            bt.logging.warning("get_current_block timed out (15s) — reconnecting")
            self.__subtensor = None  # force reconnect on next call
            return self._last_known_block
        except Exception:
            return self._last_known_block

    _cached_metagraph_line: str = ""

    def _refresh_metagraph_stats(self) -> None:
        """Fetch metagraph from RPC and cache the stats line. Called every ~12 min."""
        try:
            mg = self._metagraph
            if mg is None:
                mg = self._subtensor.metagraph(self.config.netuid)
                self._metagraph = mg
            else:
                mg.sync(subtensor=self._subtensor, lite=True)

            uid = None
            ss58 = self._validator_hotkey_ss58
            for i in range(mg.n.item()):
                if mg.hotkeys[i] == ss58:
                    uid = i
                    break
            if uid is None:
                return

            _get = lambda attr: float(getattr(mg, attr)[uid]) if hasattr(mg, attr) else 0.0
            self._cached_metagraph_parts = (
                f"UID {uid}",
                f"vtrust={_get('validator_trust'):.2f}",
                f"dividends={_get('dividends'):.4f}",
                f"emission={_get('emission'):.2f}α/tempo",
                f"stake={_get('stake'):.2f}α",
            )
            _block = self._last_known_block or 0
            self._cached_metagraph_line = f"Metagraph | block={_block} | {' | '.join(self._cached_metagraph_parts)}"
        except Exception as e:
            bt.logging.debug(f"Metagraph refresh failed: {e}")

    def _get_current_block_with_retry(self, max_attempts: int = 30) -> int:
        """Get current best/head block with retry — used at startup only."""
        import random
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        for attempt in range(1, max_attempts + 1):
            try:
                bt.logging.debug(f"get_current_head_block attempt {attempt}/{max_attempts}...")
                with ThreadPoolExecutor(1) as pool:
                    future = pool.submit(self._get_current_head_block_and_hash)
                    block, _hash, real = future.result(timeout=30)
                if real and block > 0:
                    bt.logging.debug(f"get_current_head_block: block={block}")
                    return block
            except _TE:
                bt.logging.warning(
                    f"get_current_head_block timed out after 30s (attempt {attempt}/{max_attempts}) — "
                    f"Subtensor WS may be hanging",
                )
                # Force reconnect on next attempt by clearing cached subtensor
                self.__subtensor = None
            except Exception as e:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Cannot get current block after {max_attempts} attempts: {e}"
                    ) from e
            delay = min(2 ** attempt * 3, 120) + random.uniform(0, 5)
            bt.logging.warning(
                f"Cannot get current block (attempt {attempt}/{max_attempts}), retrying in {delay:.0f}s",
            )
            time.sleep(delay)

    # ------------------------------------------------------------------
    # Main loop (unchanged)
    # ------------------------------------------------------------------

    def main_loop(self):
        """Run the validator via WebSocket subscription to current-head block headers.

        Uses substrate WebSocket subscription for real-time block tracking.
        Falls back to polling if subscription is unavailable.
        """
        bt.logging.info(
            f"Starting validator (epoch={self.config.epoch_blocks} blocks, "
            f"grace={self.config.epoch_grace_blocks} blocks, "
            f"canary_small={self.config.canary_small_count}, "
            f"canary_full={self.config.canary_full_context_count})",
        )

        # Jump to current block — never replay historical blocks after a
        # chain reset or fast-sync.  Align to the next epoch boundary so we
        # start with a clean epoch.
        current = self._get_current_block_with_retry()
        epoch_blocks = self.config.epoch_blocks
        blocks_into_epoch = current % epoch_blocks
        current_epoch_start = current - blocks_into_epoch
        start_at_current = str(os.getenv("VERATHOS_VALIDATOR_START_AT_CURRENT_BLOCK", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if start_at_current:
            self._sync_block = current
        elif blocks_into_epoch <= epoch_blocks // 4:
            # Early in epoch — start from current epoch boundary so we can
            # still schedule and run tests in this epoch.
            self._sync_block = current_epoch_start
        else:
            # Too far into epoch for a full test cycle — wait for next.
            self._sync_block = current_epoch_start + epoch_blocks
        bt.logging.info(f"Sync: current block={current}, epoch_offset={blocks_into_epoch}/{epoch_blocks}, will start processing at block {self._sync_block}")

        self._run_with_streaming()

    @staticmethod
    def _block_number_from_header(block_header: object) -> Optional[int]:
        value = getattr(block_header, "value", block_header)
        if isinstance(value, dict):
            header = value.get("header") if isinstance(value.get("header"), dict) else value
            number = header.get("number")
            if number is not None:
                try:
                    return int(number)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _close_subtensor(subtensor_obj) -> None:
        for obj in (subtensor_obj, getattr(subtensor_obj, "substrate", None)):
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _head_hash_arg(raw_hash: object, hash_bytes: bytes) -> str:
        if isinstance(raw_hash, str) and raw_hash.startswith("0x"):
            return raw_hash
        return "0x" + bytes(hash_bytes).hex()

    def _get_current_finalized_block_and_hash(
        self,
        subtensor_obj: object | None = None,
    ) -> tuple[int, bytes | None, bool]:
        target = subtensor_obj if subtensor_obj is not None else self._subtensor
        substrate = getattr(target, "substrate", None)
        if substrate is None:
            return 0, None, False
        try:
            response = substrate.rpc_request("chain_getFinalizedHead", [])
            raw_hash = response.get("result") if isinstance(response, dict) else response
            block_hash = self._coerce_block_hash(raw_hash)
            if block_hash is None:
                return 0, None, False
            header = substrate.get_block_header(
                block_hash=self._head_hash_arg(raw_hash, block_hash)
            )
            block_number = self._block_number_from_header(header)
            if block_number is None:
                return 0, None, False
            return int(block_number), block_hash, True
        except Exception as exc:
            bt.logging.debug(f"Finalized head lookup failed: {exc}")
            return 0, None, False

    def _get_current_head_block_and_hash(
        self,
        subtensor_obj: object | None = None,
    ) -> tuple[int, bytes | None, bool]:
        target = subtensor_obj if subtensor_obj is not None else self._subtensor
        try:
            method = getattr(target, "get_current_block", None)
            if callable(method):
                block_number = int(method())
            else:
                substrate = getattr(target, "substrate", None)
                if substrate is None:
                    return 0, None, False
                header = substrate.get_chain_head()
                block_number = self._block_number_from_header(header) or 0
            if block_number <= 0:
                return 0, None, False
            block_hash, real = self._get_chain_block_hash(block_number, target)
            return block_number, block_hash if real else None, real
        except Exception as exc:
            bt.logging.debug(f"Current head lookup failed: {exc}")
            return 0, None, False

    def _block_stream_watchdog_s(self) -> float:
        raw = os.getenv("VERATHOS_BLOCK_STREAM_WATCHDOG_S", "30")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 30.0
        return max(5.0, value)

    def _block_stream_fallback_poll_s(self) -> float:
        raw = getattr(
            self.config,
            "capacity_audit_worker_poll_s",
            os.getenv("VERATHOS_CAPACITY_AUDIT_WORKER_POLL_S", "2"),
        )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 2.0
        return max(0.1, value)

    def _process_current_head_block_range(
        self,
        last_block: int,
        current_block: int,
        subtensor_obj: object | None = None,
        *,
        current_hash: bytes | None = None,
        current_hash_real: bool = True,
    ) -> int:
        if current_block <= last_block:
            return last_block
        self._last_known_block = current_block
        for block_num in range(last_block + 1, current_block + 1):
            if not self._running:
                break
            if block_num == current_block and current_hash is not None:
                block_hash, block_hash_real = current_hash, current_hash_real
            else:
                block_hash, block_hash_real = self._get_chain_block_hash(
                    block_num,
                    subtensor_obj,
                )
            try:
                self.on_finalized_block(
                    block_num,
                    block_hash,
                    block_hash_real=block_hash_real,
                )
            except Exception as e:
                bt.logging.debug(f"Block {block_num} processing: {e}")
            last_block = block_num
        confirmer = getattr(self, "_confirm_capacity_audit_finalized_blocks", None)
        if callable(confirmer):
            confirmer(subtensor_obj)
        return last_block

    def _poll_current_head_catch_up(self, last_block: int) -> int:
        current, current_hash, current_hash_real = self._get_current_head_block_and_hash()
        if current <= 0:
            bt.logging.debug("Current-head catch-up skipped: no valid head block")
            return last_block

        # Show progress while waiting for sync block
        if current < self._sync_block:
            blocks_left = self._sync_block - current
            if current % 10 == 0:
                bt.logging.info(
                    f"Block {current} | waiting for epoch boundary "
                    f"(block {self._sync_block}, ~{blocks_left * 12 // 60}min)",
                )
            if current % 60 == 0:
                self._refresh_metagraph_stats()
            if current % 5 == 0 and hasattr(self, '_cached_metagraph_parts'):
                bt.logging.info(f"Metagraph | block={current} | {' | '.join(self._cached_metagraph_parts)}")

        return self._process_current_head_block_range(
            last_block,
            current,
            current_hash=current_hash,
            current_hash_real=current_hash_real,
        )

    def _run_with_streaming(self):
        """Use current-head streaming as primary path with polling catch-up."""
        import bittensor as bt

        last_block = self._sync_block - 1
        watchdog_s = self._block_stream_watchdog_s()
        fallback_poll_s = self._block_stream_fallback_poll_s()
        while self._running:
            fresh_sub = None
            try:
                bt.logging.info("Creating fresh Subtensor connection for current-head block stream...")
                SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
                fresh_sub = SubtensorCls(network=self.config.subtensor_network)
                substrate = getattr(fresh_sub, "substrate", None)
                subscribe = getattr(substrate, "subscribe_block_headers", None)
                if subscribe is None:
                    bt.logging.warning("Current-head block stream unavailable; using polling catch-up")
                    last_block = self._poll_current_head_catch_up(last_block)
                    time.sleep(fallback_poll_s)
                    continue

                state = SimpleNamespace(
                    error=None,
                    last_block=last_block,
                    last_header_at=0.0,
                    active=False,
                    started_at=time.monotonic(),
                )
                lock = threading.Lock()
                process_lock = threading.Lock()
                stop_event = threading.Event()
                last_catch_up_at = 0.0

                def callback(block_header):
                    if not self._running or stop_event.is_set():
                        raise StopIteration("Validator shutting down")
                    block_number = self._block_number_from_header(block_header)
                    if block_number is None:
                        return None
                    with process_lock:
                        with lock:
                            state.last_header_at = time.monotonic()
                            state.active = True
                            base_block = int(state.last_block)
                        try:
                            block_hash, block_hash_real = self._get_chain_block_hash(
                                block_number,
                                fresh_sub,
                            )
                            new_last = self._process_current_head_block_range(
                                base_block,
                                block_number,
                                fresh_sub,
                                current_hash=block_hash,
                                current_hash_real=block_hash_real,
                            )
                        finally:
                            with lock:
                                state.active = False
                        with lock:
                            state.last_block = new_last
                    return None

                def catch_up_if_due(now: float, last_block_snapshot: int, active: bool) -> int:
                    nonlocal last_catch_up_at
                    if active or now - last_catch_up_at < fallback_poll_s:
                        return last_block_snapshot
                    last_catch_up_at = now
                    try:
                        if not process_lock.acquire(blocking=False):
                            return last_block_snapshot
                        try:
                            with lock:
                                base_block = int(state.last_block)
                            new_last = self._poll_current_head_catch_up(base_block)
                        finally:
                            process_lock.release()
                        with lock:
                            if int(new_last) > int(state.last_block):
                                state.last_block = int(new_last)
                        if int(new_last) > int(last_block_snapshot):
                            bt.logging.debug(
                                f"Current-head stream catch-up advanced "
                                f"last_block={last_block_snapshot}->{new_last}"
                            )
                        return max(int(last_block_snapshot), int(new_last))
                    except Exception as exc:
                        bt.logging.debug(f"Current-head stream catch-up skipped: {exc}")
                        return last_block_snapshot

                def run_subscription():
                    try:
                        subscribe(callback, finalized_only=False)
                    except Exception as exc:
                        with lock:
                            state.error = exc

                thread = threading.Thread(
                    target=run_subscription,
                    name="validator-current-block-stream",
                    daemon=True,
                )
                thread.start()
                bt.logging.info(
                    f"Subscribing to current-head block headers "
                    f"(watchdog_s={watchdog_s:g}, fallback_poll_s={fallback_poll_s:g})"
                )

                while self._running and thread.is_alive():
                    time.sleep(min(1.0, max(0.1, fallback_poll_s)))
                    now = time.monotonic()
                    with lock:
                        error = state.error
                        active = bool(state.active)
                        last_header_at = float(state.last_header_at or 0.0)
                        last_block = int(state.last_block)
                        started_at = float(state.started_at)
                    last_block = catch_up_if_due(now, last_block, active)
                    if error is not None:
                        bt.logging.warning(f"Current-head block stream ended: {error}")
                        break
                    reference_at = last_header_at or started_at
                    if not active and now - reference_at > watchdog_s:
                        bt.logging.warning(
                            f"Current-head block stream stale for {now - reference_at:.1f}s; "
                            "reconnecting after polling catch-up"
                        )
                        break

                stop_event.set()
                last_block = int(getattr(state, "last_block", last_block) or last_block)
                self._close_subtensor(fresh_sub)
                fresh_sub = None
                if self._running:
                    last_block = self._poll_current_head_catch_up(last_block)
            except Exception as exc:
                bt.logging.error(f"Current-head block stream error: {exc}")
                last_block = self._poll_current_head_catch_up(last_block)
                time.sleep(fallback_poll_s)
            finally:
                if fresh_sub is not None:
                    self._close_subtensor(fresh_sub)

    def _run_with_polling(self):
        """Fallback: poll for new blocks periodically."""
        # Start one block before sync_block so the epoch boundary block
        # (sync_block itself) is included in range(last_block + 1, ...).
        # on_finalized_block() already skips blocks < _sync_block.
        last_block = self._sync_block - 1
        bt.logging.info(f"Running in polling mode (12s interval, starting at block {last_block})")
        poll_backoff = 12  # seconds, increases on error

        while self._running:
            try:
                last_block = self._poll_current_head_catch_up(last_block)
                poll_backoff = 12  # Reset on success
            except Exception as e:
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                if is_rate_limit:
                    poll_backoff = min(poll_backoff * 2, 120)
                    # Normal exponential backoff on chain rate limit.
                    bt.logging.info(f"Polling rate-limited, backing off to {poll_backoff}s")
                else:
                    bt.logging.error(f"Polling error: {e}")

            # Sleep for poll_backoff seconds (12s normal, up to 120s on rate limit)
            for _ in range(poll_backoff):
                if not self._running:
                    break
                time.sleep(1)

    def shutdown(self):
        self._running = False
        if self._capacity_audit_server is not None:
            self._capacity_audit_server.should_exit = True
        self._executor.shutdown(wait=False)
        self._control_executor.shutdown(wait=False)
        self._capacity_audit_executor.shutdown(wait=False)
        self._capacity_audit_discovery_executor.shutdown(wait=False)
        self._capacity_audit_proof_executor.shutdown(wait=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Verathos Validator Neuron")
    parser.add_argument("--wallet", default="default")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON. If omitted, derived from --subtensor-network.")
    parser.add_argument("--subtensor-network", default="test",
                        help="Bittensor network (test or finney). Selects chain config (contracts) and default RPC URL.")
    parser.add_argument("--subtensor-chain-endpoint", default=None,
                        help="Substrate RPC endpoint (e.g. ws://localhost:9944 for local subtensor).")
    parser.add_argument("--evm-rpc-url", default=None,
                        help="EVM RPC endpoint (e.g. http://localhost:9944 for local subtensor). "
                             "If omitted, uses network default (https://lite.chain.opentensor.ai for mainnet).")
    parser.add_argument("--ema-alpha", type=float, default=None,
                        help="EMA smoothing factor for scores (default: 0.1). "
                             "Higher = more responsive to recent epochs. "
                             "Also settable via VERATHOS_EMA_ALPHA env var.")
    # Auto-update
    parser.add_argument("--auto-update", action="store_true",
                        help="Enable automatic code updates from git remote. "
                             "Checks every 30 min, pulls and restarts on new commits.")
    parser.add_argument("--auto-update-interval", type=int, default=1800,
                        help="Auto-update check interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--analytics", action="store_true",
                        help="Enable analytics database (canary_results, epoch_scores, network_receipts). "
                             "Stores detailed test results and network-wide receipts for analysis.")
    parser.add_argument("--retain-backups", action="store_true",
                        help="Keep analytics backup files (.jsonl.gz) instead of auto-deleting. "
                             "By default, backup files older than 7 days are deleted after export.")
    parser.add_argument("--allow-mock-tee", action="store_true",
                        help="Allow mock TEE attestation even on mainnet (testing only).")
    parser.add_argument("--no-evm", action="store_true",
                        help="Run validator without EVM registration. Skips on-chain "
                             "registerEvm + reportOffline calls. Use this if you don't "
                             "want to fund an EVM mirror with TAO. Network still works "
                             "fine: dead miners get cleaned up via 24h lease expiry and "
                             "other validators' reportOffline votes.")
    parser.add_argument("--capacity-audit", action="store_true",
                        help="Enable hot-capacity audit windows (default mode: observe).")
    parser.add_argument("--capacity-audit-mode", default=None,
                        choices=("observe", "score_gate", "soft_gate", "enforce"),
                        help="Capacity audit verdict policy.")
    parser.add_argument("--capacity-audit-ingest-host", default=None,
                        help="Host for validator-side capacity artifact ingest.")
    parser.add_argument("--capacity-audit-ingest-port", type=int, default=None,
                        help="Port for validator-side capacity artifact ingest.")
    parser.add_argument("--capacity-audit-public-url", default=None,
                        help="Public validator audit-ingest IP:port URL to publish through axon metadata.")
    axon_group = parser.add_mutually_exclusive_group()
    axon_group.add_argument("--capacity-audit-serve-axon",
                            dest="capacity_audit_serve_axon",
                            action="store_true", default=None,
                            help="Publish validator audit ingest IP:port via Bittensor axon metadata.")
    axon_group.add_argument("--no-capacity-audit-serve-axon",
                            dest="capacity_audit_serve_axon",
                            action="store_false",
                            help="Do not publish capacity audit ingest via Bittensor axon metadata.")
    parser.add_argument("--capacity-audit-windows-per-epoch", type=int, default=None,
                        help="Number of deterministic capacity-audit windows per subnet epoch.")
    parser.add_argument("--capacity-audit-max-drain-fraction", type=float, default=None,
                        help="Maximum active endpoint fraction drained in one audit window.")
    parser.add_argument("--capacity-audit-group-stress-fraction", type=float, default=None,
                        help="Share of each audit-window budget reserved for related-slot group stress.")
    parser.add_argument("--capacity-audit-beacon-hash-count", type=int, default=None,
                        help="Number of prior chain-head hashes mixed into the audit selection beacon.")
    parser.add_argument("--capacity-audit-min-registration-age-s", type=float, default=None,
                        help="Minimum endpoint lease age before it can enter capacity-audit cohorts.")
    parser.add_argument("--capacity-audit-slot-refresh-blocks", type=int, default=None,
                        help="Background refresh cadence for cached capacity-audit eligible slots; 0 disables extra refreshes.")
    parser.add_argument("--capacity-audit-slot-snapshot-stale-blocks", type=int, default=None,
                        help="Maximum age of cached capacity-audit eligible slots before skipping a window; 0 disables staleness.")
    parser.add_argument("--capacity-audit-proof-verify-workers", type=int, default=None,
                        help="Bounded worker count for validator-side capacity proof verification.")
    parser.add_argument("--capacity-audit-lead-blocks", type=int, default=None,
                        help="Blocks between audit selection and audit start.")
    parser.add_argument("--capacity-audit-proof-challenge-delay-blocks", type=int, default=None,
                        help="Blocks between audit start and deferred proof challenge.")
    parser.add_argument("--capacity-audit-drain-seconds", type=float, default=None,
                        help="Nominal endpoint drain period before the timing deadline.")
    parser.add_argument("--capacity-audit-deadline-s", type=float, default=None,
                        help="Timing deadline, in seconds, measured from observed B_start.")
    parser.add_argument("--capacity-audit-transport-grace-s", type=float, default=None,
                        help="Additional final-receipt transport grace after the timing deadline.")
    parser.add_argument("--capacity-audit-payload-deadline-s", type=float, default=None,
                        help="Deferred proof payload timeout after the final timing receipt.")
    parser.add_argument("--capacity-audit-max-proof-payload-bytes", type=int, default=None,
                        help="Maximum capacity audit receipt/proof JSON request size.")
    parser.add_argument("--capacity-audit-require-proof-payload", action="store_true",
                        help="Treat missing deferred capacity proof payloads as hard proof misses.")
    parser.add_argument("--capacity-audit-repeat-window-epochs", type=int, default=None,
                        help="Epoch lookback window for repeated capacity-audit failures before score zeroing.")
    parser.add_argument("--capacity-audit-timing-misses-for-zero-score", type=int, default=None,
                        help="Timing misses within the repeat window required to zero score in score_gate mode.")
    parser.add_argument("--capacity-audit-hard-proof-misses-for-zero-score", type=int, default=None,
                        help="Hard proof/no-show misses within the repeat window required to zero score in score_gate mode.")
    timing_gate = parser.add_mutually_exclusive_group()
    timing_gate.add_argument("--capacity-audit-allow-timing-only-score-gate",
                             dest="capacity_audit_allow_timing_only_score_gate",
                             action="store_true", default=None,
                             help="Allow repeated timing-only misses to zero score in score_gate mode.")
    timing_gate.add_argument("--no-capacity-audit-allow-timing-only-score-gate",
                             dest="capacity_audit_allow_timing_only_score_gate",
                             action="store_false",
                             help="Do not zero score from timing-only misses; hard proof/no-show misses still count.")
    # Bittensor logging flags (--logging.debug, --logging.trace, --logging.info)
    bt.logging.add_args(parser)
    return parser.parse_args()


def main():
    from neurons.log import setup_neuron_logging, print_banner

    args = parse_args()
    setup_neuron_logging(args)

    extra_kwargs = {}
    if args.ema_alpha is not None:
        extra_kwargs["ema_alpha"] = args.ema_alpha
    if args.capacity_audit:
        extra_kwargs["capacity_audit_enabled"] = True
    if args.capacity_audit_mode is not None:
        extra_kwargs["capacity_audit_mode"] = args.capacity_audit_mode
    if args.capacity_audit_ingest_host is not None:
        extra_kwargs["capacity_audit_ingest_host"] = args.capacity_audit_ingest_host
    if args.capacity_audit_ingest_port is not None:
        extra_kwargs["capacity_audit_ingest_port"] = args.capacity_audit_ingest_port
    if args.capacity_audit_public_url is not None:
        extra_kwargs["capacity_audit_public_url"] = args.capacity_audit_public_url
    if args.capacity_audit_serve_axon is not None:
        extra_kwargs["capacity_audit_serve_axon"] = args.capacity_audit_serve_axon
    if args.capacity_audit_windows_per_epoch is not None:
        extra_kwargs["capacity_audit_windows_per_epoch"] = args.capacity_audit_windows_per_epoch
    if args.capacity_audit_max_drain_fraction is not None:
        extra_kwargs["capacity_audit_max_drain_fraction"] = args.capacity_audit_max_drain_fraction
    if args.capacity_audit_group_stress_fraction is not None:
        extra_kwargs["capacity_audit_group_stress_fraction"] = args.capacity_audit_group_stress_fraction
    if args.capacity_audit_beacon_hash_count is not None:
        extra_kwargs["capacity_audit_beacon_hash_count"] = args.capacity_audit_beacon_hash_count
    if args.capacity_audit_min_registration_age_s is not None:
        extra_kwargs["capacity_audit_min_registration_age_s"] = args.capacity_audit_min_registration_age_s
    if args.capacity_audit_slot_refresh_blocks is not None:
        extra_kwargs["capacity_audit_slot_refresh_blocks"] = args.capacity_audit_slot_refresh_blocks
    if args.capacity_audit_slot_snapshot_stale_blocks is not None:
        extra_kwargs["capacity_audit_slot_snapshot_stale_blocks"] = args.capacity_audit_slot_snapshot_stale_blocks
    if args.capacity_audit_proof_verify_workers is not None:
        extra_kwargs["capacity_audit_proof_verify_workers"] = args.capacity_audit_proof_verify_workers
    if args.capacity_audit_lead_blocks is not None:
        extra_kwargs["capacity_audit_lead_blocks"] = args.capacity_audit_lead_blocks
    if args.capacity_audit_proof_challenge_delay_blocks is not None:
        extra_kwargs["capacity_audit_proof_challenge_delay_blocks"] = args.capacity_audit_proof_challenge_delay_blocks
    if args.capacity_audit_drain_seconds is not None:
        extra_kwargs["capacity_audit_drain_seconds"] = args.capacity_audit_drain_seconds
    if args.capacity_audit_deadline_s is not None:
        extra_kwargs["capacity_audit_deadline_s"] = args.capacity_audit_deadline_s
    if args.capacity_audit_transport_grace_s is not None:
        extra_kwargs["capacity_audit_transport_grace_s"] = args.capacity_audit_transport_grace_s
    if args.capacity_audit_payload_deadline_s is not None:
        extra_kwargs["capacity_audit_payload_deadline_s"] = args.capacity_audit_payload_deadline_s
    if args.capacity_audit_max_proof_payload_bytes is not None:
        extra_kwargs["capacity_audit_max_proof_payload_bytes"] = args.capacity_audit_max_proof_payload_bytes
    if args.capacity_audit_require_proof_payload:
        extra_kwargs["capacity_audit_require_proof_payload"] = True
    if args.capacity_audit_repeat_window_epochs is not None:
        extra_kwargs["capacity_audit_repeat_window_epochs"] = args.capacity_audit_repeat_window_epochs
    if args.capacity_audit_timing_misses_for_zero_score is not None:
        extra_kwargs["capacity_audit_timing_misses_for_zero_score"] = args.capacity_audit_timing_misses_for_zero_score
    if args.capacity_audit_hard_proof_misses_for_zero_score is not None:
        extra_kwargs["capacity_audit_hard_proof_misses_for_zero_score"] = args.capacity_audit_hard_proof_misses_for_zero_score
    if args.capacity_audit_allow_timing_only_score_gate is not None:
        extra_kwargs["capacity_audit_allow_timing_only_score_gate"] = args.capacity_audit_allow_timing_only_score_gate
    config = NeuronConfig.from_env(
        wallet_name=args.wallet,
        hotkey_name=args.hotkey,
        netuid=args.netuid,
        subtensor_network=args.subtensor_network,
        **extra_kwargs,
    )
    resolved_chain_path = ChainConfig.resolve_config_path(
        args.chain_config, args.subtensor_network,
    )
    if resolved_chain_path is None:
        bt.logging.error("Provide --chain-config or --subtensor-network (test/finney)")
        sys.exit(1)
    args.chain_config = resolved_chain_path

    # Resolve EVM RPC URL: explicit --evm-rpc-url > subtensor endpoint > network default
    evm_rpc_explicit = getattr(args, "evm_rpc_url", None)
    if evm_rpc_explicit:
        rpc_override = evm_rpc_explicit
    else:
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

    # If explicit chain endpoint, also use it for Substrate (same port serves both)
    if getattr(args, "subtensor_chain_endpoint", None):
        ep = args.subtensor_chain_endpoint
        ws_ep = ep.replace("http://", "ws://").replace("https://", "wss://")
        config.subtensor_network = ws_ep  # Subtensor() accepts ws:// URL as network

    config.allow_mock_tee = getattr(args, "allow_mock_tee", False)
    config.no_evm = getattr(args, "no_evm", False)
    neuron = ValidatorNeuron(config)
    if args.analytics:
        bt.logging.info("Analytics database enabled (--analytics)")
    else:
        neuron._db._analytics = False

    def signal_handler(sig, _frame):
        bt.logging.info(f"Received signal {sig}, shutting down")
        neuron.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    neuron.setup()
    neuron._start_capacity_audit_ingest_server()
    neuron._ensure_capacity_audit_axon_served()

    # ── Startup banner ──
    network = args.subtensor_network or "test"
    print_banner(
        "Validator",
        network=network,
        netuid=config.netuid,
        wallet=args.wallet,
        hotkey=args.hotkey,
        evm=neuron.evm_addr or "",
        spec_ver=f"{version_str} ({spec_version})",
        vali_ver=f"{validator_version_str} ({validator_version})",
        ema_alpha=config.ema_alpha,
        auto_update="enabled" if args.auto_update else "disabled",
    )

    # ── Auto-updater ──
    if args.auto_update:
        from neurons.auto_update import AutoUpdater

        def _validator_busy() -> bool:
            """Don't restart during epoch close or weight setting."""
            return neuron._pending_epoch_close is not None

        neuron._auto_updater = AutoUpdater(
            role="validator",
            check_interval=args.auto_update_interval,
            busy_check=_validator_busy,
        )
        neuron._auto_updater.start()

    # Discover miners + enrich with SS58 at startup (same logic as epoch boundary)
    # so shared_state has full data immediately — proxy gets SS58 without waiting.
    try:
        neuron._epoch_miners = discover_active_miners(
            neuron._miner_client, neuron._model_client,
        )
        neuron._epoch_miners_discovery_valid = True
        neuron._enrich_miners_from_metagraph(neuron._epoch_miners)
        # Fetch hardware metadata from miners at startup (best-effort)
        from concurrent.futures import as_completed as _as_completed
        _hw_futs = {neuron._control_executor.submit(neuron._fetch_miner_hardware, m): m for m in neuron._epoch_miners}
        for f in _as_completed(_hw_futs, timeout=10):
            try:
                f.result()
            except Exception:
                pass
        for miner in neuron._epoch_miners:
            neuron._db.upsert_entry(
                address=miner.address, model_index=miner.model_index,
                model_id=miner.model_id, endpoint=miner.endpoint,
                quant=miner.quant, max_context_len=miner.max_context_len,
                epoch=0, hotkey_ss58=miner.hotkey_ss58, coldkey_ss58=miner.coldkey_ss58,
                tee_enabled=getattr(miner, "tee_enabled", False),
                tee_platform=getattr(miner, "tee_platform", ""),
                gpu_name=miner.gpu_name,
                gpu_count=miner.gpu_count,
                vram_gb=miner.vram_gb,
                compute_capability=miner.compute_capability,
                gpu_uuids=miner.gpu_uuids,
            )
        neuron._refresh_capacity_audit_slot_snapshot_from_miners(
            neuron._epoch_miners,
            block_number=getattr(neuron, "_last_known_block", 0) or 0,
            source="startup",
        )
        # Re-apply UIDs after upsert_entry created the rows.
        # _enrich_miners_from_metagraph calls set_uid (UPDATE) before
        # upsert_entry (INSERT), so the UPDATE is a no-op for new miners.
        for miner in neuron._epoch_miners:
            uid_val = neuron._db.get_uid(miner.address)
            if uid_val is None:
                try:
                    uid_val = neuron._miner_client.get_associated_uid(miner.address)
                    if uid_val is not None:
                        neuron._db.set_uid(miner.address, uid_val)
                except Exception:
                    pass
        bt.logging.info(f"Startup discovery: {len(neuron._epoch_miners)} miners enriched")
        # Refresh blacklist at boot — without this, the first weight-set after
        # restart fires before _close_epoch with an empty _blacklisted_uids set.
        neuron._refresh_blacklist({m.address for m in neuron._epoch_miners})
    except Exception as e:
        neuron._epoch_miners_discovery_valid = False
        bt.logging.debug(f"Startup discovery failed: {e} — shared state from DB only")

    # Write shared state immediately so the proxy has data before first epoch
    neuron._write_shared_state()

    bt.logging.info("Entering main loop...")
    neuron.main_loop()


if __name__ == "__main__":
    main()
