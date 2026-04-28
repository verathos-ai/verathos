#!/usr/bin/env python3
"""ValidatorNeuron — epoch-based canary testing for Verathos.

Lifecycle:
1. Ensure hotkey is linked (for reportOffline access).
2. Subscribe to finalized block headers via WebSocket.
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from typing import Dict, List, Optional, Set, Tuple

import bittensor as bt
import httpx

from neurons.canary import CanaryScheduler, CanaryTest
from neurons.config import NeuronConfig
from neurons.discovery import ActiveMiner, discover_active_miners
from neurons.version import spec_version, version_str, validator_version, validator_version_str
from neurons.receipts import (
    ServiceReceipt,
    create_receipt,
    receipt_from_dict,
    receipt_to_dict,
    verify_service_receipt,
)
from neurons.scoring import (
    CompositeScorer,
    EpochOutcome,
    ModelEntryScore,
    MinerScoreState,
    ProbationTracker,
    compute_demand_bonus,
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

# ---------------------------------------------------------------------------
# Tokenizer cache for input commitment verification
# ---------------------------------------------------------------------------

_tokenizer_cache: Dict[str, object] = {}


def _get_tokenizer(model_id: str):
    """Load and cache a tokenizer for input commitment verification.

    Tokenizers are lightweight (~1-5 MB each, CPU only, no model weights).
    Cached after first load so repeated canary tests for the same model
    are instant.
    """
    if model_id in _tokenizer_cache:
        return _tokenizer_cache[model_id]

    from transformers import AutoTokenizer
    bt.logging.info(f"Loading tokenizer for input commitment: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
        )

        # SQLite-backed validator state database
        db_path = os.path.join(
            os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
            "verathos_validator.db",
        )
        self._db = ValidatorStateDB(db_path=db_path)

        self.evm_pk = ""
        self.evm_addr = ""
        self._model_client = None
        self._miner_client = None
        self._subnet_config_client = None
        self._blacklisted_uids: set = set()
        self._burn_uid: int = 0
        self._scoring = ScoringParams()
        self._bt_module = None
        self.__subtensor = None

        # SS58 cache: EVM address (lowercase) → {hotkey_ss58, coldkey_ss58}
        self._ss58_cache: Dict[str, Dict[str, str]] = {}

        # Epoch state
        self._current_epoch: int = 0
        self._epoch_start_block: int = 0
        self._canary_scheduler: Optional[CanaryScheduler] = None
        self._epoch_miners: List[ActiveMiner] = []
        # {(miner_address, model_index): expected_receipt_count}
        self._expected_receipts: Dict[Tuple[str, int], int] = {}
        # {(miner_address, model_index): 503_skip_count} — reset each epoch
        self._busy_skips: Dict[Tuple[str, int], int] = {}
        # Miners that entered probation via busy-skips (not real proof failure)
        # — maps to list of unix timestamps when 503s occurred, used to verify
        # organic receipts overlap temporally (miner was genuinely busy then)
        self._busy_skip_probations: Dict[Tuple[str, int], List[int]] = {}
        # {model_id: ModelSpec} — cached per epoch, avoids RPC per canary
        self._model_spec_cache: Dict[str, object] = {}
        # Epoch close state
        self._pending_epoch_close: Optional[int] = None
        self._auto_updater = None  # Set by main() if --auto-update
        self._epoch_close_block: int = 0
        self._epoch_close_retry_after: float = 0.0  # monotonic time
        self._epoch_close_backoff: float = 30.0  # seconds, doubles on failure
        self._last_known_block: int = 0  # fallback for _get_current_block

        # Thread pool for concurrent canary tests
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_verifications,
        )

    @property
    def _subtensor(self):
        """Lazy Subtensor connection — only connects when actually needed."""
        if self.__subtensor is None:
            bt_log = __import__("bittensor").logging
            bt_log.info("Connecting to Subtensor...")
            bt = self._bt_module
            SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            self.__subtensor = SubtensorCls(network=self.config.subtensor_network)
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

        # Validator signing key for receipts (Ed25519, NOT Sr25519)
        from nacl.signing import SigningKey as _SK
        self._validator_hotkey_bytes = bytes(_SK(hotkey_seed[:32]).verify_key)
        self._validator_private_key = hotkey_seed

        # SS58 hotkey address for Sr25519 request signing (miner auth)
        self._validator_hotkey_ss58 = wallet.hotkey.ss58_address

        # Ensure contract-level EVM → UID mapping exists (needs wallet + subtensor + hotkey_seed above)
        self._ensure_evm_registered()
        self._ensure_validator_registry_registered()

        # Resolve subnet owner UID (burn target) — never changes at runtime
        self._burn_uid = self._resolve_burn_uid()

        # Load persisted scores from DB into in-memory scorer
        self._load_scores_from_db()

        # Block from which to start processing.  Set in main_loop() to the
        # current finalized block so we never replay historical blocks after
        # a chain reset / fast-sync.  All blocks before this are silently
        # skipped.
        self._sync_block: int = 0

    def _ensure_evm_registered(self):
        """Ensure registerEvm(uid) has been called on the current MinerRegistry."""
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
            bt.logging.error(
                f"registerEvm({uid}) on MinerRegistry failed: {e}. "
                f"The validator cannot verify miners without EVM registration. "
                f"Ensure the EVM account {self.evm_addr} has sufficient TAO for gas."
            )
            sys.exit(1)

    def _ensure_validator_registry_registered(self):
        """Register on ValidatorRegistry with empty endpoint (participation signal).

        Validators that don't run a proxy register with empty endpoint so they
        appear in the participation count. Proxies overwrite the endpoint later
        via their own register(real_url) call at startup.
        """
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

            # Register with empty endpoint if not already registered
            if not vr.is_active_validator(self.evm_addr):
                bt.logging.info("ValidatorRegistry: registering with empty endpoint (no proxy)")
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

    def on_finalized_block(self, block_number: int, block_hash: bytes):
        """Called by WebSocket subscription on each finalized block.

        Drives the epoch lifecycle: start epoch, dispatch canary tests,
        close epoch (pull receipts + score), set weights.
        """
        # Skip historical blocks — only process from sync point onward.
        if block_number < self._sync_block:
            return

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
        if block_number % 5 == 0 and hasattr(self, '_cached_metagraph_parts'):
            bt.logging.info(f"Metagraph | block={block_number} | {' | '.join(self._cached_metagraph_parts)}")

        # 1. Epoch boundary → start new epoch
        if block_number % epoch_blocks == 0:
            # If there's a pending epoch close, close it first
            if self._pending_epoch_close is not None:
                self._try_close_epoch(self._pending_epoch_close)

            self._start_new_epoch(block_number)

        # 1b. Retry failed epoch start — if we're past the boundary but the
        # canary scheduler is stale (e.g. _start_new_epoch threw on the
        # boundary block), retry with the current epoch's start block.
        elif (
            blocks_into_epoch <= 30  # only retry in first ~6 min
            and self._pending_epoch_close is not None
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
            pending = self._canary_scheduler.get_pending_tests(block_number)
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
            weights = self.scorer.get_weights()
            if weights:
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
                emission_burn = self._scoring.emission_burn
                if emission_burn > 0:
                    for uid in list(weights.keys()):
                        weights[uid] *= (1.0 - emission_burn)
                    weights[self._burn_uid] = weights.get(self._burn_uid, 0.0) + emission_burn
                    bt.logging.info(f"Emission burn: {emission_burn:.0%} to UID {self._burn_uid}")
                # Retry weight-setting up to 3 times — a single failure
                # otherwise means a 72-min gap before the next attempt.
                for _sw_attempt in range(1, 4):
                    try:
                        self._set_weights(weights)
                        self._last_weights = weights  # for shared state
                        break
                    except Exception as _sw_err:
                        if _sw_attempt == 3:
                            bt.logging.error(f"set_weights failed after 3 attempts: {_sw_err}")
                        else:
                            _sw_delay = 30 * (2 ** (_sw_attempt - 1))
                            bt.logging.warning(f"set_weights attempt {_sw_attempt}/3 failed: {_sw_err} — retrying in {_sw_delay}s")
                            time.sleep(_sw_delay)

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def _start_new_epoch(self, epoch_start_block: int):
        """Start a new epoch: discover miners, plan canary tests."""
        epoch_number = epoch_start_block // self.config.epoch_blocks
        self._current_epoch = epoch_number
        self._epoch_start_block = epoch_start_block

        # Schedule epoch close for after grace window
        self._pending_epoch_close = epoch_number
        self._epoch_close_block = (
            epoch_start_block + self.config.epoch_blocks + self.config.epoch_grace_blocks
        )

        t0 = time.monotonic()

        # Discover ALL active miners
        previous_miners = list(self._epoch_miners)  # cache for fallback
        try:
            self._epoch_miners = discover_active_miners(
                self._miner_client, self._model_client,
            )
        except Exception as e:
            bt.logging.warning(f"Discovery RPC failed: {e} — will fall back to previous miners")
            self._epoch_miners = []  # triggers fallback below
        bt.logging.info(f"Epoch {epoch_number} (block {epoch_start_block}): discovered {len(self._epoch_miners)} miner entries")

        # Enrich miners with SS58 keys from metagraph (for analytics + shared state)
        self._enrich_miners_from_metagraph(self._epoch_miners)

        if not self._epoch_miners:
            # RPC failure (e.g. 429 rate limit) — fall back to previous
            # epoch's miners so canary testing continues uninterrupted.
            # Miners don't change between epochs in practice.
            if previous_miners:
                self._epoch_miners = previous_miners
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
            self._executor.submit(_tcp_alive, m.endpoint): m
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
        self._epoch_miners = alive_miners
        if not self._epoch_miners:
            bt.logging.warning(f"Epoch {epoch_number}: no reachable miners after TCP pre-filter")
            self._canary_scheduler = None
            return

        # ── Identity verification: filter out hijacked endpoints ────
        # CRITICAL: one failing miner must NOT block others.  The per-miner
        # budget is bounded (deadline inside _verify_miner_identity), and the
        # overall as_completed loop is wrapped in try/except so a single
        # unresponsive miner can never stall epoch start for others.
        verified_miners = []
        id_futures = {}
        for miner in self._epoch_miners:
            future = self._executor.submit(self._verify_miner_identity, miner)
            id_futures[future] = miner

        # Overall deadline: scales with miner count / pool workers, but capped
        # at 120s.  Beyond that we're better off including unverified miners
        # (canary proofs will catch any bad actors) than delaying epoch start.
        _id_batches = len(self._epoch_miners) // self.config.max_concurrent_verifications + 1
        overall_deadline = min(120, max(
            self.config.identity_challenge_timeout + 10,
            _id_batches * 3 + 10,  # ~3s per batch (most pass fast) + 10s grace
        ))
        completed_futures = set()
        try:
            for future in as_completed(id_futures, timeout=overall_deadline):
                completed_futures.add(future)
                miner = id_futures[future]
                try:
                    result = future.result()
                    if result is False:
                        bt.logging.info(f"Identity FAILED for {miner.address[:10]} at {miner.endpoint} — excluding from epoch")
                        self._report_offline(miner)
                        continue
                    if result is None and self.config.identity_challenge_required:
                        bt.logging.info(f"Identity UNSUPPORTED for {miner.address[:10]} at {miner.endpoint} — excluding (required mode)")
                        continue
                    verified_miners.append(miner)
                except Exception as e:
                    bt.logging.info(f"Identity check error for {miner.address[:10]}: {e} — including miner")
                    verified_miners.append(miner)
        except _FuturesTimeout:
            # One or more futures didn't finish in time — exclude them from
            # this epoch but continue with the verified ones.  This prevents
            # a single stalled miner from blocking canary dispatch for all.
            stalled = [id_futures[f].address[:10] for f in id_futures if f not in completed_futures]
            bt.logging.warning(
                f"Identity verification timeout after {overall_deadline}s — "
                f"{len(stalled)} miner(s) stalled: {stalled}. "
                f"Proceeding with {len(verified_miners)} verified miners."
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
            future = self._executor.submit(self._fetch_miner_hardware, miner)
            hw_futures[future] = miner
        try:
            for future in as_completed(hw_futures, timeout=10):
                try:
                    future.result()
                except Exception:
                    pass  # Non-fatal — hardware metadata is optional
        except _FuturesTimeout:
            for f in hw_futures:
                if not f.done():
                    f.cancel()

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

        # Count expected receipts per (miner, model_index)
        for test in tests:
            key = (test.miner_address, test.model_index)
            self._expected_receipts[key] = self._expected_receipts.get(key, 0) + 1

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
        """Dispatch canary tests to thread pool for concurrent execution."""
        futures = {}
        for test in tests:
            future = self._executor.submit(
                self._execute_canary_test, test, self._current_epoch,
            )
            futures[future] = test

        # Don't block — results are collected at epoch close via receipts.
        # Wrap in try/except so a timeout doesn't crash the dispatch loop.
        try:
            for future in as_completed(futures, timeout=self.config.canary_inference_timeout):
                test = futures[future]
                try:
                    future.result()
                except Exception as e:
                    bt.logging.info(f"Canary test failed for {test.miner_address[:10]} model={test.model_id}: {e}")
        except _FuturesTimeout:
            stalled = [futures[f].miner_address[:10] for f in futures if not f.done()]
            bt.logging.warning(
                f"Canary dispatch timeout after {self.config.canary_inference_timeout}s — "
                f"{len(stalled)} tests stalled: {stalled}"
            )
            for f in futures:
                if not f.done():
                    f.cancel()

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

        max_retries = 3
        last_exc = None
        for attempt in range(max_retries + 1):
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
            bt.logging.info(f"Canary failed after {max_retries} retries (miner busy) for {test.miner_address[:10]} model={test.model_id}")
            key = (test.miner_address, test.model_index)
            reject_ts = int(time.time())
            if key in self._expected_receipts and self._expected_receipts[key] > 0:
                self._expected_receipts[key] -= 1
            self._busy_skips[key] = self._busy_skips.get(key, 0) + 1
            # Record 503 timestamp for temporal overlap check at epoch close
            self._busy_skip_probations.setdefault(key, []).append(reject_ts)
            if self._busy_skips[key] > 3:
                # Logged only — penalty is deferred to epoch close where
                # organic receipts can prove the miner was genuinely busy.
                bt.logging.info(f"Miner {test.miner_address[:10]} model_index={test.model_index} returned 503 on {self._busy_skips[key]} canaries (>3 after retries) — will evaluate at epoch close")

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
            with ValidatorClient(
                miner_url=test.miner_endpoint,
                config=verification_config,
                timeout=self.config.canary_inference_timeout,
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

                # Optional proof verification
                proof_verified = False
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
                            bt.logging.error(
                                f"Proof failure | {test.miner_address[:10]} "
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

                # Extract metrics from timing
                ttft_ms = timing.get("ttft_ms", 0.0)
                output_tokens = timing.get("output_tokens", 0)
                input_tokens = timing.get("input_tokens", 0)
                inference_ms = timing.get("inference_ms", 0.0)
                tokens_per_sec = (
                    output_tokens / (inference_ms / 1000)
                    if inference_ms > 0 and output_tokens > 0
                    else 0.0
                )

                # Push signed receipt to miner
                self._push_receipt_to_miner(
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
                )

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
                        prove_ms=timing.get("prove_ms"),
                        commitment_ms=timing.get("commitment_ms"),
                        verify_ms=sum(verify_timing.values()) if verify_timing else None,
                        commitment_hash=commitment.commitment_hash().hex() if commitment else None,
                        receipt_pushed=1,
                    )
                except Exception as _db_err:
                    bt.logging.debug(f"Failed to log canary result: {_db_err}")

        except Exception as e:
            # HTTP 503 (miner busy) is handled by the retry wrapper in
            # _execute_canary_test — if we get here it's a real error
            # (connection refused, timeout, non-503 HTTP error, etc.).
            # If this is a transport-level error AND the outer wrapper
            # hasn't retried yet, re-raise so it can retry once.
            if _transport_retry_allowed and isinstance(e, _transport_exc):
                raise
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
            # Repeated failures (>3 per epoch) are evaluated at epoch close.
            key = (test.miner_address, test.model_index)
            self._canary_errors[key] = self._canary_errors.get(key, 0) + 1
            if key in self._expected_receipts and self._expected_receipts[key] > 0:
                self._expected_receipts[key] -= 1
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
            self._close_epoch(epoch_number)
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

        # ── Pass 1: collect all receipts ──────────────────────────
        miner_receipts: Dict[str, List[ServiceReceipt]] = {}  # address -> receipts
        all_epoch_receipts: List[ServiceReceipt] = []

        # Pull receipts in parallel so one slow/dead miner can't block others.
        # Per-miner timeout is bounded inside _pull_epoch_receipts.
        receipt_futures = {}
        for miner in self._epoch_miners:
            if not self._running:
                break
            receipt_futures[
                self._executor.submit(self._pull_epoch_receipts, miner, epoch_number)
            ] = miner
        # Overall budget for all receipt pulls — scales with miner count,
        # floored by config, capped at 120s.
        _rp_timeout = min(120, max(
            self.config.epoch_receipt_pull_overall_timeout,
            len(self._epoch_miners) // self.config.max_concurrent_verifications * 3 + 10,
        ))
        try:
            for fut in as_completed(receipt_futures, timeout=_rp_timeout):
                miner = receipt_futures[fut]
                try:
                    receipts = fut.result()
                    miner_receipts.setdefault(miner.address, []).extend(receipts)
                    all_epoch_receipts.extend(receipts)
                except Exception as e:
                    bt.logging.debug(f"Receipt pull exception for {miner.address[:10]}: {e}")
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
                    f.cancel()

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

        # ── Read all scoring params from SubnetConfig (single RPC) ──
        # Fall back to last successfully read params (not hardcoded defaults)
        # so a transient RPC failure doesn't cause a one-epoch scoring glitch.
        if self._subnet_config_client is not None:
            try:
                self._scoring = self._subnet_config_client.get_scoring_params()
                self._last_good_scoring = self._scoring  # cache for fallback
                bt.logging.debug(
                    f"SubnetConfig scoring: tee={self._scoring.tee_bonus:.2f} "
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

        # Check blacklist in parallel (one RPC per unique address, cached 5min).
        # Parallel so one slow/429'd RPC doesn't block scoring of all miners.
        self._blacklisted_uids = set()
        if self._subnet_config_client is not None:
            _unique_addrs = {m.address for m in self._epoch_miners}
            _bl_futures = {
                self._executor.submit(
                    self._subnet_config_client.is_miner_blacklisted, addr
                ): addr
                for addr in _unique_addrs
            }
            try:
                for fut in as_completed(_bl_futures, timeout=15):
                    addr = _bl_futures[fut]
                    try:
                        if fut.result():
                            uid = self._resolve_uid(addr)
                            if uid is not None:
                                self._blacklisted_uids.add(uid)
                                # Policy enforcement, not a validator issue.
                                bt.logging.info(f"Miner {addr[:10]} (UID {uid}) is BLACKLISTED — score will be zeroed")
                    except Exception:
                        pass
            except _FuturesTimeout:
                bt.logging.warning("Blacklist check timeout (15s) — proceeding with partial results")

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
            bt.logging.warning(
                f"GPU UUID {_uuid[:16]}... used by {len(_eps)} endpoints — keeping best"
            )
            _eps.sort(key=lambda x: -x[2])  # highest EMA first
            for _addr, _midx, _ema in _eps[1:]:
                _sybil_skip.add((_addr, _midx))
                bt.logging.warning(f"  GPU dedup: skipping {_addr[:10]} model_index={_midx} (ema={_ema:.4f})")

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

            key = (miner.address, miner.model_index)
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
            expected = self._expected_receipts.get(key, 0)

            # Skip scoring if no canaries were dispatched AND no busy-skips.
            # The expected count is decremented on each 503, so expected==0
            # can mean either "nothing scheduled" or "all canaries got 503".
            # In the latter case we must still run the busy-skip evaluation.
            busy_skips_this_epoch = self._busy_skips.get(key, 0)
            if expected == 0 and busy_skips_this_epoch == 0:
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
            key = (miner.address, miner.model_index)
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
                else:
                    # No evidence of legitimate busyness — treat as evasion
                    bt.logging.info(f"Busy-skip EVASION for {miner.address[:10]} model_index={miner.model_index} ({len(reject_times)} 503s, only {len(organic_near_reject)} overlapping organic receipts)")
                    self._on_proof_failure(
                        miner.address, miner.model_index,
                        endpoint=getattr(miner, 'endpoint', ''))

            # ── Canary error evaluation ────────────────────────────
            # Tolerate ≤3 transient errors (network glitches, timeouts).
            # >3 errors with no legitimate excuse → probation + EMA halve.
            canary_errors = self._canary_errors.get(key, 0)
            if canary_errors > 3 and not had_proof_failure:
                bt.logging.info(
                    f"Canary error PENALTY for {miner.address[:10]} "
                    f"model_index={miner.model_index} ({canary_errors} errors "
                    f"in epoch — exceeds tolerance of 3)"
                )
                self._on_proof_failure(
                    miner.address, miner.model_index,
                    endpoint=getattr(miner, 'endpoint', ''))

            # Escalation: too long on probation → report offline on-chain
            if self._db.should_escalate(miner.address, miner.model_index, epoch_number):
                bt.logging.info(f"Probation ESCALATION for {miner.address[:10]} model_index={miner.model_index} -> reportOffline")
                self._report_offline(miner)

        # ── Zero undiscovered miners ───────────────────────────────
        # If a miner's lease expired or it wasn't discovered this epoch,
        # it's not serving — zero its EMA immediately.  This prevents
        # stale scores from persisting in get_weights() after miners
        # leave the network.  Transient issues (unreachable but still
        # discovered) are handled by the canary error penalty path above.
        _discovered_keys = {
            (m.address.lower(), m.model_index) for m in self._epoch_miners
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

    def _pull_epoch_receipts(
        self,
        miner: ActiveMiner,
        epoch_number: int,
    ) -> List[ServiceReceipt]:
        """Pull all receipts from a miner for the given epoch.

        GET /epoch/{epoch_number}/receipts — returns all accumulated receipts.
        Verifies signature + freshness for each receipt.
        """
        url = f"{miner.endpoint.rstrip('/')}/epoch/{epoch_number}/receipts"
        path = f"/epoch/{epoch_number}/receipts"

        try:
            from neurons.request_signing import sign_request
            auth_headers = sign_request(
                method="GET", path=path, body=b"",
                hotkey_ss58=self._validator_hotkey_ss58,
                hotkey_seed=self._validator_private_key,
            )
            resp = httpx.get(url, timeout=self.config.epoch_receipt_pull_timeout,
                             headers=auth_headers, verify=False)
            if resp.status_code != 200:
                bt.logging.debug(f"Receipt pull from {miner.address[:10]} returned {resp.status_code}")
                return []

            data = resp.json()
            receipt_dicts = data.get("receipts", [])

            verified = []
            for r_dict in receipt_dicts:
                try:
                    receipt = receipt_from_dict(r_dict)
                    if verify_service_receipt(receipt, epoch_number):
                        verified.append(receipt)
                except Exception as e:
                    bt.logging.debug(f"Invalid receipt from {miner.address[:10]}: {e}")

            bt.logging.debug(f"Pulled {len(verified)}/{len(receipt_dicts)} valid receipts from {miner.address[:10]}")
            return verified

        except Exception as e:
            bt.logging.debug(f"Receipt pull failed for {miner.address[:10]}: {e}")
            return []

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
    ):
        """Push a signed service receipt to a miner after verified inference."""
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
        )

        url = f"{miner_endpoint.rstrip('/')}/epoch/receipt"
        try:
            import json as _json
            from neurons.request_signing import sign_request as _sign
            receipt_body = _json.dumps(receipt_to_dict(receipt)).encode("utf-8")
            auth_headers = _sign(
                method="POST", path="/epoch/receipt", body=receipt_body,
                hotkey_ss58=self._validator_hotkey_ss58,
                hotkey_seed=self._validator_private_key,
            )
            resp = httpx.post(
                url,
                content=receipt_body,
                headers={**auth_headers, "content-type": "application/json"},
                timeout=self.config.miner_endpoint_timeout,
                verify=False,
            )
            if resp.status_code == 200:
                bt.logging.debug(f"Pushed receipt to {miner_address[:10]} model_index={model_index} epoch={epoch_number}")
            else:
                bt.logging.debug(f"Receipt push to {miner_address[:10]} returned {resp.status_code}")
        except Exception as e:
            bt.logging.debug(f"Failed to push receipt to {miner_address[:10]}: {e}")

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
        key = (miner_address, model_index)
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
        max_attempts = 6
        deadline = time.monotonic() + self.config.identity_challenge_timeout + 5
        for attempt in range(1, max_attempts + 1):
            if time.monotonic() >= deadline:
                break
            nonce = os.urandom(32)
            try:
                remaining = max(1.0, deadline - time.monotonic())
                resp = httpx.post(
                    url,
                    json={"nonce": nonce.hex()},
                    timeout=min(self.config.identity_challenge_timeout, remaining),
                    verify=False,
                )
                if resp.status_code == 200:
                    break
                if resp.status_code in (404, 405, 501):
                    return None  # endpoint doesn't support challenges
            except Exception as _ide:
                bt.logging.debug(f"Identity challenge exception for {miner.address[:10]}: {_ide}")
            if attempt < max_attempts and time.monotonic() < deadline:
                wait = min(30, deadline - time.monotonic())
                if wait <= 0:
                    break
                bt.logging.debug(f"Identity challenge retry {attempt}/{max_attempts} for {miner.address[:10]}, next in {wait:.0f}s")
                time.sleep(wait)

        if resp is None or resp.status_code != 200:
            bt.logging.debug(f"Identity challenge failed for {miner.address[:10]} after {attempt} attempts")
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

    def _fetch_miner_hardware(self, miner: ActiveMiner) -> None:
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
                return
            hw = resp.json().get("hardware")
            if not hw:
                return
            miner.gpu_name = hw.get("gpu_name", "")
            miner.gpu_count = hw.get("gpu_count", 0)
            miner.vram_gb = hw.get("vram_gb", 0)
            miner.compute_capability = hw.get("compute_capability", "")
            miner.gpu_uuids = hw.get("gpu_uuids", [])
        except Exception:
            pass  # Non-fatal

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
        if not miners:
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

        try:
            self._subtensor.set_weights(
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

    def _get_current_block(self) -> int:
        """Get the current finalized block number."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        try:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self._subtensor.get_current_block)
                return future.result(timeout=15)
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
        """Get current block with retry — used at startup only."""
        import random
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        for attempt in range(1, max_attempts + 1):
            try:
                bt.logging.debug(f"get_current_block attempt {attempt}/{max_attempts}...")
                # Use a thread + timeout to detect silent WS hangs.
                with ThreadPoolExecutor(1) as pool:
                    future = pool.submit(self._subtensor.get_current_block)
                    block = future.result(timeout=30)
                if block > 0:
                    bt.logging.debug(f"get_current_block: block={block}")
                    return block
            except _TE:
                bt.logging.warning(
                    f"get_current_block timed out after 30s (attempt {attempt}/{max_attempts}) — "
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
        """Run the validator via WebSocket subscription to finalized block headers.

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
        if blocks_into_epoch <= epoch_blocks // 4:
            # Early in epoch — start from current epoch boundary so we can
            # still schedule and run tests in this epoch.
            self._sync_block = current_epoch_start
        else:
            # Too far into epoch for a full test cycle — wait for next.
            self._sync_block = current_epoch_start + epoch_blocks
        bt.logging.info(f"Sync: current block={current}, epoch_offset={blocks_into_epoch}/{epoch_blocks}, will start processing at block {self._sync_block}")

        # WebSocket subscriptions are unreliable on testnet (silently hang
        # without delivering block headers).  Use polling — one RPC call per
        # 12s is well within rate limits.
        # TODO(mainnet): re-enable subscription with proper watchdog.
        self._run_with_polling()

    def _run_with_subscription(self):
        """Subscribe to finalized block headers via WebSocket.

        Creates a FRESH Subtensor/substrate connection for the subscription
        because the existing ``self._subtensor`` connection can have stale
        WS state that causes ``subscribe_block_headers`` to hang silently.
        """
        import bittensor as bt

        bt.logging.info("Creating fresh Subtensor connection for block subscription...")
        SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
        fresh_sub = SubtensorCls(network=self.config.subtensor_network)

        def callback(block_header):
            """Callback for subscribe_block_headers."""
            if not self._running:
                raise StopIteration("Validator shutting down")

            block_number = block_header["header"]["number"]
            block_hash = hashlib.sha256(
                str(block_number).encode() + str(block_header).encode()
            ).digest()

            self.on_finalized_block(block_number, block_hash)

        bt.logging.info("Subscribing to finalized block headers (fresh connection)...")
        fresh_sub.substrate.subscribe_block_headers(
            callback, finalized_only=True,
        )

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
                current = self._get_current_block()

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

                if current > last_block:
                    self._last_known_block = current
                    # Process each block we missed
                    for block_num in range(last_block + 1, current + 1):
                        if not self._running:
                            break
                        # Derive block hash (polling can't get real hash easily)
                        block_hash = hashlib.sha256(
                            f"block_{block_num}".encode()
                        ).digest()
                        try:
                            self.on_finalized_block(block_num, block_hash)
                        except Exception as e:
                            bt.logging.debug(f"Block {block_num} processing: {e}")
                        # Always advance — never re-process the same block.
                        # If on_finalized_block threw, we skip that block
                        # rather than re-processing it in a loop.
                        last_block = block_num
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
        self._executor.shutdown(wait=False)


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
        neuron._enrich_miners_from_metagraph(neuron._epoch_miners)
        # Fetch hardware metadata from miners at startup (best-effort)
        from concurrent.futures import as_completed as _as_completed
        _hw_futs = {neuron._executor.submit(neuron._fetch_miner_hardware, m): m for m in neuron._epoch_miners}
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
    except Exception as e:
        bt.logging.debug(f"Startup discovery failed: {e} — shared state from DB only")

    # Write shared state immediately so the proxy has data before first epoch
    neuron._write_shared_state()

    bt.logging.info("Entering main loop...")
    neuron.main_loop()


if __name__ == "__main__":
    main()
