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
   c. Score entries: utility × authenticated work × latency, update EMAs.
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
import asyncio
import hashlib
import logging
import math
import os
import signal
import json
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse
import ipaddress

import bittensor as bt
import httpx

from neurons.canary import (
    CanaryScheduler,
    CanaryTest,
    materialize_canary_prompt,
)
from neurons.capacity_audit import (
    CapacityAuditRuntimeConfig,
    CapacitySlot,
    PROTOCOL_VERSION,
    build_capacity_slot_group_key,
    capacity_audit_active_hold_seconds,
    capacity_audit_payload_cooldown_blocks,
    capacity_audit_uid_escalation_threshold,
    capacity_audit_window_fits_epoch,
    capacity_audit_window_triggered,
    capacity_gpu_pass_count,
    capacity_gpu_workload_spec,
    derive_audit_id,
    derive_audit_seed,
    derive_audit_seed_from_hashes,
    derive_proof_challenge_seed,
    derive_proof_seed,
    lease_id,
    match_gpu_class,
    select_capacity_audit_slots,
    slot_id,
    verify_artifact_signature,
)
from neurons.capacity_audit_combined import (
    CURRENT_COMBINED_PROOF_PROTOCOL_VERSION,
    LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
    combined_proof_protocol_version,
    is_combined_proof_payload,
    verify_combined_proof_payload,
)
from neurons.capacity_audit_head_observer import STATE_SIZE as CAPACITY_HEAD_STATE_SIZE
from neurons.config import NeuronConfig
from neurons.discovery import ActiveMiner, discover_active_miners
from neurons.subnet_runtime_config import (
    MaintenanceGraceConfig,
    ProofV3FailurePolicyConfig,
    ProofProtocolRolloutConfig,
    RuntimeSubnetConfigClient,
    apply_runtime_config_to_neuron_config,
    capacity_audit_config_from_neuron_config,
    legacy_v1_compatibility_active,
    maintenance_grace_active,
    maintenance_grace_config_from_neuron_config,
    proof_protocol_allowed,
    proof_v3_required,
    proof_protocol_rollout_config_from_neuron_config,
    proof_v3_failure_policy_config_from_neuron_config,
    select_proof_protocol_version,
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
    select_scoring_authority_receipts,
)
from neurons.validator_db import ValidatorStateDB
from neurons.proof_v3_failure_strikes import HardProofStrikeTracker

from verallm.chain.config import ChainConfig
from verallm.chain.mock import create_clients
from verallm.chain.types import ScoringParams
from verallm.chain.wallet import derive_evm_private_key, derive_evm_address
from verallm.api.client import ValidatorClient
from verallm.api.proof_protocol import (
    LEGACY_PROOF_PROTOCOL_VERSION,
    PROOF_PROTOCOL_V3,
    SUPPORTED_PROOF_PROTOCOL_VERSIONS,
)
from verallm.config import Config
from verallm.proof_policy import (
    CURRENT_PROOF_PROTOCOL_VERSION,
    evaluate_proof_policy,
    verify_with_proof_policy,
)
from verallm.proof_v3.canary_policy import (
    CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
    MAX_CANARY_FULL_PAIR_HOLD_SECONDS_V3,
    canary_prompt_token_tolerance_v3,
)
from verallm.registry import get_model, MODELS_BY_ID

logger = logging.getLogger(__name__)
_PROOF_V2_ARTIFACT_REFRESH_SECONDS = 3600.0
_PROOF_V3_ARTIFACT_REFRESH_SECONDS = 3600.0
_BITTENSOR_MAINNET_EVM_CHAIN_ID = 964
_HTTPS_MAINNET_REMEDIATION = (
    "HTTPS is required on mainnet. Configure TLS with scripts/setup_https.sh "
    "and re-register this endpoint using https://."
)
_PUBLIC_HTTPS_MAINNET_REMEDIATION = (
    "Register a publicly routable HTTPS endpoint on mainnet."
)


def _mainnet_endpoint_eligibility_reason(
    endpoint: str,
    *,
    netuid: int,
    chain_id: int | None = None,
) -> str:
    """Return a deterministic mainnet endpoint gate reason, if any.

    Testnet and local networks retain HTTP support. Mainnet mirrors the
    production proxy's existing SSRF/TLS admission policy so an endpoint that
    cannot receive organic traffic cannot retain a validator score.
    """

    try:
        resolved_netuid = int(netuid)
    except (TypeError, ValueError):
        return ""
    try:
        resolved_chain_id = int(chain_id) if chain_id is not None else None
    except (TypeError, ValueError):
        resolved_chain_id = None
    if resolved_netuid != 96:
        return ""
    if resolved_chain_id not in (None, 0, _BITTENSOR_MAINNET_EVM_CHAIN_ID):
        return ""

    try:
        parsed = urlparse(str(endpoint or ""))
    except Exception:
        return _PUBLIC_HTTPS_MAINNET_REMEDIATION
    if parsed.scheme.lower() != "https":
        return _HTTPS_MAINNET_REMEDIATION

    # Reuse the already-shipped proxy policy rather than maintaining a second
    # list of blocked mainnet networks in the scoring path.
    from neurons.miner_pool import _is_safe_endpoint

    if not _is_safe_endpoint(str(endpoint or ""), allow_private=False):
        return _PUBLIC_HTTPS_MAINNET_REMEDIATION
    return ""


class _ProofV3ValidatorConfigurationError(RuntimeError):
    """Local v3 configuration is unavailable; the miner is not at fault."""


@dataclass(frozen=True)
class _CapacityAuditIngressHead:
    """Validator-owned best-head observation captured before receipt queueing."""

    block: int
    generation: int
    observed_at: float
    source: str
    trustworthy: bool


def _capacity_audit_head_number(block_header: object) -> int:
    """Decode one substrate header without importing validator state."""

    value = getattr(block_header, "value", block_header)
    if not isinstance(value, dict):
        return 0
    header = value.get("header") if isinstance(value.get("header"), dict) else value
    number = header.get("number")
    if number is None:
        return 0
    try:
        return int(number, 0) if isinstance(number, str) else int(number)
    except (TypeError, ValueError):
        return 0


def _decode_scoring_authority_hotkey(ss58_address: str) -> bytes:
    """Decode one exact Substrate account id or fail closed."""

    from verallm.chain.wallet import ss58_decode

    authority = bytes(ss58_decode(str(ss58_address or "")))
    if len(authority) != 32:
        raise ValueError("decoded scoring authority is not 32 bytes")
    return authority


class _ProofV3FullPairBarrier:
    """Keep two full-canary precommits secret until both serves are frozen."""

    def __init__(self, pair_id: str, hold_seconds: int) -> None:
        try:
            pair_bytes = bytes.fromhex(pair_id)
        except (TypeError, ValueError):
            pair_bytes = b""
        if (
            len(pair_bytes) != 16
            or pair_id != pair_bytes.hex()
            or type(hold_seconds) is not int
            or not 0 < hold_seconds <= MAX_CANARY_FULL_PAIR_HOLD_SECONDS_V3
        ):
            raise _ProofV3ValidatorConfigurationError(
                "proof-v3 full-pair barrier metadata is malformed"
            )
        self.pair_id = pair_id
        self.hold_seconds = int(hold_seconds)
        self._condition = threading.Condition()
        self._next_slot = 0
        self._precommitted: Set[int] = set()
        self._hard_slots: Set[int] = set()
        self._exchanges: Dict[int, object] = {}
        self._completed: Set[int] = set()
        self._first_precommit_at: float | None = None
        self._failure: BaseException | None = None
        self._failure_attributed = False
        self._departed: Set[int] = set()

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("proof-v3 full-pair barrier expired")
        return remaining

    def _raise_failure_locked(self) -> None:
        if self._failure is not None:
            raise self._failure

    def wait_for_inference_turn(self, slot: int) -> None:
        if slot not in (0, 1):
            raise _ProofV3ValidatorConfigurationError(
                "proof-v3 full-pair slot is malformed"
            )
        deadline = time.monotonic() + self.hold_seconds
        with self._condition:
            while self._failure is None and self._next_slot != slot:
                self._condition.wait(self._remaining(deadline))
            self._raise_failure_locked()

    @staticmethod
    def _fail_exchanges(exchanges: Sequence[object]) -> None:
        for exchange in exchanges:
            fail_closed = getattr(exchange, "fail_closed", None)
            if callable(fail_closed):
                try:
                    fail_closed()
                except Exception:
                    pass

    def _pending_exchanges_locked(self) -> Tuple[object, ...]:
        return tuple(
            exchange
            for slot, exchange in self._exchanges.items()
            if slot not in self._completed
        )

    def mark_precommitted(
        self,
        slot: int,
        *,
        hard_audit: bool,
        exchange: object,
    ) -> None:
        if not callable(getattr(exchange, "fail_closed", None)):
            raise _ProofV3ValidatorConfigurationError(
                "proof-v3 full-pair exchange cannot fail closed"
            )
        with self._condition:
            self._raise_failure_locked()
            if slot != self._next_slot or slot in self._precommitted:
                raise RuntimeError(
                    "proof-v3 full-pair precommit order is inconsistent"
                )
            now = time.monotonic()
            self._precommitted.add(slot)
            self._exchanges[slot] = exchange
            if hard_audit:
                self._hard_slots.add(slot)
            self._next_slot += 1
            if slot == 0:
                self._first_precommit_at = now
            self._condition.notify_all()

    def wait_until_both_precommitted(self) -> None:
        with self._condition:
            if self._first_precommit_at is None:
                raise RuntimeError(
                    "proof-v3 full-pair wait preceded the first precommit"
                )
            deadline = self._first_precommit_at + self.hold_seconds
            while self._failure is None and len(self._precommitted) != 2:
                self._condition.wait(self._remaining(deadline))
            self._raise_failure_locked()
            if len(self._hard_slots) != 1:
                raise RuntimeError(
                    "proof-v3 full-pair must contain exactly one hard slot"
                )

    def mark_completed(self, slot: int) -> None:
        with self._condition:
            self._raise_failure_locked()
            if len(self._precommitted) != 2 or slot in self._completed:
                raise RuntimeError(
                    "proof-v3 full-pair completion order is inconsistent"
                )
            self._completed.add(slot)
            self._condition.notify_all()

    def worker_completed(self, slot: int) -> None:
        pending: Tuple[object, ...] = ()
        with self._condition:
            if slot not in (0, 1):
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair worker slot is malformed"
                )
            if self._failure is None and slot not in self._completed:
                self._failure = _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair worker exited before completion"
                )
                pending = self._pending_exchanges_locked()
            self._condition.notify_all()
        self._fail_exchanges(pending)

    def abort(self, failure: BaseException) -> bool:
        pending: Tuple[object, ...] = ()
        with self._condition:
            had_precommit = bool(self._precommitted)
            if self._failure is None:
                self._failure = failure
                pending = self._pending_exchanges_locked()
            self._condition.notify_all()
        self._fail_exchanges(pending)
        return had_precommit

    def claim_failure_attribution(self) -> bool:
        with self._condition:
            if self._failure_attributed:
                return False
            self._failure_attributed = True
            return True

    def depart(self, slot: int) -> bool:
        with self._condition:
            if slot not in (0, 1):
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair departure slot is malformed"
                )
            self._departed.add(slot)
            return len(self._departed) == 2


def _resolve_proof_v2_manifest_paths(
    cli_paths: Optional[List[str]],
    *,
    chain_id: int | None = None,
    netuid: int | None = None,
) -> tuple[str, ...]:
    """Resolve explicit, environment, then authenticated bundled manifests."""

    if cli_paths is not None:
        return tuple(path for path in cli_paths if path)
    environment_paths = tuple(
        path
        for path in os.environ.get("VERATHOS_PROOF_V2_MANIFESTS", "").split(
            os.pathsep
        )
        if path
    )
    if environment_paths:
        return environment_paths
    from verallm.proof_v2.runtime import bundled_proof_v2_manifest_paths

    return tuple(
        str(path)
        for path in bundled_proof_v2_manifest_paths(
            chain_id=chain_id,
            netuid=netuid,
        )
    )


def _resolve_proof_v3_release_paths(
    cli_paths: Optional[List[str]],
) -> tuple[str, ...]:
    """Resolve explicit then environment-configured v3 release descriptors."""

    if cli_paths is not None:
        return tuple(path for path in cli_paths if path)
    return tuple(
        path
        for path in os.environ.get("VERATHOS_PROOF_V3_RELEASES", "").split(
            os.pathsep
        )
        if path
    )


def _resolve_proof_v3_canary_policy_path(cli_path: Optional[str]) -> str:
    """Resolve an explicit or environment-configured signed canary policy."""

    if cli_path is not None:
        return str(cli_path).strip()
    return os.environ.get("VERATHOS_PROOF_V3_CANARY_POLICY", "").strip()


def _effective_canary_counts(
    config: object,
    signed_policy: object | None,
    *,
    hard_audit_enabled: bool,
) -> tuple[int, int, int]:
    """Resolve the exact official canary inventory for one endpoint."""

    if signed_policy is not None:
        if hard_audit_enabled:
            low = 0
            advertised_light = int(
                signed_policy.minimum_advertised_context_light_canaries
            )
            hard = 0
        else:
            low = int(signed_policy.minimum_low_context_canaries)
            advertised_light = 0
            hard = 0
    else:
        if hard_audit_enabled:
            low = 0
            advertised_light = 1
            hard = 0
        else:
            low = max(2, int(getattr(config, "canary_small_count", 0)))
            advertised_light = 0
            hard = 0
    return low, advertised_light, hard


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


def _tokenizer_chat_template_kwargs(
    tokenizer,
    *,
    enable_thinking: bool,
) -> dict[str, object]:
    """Match the miner's bounded chat-template capability selection."""

    import inspect

    try:
        source = inspect.getsource(tokenizer.apply_chat_template)
    except (AttributeError, TypeError, OSError):
        source = ""
    template = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in source or "enable_thinking" in template:
        return {"enable_thinking": enable_thinking}
    if "reasoning_effort" in template:
        return {
            "reasoning_effort": "medium" if enable_thinking else "none"
        }
    return {}


def _tokenize_proof_v3_chat(
    model_id: str,
    messages: Sequence[Mapping[str, object]],
    *,
    enable_thinking: bool,
) -> tuple[int, ...]:
    """Produce the exact token sequence the v3 miner must execute."""

    tokenizer = _get_tokenizer(model_id)
    values = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        **_tokenizer_chat_template_kwargs(
            tokenizer,
            enable_thinking=enable_thinking,
        ),
    )
    from verallm.proof_v3.request import canonical_tokenizer_token_ids_v3

    try:
        return canonical_tokenizer_token_ids_v3(values)
    except ValueError as exc:
        raise RuntimeError(
            "proof-v3 tokenizer returned malformed token ids"
        ) from exc


def _proof_v3_canary_runtime_policy(
    qualified_release,
    *,
    hard_audit: bool,
):
    """Build one validator-owned light or 100%-hard v3 canary policy."""

    from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3

    signed = (
        qualified_release.qualified_profile.profile.relation_spec.audit_policy
    )
    return RuntimeHardAuditPolicyV3(
        request_kind="canary" if hard_audit else "organic",
        effective_organic_hard_bps=signed.minimum_organic_hard_bps,
        effective_canary_hard_bps=10_000,
        effective_probation_failures=signed.probation_failures,
        nonce_selection_abi_id=signed.nonce_selection_abi_id,
        tier_selection_abi_id=signed.tier_selection_abi_id,
        selection_abi_id=signed.selection_abi_id,
    )


def _proof_v3_hard_auditor_active(
    config: object,
    validator_hotkey_ss58: str,
) -> bool:
    """Require verify mode and the exact subnet-configured validator."""

    configured_hotkey = str(
        getattr(config, "proof_v3_hard_auditor_hotkey_ss58", "") or ""
    )
    return bool(
        str(
            getattr(config, "proof_v3_verdict_source", "follower")
            or "follower"
        ).strip().lower()
        == "verify"
        and getattr(
            config,
            "proof_v3_hard_auditor_policy_enabled",
            False,
        )
        and configured_hotkey
        and configured_hotkey == str(validator_hotkey_ss58 or "")
    )


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
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **_tokenizer_chat_template_kwargs(
                    tokenizer,
                    enable_thinking=enable_thinking,
                ),
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

        self._proof_v3_failure_policy_cfg = (
            proof_v3_failure_policy_config_from_neuron_config(config)
        )
        self._hard_failure_strike_lock = threading.Lock()
        self._hard_failure_strikes = HardProofStrikeTracker(
            failure_epochs_for_penalty=(
                self._proof_v3_failure_policy_cfg
                .failure_epochs_for_penalty
            ),
            clean_hard_audit_epochs_for_reset=(
                self._proof_v3_failure_policy_cfg
                .clean_hard_audit_epochs_for_reset
            ),
        )
        self._load_hard_failure_strikes()
        self._probation_state_reset_effective_epoch: int | None = None
        raw_reset_epoch = self._db.get_meta(
            "proof_v3_probation_state_reset_effective_epoch_v1"
        )
        if raw_reset_epoch:
            self._probation_state_reset_effective_epoch = int(raw_reset_epoch)

        self.evm_pk = ""
        self.evm_addr = ""
        # Set True when validator runs without EVM registration (no on-chain
        # reportOffline / updateDemandScores). Triggered explicitly by
        # config.no_evm or implicitly when registerEvm fails (e.g. low TAO).
        self._evm_disabled: bool = bool(getattr(config, "no_evm", False))
        self._model_client = None
        self._miner_client = None
        self._chain_provider = None
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
        # EVM addresses excluded because UID ownership checks show another
        # address/hotkey now owns that numeric UID.
        self._stale_uid_addresses: Set[str] = set()
        self._miner_debug_cache_lock = threading.Lock()
        self._miner_debug_refresh_in_flight = False
        self._miner_debug_last_scheduled_at = 0.0
        from neurons.verdict_follower import VerdictSnapshotFollower

        self._verdict_snapshot_follower = VerdictSnapshotFollower()
        self._proof_v3_verdict_source_latched = "follower"
        self._proof_v3_verdict_source_latched_epoch: int | None = None
        self._owner_verdict_url_latched = ""
        self._verdict_snapshot_hex = ""
        self._verdict_snapshot_history: Dict[int, str] = {}
        try:
            from neurons.verdict_records import VerdictSnapshotV1

            persisted_snapshot = str(
                self._db.get_meta("gleipnir_verdict_snapshot_v1") or ""
            )
            if persisted_snapshot:
                decoded_snapshot = VerdictSnapshotV1.from_bytes(
                    bytes.fromhex(persisted_snapshot)
                )
                self._verdict_snapshot_hex = persisted_snapshot
                self._verdict_snapshot_history[
                    int(decoded_snapshot.epoch_number)
                ] = persisted_snapshot
            persisted_history = str(
                self._db.get_meta("gleipnir_verdict_snapshots_v1") or ""
            )
            if persisted_history:
                history = json.loads(persisted_history)
                if not isinstance(history, dict):
                    raise ValueError("verdict snapshot history must be an object")
                for raw_epoch, encoded in history.items():
                    epoch = int(raw_epoch)
                    if epoch < 0 or not isinstance(encoded, str):
                        raise ValueError("verdict snapshot history is malformed")
                    decoded = VerdictSnapshotV1.from_bytes(bytes.fromhex(encoded))
                    if int(decoded.epoch_number) != epoch:
                        raise ValueError("verdict snapshot history epoch mismatch")
                    self._verdict_snapshot_history[epoch] = encoded
                self._verdict_snapshot_history = dict(
                    sorted(self._verdict_snapshot_history.items())[-4:]
                )
        except Exception as exc:
            bt.logging.warning(
                "Ignoring malformed persisted verdict snapshot: "
                f"{type(exc).__name__}: {exc}"
            )

        # Epoch state
        self._current_epoch: int = 0
        self._epoch_start_block: int = 0
        self._canary_scheduler: Optional[CanaryScheduler] = None
        self._canary_scheduler_lock = threading.Lock()
        self._canary_accounting_lock = threading.Lock()
        self._epoch_miners: List[ActiveMiner] = []
        self._epoch_miners_discovery_valid: bool = False
        # Registered mainnet endpoints that fail the proxy's public HTTPS
        # admission policy are retained for diagnostics but never scheduled or
        # scored. This is eligibility, not probation: fixing the registration
        # restores the retained EMA without an audit-recovery cycle.
        self._endpoint_policy_gate_reasons: Dict[Tuple[str, int], str] = {}
        self._endpoint_policy_excluded_miners: Tuple[ActiveMiner, ...] = ()
        # {(lowercase miner_address, model_index): expected_receipt_count}
        self._expected_receipts: Dict[Tuple[str, int], int] = {}
        # Exact signed obligation IDs planned for each miner/model this epoch.
        self._expected_canary_obligations: Dict[
            Tuple[str, int],
            Dict[bytes, Tuple[str, int]],
        ] = {}
        self._hard_canary_obligation_ids: Set[bytes] = set()
        self._validator_canary_failures: Set[Tuple[str, int]] = set()
        self._canary_penalized_keys: Set[Tuple[str, int]] = set()
        # External validators independently replay the configured hard
        # auditor's retained proof. Missing owner receipts stay neutral;
        # present-but-invalid bundles become ordinary proof failures.
        self._shared_hard_proof_verdicts: Dict[
            Tuple[str, int],
            bool,
        ] = {}
        self._shared_hard_prefetch_lock = threading.Lock()
        self._shared_hard_prefetch_results: Dict[
            Tuple[int, str, int, bytes],
            Tuple[str, str],
        ] = {}
        self._shared_hard_prefetch_inflight: Set[
            Tuple[int, str, int, str]
        ] = set()
        self._shared_hard_prefetch_waves: Set[Tuple[int, int]] = set()
        self._shared_hard_processed_receipts: Set[bytes] = set()
        self._load_shared_hard_processed_receipts()
        self._shared_hard_processed_failures: Set[bytes] = set()
        self._load_shared_hard_processed_failures()
        # {epoch_number: {(miner_address, model_index): in_flight_count}}
        self._inflight_canaries: Dict[int, Dict[Tuple[str, int], int]] = {}
        # Submitted work is tracked before its executor worker starts so epoch
        # close can neutralize both queued and actively running validator work.
        self._queued_canaries: Dict[int, Dict[Tuple[str, int], int]] = {}
        # Exact submitted obligations remain here until their worker returns.
        # Epoch close removes only these specific unfinished obligations from
        # scoring; it must not neutralize unrelated completed canaries for the
        # same endpoint.
        self._unfinished_canary_tests: Dict[
            Tuple[int, bytes],
            CanaryTest,
        ] = {}
        # Sealing an epoch makes every late worker result stale immediately,
        # before _current_epoch advances.
        self._sealed_canary_epochs: Set[int] = set()
        # Any request actually started before its scoring boundary retains a
        # bounded terminal outcome after that boundary.  Queued/unstarted work
        # is validator-local and neutral; a miner cannot make an accepted
        # request disappear merely by delaying it into the next epoch.
        self._cross_epoch_canaries: Set[Tuple[int, bytes]] = set()
        self._closing_inflight_canaries: Dict[int, Dict[Tuple[str, int], int]] = {}
        self._proof_v3_full_pair_barriers: Dict[
            str,
            _ProofV3FullPairBarrier,
        ] = {}
        self._proof_v3_full_pair_barriers_lock = threading.Lock()
        # Only one post-commit hard audit may consume an endpoint's prover at
        # a time. Light traffic remains concurrent; this prevents one hard
        # proof from starving the next hard request's one-second precommit.
        self._proof_v3_hard_execution_locks: Dict[
            Tuple[str, int],
            threading.Lock,
        ] = {}
        self._proof_v3_hard_execution_locks_lock = threading.Lock()
        # {(lowercase miner_address, model_index): 503_skip_count} — reset each epoch
        self._busy_skips: Dict[Tuple[str, int], int] = {}
        # Signed-receipt reconciliation uses exact validator-observed intervals.
        self._busy_skip_probations: Dict[
            Tuple[str, int],
            List[Tuple[float, float, str, bytes]],
        ] = {}
        # {model_id: ModelSpec} — cached per epoch, avoids RPC per canary
        self._model_spec_cache: Dict[str, object] = {}
        # Exact Solidity ModelSpec structs are retained separately. Proof-v2
        # manifests must be matched against this lossless chain view before
        # they can be attached to a verifier client.
        self._on_chain_model_spec_cache: Dict[str, object] = {}
        # Values are VerifiedProofV2Manifest records authenticated during
        # setup against the current ModelRegistry owner/multisig authority.
        self._proof_v2_manifests: Dict[str, object] = {}
        self._proof_v2_remote_refresh_after: float = 0.0
        # Fully qualified, authority-authenticated proof-v3 releases. These
        # contain validator artifacts and signed profiles, never model weights.
        self._proof_v3_releases: Dict[str, object] = {}
        self._proof_v3_canary_policy = None
        self._proof_v3_local_release_model_ids: Set[str] = set()
        self._proof_v3_remote_refresh_after: float = 0.0
        # Digest of the last remote index whose complete releases and signed
        # policy were authenticated. The hourly refresh first compares this
        # bounded index so unchanged catalogs do not stall epoch setup.
        self._proof_v3_remote_index_sha256: bytes = b""
        # Remote miner_version tracking — opens a forgiveness window after a
        # release lands in the public repo, so miners restarting to pull the
        # new code don't get probation for "canary errors" that are really
        # vLLM-reload downtime.  Window is per-miner one-shot, lasts 2 epochs.
        self._miner_version_last_seen: int = 0
        self._miner_version_bump_at: float = 0.0
        self._miner_version_last_check: float = 0.0
        self._restart_forgiven: Set[Tuple[str, int]] = set()
        # One bounded full-context deferral, persisted in validator_meta.
        self._full_context_debt: Dict[Tuple[str, int], int] = {}
        self._load_full_context_debt()
        # Last independently verified hard-audit pass per endpoint. This is a
        # rolling security floor, not an exact per-epoch counter: random hard
        # draws continue after a pass, while a two-epoch drought forces one
        # hidden post-precommit hard mark.
        self._hard_audit_pass_lock = threading.Lock()
        self._last_hard_audit_pass_epoch: Dict[Tuple[str, int], int] = {}
        self._load_hard_audit_pass_epochs()
        # A late hard proof is security-relevant even after its throughput
        # epoch closes. Track the last source epoch reconciled into probation
        # so an out-of-order completion cannot create a duplicate/retroactive
        # clean pass after a newer failure.
        self._probation_recovery_epoch_lock = threading.Lock()
        self._last_probation_recovery_source_epoch: Dict[
            Tuple[str, int],
            int,
        ] = {}
        self._load_probation_recovery_source_epochs()
        # Epoch close state
        self._pending_epoch_close: Optional[int] = None
        # Finalized-block callbacks can overlap while receipt pulling and
        # scoring hold the first callback for tens of seconds.  The completed
        # epoch guard alone is not sufficient because two callbacks can both
        # observe the old value before either close finishes.  Keep the whole
        # close transaction single-flight so scoring, EMA updates, the audit
        # log and the weight submission are each produced exactly once.
        self._epoch_close_lock = threading.Lock()
        # Closing/scoring an epoch may wait for the designated owner's signed
        # verdict snapshot.  Keep that control-plane wait off the finalized-
        # block callback so the next epoch can be planned and its canaries can
        # be dispatched independently.  The worker receives a frozen copy of
        # every per-epoch input it consumes; it must never read state reset by
        # the next epoch's setup.
        self._epoch_close_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="epoch-close",
        )
        self._epoch_close_local = threading.local()
        self._epoch_close_futures: Dict[int, object] = {}
        self._auto_updater = None  # Set by main() if --auto-update
        self._epoch_close_block: int = 0
        self._epoch_close_retry_after: float = 0.0  # monotonic time
        self._epoch_close_backoff: float = 30.0  # seconds, doubles on failure
        self._weight_update_due: bool = False
        self._weight_update_lock = threading.Lock()
        self._weight_updates_pending: int = 0
        self._last_known_block: int = 0  # fallback for _get_current_block
        # Receipt chronology is bound to the head observed when the HTTP body
        # completes, not to a later worker-time RPC. The block stream is the
        # sole writer; receipt workers only take an atomic snapshot.
        self._capacity_audit_head_lock = threading.Lock()
        self._capacity_audit_head_block = 0
        self._capacity_audit_head_generation = 0
        self._capacity_audit_head_observed_at = 0.0
        self._capacity_audit_head_observed_monotonic = 0.0
        self._capacity_audit_head_source = ""
        self._capacity_audit_head_connected = False
        # Receipt chronology must keep advancing even when validator-side
        # native or metagraph work retains the Python GIL.  The isolated
        # observer is started before the ingest server and is the production
        # source of truth; the in-process fields above remain for narrow tests
        # and validators with capacity auditing disabled.
        self._capacity_audit_head_process = None
        self._capacity_audit_head_shared = None
        # The stream watchdog can replace a subscription while its callback is
        # still completing a slow epoch close.  Serialize block delivery across
        # subscription generations so a stale reconnect cursor cannot dispatch
        # the boundary twice or overtake the callback that already claimed it.
        self._block_dispatch_lock = threading.Lock()
        self._highest_dispatched_block: int = -1
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
        self._proof_protocol_rollout_cfg = (
            proof_protocol_rollout_config_from_neuron_config(config)
        )
        repaired_audits = self._db.reconcile_capacity_audit_duplicate_timing_misses()
        if repaired_audits["slot_rows"] or repaired_audits["history_rows"]:
            bt.logging.warning(
                "Capacity audit timing reconciliation repaired duplicate-delivery "
                f"classifications: active={repaired_audits['slot_rows']} "
                f"history={repaired_audits['history_rows']}"
            )
        self._maintenance_grace_cfg = maintenance_grace_config_from_neuron_config(config)
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
        # Weight transactions are ordered state transitions. Async epoch
        # closes may overlap, but their submissions must never race or land
        # out of epoch order.
        self._weight_update_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="weight-update",
        )
        # Slow/retrying reportOffline transactions are a best-effort chain
        # side effect. They must never consume the workers used for epoch
        # discovery, identity checks, or weight setting. Do not queue stale
        # reports behind a degraded RPC: a delayed vote could land after an
        # honest endpoint has already recovered.
        self._report_offline_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="report-offline",
        )
        self._report_offline_lock = threading.Lock()
        self._report_offline_inflight: Set[Tuple[str, int]] = set()
        self._report_offline_slots = threading.BoundedSemaphore(2)
        # Capacity-audit scheduling must not queue behind epoch identity checks:
        # short drain windows are intentionally only a few blocks long. Keep
        # scheduling serialized so adjacent hash-triggered windows observe each
        # other's freshly written drains before selecting endpoint slots.
        self._capacity_audit_executor = ThreadPoolExecutor(max_workers=1)
        self._capacity_audit_discovery_executor = ThreadPoolExecutor(max_workers=1)
        # Receipt deadlines are based on completed-body arrival. Keep signature
        # validation and SQLite work off the ingest event loop so one receipt
        # cannot delay the timestamp assigned to another.
        self._capacity_audit_receipt_executor = ThreadPoolExecutor(max_workers=4)
        self._capacity_audit_receipt_metrics_lock = threading.Lock()
        self._capacity_audit_receipt_ticket = 0
        self._capacity_audit_receipt_pending: dict[int, tuple[float, str]] = {}
        self._capacity_audit_receipt_metrics: dict[str, object] = {
            "submitted": 0,
            "completed": 0,
            "head_unavailable": 0,
            "max_queue_depth": 0,
            "max_queue_delay_s": 0.0,
            "max_processing_s": 0.0,
            "per_audit": {},
        }
        # Shared debug state is relatively expensive to serialize. Receipt
        # ingress never schedules a rebuild: its only synchronous duties are
        # validating and durably recording the already-captured arrival
        # timestamp/head. Terminal proof/window transitions use this trailing
        # edge coalescer so a burst produces one rebuild after it goes quiet.
        self._capacity_audit_state_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="capacity-state",
        )
        self._capacity_audit_state_lock = threading.Lock()
        self._capacity_audit_state_pending = False
        self._capacity_audit_state_dirty = False
        self._capacity_audit_state_requested_at = 0.0
        self._miner_debug_executor = ThreadPoolExecutor(max_workers=1)
        # Metagraph statistics are informational. A degraded RPC must never
        # stall current-head processing, audit chronology, or canary dispatch.
        self._metagraph_stats_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="metagraph-stats",
        )
        self._metagraph_stats_lock = threading.Lock()
        self._metagraph_stats_in_flight = False
        # Analytics retention is best-effort maintenance, not an epoch-close
        # dependency.  A daemon worker keeps multi-million-row archives off the
        # block subscription thread; the DB layer streams them with bounded
        # memory and short write locking.
        self._analytics_archive_lock = threading.Lock()
        self._analytics_archive_thread: threading.Thread | None = None
        proof_workers = max(
            1,
            int(getattr(config, "capacity_audit_proof_verify_workers", 4) or 4),
        )
        self._capacity_audit_proof_executor = ThreadPoolExecutor(max_workers=proof_workers)
        self._shared_hard_prefetch_executor = ThreadPoolExecutor(
            max_workers=min(8, proof_workers),
        )

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
        self._enforce_proof_v3_verdict_owner_guard()
        self._scoring = runtime.scoring
        self._last_good_scoring = runtime.scoring
        self._capacity_audit_cfg = runtime.capacity_audit
        self._proof_protocol_rollout_cfg = runtime.proof_protocol_rollout
        self._proof_v3_failure_policy_cfg = runtime.proof_v3_failure_policy
        self._maintenance_grace_cfg = runtime.maintenance_grace
        self._apply_probation_state_generation(
            runtime.proof_v3_failure_policy,
            effective_epoch=(
                runtime.effective_epoch
                if runtime.effective_epoch is not None
                else current_epoch
            ),
        )
        with self._hard_failure_strike_lock:
            self._hard_failure_strikes.configure(
                failure_epochs_for_penalty=(
                    runtime.proof_v3_failure_policy
                    .failure_epochs_for_penalty
                ),
                clean_hard_audit_epochs_for_reset=(
                    runtime.proof_v3_failure_policy
                    .clean_hard_audit_epochs_for_reset
                ),
            )
            self._save_hard_failure_strikes_locked()
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

    def _configured_proof_v3_verdict_source(self) -> str:
        source = str(
            getattr(self.config, "proof_v3_verdict_source", "follower")
            or "follower"
        ).strip().lower()
        if source not in {"verify", "follower"}:
            bt.logging.warning(
                f"Invalid proof-v3 verdict source {source!r}; using follower"
            )
            return "follower"
        return source

    def _enforce_proof_v3_verdict_owner_guard(self) -> None:
        """Require the configured owner to opt into local verification."""

        owner = str(
            getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        local = str(getattr(self, "_validator_hotkey_ss58", "") or "")
        if (
            owner
            and local == owner
            and self._configured_proof_v3_verdict_source() == "follower"
        ):
            raise RuntimeError(
                "configured proof-v3 hard auditor must explicitly set "
                "VERATHOS_PROOF_V3_VERDICT_SOURCE=verify (or "
                "--proof-v3-verdict-source=verify)"
            )

    def _latch_proof_v3_verdict_source(self, epoch_number: int) -> None:
        self._enforce_proof_v3_verdict_owner_guard()
        self._proof_v3_verdict_source_latched = (
            self._configured_proof_v3_verdict_source()
        )
        self._owner_verdict_url_latched = str(
            getattr(self.config, "owner_verdict_url", "") or ""
        ).strip()
        self._proof_v3_verdict_source_latched_epoch = int(epoch_number)
        bt.logging.info(
            "Proof-v3 verdict source latched for epoch "
            f"{int(epoch_number)}: "
            f"{self._proof_v3_verdict_source_latched}"
        )

    def _proof_v3_follower_mode_active(self) -> bool:
        return bool(
            self._epoch_close_value(
                "_proof_v3_verdict_source_latched",
                "verify",
            )
            == "follower"
        )

    def _maintenance_grace_active(
        self,
        *,
        current_epoch: int | None = None,
        action: str | None = None,
    ) -> bool:
        cfg = self._epoch_close_value("_maintenance_grace_cfg", None)
        if cfg is None:
            base_config = getattr(self, "config", None)
            cfg = (
                maintenance_grace_config_from_neuron_config(base_config)
                if base_config is not None
                else MaintenanceGraceConfig()
            )
        epoch = current_epoch
        if epoch is None:
            epoch = self._epoch_close_value("_current_epoch", None)
        if not maintenance_grace_active(cfg, current_epoch=epoch):
            return False
        if action is None:
            return True
        return bool(getattr(cfg, action, False))

    def _maintenance_grace_reason(self) -> str:
        cfg = self._epoch_close_value("_maintenance_grace_cfg", None)
        if cfg is None:
            base_config = getattr(self, "config", None)
            cfg = (
                maintenance_grace_config_from_neuron_config(base_config)
                if base_config is not None
                else MaintenanceGraceConfig()
            )
        return cfg.reason or "maintenance grace"

    def _legacy_v1_compatibility_active(
        self,
        *,
        current_epoch: int | None = None,
    ) -> bool:
        cfg = self._epoch_close_value("_proof_protocol_rollout_cfg", None)
        if cfg is None:
            base_config = getattr(self, "config", None)
            cfg = (
                proof_protocol_rollout_config_from_neuron_config(base_config)
                if base_config is not None
                else ProofProtocolRolloutConfig()
            )
        epoch = current_epoch
        if epoch is None:
            epoch = self._epoch_close_value("_current_epoch", None)
        return legacy_v1_compatibility_active(cfg, current_epoch=epoch)

    def _proof_v3_required(self) -> bool:
        """Return the proof-version requirement independently of maintenance."""

        cfg = self._epoch_close_value("_proof_protocol_rollout_cfg", None)
        if cfg is None:
            base_config = getattr(self, "config", None)
            cfg = (
                proof_protocol_rollout_config_from_neuron_config(base_config)
                if base_config is not None
                else ProofProtocolRolloutConfig()
            )
        return proof_v3_required(cfg)

    def _proof_v3_allowed(self) -> bool:
        """Return whether the epoch-pinned rollout permits proof v3."""

        cfg = self._epoch_close_value("_proof_protocol_rollout_cfg", None)
        if cfg is None:
            base_config = getattr(self, "config", None)
            cfg = (
                proof_protocol_rollout_config_from_neuron_config(base_config)
                if base_config is not None
                else ProofProtocolRolloutConfig()
            )
        return proof_protocol_allowed(cfg, 3)

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

    def _load_proof_v2_manifests(self) -> None:
        """Authenticate configured proof-v2 manifests against current chain state."""

        paths = tuple(
            getattr(self.config, "proof_v2_manifest_paths", ()) or ()
        )
        remote_base_urls = tuple(
            getattr(self.config, "proof_v2_artifact_base_urls", ()) or ()
        )
        self._proof_v2_manifests = {}
        if not paths and not remote_base_urls:
            bt.logging.warning(
                "No proof-v2 manifests configured; proof-v2 verification will "
                "fail closed"
            )
            return

        from verallm.proof_v2.runtime import load_verified_proof_v2_manifests

        verified: Dict[str, object] = {}
        if paths:
            try:
                verified.update(
                    load_verified_proof_v2_manifests(
                        paths,
                        chain_config=self.config,
                        model_registry_client=self._model_client,
                    )
                )
            except Exception as exc:
                self._proof_v2_manifests = {}
                bt.logging.error(f"Proof-v2 manifest authentication failed: {exc}")
                raise

        if remote_base_urls:
            from verallm.proof_v2.artifact_store import (
                resolve_remote_proof_v2_manifests,
            )

            try:
                remote = resolve_remote_proof_v2_manifests(
                    remote_base_urls,
                    chain_config=self.config,
                    model_registry_client=self._model_client,
                    cache_directory=getattr(
                        self.config,
                        "proof_v2_artifact_cache_dir",
                        None,
                    ),
                )
            except Exception as exc:
                bt.logging.error(
                    f"Remote proof-v2 artifact resolution failed: {exc}"
                )
            else:
                for model_id, item in remote.manifests.items():
                    existing = verified.get(model_id)
                    if existing is not None:
                        if (
                            existing.manifest.digest()
                            != item.manifest.digest()
                        ):
                            bt.logging.warning(
                                "Ignoring remote proof-v2 manifest for "
                                f"{model_id}; a different authenticated local "
                                "override is configured"
                            )
                        continue
                    verified[model_id] = item
                if remote.missing_model_ids:
                    bt.logging.warning(
                        "No published proof-v2 manifest for registered model(s): "
                        + ", ".join(remote.missing_model_ids)
                    )
                for model_id, failure in remote.failures.items():
                    bt.logging.error(
                        f"Proof-v2 artifact failed for {model_id}: {failure}"
                    )
                bt.logging.info(
                    "Loaded remote proof-v2 artifact index from "
                    f"{remote.index_source_url}"
                )

        try:
            exact_specs: Dict[str, object] = {}
            for model_id in verified:
                exact_spec = self._model_client.get_on_chain_model_spec(model_id)
                if exact_spec is None:
                    raise RuntimeError(
                        f"proof-v2 manifest model is not registered: {model_id}"
                    )
                exact_specs[model_id] = exact_spec
        except Exception as exc:
            self._proof_v2_manifests = {}
            bt.logging.error(f"Proof-v2 manifest authentication failed: {exc}")
            raise

        if not verified:
            bt.logging.warning(
                "No authenticated proof-v2 manifests are available; proof-v2 "
                "verification will fail closed"
            )
            return
        self._proof_v2_manifests = verified
        self._on_chain_model_spec_cache.update(exact_specs)
        bt.logging.info(
            f"Authenticated {len(verified)} proof-v2 manifest(s)"
        )
        if remote_base_urls:
            self._proof_v2_remote_refresh_after = (
                time.monotonic() + _PROOF_V2_ARTIFACT_REFRESH_SECONDS
            )

    def _load_proof_v3_releases(self) -> None:
        """Authenticate configured weightless v3 releases against ModelRegistry."""

        paths = tuple(
            getattr(self.config, "proof_v3_release_paths", ()) or ()
        )
        remote_base_urls = tuple(
            getattr(self.config, "proof_v3_artifact_base_urls", ()) or ()
        )
        self._proof_v3_releases = {}
        self._proof_v3_canary_policy = None
        if not paths and not remote_base_urls:
            bt.logging.warning(
                "No proof-v3 releases configured; v3 canaries are unavailable"
            )
            return
        from verallm.proof_v3.economic_release_catalog import (
            load_qualified_proof_v3_catalog,
        )

        releases = {}
        indexed_policy_path = ""
        if paths:
            try:
                releases.update(
                    load_qualified_proof_v3_catalog(
                        paths,
                        model_registry_client=self._model_client,
                    )
                )
            except Exception as exc:
                bt.logging.error(
                    f"Proof-v3 release authentication failed: {exc}"
                )
                raise
        self._proof_v3_local_release_model_ids = set(releases)
        if remote_base_urls:
            from verallm.proof_v3.artifact_store import (
                resolve_remote_proof_v3_releases,
            )

            remote = resolve_remote_proof_v3_releases(
                remote_base_urls,
                chain_config=self.config,
                model_registry_client=self._model_client,
                cache_directory=getattr(
                    self.config,
                    "proof_v3_artifact_cache_dir",
                    None,
                ),
            )
            for model_id, item in remote.releases.items():
                existing = releases.get(model_id)
                if existing is not None:
                    if (
                        existing.qualified_profile.profile.digest()
                        != item.release.qualified_profile.profile.digest()
                    ):
                        bt.logging.warning(
                            "Ignoring remote proof-v3 release for "
                            f"{model_id}; a different authenticated local "
                            "override is configured"
                        )
                    continue
                releases[model_id] = item.release
            if remote.missing_model_ids:
                bt.logging.warning(
                    "No published proof-v3 release for registered model(s): "
                    + ", ".join(remote.missing_model_ids)
                )
            for model_id, failure in remote.failures.items():
                bt.logging.error(
                    f"Proof-v3 artifact failed for {model_id}: {failure}"
                )
            bt.logging.info(
                "Loaded remote proof-v3 artifact index from "
                f"{remote.index_source_url}"
            )
            if not remote.failures:
                self._proof_v3_remote_index_sha256 = bytes(
                    getattr(remote, "index_sha256", b"") or b""
                )
            indexed_policy_path = str(
                getattr(remote, "canary_policy_path", "") or ""
            ).strip()
        self._proof_v3_releases = releases
        bt.logging.info(
            f"Authenticated {len(releases)} proof-v3 release(s)"
        )
        if self._proof_v3_required() and not releases:
            raise RuntimeError(
                "proof-v3 required mode needs at least one authenticated "
                "model release"
            )
        policy_path = str(
            getattr(self.config, "proof_v3_canary_policy_path", "") or ""
        ).strip()
        if not policy_path:
            policy_path = indexed_policy_path
        self._proof_v3_canary_policy = None
        if not policy_path:
            if self._proof_v3_required():
                raise RuntimeError(
                    "proof-v3 required mode needs an authority-signed canary policy"
                )
            bt.logging.warning(
                "No signed proof-v3 canary policy configured; validators will "
                "use bounded v1 compatibility instead of v3 hard canaries"
            )
            return
        from verallm.proof_v3.canary_policy import (
            load_signed_canary_policy_document_v3,
            qualify_canary_policy_v3,
        )

        authority = self._model_client.get_manifest_authority()
        document = load_signed_canary_policy_document_v3(policy_path)
        self._proof_v3_canary_policy = qualify_canary_policy_v3(
            document,
            qualified_releases=releases,
            expected_authority_signers=tuple(authority.signers),
            authority_threshold=int(authority.threshold),
        )
        bt.logging.info(
            "Authenticated proof-v3 canary policy "
            f"{self._proof_v3_canary_policy.policy_abi_id}"
        )
        if remote_base_urls:
            self._proof_v3_remote_refresh_after = (
                time.monotonic() + _PROOF_V3_ARTIFACT_REFRESH_SECONDS
            )

    def _refresh_remote_proof_v3_releases(
        self,
        model_ids: Set[str] | None = None,
    ) -> None:
        """Atomically adopt authenticated replacement v3 releases.

        The content-addressed objects are immutable, but the signed remote
        index may replace the release selected for an existing model.  Refresh
        the complete requested inventory and its signed canary policy as one
        unit; a partial download or stale policy leaves the last authenticated
        snapshot active.
        """

        remote_base_urls = tuple(
            getattr(self.config, "proof_v3_artifact_base_urls", ()) or ()
        )
        if not remote_base_urls:
            return
        now = time.monotonic()
        if now < getattr(self, "_proof_v3_remote_refresh_after", 0.0):
            return
        self._proof_v3_remote_refresh_after = (
            now + _PROOF_V3_ARTIFACT_REFRESH_SECONDS
        )
        requested = (
            set(model_ids)
            if model_ids is not None
            else set(self._model_client.get_model_list())
        )
        # The signed policy binds the complete qualified release inventory.
        # Refresh every already-admitted remote release as well as the models
        # active this epoch, otherwise an inactive model replacement could
        # leave the newly downloaded policy intentionally stale.
        requested.update(getattr(self, "_proof_v3_releases", {}))
        if not requested:
            return

        from verallm.proof_v3.artifact_store import (
            probe_remote_proof_v3_index,
            resolve_remote_proof_v3_releases,
        )

        try:
            probe = probe_remote_proof_v3_index(
                remote_base_urls,
                chain_config=self.config,
                cache_directory=getattr(
                    self.config,
                    "proof_v3_artifact_cache_dir",
                    None,
                ),
                timeout_seconds=5.0,
            )
            previous_index = bytes(
                getattr(self, "_proof_v3_remote_index_sha256", b"") or b""
            )
            if previous_index and probe.index_sha256 == previous_index:
                bt.logging.debug(
                    "Proof-v3 remote index is unchanged; retaining the "
                    "authenticated in-memory release snapshot"
                )
                return
            remote = resolve_remote_proof_v3_releases(
                remote_base_urls,
                chain_config=self.config,
                model_registry_client=self._model_client,
                cache_directory=getattr(
                    self.config,
                    "proof_v3_artifact_cache_dir",
                    None,
                ),
                model_ids=requested,
            )
        except Exception as exc:
            bt.logging.warning(
                "Proof-v3 artifact refresh failed; retaining authenticated "
                f"snapshot: {exc}"
            )
            return

        updated = dict(getattr(self, "_proof_v3_releases", {}))
        local_overrides = set(
            getattr(self, "_proof_v3_local_release_model_ids", set())
        )
        changed = 0
        for model_id, item in remote.releases.items():
            if model_id in local_overrides:
                continue
            previous = updated.get(model_id)
            replacement = item.release
            if (
                previous is None
                or previous.qualified_profile.profile.digest()
                != replacement.qualified_profile.profile.digest()
            ):
                changed += 1
            updated[model_id] = replacement

        policy_path = str(
            getattr(self.config, "proof_v3_canary_policy_path", "") or ""
        ).strip()
        if not policy_path:
            policy_path = str(
                getattr(remote, "canary_policy_path", "") or ""
            ).strip()
        try:
            if not policy_path:
                raise RuntimeError(
                    "refreshed proof-v3 index has no signed canary policy"
                )
            from verallm.proof_v3.canary_policy import (
                load_signed_canary_policy_document_v3,
                qualify_canary_policy_v3,
            )

            authority = self._model_client.get_manifest_authority()
            document = load_signed_canary_policy_document_v3(policy_path)
            policy = qualify_canary_policy_v3(
                document,
                qualified_releases=updated,
                expected_authority_signers=tuple(authority.signers),
                authority_threshold=int(authority.threshold),
            )
        except Exception as exc:
            bt.logging.warning(
                "Proof-v3 policy refresh failed; retaining authenticated "
                f"snapshot: {exc}"
            )
            return

        # Epoch setup calls this before dispatching the new plan. Replacing
        # both references here ensures that no newly scheduled canary can mix
        # an old policy with a new release (or the inverse).
        self._proof_v3_releases = updated
        self._proof_v3_canary_policy = policy
        if not remote.failures:
            self._proof_v3_remote_index_sha256 = bytes(
                getattr(remote, "index_sha256", b"") or b""
            )
        if changed:
            bt.logging.info(
                f"Refreshed {changed} proof-v3 release(s) and signed policy "
                f"from {remote.index_source_url}"
            )
        for model_id, failure in remote.failures.items():
            bt.logging.warning(
                f"Proof-v3 artifact refresh failed for {model_id}: {failure}"
            )

    def _refresh_remote_proof_v2_manifests(
        self,
        model_ids: Set[str] | None = None,
    ) -> None:
        """Periodically add newly published authenticated model manifests."""

        remote_base_urls = tuple(
            getattr(self.config, "proof_v2_artifact_base_urls", ()) or ()
        )
        if not remote_base_urls:
            return
        now = time.monotonic()
        if now < getattr(self, "_proof_v2_remote_refresh_after", 0.0):
            return
        self._proof_v2_remote_refresh_after = (
            now + _PROOF_V2_ARTIFACT_REFRESH_SECONDS
        )
        requested = (
            set(model_ids)
            if model_ids is not None
            else set(self._model_client.get_model_list())
        )
        missing = requested.difference(
            getattr(self, "_proof_v2_manifests", {})
        )
        if not missing:
            return

        from verallm.proof_v2.artifact_store import (
            resolve_remote_proof_v2_manifests,
        )

        try:
            remote = resolve_remote_proof_v2_manifests(
                remote_base_urls,
                chain_config=self.config,
                model_registry_client=self._model_client,
                cache_directory=getattr(
                    self.config,
                    "proof_v2_artifact_cache_dir",
                    None,
                ),
                model_ids=missing,
            )
        except Exception as exc:
            bt.logging.warning(
                f"Proof-v2 artifact refresh failed; retaining verified cache: {exc}"
            )
            return

        updated = dict(getattr(self, "_proof_v2_manifests", {}))
        added = 0
        for model_id, item in remote.manifests.items():
            previous = updated.get(model_id)
            if (
                previous is None
                or previous.manifest.digest() != item.manifest.digest()
            ):
                added += 1
            updated[model_id] = item
            exact_spec = self._model_client.get_on_chain_model_spec(model_id)
            if exact_spec is not None:
                self._on_chain_model_spec_cache[model_id] = exact_spec
        self._proof_v2_manifests = updated
        if added:
            bt.logging.info(
                f"Refreshed {added} proof-v2 manifest(s) from "
                f"{remote.index_source_url}"
            )
        for model_id, failure in remote.failures.items():
            bt.logging.warning(
                f"Proof-v2 artifact refresh failed for {model_id}: {failure}"
            )

    def _attach_verified_proof_v2_manifest(
        self,
        client: ValidatorClient,
        model_id: str,
    ) -> bool:
        """Attach one startup-authenticated manifest after exact spec binding."""

        exact_spec = getattr(self, "_on_chain_model_spec_cache", {}).get(model_id)
        if exact_spec is None:
            # Leaving the client unconfigured makes a v2 payload fail in the
            # canonical verifier while legacy v1 remains independently usable.
            client.proof_v2_manifest = None
            return False

        client._on_chain_model_spec = exact_spec
        verified = getattr(self, "_proof_v2_manifests", {}).get(model_id)
        if verified is None:
            client.proof_v2_manifest = None
            return False
        try:
            client.set_verified_proof_v2_manifest(verified.manifest)
        except Exception as exc:
            # A registry update can invalidate a manifest after startup. Keep
            # v2 fail-closed instead of converting this into a skipped check.
            client.proof_v2_manifest = None
            getattr(self, "_proof_v2_manifests", {}).pop(model_id, None)
            self._proof_v2_remote_refresh_after = 0.0
            bt.logging.warning(
                f"Proof-v2 manifest no longer matches {model_id}: {exc}"
            )
            return False
        return True

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
        self._chain_provider = getattr(self._model_client, "_provider", None)
        self._load_proof_v2_manifests()
        runtime_config_loaded = self._refresh_subnet_runtime_config(force=True)
        if select_proof_protocol_version(
            self._proof_protocol_rollout_cfg,
            peer_advertised=(1, 3),
        ) is None:
            raise RuntimeError(
                "subnet allows no inference proof protocol supported by this "
                "validator binary"
            )
        self._load_proof_v3_releases()

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
                self._subnet_config_client = SubnetConfigClient(
                    _sn_chain_config,
                    provider=self._chain_provider,
                )
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
        from bittensor_wallet import Keypair as _Keypair
        _kp = _Keypair.create_from_seed(hotkey_seed[:32].hex())
        self._validator_hotkey_bytes = _kp.public_key
        self._validator_private_key = hotkey_seed

        # SS58 hotkey address for Sr25519 request signing (miner auth)
        self._validator_hotkey_ss58 = wallet.hotkey.ss58_address
        self._enforce_proof_v3_verdict_owner_guard()

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
            vr = ValidatorRegistryClient(
                self.config,
                provider=self._chain_provider,
            )
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
    ) -> int:
        """Fill missing transient /health metadata from matching cached rows."""
        missing = [
            miner for miner in miners
            if not (getattr(miner, "gpu_name", "") or "")
            or int(getattr(miner, "gpu_count", 0) or 0) <= 0
            or int(getattr(miner, "vram_gb", 0) or 0) <= 0
        ]
        if not missing:
            return 0
        try:
            cache_getter = getattr(self._db, "get_capacity_hardware_cache_entries", None)
            rows = cache_getter() if callable(cache_getter) else self._db.get_active_entries()
        except Exception:
            return 0
        by_key = {
            (str(row.get("address", "")).lower(), int(row.get("model_index", 0) or 0)): row
            for row in rows
        }
        hydrated = 0
        for miner in missing:
            key = (str(getattr(miner, "address", "") or "").lower(), int(getattr(miner, "model_index", 0) or 0))
            row = by_key.get(key)
            if not row:
                continue
            if str(row.get("endpoint") or "") != str(getattr(miner, "endpoint", "") or ""):
                continue
            if str(row.get("model_id") or "") != str(getattr(miner, "model_id", "") or ""):
                continue
            if str(row.get("quant") or "") != str(getattr(miner, "quant", "") or ""):
                continue
            if int(row.get("max_context_len") or 0) != int(
                getattr(miner, "max_context_len", 0) or 0
            ):
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
            hydrated += 1
        return hydrated

    @staticmethod
    def _has_capacity_hardware(miner: ActiveMiner) -> bool:
        return bool(
            str(getattr(miner, "gpu_name", "") or "")
            and int(getattr(miner, "gpu_count", 0) or 0) > 0
            and int(getattr(miner, "vram_gb", 0) or 0) > 0
        )

    @staticmethod
    def _hardware_fanout_deadline_s(
        endpoint_count: int,
        worker_count: int,
        *,
        per_request_timeout_s: float = 5.0,
        grace_s: float = 5.0,
        minimum_s: float = 10.0,
        maximum_s: float = 120.0,
    ) -> float:
        workers = max(1, int(worker_count))
        waves = max(1, (max(0, int(endpoint_count)) + workers - 1) // workers)
        deadline = waves * max(0.001, float(per_request_timeout_s)) + max(0.0, float(grace_s))
        return min(float(maximum_s), max(float(minimum_s), deadline))

    def _refresh_miner_hardware_batch(
        self,
        miners: list[ActiveMiner],
        *,
        source: str,
        max_workers: int = 32,
        per_request_timeout_s: float = 5.0,
        deadline_min_s: float = 10.0,
        deadline_max_s: float = 120.0,
    ) -> tuple[list[ActiveMiner], dict[str, int | float]]:
        miners = list(miners or [])
        if not miners:
            return [], {
                "active": 0,
                "fetched": 0,
                "invalid": 0,
                "timed_out": 0,
                "cancelled": 0,
                "hydrated": 0,
                "missing": 0,
                "deadline_s": 0.0,
            }

        hydrated = self._hydrate_capacity_audit_hardware_from_cache(miners)
        ordered = sorted(miners, key=self._has_capacity_hardware)
        workers = min(max(1, int(max_workers)), len(ordered))
        deadline_s = self._hardware_fanout_deadline_s(
            len(ordered),
            workers,
            per_request_timeout_s=per_request_timeout_s,
            minimum_s=deadline_min_s,
            maximum_s=deadline_max_s,
        )
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {
            pool.submit(self._fetch_miner_hardware_status, miner, per_request_timeout_s): miner
            for miner in ordered
        }
        statuses: dict[int, str] = {}
        timed_out = 0
        cancelled = 0
        try:
            try:
                for future in as_completed(futures, timeout=deadline_s):
                    miner = futures[future]
                    try:
                        statuses[id(miner)] = str(future.result())
                    except Exception:
                        statuses[id(miner)] = "unavailable"
            except _FuturesTimeout:
                timed_out = sum(1 for future in futures if not future.done())
        finally:
            for future in futures:
                if not future.done() and future.cancel():
                    cancelled += 1
            pool.shutdown(wait=False, cancel_futures=True)

        invalid = [miner for miner in miners if statuses.get(id(miner)) == "invalid"]
        missing = sum(1 for miner in miners if not self._has_capacity_hardware(miner))
        fetched = sum(1 for status in statuses.values() if status == "fetched")
        stats: dict[str, int | float] = {
            "active": len(miners),
            "fetched": fetched,
            "invalid": len(invalid),
            "timed_out": timed_out,
            "cancelled": cancelled,
            "hydrated": hydrated,
            "missing": missing,
            "deadline_s": deadline_s,
        }
        log = bt.logging.warning if timed_out else bt.logging.info
        log(
            f"Capacity audit hardware refresh: source={source} active={len(miners)} "
            f"fetched={fetched} hydrated={hydrated} invalid={len(invalid)} "
            f"timed_out={timed_out} cancelled={cancelled} still_missing={missing} "
            f"deadline_s={deadline_s:.1f}"
        )
        return invalid, stats

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
                self._refresh_capacity_audit_slot_snapshot_from_miners(
                    miners,
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
        resolved_count = 0
        missing_hardware = 0
        unsupported_hardware = 0
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
                missing_hardware += 1
                continue
            if str(row.get("endpoint") or "") != str(slot.endpoint or ""):
                missing_hardware += 1
                continue
            if str(row.get("model_id") or "") != str(slot.model_id or ""):
                missing_hardware += 1
                continue
            if str(row.get("quant") or "") != str(slot.quant or ""):
                missing_hardware += 1
                continue
            if int(row.get("max_context_len") or 0) != int(slot.max_context_len or 0):
                missing_hardware += 1
                continue
            gpu_name = str(row.get("gpu_name") or "")
            gpu_count = int(row.get("gpu_count") or 0)
            vram_gb = int(row.get("vram_gb") or 0)
            gpu_row = match_gpu_class(gpu_name, vram_gb, self._capacity_audit_cfg)
            if gpu_row is None or not gpu_row.calibrated or capacity_gpu_pass_count(gpu_row) <= 0:
                resolved = self._capacity_audit_resolve_selected_slot_hardware(slot)
                if resolved is not None:
                    supported[sid] = resolved
                    resolved_count += 1
                elif not gpu_name or gpu_count <= 0 or vram_gb <= 0:
                    missing_hardware += 1
                else:
                    unsupported_hardware += 1
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
        bt.logging.info(
            f"Capacity audit selected-slot eligibility: selected={len(selected_slots)} "
            f"eligible={len(supported)} resolved_live={resolved_count} "
            f"missing_hardware={missing_hardware} "
            f"unsupported_or_uncalibrated={unsupported_hardware}"
        )
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
        observe_head = getattr(self, "_observe_live_capacity_audit_head", None)
        if callable(observe_head):
            live_head, trustworthy = observe_head()
            if not trustworthy:
                # Never create a timing-sensitive obligation while the
                # independent chronology source is unavailable.  In
                # particular, a reconnect must not treat a replayed block as
                # the live head and schedule an already-expired audit.
                return False
            current_block = int(live_head)
        else:
            # Compatibility for narrow fixtures without a chronology source.
            current_block = int(getattr(self, "_last_known_block", 0) or 0)
        return ValidatorNeuron._capacity_audit_start_recoverable(
            self,
            audit_block,
            current_block,
        )

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
        canary_busy_fn = getattr(self, "_owner_canary_busy_keys", None)
        canary_busy_slots = (
            canary_busy_fn() if callable(canary_busy_fn) else set()
        )
        if canary_busy_slots:
            before = len(active)
            active = [
                (slot, row)
                for slot, row in active
                if (slot.address_lower, int(slot.model_index))
                not in canary_busy_slots
            ]
            if before != len(active):
                bt.logging.info(
                    f"Capacity audit: skipped {before - len(active)} "
                    f"owner-canary-busy slot(s) at block {selection_block}"
                )
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
            payload_cooldown_blocks = capacity_audit_payload_cooldown_blocks(cfg)
            busy_slots = set(
                self._db.get_capacity_audit_selection_busy_slots(
                    selection_block=int(selection_block),
                    cooldown_blocks=payload_cooldown_blocks,
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
        # Match the miner's complete B_select-to-evidence hold. This timestamp
        # is exported to routing/canary consumers; shortening it to only the
        # post-start deadlines can admit work while the miner is still inside
        # the same audit.
        drain_until_ts = now + capacity_audit_active_hold_seconds(cfg)
        rows: list[dict] = []
        unsupported_selected = 0
        hotkey_lookup = getattr(self, "_get_miner_ss58", None)
        for selected_slot in selected:
            sid = slot_id(selected_slot)
            supported = supported_selected.get(sid)
            if supported is None:
                unsupported_selected += 1
                continue
            slot, gpu_row = supported
            miner_hotkey = (
                hotkey_lookup(slot.address_lower, "hotkey")
                if callable(hotkey_lookup) else ""
            )
            if not miner_hotkey and slot.miner_uid is not None:
                owner_lookup = getattr(self._db, "get_uid_owner", None)
                if callable(owner_lookup):
                    owner = owner_lookup(int(slot.miner_uid)) or {}
                    if str(owner.get("evm_address") or "").lower() == slot.address_lower:
                        miner_hotkey = str(owner.get("hotkey_ss58") or "")
            rows.append({
                "miner_address": slot.address_lower,
                "model_index": slot.model_index,
                "miner_uid": slot.miner_uid,
                "miner_hotkey_ss58": miner_hotkey,
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
                "gpu_index": 0,
                "pass_count": capacity_gpu_pass_count(gpu_row),
                "workload_spec": (
                    capacity_gpu_workload_spec(gpu_row)
                    if hasattr(gpu_row, "workload_version")
                    else {}
                ),
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
            incident_review_required=True,
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
            strict_timing_mode = str(getattr(self._capacity_audit_cfg, "mode", "observe") or "observe") in {
                "score_gate",
                "enforce",
            }
            observe_head = getattr(
                self,
                "_observe_live_capacity_audit_head",
                None,
            )
            if callable(observe_head):
                live_head, trustworthy = observe_head()
            else:
                live_head, trustworthy = 0, False
            if strict_timing_mode and callable(observe_head) and not trustworthy:
                stale = self._db.mark_capacity_audit_window_stale(
                    window["audit_id"],
                    reason="validator_live_head_unavailable",
                    released_at=observed_at,
                )
                if stale:
                    self._write_shared_state()
                bt.logging.warning(
                    f"Capacity audit stale window skipped: audit_id={window['audit_id'][:12]} "
                    f"B_start={window_audit_block} live_head=unavailable "
                    f"released_slots={stale}"
                )
                continue
            current_head = max(
                int(audit_block),
                int(getattr(self, "_last_known_block", 0) or 0),
                int(live_head) if trustworthy else 0,
            )
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
        quarantined = 0
        strict_timing_mode = str(
            getattr(self._capacity_audit_cfg, "mode", "observe") or "observe"
        ) in {"score_gate", "enforce"}
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
            if strict_timing_mode:
                observe_head = getattr(
                    self,
                    "_observe_live_capacity_audit_head",
                    None,
                )
                if callable(observe_head):
                    live_head, trustworthy = observe_head()
                    late_by = max(0, int(live_head) - challenge_block)
                    if not trustworthy or late_by > 1:
                        reason = (
                            "validator_proof_challenge_head_unavailable"
                            if not trustworthy
                            else f"validator_proof_challenge_observed_late:{late_by}_blocks"
                        )
                        result = self._db.quarantine_capacity_audit_window(
                            str(window["audit_id"]),
                            reason=reason,
                            reviewed_at=observed_at,
                        )
                        quarantined += int(result.get("windows_changed") or 0)
                        bt.logging.error(
                            "Capacity audit validator-incident quarantine: "
                            f"audit_id={str(window['audit_id'])[:12]} "
                            f"B_proof={challenge_block} live_head="
                            f"{int(live_head) if trustworthy else 'unavailable'} "
                            f"reason={reason}"
                        )
        if updated:
            bt.logging.info(f"Capacity audit: recorded {updated} proof challenge hashes")
        if quarantined:
            self._schedule_capacity_audit_shared_state_write()

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
            self._apply_finalized_capacity_audit_probations()
            return
        if not windows:
            self._apply_finalized_capacity_audit_probations()
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
        self._apply_finalized_capacity_audit_probations()

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
        probation_policy = getattr(
            self,
            "_capacity_audit_failure_requires_probation",
            None,
        )
        if not callable(probation_policy):
            probation_policy = (
                ValidatorNeuron._capacity_audit_failure_requires_probation.__get__(
                    self
                )
            )
        expired = self._db.expire_capacity_audit_misses(
            require_proof_payload=self._capacity_audit_cfg.require_proof_payload,
            probation_required=probation_policy(
                proof_policy_required=False,
            ),
            return_slots=True,
        )
        expired_slots = expired if isinstance(expired, list) else []
        expired_count = len(expired_slots) if isinstance(expired, list) else int(expired or 0)
        if expired_count:
            bt.logging.info(f"Capacity audit: expired {expired_count} pending slots")
            for slot in expired_slots:
                verdict = str(slot.get("verdict") or "")
                if verdict != "hard_proof_miss":
                    bt.logging.info(
                        f"Capacity audit timing miss recorded pending receipt reconciliation: "
                        f"{str(slot.get('miner_address') or '')[:10]} "
                        f"model_index={int(slot.get('model_index') or 0)} "
                        f"verdict={verdict or 'unknown'}"
                    )
            self._write_shared_state()
            self._apply_finalized_capacity_audit_probations()
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

    def _validate_capacity_audit_artifact(
        self,
        artifact: dict,
        *,
        receipt_ingress: bool = False,
    ) -> tuple[dict, Optional[str]]:
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

        row = self._db.get_capacity_audit_slot(
            audit_id,
            address,
            model_index,
            receipt_ingress=receipt_ingress,
        )
        if row is None:
            self._recover_capacity_audit_window_from_artifact(artifact)
            row = self._db.get_capacity_audit_slot(
                audit_id,
                address,
                model_index,
                receipt_ingress=receipt_ingress,
            )
        if row is None:
            return {}, "unknown audit slot"
        if str(row.get("chain_status") or "") == "reorged":
            # A reorg is terminal evidence for this window.  In particular, a
            # queued proof worker must never turn it back into a pass/failure
            # after the finality monitor has invalidated its chain anchors.
            return {}, "capacity audit window was reorged"
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
        expected_gpu_index = int(row.get("gpu_index") or 0)
        # This value is part of the validator-scheduled challenge domain.  It
        # must be signed into every pass0/final/proof artifact, not defaulted
        # when absent, otherwise artifact ABI variants can silently omit a
        # binding that changes the proof seeds.
        if "gpu_index" not in artifact:
            return {}, "missing gpu_index"
        artifact_gpu_index = artifact["gpu_index"]
        if type(artifact_gpu_index) is not int or artifact_gpu_index != expected_gpu_index:
            return {}, "gpu_index mismatch"
        row = self._recover_capacity_audit_hashes_for_artifact(row, artifact)
        return row, None

    def ingest_capacity_audit_artifact(
        self,
        artifact: dict,
        *,
        received_at: Optional[float] = None,
        ingress_head: Optional[_CapacityAuditIngressHead] = None,
        receipt_ingress: bool = False,
    ) -> tuple[int, dict]:
        """Ingest a miner-published capacity audit artifact."""
        if not self._capacity_audit_cfg.enabled:
            return 404, {"ok": False, "error": "capacity audit disabled"}
        if not isinstance(artifact, dict):
            return 400, {"ok": False, "error": "artifact must be an object"}
        artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "")
        row, error = self._validate_capacity_audit_artifact(
            artifact,
            receipt_ingress=receipt_ingress,
        )
        if error:
            return 400, {"ok": False, "error": error}
        if str(row.get("status") or "") == "validator_incident":
            return 200, {
                "ok": True,
                "verdict": "timing_excused",
                "timing_status": "excused",
                "validator_incident": True,
            }
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
                receipt_ingress=receipt_ingress,
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

            if row.get("final_received_at") is not None:
                try:
                    stored_final = json.loads(str(row.get("final_artifact") or "{}"))
                except Exception:
                    stored_final = {}
                if stored_final != artifact:
                    return 409, {
                        "ok": False,
                        "error": "final receipt already committed",
                    }
                return 200, {
                    "ok": True,
                    "verdict": str(row.get("verdict") or "pending"),
                    "timing_status": str(row.get("timing_status") or "pending"),
                }

            final_protocol_version = combined_proof_protocol_version(
                artifact.get("combined")
            )
            final_observed_block: Optional[int] = None
            if final_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
                proof_challenge_block = int(row.get("proof_challenge_block") or 0)
                if proof_challenge_block <= 0:
                    return 409, {
                        "ok": False,
                        "error": "proof challenge block required",
                    }
                if ingress_head is not None:
                    final_observed_block = int(ingress_head.block)
                    live_head_ok = bool(ingress_head.trustworthy)
                else:
                    # Direct in-process callers retain the same fail-closed
                    # interface, but the implementation now reads only the
                    # stream-owned cache and never performs worker-side RPC.
                    final_observed_block, live_head_ok = (
                        self._observe_live_capacity_audit_head()
                    )
                if not live_head_ok or final_observed_block <= 0:
                    return 503, {
                        "ok": False,
                        "error": "live chain head unavailable",
                    }
                if final_observed_block >= proof_challenge_block:
                    stored, recorded = self._db.record_capacity_audit_final(
                        audit_id=audit_id,
                        address=address,
                        model_index=model_index,
                        final_root=final_root,
                        transcript_root=transcript,
                        artifact=artifact,
                        timing_status="invalid_chronology",
                        verdict="hard_proof_miss",
                        failure_reason="v2_final_commitment_not_pre_challenge",
                        probation_required=self._capacity_audit_failure_requires_probation(
                            proof_policy_required=True,
                        ),
                        final_observed_block=final_observed_block,
                        received_at=ts,
                        receipt_ingress=receipt_ingress,
                    )
                    return 200, {
                        "ok": True,
                        "verdict": str(
                            (stored or {}).get("verdict") or "hard_proof_miss"
                        ),
                        "timing_status": "invalid_chronology",
                    }

            if row.get("pass0_root") and pass0_root and pass0_root != row.get("pass0_root"):
                stored, recorded = self._db.record_capacity_audit_final(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    final_root=final_root,
                    transcript_root=transcript,
                    artifact=artifact,
                    timing_status="invalid_transcript",
                    verdict="hard_proof_miss",
                    failure_reason="pass0_root_mismatch",
                    probation_required=self._capacity_audit_failure_requires_probation(
                        proof_policy_required=False,
                    ),
                    final_observed_block=final_observed_block,
                    received_at=ts,
                    receipt_ingress=receipt_ingress,
                )
                stored_verdict = str(
                    (stored or {}).get("verdict") or "hard_proof_miss"
                )
                return 200, {"ok": True, "verdict": stored_verdict}

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
            stored, recorded = self._db.record_capacity_audit_final(
                audit_id=audit_id,
                address=address,
                model_index=model_index,
                final_root=final_root,
                transcript_root=transcript,
                artifact=artifact,
                timing_status=timing_status,
                verdict=verdict,
                failure_reason=failure_reason,
                probation_required=(
                    verdict == "timing_miss"
                    and self._capacity_audit_failure_requires_probation(
                        proof_policy_required=False,
                    )
                ),
                final_observed_block=final_observed_block,
                received_at=ts,
                receipt_ingress=receipt_ingress,
            )
            if stored is None:
                return 404, {"ok": False, "error": "capacity audit slot not found"}
            if not recorded:
                try:
                    stored_final = json.loads(
                        str(stored.get("final_artifact") or "{}")
                    )
                except Exception:
                    stored_final = {}
                if stored_final != artifact:
                    return 409, {
                        "ok": False,
                        "error": "final receipt already committed",
                    }
                return 200, {
                    "ok": True,
                    "verdict": str(stored.get("verdict") or "pending"),
                    "timing_status": str(
                        stored.get("timing_status") or "pending"
                    ),
                }
            verdict = str(stored.get("verdict") or verdict)
            timing_status = str(stored.get("timing_status") or timing_status)
            if recorded and verdict == "timing_miss":
                bt.logging.info(
                    f"Capacity audit timing miss recorded pending receipt reconciliation: "
                    f"{address[:10]} model_index={model_index} audit_id={audit_id[:12]}"
                )
            if (
                recorded
                and verdict == "timing_pass"
                and not self._capacity_audit_cfg.require_proof_payload
            ):
                self._db.release_capacity_audit_drain(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    released_at=ts,
                )
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
                *,
                proof_policy_required: bool = False,
            ) -> tuple[int, dict]:
                self._db.record_capacity_audit_proof_verdict(
                    audit_id=audit_id,
                    address=address,
                    model_index=model_index,
                    proof_status="invalid_payload",
                    verdict="hard_proof_miss",
                    failure_reason=reason,
                    proof_verify_ms=proof_verify_ms,
                    probation_required=self._capacity_audit_failure_requires_probation(
                        proof_policy_required=proof_policy_required,
                    ),
                    received_at=ts,
                )
                self._schedule_capacity_audit_shared_state_write()
                self._apply_finalized_capacity_audit_probations()
                body = {"ok": True, "verdict": "hard_proof_miss", "proof_status": "invalid_payload"}
                if proof_verify_ms is not None:
                    body["proof_verify_ms"] = proof_verify_ms
                return 200, body

            proof = artifact.get("sampled_pass_proof")
            if is_combined_proof_payload(proof):
                combined_protocol_version = combined_proof_protocol_version(proof)
                if combined_protocol_version is None:
                    return _record_hard_payload_miss(
                        "unsupported_or_ambiguous_capacity_proof_protocol",
                        proof_policy_required=True,
                    )
                if (
                    combined_protocol_version
                    == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
                ):
                    try:
                        final_artifact = json.loads(
                            str(row.get("final_artifact") or "{}")
                        )
                    except Exception:
                        final_artifact = {}
                    final_protocol_version = combined_proof_protocol_version(
                        final_artifact.get("combined")
                        if isinstance(final_artifact, dict)
                        else None
                    )
                    if (
                        final_protocol_version
                        != CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
                    ):
                        return _record_hard_payload_miss(
                            "missing_v2_pre_challenge_final_commitment",
                            proof_policy_required=True,
                        )
                    proof_challenge_block = int(
                        row.get("proof_challenge_block") or 0
                    )
                    final_observed_block = row.get("final_observed_block")
                    if (
                        proof_challenge_block <= 0
                        or type(final_observed_block) is not int
                        or final_observed_block <= 0
                        or final_observed_block >= proof_challenge_block
                    ):
                        return _record_hard_payload_miss(
                            "v2_final_commitment_not_pre_challenge",
                            proof_policy_required=True,
                        )
                legacy_compatibility = self._legacy_v1_compatibility_active()
                if (
                    combined_protocol_version
                    == LEGACY_COMBINED_PROOF_PROTOCOL_VERSION
                    and not legacy_compatibility
                ):
                    return _record_hard_payload_miss(
                        "legacy_capacity_proof_protocol_not_accepted",
                        proof_policy_required=True,
                    )
                proof_verify_ms: Optional[float] = None
                gpu_index = int(row.get("gpu_index") or 0)
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
                    expected_workload_spec = (
                        self._capacity_audit_expected_workload_spec(row)
                    )
                    ok, reason = verify_combined_proof_payload(
                        proof=proof,
                        final_artifact=final_artifact,
                        expected_combined_transcript_root=str(row["transcript_root"]),
                        lease_id=str(row["lease_id"]),
                        gpu_index=gpu_index,
                        proof_seed_hex=proof_seed,
                        proof_challenge_seed_hex=proof_challenge_seed,
                        expected_workload_spec=expected_workload_spec,
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
                        self._schedule_capacity_audit_shared_state_write()
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
                    policy = evaluate_proof_policy(
                        protocol_version=combined_protocol_version,
                        verification_passed=False,
                        legacy_v1_compatibility_active=legacy_compatibility,
                    )
                    return _record_hard_payload_miss(
                        reason,
                        proof_verify_ms,
                        proof_policy_required=policy.probation_required,
                    )
                policy = evaluate_proof_policy(
                    protocol_version=combined_protocol_version,
                    verification_passed=True,
                    legacy_v1_compatibility_active=legacy_compatibility,
                )
                if not policy.accepted:
                    return _record_hard_payload_miss(
                        policy.reason,
                        proof_verify_ms,
                        proof_policy_required=policy.probation_required,
                    )
                self._mark_capacity_audit_verifier_healthy()

                proof_path = os.path.join(
                    self._capacity_audit_artifact_dir(audit_id),
                    f"{address}_{model_index}_proof_payload.json",
                )
                with open(proof_path, "w") as f:
                    json.dump(artifact, f, sort_keys=True)
                current_verdict = str(row.get("verdict") or "pending")
                proof_status = (
                    "legacy_combined_proof_compatibility_accepted"
                    if policy.legacy_compatibility_accepted
                    else "combined_proof_verified"
                )
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
                    self._schedule_capacity_audit_shared_state_write()
                return 200, {
                    "ok": True,
                    "verdict": current_verdict,
                    "proof_status": proof_status,
                    "proof_verify_ms": proof_verify_ms,
                }

            return _record_hard_payload_miss(
                "unsupported_proof_payload_format",
                proof_policy_required=True,
            )

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
        if str(row.get("status") or "") == "validator_incident":
            return 200, {
                "ok": True,
                "verdict": "timing_excused",
                "timing_status": "excused",
                "validator_incident": True,
            }, None
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
                    self._schedule_capacity_audit_shared_state_write()
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
        status, body, row = self._prepare_capacity_audit_proof_enqueue(
            artifact,
            received_at=ts,
        )
        if status >= 300:
            return status, body
        if self._proof_v3_follower_mode_active():
            assert row is not None
            self._db.release_capacity_audit_drain(
                audit_id=str(row["audit_id"]),
                address=str(row["miner_address"]),
                model_index=int(row["model_index"]),
                released_at=ts,
            )
            self._schedule_capacity_audit_shared_state_write()
            return status, body
        self._capacity_audit_proof_executor.submit(
            self._run_capacity_audit_proof_verification,
            artifact,
            ts,
        )
        return status, body

    def _ensure_capacity_audit_receipt_metrics(self) -> None:
        if hasattr(self, "_capacity_audit_receipt_metrics_lock"):
            return
        self._capacity_audit_receipt_metrics_lock = threading.Lock()
        self._capacity_audit_receipt_ticket = 0
        self._capacity_audit_receipt_pending = {}
        self._capacity_audit_receipt_metrics = {
            "submitted": 0,
            "completed": 0,
            "head_unavailable": 0,
            "max_queue_depth": 0,
            "max_queue_delay_s": 0.0,
            "max_processing_s": 0.0,
            "per_audit": {},
        }

    def _capacity_audit_receipt_enqueued(
        self,
        artifact: dict,
        received_at: float,
    ) -> int:
        self._ensure_capacity_audit_receipt_metrics()
        audit_id = str(artifact.get("audit_id") or "")
        with self._capacity_audit_receipt_metrics_lock:
            self._capacity_audit_receipt_ticket += 1
            ticket = int(self._capacity_audit_receipt_ticket)
            self._capacity_audit_receipt_pending[ticket] = (
                float(received_at),
                audit_id,
            )
            metrics = self._capacity_audit_receipt_metrics
            metrics["submitted"] = int(metrics["submitted"]) + 1
            depth = len(self._capacity_audit_receipt_pending)
            metrics["max_queue_depth"] = max(
                int(metrics["max_queue_depth"]),
                depth,
            )
            per_audit = metrics["per_audit"]
            assert isinstance(per_audit, dict)
            item = per_audit.setdefault(
                audit_id,
                {
                    "submitted": 0,
                    "completed": 0,
                    "head_unavailable": 0,
                    "max_queue_delay_s": 0.0,
                    "max_processing_s": 0.0,
                    "last_received_at": 0.0,
                },
            )
            item["submitted"] += 1
            item["last_received_at"] = float(received_at)
            # Bound process-lifetime diagnostic state. Active/most-recent
            # audit IDs are retained; old completed IDs are evicted first.
            if len(per_audit) > 64:
                oldest = min(
                    per_audit,
                    key=lambda key: float(per_audit[key]["last_received_at"]),
                )
                if oldest != audit_id:
                    per_audit.pop(oldest, None)
            return ticket

    def _capacity_audit_receipt_finished(
        self,
        *,
        ticket: int,
        queue_delay_s: float,
        processing_s: float,
        head_unavailable: bool,
        accepted: bool,
    ) -> None:
        self._ensure_capacity_audit_receipt_metrics()
        audit_id = ""
        with self._capacity_audit_receipt_metrics_lock:
            _received_at, audit_id = self._capacity_audit_receipt_pending.pop(
                int(ticket),
                (0.0, ""),
            )
            metrics = self._capacity_audit_receipt_metrics
            metrics["completed"] = int(metrics["completed"]) + 1
            metrics["max_queue_delay_s"] = max(
                float(metrics["max_queue_delay_s"]),
                float(queue_delay_s),
            )
            metrics["max_processing_s"] = max(
                float(metrics["max_processing_s"]),
                float(processing_s),
            )
            if head_unavailable:
                metrics["head_unavailable"] = int(metrics["head_unavailable"]) + 1
            per_audit = metrics["per_audit"]
            assert isinstance(per_audit, dict)
            item = per_audit.get(audit_id)
            if item is not None:
                item["completed"] += 1
                item["max_queue_delay_s"] = max(
                    float(item["max_queue_delay_s"]),
                    float(queue_delay_s),
                )
                item["max_processing_s"] = max(
                    float(item["max_processing_s"]),
                    float(processing_s),
                )
                if head_unavailable:
                    item["head_unavailable"] += 1
        incident_reason = ""
        if head_unavailable:
            incident_reason = "validator_live_head_unavailable_at_receipt_ingress"
        elif accepted and float(queue_delay_s) >= 5.0:
            incident_reason = f"validator_receipt_queue_delay:{float(queue_delay_s):.3f}s"
        elif accepted and float(processing_s) >= 5.0:
            incident_reason = (
                f"validator_receipt_processing_delay:{float(processing_s):.3f}s"
            )
        if audit_id and incident_reason:
            try:
                signaled = self._db.mark_capacity_audit_validator_signal(
                    audit_id,
                    reason=incident_reason,
                )
                if signaled:
                    bt.logging.error(
                        "Capacity audit validator incident signal: "
                        f"audit_id={audit_id[:12]} reason={incident_reason}"
                    )
            except Exception as exc:
                bt.logging.warning(
                    "Capacity audit incident signal persistence failed: "
                    f"audit_id={audit_id[:12]} error={exc}"
                )

    def _capacity_audit_receipt_metrics_snapshot(self) -> dict:
        self._ensure_capacity_audit_receipt_metrics()
        now = time.time()
        with self._capacity_audit_receipt_metrics_lock:
            pending = dict(self._capacity_audit_receipt_pending)
            metrics = self._capacity_audit_receipt_metrics
            return {
                "queue_depth": len(pending),
                "oldest_age_s": max(
                    (max(0.0, now - row[0]) for row in pending.values()),
                    default=0.0,
                ),
                "submitted": int(metrics["submitted"]),
                "completed": int(metrics["completed"]),
                "head_unavailable": int(metrics["head_unavailable"]),
                "max_queue_depth": int(metrics["max_queue_depth"]),
                "max_queue_delay_s": float(metrics["max_queue_delay_s"]),
                "max_processing_s": float(metrics["max_processing_s"]),
                "per_audit": {
                    key: dict(value)
                    for key, value in dict(metrics["per_audit"]).items()
                },
            }

    def _schedule_capacity_audit_shared_state_write(self) -> None:
        executor = getattr(self, "_capacity_audit_state_executor", None)
        lock = getattr(self, "_capacity_audit_state_lock", None)
        if executor is None or lock is None:
            self._write_shared_state()
            return
        with lock:
            self._capacity_audit_state_dirty = True
            self._capacity_audit_state_requested_at = time.monotonic()
            if self._capacity_audit_state_pending:
                return
            self._capacity_audit_state_pending = True

        def _flush() -> None:
            # Wait for a quiet trailing edge. The former dirty-loop started a
            # full rebuild after 50 ms and immediately repeated it whenever a
            # new completion landed during serialization; a normal cohort
            # could therefore rebuild shared state dozens of times.
            debounce_s = 0.5
            while True:
                with lock:
                    requested_at = float(self._capacity_audit_state_requested_at)
                remaining = debounce_s - (time.monotonic() - requested_at)
                if remaining > 0.0:
                    time.sleep(remaining)
                    continue
                with lock:
                    if requested_at != float(self._capacity_audit_state_requested_at):
                        continue
                    self._capacity_audit_state_dirty = False
                try:
                    self._write_shared_state()
                except Exception as exc:
                    bt.logging.warning(
                        f"Capacity audit shared-state refresh failed: {exc}"
                    )
                with lock:
                    if self._capacity_audit_state_dirty:
                        continue
                    self._capacity_audit_state_pending = False
                    return

        executor.submit(_flush)

    def _build_capacity_audit_ingest_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        app = FastAPI(title="Verathos Capacity Audit Ingest")

        async def _read_payload(
            request: Request,
        ) -> tuple[int, dict, float, _CapacityAuditIngressHead]:
            max_bytes = int(
                getattr(
                    self._capacity_audit_cfg,
                    "max_proof_payload_bytes",
                    32 * 1024 * 1024,
                )
                or 32 * 1024 * 1024
            )
            body = await request.body()
            received_at = time.time()
            ingress_head = self._capture_capacity_audit_ingress_head()
            if len(body) > max_bytes:
                return 413, {
                    "error": "payload_too_large",
                    "max_bytes": max_bytes,
                }, received_at, ingress_head
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return 400, {"error": "invalid_json"}, received_at, ingress_head
            if not isinstance(payload, dict):
                return 400, {"error": "payload_must_be_object"}, received_at, ingress_head
            return 200, payload, received_at, ingress_head

        @app.get("/capacity/audit/v1/health")
        async def _health():
            head = self._capture_capacity_audit_ingress_head()
            return {
                "status": "ok",
                "service": "verathos-capacity-audit-ingest",
                "capacity_audit": True,
                "protocol_version": PROTOCOL_VERSION,
                "receipt_chronology": {
                    "block": int(head.block),
                    "source": str(head.source),
                    "trustworthy": bool(head.trustworthy),
                    "observer_pid": int(
                        getattr(
                            getattr(self, "_capacity_audit_head_process", None),
                            "pid",
                            0,
                        )
                        or 0
                    ),
                },
                "receipt_ingest": self._capacity_audit_receipt_metrics_snapshot(),
            }

        @app.get("/v1/verdicts/current")
        async def _verdict_snapshot(epoch: int | None = None):
            """Serve owner-signed bytes directly from the owner validator."""

            if not _proof_v3_hard_auditor_active(
                self.config,
                getattr(self, "_validator_hotkey_ss58", ""),
            ):
                return JSONResponse(
                    status_code=503,
                    content={"error": "verdict snapshot unavailable"},
                    headers={"Cache-Control": "no-store", "Retry-After": "15"},
                )
            snapshot = (
                (getattr(self, "_verdict_snapshot_history", {}) or {}).get(
                    int(epoch),
                    "",
                )
                if epoch is not None
                else str(getattr(self, "_verdict_snapshot_hex", "") or "")
            )
            if not snapshot:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "verdict snapshot unavailable",
                        **({"epoch": int(epoch)} if epoch is not None else {}),
                    },
                    headers={"Cache-Control": "no-store", "Retry-After": "15"},
                )
            return JSONResponse(
                content={"snapshot": snapshot},
                headers={"Cache-Control": "public, max-age=15"},
            )

        async def _receipt(request):
            read_status, payload, received_at, ingress_head = await _read_payload(request)
            if read_status != 200:
                return JSONResponse(status_code=read_status, content=payload)
            ticket = self._capacity_audit_receipt_enqueued(payload, received_at)
            loop = asyncio.get_running_loop()
            status, body = await loop.run_in_executor(
                self._capacity_audit_receipt_executor,
                self._ingest_capacity_audit_receipt,
                payload,
                received_at,
                ingress_head,
                ticket,
            )
            return JSONResponse(status_code=status, content=body)

        async def _proof(request):
            read_status, payload, received_at, _ingress_head = await _read_payload(request)
            if read_status != 200:
                return JSONResponse(status_code=read_status, content=payload)
            status, body = self.submit_capacity_audit_proof_artifact(
                payload,
                received_at=received_at,
            )
            return JSONResponse(status_code=status, content=body)

        _receipt.__annotations__["request"] = Request
        _proof.__annotations__["request"] = Request
        app.post("/capacity/audit/v1/receipt")(_receipt)
        app.post("/capacity/audit/v1/proof")(_proof)

        return app

    def _ingest_capacity_audit_receipt(
        self,
        artifact: dict,
        received_at: float,
        ingress_head: _CapacityAuditIngressHead,
        ticket: int,
    ) -> tuple[int, dict]:
        processing_started_at = time.time()
        status = 500
        body: dict = {"ok": False, "error": "receipt worker failed"}
        try:
            status, body = self.ingest_capacity_audit_artifact(
                artifact,
                received_at=received_at,
                ingress_head=ingress_head,
                receipt_ingress=True,
            )
            return status, body
        finally:
            queue_delay_s = max(0.0, processing_started_at - received_at)
            processing_s = max(0.0, time.time() - processing_started_at)
            head_unavailable = bool(
                status == 503
                and str(body.get("error") or "") == "live chain head unavailable"
            )
            self._capacity_audit_receipt_finished(
                ticket=ticket,
                queue_delay_s=queue_delay_s,
                processing_s=processing_s,
                head_unavailable=head_unavailable,
                accepted=bool(status == 200 and body.get("ok") is True),
            )
            if queue_delay_s >= 0.5 or processing_s >= 0.5:
                bt.logging.warning(
                    "Capacity audit receipt ingest delay: "
                    f"queue_s={queue_delay_s:.3f} processing_s={processing_s:.3f} "
                    f"type={artifact.get('artifact_type') or artifact.get('type') or ''} "
                    f"audit_id={str(artifact.get('audit_id') or '')[:12]}"
                )

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
        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
        if not cfg.enabled or cfg.mode not in ("score_gate", "enforce"):
            return False
        if self._maintenance_grace_active(
            current_epoch=epoch_number,
            action="suppress_capacity_score_gate",
        ):
            if epoch_number is not None:
                logged_epoch = getattr(
                    self,
                    "_capacity_audit_grace_log_epoch",
                    None,
                )
                if logged_epoch != int(epoch_number):
                    self._capacity_audit_grace_log_epoch = int(epoch_number)
                    bt.logging.info(
                        "Capacity audit enforcement disabled for this epoch: "
                        f"{self._maintenance_grace_reason()}"
                    )
            return False
        if bool(
            self._epoch_close_value(
                "_subnet_runtime_config_authoritative",
                False,
            )
        ):
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

    def _capacity_audit_expected_workload_spec(self, row: Mapping[str, Any]) -> dict:
        """Return the validator-scheduled immutable workload for proof checks."""

        stored = row.get("workload_spec")
        if isinstance(stored, str) and stored:
            decoded = json.loads(stored)
        elif isinstance(stored, dict):
            decoded = stored
        else:
            decoded = {}
        if isinstance(decoded, dict) and decoded.get("workload_version"):
            return decoded

        # Compatibility for audit rows scheduled immediately before this DB
        # column existed.  The calibrated row is validator-owned runtime data,
        # never a parameter supplied by the proof payload.
        gpu_row = match_gpu_class(
            str(row.get("claimed_gpu_class") or ""),
            int(row.get("vram_gb") or 0),
            self._capacity_audit_cfg,
        )
        if gpu_row is None:
            raise RuntimeError("scheduled capacity workload is unavailable")
        return capacity_gpu_workload_spec(gpu_row)

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
        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
        if not cfg.enabled or cfg.mode != "score_gate":
            return ""
        if not self._capacity_audit_enforcement_enabled(epoch_number):
            return ""
        if self._proof_v3_follower_mode_active():
            return self._follower_capacity_gate_reason(
                address,
                model_index,
            )
        if getattr(self, "_capacity_audit_verifier_unhealthy", False):
            reason = str(getattr(self, "_capacity_audit_verifier_last_error", "") or "")
            suffix = f": {reason[:160]}" if reason else ""
            bt.logging.warning(
                f"Capacity audit score gate disabled while proof verifier is unhealthy{suffix}"
            )
            return ""
        since_epoch = max(
            0,
            int(epoch_number) - int(cfg.repeat_window_epochs) + 1,
            int(getattr(self, "_probation_state_reset_effective_epoch", 0) or 0),
        )
        invalid_failures = self._db.recent_invalid_capacity_proof_failures(
            address,
            model_index,
            since_epoch=since_epoch,
            require_chain_confirmed=True,
            require_consequence_eligible=True,
            require_incident_reviewed=True,
        )
        if invalid_failures >= int(cfg.invalid_proof_misses_for_zero_score):
            return f"{invalid_failures} cryptographically invalid capacity proof(s)"
        hard_failures = self._db.recent_capacity_failures(
            address,
            model_index,
            since_epoch=since_epoch,
            verdicts=("hard_proof_miss",),
            require_chain_confirmed=True,
            require_consequence_eligible=True,
            require_incident_reviewed=True,
        )
        # A no-show/timing result can still be neutralized by authenticated
        # work that was already running at the scoring boundary.  Mature these
        # transport/timing-only outcomes for one epoch before applying any
        # score or probation consequence. Cryptographically invalid hard
        # proofs remain immediate.
        hard_failures += self._db.recent_capacity_failures(
            address,
            model_index,
            since_epoch=since_epoch,
            verdicts=("no_show",),
            require_chain_confirmed=True,
            require_consequence_eligible=True,
            require_incident_reviewed=True,
            before_epoch=int(epoch_number),
        )
        if hard_failures >= int(cfg.hard_proof_misses_for_zero_score):
            return f"{hard_failures} hard capacity-audit failures"
        if cfg.allow_timing_only_score_gate:
            timing_failures = self._db.recent_capacity_failures(
                address,
                model_index,
                since_epoch=since_epoch,
                verdicts=("timing_miss",),
                require_chain_confirmed=True,
                require_consequence_eligible=True,
                require_incident_reviewed=True,
                before_epoch=int(epoch_number),
            )
            if timing_failures >= int(cfg.timing_misses_for_zero_score):
                return f"{timing_failures} timing capacity-audit misses"
        return ""

    def _capacity_audit_uid_score_gate_reason(
        self,
        uid: Optional[int],
        epoch_number: int,
    ) -> str:
        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
        if uid is None or not cfg.enabled or cfg.mode != "score_gate":
            return ""
        if not bool(getattr(cfg, "uid_escalation_enabled", False)):
            return ""
        if not self._capacity_audit_enforcement_enabled(epoch_number):
            return ""
        if self._proof_v3_follower_mode_active():
            # The signed per-entry capacity_gated bit already includes the
            # owner's endpoint and UID-quorum decision. Applying it in the
            # endpoint seam avoids recomputing owner evidence locally.
            return ""
        if getattr(self, "_capacity_audit_verifier_unhealthy", False):
            return ""
        since_epoch = max(
            0,
            int(epoch_number) - int(cfg.repeat_window_epochs) + 1,
            int(getattr(self, "_probation_state_reset_effective_epoch", 0) or 0),
        )
        counts = self._db.recent_capacity_failure_counts_for_uid(
            int(uid),
            since_epoch=since_epoch,
            require_chain_confirmed=True,
            require_consequence_eligible=True,
            require_incident_reviewed=True,
            soft_failures_before_epoch=int(epoch_number),
        )
        convicted: list[Tuple[str, int]] = []
        for key, row in counts.items():
            if int(row.get("invalid_proof_failures", 0)) >= int(
                cfg.invalid_proof_misses_for_zero_score
            ):
                convicted.append(key)
                continue
            if int(row.get("hard_failures", 0)) >= int(
                cfg.hard_proof_misses_for_zero_score
            ):
                convicted.append(key)
                continue
            if cfg.allow_timing_only_score_gate and int(
                row.get("timing_failures", 0)
            ) >= int(cfg.timing_misses_for_zero_score):
                convicted.append(key)

        active_count = self._db.active_entry_count_for_uid(int(uid))
        evidence_entry_count = len(counts)
        entry_count = max(active_count, evidence_entry_count)
        if entry_count <= 1:
            return ""
        threshold = capacity_audit_uid_escalation_threshold(entry_count, cfg)
        if len(convicted) < threshold:
            return ""
        return (
            f"{len(convicted)}/{entry_count} distinct capacity-audit entries convicted "
            f"(UID quorum={threshold})"
        )

    def _ensure_capacity_audit_entry_probation(
        self,
        address: str,
        model_index: int,
        uid: int,
    ) -> None:
        close_epoch = int(
            self._epoch_close_value(
                "_current_epoch",
                getattr(self, "_current_epoch", 0),
            )
        )
        if self._probation_reset_covers_source_epoch(close_epoch):
            return
        tracker = getattr(self, "_probation_tracker", None)
        if tracker is None or self._maintenance_grace_active(action="suppress_probation"):
            return
        key = self._miner_model_key(address, model_index)
        if tracker.is_on_probation(key):
            return
        endpoint = ""
        for miner in self._epoch_close_value("_epoch_miners", ()):
            if (
                str(getattr(miner, "address", "")).lower() == address.lower()
                and int(getattr(miner, "model_index", -1)) == int(model_index)
            ):
                endpoint = str(getattr(miner, "endpoint", "") or "")
                break
        tracker.enter_probation(key, close_epoch, endpoint=endpoint)
        self._db.enter_probation(address, model_index, close_epoch, uid=int(uid))
        self._write_shared_state()

    def _apply_capacity_audit_score_gate(
        self,
        address: str,
        model_index: int,
        uid: int,
        reason: str,
        *,
        uid_wide: bool = False,
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
        if uid_wide:
            targets = list(state.entries.items())
        else:
            entry = state.entries.get(int(model_index))
            if entry is None:
                return False
            targets = [(int(model_index), entry)]
            self._ensure_capacity_audit_entry_probation(
                address,
                int(model_index),
                int(uid),
            )
        for entry_model_index, entry in targets:
            if entry.ema_score != 0.0:
                entry.ema_score = 0.0
            self._db.save_score(
                state.address,
                entry_model_index,
                entry.ema_score,
                entry.total_epochs,
                entry.scored_epochs,
            )
        scope = "UID" if uid_wide else "entry"
        bt.logging.info(
            f"Capacity audit score gate: zeroed {scope} {uid} "
            f"entries={len(targets)} trigger={address[:10]} model_index={model_index} "
            f"({reason})"
        )
        return True

    def _apply_capacity_audit_score_gates(
        self,
        address: str,
        model_index: int,
        uid: int,
        entry_reason: str,
        uid_reason: str,
    ) -> bool:
        entry_gated = self._apply_capacity_audit_score_gate(
            address,
            model_index,
            uid,
            entry_reason,
        )
        uid_gated = self._apply_capacity_audit_score_gate(
            address,
            model_index,
            uid,
            uid_reason,
            uid_wide=True,
        )
        return entry_gated or uid_gated

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
        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
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

    def _capacity_audit_failure_requires_probation(
        self,
        *,
        proof_policy_required: bool = False,
    ) -> bool:
        """Freeze probation policy when capacity failure evidence is observed."""

        follower_mode = getattr(
            self,
            "_proof_v3_follower_mode_active",
            None,
        )
        if callable(follower_mode) and follower_mode():
            return False
        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
        if not cfg.enabled or cfg.mode not in ("score_gate", "enforce"):
            return False
        if self._maintenance_grace_active(
            action="suppress_capacity_score_gate"
        ) or self._maintenance_grace_active(action="suppress_probation"):
            return False
        if proof_policy_required:
            return True
        enforcement_enabled = getattr(
            self,
            "_capacity_audit_enforcement_enabled",
            None,
        )
        if not callable(enforcement_enabled):
            enforcement_enabled = (
                ValidatorNeuron._capacity_audit_enforcement_enabled.__get__(self)
            )
        if not enforcement_enabled():
            return False
        return True

    def _review_capacity_audit_failure_clusters(self) -> int:
        """Quarantine finalized validator-shaped cohort failures before penalties."""

        cfg = self._epoch_close_value(
            "_capacity_audit_cfg",
            self._capacity_audit_cfg,
        )
        try:
            windows = self._db.get_capacity_audit_windows_for_incident_review()
        except Exception as exc:
            bt.logging.warning(
                f"Capacity audit incident review lookup failed: {exc}"
            )
            return 0

        quarantined = 0
        reviewed = 0
        for window in windows:
            audit_id = str(window.get("audit_id") or "")
            if not audit_id:
                continue
            try:
                result = self._db.review_capacity_audit_failure_cluster(
                    audit_id,
                    enabled=bool(cfg.incident_quarantine_enabled),
                    failure_fraction=float(cfg.incident_failure_fraction),
                    min_failures=int(cfg.incident_min_failures),
                    min_distinct_miners=int(cfg.incident_min_distinct_miners),
                )
            except Exception as exc:
                bt.logging.warning(
                    f"Capacity audit incident review failed: "
                    f"audit_id={audit_id[:12]} error={exc}"
                )
                continue
            if not result.get("reviewed"):
                continue
            reviewed += 1
            if not result.get("quarantined"):
                continue
            quarantined += 1
            bt.logging.error(
                "Capacity audit validator-incident quarantine: "
                f"audit_id={audit_id[:12]} "
                f"epoch={int(result.get('epoch_number') or 0)} "
                f"failures={int(result.get('eligible_failures') or 0)}/"
                f"{int(result.get('slot_count') or 0)} "
                f"distinct_miners={int(result.get('distinct_failure_miners') or 0)}; "
                f"validator_signal={str(result.get('validator_signal') or '')}; "
                "all non-cryptographic outcomes are neutral and require manual review"
            )
        if reviewed:
            self._write_shared_state()
        return quarantined

    def _apply_finalized_capacity_audit_probations(self) -> int:
        """Apply each chain-finalized capacity proof penalty exactly once."""

        follower_mode = getattr(
            self,
            "_proof_v3_follower_mode_active",
            None,
        )
        if callable(follower_mode) and follower_mode():
            return 0
        # This review is deliberately earlier than maintenance-grace checks
        # and candidate lookup. A validator-wide ingestion incident must be
        # frozen as neutral evidence before any endpoint EMA or probation
        # state can be mutated.
        self._review_capacity_audit_failure_clusters()
        if self._maintenance_grace_active(
            action="suppress_capacity_score_gate"
        ) or self._maintenance_grace_active(action="suppress_probation"):
            return 0

        try:
            candidates = self._db.get_finalized_capacity_audit_probation_candidates()
        except Exception as exc:
            bt.logging.debug(
                f"Capacity audit probation candidate lookup failed: {exc}"
            )
            return 0
        applied = 0
        for candidate in candidates:
            audit_id = str(candidate.get("audit_id") or "")
            address = str(candidate.get("miner_address") or "").lower()
            model_index = int(candidate.get("model_index") or 0)
            if not audit_id or not address:
                continue
            result = self._db.apply_finalized_capacity_audit_probation_once(
                audit_id=audit_id,
                address=address,
                model_index=model_index,
                epoch=int(self._current_epoch),
            )
            if result is None:
                continue
            applied += 1
            if result.get("miner_exists"):
                key = self._miner_model_key(address, model_index)
                endpoint = str(result.get("endpoint") or "")
                self._probation_tracker.enter_probation(
                    key,
                    self._current_epoch,
                    endpoint=endpoint,
                )
                self.scorer.halve_ema(address, model_index)
            bt.logging.info(
                f"Finalized capacity proof failure -> probation: "
                f"audit_id={audit_id[:12]} miner={address[:10]} "
                f"model_index={model_index} "
                f"reason={str(result.get('failure_reason') or 'unknown')}"
            )
        if applied:
            self._write_shared_state()
        return applied

    def _epoch_close_value(self, name: str, default=None):
        """Read a frozen old-epoch value while an async close is running."""

        local = getattr(self, "_epoch_close_local", None)
        state = getattr(local, "state", None) if local is not None else None
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        return getattr(self, name, default)

    def _set_epoch_close_value(self, name: str, value) -> None:
        """Write close-local state without corrupting the new active epoch."""

        local = getattr(self, "_epoch_close_local", None)
        state = getattr(local, "state", None) if local is not None else None
        if state is not None and hasattr(state, name):
            setattr(state, name, value)
            return
        setattr(self, name, value)

    def _capture_epoch_close_state(self, epoch_number: int):
        """Freeze the complete per-epoch scoring input before rollover."""

        if int(epoch_number) != int(self._current_epoch):
            raise RuntimeError("cannot freeze a non-current epoch")
        with self._canary_accounting_lock:
            expected_obligations = {
                key: dict(value)
                for key, value in self._expected_canary_obligations.items()
            }
            busy_probations = {
                key: list(value)
                for key, value in self._busy_skip_probations.items()
            }
            canary_error_times = {
                key: list(value)
                for key, value in self._canary_error_times.items()
            }
        with self._shared_hard_prefetch_lock:
            shared_prefetch = dict(self._shared_hard_prefetch_results)

        state = SimpleNamespace(
            epoch_number=int(epoch_number),
            _current_epoch=int(epoch_number),
            _epoch_start_block=int(self._epoch_start_block),
            _epoch_miners=tuple(self._epoch_miners),
            _endpoint_policy_gate_reasons=dict(
                getattr(self, "_endpoint_policy_gate_reasons", {}) or {}
            ),
            _endpoint_policy_excluded_miners=tuple(
                getattr(self, "_endpoint_policy_excluded_miners", ()) or ()
            ),
            _expected_receipts=dict(self._expected_receipts),
            _expected_canary_obligations=expected_obligations,
            _hard_canary_obligation_ids=set(self._hard_canary_obligation_ids),
            _validator_canary_failures=set(self._validator_canary_failures),
            _canary_penalized_keys=set(self._canary_penalized_keys),
            _busy_skips=dict(self._busy_skips),
            _busy_skip_probations=busy_probations,
            _canary_errors=dict(self._canary_errors),
            _canary_error_times=canary_error_times,
            _shared_hard_prefetch_results=shared_prefetch,
            _shared_hard_proof_verdicts={},
            _receipt_pull_failed_keys=set(),
            _scoring=self._scoring,
            _proof_v3_releases=dict(self._proof_v3_releases),
            _proof_v3_canary_policy=self._proof_v3_canary_policy,
            _proof_protocol_rollout_cfg=self._proof_protocol_rollout_cfg,
            _proof_v3_failure_policy_cfg=(
                self._proof_v3_failure_policy_cfg
            ),
            _maintenance_grace_cfg=self._maintenance_grace_cfg,
            _capacity_audit_cfg=self._capacity_audit_cfg,
            _proof_v3_verdict_source_latched=str(
                getattr(
                    self,
                    "_proof_v3_verdict_source_latched",
                    "verify",
                )
            ),
            _owner_verdict_url_latched=str(
                getattr(self, "_owner_verdict_url_latched", "") or ""
            ),
            owner_hotkey_ss58=str(
                getattr(
                    self.config,
                    "proof_v3_hard_auditor_hotkey_ss58",
                    "",
                )
                or ""
            ),
            runtime_config_loaded=bool(
                getattr(
                    self,
                    "_subnet_runtime_config_authoritative",
                    False,
                )
            ),
            _subnet_runtime_config_authoritative=bool(
                getattr(
                    self,
                    "_subnet_runtime_config_authoritative",
                    False,
                )
            ),
            weight_update_due=bool(self._weight_update_due),
        )
        # The due update belongs to this closing epoch. A later boundary may
        # independently set the live flag again while this close is waiting.
        self._weight_update_due = False
        return state

    def _run_queued_epoch_close(self, state) -> None:
        """Finish one frozen epoch without blocking later block callbacks."""

        epoch_number = int(state.epoch_number)
        backoff = 30.0
        local = getattr(self, "_epoch_close_local", None)
        if local is None:
            local = threading.local()
            self._epoch_close_local = local
        try:
            while self._running:
                with self._epoch_close_lock:
                    if epoch_number <= int(
                        getattr(self, "_last_closed_epoch", -1)
                    ):
                        return
                    local.state = state
                    try:
                        self._closing_inflight_canaries[epoch_number] = dict(
                            self._inflight_canaries.get(epoch_number, {})
                        )
                        try:
                            self._close_epoch(epoch_number)
                        finally:
                            self._closing_inflight_canaries.pop(
                                epoch_number,
                                None,
                            )
                    except Exception as exc:
                        bt.logging.warning(
                            f"Epoch {epoch_number} close failed, retrying in "
                            f"{backoff:.0f}s without blocking the active epoch: "
                            f"{exc}"
                        )
                    else:
                        self._last_closed_epoch = epoch_number
                        if self._pending_epoch_close == epoch_number:
                            self._pending_epoch_close = None
                        # The close used its frozen old-epoch scoring policy.
                        # Restore the active epoch's policy before returning;
                        # canary setup may have refreshed it while the owner
                        # snapshot was pending.
                        active_scoring = getattr(self, "_scoring", None)
                        if active_scoring is not None:
                            self.scorer.ema_alpha = active_scoring.ema_alpha
                            self.scorer.throughput_power = (
                                active_scoring.throughput_power
                            )
                        if bool(state.weight_update_due):
                            self._schedule_weight_update()
                        return
                    finally:
                        local.state = None
                # Keep shutdown responsive and bound each sleep even after
                # repeated infrastructure failures.
                deadline = time.monotonic() + backoff
                while self._running and time.monotonic() < deadline:
                    time.sleep(min(1.0, deadline - time.monotonic()))
                backoff = min(backoff * 2.0, 300.0)
        finally:
            local.state = None
            self._epoch_close_futures.pop(epoch_number, None)
            auto_updater = getattr(self, "_auto_updater", None)
            if auto_updater is not None:
                auto_updater.notify_not_busy()

    def _queue_epoch_close(self, epoch_number: int) -> bool:
        """Queue one immutable close and return without waiting for it."""

        epoch_number = int(epoch_number)
        if epoch_number in self._epoch_close_futures:
            return False
        state = self._capture_epoch_close_state(epoch_number)
        self._epoch_close_futures[epoch_number] = None
        future = self._epoch_close_executor.submit(
            self._run_queued_epoch_close,
            state,
        )
        if epoch_number in self._epoch_close_futures:
            self._epoch_close_futures[epoch_number] = future
        bt.logging.info(
            f"Epoch {epoch_number} close queued asynchronously; next-epoch "
            "canary scheduling remains independent"
        )
        return True

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
        if block_number % self.config.set_weights_epoch_blocks == 0:
            # Weight setting must use the score from the epoch being closed.
            # When close is delayed for grace/in-flight work, retain the due
            # update and submit it immediately after that close succeeds.
            self._weight_update_due = True
        sched_count = len(self._canary_scheduler.tests) if self._canary_scheduler else 0
        bt.logging.info(
            f"Block {block_number} | epoch {blocks_into_epoch}/{epoch_blocks} "
            f"| next_epoch_in={blocks_until_next} | pending_tests={sched_count}",
        )

        # Reformat cached metagraph statistics off-thread every 20 blocks and
        # re-log them every five. Authoritative metagraph refresh already runs
        # in background epoch setup/close; no periodic RPC belongs on this
        # chronology-sensitive callback.
        if block_number % 20 == 0:
            self._schedule_metagraph_stats_refresh()
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
        self._maybe_schedule_shared_hard_prefetch(
            block_number=block_number,
            blocks_into_epoch=blocks_into_epoch,
        )

        # 1. Epoch boundary → freeze the old epoch and start the new one.
        # Receipt reconciliation, owner-snapshot waiting and scoring continue
        # on the dedicated close worker. They must never delay the next
        # epoch's setup or canary dispatch.
        if block_number % epoch_blocks == 0:
            if self._pending_epoch_close is None:
                self._start_new_epoch(block_number)
            else:
                closing_epoch = int(self._pending_epoch_close)
                bt.logging.info(
                    f"Epoch {closing_epoch} reached its boundary; freezing "
                    "its close state before the next epoch starts"
                )
                self._seal_canary_epoch_for_close(
                    closing_epoch
                )
                self._queue_epoch_close(closing_epoch)
                self._start_new_epoch(block_number)

        # 1b. Retry failed epoch start — skip while background setup is in flight.
        elif (
            blocks_into_epoch <= 30  # only retry in first ~6 min
            and self._pending_epoch_close is not None
            and self._current_epoch == block_number // epoch_blocks
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
            auto_updater = getattr(self, "_auto_updater", None)
            update_drain = bool(
                auto_updater is not None
                and getattr(auto_updater, "drain_requested", False)
            )
            if update_drain:
                # Preserve the canonical schedule while a detected version
                # bump drains already admitted work into a safe restart.
                auto_updater.notify_not_busy()
            else:
                with self._canary_scheduler_lock:
                    pending = self._canary_scheduler.get_pending_tests(
                        block_number
                    )
                pending = self._filter_stale_canary_dispatches(
                    pending,
                    processed_block=block_number,
                )
                pending = self._defer_capacity_audit_drained_canaries(
                    pending,
                    block_number,
                )
                if pending:
                    for t in pending:
                        _uid = self._db.get_uid(t.miner_address)
                        _uid_str = (
                            f"UID {_uid}" if _uid is not None else "UID ?"
                        )
                        bt.logging.debug(
                            f"Dispatching canary: {_uid_str} "
                            f"{t.miner_address[:10]} "
                            f"model={t.model_id} type={t.test_type} "
                            f"proof={t.verify_proof} "
                            f"tokens={t.max_new_tokens}",
                        )
                    self._dispatch_canary_tests(pending)

        self._schedule_miner_debug_refresh(
            current_epoch=block_number // max(1, int(epoch_blocks)),
        )

        _wd_elapsed = time.monotonic() - _wd_t0
        if _wd_elapsed > 12.0:
            bt.logging.warning(
                f"on_finalized_block took {_wd_elapsed:.1f}s for block {block_number} "
                f"(>12s; main loop is falling behind chain)"
            )

    def _schedule_weight_update(self) -> None:
        """Submit the weight update derived from the just-closed epoch."""

        model_budgets = getattr(self, "_last_model_emission_budgets", {})
        model_bucket_mode = bool(model_budgets)
        model_unallocated = 0.0
        probation_entries = {
            (str(address).lower(), int(model_index))
            for address, model_indices in self._db.get_probation_addresses().items()
            for model_index in model_indices
        }
        endpoint_policy_entries = {
            (str(address).lower(), int(model_index))
            for address, model_index in self._epoch_close_value(
                "_endpoint_policy_gate_reasons",
                getattr(self, "_endpoint_policy_gate_reasons", {}) or {},
            )
        }
        excluded_entries = probation_entries | endpoint_policy_entries
        if model_bucket_mode:
            weights, model_unallocated = self.scorer.get_model_bucket_weights(
                model_budgets,
                model_groups=getattr(self, "_last_model_emission_groups", {}),
                group_budgets=getattr(self, "_last_model_group_budgets", {}),
                excluded_entries=excluded_entries,
            )
        else:
            weights = self.scorer.get_weights(
                excluded_entries=excluded_entries,
            )

        if not weights and not (model_bucket_mode and model_unallocated > 0):
            return
        scoring = self._epoch_close_value(
            "_scoring",
            getattr(self, "_scoring", ScoringParams()),
        )
        # An asynchronous close may overlap the next epoch's runtime-config
        # refresh.  The weight vector must retain the closing epoch's signed
        # emission policy, not whichever policy is active when its owner
        # snapshot finally arrives.
        emission_burn = max(0.0, min(1.0, scoring.emission_burn))
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
            # Zero out blacklisted miners before weight normalization.
            if self._blacklisted_uids:
                for uid in self._blacklisted_uids:
                    if uid in weights:
                        weights[uid] = 0.0
                total = sum(weights.values())
                if total > 0:
                    weights = {uid: score / total for uid, score in weights.items()}
            if emission_burn > 0:
                for uid in list(weights.keys()):
                    weights[uid] *= 1.0 - emission_burn
                weights[self._burn_uid] = (
                    weights.get(self._burn_uid, 0.0) + emission_burn
                )
                bt.logging.info(
                    f"Emission burn: {emission_burn:.0%} to UID {self._burn_uid}"
                )

        def _set_weights_with_retry(_weights=dict(weights)):
            try:
                for attempt in range(1, 4):
                    try:
                        self._set_weights(_weights)
                        self._last_weights = _weights
                        return
                    except Exception as exc:
                        if attempt == 3:
                            bt.logging.error(
                                f"set_weights failed after 3 attempts: {exc}"
                            )
                        else:
                            delay = 30 * (2 ** (attempt - 1))
                            bt.logging.warning(
                                f"set_weights attempt {attempt}/3 failed: {exc} "
                                f"— retrying in {delay}s"
                            )
                            time.sleep(delay)
            finally:
                with self._weight_update_lock:
                    self._weight_updates_pending = max(
                        0,
                        int(self._weight_updates_pending) - 1,
                    )
                auto_updater = getattr(self, "_auto_updater", None)
                if auto_updater is not None:
                    auto_updater.notify_not_busy()

        with self._weight_update_lock:
            self._weight_updates_pending += 1
        try:
            self._weight_update_executor.submit(_set_weights_with_retry)
        except Exception:
            with self._weight_update_lock:
                self._weight_updates_pending = max(
                    0,
                    int(self._weight_updates_pending) - 1,
                )
            raise

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def _auto_update_busy(self) -> bool:
        """Return whether restarting could destroy live validator work."""

        if bool(getattr(self, "_epoch_setup_in_progress", False)):
            return True
        weight_lock = getattr(self, "_weight_update_lock", None)
        if weight_lock is None:
            if bool(getattr(self, "_weight_updates_pending", 0)):
                return True
        else:
            with weight_lock:
                if int(getattr(self, "_weight_updates_pending", 0)) > 0:
                    return True
        try:
            close_futures = tuple(
                getattr(self, "_epoch_close_futures", {}).values()
            )
        except RuntimeError:
            return True
        for future in close_futures:
            if future is None:
                return True
            done = getattr(future, "done", None)
            if not callable(done) or not done():
                return True
        try:
            inflight = tuple(
                getattr(self, "_inflight_canaries", {}).values()
            ) + tuple(
                getattr(self, "_closing_inflight_canaries", {}).values()
            )
            if any(bool(entries) for entries in inflight):
                return True
        except RuntimeError:
            return True
        try:
            queued = tuple(
                getattr(self, "_queued_canaries", {}).values()
            )
            if any(bool(entries) for entries in queued):
                return True
        except RuntimeError:
            return True
        return False

    def _start_new_epoch(self, epoch_start_block: int):
        """Start a new epoch — non-blocking; heavy setup runs on executor."""
        epoch_number = epoch_start_block // self.config.epoch_blocks
        self._refresh_subnet_runtime_config(current_epoch=epoch_number, force=True)
        epoch_number = epoch_start_block // self.config.epoch_blocks
        if getattr(
            self,
            "_proof_v3_verdict_source_latched_epoch",
            None,
        ) != int(epoch_number):
            self._latch_proof_v3_verdict_source(epoch_number)
        if epoch_number != self._current_epoch:
            self._reset_canary_executor()
        self._current_epoch = epoch_number
        self._epoch_start_block = epoch_start_block

        # Canary execution ends at the epoch boundary. Receipt pulling and
        # scoring happen immediately; grace is never used to finish tests.
        self._pending_epoch_close = epoch_number
        self._epoch_close_block = (
            epoch_start_block + self.config.epoch_blocks
        )
        sealed = getattr(self, "_sealed_canary_epochs", set())
        self._sealed_canary_epochs = {
            value for value in sealed if value >= epoch_number
        }

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
                auto_updater = getattr(self, "_auto_updater", None)
                if auto_updater is not None:
                    auto_updater.notify_not_busy()

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
        return bool(
            self._running
            and epoch_number == self._current_epoch
            and epoch_number
            not in getattr(self, "_sealed_canary_epochs", set())
        )

    def _filter_stale_canary_dispatches(
        self,
        pending: List[CanaryTest],
        *,
        processed_block: int,
    ) -> List[CanaryTest]:
        """Neutralize obligations reached only through historical catch-up."""

        if not pending:
            return pending
        observe_head = getattr(
            self,
            "_observe_live_capacity_audit_head",
            None,
        )
        if not callable(observe_head):
            return pending
        live_head, trustworthy = observe_head()
        if not trustworthy:
            for test in pending:
                self._neutralize_canary_obligation(test)
            bt.logging.warning(
                "Neutralized "
                f"{len(pending)} canary obligation(s) because live chronology "
                f"was unavailable at processed block {processed_block}"
            )
            return []
        oldest_allowed = int(live_head) - 1
        expired = [
            test
            for test in pending
            if int(test.target_block) < oldest_allowed
        ]
        if not expired:
            return pending
        for test in expired:
            self._neutralize_canary_obligation(test)
        bt.logging.warning(
            "Neutralized "
            f"{len(expired)} stale canary obligation(s) during block catch-up: "
            f"processed={processed_block} live_head={live_head}"
        )
        expired_ids = {id(test) for test in expired}
        return [test for test in pending if id(test) not in expired_ids]

    @staticmethod
    def _canary_execution_id(
        test: CanaryTest,
        epoch_number: int,
    ) -> Tuple[int, bytes]:
        try:
            obligation_id = bytes.fromhex(str(test.obligation_id))
        except ValueError as exc:
            raise _ProofV3ValidatorConfigurationError(
                "canary execution obligation id is malformed"
            ) from exc
        if len(obligation_id) != 16:
            raise _ProofV3ValidatorConfigurationError(
                "canary execution obligation id is malformed"
            )
        return int(epoch_number), obligation_id

    def _canary_execution_active(
        self,
        test: CanaryTest,
        epoch_number: int,
    ) -> bool:
        """Allow an already-started request to reach a terminal outcome."""

        if self._canary_epoch_active(epoch_number):
            return True
        if not self._running:
            return False
        return self._canary_execution_id(test, epoch_number) in getattr(
            self,
            "_cross_epoch_canaries",
            set(),
        )

    def _mark_cross_epoch_canary_started(
        self,
        test: CanaryTest,
        epoch_number: int,
    ) -> None:
        active = getattr(self, "_cross_epoch_canaries", None)
        if active is None:
            active = set()
            self._cross_epoch_canaries = active
        active.add(
            self._canary_execution_id(test, epoch_number)
        )

    def _mark_cross_epoch_canary_finished(
        self,
        test: CanaryTest,
        epoch_number: int,
    ) -> None:
        getattr(self, "_cross_epoch_canaries", set()).discard(
            self._canary_execution_id(test, epoch_number)
        )

    def _pending_cross_epoch_full_obligations(
        self,
        epoch_number: int,
        key: Tuple[str, int],
    ) -> Set[bytes]:
        """Return started full obligations still resolving after the boundary."""

        canonical_key = self._miner_model_key(key[0], key[1])
        active = getattr(self, "_cross_epoch_canaries", set())
        pending: Set[bytes] = set()
        for execution_id, test in tuple(
            getattr(self, "_unfinished_canary_tests", {}).items()
        ):
            if (
                execution_id[0] != int(epoch_number)
                or execution_id not in active
                or test.test_type != "full_context"
                or self._miner_model_key(
                    test.miner_address,
                    test.model_index,
                )
                != canonical_key
            ):
                continue
            pending.add(execution_id[1])
        return pending

    def _pending_cross_epoch_hard_obligations(
        self,
        epoch_number: int,
        key: Tuple[str, int],
    ) -> Set[bytes]:
        """Return hard requests from one source epoch still terminating."""

        canonical_key = self._miner_model_key(key[0], key[1])
        active = getattr(self, "_cross_epoch_canaries", set())
        pending: Set[bytes] = set()
        for execution_id, test in tuple(
            getattr(self, "_unfinished_canary_tests", {}).items()
        ):
            if (
                execution_id[0] != int(epoch_number)
                or execution_id not in active
                or not bool(test.verify_proof)
                or self._miner_model_key(
                    test.miner_address,
                    test.model_index,
                )
                != canonical_key
            ):
                continue
            pending.add(execution_id[1])
        return pending

    def _complete_cross_epoch_full_success(
        self,
        test: CanaryTest,
        epoch_number: int,
    ) -> None:
        """Clear only debt predating a successfully completed late full test."""

        if test.test_type != "full_context":
            return
        key = self._miner_model_key(
            test.miner_address,
            test.model_index,
        )
        debt_epoch = self._full_context_debt.get(key)
        if debt_epoch is None or int(debt_epoch) >= int(epoch_number):
            return
        self._full_context_debt.pop(key, None)
        self._save_full_context_debt()

    @staticmethod
    def _miner_model_key(address: str, model_index: int) -> Tuple[str, int]:
        """Canonical in-memory key for per-epoch miner/model accounting."""
        return (str(address).lower(), int(model_index))

    def _proof_v3_hard_execution_lock(
        self,
        test: CanaryTest,
    ) -> threading.Lock:
        """Return the endpoint-local lock used only by hard canaries."""

        key = self._miner_model_key(
            test.miner_address,
            test.model_index,
        )
        guard = getattr(
            self,
            "_proof_v3_hard_execution_locks_lock",
            None,
        )
        if guard is None:
            guard = threading.Lock()
            self._proof_v3_hard_execution_locks_lock = guard
        with guard:
            locks = getattr(
                self,
                "_proof_v3_hard_execution_locks",
                None,
            )
            if locks is None:
                locks = {}
                self._proof_v3_hard_execution_locks = locks
            lock = locks.get(key)
            if lock is None:
                lock = threading.Lock()
                locks[key] = lock
            return lock

    def _mark_canary_started(self, epoch_number: int, key: Tuple[str, int]) -> None:
        """Track in-flight work without changing the planned obligation set."""
        key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_canary_accounting_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._canary_accounting_lock = lock
        with lock:
            epoch_inflight = self._inflight_canaries.setdefault(
                epoch_number,
                {},
            )
            epoch_inflight[key] = epoch_inflight.get(key, 0) + 1

    def _mark_canary_finished(
        self,
        epoch_number: int,
        key: Tuple[str, int],
        test: CanaryTest | None = None,
    ) -> None:
        """Clear in-flight accounting when a canary finishes or is abandoned."""
        key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_canary_accounting_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._canary_accounting_lock = lock
        with lock:
            epoch_inflight = self._inflight_canaries.get(epoch_number)
            if not epoch_inflight:
                if test is not None:
                    getattr(
                        self,
                        "_unfinished_canary_tests",
                        {},
                    ).pop(
                        self._canary_execution_id(test, epoch_number),
                        None,
                    )
                return
            current = epoch_inflight.get(key, 0)
            if current <= 1:
                epoch_inflight.pop(key, None)
            else:
                epoch_inflight[key] = current - 1
            if not epoch_inflight:
                self._inflight_canaries.pop(epoch_number, None)
            if test is not None:
                getattr(
                    self,
                    "_unfinished_canary_tests",
                    {},
                ).pop(
                    self._canary_execution_id(test, epoch_number),
                    None,
                )

    def _mark_canary_queued(
        self,
        epoch_number: int,
        key: Tuple[str, int],
        test: CanaryTest | None = None,
    ) -> None:
        """Track submitted work before an executor worker begins it."""

        key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_canary_accounting_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._canary_accounting_lock = lock
        with lock:
            queued_by_epoch = getattr(self, "_queued_canaries", None)
            if queued_by_epoch is None:
                queued_by_epoch = {}
                self._queued_canaries = queued_by_epoch
            queued = queued_by_epoch.setdefault(epoch_number, {})
            queued[key] = queued.get(key, 0) + 1
            if test is not None:
                unfinished = getattr(
                    self,
                    "_unfinished_canary_tests",
                    None,
                )
                if unfinished is None:
                    unfinished = {}
                    self._unfinished_canary_tests = unfinished
                unfinished[
                    self._canary_execution_id(test, epoch_number)
                ] = test

    def _mark_canary_dequeued(
        self,
        epoch_number: int,
        key: Tuple[str, int],
    ) -> None:
        """Clear one queued marker when its worker starts."""

        key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_canary_accounting_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._canary_accounting_lock = lock
        with lock:
            queued_by_epoch = getattr(self, "_queued_canaries", {})
            queued = queued_by_epoch.get(epoch_number)
            if not queued:
                return
            current = queued.get(key, 0)
            if current <= 1:
                queued.pop(key, None)
            else:
                queued[key] = current - 1
            if not queued:
                queued_by_epoch.pop(epoch_number, None)

    def _seal_canary_epoch_for_close(self, epoch_number: int) -> Set[Tuple[str, int]]:
        """Fence local work without erasing miner-visible terminal outcomes."""

        sealed = getattr(self, "_sealed_canary_epochs", None)
        if sealed is None:
            sealed = set()
            self._sealed_canary_epochs = sealed
        if epoch_number in sealed:
            return set()
        sealed.add(epoch_number)

        affected: Set[Tuple[str, int]] = set()
        local_unstarted: Dict[Tuple[int, bytes], CanaryTest] = {}
        miner_rejected_pending: Dict[Tuple[int, bytes], CanaryTest] = {}
        started_inflight: Dict[Tuple[int, bytes], CanaryTest] = {}
        scheduler_lock = getattr(self, "_canary_scheduler_lock", None)
        scheduler = getattr(self, "_canary_scheduler", None)
        if (
            scheduler is not None
            and scheduler.epoch_number == epoch_number
        ):
            if scheduler_lock is None:
                pending = list(scheduler.tests)
                scheduler.tests = []
            else:
                with scheduler_lock:
                    pending = list(scheduler.tests)
                    scheduler.tests = []
            for test in pending:
                execution_id = self._canary_execution_id(
                    test,
                    epoch_number,
                )
                if tuple(getattr(test, "rejection_intervals", ()) or ()):
                    miner_rejected_pending[execution_id] = test
                else:
                    local_unstarted[execution_id] = test
                affected.add(
                    self._miner_model_key(
                        test.miner_address,
                        test.model_index,
                    )
                )

        lock = getattr(self, "_canary_accounting_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._canary_accounting_lock = lock
        with lock:
            cross_epoch = getattr(self, "_cross_epoch_canaries", set())
            for execution_id, test in tuple(
                getattr(self, "_unfinished_canary_tests", {}).items()
            ):
                if execution_id[0] == epoch_number:
                    if execution_id in cross_epoch:
                        started_inflight[execution_id] = test
                    elif tuple(
                        getattr(test, "rejection_intervals", ()) or ()
                    ):
                        miner_rejected_pending[execution_id] = test
                    else:
                        local_unstarted[execution_id] = test
                    affected.add(
                        self._miner_model_key(
                            test.miner_address,
                            test.model_index,
                        )
                    )
            affected.update(
                getattr(self, "_queued_canaries", {}).pop(
                    epoch_number,
                    {},
                ).keys()
            )
            affected.update(
                self._inflight_canaries.pop(epoch_number, {}).keys()
            )

        # Never-started local work and requests still running at the boundary
        # are removed from immutable source-epoch throughput accounting. A
        # miner-observed 503 is different: even when its retry is pending, the
        # exact obligation remains in the inventory so low work fails and full
        # work can use only the signed one-shot busy deferral. A running request
        # remains active until its configured deadline and produces either a
        # late success or an immediate miner-attributed failure.
        for test in (
            list(local_unstarted.values())
            + list(started_inflight.values())
        ):
            self._neutralize_canary_obligation(test)
        with lock:
            for execution_id in (
                set(local_unstarted) | set(miner_rejected_pending)
            ):
                self._unfinished_canary_tests.pop(execution_id, None)

        if affected:
            bt.logging.warning(
                f"Epoch {epoch_number}: neutralized "
                f"{len(local_unstarted)} unstarted validator canary task(s) "
                f"and retained {len(miner_rejected_pending)} "
                "miner-rejected obligation(s) for reconciliation; "
                f"retained {len(started_inflight)} started request(s) "
                "to a bounded terminal outcome"
            )
        return affected

    def _neutralize_canary_obligation(self, test: CanaryTest) -> None:
        """Remove one validator-unfinished obligation from score accounting."""

        key = self._miner_model_key(
            test.miner_address,
            test.model_index,
        )
        try:
            obligation_id = bytes.fromhex(test.obligation_id)
        except ValueError:
            return
        inventory = self._expected_canary_obligations.get(key)
        if inventory is None:
            return
        inventory.pop(obligation_id, None)
        getattr(
            self,
            "_hard_canary_obligation_ids",
            set(),
        ).discard(obligation_id)
        if inventory:
            self._expected_receipts[key] = len(inventory)
        else:
            self._expected_canary_obligations.pop(key, None)
            self._expected_receipts.pop(key, None)

    def _decrement_expected_receipt(self, epoch_number: int, key: Tuple[str, int]) -> None:
        """Compatibility no-op: planned 2+1 obligations are never weakened."""

        return None

    def _effective_expected_receipts(self, epoch_number: int, key: Tuple[str, int]) -> int:
        """Return the exact preplanned obligation count for an epoch."""
        key = self._miner_model_key(key[0], key[1])
        expected = self._epoch_close_value("_expected_receipts", {})
        return max(0, expected.get(key, 0))

    @staticmethod
    def _canary_debt_key(key: Tuple[str, int]) -> str:
        return f"{str(key[0]).lower()}|{int(key[1])}"

    def _load_full_context_debt(self) -> None:
        self._full_context_debt = {}
        try:
            raw = self._db.get_meta("proof_v3_full_context_debt_v1")
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                return
            for encoded, epoch in value.items():
                if not isinstance(encoded, str) or "|" not in encoded:
                    continue
                address, index = encoded.rsplit("|", 1)
                if (
                    not address
                    or type(epoch) is not int
                    or epoch < 0
                ):
                    continue
                self._full_context_debt[
                    self._miner_model_key(address, int(index))
                ] = epoch
        except Exception as exc:
            bt.logging.warning(
                f"Ignoring malformed proof-v3 full-context debt state: {exc}"
            )
            self._full_context_debt = {}

    def _save_full_context_debt(self) -> None:
        value = {
            self._canary_debt_key(key): int(epoch)
            for key, epoch in sorted(self._full_context_debt.items())
        }
        self._db.set_meta(
            "proof_v3_full_context_debt_v1",
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        )

    def _migrate_full_context_debt_for_registrations(
        self,
        previous_miners: Sequence[ActiveMiner],
        current_miners: Sequence[ActiveMiner],
    ) -> None:
        """Carry one-shot busy debt across model-index re-registration.

        A contract model index is not a stable service identity.  A miner must
        not regain its one authenticated-busy deferral merely by releasing and
        re-registering the same model under a new index.  We conservatively
        carry debt to every current entry for the same EVM address and model.
        Hard-pass credit is intentionally not migrated: a new entry must earn
        its own hidden hard pass.
        """

        if not self._full_context_debt:
            return
        prior_identity_by_key = {
            self._miner_model_key(miner.address, miner.model_index): (
                str(miner.address).lower(),
                str(miner.model_id),
            )
            for miner in previous_miners
        }
        current_by_identity: Dict[
            Tuple[str, str],
            List[Tuple[str, int]],
        ] = {}
        for miner in current_miners:
            identity = (
                str(miner.address).lower(),
                str(miner.model_id),
            )
            current_by_identity.setdefault(identity, []).append(
                self._miner_model_key(
                    miner.address,
                    miner.model_index,
                )
            )

        migrated = 0
        for old_key, debt_epoch in tuple(self._full_context_debt.items()):
            identity = prior_identity_by_key.get(old_key)
            if identity is None:
                continue
            for new_key in current_by_identity.get(identity, ()):
                if new_key in self._full_context_debt:
                    continue
                self._full_context_debt[new_key] = int(debt_epoch)
                migrated += 1
        if migrated:
            self._save_full_context_debt()
            bt.logging.info(
                "Migrated proof-v3 full-context debt to "
                f"{migrated} re-registered model entry/entries"
            )

    def _load_hard_audit_pass_epochs(self) -> None:
        self._last_hard_audit_pass_epoch = {}
        try:
            raw = self._db.get_meta("proof_v3_hard_audit_pass_epochs_v1")
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                return
            for encoded, epoch in value.items():
                if (
                    not isinstance(encoded, str)
                    or "|" not in encoded
                    or type(epoch) is not int
                    or epoch < 0
                ):
                    continue
                address, index = encoded.rsplit("|", 1)
                self._last_hard_audit_pass_epoch[
                    self._miner_model_key(address, int(index))
                ] = epoch
        except Exception as exc:
            bt.logging.warning(
                "Ignoring malformed proof-v3 hard-audit pass ledger: "
                f"{exc}"
            )
            self._last_hard_audit_pass_epoch = {}

    def _hard_failure_policy(self) -> ProofV3FailurePolicyConfig:
        policy = self._epoch_close_value(
            "_proof_v3_failure_policy_cfg",
            None,
        )
        if isinstance(policy, ProofV3FailurePolicyConfig):
            return policy
        return proof_v3_failure_policy_config_from_neuron_config(
            self.config
        )

    def _probation_reset_covers_source_epoch(self, source_epoch: int) -> bool:
        reset_epoch = getattr(
            self,
            "_probation_state_reset_effective_epoch",
            None,
        )
        return reset_epoch is not None and int(source_epoch) < int(reset_epoch)

    def _apply_probation_state_generation(
        self,
        policy: ProofV3FailurePolicyConfig,
        *,
        effective_epoch: int | None,
    ) -> bool:
        """Apply an owner-hosted probation reset once per generation."""

        target = int(policy.probation_state_generation)
        raw_current = self._db.get_meta(
            "proof_v3_probation_state_generation_v1"
        )
        current = int(raw_current or 0)
        if target < current:
            bt.logging.warning(
                "Ignoring probation state generation rollback "
                f"{current}->{target}"
            )
            return False
        if target == current:
            return False
        reset_epoch = int(
            effective_epoch
            if effective_epoch is not None
            else getattr(self, "_current_epoch", 0)
        )

        # The sidecar file is cleared before the DB transaction.  If the
        # process stops between them, the persisted generation remains old and
        # the operation safely retries on restart.  The DB transaction then
        # clears operational state and records the new generation atomically.
        tracker_entries = self._probation_tracker.clear_all()
        empty_strikes = HardProofStrikeTracker(
            failure_epochs_for_penalty=policy.failure_epochs_for_penalty,
            clean_hard_audit_epochs_for_reset=(
                policy.clean_hard_audit_epochs_for_reset
            ),
        )
        result = self._db.reset_operational_probation_state(
            target,
            effective_epoch=reset_epoch,
            hard_failure_strikes_json=empty_strikes.to_json(),
        )
        if not result.get("applied"):
            return False
        with self._hard_failure_strike_lock:
            self._hard_failure_strikes = empty_strikes
        with self._probation_recovery_epoch_lock:
            self._last_probation_recovery_source_epoch = {}
        self._probation_state_reset_effective_epoch = reset_epoch
        self._write_shared_state()
        bt.logging.info(
            "Applied probation state generation "
            f"{current}->{target} effective_epoch={reset_epoch}: "
            f"tracker={tracker_entries} "
            f"db={result['probation_entries_cleared']} "
            f"pending_capacity={result['capacity_events_consumed']}"
        )
        return True

    def _load_hard_failure_strikes(self) -> None:
        try:
            raw = self._db.get_meta("proof_v3_hard_failure_strikes_v1")
            if not raw:
                return
            policy = self._proof_v3_failure_policy_cfg
            self._hard_failure_strikes = HardProofStrikeTracker.from_json(
                raw,
                failure_epochs_for_penalty=(
                    policy.failure_epochs_for_penalty
                ),
                clean_hard_audit_epochs_for_reset=(
                    policy.clean_hard_audit_epochs_for_reset
                ),
            )
        except Exception as exc:
            # A malformed local ledger must not grant a neutral failure.
            bt.logging.warning(
                "Ignoring malformed proof-v3 hard-failure strike ledger; "
                f"falling back to immediate penalty: {exc}"
            )
            self._hard_failure_strikes = HardProofStrikeTracker(
                failure_epochs_for_penalty=1,
                clean_hard_audit_epochs_for_reset=3,
            )

    def _save_hard_failure_strikes_locked(self) -> None:
        self._db.set_meta(
            "proof_v3_hard_failure_strikes_v1",
            self._hard_failure_strikes.to_json(),
        )

    def _record_hard_failure_strike(
        self,
        key: Tuple[str, int],
        *,
        source_epoch: int,
        endpoint: str = "",
        force_penalty: bool = False,
    ) -> bool:
        """Return whether this failed source epoch carries a consequence."""

        if self._probation_reset_covers_source_epoch(source_epoch):
            bt.logging.info(
                "Ignoring pre-reset proof-v3 hard-failure consequence for "
                f"{key[0][:10]} model_index={key[1]} "
                f"source_epoch={int(source_epoch)}"
            )
            return False

        policy = self._hard_failure_policy()
        with self._hard_failure_strike_lock:
            self._hard_failure_strikes.configure(
                failure_epochs_for_penalty=(
                    policy.failure_epochs_for_penalty
                ),
                clean_hard_audit_epochs_for_reset=(
                    policy.clean_hard_audit_epochs_for_reset
                ),
            )
            penalty = self._hard_failure_strikes.record_failure(
                key,
                int(source_epoch),
                endpoint=endpoint,
                # The new epoch counter is independent of historical
                # probation.  A follower may force the consequence only when
                # it consumes an owner snapshot whose negative verdict means
                # the owner's configured threshold was already reached.
                already_on_probation=force_penalty,
            )
            self._save_hard_failure_strikes_locked()
        if not penalty:
            bt.logging.info(
                "Proof-v3 hard failure retained as neutral first strike for "
                f"{key[0][:10]} model_index={key[1]} "
                f"source_epoch={int(source_epoch)}; "
                f"threshold={policy.failure_epochs_for_penalty} "
                "(score/EMA/probation/proxy unchanged)"
            )
        return penalty

    def _record_hard_failure_clean_pass(
        self,
        key: Tuple[str, int],
        *,
        source_epoch: int,
    ) -> bool:
        policy = self._hard_failure_policy()
        with self._hard_failure_strike_lock:
            self._hard_failure_strikes.configure(
                failure_epochs_for_penalty=(
                    policy.failure_epochs_for_penalty
                ),
                clean_hard_audit_epochs_for_reset=(
                    policy.clean_hard_audit_epochs_for_reset
                ),
            )
            cleared = self._hard_failure_strikes.record_clean_pass(
                key,
                int(source_epoch),
            )
            self._save_hard_failure_strikes_locked()
        if cleared:
            bt.logging.info(
                "Proof-v3 pending hard-failure strike cleared for "
                f"{key[0][:10]} model_index={key[1]} after "
                f"{policy.clean_hard_audit_epochs_for_reset} clean hard "
                "audit epoch(s)"
            )
        return cleared

    def _hard_failure_penalty_required(
        self,
        key: Tuple[str, int],
    ) -> bool:
        with self._hard_failure_strike_lock:
            return self._hard_failure_strikes.penalty_required(key)

    def _clear_hard_failure_strike(
        self,
        key: Tuple[str, int],
    ) -> None:
        with self._hard_failure_strike_lock:
            if self._hard_failure_strikes.clear(key):
                self._save_hard_failure_strikes_locked()

    def _save_hard_audit_pass_epochs(self) -> None:
        value = {
            self._canary_debt_key(key): int(epoch)
            for key, epoch in sorted(
                self._last_hard_audit_pass_epoch.items()
            )
        }
        self._db.set_meta(
            "proof_v3_hard_audit_pass_epochs_v1",
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        )

    def _record_hard_audit_pass(
        self,
        test: CanaryTest,
        *,
        completion_epoch: int,
        source_epoch: int | None = None,
    ) -> None:
        if not bool(test.verify_proof):
            return
        key = self._miner_model_key(
            test.miner_address,
            test.model_index,
        )
        self._record_hard_failure_clean_pass(
            key,
            source_epoch=(
                int(completion_epoch)
                if source_epoch is None
                else int(source_epoch)
            ),
        )
        lock = getattr(self, "_hard_audit_pass_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._hard_audit_pass_lock = lock
        with lock:
            prior = self._last_hard_audit_pass_epoch.get(key, -1)
            if int(completion_epoch) <= prior:
                return
            self._last_hard_audit_pass_epoch[key] = int(completion_epoch)
            self._save_hard_audit_pass_epochs()

    def _load_probation_recovery_source_epochs(self) -> None:
        self._last_probation_recovery_source_epoch = {}
        try:
            raw = self._db.get_meta(
                "proof_v3_probation_recovery_source_epochs_v1"
            )
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                return
            for encoded, epoch in value.items():
                if (
                    not isinstance(encoded, str)
                    or "|" not in encoded
                    or type(epoch) is not int
                    or epoch < 0
                ):
                    continue
                address, index = encoded.rsplit("|", 1)
                self._last_probation_recovery_source_epoch[
                    self._miner_model_key(address, int(index))
                ] = epoch
        except Exception as exc:
            bt.logging.warning(
                "Ignoring malformed proof-v3 probation recovery ledger: "
                f"{exc}"
            )
            self._last_probation_recovery_source_epoch = {}

    def _save_probation_recovery_source_epochs(self) -> None:
        value = {
            self._canary_debt_key(key): int(epoch)
            for key, epoch in sorted(
                self._last_probation_recovery_source_epoch.items()
            )
        }
        self._db.set_meta(
            "proof_v3_probation_recovery_source_epochs_v1",
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        )

    def _mark_probation_recovery_failure_epoch(
        self,
        key: Tuple[str, int],
        source_epoch: int,
    ) -> None:
        canonical_key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_probation_recovery_epoch_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._probation_recovery_epoch_lock = lock
        with lock:
            prior = self._last_probation_recovery_source_epoch.get(
                canonical_key,
                -1,
            )
            if int(source_epoch) <= prior:
                return
            self._last_probation_recovery_source_epoch[canonical_key] = int(
                source_epoch
            )
            self._save_probation_recovery_source_epochs()

    def _record_probation_recovery_source_pass(
        self,
        key: Tuple[str, int],
        source_epoch: int,
    ) -> bool:
        """Count at most one clean recovery pass per ordered source epoch."""

        canonical_key = self._miner_model_key(key[0], key[1])
        lock = getattr(self, "_probation_recovery_epoch_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._probation_recovery_epoch_lock = lock
        with lock:
            prior = self._last_probation_recovery_source_epoch.get(
                canonical_key,
                -1,
            )
            if int(source_epoch) <= prior:
                return False
            if not self._probation_tracker.is_on_probation(canonical_key):
                return False
            self._last_probation_recovery_source_epoch[canonical_key] = int(
                source_epoch
            )
            self._save_probation_recovery_source_epochs()
            lifted = self._probation_tracker.record_pass(canonical_key)
            self._db.record_pass(canonical_key[0], canonical_key[1])
            if lifted:
                self._clear_hard_failure_strike(canonical_key)
            return True

    @staticmethod
    def _probation_recovery_epoch_is_clean(
        *,
        obligation_failure: bool,
        full_deferred: bool,
        proof_tests: int,
        proof_failures: int,
    ) -> bool:
        """Require an actual successful hard proof for probation recovery."""

        return bool(
            not obligation_failure
            and not full_deferred
            and int(proof_tests) > 0
            and int(proof_failures) == 0
        )

    def _reconcile_late_hard_probation_pass(
        self,
        test: CanaryTest,
        source_epoch: int,
    ) -> bool:
        """Apply one clean late source epoch after its final hard terminates."""

        if not bool(test.verify_proof):
            return False
        key = self._miner_model_key(
            test.miner_address,
            test.model_index,
        )
        current_obligation = bytes.fromhex(test.obligation_id)
        if self._pending_cross_epoch_hard_obligations(
            source_epoch,
            key,
        ) != {current_obligation}:
            return False
        failures = self._db.get_proof_v3_hard_failures(
            int(source_epoch),
            int(source_epoch),
        )
        if any(
            self._miner_model_key(
                item.get("miner_address", ""),
                item.get("model_index", -1),
            )
            == key
            for item in failures
        ):
            self._mark_probation_recovery_failure_epoch(key, source_epoch)
            return False
        recorded = self._record_probation_recovery_source_pass(
            key,
            source_epoch,
        )
        if recorded:
            bt.logging.info(
                "Probation late hard-audit pass reconciled for "
                f"{test.miner_address[:10]} model_index={test.model_index} "
                f"source_epoch={source_epoch}"
            )
        return recorded

    def _record_hard_audit_failure(
        self,
        *,
        source_epoch: int,
        miner_address: str,
        model_id: str,
        model_index: int,
        obligation_id: bytes,
        failure_code: str,
        endpoint: str = "",
    ) -> bool:
        """Persist the real failure and return whether policy applies penalty."""

        if not _proof_v3_hard_auditor_active(
            self.config,
            self._validator_hotkey_ss58,
        ):
            # A validator that cannot publish the designated owner's outcome
            # must never manufacture a free neutral strike.
            return True
        key = self._miner_model_key(miner_address, model_index)
        penalty_required = self._record_hard_failure_strike(
            key,
            source_epoch=int(source_epoch),
            endpoint=endpoint,
        )
        failure_key = (int(source_epoch), bytes(obligation_id))
        lock = getattr(self, "_hard_audit_failure_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._hard_audit_failure_lock = lock
        with lock:
            recorded = getattr(
                self,
                "_hard_audit_failure_obligations",
                None,
            )
            if recorded is None:
                recorded = set()
                self._hard_audit_failure_obligations = recorded
            if failure_key in recorded:
                if penalty_required:
                    self._mark_probation_recovery_failure_epoch(
                        key,
                        source_epoch,
                    )
                return penalty_required
            try:
                from neurons.proof_v3_hard_outcome import (
                    HardAuditFailureV3,
                    sign_hard_audit_failure_v3,
                )

                outcome = sign_hard_audit_failure_v3(
                    HardAuditFailureV3(
                        source_epoch=int(source_epoch),
                        miner_address=str(miner_address).lower(),
                        model_id=str(model_id),
                        model_index=int(model_index),
                        obligation_id=bytes(obligation_id),
                        failure_code=str(failure_code),
                        observed_at=int(time.time()),
                        validator_hotkey=self._validator_hotkey_bytes,
                    ),
                    self._validator_private_key,
                )
                self._db.store_proof_v3_hard_failure(
                    outcome.to_dict(),
                    outcome.digest(),
                )
                recorded.add(failure_key)
                if penalty_required:
                    self._mark_probation_recovery_failure_epoch(
                        key,
                        source_epoch,
                    )
                return penalty_required
            except Exception as exc:
                # Signing/persistence belongs to the validator. Never turn it
                # into miner evidence or let it suppress local enforcement.
                bt.logging.warning(
                    "Proof-v3 hard failure publication failed locally "
                    f"(NOT attributed to miner {str(miner_address)[:10]}): "
                    f"{type(exc).__name__}: {exc}"
                )
                if self._canary_epoch_active(int(source_epoch)):
                    self._validator_canary_failures.add(
                        self._miner_model_key(
                            miner_address,
                            model_index,
                        )
                    )
                return penalty_required

    def _hard_audit_due_entries(
        self,
        *,
        epoch_number: int,
        drought_epochs: int,
    ) -> Set[Tuple[str, int]]:
        due: Set[Tuple[str, int]] = set()
        with self._hard_audit_pass_lock:
            for miner in self._epoch_miners:
                key = self._miner_model_key(
                    miner.address,
                    miner.model_index,
                )
                last = self._last_hard_audit_pass_epoch.get(key)
                if last is None or int(epoch_number) - last >= drought_epochs:
                    due.add(key)
        return due

    def _load_shared_hard_processed_receipts(self) -> None:
        self._shared_hard_processed_receipts = set()
        try:
            raw = self._db.get_meta(
                "proof_v3_shared_hard_processed_receipts_v1"
            )
            values = json.loads(raw) if raw else []
            if not isinstance(values, list):
                return
            for value in values:
                if not isinstance(value, str) or len(value) != 64:
                    continue
                try:
                    decoded = bytes.fromhex(value)
                except ValueError:
                    continue
                if len(decoded) == 32:
                    self._shared_hard_processed_receipts.add(decoded)
        except Exception as exc:
            bt.logging.warning(
                "Ignoring malformed shared hard-receipt replay ledger: "
                f"{exc}"
            )
            self._shared_hard_processed_receipts = set()

    def _save_shared_hard_processed_receipts(self) -> None:
        # Three retained epochs produce only a handful of records per
        # endpoint. Keep a generous bounded tail so restart cannot replay old
        # pass/failure verdicts indefinitely.
        values = sorted(self._shared_hard_processed_receipts)
        if len(values) > 16_384:
            values = values[-16_384:]
            self._shared_hard_processed_receipts = set(values)
        self._db.set_meta(
            "proof_v3_shared_hard_processed_receipts_v1",
            json.dumps(
                [value.hex() for value in values],
                separators=(",", ":"),
            ),
        )

    def _mark_shared_hard_receipt_processed(
        self,
        receipt: ServiceReceipt,
    ) -> None:
        from neurons.proof_v3_shared_hard import (
            shared_hard_receipt_cache_key_v3,
        )

        key = shared_hard_receipt_cache_key_v3(receipt)
        processed = getattr(
            self,
            "_shared_hard_processed_receipts",
            None,
        )
        if processed is None:
            processed = set()
            self._shared_hard_processed_receipts = processed
        if key in processed:
            return
        processed.add(key)
        try:
            self._save_shared_hard_processed_receipts()
        except Exception as exc:
            bt.logging.warning(
                "Shared hard-receipt replay ledger persistence failed "
                f"locally: {exc}"
            )

    def _load_shared_hard_processed_failures(self) -> None:
        self._shared_hard_processed_failures = set()
        try:
            raw = self._db.get_meta(
                "proof_v3_shared_hard_processed_failures_v1"
            )
            values = json.loads(raw) if raw else []
            if not isinstance(values, list):
                return
            for value in values:
                if not isinstance(value, str) or len(value) != 64:
                    continue
                try:
                    decoded = bytes.fromhex(value)
                except ValueError:
                    continue
                if len(decoded) == 32:
                    self._shared_hard_processed_failures.add(decoded)
        except Exception as exc:
            bt.logging.warning(
                "Ignoring malformed shared hard-failure replay ledger: "
                f"{exc}"
            )
            self._shared_hard_processed_failures = set()

    def _save_shared_hard_processed_failures(self) -> None:
        values = sorted(self._shared_hard_processed_failures)
        if len(values) > 16_384:
            values = values[-16_384:]
            self._shared_hard_processed_failures = set(values)
        self._db.set_meta(
            "proof_v3_shared_hard_processed_failures_v1",
            json.dumps(
                [value.hex() for value in values],
                separators=(",", ":"),
            ),
        )

    def _proof_v3_hard_failure_feed_url(self) -> str:
        explicit = str(
            getattr(self.config, "proof_v3_hard_failure_url", "") or ""
        ).strip()
        if explicit:
            parsed = urlparse(explicit)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                return explicit.rstrip("/")
            return ""
        config_url = str(
            getattr(self.config, "subnet_config_url", "") or ""
        ).strip()
        parsed = urlparse(config_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ""
        return (
            f"{parsed.scheme}://{parsed.netloc}"
            "/v1/proof-v3/hard-failures"
        )

    def _fetch_shared_hard_failure_verdicts(
        self,
        *,
        current_epoch: int,
    ) -> Dict[Tuple[str, int], bool]:
        """Consume positive, owner-signed failure evidence.

        Missing/unreachable feeds and invalid records stay neutral. A valid
        failure is sufficient on its own; absence is never interpreted as a
        pass or a failure.
        """

        if self._proof_v3_follower_mode_active():
            return {}

        owner_ss58 = str(
            getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        if (
            not getattr(
                self.config,
                "proof_v3_hard_auditor_policy_enabled",
                False,
            )
            or not owner_ss58
            or owner_ss58 == self._validator_hotkey_ss58
        ):
            return {}
        feed_url = self._proof_v3_hard_failure_feed_url()
        if not feed_url:
            return {}
        minimum_epoch = max(0, int(current_epoch) - 3)
        try:
            response = httpx.get(
                feed_url,
                params={
                    "minimum_epoch": minimum_epoch,
                    "maximum_epoch": int(current_epoch),
                },
                timeout=max(
                    1.0,
                    float(
                        getattr(
                            self.config,
                            "subnet_config_timeout_seconds",
                            5.0,
                        )
                        or 5.0
                    ),
                ),
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            bt.logging.warning(
                "Owner proof-v3 hard-failure feed unavailable; hard "
                f"failure sharing remains neutral: {type(exc).__name__}"
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        rows = payload.get("failures")
        if not isinstance(rows, list) or len(rows) > 16_384:
            return {}

        from neurons.proof_v3_hard_outcome import (
            HardAuditFailureV3,
            verify_hard_audit_failure_v3,
        )

        exact: Dict[Tuple[str, str, int], List[ActiveMiner]] = {}
        identity: Dict[Tuple[str, str], List[ActiveMiner]] = {}
        for miner in self._epoch_close_value("_epoch_miners", ()):
            address = str(miner.address).lower()
            model_id = str(miner.model_id)
            exact.setdefault(
                (address, model_id, int(miner.model_index)),
                [],
            ).append(miner)
            identity.setdefault((address, model_id), []).append(miner)

        verdicts: Dict[Tuple[str, int], bool] = {}
        newly_processed = False
        for value in rows:
            try:
                outcome = HardAuditFailureV3.from_dict(value)
            except (TypeError, ValueError):
                continue
            if not minimum_epoch <= outcome.source_epoch <= int(current_epoch):
                continue
            if not verify_hard_audit_failure_v3(
                outcome,
                expected_validator_hotkey_ss58=owner_ss58,
            ):
                continue
            digest = outcome.digest()
            if digest in self._shared_hard_processed_failures:
                continue
            matches = exact.get(
                (
                    outcome.miner_address.lower(),
                    outcome.model_id,
                    outcome.model_index,
                ),
                [],
            )
            if not matches:
                # Carry failure evidence across model-index re-registration,
                # matching the persisted busy-debt rule.
                matches = identity.get(
                    (
                        outcome.miner_address.lower(),
                        outcome.model_id,
                    ),
                    [],
                )
            if not matches:
                continue
            for miner in matches:
                key = self._miner_model_key(
                    miner.address,
                    miner.model_index,
                )
                if self._record_hard_failure_strike(
                    key,
                    source_epoch=outcome.source_epoch,
                    endpoint=getattr(miner, "endpoint", ""),
                ):
                    verdicts[key] = False
            self._shared_hard_processed_failures.add(digest)
            newly_processed = True
        if newly_processed:
            try:
                self._save_shared_hard_processed_failures()
            except Exception as exc:
                bt.logging.warning(
                    "Shared hard-failure replay ledger persistence failed "
                    f"locally: {exc}"
                )
        return verdicts

    def _register_canary_obligations(
        self,
        tests: Sequence[CanaryTest],
    ) -> None:
        self._expected_receipts = {}
        self._expected_canary_obligations = {}
        self._hard_canary_obligation_ids = set()
        for test in tests:
            key = self._miner_model_key(
                test.miner_address,
                test.model_index,
            )
            try:
                obligation_id = bytes.fromhex(test.obligation_id)
            except ValueError as exc:
                raise RuntimeError("canary obligation id is malformed") from exc
            if len(obligation_id) != 16:
                raise RuntimeError("canary obligation id is malformed")
            kind = "full" if test.test_type == "full_context" else "low"
            inventory = self._expected_canary_obligations.setdefault(key, {})
            if obligation_id in inventory:
                raise RuntimeError("canary obligation id is duplicated")
            inventory[obligation_id] = (
                kind,
                int(test.target_prompt_tokens),
            )
            if (
                bool(test.verify_proof)
                and PROOF_PROTOCOL_V3
                in tuple(test.proof_protocol_versions or ())
                and test.model_id
                in getattr(self, "_proof_v3_releases", {})
            ):
                self._hard_canary_obligation_ids.add(obligation_id)
            self._expected_receipts[key] = len(inventory)
            if kind == "full" and key in self._full_context_debt:
                test.is_full_context_debt = True

    @staticmethod
    def _receipt_completes_canary_obligation(
        receipt: ServiceReceipt,
        obligation_id: bytes,
        expected: Tuple[str, int],
    ) -> bool:
        """Validate one signed receipt against one exact planned obligation."""

        expected_kind, expected_target = expected
        prompt_tokens = int(getattr(receipt, "prompt_tokens", 0) or 0)
        tolerance = canary_prompt_token_tolerance_v3(int(expected_target))
        return bool(
            receipt.is_canary
            and int(getattr(receipt, "receipt_version", 1) or 1) >= 4
            and bytes(
                getattr(receipt, "canary_obligation_id", b"") or b""
            )
            == obligation_id
            and str(getattr(receipt, "canary_kind", "") or "")
            == expected_kind
            and int(
                getattr(receipt, "canary_target_prompt_tokens", 0) or 0
            )
            == int(expected_target)
            and 0 < prompt_tokens <= int(expected_target)
            and int(expected_target) - prompt_tokens <= tolerance
            and (
                (
                    bool(receipt.proof_requested)
                    and bool(receipt.proof_verified)
                )
                or (
                    not bool(receipt.proof_requested)
                    and not bool(receipt.proof_verified)
                )
            )
            and int(receipt.tokens_generated or 0) > 0
        )

    def _completed_canary_obligations(
        self,
        own_receipts: Sequence[ServiceReceipt],
        expected: Mapping[bytes, Tuple[str, int]],
    ) -> Set[bytes]:
        completed: Set[bytes] = set()
        duplicated: Set[bytes] = set()
        for receipt in own_receipts:
            obligation_id = bytes(
                getattr(receipt, "canary_obligation_id", b"") or b""
            )
            item = expected.get(obligation_id)
            if item is None:
                continue
            if obligation_id in completed:
                duplicated.add(obligation_id)
                continue
            if self._receipt_completes_canary_obligation(
                receipt,
                obligation_id,
                item,
            ):
                completed.add(obligation_id)
        return completed.difference(duplicated)

    def _busy_evidence_covers_full_obligations(
        self,
        key: Tuple[str, int],
        missing_full: Set[bytes],
        all_receipts: Sequence[ServiceReceipt],
    ) -> bool:
        """Require authorized observed work overlapping every full 503 window."""

        if not missing_full:
            return False
        observed = [
            interval
            for receipt in all_receipts
            if int(getattr(receipt, "tokens_generated", 0) or 0) > 0
            and (
                interval := receipt_observed_interval(receipt)
            ) is not None
            and interval[1] > interval[0]
        ]
        if not observed:
            return False
        records = self._epoch_close_value(
            "_busy_skip_probations",
            {},
        ).get(key, [])
        for obligation_id in missing_full:
            windows = [
                (start, end)
                for start, end, kind, recorded_id in records
                if kind == "full" and recorded_id == obligation_id
            ]
            if not windows:
                return False
            for start, end in windows:
                if not any(
                    receipt_start <= end and receipt_end >= start
                    for receipt_start, receipt_end in observed
                ):
                    return False
        return True

    @staticmethod
    def _may_defer_full_context_obligations(
        *,
        missing_low: Set[bytes],
        missing_full: Set[bytes],
        prior_full_debt: bool,
        suppress_probation: bool,
        busy_evidence_covers: bool,
    ) -> bool:
        """Permit one authenticated-busy deferral, never a rolling rollover."""

        return bool(
            missing_full
            and not missing_low
            and not prior_full_debt
            and not suppress_probation
            and busy_evidence_covers
        )

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

    def _owner_canary_busy_keys(self) -> Set[Tuple[str, int]]:
        """Return endpoint slots with owner canaries already in flight.

        Capacity selection and canary admission share
        ``_capacity_audit_schedule_lock``.  Reading the exact started
        execution set while that lock is held closes the inverse race where a
        capacity window could be created after a canary had already entered
        normal inference.  Queued-but-unstarted canaries are intentionally not
        included: they observe the capacity drain and requeue before HTTP.
        """

        lock = getattr(self, "_canary_accounting_lock", None)

        def _snapshot() -> Set[Tuple[str, int]]:
            started = set(getattr(self, "_cross_epoch_canaries", set()))
            unfinished = getattr(self, "_unfinished_canary_tests", {})
            return {
                self._miner_model_key(test.miner_address, test.model_index)
                for execution_id, test in tuple(unfinished.items())
                if execution_id in started
            }

        if lock is None:
            return _snapshot()
        with lock:
            return _snapshot()

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
        """Requeue only when our own state authenticates the audit drain.

        The peer's HTTP 503 body is diagnostic input, never authority for an
        unscored deferral. Without an active validator-owned drain for this
        exact miner/model slot, the response follows the ordinary bounded
        busy path and cannot erase the canary obligation.
        """
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

    @staticmethod
    def _active_miner_from_shared_entry(entry) -> ActiveMiner:
        """Restore a validator-owned fallback entry without dropping identity."""

        return ActiveMiner(
            address=entry.address,
            endpoint=entry.endpoint,
            model_id=entry.model_id,
            model_index=entry.model_index,
            quant=entry.quant,
            max_context_len=entry.max_context_len,
            hotkey_ss58=str(getattr(entry, "hotkey_ss58", "") or ""),
            coldkey_ss58=str(getattr(entry, "coldkey_ss58", "") or ""),
            tee_enabled=bool(getattr(entry, "tee_enabled", False)),
            tee_platform=str(getattr(entry, "tee_platform", "") or ""),
            enclave_public_key=str(
                getattr(entry, "enclave_public_key", "") or ""
            ),
            gpu_name=str(getattr(entry, "gpu_name", "") or ""),
            gpu_count=int(getattr(entry, "gpu_count", 0) or 0),
            vram_gb=int(getattr(entry, "vram_gb", 0) or 0),
            compute_capability=str(
                getattr(entry, "compute_capability", "") or ""
            ),
            gpu_uuids=list(getattr(entry, "gpu_uuids", ()) or ()),
        )

    def _apply_mainnet_endpoint_policy(
        self,
        miners: Sequence[ActiveMiner],
        *,
        epoch_number: int,
        preserve_existing: bool = False,
    ) -> list[ActiveMiner]:
        """Persist and remove endpoints the production proxy cannot route.

        Invalid registrations remain in the validator DB for public diagnosis,
        but are inactive for shared-state routing. Their EMA is deliberately
        retained and excluded at weight construction rather than destroyed.
        """

        eligible_miners: list[ActiveMiner] = []
        endpoint_policy_excluded: list[ActiveMiner] = (
            list(getattr(self, "_endpoint_policy_excluded_miners", ()) or ())
            if preserve_existing else []
        )
        endpoint_policy_reasons: Dict[Tuple[str, int], str] = (
            dict(getattr(self, "_endpoint_policy_gate_reasons", {}) or {})
            if preserve_existing else {}
        )
        existing_excluded_keys = {
            self._miner_model_key(miner.address, miner.model_index)
            for miner in endpoint_policy_excluded
        }
        for miner in miners:
            reason = _mainnet_endpoint_eligibility_reason(
                getattr(miner, "endpoint", ""),
                netuid=getattr(self.config, "netuid", 0),
                chain_id=getattr(self.config, "chain_id", None),
            )
            if not reason:
                eligible_miners.append(miner)
                continue
            key = self._miner_model_key(miner.address, miner.model_index)
            endpoint_policy_reasons[key] = reason
            if key not in existing_excluded_keys:
                endpoint_policy_excluded.append(miner)
                existing_excluded_keys.add(key)
            self._db.upsert_entry(
                address=miner.address,
                model_index=miner.model_index,
                model_id=miner.model_id,
                endpoint=miner.endpoint,
                quant=miner.quant,
                max_context_len=miner.max_context_len,
                epoch=epoch_number,
                hotkey_ss58=getattr(miner, "hotkey_ss58", ""),
                coldkey_ss58=getattr(miner, "coldkey_ss58", ""),
            )
            miner_uid = getattr(miner, "bittensor_uid", None)
            if miner_uid is not None:
                self._db.set_uid(miner.address, int(miner_uid))
            self._db.mark_entry_inactive(miner.address, miner.model_index)
            bt.logging.info(
                f"Endpoint policy gate: {miner.address[:10]} "
                f"model_index={miner.model_index} endpoint={miner.endpoint} — {reason}"
            )
        self._endpoint_policy_gate_reasons = endpoint_policy_reasons
        self._endpoint_policy_excluded_miners = tuple(endpoint_policy_excluded)
        if endpoint_policy_excluded:
            entry_word = "entry" if len(endpoint_policy_excluded) == 1 else "entries"
            bt.logging.info(
                f"Endpoint policy: excluded {len(endpoint_policy_excluded)} "
                f"mainnet miner {entry_word} before probing and scheduling"
            )
        return eligible_miners

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

        used_fallback = False
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
                used_fallback = True
                bt.logging.warning(f"Discovery returned 0 miners — falling back to {len(previous_miners)} miners from previous epoch")
            else:
                # Fresh start with no previous miners — try shared state
                from neurons.shared_state import read_shared_state
                try:
                    shared = read_shared_state(self.config.shared_state_path)
                    if shared and shared.miner_endpoints:
                        self._epoch_miners = [
                            self._active_miner_from_shared_entry(m)
                            for m in shared.miner_endpoints
                        ]
                        self._epoch_miners_discovery_valid = False
                        used_fallback = True
                        bt.logging.warning(f"Discovery returned 0 miners, no previous epoch — falling back to {len(self._epoch_miners)} miners from shared state")
                except Exception:
                    pass

                if not self._epoch_miners:
                    self._canary_scheduler = None
                    return

        if used_fallback:
            # Prefer a fresh metagraph identity whenever it is reachable. The
            # fallback values remain intact if this best-effort refresh fails.
            self._enrich_miners_from_metagraph(self._epoch_miners)

        self._epoch_miners = (
            self._retain_epoch_miners_with_authenticated_hotkeys(
                self._epoch_miners,
                epoch_number=epoch_number,
            )
        )

        # Apply the same public-endpoint policy used by the production proxy
        # before TCP probes, identity challenges, canaries or capacity audits.
        # The URL is on-chain registry data, so the decision requires no miner
        # response and cannot be bypassed through /health metadata.
        self._epoch_miners = self._apply_mainnet_endpoint_policy(
            self._epoch_miners,
            epoch_number=epoch_number,
            preserve_existing=used_fallback,
        )

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
                        self._submit_report_offline(miner)
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

        # Capacity admission is chain-derived. Best-effort /health metadata
        # below may affect model eligibility, but never snapshot membership.
        self._refresh_capacity_audit_slot_snapshot_from_miners(
            self._epoch_miners,
            block_number=epoch_start_block,
            source="epoch_setup",
        )

        # ── Fetch hardware metadata from miner /health (best-effort) ──
        hardware_failed, _hardware_stats = self._refresh_miner_hardware_batch(
            self._epoch_miners,
            source=f"epoch_setup:{epoch_number}",
        )

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
                self._submit_report_offline(miner)
            self._epoch_miners = [m for m in self._epoch_miners if id(m) not in failed_ids]
            bt.logging.info(
                f"Hardware health: {len(self._epoch_miners)}/{len(self._epoch_miners) + len(hardware_failed)} "
                f"miners passed, {len(hardware_failed)} excluded"
            )

        if not self._epoch_miners:
            self._canary_scheduler = None
            return

        self._migrate_full_context_debt_for_registrations(
            previous_miners,
            self._epoch_miners,
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
                exact_reader = getattr(
                    self._model_client, "get_on_chain_model_spec", None
                )
                if callable(exact_reader):
                    from verallm.chain.types import on_chain_to_model_spec

                    exact_spec = exact_reader(model_id)
                    if exact_spec is None:
                        self._on_chain_model_spec_cache.pop(model_id, None)
                        spec = None
                    else:
                        self._on_chain_model_spec_cache[model_id] = exact_spec
                        spec = on_chain_to_model_spec(exact_spec)
                else:
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

        artifact_refresh_started = time.monotonic()
        self._refresh_remote_proof_v2_manifests(unique_models)
        self._refresh_remote_proof_v3_releases(unique_models)
        artifact_refresh_elapsed = time.monotonic() - artifact_refresh_started
        if artifact_refresh_elapsed > 12.0:
            bt.logging.warning(
                "Epoch setup artifact refresh exceeded one block: "
                f"{artifact_refresh_elapsed:.1f}s"
            )

        # Reset per-epoch attempt state. Exact obligations are registered after
        # the scheduler has produced the signed-policy plan.
        self._expected_receipts = {}
        self._expected_canary_obligations = {}
        self._validator_canary_failures = set()
        self._canary_penalized_keys = set()
        self._shared_hard_proof_verdicts = {}
        with self._shared_hard_prefetch_lock:
            self._shared_hard_prefetch_results = {}
            self._shared_hard_prefetch_inflight = set()
            self._shared_hard_prefetch_waves = set()
        self._busy_skips = {}
        self._busy_skip_probations = {}
        self._canary_errors: Dict[Tuple[str, int], int] = {}
        self._canary_error_times: Dict[Tuple[str, int], List[int]] = {}

        # Check if TEE is enabled on the subnet (feature flag from SubnetConfig)
        _subnet_tee_enabled = False
        if self._subnet_config_client is not None:
            tee_refresh_started = time.monotonic()
            try:
                _subnet_tee_enabled = self._subnet_config_client.is_tee_enabled_on_subnet()
            except Exception:
                pass
            tee_refresh_elapsed = time.monotonic() - tee_refresh_started
            if tee_refresh_elapsed > 12.0:
                bt.logging.warning(
                    "Epoch setup TEE feature lookup exceeded one block: "
                    f"{tee_refresh_elapsed:.1f}s"
                )
        if not _subnet_tee_enabled:
            # TEE disabled on subnet — treat all miners as non-TEE (use ZK proofs)
            for miner in self._epoch_miners:
                if getattr(miner, "tee_enabled", False):
                    bt.logging.info(
                        f"TEE disabled on subnet — forcing ZK mode for {miner.address[:10]}"
                    )
                    miner.tee_enabled = False
                    miner.tee_platform = ""

        # Plan the exact authority-signed inventory. An unsigned hosted config
        # must not silently inflate the official canary load or scoring budget.
        # The built-in compatibility path retains the protocol's 2+1 floor.
        canary_policy = getattr(self, "_proof_v3_canary_policy", None)
        hard_audit_enabled = _proof_v3_hard_auditor_active(
            self.config,
            self._validator_hotkey_ss58,
        )
        low_count, advertised_light_count, full_count = _effective_canary_counts(
            self.config,
            canary_policy,
            hard_audit_enabled=hard_audit_enabled,
        )
        if (
            self._configured_proof_v3_verdict_source() == "verify"
            and getattr(
                self.config,
                "proof_v3_hard_auditor_policy_enabled",
                False,
            )
            and not hard_audit_enabled
        ):
            bt.logging.warning(
                "This validator is in verify mode but is not the active "
                "subnet-configured hard auditor; scheduling light canaries "
                "and independently verifying retained hard bundles"
            )
        # Candidate timing covers the complete epoch. A hard request that has
        # already started may finish across a scoring boundary; unfinished
        # validator work is neutral for the closed epoch, never a miner miss.
        low_completion_reserve_blocks = 1
        full_completion_reserve_blocks = 1
        hard_audit_drought_epochs = (
            int(canary_policy.max_hard_audit_drought_epochs)
            if canary_policy is not None
            else 2
        )
        hard_audit_due_entries = (
            self._hard_audit_due_entries(
                epoch_number=epoch_number,
                drought_epochs=hard_audit_drought_epochs,
            )
            if hard_audit_enabled
            else set()
        )
        self._canary_scheduler = CanaryScheduler(
            epoch_number=epoch_number,
            epoch_start_block=epoch_start_block,
            epoch_blocks=self.config.epoch_blocks,
            validator_hotkey=self._wallet.hotkey.ss58_address,
            validator_seed=self._validator_private_key,
            small_count=low_count,
            advertised_context_light_count=advertised_light_count,
            full_context_count=full_count,
            proof_sample_rate=1.0,
            probation_entries={
                (addr, idx)
                for addr, indices in self._db.get_probation_addresses().items()
                for idx in indices
            },
            low_context_min_tokens=(
                canary_policy.low_context_min_tokens
                if canary_policy is not None
                else 512
            ),
            low_context_max_tokens=(
                canary_policy.low_context_max_tokens
                if canary_policy is not None
                else 2_048
            ),
            low_context_max_decode_tokens=(
                canary_policy.low_context_max_decode_tokens
                if canary_policy is not None
                else 192
            ),
            full_context_decode_reserve_tokens=(
                canary_policy.full_context_decode_reserve_tokens
                if canary_policy is not None
                else 256
            ),
            full_context_max_decode_tokens=(
                canary_policy.full_context_max_decode_tokens
                if canary_policy is not None
                else 96
            ),
            full_context_max_attempts=(
                canary_policy.full_context_max_attempts
                if canary_policy is not None
                else 4
            ),
            hard_candidate_target_per_epoch=(
                canary_policy.hard_auditor_candidate_target_per_epoch
                if canary_policy is not None
                else 2
            ),
            hard_candidate_bps=(
                canary_policy.hard_auditor_candidate_hard_bps
                if canary_policy is not None
                else 5_000
            ),
            advertised_context_target_bps=(
                canary_policy.advertised_context_target_bps
                if canary_policy is not None
                else 9_000
            ),
            owner_full_min_prompt_bps=(
                getattr(
                    canary_policy,
                    "owner_full_context_min_prompt_bps",
                    1_000,
                )
                if canary_policy is not None
                else 1_000
            ),
            owner_full_max_draw_bps=(
                getattr(
                    canary_policy,
                    "owner_full_context_max_draw_bps",
                    10_000,
                )
                if canary_policy is not None
                else 10_000
            ),
            owner_context_sizing_abi_id=(
                canary_policy.owner_context_sizing_abi_id
                if canary_policy is not None
                else CANARY_OWNER_CONTEXT_SIZING_ABI_V3
            ),
            hard_decode_anchor_bps=(
                canary_policy.hard_decode_anchor_bps
                if canary_policy is not None
                else 2_500
            ),
            hard_decode_tail_bps=(
                canary_policy.hard_decode_tail_bps
                if canary_policy is not None
                else 1_000
            ),
            late_decode_min_output_bps=(
                canary_policy.late_decode_min_output_bps
                if canary_policy is not None
                else 9_000
            ),
            repeat_prefix_target_bps=(
                canary_policy.repeat_prefix_target_bps
                if canary_policy is not None
                else 5_000
            ),
            repeat_prefix_min_tokens=(
                canary_policy.repeat_prefix_min_tokens
                if canary_policy is not None
                else 256
            ),
            hard_audit_enabled=hard_audit_enabled,
            hard_audit_due_entries=hard_audit_due_entries,
            low_completion_reserve_blocks=(
                low_completion_reserve_blocks
            ),
            full_completion_reserve_blocks=(
                full_completion_reserve_blocks
            ),
            hard_context_limits_by_model={
                str(model_id): int(
                    release.qualified_profile.profile.max_verified_context_tokens
                )
                for model_id, release in getattr(
                    self,
                    "_proof_v3_releases",
                    {},
                ).items()
            },
            hard_decode_limits_by_model={
                str(model_id): int(model_policy.max_hard_audit_decode_tokens)
                for model_id in getattr(
                    self,
                    "_proof_v3_releases",
                    {},
                )
                for model_policy in (
                    (
                        canary_policy.model_policy(str(model_id))
                        if canary_policy is not None
                        else None
                    ),
                )
                if model_policy is not None
            },
        )
        tests = self._canary_scheduler.plan_epoch(self._epoch_miners)
        self._register_canary_obligations(tests)

        elapsed = time.monotonic() - t0
        _unique_miners = len({m.address for m in self._epoch_miners})
        _unique_endpoints = len(self._epoch_miners)
        bt.logging.info(f"Epoch {epoch_number}: planned {len(tests)} canary tests for {_unique_miners} miners ({_unique_endpoints} endpoints) ({elapsed:.1f}s)")
        bt.logging.info(
            f"Epoch {epoch_number}: canary schedule spans the complete epoch; "
            "started hard audits may finish across its scoring boundary"
        )
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
        dispatch_epoch = int(self._current_epoch)

        def _on_done(label, future):
            try:
                future.result()
            except Exception as e:
                bt.logging.info(
                    f"Canary test failed for {label}: {e}"
                )
            finally:
                auto_updater = getattr(self, "_auto_updater", None)
                if auto_updater is not None:
                    auto_updater.notify_not_busy()

        paired: Dict[str, List[CanaryTest]] = {}
        independent: List[CanaryTest] = []
        for test in tests:
            pair_id = str(getattr(test, "full_pair_id", "") or "")
            if pair_id:
                paired.setdefault(pair_id, []).append(test)
            else:
                independent.append(test)

        for pair_id, members in paired.items():
            if (
                len(members) != 2
                or {int(item.full_pair_slot) for item in members} != {0, 1}
                or {bool(item.verify_proof) for item in members}
                != {False, True}
                or any(item.test_type != "full_context" for item in members)
            ):
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair dispatch is incomplete"
                )
            pair_targets = {
                (
                    item.miner_address.lower(),
                    item.miner_endpoint,
                    item.model_id,
                    int(item.model_index),
                    int(item.target_block),
                    item.full_pair_id,
                    int(item.full_pair_hold_seconds),
                )
                for item in members
            }
            if len(pair_targets) != 1:
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair dispatch metadata disagrees"
                )
            ordered = tuple(
                sorted(members, key=lambda item: item.full_pair_slot)
            )
            for item in ordered:
                self._mark_canary_queued(
                    dispatch_epoch,
                    self._miner_model_key(
                        item.miner_address,
                        item.model_index,
                    ),
                    item,
                )
            future = self._executor.submit(
                self._execute_canary_full_pair,
                ordered,
                dispatch_epoch,
            )
            label = (
                f"{ordered[0].miner_address[:10]} "
                f"model={ordered[0].model_id} pair={pair_id[:12]}"
            )
            future.add_done_callback(
                lambda f, value=label: _on_done(value, f)
            )

        for test in independent:
            self._mark_canary_queued(
                dispatch_epoch,
                self._miner_model_key(
                    test.miner_address,
                    test.model_index,
                ),
                test,
            )
            future = self._executor.submit(
                self._execute_canary_test, test, dispatch_epoch,
            )
            label = f"{test.miner_address[:10]} model={test.model_id}"
            future.add_done_callback(
                lambda f, value=label: _on_done(value, f)
            )

    def _execute_canary_full_pair(
        self,
        tests: Tuple[CanaryTest, CanaryTest],
        epoch_number: int,
    ) -> None:
        """Run one full pair in two workers while serializing their inference."""

        failures: List[Exception] = []
        lock = threading.Lock()
        barrier = self._proof_v3_full_pair_barrier(tests[0])
        if self._proof_v3_full_pair_barrier(tests[1]) is not barrier:
            raise _ProofV3ValidatorConfigurationError(
                "proof-v3 full-pair did not share one barrier"
            )

        def _run(test: CanaryTest) -> None:
            try:
                self._execute_canary_test(test, epoch_number)
            except Exception as exc:
                with lock:
                    failures.append(exc)
            finally:
                try:
                    barrier.worker_completed(int(test.full_pair_slot))
                finally:
                    self._release_proof_v3_full_pair_barrier(test, barrier)

        workers = tuple(
            threading.Thread(
                target=_run,
                args=(test,),
                name=(
                    "proof-v3-full-pair-"
                    f"{test.full_pair_id[:8]}-{test.full_pair_slot}"
                ),
                daemon=True,
            )
            for test in tests
        )
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if failures:
            raise RuntimeError(
                "proof-v3 full-pair worker failed"
            ) from failures[0]

    def _execute_canary_test(self, test: CanaryTest, epoch_number: int):
        """Serialize only hard proving work for one endpoint."""

        if not bool(test.verify_proof):
            return self._execute_canary_test_unlocked(test, epoch_number)
        lock = self._proof_v3_hard_execution_lock(test)
        with lock:
            return self._execute_canary_test_unlocked(test, epoch_number)

    def _execute_canary_test_unlocked(
        self,
        test: CanaryTest,
        epoch_number: int,
    ):
        """Execute or randomly reschedule one exact canary obligation.

        A 503 received before any proof-v3 precommit is a busy signal, not a
        completed test. Low obligations are rescheduled within the same epoch;
        full-context obligations receive the signed bounded number of attempts.
        Once a precommit is frozen, the proof-v3 transport converts peer
        abandonment into ``ProofV3PeerFailure`` and this wrapper never retries
        or grants busy forgiveness.
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

        key = self._miner_model_key(test.miner_address, test.model_index)
        self._mark_canary_dequeued(epoch_number, key)

        if not self._canary_epoch_active(epoch_number):
            bt.logging.debug(
                f"Skipping stale canary for {test.miner_address[:10]} "
                f"model_index={test.model_index}: test_epoch={epoch_number}, "
                f"current_epoch={self._current_epoch}"
            )
            return

        def _record_precommit_busy(
            attempt_started: float,
            attempt_finished: float,
        ) -> None:
            if not self._canary_epoch_active(epoch_number):
                # A request dispatched before the score boundary has a bounded
                # admission deadline. Delaying a precommit-free 503 across the
                # boundary cannot turn miner-controlled latency into neutral
                # validator work, regardless of the hidden hard/light mark.
                if self._canary_execution_active(test, epoch_number):
                    bt.logging.info(
                        "Cross-epoch canary returned precommit busy "
                        f"after its admission window for "
                        f"{test.miner_address[:10]} "
                        f"model_index={test.model_index}"
                    )
                    penalty_required = True
                    if bool(test.verify_proof):
                        penalty_required = self._record_hard_audit_failure(
                            source_epoch=epoch_number,
                            miner_address=test.miner_address,
                            model_id=test.model_id,
                            model_index=test.model_index,
                            obligation_id=bytes.fromhex(
                                test.obligation_id
                            ),
                            failure_code="late_precommit_busy",
                            endpoint=test.miner_endpoint,
                        )
                    if penalty_required:
                        self._on_proof_failure(
                            test.miner_address,
                            test.model_index,
                            endpoint=test.miner_endpoint,
                            source_epoch=epoch_number,
                        )
                return
            kind = "full" if test.test_type == "full_context" else "low"
            test.rejection_intervals.append(
                (attempt_started, attempt_finished)
            )
            self._busy_skips[key] = self._busy_skips.get(key, 0) + 1
            self._busy_skip_probations.setdefault(key, []).append(
                (
                    attempt_started,
                    attempt_finished,
                    kind,
                    bytes.fromhex(test.obligation_id),
                )
            )
            may_retry = (
                kind == "low"
                or test.attempt_count < max(2, int(test.max_attempts or 0))
            )
            rescheduled = False
            if may_retry and self._canary_epoch_active(epoch_number):
                current_block = int(
                    getattr(self, "_last_known_block", 0)
                    or self._epoch_start_block
                )
                with self._canary_scheduler_lock:
                    scheduler = self._canary_scheduler
                    if (
                        scheduler is not None
                        and scheduler.epoch_number == epoch_number
                    ):
                        rescheduled = scheduler.reschedule(
                            test,
                            current_block=current_block,
                        )
            bt.logging.info(
                f"Canary precommit busy for {test.miner_address[:10]} "
                f"model={test.model_id} kind={kind} "
                f"attempt={test.attempt_count}/"
                f"{test.max_attempts if kind == 'full' else 'epoch'} "
                f"rescheduled={rescheduled}"
            )

        # Capacity selection holds the same lock.  Therefore either its drain
        # exists first and this canary requeues, or this canary is marked
        # started first and the capacity scheduler excludes the endpoint.
        # There is no interval where both sides can independently admit work.
        admission_lock = getattr(self, "_capacity_audit_schedule_lock", None)
        if admission_lock is None:
            if self._requeue_capacity_audit_canary(
                test,
                epoch_number,
                phase="pre_start",
            ):
                return
            self._mark_cross_epoch_canary_started(test, epoch_number)
            self._mark_canary_started(epoch_number, key)
        else:
            with admission_lock:
                if self._requeue_capacity_audit_canary(
                    test,
                    epoch_number,
                    phase="pre_start",
                ):
                    return
                # Mark exact execution identity first so an epoch boundary
                # between dequeue and HTTP setup cannot misclassify this as
                # unstarted work.
                self._mark_cross_epoch_canary_started(test, epoch_number)
                self._mark_canary_started(epoch_number, key)

        try:
            if not self._canary_execution_active(test, epoch_number):
                return
            if self._requeue_capacity_audit_canary(
                test,
                epoch_number,
                phase=f"attempt_{test.attempt_count + 1}",
            ):
                return
            test.attempt_count += 1
            attempt_started = time.time()
            try:
                return self._execute_canary_test_once(
                    test,
                    epoch_number,
                    _transport_retry_allowed=True,
                )
            except _httpx.HTTPStatusError as exc:
                if exc.response.status_code != 503:
                    raise
                attempt_finished = time.time()
                if self._is_capacity_audit_http_503(exc):
                    if self._requeue_capacity_audit_gate_canary(
                        test,
                        epoch_number,
                        phase="precommit_busy",
                    ):
                        return
                _record_precommit_busy(attempt_started, attempt_finished)
                return
            except transport_retry_exc as exc:
                # The proof-v3 exchange reclassifies post-precommit transport
                # loss as a peer proof failure. Only a precommit-free network
                # glitch reaches here and receives one immediate retry.
                bt.logging.info(
                    f"Canary precommit transport retry for "
                    f"{test.miner_address[:10]} type={test.test_type}: "
                    f"{type(exc).__name__}"
                )
                return self._execute_canary_test_once(
                    test,
                    epoch_number,
                    _transport_retry_allowed=False,
                )
        finally:
            self._mark_cross_epoch_canary_finished(test, epoch_number)
            self._mark_canary_finished(epoch_number, key, test)

    def _proof_v3_release_for_canary(self, test: CanaryTest):
        """Return the exact qualified release for one v3 light/hard canary."""

        release = getattr(self, "_proof_v3_releases", {}).get(test.model_id)
        policy = getattr(self, "_proof_v3_canary_policy", None)
        rollout = getattr(self, "_proof_protocol_rollout_cfg", None)
        if rollout is None:
            rollout = proof_protocol_rollout_config_from_neuron_config(
                self.config
            )
        locally_ready = [LEGACY_PROOF_PROTOCOL_VERSION]
        if release is not None and policy is not None:
            locally_ready.append(PROOF_PROTOCOL_V3)
        advertised_versions = tuple(
            int(version)
            for version in (test.proof_protocol_versions or (1,))
        )
        selected = select_proof_protocol_version(
            rollout,
            locally_supported=tuple(locally_ready),
            peer_advertised=advertised_versions,
        )
        any_local_allowed = any(
            proof_protocol_allowed(rollout, version)
            for version in locally_ready
        )
        if selected is None and not any_local_allowed:
            raise _ProofV3ValidatorConfigurationError(
                "subnet allows no proof protocol supported by this validator"
            )
        if selected is None:
            from verallm.api.proof_v3_validator import ProofV3PeerFailure

            raise ProofV3PeerFailure(
                "miner advertises no mutually supported allowed proof protocol"
            )
        if selected == LEGACY_PROOF_PROTOCOL_VERSION:
            return None
        if selected != PROOF_PROTOCOL_V3:
            raise _ProofV3ValidatorConfigurationError(
                "selected proof protocol is unsupported by this validator"
            )
        assert release is not None and policy is not None
        model_policy = policy.model_policy(test.model_id)
        if model_policy is None:
            raise _ProofV3ValidatorConfigurationError(
                "signed canary policy does not cover the selected model"
            )
        profile = release.qualified_profile.profile
        if model_policy.execution_profile_digest != profile.digest():
            raise _ProofV3ValidatorConfigurationError(
                "signed canary policy profile binding is stale"
            )
        quant = str(getattr(test, "quant", "") or "").strip().lower()
        if not quant or quant not in model_policy.allowed_quantizations:
            from verallm.api.proof_v3_validator import ProofV3PeerFailure

            raise ProofV3PeerFailure(
                "advertised quantization is not qualified by the signed v3 policy"
            )
        if (
            test.verify_proof
            and int(getattr(test, "target_prompt_tokens", 0) or 0)
            > int(profile.max_verified_context_tokens)
        ):
            raise _ProofV3ValidatorConfigurationError(
                "hard canary target exceeds the signed v3 profile"
            )
        return release

    def _proof_v3_quant_qualified(self, miner: ActiveMiner) -> bool:
        """Check the advertised quant against the authenticated v3 release."""

        release = self._epoch_close_value(
            "_proof_v3_releases",
            {},
        ).get(miner.model_id)
        policy = self._epoch_close_value(
            "_proof_v3_canary_policy",
            None,
        )
        rollout = self._epoch_close_value(
            "_proof_protocol_rollout_cfg",
            None,
        )
        if rollout is None:
            rollout = proof_protocol_rollout_config_from_neuron_config(
                self.config
            )
        locally_ready = [LEGACY_PROOF_PROTOCOL_VERSION]
        if release is not None and policy is not None:
            locally_ready.append(PROOF_PROTOCOL_V3)
        selected = select_proof_protocol_version(
            rollout,
            locally_supported=tuple(locally_ready),
            peer_advertised=tuple(
                int(version)
                for version in (
                    getattr(miner, "proof_protocol_versions", ()) or (1,)
                )
            ),
        )
        if selected == LEGACY_PROOF_PROTOCOL_VERSION:
            return True
        if selected != PROOF_PROTOCOL_V3 or release is None or policy is None:
            return False
        model_policy = policy.model_policy(miner.model_id)
        if model_policy is None:
            return False
        return bool(
            model_policy.execution_profile_digest
            == release.qualified_profile.profile.digest()
            and str(getattr(miner, "quant", "") or "").strip().lower()
            in model_policy.allowed_quantizations
        )

    def _proof_v3_full_pair_barrier(
        self,
        test: CanaryTest,
    ) -> _ProofV3FullPairBarrier:
        pair_id = str(getattr(test, "full_pair_id", "") or "")
        hold_seconds = int(
            getattr(test, "full_pair_hold_seconds", 0) or 0
        )
        lock = getattr(self, "_proof_v3_full_pair_barriers_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._proof_v3_full_pair_barriers_lock = lock
            self._proof_v3_full_pair_barriers = {}
        with lock:
            barriers = self._proof_v3_full_pair_barriers
            barrier = barriers.get(pair_id)
            if barrier is None:
                barrier = _ProofV3FullPairBarrier(pair_id, hold_seconds)
                barriers[pair_id] = barrier
            elif barrier.hold_seconds != hold_seconds:
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 full-pair hold deadlines disagree"
                )
            return barrier

    def _release_proof_v3_full_pair_barrier(
        self,
        test: CanaryTest,
        barrier: _ProofV3FullPairBarrier,
    ) -> None:
        if not barrier.depart(int(test.full_pair_slot)):
            return
        lock = self._proof_v3_full_pair_barriers_lock
        with lock:
            if self._proof_v3_full_pair_barriers.get(
                test.full_pair_id
            ) is barrier:
                self._proof_v3_full_pair_barriers.pop(
                    test.full_pair_id,
                    None,
                )

    def _claim_proof_v3_pair_failure_attribution(
        self,
        test: CanaryTest,
    ) -> bool:
        pair_id = str(getattr(test, "full_pair_id", "") or "")
        if not pair_id:
            return True
        lock = getattr(self, "_proof_v3_full_pair_barriers_lock", None)
        if lock is None:
            return True
        with lock:
            barrier = self._proof_v3_full_pair_barriers.get(pair_id)
        if barrier is None:
            return True
        return barrier.claim_failure_attribution()

    def _run_proof_v3_canary(
        self,
        *,
        client: ValidatorClient,
        test: CanaryTest,
        messages: list[dict],
        qualified_release,
        prompt_token_ids: Optional[Sequence[int]] = None,
    ):
        """Execute one nonce-free light or authorized hard v3 canary."""

        try:
            if prompt_token_ids is None:
                prompt_token_ids = _tokenize_proof_v3_chat(
                    test.model_id,
                    messages,
                    enable_thinking=test.enable_thinking,
                )
            else:
                prompt_token_ids = tuple(int(value) for value in prompt_token_ids)
            miner_hotkey = self._get_miner_ss58(
                test.miner_address,
                "hotkey",
            )
            if not miner_hotkey:
                raise RuntimeError(
                    "proof-v3 miner hotkey identity is unavailable"
                )
            from verallm.proof_v3.request import (
                miner_hotkey_identity_digest_v3,
                validator_hotkey_identity_digest_v3,
            )

            validator_identity = validator_hotkey_identity_digest_v3(
                self._validator_hotkey_ss58
            )
            miner_identity = miner_hotkey_identity_digest_v3(miner_hotkey)
            runtime_policy = _proof_v3_canary_runtime_policy(
                qualified_release,
                hard_audit=bool(test.verify_proof),
            )
        except Exception as exc:
            raise _ProofV3ValidatorConfigurationError(
                "proof-v3 canary preflight failed"
            ) from exc

        request_kwargs = {
            "messages": messages,
            "prompt_token_ids": prompt_token_ids,
            "qualified_profile": qualified_release.qualified_profile,
            "validator_identity_digest": validator_identity,
            "miner_identity_digest": miner_identity,
            "runtime_policy": runtime_policy,
            "max_new_tokens": test.max_new_tokens,
            "enable_thinking": test.enable_thinking,
            "presence_penalty": float(test.presence_penalty or 0.0),
            "top_k": int(test.top_k if test.top_k is not None else -1),
            "top_p": float(test.top_p or 1.0),
            "min_p": 0.0,
        }
        from neurons.subnet_runtime_config import (
            proof_v3_timing_config_from_neuron_config,
        )

        request_kwargs["hard_proof_arrival_budget_ns"] = (
            proof_v3_timing_config_from_neuron_config(
                self.config
            ).hard_proof_arrival_budget_ns(test.max_new_tokens)
        )
        pair_id = str(getattr(test, "full_pair_id", "") or "")
        barrier = None
        exchange = None
        try:
            if not pair_id:
                result = client.run_chat_proof_v3(**request_kwargs)
            else:
                from verallm.api.proof_v3_validator import ProofV3PeerFailure
                from verallm.proof_v3.errors import ProofV3UnavailableError

                barrier = self._proof_v3_full_pair_barrier(test)
                slot = int(getattr(test, "full_pair_slot", -1))
                barrier.wait_for_inference_turn(slot)
                exchange = client.run_chat_proof_v3_precommit(
                    **request_kwargs,
                    nonce_reveal_hold_budget_ns=(
                        int(test.full_pair_hold_seconds) * 1_000_000_000
                    ),
                    expected_hard_audit=bool(test.verify_proof),
                )
                if slot == 0:
                    # The first frozen record must survive the second full
                    # inference. This authenticated hold reveals neither the
                    # local tier decision nor the nonce.
                    client.hold_chat_proof_v3_precommit(exchange)
                decision = getattr(exchange, "audit_decision", None)
                selected_hard = getattr(
                    decision,
                    "hard_audit_selected",
                    None,
                )
                if type(selected_hard) is not bool or selected_hard != bool(
                    test.verify_proof
                ):
                    exchange.fail_closed()
                    raise _ProofV3ValidatorConfigurationError(
                        "paired full-canary tier conditioning is inconsistent"
                    )
                barrier.mark_precommitted(
                    slot,
                    hard_audit=selected_hard,
                    exchange=exchange,
                )
                barrier.wait_until_both_precommitted()
                result = client.finalize_chat_proof_v3(exchange)
            hard_audit = bool(test.verify_proof)
            minimum_output = getattr(
                test,
                "minimum_observed_output_tokens",
                0,
            )
            if (
                isinstance(minimum_output, bool)
                or not isinstance(minimum_output, int)
                or minimum_output < 0
                or minimum_output > int(test.max_new_tokens)
            ):
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 canary minimum output is malformed"
                )
            # Count the exact token-id sequence accumulated by the validator's
            # SSE exchange. Never use a miner-reported aggregate count.
            observed_output_tokens = len(result.output_token_ids)
            if (
                hard_audit
                and minimum_output > 0
                and observed_output_tokens < minimum_output
            ):
                from verallm.api.proof_v3_validator import ProofV3PeerFailure

                raise ProofV3PeerFailure(
                    "proof-v3 hard canary ended before its signed late-decode "
                    f"position (observed={observed_output_tokens}, "
                    f"required={minimum_output})"
                )
            if hard_audit and (
                not result.audit_decision.hard_audit_selected
                or not result.proof_verified
            ):
                from verallm.api.proof_v3_validator import ProofV3PeerFailure

                raise ProofV3PeerFailure(
                    "proof-v3 canary did not complete its mandatory hard audit"
                )
            if not hard_audit and (
                result.audit_decision.hard_audit_selected
                or result.proof_verified
                or result.proof_wire_bytes
            ):
                raise _ProofV3ValidatorConfigurationError(
                    "proof-v3 light canary unexpectedly entered the hard path"
                )
            if barrier is not None:
                barrier.mark_completed(slot)
            return result
        except Exception as exc:
            if barrier is not None:
                if exchange is not None:
                    exchange.fail_closed()
                had_precommit = barrier.abort(exc)
                if (
                    had_precommit
                    and not isinstance(
                        exc,
                        (
                            _ProofV3ValidatorConfigurationError,
                            ProofV3UnavailableError,
                            ProofV3PeerFailure,
                        ),
                    )
                ):
                    raise ProofV3PeerFailure(
                        "paired full canary failed after its first precommit"
                    ) from exc
            raise

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
        preparation_started = time.monotonic()

        try:
            if not self._canary_execution_active(test, epoch_number):
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
                return

            bt.logging.info(
                "Canary preparation started for "
                f"{test.miner_address[:10]} model_index={test.model_index} "
                f"kind={test.test_type} hard={bool(test.verify_proof)}"
            )

            prompt = test.prompt
            prompt_token_ids: Optional[Sequence[int]] = None
            if int(test.target_prompt_tokens or 0) > 0:
                def _count_canary_tokens(text: str) -> int:
                    return len(
                        _tokenize_proof_v3_chat(
                            test.model_id,
                            [{"role": "user", "content": text}],
                            enable_thinking=test.enable_thinking,
                        )
                    )

                prompt, measured_prompt_tokens = materialize_canary_prompt(
                    test,
                    token_counter=_count_canary_tokens,
                )
                prompt_token_ids = _tokenize_proof_v3_chat(
                    test.model_id,
                    [{"role": "user", "content": prompt}],
                    enable_thinking=test.enable_thinking,
                )
                if len(prompt_token_ids) != measured_prompt_tokens:
                    raise _ProofV3ValidatorConfigurationError(
                        "validator tokenizer did not retain the canary tokenization"
                    )
            bt.logging.info(
                "Canary prompt prepared for "
                f"{test.miner_address[:10]} model_index={test.model_index} "
                f"tokens={len(prompt_token_ids) if prompt_token_ids is not None else 'compat'} "
                f"elapsed={time.monotonic() - preparation_started:.3f}s"
            )
            messages = [{"role": "user", "content": prompt}]
            proof_v3_release = ValidatorNeuron._proof_v3_release_for_canary(
                self,
                test,
            )
            bt.logging.info(
                "Canary transport starting for "
                f"{test.miner_address[:10]} model_index={test.model_index} "
                f"protocol={'v3' if proof_v3_release is not None else 'v1'} "
                f"elapsed={time.monotonic() - preparation_started:.3f}s"
            )

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
                # Hard canaries use the ordinary /chat surface but remain
                # deterministic so the signed LM-head/token relation is exact.
                do_sample = False
                temperature = test.temperature
                sampling_bps = 10_000
                enable_thinking = test.enable_thinking
                proof_v3_result = None
                if proof_v3_release is not None:
                    proof_v3_result = self._run_proof_v3_canary(
                        client=client,
                        test=test,
                        messages=messages,
                        qualified_release=proof_v3_release,
                        prompt_token_ids=prompt_token_ids,
                    )
                    full_text = proof_v3_result.text
                    commitment = None
                    proof_bundle = None
                    nonce = None
                    timing = {
                        "validator_request_start_ts": (
                            proof_v3_result.validator_request_start_ts
                        ),
                        "validator_request_end_ts": (
                            proof_v3_result.validator_request_end_ts
                        ),
                        "validator_request_ms": (
                            proof_v3_result.validator_request_ms
                        ),
                        "round_trip_ms": proof_v3_result.round_trip_ms,
                        "ttft_ms": proof_v3_result.ttft_ms,
                        "inference_ms": proof_v3_result.inference_ms,
                        "input_tokens": proof_v3_result.input_tokens,
                        "output_tokens": len(
                            proof_v3_result.output_token_ids
                        ),
                        "commitment_ms": (
                            proof_v3_result.last_token_to_precommit_ms
                        ),
                        "prove_ms": proof_v3_result.nonce_to_proof_ms,
                        "proof_wire_bytes": proof_v3_result.proof_wire_bytes,
                    }
                    canary_commitment_hash = (
                        proof_v3_result.commitment_envelope_digest
                    )
                else:
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
                        allow_unbound_output_count=test.verify_tee,
                        # Production rollout is direct v1 -> v3. Reaching this
                        # branch means no qualified v3 route was selected, so
                        # only bounded legacy compatibility may authorize it.
                        proof_protocol_version=LEGACY_PROOF_PROTOCOL_VERSION,
                    )
                    canary_commitment_hash = commitment.commitment_hash()

                bt.logging.info(
                    "Canary transport completed for "
                    f"{test.miner_address[:10]} model_index={test.model_index} "
                    f"hard={bool(test.verify_proof)} "
                    f"elapsed={time.monotonic() - preparation_started:.3f}s"
                )

                if not self._canary_execution_active(test, epoch_number):
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
                if proof_v3_result is not None:
                    proof_verified = bool(
                        test.verify_proof and proof_v3_result.proof_verified
                    )
                    if proof_verified and test.verify_proof:
                        self._record_hard_audit_pass(
                            test,
                            completion_epoch=int(self._current_epoch),
                            source_epoch=int(epoch_number),
                        )
                    elif test.verify_proof:
                        proof_failure_reason = (
                            "proof-v3 hard audit did not verify"
                        )
                        penalty_required = self._record_hard_audit_failure(
                            source_epoch=epoch_number,
                            miner_address=test.miner_address,
                            model_id=test.model_id,
                            model_index=test.model_index,
                            obligation_id=bytes.fromhex(
                                test.obligation_id
                            ),
                            failure_code="post_precommit_failure",
                            endpoint=test.miner_endpoint,
                        )
                        if penalty_required:
                            self._on_proof_failure(
                                test.miner_address,
                                test.model_index,
                                endpoint=test.miner_endpoint,
                                source_epoch=epoch_number,
                            )
                        if (
                            penalty_required
                            and self._canary_epoch_active(epoch_number)
                        ):
                            self._canary_penalized_keys.add(
                                self._miner_model_key(
                                    test.miner_address,
                                    test.model_index,
                                )
                            )
                    if proof_v3_result.verification_ms is not None:
                        verify_timing["proof_v3"] = (
                            proof_v3_result.verification_ms
                        )
                elif test.verify_proof:
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
                            self._attach_verified_proof_v2_manifest(
                                client,
                                test.model_id,
                            )
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
                        result, verify_timing, policy_decision = verify_with_proof_policy(
                            proof_bundle,
                            lambda: client.verify_proof(
                                proof_bundle, nonce,
                                expected_sampling_verification_bps=sampling_bps,
                                expected_do_sample=do_sample,
                                expected_temperature=temperature,
                                enable_thinking=enable_thinking,
                                expected_input_commitment=expected_ic,
                                expected_prompt_hash=expected_ph,
                                expected_sampler_config_hash=_expected_scfg,
                                expected_presence_penalty=_expected_pp,
                                # Pass raw sampling params so the canonical
                                # replay (high_assurance) can use them. Bound
                                # separately by sampler_config_hash.
                                expected_top_k=int(test.top_k or -1),
                                expected_top_p=float(test.top_p or 1.0),
                                expected_min_p=float(
                                    getattr(test, "min_p", 0.0) or 0.0
                                ),
                            ),
                            legacy_v1_compatibility_active=self._legacy_v1_compatibility_active(
                                current_epoch=epoch_number,
                            ),
                            expected_protocol_version=(
                                LEGACY_PROOF_PROTOCOL_VERSION
                            ),
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
                            if not self._canary_execution_active(
                                test,
                                epoch_number,
                            ):
                                return
                            # Mid-epoch cutoff: immediately put on probation
                            # and update shared state so proxy stops routing
                            self._on_proof_failure(
                                test.miner_address, test.model_index,
                                endpoint=test.miner_endpoint,
                                source_epoch=epoch_number,
                            )
                            self._canary_penalized_keys.add(
                                self._miner_model_key(
                                    test.miner_address,
                                    test.model_index,
                                )
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

                if not self._canary_execution_active(test, epoch_number):
                    bt.logging.debug(
                        f"Not pushing stale canary receipt for {test.miner_address[:10]} "
                        f"model_index={test.model_index}: test_epoch={epoch_number}, "
                        f"current_epoch={self._current_epoch}"
                    )
                    return
                late_completion = not self._canary_epoch_active(
                    epoch_number,
                )
                security_only_receipt = bool(
                    late_completion and test.verify_proof
                )
                if late_completion:
                    self._complete_cross_epoch_full_success(
                        test,
                        epoch_number,
                    )
                if security_only_receipt:
                    self._reconcile_late_hard_probation_pass(
                        test,
                        epoch_number,
                    )
                    # Verification already completed above. The original
                    # scoring epoch is immutable. Use the dedicated
                    # hard-audit route so the receipt promotes the retained
                    # bundle without entering throughput receipt accounting.
                    bt.logging.info(
                        "Proof-v3 hard audit completed after its scoring "
                        f"boundary for {test.miner_address[:10]} "
                        f"model_index={test.model_index}; verdict accepted, "
                        "security-only receipt retained"
                    )
                elif late_completion:
                    bt.logging.info(
                        "Canary completed after its scoring boundary for "
                        f"{test.miner_address[:10]} "
                        f"model_index={test.model_index}; terminal success "
                        "recorded without credit in a newer epoch"
                    )

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

                pushed_ok = self._push_receipt_to_miner(
                    miner_address=test.miner_address,
                    miner_endpoint=test.miner_endpoint,
                    model_id=test.model_id,
                    model_index=test.model_index,
                    epoch_number=epoch_number,
                    commitment_hash=canary_commitment_hash,
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
                    canary_obligation_id=bytes.fromhex(test.obligation_id),
                    canary_kind=(
                        "full"
                        if test.test_type == "full_context"
                        else "low"
                    ),
                    canary_target_prompt_tokens=test.target_prompt_tokens,
                    capture_chain_digest=(
                        proof_v3_result.capture_chain_digest
                        if proof_v3_result is not None
                        else b""
                    ),
                    security_only=security_only_receipt,
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
                        proof_verified=(
                            1
                            if proof_verified
                            else (0 if test.verify_proof else None)
                        ),
                        proof_failure_reason=proof_failure_reason,
                        prove_ms=timing.get("prove_ms"),
                        commitment_ms=timing.get("commitment_ms"),
                        verify_ms=sum(verify_timing.values()) if verify_timing else None,
                        commitment_hash=canary_commitment_hash.hex(),
                        receipt_pushed=1 if pushed_ok else 0,
                    )
                except Exception as _db_err:
                    bt.logging.debug(f"Failed to log canary result: {_db_err}")

        except Exception as e:
            if not self._canary_execution_active(test, epoch_number):
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
            from verallm.api.proof_v3_validator import ProofV3PeerFailure
            from verallm.proof_v3.errors import ProofV3UnavailableError

            if isinstance(
                e,
                (_ProofV3ValidatorConfigurationError, ProofV3UnavailableError),
            ):
                bt.logging.warning(
                    "Validator-side proof-v3 failure "
                    f"(NOT attributed to miner {test.miner_address[:10]} "
                    f"model={test.model_id}): {type(e).__name__}: {e}"
                )
                if self._canary_epoch_active(epoch_number):
                    self._validator_canary_failures.add(
                        self._miner_model_key(
                            test.miner_address, test.model_index
                        )
                    )
                return
            if isinstance(e, ProofV3PeerFailure):
                if not self._canary_execution_active(test, epoch_number):
                    return
                bt.logging.debug(
                    f"Proof-v3 failure | {test.miner_address[:10]} "
                    f"| model={test.model_id}: {e}"
                )
                if not self._claim_proof_v3_pair_failure_attribution(test):
                    bt.logging.debug(
                        "Proof-v3 paired failure already attributed | "
                        f"{test.miner_address[:10]} | model={test.model_id}"
                    )
                    return
                penalty_required = True
                if bool(test.verify_proof):
                    penalty_required = self._record_hard_audit_failure(
                        source_epoch=epoch_number,
                        miner_address=test.miner_address,
                        model_id=test.model_id,
                        model_index=test.model_index,
                        obligation_id=bytes.fromhex(test.obligation_id),
                        failure_code="post_precommit_failure",
                        endpoint=test.miner_endpoint,
                    )
                else:
                    # A nonce-free v3 request can fail because the peer's
                    # shared inference engine aborts one request while an
                    # authenticated hard proof is running.  Preserve the
                    # failure as endpoint evidence, but apply the same
                    # distinct-source-epoch strike policy as a hard peer
                    # failure instead of halving EMA on the first transient
                    # service error.  This must not publish a hard-audit
                    # outcome: no nonce or hard proof was requested here.
                    penalty_required = self._record_hard_failure_strike(
                        self._miner_model_key(
                            test.miner_address,
                            test.model_index,
                        ),
                        source_epoch=epoch_number,
                        endpoint=test.miner_endpoint,
                    )
                if penalty_required:
                    self._on_proof_failure(
                        test.miner_address,
                        test.model_index,
                        endpoint=test.miner_endpoint,
                        source_epoch=epoch_number,
                    )
                if (
                    penalty_required
                    and self._canary_epoch_active(epoch_number)
                ):
                    self._canary_penalized_keys.add(
                        self._miner_model_key(
                            test.miner_address,
                            test.model_index,
                        )
                    )
                try:
                    self._db.log_canary_result(
                        network=self.config.subtensor_network or "unknown",
                        chain_id=getattr(self.config, "chain_id", 0),
                        netuid=self.config.netuid,
                        epoch_number=epoch_number,
                        block_number=self._last_known_block or 0,
                        miner_address=test.miner_address,
                        miner_uid=self._db.get_uid(test.miner_address),
                        miner_hotkey_ss58=self._get_miner_ss58(
                            test.miner_address,
                            "hotkey",
                        ),
                        miner_coldkey_ss58=self._get_miner_ss58(
                            test.miner_address,
                            "coldkey",
                        ),
                        model_id=test.model_id,
                        model_index=test.model_index,
                        endpoint=test.miner_endpoint,
                        test_type=test.test_type,
                        test_index=test.test_index,
                        proof_requested=1 if test.verify_proof else 0,
                        enable_thinking=1 if test.enable_thinking else 0,
                        temperature=test.temperature,
                        max_new_tokens=test.max_new_tokens,
                        status="proof_failed",
                        proof_verified=0,
                        proof_failure_reason=str(e)[:500],
                    )
                except Exception as db_error:
                    bt.logging.debug(
                        f"Failed to log proof-v3 failure: {db_error}"
                    )
                return
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
                if self._canary_epoch_active(epoch_number):
                    self._validator_canary_failures.add(
                        self._miner_model_key(
                            test.miner_address, test.model_index
                        )
                    )
                return
            _uid_err = self._db.get_uid(test.miner_address)
            _uid_err_s = f"UID {_uid_err}" if _uid_err is not None else "UID ?"
            _err_msg = str(e).split("\nFor more information")[0]
            if isinstance(e, _httpx.HTTPStatusError):
                try:
                    _peer_error = e.response.json().get("error")
                except Exception:
                    _peer_error = None
                if isinstance(_peer_error, str) and _peer_error.strip():
                    _peer_error = " ".join(_peer_error.split())[:300]
                    _err_msg = f"{_err_msg}: {_peer_error}"
            bt.logging.info(f"Canary test error for {_uid_err_s} {test.miner_address[:10]} model={test.model_id}: {_err_msg}")

            # Once a miner-visible request has crossed its source boundary it
            # is no longer part of that epoch's throughput accounting, but it
            # still owes a terminal response by the configured request
            # deadline.  A peer-attributed timeout/error therefore fails
            # immediately instead of disappearing as validator-neutral work.
            if not self._canary_epoch_active(epoch_number):
                penalty_required = True
                if bool(test.verify_proof):
                    penalty_required = self._record_hard_audit_failure(
                        source_epoch=epoch_number,
                        miner_address=test.miner_address,
                        model_id=test.model_id,
                        model_index=test.model_index,
                        obligation_id=bytes.fromhex(test.obligation_id),
                        failure_code="post_precommit_failure",
                        endpoint=test.miner_endpoint,
                    )
                if penalty_required:
                    self._on_proof_failure(
                        test.miner_address,
                        test.model_index,
                        endpoint=test.miner_endpoint,
                        source_epoch=epoch_number,
                    )

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

            # Track endpoint failures for exact obligation reconciliation at
            # epoch close. Planned receipt counts are never decremented.
            if self._canary_epoch_active(epoch_number):
                key = self._miner_model_key(
                    test.miner_address,
                    test.model_index,
                )
                self._canary_errors[key] = (
                    self._canary_errors.get(key, 0) + 1
                )
                self._canary_error_times.setdefault(key, []).append(
                    int(time.time())
                )

    def _try_close_epoch(self, epoch_number: int) -> bool:
        """Attempt epoch close with exponential backoff on failure.

        On success, clears pending state and resets backoff.
        On failure (e.g. 429 rate limit), schedules retry with increasing delay
        to avoid hammering the RPC.
        """
        close_lock = getattr(self, "_epoch_close_lock", None)
        if close_lock is None:
            # Compatibility for narrowly constructed test fixtures.  Normal
            # validator instances create the lock in __init__ before block
            # subscription starts.
            close_lock = threading.Lock()
            self._epoch_close_lock = close_lock
        if not close_lock.acquire(blocking=False):
            return False

        try:
            # Guard: never close the same epoch twice — each re-close blends
            # another score into the EMA, destroying it.  This check must be
            # inside the single-flight section.
            if not hasattr(self, '_last_closed_epoch'):
                self._last_closed_epoch = -1
            if epoch_number <= self._last_closed_epoch:
                self._pending_epoch_close = None
                return True

            # The validator must schedule every obligation early enough to finish.
            # Any queued/in-flight work still present here is therefore local
            # indeterminacy, never evidence against the miner and never a reason to
            # delay the next canary epoch.
            self._seal_canary_epoch_for_close(epoch_number)

            now = time.monotonic()
            if now < self._epoch_close_retry_after:
                return False  # Still in cooldown from a previous failure

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
                if getattr(self, "_weight_update_due", False):
                    self._schedule_weight_update()
                    self._weight_update_due = False
                # If auto-update was deferred, apply it now (between epochs)
                auto_updater = getattr(self, "_auto_updater", None)
                if auto_updater is not None:
                    auto_updater.notify_not_busy()
                return True
            except Exception as e:
                self._epoch_close_retry_after = now + self._epoch_close_backoff
                bt.logging.warning(f"Epoch {epoch_number} close failed, retrying in {self._epoch_close_backoff:.0f}s: {e}")
                self._epoch_close_backoff = min(self._epoch_close_backoff * 2, 300)
                return False
        finally:
            close_lock.release()

    # Epoch duration in seconds — 360 blocks × 12s = 4320s ≈ 72 min.
    # Used by the restart-forgiveness window (2 epochs).
    _EPOCH_DURATION_SEC = 360 * 12
    # A healthy owner may need receipt reconciliation and cryptographic replay
    # before its immutable snapshot exists. Followers wait generously, but on
    # the dedicated old-epoch close worker: current-epoch scheduling never
    # waits on this timeout. Exhaustion is fail-neutral/generous.
    _FOLLOWER_OWNER_CLOSE_TIMEOUT_SEC = 180.0
    _FOLLOWER_OWNER_CLOSE_RETRY_CAP_SEC = 5.0

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
        epoch_miners = tuple(
            self._epoch_close_value("_epoch_miners", ())
        )
        expected_obligations_by_key = self._epoch_close_value(
            "_expected_canary_obligations",
            {},
        )
        validator_canary_failures = self._epoch_close_value(
            "_validator_canary_failures",
            set(),
        )
        canary_penalized_keys = self._epoch_close_value(
            "_canary_penalized_keys",
            set(),
        )
        busy_skips = self._epoch_close_value("_busy_skips", {})
        scoring = self._epoch_close_value("_scoring", self._scoring)
        bt.logging.info(
            f"Epoch {epoch_number} closing: pulling receipts from {len(epoch_miners)} miners",
        )

        # Detect a remote miner_version bump and (re-)open the restart-
        # forgiveness window before scoring this epoch's canary errors.
        try:
            self._poll_remote_miner_version()
        except Exception as e:
            bt.logging.debug(f"miner_version poll failed: {e}")

        close_state = getattr(
            getattr(self, "_epoch_close_local", None),
            "state",
            None,
        )
        if close_state is not None:
            # The old epoch already latched an authenticated runtime config at
            # setup. A concurrent new-epoch refresh must not rewrite its
            # scoring semantics while its close waits for the owner.
            runtime_config_loaded = True
        else:
            runtime_config_loaded = self._refresh_subnet_runtime_config(
                current_epoch=epoch_number,
                force=True,
            )
            scoring = self._scoring

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
                    vr = ValidatorRegistryClient(
                        self.config,
                        provider=self._chain_provider,
                    )
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
        self._set_epoch_close_value("_receipt_pull_failed_keys", set())
        miner_receipts, all_epoch_receipts = self._collect_epoch_receipts(
            epoch_number, receipt_authority,
        )
        current_shared_hard_verdicts = (
            self._verify_shared_hard_bundles_at_close(
                epoch_number,
                miner_receipts,
            )
        )
        late_shared_hard_verdicts = (
            self._verify_late_shared_hard_bundles_at_close(
                current_epoch=epoch_number,
                receipt_authority=receipt_authority,
            )
        )
        shared_hard_failure_verdicts = (
            self._fetch_shared_hard_failure_verdicts(
                current_epoch=epoch_number,
            )
        )
        shared_hard_proof_verdicts = dict(
            current_shared_hard_verdicts
        )
        for key, verdict in late_shared_hard_verdicts.items():
            # Any independently verified failure dominates a pass; otherwise
            # the newest unprocessed late pass supplies the current security
            # verdict without changing the closed epoch's throughput.
            if verdict is False or key not in shared_hard_proof_verdicts:
                shared_hard_proof_verdicts[key] = verdict
        # Positive owner-signed failure evidence always dominates a retained
        # pass. The owner refuses to publish both for one obligation; treating
        # any inconsistency conservatively prevents a downgrade.
        for key in shared_hard_failure_verdicts:
            shared_hard_proof_verdicts[key] = False
        self._set_epoch_close_value(
            "_shared_hard_proof_verdicts",
            shared_hard_proof_verdicts,
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

        # Only the epoch-latched subnet scoring authority may create organic
        # throughput and model-demand credit. Other permitted validators still
        # contribute independently authenticated canary performance samples,
        # but cannot manufacture traffic volume for a colluding miner.
        scoring_authority_hotkey = b""
        scoring_authority_ss58 = str(
            self._epoch_close_value(
                "owner_hotkey_ss58",
                getattr(
                    self.config,
                    "proof_v3_hard_auditor_hotkey_ss58",
                    "",
                ),
            )
            or ""
        )
        try:
            scoring_authority_hotkey = _decode_scoring_authority_hotkey(
                scoring_authority_ss58,
            )
        except Exception as exc:
            scoring_authority_hotkey = b""
            bt.logging.warning(
                f"Epoch {epoch_number}: scoring authority is unavailable; "
                f"organic throughput and demand credit are disabled: {exc}"
            )
        scoring_authority_receipts = select_scoring_authority_receipts(
            all_epoch_receipts,
            scoring_authority_hotkey,
        )

        # ── Compute per-model demand ──────────────────────────────
        demand_scores: Dict[str, int] = {}
        if self.config.demand_bonus_enabled:
            demand_scores = compute_model_demand(
                scoring_authority_receipts,
                epoch_number,
            )
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
                f"Runtime subnet config scoring: tee={scoring.tee_bonus:.2f} "
                f"ema={scoring.ema_alpha:.2f} tp={scoring.throughput_power:.1f} "
                f"proof_rate={scoring.proof_sample_rate:.2f} "
                f"prob_passes={scoring.probation_required_passes} "
                f"demand_max={scoring.demand_bonus_max:.2f} "
                f"burn={scoring.emission_burn:.0%}"
            )
        elif self._subnet_config_client is not None:
            try:
                scoring = self._subnet_config_client.get_scoring_params()
                self._scoring = scoring
                self._last_good_scoring = scoring  # cache for fallback
                bt.logging.debug(
                    f"SubnetConfig fallback scoring: tee={scoring.tee_bonus:.2f} "
                    f"ema={scoring.ema_alpha:.2f} tp={scoring.throughput_power:.1f} "
                    f"proof_rate={scoring.proof_sample_rate:.2f} "
                    f"prob_passes={scoring.probation_required_passes} "
                    f"demand_max={scoring.demand_bonus_max:.2f} "
                    f"burn={scoring.emission_burn:.0%}"
                )
            except Exception as e:
                if hasattr(self, "_last_good_scoring"):
                    scoring = self._last_good_scoring
                    self._scoring = scoring
                    bt.logging.info(f"SubnetConfig read failed, using last-known values (burn={scoring.emission_burn:.0%}): {e}")
                else:
                    scoring = ScoringParams()
                    self._scoring = scoring  # hardcoded defaults on first-ever failure
                    bt.logging.info(f"SubnetConfig read failed, no cache, using defaults: {e}")
        else:
            if not hasattr(self, "_last_good_scoring"):
                scoring = ScoringParams()
                self._scoring = scoring

        # Update scorer EMA alpha + throughput power from chain
        self.scorer.ema_alpha = scoring.ema_alpha
        self.scorer.throughput_power = scoring.throughput_power

        self._last_model_emission_budgets = self._build_model_emission_budgets(
            demand_scores,
        )

        # Refresh blacklist from SubnetConfig (parallel RPC per address, cached 5min).
        self._refresh_blacklist({m.address for m in epoch_miners})

        # ── GPU UUID dedup: one GPU = one endpoint ─────────────────
        # Build map: gpu_uuid -> list of (address, model_index, ema_score).
        # If any UUID appears on more than one endpoint, keep the highest-
        # scored, skip the rest.  One physical GPU can only serve one endpoint.
        _uuid_endpoints: Dict[str, List[tuple]] = {}
        for m in epoch_miners:
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
        for miner in epoch_miners:
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
            with self._hard_failure_strike_lock:
                if self._hard_failure_strikes.migrate_index(
                    miner.address,
                    miner.model_index,
                    new_endpoint=getattr(miner, "endpoint", ""),
                ):
                    self._save_hard_failure_strikes_locked()
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
            audit_uid_score_gate_reason = self._capacity_audit_uid_score_gate_reason(
                uid,
                epoch_number,
            )

            if key in self._epoch_close_value(
                "_receipt_pull_failed_keys",
                set(),
            ):
                # The peer receipt endpoint is distribution only. Our signed
                # receipts were persisted before transport and remain
                # authoritative for our exact canary obligations. Continuing
                # here prevents a miner from turning a failed receipt GET into
                # a neutral canary epoch.
                bt.logging.warning(
                    f"Peer receipt view unavailable for "
                    f"{miner.address[:10]} model_index={miner.model_index} "
                    f"at epoch {epoch_number}; enforcing validator-owned "
                    "receipts and exact missing obligations"
                )

            if key in validator_canary_failures:
                gated = self._apply_capacity_audit_model_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    model_gate_reason,
                )
                gated = self._apply_capacity_audit_score_gates(
                    miner.address,
                    miner.model_index,
                    uid,
                    audit_score_gate_reason,
                    audit_uid_score_gate_reason,
                ) or gated
                if not gated:
                    bt.logging.warning(
                        f"Skipping canary score for {miner.address[:10]} "
                        f"model_index={miner.model_index}: validator-side v3 "
                        "configuration or storage failure"
                    )
                continue

            busy_skips_this_epoch = busy_skips.get(key, 0)
            if expected == 0 and busy_skips_this_epoch == 0:
                gated = self._apply_capacity_audit_model_gate(
                    miner.address,
                    miner.model_index,
                    uid,
                    model_gate_reason,
                )
                gated = self._apply_capacity_audit_score_gates(
                    miner.address,
                    miner.model_index,
                    uid,
                    audit_score_gate_reason,
                    audit_uid_score_gate_reason,
                ) or gated
                if not gated:
                    bt.logging.info(f"Skipping score for {miner.address[:10]} model_index={miner.model_index} — 0 canaries dispatched")
                continue

            expected_inventory = dict(
                expected_obligations_by_key.get(key, {})
            )
            hard_ids = self._epoch_close_value(
                "_hard_canary_obligation_ids",
                set(),
            )
            completed_obligations = self._completed_canary_obligations(
                own_receipts,
                expected_inventory,
            )
            missing_low = {
                obligation_id
                for obligation_id, (kind, _target) in expected_inventory.items()
                if kind == "low" and obligation_id not in completed_obligations
            }
            missing_full = {
                obligation_id
                for obligation_id, (kind, _target) in expected_inventory.items()
                if kind == "full" and obligation_id not in completed_obligations
            }
            hard_missing = set(missing_full).intersection(hard_ids)
            received_obligation_ids = {
                bytes(
                    getattr(receipt, "canary_obligation_id", b"") or b""
                )
                for receipt in own_receipts
            }
            absent_hard_obligations = hard_missing.difference(
                received_obligation_ids
            )
            suppress_probation = self._maintenance_grace_active(
                current_epoch=epoch_number,
                action="suppress_probation",
            ) or self._probation_reset_covers_source_epoch(epoch_number)
            prior_full_debt = key in self._full_context_debt
            pending_full = self._pending_cross_epoch_full_obligations(
                epoch_number,
                key,
            )
            full_deferred = False
            if not missing_full and not pending_full:
                self._full_context_debt.pop(key, None)
            elif self._may_defer_full_context_obligations(
                missing_low=missing_low,
                missing_full=missing_full,
                prior_full_debt=prior_full_debt,
                suppress_probation=suppress_probation,
                busy_evidence_covers=(
                    self._busy_evidence_covers_full_obligations(
                        key,
                        missing_full,
                        all_receipts,
                    )
                ),
            ):
                self._full_context_debt[key] = epoch_number
                full_deferred = True
                for obligation_id in missing_full:
                    expected_inventory.pop(obligation_id, None)
                expected = len(expected_inventory)
                bt.logging.info(
                    f"Full-context obligation deferred once for "
                    f"{miner.address[:10]} model_index={miner.model_index}; "
                    "authorized validator-observed work overlaps every "
                    "precommit-busy interval"
                )

            hard_failure_penalty_required = False
            if not full_deferred:
                for obligation_id in absent_hard_obligations:
                    hard_failure_penalty_required = (
                        self._record_hard_audit_failure(
                            source_epoch=epoch_number,
                            miner_address=miner.address,
                            model_id=miner.model_id,
                            model_index=miner.model_index,
                            obligation_id=obligation_id,
                            failure_code="obligation_missing",
                            endpoint=getattr(miner, "endpoint", ""),
                        )
                        or hard_failure_penalty_required
                    )
            neutral_hard_obligation_ids: Set[bytes] = set()
            if hard_missing and not hard_failure_penalty_required:
                neutral_hard_obligation_ids.update(hard_missing)
                for obligation_id in hard_missing:
                    expected_inventory.pop(obligation_id, None)
                expected = len(expected_inventory)

            remaining_missing_full = set(missing_full).difference(
                neutral_hard_obligation_ids
            )
            obligation_failure = bool(
                missing_low
                or (remaining_missing_full and not full_deferred)
            )
            if obligation_failure and not suppress_probation:
                if key not in canary_penalized_keys:
                    self._on_proof_failure(
                        miner.address,
                        miner.model_index,
                        endpoint=getattr(miner, "endpoint", ""),
                        source_epoch=epoch_number,
                    )
                    canary_penalized_keys.add(key)
                bt.logging.info(
                    f"Canary obligation failure for {miner.address[:10]} "
                    f"model_index={miner.model_index}: "
                    f"missing_low={len(missing_low)} "
                    f"missing_full={len(missing_full)} "
                    f"prior_full_debt={prior_full_debt}"
                )

            # Count proof verification outcomes from own receipts
            proof_tested = [
                r for r in own_receipts if r.proof_requested
            ]
            proof_failed = [
                r for r in proof_tested if not r.proof_verified
            ]
            hard_proof_failed = [
                receipt
                for receipt in proof_failed
                if bytes(
                    getattr(receipt, "canary_obligation_id", b"") or b""
                )
                in hard_ids
            ]
            nonhard_proof_failed = [
                receipt
                for receipt in proof_failed
                if receipt not in hard_proof_failed
            ]
            shared_hard_verdict = shared_hard_proof_verdicts.get(key)
            shared_hard_tests = 1 if shared_hard_verdict is not None else 0
            shared_hard_failures = (
                1 if shared_hard_verdict is False else 0
            )
            if shared_hard_verdict is False:
                hard_failure_penalty_required = (
                    self._record_hard_failure_strike(
                        key,
                        source_epoch=epoch_number,
                        endpoint=getattr(miner, "endpoint", ""),
                        # A follower only receives a negative owner snapshot
                        # once the owner's configured threshold was reached.
                        force_penalty=self._proof_v3_follower_mode_active(),
                    )
                    or hard_failure_penalty_required
                )
            if hard_proof_failed:
                # The live owner path normally recorded these immediately.
                # Reconcile again at close so restart/replay is idempotent and
                # cannot lose the strike decision.
                for receipt in hard_proof_failed:
                    hard_failure_penalty_required = (
                        self._record_hard_audit_failure(
                            source_epoch=epoch_number,
                            miner_address=miner.address,
                            model_id=miner.model_id,
                            model_index=miner.model_index,
                            obligation_id=bytes(
                                getattr(
                                    receipt,
                                    "canary_obligation_id",
                                    b"",
                                )
                                or b""
                            ),
                            failure_code="post_precommit_failure",
                            endpoint=getattr(miner, "endpoint", ""),
                        )
                        or hard_failure_penalty_required
                    )
            if (
                (hard_proof_failed or shared_hard_failures)
                and not hard_failure_penalty_required
            ):
                neutral_hard_obligation_ids.update(
                    bytes(
                        getattr(receipt, "canary_obligation_id", b"")
                        or b""
                    )
                    for receipt in hard_proof_failed
                )
                for obligation_id in neutral_hard_obligation_ids:
                    expected_inventory.pop(obligation_id, None)
                expected = len(expected_inventory)
            if (
                not hard_proof_failed
                and shared_hard_verdict is not False
                and (
                    shared_hard_verdict is True
                    or any(
                        bytes(
                            getattr(
                                receipt,
                                "canary_obligation_id",
                                b"",
                            )
                            or b""
                        )
                        in hard_ids
                        for receipt in proof_tested
                    )
                )
            ):
                self._record_hard_failure_clean_pass(
                    key,
                    source_epoch=epoch_number,
                )

            proof_failure_penalty_required = bool(
                nonhard_proof_failed
                or (
                    (hard_proof_failed or shared_hard_failures)
                    and hard_failure_penalty_required
                )
            )

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
                expected_canary_obligations=expected_inventory,
                all_receipts=all_receipts,
                scoring_receipts=select_scoring_authority_receipts(
                    all_receipts,
                    scoring_authority_hotkey,
                ),
                proof_tests=len(proof_tested) + shared_hard_tests,
                proof_failures=len(proof_failed) + shared_hard_failures,
                proof_failure_penalty_required=(
                    proof_failure_penalty_required
                ),
                neutral_hard_obligation_ids=(
                    neutral_hard_obligation_ids
                ),
                tee_tests=len(tee_tested),
                tee_failures=len(tee_failed),
                tee_verified=len(tee_tested) > 0 and len(tee_failed) == 0,
                max_context_len=miner.max_context_len,
                quant=miner.quant,
                quant_qualified=self._proof_v3_quant_qualified(miner),
                busy_skip_count=busy_skips.get(key, 0),
            )

            # Demand bonus for this model
            demand_bonus = 1.0
            if self.config.demand_bonus_enabled:
                model_bps = demand_scores.get(miner.model_id, 0)
                demand_bonus = compute_demand_bonus(
                    model_bps, scoring.demand_bonus_max,
                )

            suppress_score_zeroing = self._maintenance_grace_active(
                current_epoch=epoch_number,
                action="suppress_score_zeroing",
            ) or self._probation_reset_covers_source_epoch(epoch_number)
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
                tee_bonus=scoring.tee_bonus,
                suppress_hard_failures=suppress_score_zeroing,
            )
            gated = self._apply_capacity_audit_model_gate(
                miner.address,
                miner.model_index,
                uid,
                model_gate_reason,
            )
            gated = self._apply_capacity_audit_score_gates(
                miner.address,
                miner.model_index,
                uid,
                audit_score_gate_reason,
                audit_uid_score_gate_reason,
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
                outcome.proof_tests > 0
                and outcome.proof_failures > 0
                and outcome.proof_failure_penalty_required
            )
            had_proof_failure = had_proof_failure or obligation_failure
            if had_proof_failure and not suppress_probation:
                # Enter or reset probation (mid-epoch may have already entered)
                self._probation_tracker.enter_probation(
                    key, epoch_number, endpoint=getattr(miner, 'endpoint', ''))
                self._db.enter_probation(
                    miner.address, miner.model_index, epoch_number,
                    uid=uid if uid is not None else -1,
                    hotkey_ss58=getattr(miner, "hotkey_ss58", "") or "",
                )
            elif self._probation_tracker.is_on_probation(key):
                if self._probation_recovery_epoch_is_clean(
                    obligation_failure=obligation_failure,
                    full_deferred=full_deferred,
                    proof_tests=outcome.proof_tests,
                    proof_failures=outcome.proof_failures,
                ):
                    # All proofs passed this epoch — record clean pass
                    self._record_probation_recovery_source_pass(
                        key,
                        epoch_number,
                    )

            # Escalation: too long on probation → report offline on-chain
            if self._db.should_escalate(miner.address, miner.model_index, epoch_number):
                bt.logging.info(f"Probation ESCALATION for {miner.address[:10]} model_index={miner.model_index} -> reportOffline")
                # Background dispatch — chain wait must not block epoch close
                self._submit_report_offline(miner)

        # ── Zero undiscovered miners ───────────────────────────────
        # If a miner's lease expired or it wasn't discovered this epoch,
        # it's not serving — zero its EMA immediately.  This prevents
        # stale scores from persisting in get_weights() after miners
        # leave the network.  Transient issues (unreachable but still
        # discovered) are handled by the canary error penalty path above.
        _discovered_keys = {
            (m.address.lower(), m.model_index) for m in epoch_miners
        }
        self._full_context_debt = {
            key: debt_epoch
            for key, debt_epoch in self._full_context_debt.items()
            if key in _discovered_keys
        }
        self._save_full_context_debt()
        try:
            self._db.gc_local_service_receipts(
                max(0, int(epoch_number) - 3)
            )
        except Exception as exc:
            bt.logging.warning(
                f"Validator-local receipt GC failed: {exc}"
            )
        try:
            self._db.gc_proof_v3_hard_failures(
                max(0, int(epoch_number) - 3)
            )
        except Exception as exc:
            bt.logging.warning(
                f"Proof-v3 hard failure GC failed: {exc}"
            )
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

        try:
            self._finalize_owner_verdict_snapshot(
                epoch_number=epoch_number,
                miner_receipts=miner_receipts,
            )
        except Exception as exc:
            bt.logging.warning(
                "Verdict snapshot publication failed locally (NOT attributed "
                f"to any miner): {type(exc).__name__}: {exc}"
            )

        # ── Write shared state for proxy (BEFORE on-chain posts) ──
        # Shared state must be written even if on-chain calls fail (429).
        self._write_shared_state()

        # ── Epoch audit log ─────────────────────────────────────
        weight_set = False  # weights are set on a separate boundary
        self._db.log_epoch(
            epoch=epoch_number,
            start_block=int(
                self._epoch_close_value("_epoch_start_block", 0)
            ),
            miner_count=len(epoch_miners),
            receipt_count=len(all_epoch_receipts),
            weight_set=weight_set,
        )
        self._db.set_meta("current_epoch", str(epoch_number))

        try:
            cleanup = self._db.compact_capacity_audit_storage(
                current_epoch=int(epoch_number),
                retain_failure_epochs=int(
                    self._epoch_close_value(
                        "_capacity_audit_cfg",
                        self._capacity_audit_cfg,
                    ).repeat_window_epochs
                ) + 2,
                retain_artifacts=os.environ.get(
                    "VERATHOS_CAPACITY_AUDIT_RETAIN_ARTIFACTS",
                    "",
                ).lower() in {"1", "true", "yes", "on"},
            )
            if any(cleanup.values()):
                bt.logging.info(f"Capacity audit storage cleanup: {cleanup}")
        except Exception as exc:
            bt.logging.warning(f"Capacity audit storage cleanup failed: {exc}")

        # Periodic analytics backup+cleanup (every ~7 days ≈ 140 epochs).
        # Never perform this on the epoch-close thread: the retained tables can
        # contain millions of rows and archiving must not stall chain progress.
        if epoch_number % 140 == 0:
            self._schedule_analytics_archive()

        elapsed = time.monotonic() - t0
        bt.logging.info(f"Epoch {epoch_number} closed in {elapsed:.1f}s")

    def _schedule_analytics_archive(self) -> bool:
        """Start one bounded background analytics archive if none is active."""

        lock = getattr(self, "_analytics_archive_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._analytics_archive_lock = lock
        with lock:
            current = getattr(self, "_analytics_archive_thread", None)
            if current is not None and current.is_alive():
                bt.logging.info(
                    "Analytics archive already active; skipping duplicate schedule"
                )
                return False
            worker = threading.Thread(
                target=self._run_analytics_archive,
                name="validator-analytics-archive",
                daemon=True,
            )
            self._analytics_archive_thread = worker
            worker.start()
            return True

    def _run_analytics_archive(self) -> None:
        """Archive retained analytics without blocking epoch processing."""

        try:
            for table, fn in [
                ("canary results", self._db.backup_and_cleanup_canary_results),
                ("network receipts", self._db.backup_and_cleanup_network_receipts),
            ]:
                archived = fn(retain_days=7)
                if archived > 0:
                    bt.logging.info(f"Analytics: archived {archived} {table}")
        except Exception as exc:
            bt.logging.warning(
                f"Analytics backup+cleanup failed without affecting epoch processing: {exc}"
            )
        finally:
            # Pruning is independent of either table archive.  A failed second
            # archive must not retain old weekly files forever and fill disk.
            if not getattr(self.config, "retain_backups", False):
                backup_dir = os.path.join(
                    os.environ.get(
                        "VERALLM_DATA_DIR",
                        os.path.expanduser("~/.verathos"),
                    ),
                    "backups",
                )
                if os.path.isdir(backup_dir):
                    import glob as _glob

                    cutoff = time.time() - (7 * 86400)
                    for pattern in (
                        "canary_results_*.jsonl.gz",
                        "network_receipts_*.jsonl.gz",
                    ):
                        for path in _glob.glob(os.path.join(backup_dir, pattern)):
                            try:
                                if os.path.getmtime(path) < cutoff:
                                    os.remove(path)
                                    bt.logging.info(
                                        f"Deleted old backup: {os.path.basename(path)}"
                                    )
                            except FileNotFoundError:
                                pass
                    stale_tmp_cutoff = time.time() - 86400
                    for pattern in (
                        "canary_results_*.jsonl.gz.*.tmp",
                        "network_receipts_*.jsonl.gz.*.tmp",
                    ):
                        for path in _glob.glob(os.path.join(backup_dir, pattern)):
                            try:
                                if os.path.getmtime(path) < stale_tmp_cutoff:
                                    os.remove(path)
                                    bt.logging.info(
                                        "Deleted stale analytics temporary file: "
                                        f"{os.path.basename(path)}"
                                    )
                            except FileNotFoundError:
                                pass

    def _shared_hard_prefetch_result(
        self,
        *,
        epoch_number: int,
        miner: ActiveMiner,
        receipt: ServiceReceipt,
    ) -> Optional[Tuple[str, str]]:
        from neurons.proof_v3_shared_hard import (
            shared_hard_receipt_cache_key_v3,
        )

        lock = getattr(self, "_shared_hard_prefetch_lock", None)
        if lock is None:
            return None
        try:
            receipt_key = shared_hard_receipt_cache_key_v3(receipt)
        except ValueError:
            return None
        cache_key = (
            int(epoch_number),
            miner.address.lower(),
            int(miner.model_index),
            receipt_key,
        )
        with lock:
            return self._epoch_close_value(
                "_shared_hard_prefetch_results",
                {},
            ).get(cache_key)

    def _store_shared_hard_prefetch_result(
        self,
        *,
        epoch_number: int,
        miner: ActiveMiner,
        receipt: ServiceReceipt,
        status: str,
        detail: str,
    ) -> None:
        from neurons.proof_v3_shared_hard import (
            shared_hard_receipt_cache_key_v3,
        )

        if status not in ("pass", "peer"):
            return
        try:
            receipt_key = shared_hard_receipt_cache_key_v3(receipt)
        except ValueError:
            return
        cache_key = (
            int(epoch_number),
            miner.address.lower(),
            int(miner.model_index),
            receipt_key,
        )
        with self._shared_hard_prefetch_lock:
            results = self._epoch_close_value(
                "_shared_hard_prefetch_results",
                {},
            )
            existing = results.get(cache_key)
            value = (status, str(detail or ""))
            if existing is not None and existing != value:
                # A deterministic replay cannot legitimately change verdict.
                results[cache_key] = (
                    "peer",
                    "retained hard bundle changed across fetches",
                )
            else:
                results[cache_key] = value

    def _prefetch_shared_hard_bundle(
        self,
        *,
        epoch_number: int,
        miner: ActiveMiner,
        owner_ss58: str,
        inflight_key: Tuple[int, str, int, str],
    ) -> None:
        """Best-effort early replay; only epoch-close receipts authorize use."""

        from neurons.proof_v3_shared_hard import (
            SharedHardBundlePeerFailure,
            SharedHardBundleUnavailable,
            verify_indexed_shared_hard_bundle_v3,
        )
        from verallm.chain.wallet import ss58_decode

        try:
            release = self._epoch_close_value(
                "_proof_v3_releases",
                {},
            ).get(miner.model_id)
            policy = self._epoch_close_value(
                "_proof_v3_canary_policy",
                None,
            )
            model_policy = (
                policy.model_policy(miner.model_id)
                if policy is not None
                else None
            )
            miner_hotkey = str(
                getattr(miner, "hotkey_ss58", "") or ""
            )
            if (
                release is None
                or model_policy is None
                or not miner_hotkey
                or model_policy.execution_profile_digest
                != release.qualified_profile.profile.digest()
                or str(miner.quant or "").strip().lower()
                not in model_policy.allowed_quantizations
            ):
                return
            owner_key = ss58_decode(owner_ss58)
            with ValidatorClient(
                miner_url=miner.endpoint,
                config=Config(block_size=256, spot_checks=25),
                timeout=5.0,
                verify_tls=False,
                validator_hotkey_ss58=self._validator_hotkey_ss58,
                validator_seed=self._validator_private_key,
            ) as client:
                index = client.fetch_proof_v3_hard_bundle_index(
                    epoch_number
                )
                candidates: list[ServiceReceipt] = []
                for entry in index["bundles"]:
                    if not isinstance(entry, dict):
                        continue
                    receipt_data = entry.get("receipt")
                    if not isinstance(receipt_data, dict):
                        continue
                    try:
                        receipt = receipt_from_dict(receipt_data)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        receipt.epoch_number == epoch_number
                        and receipt.validator_hotkey == owner_key
                        and receipt.is_canary is True
                        and receipt.proof_requested is True
                        and receipt.miner_address.lower()
                        == miner.address.lower()
                        and receipt.model_id == miner.model_id
                        and receipt.model_index == miner.model_index
                    ):
                        candidates.append(receipt)
                for receipt in candidates:
                    try:
                        verify_indexed_shared_hard_bundle_v3(
                            client=client,
                            index=index,
                            receipt=receipt,
                            qualified_release=release,
                            hard_auditor_hotkey_ss58=owner_ss58,
                            miner_hotkey_ss58=miner_hotkey,
                            expected_epoch_number=epoch_number,
                            expected_miner_address=miner.address,
                            expected_model_id=miner.model_id,
                            expected_model_index=miner.model_index,
                        )
                    except SharedHardBundleUnavailable:
                        continue
                    except SharedHardBundlePeerFailure as exc:
                        self._store_shared_hard_prefetch_result(
                            epoch_number=epoch_number,
                            miner=miner,
                            receipt=receipt,
                            status="peer",
                            detail=str(exc),
                        )
                    except (TypeError, ValueError) as exc:
                        self._store_shared_hard_prefetch_result(
                            epoch_number=epoch_number,
                            miner=miner,
                            receipt=receipt,
                            status="peer",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    else:
                        self._store_shared_hard_prefetch_result(
                            epoch_number=epoch_number,
                            miner=miner,
                            receipt=receipt,
                            status="pass",
                            detail="",
                        )
        except httpx.HTTPError:
            # The index may not exist yet or the endpoint may be momentarily
            # unavailable. Epoch-close reconciliation remains authoritative.
            pass
        except Exception as exc:
            bt.logging.debug(
                "Shared proof-v3 hard prefetch skipped locally for "
                f"{miner.address[:10]} model_index={miner.model_index}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            with self._shared_hard_prefetch_lock:
                self._shared_hard_prefetch_inflight.discard(inflight_key)

    def _fetch_owner_verdict_snapshot(
        self,
        *,
        current_epoch: int,
        at_close: bool,
    ):
        """Fetch one complete owner decision set or leave the epoch neutral."""

        from neurons.verdict_follower import VerdictSnapshotUnavailable

        follower = self._verdict_snapshot_follower
        if at_close:
            # A stopped feed at close is neutral even when a prefetch wave
            # previously succeeded. Keep the anti-replay floor separately.
            follower.clear_current()
        owner_url = str(
            self._epoch_close_value(
                "_owner_verdict_url_latched",
                "",
            )
            or ""
        ).strip()
        close_state = getattr(
            getattr(self, "_epoch_close_local", None),
            "state",
            None,
        )
        owner_ss58 = str(
            getattr(close_state, "owner_hotkey_ss58", "")
            if close_state is not None
            else getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        # Owner and follower validators observe the same chain boundary. The
        # follower's old-epoch worker polls for a bounded wall-clock interval;
        # this wait never occupies the finalized-block callback or delays the
        # next epoch's canary scheduler. An unavailable owner stays neutral.
        started = time.monotonic()
        timeout_s = (
            float(self._FOLLOWER_OWNER_CLOSE_TIMEOUT_SEC)
            if at_close
            else 0.0
        )
        attempt = 0
        last_error: Exception | None = None
        while True:
            try:
                if not owner_url:
                    raise VerdictSnapshotUnavailable(
                        "--owner-verdict-url is not configured"
                    )
                snapshot = follower.fetch(
                    owner_url,
                    expected_owner_hotkey_ss58=owner_ss58,
                    metagraph=getattr(self, "_metagraph", None),
                    closing_epoch=int(current_epoch),
                    epoch_seconds=(
                        max(1, int(self.config.epoch_blocks)) * 12.0
                    ),
                    timeout=max(
                        1.0,
                        float(
                            getattr(
                                self.config,
                                "subnet_config_timeout_seconds",
                                5.0,
                            )
                            or 5.0
                        ),
                    ),
                )
                if (
                    at_close
                    and int(snapshot.epoch_number) != int(current_epoch)
                ):
                    follower.clear_current()
                    raise VerdictSnapshotUnavailable(
                        "owner feed has not finalized the closing epoch"
                    )
                bt.logging.info(
                    "Owner verdict snapshot accepted: "
                    f"fetched_at={int(time.time())} "
                    f"snapshot_epoch={int(snapshot.epoch_number)} "
                    "owner_signature_verified=true "
                    f"entries={len(snapshot.entries)}"
                )
                return snapshot
            except VerdictSnapshotUnavailable as exc:
                last_error = exc
                if not at_close:
                    break
                elapsed = time.monotonic() - started
                remaining = timeout_s - elapsed
                if remaining <= 0:
                    break
                delay = min(
                    float(self._FOLLOWER_OWNER_CLOSE_RETRY_CAP_SEC),
                    0.25 * (1 << min(attempt, 5)),
                    remaining,
                )
                if delay <= 0:
                    break
                time.sleep(delay)
                attempt += 1
        if at_close:
            follower.clear_current()
            bt.logging.warning(
                "Owner verdict snapshot unavailable or invalid; hard and "
                "capacity decisions are neutral for this close. Check "
                "--owner-verdict-url and owner reachability, or switch to "
                "--proof-v3-verdict-source=verify. "
                f"Reason: {last_error}"
            )
        else:
            bt.logging.debug(
                "Owner verdict snapshot prefetch skipped: "
                f"{last_error}"
            )
        return None

    def _follower_verdict_entry(
        self,
        address: str,
        model_index: int,
    ):
        accepted = self._verdict_snapshot_follower.current
        if accepted is None:
            return None
        key = self._miner_model_key(address, model_index)
        miner = next(
            (
                value
                for value in self._epoch_close_value("_epoch_miners", ())
                if self._miner_model_key(
                    value.address,
                    value.model_index,
                )
                == key
            ),
            None,
        )
        if miner is None:
            return None
        for entry in accepted.snapshot.entries:
            if entry.key != key:
                continue
            if (
                entry.miner_hotkey_ss58
                != str(getattr(miner, "hotkey_ss58", "") or "")
                or entry.model_id != str(miner.model_id)
            ):
                return None
            return entry
        return None

    def _follower_hard_verdicts_at_close(
        self,
        epoch_number: int,
    ) -> Dict[Tuple[str, int], bool]:
        snapshot = self._fetch_owner_verdict_snapshot(
            current_epoch=epoch_number,
            at_close=True,
        )
        if snapshot is None:
            return {}
        verdicts: Dict[Tuple[str, int], bool] = {}
        epoch_miners = self._epoch_close_value("_epoch_miners", ())
        for miner in epoch_miners:
            entry = self._follower_verdict_entry(
                miner.address,
                miner.model_index,
            )
            if entry is None or entry.hard_verdict == -1:
                continue
            verdicts[
                self._miner_model_key(
                    miner.address,
                    miner.model_index,
                )
            ] = entry.hard_verdict == 1
        bt.logging.info(
            "Follower owner verdict apply: "
            f"pass={sum(value is True for value in verdicts.values())} "
            f"failure={sum(value is False for value in verdicts.values())} "
            f"neutral={len(epoch_miners) - len(verdicts)}"
        )
        return verdicts

    def _follower_capacity_gate_reason(
        self,
        address: str,
        model_index: int,
    ) -> str:
        entry = self._follower_verdict_entry(address, model_index)
        if entry is not None and entry.capacity_gated:
            return "capacity-audit owner feed gate"
        return ""

    def _maybe_schedule_shared_hard_prefetch(
        self,
        *,
        block_number: int,
        blocks_into_epoch: int,
    ) -> None:
        """Spread two best-effort external replay waves across the epoch."""

        if not self._proof_v3_allowed():
            return
        owner_ss58 = str(
            getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        if (
            not getattr(
                self.config,
                "proof_v3_hard_auditor_policy_enabled",
                False,
            )
            or not owner_ss58
            or owner_ss58 == self._validator_hotkey_ss58
            or not self._epoch_miners
        ):
            return
        epoch_blocks = max(1, int(self.config.epoch_blocks))
        wave_offsets = {
            max(1, epoch_blocks // 2),
            max(1, (3 * epoch_blocks) // 4),
        }
        offset = int(blocks_into_epoch)
        if offset not in wave_offsets:
            return
        epoch_number = int(block_number) // epoch_blocks
        wave_key = (epoch_number, offset)
        with self._shared_hard_prefetch_lock:
            if wave_key in self._shared_hard_prefetch_waves:
                return
            self._shared_hard_prefetch_waves.add(wave_key)

        if self._proof_v3_follower_mode_active():
            self._shared_hard_prefetch_executor.submit(
                self._fetch_owner_verdict_snapshot,
                current_epoch=epoch_number,
                at_close=False,
            )
            bt.logging.info(
                f"Owner verdict snapshot prefetch wave {offset}/{epoch_blocks}"
            )
            return

        epoch_seed, real_epoch_seed = self._get_chain_block_hash(
            epoch_number * epoch_blocks
        )
        if not real_epoch_seed:
            return
        from neurons.proof_v3_shared_hard import (
            deterministic_shared_hard_order_v3,
        )

        endpoint_keys = tuple(
            (
                miner.address.lower(),
                int(miner.model_index),
                miner.endpoint.rstrip("/"),
            )
            for miner in self._epoch_miners
        )
        try:
            ordered = deterministic_shared_hard_order_v3(
                epoch_number=epoch_number,
                epoch_seed=epoch_seed,
                endpoint_keys=endpoint_keys,
            )
        except ValueError as exc:
            bt.logging.warning(
                "Shared proof-v3 hard prefetch ordering failed locally: "
                f"{exc}"
            )
            return
        miners = {
            (
                miner.address.lower(),
                int(miner.model_index),
                miner.endpoint.rstrip("/"),
            ): miner
            for miner in self._epoch_miners
        }
        submitted = 0
        for endpoint_key in ordered:
            miner = miners[endpoint_key]
            with self._shared_hard_prefetch_lock:
                already_cached = any(
                    key[:3]
                    == (
                        epoch_number,
                        miner.address.lower(),
                        int(miner.model_index),
                    )
                    for key in self._shared_hard_prefetch_results
                )
                if already_cached or endpoint_key in (
                    key[1:] for key in self._shared_hard_prefetch_inflight
                ):
                    continue
                inflight_key = (epoch_number, *endpoint_key)
                self._shared_hard_prefetch_inflight.add(inflight_key)
            self._shared_hard_prefetch_executor.submit(
                self._prefetch_shared_hard_bundle,
                epoch_number=epoch_number,
                miner=miner,
                owner_ss58=owner_ss58,
                inflight_key=inflight_key,
            )
            submitted += 1
        if submitted:
            bt.logging.info(
                f"Shared proof-v3 hard prefetch wave {offset}/{epoch_blocks}: "
                f"scheduled={submitted}"
            )

    def _verify_shared_hard_bundles_at_close(
        self,
        epoch_number: int,
        miner_receipts: Dict[str, List[ServiceReceipt]],
    ) -> Dict[Tuple[str, int], bool]:
        """Replay the configured hard auditor's receipt-matched proof once."""

        if not self._proof_v3_allowed():
            return {}
        owner_ss58 = str(
            getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        if (
            not getattr(
                self.config,
                "proof_v3_hard_auditor_policy_enabled",
                False,
            )
            or not owner_ss58
            or owner_ss58 == self._validator_hotkey_ss58
        ):
            return {}
        if self._proof_v3_follower_mode_active():
            return self._follower_hard_verdicts_at_close(epoch_number)

        from neurons.proof_v3_shared_hard import (
            SharedHardBundlePeerFailure,
            SharedHardBundleUnavailable,
            deterministic_shared_hard_order_v3,
            verify_indexed_shared_hard_bundle_v3,
        )
        from verallm.chain.wallet import ss58_decode

        try:
            owner_key = ss58_decode(owner_ss58)
        except Exception as exc:
            bt.logging.warning(
                "Shared proof-v3 hard verification is locally unavailable: "
                f"configured hard-auditor identity is malformed: {exc}"
            )
            return {}

        work: Dict[
            Tuple[str, int, str],
            tuple[ActiveMiner, tuple[ServiceReceipt, ...]],
        ] = {}
        epoch_miners = self._epoch_close_value("_epoch_miners", ())
        for miner in epoch_miners:
            receipts = tuple(
                receipt
                for receipt in miner_receipts.get(miner.address, ())
                if (
                    receipt.model_id == miner.model_id
                    and receipt.model_index == miner.model_index
                    and receipt.validator_hotkey == owner_key
                    and receipt.is_canary
                    and receipt.proof_requested
                )
            )
            if receipts:
                work[
                    (
                        miner.address.lower(),
                        int(miner.model_index),
                        miner.endpoint.rstrip("/"),
                    )
                ] = (miner, receipts)
        if not work:
            return {}

        epoch_seed, real_epoch_seed = self._get_chain_block_hash(
            int(epoch_number) * int(self.config.epoch_blocks)
        )
        if not real_epoch_seed:
            for miner, _receipts in work.values():
                self._epoch_close_value(
                    "_validator_canary_failures",
                    set(),
                ).add(
                    self._miner_model_key(
                        miner.address,
                        miner.model_index,
                    )
                )
            bt.logging.warning(
                "Shared proof-v3 hard verification is locally unavailable: "
                "the epoch-start chain hash could not be authenticated"
            )
            return {}
        try:
            ordered = deterministic_shared_hard_order_v3(
                epoch_number=epoch_number,
                epoch_seed=epoch_seed,
                endpoint_keys=tuple(work),
            )
        except ValueError as exc:
            bt.logging.warning(
                "Shared proof-v3 hard verification ordering failed locally: "
                f"{exc}"
            )
            return {}

        def verify_one(
            item: tuple[ActiveMiner, tuple[ServiceReceipt, ...]],
        ) -> tuple[str, str]:
            miner, receipts = item
            release = self._epoch_close_value(
                "_proof_v3_releases",
                {},
            ).get(miner.model_id)
            policy = self._epoch_close_value(
                "_proof_v3_canary_policy",
                None,
            )
            model_policy = (
                policy.model_policy(miner.model_id)
                if policy is not None
                else None
            )
            miner_hotkey = str(
                getattr(miner, "hotkey_ss58", "") or ""
            )
            if (
                release is None
                or model_policy is None
                or not miner_hotkey
                or model_policy.execution_profile_digest
                != release.qualified_profile.profile.digest()
                or str(miner.quant or "").strip().lower()
                not in model_policy.allowed_quantizations
            ):
                return "local", "qualified release or endpoint identity unavailable"
            if any(receipt.proof_verified is not True for receipt in receipts):
                return "peer", "hard-auditor receipt records proof failure"

            pending: list[ServiceReceipt] = []
            for receipt in receipts:
                cached = self._shared_hard_prefetch_result(
                    epoch_number=epoch_number,
                    miner=miner,
                    receipt=receipt,
                )
                if cached is None:
                    pending.append(receipt)
                elif cached[0] == "peer":
                    return "peer", cached[1]
            if not pending:
                return "pass", ""

            retry_failure: Exception | None = None
            for attempt in range(3):
                try:
                    with ValidatorClient(
                        miner_url=miner.endpoint,
                        config=Config(block_size=256, spot_checks=25),
                        timeout=max(
                            10.0,
                            float(self.config.epoch_receipt_pull_timeout),
                        ),
                        verify_tls=False,
                        validator_hotkey_ss58=self._validator_hotkey_ss58,
                        validator_seed=self._validator_private_key,
                    ) as client:
                        index = client.fetch_proof_v3_hard_bundle_index(
                            epoch_number
                        )
                        for receipt in pending:
                            verify_indexed_shared_hard_bundle_v3(
                                client=client,
                                index=index,
                                receipt=receipt,
                                qualified_release=release,
                                hard_auditor_hotkey_ss58=owner_ss58,
                                miner_hotkey_ss58=miner_hotkey,
                                expected_epoch_number=epoch_number,
                                expected_miner_address=miner.address,
                                expected_model_id=miner.model_id,
                                expected_model_index=miner.model_index,
                            )
                            self._store_shared_hard_prefetch_result(
                                epoch_number=epoch_number,
                                miner=miner,
                                receipt=receipt,
                                status="pass",
                                detail="",
                            )
                    return "pass", ""
                except (SharedHardBundleUnavailable, httpx.HTTPError) as exc:
                    retry_failure = exc
                    if attempt < 2:
                        time.sleep(0.25 * (1 << attempt))
                    continue
                except SharedHardBundlePeerFailure as exc:
                    return "peer", str(exc)
                except (TypeError, ValueError) as exc:
                    return "peer", f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    return "local", f"{type(exc).__name__}: {exc}"
            assert retry_failure is not None
            # The endpoint returned this signed receipt during the same close
            # pass, so persistent absence after prefetch and bounded retries
            # is a peer retention failure rather than a validator timeout.
            return (
                "peer",
                f"{type(retry_failure).__name__}: {retry_failure}",
            )

        verdicts: Dict[Tuple[str, int], bool] = {}
        local_failures = 0
        peer_failures = 0
        workers = max(
            1,
            min(
                8,
                int(
                    getattr(
                        self.config,
                        "capacity_audit_proof_verify_workers",
                        4,
                    )
                    or 4
                ),
                len(ordered),
            ),
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="proof-v3-shared-hard",
        ) as executor:
            futures = {
                executor.submit(verify_one, work[endpoint_key]): endpoint_key
                for endpoint_key in ordered
            }
            for future in futures:
                endpoint_key = futures[future]
                miner, endpoint_receipts = work[endpoint_key]
                key = self._miner_model_key(
                    miner.address,
                    miner.model_index,
                )
                try:
                    status, detail = future.result()
                except Exception as exc:
                    status, detail = (
                        "local",
                        f"{type(exc).__name__}: {exc}",
                    )
                if status == "pass":
                    verdicts[key] = True
                    for receipt in endpoint_receipts:
                        self._mark_shared_hard_receipt_processed(receipt)
                elif status == "peer":
                    verdicts[key] = False
                    peer_failures += 1
                    for receipt in endpoint_receipts:
                        self._mark_shared_hard_receipt_processed(receipt)
                    bt.logging.info(
                        "Shared proof-v3 hard failure for "
                        f"{miner.address[:10]} model_index={miner.model_index}: "
                        f"{detail}"
                    )
                else:
                    local_failures += 1
                    self._epoch_close_value(
                        "_validator_canary_failures",
                        set(),
                    ).add(key)
                    bt.logging.warning(
                        "Shared proof-v3 hard verification failed locally "
                        f"(NOT attributed to miner {miner.address[:10]} "
                        f"model_index={miner.model_index}): {detail}"
                    )
                bt.logging.info(
                    "Shared proof-v3 hard replay verdict: "
                    f"address={miner.address.lower()} "
                    f"model_index={miner.model_index} verdict={status}"
                )
        bt.logging.info(
            f"Shared proof-v3 hard replay: pass="
            f"{sum(value is True for value in verdicts.values())} "
            f"peer_failure={peer_failures} local_failure={local_failures} "
            f"neutral={len(epoch_miners) - len(work)}"
        )
        return verdicts

    def _verify_late_shared_hard_bundles_at_close(
        self,
        *,
        current_epoch: int,
        receipt_authority: ValidatorAuthority | None,
    ) -> Dict[Tuple[str, int], bool]:
        """Consume unprocessed owner bundles from retained prior epochs.

        Late hard proofs are promoted through a security-only endpoint and do
        not enter the already-closed throughput receipt set. Followers still
        authenticate the embedded owner receipt against the current metagraph
        and independently verify the retained proof during its three-epoch
        retention window.
        """

        if self._proof_v3_follower_mode_active():
            return {}
        if not self._proof_v3_allowed() or receipt_authority is None:
            return {}
        owner_ss58 = str(
            getattr(
                self.config,
                "proof_v3_hard_auditor_hotkey_ss58",
                "",
            )
            or ""
        )
        if (
            not getattr(
                self.config,
                "proof_v3_hard_auditor_policy_enabled",
                False,
            )
            or not owner_ss58
            or owner_ss58 == self._validator_hotkey_ss58
        ):
            return {}

        from neurons.proof_v3_shared_hard import (
            SharedHardBundlePeerFailure,
            SharedHardBundleUnavailable,
            shared_hard_receipt_cache_key_v3,
            verify_indexed_shared_hard_bundle_v3,
        )
        from verallm.chain.wallet import ss58_decode

        try:
            owner_key = ss58_decode(owner_ss58)
        except Exception as exc:
            bt.logging.warning(
                "Late shared proof-v3 replay is locally unavailable: "
                f"configured hard-auditor identity is malformed: {exc}"
            )
            return {}

        first_epoch = max(0, int(current_epoch) - 3)
        source_epochs = range(first_epoch, int(current_epoch))
        receipt_window = max(
            4_500.0,
            4.0 * float(max(1, int(self.config.epoch_blocks))) * 12.0,
        )
        verdicts: Dict[Tuple[str, int], bool] = {}
        local_failures = 0
        for miner in self._epoch_close_value("_epoch_miners", ()):
            release = self._epoch_close_value(
                "_proof_v3_releases",
                {},
            ).get(miner.model_id)
            policy = self._epoch_close_value(
                "_proof_v3_canary_policy",
                None,
            )
            model_policy = (
                policy.model_policy(miner.model_id)
                if policy is not None
                else None
            )
            miner_hotkey = str(
                getattr(miner, "hotkey_ss58", "") or ""
            )
            if (
                release is None
                or model_policy is None
                or not miner_hotkey
                or model_policy.execution_profile_digest
                != release.qualified_profile.profile.digest()
                or str(miner.quant or "").strip().lower()
                not in model_policy.allowed_quantizations
            ):
                continue
            endpoint_key = self._miner_model_key(
                miner.address,
                miner.model_index,
            )
            try:
                with ValidatorClient(
                    miner_url=miner.endpoint,
                    config=Config(block_size=256, spot_checks=25),
                    timeout=max(
                        10.0,
                        float(self.config.epoch_receipt_pull_timeout),
                    ),
                    verify_tls=False,
                    validator_hotkey_ss58=self._validator_hotkey_ss58,
                    validator_seed=self._validator_private_key,
                ) as client:
                    for source_epoch in source_epochs:
                        index = client.fetch_proof_v3_hard_bundle_index(
                            source_epoch
                        )
                        entries = index.get("bundles")
                        if not isinstance(entries, list):
                            raise SharedHardBundlePeerFailure(
                                "shared hard proof index is malformed"
                            )
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            raw_receipt = entry.get("receipt")
                            if not isinstance(raw_receipt, dict):
                                continue
                            try:
                                receipt = receipt_from_dict(raw_receipt)
                            except (KeyError, TypeError, ValueError):
                                continue
                            if (
                                receipt.epoch_number != source_epoch
                                or receipt.validator_hotkey != owner_key
                                or receipt.miner_address.lower()
                                != miner.address.lower()
                                or receipt.model_id != miner.model_id
                                or receipt.model_index != miner.model_index
                                or receipt.is_canary is not True
                                or receipt.proof_requested is not True
                            ):
                                continue
                            if not verify_service_receipt(
                                receipt,
                                source_epoch,
                                authority=receipt_authority,
                                receipt_window_sec=receipt_window,
                            ):
                                continue
                            receipt_key = shared_hard_receipt_cache_key_v3(
                                receipt
                            )
                            if (
                                receipt_key
                                in self._shared_hard_processed_receipts
                            ):
                                continue
                            if receipt.proof_verified is not True:
                                verdicts[endpoint_key] = False
                                self._mark_shared_hard_receipt_processed(
                                    receipt
                                )
                                continue
                            try:
                                verify_indexed_shared_hard_bundle_v3(
                                    client=client,
                                    index=index,
                                    receipt=receipt,
                                    qualified_release=release,
                                    hard_auditor_hotkey_ss58=owner_ss58,
                                    miner_hotkey_ss58=miner_hotkey,
                                    expected_epoch_number=source_epoch,
                                    expected_miner_address=miner.address,
                                    expected_model_id=miner.model_id,
                                    expected_model_index=miner.model_index,
                                )
                            except SharedHardBundleUnavailable:
                                # Promotion may still be racing a completed
                                # proof. Leave it unprocessed for the next
                                # bounded polling wave.
                                continue
                            except SharedHardBundlePeerFailure as exc:
                                verdicts[endpoint_key] = False
                                self._mark_shared_hard_receipt_processed(
                                    receipt
                                )
                                bt.logging.info(
                                    "Late shared proof-v3 hard failure for "
                                    f"{miner.address[:10]} "
                                    f"model_index={miner.model_index}: {exc}"
                                )
                            else:
                                if endpoint_key not in verdicts:
                                    verdicts[endpoint_key] = True
                                self._mark_shared_hard_receipt_processed(
                                    receipt
                                )
            except httpx.HTTPError:
                local_failures += 1
            except SharedHardBundlePeerFailure as exc:
                verdicts[endpoint_key] = False
                bt.logging.info(
                    "Late shared proof-v3 index failure for "
                    f"{miner.address[:10]} model_index={miner.model_index}: "
                    f"{exc}"
                )
            except Exception as exc:
                local_failures += 1
                bt.logging.warning(
                    "Late shared proof-v3 replay failed locally "
                    f"(NOT attributed to miner {miner.address[:10]} "
                    f"model_index={miner.model_index}): "
                    f"{type(exc).__name__}: {exc}"
                )
        if verdicts or local_failures:
            bt.logging.info(
                "Late shared proof-v3 replay: "
                f"pass={sum(value is True for value in verdicts.values())} "
                f"peer_failure="
                f"{sum(value is False for value in verdicts.values())} "
                f"local_failure={local_failures}"
            )
        return verdicts

    def _load_local_epoch_receipts(
        self,
        epoch_number: int,
        receipt_authority: ValidatorAuthority | None,
    ) -> List[ServiceReceipt]:
        """Load and verify receipts signed and persisted by this validator."""

        receipt_dicts = self._db.get_local_service_receipts(epoch_number)
        receipts: List[ServiceReceipt] = []
        for value in receipt_dicts:
            try:
                receipt = receipt_from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise _ProofV3ValidatorConfigurationError(
                    "validator-local service receipt is malformed"
                ) from exc
            if receipt.validator_hotkey != self._validator_hotkey_bytes:
                raise _ProofV3ValidatorConfigurationError(
                    "validator-local service receipt has the wrong signer"
                )
            if not verify_service_receipt(
                receipt,
                epoch_number,
                authority=receipt_authority,
                receipt_window_sec=4 * self._EPOCH_DURATION_SEC,
            ):
                raise _ProofV3ValidatorConfigurationError(
                    "validator-local service receipt authentication failed"
                )
            receipts.append(receipt)
        return receipts

    def _collect_epoch_receipts(
        self,
        epoch_number: int,
        receipt_authority: ValidatorAuthority | None,
    ) -> Tuple[Dict[str, List[ServiceReceipt]], List[ServiceReceipt]]:
        """Pull epoch receipts without sharing workers with canary execution."""
        miner_receipts: Dict[str, List[ServiceReceipt]] = {}
        all_epoch_receipts: List[ServiceReceipt] = []
        pull_failed_keys: Set[Tuple[str, int]] = set()

        local_receipts = self._load_local_epoch_receipts(
            epoch_number,
            receipt_authority,
        )
        seen_sigs_by_addr: Dict[str, set[bytes]] = {}
        for receipt in local_receipts:
            address_key = receipt.miner_address.lower()
            seen_sigs_by_addr.setdefault(address_key, set()).add(
                receipt.validator_signature
            )
            miner_receipts.setdefault(receipt.miner_address, []).append(
                receipt
            )
            all_epoch_receipts.append(receipt)

        epoch_miners = tuple(
            self._epoch_close_value("_epoch_miners", ())
        )
        if not epoch_miners:
            self._set_epoch_close_value(
                "_receipt_pull_failed_keys",
                pull_failed_keys,
            )
            return miner_receipts, all_epoch_receipts

        # Receipt close is a scoring-critical phase. Do not submit these jobs
        # to self._executor: long canaries can occupy every canary worker and
        # leave all receipt pulls queued until the overall timeout fires.
        receipt_workers = max(
            1,
            min(self.config.max_concurrent_verifications, len(epoch_miners)),
        )
        receipt_executor = ThreadPoolExecutor(
            max_workers=receipt_workers,
            thread_name_prefix="receipt-pull",
        )
        receipt_futures = {}
        try:
            for miner in epoch_miners:
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
                len(epoch_miners) // self.config.max_concurrent_verifications * 3 + 10,
            ))
            # Cross-pull signature dedup: when a miner registers multiple
            # model_indices on a single physical server, every endpoint returns
            # the same receipt buffer. Filter by validator_signature so each
            # signed receipt counts once across all pulls for the same address.
            # Multi-server legit operators see no filtering (signatures differ).
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

        self._set_epoch_close_value(
            "_receipt_pull_failed_keys",
            pull_failed_keys,
        )
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
        capture_chain_digest: bytes = b"",
        canary_obligation_id: bytes = b"",
        canary_kind: str = "",
        canary_target_prompt_tokens: int = 0,
        security_only: bool = False,
    ) -> bool:
        """Push a signed scoring or late security-only receipt."""
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
            capture_chain_digest=capture_chain_digest,
            canary_obligation_id=canary_obligation_id,
            canary_kind=canary_kind,
            canary_target_prompt_tokens=canary_target_prompt_tokens,
        )

        receipt_path = (
            "/proof/v3/audit-receipt"
            if security_only
            else "/epoch/receipt"
        )
        url = f"{miner_endpoint.rstrip('/')}{receipt_path}"
        import json as _json
        import time as _time
        from neurons.request_signing import sign_request as _sign
        receipt_dict = receipt_to_dict(receipt)
        try:
            self._db.store_local_service_receipt(receipt_dict)
        except Exception as exc:
            # Local persistence is part of exact obligation accounting. Peer
            # transport may still succeed, but an active epoch with no local
            # authoritative copy is a validator fault and must never be
            # attributed to the miner at close.
            if self._canary_epoch_active(epoch_number):
                self._validator_canary_failures.add(
                    self._miner_model_key(miner_address, model_index)
                )
            bt.logging.warning(
                "Validator-local signed receipt persistence failed "
                f"(NOT attributed to miner {miner_address[:10]} "
                f"model_index={model_index}): {exc}"
            )
        else:
            reconcile = getattr(
                self,
                "_reconcile_capacity_audit_timing_excuses",
                None,
            )
            if callable(reconcile):
                try:
                    reconcile(
                        SimpleNamespace(
                            address=miner_address,
                            model_index=int(model_index),
                            model_id=model_id,
                        ),
                        [receipt],
                        int(epoch_number),
                    )
                except Exception as exc:
                    # The receipt remains durably available for ordinary
                    # epoch-close reconciliation. A local reconciliation
                    # failure is never miner evidence.
                    bt.logging.warning(
                        "Capacity overlap reconciliation failed locally "
                        f"(NOT attributed to miner {miner_address[:10]} "
                        f"model_index={model_index}): {exc}"
                    )
        receipt_body = _json.dumps(receipt_dict).encode("utf-8")
        auth_headers = _sign(
            method="POST", path=receipt_path, body=receipt_body,
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

    def _on_proof_failure(
        self,
        miner_address: str,
        model_index: int,
        endpoint: str = "",
        *,
        source_epoch: int | None = None,
    ):
        """Mid-epoch cutoff: immediately put miner on probation and notify proxy.

        Called as soon as a proof verification fails (not waiting for epoch close).
        Updates shared state so the proxy stops routing organic traffic to this miner.
        Halves EMA score on every failure — geometric decay punishes repeat
        offenders while keeping single failures recoverable.
        """
        key = self._miner_model_key(miner_address, model_index)
        close_epoch = int(
            source_epoch
            if source_epoch is not None
            else self._epoch_close_value(
                "_current_epoch",
                getattr(self, "_current_epoch", 0),
            )
        )
        if self._probation_reset_covers_source_epoch(close_epoch):
            bt.logging.info(
                "Ignoring pre-reset proof-failure consequence for "
                f"{miner_address[:10]} model_index={model_index} "
                f"source_epoch={close_epoch}"
            )
            return
        if self._maintenance_grace_active(action="suppress_probation"):
            bt.logging.info(
                f"Maintenance grace suppressed proof-failure probation for "
                f"{miner_address[:10]} model_index={model_index}: "
                f"{self._maintenance_grace_reason()}"
            )
            return
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
            self._probation_tracker.enter_probation(key, close_epoch, endpoint=endpoint)
            self._db.enter_probation(
                miner_address, model_index, close_epoch,
                uid=_uid, hotkey_ss58=_ss58,
            )
        else:
            self._probation_tracker.record_failure(key)
            self._db.record_failure(miner_address, model_index)

        # Halve EMA score immediately — don't wait for epoch close
        self.scorer.halve_ema(miner_address, model_index)
        self._db.halve_ema(miner_address, model_index)

        # A genuine mid-epoch failure must reach the proxy immediately. During
        # epoch close, however, every entry is processed in one frozen batch
        # and _close_epoch writes the complete state once at the end. Rewriting
        # the full snapshot per failed entry can otherwise stall the next
        # epoch's setup for tens of seconds on a large network.
        close_local = getattr(self, "_epoch_close_local", None)
        closing_batch = bool(
            close_local is not None
            and getattr(close_local, "state", None) is not None
        )
        if not closing_batch:
            self._write_shared_state()
            bt.logging.info(
                f"Mid-epoch cutoff: {miner_address[:10]} "
                f"model_index={model_index} on probation, shared state "
                "updated for proxy"
            )
        else:
            bt.logging.debug(
                f"Close-time cutoff batched for {miner_address[:10]} "
                f"model_index={model_index}; final shared-state publication "
                "will include it"
            )

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

    def _fetch_miner_hardware_status(
        self,
        miner: ActiveMiner,
        timeout_s: float = 5.0,
    ) -> str:
        """Fetch hardware metadata and return a structured best-effort status."""
        try:
            resp = httpx.get(
                f"{miner.endpoint.rstrip('/')}/health",
                timeout=float(timeout_s), verify=False,
            )
            if resp.status_code != 200:
                return "unavailable"
            data = resp.json()
            versions = data.get("proof_protocol_versions")
            miner.proof_protocol_versions = (
                sorted(
                    {
                        version
                        for version in versions
                        if (
                            type(version) is int
                            and version in SUPPORTED_PROOF_PROTOCOL_VERSIONS
                        )
                    }
                )
                if isinstance(versions, list)
                else []
            )
            if "hardware" not in data:
                return "missing"
            valid, hw = _normalize_health_hardware(data.get("hardware"))
            miner.gpu_name = hw["gpu_name"]
            miner.gpu_count = hw["gpu_count"]
            miner.vram_gb = hw["vram_gb"]
            miner.compute_capability = hw["compute_capability"]
            miner.gpu_uuids = hw["gpu_uuids"]
            return "fetched" if valid else "invalid"
        except Exception:
            return "unavailable"

    def _fetch_miner_hardware(self, miner: ActiveMiner) -> bool:
        """Compatibility wrapper for callers that only distinguish invalid metadata."""
        return self._fetch_miner_hardware_status(miner) != "invalid"

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
            for m in self._epoch_close_value("_epoch_miners", ())
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
        for miner in self._epoch_close_value("_epoch_miners", ()):
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
                    self._epoch_close_value(
                        "_scoring",
                        self._scoring,
                    ).demand_bonus_max,
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
        entry = getattr(self, "_ss58_cache", {}).get(miner_address.lower())
        if entry:
            return entry.get(f"{key_type}_ss58", "")
        return ""

    def _authenticated_miner_hotkey_ss58(
        self,
        miner: ActiveMiner,
        *,
        uid: Optional[int] = None,
    ) -> str:
        """Resolve a miner hotkey only from validator-authenticated state."""

        hotkey = str(getattr(miner, "hotkey_ss58", "") or "")
        if not hotkey:
            hotkey = str(self._get_miner_ss58(miner.address, "hotkey") or "")
        if hotkey or uid is None:
            return hotkey
        owner_lookup = getattr(self._db, "get_uid_owner", None)
        if not callable(owner_lookup):
            return ""
        try:
            owner = owner_lookup(int(uid)) or {}
        except Exception:
            return ""
        if (
            str(owner.get("evm_address") or "").lower()
            != str(miner.address).lower()
        ):
            return ""
        return str(owner.get("hotkey_ss58") or "")

    def _retain_epoch_miners_with_authenticated_hotkeys(
        self,
        miners: Sequence[ActiveMiner],
        *,
        epoch_number: int,
    ) -> list[ActiveMiner]:
        """Exclude entries that cannot be bound to an authenticated hotkey."""

        cache = getattr(self, "_ss58_cache", None)
        if cache is None:
            cache = {}
            self._ss58_cache = cache
        authenticated: list[ActiveMiner] = []
        missing = 0
        for miner in miners:
            try:
                cached_uid = self._db.get_uid(miner.address)
            except Exception:
                cached_uid = None
            hotkey = self._authenticated_miner_hotkey_ss58(
                miner,
                uid=cached_uid,
            )
            if not hotkey:
                missing += 1
                continue
            miner.hotkey_ss58 = hotkey
            cache_entry = cache.setdefault(miner.address.lower(), {})
            cache_entry["hotkey_ss58"] = hotkey
            coldkey = str(getattr(miner, "coldkey_ss58", "") or "")
            if coldkey:
                cache_entry["coldkey_ss58"] = coldkey
            authenticated.append(miner)
        if missing:
            bt.logging.error(
                f"Epoch {epoch_number}: excluded {missing} endpoint "
                "entry/entries before scheduling because no authenticated "
                "SS58 hotkey was available"
            )
        return authenticated

    def _stale_uid_identity(
        self,
        miner_address: str,
        uid_val: Optional[int],
        mg,
    ) -> bool:
        """True when a cached EVM→UID mapping points at a recycled UID."""
        if uid_val is None:
            return False
        addr = miner_address.lower()
        try:
            n = int(mg.n.item())
        except Exception:
            n = int(getattr(mg, "n", 0) or 0)
        if uid_val < 0 or uid_val >= n:
            return False
        current_hotkey = str(mg.hotkeys[uid_val] or "")
        try:
            cached_identity = self._db.get_cached_identity(miner_address)
        except Exception:
            cached_identity = {}
        cached_hotkey = str((cached_identity or {}).get("hotkey_ss58") or "")

        def _exclude(owner_addr: str = "") -> bool:
            try:
                ValidatorNeuron._reconcile_uid_owner_identity(
                    self,
                    int(uid_val),
                    current_hotkey,
                    str(owner_addr or "").lower(),
                )
            except Exception as exc:
                bt.logging.warning(
                    f"Failed to retire stale UID owner: uid={uid_val} "
                    f"address={addr[:10]}: {exc}"
                )
            self._stale_uid_addresses.add(addr)
            try:
                self._db.mark_address_inactive(addr)
                self._db.clear_probation_for_address(addr)
                self._db.mark_capacity_audit_address_identity_stale(addr)
            except Exception:
                pass
            tracker = getattr(self, "_probation_tracker", None)
            clear_address = getattr(tracker, "clear_address", None)
            if callable(clear_address):
                clear_address(addr)
            self._ss58_cache.pop(addr, None)
            suffix = f" current_uid_evm={owner_addr[:10]}" if owner_addr else ""
            bt.logging.warning(
                f"Stale miner UID association excluded: address={addr[:10]} "
                f"uid={uid_val} cached_hotkey={cached_hotkey[:8]} "
                f"current_hotkey={current_hotkey[:8]}{suffix}"
            )
            return True

        try:
            contract_owner = None
            owner_lookup = getattr(self._miner_client, "get_registered_evm_for_uid", None)
            if callable(owner_lookup):
                contract_owner = owner_lookup(int(uid_val))
            if contract_owner:
                owner_addr = str(contract_owner).lower()
                if owner_addr != addr:
                    try:
                        try:
                            refreshed_owner = owner_lookup(int(uid_val), refresh=True)
                        except TypeError:
                            refreshed_owner = owner_lookup(int(uid_val))
                    except Exception:
                        refreshed_owner = contract_owner
                    refreshed_addr = str(refreshed_owner or "").lower()
                    if refreshed_addr != addr:
                        return _exclude(refreshed_addr or owner_addr)
                    owner_addr = refreshed_addr

                if cached_hotkey and current_hotkey and cached_hotkey != current_hotkey:
                    try:
                        try:
                            refreshed_owner = owner_lookup(int(uid_val), refresh=True)
                        except TypeError:
                            refreshed_owner = owner_lookup(int(uid_val))
                    except Exception:
                        refreshed_owner = None
                    if not refreshed_owner or str(refreshed_owner).lower() != addr:
                        return _exclude(str(refreshed_owner or "").lower())
                return False
        except Exception as exc:
            bt.logging.debug(
                f"Contract UID owner lookup failed for address={addr[:10]} "
                f"uid={uid_val}: {exc}"
            )
        if not cached_hotkey or not current_hotkey or cached_hotkey == current_hotkey:
            return False
        return _exclude()

    def _reconcile_uid_owner_identity(
        self,
        uid: int,
        hotkey_ss58: str,
        evm_address: str,
    ) -> None:
        """Reconcile mutable validator state with the current reusable UID owner."""
        uid_i = int(uid)
        address = str(evm_address).lower()
        current_epoch = int(getattr(self, "_current_epoch", 0) or 0)
        if current_epoch <= 0:
            current_block = int(getattr(self, "_last_known_block", 0) or 0)
            epoch_blocks = max(1, int(getattr(self.config, "epoch_blocks", 360) or 360))
            current_epoch = current_block // epoch_blocks

        result = self._db.reconcile_uid_owner(
            uid_i,
            str(hotkey_ss58 or ""),
            address,
            current_epoch,
        )
        previous = result.get("previous") or {}
        previous_address = str(
            previous.get("evm_address")
            or result.get("inferred_previous_evm")
            or ""
        ).lower()
        previous_hotkey = str(
            previous.get("hotkey_ss58")
            or result.get("inferred_previous_hotkey")
            or ""
        )
        moved_from_uids = {
            int(moved_uid)
            for moved_uid in (result.get("moved_from_uids") or [])
            if int(moved_uid) != uid_i
        }

        current_owners = self._db.get_uid_owners()
        addresses = set(self._db.get_addresses_for_uid(uid_i))
        if previous_address:
            addresses.add(previous_address)
        for old_address in addresses:
            if not old_address or old_address == address:
                continue
            owns_another_uid = any(
                int(other_uid) != uid_i
                and str(owner.get("evm_address") or "").lower() == old_address
                for other_uid, owner in current_owners.items()
            )
            if owns_another_uid:
                continue
            self._stale_uid_addresses.add(old_address)
            self._db.mark_address_inactive(old_address)
            self._db.clear_probation_for_address(old_address)
            self._db.mark_capacity_audit_address_identity_stale(old_address)
            self._ss58_cache.pop(old_address, None)
            self._probation_tracker.clear_address(old_address)

        if not result.get("changed") and not result.get("address_moved"):
            return

        if moved_from_uids or (
            previous_address == address
            and previous_hotkey != str(hotkey_ss58 or "")
        ):
            self._db.reset_address_identity_state(address)
            self._probation_tracker.clear_address(address)
            self._db.mark_capacity_audit_address_identity_stale(address)

        for moved_uid in moved_from_uids:
            self.scorer.states.pop(moved_uid, None)
            self._blacklisted_uids.discard(moved_uid)
            if isinstance(getattr(self, "_last_weights", None), dict):
                self._last_weights[moved_uid] = 0.0
        self.scorer.states[uid_i] = MinerScoreState(uid=uid_i, address=address)
        if address in getattr(self, "_blacklisted_addresses", set()):
            self._blacklisted_uids.add(uid_i)
        else:
            self._blacklisted_uids.discard(uid_i)
        if isinstance(getattr(self, "_last_weights", None), dict):
            self._last_weights[uid_i] = 0.0
        bt.logging.info(
            f"UID {uid_i} owner changed; reset mutable score state for "
            f"hotkey={str(hotkey_ss58 or '')[:8]} address={address[:10]} "
            f"generation={result.get('generation')}"
        )

    def _enrich_miners_from_metagraph(self, miners: List[ActiveMiner]) -> None:
        """Enrich miners with SS58 keys from metagraph. Updates _ss58_cache.

        Called at startup and at each epoch start. Uses chain UID lookup
        first (authoritative), falls back to DB cache.
        """
        try:
            mg = self._subtensor.metagraph(self.config.netuid)
            self._metagraph = mg
            n = mg.n.item()
            valid_miners: List[ActiveMiner] = []
            for miner in miners:
                uid_val = None
                try:
                    uid_val = self._miner_client.get_associated_uid(miner.address)
                except Exception:
                    uid_val = self._db.get_uid(miner.address)
                if uid_val is not None and 0 <= int(uid_val) < n:
                    cached_identity = self._db.get_cached_identity(miner.address)
                    cached_hotkey = str(
                        (cached_identity or {}).get("hotkey_ss58")
                        or self._get_miner_ss58(miner.address, "hotkey")
                        or ""
                    )
                    current_hotkey = str(mg.hotkeys[int(uid_val)] or "")
                    if cached_hotkey and current_hotkey and cached_hotkey != current_hotkey:
                        try:
                            try:
                                refreshed_uid = self._miner_client.get_associated_uid(
                                    miner.address,
                                    refresh=True,
                                )
                            except TypeError:
                                refreshed_uid = self._miner_client.get_associated_uid(
                                    miner.address
                                )
                            if refreshed_uid is not None:
                                uid_val = refreshed_uid
                        except Exception:
                            pass
                if uid_val is not None and 0 <= int(uid_val) < int(n):
                    uid_val = int(uid_val)
                    if self._stale_uid_identity(miner.address, uid_val, mg):
                        continue
                    self._stale_uid_addresses.discard(miner.address.lower())
                    miner.hotkey_ss58 = mg.hotkeys[uid_val]
                    miner.coldkey_ss58 = mg.coldkeys[uid_val] if hasattr(mg, 'coldkeys') else ""
                    try:
                        self._reconcile_uid_owner_identity(
                            int(uid_val),
                            miner.hotkey_ss58,
                            miner.address,
                        )
                    except Exception as exc:
                        bt.logging.warning(
                            f"UID owner reconciliation failed: address={miner.address[:10]} "
                            f"uid={uid_val}: {exc}; excluding this miner refresh"
                        )
                        continue
                    self._db.set_uid(miner.address, uid_val)
                    # Preserve the authoritative lookup across the subsequent
                    # discovery-policy upsert. A first-seen ineligible entry
                    # does not yet have a DB row for set_uid() to update.
                    miner.bittensor_uid = uid_val
                    # Update cache
                    self._ss58_cache[miner.address.lower()] = {
                        "hotkey_ss58": miner.hotkey_ss58,
                        "coldkey_ss58": miner.coldkey_ss58,
                    }
                valid_miners.append(miner)
            miners[:] = valid_miners
        except Exception as e:
            bt.logging.debug(f"Metagraph enrichment failed: {e}")

    def _submit_report_offline(self, miner: ActiveMiner) -> bool:
        """Queue one bounded, deduplicated reportOffline side effect."""

        executor = getattr(self, "_report_offline_executor", None)
        lock = getattr(self, "_report_offline_lock", None)
        slots = getattr(self, "_report_offline_slots", None)
        inflight = getattr(self, "_report_offline_inflight", None)
        # Narrow unit fixtures created with __new__ retain the old synchronous
        # executor seam. Normal validator instances always use the isolated
        # bounded pool initialized in __init__.
        if executor is None or lock is None or slots is None or inflight is None:
            self._control_executor.submit(self._report_offline, miner)
            return True

        key = (str(miner.address).lower(), int(miner.model_index))
        with lock:
            if key in inflight:
                return False
            if not slots.acquire(blocking=False):
                bt.logging.warning(
                    "reportOffline queue is saturated; skipping best-effort "
                    f"chain report for {miner.address[:10]} "
                    f"model_index={miner.model_index}"
                )
                return False
            inflight.add(key)

        def _run() -> None:
            try:
                self._report_offline(miner)
            finally:
                with lock:
                    inflight.discard(key)
                slots.release()

        try:
            executor.submit(_run)
        except Exception:
            with lock:
                inflight.discard(key)
            slots.release()
            raise
        return True

    def _report_offline(self, miner: ActiveMiner):
        """Report a miner-model entry as offline."""
        if self._evm_disabled:
            bt.logging.debug(
                f"EVM disabled — skipping reportOffline for {miner.address[:10]} "
                f"model_index={miner.model_index} (other validators + 24h lease handle it)"
            )
            return
        if self._maintenance_grace_active(action="suppress_report_offline"):
            bt.logging.info(
                f"Maintenance grace suppressed reportOffline for {miner.address[:10]} "
                f"model_index={miner.model_index}: {self._maintenance_grace_reason()}"
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

    def _miner_debug_network_snapshot(self) -> Dict[int, dict]:
        """Copy already-cached weight and metagraph values for diagnostics."""
        try:
            last_weights = dict(getattr(self, "_last_weights", {}) or {})
        except RuntimeError:
            last_weights = {}

        mg = getattr(self, "_metagraph", None)
        try:
            n_value = getattr(mg, "n", 0) if mg is not None else 0
            metagraph_size = int(n_value.item() if hasattr(n_value, "item") else n_value)
        except (TypeError, ValueError, AttributeError):
            metagraph_size = 0

        def _metric(uid: int, *names: str) -> Optional[float]:
            if mg is None:
                return None
            for name in names:
                try:
                    values = getattr(mg, name)
                    value = float(values[uid])
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
            return None

        def _hotkey(uid: int) -> str:
            if mg is None:
                return ""
            try:
                return str(mg.hotkeys[uid] or "")
            except (AttributeError, IndexError, TypeError):
                return ""

        metagraph_block = None
        if mg is not None:
            try:
                block_value = getattr(mg, "block")
                metagraph_block = int(
                    block_value.item() if hasattr(block_value, "item") else block_value
                )
            except (AttributeError, TypeError, ValueError):
                pass

        uids = set(range(max(0, metagraph_size)))
        for uid in last_weights:
            try:
                uids.add(int(uid))
            except (TypeError, ValueError):
                continue

        result: Dict[int, dict] = {}
        for uid in sorted(uids):
            raw_weight = last_weights.get(uid, last_weights.get(str(uid)))
            try:
                weight = float(raw_weight) if raw_weight is not None else None
            except (TypeError, ValueError):
                weight = None
            if weight is not None and not math.isfinite(weight):
                weight = None
            result[uid] = {
                "last_validator_weight": weight,
                "metagraph_hotkey_ss58": _hotkey(uid),
                "metagraph_incentive": _metric(uid, "I", "incentive"),
                "metagraph_emission": _metric(uid, "E", "emission"),
                "metagraph_trust": _metric(uid, "trust", "T"),
                "metagraph_consensus": _metric(uid, "C", "consensus"),
                "metagraph_block": metagraph_block,
            }
        return result

    def _finalize_owner_verdict_snapshot(
        self,
        *,
        epoch_number: int,
        miner_receipts: Dict[str, List[ServiceReceipt]],
    ) -> bool:
        """Sign and persist the owner's complete current decision set."""

        if not _proof_v3_hard_auditor_active(
            self.config,
            getattr(self, "_validator_hotkey_ss58", ""),
        ):
            return False
        from neurons.verdict_records import (
            VerdictSnapshotEntryV1,
            VerdictSnapshotV1,
            canonical_verdict_entries_v1,
            sign_verdict_snapshot_v1,
            verify_verdict_snapshot_v1,
        )

        existing_hex = str(getattr(self, "_verdict_snapshot_hex", "") or "")
        if existing_hex:
            try:
                existing = VerdictSnapshotV1.from_bytes(
                    bytes.fromhex(existing_hex)
                )
                if (
                    existing.epoch_number == int(epoch_number)
                    and verify_verdict_snapshot_v1(
                        existing,
                        expected_auditor_hotkey_ss58=(
                            self._validator_hotkey_ss58
                        ),
                    )
                ):
                    # Finalized epoch bytes are immutable and survive restart.
                    return True
            except Exception:
                pass

        published_failures: set[Tuple[str, int]] = set()
        try:
            for failure in self._db.get_proof_v3_hard_failures(
                int(epoch_number),
                int(epoch_number),
            ):
                published_failures.add(
                    self._miner_model_key(
                        str(failure.get("miner_address") or ""),
                        int(failure.get("model_index", -1)),
                    )
                )
        except Exception as exc:
            raise RuntimeError(
                "owner hard-failure state is unavailable"
            ) from exc

        entries: list[VerdictSnapshotEntryV1] = []
        omitted_identity_count = 0
        snapshot_miners = list(self._epoch_close_value("_epoch_miners", ()))
        seen_snapshot_keys = {
            self._miner_model_key(miner.address, miner.model_index)
            for miner in snapshot_miners
        }
        for miner in self._epoch_close_value(
            "_endpoint_policy_excluded_miners", ()
        ):
            key = self._miner_model_key(miner.address, miner.model_index)
            if key not in seen_snapshot_keys:
                snapshot_miners.append(miner)
                seen_snapshot_keys.add(key)
        endpoint_policy_reasons = dict(
            self._epoch_close_value("_endpoint_policy_gate_reasons", {}) or {}
        )
        for miner in snapshot_miners:
            key = self._miner_model_key(
                miner.address,
                miner.model_index,
            )
            own_hard_receipts = [
                receipt
                for receipt in miner_receipts.get(miner.address, ())
                if (
                    receipt.epoch_number == int(epoch_number)
                    and receipt.model_id == miner.model_id
                    and receipt.model_index == miner.model_index
                    and receipt.validator_hotkey
                    == self._validator_hotkey_bytes
                    and receipt.is_canary
                    and receipt.proof_requested
                )
            ]
            raw_hard_failure = bool(
                key in published_failures
                or any(
                    receipt.proof_verified is not True
                    for receipt in own_hard_receipts
                )
            )
            if raw_hard_failure:
                # The raw failure remains owner-signed and retained.  A first
                # strike is operationally neutral, so followers receive no
                # negative scoring verdict until the configured threshold.
                hard_verdict = (
                    0
                    if self._hard_failure_penalty_required(key)
                    else -1
                )
            elif own_hard_receipts:
                hard_verdict = 1
            else:
                hard_verdict = -1

            uid = self._resolve_uid(miner.address)
            miner_hotkey_ss58 = self._authenticated_miner_hotkey_ss58(
                miner,
                uid=uid,
            )
            if not miner_hotkey_ss58:
                omitted_identity_count += 1
                continue
            capacity_gated = bool(
                key in endpoint_policy_reasons
                or self._capacity_audit_score_gate_reason(
                    miner.address,
                    miner.model_index,
                    epoch_number,
                    uid=uid,
                )
                or self._capacity_audit_uid_score_gate_reason(
                    uid,
                    epoch_number,
                )
            )
            entries.append(
                VerdictSnapshotEntryV1(
                    miner_address=str(miner.address).lower(),
                    model_index=int(miner.model_index),
                    miner_hotkey_ss58=miner_hotkey_ss58,
                    model_id=str(miner.model_id),
                    hard_verdict=hard_verdict,
                    hard_source_epoch=(
                        int(epoch_number) if hard_verdict != -1 else -1
                    ),
                    capacity_gated=capacity_gated,
                    probation=self._probation_tracker.is_on_probation(key),
                )
            )
        if omitted_identity_count:
            bt.logging.warning(
                "VerdictSnapshotV1 omitted "
                f"{omitted_identity_count} endpoint(s) whose validator-owned "
                "SS58 identity could not be authenticated"
            )
        snapshot = sign_verdict_snapshot_v1(
            VerdictSnapshotV1(
                auditor_hotkey=bytes(self._validator_hotkey_bytes),
                epoch_number=int(epoch_number),
                generated_at=int(time.time()),
                entries=canonical_verdict_entries_v1(entries),
            ),
            self._validator_private_key,
        )
        encoded = snapshot.to_bytes().hex()
        self._db.set_meta("gleipnir_verdict_snapshot_v1", encoded)
        history = dict(getattr(self, "_verdict_snapshot_history", {}) or {})
        history[int(epoch_number)] = encoded
        history = dict(sorted(history.items())[-4:])
        self._db.set_meta(
            "gleipnir_verdict_snapshots_v1",
            json.dumps(
                {str(epoch): value for epoch, value in history.items()},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self._verdict_snapshot_history = history
        self._verdict_snapshot_hex = encoded
        bt.logging.info(
            f"VerdictSnapshotV1 finalized: epoch={epoch_number} "
            f"entries={len(snapshot.entries)} bytes={len(snapshot.to_bytes())}"
        )
        return True

    def _schedule_miner_debug_refresh(
        self,
        *,
        current_epoch: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """Refresh the optional diagnostics file without blocking validator work."""
        if not bool(getattr(self.config, "miner_debug_enabled", False)):
            return False
        epoch = int(current_epoch if current_epoch is not None else self._current_epoch)
        if epoch <= 0:
            return False
        refresh_s = max(
            1.0,
            float(getattr(self.config, "miner_debug_refresh_seconds", 60.0) or 60.0),
        )
        now = time.monotonic()
        with self._miner_debug_cache_lock:
            if self._miner_debug_refresh_in_flight:
                return False
            if not force and now - self._miner_debug_last_scheduled_at < refresh_s:
                return False
            self._miner_debug_refresh_in_flight = True
            self._miner_debug_last_scheduled_at = now

        stale_addresses = set(getattr(self, "_stale_uid_addresses", set()) or set())
        blacklist = set(getattr(self, "_blacklisted_addresses", set()) or set())

        def _refresh() -> None:
            try:
                enforcement_enabled = self._capacity_audit_enforcement_enabled(epoch)
                suppression_reason = ""
                if not enforcement_enabled:
                    if self._maintenance_grace_active(
                        current_epoch=epoch,
                        action="suppress_capacity_score_gate",
                    ):
                        suppression_reason = self._maintenance_grace_reason()
                    elif not bool(getattr(self, "_subnet_runtime_config_authoritative", False)):
                        suppression_reason = "hosted subnet config is unavailable or invalid"
                if getattr(self, "_capacity_audit_verifier_unhealthy", False):
                    enforcement_enabled = False
                    suppression_reason = "capacity proof verifier is unhealthy"

                model_gate_reasons: Dict[Tuple[str, int], str] = {}
                for miner in list(getattr(self, "_epoch_miners", []) or []):
                    reason = self._capacity_audit_model_gate_reason(miner, epoch)
                    if reason:
                        model_gate_reasons[
                            (str(miner.address).lower(), int(miner.model_index))
                        ] = reason
                endpoint_gate_reasons = dict(
                    getattr(self, "_endpoint_policy_gate_reasons", {}) or {}
                )
                network_state = ValidatorNeuron._miner_debug_network_snapshot(self)
                snapshot = self._db.build_miner_debug_snapshots(
                    current_epoch=epoch,
                    capacity_audit_cfg=self._capacity_audit_cfg,
                    stale_addresses=stale_addresses,
                    blacklisted_addresses=blacklist,
                    model_gate_reasons=model_gate_reasons,
                    endpoint_gate_reasons=endpoint_gate_reasons,
                    capacity_audit_gate_enforced=enforcement_enabled,
                    capacity_audit_gate_suppression_reason=suppression_reason,
                    uid_network_state=network_state,
                    epoch_seconds=max(
                        1,
                        int(getattr(self.config, "epoch_blocks", 360) or 360) * 12,
                    ),
                )
                from neurons.shared_state import write_miner_debug_state
                write_miner_debug_state(
                    snapshot,
                    str(getattr(self.config, "miner_debug_state_path", "") or ""),
                )
            except Exception as exc:
                bt.logging.warning(f"Failed to refresh miner debug state: {exc}")
            finally:
                with self._miner_debug_cache_lock:
                    self._miner_debug_refresh_in_flight = False

        try:
            self._miner_debug_executor.submit(_refresh)
        except Exception:
            with self._miner_debug_cache_lock:
                self._miner_debug_refresh_in_flight = False
            return False
        return True

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
        stale_addresses = {
            str(a).lower()
            for a in getattr(self, "_stale_uid_addresses", set())
            if str(a or "").strip()
        }
        shared.stale_miner_addresses = sorted(stale_addresses)
        try:
            shared.proof_v3_hard_failures = (
                self._db.get_proof_v3_hard_failures(
                    max(0, int(self._current_epoch) - 3),
                    int(self._current_epoch),
                )
            )
        except Exception as exc:
            bt.logging.warning(
                f"Failed to export proof-v3 hard outcomes: {exc}"
            )
            shared.proof_v3_hard_failures = []
        shared.verdict_snapshot = (
            str(getattr(self, "_verdict_snapshot_hex", "") or "")
            if _proof_v3_hard_auditor_active(
                self.config,
                getattr(self, "_validator_hotkey_ss58", ""),
            )
            else ""
        )
        shared.verdict_snapshots = (
            {
                str(epoch): str(snapshot)
                for epoch, snapshot in (
                    getattr(self, "_verdict_snapshot_history", {}) or {}
                ).items()
            }
            if _proof_v3_hard_auditor_active(
                self.config,
                getattr(self, "_validator_hotkey_ss58", ""),
            )
            else {}
        )
        # Build ss58_map with UIDs so the proxy can resolve UIDs for
        # miners not in miner_endpoints (e.g. inactive/unreachable).
        uid_map_all = self._db.get_all_uids()
        ss58_with_uid: Dict[str, Dict[str, str]] = {}
        for addr, info in self._ss58_cache.items():
            if addr in stale_addresses:
                continue
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
            if m.address.lower() not in stale_addresses
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
        uid_owners = self._db.get_uid_owners()
        loaded = 0
        for (address, model_index), data in saved.items():
            uid = uid_map.get(address)
            if uid is None:
                continue
            owner = uid_owners.get(int(uid))
            if (
                owner is not None
                and str(owner.get("evm_address") or "").lower() != address.lower()
            ):
                continue

            if uid not in self.scorer.states:
                self.scorer.states[uid] = MinerScoreState(uid=uid, address=address)

            state = self.scorer.states[uid]
            if state.address.lower() != address.lower():
                continue
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
        try:
            block, _hash, real = self._get_current_head_with_timeout(15.0)
            if real and block > 0:
                return block
            return self._last_known_block
        except _FuturesTimeout:
            bt.logging.warning("get_current_block timed out (15s) — reconnecting")
            return self._last_known_block
        except Exception:
            return self._last_known_block

    def _get_current_head_with_timeout(
        self,
        timeout_seconds: float,
    ) -> tuple[int, bytes | None, bool]:
        """Bound a potentially stuck SDK head lookup without waiting on exit.

        ``ThreadPoolExecutor`` waits for its worker when used as a context
        manager, even after ``Future.result(timeout=...)`` raises. Keep the
        shutdown explicitly non-blocking and close the stale RPC connection
        so a timed-out worker can unwind in the background.
        """

        pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="validator-head-lookup",
        )
        future = pool.submit(self._get_current_head_block_and_hash)
        try:
            return future.result(timeout=max(0.001, float(timeout_seconds)))
        except _FuturesTimeout:
            future.cancel()
            stale = getattr(self, "_ValidatorNeuron__subtensor", None)
            self.__subtensor = None
            if stale is not None:
                self._close_subtensor(stale)
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    _cached_metagraph_line: str = ""

    def _schedule_metagraph_stats_refresh(self) -> bool:
        """Refresh informational metagraph stats off the block callback."""

        executor = getattr(self, "_metagraph_stats_executor", None)
        lock = getattr(self, "_metagraph_stats_lock", None)
        if executor is None or lock is None:
            # Narrow fixtures constructed without __init__ retain deterministic
            # behavior without weakening the production non-blocking path.
            self._refresh_metagraph_stats()
            return True
        with lock:
            if bool(getattr(self, "_metagraph_stats_in_flight", False)):
                return False
            self._metagraph_stats_in_flight = True

        def _refresh() -> None:
            started = time.monotonic()
            try:
                self._refresh_metagraph_stats()
            finally:
                elapsed = time.monotonic() - started
                if elapsed > 12.0:
                    bt.logging.warning(
                        f"Background metagraph refresh took {elapsed:.1f}s; "
                        "block processing remained independent"
                    )
                with lock:
                    self._metagraph_stats_in_flight = False

        try:
            executor.submit(_refresh)
        except Exception:
            with lock:
                self._metagraph_stats_in_flight = False
            return False
        return True

    def _refresh_metagraph_stats(self) -> None:
        """Format the latest authoritative metagraph without issuing RPC."""
        try:
            mg = self._metagraph
            if mg is None:
                return

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
        for attempt in range(1, max_attempts + 1):
            try:
                bt.logging.debug(f"get_current_head_block attempt {attempt}/{max_attempts}...")
                block, _hash, real = self._get_current_head_with_timeout(30.0)
                if real and block > 0:
                    bt.logging.debug(f"get_current_head_block: block={block}")
                    return block
            except _FuturesTimeout:
                bt.logging.warning(
                    f"get_current_head_block timed out after 30s (attempt {attempt}/{max_attempts}) — "
                    f"Subtensor WS may be hanging",
                )
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
        number = _capacity_audit_head_number(block_header)
        return number if number > 0 else None

    def _ensure_isolated_capacity_audit_head_state(self) -> None:
        if not hasattr(self, "_capacity_audit_head_process"):
            self._capacity_audit_head_process = None
            self._capacity_audit_head_shared = None

    def _start_isolated_capacity_audit_head_observer(self) -> None:
        """Start the sole process-isolated receipt chronology observer."""

        if not self._capacity_audit_cfg.enabled:
            return
        self._ensure_isolated_capacity_audit_head_state()
        process = self._capacity_audit_head_process
        if process is not None and process.poll() is None:
            return

        runtime_dir = os.path.join(
            os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
            "run",
        )
        os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
        state_path = os.path.join(
            runtime_dir,
            f"capacity-head-{os.getpid()}.state",
        )
        try:
            os.unlink(state_path)
        except FileNotFoundError:
            pass
        state_fd = os.open(
            state_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.ftruncate(state_fd, CAPACITY_HEAD_STATE_SIZE)
        parent_read_fd, parent_write_fd = os.pipe()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "neurons.capacity_audit_head_observer",
                    "--network",
                    str(self.config.subtensor_network),
                    "--state-path",
                    state_path,
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-fd",
                    str(parent_read_fd),
                    "--poll-seconds",
                    str(self._block_stream_fallback_poll_s()),
                    "--watchdog-seconds",
                    str(self._block_stream_watchdog_s()),
                ],
                close_fds=True,
                pass_fds=(parent_read_fd,),
            )
        except Exception:
            os.close(parent_read_fd)
            os.close(parent_write_fd)
            os.close(state_fd)
            os.unlink(state_path)
            raise
        os.close(parent_read_fd)
        self._capacity_audit_head_shared = SimpleNamespace(
            fd=state_fd,
            path=state_path,
            parent_write_fd=parent_write_fd,
        )
        self._capacity_audit_head_process = process

        deadline = time.monotonic() + max(15.0, self._block_stream_watchdog_s())
        while time.monotonic() < deadline:
            observation = self._capture_isolated_capacity_audit_ingress_head()
            if observation is not None and observation.trustworthy:
                bt.logging.info(
                    "Capacity receipt chronology observer ready: "
                    f"pid={process.pid} block={observation.block}"
                )
                return
            if process.poll() is not None:
                break
            time.sleep(0.1)
        self._stop_isolated_capacity_audit_head_observer()
        raise RuntimeError(
            "process-isolated capacity receipt chronology observer did not become ready"
        )

    def _stop_isolated_capacity_audit_head_observer(self) -> None:
        self._ensure_isolated_capacity_audit_head_state()
        process = self._capacity_audit_head_process
        shared = self._capacity_audit_head_shared
        if shared is not None:
            try:
                os.close(int(shared.parent_write_fd))
            except OSError:
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if shared is not None:
            try:
                os.close(int(shared.fd))
            except OSError:
                pass
            try:
                os.unlink(str(shared.path))
            except FileNotFoundError:
                pass
        self._capacity_audit_head_process = None
        self._capacity_audit_head_shared = None

    def _capture_isolated_capacity_audit_ingress_head(
        self,
    ) -> Optional[_CapacityAuditIngressHead]:
        """Read one consistent lock-free snapshot from the observer process."""

        self._ensure_isolated_capacity_audit_head_state()
        shared = self._capacity_audit_head_shared
        process = self._capacity_audit_head_process
        if shared is None or process is None:
            return None
        alive = process.poll() is None
        for _attempt in range(8):
            payload = os.pread(
                int(shared.fd),
                CAPACITY_HEAD_STATE_SIZE,
                0,
            )
            if len(payload) != CAPACITY_HEAD_STATE_SIZE:
                continue
            (
                before,
                block,
                observed_wall_ns,
                observed_mono_ns,
                connected,
                after,
            ) = struct.unpack("<QQQQI4xQ", payload)
            if before == after and not (int(after) & 1):
                break
        else:
            return _CapacityAuditIngressHead(
                block=0,
                generation=0,
                observed_at=0.0,
                source="isolated_head_stream",
                trustworthy=False,
            )
        age_s = (
            max(0.0, (time.monotonic_ns() - observed_mono_ns) / 1_000_000_000)
            if observed_mono_ns
            else float("inf")
        )
        return _CapacityAuditIngressHead(
            block=block,
            generation=after // 2,
            observed_at=observed_wall_ns / 1_000_000_000,
            source="isolated_head_stream",
            trustworthy=bool(
                alive
                and block > 0
                and connected
                and age_s <= self._block_stream_watchdog_s()
            ),
        )

    def _ensure_capacity_audit_head_state(self) -> None:
        """Initialize head state for narrow fixtures that bypass ``__init__``."""

        if not hasattr(self, "_capacity_audit_head_lock"):
            self._capacity_audit_head_lock = threading.Lock()
            self._capacity_audit_head_block = 0
            self._capacity_audit_head_generation = 0
            self._capacity_audit_head_observed_at = 0.0
            self._capacity_audit_head_observed_monotonic = 0.0
            self._capacity_audit_head_source = ""
            self._capacity_audit_head_connected = False

    def _begin_capacity_audit_head_generation(self) -> int:
        self._ensure_capacity_audit_head_state()
        with self._capacity_audit_head_lock:
            self._capacity_audit_head_generation += 1
            generation = int(self._capacity_audit_head_generation)
            self._capacity_audit_head_connected = False
        return generation

    def _record_capacity_audit_live_head(
        self,
        block: int,
        *,
        generation: Optional[int] = None,
        source: str,
    ) -> None:
        """Publish one current-head observation without any receipt-side RPC."""

        block_i = max(0, int(block))
        if block_i <= 0:
            return
        self._ensure_capacity_audit_head_state()
        now_wall = time.time()
        now_mono = time.monotonic()
        with self._capacity_audit_head_lock:
            current_generation = int(self._capacity_audit_head_generation)
            if generation is not None and int(generation) != current_generation:
                return
            if block_i < int(self._capacity_audit_head_block):
                return
            self._capacity_audit_head_block = block_i
            self._capacity_audit_head_observed_at = now_wall
            self._capacity_audit_head_observed_monotonic = now_mono
            self._capacity_audit_head_source = str(source)
            self._capacity_audit_head_connected = True

    def _end_capacity_audit_head_generation(self, generation: int) -> None:
        self._ensure_capacity_audit_head_state()
        with self._capacity_audit_head_lock:
            if int(generation) == int(self._capacity_audit_head_generation):
                self._capacity_audit_head_connected = False

    def _capture_capacity_audit_ingress_head(self) -> _CapacityAuditIngressHead:
        """Snapshot trustworthy chronology at completed-body HTTP ingress."""

        isolated = self._capture_isolated_capacity_audit_ingress_head()
        if isolated is not None:
            # Once configured, the isolated observer is authoritative. Never
            # hide its failure by falling back to a thread that may share the
            # exact GIL starvation condition this boundary is meant to avoid.
            return isolated

        self._ensure_capacity_audit_head_state()
        with self._capacity_audit_head_lock:
            block = int(self._capacity_audit_head_block)
            generation = int(self._capacity_audit_head_generation)
            observed_at = float(self._capacity_audit_head_observed_at)
            observed_mono = float(self._capacity_audit_head_observed_monotonic)
            source = str(self._capacity_audit_head_source)
            connected = bool(self._capacity_audit_head_connected)
        age_s = max(0.0, time.monotonic() - observed_mono) if observed_mono else float("inf")
        trustworthy = bool(
            block > 0
            and connected
            and age_s <= self._block_stream_watchdog_s()
        )
        return _CapacityAuditIngressHead(
            block=block,
            generation=generation,
            observed_at=observed_at,
            source=source,
            trustworthy=trustworthy,
        )

    def _observe_live_capacity_audit_head(self) -> tuple[int, bool]:
        """Return the stream-owned head cache; never read RPC from a worker."""

        observation = self._capture_capacity_audit_ingress_head()
        return int(observation.block), bool(observation.trustworthy)

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
        dispatch_lock = getattr(self, "_block_dispatch_lock", None)
        if dispatch_lock is None:
            # Compatibility for narrow unit fixtures and state restored from a
            # process created before this guard existed.
            dispatch_lock = threading.Lock()
            self._block_dispatch_lock = dispatch_lock
        for block_num in range(last_block + 1, current_block + 1):
            if not self._running:
                break
            with dispatch_lock:
                highest = int(
                    getattr(self, "_highest_dispatched_block", -1) or -1
                )
                if block_num <= highest:
                    last_block = max(last_block, block_num)
                    continue
                # Claim before executing. The existing stream loop advances
                # past a block even when its handler raises; preserving that
                # behavior prevents a reconnect from replaying partial side
                # effects after an exception.
                self._highest_dispatched_block = block_num
                self._last_known_block = max(
                    int(getattr(self, "_last_known_block", 0) or 0),
                    block_num,
                )
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

    def _poll_current_head_catch_up(
        self,
        last_block: int,
        subtensor_obj: object | None = None,
        *,
        head_generation: Optional[int] = None,
    ) -> int:
        if subtensor_obj is None:
            current, current_hash, current_hash_real = (
                self._get_current_head_block_and_hash()
            )
        else:
            current, current_hash, current_hash_real = (
                self._get_current_head_block_and_hash(subtensor_obj)
            )
        if current <= 0:
            bt.logging.debug("Current-head catch-up skipped: no valid head block")
            return last_block
        self._record_capacity_audit_live_head(
            current,
            generation=head_generation,
            source="poll",
        )

        # Show progress while waiting for sync block
        if current < self._sync_block:
            blocks_left = self._sync_block - current
            if current % 10 == 0:
                bt.logging.info(
                    f"Block {current} | waiting for epoch boundary "
                    f"(block {self._sync_block}, ~{blocks_left * 12 // 60}min)",
                )
            if current % 60 == 0:
                self._schedule_metagraph_stats_refresh()
            if current % 5 == 0 and hasattr(self, '_cached_metagraph_parts'):
                bt.logging.info(f"Metagraph | block={current} | {' | '.join(self._cached_metagraph_parts)}")

        return self._process_current_head_block_range(
            last_block,
            current,
            current_hash=current_hash,
            current_hash_real=current_hash_real,
        )

    def _process_current_head_subscription_header(
        self,
        block_header: object,
        last_block: int,
        rpc_subtensor: object,
        *,
        head_generation: Optional[int] = None,
    ) -> int:
        """Process one streamed header without re-entering its RPC socket.

        ``substrate-interface`` invokes subscription callbacks on the same
        websocket reader that owns the subscription. Issuing another RPC on
        that connection from inside the callback deadlocks the reader. A
        dedicated non-subscription Subtensor connection resolves the block
        hash and any catch-up blocks without contending with the validator's
        other background RPC work.
        """

        block_number = self._block_number_from_header(block_header)
        if block_number is None:
            return last_block
        # Publish chronology before hash lookup or block dispatch. Receipt
        # ingestion must remain accurate even when those slower operations
        # stall and the worker queue continues to drain.
        self._record_capacity_audit_live_head(
            block_number,
            generation=head_generation,
            source="stream",
        )
        block_hash, block_hash_real = self._get_chain_block_hash(
            block_number,
            rpc_subtensor,
        )
        return self._process_current_head_block_range(
            last_block,
            block_number,
            rpc_subtensor,
            current_hash=block_hash,
            current_hash_real=block_hash_real,
        )

    def _run_with_streaming(self):
        """Use current-head streaming as primary path with polling catch-up."""
        import bittensor as bt

        last_block = self._sync_block - 1
        watchdog_s = self._block_stream_watchdog_s()
        fallback_poll_s = self._block_stream_fallback_poll_s()
        while self._running:
            fresh_sub = None
            rpc_sub = None
            head_sub = None
            head_generation = self._begin_capacity_audit_head_generation()
            try:
                bt.logging.info("Creating fresh Subtensor connection for current-head block stream...")
                SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
                fresh_sub = SubtensorCls(network=self.config.subtensor_network)
                rpc_sub = SubtensorCls(network=self.config.subtensor_network)
                head_sub = SubtensorCls(network=self.config.subtensor_network)
                substrate = getattr(fresh_sub, "substrate", None)
                subscribe = getattr(substrate, "subscribe_block_headers", None)
                head_subscribe = getattr(
                    getattr(head_sub, "substrate", None),
                    "subscribe_block_headers",
                    None,
                )
                if subscribe is None:
                    bt.logging.warning("Current-head block stream unavailable; using polling catch-up")
                    last_block = self._poll_current_head_catch_up(
                        last_block,
                        rpc_sub,
                        head_generation=head_generation,
                    )
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
                head_state = SimpleNamespace(error=None)

                def head_callback(block_header):
                    if not self._running or stop_event.is_set():
                        raise StopIteration("Validator shutting down")
                    block_number = self._block_number_from_header(block_header)
                    if block_number is not None:
                        # This subscriber does no hash lookup or block work, so
                        # receipt chronology continues advancing even while the
                        # main callback is processing a large audit cohort.
                        self._record_capacity_audit_live_head(
                            block_number,
                            generation=head_generation,
                            source="head_stream",
                        )
                    return None

                def callback(block_header):
                    if not self._running or stop_event.is_set():
                        raise StopIteration("Validator shutting down")
                    with process_lock:
                        with lock:
                            state.last_header_at = time.monotonic()
                            state.active = True
                            base_block = int(state.last_block)
                        try:
                            new_last = (
                                self._process_current_head_subscription_header(
                                    block_header,
                                    base_block,
                                    rpc_sub,
                                    head_generation=head_generation,
                                )
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
                            new_last = self._poll_current_head_catch_up(
                                base_block,
                                rpc_sub,
                                head_generation=head_generation,
                            )
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

                def run_head_subscription():
                    if head_subscribe is None:
                        return
                    try:
                        head_subscribe(head_callback, finalized_only=False)
                    except Exception as exc:
                        with lock:
                            head_state.error = exc

                thread = threading.Thread(
                    target=run_subscription,
                    name="validator-current-block-stream",
                    daemon=True,
                )
                thread.start()
                head_thread = threading.Thread(
                    target=run_head_subscription,
                    name="validator-capacity-head-stream",
                    daemon=True,
                )
                head_thread.start()
                bt.logging.info(
                    f"Subscribing to current-head block headers "
                    f"(watchdog_s={watchdog_s:g}, fallback_poll_s={fallback_poll_s:g})"
                )

                while self._running and thread.is_alive():
                    time.sleep(min(1.0, max(0.1, fallback_poll_s)))
                    now = time.monotonic()
                    with lock:
                        error = state.error
                        head_error = head_state.error
                        active = bool(state.active)
                        last_header_at = float(state.last_header_at or 0.0)
                        last_block = int(state.last_block)
                        started_at = float(state.started_at)
                    last_block = catch_up_if_due(now, last_block, active)
                    if error is not None:
                        bt.logging.warning(f"Current-head block stream ended: {error}")
                        break
                    if head_subscribe is not None and head_error is not None:
                        bt.logging.warning(
                            f"Capacity chronology head stream ended: {head_error}"
                        )
                        break
                    reference_at = last_header_at or started_at
                    if now - reference_at > watchdog_s:
                        bt.logging.warning(
                            "Current-head block stream "
                            + ("callback stalled" if active else "stale")
                            + f" for {now - reference_at:.1f}s; "
                            "reconnecting after polling catch-up"
                        )
                        break

                stop_event.set()
                self._end_capacity_audit_head_generation(head_generation)
                last_block = int(getattr(state, "last_block", last_block) or last_block)
                self._close_subtensor(fresh_sub)
                fresh_sub = None
                self._close_subtensor(rpc_sub)
                rpc_sub = None
                self._close_subtensor(head_sub)
                head_sub = None
                if self._running:
                    last_block = self._poll_current_head_catch_up(last_block)
            except Exception as exc:
                bt.logging.error(f"Current-head block stream error: {exc}")
                last_block = self._poll_current_head_catch_up(last_block)
                time.sleep(fallback_poll_s)
            finally:
                self._end_capacity_audit_head_generation(head_generation)
                if fresh_sub is not None:
                    self._close_subtensor(fresh_sub)
                if rpc_sub is not None:
                    self._close_subtensor(rpc_sub)
                if head_sub is not None:
                    self._close_subtensor(head_sub)

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
        self._stop_isolated_capacity_audit_head_observer()
        if self._capacity_audit_server is not None:
            self._capacity_audit_server.should_exit = True
        self._executor.shutdown(wait=False)
        self._control_executor.shutdown(wait=False)
        self._weight_update_executor.shutdown(wait=False)
        self._report_offline_executor.shutdown(wait=False)
        self._epoch_close_executor.shutdown(wait=False)
        self._capacity_audit_executor.shutdown(wait=False)
        self._capacity_audit_discovery_executor.shutdown(wait=False)
        self._capacity_audit_receipt_executor.shutdown(wait=False)
        self._capacity_audit_state_executor.shutdown(wait=False)
        self._miner_debug_executor.shutdown(wait=False)
        self._metagraph_stats_executor.shutdown(wait=False)
        self._capacity_audit_proof_executor.shutdown(wait=False)
        self._shared_hard_prefetch_executor.shutdown(wait=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Verathos Validator Neuron")
    parser.add_argument("--wallet", default="default")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON. If omitted, derived from --subtensor-network.")
    parser.add_argument(
        "--proof-v2-manifest",
        action="append",
        default=None,
        help=(
            "Path to a signed proof-v2 manifest document. Repeat once per "
            "registered model. Defaults to VERATHOS_PROOF_V2_MANIFESTS "
            "split by the platform path separator."
        ),
    )
    parser.add_argument(
        "--proof-v2-artifact-base-url",
        action="append",
        default=None,
        help=(
            "HTTPS base URL for content-addressed proof-v2 artifacts. "
            "Repeat to configure fallback mirrors; defaults to "
            "VERATHOS_PROOF_V2_ARTIFACT_BASE_URLS (comma-separated)."
        ),
    )
    parser.add_argument(
        "--proof-v2-artifact-cache-dir",
        default=None,
        help=(
            "Local proof-v2 artifact cache directory; defaults to "
            "VERATHOS_PROOF_V2_ARTIFACT_CACHE_DIR or VERALLM_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--proof-v3-release",
        action="append",
        default=None,
        help=(
            "Path to a proof-v3 release descriptor. Repeat once per "
            "qualified model. Defaults to VERATHOS_PROOF_V3_RELEASES split "
            "by the platform path separator."
        ),
    )
    parser.add_argument(
        "--proof-v3-artifact-base-url",
        action="append",
        default=None,
        help=(
            "HTTPS base URL for content-addressed proof-v3 releases. "
            "Repeat for mirrors; defaults to "
            "VERATHOS_PROOF_V3_ARTIFACT_BASE_URLS."
        ),
    )
    parser.add_argument(
        "--proof-v3-artifact-cache-dir",
        default=None,
        help=(
            "Local proof-v3 release cache; defaults to "
            "VERATHOS_PROOF_V3_ARTIFACT_CACHE_DIR or VERALLM_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--proof-v3-canary-policy",
        default=None,
        help=(
            "Path to the authority-signed proof-v3 canary policy. Defaults "
            "to VERATHOS_PROOF_V3_CANARY_POLICY."
        ),
    )
    parser.add_argument(
        "--proof-v3-verdict-source",
        choices=("verify", "follower"),
        default=None,
        help=(
            "Source for hard/capacity decisions. follower is the default; "
            "owner and independent-verifier deployments explicitly select "
            "verify."
        ),
    )
    parser.add_argument(
        "--owner-verdict-url",
        default=None,
        help=(
            "Owner proxy base URL for follower mode. Defaults are selected "
            "by network; this flag is an explicit override."
        ),
    )
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
    parser.add_argument("--capacity-audit-invalid-proof-misses-for-zero-score", type=int, default=None,
                        help="Cryptographically invalid proofs required to zero one endpoint score.")
    parser.add_argument("--capacity-audit-uid-escalation-min-entries", type=int, default=None,
                        help="Minimum independently convicted entries required for UID-wide zeroing.")
    parser.add_argument("--capacity-audit-uid-escalation-fraction", type=float, default=None,
                        help="Active-entry fraction independently convicted before UID-wide zeroing.")
    parser.add_argument("--capacity-audit-uid-escalation-max-entries", type=int, default=None,
                        help="Maximum independently convicted entries required for UID-wide zeroing.")
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
    if args.proof_v3_verdict_source is not None:
        extra_kwargs["proof_v3_verdict_source"] = (
            args.proof_v3_verdict_source
        )
    if args.owner_verdict_url is not None:
        extra_kwargs["owner_verdict_url"] = args.owner_verdict_url
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
    if args.capacity_audit_invalid_proof_misses_for_zero_score is not None:
        extra_kwargs["capacity_audit_invalid_proof_misses_for_zero_score"] = args.capacity_audit_invalid_proof_misses_for_zero_score
    if args.capacity_audit_uid_escalation_min_entries is not None:
        extra_kwargs["capacity_audit_uid_escalation_min_entries"] = args.capacity_audit_uid_escalation_min_entries
    if args.capacity_audit_uid_escalation_fraction is not None:
        extra_kwargs["capacity_audit_uid_escalation_fraction"] = args.capacity_audit_uid_escalation_fraction
    if args.capacity_audit_uid_escalation_max_entries is not None:
        extra_kwargs["capacity_audit_uid_escalation_max_entries"] = args.capacity_audit_uid_escalation_max_entries
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
    config.proof_v2_manifest_paths = _resolve_proof_v2_manifest_paths(
        args.proof_v2_manifest,
        chain_id=config.chain_id,
        netuid=config.netuid,
    )
    from verallm.proof_v2.artifact_store import (
        configured_proof_v2_artifact_base_urls,
    )

    config.proof_v2_artifact_base_urls = (
        configured_proof_v2_artifact_base_urls(
            args.proof_v2_artifact_base_url,
            default_values=getattr(
                chain_config,
                "proof_v2_artifact_base_urls",
                (),
            ),
        )
    )
    config.proof_v2_artifact_cache_dir = (
        args.proof_v2_artifact_cache_dir
        or getattr(chain_config, "proof_v2_artifact_cache_dir", "")
        or None
    )
    config.proof_v3_release_paths = _resolve_proof_v3_release_paths(
        args.proof_v3_release,
    )
    from verallm.proof_v3.artifact_store import (
        configured_proof_v3_artifact_base_urls,
    )

    config.proof_v3_artifact_base_urls = (
        configured_proof_v3_artifact_base_urls(
            args.proof_v3_artifact_base_url,
            default_values=getattr(
                chain_config,
                "proof_v3_artifact_base_urls",
                (),
            ),
        )
    )
    config.proof_v3_artifact_cache_dir = (
        args.proof_v3_artifact_cache_dir
        or getattr(chain_config, "proof_v3_artifact_cache_dir", "")
        or None
    )
    config.proof_v3_canary_policy_path = _resolve_proof_v3_canary_policy_path(
        args.proof_v3_canary_policy,
    )
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
    # The receipt endpoint is not exposed until its process-isolated live-head
    # source is ready. This prevents a restart from advertising a fail-closed
    # but unusable capacity-ingest path.
    neuron._start_isolated_capacity_audit_head_observer()
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
            """Don't restart while validator-owned work is live."""
            return neuron._auto_update_busy()

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
        neuron._epoch_miners = neuron._apply_mainnet_endpoint_policy(
            neuron._epoch_miners,
            epoch_number=0,
        )
        # Fetch hardware metadata from miners at startup (best-effort).
        neuron._refresh_miner_hardware_batch(
            neuron._epoch_miners,
            source="startup",
        )
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
