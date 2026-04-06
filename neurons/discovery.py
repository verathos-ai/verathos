"""Miner discovery: enumerate active miners from the on-chain MinerRegistry."""

from __future__ import annotations

import logging
import bittensor as bt
import time
from dataclasses import dataclass, field
from typing import List


logger = logging.getLogger(__name__)


@dataclass
class ActiveMiner:
    """A miner discovered from the on-chain registry."""

    address: str
    model_id: str
    endpoint: str
    quant: str
    max_context_len: int
    model_index: int  # index in miner's model array
    hotkey_ss58: str = ""
    coldkey_ss58: str = ""

    # TEE capability (populated during discovery if miner has on-chain TEE attestation)
    tee_enabled: bool = False
    tee_platform: str = ""
    tee_model_weight_hash: str = ""  # hex
    enclave_public_key: str = ""  # hex, 32-byte X25519 public key

    # Hardware metadata (populated from miner /health during identity verification)
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    compute_capability: str = ""
    gpu_uuids: List[str] = field(default_factory=list)


def discover_active_miners(
    miner_client,
    model_client=None,
) -> List[ActiveMiner]:
    """Discover all active miners from the MinerRegistry.

    Flow:
    1. Get model list from ModelRegistry (optional — if provided, only
       discovers miners for registered models)
    2. For each model: getProvidersForModel() -> addresses
    3. For each address: getMinerModels() -> entries
    4. Filter: active=True and expiresAt > now
    5. Deduplicate by (address, model_index)

    Args:
        miner_client: MinerRegistryClient or MockMinerRegistryClient.
        model_client: Optional ModelRegistryClient for filtering by registered models.

    Returns:
        List of ActiveMiner entries, deduplicated.
    """
    now = int(time.time())
    seen = set()  # (address, model_index)
    active = []

    # Get model IDs to search for
    if model_client is not None:
        model_ids = model_client.get_model_list()
    else:
        model_ids = None

    if model_ids is not None:
        # Discover per-model
        for model_id in model_ids:
            providers = miner_client.get_providers_for_model(model_id)
            for addr in providers:
                _collect_active(miner_client, addr, now, seen, active)
    else:
        # No model filtering — would need a separate mechanism to enumerate
        # all miners. For now, log a warning.
        bt.logging.warning(
            "No model_client provided — cannot enumerate all miners. "
            "Pass a ModelRegistryClient for full discovery."
        )

    bt.logging.info(f"Discovered {len(active)} active miner-model entries")
    return active


def _collect_active(
    miner_client,
    address: str,
    now: int,
    seen: set,
    active: list,
) -> None:
    """Collect active entries for a single miner address."""
    try:
        models = miner_client.get_miner_models(address)
    except Exception as e:
        bt.logging.warning(f"Failed to get models for {address}: {e}")
        return

    for i, m in enumerate(models):
        if not m.active or m.expires_at <= now:
            continue

        # Dedup by (address, model_id, quant, endpoint) — prevents a miner
        # from registering the same model+endpoint multiple times to get
        # disproportionate canary tests / organic traffic.
        # Only active entries are deduped — inactive/expired entries are
        # skipped above and must NOT pollute the seen set.
        dedup_key = (address.lower(), m.model_id.lower(), m.quant.lower(), m.endpoint.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        miner = ActiveMiner(
            address=address,
            model_id=m.model_id,
            endpoint=m.endpoint,
            quant=m.quant,
            max_context_len=m.max_context_len,
            model_index=i,
        )
        # Enrich with TEE capability (best-effort, don't block discovery)
        try:
            if miner_client.has_tee(address):
                cap = miner_client.get_tee_capability(address)
                miner.tee_enabled = cap.enabled
                miner.tee_platform = cap.platform
                miner.tee_model_weight_hash = cap.model_weight_hash.hex() if cap.model_weight_hash else ""
                miner.enclave_public_key = cap.enclave_pub_key.hex() if cap.enclave_pub_key else ""
        except Exception:
            pass  # TEE read failed — non-fatal, miner is still active
        active.append(miner)
