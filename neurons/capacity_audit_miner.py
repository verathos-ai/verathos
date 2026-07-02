"""Miner-side hot-capacity audit worker.

The worker is non-interactive: it watches current-head chain data, derives the
audit cohort locally, and publishes signed artifacts when this miner's endpoint
slot is selected. Validators do not send per-miner challenge commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import bittensor as bt
import httpx

from neurons.capacity_audit import (
    CapacityAuditRuntimeConfig,
    CapacitySlot,
    PROTOCOL_VERSION,
    build_capacity_slot_group_key,
    capacity_audit_slot_selected,
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
    root_words_digest,
    sign_artifact,
    slot_id,
    transcript_root,
)
from neurons.capacity_audit_combined import COMBINED_PROOF_FORMAT
from neurons.capacity_audit_discovery import CapacityAuditEndpointResolver
from neurons.discovery import ActiveMiner
from neurons.subnet_runtime_config import (
    RuntimeSubnetConfigClient,
    apply_runtime_config_to_neuron_config,
    capacity_audit_config_from_neuron_config,
)


@dataclass(frozen=True)
class MinerAuditSlot:
    slot: CapacitySlot
    gpu_class_name: str
    passes: int
    deadline_s: float
    audit_id: str
    selection_block: int
    audit_block: int
    proof_challenge_block: int
    cohort_seed: str
    workload_spec: dict = field(default_factory=dict)


@dataclass
class PreparedAuditProcess:
    proc: subprocess.Popen
    out_dir: Path
    lease: str
    challenge_file: Path
    start_file: Path
    ready_file: Path


def _coerce_block_hash(raw: object) -> Optional[bytes]:
    if isinstance(raw, bytes):
        return raw if len(raw) == 32 else None
    if raw is None:
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


def _root_hex(root_words: object) -> str:
    if isinstance(root_words, str):
        text = root_words.strip()
        raw = text[2:] if text.startswith("0x") else text
        if len(raw) == 64:
            try:
                bytes.fromhex(raw)
                return raw
            except ValueError:
                pass
    return root_words_digest(root_words)


class CapacityAuditMinerWorker:
    """Background worker for miner-published capacity audit artifacts."""

    def __init__(
        self,
        *,
        config,
        miner_client,
        model_client,
        evm_address: str,
        evm_private_key: str,
        endpoint: str,
        model_id: str,
        model_index: int,
        quant: str,
        max_context_len: int,
        validator_urls: tuple[str, ...],
        local_health_url: str = "",
        audit_state_file: str = "",
        poll_interval_s: float = 2.0,
    ):
        self.config = config
        self.miner_client = miner_client
        self.model_client = model_client
        self.evm_address = evm_address.lower()
        self.evm_private_key = evm_private_key
        self.endpoint = endpoint
        self.model_id = model_id
        self.model_index = int(model_index)
        self.quant = quant
        self.max_context_len = int(max_context_len or 0)
        self.local_health_url = str(local_health_url or "").rstrip("/")
        self.audit_state_file = Path(audit_state_file) if audit_state_file else None
        self.validator_urls = tuple(u.rstrip("/") for u in validator_urls if u.strip())
        self._validator_endpoint_resolver = CapacityAuditEndpointResolver(
            config,
            manual_urls=self.validator_urls,
        )
        self.poll_interval_s = max(0.1, float(poll_interval_s))
        self.runtime_cfg = capacity_audit_config_from_neuron_config(config)
        self._subnet_runtime_config_client = RuntimeSubnetConfigClient.from_config(
            config,
            log=bt.logging,
        )
        self._subnet_runtime_config_key: tuple[int, Optional[int], str] | None = None
        self._subnet_runtime_config_authoritative = False
        self._refresh_subnet_runtime_config(force=True)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending: dict[int, list[MinerAuditSlot]] = {}
        self._seen_audits: set[str] = set()
        self._current_block_seen = 0
        self._busy_selection_block_until = 0
        self._active_audit_id = ""
        self._active_audit_until_ts = 0.0
        self._audit_start_lock = threading.Lock()
        self._prepared_audit_lock = threading.Lock()
        self._prepared_audits: dict[str, PreparedAuditProcess] = {}
        self._workspace_ext_lock = threading.Lock()
        self._workspace_ext_ready = False
        self._resolved_epoch_blocks: Optional[int] = None
        self._audit_endpoint_rejections: dict[str, set[str]] = {}

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
        key = runtime.cache_key
        if key == self._subnet_runtime_config_key:
            self._subnet_runtime_config_authoritative = True
            return True
        apply_runtime_config_to_neuron_config(runtime, self.config)
        self.runtime_cfg = runtime.capacity_audit
        self.poll_interval_s = max(0.1, float(runtime.capacity_audit_worker_poll_s))
        self._resolved_epoch_blocks = None
        self._subnet_runtime_config_key = key
        self._subnet_runtime_config_authoritative = True
        bt.logging.info(
            f"Applied runtime subnet config version={runtime.version} "
            f"effective_epoch={runtime.effective_epoch} source={runtime.source or 'server'}"
        )
        return True

    def _write_audit_state(self, payload: dict) -> None:
        if self.audit_state_file is None:
            return
        payload = dict(payload)
        payload["updated_at"] = time.time()
        tmp = self.audit_state_file.with_suffix(self.audit_state_file.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, sort_keys=True))
            os.replace(tmp, self.audit_state_file)
        except Exception as exc:
            bt.logging.warning(f"Capacity audit state write failed: {exc}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def _audit_state_until(self, audit_slot: MinerAuditSlot) -> float:
        block_budget_s = (
            max(0, audit_slot.audit_block - audit_slot.selection_block) * 12.0
            + max(0, audit_slot.proof_challenge_block - audit_slot.audit_block) * 12.0
        )
        evidence_budget_s = (
            float(self.runtime_cfg.deadline_s or 0.0)
            + float(self.runtime_cfg.transport_grace_s or 0.0)
            + float(self.runtime_cfg.payload_deadline_s or 0.0)
        )
        return time.time() + max(
            float(self.runtime_cfg.drain_seconds or 0.0),
            block_budget_s + evidence_budget_s + 30.0,
            60.0,
        )

    def _mark_audit_drain(self, audit_slot: MinerAuditSlot, *, phase: str) -> None:
        until_ts = self._audit_state_until(audit_slot)
        self._active_audit_id = audit_slot.audit_id
        self._active_audit_until_ts = max(float(self._active_audit_until_ts or 0.0), until_ts)
        self._write_audit_state({
            "active": True,
            "reason": "capacity_audit",
            "phase": phase,
            "audit_id": audit_slot.audit_id,
            "slot_id": slot_id(audit_slot.slot),
            "model_index": self.model_index,
            "B_select": audit_slot.selection_block,
            "B_start": audit_slot.audit_block,
            "B_proof": audit_slot.proof_challenge_block,
            "until_ts": until_ts,
        })

    def _clear_audit_drain(self, audit_id: str) -> None:
        if self.audit_state_file is not None:
            try:
                current = json.loads(self.audit_state_file.read_text())
                if isinstance(current, dict):
                    current_audit = str(current.get("audit_id") or "")
                    if current_audit and current_audit != audit_id:
                        return
            except FileNotFoundError:
                pass
            except Exception:
                pass
        if not self._active_audit_id or self._active_audit_id == audit_id:
            self._active_audit_id = ""
            self._active_audit_until_ts = 0.0
        self._write_audit_state({
            "active": False,
            "last_audit_id": audit_id,
        })

    def _has_active_local_audit(self) -> bool:
        audit_id = str(getattr(self, "_active_audit_id", "") or "")
        if not audit_id:
            return False
        until_ts = float(getattr(self, "_active_audit_until_ts", 0.0) or 0.0)
        if until_ts > 0.0 and time.time() > until_ts:
            self._active_audit_id = ""
            self._active_audit_until_ts = 0.0
            return False
        return True

    def _extend_busy_selection_until_current_head(
        self,
        audit_slot: MinerAuditSlot,
        *,
        subtensor=None,
    ) -> None:
        """Skip B_select blocks that arrived while this miner was busy auditing."""
        current_block = 0
        created_subtensor = None
        try:
            active_subtensor = subtensor
            if active_subtensor is None:
                created_subtensor = self._subtensor()
                active_subtensor = created_subtensor
            current_block, _head_hash = self._get_live_current_head_block_and_hash(active_subtensor)
        except Exception:
            current_block = 0
        finally:
            if created_subtensor is not None:
                self._close_subtensor(created_subtensor)
        self._busy_selection_block_until = max(
            int(self._busy_selection_block_until or 0),
            int(audit_slot.proof_challenge_block) + 1,
            int(current_block or 0),
        )

    def start(self) -> None:
        def disabled(message: str) -> None:
            if str(self.runtime_cfg.mode or "observe") != "observe":
                raise RuntimeError(message)
            bt.logging.warning(message)

        if not self.evm_private_key:
            disabled("Capacity audit miner worker disabled: missing EVM private key")
            return
        script_dir = self._workspace_script().parent
        if not self._ensure_workspace_extension(script_dir):
            disabled(
                "Capacity audit miner worker disabled: hot-capacity workspace extension unavailable"
            )
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="capacity-audit-miner", daemon=True)
        self._thread.start()
        validator_urls = self._validator_endpoint_urls(force_refresh=True)
        if not validator_urls:
            bt.logging.warning(
                "Capacity audit miner worker started without discovered validator "
                "audit endpoint yet; discovery will retry before publishing artifacts"
            )
        bt.logging.info(
            f"Capacity audit miner worker started for model_index={self.model_index} "
            f"validators={len(validator_urls)} mode={self.runtime_cfg.mode} "
            f"poll_s={self.poll_interval_s:g}"
        )

    def stop(self) -> None:
        self._running = False

    def _subtensor(self):
        SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
        return SubtensorCls(network=self.config.subtensor_network)

    def _get_block_hash(self, subtensor, block_number: int) -> Optional[bytes]:
        raw = None
        substrate = getattr(subtensor, "substrate", None)
        if substrate is not None:
            try:
                response = substrate.rpc_request("chain_getBlockHash", [int(block_number)])
                raw = response.get("result") if isinstance(response, dict) else response
            except Exception:
                raw = None
            normalized = _coerce_block_hash(raw)
            if normalized is not None:
                return normalized
        try:
            raw = subtensor.get_block_hash(block=int(block_number))
        except TypeError:
            try:
                raw = subtensor.get_block_hash(int(block_number))
            except Exception:
                raw = None
        except Exception:
            raw = None
        if raw is None and substrate is not None:
            try:
                raw = substrate.get_block_hash(block_id=int(block_number))
            except Exception:
                raw = None
        return _coerce_block_hash(raw)

    def _get_current_head_block_and_hash(self, subtensor) -> tuple[int, Optional[bytes]]:
        try:
            method = getattr(subtensor, "get_current_block", None)
            if callable(method):
                block_number = int(method())
            else:
                substrate = getattr(subtensor, "substrate", None)
                if substrate is None:
                    return 0, None
                header = substrate.get_chain_head()
                block_number = self._block_number_from_header(header) or 0
            if block_number <= 0:
                return 0, None
            block_hash = self._get_block_hash(subtensor, block_number)
            return block_number, block_hash
        except Exception as exc:
            bt.logging.debug(f"Capacity audit current head lookup failed: {exc}")
            return 0, None

    def _get_live_current_head_block_and_hash(self, subtensor) -> tuple[int, Optional[bytes]]:
        """Read best-head directly from the node RPC, bypassing cached helpers."""
        block_number = self._get_live_current_head_block(subtensor)
        if block_number > 0:
            return block_number, self._get_block_hash(subtensor, block_number)
        return self._get_current_head_block_and_hash(subtensor)

    def _get_live_current_head_block(self, subtensor) -> int:
        """Read best-head block number without resolving a block hash."""
        substrate = getattr(subtensor, "substrate", None)
        if substrate is not None:
            try:
                response = substrate.rpc_request("chain_getHeader", [])
                header = response.get("result") if isinstance(response, dict) else response
                block_number = self._block_number_from_header(header)
                if block_number and block_number > 0:
                    return int(block_number)
            except Exception as exc:
                bt.logging.debug(f"Capacity audit live head lookup failed: {exc}")
        block_number, _hash = self._get_current_head_block_and_hash(subtensor)
        return int(block_number or 0)

    def _epoch_blocks(self, subtensor=None) -> int:
        cached = getattr(self, "_resolved_epoch_blocks", None)
        if cached and cached > 0:
            return int(cached)

        configured = int(getattr(self.config, "epoch_blocks", 360) or 360)
        if getattr(self, "_subnet_runtime_config_authoritative", False):
            self._resolved_epoch_blocks = max(1, configured)
            return int(self._resolved_epoch_blocks)
        netuid = int(getattr(self.config, "netuid", 0) or 0)
        resolved = 0

        if subtensor is not None and netuid >= 0:
            probes = (
                lambda: subtensor.subnet(netuid=netuid),
                lambda: subtensor.subnet(netuid),
                lambda: subtensor.get_subnet_hyperparameters(netuid=netuid),
                lambda: subtensor.get_subnet_hyperparameters(netuid),
            )
            for probe in probes:
                try:
                    info = probe()
                except Exception:
                    continue
                tempo = getattr(info, "tempo", None)
                if tempo is None and isinstance(info, dict):
                    tempo = info.get("tempo")
                try:
                    resolved = int(tempo)
                except (TypeError, ValueError):
                    resolved = 0
                if resolved > 0:
                    break

        self._resolved_epoch_blocks = max(1, resolved or configured)
        if resolved > 0 and resolved != configured:
            bt.logging.info(
                f"Capacity audit using subnet tempo as epoch_blocks={resolved} "
                f"(config={configured})"
            )
        return int(self._resolved_epoch_blocks)

    @staticmethod
    def _block_number_from_header(block_header: object) -> Optional[int]:
        value = getattr(block_header, "value", block_header)
        if isinstance(value, dict):
            header = value.get("header") if isinstance(value.get("header"), dict) else value
            number = header.get("number")
            if number is not None:
                try:
                    if isinstance(number, str):
                        return int(number, 0)
                    return int(number)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _close_subtensor(subtensor) -> None:
        for obj in (subtensor, getattr(subtensor, "substrate", None)):
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                pass

    def _block_stream_watchdog_s(self) -> float:
        raw = os.getenv("VERATHOS_BLOCK_STREAM_WATCHDOG_S", "30")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 30.0
        return max(5.0, value)

    def _audit_already_started(self, audit_id: str) -> bool:
        with self._audit_start_lock:
            return audit_id in self._seen_audits

    def _mark_audit_started_once(self, audit_id: str) -> bool:
        with self._audit_start_lock:
            if audit_id in self._seen_audits:
                return False
            self._seen_audits.add(audit_id)
            return True

    def _store_prepared_audit(self, audit_id: str, prepared: PreparedAuditProcess) -> None:
        with self._prepared_audit_lock:
            self._prepared_audits[audit_id] = prepared

    def _pop_prepared_audit(self, audit_id: str) -> Optional[PreparedAuditProcess]:
        with self._prepared_audit_lock:
            return self._prepared_audits.pop(audit_id, None)

    def _drop_prepared_audit(self, audit_id: str, prepared: Optional[PreparedAuditProcess]) -> bool:
        with self._prepared_audit_lock:
            current = self._prepared_audits.get(audit_id)
            if current is None or (prepared is not None and current is not prepared):
                return False
            self._prepared_audits.pop(audit_id, None)
            return True

    def _get_block_hash_fresh(self, block_number: int) -> Optional[bytes]:
        subtensor = None
        try:
            subtensor = self._subtensor()
            return self._get_block_hash(subtensor, block_number)
        except Exception as exc:
            bt.logging.debug(f"Capacity audit fresh block hash lookup failed for block {block_number}: {exc}")
            return None
        finally:
            if subtensor is not None:
                self._close_subtensor(subtensor)

    def _process_block_range(
        self,
        last_block: int,
        current_block: int,
        subtensor,
        *,
        current_hash: Optional[bytes] = None,
    ) -> int:
        if current_block <= 0:
            return last_block
        if last_block <= 0:
            last_block = current_block - 1
        if current_block <= last_block:
            return last_block
        for block_number in range(last_block + 1, current_block + 1):
            if not self._running:
                break
            block_hash = current_hash if block_number == current_block else None
            if block_hash is None:
                block_hash = self._get_block_hash(subtensor, block_number)
            if block_hash is None:
                block_hash = self._get_block_hash_fresh(block_number)
            if block_hash is None:
                bt.logging.warning(
                    f"Capacity audit block stream missing chain hash for block {block_number}; "
                    "will retry on the next catch-up"
                )
                break
            self._current_block_seen = max(self._current_block_seen, int(block_number))
            self._on_block(block_number, block_hash, subtensor)
            last_block = block_number
        return last_block

    def _poll_catch_up(self, last_block: int) -> int:
        subtensor = None
        try:
            subtensor = self._subtensor()
            block, block_hash = self._get_current_head_block_and_hash(subtensor)
            return self._process_block_range(
                last_block,
                block,
                subtensor,
                current_hash=block_hash,
            )
        except Exception as exc:
            bt.logging.warning(f"Capacity audit block polling catch-up failed: {exc}")
            return last_block
        finally:
            if subtensor is not None:
                self._close_subtensor(subtensor)

    def _run(self) -> None:
        last_block = 0
        watchdog_s = self._block_stream_watchdog_s()
        while self._running:
            subtensor = None
            try:
                subtensor = self._subtensor()
                substrate = getattr(subtensor, "substrate", None)
                subscribe = getattr(substrate, "subscribe_block_headers", None)
                if subscribe is None:
                    bt.logging.warning("Capacity audit block stream unavailable; using polling catch-up")
                    last_block = self._poll_catch_up(last_block)
                    time.sleep(self.poll_interval_s)
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
                        raise StopIteration("capacity audit miner stopping")
                    block_number = self._block_number_from_header(block_header)
                    if block_number is None:
                        return None
                    with process_lock:
                        with lock:
                            state.last_header_at = time.monotonic()
                            state.active = True
                            base_block = int(state.last_block)
                        try:
                            block_hash = self._get_block_hash(subtensor, block_number)
                            new_last = self._process_block_range(
                                base_block,
                                block_number,
                                subtensor,
                                current_hash=block_hash,
                            )
                            with lock:
                                state.last_block = max(int(state.last_block), int(new_last))
                        finally:
                            with lock:
                                state.active = False
                    return None

                def catch_up_if_due(now: float, last_block_snapshot: int, active: bool) -> int:
                    nonlocal last_catch_up_at
                    if active or now - last_catch_up_at < self.poll_interval_s:
                        return last_block_snapshot
                    last_catch_up_at = now
                    try:
                        if not process_lock.acquire(blocking=False):
                            return last_block_snapshot
                        try:
                            with lock:
                                base_block = int(state.last_block)
                            new_last = self._poll_catch_up(base_block)
                        finally:
                            process_lock.release()
                        with lock:
                            if int(new_last) > int(state.last_block):
                                state.last_block = int(new_last)
                        if int(new_last) > int(last_block_snapshot):
                            bt.logging.debug(
                                f"Capacity audit stream catch-up advanced "
                                f"last_block={last_block_snapshot}->{new_last}"
                            )
                        return max(int(last_block_snapshot), int(new_last))
                    except Exception as exc:
                        bt.logging.debug(f"Capacity audit stream catch-up skipped: {exc}")
                        return last_block_snapshot

                def run_subscription():
                    try:
                        subscribe(callback, finalized_only=False)
                    except Exception as exc:
                        with lock:
                            state.error = exc

                thread = threading.Thread(
                    target=run_subscription,
                    name="capacity-audit-block-stream",
                    daemon=True,
                )
                thread.start()
                bt.logging.info(
                    f"Capacity audit block stream subscribed "
                    f"(watchdog_s={watchdog_s:g}, fallback_poll_s={self.poll_interval_s:g})"
                )

                while self._running and thread.is_alive():
                    time.sleep(min(1.0, max(0.1, self.poll_interval_s)))
                    now = time.monotonic()
                    with lock:
                        error = state.error
                        active = bool(state.active)
                        last_header_at = float(state.last_header_at or 0.0)
                        last_block = int(state.last_block)
                        started_at = float(state.started_at)
                    last_block = catch_up_if_due(now, last_block, active)
                    if error is not None:
                        bt.logging.warning(f"Capacity audit block stream ended: {error}")
                        break
                    reference_at = last_header_at or started_at
                    if not active and now - reference_at > watchdog_s:
                        bt.logging.warning(
                            f"Capacity audit block stream stale for {now - reference_at:.1f}s; "
                            "reconnecting after polling catch-up"
                        )
                        break

                stop_event.set()
                last_block = int(getattr(state, "last_block", last_block) or last_block)
                self._close_subtensor(subtensor)
                subtensor = None
                if self._running:
                    last_block = self._poll_catch_up(last_block)
            except Exception as exc:
                bt.logging.warning(f"Capacity audit miner worker error: {exc}")
                last_block = self._poll_catch_up(last_block)
                time.sleep(min(60.0, self.poll_interval_s * 2))
            finally:
                if subtensor is not None:
                    self._close_subtensor(subtensor)

    def _selection_hashes(self, selection_block: int, selection_block_hash: bytes, subtensor) -> Optional[list[bytes]]:
        count = max(1, int(self.runtime_cfg.beacon_hash_count or 1))
        hashes = [selection_block_hash]
        if count <= 1 or subtensor is None:
            return hashes
        for offset in range(1, count):
            block_hash = self._get_block_hash(subtensor, int(selection_block) - offset)
            if block_hash is None:
                bt.logging.info(
                    f"Capacity audit skipping B_select={selection_block}: "
                    f"missing beacon hash offset={offset}/{count - 1}"
                )
                return None
            hashes.append(block_hash)
        return hashes

    def _on_block(self, block_number: int, block_hash: bytes, subtensor=None) -> None:
        due = self._pending.pop(block_number, [])
        for audit_slot in due:
            prepared = self._pop_prepared_audit(audit_slot.audit_id)
            self._run_audit_slot(audit_slot, block_hash, subtensor, prepared=prepared)

        epoch_blocks = self._epoch_blocks(subtensor)
        if block_number % epoch_blocks == 0:
            current_epoch = int(block_number // epoch_blocks)
            self._refresh_subnet_runtime_config(current_epoch=current_epoch, force=True)
            epoch_blocks = self._epoch_blocks(subtensor)
        if (
            getattr(self, "_subnet_runtime_config_authoritative", False)
            and not getattr(self.runtime_cfg, "enabled", False)
        ):
            return
        if not capacity_audit_window_triggered(
            block_number,
            block_hash,
            epoch_blocks,
            self.runtime_cfg,
        ):
            return
        if not capacity_audit_window_fits_epoch(block_number, epoch_blocks, self.runtime_cfg):
            audit_block = int(block_number + self.runtime_cfg.lead_blocks)
            proof_challenge_block = int(
                audit_block + max(1, int(self.runtime_cfg.proof_challenge_delay_blocks or 1))
            )
            epoch_end = ((int(block_number) // epoch_blocks) + 1) * epoch_blocks
            bt.logging.info(
                f"Capacity audit skipping late local window: B_select={block_number} "
                f"B_start={audit_block} B_proof={proof_challenge_block} "
                f"epoch_end={epoch_end}"
            )
            return
        selection_hashes = self._selection_hashes(block_number, block_hash, subtensor)
        if selection_hashes is None:
            return
        selected = self._derive_selected_self_slots(block_number, selection_hashes, epoch_blocks)
        for audit_slot in selected:
            if self._has_active_local_audit():
                bt.logging.info(
                    f"Capacity audit skipping local slot while audit is active: "
                    f"audit_id={audit_slot.audit_id[:12]} B_select={block_number} "
                    f"B_start={audit_slot.audit_block} active_audit={self._active_audit_id[:12]}"
                )
                continue
            if int(block_number) <= int(self._busy_selection_block_until or 0):
                bt.logging.info(
                    f"Capacity audit skipping overlapping local slot: audit_id={audit_slot.audit_id[:12]} "
                    f"B_select={block_number} B_start={audit_slot.audit_block} "
                    f"busy_until_block={self._busy_selection_block_until}"
                )
                continue
            if audit_slot.audit_block <= self._current_block_seen and self._current_block_seen > block_number:
                bt.logging.info(
                    f"Capacity audit skipping stale local slot: audit_id={audit_slot.audit_id[:12]} "
                    f"B_select={block_number} B_start={audit_slot.audit_block} "
                    f"current_block={self._current_block_seen}"
                )
                continue
            self._busy_selection_block_until = max(
                int(self._busy_selection_block_until or 0),
                int(audit_slot.proof_challenge_block) + 1,
            )
            self._mark_audit_drain(audit_slot, phase="selected")
            self._pending.setdefault(int(audit_slot.audit_block), []).append(audit_slot)
            self._start_audit_waiter(audit_slot)
            bt.logging.info(
                f"Capacity audit selected local slot: audit_id={audit_slot.audit_id[:12]} "
                f"B_select={block_number} B_start={audit_slot.audit_block}"
            )

    def _start_audit_waiter(self, audit_slot: MinerAuditSlot) -> None:
        thread = threading.Thread(
            target=self._await_and_run_audit_slot,
            args=(audit_slot,),
            name=f"capacity-audit-start-{audit_slot.audit_id[:12]}",
            daemon=True,
        )
        thread.start()

    def _start_waiter_stale_head_s(self, audit_slot: MinerAuditSlot, last_head: int, start_poll_s: float) -> float:
        base_stale_head_s = max(18.0, float(self.poll_interval_s or 0.5) * 4.0)
        blocks_until_start = int(audit_slot.audit_block) - int(last_head or 0)
        if last_head > 0 and blocks_until_start <= 1:
            return min(base_stale_head_s, max(2.0, float(start_poll_s or 0.5) * 4.0))
        return base_stale_head_s

    def _await_and_run_audit_slot(self, audit_slot: MinerAuditSlot) -> None:
        """Start the workload as soon as this miner observes B_start.

        The main block-stream callback also performs selection/discovery work.
        Keeping the B_start waiter separate prevents unrelated block processing
        from delaying the timed workload once a local slot has been selected.
        """
        subtensor = None
        prepared: Optional[PreparedAuditProcess] = None
        start_poll_s = min(0.5, max(0.1, float(self.poll_interval_s or 0.5)))
        lead_wait_s = max(
            0.0,
            float(audit_slot.audit_block - audit_slot.selection_block) * 12.0,
        )
        deadline = time.time() + max(60.0, lead_wait_s + 60.0)
        last_head = 0
        last_head_at = time.time()
        last_wait_error = ""
        last_wait_error_logged_at = 0.0
        try:
            prepared = self._prepare_audit_process(audit_slot, start_timeout_s=max(60.0, lead_wait_s + 60.0))
            if prepared is None:
                self._extend_busy_selection_until_current_head(audit_slot)
                self._clear_audit_drain(audit_slot.audit_id)
                return
            self._store_prepared_audit(audit_slot.audit_id, prepared)
            while self._running and time.time() < deadline:
                if self._audit_already_started(audit_slot.audit_id):
                    if self._drop_prepared_audit(audit_slot.audit_id, prepared):
                        self._terminate_prepared_audit(prepared)
                    return
                now = time.time()
                try:
                    if subtensor is None:
                        subtensor = self._subtensor()
                    current_block = self._get_live_current_head_block(subtensor)
                    current_block = max(int(current_block or 0), int(self._current_block_seen or 0))
                    now = time.time()
                    if int(current_block or 0) > int(last_head or 0):
                        last_head = int(current_block or 0)
                        last_head_at = now
                    elif now - last_head_at > self._start_waiter_stale_head_s(audit_slot, last_head, start_poll_s):
                        bt.logging.info(
                            f"Capacity audit start waiter reconnecting stale head: "
                            f"audit_id={audit_slot.audit_id[:12]} "
                            f"last_head={last_head} B_start={audit_slot.audit_block}"
                        )
                        self._close_subtensor(subtensor)
                        subtensor = None
                        last_head_at = now
                        time.sleep(start_poll_s)
                        continue
                    if int(current_block or 0) >= int(audit_slot.audit_block):
                        audit_hash = self._get_block_hash(subtensor, audit_slot.audit_block)
                        if audit_hash is None:
                            audit_hash = self._get_block_hash_fresh(audit_slot.audit_block)
                        if audit_hash is not None:
                            self._run_audit_slot(
                                audit_slot,
                                audit_hash,
                                subtensor=subtensor,
                                prepared=prepared,
                            )
                            return
                        self._close_subtensor(subtensor)
                        subtensor = None
                except Exception as exc:
                    now = time.time()
                    last_wait_error = f"{type(exc).__name__}: {exc}"
                    if now - last_wait_error_logged_at >= 10.0:
                        last_wait_error_logged_at = now
                        bt.logging.warning(
                            f"Capacity audit start waiter RPC failure; retrying: "
                            f"audit_id={audit_slot.audit_id[:12]} "
                            f"B_start={audit_slot.audit_block} error={last_wait_error}"
                        )
                    bt.logging.debug(
                        f"Capacity audit start waiter reconnecting after RPC failure: "
                        f"audit_id={audit_slot.audit_id[:12]} last_head={last_head} "
                        f"B_start={audit_slot.audit_block}"
                    )
                    if subtensor is not None:
                        self._close_subtensor(subtensor)
                    subtensor = None
                    last_head_at = now
                    time.sleep(start_poll_s)
                    continue
                time.sleep(start_poll_s)
            suffix = f" last_error={last_wait_error}" if last_wait_error else ""
            bt.logging.warning(
                f"Capacity audit start wait timed out: audit_id={audit_slot.audit_id[:12]} "
                f"B_start={audit_slot.audit_block}{suffix}"
            )
            self._extend_busy_selection_until_current_head(audit_slot)
            self._clear_audit_drain(audit_slot.audit_id)
            if prepared is not None:
                self._drop_prepared_audit(audit_slot.audit_id, prepared)
                self._terminate_prepared_audit(prepared)
        except Exception as exc:
            bt.logging.warning(
                f"Capacity audit start waiter failed: audit_id={audit_slot.audit_id[:12]} {exc}"
            )
            self._extend_busy_selection_until_current_head(audit_slot)
            self._clear_audit_drain(audit_slot.audit_id)
            if prepared is not None:
                self._drop_prepared_audit(audit_slot.audit_id, prepared)
                self._terminate_prepared_audit(prepared)
        finally:
            if subtensor is not None:
                self._close_subtensor(subtensor)

    def _derive_selected_self_slots(
        self,
        selection_block: int,
        selection_block_hashes: list[bytes],
        epoch_blocks: int,
    ) -> list[MinerAuditSlot]:
        epoch_number = selection_block // epoch_blocks
        if not capacity_audit_window_fits_epoch(selection_block, epoch_blocks, self.runtime_cfg):
            return []

        if len(selection_block_hashes) <= 1:
            seed = derive_audit_seed(selection_block_hashes[0], epoch_number)
        else:
            seed = derive_audit_seed_from_hashes(selection_block_hashes, epoch_number)

        miner = self._discover_active_self_slot()
        if miner is None:
            return []
        registered_at = int(getattr(miner, "registered_at", 0) or 0)
        min_age = float(self.runtime_cfg.min_registration_age_s or 0.0)
        if registered_at > 0 and min_age > 0 and time.time() - registered_at < min_age:
            return []
        selection_slot = self._selection_slot_for_miner(miner)
        if not capacity_audit_slot_selected(selection_slot, seed, self.runtime_cfg):
            return []

        audit_block = int(selection_block + self.runtime_cfg.lead_blocks)
        proof_challenge_block = int(
            audit_block + max(1, int(self.runtime_cfg.proof_challenge_delay_blocks or 1))
        )
        audit_id = derive_audit_id(
            chain_id=int(getattr(self.config, "chain_id", 0) or 0),
            netuid=int(getattr(self.config, "netuid", 0) or 0),
            epoch_number=epoch_number,
            selection_block=selection_block,
            audit_block=audit_block,
            cohort_seed=seed,
        )
        supported = self._supported_local_slot(selection_slot, miner)
        if supported is None:
            bt.logging.warning(
                "Capacity audit selected this endpoint but local GPU is not calibrated "
                f"or health metadata is unavailable: audit_id={audit_id[:12]}"
            )
            return []
        supported_slot, row = supported
        return [MinerAuditSlot(
            slot=supported_slot,
            gpu_class_name=row.match_gpu_name,
            passes=capacity_gpu_pass_count(row),
            workload_spec=capacity_gpu_workload_spec(row),
            deadline_s=float(row.deadline_s or self.runtime_cfg.deadline_s),
            audit_id=audit_id,
            selection_block=selection_block,
            audit_block=audit_block,
            proof_challenge_block=proof_challenge_block,
            cohort_seed=seed,
        )]

    def _discover_active_self_slot(self) -> Optional[ActiveMiner]:
        try:
            models = self.miner_client.get_miner_models(self.evm_address)
        except Exception as exc:
            bt.logging.warning(f"Capacity audit self-slot lookup failed: {exc}")
            return None
        try:
            row = models[self.model_index]
        except Exception:
            bt.logging.warning(
                f"Capacity audit self-slot lookup found no model_index={self.model_index}"
            )
            return None
        if not bool(getattr(row, "active", False)):
            return None
        expires_at = int(getattr(row, "expires_at", 0) or 0)
        if expires_at > 0 and expires_at <= int(time.time()):
            return None
        return ActiveMiner(
            address=self.evm_address,
            model_id=str(getattr(row, "model_id", getattr(self, "model_id", "")) or getattr(self, "model_id", "")),
            endpoint=str(getattr(row, "endpoint", getattr(self, "endpoint", "")) or getattr(self, "endpoint", "")),
            quant=str(getattr(row, "quant", getattr(self, "quant", "")) or getattr(self, "quant", "")),
            max_context_len=int(getattr(row, "max_context_len", getattr(self, "max_context_len", 0)) or 0),
            model_index=self.model_index,
            expires_at=expires_at,
            registered_at=int(getattr(row, "registered_at", 0) or 0),
        )

    def _selection_slot_for_miner(self, miner: ActiveMiner) -> CapacitySlot:
        return CapacitySlot(
            chain_id=int(getattr(self.config, "chain_id", 0) or 0),
            netuid=int(getattr(self.config, "netuid", 0) or 0),
            address=miner.address,
            model_index=int(miner.model_index),
            endpoint=miner.endpoint,
            model_id=miner.model_id,
            quant=miner.quant,
            max_context_len=int(miner.max_context_len or 0),
            group_key=build_capacity_slot_group_key(
                address=miner.address,
                endpoint=miner.endpoint,
                model_id=miner.model_id,
            ),
        )

    def _supported_local_slot(
        self,
        selection_slot: CapacitySlot,
        miner: ActiveMiner,
    ) -> Optional[tuple[CapacitySlot, object]]:
        if (
            not (getattr(miner, "gpu_name", "") or "")
            or int(getattr(miner, "vram_gb", 0) or 0) <= 0
        ):
            self._enrich_hardware([miner])
        gpu_name = getattr(miner, "gpu_name", "") or ""
        gpu_count = int(getattr(miner, "gpu_count", 0) or 0)
        vram_gb = int(getattr(miner, "vram_gb", 0) or 0)
        row = match_gpu_class(gpu_name, vram_gb, self.runtime_cfg)
        if row is None or not row.calibrated or capacity_gpu_pass_count(row) <= 0:
            return None
        return (
            CapacitySlot(
                chain_id=selection_slot.chain_id,
                netuid=selection_slot.netuid,
                address=selection_slot.address,
                model_index=selection_slot.model_index,
                endpoint=selection_slot.endpoint,
                model_id=selection_slot.model_id,
                quant=selection_slot.quant,
                max_context_len=selection_slot.max_context_len,
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_gb=vram_gb,
                group_key=build_capacity_slot_group_key(
                    address=selection_slot.address,
                    endpoint=selection_slot.endpoint,
                    model_id=selection_slot.model_id,
                    gpu_name=gpu_name,
                ),
            ),
            row,
        )

    def _health_urls_for_miner(self, miner: ActiveMiner) -> list[str]:
        urls: list[str] = []
        if miner.address.lower() != self.evm_address or int(miner.model_index) != self.model_index:
            return urls
        if self.local_health_url:
            urls.append(self.local_health_url)
        if miner.endpoint:
            urls.append(miner.endpoint.rstrip("/"))
        out: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(url)
        return out

    def _enrich_hardware(self, miners: list[ActiveMiner]) -> None:
        for miner in miners:
            for base_url in self._health_urls_for_miner(miner):
                try:
                    resp = httpx.get(f"{base_url}/health", timeout=3.0)
                    if resp.status_code != 200:
                        continue
                    hw = (resp.json() or {}).get("hardware") or {}
                    miner.gpu_name = hw.get("gpu_name") or ""
                    miner.gpu_count = int(hw.get("gpu_count") or 0)
                    miner.vram_gb = int(hw.get("vram_gb") or 0)
                    miner.compute_capability = hw.get("compute_capability") or ""
                    uuids = hw.get("gpu_uuids") or []
                    miner.gpu_uuids = uuids if isinstance(uuids, list) else []
                    break
                except Exception:
                    continue

    def _workspace_script(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "hot_capacity_workspace" / "bench_combined.py"

    def _workspace_command(self, script: Path) -> list[str]:
        if script.exists():
            return [sys.executable, str(script)]
        return [
            sys.executable,
            "-c",
            "from hot_capacity_workspace.bench_combined import main; main()",
        ]

    def _workspace_build_env(self, script_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        if script_dir.exists():
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{script_dir}:{current_pythonpath}" if current_pythonpath else str(script_dir)
            )
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.exists():
                current_ld_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{torch_lib}:{current_ld_path}" if current_ld_path else str(torch_lib)
                )
        except Exception:
            pass
        env.setdefault("MAX_JOBS", "2")
        return env

    def _workspace_extension_importable(self, script_dir: Path) -> tuple[bool, str]:
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"p={str(script_dir)!r}; "
                "import pathlib; "
                "path=pathlib.Path(p); "
                "sys.path.insert(0, p) if path.exists() else None; "
                "import torch; "
                "import hot_capacity_workspace_cuda; "
                "\ntry:\n"
                "    from hot_capacity_workspace.bench_combined import main as _main\n"
                "except Exception:\n"
                "    import bench_combined as _bench_combined\n"
            ),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=script_dir if script_dir.exists() else Path(__file__).resolve().parents[1],
                env=self._workspace_build_env(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return False, str(exc)
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "")[-500:]

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)

    def _ensure_workspace_extension(self, script_dir: Path) -> bool:
        lock = getattr(self, "_workspace_ext_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._workspace_ext_lock = lock
        with lock:
            if getattr(self, "_workspace_ext_ready", False):
                return True

            importable, import_error = self._workspace_extension_importable(script_dir)
            if importable:
                self._workspace_ext_ready = True
                return True

            build_script = script_dir / "build.py"
            if not build_script.exists():
                bt.logging.warning(
                    "Capacity audit workspace wheel unavailable and private build script missing: "
                    f"{build_script}"
                )
                return False

            bt.logging.info(
                "Capacity audit workspace extension not importable; "
                f"building in-place before audits start ({import_error[:200]})"
            )
            for artifact in script_dir.glob("hot_capacity_workspace_cuda*.so"):
                try:
                    artifact.unlink()
                except OSError:
                    pass

            try:
                proc = subprocess.run(
                    [sys.executable, str(build_script), "build_ext", "--inplace"],
                    cwd=script_dir,
                    env=self._workspace_build_env(script_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900,
                )
            except subprocess.TimeoutExpired as exc:
                tail = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[-1000:]
                bt.logging.warning(f"Capacity audit workspace extension build timed out: {tail}")
                return False
            except Exception as exc:
                bt.logging.warning(f"Capacity audit workspace extension build failed to start: {exc}")
                return False

            if proc.returncode != 0:
                tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-1000:]
                bt.logging.warning(
                    f"Capacity audit workspace extension build failed: rc={proc.returncode} tail={tail}"
                )
                return False

            importable, import_error = self._workspace_extension_importable(script_dir)
            if not importable:
                bt.logging.warning(
                    "Capacity audit workspace extension built but still not importable: "
                    f"{import_error[-500:]}"
                )
                return False

            self._workspace_ext_ready = True
            bt.logging.info("Capacity audit workspace extension ready")
            return True

    def _wait_for_proof_challenge_seed(
        self,
        audit_slot: MinerAuditSlot,
        *,
        transcript: str,
        lease: str,
        subtensor,
        timeout_s: float,
    ) -> Optional[str]:
        deadline = time.time() + max(1.0, float(timeout_s))
        active_subtensor = subtensor
        owns_active_subtensor = False
        last_refresh_at = 0.0
        last_head = 0
        last_head_at = time.time()

        def close_owned_subtensor() -> None:
            nonlocal active_subtensor, owns_active_subtensor
            if owns_active_subtensor and active_subtensor is not None:
                self._close_subtensor(active_subtensor)
            active_subtensor = None
            owns_active_subtensor = False

        while self._running and time.time() < deadline:
            current_block = 0
            if active_subtensor is not None:
                current_block, _head_hash = self._get_live_current_head_block_and_hash(active_subtensor)
                now = time.time()
                if int(current_block or 0) > int(last_head or 0):
                    last_head = int(current_block or 0)
                    last_head_at = now
                elif now - last_head_at > 18.0:
                    close_owned_subtensor()
                    last_head_at = now
                    continue
                if current_block >= audit_slot.proof_challenge_block:
                    challenge_hash = self._get_block_hash(
                        active_subtensor,
                        audit_slot.proof_challenge_block,
                    )
                    if challenge_hash is not None:
                        close_owned_subtensor()
                        return derive_proof_challenge_seed(
                            transcript,
                            challenge_hash,
                            lease,
                            slot_id(audit_slot.slot),
                            0,
                        )
            now = time.time()
            if now - last_refresh_at >= 3.0:
                last_refresh_at = now
                try:
                    close_owned_subtensor()
                    active_subtensor = self._subtensor()
                    owns_active_subtensor = True
                    current_block, _head_hash = self._get_live_current_head_block_and_hash(active_subtensor)
                    if current_block >= audit_slot.proof_challenge_block:
                        challenge_hash = self._get_block_hash(
                            active_subtensor,
                            audit_slot.proof_challenge_block,
                        )
                        if challenge_hash is not None:
                            close_owned_subtensor()
                            return derive_proof_challenge_seed(
                                transcript,
                                challenge_hash,
                                lease,
                                slot_id(audit_slot.slot),
                                0,
                            )
                except Exception:
                    close_owned_subtensor()
            if current_block < audit_slot.proof_challenge_block:
                time.sleep(min(3.0, max(0.5, self.poll_interval_s)))
                continue
            time.sleep(1.0)
        close_owned_subtensor()
        return None

    def _audit_lease(self, audit_slot: MinerAuditSlot, subtensor=None) -> str:
        return lease_id(audit_slot.slot, audit_slot.selection_block // self._epoch_blocks(subtensor))

    def _audit_challenge_timeout_s(self, audit_slot: MinerAuditSlot) -> float:
        return max(
            30.0,
            (audit_slot.proof_challenge_block - audit_slot.audit_block) * 12.0
            + float(self.runtime_cfg.payload_deadline_s or 0.0),
        )

    def _workspace_audit_command(
        self,
        *,
        script: Path,
        audit_slot: MinerAuditSlot,
        lease: str,
        out_dir: Path,
        challenge_file: Path,
        proof_seed: str = "",
        ready_file: Optional[Path] = None,
        start_file: Optional[Path] = None,
        start_timeout_s: float = 0.0,
    ) -> list[str]:
        cmd = [
            *self._workspace_command(script),
            "--child",
            "--out-dir", str(out_dir),
            "--lease-id", lease,
            "--gpu-index", "0",
            "--challenge-file", str(challenge_file),
            "--challenge-timeout-s", str(self._audit_challenge_timeout_s(audit_slot)),
        ]
        if proof_seed:
            cmd.extend(["--seed-hex", proof_seed])
        if ready_file is not None:
            cmd.extend(["--ready-file", str(ready_file)])
        if start_file is not None:
            cmd.extend([
                "--start-file", str(start_file),
                "--start-timeout-s", str(max(1.0, float(start_timeout_s or 0.0))),
            ])
        workload_spec = dict(audit_slot.workload_spec or {})
        for key, value in workload_spec.items():
            if key in {"workload_version", "pass_count"}:
                continue
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def _prepare_audit_process(
        self,
        audit_slot: MinerAuditSlot,
        *,
        start_timeout_s: float,
    ) -> Optional[PreparedAuditProcess]:
        lease = self._audit_lease(audit_slot)
        out_dir = Path(tempfile.mkdtemp(prefix="verathos_capacity_audit_"))
        challenge_file = out_dir / f"{lease}_challenge.txt"
        start_file = out_dir / f"{lease}_start.json"
        ready_file = out_dir / f"{lease}_ready.json"
        script = self._workspace_script()
        if not self._ensure_workspace_extension(script.parent):
            bt.logging.warning(
                f"Capacity audit workload skipped: workspace extension unavailable "
                f"audit_id={audit_slot.audit_id[:12]}"
            )
            return None
        cmd = self._workspace_audit_command(
            script=script,
            audit_slot=audit_slot,
            lease=lease,
            out_dir=out_dir,
            challenge_file=challenge_file,
            ready_file=ready_file,
            start_file=start_file,
            start_timeout_s=start_timeout_s,
        )
        env = self._workspace_build_env(script.parent)
        bt.logging.info(
            f"Capacity audit preparing hot-start workload: audit_id={audit_slot.audit_id[:12]} "
            f"passes={audit_slot.passes} workload={(audit_slot.workload_spec or {}).get('workload_version')}"
        )
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            bt.logging.warning(f"Capacity audit workload failed to prelaunch: {exc}")
            return None
        return PreparedAuditProcess(
            proc=proc,
            out_dir=out_dir,
            lease=lease,
            challenge_file=challenge_file,
            start_file=start_file,
            ready_file=ready_file,
        )

    @staticmethod
    def _terminate_prepared_audit(prepared: PreparedAuditProcess) -> None:
        proc = prepared.proc
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        except Exception:
            pass

    def _run_audit_slot(
        self,
        audit_slot: MinerAuditSlot,
        audit_block_hash: bytes,
        subtensor=None,
        *,
        prepared: Optional[PreparedAuditProcess] = None,
    ) -> None:
        if prepared is None:
            prepared = self._pop_prepared_audit(audit_slot.audit_id)
        else:
            self._drop_prepared_audit(audit_slot.audit_id, prepared)
        if not self._mark_audit_started_once(audit_slot.audit_id):
            if prepared is not None and self._drop_prepared_audit(audit_slot.audit_id, prepared):
                self._terminate_prepared_audit(prepared)
            return
        self._mark_audit_drain(audit_slot, phase="running")
        proof_seed = derive_proof_seed(audit_block_hash, slot_id(audit_slot.slot), 0)
        if prepared is None:
            lease = self._audit_lease(audit_slot, subtensor)
            out_dir = Path(tempfile.mkdtemp(prefix="verathos_capacity_audit_"))
            challenge_file = out_dir / f"{lease}_challenge.txt"
            script = self._workspace_script()
            if not self._ensure_workspace_extension(script.parent):
                bt.logging.warning(
                    f"Capacity audit workload skipped: workspace extension unavailable "
                    f"audit_id={audit_slot.audit_id[:12]}"
                )
                self._extend_busy_selection_until_current_head(audit_slot, subtensor=subtensor)
                self._clear_audit_drain(audit_slot.audit_id)
                return
            cmd = self._workspace_audit_command(
                script=script,
                audit_slot=audit_slot,
                lease=lease,
                out_dir=out_dir,
                challenge_file=challenge_file,
                proof_seed=proof_seed,
            )
            env = self._workspace_build_env(script.parent)
            bt.logging.info(
                f"Capacity audit running workload: audit_id={audit_slot.audit_id[:12]} "
                f"passes={audit_slot.passes} workload={(audit_slot.workload_spec or {}).get('workload_version')}"
            )
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except Exception as exc:
                bt.logging.warning(f"Capacity audit workload failed to start: {exc}")
                self._extend_busy_selection_until_current_head(audit_slot, subtensor=subtensor)
                self._clear_audit_drain(audit_slot.audit_id)
                return
        else:
            proc = prepared.proc
            out_dir = prepared.out_dir
            lease = prepared.lease
            challenge_file = prepared.challenge_file
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                bt.logging.warning(
                    f"Capacity audit hot-start workload exited before B_start: "
                    f"audit_id={audit_slot.audit_id[:12]} rc={proc.returncode} "
                    f"stderr_tail={stderr[-500:]} stdout_tail={stdout[-300:]}"
                )
                self._extend_busy_selection_until_current_head(audit_slot, subtensor=subtensor)
                self._clear_audit_drain(audit_slot.audit_id)
                return
            start_payload = {
                "seed_hex": proof_seed,
                "audit_id": audit_slot.audit_id,
                "B_start": audit_slot.audit_block,
                "t": time.time(),
            }
            tmp_start = prepared.start_file.with_suffix(prepared.start_file.suffix + ".tmp")
            tmp_start.write_text(json.dumps(start_payload, sort_keys=True) + "\n")
            os.replace(tmp_start, prepared.start_file)
            bt.logging.info(
                f"Capacity audit released hot-start workload: audit_id={audit_slot.audit_id[:12]} "
                f"passes={audit_slot.passes} workload={(audit_slot.workload_spec or {}).get('workload_version')}"
            )

        pass0_sent = False
        final_sent = False
        pass0_root = ""
        final_root = ""
        transcript = ""
        final_timing_data: dict = {}
        challenge_wait_s = (
            max(0, audit_slot.proof_challenge_block - audit_slot.audit_block) * 12.0
            + float(self.runtime_cfg.payload_deadline_s or 0.0)
        )
        deadline = time.time() + max(120.0, audit_slot.deadline_s + 90.0 + challenge_wait_s)
        pass0_path = out_dir / f"{lease}_pass0.json"
        final_path = out_dir / f"{lease}_final_timing.json"

        def publish_pass0(root: str) -> bool:
            nonlocal pass0_root, pass0_sent
            candidate = str(root or "").strip()
            if not candidate:
                return False
            pass0_root = candidate
            self._publish_receipt(self._pass0_artifact(audit_slot, pass0_root))
            pass0_sent = True
            return True

        def read_pass0_file() -> bool:
            if not pass0_path.exists():
                return False
            data = json.loads(pass0_path.read_text())
            raw_root = data.get("root")
            if raw_root in (None, "", []):
                return False
            return publish_pass0(_root_hex(raw_root))

        while time.time() < deadline:
            if not pass0_sent:
                read_pass0_file()
            if not final_sent and final_path.exists():
                data = json.loads(final_path.read_text())
                final_timing_data = data if isinstance(data, dict) else {}
                if not pass0_sent:
                    if not read_pass0_file():
                        raw_pass0_root = final_timing_data.get("pass0_root")
                        if raw_pass0_root:
                            publish_pass0(_root_hex(raw_pass0_root))
                final_root = _root_hex(data.get("root") or [])
                transcript = str(data.get("transcript_root") or "")
                if not transcript:
                    transcript = transcript_root([pass0_root, final_root])
                self._publish_receipt(
                    self._final_artifact(
                        audit_slot,
                        pass0_root,
                        final_root,
                        transcript,
                        final_timing=final_timing_data,
                    )
                )
                final_sent = True
                if subtensor is None:
                    try:
                        subtensor = self._subtensor()
                    except Exception:
                        subtensor = None
                if subtensor is not None:
                    challenge_seed = self._wait_for_proof_challenge_seed(
                        audit_slot,
                        transcript=transcript,
                        lease=lease,
                        subtensor=subtensor,
                        timeout_s=challenge_wait_s,
                    )
                    if challenge_seed:
                        self._write_text_atomic(challenge_file, challenge_seed)
                    else:
                        bt.logging.warning(
                            f"Capacity audit proof challenge unavailable: "
                            f"audit_id={audit_slot.audit_id[:12]} B_proof={audit_slot.proof_challenge_block}"
                        )
                break
            if proc.poll() is not None and final_sent:
                break
            if proc.poll() is not None and not final_path.exists():
                break
            time.sleep(0.02)

        proof_assembly_timeout = max(5.0, float(self.runtime_cfg.payload_deadline_s or 0.0) + 30.0)
        try:
            stdout, stderr = proc.communicate(timeout=proof_assembly_timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        if not final_sent:
            bt.logging.warning(
                f"Capacity audit workload did not produce final receipt: "
                f"rc={proc.poll()} stderr_tail={stderr[-500:]} stdout_tail={stdout[-300:]}"
            )
        elif not pass0_sent:
            bt.logging.warning(f"Capacity audit final sent without pass0 for audit_id={audit_slot.audit_id[:12]}")
        else:
            final_summary = {}
            final_summary_path = out_dir / f"{lease}_final.json"
            if final_summary_path.exists():
                try:
                    final_summary = json.loads(final_summary_path.read_text())
                except Exception:
                    final_summary = {}
            else:
                bt.logging.warning(
                    f"Capacity audit workload missing final proof summary: "
                    f"audit_id={audit_slot.audit_id[:12]} rc={proc.poll()} "
                    f"stderr_tail={stderr[-500:]} stdout_tail={stdout[-300:]}"
                )
            proof_payload = self._proof_payload_artifact(
                audit_slot,
                pass0_root=pass0_root,
                final_root=final_root,
                transcript=transcript,
                lease=lease,
                final_summary=final_summary,
            )
            if proof_payload is None:
                bt.logging.warning(
                    f"Capacity audit proof payload missing verifier proof: "
                    f"audit_id={audit_slot.audit_id[:12]}"
                )
            else:
                self._publish_proof(proof_payload)
                bt.logging.info(f"Capacity audit artifacts published: audit_id={audit_slot.audit_id[:12]}")
        self._extend_busy_selection_until_current_head(audit_slot, subtensor=subtensor)
        self._clear_audit_drain(audit_slot.audit_id)

    def _base_artifact(self, audit_slot: MinerAuditSlot) -> dict:
        slot = audit_slot.slot
        return {
            "protocol_version": PROTOCOL_VERSION,
            "audit_id": audit_slot.audit_id,
            "slot_id": slot_id(slot),
            "address": self.evm_address,
            "model_index": self.model_index,
            "claimed_gpu_class": audit_slot.gpu_class_name,
            "gpu_index": 0,
            "B_select": audit_slot.selection_block,
            "B_start": audit_slot.audit_block,
            "B_proof": audit_slot.proof_challenge_block,
            "pass_count": audit_slot.passes,
        }

    def _pass0_artifact(self, audit_slot: MinerAuditSlot, pass0_root: str) -> dict:
        artifact = self._base_artifact(audit_slot)
        artifact.update({
            "artifact_type": "capacity_audit_pass0_receipt",
            "pass0_root": pass0_root,
            "pass0_transcript_commit": pass0_root,
        })
        artifact["miner_signature"] = sign_artifact(artifact, self.evm_private_key)
        return artifact

    def _final_artifact(
        self,
        audit_slot: MinerAuditSlot,
        pass0_root: str,
        final_root: str,
        transcript: str,
        *,
        final_timing: Optional[dict] = None,
    ) -> dict:
        artifact = self._base_artifact(audit_slot)
        combined_commit = {}
        if isinstance(final_timing, dict) and str(final_timing.get("proof_format") or "") == COMBINED_PROOF_FORMAT:
            combined_commit = {
                "format": COMBINED_PROOF_FORMAT,
                "workload_version": final_timing.get("workload_version"),
                "pass_count": final_timing.get("pass_count"),
                "capacity_transcript_root": final_timing.get("capacity_transcript_root"),
                "capacity_tail_transcript_root": final_timing.get("capacity_tail_transcript_root"),
                "fp64_transcript_root": final_timing.get("fp64_transcript_root"),
                "combined_transcript_root": final_timing.get("combined_transcript_root"),
                "capacity_params": final_timing.get("capacity_params"),
                "capacity_tail_params": final_timing.get("capacity_tail_params"),
                "fp64_params": final_timing.get("fp64_params"),
                "workspace_mode": final_timing.get("workspace_mode"),
                "timed_cuda_component_s": final_timing.get("timed_cuda_component_s"),
                "timed_wall_s": final_timing.get("timed_wall_s"),
            }
        artifact.update({
            "artifact_type": "capacity_audit_final_receipt",
            "pass0_root": pass0_root,
            "final_root": final_root,
            "final_transcript_commit": transcript,
        })
        if combined_commit:
            artifact["combined"] = combined_commit
        artifact["miner_signature"] = sign_artifact(artifact, self.evm_private_key)
        return artifact

    def _proof_payload_artifact(
        self,
        audit_slot: MinerAuditSlot,
        *,
        pass0_root: str,
        final_root: str,
        transcript: str,
        lease: str,
        final_summary: dict,
    ) -> Optional[dict]:
        proof_payload = final_summary.get("proof_payload")
        if (
            isinstance(proof_payload, dict)
            and str(proof_payload.get("format") or "") == COMBINED_PROOF_FORMAT
        ):
            capacity_proof = proof_payload.get("capacity_proof") if isinstance(proof_payload, dict) else {}
            proof_sampled = capacity_proof.get("sampled") if isinstance(capacity_proof, dict) else None
            try:
                sampled = int((proof_sampled or {}).get("pass_index"))
            except Exception:
                sampled = 0
            artifact = self._base_artifact(audit_slot)
            artifact.update({
                "artifact_type": "capacity_audit_proof_payload",
                "sampled_pass_index": sampled,
                "sampled_opening": {
                    "lease_id": lease,
                    "transcript_root": transcript,
                    "pass0_root": pass0_root,
                    "final_root": final_root,
                    "pass_index": sampled,
                },
                "sampled_pass_proof": proof_payload,
            })
            artifact["miner_signature"] = sign_artifact(artifact, self.evm_private_key)
            return artifact
        return None

    def _publish_receipt(self, artifact: dict) -> None:
        self._publish_artifact(
            "/capacity/audit/v1/receipt",
            artifact,
            attempts=3,
            retry_delay_s=0.25,
        )

    def _publish_proof(self, artifact: dict) -> None:
        attempts = max(1, int(float(self.runtime_cfg.payload_deadline_s or 0.0) // 5.0))
        self._publish_artifact(
            "/capacity/audit/v1/proof",
            artifact,
            attempts=min(12, attempts),
            retry_delay_s=5.0,
        )

    def _publish_artifact(
        self,
        path: str,
        artifact: dict,
        *,
        attempts: int = 1,
        retry_delay_s: float = 0.0,
    ) -> None:
        audit_id = str(artifact.get("audit_id") or "")
        artifact_type = str(artifact.get("artifact_type") or "")
        for attempt in range(max(1, int(attempts))):
            pending: list[str] = []
            validator_urls = self._validator_endpoint_urls(force_refresh=attempt > 0)
            if audit_id:
                validator_urls = tuple(
                    base for base in validator_urls if not self._validator_rejected_audit(base, audit_id)
                )
            if not validator_urls:
                bt.logging.warning(
                    f"Capacity audit publish has no validator endpoints: "
                    f"audit_id={audit_id[:12]} "
                    f"type={artifact_type}"
                )
                return
            for base in validator_urls:
                try:
                    resp = httpx.post(
                        f"{base}{path}",
                        json=artifact,
                        timeout=5.0,
                    )
                    if resp.status_code < 300:
                        self._record_validator_publish_result(base, True)
                        continue
                    retryable = resp.status_code in (409, 425, 429, 500, 502, 503, 504)
                    bt.logging.warning(
                        f"Capacity audit publish failed: audit_id={audit_id[:12]} "
                        f"type={artifact_type} B_select={artifact.get('B_select')} "
                        f"B_start={artifact.get('B_start')} B_proof={artifact.get('B_proof')} "
                        f"url={base}{path} status={resp.status_code} body={resp.text[:200]}"
                    )
                    if self._is_unknown_audit_slot_response(resp.status_code, resp.text):
                        self._record_validator_audit_rejection(base, audit_id)
                        continue
                    failure_kind = self._validator_endpoint_failure_kind(resp.status_code)
                    if failure_kind:
                        self._record_validator_publish_result(
                            base,
                            False,
                            failure_kind=failure_kind,
                        )
                    if retryable:
                        pending.append(base)
                except Exception as exc:
                    bt.logging.warning(
                        f"Capacity audit publish error: audit_id={audit_id[:12]} "
                        f"type={artifact_type} url={base}{path}: {exc}"
                    )
                    self._record_validator_publish_result(base, False, failure_kind="transient")
                    pending.append(base)
            if not pending or attempt + 1 >= max(1, int(attempts)):
                return
            time.sleep(max(0.1, float(retry_delay_s)))

    def _validator_rejected_audit(self, endpoint: str, audit_id: str) -> bool:
        rejected = getattr(self, "_audit_endpoint_rejections", None)
        if not isinstance(rejected, dict):
            return False
        return str(endpoint or "") in rejected.get(str(audit_id or ""), set())

    def _record_validator_audit_rejection(self, endpoint: str, audit_id: str) -> None:
        audit_id = str(audit_id or "")
        endpoint = str(endpoint or "")
        if not audit_id or not endpoint:
            return
        rejected = getattr(self, "_audit_endpoint_rejections", None)
        if not isinstance(rejected, dict):
            rejected = {}
            self._audit_endpoint_rejections = rejected
        rejected.setdefault(audit_id, set()).add(endpoint)
        bt.logging.info(
            f"Capacity audit validator did not recognize slot; "
            f"skipping later artifacts for audit_id={audit_id[:12]} endpoint={endpoint}"
        )

    def _validator_endpoint_urls(self, *, force_refresh: bool = False) -> tuple[str, ...]:
        resolver = getattr(self, "_validator_endpoint_resolver", None)
        if resolver is None:
            return tuple(getattr(self, "validator_urls", ()) or ())
        return resolver.current_urls(force_refresh=force_refresh)

    def _record_validator_publish_result(
        self,
        endpoint: str,
        success: bool,
        *,
        failure_kind: str = "hard",
    ) -> None:
        resolver = getattr(self, "_validator_endpoint_resolver", None)
        if resolver is not None:
            try:
                resolver.record_publish_result(endpoint, success, failure_kind=failure_kind)
            except TypeError:
                resolver.record_publish_result(endpoint, success)

    @staticmethod
    def _validator_endpoint_failure_kind(status_code: int) -> str:
        status = int(status_code)
        if status in {401, 403, 404}:
            return "hard"
        if status == 429 or status >= 500:
            return "transient"
        return ""

    @staticmethod
    def _is_validator_endpoint_failure(status_code: int) -> bool:
        return bool(CapacityAuditMinerWorker._validator_endpoint_failure_kind(status_code))

    @staticmethod
    def _is_unknown_audit_slot_response(status_code: int, text: str) -> bool:
        if int(status_code) != 400:
            return False
        try:
            payload = json.loads(str(text or ""))
        except Exception:
            return "unknown audit slot" in str(text or "").lower()
        if isinstance(payload, dict):
            return str(payload.get("error") or "").lower() == "unknown audit slot"
        return False
