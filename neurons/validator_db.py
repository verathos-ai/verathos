"""SQLite-backed validator state database.

Replaces scattered JSON files with a single transactional store for:
- Miner-model entry lifecycle (discovery, deregistration, model switches)
- EMA scores (survive restarts — no more 10-epoch recovery)
- Probation state (inlined with scoring, always consistent)
- UID resolution cache (fallback when RPC is unavailable)
- Epoch audit log

Follows the ``verallm/api/receipt_store.py`` pattern: SQLite WAL mode,
``threading.Lock`` around all writes, ``CREATE TABLE IF NOT EXISTS`` for
zero-migration startup.
"""

from __future__ import annotations

import json
import logging
import math
import bittensor as bt
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from neurons.shared_state import AuditDrain, MinerEntry, ValidatorSharedState

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
    "verathos_validator.db",
)

CAPACITY_AUDIT_HISTORY_RETENTION_SECONDS = 48 * 60 * 60
_SCHEMA_VERSION = "1"


def _coerce_nonnegative_int(value: object) -> int:
    """Return a DB-safe non-negative integer for optional miner metadata."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _debug_error_kind(message: object) -> str:
    """Return a stable public issue code for an upstream canary/chat error."""
    msg = str(message or "").lower()
    if not msg:
        return ""
    if "401" in msg or "unauthorized" in msg:
        return "chat_unauthorized"
    if "403" in msg or "forbidden" in msg:
        return "chat_forbidden"
    if "404" in msg or "not found" in msg:
        return "chat_not_found"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg or "network is unreachable" in msg:
        return "connection_failed"
    if "ssl" in msg or "certificate" in msg or "tls" in msg:
        return "tls_error"
    return "chat_error"


_DEBUG_ERROR_SUMMARIES = {
    "chat_unauthorized": "Inference request was unauthorized.",
    "chat_forbidden": "Inference request was forbidden.",
    "chat_not_found": "Inference route was not found.",
    "timeout": "Inference request timed out.",
    "connection_failed": "Validator could not connect to the inference endpoint.",
    "tls_error": "Inference endpoint failed TLS validation.",
    "chat_error": "Inference request failed.",
    "proof_failure": "Proof verification failed.",
    "tee_failure": "TEE attestation verification failed.",
}


def _debug_public_error_summary(kind: str) -> str:
    return _DEBUG_ERROR_SUMMARIES.get(str(kind or ""), "Request failed.")


def _debug_public_reason(value: object, fallback: str) -> str:
    """Expose only stable machine reason codes, never arbitrary exception text."""
    reason = str(value or "").strip()
    if (
        reason
        and len(reason) <= 96
        and all(ch.isalnum() or ch in {"_", "-", "."} for ch in reason)
    ):
        return reason
    return fallback


_CAPACITY_AUDIT_UNSUPPORTED_PAYLOAD_FAILURES = {
    "unsupported_proof_payload_format",
    "unsupported_combined_format",
    "unsupported_combined_workload",
}


def _debug_capacity_audit_invalid_failure(
    verdict: object,
    proof_status: object,
    failure_reason: object,
) -> bool:
    reason = str(failure_reason or "")
    return (
        str(verdict or "") == "hard_proof_miss"
        and (
            reason == "pass0_root_mismatch"
            or (
                str(proof_status or "") == "invalid_payload"
                and reason not in _CAPACITY_AUDIT_UNSUPPORTED_PAYLOAD_FAILURES
            )
        )
    )


class ValidatorStateDB:
    """SQLite-backed validator state with thread-safe writes.

    Thread-safety: all mutating operations hold ``_lock``.  SQLite WAL
    mode allows concurrent reads from other connections / threads.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, analytics: bool = True) -> None:
        self._db_path = db_path
        self._analytics = analytics
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── Schema bootstrap ─────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create database, tables, and indices if they don't exist."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS miner_entries (
                address         TEXT NOT NULL,
                model_index     INTEGER NOT NULL,

                model_id        TEXT NOT NULL,
                endpoint        TEXT NOT NULL,
                quant           TEXT NOT NULL DEFAULT '',
                max_context_len INTEGER NOT NULL DEFAULT 0,

                -- Per-address fields (denormalized, same for all rows with same address)
                bittensor_uid   INTEGER,
                hotkey_ss58     TEXT,
                coldkey_ss58    TEXT,

                first_seen_epoch INTEGER NOT NULL,
                last_seen_epoch  INTEGER NOT NULL,
                is_active        INTEGER NOT NULL DEFAULT 1,

                ema_score       REAL NOT NULL DEFAULT 0.0,
                total_epochs    INTEGER NOT NULL DEFAULT 0,
                scored_epochs   INTEGER NOT NULL DEFAULT 0,

                probation_entered_epoch     INTEGER,
                probation_consecutive_passes INTEGER NOT NULL DEFAULT 0,
                probation_required_passes   INTEGER NOT NULL DEFAULT 3,
                probation_escalation_epochs INTEGER NOT NULL DEFAULT 5,

                tee_enabled     INTEGER NOT NULL DEFAULT 0,
                tee_platform    TEXT NOT NULL DEFAULT '',

                gpu_name        TEXT NOT NULL DEFAULT '',
                gpu_count       INTEGER NOT NULL DEFAULT 0,
                vram_gb         INTEGER NOT NULL DEFAULT 0,
                compute_capability TEXT NOT NULL DEFAULT '',
                gpu_uuids       TEXT NOT NULL DEFAULT '[]',

                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL,

                PRIMARY KEY (address, model_index)
            );

            CREATE INDEX IF NOT EXISTS idx_miner_entries_model
                ON miner_entries(model_id);

            CREATE INDEX IF NOT EXISTS idx_miner_entries_active
                ON miner_entries(is_active) WHERE is_active = 1;

            CREATE TABLE IF NOT EXISTS uid_ownership (
                uid                 INTEGER PRIMARY KEY,
                hotkey_ss58         TEXT NOT NULL,
                evm_address         TEXT NOT NULL,
                generation          INTEGER NOT NULL DEFAULT 1,
                identity_start_epoch INTEGER NOT NULL DEFAULT 0,
                updated_at          REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_uid_ownership_evm
                ON uid_ownership(evm_address);

            CREATE TABLE IF NOT EXISTS epoch_log (
                epoch_number    INTEGER PRIMARY KEY,
                start_block     INTEGER NOT NULL,
                miner_count     INTEGER NOT NULL DEFAULT 0,
                receipt_count   INTEGER NOT NULL DEFAULT 0,
                weight_set      INTEGER NOT NULL DEFAULT 0,
                closed_at       REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS validator_meta (
                key     TEXT PRIMARY KEY,
                value   TEXT NOT NULL
            );

            -- Analytics: individual canary test results
            CREATE TABLE IF NOT EXISTS canary_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                network             TEXT NOT NULL,
                chain_id            INTEGER NOT NULL,
                netuid              INTEGER NOT NULL,
                epoch_number        INTEGER NOT NULL,
                block_number        INTEGER NOT NULL,
                miner_address       TEXT NOT NULL,
                miner_uid           INTEGER,
                miner_hotkey_ss58   TEXT,
                miner_coldkey_ss58  TEXT,
                model_id            TEXT NOT NULL,
                model_index         INTEGER NOT NULL,
                endpoint            TEXT NOT NULL,
                test_type           TEXT NOT NULL,
                test_index          INTEGER NOT NULL,
                proof_requested     INTEGER NOT NULL,
                enable_thinking     INTEGER NOT NULL,
                temperature         REAL NOT NULL,
                max_new_tokens      INTEGER NOT NULL,
                status              TEXT NOT NULL,
                error_message       TEXT,
                ttft_ms             REAL,
                tokens_generated    INTEGER,
                inference_ms        REAL,
                tokens_per_sec      REAL,
                prompt_tokens       INTEGER,
                proof_verified      INTEGER,
                proof_failure_reason TEXT,
                prove_ms            REAL,
                commitment_ms       REAL,
                verify_ms           REAL,
                commitment_hash     TEXT,
                receipt_pushed      INTEGER NOT NULL DEFAULT 0,
                tee_requested       INTEGER NOT NULL DEFAULT 0,
                tee_verified        INTEGER,
                created_at          REAL NOT NULL
            );

            -- Network-wide receipts from ALL validators (pulled from miners at epoch close)
            CREATE TABLE IF NOT EXISTS network_receipts (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_number        INTEGER NOT NULL,
                miner_address       TEXT NOT NULL,
                miner_hotkey_ss58   TEXT,
                miner_coldkey_ss58  TEXT,
                model_id            TEXT NOT NULL,
                model_index         INTEGER NOT NULL,
                validator_hotkey    TEXT NOT NULL,
                is_own              INTEGER NOT NULL,
                is_canary           INTEGER NOT NULL,
                ttft_ms             REAL NOT NULL,
                tokens_generated    INTEGER NOT NULL,
                generation_time_ms  REAL NOT NULL,
                tokens_per_sec      REAL NOT NULL,
                prompt_tokens       INTEGER NOT NULL DEFAULT 0,
                proof_verified      INTEGER NOT NULL DEFAULT 0,
                proof_requested     INTEGER NOT NULL DEFAULT 0,
                tee_attestation_verified INTEGER DEFAULT NULL,
                commitment_hash     TEXT,
                timestamp           INTEGER NOT NULL,
                network             TEXT NOT NULL,
                netuid              INTEGER NOT NULL,
                created_at          REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_network_receipts_epoch
                ON network_receipts(epoch_number);
            CREATE INDEX IF NOT EXISTS idx_network_receipts_miner
                ON network_receipts(miner_address, epoch_number);
            CREATE INDEX IF NOT EXISTS idx_network_receipts_validator
                ON network_receipts(validator_hotkey, epoch_number);

            CREATE INDEX IF NOT EXISTS idx_canary_epoch
                ON canary_results(epoch_number);
            CREATE INDEX IF NOT EXISTS idx_canary_miner
                ON canary_results(miner_address, epoch_number);

            -- Analytics: per-miner scores at each epoch close
            CREATE TABLE IF NOT EXISTS epoch_scores (
                epoch_number        INTEGER NOT NULL,
                miner_address       TEXT NOT NULL,
                model_index         INTEGER NOT NULL,
                model_id            TEXT NOT NULL,
                miner_uid           INTEGER,
                miner_hotkey_ss58   TEXT,
                miner_coldkey_ss58  TEXT,
                own_receipts        INTEGER NOT NULL,
                all_receipts        INTEGER NOT NULL,
                expected_receipts   INTEGER NOT NULL,
                proof_tests         INTEGER NOT NULL,
                proof_failures      INTEGER NOT NULL,
                tee_tests           INTEGER NOT NULL DEFAULT 0,
                tee_failures        INTEGER NOT NULL DEFAULT 0,
                tee_verified        INTEGER NOT NULL DEFAULT 0,
                epoch_score         REAL,
                demand_bonus        REAL NOT NULL DEFAULT 1.0,
                ema_score           REAL NOT NULL,
                peer_median_ttft_ms REAL,
                peer_median_tps     REAL,
                network             TEXT NOT NULL,
                netuid              INTEGER NOT NULL,
                created_at          REAL NOT NULL,
                PRIMARY KEY (epoch_number, miner_address, model_index)
            );

            CREATE TABLE IF NOT EXISTS capacity_audit_windows (
                audit_id             TEXT PRIMARY KEY,
                epoch_number         INTEGER NOT NULL,
                selection_block      INTEGER NOT NULL,
                audit_block          INTEGER NOT NULL,
                proof_challenge_block INTEGER NOT NULL DEFAULT 0,
                selection_block_hash TEXT NOT NULL DEFAULT '',
                audit_block_hash     TEXT NOT NULL DEFAULT '',
                proof_challenge_block_hash TEXT NOT NULL DEFAULT '',
                cohort_seed          TEXT NOT NULL DEFAULT '',
                status               TEXT NOT NULL DEFAULT 'scheduled',
                chain_status         TEXT NOT NULL DEFAULT 'pending',
                audit_start_observed_at REAL,
                proof_challenge_observed_at REAL,
                selection_finalized_at REAL,
                audit_finalized_at REAL,
                proof_challenge_finalized_at REAL,
                created_at           REAL NOT NULL,
                updated_at           REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_windows_epoch
                ON capacity_audit_windows(epoch_number);

            CREATE TABLE IF NOT EXISTS capacity_audit_slots (
                audit_id             TEXT NOT NULL,
                miner_address        TEXT NOT NULL,
                model_index          INTEGER NOT NULL,
                miner_uid            INTEGER,
                miner_hotkey_ss58    TEXT NOT NULL DEFAULT '',
                endpoint             TEXT NOT NULL,
                model_id             TEXT NOT NULL,
                quant                TEXT NOT NULL DEFAULT '',
                max_context_len      INTEGER NOT NULL DEFAULT 0,
                gpu_name             TEXT NOT NULL DEFAULT '',
                gpu_count            INTEGER NOT NULL DEFAULT 0,
                vram_gb              INTEGER NOT NULL DEFAULT 0,
                group_key            TEXT NOT NULL DEFAULT '',
                slot_id              TEXT NOT NULL,
                lease_id             TEXT NOT NULL,
                claimed_gpu_class    TEXT NOT NULL DEFAULT '',
                pass_count           INTEGER NOT NULL DEFAULT 0,
                deadline_s           REAL NOT NULL DEFAULT 0.0,
                transport_grace_s    REAL NOT NULL DEFAULT 0.0,
                payload_deadline_s   REAL NOT NULL DEFAULT 0.0,
                drain_until_ts       REAL NOT NULL DEFAULT 0.0,
                pass0_received_at    REAL,
                final_received_at    REAL,
                proof_received_at    REAL,
                pass0_root           TEXT NOT NULL DEFAULT '',
                final_root           TEXT NOT NULL DEFAULT '',
                transcript_root      TEXT NOT NULL DEFAULT '',
                timing_status        TEXT NOT NULL DEFAULT 'pending',
                proof_status         TEXT NOT NULL DEFAULT 'pending',
                proof_verify_ms      REAL,
                verdict              TEXT NOT NULL DEFAULT 'pending',
                failure_reason       TEXT,
                pass0_artifact       TEXT,
                final_artifact       TEXT,
                proof_artifact_path  TEXT,
                created_at           REAL NOT NULL,
                updated_at           REAL NOT NULL,
                PRIMARY KEY (audit_id, miner_address, model_index)
            );

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_slot
                ON capacity_audit_slots(miner_address, model_index, created_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_verdict
                ON capacity_audit_slots(verdict, created_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_drains
                ON capacity_audit_slots(drain_until_ts, verdict);

            CREATE TABLE IF NOT EXISTS capacity_audit_history (
                audit_id             TEXT NOT NULL,
                miner_address        TEXT NOT NULL,
                model_index          INTEGER NOT NULL,
                miner_uid            INTEGER,
                miner_hotkey_ss58    TEXT NOT NULL DEFAULT '',
                endpoint             TEXT NOT NULL DEFAULT '',
                model_id             TEXT NOT NULL DEFAULT '',
                quant                TEXT NOT NULL DEFAULT '',
                max_context_len      INTEGER NOT NULL DEFAULT 0,
                gpu_name             TEXT NOT NULL DEFAULT '',
                gpu_count            INTEGER NOT NULL DEFAULT 0,
                vram_gb              INTEGER NOT NULL DEFAULT 0,
                group_key            TEXT NOT NULL DEFAULT '',
                slot_id              TEXT NOT NULL DEFAULT '',
                lease_id             TEXT NOT NULL DEFAULT '',
                claimed_gpu_class    TEXT NOT NULL DEFAULT '',
                pass_count           INTEGER NOT NULL DEFAULT 0,
                epoch_number         INTEGER NOT NULL DEFAULT 0,
                selection_block      INTEGER NOT NULL DEFAULT 0,
                audit_block          INTEGER NOT NULL DEFAULT 0,
                proof_challenge_block INTEGER NOT NULL DEFAULT 0,
                verdict              TEXT NOT NULL DEFAULT '',
                timing_status        TEXT NOT NULL DEFAULT '',
                proof_status         TEXT NOT NULL DEFAULT '',
                failure_reason       TEXT,
                proof_verify_ms      REAL,
                pass0_received_at    REAL,
                final_received_at    REAL,
                proof_received_at    REAL,
                slot_created_at      REAL NOT NULL,
                slot_updated_at      REAL NOT NULL,
                archived_at          REAL NOT NULL,
                PRIMARY KEY (audit_id, miner_address, model_index)
            );

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_archived
                ON capacity_audit_history(archived_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_uid
                ON capacity_audit_history(miner_uid, archived_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_outcome
                ON capacity_audit_history(verdict, proof_status, archived_at);
        """)
        self._conn.commit()

        self._ensure_column(
            "capacity_audit_windows",
            "audit_start_observed_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "proof_challenge_block",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "proof_challenge_block_hash",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "proof_challenge_observed_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "chain_status",
            "TEXT NOT NULL DEFAULT 'pending'",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "selection_finalized_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "audit_finalized_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "proof_challenge_finalized_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "proof_verify_ms",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "miner_hotkey_ss58",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            "capacity_audit_history",
            "miner_hotkey_ss58",
            "TEXT NOT NULL DEFAULT ''",
        )

        # One-time migration: prune historical duplicate rows then install
        # the unique index. Guard it with a DB marker and the actual index
        # existence check; on production analytics DBs this table can have
        # millions of rows, so re-running the prune query on every restart
        # can block validator startup for minutes.
        dedupe_meta_key = "network_receipts_dedupe_v1"
        dedupe_index_exists = self._conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type = 'index' AND name = 'uniq_network_receipts_signature'"""
        ).fetchone() is not None

        if dedupe_index_exists:
            if self._raw_get_meta(dedupe_meta_key) != "done":
                self._raw_set_meta(dedupe_meta_key, "done")
        else:
            try:
                cur = self._conn.execute("""
                    DELETE FROM network_receipts
                    WHERE commitment_hash IS NOT NULL
                      AND id NOT IN (
                        SELECT MIN(id) FROM network_receipts
                        WHERE commitment_hash IS NOT NULL
                        GROUP BY epoch_number, validator_hotkey, commitment_hash
                      )
                """)
                removed = cur.rowcount or 0
                if removed > 0:
                    bt.logging.debug(
                        f"network_receipts: pruned {removed} historical duplicate rows"
                    )
                self._conn.commit()
            except Exception as e:
                bt.logging.warning(f"network_receipts duplicate prune failed: {e}")

            try:
                self._conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS uniq_network_receipts_signature
                        ON network_receipts(epoch_number, validator_hotkey, commitment_hash)
                """)
                self._conn.commit()
                self._raw_set_meta(dedupe_meta_key, "done")
            except Exception as e:
                bt.logging.warning(f"network_receipts unique index creation failed: {e}")

        # Ensure schema version is recorded
        existing = self._raw_get_meta("schema_version")
        if existing is None:
            self._raw_set_meta("schema_version", _SCHEMA_VERSION)

        bt.logging.info(f"Validator state DB initialized: {self._db_path}")

    # ── Miner entries ────────────────────────────────────────────────

    def upsert_entry(
        self,
        address: str,
        model_index: int,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
        epoch: int,
        hotkey_ss58: str = "",
        coldkey_ss58: str = "",
        tee_enabled: bool = False,
        tee_platform: str = "",
        gpu_name: str = "",
        gpu_count: int = 0,
        vram_gb: int = 0,
        compute_capability: str = "",
        gpu_uuids: List[str] | None = None,
    ) -> bool:
        """Insert or update a miner-model entry from discovery.

        If the entry exists but ``model_id`` has changed (model switch),
        the EMA score and probation are reset.

        Returns:
            True if the model_id changed (score was reset).
        """
        address = address.lower()
        now = time.time()
        model_switched = False
        tee_platform = tee_platform or ""
        gpu_name = gpu_name or ""
        gpu_count = _coerce_nonnegative_int(gpu_count)
        vram_gb = _coerce_nonnegative_int(vram_gb)
        compute_capability = compute_capability or ""
        _gpu_uuids_json = json.dumps(gpu_uuids or [])

        with self._lock:
            row = self._conn.execute(
                "SELECT model_id, endpoint, quant, max_context_len FROM miner_entries "
                "WHERE address = ? AND model_index = ?",
                (address, model_index),
            ).fetchone()

            if row is None:
                # New entry
                self._conn.execute(
                    """INSERT INTO miner_entries (
                        address, model_index, model_id, endpoint, quant,
                        max_context_len, first_seen_epoch, last_seen_epoch,
                        is_active, ema_score, total_epochs, scored_epochs,
                        probation_entered_epoch, probation_consecutive_passes,
                        probation_required_passes, probation_escalation_epochs,
                        tee_enabled, tee_platform,
                        gpu_name, gpu_count, vram_gb, compute_capability, gpu_uuids,
                        hotkey_ss58, coldkey_ss58, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0.0, 0, 0,
                              NULL, 0, 3, 5, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (address, model_index, model_id, endpoint, quant,
                     max_context_len, epoch, epoch,
                     1 if tee_enabled else 0, tee_platform,
                     gpu_name, gpu_count, vram_gb, compute_capability, _gpu_uuids_json,
                     hotkey_ss58, coldkey_ss58, now, now),
                )
            else:
                old_model_id = row["model_id"]
                if old_model_id != model_id:
                    # Model switch: reset score and probation
                    model_switched = True
                    self._conn.execute(
                        """UPDATE miner_entries SET
                            model_id = ?, endpoint = ?, quant = ?,
                            max_context_len = ?, last_seen_epoch = ?,
                            is_active = 1, ema_score = 0.0,
                            total_epochs = 0, scored_epochs = 0,
                            probation_entered_epoch = NULL,
                            probation_consecutive_passes = 0,
                            tee_enabled = ?, tee_platform = ?,
                            gpu_name = ?, gpu_count = ?, vram_gb = ?,
                            compute_capability = ?, gpu_uuids = ?,
                            hotkey_ss58 = COALESCE(NULLIF(?, ''), hotkey_ss58),
                            coldkey_ss58 = COALESCE(NULLIF(?, ''), coldkey_ss58),
                            updated_at = ?
                        WHERE address = ? AND model_index = ?""",
                        (model_id, endpoint, quant, max_context_len,
                         epoch, 1 if tee_enabled else 0, tee_platform,
                         gpu_name, gpu_count, vram_gb, compute_capability,
                         _gpu_uuids_json,
                         hotkey_ss58, coldkey_ss58, now, address, model_index),
                    )
                    bt.logging.info(f"Model switch for {address[:10]} idx={model_index}: {old_model_id} -> {model_id} (score reset)")
                else:
                    # Same model — update endpoint/quant/epoch, keep scores
                    registration_changed = (
                        str(row["endpoint"] or "") != endpoint
                        or str(row["quant"] or "") != quant
                        or int(row["max_context_len"] or 0) != int(max_context_len or 0)
                    )
                    self._conn.execute(
                        """UPDATE miner_entries SET
                            endpoint = ?, quant = ?, max_context_len = ?,
                            last_seen_epoch = ?, is_active = 1,
                            tee_enabled = ?, tee_platform = ?,
                            gpu_name = CASE
                                WHEN ? THEN ? ELSE COALESCE(NULLIF(?, ''), gpu_name) END,
                            gpu_count = CASE
                                WHEN ? THEN ? WHEN ? > 0 THEN ? ELSE gpu_count END,
                            vram_gb = CASE
                                WHEN ? THEN ? WHEN ? > 0 THEN ? ELSE vram_gb END,
                            compute_capability = CASE
                                WHEN ? THEN ? ELSE COALESCE(NULLIF(?, ''), compute_capability) END,
                            gpu_uuids = CASE
                                WHEN ? THEN ? WHEN ? != '[]' THEN ? ELSE gpu_uuids END,
                            hotkey_ss58 = COALESCE(NULLIF(?, ''), hotkey_ss58),
                            coldkey_ss58 = COALESCE(NULLIF(?, ''), coldkey_ss58),
                            updated_at = ?
                        WHERE address = ? AND model_index = ?""",
                        (endpoint, quant, max_context_len, epoch,
                         1 if tee_enabled else 0, tee_platform,
                         registration_changed, gpu_name, gpu_name,
                         registration_changed, gpu_count, gpu_count, gpu_count,
                         registration_changed, vram_gb, vram_gb, vram_gb,
                         registration_changed, compute_capability, compute_capability,
                         registration_changed, _gpu_uuids_json,
                         _gpu_uuids_json, _gpu_uuids_json,
                         hotkey_ss58, coldkey_ss58, now, address, model_index),
                    )
            self._conn.commit()
        return model_switched

    def mark_unseen_inactive(self, current_epoch: int) -> int:
        """Mark entries not seen in *current_epoch* as inactive.

        Returns:
            Number of entries marked inactive.
        """
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE miner_entries SET is_active = 0, updated_at = ?
                   WHERE last_seen_epoch < ? AND is_active = 1""",
                (time.time(), current_epoch),
            )
            self._conn.commit()
            count = cursor.rowcount
        if count:
            bt.logging.info(f"Marked {count} entries inactive (not seen epoch {current_epoch})")
        return count

    def mark_entry_inactive(self, address: str, model_index: int) -> bool:
        """Mark a single miner-model entry inactive locally."""
        address = address.lower()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE miner_entries SET is_active = 0, updated_at = ?
                   WHERE address = ? AND model_index = ? AND is_active = 1""",
                (time.time(), address, model_index),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def mark_address_inactive(self, address: str) -> int:
        """Mark all miner-model entries for an address inactive locally."""
        address = address.lower()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE miner_entries SET is_active = 0, updated_at = ?
                   WHERE address = ? AND is_active = 1""",
                (time.time(), address),
            )
            self._conn.commit()
            return cursor.rowcount

    def reset_address_identity_state(self, address: str) -> int:
        """Reset mutable score/probation state after the address changes hotkey owner."""
        address = address.lower()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE miner_entries SET
                       ema_score = 0.0,
                       total_epochs = 0,
                       scored_epochs = 0,
                       probation_entered_epoch = NULL,
                       probation_consecutive_passes = 0,
                       updated_at = ?
                   WHERE address = ?""",
                (time.time(), address),
            )
            self._conn.commit()
            return cursor.rowcount

    def clear_probation_for_address(self, address: str) -> int:
        """Clear mutable probation state for an identity that no longer owns a UID."""
        address = address.lower()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE miner_entries SET
                       probation_entered_epoch = NULL,
                       probation_consecutive_passes = 0,
                       updated_at = ?
                   WHERE address = ?
                     AND probation_entered_epoch IS NOT NULL""",
                (time.time(), address),
            )
            self._conn.commit()
            return cursor.rowcount

    def get_active_entries(self) -> List[dict]:
        """Return all active miner-model entries as dicts."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM miner_entries WHERE is_active = 1"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_capacity_hardware_cache_entries(self) -> List[dict]:
        """Return exact-registration rows with reusable last-good hardware metadata."""
        with self._lock:
            cursor = self._conn.execute(
                """SELECT * FROM miner_entries
                   WHERE gpu_name != '' AND gpu_count > 0 AND vram_gb > 0"""
            )
            return [dict(row) for row in cursor.fetchall()]

    def _get_all_entries(self) -> List[dict]:
        """Return all miner-model entries (active and inactive) as dicts."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM miner_entries")
            return [dict(row) for row in cursor.fetchall()]

    # ── Scoring ──────────────────────────────────────────────────────

    def save_score(
        self,
        address: str,
        model_index: int,
        ema_score: float,
        total_epochs: int,
        scored_epochs: int,
    ) -> None:
        """Persist EMA score state for a miner-model entry."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    ema_score = ?, total_epochs = ?, scored_epochs = ?,
                    updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (ema_score, total_epochs, scored_epochs, time.time(),
                 address, model_index),
            )
            self._conn.commit()

    def halve_ema(self, address: str, model_index: int) -> None:
        """Halve EMA score on proof failure (geometric decay)."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    ema_score = ema_score * 0.5, updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (time.time(), address, model_index),
            )
            self._conn.commit()

    def load_all_scores(self) -> Dict[Tuple[str, int], dict]:
        """Load all entries with scoring data.

        Returns:
            Dict mapping ``(address, model_index)`` to a dict with keys
            ``model_id``, ``ema_score``, ``total_epochs``, ``scored_epochs``.
        """
        with self._lock:
            cursor = self._conn.execute(
                """SELECT address, model_index, model_id,
                          ema_score, total_epochs, scored_epochs
                   FROM miner_entries
                   WHERE is_active = 1"""
            )
            result: Dict[Tuple[str, int], dict] = {}
            for row in cursor.fetchall():
                key = (row["address"], row["model_index"])
                result[key] = {
                    "model_id": row["model_id"],
                    "ema_score": row["ema_score"],
                    "total_epochs": row["total_epochs"],
                    "scored_epochs": row["scored_epochs"],
                }
            return result

    # ── Probation ────────────────────────────────────────────────────

    def enter_probation(
        self,
        address: str,
        model_index: int,
        epoch: int,
        *,
        uid: int = -1,
        hotkey_ss58: str = "",
    ) -> None:
        """Put an entry on probation (or reset consecutive passes if already on).

        ``uid`` and ``hotkey_ss58`` are optional and are used solely for
        operator-readable logging — the DB row itself is keyed on the
        EVM address + model_index.
        """
        address = address.lower()
        # Build a stable, operator-recognizable identifier for logs.
        if uid >= 0 and hotkey_ss58:
            who = f"UID {uid} {hotkey_ss58}"
        elif uid >= 0:
            who = f"UID {uid}"
        else:
            who = address[:10]
        with self._lock:
            row = self._conn.execute(
                """SELECT probation_entered_epoch FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, model_index),
            ).fetchone()
            if row is None:
                # Internal "shouldn't happen" trace — debug only.
                bt.logging.debug(f"enter_probation: no entry for {who} idx={model_index}")
                return

            if row["probation_entered_epoch"] is not None:
                # Already on probation — reset passes.  Validator-side
                # operation against a misbehaving miner; INFO not WARN
                # (the validator is doing its job).
                self._conn.execute(
                    """UPDATE miner_entries SET
                        probation_consecutive_passes = 0, updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (time.time(), address, model_index),
                )
                bt.logging.info(f"Probation RESET for {who} idx={model_index} (new failure during probation)")
            else:
                self._conn.execute(
                    """UPDATE miner_entries SET
                        probation_entered_epoch = ?,
                        probation_consecutive_passes = 0,
                        updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (epoch, time.time(), address, model_index),
                )
                bt.logging.info(f"Probation ENTERED for {who} idx={model_index} at epoch {epoch}")
            self._conn.commit()

    def record_pass(self, address: str, model_index: int) -> bool:
        """Record a clean epoch during probation.

        Returns True if probation is lifted (enough consecutive passes).
        """
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT probation_entered_epoch,
                          probation_consecutive_passes,
                          probation_required_passes
                   FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, model_index),
            ).fetchone()
            if row is None or row["probation_entered_epoch"] is None:
                return False

            new_passes = row["probation_consecutive_passes"] + 1
            if new_passes >= row['probation_required_passes']:
                # Lift probation
                self._conn.execute(
                    """UPDATE miner_entries SET
                        probation_entered_epoch = NULL,
                        probation_consecutive_passes = 0,
                        updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (time.time(), address, model_index),
                )
                self._conn.commit()
                bt.logging.info(f"Probation LIFTED for {address[:10]} idx={model_index} after {new_passes} passes")
                return True
            else:
                self._conn.execute(
                    """UPDATE miner_entries SET
                        probation_consecutive_passes = ?,
                        updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (new_passes, time.time(), address, model_index),
                )
                self._conn.commit()
                bt.logging.info(f"Probation pass {new_passes}/{row['probation_required_passes']} for {address[:10]} idx={model_index}")
                return False

    def record_failure(self, address: str, model_index: int) -> None:
        """Record a proof failure during probation — resets consecutive passes."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    probation_consecutive_passes = 0, updated_at = ?
                WHERE address = ? AND model_index = ?
                  AND probation_entered_epoch IS NOT NULL""",
                (time.time(), address, model_index),
            )
            self._conn.commit()

    def clear_probation(self, address: str, model_index: int) -> None:
        """Explicitly clear probation for an entry."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    probation_entered_epoch = NULL,
                    probation_consecutive_passes = 0,
                    updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (time.time(), address, model_index),
            )
            self._conn.commit()

    def is_on_probation(self, address: str, model_index: int) -> bool:
        """Check if a miner-model entry is on probation."""
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT probation_entered_epoch FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, model_index),
            ).fetchone()
        return row is not None and row["probation_entered_epoch"] is not None

    def should_escalate(
        self, address: str, model_index: int, current_epoch: int,
    ) -> bool:
        """Check if probation has lasted long enough for reportOffline escalation."""
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT probation_entered_epoch, probation_escalation_epochs
                   FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, model_index),
            ).fetchone()
        if row is None or row["probation_entered_epoch"] is None:
            return False
        return (
            current_epoch - row["probation_entered_epoch"]
        ) >= row["probation_escalation_epochs"]

    def get_probation_addresses(self) -> Dict[str, List[int]]:
        """Get all probation entries grouped by address (for shared state)."""
        with self._lock:
            cursor = self._conn.execute(
                """SELECT address, model_index FROM miner_entries
                   WHERE probation_entered_epoch IS NOT NULL"""
            )
            result: Dict[str, List[int]] = {}
            for row in cursor.fetchall():
                result.setdefault(row["address"], []).append(row["model_index"])
            return result

    def migrate_probation(
        self, address: str, old_index: int, new_index: int,
    ) -> bool:
        """Migrate probation state from old model_index to new model_index.

        Used when a miner re-registers and gets a new contract array index.

        Returns True if migration occurred.
        """
        address = address.lower()
        with self._lock:
            # Read old entry probation
            old_row = self._conn.execute(
                """SELECT probation_entered_epoch,
                          probation_consecutive_passes,
                          probation_required_passes,
                          probation_escalation_epochs
                   FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, old_index),
            ).fetchone()
            if old_row is None or old_row["probation_entered_epoch"] is None:
                return False

            # Check new entry exists
            new_row = self._conn.execute(
                "SELECT 1 FROM miner_entries WHERE address = ? AND model_index = ?",
                (address, new_index),
            ).fetchone()
            if new_row is None:
                return False

            # Copy probation to new entry
            now = time.time()
            self._conn.execute(
                """UPDATE miner_entries SET
                    probation_entered_epoch = ?,
                    probation_consecutive_passes = ?,
                    probation_required_passes = ?,
                    probation_escalation_epochs = ?,
                    updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (old_row["probation_entered_epoch"],
                 old_row["probation_consecutive_passes"],
                 old_row['probation_required_passes'],
                 old_row["probation_escalation_epochs"],
                 now, address, new_index),
            )
            # Clear old entry probation
            self._conn.execute(
                """UPDATE miner_entries SET
                    probation_entered_epoch = NULL,
                    probation_consecutive_passes = 0,
                    updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (now, address, old_index),
            )
            self._conn.commit()
            bt.logging.info(f"Probation migrated for {address[:10]}: idx {old_index} -> {new_index}")
            return True

    # ── UID cache ────────────────────────────────────────────────────

    def get_uid(self, address: str) -> Optional[int]:
        """Look up cached UID for an EVM address (from miner_entries)."""
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT bittensor_uid FROM miner_entries WHERE address = ? AND bittensor_uid IS NOT NULL LIMIT 1",
                (address,),
            ).fetchone()
        return row["bittensor_uid"] if row is not None else None

    def get_cached_identity(self, address: str) -> dict:
        """Return the most useful cached UID/SS58 identity for an EVM address."""
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT bittensor_uid, hotkey_ss58, coldkey_ss58
                   FROM miner_entries
                   WHERE address = ?
                   ORDER BY
                     CASE WHEN hotkey_ss58 IS NOT NULL AND hotkey_ss58 != '' THEN 0 ELSE 1 END,
                     is_active DESC,
                     last_seen_epoch DESC,
                     updated_at DESC
                   LIMIT 1""",
                (address,),
            ).fetchone()
        return dict(row) if row is not None else {}

    def set_uid(self, address: str, uid: int) -> None:
        """Set UID for all miner_entries rows with this address (denormalized)."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                "UPDATE miner_entries SET bittensor_uid = ?, updated_at = ? WHERE address = ?",
                (uid, time.time(), address),
            )
            self._conn.commit()

    def get_all_uids(self) -> Dict[str, int]:
        """Return all cached address -> UID mappings (one per unique address)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT address, bittensor_uid FROM miner_entries WHERE bittensor_uid IS NOT NULL"
            )
            return {row["address"]: row["bittensor_uid"] for row in cursor.fetchall()}

    def reconcile_uid_owner(
        self,
        uid: int,
        hotkey_ss58: str,
        evm_address: str,
        epoch: int,
    ) -> dict:
        """Persist the current owner of a reusable Bittensor UID slot."""
        uid_i = int(uid)
        hotkey = str(hotkey_ss58 or "")
        evm = str(evm_address or "").lower()
        epoch_i = max(0, int(epoch or 0))
        now = time.time()
        with self._lock:
            moved_rows = []
            if evm:
                moved_rows = self._conn.execute(
                    """SELECT uid FROM uid_ownership
                       WHERE LOWER(evm_address) = ? AND uid != ?""",
                    (evm, uid_i),
                ).fetchall()
            moved_from_uids = sorted(int(moved["uid"]) for moved in moved_rows)
            if moved_from_uids:
                self._conn.execute(
                    """UPDATE uid_ownership SET
                           hotkey_ss58 = '', evm_address = '',
                           generation = generation + 1,
                           identity_start_epoch = ?, updated_at = ?
                       WHERE LOWER(evm_address) = ? AND uid != ?""",
                    (epoch_i, now, evm, uid_i),
                )
            row = self._conn.execute(
                "SELECT * FROM uid_ownership WHERE uid = ?",
                (uid_i,),
            ).fetchone()
            previous = dict(row) if row is not None else None
            inferred_previous_hotkey = ""
            inferred_previous_evm = ""
            if row is None:
                cached_rows = self._conn.execute(
                    """SELECT address, hotkey_ss58
                       FROM miner_entries
                       WHERE bittensor_uid = ?
                       ORDER BY is_active DESC, last_seen_epoch DESC, updated_at DESC""",
                    (uid_i,),
                ).fetchall()
                previous_row = next(
                    (
                        cached for cached in cached_rows
                        if str(cached["address"] or "").lower() != evm
                        or (
                            str(cached["hotkey_ss58"] or "")
                            and hotkey
                            and str(cached["hotkey_ss58"] or "") != hotkey
                        )
                    ),
                    None,
                )
                if previous_row is not None:
                    inferred_previous_hotkey = str(previous_row["hotkey_ss58"] or "")
                    inferred_previous_evm = str(previous_row["address"] or "").lower()
                changed = previous_row is not None or bool(moved_from_uids)
                start_epoch = epoch_i if changed else 0
                self._conn.execute(
                    """INSERT INTO uid_ownership (
                           uid, hotkey_ss58, evm_address, generation,
                           identity_start_epoch, updated_at
                       ) VALUES (?, ?, ?, 1, ?, ?)""",
                    (uid_i, hotkey, evm, start_epoch, now),
                )
                generation = 1
            else:
                changed = (
                    str(row["hotkey_ss58"] or "") != hotkey
                    or str(row["evm_address"] or "").lower() != evm
                    or bool(moved_from_uids)
                )
                generation = int(row["generation"] or 1) + (1 if changed else 0)
                start_epoch = epoch_i if changed else int(row["identity_start_epoch"] or 0)
                self._conn.execute(
                    """UPDATE uid_ownership SET
                           hotkey_ss58 = ?, evm_address = ?, generation = ?,
                           identity_start_epoch = ?, updated_at = ?
                       WHERE uid = ?""",
                    (hotkey, evm, generation, start_epoch, now, uid_i),
                )
            self._conn.commit()
        return {
            "changed": changed,
            "uid": uid_i,
            "hotkey_ss58": hotkey,
            "evm_address": evm,
            "generation": generation,
            "identity_start_epoch": start_epoch,
            "previous": previous,
            "inferred_previous_hotkey": inferred_previous_hotkey,
            "inferred_previous_evm": inferred_previous_evm,
            "moved_from_uids": moved_from_uids,
            "address_moved": bool(moved_from_uids),
        }

    def get_uid_owner(self, uid: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM uid_ownership WHERE uid = ?",
                (int(uid),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_uid_owners(self) -> Dict[int, dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM uid_ownership").fetchall()
        return {int(row["uid"]): dict(row) for row in rows}

    def get_addresses_for_uid(self, uid: int) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT address FROM miner_entries WHERE bittensor_uid = ?",
                (int(uid),),
            ).fetchall()
        return [str(row["address"]).lower() for row in rows]

    def _capacity_uid_identity_filter_locked(self, uid: int) -> Tuple[str, tuple]:
        """Return a SQL predicate selecting only the current UID identity."""
        owner = self._conn.execute(
            "SELECT * FROM uid_ownership WHERE uid = ?",
            (int(uid),),
        ).fetchone()
        if owner is not None:
            address = str(owner["evm_address"] or "").lower()
            hotkey = str(owner["hotkey_ss58"] or "")
            start_epoch = max(0, int(owner["identity_start_epoch"] or 0))
            if hotkey:
                return (
                    """LOWER(s.miner_address) = ?
                       AND (
                         s.miner_uid = ?
                         OR (
                           s.miner_uid IS NULL
                           AND (? = 0 OR w.epoch_number > ?)
                         )
                       )
                       AND (
                         s.miner_hotkey_ss58 = ?
                         OR (
                           COALESCE(s.miner_hotkey_ss58, '') = ''
                           AND (? = 0 OR w.epoch_number > ?)
                         )
                       )""",
                    (
                        address,
                        int(uid),
                        start_epoch,
                        start_epoch,
                        hotkey,
                        start_epoch,
                        start_epoch,
                    ),
                )
            return (
                """LOWER(s.miner_address) = ?
                   AND (
                     s.miner_uid = ?
                     OR (
                       s.miner_uid IS NULL
                       AND (? = 0 OR w.epoch_number > ?)
                     )
                   )""",
                (address, int(uid), start_epoch, start_epoch),
            )

        rows = self._conn.execute(
            """SELECT DISTINCT LOWER(address) AS address
               FROM miner_entries
               WHERE bittensor_uid = ? AND is_active = 1""",
            (int(uid),),
        ).fetchall()
        addresses = [str(row["address"] or "").lower() for row in rows]
        if not addresses:
            return "0 = 1", ()
        placeholders = ",".join("?" for _ in addresses)
        return (
            f"(s.miner_uid = ? OR s.miner_uid IS NULL) "
            f"AND LOWER(s.miner_address) IN ({placeholders})",
            (int(uid), *addresses),
        )

    def _capacity_address_identity_filter_locked(
        self,
        address: str,
    ) -> Tuple[str, tuple]:
        """Return the hotkey-generation predicate for a current EVM owner."""
        owner = self._conn.execute(
            """SELECT uid, hotkey_ss58, identity_start_epoch
               FROM uid_ownership
               WHERE LOWER(evm_address) = ?
               ORDER BY updated_at DESC
               LIMIT 1""",
            (str(address).lower(),),
        ).fetchone()
        if owner is None:
            return "1 = 1", ()
        uid = int(owner["uid"])
        hotkey = str(owner["hotkey_ss58"] or "")
        start_epoch = max(0, int(owner["identity_start_epoch"] or 0))
        uid_clause = """(
             s.miner_uid = ?
             OR (
               s.miner_uid IS NULL
               AND (? = 0 OR w.epoch_number > ?)
             )
           )"""
        if not hotkey:
            return uid_clause, (uid, start_epoch, start_epoch)
        return (
            """(
                 (
                   s.miner_uid = ?
                   OR (
                     s.miner_uid IS NULL
                     AND (? = 0 OR w.epoch_number > ?)
                   )
                 )
                 AND (
                   s.miner_hotkey_ss58 = ?
                   OR (
                     COALESCE(s.miner_hotkey_ss58, '') = ''
                     AND (? = 0 OR w.epoch_number > ?)
                   )
                 )
               )""",
            (
                uid,
                start_epoch,
                start_epoch,
                hotkey,
                start_epoch,
                start_epoch,
            ),
        )

    def mark_capacity_audit_address_identity_stale(self, address: str) -> int:
        """Release unfinished audit obligations belonging to a retired identity."""
        address = address.lower()
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET verdict = 'stale_window',
                       timing_status = 'timing_excused',
                       failure_reason = 'stale_uid_identity',
                       drain_until_ts = 0,
                       updated_at = ?
                   WHERE miner_address = ?
                     AND verdict IN ('pending', 'pass0_seen', 'timing_pass')
                     AND proof_status NOT IN (
                       'combined_proof_verified', 'proof_verified',
                       'invalid_payload', 'verify_error', 'missing_payload'
                     )""",
                (now, address),
            )
            self._conn.commit()
            return cursor.rowcount

    # ── Epoch log ────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        start_block: int,
        miner_count: int,
        receipt_count: int,
        weight_set: bool,
    ) -> None:
        """Record an epoch close in the audit log."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO epoch_log
                    (epoch_number, start_block, miner_count, receipt_count,
                     weight_set, closed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (epoch, start_block, miner_count, receipt_count,
                 1 if weight_set else 0, time.time()),
            )
            self._conn.commit()

    # ── Validator metadata (key-value) ───────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        """Read a validator metadata value."""
        with self._lock:
            return self._raw_get_meta(key)

    def set_meta(self, key: str, value: str) -> None:
        """Write a validator metadata value."""
        with self._lock:
            self._raw_set_meta(key, value)

    def _raw_get_meta(self, key: str) -> Optional[str]:
        """Read meta without acquiring lock (for use during init)."""
        row = self._conn.execute(
            "SELECT value FROM validator_meta WHERE key = ?", (key,),
        ).fetchone()
        return row["value"] if row is not None else None

    def _raw_set_meta(self, key: str, value: str) -> None:
        """Write meta (caller must hold lock or be in init)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO validator_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        """Add a column to an existing SQLite table if it is missing."""
        existing = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in existing:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self._conn.commit()

    # ── Hot-capacity audit state ──────────────────────────────────────

    def create_capacity_audit_window(
        self,
        *,
        audit_id: str,
        epoch_number: int,
        selection_block: int,
        audit_block: int,
        selection_block_hash: str,
        cohort_seed: str,
        slots: List[dict],
        proof_challenge_block: int = 0,
    ) -> None:
        """Insert one audit window and its selected endpoint slots."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO capacity_audit_windows (
                    audit_id, epoch_number, selection_block, audit_block, proof_challenge_block,
                    selection_block_hash, cohort_seed, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?)""",
                (
                    audit_id,
                    int(epoch_number),
                    int(selection_block),
                    int(audit_block),
                    int(proof_challenge_block),
                    selection_block_hash,
                    cohort_seed,
                    now,
                    now,
                ),
            )
            for slot in slots:
                self._conn.execute(
                    """INSERT OR IGNORE INTO capacity_audit_slots (
                        audit_id, miner_address, model_index, miner_uid,
                        miner_hotkey_ss58, endpoint,
                        model_id, quant, max_context_len, gpu_name, gpu_count,
                        vram_gb, group_key, slot_id, lease_id, claimed_gpu_class,
                        pass_count, deadline_s, transport_grace_s,
                        payload_deadline_s, drain_until_ts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        audit_id,
                        str(slot["miner_address"]).lower(),
                        int(slot["model_index"]),
                        slot.get("miner_uid"),
                        str(slot.get("miner_hotkey_ss58", "")),
                        str(slot.get("endpoint", "")),
                        str(slot.get("model_id", "")),
                        str(slot.get("quant", "")),
                        int(slot.get("max_context_len", 0) or 0),
                        str(slot.get("gpu_name", "")),
                        int(slot.get("gpu_count", 0) or 0),
                        int(slot.get("vram_gb", 0) or 0),
                        str(slot.get("group_key", "")),
                        str(slot["slot_id"]),
                        str(slot["lease_id"]),
                        str(slot.get("claimed_gpu_class", "")),
                        int(slot.get("pass_count", 0) or 0),
                        float(slot.get("deadline_s", 0.0) or 0.0),
                        float(slot.get("transport_grace_s", 0.0) or 0.0),
                        float(slot.get("payload_deadline_s", 0.0) or 0.0),
                        float(slot.get("drain_until_ts", 0.0) or 0.0),
                        now,
                        now,
                    ),
                )
            self._conn.commit()

    def set_capacity_audit_block_hash(
        self,
        audit_id: str,
        audit_block_hash: str,
        *,
        observed_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if observed_at is None else float(observed_at)
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_windows
                   SET audit_block_hash = ?,
                       audit_start_observed_at = ?,
                       status = 'started',
                       updated_at = ?
                   WHERE audit_id = ?""",
                (audit_block_hash, ts, ts, audit_id),
            )
            self._conn.commit()

    def set_capacity_audit_proof_challenge_hash(
        self,
        audit_id: str,
        proof_challenge_block_hash: str,
        *,
        observed_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if observed_at is None else float(observed_at)
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_windows
                   SET proof_challenge_block_hash = ?,
                       proof_challenge_observed_at = ?,
                       updated_at = ?
                   WHERE audit_id = ?""",
                (proof_challenge_block_hash, ts, ts, audit_id),
            )
            self._conn.commit()

    def resolve_capacity_audit_pending_start(
        self,
        audit_id: str,
        *,
        observed_at: Optional[float] = None,
    ) -> int:
        """Resolve final receipts that arrived before B_start was observed.

        Miners may observe the chain block and finish the proof before a polling
        validator processes that same block. Such receipts are judged once the
        validator records its B_start observation time.
        """
        ts = time.time() if observed_at is None else float(observed_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET timing_status = 'pass',
                       verdict = CASE
                           WHEN verdict = 'hard_proof_miss' THEN verdict
                           ELSE 'timing_pass'
                       END,
                       failure_reason = CASE
                           WHEN verdict = 'hard_proof_miss' THEN failure_reason
                           ELSE ''
                       END,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND timing_status = 'pending_start'
                     AND final_received_at IS NOT NULL
                     AND final_received_at <= ? + deadline_s + transport_grace_s""",
                (ts, audit_id, ts),
            )
            updated = cur.rowcount or 0
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET timing_status = 'miss',
                       verdict = CASE
                           WHEN verdict = 'hard_proof_miss' THEN verdict
                           ELSE 'timing_miss'
                       END,
                       failure_reason = CASE
                           WHEN verdict = 'hard_proof_miss' THEN failure_reason
                           ELSE COALESCE(NULLIF(failure_reason, ''), 'deadline_exceeded')
                       END,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND timing_status = 'pending_start'
                     AND final_received_at IS NOT NULL
                     AND final_received_at > ? + deadline_s + transport_grace_s""",
                (ts, audit_id, ts),
            )
            updated += cur.rowcount or 0
            self._conn.commit()
        return updated

    def get_capacity_audit_windows_for_start(self, audit_block: int) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM capacity_audit_windows
                   WHERE audit_block <= ? AND status = 'scheduled'
                   ORDER BY audit_block ASC, created_at ASC""",
                (int(audit_block),),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_capacity_audit_window_stale(
        self,
        audit_id: str,
        *,
        reason: str = "validator_start_missed",
        released_at: Optional[float] = None,
    ) -> int:
        ts = time.time() if released_at is None else float(released_at)
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_windows
                   SET status = 'stale',
                       updated_at = ?
                   WHERE audit_id = ?
                     AND status = 'scheduled'""",
                (ts, audit_id),
            )
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET verdict = 'stale_window',
                       timing_status = 'not_observed',
                       failure_reason = ?,
                       drain_until_ts = CASE
                           WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                       END,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND verdict IN (
                       'pending', 'pass0_seen', 'timing_pass',
                       'timing_miss', 'no_show'
                     )""",
                (reason, ts, ts, ts, audit_id),
            )
            self._conn.commit()
        return cur.rowcount or 0

    def get_capacity_audit_windows_for_proof_challenge(self, block_number: int) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM capacity_audit_windows
                   WHERE proof_challenge_block > 0
                     AND proof_challenge_block <= ?
                     AND proof_challenge_block_hash = ''
                   ORDER BY proof_challenge_block ASC, created_at ASC""",
                (int(block_number),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_capacity_audit_windows_for_finalization(self, finalized_block: int) -> List[dict]:
        """Return windows whose current-head block hashes can now be finalized."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT *
                   FROM capacity_audit_windows
                   WHERE chain_status != 'reorged'
                     AND (
                       (selection_finalized_at IS NULL AND selection_block <= ?)
                       OR (audit_finalized_at IS NULL AND audit_block_hash != '' AND audit_block <= ?)
                       OR (
                         proof_challenge_block > 0
                         AND proof_challenge_block_hash != ''
                         AND proof_challenge_finalized_at IS NULL
                         AND proof_challenge_block <= ?
                       )
                     )
                   ORDER BY selection_block ASC, created_at ASC""",
                (int(finalized_block), int(finalized_block), int(finalized_block)),
            ).fetchall()
        return [dict(r) for r in rows]

    def record_capacity_audit_finalization(
        self,
        audit_id: str,
        *,
        selection_confirmed: Optional[bool] = None,
        audit_confirmed: Optional[bool] = None,
        proof_confirmed: Optional[bool] = None,
        observed_at: Optional[float] = None,
    ) -> None:
        """Record finalized-chain confirmation for current-head audit hashes."""
        ts = time.time() if observed_at is None else float(observed_at)
        checks = [v for v in (selection_confirmed, audit_confirmed, proof_confirmed) if v is not None]
        with self._lock:
            if any(v is False for v in checks):
                self._conn.execute(
                    """UPDATE capacity_audit_windows
                       SET chain_status = 'reorged',
                           updated_at = ?
                       WHERE audit_id = ?""",
                    (ts, audit_id),
                )
                self._conn.execute(
                    """UPDATE capacity_audit_slots
                       SET verdict = CASE
                             WHEN verdict IN ('pending', 'pass0_seen', 'timing_pass', 'timing_miss', 'no_show', 'hard_proof_miss')
                             THEN 'chain_reorged'
                             ELSE verdict
                           END,
                           failure_reason = CASE
                             WHEN verdict IN ('pending', 'pass0_seen', 'timing_pass', 'timing_miss', 'no_show', 'hard_proof_miss')
                             THEN 'audit_block_reorged'
                             ELSE failure_reason
                           END,
                           drain_until_ts = CASE
                             WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                           END,
                           updated_at = ?
                       WHERE audit_id = ?""",
                    (ts, ts, ts, audit_id),
                )
                self._conn.commit()
                return

            updates = ["updated_at = ?"]
            params: list = [ts]
            if selection_confirmed is True:
                updates.append("selection_finalized_at = COALESCE(selection_finalized_at, ?)")
                params.append(ts)
            if audit_confirmed is True:
                updates.append("audit_finalized_at = COALESCE(audit_finalized_at, ?)")
                params.append(ts)
            if proof_confirmed is True:
                updates.append("proof_challenge_finalized_at = COALESCE(proof_challenge_finalized_at, ?)")
                params.append(ts)
            params.append(audit_id)
            self._conn.execute(
                f"""UPDATE capacity_audit_windows
                    SET {', '.join(updates)}
                    WHERE audit_id = ?""",
                params,
            )
            row = self._conn.execute(
                """SELECT selection_finalized_at, audit_finalized_at,
                          proof_challenge_block, proof_challenge_block_hash,
                          proof_challenge_finalized_at
                   FROM capacity_audit_windows
                   WHERE audit_id = ?""",
                (audit_id,),
            ).fetchone()
            if row is not None:
                proof_needed = (
                    int(row["proof_challenge_block"] or 0) > 0
                    and str(row["proof_challenge_block_hash"] or "") != ""
                )
                proof_ok = (not proof_needed) or row["proof_challenge_finalized_at"] is not None
                if (
                    row["selection_finalized_at"] is not None
                    and row["audit_finalized_at"] is not None
                    and proof_ok
                ):
                    self._conn.execute(
                        """UPDATE capacity_audit_windows
                           SET chain_status = 'confirmed',
                               updated_at = ?
                           WHERE audit_id = ? AND chain_status != 'reorged'""",
                        (ts, audit_id),
                    )
            self._conn.commit()

    def expire_capacity_audit_misses(
        self,
        now: Optional[float] = None,
        *,
        require_proof_payload: bool = False,
        return_slots: bool = False,
    ) -> int | List[dict]:
        """Mark audit misses after validator deadlines.

        The miner worker emits signed pass0/final timing roots plus a deferred
        fixed-workspace proof payload. Missing payloads become hard proof misses
        only when explicitly required by rollout config.
        """
        ts = time.time() if now is None else float(now)
        expired_slots: List[dict] = []
        with self._lock:
            if return_slots:
                rows = self._conn.execute(
                    """SELECT s.audit_id, s.miner_address, s.model_index,
                              s.endpoint, 'no_show' AS verdict,
                              COALESCE(s.failure_reason, 'missing_final_receipt') AS failure_reason
                       FROM capacity_audit_slots s
                       JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                       WHERE s.verdict IN ('pending', 'pass0_seen')
                         AND w.audit_start_observed_at IS NOT NULL
                         AND ? > w.audit_start_observed_at + s.deadline_s + s.transport_grace_s""",
                    (ts,),
                ).fetchall()
                expired_slots.extend(dict(r) for r in rows)
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET timing_status = 'missing_final',
                       verdict = 'no_show',
                       failure_reason = COALESCE(failure_reason, 'missing_final_receipt'),
                       updated_at = ?
                   WHERE verdict IN ('pending', 'pass0_seen')
                     AND audit_id IN (
                       SELECT audit_id
                       FROM capacity_audit_windows
                       WHERE audit_start_observed_at IS NOT NULL
                     )
                     AND ? > (
                       SELECT w.audit_start_observed_at
                       FROM capacity_audit_windows w
                       WHERE w.audit_id = capacity_audit_slots.audit_id
                     ) + deadline_s + transport_grace_s""",
                (ts, ts),
            )
            updated = cur.rowcount or 0
            if require_proof_payload:
                if return_slots:
                    rows = self._conn.execute(
                        """SELECT s.audit_id, s.miner_address, s.model_index,
                                  s.endpoint, 'hard_proof_miss' AS verdict,
                                  COALESCE(s.failure_reason, 'missing_proof_payload') AS failure_reason
                           FROM capacity_audit_slots s
                           JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                           WHERE s.verdict = 'timing_pass'
                             AND s.proof_status = 'pending'
                             AND s.final_received_at IS NOT NULL
                             AND ? > (
                               CASE
                                 WHEN COALESCE(
                                   w.proof_challenge_observed_at,
                                   w.proof_challenge_finalized_at,
                                   s.final_received_at
                                 ) > s.final_received_at
                                 THEN COALESCE(
                                   w.proof_challenge_observed_at,
                                   w.proof_challenge_finalized_at,
                                   s.final_received_at
                                 )
                                 ELSE s.final_received_at
                               END
                             ) + s.payload_deadline_s""",
                        (ts,),
                    ).fetchall()
                    expired_slots.extend(dict(r) for r in rows)
                cur = self._conn.execute(
                    """UPDATE capacity_audit_slots
                       SET proof_status = 'missing_payload',
                           verdict = 'hard_proof_miss',
                           failure_reason = COALESCE(failure_reason, 'missing_proof_payload'),
                           updated_at = ?
                       WHERE verdict = 'timing_pass'
                         AND proof_status = 'pending'
                         AND final_received_at IS NOT NULL
                         AND ? > (
                           SELECT
                             CASE
                               WHEN COALESCE(
                                 w.proof_challenge_observed_at,
                                 w.proof_challenge_finalized_at,
                                 capacity_audit_slots.final_received_at
                               ) > capacity_audit_slots.final_received_at
                               THEN COALESCE(
                                 w.proof_challenge_observed_at,
                                 w.proof_challenge_finalized_at,
                                 capacity_audit_slots.final_received_at
                               )
                               ELSE capacity_audit_slots.final_received_at
                             END
                           FROM capacity_audit_windows w
                           WHERE w.audit_id = capacity_audit_slots.audit_id
                         ) + payload_deadline_s""",
                    (ts, ts),
                )
                updated += cur.rowcount or 0
            self._conn.commit()
        return expired_slots if return_slots else updated

    def get_capacity_audit_slot(self, audit_id: str, address: str, model_index: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                """SELECT s.*, w.epoch_number, w.selection_block, w.audit_block,
                          w.proof_challenge_block,
                          w.selection_block_hash, w.audit_block_hash,
                          w.proof_challenge_block_hash, w.cohort_seed,
                          w.status, w.chain_status,
                          w.audit_start_observed_at, w.proof_challenge_observed_at,
                          w.selection_finalized_at, w.audit_finalized_at,
                          w.proof_challenge_finalized_at
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   WHERE s.audit_id = ? AND s.miner_address = ? AND s.model_index = ?""",
                (audit_id, address.lower(), int(model_index)),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_capacity_audit_pass0(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        pass0_root: str,
        artifact: dict,
        received_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if received_at is None else float(received_at)
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET pass0_received_at = COALESCE(pass0_received_at, ?),
                       pass0_root = ?,
                       pass0_artifact = ?,
                       verdict = CASE WHEN verdict = 'pending' THEN 'pass0_seen' ELSE verdict END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?""",
                (
                    ts,
                    pass0_root,
                    json.dumps(artifact, sort_keys=True),
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()

    def record_capacity_audit_final(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        final_root: str,
        transcript_root: str,
        artifact: dict,
        timing_status: str,
        verdict: str,
        failure_reason: str = "",
        received_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if received_at is None else float(received_at)
        pass0_root = str(artifact.get("pass0_root") or "")
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET final_received_at = COALESCE(final_received_at, ?),
                       pass0_root = CASE
                           WHEN pass0_root = '' THEN ? ELSE pass0_root
                       END,
                       final_root = ?,
                       transcript_root = ?,
                       final_artifact = ?,
                       timing_status = ?,
                       verdict = ?,
                       failure_reason = COALESCE(NULLIF(?, ''), failure_reason),
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?""",
                (
                    ts,
                    pass0_root,
                    final_root,
                    transcript_root,
                    json.dumps(artifact, sort_keys=True),
                    timing_status,
                    verdict,
                    failure_reason,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()

    def record_capacity_audit_proof_verdict(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        proof_status: str,
        verdict: str,
        failure_reason: str = "",
        proof_artifact_path: str = "",
        proof_verify_ms: Optional[float] = None,
        received_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if received_at is None else float(received_at)
        verify_ms = None if proof_verify_ms is None else max(0.0, float(proof_verify_ms))
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET proof_received_at = COALESCE(proof_received_at, ?),
                       proof_status = ?,
                       proof_verify_ms = COALESCE(?, proof_verify_ms),
                       verdict = ?,
                       failure_reason = COALESCE(NULLIF(?, ''), failure_reason),
                       proof_artifact_path = COALESCE(NULLIF(?, ''), proof_artifact_path),
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?""",
                (
                    ts,
                    proof_status,
                    verify_ms,
                    verdict,
                    failure_reason,
                    proof_artifact_path,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()

    def record_capacity_audit_proof_received(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        received_at: Optional[float] = None,
    ) -> None:
        ts = time.time() if received_at is None else float(received_at)
        with self._lock:
            self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET proof_received_at = COALESCE(proof_received_at, ?),
                       proof_status = CASE
                           WHEN proof_status = 'pending' THEN 'verify_pending'
                           ELSE proof_status
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?""",
                (
                    ts,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()

    def release_capacity_audit_drain(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        released_at: Optional[float] = None,
    ) -> int:
        """Release a selected endpoint from proxy drain after audit evidence is complete."""
        ts = time.time() if released_at is None else float(released_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET drain_until_ts = CASE
                           WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?""",
                (
                    ts,
                    ts,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()
        return cur.rowcount or 0

    def release_capacity_audit_completed_drains(
        self,
        audit_id: str,
        *,
        require_proof_payload: bool,
        released_at: Optional[float] = None,
    ) -> int:
        """Release drains for slots whose evidence is complete after start catch-up."""
        ts = time.time() if released_at is None else float(released_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET drain_until_ts = CASE
                           WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                       END,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND verdict = 'timing_pass'
                     AND (
                       ? = 0
                       OR proof_status IN ('proof_verified', 'combined_proof_verified')
                     )""",
                (
                    ts,
                    ts,
                    ts,
                    audit_id,
                    1 if require_proof_payload else 0,
                ),
            )
            self._conn.commit()
        return cur.rowcount or 0

    def recent_capacity_failures(
        self,
        address: str,
        model_index: int,
        *,
        since_epoch: int,
        verdicts: Tuple[str, ...] = ("timing_miss", "hard_proof_miss", "no_show"),
        require_chain_confirmed: bool = False,
    ) -> int:
        placeholders = ",".join("?" for _ in verdicts)
        confirmation_clause = ""
        if require_chain_confirmed:
            confirmation_clause = (
                " AND w.selection_finalized_at IS NOT NULL"
                " AND w.audit_finalized_at IS NOT NULL"
                " AND ("
                "   w.proof_challenge_block <= 0"
                "   OR ("
                "     w.proof_challenge_block_hash != ''"
                "     AND w.proof_challenge_finalized_at IS NOT NULL"
                "   )"
                " )"
                " AND w.chain_status != 'reorged'"
            )
        with self._lock:
            identity_clause, identity_params = self._capacity_address_identity_filter_locked(
                address
            )
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE s.miner_address = ?
                      AND s.model_index = ?
                      AND w.epoch_number >= ?
                      AND s.verdict IN ({placeholders})
                      AND {identity_clause}
                      {confirmation_clause}""",
                (
                    address.lower(), int(model_index), int(since_epoch),
                    *verdicts, *identity_params,
                ),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def recent_invalid_capacity_proof_failures(
        self,
        address: str,
        model_index: int,
        *,
        since_epoch: int,
        require_chain_confirmed: bool = False,
    ) -> int:
        confirmation_clause = ""
        if require_chain_confirmed:
            confirmation_clause = (
                " AND w.selection_finalized_at IS NOT NULL"
                " AND w.audit_finalized_at IS NOT NULL"
                " AND ("
                "   w.proof_challenge_block <= 0"
                "   OR ("
                "     w.proof_challenge_block_hash != ''"
                "     AND w.proof_challenge_finalized_at IS NOT NULL"
                "   )"
                " )"
                " AND w.chain_status != 'reorged'"
            )
        with self._lock:
            identity_clause, identity_params = self._capacity_address_identity_filter_locked(
                address
            )
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE s.miner_address = ?
                      AND s.model_index = ?
                      AND w.epoch_number >= ?
                      AND s.verdict = 'hard_proof_miss'
                      AND (
                        s.failure_reason = 'pass0_root_mismatch'
                        OR (
                          s.proof_status = 'invalid_payload'
                          AND s.failure_reason NOT IN (
                            'unsupported_proof_payload_format',
                            'unsupported_combined_format',
                            'unsupported_combined_workload'
                          )
                        )
                      )
                      AND {identity_clause}
                      {confirmation_clause}""",
                (
                    address.lower(), int(model_index), int(since_epoch),
                    *identity_params,
                ),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def recent_capacity_failures_for_uid(
        self,
        uid: int,
        *,
        since_epoch: int,
        verdicts: Tuple[str, ...] = ("timing_miss", "hard_proof_miss", "no_show"),
        require_chain_confirmed: bool = False,
    ) -> int:
        placeholders = ",".join("?" for _ in verdicts)
        confirmation_clause = ""
        if require_chain_confirmed:
            confirmation_clause = (
                " AND w.selection_finalized_at IS NOT NULL"
                " AND w.audit_finalized_at IS NOT NULL"
                " AND ("
                "   w.proof_challenge_block <= 0"
                "   OR ("
                "     w.proof_challenge_block_hash != ''"
                "     AND w.proof_challenge_finalized_at IS NOT NULL"
                "   )"
                " )"
                " AND w.chain_status != 'reorged'"
            )
        with self._lock:
            identity_clause, identity_params = self._capacity_uid_identity_filter_locked(uid)
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE w.epoch_number >= ?
                      AND s.verdict IN ({placeholders})
                      AND ({identity_clause})
                      {confirmation_clause}""",
                (int(since_epoch), *verdicts, *identity_params),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def recent_capacity_failure_counts_for_uid(
        self,
        uid: int,
        *,
        since_epoch: int,
        require_chain_confirmed: bool = False,
    ) -> Dict[Tuple[str, int], dict]:
        """Return finalized failure counts grouped by the registered endpoint slot."""
        confirmation_clause = ""
        if require_chain_confirmed:
            confirmation_clause = (
                " AND w.selection_finalized_at IS NOT NULL"
                " AND w.audit_finalized_at IS NOT NULL"
                " AND ("
                "   w.proof_challenge_block <= 0"
                "   OR ("
                "     w.proof_challenge_block_hash != ''"
                "     AND w.proof_challenge_finalized_at IS NOT NULL"
                "   )"
                " )"
                " AND w.chain_status != 'reorged'"
            )
        with self._lock:
            identity_clause, identity_params = self._capacity_uid_identity_filter_locked(uid)
            rows = self._conn.execute(
                f"""SELECT
                        s.miner_address,
                        s.model_index,
                        SUM(CASE
                            WHEN s.verdict = 'hard_proof_miss'
                             AND (
                               s.failure_reason = 'pass0_root_mismatch'
                               OR (
                                 s.proof_status = 'invalid_payload'
                                 AND s.failure_reason NOT IN (
                                   'unsupported_proof_payload_format',
                                   'unsupported_combined_format',
                                   'unsupported_combined_workload'
                                 )
                               )
                             )
                            THEN 1 ELSE 0 END
                        ) AS invalid_proof_failures,
                        SUM(CASE
                            WHEN s.verdict IN ('hard_proof_miss', 'no_show')
                            THEN 1 ELSE 0 END
                        ) AS hard_failures,
                        SUM(CASE
                            WHEN s.verdict = 'timing_miss'
                            THEN 1 ELSE 0 END
                        ) AS timing_failures
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE w.epoch_number >= ?
                      AND ({identity_clause})
                      AND s.verdict IN ('hard_proof_miss', 'no_show', 'timing_miss')
                      {confirmation_clause}
                    GROUP BY s.miner_address, s.model_index""",
                (int(since_epoch), *identity_params),
            ).fetchall()
        return {
            (str(row["miner_address"]).lower(), int(row["model_index"])): {
                "invalid_proof_failures": int(row["invalid_proof_failures"] or 0),
                "hard_failures": int(row["hard_failures"] or 0),
                "timing_failures": int(row["timing_failures"] or 0),
            }
            for row in rows
        }

    def active_entry_count_for_uid(self, uid: int) -> int:
        with self._lock:
            owner = self._conn.execute(
                "SELECT evm_address FROM uid_ownership WHERE uid = ?",
                (int(uid),),
            ).fetchone()
            if owner is not None:
                row = self._conn.execute(
                    """SELECT COUNT(*) AS n FROM miner_entries
                       WHERE is_active = 1 AND LOWER(address) = ?""",
                    (str(owner["evm_address"] or "").lower(),),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """SELECT COUNT(*) AS n FROM miner_entries
                       WHERE is_active = 1 AND bittensor_uid = ?""",
                    (int(uid),),
                ).fetchone()
        return int(row["n"] if row is not None else 0)

    def get_capacity_audit_slots_for_epoch(
        self,
        epoch_number: int,
        *,
        address: str = "",
        model_index: Optional[int] = None,
        verdicts: Tuple[str, ...] = ("timing_miss", "no_show"),
    ) -> List[dict]:
        placeholders = ",".join("?" for _ in verdicts)
        params: list = [int(epoch_number), *verdicts]
        address_clause = ""
        if address:
            address_clause = " AND s.miner_address = ?"
            params.append(address.lower())
        model_clause = ""
        if model_index is not None:
            model_clause = " AND s.model_index = ?"
            params.append(int(model_index))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT s.*, w.epoch_number, w.selection_block, w.audit_block,
                          w.proof_challenge_block, w.chain_status,
                          w.audit_start_observed_at, w.proof_challenge_observed_at,
                          w.selection_finalized_at, w.audit_finalized_at,
                          w.proof_challenge_finalized_at
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   WHERE w.epoch_number = ?
                     AND s.verdict IN ({placeholders})
                     {address_clause}
                     {model_clause}
                   ORDER BY w.audit_block ASC, s.updated_at ASC""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_capacity_audit_timing_excused(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        reason: str,
        released_at: Optional[float] = None,
    ) -> int:
        """Neutralize a timing/no-show audit miss using signed overlap evidence."""
        ts = time.time() if released_at is None else float(released_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET verdict = 'timing_excused',
                       timing_status = CASE
                           WHEN timing_status IN ('miss', 'missing_final') THEN 'excused'
                           ELSE timing_status
                       END,
                       failure_reason = ?,
                       drain_until_ts = CASE
                           WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                       END,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND miner_address = ?
                     AND model_index = ?
                     AND verdict IN ('timing_miss', 'no_show')""",
                (
                    reason,
                    ts,
                    ts,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                ),
            )
            self._conn.commit()
        return cur.rowcount or 0

    def get_capacity_drains(self, now: Optional[float] = None) -> List[AuditDrain]:
        ts = time.time() if now is None else float(now)
        with self._lock:
            rows = self._conn.execute(
                """SELECT audit_id, miner_address, model_index, endpoint, drain_until_ts
                   FROM capacity_audit_slots
                   WHERE drain_until_ts > ?
                     AND verdict IN ('pending', 'pass0_seen', 'timing_pass')""",
                (ts,),
            ).fetchall()
        return [
            AuditDrain(
                audit_id=row["audit_id"],
                address=row["miner_address"],
                model_index=int(row["model_index"]),
                endpoint=row["endpoint"],
                until_ts=float(row["drain_until_ts"]),
            )
            for row in rows
        ]

    def get_capacity_audit_selection_busy_slots(
        self,
        *,
        selection_block: int,
        cooldown_blocks: int = 1,
    ) -> List[tuple[str, int]]:
        """Return slots that cannot fairly accept a new B_select yet.

        Miners reserve a local endpoint through B_proof and may still be
        publishing/cleaning up on the immediately following head.  The
        validator must not create a new timed obligation for the same slot in
        that short block window, otherwise an honest miner can correctly skip
        the slot locally while the validator records a no-show.
        """
        cutoff = int(selection_block) - max(0, int(cooldown_blocks))
        with self._lock:
            rows = self._conn.execute(
                """SELECT DISTINCT s.miner_address, s.model_index
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   WHERE w.proof_challenge_block >= ?
                     AND s.verdict IN (
                       'pending', 'pass0_seen', 'timing_pass',
                       'timing_miss', 'no_show', 'hard_proof_miss'
                     )
                     AND w.chain_status != 'reorged'""",
                (cutoff,),
            ).fetchall()
        return [(str(row["miner_address"]).lower(), int(row["model_index"])) for row in rows]

    def compact_capacity_audit_storage(
        self,
        *,
        current_epoch: int,
        retain_failure_epochs: int,
        retain_history_seconds: float = CAPACITY_AUDIT_HISTORY_RETENTION_SECONDS,
        retain_artifacts: bool = False,
        now: Optional[float] = None,
    ) -> dict[str, int]:
        now_ts = time.time() if now is None else float(now)
        if retain_artifacts:
            return {
                "artifact_files_deleted": 0,
                "artifact_rows_cleared": 0,
                "history_rows_archived": 0,
                "old_history_rows_deleted": 0,
                "success_rows_deleted": 0,
                "old_failure_rows_deleted": 0,
                "empty_windows_deleted": 0,
            }
        final_proof_statuses = (
            "combined_proof_verified",
            "proof_verified",
            "invalid_payload",
            "verify_error",
            "missing_payload",
        )
        failure_verdicts = ("timing_miss", "hard_proof_miss", "no_show")
        history_verdicts = (
            "timing_pass",
            "timing_excused",
            "timing_miss",
            "hard_proof_miss",
            "no_show",
            "stale_window",
            "chain_reorged",
        )
        keep_from_epoch = max(
            0,
            int(current_epoch) - max(1, int(retain_failure_epochs)) + 1,
        )
        history_cutoff = now_ts - max(0.0, float(retain_history_seconds))
        with self._lock:
            path_rows = self._conn.execute(
                f"""SELECT DISTINCT s.proof_artifact_path AS path
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE w.epoch_number <= ?
                      AND COALESCE(s.proof_artifact_path, '') != ''
                      AND (
                        s.proof_status IN ({','.join('?' for _ in final_proof_statuses)})
                        OR s.verdict IN ({','.join('?' for _ in failure_verdicts)})
                        OR s.verdict IN ('timing_excused', 'stale_window', 'chain_reorged')
                      )""",
                (int(current_epoch), *final_proof_statuses, *failure_verdicts),
            ).fetchall()
        files_deleted = 0
        for row in path_rows:
            path = str(row["path"] or "")
            if not path:
                continue
            try:
                os.remove(path)
                files_deleted += 1
                parent = os.path.dirname(path)
                if parent:
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
            except FileNotFoundError:
                pass
            except OSError as exc:
                bt.logging.debug(f"Capacity audit artifact cleanup skipped {path}: {exc}")

        with self._lock:
            history_cur = self._conn.execute(
                f"""INSERT OR REPLACE INTO capacity_audit_history (
                        audit_id, miner_address, model_index, miner_uid,
                        miner_hotkey_ss58, endpoint,
                        model_id, quant, max_context_len, gpu_name, gpu_count,
                        vram_gb, group_key, slot_id, lease_id, claimed_gpu_class,
                        pass_count, epoch_number, selection_block, audit_block,
                        proof_challenge_block, verdict, timing_status, proof_status,
                        failure_reason, proof_verify_ms, pass0_received_at,
                        final_received_at, proof_received_at, slot_created_at,
                        slot_updated_at, archived_at
                    )
                    SELECT
                        s.audit_id, LOWER(s.miner_address), s.model_index, s.miner_uid,
                        s.miner_hotkey_ss58, s.endpoint, s.model_id, s.quant,
                        s.max_context_len, s.gpu_name,
                        s.gpu_count, s.vram_gb, s.group_key, s.slot_id, s.lease_id,
                        s.claimed_gpu_class, s.pass_count, w.epoch_number,
                        w.selection_block, w.audit_block, w.proof_challenge_block,
                        s.verdict, s.timing_status, s.proof_status, s.failure_reason,
                        s.proof_verify_ms, s.pass0_received_at, s.final_received_at,
                        s.proof_received_at, s.created_at, s.updated_at, ?
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE w.epoch_number <= ?
                      AND s.verdict IN ({','.join('?' for _ in history_verdicts)})""",
                (now_ts, int(current_epoch), *history_verdicts),
            )
            history_delete_cur = self._conn.execute(
                """DELETE FROM capacity_audit_history
                   WHERE archived_at < ?""",
                (history_cutoff,),
            )
            artifact_cur = self._conn.execute(
                f"""UPDATE capacity_audit_slots
                    SET pass0_artifact = NULL,
                        final_artifact = NULL,
                        proof_artifact_path = NULL,
                        updated_at = ?
                    WHERE audit_id IN (
                        SELECT audit_id
                        FROM capacity_audit_windows
                        WHERE epoch_number <= ?
                    )
                      AND (
                        pass0_artifact IS NOT NULL
                        OR final_artifact IS NOT NULL
                        OR proof_artifact_path IS NOT NULL
                      )
                      AND (
                        final_received_at IS NOT NULL
                        OR proof_status IN ({','.join('?' for _ in final_proof_statuses)})
                        OR verdict IN ({','.join('?' for _ in failure_verdicts)})
                        OR verdict IN ('timing_excused', 'stale_window', 'chain_reorged')
                      )""",
                (now_ts, int(current_epoch), *final_proof_statuses, *failure_verdicts),
            )
            success_cur = self._conn.execute(
                """DELETE FROM capacity_audit_slots
                   WHERE audit_id IN (
                       SELECT audit_id
                       FROM capacity_audit_windows
                       WHERE epoch_number <= ?
                   )
                     AND verdict IN ('timing_pass', 'timing_excused')
                     AND proof_status IN ('combined_proof_verified', 'proof_verified')""",
                (int(current_epoch),),
            )
            failure_cur = self._conn.execute(
                f"""DELETE FROM capacity_audit_slots
                    WHERE audit_id IN (
                        SELECT audit_id
                        FROM capacity_audit_windows
                        WHERE epoch_number < ?
                    )
                      AND verdict IN ({','.join('?' for _ in failure_verdicts)})""",
                (keep_from_epoch, *failure_verdicts),
            )
            window_cur = self._conn.execute(
                """DELETE FROM capacity_audit_windows
                   WHERE NOT EXISTS (
                       SELECT 1 FROM capacity_audit_slots s
                       WHERE s.audit_id = capacity_audit_windows.audit_id
                   )"""
            )
            self._conn.commit()
        return {
            "artifact_files_deleted": files_deleted,
            "artifact_rows_cleared": artifact_cur.rowcount or 0,
            "history_rows_archived": history_cur.rowcount or 0,
            "old_history_rows_deleted": history_delete_cur.rowcount or 0,
            "success_rows_deleted": success_cur.rowcount or 0,
            "old_failure_rows_deleted": failure_cur.rowcount or 0,
            "empty_windows_deleted": window_cur.rowcount or 0,
        }

    # ── Miner debug snapshots ───────────────────────────────────────

    def build_miner_debug_snapshots(
        self,
        *,
        current_epoch: int,
        capacity_audit_cfg: object,
        stale_addresses: Optional[set[str]] = None,
        blacklisted_addresses: Optional[set[str]] = None,
        model_gate_reasons: Optional[Dict[Tuple[str, int], str]] = None,
        capacity_audit_gate_enforced: Optional[bool] = None,
        capacity_audit_gate_suppression_reason: str = "",
        uid_network_state: Optional[Dict[int, dict]] = None,
        window_hours: Tuple[int, ...] = (24, 48),
        epoch_seconds: int = 72 * 60,
    ) -> Dict[str, Any]:
        """Build cached public miner diagnostics for the proxy.

        This intentionally returns bounded aggregates and redacted last-error
        summaries only. Public callers must not trigger live DB queries.
        """
        from neurons.capacity_audit import capacity_audit_uid_escalation_threshold

        now = time.time()
        cur_epoch = max(0, int(current_epoch or 0))
        windows = tuple(sorted({max(1, min(48, int(h))) for h in window_hours}))
        epoch_s = max(1, int(epoch_seconds or 1))
        window_epochs = {
            h: max(1, int(math.ceil((h * 3600.0) / epoch_s)))
            for h in windows
        }
        min_epoch = max(0, cur_epoch - max(window_epochs.values()) + 1)
        stale = {str(a).lower() for a in (stale_addresses or set()) if str(a or "").strip()}
        blacklisted = {
            str(a).lower()
            for a in (blacklisted_addresses or set())
            if str(a or "").strip()
        }
        model_gates = {
            (str(key[0]).lower(), int(key[1])): str(reason or "")
            for key, reason in (model_gate_reasons or {}).items()
            if reason
        }
        network_by_uid = {
            int(uid): dict(values or {})
            for uid, values in (uid_network_state or {}).items()
        }

        with self._lock:
            owner_rows = [
                dict(r) for r in self._conn.execute("SELECT * FROM uid_ownership").fetchall()
            ]
            entry_rows = [
                dict(r)
                for r in self._conn.execute(
                    """SELECT *
                       FROM miner_entries
                       WHERE bittensor_uid IS NOT NULL
                         AND (
                           is_active = 1
                           OR last_seen_epoch >= ?
                           OR (
                             probation_entered_epoch IS NOT NULL
                             AND last_seen_epoch >= ?
                           )
                         )
                       ORDER BY bittensor_uid, address, model_index""",
                    (min_epoch, min_epoch),
                ).fetchall()
            ]
            canary_rows = [
                dict(r)
                for r in self._conn.execute(
                    """SELECT epoch_number, miner_uid, LOWER(miner_address) AS miner_address,
                              COALESCE(miner_hotkey_ss58, '') AS miner_hotkey_ss58,
                              model_index, endpoint, status, error_message,
                              proof_requested, proof_verified, proof_failure_reason,
                              tee_requested, tee_verified,
                              tokens_generated, tokens_per_sec, created_at
                       FROM canary_results
                       WHERE epoch_number >= ?
                         AND miner_uid IS NOT NULL
                       ORDER BY epoch_number ASC, created_at ASC""",
                    (min_epoch,),
                ).fetchall()
            ]
            receipt_rows = [
                dict(r)
                for r in self._conn.execute(
                    """SELECT MAX(e.bittensor_uid) AS miner_uid,
                              LOWER(r.miner_address) AS miner_address,
                              COALESCE(r.miner_hotkey_ss58, '') AS miner_hotkey_ss58,
                              r.model_index,
                              r.epoch_number,
                              COUNT(*) AS receipts,
                              SUM(CASE WHEN r.is_canary THEN 1 ELSE 0 END) AS canary_receipts,
                              SUM(CASE WHEN r.is_own THEN 1 ELSE 0 END) AS own_receipts,
                              SUM(CASE WHEN r.proof_requested THEN 1 ELSE 0 END) AS proof_requested,
                              SUM(CASE WHEN r.proof_verified THEN 1 ELSE 0 END) AS proof_verified,
                              SUM(r.tokens_per_sec) AS tok_s_sum,
                              MAX(r.created_at) AS last_created_at
                       FROM network_receipts r
                       LEFT JOIN miner_entries e
                         ON e.address = LOWER(r.miner_address)
                        AND e.model_index = r.model_index
                       WHERE r.epoch_number >= ?
                       GROUP BY LOWER(r.miner_address), r.model_index,
                                COALESCE(r.miner_hotkey_ss58, ''), r.epoch_number""",
                    (min_epoch,),
                ).fetchall()
            ]
            score_rows = [
                dict(r)
                for r in self._conn.execute(
                    """SELECT epoch_number, miner_uid, LOWER(miner_address) AS miner_address,
                              COALESCE(miner_hotkey_ss58, '') AS miner_hotkey_ss58,
                              model_index, epoch_score, ema_score, own_receipts,
                              all_receipts, expected_receipts, proof_tests,
                              proof_failures, tee_tests, tee_failures,
                              tee_verified, created_at
                       FROM epoch_scores
                       WHERE epoch_number >= ?
                         AND miner_uid IS NOT NULL
                       ORDER BY epoch_number ASC, created_at ASC""",
                    (min_epoch,),
                ).fetchall()
            ]
            audit_rows = [
                dict(r)
                for r in self._conn.execute(
                    """SELECT audit_id, miner_uid, LOWER(miner_address) AS miner_address,
                              miner_hotkey_ss58,
                              model_index, endpoint, model_id, gpu_name, pass_count,
                              epoch_number, verdict, timing_status, proof_status,
                              failure_reason, updated_at, source_priority
                       FROM (
                         SELECT h.audit_id,
                                COALESCE(h.miner_uid, e.bittensor_uid) AS miner_uid,
                                h.miner_hotkey_ss58,
                                h.miner_address, h.model_index,
                                h.endpoint, h.model_id, h.gpu_name, h.pass_count,
                                h.epoch_number, h.verdict, h.timing_status,
                                h.proof_status, h.failure_reason,
                                h.slot_updated_at AS updated_at, 0 AS source_priority
                         FROM capacity_audit_history h
                         LEFT JOIN miner_entries e
                           ON e.address = LOWER(h.miner_address)
                          AND e.model_index = h.model_index
                         WHERE h.epoch_number >= ?
                           AND COALESCE(h.miner_uid, e.bittensor_uid) IS NOT NULL
                         UNION ALL
                         SELECT s.audit_id,
                                COALESCE(s.miner_uid, e.bittensor_uid) AS miner_uid,
                                s.miner_hotkey_ss58,
                                s.miner_address, s.model_index,
                                s.endpoint, s.model_id, s.gpu_name, s.pass_count,
                                w.epoch_number, s.verdict, s.timing_status,
                                s.proof_status, s.failure_reason,
                                s.updated_at, 1 AS source_priority
                         FROM capacity_audit_slots s
                         JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                         LEFT JOIN miner_entries e
                           ON e.address = LOWER(s.miner_address)
                          AND e.model_index = s.model_index
                         WHERE w.epoch_number >= ?
                           AND COALESCE(s.miner_uid, e.bittensor_uid) IS NOT NULL
                           AND w.chain_status != 'reorged'
                       )
                       ORDER BY epoch_number ASC, source_priority DESC, updated_at DESC""",
                    (min_epoch, min_epoch),
                ).fetchall()
            ]

        owners_by_uid: Dict[int, dict] = {
            int(row["uid"]): row for row in owner_rows
        }
        for row in entry_rows:
            uid_value = row.get("bittensor_uid")
            if uid_value is None or int(uid_value) in owners_by_uid:
                continue
            if not int(row.get("is_active") or 0):
                continue
            owners_by_uid[int(uid_value)] = {
                "uid": int(uid_value),
                "hotkey_ss58": str(row.get("hotkey_ss58") or ""),
                "evm_address": str(row.get("address") or "").lower(),
                "generation": 0,
                "identity_start_epoch": 0,
            }

        def event_key(row: dict) -> tuple[int, str, int] | None:
            uid = row.get("miner_uid")
            if uid is None:
                return None
            uid_i = int(uid)
            address = str(row.get("miner_address") or "").lower()
            owner = owners_by_uid.get(uid_i)
            if owner is not None:
                if address != str(owner.get("evm_address") or "").lower():
                    return None
                owner_hotkey = str(owner.get("hotkey_ss58") or "")
                event_hotkey = str(row.get("miner_hotkey_ss58") or "")
                if event_hotkey and owner_hotkey and event_hotkey != owner_hotkey:
                    return None
                if (
                    not event_hotkey
                    and int(row.get("epoch_number") or 0)
                    <= int(owner.get("identity_start_epoch") or 0)
                    and int(owner.get("identity_start_epoch") or 0) > 0
                ):
                    return None
            return (
                uid_i,
                address,
                int(row.get("model_index") or 0),
            )

        filtered_entries: list[dict] = []
        for row in entry_rows:
            uid = row.get("bittensor_uid")
            if uid is None:
                continue
            owner = owners_by_uid.get(int(uid))
            if owner is not None and str(row.get("address") or "").lower() != str(
                owner.get("evm_address") or ""
            ).lower():
                continue
            filtered_entries.append(row)
        entry_rows = filtered_entries

        repeat_window = max(
            1,
            int(getattr(capacity_audit_cfg, "repeat_window_epochs", 20) or 20),
        )
        gate_since_epoch = max(0, cur_epoch - repeat_window + 1)
        gate_configured = bool(
            getattr(capacity_audit_cfg, "enabled", False)
            and str(getattr(capacity_audit_cfg, "mode", "") or "") == "score_gate"
        )
        gate_enabled = gate_configured and (
            True
            if capacity_audit_gate_enforced is None
            else bool(capacity_audit_gate_enforced)
        )
        gate_counts: Dict[tuple[int, str, int], dict[str, int]] = {}
        gate_failure_epochs_all: Dict[
            tuple[int, str, int], Dict[str, list[int]]
        ] = {}
        if gate_configured:
            confirmation_clause = (
                " AND w.selection_finalized_at IS NOT NULL"
                " AND w.audit_finalized_at IS NOT NULL"
                " AND ("
                "   w.proof_challenge_block <= 0"
                "   OR ("
                "     w.proof_challenge_block_hash != ''"
                "     AND w.proof_challenge_finalized_at IS NOT NULL"
                "   )"
                " )"
                " AND w.chain_status != 'reorged'"
            )
            with self._lock:
                gate_rows = [
                    dict(row) for row in self._conn.execute(
                    f"""SELECT
                            COALESCE(s.miner_uid, e.bittensor_uid) AS miner_uid,
                            LOWER(s.miner_address) AS miner_address,
                            s.miner_hotkey_ss58,
                            s.model_index,
                            w.epoch_number,
                            s.verdict,
                            s.proof_status,
                            s.failure_reason
                        FROM capacity_audit_slots s
                        JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                        LEFT JOIN miner_entries e
                          ON e.address = LOWER(s.miner_address)
                         AND e.model_index = s.model_index
                        WHERE w.epoch_number >= ?
                          AND COALESCE(s.miner_uid, e.bittensor_uid) IS NOT NULL
                          AND s.verdict IN ('hard_proof_miss', 'no_show', 'timing_miss')
                          {confirmation_clause}
                        ORDER BY w.epoch_number ASC""",
                    (gate_since_epoch,),
                ).fetchall()]
            for row in gate_rows:
                key = event_key(row)
                if key is None:
                    continue
                counts = gate_counts.setdefault(key, {
                    "invalid_proof_failures": 0,
                    "hard_failures": 0,
                    "timing_failures": 0,
                })
                epochs = gate_failure_epochs_all.setdefault(
                    key,
                    {"invalid_proof": [], "hard": [], "timing": []},
                )
                verdict = str(row.get("verdict") or "")
                epoch_i = int(row.get("epoch_number") or 0)
                if _debug_capacity_audit_invalid_failure(
                    verdict,
                    row.get("proof_status"),
                    row.get("failure_reason"),
                ):
                    counts["invalid_proof_failures"] += 1
                    epochs["invalid_proof"].append(epoch_i)
                if verdict in {"hard_proof_miss", "no_show"}:
                    counts["hard_failures"] += 1
                    epochs["hard"].append(epoch_i)
                if verdict == "timing_miss":
                    counts["timing_failures"] += 1
                    epochs["timing"].append(epoch_i)

        entries_by_uid: Dict[int, list[dict]] = {}
        active_count_by_uid: Dict[int, int] = {}
        for row in entry_rows:
            uid = row.get("bittensor_uid")
            if uid is None:
                continue
            uid_i = int(uid)
            entries_by_uid.setdefault(uid_i, []).append(row)
            if int(row.get("is_active") or 0):
                active_count_by_uid[uid_i] = active_count_by_uid.get(uid_i, 0) + 1

        def empty_canary() -> dict[str, Any]:
            return {
                "total": 0,
                "ok": 0,
                "errors": 0,
                "proof_requested": 0,
                "proof_verified": 0,
                "proof_failures": 0,
                "tee_requested": 0,
                "tee_verified": 0,
                "tee_failures": 0,
                "last_status": "",
                "last_error_kind": "",
                "last_error": "",
                "last_proof_failure": "",
                "last_tee_failure": "",
                "last_epoch": None,
                "error_kinds": {},
                "recent_errors": [],
                "recent_proof_failures": [],
                "recent_tee_failures": [],
            }

        def empty_audit() -> dict[str, Any]:
            return {
                "total": 0,
                "timing_pass": 0,
                "timing_excused": 0,
                "timing_miss": 0,
                "hard_proof_miss": 0,
                "no_show": 0,
                "pending": 0,
                "failure_reasons": {},
                "recent_failures": [],
            }

        def add_hint(hints: list[dict[str, str]], code: str, message: str) -> None:
            if not any(h.get("code") == code for h in hints):
                hints.append({"code": code, "message": message})

        def add_step(steps: list[str], message: str) -> None:
            if message and message not in steps:
                steps.append(message)

        def remaining_from_clear_epoch(clear_epoch: Optional[int]) -> tuple[Optional[int], Optional[float]]:
            if clear_epoch is None:
                return None, None
            epochs_remaining = max(0, int(clear_epoch) - cur_epoch)
            hours_remaining = round((epochs_remaining * epoch_s) / 3600.0, 2)
            return epochs_remaining, hours_remaining

        def category_gate_status(
            *,
            name: str,
            failure_count: int,
            threshold: int,
            failure_epochs: list[int],
        ) -> Optional[dict[str, Any]]:
            if threshold <= 0 or failure_count < threshold:
                return None
            epochs = sorted(int(e) for e in failure_epochs)
            next_clear_epoch = None
            if len(epochs) >= threshold:
                idx = min(len(epochs) - 1, max(0, int(failure_count) - int(threshold)))
                next_clear_epoch = epochs[idx] + repeat_window
            epochs_remaining, hours_remaining = remaining_from_clear_epoch(next_clear_epoch)
            return {
                "reason": name,
                "failures": int(failure_count),
                "threshold": int(threshold),
                "failure_epochs": epochs[-20:],
                "next_possible_clear_epoch": next_clear_epoch,
                "epochs_remaining_if_clean": epochs_remaining,
                "estimated_hours_remaining_if_clean": hours_remaining,
            }

        def build_entry_gate_status(
            key: tuple[int, str, int],
            gate_data: dict[str, int],
            gate_failure_epochs: dict[tuple[int, str, int], dict[str, list[int]]],
        ) -> dict[str, Any]:
            categories: list[dict[str, Any]] = []
            failure_epochs = gate_failure_epochs.get(key, {})
            invalid_status = category_gate_status(
                name="invalid_proof",
                failure_count=int(gate_data.get("invalid_proof_failures", 0) or 0),
                threshold=int(getattr(capacity_audit_cfg, "invalid_proof_misses_for_zero_score", 1) or 1),
                failure_epochs=failure_epochs.get("invalid_proof", []),
            )
            if invalid_status:
                categories.append(invalid_status)
            hard_status = category_gate_status(
                name="hard_proof_or_no_show",
                failure_count=int(gate_data.get("hard_failures", 0) or 0),
                threshold=int(getattr(capacity_audit_cfg, "hard_proof_misses_for_zero_score", 2) or 2),
                failure_epochs=failure_epochs.get("hard", []),
            )
            if hard_status:
                categories.append(hard_status)
            if bool(getattr(capacity_audit_cfg, "allow_timing_only_score_gate", True)):
                timing_status = category_gate_status(
                    name="timing",
                    failure_count=int(gate_data.get("timing_failures", 0) or 0),
                    threshold=int(getattr(capacity_audit_cfg, "timing_misses_for_zero_score", 2) or 2),
                    failure_epochs=failure_epochs.get("timing", []),
                )
                if timing_status:
                    categories.append(timing_status)

            clear_epochs = [
                int(c["next_possible_clear_epoch"])
                for c in categories
                if c.get("next_possible_clear_epoch") is not None
            ]
            next_clear_epoch = (
                max(clear_epochs)
                if categories and len(clear_epochs) == len(categories)
                else None
            )
            epochs_remaining, hours_remaining = remaining_from_clear_epoch(next_clear_epoch)
            return {
                "configured": gate_configured,
                "enabled": gate_enabled,
                "suppression_reason": (
                    str(capacity_audit_gate_suppression_reason or "")
                    if gate_configured and not gate_enabled else ""
                ),
                "active": bool(gate_enabled and categories),
                "lookback_epochs": repeat_window,
                "since_epoch": gate_since_epoch,
                "active_reasons": [str(c["reason"]) for c in categories],
                "next_possible_clear_epoch": next_clear_epoch,
                "epochs_remaining_if_clean": epochs_remaining if categories else 0,
                "estimated_hours_remaining_if_clean": hours_remaining if categories else 0.0,
                "categories": categories,
            }

        def build_window(window_h: int) -> dict[str, Any]:
            since_epoch = max(0, cur_epoch - window_epochs[window_h] + 1)
            canaries: Dict[tuple[int, str, int], dict[str, Any]] = {}
            receipts: Dict[tuple[int, str, int], dict[str, Any]] = {}
            scores: Dict[tuple[int, str, int], dict[str, Any]] = {}
            audits: Dict[tuple[int, str, int], dict[str, Any]] = {}
            gate_failure_epochs = gate_failure_epochs_all
            audit_seen: set[tuple[str, str, int]] = set()
            uids: set[int] = set(entries_by_uid) | set(network_by_uid)

            for row in canary_rows:
                if int(row.get("epoch_number") or 0) < since_epoch:
                    continue
                key = event_key(row)
                if key is None:
                    continue
                uids.add(key[0])
                item = canaries.setdefault(key, empty_canary())
                item["total"] += 1
                status = str(row.get("status") or "")
                err = row.get("error_message") or ""
                error_kind = _debug_error_kind(err)
                if status == "ok":
                    item["ok"] += 1
                else:
                    item["errors"] += 1
                    if not error_kind:
                        error_kind = "chat_error"
                    error_text = _debug_public_error_summary(error_kind)
                    error_kinds = item["error_kinds"]
                    error_kinds[error_kind] = int(error_kinds.get(error_kind, 0)) + 1
                    item["recent_errors"].append({
                        "epoch": int(row.get("epoch_number") or 0),
                        "status": status,
                        "error_kind": error_kind,
                        "error": error_text,
                    })
                    if len(item["recent_errors"]) > 10:
                        item["recent_errors"] = item["recent_errors"][-10:]
                    item["last_error_kind"] = error_kind
                    item["last_error"] = error_text
                if int(row.get("proof_requested") or 0):
                    item["proof_requested"] += 1
                    if int(row.get("proof_verified") or 0):
                        item["proof_verified"] += 1
                    elif status == "ok":
                        item["proof_failures"] += 1
                        proof_reason = _debug_public_reason(
                            row.get("proof_failure_reason"),
                            "proof_verification_failed",
                        )
                        item["last_proof_failure"] = proof_reason
                        item["recent_proof_failures"].append({
                            "epoch": int(row.get("epoch_number") or 0),
                            "reason": proof_reason,
                        })
                        item["recent_proof_failures"] = item["recent_proof_failures"][-10:]
                if int(row.get("tee_requested") or 0):
                    item["tee_requested"] += 1
                    if int(row.get("tee_verified") or 0):
                        item["tee_verified"] += 1
                    elif status == "ok":
                        item["tee_failures"] += 1
                        item["last_tee_failure"] = "tee_attestation_not_verified"
                        item["recent_tee_failures"].append({
                            "epoch": int(row.get("epoch_number") or 0),
                            "reason": "tee_attestation_not_verified",
                        })
                        item["recent_tee_failures"] = item["recent_tee_failures"][-10:]
                item["last_status"] = status
                item["last_epoch"] = int(row.get("epoch_number") or 0)

            for row in receipt_rows:
                epoch_i = int(row.get("epoch_number") or 0)
                if row.get("miner_uid") is None or epoch_i < since_epoch:
                    continue
                key = event_key(row)
                if key is None:
                    continue
                uids.add(key[0])
                item = receipts.setdefault(key, {
                    "receipts": 0,
                    "canary_receipts": 0,
                    "own_receipts": 0,
                    "proof_requested": 0,
                    "proof_verified": 0,
                    "avg_tok_s": 0.0,
                    "first_epoch": None,
                    "last_epoch": None,
                    "_tok_s_sum": 0.0,
                })
                receipt_count = int(row.get("receipts") or 0)
                item["receipts"] += receipt_count
                item["canary_receipts"] += int(row.get("canary_receipts") or 0)
                item["own_receipts"] += int(row.get("own_receipts") or 0)
                item["proof_requested"] += int(row.get("proof_requested") or 0)
                item["proof_verified"] += int(row.get("proof_verified") or 0)
                item["_tok_s_sum"] += float(row.get("tok_s_sum") or 0.0)
                item["avg_tok_s"] = round(
                    item["_tok_s_sum"] / max(1, item["receipts"]),
                    2,
                )
                item["first_epoch"] = (
                    epoch_i if item["first_epoch"] is None
                    else min(int(item["first_epoch"]), epoch_i)
                )
                item["last_epoch"] = (
                    epoch_i if item["last_epoch"] is None
                    else max(int(item["last_epoch"]), epoch_i)
                )

            for row in score_rows:
                if int(row.get("epoch_number") or 0) < since_epoch:
                    continue
                key = event_key(row)
                if key is None:
                    continue
                uids.add(key[0])
                scores[key] = {
                    "last_scored_epoch": int(row.get("epoch_number") or 0),
                    "epoch_score": (
                        None
                        if row.get("epoch_score") is None
                        else float(row.get("epoch_score"))
                    ),
                    "ema_score": float(row.get("ema_score") or 0.0),
                    "own_receipts": int(row.get("own_receipts") or 0),
                    "all_receipts": int(row.get("all_receipts") or 0),
                    "expected_receipts": int(row.get("expected_receipts") or 0),
                    "proof_tests": int(row.get("proof_tests") or 0),
                    "proof_failures": int(row.get("proof_failures") or 0),
                    "tee_tests": int(row.get("tee_tests") or 0),
                    "tee_failures": int(row.get("tee_failures") or 0),
                    "tee_verified": bool(row.get("tee_verified") or 0),
                }

            for row in audit_rows:
                if int(row.get("epoch_number") or 0) < since_epoch:
                    continue
                key = event_key(row)
                if key is None:
                    continue
                dedupe = (
                    str(row.get("audit_id") or ""),
                    key[1],
                    key[2],
                )
                if dedupe in audit_seen:
                    continue
                audit_seen.add(dedupe)
                uids.add(key[0])
                item = audits.setdefault(key, empty_audit())
                verdict = str(row.get("verdict") or "pending")
                item["total"] += 1
                if verdict in item:
                    item[verdict] += 1
                else:
                    item["pending"] += 1
                raw_reason = row.get("failure_reason") or ""
                reason = (
                    _debug_public_reason(raw_reason, "capacity_audit_failed")
                    if raw_reason else ""
                )
                if reason:
                    reasons = item["failure_reasons"]
                    reasons[reason] = int(reasons.get(reason, 0)) + 1
                if verdict in {"timing_miss", "hard_proof_miss", "no_show"}:
                    epoch_i = int(row.get("epoch_number") or 0)
                    item["recent_failures"].append({
                        "epoch": epoch_i,
                        "model_index": key[2],
                        "verdict": verdict,
                        "timing_status": str(row.get("timing_status") or ""),
                        "proof_status": str(row.get("proof_status") or ""),
                        "failure_reason": reason,
                        "gpu_name": str(row.get("gpu_name") or ""),
                        "pass_count": int(row.get("pass_count") or 0),
                    })
                    if len(item["recent_failures"]) > 20:
                        item["recent_failures"] = item["recent_failures"][-20:]

            result_uids: Dict[str, Any] = {}
            for uid in sorted(uids):
                uid_entries = [
                    e for e in entries_by_uid.get(uid, [])
                    if int(e.get("is_active") or 0)
                    or int(e.get("last_seen_epoch") or 0) >= since_epoch
                    or e.get("probation_entered_epoch") is not None
                ]
                entry_keys = {
                    (uid, str(e.get("address") or "").lower(), int(e.get("model_index") or 0))
                    for e in uid_entries
                }
                for key in set(canaries) | set(receipts) | set(scores) | set(audits) | set(gate_counts):
                    if key[0] == uid:
                        entry_keys.add(key)

                entries = []
                hints: list[dict[str, str]] = []
                active_entries = 0
                inactive_recent_entries = 0
                on_probation = 0
                latest_scored_epoch = None
                best_score = 0.0
                entry_gate_status_by_key: Dict[tuple[int, str, int], dict[str, Any]] = {}
                for key in sorted(entry_keys, key=lambda k: (k[1], k[2])):
                    _, addr, idx = key
                    db_entry = next(
                        (
                            e for e in uid_entries
                            if str(e.get("address") or "").lower() == addr
                            and int(e.get("model_index") or 0) == idx
                        ),
                        None,
                    )
                    score_data = scores.get(key, {})
                    canary_data = canaries.get(key, empty_canary())
                    audit_data = audits.get(key, empty_audit())
                    receipt_data = dict(receipts.get(key, {
                        "receipts": 0,
                        "canary_receipts": 0,
                        "own_receipts": 0,
                        "proof_requested": 0,
                        "proof_verified": 0,
                        "avg_tok_s": 0.0,
                        "first_epoch": None,
                        "last_epoch": None,
                    }))
                    receipt_data.pop("_tok_s_sum", None)
                    gate_data = gate_counts.get(key, {
                        "invalid_proof_failures": 0,
                        "hard_failures": 0,
                        "timing_failures": 0,
                    })
                    gate_status = build_entry_gate_status(
                        key,
                        gate_data,
                        gate_failure_epochs,
                    )
                    entry_gate_status_by_key[key] = gate_status
                    is_active = bool(db_entry and int(db_entry.get("is_active") or 0))
                    if is_active:
                        active_entries += 1
                    elif db_entry is not None:
                        inactive_recent_entries += 1
                    probation_epoch = (
                        db_entry.get("probation_entered_epoch")
                        if db_entry is not None else None
                    )
                    is_probation = probation_epoch is not None
                    if is_probation:
                        on_probation += 1
                    probation_consecutive = (
                        int(db_entry.get("probation_consecutive_passes") or 0)
                        if db_entry else 0
                    )
                    probation_required = (
                        int(db_entry.get("probation_required_passes") or 0)
                        if db_entry else 0
                    )
                    probation_remaining = (
                        max(0, probation_required - probation_consecutive)
                        if is_probation else 0
                    )
                    raw_ema = float(db_entry.get("ema_score") or 0.0) if db_entry else 0.0
                    display_score = 0.0 if not is_active else (raw_ema if raw_ema > 0 else 0.01)
                    is_blacklisted = addr in blacklisted
                    model_gate_reason = model_gates.get((addr, idx), "")
                    best_score = max(best_score, display_score)
                    if score_data.get("last_scored_epoch") is not None:
                        latest_scored_epoch = max(
                            latest_scored_epoch or 0,
                            int(score_data["last_scored_epoch"]),
                        )

                    entry_hint_codes: list[str] = []
                    if is_blacklisted:
                        entry_hint_codes.append("blacklisted")
                        add_hint(
                            hints,
                            "blacklisted",
                            "This miner address is currently blacklisted by subnet configuration.",
                        )
                    if model_gate_reason:
                        entry_hint_codes.append("model_gate_active")
                        add_hint(
                            hints,
                            "model_gate_active",
                            "This executor does not satisfy the current capacity model/GPU gate.",
                        )
                    if addr in stale:
                        entry_hint_codes.append("stale_uid_identity")
                        add_hint(
                            hints,
                            "stale_uid_identity",
                            "This address is stale for the current UID owner and should not be routed.",
                        )
                    if is_probation:
                        entry_hint_codes.append("on_probation")
                        add_hint(
                            hints,
                            "on_probation",
                            "One or more entries are on probation and need clean passes to recover.",
                        )
                    if is_active and int(db_entry.get("scored_epochs") or 0) == 0:
                        entry_hint_codes.append("new_entry_not_scored")
                        add_hint(
                            hints,
                            "new_entry_not_scored",
                            "A current active entry has not closed a scored epoch yet.",
                        )
                    if canary_data.get("last_error_kind"):
                        entry_hint_codes.append(str(canary_data["last_error_kind"]))
                        if canary_data["last_error_kind"] == "chat_unauthorized":
                            add_hint(
                                hints,
                                "chat_unauthorized",
                                "The endpoint is reachable but /chat rejects validator requests with 401.",
                            )
                        else:
                            add_hint(
                                hints,
                                str(canary_data["last_error_kind"]),
                                "Recent canary requests failed on the inference route.",
                            )
                    if int(canary_data.get("proof_failures") or 0):
                        entry_hint_codes.append("proof_failure")
                        add_hint(
                            hints,
                            "proof_failure",
                            "Recent synthetic proof verification failed for this executor.",
                        )
                    if int(canary_data.get("tee_failures") or 0):
                        entry_hint_codes.append("tee_failure")
                        add_hint(
                            hints,
                            "tee_failure",
                            "Recent TEE attestation verification failed for this executor.",
                        )
                    if gate_data["hard_failures"] or gate_data["timing_failures"] or gate_data["invalid_proof_failures"]:
                        entry_hint_codes.append("capacity_audit_failures")
                        add_hint(
                            hints,
                            "capacity_audit_failures",
                            "Recent capacity-audit failures are still inside the scoring lookback.",
                        )

                    entry_next_steps: list[str] = []
                    if is_blacklisted:
                        add_step(
                            entry_next_steps,
                            "Resolve the subnet blacklist reason before expecting a score.",
                        )
                    if model_gate_reason:
                        add_step(entry_next_steps, model_gate_reason)
                    if not is_active:
                        add_step(
                            entry_next_steps,
                            "This executor is not active in the cached validator state; start or re-register the intended endpoint.",
                        )
                    if addr in stale:
                        add_step(
                            entry_next_steps,
                            "Stop using this stale address for the current UID owner.",
                        )
                    if is_active and int(db_entry.get("scored_epochs") or 0) == 0:
                        add_step(
                            entry_next_steps,
                            "Keep the endpoint online until at least one scored epoch closes.",
                        )
                    if canary_data.get("last_error_kind"):
                        add_step(
                            entry_next_steps,
                            "Fix the inference route shown by canary.last_error before expecting score recovery.",
                        )
                    if int(canary_data.get("proof_failures") or 0):
                        add_step(
                            entry_next_steps,
                            "Check proof generation and keep the endpoint online for clean synthetic proofs.",
                        )
                    if int(canary_data.get("tee_failures") or 0):
                        add_step(
                            entry_next_steps,
                            "Check the TEE attestation path and registered enclave identity.",
                        )
                    if is_probation:
                        add_step(
                            entry_next_steps,
                            f"Keep this executor clean for {probation_remaining} more probation pass(es).",
                        )
                    if gate_status.get("active"):
                        clear_epoch = gate_status.get("next_possible_clear_epoch")
                        suffix = (
                            f" Earliest clear if clean: epoch {clear_epoch}."
                            if clear_epoch is not None else ""
                        )
                        add_step(
                            entry_next_steps,
                            "Fix capacity-audit execution/publishing for this executor and keep it clean."
                            + suffix,
                        )
                    elif (
                        gate_data["hard_failures"]
                        or gate_data["timing_failures"]
                        or gate_data["invalid_proof_failures"]
                    ):
                        add_step(
                            entry_next_steps,
                            "Recent audit failures are present but below the active gate threshold; keep this executor clean.",
                        )

                    entries.append({
                        "address": addr,
                        "model_index": idx,
                        "endpoint": db_entry.get("endpoint") if db_entry else "",
                        "model_id": db_entry.get("model_id") if db_entry else "",
                        "quant": db_entry.get("quant") if db_entry else "",
                        "gpu_name": db_entry.get("gpu_name") if db_entry else "",
                        "gpu_count": int(db_entry.get("gpu_count") or 0) if db_entry else 0,
                        "vram_gb": int(db_entry.get("vram_gb") or 0) if db_entry else 0,
                        "active": is_active,
                        "stale": addr in stale,
                        "blacklisted": is_blacklisted,
                        "model_gate": {
                            "active": bool(model_gate_reason),
                            "reason": model_gate_reason,
                        },
                        "first_seen_epoch": db_entry.get("first_seen_epoch") if db_entry else None,
                        "last_seen_epoch": db_entry.get("last_seen_epoch") if db_entry else None,
                        "score": display_score,
                        "ema_score": raw_ema,
                        "total_epochs": int(db_entry.get("total_epochs") or 0) if db_entry else 0,
                        "scored_epochs": int(db_entry.get("scored_epochs") or 0) if db_entry else 0,
                        "probation": {
                            "active": is_probation,
                            "entered_epoch": probation_epoch,
                            "consecutive_passes": probation_consecutive,
                            "required_passes": probation_required,
                            "passes_remaining_if_clean": probation_remaining,
                        },
                        "canary": canary_data,
                        "receipts": receipt_data,
                        "last_score": score_data,
                        "capacity_audit": {
                            **audit_data,
                            "gate_counts": gate_data,
                            "gate_status": gate_status,
                        },
                        "issue_codes": entry_hint_codes,
                        "next_steps": entry_next_steps,
                    })

                if inactive_recent_entries:
                    add_hint(
                        hints,
                        "recent_endpoint_churn",
                        "Recent inactive entries still exist for this UID; replacing endpoints does not erase the lookback immediately.",
                    )
                if active_entries == 0:
                    add_hint(hints, "no_active_endpoint", "No active endpoint is present for this UID.")

                convicted = []
                convicted_details = []
                for key, row in gate_counts.items():
                    if key[0] != uid:
                        continue
                    gate_status = entry_gate_status_by_key.get(key) or build_entry_gate_status(
                        key,
                        row,
                        gate_failure_epochs,
                    )
                    if gate_status.get("active"):
                        convicted.append(key)
                        convicted_details.append({
                            "address": key[1],
                            "model_index": key[2],
                            "active_reasons": gate_status.get("active_reasons", []),
                            "next_possible_clear_epoch": gate_status.get("next_possible_clear_epoch"),
                            "epochs_remaining_if_clean": gate_status.get("epochs_remaining_if_clean"),
                            "estimated_hours_remaining_if_clean": gate_status.get("estimated_hours_remaining_if_clean"),
                        })

                evidence_entry_count = len([k for k in gate_counts if k[0] == uid])
                entry_count = max(active_count_by_uid.get(uid, 0), evidence_entry_count)
                quorum = (
                    capacity_audit_uid_escalation_threshold(entry_count, capacity_audit_cfg)
                    if gate_enabled and entry_count > 0
                    else 0
                )
                uid_gate_active = bool(gate_enabled and entry_count > 1 and len(convicted) >= quorum)
                uid_next_clear_epoch = None
                if uid_gate_active and quorum > 0:
                    clear_epochs = sorted(
                        int(d["next_possible_clear_epoch"])
                        for d in convicted_details
                        if d.get("next_possible_clear_epoch") is not None
                    )
                    clears_needed = len(convicted) - quorum + 1
                    if clears_needed > 0 and len(clear_epochs) >= clears_needed:
                        uid_next_clear_epoch = clear_epochs[clears_needed - 1]
                uid_epochs_remaining, uid_hours_remaining = remaining_from_clear_epoch(
                    uid_next_clear_epoch
                )
                if uid_gate_active:
                    add_hint(
                        hints,
                        "uid_audit_gate_active",
                        "UID-level capacity-audit floor is active; adding new healthy endpoints does not clear old failures inside the lookback.",
                    )

                network_source = network_by_uid.get(uid, {})

                def network_float(name: str) -> Optional[float]:
                    value = network_source.get(name)
                    try:
                        number = float(value) if value is not None else None
                    except (TypeError, ValueError):
                        return None
                    return number if number is not None and math.isfinite(number) else None

                try:
                    metagraph_block = (
                        int(network_source["metagraph_block"])
                        if network_source.get("metagraph_block") is not None
                        else None
                    )
                except (TypeError, ValueError):
                    metagraph_block = None
                network = {
                    "last_validator_weight": network_float("last_validator_weight"),
                    "metagraph_hotkey_ss58": str(
                        network_source.get("metagraph_hotkey_ss58") or ""
                    ),
                    "metagraph_incentive": network_float("metagraph_incentive"),
                    "metagraph_emission": network_float("metagraph_emission"),
                    "metagraph_trust": network_float("metagraph_trust"),
                    "metagraph_consensus": network_float("metagraph_consensus"),
                    "metagraph_block": metagraph_block,
                }
                if (
                    network["last_validator_weight"] is not None
                    and network["last_validator_weight"] > 0.0
                    and network["metagraph_incentive"] is not None
                    and network["metagraph_incentive"] <= 0.0
                ):
                    add_hint(
                        hints,
                        "local_weight_not_reflected",
                        "This validator assigned a positive last weight, but the cached metagraph incentive is still zero; network consensus may not have reflected it yet.",
                    )

                uid_next_steps: list[str] = []
                if active_entries == 0:
                    add_step(
                        uid_next_steps,
                        "Bring at least one endpoint online for this UID.",
                    )
                if uid_gate_active:
                    clear_epoch = uid_next_clear_epoch
                    suffix = (
                        f" Earliest UID-wide clear if clean: epoch {clear_epoch}."
                        if clear_epoch is not None else ""
                    )
                    add_step(
                        uid_next_steps,
                        "Do not add replacement endpoints to mask failing ones; fix the convicted executors and keep the UID clean."
                        + suffix,
                    )
                if inactive_recent_entries:
                    add_step(
                        uid_next_steps,
                        "Old inactive entries still exist in the lookback; explicitly deactivate broken endpoints and wait for the window to roll.",
                    )
                if on_probation:
                    add_step(
                        uid_next_steps,
                        "One or more executors are on probation; check each entry's probation.passes_remaining_if_clean.",
                    )

                primary_issue = "healthy"
                for code in (
                    "blacklisted",
                    "uid_audit_gate_active",
                    "model_gate_active",
                    "stale_uid_identity",
                    "proof_failure",
                    "tee_failure",
                    "chat_unauthorized",
                    "chat_forbidden",
                    "chat_not_found",
                    "timeout",
                    "connection_failed",
                    "tls_error",
                    "chat_error",
                    "no_active_endpoint",
                    "on_probation",
                    "new_entry_not_scored",
                    "local_weight_not_reflected",
                    "capacity_audit_failures",
                    "recent_endpoint_churn",
                ):
                    if any(h.get("code") == code for h in hints):
                        primary_issue = code
                        break

                result_uids[str(uid)] = {
                    "uid": uid,
                    "identity": {
                        "hotkey_ss58": str(
                            (owners_by_uid.get(uid) or {}).get("hotkey_ss58") or ""
                        ),
                        "evm_address": str(
                            (owners_by_uid.get(uid) or {}).get("evm_address") or ""
                        ).lower(),
                        "generation": int(
                            (owners_by_uid.get(uid) or {}).get("generation") or 0
                        ),
                        "identity_start_epoch": int(
                            (owners_by_uid.get(uid) or {}).get("identity_start_epoch") or 0
                        ),
                    },
                    "primary_issue": primary_issue,
                    "summary": {
                        "active_entries": active_entries,
                        "recent_entries": len(entries),
                        "inactive_recent_entries": inactive_recent_entries,
                        "entries_on_probation": on_probation,
                        "best_score": best_score,
                        "latest_scored_epoch": latest_scored_epoch,
                    },
                    "network": network,
                    "uid_gate": {
                        "configured": gate_configured,
                        "enabled": gate_enabled,
                        "suppression_reason": (
                            str(capacity_audit_gate_suppression_reason or "")
                            if gate_configured and not gate_enabled else ""
                        ),
                        "active": uid_gate_active,
                        "lookback_epochs": repeat_window,
                        "since_epoch": gate_since_epoch,
                        "convicted_entries": len(convicted),
                        "entry_count": entry_count,
                        "quorum": quorum,
                        "convicted": convicted_details,
                        "next_possible_clear_epoch": uid_next_clear_epoch,
                        "epochs_remaining_if_clean": (
                            uid_epochs_remaining if uid_gate_active else 0
                        ),
                        "estimated_hours_remaining_if_clean": (
                            uid_hours_remaining if uid_gate_active else 0.0
                        ),
                        "thresholds": {
                            "invalid_proof_misses": int(getattr(capacity_audit_cfg, "invalid_proof_misses_for_zero_score", 1) or 1),
                            "hard_proof_misses": int(getattr(capacity_audit_cfg, "hard_proof_misses_for_zero_score", 2) or 2),
                            "timing_misses": int(getattr(capacity_audit_cfg, "timing_misses_for_zero_score", 2) or 2),
                            "timing_only_allowed": bool(getattr(capacity_audit_cfg, "allow_timing_only_score_gate", True)),
                            "uid_min_entries": int(getattr(capacity_audit_cfg, "uid_escalation_min_entries", 2) or 2),
                            "uid_fraction": float(getattr(capacity_audit_cfg, "uid_escalation_fraction", 0.10) or 0.10),
                            "uid_max_entries": int(getattr(capacity_audit_cfg, "uid_escalation_max_entries", 10) or 10),
                        },
                    },
                    "hints": hints,
                    "next_steps": uid_next_steps,
                    "entries": entries,
                }

            return {
                "window_h": window_h,
                "window_epochs": window_epochs[window_h],
                "since_epoch": since_epoch,
                "uids": result_uids,
            }

        return {
            "enabled": True,
            "version": 1,
            "generated_at": now,
            "epoch_number": cur_epoch,
            "windows": {str(h): build_window(h) for h in windows},
        }

    # ── Shared state derivation ──────────────────────────────────────

    def derive_shared_state(self, epoch: int) -> ValidatorSharedState:
        """Build a ``ValidatorSharedState`` from the current DB contents.

        Queries active entries, joins with UID cache, and assembles the
        structure that gets written to JSON for the proxy.
        """
        entries = self.get_active_entries()
        all_entries = self._get_all_entries()
        uids = self.get_all_uids()
        probation = self.get_probation_addresses()
        audit_drains = self.get_capacity_drains()

        # Build per-address, per-model_index scores from ALL entries
        # (active + inactive).  Inactive entries appear with ema_score=0.0
        # so the proxy can distinguish "known offline" (explicit 0.0) from
        # "never seen" (absent → proxy defaults to 1.0 for new miners).
        # Active entries with ema_score=0 (just registered, not yet scored)
        # get a small positive value so the proxy still routes to them.
        miner_scores: Dict[str, Dict[str, float]] = {}
        for e in all_entries:
            addr = e["address"]
            idx_str = str(e["model_index"])
            if addr not in miner_scores:
                miner_scores[addr] = {}
            score = e["ema_score"]
            if not e.get("is_active"):
                score = 0.0
            elif score == 0:
                score = 0.01  # active but not yet scored — allow routing
            miner_scores[addr][idx_str] = score

        # Build miner_endpoints from active entries only — inactive miners
        # should not be listed as routable endpoints.
        miner_endpoints: List[MinerEntry] = []
        for e in entries:
            miner_endpoints.append(MinerEntry(
                address=e["address"],
                endpoint=e["endpoint"],
                model_id=e["model_id"],
                model_index=e["model_index"],
                quant=e["quant"],
                max_context_len=e["max_context_len"],
                tee_enabled=bool(e["tee_enabled"]) if "tee_enabled" in e.keys() else False,
                tee_platform=e["tee_platform"] if "tee_platform" in e.keys() else "",
                gpu_name=e["gpu_name"] if "gpu_name" in e.keys() else "",
                gpu_count=e["gpu_count"] if "gpu_count" in e.keys() else 0,
                vram_gb=e["vram_gb"] if "vram_gb" in e.keys() else 0,
                gpu_uuids=json.loads(e["gpu_uuids"]) if "gpu_uuids" in e.keys() else [],
            ))

        # Look up epoch start block from log
        with self._lock:
            row = self._conn.execute(
                "SELECT start_block FROM epoch_log WHERE epoch_number = ?",
                (epoch,),
            ).fetchone()
        start_block = row["start_block"] if row is not None else 0

        return ValidatorSharedState(
            epoch_number=epoch,
            epoch_start_block=start_block,
            miner_scores=miner_scores,
            probation_miners=probation,
            miner_endpoints=miner_endpoints,
            audit_drains=audit_drains,
            updated_at=time.time(),
        )

    # ── Analytics logging ────────────────────────────────────────────

    def log_canary_result(self, **kwargs) -> None:
        """Log an individual canary test result. No-op if analytics disabled."""
        if not self._analytics:
            return
        kwargs.setdefault("created_at", time.time())
        cols = list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO canary_results ({col_names}) VALUES ({placeholders})",
                tuple(kwargs[c] for c in cols),
            )
            self._conn.commit()

    def log_epoch_score(self, **kwargs) -> None:
        """Log a per-miner score at epoch close. No-op if analytics disabled."""
        if not self._analytics:
            return
        kwargs.setdefault("created_at", time.time())
        cols = list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO epoch_scores ({col_names}) VALUES ({placeholders})",
                tuple(kwargs[c] for c in cols),
            )
            self._conn.commit()

    def log_network_receipts(self, receipts: list, own_hotkey: bytes,
                             network: str, netuid: int,
                             ss58_lookup: dict) -> int:
        """Bulk-insert all epoch receipts into network_receipts.

        Receipts signed by ``own_hotkey`` are stored with is_own=1; the rest
        with is_own=0.  Preserves the full network view in analytics —
        previously own receipts were dropped, leaving the table biased toward
        other validators' perspective on our own miners.  Scoring does NOT
        read this table; it reads the in-memory receipt pull (see
        ``_close_epoch``).

        Args:
            receipts: List of ServiceReceipt objects (all validators' receipts).
            own_hotkey: This validator's 32-byte Sr25519 pubkey.
            network: "test" or "finney".
            netuid: Subnet ID.
            ss58_lookup: Dict[str, Dict[str, str]] — address.lower() → {hotkey_ss58, coldkey_ss58}.
        Returns:
            Number of rows inserted.
        """
        if not self._analytics:
            return 0
        now = time.time()
        rows = []
        for r in receipts:
            is_own = 1 if r.validator_hotkey == own_hotkey else 0
            addr = r.miner_address.lower()
            ss58 = ss58_lookup.get(addr, {})
            rows.append((
                r.epoch_number, r.miner_address,
                ss58.get("hotkey_ss58", ""), ss58.get("coldkey_ss58", ""),
                r.model_id, r.model_index,
                r.validator_hotkey.hex() if isinstance(r.validator_hotkey, bytes) else str(r.validator_hotkey),
                is_own,
                1 if r.is_canary else 0,
                r.ttft_ms, r.tokens_generated, r.generation_time_ms,
                r.tokens_per_sec, r.prompt_tokens,
                1 if r.proof_verified else 0,
                1 if r.proof_requested else 0,
                None if getattr(r, "tee_attestation_verified", None) is None else (1 if r.tee_attestation_verified else 0),
                r.commitment_hash.hex() if isinstance(r.commitment_hash, bytes) else str(r.commitment_hash),
                r.timestamp, network, netuid, now,
            ))
        if not rows:
            return 0
        with self._lock:
            # INSERT OR IGNORE: skip rows that violate the unique index on
            # (epoch_number, validator_hotkey, commitment_hash). Falls back
            # to plain INSERT if the index does not exist.
            self._conn.executemany(
                """INSERT OR IGNORE INTO network_receipts (
                    epoch_number, miner_address, miner_hotkey_ss58, miner_coldkey_ss58,
                    model_id, model_index, validator_hotkey, is_own, is_canary,
                    ttft_ms, tokens_generated, generation_time_ms, tokens_per_sec,
                    prompt_tokens, proof_verified, proof_requested, tee_attestation_verified,
                    commitment_hash, timestamp, network, netuid, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def backup_and_cleanup_canary_results(self, retain_days: int = 7, backup_dir: str = "") -> int:
        """Export old canary_results to .jsonl.gz, then delete from live DB."""
        import gzip
        import json as _json

        cutoff = time.time() - (retain_days * 86400)
        if not backup_dir:
            backup_dir = os.path.join(
                os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")), "backups",
            )
        os.makedirs(backup_dir, exist_ok=True)

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM canary_results WHERE created_at < ?", (cutoff,),
            ).fetchall()
            if not rows:
                return 0

            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"canary_results_{ts}.jsonl.gz")
            col_names = [desc[0] for desc in self._conn.execute(
                "SELECT * FROM canary_results LIMIT 0"
            ).description]

            with gzip.open(backup_path, "wt") as f:
                for row in rows:
                    f.write(_json.dumps(dict(zip(col_names, row)), default=str) + "\n")

            self._conn.execute(
                "DELETE FROM canary_results WHERE created_at < ?", (cutoff,),
            )
            self._conn.commit()
            bt.logging.info(f"Canary results backup: {len(rows)} rows → {backup_path}")
            return len(rows)

    def backup_and_cleanup_network_receipts(self, retain_days: int = 7, backup_dir: str = "") -> int:
        """Export old network_receipts to .jsonl.gz, then delete from live DB."""
        import gzip
        import json as _json

        cutoff = time.time() - (retain_days * 86400)
        if not backup_dir:
            backup_dir = os.path.join(
                os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")), "backups",
            )
        os.makedirs(backup_dir, exist_ok=True)

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM network_receipts WHERE created_at < ?", (cutoff,),
            ).fetchall()
            if not rows:
                return 0

            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"network_receipts_{ts}.jsonl.gz")
            col_names = [desc[0] for desc in self._conn.execute(
                "SELECT * FROM network_receipts LIMIT 0"
            ).description]

            with gzip.open(backup_path, "wt") as f:
                for row in rows:
                    f.write(_json.dumps(dict(zip(col_names, row)), default=str) + "\n")

            self._conn.execute(
                "DELETE FROM network_receipts WHERE created_at < ?", (cutoff,),
            )
            self._conn.commit()
            bt.logging.info(f"Network receipts backup: {len(rows)} rows → {backup_path}")
            return len(rows)

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
