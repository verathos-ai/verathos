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
import bittensor as bt
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

from neurons.shared_state import AuditDrain, MinerEntry, ValidatorSharedState

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
    "verathos_validator.db",
)

_SCHEMA_VERSION = "1"


def _coerce_nonnegative_int(value: object) -> int:
    """Return a DB-safe non-negative integer for optional miner metadata."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


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
                "SELECT model_id FROM miner_entries "
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
                            gpu_name = COALESCE(NULLIF(?, ''), gpu_name),
                            gpu_count = CASE WHEN ? > 0 THEN ? ELSE gpu_count END,
                            vram_gb = CASE WHEN ? > 0 THEN ? ELSE vram_gb END,
                            compute_capability = COALESCE(NULLIF(?, ''), compute_capability),
                            gpu_uuids = CASE WHEN ? != '[]' THEN ? ELSE gpu_uuids END,
                            hotkey_ss58 = COALESCE(NULLIF(?, ''), hotkey_ss58),
                            coldkey_ss58 = COALESCE(NULLIF(?, ''), coldkey_ss58),
                            updated_at = ?
                        WHERE address = ? AND model_index = ?""",
                        (model_id, endpoint, quant, max_context_len,
                         epoch, 1 if tee_enabled else 0, tee_platform,
                         gpu_name, gpu_count, gpu_count, vram_gb, vram_gb, compute_capability,
                         _gpu_uuids_json, _gpu_uuids_json,
                         hotkey_ss58, coldkey_ss58, now, address, model_index),
                    )
                    bt.logging.info(f"Model switch for {address[:10]} idx={model_index}: {old_model_id} -> {model_id} (score reset)")
                else:
                    # Same model — update endpoint/quant/epoch, keep scores
                    self._conn.execute(
                        """UPDATE miner_entries SET
                            endpoint = ?, quant = ?, max_context_len = ?,
                            last_seen_epoch = ?, is_active = 1,
                            tee_enabled = ?, tee_platform = ?,
                            gpu_name = COALESCE(NULLIF(?, ''), gpu_name),
                            gpu_count = CASE WHEN ? > 0 THEN ? ELSE gpu_count END,
                            vram_gb = CASE WHEN ? > 0 THEN ? ELSE vram_gb END,
                            compute_capability = COALESCE(NULLIF(?, ''), compute_capability),
                            gpu_uuids = CASE WHEN ? != '[]' THEN ? ELSE gpu_uuids END,
                            hotkey_ss58 = COALESCE(NULLIF(?, ''), hotkey_ss58),
                            coldkey_ss58 = COALESCE(NULLIF(?, ''), coldkey_ss58),
                            updated_at = ?
                        WHERE address = ? AND model_index = ?""",
                        (endpoint, quant, max_context_len, epoch,
                         1 if tee_enabled else 0, tee_platform,
                         gpu_name, gpu_count, gpu_count, vram_gb, vram_gb, compute_capability,
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

    def get_active_entries(self) -> List[dict]:
        """Return all active miner-model entries as dicts."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM miner_entries WHERE is_active = 1"
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
                        audit_id, miner_address, model_index, miner_uid, endpoint,
                        model_id, quant, max_context_len, gpu_name, gpu_count,
                        vram_gb, group_key, slot_id, lease_id, claimed_gpu_class,
                        pass_count, deadline_s, transport_grace_s,
                        payload_deadline_s, drain_until_ts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        audit_id,
                        str(slot["miner_address"]).lower(),
                        int(slot["model_index"]),
                        slot.get("miner_uid"),
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
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE s.miner_address = ?
                      AND s.model_index = ?
                      AND w.epoch_number >= ?
                      AND s.verdict IN ({placeholders})
                      {confirmation_clause}""",
                (address.lower(), int(model_index), int(since_epoch), *verdicts),
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
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM capacity_audit_slots s
                    JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                    WHERE w.epoch_number >= ?
                      AND s.verdict IN ({placeholders})
                      AND (
                        s.miner_uid = ?
                        OR s.miner_address IN (
                          SELECT address FROM miner_entries WHERE bittensor_uid = ?
                        )
                      )
                      {confirmation_clause}""",
                (int(since_epoch), *verdicts, int(uid), int(uid)),
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
        retain_artifacts: bool = False,
    ) -> dict[str, int]:
        if retain_artifacts:
            return {
                "artifact_files_deleted": 0,
                "artifact_rows_cleared": 0,
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
        keep_from_epoch = max(
            0,
            int(current_epoch) - max(1, int(retain_failure_epochs)) + 1,
        )
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
                (time.time(), int(current_epoch), *final_proof_statuses, *failure_verdicts),
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
            "success_rows_deleted": success_cur.rowcount or 0,
            "old_failure_rows_deleted": failure_cur.rowcount or 0,
            "empty_windows_deleted": window_cur.rowcount or 0,
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
