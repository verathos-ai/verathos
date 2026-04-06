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

from neurons.shared_state import MinerEntry, ValidatorSharedState

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")),
    "verathos_validator.db",
)

_SCHEMA_VERSION = "1"


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
        """)
        self._conn.commit()

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

    def get_active_entries(self) -> List[dict]:
        """Return all active miner-model entries as dicts."""
        cursor = self._conn.execute(
            "SELECT * FROM miner_entries WHERE is_active = 1"
        )
        return [dict(row) for row in cursor.fetchall()]

    def _get_all_entries(self) -> List[dict]:
        """Return all miner-model entries (active and inactive) as dicts."""
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
        cursor = self._conn.execute(
            """SELECT address, model_index, model_id,
                      ema_score, total_epochs, scored_epochs
               FROM miner_entries"""
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
            if score == 0 and e.get("is_active"):
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
        """Bulk-insert OTHER validators' epoch receipts into network_receipts. No-op if analytics disabled.

        Skips our own receipts (is_own=1) since those are already in the
        proxy's request_log with richer detail (user, cost, auth).

        Args:
            receipts: List of ServiceReceipt objects (all validators' receipts).
            own_hotkey: This validator's 32-byte Ed25519 pubkey (to filter out).
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
            # Skip our own receipts — organic are in proxy request_log,
            # canaries are in canary_results, both with richer detail.
            if r.validator_hotkey == own_hotkey:
                continue
            addr = r.miner_address.lower()
            ss58 = ss58_lookup.get(addr, {})
            rows.append((
                r.epoch_number, r.miner_address,
                ss58.get("hotkey_ss58", ""), ss58.get("coldkey_ss58", ""),
                r.model_id, r.model_index,
                r.validator_hotkey.hex() if isinstance(r.validator_hotkey, bytes) else str(r.validator_hotkey),
                0,  # is_own = always 0 (we skip our own)
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
            self._conn.executemany(
                """INSERT INTO network_receipts (
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
        if self._conn:
            self._conn.close()
            self._conn = None
