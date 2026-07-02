"""SQLite-backed receipt storage for the miner server.

Receipts are stored both in-memory (for fast access) and on disk (for
crash recovery).  On startup, any receipts from recent epochs are loaded
from the database.  This ensures that a server restart mid-epoch does
not lose validator receipts, which would trigger receipt integrity failure
and a hard score penalty.

The database file lives next to the server data directory and is small
(~1 KB per receipt, typically <100 receipts per epoch).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Dict, List

logger = logging.getLogger(__name__)

# Keep receipts for the last N epochs (matches server GC policy)
_KEEP_EPOCHS = 3

# Default database path (overridable via VERALLM_RECEIPT_DB env var)
DEFAULT_DB_PATH = os.path.join(
    os.environ.get("VERALLM_DATA_DIR", "/tmp"),
    "verathos_receipts.db",
)


class ReceiptStore:
    """SQLite-backed receipt storage with in-memory cache.

    Thread-safe: uses a lock around all SQLite operations (SQLite in WAL
    mode handles concurrent reads, but Python's sqlite3 module is not
    thread-safe by default with shared connections).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._cache: Dict[int, List[dict]] = {}  # epoch -> [receipt_dict]
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        self._load_recent()

    def _init_db(self) -> None:
        """Create the database and table if they don't exist."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                receipt_json TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_receipts_epoch ON receipts(epoch)
        """)
        self._conn.commit()
        logger.info("Receipt store initialized: %s", self._db_path)

    def _load_recent(self) -> None:
        """Load receipts from recent epochs into memory cache."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT epoch FROM receipts ORDER BY epoch DESC LIMIT ?",
                (_KEEP_EPOCHS,),
            )
            epochs = [row[0] for row in cursor.fetchall()]

            for epoch in epochs:
                cursor = self._conn.execute(
                    "SELECT receipt_json FROM receipts WHERE epoch = ?",
                    (epoch,),
                )
                self._cache[epoch] = [
                    json.loads(row[0]) for row in cursor.fetchall()
                ]

            total = sum(len(v) for v in self._cache.values())
            if total > 0:
                logger.info(
                    "Loaded %d receipts from %d epochs on startup",
                    total, len(self._cache),
                )

    def _load_epoch_locked(self, epoch: int) -> None:
        """Load one epoch into cache. Caller must hold _lock."""
        cursor = self._conn.execute(
            "SELECT receipt_json FROM receipts WHERE epoch = ?",
            (epoch,),
        )
        self._cache[epoch] = [
            json.loads(row[0]) for row in cursor.fetchall()
        ]

    @staticmethod
    def _gc_comparable_epoch(epoch: int, current_epoch: int) -> bool:
        """Return whether two epochs are safe to compare for retention.

        Older rollout code used a different epoch divisor, producing epoch
        numbers roughly 6x larger than the current subnet tempo epoch. A
        receipt from that older scheme must not make the miner delete current
        epoch receipts just because its integer value is larger.
        """
        if current_epoch < 1024 or epoch < 1024:
            return True
        return epoch >= current_epoch // 2

    def add(self, epoch: int, receipt_dict: dict) -> int:
        """Add a receipt and persist to disk. Returns count for this epoch."""
        with self._lock:
            if epoch not in self._cache:
                self._cache[epoch] = []
            self._cache[epoch].append(receipt_dict)

            self._conn.execute(
                "INSERT INTO receipts (epoch, receipt_json) VALUES (?, ?)",
                (epoch, json.dumps(receipt_dict)),
            )
            self._conn.commit()
            return len(self._cache[epoch])

    def get(self, epoch: int) -> List[dict]:
        """Get all receipts for an epoch."""
        with self._lock:
            if epoch not in self._cache:
                self._load_epoch_locked(epoch)
            return list(self._cache.get(epoch, []))

    def count(self, epoch: int) -> int:
        """Get receipt count for an epoch."""
        with self._lock:
            if epoch not in self._cache:
                self._load_epoch_locked(epoch)
            return len(self._cache.get(epoch, []))

    def gc(self, current_epoch: int) -> None:
        """Remove receipts older than current_epoch - _KEEP_EPOCHS."""
        cutoff = current_epoch - _KEEP_EPOCHS
        with self._lock:
            stale = [
                e for e in self._cache
                if e < cutoff and self._gc_comparable_epoch(e, current_epoch)
            ]
            for e in stale:
                del self._cache[e]

            self._conn.execute(
                """
                DELETE FROM receipts
                 WHERE epoch < ?
                   AND (? < 1024 OR epoch < 1024 OR epoch >= ?)
                """,
                (cutoff, current_epoch, current_epoch // 2),
            )
            self._conn.commit()

            if stale:
                logger.debug(
                    "GC'd receipts for epochs %s (cutoff=%d)", stale, cutoff,
                )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
