"""SQLite-backed receipt storage for the miner server.

Receipts are stored both in-memory (for fast access) and on disk (for
crash recovery).  On startup, any receipts from recent epochs are loaded
from the database.  This ensures that a server restart mid-epoch does
not lose validator receipts, which would trigger receipt integrity failure
and a hard score penalty.

The database file lives next to the server data directory. Receipts remain
small; completed proof-v3 hard bundles are retained separately as bounded
binary objects for the configured epoch window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS proof_v3_hard_bundles (
                epoch INTEGER NOT NULL,
                commitment_hash BLOB NOT NULL,
                bundle_hash BLOB NOT NULL,
                proof_challenge_id BLOB NOT NULL,
                execution_profile_digest BLOB NOT NULL,
                bundle BLOB NOT NULL,
                receipt_json TEXT NOT NULL,
                PRIMARY KEY (epoch, commitment_hash)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_proof_v3_hard_bundles_epoch
            ON proof_v3_hard_bundles(epoch)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS proof_v3_pending_hard_bundles (
                commitment_hash BLOB PRIMARY KEY,
                bundle_hash BLOB NOT NULL,
                bundle BLOB NOT NULL,
                created_at REAL NOT NULL
            )
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
                self._load_epoch_locked(epoch)
            self._cache[epoch].append(receipt_dict)

            self._conn.execute(
                "INSERT INTO receipts (epoch, receipt_json) VALUES (?, ?)",
                (epoch, json.dumps(receipt_dict)),
            )
            self._conn.commit()
            return len(self._cache[epoch])

    def add_unique(
        self,
        epoch: int,
        receipt_dict: dict,
        *,
        unique_fields: tuple[str, ...],
    ) -> tuple[int, bool]:
        """Persist a receipt once for the supplied canonical identity fields.

        The lookup and insert share the store lock, so concurrent HTTP handler
        threads cannot race the same signed receipt into SQLite twice.  The
        returned boolean is ``False`` for an idempotent duplicate.
        """

        if not unique_fields:
            raise ValueError("unique_fields must not be empty")
        try:
            identity = tuple(receipt_dict[field] for field in unique_fields)
        except KeyError as exc:
            raise ValueError(
                f"receipt is missing uniqueness field: {exc.args[0]}"
            ) from exc

        with self._lock:
            if epoch not in self._cache:
                self._load_epoch_locked(epoch)
            receipts = self._cache[epoch]
            for existing in receipts:
                if tuple(existing.get(field) for field in unique_fields) == identity:
                    return len(receipts), False

            receipts.append(receipt_dict)
            self._conn.execute(
                "INSERT INTO receipts (epoch, receipt_json) VALUES (?, ?)",
                (epoch, json.dumps(receipt_dict)),
            )
            self._conn.commit()
            return len(receipts), True

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

    def add_proof_v3_hard_bundle(
        self,
        *,
        epoch: int,
        receipt_dict: dict,
        encoded_bundle: bytes,
    ) -> dict:
        """Persist one receipt-matched hard bundle immutably.

        The receipt remains the epoch/validator authority. The bundle parser
        supplies the independently verifiable request, nonce and proof
        transcript. A retry may repeat the exact pair, but the same epoch and
        envelope digest can never be replaced with different bytes.
        """

        from verallm.proof_v3.hard_bundle import (
            MAX_HARD_BUNDLE_BYTES_V3,
            RetainedHardProofBundleV3,
        )

        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or not isinstance(receipt_dict, dict)
            or not isinstance(encoded_bundle, bytes)
            or not 0 < len(encoded_bundle) <= MAX_HARD_BUNDLE_BYTES_V3
        ):
            raise ValueError("proof-v3 hard bundle input is malformed")
        bundle = RetainedHardProofBundleV3.from_canonical_bytes(
            encoded_bundle
        )
        commitment_hash = bundle.commitment_envelope_digest
        if (
            int(receipt_dict.get("epoch_number", -1)) != epoch
            or receipt_dict.get("commitment_hash") != commitment_hash.hex()
            or receipt_dict.get("proof_requested") is not True
            or receipt_dict.get("proof_verified") is not True
            or receipt_dict.get("is_canary") is not True
        ):
            raise ValueError(
                "proof-v3 hard bundle does not match its verified receipt"
            )
        bundle_hash = hashlib.sha256(encoded_bundle).digest()
        receipt_json = json.dumps(
            receipt_dict,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT bundle_hash, receipt_json
                  FROM proof_v3_hard_bundles
                 WHERE epoch = ? AND commitment_hash = ?
                """,
                (epoch, sqlite3.Binary(commitment_hash)),
            ).fetchone()
            if existing is not None:
                if (
                    bytes(existing[0]) != bundle_hash
                    or str(existing[1]) != receipt_json
                ):
                    raise ValueError(
                        "proof-v3 hard bundle identity is already retained"
                    )
            else:
                self._conn.execute(
                    """
                    INSERT INTO proof_v3_hard_bundles (
                        epoch, commitment_hash, bundle_hash,
                        proof_challenge_id, execution_profile_digest,
                        bundle, receipt_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        epoch,
                        sqlite3.Binary(commitment_hash),
                        sqlite3.Binary(bundle_hash),
                        sqlite3.Binary(bundle.proof_challenge_id),
                        sqlite3.Binary(
                            bundle.precommit_context.execution_profile_digest
                        ),
                        sqlite3.Binary(encoded_bundle),
                        receipt_json,
                    ),
                )
                self._conn.commit()
        return {
            "commitment_hash": commitment_hash.hex(),
            "bundle_hash": bundle_hash.hex(),
            "proof_challenge_id": bundle.proof_challenge_id.hex(),
            "execution_profile_digest": (
                bundle.precommit_context.execution_profile_digest.hex()
            ),
            "byte_length": len(encoded_bundle),
            "receipt": dict(receipt_dict),
        }

    def stage_proof_v3_hard_bundle(self, encoded_bundle: bytes) -> dict:
        """Persist a completed proof before its validator receipt arrives."""

        from verallm.proof_v3.hard_bundle import (
            MAX_HARD_BUNDLE_BYTES_V3,
            RetainedHardProofBundleV3,
        )

        if (
            not isinstance(encoded_bundle, bytes)
            or not 0 < len(encoded_bundle) <= MAX_HARD_BUNDLE_BYTES_V3
        ):
            raise ValueError("proof-v3 hard bundle input is malformed")
        bundle = RetainedHardProofBundleV3.from_canonical_bytes(
            encoded_bundle
        )
        commitment_hash = bundle.commitment_envelope_digest
        bundle_hash = hashlib.sha256(encoded_bundle).digest()
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT bundle_hash
                  FROM proof_v3_pending_hard_bundles
                 WHERE commitment_hash = ?
                """,
                (sqlite3.Binary(commitment_hash),),
            ).fetchone()
            if existing is not None and bytes(existing[0]) != bundle_hash:
                raise ValueError(
                    "proof-v3 pending hard bundle identity already exists"
                )
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO proof_v3_pending_hard_bundles (
                        commitment_hash, bundle_hash, bundle, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        sqlite3.Binary(commitment_hash),
                        sqlite3.Binary(bundle_hash),
                        sqlite3.Binary(encoded_bundle),
                        time.time(),
                    ),
                )
                self._conn.commit()
        return {
            "commitment_hash": commitment_hash.hex(),
            "bundle_hash": bundle_hash.hex(),
            "proof_challenge_id": bundle.proof_challenge_id.hex(),
            "execution_profile_digest": (
                bundle.precommit_context.execution_profile_digest.hex()
            ),
            "byte_length": len(encoded_bundle),
        }

    def promote_proof_v3_hard_bundle(
        self,
        *,
        epoch: int,
        receipt_dict: dict,
    ) -> dict | None:
        """Attach a staged bundle to its validator-signed epoch receipt."""

        try:
            commitment_hash = bytes.fromhex(
                str(receipt_dict.get("commitment_hash", ""))
            )
        except ValueError as exc:
            raise ValueError("proof-v3 receipt commitment is malformed") from exc
        if len(commitment_hash) != 32:
            raise ValueError("proof-v3 receipt commitment is malformed")
        with self._lock:
            row = self._conn.execute(
                """
                SELECT bundle
                  FROM proof_v3_pending_hard_bundles
                 WHERE commitment_hash = ?
                """,
                (sqlite3.Binary(commitment_hash),),
            ).fetchone()
        if row is None:
            return None
        result = self.add_proof_v3_hard_bundle(
            epoch=epoch,
            receipt_dict=receipt_dict,
            encoded_bundle=bytes(row[0]),
        )
        with self._lock:
            self._conn.execute(
                """
                DELETE FROM proof_v3_pending_hard_bundles
                 WHERE commitment_hash = ?
                """,
                (sqlite3.Binary(commitment_hash),),
            )
            self._conn.commit()
        return result

    def get_proof_v3_hard_bundle_index(self, epoch: int) -> List[dict]:
        """Return the canonical, commitment-sorted public index for an epoch."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT commitment_hash, bundle_hash, proof_challenge_id,
                       execution_profile_digest, length(bundle), receipt_json
                  FROM proof_v3_hard_bundles
                 WHERE epoch = ?
                 ORDER BY commitment_hash ASC
                """,
                (int(epoch),),
            ).fetchall()
        return [
            {
                "commitment_hash": bytes(row[0]).hex(),
                "bundle_hash": bytes(row[1]).hex(),
                "proof_challenge_id": bytes(row[2]).hex(),
                "execution_profile_digest": bytes(row[3]).hex(),
                "byte_length": int(row[4]),
                "receipt": json.loads(row[5]),
            }
            for row in rows
        ]

    def get_proof_v3_hard_bundle(
        self,
        *,
        epoch: int,
        commitment_hash: bytes,
    ) -> bytes | None:
        """Return one retained bundle after checking its stored checksum."""

        if not isinstance(commitment_hash, bytes) or len(commitment_hash) != 32:
            raise ValueError("proof-v3 commitment hash must be 32 bytes")
        with self._lock:
            row = self._conn.execute(
                """
                SELECT bundle_hash, bundle
                  FROM proof_v3_hard_bundles
                 WHERE epoch = ? AND commitment_hash = ?
                """,
                (int(epoch), sqlite3.Binary(commitment_hash)),
            ).fetchone()
        if row is None:
            return None
        encoded = bytes(row[1])
        if hashlib.sha256(encoded).digest() != bytes(row[0]):
            raise RuntimeError("retained proof-v3 hard bundle checksum failed")
        return encoded

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
            self._conn.execute(
                """
                DELETE FROM proof_v3_pending_hard_bundles
                 WHERE created_at < ?
                """,
                (time.time() - 6 * 3600.0,),
            )
            self._conn.execute(
                """
                DELETE FROM proof_v3_hard_bundles
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
