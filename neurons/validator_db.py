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
import re
import sqlite3
import statistics
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    if (
        "504 gateway" in msg
        or "gateway timeout" in msg
        or "gateway time-out" in msg
        or "status code 504" in msg
        or "http status 504" in msg
        or "response status 504" in msg
    ):
        return "reverse_proxy_timeout"
    if (
        "produced no token before the routing deadline" in msg
        or "first token timeout" in msg
        or "first-token timeout" in msg
        or "ttft deadline" in msg
    ):
        return "first_token_timeout"
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return "rate_limited"
    if "401" in msg or "unauthorized" in msg:
        return "chat_unauthorized"
    if "403" in msg or "forbidden" in msg:
        return "chat_forbidden"
    if "404" in msg or "not found" in msg:
        return "chat_not_found"
    if (
        "400 bad request" in msg
        or "status code 400" in msg
        or "http status 400" in msg
        or "422 unprocessable" in msg
        or "status code 422" in msg
        or "http status 422" in msg
    ):
        return "bad_request"
    if (
        "500 internal server" in msg
        or "502 bad gateway" in msg
        or "503 service unavailable" in msg
        or "status code 500" in msg
        or "status code 502" in msg
        or "status code 503" in msg
        or "http status 500" in msg
        or "http status 502" in msg
        or "http status 503" in msg
    ):
        return "service_unavailable"
    if (
        "empty response" in msg
        or "response body is empty" in msg
        or "hard proof is empty" in msg
        or "stream ended before" in msg
    ):
        return "empty_response"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if (
        "name or service not known" in msg
        or "temporary failure in name resolution" in msg
        or "nodename nor servname" in msg
        or "getaddrinfo failed" in msg
    ):
        return "dns_error"
    if "ssl" in msg or "certificate" in msg or "tls" in msg:
        return "tls_error"
    if "connection" in msg or "connect" in msg or "network is unreachable" in msg:
        return "connection_failed"
    return "chat_error"


_DEBUG_ERROR_SUMMARIES = {
    "chat_unauthorized": "Inference request was unauthorized.",
    "chat_forbidden": "Inference request was forbidden.",
    "chat_not_found": "Inference route was not found.",
    "reverse_proxy_timeout": "The miner's reverse proxy expired before the request deadline.",
    "first_token_timeout": "The miner produced no token before the routing deadline.",
    "rate_limited": "The miner endpoint rate-limited the validator request.",
    "bad_request": "The miner rejected the validator request as malformed or incompatible.",
    "service_unavailable": "The miner inference service returned an upstream server error.",
    "empty_response": "The miner ended the response without the required payload.",
    "timeout": "Inference request timed out.",
    "dns_error": "The registered endpoint hostname could not be resolved.",
    "connection_failed": "Validator could not connect to the inference endpoint.",
    "tls_error": "Inference endpoint failed TLS validation.",
    "chat_error": "Inference request failed.",
    "proof_failure": "Proof verification failed.",
    "tee_failure": "TEE attestation verification failed.",
}


_DEBUG_ERROR_NEXT_STEPS = {
    "chat_unauthorized": "Check validator discovery/allowlisting and request authentication on the miner.",
    "chat_forbidden": "Check the miner's validator allowlist and reverse-proxy access rules.",
    "chat_not_found": "Run the current official installer and confirm the registered endpoint exposes the inference and proof-v3 routes.",
    "reverse_proxy_timeout": "Run the current official installer or raise the upstream proxy read timeout to the minimum printed at miner startup.",
    "first_token_timeout": "Inspect the model-engine queue, GPU memory and inference logs; the endpoint accepted the request but emitted no token in time.",
    "rate_limited": "Remove unintended rate limiting from validator traffic or raise the endpoint's authenticated validator allowance.",
    "bad_request": "Update the miner to the current release and inspect route/payload compatibility in the miner log.",
    "service_unavailable": "Inspect the miner and model-engine logs for the upstream 5xx failure and restore the inference service.",
    "empty_response": "Inspect the inference/proof worker for an early exit after accepting the request.",
    "timeout": "Inspect inference load and transport timeouts; the request exceeded the validator's configured deadline.",
    "dns_error": "Correct the registered endpoint hostname and verify it resolves publicly.",
    "connection_failed": "Confirm the registered endpoint, listener and reverse proxy are online and publicly reachable.",
    "tls_error": "Repair the endpoint certificate chain, hostname coverage and system clock.",
    "chat_error": "Inspect the miner log for the matching canary request and restore the failing inference route.",
    "proof_failure": "Inspect proof-v3 generation logs and the retained failure code; update the miner before retrying if its runtime or artifact is incompatible.",
    "tee_failure": "Check the TEE attestation path and registered enclave identity.",
}


def _debug_error_next_step(kind: object) -> str:
    return _DEBUG_ERROR_NEXT_STEPS.get(
        str(kind or ""),
        "Inspect the miner log for the matching validator request.",
    )


def _debug_capacity_audit_next_step(reason: object) -> str:
    """Return bounded operator guidance for a public capacity failure code."""

    code = str(reason or "").strip().lower()
    if code == "missing_final_receipt":
        return "Check the capacity-audit worker, chain connection and final-receipt upload path."
    if code in {"missing_proof_payload", "missing_payload"}:
        return "Check the post-challenge capacity proof worker and proof-payload upload path."
    if code == "deadline_exceeded":
        return "Check capacity benchmark load, audit drain timing and host performance against the signed deadline."
    if code in {"pass0_root_mismatch", "v2_final_commitment_not_pre_challenge"}:
        return "Update the miner and inspect capacity transcript construction; its signed commitments were inconsistent."
    if code == "validator_verify_error":
        return "No miner action is indicated by this event; the validator verifier failed locally and should retry."
    if (
        code.startswith("unsupported_")
        or code.startswith("legacy_")
        or code == "missing_v2_pre_challenge_final_commitment"
    ):
        return "Update the miner to the current release; its capacity proof protocol is not accepted."
    if code:
        return f"Inspect the capacity-audit worker for failure code {code}."
    return "Inspect capacity-audit execution and publishing for this endpoint."


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
        # Receipt ingress has a hard HTTP response budget.  It must not queue
        # behind long validator-side analytics/shared-state reads merely
        # because those reads use the primary connection's Python lock.  WAL
        # permits this dedicated connection to durably record receipts while
        # the primary connection is reading.
        self._capacity_ingress_lock = threading.Lock()
        self._capacity_ingress_conn: sqlite3.Connection | None = None
        self._init_db()
        self._capacity_ingress_conn = sqlite3.connect(
            self._db_path,
            timeout=4.0,
            check_same_thread=False,
        )
        self._capacity_ingress_conn.row_factory = sqlite3.Row
        self._capacity_ingress_conn.execute("PRAGMA journal_mode=WAL")
        self._capacity_ingress_conn.execute("PRAGMA busy_timeout=4000")

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
                last_scored_epoch INTEGER,

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

            -- Signed, endpoint-free verification views for private meshes.
            -- Every accepted generation remains immutable for audit.
            CREATE TABLE IF NOT EXISTS mesh_verification_snapshots (
                coordinator_address TEXT NOT NULL,
                model_index          INTEGER NOT NULL,
                epoch_number         INTEGER NOT NULL,
                generation           INTEGER NOT NULL,
                snapshot_hash        TEXT NOT NULL,
                snapshot_json        TEXT NOT NULL,
                fetched_at           REAL NOT NULL,
                expires_at_unix      INTEGER NOT NULL,

                PRIMARY KEY (
                    coordinator_address,
                    model_index,
                    epoch_number,
                    generation
                )
            );

            CREATE INDEX IF NOT EXISTS idx_mesh_verification_snapshot_latest
                ON mesh_verification_snapshots (
                    coordinator_address,
                    model_index,
                    epoch_number,
                    generation DESC
                );

            CREATE INDEX IF NOT EXISTS idx_mesh_verification_snapshot_expiry
                ON mesh_verification_snapshots(expires_at_unix);

            -- Immutable identity of the topology that produced each completed
            -- score sample.  The EMA itself remains scoped to the registered
            -- coordinator/model slot and can therefore carry across a mesh
            -- relaunch; this table prevents an older sample from being
            -- presented as evidence for the replacement topology.
            CREATE TABLE IF NOT EXISTS score_samples (
                coordinator_address TEXT NOT NULL,
                model_index          INTEGER NOT NULL,
                score_epoch          INTEGER NOT NULL,
                model_id             TEXT NOT NULL,
                ema_score            REAL NOT NULL,
                chain_id             INTEGER,
                netuid               INTEGER,
                mesh_id              TEXT,
                snapshot_hash        TEXT,
                snapshot_generation  INTEGER,
                recorded_at          REAL NOT NULL,

                PRIMARY KEY (
                    coordinator_address,
                    model_index,
                    score_epoch
                )
            );

            CREATE INDEX IF NOT EXISTS idx_score_samples_latest
                ON score_samples (
                    coordinator_address,
                    model_index,
                    score_epoch DESC
                );

            -- Exactly-once journal for proof failures relayed by the local
            -- proxy. The event insert and financial/probation mutation share
            -- one SQLite transaction, so a replay cannot halve EMA twice.
            CREATE TABLE IF NOT EXISTS proxy_proof_failure_events (
                event_id           TEXT PRIMARY KEY,
                miner_address      TEXT NOT NULL,
                model_index        INTEGER NOT NULL,
                epoch_number       INTEGER NOT NULL,
                event_timestamp    INTEGER NOT NULL,
                applied_at         REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_proxy_proof_failure_slot
                ON proxy_proof_failure_events (
                    miner_address,
                    model_index,
                    epoch_number
                );

            -- Persistent ordered epoch-close queue.  A row is created when an
            -- epoch starts and sealed with its immutable scoring context at the
            -- next boundary.  Failed/running rows survive validator restarts;
            -- callers always retry the lowest epoch first.
            CREATE TABLE IF NOT EXISTS epoch_close_queue (
                epoch_number       INTEGER PRIMARY KEY,
                start_block        INTEGER NOT NULL,
                close_block        INTEGER NOT NULL,
                context_json       TEXT,
                status             TEXT NOT NULL DEFAULT 'collecting',
                attempt_count      INTEGER NOT NULL DEFAULT 0,
                last_error         TEXT NOT NULL DEFAULT '',
                abandoned          INTEGER NOT NULL DEFAULT 0,
                created_at         REAL NOT NULL,
                updated_at         REAL NOT NULL,
                completed_at       REAL,
                CHECK (status IN ('collecting', 'pending', 'running', 'completed'))
            );

            CREATE INDEX IF NOT EXISTS idx_epoch_close_queue_pending
                ON epoch_close_queue(status, epoch_number);

            -- Per-slot close checkpoint.  Score application and the checkpoint
            -- update share one transaction; completion is written only after
            -- every deterministic probation/penalty event for the slot ran.
            CREATE TABLE IF NOT EXISTS epoch_close_slots (
                epoch_number       INTEGER NOT NULL,
                miner_address      TEXT NOT NULL,
                model_index        INTEGER NOT NULL,
                status             TEXT NOT NULL DEFAULT 'pending',
                score_applied      INTEGER NOT NULL DEFAULT 0,
                epoch_score        REAL,
                epoch_score_is_null INTEGER NOT NULL DEFAULT 0,
                ema_score          REAL,
                total_epochs       INTEGER,
                scored_epochs      INTEGER,
                last_scored_epoch  INTEGER,
                created_at         REAL NOT NULL,
                updated_at         REAL NOT NULL,
                completed_at       REAL,
                PRIMARY KEY (epoch_number, miner_address, model_index),
                CHECK (status IN ('pending', 'completed'))
            );

            -- Deterministic close-side mutations (probation passes/failures,
            -- EMA penalties, and score gates).  The event insert is committed
            -- atomically with its miner_entries mutation.
            CREATE TABLE IF NOT EXISTS epoch_close_slot_events (
                epoch_number       INTEGER NOT NULL,
                miner_address      TEXT NOT NULL,
                model_index        INTEGER NOT NULL,
                event_kind         TEXT NOT NULL,
                payload_json       TEXT NOT NULL DEFAULT '{}',
                applied_at         REAL NOT NULL,
                PRIMARY KEY (
                    epoch_number,
                    miner_address,
                    model_index,
                    event_kind
                )
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

            -- Validator-owned signed receipts are authoritative for its own
            -- canary obligations.  They must survive restart and must not
            -- depend on the miner returning the receipt at epoch close.
            CREATE TABLE IF NOT EXISTS local_service_receipts (
                epoch_number        INTEGER NOT NULL,
                validator_signature TEXT NOT NULL,
                miner_address       TEXT NOT NULL,
                model_index         INTEGER NOT NULL,
                receipt_json        TEXT NOT NULL,
                created_at          REAL NOT NULL,
                PRIMARY KEY (epoch_number, validator_signature)
            );

            CREATE INDEX IF NOT EXISTS idx_local_service_receipts_epoch
                ON local_service_receipts(epoch_number);
            CREATE INDEX IF NOT EXISTS idx_local_service_receipts_miner
                ON local_service_receipts(
                    epoch_number, miner_address, model_index
                );

            -- Journal of every canary obligation this validator planned for
            -- an epoch, written at plan time.  A mid-epoch validator restart
            -- re-plans the epoch under a fresh secret salt; receipts signed
            -- under the discarded plan survive (locally and on the miner)
            -- and must be recognized at close as validator-produced residue
            -- rather than forgeries.  INSERT OR IGNORE keyed on
            -- (epoch, obligation_id) accumulates every plan of the epoch.
            CREATE TABLE IF NOT EXISTS planned_canary_obligations (
                epoch_number         INTEGER NOT NULL,
                obligation_id        TEXT NOT NULL,
                miner_address        TEXT NOT NULL,
                model_index          INTEGER NOT NULL,
                kind                 TEXT NOT NULL,
                target_prompt_tokens INTEGER NOT NULL,
                created_at           REAL NOT NULL,
                PRIMARY KEY (epoch_number, obligation_id)
            );

            CREATE INDEX IF NOT EXISTS idx_planned_canary_obligations_epoch
                ON planned_canary_obligations(epoch_number);

            CREATE TABLE IF NOT EXISTS proof_v3_hard_failures (
                outcome_digest      TEXT PRIMARY KEY,
                source_epoch        INTEGER NOT NULL,
                miner_address       TEXT NOT NULL,
                model_index         INTEGER NOT NULL,
                outcome_json        TEXT NOT NULL,
                created_at          REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_proof_v3_hard_failures_epoch
                ON proof_v3_hard_failures(source_epoch);

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
                incident_review_status TEXT NOT NULL DEFAULT 'clear',
                incident_reviewed_at REAL,
                incident_signal_reason TEXT NOT NULL DEFAULT '',
                incident_signaled_at REAL,
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
                gpu_index            INTEGER NOT NULL DEFAULT 0,
                roster_digest        TEXT NOT NULL DEFAULT '',
                pass_count           INTEGER NOT NULL DEFAULT 0,
                workload_spec        TEXT NOT NULL DEFAULT '{}',
                deadline_s           REAL NOT NULL DEFAULT 0.0,
                transport_grace_s    REAL NOT NULL DEFAULT 0.0,
                payload_deadline_s   REAL NOT NULL DEFAULT 0.0,
                drain_until_ts       REAL NOT NULL DEFAULT 0.0,
                pass0_received_at    REAL,
                final_received_at    REAL,
                final_observed_block INTEGER,
                proof_received_at    REAL,
                pass0_root           TEXT NOT NULL DEFAULT '',
                final_root           TEXT NOT NULL DEFAULT '',
                transcript_root      TEXT NOT NULL DEFAULT '',
                timing_status        TEXT NOT NULL DEFAULT 'pending',
                proof_status         TEXT NOT NULL DEFAULT 'pending',
                proof_verify_ms      REAL,
                verdict              TEXT NOT NULL DEFAULT 'pending',
                failure_reason       TEXT,
                probation_required   INTEGER NOT NULL DEFAULT 0,
                probation_applied_at REAL,
                pass0_artifact       TEXT,
                final_artifact       TEXT,
                proof_artifact_path  TEXT,
                created_at           REAL NOT NULL,
                updated_at           REAL NOT NULL,
                PRIMARY KEY (audit_id, miner_address, model_index, gpu_index)
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
                gpu_index            INTEGER NOT NULL DEFAULT 0,
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
                probation_required   INTEGER NOT NULL DEFAULT 0,
                probation_applied_at REAL,
                pass0_received_at    REAL,
                final_received_at    REAL,
                proof_received_at    REAL,
                slot_created_at      REAL NOT NULL,
                slot_updated_at      REAL NOT NULL,
                archived_at          REAL NOT NULL,
                PRIMARY KEY (audit_id, miner_address, model_index, gpu_index)
            );

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_archived
                ON capacity_audit_history(archived_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_uid
                ON capacity_audit_history(miner_uid, archived_at);

            CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_outcome
                ON capacity_audit_history(verdict, proof_status, archived_at);

            -- Signed mesh GPU rosters, keyed by digest so a re-signed
            -- membership change creates a new row instead of destroying the
            -- document an in-flight audit was scheduled against.
            CREATE TABLE IF NOT EXISTS capacity_rosters (
                slot_id       TEXT NOT NULL,
                roster_epoch  INTEGER NOT NULL,
                roster_digest TEXT NOT NULL,
                roster_json   TEXT NOT NULL,
                signature     TEXT NOT NULL,
                received_at   REAL NOT NULL,
                PRIMARY KEY (slot_id, roster_digest)
            );

            CREATE INDEX IF NOT EXISTS idx_capacity_rosters_slot
                ON capacity_rosters(slot_id, received_at);

            -- Mesh slots whose signed roster could not be learned (route
            -- refused/unreachable and no receipt-embedded copy). Tracks the
            -- first epoch the absence was observed so refusal can be gated
            -- after a grace window instead of exempting the entry forever.
            CREATE TABLE IF NOT EXISTS capacity_roster_probes (
                slot_id             TEXT PRIMARY KEY,
                first_missing_epoch INTEGER NOT NULL,
                last_missing_epoch  INTEGER NOT NULL
            );
        """)
        self._conn.commit()

        self._ensure_column(
            "miner_entries",
            "last_scored_epoch",
            "INTEGER",
        )
        self._ensure_column(
            "miner_entries",
            "probation_source",
            "TEXT",
        )
        self._ensure_column(
            "epoch_close_queue",
            "abandoned",
            "INTEGER NOT NULL DEFAULT 0",
        )
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
        # Existing windows predate automatic cohort anomaly review and must
        # never be retroactively reclassified on upgrade. The production
        # scheduler explicitly opts each new window into pending review.
        self._ensure_column(
            "capacity_audit_windows",
            "incident_review_status",
            "TEXT NOT NULL DEFAULT 'clear'",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "incident_reviewed_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "incident_signal_reason",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            "capacity_audit_windows",
            "incident_signaled_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "proof_verify_ms",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "final_observed_block",
            "INTEGER",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "workload_spec",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "gpu_index",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "capacity_audit_history",
            "gpu_index",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "probation_required",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "capacity_audit_slots",
            "probation_applied_at",
            "REAL",
        )
        self._ensure_column(
            "capacity_audit_history",
            "probation_required",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "capacity_audit_history",
            "probation_applied_at",
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
        self._ensure_column(
            "capacity_audit_slots",
            "roster_digest",
            "TEXT NOT NULL DEFAULT ''",
        )
        # Mesh audits fan one audit_id out to one row per roster GPU ordinal,
        # so gpu_index must join the primary key. Runs after _ensure_column so
        # the rebuilt table copy sees every current column.
        self._migrate_capacity_audit_gpu_index_pk()

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
                "SELECT model_id, endpoint, quant, max_context_len, "
                "probation_entered_epoch, probation_source FROM miner_entries "
                "WHERE address = ? AND model_index = ?",
                (address, model_index),
            ).fetchone()

            if row is None:
                # A brand-new slot for an address that is already serving out
                # probation inherits it.  Otherwise re-registering under a
                # fresh model_index is a free reset: probation costs a few
                # epochs of emission, re-registration costs at most one UID
                # burn, and with a changed endpoint URL it costs nothing.
                # Only genuinely new slots inherit; a long-standing sibling
                # entry is untouched, so an operator running several models is
                # not punished across all of them for one bad slot.
                # Availability-cause probation (a dead box, zero dishonesty
                # evidence) dies with its slot: a fresh registration is the
                # recovery, and the new slot faces normal canaries at once.
                # For-cause probation inherits — re-registering must never
                # reset a proof/integrity/evasion consequence. Legacy and
                # unknown sources read as for_cause (fail closed).
                inherited = self._conn.execute(
                    """SELECT MIN(probation_entered_epoch) AS entered
                       FROM miner_entries
                       WHERE address = ?
                         AND probation_entered_epoch IS NOT NULL
                         AND (probation_source IS NULL
                              OR probation_source NOT LIKE '%availability')""",
                    (address,),
                ).fetchone()
                inherited_epoch = (
                    inherited["entered"] if inherited is not None else None
                )
                self._conn.execute(
                    """INSERT INTO miner_entries (
                        address, model_index, model_id, endpoint, quant,
                        max_context_len, first_seen_epoch, last_seen_epoch,
                        is_active, ema_score, total_epochs, scored_epochs,
                        probation_entered_epoch, probation_consecutive_passes,
                        probation_required_passes, probation_escalation_epochs,
                        probation_source,
                        tee_enabled, tee_platform,
                        gpu_name, gpu_count, vram_gb, compute_capability, gpu_uuids,
                        hotkey_ss58, coldkey_ss58, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0.0, 0, 0,
                              ?, 0, 3, 5, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (address, model_index, model_id, endpoint, quant,
                     max_context_len, epoch, epoch, inherited_epoch,
                     "inherited_for_cause" if inherited_epoch is not None else None,
                     1 if tee_enabled else 0, tee_platform,
                     gpu_name, gpu_count, vram_gb, compute_capability, _gpu_uuids_json,
                     hotkey_ss58, coldkey_ss58, now, now),
                )
                if inherited_epoch is not None:
                    bt.logging.info(
                        f"Probation inherited by new entry {address[:10]} "
                        f"idx={model_index} from epoch {inherited_epoch}"
                    )
            else:
                old_model_id = row["model_id"]
                if old_model_id != model_id:
                    # Model switch: reset the score, keep probation.  Past
                    # performance does not carry to a different model, but a
                    # proof failure is a penalty on the operator, and clearing
                    # it here made switching model_id a free probation reset.
                    # The pass counter restarts so the clean epochs have to be
                    # served on the model now being offered.
                    model_switched = True
                    self._conn.execute(
                        """UPDATE miner_entries SET
                            model_id = ?, endpoint = ?, quant = ?,
                            max_context_len = ?, last_seen_epoch = ?,
                            is_active = 1, ema_score = 0.0,
                            total_epochs = 0, scored_epochs = 0,
                            last_scored_epoch = NULL,
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
                    if (
                        registration_changed
                        and row["probation_entered_epoch"] is not None
                        and str(row["probation_source"] or "").endswith(
                            "availability"
                        )
                    ):
                        # Availability-cause probation records missed
                        # obligations only — a dead or unreachable box,
                        # never dishonesty. A changed registration IS the
                        # recovery, so routability returns immediately;
                        # the very next canaries verify the new serve, and
                        # the EMA damage from the outage stays in place.
                        # For-cause probation never clears here: any
                        # proof, integrity, or evasion consequence serves
                        # its full consecutive-pass exit regardless of
                        # re-registration.
                        self._conn.execute(
                            """UPDATE miner_entries SET
                                probation_entered_epoch = NULL,
                                probation_consecutive_passes = 0,
                                probation_source = NULL,
                                updated_at = ?
                            WHERE address = ? AND model_index = ?""",
                            (now, address, model_index),
                        )
                        bt.logging.info(
                            f"Probation cleared for {address[:10]} "
                            f"idx={model_index}: availability-only cause "
                            f"and the registration changed"
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
                       probation_source = NULL,
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
                       probation_source = NULL,
                       updated_at = ?
                   WHERE address = ?
                     AND probation_entered_epoch IS NOT NULL""",
                (time.time(), address),
            )
            self._conn.commit()
            return cursor.rowcount

    def reset_operational_probation_state(
        self,
        generation: int,
        *,
        effective_epoch: int,
        hard_failure_strikes_json: str,
    ) -> dict:
        """Apply one monotonic, network-coordinated probation reset.

        Scores and historical audit evidence are retained.  Pending capacity
        consequences observed before this reset are consumed so they cannot
        immediately recreate the cleared probation state.
        """

        target = int(generation)
        reset_epoch = int(effective_epoch)
        if target < 0:
            raise ValueError("probation state generation must be non-negative")
        if reset_epoch < 0:
            raise ValueError("probation reset effective epoch must be non-negative")
        now = time.time()
        generation_key = "proof_v3_probation_state_generation_v1"
        with self._lock:
            raw_current = self._raw_get_meta(generation_key)
            current = int(raw_current or 0)
            if target <= current:
                return {
                    "applied": False,
                    "previous_generation": current,
                    "generation": current,
                    "probation_entries_cleared": 0,
                    "capacity_events_consumed": 0,
                }
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                probation = self._conn.execute(
                    """UPDATE miner_entries SET
                           probation_entered_epoch = NULL,
                           probation_consecutive_passes = 0,
                           probation_source = NULL,
                           updated_at = ?
                       WHERE probation_entered_epoch IS NOT NULL""",
                    (now,),
                ).rowcount or 0
                capacity = self._conn.execute(
                    """UPDATE capacity_audit_slots SET
                           probation_applied_at = ?,
                           updated_at = ?
                       WHERE probation_required != 0
                         AND probation_applied_at IS NULL""",
                    (now, now),
                ).rowcount or 0
                for key, value in (
                    (
                        "proof_v3_hard_failure_strikes_v1",
                        str(hard_failure_strikes_json),
                    ),
                    ("proof_v3_probation_recovery_source_epochs_v1", "{}"),
                    (
                        "proof_v3_probation_state_reset_effective_epoch_v1",
                        str(reset_epoch),
                    ),
                    (generation_key, str(target)),
                ):
                    self._conn.execute(
                        """INSERT OR REPLACE INTO validator_meta
                               (key, value) VALUES (?, ?)""",
                        (key, value),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "applied": True,
            "previous_generation": current,
            "generation": target,
            "probation_entries_cleared": int(probation),
            "capacity_events_consumed": int(capacity),
        }

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

    # ── Persistent epoch-close queue ─────────────────────────────────

    @staticmethod
    def _validate_epoch_close_int(value: object, field_name: str) -> int:
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError(f"{field_name} must be a non-negative 63-bit integer")
        return value

    @staticmethod
    def _canonical_epoch_close_context(context: Mapping[str, Any]) -> str:
        if not isinstance(context, Mapping):
            raise ValueError("epoch close context must be an object")
        try:
            canonical = json.dumps(
                dict(context),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("epoch close context must be canonical JSON") from exc
        if len(canonical.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("epoch close context exceeds 16 MiB")
        return canonical

    def schedule_epoch_close(
        self,
        *,
        epoch_number: int,
        start_block: int,
        close_block: int,
    ) -> dict[str, Any]:
        """Create the immutable collecting row for one epoch.

        Re-scheduling the same boundaries is idempotent.  A conflicting
        boundary is rejected so a retry cannot silently move the grace gate.
        """

        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        start_block = self._validate_epoch_close_int(start_block, "start_block")
        close_block = self._validate_epoch_close_int(close_block, "close_block")
        if close_block < start_block:
            raise ValueError("close_block must not precede start_block")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                (epoch_number,),
            ).fetchone()
            if row is not None:
                if (
                    int(row["start_block"]) != start_block
                    or int(row["close_block"]) != close_block
                ):
                    raise ValueError("conflicting epoch-close boundary")
                return dict(row)
            self._conn.execute(
                """INSERT INTO epoch_close_queue (
                       epoch_number, start_block, close_block, status,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, 'collecting', ?, ?)""",
                (epoch_number, start_block, close_block, now, now),
            )
            self._conn.commit()
            return dict(
                self._conn.execute(
                    "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                    (epoch_number,),
                ).fetchone()
            )

    def seal_epoch_close(
        self,
        *,
        epoch_number: int,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal a collecting epoch with its immutable close context."""

        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        canonical = self._canonical_epoch_close_context(context)
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                (epoch_number,),
            ).fetchone()
            if row is None:
                raise ValueError("epoch close was not scheduled")
            existing = row["context_json"]
            if existing is not None and str(existing) != canonical:
                raise ValueError("conflicting sealed epoch-close context")
            if row["status"] == "completed":
                return dict(row)
            self._conn.execute(
                """UPDATE epoch_close_queue SET
                       context_json = ?, status = 'pending', updated_at = ?
                   WHERE epoch_number = ?""",
                (canonical, now, epoch_number),
            )
            self._conn.commit()
            return dict(
                self._conn.execute(
                    "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                    (epoch_number,),
                ).fetchone()
            )

    def get_epoch_close(self, epoch_number: int) -> Optional[dict[str, Any]]:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                (epoch_number,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_pending_epoch_closes(self) -> List[dict[str, Any]]:
        """Return sealed unfinished closes in strict epoch order."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM epoch_close_queue
                   WHERE status IN ('pending', 'running')
                     AND context_json IS NOT NULL
                   ORDER BY epoch_number ASC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def abandon_collecting_epoch_closes(
        self,
        *,
        through_epoch: int,
        reason: str,
    ) -> int:
        """Terminally neutralize unsealed rows whose RAM context was lost.

        A collecting row has no immutable miner/accounting snapshot and must
        never be reconstructed from a later epoch.  Marking it abandoned is a
        fail-neutral audit record; it is intentionally not treated as a scored
        or successfully closed epoch.
        """

        through_epoch = self._validate_epoch_close_int(
            through_epoch, "through_epoch"
        )
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE epoch_close_queue SET
                       status = 'completed', abandoned = 1,
                       last_error = ?, updated_at = ?,
                       completed_at = COALESCE(completed_at, ?)
                   WHERE status = 'collecting' AND epoch_number <= ?""",
                (str(reason)[:1000], now, now, through_epoch),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def mark_epoch_close_running(self, epoch_number: int) -> dict[str, Any]:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                (epoch_number,),
            ).fetchone()
            if row is None or row["context_json"] is None:
                raise ValueError("epoch close is not sealed")
            if row["status"] == "completed":
                return dict(row)
            self._conn.execute(
                """UPDATE epoch_close_queue SET
                       status = 'running', attempt_count = attempt_count + 1,
                       last_error = '', updated_at = ?
                   WHERE epoch_number = ?""",
                (now, epoch_number),
            )
            self._conn.commit()
            return dict(
                self._conn.execute(
                    "SELECT * FROM epoch_close_queue WHERE epoch_number = ?",
                    (epoch_number,),
                ).fetchone()
            )

    def mark_epoch_close_failed(self, epoch_number: int, error: str) -> None:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        with self._lock:
            self._conn.execute(
                """UPDATE epoch_close_queue SET
                       status = CASE
                           WHEN status = 'completed' THEN status ELSE 'pending' END,
                       last_error = CASE
                           WHEN status = 'completed' THEN last_error ELSE ? END,
                       updated_at = ?
                   WHERE epoch_number = ?""",
                (str(error)[:1000], time.time(), epoch_number),
            )
            self._conn.commit()

    def mark_epoch_close_completed(self, epoch_number: int) -> None:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT context_json FROM epoch_close_queue WHERE epoch_number = ?",
                (epoch_number,),
            ).fetchone()
            if row is None or row["context_json"] is None:
                raise ValueError("epoch close is not sealed")
            try:
                context = json.loads(str(row["context_json"]))
                miners = context["miners"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("sealed epoch-close context has no miner list") from exc
            if not isinstance(miners, list):
                raise ValueError("sealed epoch-close context has no miner list")
            expected_slots = set()
            for miner in miners:
                try:
                    key = (
                        str(miner["address"]).lower(),
                        int(miner["model_index"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("sealed epoch-close miner key is invalid") from exc
                if key in expected_slots:
                    raise ValueError("sealed epoch-close context has duplicate slots")
                expected_slots.add(key)
            completed_slots = {
                (str(slot["miner_address"]), int(slot["model_index"]))
                for slot in self._conn.execute(
                    """SELECT miner_address, model_index
                       FROM epoch_close_slots
                       WHERE epoch_number = ? AND status = 'completed'""",
                    (epoch_number,),
                ).fetchall()
            }
            missing = expected_slots - completed_slots
            if missing:
                raise RuntimeError(
                    f"epoch close has {len(missing)} unfinished slot(s)"
                )
            self._conn.execute(
                """UPDATE epoch_close_queue SET
                       status = 'completed', last_error = '',
                       updated_at = ?, completed_at = COALESCE(completed_at, ?)
                   WHERE epoch_number = ?""",
                (now, now, epoch_number),
            )
            self._conn.commit()

    def get_epoch_close_slot(
        self,
        epoch_number: int,
        address: str,
        model_index: int,
    ) -> Optional[dict[str, Any]]:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        address = str(address).lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM epoch_close_slots
                   WHERE epoch_number = ? AND miner_address = ?
                     AND model_index = ?""",
                (epoch_number, address, int(model_index)),
            ).fetchone()
        return dict(row) if row is not None else None

    def mark_epoch_close_slot_completed(
        self,
        epoch_number: int,
        address: str,
        model_index: int,
    ) -> None:
        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        address = str(address).lower()
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO epoch_close_slots (
                       epoch_number, miner_address, model_index,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (epoch_number, address, int(model_index), now, now),
            )
            self._conn.execute(
                """UPDATE epoch_close_slots SET
                       status = 'completed', updated_at = ?,
                       completed_at = COALESCE(completed_at, ?)
                   WHERE epoch_number = ? AND miner_address = ?
                     AND model_index = ?""",
                (now, now, epoch_number, address, int(model_index)),
            )
            self._conn.commit()

    # ── Scoring ──────────────────────────────────────────────────────

    def save_score(
        self,
        address: str,
        model_index: int,
        ema_score: float,
        total_epochs: int,
        scored_epochs: int,
        *,
        last_scored_epoch: Optional[int] = None,
        score_provenance: Optional[Mapping[str, Any]] = None,
        epoch_close_number: Optional[int] = None,
        epoch_score: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Persist EMA score state for a miner-model entry.

        ``last_scored_epoch`` is supplied only when this write follows an
        actual epoch scoring result.  Penalties, routing/offline zeroing, and
        other score mutations omit it so they preserve the last real scoring
        evidence.  Mesh scores additionally supply ``score_provenance`` so the
        completed EMA sample is bound to the exact signed topology that was
        verified.  The slot EMA can carry across later mesh generations, but
        the sample record cannot silently follow it.
        """
        address = address.lower()
        normalized_last_scored_epoch: Optional[int] = None
        if last_scored_epoch is not None:
            if isinstance(last_scored_epoch, bool):
                raise ValueError("last_scored_epoch must be a non-negative integer")
            normalized_last_scored_epoch = int(last_scored_epoch)
            if not 0 <= normalized_last_scored_epoch < 2**63:
                raise ValueError("last_scored_epoch must be a non-negative integer")
            if int(scored_epochs) <= 0:
                raise ValueError(
                    "last_scored_epoch requires at least one scored epoch"
                )
            try:
                completed_ema = float(ema_score)
            except (TypeError, ValueError) as exc:
                raise ValueError("completed EMA score must be finite") from exc
            if not float("-inf") < completed_ema < float("inf"):
                raise ValueError("completed EMA score must be finite")
        if score_provenance is not None and normalized_last_scored_epoch is None:
            raise ValueError("score_provenance requires last_scored_epoch")

        normalized_close_epoch: Optional[int] = None
        normalized_epoch_score: Optional[float] = None
        if epoch_close_number is not None:
            if type(epoch_close_number) is not int or not 0 <= epoch_close_number < 2**63:
                raise ValueError("epoch_close_number must be a non-negative integer")
            normalized_close_epoch = epoch_close_number
            if epoch_score is not None:
                try:
                    normalized_epoch_score = float(epoch_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError("epoch_score must be finite or None") from exc
                if not math.isfinite(normalized_epoch_score):
                    raise ValueError("epoch_score must be finite or None")
        elif epoch_score is not None:
            raise ValueError("epoch_score requires epoch_close_number")

        normalized_provenance: Optional[dict[str, Any]] = None
        if score_provenance is not None:
            required = {
                "chain_id",
                "netuid",
                "coordinator_address",
                "model_index",
                "model_id",
                "mesh_id",
                "verification_snapshot_hash",
                "snapshot_generation",
                "score_epoch",
            }
            if set(score_provenance) != required:
                raise ValueError(
                    "score_provenance must contain the exact mesh score binding"
                )
            chain_id = score_provenance.get("chain_id")
            netuid = score_provenance.get("netuid")
            provenance_model_index = score_provenance.get("model_index")
            snapshot_generation = score_provenance.get("snapshot_generation")
            score_epoch = score_provenance.get("score_epoch")
            if type(chain_id) is not int or not 1 <= chain_id < 2**63:
                raise ValueError("score_provenance chain_id is invalid")
            if type(netuid) is not int or not 0 <= netuid <= 65_535:
                raise ValueError("score_provenance netuid is invalid")
            if (
                type(provenance_model_index) is not int
                or provenance_model_index < 0
                or provenance_model_index != int(model_index)
            ):
                raise ValueError("score_provenance model_index does not match score slot")
            if (
                type(score_epoch) is not int
                or score_epoch != normalized_last_scored_epoch
            ):
                raise ValueError("score_provenance score_epoch does not match score")
            if (
                type(snapshot_generation) is not int
                or not 1 <= snapshot_generation < 2**63
            ):
                raise ValueError("score_provenance snapshot_generation is invalid")
            if any(
                type(score_provenance.get(field)) is not str
                for field in (
                    "coordinator_address",
                    "model_id",
                    "mesh_id",
                    "verification_snapshot_hash",
                )
            ):
                raise ValueError("score_provenance string fields are invalid")
            provenance_address = score_provenance["coordinator_address"].lower()
            if provenance_address != address:
                raise ValueError(
                    "score_provenance coordinator_address does not match score slot"
                )
            model_id = score_provenance["model_id"]
            mesh_id = score_provenance["mesh_id"]
            snapshot_hash = score_provenance["verification_snapshot_hash"]
            if not model_id or not mesh_id:
                raise ValueError("score_provenance model_id and mesh_id are required")
            if len(snapshot_hash) != 64:
                raise ValueError(
                    "score_provenance verification_snapshot_hash is invalid"
                )
            try:
                bytes.fromhex(snapshot_hash)
            except ValueError as exc:
                raise ValueError(
                    "score_provenance verification_snapshot_hash is invalid"
                ) from exc
            normalized_provenance = {
                "chain_id": chain_id,
                "netuid": netuid,
                "coordinator_address": provenance_address,
                "model_index": provenance_model_index,
                "model_id": model_id,
                "mesh_id": mesh_id,
                "snapshot_hash": snapshot_hash.lower(),
                "snapshot_generation": snapshot_generation,
                "score_epoch": score_epoch,
            }
        with self._lock, self._conn:
            entry = self._conn.execute(
                """SELECT model_id FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, model_index),
            ).fetchone()
            if (
                normalized_provenance is not None
                and (
                    entry is None
                    or str(entry["model_id"]) != normalized_provenance["model_id"]
                )
            ):
                raise ValueError(
                    "score_provenance model_id does not match registered score slot"
                )

            if normalized_close_epoch is not None:
                now = time.time()
                self._conn.execute(
                    """INSERT OR IGNORE INTO epoch_close_slots (
                           epoch_number, miner_address, model_index,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (normalized_close_epoch, address, model_index, now, now),
                )
                checkpoint = self._conn.execute(
                    """SELECT score_applied, epoch_score, epoch_score_is_null,
                              ema_score, total_epochs, scored_epochs,
                              last_scored_epoch
                       FROM epoch_close_slots
                       WHERE epoch_number = ? AND miner_address = ?
                         AND model_index = ?""",
                    (normalized_close_epoch, address, model_index),
                ).fetchone()
                if checkpoint is not None and int(checkpoint["score_applied"]):
                    existing_epoch_score = (
                        None
                        if int(checkpoint["epoch_score_is_null"])
                        else float(checkpoint["epoch_score"])
                    )
                    incoming_state = (
                        normalized_epoch_score,
                        float(ema_score),
                        int(total_epochs),
                        int(scored_epochs),
                        normalized_last_scored_epoch,
                    )
                    existing_state = (
                        existing_epoch_score,
                        float(checkpoint["ema_score"]),
                        int(checkpoint["total_epochs"]),
                        int(checkpoint["scored_epochs"]),
                        checkpoint["last_scored_epoch"],
                    )
                    if existing_state != incoming_state:
                        self._conn.rollback()
                        raise ValueError(
                            "conflicting replay of an applied epoch-close score"
                        )
                    self._conn.commit()
                    return {
                        "applied": False,
                        "epoch_score": existing_epoch_score,
                        "ema_score": existing_state[1],
                        "total_epochs": existing_state[2],
                        "scored_epochs": existing_state[3],
                        "last_scored_epoch": existing_state[4],
                    }

            if normalized_last_scored_epoch is not None and entry is not None:
                existing_sample = self._conn.execute(
                    """SELECT model_id, ema_score, chain_id, netuid, mesh_id,
                              snapshot_hash, snapshot_generation
                       FROM score_samples
                       WHERE coordinator_address = ? AND model_index = ?
                         AND score_epoch = ?""",
                    (address, model_index, normalized_last_scored_epoch),
                ).fetchone()
                incoming_identity = (
                    (
                        normalized_provenance["model_id"],
                        normalized_provenance["chain_id"],
                        normalized_provenance["netuid"],
                        normalized_provenance["mesh_id"],
                        normalized_provenance["snapshot_hash"],
                        normalized_provenance["snapshot_generation"],
                    )
                    if normalized_provenance is not None
                    else (
                        str(entry["model_id"]) if entry is not None else "",
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                )
                if existing_sample is not None:
                    existing_identity = (
                        str(existing_sample["model_id"]),
                        existing_sample["chain_id"],
                        existing_sample["netuid"],
                        existing_sample["mesh_id"],
                        existing_sample["snapshot_hash"],
                        existing_sample["snapshot_generation"],
                    )
                    if existing_identity != incoming_identity:
                        if normalized_close_epoch is not None:
                            self._conn.rollback()
                        raise ValueError(
                            "conflicting score provenance for completed epoch"
                        )
                    if float(existing_sample["ema_score"]) != completed_ema:
                        if normalized_close_epoch is not None:
                            self._conn.rollback()
                        raise ValueError(
                            "conflicting completed EMA for score slot and epoch"
                        )
                self._conn.execute(
                    """INSERT INTO score_samples (
                           coordinator_address, model_index, score_epoch,
                           model_id, ema_score, chain_id, netuid, mesh_id,
                           snapshot_hash, snapshot_generation, recorded_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (
                           coordinator_address, model_index, score_epoch
                       ) DO NOTHING""",
                    (
                        address,
                        model_index,
                        normalized_last_scored_epoch,
                        incoming_identity[0],
                        completed_ema,
                        incoming_identity[1],
                        incoming_identity[2],
                        incoming_identity[3],
                        incoming_identity[4],
                        incoming_identity[5],
                        time.time(),
                    ),
                )
            self._conn.execute(
                """UPDATE miner_entries SET
                    ema_score = ?, total_epochs = ?, scored_epochs = ?,
                    last_scored_epoch = CASE
                        WHEN ? IS NULL THEN last_scored_epoch
                        WHEN last_scored_epoch IS NULL
                          OR ? > last_scored_epoch THEN ?
                        ELSE last_scored_epoch
                    END,
                    updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (ema_score, total_epochs, scored_epochs,
                 normalized_last_scored_epoch,
                 normalized_last_scored_epoch,
                 normalized_last_scored_epoch,
                 time.time(),
                 address, model_index),
            )
            if normalized_close_epoch is not None:
                self._conn.execute(
                    """UPDATE epoch_close_slots SET
                           score_applied = 1,
                           epoch_score = ?,
                           epoch_score_is_null = ?,
                           ema_score = ?,
                           total_epochs = ?,
                           scored_epochs = ?,
                           last_scored_epoch = ?,
                           updated_at = ?
                       WHERE epoch_number = ? AND miner_address = ?
                         AND model_index = ?""",
                    (
                        normalized_epoch_score,
                        1 if normalized_epoch_score is None else 0,
                        float(ema_score),
                        int(total_epochs),
                        int(scored_epochs),
                        normalized_last_scored_epoch,
                        time.time(),
                        normalized_close_epoch,
                        address,
                        model_index,
                    ),
                )
            try:
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            if normalized_close_epoch is not None:
                return {
                    "applied": True,
                    "epoch_score": normalized_epoch_score,
                    "ema_score": float(ema_score),
                    "total_epochs": int(total_epochs),
                    "scored_epochs": int(scored_epochs),
                    "last_scored_epoch": normalized_last_scored_epoch,
                }
            return None

    def apply_epoch_close_slot_event(
        self,
        *,
        epoch_number: int,
        address: str,
        model_index: int,
        event_kind: str,
        action: str,
    ) -> dict[str, Any]:
        """Atomically apply one deterministic close-side mutation once.

        ``event_kind`` identifies the semantic occurrence inside the epoch
        (for example ``busy_skip_penalty``).  ``action`` is deliberately a
        small closed set so replay cannot smuggle arbitrary SQL state:

        - ``probation_failure`` enters probation or resets its pass counter;
        - ``probation_pass`` records exactly one clean pass;
        - ``proof_penalty`` enters/resets probation and halves the EMA;
        - ``binding_violation`` enters/resets probation and zeroes the EMA;
        - ``score_zero`` sets the live slot EMA to zero.

        ``binding_violation`` is for a coordinator that broke a commitment it
        had already made rather than merely failing a check.  Halving assumes
        a single failure is recoverable, which is right for a miner having a
        bad epoch and wrong for one that shipped a shrunken challenge universe
        or a weight root that does not match its own receipt.

        The event row and miner entry update commit together.  Replays return
        the authoritative current entry without applying the mutation again.
        """

        epoch_number = self._validate_epoch_close_int(epoch_number, "epoch_number")
        address = str(address).lower()
        if not event_kind or len(event_kind) > 128:
            raise ValueError("event_kind must be 1..128 characters")
        if action not in {
            "probation_failure",
            "probation_pass",
            "proof_penalty",
            "binding_violation",
            "score_zero",
        }:
            raise ValueError("unsupported epoch-close slot action")
        payload = json.dumps(
            {"action": action}, sort_keys=True, separators=(",", ":")
        )
        now = time.time()

        with self._lock:
            existing_event = self._conn.execute(
                """SELECT payload_json FROM epoch_close_slot_events
                   WHERE epoch_number = ? AND miner_address = ?
                     AND model_index = ? AND event_kind = ?""",
                (epoch_number, address, int(model_index), event_kind),
            ).fetchone()
            row = self._conn.execute(
                """SELECT ema_score, total_epochs, scored_epochs,
                          last_scored_epoch, probation_entered_epoch,
                          probation_consecutive_passes,
                          probation_required_passes
                   FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (address, int(model_index)),
            ).fetchone()
            if row is None:
                return {
                    "applied": False,
                    "entry_found": False,
                    "ema_score": None,
                    "probation_entered_epoch": None,
                    "probation_consecutive_passes": 0,
                }
            if existing_event is not None:
                if str(existing_event["payload_json"]) != payload:
                    raise ValueError("conflicting epoch-close event replay")
                return {
                    "applied": False,
                    "entry_found": True,
                    "ema_score": float(row["ema_score"]),
                    "total_epochs": int(row["total_epochs"]),
                    "scored_epochs": int(row["scored_epochs"]),
                    "last_scored_epoch": row["last_scored_epoch"],
                    "probation_entered_epoch": row["probation_entered_epoch"],
                    "probation_consecutive_passes": int(
                        row["probation_consecutive_passes"]
                    ),
                }

            ema_score = float(row["ema_score"])
            entered_epoch = row["probation_entered_epoch"]
            consecutive_passes = int(row["probation_consecutive_passes"])
            if action in {
                "probation_failure", "proof_penalty", "binding_violation",
            }:
                if entered_epoch is None:
                    entered_epoch = epoch_number
                consecutive_passes = 0
            elif action == "probation_pass" and entered_epoch is not None:
                consecutive_passes += 1
                if consecutive_passes >= int(row["probation_required_passes"]):
                    entered_epoch = None
                    consecutive_passes = 0
            if action == "proof_penalty":
                ema_score *= 0.5
            elif action in {"binding_violation", "score_zero"}:
                ema_score = 0.0

            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO epoch_close_slots (
                           epoch_number, miner_address, model_index,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (epoch_number, address, int(model_index), now, now),
                )
                self._conn.execute(
                    """UPDATE miner_entries SET
                           ema_score = ?, probation_entered_epoch = ?,
                           probation_consecutive_passes = ?, updated_at = ?
                       WHERE address = ? AND model_index = ?""",
                    (
                        ema_score,
                        entered_epoch,
                        consecutive_passes,
                        now,
                        address,
                        int(model_index),
                    ),
                )
                self._conn.execute(
                    """INSERT INTO epoch_close_slot_events (
                           epoch_number, miner_address, model_index,
                           event_kind, payload_json, applied_at
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        epoch_number,
                        address,
                        int(model_index),
                        event_kind,
                        payload,
                        now,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            return {
                "applied": True,
                "entry_found": True,
                "ema_score": ema_score,
                "total_epochs": int(row["total_epochs"]),
                "scored_epochs": int(row["scored_epochs"]),
                "last_scored_epoch": row["last_scored_epoch"],
                "probation_entered_epoch": entered_epoch,
                "probation_consecutive_passes": consecutive_passes,
            }

    def _get_latest_score_samples(self) -> Dict[Tuple[str, int], dict]:
        """Return the sample selected by each slot's scoring high-water mark."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT s.coordinator_address, s.model_index,
                          s.score_epoch, s.model_id, s.ema_score,
                          s.chain_id, s.netuid, s.mesh_id, s.snapshot_hash,
                          s.snapshot_generation
                   FROM score_samples s
                   JOIN miner_entries m
                     ON m.address = s.coordinator_address
                    AND m.model_index = s.model_index
                    AND m.last_scored_epoch = s.score_epoch"""
            ).fetchall()
        return {
            (str(row["coordinator_address"]), int(row["model_index"])): dict(row)
            for row in rows
        }

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

    def zero_ema(self, address: str, model_index: int) -> None:
        """Zero the EMA score after a broken commitment."""
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    ema_score = 0.0, updated_at = ?
                WHERE address = ? AND model_index = ?""",
                (time.time(), address, model_index),
            )
            self._conn.commit()

    def load_all_scores(self) -> Dict[Tuple[str, int], dict]:
        """Load all entries with scoring data.

        Returns:
            Dict mapping ``(address, model_index)`` to a dict with keys
            ``model_id``, ``ema_score``, ``total_epochs``, ``scored_epochs``,
            and ``last_scored_epoch``.
        """
        with self._lock:
            cursor = self._conn.execute(
                """SELECT address, model_index, model_id,
                          ema_score, total_epochs, scored_epochs,
                          last_scored_epoch
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
                    "last_scored_epoch": row["last_scored_epoch"],
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
        cause: str = "for_cause",
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
                # (the validator is doing its job).  Severity only ratchets
                # up: a for_cause failure on an availability probation
                # upgrades the source; it never downgrades.
                if cause == "for_cause":
                    self._conn.execute(
                        """UPDATE miner_entries SET
                            probation_consecutive_passes = 0,
                            probation_source = 'earned_for_cause',
                            updated_at = ?
                        WHERE address = ? AND model_index = ?""",
                        (time.time(), address, model_index),
                    )
                else:
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
                        probation_source = ?,
                        updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (
                        epoch,
                        "earned_availability"
                        if cause == "availability"
                        else "earned_for_cause",
                        time.time(),
                        address,
                        model_index,
                    ),
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
                        probation_source = NULL,
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

    def ratchet_probation_source_for_cause(
        self, address: str, model_index: int
    ) -> None:
        """Upgrade an active availability probation source to for_cause.

        Up only, clocks untouched: a for_cause consequence during an
        availability probation must disqualify the row from the
        availability re-registration clear without restarting the
        probation itself. No-op for rows already for_cause, cleared rows,
        and legacy NULL sources (NULL already reads for_cause,
        fail-closed).
        """
        address = address.lower()
        with self._lock:
            self._conn.execute(
                """UPDATE miner_entries SET
                    probation_source = 'earned_for_cause', updated_at = ?
                WHERE address = ? AND model_index = ?
                  AND probation_entered_epoch IS NOT NULL
                  AND probation_source LIKE '%availability'""",
                (time.time(), address, model_index),
            )
            self._conn.commit()

    def apply_proxy_proof_failure_event(
        self,
        *,
        event_id: str,
        address: str,
        model_index: int,
        epoch_number: int,
        event_timestamp: int,
    ) -> dict[str, Any]:
        """Atomically apply one proxy proof-failure event at most once.

        Returns the authoritative post-transaction EMA and probation state so
        the validator can reconcile its in-memory scorer/tracker after a crash
        without repeating the financial mutation.
        """

        normalized_address = str(address).lower()
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                """SELECT event_id FROM proxy_proof_failure_events
                   WHERE event_id = ?""",
                (event_id,),
            ).fetchone()
            row = self._conn.execute(
                """SELECT ema_score, probation_entered_epoch,
                          probation_consecutive_passes
                   FROM miner_entries
                   WHERE address = ? AND model_index = ?""",
                (normalized_address, model_index),
            ).fetchone()
            if row is None:
                return {
                    "applied": False,
                    "entry_found": False,
                    "ema_score": None,
                    "probation_entered_epoch": None,
                    "probation_consecutive_passes": 0,
                }
            if existing is not None:
                return {
                    "applied": False,
                    "entry_found": True,
                    "ema_score": float(row["ema_score"]),
                    "probation_entered_epoch": row[
                        "probation_entered_epoch"
                    ],
                    "probation_consecutive_passes": int(
                        row["probation_consecutive_passes"]
                    ),
                }

            entered_epoch = row["probation_entered_epoch"]
            if entered_epoch is None:
                entered_epoch = int(epoch_number)
            new_ema = float(row["ema_score"]) * 0.5
            try:
                self._conn.execute(
                    """UPDATE miner_entries SET
                        probation_entered_epoch = ?,
                        probation_consecutive_passes = 0,
                        ema_score = ?,
                        updated_at = ?
                    WHERE address = ? AND model_index = ?""",
                    (
                        entered_epoch,
                        new_ema,
                        now,
                        normalized_address,
                        model_index,
                    ),
                )
                self._conn.execute(
                    """INSERT INTO proxy_proof_failure_events (
                        event_id,
                        miner_address,
                        model_index,
                        epoch_number,
                        event_timestamp,
                        applied_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        event_id,
                        normalized_address,
                        model_index,
                        epoch_number,
                        event_timestamp,
                        now,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return {
                "applied": True,
                "entry_found": True,
                "ema_score": new_ema,
                "probation_entered_epoch": entered_epoch,
                "probation_consecutive_passes": 0,
            }

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

    _CAPACITY_AUDIT_PK_INDEX_DDL = """
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_slot
            ON capacity_audit_slots(miner_address, model_index, created_at);
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_verdict
            ON capacity_audit_slots(verdict, created_at);
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_drains
            ON capacity_audit_slots(drain_until_ts, verdict);
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_archived
            ON capacity_audit_history(archived_at);
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_uid
            ON capacity_audit_history(miner_uid, archived_at);
        CREATE INDEX IF NOT EXISTS idx_capacity_audit_history_outcome
            ON capacity_audit_history(verdict, proof_status, archived_at);
    """

    def _migrate_capacity_audit_gpu_index_pk(self) -> None:
        """One-time PK rebuild adding gpu_index to the audit slot keys.

        Mesh audits store one row per roster GPU ordinal under a shared
        audit_id, which the historical ``(audit_id, miner_address,
        model_index)`` primary key cannot hold. SQLite cannot alter a primary
        key in place, so tables created before this key shape are rebuilt via
        copy-and-rename; existing rows keep their gpu_index 0. Idempotent:
        tables already keyed on gpu_index are left untouched.
        """
        rebuilt = False
        for table in ("capacity_audit_slots", "capacity_audit_history"):
            info = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            pk_cols = [row["name"] for row in info if int(row["pk"] or 0) > 0]
            if not pk_cols or "gpu_index" in pk_cols:
                continue
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            sql = str(row["sql"] if row is not None else "")
            new_sql, pk_hits = re.subn(
                r"PRIMARY KEY\s*\(\s*audit_id\s*,\s*miner_address\s*,\s*model_index\s*\)",
                "PRIMARY KEY (audit_id, miner_address, model_index, gpu_index)",
                sql,
                count=1,
            )
            new_sql, name_hits = re.subn(
                rf"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?\"?{table}\"?",
                f"CREATE TABLE {table}_pkmig",
                new_sql,
                count=1,
            )
            if pk_hits != 1 or name_hits != 1:
                # Fail-safe: an unexpected DDL shape keeps the old key rather
                # than risking a lossy rebuild.
                bt.logging.warning(
                    f"{table}: gpu_index PK migration skipped (unrecognized DDL)"
                )
                continue
            columns = ", ".join(str(r["name"]) for r in info)
            self._conn.execute(new_sql)
            self._conn.execute(
                f"INSERT INTO {table}_pkmig ({columns}) SELECT {columns} FROM {table}"
            )
            self._conn.execute(f"DROP TABLE {table}")
            self._conn.execute(f"ALTER TABLE {table}_pkmig RENAME TO {table}")
            rebuilt = True
        if rebuilt:
            self._conn.executescript(self._CAPACITY_AUDIT_PK_INDEX_DDL)
            self._conn.commit()
            bt.logging.info(
                "capacity audit tables rebuilt with per-GPU primary key"
            )

    # ── Mesh verification snapshots ─────────────────────────────────

    @staticmethod
    def _normalize_mesh_snapshot_address(address: str) -> str:
        """Return a canonical EVM coordinator address for snapshot keys."""

        if not isinstance(address, str):
            raise ValueError("coordinator_address must be a string")
        normalized = address.lower()
        if len(normalized) != 42 or not normalized.startswith("0x"):
            raise ValueError("coordinator_address must be a 20-byte EVM address")
        try:
            int(normalized[2:], 16)
        except ValueError as exc:
            raise ValueError(
                "coordinator_address must be a 20-byte EVM address"
            ) from exc
        return normalized

    @staticmethod
    def _validate_mesh_snapshot_key_int(
        value: int,
        *,
        field_name: str,
        minimum: int,
    ) -> int:
        """Validate an integer used in the immutable snapshot primary key."""

        if type(value) is not int or value < minimum or value >= 2**63:
            relation = "positive" if minimum == 1 else "non-negative"
            raise ValueError(f"{field_name} must be a {relation} 63-bit integer")
        return value

    @staticmethod
    def _normalize_mesh_snapshot_hash(snapshot_hash: str) -> str:
        """Return a canonical lowercase 32-byte snapshot digest."""

        if not isinstance(snapshot_hash, str):
            raise ValueError("snapshot_hash must be a string")
        normalized = snapshot_hash.lower()
        if normalized.startswith("0x"):
            normalized = normalized[2:]
        if len(normalized) != 64:
            raise ValueError("snapshot_hash must be a 32-byte hex digest")
        try:
            int(normalized, 16)
        except ValueError as exc:
            raise ValueError("snapshot_hash must be a 32-byte hex digest") from exc
        return normalized

    @staticmethod
    def _canonicalize_mesh_snapshot_json(
        snapshot_json: str | Mapping[str, Any],
    ) -> tuple[Any, str]:
        """Parse and canonically serialize a signed endpoint-free snapshot."""

        if isinstance(snapshot_json, str):
            try:
                payload = json.loads(snapshot_json)
            except (TypeError, ValueError) as exc:
                raise ValueError("snapshot_json must contain valid JSON") from exc
        elif isinstance(snapshot_json, Mapping):
            payload = dict(snapshot_json)
        else:
            raise ValueError("snapshot_json must be a JSON object or object string")

        if not isinstance(payload, dict):
            raise ValueError("snapshot_json must contain a JSON object")

        # Parse the closed protocol schema at the persistence boundary.  This
        # re-runs the endpoint privacy guard and prevents malformed or unsigned
        # envelopes from entering the validator's trusted cache.
        from verallm.mesh.types import canonical_json_bytes
        from verallm.mesh.verification_snapshot import MeshVerificationSnapshot

        snapshot = MeshVerificationSnapshot.from_dict(payload)
        snapshot.validate(require_signature=True)
        canonical = canonical_json_bytes(snapshot.to_dict()).decode("utf-8")
        return snapshot, canonical

    def upsert_mesh_verification_snapshot(
        self,
        *,
        coordinator_address: str,
        model_index: int,
        epoch: int,
        generation: int,
        snapshot_hash: str,
        snapshot_json: str | Mapping[str, Any],
        fetched_at: float | None = None,
        allow_mesh_change: bool = False,
    ) -> bool:
        """Pin one already-authenticated mesh verification snapshot.

        Snapshot generations are monotonic within one coordinator/model/epoch
        tuple. A lower generation is a rollback and a changed hash or payload
        at the same generation is an equivocation, so both are rejected. A
        higher generation is inserted as a new row; earlier rows are retained
        unchanged for audit.

        ``allow_mesh_change`` is reserved for validator-INITIATED re-pins
        (restart bootstrap, refusal re-pin): the validator is explicitly
        fetching and signing off on a new mesh lineage, so a differing
        ``mesh_id`` supersedes the dead lineage's rows for this epoch instead
        of being rejected as an unsolicited mid-epoch swap.

        The caller remains responsible for verifying the coordinator signature
        and its chain registration before this persistence boundary.

        Returns:
            ``True`` when a new generation was inserted and ``False`` when the
            exact snapshot was already pinned.
        """

        address = self._normalize_mesh_snapshot_address(coordinator_address)
        model_index = self._validate_mesh_snapshot_key_int(
            model_index,
            field_name="model_index",
            minimum=0,
        )
        epoch = self._validate_mesh_snapshot_key_int(
            epoch,
            field_name="epoch",
            minimum=0,
        )
        generation = self._validate_mesh_snapshot_key_int(
            generation,
            field_name="generation",
            minimum=1,
        )
        normalized_hash = self._normalize_mesh_snapshot_hash(snapshot_hash)
        snapshot, canonical_json = self._canonicalize_mesh_snapshot_json(snapshot_json)

        if snapshot.epoch != epoch:
            raise ValueError("snapshot epoch does not match the cache key")
        if snapshot.generation != generation:
            raise ValueError("snapshot generation does not match the cache key")
        payload_expiry = snapshot.expires_at_unix
        if snapshot.coordinator.coordinator_evm_address != address:
            raise ValueError("snapshot coordinator address does not match the cache key")
        if snapshot.coordinator.model_index != model_index:
            raise ValueError("snapshot model_index does not match the cache key")
        if snapshot.snapshot_hash_hex() != normalized_hash:
            raise ValueError("snapshot_hash does not match the snapshot payload")

        if fetched_at is None:
            fetched_at = time.time()
        if not isinstance(fetched_at, (int, float)) or isinstance(fetched_at, bool):
            raise ValueError("fetched_at must be a Unix timestamp")
        fetched_at = float(fetched_at)
        if fetched_at <= 0 or fetched_at == float("inf") or fetched_at != fetched_at:
            raise ValueError("fetched_at must be a finite positive Unix timestamp")
        if fetched_at >= payload_expiry:
            raise ValueError("cannot pin an already-expired mesh verification snapshot")

        with self._lock:
            try:
                # Reserve the writer before checking the high-water mark so
                # separate ValidatorStateDB instances cannot race a rollback.
                self._conn.execute("BEGIN IMMEDIATE")
                latest = self._conn.execute(
                    """SELECT generation, snapshot_hash, snapshot_json,
                              fetched_at, expires_at_unix
                       FROM mesh_verification_snapshots
                       WHERE coordinator_address = ?
                         AND model_index = ?
                         AND epoch_number = ?
                       ORDER BY generation DESC
                       LIMIT 1""",
                    (address, model_index, epoch),
                ).fetchone()

                if latest is not None:
                    latest_payload = json.loads(str(latest["snapshot_json"]))
                    latest_mesh_id = str(latest_payload.get("mesh_id", ""))
                    if snapshot.mesh_id != latest_mesh_id:
                        if not allow_mesh_change:
                            raise ValueError(
                                "mesh verification snapshot mesh_id changed within epoch"
                            )
                        # Validator-initiated adoption: the old lineage is
                        # dead and its rows would shadow the new lineage's
                        # (lower) generations in every latest-row lookup.
                        self._conn.execute(
                            """DELETE FROM mesh_verification_snapshots
                               WHERE coordinator_address = ?
                                 AND model_index = ?
                                 AND epoch_number = ?""",
                            (address, model_index, epoch),
                        )
                        latest = None

                if latest is not None and generation < int(latest["generation"]):
                    raise ValueError(
                        "mesh verification snapshot generation rollback rejected"
                    )

                if latest is not None and generation == int(latest["generation"]):
                    if normalized_hash != str(latest["snapshot_hash"]):
                        raise ValueError(
                            "conflicting mesh verification snapshot hash at pinned generation"
                        )
                    if canonical_json != str(latest["snapshot_json"]):
                        raise ValueError(
                            "conflicting mesh verification snapshot payload at pinned generation"
                        )
                    if payload_expiry != int(latest["expires_at_unix"]):
                        raise ValueError(
                            "conflicting mesh verification snapshot expiry at pinned generation"
                        )
                    self._conn.commit()
                    return False

                self._conn.execute(
                    """INSERT INTO mesh_verification_snapshots (
                           coordinator_address, model_index, epoch_number,
                           generation, snapshot_hash, snapshot_json,
                           fetched_at, expires_at_unix
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        address,
                        model_index,
                        epoch,
                        generation,
                        normalized_hash,
                        canonical_json,
                        fetched_at,
                        payload_expiry,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    def get_latest_mesh_verification_snapshot_for_epoch(
        self,
        *,
        coordinator_address: str,
        model_index: int,
        epoch: int,
    ) -> Optional[dict]:
        """Return the highest pinned generation for one epoch, if present.

        Expired rows remain queryable because expiration is a verification-time
        decision and immutable history is also used for audit.
        """

        address = self._normalize_mesh_snapshot_address(coordinator_address)
        model_index = self._validate_mesh_snapshot_key_int(
            model_index,
            field_name="model_index",
            minimum=0,
        )
        epoch = self._validate_mesh_snapshot_key_int(
            epoch,
            field_name="epoch",
            minimum=0,
        )
        with self._lock:
            row = self._conn.execute(
                """SELECT coordinator_address, model_index, epoch_number,
                          generation, snapshot_hash, snapshot_json,
                          fetched_at, expires_at_unix
                   FROM mesh_verification_snapshots
                   WHERE coordinator_address = ?
                     AND model_index = ?
                     AND epoch_number = ?
                   ORDER BY generation DESC
                   LIMIT 1""",
                (address, model_index, epoch),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_mesh_verification_snapshot_history(
        self,
        *,
        coordinator_address: str,
        model_index: int,
        epoch: int,
    ) -> List[dict]:
        """Return all pinned generations for an epoch in ascending order."""

        address = self._normalize_mesh_snapshot_address(coordinator_address)
        model_index = self._validate_mesh_snapshot_key_int(
            model_index,
            field_name="model_index",
            minimum=0,
        )
        epoch = self._validate_mesh_snapshot_key_int(
            epoch,
            field_name="epoch",
            minimum=0,
        )
        with self._lock:
            rows = self._conn.execute(
                """SELECT coordinator_address, model_index, epoch_number,
                          generation, snapshot_hash, snapshot_json,
                          fetched_at, expires_at_unix
                   FROM mesh_verification_snapshots
                   WHERE coordinator_address = ?
                     AND model_index = ?
                     AND epoch_number = ?
                   ORDER BY generation ASC""",
                (address, model_index, epoch),
            ).fetchall()
            return [dict(row) for row in rows]

    # ── Hot-capacity audit state ─────────────────────────────────

    CAPACITY_ROSTER_RETENTION_SECONDS = 30 * 24 * 60 * 60

    def record_capacity_roster(
        self,
        *,
        slot_id: str,
        roster_epoch: int,
        roster_digest: str,
        roster_json: str,
        signature: str,
        received_at: Optional[float] = None,
    ) -> None:
        """Upsert one verified mesh roster document; prunes stale rows."""
        ts = time.time() if received_at is None else float(received_at)
        with self._lock:
            self._conn.execute(
                """INSERT INTO capacity_rosters (
                       slot_id, roster_epoch, roster_digest, roster_json,
                       signature, received_at
                   ) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (slot_id, roster_digest) DO UPDATE SET
                       roster_epoch = excluded.roster_epoch,
                       roster_json = excluded.roster_json,
                       signature = excluded.signature,
                       received_at = excluded.received_at""",
                (
                    str(slot_id),
                    int(roster_epoch),
                    str(roster_digest),
                    str(roster_json),
                    str(signature),
                    ts,
                ),
            )
            self._conn.execute(
                "DELETE FROM capacity_rosters WHERE received_at < ?",
                (ts - float(self.CAPACITY_ROSTER_RETENTION_SECONDS),),
            )
            self._conn.commit()

    def get_capacity_roster(self, slot_id: str) -> Optional[dict]:
        """Return the newest stored roster row for one endpoint slot."""
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM capacity_rosters
                   WHERE slot_id = ?
                   ORDER BY received_at DESC, roster_epoch DESC
                   LIMIT 1""",
                (str(slot_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_capacity_roster_missing(
        self, slot_id: str, epoch_number: int
    ) -> None:
        """Mark one mesh slot's roster as unlearnable at this epoch.

        first_missing_epoch is preserved across calls so the gate measures
        the full refusal streak, not the latest observation.
        """
        with self._lock:
            self._conn.execute(
                """INSERT INTO capacity_roster_probes (
                       slot_id, first_missing_epoch, last_missing_epoch
                   ) VALUES (?, ?, ?)
                   ON CONFLICT (slot_id) DO UPDATE SET
                       first_missing_epoch = CASE
                           WHEN capacity_roster_probes.first_missing_epoch <= 0
                               THEN excluded.first_missing_epoch
                           ELSE capacity_roster_probes.first_missing_epoch
                       END,
                       last_missing_epoch = excluded.last_missing_epoch""",
                (str(slot_id), int(epoch_number), int(epoch_number)),
            )
            self._conn.commit()

    def clear_capacity_roster_missing(self, slot_id: str) -> None:
        """Forget a slot's refusal streak once a roster is learned."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM capacity_roster_probes WHERE slot_id = ?",
                (str(slot_id),),
            )
            self._conn.commit()

    def get_capacity_roster_missing(self, slot_id: str) -> Optional[dict]:
        """Return the refusal-streak row for one slot, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM capacity_roster_probes WHERE slot_id = ?",
                (str(slot_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_capacity_roster_for_selection(
        self,
        slot_id: str,
        max_roster_epoch: int,
    ) -> Optional[dict]:
        """Newest roster old enough for the non-interactive selection domain.

        Selection determinism requires both sides to group on a frozen
        document, so only rosters with ``roster_epoch <= current_epoch - 1``
        may feed selection; callers pass that bound as ``max_roster_epoch``.
        """
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM capacity_rosters
                   WHERE slot_id = ? AND roster_epoch <= ?
                   ORDER BY received_at DESC, roster_epoch DESC
                   LIMIT 1""",
                (str(slot_id), int(max_roster_epoch)),
            ).fetchone()
        return dict(row) if row is not None else None

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
        incident_review_required: bool = False,
    ) -> None:
        """Insert one audit window and its selected endpoint slots."""
        now = time.time()
        incident_review_status = (
            "pending" if incident_review_required else "clear"
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO capacity_audit_windows (
                    audit_id, epoch_number, selection_block, audit_block, proof_challenge_block,
                    selection_block_hash, cohort_seed, status,
                    incident_review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?)""",
                (
                    audit_id,
                    int(epoch_number),
                    int(selection_block),
                    int(audit_block),
                    int(proof_challenge_block),
                    selection_block_hash,
                    cohort_seed,
                    incident_review_status,
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
                        gpu_index, roster_digest, pass_count, workload_spec,
                        deadline_s, transport_grace_s,
                        payload_deadline_s, drain_until_ts, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )""",
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
                        int(slot.get("gpu_index", 0) or 0),
                        str(slot.get("roster_digest", "") or ""),
                        int(slot.get("pass_count", 0) or 0),
                        json.dumps(
                            slot.get("workload_spec") or {},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
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

    def get_capacity_audit_windows_for_incident_review(
        self,
        *,
        now: Optional[float] = None,
    ) -> List[dict]:
        """Return finalized windows whose complete evidence interval elapsed.

        Review is intentionally delayed through the payload deadline. This
        prevents an early chain-finality callback from penalizing the first
        late receipt before the validator can see whether the whole cohort was
        affected by one local ingestion incident.
        """

        ts = time.time() if now is None else float(now)
        with self._lock:
            rows = self._conn.execute(
                """SELECT w.audit_id, w.epoch_number,
                          COUNT(s.model_index) AS slot_count,
                          MAX(COALESCE(s.payload_deadline_s, 0.0)) AS payload_deadline_s
                   FROM capacity_audit_windows w
                   JOIN capacity_audit_slots s ON s.audit_id = w.audit_id
                   WHERE w.chain_status = 'confirmed'
                     AND w.incident_review_status = 'pending'
                     AND w.proof_challenge_observed_at IS NOT NULL
                   GROUP BY w.audit_id, w.epoch_number,
                            w.proof_challenge_observed_at
                   HAVING ? > (
                       w.proof_challenge_observed_at
                       + MAX(COALESCE(s.payload_deadline_s, 0.0))
                   )
                   ORDER BY w.proof_challenge_observed_at ASC""",
                (ts,),
            ).fetchall()
        return [dict(row) for row in rows]

    def quarantine_capacity_audit_window(
        self,
        audit_id: str,
        *,
        reason: str,
        reviewed_at: Optional[float] = None,
    ) -> dict[str, int]:
        """Fail-neutralize one window after a validator-owned timing failure.

        This path is for direct validator evidence such as observing B_proof
        only after the live chain has already advanced beyond it.  It is not a
        miner forgiveness mechanism: independently established cryptographic
        failures are preserved.
        """

        bounded_reason = str(reason or "").strip()
        if not bounded_reason or len(bounded_reason) > 128:
            raise ValueError("a bounded validator-incident reason is required")
        ts = time.time() if reviewed_at is None else float(reviewed_at)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                window_cur = self._conn.execute(
                    """UPDATE capacity_audit_windows
                       SET status = 'validator_incident',
                           incident_review_status = 'quarantined',
                           incident_reviewed_at = ?, updated_at = ?
                       WHERE audit_id = ?
                         AND chain_status != 'reorged'
                         AND incident_review_status != 'quarantined'""",
                    (ts, ts, audit_id),
                )
                slot_cur = self._conn.execute(
                    """UPDATE capacity_audit_slots
                       SET verdict = 'timing_excused',
                           timing_status = 'excused',
                           failure_reason = ?, probation_required = 0,
                           drain_until_ts = 0, updated_at = ?
                       WHERE audit_id = ?
                         AND NOT (
                           verdict = 'hard_proof_miss'
                           AND COALESCE(failure_reason, '') NOT IN (
                             '', 'v2_final_commitment_not_pre_challenge'
                           )
                         )""",
                    (bounded_reason, ts, audit_id),
                )
                history_cur = self._conn.execute(
                    """UPDATE capacity_audit_history
                       SET verdict = 'timing_excused',
                           timing_status = 'excused',
                           failure_reason = ?, probation_required = 0,
                           slot_updated_at = ?
                       WHERE audit_id = ?
                         AND NOT (
                           verdict = 'hard_proof_miss'
                           AND COALESCE(failure_reason, '') NOT IN (
                             '', 'v2_final_commitment_not_pre_challenge'
                           )
                         )""",
                    (bounded_reason, ts, audit_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "windows_changed": int(window_cur.rowcount or 0),
            "slots_neutralized": int(slot_cur.rowcount or 0),
            "history_neutralized": int(history_cur.rowcount or 0),
        }

    def mark_capacity_audit_validator_signal(
        self,
        audit_id: str,
        *,
        reason: str,
        signaled_at: Optional[float] = None,
    ) -> bool:
        """Persist bounded validator-owned incident evidence for one window."""

        bounded_reason = str(reason or "").strip()
        if not bounded_reason or len(bounded_reason) > 128:
            raise ValueError("a bounded validator-incident signal is required")
        ts = time.time() if signaled_at is None else float(signaled_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_windows
                   SET incident_signal_reason = CASE
                           WHEN incident_signal_reason = '' THEN ?
                           ELSE incident_signal_reason
                       END,
                       incident_signaled_at = COALESCE(incident_signaled_at, ?),
                       updated_at = ?
                   WHERE audit_id = ?
                     AND chain_status != 'reorged'
                     AND incident_review_status = 'pending'
                     AND incident_signal_reason = ''""",
                (bounded_reason, ts, ts, audit_id),
            )
            self._conn.commit()
        return bool(cur.rowcount or 0)

    def review_capacity_audit_failure_cluster(
        self,
        audit_id: str,
        *,
        enabled: bool,
        failure_fraction: float,
        min_failures: int,
        min_distinct_miners: int,
        reviewed_at: Optional[float] = None,
    ) -> dict[str, object]:
        """Fail-neutralize a validator-shaped cohort failure cluster once.

        Only receipt timing, missing-final, and pre-challenge chronology
        outcomes participate. Cryptographic proof failures neither contribute
        to the trigger nor get neutralized if an unrelated timing cluster
        quarantines the same window.
        """

        ts = time.time() if reviewed_at is None else float(reviewed_at)
        fraction = float(failure_fraction)
        minimum = max(1, int(min_failures))
        distinct_minimum = max(1, int(min_distinct_miners))
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("failure_fraction must be in (0.0, 1.0]")

        def _is_validator_sensitive_failure(row: sqlite3.Row) -> bool:
            verdict = str(row["verdict"] or "")
            reason = str(row["failure_reason"] or "")
            return (
                (verdict == "timing_miss" and reason == "deadline_exceeded")
                or (verdict == "no_show" and reason in ("", "missing_final_receipt"))
                or (
                    verdict == "hard_proof_miss"
                    and reason == "v2_final_commitment_not_pre_challenge"
                )
            )

        with self._lock:
            window = self._conn.execute(
                """SELECT audit_id, epoch_number, status, chain_status,
                          incident_review_status, incident_signal_reason
                   FROM capacity_audit_windows WHERE audit_id = ?""",
                (audit_id,),
            ).fetchone()
            if window is None:
                raise ValueError("capacity audit window not found")
            if str(window["chain_status"] or "") != "confirmed":
                raise ValueError("capacity audit window is not chain-confirmed")
            if str(window["incident_review_status"] or "") != "pending":
                return {
                    "audit_id": audit_id,
                    "epoch_number": int(window["epoch_number"]),
                    "reviewed": False,
                    "status": str(window["incident_review_status"] or ""),
                    "quarantined": str(window["status"] or "") == "validator_incident",
                }

            rows = self._conn.execute(
                """SELECT miner_address, miner_uid, miner_hotkey_ss58,
                          model_index, verdict, timing_status, proof_status,
                          failure_reason, probation_required
                   FROM capacity_audit_slots
                   WHERE audit_id = ?
                   ORDER BY miner_address, model_index""",
                (audit_id,),
            ).fetchall()
            failures = [row for row in rows if _is_validator_sensitive_failure(row)]
            identities = {
                (
                    f"hotkey:{str(row['miner_hotkey_ss58'])}"
                    if str(row["miner_hotkey_ss58"] or "")
                    else (
                        f"uid:{int(row['miner_uid'])}"
                        if row["miner_uid"] is not None
                        else f"address:{str(row['miner_address']).lower()}"
                    )
                )
                for row in failures
            }
            total = len(rows)
            required = max(minimum, int(math.ceil(total * fraction)))
            quarantine = bool(
                enabled
                and str(window["incident_signal_reason"] or "")
                and total > 0
                and len(failures) >= required
                and len(identities) >= distinct_minimum
            )
            reason = (
                "validator_failure_cluster:"
                f"fail={len(failures)}/{total},miners={len(identities)}"
            )

            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if quarantine:
                    self._conn.execute(
                        """UPDATE capacity_audit_windows
                           SET status = 'validator_incident',
                               incident_review_status = 'quarantined',
                               incident_reviewed_at = ?, updated_at = ?
                           WHERE audit_id = ?
                             AND incident_review_status = 'pending'""",
                        (ts, ts, audit_id),
                    )
                    # Preserve independently invalid cryptographic proofs.
                    # Every other outcome in the contaminated window is
                    # neutral evidence: neither a pass nor a failure.
                    slot_cur = self._conn.execute(
                        """UPDATE capacity_audit_slots
                           SET verdict = 'timing_excused',
                               timing_status = 'excused',
                               failure_reason = ?, probation_required = 0,
                               drain_until_ts = 0, updated_at = ?
                           WHERE audit_id = ?
                             AND NOT (
                               verdict = 'hard_proof_miss'
                               AND COALESCE(failure_reason, '') !=
                                   'v2_final_commitment_not_pre_challenge'
                             )""",
                        (reason, ts, audit_id),
                    )
                    history_cur = self._conn.execute(
                        """UPDATE capacity_audit_history
                           SET verdict = 'timing_excused',
                               timing_status = 'excused',
                               failure_reason = ?, probation_required = 0,
                               slot_updated_at = ?
                           WHERE audit_id = ?
                             AND NOT (
                               verdict = 'hard_proof_miss'
                               AND COALESCE(failure_reason, '') !=
                                   'v2_final_commitment_not_pre_challenge'
                             )""",
                        (reason, ts, audit_id),
                    )
                else:
                    self._conn.execute(
                        """UPDATE capacity_audit_windows
                           SET incident_review_status = 'clear',
                               incident_reviewed_at = ?, updated_at = ?
                           WHERE audit_id = ?
                             AND incident_review_status = 'pending'""",
                        (ts, ts, audit_id),
                    )
                    slot_cur = None
                    history_cur = None
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return {
            "audit_id": audit_id,
            "epoch_number": int(window["epoch_number"]),
            "reviewed": True,
            "status": "quarantined" if quarantine else "clear",
            "quarantined": quarantine,
            "slot_count": total,
            "eligible_failures": len(failures),
            "required_failures": required,
            "distinct_failure_miners": len(identities),
            "required_distinct_miners": distinct_minimum,
            "validator_signal": str(window["incident_signal_reason"] or ""),
            "reason": reason if quarantine else "",
            "slots_neutralized": int(slot_cur.rowcount or 0) if slot_cur else 0,
            "history_neutralized": (
                int(history_cur.rowcount or 0) if history_cur else 0
            ),
        }

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
                             WHEN verdict != 'chain_reorged' THEN 'chain_reorged'
                             ELSE verdict
                           END,
                           failure_reason = CASE
                             WHEN verdict != 'chain_reorged' THEN 'audit_block_reorged'
                             ELSE failure_reason
                           END,
                           probation_required = 0,
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
        probation_required: bool = False,
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
                    """SELECT DISTINCT s.audit_id, s.miner_address, s.model_index,
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
                       probation_required = CASE
                           WHEN ? != 0 THEN 1 ELSE probation_required
                       END,
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
                (1 if probation_required else 0, ts, ts),
            )
            updated = cur.rowcount or 0
            if require_proof_payload:
                # Security invariant: the proof deadline starts when B_proof is
                # first observed.  Finalization lag must never grant the miner
                # extra workspace-retention or proof-construction time.
                if return_slots:
                    rows = self._conn.execute(
                        """SELECT DISTINCT s.audit_id, s.miner_address, s.model_index,
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
                           probation_required = CASE
                               WHEN ? != 0 THEN 1 ELSE probation_required
                           END,
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
                    (1 if probation_required else 0, ts, ts),
                )
                updated += cur.rowcount or 0
            self._conn.commit()
        return expired_slots if return_slots else updated

    def get_capacity_audit_slot(
        self,
        audit_id: str,
        address: str,
        model_index: int,
        gpu_index: int = 0,
        *,
        receipt_ingress: bool = False,
    ) -> Optional[dict]:
        lock = self._capacity_ingress_lock if receipt_ingress else self._lock
        conn = self._capacity_ingress_conn if receipt_ingress else self._conn
        with lock:
            row = conn.execute(
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
                   WHERE s.audit_id = ? AND s.miner_address = ?
                     AND s.model_index = ? AND s.gpu_index = ?""",
                (audit_id, address.lower(), int(model_index), int(gpu_index)),
            ).fetchone()
        return dict(row) if row is not None else None

    def rollback_abandoned_transactions(self) -> tuple[str, ...]:
        """Rollback transactions left open after an interrupted DB operation.

        Every normal database helper commits or rolls back before releasing its
        Python lock.  Finding ``in_transaction`` after acquiring that lock
        therefore means an earlier operation abandoned an implicit SQLite
        transaction.  Such a transaction can retain WAL's single-writer lock
        indefinitely and prevent the validator control plane from advancing.
        """

        recovered: list[str] = []
        connections = (
            ("primary", self._lock, self._conn),
            (
                "capacity_ingress",
                self._capacity_ingress_lock,
                self._capacity_ingress_conn,
            ),
        )
        for name, lock, conn in connections:
            with lock:
                if conn is not None and conn.in_transaction:
                    conn.rollback()
                    recovered.append(name)
        return tuple(recovered)

    def record_capacity_audit_pass0(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        pass0_root: str,
        artifact: dict,
        received_at: Optional[float] = None,
        receipt_ingress: bool = False,
        gpu_index: int = 0,
    ) -> bool:
        ts = time.time() if received_at is None else float(received_at)
        lock = self._capacity_ingress_lock if receipt_ingress else self._lock
        conn = self._capacity_ingress_conn if receipt_ingress else self._conn
        with lock:
            try:
                cur = conn.execute(
                    """UPDATE capacity_audit_slots
                   SET pass0_received_at = COALESCE(pass0_received_at, ?),
                       pass0_root = ?,
                       pass0_artifact = ?,
                       verdict = CASE WHEN verdict = 'pending' THEN 'pass0_seen' ELSE verdict END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?
                     AND EXISTS (
                       SELECT 1 FROM capacity_audit_windows w
                       WHERE w.audit_id = capacity_audit_slots.audit_id
                         AND w.chain_status != 'reorged'
                         AND w.incident_review_status != 'quarantined'
                     )""",
                    (
                        ts,
                        pass0_root,
                        json.dumps(artifact, sort_keys=True),
                        ts,
                        audit_id,
                        address.lower(),
                        int(model_index),
                        int(gpu_index),
                    ),
                )
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
        return (cur.rowcount or 0) == 1

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
        probation_required: bool = False,
        final_observed_block: Optional[int] = None,
        received_at: Optional[float] = None,
        receipt_ingress: bool = False,
        gpu_index: int = 0,
    ) -> Tuple[Optional[dict], bool]:
        """Persist the first final receipt without letting retries retime it.

        Miner delivery is at-least-once: a committed request can be retried when
        its HTTP response is lost. Timing, transcript, and validator-observed
        chronology belong to the first accepted delivery. A later hard failure
        may update the verdict without replacing those commitments.
        """
        ts = time.time() if received_at is None else float(received_at)
        pass0_root = str(artifact.get("pass0_root") or "")
        observed_block = (
            None
            if final_observed_block is None
            else max(0, int(final_observed_block))
        )
        hard_override = verdict == "hard_proof_miss"
        lock = self._capacity_ingress_lock if receipt_ingress else self._lock
        conn = self._capacity_ingress_conn if receipt_ingress else self._conn
        with lock:
            try:
                cur = conn.execute(
                    """UPDATE capacity_audit_slots
                   SET final_received_at = CASE
                           WHEN final_received_at IS NULL THEN ?
                           ELSE final_received_at
                       END,
                       final_observed_block = CASE
                           WHEN final_received_at IS NULL THEN ?
                           ELSE final_observed_block
                       END,
                       pass0_root = CASE
                           WHEN final_received_at IS NULL AND pass0_root = '' THEN ?
                           ELSE pass0_root
                       END,
                       final_root = CASE
                           WHEN final_received_at IS NULL THEN ?
                           ELSE final_root
                       END,
                       transcript_root = CASE
                           WHEN final_received_at IS NULL THEN ?
                           ELSE transcript_root
                       END,
                       final_artifact = CASE
                           WHEN final_received_at IS NULL THEN ?
                           ELSE final_artifact
                       END,
                       timing_status = ?,
                       verdict = ?,
                       failure_reason = COALESCE(NULLIF(?, ''), failure_reason),
                       probation_required = CASE
                           WHEN ? != 0 THEN 1 ELSE probation_required
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?
                     AND EXISTS (
                       SELECT 1 FROM capacity_audit_windows w
                       WHERE w.audit_id = capacity_audit_slots.audit_id
                         AND w.chain_status != 'reorged'
                         AND w.incident_review_status != 'quarantined'
                     )
                     AND (
                       final_received_at IS NULL
                       OR (? = 1 AND verdict != 'hard_proof_miss')
                     )""",
                    (
                        ts,
                        observed_block,
                        pass0_root,
                        final_root,
                        transcript_root,
                        json.dumps(artifact, sort_keys=True),
                        timing_status,
                        verdict,
                        failure_reason,
                        1 if probation_required else 0,
                        ts,
                        audit_id,
                        address.lower(),
                        int(model_index),
                        int(gpu_index),
                        int(hard_override),
                    ),
                )
                row = conn.execute(
                    """SELECT *
                   FROM capacity_audit_slots
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?""",
                    (
                        audit_id,
                        address.lower(),
                        int(model_index),
                        int(gpu_index),
                    ),
                ).fetchone()
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
        return (dict(row) if row is not None else None, bool(cur.rowcount))

    def reconcile_capacity_audit_duplicate_timing_misses(
        self,
        *,
        reconciled_at: Optional[float] = None,
    ) -> dict[str, int]:
        """Repair timing misses created by late retries of on-time final receipts."""
        ts = time.time() if reconciled_at is None else float(reconciled_at)
        on_time_final = """
            s.final_received_at IS NOT NULL
            AND w.audit_start_observed_at IS NOT NULL
            AND s.deadline_s > 0
            AND s.final_received_at <= (
                w.audit_start_observed_at
                + s.deadline_s
                + MAX(0.0, s.transport_grace_s)
            )
        """
        with self._lock:
            history_cur = self._conn.execute(
                f"""UPDATE capacity_audit_history AS h
                    SET verdict = 'timing_pass',
                        timing_status = 'pass',
                        failure_reason = NULL,
                        slot_updated_at = ?
                    WHERE h.verdict = 'timing_miss'
                      AND h.failure_reason = 'deadline_exceeded'
                      AND EXISTS (
                        SELECT 1
                        FROM capacity_audit_slots s
                        JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                        WHERE s.audit_id = h.audit_id
                          AND s.miner_address = h.miner_address
                          AND s.model_index = h.model_index
                          AND s.gpu_index = h.gpu_index
                          AND s.verdict = 'timing_miss'
                          AND s.failure_reason = 'deadline_exceeded'
                          AND {on_time_final}
                      )""",
                (ts,),
            )
            slot_cur = self._conn.execute(
                f"""UPDATE capacity_audit_slots AS s
                    SET verdict = 'timing_pass',
                        timing_status = 'pass',
                        failure_reason = NULL,
                        updated_at = ?
                    WHERE s.verdict = 'timing_miss'
                      AND s.failure_reason = 'deadline_exceeded'
                      AND EXISTS (
                        SELECT 1
                        FROM capacity_audit_windows w
                        WHERE w.audit_id = s.audit_id
                          AND {on_time_final}
                      )""",
                (ts,),
            )
            self._conn.commit()
        return {
            "slot_rows": int(slot_cur.rowcount or 0),
            "history_rows": int(history_cur.rowcount or 0),
        }

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
        probation_required: bool = False,
        received_at: Optional[float] = None,
        gpu_index: int = 0,
    ) -> bool:
        ts = time.time() if received_at is None else float(received_at)
        verify_ms = None if proof_verify_ms is None else max(0.0, float(proof_verify_ms))
        verified_success = proof_status in {
            "combined_proof_verified",
            "legacy_combined_proof_compatibility_accepted",
            "legacy_combined_proof_grace_accepted",
            "proof_verified",
        }
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET proof_received_at = COALESCE(proof_received_at, ?),
                       proof_status = ?,
                       proof_verify_ms = COALESCE(?, proof_verify_ms),
                       verdict = CASE
                           WHEN ? != 0 AND timing_status = 'pass'
                               THEN 'timing_pass'
                           ELSE ?
                       END,
                       failure_reason = CASE
                           WHEN ? != 0 AND timing_status = 'pass' THEN NULL
                           ELSE COALESCE(NULLIF(?, ''), failure_reason)
                       END,
                       proof_artifact_path = COALESCE(NULLIF(?, ''), proof_artifact_path),
                       probation_required = CASE
                           WHEN ? != 0 AND timing_status = 'pass' THEN 0
                           WHEN ? != 0 THEN 1 ELSE probation_required
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?
                     AND EXISTS (
                       SELECT 1 FROM capacity_audit_windows w
                       WHERE w.audit_id = capacity_audit_slots.audit_id
                         AND w.chain_status != 'reorged'
                         AND w.incident_review_status != 'quarantined'
                     )""",
                (
                    ts,
                    proof_status,
                    verify_ms,
                    1 if verified_success else 0,
                    verdict,
                    1 if verified_success else 0,
                    failure_reason,
                    proof_artifact_path,
                    1 if verified_success else 0,
                    1 if probation_required else 0,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                    int(gpu_index),
                ),
            )
            self._conn.commit()
        return (cur.rowcount or 0) == 1

    def recently_scored_mesh_keys(self, min_epoch: int) -> set:
        """(address, model_index) pairs of active entries scored since min_epoch.

        Used to exempt still-serving registrations from the superseded-skip
        in snapshot verification: a scoring entry is serving, whatever its
        registration order.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT address, model_index FROM miner_entries
                   WHERE is_active = 1 AND last_scored_epoch >= ?""",
                (int(min_epoch),),
            ).fetchall()
        return {
            (str(r["address"]).lower(), int(r["model_index"])) for r in rows
        }

    def record_validator_outage_interval(
        self, start_ts: float, end_ts: float
    ) -> None:
        """Persist an observed validator liveness gap (stall or restart).

        The close consults these to suppress miner penalties for epochs the
        validator itself marred: a wedged or restarted validator must never
        cost miners probation, because its own dead canaries are
        indistinguishable from miner failures at scoring time.
        """
        if not (end_ts > start_ts > 0):
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS validator_outage_intervals (
                    start_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    recorded_at REAL NOT NULL
                )"""
            )
            self._conn.execute(
                "INSERT INTO validator_outage_intervals "
                "(start_ts, end_ts, recorded_at) VALUES (?, ?, ?)",
                (float(start_ts), float(end_ts), now),
            )
            self._conn.execute(
                "DELETE FROM validator_outage_intervals WHERE end_ts < ?",
                (now - 7 * 86400,),
            )
            self._conn.commit()

    def validator_outage_overlap_seconds(
        self, window_start: float, window_end: float
    ) -> float:
        """Total recorded validator-outage seconds inside a wall-clock window."""
        if window_end <= window_start:
            return 0.0
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT start_ts, end_ts FROM validator_outage_intervals "
                    "WHERE end_ts > ? AND start_ts < ?",
                    (float(window_start), float(window_end)),
                ).fetchall()
            except sqlite3.OperationalError:
                return 0.0
        total = 0.0
        for row in rows:
            s = max(float(row["start_ts"]), float(window_start))
            e = min(float(row["end_ts"]), float(window_end))
            if e > s:
                total += e - s
        return total

    def find_validator_outage_no_show_audits(
        self,
        outage_start: float,
        outage_end: float,
    ) -> Dict[int, List[str]]:
        """Audit windows blaming miners for receipts due during a validator outage.

        A slot's receipt-acceptance window is
        [audit_start_observed_at, audit_start_observed_at + deadline_s +
        transport_grace_s]. If the validator itself was down for any part of
        that window, a no_show/missing_final_receipt verdict says nothing
        about the miner - its publishes were refused at the door. Returns
        {epoch_number: [audit_id, ...]} for exactly those windows, skipping
        ones already neutralized as validator incidents.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT DISTINCT s.audit_id, w.epoch_number
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   WHERE s.verdict = 'no_show'
                     AND COALESCE(s.failure_reason, '') = 'missing_final_receipt'
                     AND w.status != 'validator_incident'
                     AND w.audit_start_observed_at IS NOT NULL
                     AND w.audit_start_observed_at < ?
                     AND w.audit_start_observed_at + s.deadline_s + s.transport_grace_s > ?
                   ORDER BY w.epoch_number, s.audit_id""",
                (float(outage_end), float(outage_start)),
            ).fetchall()
        grouped: Dict[int, List[str]] = {}
        for row in rows:
            grouped.setdefault(int(row["epoch_number"]), []).append(
                str(row["audit_id"])
            )
        return grouped

    def reconcile_capacity_audit_incident(
        self,
        audit_ids: Sequence[str],
        *,
        expected_epoch: int,
        reason: str,
        apply: bool = False,
        reconciled_at: Optional[float] = None,
    ) -> dict[str, object]:
        """Neutralize an exact validator incident without touching other audits.

        The caller must supply complete audit IDs and the expected epoch. A
        probation entry is cleared only when it began in that epoch, at least
        one targeted row applied it, and no independent same-or-later failure
        exists. EMA values are deliberately not rewritten: their later update
        history cannot be inverted safely from a mutable current value.
        """

        exact_ids = tuple(dict.fromkeys(str(value).strip() for value in audit_ids))
        if not exact_ids or any(len(value) != 64 for value in exact_ids):
            raise ValueError("complete 64-character audit IDs are required")
        if not reason or len(reason) > 96:
            raise ValueError("a bounded incident reason is required")
        ts = time.time() if reconciled_at is None else float(reconciled_at)
        placeholders = ",".join("?" for _ in exact_ids)
        epoch_i = int(expected_epoch)

        with self._lock:
            windows = self._conn.execute(
                f"""SELECT audit_id, epoch_number, status
                    FROM capacity_audit_windows
                    WHERE audit_id IN ({placeholders})
                    ORDER BY audit_id""",
                exact_ids,
            ).fetchall()
            found = {str(row["audit_id"]): int(row["epoch_number"]) for row in windows}
            if set(found) != set(exact_ids):
                missing = sorted(set(exact_ids) - set(found))
                raise ValueError(f"incident audit IDs not found: {missing}")
            wrong_epochs = {
                audit_id: epoch
                for audit_id, epoch in found.items()
                if epoch != epoch_i
            }
            if wrong_epochs:
                raise ValueError(
                    f"incident audit epoch mismatch: expected={epoch_i} actual={wrong_epochs}"
                )

            slot_rows = self._conn.execute(
                f"""SELECT audit_id, miner_address, model_index, verdict,
                           timing_status, proof_status, failure_reason,
                           probation_required, probation_applied_at
                    FROM capacity_audit_slots
                    WHERE audit_id IN ({placeholders})""",
                exact_ids,
            ).fetchall()
            history_rows = self._conn.execute(
                f"""SELECT audit_id, miner_address, model_index, verdict,
                           timing_status, proof_status, failure_reason,
                           probation_required, probation_applied_at
                    FROM capacity_audit_history
                    WHERE audit_id IN ({placeholders})""",
                exact_ids,
            ).fetchall()

            applied_keys = {
                (str(row["miner_address"]).lower(), int(row["model_index"]))
                for row in (*slot_rows, *history_rows)
                if row["probation_applied_at"] is not None
            }
            probation_candidates: list[tuple[str, int]] = []
            retained_probation: list[dict[str, object]] = []
            for address, model_index in sorted(applied_keys):
                entry = self._conn.execute(
                    """SELECT probation_entered_epoch
                       FROM miner_entries
                       WHERE address = ? AND model_index = ?""",
                    (address, model_index),
                ).fetchone()
                if entry is None or entry["probation_entered_epoch"] is None:
                    continue
                entered_epoch = int(entry["probation_entered_epoch"])
                blockers: list[str] = []
                if entered_epoch != epoch_i:
                    blockers.append("probation_started_in_other_epoch")
                other_capacity = self._conn.execute(
                    f"""SELECT 1
                        FROM capacity_audit_slots s
                        JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                        WHERE s.miner_address = ? AND s.model_index = ?
                          AND w.epoch_number >= ?
                          AND s.audit_id NOT IN ({placeholders})
                          AND s.verdict IN ('timing_miss', 'hard_proof_miss', 'no_show')
                          AND s.probation_required != 0
                        UNION ALL
                        SELECT 1
                        FROM capacity_audit_history h
                        WHERE h.miner_address = ? AND h.model_index = ?
                          AND h.epoch_number >= ?
                          AND h.audit_id NOT IN ({placeholders})
                          AND h.verdict IN ('timing_miss', 'hard_proof_miss', 'no_show')
                          AND h.probation_required != 0
                        LIMIT 1""",
                    (
                        address,
                        model_index,
                        epoch_i,
                        *exact_ids,
                        address,
                        model_index,
                        epoch_i,
                        *exact_ids,
                    ),
                ).fetchone()
                if other_capacity is not None:
                    blockers.append("independent_capacity_failure")
                hard_failure = self._conn.execute(
                    """SELECT 1 FROM proof_v3_hard_failures
                       WHERE miner_address = ? AND model_index = ?
                         AND source_epoch >= ? LIMIT 1""",
                    (address, model_index, epoch_i),
                ).fetchone()
                if hard_failure is not None:
                    blockers.append("independent_hard_proof_failure")
                canary_failure = self._conn.execute(
                    """SELECT 1 FROM canary_results
                       WHERE miner_address = ? AND model_index = ?
                         AND epoch_number >= ? AND proof_requested != 0
                         AND COALESCE(proof_verified, 0) = 0
                         AND COALESCE(proof_failure_reason, '') != ''
                       LIMIT 1""",
                    (address, model_index, epoch_i),
                ).fetchone()
                if canary_failure is not None:
                    blockers.append("independent_canary_proof_failure")
                if blockers:
                    retained_probation.append(
                        {
                            "address": address,
                            "model_index": model_index,
                            "reasons": blockers,
                        }
                    )
                else:
                    probation_candidates.append((address, model_index))

            result: dict[str, object] = {
                "apply": bool(apply),
                "expected_epoch": epoch_i,
                "audit_ids": list(exact_ids),
                "slot_rows_found": len(slot_rows),
                "history_rows_found": len(history_rows),
                "windows_to_mark": sum(
                    1 for row in windows if str(row["status"]) != "validator_incident"
                ),
                "slots_to_neutralize": sum(
                    1
                    for row in slot_rows
                    if str(row["verdict"]) != "timing_excused"
                    or str(row["timing_status"]) != "excused"
                    or str(row["failure_reason"] or "") != reason
                    or int(row["probation_required"] or 0) != 0
                ),
                "history_to_neutralize": sum(
                    1
                    for row in history_rows
                    if str(row["verdict"]) != "timing_excused"
                    or str(row["timing_status"]) != "excused"
                    or str(row["failure_reason"] or "") != reason
                    or int(row["probation_required"] or 0) != 0
                ),
                "probation_to_clear": [
                    {"address": address, "model_index": model_index}
                    for address, model_index in probation_candidates
                ],
                "probation_retained": retained_probation,
                "ema_restored": False,
            }
            if not apply:
                return result

            self._conn.execute("BEGIN IMMEDIATE")
            try:
                window_cur = self._conn.execute(
                    f"""UPDATE capacity_audit_windows
                    SET status = 'validator_incident',
                        incident_review_status = 'quarantined',
                        incident_reviewed_at = ?, updated_at = ?
                    WHERE audit_id IN ({placeholders})
                      AND status != 'validator_incident'""",
                    (ts, ts, *exact_ids),
                )
                slot_cur = self._conn.execute(
                    f"""UPDATE capacity_audit_slots
                    SET verdict = 'timing_excused', timing_status = 'excused',
                        failure_reason = ?, probation_required = 0,
                        drain_until_ts = 0, updated_at = ?
                    WHERE audit_id IN ({placeholders})
                      AND (verdict != 'timing_excused'
                           OR timing_status != 'excused'
                           OR COALESCE(failure_reason, '') != ?
                           OR probation_required != 0)""",
                    (reason, ts, *exact_ids, reason),
                )
                history_cur = self._conn.execute(
                    f"""UPDATE capacity_audit_history
                    SET verdict = 'timing_excused', timing_status = 'excused',
                        failure_reason = ?, probation_required = 0,
                        slot_updated_at = ?
                    WHERE audit_id IN ({placeholders})
                      AND (verdict != 'timing_excused'
                           OR timing_status != 'excused'
                           OR COALESCE(failure_reason, '') != ?
                           OR probation_required != 0)""",
                    (reason, ts, *exact_ids, reason),
                )
                probation_changed = 0
                for address, model_index in probation_candidates:
                    cur = self._conn.execute(
                        """UPDATE miner_entries
                       SET probation_entered_epoch = NULL,
                           probation_consecutive_passes = 0,
                           updated_at = ?
                       WHERE address = ? AND model_index = ?
                         AND probation_entered_epoch = ?""",
                        (ts, address, model_index, epoch_i),
                    )
                    probation_changed += int(cur.rowcount or 0)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            result["windows_changed"] = int(window_cur.rowcount or 0)
            result["slots_changed"] = int(slot_cur.rowcount or 0)
            result["history_changed"] = int(history_cur.rowcount or 0)
            result["probation_changed"] = probation_changed
            return result

    def apply_finalized_capacity_audit_probation_once(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        epoch: int,
        applied_at: Optional[float] = None,
    ) -> Optional[dict]:
        """Atomically apply one finalized capacity-audit probation event.

        The slot must already belong to a chain-confirmed window and have
        ``probation_required`` set when the failure was observed. The slot
        marker, persisted probation state, and EMA decay commit together so a
        restarted validator cannot apply the same audit penalty twice.
        """

        ts = time.time() if applied_at is None else float(applied_at)
        address = address.lower()
        with self._lock:
            row = self._conn.execute(
                """SELECT
                       s.endpoint,
                       s.failure_reason,
                       s.probation_required,
                       s.probation_applied_at,
                       w.chain_status,
                       m.probation_entered_epoch,
                       m.ema_score
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   LEFT JOIN miner_entries m
                     ON m.address = s.miner_address
                    AND m.model_index = s.model_index
                   WHERE s.audit_id = ?
                     AND s.miner_address = ?
                     AND s.model_index = ?
                     AND s.verdict = 'hard_proof_miss'""",
                (audit_id, address, int(model_index)),
            ).fetchone()
            if (
                row is None
                or str(row["chain_status"] or "") != "confirmed"
                or not bool(row["probation_required"])
                or row["probation_applied_at"] is not None
            ):
                return None

            miner_exists = row["ema_score"] is not None
            was_on_probation = row["probation_entered_epoch"] is not None
            if miner_exists:
                self._conn.execute(
                    """UPDATE miner_entries
                       SET probation_entered_epoch = COALESCE(
                               probation_entered_epoch, ?
                           ),
                           probation_consecutive_passes = 0,
                           ema_score = ema_score * 0.5,
                           updated_at = ?
                       WHERE address = ? AND model_index = ?""",
                    (int(epoch), ts, address, int(model_index)),
                )
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET probation_applied_at = ?,
                       updated_at = ?
                   WHERE audit_id = ?
                     AND miner_address = ?
                     AND model_index = ?
                     AND probation_applied_at IS NULL
                     AND probation_required != 0""",
                (ts, ts, audit_id, address, int(model_index)),
            )
            # Marks every ordinal row of the audit group in one statement, so
            # a mesh audit with several failing GPUs still applies exactly one
            # probation event.
            if (cur.rowcount or 0) < 1:
                self._conn.rollback()
                return None
            self._conn.commit()
            return {
                "audit_id": audit_id,
                "address": address,
                "model_index": int(model_index),
                "endpoint": str(row["endpoint"] or ""),
                "failure_reason": str(row["failure_reason"] or ""),
                "miner_exists": miner_exists,
                "was_on_probation": was_on_probation,
            }

    def get_finalized_capacity_audit_probation_candidates(self) -> List[dict]:
        """Return unapplied hard-proof failures whose chain anchors are final."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT s.audit_id, s.miner_address, s.model_index,
                          s.endpoint, s.failure_reason
                   FROM capacity_audit_slots s
                   JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                   WHERE s.verdict = 'hard_proof_miss'
                     AND s.probation_required != 0
                     AND s.probation_applied_at IS NULL
                     AND w.chain_status = 'confirmed'
                     AND w.incident_review_status != 'pending'
                   ORDER BY s.updated_at ASC, s.audit_id ASC""",
            ).fetchall()
        return [dict(row) for row in rows]

    def record_capacity_audit_proof_received(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        received_at: Optional[float] = None,
        gpu_index: int = 0,
    ) -> bool:
        ts = time.time() if received_at is None else float(received_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET proof_received_at = COALESCE(proof_received_at, ?),
                       proof_status = CASE
                           WHEN proof_status = 'pending' THEN 'verify_pending'
                           ELSE proof_status
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?
                     AND EXISTS (
                       SELECT 1 FROM capacity_audit_windows w
                       WHERE w.audit_id = capacity_audit_slots.audit_id
                         AND w.chain_status != 'reorged'
                         AND w.incident_review_status != 'quarantined'
                     )""",
                (
                    ts,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                    int(gpu_index),
                ),
            )
            self._conn.commit()
        return (cur.rowcount or 0) == 1

    def release_capacity_audit_drain(
        self,
        *,
        audit_id: str,
        address: str,
        model_index: int,
        released_at: Optional[float] = None,
        gpu_index: int = 0,
    ) -> int:
        """Release one audit opening from proxy drain after its evidence is complete.

        Keyed per gpu_index on purpose: a mesh slot's drain only ends once
        every ordinal's row has been released, so one finished GPU cannot
        un-drain the endpoint while sibling openings are still proving.
        """
        ts = time.time() if released_at is None else float(released_at)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE capacity_audit_slots
                   SET drain_until_ts = CASE
                           WHEN drain_until_ts > ? THEN ? ELSE drain_until_ts
                       END,
                       updated_at = ?
                   WHERE audit_id = ? AND miner_address = ? AND model_index = ?
                     AND gpu_index = ?""",
                (
                    ts,
                    ts,
                    ts,
                    audit_id,
                    address.lower(),
                    int(model_index),
                    int(gpu_index),
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
                       OR proof_status IN (
                         'proof_verified',
                         'combined_proof_verified',
                         'legacy_combined_proof_compatibility_accepted',
                         'legacy_combined_proof_grace_accepted'
                       )
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

    # Worst-of severity ranks for one audit's ordinal rows. A mesh audit fans
    # out to N per-GPU rows under one audit_id; strike counting collapses each
    # (audit_id, address, model_index) group to its single worst sub-verdict
    # so per-ordinal fan-out can never multiply strikes.
    _CAPACITY_FAILURE_SEVERITY_SQL = """CASE s.verdict
        WHEN 'hard_proof_miss' THEN 4
        WHEN 'timing_miss' THEN 3
        WHEN 'no_show' THEN 2
        ELSE 0 END"""
    _CAPACITY_FAILURE_SEVERITY_RANKS = {
        "hard_proof_miss": 4,
        "timing_miss": 3,
        "no_show": 2,
    }

    def recent_capacity_failures(
        self,
        address: str,
        model_index: int,
        *,
        since_epoch: int,
        verdicts: Tuple[str, ...] = ("timing_miss", "hard_proof_miss", "no_show"),
        require_chain_confirmed: bool = False,
        require_consequence_eligible: bool = False,
        require_incident_reviewed: bool = False,
        before_epoch: Optional[int] = None,
    ) -> int:
        wanted_ranks = sorted(
            {
                self._CAPACITY_FAILURE_SEVERITY_RANKS[v]
                for v in verdicts
                if v in self._CAPACITY_FAILURE_SEVERITY_RANKS
            }
        )
        if not wanted_ranks:
            return 0
        rank_placeholders = ",".join("?" for _ in wanted_ranks)
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
        eligibility_clause = (
            " AND s.probation_required != 0" if require_consequence_eligible else ""
        )
        review_clause = (
            " AND w.incident_review_status != 'pending'"
            if require_incident_reviewed else ""
        )
        before_clause = " AND w.epoch_number < ?" if before_epoch is not None else ""
        with self._lock:
            identity_clause, identity_params = self._capacity_address_identity_filter_locked(
                address
            )
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS n
                    FROM (
                      SELECT MAX({self._CAPACITY_FAILURE_SEVERITY_SQL}) AS worst
                      FROM capacity_audit_slots s
                      JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                      WHERE s.miner_address = ?
                        AND s.model_index = ?
                        AND w.epoch_number >= ?
                        {before_clause}
                        AND s.verdict IN ('timing_miss', 'hard_proof_miss', 'no_show')
                        AND {identity_clause}
                        {eligibility_clause}
                        {review_clause}
                        {confirmation_clause}
                      GROUP BY s.audit_id
                    )
                    WHERE worst IN ({rank_placeholders})""",
                (
                    address.lower(), int(model_index), int(since_epoch),
                    *((int(before_epoch),) if before_epoch is not None else ()),
                    *identity_params, *wanted_ranks,
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
        require_consequence_eligible: bool = False,
        require_incident_reviewed: bool = False,
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
        eligibility_clause = (
            " AND s.probation_required != 0" if require_consequence_eligible else ""
        )
        review_clause = (
            " AND w.incident_review_status != 'pending'"
            if require_incident_reviewed else ""
        )
        with self._lock:
            identity_clause, identity_params = self._capacity_address_identity_filter_locked(
                address
            )
            # Any ordinal's invalid proof makes the audit count once; invalid
            # is the top severity so no sibling verdict can mask it.
            row = self._conn.execute(
                f"""SELECT COUNT(DISTINCT s.audit_id) AS n
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
                      {eligibility_clause}
                      {review_clause}
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
        wanted_ranks = sorted(
            {
                self._CAPACITY_FAILURE_SEVERITY_RANKS[v]
                for v in verdicts
                if v in self._CAPACITY_FAILURE_SEVERITY_RANKS
            }
        )
        if not wanted_ranks:
            return 0
        rank_placeholders = ",".join("?" for _ in wanted_ranks)
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
                    FROM (
                      SELECT MAX({self._CAPACITY_FAILURE_SEVERITY_SQL}) AS worst
                      FROM capacity_audit_slots s
                      JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                      WHERE w.epoch_number >= ?
                        AND s.verdict IN ('timing_miss', 'hard_proof_miss', 'no_show')
                        AND ({identity_clause})
                        {confirmation_clause}
                      GROUP BY s.audit_id, s.miner_address, s.model_index
                    )
                    WHERE worst IN ({rank_placeholders})""",
                (int(since_epoch), *identity_params, *wanted_ranks),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def recent_capacity_failure_counts_for_uid(
        self,
        uid: int,
        *,
        since_epoch: int,
        require_chain_confirmed: bool = False,
        require_consequence_eligible: bool = False,
        require_incident_reviewed: bool = False,
        soft_failures_before_epoch: Optional[int] = None,
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
        eligibility_clause = (
            " AND s.probation_required != 0" if require_consequence_eligible else ""
        )
        review_clause = (
            " AND w.incident_review_status != 'pending'"
            if require_incident_reviewed else ""
        )
        soft_before = (
            int(soft_failures_before_epoch)
            if soft_failures_before_epoch is not None
            else None
        )
        no_show_epoch_clause = (
            " AND w.epoch_number < ?" if soft_before is not None else ""
        )
        timing_epoch_clause = (
            " AND w.epoch_number < ?" if soft_before is not None else ""
        )
        with self._lock:
            identity_clause, identity_params = self._capacity_uid_identity_filter_locked(uid)
            # Inner query: one row per audit with its worst-of severity across
            # the ordinal fan-out; outer query: per-slot failure totals. An
            # audit whose worst is timing_miss counts as timing even when a
            # sibling ordinal no-showed (timing_miss outranks no_show).
            rows = self._conn.execute(
                f"""SELECT
                        miner_address,
                        model_index,
                        SUM(invalid_flag) AS invalid_proof_failures,
                        SUM(CASE WHEN worst IN (4, 2) THEN 1 ELSE 0 END)
                            AS hard_failures,
                        SUM(CASE WHEN worst = 3 THEN 1 ELSE 0 END)
                            AS timing_failures
                    FROM (
                      SELECT
                          s.miner_address,
                          s.model_index,
                          MAX(CASE
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
                          ) AS invalid_flag,
                          MAX(CASE
                              WHEN s.verdict = 'hard_proof_miss' THEN 4
                              WHEN s.verdict = 'timing_miss'{timing_epoch_clause} THEN 3
                              WHEN s.verdict = 'no_show'{no_show_epoch_clause} THEN 2
                              ELSE 0
                          END) AS worst
                      FROM capacity_audit_slots s
                      JOIN capacity_audit_windows w ON w.audit_id = s.audit_id
                      WHERE w.epoch_number >= ?
                        AND ({identity_clause})
                        AND s.verdict IN ('hard_proof_miss', 'no_show', 'timing_miss')
                        {eligibility_clause}
                        {review_clause}
                        {confirmation_clause}
                      GROUP BY s.miner_address, s.model_index, s.audit_id
                    )
                    GROUP BY miner_address, model_index""",
                (
                    *((soft_before,) if soft_before is not None else ()),
                    *((soft_before,) if soft_before is not None else ()),
                    int(since_epoch),
                    *identity_params,
                ),
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
                       probation_required = 0,
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
            self._conn.execute(
                """UPDATE capacity_audit_history
                   SET verdict = 'timing_excused',
                       timing_status = CASE
                           WHEN timing_status IN ('miss', 'missing_final') THEN 'excused'
                           ELSE timing_status
                       END,
                       failure_reason = ?,
                       probation_required = 0,
                       slot_updated_at = ?
                   WHERE audit_id = ?
                     AND miner_address = ?
                     AND model_index = ?
                     AND verdict IN ('timing_miss', 'no_show')""",
                (
                    reason,
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
            # DISTINCT collapses per-ordinal mesh rows sharing one endpoint
            # drain; the endpoint stays drained while ANY ordinal row is.
            rows = self._conn.execute(
                """SELECT DISTINCT audit_id, miner_address, model_index, endpoint, drain_until_ts
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

        Miners reserve a local endpoint through B_proof and the complete
        nonce-derived payload deadline. The validator must not create a new
        timed obligation for the same slot in that interval, otherwise an
        honest miner can correctly skip the overlap locally while the
        validator records a no-show.
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
            "legacy_combined_proof_compatibility_accepted",
            "legacy_combined_proof_grace_accepted",
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
                        gpu_index, pass_count, epoch_number, selection_block, audit_block,
                        proof_challenge_block, verdict, timing_status, proof_status,
                        failure_reason, proof_verify_ms, probation_required,
                        probation_applied_at, pass0_received_at,
                        final_received_at, proof_received_at, slot_created_at,
                        slot_updated_at, archived_at
                    )
                    SELECT
                        s.audit_id, LOWER(s.miner_address), s.model_index, s.miner_uid,
                        s.miner_hotkey_ss58, s.endpoint, s.model_id, s.quant,
                        s.max_context_len, s.gpu_name,
                        s.gpu_count, s.vram_gb, s.group_key, s.slot_id, s.lease_id,
                        s.claimed_gpu_class, s.gpu_index, s.pass_count, w.epoch_number,
                        w.selection_block, w.audit_block, w.proof_challenge_block,
                        s.verdict, s.timing_status, s.proof_status, s.failure_reason,
                        s.proof_verify_ms, s.probation_required,
                        s.probation_applied_at, s.pass0_received_at, s.final_received_at,
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
                     AND proof_status IN (
                       'combined_proof_verified',
                       'legacy_combined_proof_compatibility_accepted',
                       'legacy_combined_proof_grace_accepted',
                       'proof_verified'
                     )""",
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
        endpoint_gate_reasons: Optional[Dict[Tuple[str, int], str]] = None,
        capacity_audit_gate_enforced: Optional[bool] = None,
        capacity_audit_gate_suppression_reason: str = "",
        uid_network_state: Optional[Dict[int, dict]] = None,
        mesh_excluded: Optional[Dict[Tuple[str, int], str]] = None,
        mesh_pinned: Optional[set] = None,
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
        endpoint_gates = {
            (str(key[0]).lower(), int(key[1])): str(reason or "")
            for key, reason in (endpoint_gate_reasons or {}).items()
            if reason
        }
        network_by_uid = {
            int(uid): dict(values or {})
            for uid, values in (uid_network_state or {}).items()
        }
        # Mesh (Sleipnir) health inputs from the validator's epoch state:
        # snapshot-verify failures (entry excluded from the epoch) and the
        # set of entries whose serving snapshot is pinned right now. Error
        # strings are already operator-facing one-liners; cap length so the
        # public payload stays bounded.
        mesh_excluded_map = {
            (str(k[0]).lower(), int(k[1])): str(v or "")[:160]
            for k, v in (mesh_excluded or {}).items()
        }
        mesh_pinned_keys = {
            (str(k[0]).lower(), int(k[1])) for k in (mesh_pinned or set())
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
                              tokens_generated, tokens_per_sec, receipt_pushed,
                              created_at
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

        # Newest registration per (uid, model_id). Registrations append
        # monotonically (a deploy to a new endpoint creates a HIGHER index),
        # so for the same model under the same UID every lower index is
        # SUPERSEDED - a stable fact, deliberately independent of the
        # oscillating is_active flag (chain discovery re-activates dead
        # orphans every epoch until snapshot-verify excludes them again,
        #). Superseded entries must never carry
        # "re-register" advice or active-looking mesh hints.
        latest_index_by_model: Dict[Tuple[int, str], int] = {}
        serving_index_by_model: Dict[Tuple[int, str], int] = {}
        for row in entry_rows:
            # MESH (gguf) lane only: append-supersedes is a property of the
            # mesh deploy protocol. The vLLM lane legitimately serves ONE
            # model on SEVERAL concurrent registrations - never supersession.
            if not str(row.get("quant") or "").startswith("gguf"):
                continue
            model_id_val = str(row.get("model_id") or "").strip()
            uid_val = row.get("bittensor_uid")
            if not model_id_val or uid_val is None:
                continue
            row_idx = int(row.get("model_index") or -1)
            map_key = (int(uid_val), model_id_val)
            if row_idx > latest_index_by_model.get(map_key, -1):
                latest_index_by_model[map_key] = row_idx
            # A registration that is demonstrably SERVING (pinned this epoch,
            # or active and recently scored) outranks a merely-newer index:
            # a failed replacement attempt must not dethrone the working
            # registration it failed to replace.
            row_addr = str(row.get("address") or "").lower()
            row_pinned = (row_addr, row_idx) in mesh_pinned_keys
            row_scored = (
                row.get("last_scored_epoch") is not None
                and int(row.get("last_scored_epoch") or 0) >= cur_epoch - 2
            )
            if int(row.get("is_active") or 0) and (row_pinned or row_scored):
                if row_idx > serving_index_by_model.get(map_key, -1):
                    serving_index_by_model[map_key] = row_idx
        # Intended registration per (uid, model): the newest SERVING one;
        # only when nothing in the group serves does the newest index win
        # (a migration in progress or a fully dead model).
        intended_index_by_model: Dict[Tuple[int, str], int] = {
            key: serving_index_by_model.get(key, latest)
            for key, latest in latest_index_by_model.items()
        }

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
                " AND w.incident_review_status != 'pending'"
            )
            with self._lock:
                gate_rows = [
                    dict(row) for row in self._conn.execute(
                    f"""SELECT
                            s.audit_id,
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
                          AND s.probation_required != 0
                          AND s.verdict IN ('hard_proof_miss', 'no_show', 'timing_miss')
                          {confirmation_clause}
                        ORDER BY w.epoch_number ASC""",
                    (gate_since_epoch,),
                ).fetchall()]
            # Collapse per-ordinal mesh rows to one worst-of row per audit so
            # the public diagnostics mirror the score gate's strike counting.
            worst_by_audit: Dict[tuple, tuple[int, dict]] = {}
            for row in gate_rows:
                dedupe_key = (
                    str(row.get("audit_id") or ""),
                    str(row.get("miner_address") or ""),
                    int(row.get("model_index") or 0),
                )
                current = worst_by_audit.get(dedupe_key)
                rank = self._CAPACITY_FAILURE_SEVERITY_RANKS.get(
                    str(row.get("verdict") or ""), 0
                ) + (
                    # Invalid proof evidence outranks every plain verdict.
                    10
                    if _debug_capacity_audit_invalid_failure(
                        str(row.get("verdict") or ""),
                        row.get("proof_status"),
                        row.get("failure_reason"),
                    )
                    else 0
                )
                if current is None or rank > current[0]:
                    worst_by_audit[dedupe_key] = (rank, row)
            for _rank, row in worst_by_audit.values():
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
                "receipt_delivery_failures": 0,
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
                proof_requested = bool(int(row.get("proof_requested") or 0))
                proof_verified = bool(int(row.get("proof_verified") or 0))
                err = (
                    row.get("error_message")
                    or row.get("proof_failure_reason")
                    or ""
                )
                error_kind = _debug_error_kind(err)
                if status == "proof_failed" and error_kind == "chat_error":
                    error_kind = "proof_failure"
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
                if proof_requested:
                    item["proof_requested"] += 1
                    if proof_verified:
                        item["proof_verified"] += 1
                    elif status in {"ok", "proof_failed"}:
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
                if status == "ok" and not int(row.get("receipt_pushed") or 0):
                    item["receipt_delivery_failures"] += 1
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
                # Audit rows carry model_id; keep the first non-empty one so
                # entries whose miner_entries row was filtered (dead orphans
                # outside the ownership window) still resolve their model
                # identity for display and supersession.
                if row.get("model_id") and not item.get("model_id"):
                    item["model_id"] = str(row.get("model_id"))
                verdict = str(row.get("verdict") or "pending")
                item["total"] += 1
                if verdict in item:
                    item[verdict] += 1
                else:
                    item["pending"] += 1
                # Failure reasons only from rows whose verdict IS a failure:
                # excused rows (receipt-overlap or validator-incident) keep a
                # descriptive failure_reason string, and counting them here
                # painted healthy executors with "capacity_audit_failed"
                # hints.
                if verdict in {"timing_miss", "hard_proof_miss", "no_show"}:
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
                    is_blacklisted = addr in blacklisted
                    model_gate_reason = model_gates.get((addr, idx), "")
                    endpoint_gate_reason = endpoint_gates.get((addr, idx), "")
                    display_score = (
                        0.0
                        if not is_active or endpoint_gate_reason
                        else (raw_ema if raw_ema > 0 else 0.01)
                    )
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
                    if endpoint_gate_reason:
                        entry_hint_codes.append("https_required")
                        add_hint(
                            hints,
                            "https_required",
                            "This executor's registered endpoint is not eligible on mainnet because public HTTPS is required.",
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
                    mesh_key = (addr, idx)
                    is_mesh_entry = bool(
                        str((db_entry or {}).get("quant") or "").startswith("gguf")
                    )
                    _model_id_val = (
                        str((db_entry or {}).get("model_id") or "").strip()
                        or str((audit_data or {}).get("model_id") or "").strip()
                    )
                    _latest_idx = (
                        intended_index_by_model.get((uid, _model_id_val))
                        if _model_id_val else None
                    )
                    mesh_excluded_error = mesh_excluded_map.get(mesh_key, "")
                    mesh_pinned_now = mesh_key in mesh_pinned_keys
                    # A SERVING entry is never superseded, whatever its index:
                    # the same model legitimately scales across several
                    # concurrent registrations (multiple pools/workers under
                    # one UID). Supersession only ever labels entries that
                    # are NOT serving while a newer serving (or, if none
                    # serves, newer registered) sibling exists.
                    entry_serving = mesh_pinned_now or (
                        is_active
                        and score_data.get("last_scored_epoch") is not None
                        and int(score_data.get("last_scored_epoch") or 0)
                        >= cur_epoch - 2
                    )
                    is_superseded = (
                        _latest_idx is not None
                        and _latest_idx > idx
                        and not entry_serving
                    )
                    is_stalled_replacement = (
                        _latest_idx is not None
                        and _latest_idx < idx
                        and not entry_serving
                    )
                    if is_superseded:
                        entry_hint_codes.append("superseded_by_active_entry")
                    if is_stalled_replacement:
                        entry_hint_codes.append("replacement_not_serving")
                        add_hint(
                            hints,
                            "replacement_not_serving",
                            "A newer registration exists for a model whose older registration is still the one serving.",
                        )
                    if mesh_excluded_error and not is_superseded:
                        entry_hint_codes.append("mesh_snapshot_excluded")
                        add_hint(
                            hints,
                            "mesh_snapshot_excluded",
                            "The validator could not verify this mesh's serving snapshot this epoch; the entry is excluded until the pin succeeds.",
                        )
                    elif (
                        is_mesh_entry
                        and is_active
                        and not mesh_pinned_now
                        and not is_superseded
                        and not is_stalled_replacement
                    ):
                        entry_hint_codes.append("mesh_not_pinned")
                        add_hint(
                            hints,
                            "mesh_not_pinned",
                            "This mesh entry has no pinned verification snapshot for the current epoch yet, so it is not being routed.",
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
                        error_kind = str(canary_data["last_error_kind"])
                        add_hint(
                            hints,
                            error_kind,
                            _debug_public_error_summary(error_kind),
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
                    if int(canary_data.get("receipt_delivery_failures") or 0):
                        entry_hint_codes.append("receipt_delivery_failed")
                        add_hint(
                            hints,
                            "receipt_delivery_failed",
                            "A successful canary result could not be delivered to the miner's receipt endpoint.",
                        )
                    if (
                        gate_data["hard_failures"]
                        or gate_data["timing_failures"]
                        or gate_data["invalid_proof_failures"]
                    ) and not is_superseded:
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
                    if endpoint_gate_reason:
                        add_step(entry_next_steps, endpoint_gate_reason)
                    if is_superseded:
                        add_step(
                            entry_next_steps,
                            f"Superseded by the newer registration at model "
                            f"index {_latest_idx}; the stale lease expires "
                            f"on its own - no action needed.",
                        )
                    elif is_stalled_replacement:
                        add_step(
                            entry_next_steps,
                            f"This newer registration is not serving while "
                            f"model index {_latest_idx} still is; bring this "
                            f"endpoint up or let its lease lapse.",
                        )
                    elif not is_active:
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
                            _debug_error_next_step(
                                canary_data["last_error_kind"]
                            ),
                        )
                    if int(canary_data.get("proof_failures") or 0):
                        add_step(
                            entry_next_steps,
                            _debug_error_next_step("proof_failure"),
                        )
                    if int(canary_data.get("tee_failures") or 0):
                        add_step(
                            entry_next_steps,
                            _debug_error_next_step("tee_failure"),
                        )
                    if int(canary_data.get("receipt_delivery_failures") or 0):
                        add_step(
                            entry_next_steps,
                            "Check the miner receipt-ingest route and validator authorization; inference itself succeeded.",
                        )
                    if not is_superseded:
                        # A superseded corpse's audit history is data, not a
                        # to-do list: action steps belong to the serving
                        # registration only.
                        for failure_reason in sorted(
                            (audit_data.get("failure_reasons") or {}).keys()
                        ):
                            add_step(
                                entry_next_steps,
                                _debug_capacity_audit_next_step(failure_reason),
                            )
                    if is_probation:
                        add_step(
                            entry_next_steps,
                            f"Keep this executor clean for {probation_remaining} more probation pass(es).",
                        )
                    if gate_status.get("active") and not is_superseded:
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
                    ) and not is_superseded:
                        add_step(
                            entry_next_steps,
                            "Recent audit failures are present but below the active gate threshold; keep this executor clean.",
                        )

                    entries.append({
                        "address": addr,
                        "model_index": idx,
                        "endpoint": db_entry.get("endpoint") if db_entry else "",
                        "model_id": _model_id_val,
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
                        "endpoint_gate": {
                            "active": bool(endpoint_gate_reason),
                            "reason": endpoint_gate_reason,
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
                            "source": (
                                str(db_entry.get("probation_source") or "")
                                if db_entry else ""
                            ),
                        },
                        "canary": canary_data,
                        "receipts": receipt_data,
                        "last_score": score_data,
                        "capacity_audit": {
                            **audit_data,
                            "gate_counts": gate_data,
                            "gate_status": gate_status,
                        },
                        "mesh": {
                            "is_mesh": is_mesh_entry,
                            "snapshot_pinned": mesh_pinned_now,
                            "snapshot_error": mesh_excluded_error,
                        },
                        "issue_codes": (
                            # INVARIANT BY CONSTRUCTION: a superseded corpse
                            # carries exactly one code and exactly one step.
                            # Its history (probation dict, gate counts, score)
                            # stays visible as DATA below, but no to-do or
                            # alarm code may ever leak onto it - whatever
                            # future step/hint sections get added above.
                            ["superseded_by_active_entry"]
                            if is_superseded else entry_hint_codes
                        ),
                        "next_steps": (
                            [
                                f"Superseded by the newer registration at "
                                f"model index {_latest_idx}; the stale lease "
                                f"expires on its own - no action needed."
                            ]
                            if is_superseded else entry_next_steps
                        ),
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
                uid_gate_enabled = bool(
                    gate_enabled
                    and getattr(capacity_audit_cfg, "uid_escalation_enabled", False)
                )
                quorum = (
                    capacity_audit_uid_escalation_threshold(entry_count, capacity_audit_cfg)
                    if uid_gate_enabled and entry_count > 0
                    else 0
                )
                uid_gate_active = bool(
                    uid_gate_enabled
                    and entry_count > 1
                    and len(convicted) >= quorum
                )
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
                    "https_required",
                    "mesh_snapshot_excluded",
                    "mesh_not_pinned",
                    "stale_uid_identity",
                    "reverse_proxy_timeout",
                    "first_token_timeout",
                    "rate_limited",
                    "bad_request",
                    "service_unavailable",
                    "empty_response",
                    "proof_failure",
                    "tee_failure",
                    "chat_unauthorized",
                    "chat_forbidden",
                    "chat_not_found",
                    "timeout",
                    "dns_error",
                    "connection_failed",
                    "tls_error",
                    "chat_error",
                    "receipt_delivery_failed",
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
                        "configured": bool(
                            gate_configured
                            and getattr(
                                capacity_audit_cfg,
                                "uid_escalation_enabled",
                                False,
                            )
                        ),
                        "enabled": uid_gate_enabled,
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
        latest_score_samples = self._get_latest_score_samples()
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
        miner_ema_scores: Dict[str, Dict[str, float]] = {}
        miner_score_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for e in all_entries:
            addr = e["address"]
            idx_str = str(e["model_index"])
            if addr not in miner_scores:
                miner_scores[addr] = {}
                miner_ema_scores[addr] = {}
                miner_score_metadata[addr] = {}
            raw_ema = float(e["ema_score"])
            miner_ema_scores[addr][idx_str] = raw_ema
            last_scored_epoch = e.get("last_scored_epoch")
            miner_score_metadata[addr][idx_str] = {
                "scored_epochs": max(0, int(e.get("scored_epochs") or 0)),
                "last_scored_epoch": (
                    max(0, int(last_scored_epoch))
                    if last_scored_epoch is not None
                    else None
                ),
            }
            sample = latest_score_samples.get((addr, int(e["model_index"])))
            if (
                sample is not None
                and last_scored_epoch is not None
                and int(sample["score_epoch"]) == int(last_scored_epoch)
            ):
                metadata = miner_score_metadata[addr][idx_str]
                metadata.update(
                    {
                        "score_epoch": int(sample["score_epoch"]),
                        "latest_completed_ema": float(sample["ema_score"]),
                        "ema_scope": "coordinator_model_slot",
                        "ema_carries_across_topology": True,
                    }
                )
                if sample.get("mesh_id") is not None:
                    metadata["latest_mesh_sample"] = {
                        "chain_id": int(sample["chain_id"]),
                        "netuid": int(sample["netuid"]),
                        "coordinator_address": addr,
                        "model_index": int(e["model_index"]),
                        "model_id": str(sample["model_id"]),
                        "mesh_id": str(sample["mesh_id"]),
                        "verification_snapshot_hash": str(
                            sample["snapshot_hash"]
                        ),
                        "snapshot_generation": int(
                            sample["snapshot_generation"]
                        ),
                    }
            score = raw_ema
            if not e.get("is_active"):
                score = 0.0
            elif score == 0:
                score = 0.01  # active but not yet scored — allow routing
            miner_scores[addr][idx_str] = score

        # Use only this validator's accepted receipts.  This is the same
        # validator-observed timing source used by scoring, without trusting a
        # miner-reported TPS value.  Very short completions are excluded
        # because setup/TTFT dominates them and makes decode speed noisy.
        recent_tps: Dict[Tuple[str, int], List[float]] = {}
        recent_ttft: Dict[Tuple[str, int], List[float]] = {}
        with self._lock:
            tps_rows = self._conn.execute(
                """SELECT LOWER(miner_address) AS miner_address,
                          model_index, tokens_generated,
                          generation_time_ms, ttft_ms
                   FROM network_receipts
                   WHERE epoch_number BETWEEN ? AND ?
                     AND is_own = 1
                     AND tokens_generated >= 32
                     AND generation_time_ms > ttft_ms
                     AND (proof_requested = 0 OR proof_verified = 1)""",
                (max(0, int(epoch) - 2), int(epoch)),
            ).fetchall()
        for row in tps_rows:
            key = (str(row["miner_address"]).lower(), int(row["model_index"]))
            output_tokens = int(row["tokens_generated"])
            decode_ms = float(row["generation_time_ms"]) - float(row["ttft_ms"])
            if output_tokens <= 1 or decode_ms <= 0.0:
                continue
            recent_tps.setdefault(key, []).append(
                (output_tokens - 1) / (decode_ms / 1000.0)
            )
            recent_ttft.setdefault(key, []).append(float(row["ttft_ms"]))
        miner_tps: Dict[str, Dict[str, float]] = {}
        for (addr, model_index), values in recent_tps.items():
            miner_tps.setdefault(addr, {})[str(model_index)] = float(
                statistics.median(values)
            )
        miner_ttft_ms: Dict[str, Dict[str, float]] = {}
        for (addr, model_index), values in recent_ttft.items():
            miner_ttft_ms.setdefault(addr, {})[str(model_index)] = float(
                statistics.median(values)
            )

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
            miner_ema_scores=miner_ema_scores,
            miner_score_metadata=miner_score_metadata,
            miner_tps=miner_tps,
            miner_ttft_ms=miner_ttft_ms,
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

    def store_local_service_receipt(self, receipt: dict) -> None:
        """Persist one validator-owned signed receipt before peer transport."""

        if not isinstance(receipt, dict):
            raise TypeError("local service receipt must be a mapping")
        epoch_number = int(receipt["epoch_number"])
        model_index = int(receipt["model_index"])
        miner_address = str(receipt["miner_address"]).lower()
        signature = str(receipt["validator_signature"]).lower()
        if epoch_number < 0 or model_index < 0 or not miner_address:
            raise ValueError("local service receipt identity is invalid")
        if len(signature) != 128:
            raise ValueError("local service receipt signature is invalid")
        try:
            bytes.fromhex(signature)
        except ValueError as exc:
            raise ValueError(
                "local service receipt signature is invalid"
            ) from exc
        encoded = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO local_service_receipts (
                    epoch_number, validator_signature, miner_address,
                    model_index, receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    epoch_number,
                    signature,
                    miner_address,
                    model_index,
                    encoded,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_local_service_receipts(self, epoch_number: int) -> List[dict]:
        """Return canonical validator-owned receipts for one epoch."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT receipt_json
                   FROM local_service_receipts
                   WHERE epoch_number = ?
                   ORDER BY created_at ASC, validator_signature ASC""",
                (int(epoch_number),),
            ).fetchall()
        receipts: List[dict] = []
        for row in rows:
            value = json.loads(row["receipt_json"])
            if not isinstance(value, dict):
                raise ValueError("local service receipt row is malformed")
            receipts.append(value)
        return receipts

    def gc_local_service_receipts(self, minimum_epoch: int) -> int:
        """Delete locally-owned receipts older than the retained epoch floor."""

        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM local_service_receipts WHERE epoch_number < ?",
                (max(0, int(minimum_epoch)),),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount or 0))

    def record_planned_canary_obligations(
        self,
        epoch_number: int,
        rows: List[dict],
    ) -> None:
        """Journal one plan's canary obligations for an epoch.

        Idempotent per (epoch, obligation_id); successive plans of the same
        epoch (mid-epoch restart re-plan) accumulate rather than replace, so
        the epoch close can validate receipts against every plan this
        validator actually dispatched from.
        """

        epoch_number = int(epoch_number)
        if epoch_number < 0:
            raise ValueError("planned canary obligation epoch is invalid")
        encoded: List[tuple] = []
        for row in rows:
            obligation_id = str(row["obligation_id"]).lower()
            if len(obligation_id) != 32:
                raise ValueError("planned canary obligation id is malformed")
            bytes.fromhex(obligation_id)
            kind = str(row["kind"])
            if kind not in ("low", "full"):
                raise ValueError("planned canary obligation kind is invalid")
            encoded.append(
                (
                    epoch_number,
                    obligation_id,
                    str(row["miner_address"]).lower(),
                    int(row["model_index"]),
                    kind,
                    int(row["target_prompt_tokens"]),
                    time.time(),
                )
            )
        if not encoded:
            return
        with self._lock:
            self._conn.executemany(
                """INSERT OR IGNORE INTO planned_canary_obligations (
                    epoch_number, obligation_id, miner_address,
                    model_index, kind, target_prompt_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                encoded,
            )
            self._conn.commit()

    def get_planned_canary_obligations(
        self,
        epoch_number: int,
    ) -> List[dict]:
        """Return every journaled canary obligation for one epoch."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT obligation_id, miner_address, model_index,
                          kind, target_prompt_tokens
                   FROM planned_canary_obligations
                   WHERE epoch_number = ?
                   ORDER BY created_at ASC, obligation_id ASC""",
                (int(epoch_number),),
            ).fetchall()
        return [
            {
                "obligation_id": str(row["obligation_id"]),
                "miner_address": str(row["miner_address"]),
                "model_index": int(row["model_index"]),
                "kind": str(row["kind"]),
                "target_prompt_tokens": int(row["target_prompt_tokens"]),
            }
            for row in rows
        ]

    def gc_planned_canary_obligations(self, minimum_epoch: int) -> int:
        """Delete journaled obligations older than the retained epoch floor."""

        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM planned_canary_obligations WHERE epoch_number < ?",
                (max(0, int(minimum_epoch)),),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount or 0))

    def store_proof_v3_hard_failure(
        self,
        outcome: dict,
        outcome_digest: bytes,
    ) -> None:
        """Persist one canonical owner-signed hard-audit failure."""

        if not isinstance(outcome, dict) or len(outcome_digest) != 32:
            raise ValueError("proof-v3 hard failure is malformed")
        source_epoch = int(outcome["source_epoch"])
        miner_address = str(outcome["miner_address"]).lower()
        model_index = int(outcome["model_index"])
        encoded = json.dumps(
            outcome,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO proof_v3_hard_failures (
                    outcome_digest, source_epoch, miner_address,
                    model_index, outcome_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    outcome_digest.hex(),
                    source_epoch,
                    miner_address,
                    model_index,
                    encoded,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_proof_v3_hard_failures(
        self,
        minimum_epoch: int,
        maximum_epoch: int,
    ) -> List[dict]:
        """Return signed hard-audit failures in canonical database order."""

        lower = max(0, int(minimum_epoch))
        upper = max(lower, int(maximum_epoch))
        with self._lock:
            rows = self._conn.execute(
                """SELECT outcome_json
                   FROM proof_v3_hard_failures
                   WHERE source_epoch BETWEEN ? AND ?
                   ORDER BY source_epoch ASC, outcome_digest ASC""",
                (lower, upper),
            ).fetchall()
        values: List[dict] = []
        for row in rows:
            value = json.loads(row["outcome_json"])
            if not isinstance(value, dict):
                raise ValueError("proof-v3 hard failure row is malformed")
            values.append(value)
        return values

    def gc_proof_v3_hard_failures(self, minimum_epoch: int) -> int:
        """Delete signed hard failures older than the public retention floor."""

        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM proof_v3_hard_failures WHERE source_epoch < ?",
                (max(0, int(minimum_epoch)),),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount or 0))

    def backup_and_cleanup_canary_results(self, retain_days: int = 7, backup_dir: str = "") -> int:
        """Export old canary_results to .jsonl.gz, then delete from live DB."""
        return self._backup_and_cleanup_analytics_table(
            table="canary_results",
            file_prefix="canary_results",
            log_label="Canary results",
            retain_days=retain_days,
            backup_dir=backup_dir,
        )

    def backup_and_cleanup_network_receipts(self, retain_days: int = 7, backup_dir: str = "") -> int:
        """Export old network_receipts to .jsonl.gz, then delete from live DB."""
        return self._backup_and_cleanup_analytics_table(
            table="network_receipts",
            file_prefix="network_receipts",
            log_label="Network receipts",
            retain_days=retain_days,
            backup_dir=backup_dir,
        )

    def _backup_and_cleanup_analytics_table(
        self,
        *,
        table: str,
        file_prefix: str,
        log_label: str,
        retain_days: int,
        backup_dir: str,
    ) -> int:
        """Stream one append-only analytics table into an atomic gzip archive.

        Analytics tables can contain millions of rows.  A separate read
        connection keeps normal validator writes moving under WAL while rows
        are encoded one at a time; the previous ``fetchall()`` retained every
        Python row object and could exhaust validator host memory.  Deletion is
        bounded by the captured maximum primary key, so rows appended during
        the archive are never removed.
        """
        import gzip
        import json as _json

        if table not in {"canary_results", "network_receipts"}:
            raise ValueError(f"unsupported analytics archive table: {table}")

        cutoff = time.time() - (retain_days * 86400)
        if not backup_dir:
            backup_dir = os.path.join(
                os.environ.get("VERALLM_DATA_DIR", os.path.expanduser("~/.verathos")), "backups",
            )
        os.makedirs(backup_dir, exist_ok=True)

        read_conn = sqlite3.connect(self._db_path)
        read_conn.row_factory = sqlite3.Row
        read_conn.execute("PRAGMA query_only=ON")
        tmp_path = ""
        backup_path = ""
        archived = 0
        max_id = 0
        try:
            max_row = read_conn.execute(
                f"SELECT MAX(id) AS max_id FROM {table} WHERE created_at < ?",
                (cutoff,),
            ).fetchone()
            max_id = int(max_row["max_id"] or 0) if max_row is not None else 0
            if max_id <= 0:
                return 0

            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(backup_dir, f"{file_prefix}_{ts}.jsonl.gz")
            tmp_path = (
                f"{backup_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            cursor = read_conn.execute(
                f"""SELECT * FROM {table}
                    WHERE created_at < ? AND id <= ?
                    ORDER BY id ASC""",
                (cutoff, max_id),
            )
            col_names = [desc[0] for desc in cursor.description]
            with gzip.open(tmp_path, "wt") as output:
                while True:
                    row = cursor.fetchone()
                    if row is None:
                        break
                    output.write(
                        _json.dumps(dict(zip(col_names, row)), default=str) + "\n"
                    )
                    archived += 1
            if archived <= 0:
                os.remove(tmp_path)
                return 0
            os.replace(tmp_path, backup_path)
            tmp_path = ""
        except Exception:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
            raise
        finally:
            read_conn.close()

        # The archive is durable and visible before any live rows are deleted.
        # If deletion fails, the rows remain and a later run may create a
        # duplicate archive, but analytics data is never silently lost.
        while True:
            with self._lock:
                deleted = self._conn.execute(
                    f"""DELETE FROM {table}
                        WHERE id IN (
                            SELECT id FROM {table}
                            WHERE created_at < ? AND id <= ?
                            ORDER BY id ASC
                            LIMIT 5000
                        )""",
                    (cutoff, max_id),
                ).rowcount
                self._conn.commit()
            if int(deleted or 0) < 5000:
                break
        bt.logging.info(f"{log_label} backup: {archived} rows → {backup_path}")
        return archived

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        with self._capacity_ingress_lock:
            if self._capacity_ingress_conn:
                self._capacity_ingress_conn.close()
                self._capacity_ingress_conn = None
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
