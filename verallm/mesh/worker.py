"""Lightweight private-mesh worker control plane.

This module deliberately uses only the Python standard library. It is meant to
run on Linux, macOS, and Windows before the native GGUF runtime exists.
"""

from __future__ import annotations

import json
import hashlib
import ipaddress
import logging
import math
import os
import re
import sys
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from verallm.mesh.llama_cpp import rpc_plan_from_mesh
from verallm.mesh.http_auth import (
    DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
    RequestReplayCache,
    ValidatorAllowlist,
    sign_internal_http_request,
    verify_internal_http_request,
    verify_validator_http_request,
)
from verallm.mesh.proof import (
    LLAMA_CPP_RPC_RECEIPT_PROOF_MODE,
    PROOF_SAMPLE_BPS_DENOMINATOR,
    RECEIPT_ONLY_PROOF_MODE,
    REPLAY_SEED_MODE_CLIENT,
    REPLAY_SEED_MODE_DERIVED,
    SLOT_VIEW_PROBE_MAX_ROWS,
    VALIDATOR_POSTCOMMIT_CHALLENGE_KIND,
    VERATHOS_GGML_GEMM_PROOF_MODE,
    VERATHOS_GGML_LIGHT_PROOF_MODE,
    VERATHOS_GGML_TRACE_PROOF_MODE,
    VERATHOS_GGUF_DECODE_AUDIT_MODE,
    VERATHOS_GGUF_DECODE_AUDIT_TOP_K,
    LlamaGraphOpReceipt,
    MeshStageProofReceipt,
    derive_mesh_deferred_audit_beacon,
    derive_mesh_postcommit_proof_beacon,
    derive_mesh_proof_beacon,
    derive_mesh_replay_seed,
    derive_mesh_decode_audit_positions,
    llama_graph_receipt_root_hex,
    mesh_stage_proof_receipt_root_hex,
    mesh_deferred_audit_commitment_hash,
    mesh_deferred_audit_sample_commitment_hash,
    mesh_deferred_audit_sample_value,
    mesh_validator_challenge_nonce_commitment,
    mesh_decode_audit_commitment_hash,
    mesh_decode_audit_sample_value,
    mesh_proof_gate_hash,
    mesh_proof_sample_value,
    mesh_receipt_hash,
    mesh_response_commitment_hash,
    normalize_deferred_randomness,
    normalize_proof_sample_bps,
    normalize_validator_challenge_nonce,
    normalize_validator_nonce,
    should_sample_mesh_deferred_audit,
    should_sample_mesh_decode_audit,
    should_sample_mesh_proof,
    verify_llama_graph_proof_receipts,
    verify_llama_graph_proof_receipts_for_mesh,
    verify_mesh_stage_proof_receipts_for_snapshot,
)
from verallm.mesh.types import CapabilityAd, MeshMember, MeshSpec


logger = logging.getLogger(__name__)

DEFERRED_AUDIT_BUNDLE_KIND = "verathos_mesh_deferred_audit_bundle_v1"
POSTCOMMIT_AUDIT_PATH = "/v1/mesh/proof/postcommit-audit"
# Backend exhaustion signatures for the defensive 503 mapping (B7):
# structurally unreachable for ledger-admitted requests, but a backend
# that still reports one must never surface as an unretryable 500.
_BACKEND_EXHAUSTION_RE = re.compile(
    r"Context size has been exceeded|no (?:idle )?slot"
    r"|all slots are (?:busy|processing)|KV cache is full",
    re.IGNORECASE,
)
POSTCOMMIT_ORIGIN_TTL_SECONDS = 15 * 60.0

# Capture artifacts older than every legitimate postcommit draw are pure
# scan cost: find_traces_for_window and the window-token lookup glob the
# whole trace dir per proof, so an unpruned dir makes every proof slower
# the longer the box runs (11k files after two hours of health probes,
# measured as a steadily growing proof tail).
PROOF_TRACE_RETENTION_SECONDS = POSTCOMMIT_ORIGIN_TTL_SECONDS + 300.0
# Filename tokens are window-open times in ns; values this small are
# sentinels (the warmup trace uses token 1), never real timestamps.
_PROOF_TRACE_MIN_REAL_NS = 1_000_000_000_000_000_000


def tail_group_position_offset(
    group: list[dict[str, Any]],
    *,
    positions: list[int],
    committed_ids: list[int],
    top_k: int,
) -> int | None:
    """The seq offset under which this flush group covers every audited
    position, or None when no candidate offset does.

    The group is one generation's trailing final-logit instances in decode
    order, so position ``p`` sits at ``tail_seq = p + offset``. The offset
    depends on how the reply ended: a natural stop samples the stop token
    from one extra instance past the last committed position (offset =
    total - count - 1, the common case), while a length-capped reply's last
    instance IS the last committed position (offset = total - count). Both
    are tried and validated by the same acceptance the verifier enforces on
    the opened row: the committed token must be inside the row's top-k.
    A wrong offset opens the WRONG token's logits (observed: every
    EOS-terminated audit draw missed under the capped-only mapping).
    """

    import numpy as np

    if not group:
        return None
    total = int(group[0].get("tail_total", 0) or 0)
    if total <= 0:
        return None
    by_seq: dict[int, dict[str, Any]] = {}
    for item in group:
        if int(item.get("tail_total", -1)) != total:
            return None
        by_seq[int(item.get("tail_seq", -1))] = item
    count = len(committed_ids)
    if count <= 0:
        return None

    def _row_accepts(item: dict[str, Any], token: int) -> bool:
        dst_name = str(item.get("dst_f32", ""))
        if not dst_name:
            return False
        try:
            row = np.fromfile(
                str(Path(str(item["_tail_path"])).parent / dst_name),
                dtype=np.float32,
            )
        except Exception:
            return False
        if row.size <= 0:
            return False
        k = max(1, min(int(top_k), int(row.size)))
        part = np.argpartition(row, -k)[-k:]
        return int(token) in {int(v) for v in part.tolist()}

    for offset in (total - count - 1, total - count):
        covered = True
        for position in positions:
            if not 0 <= int(position) < count:
                return None
            item = by_seq.get(int(position) + offset)
            if item is None or not _row_accepts(
                item, int(committed_ids[int(position)])
            ):
                covered = False
                break
        if covered:
            return offset
    return None


def tail_group_covers_positions(
    group: list[dict[str, Any]],
    *,
    positions: list[int],
    committed_ids: list[int],
    top_k: int,
) -> bool:
    """True when a tail-ring flush group holds every audited position's row
    under some valid seq offset (see tail_group_position_offset)."""

    return (
        tail_group_position_offset(
            group,
            positions=positions,
            committed_ids=committed_ids,
            top_k=top_k,
        )
        is not None
    )


def prewarm_slot_view_caches(
    template: list[dict[str, Any]],
    *,
    target_tokens: int,
    chunk_tokens: int = 64,
    chunk_sleep_s: float = 0.01,
    journal: "Callable[[str], None] | None" = None,
) -> None:
    """Warm the slot-view leaf/levels caches up to ``target_tokens``.

    Phase C makes leaf-set EXTENSIONS O(new + log n), but the first build at
    a novel completion length still hashes every leaf below it (measured
    1.5-1.8 s at 1500 tokens on the fleet templates). Warming once per
    template to the ceiling turns every runtime length into a prefix hit.
    Chunked so the levels builder's memo-lock holds stay short next to live
    receipt assembly; runs synchronously - callers wanting a background warm
    wrap it in a thread.
    """

    from verallm.mesh.ggml_proof import ggml_slot_view_root_from_template

    target = int(target_tokens)
    if target <= 0 or not template:
        return
    step = max(1, int(chunk_tokens))
    started = time.monotonic()
    count = 0
    while count < target:
        count = min(count + step, target)
        ggml_slot_view_root_from_template(
            template=template,
            completion_token_count=count,
        )
        if chunk_sleep_s > 0 and count < target:
            time.sleep(chunk_sleep_s)
    if journal is not None:
        journal(
            f"slot-view prewarm tokens={target} ops={len(template)}"
            f" ms={int((time.monotonic() - started) * 1000)}"
        )


def prune_stale_proof_traces(
    root: str | Path, *, retention_s: float = PROOF_TRACE_RETENTION_SECONDS
) -> int:
    """Delete capture artifacts too old for any legitimate audit draw."""

    cutoff_ns = (time.time() - retention_s) * 1e9
    removed = 0
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return 0
    for path in entries:
        name = path.name
        if not (name.startswith("trace-") or name.startswith("manifest-")):
            continue
        parts = path.stem.split("-", 2)
        try:
            ts = int(parts[1])
        except (IndexError, ValueError):
            continue
        if ts < _PROOF_TRACE_MIN_REAL_NS or ts >= cutoff_ns:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
# These caps apply only to unfinished phase-one origins, including an active
# phase-two finalization.  Completed replay artifacts must not consume them.
POSTCOMMIT_ORIGIN_MAX_ENTRIES = 4096
POSTCOMMIT_ORIGIN_MAX_ENTRIES_PER_PRINCIPAL = 256
# Pending origins include the request, response, token bindings, pinned mesh
# specification, and pinned verification snapshot.  Entry limits alone are not
# a memory bound: the public HTTP limit permits much larger bodies than a valid
# 32k-token request normally needs.  A 16 MiB single-origin ceiling leaves
# ample room for 32k prompt/completion token-id arrays and JSON text while
# rejecting pathological bodies. The byte budget alone permits four such
# origins per validator and sixteen globally; worst-case proof-replay
# reservations below impose the tighter admission bound when applicable.
POSTCOMMIT_ORIGIN_MAX_BYTES = 256 * 1024 * 1024
POSTCOMMIT_ORIGIN_MAX_BYTES_PER_PRINCIPAL = 64 * 1024 * 1024
POSTCOMMIT_ORIGIN_MAX_SINGLE_BYTES = 16 * 1024 * 1024
POSTCOMMIT_FINALIZATION_MAX_ATTEMPTS = 3
# At most the number of worst-case replies covered by the global/principal
# replay byte ceilings may be in flight at once.
POSTCOMMIT_PHASE2_MAX_ACTIVE = 4
POSTCOMMIT_PHASE2_MAX_ACTIVE_PER_PRINCIPAL = 2
# Proof collection plus canonicalization can transiently hold both the proof
# object graph and up to 128 MiB of encoded output, so serialize finalization.
POSTCOMMIT_FINALIZATION_MAX_ACTIVE = 1
POSTCOMMIT_FINALIZATION_MAX_ACTIVE_PER_PRINCIPAL = 1
POSTCOMMIT_REQUEST_MAX_BODY_BYTES = 16 * 1024
# Successful phase-two artifacts are immutable transport replay state, not
# pending work.  Phase one reserves their worst-case space up front, and
# unexpired artifacts are never evicted, so a lost response remains exactly
# replayable for the origin's original lifetime.
POSTCOMMIT_FINALIZED_MAX_ENTRIES = 1024
POSTCOMMIT_FINALIZED_MAX_ENTRIES_PER_PRINCIPAL = 256
POSTCOMMIT_FINALIZED_MAX_BYTES = 512 * 1024 * 1024
POSTCOMMIT_FINALIZED_MAX_BYTES_PER_PRINCIPAL = 256 * 1024 * 1024
POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
# Pending origins reserve a REALISTIC artifact bound, not the 128MB
# single-artifact ceiling. Worst-case reservations allowed only TWO
# overlapping pending origins per validator (2 x 128MB = the whole
# per-principal budget), and an honest validator's fast-epoch cadence -
# several full-context canaries plus retries inside one 15-minute origin
# TTL - hit the wall: the THIRD canary aborted mid-stream with
# "postcommit replay byte reservation reached" AFTER serving 11.7k
# tokens, scored the miner zero, and sent a freshly registered index to
# probation. Real artifacts run
# single-digit MB, POSTCOMMIT_FINALIZATION_MAX_ACTIVE_PER_PRINCIPAL is
# 1 so at most one artifact materialises at a time, and finalize-time
# checks re-validate REAL sizes against every hard cap above - the
# worst-case pending reservation was triple-insurance that broke honest
# traffic. 32MB permits 8 overlapping pendings per principal within the
# unchanged 256MB budget.
POSTCOMMIT_PENDING_RESERVATION_BYTES = 32 * 1024 * 1024


class _PostcommitFinalizationInProgress(RuntimeError):
    """Signal that an exact authenticated retry should be attempted later."""


class _PostcommitCapacityUnavailable(RuntimeError):
    """Signal bounded phase-two backpressure without consuming an attempt."""


# A validator pins one verification snapshot for a whole epoch.  Rejecting its
# request because the coordinator now runs a different snapshot is a refusal to
# be verified under the pinned terms, not a transient outage, so it is reported
# with a distinguishable non-retryable code that the validator prices as a
# proof failure.  Honest snapshot rotation belongs at an epoch boundary.
VERIFICATION_SNAPSHOT_MISMATCH_ERROR_CODE = "verification_snapshot_mismatch"


class _VerificationSnapshotMismatch(RuntimeError):
    """Signal that the request was pinned to a snapshot this node does not run."""


class _CaptureWindowBusy(RuntimeError):
    """The capture window did not drain within the caller's wait budget.

    Mapped by the HTTP layer to the SAME retryable slots_busy refusal the
    admission ledger sends: a request refused because an exclusive replay
    holds the capture window must be indistinguishable from one refused
    because the generation slots are saturated (canary-oracle rule)."""


JoinHandler = Callable[[dict[str, Any]], dict[str, Any]]
MeshSpecHandler = Callable[[dict[str, Any]], dict[str, Any]]
MeshUpdateHandler = Callable[[dict[str, Any]], dict[str, Any]]
MeshSpecLoader = Callable[[], MeshSpec | None]
VerificationSnapshotLoader = Callable[[], Any]


DEFAULT_WORKER_PORT = 9338
# A stage's proof endpoint SERIALIZES behind the exclusive replay window, so
# a request's proof call can legitimately queue behind another request's hard
# audit (roughly 50 s per hard proof on a representative mesh). At 4-way
# concurrency a 120 s budget expired while the stage was healthy and simply
# busy, failing honest requests ("proof endpoint failed ... timed out": 6 of
# 1666 soak requests). Budget several queued hard audits instead.
DEFAULT_PROOF_ARTIFACT_TIMEOUT = 420.0
# How long a stage waits for an exclusive capture window to drain before it
# refuses a shared join (server side, acquire_shared_capture).
SHARED_CAPTURE_WAIT_S = 60.0
# Client deadline for arming a shared window: must exceed the server's own
# wait, or the caller gives up on a stage that is still willing to serve it.
SHARED_CAPTURE_ARM_TIMEOUT_S = SHARED_CAPTURE_WAIT_S + 15.0
# ORGANIC chats wait out queued hard replays instead of erroring at 60 s: a
# hard audit's exclusive window can run roughly 50 s (worse on
# slower boxes, worse when audits queue), so the 60 s shared-join budget
# expired on healthy meshes and every overlapped chat died after a long
# silent stall. The product
# rule is "slower is fine, dead is not": organic streams stay visibly
# alive during the wait (SSE keepalive comments, below), so the only cost
# of a longer budget is latency the client can watch and cancel. Validator
# canaries keep the short wait - they already price a busy mesh through
# the retryable-503 busy machinery, and holding their nonce open longer
# only delays that verdict.
ORGANIC_CAPTURE_WAIT_S = float(
    os.environ.get("VERATHOS_MESH_ORGANIC_CAPTURE_WAIT_S", "300") or 300.0
)
# Cadence of liveness frames while a stream waits on the capture window.
# SSE comment frames are ignored by every consumer of this stream (the
# proxy, the validator canary reader, the mesh chat CLI all skip ":"
# lines) but they keep each hop's read loop fed so the wait can never be
# mistaken for a dead stream. Emitted on a generic slow-path timer, never
# labeled with a cause: a keepalive during an audit drain is byte-equal
# to one during any other slow phase (canary-oracle rule).
CAPTURE_WAIT_TICK_S = float(
    os.environ.get("VERATHOS_MESH_CAPTURE_WAIT_TICK_S", "10") or 10.0
)
SSE_KEEPALIVE_FRAME = b": keepalive\n\n"
# Backend forward timeout must cover long-context prefill (a 131k-token
# prompt can prefill for minutes). Overridable for constrained setups.
BACKEND_FORWARD_TIMEOUT_S = float(
    os.environ.get("VERATHOS_MESH_BACKEND_TIMEOUT_S", "1800")
)
# Per-connection socket timeout for the serve's HTTP handler threads.
# Bounds each individual read/write on the CLIENT socket, never total
# request duration: a long prefill spends its minutes waiting on the
# BACKEND socket (bounded above), and an SSE stream that keeps writing
# successfully can run for hours because every send completes quickly
# while the client reads. What it terminates is a handler stuck in ONE
# client-socket operation - an idle pre-request read, a half-open client
# that vanished without RST, a send to a client that stopped reading.
# A bounded socket operation ensures disconnected clients cannot leave
# handler threads holding admission reservations indefinitely.
MESH_HTTP_SOCKET_TIMEOUT_S = float(
    os.environ.get("VERATHOS_MESH_HTTP_SOCKET_TIMEOUT_S", "120") or 120.0
)
# Applied when a client sends no completion bound of its own: llama-server
# would otherwise decode without limit (see backend_openai_request).
MESH_DEFAULT_MAX_TOKENS = int(
    os.environ.get("VERATHOS_MESH_DEFAULT_MAX_TOKENS", "4096") or 4096
)
# Blob-route build-on-miss state: one dequant at a time (concurrent misses for
# 70B-class tensors would otherwise stack multi-second builds). The immutable
# manifest already loaded by each worker-server closure is used directly;
# caching one manifest globally would mix models when tests or an embedding
# process hosts more than one server.
_blob_build_lock = threading.Lock()

# Coarse hot-path timing, enabled with VERATHOS_MESH_TIMING_LOG=1. One line
# per timed section on stderr; grep VMESH_TIMING and aggregate offline. Used
# to attribute serve overhead that external profilers cannot reach (ptrace
# is blocked in most container runtimes).
_TIMING_LOG_ENABLED = os.environ.get("VERATHOS_MESH_TIMING_LOG", "") == "1"


@contextmanager
def _timed_section(label: str):
    if not _TIMING_LOG_ENABLED:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        print(
            f"VMESH_TIMING {label} {time.perf_counter() - started:.4f}",
            file=sys.stderr,
            flush=True,
        )

VERIFIED_GGUF_SAMPLER_MODE = (
    "deterministic_no_penalty_top_k_1_seed_0_prompt_cache_v2"
)
# LIGHT-tier stochastic profile: sampler params are BOUND, not rejected.
# The light relation verifies the realized token against the captured
# top-k of the raw logits, so any sampler whose support stays inside that
# top-k and whose filters never REORDER raw logits (top_k/top_p/min_p/
# temperature; penalties stay neutral) is verifiable as-is. The applied
# controls are committed in the receipt (verified_sampler_controls +
# their hash), mirroring the vLLM route's sampler_config_hash
# substitution guard. Hard tier stays greedy: the exact argmax binding
# has no sampled counterpart until canonical seeded replay lands.
VERIFIED_GGUF_SAMPLED_LIGHT_MODE = (
    "committed_stochastic_light_top_k_bounded_seeded_v1"
)
VERIFIED_GGUF_SAMPLER_CONTROLS: dict[str, Any] = {
    "temperature": 0,
    "samplers": ["top_k"],
    "top_k": 1,
    "top_p": 1,
    "min_p": 0,
    "repeat_last_n": 0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "mirostat": 0,
    "ignore_eos": False,
    "seed": 0,
    # The prompt cache stays ON for every serve, canary and organic alike.
    # A multi-turn conversation re-sends its whole history each message;
    # without prefix reuse the backend re-prefills all of it, which grows
    # quadratically over a conversation. Witness capture stays alive because
    # the patched llama-server always re-evaluates a minimum eager prompt
    # tail (VERATHOS_MIN_EAGER_PROMPT_TAIL, default 32), so even a FULL
    # cache hit on an exactly repeated prompt produces witnesses; the
    # zero-witness fallback re-serve in receipt_for remains as a safety
    # net, keyed only on the observable capture outcome, never on origin.
    "cache_prompt": True,
}


_PRIVATE_PROOF_ARTIFACT_KEYS = frozenset(
    {
        "endpoint",
        "endpoints",
        "file_path",
        "filesystem_path",
        "host",
        "hostname",
        "hotkey",
        "ip",
        "ip_address",
        "local_path",
        # NOT bare "path": Merkle openings carry their sibling lists under
        # exactly that key (pcs w-col manifest groups), and stripping it
        # silently broke EVERY payload whose challenge drew a w-col-rooted
        # tensor (glm MoE router tensors, observed as "pcs w-col
        # group membership proof failed ... path_len=0"). A filesystem
        # path stored under "path" is still removed by the VALUE checks
        # (_LOCAL_PATH_VALUE_RE); the explicit *_path key names and
        # suffixes below keep covering the metadata cases.
        "port",
        "proof_endpoint",
        "rpc_endpoint",
        "trace_path",
        "uid",
        "uri",
        "url",
        "worker_endpoint",
        "worker_hotkey",
        "worker_uid",
    }
)
_PRIVATE_PROOF_ARTIFACT_KEY_SUFFIXES = (
    "_endpoint",
    "_endpoints",
    "_file_path",
    "_filesystem_path",
    "_host",
    "_hostname",
    "_hotkey",
    "_ip",
    "_local_path",
    "_port",
    "_trace_path",
    "_uid",
    "_uri",
    "_url",
)
_NETWORK_LOCATION_VALUE_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|"
    r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+):\d{1,5}\b)",
    re.IGNORECASE,
)
_LOCAL_PATH_VALUE_RE = re.compile(
    r"(?:^|[\s=:'\"])(?:/home/|/Users/|/private/|/tmp/|/var/|"
    r"~[/\\]|[A-Za-z]:[/\\]|\\\\)",
)
_SS58_HOTKEY_VALUE_RE = re.compile(r"^5[1-9A-HJ-NP-Za-km-z]{47}$")


def _is_private_proof_artifact_key(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _PRIVATE_PROOF_ARTIFACT_KEYS or normalized.endswith(
        _PRIVATE_PROOF_ARTIFACT_KEY_SUFFIXES
    )


# Every alternative of the network regex requires ":" and every
# alternative of the path regex requires one of ":/\\~"; a string without
# any of these separators can only be private as an SS58 (fixed length 48)
# or a token substring. Proof payloads are DOMINATED by separator-free hex
# digests, so gating the regexes on these characters is what keeps the
# sanitizer linear (observed: a glm LM-head hard payload ground the
# unguarded walk for 20+ minutes while holding the GIL, wedging the proof
# port's accept queue).
_PRIVATE_VALUE_HINT_CHARS = (":", "/", "\\", "~")


def _looks_like_private_proof_artifact_value(
    value: str,
    *,
    private_tokens: Iterable[str] = (),
) -> bool:
    candidate = str(value)
    if any(ch in candidate for ch in _PRIVATE_VALUE_HINT_CHARS) and (
        _NETWORK_LOCATION_VALUE_RE.search(candidate)
        or _LOCAL_PATH_VALUE_RE.search(candidate)
    ):
        return True
    stripped = candidate.strip()
    if (
        len(stripped) == 48
        and stripped.startswith("5")
        and _SS58_HOTKEY_VALUE_RE.fullmatch(stripped)
    ):
        return True
    return any(token and token in candidate for token in private_tokens)


def _proof_artifact_scalar_list_kind(value: Any) -> str:
    """"numeric" / "string" for homogeneous scalar lists, "" otherwise.

    The walkers use this to process huge proof arrays (logit rows, opened
    vectors, merkle digest paths) in one C-speed pass instead of a
    recursive frame plus a fresh path f-string per element.
    """

    numeric = True
    strings = True
    for child in value:
        if child is None or type(child) in (int, float, bool):
            strings = False
            if not numeric:
                return ""
            continue
        if type(child) is str:
            numeric = False
            if not strings:
                return ""
            continue
        return ""
    if numeric:
        return "numeric"
    return "string" if strings else ""


def _sanitize_public_proof_artifact_value(
    value: Any,
    *,
    private_tokens: tuple[str, ...],
    path: str,
) -> Any:
    """Return an endpoint/identity-free proof value.

    Private metadata stored under mapping keys is safe to remove because it is
    not verifier input.  A private value embedded in a list is rejected instead
    of changing list positions that may be cryptographically meaningful.
    """

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if _is_private_proof_artifact_key(key):
                continue
            if isinstance(child, str) and _looks_like_private_proof_artifact_value(
                child,
                private_tokens=private_tokens,
            ):
                continue
            sanitized[key] = _sanitize_public_proof_artifact_value(
                child,
                private_tokens=private_tokens,
                path=f"{path}.{key}",
            )
        return sanitized
    if isinstance(value, (list, tuple)):
        kind = _proof_artifact_scalar_list_kind(value)
        if kind == "numeric":
            return list(value)
        if kind == "string":
            for index, child in enumerate(value):
                if _looks_like_private_proof_artifact_value(
                    child,
                    private_tokens=private_tokens,
                ):
                    raise RuntimeError(
                        f"{path}[{index}] contains private routing or "
                        "filesystem data"
                    )
            return list(value)
        sanitized_list: list[Any] = []
        for index, child in enumerate(value):
            if isinstance(child, str) and _looks_like_private_proof_artifact_value(
                child,
                private_tokens=private_tokens,
            ):
                raise RuntimeError(
                    f"{path}[{index}] contains private routing or filesystem data"
                )
            sanitized_list.append(
                _sanitize_public_proof_artifact_value(
                    child,
                    private_tokens=private_tokens,
                    path=f"{path}[{index}]",
                )
            )
        return sanitized_list
    return value


def _assert_public_proof_artifact_value(
    value: Any,
    *,
    private_tokens: tuple[str, ...] = (),
    path: str = "proof_artifact",
) -> None:
    """Fail closed if a validator proof artifact still contains private data."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _is_private_proof_artifact_key(key):
                raise RuntimeError(f"{path}.{key} is a private proof-artifact field")
            _assert_public_proof_artifact_value(
                child,
                private_tokens=private_tokens,
                path=f"{path}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        kind = _proof_artifact_scalar_list_kind(value)
        if kind == "numeric":
            return
        if kind == "string":
            for index, child in enumerate(value):
                if _looks_like_private_proof_artifact_value(
                    child,
                    private_tokens=private_tokens,
                ):
                    raise RuntimeError(
                        f"{path}[{index}] contains private routing or "
                        "filesystem data"
                    )
            return
        for index, child in enumerate(value):
            _assert_public_proof_artifact_value(
                child,
                private_tokens=private_tokens,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, str) and _looks_like_private_proof_artifact_value(
        value,
        private_tokens=private_tokens,
    ):
        raise RuntimeError(f"{path} contains private routing or filesystem data")


def _sanitize_validator_proof_artifacts(
    proof_receipts: list[dict[str, Any]],
    proof_payloads: list[dict[str, Any]],
    *,
    private_tokens: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Strip non-proof private metadata and rebind payload commitments.

    The GGML payload commitment covers the complete serialized payload, so
    removing local trace paths must also update the paired receipt commitment.
    The cryptographic proof itself and all verifier inputs remain unchanged.
    """

    from verallm.mesh.ggml_proof import ggml_proof_payload_commitment_hash

    tokens = tuple(
        sorted(
            {str(item) for item in private_tokens if str(item)},
            key=len,
            reverse=True,
        )
    )
    receipts = [dict(item) for item in proof_receipts]
    payloads: list[dict[str, Any]] = []
    receipt_indexes: dict[str, list[int]] = {}
    for index, receipt in enumerate(receipts):
        commitment = str(receipt.get("proof_commitment_hash", ""))
        receipt_indexes.setdefault(commitment, []).append(index)
    used_receipts: set[int] = set()

    for payload_index, raw_payload in enumerate(proof_payloads):
        if not isinstance(raw_payload, Mapping):
            raise RuntimeError("proof payload must be an object")
        original = dict(raw_payload)
        original_commitment = str(original.get("proof_commitment_hash", ""))
        matches = [
            index
            for index in receipt_indexes.get(original_commitment, [])
            if index not in used_receipts
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "proof payload does not have one unique paired proof receipt"
            )
        sanitized = _sanitize_public_proof_artifact_value(
            original,
            private_tokens=tokens,
            path=f"proof_payloads[{payload_index}]",
        )
        if not isinstance(sanitized, dict):
            raise RuntimeError("sanitized proof payload must be an object")
        sanitized["proof_commitment_hash"] = ggml_proof_payload_commitment_hash(
            sanitized
        )
        receipt_index = matches[0]
        used_receipts.add(receipt_index)
        receipts[receipt_index]["proof_commitment_hash"] = sanitized[
            "proof_commitment_hash"
        ]
        # A private worker signature over the pre-sanitized receipt is neither
        # public nor valid after rebinding. Public stage attestation is handled
        # separately by the endpoint-free receipt protocol.
        receipts[receipt_index]["signature"] = ""
        payloads.append(sanitized)

    _assert_public_proof_artifact_value(
        {"proof_payloads": payloads},
        private_tokens=tokens,
    )
    return receipts, payloads

# The current GGML decode proof verifies the deterministic argmax path.  Keep
# the public request semantics honest: neutral/omitted controls are accepted,
# but a caller asking for stochastic sampling or a logits transform must not be
# silently served a different sampler profile.  Canonical stochastic replay can
# replace this guard once it is implemented for the GGUF backend.
_VERIFIED_GGUF_NEUTRAL_NUMERIC_CONTROLS: dict[str, float] = {
    "temperature": 0.0,
    "top_k": 1.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_last_n": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "mirostat": 0.0,
    "seed": 0.0,
    "n": 1.0,
    "best_of": 1.0,
    "typical_p": 1.0,
    "tfs_z": 1.0,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "xtc_probability": 0.0,
}


def validate_verified_gguf_sampler_request(request: Mapping[str, Any]) -> None:
    """Reject request controls incompatible with the proved GGUF sampler.

    Missing controls use the mesh profile defaults.  Explicit controls must be
    neutral and deterministic so the signed/request-hashed semantics match the
    sampler actually executed by llama.cpp.
    """

    if not isinstance(request, Mapping):
        raise ValueError("OpenAI request must be a mapping")
    for name, expected in _VERIFIED_GGUF_NEUTRAL_NUMERIC_CONTROLS.items():
        if name not in request or request[name] is None:
            continue
        value = request[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"verified GGUF sampler requires {name}={expected:g}"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != expected:
            raise ValueError(
                f"verified GGUF sampler requires {name}={expected:g}"
            )
    if "samplers" in request and request["samplers"] is not None:
        if request["samplers"] != ["top_k"]:
            raise ValueError(
                "verified GGUF sampler requires samplers=['top_k']"
            )
    for name in ("ignore_eos", "do_sample"):
        if name in request and request[name] not in (None, False):
            raise ValueError(f"verified GGUF sampler requires {name}=false")
    # cache_prompt does not change the sampled tokens and the worker
    # overrides it with the canonical controls anyway; only require a
    # boolean so the canonical profile itself round-trips validation.
    if "cache_prompt" in request and not isinstance(
        request["cache_prompt"], (bool, type(None))
    ):
        raise ValueError("verified GGUF sampler requires boolean cache_prompt")
    for name in ("logit_bias", "grammar", "json_schema"):
        value = request.get(name)
        if value not in (None, "", {}):
            raise ValueError(
                f"verified GGUF sampler does not support non-empty {name}"
            )
    response_format = request.get("response_format")
    if response_format not in (None, {}, {"type": "text"}):
        raise ValueError(
            "verified GGUF sampler supports only text response_format"
        )


# Sampler controls a LIGHT-tier verified serve may carry non-neutrally.
# Everything else in _VERIFIED_GGUF_NEUTRAL_NUMERIC_CONTROLS stays pinned:
# penalties/dry/mirostat/xtc REORDER or transform the raw logits, which
# would let a sampled token legitimately fall outside the captured top-k
# the light relation verifies against.
_SAMPLED_LIGHT_FREE_CONTROLS = frozenset(
    {"temperature", "top_k", "top_p", "min_p", "seed"}
)


def sampled_light_controls_from_request(
    request: Mapping[str, Any], *, seed: int
) -> dict[str, Any] | None:
    """The committed stochastic profile for a light-tier verified serve.

    Returns None when the request is greedy/neutral (the canonical greedy
    profile applies). Otherwise validates that only order-preserving
    sampler controls are non-neutral, clamps the support to the decode
    audit width (the light opening only holds the captured
    top-``VERATHOS_GGUF_DECODE_AUDIT_TOP_K`` of each audited logits row,
    so a wider support could sample a token its own opening cannot
    prove), and returns the EXACT controls dict the backend runs with,
    which the receipt commits via verified_gguf_sampler_controls_hash.
    ``seed`` pins the draw (client seed or the derived replay seed) so
    audit replays re-execute the identical profile.
    """

    if not isinstance(request, Mapping):
        raise ValueError("OpenAI request must be a mapping")
    sampled = False
    for name in _SAMPLED_LIGHT_FREE_CONTROLS:
        value = request.get(name)
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(f"sampler control {name} must be numeric")
        if float(value) != _VERIFIED_GGUF_NEUTRAL_NUMERIC_CONTROLS.get(
            name, 0.0
        ):
            sampled = True
    if request.get("do_sample") is True:
        sampled = True
    if not sampled:
        return None
    # Everything OUTSIDE the free set must still pass the strict gate;
    # validate a copy with the free controls stripped so only the pinned
    # controls (penalties, transforms, n, response_format, ...) are
    # checked.
    pinned_view = {
        key: value
        for key, value in request.items()
        if key not in _SAMPLED_LIGHT_FREE_CONTROLS
        and key not in ("do_sample", "samplers")
    }
    validate_verified_gguf_sampler_request(pinned_view)

    def _pos_float(name: str, default: float, lo: float, hi: float) -> float:
        value = request.get(name)
        if value is None or isinstance(value, bool):
            return default
        numeric = float(value)
        if not math.isfinite(numeric) or not (lo <= numeric <= hi):
            raise ValueError(
                f"sampler control {name} must be within [{lo:g}, {hi:g}]"
            )
        return numeric

    temperature = _pos_float("temperature", 1.0, 0.0, 5.0)
    top_p = _pos_float("top_p", 1.0, 0.05, 1.0)
    min_p = _pos_float("min_p", 0.0, 0.0, 0.5)
    raw_top_k = request.get("top_k")
    if raw_top_k is None or isinstance(raw_top_k, bool):
        top_k = int(VERATHOS_GGUF_DECODE_AUDIT_TOP_K)
    else:
        top_k = int(raw_top_k)
        if top_k <= 0:
            top_k = int(VERATHOS_GGUF_DECODE_AUDIT_TOP_K)
    # Clamp, do not reject: the committed profile is what actually ran,
    # so an over-wide request degrades gracefully to the provable width.
    top_k = min(top_k, int(VERATHOS_GGUF_DECODE_AUDIT_TOP_K))
    if temperature == 0.0:
        # Explicit seed/top_k with temperature 0 is still greedy.
        return None
    return {
        "temperature": temperature,
        # Order matters and is committed: support restriction first
        # (top_k within the audited width, then nested top_p/min_p which
        # only narrow it), temperature scaling last, then the seeded
        # draw. No penalties, no reordering.
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_last_n": 0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "mirostat": 0,
        "ignore_eos": False,
        "seed": int(seed) & 0x7FFFFFFF,
        "cache_prompt": True,
    }


def sampled_controls_from_context(
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The committed sampled profile bound by a policy/receipt, if any."""

    if str(context.get("verified_sampler_mode", "") or "") != (
        VERIFIED_GGUF_SAMPLED_LIGHT_MODE
    ):
        return None
    controls = context.get("verified_sampler_controls")
    if not isinstance(controls, Mapping):
        raise RuntimeError(
            "sampled receipt is missing its committed sampler controls"
        )
    return dict(controls)


def finalize_verified_sampler_policy(
    policy: dict[str, Any],
    openai_request: Mapping[str, Any],
    request_id: str,
) -> dict[str, Any] | None:
    """Pick and commit the verified sampler mode for one serve.

    Greedy stays the default. A request carrying stochastic sampler
    controls gets the committed LIGHT profile instead of a rejection,
    with two hard exclusions that keep the hard tier sound: an explicit
    hard-tier demand, and any lane whose committed hard-audit rate could
    draw a hard postcommit audit (its exact-argmax relation has no
    sampled counterpart). The chosen mode + controls ride the policy into
    the receipt, so audits and replays re-apply the identical profile.
    """

    if not policy.get("verified_sampler_required"):
        return None
    seed_value = openai_request.get("seed")
    if (
        isinstance(seed_value, int)
        and not isinstance(seed_value, bool)
        and seed_value > 0
    ):
        seed = int(seed_value)
    else:
        seed = derive_mesh_replay_seed(request_id)
    controls = sampled_light_controls_from_request(openai_request, seed=seed)
    if controls is None:
        policy["verified_sampler_mode"] = VERIFIED_GGUF_SAMPLER_MODE
        return None
    verathos = openai_request.get("verathos")
    proof_tier = (
        str(verathos.get("proof_tier", "") or "")
        if isinstance(verathos, Mapping)
        else ""
    )
    if proof_tier == "hard":
        raise ValueError(
            "hard-tier requests require the deterministic sampler profile "
            "(temperature 0); sampled profiles verify on the light tier"
        )
    if int(policy.get("proof_postcommit_hard_bps", 0) or 0) != 0:
        raise ValueError(
            "sampled profiles are only served on lanes whose hard-audit "
            "rate is zero; this lane can draw hard postcommit audits"
        )
    policy["verified_sampler_mode"] = VERIFIED_GGUF_SAMPLED_LIGHT_MODE
    policy["verified_sampler_controls"] = controls
    return controls


@dataclass
class WorkerProbe:
    """Result of probing a mesh worker endpoint."""

    endpoint: str
    status: str
    service: str
    version: int
    capability: CapabilityAd
    mesh_spec_hash: str = ""
    stage_assignment_hash: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "status": self.status,
            "service": self.service,
            "version": self.version,
            "capability_hash": self.capability.ad_hash_hex(),
            "capability": self.capability.to_dict(),
            "mesh_spec_hash": self.mesh_spec_hash,
            "stage_assignment_hash": self.stage_assignment_hash,
            "latency_ms": round(self.latency_ms, 3),
        }


def normalize_endpoint(endpoint: str) -> str:
    """Normalize worker endpoint and require an explicit HTTP(S) scheme."""

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an http:// or https:// URL")
    return endpoint.rstrip("/") + "/"


def proof_payload_stage_index(proof_payload: dict[str, Any]) -> int:
    """Return the stage index carried by a GGML proof payload."""

    for key in ("stage_index",):
        if key in proof_payload:
            return int(proof_payload[key])
    for key in ("op_manifest_membership", "trace_membership", "trace"):
        section = proof_payload.get(key, {})
        if isinstance(section, dict) and "stage_index" in section:
            return int(section["stage_index"])
    return 0


def backend_openai_request(
    openai_request: dict[str, Any],
    *,
    proof_capture_required: bool = False,
    verified_sampler_required: bool | None = None,
    proof_metadata_required: bool | None = None,
    replay_seed: int | None = None,
    prompt_cache_disabled: bool = False,
    sampled_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip Verathos-only controls and add backend-only proof controls."""

    request = dict(openai_request)
    request.pop("verathos", None)
    request.pop("verathos_proof_seed", None)
    request.pop("validator_nonce", None)
    # llama-server treats a missing max_tokens as UNLIMITED (n_predict -1).
    # A greedy reasoning model can then loop for tens of thousands of tokens
    # (observed on glm-5.2: 13.5k-token runaways), and a non-streaming
    # client that times out cannot cancel the slot, so the zombie generation
    # keeps stealing decode throughput from every later request. Cap
    # explicitly whenever the client sent no bound of its own.
    raw_max = request.get("max_tokens", request.get("max_completion_tokens"))
    if not isinstance(raw_max, int) or raw_max <= 0:
        request["max_tokens"] = MESH_DEFAULT_MAX_TOKENS
    sampler_required = (
        bool(proof_capture_required)
        if verified_sampler_required is None
        else bool(verified_sampler_required)
    )
    metadata_required = (
        bool(proof_capture_required)
        if proof_metadata_required is None
        else bool(proof_metadata_required)
    )
    if sampler_required:
        if sampled_profile is not None:
            # Committed light-tier stochastic profile: apply EXACTLY what
            # the receipt binds (incl. the pinned seed), for the serve and
            # for every audit replay alike.
            request.update(dict(sampled_profile))
        else:
            validate_verified_gguf_sampler_request(request)
            request.update(VERIFIED_GGUF_SAMPLER_CONTROLS)
    elif replay_seed is not None and request.get("seed") is None:
        # Pin the sampler seed so a deferred-audit replay can reproduce the
        # committed completion even for non-greedy organic traffic. The seed
        # is derived from the committed request id (see derive_mesh_replay_seed).
        request["seed"] = int(replay_seed)
    if prompt_cache_disabled:
        # Witness-regenerating replays re-serve a prompt the backend just
        # cached in full; a FULL llama-server cache hit skips the eager
        # prefill and CUDA-graph decode replay runs no per-op hooks, so with
        # the cache on the replay would capture nothing. Applied after the
        # sampler controls so it overrides their cache_prompt=True. Live
        # serves never set this directly — the zero-witness fallback in
        # receipt_for does, keyed only on the observable capture outcome.
        request["cache_prompt"] = False
    if metadata_required:
        # Streaming needs the token ids too.  The decode audit compares the
        # model's own sampled ids against the proof-domain logits, and the
        # re-tokenization fallback is deliberately not accepted for that, so
        # without these a streamed request cannot be decode-audited at all.

        # The patched llama-server honors return_tokens on the OpenAI
        # streaming routes and emits each chunk's sampled ids as a top-level
        # "tokens" extension field (same non-standard family as "timings"),
        # which the stream accumulator already reads.  Do NOT fall back to
        # requesting logprobs for the ids: llama-server sorts the full vocab
        # per generated token for logprobs, which measured as the entire
        # -30% aggregate throughput gap at 8-way concurrency.  An unpatched
        # llama-server never reaches this path (proof capture requires the
        # patched build), and a stale build surfaces as receipts failing
        # closed on missing completion token ids.
        request["return_tokens"] = True
        if not request.get("stream"):
            request["verbose"] = True
    return request


def replay_seed_for_receipt_context(receipt_context: Mapping[str, Any]) -> int | None:
    """Return the deterministic replay seed bound by a receipt, if any."""

    mode = str(receipt_context.get("proof_replay_seed_mode", "") or "")
    if mode != REPLAY_SEED_MODE_DERIVED:
        return None
    request_id = str(receipt_context.get("request_id", "") or "")
    if not request_id:
        return None
    derived = derive_mesh_replay_seed(request_id)
    committed = int(receipt_context.get("proof_replay_seed", 0) or 0)
    if committed and committed != derived:
        raise RuntimeError("worker proof_replay_seed mismatch")
    return derived


def validate_mesh_stage_context_commitments(
    receipt_context: Mapping[str, Any],
    spec: MeshSpec,
) -> None:
    """Recompute coordinator-supplied common proof bindings from local state.

    A stage signs public proof receipts, so it must not merely echo routing,
    assignment, or model commitments supplied by the coordinator.  These
    values are intentionally excluded from the Fiat--Shamir sampling domain,
    but remain authenticated receipt bindings and must match the exact private
    mesh specification installed on the producing stage.
    """

    if not isinstance(receipt_context, Mapping):
        raise RuntimeError("proof context must be a mapping")
    if not isinstance(spec, MeshSpec):
        raise RuntimeError("local mesh spec is unavailable")
    expected = {
        "mesh_id": spec.mesh_id,
        "mesh_spec_hash": spec.spec_hash_hex(),
        "stage_assignment_hash": spec.stage_assignment_hash_hex(),
        "rpc_plan_hash": rpc_plan_from_mesh(spec).plan_hash_hex(),
        "model_package_hash": spec.model_package_hash,
        "model_tensor_manifest_root": spec.model_tensor_manifest_root,
        "model_total_layers": int(spec.total_layers),
    }
    for field, value in expected.items():
        if receipt_context.get(field) != value:
            raise RuntimeError(f"proof context {field} mismatch")


def verified_gguf_sampler_controls_hash(
    controls: Mapping[str, Any] | None = None,
) -> str:
    """Hash of the sampler controls a verified serve actually applied.

    Default (None) is the canonical greedy profile; a sampled light serve
    passes its committed per-request controls. Receipts bind this hash so
    the executed sampler profile cannot be silently substituted (the mesh
    analogue of the vLLM route's sampler_config_hash)."""

    return hashlib.sha256(
        b"VERATHOS_MESH_VERIFIED_GGUF_SAMPLER_CONTROLS_V1"
        + json.dumps(
            dict(controls) if controls is not None
            else VERIFIED_GGUF_SAMPLER_CONTROLS,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _normalize_token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    token_ids: list[int] = []
    for item in value:
        token_id = None
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            token_id = item
        elif isinstance(item, dict):
            raw_id = item.get("id")
            if isinstance(raw_id, bool):
                return None
            if isinstance(raw_id, int):
                token_id = raw_id
        if token_id is None:
            return None
        token_ids.append(int(token_id))
    return token_ids


def completion_token_ids_from_response(response: dict[str, Any]) -> tuple[list[int], str]:
    """Extract generated token ids from llama.cpp response carriers."""

    carriers: list[tuple[str, Any]] = [
        ("tokens", response.get("tokens")),
    ]
    verbose = response.get("__verbose")
    if isinstance(verbose, dict):
        carriers.append(("__verbose.tokens", verbose.get("tokens")))
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            carriers.append(("choices[0].tokens", first_choice.get("tokens")))
            carriers.append(
                (
                    "choices[0].verathos_generated_tokens",
                    first_choice.get("verathos_generated_tokens"),
                )
            )
            logprobs = first_choice.get("logprobs")
            if isinstance(logprobs, dict):
                carriers.append(
                    (
                        "choices[0].logprobs.content",
                        logprobs.get("content"),
                    )
                )
    for source, candidate in carriers:
        token_ids = _normalize_token_ids(candidate)
        if token_ids is not None:
            return token_ids, source
    return [], ""


def completion_text_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message.get("content", ""))
    if isinstance(first.get("text"), str):
        return str(first.get("text", ""))
    return ""


def fetch_completion_token_ids_from_backend(
    backend_base_url: str,
    response: dict[str, Any],
) -> tuple[list[int], str]:
    """Tokenize visible assistant text when llama.cpp does not return token ids."""

    content = completion_text_from_response(response)
    if content == "":
        return [], ""
    token_payload = post_json(
        urljoin(backend_base_url.rstrip("/") + "/", "tokenize"),
        {
            "content": content,
            "add_special": False,
            "parse_special": False,
        },
        timeout=30.0,
    )
    token_ids = _normalize_token_ids(token_payload.get("tokens"))
    if token_ids is None:
        raise RuntimeError("llama.cpp /tokenize response missing completion token ids")
    return token_ids, "llama_cpp_completion_tokenize_v1"


def _token_ids_hash(tag: bytes, token_ids: list[int]) -> str:
    return hashlib.sha256(
        tag
        + json.dumps(token_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def completion_token_ids_hash(token_ids: list[int]) -> str:
    return _token_ids_hash(b"VERATHOS_MESH_COMPLETION_TOKEN_IDS_V1", token_ids)


def prompt_token_ids_hash(token_ids: list[int]) -> str:
    return _token_ids_hash(b"VERATHOS_MESH_PROMPT_TOKEN_IDS_V1", token_ids)


def prompt_template_hash(prompt: Any) -> str:
    return hashlib.sha256(
        b"VERATHOS_MESH_PROMPT_TEMPLATE_V1"
        + json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def strip_backend_verification_fields(response: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(response)
    cleaned.pop("__verbose", None)
    choices = cleaned.get("choices")
    if isinstance(choices, list):
        cleaned_choices = []
        for choice in choices:
            if isinstance(choice, dict):
                item = dict(choice)
                item.pop("logprobs", None)
                item.pop("verathos_generated_tokens", None)
                item.pop("verathos_slot_id", None)
                cleaned_choices.append(item)
            else:
                cleaned_choices.append(choice)
        cleaned["choices"] = cleaned_choices
    return cleaned


def semantic_openai_response_hash(response: dict[str, Any]) -> str:
    """Hash generated output while excluding coordinator-chosen metadata.

    The same commitment is used by deterministic replay checks and the proof
    sampling gate.  IDs, timestamps, usage counters, finish reasons, choice
    indexes, tool-call IDs, and arbitrary extension fields are deliberately
    excluded because changing them does not require another inference.
    """

    def stable_message(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        stable: dict[str, Any] = {}
        for field in ("content", "refusal"):
            if field in value:
                stable[field] = value[field]
        function_call = value.get("function_call")
        if isinstance(function_call, dict):
            stable["function_call"] = {
                field: function_call[field]
                for field in ("name", "arguments")
                if field in function_call
            }
        tool_calls = value.get("tool_calls")
        if isinstance(tool_calls, list):
            stable_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                item: dict[str, Any] = {}
                if "type" in tool_call:
                    item["type"] = tool_call["type"]
                function = tool_call.get("function")
                if isinstance(function, dict):
                    item["function"] = {
                        field: function[field]
                        for field in ("name", "arguments")
                        if field in function
                    }
                stable_tool_calls.append(item)
            stable["tool_calls"] = stable_tool_calls
        return stable

    choices = response.get("choices", [])
    stable_choices = []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            item: dict[str, Any] = {}
            for field in ("message", "delta"):
                if field in choice:
                    item[field] = stable_message(choice[field])
            if "text" in choice:
                item["text"] = choice["text"]
            stable_choices.append(item)
    body = {"choices": stable_choices}
    return hashlib.sha256(
        b"VERATHOS_MESH_SEMANTIC_OPENAI_RESPONSE_V1"
        + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare_backend_response_for_receipt(
    response: dict[str, Any],
    *,
    proof_capture_required: bool,
    stream: bool,
) -> tuple[dict[str, Any], list[int], str]:
    token_ids, token_source = completion_token_ids_from_response(response)
    if proof_capture_required and not stream:
        response = strip_backend_verification_fields(response)
    return response, token_ids, token_source


def slot_id_from_backend_response(response: dict[str, Any]) -> int:
    """Extract the llama-server slot id that served a completion, or -1."""

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and first.get("verathos_slot_id") is not None:
            try:
                return int(first["verathos_slot_id"])
            except (TypeError, ValueError):
                return -1
    if response.get("id_slot") is not None:
        try:
            return int(response["id_slot"])
        except (TypeError, ValueError):
            return -1
    verbose = response.get("__verbose")
    if isinstance(verbose, dict) and verbose.get("id_slot") is not None:
        try:
            return int(verbose["id_slot"])
        except (TypeError, ValueError):
            return -1
    return -1


def wait_for_backend_model_loaded(
    backend_url: str,
    *,
    deadline_s: float = 3600.0,
    poll_s: float = 10.0,
    opener: Callable[..., Any] = urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until the llama backend reports the model loaded, or time out.

    llama.cpp's ``/health`` answers 503 while weights are still loading and
    200 only once the model is resident, so a 200 is the load-complete
    signal. Returns True on 200, False when ``backend_url`` is empty or the
    deadline passes; callers treat False as "proceed anyway" because every
    caller is idempotent background work, never a correctness gate.
    """

    if not backend_url:
        return False
    deadline = clock() + max(0.0, float(deadline_s))
    while True:
        try:
            with opener(
                backend_url.rstrip("/") + "/health", timeout=5.0
            ) as response:
                if int(getattr(response, "status", 0) or 0) == 200:
                    return True
        except Exception:
            pass
        if clock() >= deadline:
            return False
        sleeper(max(0.1, float(poll_s)))


def fetch_prompt_token_ids_from_backend(
    backend_base_url: str,
    openai_request: dict[str, Any],
) -> tuple[list[int], str, str]:
    """Use llama.cpp chat-template and tokenizer endpoints for prompt binding."""

    template_request = backend_openai_request(openai_request, proof_capture_required=False)
    template_request.pop("stream", None)
    template_payload = post_json(
        urljoin(backend_base_url.rstrip("/") + "/", "apply-template"),
        template_request,
        timeout=30.0,
    )
    if "prompt" not in template_payload:
        raise RuntimeError("llama.cpp /apply-template response missing prompt")
    prompt = template_payload["prompt"]
    token_payload = post_json(
        urljoin(backend_base_url.rstrip("/") + "/", "tokenize"),
        {
            "content": prompt,
            "add_special": True,
            "parse_special": True,
        },
        timeout=30.0,
    )
    token_ids = _normalize_token_ids(token_payload.get("tokens"))
    if token_ids is None:
        raise RuntimeError("llama.cpp /tokenize response missing token id list")
    return token_ids, "llama_cpp_apply_template_tokenize_v1", prompt_template_hash(prompt)


def validator_nonce_from_request(openai_request: dict[str, Any]) -> str:
    """Extract a Verathos validator nonce from an OpenAI-compatible request."""

    verathos = openai_request.get("verathos", {})
    if isinstance(verathos, dict):
        nonce = verathos.get("validator_nonce") or verathos.get("nonce")
        if nonce:
            return str(nonce)
    nonce = openai_request.get("validator_nonce")
    return str(nonce or "")


def validator_challenge_commitment_from_request(
    openai_request: dict[str, Any],
) -> str:
    """Extract the hidden postcommit challenge commitment from a request."""

    verathos = openai_request.get("verathos", {})
    if not isinstance(verathos, dict):
        return ""
    return str(verathos.get("challenge_nonce_commitment") or "")


def validator_request_id_from_request(openai_request: dict[str, Any]) -> str:
    """Extract the validator-chosen request id from a signed mesh request."""

    verathos = openai_request.get("verathos", {})
    if not isinstance(verathos, dict):
        return ""
    return str(verathos.get("validator_request_id") or "")


def normalize_validator_request_id(value: str) -> str:
    """Require the canonical 32-byte hex request id generated by a validator."""

    raw = str(value or "").strip()
    if len(raw) != 64:
        raise ValueError(
            "validator_request_id must be 32 bytes encoded as lowercase hex"
        )
    try:
        decoded = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError("validator_request_id must be lowercase hex") from exc
    if len(decoded) != 32 or decoded.hex() != raw:
        raise ValueError(
            "validator_request_id must be 32 bytes encoded as lowercase hex"
        )
    return raw


def mesh_request_id_from_request(
    openai_request: dict[str, Any],
    *,
    require_validator_request_id: bool = False,
    fallback: str = "",
) -> str:
    """Choose a request id without letting a coordinator grind signed traffic."""

    validator_request_id = validator_request_id_from_request(openai_request)
    if validator_request_id:
        return normalize_validator_request_id(validator_request_id)
    if require_validator_request_id:
        raise ValueError("validator_request_id is required")
    return str(fallback or uuid.uuid4())


def decode_audit_stage_index_for_spec(spec: MeshSpec) -> int:
    """Return the unique compute stage that owns the model's final layer.

    The stage index is committed before Fiat--Shamir sampling so a coordinator
    cannot move the expensive final-logit opening to a more convenient worker
    after seeing the challenge.
    """

    owners = {
        int(member.stage_index)
        for member in spec.members
        if int(member.layers.end) > int(member.layers.start)
        and int(member.layers.end) == int(spec.total_layers)
    }
    if len(owners) != 1:
        raise RuntimeError("mesh must have exactly one final compute stage")
    return next(iter(owners))


def verify_mesh_proof_sampling_fields(
    receipt: dict[str, Any],
    openai_request: dict[str, Any],
) -> None:
    """Recompute and verify the mesh proof sampling gate in a receipt."""

    if not receipt.get("proof_capture_required"):
        if receipt.get("proof_sampled") not in (False, None):
            raise RuntimeError("worker proof_sampled mismatch")
        if receipt.get("proof_required") not in (False, None):
            raise RuntimeError("worker proof_required mismatch")
        if receipt.get("decode_audit_required") not in (False, None):
            raise RuntimeError("worker decode_audit_required mismatch")
        return

    sample_bps = normalize_proof_sample_bps(
        int(receipt.get("proof_sample_bps", PROOF_SAMPLE_BPS_DENOMINATOR))
    )
    decode_bps = normalize_proof_sample_bps(int(receipt.get("decode_audit_bps", 0)))
    effective_sample_bps = max(sample_bps, decode_bps)
    if int(receipt.get("proof_sample_denominator", PROOF_SAMPLE_BPS_DENOMINATOR)) != (
        PROOF_SAMPLE_BPS_DENOMINATOR
    ):
        raise RuntimeError("worker proof_sample_denominator mismatch")

    gate_hash = mesh_proof_gate_hash(receipt)
    if receipt.get("proof_gate_hash") != gate_hash:
        raise RuntimeError("worker proof_gate_hash mismatch")

    if receipt.get("proof_postcommit"):
        if receipt.get("proof_challenge_kind") != (
            VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
        ):
            raise RuntimeError("worker proof_challenge_kind mismatch")
        if validator_nonce_from_request(openai_request):
            raise RuntimeError(
                "postcommit phase-one request exposed the validator nonce"
            )
        request_id = normalize_validator_request_id(
            validator_request_id_from_request(openai_request)
        )
        snapshot_hash = _require_sha256_text(
            "verification_snapshot_hash",
            receipt.get("verification_snapshot_hash", ""),
        )
        request_commitment = _require_sha256_text(
            "challenge_nonce_commitment",
            validator_challenge_commitment_from_request(openai_request),
        )
        if receipt.get("proof_challenge_nonce_commitment") != (
            request_commitment
        ):
            raise RuntimeError(
                "worker proof_challenge_nonce_commitment mismatch"
            )
        if str(receipt.get("request_id", "")) != request_id:
            raise RuntimeError("worker validator_request_id mismatch")
        if not bool(receipt.get("proof_postcommit_finalized", False)):
            if receipt.get("proof_postcommit_origin_receipt_hash"):
                raise RuntimeError(
                    "worker proof_postcommit_origin_receipt_hash mismatch"
                )
            if receipt.get("proof_postcommit_challenge_nonce"):
                raise RuntimeError(
                    "worker proof_postcommit_challenge_nonce mismatch"
                )
            if receipt.get("proof_beacon", ""):
                raise RuntimeError("worker proof_beacon mismatch")
            if int(receipt.get("proof_sample_value", -1)) != -1:
                raise RuntimeError("worker proof_sample_value mismatch")
            if bool(receipt.get("proof_sampled", False)):
                raise RuntimeError("worker proof_sampled mismatch")
            if bool(receipt.get("proof_required", False)):
                raise RuntimeError("worker proof_required mismatch")
            if bool(receipt.get("decode_audit_required", False)):
                raise RuntimeError("worker decode_audit_required mismatch")
            if decode_bps > 0:
                commitment = mesh_decode_audit_commitment_hash(receipt)
                if receipt.get("decode_audit_commitment_hash") != commitment:
                    raise RuntimeError(
                        "worker decode_audit_commitment_hash mismatch"
                    )
            return

        challenge_nonce = normalize_validator_challenge_nonce(
            str(receipt.get("proof_postcommit_challenge_nonce", ""))
        )
        expected_commitment = mesh_validator_challenge_nonce_commitment(
            challenge_nonce,
            validator_request_id=request_id,
            verification_snapshot_hash=snapshot_hash,
        )
        if request_commitment != expected_commitment:
            raise RuntimeError("worker postcommit challenge reveal mismatch")
        origin_hash = _require_sha256_text(
            "proof_postcommit_origin_receipt_hash",
            receipt.get("proof_postcommit_origin_receipt_hash", ""),
        )
        beacon = derive_mesh_postcommit_proof_beacon(
            origin_receipt_hash=origin_hash,
            proof_gate_hash=gate_hash,
            challenge_nonce=challenge_nonce,
        )
        # The tier draw runs against the receipt-bound hard-audit rate (a
        # pre-tiering receipt has none and keeps the base rate = full hard).
        hard_bps = normalize_proof_sample_bps(
            int(receipt.get("proof_postcommit_hard_bps", sample_bps))
        )
        expected_sample_value = (
            0
            if hard_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
            else mesh_proof_sample_value(beacon)
            if hard_bps > 0
            else -1
        )
        expected_sampled = (
            hard_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
            or (
                hard_bps > 0
                and should_sample_mesh_proof(
                    beacon=beacon,
                    sample_bps=hard_bps,
                )
            )
        )
        # Stricter-only upgrade: a validator hard demand (or a miner choosing
        # to hard-prove anyway) makes proof_sampled True on a light draw.
        # The reverse, claiming light on a hard draw, stays a mismatch.
        if (
            not expected_sampled
            and str(receipt.get("proof_audit_tier", "")) == "hard"
            and bool(receipt.get("proof_sampled", False))
        ):
            expected_sampled = True
        expected_challenge_kind = VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
    if (
        receipt.get("proof_deferred")
        and receipt.get("proof_challenge_kind") == "deferred_future_randomness_v1"
    ):
        if receipt.get("proof_beacon", ""):
            raise RuntimeError("worker proof_beacon mismatch")
        if int(receipt.get("proof_sample_value", -1)) != -1:
            raise RuntimeError("worker proof_sample_value mismatch")
        if bool(receipt.get("proof_deferred_obligation", False)) is not True:
            raise RuntimeError("worker proof_deferred_obligation mismatch")
        expected_required = (
            normalize_proof_sample_bps(
                int(receipt.get("proof_deferred_audit_bps", 0))
            )
            >= PROOF_SAMPLE_BPS_DENOMINATOR
        )
        if bool(receipt.get("proof_deferred_required", False)) != expected_required:
            raise RuntimeError("worker proof_deferred_required mismatch")
        if bool(receipt.get("proof_sampled", False)):
            raise RuntimeError("worker proof_sampled mismatch")
        if bool(receipt.get("proof_required", False)):
            raise RuntimeError("worker proof_required mismatch")
        if bool(receipt.get("decode_audit_required", False)):
            raise RuntimeError("worker decode_audit_required mismatch")
        deferred_audit_decision(receipt)
        return

    if receipt.get("proof_postcommit"):
        pass
    elif effective_sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR:
        from verallm.mesh.ggml_proof import (
            derive_every_request_trace_beacon,
            derive_every_request_trace_beacon_v2,
        )

        nonce_raw = validator_nonce_from_request(openai_request)
        if nonce_raw:
            # V2 mixes the validator nonce so which-op selection is not
            # derived purely from miner-controlled commitment input.
            beacon = derive_every_request_trace_beacon_v2(
                gate_hash,
                normalize_validator_nonce(nonce_raw),
            )
        else:
            beacon = derive_every_request_trace_beacon(gate_hash)
        expected_sample_value = 0 if sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR else (
            mesh_proof_sample_value(beacon) if sample_bps > 0 else -1
        )
        expected_sampled = sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR or (
            sample_bps > 0
            and should_sample_mesh_proof(beacon=beacon, sample_bps=sample_bps)
        )
        expected_challenge_kind = "inline_every_request_v1"
    elif effective_sample_bps > 0:
        nonce_raw = validator_nonce_from_request(openai_request)
        if nonce_raw:
            normalize_validator_nonce(nonce_raw)
            beacon = derive_mesh_proof_beacon(gate_hash, nonce_raw)
            expected_sample_value = (
                mesh_proof_sample_value(beacon) if sample_bps > 0 else -1
            )
            expected_sampled = (
                should_sample_mesh_proof(beacon=beacon, sample_bps=sample_bps)
                if sample_bps > 0
                else False
            )
            expected_challenge_kind = "fiat_shamir_inline_v1"
        else:
            beacon = b""
            expected_sample_value = -1
            expected_sampled = False
            expected_challenge_kind = "deferred_future_randomness_v1"
    else:
        beacon = b""
        expected_sample_value = -1
        expected_sampled = False
        expected_challenge_kind = "disabled"

    if receipt.get("proof_challenge_kind") != expected_challenge_kind:
        raise RuntimeError("worker proof_challenge_kind mismatch")
    if receipt.get("proof_beacon", "") != beacon.hex():
        raise RuntimeError("worker proof_beacon mismatch")
    if int(receipt.get("proof_sample_value", -1)) != expected_sample_value:
        raise RuntimeError("worker proof_sample_value mismatch")
    if bool(receipt.get("proof_sampled", False)) != expected_sampled:
        raise RuntimeError("worker proof_sampled mismatch")
    if decode_bps > 0:
        if int(receipt.get("decode_audit_stage_index", -1)) < 0:
            raise RuntimeError("worker decode_audit_stage_index missing")
        if receipt.get("decode_audit_mode") != VERATHOS_GGUF_DECODE_AUDIT_MODE:
            raise RuntimeError("worker decode_audit_mode mismatch")
        commitment = mesh_decode_audit_commitment_hash(receipt)
        if receipt.get("decode_audit_commitment_hash") != commitment:
            raise RuntimeError("worker decode_audit_commitment_hash mismatch")
        expected_decode_sample_value = (
            0
            if decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
            else mesh_decode_audit_sample_value(beacon)
            if beacon
            else -1
        )
        if decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR:
            expected_decode_sampled = True
        elif beacon:
            expected_decode_sampled = should_sample_mesh_decode_audit(
                beacon=beacon,
                sample_bps=decode_bps,
            )
        else:
            expected_decode_sampled = False
        if receipt.get("proof_postcommit") and not expected_sampled:
            # Postcommit lane: decode openings ride the hard tier only -
            # a light draw owes nothing at reveal time, so the finalized
            # receipt must not claim one.
            expected_decode_sampled = False
        if int(receipt.get("decode_audit_sample_value", -1)) != expected_decode_sample_value:
            raise RuntimeError("worker decode_audit_sample_value mismatch")
        if bool(receipt.get("decode_audit_sampled", False)) != expected_decode_sampled:
            raise RuntimeError("worker decode_audit_sampled mismatch")
        if bool(receipt.get("decode_audit_required", False)) != expected_decode_sampled:
            raise RuntimeError("worker decode_audit_required mismatch")
        expected_positions = (
            derive_mesh_decode_audit_positions(
                beacon=beacon,
                decode_commitment_hash=commitment,
                completion_token_count=int(receipt.get("completion_token_count", 0)),
            )
            if expected_decode_sampled
            else []
        )
        actual_positions = [int(item) for item in receipt.get("decode_audit_positions", [])]
        if actual_positions != expected_positions:
            raise RuntimeError("worker decode_audit_positions mismatch")
    else:
        expected_decode_sampled = False
        if receipt.get("decode_audit_required") not in (False, None):
            raise RuntimeError("worker decode_audit_required mismatch")
        if receipt.get("decode_audit_sampled") not in (False, None):
            raise RuntimeError("worker decode_audit_sampled mismatch")
    if bool(receipt.get("proof_required", False)) != (
        expected_sampled or expected_decode_sampled
    ):
        raise RuntimeError("worker proof_required mismatch")


def deferred_audit_decision(
    receipt: dict[str, Any],
    *,
    randomness: str | bytes = "",
) -> dict[str, Any]:
    """Verify and optionally evaluate a deferred future-randomness audit gate."""

    if not receipt.get("proof_deferred"):
        return {
            "configured": False,
            "sampled": False,
            "sample_value": -1,
            "beacon": "",
        }
    if receipt.get("proof_deferred_mode") != "future_randomness_v1":
        raise RuntimeError("worker proof_deferred_mode mismatch")
    bps = normalize_proof_sample_bps(int(receipt.get("proof_deferred_audit_bps", 0)))
    if bool(receipt.get("proof_deferred_obligation", False)) is not True:
        raise RuntimeError("worker proof_deferred_obligation mismatch")
    if bool(receipt.get("proof_deferred_required", False)) != (
        bps >= PROOF_SAMPLE_BPS_DENOMINATOR
    ):
        raise RuntimeError("worker proof_deferred_required mismatch")
    denominator = int(
        receipt.get("proof_sample_denominator", PROOF_SAMPLE_BPS_DENOMINATOR)
    )
    if denominator != PROOF_SAMPLE_BPS_DENOMINATOR:
        raise RuntimeError("worker proof_sample_denominator mismatch")
    sample_commitment = mesh_deferred_audit_sample_commitment_hash(receipt)
    if receipt.get("proof_deferred_sample_commitment_hash") != sample_commitment:
        raise RuntimeError(
            "worker proof_deferred_sample_commitment_hash mismatch"
        )
    commitment = mesh_deferred_audit_commitment_hash(receipt)
    if receipt.get("proof_deferred_commitment_hash") != commitment:
        raise RuntimeError("worker proof_deferred_commitment_hash mismatch")
    if bps <= 0:
        raise RuntimeError("worker proof_deferred_audit_bps mismatch")
    if not randomness:
        return {
            "configured": True,
            "sampled": False,
            "sample_value": -1,
            "beacon": "",
            "commitment_hash": commitment,
            "sample_commitment_hash": sample_commitment,
            "proof_sampled": False,
            "proof_sample_value": -1,
            "decode_audit_sampled": False,
            "decode_audit_sample_value": -1,
        }
    randomness_bytes = normalize_deferred_randomness(randomness)
    beacon = derive_mesh_deferred_audit_beacon(
        sample_commitment,
        randomness_bytes,
        randomness_round=str(receipt.get("proof_deferred_randomness_round", "")),
    )
    deferred_sample_value = mesh_deferred_audit_sample_value(
        beacon,
        denominator=denominator,
    )
    deferred_sampled = should_sample_mesh_deferred_audit(
        beacon=beacon,
        sample_bps=bps,
        denominator=denominator,
    )
    proof_bps = normalize_proof_sample_bps(int(receipt.get("proof_sample_bps", 0)))
    decode_bps = normalize_proof_sample_bps(int(receipt.get("decode_audit_bps", 0)))
    proof_sample_value = (
        mesh_proof_sample_value(beacon)
        if proof_bps > 0
        else -1
    )
    proof_sampled = (
        should_sample_mesh_proof(
            beacon=beacon,
            sample_bps=proof_bps,
        )
        if proof_bps > 0
        else False
    )
    decode_sample_value = (
        mesh_decode_audit_sample_value(beacon)
        if decode_bps > 0
        else -1
    )
    decode_sampled = (
        should_sample_mesh_decode_audit(
            beacon=beacon,
            sample_bps=decode_bps,
        )
        if decode_bps > 0
        else False
    )
    sampled = bool(proof_sampled or decode_sampled)
    sample_candidates = [
        value for value in (proof_sample_value, decode_sample_value) if value >= 0
    ]
    sample_value = min(sample_candidates) if sample_candidates else deferred_sample_value
    if receipt.get("proof_deferred_randomness"):
        if receipt.get("proof_deferred_randomness") != randomness_bytes.hex():
            raise RuntimeError("worker proof_deferred_randomness mismatch")
    if receipt.get("proof_deferred_beacon"):
        if receipt.get("proof_deferred_beacon") != beacon.hex():
            raise RuntimeError("worker proof_deferred_beacon mismatch")
    if int(receipt.get("proof_deferred_sample_value", -1)) not in (-1, sample_value):
        raise RuntimeError("worker proof_deferred_sample_value mismatch")
    if bool(receipt.get("proof_deferred_sampled", False)) not in (False, sampled):
        raise RuntimeError("worker proof_deferred_sampled mismatch")
    return {
        "configured": True,
        "sampled": bool(sampled),
        "sample_value": int(sample_value),
        "beacon": beacon.hex(),
        "commitment_hash": commitment,
        "sample_commitment_hash": sample_commitment,
        "randomness": randomness_bytes.hex(),
        "proof_sampled": bool(proof_sampled),
        "proof_sample_value": int(proof_sample_value),
        "decode_audit_sampled": bool(decode_sampled),
        "decode_audit_sample_value": int(decode_sample_value),
        "deferred_sampled": bool(deferred_sampled),
        "deferred_sample_value": int(deferred_sample_value),
    }


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _bounded_canonical_json_bytes(
    value: Any,
    *,
    max_bytes: int,
    limit_error: str,
) -> bytes:
    """Serialize deterministic compact JSON without an unbounded full clone."""

    if max_bytes <= 0:
        raise RuntimeError(limit_error)
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload = bytearray()
    total_bytes = 0
    for chunk in encoder.iterencode(value):
        encoded = chunk.encode("utf-8")
        total_bytes += len(encoded)
        if total_bytes > max_bytes:
            raise RuntimeError(limit_error)
        payload.extend(encoded)
    return bytes(payload)


def _artifact_response(
    artifact: dict[str, Any],
    openai_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = openai_response if openai_response is not None else artifact.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("mesh artifact missing response")
    return response


def _artifact_completion_token_binding(
    artifact: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[list[int], str]:
    completion_ids = _normalize_token_ids(artifact.get("completion_token_ids"))
    completion_source = str(receipt.get("completion_token_source", ""))
    if completion_ids is not None:
        return completion_ids, completion_source
    decode_ids = _normalize_token_ids(receipt.get("decode_audit_completion_token_ids"))
    decode_source = str(receipt.get("decode_audit_completion_token_source", ""))
    if decode_ids is not None:
        return decode_ids, decode_source
    return [], ""


def deferred_audit_context_from_receipt(
    receipt: dict[str, Any],
    *,
    randomness: str | bytes,
    completion_token_ids: list[int] | None = None,
    completion_token_source: str = "",
) -> dict[str, Any]:
    """Build the deterministic proof context for a future-randomness audit."""

    if receipt.get("receipt_hash") and receipt.get("receipt_hash") != mesh_receipt_hash(receipt):
        raise RuntimeError("deferred audit origin receipt_hash mismatch")
    if receipt.get("mesh_response_commitment_hash") and receipt.get(
        "mesh_response_commitment_hash"
    ) != mesh_response_commitment_hash(receipt):
        raise RuntimeError("deferred audit origin response commitment mismatch")

    decision = deferred_audit_decision(receipt, randomness=randomness)
    if not decision.get("configured"):
        raise RuntimeError("deferred audit is not configured")

    ctx = _json_clone(receipt)
    ctx["proof_challenge_kind"] = "deferred_future_randomness_v1"
    ctx["proof_deferred_randomness"] = str(decision.get("randomness", ""))
    ctx["proof_deferred_beacon"] = str(decision.get("beacon", ""))
    ctx["proof_deferred_sample_value"] = int(decision.get("sample_value", -1))
    ctx["proof_deferred_sampled"] = bool(decision.get("sampled", False))
    ctx["proof_beacon"] = str(decision.get("beacon", "")) if decision.get("sampled") else ""
    ctx["proof_sampled"] = bool(decision.get("proof_sampled", False))
    ctx["proof_sample_value"] = int(decision.get("proof_sample_value", -1))
    ctx["decode_audit_sampled"] = bool(decision.get("decode_audit_sampled", False))
    ctx["decode_audit_required"] = bool(decision.get("decode_audit_sampled", False))
    ctx["decode_audit_sample_value"] = int(decision.get("decode_audit_sample_value", -1))
    # Respect the scope the serve committed: keep the no-replay candidate-set
    # scope when the receipt was built that way (single-node capture); only
    # force the manifest-challenge (replay) scope otherwise.
    if (
        ctx.get("proof_op_manifest_root")
        and str(ctx.get("proof_trace_scope", "")) != "trace_candidate_set_v1"
    ):
        ctx["proof_trace_scope"] = "op_manifest_challenge_v1"
    if ctx["decode_audit_required"]:
        if not ctx.get("decode_audit_commitment_hash"):
            ctx["decode_audit_commitment_hash"] = mesh_decode_audit_commitment_hash(ctx)
        ctx["decode_audit_positions"] = derive_mesh_decode_audit_positions(
            beacon=bytes.fromhex(str(decision["beacon"])),
            decode_commitment_hash=str(ctx.get("decode_audit_commitment_hash", "")),
            completion_token_count=int(ctx.get("completion_token_count", 0)),
        )
        ids = [int(item) for item in (completion_token_ids or [])]
        if ids and completion_token_source:
            if len(ids) != int(ctx.get("completion_token_count", 0)):
                raise RuntimeError("deferred audit completion token count mismatch")
            ctx["decode_audit_completion_token_ids"] = ids
            ctx["decode_audit_completion_token_ids_hash"] = completion_token_ids_hash(ids)
            ctx["decode_audit_completion_token_count"] = len(ids)
            ctx["decode_audit_completion_token_source"] = str(completion_token_source)
    else:
        ctx["decode_audit_positions"] = []
        ctx.pop("decode_audit_completion_token_ids", None)
        ctx.pop("decode_audit_completion_token_ids_hash", None)
        ctx.pop("decode_audit_completion_token_count", None)
        ctx.pop("decode_audit_completion_token_source", None)

    ctx["proof_required"] = bool(ctx["proof_sampled"] or ctx["decode_audit_required"])
    ctx["proof_receipt_root"] = ""
    ctx["proof_receipt_count"] = 0
    ctx["proof_receipt_verified"] = False
    ctx["proof_verifier_ms"] = 0.0
    ctx["proof_replay_started_unix_ns"] = 0
    ctx["proof_replay_ended_unix_ns"] = 0
    ctx["decode_audit_receipt_root"] = ""
    ctx["decode_audit_verified"] = False
    ctx["decode_audit_verifier_ms"] = 0.0
    ctx["verified"] = False
    ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
    ctx["receipt_hash"] = mesh_receipt_hash(ctx)
    return ctx


def _expected_deferred_proof_stage_indexes(
    receipt: dict[str, Any],
    *,
    spec: MeshSpec | None,
    required_proof_stage_indexes: Iterable[int] | None,
    require_spec_proof_stage_coverage: bool,
) -> set[int]:
    if required_proof_stage_indexes is not None:
        return {int(item) for item in required_proof_stage_indexes}
    if (
        spec is not None
        and require_spec_proof_stage_coverage
        and bool(receipt.get("proof_required", False))
    ):
        receipt_stage = int(receipt.get("stage_index", -1))
        receipt_member = next(
            (
                member
                for member in spec.members
                if int(member.stage_index) == receipt_stage
                and str(member.endpoint) == str(receipt.get("endpoint", ""))
            ),
            None,
        )
        if receipt_member is not None and receipt_member.role != "coordinator":
            return (
                {receipt_stage}
                if receipt_member.layers.end > receipt_member.layers.start
                else set()
            )
        return {
            int(member.stage_index)
            for member in spec.members
            if int(member.layers.end) > int(member.layers.start)
        }
    return set()


def finalize_deferred_audit_context(
    audit_context: dict[str, Any],
    *,
    proof_receipts: list[dict[str, Any]],
    proof_payloads: list[dict[str, Any]],
    spec: MeshSpec | None = None,
    required_proof_stage_indexes: Iterable[int] | None = None,
    require_spec_proof_stage_coverage: bool = True,
) -> dict[str, Any]:
    """Attach and verify post-hoc proof roots to a derived audit context."""

    ctx = _json_clone(audit_context)
    if not ctx.get("proof_required"):
        if proof_receipts or proof_payloads:
            raise RuntimeError("deferred audit proof supplied for unsampled receipt")
        ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
        ctx["receipt_hash"] = mesh_receipt_hash(ctx)
        return ctx

    if not isinstance(proof_receipts, list) or not proof_receipts:
        raise RuntimeError("deferred sampled proof receipts are required")
    if not isinstance(proof_payloads, list) or not proof_payloads:
        raise RuntimeError("deferred sampled proof payloads are required")
    parsed = [LlamaGraphOpReceipt.from_dict(item) for item in proof_receipts]
    ctx["proof_mode"] = VERATHOS_GGML_GEMM_PROOF_MODE
    ctx["proof_receipt_root"] = llama_graph_receipt_root_hex(parsed)
    ctx["proof_receipt_count"] = len(parsed)
    expected_stages = _expected_deferred_proof_stage_indexes(
        ctx,
        spec=spec,
        required_proof_stage_indexes=required_proof_stage_indexes,
        require_spec_proof_stage_coverage=require_spec_proof_stage_coverage,
    )
    if spec is not None:
        verify_llama_graph_proof_receipts_for_mesh(
            ctx,
            parsed,
            spec,
            required_stage_indexes=expected_stages,
        )
    else:
        verify_llama_graph_proof_receipts(ctx, parsed)

    from verallm.mesh.ggml_proof import (
        ggml_decode_audit_receipt_root,
        verify_ggml_decode_audit_payloads,
        verify_ggml_gemm_proof_payloads,
    )

    proof_result = verify_ggml_gemm_proof_payloads(
        proof_payloads,
        [item.to_dict() for item in parsed],
        mesh_receipt=ctx,
    )
    if not proof_result.verified:
        raise RuntimeError("deferred audit proof verification failed: " + proof_result.message)
    ctx["proof_receipt_verified"] = True
    ctx["verified"] = True
    ctx["proof_verifier_ms"] = 0.0

    if ctx.get("decode_audit_required"):
        completion_ids = _normalize_token_ids(ctx.get("decode_audit_completion_token_ids"))
        if completion_ids is None:
            raise RuntimeError("deferred decode audit completion tokens are required")
        ctx["decode_audit_receipt_root"] = ggml_decode_audit_receipt_root(proof_payloads)
        decode_result = verify_ggml_decode_audit_payloads(
            ctx,
            proof_payloads,
            completion_token_ids=completion_ids,
        )
        if not decode_result.verified:
            raise RuntimeError(
                "deferred decode audit verification failed: " + decode_result.message
            )
        ctx["decode_audit_verified"] = True
        ctx["decode_audit_verifier_ms"] = 0.0

    ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
    ctx["receipt_hash"] = mesh_receipt_hash(ctx)
    return ctx


def mesh_deferred_audit_bundle_hash(bundle: dict[str, Any]) -> str:
    body = dict(bundle)
    body.pop("bundle_hash", None)
    return hashlib.sha256(
        b"VERATHOS_MESH_DEFERRED_AUDIT_BUNDLE_V1"
        + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_deferred_audit_bundle(
    *,
    origin_receipt: dict[str, Any],
    audit_receipt: dict[str, Any],
    randomness: str | bytes,
    proof_receipts: list[dict[str, Any]],
    proof_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = deferred_audit_decision(origin_receipt, randomness=randomness)
    bundle = {
        "version": 1,
        "kind": DEFERRED_AUDIT_BUNDLE_KIND,
        "origin_receipt_hash": str(origin_receipt.get("receipt_hash", "")),
        "deferred_commitment_hash": str(
            origin_receipt.get("proof_deferred_commitment_hash", "")
        ),
        "deferred_sample_commitment_hash": str(
            origin_receipt.get("proof_deferred_sample_commitment_hash", "")
        ),
        "randomness": str(decision.get("randomness", "")),
        "randomness_round": str(origin_receipt.get("proof_deferred_randomness_round", "")),
        "decision": decision,
        "audit_receipt": audit_receipt,
        "proof_receipts": proof_receipts,
        "proof_payloads": proof_payloads,
    }
    bundle["bundle_hash"] = mesh_deferred_audit_bundle_hash(bundle)
    return bundle


def verify_deferred_mesh_audit_bundle(
    bundle: dict[str, Any],
    artifact: dict[str, Any],
    openai_request: dict[str, Any],
    *,
    openai_response: dict[str, Any] | None = None,
    spec: MeshSpec | None = None,
    member_index: int | None = None,
    required_proof_stage_indexes: Iterable[int] | None = None,
    require_spec_proof_stage_coverage: bool = True,
) -> bool:
    """Verify a post-hoc future-randomness audit bundle externally."""

    if int(bundle.get("version", 0)) != 1:
        raise RuntimeError("deferred audit bundle version mismatch")
    if bundle.get("kind") != DEFERRED_AUDIT_BUNDLE_KIND:
        raise RuntimeError("deferred audit bundle kind mismatch")
    if bundle.get("bundle_hash") != mesh_deferred_audit_bundle_hash(bundle):
        raise RuntimeError("deferred audit bundle_hash mismatch")
    if not isinstance(artifact.get("receipt"), dict):
        raise RuntimeError("deferred audit origin artifact missing receipt")
    receipt = artifact["receipt"]
    randomness = str(bundle.get("randomness", ""))
    verify_mesh_inference_artifact(
        artifact,
        openai_request,
        openai_response=openai_response,
        spec=spec,
        member_index=member_index,
        deferred_randomness=randomness,
        require_deferred_proof_if_sampled=False,
    )
    if bundle.get("origin_receipt_hash") != receipt.get("receipt_hash"):
        raise RuntimeError("deferred audit origin_receipt_hash mismatch")
    if bundle.get("deferred_commitment_hash") != receipt.get(
        "proof_deferred_commitment_hash"
    ):
        raise RuntimeError("deferred audit commitment hash mismatch")
    if bundle.get("deferred_sample_commitment_hash") != receipt.get(
        "proof_deferred_sample_commitment_hash"
    ):
        raise RuntimeError("deferred audit sample commitment hash mismatch")

    completion_ids, completion_source = _artifact_completion_token_binding(artifact, receipt)
    expected_context = deferred_audit_context_from_receipt(
        receipt,
        randomness=randomness,
        completion_token_ids=completion_ids,
        completion_token_source=completion_source,
    )
    if not expected_context.get("proof_required"):
        raise RuntimeError("deferred audit bundle supplied for unsampled receipt")

    proof_receipts = bundle.get("proof_receipts", [])
    proof_payloads = bundle.get("proof_payloads", [])
    if not isinstance(proof_receipts, list):
        raise RuntimeError("deferred audit proof_receipts must be a list")
    if not isinstance(proof_payloads, list):
        raise RuntimeError("deferred audit proof_payloads must be a list")
    finalized = finalize_deferred_audit_context(
        expected_context,
        proof_receipts=proof_receipts,
        proof_payloads=proof_payloads,
        spec=spec,
        required_proof_stage_indexes=required_proof_stage_indexes,
        require_spec_proof_stage_coverage=require_spec_proof_stage_coverage,
    )
    claimed_receipt = bundle.get("audit_receipt")
    if not isinstance(claimed_receipt, dict):
        raise RuntimeError("deferred audit bundle missing audit_receipt")
    if claimed_receipt != finalized:
        raise RuntimeError("deferred audit audit_receipt mismatch")
    return True


_POSTCOMMIT_MUTABLE_RECEIPT_FIELDS = frozenset(
    {
        "receipt_hash",
        "signature",
        "mesh_response_commitment_hash",
        "proof_required",
        "proof_sampled",
        "proof_sample_value",
        "proof_beacon",
        "proof_mode",
        "proof_receipt_root",
        "proof_receipt_count",
        "proof_receipt_verified",
        "proof_verifier_ms",
        "proof_replay_started_unix_ns",
        "proof_replay_ended_unix_ns",
        "verified",
        "decode_audit_required",
        "decode_audit_sampled",
        "decode_audit_sample_value",
        "decode_audit_positions",
        "decode_audit_completion_token_ids",
        "decode_audit_completion_token_ids_hash",
        "decode_audit_completion_token_count",
        "decode_audit_completion_token_source",
        "decode_audit_receipt_root",
        "decode_audit_verified",
        "decode_audit_verifier_ms",
        "proof_postcommit_origin_receipt_hash",
        "proof_postcommit_challenge_nonce",
        "proof_postcommit_finalized",
        "proof_audit_tier",
    }
)


def _require_sha256_text(name: str, value: Any) -> str:
    raw = str(value or "")
    if (
        len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return raw


def postcommit_audit_decision(
    origin_receipt: dict[str, Any],
    openai_request: dict[str, Any],
    *,
    challenge_nonce: str | bytes,
    force_hard: bool = False,
) -> dict[str, Any]:
    """Verify one frozen origin and derive its one-time proof challenge.

    The audit tier (hard sumcheck vs light openings) is drawn from the
    nonce-derived beacon against the receipt's committed hard-audit rate, so
    it is fixed at reveal time and unpredictable before it, exactly v3's
    postcommit tier draw. ``force_hard`` is the validator's signed
    stricter-only override (canary hard slots); there is no light override,
    a demand can never fall below the drawn tier.
    """

    if origin_receipt.get("proof_postcommit") is not True:
        raise RuntimeError("mesh origin is not configured for postcommit proof")
    if origin_receipt.get("proof_challenge_kind") != (
        VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
    ):
        raise RuntimeError("mesh origin postcommit challenge kind mismatch")
    if origin_receipt.get("proof_postcommit_finalized") not in (False, None):
        raise RuntimeError("mesh origin is already postcommit-finalized")
    if origin_receipt.get("proof_beacon", ""):
        raise RuntimeError("mesh origin must not expose a proof beacon")
    if origin_receipt.get("proof_required") not in (False, None):
        raise RuntimeError("mesh origin must not claim a selected proof")
    if origin_receipt.get("proof_sampled") not in (False, None):
        raise RuntimeError("mesh origin must not claim a sampled base proof")
    if origin_receipt.get("decode_audit_required") not in (False, None):
        raise RuntimeError("mesh origin must not claim a decode proof")
    if origin_receipt.get("proof_receipt_root"):
        raise RuntimeError("mesh origin must not carry proof receipts")

    request_id = normalize_validator_request_id(
        validator_request_id_from_request(openai_request)
    )
    if str(origin_receipt.get("request_id", "")) != request_id:
        raise RuntimeError("mesh origin validator_request_id mismatch")
    snapshot_hash = _require_sha256_text(
        "verification_snapshot_hash",
        origin_receipt.get("verification_snapshot_hash", ""),
    )
    requested_snapshot_hash = str(
        (openai_request.get("verathos") or {}).get(
            "verification_snapshot_hash",
            "",
        )
    )
    if requested_snapshot_hash != snapshot_hash:
        raise RuntimeError("mesh origin verification snapshot mismatch")

    request_commitment = _require_sha256_text(
        "challenge_nonce_commitment",
        validator_challenge_commitment_from_request(openai_request),
    )
    receipt_commitment = _require_sha256_text(
        "proof_challenge_nonce_commitment",
        origin_receipt.get("proof_challenge_nonce_commitment", ""),
    )
    if receipt_commitment != request_commitment:
        raise RuntimeError("mesh origin challenge commitment mismatch")
    expected_commitment = mesh_validator_challenge_nonce_commitment(
        challenge_nonce,
        validator_request_id=request_id,
        verification_snapshot_hash=snapshot_hash,
    )
    if receipt_commitment != expected_commitment:
        raise RuntimeError("validator challenge nonce reveal mismatch")

    gate_hash = mesh_proof_gate_hash(origin_receipt)
    if str(origin_receipt.get("proof_gate_hash", "")) != gate_hash:
        raise RuntimeError("mesh origin proof_gate_hash mismatch")
    origin_hash = _require_sha256_text(
        "origin_receipt_hash",
        origin_receipt.get("receipt_hash", ""),
    )
    if origin_hash != mesh_receipt_hash(origin_receipt):
        raise RuntimeError("mesh origin receipt_hash mismatch")
    beacon = derive_mesh_postcommit_proof_beacon(
        origin_receipt_hash=origin_hash,
        proof_gate_hash=gate_hash,
        challenge_nonce=challenge_nonce,
    )

    proof_bps = normalize_proof_sample_bps(
        int(origin_receipt.get("proof_sample_bps", 0))
    )
    # The hard-audit rate governs the postcommit tier draw. Receipts from a
    # pre-tiering coordinator carry no rate, which means the old behavior:
    # every sampled postcommit audit is hard.
    hard_bps = normalize_proof_sample_bps(
        int(origin_receipt.get("proof_postcommit_hard_bps", proof_bps))
    )
    decode_bps = normalize_proof_sample_bps(
        int(origin_receipt.get("decode_audit_bps", 0))
    )
    proof_sample_value = (
        0
        if hard_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        else mesh_proof_sample_value(beacon)
        if hard_bps > 0
        else -1
    )
    proof_sampled = (
        hard_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        or (
            hard_bps > 0
            and should_sample_mesh_proof(
                beacon=beacon,
                sample_bps=hard_bps,
            )
        )
    )
    if force_hard:
        # The stricter-only override belongs to greedy canary slots. A
        # sampled light receipt has no exact-argmax relation to escalate
        # into; surfacing a policy error here beats a fake proof failure.
        if str(origin_receipt.get("verified_sampler_mode", "") or "") == (
            VERIFIED_GGUF_SAMPLED_LIGHT_MODE
        ):
            raise RuntimeError(
                "hard audit demand cannot apply to a sampled light receipt"
            )
        proof_sampled = True
    decode_sample_value = (
        0
        if decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        else mesh_decode_audit_sample_value(beacon)
        if decode_bps > 0
        else -1
    )
    # Decode-audit openings ride the HARD tier only: the protocol is
    # light or hard, nothing else, and the
    # hard relation already proves the decode. A light draw must carry
    # nothing that can fail on an honest serve - the light postcommit
    # obligation is the synthesized structural payload, which needs no
    # witness re-capture. (A light-draw decode lane false-failed an
    # honest hybrid-model serve: its serve-tail capture was lost to
    # batching and the replay probes reconstruct different recurrent
    # state, so the served token ranked ~293+ in replay logits.)
    decode_sampled = proof_sampled and (
        decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        or (
            decode_bps > 0
            and should_sample_mesh_decode_audit(
                beacon=beacon,
                sample_bps=decode_bps,
            )
        )
    )
    # v3 protocol shape : a light draw carries NO
    # postcommit obligation - the origin receipt with its inline light
    # proof already is the light tier. Only a hard draw (or a validator
    # hard demand) owes anything at reveal time, and the decode openings
    # ride inside it.
    audit_tier = "hard" if proof_sampled else "light"
    light_obligated = False
    return {
        "origin_receipt_hash": origin_hash,
        "challenge_nonce": normalize_validator_challenge_nonce(
            challenge_nonce
        ).hex(),
        "challenge_nonce_commitment": receipt_commitment,
        "proof_gate_hash": gate_hash,
        "beacon": beacon.hex(),
        "proof_sample_value": int(proof_sample_value),
        "proof_sampled": bool(proof_sampled),
        "decode_audit_sample_value": int(decode_sample_value),
        "decode_audit_sampled": bool(decode_sampled),
        "proof_required": bool(
            proof_sampled or decode_sampled or light_obligated
        ),
        "audit_tier": audit_tier,
    }


def postcommit_audit_context_from_receipt(
    origin_receipt: dict[str, Any],
    openai_request: dict[str, Any],
    *,
    challenge_nonce: str | bytes,
    completion_token_ids: list[int] | None = None,
    completion_token_source: str = "",
    force_hard: bool = False,
) -> dict[str, Any]:
    """Build the exact final proof context from a signed origin receipt."""

    if origin_receipt.get("mesh_response_commitment_hash") != (
        mesh_response_commitment_hash(origin_receipt)
    ):
        raise RuntimeError("mesh origin response commitment mismatch")
    decision = postcommit_audit_decision(
        origin_receipt,
        openai_request,
        challenge_nonce=challenge_nonce,
        force_hard=force_hard,
    )
    ctx = _json_clone(origin_receipt)
    ctx.pop("signature", None)
    ctx["proof_postcommit_origin_receipt_hash"] = decision[
        "origin_receipt_hash"
    ]
    ctx["proof_postcommit_challenge_nonce"] = decision["challenge_nonce"]
    ctx["proof_postcommit_finalized"] = True
    ctx["proof_beacon"] = decision["beacon"]
    ctx["proof_sample_value"] = int(decision["proof_sample_value"])
    ctx["proof_sampled"] = bool(decision["proof_sampled"])
    ctx["proof_audit_tier"] = str(decision["audit_tier"])
    ctx["decode_audit_sample_value"] = int(
        decision["decode_audit_sample_value"]
    )
    ctx["decode_audit_sampled"] = bool(decision["decode_audit_sampled"])
    ctx["decode_audit_required"] = bool(decision["decode_audit_sampled"])
    ctx["proof_required"] = bool(decision["proof_required"])

    if ctx["decode_audit_required"]:
        commitment = mesh_decode_audit_commitment_hash(ctx)
        if ctx.get("decode_audit_commitment_hash") != commitment:
            raise RuntimeError("mesh origin decode audit commitment mismatch")
        ctx["decode_audit_positions"] = derive_mesh_decode_audit_positions(
            beacon=bytes.fromhex(decision["beacon"]),
            decode_commitment_hash=commitment,
            completion_token_count=int(ctx.get("completion_token_count", 0)),
        )
        ids = [int(item) for item in (completion_token_ids or [])]
        if (
            not ids
            or not completion_token_source
            or len(ids) != int(ctx.get("completion_token_count", 0))
        ):
            raise RuntimeError(
                "postcommit decode audit completion token binding is missing"
            )
        ctx["decode_audit_completion_token_ids"] = ids
        ctx["decode_audit_completion_token_ids_hash"] = (
            completion_token_ids_hash(ids)
        )
        ctx["decode_audit_completion_token_count"] = len(ids)
        ctx["decode_audit_completion_token_source"] = str(
            completion_token_source
        )
    else:
        ctx["decode_audit_positions"] = []
        for field in (
            "decode_audit_completion_token_ids",
            "decode_audit_completion_token_ids_hash",
            "decode_audit_completion_token_count",
            "decode_audit_completion_token_source",
        ):
            ctx.pop(field, None)

    ctx["proof_receipt_root"] = ""
    ctx["proof_receipt_count"] = 0
    ctx["proof_receipt_verified"] = False
    ctx["proof_verifier_ms"] = 0.0
    ctx["proof_replay_started_unix_ns"] = 0
    ctx["proof_replay_ended_unix_ns"] = 0
    ctx["decode_audit_receipt_root"] = ""
    ctx["decode_audit_verified"] = False
    ctx["decode_audit_verifier_ms"] = 0.0
    ctx["verified"] = False
    ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
    ctx["receipt_hash"] = mesh_receipt_hash(ctx)
    return ctx


def finalize_postcommit_audit_context(
    audit_context: dict[str, Any],
    *,
    proof_receipts: list[dict[str, Any]],
    proof_payloads: list[dict[str, Any]],
    verification_snapshot: Any | None = None,
    spec: MeshSpec | None = None,
) -> dict[str, Any]:
    """Attach and independently verify proofs selected after origin commit."""

    ctx = _json_clone(audit_context)
    if not ctx.get("proof_required"):
        if proof_receipts or proof_payloads:
            raise RuntimeError("proofs supplied for an unsampled postcommit")
        ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
        ctx["receipt_hash"] = mesh_receipt_hash(ctx)
        return ctx
    if not isinstance(proof_receipts, list) or not proof_receipts:
        raise RuntimeError("postcommit proof receipts are required")
    if not isinstance(proof_payloads, list) or not proof_payloads:
        raise RuntimeError("postcommit proof payloads are required")

    opaque = str(ctx.get("proof_receipt_format", "")) == "opaque_stage_v2"
    if opaque:
        if verification_snapshot is None:
            raise RuntimeError(
                "postcommit opaque receipts require a verification snapshot"
            )
        parsed_stage = [
            MeshStageProofReceipt.from_dict(item) for item in proof_receipts
        ]
        ctx["proof_receipt_root"] = mesh_stage_proof_receipt_root_hex(
            parsed_stage
        )
        ctx["proof_receipt_count"] = len(parsed_stage)
        verify_mesh_stage_proof_receipts_for_snapshot(
            ctx,
            parsed_stage,
            verification_snapshot,
            require_complete_coverage=True,
        )
        normalized_receipts = [item.to_dict() for item in parsed_stage]
    else:
        parsed = [LlamaGraphOpReceipt.from_dict(item) for item in proof_receipts]
        ctx["proof_receipt_root"] = llama_graph_receipt_root_hex(parsed)
        ctx["proof_receipt_count"] = len(parsed)
        if spec is not None:
            verify_llama_graph_proof_receipts_for_mesh(ctx, parsed, spec)
        else:
            verify_llama_graph_proof_receipts(ctx, parsed)
        normalized_receipts = [item.to_dict() for item in parsed]

    from verallm.mesh.ggml_proof import (
        ggml_decode_audit_receipt_root,
        verify_ggml_decode_audit_payloads,
        verify_ggml_gemm_proof_payloads,
    )

    light_tier = str(ctx.get("proof_audit_tier", "")) == "light"
    if light_tier:
        from verallm.mesh.ggml_proof import verify_mesh_proof_payloads_any_tier
        from verallm.mesh.proof import VERATHOS_GGML_LIGHT_PROOF_MODE

        result = verify_mesh_proof_payloads_any_tier(
            proof_payloads,
            normalized_receipts,
            mesh_receipt=ctx,
            completion_token_ids=_normalize_token_ids(
                ctx.get("decode_audit_completion_token_ids")
            ),
            postcommit_light_ok=True,
        )
        if not result.verified:
            raise RuntimeError(
                "postcommit light proof verification failed: " + result.message
            )
        ctx["proof_mode"] = VERATHOS_GGML_LIGHT_PROOF_MODE
    else:
        result = verify_ggml_gemm_proof_payloads(
            proof_payloads,
            normalized_receipts,
            mesh_receipt=ctx,
        )
        if not result.verified:
            raise RuntimeError(
                "postcommit proof verification failed: " + result.message
            )
        ctx["proof_mode"] = VERATHOS_GGML_GEMM_PROOF_MODE
    ctx["proof_receipt_verified"] = True
    ctx["verified"] = True
    ctx["proof_verifier_ms"] = round(float(result.verifier_ms), 3)

    if ctx.get("decode_audit_required"):
        completion_ids = _normalize_token_ids(
            ctx.get("decode_audit_completion_token_ids")
        )
        if completion_ids is None:
            raise RuntimeError(
                "postcommit decode audit completion tokens are required"
            )
        ctx["decode_audit_receipt_root"] = ggml_decode_audit_receipt_root(
            proof_payloads
        )
        decode_result = verify_ggml_decode_audit_payloads(
            ctx,
            proof_payloads,
            completion_token_ids=completion_ids,
            receipts=normalized_receipts,
        )
        if not decode_result.verified:
            raise RuntimeError(
                "postcommit decode audit verification failed: "
                + decode_result.message
            )
        ctx["decode_audit_verified"] = True
        ctx["decode_audit_verifier_ms"] = round(
            float(decode_result.verifier_ms),
            3,
        )

    ctx["mesh_response_commitment_hash"] = mesh_response_commitment_hash(ctx)
    ctx["receipt_hash"] = mesh_receipt_hash(ctx)
    return ctx


def verify_mesh_postcommit_artifact(
    final_artifact: dict[str, Any],
    origin_artifact: dict[str, Any],
    openai_request: dict[str, Any],
    *,
    challenge_nonce: str | bytes,
    openai_response: dict[str, Any] | None = None,
    spec: MeshSpec | None = None,
    expected_coordinator_hotkey: str = "",
    expected_coordinator_uid: int | None = None,
    expected_validator_hotkey: str = "",
    require_coordinator_signature: bool = False,
    verification_snapshot: Any | None = None,
    verification_snapshot_now_unix: int | None = None,
    audit_tier: str = "",
) -> bool:
    """Verify origin commitment, reveal binding, final proof, and signatures.

    ``audit_tier`` is the tier the CALLER demanded in its signed postcommit
    request: "" for the default nonce-derived draw, "hard" for a forced hard
    audit (canary hard slots). It is verifier-supplied, never read from the
    artifact, and feeds the recomputed expected context so a miner cannot
    answer a hard demand with a light artifact.
    """

    if audit_tier not in ("", "hard"):
        raise RuntimeError("postcommit audit tier demand must be '' or 'hard'")

    response = (
        openai_response
        if openai_response is not None
        else origin_artifact.get("response")
    )
    if not isinstance(response, dict):
        raise RuntimeError("postcommit origin response is missing")
    verify_mesh_inference_artifact(
        origin_artifact,
        openai_request,
        openai_response=response,
        spec=spec,
        require_configured_proof=True,
        require_cryptographic_proof=False,
        expected_coordinator_hotkey=expected_coordinator_hotkey,
        expected_coordinator_uid=expected_coordinator_uid,
        expected_validator_hotkey=expected_validator_hotkey,
        require_coordinator_signature=require_coordinator_signature,
        verification_snapshot=verification_snapshot,
        verification_snapshot_now_unix=verification_snapshot_now_unix,
        require_validator_request_id=True,
    )
    origin_receipt = origin_artifact.get("receipt")
    final_receipt = final_artifact.get("receipt")
    if not isinstance(origin_receipt, dict) or not isinstance(final_receipt, dict):
        raise RuntimeError("postcommit artifact receipt is missing")
    if expected_validator_hotkey:
        for label, receipt in (
            ("origin", origin_receipt),
            ("final", final_receipt),
        ):
            if str(receipt.get("proof_validator_hotkey", "")) != str(
                expected_validator_hotkey
            ):
                raise RuntimeError(
                    f"postcommit {label} validator hotkey mismatch"
                )
    completion_ids, completion_source = _artifact_completion_token_binding(
        origin_artifact,
        origin_receipt,
    )
    expected = postcommit_audit_context_from_receipt(
        origin_receipt,
        openai_request,
        challenge_nonce=challenge_nonce,
        completion_token_ids=completion_ids,
        completion_token_source=completion_source,
        force_hard=audit_tier == "hard",
    )

    frozen_origin = {
        key: value
        for key, value in origin_receipt.items()
        if key not in _POSTCOMMIT_MUTABLE_RECEIPT_FIELDS
    }
    frozen_final = {
        key: value
        for key, value in final_receipt.items()
        if key not in _POSTCOMMIT_MUTABLE_RECEIPT_FIELDS
    }
    if frozen_final != frozen_origin:
        raise RuntimeError("postcommit final receipt changed frozen origin state")
    for field in (
        "proof_postcommit_origin_receipt_hash",
        "proof_postcommit_challenge_nonce",
        "proof_postcommit_finalized",
        "proof_beacon",
        "proof_sample_value",
        "proof_sampled",
        "proof_audit_tier",
        "decode_audit_sample_value",
        "decode_audit_sampled",
        "decode_audit_required",
        "decode_audit_positions",
        "proof_required",
    ):
        if final_receipt.get(field) != expected.get(field):
            raise RuntimeError(f"postcommit final receipt {field} mismatch")

    final_response = final_artifact.get("response")
    if final_response is not None and final_response != response:
        raise RuntimeError("postcommit final response changed the origin response")
    return bool(
        verify_mesh_inference_artifact(
            final_artifact,
            openai_request,
            openai_response=response,
            spec=spec,
            require_configured_proof=True,
            require_cryptographic_proof=True,
            expected_coordinator_hotkey=expected_coordinator_hotkey,
            expected_coordinator_uid=expected_coordinator_uid,
            expected_validator_hotkey=expected_validator_hotkey,
            require_coordinator_signature=require_coordinator_signature,
            verification_snapshot=verification_snapshot,
            verification_snapshot_now_unix=verification_snapshot_now_unix,
            require_validator_request_id=True,
        )
    )


def verify_mesh_inference_artifact(
    artifact: dict[str, Any],
    openai_request: dict[str, Any],
    *,
    openai_response: dict[str, Any] | None = None,
    spec: MeshSpec | None = None,
    member_index: int | None = None,
    require_configured_proof: bool = False,
    require_cryptographic_proof: bool = False,
    required_proof_stage_indexes: Iterable[int] | None = None,
    require_spec_proof_stage_coverage: bool = True,
    deferred_randomness: str | bytes = "",
    require_deferred_proof_if_sampled: bool = False,
    expected_coordinator_hotkey: str = "",
    expected_coordinator_uid: int | None = None,
    expected_validator_hotkey: str = "",
    require_coordinator_signature: bool = False,
    verification_snapshot: Any | None = None,
    verification_snapshot_now_unix: int | None = None,
    require_validator_request_id: bool = False,
) -> bool:
    """Verify a mesh inference artifact without trusting the coordinator.

    ``artifact`` may be either a ``/v1/mesh/inference`` response containing a
    top-level ``response`` field or a ``verathos_mesh`` metadata object from a
    chat completion response, in which case ``openai_response`` must be the
    response with that metadata removed.
    """

    receipt = artifact.get("receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("mesh artifact missing receipt")
    response = openai_response if openai_response is not None else artifact.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("mesh artifact missing response")

    validator_request_id = validator_request_id_from_request(openai_request)
    strict_validator_request_binding = bool(
        require_validator_request_id or validator_request_id
    )
    if validator_request_id:
        try:
            expected_request_id = normalize_validator_request_id(
                validator_request_id
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if str(receipt.get("request_id", "")) != expected_request_id:
            raise RuntimeError("mesh artifact validator_request_id mismatch")
    elif strict_validator_request_binding:
        raise RuntimeError("mesh request validator_request_id missing")

    request_hash = hashlib.sha256(
        json.dumps(openai_request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    response_hash = hashlib.sha256(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if receipt.get("request_hash") != request_hash:
        raise RuntimeError("mesh artifact request_hash mismatch")
    if receipt.get("response_hash") != response_hash:
        raise RuntimeError("mesh artifact response_hash mismatch")
    expected_semantic_response_hash = semantic_openai_response_hash(response)
    claimed_semantic_response_hash = str(
        receipt.get("semantic_response_hash", "")
    )
    if claimed_semantic_response_hash:
        if claimed_semantic_response_hash != expected_semantic_response_hash:
            raise RuntimeError("mesh artifact semantic_response_hash mismatch")
    elif strict_validator_request_binding:
        raise RuntimeError("mesh artifact semantic_response_hash missing")
    if receipt.get("receipt_hash") != mesh_receipt_hash(receipt):
        raise RuntimeError("mesh artifact receipt_hash mismatch")
    if receipt.get("mesh_response_commitment_hash") != mesh_response_commitment_hash(
        receipt
    ):
        raise RuntimeError("mesh artifact response commitment mismatch")

    if verification_snapshot is not None:
        from verallm.mesh.verification_snapshot import (
            MeshVerificationSnapshot,
            assert_endpoint_free_payload,
            verify_mesh_verification_snapshot_signature,
        )

        snapshot = (
            verification_snapshot
            if isinstance(verification_snapshot, MeshVerificationSnapshot)
            else MeshVerificationSnapshot.from_dict(verification_snapshot)
        )
        snapshot_now_unix = (
            int(time.time())
            if verification_snapshot_now_unix is None
            else int(verification_snapshot_now_unix)
        )
        if snapshot_now_unix <= 0:
            raise RuntimeError(
                "mesh artifact snapshot validation time is invalid"
            )
        snapshot.validate(require_signature=True)
        snapshot.validate_freshness(now_unix=snapshot_now_unix)
        if not verify_mesh_verification_snapshot_signature(
            snapshot,
            expected_hotkey=snapshot.coordinator.coordinator_hotkey,
            expected_epoch=snapshot.epoch,
            expected_mesh_id=snapshot.mesh_id,
            expected_generation=snapshot.generation,
            expected_coordinator=snapshot.coordinator,
            expected_model=snapshot.model,
            expected_policy=snapshot.policy,
            now_unix=snapshot_now_unix,
        ):
            raise RuntimeError("mesh artifact verification snapshot signature invalid")
        if str(receipt.get("verification_snapshot_hash", "")) != (
            snapshot.snapshot_hash_hex()
        ):
            raise RuntimeError("mesh artifact verification snapshot hash mismatch")
        if int(receipt.get("verification_snapshot_generation", -1)) != int(
            snapshot.generation
        ):
            raise RuntimeError(
                "mesh artifact verification snapshot generation mismatch"
            )
        if int(receipt.get("verification_snapshot_epoch", -1)) != int(snapshot.epoch):
            raise RuntimeError("mesh artifact verification snapshot epoch mismatch")
        if str(receipt.get("mesh_id", "")) != snapshot.mesh_id:
            raise RuntimeError("mesh artifact snapshot mesh_id mismatch")
        if int(receipt.get("uid", -1)) != snapshot.coordinator.coordinator_uid:
            raise RuntimeError("mesh artifact snapshot coordinator uid mismatch")
        if str(receipt.get("hotkey", "")) != (
            snapshot.coordinator.coordinator_hotkey
        ):
            raise RuntimeError("mesh artifact snapshot coordinator hotkey mismatch")
        if str(receipt.get("model_package_hash", "")) != (
            snapshot.model.model_package_hash
        ):
            raise RuntimeError("mesh artifact snapshot package hash mismatch")
        if str(receipt.get("model_tensor_manifest_root", "")) != (
            snapshot.model.model_tensor_manifest_root
        ):
            raise RuntimeError("mesh artifact snapshot tensor root mismatch")
        if receipt.get("rpc_endpoints") != []:
            raise RuntimeError("mesh artifact disclosed private RPC endpoints")
        if str(receipt.get("proof_policy_profile", "")) != snapshot.policy.profile:
            raise RuntimeError("mesh artifact proof policy profile mismatch")
        if snapshot.policy.base_proof_sample_bps != PROOF_SAMPLE_BPS_DENOMINATOR:
            raise RuntimeError("mesh snapshot must require base proof on every response")
        # Decode openings ride the hard tier; a light draw owes nothing at
        # reveal, so the policy may pin any decode rate down to zero. What
        # matters is that every wire value equals the pinned rate — those
        # equality checks live below and in the request builder.
        if not 0 <= int(snapshot.policy.decode_sample_bps) <= (
            PROOF_SAMPLE_BPS_DENOMINATOR
        ):
            raise RuntimeError("mesh snapshot decode rate is out of range")
        if snapshot.policy.deferred_proof_enabled:
            raise RuntimeError("mesh snapshot active policy must disable deferred proof")
        if str(snapshot.policy.challenge_scheme) != (
            VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
        ):
            raise RuntimeError(
                "mesh snapshot challenge scheme must require validator postcommit"
            )
        if int(receipt.get("proof_sample_bps", -1)) != int(
            snapshot.policy.base_proof_sample_bps
        ):
            raise RuntimeError("mesh artifact base proof policy mismatch")
        expected_hard_bps = int(
            getattr(snapshot.policy, "postcommit_hard_audit_bps", 10_000)
        )
        # Bound so a coordinator cannot self-declare a lower hard-audit rate
        # than the pinned snapshot policy and quietly resolve everything
        # light; absent (pre-tiering receipt) is only acceptable when the
        # policy itself is pre-tiering full-hard.
        receipt_hard_bps = int(
            receipt.get("proof_postcommit_hard_bps", 10_000)
        )
        if receipt_hard_bps != expected_hard_bps:
            raise RuntimeError(
                "mesh artifact postcommit hard-audit rate mismatch"
            )
        if receipt.get("proof_configured_required") is not True:
            raise RuntimeError("mesh artifact proof policy is not configured")
        if receipt.get("proof_capture_required") is not True:
            raise RuntimeError("mesh artifact base proof capture is not active")
        if receipt.get("proof_postcommit") is not True:
            raise RuntimeError("mesh artifact postcommit proof policy is not active")
        if receipt.get("proof_challenge_kind") != (
            VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
        ):
            raise RuntimeError("mesh artifact postcommit challenge kind mismatch")
        if int(receipt.get("proof_policy_version", 0)) != 2:
            raise RuntimeError("mesh artifact postcommit policy version mismatch")
        postcommit_origin = not bool(
            receipt.get("proof_postcommit_finalized", False)
        )
        if postcommit_origin:
            if receipt.get("proof_sampled") not in (False, None):
                raise RuntimeError(
                    "mesh postcommit origin must not claim a sampled proof"
                )
            if receipt.get("proof_required") not in (False, None):
                raise RuntimeError(
                    "mesh postcommit origin must not require a revealed proof"
                )
            if receipt.get("proof_beacon", ""):
                raise RuntimeError(
                    "mesh postcommit origin must not expose its proof beacon"
                )
        else:
            # A finalized light-tier resolution legitimately has
            # proof_sampled=False (the nonce draw chose light); its tier
            # fields are bound to the recomputed decision upstream. v3
            # shape: a light draw owes NOTHING at reveal time - the origin
            # receipt with its inline light proof is the light tier, so a
            # finalized light resolution carries no revealed obligation.
            finalized_light = (
                str(receipt.get("proof_audit_tier", "")) == "light"
            )
            if receipt.get("proof_sampled") is not True and not finalized_light:
                raise RuntimeError("mesh artifact base proof was not sampled")
            if finalized_light:
                if receipt.get("proof_required") not in (False, None):
                    raise RuntimeError(
                        "mesh light resolution must not require a "
                        "revealed proof"
                    )
            elif receipt.get("proof_required") is not True:
                raise RuntimeError("mesh artifact base proof is not required")
        if receipt.get("verified_sampler_required") is not True:
            raise RuntimeError("mesh artifact verified sampler is not active")
        sampler_mode = str(receipt.get("verified_sampler_mode", "") or "")
        if sampler_mode == VERIFIED_GGUF_SAMPLER_MODE:
            if receipt.get("verified_sampler_controls_hash") != (
                verified_gguf_sampler_controls_hash()
            ):
                raise RuntimeError(
                    "mesh artifact verified sampler controls mismatch"
                )
        elif sampler_mode == VERIFIED_GGUF_SAMPLED_LIGHT_MODE:
            # Committed stochastic profile: sound ONLY on the light tier
            # (top-k membership); the exact-argmax hard relation has no
            # sampled counterpart, so a sampled receipt must be un-drawable
            # for hard and must never claim a hard resolution.
            if str(receipt.get("proof_audit_tier", "")) == "hard":
                raise RuntimeError(
                    "sampled receipts cannot satisfy the hard audit tier"
                )
            if int(receipt.get("proof_postcommit_hard_bps", 0) or 0) != 0:
                raise RuntimeError(
                    "sampled receipts require a zero hard-audit rate"
                )
            controls = receipt.get("verified_sampler_controls")
            if not isinstance(controls, Mapping):
                raise RuntimeError(
                    "sampled receipt is missing its committed sampler controls"
                )
            top_k = int(controls.get("top_k", 0) or 0)
            if not (0 < top_k <= VERATHOS_GGUF_DECODE_AUDIT_TOP_K):
                raise RuntimeError(
                    "sampled receipt sampler support exceeds the decode "
                    "audit width"
                )
            if receipt.get("verified_sampler_controls_hash") != (
                verified_gguf_sampler_controls_hash(controls)
            ):
                raise RuntimeError(
                    "mesh artifact verified sampler controls mismatch"
                )
        else:
            raise RuntimeError("mesh artifact verified sampler mode mismatch")
        request_policy = openai_request.get("verathos", {})
        if not isinstance(request_policy, dict):
            request_policy = {}
        if str(request_policy.get("verification_snapshot_hash", "")) != (
            snapshot.snapshot_hash_hex()
        ):
            raise RuntimeError("mesh request verification snapshot hash mismatch")
        # Exactly one approved decode rate exists per snapshot.  Accepting a
        # range here would put a caller-chosen value back on the wire, which
        # is the signal that let a coordinator separate canaries from organic
        # traffic in the first place.
        requested_decode_bps = request_policy.get("decode_audit_bps")
        expected_decode_bps = snapshot.policy.decode_sample_bps
        if (
            requested_decode_bps is not None
            and int(requested_decode_bps) != int(expected_decode_bps)
        ):
            raise RuntimeError("mesh request decode policy is not snapshot-approved")
        if int(receipt.get("decode_audit_bps", -1)) != int(expected_decode_bps):
            raise RuntimeError("mesh artifact decode proof policy mismatch")
        if int(receipt.get("decode_audit_top_k", -1)) != (
            VERATHOS_GGUF_DECODE_AUDIT_TOP_K
        ):
            raise RuntimeError("mesh artifact decode audit top-k mismatch")
        if int(receipt.get("proof_ops_per_request", -1)) != int(
            snapshot.policy.proof_ops_per_request
        ):
            raise RuntimeError("mesh artifact proof op policy mismatch")
        if int(receipt.get("proof_trace_candidates_per_request", -1)) != int(
            snapshot.policy.proof_trace_candidates_per_request
        ):
            raise RuntimeError("mesh artifact proof trace candidate policy mismatch")
        if bool(receipt.get("proof_deferred", False)) != bool(
            snapshot.policy.deferred_proof_enabled
        ):
            raise RuntimeError("mesh artifact deferred proof policy mismatch")
        if str(receipt.get("proof_trace_manifest_format", "")) != (
            snapshot.policy.trace_manifest_format
        ):
            raise RuntimeError("mesh artifact trace manifest policy mismatch")
        proof_receipt_payload = artifact.get("proof_receipts", [])
        proof_payload_payload = artifact.get("proof_payloads", [])
        public_proof_artifacts = {
            "proof_receipts": proof_receipt_payload,
            "proof_payloads": proof_payload_payload,
        }
        assert_endpoint_free_payload(
            public_proof_artifacts,
            path="mesh_artifact",
        )
        _assert_public_proof_artifact_value(
            public_proof_artifacts,
            path="mesh_artifact",
        )
        expected_coordinator_hotkey = snapshot.coordinator.coordinator_hotkey
        expected_coordinator_uid = snapshot.coordinator.coordinator_uid
        require_coordinator_signature = True

    if expected_coordinator_hotkey:
        if str(receipt.get("hotkey", "")) != expected_coordinator_hotkey:
            raise RuntimeError("mesh artifact coordinator hotkey mismatch")
    if expected_coordinator_uid is not None:
        if int(receipt.get("uid", -1)) != int(expected_coordinator_uid):
            raise RuntimeError("mesh artifact coordinator uid mismatch")
    if expected_validator_hotkey:
        if str(receipt.get("proof_validator_hotkey", "")) != str(
            expected_validator_hotkey
        ):
            raise RuntimeError("mesh artifact validator hotkey mismatch")
    if require_coordinator_signature or expected_coordinator_hotkey:
        signer = expected_coordinator_hotkey or str(receipt.get("hotkey", ""))
        signature = str(receipt.get("signature", ""))
        if not signer or not signature:
            raise RuntimeError("mesh artifact coordinator signature missing")
        from verallm.mesh.receipt_signing import verify_receipt_signature

        if not verify_receipt_signature(
            str(receipt.get("receipt_hash", "")),
            signature,
            signer,
        ):
            raise RuntimeError("mesh artifact coordinator signature invalid")

    prompt_source = str(receipt.get("prompt_token_source", ""))
    if prompt_source:
        prompt_ids = _normalize_token_ids(artifact.get("prompt_token_ids"))
        if prompt_ids is None:
            raise RuntimeError("mesh artifact prompt_token_ids must be a token id list")
        if int(receipt.get("prompt_token_count", -1)) != len(prompt_ids):
            raise RuntimeError("mesh artifact prompt_token_count mismatch")
        if receipt.get("prompt_token_ids_hash") != prompt_token_ids_hash(prompt_ids):
            raise RuntimeError("mesh artifact prompt_token_ids_hash mismatch")
        if not receipt.get("prompt_template_hash"):
            raise RuntimeError("mesh artifact prompt_template_hash missing")
    elif receipt.get("proof_metadata_required"):
        raise RuntimeError("mesh artifact missing prompt token binding")

    completion_ids: list[int] = []
    completion_source = str(receipt.get("completion_token_source", ""))
    if completion_source:
        parsed_completion_ids = _normalize_token_ids(artifact.get("completion_token_ids"))
        if parsed_completion_ids is None:
            raise RuntimeError("mesh artifact completion_token_ids must be a token id list")
        completion_ids = parsed_completion_ids
        if int(receipt.get("completion_token_count", -1)) != len(completion_ids):
            raise RuntimeError("mesh artifact completion_token_count mismatch")
        if receipt.get("completion_token_ids_hash") != completion_token_ids_hash(
            completion_ids
        ):
            raise RuntimeError("mesh artifact completion_token_ids_hash mismatch")
    elif receipt.get("proof_metadata_required") and not openai_request.get("stream"):
        raise RuntimeError("mesh artifact missing completion token binding")
    elif receipt.get("decode_audit_required"):
        parsed_decode_ids = _normalize_token_ids(
            receipt.get("decode_audit_completion_token_ids")
        )
        if parsed_decode_ids is None:
            parsed_decode_ids = _normalize_token_ids(artifact.get("completion_token_ids"))
        if parsed_decode_ids is None:
            raise RuntimeError("mesh artifact missing decode audit completion tokens")
        completion_ids = parsed_decode_ids
        if int(receipt.get("completion_token_count", -1)) != len(completion_ids):
            raise RuntimeError("mesh artifact decode audit completion_token_count mismatch")
        if int(receipt.get("decode_audit_completion_token_count", -1)) != len(
            completion_ids
        ):
            raise RuntimeError(
                "mesh artifact decode audit completion token count mismatch"
            )
        if receipt.get("decode_audit_completion_token_ids_hash") != (
            completion_token_ids_hash(completion_ids)
        ):
            raise RuntimeError(
                "mesh artifact decode audit completion token hash mismatch"
            )

    if receipt.get("decode_audit_required"):
        if not completion_ids:
            raise RuntimeError("mesh artifact missing decode audit completion tokens")
        if not receipt.get("decode_audit_completion_token_ids_hash"):
            raise RuntimeError("mesh artifact missing decode audit completion token hash")
        if int(receipt.get("decode_audit_completion_token_count", -1)) != len(
            completion_ids
        ):
            raise RuntimeError(
                "mesh artifact decode audit completion token count mismatch"
            )
        if receipt.get("decode_audit_completion_token_ids_hash") != (
            completion_token_ids_hash(completion_ids)
        ):
            raise RuntimeError(
                "mesh artifact decode audit completion token hash mismatch"
            )

    verify_mesh_proof_sampling_fields(receipt, openai_request)
    deferred_decision = deferred_audit_decision(
        receipt,
        randomness=deferred_randomness,
    )

    if spec is not None:
        expected_fields: dict[str, Any] = {
            "mesh_id": spec.mesh_id,
            "mesh_spec_hash": spec.spec_hash_hex(),
            "stage_assignment_hash": spec.stage_assignment_hash_hex(),
            "model_package_hash": spec.model_package_hash,
            "model_tensor_manifest_root": spec.model_tensor_manifest_root,
            "rpc_endpoints": [],
            "rpc_plan_hash": rpc_plan_from_mesh(spec).plan_hash_hex(),
        }
        if member_index is not None:
            member = spec.members[int(member_index)]
            expected_fields.update(
                {
                    "uid": member.uid,
                    "hotkey": member.hotkey,
                    "endpoint": member.endpoint,
                    "stage_index": member.stage_index,
                    "layer_start": member.layers.start,
                    "layer_end": member.layers.end,
                }
            )
        for field, expected in expected_fields.items():
            if receipt.get(field) != expected:
                raise RuntimeError(f"mesh artifact receipt {field} mismatch")
        if int(receipt.get("decode_audit_bps", 0)) > 0:
            expected_decode_stage = decode_audit_stage_index_for_spec(spec)
            if int(receipt.get("decode_audit_stage_index", -1)) != expected_decode_stage:
                raise RuntimeError(
                    "mesh artifact decode_audit_stage_index mismatch"
                )

    # A light-tier postcommit resolution is legitimate ONLY when the
    # receipt's tier fields say so, and those fields are bound to the
    # verifier-recomputed nonce decision by verify_mesh_postcommit_artifact
    # BEFORE this strict pass runs (the only require_cryptographic_proof
    # caller). A forged light claim fails that field comparison first.
    light_tier_receipt = bool(
        str(receipt.get("proof_audit_tier", "")) == "light"
        and receipt.get("proof_sampled") is not True
        and str(receipt.get("proof_challenge_kind", ""))
        == VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
    )
    if require_cryptographic_proof and not light_tier_receipt:
        if receipt.get("proof_required") is not True:
            raise RuntimeError("mesh artifact cryptographic proof was not sampled")
        if receipt.get("proof_mode") != VERATHOS_GGML_GEMM_PROOF_MODE:
            raise RuntimeError("mesh artifact cryptographic GGML GEMM proof required")
        if not receipt.get("model_tensor_manifest_root"):
            raise RuntimeError("mesh artifact GGUF tensor manifest root required")
    elif require_cryptographic_proof:
        # v3 shape: a light draw owes no revealed obligation - the origin
        # receipt's inline light proof is the light tier, so the finalized
        # resolution carries no proof material at all. The tier fields
        # themselves are bound to the verifier-recomputed nonce decision
        # (see comment above), so this branch cannot be forged into.
        if receipt.get("proof_required") not in (False, None):
            raise RuntimeError(
                "mesh light resolution must not require a revealed proof"
            )

    proof_receipts = artifact.get("proof_receipts", [])
    proof_payloads = artifact.get("proof_payloads", [])
    if required_proof_stage_indexes is None:
        if (
            spec is not None
            and require_spec_proof_stage_coverage
            and bool(receipt.get("proof_required", False))
        ):
            if member_index is not None:
                selected_member = spec.members[int(member_index)]
                expected_proof_stage_indexes = (
                    {int(selected_member.stage_index)}
                    if selected_member.layers.end > selected_member.layers.start
                    else set()
                )
            else:
                expected_proof_stage_indexes = {
                    int(member.stage_index)
                    for member in spec.members
                    if int(member.layers.end) > int(member.layers.start)
                }
        else:
            expected_proof_stage_indexes = set()
    else:
        expected_proof_stage_indexes = {
            int(item) for item in required_proof_stage_indexes
        }
    if receipt.get("proof_receipt_root"):
        if not isinstance(proof_receipts, list):
            raise RuntimeError("mesh artifact proof_receipts must be a list")
        if str(receipt.get("proof_receipt_format", "")) == "opaque_stage_v2":
            if verification_snapshot is None:
                raise RuntimeError(
                    "opaque mesh proof receipts require a verification snapshot"
                )
            public_stage_receipts = verify_mesh_stage_proof_receipts_for_snapshot(
                receipt,
                proof_receipts,
                snapshot,
                require_complete_coverage=bool(
                    receipt.get("proof_required", False)
                ),
            )
            if int(receipt.get("decode_audit_bps", 0)) > 0:
                final_snapshot_stage = next(
                    (
                        stage
                        for stage in snapshot.stages
                        if int(stage.layer_end) == int(snapshot.model.total_layers)
                    ),
                    None,
                )
                final_stage_indexes = {
                    int(item.stage_index)
                    for item in public_stage_receipts
                    if final_snapshot_stage is not None
                    and item.stage_id == final_snapshot_stage.stage_id
                }
                if len(final_stage_indexes) != 1 or int(
                    receipt.get("decode_audit_stage_index", -1)
                ) != next(iter(final_stage_indexes), -1):
                    raise RuntimeError("mesh artifact decode stage owner mismatch")
        elif verification_snapshot is not None:
            raise RuntimeError("mesh artifact exposed private stage proof receipts")
        elif spec is not None:
            verify_llama_graph_proof_receipts_for_mesh(
                receipt,
                proof_receipts,
                spec,
                required_stage_indexes=expected_proof_stage_indexes,
            )
        else:
            verify_llama_graph_proof_receipts(receipt, proof_receipts)
        if receipt.get("proof_mode") != VERATHOS_GGML_TRACE_PROOF_MODE:
            if not isinstance(proof_payloads, list):
                raise RuntimeError("mesh artifact proof_payloads must be a list")
            from verallm.mesh.ggml_proof import (
                verify_mesh_proof_payloads_any_tier,
            )

            result = verify_mesh_proof_payloads_any_tier(
                proof_payloads,
                proof_receipts,
                mesh_receipt=receipt,
                completion_token_ids=completion_ids or None,
                postcommit_light_ok=light_tier_receipt,
            )
            if not result.verified:
                raise RuntimeError("mesh artifact proof verification failed: " + result.message)
        if receipt.get("proof_receipt_verified") is not True:
            raise RuntimeError("mesh artifact proof receipt verification flag mismatch")
        if require_cryptographic_proof and receipt.get("verified") is not True:
            raise RuntimeError("mesh artifact verified flag mismatch")
    elif receipt.get("proof_required"):
        raise RuntimeError("mesh artifact missing required proof receipts")
    elif (
        require_deferred_proof_if_sampled
        and bool(deferred_decision.get("sampled", False))
    ):
        raise RuntimeError("mesh artifact missing deferred sampled proof receipts")
    elif require_configured_proof and (
        receipt.get("proof_configured_required") is not True
        or receipt.get("proof_capture_required") is not True
    ):
        raise RuntimeError("mesh artifact proof capture was not configured")

    if receipt.get("decode_audit_required"):
        if not isinstance(proof_payloads, list) or not proof_payloads:
            raise RuntimeError("mesh artifact missing decode audit proof payloads")
        if not completion_ids:
            raise RuntimeError("mesh artifact missing decode audit completion tokens")
        from verallm.mesh.ggml_proof import verify_ggml_decode_audit_payloads

        decode_result = verify_ggml_decode_audit_payloads(
            receipt,
            proof_payloads,
            completion_token_ids=completion_ids,
            receipts=proof_receipts,
        )
        if not decode_result.verified:
            raise RuntimeError(
                "mesh artifact decode audit verification failed: "
                + decode_result.message
            )
        if receipt.get("decode_audit_verified") is not True:
            raise RuntimeError("mesh artifact decode audit verification flag mismatch")

    return True


def _sse_event_bytes(event: str, payload: dict[str, Any] | str) -> bytes:
    lines = []
    if event:
        lines.append(f"event: {event}")
    if isinstance(payload, str):
        data = payload
    else:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _parse_sse_block(block: bytes) -> tuple[str, list[str]]:
    event = ""
    data_lines: list[str] = []
    for raw_line in block.decode("utf-8", errors="replace").splitlines():
        if raw_line.startswith("event:"):
            event = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            data_lines.append(raw_line[len("data:"):].strip())
    return event, data_lines


def _read_sse_block(resp) -> bytes:
    lines = []
    while True:
        line = resp.readline()
        if not line:
            return b"".join(lines)
        lines.append(line)
        if line in (b"\n", b"\r\n"):
            return b"".join(lines)


def _accumulate_openai_stream_chunk(state: dict[str, Any], chunk: dict[str, Any]) -> None:
    state.setdefault("id", chunk.get("id") or "")
    state.setdefault("created", chunk.get("created") or 0)
    if chunk.get("model"):
        state["model"] = chunk["model"]
    if chunk.get("usage"):
        state["usage"] = chunk["usage"]
    timings = chunk.get("timings")
    if isinstance(timings, dict):
        state["timings"] = timings
    chunk_tokens = _normalize_token_ids(chunk.get("tokens"))
    if chunk_tokens:
        state.setdefault("completion_token_ids", []).extend(chunk_tokens)
    choices = chunk.get("choices", [])
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_tokens = _normalize_token_ids(choice.get("tokens"))
        if choice_tokens:
            state.setdefault("completion_token_ids", []).extend(choice_tokens)
        # The carrier llama.cpp actually uses while streaming.  Entries are
        # dicts with an "id", which _normalize_token_ids already understands.
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            logprob_tokens = _normalize_token_ids(logprobs.get("content"))
            if logprob_tokens:
                state.setdefault("completion_token_ids", []).extend(
                    logprob_tokens
                )
        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            role = delta.get("role")
            if role:
                state["role"] = role
            content = delta.get("content")
            if content:
                state.setdefault("content_parts", []).append(str(content))
            # Reasoning models stream their thinking as a separate
            # delta.reasoning_content channel (llama-server
            # --reasoning-format deepseek). It must survive aggregation:
            # the aggregate is what the receipt hash commits to, and the
            # validator counts reasoning as real generated output.
            reasoning = delta.get("reasoning_content")
            if reasoning:
                state.setdefault("reasoning_parts", []).append(str(reasoning))
        if choice.get("finish_reason") is not None:
            state["finish_reason"] = choice.get("finish_reason")
        if choice.get("verathos_slot_id") is not None:
            try:
                state["slot_id"] = int(choice["verathos_slot_id"])
            except (TypeError, ValueError):
                pass


def _stream_aggregate_response(
    *,
    request_id: str,
    openai_request: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    content = "".join(state.get("content_parts", []))
    usage = state.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    timings = state.get("timings")
    if (
        not usage
        and isinstance(timings, dict)
        and timings.get("predicted_n") is not None
    ):
        completion_tokens = int(timings.get("predicted_n") or 0)
        prompt_tokens = int(timings.get("prompt_n") or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    message: dict[str, Any] = {
        "role": state.get("role") or "assistant",
        "content": content,
    }
    # Aggregated BEFORE the receipt path serializes the response into
    # response_hash: reasoning_content is part of the committed response,
    # never a post-hash mutation.
    reasoning = "".join(state.get("reasoning_parts", []))
    if reasoning:
        message["reasoning_content"] = reasoning
    aggregate = {
        "id": state.get("id") or f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(state.get("created") or time.time()),
        "model": state.get("model") or openai_request.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": state.get("finish_reason") or "stop",
            }
        ],
        "usage": usage,
    }
    if state.get("slot_id") is not None:
        aggregate["choices"][0]["verathos_slot_id"] = int(state["slot_id"])
    # Surface the ids the stream carried so the receipt path finds them
    # through the same accessor a non-streamed response uses.
    token_ids = state.get("completion_token_ids")
    if isinstance(token_ids, list) and token_ids:
        aggregate["tokens"] = [int(token_id) for token_id in token_ids]
    return aggregate


def make_worker_server(
    *,
    capability: CapabilityAd,
    host: str = "127.0.0.1",
    port: int = DEFAULT_WORKER_PORT,
    mesh_spec: MeshSpec | None = None,
    mesh_spec_loader: MeshSpecLoader | None = None,
    join_handler: JoinHandler | None = None,
    mesh_spec_handler: MeshSpecHandler | None = None,
    mesh_update_handler: MeshUpdateHandler | None = None,
    backend_url: str = "",
    proof_url: str = "",
    require_proof: bool = False,
    proof_trace_enable_file: str | Path = "",
    proof_trace_dir: str | Path = "",
    proof_gguf_manifest_path: str | Path = "",
    proof_tolerance_abs: float = 8e-2,
    proof_tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    proof_spot_checks: int = 8,
    proof_warmup: bool = False,
    proof_decode_projection_warmup: bool = False,
    proof_sample_bps: int = PROOF_SAMPLE_BPS_DENOMINATOR,
    defer_proof: bool = False,
    proof_ops_per_request: int = 1,
    proof_trace_candidates_per_request: int = 8,
    decode_audit_bps: int = 0,
    decode_audit_top_k: int = 8,
    proof_artifact_timeout: float = DEFAULT_PROOF_ARTIFACT_TIMEOUT,
    prompt_binding_cache_enabled: bool = False,
    llama_n_parallel: int = 1,
    llama_n_ubatch: int = 0,
    slot_view_template_warmup: bool = False,
    # Local-stage serving: llama-server computes the first member's stage
    # on the coordinator's local devices, so that member (which has NO
    # rpc_endpoint) must still be armed remotely over its trace-capture
    # channel. Never set for legacy self-capturing stages.
    local_stage_capture: bool = False,
    proof_trace_manifest_format: str = "",
    receipt_signer: "Callable[[str], str] | None" = None,
    stage_receipt_signer: "Callable[[str], str] | None" = None,
    stage_proof_key: str = "",
    server_role: str = "auto",
    validator_auth_enabled: bool = False,
    validator_allowlist_path: str | Path = "",
    validator_allowlist_max_age_seconds: float = (
        DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS
    ),
    require_validator_nonce: bool = False,
    allow_loopback_dev_validator_routes: bool = False,
    internal_auth_secret: str | bytes = "",
    evm_address: str = "",
    evm_private_key: str = "",
    # Alternative to ``evm_private_key`` for a driver that holds no wallet:
    # signs the identity-challenge payload through its pool manager, which
    # holds the coordinator EVM key. Mutually exclusive with the local key.
    evm_challenge_signer: Callable[[bytes], str] | None = None,
    # The fitted unified-KV budget this serve launched llama with (0 =
    # model trained maximum, resolved from llama /props); sizes the
    # admission ledger.
    llama_ctx_budget: int = 0,
    # Drain state file written by the pool worker daemon's capacity-audit
    # worker. While it marks an active window, admission returns the
    # ORDINARY busy 503 — byte-identical to a saturation rejection, so the
    # refusal can never fingerprint an audit window (canary-oracle rule).
    capacity_drain_file: str | Path = "",
    slot_state_dir: str | Path = "",
    # Signed capacity roster file (same daemon); served on GET
    # /capacity/roster so validators can pull the mesh's GPU obligation.
    capacity_roster_file: str | Path = "",
    max_request_body_bytes: int = 128 * 1024 * 1024,
    verification_snapshot_loader: VerificationSnapshotLoader | None = None,
) -> ThreadingHTTPServer:
    """Create a worker HTTP server without starting it.

    ``receipt_signer`` signs the coordinator's top-level receipt.  A distinct
    ``stage_receipt_signer`` signs endpoint-free proof receipts on the worker
    that produced them; the two signature domains and keys must not be reused.
    """

    capability.validate()
    server_role = str(server_role or "auto").strip().lower()
    if server_role not in {"auto", "coordinator", "worker"}:
        raise ValueError("server_role must be auto, coordinator, or worker")
    validator_auth_enabled = bool(validator_auth_enabled)
    require_validator_nonce = bool(require_validator_nonce)
    allow_loopback_dev_validator_routes = bool(
        allow_loopback_dev_validator_routes
    )
    max_request_body_bytes = int(max_request_body_bytes)
    if max_request_body_bytes <= 0:
        raise ValueError("max_request_body_bytes must be positive")
    if validator_auth_enabled and not validator_allowlist_path:
        validator_allowlist_path = os.environ.get(
            "VERATHOS_VALIDATORS_PATH",
            str(Path.home() / ".verathos" / "validators.json"),
        )
    validator_allowlist = (
        ValidatorAllowlist(
            validator_allowlist_path,
            reload_interval_seconds=0,
            max_file_age_seconds=validator_allowlist_max_age_seconds,
        )
        if validator_auth_enabled
        else None
    )
    validator_request_replays = RequestReplayCache()
    internal_request_replays = RequestReplayCache()
    validator_nonce_replays = RequestReplayCache()
    validator_challenge_commitment_replays = RequestReplayCache()
    postcommit_origin_lock = threading.Lock()
    postcommit_origins: dict[
        tuple[str, str, str],
        tuple[
            float,
            bytes,
            int,
            int,
        ],
    ] = {}
    postcommit_finalized: dict[
        tuple[str, str, str],
        tuple[
            float,
            int,
            tuple[str, str, str],
            bytes,
        ],
    ] = {}
    postcommit_finalized_sequence = 0
    postcommit_finalizations: dict[
        tuple[str, str, str],
        tuple[tuple[str, str, str], object],
    ] = {}
    postcommit_active_requests: dict[str, int] = {}
    internal_auth_configured = bool(internal_auth_secret)
    if allow_loopback_dev_validator_routes and validator_auth_enabled:
        raise ValueError(
            "loopback development mode cannot be combined with validator auth"
        )
    if verification_snapshot_loader is not None and server_role != "worker":
        if not validator_auth_enabled or not require_validator_nonce:
            raise ValueError(
                "snapshot-bound coordinators require validator auth and fresh nonces"
            )
        if allow_loopback_dev_validator_routes:
            raise ValueError(
                "snapshot-bound coordinators cannot enable unsigned development routes"
            )
    if (
        verification_snapshot_loader is not None
        and server_role == "worker"
        and not internal_auth_configured
        and not allow_loopback_dev_validator_routes
    ):
        raise ValueError(
            "snapshot-bound workers require internal HTTP authentication"
        )

    stage_proof_key = str(stage_proof_key or "").strip()
    if bool(stage_receipt_signer) != bool(stage_proof_key):
        raise ValueError(
            "stage_receipt_signer and stage_proof_key must be configured together"
        )
    if stage_receipt_signer is not None:
        from verallm.mesh.receipt_signing import (
            STAGE_PROOF_KEY_SCHEME,
            validate_stage_proof_public_key,
            verify_stage_proof_receipt_signature,
        )

        validate_stage_proof_public_key(stage_proof_key, STAGE_PROOF_KEY_SCHEME)
        stage_probe_hash = "00" * 32
        try:
            stage_probe_signature = str(stage_receipt_signer(stage_probe_hash) or "")
        except Exception as exc:
            raise ValueError("stage proof receipt signer is unavailable") from exc
        if not verify_stage_proof_receipt_signature(
            stage_probe_hash,
            stage_probe_signature,
            stage_proof_key,
            STAGE_PROOF_KEY_SCHEME,
        ):
            raise ValueError("stage proof receipt signer does not match its proof key")

    evm_address = str(evm_address or "").strip()
    evm_private_key = str(evm_private_key or "").strip()
    if evm_challenge_signer is not None:
        if evm_private_key:
            raise ValueError(
                "configure either evm_private_key or evm_challenge_signer, not both"
            )
        if not evm_address:
            raise ValueError("evm_challenge_signer requires evm_address")
    elif bool(evm_address) != bool(evm_private_key):
        raise ValueError("evm_address and evm_private_key must be configured together")
    if evm_private_key:
        from eth_account import Account

        derived_address = Account.from_key(evm_private_key).address
        if derived_address.lower() != evm_address.lower():
            raise ValueError("evm_private_key does not match evm_address")
        evm_address = derived_address
    secure_receipt_signing = bool(
        validator_auth_enabled and server_role != "worker"
    )
    if secure_receipt_signing and receipt_signer is None:
        raise ValueError("validator-authenticated coordinators require receipt signing")
    if secure_receipt_signing:
        from verallm.mesh.receipt_signing import verify_receipt_signature

        signer_probe_hash = "00" * 32
        try:
            signer_probe_signature = str(receipt_signer(signer_probe_hash) or "")
        except Exception as exc:
            # Include the cause: "unavailable" alone hid a TLS verification
            # failure on the delegated signing channel for a whole debug
            # round.
            raise ValueError(
                f"coordinator receipt signer is unavailable: {exc}"
            ) from exc
        if not verify_receipt_signature(
            signer_probe_hash,
            signer_probe_signature,
            capability.hotkey,
        ):
            raise ValueError("coordinator receipt signer does not match its hotkey")
        if not evm_address or (
            not evm_private_key and evm_challenge_signer is None
        ):
            raise ValueError(
                "validator-authenticated coordinators require EVM identity challenge keys"
            )
    backend_url = backend_url.rstrip("/")
    proof_url = proof_url.rstrip("/")
    proof_sample_bps = normalize_proof_sample_bps(proof_sample_bps)
    defer_proof = bool(defer_proof)
    decode_audit_bps = normalize_proof_sample_bps(decode_audit_bps)
    proof_ops_per_request = int(proof_ops_per_request)
    proof_trace_candidates_per_request = int(proof_trace_candidates_per_request)
    decode_audit_top_k = max(1, int(decode_audit_top_k))
    proof_artifact_timeout = float(proof_artifact_timeout)
    llama_n_parallel = max(1, int(llama_n_parallel))
    llama_n_ubatch = max(0, int(llama_n_ubatch))
    # Candidate witnesses are captured during serve when proof sampling is on
    # and a candidate count is requested; the audit then proves a stored
    # candidate with no replay. This applies to every compute stage — a
    # single-node coordinator, each rpc-worker, and a coordinator computing
    # some layers alongside remote workers. The coordinator aggregates each
    # stage's candidate commitment (collect_trace_commitments) into one mesh
    # root; multi-slot (--parallel > 1) still uses slot-view per-slot
    # attribution.
    _mesh_has_remote_members = bool(
        mesh_spec is not None
        and any(
            member.endpoint != capability.endpoint
            for member in mesh_spec.members
        )
    )
    proof_trace_candidate_capture = (
        bool(require_proof) and int(proof_trace_candidates_per_request) > 0
    )
    # Slot-view attribution is the compact organic commitment format for local
    # proof-traced llama.cpp backends. Slot-view (template + replay) is
    # required for multi-slot (--parallel > 1) per-slot attribution, and for
    # ANY decode-audited serve regardless of parallelism. The audit
    # regenerates under the exclusive replay with teacher-forced probes and
    # the slot-view selection carries the decode-audit leaves.

    # Single-slot decode-audited serves cannot keep the candidate-capture
    # path: op hooks do not run on cudaGraphLaunch, so graph-replayed decode
    # steps never capture candidate witnesses during serve, and the legacy
    # sampled solo re-serve can NEVER semantically match a graph-served
    # original (graph replay reuses capture-time kernel launch geometry
    # while an eager re-serve re-picks it; the shifted reduction order flips
    # a near-tie greedy token deterministically; live-measured: eager and
    # graph serves of the same prompt return different content, so the
    # strict semantic-equality gate rejected every warmed-graph serve).
    # Teacher-forced probes bind to committed token ids with top-k
    # acceptance, which absorbs exactly this jitter.

    # The audit disjunct is gated on the v3 manifest format because
    # slot-view leaves need v3 rows + graph markers to exist at all; the
    # CLI's decide_trace_manifest_format always yields compact-raw-v3 when
    # the audit and base gates are configured, so every production audited
    # serve takes slot-view. Direct callers with legacy fixture formats
    # (unit tests, capture-less builds) keep the candidate-capture flow,
    # which is sound whenever serve-time capture covers the audited ops.

    # Non-audited single-slot serves keep the candidate-capture decode path
    # (trace_candidate_set_v1, O(1) in generation length, MoE-safe).
    slot_view_required = bool(
        backend_url
        and proof_trace_dir
        and (
            llama_n_parallel > 1
            or (
                decode_audit_bps > 0
                and proof_trace_manifest_format == "compact-raw-v3"
            )
            or (
                slot_view_template_warmup
                and not proof_trace_candidate_capture
            )
        )
    )
    if slot_view_required and llama_n_ubatch <= 0:
        # llama-server defaults to 512 when the operator does not pass
        # --ubatch-size/-ub. The CLI normally supplies this value explicitly;
        # keep direct test/in-process users on the same default.
        llama_n_ubatch = 512
    organic_capture_skip_enabled = os.environ.get(
        "VERATHOS_MESH_ORGANIC_CAPTURE_SKIP", "1"
    ).strip().lower() not in ("0", "false", "no")

    def organic_capture_structurally_unwitnessed() -> bool:
        """True when a serve-time shared-capture join can produce no witness.

        At --parallel > 1 under the v3 slot-view profile the serve-time
        capture channel is structurally empty: the C-side dump budget is
        zeroed for slot-view serves (the CLI sets TRACE_MAX_OPS_PER_CAPTURE
        and _PER_GRAPH to 0), the tail ring is off (its position binding
        only holds with one slot), and base light leaves are synthesized
        from committed leaf metadata. Every witness a deferred/postcommit
        audit needs comes from the exclusive probe window it opens itself.
        An organic serve joining the shared window therefore contributes
        nothing except drain latency for the next hard audit - it forces
        `exclusive_capture` to wait out the whole stream. Skipping the join
        under exactly these structural conditions is what keeps hard audits
        from blocking organic chat.
        """

        return (
            organic_capture_skip_enabled
            and slot_view_required
            and proof_trace_manifest_format == "compact-raw-v3"
            and llama_n_parallel > 1
        )

    # llama.cpp slot-state save/restore turns the hard-audit probe cost from
    # O(context) into O(restore): a committed serve's final KV state is saved
    # once (validator lane by default - those are the serves later
    # postcommit-audited), and the probe window restores it instead of
    # re-prefilling a possibly-evicted prompt EAGER (the 15-20 min glm case).
    # Purely miner-local acceleration: a wrong/stale/corrupt restore produces
    # wrong logits and the proof fails closed, so availability is never
    # eligibility.
    slot_state_root = Path(slot_state_dir) if slot_state_dir else None
    slot_state_save_enabled = os.environ.get(
        "VERATHOS_MESH_SLOT_STATE_SAVE", "1"
    ).strip().lower() not in ("0", "false", "no")
    slot_state_organic_enabled = os.environ.get(
        "VERATHOS_MESH_SLOT_STATE_ORGANIC", "0"
    ).strip().lower() in ("1", "true", "yes")
    slot_state_min_tokens = int(
        os.environ.get("VERATHOS_MESH_SLOT_STATE_MIN_TOKENS", "8192") or 8192
    )
    slot_state_max_bytes = int(
        os.environ.get(
            "VERATHOS_MESH_SLOT_STATE_MAX_BYTES", str(64 * 1024**3)
        )
        or 64 * 1024**3
    )
    slot_state_save_timeout_s = float(
        os.environ.get("VERATHOS_MESH_SLOT_STATE_SAVE_TIMEOUT_S", "120") or 120
    )
    slot_state_restore_timeout_s = float(
        os.environ.get("VERATHOS_MESH_SLOT_STATE_RESTORE_TIMEOUT_S", "60") or 60
    )

    def _slot_state_stem(request_id: str) -> str:
        return hashlib.sha256(
            str(request_id).encode("utf-8")
        ).hexdigest()[:40]

    def _slot_state_paths(request_id: str) -> tuple[Path, Path] | None:
        if slot_state_root is None:
            return None
        stem = _slot_state_stem(request_id)
        return (
            slot_state_root / f"slot-state-{stem}.vseq",
            slot_state_root / f"slot-state-{stem}.json",
        )

    def _sweep_slot_states() -> None:
        if slot_state_root is None:
            return
        now = time.time()
        entries: list[tuple[float, int, Path]] = []
        for path in slot_state_root.glob("slot-state-*"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if now - stat.st_mtime > PROOF_TRACE_RETENTION_SECONDS:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _mtime, size, _path in entries)
        if total <= slot_state_max_bytes:
            return
        for _mtime, size, path in sorted(entries):
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            if total <= slot_state_max_bytes:
                break

    if slot_state_root is not None:
        try:
            slot_state_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "slot-state dir %s is not writable; save/restore disabled",
                slot_state_root,
            )
            slot_state_root = None

    if slot_state_root is not None and slot_state_save_enabled:

        def _slot_state_janitor() -> None:
            _sweep_slot_states()
            while True:
                time.sleep(60.0)
                try:
                    _sweep_slot_states()
                except Exception as exc:
                    logger.debug("slot-state sweep failed: %s", exc)

        threading.Thread(
            target=_slot_state_janitor,
            name="slot-state-janitor",
            daemon=True,
        ).start()

    def maybe_save_slot_state(
        *,
        request_id: str,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        slot_id: int,
        validator_authenticated: bool,
    ) -> None:
        """Post-response, async: persist the committed slot's KV state.

        Validator-lane serves only by default (they are the postcommit-audit
        universe and arrive a few times per epoch); organics opt in via
        VERATHOS_MESH_SLOT_STATE_ORGANIC=1 above the min-token gate. The
        save runs on a daemon thread because llama serializes it behind the
        slot going idle, and its file write blocks the task loop - never
        the serve epilogue. A save that raced a follow-up request in the
        same slot is detected by token count and deleted (stale); a stale
        file that slips through only costs the restore its speedup - llama
        re-evals from the first divergent position, and the proof math never
        trusts restored state.
        """

        if (
            slot_state_root is None
            or not slot_state_save_enabled
            or not slot_view_required
            or int(slot_id) < 0
            or not prompt_token_ids
        ):
            return
        expected_tokens = len(prompt_token_ids) + len(completion_token_ids)
        if not validator_authenticated:
            if not slot_state_organic_enabled:
                return
            if expected_tokens < slot_state_min_tokens:
                return
        paths = _slot_state_paths(request_id)
        if paths is None:
            return
        state_path, sidecar_path = paths
        prompt_hash = prompt_token_ids_hash(list(prompt_token_ids))
        completion_hash = completion_token_ids_hash(
            list(completion_token_ids)
        )

        def _save() -> None:
            try:
                raw = post_json(
                    f"{backend_url}/slots/{int(slot_id)}?action=save",
                    {"filename": state_path.name},
                    timeout=slot_state_save_timeout_s,
                )
                n_saved = int(raw.get("n_saved", -1) or -1)
                # The slot may have picked up another request between the
                # response epilogue and the deferred save; llama then saved
                # that conversation's state. A shorter sequence cannot
                # contain our prefixes and a much longer one is another
                # conversation - either way the file is stale. (An equal
                # count with different content is harmless: restore falls
                # back to eager re-eval from the divergence point.)
                # Lower bound is expected-1: the FINAL sampled token is
                # appended to the completion but never fed back as decode
                # input, so the slot cache systematically ends one short of
                # prompt+completion (live: every save reported exactly -1).
                # The longest teacher-forced probe prefix is
                # prompt+committed[:len-1] - exactly that cache - so the
                # save still covers every probe.
                if not (
                    expected_tokens - 1 <= n_saved <= expected_tokens + 8
                ):
                    _journal_capture_event(
                        f"slot-state save stale request={request_id[:8]} "
                        f"n_saved={n_saved} expected={expected_tokens}"
                    )
                    try:
                        state_path.unlink()
                    except OSError:
                        pass
                    return
                sidecar_path.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "request_id": str(request_id),
                            "slot_id": int(slot_id),
                            "n_saved": n_saved,
                            "expected_tokens": expected_tokens,
                            "prompt_token_ids_hash": prompt_hash,
                            "completion_token_ids_hash": completion_hash,
                            "model_id": str(
                                getattr(mesh_spec, "model_id", "") or ""
                            ),
                            "created_unix_ns": time.time_ns(),
                            "n_written": int(raw.get("n_written", 0) or 0),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                _journal_capture_event(
                    f"slot-state saved request={request_id[:8]} "
                    f"slot={int(slot_id)} tokens={n_saved} "
                    f"bytes={int(raw.get('n_written', 0) or 0)}"
                )
            except Exception as exc:
                logger.debug(
                    "slot-state save failed for %s: %s", request_id[:8], exc
                )

        threading.Thread(
            target=_save,
            name=f"slot-state-save-{request_id[:8]}",
            daemon=True,
        ).start()

    def _backend_slots_snapshot() -> list[dict[str, Any]]:
        try:
            with urlopen(f"{backend_url}/slots", timeout=5.0) as response:
                slots = json.loads(response.read())
            return slots if isinstance(slots, list) else []
        except Exception:
            return []

    def _maybe_restore_slot_state_for_probes(
        receipt_context: dict[str, Any],
        *,
        probe_slot: int,
        probe_cache: bool,
        timing: dict[str, Any] | None,
    ) -> int:
        """Restore the audited serve's saved KV before teacher-forced probes.

        Fail-open at every step: no state file, hash mismatch, no idle slot,
        HTTP error, no-KV-space - each falls through to today's prompt-cache
        path unchanged. A hot committed slot (its cached token count equals
        the save) skips the restore I/O entirely. Returns the slot the
        probes should pin (the restored slot when a restore ran).
        """

        outcome = "off"
        restore_ms = 0
        try:
            if (
                slot_state_root is None
                or not slot_state_save_enabled
                or not probe_cache
                or not backend_url
            ):
                return probe_slot
            outcome = "miss"
            request_id = str(receipt_context.get("request_id", "") or "")
            paths = _slot_state_paths(request_id)
            if paths is None:
                return probe_slot
            state_path, sidecar_path = paths
            if not state_path.exists() or not sidecar_path.exists():
                return probe_slot
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                outcome = "error"
                return probe_slot
            expected_prompt_hash = str(
                receipt_context.get("prompt_token_ids_hash", "") or ""
            )
            expected_completion_hash = str(
                receipt_context.get("completion_token_ids_hash", "") or ""
            )
            if expected_prompt_hash and str(
                sidecar.get("prompt_token_ids_hash", "")
            ) != expected_prompt_hash:
                outcome = "mismatch"
                return probe_slot
            if expected_completion_hash and str(
                sidecar.get("completion_token_ids_hash", "")
            ) != expected_completion_hash:
                outcome = "mismatch"
                return probe_slot
            n_saved = int(sidecar.get("n_saved", 0) or 0)
            slots = _backend_slots_snapshot()
            by_id = {int(item.get("id", -1)): item for item in slots}
            committed = by_id.get(int(probe_slot))
            if (
                committed is not None
                and not committed.get("is_processing")
                and int(committed.get("n_prompt_tokens", -1) or -1) == n_saved
            ):
                # The committed slot still holds this conversation's cache;
                # probes will prefix-hit without any restore I/O.
                outcome = "hot"
                return probe_slot
            target_slot = -1
            if committed is not None and not committed.get("is_processing"):
                target_slot = int(probe_slot)
            else:
                for item in slots:
                    if not item.get("is_processing"):
                        target_slot = int(item.get("id", -1))
                        break
            if target_slot < 0:
                # Every slot busy: a restore task would defer behind them.
                # Plain probes queue the same way, so just fall through.
                outcome = "busy"
                return probe_slot
            restore_started = time.monotonic()
            raw = post_json(
                f"{backend_url}/slots/{target_slot}?action=restore",
                {"filename": state_path.name},
                timeout=slot_state_restore_timeout_s,
            )
            restore_ms = int((time.monotonic() - restore_started) * 1000)
            n_restored = int(raw.get("n_restored", 0) or 0)
            if n_restored <= 0:
                outcome = "error"
                return probe_slot
            outcome = "hit"
            _journal_capture_event(
                f"slot-state restore request={request_id[:8]} "
                f"slot={target_slot} tokens={n_restored} ms={restore_ms}"
            )
            return target_slot
        except Exception as exc:
            outcome = "error"
            logger.info(
                "slot-state restore failed (fail-open to prompt cache): %s",
                exc,
            )
            return probe_slot
        finally:
            if timing is not None:
                timing["restore"] = outcome
                timing["restore_ms"] = restore_ms
    if proof_ops_per_request < 0:
        raise ValueError("proof_ops_per_request must be >= 0")
    if proof_trace_candidates_per_request < 0:
        raise ValueError("proof_trace_candidates_per_request must be >= 0")
    if proof_artifact_timeout <= 0:
        raise ValueError("proof_artifact_timeout must be > 0")
    proof_trace_candidates_per_request = max(
        proof_trace_candidates_per_request,
        proof_ops_per_request,
    )
    if require_proof and proof_sample_bps > 0 and proof_ops_per_request < 1:
        raise ValueError("proof_ops_per_request must be >= 1 when proof sampling is enabled")
    proof_trace_enable_path = Path(proof_trace_enable_file) if proof_trace_enable_file else None
    proof_trace_root = Path(proof_trace_dir) if proof_trace_dir else None
    proof_trace_janitor_stop = threading.Event()
    proof_trace_janitor_thread: threading.Thread | None = None
    proof_warmup_ms = 0.0
    if proof_trace_root is not None:
        proof_trace_root.mkdir(parents=True, exist_ok=True)

        def _proof_trace_janitor(
            root=proof_trace_root,
            stop=proof_trace_janitor_stop,
        ):
            while not stop.wait(60.0):
                try:
                    removed = prune_stale_proof_traces(root)
                except Exception as exc:
                    logger.debug("proof trace prune failed: %s", exc)
                    continue
                if removed:
                    logger.debug(
                        "pruned %d stale proof trace artifacts from %s",
                        removed,
                        root,
                    )

        proof_trace_janitor_thread = threading.Thread(
            target=_proof_trace_janitor,
            name="proof-trace-janitor",
            daemon=True,
        )
        if organic_capture_structurally_unwitnessed():
            logger.info(
                "organic capture skip active: parallel=%d slot-view v3 "
                "(serve-time shared joins are structurally witness-free; "
                "audits use exclusive probe windows; disable with "
                "VERATHOS_MESH_ORGANIC_CAPTURE_SKIP=0)",
                llama_n_parallel,
            )
        if proof_warmup:
            from verallm.mesh.ggml_proof import warm_ggml_proof_adapter

            proof_warmup_ms = warm_ggml_proof_adapter(
                proof_block_size=proof_block_size,
                spot_checks=proof_spot_checks,
            )
    proof_gguf_manifest = None
    proof_decode_projection_warmup_ms = 0.0
    proof_cache_warm_required = bool(
        server_role == "worker"
        and require_proof
        and proof_gguf_manifest_path
        and os.environ.get("VERATHOS_PROOF_CACHE_BACKGROUND_WARM", "1") != "0"
    )
    proof_cache_warm_state: dict[str, Any] = {
        "state": "warming" if proof_cache_warm_required else "not_required",
        "error": "",
    }
    if proof_gguf_manifest_path:
        from verallm.mesh.gguf_manifest import load_gguf_tensor_manifest

        proof_gguf_manifest = load_gguf_tensor_manifest(proof_gguf_manifest_path)
        if proof_decode_projection_warmup:
            from verallm.mesh.ggml_proof import warm_gguf_decode_projection_cache

            warmup = warm_gguf_decode_projection_cache(proof_gguf_manifest)
            proof_decode_projection_warmup_ms = float(warmup.get("elapsed_ms", 0.0))
        if os.environ.get("VERATHOS_PROOF_CACHE_BACKGROUND_WARM", "1") != "0":
            # A hard audit whose beacon draws a tensor with no persisted
            # weight-Merkle blob rebuilds the tree in-process: 20-30 s per
            # draw measured live, making audit latency a lottery until every
            # tensor has been drawn once. Warm the whole cache once in the
            # background instead; the pass is idempotent and skips blobs
            # that already exist, so restarts cost one directory scan.
            def _background_proof_cache_warm(manifest=proof_gguf_manifest):
                import shutil as _shutil

                from verallm.mesh.gguf_manifest import (
                    _should_persist_i8,
                    prewarm_proof_weight_cache_to_convergence,
                    proof_weight_cache_dir,
                )

                try:
                    cache_root = proof_weight_cache_dir()
                    if cache_root is None:
                        raise RuntimeError("proof-weight cache directory is unavailable")
                    # The warm's dequant transients must never stack on top
                    # of the backend's model load inside one memory budget:
                    # on the 24GB-GPU miner box class (30-32GB RAM) the sum
                    # OOM-kills the worker mid-formation. Wait for the
                    # backend to finish loading first; the warm is
                    # idempotent. Proof workers now fail closed if the model
                    # never becomes ready; exposing a cold per-draw fallback
                    # is precisely the post-relaunch audit race this gate
                    # removes. Cache-disabled non-worker roles remain ungated.
                    backend_loaded = wait_for_backend_model_loaded(backend_url)
                    if proof_cache_warm_required and backend_url and not backend_loaded:
                        raise RuntimeError(
                            "backend model did not become ready before proof-cache warm"
                        )
                    # Worst-case ADDITIONAL bytes for what the active profile
                    # actually persists: i8 blobs plus tree nodes (~45% of i8
                    # at the canonical chunk size). Under the compact profile
                    # that is only the final projection, so the estimate stays
                    # honest instead of sizing for blobs that never land.
                    i8_total = sum(
                        int(record.get("proof_i8_nbytes", 0) or 0)
                        for record in manifest.get("tensors", [])
                        if isinstance(record, Mapping)
                        and _should_persist_i8(record)
                    )
                    needed = int(i8_total * 1.5)
                    free = _shutil.disk_usage(cache_root).free
                    if free < needed + (10 << 30):
                        raise RuntimeError(
                            "insufficient disk for proof-cache warm: "
                            f"{free / 1e9:.0f} GB free < "
                            f"{needed / 1e9:.0f} GB worst-case + 10 GB headroom"
                        )
                    stats = prewarm_proof_weight_cache_to_convergence(manifest)
                    proof_cache_warm_state.update(
                        state="ready",
                        error="",
                        stats=stats,
                    )
                    if stats.get("cached") or stats.get("merkle"):
                        logger.info(
                            "proof weight cache warmed to convergence: %s", stats
                        )
                except Exception as exc:
                    proof_cache_warm_state.update(
                        state="failed",
                        error=str(exc)[:500],
                    )
                    if proof_cache_warm_required:
                        logger.error(
                            "proof cache warm failed; worker will remain "
                            "non-serving",
                            exc_info=True,
                        )
                    else:
                        logger.warning(
                            "background proof cache warm failed; hard audits "
                            "fall back to per-draw builds",
                            exc_info=True,
                        )

            threading.Thread(
                target=_background_proof_cache_warm,
                name="proof-cache-warm",
                daemon=True,
            ).start()
    prompt_binding_cache: dict[str, tuple[list[int], str, str]] = {}
    prompt_binding_cache_lock = threading.Lock()
    slot_view_template_cache: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = {}
    slot_view_template_cache_lock = threading.Lock()

    def current_mesh_spec() -> MeshSpec | None:
        if mesh_spec_loader is not None:
            return mesh_spec_loader()
        return mesh_spec

    def current_verification_snapshot(
        pinned_spec: MeshSpec | None = None,
    ):
        """Load and bind the signed public snapshot to this exact runtime."""

        if verification_snapshot_loader is None:
            raise RuntimeError("verification snapshot unavailable")
        from verallm.mesh.verification_snapshot import (
            MeshVerificationSnapshot,
            verify_mesh_verification_snapshot_signature,
        )

        loaded = verification_snapshot_loader()
        snapshot = (
            loaded
            if isinstance(loaded, MeshVerificationSnapshot)
            else MeshVerificationSnapshot.from_dict(loaded)
        )
        spec = pinned_spec if pinned_spec is not None else current_mesh_spec()
        if spec is None:
            raise RuntimeError("mesh spec is unavailable")
        snapshot.validate(require_signature=True)
        if snapshot.mesh_id != spec.mesh_id:
            raise RuntimeError("snapshot mesh_id does not match runtime mesh")
        if snapshot.coordinator.coordinator_uid != spec.coordinator_uid:
            raise RuntimeError("snapshot coordinator uid does not match runtime mesh")
        if snapshot.coordinator.coordinator_hotkey != spec.coordinator_hotkey:
            raise RuntimeError("snapshot coordinator hotkey does not match runtime mesh")
        if snapshot.coordinator.coordinator_evm_address != evm_address.lower():
            raise RuntimeError(
                "snapshot coordinator EVM address does not match serving identity"
            )
        expected_model = {
            "model_id": spec.model_id,
            "model_package_hash": spec.model_package_hash,
            "model_tensor_manifest_root": spec.model_tensor_manifest_root,
            "tokenizer_hash": spec.tokenizer_hash,
            "total_layers": spec.total_layers,
            "max_context_len": spec.max_context_len,
            "quantization_scheme": spec.quantization_scheme,
            "activation_dtype": spec.activation_dtype,
        }
        if snapshot.model.to_dict() != expected_model:
            raise RuntimeError("snapshot model anchors do not match runtime mesh")
        runtime_stage_ranges = sorted(
            (int(member.layers.start), int(member.layers.end))
            for member in spec.members
            if int(member.layers.end) > int(member.layers.start)
        )
        snapshot_stage_ranges = sorted(
            (int(stage.layer_start), int(stage.layer_end))
            for stage in snapshot.stages
        )
        if snapshot_stage_ranges != runtime_stage_ranges:
            raise RuntimeError("snapshot stage coverage does not match runtime mesh")
        snapshot_by_range = {
            (int(stage.layer_start), int(stage.layer_end)): stage
            for stage in snapshot.stages
        }
        for member in spec.members:
            if int(member.layers.end) <= int(member.layers.start):
                continue
            stage = snapshot_by_range[
                (int(member.layers.start), int(member.layers.end))
            ]
            expected_proof_key = str(member.proof_key or member.hotkey).strip()
            if stage.proof_key != expected_proof_key:
                raise RuntimeError(
                    "snapshot stage proof key does not match runtime mesh"
                )
        policy = snapshot.policy
        if policy.base_proof_sample_bps != int(proof_sample_bps):
            raise RuntimeError("snapshot base proof policy does not match runtime")
        # `decode_sample_bps` raises unless the organic and canary rates agree,
        # which is what keeps the wire value from identifying a canary.  The
        # absolute rate an active mesh must run is enforced validator-side, in
        # `verify_mesh_inference_artifact` and in the canary client, so it is
        # not repeated here.
        if policy.decode_sample_bps != int(decode_audit_bps):
            raise RuntimeError("snapshot decode policy does not match runtime")
        if policy.proof_ops_per_request != int(proof_ops_per_request):
            raise RuntimeError("snapshot proof op policy does not match runtime")
        if policy.proof_trace_candidates_per_request != int(
            proof_trace_candidates_per_request
        ):
            raise RuntimeError(
                "snapshot proof trace candidate policy does not match runtime"
            )
        if int(decode_audit_top_k) != VERATHOS_GGUF_DECODE_AUDIT_TOP_K:
            raise RuntimeError("snapshot runtime decode audit top-k must be 8")
        if policy.deferred_proof_enabled != bool(defer_proof):
            raise RuntimeError("snapshot deferred policy does not match runtime")
        if spec.proof_trace_manifest_format and (
            policy.trace_manifest_format != spec.proof_trace_manifest_format
        ):
            raise RuntimeError(
                "snapshot trace manifest format does not match runtime mesh"
            )
        if not verify_mesh_verification_snapshot_signature(
            snapshot,
            expected_hotkey=spec.coordinator_hotkey,
            expected_epoch=snapshot.epoch,
            expected_mesh_id=spec.mesh_id,
            now_unix=int(time.time()),
        ):
            raise RuntimeError("verification snapshot signature is invalid")
        return snapshot

    # Capture window manager. Organic requests share one capture window
    # (single token, so backend counters and manifest files stay stable while
    # requests overlap under --parallel N). Audit replays take an exclusive
    # window: they wait for in-flight organic captures to drain so selected
    # witness dumps can never contain another request's activations, and so
    # backend-local graph ordinals restart at 1 for deterministic targeting.
    capture_cond = threading.Condition()
    capture_state: dict[str, Any] = {
        "shared": set(),  # request ids holding the shared window
        "shared_armed_ns": {},  # request id -> arm timestamp (staleness expiry)
        "exclusive": "",  # request id holding the exclusive window
        "writers_waiting": 0,
        "token": "",  # digits window token currently written to the enable file
        # holder -> [armed_ns, disarmed_ns]. The payload build runs AFTER the
        # exclusive window disarms, and under concurrent audits the NEXT
        # audit's arm overwrites any single global timestamp before the
        # previous audit's payload build reads it, so the build then scans
        # the WRONG window's dumps (observed: an audit whose window
        # dumped 3 instances reported "window captured 1 instances over
        # graph ords [1]" because it read the successor's window). Window
        # bounds must therefore be per holder.
        "exclusive_window_ns_by_holder": {},
    }
    _EXCLUSIVE_WINDOW_HISTORY_LIMIT = 128

    def _record_exclusive_window_armed_locked(holder: str) -> int:
        now_ns = time.time_ns()
        windows = capture_state["exclusive_window_ns_by_holder"]
        windows[str(holder)] = [now_ns, 0]
        while len(windows) > _EXCLUSIVE_WINDOW_HISTORY_LIMIT:
            windows.pop(next(iter(windows)))
        return now_ns

    def _record_exclusive_window_disarmed_locked(holder: str) -> None:
        window = capture_state["exclusive_window_ns_by_holder"].get(str(holder))
        if window is not None and not window[1]:
            window[1] = time.time_ns()
    # A holder whose disarm never arrives (coordinator crash, dropped POST,
    # network partition) must not strand the capture window forever: a stale
    # shared holder blocks fresh window tokens (so captures silently stop) and
    # a stale exclusive claim deadlocks audits. Expire both after this budget.
    capture_stale_ns = int(
        float(os.environ.get("VERATHOS_MESH_CAPTURE_STALE_S", "120")) * 1e9
    )

    def _purge_stale_capture_holders_locked() -> None:
        now_ns = time.time_ns()
        armed = capture_state["shared_armed_ns"]
        stale = [
            holder
            for holder in capture_state["shared"]
            if now_ns - int(armed.get(holder, 0) or 0) > capture_stale_ns
        ]
        for holder in stale:
            capture_state["shared"].discard(holder)
            armed.pop(holder, None)
        if capture_state["exclusive"]:
            exclusive_ns = int(capture_state.get("exclusive_armed_ns", 0) or 0)
            if now_ns - exclusive_ns > capture_stale_ns:
                capture_state["exclusive"] = ""
        if (
            (stale or not capture_state["shared"])
            and not capture_state["shared"]
            and not capture_state["exclusive"]
            and capture_state["token"]
        ):
            _clear_capture_token()

    def _journal_capture_event(event: str) -> None:
        # Window-transition journal: one line per token write/clear, so a
        # zero-capture audit window can be diagnosed post-hoc against the
        # C-side dumps (which only exist when the token matched). Writes
        # happen only on capture window transitions, never per token.
        if proof_trace_enable_path is None:
            return
        try:
            with open(
                proof_trace_enable_path.parent / "token-journal.log",
                "a",
                encoding="utf-8",
            ) as journal:
                journal.write(f"{time.time_ns()} {event}\n")
        except OSError:
            pass

    def _write_capture_token(token: str) -> None:
        base_token = token.split("|", 1)[0]
        if proof_trace_enable_path is not None:
            proof_trace_enable_path.parent.mkdir(parents=True, exist_ok=True)
            proof_trace_enable_path.write_text(token, encoding="utf-8")
            _journal_capture_event(f"write {token}")
        # An orchestration-only all-RPC coordinator has no local trace file,
        # but it still brokers one capture window across all remote stages.
        capture_state["token"] = base_token

    def _clear_capture_token() -> None:
        capture_state["token"] = ""
        if proof_trace_enable_path is None:
            return
        try:
            proof_trace_enable_path.unlink()
            _journal_capture_event("clear")
        except FileNotFoundError:
            _journal_capture_event("clear-missing")

    def _selected_capture_token(
        window_token: str,
        selected_manifest_indexes: list[int] | None,
        selected_ops: list[str] | None,
        selected_anchor_rows: list[int] | None = None,
    ) -> str:
        token = str(window_token or time.time_ns())
        selected = [int(item) for item in (selected_manifest_indexes or [])]
        ops = [str(item) for item in (selected_ops or [])]
        anchor_rows = [int(item) for item in (selected_anchor_rows or [])]
        if selected:
            token += "|selected=" + ",".join(str(item) for item in selected)
        if ops:
            token += "|selected_v3=" + ",".join(ops)
        if anchor_rows:
            # The runtime dumps exactly these per-stage stream rows to
            # anchor-*.vrows during the replay; the bounded anchored hard
            # audit opens them against the frozen commitments.
            token += "|anchor_rows=" + ",".join(
                str(item) for item in anchor_rows
            )
        return token

    def acquire_shared_capture(
        request_id: str,
        *,
        window_token: str = "",
        timeout: float | None = None,
        on_wait_tick: Callable[[], None] | None = None,
    ) -> str:
        """Join the shared capture window; returns the active window token.

        The window token doubles as the trace file suffix on every stage, so
        the coordinator mints it once and passes it to remote workers. It
        only changes when the window fully drains (no straddling graphs).

        ``on_wait_tick`` runs roughly every CAPTURE_WAIT_TICK_S while the
        join is blocked behind an exclusive replay, OUTSIDE the condition
        lock (it does socket I/O: SSE keepalives). If it raises - the
        client hung up mid-wait - the join aborts immediately instead of
        holding its admission reservation for the rest of the budget."""

        # Resolved at call time (module attribute, not a bound default) so
        # tests and operators can tune the budget without rebuilding the
        # server closure.
        wait_budget = float(SHARED_CAPTURE_WAIT_S if timeout is None else timeout)
        deadline = time.monotonic() + wait_budget
        last_tick = time.monotonic()
        while True:
            with capture_cond:
                _purge_stale_capture_holders_locked()
                if (
                    not capture_state["exclusive"]
                    and capture_state["writers_waiting"] <= 0
                ):
                    first = not capture_state["shared"]
                    capture_state["shared"].add(str(request_id))
                    capture_state["shared_armed_ns"][str(request_id)] = (
                        time.time_ns()
                    )
                    if first:
                        _write_capture_token(str(window_token or time.time_ns()))
                    return str(capture_state["token"])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _CaptureWindowBusy(
                        "timed out waiting for exclusive trace capture to finish"
                    )
                slice_s = (
                    min(remaining, float(CAPTURE_WAIT_TICK_S))
                    if on_wait_tick is not None
                    else remaining
                )
                capture_cond.wait(timeout=slice_s)
            if on_wait_tick is not None:
                now = time.monotonic()
                if now - last_tick >= float(CAPTURE_WAIT_TICK_S):
                    on_wait_tick()
                    last_tick = now

    def release_shared_capture(request_id: str) -> None:
        with capture_cond:
            capture_state["shared"].discard(str(request_id))
            capture_state["shared_armed_ns"].pop(str(request_id), None)
            if not capture_state["shared"] and not capture_state["exclusive"]:
                _clear_capture_token()
            capture_cond.notify_all()

    # Capture-skipping organic serves are invisible to the exclusive window's
    # shared-holder drain (that is the point - no drain latency), but a hard
    # audit whose probes keep colliding with free-running organic decode
    # still needs ONE bounded way to run solo. This registry counts those
    # serves; the strict-quiesce event gates NEW ones behind the same
    # wait/keepalive/busy bytes as a shared-capture join (canary-oracle
    # rule: the refusal must be byte-identical to a saturation refusal).
    organic_inflight_cond = threading.Condition()
    organic_inflight_state = {"count": 0}
    strict_quiesce_event = threading.Event()
    strict_quiesce_wait_s = float(
        os.environ.get("VERATHOS_MESH_STRICT_QUIESCE_WAIT_S", "180") or 180.0
    )

    @contextmanager
    def _organic_inflight_tracked():
        with organic_inflight_cond:
            organic_inflight_state["count"] += 1
        try:
            yield
        finally:
            with organic_inflight_cond:
                organic_inflight_state["count"] = max(
                    0, organic_inflight_state["count"] - 1
                )
                organic_inflight_cond.notify_all()

    def _wait_out_strict_quiesce(
        timeout: float | None,
        on_wait_tick: Callable[[], None] | None,
    ) -> None:
        """Hold a new capture-skipping serve while a strict quiesce runs.

        Same wait shape as acquire_shared_capture: keepalive ticks outside
        any lock, and on budget exhaustion the SAME retryable busy refusal
        an admission-ledger saturation produces.
        """

        if not strict_quiesce_event.is_set():
            return
        wait_budget = float(
            SHARED_CAPTURE_WAIT_S if timeout is None else timeout
        )
        deadline = time.monotonic() + wait_budget
        last_tick = time.monotonic()
        while strict_quiesce_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _CaptureWindowBusy(
                    "timed out waiting for exclusive trace capture to finish"
                )
            # Event.wait waits for SET; the release condition here is
            # CLEARED, so poll in bounded slices (the quiesce is rare and
            # minutes-bounded; 250 ms of release latency is immaterial)
            # while keepalive ticks keep the client's read loop fed.
            time.sleep(min(remaining, 0.25))
            if on_wait_tick is not None:
                now = time.monotonic()
                if now - last_tick >= float(CAPTURE_WAIT_TICK_S):
                    on_wait_tick()
                    last_tick = now

    @contextmanager
    def exclusive_capture(
        request_id: str,
        *,
        window_token: str = "",
        selected_manifest_indexes: list[int] | None = None,
        selected_ops: list[str] | None = None,
        selected_anchor_rows: list[int] | None = None,
        require_local_trace: bool = True,
        timeout: float = 120.0,
        timing: dict[str, Any] | None = None,
    ):
        if require_local_trace and proof_trace_enable_path is None:
            raise RuntimeError("trace capture is not configured")
        deadline = time.monotonic() + timeout
        drain_started = time.monotonic()
        with capture_cond:
            capture_state["writers_waiting"] += 1
            try:
                _purge_stale_capture_holders_locked()
                while capture_state["shared"] or capture_state["exclusive"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "timed out waiting for organic trace captures to drain"
                        )
                    capture_cond.wait(timeout=remaining)
                    _purge_stale_capture_holders_locked()
                capture_state["exclusive"] = str(request_id)
            finally:
                capture_state["writers_waiting"] -= 1
            if timing is not None:
                timing["drain_ms"] = int(
                    (time.monotonic() - drain_started) * 1000
                )
            _write_capture_token(
                _selected_capture_token(
                    window_token,
                    selected_manifest_indexes,
                    selected_ops,
                    selected_anchor_rows,
                )
            )
            capture_state["exclusive_armed_ns"] = (
                _record_exclusive_window_armed_locked(request_id)
            )
            if timing is not None:
                timing["armed_unix_ns"] = int(
                    capture_state["exclusive_armed_ns"]
                )
            token = str(capture_state["token"])
        try:
            yield token
        finally:
            with capture_cond:
                capture_state["exclusive"] = ""
                _record_exclusive_window_disarmed_locked(request_id)
                _clear_capture_token()
                capture_cond.notify_all()

    def set_local_trace_capture(
        enabled: bool,
        *,
        request_id: str = "",
        mode: str = "",
        window_token: str = "",
        selected_manifest_indexes: list[int] | None = None,
        selected_ops: list[str] | None = None,
        selected_anchor_rows: list[int] | None = None,
    ) -> None:
        """Legacy-compatible local capture toggle (HTTP endpoint surface).

        Selected captures map to a non-blocking exclusive claim; plain
        captures map to shared-window membership keyed by request id.
        """

        if proof_trace_enable_path is None:
            return
        selected = [int(item) for item in (selected_manifest_indexes or [])]
        ops = [str(item) for item in (selected_ops or [])]
        effective_mode = mode or ("exclusive" if (selected or ops) else "shared")
        holder = str(request_id or "legacy")
        _journal_capture_event(
            f"set enabled={bool(enabled)} mode={effective_mode} "
            f"holder={holder} ops={len(ops)} indexes={len(selected)}"
        )
        if enabled:
            if effective_mode == "exclusive":
                with capture_cond:
                    _purge_stale_capture_holders_locked()
                    if capture_state["shared"] or (
                        capture_state["exclusive"]
                        and capture_state["exclusive"] != holder
                    ):
                        raise RuntimeError(
                            "trace capture is busy; retry exclusive capture later"
                        )
                    capture_state["exclusive"] = holder
                    _write_capture_token(
                        _selected_capture_token(
                            window_token,
                            selected,
                            ops,
                            [int(item) for item in (selected_anchor_rows or [])],
                        )
                    )
                    capture_state["exclusive_armed_ns"] = (
                        _record_exclusive_window_armed_locked(holder)
                    )
                return
            acquire_shared_capture(holder, window_token=window_token)
            return
        with capture_cond:
            if capture_state["exclusive"] in ("", holder):
                capture_state["exclusive"] = ""
            _record_exclusive_window_disarmed_locked(holder)
            capture_state["shared"].discard(holder)
            capture_state["shared_armed_ns"].pop(holder, None)
            if not capture_state["shared"] and not capture_state["exclusive"]:
                _clear_capture_token()
            capture_cond.notify_all()

    def set_remote_trace_capture(
        member_endpoint: str,
        *,
        spec: MeshSpec,
        request_id: str,
        enabled: bool,
        mode: str = "",
        window_token: str = "",
        selected_manifest_indexes: list[int] | None = None,
        selected_ops: list[str] | None = None,
        selected_anchor_rows: list[int] | None = None,
        anchor_beacon: str = "",
        retries: int = 0,
        retry_delay: float = 0.5,
        timeout: float = 3.0,
    ) -> None:
        body = {
            "mesh_spec_hash": spec.spec_hash_hex(),
            "request_id": request_id,
            "enabled": bool(enabled),
            "selected_manifest_indexes": [
                int(item) for item in (selected_manifest_indexes or [])
            ],
        }
        if mode:
            body["mode"] = str(mode)
        if window_token:
            body["window_token"] = str(window_token)
        if selected_ops:
            body["selected_ops"] = [str(item) for item in selected_ops]
        if selected_anchor_rows:
            body["selected_anchor_rows"] = [
                int(item) for item in selected_anchor_rows
            ]
        if anchor_beacon:
            body["anchor_beacon"] = str(anchor_beacon)
        attempt = 0
        while True:
            try:
                post_json(
                    member_endpoint.rstrip("/") + "/v1/mesh/trace-capture",
                    body,
                    timeout=float(timeout),
                    internal_auth_secret=internal_auth_secret,
                )
                return
            except Exception:
                attempt += 1
                if attempt > max(0, int(retries)):
                    raise
                time.sleep(retry_delay)

    def dedupe_selected_ops(ops: list[str]) -> list[str]:
        """Preserve graph-priority order while removing duplicate native filters."""

        selected: list[str] = []
        seen: set[str] = set()
        for op in ops:
            value = str(op)
            if value in seen:
                continue
            seen.add(value)
            selected.append(value)
        return selected

    def bind_decode_audit_completion_tokens(
        receipt_context: dict[str, Any],
        token_ids: list[int],
        token_source: str,
    ) -> None:
        if not receipt_context.get("decode_audit_required"):
            return
        if not token_source:
            raise RuntimeError("decode audit replay missing completion token ids")
        ids = [int(item) for item in token_ids]
        expected_count = int(receipt_context.get("completion_token_count", 0) or 0)
        if expected_count != len(ids):
            raise RuntimeError("decode audit replay completion token count mismatch")
        receipt_context["decode_audit_completion_token_ids"] = ids
        receipt_context["decode_audit_completion_token_ids_hash"] = (
            completion_token_ids_hash(ids)
        )
        receipt_context["decode_audit_completion_token_count"] = len(ids)
        receipt_context["decode_audit_completion_token_source"] = str(token_source)

    def proof_policy_context(
        openai_request: dict[str, Any] | None = None,
        *,
        requested_require_proof: bool = False,
        pinned_verification_snapshot: Any | None = None,
        validator_principal: str = "",
        require_postcommit: bool = False,
        validator_authenticated: bool = False,
    ) -> dict[str, Any]:
        verathos = (
            openai_request.get("verathos", {})
            if isinstance(openai_request, dict)
            else {}
        )
        if not isinstance(verathos, dict):
            verathos = {}
        challenge_commitment = str(
            verathos.get("challenge_nonce_commitment", "") or ""
        )
        if challenge_commitment:
            _require_sha256_text(
                "challenge_nonce_commitment",
                challenge_commitment,
            )
        if require_postcommit:
            if not challenge_commitment:
                raise RuntimeError(
                    "authenticated validator request requires a hidden "
                    "postcommit challenge commitment"
                )
            if validator_nonce_from_request(openai_request or {}):
                raise RuntimeError(
                    "authenticated validator request must not reveal its "
                    "challenge nonce before the origin receipt"
                )
            if not str(validator_principal or ""):
                raise RuntimeError(
                    "authenticated validator principal is unavailable"
                )
        snapshot_binding: dict[str, Any] = {}
        if verification_snapshot_loader is not None and server_role != "worker":
            snapshot = (
                pinned_verification_snapshot
                if pinned_verification_snapshot is not None
                else current_verification_snapshot()
            )
            expected_snapshot_hash = snapshot.snapshot_hash_hex()
            requested_snapshot_hash = str(
                verathos.get("verification_snapshot_hash", "") or ""
            )
            if requested_snapshot_hash != expected_snapshot_hash:
                raise _VerificationSnapshotMismatch(
                    "request verification snapshot hash does not match coordinator"
                )
            if (
                require_postcommit
                and str(snapshot.policy.challenge_scheme)
                != VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
            ):
                raise RuntimeError(
                    "verification snapshot does not approve the postcommit "
                    "challenge scheme"
                )
            snapshot_binding = {
                "verification_snapshot_hash": expected_snapshot_hash,
                "verification_snapshot_generation": int(snapshot.generation),
                "verification_snapshot_epoch": int(snapshot.epoch),
                "model_index": int(snapshot.coordinator.model_index),
                "model_total_layers": int(snapshot.model.total_layers),
                "proof_policy_profile": snapshot.policy.profile,
                "proof_receipt_format": "opaque_stage_v2",
                "proof_trace_manifest_format": (
                    snapshot.policy.trace_manifest_format
                ),
                # The signed hard-audit rate for the postcommit tier draw.
                # It rides the origin receipt so both sides derive the same
                # hard/light decision from the revealed nonce.
                "proof_postcommit_hard_bps": int(
                    getattr(snapshot.policy, "postcommit_hard_audit_bps", 10_000)
                ),
            }
        requested_proof_bps = (
            normalize_proof_sample_bps(int(verathos["proof_sample_bps"]))
            if "proof_sample_bps" in verathos
            else None
        )
        requested_decode_bps = (
            normalize_proof_sample_bps(int(verathos["decode_audit_bps"]))
            if "decode_audit_bps" in verathos
            else None
        )
        requested_ops = (
            max(1, int(verathos["proof_ops_per_request"]))
            if "proof_ops_per_request" in verathos
            else None
        )
        requested_candidates = (
            max(1, int(verathos["proof_trace_candidates_per_request"]))
            if "proof_trace_candidates_per_request" in verathos
            else None
        )
        requested_decode_top_k = (
            max(1, int(verathos["decode_audit_top_k"]))
            if "decode_audit_top_k" in verathos
            else None
        )
        requested_defer_proof = bool(
            verathos.get("defer_proof", False)
            or verathos.get("proof_deferred", False)
        )
        if requested_defer_proof and not defer_proof:
            raise RuntimeError(
                "deferred proof is disabled; the coordinator must opt in explicitly"
            )
        if snapshot_binding:
            if requested_ops is not None and requested_ops != int(
                snapshot.policy.proof_ops_per_request
            ):
                raise RuntimeError(
                    "request proof op count is not snapshot-approved"
                )
            if requested_candidates is not None and requested_candidates != int(
                snapshot.policy.proof_trace_candidates_per_request
            ):
                raise RuntimeError(
                    "request proof trace candidate count is not snapshot-approved"
                )
        effective_defer_proof = bool(defer_proof)
        effective_proof_bps = max(
            int(proof_sample_bps),
            int(requested_proof_bps or 0),
        )
        effective_decode_bps = max(
            int(decode_audit_bps),
            int(requested_decode_bps or 0),
        )
        effective_ops_per_request = max(
            int(proof_ops_per_request),
            int(requested_ops or 0),
        )
        effective_candidates_per_request = max(
            int(proof_trace_candidates_per_request),
            int(requested_candidates or 0),
            effective_ops_per_request,
        )
        effective_decode_top_k = max(
            int(decode_audit_top_k),
            int(requested_decode_top_k or 0),
        )
        requested_any_proof = any(
            value is not None
            for value in (
                requested_proof_bps,
                requested_decode_bps,
                requested_ops,
                requested_candidates,
            )
        )
        configured = bool(require_proof or requested_require_proof or requested_any_proof)
        audit_configured = bool(configured and effective_decode_bps > 0)
        effective_bps = max(
            int(effective_proof_bps),
            int(effective_decode_bps if audit_configured else 0),
        )
        postcommit_enabled = bool(
            require_postcommit and configured and effective_bps > 0
        )
        if require_postcommit and not postcommit_enabled:
            raise RuntimeError(
                "authenticated validator inference requires configured "
                "postcommit proof capture"
            )
        if not configured or effective_bps <= 0:
            capture_required = False
            challenge_kind = "disabled"
        elif postcommit_enabled:
            capture_required = True
            challenge_kind = VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
        elif effective_defer_proof:
            capture_required = True
            challenge_kind = "deferred_future_randomness_v1"
        elif effective_bps >= PROOF_SAMPLE_BPS_DENOMINATOR:
            # inline_every_request_v1 derives its beacon as
            # SHA256(tag || gate_hash), which the coordinator can compute
            # itself and grind by regenerating the response until it likes the
            # draw.  It survives for the loopback operator self-test, where
            # grinding buys an operator nothing but a self-satisfying test.
            # It must never serve an authenticated validator: reaching this
            # branch for one means the postcommit requirement was lost
            # upstream, which is the auth-regression case, so refuse rather
            # than silently downgrade to a grindable challenge.
            if bool(validator_authenticated):
                raise RuntimeError(
                    "snapshot-bound mesh inference must not fall back to the "
                    "grindable inline challenge"
                )
            capture_required = True
            challenge_kind = "inline_every_request_v1"
        else:
            capture_required = True
            nonce_raw = validator_nonce_from_request(openai_request or {})
            challenge_kind = "deferred_future_randomness_v1"
            if nonce_raw:
                try:
                    normalize_validator_nonce(nonce_raw)
                    challenge_kind = "fiat_shamir_inline_v1"
                except ValueError:
                    challenge_kind = "fiat_shamir_inline_v1"
        # Slot-view receipts need committed prompt token counts for the
        # deterministic solo-replay arithmetic, so request token metadata.
        slot_view_capture = bool(slot_view_required and capture_required)
        metadata_required = bool(audit_configured or slot_view_capture)
        # Under continuous batching a co-batched run and the solo audit replay
        # can segment a sampled completion into different token ids (same
        # text), which would desync the decode-step layout. Force the verified
        # greedy sampler so the committed completion replays token-for-token.
        verified_sampler_required = bool(audit_configured or slot_view_capture)
        deferred_audit_bps = (
            int(effective_bps)
            if capture_required
            and 0 < int(effective_bps)
            and (
                effective_defer_proof
                or (
                    int(effective_bps) < PROOF_SAMPLE_BPS_DENOMINATOR
                    and not postcommit_enabled
                )
            )
            else 0
        )
        deferred_round = ""
        proof_tier_request = ""
        if isinstance(verathos, dict):
            deferred_round = str(verathos.get("deferred_randomness_round", "") or "")
            proof_tier_request = str(verathos.get("proof_tier", "") or "")
        from verallm.mesh.verification_snapshot import (
            MESH_POSTCOMMIT_HARD_AUDIT_BPS,
        )

        return {
            # Explicit on EVERY receipt: a receipt without a hard-audit
            # rate makes the postcommit decision fall back to the
            # pre-tiering "every sampled audit is hard" behavior, which
            # put a 10% inline hard tax on dev-pool organic chats. The
            # producer policy is 0 - organic traffic is never
            # hard-audited; a bound snapshot's signed rate (spread below)
            # takes precedence when present.
            "proof_postcommit_hard_bps": MESH_POSTCOMMIT_HARD_AUDIT_BPS,
            **snapshot_binding,
            "proof_policy_version": 2 if postcommit_enabled else 1,
            "proof_configured_required": configured,
            "proof_capture_required": capture_required,
            "verified_sampler_required": bool(verified_sampler_required),
            "proof_metadata_required": metadata_required,
            "proof_required": False,
            "proof_sample_bps": int(effective_proof_bps),
            "proof_sample_denominator": PROOF_SAMPLE_BPS_DENOMINATOR,
            "proof_ops_per_request": int(effective_ops_per_request),
            "proof_trace_candidates_per_request": int(effective_candidates_per_request),
            "proof_challenge_kind": challenge_kind,
            # Upgrade-only tier request: callers (the pool's validator-mode
            # self-test, a probing operator) can force the hard relation on
            # the organic inline lane. It can never downgrade a validator
            # kind to light; the tier router rejects light for those anyway.
            "proof_tier_hard_requested": proof_tier_request == "hard",
            "proof_challenge_nonce_commitment": (
                challenge_commitment if postcommit_enabled else ""
            ),
            "proof_validator_hotkey": (
                str(validator_principal) if postcommit_enabled else ""
            ),
            "proof_postcommit": bool(postcommit_enabled),
            "proof_postcommit_origin_receipt_hash": "",
            "proof_postcommit_challenge_nonce": "",
            "proof_postcommit_finalized": False,
            "proof_deferred": bool(deferred_audit_bps > 0),
            "proof_deferred_obligation": bool(deferred_audit_bps > 0),
            "proof_deferred_required": bool(
                deferred_audit_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
            ),
            "proof_deferred_mode": (
                "future_randomness_v1" if deferred_audit_bps > 0 else ""
            ),
            "proof_deferred_audit_bps": int(deferred_audit_bps),
            "proof_deferred_sample_commitment_hash": "",
            "proof_deferred_commitment_hash": "",
            "proof_deferred_randomness_round": deferred_round,
            "proof_deferred_randomness": "",
            "proof_deferred_beacon": "",
            "proof_deferred_sample_value": -1,
            "proof_deferred_sampled": False,
            "decode_audit_configured": audit_configured,
            "decode_audit_mode": VERATHOS_GGUF_DECODE_AUDIT_MODE if audit_configured else "",
            "decode_audit_bps": int(effective_decode_bps if audit_configured else 0),
            "decode_audit_top_k": int(effective_decode_top_k),
            "decode_audit_required": False,
            "decode_audit_sampled": False,
            "decode_audit_sample_value": -1,
            "decode_audit_positions": [],
            "decode_audit_commitment_hash": "",
            "decode_audit_receipt_root": "",
            "decode_audit_verified": False,
        }

    # Sticky remote-capture hold. Arming every serve's shared window on
    # every remote stage cost four HTTP round-trips per request through the
    # workers' control planes; under concurrency the windows overlap almost
    # always, so the transitions 0->1 and ->0 are the only ones that need
    # the wire. The stable holder id keeps the remote refcount at one, and
    # a periodic re-arm refreshes the remote stale timer
    # (VERATHOS_MESH_CAPTURE_STALE_S, default 120 s) for long windows.
    _REMOTE_HOLD_ID = "coordinator-shared-hold"
    _REMOTE_HOLD_REFRESH_S = 30.0
    _remote_hold_lock = threading.Lock()
    _remote_hold_state = {"count": 0, "token": "", "armed_monotonic": 0.0}

    def _arm_remote_members(
        *,
        spec: MeshSpec,
        remote_members: list[MeshMember],
        window_token: str,
        enabled: bool,
    ) -> None:
        with ThreadPoolExecutor(max_workers=min(8, len(remote_members))) as pool:
            futures = {
                pool.submit(
                    set_remote_trace_capture,
                    same_host_loopback(member.endpoint),
                    spec=spec,
                    request_id=_REMOTE_HOLD_ID,
                    enabled=enabled,
                    mode="shared",
                    window_token=window_token if enabled else "",
                    # Joining a shared window BLOCKS server-side while an
                    # exclusive replay drains (acquire_shared_capture waits up
                    # to SHARED_CAPTURE_WAIT_S). A 3 s client deadline made
                    # every organic request that overlapped a hard proof fail
                    # with "failed to arm trace capture: timed out" - measured
                    # as 5 of 1666 soak requests at 4-way concurrency. Wait
                    # out the window the server is willing to impose; a
                    # DISARM never blocks, so it keeps the short deadline.
                    timeout=(
                        SHARED_CAPTURE_ARM_TIMEOUT_S if enabled else 3.0
                    ),
                    retries=2 if enabled else 0,
                    retry_delay=1.0,
                ): member
                for member in remote_members
            }
            for future in as_completed(futures):
                member = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    if enabled:
                        raise RuntimeError(
                            "failed to arm trace capture on "
                            f"{member.endpoint}: {exc}"
                        ) from exc
                    # Best-effort disarm; the remote stale timer reclaims a
                    # missed release.

    def _hold_remote_shared_capture(
        *,
        spec: MeshSpec,
        remote_members: list[MeshMember],
        window_token: str,
    ) -> None:
        with _remote_hold_lock:
            now = time.monotonic()
            need_arm = (
                _remote_hold_state["count"] == 0
                or _remote_hold_state["token"] != window_token
                or now - _remote_hold_state["armed_monotonic"]
                > _REMOTE_HOLD_REFRESH_S
            )
            if need_arm:
                _arm_remote_members(
                    spec=spec,
                    remote_members=remote_members,
                    window_token=window_token,
                    enabled=True,
                )
                _remote_hold_state["token"] = window_token
                _remote_hold_state["armed_monotonic"] = now
            _remote_hold_state["count"] += 1

    @contextmanager
    def _suspended_remote_shared_hold(*, spec: MeshSpec | None):
        """Drop the sticky shared hold so an exclusive audit window can arm.

        The coordinator keeps a shared capture hold armed on every worker to
        avoid per-request arm/disarm round trips. A worker's exclusive arm
        requires an idle capture state, so with the hold in place the
        SELECTOR never lands and the worker keeps serving under a bare
        window token — which dumps nothing selectable and starves every
        audit (observed: "no single-row decode instance captured").
        Suspend the hold for the duration of the exclusive window, then
        restore it so steady-state serving keeps its round-trip savings.
        """

        remote_members = (
            [
                member
                for member in spec.members
                # rpc members compute in a separate process and need
                # remote arming; in local-stage mode the first member has NO
                # rpc endpoint but its layers run inside the coordinator's
                # llama-server, so it too must be armed over its
                # trace-capture channel (it never sees the request itself).
                if (
                    member.rpc_endpoint
                    or (
                        local_stage_capture
                        and member.layers.end > member.layers.start
                    )
                )
                and member.endpoint != capability.endpoint
            ]
            if spec is not None
            else []
        )
        suspended = False
        with _remote_hold_lock:
            if remote_members and _remote_hold_state["count"] > 0:
                _arm_remote_members(
                    spec=spec,
                    remote_members=remote_members,
                    window_token="",
                    enabled=False,
                )
                _remote_hold_state["token"] = ""
                _remote_hold_state["armed_monotonic"] = 0.0
                suspended = True
        try:
            yield
        finally:
            if suspended:
                with _remote_hold_lock:
                    if _remote_hold_state["count"] > 0:
                        token = str(time.time_ns())
                        try:
                            _arm_remote_members(
                                spec=spec,
                                remote_members=remote_members,
                                window_token=token,
                                enabled=True,
                            )
                            _remote_hold_state["token"] = token
                            _remote_hold_state["armed_monotonic"] = (
                                time.monotonic()
                            )
                        except Exception:
                            # Next request re-arms; a missed re-arm only
                            # costs one round trip, never correctness.
                            _remote_hold_state["token"] = ""
                            _remote_hold_state["armed_monotonic"] = 0.0

    def _release_remote_shared_capture(*, spec: MeshSpec | None) -> None:
        with _remote_hold_lock:
            _remote_hold_state["count"] = max(
                0, _remote_hold_state["count"] - 1
            )
            if _remote_hold_state["count"] > 0 or spec is None:
                return
            remote_members = [
                member
                for member in spec.members
                # rpc members compute in a separate process and need
                # remote arming; in local-stage mode the first member has NO
                # rpc endpoint but its layers run inside the coordinator's
                # llama-server, so it too must be armed over its
                # trace-capture channel (it never sees the request itself).
                if (
                    member.rpc_endpoint
                    or (
                        local_stage_capture
                        and member.layers.end > member.layers.start
                    )
                )
                and member.endpoint != capability.endpoint
            ]
            _remote_hold_state["token"] = ""
            _remote_hold_state["armed_monotonic"] = 0.0
            if remote_members:
                _arm_remote_members(
                    spec=spec,
                    remote_members=remote_members,
                    window_token="",
                    enabled=False,
                )

    @contextmanager
    def capture_trace_during_backend(
        *,
        spec: MeshSpec | None,
        request_id: str,
        proof_capture_required: bool,
        capture_wait_timeout: float | None = None,
        on_capture_wait_tick: Callable[[], None] | None = None,
    ):
        if not proof_capture_required:
            # Serves that hold no shared capture (capture-skipping organics
            # and capture-less policies) still honor a strict quiesce: an
            # escalated hard audit needs the decode stream solo once, and
            # these serves are exactly the ones its drain cannot see.
            _wait_out_strict_quiesce(capture_wait_timeout, on_capture_wait_tick)
            with _organic_inflight_tracked():
                yield ""
            return
        remote_members = (
            [
                member
                for member in spec.members
                # rpc members compute in a separate process and need
                # remote arming; in local-stage mode the first member has NO
                # rpc endpoint but its layers run inside the coordinator's
                # llama-server, so it too must be armed over its
                # trace-capture channel (it never sees the request itself).
                if (
                    member.rpc_endpoint
                    or (
                        local_stage_capture
                        and member.layers.end > member.layers.start
                    )
                )
                and member.endpoint != capability.endpoint
            ]
            if spec is not None
            else []
        )
        capture_window_acquired = False
        remote_hold_acquired = False
        window_token = ""
        if proof_trace_enable_path is not None or remote_members:
            window_token = acquire_shared_capture(
                request_id,
                timeout=capture_wait_timeout,
                on_wait_tick=on_capture_wait_tick,
            )
            capture_window_acquired = True
        try:
            if remote_members:
                _hold_remote_shared_capture(
                    spec=spec,
                    remote_members=remote_members,
                    window_token=window_token,
                )
                remote_hold_acquired = True
            yield window_token
        finally:
            if remote_hold_acquired:
                _release_remote_shared_capture(spec=spec)
            # Keep the coordinator's shared claim until every remote stage is
            # disarmed, so an exclusive replay cannot overtake partial cleanup.
            if capture_window_acquired:
                release_shared_capture(request_id)

    def forward_to_backend(
        openai_request: dict[str, Any],
        *,
        spec: MeshSpec | None,
        request_id: str,
        proof_capture_required: bool,
        capture_info: dict[str, Any] | None = None,
        skip_trace_capture: bool = False,
        capture_wait_timeout: float | None = None,
    ) -> dict[str, Any]:
        if not backend_url:
            raise RuntimeError("no local backend configured; pass --backend-url")
        if openai_request.get("stream"):
            raise RuntimeError("use forward_stream_to_backend for streaming requests")
        with capture_trace_during_backend(
            spec=spec,
            request_id=request_id,
            proof_capture_required=proof_capture_required and not skip_trace_capture,
            capture_wait_timeout=capture_wait_timeout,
        ) as window_token:
            if capture_info is not None:
                capture_info["window_token"] = str(window_token or "")
            return post_json(f"{backend_url}/v1/chat/completions", openai_request, timeout=BACKEND_FORWARD_TIMEOUT_S)

    def forward_stream_to_backend(
        openai_request: dict[str, Any],
        *,
        spec: MeshSpec | None,
        request_id: str,
        proof_capture_required: bool,
        emit_raw_sse,
        capture_info: dict[str, Any] | None = None,
        skip_trace_capture: bool = False,
        capture_wait_timeout: float | None = None,
    ) -> dict[str, Any]:
        if not backend_url:
            raise RuntimeError("streaming mesh inference requires a local backend")
        stream_request = dict(openai_request)
        stream_request["stream"] = True
        req = Request(
            f"{backend_url}/v1/chat/completions",
            data=json.dumps(stream_request, sort_keys=True).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        stream_state: dict[str, Any] = {"model": stream_request.get("model", "")}

        def emit_wait_keepalive() -> None:
            # A pure SSE comment: every consumer of this stream skips ":"
            # lines, but the bytes keep the client's read loop (and every
            # intermediary's) fed while the join waits behind an exclusive
            # replay. A failed write means the client hung up, and the
            # raised exception aborts the wait instead of squatting on the
            # admission reservation for the rest of the budget.
            emit_raw_sse(SSE_KEEPALIVE_FRAME)

        with capture_trace_during_backend(
            spec=spec,
            request_id=request_id,
            proof_capture_required=(
                proof_capture_required and not skip_trace_capture
            ),
            capture_wait_timeout=capture_wait_timeout,
            on_capture_wait_tick=emit_wait_keepalive,
        ) as window_token:
            if capture_info is not None:
                capture_info["window_token"] = str(window_token or "")
            # Relay chunks raw and only COLLECT the data payloads; parsing
            # every chunk inline put a json.loads on the per-token path of
            # every concurrent stream (hundreds of parses per second through
            # one interpreter). The aggregate needs no mid-stream state, so
            # one parse pass after the stream closes is equivalent.
            pending_chunks: list[str] = []
            saw_done_sentinel = False
            try:
                with urlopen(req, timeout=BACKEND_FORWARD_TIMEOUT_S) as resp:
                    while True:
                        block = _read_sse_block(resp)
                        if not block:
                            break
                        _, data_lines = _parse_sse_block(block)
                        if data_lines:
                            raw_data = "\n".join(data_lines).strip()
                            if raw_data == "[DONE]":
                                # The worker terminates the client stream
                                # itself after its own done event; the
                                # backend's sentinel must not pass through.
                                saw_done_sentinel = True
                                continue
                            pending_chunks.append(raw_data)
                        emit_raw_sse(block)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} from backend stream: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"failed to connect to backend stream: {exc.reason}") from exc
            except (IncompleteRead, ConnectionError, TimeoutError) as exc:
                raise RuntimeError(
                    f"backend stream aborted mid-serve ({exc!r}); llama-server "
                    "crashed or dropped the connection before completing the "
                    "serve, so the partial stream cannot be receipted"
                ) from exc
        if not saw_done_sentinel:
            # llama-server terminates every successful OpenAI-compat stream
            # with the [DONE] sentinel, sent AFTER the slot-stamped final
            # chunk. A stream that ends without it was cut short: either a
            # mid-stream error event (emitted as a data payload, then the
            # connection closes) or a backend crash (bare EOF). Aggregating
            # the partial chunks would commit a receipt to a completion the
            # backend never finished, and at --parallel > 1 one with no slot
            # id to bind the proof to, so this fails closed instead
            # (observed: a CUDA abort mid-stream surfaced as a
            # misleading "runtime must be built with the Verathos server
            # patches" error out of the receipt path).
            backend_error: Any = None
            for raw_data in reversed(pending_chunks):
                try:
                    parsed = json.loads(raw_data)
                except json.JSONDecodeError:
                    # A crash can tear the last chunk mid-write; look one
                    # chunk further back for an intact error payload.
                    continue
                if isinstance(parsed, dict) and parsed.get("error") is not None:
                    backend_error = parsed["error"]
                break
            if backend_error is not None:
                raise RuntimeError(
                    "backend stream failed mid-serve: "
                    + json.dumps(backend_error, sort_keys=True)
                )
            raise RuntimeError(
                "backend stream ended without the [DONE] sentinel; "
                "llama-server terminated before completing the serve "
                "(backend crash or dropped connection), so the partial "
                "stream cannot be receipted"
            )
        for raw_data in pending_chunks:
            try:
                parsed = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                _accumulate_openai_stream_chunk(stream_state, parsed)
        return _stream_aggregate_response(
            request_id=request_id,
            openai_request=stream_request,
            state=stream_state,
        )

    def local_anchor_inventory_digest(receipt_context: Mapping[str, Any]) -> str:
        """Digest this capture window's streaming anchors, or "" if none.

        Fails closed: a runtime that wrote anchor artifacts which do not
        rebuild their committed roots raises here, before the receipt can
        commit to an inventory an audit could never open against.
        """

        if proof_trace_root is None:
            return ""
        from verallm.mesh.anchor_streams import (
            anchor_inventory_digest,
            load_anchor_streams,
        )

        streams = load_anchor_streams(proof_trace_root)
        return anchor_inventory_digest(streams)

    def anchor_rows_to_arm(receipt_context: Mapping[str, Any]) -> list[int]:
        """Union of joint per-op audit rows across every anchored tensor.

        Superset arming: the audited op's tensor is not known at arm time,
        so every anchored stage dumps its rows at the union indexes during
        the replay. The prover opens only the rows its op needs, and the
        loader leaf-verifies every dumped row regardless, so extra rows are
        harmless. Returns [] when anchors or a v2 beacon are absent, which
        keeps the replay on the full-dump path.
        """

        if proof_trace_root is None:
            return []
        beacon = str(receipt_context.get("proof_beacon", ""))
        if len(beacon) != 64:
            return []
        from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_op
        from verallm.mesh.anchor_streams import load_anchor_streams

        try:
            streams = load_anchor_streams(proof_trace_root)
        except Exception:
            return []
        rows: set[int] = set()
        for stage_id, stream in streams.items():
            if not stage_id.endswith(":src1"):
                continue
            dst = streams.get(stage_id[: -len(":src1")] + ":dst")
            if dst is None:
                continue
            try:
                rows.update(
                    select_anchor_audit_rows_for_op(
                        beacon=beacon,
                        src1_commitment=stream.commitment,
                        dst_commitment=dst.commitment,
                    )
                )
            except Exception:
                continue
        return sorted(rows)

    def load_anchored_rows_for_trace(
        trace: "GgmlMulMatTrace",
        receipt_context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Anchored witness for one audited op, or None for the full path.

        DORMANT IN PRODUCTION, deliberately. Two independent reasons, both
        verified, and both must be lifted together before this can run:

        1. Production always stamps SLOT_VIEW_SCOPE, which dispatches to
           make_embedded_slot_view_proof_payload; that maker never passes
           anchored_rows, so this function is not reached from the serving
           path at all.
        2. Nothing arms anchor capture, so no streams exist and the origin
           receipt carries no proof_anchor_inventory_digest.

        Lifting them would buy NO latency. Slot-view requires
        src1_shape[1] == 1, so the audited op is a single row and the
        activation side this bounds is already 0.09 s of a 3.1 s prove;
        92% is PCS proving over the weight geometry, which no amount of
        row bounding touches. See docs/architecture/mesh_hardening_progress.md
        "the flatness premise was false".

        The machinery is retained because it is tested and because its row
        selection is the sound one for any FUTURE multi-row audit: rows
        derive from the frozen op-manifest entry, not from commitments the
        prover generates after seeing the nonce.

        Every miss falls back to the full-dump relation rather than failing
        the audit: the full path is the heavier, strictly-stronger shape,
        so falling back is always sound.
        """

        if proof_trace_root is None:
            return None
        beacon = str(receipt_context.get("proof_beacon", ""))
        origin_digest = str(
            receipt_context.get("proof_anchor_inventory_digest", "")
        )
        if len(beacon) != 64 or not origin_digest:
            return None
        from verallm.mesh.anchor_audit import select_anchor_audit_rows_for_op
        from verallm.mesh.anchor_streams import (
            anchor_inventory_digest,
            load_anchor_row_dumps,
            load_anchor_streams,
        )

        # Scope to the replay window by BASE token prefix: every stage
        # composes its own full token (local selection markers differ per
        # member), but the minted base is shared mesh-wide.
        window_token = str(
            receipt_context.get("proof_replay_window_token", "")
        )
        if not window_token:
            return None
        try:
            streams = load_anchor_streams(
                proof_trace_root, capture_token_prefix=window_token
            )
            if anchor_inventory_digest(streams) != origin_digest:
                logger.warning(
                    "replay anchor inventory diverged from the frozen "
                    "origin digest; hard audit stays on the full path"
                )
                return None
            tensor_name = trace.src0_name or trace.tensor_name
            src1 = streams.get(f"{tensor_name}:src1")
            dst = streams.get(f"{tensor_name}:dst")
            if src1 is None or dst is None:
                return None
            selected = select_anchor_audit_rows_for_op(
                beacon=beacon,
                src1_commitment=src1.commitment,
                dst_commitment=dst.commitment,
            )
            dumps = load_anchor_row_dumps(
                proof_trace_root, streams, capture_token_prefix=window_token
            )
            for stage_id in (src1.stage_id, dst.stage_id):
                stage_rows = dumps.get(stage_id, {})
                if any(index not in stage_rows for index in selected):
                    return None
            return {"streams": streams, "row_dumps": dumps}
        except Exception:
            logger.warning(
                "anchored witness unavailable; hard audit stays on the "
                "full path",
                exc_info=True,
            )
            return None

    def _with_stage_boundary_roots(
        receipt_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach this stage's captured activation boundary roots.

        The patched RPC server writes ``boundary.jsonl`` next to the op
        manifest whenever ``VERATHOS_GGML_BOUNDARY_CAPTURE=1``; the roots
        commit everything that crossed the stage's RPC edge during the
        capture window, and the verifier chains them across stages. A
        worker without the capture build simply has no file and the
        receipts keep empty roots, which the staged
        ``VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN`` gate tolerates.
        """

        if proof_trace_root is None:
            return receipt_context
        from verallm.mesh.ggml_proof import read_boundary_roots_for_window

        roots = read_boundary_roots_for_window(
            proof_trace_root,
            start_unix_ns=int(
                receipt_context.get("inference_started_unix_ns", 0) or 0
            ),
        )
        if not roots:
            return receipt_context
        return {**receipt_context, **roots}

    def make_embedded_slot_view_proof_payload(
        receipt_context: dict[str, Any],
        slot_view_context: dict[str, Any],
        *,
        include_proof: bool = False,
        proof_required: bool = True,
    ) -> dict[str, Any]:
        """Prove beacon-selected slot-view leaves from solo replay dumps."""

        assembly_started = time.monotonic()

        def _journal_assembly(result: dict[str, Any]) -> dict[str, Any]:
            _journal_capture_event(
                "assembly"
                f" ms={int((time.monotonic() - assembly_started) * 1000)}"
                f" receipts={len(result.get('proof_receipts', []) or [])}"
                f" mode={str(result.get('proof_mode', '') or '-')}"
            )
            return result

        from verallm.mesh.ggml_proof import (
            ORGANIC_LIGHT_CHALLENGE_KIND,
            find_traces_for_window,
            ggml_slot_view_selection_payload_from_template,
            prove_ggml_light_trace_verified,
            prove_ggml_mul_mat_trace,
            slot_view_membership_payload_from_template,
        )

        organic_light = bool(
            (
                str(receipt_context.get("proof_challenge_kind", ""))
                == ORGANIC_LIGHT_CHALLENGE_KIND
                and not bool(
                    receipt_context.get("proof_tier_hard_requested", False)
                )
            )
            or str(receipt_context.get("proof_audit_tier", "")) == "light"
        )

        template = local_slot_view_template(receipt_context, slot_view_context)
        selection = ggml_slot_view_selection_payload_from_template(
            template=template,
            receipt_context={
                **receipt_context,
                "proof_ops_per_request": int(
                    receipt_context.get("proof_ops_per_request")
                    or proof_ops_per_request
                ),
            },
        )
        request_holder = str(receipt_context.get("request_id", "") or "")
        _t_acquire = time.perf_counter()
        tail_flush_ns = (
            int(receipt_context.get("proof_tail_flush_ns", 0) or 0)
            if organic_light
            else 0
        )
        tail_meta: dict[str, tuple[int, int]] = {}
        if tail_flush_ns:
            # Probe-free light tier: decode-audit leaves open rows from the
            # serve-time tail flush the selection verified as covering
            # (flush id > 0); base light leaves make no value claim and are
            # synthesized from committed leaf metadata below, so a
            # base-only receipt (flush id -1) needs no capture at all.
            from verallm.mesh.ggml_proof import GgmlMulMatTrace

            replay_traces = []
            tail_seq_offset: int | None = None
            if tail_flush_ns > 0:
                tail_items = _scan_tail_groups().get(tail_flush_ns, [])
                # Same offset resolution the covering check ran at selection
                # time: the group's last instance is the stop draw on a
                # naturally ended reply but the last committed position on a
                # capped one, and only the acceptance predicate can tell.
                tail_seq_offset = tail_group_position_offset(
                    tail_items,
                    positions=[
                        int(p)
                        for p in receipt_context.get(
                            "decode_audit_positions", []
                        )
                        or []
                    ],
                    committed_ids=[
                        int(t)
                        for t in receipt_context.get(
                            "decode_audit_completion_token_ids", []
                        )
                        or []
                    ],
                    top_k=int(
                        receipt_context.get(
                            "decode_audit_top_k", decode_audit_top_k
                        )
                    ),
                )
                for item in tail_items:
                    try:
                        trace = GgmlMulMatTrace.from_json(
                            str(item["_tail_path"])
                        )
                    except Exception:
                        continue
                    tail_meta[str(trace.path)] = (
                        int(item.get("tail_seq", -1)),
                        int(item.get("tail_total", 0) or 0),
                    )
                    replay_traces.append(trace)
        else:
            with capture_cond:
                holder_window = list(
                    capture_state["exclusive_window_ns_by_holder"].get(
                        request_holder, ()
                    )
                )
            if holder_window:
                armed_ns = int(holder_window[0] or 0)
                disarmed_ns = int(holder_window[1] or 0)
            else:
                # Pre-per-holder fallback (a coordinator older than this
                # worker): the global timestamp is correct only while audits
                # never overlap.
                armed_ns = int(capture_state.get("exclusive_armed_ns", 0) or 0)
                disarmed_ns = 0
            if armed_ns <= 0:
                if proof_required:
                    raise RuntimeError(
                        "slot view proof needs a selected replay capture first"
                    )
                return {"proof_mode": "", "verified": False, "proof_receipts": []}
            replay_traces = find_traces_for_window(
                proof_trace_root,
                start_unix_ns=armed_ns,
                end_unix_ns=disarmed_ns,
            )
        if _TIMING_LOG_ENABLED:
            print(
                f"VMESH_TIMING light_trace_acquire tail={tail_flush_ns} "
                f"{time.perf_counter() - _t_acquire:.4f}",
                file=sys.stderr,
                flush=True,
            )
        _t_prove = time.perf_counter()
        stage_index = int(receipt_context.get("stage_index", 0))
        proofs = []
        errors: list[str] = []
        decode_errors: list[str] = []
        # v3 leaves carry no graph ordinal; the committed solo layout's
        # ordinal is arithmetic over the token index and is only used below
        # as a deterministic tie-break anchor among captured instances.
        try:
            from verallm.mesh.ggml_proof import solo_prefill_graph_count

            leaf_prefill_graphs = solo_prefill_graph_count(
                prompt_token_count=int(
                    receipt_context.get("prompt_token_count", 0) or 0
                ),
                n_ubatch=int(
                    receipt_context.get("proof_runtime_ubatch_size", 0) or 0
                ),
            )
        except ValueError:
            leaf_prefill_graphs = 1
        for item in selection.get("selected", []):
            intra_index = int(item["replay_intra_index"])
            token_index = int(item["replay_token_index"])
            leaf_name = str(item["leaf"].get("tensor_name", ""))
            leaf_graph_ord = leaf_prefill_graphs + token_index + 1
            decode_positions_for_trace = [
                int(position)
                for position in item.get("decode_audit_positions", []) or []
            ]
            if tail_flush_ns and not decode_positions_for_trace:
                # Probe-free base light leaf: the light payload makes no
                # value claim (openings exist only on decode-audit leaves)
                # and the verifier binds the trace purely by leaf metadata
                # (intra index, tensor name, k/n dims, source types, one
                # activation row), all of which the committed leaf itself
                # carries. Synthesize the trace instead of manufacturing a
                # witness file with a teacher-forced probe.
                from verallm.mesh.ggml_proof import (
                    GgmlMulMatTrace,
                    SlotViewLeaf,
                )

                leaf_obj = SlotViewLeaf.from_mapping(item["leaf"])
                trace = GgmlMulMatTrace(
                    path=Path(
                        f"slot-view-leaf-{int(item['leaf_index'])}.synthetic"
                    ),
                    created_unix_ns=time.time_ns(),
                    graph_id=f"slot-view-leaf-{int(item['leaf_index'])}",
                    op_index=int(item["leaf_index"]),
                    tensor_name=leaf_obj.tensor_name,
                    src0_shape=(int(leaf_obj.k_dim), int(leaf_obj.n_dim), 1, 1),
                    src1_shape=(int(leaf_obj.k_dim), 1, 1, 1),
                    dst_shape=(int(leaf_obj.n_dim), 1, 1, 1),
                    src0_f32_path=None,
                    src1_f32_path=None,
                    dst_f32_path=None,
                    source_types=dict(leaf_obj.source_types),
                    src0_name=leaf_obj.src0_name,
                    src1_name="",
                    dst_name="",
                    backend=leaf_obj.backend,
                    device=leaf_obj.device,
                    # Informational for slot-view scope, but the light
                    # payload serializer requires a real integer.
                    manifest_index=int(item["leaf_index"]),
                    graph_seq=-1,
                    intra_graph_index=int(leaf_obj.intra_graph_index),
                )
                membership = slot_view_membership_payload_from_template(
                    template=template,
                    leaf_index=int(item["leaf_index"]),
                    completion_token_count=int(
                        receipt_context.get("completion_token_count", 0)
                    ),
                    stage_index=stage_index,
                )
                try:
                    proofs.append(
                        prove_ggml_light_trace_verified(
                            trace,
                            receipt_context,
                            proof_block_size=proof_block_size,
                            slot_view_membership=membership,
                            decode_audit_positions=[],
                            decode_audit_token_ids=[],
                            decode_audit_top_k=int(
                                receipt_context.get(
                                    "decode_audit_top_k", decode_audit_top_k
                                )
                            ),
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "synthetic slot view leaf proof failed for leaf %s",
                        item.get("leaf_index"),
                    )
                    errors.append(f"leaf-{item.get('leaf_index')}: {exc}")
                continue
            # The challenged op is identified by its committed WEIGHT NAME,
            # never by intra ordinal: architectures with length-dependent op
            # streams (glm-dsa's sparse-attention indexer path switches with
            # KV state) shift every intra between the serve graph the
            # template came from and the replay graph, so an intra-exact
            # match misses the very instance the audit needs. The name plus
            # single activation row plus the verifier's dim/type checks bind
            # the instance to the committed weight just as tightly. Clean
            # decode steps are single-row. The exact decode-step count can
            # be one short of the committed token count when llama.cpp fuses
            # the first decoded token into the prefill graph (or reuses a
            # graph), so we do NOT require an exact ordinal for ordinary
            # GEMM checks. Decode audit leaves are centered around the
            # committed token graph, so pick the closest captured instance
            # instead of modulo-wrapping high token positions into early
            # decode traces.
            single_row = sorted(
                (
                    trace
                    for trace in replay_traces
                    if trace.tensor_name == leaf_name
                    and tuple(trace.src1_shape[1:2]) == (1,)
                ),
                key=lambda t: int(t.graph_seq),
            )
            # A probe window may only capture bounded multi-row instances: a
            # recurrent model can expose the verified runtime's eager prompt
            # tail as one GEMM. The shared ceiling matches the native filter
            # and verifier; tail-backed organic witnesses stay single-row.
            small_rows = sorted(
                (
                    trace
                    for trace in replay_traces
                    if trace.tensor_name == leaf_name
                    and len(trace.src1_shape) > 1
                    and 1
                    <= int(trace.src1_shape[1])
                    <= SLOT_VIEW_PROBE_MAX_ROWS
                ),
                key=lambda t: int(t.graph_seq),
            )
            candidate_pool = single_row if tail_flush_ns else small_rows
            if not candidate_pool:
                same_name = sorted(
                    (
                        (int(trace.graph_seq), tuple(trace.src1_shape))
                        for trace in replay_traces
                        if trace.tensor_name == leaf_name
                    ),
                    key=lambda item: item[0],
                )
                window_ords = sorted(
                    {int(trace.graph_seq) for trace in replay_traces}
                )
                message = (
                    f"no single-row decode instance captured for slot view op "
                    f"{leaf_name} (template intra {intra_index}); window "
                    f"captured {len(replay_traces)} instances over graph ords "
                    f"{window_ords[:16]}, same-name instances "
                    f"{same_name[:12]}"
                )
                errors.append(message)
                if decode_positions_for_trace:
                    decode_errors.append(message)
                continue
            if decode_positions_for_trace and tail_flush_ns and tail_meta:
                # Tail-ring assembly: instance order within the flush group
                # is exact (one final-logit instance per decode graph), so
                # the audited position maps to seq = position + offset with
                # the offset resolved once per receipt above. Content was
                # already checked against the committed token before the
                # probe window was skipped.
                position = min(int(p) for p in decode_positions_for_trace)
                trace = None
                if tail_seq_offset is not None:
                    want_seq = int(position) + int(tail_seq_offset)
                    for candidate in single_row:
                        meta = tail_meta.get(str(candidate.path))
                        if meta is None:
                            continue
                        seq, total = meta
                        if total > 0 and int(seq) == want_seq:
                            trace = candidate
                            break
                if trace is None:
                    raise RuntimeError(
                        "tail-light trace missing for audited position "
                        f"{position}"
                    )
            elif decode_positions_for_trace:
                # Whether llama.cpp fuses the first decoded token into the
                # prefill graph varies, so the committed graph_ord can sit
                # one off the captured sequence — and an off-by-one opens
                # the WRONG token's logits (observed: committed token
                # ranked 954 in the adjacent row). The capture already dumps
                # a small ord neighborhood; select the instance whose argmax
                # equals the committed token at the audited position. This
                # adds no prover freedom: it is exactly the acceptance
                # criterion the verifier enforces on the opened row.
                committed_ids = [
                    int(t)
                    for t in receipt_context.get(
                        "decode_audit_completion_token_ids", []
                    )
                    or []
                ]
                position = min(int(p) for p in decode_positions_for_trace)
                expected_token = (
                    committed_ids[position]
                    if 0 <= position < len(committed_ids)
                    else None
                )
                import numpy as np

                audit_top_k = max(
                    1,
                    int(
                        receipt_context.get(
                            "decode_audit_top_k", decode_audit_top_k
                        )
                        or decode_audit_top_k
                    ),
                )
                # Rank candidates by what the VERIFIER will accept on the
                # opened row: argmax equality first, then membership in the
                # committed top-k. Falling back to an arbitrary instance
                # emitted an opening for the wrong decode step, which then
                # failed far away as "streamed token is not in the committed
                # top-k" (observed at the max-tokens truncation
                # boundary, where the probe's extra forward sits next to the
                # audited position).
                argmax_hits: list[Any] = []
                topk_hits: list[Any] = []
                batched_hits = 0
                single_row_seen = 0
                if expected_token is not None:
                    for candidate in candidate_pool:
                        try:
                            rows = np.fromfile(
                                str(candidate.path).replace(
                                    ".json", "-dst.f32"
                                ),
                                dtype=np.float32,
                            )
                        except Exception:
                            continue
                        vocab = int(candidate.dst_shape[0]) if candidate.dst_shape else 0
                        if vocab <= 0 or rows.size % vocab:
                            continue
                        matrix = rows.reshape(-1, vocab)
                        contains_token = any(
                            int(np.argmax(row)) == int(expected_token)
                            for row in matrix
                        )
                        k = min(audit_top_k, vocab)
                        if not contains_token:
                            contains_token = any(
                                int(expected_token)
                                in set(
                                    int(idx)
                                    for idx in np.argpartition(row, -k)[-k:]
                                )
                                for row in matrix
                            )
                            in_topk_only = contains_token
                        else:
                            in_topk_only = False
                        if matrix.shape[0] != 1:
                            # A batched serve-time instance can contain the
                            # audited row among its rows and outrank the
                            # replay's single-row probe instance on the
                            # graph-ord tiebreak - but the decode-audit
                            # prover categorically refuses multi-row traces,
                            # so selecting one is a guaranteed leaf failure
                            # ("requires a single-row LM-head GEMM trace",
                            #
                            # probation resets on a 4-GPU glm mesh whose
                            # deferred audits collided with batching). Only
                            # single-row instances are provable candidates;
                            # a batched instance that DOES carry the token
                            # is counted so the failure message can name a
                            # batching collision (retryable: a fresh probe
                            # re-captures the row) instead of a mismatch.
                            if contains_token:
                                batched_hits += 1
                            continue
                        single_row_seen += 1
                        if contains_token and not in_topk_only:
                            argmax_hits.append(candidate)
                        elif contains_token:
                            topk_hits.append(candidate)
                pool = argmax_hits or topk_hits
                if not pool:
                    # No provable instance carries the committed token:
                    # opening one anyway would ship a proof we already know
                    # the verifier must reject. Name the real condition -
                    # BATCHING-COLLISION means the audited row exists but
                    # only inside a batched instance (another slot decoded
                    # in the probe's ubatch), which a fresh probe draw fixes.
                    collision = batched_hits > 0
                    raise RuntimeError(
                        "decode audit found no captured logits containing the "
                        f"committed token {expected_token} at position "
                        f"{position} of {len(committed_ids)} "
                        f"({len(candidate_pool)} candidate instances for "
                        f"{leaf_name}; single_row={single_row_seen} "
                        f"batched_with_token={batched_hits}"
                        f"{'; BATCHING-COLLISION' if collision else ''})"
                    )
                trace = min(
                    pool,
                    key=lambda t: abs(int(t.graph_seq) - int(leaf_graph_ord)),
                )
            else:
                trace = candidate_pool[token_index % len(candidate_pool)]
            membership = slot_view_membership_payload_from_template(
                template=template,
                leaf_index=int(item["leaf_index"]),
                completion_token_count=int(
                    receipt_context.get("completion_token_count", 0)
                ),
                stage_index=stage_index,
            )
            try:
                if organic_light:
                    proofs.append(
                        prove_ggml_light_trace_verified(
                            trace,
                            receipt_context,
                            proof_block_size=proof_block_size,
                            slot_view_membership=membership,
                            decode_audit_positions=decode_positions_for_trace,
                            decode_audit_token_ids=[
                                int(token_id)
                                for token_id in receipt_context.get(
                                    "decode_audit_completion_token_ids",
                                    [],
                                )
                                or []
                            ],
                            decode_audit_top_k=int(
                                receipt_context.get(
                                    "decode_audit_top_k", decode_audit_top_k
                                )
                            ),
                        )
                    )
                    continue
                proofs.append(
                    prove_ggml_mul_mat_trace(
                        trace,
                        receipt_context,
                        tolerance_abs=proof_tolerance_abs,
                        tolerance_rel=proof_tolerance_rel,
                        proof_block_size=proof_block_size,
                        spot_checks=proof_spot_checks,
                        include_proof=include_proof,
                        slot_view_membership=membership,
                        gguf_manifest=proof_gguf_manifest,
                        decode_audit_positions=decode_positions_for_trace,
                        decode_audit_token_ids=[
                            int(token_id)
                            for token_id in receipt_context.get(
                                "decode_audit_completion_token_ids",
                                [],
                            )
                            or []
                        ],
                        decode_audit_top_k=int(
                            receipt_context.get("decode_audit_top_k", decode_audit_top_k)
                        ),
                        verify_before_return=False,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "slot view leaf proof failed for %s (decode positions %s)",
                    trace.path.name,
                    decode_positions_for_trace,
                )
                errors.append(f"{trace.path.name}: {exc}")
                if decode_positions_for_trace:
                    decode_errors.append(f"{trace.path.name}: {exc}")
        if _TIMING_LOG_ENABLED:
            print(
                f"VMESH_TIMING light_prove_loop n={len(proofs)} "
                f"{time.perf_counter() - _t_prove:.4f}",
                file=sys.stderr,
                flush=True,
            )
        if decode_errors:
            # A failed decode-audit leaf is fatal: the openings are a receipt
            # obligation, and a partial payload would only fail verification
            # downstream with a less informative "missing positions" message.
            raise RuntimeError(
                "slot view decode audit leaf failed: "
                + "; ".join(decode_errors[:3])
            )
        if errors and proof_required:
            # Every beacon-selected leaf is an obligation: the verifier
            # recomputes the Fiat-Shamir base challenge set and rejects a
            # payload missing any of it. Failing here reports the ACTUAL
            # cause instead of the downstream "missing base challenge
            # indexes", which says nothing about why the leaf was dropped.
            raise RuntimeError(
                "slot view selected leaf failed: " + "; ".join(errors[:3])
            )
        if proofs:
            payload: dict[str, Any] = {
                "proof_mode": proofs[0].proof_mode,
                "verified": all(item.verified for item in proofs),
                "proof_receipts": [item.receipt.to_dict() for item in proofs],
                "trace_paths": [item.trace_path for item in proofs],
            }
            if include_proof or organic_light:
                payload["proof_payloads"] = [item.proof_payload for item in proofs]
                payload["proof_verifier_ms"] = sum(item.verifier_ms for item in proofs)
            return _journal_assembly(payload)
        if proof_required:
            raise RuntimeError(
                "no slot view GGML proof produced: " + "; ".join(errors[:3])
            )
        return {"proof_mode": "", "verified": False, "proof_receipts": []}

    def make_embedded_proof_payload(
        receipt_context: dict[str, Any],
        *,
        include_proof: bool = False,
        proof_required: bool = True,
        openai_request: dict[str, Any] | None = None,
        openai_response: dict[str, Any] | None = None,
        slot_view_context: dict[str, Any] | None = None,
        pinned_spec: MeshSpec | None = None,
    ) -> dict[str, Any]:
        if proof_trace_root is None:
            if proof_required:
                raise RuntimeError("embedded proof is not configured")
            return {"proof_mode": "", "verified": False, "proof_receipts": []}
        validate_local_proof_context(
            receipt_context,
            require_proof_gate=True,
            pinned_spec=pinned_spec,
        )
        _t_bound = time.perf_counter()
        receipt_context = _with_stage_boundary_roots(receipt_context)
        if _TIMING_LOG_ENABLED:
            print(
                f"VMESH_TIMING light_boundary_roots "
                f"{time.perf_counter() - _t_bound:.4f}",
                file=sys.stderr,
                flush=True,
            )

        from verallm.mesh.ggml_proof import (
            ORGANIC_LIGHT_CHALLENGE_KIND,
            SLOT_VIEW_SCOPE,
            _ordered_traces,
            _walk_to_provable_index,
            _witness_traces,
            derive_every_request_trace_beacon,
            derive_standalone_trace_beacon,
            find_op_manifest_entries_for_window,
            find_traces_for_window,
            ggml_op_manifest_root,
            ggml_trace_commitment_root,
            op_manifest_membership_payload,
            prove_ggml_light_trace_verified,
            select_manifest_challenge_indexes,
            select_decode_manifest_entries,
            select_trace_challenge_indexes,
            trace_membership_payload,
            prove_ggml_mul_mat_trace,
        )

        if slot_view_scope_for(receipt_context) == SLOT_VIEW_SCOPE:
            if not slot_view_context:
                raise RuntimeError(
                    "slot view receipts need a slot_view_context for proofs"
                )
            return publicize_stage_proof_payload(
                make_embedded_slot_view_proof_payload(
                    receipt_context,
                    slot_view_context,
                    include_proof=include_proof,
                    proof_required=proof_required,
                ),
                receipt_context,
                pinned_spec=pinned_spec,
            )

        def replay_selected_manifest_indexes(indexes: list[int]) -> list[Any]:
            if not indexes:
                return []
            if not backend_url or proof_trace_enable_path is None:
                return []
            if openai_request is None or openai_response is None:
                return []
            replay_request = backend_openai_request(
                openai_request,
                proof_capture_required=True,
                prompt_cache_disabled=True,
                verified_sampler_required=bool(
                    receipt_context.get("verified_sampler_required")
                    or receipt_context.get("decode_audit_required")
                ),
                proof_metadata_required=bool(
                    receipt_context.get("proof_metadata_required")
                    or receipt_context.get("decode_audit_required")
                ),
                sampled_profile=sampled_controls_from_context(receipt_context),
            )
            if (
                receipt_context.get("decode_audit_required")
                and not receipt_context.get("decode_audit_completion_token_ids")
            ):
                replay_request["logprobs"] = True
                replay_request["top_logprobs"] = 1
            started_ns = time.time_ns()
            set_local_trace_capture(
                True,
                selected_manifest_indexes=[int(item) for item in indexes],
            )
            try:
                raw_replay_response = post_json(
                    f"{backend_url}/v1/chat/completions",
                    replay_request,
                    timeout=120.0,
                )
            finally:
                set_local_trace_capture(False)
            ended_ns = time.time_ns()
            previous_started_ns = int(
                receipt_context.get("proof_replay_started_unix_ns", 0) or 0
            )
            previous_ended_ns = int(
                receipt_context.get("proof_replay_ended_unix_ns", 0) or 0
            )
            receipt_context["proof_replay_started_unix_ns"] = (
                min(previous_started_ns, started_ns)
                if previous_started_ns
                else started_ns
            )
            receipt_context["proof_replay_ended_unix_ns"] = max(
                previous_ended_ns,
                ended_ns,
            )
            replay_response, replay_token_ids, replay_token_source = (
                prepare_backend_response_for_receipt(
                    raw_replay_response,
                    proof_capture_required=True,
                    stream=False,
                )
            )
            if not receipt_context.get("decode_audit_required") and not replay_token_source:
                replay_token_ids, replay_token_source = (
                    fetch_completion_token_ids_from_backend(backend_url, replay_response)
                )
            expected_response_hash = semantic_openai_response_hash(openai_response)
            replay_response_hash = semantic_openai_response_hash(replay_response)
            if replay_response_hash != expected_response_hash:
                raise RuntimeError(
                    "selected GGML witness replay semantic response mismatch"
                )
            bind_decode_audit_completion_tokens(
                receipt_context,
                replay_token_ids,
                replay_token_source,
            )
            if str(receipt_context.get("completion_token_source", "")):
                if not replay_token_source:
                    raise RuntimeError("selected GGML witness replay missing token ids")
                if completion_token_ids_hash(replay_token_ids) != str(
                    receipt_context.get("completion_token_ids_hash", "")
                ):
                    raise RuntimeError("selected GGML witness replay token hash mismatch")
            return find_traces_for_window(
                proof_trace_root,
                start_unix_ns=started_ns,
                end_unix_ns=ended_ns,
            )

        def trace_for_manifest_entry(entry, available_traces: list[Any]) -> Any | None:
            return next(
                (
                    item
                    for item in _ordered_traces(available_traces)
                    if entry.matches_trace(item)
                ),
                None,
            )

        def ensure_manifest_entry_trace(entry) -> tuple[Any, int]:
            trace_index = next(
                (
                    idx
                    for idx, item in enumerate(candidates)
                    if entry.matches_trace(item)
                ),
                -1,
            )
            trace = trace_for_manifest_entry(entry, traces)
            if trace is None and replay_traces:
                trace = trace_for_manifest_entry(entry, replay_traces)
            if trace is None:
                local_replay_traces = replay_selected_manifest_indexes(
                    [int(entry.manifest_index)]
                )
                if local_replay_traces:
                    traces.extend(local_replay_traces)
                    trace = trace_for_manifest_entry(entry, local_replay_traces)
            if trace is None:
                raise RuntimeError(
                    "manifest-selected GGML op was not captured as a proof witness: "
                    f"manifest_index={entry.manifest_index}"
                )
            return trace, trace_index

        traces = find_traces_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        manifest_entries = find_op_manifest_entries_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        replay_traces = []
        replay_started_ns = int(receipt_context.get("proof_replay_started_unix_ns", 0) or 0)
        replay_ended_ns = int(receipt_context.get("proof_replay_ended_unix_ns", 0) or 0)
        if replay_started_ns and replay_ended_ns and replay_ended_ns >= replay_started_ns:
            replay_traces = find_traces_for_window(
                proof_trace_root,
                start_unix_ns=replay_started_ns,
                end_unix_ns=replay_ended_ns,
            )
        if not traces and not manifest_entries:
            if proof_required:
                raise RuntimeError("no GGML proof trace or op manifest found for inference window")
            return {"proof_mode": "", "verified": False, "proof_receipts": []}
        errors: list[str] = []
        proofs = []
        proof_limit = max(
            1,
            int(receipt_context.get("proof_ops_per_request") or proof_ops_per_request),
        )
        candidate_limit = max(
            proof_limit,
            int(
                receipt_context.get("proof_trace_candidates_per_request")
                or proof_trace_candidates_per_request
            ),
        )
        candidates = _ordered_traces(_witness_traces(traces))[:candidate_limit]
        trace_root = ggml_trace_commitment_root(candidates)
        manifest_ordered = []
        manifest_root = ""
        if manifest_entries:
            from verallm.mesh.ggml_proof import _ordered_manifest_entries

            manifest_ordered = _ordered_manifest_entries(manifest_entries)
            manifest_root = ggml_op_manifest_root(manifest_ordered)
        beacon = str(receipt_context.get("proof_beacon", ""))
        if not beacon:
            gate_hash = str(receipt_context.get("proof_gate_hash", ""))
            beacon = (
                derive_every_request_trace_beacon(gate_hash).hex()
                if gate_hash
                else derive_standalone_trace_beacon(trace_root).hex()
            )
        selected_items = []
        # Base GGML proof selection and decode auditing are independent.  A
        # decode-sampled response carries the ordinary challenge as well as
        # the final-logit opening; it must never replace the base proof.
        proof_sampled_for_gemm = bool(receipt_context.get("proof_sampled", True))
        use_manifest_challenge = (
            manifest_ordered
            and manifest_root
            and proof_sampled_for_gemm
            and str(receipt_context.get("proof_trace_scope", ""))
            == "op_manifest_challenge_v1"
        )
        if use_manifest_challenge:
            selected_manifest_indexes = select_manifest_challenge_indexes(
                beacon=beacon,
                op_manifest_root=manifest_root,
                op_manifest_count=len(manifest_ordered),
                proof_ops_per_request=proof_limit,
                stage_index=int(receipt_context.get("stage_index", 0)),
            )
            for selected_manifest_index in selected_manifest_indexes:
                # Walk past 1-D gate-vector ops (MoE) that have no committed
                # proof matrix; same committed manifest set, deterministic.
                selected_manifest_index = _walk_to_provable_index(
                    manifest_ordered, selected_manifest_index, lambda e: e.src0_shape
                )
                entry = manifest_ordered[selected_manifest_index]
                trace, trace_index = ensure_manifest_entry_trace(entry)
                selected_items.append((trace, trace_index, []))
        elif proof_sampled_for_gemm:
            selected_indexes = select_trace_challenge_indexes(
                beacon=beacon,
                trace_set_root=trace_root,
                trace_set_count=len(candidates),
                proof_ops_per_request=proof_limit,
                stage_index=int(receipt_context.get("stage_index", 0)),
            )
            remapped: list[int] = []
            for selected_index in selected_indexes:
                index = _walk_to_provable_index(
                    candidates, selected_index, lambda t: t.src0_shape
                )
                if index not in remapped:
                    remapped.append(index)
            selected_items = [
                (candidates[index], index, []) for index in remapped
            ]

        decode_positions = [
            int(item)
            for item in receipt_context.get("decode_audit_positions", []) or []
        ]
        decode_token_ids = [
            int(item)
            for item in receipt_context.get("decode_audit_completion_token_ids", []) or []
        ]
        decode_active = bool(
            receipt_context.get("decode_audit_required", False)
            and int(receipt_context.get("stage_index", 0))
            == int(
                receipt_context.get(
                    "decode_audit_stage_index",
                    receipt_context.get("stage_index", 0),
                )
            )
        )
        if decode_active:
            if not decode_positions:
                raise RuntimeError("decode audit is required but no positions were selected")
            if not decode_token_ids:
                raise RuntimeError("decode audit is required but completion tokens are missing")
            if not manifest_entries:
                raise RuntimeError("decode audit requires a GGML op manifest")
            decode_entries = select_decode_manifest_entries(
                manifest_entries,
                completion_token_ids=decode_token_ids,
                positions=decode_positions,
            )
            by_path = {str(item[0].path): idx for idx, item in enumerate(selected_items)}
            for position, entry in sorted(decode_entries.items()):
                trace, trace_index = ensure_manifest_entry_trace(entry)
                key = str(trace.path)
                if key in by_path:
                    selected_items[by_path[key]][2].append(int(position))
                else:
                    by_path[key] = len(selected_items)
                    selected_items.append((trace, trace_index, [int(position)]))

        # Organic requests ride the LIGHT tier by default: openings-only
        # payloads that stay milliseconds at any context, per the tier
        # design (hard stays the validator-audit relation and can be forced
        # via the upgrade-only proof_tier request). On the postcommit lane
        # the tier is the nonce-derived draw stamped into the audit context
        # (proof_audit_tier), which both sides recompute from the reveal.
        # Requires the op manifest so light membership stays anchored;
        # legacy manifest-less captures keep the hard path.
        organic_light = bool(
            (
                (
                    str(receipt_context.get("proof_challenge_kind", ""))
                    == ORGANIC_LIGHT_CHALLENGE_KIND
                    and not bool(
                        receipt_context.get("proof_tier_hard_requested", False)
                    )
                )
                or str(receipt_context.get("proof_audit_tier", "")) == "light"
            )
            and manifest_entries
        )
        for trace, selected_index, decode_positions_for_trace in selected_items:
            try:
                manifest_membership = (
                    op_manifest_membership_payload(
                        manifest_entries,
                        trace,
                        stage_index=int(receipt_context.get("stage_index", 0)),
                    )
                    if manifest_entries
                    else None
                )
                if organic_light:
                    proofs.append(
                        prove_ggml_light_trace_verified(
                            trace,
                            receipt_context,
                            proof_block_size=proof_block_size,
                            op_manifest_membership=manifest_membership,
                            decode_audit_positions=decode_positions_for_trace,
                            decode_audit_token_ids=decode_token_ids,
                            decode_audit_top_k=int(
                                receipt_context.get(
                                    "decode_audit_top_k", decode_audit_top_k
                                )
                            ),
                        )
                    )
                    continue
                # Bounded anchored witness when the replay dumped this
                # tensor's beacon-selected rows; any miss keeps the full
                # dump relation, so the audit never weakens or fails on an
                # unavailable anchor. Decode audits stay on their own lane.
                anchored_rows = (
                    load_anchored_rows_for_trace(trace, receipt_context)
                    if proof_gguf_manifest is not None
                    and not decode_positions_for_trace
                    else None
                )
                proofs.append(
                    prove_ggml_mul_mat_trace(
                        trace,
                        receipt_context,
                        tolerance_abs=proof_tolerance_abs,
                        tolerance_rel=proof_tolerance_rel,
                        proof_block_size=proof_block_size,
                        spot_checks=proof_spot_checks,
                        include_proof=include_proof,
                        trace_membership=trace_membership_payload(
                            candidates,
                            selected_index,
                            stage_index=int(receipt_context.get("stage_index", 0)),
                        )
                        if selected_index >= 0
                        else None,
                        op_manifest_membership=manifest_membership,
                        gguf_manifest=proof_gguf_manifest,
                        decode_audit_positions=decode_positions_for_trace,
                        decode_audit_token_ids=decode_token_ids,
                        decode_audit_top_k=int(
                            receipt_context.get("decode_audit_top_k", decode_audit_top_k)
                        ),
                        anchored_rows=anchored_rows,
                        verify_before_return=False,
                    )
                )
            except Exception as exc:
                errors.append(f"{trace.path.name}: {exc}")
        if proofs:
            payload: dict[str, Any] = {
                "proof_mode": proofs[0].proof_mode,
                "verified": all(item.verified for item in proofs),
                "proof_receipts": [item.receipt.to_dict() for item in proofs],
                "trace_paths": [item.trace_path for item in proofs],
            }
            if include_proof or organic_light:
                # Light payloads ARE the proof: they must always travel with
                # their receipts or the aggregate verifier has nothing to pair.
                payload["proof_payloads"] = [item.proof_payload for item in proofs]
                payload["proof_verifier_ms"] = sum(item.verifier_ms for item in proofs)
            return publicize_stage_proof_payload(
                payload,
                receipt_context,
                pinned_spec=pinned_spec,
            )
        if proof_required:
            raise RuntimeError("no GGML proof trace verified: " + "; ".join(errors[:3]))
        return {"proof_mode": "", "verified": False, "proof_receipts": []}

    def make_embedded_trace_commitment_payload(
        receipt_context: dict[str, Any],
        slot_view_context: dict[str, Any] | None = None,
        *,
        pinned_spec: MeshSpec | None = None,
    ) -> dict[str, Any]:
        if proof_trace_root is None:
            raise RuntimeError("embedded proof is not configured")
        validate_local_proof_context(
            receipt_context,
            pinned_spec=pinned_spec,
        )

        from verallm.mesh.ggml_proof import (
            SLOT_VIEW_SCOPE,
            _ordered_traces,
            _witness_traces,
            find_traces_for_window,
            ggml_op_manifest_summary_for_window,
            ggml_trace_commitment_root,
        )

        if slot_view_scope_for(receipt_context) == SLOT_VIEW_SCOPE:
            if not slot_view_context:
                raise RuntimeError(
                    "slot view receipts need a slot_view_context for commitments"
                )
            slot_view_root, slot_view_count = local_slot_view_root_count(
                receipt_context,
                slot_view_context,
            )
            if not slot_view_root or slot_view_count <= 0:
                raise RuntimeError(
                    "no slot view leaves found for the request's capture window"
                )
            return {
                "version": 1,
                "stage_index": int(receipt_context.get("stage_index", 0)),
                "trace_commitment_root": "",
                "trace_commitment_count": 0,
                "op_manifest_root": slot_view_root,
                "op_manifest_count": int(slot_view_count),
                "op_manifest_scope": SLOT_VIEW_SCOPE,
            }

        traces = find_traces_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        manifest_root, manifest_count = ggml_op_manifest_summary_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        proof_limit = max(
            1,
            int(receipt_context.get("proof_ops_per_request") or proof_ops_per_request),
        )
        candidate_limit = max(
            proof_limit,
            int(
                receipt_context.get("proof_trace_candidates_per_request")
                or proof_trace_candidates_per_request
            ),
        )
        selected = _ordered_traces(_witness_traces(traces))[:candidate_limit]
        payload = {
            "version": 1,
            "stage_index": int(receipt_context.get("stage_index", 0)),
            "trace_commitment_root": ggml_trace_commitment_root(selected) if selected else "",
            "trace_commitment_count": len(selected),
        }
        if manifest_root:
            payload["op_manifest_root"] = manifest_root
            payload["op_manifest_count"] = int(manifest_count)
        # An empty local stage is legitimate when this node offloads all
        # compute to remote rpc-workers (coordinator with --llama-device
        # RPC*): it contributes nothing and the workers' commitments carry the
        # proof. The aggregate-level check (proof capture requires a commitment
        # OR op manifest) still fails a genuinely empty mesh.
        if not selected and not manifest_root and not _mesh_has_remote_members:
            raise RuntimeError("no GGML proof trace or op manifest found for inference window")
        return payload

    def make_embedded_proof_selection_payload(
        receipt_context: dict[str, Any],
        slot_view_context: dict[str, Any] | None = None,
        *,
        pinned_spec: MeshSpec | None = None,
        serve_n_parallel: int = 1,
    ) -> dict[str, Any]:
        if proof_trace_root is None:
            raise RuntimeError("embedded proof is not configured")
        validate_local_proof_context(
            receipt_context,
            pinned_spec=pinned_spec,
        )

        from verallm.mesh.ggml_proof import (
            SLOT_VIEW_SCOPE,
            find_op_manifest_entries_for_window,
            find_traces_for_window,
            ggml_proof_selection_payload,
            ggml_slot_view_selection_payload_from_template,
        )

        if slot_view_scope_for(receipt_context) == SLOT_VIEW_SCOPE:
            if not slot_view_context:
                raise RuntimeError(
                    "slot view receipts need a slot_view_context for selection"
                )
            template = local_slot_view_template(receipt_context, slot_view_context)
            selection = ggml_slot_view_selection_payload_from_template(
                template=template,
                receipt_context={
                    **receipt_context,
                    "proof_ops_per_request": int(
                        receipt_context.get("proof_ops_per_request")
                        or proof_ops_per_request
                    ),
                },
            )
            selected_items = selection.get("selected", []) or []
            if _TIMING_LOG_ENABLED:
                _journal_capture_event(
                    "tail-light member gate: enabled=%s organic=%s kind=%s "
                    "selected=%d audits=%d"
                    % (
                        _light_tail_capture_enabled(),
                        _organic_light_receipt(receipt_context),
                        str(receipt_context.get("proof_challenge_kind", "")),
                        len(selected_items),
                        sum(
                            1
                            for item in selected_items
                            if item.get("decode_audit_positions")
                        ),
                    )
                )
            if (
                _light_tail_capture_enabled()
                and _organic_light_receipt(receipt_context)
                and selected_items
            ):
                # Probe-free light tier. Base light leaves make no value
                # claim - their payload is synthesized from the committed
                # leaf metadata - so only decode-audit (final-logit) leaves
                # need serve-time material, and the tail ring's flush holds
                # exactly those rows. Advertise readiness so the
                # coordinator can skip the exclusive window + probes; any
                # miss simply omits the field and the probe path runs.
                audit_items = [
                    item
                    for item in selected_items
                    if item.get("decode_audit_positions")
                ]
                if not audit_items:
                    selection["tail_flush_ns"] = -1
                    _journal_capture_event("tail-light ready base-only")
                else:
                    tail_probe_ctx = dict(receipt_context)
                    if wait_for_light_tail_material(
                        tail_probe_ctx,
                        serve_n_parallel=serve_n_parallel,
                    ):
                        selection["tail_flush_ns"] = int(
                            tail_probe_ctx["proof_tail_flush_ns"]
                        )
            return selection

        traces = find_traces_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        manifest_entries = find_op_manifest_entries_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0)),
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0)),
        )
        return ggml_proof_selection_payload(
            traces=traces,
            manifest_entries=manifest_entries,
            receipt_context={
                **receipt_context,
                "proof_ops_per_request": int(
                    receipt_context.get("proof_ops_per_request")
                    or proof_ops_per_request
                ),
            },
        )

    def local_member_for_spec(spec: MeshSpec | None) -> MeshMember | None:
        if spec is None:
            return None
        return next(
            (item for item in spec.members if item.endpoint == capability.endpoint),
            None,
        )

    def same_host_loopback(url: str) -> str:
        """Dial co-located members via loopback, not the advertised address.

        Members advertise the address the POOL reaches them at. A machine
        behind NAT without hairpin routing cannot reach its OWN advertised
        address, so a same-host dial silently hangs until the caller's
        timeout (observed: the driver's serving self-test spent its
        whole 120s dialing the co-located proof stage via the public IP,
        with an idle GPU and nothing logged, and the mesh never reached
        serving). Datacenter boxes never hit this because their advertised
        address is bound on a local interface.
        """

        own = str(capability.endpoint or "")
        own_host = own.split("://")[-1].rsplit(":", 1)[0]
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if own_host and parsed.hostname == own_host and parsed.port:
                return f"{parsed.scheme}://127.0.0.1:{parsed.port}"
        except Exception:
            return url
        return url

    def proof_members_for_spec(spec: MeshSpec) -> list[MeshMember]:
        """Return the stages this server must prove for one artifact.

        A coordinator aggregates every non-empty compute stage.  A leaf stage
        serves only its own internal artifact; requiring it to contact every
        peer would both disclose topology and duplicate coordinator fan-in.
        """

        compute = [
            member
            for member in sorted(spec.members, key=lambda item: item.stage_index)
            if member.layers.end > member.layers.start
        ]
        local = local_member_for_spec(spec)
        if local is None:
            raise RuntimeError("local capability is not present in the mesh spec")
        if local.role == "coordinator":
            return compute
        return [local] if local.layers.end > local.layers.start else []

    def private_proof_tokens_for_spec(spec: MeshSpec | None) -> tuple[str, ...]:
        if spec is None:
            return ()
        tokens: set[str] = set()
        for member in spec.members:
            tokens.update(
                {
                    str(member.endpoint or ""),
                    str(member.proof_endpoint or ""),
                    str(member.rpc_endpoint or ""),
                    str(member.hotkey or ""),
                    str(member.proof_key or ""),
                }
            )
        return tuple(sorted((item for item in tokens if item), key=len, reverse=True))

    def receipt_context_for_member(
        receipt_context: dict[str, Any],
        member: MeshMember,
        *,
        spec: MeshSpec | None = None,
        verification_snapshot: Any | None = None,
    ) -> dict[str, Any]:
        ctx = dict(receipt_context)
        ctx.update(
            {
                "uid": member.uid,
                "hotkey": member.hotkey,
                "endpoint": member.endpoint,
                "stage_index": member.stage_index,
                "layer_start": member.layers.start,
                "layer_end": member.layers.end,
            }
        )
        if str(ctx.get("proof_receipt_format", "")) == "opaque_stage_v2":
            pinned_spec = spec if spec is not None else current_mesh_spec()
            snapshot = (
                verification_snapshot
                if verification_snapshot is not None
                else current_verification_snapshot(pinned_spec)
            )
            if str(ctx.get("verification_snapshot_hash", "")) != (
                snapshot.snapshot_hash_hex()
            ):
                raise RuntimeError("proof context verification snapshot mismatch")
            stage = next(
                (
                    item
                    for item in snapshot.stages
                    if int(item.layer_start) == int(member.layers.start)
                    and int(item.layer_end) == int(member.layers.end)
                ),
                None,
            )
            if stage is None:
                raise RuntimeError("proof member is absent from verification snapshot")
            expected_key = str(member.proof_key or member.hotkey).strip()
            if stage.proof_key != expected_key:
                raise RuntimeError("proof member key does not match verification snapshot")
            ctx.update(
                {
                    "stage_id": stage.stage_id,
                    "stage_proof_key": stage.proof_key,
                    "stage_proof_key_scheme": stage.proof_key_scheme,
                    "stage_proof_commitment": stage.proof_commitment,
                    "model_index": int(snapshot.coordinator.model_index),
                    "model_total_layers": int(snapshot.model.total_layers),
                }
            )
        return ctx

    def validate_local_proof_context(
        receipt_context: dict[str, Any],
        *,
        require_proof_gate: bool = False,
        pinned_spec: MeshSpec | None = None,
    ) -> None:
        spec = pinned_spec if pinned_spec is not None else current_mesh_spec()
        opaque_stage_receipt = (
            str(receipt_context.get("proof_receipt_format", ""))
            == "opaque_stage_v2"
        )
        if opaque_stage_receipt:
            if spec is None:
                raise RuntimeError("local mesh spec is unavailable")
            validate_mesh_stage_context_commitments(receipt_context, spec)
        member = local_member_for_spec(spec)
        if member is None:
            if opaque_stage_receipt:
                raise RuntimeError(
                    "local capability is absent from the mesh specification"
                )
            return
        expected = {
            "uid": member.uid,
            "hotkey": member.hotkey,
            "endpoint": member.endpoint,
            "stage_index": member.stage_index,
            "layer_start": member.layers.start,
            "layer_end": member.layers.end,
        }
        for field, value in expected.items():
            if receipt_context.get(field) != value:
                raise RuntimeError(f"proof context {field} mismatch")
        if str(receipt_context.get("proof_receipt_format", "")) != "opaque_stage_v2":
            return
        expected_stage_key = str(member.proof_key or member.hotkey).strip()
        if not stage_proof_key or stage_receipt_signer is None:
            raise RuntimeError("secure stage proof receipt signing is not configured")
        if stage_proof_key != expected_stage_key:
            raise RuntimeError("configured stage proof key does not match runtime mesh")
        secure_expected = {
            "stage_proof_key": stage_proof_key,
            "stage_proof_key_scheme": "sr25519",
            "model_total_layers": int(spec.total_layers),
        }
        for field, value in secure_expected.items():
            if receipt_context.get(field) != value:
                raise RuntimeError(f"proof context {field} mismatch")
        for field in (
            "stage_id",
            "verification_snapshot_hash",
            "stage_proof_commitment",
        ):
            value = str(receipt_context.get(field, ""))
            if field == "stage_id":
                valid = value.startswith("stg_") and len(value) == 36
            else:
                valid = len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
            if not valid:
                raise RuntimeError(f"proof context {field} is invalid")
        if require_proof_gate:
            gate_hash = str(receipt_context.get("proof_gate_hash", ""))
            if not (
                len(gate_hash) == 64
                and all(ch in "0123456789abcdef" for ch in gate_hash)
            ):
                raise RuntimeError("proof context proof_gate_hash is invalid")
        if type(receipt_context.get("model_index")) is not int or int(
            receipt_context["model_index"]
        ) < 0:
            raise RuntimeError("proof context model_index is invalid")

    def publicize_stage_proof_payload(
        payload: dict[str, Any],
        receipt_context: dict[str, Any],
        *,
        pinned_spec: MeshSpec | None = None,
    ) -> dict[str, Any]:
        """Sanitize, bind, and sign proof receipts on their producing worker."""

        if str(receipt_context.get("proof_receipt_format", "")) != "opaque_stage_v2":
            return payload
        validate_local_proof_context(
            receipt_context,
            require_proof_gate=True,
            pinned_spec=pinned_spec,
        )
        raw_receipts = payload.get("proof_receipts", [])
        raw_payloads = payload.get("proof_payloads", [])
        if not isinstance(raw_receipts, list) or not isinstance(raw_payloads, list):
            raise RuntimeError("stage proof payload contains invalid receipt lists")
        if not raw_receipts:
            clean = dict(payload)
            clean.pop("trace_paths", None)
            return clean

        private_receipts = [LlamaGraphOpReceipt.from_dict(item) for item in raw_receipts]
        spec = pinned_spec if pinned_spec is not None else current_mesh_spec()
        private_tokens = private_proof_tokens_for_spec(spec)
        if raw_payloads:
            sanitized_receipts, sanitized_payloads = _sanitize_validator_proof_artifacts(
                [item.to_dict() for item in private_receipts],
                raw_payloads,
                private_tokens=private_tokens,
            )
            private_receipts = [
                LlamaGraphOpReceipt.from_dict(item) for item in sanitized_receipts
            ]
        else:
            sanitized_payloads = []

        from verallm.mesh.receipt_signing import verify_stage_proof_receipt_signature

        public_receipts: list[MeshStageProofReceipt] = []
        for private_receipt in private_receipts:
            for field in (
                "request_id",
                "mesh_id",
                "mesh_spec_hash",
                "stage_assignment_hash",
                "rpc_plan_hash",
                "model_package_hash",
                "model_tensor_manifest_root",
                "request_hash",
                "response_hash",
            ):
                if getattr(private_receipt, field) != receipt_context.get(field):
                    raise RuntimeError(f"private proof receipt {field} mismatch")
            for field in ("stage_index", "layer_start", "layer_end"):
                if getattr(private_receipt, field) != int(receipt_context[field]):
                    raise RuntimeError(f"private proof receipt {field} mismatch")
            public = MeshStageProofReceipt.from_private_receipt(
                private_receipt,
                stage_id=str(receipt_context["stage_id"]),
                model_index=int(receipt_context["model_index"]),
                model_total_layers=int(receipt_context["model_total_layers"]),
                verification_snapshot_hash=str(
                    receipt_context["verification_snapshot_hash"]
                ),
                proof_gate_hash=str(receipt_context["proof_gate_hash"]),
                stage_proof_commitment=str(
                    receipt_context["stage_proof_commitment"]
                ),
            )
            signature = str(stage_receipt_signer(public.body_hash().hex()) or "")
            signed = replace(public, signature=signature)
            signed.validate(require_signature=True)
            if not verify_stage_proof_receipt_signature(
                signed.body_hash().hex(),
                signed.signature,
                stage_proof_key,
                str(receipt_context["stage_proof_key_scheme"]),
            ):
                raise RuntimeError("stage proof receipt signature is invalid")
            public_receipts.append(signed)

        clean = dict(payload)
        clean["proof_receipts"] = [item.to_dict() for item in public_receipts]
        clean["proof_payloads"] = sanitized_payloads
        clean.pop("trace_paths", None)
        _assert_public_proof_artifact_value(
            {
                "proof_receipts": clean["proof_receipts"],
                "proof_payloads": clean["proof_payloads"],
            },
            private_tokens=private_tokens,
        )
        return clean

    def member_for_proof_url(spec: MeshSpec | None, url: str) -> MeshMember | None:
        if spec is None or not url:
            return None
        try:
            normalized = normalize_endpoint(url).rstrip("/")
        except ValueError:
            normalized = url.rstrip("/")
        return next(
            (
                member
                for member in spec.members
                if member.proof_endpoint.rstrip("/") == normalized
                or member.endpoint.rstrip("/") == normalized
            ),
            None,
        )

    def slot_view_scope_for(receipt_context: Mapping[str, Any]) -> str:
        return str(receipt_context.get("proof_op_manifest_scope", "") or "")

    def build_slot_view_context(
        receipt_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the per-request slot-view context (committed counts only).

        The deterministic decode-step slot view derives everything it needs
        from the receipt's committed counts plus each stage's local op
        template, so the context is just a sentinel that carries the window
        file token forward. Each stage rebuilds its own leaves locally.
        """

        from verallm.mesh.ggml_proof import (
            SLOT_VIEW_SCOPE,
            window_capture_file_token,
        )

        if slot_view_scope_for(receipt_context) != SLOT_VIEW_SCOPE:
            return None
        if proof_trace_root is None:
            raise RuntimeError("slot view requires --proof-trace-dir")
        file_token = str(receipt_context.get("proof_capture_window", "") or "")
        if not file_token:
            started_ns = int(receipt_context.get("inference_started_unix_ns", 0))
            file_token = window_capture_file_token(
                proof_trace_root,
                started_unix_ns=started_ns,
            )
        return {"version": 2, "file_token": str(file_token or "")}

    _slot_view_prewarm_lock = threading.Lock()
    _slot_view_prewarm_started: set[tuple] = set()

    def _maybe_prewarm_slot_view_caches(
        template: list[dict[str, Any]], dedup_key: tuple
    ) -> None:
        """Background-warm the slot-view caches once per template.

        Fires on the first template build of the process (the launch
        canary), so by the time organic traffic arrives every completion
        length up to the ceiling is a prefix hit instead of a one-time
        1.5-1.8 s first-seen-length build. VERATHOS_MESH_SLOT_VIEW_
        PREWARM_TOKENS overrides the ceiling; 0 disables.
        """

        raw = os.environ.get(
            "VERATHOS_MESH_SLOT_VIEW_PREWARM_TOKENS", "4096"
        ).strip()
        try:
            target = int(raw)
        except ValueError:
            target = 4096
        if target <= 0:
            return
        with _slot_view_prewarm_lock:
            if dedup_key in _slot_view_prewarm_started:
                return
            _slot_view_prewarm_started.add(dedup_key)

        def _run() -> None:
            try:
                prewarm_slot_view_caches(
                    template,
                    target_tokens=target,
                    journal=_journal_capture_event,
                )
            except Exception:
                logger.exception("slot view cache prewarm failed")

        threading.Thread(
            target=_run, name="slot-view-prewarm", daemon=True
        ).start()

    def local_slot_view_template(
        receipt_context: dict[str, Any],
        slot_ctx: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return this stage's cached slot-view op template.

        The op template comes from this stage's local v3 manifest (model
        invariant, so a first-graph template from the request's capture window
        suffices). The request layout is derived separately from committed
        prompt/completion counts and ubatch size.
        """

        from verallm.mesh.ggml_proof import (
            find_slot_view_template_for_window,
        )

        if proof_trace_root is None:
            raise RuntimeError("embedded proof is not configured")
        file_token = str((slot_ctx or {}).get("file_token", "") or "")
        if not file_token:
            file_token = str(receipt_context.get("proof_capture_window", "") or "")
        spec = current_mesh_spec()
        cache_key = (
            str(proof_trace_root),
            str(file_token or ""),
            str(spec.model_tensor_manifest_root if spec else ""),
            int(receipt_context.get("stage_index", 0)),
            int(receipt_context.get("layer_start", 0)),
            int(receipt_context.get("layer_end", 0)),
        )
        base_cache_key = (
            str(proof_trace_root),
            "",
            str(spec.model_tensor_manifest_root if spec else ""),
            int(receipt_context.get("stage_index", 0)),
            int(receipt_context.get("layer_start", 0)),
            int(receipt_context.get("layer_end", 0)),
        )
        with slot_view_template_cache_lock:
            cached = slot_view_template_cache.get(cache_key)
            if cached is None and cache_key != base_cache_key:
                cached = slot_view_template_cache.get(base_cache_key)
            if cached:
                return cached
        template = find_slot_view_template_for_window(
            proof_trace_root,
            start_unix_ns=int(receipt_context.get("inference_started_unix_ns", 0))
            if not file_token
            else 0,
            end_unix_ns=int(receipt_context.get("inference_ended_unix_ns", 0))
            if not file_token
            else 0,
            file_token=file_token,
        )
        # The template must cover the stage's WHOLE layer span. A capture
        # window that holds only part of a forward (the launch self-test's
        # probe window, a window opened mid-forward) yields a structurally
        # valid but SHRUNKEN template - observed on a 4-GPU glm serve
        # that pinned a 211-op CUDA3-only template where the full graph has
        # 931 ops over 4 devices. Every draw then lands in the last layers
        # and the verifier's per-layer floor rejects the proof, which is the
        # guard's whole point (a miner must not be able to shrink its own
        # challenge universe). Widen the search to every window before
        # accepting one, and never cache a short template.
        from verallm.mesh.ggml_proof import MIN_PROOF_OPS_PER_LAYER

        stage_layers = max(
            0,
            int(receipt_context.get("layer_end", 0))
            - int(receipt_context.get("layer_start", 0)),
        )
        floor = stage_layers * MIN_PROOF_OPS_PER_LAYER
        if floor and len(template) < floor and file_token:
            widened = find_slot_view_template_for_window(
                proof_trace_root,
                start_unix_ns=0,
                end_unix_ns=0,
                file_token="",
            )
            if len(widened) > len(template):
                logger.info(
                    "slot-view template widened past window %s: %d -> %d ops",
                    file_token,
                    len(template),
                    len(widened),
                )
                template = widened
        if not template:
            raise RuntimeError(
                "no v3 manifest rows for the slot view op template; the "
                "runtime must emit compact-raw-v3 rows"
            )
        if floor and len(template) < floor:
            raise RuntimeError(
                f"slot view template covers {len(template)} ops for "
                f"{stage_layers} layers, below the {floor}-op challenge "
                "floor; no capture window holds a whole forward yet"
            )
        # Every template op must name a tensor of THIS model. A trace
        # directory that ever held another model's manifests would otherwise
        # yield a structurally valid template for the wrong graph, and the
        # failure surfaces far away as "no instance captured" or a weight
        # shape mismatch (observed after a qwen -> glm model swap on one
        # worker). Trace dirs are per-model now; this keeps any residue from
        # being provable.
        if proof_gguf_manifest is not None:
            from verallm.mesh.ggml_proof import _slot_view_op_is_committed_weight

            known = {
                str(item.get("name", ""))
                for item in proof_gguf_manifest.get("tensors", [])
                if isinstance(item, Mapping)
            }
            if known:
                # Only committed-weight rows are checkable. A template also
                # carries scheduler input copies ("<backend>#<tensor>#<n>",
                # e.g. CUDA0#attn_inp_k_rot#3 on a split serve) and other
                # activation GEMMs; those are never provable leaves and never
                # appear in the model manifest, so treating them as foreign
                # rejected an honest template and failed the launch self-test
                # (observed: the mesh sat in error for hours).
                foreign = sorted(
                    {
                        str(op.get("tensor_name", ""))
                        for op in template
                        if _slot_view_op_is_committed_weight(op)
                        and str(op.get("tensor_name", "")) not in known
                    }
                )
                if foreign:
                    raise RuntimeError(
                        "slot view template names tensors that are not in "
                        f"this model's manifest: {foreign[:6]} "
                        f"({len(foreign)} of {len(template)} ops); the trace "
                        "directory holds another model's manifests"
                    )
        device_counts: dict[str, int] = {}
        for op in template:
            key = str(op.get("device", ""))
            device_counts[key] = device_counts.get(key, 0) + 1
        logger.info(
            "slot-view template loaded: %d ops, devices %s, file_token=%r, "
            "window=[%s..%s]",
            len(template),
            device_counts,
            file_token,
            receipt_context.get("inference_started_unix_ns", 0)
            if not file_token
            else "-",
            receipt_context.get("inference_ended_unix_ns", 0)
            if not file_token
            else "-",
        )
        with slot_view_template_cache_lock:
            if len(slot_view_template_cache) > 64:
                slot_view_template_cache.clear()
            slot_view_template_cache[cache_key] = template
            slot_view_template_cache[base_cache_key] = template
        _maybe_prewarm_slot_view_caches(template, base_cache_key)
        return template

    def local_slot_view_leaves(
        receipt_context: dict[str, Any],
        slot_ctx: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Build this stage's deterministic decode-step slot view."""

        from verallm.mesh.ggml_proof import build_slot_view_leaves

        template = local_slot_view_template(receipt_context, slot_ctx)
        # Preserve the legacy materialized-leaf path for tests and any caller
        # that still needs full leaves; organic commitments use the faster
        # root/count helper below.
        entries = []
        for idx, op in enumerate(template):
            from verallm.mesh.ggml_proof import GgmlOpManifestEntry

            entries.append(
                GgmlOpManifestEntry(
                    path=Path("slot-view-template"),
                    created_unix_ns=idx,
                    manifest_index=idx,
                    graph_id=f"slot-view-template-{idx}",
                    op_index=idx,
                    op_type="GGML_OP_MUL_MAT",
                    tensor_name=str(op.get("tensor_name", "")),
                    src0_name=str(op.get("src0_name", "")),
                    src1_name="",
                    dst_name="",
                    src0_shape=(int(op["k_dim"]), int(op["n_dim"]), 1, 1),
                    src1_shape=(int(op["k_dim"]), 1, 1, 1),
                    dst_shape=(int(op["n_dim"]), 1, 1, 1),
                    source_types={
                        str(k): str(v)
                        for k, v in dict(op.get("source_types", {}) or {}).items()
                    },
                    backend=str(op.get("backend", "")),
                    device=str(op.get("device", "")),
                    proof_eligible=True,
                    graph_seq=1,
                    intra_graph_index=int(op["intra_graph_index"]),
                )
            )
        return build_slot_view_leaves(
            manifest_entries=entries,
            completion_token_count=int(receipt_context.get("completion_token_count", 0)),
        )

    def local_slot_view_root_count(
        receipt_context: dict[str, Any],
        slot_ctx: Mapping[str, Any] | None = None,
    ) -> tuple[str, int]:
        from verallm.mesh.ggml_proof import ggml_slot_view_root_from_template

        template = local_slot_view_template(receipt_context, slot_ctx)
        return ggml_slot_view_root_from_template(
            template=template,
            completion_token_count=int(receipt_context.get("completion_token_count", 0)),
        )

    def can_skip_organic_slot_view_capture(spec: MeshSpec | None) -> bool:
        """Return true when organic slot-view commitment can use cached template.

        The skipped work is only C-side metadata emission. Deferred audits still
        run exclusive selected-op replays and produce verifier-checkable witness
        proofs. Remote RPC meshes continue to arm capture until remote template
        readiness is explicit.

        NOTE: skipping capture means the deferred audit must REGENERATE the
        completion to materialize the witness. That is O(generation-length) and
        diverges on MoE models (non-deterministic expert routing), so when
        candidate witness capture is requested (proof_trace_candidates > 0) we
        must NOT skip — arm capture during serve and prove from stored
        candidates. Only skip when no candidate capture was asked for.

        Exception: under the v3 slot-view multi-slot profile the candidate
        request is moot — the CLI zeroes the C-side dump budget for slot-view
        serves, so an armed window stores no candidates anyway and audits use
        teacher-forced probes in their own exclusive window. There the
        candidate short-circuit must not force organic serves back into the
        shared join (see organic_capture_structurally_unwitnessed).
        """

        if proof_trace_candidate_capture and not (
            organic_capture_structurally_unwitnessed()
        ):
            return False
        if not slot_view_required or proof_trace_root is None:
            return False
        if spec is not None:
            remote_members = [
                member
                for member in spec.members
                # rpc members compute in a separate process and need
                # remote arming; in local-stage mode the first member has NO
                # rpc endpoint but its layers run inside the coordinator's
                # llama-server, so it too must be armed over its
                # trace-capture channel (it never sees the request itself).
                if (
                    member.rpc_endpoint
                    or (
                        local_stage_capture
                        and member.layers.end > member.layers.start
                    )
                )
                and member.endpoint != capability.endpoint
            ]
            if remote_members:
                return False
            member = next(
                (item for item in spec.members if item.endpoint == capability.endpoint),
                None,
            )
            stage_index = int(member.stage_index) if member else 0
            layer_start = int(member.layers.start) if member else 0
            layer_end = int(member.layers.end) if member else 0
            model_root = str(spec.model_tensor_manifest_root)
        else:
            stage_index = 0
            layer_start = 0
            layer_end = 0
            model_root = ""
        key = (
            str(proof_trace_root),
            "",
            model_root,
            stage_index,
            layer_start,
            layer_end,
        )
        with slot_view_template_cache_lock:
            return bool(slot_view_template_cache.get(key))

    def organic_serve_skip_trace_capture(
        policy: Mapping[str, Any],
        spec: MeshSpec | None,
        *,
        validator_authenticated: bool,
    ) -> bool:
        """Serve-time capture-skip decision for one serve (any lane).

        Validator-lane serves skip under EXACTLY the same structural-zero
        conditions as organics: their shared join stores nothing either
        (dump budget 0, tail ring off, light leaves synthesized), and their
        audits regenerate every witness in their own exclusive probe
        window. Keeping the join was launch conservatism, and it produced
        a real tail: a 12-minute glm full-context canary held the shared
        window for its whole serve, so a hard audit landing mid-canary
        drained up to the 120s stale purge before probing under a loaded
        engine. Outside the
        structural-zero profile validator-lane serves still join
        unconditionally. Decode-audit configured pools skip only under
        the structural-zero profile; every other guard lives in
        can_skip_organic_slot_view_capture.
        """

        if not policy.get("proof_capture_required"):
            return False
        if validator_authenticated and not (
            organic_capture_structurally_unwitnessed()
        ):
            return False
        if policy.get("decode_audit_configured") and not (
            organic_capture_structurally_unwitnessed()
        ):
            return False
        return can_skip_organic_slot_view_capture(spec)

    def post_proof_request(
        endpoint: str,
        *,
        receipt_context: dict[str, Any],
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        include_proof: bool = True,
        slot_view_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "receipt_context": receipt_context,
            "openai_request": openai_request,
            "openai_response": openai_response,
            "include_proof": bool(include_proof),
        }
        if slot_view_context:
            body["slot_view_context"] = slot_view_context
        return post_json(
            endpoint.rstrip("/") + "/v1/mesh/proof/receipt",
            body,
            timeout=proof_artifact_timeout,
            internal_auth_secret=internal_auth_secret,
        )

    def post_trace_commitment_request(
        endpoint: str,
        *,
        receipt_context: dict[str, Any],
        slot_view_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"receipt_context": receipt_context}
        if slot_view_context:
            body["slot_view_context"] = slot_view_context
        return post_json(
            endpoint.rstrip("/") + "/v1/mesh/proof/commitment",
            body,
            timeout=proof_artifact_timeout,
            internal_auth_secret=internal_auth_secret,
        )

    def post_proof_selection_request(
        endpoint: str,
        *,
        receipt_context: dict[str, Any],
        slot_view_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # serve_n_parallel rides BESIDE the context, never inside it:
        # postcommit replay demands byte-exact receipt_context
        # reproduction, and only the coordinator knows the slot count.
        body: dict[str, Any] = {
            "receipt_context": receipt_context,
            "serve_n_parallel": int(llama_n_parallel),
        }
        if slot_view_context:
            body["slot_view_context"] = slot_view_context
        return post_json(
            endpoint.rstrip("/") + "/v1/mesh/proof/selection",
            body,
            timeout=proof_artifact_timeout,
            internal_auth_secret=internal_auth_secret,
        )

    def trace_commitment_root_from_payloads(
        payloads: list[dict[str, Any]],
    ) -> tuple[str, int, str, int]:
        from verallm.mesh.ggml_proof import (
            mesh_op_manifest_aggregate_root,
            mesh_trace_commitment_aggregate_root,
        )

        entries = []
        manifest_entries = []
        count = 0
        manifest_count = 0
        for payload in payloads:
            root = str(payload.get("trace_commitment_root", ""))
            if root:
                entry = {
                    "stage_index": int(payload.get("stage_index", 0)),
                    "trace_commitment_root": root,
                    "trace_commitment_count": int(payload.get("trace_commitment_count", 0)),
                }
                entries.append(entry)
                count += entry["trace_commitment_count"]
            manifest_root = str(payload.get("op_manifest_root", ""))
            if manifest_root:
                manifest_entry = {
                    "stage_index": int(payload.get("stage_index", 0)),
                    "op_manifest_root": manifest_root,
                    "op_manifest_count": int(payload.get("op_manifest_count", 0)),
                }
                manifest_entries.append(manifest_entry)
                manifest_count += manifest_entry["op_manifest_count"]
        trace_root = mesh_trace_commitment_aggregate_root(entries) if entries else ""
        manifest_root = (
            mesh_op_manifest_aggregate_root(manifest_entries)
            if manifest_entries
            else ""
        )
        return trace_root, count, manifest_root, manifest_count

    def collect_trace_commitments(*args: Any, **kwargs: Any):
        with _timed_section("collect_trace_commitments"):
            return _collect_trace_commitments_inner(*args, **kwargs)

    def _collect_trace_commitments_inner(
        *,
        receipt_context: dict[str, Any],
        spec: MeshSpec | None,
        slot_view_context: dict[str, Any] | None = None,
        pinned_verification_snapshot: Any | None = None,
    ) -> tuple[str, int, str, int]:
        payloads: list[dict[str, Any]] = []
        if proof_url:
            member = member_for_proof_url(spec, proof_url)
            ctx = (
                receipt_context_for_member(
                    receipt_context,
                    member,
                    spec=spec,
                    verification_snapshot=pinned_verification_snapshot,
                )
                if member
                else receipt_context
            )
            payloads.append(
                post_trace_commitment_request(
                    proof_url,
                    receipt_context=ctx,
                    slot_view_context=slot_view_context,
                )
            )
            return trace_commitment_root_from_payloads(payloads)

        if spec is not None:
            local_member = local_member_for_spec(spec)
            jobs = []
            for member in proof_members_for_spec(spec):
                is_local_member = (
                    local_member is not None
                    and member.stage_index == local_member.stage_index
                )
                if not member.proof_endpoint and not (
                    is_local_member and proof_trace_root is not None
                ):
                    raise RuntimeError(
                        "proof coverage is required for every compute stage; "
                        f"stage {member.stage_index} has no proof source"
                    )
                jobs.append(
                    (
                        member,
                        receipt_context_for_member(
                            receipt_context,
                            member,
                            spec=spec,
                            verification_snapshot=pinned_verification_snapshot,
                        ),
                    )
                )
            if jobs:
                with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
                    futures = {}
                    for member, ctx in jobs:
                        if (
                            local_member is not None
                            and member.stage_index == local_member.stage_index
                        ):
                            if proof_trace_root is None:
                                raise RuntimeError(
                                    "proof endpoint is configured for the local member "
                                    "but --proof-trace-dir is missing"
                                )
                            futures[
                                pool.submit(
                                    make_embedded_trace_commitment_payload,
                                    ctx,
                                    slot_view_context,
                                    pinned_spec=spec,
                                )
                            ] = member
                        else:
                            futures[
                                pool.submit(
                                    post_trace_commitment_request,
                                    same_host_loopback(member.proof_endpoint),
                                    receipt_context=ctx,
                                    slot_view_context=slot_view_context,
                                )
                            ] = member
                    ordered_payloads: dict[int, dict[str, Any]] = {}
                    for future in as_completed(futures):
                        member = futures[future]
                        try:
                            ordered_payloads[member.stage_index] = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"trace commitment failed for stage {member.stage_index} "
                                f"at {member.proof_endpoint or member.endpoint}: {exc}"
                            ) from exc
                    payloads.extend(
                        ordered_payloads[index]
                        for index in sorted(ordered_payloads)
                    )
                return trace_commitment_root_from_payloads(payloads)

        if proof_trace_root is not None:
            payloads.append(
                make_embedded_trace_commitment_payload(
                    receipt_context,
                    slot_view_context,
                    pinned_spec=spec,
                )
            )
            return trace_commitment_root_from_payloads(payloads)
        return "", 0, "", 0

    def parse_proof_payloads(
        payloads: list[dict[str, Any]],
        *,
        receipt_context: dict[str, Any],
        proof_required: bool,
        expected_stage_indexes: set[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, bool, float]:
        raw_receipts: list[dict[str, Any]] = []
        proof_payloads: list[dict[str, Any]] = []
        proof_modes: set[str] = set()
        for payload in payloads:
            items = payload.get("proof_receipts", [])
            if not isinstance(items, list):
                raise RuntimeError("proof endpoint returned non-list proof_receipts")
            raw_receipts.extend(items)
            full_payloads = payload.get("proof_payloads", [])
            if full_payloads and not isinstance(full_payloads, list):
                raise RuntimeError("proof endpoint returned non-list proof_payloads")
            proof_payloads.extend(full_payloads)
            mode = str(payload.get("proof_mode") or VERATHOS_GGML_TRACE_PROOF_MODE)
            if items or full_payloads:
                proof_modes.add(mode)
        if proof_required and not raw_receipts:
            raise RuntimeError("proof receipts are required but proof endpoint returned none")
        opaque_stage_receipts = (
            str(receipt_context.get("proof_receipt_format", ""))
            == "opaque_stage_v2"
        )
        receipts: list[LlamaGraphOpReceipt] | list[MeshStageProofReceipt]
        if opaque_stage_receipts:
            receipts = [MeshStageProofReceipt.from_dict(item) for item in raw_receipts]
        else:
            receipts = [LlamaGraphOpReceipt.from_dict(item) for item in raw_receipts]
        covered_stage_indexes = {item.stage_index for item in receipts}
        missing = sorted(expected_stage_indexes - covered_stage_indexes)
        if proof_required and missing:
            raise RuntimeError(
                "missing proof receipts for mesh stage indexes: "
                + ",".join(str(item) for item in missing)
            )
        if len(proof_modes) > 1:
            raise RuntimeError("proof endpoints returned mixed proof modes")
        proof_mode = next(iter(proof_modes), VERATHOS_GGML_TRACE_PROOF_MODE)
        verified = False
        verifier_ms = 0.0
        if raw_receipts:
            if proof_mode != VERATHOS_GGML_TRACE_PROOF_MODE:
                # The any-tier router enforces the light/hard security rule
                # on the signed challenge kind (light only for the organic
                # inline lane, hard for every validator kind) and delegates
                # hard payloads to the full GEMM verifier.
                from verallm.mesh.ggml_proof import (
                    verify_mesh_proof_payloads_any_tier,
                )

                result = verify_mesh_proof_payloads_any_tier(
                    proof_payloads,
                    [item.to_dict() for item in receipts],
                    mesh_receipt=receipt_context,
                    completion_token_ids=[
                        int(item)
                        for item in receipt_context.get(
                            "decode_audit_completion_token_ids", []
                        )
                        or []
                    ]
                    or None,
                    # The audit context is built locally from the verified
                    # reveal (never miner-supplied), so its stamped tier is
                    # the trusted draw result here.
                    postcommit_light_ok=(
                        str(receipt_context.get("proof_audit_tier", ""))
                        == "light"
                    ),
                )
                verifier_ms = result.verifier_ms
                verified = result.verified
                if proof_required and not result.verified:
                    raise RuntimeError("proof verification failed: " + result.message)
            else:
                verified = all(bool(payload.get("verified", False)) for payload in payloads)
        return (
            [item.to_dict() for item in receipts],
            proof_payloads,
            proof_mode,
            verified,
            verifier_ms,
        )

    def _light_tail_capture_enabled() -> bool:
        return os.environ.get(
            "VERATHOS_MESH_LIGHT_TAIL_CAPTURE", "1"
        ).strip().lower() not in ("0", "false", "no")

    def _organic_light_receipt(receipt_context: Mapping[str, Any]) -> bool:
        from verallm.mesh.ggml_proof import ORGANIC_LIGHT_CHALLENGE_KIND

        return bool(
            (
                str(receipt_context.get("proof_challenge_kind", ""))
                == ORGANIC_LIGHT_CHALLENGE_KIND
                and not bool(
                    receipt_context.get("proof_tier_hard_requested", False)
                )
            )
            or str(receipt_context.get("proof_audit_tier", "")) == "light"
        )

    # Tail-ring flushes land within ~flush-idle + write time of the last
    # decoded token; the deadline covers a lagging flush thread without
    # stalling the receipt when the runtime predates the ring.
    _TAIL_WAIT_DEADLINE_S = 0.5
    _TAIL_SCAN_WINDOW_S = 90.0

    def _scan_tail_groups() -> dict[int, list[dict[str, Any]]]:
        """Tail flush groups (flush_ns -> entries) from the local trace dir."""

        groups: dict[int, list[dict[str, Any]]] = {}
        if proof_trace_root is None:
            return groups
        cutoff = time.time() - _TAIL_SCAN_WINDOW_S
        try:
            entries = list(os.scandir(proof_trace_root))
        except OSError:
            return groups
        for entry in entries:
            name = entry.name
            if not name.startswith("trace-") or not name.endswith(".json"):
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            try:
                data = json.loads(Path(entry.path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if not data.get("tail_ring"):
                continue
            flush_ns = int(data.get("tail_flush_ns", 0) or 0)
            if flush_ns <= 0:
                continue
            data["_tail_path"] = entry.path
            groups.setdefault(flush_ns, []).append(data)
        return groups

    def wait_for_light_tail_material(
        receipt_context: dict[str, Any],
        *,
        serve_n_parallel: int = 1,
    ) -> bool:
        """Poll for a serve-time tail flush covering every audit position.

        On success the flush id is pinned into the receipt context so the
        payload maker assembles from exactly that group; on miss the caller
        falls back to the exclusive-window probe path unchanged."""

        if proof_trace_root is None:
            return False
        if int(serve_n_parallel or 1) > 1:
            # Concurrent slots interleave generations through the one
            # process-wide ring, so instance-order position binding is
            # void. Deterministic no (never a 0.5s wait): audit-drawn
            # lights take the certified probe path at --parallel > 1.
            # Carried OUT OF BAND (a call parameter locally, a body field
            # beside receipt_context on the selection wire): postcommit
            # replay demands byte-exact context reproduction, so nothing
            # may be stamped into the context itself.
            return False
        positions = [
            int(item)
            for item in receipt_context.get("decode_audit_positions", []) or []
        ]
        committed_ids = [
            int(item)
            for item in receipt_context.get(
                "decode_audit_completion_token_ids", []
            )
            or []
        ]
        if not positions or not committed_ids:
            return False
        top_k = int(
            receipt_context.get("decode_audit_top_k", decode_audit_top_k)
        )
        started = time.monotonic()
        deadline = started + _TAIL_WAIT_DEADLINE_S
        while True:
            groups = _scan_tail_groups()
            for flush_ns in sorted(groups, reverse=True):
                if tail_group_covers_positions(
                    groups[flush_ns],
                    positions=positions,
                    committed_ids=committed_ids,
                    top_k=top_k,
                ):
                    receipt_context["proof_tail_flush_ns"] = int(flush_ns)
                    _journal_capture_event(
                        f"tail-light ready flush={flush_ns}"
                        f" positions={len(positions)}"
                        f" wait_ms={int((time.monotonic() - started) * 1000)}"
                    )
                    return True
            if time.monotonic() >= deadline:
                # Journal enough to diagnose WHICH side failed post-hoc: no
                # groups at all (flush never fired), a short group (ring
                # split mid-generation), or a full group that rejects every
                # offset (mapping or content mismatch).
                newest_total = 0
                if groups:
                    newest = groups[max(groups)]
                    if newest:
                        newest_total = int(
                            newest[0].get("tail_total", 0) or 0
                        )
                _journal_capture_event(
                    "tail-light miss: no covering flush"
                    f" groups={len(groups)} newest_total={newest_total}"
                    f" committed={len(committed_ids)}"
                    f" positions={sorted(int(p) for p in positions)[:8]}"
                )
                return False
            time.sleep(0.025)

    def maybe_replay_missing_selected_witnesses(
        *,
        receipt_context: dict[str, Any],
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        spec: MeshSpec | None,
        slot_view_context: dict[str, Any] | None = None,
        pinned_verification_snapshot: Any | None = None,
    ) -> None:
        if not backend_url:
            return
        if not receipt_context.get("proof_required"):
            return
        if (
            not receipt_context.get("decode_audit_required")
            and str(receipt_context.get("proof_trace_scope", ""))
            != "op_manifest_challenge_v1"
        ):
            return

        replay_openai_request = dict(openai_request)
        replay_openai_request["stream"] = False
        committed_slot = int(receipt_context.get("proof_slot_id", -1))
        if committed_slot >= 0:
            # Pin the audit replay to the slot the serve committed: at
            # --parallel > 1 the KV layout is slot-dependent, and a replay
            # on a different slot can tie-flip a greedy token, which the
            # strict semantic equality check would then reject on an HONEST
            # serve (observed).
            replay_openai_request["id_slot"] = committed_slot

        def replay_decode_tokens_if_needed() -> None:
            if not receipt_context.get("decode_audit_required"):
                return
            if receipt_context.get("decode_audit_completion_token_ids"):
                return
            replay_request = backend_openai_request(
                replay_openai_request,
                proof_capture_required=True,
                prompt_cache_disabled=True,
                verified_sampler_required=True,
                proof_metadata_required=True,
                sampled_profile=sampled_controls_from_context(receipt_context),
            )
            replay_started_ns = time.time_ns()
            raw_replay_response = post_json(
                f"{backend_url}/v1/chat/completions",
                replay_request,
                timeout=120.0,
            )
            replay_ended_ns = time.time_ns()
            receipt_context["proof_replay_started_unix_ns"] = int(replay_started_ns)
            receipt_context["proof_replay_ended_unix_ns"] = int(replay_ended_ns)
            replay_response, replay_token_ids, replay_token_source = (
                prepare_backend_response_for_receipt(
                    raw_replay_response,
                    proof_capture_required=True,
                    stream=False,
                )
            )
            if not replay_token_source:
                replay_token_ids, replay_token_source = (
                    fetch_completion_token_ids_from_backend(backend_url, replay_response)
                )
            expected_response_hash = semantic_openai_response_hash(openai_response)
            replay_response_hash = semantic_openai_response_hash(replay_response)
            if replay_response_hash != expected_response_hash:
                raise RuntimeError(
                    "decode audit token replay semantic response mismatch"
                )
            bind_decode_audit_completion_tokens(
                receipt_context,
                replay_token_ids,
                replay_token_source,
            )

        from verallm.mesh.ggml_proof import SLOT_VIEW_SCOPE

        slot_view = slot_view_scope_for(receipt_context) == SLOT_VIEW_SCOPE

        selection_jobs: list[tuple[MeshMember | None, dict[str, Any]]] = []
        if proof_url:
            member = member_for_proof_url(spec, proof_url)
            ctx = (
                receipt_context_for_member(
                    receipt_context,
                    member,
                    spec=spec,
                    verification_snapshot=pinned_verification_snapshot,
                )
                if member
                else receipt_context
            )
            try:
                selection_jobs.append(
                    (
                        member,
                        post_proof_selection_request(
                            proof_url,
                            receipt_context=ctx,
                            slot_view_context=slot_view_context,
                        ),
                    )
                )
            except Exception:
                if receipt_context.get("proof_required"):
                    raise
                return
        elif spec is not None:
            local_member = local_member_for_spec(spec)
            for member in proof_members_for_spec(spec):
                is_local_member = (
                    local_member is not None
                    and member.stage_index == local_member.stage_index
                )
                if not member.proof_endpoint and not (
                    is_local_member and proof_trace_root is not None
                ):
                    if receipt_context.get("proof_required"):
                        raise RuntimeError(
                            "proof coverage is required for every compute stage; "
                            f"stage {member.stage_index} has no proof source"
                        )
                    continue
                ctx = receipt_context_for_member(
                    receipt_context,
                    member,
                    spec=spec,
                    verification_snapshot=pinned_verification_snapshot,
                )
                if local_member is not None and member.stage_index == local_member.stage_index:
                    if proof_trace_root is None:
                        continue
                    selection = make_embedded_proof_selection_payload(
                        ctx,
                        slot_view_context,
                        pinned_spec=spec,
                        serve_n_parallel=llama_n_parallel,
                    )
                else:
                    try:
                        selection = post_proof_selection_request(
                            same_host_loopback(member.proof_endpoint),
                            receipt_context=ctx,
                            slot_view_context=slot_view_context,
                        )
                    except Exception:
                        if receipt_context.get("proof_required"):
                            raise
                        return
                selection_jobs.append((member, selection))
        elif proof_trace_root is not None:
            selection_jobs.append(
                (
                    None,
                    make_embedded_proof_selection_payload(
                        receipt_context,
                        slot_view_context,
                        pinned_spec=spec,
                        serve_n_parallel=llama_n_parallel,
                    ),
                )
            )

        local_selected: list[int] = []
        local_selected_ops: list[str] = []
        remote_selected: list[tuple[MeshMember, list[int], list[str]]] = []
        local_member = local_member_for_spec(spec)
        for member, selection in selection_jobs:
            missing = [int(item) for item in selection.get("missing_manifest_indexes", [])]
            ops = [str(item) for item in selection.get("selected_ops", [])]
            if not missing and not ops:
                continue
            if (
                member is None
                or local_member is not None
                and member.stage_index == local_member.stage_index
            ):
                if slot_view:
                    local_selected_ops.extend(ops)
                else:
                    local_selected.extend(missing)
            else:
                remote_selected.append(
                    (member, missing if not slot_view else [], ops if slot_view else [])
                )
        local_selected = sorted(set(local_selected))
        local_selected_ops = dedupe_selected_ops(local_selected_ops)
        remote_selected = [
            (member, sorted(set(indexes)), dedupe_selected_ops(ops))
            for member, indexes, ops in remote_selected
            if indexes or ops
        ]
        if not local_selected and not local_selected_ops and not remote_selected:
            return

        if slot_view and _TIMING_LOG_ENABLED:
            _journal_capture_event(
                "tail-light gate: jobs=%d organic=%s kind=%s tails=%s"
                % (
                    len(selection_jobs or []),
                    _organic_light_receipt(receipt_context),
                    str(receipt_context.get("proof_challenge_kind", "")),
                    [
                        int(s.get("tail_flush_ns", 0) or 0)
                        for _m, s in (selection_jobs or [])
                    ],
                )
            )
        if (
            slot_view
            and selection_jobs
            and _light_tail_capture_enabled()
            and _organic_light_receipt(receipt_context)
            and all(
                int(selection.get("tail_flush_ns", 0) or 0) != 0
                for _member, selection in selection_jobs
            )
        ):
            # Probe-free light tier: every stage's selection advertised a
            # serve-time tail flush covering all its audited positions
            # (only all-audit-leaf selections do), so the exclusive window
            # and teacher-forced probes are skipped entirely. The flush id
            # rides the shared context into each member's collection
            # request, where assembly reads exactly that group. Any stage
            # that missed leaves tail_flush_ns unset and this receipt
            # rides the probe path below unchanged.
            receipt_context["proof_tail_flush_ns"] = max(
                int(selection.get("tail_flush_ns", 0) or 0)
                for _member, selection in selection_jobs
            )
            _journal_capture_event(
                "tail-light adopt flush="
                f"{receipt_context['proof_tail_flush_ns']}"
            )
            return

        replay_seed = replay_seed_for_receipt_context(receipt_context)
        request_id = str(receipt_context.get("request_id", ""))

        def run_replay_once() -> tuple[dict[str, Any], int, int]:
            replay_request = backend_openai_request(
                replay_openai_request,
                proof_capture_required=True,
                prompt_cache_disabled=True,
                verified_sampler_required=bool(
                    receipt_context.get("verified_sampler_required")
                    or receipt_context.get("decode_audit_required")
                ),
                proof_metadata_required=bool(
                    receipt_context.get("proof_metadata_required")
                    or receipt_context.get("decode_audit_required")
                ),
                replay_seed=replay_seed,
                sampled_profile=sampled_controls_from_context(receipt_context),
            )
            if (
                receipt_context.get("decode_audit_required")
                and not receipt_context.get("decode_audit_completion_token_ids")
            ):
                replay_request["logprobs"] = True
                replay_request["top_logprobs"] = 1
            started_ns = time.time_ns()
            raw = post_json(
                f"{backend_url}/v1/chat/completions",
                replay_request,
                timeout=120.0,
            )
            return raw, started_ns, time.time_ns()

        def build_teacher_forced_probe_plan() -> tuple[
            list[int], list[int], list[tuple[int, int]]
        ]:
            """Probe plan for the slot-view replay window.

            Returns (prompt_ids, committed_ids, probes) where probes is the
            deduplicated (prefix_len, n_predict) list in EXECUTION order.
            Hoisted out of the probe runner because the capture arming needs
            the plan too: the window's graph ordinals are pure arithmetic
            over this list, and the C-side dump filter must be armed with
            exactly those ordinals or the probes dump nothing selectable.
            """

            committed_ids = [
                int(item)
                for item in receipt_context.get(
                    "decode_audit_completion_token_ids", []
                )
                or []
            ]
            if receipt_context.get("decode_audit_required") and not committed_ids:
                raise RuntimeError(
                    "decode audit requires committed completion token ids"
                )
            prompt_ids, _prompt_source, _prompt_root = prompt_binding_for_request(
                openai_request,
                proof_metadata_required=True,
            )
            expected_prompt_hash = str(
                receipt_context.get("prompt_token_ids_hash", "") or ""
            )
            if expected_prompt_hash and prompt_token_ids_hash(
                list(prompt_ids)
            ) != expected_prompt_hash:
                raise RuntimeError(
                    "teacher-forced probe prompt ids do not match the receipt"
                )
            positions = sorted(
                {
                    int(item)
                    for item in receipt_context.get("decode_audit_positions", [])
                    or []
                }
            )
            # Ascending prefix order: each probe extends the slot's prompt
            # cache instead of rolling it back, so the whole plan costs one
            # bounded sweep (recurrent contexts replay checkpoint-to-end
            # once; attention contexts only re-eval the eager tail), and
            # the final probe leaves the cache positioned for the next
            # conversation turn.
            final_prefix = max(0, len(committed_ids) - 1) if committed_ids else 0
            ordered = [
                (prefix, 2)
                for prefix in sorted({*positions, final_prefix})
                if 0 <= prefix <= final_prefix
            ] or [(final_prefix, 2)]
            return list(prompt_ids), committed_ids, ordered

        def probe_window_selected_ops(
            ops: list[str],
            probe_plan: tuple[list[int], list[int], list[tuple[int, int]]],
        ) -> list[str]:
            """Rewrite selected-op arming pairs for the probe window.

            The selection payload's "graph_ord:intra" pairs carry the
            COMMITTED serve's ordinals, but the exclusive probe window runs
            a different, deterministic graph sequence (the C-side counter
            resets when the capture token changes): per probe, its prefill
            ubatch chunks then one decode graph. Arming committed ordinals
            here can only match by coincidence, and whether the fallback
            probe ordinals survived the emission budget varied with the
            beacon-derived selection (observed: the LM-head instance
            capture was intermittent across identical runs). Keep the
            payload's intras, replace the ordinals with the probe window's
            own: the last prefill graph (the audited LM-head row) and the
            decode graph (single-row instances for every deep op) of every
            probe, plus a growing ordinal slack for llama.cpp graph
            fusion/reuse drift.
            """

            from verallm.mesh.ggml_proof import (
                SLOT_VIEW_SELECTED_OP_LIMIT,
                SLOT_VIEW_SELECTED_OP_TOKEN_BUDGET,
                solo_prefill_graph_count,
            )

            # Name-armed entries (n:<weight name>) are ordinal-free, so the
            # probe-window rewrite must carry them through verbatim: they are
            # the arming that still matches when the probe graph's intras
            # drift from the template (glm-dsa length-dependent op streams).
            name_ops: list[str] = []
            for op in ops:
                if op.startswith("n:") and op not in name_ops:
                    name_ops.append(op)
            intras = sorted(
                {
                    int(op.split(":", 1)[1])
                    for op in ops
                    if not op.startswith("n:")
                    and ":" in op
                    and op.split(":", 1)[1].lstrip("-").isdigit()
                }
            )
            prompt_ids, _committed_ids, probes = probe_plan
            n_ubatch = int(
                receipt_context.get("proof_runtime_ubatch_size", 0) or 0
            )
            if not intras or not probes or n_ubatch <= 0:
                if intras and probes:
                    logger.info(
                        "probe-window arming fallback: no runtime ubatch size "
                        "in the receipt context; keeping committed ordinals"
                    )
                return list(ops)
            bases: list[tuple[int, int]] = []
            cursor = 0
            for prefix_len, _n_predict in probes:
                prefix_total = len(prompt_ids) + max(0, int(prefix_len))
                chunks = solo_prefill_graph_count(
                    prompt_token_count=max(1, prefix_total),
                    n_ubatch=n_ubatch,
                )
                last_prefill = cursor + int(chunks)
                decode = last_prefill + 1
                bases.append((last_prefill, decode))
                cursor = decode
            exact: list[int] = []
            for last_prefill, decode in bases:
                exact.extend((last_prefill, decode))
            slack: list[int] = []
            for index, (last_prefill, decode) in enumerate(bases):
                for ordinal in range(
                    last_prefill - 1 - index, decode + 2 + index
                ):
                    if ordinal >= 1 and ordinal not in exact:
                        slack.append(ordinal)
            candidates: list[int] = []
            seen_ordinals: set[int] = set()
            for ordinal in [*exact, *slack]:
                if ordinal not in seen_ordinals:
                    seen_ordinals.add(ordinal)
                    candidates.append(ordinal)
            name_chars = sum(len(op) + 1 for op in name_ops)
            ordinal_budget = max(
                1, int(SLOT_VIEW_SELECTED_OP_TOKEN_BUDGET) - name_chars
            )
            ordinal_limit = max(
                1, int(SLOT_VIEW_SELECTED_OP_LIMIT) - len(name_ops)
            )
            selected_ops: list[str] = []
            chars = 0
            # Ord-agnostic wildcards first: on split meshes every pass fans
            # out into many member subgraphs (rpc view-boundary splits,
            # hyper-connection islands), so the computed probe ordinals can
            # sit tens of graphs away from the real decode instance. The C
            # side dumps wildcard matches only for small-row instances.
            for intra in intras:
                op = f"*:{intra}"
                extra = len(op) + (1 if selected_ops else 0)
                selected_ops.append(op)
                chars += extra
            for rank in range(len(candidates)):
                for intra in intras:
                    op = f"{candidates[rank]}:{intra}"
                    extra = len(op) + (1 if selected_ops else 0)
                    if selected_ops and chars + extra > ordinal_budget:
                        return [*selected_ops, *name_ops]
                    selected_ops.append(op)
                    chars += extra
                    if len(selected_ops) >= ordinal_limit:
                        return [*selected_ops, *name_ops]
            return [*selected_ops, *name_ops]

        def run_teacher_forced_probes(
            probe_plan: tuple[list[int], list[int], list[tuple[int, int]]],
            *,
            timing: dict[str, Any] | None = None,
        ) -> None:
            # Slot-view audits regenerate witnesses WITHOUT re-sampling. A
            # sampled solo re-serve can tie-flip a greedy token against the
            # co-batched serve (slot-dependent KV layout, eager-vs-graph
            # kernel numerics), which the strict semantic equality check
            # would then reject on an HONEST serve (observed). Teacher
            # forcing prefills exact committed prefixes instead: nothing is
            # sampled, so nothing can diverge, and per-op numeric jitter
            # only moves logit margins, which the top-k acceptance absorbs.

            # Probe shape: prefill(prompt + committed[:p]) with n_predict=2.
            # Both smaller values fail on real llama.cpp (observed):
            # n_predict=0 never evaluates the output head at all, and
            # n_predict=1 samples its one token straight from the prefill's
            # last row, so NO decode graph runs and the single-row instances
            # the slot-view leaves need never exist. With 2 there is exactly
            # one decode graph, giving every selected op a single-row
            # instance; the sampled tokens themselves are discarded.
            # Instance choice is by committed-token content, so whether the
            # audited row lands in the prefill tail or the decode step does
            # not matter.
            # Probes reuse the committed slot's prompt cache: the audited
            # prefixes are prefixes of the state that slot just served, so
            # replay cost is bounded by the distance to the nearest cache
            # checkpoint instead of the whole conversation. The minimum
            # eager prompt tail in the patched llama-server guarantees the
            # re-eval always produces capture witnesses, and every witness
            # the audit needs is a single-row instance the wildcard arming
            # dumps regardless of graph ordinals.
            prompt_ids, committed_ids, probes = probe_plan
            probe_cache = os.environ.get(
                "VERATHOS_MESH_PROBE_PROMPT_CACHE", "1"
            ).strip().lower() not in ("0", "false", "no")
            probe_slot = int(receipt_context.get("proof_slot_id", -1))
            probe_slot = _maybe_restore_slot_state_for_probes(
                receipt_context,
                probe_slot=probe_slot,
                probe_cache=probe_cache,
                timing=timing,
            )
            started_ns = time.time_ns()
            for prefix_len, n_predict in probes:
                probe_started = time.monotonic()
                prefix = list(prompt_ids) + committed_ids[: int(prefix_len)]
                body: dict[str, Any] = {
                    "prompt": prefix,
                    "n_predict": int(n_predict),
                    # A teacher-forced probe exists to RUN the ops, not to
                    # sample content. When the token greedily sampled off the
                    # prefill logits is a stop token, llama-server ends the
                    # request before the decode forward ever executes, the
                    # window holds no single-row instances, and every leaf
                    # witness starves (observed on glm-5.2 audits whose
                    # audited position sat at the completion tail). Decoding
                    # past EOS costs one forward and guarantees the witness.
                    "ignore_eos": True,
                    "temperature": 0,
                    "top_k": 1,
                    "cache_prompt": bool(probe_cache),
                }
                if probe_cache and probe_slot >= 0:
                    body["id_slot"] = probe_slot
                raw = post_json(
                    f"{backend_url}/completion",
                    body,
                    timeout=300.0,
                )
                timings = raw.get("timings", {}) if isinstance(raw, dict) else {}
                if timing is not None:
                    timing.setdefault("probe_ms", []).append(
                        int((time.monotonic() - probe_started) * 1000)
                    )
                _journal_capture_event(
                    f"probe prefix_total={len(prefix)} n_predict={int(n_predict)}"
                    f" cache={int(probe_cache)}"
                    f" evaluated={int(timings.get('prompt_n', -1) or -1)}"
                )
            _journal_capture_event("probes-done")
            # Re-runs extend the span forward; the first run's start stands
            # (the receipt's replay span covers every in-window probe pass).
            if not receipt_context.get("proof_replay_started_unix_ns"):
                receipt_context["proof_replay_started_unix_ns"] = int(started_ns)
            receipt_context["proof_replay_ended_unix_ns"] = time.time_ns()

        started_remote: list[MeshMember] = []
        local_replay_required = bool(local_selected or local_selected_ops)
        if local_replay_required and proof_trace_enable_path is None:
            raise RuntimeError("selected local replay requires trace capture")
        probe_plan: tuple[list[int], list[int], list[tuple[int, int]]] | None = (
            None
        )
        local_arm_ops = local_selected_ops
        if slot_view:
            probe_plan = build_teacher_forced_probe_plan()
            local_arm_ops = probe_window_selected_ops(
                local_selected_ops, probe_plan
            )
            logger.info(
                "slot-view probe window: probes=%s local_arm_ops=%d/%d "
                "remote_members=%d",
                [prefix for prefix, _ in probe_plan[2]],
                len(local_arm_ops),
                len(local_selected_ops),
                len(remote_selected),
            )
        def _local_probe_window_needs() -> tuple[set[str], list[dict[str, Any]]]:
            """Leaf names + decode-audit needs this coordinator's window must hold.

            Mirrors the payload build's acceptance exactly: every LOCAL
            selection leaf needs a bounded eager-tail instance, and every
            decode-audit leaf additionally needs a strict single-row
            instance whose logits carry the committed token (argmax or
            committed top-k). Remote members' dumps land on their own trace
            dirs and cannot be scanned from here; local coverage is the
            collision signal (one continuous-batching schedule spans all
            stages, and the LM-head leaf is local in every production
            topology).
            """

            leaf_names: set[str] = set()
            audit_needs: list[dict[str, Any]] = []
            committed_ids = [
                int(item)
                for item in receipt_context.get(
                    "decode_audit_completion_token_ids", []
                )
                or []
            ]
            audit_top_k = max(
                1,
                int(
                    receipt_context.get("decode_audit_top_k", decode_audit_top_k)
                    or decode_audit_top_k
                ),
            )
            for member, selection in selection_jobs:
                is_local = member is None or (
                    local_member is not None
                    and member.stage_index == local_member.stage_index
                )
                if not is_local:
                    continue
                for item in selection.get("selected", []) or []:
                    leaf = item.get("leaf") or {}
                    name = str(leaf.get("tensor_name", "") or "")
                    if not name:
                        continue
                    leaf_names.add(name)
                    positions = [
                        int(position)
                        for position in item.get("decode_audit_positions", [])
                        or []
                    ]
                    if positions and committed_ids:
                        position = min(positions)
                        if 0 <= position < len(committed_ids):
                            audit_needs.append(
                                {
                                    "leaf": name,
                                    "position": position,
                                    "expected_token": int(
                                        committed_ids[position]
                                    ),
                                    "top_k": audit_top_k,
                                }
                            )
            return leaf_names, audit_needs

        def _probe_window_witnesses_missing(
            armed_unix_ns: int,
            leaf_names: set[str],
            audit_needs: list[dict[str, Any]],
        ) -> str:
            """Return '' when the armed window already holds every needed
            instance, else a description of the first miss (the same
            condition the payload build would later fail on)."""

            if proof_trace_root is None or not (leaf_names or audit_needs):
                return ""
            from verallm.mesh.ggml_proof import find_traces_for_window

            traces = find_traces_for_window(
                proof_trace_root,
                start_unix_ns=int(armed_unix_ns),
                end_unix_ns=0,
            )
            by_name: dict[str, list[Any]] = {}
            for trace in traces:
                by_name.setdefault(str(trace.tensor_name), []).append(trace)
            for name in sorted(leaf_names):
                pool = [
                    trace
                    for trace in by_name.get(name, [])
                    if len(trace.src1_shape) > 1
                    and 1
                    <= int(trace.src1_shape[1])
                    <= SLOT_VIEW_PROBE_MAX_ROWS
                ]
                if not pool:
                    return (
                        "no bounded probe instance for "
                        f"{name} (max rows {SLOT_VIEW_PROBE_MAX_ROWS})"
                    )
            import numpy as np

            for need in audit_needs:
                hit = False
                for trace in by_name.get(str(need["leaf"]), []):
                    if not (
                        len(trace.src1_shape) > 1
                        and 1
                        <= int(trace.src1_shape[1])
                        <= SLOT_VIEW_PROBE_MAX_ROWS
                    ):
                        continue
                    try:
                        rows = np.fromfile(
                            str(trace.path).replace(".json", "-dst.f32"),
                            dtype=np.float32,
                        )
                    except Exception:
                        continue
                    vocab = int(trace.dst_shape[0]) if trace.dst_shape else 0
                    if vocab <= 0 or rows.size % vocab:
                        continue
                    matrix = rows.reshape(-1, vocab)
                    if matrix.shape[0] != 1:
                        continue
                    row = matrix[0]
                    expected = int(need["expected_token"])
                    if int(np.argmax(row)) == expected:
                        hit = True
                        break
                    k = min(int(need["top_k"]), vocab)
                    if expected in {
                        int(idx) for idx in np.argpartition(row, -k)[-k:]
                    }:
                        hit = True
                        break
                if not hit:
                    return (
                        "no single-row instance carrying committed token "
                        f"{need['expected_token']} at position "
                        f"{need['position']} for {need['leaf']}"
                    )
            return ""

        probe_rerun_budget = max(
            0, int(os.environ.get("VERATHOS_MESH_PROBE_RERUNS", "2") or 2)
        )
        window_timing: dict[str, Any] = {}
        telemetry = receipt_context.setdefault(
            "_mesh_audit_telemetry",
            {"windows": 0, "reprobes": 0, "escalations": 0},
        )
        window_started = time.monotonic()
        replay_anchor_rows = anchor_rows_to_arm(receipt_context)
        with _suspended_remote_shared_hold(spec=spec), exclusive_capture(
            request_id,
            selected_manifest_indexes=local_selected,
            selected_ops=local_arm_ops,
            selected_anchor_rows=replay_anchor_rows,
            require_local_trace=local_replay_required,
            timing=window_timing,
        ) as replay_window_token:
            try:
                if spec is not None and remote_selected:
                    # One window open pays every stage's arming ONCE and in
                    # parallel; the old serial 10x1.0s retry ladder could
                    # spend tens of exclusive seconds on a flaky member
                    # while organics queued behind the window. Fail fast to
                    # the outer draw instead.
                    arm_started = time.monotonic()

                    def _arm_remote_exclusive(
                        job: tuple[MeshMember, list[int], list[str]],
                    ) -> MeshMember:
                        member, indexes, ops = job
                        arm_ops = (
                            probe_window_selected_ops(ops, probe_plan)
                            if slot_view and probe_plan is not None
                            else ops
                        )
                        set_remote_trace_capture(
                            same_host_loopback(member.endpoint),
                            spec=spec,
                            request_id=request_id,
                            enabled=True,
                            mode="exclusive",
                            window_token=replay_window_token.split("|", 1)[0],
                            selected_manifest_indexes=indexes,
                            selected_ops=arm_ops,
                            # Commitments never travel: the remote worker
                            # derives its own anchor rows from this beacon
                            # against its own local streams.
                            anchor_beacon=str(
                                receipt_context.get("proof_beacon", "")
                            ),
                            retries=2,
                            retry_delay=0.5,
                        )
                        return member

                    arm_errors: list[Exception] = []
                    with ThreadPoolExecutor(
                        max_workers=min(8, len(remote_selected))
                    ) as arm_pool:
                        futures = [
                            arm_pool.submit(_arm_remote_exclusive, job)
                            for job in remote_selected
                        ]
                        for future in futures:
                            try:
                                started_remote.append(future.result())
                            except Exception as exc:
                                arm_errors.append(exc)
                    window_timing["arm_remote_ms"] = int(
                        (time.monotonic() - arm_started) * 1000
                    )
                    if arm_errors:
                        raise arm_errors[0]
                if slot_view:
                    assert probe_plan is not None
                    run_teacher_forced_probes(
                        probe_plan, timing=window_timing
                    )
                    # In-window witness verification: a probe whose
                    # single-row instance was lost to a batching collision
                    # is cheapest to fix RIGHT NOW - the slot cache is hot,
                    # so a re-probe costs one bounded eager tail instead of
                    # a whole fresh window (drain + arming) later.
                    leaf_names, audit_needs = _local_probe_window_needs()
                    armed_ns = int(window_timing.get("armed_unix_ns", 0) or 0)
                    missing = _probe_window_witnesses_missing(
                        armed_ns, leaf_names, audit_needs
                    )
                    reruns = 0
                    while missing and reruns < probe_rerun_budget:
                        reruns += 1
                        telemetry["reprobes"] += 1
                        _journal_capture_event(
                            f"probe-rerun {reruns}/{probe_rerun_budget}: "
                            f"{missing}"
                        )
                        run_teacher_forced_probes(
                            probe_plan, timing=window_timing
                        )
                        missing = _probe_window_witnesses_missing(
                            armed_ns, leaf_names, audit_needs
                        )
                    window_timing["reprobes"] = reruns
                    if missing:
                        # Bounded strict-quiesce escalation: gate NEW
                        # capture-skipping organics (byte-identical busy on
                        # timeout), wait out the in-flight ones, then one
                        # final solo probe pass. The common case never
                        # reaches this; sustained collision under load does,
                        # once, instead of six drain-paying windows.
                        telemetry["escalations"] += 1
                        window_timing["escalations"] = 1
                        logger.info(
                            "mesh-audit escalation request=%s: %s",
                            request_id[:8],
                            missing,
                        )
                        _journal_capture_event(
                            f"strict-quiesce start: {missing}"
                        )
                        strict_quiesce_event.set()
                        try:
                            drain_deadline = (
                                time.monotonic() + strict_quiesce_wait_s
                            )
                            with organic_inflight_cond:
                                while organic_inflight_state["count"] > 0:
                                    remaining = (
                                        drain_deadline - time.monotonic()
                                    )
                                    if remaining <= 0:
                                        break
                                    organic_inflight_cond.wait(
                                        timeout=min(remaining, 5.0)
                                    )
                            run_teacher_forced_probes(
                                probe_plan, timing=window_timing
                            )
                            missing = _probe_window_witnesses_missing(
                                armed_ns, leaf_names, audit_needs
                            )
                        finally:
                            strict_quiesce_event.clear()
                            _journal_capture_event(
                                "strict-quiesce end"
                                + (f" still-missing: {missing}" if missing else "")
                            )
                        if missing:
                            logger.info(
                                "mesh-audit window request=%s still missing "
                                "witnesses after strict quiesce: %s",
                                request_id[:8],
                                missing,
                            )
                else:
                    if slot_view_required:
                        # Production profile assertion: every audited serve
                        # on a slot-view server takes teacher-forced probes.
                        # The legacy full re-serve can never semantically
                        # match a graph-served original (eager re-serve
                        # re-picks kernel geometry; near-tie greedy tokens
                        # flip), so reaching it here is a routing bug, not
                        # a fallback. Test fixtures with legacy manifest
                        # formats keep it (slot_view_required is false
                        # there).
                        logger.warning(
                            "legacy full-replay path reached under the "
                            "slot-view production profile; refusing "
                            "(request=%s scope=%s)",
                            request_id[:8],
                            str(receipt_context.get("proof_trace_scope", "")),
                        )
                        raise RuntimeError(
                            "legacy replay path is unreachable under the "
                            "slot-view production profile"
                        )
                    raw_replay_response, replay_started_ns, replay_ended_ns = (
                        run_replay_once()
                    )
                    receipt_context["proof_replay_started_unix_ns"] = int(
                        replay_started_ns
                    )
                    receipt_context["proof_replay_ended_unix_ns"] = int(
                        replay_ended_ns
                    )
                receipt_context["proof_replay_window_token"] = (
                    replay_window_token.split("|", 1)[0]
                )
            finally:
                if spec is not None and started_remote:
                    for member in started_remote:
                        try:
                            set_remote_trace_capture(
                                same_host_loopback(member.endpoint),
                                spec=spec,
                                request_id=request_id,
                                enabled=False,
                                mode="exclusive",
                            )
                        except Exception:
                            pass
                telemetry["windows"] += 1
                with organic_inflight_cond:
                    inflight_now = int(organic_inflight_state["count"])
                logger.info(
                    "mesh-audit-window request=%s drain_ms=%d "
                    "arm_remote_ms=%d restore=%s restore_ms=%d probes=%d "
                    "probe_ms=%s reprobes=%d escalations=%d inflight=%d "
                    "window_ms=%d",
                    request_id[:8],
                    int(window_timing.get("drain_ms", 0) or 0),
                    int(window_timing.get("arm_remote_ms", 0) or 0),
                    str(window_timing.get("restore", "off")),
                    int(window_timing.get("restore_ms", 0) or 0),
                    len(window_timing.get("probe_ms", []) or []),
                    [
                        int(item)
                        for item in window_timing.get("probe_ms", []) or []
                    ],
                    int(window_timing.get("reprobes", 0) or 0),
                    int(window_timing.get("escalations", 0) or 0),
                    inflight_now,
                    int((time.monotonic() - window_started) * 1000),
                )

        if slot_view:
            # Teacher-forced probes never sample, so there is no replay
            # response to compare; the binding is the committed prompt and
            # completion token id hashes the probes were derived from.
            return
        replay_response, replay_token_ids, replay_token_source = (
            prepare_backend_response_for_receipt(
                raw_replay_response,
                proof_capture_required=True,
                stream=False,
            )
        )
        if not receipt_context.get("decode_audit_required") and not replay_token_source:
            replay_token_ids, replay_token_source = (
                fetch_completion_token_ids_from_backend(backend_url, replay_response)
            )
        expected_response_hash = semantic_openai_response_hash(openai_response)
        replay_response_hash = semantic_openai_response_hash(replay_response)
        if replay_response_hash != expected_response_hash:
            raise RuntimeError("selected mesh witness replay semantic response mismatch")
        bind_decode_audit_completion_tokens(
            receipt_context,
            replay_token_ids,
            replay_token_source,
        )
        if str(receipt_context.get("completion_token_source", "")):
            if not replay_token_source:
                raise RuntimeError("selected mesh witness replay missing token ids")
            if completion_token_ids_hash(replay_token_ids) != str(
                receipt_context.get("completion_token_ids_hash", "")
            ):
                raise RuntimeError("selected mesh witness replay token hash mismatch")

    def collect_proof_receipts(*args: Any, **kwargs: Any):
        with _timed_section("collect_proof_receipts"):
            started = time.monotonic()
            try:
                result = _collect_proof_receipts_inner(*args, **kwargs)
            except RuntimeError as exc:
                if "no single-row decode instance captured" not in str(exc):
                    raise
                # Concurrency collision, retryable: the exclusive probe
                # window replays its decode while OTHER slots keep
                # decoding, and continuous batching can merge them into
                # one multi-column graph, so the probe's single-row
                # instance never lands (live at 12-way load: the same-op
                # instance arrived shaped (5120, 4)). A re-armed window
                # rarely collides twice, so one retry turns a hard
                # failure into a served, verified reply. The principled
                # follow-up is column extraction from batched probe
                # instances via the slot-view batch slices.
                _journal_capture_event("probe-batch-collision retry")
                result = _collect_proof_receipts_inner(*args, **kwargs)
            try:
                if result and result[0]:
                    _journal_capture_event(
                        "collect-receipts"
                        f" ms={int((time.monotonic() - started) * 1000)}"
                        f" payloads={len(result[0])}"
                    )
            except Exception:
                pass
            return result

    def _collect_proof_receipts_inner(
        *,
        receipt_context: dict[str, Any],
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        proof_required: bool,
        spec: MeshSpec | None,
        slot_view_context: dict[str, Any] | None = None,
        pinned_verification_snapshot: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, bool, float]:
        if not proof_required:
            return [], [], "", False, 0.0
        from verallm.mesh.ggml_proof import SLOT_VIEW_SCOPE

        if (
            slot_view_context is None
            and slot_view_scope_for(receipt_context) == SLOT_VIEW_SCOPE
            and backend_url
        ):
            # Deferred audits arrive with only the receipt; rebuild the
            # request's batch slices from the coordinator's trace dir.
            slot_view_context = build_slot_view_context(receipt_context)
        maybe_replay_missing_selected_witnesses(
            receipt_context=receipt_context,
            openai_request=openai_request,
            openai_response=openai_response,
            spec=spec,
            slot_view_context=slot_view_context,
            pinned_verification_snapshot=pinned_verification_snapshot,
        )
        payloads: list[dict[str, Any]] = []
        expected_stage_indexes: set[int] = set()
        if proof_url:
            member = member_for_proof_url(spec, proof_url)
            ctx = (
                receipt_context_for_member(
                    receipt_context,
                    member,
                    spec=spec,
                    verification_snapshot=pinned_verification_snapshot,
                )
                if member
                else receipt_context
            )
            if member is not None:
                expected_stage_indexes.add(member.stage_index)
            payloads.append(
                post_proof_request(
                    proof_url,
                    receipt_context=ctx,
                    openai_request=openai_request,
                    openai_response=openai_response,
                    slot_view_context=slot_view_context,
                )
            )
            return parse_proof_payloads(
                payloads,
                receipt_context=receipt_context,
                proof_required=proof_required,
                expected_stage_indexes=expected_stage_indexes,
            )

        if spec is not None:
            local_member = local_member_for_spec(spec)
            proof_jobs = []
            for member in proof_members_for_spec(spec):
                expected_stage_indexes.add(member.stage_index)
                is_local_member = (
                    local_member is not None
                    and member.stage_index == local_member.stage_index
                )
                if not member.proof_endpoint and not (
                    is_local_member and proof_trace_root is not None
                ):
                    raise RuntimeError(
                        "proof coverage is required for every compute stage; "
                        f"stage {member.stage_index} has no proof source"
                    )
                ctx = receipt_context_for_member(
                    receipt_context,
                    member,
                    spec=spec,
                    verification_snapshot=pinned_verification_snapshot,
                )
                proof_jobs.append((member, ctx))
            if proof_jobs:
                with ThreadPoolExecutor(max_workers=min(8, len(proof_jobs))) as pool:
                    futures = {}
                    for member, ctx in proof_jobs:
                        if (
                            local_member is not None
                            and member.stage_index == local_member.stage_index
                        ):
                            if proof_trace_root is None:
                                if proof_required:
                                    raise RuntimeError(
                                        "proof endpoint is configured for the local member "
                                        "but --proof-trace-dir is missing"
                                    )
                                continue
                            futures[
                                pool.submit(
                                    make_embedded_proof_payload,
                                    ctx,
                                    include_proof=True,
                                    proof_required=proof_required,
                                    openai_request=openai_request,
                                    openai_response=openai_response,
                                    slot_view_context=slot_view_context,
                                    pinned_spec=spec,
                                )
                            ] = member
                        else:
                            futures[
                                pool.submit(
                                    post_proof_request,
                                    same_host_loopback(member.proof_endpoint),
                                    receipt_context=ctx,
                                    openai_request=openai_request,
                                    openai_response=openai_response,
                                    slot_view_context=slot_view_context,
                                )
                            ] = member
                    ordered_payloads: dict[int, dict[str, Any]] = {}
                    for future in as_completed(futures):
                        member = futures[future]
                        try:
                            ordered_payloads[member.stage_index] = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"proof endpoint failed for stage {member.stage_index} "
                                f"at {member.proof_endpoint or member.endpoint}: {exc}"
                            ) from exc
                    payloads.extend(
                        ordered_payloads[index]
                        for index in sorted(ordered_payloads)
                    )
                if payloads:
                    return parse_proof_payloads(
                        payloads,
                        receipt_context=receipt_context,
                        proof_required=proof_required,
                        expected_stage_indexes=expected_stage_indexes,
                    )

        if proof_trace_root is None:
            if proof_required:
                raise RuntimeError(
                    "proof receipts are required but no proof endpoint is configured; "
                    "pass --proof-url or --proof-trace-dir"
                )
            return [], [], "", False, 0.0

        payloads.append(
            make_embedded_proof_payload(
                receipt_context,
                include_proof=True,
                proof_required=proof_required,
                openai_request=openai_request,
                openai_response=openai_response,
                pinned_spec=spec,
            )
        )
        return parse_proof_payloads(
            payloads,
            receipt_context=receipt_context,
            proof_required=proof_required,
            expected_stage_indexes=expected_stage_indexes,
        )

    def rerun_starved_capture_serve(
        receipt_body: dict[str, Any],
        *,
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        spec: MeshSpec | None,
    ) -> tuple[str, int]:
        """Re-serve once with the prompt cache disabled after a starved capture.

        A FULL llama-server prompt-cache hit skips the eager prefill entirely,
        and CUDA-graph decode replay runs no per-op hooks, so an exactly
        repeated prompt yields zero capture witnesses. Ordinary multi-turn
        traffic never hits this — each new message appends tokens that prefill
        eagerly — so the cache stays on for every serve and this fallback pays
        the full re-prefill only for exact duplicates. The trigger is the
        observable capture outcome alone, identical for canary and organic
        traffic; keying it on anything request-derived would hand miners a
        canary distinguisher.

        Returns the fallback capture window token and the serving slot id,
        or ("", -1) when no fallback serve could run.
        """

        if not backend_url:
            return "", -1
        retry_openai_request = dict(openai_request)
        retry_openai_request["stream"] = False
        retry_request = backend_openai_request(
            retry_openai_request,
            proof_capture_required=True,
            verified_sampler_required=bool(
                receipt_body.get("verified_sampler_required")
            ),
            proof_metadata_required=bool(
                receipt_body.get("proof_metadata_required")
            ),
            replay_seed=replay_seed_for_receipt_context(receipt_body),
            prompt_cache_disabled=True,
            sampled_profile=sampled_controls_from_context(receipt_body),
        )
        capture_info: dict[str, Any] = {}
        raw_retry = forward_to_backend(
            retry_request,
            spec=spec,
            request_id=str(receipt_body.get("request_id", "")),
            proof_capture_required=True,
            capture_info=capture_info,
        )
        retry_response, _, _ = prepare_backend_response_for_receipt(
            raw_retry,
            proof_capture_required=True,
            stream=False,
        )
        # The deterministic sampler must reproduce the committed completion
        # byte-for-byte; anything else means the receipt would commit a
        # response the fallback witnesses do not cover.
        if semantic_openai_response_hash(retry_response) != (
            semantic_openai_response_hash(openai_response)
        ):
            raise RuntimeError(
                "prompt-cache fallback serve semantic response mismatch"
            )
        receipt_body["proof_capture_fallback"] = "prompt_cache_disabled_v1"
        return (
            str(capture_info.get("window_token", "") or ""),
            slot_id_from_backend_response(raw_retry),
        )

    def receipt_for(
        *,
        request_id: str,
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        prompt_token_ids: list[int] | None = None,
        prompt_token_source: str = "",
        prompt_template_root: str = "",
        completion_token_ids: list[int] | None = None,
        completion_token_source: str = "",
        stage_index: int,
        layer_start: int,
        layer_end: int,
        spec: MeshSpec | None,
        proof_required: bool,
        proof_policy: dict[str, Any] | None = None,
        inference_started_unix_ns: int = 0,
        inference_ended_unix_ns: int = 0,
        slot_id: int = -1,
        capture_window_token: str = "",
        pinned_verification_snapshot: Any | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with _timed_section("receipt_for_total"):
            return _receipt_for_inner(
                request_id=request_id,
                openai_request=openai_request,
                openai_response=openai_response,
                prompt_token_ids=prompt_token_ids,
                prompt_token_source=prompt_token_source,
                prompt_template_root=prompt_template_root,
                completion_token_ids=completion_token_ids,
                completion_token_source=completion_token_source,
                stage_index=stage_index,
                layer_start=layer_start,
                layer_end=layer_end,
                spec=spec,
                proof_required=proof_required,
                proof_policy=proof_policy,
                inference_started_unix_ns=inference_started_unix_ns,
                inference_ended_unix_ns=inference_ended_unix_ns,
                slot_id=slot_id,
                capture_window_token=capture_window_token,
                pinned_verification_snapshot=pinned_verification_snapshot,
            )

    def _receipt_for_inner(
        *,
        request_id: str,
        openai_request: dict[str, Any],
        openai_response: dict[str, Any],
        prompt_token_ids: list[int] | None = None,
        prompt_token_source: str = "",
        prompt_template_root: str = "",
        completion_token_ids: list[int] | None = None,
        completion_token_source: str = "",
        stage_index: int,
        layer_start: int,
        layer_end: int,
        spec: MeshSpec | None,
        proof_required: bool,
        proof_policy: dict[str, Any] | None = None,
        inference_started_unix_ns: int = 0,
        inference_ended_unix_ns: int = 0,
        slot_id: int = -1,
        capture_window_token: str = "",
        pinned_verification_snapshot: Any | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # Restate usage from the token ids the proof will bind BEFORE the
        # response is serialized into response_hash. Engine telemetry
        # under-reports prompt tokens on prefix-cache hits (validators
        # require usage to exactly restate the bound counts), and mutating
        # usage after hashing breaks the hash commitment instead
        # (observed as "response_hash mismatch" canaries when the
        # first version of this fix ran post-serialization).
        if isinstance(openai_response, dict) and (
            (prompt_token_source and prompt_token_ids is not None)
            or (completion_token_source and completion_token_ids is not None)
        ):
            usage = openai_response.get("usage")
            if not isinstance(usage, dict):
                usage = {}
                openai_response["usage"] = usage
            if prompt_token_source and prompt_token_ids is not None:
                usage["prompt_tokens"] = len(prompt_token_ids)
            if completion_token_source and completion_token_ids is not None:
                usage["completion_tokens"] = len(completion_token_ids)
            if (
                type(usage.get("prompt_tokens")) is int
                and type(usage.get("completion_tokens")) is int
            ):
                usage["total_tokens"] = (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                )
        request_bytes = json.dumps(openai_request, sort_keys=True, separators=(",", ":")).encode()
        response_bytes = json.dumps(openai_response, sort_keys=True, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(request_bytes).hexdigest()
        response_hash = hashlib.sha256(response_bytes).hexdigest()
        semantic_response_hash = semantic_openai_response_hash(openai_response)
        rpc_plan = rpc_plan_from_mesh(spec) if spec else None
        rpc_endpoints = rpc_plan.rpc_endpoints if rpc_plan else []
        receipt_body = {
            "version": 1,
            "request_id": request_id,
            "mesh_id": spec.mesh_id if spec else "",
            "mesh_spec_hash": spec.spec_hash_hex() if spec else "",
            "stage_assignment_hash": spec.stage_assignment_hash_hex() if spec else "",
            "model_package_hash": spec.model_package_hash if spec else "",
            "model_tensor_manifest_root": spec.model_tensor_manifest_root if spec else "",
            # Private RPC routes are committed by rpc_plan_hash but never
            # disclosed in validator-facing receipts.
            "rpc_endpoints": [],
            "rpc_plan_hash": rpc_plan.plan_hash_hex() if rpc_plan else "",
            "runtime": "llama_cpp_rpc" if rpc_endpoints else "openai_backend",
            "uid": capability.uid,
            "hotkey": capability.hotkey,
            "endpoint": capability.endpoint,
            "stage_index": int(stage_index),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "request_hash": request_hash,
            "response_hash": response_hash,
            "semantic_response_hash": semantic_response_hash,
            "inference_started_unix_ns": int(inference_started_unix_ns),
            "inference_ended_unix_ns": int(inference_ended_unix_ns),
            "proof_mode": LLAMA_CPP_RPC_RECEIPT_PROOF_MODE
            if rpc_endpoints
            else RECEIPT_ONLY_PROOF_MODE,
            "proof_required": bool(proof_required),
            "proof_receipt_root": "",
            "proof_receipt_count": 0,
            "proof_receipt_verified": False,
            "verified": False,
        }
        if proof_policy is not None:
            receipt_body.update(proof_policy)
        if receipt_body.get("verified_sampler_required"):
            sampled_controls = sampled_controls_from_context(receipt_body)
            if sampled_controls is not None:
                receipt_body["verified_sampler_controls_hash"] = (
                    verified_gguf_sampler_controls_hash(sampled_controls)
                )
            else:
                receipt_body["verified_sampler_mode"] = (
                    VERIFIED_GGUF_SAMPLER_MODE
                )
                receipt_body["verified_sampler_controls_hash"] = (
                    verified_gguf_sampler_controls_hash()
                )
        prompt_ids = list(prompt_token_ids or [])
        if prompt_token_source:
            receipt_body["prompt_token_ids_hash"] = prompt_token_ids_hash(prompt_ids)
            receipt_body["prompt_token_count"] = len(prompt_ids)
            receipt_body["prompt_token_source"] = prompt_token_source
            receipt_body["prompt_template_hash"] = prompt_template_root
        elif receipt_body.get("proof_metadata_required"):
            raise RuntimeError(
                "verified GGUF request missing prompt token ids; "
                "llama.cpp backend must support apply-template and tokenize"
            )
        token_ids = list(completion_token_ids or [])
        if completion_token_source:
            receipt_body["completion_token_ids_hash"] = completion_token_ids_hash(token_ids)
            receipt_body["completion_token_count"] = len(token_ids)
            receipt_body["completion_token_source"] = completion_token_source
        else:
            usage = openai_response.get("usage") if isinstance(openai_response, dict) else None
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                receipt_body["completion_token_count"] = int(usage.get("completion_tokens") or 0)
        if (
            receipt_body.get("proof_metadata_required")
            and not completion_token_source
            and not openai_request.get("stream")
            and not receipt_body.get("decode_audit_configured")
        ):
            raise RuntimeError(
                "verified GGUF response missing completion token ids; "
                "llama.cpp backend must expose token ids via the Verathos "
                "llama-server token extension, __verbose.tokens, or logprobs"
            )
        if (
            receipt_body.get("proof_capture_required")
            and int(receipt_body.get("completion_token_count", 0) or 0) <= 0
        ):
            # An empty completion has nothing to prove, and the slot view's
            # own guard surfaces as an opaque HTTP 500 several hops away
            # ("slot view requires at least one decoded token"). Name the
            # real condition here, where the cause is still visible.
            raise RuntimeError(
                "the model returned an empty reply (no tokens decoded), so "
                "there is nothing to prove. Reasoning models can answer "
                "with an empty message when thinking is disabled or the "
                "token budget is exhausted: retry with thinking enabled or "
                "a larger max_tokens."
            )
        slot_view_ctx: dict[str, Any] | None = None
        if receipt_body.get("proof_capture_required"):
            if slot_view_required and backend_url:
                from verallm.mesh.ggml_proof import SLOT_VIEW_SCOPE

                if int(slot_id) < 0:
                    if llama_n_parallel > 1:
                        raise RuntimeError(
                            "verified GGUF inference with --parallel > 1 needs the "
                            "llama-server slot id; the runtime must be built with "
                            "the Verathos server patches"
                        )
                    slot_id = 0
                receipt_body["proof_op_manifest_scope"] = SLOT_VIEW_SCOPE
                receipt_body["proof_slot_id"] = int(slot_id)
                receipt_body["proof_runtime_ubatch_size"] = int(llama_n_ubatch)
                receipt_body["proof_capture_window"] = str(capture_window_token or "")
                with _timed_section("build_slot_view_context"):
                    slot_view_ctx = build_slot_view_context(receipt_body)
            root, count, manifest_root, manifest_count = collect_trace_commitments(
                receipt_context=receipt_body,
                spec=spec,
                slot_view_context=slot_view_ctx,
                pinned_verification_snapshot=pinned_verification_snapshot,
            )
            if not root and not manifest_root:
                # Zero witnesses: with the prompt cache on this is a FULL
                # cache hit (exactly repeated prompt), which skips the eager
                # prefill that capture hooks observe. Re-serve once with the
                # cache disabled and collect from the fallback window.
                fallback_window, fallback_slot = rerun_starved_capture_serve(
                    receipt_body,
                    openai_request=openai_request,
                    openai_response=openai_response,
                    spec=spec,
                )
                if fallback_window:
                    if slot_view_ctx is not None:
                        if int(fallback_slot) >= 0:
                            receipt_body["proof_slot_id"] = int(fallback_slot)
                        receipt_body["proof_capture_window"] = fallback_window
                        slot_view_ctx = build_slot_view_context(receipt_body)
                    root, count, manifest_root, manifest_count = (
                        collect_trace_commitments(
                            receipt_context=receipt_body,
                            spec=spec,
                            slot_view_context=slot_view_ctx,
                            pinned_verification_snapshot=pinned_verification_snapshot,
                        )
                    )
            if not root and not manifest_root:
                raise RuntimeError(
                    "proof capture requires a GGML trace commitment or op manifest"
                )
            receipt_body["proof_trace_commitment_root"] = root
            receipt_body["proof_trace_commitment_count"] = int(count)
            if manifest_root:
                receipt_body["proof_op_manifest_root"] = manifest_root
                receipt_body["proof_op_manifest_count"] = int(manifest_count)
            # Streaming execution anchors, when the runtime produced them:
            # the ordered inventory digest freezes every anchored stage's
            # root BEFORE the nonce. It rides the receipt body, so it is
            # covered by receipt_hash and therefore by the postcommit
            # beacon (origin_receipt_hash), which is what makes the
            # nonce-selected row openings bind to this exact serve. It is
            # deliberately NOT in the grind-resistant gate hash: a
            # coordinator picks its own capture, and the nonce is what
            # makes that choice worthless.
            anchor_digest = local_anchor_inventory_digest(receipt_body)
            if anchor_digest:
                receipt_body["proof_anchor_inventory_digest"] = anchor_digest
            # Prefer the no-replay candidate-set scope for single-node serves
            # that captured real candidate witnesses (root present, count > 0),
            # including deferred proofs — the committed candidate root lets the
            # later beacon select a stored witness without regenerating, which
            # is O(1) in generation length and works on MoE. Multi-member
            # meshes and full decode audits keep the manifest-challenge scope.
            snapshot_bound_proofs = (
                str(receipt_body.get("proof_receipt_format", ""))
                == "opaque_stage_v2"
            )
            candidate_available = (
                bool(root)
                and int(count) > 0
                and int(receipt_body.get("decode_audit_bps", 0))
                < PROOF_SAMPLE_BPS_DENOMINATOR
            )
            # Original behavior: candidate scope for non-deferred captures.
            # New: single-node captures also use it for DEFERRED proofs (the
            # committed candidate root removes the O(n)/MoE-breaking replay).
            use_candidate_scope = candidate_available and (
                not receipt_body.get("proof_deferred")
                or proof_trace_candidate_capture
            )
            if snapshot_bound_proofs:
                if not manifest_root or int(manifest_count) <= 0:
                    raise RuntimeError(
                        "snapshot-bound proofs require a committed GGML op manifest"
                    )
                receipt_body["proof_trace_scope"] = "op_manifest_challenge_v1"
            elif manifest_root and not use_candidate_scope:
                receipt_body["proof_trace_scope"] = "op_manifest_challenge_v1"
            else:
                receipt_body["proof_trace_scope"] = "trace_candidate_set_v1"
        if int(receipt_body.get("decode_audit_bps", 0)) > 0:
            if not receipt_body.get("proof_capture_required"):
                raise RuntimeError("decode audit requires proof capture")
            receipt_body["decode_audit_stage_index"] = (
                decode_audit_stage_index_for_spec(spec)
                if spec is not None
                else int(stage_index)
            )
            receipt_body["decode_audit_commitment_hash"] = (
                mesh_decode_audit_commitment_hash(receipt_body)
            )
        else:
            # The secure transcript's policy context binds the stage index
            # unconditionally (it is a static property of the spec, not of
            # the draw), so a zero decode rate still stamps it - without
            # this every light proof failed "secure GGML transcript is
            # missing decode_audit_stage_index" once the rate moved to 0
            # under the current protocol.
            receipt_body["decode_audit_stage_index"] = (
                decode_audit_stage_index_for_spec(spec)
                if spec is not None
                else int(stage_index)
            )

        sample_bps = int(receipt_body.get("proof_sample_bps", 0))
        decode_bps = int(receipt_body.get("decode_audit_bps", 0))
        effective_sample_bps = max(sample_bps, decode_bps)
        if receipt_body.get("proof_capture_required") and effective_sample_bps > 0:
            from verallm.mesh.ggml_proof import (
                derive_every_request_trace_beacon,
                derive_every_request_trace_beacon_v2,
            )

            gate_hash = mesh_proof_gate_hash(receipt_body)
            if receipt_body.get("proof_postcommit"):
                beacon = b""
                beacon_available = False
            elif (
                receipt_body.get("proof_deferred")
                and receipt_body.get("proof_challenge_kind")
                == "deferred_future_randomness_v1"
            ):
                beacon = b""
                beacon_available = False
            elif effective_sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR:
                nonce_raw = validator_nonce_from_request(openai_request)
                if nonce_raw:
                    beacon = derive_every_request_trace_beacon_v2(
                        gate_hash,
                        normalize_validator_nonce(nonce_raw),
                    )
                else:
                    beacon = derive_every_request_trace_beacon(gate_hash)
                beacon_available = True
            else:
                nonce_raw = validator_nonce_from_request(openai_request)
                if nonce_raw:
                    normalize_validator_nonce(nonce_raw)
                    beacon = derive_mesh_proof_beacon(gate_hash, nonce_raw)
                    beacon_available = True
                else:
                    beacon = b""
                    beacon_available = False
            proof_sample_value = (
                0
                if sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR and beacon_available
                else mesh_proof_sample_value(beacon)
                if sample_bps > 0 and beacon_available
                else -1
            )
            proof_sampled = (
                sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR and beacon_available
                or (
                    sample_bps > 0
                    and beacon_available
                    and should_sample_mesh_proof(beacon=beacon, sample_bps=sample_bps)
                )
            )
            decode_sample_value = (
                0
                if decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR and beacon_available
                else mesh_decode_audit_sample_value(beacon)
                if decode_bps > 0 and beacon_available
                else -1
            )
            decode_sampled = (
                decode_bps >= PROOF_SAMPLE_BPS_DENOMINATOR and beacon_available
                or (
                    decode_bps > 0
                    and beacon_available
                    and should_sample_mesh_decode_audit(
                        beacon=beacon,
                        sample_bps=decode_bps,
                    )
                )
            )
            receipt_body["proof_required"] = bool(proof_sampled or decode_sampled)
            receipt_body["proof_sampled"] = bool(proof_sampled)
            receipt_body["proof_sample_value"] = int(proof_sample_value)
            receipt_body["proof_gate_hash"] = gate_hash
            receipt_body["proof_beacon"] = beacon.hex() if beacon_available else ""
            receipt_body["decode_audit_sampled"] = bool(decode_sampled)
            receipt_body["decode_audit_required"] = bool(decode_sampled)
            receipt_body["decode_audit_sample_value"] = int(decode_sample_value)
            receipt_body["decode_audit_positions"] = (
                derive_mesh_decode_audit_positions(
                    beacon=beacon,
                    decode_commitment_hash=str(
                        receipt_body.get("decode_audit_commitment_hash", "")
                    ),
                    completion_token_count=int(
                        receipt_body.get("completion_token_count", 0)
                    ),
                )
                if decode_sampled
                else []
            )
            if receipt_body.get("proof_deferred"):
                receipt_body["proof_deferred_sample_commitment_hash"] = (
                    mesh_deferred_audit_sample_commitment_hash(receipt_body)
                )
                receipt_body["proof_deferred_commitment_hash"] = (
                    mesh_deferred_audit_commitment_hash(receipt_body)
                )
        else:
            receipt_body["proof_sampled"] = False
            receipt_body["proof_sample_value"] = -1
            receipt_body["proof_gate_hash"] = ""
            receipt_body["proof_beacon"] = ""
            receipt_body["decode_audit_sampled"] = False
            receipt_body["decode_audit_sample_value"] = -1
            receipt_body["decode_audit_required"] = False
            receipt_body["decode_audit_positions"] = []

        proof_context = dict(receipt_body)
        if receipt_body.get("decode_audit_required") and token_ids:
            bind_decode_audit_completion_tokens(
                proof_context,
                token_ids,
                completion_token_source,
            )
        (
            proof_receipts,
            proof_payloads,
            proof_mode,
            proof_verified,
            proof_verifier_ms,
        ) = server.verathos_proof_receipt_collector(
            receipt_context=proof_context,
            openai_request=openai_request,
            openai_response=openai_response,
            proof_required=bool(receipt_body.get("proof_required")),
            spec=spec,
            slot_view_context=slot_view_ctx,
            pinned_verification_snapshot=pinned_verification_snapshot,
        )
        for replay_field in (
            "proof_replay_started_unix_ns",
            "proof_replay_ended_unix_ns",
            "decode_audit_completion_token_ids",
            "decode_audit_completion_token_ids_hash",
            "decode_audit_completion_token_count",
            "decode_audit_completion_token_source",
        ):
            if replay_field in proof_context:
                receipt_body[replay_field] = proof_context[replay_field]
        if proof_receipts:
            receipt_body["proof_mode"] = proof_mode
            if str(receipt_body.get("proof_receipt_format", "")) == "opaque_stage_v2":
                snapshot = (
                    pinned_verification_snapshot
                    if pinned_verification_snapshot is not None
                    else current_verification_snapshot(spec)
                )
                private_tokens = private_proof_tokens_for_spec(spec)
                public_receipts = [
                    MeshStageProofReceipt.from_dict(item) for item in proof_receipts
                ]
                receipt_body["proof_receipt_root"] = (
                    mesh_stage_proof_receipt_root_hex(public_receipts)
                )
                receipt_body["proof_receipt_count"] = len(public_receipts)
                receipt_body["verification_snapshot_hash"] = (
                    snapshot.snapshot_hash_hex()
                )
                receipt_body["verification_snapshot_generation"] = int(
                    snapshot.generation
                )
                receipt_body["verification_snapshot_epoch"] = int(snapshot.epoch)
                verify_mesh_stage_proof_receipts_for_snapshot(
                    receipt_body,
                    public_receipts,
                    snapshot,
                    require_complete_coverage=True,
                )
                _assert_public_proof_artifact_value(
                    {
                        "proof_receipts": proof_receipts,
                        "proof_payloads": proof_payloads,
                    },
                    private_tokens=private_tokens,
                )
            else:
                parsed = [
                    LlamaGraphOpReceipt.from_dict(item) for item in proof_receipts
                ]
                receipt_body["proof_receipt_root"] = llama_graph_receipt_root_hex(parsed)
                receipt_body["proof_receipt_count"] = len(parsed)
                if spec is not None:
                    verify_llama_graph_proof_receipts_for_mesh(
                        receipt_body,
                        parsed,
                        spec,
                    )
                else:
                    verify_llama_graph_proof_receipts(receipt_body, parsed)
            receipt_body["proof_receipt_verified"] = True
            receipt_body["verified"] = bool(proof_verified)
            receipt_body["proof_verifier_ms"] = round(float(proof_verifier_ms), 3)
        elif receipt_body.get("proof_required"):
            raise RuntimeError("proof receipts are required")
        if receipt_body.get("decode_audit_required"):
            from verallm.mesh.ggml_proof import (
                ggml_decode_audit_receipt_root,
                verify_ggml_decode_audit_payloads,
            )

            receipt_body["decode_audit_receipt_root"] = (
                ggml_decode_audit_receipt_root(proof_payloads)
            )
            decode_token_ids = [
                int(item)
                for item in receipt_body.get("decode_audit_completion_token_ids", [])
            ]
            if not decode_token_ids:
                raise RuntimeError("decode audit missing completion token ids")
            decode_result = verify_ggml_decode_audit_payloads(
                receipt_body,
                proof_payloads,
                completion_token_ids=decode_token_ids,
                receipts=proof_receipts,
            )
            receipt_body["decode_audit_verified"] = bool(decode_result.verified)
            receipt_body["decode_audit_verifier_ms"] = round(
                float(decode_result.verifier_ms),
                3,
            )
            if not decode_result.verified:
                raise RuntimeError(
                    "decode audit verification failed: " + decode_result.message
                )
        receipt_body["mesh_response_commitment_hash"] = mesh_response_commitment_hash(receipt_body)
        receipt_body["receipt_hash"] = mesh_receipt_hash(receipt_body)
        # Sign the canonical receipt hash with the serving hotkey (if configured
        # — operator-run coordinators leave the placeholder hotkey and an empty
        # signature). This binds the receipt to the miner's on-chain identity so
        # a validator can verify it against the metagraph, exactly as the vLLM
        # path is verified.
        if receipt_signer is not None:
            try:
                signature = str(receipt_signer(receipt_body["receipt_hash"]) or "")
                if not signature:
                    raise RuntimeError("receipt signer returned an empty signature")
                if secure_receipt_signing:
                    from verallm.mesh.receipt_signing import verify_receipt_signature

                    if not verify_receipt_signature(
                        receipt_body["receipt_hash"],
                        signature,
                        capability.hotkey,
                    ):
                        raise RuntimeError(
                            "receipt signature does not match coordinator hotkey"
                        )
                receipt_body["signature"] = signature
            except Exception as exc:
                if secure_receipt_signing:
                    raise RuntimeError("receipt signing failed") from exc
                logger.warning("receipt signing failed: %s", exc)
        return receipt_body, proof_receipts, proof_payloads

    def verify_worker_result(
        payload: dict[str, Any],
        request: dict[str, Any],
        *,
        spec: MeshSpec,
        member_index: int,
    ) -> None:
        verify_mesh_inference_artifact(
            payload,
            request,
            spec=spec,
            member_index=member_index,
            require_configured_proof=require_proof,
        )

    def assert_validator_artifact_privacy(
        artifact: Mapping[str, Any],
        *,
        spec: MeshSpec | None,
    ) -> None:
        """Ensure only the public coordinator identity crosses the boundary."""

        receipt = artifact.get("receipt")
        if not isinstance(receipt, Mapping):
            raise RuntimeError("validator artifact is missing its coordinator receipt")
        if int(receipt.get("uid", -1)) != int(capability.uid):
            raise RuntimeError("validator artifact exposes a non-coordinator uid")
        if str(receipt.get("hotkey", "")) != str(capability.hotkey):
            raise RuntimeError("validator artifact exposes a non-coordinator hotkey")
        if str(receipt.get("endpoint", "")) != str(capability.endpoint):
            raise RuntimeError("validator artifact exposes a non-coordinator endpoint")
        if receipt.get("rpc_endpoints") != []:
            raise RuntimeError("validator artifact exposes private RPC endpoints")
        private_tokens = private_proof_tokens_for_spec(spec)
        _assert_public_proof_artifact_value(
            {
                "proof_receipts": artifact.get("proof_receipts", []),
                "proof_payloads": artifact.get("proof_payloads", []),
            },
            private_tokens=private_tokens,
        )

    def verathos_mesh_metadata(
        routed: dict[str, Any],
        *,
        include_proof_payloads: bool = False,
    ) -> dict[str, Any]:
        """Build public proof metadata for a chat completion.

        Authenticated validator streams need the complete sanitized proof
        payloads in the terminal event so the proxy can independently verify
        the already-streamed completion before it finalizes scoring.  Other
        callers retain the compact reference-only response.
        """

        receipt = routed["receipt"]
        payload = {
            "receipt": receipt,
            "receipt_verified": True,
            "runtime": receipt.get("runtime", "openai_backend"),
            "model_package_hash": receipt.get("model_package_hash", ""),
            "model_tensor_manifest_root": receipt.get("model_tensor_manifest_root", ""),
            "rpc_endpoints": receipt.get("rpc_endpoints", []),
            "rpc_plan_hash": receipt.get("rpc_plan_hash", ""),
            "proof_required": receipt.get("proof_required", False),
            "proof_receipt_root": receipt.get("proof_receipt_root", ""),
            "proof_receipt_count": receipt.get("proof_receipt_count", 0),
            "proof_receipt_format": receipt.get("proof_receipt_format", ""),
            "proof_receipt_verified": receipt.get("proof_receipt_verified", False),
            "verification_snapshot_hash": receipt.get(
                "verification_snapshot_hash",
                "",
            ),
            "verification_snapshot_generation": receipt.get(
                "verification_snapshot_generation",
                0,
            ),
            "verification_snapshot_epoch": receipt.get(
                "verification_snapshot_epoch",
                0,
            ),
            "mesh_response_commitment_hash": receipt.get(
                "mesh_response_commitment_hash",
                "",
            ),
            "verified": receipt.get("verified", False),
            "proof_mode": receipt.get("proof_mode", "receipt_only"),
            "proof_policy_version": receipt.get("proof_policy_version", 1),
            "proof_configured_required": receipt.get("proof_configured_required", False),
            "proof_capture_required": receipt.get("proof_capture_required", False),
            "verified_sampler_required": receipt.get("verified_sampler_required", False),
            "proof_metadata_required": receipt.get("proof_metadata_required", False),
            "proof_deferred": receipt.get("proof_deferred", False),
            "proof_deferred_obligation": receipt.get(
                "proof_deferred_obligation",
                False,
            ),
            "proof_deferred_required": receipt.get("proof_deferred_required", False),
            "proof_sample_bps": receipt.get("proof_sample_bps", PROOF_SAMPLE_BPS_DENOMINATOR),
            "proof_sample_denominator": receipt.get(
                "proof_sample_denominator",
                PROOF_SAMPLE_BPS_DENOMINATOR,
            ),
            "proof_ops_per_request": receipt.get("proof_ops_per_request", 0),
            "proof_trace_candidates_per_request": receipt.get(
                "proof_trace_candidates_per_request",
                receipt.get("proof_ops_per_request", 0),
            ),
            "proof_challenge_kind": receipt.get("proof_challenge_kind", "disabled"),
            "proof_sampled": receipt.get("proof_sampled", False),
            "proof_sample_value": receipt.get("proof_sample_value", -1),
            "proof_gate_hash": receipt.get("proof_gate_hash", ""),
            "proof_beacon": receipt.get("proof_beacon", ""),
            "proof_trace_commitment_root": receipt.get("proof_trace_commitment_root", ""),
            "proof_trace_commitment_count": receipt.get("proof_trace_commitment_count", 0),
            "proof_trace_scope": receipt.get("proof_trace_scope", ""),
            "proof_op_manifest_root": receipt.get("proof_op_manifest_root", ""),
            "proof_op_manifest_count": receipt.get("proof_op_manifest_count", 0),
            "verified_sampler_mode": receipt.get("verified_sampler_mode", ""),
            "verified_sampler_controls_hash": receipt.get(
                "verified_sampler_controls_hash",
                "",
            ),
            # The applied profile for sampled light serves (empty when
            # greedy): lets clients see exactly which sampler ran and
            # recompute the committed controls hash.
            "verified_sampler_controls": dict(
                receipt.get("verified_sampler_controls") or {}
            ),
            "decode_audit_configured": receipt.get("decode_audit_configured", False),
            "decode_audit_mode": receipt.get("decode_audit_mode", ""),
            "decode_audit_bps": receipt.get("decode_audit_bps", 0),
            "decode_audit_stage_index": receipt.get(
                "decode_audit_stage_index",
                -1,
            ),
            "decode_audit_top_k": receipt.get("decode_audit_top_k", 0),
            "decode_audit_commitment_hash": receipt.get(
                "decode_audit_commitment_hash",
                "",
            ),
            "decode_audit_required": receipt.get("decode_audit_required", False),
            "decode_audit_sampled": receipt.get("decode_audit_sampled", False),
            "decode_audit_sample_value": receipt.get("decode_audit_sample_value", -1),
            "decode_audit_positions": receipt.get("decode_audit_positions", []),
            "decode_audit_receipt_root": receipt.get("decode_audit_receipt_root", ""),
            "decode_audit_verified": receipt.get("decode_audit_verified", False),
        }
        if receipt.get("prompt_token_source"):
            payload["prompt_token_ids_hash"] = receipt.get("prompt_token_ids_hash", "")
            payload["prompt_token_count"] = receipt.get("prompt_token_count", 0)
            payload["prompt_token_source"] = receipt.get("prompt_token_source", "")
            payload["prompt_template_hash"] = receipt.get("prompt_template_hash", "")
            payload["prompt_token_ids"] = routed.get("prompt_token_ids", [])
        if receipt.get("completion_token_source"):
            payload["completion_token_ids_hash"] = receipt.get(
                "completion_token_ids_hash",
                "",
            )
            payload["completion_token_count"] = receipt.get("completion_token_count", 0)
            payload["completion_token_source"] = receipt.get("completion_token_source", "")
            payload["completion_token_ids"] = routed.get("completion_token_ids", [])
        proof_receipts = routed.get("proof_receipts", [])
        if proof_receipts:
            payload["proof_receipts"] = proof_receipts
        proof_payloads = routed.get("proof_payloads", [])
        if proof_payloads:
            refs = []
            for index, proof_payload in enumerate(proof_payloads):
                if not isinstance(proof_payload, dict):
                    continue
                commitment = str(proof_payload.get("proof_commitment_hash", ""))
                if not commitment:
                    from verallm.mesh.ggml_proof import ggml_proof_payload_commitment_hash

                    commitment = ggml_proof_payload_commitment_hash(proof_payload)
                trace_meta = proof_payload.get("trace", {})
                openings = proof_payload.get("decode_audit_openings", [])
                refs.append(
                    {
                        "index": int(index),
                        "proof_commitment_hash": commitment,
                        "proof_kind": str(proof_payload.get("proof_kind", "gemm")),
                        "stage_index": proof_payload_stage_index(proof_payload),
                        "decode_audit_opening_count": (
                            len(openings) if isinstance(openings, list) else 0
                        ),
                        "payload_json_bytes": len(
                            json.dumps(
                                proof_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ).encode("utf-8")
                        ),
                    }
                )
            payload["proof_payload_count"] = len(proof_payloads)
            payload["proof_payloads_inline"] = bool(include_proof_payloads)
            payload["proof_payload_refs"] = refs
            if include_proof_payloads:
                payload["proof_payloads"] = proof_payloads
        return payload

    def make_deferred_audit_bundle_payload(body: dict[str, Any]) -> dict[str, Any]:
        artifact = body.get("artifact")
        if not isinstance(artifact, dict):
            raise RuntimeError("artifact must be an object")
        openai_request = body.get("openai_request")
        if not isinstance(openai_request, dict):
            raise RuntimeError("openai_request must be an object")
        openai_response = body.get("openai_response")
        if openai_response is not None and not isinstance(openai_response, dict):
            raise RuntimeError("openai_response must be an object")
        randomness = str(body.get("deferred_randomness") or body.get("randomness") or "")
        if not randomness:
            raise RuntimeError("deferred_randomness is required")
        response = _artifact_response(artifact, openai_response)
        spec = current_mesh_spec()
        verify_mesh_inference_artifact(
            artifact,
            openai_request,
            openai_response=response,
            spec=spec,
            deferred_randomness=randomness,
            require_deferred_proof_if_sampled=False,
        )
        receipt = artifact.get("receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError("artifact missing receipt")
        completion_ids, completion_source = _artifact_completion_token_binding(
            artifact,
            receipt,
        )
        audit_context = deferred_audit_context_from_receipt(
            receipt,
            randomness=randomness,
            completion_token_ids=completion_ids,
            completion_token_source=completion_source,
        )
        if not audit_context.get("proof_required"):
            raise RuntimeError("deferred randomness did not sample this artifact")
        (
            proof_receipts,
            proof_payloads,
            _proof_mode,
            _proof_verified,
            _proof_verifier_ms,
        ) = server.verathos_proof_receipt_collector(
            receipt_context=audit_context,
            openai_request=openai_request,
            openai_response=response,
            proof_required=True,
            spec=spec,
        )
        audit_context["proof_replay_started_unix_ns"] = 0
        audit_context["proof_replay_ended_unix_ns"] = 0
        # Replay-local capture bookkeeping is not part of the committed
        # receipt, so drop it before finalizing or the re-finalize in
        # verify_deferred_mesh_audit_bundle (which rebuilds a clean context
        # from the origin receipt) will not match.
        audit_context.pop("proof_replay_window_token", None)
        audit_context.pop("_mesh_audit_telemetry", None)
        finalized = finalize_deferred_audit_context(
            audit_context,
            proof_receipts=proof_receipts,
            proof_payloads=proof_payloads,
            spec=spec,
        )
        bundle = build_deferred_audit_bundle(
            origin_receipt=receipt,
            audit_receipt=finalized,
            randomness=randomness,
            proof_receipts=proof_receipts,
            proof_payloads=proof_payloads,
        )
        verify_deferred_mesh_audit_bundle(
            bundle,
            artifact,
            openai_request,
            openai_response=response,
            spec=spec,
        )
        return bundle

    def build_postcommit_audit_artifact(
        claim: dict[str, Any],
        *,
        principal: str,
    ) -> dict[str, Any]:
        origin_artifact = claim["origin_artifact"]
        openai_request = claim["openai_request"]
        challenge_nonce = claim["challenge_nonce"]
        audit_tier = str(claim.get("audit_tier", "") or "")
        spec = claim["spec"]
        snapshot = claim["snapshot"]
        snapshot_now_unix = claim["snapshot_now_unix"]
        origin_receipt = origin_artifact.get("receipt")
        if not isinstance(origin_receipt, dict):
            raise RuntimeError("cached postcommit origin receipt is missing")
        response = _artifact_response(origin_artifact)
        if spec is None or snapshot is None:
            raise RuntimeError(
                "postcommit origin is missing its pinned mesh verification state"
            )
        verify_mesh_inference_artifact(
            origin_artifact,
            openai_request,
            openai_response=response,
            spec=spec,
            require_configured_proof=True,
            require_cryptographic_proof=False,
            expected_coordinator_hotkey=capability.hotkey,
            expected_coordinator_uid=capability.uid,
            require_coordinator_signature=True,
            verification_snapshot=snapshot,
            verification_snapshot_now_unix=snapshot_now_unix,
            require_validator_request_id=True,
        )
        completion_ids, completion_source = _artifact_completion_token_binding(
            origin_artifact,
            origin_receipt,
        )
        # The replay probe regenerates the audited LM-head row as its own
        # single-row instance - unless another slot decoded inside the same
        # ubatch, in which case the row only exists inside a batched
        # (unprovable) instance and selection fails with "no captured
        # logits". That collision is transient serve-time contention, not a
        # serve defect, so the full replay is re-drawn a bounded number of
        # times; a genuine token mismatch fails identically on every draw
        # and still surfaces after the last attempt.
        _AUDIT_PROBE_DRAWS = 3
        audit_total_started = time.monotonic()
        audit_windows_total = 0
        audit_reprobes_total = 0
        audit_escalations_total = 0
        for _draw in range(_AUDIT_PROBE_DRAWS):
            audit_context = postcommit_audit_context_from_receipt(
                origin_receipt,
                openai_request,
                challenge_nonce=challenge_nonce,
                completion_token_ids=completion_ids,
                completion_token_source=completion_source,
                force_hard=audit_tier == "hard",
            )
            try:
                (
                    proof_receipts,
                    proof_payloads,
                    _proof_mode,
                    _proof_verified,
                    _proof_verifier_ms,
                ) = server.verathos_proof_receipt_collector(
                    receipt_context=audit_context,
                    openai_request=openai_request,
                    openai_response=response,
                    proof_required=bool(audit_context.get("proof_required")),
                    spec=spec,
                    pinned_verification_snapshot=snapshot,
                )
            except Exception as exc:
                telemetry = audit_context.get("_mesh_audit_telemetry") or {}
                audit_windows_total += int(telemetry.get("windows", 0) or 0)
                audit_reprobes_total += int(telemetry.get("reprobes", 0) or 0)
                audit_escalations_total += int(
                    telemetry.get("escalations", 0) or 0
                )
                if (
                    "decode audit found no captured logits" in str(exc)
                    and _draw + 1 < _AUDIT_PROBE_DRAWS
                ):
                    logger.info(
                        "postcommit audit probe draw %d/%d lost its "
                        "single-row LM-head instance to batching; "
                        "re-drawing: %s",
                        _draw + 1,
                        _AUDIT_PROBE_DRAWS,
                        exc,
                    )
                    time.sleep(0.25 * (_draw + 1))
                    continue
                raise
            break
        telemetry = audit_context.get("_mesh_audit_telemetry") or {}
        audit_windows_total += int(telemetry.get("windows", 0) or 0)
        audit_reprobes_total += int(telemetry.get("reprobes", 0) or 0)
        audit_escalations_total += int(telemetry.get("escalations", 0) or 0)
        logger.info(
            "mesh-audit-total request=%s tier=%s draws=%d windows_opened=%d "
            "reprobes=%d escalations=%d total_ms=%d",
            str(origin_receipt.get("request_id", ""))[:8],
            audit_tier or "base",
            _draw + 1,
            audit_windows_total,
            audit_reprobes_total,
            audit_escalations_total,
            int((time.monotonic() - audit_total_started) * 1000),
        )
        audit_context.pop("proof_replay_window_token", None)
        audit_context.pop("_mesh_audit_telemetry", None)
        finalized = finalize_postcommit_audit_context(
            audit_context,
            proof_receipts=proof_receipts,
            proof_payloads=proof_payloads,
            verification_snapshot=snapshot,
            spec=spec,
        )
        if receipt_signer is None:
            raise RuntimeError("postcommit coordinator receipt signer is missing")
        signature = str(receipt_signer(finalized["receipt_hash"]) or "")
        if not signature:
            raise RuntimeError("postcommit receipt signer returned an empty signature")
        if secure_receipt_signing:
            from verallm.mesh.receipt_signing import verify_receipt_signature

            if not verify_receipt_signature(
                finalized["receipt_hash"],
                signature,
                capability.hotkey,
            ):
                raise RuntimeError(
                    "postcommit receipt signature does not match coordinator"
                )
        finalized["signature"] = signature
        final_artifact = {
            "response": response,
            "receipt": finalized,
            "proof_receipts": proof_receipts,
            "proof_payloads": proof_payloads,
        }
        for field in ("prompt_token_ids", "completion_token_ids"):
            if field in origin_artifact:
                final_artifact[field] = origin_artifact[field]
        assert_validator_artifact_privacy(final_artifact, spec=spec)
        verify_mesh_postcommit_artifact(
            final_artifact,
            origin_artifact,
            openai_request,
            challenge_nonce=challenge_nonce,
            spec=spec,
            expected_coordinator_hotkey=capability.hotkey,
            expected_coordinator_uid=capability.uid,
            expected_validator_hotkey=principal,
            require_coordinator_signature=True,
            verification_snapshot=snapshot,
            verification_snapshot_now_unix=snapshot_now_unix,
            audit_tier=audit_tier,
        )
        return final_artifact

    def make_postcommit_audit_artifact(
        body: dict[str, Any],
        *,
        principal: str,
    ) -> bytes:
        claim = claim_postcommit_origin(body, principal=principal)
        cached_final = claim.get("cached_final")
        if isinstance(cached_final, bytes):
            return cached_final
        try:
            final_artifact = build_postcommit_audit_artifact(
                claim,
                principal=principal,
            )
            return complete_postcommit_finalization(claim, final_artifact)
        except Exception:
            release_postcommit_finalization(claim)
            raise

    def prompt_binding_for_request(
        openai_request: dict[str, Any],
        *,
        proof_metadata_required: bool,
    ) -> tuple[list[int], str, str]:
        if not proof_metadata_required:
            return [], "", ""
        if not backend_url:
            raise RuntimeError("verified GGUF request needs a local llama.cpp backend")
        template_request = backend_openai_request(openai_request, proof_capture_required=False)
        template_request.pop("stream", None)
        cache_key = hashlib.sha256(
            b"VERATHOS_MESH_PROMPT_BINDING_CACHE_V1"
            + json.dumps(
                template_request,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if prompt_binding_cache_enabled:
            with prompt_binding_cache_lock:
                cached = prompt_binding_cache.get(cache_key)
            if cached is not None:
                token_ids, token_source, template_root = cached
                return list(token_ids), token_source, template_root
        token_ids, token_source, template_root = fetch_prompt_token_ids_from_backend(
            backend_url,
            openai_request,
        )
        if prompt_binding_cache_enabled:
            with prompt_binding_cache_lock:
                prompt_binding_cache[cache_key] = (list(token_ids), token_source, template_root)
        return token_ids, token_source, template_root

    def bind_replay_seed_policy(
        policy: dict[str, Any],
        openai_request: dict[str, Any],
        request_id: str,
    ) -> int | None:
        """Derive and commit the replay seed for capture-enabled requests."""

        if not policy.get("proof_capture_required"):
            return None
        if policy.get("verified_sampler_required"):
            # The verified sampler profile already pins seed=0.
            policy["proof_replay_seed_mode"] = ""
            policy["proof_replay_seed"] = 0
            return None
        if openai_request.get("seed") is not None:
            policy["proof_replay_seed_mode"] = REPLAY_SEED_MODE_CLIENT
            policy["proof_replay_seed"] = 0
            return None
        seed = derive_mesh_replay_seed(request_id)
        policy["proof_replay_seed_mode"] = REPLAY_SEED_MODE_DERIVED
        policy["proof_replay_seed"] = int(seed)
        return int(seed)

    def _capture_wait_budget(validator_authenticated: bool) -> float:
        """Shared-join wait for this lane, resolved per request.

        Organic chats outwait queued hard replays (their stream stays
        alive via keepalives; a non-stream caller simply sees a slower
        response). Validator canaries keep the short budget so a busy
        mesh resolves into the retryable busy verdict they already price."""

        return float(
            SHARED_CAPTURE_WAIT_S
            if validator_authenticated
            else ORGANIC_CAPTURE_WAIT_S
        )

    def route_to_mesh(
        openai_request: dict[str, Any],
        *,
        require_validator_request_id: bool = False,
        validator_principal: str = "",
        require_postcommit: bool = False,
        validator_authenticated: bool = False,
        pinned_mesh_spec: MeshSpec | None = None,
        pinned_verification_snapshot: Any | None = None,
    ) -> dict[str, Any]:
        spec = (
            pinned_mesh_spec
            if pinned_mesh_spec is not None
            else current_mesh_spec()
        )
        policy = proof_policy_context(
            openai_request,
            pinned_verification_snapshot=pinned_verification_snapshot,
            validator_principal=validator_principal,
            require_postcommit=require_postcommit,
            validator_authenticated=validator_authenticated,
        )
        request_id = mesh_request_id_from_request(
            openai_request,
            require_validator_request_id=require_validator_request_id,
        )
        replay_seed = bind_replay_seed_policy(policy, openai_request, request_id)
        sampled_profile = finalize_verified_sampler_policy(
            policy, openai_request, request_id
        )
        backend_request = backend_openai_request(
            openai_request,
            proof_capture_required=policy["proof_capture_required"],
            verified_sampler_required=policy["verified_sampler_required"],
            proof_metadata_required=policy["proof_metadata_required"],
            replay_seed=replay_seed,
            sampled_profile=sampled_profile,
        )
        if spec is None:
            prompt_ids, prompt_source, prompt_root = prompt_binding_for_request(
                openai_request,
                proof_metadata_required=policy["proof_metadata_required"],
            )
            capture_info: dict[str, Any] = {}
            skip_trace_capture = organic_serve_skip_trace_capture(
                policy,
                None,
                validator_authenticated=validator_authenticated,
            )
            started_ns = time.time_ns()
            raw_response = forward_to_backend(
                backend_request,
                spec=None,
                request_id=request_id,
                proof_capture_required=policy["proof_capture_required"],
                capture_info=capture_info,
                skip_trace_capture=skip_trace_capture,
                capture_wait_timeout=_capture_wait_budget(validator_authenticated),
            )
            ended_ns = time.time_ns()
            local_response, token_ids, token_source = prepare_backend_response_for_receipt(
                raw_response,
                proof_capture_required=policy["proof_capture_required"],
                stream=False,
            )
            if (
                policy["proof_metadata_required"]
                and not policy.get("decode_audit_configured")
                and not token_source
            ):
                token_ids, token_source = fetch_completion_token_ids_from_backend(
                    backend_url,
                    local_response,
                )
            receipt, proof_receipts, proof_payloads = receipt_for(
                request_id=request_id,
                openai_request=openai_request,
                openai_response=local_response,
                prompt_token_ids=prompt_ids,
                prompt_token_source=prompt_source,
                prompt_template_root=prompt_root,
                completion_token_ids=token_ids,
                completion_token_source=token_source,
                stage_index=0,
                layer_start=0,
                layer_end=0,
                spec=None,
                proof_required=policy["proof_required"],
                proof_policy=policy,
                inference_started_unix_ns=started_ns,
                inference_ended_unix_ns=ended_ns,
                slot_id=slot_id_from_backend_response(raw_response),
                capture_window_token=capture_info.get("window_token", ""),
                pinned_verification_snapshot=pinned_verification_snapshot,
            )
            maybe_save_slot_state(
                request_id=request_id,
                prompt_token_ids=prompt_ids,
                completion_token_ids=token_ids,
                slot_id=slot_id_from_backend_response(raw_response),
                validator_authenticated=validator_authenticated,
            )
            return {
                "response": local_response,
                "receipt": receipt,
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": token_ids,
                "proof_receipts": proof_receipts,
                "proof_payloads": proof_payloads,
            }

        if backend_url:
            prompt_ids, prompt_source, prompt_root = prompt_binding_for_request(
                openai_request,
                proof_metadata_required=policy["proof_metadata_required"],
            )
            capture_info = {}
            skip_trace_capture = organic_serve_skip_trace_capture(
                policy,
                spec,
                validator_authenticated=validator_authenticated,
            )
            started_ns = time.time_ns()
            raw_response = forward_to_backend(
                backend_request,
                spec=spec,
                request_id=request_id,
                proof_capture_required=policy["proof_capture_required"],
                capture_info=capture_info,
                skip_trace_capture=skip_trace_capture,
                capture_wait_timeout=_capture_wait_budget(validator_authenticated),
            )
            ended_ns = time.time_ns()
            local_response, token_ids, token_source = prepare_backend_response_for_receipt(
                raw_response,
                proof_capture_required=policy["proof_capture_required"],
                stream=False,
            )
            if (
                policy["proof_metadata_required"]
                and not policy.get("decode_audit_configured")
                and not token_source
            ):
                token_ids, token_source = fetch_completion_token_ids_from_backend(
                    backend_url,
                    local_response,
                )
            member = next(
                (item for item in spec.members if item.endpoint == capability.endpoint),
                spec.members[0],
            )
            receipt, proof_receipts, proof_payloads = receipt_for(
                request_id=request_id,
                openai_request=openai_request,
                openai_response=local_response,
                prompt_token_ids=prompt_ids,
                prompt_token_source=prompt_source,
                prompt_template_root=prompt_root,
                completion_token_ids=token_ids,
                completion_token_source=token_source,
                stage_index=member.stage_index,
                layer_start=member.layers.start,
                layer_end=member.layers.end,
                spec=spec,
                proof_required=policy["proof_required"],
                proof_policy=policy,
                inference_started_unix_ns=started_ns,
                inference_ended_unix_ns=ended_ns,
                slot_id=slot_id_from_backend_response(raw_response),
                capture_window_token=capture_info.get("window_token", ""),
                pinned_verification_snapshot=pinned_verification_snapshot,
            )
            maybe_save_slot_state(
                request_id=request_id,
                prompt_token_ids=prompt_ids,
                completion_token_ids=token_ids,
                slot_id=slot_id_from_backend_response(raw_response),
                validator_authenticated=validator_authenticated,
            )
            return {
                "response": local_response,
                "receipt": receipt,
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": token_ids,
                "proof_receipts": proof_receipts,
                "proof_payloads": proof_payloads,
            }

        members = sorted(spec.members, key=lambda item: item.stage_index)
        candidates = [member for member in members if member.endpoint != capability.endpoint]
        if not candidates:
            raise RuntimeError("no mesh worker with backend is available")
        last_error: Exception | None = None
        for member in candidates:
            try:
                payload = post_json(
                    same_host_loopback(member.endpoint.rstrip("/"))
                    + "/v1/mesh/inference",
                    {
                        "request_id": request_id,
                        "mesh_spec_hash": spec.spec_hash_hex(),
                        "openai_request": openai_request,
                        "require_proof": require_proof,
                    },
                    timeout=120.0,
                    internal_auth_secret=internal_auth_secret,
                )
                verify_worker_result(
                    payload,
                    openai_request,
                    spec=spec,
                    member_index=spec.members.index(member),
                )
                return payload
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"all mesh workers failed: {last_error}")

    def route_to_mesh_stream(
        openai_request: dict[str, Any],
        emit_raw_sse,
        *,
        require_validator_request_id: bool = False,
        validator_principal: str = "",
        require_postcommit: bool = False,
        validator_authenticated: bool = False,
        pinned_mesh_spec: MeshSpec | None = None,
        pinned_verification_snapshot: Any | None = None,
    ) -> dict[str, Any]:
        spec = (
            pinned_mesh_spec
            if pinned_mesh_spec is not None
            else current_mesh_spec()
        )
        if not backend_url:
            raise RuntimeError("streaming mesh inference requires a local backend")
        original_request = {**openai_request, "stream": True}
        policy = proof_policy_context(
            original_request,
            pinned_verification_snapshot=pinned_verification_snapshot,
            validator_principal=validator_principal,
            require_postcommit=require_postcommit,
            validator_authenticated=validator_authenticated,
        )
        request_id = mesh_request_id_from_request(
            original_request,
            require_validator_request_id=require_validator_request_id,
        )
        replay_seed = bind_replay_seed_policy(policy, original_request, request_id)
        sampled_profile = finalize_verified_sampler_policy(
            policy, original_request, request_id
        )
        backend_request = backend_openai_request(
            original_request,
            proof_capture_required=policy["proof_capture_required"],
            verified_sampler_required=policy["verified_sampler_required"],
            proof_metadata_required=policy["proof_metadata_required"],
            replay_seed=replay_seed,
            sampled_profile=sampled_profile,
        )
        prompt_ids, prompt_source, prompt_root = prompt_binding_for_request(
            original_request,
            proof_metadata_required=policy["proof_metadata_required"],
        )
        capture_info: dict[str, Any] = {}
        skip_trace_capture = organic_serve_skip_trace_capture(
            policy,
            spec,
            validator_authenticated=validator_authenticated,
        )
        started_ns = time.time_ns()
        raw_response = forward_stream_to_backend(
            backend_request,
            spec=spec,
            request_id=request_id,
            proof_capture_required=policy["proof_capture_required"],
            emit_raw_sse=emit_raw_sse,
            capture_info=capture_info,
            skip_trace_capture=skip_trace_capture,
            capture_wait_timeout=_capture_wait_budget(validator_authenticated),
        )
        ended_ns = time.time_ns()
        response, token_ids, token_source = prepare_backend_response_for_receipt(
            raw_response,
            proof_capture_required=policy["proof_capture_required"],
            stream=True,
        )
        member = None
        if spec:
            member = next(
                (item for item in spec.members if item.endpoint == capability.endpoint),
                spec.members[0],
            )
        receipt, proof_receipts, proof_payloads = receipt_for(
            request_id=request_id,
            openai_request=original_request,
            openai_response=response,
            prompt_token_ids=prompt_ids,
            prompt_token_source=prompt_source,
            prompt_template_root=prompt_root,
            completion_token_ids=token_ids,
            completion_token_source=token_source,
            stage_index=member.stage_index if member else 0,
            layer_start=member.layers.start if member else 0,
            layer_end=member.layers.end if member else 0,
            spec=spec,
            proof_required=policy["proof_required"],
            proof_policy=policy,
            inference_started_unix_ns=started_ns,
            inference_ended_unix_ns=ended_ns,
            slot_id=slot_id_from_backend_response(raw_response),
            capture_window_token=capture_info.get("window_token", ""),
            pinned_verification_snapshot=pinned_verification_snapshot,
        )
        maybe_save_slot_state(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            completion_token_ids=token_ids,
            slot_id=slot_id_from_backend_response(raw_response),
            validator_authenticated=validator_authenticated,
        )
        return {
            "response": response,
            "receipt": receipt,
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": token_ids,
            "proof_receipts": proof_receipts,
            "proof_payloads": proof_payloads,
        }

    # Validator-pushed service receipts: mesh parity with the vLLM server's
    # POST /epoch/receipt + GET /epoch/{n}/receipts routes, so validators can
    # push canary/organic receipts and pull the full cross-validator set at
    # epoch close. Lazily created; own DB file so a co-hosted vLLM miner's
    # receipt store is never mixed with the mesh one.
    _receipt_store_holder: dict[str, Any] = {}

    def _mesh_receipt_store():
        store = _receipt_store_holder.get("store")
        if store is None:
            import os as _os

            from verallm.api.receipt_store import ReceiptStore

            data_dir = Path(
                _os.environ.get(
                    "VERALLM_DATA_DIR",
                    str(Path.home() / ".verathos"),
                )
            ).expanduser()
            data_dir.mkdir(parents=True, exist_ok=True)
            store = ReceiptStore(
                db_path=str(data_dir / "verathos_mesh_receipts.db")
            )
            _receipt_store_holder["store"] = store
        return store

    def validate_authenticated_service_receipt(
        body: dict[str, Any],
        *,
        principal: str,
    ) -> tuple[int, dict[str, Any]]:
        """Verify and canonicalize a validator receipt before persistence."""

        if not principal:
            raise PermissionError(
                "receipt submission requires validator authentication"
            )

        from scalecodec.utils.ss58 import ss58_encode

        from neurons.receipts import (
            ValidatorAuthority,
            receipt_from_dict,
            receipt_to_dict,
            verify_service_receipt,
        )

        receipt = receipt_from_dict(body)
        embedded_principal = ss58_encode(receipt.validator_hotkey)
        if embedded_principal != principal:
            raise PermissionError(
                "receipt signer does not match the authenticated validator"
            )

        authority = ValidatorAuthority(
            ss58_to_uid={principal: 0},
            validator_permit=[True],
            stakes=[1.0],
            min_stake=0.0,
        )
        if not verify_service_receipt(
            receipt,
            receipt.epoch_number,
            authority=authority,
        ):
            raise ValueError("receipt signature or freshness is invalid")

        if type(receipt.model_index) is not int or receipt.model_index < 0:
            raise ValueError("receipt model_index must be a non-negative integer")
        if type(receipt.epoch_number) is not int or receipt.epoch_number < 0:
            raise ValueError("receipt epoch_number must be a non-negative integer")
        if type(receipt.timestamp) is not int or receipt.timestamp <= 0:
            raise ValueError("receipt timestamp must be a positive integer")
        if type(receipt.tokens_generated) is not int or receipt.tokens_generated < 0:
            raise ValueError(
                "receipt tokens_generated must be a non-negative integer"
            )
        if type(receipt.prompt_tokens) is not int or receipt.prompt_tokens < 0:
            raise ValueError("receipt prompt_tokens must be a non-negative integer")
        if len(receipt.commitment_hash) != 32:
            raise ValueError("receipt commitment_hash must be 32 bytes")
        for field_name in (
            "ttft_ms",
            "generation_time_ms",
            "tokens_per_sec",
            "observed_start_ts",
            "observed_end_ts",
        ):
            value = getattr(receipt, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"receipt {field_name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"receipt {field_name} must be finite and non-negative"
                )
        if type(receipt.receipt_version) is not int or not (
            2 <= receipt.receipt_version <= 4
        ):
            # Version 2 introduced validator-observed timing; version 4 added
            # the exact canary-obligation fields stamped by current validators.
            raise ValueError(
                "authenticated mesh receipts must use version 2 through 4"
            )
        if receipt.timing_source != "validator_observed":
            raise ValueError(
                "authenticated mesh receipts require validator-observed timing"
            )
        if (
            receipt.observed_start_ts <= 0
            or receipt.observed_end_ts <= 0
            or receipt.observed_end_ts < receipt.observed_start_ts
        ):
            raise ValueError("receipt validator-observed timing is invalid")
        for field_name in (
            "proof_verified",
            "proof_requested",
            "is_canary",
        ):
            if type(getattr(receipt, field_name)) is not bool:
                raise ValueError(f"receipt {field_name} must be boolean")
        if receipt.tee_attestation_verified is not None and type(
            receipt.tee_attestation_verified
        ) is not bool:
            raise ValueError("receipt tee_attestation_verified must be boolean or null")

        if not evm_address or receipt.miner_address.lower() != evm_address.lower():
            raise PermissionError(
                "receipt miner address does not match this coordinator"
            )

        spec = current_mesh_spec()
        if spec is not None:
            if receipt.model_id != spec.model_id:
                raise PermissionError(
                    "receipt model does not match this coordinator"
                )
            # The spec's epoch is the LAUNCH epoch and is never updated by
            # snapshot rotation, so this equality only holds for a mesh's
            # first epoch (observed: every receipt push after the
            # first rotation was rejected 403 and the miner closed each
            # epoch with 0/0 receipts). Snapshot-bound meshes enforce the
            # CURRENT epoch against the rotated snapshot below; this check
            # only guards meshes without a snapshot loader.
            if (
                verification_snapshot_loader is None
                and int(spec.epoch) > 0
                and receipt.epoch_number != int(spec.epoch)
            ):
                raise PermissionError(
                    "receipt epoch does not match the active mesh"
                )

        if verification_snapshot_loader is not None:
            snapshot = current_verification_snapshot(spec)
            if receipt.model_id != snapshot.model.model_id:
                raise PermissionError(
                    "receipt model does not match the verification snapshot"
                )
            if receipt.model_index != snapshot.coordinator.model_index:
                raise PermissionError(
                    "receipt model index does not match the verification snapshot"
                )
            if receipt.epoch_number != snapshot.epoch:
                raise PermissionError(
                    "receipt epoch does not match the verification snapshot"
                )

        return receipt.epoch_number, receipt_to_dict(receipt)

    def effective_server_role() -> str:
        if server_role != "auto":
            return server_role
        spec = current_mesh_spec()
        if spec is not None:
            member = local_member_for_spec(spec)
            if member is not None:
                return "coordinator" if member.role == "coordinator" else "worker"
        return "coordinator" if backend_url else "worker"

    def is_validator_route(method: str, path: str) -> bool:
        if effective_server_role() != "coordinator":
            return False
        if method == "GET":
            return bool(re.fullmatch(r"/epoch/\d+/receipts", path)) or path == (
                "/v1/mesh/verification-snapshot"
            )
        return path in {
            "/epoch/receipt",
            "/v1/chat/completions",
            "/v1/mesh/inference",
            "/v1/mesh/proof/deferred-audit",
            POSTCOMMIT_AUDIT_PATH,
        }

    def is_public_route(method: str, path: str) -> bool:
        # /capacity/roster is the validator's unauthenticated PULL path for
        # the signed GPU roster; authenticity rides the embedded
        # coordinator-EVM signature, so an auth wall here would only stop
        # validators from ever learning the mesh's audit obligation.
        return (
            method == "GET" and path in ("/health", "/capacity/roster")
        ) or (method == "POST" and path == "/identity/challenge")

    def claim_fresh_validator_nonce(
        openai_request: dict[str, Any],
        *,
        principal: str,
    ) -> None:
        nonce = normalize_validator_nonce(
            validator_nonce_from_request(openai_request)
        )
        fingerprint = hashlib.sha256(
            b"verathos.mesh.validator-nonce.v1\x00"
            + principal.encode("utf-8")
            + nonce
        ).digest()
        now = time.time()
        if not validator_nonce_replays.claim(
            fingerprint,
            expires_at=now + 3600.0,
            now=now,
        ):
            raise ValueError("validator_nonce was already used")

    def claim_fresh_validator_challenge_commitment(
        openai_request: dict[str, Any],
        *,
        principal: str,
    ) -> None:
        try:
            commitment = _require_sha256_text(
                "challenge_nonce_commitment",
                validator_challenge_commitment_from_request(openai_request),
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        fingerprint = hashlib.sha256(
            b"verathos.mesh.validator-challenge-commitment.v1\x00"
            + principal.encode("utf-8")
            + bytes.fromhex(commitment)
        ).digest()
        now = time.time()
        if not validator_challenge_commitment_replays.claim(
            fingerprint,
            expires_at=now + POSTCOMMIT_ORIGIN_TTL_SECONDS,
            now=now,
        ):
            raise ValueError(
                "validator challenge commitment was already used"
            )

    def prune_postcommit_cache_locked(*, now: float) -> None:
        expired_origins = [
            item_key
            for item_key, stored in postcommit_origins.items()
            if stored[0] <= now
        ]
        for item_key in expired_origins:
            del postcommit_origins[item_key]
        expired_finalized = [
            item_key
            for item_key, stored in postcommit_finalized.items()
            if stored[0] <= now
        ]
        for item_key in expired_finalized:
            del postcommit_finalized[item_key]

    def decode_postcommit_origin(payload: bytes) -> dict[str, Any]:
        origin = json.loads(payload)
        expected_fields = {
            "artifact",
            "openai_request",
            "mesh_spec",
            "verification_snapshot",
            "snapshot_now_unix",
        }
        if not isinstance(origin, dict) or set(origin) != expected_fields:
            raise RuntimeError("cached postcommit origin is invalid")
        if not isinstance(origin["artifact"], dict):
            raise RuntimeError("cached postcommit origin artifact is invalid")
        if not isinstance(origin["openai_request"], dict):
            raise RuntimeError("cached postcommit origin request is invalid")
        if origin["mesh_spec"] is not None and not isinstance(
            origin["mesh_spec"],
            dict,
        ):
            raise RuntimeError("cached postcommit origin mesh spec is invalid")
        if origin["verification_snapshot"] is not None and not isinstance(
            origin["verification_snapshot"],
            dict,
        ):
            raise RuntimeError("cached postcommit origin snapshot is invalid")
        if type(origin["snapshot_now_unix"]) is not int:
            raise RuntimeError("cached postcommit origin timestamp is invalid")
        return origin

    def acquire_postcommit_request_slot(principal: str) -> None:
        with postcommit_origin_lock:
            principal_active = postcommit_active_requests.get(principal, 0)
            if (
                principal_active
                >= POSTCOMMIT_PHASE2_MAX_ACTIVE_PER_PRINCIPAL
                or sum(postcommit_active_requests.values())
                >= POSTCOMMIT_PHASE2_MAX_ACTIVE
            ):
                raise _PostcommitCapacityUnavailable(
                    "postcommit request capacity is temporarily unavailable"
                )
            postcommit_active_requests[principal] = principal_active + 1

    def release_postcommit_request_slot(principal: str) -> None:
        with postcommit_origin_lock:
            principal_active = postcommit_active_requests.get(principal, 0)
            if principal_active <= 1:
                postcommit_active_requests.pop(principal, None)
            else:
                postcommit_active_requests[principal] = principal_active - 1

    def require_postcommit_replay_reservation_locked(principal: str) -> None:
        """Reserve worst-case replay space before returning a signed origin."""

        principal_pending = sum(
            1 for item_key in postcommit_origins if item_key[0] == principal
        )
        principal_finalized = [
            stored
            for item_key, stored in postcommit_finalized.items()
            if item_key[0] == principal
        ]
        if (
            principal_pending + len(principal_finalized) + 1
            > POSTCOMMIT_FINALIZED_MAX_ENTRIES_PER_PRINCIPAL
        ):
            raise RuntimeError(
                "postcommit replay reservation limit reached for validator"
            )
        if (
            len(postcommit_origins) + len(postcommit_finalized) + 1
            > POSTCOMMIT_FINALIZED_MAX_ENTRIES
        ):
            raise RuntimeError("postcommit replay reservation cache is full")
        principal_reserved_bytes = (
            sum(len(stored[3]) for stored in principal_finalized)
            + principal_pending * POSTCOMMIT_PENDING_RESERVATION_BYTES
        )
        if (
            principal_reserved_bytes
            + POSTCOMMIT_PENDING_RESERVATION_BYTES
            > POSTCOMMIT_FINALIZED_MAX_BYTES_PER_PRINCIPAL
        ):
            raise RuntimeError(
                "postcommit replay byte reservation reached for validator"
            )
        global_reserved_bytes = (
            sum(len(stored[3]) for stored in postcommit_finalized.values())
            + len(postcommit_origins)
            * POSTCOMMIT_PENDING_RESERVATION_BYTES
        )
        if (
            global_reserved_bytes + POSTCOMMIT_PENDING_RESERVATION_BYTES
            > POSTCOMMIT_FINALIZED_MAX_BYTES
        ):
            raise RuntimeError("postcommit replay byte reservation cache is full")

    def require_postcommit_finalized_room_locked(
        *,
        key: tuple[str, str, str],
        artifact_size: int,
    ) -> None:
        principal = key[0]
        if artifact_size > POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES:
            raise RuntimeError("postcommit final artifact exceeds replay cache limit")
        other_pending = [
            item_key for item_key in postcommit_origins if item_key != key
        ]
        principal_other_pending = sum(
            1 for item_key in other_pending if item_key[0] == principal
        )
        principal_finalized = [
            stored
            for item_key, stored in postcommit_finalized.items()
            if item_key[0] == principal
        ]
        if (
            len(principal_finalized) + principal_other_pending + 1
            > POSTCOMMIT_FINALIZED_MAX_ENTRIES_PER_PRINCIPAL
        ):
            raise RuntimeError(
                "postcommit replay entry reservation invariant failed for validator"
            )
        if (
            len(postcommit_finalized) + len(other_pending) + 1
            > POSTCOMMIT_FINALIZED_MAX_ENTRIES
        ):
            raise RuntimeError("postcommit replay entry reservation invariant failed")
        principal_bytes = (
            sum(len(stored[3]) for stored in principal_finalized)
            + principal_other_pending * POSTCOMMIT_PENDING_RESERVATION_BYTES
            + artifact_size
        )
        global_bytes = (
            sum(len(stored[3]) for stored in postcommit_finalized.values())
            + len(other_pending) * POSTCOMMIT_PENDING_RESERVATION_BYTES
            + artifact_size
        )
        if principal_bytes > POSTCOMMIT_FINALIZED_MAX_BYTES_PER_PRINCIPAL:
            raise RuntimeError(
                "postcommit replay reservation invariant failed for validator"
            )
        if global_bytes > POSTCOMMIT_FINALIZED_MAX_BYTES:
            raise RuntimeError("postcommit replay reservation invariant failed")

    def store_postcommit_origin(
        artifact: dict[str, Any],
        openai_request: dict[str, Any],
        *,
        principal: str,
        spec: MeshSpec | None,
        verification_snapshot: Any | None,
    ) -> None:
        receipt = artifact.get("receipt")
        if not isinstance(receipt, dict) or not receipt.get("proof_postcommit"):
            return
        request_id = normalize_validator_request_id(
            str(receipt.get("request_id", ""))
        )
        receipt_hash = _require_sha256_text(
            "origin_receipt_hash",
            receipt.get("receipt_hash", ""),
        )
        if str(receipt.get("proof_validator_hotkey", "")) != principal:
            raise RuntimeError("postcommit origin validator principal mismatch")
        key = (principal, request_id, receipt_hash)
        now = time.time()
        deadline = now + POSTCOMMIT_ORIGIN_TTL_SECONDS
        stored_origin = _bounded_canonical_json_bytes(
            {
                "artifact": artifact,
                "openai_request": openai_request,
                "mesh_spec": spec.to_dict() if spec is not None else None,
                "verification_snapshot": (
                    verification_snapshot.to_dict()
                    if verification_snapshot is not None
                    else None
                ),
                "snapshot_now_unix": int(now),
            },
            max_bytes=POSTCOMMIT_ORIGIN_MAX_SINGLE_BYTES,
            limit_error="postcommit origin exceeds single-origin cache limit",
        )
        stored_size = len(stored_origin)
        with postcommit_origin_lock:
            prune_postcommit_cache_locked(now=now)
            if (
                key in postcommit_origins
                or key in postcommit_finalized
                or key in postcommit_finalizations
            ):
                raise RuntimeError("postcommit request state was already stored")
            principal_entries = sum(
                1
                for item_key in postcommit_origins
                if item_key[0] == principal
            )
            if (
                principal_entries
                >= POSTCOMMIT_ORIGIN_MAX_ENTRIES_PER_PRINCIPAL
            ):
                raise RuntimeError(
                    "postcommit origin cache limit reached for validator"
                )
            if len(postcommit_origins) >= POSTCOMMIT_ORIGIN_MAX_ENTRIES:
                raise RuntimeError("postcommit origin cache is full")
            principal_bytes = sum(
                len(stored[1])
                for item_key, stored in postcommit_origins.items()
                if item_key[0] == principal
            )
            if (
                principal_bytes + stored_size
                > POSTCOMMIT_ORIGIN_MAX_BYTES_PER_PRINCIPAL
            ):
                raise RuntimeError(
                    "postcommit origin byte limit reached for validator"
                )
            global_bytes = sum(
                len(stored[1]) for stored in postcommit_origins.values()
            )
            if global_bytes + stored_size > POSTCOMMIT_ORIGIN_MAX_BYTES:
                raise RuntimeError("postcommit origin byte cache is full")
            require_postcommit_replay_reservation_locked(principal)
            postcommit_origins[key] = (
                deadline,
                stored_origin,
                int(now),
                0,
            )

    def claim_postcommit_origin(
        body: dict[str, Any],
        *,
        principal: str,
    ) -> dict[str, Any]:
        expected_fields = {
            "validator_request_id",
            "origin_receipt_hash",
            "mesh_response_commitment_hash",
            "verification_snapshot_hash",
            "challenge_nonce",
        }
        # audit_tier is the validator's OPTIONAL stricter-only tier demand
        # ("hard" forces the hard relation on this reveal; canary hard
        # slots). It rides the same signed body, so a miner cannot inject
        # or strip it, and there is no "light" demand: a reveal can never
        # fall below the nonce-derived tier draw.
        if set(body) - {"audit_tier"} != expected_fields:
            raise RuntimeError(
                "postcommit request fields do not match the closed schema"
            )
        audit_tier = str(body.get("audit_tier", "") or "")
        if audit_tier not in ("", "hard"):
            raise RuntimeError(
                "postcommit audit_tier demand must be omitted or 'hard'"
            )
        request_id = normalize_validator_request_id(
            str(body.get("validator_request_id", ""))
        )
        origin_hash = _require_sha256_text(
            "origin_receipt_hash",
            body.get("origin_receipt_hash", ""),
        )
        response_commitment = _require_sha256_text(
            "mesh_response_commitment_hash",
            body.get("mesh_response_commitment_hash", ""),
        )
        snapshot_hash = _require_sha256_text(
            "verification_snapshot_hash",
            body.get("verification_snapshot_hash", ""),
        )
        challenge_nonce = normalize_validator_challenge_nonce(
            str(body.get("challenge_nonce", ""))
        ).hex()
        key = (principal, request_id, origin_hash)
        reveal_identity = (
            response_commitment,
            snapshot_hash,
            challenge_nonce,
            audit_tier,
        )
        now = time.time()
        cached_final_artifact: bytes | None = None
        stored = None
        with postcommit_origin_lock:
            prune_postcommit_cache_locked(now=now)
            finalized = postcommit_finalized.get(key)
            if finalized is not None:
                _, _, finalized_reveal, final_artifact = finalized
                if finalized_reveal != reveal_identity:
                    raise RuntimeError(
                        "postcommit retry conflicts with finalized artifact"
                    )
                cached_final_artifact = final_artifact
            else:
                finalization = postcommit_finalizations.get(key)
                if finalization is not None:
                    active_reveal, _ = finalization
                    if active_reveal != reveal_identity:
                        raise RuntimeError(
                            "postcommit retry conflicts with active finalization"
                        )
                    raise _PostcommitFinalizationInProgress(
                        "postcommit finalization is already in progress"
                    )
                stored = postcommit_origins.get(key)
                if stored is None:
                    raise RuntimeError(
                        "postcommit origin is missing or expired"
                    )
        if cached_final_artifact is not None:
            return {"cached_final": cached_final_artifact}
        if stored is None:
            raise RuntimeError("postcommit origin state is unavailable")
        (
            expires_at,
            stored_origin,
            snapshot_now_unix,
            finalization_attempts,
        ) = stored
        decoded_origin = decode_postcommit_origin(stored_origin)
        artifact = decoded_origin["artifact"]
        openai_request = decoded_origin["openai_request"]
        stored_spec = decoded_origin["mesh_spec"]
        stored_snapshot = decoded_origin["verification_snapshot"]
        if int(decoded_origin["snapshot_now_unix"]) != snapshot_now_unix:
            raise RuntimeError("cached postcommit origin timestamp mismatch")
        receipt = artifact.get("receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError("cached postcommit origin receipt is missing")
        if receipt.get("mesh_response_commitment_hash") != response_commitment:
            raise RuntimeError("postcommit response commitment mismatch")
        if receipt.get("verification_snapshot_hash") != snapshot_hash:
            raise RuntimeError("postcommit verification snapshot mismatch")
        if str(receipt.get("proof_validator_hotkey", "")) != principal:
            raise RuntimeError("postcommit validator principal mismatch")
        # This validates the reveal against the request commitment before any
        # expensive proof selection/replay starts.
        postcommit_audit_decision(
            receipt,
            openai_request,
            challenge_nonce=challenge_nonce,
            force_hard=audit_tier == "hard",
        )
        pinned_spec = (
            MeshSpec.from_dict(stored_spec)
            if stored_spec is not None
            else None
        )
        pinned_snapshot = None
        if stored_snapshot is not None:
            from verallm.mesh.verification_snapshot import (
                MeshVerificationSnapshot,
            )

            pinned_snapshot = MeshVerificationSnapshot.from_dict(
                stored_snapshot
            )
        cached_final_artifact = None
        claim_token = None
        with postcommit_origin_lock:
            now = time.time()
            prune_postcommit_cache_locked(now=now)
            finalized = postcommit_finalized.get(key)
            if finalized is not None:
                _, _, finalized_reveal, final_artifact = finalized
                if finalized_reveal != reveal_identity:
                    raise RuntimeError(
                        "postcommit retry conflicts with finalized artifact"
                    )
                cached_final_artifact = final_artifact
            else:
                finalization = postcommit_finalizations.get(key)
                if finalization is not None:
                    active_reveal, _ = finalization
                    if active_reveal != reveal_identity:
                        raise RuntimeError(
                            "postcommit retry conflicts with active finalization"
                        )
                    raise _PostcommitFinalizationInProgress(
                        "postcommit finalization is already in progress"
                    )
                if postcommit_origins.get(key) is not stored:
                    raise RuntimeError(
                        "postcommit origin state changed during validation"
                    )
                if finalization_attempts >= POSTCOMMIT_FINALIZATION_MAX_ATTEMPTS:
                    raise RuntimeError("postcommit finalization retry limit reached")
                principal_finalizations = sum(
                    1
                    for item_key in postcommit_finalizations
                    if item_key[0] == principal
                )
                if (
                    principal_finalizations
                    >= POSTCOMMIT_FINALIZATION_MAX_ACTIVE_PER_PRINCIPAL
                    or len(postcommit_finalizations)
                    >= POSTCOMMIT_FINALIZATION_MAX_ACTIVE
                ):
                    raise _PostcommitCapacityUnavailable(
                        "postcommit proof capacity is temporarily unavailable"
                    )
                claim_token = object()
                postcommit_origins[key] = (
                    *stored[:-1],
                    finalization_attempts + 1,
                )
                postcommit_finalizations[key] = (reveal_identity, claim_token)
        if cached_final_artifact is not None:
            return {"cached_final": cached_final_artifact}
        if claim_token is None:
            raise RuntimeError("postcommit finalization claim is unavailable")
        return {
            "cached_final": None,
            "origin_artifact": artifact,
            "openai_request": openai_request,
            "challenge_nonce": challenge_nonce,
            "audit_tier": audit_tier,
            "spec": pinned_spec,
            "snapshot": pinned_snapshot,
            "snapshot_now_unix": int(snapshot_now_unix),
            "cache_key": key,
            "reveal_identity": reveal_identity,
            "claim_token": claim_token,
        }

    def release_postcommit_finalization(claim: dict[str, Any]) -> None:
        key = claim["cache_key"]
        claim_token = claim["claim_token"]
        with postcommit_origin_lock:
            finalization = postcommit_finalizations.get(key)
            if finalization is not None and finalization[1] is claim_token:
                del postcommit_finalizations[key]

    def complete_postcommit_finalization(
        claim: dict[str, Any],
        final_artifact: dict[str, Any],
    ) -> bytes:
        nonlocal postcommit_finalized_sequence

        key = claim["cache_key"]
        reveal_identity = claim["reveal_identity"]
        claim_token = claim["claim_token"]
        stored_final = _bounded_canonical_json_bytes(
            final_artifact,
            max_bytes=POSTCOMMIT_FINALIZED_MAX_ARTIFACT_BYTES,
            limit_error="postcommit final artifact exceeds replay cache limit",
        )
        now = time.time()
        with postcommit_origin_lock:
            finalization = postcommit_finalizations.get(key)
            if finalization is None or finalization[1] is not claim_token:
                raise RuntimeError("postcommit finalization claim was lost")
            origin = postcommit_origins.get(key)
            if origin is None:
                del postcommit_finalizations[key]
                raise RuntimeError("postcommit origin disappeared during finalization")
            expires_at = origin[0]
            if expires_at <= now:
                del postcommit_origins[key]
                del postcommit_finalizations[key]
                raise RuntimeError("postcommit origin expired during finalization")
            prune_postcommit_cache_locked(now=now)
            require_postcommit_finalized_room_locked(
                key=key,
                artifact_size=len(stored_final),
            )
            postcommit_finalized_sequence += 1
            postcommit_finalized[key] = (
                expires_at,
                postcommit_finalized_sequence,
                reveal_identity,
                stored_final,
            )
            del postcommit_origins[key]
            del postcommit_finalizations[key]
        return stored_final

    # Uniform token-ledger admission (verallm/mesh/admission.py). The
    # backend's unified KV is ONE shared pool: llama admits requests into
    # slots without reserving context and then errors the stream when the
    # pool runs dry ("Context size has been exceeded" killed two
    # overlapping 88k canaries as unretryable 500s).
    # The coordinator knows every request's demand up front, so what
    # cannot fit RIGHT NOW gets an instant clean 503 instead - uniformly,
    # with zero lane inspection: canaries are byte-indistinguishable from
    # organic traffic by design and any priority would be a cheating
    # oracle. No queue (owner decision): routers fail organic traffic
    # over instantly, validators reschedule canaries via the busy
    # machinery, and epoch-close busy evidence keeps 503s honest.
    from verallm.mesh.admission import (
        Admission,
        KVAdmissionLedger,
        ReservationGuard,
    )

    _admission_state: dict[str, Any] = {"ledger": None}
    _admission_lock = threading.Lock()

    def _admission_ledger() -> KVAdmissionLedger:
        if _admission_state.get("resolved"):
            return _admission_state["ledger"]
        with _admission_lock:
            ledger = _admission_state["ledger"]
            if ledger is None:
                # ONE persistent ledger from the very first request:
                # slots-only until the real capacity is known. A fresh
                # instance per request while llama loads lost all
                # in-flight state - the busy 503 could never fire during
                # warm-up (caught by the route admission tests).
                ledger = KVAdmissionLedger(
                    capacity_tokens=1_000_000_000,
                    slots=max(1, int(llama_n_parallel)),
                    per_request_cap=0,
                )
                _admission_state["ledger"] = ledger
            if _admission_state.get("resolved"):
                return ledger
            # Backoff between /props attempts: while llama is still
            # loading, every request would otherwise pay a 5s probe.
            now = time.monotonic()
            if now < float(_admission_state.get("props_retry_after", 0)):
                return ledger
            _admission_state["props_retry_after"] = now + 10.0
            # The serve process launches llama-server itself, so its own
            # --llama-ctx-size IS the fitted budget (the KV auto-fit
            # ladder respawns the whole serve on descent). 0 = model
            # trained maximum: ask llama once; a backend that is not up
            # yet degrades to slots-only admission until it answers.
            capacity = int(llama_ctx_budget or 0)
            resolved = capacity > 0
            if not resolved and backend_url:
                try:
                    with urlopen(
                        f"{backend_url}/props", timeout=5.0
                    ) as response:
                        props = json.loads(response.read())
                    capacity = int(
                        props.get("n_ctx")
                        or (
                            props.get("default_generation_settings") or {}
                        ).get("n_ctx")
                        or 0
                    )
                    resolved = capacity > 0
                except Exception:
                    capacity = 0
            if not resolved and backend_url:
                return ledger  # slots-only until llama answers
            # The per-request cap is the PHYSICAL serving budget (llama's
            # unified KV pool), never the chain contract: kv-unified means
            # one request may legitimately use the model's full launched
            # context, and the ledger's shared-capacity accounting already
            # refuses anything that cannot fit. Reading the registered
            # max_context_len here made a RE-deploy impossible - the
            # measurement mesh of an already-registered model inherited
            # the OLD registration's contract and 400'd every probe above
            # it. Advertised-context
            # enforcement is the validator's scoring concern, not
            # admission's.
            # Upgrade IN PLACE: replacing the instance would orphan
            # in-flight reservations (their releases would land on the
            # new ledger and silently reset the accounting).
            ledger.configure(
                capacity_tokens=(
                    capacity if capacity > 0 else 1_000_000_000
                ),
                per_request_cap=0,
            )
            _admission_state["resolved"] = True
            return ledger

    def admission_demand_tokens(openai_request: dict[str, Any]) -> int:
        """Uniform reservation size: prompt tokens + decode budget.

        Uses the SAME cache-backed tokenize path the proof binding uses,
        so the later binding call is a cache hit (one /tokenize round
        trip per request). Identical for every request; nothing here
        looks at who is asking.
        """

        prompt_tokens = 0
        if backend_url:
            # BOUNDED tokenize: llama serializes /tokenize behind a
            # saturated prefill, so an unbounded call stalls the very
            # verdict that must be instant under load. On timeout the char
            # estimate below - which
            # OVERESTIMATES tokens for natural text - keeps admission
            # conservative; the abandoned tokenize still completes in
            # its thread and warms the binding cache for the proof path.
            outcome: dict[str, int] = {}

            def _tokenize() -> None:
                try:
                    token_ids, _source, _root = prompt_binding_for_request(
                        openai_request,
                        proof_metadata_required=True,
                    )
                    outcome["tokens"] = len(token_ids)
                except Exception:
                    pass

            tokenize_thread = threading.Thread(target=_tokenize, daemon=True)
            tokenize_thread.start()
            tokenize_thread.join(3.0)
            prompt_tokens = int(outcome.get("tokens", 0))
        if prompt_tokens <= 0:
            # Tokenizer unavailable: rough character estimate keeps the
            # ledger conservative instead of blind.
            try:
                rendered = json.dumps(
                    openai_request.get("messages", []), ensure_ascii=False
                )
            except (TypeError, ValueError):
                rendered = ""
            prompt_tokens = max(1, len(rendered) // 3)
        max_tokens = int(openai_request.get("max_tokens") or 0)
        if max_tokens <= 0:
            ledger = _admission_ledger()
            cap = int(ledger.per_request_cap or 0)
            max_tokens = max(0, cap - prompt_tokens) if cap else 0
        return prompt_tokens + max_tokens

    class MeshWorkerHandler(BaseHTTPRequestHandler):
        server_version = "VerathosMeshWorker/0.1"
        # socketserver applies this as the per-operation timeout on every
        # accepted connection (StreamRequestHandler.setup calls
        # settimeout). http.server catches the resulting timeout in
        # handle_one_request and closes the connection, so idle
        # keep-alive reads, half-open clients, and sends to a client
        # that stopped reading all terminate the handler thread within a
        # bounded period instead of parking it forever - and do_POST's
        # finally has released any admission reservation on the way out.
        # Active SSE streams are untouched: the bound is per write, and
        # a client that is reading drains each write immediately.
        timeout = MESH_HTTP_SOCKET_TIMEOUT_S

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
            self._send_json_bytes(status_code, body)

        def _send_json_bytes(self, status_code: int, body: bytes) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_sse_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

        def _write_sse(self, payload: bytes) -> None:
            self.wfile.write(payload)
            self.wfile.flush()

        def _read_json_body(self) -> dict[str, Any]:
            raw = self._read_raw_body()
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _read_raw_body(self) -> bytes:
            cached = getattr(self, "_verathos_raw_body", None)
            if cached is not None:
                return cached
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported")
            get_all = getattr(self.headers, "get_all", None)
            lengths = get_all("Content-Length", []) if callable(get_all) else []
            if len(lengths) > 1:
                raise ValueError("duplicate Content-Length header")
            try:
                length = int(lengths[0] if lengths else self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("invalid Content-Length header") from None
            if length < 0:
                raise ValueError("invalid Content-Length header")
            if length > max_request_body_bytes:
                raise ValueError("request body exceeds configured limit")
            if (
                (urlparse(self.path).path.rstrip("/") or "/")
                == POSTCOMMIT_AUDIT_PATH
                and length > POSTCOMMIT_REQUEST_MAX_BODY_BYTES
            ):
                raise ValueError("postcommit request body exceeds configured limit")
            if length <= 0:
                self._verathos_raw_body = b""
                return b""
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("request body ended before Content-Length")
            self._verathos_raw_body = raw
            return raw

        def _authorize(self, *, method: str, raw_body: bytes) -> bool:
            signed_path = urlparse(self.path).path or "/"
            route_path = signed_path.rstrip("/") or "/"
            if is_public_route(method, route_path):
                return True
            if is_validator_route(method, route_path):
                try:
                    loopback_client = ipaddress.ip_address(
                        str(self.client_address[0]).split("%", 1)[0]
                    ).is_loopback
                except ValueError:
                    loopback_client = False
                # Pool operator self-tests are local and authenticate with the
                # mesh control HMAC. They never need a validator key, and remote
                # workers cannot use this alternate path because it is loopback
                # only. Try it before validator auth when its headers are present.
                if internal_auth_configured and loopback_client:
                    internal_result = verify_internal_http_request(
                        secret=internal_auth_secret,
                        method=method,
                        path=signed_path,
                        body=raw_body,
                        headers=self.headers,
                        replay_cache=internal_request_replays,
                    )
                    if internal_result.ok:
                        self._verathos_internal_principal = (
                            internal_result.principal or "mesh-internal"
                        )
                        return True
                if validator_auth_enabled:
                    assert validator_allowlist is not None
                    result = verify_validator_http_request(
                        method=method,
                        path=signed_path,
                        body=raw_body,
                        headers=self.headers,
                        allowlist=validator_allowlist,
                        replay_cache=validator_request_replays,
                    )
                    if not result.ok:
                        self._send_json(result.status_code, {"error": result.reason})
                        return False
                    self._verathos_validator_principal = result.principal or ""
                    return True
                if allow_loopback_dev_validator_routes and loopback_client:
                    return True
                self._send_json(403, {"error": "validator authentication is required"})
                return False
            if internal_auth_configured and not is_validator_route(method, route_path):
                result = verify_internal_http_request(
                    secret=internal_auth_secret,
                    method=method,
                    path=signed_path,
                    body=raw_body,
                    headers=self.headers,
                    replay_cache=internal_request_replays,
                )
                if not result.ok:
                    self._send_json(result.status_code, {"error": result.reason})
                    return False
            return True

        def _is_authenticated_validator_request(self) -> bool:
            return bool(
                validator_auth_enabled
                and effective_server_role() == "coordinator"
                and str(getattr(self, "_verathos_validator_principal", ""))
            )

        def _request_principal(self) -> str:
            return str(
                getattr(self, "_verathos_validator_principal", "")
                or getattr(self, "_verathos_internal_principal", "")
                or ""
            )

        def _send_snapshot_mismatch(self, *, stream: bool = False) -> None:
            """Report a pinned-snapshot refusal as a distinct, non-retryable class.

            Reported without the coordinator's own snapshot hash: the validator
            already knows which snapshot it pinned, and echoing the live one
            would let an unauthenticated caller enumerate rotations.
            """

            body = {
                "error": "request verification snapshot is not served here",
                "error_code": VERIFICATION_SNAPSHOT_MISMATCH_ERROR_CODE,
                "retryable": False,
            }
            if stream:
                try:
                    self._write_sse(
                        _sse_event_bytes(
                            "error", {"event": "error", **body}
                        )
                    )
                    self._write_sse(_sse_event_bytes("", "[DONE]"))
                except Exception:
                    pass
                return
            self._send_json(409, body)

        def _send_mesh_failure(
            self,
            exc: Exception,
            *,
            stream: bool = False,
        ) -> None:
            """Log private routing detail locally but never return it to validators."""

            logger.exception("mesh inference request failed", exc_info=exc)
            if isinstance(exc, _VerificationSnapshotMismatch):
                self._send_snapshot_mismatch(stream=stream)
                return
            if isinstance(exc, _CaptureWindowBusy):
                # The capture window did not drain within this lane's wait
                # budget. Every field below matches the admission ledger's
                # busy rejection EXACTLY: a refusal caused by an exclusive
                # replay must be indistinguishable from a saturation
                # refusal, or it hands callers an oracle for audit windows
                # (canary-oracle rule). Retryable: the window drains on its
                # own and a retry then verifies fine.
                busy_body = {
                    "error": (
                        f"all {int(llama_n_parallel)} generation "
                        "slots are busy; retry or fail over"
                    ),
                    "type": "slots_busy",
                    "retryable": True,
                }
                if stream:
                    # Headers are already on the wire, so the busy verdict
                    # rides an SSE error event instead of a 503 status.
                    try:
                        self._write_sse(
                            _sse_event_bytes(
                                "error", {"event": "error", **busy_body}
                            )
                        )
                        self._write_sse(_sse_event_bytes("", "[DONE]"))
                    except Exception:
                        pass
                    return
                self._send_json(503, busy_body)
                return
            if not stream and _BACKEND_EXHAUSTION_RE.search(str(exc)):
                # Defensive only: the admission ledger makes KV/slot
                # exhaustion structurally unreachable for admitted
                # requests, but if the backend still reports it (an
                # accounting hole, or internal work sharing the pool)
                # the caller must see a clean RETRYABLE busy - a bare
                # 500 here cost an epoch.
                # The nonce is already burnt; validator retries use a
                # fresh prompt + nonce, so this is safe pre-first-byte.
                # Mid-stream (stream=True) nothing better than the SSE
                # error exists.
                self._send_json(
                    503,
                    {
                        "error": (
                            "generation capacity is momentarily "
                            "exhausted; retry or fail over"
                        ),
                        "type": "slots_busy",
                        "retryable": True,
                    },
                )
                return
            public_error = (
                "mesh inference failed"
                if self._is_authenticated_validator_request()
                else str(exc)
            )
            if stream:
                try:
                    self._write_sse(
                        _sse_event_bytes(
                            "error",
                            {"event": "error", "error": public_error},
                        )
                    )
                    self._write_sse(_sse_event_bytes("", "[DONE]"))
                except Exception:
                    pass
                return
            self._send_json(500, {"error": public_error})

        def _prepare_request(self, method: str) -> bool:
            try:
                raw = self._read_raw_body() if method == "POST" else b""
            except ValueError as exc:
                status = 413 if "exceeds configured limit" in str(exc) else 400
                self._send_json(status, {"error": str(exc)})
                return False
            return self._authorize(method=method, raw_body=raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._prepare_request("GET"):
                return
            if self.path.rstrip("/") == "/v1/mesh/verification-snapshot":
                if verification_snapshot_loader is None:
                    self._send_json(404, {"error": "verification snapshot unavailable"})
                    return
                try:
                    snapshot = current_verification_snapshot()
                    self._send_json(200, snapshot.to_dict())
                except Exception as exc:
                    self._send_json(503, {"error": str(exc)})
                return
            blob_match = re.match(
                r"^/v1/mesh/proof-blob/([0-9a-f]{64})(\.i8|\.i8\.json|\.f32)$",
                self.path.rstrip("/"),
            )
            if blob_match:
                # Serve content-addressed proof-weight blobs to mesh members.
                # The client verifies the sha256 itself, so this route needs no
                # trust: a member without the GGUF fetches exactly its slice
                # and checks it against the manifest's committed hashes.
                from verallm.mesh.gguf_manifest import proof_weight_cache_dir

                sha, suffix = blob_match.group(1), blob_match.group(2)
                root = proof_weight_cache_dir()
                path = (root / sha[:2] / (sha + suffix)) if root is not None else None
                if (path is None or not path.is_file()) and proof_gguf_manifest_path:
                    # BUILD ON MISS: this node has the GGUF (it serves the
                    # manifest), so it can dequantize the one requested tensor,
                    # bank it, and serve it — instead of 404ing and stranding a
                    # file-less member whose fallback path (open the model
                    # file) cannot exist on its box. Pre-building every blob is
                    # not an option for 70B-class models (~1 byte/param of i8);
                    # lazily banking only sampled tensors keeps disk bounded
                    # by real usage. Serialized: concurrent misses would each
                    # pay the same multi-second dequant.
                    try:
                        from verallm.mesh.gguf_manifest import (
                            build_proof_blob_for_committed_sha,
                            load_gguf_tensor_manifest,
                        )

                        with _blob_build_lock:
                            # Re-check after acquiring: a concurrent miss for
                            # the same sha may have built it while we queued —
                            # skip re-paying the multi-second dequant.
                            if path is not None and path.is_file():
                                pass
                            else:
                                manifest = proof_gguf_manifest
                                if manifest is None:
                                    manifest = load_gguf_tensor_manifest(
                                        proof_gguf_manifest_path
                                    )
                                build_proof_blob_for_committed_sha(
                                    manifest,
                                    sha,
                                    suffix,
                                )
                    except Exception as exc:
                        self._send_json(500, {"error": f"blob build failed: {exc}"})
                        return
                if path is None or not path.is_file():
                    self._send_json(404, {"error": "blob not cached here"})
                    return
                size = path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                # Proof matrices can be hundreds of MiB. Stream the committed
                # cache file instead of duplicating it in the server's heap.
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        self.wfile.write(chunk)
                return
            if self.path.rstrip("/") == "/v1/mesh/manifest":
                # Serve the GGUF tensor manifest to FILE-LESS members: a machine
                # joining a mesh without the model needs only this small JSON
                # (plus on-demand proof blobs above) — never the GGUF itself.
                # Needs no trust: the fetcher checks the manifest's Merkle root
                # against the mesh spec it received via the join token.
                mpath = Path(proof_gguf_manifest_path) if proof_gguf_manifest_path else None
                if mpath is None or not mpath.is_file():
                    self._send_json(404, {"error": "no tensor manifest configured here"})
                    return
                try:
                    from verallm.mesh.gguf_manifest import (
                        load_gguf_tensor_manifest,
                        strip_gguf_manifest_runtime_paths,
                    )

                    manifest = proof_gguf_manifest
                    if manifest is None:
                        manifest = load_gguf_tensor_manifest(mpath)
                    raw = (
                        json.dumps(
                            strip_gguf_manifest_runtime_paths(manifest),
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n"
                    ).encode("utf-8")
                except Exception as exc:
                    self._send_json(500, {"error": f"manifest export failed: {exc}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            receipts_match = re.match(r"^/epoch/(\d+)/receipts$", self.path.rstrip("/"))
            if receipts_match:
                try:
                    epoch = int(receipts_match.group(1))
                    receipts = _mesh_receipt_store().get(epoch)
                    self._send_json(200, {
                        "epoch": epoch,
                        "receipt_count": len(receipts),
                        "receipts": receipts,
                    })
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
                return
            if self.path.rstrip("/") == "/health":
                warm_state = str(proof_cache_warm_state.get("state", ""))
                if proof_cache_warm_required and warm_state != "ready":
                    self._send_json(
                        503,
                        {
                            "status": "warming" if warm_state == "warming" else "error",
                            "service": "verathos-mesh-worker",
                            "proof_cache_warm_state": warm_state,
                            **(
                                {
                                    "error": (
                                        "proof-weight cache prewarm failed; "
                                        "see worker logs"
                                    )
                                }
                                if warm_state == "failed"
                                else {}
                            ),
                        },
                    )
                    return
                spec = current_mesh_spec()
                mesh_spec_hash = spec.spec_hash_hex() if spec else ""
                stage_assignment_hash = spec.stage_assignment_hash_hex() if spec else ""
                configured_proof_bps = max(
                    int(proof_sample_bps),
                    int(decode_audit_bps),
                )
                if not require_proof or configured_proof_bps <= 0:
                    configured_challenge_kind = "disabled"
                elif (
                    effective_server_role() == "coordinator"
                    and validator_auth_enabled
                    and require_validator_nonce
                    and verification_snapshot_loader is not None
                ):
                    configured_challenge_kind = (
                        VALIDATOR_POSTCOMMIT_CHALLENGE_KIND
                    )
                elif defer_proof:
                    configured_challenge_kind = "deferred_future_randomness_v1"
                elif configured_proof_bps >= PROOF_SAMPLE_BPS_DENOMINATOR:
                    configured_challenge_kind = "inline_every_request_v1"
                else:
                    configured_challenge_kind = "fiat_shamir_inline_v1"
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "verathos-mesh-worker",
                        "version": 1,
                        "time_unix": int(time.time()),
                        "backend_configured": bool(backend_url),
                        "proof_adapter_configured": bool(proof_url),
                        "embedded_proof_configured": bool(proof_trace_root),
                        "proof_warmup_ms": round(proof_warmup_ms, 3),
                        "proof_decode_projection_warmup_ms": round(
                            proof_decode_projection_warmup_ms,
                            3,
                        ),
                        "proof_cache_warm_state": warm_state,
                        "proof_required": bool(require_proof),
                        "defer_proof": bool(defer_proof),
                        "proof_sample_bps": int(proof_sample_bps),
                        "proof_sample_denominator": PROOF_SAMPLE_BPS_DENOMINATOR,
                        "proof_ops_per_request": int(proof_ops_per_request),
                        "proof_trace_candidates_per_request": int(
                            proof_trace_candidates_per_request
                        ),
                        "proof_artifact_timeout": float(proof_artifact_timeout),
                        "proof_challenge_kind": configured_challenge_kind,
                        "decode_audit_bps": int(decode_audit_bps),
                        "decode_audit_top_k": int(decode_audit_top_k),
                        "mesh_spec_hash": mesh_spec_hash,
                        "stage_assignment_hash": stage_assignment_hash,
                    },
                )
                return

            if self.path.rstrip("/") == "/capability":
                payload = {
                    "capability": capability.to_dict(),
                    "capability_hash": capability.ad_hash_hex(),
                    "body_hash": capability.body_hash_hex(),
                }
                # Admission diagnostics ride this route ONLY behind the
                # internal HMAC wall (pool operator): with internal auth
                # configured, _authorize has already verified the signed
                # request before we get here. Never expose the ledger on
                # an open route - real in-flight occupancy next to the
                # drain-file busy masking would hand validators an
                # oracle for capacity-audit windows.
                if internal_auth_configured:
                    ledger = _admission_state.get("ledger")
                    if ledger is not None:
                        payload["admission"] = ledger.snapshot()
                self._send_json(200, payload)
                return

            if self.path.rstrip("/") == "/capacity/roster":
                # Public pull route for the signed capacity roster: how a
                # validator learns which physical GPUs stand behind this
                # mesh's chain entry WITHOUT waiting for a first audit
                # receipt. The document is public evidence (worker ids, GPU
                # names, /24 hints — never full member addresses) and its
                # authenticity rides the embedded coordinator-EVM signature,
                # so no auth wall is needed.
                payload = None
                if capacity_roster_file:
                    try:
                        payload = json.loads(
                            Path(capacity_roster_file).read_text()
                        )
                    except Exception:
                        payload = None
                if (
                    isinstance(payload, dict)
                    and payload.get("roster")
                    and payload.get("roster_signature")
                ):
                    self._send_json(200, payload)
                else:
                    self._send_json(404, {"error": "not found"})
                return

            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._prepare_request("POST"):
                return
            # Ledger reservations are released HERE: every route path
            # (success, error, mid-stream abort, socket timeout) ends
            # this method, and a leaked reservation would permanently
            # shrink the pool. The guard releases exactly once from
            # whichever exit runs first; finish() repeats the call at
            # connection close for exits that escape this finally, and
            # the dead-owner sweep in service_actions is the last
            # backstop for a thread that dies holding its reservation.
            self._admission_guard = None
            try:
                self._do_post_routes()
            finally:
                guard = getattr(self, "_admission_guard", None)
                self._admission_guard = None
                if guard is not None:
                    guard.release()

        def finish(self) -> None:
            # Connection-teardown backstop for the admission guard: if a
            # request thread unwound past do_POST's finally without
            # releasing (a thread-killing error inside that finally),
            # the reservation still dies with the connection. Idempotent
            # by construction - in the normal lifecycle the guard was
            # already released and cleared per request.
            guard = getattr(self, "_admission_guard", None)
            self._admission_guard = None
            if guard is not None:
                try:
                    if guard.release():
                        logger.error(
                            "admission reservation released at connection "
                            "close instead of request end (tokens=%s); "
                            "request-path release was skipped",
                            guard.tokens,
                        )
                except Exception:
                    logger.exception(
                        "admission release failed at connection close"
                    )
            super().finish()

        def _admit_or_reject(self, openai_request: dict[str, Any]) -> bool:
            """Uniform admission; sends the busy 503 itself on refusal.

            Runs BEFORE any nonce claim or SSE byte so a refusal is a
            clean precommit-free busy signal (validator busy machinery /
            router failover). OVERSIZED gets a uniform 400: the
            coordinator owns per-request contract enforcement now that
            the unified KV is sized beyond the contract.
            """

            if capacity_drain_file:
                from verallm.mesh.capacity_audit_worker import (
                    capacity_drain_active,
                )

                if capacity_drain_active(capacity_drain_file):
                    # Every field below matches the ledger's busy rejection
                    # EXACTLY. Anything distinguishable (an extra field, a
                    # different message, an audit id) would hand validators
                    # an oracle for audit windows.
                    self._send_json(
                        503,
                        {
                            "error": (
                                f"all {int(llama_n_parallel)} generation "
                                "slots are busy; retry or fail over"
                            ),
                            "type": "slots_busy",
                            "retryable": True,
                        },
                    )
                    return False
            demand = admission_demand_tokens(openai_request)
            ledger = _admission_ledger()
            verdict = ledger.try_admit(demand)
            if verdict is Admission.BUSY:
                self._send_json(
                    503,
                    {
                        "error": (
                            f"all {int(llama_n_parallel)} generation "
                            "slots are busy; retry or fail over"
                        ),
                        "type": "slots_busy",
                        "retryable": True,
                    },
                )
                return False
            if verdict is Admission.OVERSIZED:
                # OVERSIZED now means the demand can NEVER fit the
                # launched unified KV pool (the per-request cap is the
                # physical serving budget; the registered contract is a
                # scoring concern, not admission's). Refusing here keeps
                # a never-fits request from consuming KV the ledger never
                # accounted (the mid-flight exhaustion class through the
                # back door). Uniform for every caller.
                self._send_json(
                    400,
                    {
                        "error": (
                            "request exceeds the serving context "
                            f"budget ({ledger.per_request_cap} tokens)"
                        ),
                        "type": "exceed_context_size_error",
                    },
                )
                return False
            self._admission_guard = ReservationGuard(ledger, demand)
            return True

        def _do_post_routes(self) -> None:
            if self.path.rstrip("/") == "/identity/challenge":
                if not evm_address or (
                    not evm_private_key and evm_challenge_signer is None
                ):
                    self._send_json(
                        501,
                        {"error": "identity challenge is not configured"},
                    )
                    return
                try:
                    nonce_text = self._read_json_body().get("nonce", "")
                    nonce = bytes.fromhex(str(nonce_text))
                    if len(nonce) != 32:
                        raise ValueError("nonce must be 32 bytes encoded as hex")
                    if evm_private_key:
                        from eth_account import Account
                        from eth_account.messages import encode_defunct

                        address_bytes = bytes.fromhex(evm_address[2:])
                        signed = Account.sign_message(
                            encode_defunct(primitive=nonce + address_bytes),
                            private_key=evm_private_key,
                        )
                        signature_hex = signed.signature.hex()
                    else:
                        # Wallet-less driver: the pool manager signs the
                        # same EIP-191 payload with the coordinator EVM key.
                        try:
                            signature_hex = evm_challenge_signer(nonce)
                        except Exception as exc:
                            self._send_json(
                                502,
                                {
                                    "error": (
                                        "delegated identity signing failed: "
                                        f"{exc}"
                                    )
                                },
                            )
                            return
                    self._send_json(
                        200,
                        {
                            "address": evm_address,
                            "signature": signature_hex,
                        },
                    )
                except (TypeError, ValueError) as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if self.path.rstrip("/") == "/epoch/receipt":
                try:
                    body = self._read_json_body()
                    store = _mesh_receipt_store()
                    if validator_auth_enabled:
                        epoch, canonical = validate_authenticated_service_receipt(
                            body,
                            principal=str(
                                getattr(
                                    self,
                                    "_verathos_validator_principal",
                                    "",
                                )
                            ),
                        )
                        count, added = store.add_unique(
                            epoch,
                            canonical,
                            unique_fields=("validator_signature",),
                        )
                    else:
                        epoch = int(body.get("epoch_number"))
                        if not body.get("validator_signature") or not body.get(
                            "miner_address"
                        ):
                            self._send_json(
                                400,
                                {
                                    "error": (
                                        "receipt missing validator_signature "
                                        "or miner_address"
                                    )
                                },
                            )
                            return
                        canonical = body
                        count = store.add(epoch, canonical)
                        added = True
                    store.gc(epoch)
                    self._send_json(
                        200,
                        {
                            "status": "accepted" if added else "duplicate",
                            "epoch": epoch,
                            "count": count,
                        },
                    )
                except PermissionError as e:
                    self._send_json(403, {"error": f"invalid receipt: {e}"})
                except Exception as e:
                    self._send_json(400, {"error": f"invalid receipt: {e}"})
                return
            if self.path.rstrip("/") == "/v1/mesh/join":
                if join_handler is None:
                    self._send_json(404, {"error": "join endpoint disabled"})
                    return
                try:
                    self._send_json(200, join_handler(self._read_json_body()))
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/spec":
                if mesh_spec_handler is None:
                    self._send_json(404, {"error": "mesh spec endpoint disabled"})
                    return
                try:
                    self._send_json(200, mesh_spec_handler(self._read_json_body()))
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/update":
                if mesh_update_handler is None:
                    self._send_json(404, {"error": "mesh update endpoint disabled"})
                    return
                try:
                    self._send_json(200, mesh_update_handler(self._read_json_body()))
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/trace-capture":
                try:
                    body = self._read_json_body()
                    spec = current_mesh_spec()
                    if spec and body.get("mesh_spec_hash") != spec.spec_hash_hex():
                        self._send_json(409, {"error": "mesh_spec_hash mismatch"})
                        return
                    if proof_trace_enable_path is None:
                        self._send_json(409, {"error": "trace capture is not configured"})
                        return
                    enabled = bool(body.get("enabled", False))
                    raw_selected = body.get("selected_manifest_indexes", [])
                    if raw_selected is None:
                        raw_selected = []
                    if not isinstance(raw_selected, list):
                        self._send_json(
                            400,
                            {"error": "selected_manifest_indexes must be a list"},
                        )
                        return
                    raw_ops = body.get("selected_ops", [])
                    if raw_ops is None:
                        raw_ops = []
                    if not isinstance(raw_ops, list):
                        self._send_json(400, {"error": "selected_ops must be a list"})
                        return
                    selected = [int(item) for item in raw_selected]
                    selected_ops = [str(item) for item in raw_ops]
                    raw_anchor_rows = body.get("selected_anchor_rows", [])
                    if raw_anchor_rows is None:
                        raw_anchor_rows = []
                    if not isinstance(raw_anchor_rows, list):
                        self._send_json(
                            400,
                            {"error": "selected_anchor_rows must be a list"},
                        )
                        return
                    selected_anchor_rows = [
                        int(item) for item in raw_anchor_rows
                    ]
                    anchor_beacon = str(body.get("anchor_beacon", "") or "")
                    if enabled and anchor_beacon and not selected_anchor_rows:
                        # The coordinator never learns this member's anchor
                        # commitments; the member derives its own audit rows
                        # from the shared beacon against its local streams.
                        selected_anchor_rows = anchor_rows_to_arm(
                            {"proof_beacon": anchor_beacon}
                        )
                    try:
                        set_local_trace_capture(
                            enabled,
                            request_id=str(body.get("request_id", "")),
                            mode=str(body.get("mode", "") or ""),
                            window_token=str(body.get("window_token", "") or ""),
                            selected_manifest_indexes=selected,
                            selected_ops=selected_ops,
                            selected_anchor_rows=selected_anchor_rows,
                        )
                    except RuntimeError as exc:
                        if "busy" in str(exc):
                            self._send_json(409, {"error": str(exc)})
                            return
                        raise
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "capture_enabled": enabled and proof_trace_enable_path.exists(),
                            "request_id": str(body.get("request_id", "")),
                            "selected_manifest_indexes": selected if enabled else [],
                            "selected_ops": selected_ops if enabled else [],
                        },
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/proof/receipt":
                try:
                    body = self._read_json_body()
                    ctx = body.get("receipt_context", {})
                    if not isinstance(ctx, dict):
                        self._send_json(400, {"error": "receipt_context must be an object"})
                        return
                    if proof_trace_root is None:
                        self._send_json(409, {"error": "embedded proof is not configured"})
                        return
                    self._send_json(
                        200,
                        make_embedded_proof_payload(
                            ctx,
                            include_proof=bool(body.get("include_proof")),
                            proof_required=True,
                            openai_request=body.get("openai_request")
                            if isinstance(body.get("openai_request"), dict)
                            else None,
                            openai_response=body.get("openai_response")
                            if isinstance(body.get("openai_response"), dict)
                            else None,
                            slot_view_context=body.get("slot_view_context")
                            if isinstance(body.get("slot_view_context"), dict)
                            else None,
                        ),
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/proof/commitment":
                try:
                    body = self._read_json_body()
                    ctx = body.get("receipt_context", {})
                    if not isinstance(ctx, dict):
                        self._send_json(400, {"error": "receipt_context must be an object"})
                        return
                    if proof_trace_root is None:
                        self._send_json(409, {"error": "embedded proof is not configured"})
                        return
                    self._send_json(
                        200,
                        make_embedded_trace_commitment_payload(
                            ctx,
                            body.get("slot_view_context")
                            if isinstance(body.get("slot_view_context"), dict)
                            else None,
                        ),
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/proof/selection":
                try:
                    body = self._read_json_body()
                    ctx = body.get("receipt_context", {})
                    if not isinstance(ctx, dict):
                        self._send_json(400, {"error": "receipt_context must be an object"})
                        return
                    if proof_trace_root is None:
                        self._send_json(409, {"error": "embedded proof is not configured"})
                        return
                    self._send_json(
                        200,
                        make_embedded_proof_selection_payload(
                            ctx,
                            body.get("slot_view_context")
                            if isinstance(body.get("slot_view_context"), dict)
                            else None,
                            serve_n_parallel=int(
                                body.get("serve_n_parallel", 1) or 1
                            ),
                        ),
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == POSTCOMMIT_AUDIT_PATH:
                if not self._is_authenticated_validator_request():
                    self._send_json(
                        403,
                        {"error": "validator authentication is required"},
                    )
                    return
                principal = self._request_principal()
                request_slot_acquired = False
                try:
                    acquire_postcommit_request_slot(principal)
                    request_slot_acquired = True
                    self._send_json_bytes(
                        200,
                        make_postcommit_audit_artifact(
                            self._read_json_body(),
                            principal=principal,
                        ),
                    )
                except _PostcommitFinalizationInProgress:
                    self._send_json(
                        503,
                        {
                            "error": "postcommit finalization is in progress",
                            "error_code": "postcommit_finalization_in_progress",
                            "retryable": True,
                        },
                    )
                except _PostcommitCapacityUnavailable:
                    self._send_json(
                        503,
                        {
                            "error": "postcommit capacity is temporarily unavailable",
                            "error_code": "postcommit_capacity_unavailable",
                            "retryable": True,
                        },
                    )
                except Exception as exc:
                    self._send_mesh_failure(exc)
                finally:
                    if request_slot_acquired:
                        release_postcommit_request_slot(principal)
                return

            if self.path.rstrip("/") == "/v1/mesh/proof/deferred-audit":
                try:
                    self._send_json(
                        200,
                        make_deferred_audit_bundle_payload(self._read_json_body()),
                    )
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if self.path.rstrip("/") == "/v1/mesh/inference":
                try:
                    body = self._read_json_body()
                    spec = current_mesh_spec()
                    external_validator_request = (
                        self._is_authenticated_validator_request()
                    )
                    if external_validator_request and "mesh_spec_hash" in body:
                        self._send_json(
                            400,
                            {"error": "mesh_spec_hash is internal-only"},
                        )
                        return
                    if (
                        not external_validator_request
                        and spec
                        and body.get("mesh_spec_hash") != spec.spec_hash_hex()
                    ):
                        self._send_json(409, {"error": "mesh_spec_hash mismatch"})
                        return
                    if (
                        external_validator_request
                        and verification_snapshot_loader is None
                    ):
                        self._send_json(
                            503,
                            {
                                "error": (
                                    "authenticated mesh inference requires "
                                    "a verification snapshot"
                                )
                            },
                        )
                        return
                    pinned_snapshot = (
                        current_verification_snapshot(spec)
                        if external_validator_request
                        and verification_snapshot_loader is not None
                        else None
                    )
                    postcommit_validator_request = bool(
                        external_validator_request
                    )
                    openai_request = body.get("openai_request")
                    if not isinstance(openai_request, dict):
                        self._send_json(400, {"error": "openai_request must be an object"})
                        return
                    # Same uniform admission as the chat route - this
                    # route previously had NONE and llama broke overflow
                    # streams mid-flight. Before the nonce claim: a busy
                    # 503 must stay a precommit-free signal.
                    if not self._admit_or_reject(openai_request):
                        return
                    if require_validator_nonce:
                        try:
                            if postcommit_validator_request:
                                claim_fresh_validator_challenge_commitment(
                                    openai_request,
                                    principal=self._request_principal(),
                                )
                            else:
                                claim_fresh_validator_nonce(
                                    openai_request,
                                    principal=(
                                        self._request_principal()
                                        or "unauthenticated"
                                    ),
                                )
                        except ValueError as exc:
                            self._send_json(400, {"error": str(exc)})
                            return
                    try:
                        request_id = mesh_request_id_from_request(
                            openai_request,
                            require_validator_request_id=(
                                external_validator_request
                            ),
                            fallback=str(body.get("request_id") or ""),
                        )
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    policy = proof_policy_context(
                        openai_request,
                        requested_require_proof=bool(body.get("require_proof")),
                        pinned_verification_snapshot=pinned_snapshot,
                        validator_principal=self._request_principal(),
                        require_postcommit=postcommit_validator_request,
                        validator_authenticated=(
                            self._is_authenticated_validator_request()
                        ),
                    )
                    replay_seed = bind_replay_seed_policy(
                        policy,
                        openai_request,
                        request_id,
                    )
                    sampled_profile = finalize_verified_sampler_policy(
                        policy,
                        openai_request,
                        request_id,
                    )
                    backend_request = backend_openai_request(
                        openai_request,
                        proof_capture_required=policy["proof_capture_required"],
                        verified_sampler_required=policy["verified_sampler_required"],
                        proof_metadata_required=policy["proof_metadata_required"],
                        replay_seed=replay_seed,
                        sampled_profile=sampled_profile,
                    )
                    prompt_ids, prompt_source, prompt_root = prompt_binding_for_request(
                        openai_request,
                        proof_metadata_required=policy["proof_metadata_required"],
                    )
                    capture_info: dict[str, Any] = {}
                    skip_trace_capture = organic_serve_skip_trace_capture(
                        policy,
                        spec,
                        validator_authenticated=(
                            self._is_authenticated_validator_request()
                        ),
                    )
                    started_ns = time.time_ns()
                    raw_response = forward_to_backend(
                        backend_request,
                        spec=spec,
                        request_id=request_id,
                        proof_capture_required=policy["proof_capture_required"],
                        capture_info=capture_info,
                        skip_trace_capture=skip_trace_capture,
                    )
                    ended_ns = time.time_ns()
                    response, token_ids, token_source = prepare_backend_response_for_receipt(
                        raw_response,
                        proof_capture_required=policy["proof_capture_required"],
                        stream=False,
                    )
                    if (
                        policy["proof_metadata_required"]
                        and not policy.get("decode_audit_configured")
                        and not token_source
                    ):
                        token_ids, token_source = fetch_completion_token_ids_from_backend(
                            backend_url,
                            response,
                        )
                    member = None
                    if spec:
                        member = next(
                            (item for item in spec.members if item.endpoint == capability.endpoint),
                            None,
                        )
                    receipt, proof_receipts, proof_payloads = receipt_for(
                        request_id=request_id,
                        openai_request=openai_request,
                        openai_response=response,
                        prompt_token_ids=prompt_ids,
                        prompt_token_source=prompt_source,
                        prompt_template_root=prompt_root,
                        completion_token_ids=token_ids,
                        completion_token_source=token_source,
                        stage_index=member.stage_index if member else 0,
                        layer_start=member.layers.start if member else 0,
                        layer_end=member.layers.end if member else 0,
                        spec=spec,
                        proof_required=policy["proof_required"],
                        proof_policy=policy,
                        inference_started_unix_ns=started_ns,
                        inference_ended_unix_ns=ended_ns,
                        slot_id=slot_id_from_backend_response(raw_response),
                        capture_window_token=capture_info.get("window_token", ""),
                        pinned_verification_snapshot=pinned_snapshot,
                    )
                    maybe_save_slot_state(
                        request_id=request_id,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=token_ids,
                        slot_id=slot_id_from_backend_response(raw_response),
                        validator_authenticated=(
                            self._is_authenticated_validator_request()
                        ),
                    )
                    payload = {"response": response, "receipt": receipt}
                    if prompt_source:
                        payload["prompt_token_ids"] = prompt_ids
                    payload["completion_token_ids"] = token_ids
                    if proof_receipts:
                        payload["proof_receipts"] = proof_receipts
                    if proof_payloads:
                        payload["proof_payloads"] = proof_payloads
                    if external_validator_request:
                        assert_validator_artifact_privacy(payload, spec=spec)
                    if postcommit_validator_request:
                        store_postcommit_origin(
                            payload,
                            openai_request,
                            principal=self._request_principal(),
                            spec=spec,
                            verification_snapshot=pinned_snapshot,
                        )
                    self._send_json(200, payload)
                except Exception as exc:
                    self._send_mesh_failure(exc)
                return

            if self.path.rstrip("/") == "/v1/chat/completions":
                body: dict[str, Any] = {}
                # Token-ledger admission BEFORE any nonce is claimed (a
                # busy 503 must never burn a validator nonce). Over-LIMIT
                # prompts are a different, permanent failure: a uniform
                # 400 at admission (context shift stays off so verified
                # traffic is never silently truncated).
                try:
                    body = self._read_json_body()
                    if not self._admit_or_reject(body):
                        return
                    authenticated_validator = (
                        self._is_authenticated_validator_request()
                    )
                    verathos_request = body.get("verathos", {})
                    requested_snapshot_hash = (
                        str(
                            verathos_request.get(
                                "verification_snapshot_hash",
                                "",
                            )
                            or ""
                        )
                        if isinstance(verathos_request, dict)
                        else ""
                    )
                    # Local pool self-tests use HMAC instead of validator auth,
                    # but their requested snapshot still has to stay fixed from
                    # policy selection through proof-receipt aggregation.
                    internal_operator_snapshot_request = bool(
                        requested_snapshot_hash
                        and str(
                            getattr(
                                self,
                                "_verathos_internal_principal",
                                "",
                            )
                            or ""
                        )
                    )
                    snapshot_bound_request = bool(
                        authenticated_validator
                        or internal_operator_snapshot_request
                    )
                    if (
                        snapshot_bound_request
                        and verification_snapshot_loader is None
                    ):
                        self._send_json(
                            503,
                            {
                                "error": (
                                    (
                                        "authenticated mesh inference requires "
                                        if authenticated_validator
                                        else "snapshot-bound mesh inference requires "
                                    )
                                    + "a verification snapshot"
                                )
                            },
                        )
                        return
                    pinned_spec = (
                        current_mesh_spec()
                        if snapshot_bound_request
                        else None
                    )
                    pinned_snapshot = (
                        current_verification_snapshot(pinned_spec)
                        if snapshot_bound_request
                        and verification_snapshot_loader is not None
                        else None
                    )
                    postcommit_validator_request = bool(
                        authenticated_validator
                    )
                    if require_validator_nonce:
                        try:
                            if postcommit_validator_request:
                                claim_fresh_validator_challenge_commitment(
                                    body,
                                    principal=self._request_principal(),
                                )
                            else:
                                claim_fresh_validator_nonce(
                                    body,
                                    principal=(
                                        self._request_principal()
                                        or "unauthenticated"
                                    ),
                                )
                        except ValueError as exc:
                            self._send_json(400, {"error": str(exc)})
                            return
                    if authenticated_validator:
                        try:
                            mesh_request_id_from_request(
                                body,
                                require_validator_request_id=True,
                            )
                        except ValueError as exc:
                            self._send_json(400, {"error": str(exc)})
                            return
                    if body.get("stream"):
                        self._send_sse_headers()

                        def emit_raw_sse(payload: bytes) -> None:
                            self._write_sse(payload)

                        routed = route_to_mesh_stream(
                            body,
                            emit_raw_sse,
                            require_validator_request_id=(
                                authenticated_validator
                            ),
                            validator_principal=self._request_principal(),
                            require_postcommit=postcommit_validator_request,
                            validator_authenticated=authenticated_validator,
                            pinned_mesh_spec=pinned_spec,
                            pinned_verification_snapshot=pinned_snapshot,
                        )
                        if authenticated_validator:
                            assert_validator_artifact_privacy(
                                routed,
                                spec=pinned_spec,
                            )
                        if postcommit_validator_request:
                            store_postcommit_origin(
                                routed,
                                body,
                                principal=self._request_principal(),
                                spec=pinned_spec,
                                verification_snapshot=pinned_snapshot,
                            )
                        self._write_sse(
                            _sse_event_bytes(
                                "done",
                                {
                                    "event": "done",
                                    "response": routed["response"],
                                    "verathos_mesh": verathos_mesh_metadata(
                                        routed,
                                        include_proof_payloads=(
                                            authenticated_validator
                                            and not postcommit_validator_request
                                        ),
                                    ),
                                },
                            )
                        )
                        self._write_sse(_sse_event_bytes("", "[DONE]"))
                        return
                    routed = route_to_mesh(
                        body,
                        require_validator_request_id=(
                            authenticated_validator
                        ),
                        validator_principal=self._request_principal(),
                        require_postcommit=postcommit_validator_request,
                        validator_authenticated=authenticated_validator,
                        pinned_mesh_spec=pinned_spec,
                        pinned_verification_snapshot=pinned_snapshot,
                    )
                    if authenticated_validator:
                        assert_validator_artifact_privacy(
                            routed,
                            spec=pinned_spec,
                        )
                    if postcommit_validator_request:
                        store_postcommit_origin(
                            routed,
                            body,
                            principal=self._request_principal(),
                            spec=pinned_spec,
                            verification_snapshot=pinned_snapshot,
                        )
                    response = dict(routed["response"])
                    response["verathos_mesh"] = verathos_mesh_metadata(
                        routed,
                        include_proof_payloads=(
                            authenticated_validator
                            and not postcommit_validator_request
                        ),
                    )
                    self._send_json(200, response)
                except Exception as exc:
                    self._send_mesh_failure(
                        exc,
                        stream=bool(body.get("stream")),
                    )
                return

            if self.path.rstrip("/") != "/v1/stage/handshake":
                self._send_json(404, {"error": "not found"})
                return

            try:
                body = self._read_json_body()
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
                return

            requested_mesh_hash = str(body.get("mesh_spec_hash", ""))
            spec = current_mesh_spec()
            mesh_spec_hash = spec.spec_hash_hex() if spec else ""
            stage_assignment_hash = spec.stage_assignment_hash_hex() if spec else ""
            if mesh_spec_hash and requested_mesh_hash != mesh_spec_hash:
                self._send_json(
                    409,
                    {"error": "mesh_spec_hash mismatch"},
                )
                return

            self._send_json(
                200,
                {
                    "status": "accepted",
                    "service": "verathos-mesh-worker",
                    "version": 1,
                    "capability_hash": capability.ad_hash_hex(),
                    "mesh_spec_hash": mesh_spec_hash,
                    "stage_assignment_hash": stage_assignment_hash,
                },
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def log_error(self, fmt: str, *args: Any) -> None:
            # log_message is silenced (per-request noise), but handler
            # errors - a socket timeout discarding a wedged connection -
            # must stay visible in the serve log.
            logger.warning("mesh http handler: " + fmt, *args)

    class MeshWorkerHTTPServer(ThreadingHTTPServer):
        """ThreadingHTTPServer plus the admission-ledger leak sweep.

        service_actions runs inside serve_forever's existing poll loop,
        so the sweep needs no extra thread and stops with the server. It
        is defense in depth: with socket timeouts and per-request
        release guards in place it should never fire, and every hit is
        logged at ERROR because it marks a release-path bug that would
        previously have leaked pool capacity until relaunch.
        """

        daemon_threads = True
        admission_reap_interval_s = 15.0
        _admission_reap_due = 0.0

        def server_close(self) -> None:
            # The janitor is scoped to this server.  Leaving its daemon
            # thread alive across relaunches accumulates stale workers in
            # long-running managers and aggregate tests.
            proof_trace_janitor_stop.set()
            super().server_close()
            if (
                proof_trace_janitor_thread is not None
                and proof_trace_janitor_thread.is_alive()
                and proof_trace_janitor_thread is not threading.current_thread()
            ):
                proof_trace_janitor_thread.join(timeout=1.0)

        def service_actions(self) -> None:
            super().service_actions()
            now = time.monotonic()
            if now < self._admission_reap_due:
                return
            self._admission_reap_due = now + float(
                self.admission_reap_interval_s
            )
            ledger = _admission_state.get("ledger")
            if ledger is None:
                return
            for leak in ledger.reap_dead_owners():
                logger.error(
                    "admission reservation leaked by a dead handler "
                    "thread; released by sweep: tokens=%s owner=%s "
                    "held_seconds=%s",
                    leak.get("tokens"),
                    leak.get("owner"),
                    leak.get("held_seconds"),
                )

    server = MeshWorkerHTTPServer((host, int(port)), MeshWorkerHandler)
    server.daemon_threads = True
    server.verathos_proof_trace_janitor_stop = proof_trace_janitor_stop
    server.verathos_proof_trace_janitor_thread = proof_trace_janitor_thread
    if proof_trace_janitor_thread is not None:
        proof_trace_janitor_thread.start()
    # Test/diagnostic accessor: resolves (and lazily creates) this
    # serve's admission ledger without going through a request.
    server.verathos_admission_ledger = _admission_ledger
    server.verathos_slot_view_template_warmup = bool(
        slot_view_template_warmup
        and backend_url
        and require_proof
        and slot_view_required
        and proof_trace_root is not None
    )
    server.verathos_mesh_spec_loader = current_mesh_spec
    server.verathos_proof_cache_warm_required = proof_cache_warm_required
    server.verathos_proof_cache_warm_state = proof_cache_warm_state
    # Keep the proof collector on the server instance so failure/retry tests
    # can inject a bounded fault into one coordinator without mutating global
    # module state or affecting another concurrently running server.
    server.verathos_proof_receipt_collector = collect_proof_receipts
    # Test/diagnostic accessors for the bounded-audit-window machinery:
    # closure state that unit tests must reach without HTTP choreography.
    server.verathos_organic_capture_skip = organic_serve_skip_trace_capture
    server.verathos_can_skip_organic_capture = can_skip_organic_slot_view_capture
    server.verathos_slot_view_template_cache = slot_view_template_cache
    server.verathos_strict_quiesce = strict_quiesce_event
    server.verathos_organic_inflight = organic_inflight_state
    server.verathos_organic_inflight_tracked = _organic_inflight_tracked
    server.verathos_organic_inflight_cond = organic_inflight_cond
    server.verathos_wait_out_strict_quiesce = _wait_out_strict_quiesce
    server.verathos_save_slot_state = maybe_save_slot_state
    server.verathos_restore_slot_state = _maybe_restore_slot_state_for_probes
    server.verathos_sweep_slot_states = _sweep_slot_states
    return server


def serve_worker(
    *,
    capability: CapabilityAd,
    host: str = "0.0.0.0",
    port: int = DEFAULT_WORKER_PORT,
    mesh_spec: MeshSpec | None = None,
    mesh_spec_loader: MeshSpecLoader | None = None,
    join_handler: JoinHandler | None = None,
    mesh_spec_handler: MeshSpecHandler | None = None,
    mesh_update_handler: MeshUpdateHandler | None = None,
    backend_url: str = "",
    proof_url: str = "",
    require_proof: bool = False,
    proof_trace_enable_file: str | Path = "",
    proof_trace_dir: str | Path = "",
    proof_gguf_manifest_path: str | Path = "",
    proof_tolerance_abs: float = 8e-2,
    proof_tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    proof_spot_checks: int = 8,
    proof_warmup: bool = False,
    proof_decode_projection_warmup: bool = False,
    proof_sample_bps: int = PROOF_SAMPLE_BPS_DENOMINATOR,
    defer_proof: bool = False,
    proof_ops_per_request: int = 1,
    proof_trace_candidates_per_request: int = 8,
    decode_audit_bps: int = 0,
    decode_audit_top_k: int = 8,
    proof_artifact_timeout: float = DEFAULT_PROOF_ARTIFACT_TIMEOUT,
    prompt_binding_cache_enabled: bool = False,
    llama_n_parallel: int = 1,
    llama_n_ubatch: int = 0,
    slot_view_template_warmup: bool = False,
    local_stage_capture: bool = False,
    proof_trace_manifest_format: str = "",
    receipt_signer: "Callable[[str], str] | None" = None,
    stage_receipt_signer: "Callable[[str], str] | None" = None,
    stage_proof_key: str = "",
    server_role: str = "auto",
    validator_auth_enabled: bool = False,
    validator_allowlist_path: str | Path = "",
    validator_allowlist_max_age_seconds: float = (
        DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS
    ),
    require_validator_nonce: bool = False,
    allow_loopback_dev_validator_routes: bool = False,
    internal_auth_secret: str | bytes = "",
    evm_address: str = "",
    evm_private_key: str = "",
    evm_challenge_signer: Callable[[bytes], str] | None = None,
    llama_ctx_budget: int = 0,
    capacity_drain_file: str | Path = "",
    slot_state_dir: str | Path = "",
    capacity_roster_file: str | Path = "",
    max_request_body_bytes: int = 128 * 1024 * 1024,
    verification_snapshot_loader: VerificationSnapshotLoader | None = None,
) -> None:
    """Run a worker server until interrupted."""

    server = make_worker_server(
        receipt_signer=receipt_signer,
        stage_receipt_signer=stage_receipt_signer,
        stage_proof_key=stage_proof_key,
        server_role=server_role,
        validator_auth_enabled=validator_auth_enabled,
        validator_allowlist_path=validator_allowlist_path,
        validator_allowlist_max_age_seconds=(
            validator_allowlist_max_age_seconds
        ),
        require_validator_nonce=require_validator_nonce,
        allow_loopback_dev_validator_routes=allow_loopback_dev_validator_routes,
        internal_auth_secret=internal_auth_secret,
        evm_address=evm_address,
        evm_private_key=evm_private_key,
        evm_challenge_signer=evm_challenge_signer,
        llama_ctx_budget=llama_ctx_budget,
        capacity_drain_file=capacity_drain_file,
        slot_state_dir=slot_state_dir,
        capacity_roster_file=capacity_roster_file,
        max_request_body_bytes=max_request_body_bytes,
        verification_snapshot_loader=verification_snapshot_loader,
        capability=capability,
        host=host,
        port=port,
        mesh_spec=mesh_spec,
        mesh_spec_loader=mesh_spec_loader,
        join_handler=join_handler,
        mesh_spec_handler=mesh_spec_handler,
        mesh_update_handler=mesh_update_handler,
        backend_url=backend_url,
        proof_url=proof_url,
        require_proof=require_proof,
        proof_trace_enable_file=proof_trace_enable_file,
        proof_trace_dir=proof_trace_dir,
        proof_gguf_manifest_path=proof_gguf_manifest_path,
        proof_tolerance_abs=proof_tolerance_abs,
        proof_tolerance_rel=proof_tolerance_rel,
        proof_block_size=proof_block_size,
        proof_spot_checks=proof_spot_checks,
        proof_warmup=proof_warmup,
        proof_decode_projection_warmup=proof_decode_projection_warmup,
        proof_sample_bps=proof_sample_bps,
        defer_proof=defer_proof,
        proof_ops_per_request=proof_ops_per_request,
        proof_trace_candidates_per_request=proof_trace_candidates_per_request,
        decode_audit_bps=decode_audit_bps,
        decode_audit_top_k=decode_audit_top_k,
        proof_artifact_timeout=proof_artifact_timeout,
        prompt_binding_cache_enabled=prompt_binding_cache_enabled,
        llama_n_parallel=llama_n_parallel,
        llama_n_ubatch=llama_n_ubatch,
        slot_view_template_warmup=slot_view_template_warmup,
        local_stage_capture=local_stage_capture,
        proof_trace_manifest_format=proof_trace_manifest_format,
    )
    print(f"verathos mesh worker listening on http://{host}:{port}", flush=True)

    def warm_slot_view_template() -> None:
        if not bool(getattr(server, "verathos_slot_view_template_warmup", False)):
            return
        spec_loader = getattr(server, "verathos_mesh_spec_loader", None)
        spec = spec_loader() if callable(spec_loader) else mesh_spec
        if spec is not None and rpc_plan_from_mesh(spec).rpc_endpoints:
            return
        bind_host = server.server_address[0]
        if bind_host in {"", "0.0.0.0", "::"}:
            bind_host = "127.0.0.1"
        local_port = int(server.server_address[1])
        endpoint = f"http://{bind_host}:{local_port}/v1/mesh/inference"
        model = spec.model_id if spec is not None else "verathos-slot-view-warmup"
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with one short word.",
                }
            ],
            "stream": False,
            "max_tokens": 4,
            "temperature": 0,
            "samplers": ["top_k"],
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "repeat_last_n": 0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "mirostat": 0,
            "ignore_eos": False,
            "seed": 0,
            "cache_prompt": False,
        }
        payload = {
            "request_id": f"slot-view-warmup-{time.time_ns()}",
            "mesh_spec_hash": spec.spec_hash_hex() if spec is not None else "",
            "openai_request": request,
            "require_proof": True,
        }
        for attempt in range(60):
            try:
                post_json(endpoint, payload, timeout=30.0)
                return
            except Exception:
                time.sleep(min(0.25 + attempt * 0.05, 2.0))

    if bool(getattr(server, "verathos_slot_view_template_warmup", False)):
        threading.Thread(target=warm_slot_view_template, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def serve_worker_in_thread(
    *,
    capability: CapabilityAd,
    host: str = "127.0.0.1",
    port: int = 0,
    mesh_spec: MeshSpec | None = None,
    mesh_spec_loader: MeshSpecLoader | None = None,
    join_handler: JoinHandler | None = None,
    mesh_spec_handler: MeshSpecHandler | None = None,
    mesh_update_handler: MeshUpdateHandler | None = None,
    backend_url: str = "",
    proof_url: str = "",
    require_proof: bool = False,
    proof_trace_enable_file: str | Path = "",
    proof_trace_dir: str | Path = "",
    proof_gguf_manifest_path: str | Path = "",
    proof_tolerance_abs: float = 8e-2,
    proof_tolerance_rel: float = 4e-2,
    proof_block_size: int = 64,
    proof_spot_checks: int = 8,
    proof_warmup: bool = False,
    proof_decode_projection_warmup: bool = False,
    proof_sample_bps: int = PROOF_SAMPLE_BPS_DENOMINATOR,
    defer_proof: bool = False,
    proof_ops_per_request: int = 1,
    proof_trace_candidates_per_request: int = 8,
    decode_audit_bps: int = 0,
    decode_audit_top_k: int = 8,
    proof_artifact_timeout: float = DEFAULT_PROOF_ARTIFACT_TIMEOUT,
    prompt_binding_cache_enabled: bool = False,
    llama_n_parallel: int = 1,
    llama_ctx_budget: int = 0,
    llama_n_ubatch: int = 0,
    slot_view_template_warmup: bool = False,
    local_stage_capture: bool = False,
    proof_trace_manifest_format: str = "",
    receipt_signer: "Callable[[str], str] | None" = None,
    stage_receipt_signer: "Callable[[str], str] | None" = None,
    stage_proof_key: str = "",
    server_role: str = "auto",
    validator_auth_enabled: bool = False,
    validator_allowlist_path: str | Path = "",
    validator_allowlist_max_age_seconds: float = (
        DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS
    ),
    require_validator_nonce: bool = False,
    allow_loopback_dev_validator_routes: bool = False,
    internal_auth_secret: str | bytes = "",
    evm_address: str = "",
    evm_private_key: str = "",
    capacity_drain_file: str | Path = "",
    slot_state_dir: str | Path = "",
    capacity_roster_file: str | Path = "",
    max_request_body_bytes: int = 128 * 1024 * 1024,
    verification_snapshot_loader: VerificationSnapshotLoader | None = None,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start a worker server in a background thread for tests and local tools."""

    server = make_worker_server(
        capability=capability,
        receipt_signer=receipt_signer,
        stage_receipt_signer=stage_receipt_signer,
        stage_proof_key=stage_proof_key,
        server_role=server_role,
        validator_auth_enabled=validator_auth_enabled,
        validator_allowlist_path=validator_allowlist_path,
        validator_allowlist_max_age_seconds=(
            validator_allowlist_max_age_seconds
        ),
        require_validator_nonce=require_validator_nonce,
        allow_loopback_dev_validator_routes=allow_loopback_dev_validator_routes,
        internal_auth_secret=internal_auth_secret,
        evm_address=evm_address,
        evm_private_key=evm_private_key,
        capacity_drain_file=capacity_drain_file,
        slot_state_dir=slot_state_dir,
        capacity_roster_file=capacity_roster_file,
        max_request_body_bytes=max_request_body_bytes,
        verification_snapshot_loader=verification_snapshot_loader,
        host=host,
        port=port,
        mesh_spec=mesh_spec,
        mesh_spec_loader=mesh_spec_loader,
        join_handler=join_handler,
        mesh_spec_handler=mesh_spec_handler,
        mesh_update_handler=mesh_update_handler,
        backend_url=backend_url,
        proof_url=proof_url,
        require_proof=require_proof,
        proof_trace_enable_file=proof_trace_enable_file,
        proof_trace_dir=proof_trace_dir,
        proof_gguf_manifest_path=proof_gguf_manifest_path,
        proof_tolerance_abs=proof_tolerance_abs,
        proof_tolerance_rel=proof_tolerance_rel,
        proof_block_size=proof_block_size,
        proof_spot_checks=proof_spot_checks,
        proof_warmup=proof_warmup,
        proof_decode_projection_warmup=proof_decode_projection_warmup,
        proof_sample_bps=proof_sample_bps,
        defer_proof=defer_proof,
        proof_ops_per_request=proof_ops_per_request,
        proof_trace_candidates_per_request=proof_trace_candidates_per_request,
        decode_audit_bps=decode_audit_bps,
        decode_audit_top_k=decode_audit_top_k,
        proof_artifact_timeout=proof_artifact_timeout,
        prompt_binding_cache_enabled=prompt_binding_cache_enabled,
        llama_n_parallel=llama_n_parallel,
        llama_ctx_budget=llama_ctx_budget,
        llama_n_ubatch=llama_n_ubatch,
        slot_view_template_warmup=slot_view_template_warmup,
        local_stage_capture=local_stage_capture,
        proof_trace_manifest_format=proof_trace_manifest_format,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _fetch_json(
    url: str,
    *,
    timeout: float = 3.0,
    internal_auth_secret: str | bytes = "",
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if internal_auth_secret:
        headers.update(
            sign_internal_http_request(
                secret=internal_auth_secret,
                method="GET",
                path=urlparse(url).path or "/",
                body=b"",
            )
        )
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"failed to connect to {url}: {exc.reason}") from exc

    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return data


# Pinned TLS trust shared with the keep-alive client below: the urllib
# openers pin per-host contexts inside handler closures, which a raw
# http.client connection cannot reach. Installers register the same
# context here so keep-alive requests verify against the identical pin.
_PINNED_TLS_CONTEXTS: dict[tuple[str, int], Any] = {}
# (scheme, host, port) -> one idle keep-alive connection. Signing-heavy
# paths (delegated receipt/artifact signs) pay a full TCP+TLS handshake
# per call otherwise - measured 326ms per sign from a WAN worker to the
# manager, with most of it spent on TLS setup.
_KEEPALIVE_LOCK = threading.Lock()
_KEEPALIVE_CONNS: dict[tuple[str, str, int], Any] = {}


def register_pinned_tls_context(host: str, port: int, context: Any) -> None:
    _PINNED_TLS_CONTEXTS[(str(host), int(port or 443))] = context


def _post_json_keepalive(
    url: str,
    body: bytes,
    request_headers: dict[str, str],
    *,
    timeout: float,
    fallback_on_error: bool = True,
) -> tuple[int, bytes] | None:
    """POST over a pooled keep-alive connection; None = caller falls back.

    Only https hosts with a REGISTERED pinned context are eligible: a raw
    connection with default PKI trust would reject the manager's
    self-signed certificate that the urllib opener path pins, so hosts
    without a registered pin keep the opener path untouched.
    """

    import http.client as _http_client

    parsed = urlparse(url)
    scheme = str(parsed.scheme)
    host = parsed.hostname or ""
    port = int(parsed.port or (443 if scheme == "https" else 80))
    if scheme != "https" or not host:
        return None
    context = _PINNED_TLS_CONTEXTS.get((host, port))
    if context is None:
        return None
    key = (scheme, host, port)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    for attempt in (0, 1):
        with _KEEPALIVE_LOCK:
            conn = _KEEPALIVE_CONNS.pop(key, None)
        fresh = conn is None
        if conn is None:
            conn = _http_client.HTTPSConnection(
                host, port, timeout=timeout, context=context
            )
        else:
            conn.timeout = timeout
            if getattr(conn, "sock", None) is not None:
                try:
                    conn.sock.settimeout(timeout)
                except OSError:
                    pass
        try:
            conn.request("POST", path, body=body, headers=request_headers)
            resp = conn.getresponse()
            raw = resp.read()
            status = int(resp.status)
            reusable = not resp.will_close
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            if fresh:
                # A fresh connection failing is a real transport error -
                # ordinary callers retain the urllib fallback. Deadline-
                # sensitive delegated signing disables that duplicate network
                # attempt and supplies its own bounded, idempotent retry loop.
                if not fallback_on_error:
                    raise RuntimeError(
                        f"failed to connect to {url}: {exc}"
                    ) from exc
                return None
            continue  # stale idle connection; retry once on a fresh one
        if reusable:
            with _KEEPALIVE_LOCK:
                if key not in _KEEPALIVE_CONNS:
                    _KEEPALIVE_CONNS[key] = conn
                    conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return status, raw
    return None


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 3.0,
    headers: Mapping[str, str] | None = None,
    internal_auth_secret: str | bytes = "",
    keepalive: bool = False,
    keepalive_fallback: bool = True,
) -> dict[str, Any]:
    # Compact separators: proof-payload sign requests carry multi-MiB
    # element-heavy JSON, and default spaced separators inflate the wire
    # body 5-30% past caps checked on canonical (compact) JSON.
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update({str(key): str(value) for key, value in headers.items()})
    if internal_auth_secret:
        request_headers.update(
            sign_internal_http_request(
                secret=internal_auth_secret,
                method="POST",
                path=urlparse(url).path or "/",
                body=body,
            )
        )
    if keepalive:
        pooled = _post_json_keepalive(
            url,
            body,
            request_headers,
            timeout=timeout,
            fallback_on_error=keepalive_fallback,
        )
        if pooled is not None:
            status, raw = pooled
            if status >= 400:
                detail = raw.decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {status} from {url}: {detail}")
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError(f"{url} returned non-object JSON")
            return data
    req = Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"failed to connect to {url}: {exc.reason}") from exc

    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return data


def probe_worker(
    endpoint: str,
    *,
    timeout: float = 3.0,
    internal_auth_secret: str | bytes = "",
) -> WorkerProbe:
    """Fetch health and capability metadata from a worker endpoint."""

    base = normalize_endpoint(endpoint)
    start = time.perf_counter()
    health = _fetch_json(urljoin(base, "health"), timeout=timeout)
    cap_payload = _fetch_json(
        urljoin(base, "capability"),
        timeout=timeout,
        internal_auth_secret=internal_auth_secret,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    capability = CapabilityAd.from_dict(cap_payload["capability"])
    if cap_payload.get("capability_hash") != capability.ad_hash_hex():
        raise RuntimeError("capability hash mismatch")

    return WorkerProbe(
        endpoint=base.rstrip("/"),
        status=str(health.get("status", "")),
        service=str(health.get("service", "")),
        version=int(health.get("version", 0)),
        capability=capability,
        mesh_spec_hash=str(health.get("mesh_spec_hash", "")),
        stage_assignment_hash=str(health.get("stage_assignment_hash", "")),
        latency_ms=latency_ms,
    )
