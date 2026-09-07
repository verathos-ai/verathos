"""Worker pool + driver placement for mesh orchestration (v1).

Design: docs/architecture/mesh_orchestration_ux.md.

Two planes:
- The pool MANAGER (one per UID, `mesh pool serve`) holds the worker
  registry, RTT data and mesh assignments, and exposes the dashboard API.
  It is never in the per-token data path.
- Each mesh's DRIVER is one of its own workers: it runs the (stage-less)
  mesh coordinator plus its local rpc-worker, so its stage costs no network
  and remote stages cost one RTT.

Workers only dial OUT to the manager (join/heartbeat/report); commands come
back on the heartbeat response, so a NAT'd worker needs no inbound port for
pool membership. The driver's data path (llama RPC + proof HTTP to every
member) is the only inbound requirement, satisfied per-worker by the
advertised rpc/proof endpoints.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import queue
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from verallm.mesh.delegated_signing import (
    CAPACITY_ARTIFACT_TYPES,
    CAPACITY_AUDIT_ARTIFACT_PURPOSE,
    IDENTITY_CHALLENGE_PURPOSE,
    MAX_CAPACITY_ARTIFACT_BYTES,
    coordinator_sign_request_with_retry,
    coordinator_sign_retry_profile,
    coordinator_sign_message,
    delegate_keypair_from_worker_request,
    identity_challenge_message,
)
from verallm.mesh.http_auth import (
    DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
    read_fresh_validator_allowlist,
)
from verallm.mesh.types import CapabilityAd, MeshSpec, canonical_json_bytes
from verallm.mesh.private_files import (
    read_owner_only_text,
    write_owner_only_json,
    write_owner_only_text,
)
from verallm.mesh.llama_cpp import llama_tensor_split_layer_ranges
from verallm.mesh.proof import (
    PROOF_SAMPLE_BPS_DENOMINATOR,
    VERATHOS_GGML_GEMM_PROOF_MODE,
    VERATHOS_GGML_LIGHT_PROOF_MODE,
)
from verallm.mesh.state import create_mesh_state, join_mesh
from verallm.mesh.worker import normalize_endpoint, post_json

logger = logging.getLogger(__name__)

POOL_TOKEN_PREFIX = "vtpool_"
POOL_STATE_FILE = "pool-state.json"
POOL_TOKEN_FILE = "pool-token.txt"
POOL_ADMIN_TOKEN_FILE = "pool-admin-token.txt"
POOL_WORKER_COMMAND_JOURNAL_FILE = "pool-command-journal.json"
POOL_TOKEN_SCOPE_WORKER = "worker"
POOL_TOKEN_SCOPE_MANAGEMENT = "management"
POOL_SERVING_MODE_DEV = "dev"
# A pool is always MINER-side infrastructure. "subnet" means this pool is
# registered on the subnet as a miner and signs the snapshots validators
# verify; "dev" means local only, nothing scoreable. The mode was originally
# spelled "validator", which read as if the miner were running a validator -
# that value is still accepted so existing pool state files keep loading.
POOL_SERVING_MODE_SUBNET = "subnet"
POOL_SERVING_MODE_SUBNET_LEGACY = "validator"
# KV pool size varies between restarts (VRAM fragmentation, CUDA graph
# overhead), so a measured context is never bit-identical run to run. The
# vLLM miner path absorbs the same jitter with a 10% tolerance
# (neurons/miner.py _ctx_close_enough) to avoid churning on-chain entries and
# causing model-index drift; the mesh uses the SAME figure deliberately.
MESH_CTX_JITTER_TOLERANCE = 0.10

POOL_SERVING_MODES = frozenset(
    {POOL_SERVING_MODE_DEV, POOL_SERVING_MODE_SUBNET}
)
# Private OpenAI API limits (fixed one-minute windows). Generous for one
# operator's own tools; the point is that an internet-exposed manager port
# cannot be brute-forced or flooded through the OpenAI surface.
POOL_API_RATE_LIMIT_PER_MIN = 120
POOL_API_AUTH_FAILURES_PER_MIN = 30
# 0.5s: a queued chat is picked up on the driver's next heartbeat, so this is
# a direct TTFT component (avg pickup = half the interval). The per-beat work
# is one small POST — trivial at pool scale, and still 60x inside
# WORKER_STALE_S.
DEFAULT_HEARTBEAT_S = 0.5
WORKER_STALE_S = 30.0
# The validator rewrites shared state throughout an epoch and on its periodic
# discovery refresh.  Past this window the score may still be historically
# useful, but it must not look current in the operator dashboard.
VALIDATOR_SCORE_STALE_S = 10 * 60.0
# Blocks per scoring epoch (bittensor tempo; mirrors the validator's
# epoch_blocks default). Snapshot epoch rotation keys on this.
VALIDATOR_EPOCH_BLOCKS = 360
# A deploy stage with no heartbeat for this long is a dead deploy process.
DEPLOY_STATUS_TTL_S = 900


def _call_bounded(fn, *, timeout_s: float):
    """Run ``fn`` with a hard deadline; raise TimeoutError when it blocks.

    For library calls with no timeout parameter of their own (a chain
    client reading a silently-dead websocket blocks recv forever). The
    abandoned worker thread is a daemon: it dies with the process, and the
    caller drops the wedged client so nothing reuses it.
    """

    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # propagate to the caller's thread
            result["error"] = exc

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()
    runner.join(timeout_s)
    if "value" in result:
        return result["value"]
    if "error" in result:
        raise result["error"]
    raise TimeoutError(f"call exceeded {timeout_s:.0f}s deadline")
DEFAULT_RTT_MS = 50.0
VRAM_HEADROOM = 1.3
# One coherent operator-chat budget.  It covers one lost long-poll response,
# the complete coordinator inference/proof timeout, and bounded final delivery
# retries.  The mesh lock outlives the browser deadline by a cleanup margin.
CHAT_POLL_WAIT_S = 25.0
CHAT_POLL_HTTP_TIMEOUT_S = 35.0
CHAT_DELIVERY_LEASE_S = 15.0
CHAT_POLL_RETRY_S = 1.0
CHAT_COORDINATOR_TIMEOUT_S = 300.0
CHAT_OPERATOR_DEADLINE_S = 420.0
CHAT_ACTIVE_MAX_S = 450.0
# Pre-registration probes replay the validator's full-context canary shape,
# whose inference budget is 900 s; ordinary operator chats stay on the
# shorter deadlines above. Management-authed callers opt in per request.
PROBE_DEADLINE_MAX_S = 960.0


def _requested_chat_proof_tier(body: Mapping[str, Any]) -> str:
    """The proof tier an operator chat asked for, refusing unsafe downgrades.

    Returns "light" only for a plain operator chat that explicitly asked
    for it, which is a latency-comparison convenience on the operator's
    own mesh. Probes and self-tests must exercise and ASSERT the hard
    relation, so a probe body can never resolve to light no matter what
    it requests; everything else falls through to the caller's default
    (hard on the manager-pinned lane).
    """

    requested = str(body.get("proof_tier", "") or "").strip().lower()
    if requested not in ("", "auto", "light", "hard"):
        raise ValueError("proof_tier must be auto, light, or hard")
    if requested == "light" and bool(body.get("probe", False)):
        raise ValueError("a probe cannot downgrade its proof tier to light")
    return requested if requested in ("light", "hard") else ""


def _chat_sampler_from_body(body: Mapping[str, Any]) -> dict[str, Any] | None:
    """Numeric sampler passthrough for an operator/private-API chat.

    The coordinator owns validation and normalization (committed light
    profile, decode-audit-width clamp); the manager only carries clean
    numeric fields so nothing else rides along."""

    sampler = body.get("sampler")
    if not isinstance(sampler, Mapping):
        return None
    fields: dict[str, Any] = {}
    for name in ("temperature", "top_k", "top_p", "min_p", "seed"):
        value = sampler.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fields[name] = value
    return fields or None


def chat_deadline_seconds(body: Mapping[str, Any]) -> float:
    """The operator chat deadline for one request body.

    ``probe: true`` (management-authed routes only) raises the ceiling to
    PROBE_DEADLINE_MAX_S; an explicit ``timeout`` is honored up to the
    applicable ceiling, anything invalid falls back to the ceiling.
    """
    cap = (
        PROBE_DEADLINE_MAX_S
        if bool(body.get("probe", False))
        else CHAT_OPERATOR_DEADLINE_S
    )
    try:
        timeout_s = float(body.get("timeout", cap))
    except (TypeError, ValueError, OverflowError):
        return cap
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= cap:
        return cap
    return timeout_s
CHAT_COMPLETED_TTL_S = 5 * 60.0
WORKER_REPORT_DELIVERY_MAX_S = 5 * 60.0
COMMAND_STOP_QUIESCE_S = 30.0
# A live driver long-polls /v1/pool/chat-poll every <=25s. If it hasn't polled
# in this long it is not picking up chats (crashed, or running pre-long-poll
# code) — queuing into the void just hangs, so we refuse fast with a clear
# message instead.
CHAT_POLL_STALE_S = 60.0
# Cadence for the driver's keep-warm touch (see _keepwarm). A GGUF driver on a
# memory-pressured host (Apple unified memory especially) has its weights paged
# out between requests; a periodic 1-token forward keeps every layer resident so
# TTFT stays flat instead of paying a cold re-fault. Model/hardware agnostic.
KEEPWARM_INTERVAL_S = 15.0
# Driver self-audit cadence: a full verified HARD request through the local
# coordinator (see _self_audit). 0 disables. The timeout covers a hard
# sumcheck + decode-audit replay on the largest meshes.
SELF_AUDIT_INTERVAL_S = 900.0
SELF_AUDIT_TIMEOUT_S = 300.0


def self_audit_skippable(message: str) -> bool:
    """Self-audit contention is a skip, never a strike: busy slots, an
    exclusive replay window held by a real canary, or a mid-rotation
    snapshot. Only genuine verification failures may fail the mesh."""

    lowered = str(message).lower()
    return any(
        token in lowered
        for token in (
            "slots busy",
            "503",
            "exclusive",
            "replay window",
            "snapshot no longer matches",
            "rotation",
        )
    )
MAX_POOL_REQUEST_BODY_BYTES = 4 * 1024 * 1024
#: The delegated-sign route carries whole sampled proof payloads: a 4-GPU
#: A100-class combined proof measures ~11.2MiB PER GPU.
MAX_POOL_SIGN_REQUEST_BODY_BYTES = 24 * 1024 * 1024
POOL_REQUEST_SOCKET_TIMEOUT_S = 30.0
WORKER_CONTROL_MAX_CLOCK_SKEW_S = 90
WORKER_CONTROL_NONCE_TTL_S = 5 * 60.0
_WORKER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_WORKER_AUTH_NONCE_RE = re.compile(r"[0-9a-f]{64}")
_COMMAND_ID_RE = re.compile(r"cmd-[0-9a-f]{32}")
_COMMAND_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _worker_control_body_hash_hex(body: Mapping[str, Any]) -> str:
    unsigned = {
        str(key): value
        for key, value in body.items()
        if key != "worker_auth_signature"
    }
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _signed_worker_control_body(
    body: Mapping[str, Any],
    *,
    action: str,
    keypair: Any,
    proof_key: str,
) -> dict[str, Any]:
    """Sign one worker->manager request with the persistent stage identity."""

    from verallm.mesh.receipt_signing import sign_worker_control_body_hash

    payload = {
        **dict(body),
        "worker_auth_action": str(action),
        "worker_auth_timestamp": int(time.time()),
        "worker_auth_nonce": secrets.token_hex(32),
        "worker_proof_key": str(proof_key),
    }
    payload["worker_auth_signature"] = sign_worker_control_body_hash(
        _worker_control_body_hash_hex(payload),
        keypair,
        expected_proof_key=str(proof_key),
    )
    return payload


def model_trace_dir(workdir: Path | str, model_id: str) -> Path:
    """Per-MODEL proof trace directory under a worker workdir.

    Trace manifests describe one model's op graph. A worker serves several
    models over its lifetime from one workdir, so a shared traces/ directory
    lets a template loader pick up a manifest written by a previous model
    whose tensor names do not exist in the current graph. Scoping by model
    id makes that impossible.
    """

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_id or "default")).strip("-")
    return Path(workdir) / "traces" / (slug or "default")


def _normalize_evm_address(value: object) -> str:
    """Return one canonical coordinator address or reject an unsafe mapping.

    Pool ownership uses an SS58 account, while validator scores are keyed by
    the coordinator miner's EVM address.  They are deliberately not inferred
    from one another: an explicit, well-formed EVM identity is required before
    the dashboard can associate any validator score with a mesh.
    """

    address = str(value or "").strip().lower()
    if not address:
        return ""
    if not re.fullmatch(r"0x[0-9a-f]{40}", address):
        raise ValueError("coordinator address must be a 20-byte 0x EVM address")
    return address


def _normalize_pool_serving_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == POOL_SERVING_MODE_SUBNET_LEGACY:
        # Pools minted before the rename persist "validator"; same mode.
        mode = POOL_SERVING_MODE_SUBNET
    if mode not in POOL_SERVING_MODES:
        raise ValueError("pool serving_mode must be explicitly dev or subnet")
    return mode


# bittensor network names by EVM chain id, for pools whose PM2 unit predates
# the --subtensor-network flag: the chain id is already in validator_binding,
# so the metagraph-backed status endpoints can still resolve a network.
_SUBTENSOR_NETWORK_BY_CHAIN_ID = {945: "test", 964: "finney"}


def subtensor_network_for_chain_id(chain_id: object) -> str:
    try:
        return _SUBTENSOR_NETWORK_BY_CHAIN_ID.get(int(chain_id), "")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def is_subnet_serving_mode(value: object) -> bool:
    """True when ``value`` names the subnet serving mode.

    The single mode predicate for every UI/flow branch: state files persist
    "subnet" since the rename but pools minted before it still carry
    "validator", and comparing either literal directly has already shipped
    one bug (an already-subnet pool offered "recreate as a subnet pool").
    Unknown or empty values are simply not subnet mode, never an error.
    """

    try:
        return _normalize_pool_serving_mode(value) == POOL_SERVING_MODE_SUBNET
    except ValueError:
        return False


def _require_digest(value: object, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field_name} must be a lowercase 32-byte hex digest")
    return digest


def _require_max_context_len(
    value: object,
    *,
    field_name: str = "max_context_len",
) -> int:
    if type(value) is not int or not 1 <= value < 2**32:
        raise ValueError(f"{field_name} must be a positive uint32 integer")
    return int(value)


def _validator_binding(
    *,
    chain_id: object,
    netuid: object,
    coordinator_uid: object,
    epoch: object,
    snapshot_ttl_seconds: object,
) -> dict[str, int]:
    """Validate the chain facts a pool may stamp into signed snapshots."""

    raw_values = {
        "chain_id": chain_id,
        "netuid": netuid,
        "coordinator_uid": coordinator_uid,
        "epoch": epoch,
        "snapshot_ttl_seconds": snapshot_ttl_seconds,
    }
    values: dict[str, int] = {}
    for name, value in raw_values.items():
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            values[name] = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= values["chain_id"] < 2**64:
        raise ValueError("chain_id must be in [1, 2^64)")
    if not 0 <= values["netuid"] <= 65_535:
        raise ValueError("netuid must fit uint16")
    if not 0 <= values["coordinator_uid"] < 2**32:
        raise ValueError("coordinator_uid must fit uint32")
    if not 0 <= values["epoch"] < 2**63:
        raise ValueError("epoch must be in [0, 2^63)")
    if not 60 <= values["snapshot_ttl_seconds"] <= 7 * 24 * 3600:
        raise ValueError("snapshot_ttl_seconds must be between 60s and 7 days")
    return values


# Proof profile every pool-launched mesh runs (the validated candidate
# profile; the coordinator pins the manifest format into the spec).
PROOF_FLAGS = [
    "--require-proof",
    "--proof-sample-bps", str(PROOF_SAMPLE_BPS_DENOMINATOR),
    # ONE proven op per request (speed: proof work per request stays flat)
    # drawn from a WIDE candidate window. The old 8-candidate window only
    # ever covered the first few (layer-0 attention) ops, so MoE expert
    # GEMMs (captured at higher op indices) could never be challenged: a
    # cheater could fake experts freely. A wide window puts the whole
    # captured forward (dense + expert planes) in the beacon-selectable set
    # so experts are challenged across requests. Affordable now that expert
    # weight loads dequant a SINGLE plane (~25ms cold, cache hit after)
    # instead of the full 3-D tensor + all-plane Merkle (minutes) that
    # stalled the drive when this window was first widened.
    "--proof-ops-per-request", "1",
    "--proof-trace-candidates-per-request", "1024",
    "--proof-tolerance-abs", "0.08",
    "--proof-tolerance-rel", "0.04",
    # ZERO by protocol: there are exactly two proof
    # tiers, light and hard, and the light tier carries no decode
    # obligation - decode verification lives in hard draws and forced hard
    # canaries, which prove the LM head outright. The canonical policy
    # (neurons/validator.py) pins BOTH organic and canary rates to the
    # SAME value so the wire cannot identify a canary; the signed snapshot
    # embeds that policy and the coordinator refuses to serve chain-bound
    # when the runtime disagrees ("snapshot decode policy does not match
    # runtime"). A non-zero rate forced a teacher-forced replay on every
    # inline light turn (tail capture is structurally unavailable at
    # --parallel > 1).
    "--decode-audit-bps", "0",
    "--decode-audit-top-k", "8",
]


def _serve_parallel_slots() -> int:
    """Backend slots per mesh serve: at least 8, env-tunable upward.

    Production meshes serve concurrent traffic; a single slot silently
    serializes every user behind one generation. Eight is the operator-set
    floor; VERATHOS_MESH_SERVE_PARALLEL raises it for bigger boxes and is
    clamped up to the floor rather than allowed to sneak back below it.
    """

    raw = os.environ.get("VERATHOS_MESH_SERVE_PARALLEL", "").strip()
    try:
        value = int(raw) if raw else 8
    except ValueError:
        value = 8
    return max(8, value)


# Descent ladder for the measured KV auto-fit: tried top-down after an
# allocation failure at the model's own maximum. Bounded and coarse on
# purpose: every step re-loads the model, so precision costs minutes.
_KV_FIT_LADDER = (131072, 65536, 32768, 16384)
_KV_ALLOC_FAILURE_RE = re.compile(
    r"out of memory|cudamalloc|failed to allocate|alloc(?:ation)? fail"
    r"|not enough (?:memory|space)|insufficient memory",
    re.IGNORECASE,
)


def _serve_ctx_budget(max_context_len: int) -> int:
    """The unified KV FLOOR for one mesh serve: the per-request contract.

    0 tells llama-server to use the model's trained maximum. A pinned
    registry max_context_len is the advertised per-request maximum and
    the minimum the serve must hold. The ACTUAL budget is raised to the
    machine's MEASURED fit at drive time (cached KV fit, else the free
    VRAM estimate): whatever headroom the hardware really has serves
    concurrency - overlapping validator canaries and private-API traffic
    - instead of sitting idle. The auto-fit ladder plus the first-batch
    probe shrink a too-optimistic estimate back down, never below this
    contract. VERATHOS_MESH_CTX_BUDGET overrides for VRAM-constrained
    machines (never below 8192: a budget smaller than one real request
    is a misconfiguration, not a tuning).
    """

    raw = os.environ.get("VERATHOS_MESH_CTX_BUDGET", "").strip()
    if raw:
        try:
            return max(8192, int(raw))
        except ValueError:
            pass
    return int(max_context_len or 0)


def _kv_bytes_per_token(gguf_path: str | Path) -> int:
    """KV-cache bytes per token for this GGUF, from its own header.

    n_layers x n_kv_heads x (key_length + value_length) x 2 bytes (f16), the
    layout llama.cpp allocates. Returns 0 when the header cannot be read, and
    callers then fall back to the measured ladder.
    """

    try:
        import gguf
        import numpy as np

        reader = gguf.GGUFReader(str(gguf_path))
        fields = reader.fields

        def _val(suffix: str) -> int:
            arch = str(_gguf_field_str(fields, "general.architecture") or "")
            field = fields.get(f"{arch}.{suffix}")
            if field is None:
                return 0
            try:
                return int(np.asarray(field.parts[field.data[0]]).ravel()[0])
            except Exception:
                return 0

        layers = _val("block_count")
        kv_heads = _val("attention.head_count_kv") or 1
        k_dim = _val("attention.key_length")
        v_dim = _val("attention.value_length") or k_dim
        if layers <= 0 or k_dim <= 0:
            return 0
        return int(layers * kv_heads * (k_dim + v_dim) * 2)
    except Exception as exc:  # pragma: no cover - header read failure
        logger.debug("KV size estimate unavailable for %s: %s", gguf_path, exc)
        return 0


def _gguf_field_str(fields: Mapping[str, Any], key: str) -> str:
    field = fields.get(key)
    if field is None:
        return ""
    try:
        return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
    except Exception:
        return ""


def _gguf_trained_context(gguf_path: str | Path) -> int:
    """The model's trained context length from its GGUF header, or 0."""

    try:
        import gguf
        import numpy as np

        reader = gguf.GGUFReader(str(gguf_path))
        fields = reader.fields
        arch = _gguf_field_str(fields, "general.architecture")
        field = fields.get(f"{arch}.context_length")
        if field is None:
            return 0
        return int(np.asarray(field.parts[field.data[0]]).ravel()[0])
    except Exception:
        return 0


#: Per-GPU VRAM held back from the unified KV budget so the hot capacity
#: audit workload (~500MB arena + a fresh torch CUDA context) can run BESIDE
#: the resident model. Without it the KV auto-fit packs the card wall to
#: wall and every audit workload dies in cudaMalloc before B_start — a
#: permanent no-show while serving looks healthy.
CAPACITY_AUDIT_VRAM_RESERVE_MB = 3072
#: Residual margin when the workspace is ALREADY HELD by a resident holder
#: process: free VRAM then excludes the workspace by construction, and the
#: full reserve would double-count it.
CAPACITY_AUDIT_HELD_RESERVE_MB = 512


def _effective_audit_reserve_mb(local_gpu_index: int | None = None) -> int:
    """Launch-time audit reserve, holder-aware PER GPU.

    A partially-held box (one GPU too full for its holder) must keep the
    full reserve on the unheld GPU; a box-wide any() under-reserves it and
    can make audits run out of memory.
    """
    try:
        from verallm.mesh.capacity_audit_worker import workspace_holders

        holders = workspace_holders()
        if local_gpu_index is not None:
            return (
                CAPACITY_AUDIT_HELD_RESERVE_MB
                if holders.active(int(local_gpu_index))
                else CAPACITY_AUDIT_VRAM_RESERVE_MB
            )
        indexes = range(len(_gpu_free_vram_mb()) or 1)
        if all(holders.active(i) for i in indexes):
            return CAPACITY_AUDIT_HELD_RESERVE_MB
    except Exception:
        pass
    return CAPACITY_AUDIT_VRAM_RESERVE_MB


def _extra_arg_value_index(cmd: list[str], flag: str) -> int:
    """Index of the ``--llama-extra-arg=<value>`` entry following ``flag``.

    Extra args ride as paired ``--llama-extra-arg=-ub`` /
    ``--llama-extra-arg=4096`` entries; returns -1 when the flag is absent.
    """

    marker = f"--llama-extra-arg={flag}"
    for i, item in enumerate(cmd):
        if item == marker and i + 1 < len(cmd):
            return i + 1
    return -1


def _gpu_free_vram_mb() -> list[int]:
    """Per-GPU free VRAM (MB) as nvidia-smi reports it now."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return []
        return [int(line) for line in out.stdout.split() if line.strip().isdigit()]
    except Exception:
        return []




def _kv_tokens_that_fit(
    gguf_path: str | Path, model_bytes: int, *, utilization: float = 0.90
) -> int:
    """Largest unified KV budget (tokens) that free VRAM can hold.

    The vLLM gpu_memory_utilization idea: rather than starting at the model's
    trained maximum and stepping DOWN a coarse ladder (1M -> 131072 for a
    model whose spare VRAM actually holds ~400k), size the budget from what
    is really free. Weights and a utilization margin come off the top; the
    rest becomes KV. The launch still verifies by measurement and descends on
    an allocation failure, so an imperfect estimate costs a retry, never a
    wrong claim.
    """

    per_token = _kv_bytes_per_token(gguf_path)
    if per_token <= 0:
        return 0
    per_gpu_free = _gpu_free_vram_mb()
    free = int(sum(per_gpu_free)) * 1024 * 1024
    if free <= 0:
        return 0
    # Hold back the capacity-audit workspace on every GPU BEFORE sizing KV:
    # the audit workload must fit beside the resident model at any window.
    audit_reserve = (
        len(per_gpu_free) * _effective_audit_reserve_mb() * 1024 * 1024
    )
    spare = (
        int(free * max(0.1, min(0.98, utilization)))
        - int(model_bytes or 0)
        - audit_reserve
    )
    if spare <= 0:
        return 0
    tokens = spare // per_token
    # Round down to a 4096 multiple: llama pads, and a round number is far
    # easier to reason about in logs and dashboards.
    tokens = int(tokens // 4096 * 4096)
    # NEVER exceed, and never quietly cap, the model's trained context. If
    # everything fits, 0 (use the trained maximum) is the right answer and
    # this estimate must stay out of the way: models that already served
    # their full context (1M on DeepSeek Flash, 262k on Qwen) must keep
    # doing so. This estimate exists only for models that cannot fit theirs.
    trained = _gguf_trained_context(gguf_path)
    if trained and tokens >= trained:
        return 0
    return tokens


def _tolerance_flags(command: dict[str, Any]) -> list[str]:
    """Per-model proof tolerance overrides, appended AFTER PROOF_FLAGS so
    argparse takes them (last value wins).

    The witness float-recompute band is quantization-dependent: sub-q4 GGUF
    quants (q3_K and below) dequantize in-kernel with coarser blocks, so an
    HONEST node's recompute drifts slightly wider than q4_K's (observed rel
    ~0.044 on Metal q3_K vs the 0.04 default; a substituted output diverges by
    orders of magnitude, so a small widening keeps the check sound). The
    override lives in the driver's catalog entry and is fanned to every stage
    via the drive/join commands — all members of a mesh must verify with the
    same band.
    """
    flags: list[str] = []
    if command.get("proof_tolerance_abs") is not None:
        flags += ["--proof-tolerance-abs", str(float(command["proof_tolerance_abs"]))]
    if command.get("proof_tolerance_rel") is not None:
        flags += ["--proof-tolerance-rel", str(float(command["proof_tolerance_rel"]))]
    return flags


_TOLERANCE_KEYS = ("proof_tolerance_abs", "proof_tolerance_rel")
# Per-model llama batching. ubatch is the mixed-batch TTFT floor: prefill
# fairness admits a small chat into the very next pass, but its first token
# still waits that whole pass (ub tokens / prefill throughput). Representative
# measurements showed no gain at ub 4096 and substantially lower chat TTFT at
# ub 1024 under full-context canaries.
# Model-dependent, so it travels the SAME hops as the proof tolerances.
_LLAMA_BATCH_KEYS = ("llama_ubatch", "llama_batch")
_OPERATOR_TUNING_KEYS = _TOLERANCE_KEYS + _LLAMA_BATCH_KEYS


def _tolerance_fields(src: Mapping[str, Any]) -> dict[str, float]:
    """Extract the per-model proof tolerance overrides from a catalog entry /
    registry entry / mesh record, for fanning to the next hop. One definition
    of the key set — the override travels catalog -> advert -> registry ->
    mesh -> drive/join command, and any hop that hand-copies the keys is a hop
    that can silently drop them (the q3_K intermittent-failure bug)."""
    return {k: float(src[k]) for k in _TOLERANCE_KEYS if src.get(k) is not None}


def _llama_batch_fields(src: Mapping[str, Any]) -> dict[str, int]:
    """Per-model llama batch overrides, same hop discipline as tolerances."""
    return {
        k: int(src[k]) for k in _LLAMA_BATCH_KEYS if src.get(k) is not None
    }


def _operator_tuning_fields(src: Mapping[str, Any]) -> dict[str, Any]:
    """Every operator-pinned per-model tuning field, for hop fan-out and for
    surviving registry re-registration."""
    return {**_tolerance_fields(src), **_llama_batch_fields(src)}


def _llama_batch_args(command: Mapping[str, Any]) -> list[str]:
    """The coordinator's -b/-ub llama args for this mesh/command.

    Precedence: explicit env override (operator break-glass on one box) >
    per-model registry/mesh value > defaults (-ub 4096, -b 2x ub)."""

    ub = (
        int(os.environ.get("VERATHOS_MESH_LLAMA_UBATCH", "") or 0)
        or int(command.get("llama_ubatch") or 0)
        or 4096
    )
    batch = (
        int(os.environ.get("VERATHOS_MESH_LLAMA_BATCH", "") or 0)
        or int(command.get("llama_batch") or 0)
        or 2 * ub
    )
    return [
        "--llama-extra-arg=-b",
        f"--llama-extra-arg={batch}",
        "--llama-extra-arg=-ub",
        f"--llama-extra-arg={ub}",
    ]


def _first_batch_deadline_s(model_bytes: int = 0) -> float:
    """Deadline for the first-batch probe, covering llama's whole model load.

    The probe answers only after every shard byte has streamed off disk
    (health returns 200 long before that), so the budget must scale with
    model size: 600s fits every small model on healthy NVMe, but a 120GB
    glm on a shared host with contended I/O (or a cgroup page cache
    smaller than the model) can read as slowly as ~50 MB/s, so a fixed 600s
    deadline may expire while llama is still loading shards. Bootstrap budget, not a
    validation gate: allocation failures still exit early via the
    backend-log scrape, so a generous deadline never masks a real death.
    Env override is the per-box break-glass and wins outright."""

    env_budget = os.environ.get("VERATHOS_MESH_FIRST_BATCH_DEADLINE_S", "")
    if env_budget:
        try:
            return float(env_budget)
        except ValueError:
            pass
    return min(7200.0, max(600.0, int(model_bytes or 0) / 50e6))


def _formation_silence_budget_s(model_bytes: int = 0) -> float:
    """Driver-silence budget during formation, scaled to the model.

    A driver loading a large GGUF can starve its own poll loop well past a
    flat 120s floor even while the load is healthy, and a false reap here
    does not stop the drive worker-side — it completes into a serve the
    failed record no longer tracks. 50 MB/s is the same conservative load
    rate the first-batch deadline uses; the cap keeps genuinely dead
    drivers from wedging a launch for more than 15 minutes.
    """

    return min(
        900.0,
        max(120.0, 4 * WORKER_STALE_S, int(model_bytes or 0) / 50e6),
    )


def _release_fetch_memory(model_id: str) -> None:
    """Return download-inflated heap to the OS after a model fetch.

    The fetch and manifest phases stream multi-GB files through this
    process, and glibc keeps the freed arenas mapped afterwards, so the
    daemon's RSS can sit gigabytes above its real working set. The daemon
    then enters the memory-heaviest phase of its life (the drive spawns
    llama-server plus the proof sidecar), and on a cgroup-limited box the
    OOM killer picks the fattest task — a daemon SIGKILLed mid-drive
    leaves an untracked serve behind. Trim is a no-op outside glibc
    (macOS workers), and every probe here is best-effort.
    """

    import gc

    gc.collect()
    trimmed = False
    try:
        import ctypes

        trimmed = bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except Exception:
        pass
    rss_mb = -1
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    logger.info(
        "fetch heap released for %s: malloc_trim=%s rss_now=%sMB",
        model_id,
        trimmed,
        rss_mb if rss_mb >= 0 else "unknown",
    )


def _registration_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """Registration fields for a status payload, single-slot compatible."""

    registrations = dict(state.get("mesh_registrations") or {})
    legacy = state.get("mesh_registration")
    if isinstance(legacy, dict) and legacy.get("model_id"):
        registrations.setdefault(str(legacy["model_id"]), dict(legacy))
    single = (
        dict(next(iter(registrations.values())))
        if len(registrations) == 1
        else {}
    )
    return {
        "mesh_registration": single,
        "mesh_registrations": registrations,
    }


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except OSError:
        return False


class _QuietHandshakeFailure(Exception):
    """A TLS client died during handshake; log one line, not a traceback."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_descends_from(pid: int, roots: set[int]) -> bool:
    """Attribute a Linux listener to a currently tracked process tree."""
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        try:
            # comm may contain spaces/parentheses; fields after its final ')'
            # begin with state and ppid.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _capability_subnet_driver_ready(capability: Mapping[str, Any]) -> bool:
    """Read a worker's subnet-driver readiness advert.

    Tolerates the pre-rename key so workers running older code stay
    placeable while a pool upgrades. (Renamed: this is miner-side readiness
    to DRIVE a subnet mesh -- the old "validator_*" name wrongly implied
    the worker validates something.)
    """

    if "subnet_driver_ready" in capability:
        return bool(capability.get("subnet_driver_ready"))
    return bool(capability.get("validator_driver_ready"))


def _shipped_model_fetch_spec(
    model_id: str, chain_id: int | None = None
) -> dict[str, Any] | None:
    """Download source for a model from the SHIPPED mesh catalogue, if any.

    The pool otherwise learns sources only from worker catalog adverts and
    the management registration route, so a model that ships in
    verallm/registry/models.py (hf_repo + hf_files and all) was unlaunchable
    on a pool whose workers happen not to hold it yet. Shape matches the
    worker-advertised registry entries so every downstream consumer
    (auto-fetch, disk-fit check) reads it unchanged.

    When the catalogue records the owner-built tensor-manifest root, the
    spec also carries it plus the chain's gleipnir store URLs so the driver
    DOWNLOADS the manifest instead of rebuilding it.
    """

    try:
        from verallm.registry.models import (
            mesh_model_manifest_root,
            mesh_model_source,
        )

        found = mesh_model_source(str(model_id))
        manifest_root = mesh_model_manifest_root(str(model_id))
    except Exception as exc:  # pragma: no cover - catalogue import failure
        logger.debug("shipped mesh catalogue unavailable for %s: %s", model_id, exc)
        return None
    if not found:
        return None
    hf_repo, hf_files, model_bytes, layers = found
    if not hf_repo or not hf_files:
        return None
    spec: dict[str, Any] = {
        "hf_repo": str(hf_repo),
        "hf_files": [str(f) for f in hf_files],
        "layers": int(layers or 0),
        "model_bytes": int(model_bytes or 0),
    }
    if manifest_root:
        from verallm.mesh.manifest_store import (
            all_default_store_urls,
            default_store_urls_for_chain,
        )

        spec["model_tensor_manifest_root"] = manifest_root
        # A pool without a chain binding still needs the owner's stores:
        # the root is content-addressed, so the union is safe to query.
        store_urls = default_store_urls_for_chain(chain_id) or all_default_store_urls()
        if store_urls:
            spec["manifest_urls"] = list(store_urls)
    return spec


def _proc_net_available() -> bool:
    """Whether this box exposes /proc/net/tcp (Linux) for listener lookup."""

    return Path("/proc/net/tcp").exists()


def _listening_inodes_from_proc(
    port: int, sources: Sequence[str] = ("/proc/net/tcp", "/proc/net/tcp6")
) -> set[int]:
    """Socket inodes LISTENing on `port`, read straight from /proc.

    No external binary: /proc/net/tcp{,6} is always there on Linux, while
    lsof frequently is not (plain container images ship without it).
    """

    inodes: set[int] = set()
    for name in sources:
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A = TCP_LISTEN
                continue
            local = fields[1].rsplit(":", 1)
            if len(local) != 2:
                continue
            try:
                if int(local[1], 16) != int(port):
                    continue
                inodes.add(int(fields[9]))
            except ValueError:
                continue
    return inodes


def _pids_holding_inodes(inodes: set[int]) -> set[int]:
    """PIDs holding any of these socket inodes, via /proc/<pid>/fd."""

    if not inodes:
        return set()
    wanted = {f"socket:[{inode}]" for inode in inodes}
    pids: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            handles = list(fd_dir.iterdir())
        except OSError:
            continue  # process gone, or not ours to inspect
        for handle in handles:
            try:
                if os.readlink(handle) in wanted:
                    pids.add(int(entry.name))
                    break
            except OSError:
                continue
    return pids


def _listeners_on_port(port: int) -> tuple[set[int], bool]:
    """(pids, occupied) for TCP listeners on `port`.

    `occupied` can be True with an empty pid set: a listener exists but its
    owner could not be attributed (another user's process under a non-root
    caller). Callers must treat that as occupied and fail closed, never as
    free.

    Linux reads /proc directly; elsewhere (macOS workers) it falls back to
    lsof, which ships with the OS there. A box with neither raises, keeping
    the original fail-closed contract instead of guessing the port is free.
    """

    if _proc_net_available():
        inodes = _listening_inodes_from_proc(port)
        return _pids_holding_inodes(inodes), bool(inodes)
    try:
        probe = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"cannot verify ownership of listeners on tcp:{port}: {exc}"
        ) from exc
    # lsof exits 1 when no file matches. Any other non-zero status is an
    # inspection failure, not evidence that the port is free.
    if probe.returncode not in {0, 1}:
        detail = str(probe.stderr or "").strip()[:300]
        raise RuntimeError(
            f"cannot verify ownership of listeners on tcp:{port}"
            + (f": {detail}" if detail else "")
        )
    pids = {int(p) for p in probe.stdout.split() if p.strip().isdigit()}
    return pids, bool(pids)


def _serve_memory_high_bytes(cgroup_base: Path = Path("/sys/fs/cgroup")) -> int:
    """memory.high for serve children: the container limit minus a margin.

    Returns 0 when confinement should not run: no container limit (bare
    metal / "max"), unreadable cgroupfs, or an explicit 0 override via
    VERATHOS_SERVE_MEMORY_HIGH_BYTES. The margin keeps the worker daemon,
    proof lanes, and audit workspace outside the reclaim storm.
    """

    override = os.environ.get("VERATHOS_SERVE_MEMORY_HIGH_BYTES", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            return 0
    try:
        raw = (cgroup_base / "memory.max").read_text().strip()
    except OSError:
        return 0
    if raw == "max":
        return 0
    try:
        limit = int(raw)
    except ValueError:
        return 0
    margin = 4 << 30
    floor = 8 << 30
    high = limit - margin
    if high < floor:
        high = max(limit // 2, 1 << 30)
    return high


def _confine_serve_process_memory(
    pid: int, cgroup_base: Path = Path("/sys/fs/cgroup")
) -> str:
    """Place a serve child under a memory.high sub-cgroup. Fail-open.

    A GGUF load streams the whole model through page cache faster than a
    memory-capped container reclaims; hitting memory.max OOM-kills the
    worker tree mid-formation (observed on the 24GB-GPU box class with
    30-32GiB limits). memory.high on a child group turns that death into
    reclaim + throttle: the load slows down and completes. Returns the
    cgroup path on success, "" when unavailable (bare metal, cgroup v1,
    read-only cgroupfs) - the child then runs unconfined exactly as
    before this fix.
    """

    high = _serve_memory_high_bytes(cgroup_base)
    if high <= 0:
        return ""
    try:
        subtree = cgroup_base / "cgroup.subtree_control"
        if not os.access(subtree, os.W_OK):
            return ""
        try:
            subtree.write_text("+memory")
        except OSError:
            pass  # already enabled or partially delegated; mkdir decides
        # Opportunistic sweep of empty groups left by exited children.
        for stale in cgroup_base.glob("verathos-serve-*"):
            try:
                stale.rmdir()
            except OSError:
                pass
        cgroup = cgroup_base / f"verathos-serve-{int(pid)}"
        cgroup.mkdir(exist_ok=True)
        (cgroup / "memory.high").write_text(str(int(high)))
        (cgroup / "cgroup.procs").write_text(str(int(pid)))
        return str(cgroup)
    except OSError:
        return ""


def _pcs_prover_threads() -> int:
    """Default thread count for the PCS/IPA prover inside worker units.

    Capped rather than "all cores": the prover runs alongside the serving
    backend on the same box, and oversubscribing the machine slows the very
    request whose proof is being built.

    The cap looks low, and a single-prover microbenchmark says it is.
    Measured on a 30-core 2x5090 box, median warm prove of one
    production-shaped op:

        threads      4     8    12    15    16    20    24
        attn_q    4.36  3.64  3.51  3.37  3.30  3.28  4.14
        LM head   ----  6.01  ----  ----  5.57  5.39  ----

    Do NOT raise it on the strength of that table. This value is PER
    WORKER UNIT, and a box runs one unit per GPU, so N co-located workers
    multiply it: cores // 2 on this box would put 2 x 15 = 30 prover
    threads on 30 cores before the serving backend gets any. An end-to-end
    A/B could not confirm any gain either, because hard-proof wall time is
    dominated by WHICH tensor the beacon draws (LM head ~6 s vs attn_q
    ~4 s per stage) and whether its weight cache is warm: at a fixed 8
    threads, three consecutive identical runs gave 15.4/30.1/15.4,
    16.6/20.7/25.6 and 6.7/8.8/10.6 seconds. Any future change here needs
    a sibling-worker-aware formula and a benchmark that controls for
    tensor selection.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cores = os.cpu_count() or 1
    return max(1, min(8, cores - 1))


def _fetch_dest_dir(model_id: str = "") -> Path:
    """Where auto-fetched models land on this worker."""
    root = Path.home() / ".verathos" / "mesh-models"
    return root / model_id if model_id else root


def _fetch_dest_free_gb() -> float:
    """Free disk at the auto-fetch destination in GB (nearest existing parent)."""
    p = _fetch_dest_dir()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except OSError:
        return 0.0


@dataclass(frozen=True)
class MeshPoolToken:
    """UID-scoped worker or operator credential for one pool."""

    pool_id: str
    manager_endpoint: str
    pool_secret: str
    scope: str = POOL_TOKEN_SCOPE_WORKER
    # SHA256 (hex) of the manager's TLS certificate in DER form. The token
    # is handed over a secure channel anyway (it carries the pool secret),
    # so it doubles as the TLS trust root for remote workers: no CA files
    # to distribute, no reliance on public PKI for rented boxes without a
    # domain. Empty for loopback/plain-HTTP pools and legacy tokens.
    manager_ca_sha256: str = ""

    def encode(self) -> str:
        if self.scope not in {
            POOL_TOKEN_SCOPE_WORKER,
            POOL_TOKEN_SCOPE_MANAGEMENT,
        }:
            raise ValueError("unsupported mesh pool token scope")
        payload = {
            "pool_id": self.pool_id,
            "manager_endpoint": self.manager_endpoint,
            "pool_secret": self.pool_secret,
            "scope": self.scope,
        }
        if self.manager_ca_sha256:
            payload["manager_ca_sha256"] = self.manager_ca_sha256
        raw = canonical_json_bytes(payload)
        return POOL_TOKEN_PREFIX + base64.urlsafe_b64encode(raw).decode()

    @classmethod
    def decode(cls, token: str) -> "MeshPoolToken":
        raw = str(token).strip()
        if not raw.startswith(POOL_TOKEN_PREFIX):
            raise ValueError("not a mesh pool token")
        data = json.loads(base64.urlsafe_b64decode(raw[len(POOL_TOKEN_PREFIX):]))
        decoded = cls(
            pool_id=str(data["pool_id"]),
            manager_endpoint=str(data["manager_endpoint"]),
            pool_secret=str(data["pool_secret"]),
            # Tokens minted before credential separation were distributed to
            # workers, so fail safely by treating them as worker-scoped.
            scope=str(data.get("scope", POOL_TOKEN_SCOPE_WORKER)),
            manager_ca_sha256=str(data.get("manager_ca_sha256", "") or ""),
        )
        if decoded.scope not in {
            POOL_TOKEN_SCOPE_WORKER,
            POOL_TOKEN_SCOPE_MANAGEMENT,
        }:
            raise ValueError("unsupported mesh pool token scope")
        return decoded


def load_pool_token_file(path: str | Path) -> MeshPoolToken:
    """Load a pool token without accepting a link or shared secret file."""

    raw = read_owner_only_text(path, label="pool token file").strip()
    if not raw:
        raise ValueError("pool token file is empty")
    return MeshPoolToken.decode(raw)


def _known_pools_file() -> Path:
    # Resolved per call, never at import: tests (and anything else that
    # redirects Path.home) must isolate, not write the real registry.
    return Path.home() / ".verathos" / "known-pools.json"


def record_known_pool(pool_dir: Path) -> None:
    """Best-effort registry of pool state dirs created on this machine.

    Zero-flag commands (`verathos mesh chat`, `fleet`, ...) discover pools
    through this file, so a human never has to remember where a state dir
    lives. Purely a convenience index: losing it costs nothing but the
    zero-flag lookup, and secrets never enter it (only paths). Entries
    whose directories vanished are pruned on every write.
    """

    import tempfile

    resolved_dir = Path(pool_dir).resolve()
    try:
        if resolved_dir.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            return  # throwaway pools (tests, scratch) never enter the registry
    except (OSError, ValueError):
        pass
    registry = _known_pools_file()
    try:
        entries: list[str] = []
        if registry.exists():
            loaded = json.loads(registry.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                entries = [str(item) for item in loaded]
        resolved = str(resolved_dir)
        if resolved not in entries:
            entries.append(resolved)
        entries = [item for item in entries if Path(item).is_dir()]
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(
            json.dumps(entries, indent=1) + "\n", encoding="utf-8"
        )
    except Exception:
        logger.debug("known-pools registry update failed", exc_info=True)


def known_pool_dirs() -> list[Path]:
    """Registered pool dirs that still exist, newest last."""

    try:
        loaded = json.loads(_known_pools_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(loaded, list):
        return []
    return [
        Path(item) for item in loaded if isinstance(item, str) and Path(item).is_dir()
    ]


def create_pool_state(
    root: str | Path,
    *,
    manager_endpoint: str,
    serving_mode: str,
    owner_account: str = "",
    coordinator_address: str = "",
    validator_shared_state_path: str | Path = "",
    chain_id: int | None = None,
    netuid: int | None = None,
    coordinator_uid: int | None = None,
    epoch: int | None = None,
    snapshot_ttl_seconds: int = 86_400,
) -> tuple[Path, MeshPoolToken]:
    """Persist a new pool and return its directory + join token."""

    mode = _normalize_pool_serving_mode(serving_mode)
    normalized_manager_endpoint = normalize_endpoint(
        manager_endpoint
    ).rstrip("/")
    manager_url = urlparse(normalized_manager_endpoint)
    if (
        mode == POOL_SERVING_MODE_SUBNET
        and manager_url.scheme != "https"
        and manager_url.hostname not in {"127.0.0.1", "::1", "localhost"}
    ):
        raise ValueError(
            "remote subnet pool manager endpoints must use HTTPS"
        )
    normalized_coordinator = _normalize_evm_address(coordinator_address)
    normalized_owner = str(owner_account or "").strip()
    normalized_shared_state = str(validator_shared_state_path or "").strip()
    if mode == POOL_SERVING_MODE_SUBNET and not normalized_owner:
        raise ValueError("subnet pools require owner_account")
    if normalized_owner:
        from verallm.mesh.receipt_signing import _keypair_from_ss58

        try:
            _keypair_from_ss58(normalized_owner)
        except Exception as exc:
            raise ValueError("owner_account must be a valid SS58 address") from exc
    validator_binding: dict[str, int] | None = None
    if mode == POOL_SERVING_MODE_SUBNET:
        if not normalized_coordinator:
            raise ValueError("subnet pools require coordinator_address")
        # validator_shared_state_path stays optional: it only feeds the
        # operator-board score view, which falls back to the chain score
        # cache when unset (public operators have no validator install).
        missing = [
            name
            for name, value in (
                ("chain_id", chain_id),
                ("netuid", netuid),
                ("coordinator_uid", coordinator_uid),
                ("epoch", epoch),
            )
            if value is None
        ]
        if missing:
            raise ValueError("subnet pools require " + ", ".join(missing))
        validator_binding = _validator_binding(
            chain_id=chain_id,
            netuid=netuid,
            coordinator_uid=coordinator_uid,
            epoch=epoch,
            snapshot_ttl_seconds=snapshot_ttl_seconds,
        )
    elif any(value is not None for value in (chain_id, netuid, coordinator_uid, epoch)):
        raise ValueError("chain binding fields are only valid for validator pools")

    pool_id = "pool-" + hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:12]
    token = MeshPoolToken(
        pool_id=pool_id,
        manager_endpoint=normalized_manager_endpoint,
        pool_secret=secrets.token_urlsafe(32),
        scope=POOL_TOKEN_SCOPE_WORKER,
    )
    admin_token = MeshPoolToken(
        pool_id=pool_id,
        manager_endpoint=token.manager_endpoint,
        pool_secret=secrets.token_urlsafe(32),
        scope=POOL_TOKEN_SCOPE_MANAGEMENT,
    )
    out = Path(root) / pool_id
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "pool_id": pool_id,
        "serving_mode": mode,
        "manager_endpoint": token.manager_endpoint,
        "pool_secret": token.pool_secret,
        "management_secret": admin_token.pool_secret,
        "created_at_unix": int(time.time()),
        "workers": {},
        "meshes": {},
        "snapshot_generation": 0,
    }
    if normalized_coordinator:
        state["coordinator_address"] = normalized_coordinator
    if normalized_owner:
        state["owner_account"] = normalized_owner
    if normalized_shared_state:
        state["validator_shared_state_path"] = normalized_shared_state
    if validator_binding is not None:
        state["validator_binding"] = validator_binding
        state["validator_epoch_floor"] = int(validator_binding["epoch"])
    write_owner_only_json(out / POOL_STATE_FILE, state)
    write_owner_only_text(out / POOL_TOKEN_FILE, token.encode() + "\n")
    write_owner_only_text(out / POOL_ADMIN_TOKEN_FILE, admin_token.encode() + "\n")
    record_known_pool(out)
    return out, token


def upgrade_pool_state_to_subnet(
    pool_dir: str | Path,
    *,
    owner_account: str,
    coordinator_address: str,
    validator_shared_state_path: str | Path,
    chain_id: int | None,
    netuid: int | None,
    coordinator_uid: int | None,
    epoch: int | None,
    snapshot_ttl_seconds: int = 86_400,
    manager_endpoint: str | None = None,
) -> tuple[Path, MeshPoolToken, bool]:
    """Flip a dev pool to subnet mode IN PLACE.

    The old path was an honest teardown-and-recreate (new pool id, new
    secrets, every worker re-joined by hand), but nothing actually requires
    that: workers already sign every request with their stage key even in
    dev mode, workdirs and proof caches are keyed by GPU group rather than
    pool id, and the join path binds an unbound worker's stage identity on
    its next rejoin. What a subnet pool genuinely needs is the chain
    binding, which this writes with the SAME invariants create_pool_state
    enforces. pool_id, both secrets, worker records, meshes (all stopped),
    and the model registry are preserved.

    Returns ``(pool_dir, worker_token, endpoint_changed)``. The worker
    token only changes its embedded endpoint when ``manager_endpoint``
    differs (e.g. http -> https for a remote manager); the secret is never
    rotated, so an unchanged endpoint keeps every existing token valid.

    The caller must ensure NO manager is running over this state dir: a
    live manager persists its in-memory state lazily and would clobber
    this write.
    """

    pool_dir = Path(pool_dir)
    state_path = pool_dir / POOL_STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if is_subnet_serving_mode(state.get("serving_mode")):
        raise ValueError(
            "this pool is already a subnet pool; nothing to upgrade"
        )
    running = sorted(
        key
        for key, mesh in (state.get("meshes") or {}).items()
        if isinstance(mesh, Mapping) and mesh.get("status") != "stopped"
    )
    if running:
        raise ValueError(
            "stop the running meshes first (their snapshots must sign the "
            "new chain binding): " + ", ".join(running)
        )
    endpoint = normalize_endpoint(
        str(manager_endpoint or state.get("manager_endpoint", ""))
    ).rstrip("/")
    manager_url = urlparse(endpoint)
    if (
        manager_url.scheme != "https"
        and manager_url.hostname not in {"127.0.0.1", "::1", "localhost"}
    ):
        raise ValueError(
            "remote subnet pool manager endpoints must use HTTPS; pass a "
            "https manager_endpoint (or front the manager with TLS) before "
            "upgrading"
        )
    normalized_coordinator = _normalize_evm_address(coordinator_address)
    if not normalized_coordinator:
        raise ValueError("subnet pools require coordinator_address")
    normalized_owner = str(owner_account or "").strip()
    if not normalized_owner:
        raise ValueError("subnet pools require owner_account")
    from verallm.mesh.receipt_signing import _keypair_from_ss58

    try:
        _keypair_from_ss58(normalized_owner)
    except Exception as exc:
        raise ValueError("owner_account must be a valid SS58 address") from exc
    normalized_shared_state = str(validator_shared_state_path or "").strip()
    # Optional: score-view enrichment only; see create_pool_state.
    missing = [
        name
        for name, value in (
            ("chain_id", chain_id),
            ("netuid", netuid),
            ("coordinator_uid", coordinator_uid),
            ("epoch", epoch),
        )
        if value is None
    ]
    if missing:
        raise ValueError("subnet pools require " + ", ".join(missing))
    validator_binding = _validator_binding(
        chain_id=chain_id,
        netuid=netuid,
        coordinator_uid=coordinator_uid,
        epoch=epoch,
        snapshot_ttl_seconds=snapshot_ttl_seconds,
    )
    endpoint_changed = endpoint != str(state.get("manager_endpoint", ""))
    state.update(
        {
            "serving_mode": POOL_SERVING_MODE_SUBNET,
            "manager_endpoint": endpoint,
            "coordinator_address": normalized_coordinator,
            "owner_account": normalized_owner,
            "validator_shared_state_path": normalized_shared_state,
            "validator_binding": validator_binding,
            "validator_epoch_floor": int(validator_binding["epoch"]),
        }
    )
    token = MeshPoolToken(
        pool_id=str(state["pool_id"]),
        manager_endpoint=endpoint,
        pool_secret=str(state["pool_secret"]),
        scope=POOL_TOKEN_SCOPE_WORKER,
    )
    admin_token = MeshPoolToken(
        pool_id=token.pool_id,
        manager_endpoint=endpoint,
        pool_secret=str(state["management_secret"]),
        scope=POOL_TOKEN_SCOPE_MANAGEMENT,
    )
    write_owner_only_json(state_path, state)
    # Idempotent either way; on an endpoint change this is what re-mints
    # the printed tokens (same secrets, new embedded endpoint).
    write_owner_only_text(pool_dir / POOL_TOKEN_FILE, token.encode() + "\n")
    write_owner_only_text(
        pool_dir / POOL_ADMIN_TOKEN_FILE, admin_token.encode() + "\n"
    )
    record_known_pool(pool_dir)
    return pool_dir, token, endpoint_changed


def _is_loopback(endpoint: str) -> bool:
    host = endpoint.split("//", 1)[-1].rsplit(":", 1)[0].strip("[]")
    return host in ("127.0.0.1", "localhost", "::1", "")


def _tcp_rtt_ms(endpoint: str, *, timeout: float = 3.0) -> float | None:
    """One TCP connect round-trip to host:port (any scheme prefix tolerated)."""

    target = endpoint.split("//", 1)[-1].rstrip("/")
    host, _, port_raw = target.rpartition(":")
    try:
        port = int(port_raw)
    except ValueError:
        return None
    started = time.monotonic()
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host or "127.0.0.1", port))
    except OSError:
        return None
    finally:
        sock.close()
    return (time.monotonic() - started) * 1000.0


class PoolManager:
    """Registry + assignment brain behind the pool HTTP routes."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        coordinator_address: str | None = None,
        validator_shared_state_path: str | Path | None = None,
        manifest_base_urls: Sequence[str] = (),
    ):
        # Where drivers may fetch a ready-made tensor manifest instead of
        # rebuilding one. Rebuilding hashes every tensor: minutes for a 7B,
        # the better part of an hour for a 238GB GGUF, on every fresh box.
        self.manifest_base_urls = tuple(
            str(url).strip().rstrip("/") for url in (manifest_base_urls or ()) if str(url).strip()
        )
        self.state_path = Path(state_dir) / POOL_STATE_FILE
        self.state = json.loads(self.state_path.read_text())
        # Operator prompts are request-scoped secrets.  Older pool versions
        # accidentally placed them under each persisted worker record; no
        # waiter/context survives a manager restart, so discard that stale
        # material during migration and keep all future chats in memory only.
        removed_persisted_chats = False
        for worker in (self.state.get("workers") or {}).values():
            if not isinstance(worker, dict):
                continue
            if "chat_pending" in worker:
                worker.pop("chat_pending", None)
                removed_persisted_chats = True
            # Long-poll liveness belongs to this manager process. A timestamp
            # persisted by an unrelated state save before restart must not let
            # the dashboard queue a secret prompt to a driver that has not
            # polled the new process yet.
            if worker.pop("last_chat_poll_unix", None) is not None:
                removed_persisted_chats = True
            normalized_commands = [
                self._command_with_id(command)
                for command in (worker.get("commands") or [])
                if isinstance(command, Mapping)
            ]
            if normalized_commands != (worker.get("commands") or []):
                worker["commands"] = normalized_commands
                removed_persisted_chats = True
            inflight = worker.get("command_inflight")
            if isinstance(inflight, Mapping):
                normalized_inflight = self._command_with_id(inflight)
                if normalized_inflight != inflight:
                    worker["command_inflight"] = normalized_inflight
                    removed_persisted_chats = True
        # A deploy marker surviving into a NEW manager process means the
        # manager went down mid-deploy: the deploy CLI driving it is now
        # orphaned (its chain-bound relaunch/registration never completes)
        # and must be re-run. Say so loudly instead of leaving a mesh
        # stuck in measurement.
        for _model_id, _entry in (self.state.get("deploys") or {}).items():
            if not isinstance(_entry, Mapping):
                continue
            _age = int(time.time()) - int(
                _entry.get("updated_at_unix", 0) or 0
            )
            if _age <= DEPLOY_STATUS_TTL_S:
                logger.warning(
                    "a deploy for %s was in flight when the previous "
                    "manager went down (stage %r, %ds ago); its deploy "
                    "client is orphaned - re-run `mesh deploy %s`",
                    _model_id,
                    str(_entry.get("stage", "") or ""),
                    _age,
                    _model_id,
                )
        raw_serving_mode = str(self.state.get("serving_mode", "") or "").strip()
        self.serving_mode = (
            _normalize_pool_serving_mode(raw_serving_mode)
            if raw_serving_mode
            else ""
        )
        migrated = removed_persisted_chats
        # Migrate the old mis-named field (stored the OWNER ACCOUNT, which is
        # usually a coldkey — never actually a hotkey).
        if "owner_hotkey" in self.state and not self.state.get("owner_account"):
            self.state["owner_account"] = self.state.pop("owner_hotkey")
            migrated = True
        # The pre-separation pool secret is a worker credential and may already
        # be present on every member. Never promote it to management. Existing
        # pools receive a new local-only operator credential on first startup.
        if not str(self.state.get("management_secret", "")):
            self.state["management_secret"] = secrets.token_urlsafe(32)
            migrated = True
        stored_coordinator_address = _normalize_evm_address(
            self.state.get("coordinator_address", "")
        )
        if self.serving_mode == POOL_SERVING_MODE_SUBNET:
            if not stored_coordinator_address:
                raise ValueError("subnet pool state has no coordinator_address")
            if coordinator_address is not None:
                supplied_coordinator_address = _normalize_evm_address(
                    coordinator_address
                )
                if supplied_coordinator_address != stored_coordinator_address:
                    raise ValueError(
                        "subnet pool coordinator_address override does not match state"
                    )
            configured_address = stored_coordinator_address
        else:
            configured_address = (
                coordinator_address
                if coordinator_address is not None
                else stored_coordinator_address
            )
        configured_shared_state = (
            validator_shared_state_path
            if validator_shared_state_path is not None
            else self.state.get("validator_shared_state_path", "")
        )
        self.coordinator_address = _normalize_evm_address(configured_address)
        self.validator_shared_state_path = str(configured_shared_state or "").strip()
        self.validator_binding: dict[str, int] = {}
        if self.serving_mode == POOL_SERVING_MODE_SUBNET:
            # validator_shared_state_path is optional (score-view
            # enrichment only; the empty-path branch in the score reader
            # degrades to the chain score cache).
            binding = self.state.get("validator_binding")
            if not isinstance(binding, Mapping):
                raise ValueError("subnet pool state has no validator_binding")
            self.validator_binding = _validator_binding(
                chain_id=binding.get("chain_id"),
                netuid=binding.get("netuid"),
                coordinator_uid=binding.get("coordinator_uid"),
                epoch=binding.get("epoch"),
                snapshot_ttl_seconds=binding.get("snapshot_ttl_seconds"),
            )
            mesh_epochs = [
                int((mesh.get("validator_binding") or {}).get("epoch"))
                for mesh in (self.state.get("meshes") or {}).values()
                if (mesh.get("validator_binding") or {}).get("epoch") is not None
            ]
            if "validator_epoch_floor" not in self.state:
                self.state["validator_epoch_floor"] = max(
                    mesh_epochs or [int(self.validator_binding["epoch"])]
                )
                migrated = True
        # Reconcile meshes a restart made unrecoverable. Persisted state can
        # outlive the processes it describes: a stopping tombstone whose
        # workers already detached will never receive another "stopped"
        # report, and a mesh whose driver record is gone entirely has nothing
        # left that could serve it OR confirm its teardown. Leaving either in
        # place is how a long-stopped mesh reappeared in fleet output after
        # a manager restart.
        workers_state = self.state.setdefault("workers", {})
        for mesh_key in list((self.state.get("meshes") or {}).keys()):
            mesh = self.state["meshes"][mesh_key]
            if not isinstance(mesh, dict):
                continue
            members = [str(m) for m in (mesh.get("members") or [])]
            driver = str(mesh.get("driver", "") or "")
            if self._finish_mesh_stop_locked(mesh_key):
                migrated = True
                continue
            if driver and driver not in workers_state:
                survivors = [m for m in members if m in workers_state]
                if survivors:
                    # _drop_mesh, NOT _fail_mesh_locked: the mesh must stay
                    # in the removable "stopping" state so the survivors'
                    # eventual stopped reports can retire it. An "error"
                    # status here would pin the entry forever (nothing
                    # removes a tombstone that is not "stopping").
                    self._drop_mesh(mesh_key)
                else:
                    self.state["meshes"].pop(mesh_key, None)
                migrated = True
        self.lock = threading.Lock()
        # Chain renewal runs outside the manager lock.  Deploy can suspend a
        # model's renewals persistently, then wait for this in-memory claim to
        # drain before deactivating its slot for recalibration.  This closes
        # the only race in which an already-started renewModel could
        # reactivate the old contract after deploy took it offline.
        self._lease_renewals_inflight: set[str] = set()
        if migrated:
            self._save()
        admin_token = MeshPoolToken(
            pool_id=str(self.state["pool_id"]),
            manager_endpoint=str(self.state["manager_endpoint"]),
            pool_secret=str(self.state["management_secret"]),
            scope=POOL_TOKEN_SCOPE_MANAGEMENT,
        )
        write_owner_only_text(
            self.state_path.with_name(POOL_ADMIN_TOKEN_FILE),
            admin_token.encode() + "\n",
        )
        # Operator chat is routed through the driver's OUTBOUND heartbeat (the
        # only channel that survives NAT/firewalls): handle_chat queues a
        # request for the driver and blocks on an Event here until the driver
        # POSTs the result back to /v1/pool/chat-result. Ephemeral, never
        # persisted — a manager restart just abandons any in-flight chat.
        self.chat_waiters: dict[str, dict[str, Any]] = {}
        # Active operator chats per mesh (mesh_key -> {chat_id: taken
        # monotonic}). Capped at the serve slot count: the backend serves
        # --parallel slots concurrently, so the chat lane admits exactly
        # that many and the next Send gets a clean "cap reached" instead
        # of wedging the coordinator. Entries whose driver died mid-flight
        # would otherwise pin a slot until the SSE deadline (300s), so a
        # lock older than CHAT_ACTIVE_MAX_S (and past its own request
        # deadline) is reclaimed: past that, no legitimate inline proof is
        # still running — the chat is dead.
        self.chat_active: dict[str, dict[str, float]] = {}
        # Streaming chat: chat_id -> queue the SSE handler drains. The driver
        # streams token deltas to /v1/pool/chat-chunk, which push onto the
        # queue; the final /v1/pool/chat-result closes it. Tokens reach the
        # browser at generation speed; the proof metadata lands on the final.
        self.chat_streams: dict[str, queue.Queue] = {}
        # Per-driver wakeups for the chat long-poll: queuing a chat sets the
        # driver's event so its blocked /v1/pool/chat-poll returns in one RTT
        # instead of waiting out a heartbeat cycle (pickup was the bulk of TTFT).
        self.chat_signals: dict[str, threading.Event] = {}
        # Ephemeral authorization/lifecycle state. Worker control signatures
        # are replay-protected in memory; manager restarts may accept a recent
        # signed request again, but every state mutation remains idempotent or
        # bound to the worker's persistent stage identity.
        self.worker_auth_nonces: dict[str, dict[str, float]] = {}
        # Recent lifecycle report acknowledgements make worker retries safe
        # even when the first HTTP response is lost after state mutation.
        self.worker_report_completed: dict[str, dict[str, Any]] = {}
        # chat_id -> assigned driver, mesh, proof-stage expectation, delivery
        # sequence, and queued/running/done state.
        self.chat_contexts: dict[str, dict[str, Any]] = {}
        # Idempotent final-result acknowledgements survive stream cleanup long
        # enough for a driver to retry after a lost HTTP response.
        self.chat_completed: dict[str, dict[str, Any]] = {}
        # worker_id -> pending operator requests.  Deliberately excluded from
        # pool-state.json so prompts are never written to disk.
        self.chat_pending: dict[str, list[dict[str, Any]]] = {}

    def _chat_signal(self, worker_id: str) -> threading.Event:
        with self.lock:
            ev = self.chat_signals.get(worker_id)
            if ev is None:
                ev = threading.Event()
                self.chat_signals[worker_id] = ev
            return ev

    def _remove_pending_chat_locked(self, worker_id: str, chat_id: str) -> None:
        pending = self.chat_pending.get(worker_id, [])
        remaining = [
            item for item in pending if str(item.get("chat_id", "")) != chat_id
        ]
        if remaining:
            self.chat_pending[worker_id] = remaining
        else:
            self.chat_pending.pop(worker_id, None)

    def _expire_unowned_chat_locked(self, chat_id: str) -> None:
        """Fail and release a queued/leased request whose deadline elapsed.

        The worker has not acknowledged pickup in either state, so no
        inference owns the request. Releasing it here prevents an expired
        prompt from retaining the per-mesh slot until the HTTP/SSE handler's
        outer timeout runs.
        """

        context = self.chat_contexts.get(chat_id)
        if context is None or str(context.get("state", "") or "") not in {
            "queued",
            "leased",
        }:
            return
        driver = str(context.get("driver", "") or "")
        self._remove_pending_chat_locked(driver, chat_id)
        self.chat_contexts.pop(chat_id, None)
        self._clear_chat_active(chat_id)
        error = "mesh request expired before the driver acknowledged pickup"
        waiter = self.chat_waiters.get(chat_id)
        if waiter is not None:
            waiter["result"] = {
                "status": "error",
                "error": error,
                "verified": False,
                "receipt_verified": False,
            }
            waiter["event"].set()
        stream_q = self.chat_streams.get(chat_id)
        if stream_q is not None:
            stream_q.put({"type": "error", "error": error})

    def _lease_ready_chats_locked(
        self,
        worker_id: str,
        *,
        now_mono: float,
    ) -> tuple[list[dict[str, Any]], float | None]:
        """Lease queued/expired chats without consuming their prompt payloads."""

        leased: list[dict[str, Any]] = []
        next_deadline: float | None = None
        expired: list[str] = []
        for item in list(self.chat_pending.get(worker_id, [])):
            chat_id = str(item.get("chat_id", "") or "")
            context = self.chat_contexts.get(chat_id)
            if context is None or str(context.get("driver", "")) != worker_id:
                continue
            request_deadline = float(
                context.get("request_deadline_mono", 0.0) or 0.0
            )
            if request_deadline and request_deadline <= now_mono:
                expired.append(chat_id)
                continue
            state = str(context.get("state", "") or "")
            deadline = float(context.get("delivery_lease_deadline_mono", 0.0) or 0.0)
            eligible = state == "queued" or (
                state == "leased" and deadline <= now_mono
            )
            if not eligible:
                if state == "leased":
                    next_deadline = (
                        deadline
                        if next_deadline is None
                        else min(next_deadline, deadline)
                    )
                continue
            token = secrets.token_hex(32)
            attempt = int(context.get("delivery_attempt", 0) or 0) + 1
            deadline = now_mono + CHAT_DELIVERY_LEASE_S
            if request_deadline:
                deadline = min(deadline, request_deadline)
            context.update(
                {
                    "state": "leased",
                    "delivery_token": token,
                    "delivery_attempt": attempt,
                    "delivery_lease_deadline_mono": deadline,
                }
            )
            leased.append(
                {
                    **dict(item),
                    "delivery_token": token,
                    "delivery_attempt": attempt,
                    "delivery_expires_at_unix_ms": int(
                        (time.time() + max(0.0, deadline - now_mono)) * 1000
                    ),
                }
            )
            next_deadline = (
                deadline if next_deadline is None else min(next_deadline, deadline)
            )
        for chat_id in expired:
            self._expire_unowned_chat_locked(chat_id)
        return leased, next_deadline

    def _guard_chat_slot(self, mesh_key: str, worker: dict[str, Any]) -> None:
        """Admit a chat up to the mesh's slot capacity, or refuse precisely.

        Caller holds ``self.lock``. The backend serves ``--parallel`` slots
        concurrently, so the chat lane admits exactly that many per mesh.
        Failure modes are distinguished so the operator never sees a
        misleading "cap reached" when the truth is a dead driver:

        * An active entry younger than ``CHAT_ACTIVE_MAX_S`` (or still
          inside its own request deadline) counts against the cap. Older
          -> that chat's driver died mid-flight; reclaim its slot (and its
          orphaned waiter/stream) and proceed.
        * The driver has not long-polled within ``CHAT_POLL_STALE_S`` -> it
          is not picking chats up at all (crashed or pre-long-poll code).
          Refuse immediately instead of queuing into the void and hanging
          until the SSE deadline.
        """
        active = self.chat_active.get(mesh_key) or {}
        now = time.monotonic()
        for active_cid in list(active):
            age = now - float(active.get(active_cid, 0.0) or 0.0)
            # A probe chat may hold its slot far beyond CHAT_ACTIVE_MAX_S:
            # honor the active chat's own request deadline (plus delivery
            # slack) before declaring its driver dead.
            active_context = self.chat_contexts.get(active_cid) or {}
            active_deadline = float(
                active_context.get("request_deadline_mono", 0.0) or 0.0
            )
            if age < CHAT_ACTIVE_MAX_S or (
                active_deadline and now < active_deadline + 30.0
            ):
                continue
            active.pop(active_cid, None)
            self.chat_waiters.pop(active_cid, None)
            orphaned_stream = self.chat_streams.pop(active_cid, None)
            if hasattr(orphaned_stream, "put"):
                orphaned_stream.put(
                    {
                        "type": "error",
                        "error": "the previous mesh test expired before completion",
                    }
                )
            orphaned_context = self.chat_contexts.pop(active_cid, None)
            if orphaned_context is not None:
                self._remove_pending_chat_locked(
                    str(orphaned_context.get("driver", "") or ""),
                    active_cid,
                )
        if not active:
            self.chat_active.pop(mesh_key, None)
        slot_cap = _serve_parallel_slots()
        if len(active) >= slot_cap:
            raise ValueError(
                f"{len(active)} chats are already running on this mesh "
                f"(slot capacity {slot_cap}); wait for one to finish"
            )
        last_poll = float(worker.get("last_chat_poll_unix", 0.0) or 0.0)
        if time.time() - last_poll > CHAT_POLL_STALE_S:
            raise ValueError(
                "mesh driver is not picking up chats; its worker is unreachable "
                "or running old code; restart the worker for this machine"
            )

    def _mark_chat_active(self, mesh_key: str, chat_id: str) -> None:
        """Take one chat slot and stamp it. Caller holds ``self.lock``."""
        self.chat_active.setdefault(mesh_key, {})[chat_id] = time.monotonic()

    def _clear_chat_active(self, chat_id: str) -> None:
        """Release any slot held by ``chat_id``. Caller holds ``self.lock``."""
        for mk in list(self.chat_active):
            if self.chat_active[mk].pop(chat_id, None) is not None:
                if not self.chat_active[mk]:
                    self.chat_active.pop(mk, None)

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        write_owner_only_json(self.state_path, self.state)

    @staticmethod
    def _command_with_id(command: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(command)
        command_id = str(payload.get("command_id", "") or "")
        if not _COMMAND_ID_RE.fullmatch(command_id):
            payload["command_id"] = "cmd-" + uuid.uuid4().hex
        supplied_digest = str(payload.pop("command_digest", "") or "")
        command_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if supplied_digest and not secrets.compare_digest(
            supplied_digest,
            command_digest,
        ):
            raise ValueError("persisted command digest does not match its payload")
        payload["command_digest"] = command_digest
        return payload

    def _queue_worker_command(
        self,
        worker: dict[str, Any],
        command: Mapping[str, Any],
    ) -> str:
        """Persist one idempotently-addressable command until completion.

        Normal commands are serialized behind the current command.  ``stop``
        is the only pre-emptive command: once a mesh is being torn down, a
        half-finished drive/join/fetch must not keep the process-owning worker
        busy while the manager advertises it as reusable.
        """

        requested = dict(command)
        action = str(requested.get("action", "") or "")
        mesh_key = str(requested.get("mesh_key", "") or "")
        inflight = worker.get("command_inflight")
        candidates = [
            *(worker.get("commands") or []),
            *([inflight] if isinstance(inflight, Mapping) else []),
        ]
        for existing in candidates:
            if (
                isinstance(existing, dict)
                and str(existing.get("action", "") or "") == action
                and str(existing.get("mesh_key", "") or "") == mesh_key
            ):
                normalized = self._command_with_id(existing)
                existing.clear()
                existing.update(normalized)
                return str(normalized["command_id"])
        payload = self._command_with_id(requested)
        commands = worker.setdefault("commands", [])
        if action == "stop":
            # A stop invalidates every not-yet-started command for this mesh.
            commands[:] = [
                queued
                for queued in commands
                if str((queued or {}).get("mesh_key", "") or "") != mesh_key
            ]
            if (
                isinstance(inflight, Mapping)
                and str(inflight.get("mesh_key", "") or "") == mesh_key
            ):
                worker["last_superseded_command"] = {
                    "command_id": str(inflight.get("command_id", "") or ""),
                    "action": str(inflight.get("action", "") or ""),
                    "mesh_key": mesh_key,
                    "superseded_at_unix": int(time.time()),
                }
                worker["command_inflight"] = payload
            else:
                commands.insert(0, payload)
        else:
            commands.append(payload)
        return str(payload["command_id"])

    def _auth(self, body: dict[str, Any]) -> None:
        if not secrets.compare_digest(
            str(body.get("pool_secret", "")),
            str(self.state["pool_secret"]),
        ):
            raise PermissionError("invalid pool token")

    def _auth_worker(
        self,
        body: dict[str, Any],
        *,
        action: str,
        allow_missing: bool = False,
        allow_unbound_join: bool = False,
    ) -> tuple[str, str]:
        """Authenticate one worker request and bind it to its stage key.

        Development pools retain the shared-token convenience path. Validator
        pools require every worker control request to carry a fresh Sr25519
        signature from the persistent stage proof key pinned at join.
        """

        self._auth(body)
        worker_id = str(body.get("worker_id", "") or "")
        if not _WORKER_ID_RE.fullmatch(worker_id):
            raise PermissionError("invalid worker identity")
        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            return worker_id, ""

        worker_session_id = str(body.get("worker_session_id", "") or "")
        if not _WORKER_AUTH_NONCE_RE.fullmatch(worker_session_id):
            raise PermissionError("worker session identity is invalid")

        if str(body.get("worker_auth_action", "")) != action:
            raise PermissionError("worker control action mismatch")
        timestamp = body.get("worker_auth_timestamp")
        if type(timestamp) is not int:
            raise PermissionError("worker control timestamp must be an integer")
        now = int(time.time())
        if abs(now - timestamp) > WORKER_CONTROL_MAX_CLOCK_SKEW_S:
            raise PermissionError("worker control timestamp is stale")
        nonce = str(body.get("worker_auth_nonce", "") or "")
        if not _WORKER_AUTH_NONCE_RE.fullmatch(nonce):
            raise PermissionError("worker control nonce is invalid")
        proof_key = str(body.get("worker_proof_key", "") or "")
        signature = str(body.get("worker_auth_signature", "") or "")
        from verallm.mesh.receipt_signing import verify_worker_control_body_hash

        if not verify_worker_control_body_hash(
            _worker_control_body_hash_hex(body),
            signature,
            proof_key,
        ):
            raise PermissionError("worker control signature is invalid")

        with self.lock:
            worker = self.state.get("workers", {}).get(worker_id)
            pinned_key = str((worker or {}).get("worker_proof_key", "") or "")
            if worker is None and not allow_missing:
                raise PermissionError("worker is not registered")
            if pinned_key and pinned_key != proof_key:
                raise PermissionError("worker stage identity does not match registration")
            if worker is not None and not pinned_key and not allow_unbound_join:
                raise PermissionError("worker must rejoin to bind its stage identity")
            pinned_session = str(
                (worker or {}).get("worker_session_id", "") or ""
            )
            if (
                worker is not None
                and action != "join"
                and pinned_session
                and not secrets.compare_digest(pinned_session, worker_session_id)
            ):
                raise PermissionError("worker control request is from a stale session")

            seen = self.worker_auth_nonces.setdefault(worker_id, {})
            cutoff = time.time() - WORKER_CONTROL_NONCE_TTL_S
            for old_nonce in [
                value for value, observed in seen.items() if observed < cutoff
            ]:
                seen.pop(old_nonce, None)
            if nonce in seen:
                raise PermissionError("worker control request was replayed")
            if len(seen) >= 4096:
                oldest = min(seen, key=seen.get)
                seen.pop(oldest, None)
            seen[nonce] = time.time()
        return worker_id, proof_key

    # -- wallet auth (operator dashboard) -----------------------------------

    # A pool is owned by one Bittensor account. The operator signs a single-use
    # challenge with that account and receives an opaque session token (only
    # its sha256 is stored). Viewing is public; managing needs the owner
    # session or the separate local management secret. Worker credentials are
    # intentionally never accepted for operator actions.

    AUTH_NONCE_TTL_S = 600.0
    AUTH_SESSION_TTL_S = 24 * 3600.0

    def handle_auth_challenge(self, body: dict[str, Any]) -> dict[str, Any]:
        # The signer is a Bittensor SS58 account. It may be the operator's
        # coldkey or hotkey; ownership is the account bound to this pool.
        account = str(body.get("account", "") or body.get("hotkey", "")).strip()
        if not (40 <= len(account) <= 50):
            raise ValueError("account must be an SS58 address")
        nonce = secrets.token_hex(16)
        with self.lock:
            now = time.time()
            self.auth_nonces = {
                n: v for n, v in getattr(self, "auth_nonces", {}).items()
                if v["exp"] > now
            }
            while len(self.auth_nonces) >= 1024:
                self.auth_nonces.pop(next(iter(self.auth_nonces)))
            self.auth_nonces[nonce] = {
                "account": account,
                "exp": now + self.AUTH_NONCE_TTL_S,
            }
            pool_id = self.state["pool_id"]
        message = (
            "verathos-operator-login\n"
            f"pool: {pool_id}\n"
            f"account: {account}\n"
            f"nonce: {nonce}"
        )
        return {"status": "ok", "message": message, "nonce": nonce}

    def handle_auth_verify(self, body: dict[str, Any]) -> dict[str, Any]:
        account = str(body.get("account", "") or body.get("hotkey", "")).strip()
        nonce = str(body.get("nonce", ""))
        signature = str(body.get("signature", "")).strip()
        with self.lock:
            entry = getattr(self, "auth_nonces", {}).pop(nonce, None)
        if entry is None or entry["exp"] < time.time() or entry["account"] != account:
            raise PermissionError("challenge expired or unknown; request a new one")
        pool_id = self.state["pool_id"]
        message = (
            "verathos-operator-login\n"
            f"pool: {pool_id}\n"
            f"account: {account}\n"
            f"nonce: {nonce}"
        ).encode()
        sig_hex = signature[2:] if signature.startswith("0x") else signature
        try:
            sig = bytes.fromhex(sig_hex)
        except ValueError:
            raise PermissionError("signature is not hex")
        # Wallets differ in what exactly signRaw signs: some sign the raw
        # bytes, most wrap them as <Bytes>...</Bytes> (and some prefix the
        # signature with a multisig type byte). Accept the established
        # Substrate wallet variants.
        from verallm.mesh.receipt_signing import _keypair_from_ss58

        try:
            kp = _keypair_from_ss58(account)
        except Exception as exc:
            raise PermissionError("account is not a valid SS58 address") from exc
        msgs = (message, b"<Bytes>" + message + b"</Bytes>")
        sigs = [sig] + ([sig[1:]] if len(sig) in (65, 66) else [])
        ok = any(
            self._sig_ok(kp, m, sg) for m in msgs for sg in sigs
        )
        if not ok:
            raise PermissionError("signature does not verify for this account")
        with self.lock:
            owner = str(self.state.get("owner_account", "") or "")
            if not owner:
                # First verified wallet CLAIMS the pool only with the local
                # management credential. The worker enrollment credential may
                # be held by every pool member and must never claim ownership.
                supplied_management_secret = str(
                    body.get("management_secret", "")
                    or body.get("pool_secret", "")  # legacy field name only
                )
                if not secrets.compare_digest(
                    supplied_management_secret,
                    str(self.state["management_secret"]),
                ):
                    raise PermissionError(
                        "pool is unclaimed: include the pool admin token once "
                        "to bind this account as its owner"
                    )
                self.state["owner_account"] = owner = account
            ttl = self.AUTH_SESSION_TTL_S
            token = secrets.token_urlsafe(32)
            sessions = self.state.setdefault("sessions", {})
            now = time.time()
            for h in [h for h, v in sessions.items() if v.get("exp", 0) < now]:
                sessions.pop(h, None)
            sessions[hashlib.sha256(token.encode()).hexdigest()] = {
                "account": account,
                "exp": now + ttl,
            }
            self._save()
        return {
            "status": "ok",
            "token": token,
            "account": account,
            "is_owner": account == owner,
            "expires_in": int(ttl),
        }

    def handle_auth_session(self, body: dict[str, Any]) -> dict[str, Any]:
        """Who-am-I for a session token. The SERVER compares the session
        account to owner_account (both stored here in one canonical form), so
        the UI never has to do fragile client-side SS58 string matching."""
        account = self._session_account(str(body.get("session", "")))
        owner = str(self.state.get("owner_account", "") or "")
        return {
            "status": "ok",
            "account": account,
            "is_owner": bool(account and account == owner),
            "signed_in": bool(account),
        }

    def handle_auth_logout(self, body: dict[str, Any]) -> dict[str, Any]:
        """Revoke a browser session immediately instead of waiting for TTL."""
        token = str(body.get("session", ""))
        if token:
            digest = hashlib.sha256(token.encode()).hexdigest()
            with self.lock:
                sessions = self.state.get("sessions") or {}
                if digest in sessions:
                    sessions.pop(digest, None)
                    self._save()
        return {"status": "ok"}

    def _session_account(self, token: str) -> str:
        if not token:
            return ""
        h = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            entry = (self.state.get("sessions") or {}).get(h)
            if entry and entry.get("exp", 0) > time.time():
                return str(entry.get("account", ""))
        return ""

    @staticmethod
    def _sig_ok(kp: Any, message: bytes, sig: bytes) -> bool:
        try:
            return bool(kp.verify(message, sig))
        except Exception:
            return False

    def _auth_manage(self, body: dict[str, Any]) -> None:
        """Management auth: local admin secret or verified owner session."""
        supplied_management_secret = str(
            body.get("management_secret", "")
            or body.get("pool_secret", "")  # legacy transport field only
        )
        if secrets.compare_digest(
            supplied_management_secret,
            str(self.state["management_secret"]),
        ):
            return
        account = self._session_account(str(body.get("session", "")))
        owner = str(self.state.get("owner_account", "") or "")
        if account and owner and account == owner:
            return
        raise PermissionError(
            "management needs the pool admin token or a verified owner-wallet session"
        )

    @staticmethod
    def _validator_state_freshness(shared: Any) -> dict[str, Any]:
        updated_at = float(getattr(shared, "updated_at", 0.0) or 0.0)
        if updated_at <= 0:
            return {
                "updated_at": updated_at,
                "age_seconds": None,
                "stale": True,
            }
        age_seconds = max(0.0, time.time() - updated_at)
        return {
            "updated_at": updated_at,
            "age_seconds": round(age_seconds, 1),
            "stale": age_seconds > VALIDATOR_SCORE_STALE_S,
        }

    @staticmethod
    def _unavailable_validator_score(
        reason: str,
        *,
        shared: Any | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "available": False,
            "source": "validator_ema",
            "reason": reason,
        }
        if shared is not None:
            raw_epoch = getattr(shared, "epoch_number", None)
            result["epoch_number"] = (
                raw_epoch
                if type(raw_epoch) is int and 0 <= raw_epoch < 2**63
                else None
            )
            result.update(PoolManager._validator_state_freshness(shared))
        return result

    def _validator_scores_for_meshes(
        self,
        meshes: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Resolve slot EMA and its latest exact mesh scoring provenance.

        The EMA is intentionally keyed by ``(coordinator EVM address,
        model_index)`` and therefore carries across topology generations.  Its
        latest completed sample separately carries the signed mesh id,
        snapshot hash/generation, epoch, and network identity that produced
        it.  The operator response exposes both records and an exact-match bit
        rather than relabeling historical evidence as the current topology.
        """

        keys = [str(key) for key in meshes]
        if not self.coordinator_address:
            unavailable = self._unavailable_validator_score(
                "coordinator identity is not configured"
            )
            return {key: dict(unavailable) for key in keys}
        if not self.validator_shared_state_path:
            unavailable = self._unavailable_validator_score(
                "validator score source is not configured"
            )
            return {key: dict(unavailable) for key in keys}
        try:
            from neurons.shared_state import read_shared_state

            shared = read_shared_state(self.validator_shared_state_path)
        except Exception as exc:
            logger.info("operator validator score state unavailable: %s", exc)
            shared = None
        if shared is None:
            unavailable = self._unavailable_validator_score(
                "validator score state is unavailable"
            )
            return {key: dict(unavailable) for key in keys}

        if self.serving_mode == POOL_SERVING_MODE_SUBNET:
            expected_chain_id = int(self.validator_binding.get("chain_id", -1))
            expected_netuid = int(self.validator_binding.get("netuid", -1))
            shared_chain_id = getattr(shared, "chain_id", None)
            shared_netuid = getattr(shared, "netuid", None)
            if type(shared_chain_id) is not int or type(shared_netuid) is not int:
                unavailable = self._unavailable_validator_score(
                    "validator score state has no chain/network binding",
                    shared=shared,
                )
                return {key: dict(unavailable) for key in keys}
            if (
                shared_chain_id != expected_chain_id
                or shared_netuid != expected_netuid
            ):
                unavailable = self._unavailable_validator_score(
                    "validator score state belongs to a different chain or subnet",
                    shared=shared,
                )
                return {key: dict(unavailable) for key in keys}

        coordinator = self.coordinator_address
        endpoints = [
            entry
            for entry in (getattr(shared, "miner_endpoints", []) or [])
            if str(getattr(entry, "address", "")).strip().lower() == coordinator
        ]

        def entry_model_index(entry: Any) -> int | None:
            try:
                value = int(getattr(entry, "model_index", -1))
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None

        scores: dict[str, Any] = {}
        ema_identity_present = False
        for address, per_model in (
            getattr(shared, "miner_ema_scores", {}) or {}
        ).items():
            if str(address).strip().lower() == coordinator and isinstance(per_model, dict):
                ema_identity_present = True
                scores.update(per_model)
        score_metadata: dict[str, Any] = {}
        for address, per_model in (
            getattr(shared, "miner_score_metadata", {}) or {}
        ).items():
            if (
                str(address).strip().lower() == coordinator
                and isinstance(per_model, dict)
            ):
                score_metadata.update(per_model)
        probation: set[int] = set()
        for address, model_indices in (
            getattr(shared, "probation_miners", {}) or {}
        ).items():
            if str(address).strip().lower() != coordinator:
                continue
            for model_index in model_indices or []:
                try:
                    probation.add(int(model_index))
                except (TypeError, ValueError):
                    continue
        blacklisted = coordinator in {
            str(address).strip().lower()
            for address in (getattr(shared, "blacklisted_addresses", []) or [])
        }
        ss58_identity: Mapping[str, Any] = {}
        for address, identity in (getattr(shared, "ss58_map", {}) or {}).items():
            if (
                str(address).strip().lower() == coordinator
                and isinstance(identity, Mapping)
            ):
                ss58_identity = identity
                break

        results: dict[str, dict[str, Any]] = {}
        raw_shared_epoch = getattr(shared, "epoch_number", None)
        if type(raw_shared_epoch) is not int or not 0 <= raw_shared_epoch < 2**63:
            unavailable = self._unavailable_validator_score(
                "validator score state has an invalid epoch binding",
                shared=shared,
            )
            return {key: dict(unavailable) for key in keys}
        shared_epoch = raw_shared_epoch
        for key, mesh in meshes.items():
            model_id = str(mesh.get("model_id", ""))
            mesh_epoch: int | None = None
            current_chain_id: int | None = None
            current_netuid: int | None = None
            if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                binding = mesh.get("validator_binding")
                if not isinstance(binding, Mapping):
                    results[str(key)] = self._unavailable_validator_score(
                        "mesh has no validator epoch binding",
                        shared=shared,
                    )
                    continue
                raw_mesh_epoch = binding.get("epoch")
                raw_chain_id = binding.get(
                    "chain_id", self.validator_binding["chain_id"]
                )
                raw_netuid = binding.get(
                    "netuid", self.validator_binding["netuid"]
                )
                if any(
                    type(value) is not int
                    for value in (raw_mesh_epoch, raw_chain_id, raw_netuid)
                ):
                    results[str(key)] = self._unavailable_validator_score(
                        "mesh validator epoch binding is invalid",
                        shared=shared,
                    )
                    continue
                mesh_epoch = raw_mesh_epoch
                current_chain_id = raw_chain_id
                current_netuid = raw_netuid
            pinned_raw = mesh.get("model_index")
            pinned_index: int | None = None
            if pinned_raw is not None:
                try:
                    pinned_index = int(pinned_raw)
                except (TypeError, ValueError):
                    results[str(key)] = self._unavailable_validator_score(
                        "mesh model index is invalid", shared=shared
                    )
                    continue
                if pinned_index < 0:
                    results[str(key)] = self._unavailable_validator_score(
                        "mesh model index is invalid", shared=shared
                    )
                    continue

            matching = [
                entry
                for entry in endpoints
                if bool(getattr(entry, "mesh_enabled", False))
                and str(getattr(entry, "model_id", "")) == model_id
            ]
            if pinned_index is not None:
                same_slot = [
                    entry
                    for entry in endpoints
                    if entry_model_index(entry) == pinned_index
                ]
                # An active endpoint at the pinned slot is a strong
                # contradiction if its model/runtime does not match this mesh.
                if same_slot and not any(
                    entry in matching
                    and entry_model_index(entry) == pinned_index
                    for entry in same_slot
                ):
                    results[str(key)] = self._unavailable_validator_score(
                        "pinned model index does not match this mesh",
                        shared=shared,
                    )
                    continue
                model_index = pinned_index
                selected = [
                    entry
                    for entry in matching
                    if entry_model_index(entry) == model_index
                ]
                if not selected:
                    results[str(key)] = self._unavailable_validator_score(
                        "pinned mesh slot is not active in validator state",
                        shared=shared,
                    )
                    continue
            else:
                model_indices = {
                    model_index
                    for entry in matching
                    if (model_index := entry_model_index(entry)) is not None
                }
                if not model_indices:
                    results[str(key)] = self._unavailable_validator_score(
                        "coordinator model mapping is unavailable",
                        shared=shared,
                    )
                    continue
                if len(model_indices) != 1:
                    results[str(key)] = self._unavailable_validator_score(
                        "coordinator model mapping is ambiguous",
                        shared=shared,
                    )
                    continue
                model_index = next(iter(model_indices))
                selected = [
                    entry
                    for entry in matching
                    if entry_model_index(entry) == model_index
                ]

            raw_score = scores.get(str(model_index))
            if not ema_identity_present:
                results[str(key)] = self._unavailable_validator_score(
                    "validator EMA is unavailable in this shared state",
                    shared=shared,
                )
                continue
            try:
                current_slot_ema = float(raw_score)
            except (TypeError, ValueError):
                results[str(key)] = self._unavailable_validator_score(
                    "validator has not scored this coordinator model yet",
                    shared=shared,
                )
                continue
            if not math.isfinite(current_slot_ema):
                results[str(key)] = self._unavailable_validator_score(
                    "validator score is invalid", shared=shared
                )
                continue
            metadata = score_metadata.get(str(model_index))
            if not isinstance(metadata, Mapping):
                results[str(key)] = self._unavailable_validator_score(
                    "validator score has no scoring provenance",
                    shared=shared,
                )
                continue
            scored_epochs = metadata.get("scored_epochs")
            last_scored_epoch = metadata.get("last_scored_epoch")
            score_epoch = metadata.get("score_epoch")
            latest_completed_ema = metadata.get("latest_completed_ema")
            if (
                type(scored_epochs) is not int
                or scored_epochs <= 0
                or type(last_scored_epoch) is not int
                or type(score_epoch) is not int
                or score_epoch != last_scored_epoch
            ):
                results[str(key)] = self._unavailable_validator_score(
                    "validator has not scored this coordinator model yet",
                    shared=shared,
                )
                continue
            if score_epoch > shared_epoch:
                results[str(key)] = self._unavailable_validator_score(
                    "validator score provenance is from a future epoch",
                    shared=shared,
                )
                continue
            try:
                completed_ema = float(latest_completed_ema)
            except (TypeError, ValueError):
                results[str(key)] = self._unavailable_validator_score(
                    "validator score has no completed EMA sample",
                    shared=shared,
                )
                continue
            if not math.isfinite(completed_ema):
                results[str(key)] = self._unavailable_validator_score(
                    "validator completed EMA sample is invalid",
                    shared=shared,
                )
                continue
            if (
                metadata.get("ema_scope") != "coordinator_model_slot"
                or metadata.get("ema_carries_across_topology") is not True
            ):
                results[str(key)] = self._unavailable_validator_score(
                    "validator score has no explicit slot EMA scope",
                    shared=shared,
                )
                continue

            latest_mesh_sample = metadata.get("latest_mesh_sample")
            if not isinstance(latest_mesh_sample, Mapping):
                results[str(key)] = self._unavailable_validator_score(
                    "validator score has no bound mesh sample provenance",
                    shared=shared,
                )
                continue
            expected_sample_fields = {
                "chain_id",
                "netuid",
                "coordinator_address",
                "model_index",
                "model_id",
                "mesh_id",
                "verification_snapshot_hash",
                "snapshot_generation",
            }
            if set(latest_mesh_sample) != expected_sample_fields or any(
                type(latest_mesh_sample.get(field)) is not int
                for field in (
                    "chain_id",
                    "netuid",
                    "model_index",
                    "snapshot_generation",
                )
            ):
                results[str(key)] = self._unavailable_validator_score(
                    "validator mesh score provenance is invalid",
                    shared=shared,
                )
                continue
            sample_chain_id = latest_mesh_sample["chain_id"]
            sample_netuid = latest_mesh_sample["netuid"]
            sample_model_index = latest_mesh_sample["model_index"]
            sample_generation = latest_mesh_sample["snapshot_generation"]
            sample_coordinator = str(
                latest_mesh_sample.get("coordinator_address", "") or ""
            ).strip().lower()
            sample_model_id = str(
                latest_mesh_sample.get("model_id", "") or ""
            )
            scored_mesh_id = str(
                latest_mesh_sample.get("mesh_id", "") or ""
            )
            scored_snapshot_hash = str(
                latest_mesh_sample.get("verification_snapshot_hash", "") or ""
            )
            if (
                not 1 <= sample_chain_id < 2**63
                or not 0 <= sample_netuid <= 65_535
                or sample_model_index < 0
                or sample_coordinator != coordinator
                or sample_model_index != model_index
                or sample_model_id != model_id
                or not scored_mesh_id
                or not 1 <= sample_generation < 2**63
                or not _COMMAND_DIGEST_RE.fullmatch(scored_snapshot_hash)
            ):
                results[str(key)] = self._unavailable_validator_score(
                    "validator mesh score provenance does not match this slot",
                    shared=shared,
                )
                continue
            if self.serving_mode == POOL_SERVING_MODE_SUBNET and (
                sample_chain_id != int(self.validator_binding["chain_id"])
                or sample_netuid != int(self.validator_binding["netuid"])
            ):
                results[str(key)] = self._unavailable_validator_score(
                    "validator mesh score provenance belongs to another network",
                    shared=shared,
                )
                continue

            current_mesh_id = str(mesh.get("mesh_id", "") or "") or None
            current_snapshot_hash = str(
                mesh.get("verification_snapshot_hash", "") or ""
            ) or None
            current_generation_raw = mesh.get("snapshot_generation")
            current_generation = (
                current_generation_raw
                if type(current_generation_raw) is int
                else None
            )
            current_topology = {
                "chain_id": current_chain_id,
                "netuid": current_netuid,
                "coordinator_address": coordinator,
                "model_index": model_index,
                "model_id": model_id,
                "mesh_id": current_mesh_id,
                "verification_snapshot_hash": current_snapshot_hash,
                "snapshot_generation": current_generation,
                "epoch": mesh_epoch,
            }
            topology_matches_latest_score = bool(
                current_chain_id == sample_chain_id
                and current_netuid == sample_netuid
                and mesh_epoch == score_epoch
                and current_mesh_id == scored_mesh_id
                and current_snapshot_hash == scored_snapshot_hash
                and current_generation == sample_generation
            )

            uid = next(
                (
                    int(getattr(entry, "uid"))
                    for entry in selected
                    if getattr(entry, "uid", None) is not None
                ),
                None,
            )
            if uid is None:
                try:
                    uid = int(ss58_identity.get("uid"))
                except (AttributeError, TypeError, ValueError):
                    uid = None
            results[str(key)] = {
                "available": True,
                "source": "validator_ema",
                # The current slot EMA includes any post-close penalties.  It
                # is deliberately not presented as evidence for the current
                # topology; the exact completed sample is nested separately.
                "score": current_slot_ema,
                "score_kind": "current_coordinator_model_slot_ema",
                "coordinator_address": coordinator,
                "model_index": model_index,
                "uid": uid,
                "epoch_number": shared_epoch,
                "scored_epochs": scored_epochs,
                "last_scored_epoch": last_scored_epoch,
                "score_epoch": score_epoch,
                "ema_scope": "coordinator_model_slot",
                "ema_carries_across_topology": True,
                "latest_score": {
                    "ema_at_completion": completed_ema,
                    "score_epoch": score_epoch,
                    "chain_id": sample_chain_id,
                    "netuid": sample_netuid,
                    "coordinator_address": sample_coordinator,
                    "model_index": sample_model_index,
                    "model_id": sample_model_id,
                    "scored_mesh_id": scored_mesh_id,
                    "scored_verification_snapshot_hash": scored_snapshot_hash,
                    "scored_snapshot_generation": sample_generation,
                },
                "current_topology": current_topology,
                "current_topology_matches_latest_score": (
                    topology_matches_latest_score
                ),
                "score_adjusted_since_completion": not math.isclose(
                    current_slot_ema,
                    completed_ema,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "epochs_since_score": shared_epoch - score_epoch,
                **self._validator_state_freshness(shared),
                "probation": model_index in probation,
                "blacklisted": blacklisted,
            }
        return results

    def handle_operator_overview(self, body: dict[str, Any]) -> dict[str, Any]:
        """Sanitized pool view for the operator UI.

        No secrets, endpoints, commands, or chat queues are returned. Managing
        credentials reveal actionable error text, while public viewers receive
        only safe state labels. Mutations still require management auth.
        """
        managed = False
        try:
            self._auth_manage(body)
            managed = True
        except PermissionError:
            pass
        # Anonymous viewers share one fixed-window budget; managing
        # credentials are never throttled (their own board polls this).
        if not managed and not self.api_request_allowed(
            "operator-overview", limit=POOL_API_RATE_LIMIT_PER_MIN
        ):
            raise PermissionError(
                "operator overview rate limit reached; retry shortly"
            )
        status = self.handle_status(
            {"management_secret": self.state["management_secret"]}
        )

        def public_worker_status(value: Any) -> str:
            raw = str(value or "idle")
            if raw in {
                "idle", "assigned", "serving", "joining", "driving",
                "fetching", "preparing backend", "error",
            }:
                return raw
            # Progress-bearing transients stay verbatim; keep every fetch
            # shape in sync with the worker's status strings — an unknown
            # legit status coercing to "error" is a recurring bug class
            # (multi-file GGUF fetch: "fetching file 2/6 20%").
            if re.fullmatch(
                r"(?:driving \((?:warming proofs(?: \d{1,3}%)?|\d{1,3}%)\)"
                r"|fetching(?: file \d{1,3}/\d{1,3})?(?: \d{1,3}%| verifying)?)",
                raw,
            ):
                return raw
            return "error"

        viewer_safe_workers = {}
        for wid, w in status["workers"].items():
            capability = w.get("capability") or {}
            viewer_safe_workers[wid] = {
                "status": w.get("status") if managed else public_worker_status(
                    w.get("status")
                ),
                "stale": bool(w.get("stale")),
                "mesh": w.get("mesh"),
                "rtt_ms": dict(w.get("rtt_ms") or {}),
                "peer_rtt_ms": dict(w.get("peer_rtt_ms") or {}),
                "capability": {
                    k: capability.get(k)
                    for k in (
                        "gpu_name", "vram_gb", "gpu_count", "per_gpu_vram_gb",
                        "gpu_names", "free_disk_gb", "rpc_device",
                        "member_only", "subnet_driver_ready",
                        "validator_driver_ready",  # pre-rename workers
                    )
                    if capability.get(k) is not None
                },
                "catalog": [
                    {
                        k: entry.get(k)
                        for k in ("model_id", "model_bytes", "layers")
                        if entry.get(k) is not None
                    }
                    for entry in (w.get("catalog") or [])
                    if entry.get("model_id")
                ],
            }
        viewer_safe_meshes = {}
        if managed:
            validator_scores = self._validator_scores_for_meshes(status["meshes"])
        else:
            unavailable = self._unavailable_validator_score(
                "owner access is required to view validator scores"
            )
            validator_scores = {
                str(key): dict(unavailable) for key in status["meshes"]
            }
        for key, mesh in status["meshes"].items():
            viewer_safe_meshes[key] = {
                k: mesh.get(k)
                for k in (
                    "mesh_key", "model_id", "members", "driver", "status",
                    "serving", "driver_verified", "driver_stale",
                    "member_stale", "stale_members", "routing_ready",
                    "created_at_unix", "model_index",
                )
                if mesh.get(k) is not None
            }
            # Existing single-worker meshes predate the write-side fix and
            # carry no driver_verified key: their drive() already verified
            # generation, so derive true instead of "pending" forever.
            if (
                not viewer_safe_meshes[key].get("driver_verified")
                and str(mesh.get("status", "")) == "serving"
                and len(mesh.get("members", []) or []) <= 1
            ):
                viewer_safe_meshes[key]["driver_verified"] = True
            viewer_safe_meshes[key]["validator_score"] = validator_scores[key]
            if mesh.get("error"):
                viewer_safe_meshes[key]["error"] = (
                    str(mesh["error"])
                    if managed
                    else "mesh reported an error; owner access shows details"
                )
        viewer_safe_models = {}
        for model_id, entry in (status.get("models") or {}).items():
            try:
                model_bytes = int(entry.get("model_bytes", 0) or 0)
                layers = int(entry.get("layers", 0) or 0)
                raw_max_context_len = entry.get("max_context_len", 0)
                max_context_len = (
                    raw_max_context_len
                    if (
                        type(raw_max_context_len) is int
                        and 1 <= raw_max_context_len < 2**32
                    )
                    else 0
                )
            except (TypeError, ValueError, OverflowError):
                model_bytes = 0
                layers = 0
                max_context_len = 0
            launch_ready = model_bytes > 0
            if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                launch_ready = (
                    launch_ready
                    and entry.get("model_index") is not None
                    and layers > 0
                    and max_context_len > 0
                    and bool(str(entry.get("quantization_scheme", "")).strip())
                    and all(
                        bool(re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))))
                        for field in (
                            "model_package_hash",
                            "model_tensor_manifest_root",
                            "tokenizer_hash",
                        )
                    )
                )
            viewer_safe_models[model_id] = {
                **{
                    k: entry.get(k)
                    for k in ("model_bytes", "layers", "max_context_len")
                    if entry.get(k) is not None
                },
                "launch_ready": launch_ready,
            }
        # Models a worker holds locally (catalog entry without a download
        # source) are launchable in dev pools even though the registry
        # never learned them; hiding them made the fleet view contradict
        # a serving mesh.
        for worker in status["workers"].values():
            for item in worker.get("catalog") or []:
                model_id = str(item.get("model_id", "") or "")
                if not model_id or model_id in viewer_safe_models:
                    continue
                model_bytes = int(item.get("model_bytes", 0) or 0)
                viewer_safe_models[model_id] = {
                    **{
                        k: item.get(k)
                        for k in ("model_bytes", "layers")
                        if item.get(k) is not None
                    },
                    "launch_ready": (
                        self.serving_mode != POOL_SERVING_MODE_SUBNET
                        and model_bytes > 0
                    ),
                }
        return {
            "status": "ok",
            "pool_id": status["pool_id"],
            "serving_mode": self.serving_mode,
            "owner_account": str(self.state.get("owner_account", "") or ""),
            "subtensor_network": self._subtensor_network(),
            **(
                {
                    # Wallet names and the stored registration (public URL,
                    # lease expiry) are operator detail; public viewers get
                    # the mode and network only.
                    "wallet_name": str(
                        self.state.get("wallet_name", "") or ""
                    ),
                    "wallet_hotkey": str(
                        self.state.get("wallet_hotkey", "") or ""
                    ),
                    "coordinator_hotkey_ss58": str(
                        self.state.get("coordinator_hotkey_ss58", "") or ""
                    ),
                    **_registration_view(self.state),
                }
                if managed
                else {}
            ),
            **(
                {
                    "validator_binding": dict(self.validator_binding),
                    "active_validator_epochs": sorted(
                        {
                            int(
                                (mesh.get("validator_binding") or {}).get(
                                    "epoch"
                                )
                            )
                            for mesh in status["meshes"].values()
                            if (mesh.get("validator_binding") or {}).get(
                                "epoch"
                            )
                            is not None
                        }
                    ),
                }
                if self.serving_mode == POOL_SERVING_MODE_SUBNET
                else {}
            ),
            "workers": viewer_safe_workers,
            "meshes": viewer_safe_meshes,
            "models": viewer_safe_models,
        }

    def handle_api_keys(self, body: dict[str, Any]) -> dict[str, Any]:
        """Mint/list/revoke private-API keys (management-authed).

        Keys authenticate the pool's OpenAI-compatible surface
        (GET /v1/models, POST /v1/chat/completions); they are stored as
        sha256 hashes in the owner-only pool state and the cleartext key
        is returned exactly once, at mint time.
        """
        self._auth_manage(body)
        from verallm.mesh import apikeys

        action = str(body.get("action", "list") or "list")
        with self.lock:
            if action == "create":
                key, record = apikeys.mint_pool_api_key(
                    self.state, name=str(body.get("name", "") or "")
                )
                self._save()
                return {"status": "ok", "api_key": key, "record": record}
            if action == "revoke":
                revoked = apikeys.revoke_pool_api_key(
                    self.state, str(body.get("key_id", "") or "")
                )
                if revoked:
                    self._save()
                return {"status": "ok", "revoked": revoked}
            if action == "list":
                return {
                    "status": "ok",
                    "keys": apikeys.list_pool_api_keys(self.state),
                }
        raise ValueError("action must be create, list, or revoke")

    def authenticate_api_key(self, authorization_header: str) -> str:
        """Bearer auth for the private OpenAI surface.

        Returns the key's stored hash ("" on failure) so the caller can
        rate limit per key without touching the cleartext."""
        from verallm.mesh import apikeys

        with self.lock:
            return apikeys.authenticate_pool_api_key(
                self.state, authorization_header
            )

    def api_request_allowed(self, bucket: str, *, limit: int) -> bool:
        """Fixed one-minute-window counter behind the private API limits."""

        window = int(time.time() // 60)
        with self.lock:
            buckets = getattr(self, "_api_rate_buckets", None)
            if buckets is None:
                buckets = self._api_rate_buckets = {}
            seen_window, count = buckets.get(bucket, (window, 0))
            if seen_window != window:
                count = 0
            if count >= limit:
                buckets[bucket] = (window, count)
                return False
            buckets[bucket] = (window, count + 1)
            if len(buckets) > 4096:
                for stale in [
                    key
                    for key, (w, _c) in buckets.items()
                    if w != window
                ]:
                    buckets.pop(stale, None)
            return True

    def handle_operator_access(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the worker onboarding token to an authenticated operator."""
        self._auth_manage(body)
        token = MeshPoolToken(
            pool_id=str(self.state["pool_id"]),
            manager_endpoint=str(self.state["manager_endpoint"]),
            pool_secret=str(self.state["pool_secret"]),
            scope=POOL_TOKEN_SCOPE_WORKER,
        )
        return {
            "status": "ok",
            "pool_id": token.pool_id,
            "manager_endpoint": token.manager_endpoint,
            "pool_token": token.encode(),
        }

    def update_chain_identity(
        self,
        *,
        subtensor_network: str = "",
        wallet_name: str = "",
        wallet_hotkey: str = "",
        coordinator_hotkey_ss58: str = "",
    ) -> None:
        """Persist the chain identity the manager actually runs with.

        Wallet name/hotkey and the subtensor network historically lived only
        in the manager's PM2 argv, so nothing on disk could say which hotkey
        a pool is bound to and every status endpoint that needed the network
        (handle_operator_score) came up empty. Called from ``mesh pool
        serve`` startup, before the server loop, so it backfills existing
        pools without a recreate. Only fills/updates non-empty values.
        """

        updates = {
            "subtensor_network": str(subtensor_network or "").strip(),
            "wallet_name": str(wallet_name or "").strip(),
            "wallet_hotkey": str(wallet_hotkey or "").strip(),
            "coordinator_hotkey_ss58": str(
                coordinator_hotkey_ss58 or ""
            ).strip(),
        }
        changed = False
        for key, value in updates.items():
            if value and self.state.get(key) != value:
                self.state[key] = value
                changed = True
        if changed:
            self._save()

    def _subtensor_network(self) -> str:
        """The bittensor network name for metagraph reads ("" if unknown)."""

        explicit = str(
            self.state.get("subtensor_network", "")
            or self.state.get("subtensor", "")
            or ""
        ).strip()
        if explicit:
            return explicit
        return subtensor_network_for_chain_id(
            (self.validator_binding or {}).get("chain_id")
        )

    def _epoch_blocks(self) -> int:
        """Epoch length in blocks, from the hosted subnet config (cached).

        Validators follow the AUTHORITATIVE hosted subnet-config for
        epoch_blocks; a manager that hardcodes the historical 360 default
        signs snapshots under a different epoch numbering the moment the
        owner changes the epoch length, and every canary then rejects the
        mesh ("snapshot still binds the previous epoch"
        when testnet moved to 180-block epochs). Resolved from the chain's
        gleipnir base, cached 5 minutes, 360 on any failure.
        """

        cache = getattr(self, "_epoch_blocks_cache", None)
        now = time.time()
        if cache and now < cache["exp"]:
            return int(cache["value"])
        value = VALIDATOR_EPOCH_BLOCKS
        try:
            from verallm.mesh.manifest_store import (
                default_store_urls_for_chain,
            )

            chain_id = int(
                (self.validator_binding or {}).get("chain_id", 0) or 0
            )
            for base in default_store_urls_for_chain(chain_id):
                try:
                    raw = urllib.request.urlopen(
                        base.rstrip("/") + "/subnet-config.json", timeout=10.0
                    ).read()
                    blocks = int(
                        (json.loads(raw).get("epoch") or {}).get(
                            "epoch_blocks", 0
                        )
                        or 0
                    )
                    if blocks > 0:
                        value = blocks
                        break
                except Exception as exc:
                    logger.debug(
                        "hosted epoch_blocks fetch failed via %s: %s",
                        base,
                        exc,
                    )
        except Exception:  # pragma: no cover - never block epoch math
            pass
        self._epoch_blocks_cache = {"value": value, "exp": now + 300.0}
        return int(value)

    def _current_chain_epoch(self) -> int | None:
        """Current scoring epoch from the chain (cached; None when unknown).

        One block-number read per interval, success cached briefly and
        failure backed off serving the stale value, so the rate-limited
        public RPCs never see a hot loop.
        """

        network = self._subtensor_network()
        if not network:
            return None
        cache = getattr(self, "_epoch_cache", None)
        now = time.time()
        if cache and cache["exp"] > now:
            return cache["epoch"]
        try:
            import bittensor as bt  # heavy; only when chain-configured

            # Reuse one client across polls: a fresh websocket handshake
            # per query trips public-entrypoint rate limits (observed
            # HTTP 429 on the test entrypoint, stalling epoch rotation).
            sub = getattr(self, "_epoch_subtensor", None)
            if sub is None:
                from verallm.mesh.allowlist import normalize_subtensor_network

                _Ctor = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
                sub = _Ctor(network=normalize_subtensor_network(network))
                self._epoch_subtensor = sub
            # HARD deadline on the read: a reused client whose websocket
            # died silently blocks recv forever, which froze the follower
            # thread for hours (observed: snapshot never rotated, no
            # error ever logged). The fresh-per-poll code never had this
            # hazard because handshakes carry their own timeouts.
            block = _call_bounded(
                lambda: int(sub.get_current_block()), timeout_s=15.0
            )
        except Exception as exc:
            logger.warning("chain epoch read failed (%s): %s", network, exc)
            # Drop the client so the next attempt reconnects fresh.
            self._epoch_subtensor = None
            stale = cache["epoch"] if cache else None
            self._epoch_cache = {"epoch": stale, "exp": now + 120.0}
            return stale
        epoch_blocks = self._epoch_blocks()
        epoch = block // epoch_blocks
        boundary_s = (
            (epoch + 1) * epoch_blocks - block
        ) * 12.0
        # Never cache past the epoch boundary: validators pin snapshots in
        # a short grace window at epoch START, so the follower must observe
        # the flip within seconds (a full cache interval of lag lost the
        # mesh an entire epoch, observed).
        self._epoch_cache = {
            "epoch": epoch,
            "exp": now + max(5.0, min(60.0, boundary_s)),
            "block": block,
            "at": now,
        }
        return epoch

    def seconds_to_next_epoch(self) -> float | None:
        """Estimated seconds until the next epoch boundary (None if unknown)."""

        cache = getattr(self, "_epoch_cache", None)
        if not cache or cache.get("block") is None:
            return None
        eta = (
            (cache["epoch"] + 1) * self._epoch_blocks() - cache["block"]
        ) * 12.0
        return max(0.0, eta - (time.time() - cache["at"]))

    def follow_chain_epoch(self) -> None:
        """Keep chain-bound meshes signed for the CURRENT scoring epoch.

        Validators pin each mesh's signed verification snapshot per epoch
        and exclude any mesh whose snapshot binds a different one (the
        anti-replay guard), so a long-lived mesh must refresh its binding
        every ~72 minutes. The refresh is a LOCAL re-sign on the driver
        (queued as a ``rotate`` command): no chain writes, no relaunch, the
        model stays loaded. Called periodically by the manager's epoch
        follower thread; every step is idempotent and a missed tick only
        delays the next validator pin by one poll interval.
        """

        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            return
        epoch = self._current_chain_epoch()
        if epoch is None:
            return
        with self.lock:
            dirty = False
            binding = dict(self.validator_binding or {})
            if binding and int(binding.get("epoch", -1)) < epoch:
                # Stage for future launches too, respecting the same floor
                # the manual set-epoch enforces.
                floor = int(
                    self.state.get(
                        "validator_epoch_floor", binding.get("epoch", -1)
                    )
                )
                if epoch > floor:
                    binding["epoch"] = epoch
                    self.validator_binding = _validator_binding(**binding)
                    self.state["validator_binding"] = dict(
                        self.validator_binding
                    )
                    dirty = True
            for mesh_key, mesh in self.state["meshes"].items():
                if (
                    _normalize_pool_serving_mode(mesh.get("serving_mode"))
                    != POOL_SERVING_MODE_SUBNET
                ):
                    continue
                if mesh.get("model_index") is None:
                    continue
                if mesh.get("status") != "serving":
                    continue
                mesh_binding = dict(mesh.get("validator_binding") or {})
                if int(mesh_binding.get("epoch", -1)) >= epoch:
                    continue
                driver = self.state["workers"].get(
                    str(mesh.get("driver", "") or "")
                )
                if driver is None:
                    continue
                self._queue_worker_command(
                    driver,
                    {
                        "action": "rotate",
                        "mesh_key": str(mesh_key),
                        "epoch": int(epoch),
                    },
                )
                dirty = True
            if dirty:
                self._save()

    def _score_cache_path(self) -> Path:
        return self.state_path.parent / "chain-score-cache.json"

    def _load_persisted_score(self, cache_key: tuple[str, str]) -> None:
        """Seed the in-memory score cache from disk, already expired.

        The metagraph read takes 10-30s cold and the in-memory cache dies
        with the process, so every manager restart used to render "chain
        status unavailable" until a full chain round-trip completed. The
        last good value is served immediately (marked stale) while a
        background refresh replaces it.
        """
        try:
            raw = json.loads(self._score_cache_path().read_text())
        except (OSError, ValueError):
            return
        val = raw.get("val")
        if raw.get("key") != "|".join(cache_key) or not isinstance(val, dict):
            return
        if val.get("available"):
            self._score_cache = {
                **getattr(self, "_score_cache", {}),
                cache_key: {"val": val, "exp": 0.0, "good": True},
            }

    def _persist_score(
        self, cache_key: tuple[str, str], val: dict[str, Any]
    ) -> None:
        try:
            write_owner_only_json(
                self._score_cache_path(),
                {
                    "key": "|".join(cache_key),
                    "val": val,
                    "ts": time.time(),
                },
            )
        except OSError:  # pragma: no cover - disk-full etc., cache only
            pass

    def handle_operator_score(self, body: dict[str, Any]) -> dict[str, Any]:
        """Best-effort UID + on-chain score for the pool's miner identity.

        The manager may run without chain access (pure dev pool), so this is
        allowed to say "not available" rather than fabricate a number — the UI
        then simply shows the fleet health it can measure locally. The netuid
        comes from the pool's validator_binding and the network from the
        persisted chain identity (chain-id fallback for pools started before
        the identity was persisted). A successful metagraph read that finds
        NO UID reports ``registered: false`` so the UI can distinguish
        "deregistered" from "chain unreachable".

        This never blocks a caller behind the chain: a fresh cache answers
        directly, an expired-but-good cache (including the value persisted
        across restarts) answers stale NOW and refreshes on a background
        thread, and while the first-ever fetch is in flight callers get
        ``pending: true`` so the UI can say "fetching" instead of
        "unavailable". Only the very first fetch with no history runs on
        the caller's thread (the startup warmer normally is that caller).
        """
        account = self._session_account(str(body.get("session", ""))) or str(
            self.state.get("owner_account", "") or ""
        )
        hotkey_ss58 = str(self.state.get("coordinator_hotkey_ss58", "") or "")
        binding_netuid = (self.validator_binding or {}).get("netuid")
        netuid = (
            binding_netuid
            if binding_netuid is not None
            else self.state.get("netuid")
        )
        network = self._subtensor_network()
        if (not account and not hotkey_ss58) or netuid is None or not network:
            return {"status": "ok", "available": False, "uid": None}
        cache_key = (account, hotkey_ss58)
        if cache_key not in getattr(self, "_score_cache", {}):
            self._load_persisted_score(cache_key)
        hit = getattr(self, "_score_cache", {}).get(cache_key)
        if hit and hit["exp"] > time.time():
            return {"status": "ok", **hit["val"]}
        if hit and hit.get("good"):
            # Serve the last good value immediately and refresh behind the
            # caller's back: board clients wait ~3s while the metagraph
            # read takes 10-30s, so a blocking refresh here rendered
            # "unavailable" on every cache expiry and every restart.
            self._spawn_score_refresh(
                cache_key, account, hotkey_ss58, netuid, network
            )
            return {"status": "ok", **hit["val"], "stale": True}
        if cache_key in getattr(self, "_score_refresh_inflight", set()):
            return {
                "status": "ok",
                "available": False,
                "uid": None,
                "pending": True,
            }
        return self._refresh_operator_score(
            cache_key, account, hotkey_ss58, netuid, network
        )

    def _spawn_score_refresh(
        self,
        cache_key: tuple[str, str],
        account: str,
        hotkey_ss58: str,
        netuid: Any,
        network: str,
    ) -> None:
        thread = threading.Thread(
            target=self._refresh_operator_score,
            args=(cache_key, account, hotkey_ss58, netuid, network),
            daemon=True,
            name="pool-score-refresh",
        )
        # Kept for tests and diagnostics; the duplicate-suppression lives
        # in _refresh_operator_score's inflight claim, not here.
        self._score_refresh_thread = thread
        thread.start()

    def _refresh_operator_score(
        self,
        cache_key: tuple[str, str],
        account: str,
        hotkey_ss58: str,
        netuid: Any,
        network: str,
    ) -> dict[str, Any]:
        # The public subtensor RPCs (testnet especially) are heavily rate
        # limited, and this box already hits them from the allowlist
        # refresher and the lease renewer. Cache successes LONG, keep
        # serving the last good value through failures (marked stale), and
        # never retry a failure faster than the backoff: on-chain incentive
        # moves per epoch (~72 min), so freshness is cheap to give up.
        score_success_ttl_s = 600.0
        score_failure_retry_s = 120.0
        lock = getattr(self, "_score_refresh_lock", None)
        if lock is None:
            lock = self._score_refresh_lock = threading.Lock()
        with lock:
            inflight = getattr(self, "_score_refresh_inflight", None)
            if inflight is None:
                inflight = self._score_refresh_inflight = set()
            if cache_key in inflight:
                return {
                    "status": "ok",
                    "available": False,
                    "uid": None,
                    "pending": True,
                }
            inflight.add(cache_key)
        try:
            return self._refresh_operator_score_locked(
                cache_key,
                account,
                hotkey_ss58,
                netuid,
                network,
                score_success_ttl_s,
                score_failure_retry_s,
            )
        finally:
            with lock:
                inflight.discard(cache_key)

    def _refresh_operator_score_locked(
        self,
        cache_key: tuple[str, str],
        account: str,
        hotkey_ss58: str,
        netuid: Any,
        network: str,
        score_success_ttl_s: float,
        score_failure_retry_s: float,
    ) -> dict[str, Any]:
        cache = getattr(self, "_score_cache", {})
        hit = cache.get(cache_key)
        val: dict[str, Any] = {"available": False, "uid": None}
        try:
            import bittensor as bt  # heavy; only when chain-configured

            _Ctor = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            sub = _Ctor(network=network)
            mg = sub.metagraph(int(netuid))
            hotkeys = [str(x) for x in list(mg.hotkeys)]
            coldkeys = [str(x) for x in list(getattr(mg, "coldkeys", []))]
            # The pool's miner identity is the coordinator HOTKEY when known;
            # otherwise the proven account, which is either a hotkey (its own
            # UID) or a coldkey (owns the hotkeys at one or more UIDs).
            uids = []
            role = "hotkey"
            if hotkey_ss58 and hotkey_ss58 in hotkeys:
                uids = [hotkeys.index(hotkey_ss58)]
            elif account in hotkeys:
                uids = [hotkeys.index(account)]
            elif account in coldkeys:
                uids = [i for i, c in enumerate(coldkeys) if c == account]
                role = "coldkey"
            base = {
                "netuid": int(netuid),
                "network": network,
                "queried_hotkey": hotkey_ss58 or None,
            }
            if uids:
                # Metagraph attribute names vary across bittensor versions
                # (long names vs single-letter tensors; trust is absent in
                # some releases entirely). Read defensively: a missing
                # metric renders as None, never an AttributeError that
                # blanks the whole panel (observed: 'Metagraph'
                # object has no attribute 'trust').
                def metric(names, uid, digits):
                    for name in names:
                        values = getattr(mg, name, None)
                        if values is None:
                            continue
                        try:
                            if uid < len(values):
                                return round(float(values[uid]), digits)
                        except (TypeError, ValueError, IndexError):
                            continue
                    return None

                def one(uid):
                    return {
                        "uid": int(uid),
                        "hotkey": hotkeys[uid] if uid < len(hotkeys) else None,
                        "score": metric(("trust", "T"), uid, 3),
                        "incentive": metric(("incentive", "I"), uid, 4),
                        "stake": metric(("stake", "S"), uid, 2),
                        "emission": metric(("emission", "E"), uid, 6),
                    }
                rows = [one(u) for u in uids]
                primary = max(rows, key=lambda r: r["incentive"] or 0.0)
                val = {"available": True, "registered": True, **base,
                       **primary, "uids": rows, "role": role}
            else:
                # The chain answered: this identity holds no UID. Serving
                # keeps working (registration is a deploy-time concern);
                # the UI renders the no-earnings warning from this.
                val = {"available": True, "registered": False,
                       "uid": None, **base}
            self._score_cache = {
                **cache,
                cache_key: {
                    "val": val,
                    "exp": time.time() + score_success_ttl_s,
                    "good": True,
                },
            }
            self._persist_score(cache_key, val)
            return {"status": "ok", **val}
        except Exception as exc:
            # bittensor's import reconfigures logging in-process, so a log
            # line alone can vanish; the reason rides the response too and
            # the board/dashboard can say WHY chain status is unavailable.
            val["error"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning(
                "operator score lookup failed (%s rate limits are common; "
                "serving the last good value if any): %s",
                network,
                exc,
            )
        if hit and hit.get("good"):
            # Stale beats blank: the board keeps showing the last real
            # UID/incentive with a stale marker instead of "unavailable".
            val = {**hit["val"], "stale": True}
            self._score_cache = {
                **cache,
                cache_key: {
                    "val": val,
                    "exp": time.time() + score_failure_retry_s,
                    "good": True,
                },
            }
        else:
            self._score_cache = {
                **cache,
                cache_key: {
                    "val": val,
                    "exp": time.time() + score_failure_retry_s,
                },
            }
        return {"status": "ok", **val}

    # -- worker-facing -----------------------------------------------------

    def _drop_mesh(self, mesh_key: str) -> None:
        """Begin a confirmed teardown of every mesh member (caller holds lock).

        The mesh remains as a ``stopping`` tombstone and its workers remain
        reserved until each signed ``stopped`` report completes its stop
        command.  Marking a worker idle when merely *queueing* stop can
        double-book VRAM while the old coordinator/RPC processes still run.
        """
        mesh = self.state["meshes"].get(mesh_key)
        if not mesh:
            return
        mesh["status"] = "stopping"
        mesh["remove_after_stop"] = True
        for mid in mesh.get("members", []):
            mw = self.state["workers"].get(mid)
            if mw is not None and mw.get("mesh") == mesh_key:
                self._queue_worker_command(
                    mw,
                    {"action": "stop", "mesh_key": mesh_key},
                )
                mw["status"] = "stopping"
        self._finish_mesh_stop_locked(mesh_key)

    def _finish_mesh_stop_locked(self, mesh_key: str) -> bool:
        """Remove a stopping tombstone after no worker remains assigned."""

        mesh = self.state["meshes"].get(mesh_key)
        if (
            not isinstance(mesh, Mapping)
            or str(mesh.get("status", "") or "") != "stopping"
            or not bool(mesh.get("remove_after_stop"))
        ):
            return False
        if any(
            str((self.state["workers"].get(member_id) or {}).get("mesh", "") or "")
            == mesh_key
            for member_id in (mesh.get("members") or [])
        ):
            return False
        self.state["meshes"].pop(mesh_key, None)
        return True

    def handle_join(self, body: dict[str, Any]) -> dict[str, Any]:
        if (
            self.serving_mode == POOL_SERVING_MODE_DEV
            and not str(body.get("worker_id", "") or "")
        ):
            body = {**body, "worker_id": "w-" + uuid.uuid4().hex[:10]}
        worker_id, proof_key = self._auth_worker(
            body,
            action="join",
            allow_missing=True,
            allow_unbound_join=True,
        )
        capability = dict(body.get("capability") or {})
        if (
            self.serving_mode == POOL_SERVING_MODE_SUBNET
            and str(capability.get("stage_proof_key", "") or "") != proof_key
        ):
            raise PermissionError(
                "worker capability stage identity does not match signed join"
            )
        if self.serving_mode == POOL_SERVING_MODE_SUBNET and not bool(
            capability.get("member_only")
        ):
            # PRODUCTION JOINS ONLY: a driver-capable worker on a subnet
            # pool must advertise an address validators can actually dial.
            # Non-interactive joins on rented docker boxes auto-detected
            # the CONTAINER IP (172.x) and produced workers that looked
            # healthy until deploy failed its endpoint preflight - twice
            # in one day. Refuse at join, with the fix
            # in the message; LAN/member-only slices stay legitimate via
            # --member-only.
            import ipaddress

            advertised_hosts = set()
            for ep in (body.get("endpoints") or {}).values():
                host = str(ep or "")
                host = host.split("//", 1)[-1].split("/", 1)[0]
                host = host.rsplit(":", 1)[0] if ":" in host else host
                if host:
                    advertised_hosts.add(host)
            for host in sorted(advertised_hosts):
                try:
                    addr = ipaddress.ip_address(host)
                except ValueError:
                    continue  # hostname: resolvable names are the operator's call
                if addr.is_private or addr.is_loopback or addr.is_link_local:
                    raise PermissionError(
                        f"subnet pool join refused: '{worker_id}' advertises "
                        f"{host}, which no validator can dial. Re-run the join "
                        "with --advertise-host <public-ip> and the box's "
                        "mapped --mesh-port/--proof-port/--rpc-port (verify "
                        "the provider's port mapping first), or pass "
                        "--member-only for a LAN-only slice worker."
                    )
        with self.lock:
            existing_worker = self.state["workers"].get(worker_id)
            # _auth_worker() deliberately releases the manager lock before the
            # comparatively large join payload is installed.  Re-check the
            # identity here so two concurrent first joins for the same worker
            # id cannot both observe an unbound slot and let the last writer
            # replace the first worker's pinned stage key.
            if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                existing_key = str(
                    (existing_worker or {}).get("worker_proof_key", "") or ""
                )
                if existing_key and existing_key != proof_key:
                    raise PermissionError(
                        "worker stage identity does not match registration"
                    )
                duplicate_owner = next(
                    (
                        other_id
                        for other_id, other in self.state["workers"].items()
                        if other_id != worker_id
                        and str(other.get("worker_proof_key", "") or "")
                        == proof_key
                    ),
                    None,
                )
                if duplicate_owner is not None:
                    raise PermissionError(
                        "worker stage identity is already registered to "
                        f"{duplicate_owner}"
                    )
            # A (re)joining worker restarted, so any mesh it belonged to is
            # no longer trustworthy. Begin a confirmed stop on every member.
            # Preserve this worker's control record below so the new process
            # receives that stop instead of silently becoming reusable while
            # an orphaned backend may still own its GPU/ports.
            incoming_session = str(body.get("worker_session_id", "") or "")
            same_session = bool(
                existing_worker is not None
                and incoming_session
                and secrets.compare_digest(
                    incoming_session,
                    str(existing_worker.get("worker_session_id", "") or ""),
                )
            )
            if not same_session:
                for mk in [
                    k for k, m in self.state["meshes"].items()
                    if worker_id in m.get("members", [])
                ]:
                    self._drop_mesh(mk)
            if existing_worker is not None and not same_session:
                legacy_mesh = str(existing_worker.get("mesh", "") or "")
                if legacy_mesh and legacy_mesh not in self.state["meshes"]:
                    self._queue_worker_command(
                        existing_worker,
                        {"action": "stop", "mesh_key": legacy_mesh},
                    )
                    existing_worker["status"] = "stopping"

            worker_record = {
                "worker_id": worker_id,
                **(
                    {"worker_session_id": str(body["worker_session_id"])}
                    if _WORKER_AUTH_NONCE_RE.fullmatch(
                        str(body.get("worker_session_id", "") or "")
                    )
                    else {}
                ),
                "capability": capability,
                "catalog": list(body.get("catalog") or []),
                "endpoints": dict(body.get("endpoints") or {}),
                "status": "idle",
                "mesh": "",
                "last_seen_unix": int(time.time()),
                "rtt_ms": {},
                "peer_rtt_ms": {},
                "commands": [],
                "command_inflight": None,
                **({"worker_proof_key": proof_key} if proof_key else {}),
            }
            if existing_worker is not None:
                for field_name in (
                    "status",
                    "mesh",
                    "commands",
                    "command_inflight",
                    "last_command_ack",
                    "last_command_completion",
                    "last_superseded_command",
                ):
                    if field_name in existing_worker:
                        worker_record[field_name] = existing_worker[field_name]
            self.state["workers"][worker_id] = worker_record
            # Pool-wide model registry: any catalog entry that names a download
            # source teaches the pool where to fetch that model, so a DRIVER
            # that lacks it can auto-fetch at launch instead of being refused.
            for c in body.get("catalog") or []:
                if c.get("model_id") and c.get("hf_repo") and c.get("hf_files"):
                    model_id = str(c["model_id"])
                    registry = self.state.setdefault("model_registry", {})
                    # Never let a worker advert erase operator-pinned chain/model
                    # anchors.  It may teach the pool a download source, but only
                    # the management-authenticated registration route may bind a
                    # validator-served model.
                    existing = dict(registry.get(model_id) or {})
                    advertised = {
                        "hf_repo": str(c["hf_repo"]),
                        "hf_files": [str(f) for f in c["hf_files"]],
                        "layers": int(c.get("layers", 0) or 0),
                        "model_bytes": int(c.get("model_bytes", 0) or 0),
                        # Tolerances must survive this hop too: a mesh launched
                        # via auto-fetch reads them from the REGISTRY (the
                        # driver has no catalog entry yet), and dropping them
                        # here would re-run the default band on exactly the
                        # models that need the override.
                        **_operator_tuning_fields(c),
                    }
                    if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                        # Worker credentials are intentionally weaker than the
                        # management credential. In validator mode they may fill
                        # an absent download hint, never overwrite an operator's
                        # model registration or proof tolerances.
                        registry[model_id] = {
                            **advertised,
                            **existing,
                        }
                    else:
                        registry[model_id] = {
                            **existing,
                            **advertised,
                        }
            probe_target = self.state["workers"][worker_id]["endpoints"].get("proof", "")
            self._save()

        def _measure() -> None:
            ms = _tcp_rtt_ms(probe_target, timeout=2.0)
            if ms is None:
                return
            with self.lock:
                worker = self.state["workers"].get(worker_id)
                if worker is not None:
                    worker["rtt_ms"]["manager"] = round(ms, 2)
                    self._save()

        # DNS/connect can stall (e.g. unresolvable hosts); never block a join.
        threading.Thread(target=_measure, daemon=True).start()
        response = {
            "status": "joined",
            "worker_id": worker_id,
            "pool_id": self.state["pool_id"],
        }
        if self.serving_mode == POOL_SERVING_MODE_SUBNET:
            # Chain coordinates so a token-only worker can keep its own
            # validator allowlist fresh without any flag: the token really
            # is the whole setup. (The allowlist itself stays a local chain
            # read on the worker; only the WHERE comes from the pool.)
            network = self._subtensor_network()
            netuid = (self.validator_binding or {}).get("netuid")
            if network:
                response["subtensor_network"] = network
            if netuid is not None:
                response["netuid"] = int(netuid)
        return response

    def _reconcile_orphaned_meshes_locked(self) -> bool:
        """Flip active-status mesh records that no worker is assigned to.

        Every live mesh keeps member back-references (``worker["mesh"] ==
        mesh_key``) from launch through teardown, so a record in an active
        status that NO worker references any more can never advance again:
        nothing will ever report for it. Such records previously rendered
        as "serving" indefinitely — after a roll to a new mesh key,
        pool-state.json kept the dead key "serving" and misled operators
        and tooling that read pool state instead of asking the
        coordinator. Orphans arise from crash recovery,
        records persisted by older code, and supersession — the healing
        belongs in code, not in hand-edited state files.

        "stopping" tombstones are owned by ``_finish_mesh_stop_locked``;
        "error"/"stopped" records already tell the truth and keep their
        operator-facing message. Returns True when anything changed.
        """

        changed = False
        for mesh_key, mesh in list(self.state["meshes"].items()):
            if str(mesh.get("status", "") or "") not in (
                "serving",
                "driving",
                "joining",
                "fetching",
            ):
                continue
            if any(
                str(
                    (self.state["workers"].get(member_id) or {}).get(
                        "mesh", ""
                    )
                    or ""
                )
                == mesh_key
                for member_id in (mesh.get("members") or [])
            ):
                continue
            mesh["status"] = "error"
            mesh["error"] = (
                "no worker is assigned to this mesh any more (superseded "
                "by a newer launch or recovered from a stale record); it "
                "cannot serve — run `mesh stop` to clear it"
            )
            logger.warning(
                "mesh %s reconciled to error: no member worker references "
                "it any more (members=%s)",
                mesh_key,
                list(mesh.get("members") or []),
            )
            changed = True
        return changed

    def _fail_mesh_locked(self, mesh: dict[str, Any], message: str) -> None:
        """Mark one mesh failed and reserve members until stop completes."""

        mesh["status"] = "error"
        mesh["error"] = str(message)[:500]
        for worker_id in mesh.get("members", []):
            worker = self.state["workers"].get(worker_id)
            if worker is None:
                continue
            self._queue_worker_command(
                worker,
                {"action": "stop", "mesh_key": mesh["mesh_key"]},
            )
            worker["status"] = "stopping"

    def _queue_member_join(
        self,
        mesh: dict[str, Any],
        member_id: str,
    ) -> None:
        """Queue exactly one serialized member admission command."""

        member = self.state["workers"].get(member_id)
        if member is None:
            raise ValueError(f"mesh member is unavailable: {member_id}")
        self._queue_worker_command(
            member,
            {
                "action": "join",
                "mesh_key": mesh["mesh_key"],
                "join_token": mesh["join_token"],
                # Peer for content-addressed proof-weight blobs: a member
                # without the GGUF fetches its slice from the driver,
                # hash-verified.
                "coordinator_endpoint": mesh["coordinator_endpoint"],
                **_operator_tuning_fields(mesh),
            },
        )

    def _maybe_mark_mesh_serving(self, mesh: dict[str, Any]) -> bool:
        expected = [
            member_id
            for member_id in mesh.get("members", [])
            if member_id != mesh.get("driver")
        ]
        if (
            set(mesh.get("serving", [])) >= set(expected)
            and bool(mesh.get("driver_verified", False))
        ):
            changed = mesh.get("status") != "serving"
            mesh["status"] = "serving"
            driver = self.state["workers"].get(str(mesh.get("driver", "")))
            if driver is not None:
                changed = changed or driver.get("status") != "serving"
                driver["status"] = "serving"
            mesh.pop("pending_join_members", None)
            if changed:
                self._verify_advertised_endpoint_async(mesh)
            return changed
        return False

    def _verify_advertised_endpoint_async(self, mesh: dict[str, Any]) -> None:
        """Verify a chain-bound mesh's public endpoint FROM THE MANAGER.

        Validators dial the registered endpoint; a mesh whose provider lost
        its port forwarding comes up "serving" while being dark to every
        validator and silently rides its probation forever.
        The manager sits off-box, so its dial sees what validators see -
        including hairpin-less NATs where the box itself could never
        self-verify. Async so the heartbeat handler never blocks on a
        network dial; on persistent failure the mesh flips to ERROR with
        the actionable cause instead of pretending to serve.
        """

        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            return
        if mesh.get("model_index") is None:
            return
        endpoint = str(mesh.get("coordinator_endpoint", "") or "").rstrip("/")
        mesh_key = str(mesh.get("mesh_key", "") or "")
        if not endpoint or not mesh_key:
            return

        def _probe() -> None:
            import urllib.request as _request

            last_error = ""
            for _attempt in range(6):
                try:
                    with _request.urlopen(
                        endpoint + "/health", timeout=8.0
                    ) as resp:
                        if resp.status == 200:
                            return
                        last_error = f"HTTP {resp.status}"
                except Exception as exc:
                    last_error = str(exc)[:120]
                time.sleep(5.0)
            with self.lock:
                current = (self.state.get("meshes") or {}).get(mesh_key)
                if (
                    current is None
                    or current.get("status") != "serving"
                    or str(
                        current.get("coordinator_endpoint", "") or ""
                    ).rstrip("/")
                    != endpoint
                ):
                    return
                current["status"] = "error"
                current["error"] = (
                    f"advertised endpoint {endpoint} is unreachable from the "
                    "pool manager, so validators cannot canary this mesh "
                    f"({last_error}); check the provider port forwarding / "
                    "firewall and relaunch"
                )
                logger.warning(
                    "mesh %s marked error: advertised endpoint %s "
                    "unreachable from the manager (%s)",
                    mesh_key,
                    endpoint,
                    last_error,
                )

        threading.Thread(
            target=_probe,
            name=f"endpoint-verify-{mesh_key}",
            daemon=True,
        ).start()

    def _record_serial_member_serving(
        self,
        mesh: dict[str, Any],
        worker_id: str,
    ) -> bool:
        """Advance admission only for the one member currently in flight.

        The explicit in-flight member is the sole expected member that is
        neither already serving nor still pending. This makes a repeated
        report/heartbeat idempotent and prevents a later worker from skipping
        ahead by reporting ``serving`` before it received its join command.
        """

        expected = [
            member_id
            for member_id in mesh.get("members", [])
            if member_id != mesh.get("driver")
        ]
        if worker_id not in expected:
            return False
        serving = mesh.setdefault("serving", [])
        if worker_id in serving:
            return False
        pending = mesh.setdefault("pending_join_members", [])
        in_flight = [
            member_id
            for member_id in expected
            if member_id not in serving and member_id not in pending
        ]
        if in_flight != [worker_id]:
            return False
        serving.append(worker_id)
        if pending:
            self._queue_member_join(mesh, str(pending.pop(0)))
        self._maybe_mark_mesh_serving(mesh)
        return True

    def _capacity_roster_cache_path(self) -> Path:
        return Path(self.state_path).parent / "capacity-roster-cache.json"

    def _load_capacity_roster_cache(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._capacity_roster_cache_path().read_text())
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_capacity_roster_cache(self, cache: Mapping[str, Any]) -> None:
        try:
            write_owner_only_json(self._capacity_roster_cache_path(), dict(cache))
        except OSError as exc:
            logger.debug("capacity roster cache not persisted: %s", exc)

    def _capacity_audit_context_locked(
        self, worker_id: str, worker: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """The capacity-audit context a mesh-bound worker needs, or None.

        Delivered on every heartbeat to workers serving a CHAIN-BOUND subnet
        mesh: the confirmed chain slot, the signed GPU roster, and this
        worker's share of the global ordinals. The roster is built from the
        capabilities every member advertised at join and signed with the
        coordinator EVM key (the chain entry's address), cached until the
        membership or capabilities change. Caller holds the state lock.
        """

        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            return None
        mesh_key = str(worker.get("mesh", "") or "")
        mesh = self.state.get("meshes", {}).get(mesh_key)
        if not isinstance(mesh, Mapping) or mesh.get("model_index") is None:
            return None
        model_id = str(mesh.get("model_id", "") or "")
        registration = (
            self.state.get("mesh_registrations") or {}
        ).get(model_id) or {}
        # Only a lease-confirmed registration is a chain contract; a mesh
        # launched on a pending prediction must not derive audit windows
        # for a slot the chain never saw.
        if (
            registration.get("index") is None
            or int(registration["index"]) != int(mesh["model_index"])
        ):
            return None
        # Only the mesh named by the stored registration is the chain
        # contract. A MEASUREMENT mesh for an already-registered model
        # carries the same model_index, and arming audits on it can make the
        # calibrated workload saturate the GPUs mid-probe-gate. The snapshot was
        # re-issued mid-gate, and the gate failed snapshot_binding (the
        # validator refused every receipt as "unknown audit slot" anyway,
        # because the roster it scheduled against was the registered
        # mesh's).
        if str(registration.get("mesh_key", "") or "") != mesh_key:
            return None
        binding = mesh.get("validator_binding") or {}
        wallet_name = str(self.state.get("wallet_name", "") or "")
        wallet_hotkey = str(self.state.get("wallet_hotkey", "") or "")
        if not wallet_name or not wallet_hotkey or not binding:
            return None

        roster_workers: list[dict[str, Any]] = []
        for wid in mesh.get("members") or []:
            member = self.state.get("workers", {}).get(wid) or {}
            capability = member.get("capability") or {}
            gpu_names = [
                str(name)
                for name in (capability.get("gpu_names") or [])
                if str(name or "")
            ]
            per_gpu = [
                int(value or 0)
                for value in (capability.get("per_gpu_vram_gb") or [])
            ]
            if not gpu_names and str(capability.get("gpu_name", "") or ""):
                gpu_names = [str(capability["gpu_name"])]
                per_gpu = [int(capability.get("vram_gb", 0) or 0)]
            endpoints = member.get("endpoints") or {}
            roster_workers.append(
                {
                    "worker_id": wid,
                    "backend": capability.get("rpc_device", ""),
                    "gpu_names": gpu_names,
                    "per_gpu_vram_gb": per_gpu,
                    "host": str(
                        endpoints.get("proof", "")
                        or endpoints.get("rpc", "")
                        or ""
                    ),
                }
            )

        cache = getattr(self, "_capacity_roster_cache", None)
        if cache is None:
            cache = self._load_capacity_roster_cache()
            self._capacity_roster_cache = cache
        # The binding EPOCH is deliberately excluded: it advances on every
        # relaunch, and a re-signed roster at the CURRENT epoch is invisible
        # to selection (both sides only group on rosters frozen before the
        # epoch) while its digest invalidates every in-flight receipt
        # ("roster_digest mismatch"). The
        # roster document rotates ONLY when its CONTENT would change:
        # membership, capabilities, chain identity, or the serving contract.
        binding_identity = {
            k: v for k, v in dict(binding).items() if k != "epoch"
        }
        cache_key = hashlib.sha256(
            canonical_json_bytes(
                {
                    "model_index": int(mesh["model_index"]),
                    "binding": binding_identity,
                    "workers": roster_workers,
                    "registration": {
                        "endpoint": str(registration.get("endpoint", "") or ""),
                        "quant": str(registration.get("quant", "") or ""),
                        "max_context_len": int(
                            registration.get("max_context_len", 0) or 0
                        ),
                    },
                }
            )
        ).hexdigest()
        # Keyed by the CHAIN SLOT, not the mesh key: relaunches mint fresh
        # mesh keys while the slot (and its audit obligation) is the same.
        slot_cache_key = f"idx{int(mesh['model_index'])}"
        cached = cache.get(slot_cache_key)
        if not cached or cached.get("cache_key") != cache_key:
            try:
                from verallm.chain.wallet import (
                    derive_evm_address,
                    derive_evm_private_key,
                )
                from verallm.mesh.capacity_roster import (
                    build_roster,
                    sign_roster,
                )
                from verallm.mesh.receipt_signing import (
                    load_hotkey_keypair,
                    load_hotkey_seed,
                )

                keypair = load_hotkey_keypair(wallet_name, wallet_hotkey)
                hotkey_seed = load_hotkey_seed(
                    wallet_name, wallet_hotkey, keypair=keypair
                )
                evm_address = derive_evm_address(hotkey_seed).lower()
                roster = build_roster(
                    chain_id=int(binding["chain_id"]),
                    netuid=int(binding["netuid"]),
                    address=evm_address,
                    model_index=int(mesh["model_index"]),
                    roster_epoch=int(binding["epoch"]),
                    workers=roster_workers,
                )
                signature = sign_roster(
                    roster, derive_evm_private_key(hotkey_seed)
                )
            except Exception as exc:
                logger.warning(
                    "capacity roster build failed for mesh %s: %s",
                    mesh_key,
                    exc,
                )
                return None
            cached = {
                "cache_key": cache_key,
                "roster": roster,
                "signature": signature,
                "evm_address": evm_address,
            }
            cache[slot_cache_key] = cached
            # Slots come and go; never let dead entries accumulate.
            live_slot_keys = {
                f"idx{int(m['model_index'])}"
                for m in (self.state.get("meshes") or {}).values()
                if m.get("model_index") is not None
            }
            for stale_key in [k for k in cache if k not in live_slot_keys]:
                cache.pop(stale_key, None)
            # Survive manager restarts: a re-signed roster at the restart
            # epoch would orphan every audit scheduled against the old one.
            self._save_capacity_roster_cache(cache)

        from verallm.mesh.capacity_roster import roster_cuda_gpus

        local_gpus = [
            {
                "ordinal": gpu.ordinal,
                "local_gpu_index": gpu.local_gpu_index,
                "gpu_name": gpu.gpu_name,
                "vram_gb": gpu.vram_gb,
            }
            for gpu in roster_cuda_gpus(cached["roster"])
            if gpu.worker_id == worker_id
        ]
        return {
            "mesh_key": mesh_key,
            "slot": {
                "chain_id": int(binding["chain_id"]),
                "netuid": int(binding["netuid"]),
                "address": cached["evm_address"],
                "model_index": int(mesh["model_index"]),
                "endpoint": str(registration.get("endpoint", "") or ""),
                "model_id": model_id,
                "quant": str(registration.get("quant", "") or ""),
                "max_context_len": int(
                    registration.get("max_context_len", 0) or 0
                ),
            },
            "roster": cached["roster"],
            "roster_signature": cached["signature"],
            "local_gpus": local_gpus,
        }

    def handle_heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
        worker_id, _proof_key = self._auth_worker(
            body,
            action="heartbeat",
            allow_missing=True,
        )
        with self.lock:
            worker = self.state["workers"].get(worker_id)
            if worker is None:
                return {"status": "unknown-worker"}
            dirty = False
            worker["last_seen_unix"] = int(time.time())
            mesh_key = str(worker.get("mesh", "") or "")
            mesh = self.state["meshes"].get(mesh_key) if mesh_key else None
            runtime_error = str(body.get("runtime_error", "") or "").strip()
            if "subnet_driver_ready" in body or "validator_driver_ready" in body:
                worker.setdefault("capability", {})["subnet_driver_ready"] = bool(
                    body.get(
                        "subnet_driver_ready",
                        # pre-rename workers still heartbeat the old key
                        body.get("validator_driver_ready"),
                    )
                )
            if body.get("status") and not runtime_error:
                reported = str(body["status"])
                # A worker reserved for a live mesh (launch accepted; its own
                # command not picked up yet — e.g. a member waiting for the
                # driver to finish fetching/loading) still self-reports "idle".
                # Don't let that clobber the reservation, or the machine could
                # be double-booked into a second mesh mid-formation.
                reserved = worker.get("mesh") and worker["mesh"] in self.state["meshes"]
                if not (reported == "idle" and reserved):
                    previous = str(worker.get("status", "") or "")
                    worker["status"] = reported
                    # Persist status TRANSITIONS (idle->fetching->serving...)
                    # so pool-state.json stops lying to tooling that reads
                    # the file instead of the live status route (handoff
                    # #6). Progress ticks ("fetching 42%") share a first
                    # word and stay off the disk-write path — persisting
                    # every beat is exactly what the lazy-save design
                    # exists to avoid.
                    if previous.split(" ", 1)[:1] != reported.split(" ", 1)[:1]:
                        dirty = True
                # Heartbeat status is observational only. Admission and
                # command completion require the signed, command-bound report
                # route; otherwise a member could claim ``serving`` before the
                # driver has even published its mesh runtime and join token.
            if (
                runtime_error
                and mesh is not None
                and mesh.get("status") not in ("error", "stopped", "stopping")
            ):
                self._fail_mesh_locked(
                    mesh,
                    f"worker {worker_id} runtime failed: {runtime_error}",
                )
                dirty = True
            # The worker measures its own round-trip to the coordinator (the
            # meaningful "distance"); firewalls/NAT stop the manager probing
            # back, so trust the worker's number.
            if body.get("rtt_manager") is not None:
                worker["rtt_ms"]["manager"] = round(float(body["rtt_manager"]), 2)
            # Worker->worker probe results, keyed by peer worker id, feed both
            # the recommender (pick low-mutual-latency pipelines) and the map
            # (cluster network-close machines). Replace (not merge) so stale
            # peers drop when they are no longer probed.
            if "probe_results" in body:
                worker["peer_rtt_ms"] = {
                    str(pid): round(float(ms), 2)
                    for pid, ms in (body["probe_results"] or {}).items()
                    if ms is not None
                }
            command_ack = str(body.get("command_ack", "") or "")
            if command_ack and not _COMMAND_ID_RE.fullmatch(command_ack):
                raise ValueError("command_ack is not a valid command id")
            inflight = worker.get("command_inflight")
            if not isinstance(inflight, dict):
                inflight = None
            acknowledged = ""
            if (
                inflight is not None
                and command_ack
                and str(inflight.get("command_id", "") or "") == command_ack
            ):
                if str(worker.get("last_command_ack", "") or "") != command_ack:
                    worker["last_command_ack"] = command_ack
                    dirty = True
                acknowledged = command_ack
            elif (
                command_ack
                and command_ack == str(worker.get("last_command_ack", "") or "")
            ):
                # The prior heartbeat response may have been lost. Re-echo
                # only an ACK the manager actually recorded, never arbitrary
                # syntactically valid input supplied by the worker.
                acknowledged = command_ack
            if inflight is None and worker.get("commands"):
                queued = worker["commands"].pop(0)
                inflight = self._command_with_id(queued)
                worker["command_inflight"] = inflight
                dirty = True
            elif inflight is not None:
                normalized = self._command_with_id(inflight)
                if normalized != inflight:
                    worker["command_inflight"] = normalized
                    inflight = normalized
                    dirty = True
            command = dict(inflight) if inflight is not None else None
            if dirty:
                # Persist the in-flight command BEFORE writing the HTTP
                # response. Receipt ACKs are informational; only an exact
                # command-bound terminal report clears this record.
                self._save()
            # Tell the worker which peers to probe next. Use the peer's
            # PROOF HTTP endpoint, not its rpc port: the ggml rpc-server
            # accepts a single client, so while a mesh holds it every new
            # connect sits unaccepted until the probe times out - the
            # latency map went blank the moment every GPU was serving.
            # RTT measures network distance; any always-accepting TCP
            # port on the peer works. Skip loopback-advertised peers
            # (127.0.0.1/localhost) - probing those would hit the
            # prober's own machine, not the peer.
            probe = {}
            for wid, w in self.state["workers"].items():
                if wid == worker_id:
                    continue
                endpoints = w.get("endpoints", {}) or {}
                ep = str(endpoints.get("proof", "") or "") or str(
                    endpoints.get("rpc", "") or ""
                )
                if ep and not _is_loopback(ep):
                    probe[wid] = ep
            capacity_context = self._capacity_audit_context_locked(
                worker_id, worker
            )
        # last_seen/rtt are refreshed every beat and rebuilt on reconnect, so
        # NOT persisting them keeps the beat off the disk-write path (the state
        # file write under the lock was serializing all workers' heartbeats).
        # Chat is delivered by the long-poll (/v1/pool/chat-poll), not here.
        self._maybe_auto_relaunch_registered(worker_id)
        return {
            "status": "ok",
            "command": command,
            "command_acknowledged": acknowledged,
            "probe": probe,
            **(
                {"capacity_audit_context": capacity_context}
                if capacity_context
                else {}
            ),
        }

    # Registered models must SERVE: a mesh that dies (worker crash, killed
    # serve, host reboot) leaves an on-chain registration with nothing
    # behind it, and until now the recovery was a manual relaunch. When an
    # idle worker that already holds the model's files heartbeats in, the
    # manager relaunches the mesh itself. Guards: chain-bound
    # registrations only, never while any record for the model is still
    # live, never after an operator stop (until the next explicit
    # launch), one attempt per model per cooldown window so a
    # crash-looping serve cannot be hammered.
    AUTO_RELAUNCH_COOLDOWN_S = 600.0
    # A worker restart briefly removes its old assignment before the new
    # session has finished joining.  Give that session a small reconnect
    # window before treating the registration as needing placement elsewhere.
    AUTO_RELAUNCH_DELAY_S = 10.0

    def _maybe_auto_relaunch_registered(self, worker_id: str) -> None:
        now = time.monotonic()
        last_map = getattr(self, "_auto_relaunch_last", None)
        if last_map is None:
            last_map = self._auto_relaunch_last = {}
        candidate_model = ""
        with self.lock:
            worker = self.state["workers"].get(worker_id)
            if worker is None or worker.get("status") != "idle":
                return
            catalog_models = {
                str(item.get("model_id", "") or "")
                for item in (worker.get("catalog") or [])
            }
            if not catalog_models:
                return
            live_models = set()
            for mesh_key, mesh in self.state["meshes"].items():
                model_id = str(mesh.get("model_id", "") or "")
                status = str(mesh.get("status", "") or "")
                assigned = any(
                    str((self.state["workers"].get(member_id) or {}).get(
                        "mesh", ""
                    ) or "") == mesh_key
                    for member_id in (mesh.get("members") or [])
                )
                # A detached error/stopped record is a removable tombstone.
                # Every other record, and every record whose workers are
                # still assigned, blocks relaunch until teardown converges.
                if status not in ("error", "stopped") or assigned:
                    live_models.add(model_id)
            for model_id, registration in self._registrations_locked().items():
                if registration.get("index") is None:
                    continue
                if registration.get("suspended_by_operator"):
                    continue
                if model_id in live_models:
                    continue
                if model_id not in catalog_models:
                    continue
                if now - last_map.get(model_id, 0.0) < (
                    self.AUTO_RELAUNCH_COOLDOWN_S
                ):
                    continue
                last_map[model_id] = now
                candidate_model = model_id
                break
            if not candidate_model:
                return
            management_secret = str(self.state["management_secret"])
        model_id = candidate_model
        logger.warning(
            "auto-relaunch: registered model %s has no live mesh; "
            "relaunch triggered by idle worker %s (its catalog holds the "
            "model); selecting a feasible worker set",
            model_id,
            worker_id,
        )

        def _relaunch() -> None:
            time.sleep(self.AUTO_RELAUNCH_DELAY_S)
            try:
                preferred_driver = ""
                with self.lock:
                    registration = self._registrations_locked().get(model_id)
                    registered_endpoint = ""
                    if isinstance(registration, Mapping):
                        registered_endpoint = str(
                            registration.get("endpoint", "") or ""
                        ).strip()
                    if registered_endpoint:
                        try:
                            registered_endpoint = normalize_endpoint(
                                registered_endpoint
                            )
                        except ValueError:
                            registered_endpoint = ""
                    if registered_endpoint:
                        for candidate_id, candidate in self.state[
                            "workers"
                        ].items():
                            if candidate.get("status") != "idle":
                                continue
                            candidate_models = {
                                str(item.get("model_id", "") or "")
                                for item in (candidate.get("catalog") or [])
                            }
                            if model_id not in candidate_models:
                                continue
                            candidate_endpoint = str(
                                (candidate.get("endpoints") or {}).get(
                                    "mesh", ""
                                )
                                or ""
                            ).strip()
                            if not candidate_endpoint:
                                continue
                            try:
                                candidate_endpoint = normalize_endpoint(
                                    candidate_endpoint
                                )
                            except ValueError:
                                continue
                            if candidate_endpoint == registered_endpoint:
                                preferred_driver = str(candidate_id)
                                break
                self.handle_launch(
                    {
                        "management_secret": management_secret,
                        "model_id": model_id,
                        "_auto_relaunch": True,
                        **(
                            {"_preferred_driver": preferred_driver}
                            if preferred_driver
                            else {}
                        ),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "auto-relaunch of %s on %s failed (next attempt after "
                    "cooldown): %s",
                    model_id,
                    worker_id,
                    str(exc)[:200],
                )

        threading.Thread(target=_relaunch, daemon=True).start()

    def handle_report(self, body: dict[str, Any]) -> dict[str, Any]:
        worker_id, _proof_key = self._auth_worker(body, action="report")
        event = str(body.get("event", ""))
        mesh_key = str(body.get("mesh_key", ""))
        command_id = str(body.get("command_id", "") or "")
        command_digest = str(body.get("command_digest", "") or "")
        if not _COMMAND_ID_RE.fullmatch(command_id):
            raise ValueError("worker report requires a valid command_id")
        if not _COMMAND_DIGEST_RE.fullmatch(command_digest):
            raise ValueError("worker report requires a valid command_digest")
        report_key = f"{worker_id}:{command_id}:{mesh_key}:{event}"
        report_fingerprint = hashlib.sha256(
            canonical_json_bytes(
                {
                    str(key): value
                    for key, value in body.items()
                    if key
                    not in {
                        "pool_secret",
                        "worker_auth_action",
                        "worker_auth_timestamp",
                        "worker_auth_nonce",
                        "worker_auth_signature",
                        "worker_proof_key",
                        "worker_session_id",
                    }
                }
            )
        ).hexdigest()
        with self.lock:
            now = time.time()
            self.worker_report_completed = {
                key: value
                for key, value in self.worker_report_completed.items()
                if float(value.get("expires", 0.0) or 0.0) > now
            }
            if report_key in self.worker_report_completed:
                completed = self.worker_report_completed[report_key]
                if completed.get("fingerprint") != report_fingerprint:
                    raise ValueError("conflicting retry for completed worker report")
                return dict(completed.get("response") or {
                    "status": "ok",
                    "duplicate": True,
                })
            worker = self.state["workers"].get(worker_id)
            if worker is None:
                return {"status": "unknown"}

            last_completion = worker.get("last_command_completion")
            if (
                isinstance(last_completion, Mapping)
                and str(last_completion.get("command_id", "") or "") == command_id
            ):
                if not secrets.compare_digest(
                    str(last_completion.get("command_digest", "") or ""),
                    command_digest,
                ):
                    raise ValueError("completed command digest conflict")
                if (
                    str(last_completion.get("mesh_key", "") or "") == mesh_key
                    and str(last_completion.get("event", "") or "") == event
                ):
                    if str(last_completion.get("report_fingerprint", "") or "") != (
                        report_fingerprint
                    ):
                        raise ValueError(
                            "conflicting retry for durably completed worker report"
                        )
                    return {
                        "status": "ok",
                        "duplicate": True,
                        "command_completed": command_id,
                    }
                return {"status": "stale", "command_id": command_id}

            inflight = worker.get("command_inflight")
            if not isinstance(inflight, Mapping) or str(
                inflight.get("command_id", "") or ""
            ) != command_id:
                return {"status": "stale", "command_id": command_id}
            expected_digest = str(inflight.get("command_digest", "") or "")
            if not _COMMAND_DIGEST_RE.fullmatch(expected_digest) or not (
                secrets.compare_digest(expected_digest, command_digest)
            ):
                raise PermissionError("worker report command digest mismatch")
            if str(inflight.get("mesh_key", "") or "") != mesh_key:
                raise PermissionError("worker report mesh does not match its command")

            action = str(inflight.get("action", "") or "")
            allowed_events = {
                "drive": {"drive_ready", "serving", "error"},
                "join": {"serving", "error"},
                "fetch": {"fetched", "error"},
                "stop": {"stopped", "error"},
                "rotate": {"rotated", "error"},
            }
            if event not in allowed_events.get(action, set()):
                raise ValueError(
                    f"worker report event {event or '(empty)'} is invalid for "
                    f"command action {action or '(empty)'}"
                )

            mesh = self.state["meshes"].get(mesh_key)
            if mesh is None and action != "stop":
                # mesh_gone tells the worker its spawn serves a record that
                # no longer exists; the worker MUST tear that serve down. A
                # bare "stale" (command-identity races below) must not — the
                # two cases carry different obligations.
                return {
                    "status": "stale",
                    "command_id": command_id,
                    "mesh_gone": True,
                }
            if (
                mesh is not None
                and action in ("drive", "join")
                and str(mesh.get("status", "") or "")
                in ("error", "stopped", "stopping")
            ):
                # A drive/join completing AFTER its mesh was failed or torn
                # down must not resurrect the record. Without this, a driver
                # that finishes forming after the silent-driver reap keeps
                # an unsupervised serve on the GPU that no pool state
                # references: coordinator + llama hold the model's full
                # VRAM for a mesh the manager already failed, and shadow
                # the worker's ports against every later launch. Tell the
                # worker to tear its spawn down instead.
                return {
                    "status": "stale",
                    "command_id": command_id,
                    "mesh_gone": True,
                }
            terminal = True
            fail_message = ""
            response: dict[str, Any] = {"status": "ok"}

            if action == "stop":
                if event == "stopped":
                    worker["status"] = "idle"
                    worker["mesh"] = ""
                else:
                    fail_message = str(
                        body.get("message", "") or "stop failed"
                    )[:500]
                    # A stop that ERRORS must still converge. The dominant
                    # error class is "worker restarted while the command was
                    # running" - the restart itself already killed the serve
                    # (the fresh daemon's generation reap owns that), and
                    # leaving the assignment in place wedged the mesh in
                    # "stopping" for 40+ minutes while every relaunch was
                    # refused. Release the worker; the
                    # error text still reaches the operator via the mesh
                    # record below.
                    worker["status"] = "idle"
                    worker["mesh"] = ""
                    if mesh is not None:
                        # A teardown tombstone must stay "stopping": the
                        # error text alone informs the operator, while an
                        # "error" status would make the tombstone
                        # unremovable (removal is gated on "stopping") and
                        # wedge the mesh AND its workers until a manual
                        # re-stop.
                        if not bool(mesh.get("remove_after_stop")):
                            mesh["status"] = "error"
                        mesh["error"] = (
                            f"worker {worker_id} could not stop safely: {fail_message}"
                        )[:500]
            else:
                assert mesh is not None
                members = [str(value) for value in (mesh.get("members") or [])]
                if worker_id not in members:
                    raise PermissionError("worker does not belong to the reported mesh")
                if str(worker.get("mesh", "") or "") != mesh_key:
                    raise PermissionError("worker is not assigned to the reported mesh")
                driver = str(mesh.get("driver", "") or "")
                if (
                    event in {"drive_ready", "fetched", "rotated"}
                    and worker_id != driver
                ):
                    raise PermissionError(f"only the mesh driver may report {event}")

            if action == "drive" and event == "drive_ready":
                mesh_id = str(body.get("mesh_id", "") or "")
                join_token = str(body.get("join_token", "") or "")
                coordinator_endpoint = str(
                    body.get("coordinator_endpoint", "") or ""
                )
                verification_snapshot_hash = str(
                    body.get("verification_snapshot_hash", "") or ""
                )
                # The driver measured how much context this hardware actually
                # holds. Record it against the MODEL so register-model can
                # commit that exact number on chain (a miner is scored on
                # context and canaried at its registered maximum, so a typed
                # guess either loses score or fails an honest canary).
                measured_ctx_budget = body.get("measured_ctx_budget")
                if (
                    type(measured_ctx_budget) is int
                    and 1 <= measured_ctx_budget < 2**32
                ):
                    # Bind deploy calibration to this exact mesh generation.
                    # The model-level copy remains a compatibility/status
                    # summary, but replacement deploys must consume this
                    # per-mesh value so stale measurements from prior hardware
                    # cannot be reused.
                    mesh["measured_ctx_budget"] = int(measured_ctx_budget)
                    registry = self.state.setdefault("model_registry", {})
                    model_entry = registry.setdefault(
                        str(mesh.get("model_id", "") or ""), {}
                    )
                    model_entry["measured_ctx_budget"] = int(measured_ctx_budget)
                if not mesh_id or not join_token or not coordinator_endpoint:
                    raise ValueError(
                        "drive_ready requires mesh_id, join_token, and "
                        "coordinator_endpoint"
                    )
                non_driver = [
                    member_id
                    for member_id in mesh["members"]
                    if member_id != mesh["driver"]
                ]
                if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                    # UNREGISTERED subnet launches (no chain slot) run the
                    # operator measurement lane: they sign no snapshot by
                    # design, so requiring - or accepting - one here would
                    # either dead-loop the drive (report rejected, drive
                    # re-runs forever) or let a snapshot appear with no
                    # chain anchors behind it.
                    mesh_chain_bound = mesh.get("model_index") is not None
                    if non_driver and verification_snapshot_hash:
                        raise ValueError(
                            "validator multi-worker drive_ready must defer its "
                            "verification snapshot until final admission"
                        )
                    if not mesh_chain_bound and verification_snapshot_hash:
                        raise ValueError(
                            "unregistered subnet drive_ready must not carry a "
                            "verification snapshot hash"
                        )
                    if (
                        not non_driver
                        and mesh_chain_bound
                        and not _COMMAND_DIGEST_RE.fullmatch(
                            verification_snapshot_hash
                        )
                    ):
                        raise ValueError(
                            "validator single-worker drive_ready requires a "
                            "verification snapshot hash"
                        )
                existing_drive = (
                    str(mesh.get("mesh_id", "") or ""),
                    str(mesh.get("join_token", "") or ""),
                    str(mesh.get("coordinator_endpoint", "") or ""),
                    str(mesh.get("verification_snapshot_hash", "") or ""),
                )
                reported_drive = (
                    mesh_id,
                    join_token,
                    coordinator_endpoint,
                    verification_snapshot_hash,
                )
                if any(existing_drive):
                    if existing_drive != reported_drive:
                        raise ValueError(
                            "driver attempted to replace an established mesh runtime"
                        )
                    response["duplicate"] = True
                else:
                    if str(mesh.get("status", "")) != "driving":
                        raise ValueError(
                            "drive_ready is invalid in the current mesh state"
                        )
                    mesh["mesh_id"] = mesh_id
                    mesh["join_token"] = join_token
                    mesh["coordinator_endpoint"] = coordinator_endpoint
                    # Remember the internal mesh identity on the durable
                    # registration record: a future relaunch reuses it so
                    # the coordinator resumes this mesh's snapshot chain.
                    ready_registration = (
                        self.state.get("mesh_registrations") or {}
                    ).get(str(mesh.get("model_id", "") or ""))
                    if isinstance(ready_registration, dict) and mesh_chain_bound:
                        ready_registration["mesh_id"] = mesh_id
                    if verification_snapshot_hash:
                        mesh["verification_snapshot_hash"] = (
                            verification_snapshot_hash
                        )
                        reported_generation = body.get("snapshot_generation")
                        if (
                            type(reported_generation) is int
                            and reported_generation >= 1
                        ):
                            # A resumed chain's generation, not the launch
                            # counter, is what the served snapshot binds.
                            mesh["snapshot_generation"] = reported_generation
                    worker["mesh"] = mesh["mesh_key"]
                    # The driver runs the coordinator AND computes its own stage,
                    # so it is a serving member too — not merely "driving".
                    if not non_driver:
                        # Single-worker: drive() already verified generation,
                        # so the end-to-end flag is true by construction —
                        # leaving it unset kept the operator console on
                        # "pending" forever.
                        mesh["status"] = "serving"
                        mesh["driver_verified"] = True
                        worker["status"] = "serving"
                        self._verify_advertised_endpoint_async(mesh)
                    else:
                        mesh["status"] = "joining"
                        worker["status"] = "driving"
                        mesh["driver_verified"] = False
                        # Admit members serially in committed RPC order.
                        mesh["pending_join_members"] = list(non_driver[1:])
                        self._queue_member_join(mesh, non_driver[0])
                # Multi-box drive_ready is a durable phase, not completion;
                # the driver's later serving report completes the command.
                terminal = not non_driver
            elif event == "serving":
                # Members report "serving" after they join; the driver reports it
                # only after verifying the fully-joined backend actually answers.
                # The mesh is serving when every member has joined AND the driver
                # has verified end-to-end, so its first chat is never empty.
                if not str(mesh.get("mesh_id", "") or "") or str(
                    mesh.get("status", "") or ""
                ) not in {"joining", "serving"}:
                    raise ValueError(
                        "serving report is invalid before drive_ready and member admission"
                    )
                if worker_id == mesh.get("driver"):
                    if action != "drive":
                        raise ValueError(
                            "driver serving report requires its drive command"
                        )
                    if (
                        self.serving_mode == POOL_SERVING_MODE_SUBNET
                        and mesh.get("model_index") is not None
                    ):
                        # Chain-bound only: an unregistered subnet mesh signs
                        # no snapshot (operator measurement lane).
                        verification_snapshot_hash = str(
                            body.get("verification_snapshot_hash", "") or ""
                        )
                        if not _COMMAND_DIGEST_RE.fullmatch(
                            verification_snapshot_hash
                        ):
                            raise ValueError(
                                "validator driver serving requires a verification "
                                "snapshot hash"
                            )
                        established_snapshot_hash = str(
                            mesh.get("verification_snapshot_hash", "") or ""
                        )
                        if (
                            established_snapshot_hash
                            and established_snapshot_hash
                            != verification_snapshot_hash
                        ):
                            raise ValueError(
                                "validator driver attempted to replace the "
                                "verification snapshot"
                            )
                        mesh["verification_snapshot_hash"] = (
                            verification_snapshot_hash
                        )
                    mesh["driver_verified"] = True
                    self._maybe_mark_mesh_serving(mesh)
                else:
                    if action != "join":
                        raise ValueError(
                            "member serving report requires its join command"
                        )
                    already_serving = worker_id in mesh.setdefault("serving", [])
                    advanced = self._record_serial_member_serving(mesh, worker_id)
                    if not (already_serving or advanced):
                        raise ValueError(
                            "mesh member serving report arrived out of admission order"
                        )
                worker["status"] = "serving"
                worker["mesh"] = mesh["mesh_key"]
            elif action == "fetch" and event == "fetched":
                # The driver finished downloading the model (auto-fetch) and
                # advertises its new catalog entry; now the mesh can drive.
                entry = dict(body.get("entry") or {})
                if str(entry.get("model_id", "") or "") != str(
                    mesh.get("model_id", "") or ""
                ):
                    raise ValueError("fetched model does not match the assigned mesh")
                if mesh.get("status") != "fetching":
                    if any(
                        str(item.get("model_id", "") or "")
                        == str(mesh.get("model_id", "") or "")
                        for item in worker.setdefault("catalog", [])
                    ):
                        response["duplicate"] = True
                    else:
                        raise ValueError(
                            "fetched report is invalid in the current mesh state"
                        )
                else:
                    catalog = worker.setdefault("catalog", [])
                    catalog[:] = [
                        item
                        for item in catalog
                        if str(item.get("model_id", "") or "")
                        != str(entry["model_id"])
                    ]
                    catalog.append(entry)
                    self._queue_drive(mesh["mesh_key"])
            elif action == "rotate" and event == "rotated":
                rotated_hash = str(
                    body.get("verification_snapshot_hash", "") or ""
                )
                rotated_generation = body.get("snapshot_generation")
                rotated_epoch = body.get("epoch")
                if (
                    not rotated_hash
                    or type(rotated_generation) is not int
                    or type(rotated_epoch) is not int
                ):
                    raise ValueError(
                        "rotated report requires verification_snapshot_hash, "
                        "snapshot_generation, and epoch"
                    )
                mesh["verification_snapshot_hash"] = rotated_hash
                mesh["snapshot_generation"] = int(rotated_generation)
                rotated_binding = dict(mesh.get("validator_binding") or {})
                rotated_binding["epoch"] = int(rotated_epoch)
                mesh["validator_binding"] = rotated_binding
                mesh.pop("rotation_error", None)
            elif event == "error":
                fail_message = str(
                    body.get("message", "") or "worker command failed"
                )[:500]

            if terminal:
                worker["command_inflight"] = None
                worker["last_command_completion"] = {
                    "command_id": command_id,
                    "command_digest": command_digest,
                    "action": action,
                    "mesh_key": mesh_key,
                    "event": event,
                    "report_fingerprint": report_fingerprint,
                    "completed_at_unix": int(now),
                }
                response["command_completed"] = command_id
            else:
                response["command_phase"] = event

            if fail_message and action == "rotate" and mesh is not None:
                # A failed rotation must never take down a serving mesh: it
                # keeps serving under the previous snapshot and the epoch
                # follower retries on its next tick.
                mesh["rotation_error"] = fail_message
            elif fail_message and action != "stop" and mesh is not None:
                self._fail_mesh_locked(mesh, fail_message)
            if action == "stop" and event in ("stopped", "error"):
                # Errored stops release the worker above, so the tombstone
                # can and must retire here too - otherwise it wedges.
                self._finish_mesh_stop_locked(mesh_key)

            self.worker_report_completed[report_key] = {
                "fingerprint": report_fingerprint,
                "response": {**response, "duplicate": True},
                "expires": now + WORKER_REPORT_DELIVERY_MAX_S,
            }
            self._save()
        return response

    # -- operator / dashboard ---------------------------------------------

    def _mesh_routing_health(
        self,
        mesh: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return current health for every worker required by one mesh.

        A coordinator process can remain alive after a remote RPC stage dies,
        so the persisted ``status == serving`` flag alone is not sufficient to
        route an operator test.  Treat missing workers and stale heartbeats as
        unavailable stages and expose that distinction to the dashboard.
        Caller holds ``self.lock`` when concurrent mutation is possible.
        """

        observed_at = time.time() if now is None else float(now)
        driver = str(mesh.get("driver", ""))
        required = list(
            dict.fromkeys(
                [
                    *(str(worker_id) for worker_id in (mesh.get("members") or [])),
                    *([driver] if driver else []),
                ]
            )
        )
        stale_members: list[str] = []
        for worker_id in required:
            worker = self.state["workers"].get(worker_id)
            if worker is None:
                stale_members.append(worker_id)
                continue
            last_seen = float(worker.get("last_seen_unix", 0.0) or 0.0)
            if observed_at - last_seen > WORKER_STALE_S:
                stale_members.append(worker_id)
        driver_stale = not driver or driver in stale_members
        member_stale = any(worker_id != driver for worker_id in stale_members)
        return {
            "driver_stale": driver_stale,
            "member_stale": member_stale,
            "stale_members": stale_members,
            "routing_ready": (
                str(mesh.get("status", "")) == "serving"
                and bool(driver)
                and bool(required)
                and not stale_members
            ),
        }

    def _require_mesh_routing_ready(self, mesh: Mapping[str, Any]) -> None:
        if mesh.get("status") != "serving":
            raise ValueError("mesh is not serving yet")
        health = self._mesh_routing_health(mesh)
        if health["routing_ready"]:
            return
        if health["driver_stale"]:
            raise ValueError(
                "mesh driver is offline; restart its worker before testing"
            )
        stale = ", ".join(health["stale_members"]) or "unknown"
        raise ValueError(
            f"mesh has an offline member stage ({stale}); restore every "
            "member before testing"
        )

    def ensure_worker_source_bundle(self, *, max_age_s: float = 600.0) -> Path:
        """The repo source tarball served to joining workers (cached).

        Regenerated when older than ``max_age_s`` so code syncs reach new
        workers without a manager restart. Excludes VCS state, venvs,
        caches, build trees, and model/pool data — joining workers build
        their own binaries (join_pool.sh), exactly like a git checkout.
        ``dist/`` MUST ship: it is the sanctioned prebuilt-wheel channel
        (zkllm, hot-capacity workspace, pcs prebuilts) that join_pool.sh
        installs from — a git checkout has it, so the bundle must too.
        Excluding it silently left every bundle-installed worker without
        the capacity-audit wheel (audits disabled) and aborted the join
        outright before the wheel lookup was made non-fatal.
        """

        from verallm.mesh.onboarding import find_repo_root

        repo = find_repo_root()
        bundle = Path(self.state_path).parent / "worker-src.tar.gz"
        if (
            bundle.is_file()
            and time.time() - bundle.stat().st_mtime < max_age_s
        ):
            return bundle
        import subprocess

        # Provenance stamp: workers run whatever tree state this tar
        # captures - INCLUDING uncommitted edits. A mid-development join
        # A partial source snapshot can make decode audits fail without
        # recording what the box actually ran. The stamp rides inside the
        # bundle as bundle-stamp.json and the dirty state is loud in the
        # manager log.
        stamp: dict[str, Any] = {"built_at_unix": int(time.time())}
        for key, argv in (
            ("git_head", ["git", "-C", str(repo), "rev-parse", "HEAD"]),
            (
                "git_dirty_files",
                ["git", "-C", str(repo), "status", "--porcelain"],
            ),
        ):
            try:
                probe = subprocess.run(
                    argv, capture_output=True, text=True, timeout=10.0
                )
                if probe.returncode == 0:
                    text_out = probe.stdout.strip()
                    stamp[key] = (
                        len(text_out.splitlines())
                        if key == "git_dirty_files"
                        else text_out
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass
        dirty = int(stamp.get("git_dirty_files", 0) or 0)
        if dirty:
            logger.warning(
                "WORKER-BUNDLE ships a DIRTY tree: %d uncommitted files on "
                "top of %s - joining workers will run unreviewed edits",
                dirty,
                str(stamp.get("git_head", "unknown"))[:12],
            )
        stamp_path = Path(self.state_path).parent / "bundle-stamp.json"
        stamp_path.write_text(json.dumps(stamp, indent=1), encoding="utf-8")

        tmp = bundle.with_suffix(".tmp")
        completed = subprocess.run(
            [
                "tar", "-czf", str(tmp),
                "--exclude=.git",
                "--exclude=.venv*",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=node_modules",
                "--exclude=build",
                "--exclude=models",
                "--exclude=.verathos",
                "--exclude=*.gguf",
                "--exclude=*.log",
                "-C", str(repo.parent),
                repo.name,
                "-C", str(stamp_path.parent),
                "--transform",
                f"s|^bundle-stamp.json|{repo.name}/bundle-stamp.json|",
                "bundle-stamp.json",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"tar failed: {completed.stderr.strip()[:300]}"
            )
        os.replace(tmp, bundle)
        return bundle

    def handle_deploy_status(self, body: dict[str, Any]) -> dict[str, Any]:
        """Deploy progress marker so boards stop offering a racing deploy.

        The deploy CLI runs as its own process; without this record the
        registration menu kept advertising "register on the subnet" for a
        model whose probe gate was ALREADY running - one
        keypress away from two deploys racing the same chain slot. Stage
        "done"/"failed" clears the marker; stale markers (a killed deploy)
        expire via DEPLOY_STATUS_TTL_S at read time.
        """

        self._auth_manage(body)
        model_id = str(body.get("model_id", "") or "")
        stage = str(body.get("stage", "") or "")[:80]
        if not model_id:
            raise ValueError("model_id is required")
        with self.lock:
            deploys = self.state.setdefault("deploys", {})
            if stage in ("done", "failed"):
                deploys.pop(model_id, None)
            else:
                deploys[model_id] = {
                    "stage": stage,
                    "updated_at_unix": int(time.time()),
                }
            self._save()
        return {"status": "ok"}

    def _live_deploys(self) -> dict[str, dict[str, Any]]:
        now = int(time.time())
        result: dict[str, dict[str, Any]] = {}
        for model_id, entry in (self.state.get("deploys") or {}).items():
            if now - int(entry.get("updated_at_unix", 0) or 0) <= (
                DEPLOY_STATUS_TTL_S
            ):
                result[str(model_id)] = dict(entry)
        return result

    def handle_status(self, body: dict[str, Any]) -> dict[str, Any]:
        # This raw view includes private worker endpoints and model catalog
        # details. Workers do not need it; the public dashboard uses the
        # sanitized operator overview instead.
        self._auth_manage(body)
        now = int(time.time())
        with self.lock:
            workers = {
                wid: {
                    k: v
                    for k, v in w.items()
                    if k
                    not in (
                        "commands",
                        "command_inflight",
                        "chat_pending",
                        "worker_session_id",
                        "last_command_ack",
                        "last_command_completion",
                        "last_superseded_command",
                    )
                }
                | {"stale": now - int(w.get("last_seen_unix", 0)) > WORKER_STALE_S}
                for wid, w in self.state["workers"].items()
            }
            meshes = {}
            # Heal records nothing can ever advance BEFORE rendering them:
            # the board must show reality, and pool-state.json is read by
            # tooling that never dials the coordinator (regression).
            dirty = self._reconcile_orphaned_meshes_locked()
            for k, m in self.state["meshes"].items():
                if m.get("status") == "stopped":
                    continue
                dw = self.state["workers"].get(str(m.get("driver", "")), {})
                silent_s = now - int(dw.get("last_seen_unix", 0))
                # A mesh stuck in a FORMATION state (driving/joining/fetching)
                # whose driver has been silent for minutes can never complete:
                # the drive command lives in the dead worker, and nothing else
                # will ever move the mesh out of "driving". Reap it so the
                # operator sees an actionable error instead of a permanent
                # "driving" (a dead driver during formation wedged a mesh for
                # 10+ minutes exactly this way). Generous threshold so a
                # worker riding out a network blip is left alone; "serving"
                # meshes are never reaped (driver_stale below flags those).
                _mesh_model_bytes = int(
                    (
                        (self.state.get("model_registry") or {}).get(
                            str(m.get("model_id", "") or ""), {}
                        )
                        or {}
                    ).get("model_bytes", 0)
                    or 0
                )
                _silent_budget_s = _formation_silence_budget_s(
                    _mesh_model_bytes
                )
                if (
                    m.get("status") in ("driving", "joining", "fetching")
                    and silent_s > _silent_budget_s
                ):
                    self._fail_mesh_locked(
                        m,
                        f"driver worker went silent for {silent_s}s during "
                        "formation (budget "
                        f"{int(_silent_budget_s)}s); restart its worker and "
                        "relaunch",
                    )
                    for member_id in m.get("members", []):
                        if member_id in workers:
                            workers[member_id]["status"] = str(
                                self.state["workers"][member_id].get(
                                    "status",
                                    "error",
                                )
                            )
                    dirty = True
                m2 = dict(m)
                # Join credentials remain manager-internal. Management status
                # needs topology and errors, never the reusable raw token.
                m2.pop("join_token", None)
                m2.pop("pending_join_members", None)
                # A "serving" row whose driver stopped heartbeating is dead in
                # practice. The same is true for every non-driver RPC stage:
                # llama-server may stay up while its pipeline is incomplete.
                m2.update(self._mesh_routing_health(m, now=now))
                meshes[k] = m2
            if dirty:
                self._save()
            return {
                "status": "ok",
                "pool_id": self.state["pool_id"],
                "serving_mode": self.serving_mode,
                "deploys": self._live_deploys(),
                "coordinator_address": str(
                    self.state.get("coordinator_address", "") or ""
                ),
                **(
                    {
                        "validator_binding": dict(self.validator_binding),
                        "validator_epoch_floor": int(
                            self.state.get(
                                "validator_epoch_floor",
                                self.validator_binding["epoch"],
                            )
                        ),
                    }
                    if self.serving_mode == POOL_SERVING_MODE_SUBNET
                    else {}
                ),
                # Chain identity persisted at manager start (empty on pools
                # whose PM2 unit predates it) so status consumers can say
                # WHICH hotkey/network this pool serves as, without argv
                # archaeology.
                "subtensor_network": self._subtensor_network(),
                "wallet_name": str(self.state.get("wallet_name", "") or ""),
                "wallet_hotkey": str(
                    self.state.get("wallet_hotkey", "") or ""
                ),
                "coordinator_hotkey_ss58": str(
                    self.state.get("coordinator_hotkey_ss58", "") or ""
                ),
                # The on-chain registrations the deploy pipeline stored
                # (stage 15), keyed per model; the lease renewer keeps each
                # expires_at fresh. Empty until a deploy has completed.
                **_registration_view(self.state),
                # Public https listener for the private API (0 = loopback
                # only); the board prints the reachable URL from this.
                "api_tls_port": int(self.state.get("api_tls_port", 0) or 0),
                "workers": workers,
                "meshes": meshes,
                # Registered download sources: models the pool can serve even
                # if no worker holds the file yet (drivers auto-fetch).
                "models": dict(self.state.get("model_registry") or {}),
            }

    @staticmethod
    def _link_class(rtt_ms: float) -> tuple[int, str]:
        """Classify a member link by RTT for placement ranking + UX.

        Every decode token crosses the driver<->member link, so RTT is a
        hard per-token latency floor: a 107ms cross-region pair is capped
        near ~4-5 tok/s no matter how fast the GPUs are (measured on the
        H100(FI)+A100(US) attempt). Recommend against, never hard-block.
        """
        if rtt_ms < 2:
            return 0, "local"
        if rtt_ms < 10:
            return 1, "lan"
        if rtt_ms < 40:
            return 2, "regional"
        if rtt_ms < 90:
            return 3, "far"
        return 4, "cross-region"

    def recommend(self, model_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Ranked launch suggestions for one model, plus a per-worker reason.

        The reasons make an empty result explainable in the dashboard
        ("busy", "offline", "model not on this machine") instead of a bare
        "no placement" that reads as a bug.
        """

        reasons: dict[str, str] = {}
        with self.lock:
            now = int(time.time())
            registry = (self.state.get("model_registry") or {}).get(model_id, {})
            if not registry.get("hf_repo"):
                # Same shipped-catalogue fallback the launch path uses. Without
                # it, placement for a model no worker happens to hold has no
                # facts at all and returns "no feasible worker set", so the
                # launch never even reaches the auto-fetch it would have run.
                registry = _shipped_model_fetch_spec(
                    model_id,
                    chain_id=self.validator_binding.get("chain_id"),
                ) or registry
            fetch_gb = int(registry.get("model_bytes", 0) or 0) / 1e9
            fetchable = bool(registry.get("hf_repo"))
            # EVERY idle worker is a potential member: only the DRIVER needs the
            # model (it streams layer slices to file-less members over RPC), so
            # a machine without the model still contributes its VRAM to the
            # split. A driver is a non-member-only worker that has the model OR
            # can fetch it (registry-known source).
            candidates = []
            model_bytes = 0
            for wid, w in self.state["workers"].items():
                if now - int(w.get("last_seen_unix", 0)) > WORKER_STALE_S:
                    reasons[wid] = "offline"
                    continue
                if w.get("status") != "idle":
                    reasons[wid] = f"busy ({w.get('status')})"
                    continue
                entry = next(
                    (c for c in w.get("catalog", []) if c.get("model_id") == model_id),
                    None,
                )
                if entry is not None:
                    model_bytes = max(model_bytes, int(entry.get("model_bytes", 0) or 0))
                member_only = bool(w.get("capability", {}).get("member_only"))
                subnet_driver_ready = _capability_subnet_driver_ready(
                    w.get("capability", {})
                )
                has_model = entry is not None
                # A fetch-driver needs disk for the download itself (plus a
                # little margin) — huggingface_hub only WARNS on low disk and
                # then dies mid-file, so refuse the placement up front.
                free_disk = w.get("capability", {}).get("free_disk_gb")
                disk_ok = (
                    free_disk is None  # older worker without the advert
                    or float(free_disk) >= fetch_gb * 1.05 + 1.0
                )
                can_drive = (
                    not member_only
                    and (
                        self.serving_mode != POOL_SERVING_MODE_SUBNET
                        or subnet_driver_ready
                    )
                    and (has_model or (fetchable and disk_ok))
                )
                if member_only:
                    reasons[wid] = "available (member-only slice)"
                elif (
                    self.serving_mode == POOL_SERVING_MODE_SUBNET
                    and not subnet_driver_ready
                ):
                    reasons[wid] = (
                        "member only: no fresh, non-empty validator "
                        "allowlist on this worker (it needs "
                        "--subtensor-network/--netuid, not a wallet)"
                    )
                elif has_model:
                    reasons[wid] = "available (can drive)"
                elif fetchable and not disk_ok:
                    reasons[wid] = (
                        f"member only: downloading needs ~{fetch_gb:.0f} GB free "
                        f"disk, has {float(free_disk):.0f} GB"
                    )
                elif fetchable:
                    reasons[wid] = "available (downloads model to drive)"
                else:
                    reasons[wid] = "available (file-less member slice)"
                candidates.append(
                    {
                        "worker_id": wid,
                        "vram_gb": float(w.get("capability", {}).get("vram_gb", 0) or 0),
                        "peer_rtt_ms": dict(w.get("peer_rtt_ms", {})),
                        "rtt_manager": float((w.get("rtt_ms", {}) or {}).get("manager", 0) or 0),
                        "can_drive": can_drive,
                        "needs_fetch": can_drive and not has_model,
                    }
                )
        if not candidates:
            return [], reasons
        need_bytes = model_bytes or int(fetch_gb * 1e9)
        if need_bytes <= 0:
            return [], reasons  # unknown model size — can't size a placement
        need_gb = need_bytes / 1e9 * VRAM_HEADROOM

        def pair_rtt(a: dict[str, Any], b: dict[str, Any]) -> float:
            direct = a["peer_rtt_ms"].get(b["worker_id"])
            if direct is not None:
                return float(direct)
            if a["rtt_manager"] and b["rtt_manager"]:
                return a["rtt_manager"] + b["rtt_manager"]
            return DEFAULT_RTT_MS

        def tag(s: dict[str, Any], driver: dict[str, Any]) -> dict[str, Any]:
            # Only annotate a fetch (keeps the has-model output shape stable).
            if driver["needs_fetch"]:
                s["fetch"] = True
                s["download_gb"] = round(fetch_gb, 1)
            return s

        suggestions: list[dict[str, Any]] = []
        drivers = [c for c in candidates if c["can_drive"]]
        for c in drivers:
            if c["vram_gb"] >= need_gb:
                suggestions.append(tag(
                    {
                        "workers": [c["worker_id"]],
                        "driver": c["worker_id"],
                        "max_rtt_ms": 0.0,
                        "link_class": "local",
                        "driver_vram_gb": c["vram_gb"],
                        "total_vram_gb": c["vram_gb"],
                    },
                    c,
                ))
            else:
                # Say WHY there is no single-box row for this machine
                # instead of silently omitting it (a "missing" a100 row
                # reads as a bug when the real cause is VRAM headroom).
                reasons[c["worker_id"]] = (
                    f"pairs only: solo needs ~{need_gb:.0f} GB VRAM "
                    f"(advertises {c['vram_gb']:.0f} GB)"
                )
        # Pairs: a driver-capable worker + ANY other idle worker (the second may
        # be file-less). Dedup by worker set, preferring a driver that already
        # has the model over one that must download.
        best_pair: dict[frozenset, tuple[tuple, dict[str, Any]]] = {}
        for d in drivers:
            for m in candidates:
                if m["worker_id"] == d["worker_id"]:
                    continue
                if d["vram_gb"] + m["vram_gb"] < need_gb:
                    continue
                key = frozenset((d["worker_id"], m["worker_id"]))
                rtt = max(pair_rtt(d, m), pair_rtt(m, d))
                link_rank, link_class = self._link_class(rtt)
                cand = tag(
                    {
                        "workers": [d["worker_id"], m["worker_id"]],
                        "driver": d["worker_id"],
                        "max_rtt_ms": round(rtt, 2),
                        "link_class": link_class,
                        "driver_vram_gb": d["vram_gb"],
                        "total_vram_gb": d["vram_gb"] + m["vram_gb"],
                    },
                    d,
                )
                if link_rank >= 3:
                    cand["warn"] = (
                        f"{rtt:.0f} ms link: every decode token crosses it "
                        f"(~{max(1, int(1000 / (2 * rtt)))} tok/s ceiling); "
                        "co-located workers run at full speed"
                    )
                # Best driver for this worker set: no-fetch first, then bigger.
                rank = (d["needs_fetch"], -d["vram_gb"])
                prev = best_pair.get(key)
                if prev is None or rank < prev[0]:
                    best_pair[key] = (rank, cand)
        suggestions.extend(v[1] for v in best_pair.values())
        suggestions.sort(
            key=lambda s: (
                len(s["workers"]),
                self._link_class(s["max_rtt_ms"])[0],
                s["max_rtt_ms"],
                # BEST fit, not biggest: among placements that fit (with the
                # VRAM_HEADROOM KV allowance already applied), prefer the
                # SMALLEST unit. Biggest-first parked a 21 GB model on the
                # pool's only 320 GB box — the one box that can hold the
                # largest catalogue models while a smaller unit sat idle.
                # Fit quality outranks fetch convenience:
                # a one-time download beats squatting the big box for the
                # model's whole serving life.
                float(s.get("total_vram_gb", 0)),
                s.get("fetch", False),
            )
        )
        return suggestions, reasons

    def handle_recommend(self, body: dict[str, Any]) -> dict[str, Any]:
        # Placement advice contains only the same sanitized fleet facts shown
        # in the public overview. Launching the suggestion remains protected.
        suggestions, reasons = self.recommend(str(body.get("model_id", "")))
        return {"status": "ok", "suggestions": suggestions, "reasons": reasons}

    def handle_launch(self, body: dict[str, Any]) -> dict[str, Any]:
        self._auth_manage(body)
        auto_relaunch = bool(body.get("_auto_relaunch"))
        deploy_measurement_unbound = bool(
            body.get("_deploy_measurement_unbound")
        )
        if auto_relaunch and deploy_measurement_unbound:
            raise ValueError(
                "automatic relaunch cannot request an unbound deploy measurement"
            )
        if not self.serving_mode:
            raise ValueError(
                "pool serving mode is unconfigured; recreate it explicitly as dev or subnet"
            )
        model_id = str(body.get("model_id", ""))
        if not model_id:
            raise ValueError("model_id is required")
        model_registration = dict(
            (self.state.get("model_registry") or {}).get(model_id) or {}
        )
        # A subnet pool launches a model in one of two states. CHAIN-BOUND
        # (the model has an on-chain slot) is the scoreable one: it signs a
        # verification snapshot and serves validators. UNREGISTERED is the
        # verification state an operator needs FIRST, because the registered
        # max_context_len must be the auto-fit's MEASURED value and that is
        # only known after a launch: it serves the operator lane (probe,
        # chat, self-test) so the operator can prove the model works and read
        # its measured context, and it signs no snapshot and claims no chain
        # endpoint, so it can never take scored traffic.
        chain_bound = False
        if self.serving_mode == POOL_SERVING_MODE_SUBNET:
            if not self.validator_binding:
                raise ValueError("subnet pool has no chain binding")
            if model_registration.get("model_index") is not None:
                confirmed = bool(model_registration.get("chain_committed"))
                if not confirmed:
                    # The lease record is only ever written after a
                    # successful chain registration, so its presence at
                    # the same index IS confirmation - both for legacy
                    # records that predate the flag and for a registered
                    # model whose re-deploy aborted after re-stamping the
                    # binding as pending.
                    lease = (
                        self.state.get("mesh_registrations") or {}
                    ).get(model_id) or {}
                    confirmed = (
                        lease.get("index") is not None
                        and int(lease["index"])
                        == int(model_registration["model_index"])
                    )
                    if confirmed:
                        with self.lock:
                            stored_entry = (
                                self.state.get("model_registry") or {}
                            ).get(model_id)
                            if isinstance(stored_entry, dict):
                                stored_entry["chain_committed"] = True
                                self._save()
                if not confirmed and not bool(
                    body.get("pending_binding_ok")
                ):
                    # A predicted binding whose deploy never reached the
                    # chain (aborted mid-flight) is not a contract: the
                    # model was NEVER registered, so this launch must
                    # measure fresh on its assigned workers instead of
                    # inheriting a stale prediction as its floor. Purge it so the
                    # state heals in code, not by hand.
                    logger.warning(
                        "discarding unconfirmed chain binding for %s "
                        "(model_index=%s, max_context_len=%s): its deploy "
                        "never completed the chain registration",
                        model_id,
                        model_registration.get("model_index"),
                        model_registration.get("max_context_len"),
                    )
                    with self.lock:
                        stored_entry = (
                            self.state.get("model_registry") or {}
                        ).get(model_id)
                        if isinstance(stored_entry, dict):
                            for field_name in (
                                "model_index",
                                "chain_committed",
                                "max_context_len",
                                "measured_ctx_budget",
                            ):
                                stored_entry.pop(field_name, None)
                            self._save()
                    for field_name in (
                        "model_index",
                        "chain_committed",
                        "max_context_len",
                        "measured_ctx_budget",
                    ):
                        model_registration.pop(field_name, None)
            chain_bound = (
                model_registration.get("model_index") is not None
                and not deploy_measurement_unbound
            )
            if chain_bound:
                for field_name in (
                    "model_package_hash",
                    "model_tensor_manifest_root",
                    "tokenizer_hash",
                ):
                    _require_digest(
                        model_registration.get(field_name),
                        field_name=field_name,
                    )
                if int(model_registration.get("layers", 0) or 0) <= 0:
                    raise ValueError(
                        "subnet pool model has no positive layer count"
                    )
                _require_max_context_len(
                    model_registration.get("max_context_len")
                )
                if not str(
                    model_registration.get("quantization_scheme", "")
                ).strip():
                    raise ValueError(
                        "subnet pool model has no quantization_scheme"
                    )
        members = [str(w) for w in (body.get("workers") or [])]
        driver = str(body.get("driver", "") or "")
        if not members:
            picks, _ = self.recommend(model_id)
            if not picks:
                raise ValueError("no feasible worker set for this model")
            preferred_driver = str(
                body.get("_preferred_driver", "") or ""
            )
            if preferred_driver:
                picks = sorted(
                    picks,
                    key=lambda pick: (
                        str(pick.get("driver", "") or "")
                        != preferred_driver
                    ),
                )
            members, driver = picks[0]["workers"], picks[0]["driver"]
        if len(set(members)) != len(members):
            raise ValueError("mesh workers must be unique")
        if len(members) > 4096:
            raise ValueError("mesh cannot contain more than 4096 workers")
        if not driver:
            driver = members[0]
        if driver not in members:
            raise ValueError("driver must be one of the mesh workers")
        with self.lock:
            if auto_relaunch:
                registration = self._registrations_locked().get(model_id)
                if not isinstance(registration, dict) or registration.get(
                    "index"
                ) is None:
                    raise ValueError(
                        "auto-relaunch requires a chain-bound registration"
                    )
                if registration.get("suspended_by_operator"):
                    raise ValueError("auto-relaunch is suspended by operator")

                terminal_meshes = []
                for key, existing in self.state["meshes"].items():
                    if str(existing.get("model_id", "") or "") != model_id:
                        continue
                    assigned = any(
                        str((self.state["workers"].get(member_id) or {}).get(
                            "mesh", ""
                        ) or "") == key
                        for member_id in (existing.get("members") or [])
                    )
                    status = str(existing.get("status", "") or "")
                    if status not in ("error", "stopped") or assigned:
                        raise ValueError(
                            "auto-relaunch refused while another mesh record "
                            "is live or still owns workers"
                        )
                    terminal_meshes.append(key)
                for key in terminal_meshes:
                    self.state["meshes"].pop(key, None)

            if self.serving_mode == POOL_SERVING_MODE_SUBNET and chain_bound:
                model_index = int(model_registration["model_index"])
                conflicting = [
                    str(existing.get("mesh_key", key))
                    for key, existing in self.state["meshes"].items()
                    if int(existing.get("model_index", -1)) == model_index
                ]
                if conflicting:
                    raise ValueError(
                        "subnet pool already has a mesh for model index "
                        f"{model_index}; stop {conflicting[0]} and wait for "
                        "every worker to confirm shutdown before relaunching"
                    )
            now = int(time.time())
            for wid in members:
                worker = self.state["workers"].get(wid)
                if worker is None:
                    raise ValueError(f"unknown worker: {wid}")
                if worker.get("status") != "idle":
                    raise ValueError(f"worker busy: {wid}")
                # A stale worker is either offline (mesh would hang forming) or
                # mid-restart: its imminent (re)join would DROP this brand-new
                # mesh as leftover state, silently un-launching it.
                if now - int(worker.get("last_seen_unix", 0)) > WORKER_STALE_S:
                    raise ValueError(
                        f"worker offline (no heartbeat for >{int(WORKER_STALE_S)}s): "
                        f"{wid}; wait for it to reconnect, then launch again"
                    )
            ordered_member_ids = [driver] + [
                worker_id for worker_id in members if worker_id != driver
            ]
            # One weight per llama DEVICE: a multi-GPU worker's rpc-server
            # exposes one device per GPU, so the split viability runs over
            # the flattened per-GPU weights in committed member order.
            ordered_device_vram: list[int] = []
            for worker_id in ordered_member_ids:
                capability = self.state["workers"][worker_id].get(
                    "capability", {}
                )
                per_gpu = [
                    int(v) for v in capability.get("per_gpu_vram_gb") or []
                ]
                if not per_gpu:
                    per_gpu = [int(capability.get("vram_gb", 0) or 0)]
                ordered_device_vram.extend(per_gpu)
            if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                if any(weight <= 0 for weight in ordered_device_vram):
                    raise ValueError(
                        "subnet mesh workers must advertise positive vram_gb"
                    )
                # An unregistered launch has no pool registry entry; the
                # layer count then comes from a worker's catalog advert or
                # the shipped catalogue (same sources the recommender uses).
                viability_layers = int(model_registration.get("layers", 0) or 0)
                if viability_layers <= 0:
                    for worker_id in ordered_member_ids:
                        for advert in (
                            self.state["workers"][worker_id].get("catalog")
                            or []
                        ):
                            if advert.get("model_id") == model_id:
                                viability_layers = int(
                                    advert.get("layers", 0) or 0
                                )
                                break
                        if viability_layers > 0:
                            break
                if viability_layers <= 0:
                    shipped = _shipped_model_fetch_spec(
                        model_id,
                        chain_id=self.validator_binding.get("chain_id"),
                    ) or {}
                    viability_layers = int(shipped.get("layers", 0) or 0)
                if viability_layers > 0:
                    try:
                        llama_tensor_split_layer_ranges(
                            viability_layers,
                            ordered_device_vram,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"subnet mesh tensor split is not viable: {exc}"
                        ) from exc
                elif chain_bound:
                    raise ValueError(
                        "subnet pool model has no positive layer count"
                    )
            # Only the DRIVER needs the model on disk: its llama-server reads
            # the GGUF and streams layer slices to members over RPC, and members
            # prove from on-demand content-addressed blobs (file-less join). A
            # driver without the model auto-fetches it if the pool knows a
            # source (model_registry, merged from workers' catalog adverts).
            driver_w = self.state["workers"].get(driver, {})
            if (
                self.serving_mode == POOL_SERVING_MODE_SUBNET
                and not _capability_subnet_driver_ready(
                    driver_w.get("capability", {})
                )
            ):
                raise ValueError(
                    f"driver '{driver}' has no fresh, non-empty validator "
                    "allowlist (start the worker with "
                    "--subtensor-network/--netuid; no wallet needed)"
                )
            driver_has = any(
                c.get("model_id") == model_id for c in driver_w.get("catalog", [])
            )
            fetch_spec: dict[str, Any] | None = None
            if not driver_has:
                fetch_spec = (self.state.get("model_registry") or {}).get(model_id)
                if not (fetch_spec and fetch_spec.get("hf_repo")):
                    # Last resort: the SHIPPED mesh catalogue already knows the
                    # source of every model we ship. Without this, a pool whose
                    # workers have never advertised the model refuses to launch
                    # it even though the download source is compiled in, which
                    # made a fresh box unable to serve a catalogued model until
                    # someone hand-fed it a catalog.
                    fetch_spec = _shipped_model_fetch_spec(
                        model_id,
                        chain_id=self.validator_binding.get("chain_id"),
                    ) or fetch_spec
                if not (fetch_spec and fetch_spec.get("hf_repo")):
                    have = {c.get("model_id") for c in driver_w.get("catalog", [])}
                    listed = ", ".join(sorted(m for m in have if m)) or "(none)"
                    raise ValueError(
                        f"driver '{driver}' does not have model '{model_id}' "
                        f"(it has: {listed}) and no download source is known for "
                        "it; pick a driver that has the model. (Members don't "
                        "need the model; only the driver loads it.)"
                    )
                # A fetch source that carries no manifest store (the shipped
                # catalogue, or a worker advert) still gets this pool's
                # configured store, so a fresh driver downloads the manifest
                # instead of spending the better part of an hour rebuilding
                # it. The root is verified against the registered anchor
                # either way, so a wrong manifest can only fail, never pass.
                if not fetch_spec.get("manifest_urls") and self.manifest_base_urls:
                    fetch_spec = {
                        **dict(fetch_spec),
                        "manifest_urls": list(self.manifest_base_urls),
                    }
                # Deploy passes the CHAIN manifest root with the measurement
                # launch. Pool registration deliberately happens only after
                # measurement (it commits the measured context), so for a
                # first launch on a fresh pool the registry entry cannot
                # carry the root yet, and without it the driver skipped the
                # manifest store and silently rebuilt the manifest locally:
                # subnet-owner work on a miner box. Management-authed input,
                # and a wrong root can only make the store fetch fail back
                # to the local build, never pass a bad manifest (binding is
                # hash-verified end to end).
                launch_root = str(
                    body.get("model_tensor_manifest_root", "") or ""
                )
                if launch_root and not fetch_spec.get(
                    "model_tensor_manifest_root"
                ):
                    fetch_spec = {
                        **dict(fetch_spec),
                        "model_tensor_manifest_root": _require_digest(
                            launch_root,
                            field_name="model_tensor_manifest_root",
                        ),
                    }
                from verallm.mesh.manifest_store import (
                    normalize_mesh_manifest_base_urls,
                )

                launch_urls = normalize_mesh_manifest_base_urls(
                    body.get("manifest_urls") or ()
                )
                if launch_urls and not fetch_spec.get("manifest_urls"):
                    fetch_spec = {
                        **dict(fetch_spec),
                        "manifest_urls": list(launch_urls),
                    }
                # Refuse a download that cannot fit — otherwise it dies at
                # whatever percent the disk fills (hf_hub only warns).
                dl_gb = int(fetch_spec.get("model_bytes", 0) or 0) / 1e9
                free_disk = driver_w.get("capability", {}).get("free_disk_gb")
                if dl_gb and free_disk is not None and float(free_disk) < dl_gb * 1.05 + 1.0:
                    raise ValueError(
                        f"driver '{driver}' has {float(free_disk):.1f} GB free disk "
                        f"but downloading '{model_id}' needs ~{dl_gb:.1f} GB; free "
                        "up space on it or pick a driver that has the model"
                    )
                if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                    # Manifests are owner-built and published to the store;
                    # a driver REBUILD is subnet-owner work on a miner box
                    # and always means a broken fetch path. Refuse at launch
                    # (never even start fetching) instead of silently
                    # rebuilding for hours at the end of a large download.
                    if not str(
                        fetch_spec.get("model_tensor_manifest_root", "") or ""
                    ):
                        raise ValueError(
                            f"refusing subnet launch of '{model_id}': no "
                            "owner-built tensor-manifest root is known (pool "
                            "registry, shipped catalogue, or launch body), so "
                            "the driver would have to rebuild the manifest "
                            "locally. Publish the manifest to the gleipnir "
                            "store and register the root first."
                        )
                    fetch_spec = {
                        **dict(fetch_spec),
                        "require_published_manifest": True,
                    }
            if driver_w.get("capability", {}).get("member_only"):
                raise ValueError(
                    f"worker {driver} is member-only (serves stages, cannot drive); "
                    "pick a different driver"
                )
            mesh_key = "m-" + uuid.uuid4().hex[:10]
            resume_mesh_id = ""
            if chain_bound:
                # Mesh identity is logical, not instance-scoped: a relaunch
                # of the same chain-bound registration keeps its mesh key
                # AND its internal mesh id, so the coordinator lands in the
                # same mesh state dir, finds its persisted verification
                # snapshot, and resumes that chain instead of minting a new
                # one. A fresh identity would present the identical model as
                # different verification terms mid-epoch, which a
                # validator's pinned canary prices as a proof failure.
                previous_registration = (
                    self.state.get("mesh_registrations") or {}
                ).get(model_id)
                previous_key = (
                    str(previous_registration.get("mesh_key", "") or "")
                    if isinstance(previous_registration, dict)
                    else ""
                )
                if previous_key and previous_key not in (
                    self.state.get("meshes") or {}
                ):
                    mesh_key = previous_key
                    resume_mesh_id = str(
                        previous_registration.get("mesh_id", "") or ""
                    )
            # Per-model proof tolerance overrides ride from the driver's
            # catalog entry into every stage's serve flags (see
            # _tolerance_flags). A driver that must FETCH has no catalog entry
            # yet — fall back to the pool registry (merged from the advert of
            # whichever worker taught the pool this model).
            driver_entry = next(
                (c for c in driver_w.get("catalog", []) if c.get("model_id") == model_id),
                {},
            )
            tolerances = {
                **_operator_tuning_fields(
                    (self.state.get("model_registry") or {}).get(model_id, {})
                ),
                **_operator_tuning_fields(driver_entry),
            }
            model_binding = (
                {"model_index": int(model_registration["model_index"])}
                if (
                    model_registration.get("model_index") is not None
                    and not deploy_measurement_unbound
                )
                else {}
            )
            if (
                not deploy_measurement_unbound
                and model_registration.get("max_context_len") is not None
            ):
                model_binding["max_context_len"] = _require_max_context_len(
                    model_registration["max_context_len"]
                )
            elif body.get("max_context_len") is not None:
                # Deploy passes the operator's intended registration budget
                # so the MEASUREMENT launch serves at that context instead
                # of probing the VRAM maximum. Health-check-passing at the
                # VRAM max is not serving-survives: the first full batch
                # allocates compute buffers on TOP of the KV, which is
                # exactly where a 389k-token "fit" OOM-crashed a mesh whose
                # actual contract was 98k. Measurement
                # still reports the measured value; registration caps an
                # override at what was measured, unchanged.
                model_binding["max_context_len"] = _require_max_context_len(
                    body["max_context_len"]
                )
            if self.serving_mode == POOL_SERVING_MODE_SUBNET and chain_bound:
                # Only a CHAIN-BOUND mesh carries the on-chain anchors and a
                # snapshot generation: an unregistered measurement launch has
                # neither (it signs no snapshot and serves only the operator
                # lane), and stamping placeholders here would let a mesh
                # masquerade as verifiable.
                self.state["snapshot_generation"] = int(
                    self.state.get("snapshot_generation", 0) or 0
                ) + 1
                self.state["validator_epoch_floor"] = max(
                    int(self.state.get("validator_epoch_floor", -1)),
                    int(self.validator_binding["epoch"]),
                )
                model_binding.update(
                    {
                        "model_package_hash": str(
                            model_registration["model_package_hash"]
                        ),
                        "model_tensor_manifest_root": str(
                            model_registration["model_tensor_manifest_root"]
                        ),
                        "tokenizer_hash": str(model_registration["tokenizer_hash"]),
                        "quantization_scheme": str(
                            model_registration["quantization_scheme"]
                        ),
                        "total_layers": int(model_registration["layers"]),
                        "snapshot_generation": int(
                            self.state["snapshot_generation"]
                        ),
                    }
                )
            self.state["meshes"][mesh_key] = {
                "mesh_key": mesh_key,
                "model_id": model_id,
                "members": members,
                "driver": driver,
                "status": "fetching" if fetch_spec else "driving",
                "serving": [],
                "serving_mode": self.serving_mode,
                "created_at_unix": int(time.time()),
                **({"resume_mesh_id": resume_mesh_id} if resume_mesh_id else {}),
                **(
                    {
                        "validator_binding": dict(self.validator_binding),
                        "coordinator_address": self.coordinator_address,
                        # A driver that holds no wallet cannot look up the
                        # hotkey it serves under, and the spec it stamps must
                        # carry the same SS58 the receipts verify against.
                        "coordinator_hotkey": str(
                            self.state.get("coordinator_hotkey_ss58", "") or ""
                        ),
                    }
                    if self.serving_mode == POOL_SERVING_MODE_SUBNET
                    else {}
                ),
                **tolerances,
                **model_binding,
            }
            if chain_bound:
                # The stored registration's mesh_key must follow the model
                # to its NEW mesh: the lease renewer reads it, and a stale
                # key makes it read a healthy relaunched model as "gone"
                # and let the live chain entry expire.
                launched_registration = (
                    self.state.get("mesh_registrations") or {}
                ).get(model_id)
                if isinstance(launched_registration, dict):
                    launched_registration["mesh_key"] = mesh_key
            if not auto_relaunch and not deploy_measurement_unbound:
                # An explicit launch lifts any operator suspension: the
                # operator has re-stated that this model should serve, so
                # auto-relaunch may guard it again. An automatic launch must
                # never clear intent that raced its delayed attempt.
                _launch_registration = self._registrations_locked().get(model_id)
                if isinstance(_launch_registration, dict):
                    _launch_registration.pop("suspended_by_operator", None)
            for wid in members:
                self.state["workers"][wid]["status"] = "assigned"
                self.state["workers"][wid]["mesh"] = mesh_key
            if fetch_spec:
                # Driver downloads the model first; its "fetched" report (see
                # handle_report) dispatches the drive. Progress rides the
                # worker's heartbeat status ("fetching 42%") into the dashboard.
                self._queue_worker_command(
                    self.state["workers"][driver],
                    {
                        "action": "fetch",
                        "mesh_key": mesh_key,
                        "model_id": model_id,
                        "spec": dict(fetch_spec),
                    },
                )
            else:
                self._queue_drive(mesh_key)
            self._save()
        return {"status": "launching", "mesh_key": mesh_key, "driver": driver, "workers": members}

    def _queue_drive(self, mesh_key: str) -> None:
        """Queue the drive command for a mesh's driver. Caller holds the lock."""
        mesh = self.state["meshes"][mesh_key]
        mesh["status"] = "driving"
        # Per-member llama device counts in committed stage order (driver
        # first). The driver sizes --llama-device RPC0..RPCn-1 from the SUM
        # before any member joins, so this must ride the drive command.
        ordered_ids = [mesh["driver"]] + [
            wid for wid in mesh["members"] if wid != mesh["driver"]
        ]
        member_device_counts = [
            max(
                1,
                int(
                    self.state["workers"]
                    .get(wid, {})
                    .get("capability", {})
                    .get("gpu_count", 1)
                    or 1
                ),
            )
            for wid in ordered_ids
        ]
        self._queue_worker_command(
            self.state["workers"][mesh["driver"]],
            {
                "action": "drive",
                "mesh_key": mesh_key,
                "model_id": mesh["model_id"],
                "member_count": len(mesh["members"]),
                "member_device_counts": member_device_counts,
                "serving_mode": str(mesh.get("serving_mode", "")),
                **(
                    # Relaunch of the same registration: the driver resumes
                    # the previous mesh identity and its snapshot chain.
                    {"resume_mesh_id": str(mesh["resume_mesh_id"])}
                    if mesh.get("resume_mesh_id")
                    else {}
                ),
                **(
                    {"max_context_len": int(mesh["max_context_len"])}
                    if int(mesh.get("max_context_len", 0) or 0) > 0
                    else {}
                ),
                **(
                    {
                        "validator_binding": dict(mesh["validator_binding"]),
                        "coordinator_address": str(mesh["coordinator_address"]),
                        # The identity the mesh spec must be stamped with:
                        # a token-only driver cannot derive it from a local
                        # wallet, and stamping the placeholder instead fails
                        # every receipt identity check.
                        "coordinator_hotkey": str(
                            mesh.get("coordinator_hotkey", "") or ""
                        ),
                    }
                    if mesh.get("serving_mode") == POOL_SERVING_MODE_SUBNET
                    else {}
                ),
                **(
                    # Chain anchors ride only on a CHAIN-BOUND mesh; an
                    # unregistered measurement launch has none and the driver
                    # then serves the operator lane without signing a
                    # snapshot (worker drive() keys the same way).
                    {
                        "model_index": int(mesh["model_index"]),
                        "model_package_hash": str(mesh["model_package_hash"]),
                        "model_tensor_manifest_root": str(
                            mesh["model_tensor_manifest_root"]
                        ),
                        "tokenizer_hash": str(mesh["tokenizer_hash"]),
                        "quantization_scheme": str(mesh["quantization_scheme"]),
                        "total_layers": int(mesh["total_layers"]),
                        "snapshot_generation": int(mesh["snapshot_generation"]),
                    }
                    if mesh.get("serving_mode") == POOL_SERVING_MODE_SUBNET
                    and mesh.get("model_index") is not None
                    else {}
                ),
                **_operator_tuning_fields(mesh),
            },
        )

    def handle_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Operator self-test, routed through the driver's outbound channel.

        In production the mesh inference endpoint is served to validators only
        (same as the vLLM miner). The coordinator/operator is allowed to test
        its own mesh. The manager cannot reach the coordinator directly — it
        sits behind the worker's firewall/NAT — so instead of proxying inbound
        (which needs a relay tunnel that dies on every restart) we queue the
        request for the driver worker. The driver runs it against its own
        LOCAL coordinator (always reachable on 127.0.0.1) and POSTs the result
        back to /v1/pool/chat-result. No tunnels, works through any firewall.
        """
        self._auth_manage(body)
        mesh_key = str(body.get("mesh_key", ""))
        # Probes (pre-registration gate) may run canary-shaped full-context
        # requests; chat_deadline_seconds gives them the longer ceiling.
        timeout_s = chat_deadline_seconds(body)
        request_deadline_mono = time.monotonic() + timeout_s
        request_expires_at_unix_ms = int((time.time() + timeout_s) * 1000)
        with self.lock:
            mesh = self.state["meshes"].get(mesh_key)
            if mesh is None:
                raise ValueError("unknown mesh")
            self._require_mesh_routing_ready(mesh)
            driver = str(mesh.get("driver", ""))
            worker = self.state["workers"].get(driver)
            if worker is None:
                raise ValueError("mesh driver is not connected")
            self._guard_chat_slot(mesh_key, worker)
            chat_id = "c-" + uuid.uuid4().hex[:12]
            self.chat_pending.setdefault(driver, []).append(
                {
                    "chat_id": chat_id,
                    "model": str(mesh.get("model_id", "")),
                    "messages": body.get("messages")
                    or [{"role": "user", "content": str(body.get("prompt", "Hello"))}],
                    "max_tokens": int(body.get("max_tokens", 1024)),
                    "thinking": bool(body.get("thinking", True)),
                    "proof_tier": _requested_chat_proof_tier(body),
                    **(
                        {"sampler": _chat_sampler_from_body(body)}
                        if _chat_sampler_from_body(body)
                        else {}
                    ),
                    # A blocking chat may still ask the driver to stream from
                    # its coordinator: probes measure TTFT from the first
                    # delta; the relayed chunks are discarded manager-side.
                    "stream": bool(body.get("stream", False)),
                    "request_expires_at_unix_ms": request_expires_at_unix_ms,
                    # The driver's per-POST budget against its coordinator;
                    # defaults to CHAT_COORDINATOR_TIMEOUT_S when absent.
                    "coordinator_timeout_s": timeout_s,
                    **(
                        {
                            "verification_snapshot_hash": str(
                                mesh.get("verification_snapshot_hash", "") or ""
                            )
                        }
                        if self.serving_mode == POOL_SERVING_MODE_SUBNET
                        else {}
                    ),
                }
            )
            waiter: dict[str, Any] = {"event": threading.Event(), "result": None}
            self.chat_waiters[chat_id] = waiter
            self.chat_contexts[chat_id] = {
                "driver": driver,
                "mesh_key": mesh_key,
                "expected_stage_count": len(mesh.get("members") or []),
                "proof_tier": _requested_chat_proof_tier(body),
                "state": "queued",
                "delivery_token": "",
                "delivery_attempt": 0,
                "delivery_lease_deadline_mono": 0.0,
                "request_deadline_mono": request_deadline_mono,
                "request_expires_at_unix_ms": request_expires_at_unix_ms,
                "last_seq": 0,
                "stream": False,
                "client_disconnected": False,
            }
            self._mark_chat_active(mesh_key, chat_id)
        self._chat_signal(driver).set()  # wake the driver's long-poll now
        # Block OUTSIDE the lock (ThreadingHTTPServer gives us our own thread,
        # so heartbeats/status keep flowing) until the driver reports back.
        got = waiter["event"].wait(timeout_s)
        with self.lock:
            self.chat_waiters.pop(chat_id, None)
            if not got:
                context = self.chat_contexts.get(chat_id)
                if context is not None and context.get("state") in {
                    "queued",
                    "leased",
                }:
                    self._remove_pending_chat_locked(driver, chat_id)
                    self.chat_contexts.pop(chat_id, None)
                    self._clear_chat_active(chat_id)
                elif context is not None:
                    context["client_disconnected"] = True
                self._save()
            else:
                self.chat_contexts.pop(chat_id, None)
        if not got:
            raise ValueError("mesh did not respond in time")
        return waiter["result"]

    def handle_chat_result(self, body: dict[str, Any]) -> dict[str, Any]:
        """A driver worker delivering the final completion for a queued chat.

        Both the blocking path (handle_chat) and the streaming path
        (start_chat_stream) terminate here: the final carries the proof
        metadata, so it unblocks the waiter and/or closes the SSE stream.
        """
        worker_id, _proof_key = self._auth_worker(
            body,
            action="chat-result",
        )
        chat_id = str(body.get("chat_id", ""))
        with self.lock:
            now = time.time()
            self.chat_completed = {
                key: value
                for key, value in self.chat_completed.items()
                if float(value.get("expires", 0.0) or 0.0) > now
            }
            context = self.chat_contexts.get(chat_id)
            if context is None:
                completed = self.chat_completed.get(chat_id)
                if completed and completed.get("driver") == worker_id:
                    return {"status": "ok", "duplicate": True}
                return {"status": "stale"}
            if str(context.get("driver", "")) != worker_id:
                raise PermissionError("chat result does not come from its assigned driver")
            if context.get("state") in {"queued", "leased"}:
                return {"status": "stale"}
            last_seq = int(context.get("last_seq", 0) or 0)
            seq = body.get("seq")
            if type(seq) is not int:
                if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                    raise ValueError("chat result seq must be an integer")
                seq = last_seq + 1
            if context.get("state") == "done":
                if seq == last_seq:
                    return {"status": "ok", "duplicate": True}
                return {"status": "stale"}
            if seq != last_seq + 1:
                return {"status": "gap", "expected_seq": last_seq + 1}

            error = str(body.get("error", "") or "")
            request_deadline = float(
                context.get("request_deadline_mono", 0.0) or 0.0
            )
            if (
                not error
                and request_deadline
                and time.monotonic() >= request_deadline
            ):
                error = "driver final arrived after the operator request deadline"
            expected_stage_count = int(
                context.get("expected_stage_count", 0) or 0
            )
            if not error and self.serving_mode == POOL_SERVING_MODE_SUBNET:
                proof_stages = body.get("proof_stages")
                receipt_count = body.get("receipts")
                proof_mode = str(body.get("proof_mode", "") or "")
                proof_receipt_root = str(
                    body.get("proof_receipt_root", "") or ""
                )
                response_commitment = str(
                    body.get("mesh_response_commitment_hash", "") or ""
                )
                reported_snapshot_hash = str(
                    body.get("verification_snapshot_hash", "") or ""
                )
                mesh = self.state["meshes"].get(
                    str(context.get("mesh_key", "") or "")
                )
                expected_snapshot_hash = str(
                    (mesh or {}).get("verification_snapshot_hash", "") or ""
                )
                # An UNREGISTERED subnet mesh (no chain slot) signs no
                # snapshot; its operator-lane chats still carry full proof
                # coverage, verified against the local manifest instead of a
                # snapshot binding. Requiring the hash here failed every
                # measurement-lane probe outright.
                mesh_chain_bound = (
                    mesh is not None and mesh.get("model_index") is not None
                )
                snapshot_binding_ok = (
                    bool(expected_snapshot_hash)
                    and secrets.compare_digest(
                        reported_snapshot_hash,
                        expected_snapshot_hash,
                    )
                    if mesh_chain_bound
                    else not reported_snapshot_hash
                )
                # Expected proof mode mirrors the driver's tier resolution:
                # a chain-bound operator-lane chat (and any explicit hard
                # request) upgrades to the HARD relation and must report the
                # GEMM mode; a plain chat on the UNREGISTERED measurement
                # lane runs the organic light tier, whose light proof mode
                # the validator itself accepts for light receipts. The GEMM
                # mode always passes - an upgrade beyond the request is
                # never a downgrade.
                requested_tier = str(context.get("proof_tier", "") or "")
                hard_expected = (
                    mesh_chain_bound and requested_tier != "light"
                ) or requested_tier == "hard"
                proof_mode_ok = (
                    proof_mode == VERATHOS_GGML_GEMM_PROOF_MODE
                    if hard_expected
                    else proof_mode
                    in (
                        VERATHOS_GGML_GEMM_PROOF_MODE,
                        VERATHOS_GGML_LIGHT_PROOF_MODE,
                    )
                )
                failed_coverage = [
                    name
                    for name, ok in (
                        ("verified", body.get("verified") is True),
                        (
                            "receipt_verified",
                            body.get("receipt_verified") is True,
                        ),
                        (
                            "proof_stages",
                            type(proof_stages) is int
                            and proof_stages == expected_stage_count,
                        ),
                        (
                            # AT LEAST one stage-proof receipt per stage:
                            # per-stage coverage is enforced on the
                            # coordinator (require_complete_coverage
                            # against the snapshot/manifest), and one
                            # stage legitimately produces several
                            # receipts (postcommit light emits prefill +
                            # tail capture windows; observed 2
                            # receipts for 1 stage). Exact equality
                            # failed every chain-bound light chat.
                            "receipts",
                            type(receipt_count) is int
                            and receipt_count >= expected_stage_count,
                        ),
                        ("proof_mode", proof_mode_ok),
                        (
                            "proof_receipt_root",
                            bool(
                                _COMMAND_DIGEST_RE.fullmatch(
                                    proof_receipt_root
                                )
                            ),
                        ),
                        (
                            "mesh_response_commitment_hash",
                            bool(
                                _COMMAND_DIGEST_RE.fullmatch(
                                    response_commitment
                                )
                            ),
                        ),
                        ("snapshot_binding", snapshot_binding_ok),
                    )
                    if not ok
                ]
                if failed_coverage:
                    error = (
                        "driver final did not carry the expected "
                        + (
                            "snapshot-bound "
                            if mesh_chain_bound
                            else "manifest-bound "
                        )
                        + "GGML proof coverage for all "
                        f"{expected_stage_count} declared compute stages "
                        f"(failed: {', '.join(failed_coverage)})"
                    )

            result = {
                "status": "error" if error else "ok",
                "content": str(body.get("content", "")),
                "reasoning_content": str(body.get("reasoning_content", "") or ""),
                "verified": bool(body.get("verified")) and not bool(error),
                "receipt_verified": (
                    bool(body.get("receipt_verified")) and not bool(error)
                ),
                "receipts": body.get("receipts"),
                "proof_stages": body.get("proof_stages"),
                "expected_stage_count": expected_stage_count,
                "proof_mode": body.get("proof_mode"),
                "proof_receipt_root": body.get("proof_receipt_root"),
                "verification_snapshot_hash": body.get(
                    "verification_snapshot_hash"
                ),
                "mesh_response_commitment_hash": body.get(
                    "mesh_response_commitment_hash"
                ),
                "verification_scope": "coordinator",
                "deferred": body.get("deferred"),
                "deferred_obligation": body.get("deferred_obligation"),
                "usage": body.get("usage", {}),
                "engine_tps": body.get("engine_tps"),
                "prompt_tps": body.get("prompt_tps"),
                # Driver-observed timing telemetry (no artifact content):
                # ttft_s is the first visible delta on the driver's loopback
                # dial, a lower bound on validator-observed TTFT.
                "ttft_s": body.get("ttft_s"),
                "total_s": body.get("total_s"),
                "pickup_s": body.get("pickup_s"),
                "proof_wall_s": body.get("proof_wall_s"),
                "error": error,
            }
            context["state"] = "done"
            context["last_seq"] = seq
            waiter = self.chat_waiters.get(chat_id)
            stream_q = self.chat_streams.get(chat_id)
            # Safety net: the mesh is free for the next test as soon as its
            # result lands, even if a client vanished without cleanup.
            self._clear_chat_active(chat_id)
            self.chat_completed[chat_id] = {
                "driver": worker_id,
                "expires": now + CHAT_COMPLETED_TTL_S,
            }
            if waiter is None and stream_q is None:
                self.chat_contexts.pop(chat_id, None)
        if waiter is not None:
            waiter["result"] = result
            waiter["event"].set()
        if stream_q is not None:
            stream_q.put({"type": "done", **result})
        return {"status": "ok"}

    def start_chat_stream(self, body: dict[str, Any]) -> tuple[str, queue.Queue]:
        """Queue a STREAMING chat for the mesh driver; return (chat_id, queue).

        The SSE handler drains the queue: token deltas arrive via
        /v1/pool/chat-chunk, the proof-bearing final via /v1/pool/chat-result.
        """
        self._auth_manage(body)
        mesh_key = str(body.get("mesh_key", ""))
        # Same probe opt-in as handle_chat: canary-shaped full-context probes
        # need the longer deadline, interactive chats keep the short one.
        timeout_s = chat_deadline_seconds(body)
        request_deadline_mono = time.monotonic() + timeout_s
        request_expires_at_unix_ms = int((time.time() + timeout_s) * 1000)
        with self.lock:
            mesh = self.state["meshes"].get(mesh_key)
            if mesh is None:
                raise ValueError("unknown mesh")
            self._require_mesh_routing_ready(mesh)
            driver = str(mesh.get("driver", ""))
            worker = self.state["workers"].get(driver)
            if worker is None:
                raise ValueError("mesh driver is not connected")
            self._guard_chat_slot(mesh_key, worker)
            chat_id = "c-" + uuid.uuid4().hex[:12]
            self.chat_pending.setdefault(driver, []).append(
                {
                    "chat_id": chat_id,
                    "model": str(mesh.get("model_id", "")),
                    "messages": body.get("messages")
                    or [{"role": "user", "content": str(body.get("prompt", "Hello"))}],
                    "max_tokens": int(body.get("max_tokens", 1024)),
                    "thinking": bool(body.get("thinking", True)),
                    "proof_tier": _requested_chat_proof_tier(body),
                    **(
                        {"sampler": _chat_sampler_from_body(body)}
                        if _chat_sampler_from_body(body)
                        else {}
                    ),
                    "stream": True,
                    "queued_unix_ns": time.time_ns(),
                    "request_expires_at_unix_ms": request_expires_at_unix_ms,
                    "coordinator_timeout_s": timeout_s,
                    **(
                        {
                            "verification_snapshot_hash": str(
                                mesh.get("verification_snapshot_hash", "") or ""
                            )
                        }
                        if self.serving_mode == POOL_SERVING_MODE_SUBNET
                        else {}
                    ),
                }
            )
            q: queue.Queue = queue.Queue()
            self.chat_streams[chat_id] = q
            self.chat_contexts[chat_id] = {
                "driver": driver,
                "mesh_key": mesh_key,
                "expected_stage_count": len(mesh.get("members") or []),
                "proof_tier": _requested_chat_proof_tier(body),
                "state": "queued",
                "delivery_token": "",
                "delivery_attempt": 0,
                "delivery_lease_deadline_mono": 0.0,
                "request_deadline_mono": request_deadline_mono,
                "request_expires_at_unix_ms": request_expires_at_unix_ms,
                "last_seq": 0,
                "stream": True,
                "client_disconnected": False,
            }
            self._mark_chat_active(mesh_key, chat_id)
        self._chat_signal(driver).set()  # wake the driver's long-poll now
        return chat_id, q

    def handle_chat_poll(self, body: dict[str, Any]) -> dict[str, Any]:
        """Lease queued chats; pickup acknowledgement transfers ownership.

        Cuts chat pickup from a heartbeat cycle to one network RTT.
        """
        worker_id, _proof_key = self._auth_worker(
            body,
            action="chat-poll",
        )
        try:
            requested_wait = float(body.get("wait", CHAT_POLL_WAIT_S))
        except (TypeError, ValueError, OverflowError):
            requested_wait = CHAT_POLL_WAIT_S
        if not math.isfinite(requested_wait):
            requested_wait = CHAT_POLL_WAIT_S
        requested_wait = min(max(requested_wait, 0.0), CHAT_POLL_WAIT_S)
        ev = self._chat_signal(worker_id)

        # Scan before waiting: the signal is only a latency hint and may be
        # lost across handler scheduling.  A leased prompt stays in memory and
        # is re-issued with a fresh delivery token after its lease expires.
        with self.lock:
            worker = self.state["workers"].get(worker_id)
            if worker is not None:
                worker["last_chat_poll_unix"] = time.time()
            chat, next_deadline = self._lease_ready_chats_locked(
                worker_id,
                now_mono=time.monotonic(),
            )
            if chat:
                return {"status": "ok", "chat": chat}
            ev.clear()
            wait_s = requested_wait
            if next_deadline is not None:
                wait_s = min(
                    wait_s,
                    max(0.0, next_deadline - time.monotonic()),
                )

        # Wait OUTSIDE the lock so heartbeats/other workers keep flowing.
        ev.wait(timeout=wait_s)
        with self.lock:
            worker = self.state["workers"].get(worker_id)
            if worker is not None:
                worker["last_chat_poll_unix"] = time.time()
            chat, _next_deadline = self._lease_ready_chats_locked(
                worker_id,
                now_mono=time.monotonic(),
            )
            if not chat:
                ev.clear()
        return {"status": "ok", "chat": chat}

    def handle_chat_pickup(self, body: dict[str, Any]) -> dict[str, Any]:
        """Idempotently acknowledge one leased prompt before inference starts."""

        worker_id, _proof_key = self._auth_worker(
            body,
            action="chat-pickup",
        )
        chat_id = str(body.get("chat_id", "") or "")
        delivery_token = str(body.get("delivery_token", "") or "")
        if not _WORKER_AUTH_NONCE_RE.fullmatch(delivery_token):
            raise ValueError("chat delivery token is invalid")
        with self.lock:
            context = self.chat_contexts.get(chat_id)
            if context is None or context.get("state") == "done":
                return {"status": "stale"}
            if str(context.get("driver", "") or "") != worker_id:
                raise PermissionError("chat pickup does not come from its assigned driver")
            state = str(context.get("state", "") or "")
            if state == "running":
                if str(context.get("accepted_delivery_token", "") or "") == delivery_token:
                    return {"status": "ok", "duplicate": True}
                return {"status": "stale"}
            now_mono = time.monotonic()
            request_deadline = float(
                context.get("request_deadline_mono", 0.0) or 0.0
            )
            if request_deadline and now_mono >= request_deadline:
                self._expire_unowned_chat_locked(chat_id)
                return {"status": "expired"}
            if (
                state != "leased"
                or str(context.get("delivery_token", "") or "")
                != delivery_token
            ):
                return {"status": "stale"}
            lease_deadline = float(
                context.get("delivery_lease_deadline_mono", 0.0) or 0.0
            )
            if not lease_deadline or now_mono >= lease_deadline:
                return {"status": "stale"}
            context["state"] = "running"
            context["accepted_delivery_token"] = delivery_token
            context["pickup_acked_unix_ns"] = time.time_ns()
            self._remove_pending_chat_locked(worker_id, chat_id)
            mesh_key = str(context.get("mesh_key", "") or "")
            if chat_id in (self.chat_active.get(mesh_key) or {}):
                self.chat_active[mesh_key][chat_id] = time.monotonic()
        return {"status": "ok"}

    def push_chat_chunk(self, body: dict[str, Any]) -> dict[str, Any]:
        """A driver worker delivering one streamed token delta or phase marker."""
        worker_id, _proof_key = self._auth_worker(
            body,
            action="chat-chunk",
        )
        chat_id = str(body.get("chat_id", ""))
        with self.lock:
            context = self.chat_contexts.get(chat_id)
            if context is None or context.get("state") == "done":
                return {"status": "stale"}
            if str(context.get("driver", "")) != worker_id:
                raise PermissionError("chat chunk does not come from its assigned driver")
            if context.get("state") != "running":
                return {"status": "stale"}
            request_deadline = float(
                context.get("request_deadline_mono", 0.0) or 0.0
            )
            if request_deadline and time.monotonic() >= request_deadline:
                context["client_disconnected"] = True
                if not context.get("deadline_notified"):
                    context["deadline_notified"] = True
                    stream_q = self.chat_streams.get(chat_id)
                    if stream_q is not None:
                        stream_q.put(
                            {
                                "type": "error",
                                "error": "mesh request exceeded its operator deadline",
                            }
                        )
                return {"status": "expired"}
            if context.get("client_disconnected"):
                return {"status": "cancelled"}
            last_seq = int(context.get("last_seq", 0) or 0)
            seq = body.get("seq")
            if type(seq) is not int:
                if self.serving_mode == POOL_SERVING_MODE_SUBNET:
                    raise ValueError("chat chunk seq must be an integer")
                seq = last_seq + 1
            if seq <= last_seq:
                return {"status": "ok", "duplicate": True, "seq": seq}
            if seq != last_seq + 1:
                return {"status": "gap", "expected_seq": last_seq + 1}
            q = self.chat_streams.get(chat_id)
            if q is None and context.get("stream"):
                # An SSE chat whose browser vanished: tell the driver to stop.
                return {"status": "cancelled"}
            # A BLOCKING chat may still stream driver-side (probes measure
            # TTFT that way); accept and discard the deltas, tracking seq so
            # the final's ordering check holds.
            context["last_seq"] = seq
            mesh_key = str(context.get("mesh_key", "") or "")
            if chat_id in (self.chat_active.get(mesh_key) or {}):
                self.chat_active[mesh_key][chat_id] = time.monotonic()
        if q is None:
            return {"status": "ok", "seq": seq}
        if body.get("phase"):
            q.put({"type": "phase", "phase": str(body["phase"])})
        elif body.get("thinking"):
            q.put({"type": "thinking", "thinking": str(body["thinking"])})
        else:
            q.put({"type": "delta", "delta": str(body.get("delta", ""))})
        return {"status": "ok", "seq": seq}

    def end_chat_stream(self, chat_id: str) -> None:
        with self.lock:
            self.chat_streams.pop(chat_id, None)
            context = self.chat_contexts.get(chat_id)
            if context is None:
                self._clear_chat_active(chat_id)
                return
            driver = str(context.get("driver", "") or "")
            if context.get("state") in {"queued", "leased"}:
                self._remove_pending_chat_locked(driver, chat_id)
                self.chat_contexts.pop(chat_id, None)
                self._clear_chat_active(chat_id)
            elif context.get("state") == "done":
                self.chat_contexts.pop(chat_id, None)
                self._clear_chat_active(chat_id)
            else:
                # The driver already owns the request. Keep the mesh slot until
                # it acknowledges a terminal result; the next chunk observes
                # this flag and closes the local coordinator stream.
                context["client_disconnected"] = True

    def handle_stop(self, body: dict[str, Any]) -> dict[str, Any]:
        self._auth_manage(body)
        mesh_key = str(body.get("mesh_key", ""))
        with self.lock:
            mesh = self.state["meshes"].get(mesh_key)
            if mesh is None:
                raise ValueError("unknown mesh")
            # An operator stop is intent, not an outage: suspend the
            # model's auto-relaunch until the next explicit launch, or the
            # manager would immediately resurrect what the operator just
            # tore down.
            model_id = str(mesh.get("model_id", "") or "")
            registration = self._registrations_locked().get(model_id)
            if isinstance(registration, dict):
                registration["suspended_by_operator"] = True
            # Keep a stopping tombstone until every worker confirms process
            # shutdown. This prevents a visually convenient early removal from
            # making GPUs reusable while old backends still own their ports.
            self._drop_mesh(mesh_key)
            self._save()
        return {"status": "stopping", "mesh_key": mesh_key}

    def handle_remove_worker(self, body: dict[str, Any]) -> dict[str, Any]:
        """Remove an OFFLINE worker's record (operator cleanup).

        Live workers are refused: they would re-register on their next
        heartbeat anyway, so 'removing' one only produces a confusing
        flicker. Meshes the dead worker belonged to are dropped (its
        surviving members get stop commands via _drop_mesh).
        """
        self._auth_manage(body)
        worker_id = str(body.get("worker_id", ""))
        with self.lock:
            worker = self.state["workers"].get(worker_id)
            if worker is None:
                raise ValueError(f"unknown worker: {worker_id}")
            age = int(time.time()) - int(worker.get("last_seen_unix", 0))
            if age <= WORKER_STALE_S:
                raise ValueError(
                    f"worker {worker_id} is online (heartbeat {age}s ago); "
                    "stop its process first, then remove it"
                )
            affected_meshes = [
                k for k, m in self.state["meshes"].items()
                if worker_id in m.get("members", [])
            ]
            for mk in affected_meshes:
                self._drop_mesh(mk)
            self.state["workers"].pop(worker_id, None)
            for mk in affected_meshes:
                self._finish_mesh_stop_locked(mk)
            self._save()
        return {"status": "removed", "worker_id": worker_id}

    def _registrations_locked(self) -> dict[str, dict[str, Any]]:
        """The per-model on-chain registration map. Caller holds the lock.

        A pool serves SEVERAL meshes, each registered at its own index with
        its own endpoint and lease. The store was a single slot for its
        first year ("mesh_registration"), which made every second serving
        mesh invisible to the board's registration flow and the lease
        renewer; legacy slots migrate in place on first touch.
        """

        registrations = dict(self.state.get("mesh_registrations") or {})
        legacy = self.state.get("mesh_registration")
        if isinstance(legacy, dict) and legacy.get("model_id"):
            registrations.setdefault(
                str(legacy["model_id"]), dict(legacy)
            )
            self.state.pop("mesh_registration", None)
            self.state["mesh_registrations"] = registrations
        return registrations

    def handle_registration_state(self, body: dict[str, Any]) -> dict[str, Any]:
        """Store or read the on-chain registrations the lease renewer tracks.

        Deploy runs on whatever machine holds the admin token; the renewer
        runs inside this manager process. Persisting the registered tuples
        here (owner-only pool state) is what hands the 24h leases over.
        Keyed per model: a pool with a glm mesh on one machine and a qwen
        mesh on another holds BOTH registrations.
        """
        self._auth_manage(body)
        registration = body.get("registration")
        with self.lock:
            registrations = self._registrations_locked()
            if "suspend_renewal" in body:
                model_id = str(body.get("model_id", "") or "")
                if not model_id:
                    raise ValueError("model_id is required to suspend renewal")
                current = registrations.get(model_id)
                if not isinstance(current, dict):
                    raise ValueError(
                        f"no stored registration for {model_id}; cannot "
                        "coordinate renewal suspension"
                    )
                current["renewal_suspended"] = bool(body["suspend_renewal"])
                registrations[model_id] = current
                self.state["mesh_registrations"] = registrations
                renewal_in_progress = (
                    model_id in self._lease_renewals_inflight
                )
                self._save()
                return {
                    "status": "ok",
                    "registration": dict(current),
                    "registrations": {
                        key: dict(value) for key, value in registrations.items()
                    },
                    "renewal_in_progress": renewal_in_progress,
                }
            if body.get("clear"):
                # `mesh retire` deactivated the entry on chain; a stored
                # registration would keep the lease renewer resurrecting it.
                model_id = str(body.get("model_id", "") or "")
                cleared: dict[str, Any] | None = None
                if model_id:
                    cleared = registrations.pop(model_id, None)
                elif len(registrations) == 1:
                    # Pre-multi-model retire clients cleared "the" slot.
                    cleared = registrations.pop(next(iter(registrations)))
                elif registrations:
                    raise ValueError(
                        "several registrations are stored; clear needs "
                        "model_id"
                    )
                self.state["mesh_registrations"] = registrations
                # Retirement symmetry: the chain entry is deactivated, so
                # the model's stored binding is no longer a contract. A
                # launch finding it would serve chain-bound at a DEAD
                # index; the next deploy must re-measure fresh instead.
                cleared_model = str(
                    (cleared or {}).get("model_id", "") or model_id
                )
                model_entry = (
                    self.state.get("model_registry") or {}
                ).get(cleared_model)
                if isinstance(model_entry, dict):
                    for field_name in (
                        "model_index",
                        "chain_committed",
                        "max_context_len",
                        "measured_ctx_budget",
                    ):
                        model_entry.pop(field_name, None)
                self._save()
                return {
                    "status": "ok",
                    "cleared": cleared or None,
                    "registration": None,
                    "registrations": registrations,
                }
            if registration is not None:
                if not isinstance(registration, Mapping):
                    raise ValueError("registration must be a JSON object")
                required = (
                    "model_id",
                    "endpoint",
                    "quant",
                    "max_context_len",
                    "model_spec_ref",
                    "index",
                    "mesh_key",
                    "expires_at",
                )
                missing = [key for key in required if key not in registration]
                if missing:
                    raise ValueError(
                        "registration is missing: " + ", ".join(missing)
                    )
                registrations[str(registration["model_id"])] = dict(
                    registration
                )
                self.state["mesh_registrations"] = registrations
                # This write only happens AFTER a successful chain
                # registration (deploy stage 15 / renew), so it is the
                # one place a predicted model binding becomes a trusted
                # chain contract. Launches refuse unconfirmed bindings.
                model_entry = (
                    self.state.get("model_registry") or {}
                ).get(str(registration["model_id"]))
                if (
                    isinstance(model_entry, dict)
                    and model_entry.get("model_index") is not None
                    and int(model_entry["model_index"])
                    == int(registration["index"])
                ):
                    model_entry["chain_committed"] = True
                self._save()
            stored = dict(registrations)
        single = (
            dict(next(iter(stored.values()))) if len(stored) == 1 else None
        )
        return {
            "status": "ok",
            # Single-slot compat for pre-multi-model readers.
            "registration": single,
            "registrations": stored,
        }

    def handle_coordinator_sign(self, body: dict[str, Any]) -> dict[str, Any]:
        """Sign one mesh message with the coordinator hotkey, for its driver.

        Workers join with a token and nothing else. A worker the manager then
        selects to DRIVE a subnet mesh still owes the validator two
        coordinator-hotkey signatures (the verification snapshot and every
        receipt), so before this route the operator had to copy hotkey
        material onto each box that might drive -- which made the one-command
        join a half-truth.

        The manager already holds that wallet for the on-chain lease renewal,
        so it signs on the driver's behalf. It never signs arbitrary bytes:
        the caller sends a purpose name and a hash, and the message is rebuilt
        here from a fixed domain table, so nothing outside mesh snapshots and
        mesh receipts is reachable through this route -- least of all a
        substrate extrinsic.
        """

        worker_id, _ = self._auth_worker(body, action="coordinator-sign")
        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            raise PermissionError("coordinator signing is a subnet-pool operation")
        purpose = str(body.get("purpose", "") or "")
        mesh_key = str(body.get("mesh_key", "") or "")
        # Shape is settled before any key is touched: either a mesh message
        # rebuilt from the fixed domain table, or a 32-byte challenge nonce.
        nonce = b""
        message = b""
        artifact: dict[str, Any] = {}
        if purpose == IDENTITY_CHALLENGE_PURPOSE:
            nonce_hex = str(body.get("nonce", "") or "")
            try:
                nonce = bytes.fromhex(nonce_hex)
            except ValueError as exc:
                raise ValueError("identity challenge nonce must be hex") from exc
            if len(nonce) != 32:
                raise ValueError("identity challenge nonce must be 32 bytes")
        elif purpose == CAPACITY_AUDIT_ARTIFACT_PURPOSE:
            # The worker sends the full unsigned artifact DICT; the EIP-191
            # text is rebuilt HERE from the fixed capacity-audit framing
            # (never caller bytes), so nothing outside a capacity artifact
            # -- least of all an RLP transaction -- is reachable.
            raw_artifact = body.get("artifact")
            if not isinstance(raw_artifact, Mapping):
                raise ValueError("capacity artifact must be a JSON object")
            artifact = dict(raw_artifact)
            artifact.pop("miner_signature", None)
            artifact.pop("signature", None)
            from neurons.capacity_audit import canonical_json
            from verallm.mesh.delegated_signing import (
                max_capacity_artifact_bytes,
            )

            artifact_type = str(artifact.get("artifact_type", "") or "")
            if artifact_type not in CAPACITY_ARTIFACT_TYPES:
                raise ValueError("unknown capacity artifact type")
            if (
                len(canonical_json(artifact).encode("utf-8"))
                > max_capacity_artifact_bytes(artifact_type)
            ):
                raise ValueError("capacity artifact is too large")
            for field_name in ("audit_id", "slot_id", "worker_id"):
                if not str(artifact.get(field_name, "") or ""):
                    raise ValueError(
                        f"capacity artifact is missing {field_name}"
                    )
            if str(artifact.get("worker_id", "") or "") != worker_id:
                # A worker may only obtain signatures over artifacts naming
                # ITSELF; without this one member could sign openings for a
                # sibling GPU it does not have.
                raise PermissionError(
                    "capacity artifact worker_id must be the requesting worker"
                )
        else:
            message = coordinator_sign_message(
                purpose, str(body.get("body_hash", "") or "")
            )

        with self.lock:
            wallet_name = str(self.state.get("wallet_name", "") or "")
            wallet_hotkey = str(self.state.get("wallet_hotkey", "") or "")
            mesh = dict(self.state.get("meshes", {}).get(mesh_key) or {})
            worker = dict(self.state.get("workers", {}).get(worker_id) or {})
        if not wallet_name or not wallet_hotkey:
            raise PermissionError(
                "this pool manager runs without a coordinator wallet, so it "
                "cannot sign for its drivers"
            )
        if not mesh:
            raise PermissionError("unknown mesh")
        # Authority, not just shape: only the worker THIS manager appointed as
        # the mesh's driver can obtain coordinator signatures for it. A
        # member-only stage worker is refused even though its token is valid.
        # Capacity artifacts are the one deliberate exception: audit openings
        # run on EVERY member's GPUs simultaneously and each member signs its
        # own artifacts, so mesh-bound membership (checked below) is the
        # authority, not driver appointment.
        if (
            purpose != CAPACITY_AUDIT_ARTIFACT_PURPOSE
            and str(mesh.get("driver", "") or "") != worker_id
        ):
            raise PermissionError("only the assigned mesh driver may request this")
        if str(worker.get("mesh", "") or "") != mesh_key:
            raise PermissionError("worker is not currently bound to that mesh")
        if purpose == CAPACITY_AUDIT_ARTIFACT_PURPOSE:
            # The artifact must name the mesh's CONFIRMED chain slot: signing
            # for an unregistered measurement mesh, or for a different
            # model_index than the chain entry, would mint audit evidence for
            # a slot this mesh does not serve.
            mesh_model_index = mesh.get("model_index")
            if mesh_model_index is None:
                raise PermissionError(
                    "capacity artifacts require a chain-bound mesh"
                )
            if int(artifact.get("model_index", -1)) != int(mesh_model_index):
                raise PermissionError(
                    "capacity artifact model_index does not match this mesh's "
                    "chain entry"
                )

        from verallm.mesh.receipt_signing import load_hotkey_keypair

        keypair = load_hotkey_keypair(wallet_name, wallet_hotkey)
        if purpose == IDENTITY_CHALLENGE_PURPOSE:
            from eth_account import Account
            from eth_account.messages import encode_defunct

            from verallm.chain.wallet import (
                derive_evm_address,
                derive_evm_private_key,
            )
            from verallm.mesh.receipt_signing import load_hotkey_seed

            hotkey_seed = load_hotkey_seed(
                wallet_name, wallet_hotkey, keypair=keypair
            )
            evm_address = derive_evm_address(hotkey_seed)
            signed = Account.sign_message(
                encode_defunct(
                    primitive=identity_challenge_message(nonce, evm_address)
                ),
                private_key=derive_evm_private_key(hotkey_seed),
            )
            return {
                "status": "ok",
                "signature": signed.signature.hex(),
                "evm_address": evm_address,
            }
        if purpose == CAPACITY_AUDIT_ARTIFACT_PURPOSE:
            from neurons.capacity_audit import sign_artifact

            from verallm.chain.wallet import (
                derive_evm_address,
                derive_evm_private_key,
            )
            from verallm.mesh.receipt_signing import load_hotkey_seed

            hotkey_seed = load_hotkey_seed(
                wallet_name, wallet_hotkey, keypair=keypair
            )
            evm_address = derive_evm_address(hotkey_seed)
            artifact_address = str(artifact.get("address", "") or "").lower()
            if artifact_address != evm_address.lower():
                raise PermissionError(
                    "capacity artifact address must be the coordinator EVM "
                    "address this pool registered on chain"
                )
            signature = sign_artifact(
                artifact, derive_evm_private_key(hotkey_seed)
            )
            return {
                "status": "ok",
                "signature": signature,
                "evm_address": evm_address,
            }
        signature = keypair.sign(message)
        signature_hex = (
            bytes(signature).hex()
            if isinstance(signature, (bytes, bytearray))
            else str(signature).removeprefix("0x")
        )
        return {
            "status": "ok",
            "signature": signature_hex.lower(),
            "coordinator_hotkey": str(keypair.ss58_address),
        }

    def handle_register_model(self, body: dict[str, Any]) -> dict[str, Any]:
        """Teach the pool a model no worker has yet (operator 'add model').

        Without this, the registry only learns models from the catalog advert
        of a worker that already holds the file, so a brand-new model could
        never enter the pool through the app: someone had to download it and
        hand-write a catalog entry first. Registering just the download
        source lets any capable driver auto-fetch it at launch.
        """
        self._auth_manage(body)
        model_id = str(body.get("model_id", ""))
        hf_repo = str(body.get("hf_repo", ""))
        hf_files = [str(f) for f in (body.get("hf_files") or [])]
        if not model_id or not hf_repo or not hf_files:
            raise ValueError("model_id, hf_repo and hf_files are required")
        entry = {
            "hf_repo": hf_repo,
            "hf_files": hf_files,
            "layers": int(body.get("layers", 0) or 0),
            "model_bytes": int(body.get("model_bytes", 0) or 0),
            **_operator_tuning_fields(body),
        }
        if entry.get("llama_ubatch") is not None:
            ubatch = int(entry["llama_ubatch"])
            if not 128 <= ubatch <= 8192:
                raise ValueError("llama_ubatch must be within [128, 8192]")
        if entry.get("llama_batch") is not None:
            batch = int(entry["llama_batch"])
            floor = int(entry.get("llama_ubatch") or 128)
            if not floor <= batch <= 16384:
                raise ValueError(
                    "llama_batch must be within [llama_ubatch, 16384]"
                )
        if "max_context_len" in body:
            entry["max_context_len"] = _require_max_context_len(
                body.get("max_context_len")
            )
        # GLEIPNIR-style manifest distribution next to the model source:
        # drivers try these before rebuilding the manifest locally. Pure
        # discovery metadata; every download still has to reproduce the
        # registered model_tensor_manifest_root or it is refused. When the
        # registration names no URLs, the pool's own configured store
        # (VERATHOS_MESH_MANIFEST_BASE_URLS, canonically the same hosting
        # directory as the GLEIPNIR artifact store) is stamped in so every
        # driver benefits without per-model setup.
        from verallm.mesh.manifest_store import (
            configured_mesh_manifest_base_urls,
            normalize_mesh_manifest_base_urls,
        )

        manifest_urls = normalize_mesh_manifest_base_urls(
            body.get("manifest_urls") or ()
        ) or configured_mesh_manifest_base_urls()
        if len(manifest_urls) > 8:
            raise ValueError("at most 8 manifest_urls are accepted")
        if manifest_urls:
            entry["manifest_urls"] = list(manifest_urls)
        if body.get("model_index") is not None:
            model_index = int(body["model_index"])
            if model_index < 0:
                raise ValueError("model_index must be non-negative")
            entry["model_index"] = model_index
            # A binding written here is a PREDICTION (deploy stamps it
            # before the chain write so the relaunch's snapshot can bind
            # the index). It becomes trusted only when the post-chain
            # lease write confirms it (handle_registration_state). An
            # aborted deploy's leftover prediction must never launch as
            # a chain contract -
            # left model_index 45 + a poisoned 8192 contract that every
            # later launch inherited as its floor.
            entry["chain_committed"] = False
        anchor_fields = {
            "model_package_hash": body.get("model_package_hash"),
            "model_tensor_manifest_root": body.get("model_tensor_manifest_root"),
            "tokenizer_hash": body.get("tokenizer_hash"),
        }
        supplied_anchors = any(value not in (None, "") for value in anchor_fields.values())
        if supplied_anchors or self.serving_mode == POOL_SERVING_MODE_SUBNET:
            for field_name, value in anchor_fields.items():
                entry[field_name] = _require_digest(value, field_name=field_name)
            quantization_scheme = str(body.get("quantization_scheme", "") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", quantization_scheme):
                raise ValueError(
                    "quantization_scheme must be a bounded protocol token"
                )
            entry["quantization_scheme"] = quantization_scheme
        if (
            self.serving_mode == POOL_SERVING_MODE_SUBNET
            and "model_index" not in entry
        ):
            # SOURCE-ONLY registration (the board's launch flow teaching a
            # download source before anything measured or registered): the
            # chain binding (model_index + measured max_context_len)
            # arrives via deploy AFTER measurement, and an unregistered
            # model cannot carry it yet. Requiring it here made every
            # catalog launch on a subnet pool fail with HTTP 400. The launch
            # treats an index-less entry as the
            # measurement/operator lane, exactly as intended. A stored
            # chain binding must survive a later source-only update:
            # entries replace wholesale below, and wiping model_index
            # would silently flip a registered model back to the
            # measurement lane.
            stored = dict(
                (self.state.get("model_registry") or {}).get(model_id) or {}
            )
            for keep in (
                "model_index",
                "chain_committed",
                "max_context_len",
                "measured_ctx_budget",
                "model_package_hash",
                "model_tensor_manifest_root",
                "tokenizer_hash",
                "quantization_scheme",
            ):
                if keep in stored and keep not in entry:
                    entry[keep] = stored[keep]
        if (
            self.serving_mode == POOL_SERVING_MODE_SUBNET
            and body.get("model_index") is not None
        ):
            # max_context_len is the on-chain contract AND the serve's KV
            # budget (_serve_ctx_budget), and validators canary at exactly
            # this number. It must therefore be what the hardware MEASURED,
            # so an unregistered launch records measured_ctx_budget and this
            # is where it becomes the committed value. An explicit override
            # is honored but may not exceed what was measured.
            stored = dict(
                (self.state.get("model_registry") or {}).get(model_id) or {}
            )
            measured = int(
                body.get("measured_ctx_budget", 0)
                or stored.get("measured_ctx_budget", 0)
                or 0
            )
            if measured > 0:
                entry["measured_ctx_budget"] = measured
            if "max_context_len" not in entry:
                if measured <= 0:
                    raise ValueError(
                        "subnet pool models require max_context_len, and no "
                        "launch has measured this model's context yet: run "
                        "`mesh pool launch --model-id <id>` first so the KV "
                        "auto-fit measures what this hardware holds, then "
                        "register (the measured value is used automatically)"
                    )
                entry["max_context_len"] = measured
            else:
                requested = int(entry["max_context_len"])
                ceiling = int(measured * (1.0 + MESH_CTX_JITTER_TOLERANCE))
                if measured > 0 and requested > ceiling:
                    raise ValueError(
                        f"max_context_len {requested} exceeds the measured "
                        f"context this hardware holds ({measured}) by more "
                        f"than {int(MESH_CTX_JITTER_TOLERANCE * 100)}%; a "
                        "validator canaries at the registered maximum, so "
                        "registering more than the mesh can serve fails an "
                        "honest audit"
                    )
        with self.lock:
            # Operator-pinned proof tolerances survive EVERY re-register that
            # does not explicitly override them. The deploy re-registers the
            # model after measurement with a body that carries no tolerance
            # fields; replacing the entry wholesale silently dropped the
            # q2_k_xl override, the next relaunch falls back to the default
            # band and can re-probate an honest mesh.
            stored_tunings = _operator_tuning_fields(
                (self.state.get("model_registry") or {}).get(model_id) or {}
            )
            for key, value in stored_tunings.items():
                entry.setdefault(key, value)
            self.state.setdefault("model_registry", {})[model_id] = entry
            self._save()
        return {"status": "ok", "model_id": model_id, "registry": entry}

    def handle_set_epoch(self, body: dict[str, Any]) -> dict[str, Any]:
        """Stage a later validator epoch for the next mesh launch.

        Existing mesh records retain their copied binding. This lets an
        operator stage the upcoming epoch while the current mesh still serves,
        then stop and relaunch near the boundary without mutating a live signed
        snapshot.
        """

        self._auth_manage(body)
        if self.serving_mode != POOL_SERVING_MODE_SUBNET:
            raise ValueError("epoch binding is only valid for validator pools")
        epoch = body.get("epoch")
        if type(epoch) is not int or epoch < 0 or epoch >= 2**63:
            raise ValueError("epoch must be an integer in [0, 2^63)")
        with self.lock:
            current = int(self.validator_binding.get("epoch", -1))
            if epoch == current:
                raise ValueError("new epoch must differ from the current binding")
            floor = int(
                self.state.get(
                    "validator_epoch_floor",
                    current,
                )
            )
            if epoch <= floor:
                raise ValueError(
                    "new epoch must be later than every epoch already used "
                    "by this pool"
                )
            updated = dict(self.validator_binding)
            updated["epoch"] = epoch
            self.validator_binding = _validator_binding(**updated)
            self.state["validator_binding"] = dict(self.validator_binding)
            self._save()
        return {
            "status": "staged",
            "previous_epoch": current,
            "epoch": epoch,
            "corrected": epoch < current,
        }


def ensure_pool_api_tls_cert(state_dir: str | Path) -> tuple[Path, Path]:
    """Self-signed cert/key for the private API's https listener.

    Minted once into the pool dir (owner-only) via the openssl binary and
    reused across restarts, mirroring the stage-key lifecycle. Self-signed
    is the honest default on rented boxes with no domain: clients pin or
    accept the printed fingerprint (CA distribution is the open console
    P3 item).
    """

    import subprocess

    pool_dir = Path(state_dir)
    certfile = pool_dir / "api-tls-cert.pem"
    keyfile = pool_dir / "api-tls-key.pem"
    if certfile.is_file() and keyfile.is_file():
        return certfile, keyfile
    completed = subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(keyfile), "-out", str(certfile),
            "-days", "1825", "-nodes", "-subj", "/CN=verathos-pool-api",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not keyfile.is_file():
        raise RuntimeError(
            "could not mint the API TLS certificate (is the openssl "
            f"binary installed?): {completed.stderr.strip()[:300]}"
        )
    os.chmod(keyfile, 0o600)
    os.chmod(certfile, 0o644)
    return certfile, keyfile


def serve_pool_manager(
    state_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 9500,
    coordinator_address: str | None = None,
    validator_shared_state_path: str | Path | None = None,
    tls_certfile: str | Path | None = None,
    tls_keyfile: str | Path | None = None,
    api_tls_port: int = 0,
    api_tls_certfile: str | Path | None = None,
    api_tls_keyfile: str | Path | None = None,
    manifest_base_urls: Sequence[str] = (),
    announce_chain_config: str | Path | None = None,
) -> ThreadingHTTPServer:
    """Blockingless pool manager HTTP(S) server (call serve_forever yourself).

    ``api_tls_port`` opens a SECOND listener serving the same handler over
    TLS, so the private OpenAI API (and dashboard) are reachable from the
    internet while workers keep dialing the plain-http port; certs default
    to the pool's auto-minted self-signed pair. The extra server rides
    ``server.api_tls_server`` and the caller runs its ``serve_forever`` on
    a thread.
    """

    if bool(tls_certfile) != bool(tls_keyfile):
        raise ValueError("TLS requires both a certificate and private-key file")
    if tls_keyfile:
        # Fail before binding a remotely reachable manager with a private key
        # that another local user can read.
        read_owner_only_text(tls_keyfile, label="TLS private key")

    manager = PoolManager(
        state_dir,
        coordinator_address=coordinator_address,
        validator_shared_state_path=validator_shared_state_path,
        manifest_base_urls=manifest_base_urls,
    )
    routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "/v1/pool/join": manager.handle_join,
        "/v1/pool/heartbeat": manager.handle_heartbeat,
        "/v1/pool/report": manager.handle_report,
        "/v1/pool/status": manager.handle_status,
        "/v1/pool/recommend": manager.handle_recommend,
        "/v1/pool/launch": manager.handle_launch,
        "/v1/pool/stop": manager.handle_stop,
        "/v1/pool/chat": manager.handle_chat,
        "/v1/pool/chat-result": manager.handle_chat_result,
        "/v1/pool/chat-chunk": manager.push_chat_chunk,
        "/v1/pool/chat-poll": manager.handle_chat_poll,
        "/v1/pool/chat-pickup": manager.handle_chat_pickup,
        "/v1/pool/register-model": manager.handle_register_model,
        "/v1/pool/deploy-status": manager.handle_deploy_status,
        "/v1/pool/coordinator-sign": manager.handle_coordinator_sign,
        "/v1/pool/registration-state": manager.handle_registration_state,
        "/v1/pool/set-epoch": manager.handle_set_epoch,
        "/v1/pool/remove-worker": manager.handle_remove_worker,
        "/v1/auth/challenge": manager.handle_auth_challenge,
        "/v1/auth/verify": manager.handle_auth_verify,
        "/v1/auth/logout": manager.handle_auth_logout,
        "/v1/auth/session": manager.handle_auth_session,
        "/v1/operator/overview": manager.handle_operator_overview,
        "/v1/operator/access": manager.handle_operator_access,
        "/v1/operator/score": manager.handle_operator_score,
        "/v1/pool/api-keys": manager.handle_api_keys,
    }

    class PoolHandler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(POOL_REQUEST_SOCKET_TIMEOUT_S)
            # Deferred TLS handshake (see the wrap_socket call): runs here
            # on the handler thread, bounded by the timeout above, so a
            # stalled or non-TLS client costs one thread for one timeout
            # instead of wedging the accept loop.
            if isinstance(self.connection, ssl.SSLSocket):
                try:
                    self.connection.do_handshake()
                except (ssl.SSLError, OSError) as exc:
                    # Port scanners, health probes, and aborted clients all
                    # die here; a full traceback per attempt floods the log
                    # (hundreds of SSLEOFError dumps).
                    raise _QuietHandshakeFailure(str(exc)) from None

        def log_message(self, *args: Any) -> None:  # quiet
            del args

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_bootstrap_script(self) -> None:
            """Serve the public worker-join bootstrap (no secrets inside)."""

            from verallm.mesh.onboarding import find_repo_root

            script = find_repo_root() / "scripts" / "pool_bootstrap.sh"
            try:
                raw = script.read_bytes()
            except OSError:
                self._send(404, {"error": "bootstrap script unavailable"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/x-shellscript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_worker_source_bundle(self) -> None:
            """Stream the repo source tarball to a WORKER-token-authed caller.

            The source tree is private; the bundle route therefore requires
            the same credential that lets a machine join the pool at all.
            """

            header = str(self.headers.get("Authorization", "") or "")
            raw_token = header[len("Bearer "):] if header.startswith("Bearer ") else ""
            try:
                token = MeshPoolToken.decode(raw_token)
            except Exception:
                self._send(401, {"error": "worker token required"})
                return
            if token.pool_id != str(manager.state.get("pool_id", "")) or not (
                secrets.compare_digest(
                    token.pool_secret,
                    str(manager.state.get("pool_secret", "")),
                )
            ):
                self._send(403, {"error": "wrong pool token"})
                return
            try:
                bundle = manager.ensure_worker_source_bundle()
            except Exception as exc:
                self._send(500, {"error": f"bundle unavailable: {exc}"})
                return
            size = bundle.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with bundle.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile)

        def _send_html(self, code: int, html: str) -> None:
            raw = html.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                "form-action 'self'",
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported")
            get_all = getattr(self.headers, "get_all", None)
            lengths = get_all("Content-Length", []) if callable(get_all) else []
            if len(lengths) > 1:
                raise ValueError("duplicate Content-Length header")
            try:
                length = int(
                    lengths[0]
                    if lengths
                    else self.headers.get("Content-Length", "0")
                )
            except (TypeError, ValueError):
                raise ValueError("invalid Content-Length header") from None
            if length < 0:
                raise ValueError("invalid Content-Length header")
            body_cap = (
                MAX_POOL_SIGN_REQUEST_BODY_BYTES
                if (self.path or "").rstrip("/").endswith(
                    "/pool/coordinator-sign"
                )
                else MAX_POOL_REQUEST_BODY_BYTES
            )
            if length > body_cap:
                raise OverflowError("request body exceeds configured limit")
            raw = self.rfile.read(length) if length else b"{}"
            if length and len(raw) != length:
                raise ValueError("request body ended before Content-Length")

            def reject_duplicate_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key: {key}")
                    result[key] = value
                return result

            data = json.loads(
                raw,
                object_pairs_hook=reject_duplicate_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value is forbidden: {value}")
                ),
            )
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _openai_error(self, code: int, message: str) -> None:
            self._send(
                code,
                {
                    "error": {
                        "message": str(message),
                        "type": (
                            "invalid_request_error"
                            if code < 500
                            else "server_error"
                        ),
                    }
                },
            )

        def _require_api_key(self) -> bool:
            key_hash = manager.authenticate_api_key(
                self.headers.get("Authorization", "")
            )
            if not key_hash:
                # Per-IP backoff on failed auth: an exposed port must not
                # be brute-forceable through this surface.
                client_ip = (
                    str(self.client_address[0]) if self.client_address else "?"
                )
                if not manager.api_request_allowed(
                    f"authfail:{client_ip}",
                    limit=POOL_API_AUTH_FAILURES_PER_MIN,
                ):
                    self._openai_error(
                        429,
                        "too many failed authentication attempts from this "
                        "address; retry in a minute",
                    )
                    return False
                self._openai_error(
                    401,
                    "missing or invalid API key; mint one with "
                    "`verathos mesh apikey create`",
                )
                return False
            if not manager.api_request_allowed(
                f"key:{key_hash}", limit=POOL_API_RATE_LIMIT_PER_MIN
            ):
                self._openai_error(
                    429,
                    "rate limit exceeded "
                    f"({POOL_API_RATE_LIMIT_PER_MIN} requests/min per key); "
                    "retry in a minute",
                )
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            # Strip the query string before matching browser routes.
            path = self.path.split("?", 1)[0]
            if path in (
                "/", "/dashboard", "/index.html", "/operator", "/operator/"
            ):
                from verallm.mesh.pool_dashboard import DASHBOARD_HTML

                self._send_html(200, DASHBOARD_HTML)
            elif path == "/healthz":
                self._send(200, {"status": "ok", "pool_id": manager.state["pool_id"]})
            elif path == "/install.sh":
                # Public bootstrap for the one-command worker join. Carries
                # no secrets: the caller supplies the join token, and the
                # source bundle below requires it. verathos.ai/install.sh
                # mirrors the same script after the public release.
                self._send_bootstrap_script()
            elif path == "/v1/pool/worker-src":
                self._send_worker_source_bundle()
            elif path == "/v1/models":
                if not self._require_api_key():
                    return
                from verallm.mesh.openai_api import openai_models_payload

                status = manager.handle_status(
                    {"management_secret": manager.state["management_secret"]}
                )
                self._send(200, openai_models_payload(status))
            else:
                self._send(404, {"error": "not found"})

        def _chat_stream(self) -> None:
            """Server-Sent Events: relay the driver's streamed token deltas to
            the browser, then the proof-bearing final event."""
            try:
                body = self._read_json_body()
                stream_deadline_s = chat_deadline_seconds(body)
                chat_id, chunk_q = manager.start_chat_stream(body)
            except OverflowError as exc:
                self._send(413, {"error": str(exc)})
                return
            except PermissionError as exc:
                self._send(403, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._send(500, {"error": str(exc)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            deadline = time.monotonic() + stream_deadline_s
            finished = False
            try:
                while time.monotonic() < deadline:
                    try:
                        item = chunk_q.get(timeout=5.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")  # hold the connection
                        self.wfile.flush()
                        continue
                    self.wfile.write(("data: " + json.dumps(item) + "\n\n").encode())
                    self.wfile.flush()
                    if item.get("type") in ("done", "error"):
                        finished = True
                        break
                if not finished:
                    # Timing out silently strands the browser on a dead stream
                    # ("Load failed"); tell it what happened instead.
                    self.wfile.write(
                        ("data: " + json.dumps({
                            "type": "error",
                            "error": (
                                "mesh did not respond within "
                                f"{int(CHAT_OPERATOR_DEADLINE_S)}s; the driver "
                                "may be overloaded or its backend hung"
                            ),
                        }) + "\n\n").encode()
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                manager.end_chat_stream(chat_id)

        def _openai_chat(self) -> None:
            """The private pool API: OpenAI chat completions over the
            existing queue/driver plumbing. Bearer-key authed, proofs
            always on (organic light tier, greedy or the committed
            sampled profile)."""

            from verallm.mesh import openai_api

            if not self._require_api_key():
                return
            try:
                request = self._read_json_body()
            except OverflowError as exc:
                self._openai_error(413, str(exc))
                return
            except ValueError as exc:
                self._openai_error(400, str(exc))
                return
            secret = manager.state["management_secret"]
            request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            try:
                status = manager.handle_status({"management_secret": secret})
                mesh_key, model_id = openai_api.resolve_mesh_for_model(
                    status, request.get("model", "")
                )
                tools = request.get("tools") or []
                tools_active = bool(tools) and openai_api.tool_choice_mode(
                    request.get("tool_choice")
                ) != "none"
                stream = bool(request.get("stream", False))
                if tools_active and stream:
                    raise ValueError(
                        "streaming with tools is not supported; send "
                        "stream=false for tool calls"
                    )
                if tools_active:
                    decision_request = {
                        **request,
                        "messages": openai_api.build_tool_decision_messages(
                            list(request.get("messages") or []),
                            list(tools),
                            request.get("tool_choice"),
                            request.get("parallel_tool_calls"),
                        ),
                        "stream": False,
                    }
                    pool_body = openai_api.pool_chat_body_from_openai(
                        decision_request, mesh_key=mesh_key
                    )
                else:
                    pool_body = openai_api.pool_chat_body_from_openai(
                        request, mesh_key=mesh_key
                    )
            except ValueError as exc:
                self._openai_error(400, str(exc))
                return
            pool_body["management_secret"] = secret
            if not stream:
                try:
                    result = manager.handle_chat(pool_body)
                except ValueError as exc:
                    self._openai_error(502, str(exc))
                    return
                if not isinstance(result, dict) or result.get("error"):
                    self._openai_error(
                        502,
                        str(
                            (result or {}).get("error", "")
                            or "mesh chat failed"
                        ),
                    )
                    return
                if tools_active:
                    message, finish_reason = (
                        openai_api.normalize_tool_decision(
                            str(result.get("content", "") or ""),
                            list(tools),
                            request.get("tool_choice"),
                            request.get("parallel_tool_calls"),
                        )
                    )
                    self._send(
                        200,
                        openai_api.openai_response_from_result(
                            result,
                            model=model_id,
                            request_id=request_id,
                            message=message,
                            finish_reason=finish_reason,
                        ),
                    )
                    return
                self._send(
                    200,
                    openai_api.openai_response_from_result(
                        result, model=model_id, request_id=request_id
                    ),
                )
                return
            # Streaming: relay the driver's SSE deltas as OpenAI chunks.
            try:
                stream_deadline_s = chat_deadline_seconds(pool_body)
                chat_id, chunk_q = manager.start_chat_stream(pool_body)
            except ValueError as exc:
                self._openai_error(400, str(exc))
                return
            except PermissionError as exc:
                self._openai_error(403, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            def _emit(payload: dict[str, Any]) -> None:
                self.wfile.write(
                    ("data: " + json.dumps(payload) + "\n\n").encode()
                )
                self.wfile.flush()

            deadline = time.monotonic() + stream_deadline_s
            sent_role = False
            finished = False
            try:
                while time.monotonic() < deadline:
                    try:
                        item = chunk_q.get(timeout=5.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    kind = str(item.get("type", ""))
                    if kind == "phase":
                        continue
                    if kind == "thinking":
                        # Reasoning deltas ride the OpenAI-standard
                        # delta.reasoning_content channel (same shape
                        # llama-server emits with --reasoning-format
                        # deepseek), so clients see the think phase live.
                        delta = {}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        delta["reasoning_content"] = str(
                            item.get("thinking", "") or ""
                        )
                        _emit(
                            openai_api.openai_chunk(
                                request_id=request_id,
                                model=model_id,
                                delta=delta,
                            )
                        )
                        continue
                    if kind == "delta":
                        delta: dict[str, Any] = {}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        delta["content"] = str(item.get("delta", "") or "")
                        _emit(
                            openai_api.openai_chunk(
                                request_id=request_id,
                                model=model_id,
                                delta=delta,
                            )
                        )
                        continue
                    if kind == "done":
                        result = dict(item.get("result") or item)
                        if result.get("error"):
                            _emit(
                                {
                                    "error": {
                                        "message": str(result["error"]),
                                        "type": "server_error",
                                    }
                                }
                            )
                            finished = True
                            break
                        _emit(
                            openai_api.openai_chunk(
                                request_id=request_id,
                                model=model_id,
                                finish_reason="stop",
                                extra={
                                    **(
                                        {"usage": result["usage"]}
                                        if isinstance(
                                            result.get("usage"), Mapping
                                        )
                                        else {}
                                    ),
                                    "verathos": {
                                        key: result.get(key)
                                        for key in (
                                            "verified",
                                            "receipt_verified",
                                            "proof_mode",
                                            "proof_receipt_root",
                                        )
                                        if result.get(key) is not None
                                    },
                                },
                            )
                        )
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        finished = True
                        break
                    if kind == "error":
                        _emit(
                            {
                                "error": {
                                    "message": str(
                                        item.get("error", "") or "mesh error"
                                    ),
                                    "type": "server_error",
                                }
                            }
                        )
                        finished = True
                        break
                if not finished:
                    _emit(
                        {
                            "error": {
                                "message": "mesh did not respond in time",
                                "type": "server_error",
                            }
                        }
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                manager.end_chat_stream(chat_id)

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/v1/pool/chat-stream":
                self._chat_stream()
                return
            if self.path == "/v1/chat/completions":
                self._openai_chat()
                return
            handler = routes.get(self.path)
            if handler is None:
                self._send(404, {"error": "not found"})
                return
            try:
                body = self._read_json_body()
                self._send(200, handler(body))
            except OverflowError as exc:
                self._send(413, {"error": str(exc)})
            except PermissionError as exc:
                self._send(403, {"error": str(exc)})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                self._send(500, {"error": str(exc)})

    def _pool_http_server(address: tuple[str, int]) -> Any:
        # The default listen backlog (5) drops concurrent connects under a
        # normal burst: four simultaneous delegated-sign requests (one per
        # GPU of a 4-GPU mesh audit window) plus heartbeats and roster
        # pulls can overflow it so extra clients see a TLS EOF and payloads
        # go unsigned.
        base = ThreadingHTTPServer

        def _quiet_handle_error(self, request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, _QuietHandshakeFailure):
                logger.warning(
                    "tls handshake failed from %s: %s", client_address, exc
                )
                return
            base.handle_error(self, request, client_address)

        if isinstance(base, type):
            cls = type(
                "PoolThreadingHTTPServer",
                (base,),
                {
                    "request_queue_size": 128,
                    "daemon_threads": True,
                    "handle_error": _quiet_handle_error,
                },
            )
            return cls(address, PoolHandler)
        return base(address, PoolHandler)  # test stubs replace the class

    server = _pool_http_server((host, int(port)))
    if tls_certfile and tls_keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.options |= ssl.OP_NO_COMPRESSION
        context.load_cert_chain(
            certfile=str(Path(tls_certfile).expanduser()),
            keyfile=str(Path(tls_keyfile).expanduser()),
        )
        # NEVER handshake in accept(): with the default
        # do_handshake_on_connect=True the TLS handshake runs on the
        # serve_forever thread BEFORE any per-connection timeout exists, so
        # one client that connects and sends nothing wedges the whole
        # listener forever. The
        # handshake runs in PoolHandler.setup() instead, on the handler
        # thread, bounded by the request socket timeout.
        server.socket = context.wrap_socket(
            server.socket, server_side=True, do_handshake_on_connect=False
        )
        server.tls_enabled = True  # type: ignore[attr-defined]
    else:
        server.tls_enabled = False  # type: ignore[attr-defined]
    server.pool_manager = manager  # type: ignore[attr-defined]
    server.api_tls_server = None  # type: ignore[attr-defined]
    if int(api_tls_port):
        if not (api_tls_certfile and api_tls_keyfile):
            api_tls_certfile, api_tls_keyfile = ensure_pool_api_tls_cert(
                manager.state_path.parent
            )
        api_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        api_context.minimum_version = ssl.TLSVersion.TLSv1_2
        api_context.options |= ssl.OP_NO_COMPRESSION
        api_context.load_cert_chain(
            certfile=str(Path(api_tls_certfile).expanduser()),
            keyfile=str(Path(api_tls_keyfile).expanduser()),
        )
        api_server = _pool_http_server((host, int(api_tls_port)))
        # Same accept-wedge defense as the primary listener above.
        api_server.socket = api_context.wrap_socket(
            api_server.socket, server_side=True, do_handshake_on_connect=False
        )
        api_server.tls_enabled = True  # type: ignore[attr-defined]
        server.api_tls_server = api_server  # type: ignore[attr-defined]
        # Persist so the board/CLI can print the working https URL.
        with manager.lock:
            if manager.state.get("api_tls_port") != int(api_tls_port):
                manager.state["api_tls_port"] = int(api_tls_port)
                manager._save()
    else:
        with manager.lock:
            if manager.state.pop("api_tls_port", None) is not None:
                manager._save()
    # Warm the chain-score cache off the request path: the first metagraph
    # read takes 10-30s, so without this the board's first render after
    # every (re)start said "chain status unavailable". Chain-blind dev
    # pools return instantly from this call.
    threading.Thread(
        target=lambda: manager.handle_operator_score({}),
        daemon=True,
        name="pool-score-warmer",
    ).start()
    if manager.serving_mode == POOL_SERVING_MODE_SUBNET:
        # Keep chain-bound meshes signed for the CURRENT scoring epoch:
        # validators exclude any snapshot bound to another epoch, and the
        # refresh is a local driver-side re-sign (no chain writes, no
        # relaunch). Validators pin snapshots in a short grace window at
        # epoch START (~10 blocks), so the poll must be tight enough that
        # boundary detection plus one worker heartbeat lands the fresh
        # signature inside that window.
        def _epoch_follow_loop() -> None:
            while True:
                try:
                    manager.follow_chain_epoch()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("epoch follower tick failed: %s", exc)
                # Tighten the poll approaching a boundary so the re-sign
                # lands inside the validators' epoch-start pin window.
                try:
                    eta = manager.seconds_to_next_epoch()
                except Exception:  # pragma: no cover - defensive
                    eta = None
                sleep_s = 60.0 if eta is None else min(60.0, max(5.0, eta - 6.0))
                time.sleep(sleep_s)

        threading.Thread(
            target=_epoch_follow_loop,
            daemon=True,
            name="pool-epoch-follower",
        ).start()
        if announce_chain_config and os.environ.get(
            "VERATHOS_POOL_NO_ANNOUNCE", ""
        ) != "1":
            threading.Thread(
                target=_coordinator_announce_loop,
                args=(manager, str(announce_chain_config)),
                daemon=True,
                name="pool-coordinator-announce",
            ).start()
    return server


#: Re-announce at least this often even when no epoch boundary is seen
#: (covers chains whose epoch read fails); matches roughly one epoch.
COORDINATOR_ANNOUNCE_FALLBACK_S = 72 * 60.0
COORDINATOR_ANNOUNCE_PATH = "/v1/operator/announce"


def _coordinator_announce_loop(manager: "PoolManager", chain_config_path: str) -> None:
    """Register this coordinator's control endpoint with validator proxies.

    Address-book only: proxies keep the endpoint PRIVATE (it is never
    served to anyone) and pull the sanitized viewer-lane operator overview
    themselves. Targets come from the on-chain ValidatorRegistry, so
    nothing is hardcoded and the same code serves every network. The
    announce is self-attested (signed with the coordinator hotkey) and
    never feeds scoring or validation; any failure here must never affect
    serving, so the whole body is defensive.

    Cadence: once at startup, then on every scoring-epoch change, with a
    ~72 min fallback when the epoch cannot be read.
    """

    last_epoch: int | None = None
    last_sent = 0.0
    warned: set[str] = set()
    tick_failures = 0
    while True:
        try:
            now = time.time()
            epoch = manager._current_chain_epoch()
            due = (
                last_sent == 0.0
                or (epoch is not None and epoch != last_epoch)
                or now - last_sent >= COORDINATOR_ANNOUNCE_FALLBACK_S
            )
            if due:
                with manager.lock:
                    endpoint = str(manager.state.get("manager_endpoint", "") or "")
                    pool_id = str(manager.state.get("pool_id", "") or "")
                    wallet_name = str(manager.state.get("wallet_name", "") or "")
                    wallet_hotkey = str(manager.state.get("wallet_hotkey", "") or "")
                    hotkey_ss58 = str(
                        manager.state.get("coordinator_hotkey_ss58", "") or ""
                    )
                network = manager._subtensor_network()
                if not (endpoint and wallet_name and wallet_hotkey and hotkey_ss58):
                    logger.debug(
                        "coordinator announce skipped: chain identity or "
                        "manager endpoint not configured yet"
                    )
                elif endpoint.startswith(("http://127.", "http://localhost")):
                    logger.debug(
                        "coordinator announce skipped: loopback manager endpoint"
                    )
                else:
                    # Heavy imports stay lazy: the module must import
                    # stdlib-only, and dev pools never reach this branch.
                    from dataclasses import replace as _dc_replace

                    from verallm.chain.config import ChainConfig

                    from verallm.chain.validator_registry import (
                        ValidatorRegistryClient,
                    )

                    config = ChainConfig.from_json(chain_config_path)
                    # Registry reads ride the pool's OWN node when one is
                    # configured; the bundled config default is the public
                    # RPC, which rate-limits and must never be a dependency.
                    own_rpc = ChainConfig.resolve_rpc_url(network, None)
                    if own_rpc:
                        config = _dc_replace(config, rpc_url=own_rpc)
                    registry = ValidatorRegistryClient(config)
                    targets = [
                        str(info.proxy_endpoint).rstrip("/")
                        for _addr, info in registry.get_proxy_validators()
                        if str(info.proxy_endpoint).startswith(("http://", "https://"))
                    ]
                    if targets:
                        body = json.dumps(
                            {
                                "hotkey_ss58": hotkey_ss58,
                                "endpoint": endpoint,
                                "pool_id": pool_id,
                                "netuid": int(config.netuid),
                                "subtensor_network": network,
                                "timestamp": int(now),
                            },
                            sort_keys=True,
                        ).encode("utf-8")
                        from neurons.request_signing import sign_request

                        from verallm.mesh.receipt_signing import load_hotkey_seed

                        headers = sign_request(
                            "POST",
                            COORDINATOR_ANNOUNCE_PATH,
                            body,
                            hotkey_ss58,
                            load_hotkey_seed(wallet_name, wallet_hotkey),
                        )
                        headers["Content-Type"] = "application/json"
                        ok = 0
                        for target in targets:
                            try:
                                request = urllib.request.Request(
                                    target + COORDINATOR_ANNOUNCE_PATH,
                                    data=body,
                                    headers=headers,
                                    method="POST",
                                )
                                with urllib.request.urlopen(
                                    request, timeout=10.0
                                ) as response:
                                    if response.status == 200:
                                        ok += 1
                                        warned.discard(target)
                            except (OSError, urllib.error.URLError) as exc:
                                if target not in warned:
                                    warned.add(target)
                                    logger.warning(
                                        "coordinator announce to %s failed: %s",
                                        target,
                                        exc,
                                    )
                        logger.debug(
                            "coordinator announce: %d/%d proxies ok", ok, len(targets)
                        )
                        # A tick where EVERY proxy failed keeps the loop due:
                        # retry on the next 60s tick instead of going dark
                        # until the epoch boundary (warned-set keeps logs calm).
                        if ok > 0:
                            last_epoch = epoch
                            last_sent = now
                    else:
                        logger.debug(
                            "coordinator announce: no proxy validators on chain"
                        )
                        last_sent = now  # retry on fallback cadence, not hot
        except Exception as exc:  # pragma: no cover - must never kill serving
            tick_failures += 1
            # Never silent: a broken announce means the operator console
            # loses this pool. Warn immediately, then once every ~30 min.
            if tick_failures == 1 or tick_failures % 30 == 0:
                logger.warning(
                    "coordinator announce tick failed (%d): %s",
                    tick_failures,
                    exc,
                )
            else:
                logger.debug("coordinator announce tick failed: %s", exc)
        else:
            tick_failures = 0
        time.sleep(60.0)


# -- worker side -----------------------------------------------------------


@dataclass
class PoolWorkerConfig:
    """Everything one pool worker needs to serve any catalog model."""

    token: MeshPoolToken
    repo_root: Path
    workdir: Path
    advertise_host: str
    rpc_port: int
    proof_port: int
    mesh_port: int  # coordinator control port when this worker drives
    # Backend binaries. Empty = AUTO: the worker detects its GPU architecture,
    # reuses a canonical patched build that matches the shipped llama.cpp
    # patch, or builds one (plug-and-play; a hand-picked path for the wrong
    # arch crash-looped a 4090 362x). Explicit paths still win when set.
    llama_server_binary: str = ""
    rpc_worker_binary: str = ""
    rpc_device: str = "CUDA0"
    catalog: list[dict[str, Any]] = field(default_factory=list)
    # File the catalog was loaded from; auto-fetched models are appended there
    # so they survive a worker restart. Empty = in-memory only.
    catalog_path: str = ""
    # Bittensor wallet/hotkey used only by this machine when it is the mesh
    # coordinator/miner. Proof-producing stages use the separate key below.
    wallet_name: str = ""
    wallet_hotkey: str = ""
    # Local validator allowlist maintained by the isolated miner/metagraph
    # refresh. Required only when this worker is selected to drive a
    # validator-served mesh; stage-only workers never receive validator calls.
    validator_allowlist_path: str = ""
    validator_allowlist_max_age_seconds: float = (
        DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS
    )
    # Optional path for this physical worker's persistent Sr25519 stage key.
    # Empty selects <workdir>/stage-proof-key.seed, created owner-only once.
    stage_proof_key_file: Path | None = None
    worker_id: str = ""
    gpu_name: str = ""
    vram_gb: int = 0
    # Per-GPU breakdown for multi-GPU workers. Empty means "one GPU worth
    # vram_gb" (legacy single-GPU workers and hand-written configs).
    gpu_names: list[str] = field(default_factory=list)
    per_gpu_vram_gb: list[int] = field(default_factory=list)
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    # A member-only worker serves pipeline stages but cannot drive (host the
    # coordinator + llama-server). Concretely: the Mac's Metal llama.cpp build
    # has a broken RPC client (recv failed in ggml_backend_rpc_add_server), so
    # driving wedges at "driving" forever while it works fine as a member.
    member_only: bool = False
    # Explicit private CA trust must survive into a wallet-free driver's
    # signing subprocess, not just the parent worker's HTTP opener.
    manager_ca_file: str = ""

    def __post_init__(self) -> None:
        if bool(self.wallet_name) != bool(self.wallet_hotkey):
            raise ValueError(
                "wallet_name and wallet_hotkey must be configured together"
            )
        max_age = float(self.validator_allowlist_max_age_seconds)
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError(
                "validator_allowlist_max_age_seconds must be positive"
            )
        self.validator_allowlist_max_age_seconds = max_age

    @property
    def subnet_driver_ready(self) -> bool:
        """Whether this worker can DRIVE a subnet-serving mesh.

        This is miner-side readiness (the mesh gets validated; it never
        validates anything). A fresh allowlist of validator hotkeys is the
        real local requirement: it is what lets the driver authenticate
        INCOMING validator calls, and nothing can supply it remotely. The
        coordinator hotkey is not a local requirement -- a worker without
        one signs through its manager
        (verallm/mesh/delegated_signing.py), which is what lets a machine
        join with only a token and still drive.
        """

        allowlist = str(self.validator_allowlist_path or "").strip()
        if not allowlist:
            return False
        try:
            hotkeys = read_fresh_validator_allowlist(
                Path(allowlist).expanduser(),
                max_file_age_seconds=self.validator_allowlist_max_age_seconds,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return False
        return bool(hotkeys)

    @property
    def gpu_count(self) -> int:
        return max(1, len(self.per_gpu_vram_gb))

    @property
    def gpu_split_weights(self) -> list[int]:
        """Per-GPU VRAM list driving the mesh tensor split for this worker."""
        if self.per_gpu_vram_gb:
            return [int(v) for v in self.per_gpu_vram_gb]
        return [int(self.vram_gb)]

    def endpoints(self) -> dict[str, str]:
        return {
            "rpc": f"{self.advertise_host}:{self.rpc_port}",
            "proof": f"http://{self.advertise_host}:{self.proof_port}",
            "mesh": f"http://{self.advertise_host}:{self.mesh_port}",
        }

    def catalog_entry(self, model_id: str) -> dict[str, Any]:
        entry = next((c for c in self.catalog if c.get("model_id") == model_id), None)
        if entry is None:
            raise ValueError(f"model not in local catalog: {model_id}")
        return entry


def _acquire_owner_manifest(
    *,
    model_id: str,
    spec: Mapping[str, Any],
    dest_dir: Path,
    progress: Any | None = None,
) -> str:
    """The ONLY sources of a tensor manifest on a fetching box.

    Manifests are owner-built; a fetching box downloads one from the
    store or accepts an owner-built file pre-staged next to the model,
    and NEVER builds one itself - a rebuild here is always a broken
    distribution path, dev pools included. Every acquired manifest is
    verified: a registered root must match, and all later proofs bind
    to it, so both sources carry identical trust.
    """

    from verallm.mesh.gguf_manifest import (
        load_gguf_tensor_manifest,
        save_gguf_tensor_manifest,
    )
    from verallm.mesh.manifest_store import (
        MeshManifestStoreError,
        configured_mesh_manifest_base_urls,
        fetch_mesh_tensor_manifest,
    )

    registered_root = str(
        spec.get("model_tensor_manifest_root", "") or ""
    ).lower()
    store_error: MeshManifestStoreError | None = None
    if registered_root:
        base_urls = configured_mesh_manifest_base_urls(
            spec.get("manifest_urls") or ()
        )
        if base_urls:
            if progress is not None:
                progress("fetching manifest")
            try:
                fetched = fetch_mesh_tensor_manifest(
                    model_id,
                    expected_tensor_manifest_root=registered_root,
                    base_urls=base_urls,
                )
                return str(
                    save_gguf_tensor_manifest(
                        load_gguf_tensor_manifest(fetched),
                        dest_dir / "tensor-manifest.json",
                    )
                )
            except MeshManifestStoreError as exc:
                store_error = exc
                logger.warning(
                    "manifest store fetch failed for %s (%s); checking for "
                    "a pre-staged owner manifest",
                    model_id,
                    exc,
                )
    staged = dest_dir / "tensor-manifest.json"
    if staged.is_file():
        staged_manifest = load_gguf_tensor_manifest(str(staged))
        staged_root = str(
            staged_manifest.get("tensor_manifest_root", "") or ""
        ).lower()
        if registered_root and staged_root != registered_root:
            raise ValueError(
                f"pre-staged tensor manifest for {model_id} has root "
                f"{staged_root[:16]}..., registered root is "
                f"{registered_root[:16]}...; refusing it"
            )
        return str(staged)
    if store_error is not None:
        raise ValueError(
            f"manifest store fetch failed for {model_id}: {store_error}; "
            "no pre-staged owner-built tensor-manifest.json is available. "
            "A fetching box never rebuilds a manifest - fix the store path "
            "or pre-stage the manifest next to the model"
        ) from store_error
    raise ValueError(
        f"no owner-built tensor manifest available for {model_id}: the "
        "store fetch is unconfigured and nothing is pre-staged. A "
        "fetching box never builds manifests - publish the manifest to "
        "the store or pre-stage the owner-built tensor-manifest.json "
        "next to the model"
    )


class LocalMeshRunner:
    """Executes drive/join/stop commands by spawning the validated serve CLIs."""

    def __init__(self, config: PoolWorkerConfig):
        self.config = config
        self.procs: list[subprocess.Popen] = []
        self.mesh_dir: Path | None = None
        self._hotkey_ss58: str = ""  # resolved lazily from the signing wallet
        from verallm.mesh.receipt_signing import ensure_stage_proof_key_file

        configured_stage_key = config.stage_proof_key_file
        self._stage_proof_key_file = Path(
            configured_stage_key or (config.workdir / "stage-proof-key.seed")
        ).expanduser().resolve()
        stage_keypair = ensure_stage_proof_key_file(self._stage_proof_key_file)
        self._stage_proof_keypair = stage_keypair
        self._stage_proof_key_ss58 = str(stage_keypair.ss58_address)
        self._prewarm_stop = threading.Event()
        self._prewarm_done = threading.Event()
        self._prewarm_result: dict[str, Any] = {}
        # Byte offsets into the serve logs, marked at each spawn attempt:
        # error scraping must never read a PREVIOUS attempt's crash lines.
        self._backend_log_scrape_offsets: dict[str, int] = {}
        self._drive_command: dict[str, Any] = {}
        self._snapshot_finalize_lock = threading.Lock()
        self._command_fence_lock = threading.Lock()
        self._active_command_id = ""
        self._thread_command = threading.local()
        # Set by ``pool_worker_loop`` once the authenticated manager channel
        # exists. Workers that hold no coordinator wallet sign through it.
        self._coordinator_sign_request: Callable[
            [str, dict[str, Any]], dict[str, Any]
        ] | None = None

    def _coordinator_keypair(
        self, *, coordinator_hotkey: str, mesh_key: str
    ) -> Any:
        """Resolve the coordinator signing identity for this driver.

        A machine that was given the coordinator wallet uses it directly.
        A machine that joined with only a token signs through its manager,
        which holds that wallet anyway for the on-chain lease renewal.
        """

        cfg = self.config
        if cfg.wallet_name and cfg.wallet_hotkey:
            from verallm.mesh.receipt_signing import load_hotkey_keypair

            return load_hotkey_keypair(cfg.wallet_name, cfg.wallet_hotkey)
        request = self._coordinator_sign_request
        if request is None:
            raise RuntimeError(
                "subnet mesh driver has neither a coordinator wallet nor a "
                "manager channel to sign through"
            )
        return delegate_keypair_from_worker_request(
            coordinator_hotkey=coordinator_hotkey,
            mesh_key=mesh_key,
            request=request,
        )

    def fence_command(self, command_id: str) -> None:
        """Invalidate every older runner generation before execution starts."""

        if not _COMMAND_ID_RE.fullmatch(str(command_id)):
            raise ValueError("runner command fence requires a valid command id")
        with self._command_fence_lock:
            self._active_command_id = str(command_id)

    def enter_command(self, command_id: str) -> None:
        """Bind the current execution thread to the active command fence."""

        command_id = str(command_id)
        with self._command_fence_lock:
            if not self._active_command_id:
                self._active_command_id = command_id
            if self._active_command_id != command_id:
                raise RuntimeError("mesh command was superseded before execution")
        self._thread_command.command_id = command_id

    def _assert_command_active(self) -> None:
        command_id = str(getattr(self._thread_command, "command_id", "") or "")
        if not command_id:
            return
        with self._command_fence_lock:
            if self._active_command_id != command_id:
                raise RuntimeError("mesh command was superseded by teardown")

    def _spawn(
        self, cmd: list[str], log_name: str, *, env: dict[str, str] | None = None
    ) -> subprocess.Popen:
        log_path = self.config.workdir / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command_id = str(
            getattr(self._thread_command, "command_id", "") or ""
        )
        # Hold the fence across process creation and registration. A stop can
        # then either invalidate us before Popen, or observe and terminate the
        # newly tracked group after Popen—never slip through between them.
        spawn_env = dict(env if env is not None else os.environ)
        # The IPA/PCS prover is the hot path of every stage proof and is
        # single-threaded unless told otherwise; a stage proof is the thing
        # that holds the mesh's serving slot, so the default matters. Left
        # overridable for operators who share a box with other work.
        spawn_env.setdefault("VERATHOS_PCS_V2_THREADS", str(_pcs_prover_threads()))
        with self._command_fence_lock:
            if command_id and self._active_command_id != command_id:
                raise RuntimeError("mesh command was superseded by teardown")
            with open(log_path, "ab") as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.config.repo_root),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    env=spawn_env,
                )
            self.procs.append(proc)
            self._record_spawned_group(proc)
        confined = _confine_serve_process_memory(proc.pid)
        if confined:
            logger.info(
                "serve child %s confined to %s (memory.high=%d)",
                log_name,
                confined,
                _serve_memory_high_bytes(),
            )
        return proc

    def _proc_registry_path(self) -> Path:
        return Path(self.config.workdir) / "runner-procs.json"

    def _record_spawned_group(self, proc: subprocess.Popen) -> None:
        """Persist the spawned process group for cross-generation reaping.

        Serve processes run in their own sessions (start_new_session), so
        a pm2 restart of the WORKER orphans them: the fresh runner then
        cannot stop the old mesh - _free_own_ports correctly refuses to
        kill listeners it did not spawn, and the stop wedges forever on
        the previous generation's llama. The
        registry proves ownership across generations.
        """

        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return
        path = self._proc_registry_path()
        try:
            records = json.loads(path.read_text())
        except (OSError, ValueError):
            records = []
        records = [r for r in records if isinstance(r, dict)]
        records.append({"pid": int(proc.pid), "pgid": int(pgid)})
        from verallm.mesh.private_files import write_owner_only_json

        try:
            write_owner_only_json(path, records)
        except OSError as exc:
            logger.debug("proc registry not persisted: %s", exc)

    def _reap_previous_generation(self) -> None:
        """Terminate serve process groups a PREVIOUS runner left behind.

        Runs once at construction, before this runner spawns anything, so
        every group in the registry is provably a leftover. Verification
        before killing: the pid must still lead its recorded process
        group (a recycled pid fails this) - only then is the whole group
        signalled.
        """

        path = self._proc_registry_path()
        try:
            records = json.loads(path.read_text())
        except (OSError, ValueError):
            # No registry — but grandchild sessions (llama under its own
            # supervisor session) never enter it, so sweep the ports anyway.
            self._reap_port_squatters()
            return
        stale: list[int] = []
        try:
            own_pgid = os.getpgid(0)
        except OSError:
            own_pgid = -1
        for record in records if isinstance(records, list) else []:
            try:
                pid = int(record.get("pid", 0))
                pgid = int(record.get("pgid", 0))
            except (TypeError, ValueError, AttributeError):
                continue
            if pid <= 0 or pgid <= 0:
                continue
            if pgid == own_pgid:
                # NEVER killpg our own group. A pm2 fork-mode restart can
                # spawn the fresh daemon into the SAME process group the
                # previous generation recorded, and killing it is silent
                # suicide -
                # redeploy died with nothing in stderr and pm2 marked the
                # worker errored, while the second restart (registry pids
                # already dead) always survived. Port-level reaping below
                # still clears any genuine leftovers.
                logger.warning(
                    "previous-generation registry names this process's own "
                    "group %d; skipping the group kill and relying on the "
                    "port sweep",
                    pgid,
                )
                continue
            try:
                if os.getpgid(pid) != pgid:
                    continue  # pid recycled; not ours anymore
            except OSError:
                # The recorded anchor pid is gone, but the GROUP can still
                # hold survivors: an OOM kill takes the serve child while
                # its llama-server lives on in the same pgid, squatting
                # the whole GPU and poisoning every later launch on the
                # box. A dead anchor therefore proves nothing; reap the
                # group whenever anything in it is still alive.
                if not _process_group_alive(pgid):
                    continue  # genuinely all gone
            stale.append(pgid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                continue
        if stale:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if not any(_process_group_alive(pgid) for pgid in stale):
                    break
                time.sleep(0.5)
            for pgid in stale:
                if _process_group_alive(pgid):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
            logger.warning(
                "reaped %d serve process group(s) left by a previous "
                "worker generation",
                len(stale),
            )
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self._reap_port_squatters()

    def _reap_port_squatters(self) -> None:
        """Kill leftover listeners on this worker's OWN configured ports.

        The group registry misses grandchildren that llama's supervisor
        starts in their own sessions: a pm2 restart then leaves a PPID-1
        llama-server holding the port and its VRAM, the new generation's
        safe-stop refuses the unowned listener, and every relaunch wedges. At CONSTRUCTION —
        before this generation spawned anything — any listener on the
        ports this workdir exclusively owns is provably a leftover.
        """

        ports = {
            int(self.config.mesh_port),
            int(self.config.mesh_port) + 1,
            int(self.config.proof_port),
            int(self.config.rpc_port),
        }
        # Tool-free discovery: `ss` is exactly as absent as lsof on plain
        # container images, so use the same /proc-based helper the ownership checks
        # use. The sweep itself stays best-effort per port; the fail-closed
        # contract lives in the safe-stop ownership checks.
        victims: set[int] = set()
        unattributed: list[int] = []
        for port in sorted(ports):
            try:
                pids, occupied = _listeners_on_port(port)
            except Exception as exc:
                logger.debug("port squatter scan failed for %d: %s", port, exc)
                continue
            if occupied and not pids:
                unattributed.append(port)
            for pid in pids:
                if pid and pid != os.getpid():
                    victims.add(pid)
        if unattributed:
            logger.warning(
                "leftover listener(s) on port(s) %s could not be attributed "
                "to a pid; leaving them alone (safe-stop fails closed on "
                "them)",
                unattributed,
            )
        if not victims:
            return
        for pid in victims:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if not any(_pid_alive(pid) for pid in victims):
                break
            time.sleep(0.5)
        for pid in victims:
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        logger.warning(
            "reaped %d leftover listener(s) on this worker's ports %s",
            len(victims),
            sorted(ports),
        )

    def _coordinator_hotkey(self) -> str:
        """The SS58 to stamp on the mesh spec (== receipt.hotkey). In miner mode
        it must be the signing hotkey's address so a receipt's signature
        verifies against it; otherwise the operator placeholder.

        A configured wallet that fails to load RAISES instead of silently
        falling back: the spawned coordinator would crash on the same wallet
        anyway (cmd_serve loads it unguarded), turning a misconfiguration
        into a 900s opaque readiness hang — and even if it didn't, a spec
        stamped with the placeholder while receipts carry a real signature
        would fail validator identity checks.
        """
        from verallm.mesh.receipt_signing import PLACEHOLDER_HOTKEY, load_hotkey_keypair

        if self.config.wallet_name and self.config.wallet_hotkey:
            if not getattr(self, "_hotkey_ss58", ""):
                try:
                    self._hotkey_ss58 = load_hotkey_keypair(
                        self.config.wallet_name, self.config.wallet_hotkey
                    ).ss58_address
                except Exception as exc:
                    raise RuntimeError(
                        f"signing wallet {self.config.wallet_name}/"
                        f"{self.config.wallet_hotkey} failed to load: {exc}"
                    ) from exc
            return self._hotkey_ss58
        # No local wallet: on a subnet mesh the manager names the hotkey it
        # will sign as, and the spec must carry exactly that SS58 or every
        # receipt signature fails its identity check. Only a mesh with no
        # coordinator identity at all falls back to the placeholder.
        delegated = str(self._drive_command.get("coordinator_hotkey", "") or "")
        return delegated or PLACEHOLDER_HOTKEY

    def _wallet_flags(self) -> list[str]:
        """Pass the miner identity only to a coordinator serve process."""
        if self.config.wallet_name and self.config.wallet_hotkey:
            return [
                "--wallet-name", self.config.wallet_name,
                "--wallet-hotkey", self.config.wallet_hotkey,
            ]
        return []

    def _coordinator_signing_flags(
        self, mesh_dir: Path, command: Mapping[str, Any]
    ) -> list[str]:
        """Delegation channel for a driver that holds no coordinator wallet.

        The pool manager keeps the coordinator wallet (it is required for
        subnet serving); this only hands the spawned serve process the
        material to request signatures FROM it over the authenticated worker
        channel. Written owner-only, never argv: a pool secret on a command
        line is readable by every local process.
        """

        if self.config.wallet_name and self.config.wallet_hotkey:
            return []
        context = dict(getattr(self, "_delegation_context", {}) or {})
        if not context:
            return []
        coordinator_hotkey = str(command.get("coordinator_hotkey", "") or "")
        if not coordinator_hotkey:
            return []
        from verallm.mesh.private_files import write_owner_only_json

        path = mesh_dir / "coordinator-signing.json"
        write_owner_only_json(
            path,
            {
                **context,
                "mesh_key": str(command.get("mesh_key", "") or ""),
                "coordinator_hotkey": coordinator_hotkey,
                "evm_address": _normalize_evm_address(
                    command.get("coordinator_address")
                ),
                "stage_proof_key_file": str(self._stage_proof_key_file),
            },
        )
        return ["--coordinator-sign-file", str(path)]

    def _validator_coordinator_flags(
        self,
        command: Mapping[str, Any],
    ) -> list[str]:
        cfg = self.config
        return [
            "--server-role",
            "coordinator",
            "--validator-auth",
            "--require-validator-nonce",
            "--validator-allowlist-path",
            str(cfg.validator_allowlist_path),
            "--validator-allowlist-max-age-seconds",
            f"{cfg.validator_allowlist_max_age_seconds:g}",
            "--require-verification-snapshot",
            "--evm-address",
            _normalize_evm_address(command.get("coordinator_address")),
        ]

    def _worker_serve_cmd(
        self,
        mesh_dir: Path,
        entry: dict[str, Any],
        *,
        rpc_worker: bool = True,
    ) -> list[str]:
        return [
            sys.executable, "-m", "neurons.cli", "mesh", "serve",
            "--mesh", str(mesh_dir),
            # Bind all interfaces, never the advertise host: behind NAT or a
            # port-mapping fabric the advertised public address is not a
            # local interface, and binding it fails the whole serve process.
            # Remote parties still dial the ADVERTISED endpoints (set at
            # join_mesh time), which keep the advertise host.
            "--host", "0.0.0.0",
            "--port", str(self.config.proof_port),
            # Local-stage member: the driver's llama-server computes this
            # stage on local devices and writes into this trace dir, so no
            # rpc-server runs; this process is the proof server + capture
            # arming owner only.
            *(
                [
                    "--rpc-worker",
                    "--rpc-worker-binary", self.config.rpc_worker_binary,
                    "--rpc-host", "0.0.0.0",
                    "--rpc-port", str(self.config.rpc_port),
                    "--rpc-device", self.config.rpc_device,
                ]
                if rpc_worker
                else []
            ),
            "--proof-trace-dir",
            str(model_trace_dir(self.config.workdir, str(entry.get("model_id", "")))),
            *PROOF_FLAGS,
            "--proof-gguf-manifest", str(entry["manifest"]),
            "--stage-proof-key-file", str(self._stage_proof_key_file),
            "--server-role", "worker",
        ]

    def _free_own_ports(self, *, driving: bool) -> None:
        """Clear tracked children, then reclaim the runner's own ports.

        These ports belong EXCLUSIVELY to this workdir's configuration, so
        any listener still holding one after the tracked children are gone
        is this runner's own descendant that escaped the process group —
        llama-server runs in its own session under the coordinator's
        supervisor, so a coordinator that dies hard (SIGKILL after a stop
        timeout, a crash) orphans it while it keeps the port and the VRAM.
        Refusing to kill it wedges stop/relaunch into a permanent error until
        a daemon restart sweeps it. The same owned-by-definition invariant as
        the construction-time sweep applies at stop time.
        """
        self._terminate_tracked_processes()
        cfg = self.config
        ports = {cfg.rpc_port, cfg.proof_port}
        if driving:
            ports |= {cfg.mesh_port, cfg.mesh_port + 1}
        # Grace-poll first: the children were JUST signalled, and a dying
        # llama's listen socket stays visible for a moment. Only listeners
        # that PERSIST past the grace window get escalated.
        deadline = time.monotonic() + 15.0
        for port in sorted(ports):
            while True:
                pids, occupied = _listeners_on_port(port)
                if not pids and not occupied:
                    break
                if time.monotonic() < deadline:
                    time.sleep(0.5)
                    continue
                if pids:
                    # Fence: the stop must still be the active command; a
                    # future dispatcher change that allows a launch beside
                    # a stop must not let this SIGKILL a fresh llama.
                    fence = getattr(self, "_assert_command_active", None)
                    if callable(fence):
                        fence()
                    rendered = ", ".join(str(pid) for pid in sorted(pids))
                    logger.warning(
                        "reclaiming own port tcp:%s from escaped "
                        "descendant(s): pid %s",
                        port,
                        rendered,
                    )
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except OSError:
                            pass
                    kill_deadline = time.monotonic() + 10.0
                    while time.monotonic() < kill_deadline:
                        remaining, _ = _listeners_on_port(port)
                        if not remaining:
                            break
                        time.sleep(0.5)
                    remaining, _ = _listeners_on_port(port)
                    if remaining:
                        raise RuntimeError(
                            f"could not reclaim tcp:{port}: pid "
                            f"{sorted(remaining)} survived SIGKILL"
                        )
                    break
                raise RuntimeError(
                    f"refusing to free tcp:{port}: a listener exists but its "
                    "owning process could not be identified; stop the owning "
                    "service explicitly"
                )

    def _terminate_tracked_processes(self) -> None:
        """Kill only process groups spawned by this runner."""

        tracked = list(self.procs)
        for proc in tracked:
            # Once the tracked session leader has been reaped, its numeric PID
            # may be reused by an unrelated process group. Do not guess. Any
            # surviving listener is detected below and refused fail-closed.
            if proc.poll() is not None:
                continue
            try:
                # Each serve runs in its own session (start_new_session), so
                # killing the group also reaps its llama-server / rpc-server
                # children instead of orphaning them on the GPU.
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        for proc in tracked:
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=5.0)
            except Exception:
                # The subsequent listener scan is the fail-closed authority.
                # Do not guess at any PID after the bounded reap attempt.
                pass
        self.procs.clear()

    def _fetch_manifest_entry(
        self,
        model_id: str,
        coordinator: str,
        expected_root: str,
        expected_package_hash: str,
        *,
        internal_auth_secret: str,
    ) -> dict[str, Any]:
        """Fetch the tensor manifest from the coordinator for a file-less join.

        The manifest is the only local artifact a member needs (its rpc worker
        receives layer slices over RPC; its proof sidecar fetches weight blobs
        on demand, sha-verified). Integrity: the fetched manifest's Merkle root
        must equal the root committed in the mesh spec we joined against — a
        coordinator cannot swap in a different model's manifest.

        When the coordinator route is unavailable, the content-addressed
        manifest store (GLEIPNIR-style base URLs) is the fallback source;
        the same committed-root check gates it.
        """
        from verallm.mesh.gguf_manifest import (
            load_gguf_tensor_manifest,
            save_gguf_tensor_manifest,
            strip_gguf_manifest_runtime_paths,
        )

        from verallm.mesh.http_auth import sign_internal_http_request
        from verallm.mesh.model_spec import gguf_package_hash

        def _fetch_from_store(reason: str) -> dict[str, Any]:
            from verallm.mesh.manifest_store import (
                MeshManifestStoreError,
                configured_mesh_manifest_base_urls,
                fetch_mesh_tensor_manifest,
            )

            base_urls = configured_mesh_manifest_base_urls()
            if not base_urls:
                raise ValueError(
                    f"cannot fetch the tensor manifest for {model_id}: "
                    f"{reason}, and no mesh manifest base URLs are "
                    "configured (VERATHOS_MESH_MANIFEST_BASE_URLS)"
                )
            if not expected_root:
                raise ValueError(
                    "store manifest fetch requires the mesh spec's "
                    "committed tensor manifest root"
                )
            try:
                fetched = fetch_mesh_tensor_manifest(
                    model_id,
                    expected_tensor_manifest_root=expected_root,
                    expected_package_hash=expected_package_hash,
                    base_urls=base_urls,
                )
            except MeshManifestStoreError as exc:
                raise ValueError(
                    f"manifest store fetch failed for {model_id}: {exc}"
                ) from exc
            dest = (
                self.config.workdir
                / "manifests"
                / f"{model_id}.tensor-manifest.json"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_gguf_tensor_manifest(
                strip_gguf_manifest_runtime_paths(
                    load_gguf_tensor_manifest(fetched)
                ),
                dest,
            )
            return {"model_id": model_id, "manifest": str(dest)}

        if not coordinator:
            return _fetch_from_store(
                "model not in local catalog and no coordinator endpoint"
            )

        manifest_path = "/v1/mesh/manifest"
        req = urllib.request.Request(
            coordinator.rstrip("/") + manifest_path,
            method="GET",
            headers=sign_internal_http_request(
                secret=internal_auth_secret,
                method="GET",
                path=manifest_path,
                body=b"",
            ),
        )
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                raw = resp.read()
        except (OSError, urllib.error.URLError) as exc:
            return _fetch_from_store(f"coordinator manifest route failed ({exc})")
        dest = self.config.workdir / "manifests" / f"{model_id}.tensor-manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        manifest = load_gguf_tensor_manifest(dest)
        got_root = str(manifest.get("tensor_manifest_root", ""))
        if expected_root and got_root != expected_root:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"coordinator manifest root {got_root[:16]}… does not match the "
                f"mesh spec's committed root {expected_root[:16]}...; refusing to join"
            )
        got_package_hash = gguf_package_hash(manifest).hex()
        if expected_package_hash and got_package_hash != expected_package_hash:
            dest.unlink(missing_ok=True)
            raise ValueError(
                "coordinator manifest package hash does not match the mesh spec; "
                "refusing to join"
            )
        # Defense in depth: the coordinator route already omits its local
        # paths, but never persist or pass any uncommitted locator received
        # from the network. A file-less member must use verified proof blobs.
        save_gguf_tensor_manifest(
            strip_gguf_manifest_runtime_paths(manifest),
            dest,
        )
        return {"model_id": model_id, "manifest": str(dest)}

    @staticmethod
    def _ready_timeout_s() -> float:
        """How long a drive/join waits for the backend to answer 200.

        Cold-loading is dominated by moving the members' layer shares over
        RPC: a 22GB MoE whose member sits on WiFi (~30MB/s) legitimately
        takes ~12min before first 200, so 900s failed HEALTHY meshes. Tune
        via VERATHOS_MESH_READY_TIMEOUT_S when a topology needs more.
        """
        return float(os.environ.get("VERATHOS_MESH_READY_TIMEOUT_S", "1800"))

    def _await_proof_cache_prewarm(self) -> None:
        """Require the driver cache warm to finish and converge before serve."""

        timeout = self._ready_timeout_s()
        if not self._prewarm_done.wait(timeout):
            raise RuntimeError(
                "proof-weight cache did not finish warming within "
                f"{timeout:.0f}s"
            )
        error = str(self._prewarm_result.get("error", "") or "").strip()
        if error:
            raise RuntimeError(f"proof-weight cache prewarm failed: {error}")
        if not bool(self._prewarm_result.get("converged")):
            raise RuntimeError(
                "proof-weight cache prewarm completed without a convergence "
                "certificate"
            )

    def _preflight_backend(self) -> None:
        """Fail fast, with the real reason, before a mesh commits.

        Catches the whole class the 4090 hit (a binary built for another
        GPU generation / missing shared libs) at launch time instead of a
        362x crash-loop behind a healthy-looking 'driving' status: the
        binary must exist, execute (--version), and — when it carries our
        build manifest — match this machine's detected GPU arch.
        """
        cfg = self.config
        for label, path_str in (
            ("llama-server", cfg.llama_server_binary),
            ("rpc-worker", cfg.rpc_worker_binary),
        ):
            path = Path(path_str)
            if not path.is_file():
                raise RuntimeError(f"{label} binary missing: {path}")
            if label == "llama-server":
                try:
                    rc = subprocess.run(
                        [str(path), "--version"], capture_output=True, timeout=20
                    ).returncode
                except Exception as exc:
                    raise RuntimeError(f"{label} does not execute: {exc}")
                if rc != 0:
                    raise RuntimeError(
                        f"{label} fails to start (--version exit {rc}); "
                        "likely missing shared libraries — rebuild or clear "
                        f"{path.parent}"
                    )
            manifest = path.parent / "verathos-build.json"
            if manifest.is_file():
                try:
                    built = json.loads(manifest.read_text())
                except Exception:
                    continue
                backend, arch = _detect_backend_arch()
                if (
                    built.get("arch")
                    and arch
                    and built.get("backend") == backend
                    and built["arch"] != arch
                ):
                    raise RuntimeError(
                        f"{label} was built for {built['arch']} but this "
                        f"GPU is {arch}; remove --llama-server-binary/"
                        "--rpc-worker-binary to let the worker auto-build "
                        "the matching backend"
                    )

    def _mark_backend_log_offsets(self) -> None:
        """Record where the serve logs end RIGHT NOW.

        Called before (re)spawning a backend so ``_scrape_backend_error``
        only ever reports the current attempt. The serve logs are opened
        in append mode across KV-ladder rungs; without the mark, a
        previous rung's cudaMalloc line in the shared tail made a later
        rung's transport hiccup look like an allocation death - the
        probe raised instantly instead of waiting out a loading llama,
        and the ladder mis-descended.
        """

        for name in ("pool-driver-worker.log", "pool-driver-coordinator.log"):
            path = self.config.workdir / name
            try:
                self._backend_log_scrape_offsets[name] = path.stat().st_size
            except OSError:
                self._backend_log_scrape_offsets[name] = 0

    def _scrape_backend_error(self) -> str:
        """Pull the first meaningful crash lines out of the spawned serve logs.

        The operator should see "GGML_ASSERT(ok) failed at ggml-cuda.cu:4746"
        in the dashboard, not a bare ready-timeout: the reason is sitting in
        the logs this worker itself created. Only reads past the offsets
        marked at the current attempt's spawn (see
        ``_mark_backend_log_offsets``); earlier attempts' crashes are not
        this attempt's error.
        """
        signatures = re.compile(
            r"GGML_ASSERT|CUDA error|Remote RPC server crashed|out of memory"
            r"|ggml_abort|Segmentation fault|refusing to run",
            re.IGNORECASE,
        )
        found: list[str] = []
        for name in ("pool-driver-worker.log", "pool-driver-coordinator.log"):
            path = self.config.workdir / name
            offset = self._backend_log_scrape_offsets.get(name, 0)
            try:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    text = handle.read().decode(errors="replace")
            except OSError:
                continue
            for line in text.splitlines()[-200:]:
                if signatures.search(line) and line.strip() not in found:
                    found.append(line.strip()[:180])
        return "; ".join(found[-2:]) if found else ""

    def _end_kv_ladder_attempt(self) -> None:
        """Fully end a failed KV-fit attempt before retrying OR raising.

        Ends the whole attempt, not just reaps the dead: a probe failure
        leaves the coordinator serve process ALIVE and supervising a
        crash-looping llama, and two coordinators cannot share the mesh
        port. Escalates to SIGKILL because a llama wedged in CUDA
        teardown ignores SIGTERM - and while it lives it keeps its VRAM,
        making the next (smaller) rung OOM below the machine's true fit,
        and keeps its port, which the worker's safe-stop then refuses to
        kill as an unowned listener (both).
        """

        for proc in self.procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        for proc in self.procs:
            try:
                proc.wait(timeout=20)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:
                    pass
        self.procs = [p for p in self.procs if p.poll() is None]

    def _llama_first_batch_probe(
        self, llama_port: int, model_bytes: int = 0
    ) -> None:
        """Force the prefill compute-buffer allocation at full micro-batch.

        The health check proves weights + KV allocated; it does NOT prove
        the budget survives serving. llama.cpp allocates its prompt
        processing buffers lazily at the FIRST request, sized by ubatch,
        on top of everything else -- observed: an estimated 389k-token
        budget passed health, then the first batch asked for a further
        33GB on one device and crash-looped the mesh. Called inside the
        KV auto-fit ladder so an allocation death descends to the next
        rung instead of shipping a budget that dies on request one.
        """

        # Longer than the 4096-token ubatch so the full prefill graph
        # shape (and its worst-case compute buffer) is exercised.
        body = json.dumps(
            {
                "prompt": " a" * 5120,
                "n_predict": 1,
                "temperature": 0,
                "cache_prompt": False,
            }
        ).encode()
        first_batch_budget_s = _first_batch_deadline_s(model_bytes)
        deadline = time.monotonic() + first_batch_budget_s
        last = ""
        while time.monotonic() < deadline:
            self._assert_command_active()
            request = urllib.request.Request(
                f"http://127.0.0.1:{llama_port}/completion",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180.0) as resp:
                    resp.read()
                return
            except Exception as exc:
                last = str(exc)
            detail = self._scrape_backend_error()
            if detail and _KV_ALLOC_FAILURE_RE.search(detail):
                # The backend died allocating: no amount of waiting brings
                # it back at this budget. Hand the ladder its descend signal.
                raise RuntimeError(
                    f"first-batch probe failed: {last}; {detail}"
                )
            time.sleep(2.0)
        raise RuntimeError(
            "first-batch probe did not complete within its deadline: "
            f"{last}; {self._scrape_backend_error()}"
        )

    def _kv_fit_path(self, command: Mapping[str, Any]) -> Path:
        model_id = str(command.get("model_id", "") or "model")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id)
        return Path(self.config.workdir) / f"kv-fit-{safe}.json"

    def _cached_kv_fit(self, command: Mapping[str, Any]) -> tuple[int, bool]:
        """(measured unified-KV budget, was it DESCENT-validated) or (0, False).

        A descent-validated value is a real ceiling: the ladder probed
        above it and failed. A first-try value merely means "this
        estimate happened to work" - and the estimate can be poisoned by
        transiently-occupied VRAM (a dying previous serve still holding
        the model produced a bogus 8192 cache that then nearly became an
        on-chain registration).
        """

        try:
            payload = json.loads(self._kv_fit_path(command).read_text())
            # A fit measured without today's audit reserve packed the card
            # wall to wall; trusting it would OOM every audit workload.
            # Force a fresh reserve-aware measurement instead.
            if (
                int(payload.get("audit_reserve_mb", 0) or 0)
                != _effective_audit_reserve_mb()
            ):
                return 0, False
            return (
                max(0, int(payload.get("ctx_budget", 0) or 0)),
                bool(payload.get("descended", True)),
            )
        except (OSError, ValueError, TypeError):
            return 0, False

    def _store_kv_fit(
        self,
        command: Mapping[str, Any],
        ctx_budget: int,
        *,
        descended: bool,
    ) -> None:
        try:
            self._kv_fit_path(command).write_text(
                json.dumps(
                    {
                        "ctx_budget": int(ctx_budget),
                        "descended": bool(descended),
                        # The reserve REGIME the measurement ran under: a
                        # fit measured with holders resident (512MB margin)
                        # must not be trusted after a restart where holders
                        # failed to spawn.
                        "audit_reserve_mb": _effective_audit_reserve_mb(),
                    }
                )
            )
        except OSError as exc:
            logger.debug("kv fit not cached: %s", exc)

    def _wait_http(
        self,
        url: str,
        *,
        timeout: float = 900.0,
        ready_on_200: bool = False,
        watch_procs: bool = True,
        failure_file: Path | None = None,
    ) -> None:
        """Poll GET ``url`` until it is ready, then return.

        ``ready_on_200=False`` (control endpoints): return as soon as the server
        is *reachable* — any HTTP response means the process is up, which is all
        the coordinator/proof endpoints need.

        ``ready_on_200=True`` (the llama backend): return ONLY on HTTP 200.
        llama.cpp answers ``503 {"error":"Loading model"}`` for the entire cold
        load, so "reachable" is emphatically NOT "ready". The old code POSTed
        (llama /health is GET-only -> 404) and returned on any "HTTP ..." error,
        so it fired the instant the port bound — that is exactly what made a
        mesh report "serving" while the model was still loading and every chat
        came back 503.
        """
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            self._assert_command_active()
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200 or not ready_on_200:
                        return
                    last = f"HTTP {resp.status}"
            except urllib.error.HTTPError as exc:
                if not ready_on_200:
                    return  # reachable == ready for control endpoints
                last = f"HTTP {exc.code}"  # 503 "Loading model" -> keep waiting
            except Exception as exc:
                last = str(exc)
            if failure_file is not None and failure_file.is_file():
                try:
                    failure = json.loads(failure_file.read_text())
                except Exception:
                    failure = {}
                detail = str(failure.get("error", "") or "").strip()
                if not detail:
                    detail = self._scrape_backend_error()
                crashes = int(failure.get("crashes", 0) or 0)
                component = failure.get("component", "?")
                if crashes > 0:
                    message = (
                        f"backend {component} crash-looped {crashes}x "
                        f"(exit {failure.get('exit_code')})"
                    )
                else:
                    message = f"backend {component} failed"
                raise RuntimeError(
                    message + (f": {detail}" if detail else "; see serve logs")
                )
            if watch_procs:
                for proc in self.procs:
                    code = proc.poll()
                    if code is not None:
                        # The process we are waiting on is DEAD: waiting out
                        # the full timeout turns an instant, diagnosable crash
                        # (missing native extension, bad flag, OOM at load)
                        # into a 15-minute opaque hang. Fail now and point at
                        # the log.
                        raise RuntimeError(
                            f"mesh process exited with code {code} while waiting "
                            f"for {url}; check the serve logs in {self.config.workdir}"
                        )
            time.sleep(1.0)
        raise RuntimeError(f"endpoint not ready within {timeout:.0f}s: {url}: {last}")

    def _wait_tcp(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Wait for the owned RPC listener without consuming its accept queue."""

        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            self._assert_command_active()
            for proc in self.procs:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"mesh process exited with code {code} while waiting "
                        f"for tcp://{host}:{port}; check the serve logs in "
                        f"{self.config.workdir}"
                    )
            if host in {"127.0.0.1", "::1", "localhost"} and _proc_net_available():
                pids, occupied = _listeners_on_port(port)
                if occupied:
                    roots = {int(proc.pid) for proc in self.procs}
                    if pids and all(_pid_descends_from(pid, roots) for pid in pids):
                        return
                    raise RuntimeError(f"RPC listener on tcp:{port} is not owned by this launch")
                last = "owned RPC listener not bound yet"
                time.sleep(0.2)
                continue
            try:
                with socket.create_connection((host, int(port)), timeout=2.0):
                    return
            except OSError as exc:
                last = str(exc)
            for proc in self.procs:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"mesh process exited with code {code} while waiting "
                        f"for tcp://{host}:{port}; check the serve logs in "
                        f"{self.config.workdir}"
                    )
            time.sleep(0.2)
        raise RuntimeError(
            f"endpoint not ready within {timeout:.0f}s: "
            f"tcp://{host}:{port}: {last}"
        )

    def runtime_failure(self) -> str:
        """Return a bounded child-process failure for manager heartbeats."""

        failures = []
        for proc in self.procs:
            code = proc.poll()
            if code is not None:
                failures.append(f"pid {proc.pid} exited with code {code}")
        return "; ".join(failures)[:500]

    def _finalize_verification_snapshot(self) -> Any | None:
        with self._snapshot_finalize_lock:
            return self._finalize_verification_snapshot_locked()

    def _finalize_verification_snapshot_locked(self) -> Any | None:
        """Sign the public snapshot after the expected stage set is immutable."""

        self._assert_command_active()
        command = dict(self._drive_command)
        mode = _normalize_pool_serving_mode(command.get("serving_mode"))
        if mode == POOL_SERVING_MODE_DEV:
            return None
        if command.get("model_index") is None:
            # Unregistered subnet model: no chain slot means no coordinator
            # identity to sign, and nothing for a validator to verify yet.
            return None
        if self.mesh_dir is None:
            raise RuntimeError("driver mesh state is unavailable")
        # A later stop/drive may replace ``self.mesh_dir`` after fencing this
        # command.  Keep every read/write in this finalization bound to the
        # directory that belonged to the active drive at entry.
        mesh_dir = Path(self.mesh_dir)
        cfg = self.config
        if not cfg.subnet_driver_ready:
            raise RuntimeError(
                "subnet mesh driver requires a fresh, non-empty "
                "validator allowlist"
            )

        from verallm.chain.wallet import derive_evm_address
        from verallm.mesh.llama_cpp import rpc_plan_from_mesh
        from verallm.mesh.private_files import write_owner_only_json
        from verallm.mesh.receipt_signing import load_hotkey_keypair, load_hotkey_seed
        from verallm.mesh.state import (
            load_mesh_state,
            save_mesh_state,
            state_internal_auth_secret,
            state_mesh_spec,
        )
        from verallm.mesh.verification_snapshot import (
            MeshCoordinatorIdentity,
            MeshVerificationPolicy,
            MeshVerificationSnapshot,
            build_mesh_verification_snapshot,
            derive_mesh_verification_stage_bindings,
            sign_mesh_verification_snapshot,
            verify_mesh_verification_snapshot_signature,
        )

        expected_stage_count = int(command.get("member_count", 0) or 0)
        if expected_stage_count <= 0:
            raise RuntimeError("subnet mesh has no expected compute stages")
        deadline = time.monotonic() + self._ready_timeout_s()
        state: dict[str, Any] = {}
        spec = None
        while time.monotonic() < deadline:
            self._assert_command_active()
            state = load_mesh_state(mesh_dir)
            spec = state_mesh_spec(state)
            compute_members = [
                member
                for member in spec.members
                if member.layers.end > member.layers.start
            ]
            if len(compute_members) == expected_stage_count:
                break
            if len(compute_members) > expected_stage_count:
                raise RuntimeError(
                    "mesh admitted more compute stages than the pool assignment"
                )
            time.sleep(0.1)
        else:
            raise RuntimeError("final mesh member assignment did not arrive before timeout")
        self._assert_command_active()
        assert spec is not None
        coordinator_members = [
            member for member in spec.members if member.role == "coordinator"
        ]
        if (
            len(coordinator_members) != 1
            or coordinator_members[0].layers.end
            != coordinator_members[0].layers.start
        ):
            raise RuntimeError(
                "subnet mesh coordinator must be orchestration-only"
            )
        # LOCAL STAGE: exactly one compute member with no rpc endpoint means
        # the coordinator's llama-server computes that stage on local devices
        # (zero RPC on one box). The verification snapshot is endpoint-free
        # (stage keys + layer ranges + model anchors), so nothing a validator
        # verifies changes with the transport. Any multi-stage mesh, and any
        # mixed shape (some stages rpc, some not), keeps the strict all-RPC
        # requirement below.
        local_stage_mesh = (
            len(compute_members) == 1
            and not compute_members[0].rpc_endpoint
        )
        if not local_stage_mesh:
            if any(
                not member.rpc_endpoint or member.rpc_split_weight <= 0
                for member in compute_members
            ):
                raise RuntimeError(
                    "validator mesh compute stages require committed RPC endpoints "
                    "and positive split weights"
                )
            try:
                rpc_plan = rpc_plan_from_mesh(spec)
            except ValueError as exc:
                raise RuntimeError(
                    f"validator mesh RPC placement is invalid: {exc}"
                ) from exc
            if len(rpc_plan.rpc_endpoints) != expected_stage_count:
                raise RuntimeError(
                    "validator mesh RPC plan does not cover every compute stage"
                )

        binding = _validator_binding(**dict(command.get("validator_binding") or {}))
        if spec.coordinator_uid != binding["coordinator_uid"]:
            raise RuntimeError("runtime coordinator UID does not match pool chain binding")
        if spec.epoch != binding["epoch"]:
            raise RuntimeError("runtime epoch does not match pool chain binding")
        expected_model = {
            "model_package_hash": _require_digest(
                command.get("model_package_hash"),
                field_name="model_package_hash",
            ),
            "model_tensor_manifest_root": _require_digest(
                command.get("model_tensor_manifest_root"),
                field_name="model_tensor_manifest_root",
            ),
            "tokenizer_hash": _require_digest(
                command.get("tokenizer_hash"),
                field_name="tokenizer_hash",
            ),
            "quantization_scheme": str(command.get("quantization_scheme", "")),
            "total_layers": int(command.get("total_layers", 0) or 0),
            "max_context_len": _require_max_context_len(
                command.get("max_context_len")
            ),
        }
        for field_name, expected in expected_model.items():
            if getattr(spec, field_name) != expected:
                raise RuntimeError(
                    f"runtime {field_name} does not match pool model binding"
                )

        declared_address = _normalize_evm_address(command.get("coordinator_address"))
        keypair = self._coordinator_keypair(
            coordinator_hotkey=spec.coordinator_hotkey,
            mesh_key=str(command.get("mesh_key", "") or ""),
        )
        if str(keypair.ss58_address) != spec.coordinator_hotkey:
            raise RuntimeError("coordinator wallet does not match runtime mesh hotkey")
        if cfg.wallet_name and cfg.wallet_hotkey:
            # Local wallet: cross-check that the same hotkey also produces the
            # EVM identity the pool bound on chain. With delegated signing the
            # manager holds that wallet and is the authority for both halves,
            # so there is nothing for the driver to cross-check here.
            hotkey_seed = load_hotkey_seed(
                cfg.wallet_name,
                cfg.wallet_hotkey,
                keypair=keypair,
            )
            evm_address = derive_evm_address(hotkey_seed).lower()
            if evm_address != declared_address:
                raise RuntimeError(
                    "coordinator wallet EVM identity does not match pool chain binding"
                )

        issued_at = int(time.time())
        from verallm.mesh.verification_snapshot import (
            MESH_POSTCOMMIT_HARD_AUDIT_BPS,
        )

        policy = MeshVerificationPolicy(
            profile="gguf_mesh_v1",
            trace_manifest_format=spec.proof_trace_manifest_format,
            base_proof_sample_bps=10_000,
            # Identical on both request paths so the rate cannot identify a
            # canary at phase one. ZERO by protocol:
            # the light tier carries nothing that needs serve capture -
            # decode verification lives in hard draws and forced hard
            # canaries, which prove the LM head outright. A non-zero rate
            # forced a teacher-forced replay on EVERY inline light turn
            # (tail capture is structurally unavailable at --parallel > 1),
            # costing ~0.4s on a 27b turn and ~1s on glm for zero added
            # coverage beyond what hard draws already give.
            organic_decode_sample_bps=0,
            canary_decode_sample_bps=0,
            proof_ops_per_request=1,
            proof_trace_candidates_per_request=1_024,
            deferred_proof_enabled=False,
            postcommit_hard_audit_bps=MESH_POSTCOMMIT_HARD_AUDIT_BPS,
        )
        coordinator_identity = MeshCoordinatorIdentity(
            chain_id=binding["chain_id"],
            netuid=binding["netuid"],
            coordinator_uid=binding["coordinator_uid"],
            coordinator_hotkey=spec.coordinator_hotkey,
            coordinator_evm_address=declared_address,
            model_index=int(command.get("model_index")),
        )
        stage_bindings = derive_mesh_verification_stage_bindings(
            spec,
            internal_auth_secret=state_internal_auth_secret(state),
        )
        generation = int(command.get("snapshot_generation"))
        snapshot_path = mesh_dir / "verification_snapshot.json"
        if state.get("mesh_finalized"):
            if not snapshot_path.is_file():
                raise RuntimeError("finalized mesh has no verification snapshot")
            existing = MeshVerificationSnapshot.from_dict(
                json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            # The chain position (generation/epoch) is anchored by the
            # persisted snapshot itself, not by this command: a resumed or
            # rotated chain is legitimately ahead of the launch command's
            # counter, and re-deriving from the command made every re-drive
            # after a rotation fail this equality.
            expected = build_mesh_verification_snapshot(
                spec,
                coordinator=coordinator_identity,
                policy=policy,
                generation=int(existing.generation),
                epoch=int(existing.epoch),
                issued_at_unix=existing.issued_at_unix,
                expires_at_unix=existing.expires_at_unix,
                stage_bindings=stage_bindings,
            )
            if (
                existing.to_dict(include_signature=False)
                != expected.to_dict(include_signature=False)
            ):
                raise RuntimeError(
                    "persisted verification snapshot does not match the final mesh"
                )
            if not verify_mesh_verification_snapshot_signature(
                existing,
                expected_hotkey=spec.coordinator_hotkey,
                expected_epoch=int(existing.epoch),
                expected_generation=int(existing.generation),
                expected_coordinator=coordinator_identity,
                expected_model=expected.model,
                expected_policy=policy,
                now_unix=int(time.time()),
            ):
                raise RuntimeError("persisted verification snapshot is invalid")
            if state.get("verification_snapshot_hash") != existing.snapshot_hash_hex():
                raise RuntimeError("persisted verification snapshot hash mismatch")
            self._assert_command_active()
            return existing

        signed = resume_mesh_verification_snapshot_chain(
            snapshot_path,
            spec=spec,
            coordinator_identity=coordinator_identity,
            policy=policy,
            stage_bindings=stage_bindings,
            epoch=binding["epoch"],
            keypair=keypair,
        )
        if signed is None:
            snapshot = build_mesh_verification_snapshot(
                spec,
                coordinator=coordinator_identity,
                policy=policy,
                generation=generation,
                epoch=binding["epoch"],
                issued_at_unix=issued_at,
                expires_at_unix=issued_at + binding["snapshot_ttl_seconds"],
                stage_bindings=stage_bindings,
            )
            signed = sign_mesh_verification_snapshot(snapshot, keypair)
        self._assert_command_active()

        # Re-read immediately before publication. The admission handler caps the
        # compute-stage count, and this hash guard also catches any reassignment
        # racing snapshot construction.
        latest = load_mesh_state(mesh_dir)
        latest_spec = state_mesh_spec(latest)
        if latest_spec.spec_hash_hex() != spec.spec_hash_hex():
            raise RuntimeError("mesh assignment changed while snapshot was being signed")
        # Serialize publication with ``fence_command``.  Either the old drive
        # publishes both files before teardown wins the fence, or teardown wins
        # and this command publishes nothing after it has been superseded.
        command_id = str(
            getattr(self._thread_command, "command_id", "") or ""
        )
        with self._command_fence_lock:
            if command_id and self._active_command_id != command_id:
                raise RuntimeError("mesh command was superseded by teardown")
            write_owner_only_json(
                snapshot_path,
                signed.to_dict(),
            )
            latest["mesh_finalized"] = True
            latest["verification_snapshot_hash"] = signed.snapshot_hash_hex()
            latest["verification_snapshot_generation"] = int(signed.generation)
            save_mesh_state(mesh_dir, latest)
        return signed

    def _local_request_context(self) -> tuple[str, dict[str, Any]]:
        """Return the mesh HMAC secret and snapshot binding for a local self-test."""

        if self.mesh_dir is None:
            return "", {}
        from verallm.mesh.state import (
            load_mesh_state,
            state_internal_auth_secret,
        )
        from verallm.mesh.verification_snapshot import MeshVerificationSnapshot

        state = load_mesh_state(self.mesh_dir)
        secret = state_internal_auth_secret(state)
        snapshot_path = Path(self.mesh_dir) / "verification_snapshot.json"
        if not snapshot_path.is_file():
            return secret, {}
        snapshot = MeshVerificationSnapshot.from_dict(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        return secret, {
            "validator_nonce": secrets.token_hex(32),
            "validator_request_id": secrets.token_hex(32),
            "verification_snapshot_hash": snapshot.snapshot_hash_hex(),
            "verification_snapshot_generation": int(snapshot.generation),
        }

    def _verify_serving(self, coordinator_endpoint: str, *, timeout: float = 120.0) -> None:
        """Prove the FULL path works before we report serving.

        A green llama /health only means the model loaded — it does not mean the
        coordinator -> llama -> rpc -> proof pipeline can actually answer. So we
        run one short real generation through the coordinator and require it to
        come back verified. If it can't, drive() raises and the mesh shows an
        honest ``error`` with the reason instead of a hollow "serving" that 503s
        or fails proof on the operator's first message.

        The generation must be MORE than one token: a 1-token completion is
        sampled straight from the prefill graph's logits, so no decode-shaped
        graph ever executes and the slot-view template can never assemble its
        full multi-device challenge universe on a fresh trace dir (live: a
        4-GPU glm drive pinned a 211-op CUDA3-only template against the
        237-op per-layer floor because the self-test's window held only
        prefill sub-graphs). A few decode steps make the self-test's own
        capture window template-complete.
        """
        self._assert_command_active()
        payload = {
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        internal_secret, snapshot_binding = self._local_request_context()
        if snapshot_binding:
            payload["verathos"] = snapshot_binding
        # A fresh mesh's FIRST request is also the one that seeds the proof
        # capture state: the slot-view template needs a capture window that
        # holds a whole forward, and the first request's window can arm
        # mid-forward (observed on a 4-GPU glm serve: a 211-op
        # CUDA3-only window against a 237-op challenge floor). Each retried
        # self-test request arms from the start of its own forward, so the
        # full-forward window the template needs materializes within an
        # attempt or two. Only this known-transient warmup class retries;
        # every other failure stays fatal and honest.
        transient_markers = (
            "no capture window holds a whole forward yet",
            "no single-row decode instance captured",
        )
        # Capacity-audit drain windows 503 EVERY request by design for a
        # few minutes, and a launch can land inside one. The 503 is
        # deliberately byte-identical to genuine busy, so both wait the
        # same bounded patience instead of failing the drive.
        drain_markers = (
            '"slots_busy"',
            "generation capacity is momentarily exhausted",
        )
        attempts = 6
        drain_budget = 20  # x 20s sleep ~= 6.5 min, outlasts any window
        warm_seen = 0
        drain_seen = 0
        last_error: RuntimeError | None = None
        while True:
            self._assert_command_active()
            try:
                body = post_json(
                    coordinator_endpoint + "/v1/chat/completions",
                    payload,
                    timeout=timeout,
                    internal_auth_secret=internal_secret,
                )
            except RuntimeError as exc:
                error = RuntimeError(f"backend self-test failed: {exc}")
                error.__cause__ = exc
                if any(marker in str(exc) for marker in drain_markers):
                    drain_seen += 1
                    if drain_seen > drain_budget:
                        raise error
                    logger.warning(
                        "self-test waiting out a capacity drain / busy "
                        "window (%d/%d): %s",
                        drain_seen,
                        drain_budget,
                        str(exc)[:160],
                    )
                    time.sleep(20.0)
                    continue
                if not any(marker in str(exc) for marker in transient_markers):
                    raise error
                warm_seen += 1
                last_error = error
                if warm_seen >= attempts:
                    raise last_error
                logger.warning(
                    "self-test hit a warmup-transient proof state "
                    "(attempt %d/%d): %s",
                    warm_seen,
                    attempts,
                    str(exc)[:200],
                )
                time.sleep(min(10.0, 2.0 * warm_seen))
                continue
            break
        self._assert_command_active()
        if body.get("error"):
            raise RuntimeError(f"backend not serving cleanly: {str(body['error'])[:200]}")
        mesh_meta = body.get("verathos_mesh", {})
        if mesh_meta.get("proof_required") and not mesh_meta.get("verified"):
            raise RuntimeError(
                "backend generates but proof did not verify: "
                f"{str(mesh_meta.get('proof_error') or body.get('choices'))[:200]}"
            )

    def verify_backend_ready(self) -> str:
        """Wait for a MULTI-box split backend to be truly ready, then prove it.

        The driver's llama-server can't bind until the other members join, which
        only happens after drive_ready is reported — so this runs asynchronously
        after drive() returns (waiting inline would deadlock). Same two-step gate
        as single-box: a real 200 from the backend, then a verified generation.
        """
        cfg = self.config
        self._assert_command_active()
        self._wait_http(
            f"http://127.0.0.1:{cfg.mesh_port + 1}/health",
            timeout=self._ready_timeout_s(), ready_on_200=True,
            failure_file=(self.mesh_dir / "backend-failure.json") if self.mesh_dir else None,
        )
        # Same guard as single-box: never report serving with a cold proof
        # cache (first proofs would stall for tens of seconds).
        self._await_proof_cache_prewarm()
        self._assert_command_active()
        # Publish as late as possible so a slow distributed model load does not
        # consume the signed snapshot's TTL before the mesh can serve requests.
        snapshot = self._finalize_verification_snapshot()
        verification_snapshot_hash = (
            snapshot.snapshot_hash_hex() if snapshot is not None else ""
        )
        # Dial loopback: the coordinator's internal-HMAC lane on validator
        # routes only accepts loopback clients, and the coordinator binds
        # 0.0.0.0 so 127.0.0.1 always reaches it.
        self._verify_serving(
            f"http://127.0.0.1:{cfg.mesh_port}"
        )
        self._assert_command_active()
        return verification_snapshot_hash

    def drive(
        self, command: dict[str, Any], progress: Any | None = None
    ) -> dict[str, Any]:
        self.enter_command(str(command.get("command_id", "") or ""))
        cfg = self.config
        mode = _normalize_pool_serving_mode(command.get("serving_mode"))
        # Only a model with an on-chain slot is scoreable; an unregistered
        # subnet model launches for operator verification (see handle_launch).
        chain_bound = command.get("model_index") is not None
        self._drive_command = dict(command)
        validator_binding: dict[str, int] = {}
        max_context_len = 0
        if mode == POOL_SERVING_MODE_SUBNET:
            validator_binding = _validator_binding(
                **dict(command.get("validator_binding") or {})
            )
            # A chain-bound mesh serves at its REGISTERED context. An
            # unregistered measurement launch has no registered value yet;
            # the KV auto-fit measures one and reports it via drive_ready.
            if chain_bound:
                max_context_len = _require_max_context_len(
                    command.get("max_context_len")
                )
            elif command.get("max_context_len") is not None:
                max_context_len = _require_max_context_len(
                    command.get("max_context_len")
                )
            if (
                (not cfg.wallet_name or not cfg.wallet_hotkey)
                and self._coordinator_sign_request is None
            ):
                # Fail before any spawn: this driver can produce NO
                # coordinator signature, neither locally nor through its
                # manager, so the mesh could never pass a validator.
                raise RuntimeError(
                    "subnet mesh driver has neither a coordinator wallet "
                    "nor a manager signing channel"
                )
            if not cfg.subnet_driver_ready:
                raise RuntimeError(
                    "subnet mesh driver requires a fresh, non-empty "
                    "--validator-allowlist-path"
                )
            # HARD GATE (operator order): a CUDA box may not load ANY
            # model without the working GPU BLAKE3 Merkle kernel. The old
            # behavior warned and served with 6-200s CPU tree builds —
            # multi-second light proofs in production. Verified by a REAL
            # kernel launch, not an import.
            if str(cfg.rpc_device or "").upper().startswith("CUDA"):
                from verallm.mesh.gguf_manifest import (
                    _gpu_merkle_hash_available,
                )

                if not _gpu_merkle_hash_available():
                    raise RuntimeError(
                        "GPU BLAKE3 Merkle kernel unavailable on this CUDA "
                        "worker; refusing to serve with CPU proof "
                        "fallback. Rebuild the zkllm extension for this "
                        "box: python zkllm/cuda/build.py (then restart "
                        "the worker)"
                    )
        self._preflight_backend()
        self._free_own_ports(driving=True)
        entry = cfg.catalog_entry(str(command["model_id"]))
        if mode == POOL_SERVING_MODE_DEV:
            raw_max_context_len = command.get(
                "max_context_len",
                entry.get("max_context_len"),
            )
            if raw_max_context_len not in (None, 0):
                max_context_len = _require_max_context_len(raw_max_context_len)
        if max_context_len:
            self._drive_command["max_context_len"] = max_context_len
        self._prewarm_stop = threading.Event()
        self._prewarm_done = threading.Event()
        self._prewarm_result = {}
        from verallm.mesh.gguf_manifest import (
            bind_gguf_manifest_to_local_model,
            load_gguf_tensor_manifest,
            save_gguf_tensor_manifest,
        )
        from verallm.mesh.model_spec import (
            gguf_package_hash,
            verify_gguf_model_files,
        )

        manifest = load_gguf_tensor_manifest(entry["manifest"])
        manifest_root = str(manifest.get("tensor_manifest_root", ""))
        manifest_package_hash = (
            gguf_package_hash(manifest).hex()
            if mode == POOL_SERVING_MODE_SUBNET
            or bool(manifest.get("model_file_sha256"))
            else ""
        )
        if mode == POOL_SERVING_MODE_SUBNET and chain_bound:
            expected_manifest_root = _require_digest(
                command.get("model_tensor_manifest_root"),
                field_name="model_tensor_manifest_root",
            )
            if manifest_root != expected_manifest_root:
                raise RuntimeError(
                    "local GGUF tensor manifest root does not match the registered model"
                )
            package_hash = _require_digest(
                command.get("model_package_hash"),
                field_name="model_package_hash",
            )
            if manifest_package_hash != package_hash:
                raise RuntimeError(
                    "local GGUF package hash does not match the registered model"
                )
            tokenizer_hash = _require_digest(
                command.get("tokenizer_hash"),
                field_name="tokenizer_hash",
            )
            quantization_scheme = str(
                command.get("quantization_scheme", "") or ""
            ).strip()
            total_layers = int(command.get("total_layers", 0) or 0)
            if total_layers != int(entry["layers"]):
                raise RuntimeError(
                    "local GGUF layer count does not match the registered model"
                )
            coordinator_uid = validator_binding["coordinator_uid"]
            epoch = validator_binding["epoch"]
        elif mode == POOL_SERVING_MODE_SUBNET:
            # UNREGISTERED measurement launch: there is no on-chain slot to
            # verify against yet, so the LOCAL manifest is the identity. The
            # proofs still bind to this manifest, and a later registration
            # (deploy) re-derives the anchors from chain and re-verifies.
            package_hash = str(manifest_package_hash or "")
            if not package_hash:
                raise RuntimeError(
                    "unregistered subnet launch requires a GGUF manifest "
                    "with a package hash"
                )
            tokenizer_hash = ""
            quantization_scheme = "unknown"
            total_layers = int(entry["layers"])
            coordinator_uid = validator_binding["coordinator_uid"]
            epoch = validator_binding["epoch"]
        else:
            configured_package_hash = str(
                entry.get("model_package_hash")
                or entry.get("package_hash")
                or ""
            )
            if configured_package_hash:
                configured_package_hash = _require_digest(
                    configured_package_hash,
                    field_name="catalog package_hash",
                )
                if configured_package_hash != manifest_package_hash:
                    raise RuntimeError(
                        "catalog GGUF package hash does not match the tensor manifest"
                    )
            package_hash = str(
                configured_package_hash
                or manifest_package_hash
                or hashlib.sha256(str(command["model_id"]).encode()).hexdigest()
            )
            tokenizer_hash = ""
            quantization_scheme = "unknown"
            total_layers = int(entry["layers"])
            coordinator_uid = 0
            epoch = 0

        # Manifests are portable commitments, so a copy may retain absolute
        # model paths from the machine that built it. Bind a runtime-only copy
        # to this driver's GGUF, but only after hashing the local file(s)
        # against the manifest/package identity. This is what lets the driver
        # service an on-demand proof blob for a file-less member safely.
        if manifest.get("model_file_sha256"):
            manifest = bind_gguf_manifest_to_local_model(
                manifest,
                entry["llama_model"],
                expected_package_hash=manifest_package_hash,
            )
            runtime_manifest_name = (
                hashlib.sha256(str(command["model_id"]).encode()).hexdigest()[:16]
                + "-"
                + manifest_root[:16]
                + ".tensor-manifest.json"
            )
            runtime_manifest_path = save_gguf_tensor_manifest(
                manifest,
                cfg.workdir / "runtime-manifests" / runtime_manifest_name,
            )
            entry = {**entry, "manifest": str(runtime_manifest_path)}
        elif mode == POOL_SERVING_MODE_SUBNET:
            # Compatibility for legacy/test manifests without package file
            # metadata. Current manifests always take the verified binding
            # path above.
            verify_gguf_model_files(manifest)

        # Warm the proof-weight cache CONCURRENTLY with the model load and
        # refuse to report ready until it finishes. It must use the verified
        # runtime manifest, never the copied manifest's foreign paths.
        threading.Thread(
            target=_prewarm_proof_cache,
            args=(cfg, str(command["model_id"]), self._prewarm_stop),
            kwargs={
                "progress": progress,
                "done": self._prewarm_done,
                "result": self._prewarm_result,
                "entry_override": entry,
            },
            daemon=True,
        ).start()
        coordinator_endpoint = f"http://{cfg.advertise_host}:{cfg.mesh_port}"
        spec = MeshSpec.new_private_mesh(
            coordinator_uid=coordinator_uid,
            coordinator_hotkey=self._coordinator_hotkey(),
            endpoint=coordinator_endpoint,
            model_id=str(command["model_id"]),
            model_package_ref=str(entry["llama_model"]),
            model_package_hash=package_hash,
            model_tensor_manifest_root=manifest_root,
            tokenizer_hash=tokenizer_hash,
            quantization_scheme=quantization_scheme,
            max_context_len=max_context_len,
            total_layers=total_layers,
            epoch=epoch,
            mesh_id=str(command.get("resume_mesh_id", "") or ""),
        )
        spec.proof_trace_manifest_format = "compact-raw-v3"
        spec.validate()
        mesh_dir, state, token = create_mesh_state(
            spec=spec,
            root=cfg.workdir / "driver",
        )
        self.mesh_dir = mesh_dir
        member_count = int(command.get("member_count", 1))
        from verallm.mesh.state import save_mesh_state

        state["expected_compute_stage_count"] = member_count
        state["serving_mode"] = mode
        state["verification_snapshot_required"] = (
            mode == POOL_SERVING_MODE_SUBNET
        )
        save_mesh_state(mesh_dir, state)
        # One llama device per member GPU: each member's rpc-server exposes
        # its whole GPU group, and llama numbers remote devices RPC0..RPCn-1
        # globally in --rpc endpoint (= committed stage) order.
        member_device_counts = [
            max(1, int(count))
            for count in (
                command.get("member_device_counts")
                or [1] * member_count
            )
        ]
        if len(member_device_counts) != member_count:
            member_device_counts = [1] * member_count
        total_devices = sum(member_device_counts)
        # LOCAL STAGE: a single-worker mesh needs no RPC at all - the
        # driver's llama-server computes the member's stage directly on the
        # local CUDA devices and captures into the member's trace dir, which
        # removes the per-token loopback rpc-server hop. This applies in
        # VALIDATOR (subnet production) mode too: the verification snapshot
        # is endpoint-free by construction (stage keys + layer ranges +
        # model anchors), so the committed verification surface is identical
        # whether the stage ran behind an rpc-server or inside llama-server.
        local_stage = member_count <= 1
        if local_stage:
            # The worker's own in-mask device list ("CUDA0,CUDA1", "MTL0"):
            # exactly what its rpc-server would have exposed.
            devices = cfg.rpc_device
        else:
            devices = ",".join(f"RPC{i}" for i in range(total_devices))
        coord_cmd = [
            sys.executable, "-m", "neurons.cli", "mesh", "serve",
            "--mesh", str(mesh_dir),
            # Bind all interfaces: validators and members dial the advertise
            # address, while the driver's own self-tests (_verify_serving and
            # operator chats) must dial 127.0.0.1 because the coordinator's
            # internal-HMAC lane on validator routes only accepts loopback
            # clients. Binding only the advertise host would 403 every
            # self-test on any machine whose advertise host is a real address.
            "--host", "0.0.0.0", "--port", str(cfg.mesh_port),
            "--llama-model", str(entry["llama_model"]),
            "--llama-server-binary", cfg.llama_server_binary,
            "--llama-host", "127.0.0.1", "--llama-port", str(cfg.mesh_port + 1),
            "--llama-device", devices,
            "--llama-n-gpu-layers", "all",
            "--llama-min-rpc-workers", "0" if local_stage else str(member_count),
            *(
                [
                    "--llama-capture-trace-dir",
                    str(
                        model_trace_dir(
                            cfg.workdir, str(entry.get("model_id", ""))
                        )
                    ),
                    # Deterministic split over the member's committed per-GPU
                    # weights; with no rpc plan there is nothing to derive it
                    # from and llama's free placement would be uncommitted.
                    *(
                        [
                            "--llama-tensor-split",
                            ",".join(
                                str(w) for w in cfg.gpu_split_weights
                            ),
                        ]
                        if len(cfg.gpu_split_weights) > 1
                        else []
                    ),
                ]
                if local_stage
                else []
            ),
            # Unified KV, vLLM-parity: --ctx-size is ONE shared budget the
            # --parallel slots draw from (llama would otherwise partition
            # it statically per slot, shrinking every request). A single
            # request may therefore use the model's FULL advertised
            # context while concurrent smaller ones pack beside it, and a
            # prompt past the remaining budget fails loudly
            # (exceed_context_size_error), never truncated. The budget: a
            # registry max_context_len when the model is pinned (the
            # advertised per-request maximum IS the budget), else 0 =
            # llama resolves the model's own trained maximum.
            "--llama-extra-arg=--kv-unified",
            # MoE prefill throughput is dominated by tokens-per-expert-pass:
            # at the default n_ubatch 512 each pass streams essentially the
            # whole expert weight set for ~2 tokens per routed expert
            # (live-measured 102 tok/s over an 88k prefill on 4x A100 for a
            # 256-expert model; published anchors put ub 4096 at 2-4x that).
            # Batch stays 2x ubatch so the layer-split pipeline keeps two
            # ubatches in flight. Changing ubatch changes prefill graph
            # shapes: the slot-view template re-derives at launch and the
            # name-keyed capture arming is shape-agnostic by design.

            # Per-model llama batching (see _llama_batch_args: env
            # break-glass > registry/mesh per-model value > defaults).
            *_llama_batch_args(command),
            "--llama-ctx-size",
            str(_serve_ctx_budget(max_context_len)),
            # Concurrent serving. Decode audits at --parallel are certified
            # (probe windows arm their own graph ordinals; audit-window
            # bounds are per holder) and receipts carry the llama-server
            # slot id above one slot. The value stays EXPLICIT so the
            # coordinator and llama.cpp never disagree about slot count;
            # the tail-ring light fast path self-disables above one slot
            # (its position binding is instance-order within a single
            # slot) and audit-drawn lights take the certified probe path.
            "--llama-extra-arg=--parallel",
            f"--llama-extra-arg={_serve_parallel_slots()}",
            *PROOF_FLAGS,
            # The coordinator needs its own trace dir even in all-RPC meshes
            # where every stage trace lives on the members: slot-view scope
            # (required for any decode-audited serve) is gated on
            # proof_trace_dir for its window-token plumbing. The certified
            # 2-RPC-worker harnesses always passed this flag; omitting it
            # here silently pinned the pool to the legacy candidate scope,
            # whose missing-witness solo re-serve can never semantically
            # match a graph-served original. Fresh per-mesh dir, never the
            # member's traces dir: the two processes must not share capture
            # enable files.
            "--proof-trace-dir", str(mesh_dir / "coordinator-traces"),
            "--proof-gguf-manifest", str(entry["manifest"]),
            # Slot-state save/restore: hard-audit probes restore the audited
            # serve's KV instead of re-prefilling an evicted context eager.
            # The serve CLI mkdirs it and activates llama's --slot-save-path
            # + --slots on the loopback-bound aux port.
            "--slot-state-dir", str(mesh_dir / "slot-states"),
            *self._wallet_flags(),
        ]
        # An UNREGISTERED subnet model (no on-chain slot yet) runs the
        # operator-lane coordinator: it must answer probe/chat/self-test over
        # the internal-HMAC lane so the operator can verify the model and
        # read its measured context BEFORE registering. A snapshot-bound
        # coordinator refuses those routes by design, and there is no
        # model_index to build a coordinator identity from, so it cannot be
        # snapshot-bound until the model is registered.
        if mode == POOL_SERVING_MODE_SUBNET and chain_bound:
            coord_cmd += self._validator_coordinator_flags(command)
            coord_cmd += self._coordinator_signing_flags(mesh_dir, command)
            # The daemon's capacity-audit worker writes the drain state
            # into the worker's own workdir; the serve process gates
            # admission on it so canaries see an ordinary busy 503 while
            # an audit window saturates the GPUs.
            from verallm.mesh.capacity_audit_worker import (
                capacity_drain_file_path,
                capacity_roster_file_path,
            )

            coord_cmd += [
                "--capacity-drain-file",
                str(capacity_drain_file_path(self.config.workdir)),
                "--capacity-roster-file",
                str(capacity_roster_file_path(self.config.workdir)),
            ]
        else:
            coord_cmd += [
                "--server-role", "coordinator",
                "--allow-loopback-dev-validator-routes",
            ]
        coord_cmd += _tolerance_flags(command)
        # VERATHOS_LLAMA_RPC_TCP_PROBE=1 makes the coordinator's llama-server
        # supervisor wait until the rpc worker's port actually accepts before
        # spawning llama-server. Without it the supervisor spawns llama-server
        # the instant the rpc endpoint is *registered* (which drive() does just
        # below, before the rpc process is even up) — llama-server then dies on
        # "invalid device RPC0" and the retry loop wedges the llama port.
        coord_env = {**os.environ, "VERATHOS_LLAMA_RPC_TCP_PROBE": "1"}
        # vLLM-style auto-fit by MEASUREMENT: llama-server is the only
        # accurate judge of whether weights + a unified KV budget fit this
        # machine (exact KV math spans GQA, MLA, and recurrent hybrids).
        # Start at the full budget (0 = the model's trained maximum); on
        # an ALLOCATION failure descend a bounded ladder and cache the
        # fitted value per model so later launches start right. A
        # registry-pinned budget never descends: the pin is the advertised
        # per-request contract, and a machine that cannot hold it must
        # fail the launch loudly instead of quietly serving less.
        ctx_value_index = coord_cmd.index("--llama-ctx-size") + 1
        requested_budget = int(coord_cmd[ctx_value_index])
        # The registered context is the per-request CONTRACT and the
        # ladder's FLOOR: the serve budget starts from the machine's
        # MEASURED fit (cached, else estimated from free VRAM) so real
        # headroom serves concurrency - overlapping validator canaries
        # and private-API traffic - instead of sitting idle. The ladder
        # plus the first-batch probe shrink a too-optimistic start, never
        # below the contract. A cached fit below the contract cannot
        # serve chain-bound and is ignored.
        ctx_floor = int(max_context_len or 0)
        cached_fit, cached_descended = self._cached_kv_fit(command)
        if cached_fit and cached_fit < ctx_floor:
            cached_fit = 0
        estimated = _kv_tokens_that_fit(
            str(entry.get("llama_model", "") or ""),
            int(entry.get("model_bytes", 0) or 0),
        )
        if (
            cached_fit
            and not cached_descended
            and estimated >= max(8192, 2 * cached_fit)
        ):
            # First-try cache far below what the hardware now reports:
            # almost certainly measured against transiently-occupied
            # VRAM (a dying previous serve). Re-measure from the fresh
            # estimate; the ladder + first-batch probe correct any
            # overshoot. A DESCENT-validated cache is a probed ceiling
            # and is never second-guessed this way.
            logger.warning(
                "discarding suspicious first-try KV fit %d (fresh "
                "estimate %d); re-measuring",
                cached_fit,
                estimated,
            )
            cached_fit = 0
        if cached_fit:
            if requested_budget == 0 or cached_fit > requested_budget:
                coord_cmd[ctx_value_index] = str(cached_fit)
        else:
            # No trusted measurement yet: size the FIRST attempt from
            # free VRAM instead of the trained maximum (a 1M-context
            # model on 4x80GB otherwise OOMs at load) or the bare
            # contract (which would waste every token of real headroom).
            if estimated >= 8192 and (
                requested_budget == 0 or estimated > requested_budget
            ):
                logger.info(
                    "unified KV budget estimated from free VRAM: %d tokens",
                    estimated,
                )
                coord_cmd[ctx_value_index] = str(estimated)
        ladder_descended = False
        while True:
            # Scope error scraping to THIS attempt: the logs append across
            # rungs (and drives), and a stale cudaMalloc line must not
            # classify this rung's failure.
            self._mark_backend_log_offsets()
            self._spawn(
                coord_cmd, "pool-driver-coordinator.log", env=coord_env
            )
            try:
                # Loopback dial: the coordinator is our own child on this
                # box; the advertise host may not hairpin inside a NAT.
                self._wait_http(f"http://127.0.0.1:{cfg.mesh_port}/health")
                if local_stage:
                    # Health is "loads", not "serves". Prove the budget
                    # survives its first full batch while the ladder can
                    # still descend. (Multi-box meshes cannot answer until
                    # their members join, so only a local-stage driver can
                    # probe here.)
                    self._llama_first_batch_probe(
                        cfg.mesh_port + 1,
                        int(entry.get("model_bytes", 0) or 0),
                    )
                    # The probe just forced the lazy prefill compute-buffer
                    # allocation, so free VRAM NOW is the serve's true
                    # steady state. The capacity-audit workload must fit
                    # beside it at any window; when the KV budget is pinned
                    # at the registered contract the marginal consumer is
                    # the ubatch-sized compute buffer, so descend that.
                    free_mb = _gpu_free_vram_mb()
                    if free_mb and min(free_mb) < _effective_audit_reserve_mb():
                        ub_idx = _extra_arg_value_index(coord_cmd, "-ub")
                        b_idx = _extra_arg_value_index(coord_cmd, "-b")
                        current_ub = (
                            int(coord_cmd[ub_idx].split("=", 1)[1])
                            if ub_idx >= 0
                            else 0
                        )
                        if current_ub > 512:
                            new_ub = max(512, current_ub // 2)
                            self._end_kv_ladder_attempt()
                            logger.warning(
                                "capacity-audit reserve unmet after first "
                                "batch (min free %dMB < %dMB); halving "
                                "micro-batch %d -> %d",
                                min(free_mb),
                                _effective_audit_reserve_mb(),
                                current_ub,
                                new_ub,
                            )
                            coord_cmd[ub_idx] = f"--llama-extra-arg={new_ub}"
                            if b_idx >= 0:
                                coord_cmd[b_idx] = (
                                    f"--llama-extra-arg={2 * new_ub}"
                                )
                            continue
                        logger.warning(
                            "capacity-audit reserve unmet even at the "
                            "micro-batch floor (min free %dMB < %dMB); "
                            "audit workloads on this placement will "
                            "likely OOM",
                            min(free_mb),
                            _effective_audit_reserve_mb(),
                        )
                break
            except RuntimeError as exc:
                detail = f"{exc}; {self._scrape_backend_error()}"
                # End the attempt on EVERY exit, not only before a retry:
                # a raise that leaves the dead attempt's llama alive keeps
                # VRAM occupied (the next rung then OOMs below its true
                # fit) and keeps its port held, which the worker's
                # safe-stop later refuses to kill as an unowned listener.
                self._end_kv_ladder_attempt()
                if not _KV_ALLOC_FAILURE_RE.search(detail):
                    raise
                current = int(coord_cmd[ctx_value_index])
                if ctx_floor and current <= ctx_floor:
                    # The context cannot go below the registered contract,
                    # but the ubatch compute buffer can still shrink before
                    # declaring the machine unable to serve its contract
                    # (same descent as the no-floor branch below).
                    ub_idx = _extra_arg_value_index(coord_cmd, "-ub")
                    b_idx = _extra_arg_value_index(coord_cmd, "-b")
                    current_ub = (
                        int(coord_cmd[ub_idx].split("=", 1)[1])
                        if ub_idx >= 0
                        else 0
                    )
                    if current_ub > 512:
                        new_ub = max(512, current_ub // 2)
                        logger.warning(
                            "KV budget pinned at the %d-token contract but "
                            "the compute buffer OOMs; halving micro-batch "
                            "%d -> %d and retrying",
                            current,
                            current_ub,
                            new_ub,
                        )
                        coord_cmd[ub_idx] = f"--llama-extra-arg={new_ub}"
                        if b_idx >= 0:
                            coord_cmd[b_idx] = (
                                f"--llama-extra-arg={2 * new_ub}"
                            )
                        ladder_descended = True
                        continue
                    raise RuntimeError(
                        "unified KV budget does not fit this machine even "
                        f"at the registered contract of {current} tokens: "
                        f"{detail}"
                    )
                smaller = next(
                    (
                        step
                        for step in _KV_FIT_LADDER
                        if (current == 0 or step < current)
                        and step >= ctx_floor
                    ),
                    ctx_floor if ctx_floor else None,
                )
                if not smaller:
                    # Context is at its floor but the OTHER VRAM consumer -
                    # the ubatch-sized compute buffer - may still shrink.
                    # The audit-reserve branch below already halves it; a
                    # model whose weights leave less headroom than ubatch
                    # needs the same descent here or it never fits at all.
                    ub_idx = _extra_arg_value_index(coord_cmd, "-ub")
                    b_idx = _extra_arg_value_index(coord_cmd, "-b")
                    current_ub = (
                        int(coord_cmd[ub_idx].split("=", 1)[1])
                        if ub_idx >= 0
                        else 0
                    )
                    if current_ub > 512:
                        new_ub = max(512, current_ub // 2)
                        logger.warning(
                            "unified KV budget at its floor (%d tokens) but "
                            "the compute buffer still OOMs; halving "
                            "micro-batch %d -> %d and retrying",
                            current,
                            current_ub,
                            new_ub,
                        )
                        coord_cmd[ub_idx] = f"--llama-extra-arg={new_ub}"
                        if b_idx >= 0:
                            coord_cmd[b_idx] = (
                                f"--llama-extra-arg={2 * new_ub}"
                            )
                        ladder_descended = True
                        continue
                    raise RuntimeError(
                        "unified KV budget does not fit this machine even "
                        f"at {current} tokens: {detail}"
                    )
                logger.warning(
                    "unified KV budget %s does not fit; retrying at %d",
                    "model-max" if current == 0 else str(current),
                    smaller,
                )
                coord_cmd[ctx_value_index] = str(smaller)
                ladder_descended = True
        fitted_budget = int(coord_cmd[ctx_value_index])
        if fitted_budget and fitted_budget != requested_budget:
            self._store_kv_fit(
                command,
                fitted_budget,
                # A value the ladder descended TO is a probed ceiling; a
                # first-try success only proves "this worked right now".
                descended=bool(ladder_descended),
            )
        # The driver is also the first member. All-RPC: a local rpc worker.
        # Local stage: NO rpc endpoint - the coordinator's llama-server
        # computes this stage; the member process is proof server + arming
        # owner only.
        worker_dir, _ = join_mesh(
            token=token.encode(),
            endpoint=f"http://{cfg.advertise_host}:{cfg.proof_port}",
            root=cfg.workdir / "member",
            hotkey=self._stage_proof_key_ss58,
            package_hash=package_hash,
            gpu_name=cfg.gpu_name,
            vram_gb=int(cfg.vram_gb),
            per_gpu_vram_gb=cfg.gpu_split_weights,
            rpc_endpoint=(
                "" if local_stage else f"{cfg.advertise_host}:{cfg.rpc_port}"
            ),
            proof_endpoint=f"http://{cfg.advertise_host}:{cfg.proof_port}",
            timeout=10.0,
            self_advertise_host=str(cfg.advertise_host),
        )
        self._spawn(
            self._worker_serve_cmd(worker_dir, entry, rpc_worker=not local_stage)
            + _tolerance_flags(command),
            "pool-driver-worker.log",
        )
        # The local member must be reachable before other members join, so it
        # receives the coordinator's spec-update broadcasts (a member that
        # misses a reassignment serves a stale spec hash and 409s arming).
        # Loopback dial: this is our own child; the advertise host may not
        # hairpin from inside a NAT.
        self._wait_http(
            f"http://127.0.0.1:{cfg.proof_port}/health",
            ready_on_200=True,
        )
        # For a SINGLE-worker mesh, wait for the llama backend to actually bind
        # before drive_ready so "serving" means the mesh can answer (no
        # connection-reset on the first chat). For a MULTI-worker mesh the
        # backend can't come up until the other members join — and they only
        # join AFTER drive_ready — so waiting here would deadlock. There the
        # backend readiness is reflected when members report "serving".
        if member_count <= 1:
            # Cold-loading a 30B-class GGUF from disk can take several minutes;
            # 240s was too tight and failed healthy drives mid-load. Require a
            # real 200 (not mere reachability) so we wait out llama's 503 load
            # instead of reporting serving while the model is still loading.
            self._finalize_verification_snapshot()
            self._wait_http(
                f"http://127.0.0.1:{cfg.mesh_port + 1}/health",
                timeout=self._ready_timeout_s(), ready_on_200=True,
                failure_file=self.mesh_dir / "backend-failure.json",
            )
            if progress is not None:
                progress("warming proofs")
            self._await_proof_cache_prewarm()
            # /health green only means loaded; prove the full verified path can
            # actually answer before we hand back drive_ready. Loopback dial:
            # the internal-HMAC lane rejects non-loopback clients.
            self._verify_serving(
                f"http://127.0.0.1:{cfg.mesh_port}"
            )
        result = {
            "event": "drive_ready",
            "mesh_id": spec.mesh_id,
            "join_token": token.encode(),
            "coordinator_endpoint": coordinator_endpoint,
        }
        # The context this machine actually holds, MEASURED by the KV auto-fit
        # during this launch. A miner is scored on context and validators
        # canary it at the registered maximum, so the registered value must
        # be this number - never a hand-typed guess. register-model consumes
        # it. A zero fit means the model's TRAINED maximum fit whole; that
        # trained value then IS the measurement (a registration may never
        # exceed it anyway).
        measured_ctx, _measured_descended = self._cached_kv_fit(command)
        if measured_ctx <= 0:
            measured_ctx = _gguf_trained_context(
                str(entry.get("llama_model", "") or "")
            )
        if measured_ctx > 0:
            result["measured_ctx_budget"] = int(measured_ctx)
        if mode == POOL_SERVING_MODE_SUBNET and chain_bound and member_count <= 1:
            _secret, snapshot_binding = self._local_request_context()
            snapshot_hash = str(
                snapshot_binding.get("verification_snapshot_hash", "") or ""
            )
            if not _COMMAND_DIGEST_RE.fullmatch(snapshot_hash):
                raise RuntimeError(
                    "subnet mesh drive completed without a signed snapshot"
                )
            result["verification_snapshot_hash"] = snapshot_hash
            # A resumed or rotated chain is ahead of the manager's launch
            # counter; report the generation actually served so the manager's
            # mesh record stays truthful for scoring-sample ordering.
            generation_val = snapshot_binding.get(
                "verification_snapshot_generation"
            )
            if type(generation_val) is int and generation_val >= 1:
                result["snapshot_generation"] = generation_val
        return result

    def join(self, command: dict[str, Any]) -> dict[str, Any]:
        self.enter_command(str(command.get("command_id", "") or ""))
        cfg = self.config
        self._preflight_backend()
        self._free_own_ports(driving=False)
        token = str(command["join_token"])
        worker_dir, joined = join_mesh(
            token=token,
            endpoint=f"http://{cfg.advertise_host}:{cfg.proof_port}",
            root=cfg.workdir / "member",
            hotkey=self._stage_proof_key_ss58,
            package_hash="",
            gpu_name=cfg.gpu_name,
            vram_gb=int(cfg.vram_gb),
            per_gpu_vram_gb=cfg.gpu_split_weights,
            rpc_endpoint=f"{cfg.advertise_host}:{cfg.rpc_port}",
            proof_endpoint=f"http://{cfg.advertise_host}:{cfg.proof_port}",
            timeout=10.0,
            self_advertise_host=str(cfg.advertise_host),
        )
        coordinator = str(command.get("coordinator_endpoint", "") or "")
        try:
            entry = cfg.catalog_entry(str(joined.model_id))
        except ValueError:
            from verallm.mesh.state import MeshJoinToken

            join_token = MeshJoinToken.decode(token)
            # FILE-LESS member: an rpc worker never reads the GGUF (the driver's
            # llama-server streams layer slices over RPC at load), and the proof
            # sidecar proves from content-addressed blobs it fetches on demand.
            # All it needs locally is the tensor manifest — fetch that from the
            # coordinator and bind it to the mesh spec's committed Merkle root.
            entry = self._fetch_manifest_entry(
                str(joined.model_id),
                coordinator,
                str(joined.model_tensor_manifest_root),
                str(joined.model_package_hash),
                internal_auth_secret=join_token.join_secret,
            )
        # Mirror the driver's runtime-manifest binding for members that can
        # reach the GGUF locally: a portable manifest copy carries no local
        # model_file paths, and without them the proof sidecar loses the
        # exact quantized-weight recomputation fallback. The float path
        # alone can exceed tolerance on single-row decode GEMMs
        # (observed on q4_K ffn tensors under slot-view selection).
        # A file-less catalog does not mean the file is absent: units on one
        # host share the auto-fetch dir, so probe it too. Binding is
        # hash-verified end to end, so a stale or wrong local file falls
        # back to the portable manifest instead of failing the join.
        local_model_for_binding = str(entry.get("llama_model", "") or "")
        if not local_model_for_binding:
            # rglob, not glob: hf_hub_download preserves the repo's own
            # sub-path, so a fetched multi-file model lands under a quant
            # subdirectory (e.g. UD-IQ2_M/) rather than flat in the dir.
            shared_ggufs = sorted(
                _fetch_dest_dir(str(joined.model_id)).rglob("*.gguf")
            )
            if shared_ggufs:
                local_model_for_binding = str(shared_ggufs[0])
        if local_model_for_binding:
            from verallm.mesh.gguf_manifest import (
                bind_gguf_manifest_to_local_model,
                load_gguf_tensor_manifest,
                save_gguf_tensor_manifest,
            )
            from verallm.mesh.model_spec import gguf_package_hash

            try:
                member_manifest = load_gguf_tensor_manifest(entry["manifest"])
                if member_manifest.get("model_file_sha256"):
                    member_manifest = bind_gguf_manifest_to_local_model(
                        member_manifest,
                        local_model_for_binding,
                        expected_package_hash=gguf_package_hash(
                            member_manifest
                        ).hex(),
                    )
                    runtime_manifest_name = (
                        hashlib.sha256(
                            str(joined.model_id).encode()
                        ).hexdigest()[:16]
                        + "-"
                        + str(
                            member_manifest.get("tensor_manifest_root", "")
                        )[:16]
                        + ".tensor-manifest.json"
                    )
                    runtime_manifest_path = save_gguf_tensor_manifest(
                        member_manifest,
                        cfg.workdir
                        / "runtime-manifests"
                        / runtime_manifest_name,
                    )
                    entry = {**entry, "manifest": str(runtime_manifest_path)}
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "member runtime-manifest binding skipped (%s); exact "
                    "recomputation fallback unavailable on this member",
                    exc,
                )
        self.mesh_dir = worker_dir
        member_env = dict(os.environ)
        if coordinator:
            # Lets the proof sidecar fetch content-addressed weight blobs it
            # doesn't have locally from the driver (hash-verified), so a
            # member needs only its slice — never the full GGUF.
            member_env["VERALLM_PROOF_BLOB_PEERS"] = coordinator
            member_env["VERALLM_PROOF_BLOB_AUTH_STATE"] = str(worker_dir)
        self._spawn(
            self._worker_serve_cmd(worker_dir, entry) + _tolerance_flags(command),
            "pool-member-worker.log",
            env=member_env,
        )
        # Loopback dials: these are our own children; the advertise host may
        # not hairpin from inside a NAT.
        self._wait_http(
            f"http://127.0.0.1:{cfg.proof_port}/health",
            timeout=self._ready_timeout_s(),
            ready_on_200=True,
        )
        self._wait_tcp(
            "127.0.0.1",
            cfg.rpc_port,
            timeout=self._ready_timeout_s(),
        )
        return {"event": "serving"}

    def fetch(self, command: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
        """Download a model this worker lacks, so it can DRIVE a mesh for it.

        Members never need this (they join file-lessly); only a driver reads
        the GGUF. Downloads from the registry-advertised HF repo, then
        acquires the OWNER-BUILT tensor manifest (store download, or a file
        pre-staged next to the model) — a fetching box never builds one.
        Progress lands in the worker's heartbeat status string
        ("fetching 42%") for the dashboard.
        """
        self.enter_command(str(command.get("command_id", "") or ""))
        spec = dict(command.get("spec") or {})
        model_id = str(command["model_id"])
        repo = str(spec.get("hf_repo", ""))
        files = [str(f) for f in (spec.get("hf_files") or [])]
        if not repo or not files:
            raise ValueError(f"no download source known for {model_id}")
        from huggingface_hub import hf_hub_download

        from verallm.mesh.gguf_manifest import (
            load_gguf_tensor_manifest,
            save_gguf_tensor_manifest,
        )

        dest_dir = _fetch_dest_dir(model_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        total = int(spec.get("model_bytes", 0) or 0)
        # Fail BEFORE downloading if the model can't fit: huggingface_hub only
        # warns on low disk, then dies mid-file after minutes of progress.
        # (The manager pre-checks the advertised free disk too, but disk may
        # have shrunk since this worker joined.)
        already = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
        free = shutil.disk_usage(dest_dir).free
        need = total - already
        if need > 0 and free < need * 1.05 + 1e9:
            raise RuntimeError(
                f"not enough disk to download {model_id}: needs ~{need / 1e9:.1f} GB "
                f"more, only {free / 1e9:.1f} GB free at {dest_dir}"
            )
        done_evt = threading.Event()
        # A percentage alone lies about what the fetch is doing: bytes already
        # on disk from an earlier attempt read as "99%" the instant the fetch
        # starts, and stay there for minutes while huggingface_hub checksums
        # those files instead of moving any bytes. The file counter makes the
        # difference visible ("file 2/6 99%" advancing with no percent change
        # is verification, not a stall).
        phase = {"label": ""}

        def _watch() -> None:
            while not done_evt.is_set():
                got = sum(
                    f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()
                )
                if total and progress is not None:
                    pct = min(99, int(got * 100 / total))
                    label = phase["label"]
                    progress(f"{label} {pct}%" if label else f"{pct}%")
                done_evt.wait(2.0)

        threading.Thread(target=_watch, daemon=True).start()
        try:
            for index, fn in enumerate(files, start=1):
                phase["label"] = f"file {index}/{len(files)}"
                hf_hub_download(repo_id=repo, filename=fn, local_dir=str(dest_dir))
        finally:
            done_evt.set()
        model_path = dest_dir / files[0]
        manifest_path = _acquire_owner_manifest(
            model_id=model_id,
            spec=spec,
            dest_dir=dest_dir,
            progress=progress,
        )
        entry = {
            "model_id": model_id,
            "llama_model": str(model_path),
            "manifest": str(manifest_path),
            "layers": int(spec.get("layers", 0) or 0),
            "model_bytes": total
            or sum((dest_dir / f).stat().st_size for f in files),
            "hf_repo": repo,
            "hf_files": files,
            # Keep the per-model proof tolerances with the model: future
            # launches read them from THIS driver's catalog entry.
            **_operator_tuning_fields(spec),
        }
        self.config.catalog.append(entry)
        if self.config.catalog_path:
            Path(self.config.catalog_path).write_text(json.dumps(self.config.catalog, indent=1))
        _release_fetch_memory(model_id)
        return {"event": "fetched", "entry": entry}

    def stop(self, command: dict[str, Any]) -> dict[str, Any]:
        self.enter_command(str(command.get("command_id", "") or ""))
        self._prewarm_stop.set()
        self._terminate_tracked_processes()
        # Never claim a clean stop while an orphan from an earlier worker
        # process still owns one of this worker's configured listener ports.
        self._free_own_ports(driving=True)
        return {"event": "stopped"}

    def rotate_snapshot(self, command: dict[str, Any]) -> dict[str, Any]:
        """Re-sign the published verification snapshot for a new epoch.

        Validators pin a mesh's signed snapshot per scoring epoch and refuse
        every other epoch, so a long-lived mesh must refresh the binding at
        each boundary. The content stays byte-identical to the finalized
        snapshot (same mesh, model anchors, stages, policy); only epoch,
        generation, and the freshness window change, re-signed with the same
        coordinator hotkey. The serve path re-reads the snapshot file on
        every request, so rotation needs no relaunch and never writes to the
        chain. This runs BESIDE the live drive command: it must neither
        fence nor enter the command (that would supersede the serving
        drive); the fence lock only guards the file swap against a racing
        teardown.
        """

        from verallm.mesh.private_files import write_owner_only_json
        from verallm.mesh.receipt_signing import load_hotkey_keypair
        from verallm.mesh.state import load_mesh_state, save_mesh_state
        from verallm.mesh.verification_snapshot import (
            MeshVerificationSnapshot,
            rotate_mesh_verification_snapshot,
            verify_mesh_verification_snapshot_signature,
        )

        epoch = command.get("epoch")
        if type(epoch) is not int or epoch < 0 or epoch >= 2**63:
            raise RuntimeError("rotate requires an integer epoch")
        if self.mesh_dir is None:
            raise RuntimeError("driver mesh state is unavailable")
        mesh_dir = Path(self.mesh_dir)
        with self._command_fence_lock:
            fence_at_entry = self._active_command_id
        state = load_mesh_state(mesh_dir)
        if not state.get("mesh_finalized"):
            raise RuntimeError("mesh has no finalized snapshot to rotate")
        snapshot_path = mesh_dir / "verification_snapshot.json"
        existing = MeshVerificationSnapshot.from_dict(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        if int(existing.epoch) == epoch:
            # Idempotent: a redelivered command must not burn a generation.
            return {
                "event": "rotated",
                "epoch": int(epoch),
                "verification_snapshot_hash": existing.snapshot_hash_hex(),
                "snapshot_generation": int(existing.generation),
            }
        keypair = self._coordinator_keypair(
            coordinator_hotkey=existing.coordinator.coordinator_hotkey,
            mesh_key=str(command.get("mesh_key", "") or ""),
        )
        if str(keypair.ss58_address) != existing.coordinator.coordinator_hotkey:
            raise RuntimeError(
                "coordinator wallet does not match the snapshot hotkey"
            )
        # Signature-only check on the outgoing snapshot: rotating an already
        # EXPIRED snapshot is exactly the recovery this command exists for,
        # so freshness is deliberately not enforced here (now_unix=None).
        if not verify_mesh_verification_snapshot_signature(
            existing,
            expected_hotkey=existing.coordinator.coordinator_hotkey,
            expected_epoch=int(existing.epoch),
        ):
            raise RuntimeError("persisted verification snapshot is invalid")
        signed = rotate_mesh_verification_snapshot(
            existing, epoch=int(epoch), keypair=keypair
        )
        with self._command_fence_lock:
            if self._active_command_id != fence_at_entry:
                # A stop/drive fenced in while we were signing: the mesh is
                # being torn down or replaced; publish nothing.
                raise RuntimeError("mesh command fence changed during rotation")
            latest = load_mesh_state(mesh_dir)
            if not latest.get("mesh_finalized"):
                raise RuntimeError("mesh was torn down during rotation")
            write_owner_only_json(snapshot_path, signed.to_dict())
            latest["verification_snapshot_hash"] = signed.snapshot_hash_hex()
            latest["verification_snapshot_generation"] = int(signed.generation)
            save_mesh_state(mesh_dir, latest)
        return {
            "event": "rotated",
            "epoch": int(epoch),
            "verification_snapshot_hash": signed.snapshot_hash_hex(),
            "snapshot_generation": int(signed.generation),
        }


def resume_mesh_verification_snapshot_chain(
    snapshot_path: Path,
    *,
    spec: Any,
    coordinator_identity: Any,
    policy: Any,
    stage_bindings: Any,
    epoch: int,
    keypair: Any,
) -> Any:
    """Resume a relaunched mesh's persisted verification snapshot chain.

    Mesh identity is logical: a relaunch of the same registration keeps its
    mesh id, so the snapshot signed by the previous instance still binds this
    exact assignment. Re-serving it unchanged (same epoch) keeps a
    validator's epoch pin valid across an honest restart; rotating it (the
    epoch moved while the mesh was down) is the standard boundary rotation
    arriving late, and rotating an EXPIRED snapshot is that command's own
    recovery case, so freshness is deliberately not enforced. Returns None
    when no usable chain exists — a changed assignment, an unreadable file,
    or a foreign signature — and the caller mints a new chain.
    """

    from verallm.mesh.verification_snapshot import (
        MeshVerificationSnapshot,
        build_mesh_verification_snapshot,
        rotate_mesh_verification_snapshot,
        verify_mesh_verification_snapshot_signature,
    )

    if not snapshot_path.is_file():
        return None
    try:
        existing = MeshVerificationSnapshot.from_dict(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        expected = build_mesh_verification_snapshot(
            spec,
            coordinator=coordinator_identity,
            policy=policy,
            generation=int(existing.generation),
            epoch=int(existing.epoch),
            issued_at_unix=int(existing.issued_at_unix),
            expires_at_unix=int(existing.expires_at_unix),
            stage_bindings=stage_bindings,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "persisted verification snapshot unusable (%s); minting a new chain",
            exc,
        )
        return None
    if existing.to_dict(include_signature=False) != expected.to_dict(
        include_signature=False
    ):
        # Loud on purpose: a resume attempt that cannot adopt means the
        # relaunch presents new verification terms and every validator
        # pin on this slot fails until re-pin - if the assignment did
        # not really change, an identity input (mesh id, stage secret,
        # anchors) regressed to instance-scoped.
        logger.warning(
            "persisted verification snapshot binds a different assignment; "
            "minting a new chain"
        )
        return None
    if not verify_mesh_verification_snapshot_signature(
        existing,
        expected_hotkey=spec.coordinator_hotkey,
        expected_epoch=int(existing.epoch),
    ):
        logger.warning(
            "persisted verification snapshot signature invalid; minting a "
            "new chain"
        )
        return None
    if int(existing.epoch) == int(epoch):
        logger.info(
            "verification snapshot resumed across relaunch: generation=%d "
            "epoch=%d hash=%s",
            int(existing.generation),
            int(existing.epoch),
            existing.snapshot_hash_hex()[:16],
        )
        return existing
    signed = rotate_mesh_verification_snapshot(
        existing, epoch=int(epoch), keypair=keypair
    )
    logger.info(
        "verification snapshot rotated on relaunch: generation=%d "
        "epoch=%d->%d",
        int(signed.generation),
        int(existing.epoch),
        int(epoch),
    )
    return signed


def _run_pool_chat(
    manager: str,
    secret: str,
    worker_id: str,
    config: PoolWorkerConfig,
    chat: dict[str, Any],
    *,
    runner: Any | None = None,
) -> None:
    """Run one operator chat against the local coordinator; report result out.

    Only the driver worker receives these. The coordinator binds all
    interfaces, and this local control path must dial 127.0.0.1 because the
    coordinator's internal-HMAC lane on validator routes only accepts
    loopback clients.
    """
    coordinator = f"http://127.0.0.1:{config.mesh_port}"
    chat_id = str(chat.get("chat_id", ""))
    try:
        request_expires_at_unix_ms = int(
            chat.get("request_expires_at_unix_ms", 0) or 0
        )
    except (TypeError, ValueError, OverflowError):
        request_expires_at_unix_ms = 0

    def remaining_request_s() -> float:
        if request_expires_at_unix_ms <= 0:
            return CHAT_COORDINATOR_TIMEOUT_S
        return max(
            0.0,
            (request_expires_at_unix_ms - int(time.time() * 1000)) / 1000.0,
        )

    chat_worker_session = str(chat.get("_worker_session_id", "") or "")
    base = {
        "pool_secret": secret,
        "worker_id": worker_id,
        "chat_id": chat_id,
        **(
            {"worker_session_id": chat_worker_session}
            if chat_worker_session
            else {}
        ),
    }
    stage_keypair = getattr(runner, "_stage_proof_keypair", None)
    stage_proof_key = str(
        getattr(runner, "_stage_proof_key_ss58", "") or ""
    )

    def manager_body(
        action: str,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {**base, **dict(fields or {})}
        if stage_keypair is not None and stage_proof_key:
            return _signed_worker_control_body(
                payload,
                action=action,
                keypair=stage_keypair,
                proof_key=stage_proof_key,
            )
        return payload
    payload = {
        "model": str(chat.get("model", "")),
        "messages": chat.get("messages") or [{"role": "user", "content": "Hello"}],
        "max_tokens": int(chat.get("max_tokens", 1024)),
        "temperature": 0,
        "stream": bool(chat.get("stream")),
        # Reasoning models plan in a hidden think block first; the operator
        # can turn that off for fast direct answers (Qwen3-style templates).
        **(
            {"chat_template_kwargs": {"enable_thinking": False}}
            if chat.get("thinking") is False
            else {}
        ),
        # Inline proof, same semantics as the vLLM miner: every response ships
        # with its GEMM proof generated AND verified before the reply is final.
        # On 30B-class models the inline proof costs tens of seconds (per-
        # request weight dequant); the manager serializes operator chats per
        # mesh so those can't pile up and wedge the coordinator.
    }
    # Sampler passthrough (private API / operator chats): the coordinator
    # normalizes these into the committed LIGHT sampled profile and binds
    # the applied controls into the signed receipt; hard-tier lanes reject
    # them there. Absent sampler = the deterministic greedy profile above.
    sampler = chat.get("sampler")
    if isinstance(sampler, Mapping):
        for name in ("temperature", "top_k", "top_p", "min_p", "seed"):
            value = sampler.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[name] = value
    internal_secret = ""
    snapshot_binding: dict[str, Any] = {}
    request_context_error = ""
    if runner is not None and hasattr(runner, "_local_request_context"):
        try:
            internal_secret, snapshot_binding = runner._local_request_context()
        except Exception as exc:
            request_context_error = (
                f"cannot load the mesh verification snapshot: {exc}"
            )[:500]
    expected_snapshot_hash = str(
        chat.get("verification_snapshot_hash", "") or ""
    )
    if expected_snapshot_hash:
        actual_snapshot_hash = str(
            snapshot_binding.get("verification_snapshot_hash", "") or ""
        )
        if not _COMMAND_DIGEST_RE.fullmatch(expected_snapshot_hash):
            request_context_error = "manager supplied an invalid snapshot hash"
        elif not secrets.compare_digest(
            expected_snapshot_hash,
            actual_snapshot_hash,
        ):
            request_context_error = (
                "local coordinator snapshot no longer matches the "
                "manager-pinned mesh snapshot"
            )
    if snapshot_binding:
        payload["verathos"] = snapshot_binding
    requested_chat_tier = str(chat.get("proof_tier", ""))
    if (
        expected_snapshot_hash and requested_chat_tier != "light"
    ) or requested_chat_tier == "hard":
        # The manager-pinned (validator-mode) operator lane is the self-test
        # and probe-gate path: it must exercise and assert the HARD relation,
        # so it opts out of the organic light tier explicitly. Organic
        # serving traffic never carries this upgrade request and stays light.
        # An operator may opt a plain chat back down to light for latency
        # comparisons; probes and self-tests never reach here with "light"
        # because _requested_chat_proof_tier refuses to downgrade them.
        # An EXPLICIT hard request upgrades on every lane, dev included:
        # the upgrade-only request is safe by construction and is how an
        # operator feels the hard-relation latency without a validator pool.
        payload.setdefault("verathos", {})["proof_tier"] = "hard"

    def post_control(
        path: str,
        *,
        action: str,
        fields: Mapping[str, Any],
        attempts: int = 4,
    ) -> dict[str, Any]:
        """Deliver one idempotent manager event with bounded retries."""

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                remaining = remaining_request_s()
                terminal_cleanup = path.endswith("/chat-result")
                if remaining <= 0 and not terminal_cleanup:
                    raise RuntimeError(
                        "operator chat expired before control delivery"
                    )
                # A terminal result is also the manager's cleanup ACK. Allow
                # a short bounded delivery after the user-visible
                # deadline so an expired running chat does not pin the mesh
                # slot; no further inference or chunk delivery is permitted.
                timeout_s = min(15.0, remaining) if remaining > 0 else 5.0
                response = post_json(
                    manager + path,
                    manager_body(action, fields),
                    timeout=timeout_s,
                )
                status = str(response.get("status", "") or "")
                if status == "ok":
                    return response
                if status in {"cancelled", "stale", "expired"}:
                    raise RuntimeError(f"operator chat was {status}")
                if status == "gap":
                    raise RuntimeError(
                        "manager rejected an out-of-order chat event "
                        f"(expected seq {response.get('expected_seq')})"
                    )
                raise RuntimeError(f"manager rejected chat event: {status or 'unknown'}")
            except Exception as exc:
                last_error = exc
                if any(
                    terminal in str(exc)
                    for terminal in ("cancelled", "stale", "expired")
                ):
                    raise
                can_retry = (
                    attempt + 1 < attempts
                    and (
                        remaining_request_s() > 0
                        or (path.endswith("/chat-result") and attempt == 0)
                    )
                )
                if not can_retry:
                    break
                time.sleep(0.1 * (2**attempt))
        assert last_error is not None
        raise RuntimeError(
            f"could not deliver chat event after {attempts} attempts: "
            f"{last_error}"
        )

    def _apply_final(out: dict[str, Any], response: dict[str, Any], acc: str) -> None:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        mesh_meta = response.get("verathos_mesh", {})
        proof_receipts = mesh_meta.get("proof_receipts", [])
        # Two real receipt families: GGML op receipts carry a numeric
        # stage_index, opaque stage receipts a string stage_id. Counting only
        # stage_id undercounted live proofs to zero, which the validator-mode
        # manager assertion then refused (caught in live certification).
        proof_stage_ids = set()
        for item in proof_receipts:
            if not isinstance(item, Mapping):
                continue
            if item.get("stage_index") is not None:
                try:
                    proof_stage_ids.add(f"index:{int(item['stage_index'])}")
                    continue
                except (TypeError, ValueError):
                    pass
            if str(item.get("stage_id", "")):
                proof_stage_ids.add(f"id:{item['stage_id']}")
        # Reasoning models: the coordinator's final message carries the
        # thinking segment as reasoning_content. Surface it on the pool
        # result so the private OpenAI API can pass it through.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            out["reasoning_content"] = reasoning
        out.update(
            {
                "content": message.get("content", acc),
                "verified": mesh_meta.get("verified"),
                "receipt_verified": mesh_meta.get("proof_receipt_verified"),
                "receipts": mesh_meta.get("proof_receipt_count"),
                "proof_stages": len(proof_stage_ids),
                "proof_mode": mesh_meta.get("proof_mode"),
                "proof_receipt_root": mesh_meta.get("proof_receipt_root"),
                "verification_snapshot_hash": mesh_meta.get(
                    "verification_snapshot_hash"
                ),
                "mesh_response_commitment_hash": mesh_meta.get(
                    "mesh_response_commitment_hash"
                ),
                "deferred": mesh_meta.get("proof_deferred"),
                "deferred_obligation": mesh_meta.get("proof_deferred_obligation"),
                "usage": response.get("usage", {}),
            }
        )
        # The engine's own decode rate. The dashboard used to derive tok/s from
        # wall time INCLUDING the proving phase, understating a 40 tok/s engine
        # as ~30; the authoritative number comes from llama's timings.
        timings = response.get("timings") or {}
        if timings.get("predicted_per_second"):
            out["engine_tps"] = round(float(timings["predicted_per_second"]), 1)
        if timings.get("prompt_per_second"):
            out["prompt_tps"] = round(float(timings["prompt_per_second"]), 1)

    # Per-chat coordinator POST budget: probes carry a longer one than the
    # interactive default (see handle_chat), bounded by the probe ceiling.
    try:
        coordinator_timeout_s = float(chat.get("coordinator_timeout_s", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        coordinator_timeout_s = 0.0
    if not math.isfinite(coordinator_timeout_s) or coordinator_timeout_s <= 0:
        coordinator_timeout_s = CHAT_COORDINATOR_TIMEOUT_S
    coordinator_timeout_s = min(coordinator_timeout_s, PROBE_DEADLINE_MAX_S)

    out = dict(base)
    if not payload["stream"]:
        chat_started = time.monotonic()
        try:
            if request_context_error:
                raise RuntimeError(request_context_error)
            remaining = remaining_request_s()
            if remaining <= 0:
                raise TimeoutError("operator chat expired before inference started")
            response = post_json(
                coordinator + "/v1/chat/completions",
                payload,
                timeout=min(coordinator_timeout_s, remaining),
                internal_auth_secret=internal_secret,
            )
            _apply_final(out, response, "")
        except Exception as exc:
            out["error"] = str(exc)[:500]
        # Whole-request wall time; a non-streaming POST has no first-delta,
        # so ttft_s stays the -1.0 sentinel.
        out["ttft_s"] = -1.0
        out["total_s"] = round(time.monotonic() - chat_started, 3)
        try:
            post_control(
                "/v1/pool/chat-result",
                action="chat-result",
                fields={**out, "seq": 1},
            )
        except Exception as exc:
            logger.error("POOLCHAT-FINAL-DELIVERY %s: %s", chat_id, exc)
        return

    # Streaming: read the coordinator's SSE, forward each token delta to the
    # manager immediately, then send the proof-bearing final on the "done"
    # event (which carries verathos_mesh + the full response).
    acc = ""
    pending = ""
    think_pending = {"s": "", "t": time.monotonic()}
    acked_seq = 0
    last_flush = time.monotonic()
    # One-line latency telemetry per chat: where TTFT actually goes.
    queued_ns = int(chat.get("queued_unix_ns", 0) or 0)
    t_start = time.monotonic()
    pickup_s = (time.time_ns() - queued_ns) / 1e9 if queued_ns else -1.0
    t_conn = t_first = t_posted = -1.0
    t_gen_done: float | None = None

    def deliver_stream_event(fields: Mapping[str, Any]) -> None:
        nonlocal acked_seq
        seq = acked_seq + 1
        post_control(
            "/v1/pool/chat-chunk",
            action="chat-chunk",
            fields={**dict(fields), "seq": seq},
        )
        acked_seq = seq

    def flush_pending() -> None:
        nonlocal pending, last_flush
        if not pending:
            return
        deliver_stream_event({"delta": pending})
        pending = ""
        last_flush = time.monotonic()

    def flush_thinking() -> None:
        if not think_pending["s"]:
            return
        deliver_stream_event({"thinking": think_pending["s"]})
        think_pending["s"] = ""
        think_pending["t"] = time.monotonic()

    try:
        if request_context_error:
            raise RuntimeError(request_context_error)
        remaining = remaining_request_s()
        if remaining <= 0:
            raise TimeoutError("operator chat expired before inference started")
        request_body = json.dumps(payload, sort_keys=True).encode()
        request_headers = {"Content-Type": "application/json"}
        if internal_secret:
            from verallm.mesh.http_auth import sign_internal_http_request

            request_headers.update(
                sign_internal_http_request(
                    secret=internal_secret,
                    method="POST",
                    path="/v1/chat/completions",
                    body=request_body,
                )
            )
        req = urllib.request.Request(
            coordinator + "/v1/chat/completions",
            data=request_body,
            headers=request_headers,
        )
        with urllib.request.urlopen(
            req,
            timeout=min(coordinator_timeout_s, remaining),
        ) as resp:
            t_conn = time.monotonic() - t_start
            for raw in resp:
                if remaining_request_s() <= 0:
                    raise TimeoutError("operator chat exceeded its request deadline")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    # NEVER break here: the proof-bearing final ("event: done")
                    # is what ends a chat, and a stray upstream [DONE] slipping
                    # through ahead of it would make us drop the proof metadata
                    # (the intermittent verified=None). The coordinator closes
                    # the connection after its final, which ends this loop.
                    continue
                try:
                    obj = json.loads(body)
                except Exception:
                    # Never drop a line silently: the proof-bearing final event
                    # is the largest line in the stream, and losing it returns
                    # the text with verified=None — log enough to diagnose.
                    logger.warning(
                        "POOLCHAT-PARSE-FAIL %s len=%d head=%r",
                        chat_id, len(body), body[:120],
                    )
                    continue
                if obj.get("event") == "done":
                    # The coordinator's terminal event nests the OpenAI body in
                    # "response" and puts the proof in a sibling "verathos_mesh";
                    # fold them together so _apply_final finds both. This event
                    # ends the chat (never rely on a trailing [DONE] marker).
                    flush_thinking()
                    flush_pending()
                    response = dict(obj.get("response", {}) or {})
                    response.setdefault("verathos_mesh", obj.get("verathos_mesh", {}))
                    _apply_final(out, response, acc)
                    if t_gen_done is not None:
                        # Measured proving tail (generation end to the
                        # proof-bearing final). The subtraction estimate the
                        # chat falls back to mislabels per-token relay
                        # overhead as proving (a Mac reply showed 0.9s
                        # "proof" while the receipts took 25ms).
                        out["proof_wall_s"] = round(
                            time.monotonic() - t_gen_done, 3
                        )
                    break
                if obj.get("event") == "error" or (
                    "error" in obj and "choices" not in obj
                ):
                    # A mid-stream failure (e.g. proof verification error) MUST
                    # surface as an error — swallowing it would return the text
                    # as if it were fine, silently unverified.
                    err = obj.get("error")
                    if isinstance(err, dict):
                        err = err.get("message") or json.dumps(err)
                    flush_thinking()
                    flush_pending()
                    out["error"] = str(err or "mesh stream error")[:500]
                    continue
                if obj.get("timings") is not None:
                    # llama.cpp's final timing chunk: generation is complete and
                    # the proof is now being generated + verified — surface that
                    # phase so the operator sees why the badge hasn't landed yet.
                    # Also the only place the engine's true decode rate appears
                    # in the stream (the coordinator's done event omits it).
                    t = obj["timings"]
                    if t.get("predicted_per_second"):
                        out["engine_tps"] = round(float(t["predicted_per_second"]), 1)
                    if t.get("prompt_per_second"):
                        out["prompt_tps"] = round(float(t["prompt_per_second"]), 1)
                    t_gen_done = time.monotonic()
                    flush_thinking()
                    flush_pending()
                    deliver_stream_event({"phase": "proving"})
                d0 = (obj.get("choices") or [{}])[0].get("delta", {})
                # Thinking models (Qwen3.6 etc.): llama-server routes the
                # reasoning stream into reasoning_content, NOT content. If we
                # drop it, the operator watches an empty box for the whole
                # think phase and the chat looks dead. Relay it as a separate
                # "thinking" stream so the dashboard renders it dimmed.
                think = d0.get("reasoning_content")
                if think:
                    if t_first < 0:
                        t_first = time.monotonic() - t_start
                    think_pending["s"] += think
                    if time.monotonic() - think_pending["t"] >= 0.25:
                        deliver_stream_event(
                            {"thinking": think_pending["s"]}
                        )
                        think_pending["s"] = ""
                        think_pending["t"] = time.monotonic()
                delta = d0.get("content")
                if delta:
                    # Preserve upstream order: any final reasoning fragment
                    # must reach the browser before the first answer token.
                    flush_thinking()
                    if t_first < 0:
                        t_first = time.monotonic() - t_start
                        flush_pending()  # ship the FIRST token immediately
                        acc += delta
                        pending += delta
                        flush_pending()
                        t_posted = time.monotonic() - t_start
                        continue
                    acc += delta
                    pending += delta
                    # Coalesce deltas into ~50ms batches so the browser sees
                    # smooth streaming without one HTTP round-trip per token
                    # (which otherwise throttles the relay to the POST rate).
                    if time.monotonic() - last_flush >= 0.05:
                        flush_pending()
        flush_thinking()
        flush_pending()
        out.setdefault("content", acc)
        if out.get("verified") is None and not out.get("error"):
            # The stream ended without the coordinator's proof-bearing final.
            # Say so explicitly — a silent "unverified" badge misreads as a
            # proof failure when the truth is lost metadata.
            logger.warning("POOLCHAT-NO-FINAL %s", chat_id)
            out["error"] = (
                "stream ended before the proof-bearing final event; "
                "the response text is complete but its proof metadata was lost"
            )
        logger.info(
            "POOLCHAT-TIMING %s pickup=%.2fs coord_headers=%.2fs "
            "first_token=%.2fs first_relayed=%.2fs total=%.2fs verified=%s",
            chat_id, pickup_s, t_conn, t_first, t_posted,
            time.monotonic() - t_start, out.get("verified"),
        )
    except Exception as exc:
        for flush in (flush_thinking, flush_pending):
            try:
                flush()
            except Exception:
                pass
        out["error"] = str(exc)[:500]
    # Timing telemetry rides the final so probes and the dashboard see it:
    # ttft_s is the first visible delta (thinking or content) on the driver's
    # loopback dial, -1.0 when none arrived.
    out["ttft_s"] = round(t_first, 3) if t_first >= 0 else -1.0
    out["total_s"] = round(time.monotonic() - t_start, 3)
    if pickup_s >= 0:
        out["pickup_s"] = round(pickup_s, 3)
    try:
        post_control(
            "/v1/pool/chat-result",
            action="chat-result",
            fields={**out, "seq": acked_seq + 1},
        )
    except Exception as exc:
        logger.error("POOLCHAT-FINAL-DELIVERY %s: %s", chat_id, exc)


def _prewarm_proof_cache(
    cfg: "PoolWorkerConfig",
    model_id: str,
    stop_event: threading.Event,
    *,
    progress: Any | None = None,
    done: threading.Event | None = None,
    result: dict[str, Any] | None = None,
    entry_override: Mapping[str, Any] | None = None,
) -> None:
    """Build the driver's proof-weight blobs before it reports serving.

    Proof challenges sample different tensors per request; without a warm
    cache each newly drawn tensor pays a multi-second dequant DURING the
    chat (a 35B's first proofs took ~20s, which reads as broken). Pre-
    building every provable blob from the local GGUF moves that cost off
    the request path. A second idempotence pass certifies convergence before
    the caller may publish serving. Content-addressed writes make racing the
    serve process's own build harmless. Skips file-less members.
    """
    try:
        if entry_override is not None:
            entry = dict(entry_override)
        else:
            try:
                entry = cfg.catalog_entry(model_id)
            except ValueError:
                if result is not None:
                    result.update(converged=True, skipped_fileless=True)
                return  # file-less member: its child proof server gates warmup
        stats = _prewarm_records(entry, stop_event, progress, model_id)
        if result is not None:
            result.update(converged=True, stats=stats)
    except Exception as exc:
        if result is not None:
            result.update(converged=False, error=str(exc)[:500])
        logger.warning("proof cache prewarm failed for %s: %s", model_id, exc)
    finally:
        if done is not None:
            done.set()


def _prewarm_records(
    entry: Mapping[str, Any],
    stop_event: threading.Event,
    progress: Any | None,
    model_id: str,
) -> dict[str, int]:
    from verallm.mesh.gguf_manifest import (
        load_gguf_tensor_manifest,
        prewarm_proof_weight_cache_to_convergence,
    )

    if stop_event.is_set():
        raise RuntimeError("proof cache prewarm was cancelled")
    manifest = load_gguf_tensor_manifest(entry["manifest"])
    warm_progress = (
        (lambda _detail: progress("warming proofs"))
        if progress is not None
        else None
    )
    stats = prewarm_proof_weight_cache_to_convergence(
        manifest,
        progress=warm_progress,
    )
    if stop_event.is_set():
        raise RuntimeError("proof cache prewarm was cancelled")
    logger.info(
        "proof-weight cache prewarmed to convergence for %s: %s",
        model_id,
        stats,
    )
    return stats


def _detect_backend_arch() -> tuple[str, str]:
    """Detect the compute backend and GPU architecture of this machine.

    Returns ("cuda", "sm89"), ("metal", "apple"), or ("cpu", "") when no GPU
    stack is present. The arch label keys the canonical build directory so a
    box never runs a binary compiled for a different GPU generation (an
    arch-mismatched hand-picked binary crash-looped a 4090 362 times with
    GGML_ASSERT(ok) while the dashboard showed "driving").
    """
    if sys.platform == "darwin":
        return "metal", "apple"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if out:
            major, _, minor = out[0].strip().partition(".")
            return "cuda", f"sm{int(major)}{int(minor or 0)}"
    except Exception:
        pass
    return "cpu", ""


def _mesh_patch_sha(repo_root: Path, backend: str) -> str:
    """Digest every patch this backend's build applies, in apply order.

    The build manifest compares this against the recorded value, so adding
    a patch (for example the streaming anchors) invalidates stale binaries
    instead of silently serving a runtime without the new capture.
    """

    names = (
        [
            "0002-verathos-proof-capture-metal.patch",
        ]
        if backend == "metal"
        else [
            "0001-verathos-proof-capture-cuda-cpu-rpc-server.patch",
            "0003-verathos-streaming-execution-anchors.patch",
            "0005-verathos-mmid-decode-intra-parity.patch",
            "0007-verathos-rpc-foreign-view-serialization.patch",
            "0008-verathos-wildcard-op-arming.patch",
            "0009-verathos-name-keyed-op-arming.patch",
        ]
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(
            (Path(repo_root) / "patches" / "llama.cpp" / name).read_bytes()
        )
    return digest.hexdigest()


def _mesh_runtime_build_inputs(repo_root: Path, backend: str) -> dict[str, str]:
    patch_root = Path(repo_root) / "patches" / "llama.cpp"
    return {
        "patch_sha256": _mesh_patch_sha(repo_root, backend),
        "upstream_base": (patch_root / "UPSTREAM_BASE.txt").read_text().strip(),
        "build_script_sha256": hashlib.sha256(
            (patch_root / "build.sh").read_bytes()
        ).hexdigest(),
    }


def _auto_backend_binaries(
    cfg: "PoolWorkerConfig", status_cb: Any
) -> tuple[str, str]:
    """Resolve (llama-server, verathos-rpc-server) for THIS machine's GPU.

    Reuses the canonical per-arch build when its recorded patch hash matches
    the patch shipped in the repo; otherwise builds one via
    patches/llama.cpp/build.sh (one-time per box per patch version) and
    records a build manifest so stale builds are detected instead of served.
    """
    backend, arch = _detect_backend_arch()
    if backend == "cpu":
        raise RuntimeError(
            "no GPU backend detected (nvidia-smi missing and not macOS); "
            "pass --llama-server-binary/--rpc-worker-binary explicitly"
        )
    label = f"{backend}-{arch}"
    src = Path.home() / ".cache" / "verathos-mesh-runtime" / f"auto-{label}"
    bin_dir = src / f"build-verathos-{backend}" / "bin"
    llama = bin_dir / "llama-server"
    rpc = bin_dir / "verathos-rpc-server"
    manifest_path = bin_dir / "verathos-build.json"
    want = {
        **_mesh_runtime_build_inputs(cfg.repo_root, backend),
        "backend": backend,
        "arch": arch,
    }
    if llama.is_file() and rpc.is_file() and manifest_path.is_file():
        try:
            have = json.loads(manifest_path.read_text())
        except Exception:
            have = {}
        if all(have.get(k) == v for k, v in want.items()):
            logger.info("backend build ready: %s (%s)", bin_dir, label)
            return str(llama), str(rpc)
        logger.info(
            "backend build at %s is stale (patch or arch changed); rebuilding",
            bin_dir,
        )
    status_cb(f"building backend ({label}, one-time)")
    build_sh = Path(cfg.repo_root) / "patches" / "llama.cpp" / "build.sh"
    log_path = Path(cfg.workdir) / "backend-build.log"
    Path(cfg.workdir).mkdir(parents=True, exist_ok=True)
    logger.info("building patched llama.cpp for %s; log: %s", label, log_path)
    build_command = [
        "bash",
        str(build_sh),
        "--backend",
        backend,
        "--src",
        str(src),
    ]
    if backend == "cuda" and arch.startswith("sm"):
        build_command.extend(["--cuda-architectures", arch.removeprefix("sm")])
    with open(log_path, "ab") as log:
        rc = subprocess.run(
            build_command,
            stdout=log, stderr=log,
        ).returncode
    if rc != 0 or not llama.is_file() or not rpc.is_file():
        raise RuntimeError(
            f"backend auto-build failed (exit {rc}) for {label}; see {log_path}"
        )
    manifest_path.write_text(json.dumps(want))
    logger.info("backend build complete: %s", bin_dir)
    return str(llama), str(rpc)


def _pick_reachable_manager_endpoint(advertised: str, pool_id: str) -> str:
    """The advertised manager endpoint, or its loopback twin if only that
    answers for OUR pool.

    A worker on the coordinator machine dials the pool's PUBLIC address
    from the join token, and many NATs cannot hairpin a box to its own
    public IP — the worker then retries a join forever against an
    endpoint it can never reach while the manager listens fine on
    loopback (it binds 0.0.0.0). Remote workers are unaffected: their
    loopback probe answers for no pool (or the wrong one) and the
    advertised endpoint stays in charge.
    """

    from urllib.parse import urlparse

    def _healthy(endpoint: str) -> bool:
        try:
            with urllib.request.urlopen(
                endpoint.rstrip("/") + "/healthz",
                timeout=3.0,
                context=(
                    ssl._create_unverified_context()
                    if endpoint.startswith("https://")
                    else None
                ),
            ) as resp:
                payload = json.loads(resp.read() or b"{}")
        except Exception:
            return False
        return str(payload.get("pool_id", "") or "") == pool_id

    if _is_loopback(advertised) or _healthy(advertised):
        return advertised
    parsed = urlparse(advertised.rstrip("/"))
    loopback = (
        f"{parsed.scheme or 'http'}://127.0.0.1:{parsed.port or 9500}"
    )
    if _healthy(loopback):
        logger.info(
            "pool manager %s is unreachable from this box (hairpin NAT?); "
            "using %s, which answers for pool %s",
            advertised,
            loopback,
            pool_id,
        )
        return loopback
    return advertised


_PR_SET_CHILD_SUBREAPER = 36


def enable_child_subreaper() -> bool:
    """Make this daemon adopt orphaned descendants (Linux only).

    Rented mesh containers run ``sleep infinity`` as PID 1, which never
    reaps: when a serve process is killed mid-roll before collecting its
    llama-server/rpc children, those children re-parent to PID 1 and stay
    ``<defunct>`` forever, accumulating across relaunches toward PID
    exhaustion. With the worker
    daemon registered as subreaper they re-parent HERE instead, where
    ``reap_untracked_zombies`` collects their statuses each beat.

    macOS has no subreaper concept and launchd already reaps orphans, so
    non-Linux platforms skip silently.
    """

    if sys.platform != "linux":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return int(libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)) == 0
    except Exception:
        return False


def reap_untracked_zombies(
    tracked_pids: Callable[[], Iterable[int]],
    pending: frozenset[int] = frozenset(),
) -> tuple[list[int], frozenset[int]]:
    """Reap dead children that no live ``subprocess.Popen`` owns.

    A bare ``os.waitpid(-1, WNOHANG)`` loop would STEAL exit statuses
    from the Popen children this daemon tracks (serve processes in the
    runner registry, audit workspace holders, bench children), leaving
    their ``poll()``/``returncode`` permanently wedged on ``None``.
    Instead each dead child is PEEKED with ``os.waitid(..., WNOWAIT)`` -
    the zombie stays reapable by its true owner - and is only reaped
    here when BOTH hold:

    - the pid is not owned by any tracked Popen (``tracked_pids()`` is
      re-collected on every iteration so a Popen created mid-sweep is
      still honoured), and
    - the pid was already pending on the PREVIOUS sweep. A child of an
      untracked in-flight waiter (``subprocess.run`` inside a helper
      thread) is a zombie only for the microseconds before that waiter
      collects it; stealing it in that window would misreport a failed
      git/build as success (CPython maps ECHILD in ``Popen._try_wait``
      to returncode 0). A zombie that survives a whole sweep interval
      has no waiter left.

    Returns ``(reaped_pids, pending)``; feed ``pending`` back into the
    next call. ``waitid(P_ALL, WNOWAIT)`` re-reports the same child
    until someone reaps it, so the loop stops at the first child it must
    leave alone - anything queued behind it is drained by later sweeps.
    ``ChildProcessError`` (no children at all) ends the sweep quietly.
    """

    reaped: list[int] = []
    next_pending: set[int] = set()
    if not hasattr(os, "waitid"):  # pragma: no cover - non-POSIX
        return reaped, frozenset()
    while True:
        try:
            info = os.waitid(
                os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT
            )
        except ChildProcessError:
            break  # no children at all
        except OSError:
            break
        if info is None:
            break  # children exist, none dead
        pid = int(info.si_pid)
        try:
            owned = pid in {int(p) for p in tracked_pids()}
        except Exception:
            break
        if owned:
            # A live Popen owns this exit status; its poll()/wait()
            # must be the one to observe it.
            break
        if pid not in pending:
            next_pending.add(pid)
            break
        try:
            os.waitpid(pid, 0)  # already a zombie: returns immediately
        except (ChildProcessError, OSError):
            break
        reaped.append(pid)
    return reaped, frozenset(next_pending)


def pool_worker_loop(
    config: PoolWorkerConfig,
    *,
    runner: Any | None = None,
    stop_event: threading.Event | None = None,
    max_beats: int | None = None,
    on_join_info: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Join the pool and execute manager commands until stopped.

    ``on_join_info`` receives the join response once; token-only workers
    use it to learn the pool's chain coordinates (subtensor network,
    netuid) without any local flag.
    """

    # One daemon per workdir, enforced with an exclusive lock held for the
    # process lifetime. Two daemons on one workdir race their control joins
    # (each rotates the session pin, 403-ing the other's delegated signing)
    # and double every capacity-audit workload (guaranteed OOM beside a
    # resident model) —
    # daemons for 30 minutes of silent no_show windows.
    lock_path = Path(config.workdir) / "pool-worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    worker_lock_handle = open(lock_path, "w")
    try:
        import fcntl

        fcntl.flock(worker_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError(
            f"another pool worker daemon already owns {config.workdir} "
            f"(lock {lock_path}): {exc}"
        ) from None
    worker_lock_handle.write(f"{os.getpid()}\n")
    worker_lock_handle.flush()

    # Orphaned grandchildren (llama-server/rpc whose serve was killed
    # mid-roll) re-parent to this daemon instead of the container's
    # non-reaping PID 1; the per-beat zombie sweep then collects them.
    if enable_child_subreaper():
        logger.info("pool worker daemon registered as child subreaper")

    # Hold the capacity-audit workspace on every local CUDA device BEFORE
    # any serve launches: llama.cpp grows into all free VRAM over time, so
    # only memory that is already allocated reliably survives to a window.
    if str(getattr(config, "subtensor_network", "") or "").strip():
        try:
            from verallm.mesh.capacity_audit_worker import workspace_holders

            devices = []
            for i, dev in enumerate(
                str(getattr(config, "rpc_device", "") or "").split(",")
            ):
                name = dev.strip().upper()
                if not name.startswith("CUDA"):
                    continue
                # "CUDA1" names the runtime device index, which is the
                # worker-LOCAL index; positions only coincide for
                # zero-based lists.
                suffix = name[4:]
                devices.append(int(suffix) if suffix.isdigit() else i)
            if devices:
                workspace_holders().ensure(devices)
        except Exception as exc:
            logger.warning(
                "audit workspace holders unavailable at daemon start: %s",
                exc,
            )

    runner = runner or LocalMeshRunner(config)
    # Previous-generation reaping (process groups + own-port sweep) runs
    # ONLY here, under the daemon flock: any LocalMeshRunner construction
    # used to sweep, and a TEST runner configured with production-like
    # port numbers could otherwise terminate a daemon's listeners.
    reap = getattr(runner, "_reap_previous_generation", None)
    if callable(reap):
        reap()
    stop_event = stop_event or threading.Event()
    manager = _pick_reachable_manager_endpoint(
        config.token.manager_endpoint.rstrip("/"),
        config.token.pool_id,
    )
    secret = config.token.pool_secret
    stage_keypair = getattr(runner, "_stage_proof_keypair", None)
    stage_proof_key = str(
        getattr(runner, "_stage_proof_key_ss58", "") or ""
    )
    requested_worker_id = str(config.worker_id or "").strip()
    if not requested_worker_id:
        if stage_proof_key:
            requested_worker_id = (
                "w-"
                + hashlib.sha256(stage_proof_key.encode("utf-8")).hexdigest()[:10]
            )
        else:
            requested_worker_id = "w-" + uuid.uuid4().hex[:10]
    if not _WORKER_ID_RE.fullmatch(requested_worker_id):
        raise ValueError("worker_id must be a bounded protocol identifier")

    worker_run_id = uuid.uuid4().hex
    worker_session_id = secrets.token_hex(32)
    command_journal_path = (
        Path(config.workdir) / POOL_WORKER_COMMAND_JOURNAL_FILE
    )
    command_journal_lock = threading.RLock()
    if command_journal_path.exists():
        try:
            command_journal = json.loads(
                read_owner_only_text(
                    command_journal_path,
                    label="pool worker command journal",
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "pool worker command journal is unreadable; refusing to "
                "risk duplicate command execution"
            ) from exc
        journal_worker_id = str(command_journal.get("worker_id", "") or "")
        if journal_worker_id != requested_worker_id:
            # Say whose workdir this is and how to resolve it: the bare
            # refusal left an operator with no way to tell which id was
            # wrong, and the two remedies (reuse the id, or give the new
            # worker its own workdir) are not guessable from the message.
            raise RuntimeError(
                f"workdir {config.workdir} belongs to worker id "
                f"'{journal_worker_id}', not '{requested_worker_id}'; either "
                f"start this worker as '{journal_worker_id}' or point it at a "
                "fresh workdir (one workdir = one worker identity)"
            )
        if not isinstance(command_journal.get("commands"), dict):
            raise RuntimeError("pool worker command journal is malformed")
    else:
        command_journal = {
            "version": 1,
            "worker_id": requested_worker_id,
            "commands": {},
        }

    def _save_command_journal_locked() -> None:
        commands = command_journal["commands"]
        if len(commands) > 512:
            # Only terminal completions may be aged out. Dropping a running,
            # phase-complete, or report-pending record can turn manager
            # redelivery after a restart into duplicate side effects.
            protected = [
                command_id
                for command_id, entry in commands.items()
                if str((entry or {}).get("state", "") or "") != "completed"
            ]
            if len(protected) > 512:
                raise RuntimeError(
                    "pool worker command journal has too many unfinished commands"
                )
            completed = sorted(
                (
                    command_id
                    for command_id in commands
                    if command_id not in protected
                ),
                key=lambda command_id: float(
                    (commands[command_id] or {}).get("updated_at_unix", 0.0)
                    or 0.0
                ),
                reverse=True,
            )
            keep = [*protected, *completed[: 512 - len(protected)]]
            command_journal["commands"] = {
                command_id: commands[command_id] for command_id in keep
            }
        write_owner_only_json(command_journal_path, command_journal)

    def _claim_journal_command(
        command: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        command_id = str(command.get("command_id", "") or "")
        command_digest = str(command.get("command_digest", "") or "")
        with command_journal_lock:
            existing = command_journal["commands"].get(command_id)
            if isinstance(existing, Mapping):
                if not secrets.compare_digest(
                    str(existing.get("command_digest", "") or ""),
                    command_digest,
                ):
                    return "conflict", None
                state = str(existing.get("state", "") or "")
                report = existing.get("report")
                if state in {"report_pending", "completed"} and isinstance(
                    report,
                    Mapping,
                ):
                    if (
                        state == "report_pending"
                        and not bool(existing.get("report_terminal", True))
                        and str(existing.get("worker_run_id", "") or "")
                        != worker_run_id
                    ):
                        return "interrupted", None
                    return state, dict(report)
                if state == "phase_completed" and isinstance(report, Mapping):
                    if str(existing.get("worker_run_id", "") or "") == worker_run_id:
                        return state, dict(report)
                    return "interrupted", None
                if state == "running":
                    if str(existing.get("worker_run_id", "") or "") == worker_run_id:
                        return "running", None
                    return "interrupted", None
            command_journal["commands"][command_id] = {
                "command_digest": command_digest,
                "action": str(command.get("action", "") or ""),
                "mesh_key": str(command.get("mesh_key", "") or ""),
                "member_count": int(command.get("member_count", 0) or 0),
                "state": "running",
                "worker_run_id": worker_run_id,
                "updated_at_unix": time.time(),
            }
            _save_command_journal_locked()
            return "new", None

    def _journal_report_pending(
        command_id: str,
        report: Mapping[str, Any],
    ) -> None:
        with command_journal_lock:
            existing = command_journal["commands"].get(command_id)
            if not isinstance(existing, dict):
                raise RuntimeError("cannot persist report for an unknown command")
            existing.update(
                {
                    "state": "report_pending",
                    "report": dict(report),
                    "report_terminal": not (
                        str(existing.get("action", "") or "") == "drive"
                        and str(report.get("event", "") or "") == "drive_ready"
                        and int(existing.get("member_count", 0) or 0) > 1
                    ),
                    "updated_at_unix": time.time(),
                }
            )
            _save_command_journal_locked()

    def _journal_report_delivered(
        command_id: str,
        *,
        stale: bool = False,
    ) -> None:
        with command_journal_lock:
            existing = command_journal["commands"].get(command_id)
            if not isinstance(existing, dict):
                return
            existing.update(
                {
                    "state": (
                        "completed"
                        if stale or bool(existing.get("report_terminal", True))
                        else "phase_completed"
                    ),
                    "updated_at_unix": time.time(),
                }
            )
            _save_command_journal_locked()

    def worker_request(
        action: str,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "pool_secret": secret,
            "worker_id": requested_worker_id,
            "worker_session_id": worker_session_id,
            **dict(fields or {}),
        }
        if stage_keypair is not None and stage_proof_key:
            return _signed_worker_control_body(
                payload,
                action=action,
                keypair=stage_keypair,
                proof_key=stage_proof_key,
            )
        return payload

    def manager_call(action: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        """One authenticated worker->manager call, decoded."""

        def _attempt(timeout_s: float) -> dict[str, Any]:
            return post_json(
                f"{manager}/v1/pool/{action}",
                # Rebuild on every retry so signed worker auth carries a fresh
                # nonce/timestamp while the coordinator-sign digest is stable.
                worker_request(action, fields),
                timeout=timeout_s,
                # Keep-alive kills the per-call TCP+TLS handshake (measured
                # 326ms/call from a WAN worker, ~218ms TLS); falls back to
                # the pinned urllib opener when no context is registered.
                keepalive=True,
                keepalive_fallback=False,
            )

        if action == "coordinator-sign":
            attempts, timeout_s, delays = coordinator_sign_retry_profile(
                fields.get("purpose")
            )
            return coordinator_sign_request_with_retry(
                _attempt,
                attempts=attempts,
                attempt_timeout_s=timeout_s,
                retry_delays_s=delays,
            )
        return _attempt(20.0)

    # Lets a driver that holds no coordinator wallet obtain the coordinator
    # signatures a subnet mesh owes the validator. See the module docstring
    # of verallm/mesh/delegated_signing.py for why this is not a general
    # signing oracle.
    runner._coordinator_sign_request = manager_call
    # The same channel in file form, for the serve subprocess a driving
    # worker spawns (it signs receipts and identity challenges itself).
    runner._delegation_context = {
        "manager_endpoint": manager,
        "pool_secret": secret,
        "worker_id": requested_worker_id,
        "worker_session_id": worker_session_id,
        # TLS trust for the serve subprocess: the daemon installed its
        # pinned opener at join, but the subprocess starts bare and an
        # https manager with a self-signed cert then fails every delegated
        # signing call.
        "pool_id": config.token.pool_id,
        "manager_ca_sha256": str(
            getattr(config.token, "manager_ca_sha256", "") or ""
        ),
        "manager_ca_file": str(config.manager_ca_file or ""),
    }

    def deliver_worker_report(
        fields: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Retry one idempotent lifecycle report across manager interruptions."""

        deadline = time.monotonic() + WORKER_REPORT_DELIVERY_MAX_S
        last_error: Exception | None = None
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                response = post_json(
                    manager + "/v1/pool/report",
                    worker_request("report", fields),
                    timeout=10.0,
                )
                response_status = str(response.get("status", "") or "")
                if response_status in {"ok", "stale"}:
                    return dict(response)
                last_error = RuntimeError(
                    "manager rejected worker report: "
                    f"{response_status or 'unknown'}"
                )
            except Exception as exc:
                last_error = exc
            stop_event.wait(CHAT_POLL_RETRY_S)
        logger.error(
            "worker lifecycle report was not delivered before its deadline: %s",
            last_error or "worker stopped",
        )
        return None

    # AUTO backend resolution: join immediately (the dashboard shows this
    # worker with a "building backend" status) and resolve/build the patched
    # llama.cpp for this box's GPU arch in the background. The worker refuses
    # nothing by hand — its non-idle status keeps the manager from placing a
    # mesh on it until the build lands; a failed build heartbeats the error.
    backend_ready = {
        "done": bool(config.llama_server_binary and config.rpc_worker_binary),
        "note": "",
        "error": "",
    }
    if not backend_ready["done"]:
        def _resolve_backend() -> None:
            try:
                llama_bin, rpc_bin = _auto_backend_binaries(
                    config,
                    lambda text: backend_ready.__setitem__("note", text),
                )
                config.llama_server_binary = llama_bin
                config.rpc_worker_binary = rpc_bin
            except Exception as exc:
                backend_ready["error"] = f"backend unavailable: {exc}"
                logger.error("%s", backend_ready["error"])
            finally:
                backend_ready["done"] = True

        threading.Thread(target=_resolve_backend, daemon=True).start()
    def join_payload() -> dict[str, Any]:
        # Rebuilt per call: an auto-fetched model appends to config.catalog and
        # must be advertised on the next (re)join. hf_repo/hf_files/layers feed
        # the manager's model_registry so OTHER machines can auto-fetch too;
        # local file paths never leave this box.
        return worker_request(
            "join",
            {
            "capability": {
                "gpu_name": config.gpu_name,
                "vram_gb": int(config.vram_gb),
                "gpu_count": config.gpu_count,
                "per_gpu_vram_gb": config.gpu_split_weights,
                "gpu_names": [str(n) for n in config.gpu_names],
                "rpc_device": config.rpc_device,
                "member_only": bool(config.member_only),
                "subnet_driver_ready": config.subnet_driver_ready,
                # pre-rename managers read the old key
                "validator_driver_ready": config.subnet_driver_ready,
                **(
                    {"stage_proof_key": stage_proof_key}
                    if stage_proof_key
                    else {}
                ),
                # Free disk where auto-fetched models land, so the manager can
                # refuse a download that cannot fit instead of letting it die
                # mid-file (huggingface_hub only WARNS on low disk, then the
                # download fails at whatever percent the disk filled).
                "free_disk_gb": _fetch_dest_free_gb(),
            },
            "catalog": [
                {
                    "model_id": c.get("model_id"),
                    "model_bytes": int(c.get("model_bytes", 0) or 0),
                    **({"layers": int(c["layers"])} if c.get("layers") else {}),
                    **({"hf_repo": str(c["hf_repo"])} if c.get("hf_repo") else {}),
                    **({"hf_files": [str(f) for f in c["hf_files"]]} if c.get("hf_files") else {}),
                    # Per-model proof tolerance overrides must reach the manager
                    # so it can fan them to every mesh stage; without this the
                    # override sat unused in the local catalog and every stage
                    # ran the default band (the q3_K intermittent-failure bug).
                    **_operator_tuning_fields(c),
                }
                for c in config.catalog
            ],
            "endpoints": config.endpoints(),
            },
        )

    while True:
        try:
            joined = post_json(manager + "/v1/pool/join", join_payload(), timeout=10.0)
            break
        except Exception as exc:
            if stop_event.is_set():
                return
            # A 4xx from /v1/pool/join is the manager's DECISION about this
            # exact payload (undialable advertise address, bad token, policy
            # refusal) — retrying the same payload can never succeed, and the
            # 5s retry loop used to spin forever while the installer had
            # already reported success. Stay stopped
            # (units run with --no-autorestart) so the operator sees it.
            match = re.search(r"HTTP (4\d\d) from", str(exc))
            if match and int(match.group(1)) not in (408, 429):
                logger.error("pool join REFUSED (fatal, not retrying): %s", exc)
                raise RuntimeError(f"pool join refused: {exc}") from exc
            logger.warning("pool join failed (%s); retrying in 5s", exc)
            stop_event.wait(5.0)
    logger.info("pool join accepted: worker '%s'", joined.get("worker_id"))
    worker_id = str(joined["worker_id"])
    if worker_id != requested_worker_id:
        raise RuntimeError("pool manager returned a different worker identity")
    if on_join_info is not None:
        # Hand the CLI the pool-provided chain coordinates (token-only
        # workers start their allowlist refresher from these).
        try:
            on_join_info(dict(joined))
        except Exception:
            logger.exception("join-info callback failed; continuing")
    # Capacity-audit worker lifecycle, driven entirely by the manager's
    # heartbeat context: a worker bound to a chain-bound subnet mesh gets the
    # signed roster + its ordinal share and starts auditing; when the mesh
    # stops (context disappears) or the roster changes, the audit worker is
    # stopped/replaced. Keyed by the context digest so identical beats are
    # free.
    capacity_audit_state: dict[str, Any] = {"digest": "", "worker": None}

    def _sync_capacity_audit_worker(context: Any) -> None:
        import hashlib as _hashlib

        try:
            digest = (
                _hashlib.sha256(canonical_json_bytes(dict(context))).hexdigest()
                if isinstance(context, Mapping)
                else ""
            )
        except Exception:
            digest = ""
        if digest == capacity_audit_state["digest"]:
            return
        previous = capacity_audit_state["worker"]
        if previous is not None:
            try:
                previous.stop()
            except Exception:
                pass
        capacity_audit_state["digest"] = digest
        capacity_audit_state["worker"] = None
        if not digest:
            return
        try:
            from verallm.mesh.capacity_audit_worker import (
                MeshCapacityAuditWorker,
            )
            from verallm.mesh.delegated_signing import (
                delegate_capacity_signer_from_worker_request,
            )

            context = dict(context)
            slot_ctx = dict(context.get("slot") or {})
            signer = delegate_capacity_signer_from_worker_request(
                evm_address=str(slot_ctx.get("address", "") or ""),
                mesh_key=str(context.get("mesh_key", "") or ""),
                request=manager_call,
            )
            audit_worker = MeshCapacityAuditWorker(
                context=context,
                workdir=config.workdir,
                repo_root=config.repo_root,
                sign_artifact_remote=signer.sign_artifact,
                subtensor_network=str(
                    joined.get("subtensor_network", "") or ""
                ),
                netuid=int(joined.get("netuid", 0) or 0),
                worker_id=worker_id,
            )
            if audit_worker.start():
                capacity_audit_state["worker"] = audit_worker
        except Exception as exc:
            from verallm.mesh.capacity_audit_worker import (
                CapacityAuditWorkspaceError,
            )

            if isinstance(exc, CapacityAuditWorkspaceError):
                # Fail closed: a subnet worker without a working audit
                # workspace serves audit-blind and probates with nothing
                # in its own logs. Dying here makes the defect loud in
                # pm2 and in the manager's staleness view instead of
                # burning epochs silently.
                logger.critical(
                    "capacity audit workspace unavailable; refusing to "
                    "serve audit-blind: %s",
                    exc,
                )
                os._exit(78)
            # Transient failures must retry on the next heartbeat: the
            # digest memo would otherwise pin this failed start forever.
            capacity_audit_state["digest"] = ""
            logger.exception(
                "capacity audit worker failed to start; will retry on "
                "the next context sync"
            )

    status = "idle"
    active_mesh = ""
    beats = 0
    exec_threads: set[threading.Thread] = set()
    exec_threads_lock = threading.Lock()
    pending_command_ack = ""
    claimed_command_ids: dict[str, float] = {}
    completed_command_reports: dict[str, dict[str, Any]] = {}
    command_runtime_lock = threading.Lock()
    report_delivery_inflight: set[str] = set()
    phase_continuation_started: set[str] = set()
    worker_command_state = {"active_id": ""}

    def _log_termination_signal(signum: int, frame: Any) -> None:
        # A daemon killed mid-command dies silently today, and the forensic
        # cost of that silence is real: the manager only ever sees "went
        # silent". Name the signal and the in-flight command before dying.
        # (SIGKILL is uncatchable — cgroup OOM kills stay silent; this
        # covers every orderly stop: pm2 stop/restart, operator kill.)
        with command_runtime_lock:
            active = worker_command_state["active_id"]
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        logger.warning(
            "worker daemon received %s while %s; supervised serves survive "
            "in their own sessions and the next daemon start reaps them",
            signal_name,
            f"executing command {active[:8]}" if active else "idle",
        )
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _log_termination_signal)
    except ValueError:
        pass  # not the main thread (tests drive the daemon in threads)

    def _accepted_drive_ready_phase(
        report: Mapping[str, Any],
        response: Mapping[str, Any] | None,
    ) -> bool:
        return bool(
            response
            and str(response.get("status", "") or "") == "ok"
            and str(response.get("command_phase", "") or "") == "drive_ready"
            and str(report.get("event", "") or "") == "drive_ready"
        )

    def _mesh_gone_teardown(
        report: Mapping[str, Any],
        response: Mapping[str, Any] | None,
    ) -> None:
        """Tear down a local serve whose mesh the manager no longer tracks.

        A ``mesh_gone`` report response means the record this command spawned
        for was failed, stopped, or deleted manager-side while the command
        was still running. Whatever the command left behind (coordinator +
        llama on a driver, rpc stage on a member) is an unsupervised serve
        holding GPU memory that no pool state references and no operator
        command can reach; it also shadows the worker's ports against every
        later launch. Reap it and idle the worker.
        """
        nonlocal status, active_mesh
        if not (response and bool(response.get("mesh_gone"))):
            return
        event = str(report.get("event", "") or "")
        if event == "stopped":
            return  # a stop already tore everything down
        mesh_key = str(report.get("mesh_key", "") or "")
        command_id = str(report.get("command_id", "") or "")
        logger.error(
            "manager no longer tracks mesh %s (command %s, event %s): "
            "tearing down the local serve so it cannot linger unsupervised "
            "on the GPU",
            mesh_key,
            command_id[:8],
            event,
        )
        try:
            runner.stop({"command_id": command_id, "mesh_key": mesh_key})
        except Exception as exc:
            logger.error(
                "mesh_gone teardown for %s failed: %s",
                mesh_key,
                str(exc)[:200],
            )
            return
        with command_runtime_lock:
            still_active = worker_command_state["active_id"] == command_id
        if still_active:
            status = "idle"
            active_mesh = ""
            warm_state["driver_serving"] = False

    def _remember_and_deliver_command_report(
        command_id: str,
        fields: Mapping[str, Any],
        *,
        phase_command: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        nonlocal pending_command_ack
        report_fields = dict(fields)
        _journal_report_pending(command_id, report_fields)
        with command_runtime_lock:
            completed_command_reports[command_id] = report_fields
            report_delivery_inflight.add(command_id)
            while len(completed_command_reports) > 4096:
                completed_command_reports.pop(next(iter(completed_command_reports)))
        response: dict[str, Any] | None = None
        try:
            response = deliver_worker_report(report_fields)
            if response is not None:
                _journal_report_delivered(
                    command_id,
                    stale=str(response.get("status", "") or "") == "stale",
                )
                # A delivered command-bound report is stronger than the
                # informational heartbeat receipt ACK. Do not keep emitting a
                # stale ACK after a fast terminal report already cleared the
                # manager's in-flight record.
                if pending_command_ack == command_id:
                    pending_command_ack = ""
        finally:
            with command_runtime_lock:
                report_delivery_inflight.discard(command_id)
        if (
            phase_command is not None
            and _accepted_drive_ready_phase(report_fields, response)
        ):
            _start_multibox_verifier(phase_command)
        _mesh_gone_teardown(report_fields, response)
        return response

    def _retry_completed_command_report(
        command_id: str,
        *,
        phase_command: Mapping[str, Any] | None = None,
    ) -> None:
        with command_runtime_lock:
            report_fields = completed_command_reports.get(command_id)
            if report_fields is None or command_id in report_delivery_inflight:
                return
            report_delivery_inflight.add(command_id)

        def _deliver() -> None:
            response: dict[str, Any] | None = None
            try:
                response = deliver_worker_report(report_fields)
                if response is not None:
                    _journal_report_delivered(
                        command_id,
                        stale=str(response.get("status", "") or "") == "stale",
                    )
            finally:
                with command_runtime_lock:
                    report_delivery_inflight.discard(command_id)
            if (
                phase_command is not None
                and _accepted_drive_ready_phase(report_fields, response)
            ):
                _start_multibox_verifier(phase_command)
            _mesh_gone_teardown(report_fields, response)

        threading.Thread(target=_deliver, daemon=True).start()

    def _start_multibox_verifier(command: Mapping[str, Any]) -> None:
        """Start the accepted drive-ready continuation exactly once."""

        command_id = str(command.get("command_id", "") or "")
        command_digest = str(command.get("command_digest", "") or "")
        mesh_key = str(command.get("mesh_key", "") or "")
        if (
            str(command.get("action", "") or "") != "drive"
            or int(command.get("member_count", 1) or 1) <= 1
        ):
            return
        with command_runtime_lock:
            if (
                worker_command_state["active_id"] != command_id
                or command_id in phase_continuation_started
            ):
                return
            phase_continuation_started.add(command_id)

        def _verify_multibox() -> None:
            nonlocal status
            try:
                enter_command = getattr(runner, "enter_command", None)
                if callable(enter_command):
                    enter_command(command_id)
                verification_snapshot_hash = str(
                    runner.verify_backend_ready() or ""
                )
                out: dict[str, Any] = {"event": "serving"}
                if (
                    _normalize_pool_serving_mode(command.get("serving_mode"))
                    == POOL_SERVING_MODE_SUBNET
                    and command.get("model_index") is not None
                ):
                    # The initial deploy measures capacity before registration;
                    # only the subsequent chain-bound launch has a snapshot.
                    if not _COMMAND_DIGEST_RE.fullmatch(
                        verification_snapshot_hash
                    ):
                        raise RuntimeError(
                            "validator split backend became ready without a "
                            "signed snapshot"
                        )
                    out["verification_snapshot_hash"] = (
                        verification_snapshot_hash
                    )
                new_verify_status = "serving"
                new_verify_warm = True
            except Exception as exc:
                out = {
                    "event": "error",
                    "message": (
                        "split backend never became ready: "
                        f"{str(exc)[:400]}"
                    ),
                }
                new_verify_status = "error"
                new_verify_warm = False
            with command_runtime_lock:
                verify_still_active = (
                    worker_command_state["active_id"] == command_id
                )
            if verify_still_active:
                status = new_verify_status
                warm_state["driver_serving"] = new_verify_warm
            _remember_and_deliver_command_report(
                command_id,
                {
                    "command_id": command_id,
                    "command_digest": command_digest,
                    "mesh_key": mesh_key,
                    **out,
                },
            )

        def _verify_tracked() -> None:
            try:
                _verify_multibox()
            finally:
                with exec_threads_lock:
                    exec_threads.discard(threading.current_thread())

        verify_thread = threading.Thread(target=_verify_tracked, daemon=True)
        with exec_threads_lock:
            exec_threads.add(verify_thread)
        verify_thread.start()

    # RTT measurement runs in ITS OWN thread. Probing peers inline made every
    # heartbeat wait out TCP timeouts to unreachable peers (1.5s each), which
    # stretched the beat cadence to multiple seconds — and queued operator
    # chats ride the heartbeat, so that lag was the bulk of chat TTFT.
    probe_state: dict[str, Any] = {"targets": {}, "results": {}, "rtt_manager": None}
    probe_lock = threading.Lock()

    def _prober() -> None:
        while not stop_event.is_set():
            rtt_mgr = _tcp_rtt_ms(manager, timeout=4.0)
            with probe_lock:
                targets = dict(probe_state["targets"])
            results = {
                peer_id: _tcp_rtt_ms(endpoint, timeout=1.5)
                for peer_id, endpoint in targets.items()
            }
            with probe_lock:
                probe_state["results"] = results
                probe_state["rtt_manager"] = rtt_mgr
            stop_event.wait(10.0)  # RTTs drift slowly; keep beats unblocked

    threading.Thread(target=_prober, daemon=True).start()

    # Keep-warm coordination. warm_state["driver_serving"] is True only while
    # THIS worker drives a serving mesh (has a local llama-server at mesh_port+1).
    # warm_lock makes the backend single-tenant between keep-warm and real chats:
    # a chat arms proof capture on the rpc-server, and a keep-warm graph running
    # concurrently would land in that capture window ("no GGML proof trace
    # verified"). A chat takes the lock for its whole duration; keep-warm only
    # pings when it can take the lock uncontended, so the two never overlap.
    warm_state = {"driver_serving": False, "last_chat_unix": 0.0}
    warm_lock = threading.Lock()

    def _run_chat_tracked(chat: dict[str, Any]) -> None:
        with warm_lock:
            try:
                _run_pool_chat(
                    manager,
                    secret,
                    worker_id,
                    config,
                    {**chat, "_worker_session_id": worker_session_id},
                    runner=runner,
                )
            finally:
                # Real traffic already touched every layer — the keep-warm
                # ping skips its next tick(s) instead of re-warming warm pages.
                warm_state["last_chat_unix"] = time.time()

    chat_claim_lock = threading.Lock()
    chat_claiming_or_running: set[str] = set()
    chat_recent_completed: dict[str, float] = {}

    def _release_chat_claim(chat_id: str, *, completed: bool) -> None:
        with chat_claim_lock:
            chat_claiming_or_running.discard(chat_id)
            if completed:
                chat_recent_completed[chat_id] = (
                    time.monotonic() + CHAT_COMPLETED_TTL_S
                )

    def _run_claimed_chat(chat: dict[str, Any], chat_id: str) -> None:
        try:
            _run_chat_tracked(chat)
        finally:
            # Even if terminal delivery exhausts its bounded retries, never run
            # the same prompt twice on this worker.  The manager owns eventual
            # stale-lock cleanup for that exceptional case.
            _release_chat_claim(chat_id, completed=True)

    def _claim_and_start_chat(chat: dict[str, Any]) -> None:
        chat_id = str(chat.get("chat_id", "") or "")
        delivery_token = str(chat.get("delivery_token", "") or "")
        if not chat_id or not _WORKER_AUTH_NONCE_RE.fullmatch(delivery_token):
            logger.error("pool chat lease is malformed; refusing to execute it")
            return
        with chat_claim_lock:
            now = time.monotonic()
            for completed_id in [
                value
                for value, expiry in chat_recent_completed.items()
                if expiry <= now
            ]:
                chat_recent_completed.pop(completed_id, None)
            if (
                chat_id in chat_claiming_or_running
                or chat_id in chat_recent_completed
            ):
                return
            chat_claiming_or_running.add(chat_id)

        try:
            delivery_expires_at_ms = int(
                chat.get("delivery_expires_at_unix_ms", 0) or 0
            )
            request_expires_at_ms = int(
                chat.get("request_expires_at_unix_ms", 0) or 0
            )
        except (TypeError, ValueError, OverflowError):
            delivery_expires_at_ms = request_expires_at_ms = 0
        now_unix_ms = int(time.time() * 1000)
        expiry_candidates = [
            value
            for value in (delivery_expires_at_ms, request_expires_at_ms)
            if value > 0
        ]
        if not expiry_candidates or min(expiry_candidates) <= now_unix_ms:
            _release_chat_claim(chat_id, completed=False)
            return
        deadline = time.monotonic() + min(
            CHAT_DELIVERY_LEASE_S,
            max(0.0, (min(expiry_candidates) - now_unix_ms) / 1000.0),
        )
        accepted = False
        try:
            while not stop_event.is_set() and time.monotonic() < deadline:
                try:
                    response = post_json(
                        manager + "/v1/pool/chat-pickup",
                        worker_request(
                            "chat-pickup",
                            {
                                "chat_id": chat_id,
                                "delivery_token": delivery_token,
                            },
                        ),
                        timeout=15.0,
                    )
                except Exception:
                    stop_event.wait(CHAT_POLL_RETRY_S)
                    continue
                status_value = str(response.get("status", "") or "")
                if status_value == "ok":
                    accepted = True
                    break
                if status_value in {"stale", "cancelled", "expired"}:
                    return
                stop_event.wait(CHAT_POLL_RETRY_S)
            if not accepted:
                return
            threading.Thread(
                target=_run_claimed_chat,
                args=(chat, chat_id),
                daemon=True,
            ).start()
        finally:
            if not accepted:
                _release_chat_claim(chat_id, completed=False)

    def _chat_poll_loop() -> None:
        # Blocks on the manager until a chat is queued for us, then runs each
        # against our LOCAL coordinator in its own thread (a slow generation
        # never stalls the next pickup). Pickup ~= one network RTT.
        while not stop_event.is_set():
            try:
                resp = post_json(
                    manager + "/v1/pool/chat-poll",
                    worker_request("chat-poll", {"wait": CHAT_POLL_WAIT_S}),
                    timeout=CHAT_POLL_HTTP_TIMEOUT_S,
                )
            except Exception:
                stop_event.wait(CHAT_POLL_RETRY_S)
                continue
            for chat in resp.get("chat") or []:
                _claim_and_start_chat(dict(chat))

    threading.Thread(target=_chat_poll_loop, daemon=True).start()

    def _keepwarm() -> None:
        # A GGUF model's weights get paged out between requests on a
        # memory-pressured host (Apple unified memory is the worst case: a 20GB
        # model competes with everything and macOS evicts idle pages), so the
        # next request pays a multi-second cold re-fault. A periodic 1-token
        # forward through the driver's own engine touches every layer's weights,
        # keeping them resident. We hit the INNER llama-server directly
        # (mesh_port+1), NOT the coordinator, so no proof capture is armed — and
        # we skip when a real chat is inflight so we never collide with one.
        # Backend-agnostic: it is just an OpenAI-style completion, so it warms
        # Metal, CUDA, and every RPC member in the pipeline alike.
        inner = f"http://127.0.0.1:{config.mesh_port + 1}/v1/chat/completions"
        payload = json.dumps({
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }).encode()
        interval = float(
            os.environ.get("VERATHOS_KEEPWARM_INTERVAL_S", str(KEEPWARM_INTERVAL_S))
        )
        if interval <= 0:
            return  # keep-warm disabled (e.g. CUDA-only boxes that never page)
        while not stop_event.is_set():
            stop_event.wait(interval)
            if stop_event.is_set() or not warm_state["driver_serving"]:
                continue
            if time.time() - warm_state["last_chat_unix"] < interval:
                continue  # real traffic within the window keeps it warm already
            if not warm_lock.acquire(blocking=False):
                continue  # a chat holds the backend; skip this tick, never queue
            try:
                req = urllib.request.Request(
                    inner, data=payload, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=30.0).read()
            except Exception:
                pass  # best-effort; a failed ping must never disturb the loop
            finally:
                warm_lock.release()

    threading.Thread(target=_keepwarm, daemon=True).start()

    # Driver self-audit: light proofs verify a DEGENERATE mesh's output
    # (proof-valid garbage -
    # mid-life into token-0 spam that light tier kept verifying; only the
    # decode audit refused). The driver periodically runs one full
    # verified HARD request through its own coordinator; two consecutive
    # failures surface as runtime_error on the heartbeat, the manager
    # fails the mesh, and the operator/deploy relaunches a clean one
    # instead of serving garbage to validators and users.
    audit_state = {"consecutive": 0, "failure": ""}

    def _self_audit() -> None:
        interval = float(
            os.environ.get(
                "VERATHOS_SELF_AUDIT_INTERVAL_S", str(SELF_AUDIT_INTERVAL_S)
            )
        )
        if interval <= 0:
            return
        coordinator = f"http://127.0.0.1:{config.mesh_port}/v1/chat/completions"
        while not stop_event.is_set():
            stop_event.wait(interval)
            if stop_event.is_set() or not warm_state["driver_serving"]:
                continue
            # A capacity-audit drain 503s every chat BY DESIGN (byte-identical
            # busy signal); a self-audit landing in that window can read its
            # own drain as a backend failure. Skip the tick; the next one
            # probes normally.
            try:
                from verallm.mesh.capacity_audit_worker import (
                    capacity_drain_active,
                    capacity_drain_file_path,
                )

                if capacity_drain_active(
                    capacity_drain_file_path(config.workdir)
                ):
                    continue
            except Exception:
                pass
            if not warm_lock.acquire(blocking=False):
                continue  # a real chat is running; audit next tick
            try:
                internal_secret = ""
                snapshot_binding: dict[str, Any] = {}
                context_fn = getattr(runner, "_local_request_context", None)
                if callable(context_fn):
                    try:
                        internal_secret, snapshot_binding = context_fn()
                    except Exception:
                        continue  # snapshot mid-rotation; next tick retries
                payload: dict[str, Any] = {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Briefly name two properties a cryptographic "
                                "hash function must have."
                            ),
                        }
                    ],
                    "max_tokens": 160,
                    "temperature": 0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "verathos": {**snapshot_binding, "proof_tier": "hard"},
                }
                failure = ""
                try:
                    response = post_json(
                        coordinator,
                        payload,
                        timeout=SELF_AUDIT_TIMEOUT_S,
                        internal_auth_secret=internal_secret,
                    )
                    mesh_meta = (
                        response.get("verathos_mesh")
                        if isinstance(response, dict)
                        else None
                    ) or {}
                    if not bool(mesh_meta.get("verified")):
                        failure = "self-audit reply did not verify"
                except Exception as exc:
                    failure = str(exc)[:300]
                if failure and self_audit_skippable(failure):
                    logger.info("self-audit skipped (contention): %s", failure)
                    continue
                if failure:
                    audit_state["consecutive"] += 1
                    logger.warning(
                        "SELF-AUDIT FAILED (%d consecutive): %s",
                        audit_state["consecutive"],
                        failure,
                    )
                    if audit_state["consecutive"] >= 2:
                        audit_state["failure"] = failure
                else:
                    audit_state["consecutive"] = 0
                    audit_state["failure"] = ""
            finally:
                warm_lock.release()

    threading.Thread(target=_self_audit, daemon=True).start()

    def _tracked_child_pids() -> set[int]:
        """Pids whose exit statuses belong to live Popen owners, not us."""

        pids: set[int] = set()
        for proc in list(getattr(runner, "procs", None) or []):
            pid = getattr(proc, "pid", None)
            if pid is not None:
                pids.add(int(pid))
        try:
            from verallm.mesh.capacity_audit_worker import (
                tracked_child_pids as _audit_child_pids,
            )

            pids.update(_audit_child_pids())
        except Exception:
            pass
        return pids

    zombie_pending: frozenset[int] = frozenset()

    while not stop_event.is_set():
        if max_beats is not None and beats >= max_beats:
            return
        beats += 1
        # Adopted-orphan sweep runs at the TOP of the beat so zombies are
        # collected even while the manager is unreachable (rolls that
        # orphan backends do not wait for healthy heartbeats).
        try:
            reaped_zombies, zombie_pending = reap_untracked_zombies(
                _tracked_child_pids, zombie_pending
            )
        except Exception as exc:
            reaped_zombies = []
            logger.warning("zombie sweep failed: %s", exc)
        if reaped_zombies:
            print(
                f"reaped adopted orphan child processes: {reaped_zombies}",
                flush=True,
            )
        with probe_lock:
            rtt = probe_state["rtt_manager"]
            probe_results = dict(probe_state["results"])
        runtime_error = ""
        if active_mesh and status != "stopping":
            health_check = getattr(runner, "runtime_failure", None)
            if callable(health_check):
                try:
                    runtime_error = str(health_check() or "")[:500]
                except Exception as exc:
                    runtime_error = f"cannot inspect mesh child processes: {exc}"[:500]
            if not runtime_error and audit_state["failure"]:
                # Two consecutive self-audit failures: the mesh serves
                # output its own decode audit refuses (degenerate serve
                # state). Failing the mesh beats serving proof-valid
                # garbage until a validator audit lands.
                runtime_error = f"self-audit: {audit_state['failure']}"[:500]
        try:
            beat = post_json(
                manager + "/v1/pool/heartbeat",
                worker_request(
                    "heartbeat",
                    {
                    "status": (
                        ("error" if runtime_error else status)
                        if backend_ready["done"] and not backend_ready["error"]
                        else (
                            backend_ready["error"]
                            or backend_ready["note"]
                            or "preparing backend"
                        )
                    ),
                    "rtt_manager": rtt,
                    "probe_results": probe_results,
                    "subnet_driver_ready": config.subnet_driver_ready,
                    # pre-rename managers read the old key
                    "validator_driver_ready": config.subnet_driver_ready,
                    "runtime_error": runtime_error,
                    **(
                        {"command_ack": pending_command_ack}
                        if pending_command_ack
                        else {}
                    ),
                    },
                ),
                timeout=10.0,
            )
        except Exception:
            stop_event.wait(config.heartbeat_s)
            continue
        acknowledged = str(beat.get("command_acknowledged", "") or "")
        if pending_command_ack and acknowledged == pending_command_ack:
            pending_command_ack = ""
        _sync_capacity_audit_worker(beat.get("capacity_audit_context"))
        if beat.get("status") == "unknown-worker":
            # The manager was restarted/reset and no longer knows us. Re-join
            # so we reappear in the pool instead of heartbeating into the void
            # forever (this is what silently dropped workers on manager resets).
            # If this process was serving a mesh, that mesh no longer exists
            # on the manager side — tear our runtime down BEFORE rejoining as
            # idle. Skipping this left orphaned backends owning the GPU and
            # its ports while the manager placed a fresh mesh on an "idle"
            # worker (the resurrected-mesh / double-booked-GPU incident).
            if active_mesh or status not in ("idle", "stopping"):
                fence_command = getattr(runner, "fence_command", None)
                stop_runner = getattr(runner, "stop", None)
                reset_command_id = "cmd-" + uuid.uuid4().hex
                # Claim the active command BEFORE tearing down: a surviving
                # drive/join thread gates its status write-back on this id,
                # and without the claim it would overwrite the post-teardown
                # "idle" with "serving"/"error" — stranding a worker the
                # manager then refuses to place anything on, forever.
                with command_runtime_lock:
                    worker_command_state["active_id"] = reset_command_id
                warm_state["driver_serving"] = False
                if callable(stop_runner):
                    try:
                        if callable(fence_command):
                            fence_command(reset_command_id)
                        stop_runner(
                            {
                                "command_id": reset_command_id,
                                "action": "stop",
                                "mesh_key": active_mesh,
                            }
                        )
                    except Exception as exc:
                        logger.error(
                            "mesh teardown after manager reset failed: %s", exc
                        )
                status, active_mesh = "idle", ""
            try:
                post_json(manager + "/v1/pool/join", join_payload(), timeout=10.0)
                status, active_mesh = "idle", ""
            except Exception:
                pass
            stop_event.wait(config.heartbeat_s)
            continue
        # Hand the freshest probe targets to the prober thread; results ride a
        # later heartbeat (RTT is display/placement data, never latency-critical).
        with probe_lock:
            probe_state["targets"] = dict(beat.get("probe") or {})
        # Operator chat is delivered by the dedicated long-poll thread below,
        # not the heartbeat (near-zero pickup latency).
        command = beat.get("command") or None
        if command:
            command = dict(command)
            command_id = str(command.get("command_id", "") or "")
            if not _COMMAND_ID_RE.fullmatch(command_id):
                logger.error(
                    "pool command has no valid command_id; refusing it"
                )
                stop_event.wait(config.heartbeat_s)
                continue
            supplied_digest = str(command.pop("command_digest", "") or "")
            command_digest = hashlib.sha256(
                canonical_json_bytes(command)
            ).hexdigest()
            if not (
                _COMMAND_DIGEST_RE.fullmatch(supplied_digest)
                and secrets.compare_digest(supplied_digest, command_digest)
            ):
                logger.error(
                    "pool command %s has an invalid payload digest; refusing it",
                    command_id,
                )
                stop_event.wait(config.heartbeat_s)
                continue
            command["command_digest"] = supplied_digest
            pending_command_ack = command_id
            journal_state, saved_report = _claim_journal_command(command)
            if journal_state == "conflict":
                logger.error(
                    "pool command id %s was reused with a different digest; "
                    "refusing it",
                    command_id,
                )
                stop_event.wait(config.heartbeat_s)
                continue
            if saved_report is not None:
                claimed_command_ids[command_id] = time.monotonic()
                if journal_state == "phase_completed":
                    # The manager durably acknowledged drive_ready and retains
                    # the drive command until the verifier reports serving.
                    # Redelivery is a continuation prompt, not a reason to POST
                    # drive_ready on every heartbeat.
                    _start_multibox_verifier(command)
                else:
                    with command_runtime_lock:
                        completed_command_reports[command_id] = saved_report
                    _retry_completed_command_report(
                        command_id,
                        phase_command=(
                            command
                            if journal_state == "report_pending"
                            else None
                        ),
                    )
                stop_event.wait(config.heartbeat_s)
                continue
            if journal_state == "interrupted":
                claimed_command_ids[command_id] = time.monotonic()
                interrupted_report = {
                    "command_id": command_id,
                    "command_digest": command_digest,
                    "mesh_key": str(command.get("mesh_key", "") or ""),
                    "event": "error",
                    "message": (
                        "worker restarted while this command was running; "
                        "refusing duplicate execution and tearing the mesh down"
                    ),
                }
                threading.Thread(
                    target=_remember_and_deliver_command_report,
                    args=(command_id, interrupted_report),
                    daemon=True,
                ).start()
                stop_event.wait(config.heartbeat_s)
                continue
            if journal_state == "running" or command_id in claimed_command_ids:
                _retry_completed_command_report(command_id)
                stop_event.wait(config.heartbeat_s)
                continue
            claimed_command_ids[command_id] = time.monotonic()
            while len(claimed_command_ids) > 4096:
                claimed_command_ids.pop(next(iter(claimed_command_ids)))
            action = str(command.get("action", ""))
            mesh_key = str(command.get("mesh_key", ""))
            with exec_threads_lock:
                exec_threads = {thread for thread in exec_threads if thread.is_alive()}
                live_exec_threads = list(exec_threads)
            if live_exec_threads and action not in ("stop", "rotate"):
                # Never run two long commands at once; stop is always allowed
                # (it kills the processes the stuck command is waiting on) and
                # rotate runs BESIDE a live drive by design (the drive thread
                # stays alive for as long as the mesh serves).
                _remember_and_deliver_command_report(
                    command_id,
                    {
                        "command_id": command_id,
                        "command_digest": command_digest,
                        "mesh_key": mesh_key,
                        "event": "error",
                        "message": "worker is still executing a previous command",
                    },
                )
            else:
                # Execute in a thread so HEARTBEATS KEEP FLOWING during long
                # operations (a 30B-class drive loads ~20GB before the backend
                # is up). A blocked loop made the worker look stale/dead in
                # the dashboard while it was simply loading a model.
                fence_command = getattr(runner, "fence_command", None)
                if callable(fence_command) and action != "rotate":
                    # For stop this runs synchronously before its thread is
                    # scheduled, so an older drive/join cannot spawn another
                    # child after teardown has begun. Rotate must NOT fence:
                    # it runs beside the live drive and superseding it would
                    # invalidate the serving command's own state writes.
                    fence_command(command_id)
                if action != "rotate":
                    with command_runtime_lock:
                        worker_command_state["active_id"] = command_id
                if action in ("drive", "join", "fetch"):
                    status = {"drive": "driving", "join": "joining", "fetch": "fetching"}[action]
                elif action == "stop":
                    status = "stopping"

                def _exec(
                    cmd=command,
                    act=action,
                    mk=mesh_key,
                    cid=command_id,
                    cdigest=command_digest,
                ) -> None:
                    nonlocal status, active_mesh
                    new_warm: bool | None = None
                    try:
                        if act == "drive":
                            # After drive the driver serves its own stage too.
                            # The progress callback is only live WHILE drive()
                            # runs: a prewarm that outlives the ready-timeout
                            # must not keep clobbering "serving" afterwards.
                            _prog_live = {"on": True}

                            def _dprog(p: str) -> None:
                                nonlocal status
                                with command_runtime_lock:
                                    active = worker_command_state["active_id"] == cid
                                if _prog_live["on"] and active:
                                    status = f"driving ({p})"

                            report = runner.drive(cmd, progress=_dprog)
                            _prog_live["on"] = False
                            if int(cmd.get("member_count", 1)) <= 1:
                                # Single-box: drive() already waited for a real
                                # 200 AND ran a verified generation, so it truly
                                # serves now. Only a DRIVER has a local llama to
                                # keep warm (members run an rpc-server).
                                new_status, new_mesh = "serving", mk
                                new_warm = True
                            else:
                                # Multi-box: the split backend can't answer until
                                # the OTHER members join (after this drive_ready is
                                # reported). Stay "driving"; the verifier below
                                # flips us to serving once the whole pipeline
                                # answers verified.
                                new_status, new_mesh = "driving", mk
                        elif act == "join":
                            report = runner.join(cmd)
                            new_status, new_mesh = "serving", mk
                        elif act == "fetch":
                            # Driver lacks the model: download it, then wait for
                            # the manager's follow-up drive command (dispatched
                            # when our "fetched" report lands).
                            def _prog(p: str) -> None:
                                nonlocal status
                                with command_runtime_lock:
                                    active = worker_command_state["active_id"] == cid
                                if active:
                                    status = f"fetching {p}"

                            report = runner.fetch(cmd, progress=_prog)
                            new_status, new_mesh = "idle", mk
                        elif act == "rotate":
                            # Epoch re-sign of the served snapshot: instant,
                            # runs beside the live drive, and must never
                            # disturb the serving status either way.
                            report = runner.rotate_snapshot(cmd)
                            new_status, new_mesh = status, active_mesh
                        elif act == "stop":
                            report = runner.stop(cmd)
                            deadline = time.monotonic() + COMMAND_STOP_QUIESCE_S
                            current_thread = threading.current_thread()
                            while time.monotonic() < deadline:
                                with exec_threads_lock:
                                    older_threads = [
                                        thread
                                        for thread in exec_threads
                                        if thread is not current_thread
                                        and thread.is_alive()
                                    ]
                                if not older_threads:
                                    break
                                for older_thread in older_threads:
                                    older_thread.join(timeout=0.2)
                            else:
                                raise RuntimeError(
                                    "superseded mesh command did not quiesce after stop"
                                )
                            new_status, new_mesh = "idle", ""
                            new_warm = False
                        else:
                            report = {"event": "error", "message": f"unknown action {act}"}
                            new_status, new_mesh = status, active_mesh
                    except Exception as exc:
                        report = {"event": "error", "message": str(exc)[:500]}
                        if act == "drive" and _KV_ALLOC_FAILURE_RE.search(
                            str(exc)
                        ):
                            # The stored KV fit passed the health check but
                            # the serve then died on an allocation (compute
                            # buffers land on TOP of the KV at the first
                            # full batch). Keeping the cache would relaunch
                            # straight into the same crash; dropping it
                            # makes the next attempt re-measure.
                            try:
                                runner._kv_fit_path(cmd).unlink(
                                    missing_ok=True
                                )
                                logger.warning(
                                    "dropped cached KV fit after an "
                                    "allocation failure: %s",
                                    str(exc)[:200],
                                )
                            except Exception:
                                pass
                        if act == "rotate":
                            # A failed rotation leaves the mesh serving under
                            # its previous snapshot; the manager's follower
                            # retries. Never flip a serving worker to error
                            # or kill its warm loop over it.
                            new_status, new_mesh = status, active_mesh
                        else:
                            new_status, new_mesh = "error", active_mesh
                            new_warm = False
                    with command_runtime_lock:
                        is_active_command = (
                            worker_command_state["active_id"] == cid
                        )
                    if is_active_command:
                        status, active_mesh = new_status, new_mesh
                        if new_warm is not None:
                            warm_state["driver_serving"] = new_warm
                    _remember_and_deliver_command_report(
                        cid,
                        {
                            "command_id": cid,
                            "command_digest": cdigest,
                            "mesh_key": mk or active_mesh,
                            **report,
                        },
                        phase_command=(
                            cmd
                            if (
                                act == "drive"
                                and int(cmd.get("member_count", 1)) > 1
                                and report.get("event") == "drive_ready"
                            )
                            else None
                        ),
                    )

                def _exec_tracked() -> None:
                    try:
                        _exec()
                    finally:
                        with exec_threads_lock:
                            exec_threads.discard(threading.current_thread())

                command_thread = threading.Thread(
                    target=_exec_tracked,
                    daemon=True,
                )
                with exec_threads_lock:
                    exec_threads.add(command_thread)
                command_thread.start()
        stop_event.wait(config.heartbeat_s)
